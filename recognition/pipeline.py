# -*- coding: utf-8 -*-
"""统一的截图识别入口。

优先使用 YOLOv5 检测模型（VinXiangQi 的预训练权重），识别率远高于
模板匹配；模型或 onnxruntime 不可用时，回退到几何定位 + 模板匹配
（对浅色棋面 + 深/红文字的皮肤尚可）；两者都失败时由用户用手动摆局兜底。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from recognition.yolo import CLASS_TO_FEN, Detection, YoloDetector, default_model_path
from recognition.yolo_board import (BoardResult, board_from_detections,
                                    board_plausibility)
from recognition import skin


@dataclass
class RecognizeResult:
    grid: List[List[Optional[str]]]          # FEN 顺序（row0 = 黑方底线）
    red_at_top: bool = False                 # 截图是否红方在上
    source: str = "yolo"                     # yolo / geometric / none
    n_pieces: int = 0
    n_ambiguous: int = 0
    detections: List[Detection] = field(default_factory=list)
    board_box: Optional[tuple] = None
    warning: Optional[str] = None            # 给用户看的提示
    error: Optional[str] = None
    corrections: list = field(default_factory=list)  # 皮肤模板修正 [(row,col,旧,新,分)]
    skin_classes: int = 0                    # 皮肤模板覆盖的类别数（0=未校准）


_detector: Optional[YoloDetector] = None
_detector_path: Optional[str] = None
_skin_cache = {"path": None, "mtime": None, "templates": None}
# 裁剪框缓存：同一窗口（同尺寸）会话内棋盘位置固定。裁剪重试成功一次后，
# 后续帧直接裁剪识别——既绕开"皮肤边框被算进棋盘"的模型缺陷，又比全窗更快。
_crop_cache = {"shape": None, "box": None}


def get_skin_templates(path: Optional[str] = None) -> Optional[dict]:
    """加载皮肤模板（带 mtime 缓存；文件被重新采集后自动刷新）。"""
    p = path or skin.default_skin_path()
    try:
        mtime = os.path.getmtime(p) if os.path.exists(p) else None
    except OSError:
        mtime = None
    if mtime is None:
        return None
    if (_skin_cache["path"] == p and _skin_cache["mtime"] == mtime
            and _skin_cache["templates"] is not None):
        return _skin_cache["templates"]
    tpl = skin.load_templates(p)
    _skin_cache.update(path=p, mtime=mtime, templates=tpl)
    return tpl


def get_detector(model_path: Optional[str] = None,
                 conf: float = 0.4, iou: float = 0.5) -> Optional[YoloDetector]:
    """惰性创建并复用 YOLO 检测器（模型加载约数百毫秒）。"""
    global _detector, _detector_path
    path = model_path or default_model_path()
    if path is None:
        return None
    if _detector is not None and _detector_path == path:
        return _detector
    try:
        _detector = YoloDetector(path, conf=conf, iou=iou)
        _detector_path = path
    except Exception:
        _detector = None
    return _detector


def yolo_available() -> bool:
    return get_detector() is not None


def recognize_screenshot(img_bgr: np.ndarray,
                         use_yolo: bool = True,
                         model_path: Optional[str] = None,
                         skin_path: Optional[str] = None,
                         use_skin: bool = True) -> RecognizeResult:
    """识别整张截图。返回 RecognizeResult（grid 为 FEN 顺序）。

    model_path 为 None 时自动挑选 models/ 下最优模型；
    use_skin 且存在皮肤模板（用户此前核对过的局面采集）时做二次校验，
    修复该皮肤上的系统性误判。窗口探测等场景应传 use_skin=False——
    未经验证的模板可能引入新的误判。
    """
    templates = get_skin_templates(skin_path) if (use_yolo and use_skin) else None
    if use_yolo:
        det = get_detector(model_path)
        if det is not None:
            try:
                # 命中上次成功的裁剪框：直接裁剪识别（快路径）
                cbox = _crop_cache.get("box")
                if (cbox is not None
                        and _crop_cache.get("shape") == img_bgr.shape[:2]):
                    cached = _recognize_cropped_image(img_bgr, cbox, det,
                                                      templates)
                    if cached is not None:
                        return cached
                    _crop_cache["box"] = None     # 失效（移动/换局）：走完整流程
                dets = det.detect(img_bgr)
                br = board_from_detections(dets, img=img_bgr,
                                           skin_templates=templates)
                if br is not None and br.n_pieces >= 4:
                    # 统一"先裁剪再识别"：不论整窗结果好坏，都优先在裁剪图上
                    # 出结果（裁剪排除鼓/旗/凳装饰与工具栏——那是"多出一个
                    # 炮/相"这类误检的来源，准确率更高；实测同皮肤裁剪后
                    # 识别完全正确）。裁剪框：整窗框可信则贴棋盘框+半格，
                    # 否则用"棋子范围+1.5 格"（某边棋子漏检时更耐操）
                    ok_box, _why = board_plausibility(br)
                    cbox = (_tight_crop_box(br.board_box, img_bgr.shape)
                            if ok_box else None)
                    if cbox is None:
                        cbox = _crop_box_from_dets(dets, img_bgr.shape)
                    if cbox is not None:
                        cropped = _recognize_cropped_image(img_bgr, cbox, det,
                                                           templates)
                        if cropped is not None:
                            tight = (_tight_crop_box(cropped.board_box,
                                                     img_bgr.shape) or cbox)
                            _crop_cache.update(shape=img_bgr.shape[:2],
                                               box=tight)
                            return cropped
                    # 裁剪不可用（无框/被裁坏/校验不过）：整窗结果兜底
                    if ok_box:
                        tight = _tight_crop_box(br.board_box, img_bgr.shape)
                        if tight is not None:
                            _crop_cache.update(shape=img_bgr.shape[:2],
                                               box=tight)
                        return _from_yolo(br, dets, skin_classes=len(templates or {}))
                    return _from_yolo(br, dets, skin_classes=len(templates or {}))
                # 棋子太少：可能是模型不认识该皮肤，继续尝试几何方案
                geo = _recognize_geometric(img_bgr)
                if geo is not None and geo.n_pieces > (br.n_pieces if br else 0):
                    geo.warning = ("检测模型对这张截图只找到 "
                                   f"{br.n_pieces if br else 0} 个棋子，已改用备用识别。"
                                   "建议核对，或使用「手动摆局」。")
                    return geo
                if br is not None:
                    return _from_yolo(br, dets)
                return RecognizeResult(grid=_empty(), source="none",
                                       error="未在截图中定位到棋盘")
            except Exception as e:
                return RecognizeResult(grid=_empty(), source="none",
                                       error=f"检测模型执行失败: {e}")
        # 模型不可用 -> 几何方案
        geo = _recognize_geometric(img_bgr)
        if geo is not None:
            geo.warning = ("未找到检测模型（models/*.onnx），"
                           "已改用备用识别，准确率较低。")
            return geo
        return RecognizeResult(
            grid=_empty(), source="none",
            error="未找到检测模型，且备用识别未能定位棋盘。"
                  "请运行 tools/download_models.py，或使用「手动摆局」")

    geo = _recognize_geometric(img_bgr)
    if geo is not None:
        return geo
    return RecognizeResult(grid=_empty(), source="none", error="未能定位棋盘")


def _empty():
    return [[None] * 9 for _ in range(10)]


def _crop_box_from_dets(dets: List[Detection], shape) -> Optional[tuple]:
    """按棋子检测框估算棋盘区域（裁剪预处理用）；估不出返回 None。

    YOLO 的 board 类在某些皮肤上会把装饰边框算进棋盘，而棋子检测是
    逐棋子的、不受边框影响——用棋子范围反推棋盘区域最可靠。
    """
    pieces = [d for d in dets
              if d.label != "board" and d.label in CLASS_TO_FEN
              and d.score >= 0.25]
    if len(pieces) < 4:
        return None
    x1 = min(p.box[0] for p in pieces)
    y1 = min(p.box[1] for p in pieces)
    x2 = max(p.box[2] for p in pieces)
    y2 = max(p.box[3] for p in pieces)
    w, h = x2 - x1, y2 - y1
    if w < 60 or h < 60:
        return None
    # 外扩约一格半：保证边行棋子完整、含少量棋盘边缘余量
    pad = max(w / 8.0, h / 9.0) * 1.5
    H, W = shape[0], shape[1]
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(W, x2 + pad))
    y2 = int(min(H, y2 + pad))
    if (x2 - x1) * (y2 - y1) > W * H * 0.97:
        return None                    # 裁剪没有实际意义（原图已经很紧）
    return (x1, y1, x2, y2)


def _tight_crop_box(board_box, shape, margin_cells: float = 0.5):
    """贴着棋盘框的紧凑裁剪（+半格余量）。返回 (x1,y1,x2,y2) 或 None。

    用户实测：裁到棋盘边（排除鼓/旗/凳等装饰与工具栏）识别更稳——
    装饰物正是"多出一个炮/相"这类误检的来源之一，且棋盘占满输入。
    """
    bx1, by1, bx2, by2 = board_box
    cell = max((bx2 - bx1) / 8.0, (by2 - by1) / 9.0)
    m = cell * margin_cells
    H, W = shape[0], shape[1]
    x1 = int(max(0, bx1 - m))
    y1 = int(max(0, by1 - m))
    x2 = int(min(W, bx2 + m))
    y2 = int(min(H, by2 + m))
    if x2 - x1 < 50 or y2 - y1 < 50:
        return None
    return (x1, y1, x2, y2)


def _recognize_cropped_image(img_bgr: np.ndarray, cbox, det,
                             templates) -> Optional[RecognizeResult]:
    """在裁剪图上识别并把坐标换算回原图；校验不过返回 None。"""
    x1, y1, x2, y2 = cbox
    sub = img_bgr[y1:y2, x1:x2]
    try:
        dets2 = det.detect(sub)
        br2 = board_from_detections(dets2, img=sub, skin_templates=templates)
    except Exception:
        return None
    if br2 is None or br2.n_pieces < 4:
        return None
    if not board_plausibility(br2)[0]:
        return None
    # 裁剪可信度：识别出的棋盘框贴着裁剪边缘 -> 棋盘很可能被裁掉了一部分
    # （棋子范围估出的裁剪框在误检稀疏时会偏小）。这种"看着合法但缺一块"
    # 的局面不能采用，更不能进缓存被后续帧反复复用
    _bb = br2.board_box
    _t = 6.0
    if (_bb[0] <= _t or _bb[1] <= _t
            or _bb[2] >= (x2 - x1) - _t or _bb[3] >= (y2 - y1) - _t):
        return None
    # 坐标全部换算回原图：代点点击与皮肤模板采集都依赖原图坐标系
    bb = br2.board_box
    br2.board_box = (bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1)
    shifted = [Detection(label=d.label, score=d.score,
                         box=(d.box[0] + x1, d.box[1] + y1,
                              d.box[2] + x1, d.box[3] + y1))
               for d in dets2]
    res = _from_yolo(br2, shifted, skin_classes=len(templates or {}))
    res.warning = ((res.warning + "；" if res.warning else "")
                   + "已按棋盘区域裁剪后识别")
    return res


def _from_yolo(br: BoardResult, dets: List[Detection],
               skin_classes: int = 0) -> RecognizeResult:
    warns = []
    if br.n_corrected:
        warns.append(f"皮肤模板修正了 {br.n_corrected} 处模型的判断，请重点复核")
    if br.n_ambiguous:
        warns.append(f"有 {br.n_ambiguous} 处多个棋子落到同一格")
    if len(br.low_conf) > 3:
        warns.append(f"有 {len(br.low_conf)} 个棋子置信度偏低")
    warning = "；".join(warns) + "。" if warns else None
    return RecognizeResult(grid=br.grid, red_at_top=br.red_at_top, source="yolo",
                           n_pieces=br.n_pieces, n_ambiguous=br.n_ambiguous,
                           detections=dets, board_box=br.board_box,
                           warning=warning, corrections=list(br.corrections),
                           skin_classes=skin_classes)


def _recognize_geometric(img_bgr: np.ndarray) -> Optional[RecognizeResult]:
    """回退方案：霍夫/投影定位棋盘 + 透视变换 + 模板匹配识别棋子。"""
    try:
        from recognition.classifier import classify_point
        from recognition.detector import BoardLocator, auto_detect_corners
    except Exception:
        return None
    corners = auto_detect_corners(img_bgr)
    if corners is None:
        return None
    warped = BoardLocator(corners).warp(img_bgr)
    raw = [[None] * 9 for _ in range(10)]
    for r in range(10):
        for c in range(9):
            state, letter = classify_point(warped, r, c)
            if state == "piece":
                raw[r][c] = letter
    red_top = sum(1 for r in range(5) for c in range(9)
                  if raw[r][c] and raw[r][c].isupper())
    red_bottom = sum(1 for r in range(5, 10) for c in range(9)
                     if raw[r][c] and raw[r][c].isupper())
    red_at_top = red_top > red_bottom
    grid = raw
    if red_at_top:
        grid = [row[:] for row in raw][::-1]
        for row in grid:
            row.reverse()
    n = sum(1 for r in range(10) for c in range(9) if grid[r][c])
    return RecognizeResult(grid=grid, red_at_top=red_at_top, source="geometric",
                           n_pieces=n, board_box=None)
