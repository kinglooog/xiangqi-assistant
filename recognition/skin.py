# -*- coding: utf-8 -*-
"""皮肤模板：自校准 + 二次校验。

背景：YOLO 检测模型主要在天天象棋风格的棋子上训练，换成其它皮肤
（如 JJ 象棋的部分棋子主题）会出现系统性误判（如 車→象、車→炮）。

机制：
1. 【采集】用户在导入对话框核对/修正识别结果并点 OK 后，按最终局面
   （含人工修正）自动裁剪每个棋子的棋面模板并保存。人工修正恰好
   提供了正确标签——用得越多，模板越准。
2. 【校验】之后的每次识别，把每个检出棋子与皮肤模板比对；
   当模板结果与模型结果**明显不一致**（相关系数差距清晰）时，
   以模板为准修正，并计入修正列表供界面提示。
3. 【安全】模板相关度不足或前两名差距模糊时不干预——保证在
   模板不匹配的皮肤（或从未校准）时不劣化原识别。

模板存储：models/skin_templates.npz（每类若干张 64x64 二值掩码，
每类上限 6 张、保留最新，适配用户中途更换皮肤的情况）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from recognition.yolo import CLASS_TO_FEN

TEMPLATE_SIZE = 64
MAX_PER_CLASS = 6
CROP_SHRINK = 0.72        # 检测框向中心收缩比例（聚焦棋面、去外圈）
MIN_CORR = 0.30           # 低于该相关度的模板意见不予采纳
MIN_MARGIN = 0.08         # 最佳与次佳类相关度差需超过该值才采信


def default_skin_path() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models")
    return os.path.join(base, "skin_templates.npz")


def norm_crop(img_bgr: np.ndarray, box: Tuple[float, float, float, float],
              shrink: float = CROP_SHRINK) -> Optional[np.ndarray]:
    """按检测框裁剪棋面：灰度 -> Otsu 笔画掩码 -> 边界框归一化 64x64。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sw, sh = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    ix1, iy1 = int(max(0, cx - sw)), int(max(0, cy - sh))
    ix2, iy2 = int(cx + sw), int(cy + sh)
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return None
    patch = cv2.cvtColor(img_bgr[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(patch, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) < 12:
        return None
    x0, x1b, y0, y1b = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1b + 1, x0:x1b + 1]
    ch, cw = crop.shape
    s = max(ch, cw)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - ch) // 2:(s - ch) // 2 + ch, (s - cw) // 2:(s - cw) // 2 + cw] = crop
    n = cv2.resize(sq, (TEMPLATE_SIZE, TEMPLATE_SIZE),
                   interpolation=cv2.INTER_AREA)
    return (n > 100).astype(np.float32)


def _prep(mask: np.ndarray) -> Optional[np.ndarray]:
    m = cv2.GaussianBlur(mask, (5, 5), 1.2)
    sd = m.std()
    if sd < 1e-6:
        return None
    return (m - m.mean()) / sd


def _match(mask: np.ndarray,
           templates: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float, float]:
    """与全部模板比对（黑方倒置文字兼容：掩码正反各试一次）。

    返回 (最佳类别, 最佳相关度, 与次佳类别的差距)。
    """
    m = _prep(mask)
    if m is None:
        return None, 0.0, 0.0
    variants = (m, cv2.rotate(m, cv2.ROTATE_180))
    best, second = (None, -9.0), (None, -9.0)
    for letter, lst in templates.items():
        top = -9.0
        for t in lst:
            for v in variants:
                sc = float((v * t).mean())
                if sc > top:
                    top = sc
        if top > best[1]:
            second = best
            best = (letter, top)
        elif top > second[1]:
            second = (letter, top)
    return best[0], best[1], best[1] - second[1]


# ------------------------------------------------------------------ 采集/存取
def harvest(img_bgr: np.ndarray, detections, board_box, red_at_top: bool,
            grid_fen: List[List[Optional[str]]],
            templates: Dict[str, List[np.ndarray]] = None
            ) -> Dict[str, List[np.ndarray]]:
    """按（经人工确认的）局面采集各类棋子模板。

    grid_fen 为 FEN 顺序的最终局面（用户修正后）。
    返回合并后的模板字典（每类保留最新 MAX_PER_CLASS 张）。
    """
    templates = {k: list(v) for k, v in (templates or {}).items()}
    bx1, by1, bx2, by2 = board_box
    gw, gh = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
    # 与识别同源的"框外近一格"过滤 + 单元格映射（含边行回夹）：
    # 两边映射不一致时，会把邻格棋子当成本格类别采集，污染模板
    from recognition.yolo_board import map_center_to_cell
    mx, my = gw * 0.6, gh * 0.6
    for p in detections:
        if p.label == "board":
            continue
        cx, cy = p.center
        if not (bx1 - mx <= cx <= bx2 + mx and by1 - my <= cy <= by2 + my):
            continue
        cell = map_center_to_cell(cx, cy, board_box, gw, gh)
        if cell is None:
            continue
        row, col = cell
        if red_at_top:                       # 与 board_from_detections 同样的翻转
            row, col = 9 - row, 8 - col
        letter = grid_fen[row][col]
        if letter is None:
            continue
        mask = norm_crop(img_bgr, p.box)
        if mask is None:
            continue
        lst = templates.setdefault(letter, [])
        lst.append(mask)
        if len(lst) > MAX_PER_CLASS:
            del lst[0]                       # 保留最新
    return templates


def save_templates(path: str, templates: Dict[str, List[np.ndarray]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays, meta = {}, []
    for i, (letter, lst) in enumerate(sorted(templates.items())):
        if not lst:
            continue
        arrays[f"cls_{i}"] = np.stack(lst)
        meta.append(letter)
    np.savez_compressed(path, meta=np.array(meta), **arrays)


def load_templates(path: str) -> Optional[Dict[str, List[np.ndarray]]]:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        meta = [str(x) for x in z["meta"]]
        out: Dict[str, List[np.ndarray]] = {}
        for i, letter in enumerate(meta):
            out[letter] = [m for m in z[f"cls_{i}"]]
        return out or None
    except Exception:
        return None


# ------------------------------------------------------------------ 校验
def verify_pieces(img_bgr: np.ndarray, detections, board_box, red_at_top: bool,
                  templates: Dict[str, List[np.ndarray]]
                  ) -> List[Tuple[int, int, str, str, float]]:
    """用皮肤模板逐子校验模型判断。

    返回修正列表 [(row, col, 模型判, 模板判, 相关度)]，坐标为 **FEN 顺序**
    （便于界面直接定位）。只在模板意见清晰时给出修正。
    """
    bx1, by1, bx2, by2 = board_box
    gw, gh = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
    corrections: List[Tuple[int, int, str, str, float]] = []
    for p in detections:
        if p.label == "board" or p.label not in CLASS_TO_FEN:
            continue
        cx, cy = p.center
        col = int(round((cx - bx1) / gw))
        row = int(round((cy - by1) / gh))
        if not (0 <= col <= 8 and 0 <= row <= 9):
            continue
        fen_row, fen_col = (9 - row, 8 - col) if red_at_top else (row, col)
        model_letter = CLASS_TO_FEN[p.label]
        mask = norm_crop(img_bgr, p.box)
        if mask is None:
            continue
        pred, corr, margin = _match(mask, templates)
        if pred is None or corr < MIN_CORR or margin < MIN_MARGIN:
            continue                          # 模板意见不清晰：不干预
        if pred != model_letter:
            corrections.append((fen_row, fen_col, model_letter, pred, corr))
    return corrections
