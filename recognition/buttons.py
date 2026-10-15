# -*- coding: utf-8 -*-
"""对局结束/结算界面的按钮检测（模板匹配）。

用于"一局结束后自动点叉 → 点再来一局 → 进入新对局"的自动化。

做法：把用户提供的真实截图裁出的按钮模板（assets/jj/*.png）在
窗口画面上做多尺度归一化相关匹配（matchTemplate TM_CCOEFF_NORMED），
超过阈值即认为按钮出现，并把匹配位置换算为屏幕点击坐标。

模板与分辨率相关，故做多尺度搜索（0.6~1.5 倍）以适配不同窗口大小。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "jj")

# 模板名 -> (文件名, 匹配阈值)
TEMPLATES = {
    "close": ("close_btn.png", 0.86),   # 结算弹窗内的叉
    "again": ("again_btn.png", 0.70),   # "再来一局"按钮
}

# 搜索区域（画面相对比例 left, top, right, bottom）。
# 关键安全约束：close 只在中部区域寻找——X 字形过于通用，
# 若全图搜索会匹配到窗口/应用自身的关闭按钮，点击会导致直接退出游戏。
ROI = {
    "close": (0.08, 0.10, 0.94, 0.92),
    "again": (0.03, 0.35, 0.97, 0.99),
}

SCALES = (0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5)

# 模板标定来源：参考截图宽度与其上按钮的像素宽度。
# 模板匹配对缩放极敏感（窗口比参考图更宽时按钮按比例变大，
# 超出固定缩放档就匹配不上），因此按"期望缩放"自适应搜索。
REF_WIDTH = 613.0
REF_BTN_WIDTH = {"close": 24.0, "again": 284.0}
# 参考图中「按钮宽 / 棋盘宽」的比值——用棋盘框推算期望尺寸，
# 比用窗口宽度可靠（窗口可能含模拟器工具栏、黑边）
REF_BTN_BOARD_RATIO = {"close": 24.0 / 505.0, "again": 284.0 / 505.0}
REF_BOARD_WIDTH = 505.0

# 按钮中心相对**画面**的标定位置（用于模板匹配失败时的坐标兜底点击）。
# 标定依据：两张真实结算截图（613x1090 与 613x1089）——
# 按钮绝对位置几乎相同（y≈1016）而棋盘大小不同（600 / 477 像素高），
# 说明「再来一局」锚定在应用界面底部，与棋盘大小无关，
# 因此按画面比例换算（横向居中 0.50，纵向距底 6.7%）。
FRAME_ANCHOR = {
    "again": (0.50, 0.933),
}


def _load(name: str) -> Optional[np.ndarray]:
    path = os.path.join(TEMPLATE_DIR, name)
    if not os.path.isfile(path):
        return None
    tpl = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return tpl


_cache: dict = {}


def get_template(key: str) -> Optional[np.ndarray]:
    if key not in _cache:
        fn, _th = TEMPLATES[key]
        _cache[key] = _load(fn)
    return _cache[key]


def expected_scales(key: str, tpl_w: int, frame_w: int,
                    board_box=None) -> List[float]:
    """按参考比例推算期望缩放，并给出邻域档位（自适应分辨率）。

    优先用棋盘框换算（与游戏画布真实缩放一致）；没有棋盘框时退回
    按窗口宽度估算。
    """
    scales: List[float] = []
    exp = None
    ratio = REF_BTN_BOARD_RATIO.get(key)
    if board_box is not None and ratio and tpl_w > 0:
        bw = board_box[2] - board_box[0]
        if bw > 20:
            est_btn_w = bw * ratio * (tpl_w / (REF_BTN_BOARD_RATIO[key] * REF_BOARD_WIDTH))
            exp = est_btn_w / tpl_w
    if exp is None:
        ref_btn = REF_BTN_WIDTH.get(key)
        if ref_btn and frame_w > 0 and tpl_w > 0:
            exp = (frame_w * (ref_btn / REF_WIDTH)) / tpl_w
    if exp:
        scales += [exp * 0.85, exp * 0.95, exp, exp * 1.06, exp * 1.18]
    scales += list(SCALES)
    out: List[float] = []
    for s2 in scales:
        if 0.2 <= s2 <= 3.2 and all(abs(s2 - o) > 0.03 for o in out):
            out.append(s2)
    return out


def settle_button_point(key: str, frame_shape
                        ) -> Optional[Tuple[int, int]]:
    """按画面尺寸换算结算按钮中心坐标（坐标兜底点击用）。

    标定自两张真实结算截图：按钮锚定在应用界面底部（距底约 6.7%）、
    横向居中，与棋盘大小无关。
    """
    anchor = FRAME_ANCHOR.get(key)
    if anchor is None or frame_shape is None or len(frame_shape) < 2:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        return None
    rx, ry = anchor
    return (int(w * rx), int(h * ry))


def find_button(img_bgr: np.ndarray, key: str,
                threshold: float = None,
                board_box=None) -> Optional[Tuple[int, int, float]]:
    """在画面中查找按钮，返回 (中心x, 中心y, 置信度)；未找到返回 None。

    多尺度归一化相关匹配；对整幅做直方图无关的 TM_CCOEFF_NORMED。
    """
    tpl0 = get_template(key)
    if tpl0 is None:
        return None
    th = threshold if threshold is not None else TEMPLATES[key][1]
    if img_bgr is None:
        return None
    ih, iw = img_bgr.shape[:2]
    # 只在安全区域内匹配（返回坐标仍换算为全画面坐标）
    l, t, r, b = ROI.get(key, (0.0, 0.0, 1.0, 1.0))
    x0, y0 = int(iw * l), int(ih * t)
    x1, y1 = int(iw * r), int(ih * b)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    rh, rw = roi.shape[:2]
    best = (None, -1.0)
    for s in expected_scales(key, tpl0.shape[1], iw, board_box):
        tw, thh = int(tpl0.shape[1] * s), int(tpl0.shape[0] * s)
        if tw < 12 or thh < 8 or tw >= rw or thh >= rh:
            continue
        tpl = cv2.resize(tpl0, (tw, thh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
        if max_v > best[1]:
            best = ((max_l[0] + tw // 2, max_l[1] + thh // 2), float(max_v))
    if best[0] is None or best[1] < th:
        return None
    # ROI 内坐标 -> 全画面坐标
    return (int(best[0][0] + x0), int(best[0][1] + y0), float(best[1]))


def find_any(img_bgr: np.ndarray, keys: List[str]
             ) -> Optional[Tuple[str, int, int, float]]:
    """在多个按钮中找出置信度最高的一个。"""
    best = None
    for key in keys:
        hit = find_button(img_bgr, key)
        if hit is None:
            continue
        if best is None or hit[2] > best[3]:
            best = (key, hit[0], hit[1], hit[2])
    return best
