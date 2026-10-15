# -*- coding: utf-8 -*-
"""棋盘定位与网格提取。

统一采用"标准棋盘坐标系"：90 个交点映射到一张标准图上，
- 交点间距 CELL=60 像素，边距 MARGIN=40；
- 标准图尺寸 W = 40*2 + 60*8 = 560，H = 40*2 + 60*9 = 620；
- 交点 (row, col) 位于 (MARGIN + col*CELL, MARGIN + row*CELL)，row0 在上（黑方底线）。

输入为 OpenCV BGR 图像。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

CELL = 60
MARGIN = 40
CANVAS_W = MARGIN * 2 + CELL * 8
CANVAS_H = MARGIN * 2 + CELL * 9

Point = Tuple[float, float]


def canonical_point(row: int, col: int) -> Tuple[int, int]:
    """标准图上的交点像素坐标。"""
    return (MARGIN + col * CELL, MARGIN + row * CELL)


class BoardLocator:
    """一个已定位棋盘的透视变换（截图坐标 <-> 标准坐标）。"""

    def __init__(self, corners: Sequence[Point]):
        """corners: 截图上棋盘 9x10 网格的四个角交点，
        顺序：左上、右上、右下、左下（左上 = 交点(col0,row0)，即黑方 a9）。"""
        assert len(corners) == 4
        self.corners = [(float(x), float(y)) for x, y in corners]
        dst = np.float32([
            canonical_point(0, 0), canonical_point(0, 8),
            canonical_point(9, 8), canonical_point(9, 0),
        ])
        src = np.float32(self.corners)
        self.to_canvas = cv2.getPerspectiveTransform(src, dst)
        self.to_orig = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img_bgr: np.ndarray) -> np.ndarray:
        """把截图透视变换成标准棋盘图。"""
        return cv2.warpPerspective(img_bgr, self.to_canvas, (CANVAS_W, CANVAS_H))

    def canvas_to_orig(self, row: int, col: int) -> Point:
        """标准交点 -> 截图坐标。"""
        p = np.float32([[canonical_point(row, col)]])
        q = cv2.perspectiveTransform(p, self.to_orig)
        return (float(q[0][0][0]), float(q[0][0][1]))


def _line_binary(img_bgr: np.ndarray) -> np.ndarray:
    """提取深色线条的二值图。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    return binary


def _profile_peaks(profile: np.ndarray, min_score: float) -> List[int]:
    """从投影曲线中取峰（局部极大且高于阈值），合并相邻。"""
    peaks = []
    n = len(profile)
    for i in range(1, n - 1):
        if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] \
                and profile[i] >= min_score:
            peaks.append(i)
    # 合并间距过小的峰（保留得分高者）
    merged: List[int] = []
    for p in peaks:
        if merged and p - merged[-1] <= 8:
            if profile[p] > profile[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def auto_detect_corners(img_bgr: np.ndarray) -> Optional[List[Point]]:
    """自动检测棋盘 4 个角交点。

    思路：形态学长核提取横/竖线 -> 投影找峰 -> 由等距格点外推补全缺失线 ->
    按线条覆盖率验证，输出 9 列 x 10 行的完整格线坐标。
    成功返回 [左上, 右上, 右下, 左下]，失败返回 None（调用方转手动校准）。
    """
    h, w = img_bgr.shape[:2]
    binary = _line_binary(img_bgr)

    # 长核开运算：只保留贯穿性直线（棋盘格线被棋子断开处用中等核 + 投影弥补）
    kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 8)))
    kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 8), 1))
    vlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern_v)
    hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern_h)

    col_profile = vlines.sum(axis=0) / 255.0
    row_profile = hlines.sum(axis=1) / 255.0
    xs = _profile_peaks(col_profile, min_score=h * 0.10)
    ys = _profile_peaks(row_profile, min_score=w * 0.10)
    if len(xs) < 5 or len(ys) < 3:
        return None

    xs_full = _complete_lattice(binary, xs, 9, w, axis=0)
    ys_full = _complete_lattice(binary, ys, 10, h, axis=1)
    if xs_full is None or ys_full is None:
        return None

    # 用原始二值图的线条覆盖率验证（棋子遮挡下也应有足够线条像素）
    if not _verify_lines(binary, xs_full, ys_full):
        return None
    return [(xs_full[0], ys_full[0]), (xs_full[-1], ys_full[0]),
            (xs_full[-1], ys_full[-1]), (xs_full[0], ys_full[-1])]


def _complete_lattice(binary: np.ndarray, peaks: List[int], count: int,
                      span: int, axis: int, cell: int = CELL) -> Optional[List[int]]:
    """把部分检测到的格线峰补全为 count 条真实存在的等距格线。

    允许缩放：先估计实际间距（相对棋盘宽度的比例），再对齐相位。
    """
    if len(peaks) < 3:
        return None
    # 估计间距：取相邻峰间距的中位数（忽略跨缺失线的间隔：
    # 用所有间距对最小非零间隔的倍数关系拟合）
    diffs = np.diff(peaks).astype(float)
    base = float(np.min(diffs))
    if base < 12:
        return None
    # 间距 = base 或 base 的整数倍（中间有缺失）
    spacings = [d / round(d / base) for d in diffs if round(d / base) >= 1]
    spacing = float(np.median(spacings))
    if not (0.6 * cell <= spacing <= 1.6 * cell):
        return None
    # 相位：所有峰对 spacing 取模的圆均值
    phases = [p % spacing for p in peaks]
    ang = np.mean([np.exp(1j * 2 * np.pi * ph / spacing) for ph in phases])
    phase = float((np.angle(ang) / (2 * np.pi) * spacing) % spacing)
    # 生成候选线并按支持度筛选
    cands = []
    k = 0
    while phase + k * spacing < span:
        cands.append(int(round(phase + k * spacing)))
        k += 1
    return _select_count(binary, cands, peaks, count, spacing, axis)


def _line_support(binary: np.ndarray, pos: int, axis: int) -> float:
    """某条候选格线上的深色像素覆盖率（容忍棋子遮挡）。axis=0 竖线 / 1 横线。"""
    h, w = binary.shape[:2]
    tol = 3
    if axis == 0:
        x0, x1 = max(0, pos - tol), min(w, pos + tol + 1)
        return float(binary[:, x0:x1].max(axis=1).mean() / 255.0)
    y0, y1 = max(0, pos - tol), min(h, pos + tol + 1)
    return float(binary[y0:y1, :].max(axis=0).mean() / 255.0)


def _select_count(binary: np.ndarray, cands: List[int], peaks: List[int],
                  count: int, spacing: float, axis: int) -> Optional[List[int]]:
    """从候选线中选出连续 count 条：每条必须有足够线条支持度，
    且与检测峰的对齐误差最小。"""
    if len(cands) < count:
        return None
    support = {c: _line_support(binary, c, axis) for c in cands}
    best, best_key = None, None
    for i in range(len(cands) - count + 1):
        grp = cands[i:i + count]
        if any(support[c] < 0.15 for c in grp):
            continue
        err = 0.0
        for p in peaks:
            d = min(abs(p - g) for g in grp)
            err += min(d, spacing / 2)
        err /= len(peaks)
        total_support = sum(support[c] for c in grp)
        key = (round(err, 1), -round(total_support, 2))
        if best_key is None or key < best_key:
            best_key, best = key, grp
    if best is None:
        return None
    return best


def _verify_lines(binary: np.ndarray, xs: List[int], ys: List[int]) -> bool:
    """验证格线：每条竖/横线在二值图上的深色像素占比（容忍棋子遮挡）。"""
    h, w = binary.shape[:2]
    tol = 3
    for x in xs:
        x0, x1 = max(0, x - tol), min(w, x + tol + 1)
        col = binary[:, x0:x1].max(axis=1) / 255.0
        if col.mean() < 0.18:
            return False
    for y in ys:
        y0, y1 = max(0, y - tol), min(h, y + tol + 1)
        row = binary[y0:y1, :].max(axis=0) / 255.0
        if row.mean() < 0.18:
            return False
    return True
