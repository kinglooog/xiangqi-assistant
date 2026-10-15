# -*- coding: utf-8 -*-
"""棋子识别：空位判断、红黑方判断、棋子文字识别（模板匹配）。

针对常见网络象棋皮肤（JJ象棋、天天象棋等圆形棋子，深/红文字 + 浅色棋面）调优。
识别结果始终允许人工校正，本模块只需把大概率正确的局面呈现给用户。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from recognition.detector import CANVAS_H, CANVAS_W, CELL, MARGIN, canonical_point

RED_CHARS = {"K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵"}
BLACK_CHARS = {"k": "将", "a": "士", "b": "象", "n": "马", "r": "车", "c": "炮", "p": "卒"}

TEMPLATE_SIZE = 48
PATCH_R = 30  # 交点采样半径

_WINDOWS_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
]


# ------------------------------------------------------------------ 模板生成
_template_cache: Optional[Dict[str, np.ndarray]] = None


def build_templates() -> Dict[str, np.ndarray]:
    """渲染所有棋子字符模板（二值图，白=笔画），按 FEN 字母索引。

    每个字符用多个字体渲染，运行时匹配取最高分。
    """
    global _template_cache
    if _template_cache is not None:
        return _template_cache
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError("需要 Pillow: pip install Pillow")

    from core.board import PIECE_NAMES
    templates: Dict[str, List[np.ndarray]] = {}
    size = TEMPLATE_SIZE * 3  # 高分辨率渲染再缩小
    fonts = [p for p in _WINDOWS_FONTS if os.path.exists(p)]
    if not fonts:
        fonts = [None]  # PIL 默认字体（可能无中文，尽量用系统字体）

    for letter, cn in PIECE_NAMES.items():
        for font_path in fonts:
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, int(size * 0.8))
                else:
                    font = ImageFont.load_default()
                img = Image.new("L", (size, size), 0)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), cn, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
                          cn, fill=255, font=font)
            except Exception:
                continue
            arr = np.array(img)
            _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
            tpl = _norm_bbox(binary)
            if tpl is not None:
                # 预处理：模糊 + 中心化（皮尔逊相关要求两侧都中心化）
                tpl = cv2.GaussianBlur(tpl, (5, 5), 1.2)
                tpl = tpl - tpl.mean()
                std = tpl.std()
                if std > 1e-6:
                    tpl = tpl / std
                    templates.setdefault(letter, []).append(tpl)
    _template_cache = {k: v for k, v in templates.items()}
    return _template_cache


# ------------------------------------------------------------------ 像素分析
def _remove_grid_lines(mask: np.ndarray) -> np.ndarray:
    """从二值掩码中剔除横、竖、斜（九宫）格线。

    格线贯穿整个采样窗口，而文字笔画长度有限：
    用长核（45px > 窗口宽 75%）做形态学开运算提取格线后扣除。
    """
    if mask.max() == 0:
        return mask
    bin_ = (mask > 100).astype(np.uint8)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)),   # 横线
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)),   # 竖线
        np.eye(45, dtype=np.uint8),                           # 撇向斜线
        np.fliplr(np.eye(45, dtype=np.uint8)),                # 捺向斜线
    ]
    lines = np.zeros_like(bin_)
    for k in kernels:
        lines |= cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k)
    return ((bin_ - lines) > 0).astype(np.uint8) * 255


CENTER_R = 24  # 中心区域半径（只含文字，不含外圈圆环）


def _center_crop(img: np.ndarray, radius: int) -> np.ndarray:
    """只保留采样窗口中心圆形区域。"""
    out = np.zeros_like(img)
    ch, cw = img.shape[:2]
    yy, xx = np.ogrid[:ch, :cw]
    out[(yy - (ch - 1) / 2) ** 2 + (xx - (cw - 1) / 2) ** 2 <= radius ** 2] = \
        img[(yy - (ch - 1) / 2) ** 2 + (xx - (cw - 1) / 2) ** 2 <= radius ** 2]
    return out


def _text_mask(patch_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """提取棋子文字笔画掩码。返回 (二值掩码, 'r'/'b' 判色)。

    做法：中心圆形区域 Otsu 二值化（亮棋面 vs 深文字，与文字颜色无关），
    再剔除贯穿的格线；颜色由掩码内红色像素占比决定。
    """
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray = _center_crop(gray, CENTER_R)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = _remove_grid_lines(mask)
    mask = _center_crop(mask, CENTER_R)

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_pixels = ((h <= 14) | (h >= 165)) & (s > 60) & (v > 60)
    inside = mask > 0
    n_total = int(inside.sum())
    if n_total < 100:
        return np.zeros(patch_bgr.shape[:2], np.uint8), ""
    n_red = int((red_pixels & inside).sum())
    color = "r" if n_red > n_total * 0.35 else "b"
    return mask, color


def _has_piece(patch_bgr: np.ndarray) -> bool:
    """判断交点处是否有棋子：Otsu 掩码笔画数量是否足够。"""
    mask, _ = _text_mask(patch_bgr)
    return mask.sum() / 255 > 150


def _norm_bbox(mask: np.ndarray, size: int = TEMPLATE_SIZE) -> Optional[np.ndarray]:
    """按笔画边界框裁剪并居中缩放到 size x size（浮点 0-1）。

    关键的尺度归一化：掩码中的字通常只占采样窗口的六成左右，
    而模板字符几乎撑满画布，不做边界框对齐会导致大面积误匹配。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), crop.dtype)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32) / 255.0


def _match_char(mask: np.ndarray, color: str) -> Tuple[Optional[str], float]:
    """把文字掩码与模板库匹配。返回 (FEN字母, 相似度)。

    掩码与模板都做边界框归一化 + 高斯模糊后算皮尔逊相关系数，
    对笔画宽度、字符大小、抗锯齿、黑白字色均鲁棒。
    网络象棋中黑方棋子文字常为 180 度倒置（截图视角），
    因此掩码正反两个方向都参与匹配。
    """
    templates = build_templates()
    letters = RED_CHARS if color == "r" else BLACK_CHARS
    base = _norm_bbox(mask)
    if base is None:
        return None, 0.0
    base = cv2.GaussianBlur(base, (5, 5), 1.2)
    base = (base - base.mean()) / (base.std() + 1e-6)
    if not np.isfinite(base).all():
        return None, 0.0
    variants = (base, cv2.rotate(base, cv2.ROTATE_180))
    best_letter, best_score = None, -9.0
    for letter in letters:
        for tpl in templates.get(letter, []):
            for patch in variants:
                score = float((patch * tpl).mean())
                if score > best_score:
                    best_score, best_letter = score, letter
    return best_letter, best_score


# ------------------------------------------------------------------ 顶层接口
def classify_point(warped: np.ndarray, row: int, col: int,
                   min_score: float = 0.5) -> Tuple[str, Optional[str]]:
    """识别标准棋盘图上某个交点。

    返回 (状态, FEN字母)：
    - ('empty', None)            空位
    - ('piece', 'R'/'k'/...)     棋子（大写红方、小写黑方）
    - ('unknown', None)          有棋子但无法识别（GUI 提示人工修正）
    """
    x, y = canonical_point(row, col)
    x0, y0 = max(0, x - PATCH_R), max(0, y - PATCH_R)
    x1, y1 = min(CANVAS_W, x + PATCH_R), min(CANVAS_H, y + PATCH_R)
    patch = warped[y0:y1, x0:x1]
    if patch.size == 0:
        return ("unknown", None)
    if not _has_piece(patch):
        return ("empty", None)
    mask, color = _text_mask(patch)
    if not color:
        return ("unknown", None)
    letter, score = _match_char(mask, color)
    if letter is None or score < min_score:
        return ("unknown", None)
    return ("piece", letter)


def recognize_board(warped: np.ndarray) -> Tuple[List[List[Optional[str]]], float]:
    """识别整张标准棋盘图。

    返回 (10x9 棋子字母矩阵（None=空）, 平均匹配置信度)。
    """
    grid: List[List[Optional[str]]] = [[None] * 9 for _ in range(10)]
    scores: List[float] = []
    for row in range(10):
        for col in range(9):
            state, letter = classify_point(warped, row, col)
            if state == "piece":
                grid[row][col] = letter
                # 重新算一次分值太浪费，这里用简单标记
                scores.append(1.0)
            elif state == "unknown":
                scores.append(0.0)
    conf = (sum(scores) / len(scores)) if scores else 0.0
    return grid, conf


def grid_to_fen(grid: List[List[Optional[str]]], side_to_move: str = "w") -> str:
    """识别矩阵 -> FEN。row0 为黑方底线，与 FEN 行序一致。"""
    rows = []
    for r in range(10):
        row_str, empty = "", 0
        for c in range(9):
            p = grid[r][c]
            if p is None:
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += p
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    return f"{'/'.join(rows)} {side_to_move} - - 0 1"
