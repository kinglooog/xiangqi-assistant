# -*- coding: utf-8 -*-
"""合成 JJ 风格棋盘截图（用于识别流水线的自动化测试）。

按真实截图的典型样式渲染：木纹底色、深棕格线、圆形棋子、
红方文字正放、黑方文字 180 度倒置。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from recognition.detector import CANVAS_H, CANVAS_W, CELL, MARGIN, canonical_point

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise RuntimeError("需要 Pillow: pip install Pillow")

from core.board import PIECE_NAMES

WOOD_BG = (245, 222, 184)      # 底色 BGR 浅木色
GRID_COLOR = (60, 60, 110)     # 格线 BGR 深棕
RED_TEXT = (40, 40, 200)       # 红字 BGR
BLACK_TEXT = (30, 30, 30)      # 黑字 BGR
FACE_COLOR = (205, 225, 240)   # 棋面 BGR（略偏黄铜）

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]


def render_board(fen_board_rows: List[List[Optional[str]]],
                 pad: int = 40, rotate_black: bool = True) -> np.ndarray:
    """把 10x9 字母矩阵渲染成一张"截图"。返回 BGR 图像。

    fen_board_rows: row0 为黑方底线（与 FEN/识别矩阵同向）。
    rotate_black: 黑方棋子文字是否倒置（模拟 JJ 视角）。
    """
    w = CANVAS_W + pad * 2
    h = CANVAS_H + pad * 2
    img = Image.new("RGB", (w, h), WOOD_BG[::-1])
    draw = ImageDraw.Draw(img)

    # 格线
    for c in range(9):
        x = pad + MARGIN + c * CELL
        draw.line([(x, pad + MARGIN), (x, pad + MARGIN + CELL * 9)],
                  fill=GRID_COLOR[::-1], width=2)
    for r in range(10):
        y = pad + MARGIN + r * CELL
        draw.line([(pad + MARGIN, y), (pad + MARGIN + CELL * 8, y)],
                  fill=GRID_COLOR[::-1], width=2)
    # 九宫斜线
    for (c1, r1), (c2, r2) in (((3, 0), (5, 2)), ((5, 0), (3, 2)),
                               ((3, 7), (5, 9)), ((5, 7), (3, 9))):
        draw.line([(pad + MARGIN + c1 * CELL, pad + MARGIN + r1 * CELL),
                   (pad + MARGIN + c2 * CELL, pad + MARGIN + r2 * CELL)],
                  fill=GRID_COLOR[::-1], width=2)

    font_path = next((p for p in _FONT_CANDIDATES if __import__("os").path.exists(p)), None)
    font_size = int(CELL * 0.62)

    for r in range(10):
        for c in range(9):
            letter = fen_board_rows[r][c]
            if letter is None:
                continue
            x, y = pad + MARGIN + c * CELL, pad + MARGIN + r * CELL
            radius = int(CELL * 0.46)
            # 棋子底圆 + 边圈
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=FACE_COLOR[::-1], outline=GRID_COLOR[::-1], width=2)
            cn = PIECE_NAMES[letter]
            color = RED_TEXT[::-1] if letter.isupper() else BLACK_TEXT[::-1]
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), cn, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            char_img = Image.new("RGB", (tw + 4, th + 4), FACE_COLOR[::-1])
            ImageDraw.Draw(char_img).text((2 - bbox[0], 2 - bbox[1]), cn,
                                          fill=color, font=font)
            if rotate_black and letter.islower():
                char_img = char_img.rotate(180)
            img.paste(char_img, (x - char_img.width // 2, y - char_img.height // 2))

    # 底部加一条状态栏，模拟真实截图布局
    arr = np.array(img)[:, :, ::-1].copy()
    return arr
