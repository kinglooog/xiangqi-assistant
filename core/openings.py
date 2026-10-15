# -*- coding: utf-8 -*-
"""常用开局走法库。

每个开局定义一串**红方视角**的 ICCS 着法；轮到黑方时自动做镜像
（纵线不变、横线号 9-n，即 180 度旋转——象棋的天然对称）。

用途：接管时指定开局，软件在轮到我方时优先走这些着法，
走完（或着法在当前局面不可行）后恢复由引擎选择最优着法。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# (显示名, 着法说明, 红方着法序列)
OPENINGS: List[Tuple[str, str, List[str]]] = [
    ("", "引擎选最优（默认）", []),
    ("中炮（当头炮）", "炮二平五", ["h2e2"]),
    ("过宫炮", "炮二平六", ["h2d2"]),
    ("士角炮", "炮二平四", ["h2f2"]),
    ("巡河炮", "炮二进二", ["h2h4"]),
    ("起马局", "马二进三", ["h0g2"]),
    ("飞相局", "相三进五", ["g0e2"]),
    ("仙人指路", "兵七进一", ["c3c4"]),
    ("双铁滑车", "车一进一、车九进一", ["i0i1", "a0a1"]),
    ("御驾亲征", "帅五进一", ["e0e1"]),
]

_BY_NAME = {name: moves for name, _tip, moves in OPENINGS}


def _mirror_square(sq: str) -> str:
    """单格镜像：纵线不变，横线号 9-n（180 度旋转）。"""
    return f"{sq[0]}{9 - int(sq[1])}"


def mirror_iccs(mv: str) -> str:
    """红方着法 -> 黑方镜像着法（如 h2e2 炮二平五 -> h7e7 炮8平5）。"""
    if len(mv) != 4:
        return mv
    return _mirror_square(mv[:2]) + _mirror_square(mv[2:4])


def opening_moves(name: str, side: str) -> List[str]:
    """取指定开局在某一方的着法序列（side: 'w' 红 / 'b' 黑）。

    黑方时自动镜像，因此同一套开局红黑通用。
    """
    moves = _BY_NAME.get(name)
    if not moves:
        return []
    if side == "b":
        return [mirror_iccs(m) for m in moves]
    return list(moves)


def opening_tip(name: str) -> str:
    for n, tip, _m in OPENINGS:
        if n == name:
            return tip
    return ""


def opening_names() -> List[str]:
    return [name for name, _tip, _moves in OPENINGS]
