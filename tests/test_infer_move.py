# -*- coding: utf-8 -*-
"""转移约束推断测试：用"当前局面+合法走法"解释观察帧，
容错同色异子误读（車→象/炮 类皮肤误差），歧义靠起点空缺消解。"""
from core.board import Board, infer_move_from_observation, move_to_iccs


def test_misread_tolerated():
    """車 i0->i2（车一进二），终点車被模型误读成炮 -> 仍应正确推断。"""
    b = Board()
    g = [row[:] for row in b.grid]
    g[7][8] = g[9][8]
    g[9][8] = None
    g[7][8] = 'C'                      # 同色异子误读
    mv, score = infer_move_from_observation(b, g)
    assert mv == ((9, 8), (7, 8)), f"应推断 车一进二, got {mv}"
    assert score >= 0.99


def test_exact_move():
    """无误读的普通走法。"""
    b = Board()
    g = [row[:] for row in b.grid]
    g[7][8] = g[9][8]
    g[9][8] = None
    mv, score = infer_move_from_observation(b, g)
    assert mv == ((9, 8), (7, 8))
    assert score >= 0.99


def test_capture():
    """炮吃马（隔炮架）并带误读：吃子形态也能推断。"""
    b = Board("4n4/9/9/9/4P4/9/9/9/4C4/4K4 w - - 0 1")
    g = [row[:] for row in b.grid]
    g[0][4] = 'B'                      # 吃掉黑马后，炮被误读成相
    g[8][4] = None
    mv, score = infer_move_from_observation(b, g)
    assert mv == ((8, 4), (0, 4)), f"应推断 炮吃马, got {mv}"


def test_two_knights_disambiguated_by_origin():
    """两匹马都能到同一格：观察帧里哪个起点空了就走的那匹。"""
    b = Board("4k4/9/9/9/9/9/9/1N1N5/9/3K5 w - - 0 1")
    g = [row[:] for row in b.grid]
    g[7][1] = None                     # (7,1) 的马走了
    g[5][2] = 'N'
    mv, score = infer_move_from_observation(b, g)
    assert mv == ((7, 1), (5, 2))


def test_unexplainable_returns_none():
    """观察帧无法用任何合法走法解释（如棋子凭空消失）-> None。"""
    b = Board()
    g = [row[:] for row in b.grid]
    g[6][8] = g[9][8]                  # 車瞬移到有自家兵的格子（不合法）
    g[9][8] = None
    g[6][6] = None                     # 还多丢一个兵
    mv, _ = infer_move_from_observation(b, g)
    assert mv is None


def test_empty_board_returns_none():
    b = Board()
    mv, _ = infer_move_from_observation(b, [[None] * 9 for _ in range(10)])
    assert mv is None


def test_no_change_returns_none_or_high():
    """观察帧与当前局面相同：无走法可解释 -> None（上层按无变化处理）。"""
    b = Board()
    g = [row[:] for row in b.grid]
    mv, _ = infer_move_from_observation(b, g)
    assert mv is None
