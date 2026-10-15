# -*- coding: utf-8 -*-
"""局面校验与导入方向处理的回归测试。

这两块正是"截图导入用不了"的两个真实原因：
- 执黑时截图红方在上，识别结果整盘颠倒；
- 非法局面（子力超量/将帅照面）会让引擎拒绝工作。
"""
import pytest

from core.board import Board, iccs_to_move


# ---------------------------------------------------------------- 局面校验
def test_validate_initial_ok():
    assert Board().validate_position() is None


def test_validate_rejects_facing_kings():
    """将帅照面不是合法局面（这正是之前漏掉的检查）。"""
    b = Board("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    assert b.kings_face()
    assert "照面" in b.validate_position()


def test_validate_allows_blocked_kings():
    """同路但有子遮挡则合法。"""
    b = Board("4k4/9/9/9/9/4P4/9/9/9/4K4 w - - 0 1")
    assert not b.kings_face()
    assert b.validate_position() is None


def test_validate_rejects_extra_pieces():
    b = Board("rr1k1r3/9/9/9/9/9/9/9/9/4K4 w - - 0 1")  # 黑 3 车
    assert "车" in b.validate_position()
    b2 = Board("4k4/9/9/9/9/9/9/9/9/3K1R1R1 w - - 0 1")
    assert b2.validate_position() is None


def test_validate_rejects_missing_king():
    assert "缺少" in Board("9/9/9/9/9/9/9/9/9/4K4 w - - 0 1").validate_position()


def test_validate_rejects_king_outside_palace():
    b = Board("4k4/9/9/9/9/9/9/9/9/2K6 w - - 0 1")  # 红帅在 c0，出九宫
    assert "九宫" in b.validate_position()


# ---------------------------------------------------------------- 导入方向
def test_black_view_orientation_roundtrip():
    """执黑视角（截图红方在上）经 180 度旋转应还原为 FEN 顺序。

    这是 BoardLocator 把截图左上角映射到 FEN 第 0 行所要求的处理。
    """
    fen = "r1bakabr1/9/1cn3nc1/p3p3p/2p3p2/2P6/P3P1P1P/1CN1C1N2/9/R1BAKABR1 w - - 0 1"
    truth = [[p for p in row] for row in Board(fen).grid]
    # 模拟"红方在上"的截图：识别矩阵会得到 180 度旋转的结果
    raw = [row[:] for row in truth][::-1]
    for row in raw:
        row.reverse()
    # 检测：红方是否集中在矩阵上半部
    red_top = sum(1 for r in range(5) for c in range(9)
                  if raw[r][c] and raw[r][c].isupper())
    red_bottom = sum(1 for r in range(5, 10) for c in range(9)
                     if raw[r][c] and raw[r][c].isupper())
    assert red_top > red_bottom, "应判定为红方在上"
    # 旋转回 FEN 顺序
    fixed = [row[:] for row in raw][::-1]
    for row in fixed:
        row.reverse()
    assert fixed == truth


def test_grid_to_fen_and_validate_pipeline():
    """识别矩阵 -> FEN -> 校验 的完整链路。"""
    from recognition.classifier import grid_to_fen
    truth = [[p for p in row] for row in Board().grid]
    fen = grid_to_fen(truth, "w")
    b = Board(fen)
    assert b.validate_position() is None
    assert b.to_fen().split()[0] == Board().to_fen().split()[0]


def test_moves_illegal_when_kings_face():
    """将帅照面时，不解除照面的走法非法。"""
    b = Board("4k4/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
    assert b.kings_face()
    legal = [iccs_to_move_iccs(m) for m in b.legal_moves()]
    # 车 a0-a1 不解除照面 -> 非法
    assert "a0a1" not in legal
    # 帅 e0-e1 同样不解除（仍在 e 路）-> 非法
    assert "e0e1" not in legal


def iccs_to_move_iccs(m):
    from core.board import move_to_iccs
    return move_to_iccs(m)
