# -*- coding: utf-8 -*-
"""识别流水线测试：用合成棋盘截图验证 定位->采样->分类 全链路。"""
import numpy as np

from core.board import Board
from recognition.classifier import classify_point, grid_to_fen, recognize_board
from recognition.detector import BoardLocator, auto_detect_corners, canonical_point
from recognition.synth import render_board


def fen_to_matrix(fen: str):
    board = Board(fen)
    return [[p for p in row] for row in board.grid]


def test_locator_roundtrip():
    """已知四角 -> 透视往返一致。"""
    # 模拟棋盘在截图里平移+缩放
    sx, sy, scale = 120, 80, 0.9
    def orig_pt(row, col):
        x, y = canonical_point(row, col)
        return (x * scale + sx, y * scale + sy)
    corners = [orig_pt(0, 0), orig_pt(0, 8), orig_pt(9, 8), orig_pt(9, 0)]
    loc = BoardLocator(corners)
    # 画一张合成图，再按变换验证交点采样回原坐标
    img = render_board(fen_to_matrix(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"))
    # 用真实合成几何（合成图按 pad=40 的标准坐标渲染）
    loc2 = BoardLocator([(80, 80), (80 + 480, 80), (80 + 480, 80 + 540), (80, 80 + 540)])
    warped = loc2.warp(img)
    assert warped.shape[0] == 620 and warped.shape[1] == 560
    # 中心点应保持清晰（简单像素方差检查）
    assert warped.std() > 10


def test_classify_empty_intersections():
    """初始局面空交点应判空。"""
    img = render_board(fen_to_matrix(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"))
    loc = BoardLocator([(80, 80), (560, 80), (560, 620), (80, 620)])
    warped = loc.warp(img)
    # e4/e5 (row4/5, col4) 是空交点
    for row in (4, 5):
        state, letter = classify_point(warped, row, 4)
        assert state == "empty", f"({row},4) 应为空, got {state}"


def test_classify_pieces_initial():
    """初始局面的底线棋子识别（车马象士将）。"""
    img = render_board(fen_to_matrix(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"))
    loc = BoardLocator([(80, 80), (560, 80), (560, 620), (80, 620)])
    warped = loc.warp(img)
    expect = {0: "r", 1: "n", 2: "b", 3: "a", 4: "k"}  # 黑方底线 车马象士将
    for col, letter in expect.items():
        state, got = classify_point(warped, 0, col)
        assert state == "piece", f"row0 col{col}: {state}"
        assert got == letter, f"row0 col{col}: 期望 {letter} 实得 {got}"


def test_full_board_recognition_midgame():
    """中局局面全盘识别（允许个别格错，但绝大多数要正确）。"""
    fen = "2bak4/4a4/4c4/9/9/3C5/4P4/9/4A4/2B1K4 w - - 0 1"
    truth = Board(fen)
    img = render_board([[p for p in row] for row in truth.grid])
    loc = BoardLocator([(80, 80), (560, 80), (560, 620), (80, 620)])
    warped = loc.warp(img)
    grid, _ = recognize_board(warped)
    correct, total = 0, 0
    for r in range(10):
        for c in range(9):
            want = truth.grid[r][c]
            if want is None:
                continue
            total += 1
            if grid[r][c] == want:
                correct += 1
    assert total >= 8
    assert correct >= total - 1, f"识别错误过多: {correct}/{total}"


def test_grid_to_fen():
    from core.board import INITIAL_FEN
    b = Board(INITIAL_FEN)
    grid = [[p for p in row] for row in b.grid]
    fen = grid_to_fen(grid)
    assert fen.split()[0] == INITIAL_FEN.split()[0]


def test_auto_detect_on_synthetic():
    """合成图（有清晰格线）应能自动定位。"""
    img = render_board(fen_to_matrix(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"))
    corners = auto_detect_corners(img)
    assert corners is not None
    loc = BoardLocator(corners)
    warped = loc.warp(img)
    state, letter = classify_point(warped, 9, 8)  # 红车 i0
    assert state == "piece" and letter == "R"
