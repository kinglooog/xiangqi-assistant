# -*- coding: utf-8 -*-
"""LiveObserver 状态机测试：稳定判定、走子 diff、吃子、chaos、方向变化。"""
import pytest

from core.board import Board, iccs_to_move
from live import LiveObserver


def grid_of(fen: str):
    return [[p for p in row] for row in Board(fen).grid]


def apply_move(grid, move):
    """在矩阵上执行一步（吃子即覆盖）。"""
    g = [row[:] for row in grid]
    (fr, fc), (tr, tc) = move
    g[tr][tc] = g[fr][fc]
    g[fr][fc] = None
    return g


def feed_stable(obs, grid, n=2, red_at_top=False):
    events = []
    for _ in range(n):
        events = obs.feed(grid, red_at_top)
    return events


OPENING = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def test_baseline_on_first_stable():
    obs = LiveObserver()
    g = grid_of(OPENING)
    assert obs.feed(g, False) == []          # 第一帧：等待
    ev = obs.feed(g, False)                  # 第二帧：稳定
    assert ev and ev[0]["type"] == "baseline"
    # 之后相同画面不再有事件
    assert obs.feed(g, False) == []


def test_change_detected_after_stability():
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    # 黑方应了一手 马8进7 —— 单帧出现不稳定，第二帧才稳定
    mv = iccs_to_move("h9g7")
    g1 = apply_move(g0, mv)
    assert obs.feed(g1, False) == []                     # 单帧不确认
    ev = obs.feed(g1, False)
    assert ev and ev[0]["type"] == "changed"
    assert ev[0]["grid"] == g1                           # 走子解释移交推断层


def test_unstable_flicker_ignored():
    """中间闪现一帧错误识别（如动画半途），不产生事件也不丢基准。"""
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    g1 = apply_move(g0, iccs_to_move("h9g7"))
    g_bad = [row[:] for row in g1]
    g_bad[3][0] = None                                    # 造一个缺子的坏帧
    obs.feed(g_bad, False)
    assert obs.feed(g0, False) == []                      # 回到原局面：无事件
    ev = feed_stable(obs, g1)
    assert ev and ev[0]["type"] == "changed"              # 真实走子仍可检出


def test_capture_change():
    """吃子（落点原有对方棋子）同样表现为 changed 事件。"""
    fen = "4n4/9/9/9/4P4/9/9/9/4C4/4K4 w - - 0 1"
    obs = LiveObserver()
    g0 = grid_of(fen)
    feed_stable(obs, g0)
    mv = iccs_to_move("e1e9")                             # 炮吃马（有炮架）
    g1 = apply_move(g0, mv)
    ev = feed_stable(obs, g1)
    assert ev and ev[0]["type"] == "changed"
    assert ev[0]["grid"] == g1


def test_multi_change_reports_changed():
    """多格同时变化：观察器只报 changed，是否合法由推断层判断。"""
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    g1 = [row[:] for row in g0]
    g1[3][0] = None                                       # 卒没了
    g1[6][0] = None                                       # 兵也没了
    ev = feed_stable(obs, g1)
    assert ev and ev[0]["type"] == "changed"
    assert ev[0]["grid"] == g1


def test_chaos_then_force_baseline():
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    g1 = grid_of("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")     # 完全不同的局面
    ev = feed_stable(obs, g1)
    assert ev and ev[0]["type"] == "changed"   # 是否 chaos 由推断层判断
    obs.force_baseline(g1, False)
    assert obs.feed(g1, False) == []
    g2 = apply_move(g1, iccs_to_move("e9e8"))
    ev = feed_stable(obs, g2)
    assert ev and ev[0]["type"] == "changed"


def test_orientation_change_requires_rebaseline():
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    ev = obs.feed(g0, True)                               # 方向翻转
    assert ev and ev[0]["type"] == "orientation_changed"
    # 翻转后的画面按新基准重新建立
    ev = feed_stable(obs, [row[::-1] for row in g0[::-1]], red_at_top=True)
    assert ev and ev[0]["type"] == "baseline"


def test_board_missing_resets_candidate():
    obs = LiveObserver()
    g0 = grid_of(OPENING)
    feed_stable(obs, g0)
    assert obs.feed(None, False) == []                    # 棋盘丢失：静默
    g1 = apply_move(g0, iccs_to_move("h9g7"))
    ev = feed_stable(obs, g1)
    assert ev and ev[0]["type"] == "changed"
