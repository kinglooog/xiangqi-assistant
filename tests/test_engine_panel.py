# -*- coding: utf-8 -*-
"""引擎评估面板的红黑视角换算测试。

这是最容易出错、且用户最容易看混的地方：引擎给出的分数是「行棋方视角」，
面板必须换算成「红方视角」再显示，否则领先方会被显示反。
"""
import pytest

from core.board import BLACK, RED
from gui.engine_panel import EnginePanel, cp_to_winrate, red_cp


# ---------------------------------------------------------------- 视角换算
def test_red_cp_red_to_move():
    """红方行棋：引擎分数即红方视角，正数=红优。"""
    assert red_cp(150, RED) == 150
    assert red_cp(-80, RED) == -80


def test_red_cp_black_to_move_flips_sign():
    """黑方行棋：引擎分数是黑方视角，必须取反才是红方视角。"""
    assert red_cp(150, BLACK) == -150
    assert red_cp(-80, BLACK) == 80


def test_red_cp_none():
    assert red_cp(None, RED) is None


# ---------------------------------------------------------------- 文字描述
@pytest.fixture
def panel(qapp):
    return EnginePanel()


def test_lead_text_names_the_leading_side(panel):
    """文字必须显式写出谁领先，不能只给一个裸数字。"""
    assert "红方领先" in panel._fmt_lead(120)
    assert "黑方领先" in panel._fmt_lead(-120)
    assert panel._fmt_lead(0) == "双方均势"
    assert panel._fmt_lead(0.5) == "双方均势"


def test_lead_text_mate(panel):
    assert "红方" in panel._fmt_lead(3, "mate") and "步内取胜" in panel._fmt_lead(3, "mate")
    assert "黑方" in panel._fmt_lead(-3, "mate")


def test_winrate_monotonic():
    assert cp_to_winrate(0) == pytest.approx(50.0)
    assert cp_to_winrate(600) > 85
    assert cp_to_winrate(-600) < 15


# ---------------------------------------------------------------- 端到端面板
def _feed(panel, score, stm, my_color, kind="cp", depth=20):
    panel.set_my_color(my_color)
    panel.set_stm(stm)
    panel._on_info({"multipv": 1, "depth": depth, "score": score,
                    "score_kind": kind, "stm": stm, "pv": ["a0a1"], "pv_cn": []})


def test_panel_black_advantage_shown_correctly(panel):
    """黑方行棋、黑方大优 -> 应显示"黑方领先"，且执黑者看到"你领先"。"""
    _feed(panel, 300, BLACK, BLACK)      # 引擎+300 是黑方视角 -> 红方视角 -300
    assert "黑方领先" in panel.eval_label.text()
    assert "你领先" in panel.myside_label.text()
    assert "黑方胜率" in panel.winrate_label.text()


def test_panel_red_advantage_for_black_player(panel):
    """红方大优、玩家执黑 -> 显示红方领先，且执黑者看到"你落后"。"""
    _feed(panel, 300, RED, BLACK)
    assert "红方领先" in panel.eval_label.text()
    assert "你落后" in panel.myside_label.text()


def test_panel_mate_against_me(panel):
    """黑方行棋且黑方即将杀死红方 -> 执红者看到"你即将被将死"。"""
    _feed(panel, -2, BLACK, RED, kind="mate")   # 引擎 -2 是黑方视角 -> 红方视角 +2
    # 红方视角 +2 表示红方两步内取胜
    assert "红方" in panel.eval_label.text()
    assert "你即将取胜" in panel.myside_label.text()


def test_panel_turn_label(panel):
    panel.set_stm(RED)
    assert "红方" in panel.turn_label.text()
    panel.set_stm(BLACK)
    assert "黑方" in panel.turn_label.text()


def test_pv_list_marks_side_to_move(panel):
    """候选着法属于当前行棋方，列表需标出是红方还是黑方的着法。"""
    _feed(panel, 100, BLACK, BLACK)
    item = panel.pv_list.item(0).text()
    assert "[黑方]" in item
    _feed(panel, 100, RED, RED)
    item = panel.pv_list.item(0).text()
    assert "[红方]" in item


def test_panel_clears_on_new_search(panel):
    _feed(panel, 300, RED, RED)
    assert "红方领先" in panel.eval_label.text()
    panel.clear_analysis()
    assert panel.eval_label.text() == "评估：等待分析…"
    assert panel.pv_list.count() == 0
