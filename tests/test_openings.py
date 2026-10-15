# -*- coding: utf-8 -*-
"""开局走法功能测试：镜像换算、优先走开局、每局复位、不可行时回退引擎。"""
import pytest

from core.board import Board, iccs_to_move, move_to_iccs, move_to_chinese
from core.openings import OPENINGS, mirror_iccs, opening_moves, opening_names

OPENING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


# ---------------------------------------------------------------- 开局库
def test_opening_library_legal_for_both_sides():
    """每个开局的首着在红黑双方都必须合法，且中文记法可读。"""
    for name, _tip, moves in OPENINGS:
        if not name:
            continue
        assert moves, f"{name} 缺少着法"
        for side in ("w", "b"):
            seq = opening_moves(name, side)
            b = Board(OPENING_FEN)
            b.side_to_move = side
            mv = iccs_to_move(seq[0])
            assert b.is_legal_move(mv), f"{name} 执{side} 的首着非法: {seq[0]}"
            assert move_to_chinese(b, mv)


def test_mirror_iccs_roundtrip():
    """镜像换算：红方着法 <-> 黑方着法的 180 度对称。"""
    assert mirror_iccs("h2e2") == "h7e7"      # 炮二平五 -> 炮8平5
    assert mirror_iccs("h7e7") == "h2e2"      # 再镜像回原样
    assert mirror_iccs("h0g2") == "h9g7"      # 马二进三 -> 马8进7
    assert mirror_iccs("c3c4") == "c6c5"      # 兵七进一 -> 卒3进1


def test_mirrored_opening_is_the_same_shape():
    """镜像后的着法在中文记法上与红方对应（如 炮二平五 <-> 炮8平5）。"""
    b = Board(OPENING_FEN)
    assert move_to_chinese(b, iccs_to_move("h2e2")) == "炮二平五"
    b.side_to_move = "b"
    assert move_to_chinese(b, iccs_to_move("h7e7")) == "炮8平5"


def test_opening_names_include_default_first():
    names = opening_names()
    assert names[0] == "", "第一项应为默认（引擎选最优）"
    for want in ("双铁滑车", "巡河炮", "过宫炮", "士角炮",
                 "起马局", "飞相局", "御驾亲征"):
        assert want in names, f"缺少开局：{want}"


# ---------------------------------------------------------------- 接管应用
def _win(qapp):
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    return w


def test_opening_played_instead_of_engine_move(qapp):
    """接管中选定开局：轮到我方时应走开局着法而非引擎着法。"""
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("w")
    w.game.reset(OPENING_FEN)
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("过宫炮"))
    w._reset_opening()
    assert w._opening_queue == ["h2d2"]           # 炮二平六

    pending = w._pending_opening_move()
    assert pending is not None
    mv, cn = pending
    assert move_to_iccs(mv) == "h2d2" and cn == "炮二平六"

    # 走掉后队列推进，不再有待走着法 -> 回退引擎
    w.game.play(mv)
    w._confirm_opening_move(mv)
    assert w._opening_idx == 1
    assert w._pending_opening_move() is None
    w.engine.stop() if w.engine else None


def test_black_side_uses_mirrored_opening(qapp):
    """执黑时开局自动镜像（炮二平五 -> 炮8平5）。"""
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("b")
    w.game.reset(OPENING_FEN)
    w.game.board.side_to_move = "b"
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("中炮（当头炮）"))
    w._reset_opening()
    assert w._opening_queue == ["h7e7"]
    mv, cn = w._pending_opening_move()
    assert move_to_iccs(mv) == "h7e7" and cn == "炮8平5"
    w.engine.stop() if w.engine else None


def test_multi_move_opening_plays_in_sequence(qapp):
    """双铁滑车：两着依次走完（车一进一 -> 车九进一）。"""
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("w")
    w.game.reset(OPENING_FEN)
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("双铁滑车"))
    w._reset_opening()
    assert w._opening_queue == ["i0i1", "a0a1"]
    mv1, cn1 = w._pending_opening_move()
    assert move_to_iccs(mv1) == "i0i1" and cn1 == "车一进一"
    w.game.play(mv1)
    w._confirm_opening_move(mv1)
    # 对手应一手后，我方走第二着
    w.game.play(iccs_to_move("h9g7"))
    mv2, cn2 = w._pending_opening_move()
    assert move_to_iccs(mv2) == "a0a1" and cn2 == "车九进一"
    w.engine.stop() if w.engine else None


def test_new_game_reapplies_opening(qapp):
    """每局都重新应用开局（新开局采纳后队列复位）。"""
    from tests.test_auto_continue import _Res
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("w")
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("飞相局"))
    w._reset_opening()
    mv, _cn = w._pending_opening_move()
    w.game.play(mv)
    w._confirm_opening_move(mv)
    assert w._opening_idx == 1

    w.game.reset(OPENING_FEN)
    w._await_new_game = True
    w._adopt_new_game(_Res([[p for p in row] for row in Board().grid]))
    assert w._opening_idx == 0, "新一局应重新从开局第一着开始"
    mv2, cn2 = w._pending_opening_move()
    assert move_to_iccs(mv2) == "g0e2" and cn2 == "相三进五"
    w.engine.stop() if w.engine else None


def test_unavailable_opening_move_skipped(qapp):
    """开局着法在当前局面不可行时跳过并回退引擎（不卡死）。"""
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("w")
    # 局面里红炮二（h2）位置被占：无法走 炮二平五 -> 应跳过
    # 仍在开局阶段（仅红方 7 路炮缺失），但该炮位已空 -> 中炮不可行
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C7/9/RNBAKABNR w - - 0 1")
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("中炮（当头炮）"))
    w._reset_opening()
    assert w._pending_opening_move() is None, "不可行的开局着法应被跳过"
    assert w._opening_idx == 1
    w.engine.stop() if w.engine else None


def test_no_opening_when_not_takeover(qapp):
    """未接管时选定的开局不生效（避免干扰手动/观战）。"""
    w = _win(qapp)
    w._takeover = False
    w.board.set_my_color("w")
    w.game.reset(OPENING_FEN)
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("中炮（当头炮）"))
    w._reset_opening()
    assert w._pending_opening_move() is None
    w.engine.stop() if w.engine else None


def test_opening_selection_persisted(qapp, tmp_path, monkeypatch):
    """开局选择写入配置（下次启动保留）。"""
    import gui.main_window as gmw
    monkeypatch.setattr(gmw, "CONFIG_PATH", str(tmp_path / "cfg.json"))
    w = _win(qapp)
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("起马局"))
    w._on_opening_changed(w.combo_opening.currentIndex())
    import json
    cfg = json.load(open(str(tmp_path / "cfg.json"), encoding="utf-8"))
    assert cfg.get("opening") == "起马局"
    w.engine.stop() if w.engine else None


def test_opening_not_consumed_on_opponent_turn(qapp):
    """回归（实际 bug）：引擎在对手回合也会给出 bestmove，
    此时绝不能消耗/跳过我方开局着法——否则双铁滑车只会走一边的車。

    日志实证：黑方走完车9进1后轮到红方，此时引擎出着被当成"我方回合"，
    第二着 a9a8 在红方回合判为不可行而被永久跳过。
    """
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color("b")
    w.game.reset(OPENING_FEN)
    w.game.board.side_to_move = "b"
    w.combo_opening.setCurrentIndex(w.combo_opening.findData("双铁滑车"))
    w._reset_opening()
    assert w._opening_queue == ["i9i8", "a9a8"]

    # 我方回合：走第一着
    mv1, cn1 = w._pending_opening_move()
    assert move_to_iccs(mv1) == "i9i8"
    w.game.play(mv1)
    w._confirm_opening_move(mv1)
    assert w._opening_idx == 1

    # 现在轮到红方（对手）：不应返回任何开局着法，也不应推进队列
    assert w.game.side_to_move == "w"
    assert w._pending_opening_move() is None, "对手回合不应取用我方开局着法"
    assert w._opening_idx == 1, "对手回合不应消耗开局队列"

    # 对手应着后回到我方回合：第二着仍可用
    w.game.play(iccs_to_move("h2e2"))
    assert w.game.side_to_move == "b"
    mv2, cn2 = w._pending_opening_move()
    assert mv2 is not None and move_to_iccs(mv2) == "a9a8", \
        f"第二着应仍可走，实际 {mv2}"
    assert cn2 == "车1进1"
    w.engine.stop() if w.engine else None


def test_frame_validation_ignores_side_to_move(qapp):
    """回归（实际 bug）：帧校验不能因"我方正被将军"而丢弃画面。

    帧本身不含行棋方信息，早先的实现默认按红方判，导致"轮到黑方且黑方
    被将军"（对手将军的正常局面）被误判为非法并持续丢帧，
    表现为"检测不到对方行棋"。
    """
    from recognition.yolo_board import grid_plausibility, board_plausibility
    from core.board import Board
    # 轮到黑方，黑将正被红车将军（正常局面：红方刚将军）
    b = Board("4k4/9/9/9/9/9/9/9/9/3KR4 b - - 0 1")
    grid = [row[:] for row in b.grid]
    ok, reason = grid_plausibility(grid)          # 帧校验（无行棋方信息）
    assert ok, f"该帧不应被丢弃：{reason}"

    # 用同一个矩阵但明确按红方走棋（此时确实非法）-> 交给 Board.validate_position 拦
    b2 = Board("4k4/9/9/9/9/9/9/9/9/3KR4 w - - 0 1")
    assert b2.validate_position() is not None, "引擎侧校验仍应拦住非法轮次"


def test_board_plausibility_accepts_checked_side(qapp):
    """探测阶段的棋盘合理性校验同样不应因"被将军"而拒绝。"""
    from recognition.yolo_board import board_plausibility

    class R:
        red_at_top = False
        n_pieces = 4
    r = R()
    r.board_box = (100, 100, 604, 700)          # 9:10 比例正常
    r.grid = [[None] * 9 for _ in range(10)]
    from core.board import Board
    b = Board("4k4/9/9/9/9/9/9/9/9/3KR4 b - - 0 1")
    r.grid = [row[:] for row in b.grid]
    r.n_pieces = 4
    ok, reason = board_plausibility(r)
    assert ok, f"被将军的合法局面不应被拒绝：{reason}"
