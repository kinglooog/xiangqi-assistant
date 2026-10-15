# -*- coding: utf-8 -*-
"""棋盘规则与记法测试。所有测试局面均手工推演验证过。"""
from core.board import (
    INITIAL_FEN, Board, iccs_to_move, iccs_to_sq, move_to_iccs, move_to_chinese,
)
from core.game import Game


def dests(board, frm_iccs):
    """从 frm 出发的所有合法走法目标（ICCS）。"""
    frm = iccs_to_sq(frm_iccs)
    return sorted(move_to_iccs(m)[2:] for m in board.legal_moves()
                  if m[0] == frm)


def test_fen_roundtrip():
    b = Board()
    assert b.to_fen().split()[0] == INITIAL_FEN.split()[0]
    assert b.side_to_move == "w"


def test_initial_setup():
    b = Board()
    pieces = [p for row in b.grid for p in row if p]
    assert len(pieces) == 32
    assert sum(1 for p in pieces if p.isupper()) == 16


def test_initial_move_count():
    """开局红方合法着法 44 种（经典常数，与 Pikafish 等引擎一致）。

    逐子校验：车 2+2、马 2+2、相 2+2、仕 1+1、帅 1、
    炮（上5含隔架吃马+下1+左1+右5）×2=24、兵 5，合计 44。
    """
    b = Board()
    assert len(b.legal_moves()) == 44
    # 炮八进七：隔自家炮架吃黑马，开局即存在
    assert iccs_to_move("b2b9") in b.legal_moves()


def test_opening_moves():
    b = Board()
    assert iccs_to_move("h2e2") in b.legal_moves()   # 炮二平五
    assert iccs_to_move("h0g2") in b.legal_moves()   # 马二进三
    assert iccs_to_move("b0c2") in b.legal_moves()   # 马八进七


def test_horse_moves_and_leg():
    """马走日、蹩马腿。黑马 e9、黑将 f9，红帅 d0。"""
    b = Board("4nk3/9/9/9/9/9/9/9/9/3K5 b - - 0 1")
    assert dests(b, "e9") == ["c8", "d7", "f7"]
    # 腿位 e8 放红兵 -> 上跳两只被蹩；g8 被自家黑将占 -> 只剩 c8
    b2 = Board("4nk3/4P4/9/9/9/9/9/9/9/3K5 b - - 0 1")
    assert dests(b2, "e9") == ["c8"]


def test_elephant_river_and_eye():
    """象不过河、塞象眼。"""
    b = Board("4bk3/9/9/9/9/9/9/9/9/3K5 b - - 0 1")
    assert dests(b, "e9") == ["c7", "g7"]
    # 象眼 d8/f8 各放黑卒 -> 象无路可走
    b2 = Board("4bk3/3p1p3/9/9/9/9/9/9/9/3K5 b - - 0 1")
    assert dests(b2, "e9") == []


def test_advisor_palace():
    b = Board("4ak3/9/9/9/9/9/9/9/9/3K5 b - - 0 1")
    assert dests(b, "e9") == ["d8", "f8"]


def test_pawn_rules():
    """黑卒 d5 未过河只能进；黑卒 d4 已过河可横移。"""
    b = Board("9/9/9/9/3p5/9/9/9/4k4/3K5 b - - 0 1")
    assert dests(b, "d5") == ["d4"]
    b2 = Board("9/9/9/9/9/3p5/9/9/4k4/3K5 b - - 0 1")
    assert dests(b2, "d4") == ["c4", "d3", "e4"]


def test_cannon_screen():
    """炮吃子必须有炮架。"""
    # 红炮 e0、红兵 e1 作炮架、黑马 e9
    b = Board("4n4/9/9/9/9/9/9/9/4P4/3KC4 w - - 0 1")
    assert "e0e9" in [move_to_iccs(m) for m in b.legal_moves()]
    # 无炮架时不能吃马
    b2 = Board("4n4/9/9/9/9/9/9/9/9/3KC4 w - - 0 1")
    assert "e0e9" not in [move_to_iccs(m) for m in b2.legal_moves()]


def test_flying_general_block():
    """将帅对脸：红车离开 d 线会露将，非法；沿 d 线走合法。"""
    b = Board("3k5/9/9/9/9/9/9/9/3R5/3K5 w - - 0 1")
    rook_moves = [m for m in b.legal_moves() if move_to_iccs(m)[:2] == "d1"]
    # 除吃将 d1d9 外，车不能横移离开 d 线
    assert all(m[1][1] == 3 for m in rook_moves if move_to_iccs(m) != "d1d9")
    assert "d1d8" in [move_to_iccs(m) for m in rook_moves]


def test_flying_general_capture():
    b = Board("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    assert iccs_to_move("e0e9") in b.legal_moves()


def test_in_check():
    assert Board("4k4/9/9/9/4R4/9/9/9/9/4K4 b - - 0 1").in_check("b")
    assert not Board("4k4/9/9/9/9/9/9/9/9/3K5 b - - 0 1").in_check("b")
    # 红马 d7 跳吃 e9（马腿 d8 无阻塞）
    assert Board("4k4/9/3N5/9/9/9/9/9/9/3K5 b - - 0 1").in_check("b")


def test_must_resolve_check():
    """被将军时所有合法走法都必须解将。"""
    b = Board("4k4/9/9/9/4R4/9/9/9/9/4K4 b - - 0 1")
    assert b.legal_moves()  # 确实有解
    for m in b.legal_moves():
        probe = Board(b.to_fen())
        probe.make_move(m)
        assert not probe.in_check("b")


def test_checkmate():
    """双车+马杀：黑将 d9 被红车 a9 沿横线将军，所有逃格均被控。"""
    b = Board("R2k5/9/9/2N6/9/9/9/9/5K3/4R4 b - - 0 1")
    assert b.in_check("b")
    assert b.legal_moves() == []
    over, result = b.is_game_over()
    assert over and "红方胜" in result and "将死" in result


def test_stalemate_is_loss():
    """困毙：黑将 d9 未被将军但无子可走，判负。"""
    b = Board("3k5/9/1N7/2N6/9/9/9/9/5K3/4R4 b - - 0 1")
    assert not b.in_check("b")
    assert b.legal_moves() == []
    over, result = b.is_game_over()
    assert over and "红方胜" in result and "困毙" in result


def test_game_undo_and_history():
    g = Game()
    assert g.play(iccs_to_move("h2e2"))
    assert g.play(iccs_to_move("h9g7"))
    assert len(g.history) == 2
    assert g.move_list_chinese() == ["炮二平五", "马8进7"]
    assert g.move_list_iccs() == ["h2e2", "h9g7"]
    g.undo()
    assert len(g.history) == 1
    assert g.board.side_to_move == "b"
    g.undo()
    g.undo()
    assert len(g.history) == 0
    assert g.board.to_fen().split()[0] == INITIAL_FEN.split()[0]


def test_illegal_move_rejected():
    g = Game()
    assert not g.play(iccs_to_move("h0h9"))  # 车不能跳过马? 直线被马挡
    assert len(g.history) == 0


def test_chinese_notation():
    b = Board()
    assert move_to_chinese(b, iccs_to_move("h2e2")) == "炮二平五"
    assert move_to_chinese(b, iccs_to_move("h0g2")) == "马二进三"
    assert move_to_chinese(b, iccs_to_move("b0c2")) == "马八进七"
    assert move_to_chinese(b, iccs_to_move("c3c4")) == "兵七进一"
    assert move_to_chinese(b, iccs_to_move("i0i1")) == "车一进一"


def test_chinese_notation_front_back():
    """同线两车用前/后前缀。红方纵线: a=九...f=四。"""
    b = Board("9/9/9/9/9/9/9/9/R8/R3K4 w - - 0 1")
    assert move_to_chinese(b, ((8, 0), (8, 5))) == "前车平四"
    assert move_to_chinese(b, ((9, 0), (9, 5))) == "后车平四"
