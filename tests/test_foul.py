# -*- coding: utf-8 -*-
"""禁止着法（长将/长捉）检测与代点规避测试。

背景：Pikafish 没有中国规则设置（UCI 选项里没有，实测确认），只把
局面重复当成和棋——劣势时它一直将军"求个和棋"，在平台上直接判负。
core/rules.py 负责在软件层检测我方犯规循环，gui._foul_avoid 负责换
着法脱出。
"""
from core.board import RED, Board, move_to_iccs

# ---- 长将循环局面：红車(5,4) 追着黑将 (0,3)<->(0,4) 将军，红帅在 (9,5)
LOOP_FEN = "3k5/9/9/9/9/4R4/9/9/9/5K3 w - - 0 1"
LOOP_MOVES = [((5, 4), (5, 3)),     # 红車 将
              ((0, 3), (0, 4)),     # 黑将 避
              ((5, 3), (5, 4)),     # 红車 将
              ((0, 4), (0, 3))]     # 黑将 避（回到起点，循环成立）

# ---- 允许着法循环：红車与黑馬各自来回挪动，双方都不将军也不捉
ALLOWED_FEN = "4k4/9/9/8n/9/4P4/9/9/R8/4K4 w - - 0 1"
ALLOWED_MOVES = [((8, 0), (7, 0)),
                 ((3, 8), (5, 7)),
                 ((7, 0), (8, 0)),
                 ((5, 7), (3, 8))]

# ---- 长捉循环：红車来回追吃无根黑馬（我方的每一着都新形成白吃威胁）
CHASE_FEN = "4k4/9/9/9/R3P4/7n1/9/9/9/4K4 w - - 0 1"
CHASE_MOVES = [((4, 0), (5, 0)),    # 红車 捉 馬(5,7)
               ((5, 7), (3, 8)),    # 黑馬 逃
               ((5, 0), (3, 0)),    # 红車 捉 馬(3,8)
               ((3, 8), (5, 7)),    # 黑馬 逃
               ((3, 0), (5, 0)),    # 红車 再捉 馬(5,7)
               ((5, 7), (3, 8)),    # 黑馬 再逃
               ((5, 0), (3, 0)),    # 红車 再捉 馬(3,8)
               ((3, 8), (5, 7))]    # 黑馬 回位（局面二次重复）


def test_position_keys_and_repeat():
    from core.rules import position_keys, repeating_moves
    b = Board(LOOP_FEN)
    for m in LOOP_MOVES:
        b.make_move(m)
    keys = position_keys(LOOP_FEN, LOOP_MOVES)
    assert len(keys) == 5
    assert keys[0] == keys[-1], "循环走完应回到起点局面"
    reps = repeating_moves(b, keys)
    assert ((5, 4), (5, 3)) in reps       # 继续将军 -> 重复
    assert ((5, 4), (6, 4)) not in reps   # 换个位置 -> 脱出


def test_is_check_move():
    from core.rules import is_check_move
    b = Board(LOOP_FEN)
    assert is_check_move(b, ((5, 4), (5, 3))) is True
    assert is_check_move(b, ((5, 4), (6, 4))) is False


def test_is_chase_move_undefended_is_chase():
    from core.rules import is_chase_move
    b = Board("4k4/9/9/9/R8/4n4/9/9/9/4K4 w - - 0 1")
    # 红車 (4,0)->(5,0) 新形成对无根黑馬的白吃威胁 -> 捉
    assert is_chase_move(b, ((4, 0), (5, 0))) is True


def test_is_chase_move_defended_is_exchange_not_chase():
    from core.rules import is_chase_move
    b = Board("4k4/9/9/9/R8/4nr3/9/9/9/4K4 w - - 0 1")
    # 黑馬有黑車保护：吃馬会被吃回（400-900<0），只能算兑，不算捉
    assert is_chase_move(b, ((4, 0), (5, 0))) is False


def test_long_check_loop_detected():
    from core.rules import detect_our_foul
    b = Board(LOOP_FEN)
    for m in LOOP_MOVES:
        b.make_move(m)
    assert detect_our_foul(b, LOOP_FEN, LOOP_MOVES, "w") == "长将"
    # 轮次不符（回放与当前行棋方不一致）时不下判定
    assert detect_our_foul(b, LOOP_FEN, LOOP_MOVES, "b") is None
    # 只有半圈（未形成重复）时不判定
    assert detect_our_foul(Board(LOOP_FEN), LOOP_FEN, LOOP_MOVES[:2], "w") is None


def test_long_chase_loop_detected():
    from core.rules import detect_our_foul
    b = Board(CHASE_FEN)
    for m in CHASE_MOVES:
        b.make_move(m)
    assert detect_our_foul(b, CHASE_FEN, CHASE_MOVES, "w") == "长捉"


def test_allowed_repetition_not_flagged():
    from core.rules import detect_our_foul
    b = Board(ALLOWED_FEN)
    for m in ALLOWED_MOVES:
        b.make_move(m)
    # 循环里我方每着都是闲着（长兑/长跟类）-> 不算犯规
    assert detect_our_foul(b, ALLOWED_FEN, ALLOWED_MOVES, "w") is None


def test_foul_avoid_switches_to_fresh_pv_move(qapp, monkeypatch):
    """代点时检测到长将循环 -> 换成 MultiPV 里不会重复的候选着法。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    w._takeover = True
    w._search_fen = None
    w._last_sync_res = None
    w._opening_queue = []
    w._opening_idx = 0
    w.game.reset(LOOP_FEN)
    for m in LOOP_MOVES:
        assert w.game.play(m)
    w.board.set_game(w.game)
    w.board.set_my_color(RED)
    loop_iccs = move_to_iccs(((5, 4), (5, 3)))
    fresh_iccs = move_to_iccs(((5, 4), (6, 4)))
    w.panel.pv_lines = [
        {"multipv": 1, "pv": [loop_iccs, "h9g7"], "score_str": "+0.00"},
        {"multipv": 2, "pv": [fresh_iccs], "score_str": "-1.20"},
    ]
    calls = []
    monkeypatch.setattr(w, "_click_worker", lambda m: calls.append(m))
    w._on_engine_bestmove(loop_iccs)
    assert calls, "应照常进入代点流程"
    assert move_to_iccs(calls[0]) == fresh_iccs, "应换成不重复的候选着法"
    w.engine.stop() if w.engine else None


def test_foul_avoid_keeps_move_for_allowed_repetition(qapp):
    """允许着法（长兑/长跟）的重复不换着法，照常走引擎的选择。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    w.game.reset(ALLOWED_FEN)
    for m in ALLOWED_MOVES:
        assert w.game.play(m)
    res = w._foul_avoid(((8, 0), (7, 0)), "")
    assert res is not None and res[0] == ((8, 0), (7, 0))
    w.engine.stop() if w.engine else None


def test_go_builds_searchmoves_last():
    """searchmoves 必须放在 go 命令最后（实测放前面会吞掉 depth）。"""
    from engine.uci import UciEngine
    e = UciEngine.__new__(UciEngine)
    sent = []
    e._send = sent.append
    e.go(depth=24, movetime_ms=3000, searchmoves=["b0c2", "h2e2"])
    assert sent == ["go depth 24 movetime 3000 searchmoves b0c2 h2e2"]
