# -*- coding: utf-8 -*-
"""后台预计算（ponder）测试：挂起、命中复用、未命中丢弃、各守卫。

不启动真实引擎：_FakeEngine 记录调用序列。协议本身已在真机实测：
`go ponder` 期间 bestmove 被引擎压住；ponderhit 后 0.02s 出着；
未命中时 stop 也能 0.02s 取回（丢弃即可）。
"""
from core.board import RED, iccs_to_move

OPEN_BLACK_TURN = ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                   "1C5C1/9/RNBAKABNR b - - 0 1")
OPEN_RED_TURN = ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self._running = True

    def is_running(self):
        return self._running

    def stop_search(self):
        self.calls.append(("stop",))

    def position(self, fen, moves=None):
        self.calls.append(("position", fen, tuple(moves or [])))

    def go(self, **kw):
        self.calls.append(("go", kw))

    def ponderhit(self):
        self.calls.append(("ponderhit",))

    def stop(self):
        self._running = False


def _win_with_fake(qapp):
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    w._takeover = True
    try:
        w.engine.stop()
    except Exception:
        pass
    w.engine = _FakeEngine()
    return w


def test_arm_ponder_sends_pondered_search(qapp):
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "h9g7"
    w._arm_ponder()
    assert w._pondering is True
    assert w._ponder_expected == "h9g7"
    pos = [c for c in w.engine.calls if c[0] == "position"][-1]
    assert pos[2] == ("h9g7",), "position 最后一步应是预计的对手着法"
    go_kw = [c[1] for c in w.engine.calls if c[0] == "go"][-1]
    assert go_kw.get("ponder") is True, "应发 go ponder"
    assert not go_kw.get("movetime_ms"), "预计算不设时间上限（对手想多久算多久）"
    w.engine.stop()


def test_arm_ponder_skips_when_not_opponent_turn(qapp):
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_RED_TURN)          # 轮到我方
    w._expected_reply = "h2e2"
    w._arm_ponder()
    assert w._pondering is False
    assert not any(c[0] == "go" for c in w.engine.calls)
    w.engine.stop()


def test_arm_ponder_skips_when_prediction_stale(qapp):
    """预测着法在当前局面不合法（局面变了）时不挂预计算。"""
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "a0a1"           # 红方着法，黑方回合走不了
    w._arm_ponder()
    assert w._pondering is False
    w.engine.stop()


def test_resolve_ponder_hit_reuses_deep_search(qapp):
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "h9g7"
    w._arm_ponder()
    assert w.game.play(iccs_to_move("h9g7"))     # 对手真走了预计着
    w._resolve_ponder(iccs_to_move("h9g7"))
    assert w._ponder_hit_pending is True
    assert w._pondering is False
    assert ("ponderhit",) in w.engine.calls, "命中应发 ponderhit"
    assert w._search_fen == w.game.board.to_fen()
    assert w._search_in_flight is True
    w._finish_ponder_hit()                        # 宽限到期：冻结深算取着
    assert w.engine.calls[-1] == ("stop",)
    w.engine.stop()


def test_resolve_ponder_miss_discards_result(qapp):
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "h9g7"
    w._arm_ponder()
    assert w.game.play(iccs_to_move("b9c7"))      # 对手走了别的着法
    w._resolve_ponder(iccs_to_move("b9c7"))
    assert w._pondering is False and w._ponder_hit_pending is False
    assert ("stop",) in w.engine.calls
    assert w._ponder_discard is True, "应丢弃预搜索的残留 bestmove"
    w._on_engine_bestmove("h2e2")                 # 残留 bestmove：被吞掉
    assert w._ponder_discard is False
    w.engine.stop()


def test_bestmove_swallowed_while_pondering(qapp, monkeypatch):
    w = _win_with_fake(qapp)
    calls = []
    monkeypatch.setattr(w, "_click_worker", lambda m: calls.append(m))
    w._pondering = True
    w._on_engine_bestmove("h2e2")
    assert not calls, "预计算期间不应应用任何 bestmove"
    w.engine.stop()


def test_trigger_analysis_skipped_during_ponder_flow(qapp):
    w = _win_with_fake(qapp)
    w.chk_analyze.setChecked(True)
    w._ponder_hit_pending = True
    w._trigger_analysis()
    assert not any(c[0] == "go" for c in w.engine.calls), \
        "命中待取着法时不应重发 go（打断深算）"
    w.engine.stop()


def test_drop_ponder_clears_state(qapp):
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "h9g7"
    w._arm_ponder()
    assert w._pondering is True
    w._drop_ponder()
    assert w._pondering is False and w._ponder_hit_pending is False
    assert w._ponder_expected == "" and w._expected_reply is None
    assert w._ponder_discard is True
    w.engine.stop()


def test_go_ponder_command_and_ponderhit():
    from engine.uci import UciEngine
    e = UciEngine.__new__(UciEngine)
    sent = []
    e._send = sent.append
    e.go(depth=24, ponder=True, searchmoves=["a0a1"])
    assert sent == ["go ponder depth 24 searchmoves a0a1"]
    e.ponderhit()
    assert sent[-1] == "ponderhit"


def test_ponder_info_stm_keeps_red_perspective(qapp):
    """预计算期间 info 的视角标记 must 是"被搜索局面的行棋方"=我方。

    实际 bug：轮转后红方胜率 90% 跳到黑方 90%——预计算根局面是我方
    该走（引擎分数以我方为正），却被贴上旧视角（对手），胜率条翻面。
    """
    w = _win_with_fake(qapp)
    w.board.set_my_color(RED)
    w.panel.set_my_color(RED)
    w.game.reset(OPEN_BLACK_TURN)
    w._expected_reply = "h9g7"
    # 复现真实时序：此前刚跑过"对手局面"的分析（视角=黑），并且
    # _search_fen 与当前局面一致（info 不会被过期守卫拦掉）
    w._search_stm = "b"
    w._search_fen = w.game.board.to_fen()
    w._arm_ponder()
    assert w._search_stm == RED, "预计算根局面是我方该走，视角应更新为我方"
    # 预计算 info 以红方（我方）为正分：胜率条必须仍显示红方占优
    w._engine_info_cb({"score": 300, "score_str": "+3.00", "multipv": 1,
                       "depth": 20, "pv": ["g3g4"], "score_kind": "cp"})
    assert w.panel.pv_lines, "预计算 info 应显示到面板"
    top = w.panel.pv_lines[0]
    assert top.get("stm") == RED, "info 应贴我方视角"
    assert w.panel.eval_bar.value > 50, \
        f"红方占优时胜率条应仍偏红，实际 {w.panel.eval_bar.value}"
    assert "红方胜率" in w.panel.winrate_label.text()
    w.engine.stop()


def test_ponder_pv_translated_from_searched_position(qapp):
    """预计算 PV 的中文翻译必须基于被搜索局面（预计对手走后）。

    实际 bug：用当前局面回放时第一步就不合法，面板回退显示 ICCS
    编码串（"黑方的候选着法变成 b9c7 这种编码"）。
    """
    from core.board import BLACK
    w = _win_with_fake(qapp)
    w.board.set_my_color(BLACK)
    w.panel.set_my_color(BLACK)
    w.game.reset(OPEN_RED_TURN)          # 对手（红）该走：我方执黑
    w.board.set_game(w.game)
    w.panel.set_stm("w")                 # 面板视角=当前轮次（红）
    w._expected_reply = "h2e2"           # 预计红方走炮二平五
    w._search_stm = "w"
    w._search_fen = w.game.board.to_fen()
    w._arm_ponder()
    assert w._search_root_fen is not None, "应记录被搜索局面"
    # 预计算局面（红方已走 h2e2）里的黑方候选着法
    w._engine_info_cb({"score": -80, "score_str": "-0.80", "multipv": 1,
                       "depth": 18, "pv": ["b9c7", "h2e2"], "score_kind": "cp"})
    top = w.panel.pv_lines[0]
    assert top.get("pv_cn"), "PV 应翻译为中文记法，而不是编码串"
    assert not str(top["pv_cn"][0]).startswith("b9"), \
        f"不应显示 ICCS 编码串，实际 {top['pv_cn']}"
    text = w.panel.pv_list.item(0).text()
    assert "[黑方]" in text, f"候选着法标签应为被搜索方（黑），实际：{text}"
    w.engine.stop()
