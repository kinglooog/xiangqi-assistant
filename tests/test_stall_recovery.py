# -*- coding: utf-8 -*-
"""卡死自愈测试：重同步的轮次推断（合法着法优先）+ 看门狗 + 代点拦截限时。

背景：实际对局中出现过"该我方走却判成对手回合"导致永久等待，
以及"点击未生效但同一着法被永久拦截"两种停滞。
"""
import time

import pytest

from PyQt6.QtWidgets import QApplication

from core.board import RED, BLACK, Board, iccs_to_move


def _win(qapp):
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345          # 避免触发真实抓屏
    return w


def _res(grid, red_at_top=False):
    class R:
        pass
    r = R()
    r.grid = grid
    r.red_at_top = red_at_top
    r.n_pieces = sum(1 for row in grid for p in row if p)
    r.corrections = []
    r.source = "yolo"
    r.board_box = (100, 100, 900, 1000)
    r.detections = []
    return r


def test_resync_uses_legal_move_to_decide_turn(qapp):
    """变化能被我方着法解释时，轮次应判给对手（走子方=我方）。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")   # 轮到黑方（我方）
    # 观察帧：黑方走了一步马8进7
    b = Board(w.game.board.to_fen())
    b.make_move(iccs_to_move("h9g7"))
    w._last_auto_resync = 0.0
    w._auto_resync_to_window(_res([row[:] for row in b.grid]), None)
    assert w.game.side_to_move == RED, "黑方刚走完，应轮到红方"
    w.engine.stop() if w.engine else None


def test_resync_infers_opponent_move_when_waiting(qapp):
    """等在等对手时对手走了一步 -> 重同步后应轮到"我方"（不卡死）。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)                 # 我方执黑
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")  # 轮到红方（对手）
    b = Board(w.game.board.to_fen())
    b.make_move(iccs_to_move("h2e2"))            # 对手走炮二平五
    w._last_auto_resync = 0.0
    w._auto_resync_to_window(_res([row[:] for row in b.grid]), None)
    assert w.game.side_to_move == BLACK, "对手刚走完，应轮到我方"
    w.engine.stop() if w.engine else None


def test_resync_bias_when_unexplainable_while_waiting(qapp):
    """变化无法解释且此前在等对手 -> 保守判给"我方"，宁可无效点击也不卡死。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")   # 等对手
    garbled = [[None] * 9 for _ in range(10)]
    garbled[0][4] = "k"
    garbled[9][3] = "K"
    garbled[5][5] = "R"          # 无法用任何一步解释的乱局
    garbled[3][2] = "p"
    w._last_auto_resync = 0.0
    w._auto_resync_to_window(_res(garbled), None)
    assert w.game.side_to_move == BLACK, "无从判断时应偏向我方，避免无限等待"
    w.engine.stop() if w.engine else None


def test_resync_clears_click_guard(qapp):
    """重同步必须清空代点拦截标记（否则同一着法永久被拦 -> 卡死）。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    w._clicked_key = ("old-fen", "h9g7")
    w._auto_click_pending = {"move": iccs_to_move("h9g7"), "at": 0}
    w._pending_reply = iccs_to_move("h2e2")
    b = Board(w.game.board.to_fen())
    b.make_move(iccs_to_move("h9g7"))
    w._last_auto_resync = 0.0
    w._auto_resync_to_window(_res([row[:] for row in b.grid]), None)
    assert w._clicked_key is None
    assert w._auto_click_pending is None
    assert w._pending_reply is None
    w.engine.stop() if w.engine else None


def test_click_guard_is_time_limited(qapp, monkeypatch):
    """同一着法只拦 6 秒：点击静默失败后能重试，而不是永久卡死。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    mv = iccs_to_move("h9g7")
    calls = []
    monkeypatch.setattr(w, "_click_worker", lambda m: calls.append(m))
    w._auto_click_move(mv, "马8进7", "")
    assert len(calls) == 1
    w._auto_click_move(mv, "马8进7", "")
    assert len(calls) == 1, "6 秒内同一着法应被拦截"
    w._clicked_at = time.time() - 7          # 模拟已过 6 秒
    w._auto_click_move(mv, "马8进7", "")
    assert len(calls) == 2, "超时后应允许重试"
    w.engine.stop() if w.engine else None


def test_stall_watchdog_retriggers(qapp, monkeypatch):
    """我方回合长时间无动作 -> 看门狗重新触发分析。"""
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")   # 我方回合
    triggered = []
    monkeypatch.setattr(w, "_trigger_analysis", lambda: triggered.append(1))
    # 清掉构造期间可能残留的分析状态，避免干扰计时判定
    w._search_in_flight = False
    w._clicking = False
    w._auto_click_pending = None
    w._last_our_turn_ts = 0.0
    w._stall_watchdog()
    assert triggered == [], "首次只计时，不触发"
    w._last_our_turn_ts = time.time() - 7
    w._stall_watchdog()
    assert triggered == [1], "超时应重新触发分析"
    w.engine.stop() if w.engine else None


def test_stall_watchdog_silent_when_not_our_turn(qapp, monkeypatch):
    w = _win(qapp)
    w._takeover = True
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")   # 对手回合
    triggered = []
    monkeypatch.setattr(w, "_trigger_analysis", lambda: triggered.append(1))
    w._search_in_flight = False
    w._auto_click_pending = None
    w._last_our_turn_ts = time.time() - 20
    w._stall_watchdog()
    assert triggered == []
    w.engine.stop() if w.engine else None


def test_detection_miss_not_adopted_as_real_capture(qapp, monkeypatch):
    """回归（实际 bug）：单格差异持续存在，但低阈值复核发现棋子仍在
    -> 判定为检测漏检，保持本地棋盘，绝不把漏检当成真丢子。

    实际发生的链路：红車在 a0 被漏检 -> 单格差异 4 秒未补全 ->
    旧逻辑"以 JJ 为准"采纳了缺子的画面 -> 本地少一个車 ->
    引擎评估跳到 +8.55（误以为对手少子）-> 下出错误着法。
    """
    import time as _t
    from gui.main_window import MainWindow
    from core.board import Board, move_to_iccs
    from recognition.yolo_board import grid_plausibility

    w = MainWindow()
    w._sync_hwnd = 12345
    w.board.set_my_color(BLACK)
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
    w.game.reset(fen)
    local_grid = [row[:] for row in w.game.board.grid]

    # 观察帧：a0 的红車"消失"（模拟漏检）
    observed = [row[:] for row in local_grid]
    observed[9][0] = None

    class _Res:
        source = "yolo"
        n_pieces = 15          # 少于本地，但不是 0（帧未被丢弃）
        error = None
        red_at_top = False
        board_box = (100, 100, 605, 700)
        corrections = []
        detections = []
    res = _Res()
    res.grid = observed

    # 低阈值复核结果：棋子其实还在（找回漏检）
    class _Low:
        source = "yolo"
        error = None
        red_at_top = False
        board_box = (100, 100, 605, 700)
        corrections = []
        detections = []
        n_pieces = 16
    low = _Low()
    low.grid = [row[:] for row in local_grid]     # 完整（含 a0 的車）
    monkeypatch.setattr(w, "_recognize_low_conf", lambda img: low)

    adopted = []
    monkeypatch.setattr(w, "_auto_resync_to_window",
                        lambda r, m: adopted.append((r, m)))

    # 构造"单格差异已持续 5 秒"的状态并触发复核分支
    w._partial_key = (9, 0, "R", None)
    w._partial_since = _t.time() - 5
    w._handle_sync_events([{"type": "changed", "grid": observed}], res)

    assert adopted == [], "漏检被误当成真丢子并触发了重新同步"
    assert w.game.board.grid[9][0] == "R", "本地棋盘不应丢掉红車"
    w.engine.stop() if w.engine else None


def test_real_removal_still_resyncs(qapp, monkeypatch):
    """低阈值复核也确认棋子不在时 -> 仍按 JJ 局面重新同步（不误判为漏检）。"""
    import time as _t
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    observed = [row[:] for row in w.game.board.grid]
    observed[9][0] = None

    class _Res:
        source = "yolo"
        n_pieces = 15
        error = None
        red_at_top = False
        board_box = (100, 100, 605, 700)
        corrections = []
        detections = []
    res = _Res()
    res.grid = observed

    class _Low:
        source = "yolo"
        error = None
        red_at_top = False
        board_box = (100, 100, 605, 700)
        corrections = []
        detections = []
        n_pieces = 15
    low = _Low()
    low.grid = [row[:] for row in observed]        # 复核也找不到 -> 真的没了
    monkeypatch.setattr(w, "_recognize_low_conf", lambda img: low)

    adopted = []
    monkeypatch.setattr(w, "_auto_resync_to_window", lambda r, m: adopted.append(m))
    w._partial_key = (9, 0, "R", None)
    w._partial_since = _t.time() - 5
    w._handle_sync_events([{"type": "changed", "grid": observed}], res)
    assert len(adopted) == 1, "确认真实丢子后应重新同步"
    w.engine.stop() if w.engine else None


def test_engine_death_is_recovered(qapp, monkeypatch):
    """回归：引擎进程退出后，必须自动重启并重新分析，不能静默卡死。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w._takeover = True
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    # 模拟引擎已退出
    class DeadEngine:
        def is_running(self):
            return False

        def stop(self):
            pass
    w.engine = DeadEngine()
    restarted, triggered = [], []
    monkeypatch.setattr(w, "_start_engine", lambda: restarted.append(1))
    monkeypatch.setattr(w, "_trigger_analysis", lambda: triggered.append(1))
    w._search_in_flight = True              # 之前有搜索在飞（永久收不到 bestmove）
    w._search_started_at = time.time() - 100
    w._stall_watchdog()
    assert restarted == [1], "引擎退出未被检测到"
    assert triggered == [1], "重启后应重新触发分析"
    assert w._search_in_flight is False, "应清除卡死的搜索状态"


def test_hung_search_is_recovered(qapp, monkeypatch):
    """回归：搜索超过 15 秒没有 bestmove（引擎挂死）也要重启救援。

    旧逻辑对"搜索在飞"一律只刷新计时，导致永远卡在等待分析。
    """
    from gui.main_window import MainWindow
    w = MainWindow()
    w._takeover = True
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    alive = type("E", (), {"is_running": lambda self: True,
                           "stop": lambda self: None})()
    w.engine = alive
    restarted, triggered = [], []
    monkeypatch.setattr(w, "_start_engine", lambda: restarted.append(1))
    monkeypatch.setattr(w, "_trigger_analysis", lambda: triggered.append(1))
    w._search_in_flight = True
    w._search_started_at = time.time() - 20      # 已超 15 秒无响应
    w._stall_watchdog()
    assert restarted == [1], "挂死的搜索未被救援"
    assert triggered == [1]
    # 未超时则不应重启
    w2 = MainWindow()
    w2.engine = alive
    w2._search_in_flight = True
    w2._search_started_at = time.time() - 3
    r2 = []
    monkeypatch.setattr(w2, "_start_engine", lambda: r2.append(1))
    w2._stall_watchdog()
    assert r2 == [], "正常运行中的搜索不应被重启"
    w2.engine.stop() if hasattr(w2.engine, "stop") else None


def test_validate_rejects_pawn_on_back_rank():
    """兵/卒不能停在己方底线（引擎会拒收这类局面并直接退出）。"""
    from core.board import Board
    b = Board("4k4/9/9/9/9/9/9/9/9/P2K5 w - - 0 1")
    assert "底线" in b.validate_position()
    b2 = Board("p3k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1")
    assert "底线" in b2.validate_position()
    assert Board().validate_position() is None


def test_validate_rejects_non_moving_side_in_check():
    """非行棋方被将军是非法局面（Pikafish 会拒收并退出，实测确认）。"""
    from core.board import Board
    # 轮到红走，但黑将正被红车将军 -> 非法
    b = Board("4k4/9/9/9/9/9/9/9/9/3KR4 w - - 0 1")
    assert "非行棋方" in b.validate_position()
    # 同一局面轮到黑走（被将的一方走）-> 合法
    b2 = Board("4k4/9/9/9/9/9/9/9/9/3KR4 b - - 0 1")
    assert b2.validate_position() is None


def test_turn_auto_corrected_by_check(qapp):
    """轮次猜错导致"非行棋方被将军"时，应自动翻转轮次（而非拒绝）。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345
    # 轮到红走但黑将被将军（轮次有误）-> 应自动改成黑方走
    w.game.reset("4k4/9/9/9/9/9/9/9/9/3KR4 w - - 0 1")
    assert w.game.board.validate_position() is not None
    fixed = w._fix_turn_by_check()
    assert fixed is True
    assert w.game.side_to_move == "b", "被将军的一方（黑）应成为走棋方"
    assert w.game.board.validate_position() is None, "纠正后应成为合法局面"
    # 合法局面不应被改动
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
    assert w._fix_turn_by_check() is False
    assert w.game.side_to_move == "w"
    w.engine.stop() if w.engine else None


def test_engine_kill_loop_is_broken(qapp, monkeypatch):
    """同一局面反复杀死引擎时，不应无限"重启->再死"，而是停手并提示。"""
    import time as _t
    from gui.main_window import MainWindow
    w = MainWindow()
    w._takeover = True
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    dead = type("E", (), {"is_running": lambda self: False,
                          "stop": lambda self: None})()
    w.engine = dead
    triggered = []
    monkeypatch.setattr(w, "_start_engine", lambda: None)
    monkeypatch.setattr(w, "_trigger_analysis", lambda: triggered.append(1))

    # 第一次：正常重启并重新分析，并把该局面记为"杀死引擎的局面"
    w._stall_watchdog()
    assert triggered == [1], "首次应重启并重新分析"
    assert w._engine_kill_fen == w.game.board.to_fen()

    # 第二次（同一局面）：不再重复送分析，避免死循环
    w.engine = dead
    w._search_in_flight = True
    w._stall_watchdog()
    assert triggered == [1], "同一局面不应反复重复送分析（会死循环）"


def test_flicker_recovered_by_low_threshold(qapp, monkeypatch):
    """回归：检测抖动导致棋子"凭空消失"时，低阈值复核应找回并解释变化。

    日志实证：同一格 (0,0,'r',None) 下一帧又变回 (0,0,None,'r')，
    说明黑車被间歇性漏检——多格路径原先直接采纳抖动画面，导致丢子。
    """
    from gui.main_window import MainWindow
    from core.board import Board
    w = MainWindow()
    w._sync_hwnd = 12345
    w.board.set_my_color(RED_ if False else "w")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
    w.game.reset(fen)
    local = [row[:] for row in w.game.board.grid]
    # 观察帧：黑方两車被漏检（抖动），外加一步真实走子 马8进7
    observed = [row[:] for row in local]
    observed[0][0] = None          # 黑車 a9 漏检
    observed[0][8] = None          # 黑車 i9 漏检
    observed[0][1] = None          # 马8进7 的起点
    observed[2][2] = "n"           # 马8进7 的终点

    class R:
        source = "yolo"
        n_pieces = 15
        error = None
        red_at_top = False
        board_box = (100, 100, 605, 700)
        corrections = []
        detections = []
    res = R()
    res.grid = observed

    # 低阈值复核：找回两車，且只有 马8进7 一处变化 -> 可被合法走法解释
    class L(R):
        pass
    low = L()
    low.grid = [row[:] for row in local]
    low.grid[0][1] = None
    low.grid[2][2] = "n"
    low.n_pieces = 32
    monkeypatch.setattr(w, "_recognize_low_conf", lambda img: low)

    adopted = []
    monkeypatch.setattr(w, "_auto_resync_to_window",
                        lambda r, m: adopted.append(m))
    w._handle_sync_events([{"type": "changed", "grid": observed}], res, None)

    assert adopted == [], "抖动不应触发整体重新同步（会丢子）"
    # 黑方纵线号自黑方右侧起数：col 1 为"马2"，走到 col 2 即"马2进3"
    assert w.game.move_list_chinese()[-1:] == ["马2进3"], \
        f"应按复核结果同步走子，实际 {w.game.move_list_chinese()}"
    assert w.game.board.grid[0][0] == "r" and w.game.board.grid[0][8] == "r", \
        "两車不应丢失"
    w.engine.stop() if w.engine else None


def test_click_cancelled_when_window_ahead_of_board(qapp, monkeypatch):
    """回归（漏招根因）：对手已走子但尚未同步时，不能按旧局面点击。

    否则会"按旧坐标点新局面"——起点棋子已不在原位，可能触发完全不同的
    走法（真实漏招）。正确行为是放弃本次代点、先同步对手着法。
    """
    from gui.main_window import MainWindow
    from core.board import Board, iccs_to_move, move_to_iccs
    w = MainWindow()
    w._sync_hwnd = 12345
    w._takeover = True
    w.board.set_my_color("w")
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")

    # 最新观察帧：对手（黑）已走马8进7，但本地棋盘还没同步
    ahead = [row[:] for row in w.game.board.grid]
    ahead[0][1] = None
    ahead[2][2] = "n"

    class R:
        pass
    r = R()
    r.grid = ahead
    w._last_sync_res = r

    calls = []
    monkeypatch.setattr(w, "_click_window_point",
                        lambda x, y, what: calls.append(what) or True)
    w._auto_click_move(iccs_to_move("h2e2"), "炮二平五", "")
    assert calls == [], f"窗口局面已变化时不应点击: {calls}"
    assert w._auto_click_pending is None
    w.engine.stop() if w.engine else None


def test_click_allowed_when_window_matches(qapp, monkeypatch):
    """窗口与本地棋盘一致时应正常代点（不误拦）。"""
    from gui.main_window import MainWindow
    from core.board import iccs_to_move
    w = MainWindow()
    w._sync_hwnd = 12345
    w._takeover = True
    w.board.set_my_color("w")
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")

    class R:
        pass
    r = R()
    r.grid = [row[:] for row in w.game.board.grid]
    w._last_sync_res = r
    started = []
    monkeypatch.setattr(w, "_click_worker", lambda mv: started.append(mv))
    w._auto_click_move(iccs_to_move("h2e2"), "炮二平五", "")
    assert len(started) == 1, "窗口一致时应正常发起代点"
    w.engine.stop() if w.engine else None


def test_blunder_audit_logs_big_eval_drop(qapp):
    """漏招审计：我方走后评估骤降应在日志中告警（便于发现识别导致的问题）。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w.board.set_my_color("w")
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR w - - 0 1")
    logs = []
    w._sync_log = lambda t: logs.append(t)

    # 我方待走：评估 +0.3（我方视角）
    w.panel.pv_lines = [{"multipv": 1, "score": 30, "score_kind": "cp", "stm": "w"}]
    w._audit_own_eval()
    assert w._own_eval_before == 30

    # 轮到我方走子后（对手回合）：引擎给出 -2.0（我方视角 = -(-200) ？）
    # 对手视角 -200 表示对手差，即我方优 -> 不算漏招；这里模拟我方骤降：
    w.panel.pv_lines = [{"multipv": 1, "score": 200, "score_kind": "cp", "stm": "b"}]
    w.game.board.side_to_move = "b"
    w._audit_own_eval()
    assert any("漏招审计" in t for t in logs), f"应记录漏招告警，实际 {logs}"
    w.engine.stop() if w.engine else None


def test_restart_takeover_resets_session_state(qapp):
    """重新接管必须复位上一会话的状态，否则会以各种方式静默不走棋。"""
    w = _win(qapp)
    w._await_new_game = True
    w._settling = True
    w._engine_kill_fen = "poison-fen"
    w._close_clicks = 3
    w._coord_clicks = 2
    w._auto_click_pending = {"move": iccs_to_move("h2e2"), "at": time.time()}
    w._pending_reply = iccs_to_move("h2e2")
    w._lift_key = (9, 0, "R")
    w._dropped_frames = 7
    w._last_auto_resync = time.time() - 2
    w._reset_session_state()
    assert w._await_new_game is False
    assert w._settling is False
    assert w._engine_kill_fen is None, "引擎崩溃黑名单必须随新会话解除"
    assert w._close_clicks == 0 and w._coord_clicks == 0
    assert w._auto_click_pending is None
    assert w._pending_reply is None
    assert w._lift_key is None
    assert w._dropped_frames == 0
    assert w._last_auto_resync == 0.0
    w.engine.stop() if w.engine else None


def test_click_worker_failure_clears_pending(qapp, monkeypatch):
    """代点失败（窗口不可用/被遮挡）必须清掉待确认代点，
    否则看门狗把它当"有动作在跑"而永不救援。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    mv = iccs_to_move("h9g7")
    # 坐标换算失败路径：无识别结果 -> _cell_screen_pos 返回 None
    w._last_sync_res = None
    monkeypatch.setattr(w, "_window_matches_board", lambda: True)
    w._clicking = False
    w._auto_click_move(mv, "马8进7", "")
    deadline = time.time() + 5
    while w._clicking and time.time() < deadline:
        time.sleep(0.02)
    assert w._auto_click_pending is None, "坐标换算失败后 pending 应被清除"
    w.engine.stop() if w.engine else None


def test_click_confirm_giveup_clears_pending(qapp):
    """代点重试耗尽放弃后必须清 pending，把救援权交还看门狗。"""
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    w._auto_click_retries = 1          # 已重试过一次
    w._auto_click_pending = {"move": iccs_to_move("h9g7"),
                             "at": time.time() - 10, "warned": False}
    w._check_auto_click_confirm()
    assert w._auto_click_pending is None, "放弃代点后 pending 应被清除"
    w.engine.stop() if w.engine else None


def test_looks_like_emulator():
    from gui.main_window import MainWindow
    assert MainWindow._looks_like_emulator("MuMu安卓设备")
    assert MainWindow._looks_like_emulator("MuMu模拟器")
    assert MainWindow._looks_like_emulator("雷电模拟器")
    assert not MainWindow._looks_like_emulator("Windows PowerShell")


def test_relocate_window_silent_switches_hwnd(qapp, monkeypatch):
    """句柄失效时按标题静默重定位到新窗口并重建基准。"""
    import numpy as np
    import capture_win
    w = _win(qapp)
    w._takeover = True
    w._takeover_title = "MuMu安卓设备"
    monkeypatch.setattr(capture_win, "list_windows", lambda: [
        {"title": "MuMu安卓设备", "hwnd": 99999, "minimized": False},
        {"title": "其它窗口", "hwnd": 88888, "minimized": False},
    ])
    monkeypatch.setattr(capture_win, "capture_window_client",
                        lambda h: np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(capture_win, "set_topmost", lambda h, on: None)

    class _R:
        source = "yolo"
        n_pieces = 32
        grid = Board().grid
        red_at_top = False
        board_box = (10, 10, 500, 600)
    import recognition.pipeline as _p
    import recognition.yolo_board as _yb
    monkeypatch.setattr(_p, "recognize_screenshot",
                        lambda img, model_path=None: _R())
    monkeypatch.setattr(_yb, "board_plausibility", lambda r: (True, ""))
    okr = w._relocate_window_silent()
    assert okr is True
    assert w._sync_hwnd == 99999 and w._takeover_hwnd == 99999
    w.engine.stop() if w.engine else None


def test_no_board_heal_saves_frame_and_logs(qapp, monkeypatch, tmp_path):
    """无棋盘持续 15 秒：存档画面（到临时目录）并写入详因日志。"""
    import numpy as np
    import capture_win
    w = _win(qapp)
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path))
    monkeypatch.setattr(capture_win, "is_window_valid", lambda h: True)
    monkeypatch.setattr(capture_win, "is_minimized", lambda h: False)
    monkeypatch.setattr(capture_win, "window_title", lambda h: "MuMu安卓设备")
    monkeypatch.setattr(w, "_relocate_window_silent", lambda: False)

    class _Res:
        source = "none"
        error = "未在截图中定位到棋盘"
    w._no_board_since = time.time() - 20
    w._no_board_heal(_Res(), np.zeros((50, 50, 3), np.uint8))
    assert (tmp_path / "xq_sync_no_board.png").exists(), "应存档无棋盘画面"
    assert w._no_board_saved is True
    w.engine.stop() if w.engine else None


def test_grab_fail_heal_restores_minimized(qapp, monkeypatch):
    """窗口最小化导致的抓屏失败：自动恢复窗口而不是静默卡住。"""
    import capture_win
    w = _win(qapp)
    calls = []
    monkeypatch.setattr(capture_win, "is_window_valid", lambda h: True)
    monkeypatch.setattr(capture_win, "is_minimized", lambda h: True)
    monkeypatch.setattr(capture_win, "restore_window",
                        lambda h: (calls.append(h), True)[1])
    w._grab_fail_heal()
    assert calls == [w._sync_hwnd], "应尝试恢复最小化的对局窗口"
    w.engine.stop() if w.engine else None


def test_synced_move_clears_stale_arrow(qapp):
    """窗口画面同步出着法后必须清掉旧建议箭头。

    实际 bug：同步路径不清理 best_move——整段对手回合都指着一个已经
    走掉的着法，对手走完后棋盘变了、箭头坐标却没变，看起来就是乱指。
    """
    w = _win(qapp)
    w.board.set_my_color(BLACK)
    w.game.reset("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/"
                 "1C5C1/9/RNBAKABNR b - - 0 1")
    # 一个"我方（黑）已走过马8进7"的观察帧
    observed = [row[:] for row in w.game.board.grid]
    observed[0][1] = None
    observed[2][2] = "n"
    w.board.set_game(w.game)
    w.board.set_best_move(((0, 1), (2, 2)))       # 摆一个旧箭头
    w._handle_sync_events([{"type": "changed", "grid": observed}],
                          _res(observed))
    assert w.board.best_move is None, "同步出着法后旧箭头应被清掉"
    w.engine.stop() if w.engine else None


def test_detect_systematic_shift():
    """识别整体偏移（遮挡伪影）的检测：偏移帧报出、正常帧不误报。"""
    from gui.main_window import MainWindow
    local = [[None] * 9 for _ in range(10)]
    for (r, c, p) in ((0, 3, "a"), (0, 4, "k"), (1, 4, "a"),
                      (3, 0, "p"), (3, 1, "R"), (7, 6, "N"),
                      (9, 4, "K"), (9, 5, "A")):
        local[r][c] = p
    # 观测 = 本地整体左移一格，且遮挡少了几颗
    shifted = [[None] * 9 for _ in range(10)]
    for r in range(10):
        for c in range(1, 9):
            shifted[r][c - 1] = local[r][c]
    shifted[0][0] = None
    s = MainWindow._detect_systematic_shift(local, shifted)
    assert s is not None, "整体偏移帧应当被识别出来"
    assert "列-1" in s
    # 完全一致 -> 不报
    assert MainWindow._detect_systematic_shift(local, local) is None
    # 正常的一步着法差异 -> 不报
    moved = [row[:] for row in local]
    moved[7][6] = None
    moved[5][5] = "N"
    assert MainWindow._detect_systematic_shift(local, moved) is None


def test_auto_resync_refuses_shifted_frame(qapp):
    """疑似整体偏移的观测帧不得被采纳为真相（棋局保持不变）。"""
    w = _win(qapp)
    w.board.set_my_color(RED)
    w.game.reset("3a1k3/9/9/2C6/9/9/4P4/9/9/3AKA3 w - - 0 1")
    local = [row[:] for row in w.game.board.grid]
    shifted = [[None] * 9 for _ in range(10)]
    for r in range(10):
        for c in range(1, 9):
            shifted[r][c - 1] = local[r][c]
    before = w.game.board.to_fen()
    w._last_auto_resync = 0.0
    w._auto_resync_to_window(_res(shifted), None)
    assert w.game.board.to_fen() == before, "偏移帧不应被采纳"
    assert getattr(w, "_shift_guard_n", 0) == 1
    w.engine.stop() if w.engine else None
