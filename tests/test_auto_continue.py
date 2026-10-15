# -*- coding: utf-8 -*-
"""自动续局测试：结算按钮检测（含安全区域约束）+ 新开局采纳与阵营推断。

重要安全用例：X 字形模板过于通用，若全画面搜索会匹配到窗口/应用自身的
关闭按钮，点击会直接退出游戏——因此 close 只允许在画面中部安全区域匹配。
"""
import os

import cv2
import numpy as np
import pytest

from core.board import BLACK, RED

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets", "jj")
CLOSE = os.path.join(SAMPLES, "close_btn.png")
AGAIN = os.path.join(SAMPLES, "again_btn.png")

requires_tpl = pytest.mark.skipif(
    not (os.path.isfile(CLOSE) and os.path.isfile(AGAIN)),
    reason="缺少按钮模板")


def _tpl(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _scene(with_close=True, close_in_corner=False, with_again=True):
    """构造"结算界面"模拟图：棋盘 + 弹窗叉 + 再来一局按钮。

    close_in_corner=True 时把叉放到窗口右上角（模拟"应用自身的关闭按钮"），
    用于验证安全区域约束——这种叉绝不能被匹配。
    """
    from recognition.synth import render_board
    from core.board import Board
    board_img = render_board([[p for p in row] for row in Board().grid])
    h, w = board_img.shape[:2]
    scene = np.full((h + 300, w, 3), 40, np.uint8)
    scene[:h] = board_img
    again_pos = (w // 2, h + 140)
    if with_again:
        again = _tpl(AGAIN)
        y = h + 120
        x = (w - again.shape[1]) // 2
        scene[y:y + again.shape[0], x:x + again.shape[1]] = again
        again_pos = (x + again.shape[1] // 2, y + again.shape[0] // 2)
    close_pos = None
    if with_close:
        close = _tpl(CLOSE)
        ch, cw = close.shape[:2]
        if close_in_corner:
            cx, cy = w - cw - 4, 4                    # 右上角：应用关闭按钮
        else:
            cx, cy = int(w * 0.78), int(h * 0.28)     # 弹窗内的叉
        scene[cy:cy + ch, cx:cx + cw] = close
        close_pos = (cx + cw // 2, cy + ch // 2)
    return scene, close_pos, again_pos


class _Res:
    def __init__(self, grid, red_at_top=False):
        self.grid = grid
        self.red_at_top = red_at_top
        self.n_pieces = sum(1 for row in grid for p in row if p)
        self.corrections = []
        self.source = "yolo"
        self.board_box = (100, 100, 900, 1000)
        self.detections = []


# ---------------------------------------------------------------- 按钮检测
@requires_tpl
def test_find_close_and_again():
    from recognition.buttons import find_button
    scene, close_pos, again_pos = _scene()
    hit_c = find_button(scene, "close")
    assert hit_c is not None and hit_c[2] > 0.8
    assert abs(hit_c[0] - close_pos[0]) < 8 and abs(hit_c[1] - close_pos[1]) < 8
    hit_a = find_button(scene, "again")
    assert hit_a is not None and hit_a[2] > 0.7
    assert abs(hit_a[0] - again_pos[0]) < 12 and abs(hit_a[1] - again_pos[1]) < 12


@requires_tpl
def test_corner_close_button_is_never_matched():
    """回归：窗口/应用自身的角落关闭按钮绝不能被匹配（否则直接退游戏）。"""
    from recognition.buttons import find_button
    scene, _cp, _ap = _scene(close_in_corner=True)
    assert find_button(scene, "close") is None, \
        "角落的关闭按钮被误匹配——会导致误点应用关闭、直接退出游戏"


@requires_tpl
def test_find_buttons_scale_robust():
    """窗口尺寸变化（0.8~1.3 倍）仍能检出。"""
    from recognition.buttons import find_button
    scene, _c, _a = _scene()
    for s in (0.8, 1.0, 1.3):
        m = cv2.resize(scene, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        assert find_button(m, "again") is not None, f"x{s} 再来一局未检出"
        assert find_button(m, "close") is not None, f"x{s} 叉未检出"


@requires_tpl
def test_no_false_positive_on_plain_board():
    """纯棋盘画面不应误检按钮。"""
    from recognition.buttons import find_button
    from recognition.synth import render_board
    from core.board import Board
    img = render_board([[p for p in row] for row in Board().grid])
    assert find_button(img, "close") is None
    assert find_button(img, "again") is None


# ---------------------------------------------------------------- 续局决策
def _win(qapp):
    from gui.main_window import MainWindow
    w = MainWindow()
    w._sync_hwnd = 12345                 # 不触发真实抓屏
    clicks = []
    w._click_window_point = lambda x, y, what: (clicks.append((what, x, y)) or True)
    w._clicks = clicks
    return w


@requires_tpl
def test_continue_prefers_again_without_clicking_close(qapp):
    """结算界面：能直接看到「再来一局」时不点叉，直接点它。"""
    from core.board import Board
    w = _win(qapp)
    w.chk_continue.setChecked(True)
    w.game.result = "红方胜（黑方被将死）"
    scene, _cp, _ap = _scene()           # 同时含叉与再来一局
    assert w._auto_continue_tick(scene) is True
    assert len(w._clicks) == 1
    assert w._clicks[0][0] == "再来一局"
    assert w._await_new_game is True
    assert w._close_clicks == 0, "能直接点再来一局时不应去点叉"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_continue_never_clicks_corner_close(qapp):
    """中局/无结算迹象时：角落的关闭按钮绝不能被点击（防误退游戏）。"""
    from core.board import Board
    w = _win(qapp)
    w.chk_continue.setChecked(True)
    w.game.result = None                  # 对局未结束
    scene, _cp, _ap = _scene(with_again=False, close_in_corner=True)
    assert w._auto_continue_tick(scene) is False
    assert w._clicks == [], f"误点了按钮: {w._clicks}"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_continue_clicks_close_only_when_game_over_and_central(qapp):
    """对局已结束、棋盘被遮挡、且「再来一局」不可见时，才点中部的叉（限 3 次）。"""
    from core.board import Board
    w = _win(qapp)
    w.chk_continue.setChecked(True)
    w.game.result = "红方胜（黑方被将死）"
    w._no_board_since = 1.0               # 棋盘读不到（被弹窗遮挡）
    w._coord_clicks = 2                   # 坐标兜底已用尽，隔离关弹窗分支
    scene, _cp, _ap = _scene(with_again=False, close_in_corner=False)
    for i in range(5):
        w._auto_continue_tick(scene)
    assert [c[0] for c in w._clicks] == ["关闭结算弹窗"] * 3, \
        f"应最多点 3 次叉，实际 {w._clicks}"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_continue_does_not_click_close_when_board_readable(qapp):
    """对局已结束但棋盘仍可读（无弹窗遮挡）时，不点叉。"""
    from core.board import Board
    w = _win(qapp)
    w.chk_continue.setChecked(True)
    w.game.result = "红方胜（黑方被将死）"
    w._no_board_since = 0.0               # 棋盘可读
    scene, _cp, _ap = _scene(with_again=False, close_in_corner=False)
    assert w._auto_continue_tick(scene) is False
    assert w._clicks == [], f"棋盘可读时不应点叉：{w._clicks}"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_continue_noop_midgame_without_buttons(qapp):
    """中局正常棋盘：续局逻辑应完全不动作。"""
    from core.board import Board
    from recognition.synth import render_board
    w = _win(qapp)
    w.chk_continue.setChecked(True)
    w.game.result = None
    img = render_board([[p for p in row] for row in Board().grid])
    assert w._auto_continue_tick(img) is False
    assert w._clicks == []
    w.engine.stop() if w.engine else None


# ---------------------------------------------------------------- 新开局采纳
def test_adopt_new_game_infers_side_and_turn(qapp):
    """新开局：红在上 -> 我方执黑且等对方先走；红在下 -> 执红先走。"""
    from core.board import Board
    w = _win(qapp)
    grid = [[p for p in row] for row in Board().grid]
    w._await_new_game = True
    w._adopt_new_game(_Res(grid, red_at_top=False))
    assert w.board.my_color == RED
    assert w.game.side_to_move == RED          # 红先行
    assert w._await_new_game is False
    assert len(w.game.history) == 0

    w._await_new_game = True
    w._adopt_new_game(_Res(grid, red_at_top=True))
    assert w.board.my_color == BLACK
    assert w.game.side_to_move == RED          # 对方（红）先走
    w.engine.stop() if w.engine else None


def test_adopt_new_game_resets_previous_state(qapp):
    """新开局不受上一局影响：历史与各类残留状态全部复位。"""
    from core.board import Board, iccs_to_move
    w = _win(qapp)
    w.game.play(iccs_to_move("h2e2"))
    w._auto_click_pending = {"move": iccs_to_move("h2e2"), "at": 0, "warned": False}
    w._pending_reply = iccs_to_move("h9g7")
    w._last_mover_side = "w"
    w._partial_key = (1, 2, "R", None)
    w._close_clicks = 2
    grid = [[p for p in row] for row in Board().grid]
    w._await_new_game = True
    w._adopt_new_game(_Res(grid))
    assert len(w.game.history) == 0, "新开局应清空历史"
    assert w._auto_click_pending is None
    assert w._pending_reply is None
    assert w._last_mover_side is None
    assert w._partial_key is None
    assert w._close_clicks == 0
    w.engine.stop() if w.engine else None


@requires_tpl
def test_continue_triggered_even_when_board_partially_readable(qapp):
    """回归：结算弹窗只挡棋盘中部、上下仍可识别时，续局逻辑也必须触发。

    曾经的 bug：只在"棋盘完全识别不到"时才尝试续局按钮，
    而实际弹窗常只遮住中部 -> 续局从不触发 -> 不会自动开下一局。
    """
    from gui.main_window import MainWindow
    from live import LiveObserver
    w = MainWindow()
    w._sync_hwnd = 12345
    w._sct = object()                      # 占位，避免 _grab_window 早退
    w.chk_continue.setChecked(True)
    w.game.result = "红方胜（黑方被将死）"
    # 棋盘"部分可读"：识别成功（n_pieces=18），但结算界面按钮可见
    scene, _cp, _ap = _scene()
    clicks = []
    w._click_window_point = lambda x, y, what: (clicks.append(what) or True)

    class FakeRes:
        source = "yolo"
        n_pieces = 18                      # 部分可读（>=4，不触发"读不到"分支）
        error = None
        grid = [[None] * 9 for _ in range(10)]
        board_box = (100, 100, 600, 700)
        red_at_top = False
        corrections = []
        detections = []

    import recognition.pipeline as _pl
    orig = _pl.recognize_screenshot
    _pl.recognize_screenshot = lambda img, model_path=None, **k: FakeRes()
    w._grab_window = lambda: scene
    try:
        w._sync_tick_inner()
    finally:
        _pl.recognize_screenshot = orig
    assert "再来一局" in clicks, f"部分可读时续局未触发，点击记录: {clicks}"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_coord_fallback_uses_frame_anchor(qapp):
    """模板未匹配时，按画面锚点坐标兜底点击「再来一局」。"""
    import time as _t
    from gui.main_window import MainWindow
    from recognition.buttons import settle_button_point
    w = MainWindow()
    w._sync_hwnd = 12345
    w.chk_continue.setChecked(True)
    w.game.result = "红方胜（黑方被将死）"
    w._no_board_since = _t.time() - 10     # 棋盘遮挡已超 3 秒
    clicks = []
    w._click_window_point = lambda x, y, what: (clicks.append((what, x, y)) or True)
    # 画面里没有任何按钮（模板必然不匹配）
    img = np.full((900, 800, 3), 30, np.uint8)
    assert w._auto_continue_tick(img) is True
    assert len(clicks) == 1 and "坐标定位" in clicks[0][0], clicks
    exp = settle_button_point("again", img.shape)
    assert abs(clicks[0][1] - exp[0]) <= 1 and abs(clicks[0][2] - exp[1]) <= 1
    assert w._await_new_game is True
    w.engine.stop() if w.engine else None


def test_settle_button_point_matches_both_references(qapp):
    """画面锚点换算精度：两张真实结算截图都应还原出按钮实测位置。

    这两张截图的棋盘大小不同（600 / 477 像素高）而按钮位置几乎相同，
    证明按钮锚定在界面底部、与棋盘大小无关——按画面比例换算才正确。
    """
    from recognition.buttons import settle_button_point
    cases = [((1090, 613), (321, 1017)), ((1089, 613), (305, 1015))]
    for shape, actual in cases:
        pt = settle_button_point("again", shape)
        assert pt is not None
        assert abs(pt[0] - actual[0]) <= 20, f"横向偏差过大 {pt} vs {actual}"
        assert abs(pt[1] - actual[1]) <= 6, f"纵向偏差过大 {pt} vs {actual}"


@requires_tpl
def test_continue_clicks_when_board_fully_visible_and_no_result(qapp):
    """回归（实际 bug）：结算界面棋盘完整可见、且本地判不出胜负
    （对手认输）时，也必须能点「再来一局」。

    旧逻辑要求"对局已结束"或"棋盘读不到"，两者都不满足就不触发，
    导致按钮明明可见却不点。
    """
    import recognition.pipeline as _pl
    from gui.main_window import MainWindow
    from core.board import Board
    w = MainWindow()
    w._sync_hwnd = 12345
    w._sct = object()
    w.chk_continue.setChecked(True)
    w.game.result = None                    # 本地判不出胜负
    scene, _cp, _ap = _scene()              # 棋盘完整可见 + 按钮可见
    clicks = []
    w._click_window_point = lambda x, y, what: (clicks.append(what) or True)

    class FakeRes:
        source = "yolo"
        n_pieces = 32                       # 棋盘完整可读
        error = None
        grid = [[p for p in row] for row in Board().grid]
        board_box = (95, 154, 518, 631)
        red_at_top = False
        corrections = []
        detections = []

    orig = _pl.recognize_screenshot
    _pl.recognize_screenshot = lambda img, model_path=None, **k: FakeRes()
    w._grab_window = lambda: scene
    try:
        for _ in range(4):                  # 隔帧检查，多跑几拍
            w._sync_tick_inner()
    finally:
        _pl.recognize_screenshot = orig
    assert "再来一局" in clicks, f"棋盘完整可见时未点再来一局，点击记录: {clicks}"
    w.engine.stop() if w.engine else None


@requires_tpl
def test_probe_accepts_settlement_screen(qapp, monkeypatch):
    """回归（实际 bug）：结算界面上棋盘被弹窗遮挡 -> 棋子不全、将帅被挡，
    合理性校验必然失败。此时只要看到「再来一局」就必须接受该窗口，
    否则接管根本建立不起来，续局逻辑永远不会运行。
    """
    import capture_win
    from gui.main_window import MainWindow
    from core.board import Board

    w = MainWindow()
    scene, _cp, _ap = _scene()               # 含「再来一局」的结算画面
    # 结算画面下的"残缺识别结果"：棋子不全、缺少将帅
    partial = [[None] * 9 for _ in range(10)]
    for r, c, p in [(0, 0, "r"), (0, 8, "r"), (2, 1, "c"), (3, 0, "p"),
                    (6, 4, "P"), (7, 1, "C"), (9, 0, "R"), (9, 8, "R")]:
        partial[r][c] = p

    class _Res:
        source = "yolo"
        n_pieces = 8
        error = None
        red_at_top = False
        board_box = (95, 154, 518, 631)
        corrections = []
        detections = []
    res = _Res()
    res.grid = partial

    monkeypatch.setattr(capture_win, "capture_window_client",
                        lambda hwnd: scene)
    monkeypatch.setattr(capture_win, "list_windows",
                        lambda: [{"hwnd": 1, "title": "MuMu安卓设备",
                                  "rect": (0, 0, 900, 1100), "minimized": False,
                                  "pid": 4242, "cls": "MuMu"}])
    monkeypatch.setattr(capture_win, "get_client_rect",
                        lambda hwnd: (0, 0, 900, 1100))
    import recognition.pipeline as _pl
    monkeypatch.setattr(_pl, "recognize_screenshot",
                        lambda img, model_path=None, **k: res)
    captured = []
    w.takeoverFound.connect(lambda p: captured.append(p))
    w._probe_worker()                        # 直接跑探测
    assert captured, "结算界面被探测拒绝，接管建立不起来"
    assert captured[0].get("settling") is True
    w.engine.stop() if w.engine else None


@requires_tpl
def test_settlement_takeover_starts_continue(qapp):
    """结算界面接管后：不弹对话框、直接连窗口，且续局逻辑能点按钮。"""
    import recognition.pipeline as _pl
    from gui.main_window import MainWindow
    w = MainWindow()
    scene, _cp, _ap = _scene()
    clicks = []
    w._click_window_point = lambda x, y, what: (clicks.append(what) or True)

    class _Res:
        source = "yolo"
        n_pieces = 8
        error = None
        red_at_top = False
        board_box = (95, 154, 518, 631)
        corrections = []
        detections = []
    res = _Res()
    res.grid = [[None] * 9 for _ in range(10)]
    w._on_takeover_found({"hwnd": 999, "res": res, "title": "MuMu",
                          "img": scene, "settling": True})
    assert w._sync_hwnd == 999
    assert w._settling is True
    assert w._sync_timer.isActive()
    # 续局：即便没有棋盘、没有胜负结果，也应点「再来一局」
    assert w._auto_continue_tick(scene) is True
    assert "再来一局" in clicks
    w._stop_sync()
    w.engine.stop() if w.engine else None
