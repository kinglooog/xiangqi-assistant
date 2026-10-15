# -*- coding: utf-8 -*-
"""接管核对界面的「翻转棋盘」功能测试。

场景：红黑上下方向识别反了——在"确认接管局面"核对界面里整盘翻转
180° 后再逐格修正（用户要求：翻转入口放这里，接管确认框里不要）；
采纳时方向标记 red_at_top 同步取反。
"""
from core.board import RED, Board


def test_verify_dialog_flip_rotates_editor(qapp):
    from gui.takeover_dialog import VerifyBoardDialog
    g = [row[:] for row in Board().grid]
    d = VerifyBoardDialog(grid=g, my_color=RED)
    assert d.flipped is False
    d._flip_board()
    assert d.flipped is True
    assert d.editor.get_grid() == [row[::-1] for row in g[::-1]], "整盘应旋转 180°"
    d._flip_board()
    assert d.flipped is False
    assert d.editor.get_grid() == g, "翻两次应回到原状"
    d.deleteLater()


def test_takeover_dialog_has_no_flip_button(qapp):
    """翻转入口只在核对界面（接管确认框里不加，用户明确要求）。"""
    from gui.takeover_dialog import TakeoverDialog
    d = TakeoverDialog(red_at_top=True, window_title="t", n_pieces=32)
    assert not hasattr(d, "btn_flip")
    d.deleteLater()


def test_adoption_applies_verify_flip(qapp, monkeypatch):
    """核对界面翻转后采纳：res.red_at_top 取反，棋局采用翻转后的网格。"""
    from PyQt6.QtWidgets import QDialog
    from gui import takeover_dialog as _td
    from gui.main_window import MainWindow

    w = MainWindow()
    w._sync_hwnd = 12345
    try:
        w.engine.stop()
    except Exception:
        pass
    w.engine = None                      # 采纳流程里的分析触发直接短路

    grid = [row[:] for row in Board().grid]
    flipped = [row[::-1] for row in grid[::-1]]

    class FakeTakeover:
        def __init__(self, *a, **k):
            self.my_color = "w"
            self.my_turn_now = True

        def exec(self):
            return QDialog.DialogCode.Accepted

    class FakeVerify:
        flipped = True

        def __init__(self, *a, **k):
            self.confirmed_grid = [r[:] for r in flipped]

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(_td, "TakeoverDialog", FakeTakeover)
    monkeypatch.setattr(_td, "VerifyBoardDialog", FakeVerify)

    res_grid = [r[:] for r in grid]      # 类体推导式拿不到函数局部变量

    class _Res:
        red_at_top = True
        grid = res_grid
        n_pieces = 32
        board_box = (0.0, 0.0, 100.0, 100.0)
        detections = []
        corrections = []
        error = None

    res = _Res()
    w._on_takeover_found({"hwnd": 99999, "res": res, "title": "t", "img": None})
    assert res.red_at_top is False, "翻转后方向标记应取反"
    assert w.game.board.grid == flipped, "棋局应采用翻转后的网格"
    w._stop_sync()
