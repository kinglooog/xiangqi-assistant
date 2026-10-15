# -*- coding: utf-8 -*-
"""接管确认界面的「翻转棋盘」功能测试。

场景：红黑上下方向被识别反了（如把红方在上误判/漏判），用户点
「翻转棋盘」后：网格翻 180°、阵营推断跟着翻，后续点击坐标换算
与同步基准都基于翻转后的 red_at_top。
"""
from core.board import RED, BLACK, Board


def test_takeover_dialog_flip_toggles_orientation_and_side(qapp):
    from gui.takeover_dialog import TakeoverDialog
    d = TakeoverDialog(red_at_top=True, window_title="测试窗口", n_pieces=32)
    assert d.flip_orientation is False
    assert "黑方" in d.rb_my_auto.text()          # 红在上 -> 推断执黑
    d._toggle_flip()
    assert d.flip_orientation is True
    assert "红方" in d.rb_my_auto.text(), "翻转后推断应变执红"
    d.accept()
    assert d.my_color == RED
    d.deleteLater()


def test_takeover_dialog_flip_twice_resets(qapp):
    from gui.takeover_dialog import TakeoverDialog
    d = TakeoverDialog(red_at_top=True, window_title="测试窗口", n_pieces=32)
    d._toggle_flip()
    d._toggle_flip()
    assert d.flip_orientation is False, "翻两次应回到原方向"
    d.accept()
    assert d.my_color == BLACK
    d.deleteLater()


def test_apply_orientation_flip_rotates_grid(qapp):
    """翻转：red_at_top 取反 + 网格旋转 180°（行列同时反转）。"""
    from gui.main_window import MainWindow

    class _R:
        pass

    r = _R()
    r.red_at_top = True
    r.grid = [row[:] for row in Board().grid]
    expect = [row[::-1] for row in r.grid[::-1]]
    MainWindow._apply_orientation_flip(r)
    assert r.red_at_top is False
    assert r.grid == expect, "网格应旋转 180°"
    # 再翻一次回到原状
    MainWindow._apply_orientation_flip(r)
    assert r.red_at_top is True
    assert r.grid == Board().grid
