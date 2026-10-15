# -*- coding: utf-8 -*-
"""一键接管端到端联调（需要真实显示器与鼠标输入权限）：

1. 开一个**可点击**的真实窗口显示 JJ 对局（点击棋子格→目标格会真实走子）
2. 主窗口走「一键接管」全流程：自动探测窗口 -> 采纳局面 -> 置顶 ->
   引擎出着自动代点 -> 画面回读确认 -> 黑方应着 -> 引擎继续
3. 断言：两步引擎着法被点击并同步，本地与窗口局面一致
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PyQt6.QtWidgets import QDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

import capture_win
capture_win.set_dpi_aware()

from core.board import Board, move_to_iccs
from recognition.synth import render_board

TITLE = "代点测试窗口"

# 渲染坐标（与 recognition.synth.render_board 一致）
ORIGIN = 80      # pad40 + margin40
CELL = 60

OPENING = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


class ClickableBoard(QLabel):
    """显示棋盘；两次点击（起点/终点）构成走法并真实应用到局面。"""

    def __init__(self):
        super().__init__()
        self.board = Board(OPENING)
        self.pending = None          # 已点击的起点
        self.accepted_moves = []     # 被接受的走法 (iccs)
        self.refresh()

    def refresh(self):
        # rotate_black=False：合成渲染的倒置黑子不在模型训练分布内，
        # 会造成识别噪音；本测试验证的是同步/代点管线，不是皮肤适应性
        img = render_board([[p for p in row] for row in self.board.grid],
                           rotate_black=False)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dpr = self.devicePixelRatioF() or 1.0
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                    rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        qi.setDevicePixelRatio(dpr)   # 图像物理像素 1:1 上屏，不被 Qt 缩放
        self.setPixmap(QPixmap.fromImage(qi))
        self.setFixedSize(round(rgb.shape[1] / dpr), round(rgb.shape[0] / dpr))

    def _cell_at(self, pos) -> tuple:
        dpr = self.devicePixelRatioF()
        x, y = pos.x() * dpr, pos.y() * dpr
        c = round((x - ORIGIN) / CELL)
        r = round((y - ORIGIN) / CELL)
        if not (0 <= r <= 9 and 0 <= c <= 8):
            return None
        if abs(x - (ORIGIN + c * CELL)) > CELL * 0.45:
            return None
        if abs(y - (ORIGIN + r * CELL)) > CELL * 0.45:
            return None
        return (r, c)

    def mousePressEvent(self, event):
        cell = self._cell_at(event.position())
        if cell is None:
            return
        if self.pending is None:
            if self.board.piece_at(*cell) is not None:
                self.pending = cell
        else:
            mv = (self.pending, cell)
            if self.board.is_legal_move(mv):
                self.board.make_move(mv)
                self.accepted_moves.append(move_to_iccs(mv))
            self.pending = None
            self.refresh()


def main():
    app = QApplication(sys.argv)
    # 防自污染：e2e 的自动确认桩会用模型标签采模板，删除并禁用保存
    if os.path.exists('models/skin_templates.npz'):
        os.remove('models/skin_templates.npz')
    import recognition.skin as _skin
    _skin.save_templates = lambda *a, **k: None

    win = ClickableBoard()
    win.setWindowTitle(TITLE)
    win.move(60, 60)
    win.show()
    capture_win.set_topmost(int(win.winId()), True)   # 防被其它窗口遮挡
    # 关键安全：把窗口枚举限制为测试窗口自身，
    # 避免测试误接管用户正在运行的真实对局（MuMu 尺寸更大，会被优先选中）
    _real_list = capture_win.list_windows
    capture_win.list_windows = lambda: [x for x in _real_list()
                                        if x["title"] == TITLE]
    app.processEvents()
    time.sleep(0.8)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)

    from gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1280, 840)
    w._refresh_windows()

    # 桩：新版接管对话框（探测后弹出）——阵营按方向自动推断
    # （测试窗口红在下方 -> 推断执红，与原假设一致）
    class FakeDialog:
        def __init__(self, parent=None, red_at_top=False, window_title="",
                     n_pieces=0):
            self.my_color = "b" if red_at_top else "w"
            self.my_turn_now = True

        def exec(self):
            return QDialog.DialogCode.Accepted

    import gui.main_window as gmw
    import gui.takeover_dialog as _td
    gmw.TakeoverDialog = FakeDialog
    _td.TakeoverDialog = FakeDialog

    # 核对对话框桩：自动确认（ e2e 验证的是管线；人工核对价值在真实使用中体现）
    class FakeVerify:
        confirmed_grid = None

        def __init__(self, parent=None, grid=None, my_color="w"):
            self.confirmed_grid = [row[:] for row in grid]

        def exec(self):
            return QDialog.DialogCode.Accepted

    _td.VerifyBoardDialog = FakeVerify   # 处理器从 _td 模块取类，必须打在这里
    gmw.VerifyBoardDialog = FakeVerify

    # 测试窗口与软件同进程；真实使用中 MuMu 是独立进程（PID 过滤只排自己）。
    # 这里把 pid 打桩成不同值以模拟真实场景。
    import os as _os
    _real_getpid = _os.getpid
    _os.getpid = lambda: 999999

    w._toggle_takeover()              # 一键接管：探测 -> 采纳 -> 置顶 -> 自动

    # 等接管成功（探测需逐窗口试，给足时间）
    t0 = time.time()
    while time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.05)
        if w._takeover and "接管成功" in w.lbl_sync.text():
            break
    print("接管状态:", w.lbl_sync.text())
    assert w._takeover and "接管成功" in w.lbl_sync.text(), "接管未成功"
    assert w.game.history == []
    assert win.board.to_fen().split()[0] == w.game.board.to_fen().split()[0], \
        "采纳的局面与窗口不一致"

    # 主循环：引擎自动代点红方；黑方由本脚本替走；两步引擎着法
    t0 = time.time()
    while len(win.accepted_moves) < 2 and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.05)
        # 黑方回合且本地已与窗口同步 -> 脚本替黑走一手，让引擎继续
        if (win.board.side_to_move == "b"
                and len(w.game.history) == len(win.accepted_moves)
                and win.board.to_fen().split()[0]
                    == w.game.board.to_fen().split()[0]):
            black_moves = win.board.legal_moves()
            if black_moves:
                win.board.make_move(black_moves[0])
                win.refresh()
                time.sleep(0.3)

    # 代点后观察回读有 1~2 tick 延迟，等本地记录跟上
    t0 = time.time()
    while len(w.game.history) < len(win.accepted_moves) + 1 and \
            time.time() - t0 < 15:
        app.processEvents()
        time.sleep(0.05)

    print("窗口接受的代点走法:", win.accepted_moves)
    print("本地同步的着法:", w.game.move_list_chinese())
    assert len(win.accepted_moves) >= 2, "代点未被测试窗口接受"
    assert win.accepted_moves[0] == move_to_iccs(w.game.history[0][0]), \
        "第一着代点与本地记录不一致"

    # 最后一着代点的观察回读需要 1~2 tick，等局面对齐
    t0 = time.time()
    while (w.game.board.to_fen().split()[0]
            != win.board.to_fen().split()[0]) and time.time() - t0 < 15:
        app.processEvents()
        time.sleep(0.05)

    # 本地棋盘应与窗口局面一致（经观察同步）
    assert w.game.board.to_fen().split()[0] == win.board.to_fen().split()[0], \
        "本地棋盘与测试窗口局面不同步"

    w._stop_sync()
    capture_win.set_topmost(int(win.winId()), False)
    capture_win.list_windows = _real_list        # 还原窗口枚举
    win.close()
    w.engine.stop() if w.engine else None
    print("\n一键接管端到端联调通过：自动找窗 -> 识别采纳 -> 自动走子 -> 回读确认 ✓")


if __name__ == "__main__":
    main()
