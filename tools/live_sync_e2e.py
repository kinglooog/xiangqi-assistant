# -*- coding: utf-8 -*-
"""一期自动观察的端到端联调（需要真实显示器）：

1. 开一个真实窗口显示 JJ 开局棋盘（模拟 MuMu 画面）
2. MainWindow 连接该窗口开始观察（抓屏 -> YOLO -> LiveObserver）
3. 程序切换画面模拟对手走 马8进7 -> 验证着法被自动同步进棋局
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

import capture_win
capture_win.set_dpi_aware()

from core.board import Board, iccs_to_move, move_to_iccs, move_to_chinese
from recognition.synth import render_board

TITLE = "同步测试窗口"

OPENING = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
AFTER = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def show_board(label: QLabel, fen: str):
    b = Board(fen)
    img = render_board([[p for p in row] for row in b.grid])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg))


def main():
    app = QApplication(sys.argv)

    # 1) 真实窗口显示开局
    label = QLabel()
    label.setWindowTitle(TITLE)
    show_board(label, OPENING)
    label.move(60, 60)
    label.show()
    app.processEvents()
    time.sleep(0.6)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)

    # 2) 主窗口连接观察
    from gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1280, 840)
    w.game.reset(OPENING)
    w.board.set_game(w.game)
    w.board.set_my_color("w")
    w._refresh_windows()
    idx = w.combo_window.findText(TITLE)
    assert idx >= 0, f"未枚举到测试窗口: {[w.combo_window.itemText(i) for i in range(w.combo_window.count())]}"
    w.combo_window.setCurrentIndex(idx)
    w._toggle_sync()
    assert w._sync_hwnd is not None

    # 等基准建立（数个 tick）。注意不能匹配"建立基准"——初始提示里也有这词
    t0 = time.time()
    while time.time() - t0 < 15:
        app.processEvents()
        time.sleep(0.05)
        if "已锁定" in w.lbl_sync.text():
            break
    print("基准:", w.lbl_sync.text())
    assert "已锁定" in w.lbl_sync.text()
    assert len(w.game.history) == 0

    # 3a) 红方（我方）在 JJ 上走炮二平五 -> 应被自动记录
    b = Board(OPENING)
    mv_red = iccs_to_move("h2e2")
    b.make_move(mv_red)
    show_board(label, b.to_fen())
    label.raise_()
    app.processEvents()
    time.sleep(0.3)
    for _ in range(8):
        app.processEvents()
        time.sleep(0.05)
    t0 = time.time()
    while time.time() - t0 < 20:
        app.processEvents()
        time.sleep(0.05)
        if w.game.history:
            break
    print("我方着法同步:", w.game.move_list_chinese())
    assert w.game.history and move_to_iccs(w.game.history[-1][0]) == "h2e2"

    # 3b) 黑方应 马8进7 -> 同样被自动记录
    mv_black = iccs_to_move("h9g7")
    b.make_move(mv_black)
    show_board(label, b.to_fen())
    label.raise_()
    app.processEvents()
    time.sleep(0.3)
    for _ in range(8):
        app.processEvents()
        time.sleep(0.05)
    t0 = time.time()
    while time.time() - t0 < 20:
        app.processEvents()
        time.sleep(0.05)
        if len(w.game.history) >= 2:
            break
    print("双方着法:", w.game.move_list_chinese())
    assert len(w.game.history) >= 2
    assert move_to_iccs(w.game.history[-1][0]) == "h9g7"

    w._stop_sync()
    label.close()
    w.engine.stop() if w.engine else None
    print("\n端到端联调通过：抓屏 -> 识别 -> 双向着法自动同步 ✓")


if __name__ == "__main__":
    main()
