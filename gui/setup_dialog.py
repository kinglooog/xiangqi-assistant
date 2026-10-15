# -*- coding: utf-8 -*-
"""手动摆局对话框：完全不依赖图像识别。

用调色板点选棋子后点击棋盘摆放，支持任意中局/残局。
这是截图识别失效时的可靠入口，也是最快的自定义局面方式。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QPushButton, QRadioButton, QVBoxLayout)

from core.board import BLACK, RED, Board, side_of
from gui.board_editor import BoardEditorWidget, PiecePalette
from recognition.classifier import grid_to_fen


class SetupDialog(QDialog):
    """返回 self.fen（FEN 顺序）与 self.my_color。"""

    def __init__(self, parent=None, fen: str = None, my_color: str = RED):
        super().__init__(parent)
        self.setWindowTitle("手动摆局")
        self.resize(760, 700)
        self.fen: Optional[str] = None
        self.my_color: Optional[str] = None

        root = QVBoxLayout(self)
        tip = QLabel("点选下方棋子 → 点击棋盘落子；右键擦除。"
                     "支持「移动」工具拖动已有棋子。")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.editor = BoardEditorWidget()
        self.editor.setMinimumSize(420, 480)
        root.addWidget(self.editor, 1)

        self.palette = PiecePalette()
        self.palette.tool_changed.connect(self.editor.set_tool)
        root.addWidget(self.palette)

        preset = QHBoxLayout()
        self.btn_initial = QPushButton("设为开局")
        self.btn_clear = QPushButton("清空")
        self.btn_flip = QPushButton("翻转显示")
        preset.addWidget(self.btn_initial)
        preset.addWidget(self.btn_clear)
        preset.addWidget(self.btn_flip)
        preset.addStretch(1)
        root.addLayout(preset)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("我方执子:"))
        self.rb_red = QRadioButton("红方")
        self.rb_black = QRadioButton("黑方")
        self.rb_red.setChecked(my_color != BLACK)
        self.rb_black.setChecked(my_color == BLACK)
        opts.addWidget(self.rb_red)
        opts.addWidget(self.rb_black)
        opts.addSpacing(16)
        opts.addWidget(QLabel("当前轮到:"))
        self.rb_turn_red = QRadioButton("红方")
        self.rb_turn_black = QRadioButton("黑方")
        self.rb_turn_red.setChecked(True)
        opts.addWidget(self.rb_turn_red)
        opts.addWidget(self.rb_turn_black)
        opts.addStretch(1)
        root.addLayout(opts)

        self.lbl_stat = QLabel("")
        root.addWidget(self.lbl_stat)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.btn_initial.clicked.connect(self.editor.set_initial)
        self.btn_clear.clicked.connect(self.editor.clear)
        self.btn_flip.clicked.connect(self._flip_display)
        self.editor.changed.connect(self._update_stat)

        if fen:
            try:
                self.editor.set_grid([[p for p in row] for row in Board(fen).grid])
            except Exception:
                self.editor.set_initial()
        else:
            self.editor.set_initial()
        self._update_stat()

    def _flip_display(self):
        self.editor.set_flip_display(not self.editor.flip_display)

    def _update_stat(self):
        grid = self.editor.get_grid()
        n_red = sum(1 for row in grid for p in row
                    if p and side_of(p) == RED)
        n_black = sum(1 for row in grid for p in row
                      if p and side_of(p) == BLACK)
        self.lbl_stat.setText(f"红方 {n_red} 子，黑方 {n_black} 子"
                              f"（标准开局各 16 子）")

    def _accept(self):
        grid = self.editor.get_grid()
        if not any(p for row in grid for p in row):
            self.lbl_stat.setText("⚠ 棋盘为空，请先摆放棋子")
            return
        stm = RED if self.rb_turn_red.isChecked() else BLACK
        fen = grid_to_fen(grid, stm)
        board = Board(fen)
        problem = board.validate_position()
        if problem:
            self.lbl_stat.setText(f"⚠ 局面不合法：{problem}（请检查摆放）")
            return
        self.fen = fen
        self.my_color = RED if self.rb_red.isChecked() else BLACK
        self.accept()
