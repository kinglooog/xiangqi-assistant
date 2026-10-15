# -*- coding: utf-8 -*-
"""手动摆局编辑器：不依赖任何图像识别，点选棋子后点击棋盘落子。

这是截图识别的可靠兜底——无论对方使用什么棋子皮肤，
用户都能在十几秒内准确摆出任意局面（含中局残局）。

坐标系与 FEN 一致：row0 为黑方底线（棋盘上方显示）。
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (QButtonGroup, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
                             QWidget)

from core.board import PIECE_NAMES, RED, BLACK, side_of

BOARD_BG = QColor(238, 216, 178)
LINE_COLOR = QColor(90, 60, 30)
RED_COLOR = QColor(190, 30, 30)
BLACK_COLOR = QColor(30, 30, 30)
FACE_COLOR = QColor(248, 236, 208)

ERASE_TOOL = "ERASE"
MOVE_TOOL = "MOVE"


class BoardEditorWidget(QWidget):
    """棋盘编辑器：先选棋子（或"空"），再点棋盘格子放置。"""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid: List[List[Optional[str]]] = [[None] * 9 for _ in range(10)]
        self.tool = "P"           # 当前工具：FEN 字母 / ERASE_TOOL / MOVE_TOOL
        self.flip_display = False  # True 时上下颠倒显示（配合"红方在截图上方"）
        self._picked = None       # MOVE 工具拾起的格子
        self.marks: list = []     # 需要复核的格子（数据坐标），画橙色圈
        self.setMinimumSize(360, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---------------------------------------------------------------- 数据
    def set_grid(self, grid: List[List[Optional[str]]]) -> None:
        self.grid = [row[:] for row in grid]
        self._picked = None
        self.marks = []
        self.update()
        self.changed.emit()

    def set_marks(self, cells: list) -> None:
        """标出需要人工复核的格子（皮肤模板修正处等）。"""
        self.marks = list(cells)
        self.update()

    def get_grid(self) -> List[List[Optional[str]]]:
        return [row[:] for row in self.grid]

    def set_flip_display(self, flip: bool) -> None:
        """设置显示方向（内部数据始终是 FEN 顺序：row0 黑方底线）。"""
        self.flip_display = bool(flip)
        self.update()

    def _to_display(self, row: int, col: int):
        """数据坐标 -> 显示坐标。"""
        if self.flip_display:
            return 9 - row, 8 - col
        return row, col

    def _to_data(self, row: int, col: int):
        """显示坐标 -> 数据坐标。"""
        return self._to_display(row, col)

    def clear(self) -> None:
        self.grid = [[None] * 9 for _ in range(10)]
        self._picked = None
        self.update()
        self.changed.emit()

    def set_initial(self) -> None:
        from core.board import INITIAL_FEN, Board
        b = Board(INITIAL_FEN)
        self.set_grid([[p for p in row] for row in b.grid])

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._picked = None
        self.update()

    # ---------------------------------------------------------------- 几何
    def _metrics(self):
        w, h = self.width(), self.height()
        cell = min(w / 9.6, h / 10.6)
        x0 = (w - cell * 8) / 2
        y0 = (h - cell * 9) / 2
        return cell, x0, y0

    def _cell_at(self, pos):
        """显示坐标 -> 数据坐标。"""
        cell, x0, y0 = self._metrics()
        dcol = round((pos.x() - x0) / cell)
        drow = round((pos.y() - y0) / cell)
        if not (0 <= drow <= 9 and 0 <= dcol <= 8):
            return None
        if abs(pos.x() - (x0 + dcol * cell)) > cell * 0.5:
            return None
        if abs(pos.y() - (y0 + drow * cell)) > cell * 0.5:
            return None
        return self._to_data(drow, dcol)

    # ---------------------------------------------------------------- 绘制
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(40, 32, 24))
        cell, x0, y0 = self._metrics()
        p.fillRect(QRectF(x0 - cell * 0.5, y0 - cell * 0.5,
                          cell * 9, cell * 10), BOARD_BG)

        p.setPen(QPen(LINE_COLOR, 1.4))
        for r in range(10):
            p.drawLine(QRectF(x0, y0 + r * cell, 0, 0).topLeft(),
                       QRectF(x0 + 8 * cell, y0 + r * cell, 0, 0).topLeft())
        for c in range(9):
            p.drawLine(QRectF(x0 + c * cell, y0, 0, 0).topLeft(),
                       QRectF(x0 + c * cell, y0 + 9 * cell, 0, 0).topLeft())
        # 九宫斜线
        for r1, c1, r2, c2 in ((0, 3, 2, 5), (0, 5, 2, 3), (7, 3, 9, 5), (7, 5, 9, 3)):
            p.drawLine(QRectF(x0 + c1 * cell, y0 + r1 * cell, 0, 0).topLeft(),
                       QRectF(x0 + c2 * cell, y0 + r2 * cell, 0, 0).topLeft())

        if self._picked:
            p.setBrush(QBrush(QColor(240, 200, 60, 140)))
            p.setPen(Qt.PenStyle.NoPen)
            dr, dc = self._to_display(*self._picked)
            p.drawEllipse(QRectF(x0 + dc * cell - cell * 0.5, y0 + dr * cell - cell * 0.5,
                                 cell, cell))

        # 需复核的格子：橙色圈
        if self.marks:
            pen = QPen(QColor(255, 140, 0), max(2, int(cell * 0.07)))
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for (r, c) in self.marks:
                dr, dc = self._to_display(r, c)
                p.drawEllipse(QRectF(x0 + dc * cell - cell * 0.46,
                                     y0 + dr * cell - cell * 0.46,
                                     cell * 0.92, cell * 0.92))

        for r in range(10):
            for c in range(9):
                piece = self.grid[r][c]
                if piece is None:
                    continue
                dr, dc = self._to_display(r, c)
                self._draw_piece(p, cell, x0 + dc * cell, y0 + dr * cell, piece)

    def _draw_piece(self, p, cell, cx, cy, piece):
        rad = cell * 0.46
        p.setBrush(QBrush(FACE_COLOR))
        p.setPen(QPen(QColor(120, 80, 40), 1.3))
        p.drawEllipse(QRectF(cx - rad, cy - rad, rad * 2, rad * 2))
        font = QFont("Microsoft YaHei", max(7, int(cell * 0.34)))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(RED_COLOR if side_of(piece) == RED else BLACK_COLOR))
        p.drawText(QRectF(cx - rad, cy - rad, rad * 2, rad * 2),
                   Qt.AlignmentFlag.AlignCenter, PIECE_NAMES[piece])

    # ---------------------------------------------------------------- 交互
    def mousePressEvent(self, event):
        sq = self._cell_at(event.position())
        if sq is None:
            return
        r, c = sq
        if self.tool == MOVE_TOOL:
            if self._picked is None:
                if self.grid[r][c] is not None:
                    self._picked = sq
            elif self._picked == sq:
                self._picked = None
            else:
                pr, pc = self._picked
                self.grid[r][c] = self.grid[pr][pc]
                self.grid[pr][pc] = None
                self._picked = None
                self.changed.emit()
        elif self.tool == ERASE_TOOL:
            if self.grid[r][c] is not None:
                self.grid[r][c] = None
                self.changed.emit()
        else:
            # 放置棋子；再点同一格同一棋子则清空（便于纠错）
            if self.grid[r][c] == self.tool:
                self.grid[r][c] = None
            else:
                self.grid[r][c] = self.tool
            self.changed.emit()
        self.update()

    def contextMenuEvent(self, event):
        """右键擦除，方便快速纠错。"""
        sq = self._cell_at(event.pos())
        if sq is not None and self.grid[sq[0]][sq[1]] is not None:
            self.grid[sq[0]][sq[1]] = None
            self.changed.emit()
            self.update()


class PiecePalette(QWidget):
    """棋子/工具选择面板。选中项即当前工具。"""

    tool_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = {}
        self._current = "P"
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        def add_row(title, items):
            if title:
                lbl = QLabel(title)
                f = QFont()
                f.setBold(True)
                lbl.setFont(f)
                root.addWidget(lbl)
            row = QHBoxLayout()
            row.setSpacing(3)
            for key, text, color in items:
                btn = QToolButton()
                btn.setText(text)
                btn.setCheckable(True)
                btn.setFixedSize(40, 38)
                fnt = QFont("Microsoft YaHei", 13)
                fnt.setBold(True)
                btn.setFont(fnt)
                btn.setStyleSheet(
                    f"QToolButton {{ color: {color}; border: 1px solid #999;"
                    f" border-radius: 4px; background: #f6f0e2; }}"
                    f"QToolButton:checked {{ background: #ffd76e;"
                    f" border: 2px solid #b8860b; }}")
                btn.clicked.connect(lambda _=False, k=key: self._select(k))
                self._group.addButton(btn)
                self._buttons[key] = btn
                row.addWidget(btn)
            row.addStretch(1)
            root.addLayout(row)

        red_items = [(k, PIECE_NAMES[k], "#b81e1e")
                     for k in ("K", "A", "B", "N", "R", "C", "P")]
        black_items = [(k, PIECE_NAMES[k], "#1e1e1e")
                       for k in ("k", "a", "b", "n", "r", "c", "p")]
        add_row("红方", red_items)
        add_row("黑方", black_items)
        add_row("工具", [(ERASE_TOOL, "擦除", "#555"),
                        (MOVE_TOOL, "移动", "#555")])

        self._select("P")

    def _select(self, key: str) -> None:
        self._current = key
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)
        self.tool_changed.emit(key)

    def current_tool(self) -> str:
        return self._current
