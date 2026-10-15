# -*- coding: utf-8 -*-
"""棋盘控件：绘制棋盘/棋子/高亮/最佳着法箭头，处理选子与落子。"""
from __future__ import annotations

import time
from typing import Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
                         QPolygonF)
from PyQt6.QtWidgets import QWidget

from core.board import PIECE_NAMES, RED, BLACK, side_of

Pt = Tuple[int, int]

BOARD_COLOR = QColor(231, 208, 168)
LINE_COLOR = QColor(90, 60, 30)
RED_COLOR = QColor(190, 30, 30)
BLACK_COLOR = QColor(30, 30, 30)
FACE_COLOR = QColor(244, 228, 196)
FACE_BORDER = QColor(120, 80, 40)
LAST_MOVE_COLOR = QColor(60, 160, 60, 110)
SEL_COLOR = QColor(240, 200, 60, 150)
ARROW_COLOR = QColor(240, 90, 30, 190)


class BoardWidget(QWidget):
    """复刻棋盘。点击选子/落子记录着法；支持拖拽。"""

    move_attempted = pyqtSignal(object)   # Move；由主窗口负责合法性检查与执行
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = None
        self.my_color: Optional[str] = None   # None = 红下黑上默认视角
        self.view_flip = False                # 手动翻转视角
        self.selected: Optional[Pt] = None
        self.last_move = None                 # Move
        self.best_move = None                 # Move（引擎建议）
        self.dragging: Optional[Pt] = None
        self.drag_pos = QPointF()
        # 走子动画：让用户看清对方走了什么
        self._anim = None                     # {'piece','move','t','dur'}
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)      # ~60fps
        self._anim_timer.timeout.connect(self._tick_anim)
        # 绝杀/将军特效
        self.fx = None                        # {'text','sub','color','king','t','dur'}
        self._fx_timer = QTimer(self)
        self._fx_timer.setInterval(16)
        self._fx_timer.timeout.connect(self._tick_fx)
        self.on_piece_grabbed = None          # 抬子回调（主窗口接音效）
        self.setMinimumSize(420, 460)
        self.setMouseTracking(True)

    # ---------------------------------------------------------------- 动画
    ANIM_MS = 300          # 单步走子动画时长

    def animate_move(self, move, piece: str, duration_ms: int = None) -> None:
        """播放一步走子的滑动动画（棋子从起点滑到终点）。"""
        if move is None or piece is None:
            return
        self._anim = {
            "piece": piece,
            "move": move,
            "dur": max(1, duration_ms or self.ANIM_MS),
            "start": time.monotonic(),   # 真实时钟：不受引擎高负载拖慢影响
        }
        self.last_move = None          # 动画期间不显示上一步高亮，避免重影
        self._anim_timer.start()
        self.update()

    def cancel_anim(self) -> None:
        """立即中止动画（悔棋、换局面时调用）。"""
        if self._anim is not None:
            self._anim = None
        self._anim_timer.stop()
        self.update()

    def _tick_anim(self):
        if self._anim is None:
            self._anim_timer.stop()
            return
        if (time.monotonic() - self._anim["start"]) * 1000 >= self._anim["dur"]:
            fin = self._anim
            self._anim = None
            self._anim_timer.stop()
            self.last_move = fin["move"]      # 动画结束再显示落点高亮
        self.update()

    # ---------------------------------------------------------------- 特效
    def start_effect(self, text: str, color: QColor, king: Optional[Pt] = None,
                     dur_ms: int = 2800, sub: str = "", scale: float = 1.0) -> None:
        """播放全盘特效（绝杀/将军）。

        text: 主文字（如"绝杀"）；sub: 副文字（如"黑方被将死"）；
        king: 被将死/被将军一方的帅(将)所在格，将以此为中心扩散色环。
        """
        self.fx = {"text": text, "sub": sub, "color": QColor(color),
                   "king": king, "start": time.monotonic(),
                   "dur": max(400, dur_ms), "scale": scale}
        self._fx_timer.start()
        self.update()

    def cancel_effect(self) -> None:
        if self.fx is not None:
            self.fx = None
        self._fx_timer.stop()
        self.update()

    def _tick_fx(self):
        if self.fx is None:
            self._fx_timer.stop()
            return
        if (time.monotonic() - self.fx["start"]) * 1000 >= self.fx["dur"]:
            self.fx = None
            self._fx_timer.stop()
        self.update()

    def _draw_fx(self, painter, cell):
        """特效绘制：全盘压暗 + 王位扩散色环 + 大字弹入。"""
        fx = self.fx
        t = min(1.0, (time.monotonic() - fx["start"]) * 1000 / fx["dur"])
        elapsed = fx["elapsed"] = t * fx["dur"]
        # 压暗：前 12% 渐入，最后 20% 渐出
        dim = min(1.0, fx["elapsed"] / (fx["dur"] * 0.12)) \
            * (1.0 - max(0.0, (t - 0.8) / 0.2))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, int(140 * dim))))
        painter.drawRect(self.rect())

        # 王位扩散环
        if fx["king"] is not None:
            center = self.cell_to_view(*fx["king"])
            for i in range(3):
                phase = (t * 1.6 + i * 0.33) % 1.0
                ring_alpha = int(230 * (1 - phase) * max(0.25, dim))
                if ring_alpha <= 4:
                    continue
                painter.setPen(QPen(QColor(fx["color"].red(),
                                           fx["color"].green(),
                                           fx["color"].blue(), ring_alpha),
                                    max(2, int(cell * 0.09 * (1 - phase * 0.5)))))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                radius = cell * (0.5 + phase * 2.1)
                painter.drawEllipse(center, radius, radius)
            # 王位常亮标记
            painter.setPen(QPen(fx["color"], max(2, int(cell * 0.08))))
            painter.setBrush(QBrush(QColor(fx["color"].red(), fx["color"].green(),
                                           fx["color"].blue(), int(70 * dim))))
            painter.drawEllipse(center, cell * 0.5, cell * 0.5)

        # 主文字：弹性放大进入（前 25%），结尾淡出
        appear = min(1.0, fx["elapsed"] / (fx["dur"] * 0.25))
        if appear < 1.0:
            back = 1 + 2.2 * ((appear - 1) ** 3) + 1.2 * ((appear - 1) ** 2)
        else:
            back = 1.0
        scale = fx["scale"] * (0.3 + 0.7 * appear) * back
        fade_out = 1.0 - max(0.0, (t - 0.82) / 0.18)
        alpha = int(255 * min(1.0, fx["elapsed"] / 120) * fade_out)

        cx, cy = self.width() / 2, self.height() * 0.42
        text = fx["text"]
        font = QFont("Microsoft YaHei", max(10, int(cell * 1.5 * scale)))
        font.setBold(True)
        painter.setFont(font)
        tw = painter.fontMetrics().horizontalAdvance(text)
        # 阴影 + 主体
        painter.setPen(QPen(QColor(0, 0, 0, int(alpha * 0.65))))
        off = max(1.5, cell * 0.05)
        painter.drawText(QPointF(cx - tw / 2 + off, cy + off), text)
        painter.setPen(QPen(QColor(fx["color"].red(), fx["color"].green(),
                                   fx["color"].blue(), alpha)))
        painter.drawText(QPointF(cx - tw / 2, cy), text)
        if fx["sub"]:
            sub_font = QFont("Microsoft YaHei", max(9, int(cell * 0.42)))
            painter.setFont(sub_font)
            painter.setPen(QPen(QColor(235, 235, 235, alpha)))
            w = painter.fontMetrics().horizontalAdvance(fx["sub"])
            painter.drawText(QPointF(cx - w / 2, cy + cell * 0.75), fx["sub"])

    # ---------------------------------------------------------------- 数据
    def set_game(self, game):
        self.game = game
        self.selected = None
        self.last_move = None
        self.best_move = None
        self.cancel_anim()
        self.cancel_effect()
        self.update()

    def set_my_color(self, color: Optional[str]) -> None:
        self.my_color = color
        self.update()

    def set_last_move(self, move) -> None:
        self.last_move = move
        self.update()

    def set_best_move(self, move) -> None:
        self.best_move = move
        self.update()

    def flipped(self) -> bool:
        """黑方在下方显示时为 True。"""
        base = self.my_color == BLACK
        return base != bool(self.view_flip)

    # ---------------------------------------------------------------- 几何
    def _metrics(self):
        """返回 (格距, 左上交点 x, y)（视图坐标）。

        高度按 11 格算：9 格棋盘 + 上下各约 1 格坐标区。"""
        w, h = self.width(), self.height()
        cell = min(w / 10.0, h / 12.0)
        x0 = (w - cell * 8) / 2
        y0 = (h - cell * 9) / 2
        return cell, x0, y0

    def cell_to_view(self, row: int, col: int) -> QPointF:
        cell, x0, y0 = self._metrics()
        if self.flipped():
            row, col = 9 - row, 8 - col
        return QPointF(x0 + col * cell, y0 + row * cell)

    def view_to_cell(self, pos: QPointF) -> Optional[Pt]:
        cell, x0, y0 = self._metrics()
        col = round((pos.x() - x0) / cell)
        row = round((pos.y() - y0) / cell)
        if not (0 <= row <= 9 and 0 <= col <= 8):
            return None
        # 点须落在交点附近
        dx = abs(pos.x() - (x0 + col * cell))
        dy = abs(pos.y() - (y0 + row * cell))
        if dx > cell * 0.45 or dy > cell * 0.45:
            return None
        if self.flipped():
            row, col = 9 - row, 8 - col
        return (row, col)

    # ---------------------------------------------------------------- 绘制
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(60, 45, 28))

        cell, x0, y0 = self._metrics()
        # 底色
        painter.fillRect(QRectF(x0 - cell * 0.55, y0 - cell * 0.55,
                                cell * 8 + cell * 1.1, cell * 9 + cell * 1.1),
                         BOARD_COLOR)

        pen = QPen(LINE_COLOR, 1.6)
        painter.setPen(pen)
        # 横线 10 条
        for r in range(10):
            p1 = self.cell_to_view(r, 0)
            p2 = self.cell_to_view(r, 8)
            painter.drawLine(p1, p2)
        # 竖线：两侧直通，中间因楚河汉界断开
        for c in range(9):
            top1 = self.cell_to_view(0, c)
            top2 = self.cell_to_view(4, c)
            bot1 = self.cell_to_view(5, c)
            bot2 = self.cell_to_view(9, c)
            painter.drawLine(top1, top2)
            painter.drawLine(bot1, bot2)
        # 九宫斜线
        for r1, c1, r2, c2 in ((0, 3, 2, 5), (0, 5, 2, 3),
                               (7, 3, 9, 5), (7, 5, 9, 3)):
            painter.drawLine(self.cell_to_view(r1, c1), self.cell_to_view(r2, c2))
        # 外框（粗）
        pen.setWidthF(3.0)
        painter.setPen(pen)
        tl = self.cell_to_view(0, 0)
        br = self.cell_to_view(9, 8)
        painter.drawRect(QRectF(tl.x() - cell * 0.14, tl.y() - cell * 0.14,
                                br.x() - tl.x() + cell * 0.28,
                                br.y() - tl.y() + cell * 0.28))
        pen.setWidthF(1.6)
        painter.setPen(pen)

        # 楚河 汉界
        font = QFont("Microsoft YaHei", int(cell * 0.32))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(LINE_COLOR))
        mid_y = (self.cell_to_view(4, 0).y() + self.cell_to_view(5, 0).y()) / 2
        painter.drawText(QPointF(x0 + cell * 1.2, mid_y + cell * 0.15), "楚  河")
        painter.drawText(QPointF(x0 + cell * 5.6, mid_y + cell * 0.15), "汉  界")

        # 兵/炮位标记
        self._draw_position_marks(painter, cell, x0, y0)

        # 红黑双方纵线坐标（红方中文数字、黑方阿拉伯数字）
        self._draw_coordinates(painter, cell, x0, y0)

        # 上一步高亮
        if self.last_move:
            for pt in self.last_move:
                c = self.cell_to_view(*pt)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(LAST_MOVE_COLOR))
                painter.drawEllipse(c, cell * 0.48, cell * 0.48)

        # 炮/兵位标记在棋子之下绘制完成，开始画棋子
        anim_from = self._anim["move"][0] if self._anim else None
        if self.game is not None:
            for r in range(10):
                for c in range(9):
                    if self._anim and (r, c) == anim_from:
                        continue          # 动画中的棋子稍后单独绘制
                    piece = self.game.board.piece_at(r, c)
                    if piece is None:
                        continue
                    if self.dragging == (r, c):
                        continue
                    self._draw_piece(painter, cell, self.cell_to_view(r, c), piece)

        # 选中高亮 + 可走点
        if self.selected:
            c = self.cell_to_view(*self.selected)
            painter.setBrush(QBrush(SEL_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(c, cell * 0.5, cell * 0.5)
            if self.game is not None:
                piece = self.game.board.piece_at(*self.selected)
                if piece is not None:
                    painter.setBrush(QBrush(QColor(30, 120, 60, 170)))
                    for m in self.game.board.legal_moves():
                        if m[0] == self.selected:
                            t = self.cell_to_view(*m[1])
                            painter.drawEllipse(t, cell * 0.13, cell * 0.13)

        # 走子动画中的棋子：在起点与终点之间插值滑动
        if self._anim:
            self._draw_animating_piece(painter, cell)

        # 拖拽中的棋子
        if self.dragging and self.game is not None:
            piece = self.game.board.piece_at(*self.dragging)
            if piece is not None:
                self._draw_piece(painter, cell * 1.08, self.drag_pos, piece)

        # 引擎最佳着法箭头
        if self.best_move:
            self._draw_arrow(painter, cell, self.best_move)

        # 绝杀/将军特效（最上层）
        if self.fx:
            self._draw_fx(painter, cell)

    def _draw_position_marks(self, painter, cell, x0, y0):
        """兵位与炮位的折角标记。"""
        pen = QPen(LINE_COLOR, 1.3)
        painter.setPen(pen)
        marks = {(2, 1), (2, 7), (3, 0), (3, 2), (3, 4), (3, 6), (3, 8),
                 (6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (7, 1), (7, 7)}
        for (r, c) in marks:
            center = self.cell_to_view(r, c)
            d = cell * 0.09
            g = cell * 0.16
            for sx in (-1, 1):
                for sy in (-1, 1):
                    # 边线上的点只画内侧
                    if (c == 0 and sx < 0) or (c == 8 and sx > 0):
                        continue
                    x = center.x() + sx * g
                    y = center.y() + sy * g
                    painter.drawLine(QPointF(x, y), QPointF(x + sx * d, y))
                    painter.drawLine(QPointF(x, y), QPointF(x, y + sy * d))

    def _draw_coordinates(self, painter, cell, x0, y0):
        """棋盘上下两侧的纵线坐标：红方一侧中文数字（一~九，自红方右起），
        黑方一侧阿拉伯数字（1~9，自黑方右起）。视角翻转时随之交换。"""
        red_nums = ["九", "八", "七", "六", "五", "四", "三", "二", "一"]
        flip = self.flipped()
        red_bottom = not flip
        font = QFont("Microsoft YaHei", max(8, int(cell * 0.22)))
        painter.setFont(font)
        for c in range(9):
            vx = x0 + ((8 - c) if flip else c) * cell
            y_red = y0 + 9 * cell + cell * 0.5 if red_bottom else y0 - cell * 0.5
            y_black = y0 - cell * 0.5 if red_bottom else y0 + 9 * cell + cell * 0.5
            painter.setPen(QPen(RED_COLOR))
            painter.drawText(QRectF(vx - cell / 2, y_red - cell * 0.3, cell, cell * 0.6),
                             Qt.AlignmentFlag.AlignCenter, red_nums[c])
            painter.setPen(QPen(BLACK_COLOR))
            painter.drawText(QRectF(vx - cell / 2, y_black - cell * 0.3, cell, cell * 0.6),
                             Qt.AlignmentFlag.AlignCenter, str(c + 1))

    def _draw_animating_piece(self, painter, cell):
        """绘制滑动中的棋子：起点->终点插值，带缓出效果，
        并给起点一个淡色残影，方便看清是从哪走到哪。"""
        anim = self._anim
        frm, to = anim["move"]
        t = min(1.0, (time.monotonic() - anim["start"]) * 1000 / anim["dur"])
        ease = 1 - (1 - t) ** 3                     # ease-out cubic
        p1 = self.cell_to_view(*frm)
        p2 = self.cell_to_view(*to)
        cur = QPointF(p1.x() + (p2.x() - p1.x()) * ease,
                      p1.y() + (p2.y() - p1.y()) * ease)

        # 起点残影（渐隐）
        ghost = QColor(120, 170, 235, int(150 * (1 - t)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(ghost))
        painter.drawEllipse(p1, cell * 0.44, cell * 0.44)

        # 终点落点圈（渐显）
        target = QColor(240, 150, 40, int(200 * t))
        painter.setBrush(QBrush(target))
        painter.drawEllipse(p2, cell * 0.5, cell * 0.5)

        # 滑动中的棋子略微放大，突出动感
        self._draw_piece(painter, cell * (1.0 + 0.12 * (1 - abs(t - 0.5) * 2)),
                         cur, anim["piece"])

    def _draw_piece(self, painter, cell, center: QPointF, piece: str):
        radius = cell * 0.46
        red = side_of(piece) == RED
        painter.setPen(QPen(FACE_BORDER, 1.4))
        painter.setBrush(QBrush(FACE_COLOR))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(FACE_BORDER, 0.9))
        painter.drawEllipse(center, radius * 0.86, radius * 0.86)
        color = RED_COLOR if red else BLACK_COLOR
        font = QFont("Microsoft YaHei", int(cell * 0.34))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color))
        name = PIECE_NAMES[piece]
        painter.drawText(QRectF(center.x() - radius, center.y() - radius,
                                radius * 2, radius * 2),
                         Qt.AlignmentFlag.AlignCenter, name)

    def _draw_arrow(self, painter, cell, move):
        p1 = self.cell_to_view(*move[0])
        p2 = self.cell_to_view(*move[1])
        pen = QPen(ARROW_COLOR, cell * 0.14, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QBrush(ARROW_COLOR))
        # 箭杆缩短留出箭头
        import math
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        dist = math.hypot(dx, dy) or 1
        ux, uy = dx / dist, dy / dist
        head = cell * 0.42
        start = QPointF(p1.x() + ux * cell * 0.42, p1.y() + uy * cell * 0.42)
        end = QPointF(p2.x() - ux * head, p2.y() - uy * head)
        painter.drawLine(start, end)
        # 箭头三角
        perp = QPointF(-uy, ux)
        tip = p2
        b1 = QPointF(end.x() + perp.x() * head * 0.55, end.y() + perp.y() * head * 0.55)
        b2 = QPointF(end.x() - perp.x() * head * 0.55, end.y() - perp.y() * head * 0.55)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([tip, b1, b2]))

    # ---------------------------------------------------------------- 交互
    def mousePressEvent(self, event):
        if self.game is None or self.game.result is not None:
            return
        pos = event.position()
        sq = self.view_to_cell(pos)
        self._press_toggle = False
        if sq is None:
            self.selected = None
            self.update()
            return
        # 已有选中棋子且点击其它位置 -> 尝试点击走子
        if self.selected is not None and sq != self.selected:
            move = (self.selected, sq)
            if self.game.board.is_legal_move(move):
                self.selected = None
                self.move_attempted.emit(move)
                self.update()
                return
            piece = self.game.board.piece_at(*sq)
            if piece is not None and side_of(piece) == self.game.side_to_move:
                # 改选另一枚己方棋子
                self.selected = sq
                self.dragging = sq
                self.drag_pos = pos
            else:
                self.selected = None
                self.status_message.emit("非法着法")
            self.update()
            return
        piece = self.game.board.piece_at(*sq)
        if piece is not None and side_of(piece) == self.game.side_to_move:
            self.dragging = sq
            self.drag_pos = pos
            # 按下前它已被选中 -> 松开时原地抬起则取消选择
            self._press_toggle = (self.selected == sq)
            self.selected = sq
            if self.on_piece_grabbed:
                try:
                    self.on_piece_grabbed()
                except Exception:
                    pass
        else:
            self.selected = None
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging is not None:
            self.drag_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging is None:
            return
        frm = self.dragging
        self.dragging = None
        to = self.view_to_cell(event.position())
        if to is None or to == frm:
            # 原地抬起：视为选子/取消选子（点击切换）
            self.selected = None if getattr(self, "_press_toggle", False) else frm
            self.update()
            return
        move = (frm, to)
        if self.game.board.is_legal_move(move):
            self.selected = None
            self.move_attempted.emit(move)
        else:
            self.selected = frm
            self.status_message.emit("非法着法")
        self.update()
