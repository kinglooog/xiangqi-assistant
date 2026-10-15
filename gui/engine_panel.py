# -*- coding: utf-8 -*-
"""引擎分析面板：评估条、实时信息、MultiPV 候选着法列表。

**红黑显示约定（避免搞混）**：
- 引擎给出的分数是「行棋方视角」，本面板一律换算成**红方视角**再显示，
  即：正数 = 红方占优，负数 = 黑方占优。文字里显式写出"红方领先/黑方领先"。
- 评估条左端固定为红方、右端固定为黑方，红色填充长度 = 红方胜率。
- 候选着法列表每一条都标注该着法属于哪一方（即当前行棋方）。
"""
from __future__ import annotations

import math
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QVBoxLayout, QWidget)

from core.board import RED, BLACK


def cp_to_winrate(cp: float) -> float:
    """百分兵评估 -> 红方胜率(%)，logistic 曲线。"""
    return 100.0 / (1.0 + math.exp(-cp / 300.0))


def red_cp(score: Optional[int], stm: str) -> Optional[float]:
    """把「行棋方视角」的分数换算成「红方视角」。"""
    if score is None:
        return None
    return float(score) if stm == RED else -float(score)


class _EvalBar(QWidget):
    """红黑评估条：左红右黑，红色填充比例 = 红方胜率。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 50.0          # 红方胜率 0-100
        self.setFixedHeight(22)
        self.setMinimumWidth(180)

    def set_value(self, v: float) -> None:
        self.value = max(0.0, min(100.0, v))
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QBrush, QPen
        from PyQt6.QtCore import QRectF
        p = QPainter(self)
        w, h = self.width(), self.height()
        # 黑方底
        p.fillRect(0, 0, w, h, QColor(40, 40, 40))
        # 红方填充（从左到右）
        red_w = int(w * self.value / 100.0)
        p.fillRect(0, 0, red_w, h, QColor(200, 45, 45))
        # 中线
        p.setPen(QPen(QColor(230, 230, 230), 2))
        p.drawLine(w // 2, 0, w // 2, h)
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()


class EnginePanel(QWidget):
    """右侧分析面板。所有 update_* 由主窗口通过 Qt 信号跨线程调用。"""

    info_arrived = pyqtSignal(dict)
    bestmove_arrived = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.my_color: Optional[str] = None
        self.pv_lines: List[dict] = []
        self._stm = RED
        self._engine_name = "引擎"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("引擎分析")
        font = QFont()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # 评估条 + 红黑端点标注
        bar_row = QHBoxLayout()
        bar_row.setSpacing(4)
        lbl_red = QLabel("红")
        lbl_red.setStyleSheet("color:#c62828; font-weight:bold;")
        lbl_red.setFixedWidth(16)
        bar_row.addWidget(lbl_red)
        self.eval_bar = _EvalBar()
        bar_row.addWidget(self.eval_bar, 1)
        lbl_black = QLabel("黑")
        lbl_black.setStyleSheet("color:#e0e0e0; background:#222;"
                                "font-weight:bold; padding:0 3px;")
        lbl_black.setFixedWidth(16)
        bar_row.addWidget(lbl_black)
        layout.addLayout(bar_row)

        # 主评估（红方视角，显式标出谁领先）
        self.eval_label = QLabel("评估：等待分析…")
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        self.eval_label.setFont(font2)
        self.eval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eval_label.setWordWrap(True)
        layout.addWidget(self.eval_label)

        self.winrate_label = QLabel("")
        self.winrate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.winrate_label)

        # 我方视角一句话结论
        self.myside_label = QLabel("")
        self.myside_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.myside_label.setWordWrap(True)
        layout.addWidget(self.myside_label)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        self.turn_label = QLabel("当前轮到：—")
        self.turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.turn_label)

        self.info_label = QLabel("深度 --   NPS --")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        layout.addWidget(QLabel("候选着法（MultiPV，按红方视角评估）:"))
        self.pv_list = QListWidget()
        self.pv_list.setMinimumHeight(150)
        layout.addWidget(self.pv_list, 1)

        layout.addWidget(QLabel("着法记录:"))
        self.move_list = QListWidget()
        layout.addWidget(self.move_list, 1)

        self.info_arrived.connect(self._on_info)
        self.bestmove_arrived.connect(self._on_bestmove)

    # ---------------------------------------------------------------- 外部调用
    @pyqtSlot(dict)
    def update_info(self, info: dict) -> None:
        self.info_arrived.emit(info)

    @pyqtSlot(object)
    def update_bestmove(self, move) -> None:
        self.bestmove_arrived.emit(move)

    def set_my_color(self, color: Optional[str]) -> None:
        self.my_color = color
        self._refresh_myside()

    def set_stm(self, stm: str) -> None:
        """设置当前行棋方（红/黑）。"""
        self._stm = stm
        who = "红方" if stm == RED else "黑方"
        self.turn_label.setText(f"当前轮到：<b>{who}</b>走子")
        self._refresh_myside()

    def set_engine_name(self, name: str) -> None:
        self._engine_name = name
        self._refresh_info()

    def clear_analysis(self) -> None:
        self.pv_lines.clear()
        self.pv_list.clear()
        self.eval_bar.set_value(50)
        self.eval_label.setText("评估：等待分析…")
        self.winrate_label.setText("")
        self.info_label.setText(f"深度 --   NPS --\n{self._engine_name}")

    # ---------------------------------------------------------------- 评估换算
    def _fmt_lead(self, cp: Optional[float], kind: str = "cp") -> str:
        """红方视角分数 -> 文字（显式标出谁领先）。"""
        if cp is None:
            return "—"
        if kind == "mate":
            n = int(abs(cp))
            who = "红方" if cp > 0 else "黑方"
            return f"{who} {n} 步内取胜"
        if abs(cp) < 1:
            return "双方均势"
        who = "红方" if cp > 0 else "黑方"
        return f"{who}领先 {abs(cp) / 100:.2f}"

    def _refresh_myside(self) -> None:
        if self.my_color is None:
            self.myside_label.setText("（未设定我方执子）")
            return
        side = "红方" if self.my_color == RED else "黑方"
        if not self.pv_lines:
            self.myside_label.setText(f"你执{side}")
            return
        top = self.pv_lines[0]
        cp = red_cp(top.get("score"), top.get("stm", self._stm))
        if cp is None:
            self.myside_label.setText(f"你执{side}")
            return
        kind = top.get("score_kind", "cp")
        if kind == "mate":
            mine = (cp > 0) == (self.my_color == RED)
            txt = "你即将取胜" if mine else "你即将被将死"
        elif abs(cp) < 1:
            txt = "局势均势"
        else:
            mine = (cp > 0) == (self.my_color == RED)
            txt = f"你{'领先' if mine else '落后'} {abs(cp) / 100:.2f}"
        self.myside_label.setText(f"你执{side}：{txt}")
        self.myside_label.setStyleSheet(
            "color:#1b5e20; font-weight:bold;"
            if ("领先" in txt or txt == "你即将取胜")
            else "color:#b71c1c; font-weight:bold;")

    def _refresh_info(self) -> None:
        if not self.pv_lines:
            self.info_label.setText(f"深度 --   NPS --\n{self._engine_name}")
            return
        top = self.pv_lines[0]
        self.info_label.setText(
            f"深度 {top.get('depth', '--')}/{top.get('seldepth', '--')}   "
            f"NPS {(top.get('nps') or 0) // 1000}k   "
            f"置换表 {(top.get('hashfull') or 0) / 10:.0f}%\n"
            f"{self._engine_name}")

    # ---------------------------------------------------------------- 内部
    @pyqtSlot(dict)
    def _on_info(self, info: dict) -> None:
        self.pv_lines = [l for l in self.pv_lines
                         if l.get("multipv") != info.get("multipv")]
        self.pv_lines.append(info)
        self.pv_lines.sort(key=lambda l: l.get("multipv", 1))

        top = self.pv_lines[0]
        cp = red_cp(top.get("score"), top.get("stm", self._stm))
        kind = top.get("score_kind", "cp")
        if cp is not None:
            self.eval_label.setText("评估：" + self._fmt_lead(cp, kind))
            if kind == "mate":
                wr = 100.0 if cp > 0 else 0.0
            else:
                wr = cp_to_winrate(cp)
            self.eval_bar.set_value(wr)
            self.winrate_label.setText(
                f"红方胜率 {wr:.0f}%　·　黑方胜率 {100 - wr:.0f}%")
        self._refresh_info()
        self._refresh_myside()
        self._refresh_pv_list()

    @pyqtSlot(object)
    def _on_bestmove(self, move) -> None:
        pass  # 主窗口直接处理

    def _refresh_pv_list(self) -> None:
        self.pv_list.clear()
        mover = "红" if self._stm == RED else "黑"
        for info in self.pv_lines:
            cp = red_cp(info.get("score"), info.get("stm", self._stm))
            kind = info.get("score_kind", "cp")
            lead = self._fmt_lead(cp, kind)
            pv_cn = info.get("pv_cn") or info.get("pv") or []
            seq = " ".join(pv_cn[:6])
            item = QListWidgetItem(
                f"{info.get('multipv', 1)}. [{mover}方] {lead}   {seq}")
            self.pv_list.addItem(item)
