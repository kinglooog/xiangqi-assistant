# -*- coding: utf-8 -*-
"""一键接管的唯一选择对话框：我执哪方 + 现在轮到谁（合并为一个选择）。"""
from __future__ import annotations

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QPushButton,
                             QRadioButton, QVBoxLayout)

from core.board import BLACK, RED

# (我方执子, 当前轮到我方?)
OPTIONS = [
    ("w", True,  "我执红方，现在轮到红方（软件立即开始走）"),
    ("w", False, "我执红方，现在轮到黑方（等对手走）"),
    ("b", True,  "我执黑方，现在轮到黑方（软件立即开始走）"),
    ("b", False, "我执黑方，现在轮到红方（等对手走）"),
]


class TakeoverDialog(QDialog):
    """接管确认（探测成功后弹出）：阵营自动推断，只需确认轮到谁。

    阵营推断依据：JJ（及绝大多数象棋 App）总是把"你"的棋子放在
    你屏幕的下半方。识别到红方在上 -> 你执黑；红方在下 -> 你执红。
    若你在观战/代他人走，可手动改选阵营。
    """

    def __init__(self, parent=None, red_at_top: bool = False,
                 window_title: str = "", n_pieces: int = 0):
        super().__init__(parent)
        self.setWindowTitle("一键接管 · 确认")
        self.resize(470, 300)
        derived = BLACK if red_at_top else RED
        self.my_color: str = derived
        self.my_turn_now: bool = True

        layout = QVBoxLayout(self)
        info = QLabel(f"已找到对局窗口：「{window_title[:30]}」"
                      f"（识别 {n_pieces} 颗棋子）")
        info.setWordWrap(True)
        layout.addWidget(info)

        ori = QLabel(f"检测到 {'红方在上方' if red_at_top else '红方在下方'}"
                     f" —— 按惯例你的棋子在"
                     f"{'上' if red_at_top else '下'}半屏，"
                     f"推断你执{'黑方' if derived == BLACK else '红方'}。")
        ori.setWordWrap(True)
        ori.setStyleSheet("color:#1b5e20;")
        layout.addWidget(ori)

        layout.addSpacing(4)
        h1 = QLabel("我执（一般不用改，自动按 JJ 惯例推断）：")
        layout.addWidget(h1)
        self.rb_my_auto = QRadioButton(
            f"自动推断：我执{'黑方' if derived == BLACK else '红方'}")
        self.rb_my_auto.setChecked(True)
        layout.addWidget(self.rb_my_auto)
        self.rb_my_flip = QRadioButton(
            f"反过来：我执{'红方' if derived == BLACK else '黑方'}"
            "（观战/代他人走时选这个）")
        layout.addWidget(self.rb_my_flip)

        layout.addSpacing(4)
        h2 = QLabel("现在轮到谁走：")
        f = h2.font()
        f.setBold(True)
        h2.setFont(f)
        layout.addWidget(h2)
        self.rb_turn_me = QRadioButton("轮到我方（软件立即开始走子）")
        self.rb_turn_me.setChecked(True)
        layout.addWidget(self.rb_turn_me)
        self.rb_turn_opp = QRadioButton("轮到对方（等对手走完自动接上）")
        layout.addWidget(self.rb_turn_opp)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始接管")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        if self.rb_my_flip.isChecked():
            self.my_color = RED if self.my_color == BLACK else BLACK
        self.my_turn_now = self.rb_turn_me.isChecked()
        super().accept()




class VerifyBoardDialog(QDialog):
    """接管局面核对：复刻盘与 JJ 对照，错处点调色板修正后 OK。

    这一步同时是皮肤校准：人工核对过的局面会作为模板采集，
    之后你皮肤上的识别误差（如 車↔炮）会被自动纠正。
    """

    def __init__(self, parent=None, grid=None, my_color=RED):
        super().__init__(parent)
        self.setWindowTitle("确认接管局面（一次性，约 10 秒）")
        self.resize(720, 760)
        self.confirmed_grid = None
        self.flipped = False        # 用户是否整盘翻转了 180°（红黑上下反了）

        from gui.board_editor import BoardEditorWidget, PiecePalette

        root = QVBoxLayout(self)
        tip = QLabel("对照 JJ 真实局面：识别错的棋子——左边选棋子后点击棋盘修正"
                     "（右键擦除）。全对就直接点 OK。")
        tip.setWordWrap(True)
        root.addWidget(tip)

        # 方向识别反了（复刻盘与 JJ 上下相反）时：先整盘翻转，再逐格修正
        self.btn_flip = QPushButton(
            "🔄 翻转棋盘（复刻盘与 JJ 上下相反时点这里，翻转后再逐格修正）")
        self.btn_flip.clicked.connect(self._flip_board)
        root.addWidget(self.btn_flip)

        self.editor = BoardEditorWidget()
        self.editor.setMinimumSize(400, 460)
        root.addWidget(self.editor, 1)

        self.palette = PiecePalette()
        self.palette.tool_changed.connect(self.editor.set_tool)
        root.addWidget(self.palette)

        if grid is not None:
            self.editor.set_grid([row[:] for row in grid])
        else:
            self.editor.set_initial()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认，开始下棋")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _flip_board(self) -> None:
        """整盘旋转 180°（行列同时反转）；再点一次翻回去。"""
        grid = self.editor.get_grid()
        self.editor.set_grid([row[::-1] for row in grid[::-1]])
        self.flipped = not self.flipped

    def accept(self):
        self.confirmed_grid = self.editor.get_grid()
        super().accept()
