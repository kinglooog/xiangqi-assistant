# -*- coding: utf-8 -*-
"""截图导入对话框：校准棋盘 -> 识别 -> 人工校对 -> 设定我方与轮次。

流程：
1. 载入截图（文件或剪贴板粘贴）。
2. 自动定位四角；失败或用户点击"手动校准"时，依次点击
   左上/右上/右下/左下四个角交点。
3. 透视变换 + 识别 90 交点 -> 显示识别结果棋盘。
4. 用户可点击格位循环修正棋子；选择我方执子颜色与当前轮到谁。
5. 确认后主窗口得到 FEN + 我方颜色。
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox,
                             QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QRadioButton, QVBoxLayout, QWidget)

from core.board import PIECE_NAMES, RED, BLACK
from gui.board_editor import BoardEditorWidget, PiecePalette
from recognition.classifier import grid_to_fen
from recognition.detector import BoardLocator, auto_detect_corners
from recognition.pipeline import recognize_screenshot, yolo_available

_CORNER_NAMES = ["左上角交点", "右上角交点", "右下角交点", "左下角交点"]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def load_image_file(path: str) -> Optional[np.ndarray]:
    """读取图片文件为 BGR ndarray（支持中文/空格路径）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_from_mime(mime) -> Optional[np.ndarray]:
    """从拖拽/剪贴板数据里取出图片。支持文件路径与图片数据。"""
    from PyQt6.QtCore import QUrl
    if mime is None:
        return None
    # 1) 图片数据（截图工具直接拖入）
    if mime.hasImage():
        img = mime.imageData()
        if isinstance(img, QImage) and not img.isNull():
            return _qimage_to_bgr_static(img)
    # 2) 本地文件
    if mime.hasUrls():
        for url in mime.urls():
            if not isinstance(url, QUrl):
                continue
            if url.isLocalFile():
                p = url.toLocalFile()
                if p.lower().endswith(IMAGE_EXTS) or os.path.isfile(p):
                    arr = load_image_file(p)
                    if arr is not None:
                        return arr
    # 3) 文本里的文件路径（从资源管理器复制文件后粘贴）
    if mime.hasText():
        for line in mime.text().splitlines():
            line = line.strip().strip('"')
            if line.lower().endswith(IMAGE_EXTS) and os.path.isfile(line):
                arr = load_image_file(line)
                if arr is not None:
                    return arr
    return None


def _qimage_to_bgr_static(image: QImage) -> np.ndarray:
    """QImage -> BGR ndarray（模块级版本，供拖拽/剪贴板共用）。"""
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    bpl = image.bytesPerLine()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    buf = np.frombuffer(ptr, np.uint8)
    arr = buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)


class ImportDialog(QDialog):
    """返回 self.fen / self.my_color / self.flip_view（True=截图为我方在上）。"""

    def __init__(self, parent=None, model_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("导入对局截图")
        self.resize(1150, 760)
        self.setAcceptDrops(True)
        self.model_name = model_name
        self.img_bgr: Optional[np.ndarray] = None
        self.corners: Optional[list] = None
        self.click_points: List[QPointF] = []
        self.warped: Optional[np.ndarray] = None
        self.grid: Optional[list] = None
        self.fen: Optional[str] = None
        self.my_color: Optional[str] = None
        self.auto_ok = False
        self.screenshot_red_top = False
        self.board_box = None
        self._last_result = None
        self._n_piece = self._n_unknown = 0

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.image_label = QLabel("把截图拖到这里\n\n"
                                  "（也可以点下方「打开图片」或按 Ctrl+V 粘贴）")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 560)
        self.image_label.setStyleSheet("border: 1px solid #888; background: #333;")
        self.image_label.mousePressEvent = self._on_image_click
        left.addWidget(self.image_label, 1)

        btn_bar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片...")
        self.btn_paste = QPushButton("粘贴截图 (Ctrl+V)")
        self.btn_auto = QPushButton("自动识别棋盘")
        self.btn_manual = QPushButton("手动校准四角")
        self.btn_redo = QPushButton("重新识别棋子")
        for b in (self.btn_open, self.btn_paste, self.btn_auto,
                  self.btn_manual, self.btn_redo):
            btn_bar.addWidget(b)
        left.addLayout(btn_bar)
        self.hint_label = QLabel("步骤：载入截图 → 定位棋盘 → 校对识别结果 → 确认")
        left.addWidget(self.hint_label)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(QLabel("复刻棋盘（识别结果；用下方调色板点选棋子后点击棋盘修正）:"))
        self.editor = BoardEditorWidget()
        right.addWidget(self.editor, 1)

        self.palette = PiecePalette()
        self.palette.tool_changed.connect(self.editor.set_tool)
        right.addWidget(self.palette)

        preset_bar = QHBoxLayout()
        self.btn_clear = QPushButton("清空棋盘")
        self.btn_initial = QPushButton("设为开局")
        preset_bar.addWidget(self.btn_clear)
        preset_bar.addWidget(self.btn_initial)
        preset_bar.addStretch(1)
        right.addLayout(preset_bar)

        color_bar = QHBoxLayout()
        color_bar.addWidget(QLabel("我方执子:"))
        self.rb_red = QRadioButton("红方")
        self.rb_black = QRadioButton("黑方")
        self.rb_red.setChecked(True)
        color_bar.addWidget(self.rb_red)
        color_bar.addWidget(self.rb_black)
        right.addLayout(color_bar)

        turn_bar = QHBoxLayout()
        turn_bar.addWidget(QLabel("当前轮到:"))
        self.rb_turn_red = QRadioButton("红方")
        self.rb_turn_black = QRadioButton("黑方")
        self.rb_turn_red.setChecked(True)
        turn_bar.addWidget(self.rb_turn_red)
        turn_bar.addWidget(self.rb_turn_black)
        right.addLayout(turn_bar)

        self.chk_red_top = QCheckBox("截图红方在上（执黑时勾选，棋盘会随之翻转）")
        self.chk_red_top.toggled.connect(self._on_red_top_toggled)
        right.addWidget(self.chk_red_top)

        self.lbl_conf = QLabel("")
        right.addWidget(self.lbl_conf)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        root.addLayout(right, 2)

        self.btn_open.clicked.connect(self._open_file)
        self.btn_paste.clicked.connect(self._paste_clipboard)
        self.btn_auto.clicked.connect(self._auto_detect)
        self.btn_manual.clicked.connect(self._manual_calibrate)
        self.btn_redo.clicked.connect(self._run_recognition)
        self.btn_clear.clicked.connect(self.editor.clear)
        self.btn_initial.clicked.connect(self.editor.set_initial)

    # ---------------------------------------------------------------- 载入
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择对局截图", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
        if not path:
            return
        if not self.load_from_path(path):
            QMessageBox.warning(self, "导入", "无法读取该图片")

    def _paste_clipboard(self):
        """粘贴截图：支持剪贴板里的图片数据，也支持复制的图片文件路径。"""
        from PyQt6.QtWidgets import QApplication
        mime = QApplication.clipboard().mimeData()
        img = image_from_mime(mime)
        if img is None:
            QMessageBox.information(
                self, "导入",
                "剪贴板中没有可用的图片。\n"
                "可以用截图工具截图后按 Ctrl+V，或直接把图片文件拖进本窗口。")
            return
        self._set_image(img)

    def _qimage_to_bgr(self, image: QImage) -> np.ndarray:
        return _qimage_to_bgr_static(image)

    # ---------------------------------------------------------------- 拖拽
    def dragEnterEvent(self, event):
        """拖入图片文件或图片数据时接受。"""
        md = event.mimeData()
        if md.hasImage() or md.hasUrls() or md.hasText():
            event.acceptProposedAction()
            self.hint_label.setText("松开鼠标即可导入该截图")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        img = image_from_mime(event.mimeData())
        if img is None:
            self.hint_label.setText("拖入的内容不是可识别的图片，请拖入截图文件")
            event.ignore()
            return
        event.acceptProposedAction()
        self._set_image(img)

    def load_from_path(self, path: str) -> bool:
        """从文件路径载入截图（供拖拽/外部调用）。"""
        img = load_image_file(path)
        if img is None:
            return False
        self._set_image(img)
        return True

    def _set_image(self, img: np.ndarray):
        self.img_bgr = img
        self.corners = None
        self.click_points = []
        self.warped = None
        self.grid = None
        self.board_box = None
        self._refresh_image()
        self._run_recognition()

    # ---------------------------------------------------------------- 识别
    def _model_path(self):
        """按设置解析检测模型路径；空表示自动选择。"""
        if not self.model_name:
            return None
        import os
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models")
        p = os.path.join(base, self.model_name)
        return p if os.path.exists(p) else None

    def _run_recognition(self):
        """优先 YOLO 检测模型；不可用或效果差时回退几何方案。"""
        if self.img_bgr is None:
            return
        self.board_box = None
        self.corners = None
        res = recognize_screenshot(self.img_bgr, model_path=self._model_path())
        self._last_result = res
        self._n_piece = res.n_pieces
        self._n_unknown = 0
        self.screenshot_red_top = res.red_at_top
        self.grid = res.grid
        self.editor.set_flip_display(res.red_at_top)
        self.editor.set_grid(res.grid)
        self.editor.set_marks([(r, c) for (r, c, *_rest) in res.corrections])
        self.chk_red_top.setChecked(res.red_at_top)
        if res.source == "yolo":
            self.board_box = res.board_box
        self._update_quality_hint(res)
        self._refresh_image()

    def _auto_detect(self):
        """「重新识别」按钮：重跑识别流水线。"""
        self._run_recognition()

    def _manual_calibrate(self):
        """回退到手动四角校准 + 几何识别（YOLO 不可用时的备用手段）。"""
        if self.img_bgr is None:
            return
        self.corners = None
        self.board_box = None
        self.click_points = []
        self._manual_mode = True
        self.hint_label.setText("请依次点击棋盘 " +
                                "、".join(_CORNER_NAMES))
        self._refresh_image()

    def _on_image_click(self, event):
        if self.img_bgr is None:
            return
        if self.corners is not None and self.grid is not None and not self._manual_mode:
            return
        if len(self.click_points) >= 4:
            return
        # 显示坐标 -> 原图坐标
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        lw, lh = self.image_label.width(), self.image_label.height()
        scale = min(lw / self.img_bgr.shape[1], lh / self.img_bgr.shape[0])
        ox = (lw - self.img_bgr.shape[1] * scale) / 2
        oy = (lh - self.img_bgr.shape[0] * scale) / 2
        pos = event.position()
        x = (pos.x() - ox) / scale
        y = (pos.y() - oy) / scale
        self.click_points.append(QPointF(x, y))
        if len(self.click_points) == 4:
            corners = [(p.x(), p.y()) for p in self.click_points]
            self._apply_corners(corners)
            self.hint_label.setText("手动校准完成，已识别棋子。请校对后确认。")
        else:
            self.hint_label.setText("请点击 " + _CORNER_NAMES[len(self.click_points)])
        self._refresh_image()

    _manual_mode = False

    def _apply_corners(self, corners):
        """手动校准得到的四角 -> 透视变换 -> 几何识别（模板匹配）。"""
        self._manual_mode = bool(self.click_points)
        self.corners = corners
        self.board_box = None
        locator = BoardLocator(corners)
        self.warped = locator.warp(self.img_bgr)
        raw = self._recognize_plain()
        red_top = sum(1 for r in range(5) for c in range(9)
                      if raw[r][c] and raw[r][c].isupper())
        red_bottom = sum(1 for r in range(5, 10) for c in range(9)
                         if raw[r][c] and raw[r][c].isupper())
        self.screenshot_red_top = red_top > red_bottom
        grid = raw
        if self.screenshot_red_top:
            grid = [row[:] for row in raw][::-1]
            for row in grid:
                row.reverse()
        self.grid = grid
        self.editor.set_flip_display(self.screenshot_red_top)
        self.editor.set_grid(grid)
        self.chk_red_top.setChecked(self.screenshot_red_top)
        self._update_quality_hint()
        self._refresh_image()

    def _recognize_plain(self):
        """按 warped 图像的原始朝向逐格做模板匹配（不做方向判断）。"""
        from recognition.classifier import classify_point
        grid = [[None] * 9 for _ in range(10)]
        self._n_piece = self._n_unknown = 0
        for r in range(10):
            for c in range(9):
                state, letter = classify_point(self.warped, r, c)
                if state == "piece":
                    grid[r][c] = letter
                    self._n_piece += 1
                elif state == "unknown":
                    self._n_unknown += 1
        return grid

    def _update_quality_hint(self, res=None):
        res = res if res is not None else getattr(self, "_last_result", None)
        source = getattr(res, "source", "geometric") if res else "geometric"
        warning = getattr(res, "warning", None) if res else None
        error = getattr(res, "error", None) if res else None
        n_piece = self._n_piece

        if error and n_piece == 0:
            self.lbl_conf.setText(
                f"⚠ {error}\n可点「手动校准四角」重试，或直接在右侧调色板手动摆放。")
            self.lbl_conf.setStyleSheet("color: #b00;")
        elif n_piece == 0:
            self.lbl_conf.setText(
                "⚠ 未识别出任何棋子。请直接用下方调色板手动摆放"
                "（或点「设为开局」后微调）。")
            self.lbl_conf.setStyleSheet("color: #b00;")
        elif n_piece < 12:
            kind = "检测模型" if source == "yolo" else "备用识别"
            self.lbl_conf.setText(
                f"⚠ {kind}只找到 {n_piece} 个棋子，很可能不准。"
                f"请对照左侧原图用调色板修正，或改用「手动摆局」。")
            self.lbl_conf.setStyleSheet("color: #b60;")
        else:
            kind = "检测模型" if source == "yolo" else "备用识别（准确率较低）"
            n_corr = len(getattr(res, "corrections", []) or [])
            n_cls = getattr(res, "skin_classes", 0) or 0
            msg = (f"{kind}识别出 {n_piece} 个棋子"
                   + (f"，{self._n_unknown} 格不确定" if self._n_unknown else "")
                   + "。请对照左侧原图核对，错处用调色板点选后点击棋盘修正。")
            tail = []
            if n_corr:
                tail.append(f"皮肤模板已自动修正 {n_corr} 处（棋盘上橙圈标注，请复核）")
            if n_cls:
                tail.append(f"皮肤模板已校准 {n_cls} 类；点 OK 后会按本局继续更新模板")
            if tail:
                msg += "\n" + "；".join(tail) + "。"
            if warning:
                msg = "⚠ " + warning + "\n" + msg
            self.lbl_conf.setText(msg)
            self.lbl_conf.setStyleSheet(
                "color: #060;" if source == "yolo" and not warning else "color: #b60;")

    def _on_red_top_toggled(self, checked):
        """手动纠正截图方向（红方是否在截图上方）。"""
        if self.warped is None or self.grid is None:
            return
        if checked == getattr(self, "screenshot_red_top", False):
            return
        self.screenshot_red_top = checked
        grid = [row[:] for row in self.grid][::-1]
        for row in grid:
            row.reverse()
        self.grid = grid
        self.editor.set_flip_display(checked)
        self.editor.set_grid(grid)

    # ---------------------------------------------------------------- 绘制
    def _refresh_image(self):
        if self.img_bgr is None:
            return
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 210, 0), 3)
        painter.setPen(pen)
        if self.board_box:
            # YOLO 检测到的棋盘框 + 90 个交点
            bx1, by1, bx2, by2 = self.board_box
            gw, gh = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
            painter.drawRect(QRectF(bx1, by1, bx2 - bx1, by2 - by1))
            for r in range(10):
                for c in range(9):
                    painter.drawEllipse(
                        QPointF(bx1 + c * gw, by1 + r * gh), 2.5, 2.5)
        if self.corners:
            # 画出四角与网格
            for (x, y) in self.corners:
                painter.drawEllipse(QPointF(x, y), 6, 6)
            locator = BoardLocator(self.corners)
            for r in range(10):
                for c in range(9):
                    x, y = locator.canvas_to_orig(r, c)
                    painter.drawEllipse(QPointF(x, y), 1.5, 1.5)
        for i, p in enumerate(self.click_points):
            painter.setPen(QPen(QColor(0, 255, 120), 3))
            painter.drawEllipse(p, 7, 7)
            painter.drawText(p + QPointF(10, -10), _CORNER_NAMES[i])
        painter.end()
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    # ---------------------------------------------------------------- 确认
    def _accept(self):
        grid = self.editor.get_grid()
        if not any(p for row in grid for p in row):
            QMessageBox.information(self, "导入", "棋盘是空的，请先摆放棋子。")
            return
        my = RED if self.rb_red.isChecked() else BLACK
        stm = RED if self.rb_turn_red.isChecked() else BLACK
        # 编辑器内部数据始终为 FEN 顺序（row0 = 黑方底线），
        # 显示方向的翻转只影响呈现，不影响这里。
        fen = grid_to_fen(grid, side_to_move=stm)
        from core.board import Board
        board = Board(fen)
        problem = board.validate_position()
        if problem:
            QMessageBox.warning(
                self, "导入",
                f"局面不合法：{problem}。\n"
                f"请在右侧棋盘上用调色板修正（右键可快速擦除），再点 OK。")
            return
        self.fen = fen
        self.my_color = my
        self._harvest_skin_templates(grid)
        self.accept()

    def _harvest_skin_templates(self, grid):
        """按本局（含人工修正）采集皮肤模板，供后续识别二次校验。

        人工修正是天然的准确标签：用得越多、修得越多，模板越贴近
        你正在用的棋子皮肤。
        """
        res = getattr(self, "_last_result", None)
        if (res is None or res.source != "yolo" or not res.detections
                or not res.board_box):
            return
        try:
            from recognition import skin
            path = skin.default_skin_path()
            existing = skin.load_templates(path) or {}
            merged = skin.harvest(self.img_bgr, res.detections, res.board_box,
                                  res.red_at_top, grid, existing)
            skin.save_templates(path, merged)
            self.lbl_conf.setText(
                self.lbl_conf.text()
                + f"\n✓ 已按本局更新皮肤模板（共 {len(merged)} 类），"
                  f"之后的识别会自动用它们校验。")
        except Exception as e:
            self.lbl_conf.setText(self.lbl_conf.text() + f"\n（皮肤模板更新失败: {e}）")
