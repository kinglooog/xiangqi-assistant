# -*- coding: utf-8 -*-
"""截图导入路径的回归测试：剪贴板粘贴、文件拖拽、图片数据拖拽。

背景：此前「粘贴截图」会直接闪退。根因是 QImage -> ndarray 的转换写错：
PyQt6 的 constBits() 返回 sip.voidptr，直接 np.array(ptr) 只得到长度为 1 的
object 数组，随后 reshape 抛 ValueError；该异常发生在按钮槽函数里，
PyQt 默认会让进程直接 abort（表现为闪退）。
"""
import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage

from gui.import_dialog import (_qimage_to_bgr_static, image_from_mime,
                              load_image_file)

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "samples")


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """ndarray(BGR) -> QImage，用于模拟剪贴板/拖拽里的图片数据。"""
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------- 转换正确性
@pytest.mark.parametrize("w,h", [(211, 137), (400, 300), (255, 255), (64, 64)])
def test_qimage_roundtrip_is_pixel_exact(qapp, w, h):
    """QImage -> ndarray 必须逐像素还原（含 bytesPerLine 有填充的宽度）。"""
    rng = np.random.default_rng(1234)
    bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    back = _qimage_to_bgr_static(_bgr_to_qimage(bgr))
    assert back.shape == (h, w, 3)
    assert np.array_equal(back, bgr), "转换后像素不一致"


def test_qimage_roundtrip_real_screenshot(qapp):
    """真实截图往返一致（尺寸非 4 字节对齐，覆盖填充分支）。"""
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    img = cv2.imread(path)
    back = _qimage_to_bgr_static(_bgr_to_qimage(img))
    assert np.array_equal(back, img)


def test_qimage_conversion_no_object_array(qapp):
    """回归保护：不得退回到 np.array(voidptr) 的错误写法。"""
    img = np.full((40, 40, 3), 128, np.uint8)
    out = _qimage_to_bgr_static(_bgr_to_qimage(img))
    assert out.dtype == np.uint8
    assert out.shape == (40, 40, 3)
    assert out.size == 40 * 40 * 3


# ---------------------------------------------------------------- 文件读取
def test_load_image_file_unicode_path(qapp, tmp_path):
    """中文/空格路径也要能读（用 imdecode 而非 cv2.imread）。"""
    p = tmp_path / "中文 目录 截图.png"
    img = np.zeros((30, 50, 3), np.uint8)
    img[..., 1] = 200
    cv2.imencode(".png", img)[1].tofile(str(p))
    got = load_image_file(str(p))
    assert got is not None and got.shape == (30, 50, 3)
    assert np.array_equal(got, img)


def test_load_image_file_missing(qapp):
    assert load_image_file("不存在的文件_xyz.png") is None


# ---------------------------------------------------------------- mime 解析
def test_image_from_mime_file_url(qapp):
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(path)])
    got = image_from_mime(md)
    assert got is not None
    assert np.array_equal(got, cv2.imread(path))


def test_image_from_mime_image_data(qapp):
    img = np.full((60, 80, 3), 77, np.uint8)
    md = QMimeData()
    md.setImageData(_bgr_to_qimage(img))
    got = image_from_mime(md)
    assert got is not None and got.shape == (60, 80, 3)
    assert np.array_equal(got, img)


def test_image_from_mime_text_path(qapp):
    """从资源管理器复制图片文件后粘贴，mime 里可能是文本路径。"""
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    md = QMimeData()
    md.setText(f'"{os.path.abspath(path)}"')
    assert image_from_mime(md) is not None


def test_image_from_mime_empty(qapp):
    assert image_from_mime(QMimeData()) is None
    assert image_from_mime(None) is None


def test_image_from_mime_non_image_file(qapp, tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello", encoding="utf-8")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(p))])
    assert image_from_mime(md) is None


# ---------------------------------------------------------------- 拖拽事件
def _make_drop_events(md):
    enter = QDragEnterEvent(QPoint(50, 50), Qt.DropAction.CopyAction, md,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    drop = QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, md,
                      Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)
    return enter, drop


def test_import_dialog_accepts_file_drop(qapp):
    """把截图文件拖进导入对话框应完成识别。"""
    from gui.import_dialog import ImportDialog
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(path)])
    dlg = ImportDialog()
    enter, drop = _make_drop_events(md)
    dlg.dragEnterEvent(enter)
    assert enter.isAccepted(), "拖入图片文件时应接受"
    dlg.dropEvent(drop)
    assert dlg.img_bgr is not None
    assert np.array_equal(dlg.img_bgr, cv2.imread(path))
    assert drop.isAccepted()


def test_import_dialog_accepts_image_data_drop(qapp):
    from gui.import_dialog import ImportDialog
    md = QMimeData()
    md.setImageData(_bgr_to_qimage(np.full((50, 70, 3), 90, np.uint8)))
    dlg = ImportDialog()
    enter, drop = _make_drop_events(md)
    dlg.dragEnterEvent(enter)
    dlg.dropEvent(drop)
    assert dlg.img_bgr is not None and dlg.img_bgr.shape == (50, 70, 3)


def test_import_dialog_rejects_non_image_drop(qapp):
    from gui.import_dialog import ImportDialog
    md = QMimeData()
    md.setText("just some text")
    dlg = ImportDialog()
    enter, drop = _make_drop_events(md)
    dlg.dragEnterEvent(enter)
    dlg.dropEvent(drop)
    assert dlg.img_bgr is None, "非图片内容不应被载入"


def test_import_dialog_paste_empty_clipboard_no_crash(qapp):
    """空剪贴板粘贴不得抛异常（此前该路径会闪退）。"""
    from PyQt6.QtWidgets import QApplication
    from gui.import_dialog import ImportDialog
    QApplication.clipboard().setMimeData(QMimeData())
    dlg = ImportDialog()
    dlg._paste_clipboard()          # 不应抛异常
    assert dlg.img_bgr is None


def test_import_dialog_paste_image(qapp):
    from PyQt6.QtWidgets import QApplication
    from gui.import_dialog import ImportDialog
    img = np.full((64, 96, 3), 55, np.uint8)
    QApplication.clipboard().setImage(_bgr_to_qimage(img))
    dlg = ImportDialog()
    dlg._paste_clipboard()
    assert dlg.img_bgr is not None and dlg.img_bgr.shape == (64, 96, 3)
