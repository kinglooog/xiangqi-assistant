# -*- coding: utf-8 -*-
"""pytest 公共配置。

GUI 相关测试需要 QApplication 存在才能创建 QWidget。
用离屏平台避免弹出窗口，且整个测试会话共用一个实例。
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """屏蔽所有模态弹窗。

    QMessageBox.information/warning/critical 是模态的，测试环境没人点确认
    会永久阻塞（曾导致测试挂死）。这里替换为记录调用的空实现，
    测试若需要可断言 recorded 内容。
    """
    from PyQt6.QtWidgets import QMessageBox
    recorded = []

    def _fake(kind):
        def _f(*args, **kwargs):
            recorded.append((kind, args[1] if len(args) > 1 else "",))
            return QMessageBox.StandardButton.Ok
        return _f

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(_fake(kind)))
    yield recorded
