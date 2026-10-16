# -*- coding: utf-8 -*-
"""界面默认值与静音按钮：启动默认静音、开局默认"引擎最优"。

用户要求：
- 打开软件默认静音（按钮可随时开声）
- 开局走法默认"引擎最优"，不要一开软件就是双铁滑车
"""
from core.board import RED


def test_mute_button_default_muted_and_toggle(qapp):
    from gui.main_window import MainWindow
    w = MainWindow()
    # 启动默认静音
    assert w.config["sound"] is False and w.config["voice"] is False
    assert w.sound.enabled is False and w.sound.voice is False
    assert w.btn_mute.isChecked() and "静音" in w.btn_mute.text()
    # 点一下开声（音效+播报一并恢复）
    w.btn_mute.click()
    assert w.config["sound"] is True and w.config["voice"] is True
    assert w.sound.enabled is True and w.sound.voice is True
    assert not w.btn_mute.isChecked() and "声音开" in w.btn_mute.text()
    # 再点回静音
    w.btn_mute.click()
    assert w.config["sound"] is False and w.sound.enabled is False
    assert w.btn_mute.isChecked()
    w.engine.stop() if w.engine else None


def test_opening_default_engine_best(qapp):
    """启动时开局走法必须回到"引擎最优"（即使配置里存着双铁滑车）。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    assert w.combo_opening.currentData() == "", "启动应默认引擎最优"
    assert w.config["opening"] == ""
    w.engine.stop() if w.engine else None
