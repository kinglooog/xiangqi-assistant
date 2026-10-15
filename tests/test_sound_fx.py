# -*- coding: utf-8 -*-
"""音效与绝杀/将军特效的回归测试。"""
import os

import pytest
from PyQt6.QtCore import QPoint, QPointF, QEvent, Qt
from PyQt6.QtGui import QMouseEvent

from gui.board_widget import BoardWidget
from gui.sound import SoundManager, synth_sound


# ---------------------------------------------------------------- 音效合成
@pytest.mark.parametrize("name", SoundManager.NAMES)
def test_synth_valid_wav(name):
    data = synth_sound(name)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert len(data) > 44


def test_manager_disabled_does_not_play(qapp, monkeypatch):
    calls = []
    import gui.sound as gs
    monkeypatch.setattr(gs, "winsound_ok", True, raising=False)
    sm = SoundManager(enabled=False)
    monkeypatch.setattr(sm, "_winsound",
                        type("W", (), {"SND_ASYNC": 1, "SND_MEMORY": 2,
                                       "PlaySound": staticmethod(
                                           lambda d, f: calls.append((d, f)))})())
    sm.play("drop")
    assert calls == [], "禁用状态不应播放"


def test_manager_play_async(qapp, monkeypatch):
    """优先播放 assets/sounds 真实素材（SND_FILENAME），素材缺失回退合成音。"""
    calls = []
    sm = SoundManager(enabled=True)
    fake = type("W", (), {"SND_ASYNC": 1, "SND_MEMORY": 2, "SND_FILENAME": 4,
                          "PlaySound": staticmethod(
                              lambda d, f: calls.append((d, f)))})()
    monkeypatch.setattr(sm, "_winsound", fake)
    sm.play("drop")
    assert len(calls) == 1
    data, flags = calls[0]
    assert isinstance(data, str) and data.endswith("drop.wav")
    assert flags == 1 | 4, "素材文件应用 SND_FILENAME|SND_ASYNC"
    # 素材缺失 -> 回退合成音（内存 WAV）
    monkeypatch.setattr(sm, "_asset", lambda fn: None)
    calls.clear()
    sm.play("drop")
    assert calls and calls[0][0][:4] == b"RIFF" and calls[0][1] == 1 | 2


def test_speak_uses_voice_file(qapp, monkeypatch):
    """绝杀/将军/困毙有预生成的自然语音文件，直接播放。"""
    calls = []
    sm = SoundManager(enabled=True, voice=True)
    monkeypatch.setattr(sm, "_winsound",
                        type("W", (), {"SND_ASYNC": 1, "SND_MEMORY": 2,
                                       "SND_FILENAME": 4,
                                       "PlaySound": staticmethod(
                                           lambda d, f: calls.append(d))})())
    for text, fname in (("绝杀", "voice_js.wav"), ("将军", "voice_jj.wav"),
                        ("困毙", "voice_kb.wav")):
        sm.speak(text)
        assert calls[-1].endswith(fname), calls[-1]


def test_speak_fallback_to_tts(qapp, monkeypatch):
    """没有语音文件的文本走系统 TTS 兜底。"""
    import gui.sound as gs
    seen = []

    class FakeProc:
        def __init__(self, *a, **k):
            seen.append(a[0] if a else None)

    monkeypatch.setattr(gs.subprocess, "Popen", FakeProc)
    sm = SoundManager(enabled=True, voice=True)
    sm.speak("未知文本")
    assert seen and any("未知文本" in " ".join(c) for c in seen if c)


# ---------------------------------------------------------------- 特效生命周期
def _click(board, sq):
    p = board.cell_to_view(*sq)
    board.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(p),
                                      Qt.MouseButton.LeftButton,
                                      Qt.MouseButton.LeftButton,
                                      Qt.KeyboardModifier.NoModifier))
    board.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(p),
                                        Qt.MouseButton.LeftButton,
                                        Qt.MouseButton.LeftButton,
                                        Qt.KeyboardModifier.NoModifier))


def test_effect_lifecycle(qapp):
    from core.board import RED
    w = BoardWidget()
    from core.game import Game
    g = Game()
    w.set_game(g)
    w.resize(600, 660)
    w.start_effect("将军", __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(240, 190, 40),
                   king=(0, 4), dur_ms=200)
    assert w.fx is not None and w.fx["text"] == "将军"
    deadline = __import__("time").time() + 3
    import time
    while w.fx is not None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert w.fx is None, "特效应在时长结束后自动清除"


def test_cancel_effect(qapp):
    from core.game import Game
    w = BoardWidget()
    w.set_game(Game())
    w.start_effect("绝杀", __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(220, 40, 40))
    assert w.fx is not None
    w.cancel_effect()
    assert w.fx is None


# ---------------------------------------------------------------- 主窗口集成
def test_mate_triggers_effect_and_voice(qapp, monkeypatch):
    """绝杀一步：触发 绝杀 特效 + mate 音效 + 语音播报。"""
    import time
    from PyQt6.QtGui import QColor
    from gui.main_window import MainWindow
    from core.board import iccs_to_move

    w = MainWindow()
    plays, speaks = [], []
    monkeypatch.setattr(w.sound, "play", lambda n: plays.append(n))
    monkeypatch.setattr(w.sound, "speak", lambda t: speaks.append(t))
    # 红一步绝杀：车 a7-a9 锁死横线（马控 d8、车控将脸线，与已验证杀局同构）
    w.game.reset("3k5/9/R8/2N6/9/9/9/9/5K3/4R4 w - - 0 1")
    w.board.set_game(w.game)
    w.board.set_my_color("w")
    w.board.resize(620, 700)
    assert w._play_move(iccs_to_move("a7a9"), source="user") is None
    assert w.game.result is not None and "将死" in w.game.result
    qapp.processEvents()
    assert w.board.fx is not None and w.board.fx["text"] == "绝杀"
    assert "mate" in plays
    # 语音延迟 800ms 播报
    deadline = time.time() + 3
    while "绝杀" not in speaks and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert "绝杀" in speaks
    # 特效播放结束后自动清除
    deadline = time.time() + 6
    while w.board.fx is not None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.05)
    assert w.board.fx is None
    w.engine.stop() if w.engine else None


def test_check_triggers_check_fx(qapp, monkeypatch):
    """普通将军：将军 特效 + check 音效，不触发绝杀。"""
    from gui.main_window import MainWindow
    from core.board import iccs_to_move

    w = MainWindow()
    plays, speaks = [], []
    monkeypatch.setattr(w.sound, "play", lambda n: plays.append(n))
    monkeypatch.setattr(w.sound, "speak", lambda t: speaks.append(t))
    w.game.reset("3k5/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1")
    w.board.set_game(w.game)
    w.board.resize(620, 700)
    w._play_move(iccs_to_move("e1e9"), source="user")
    qapp.processEvents()
    assert w.game.result is None, "该局面应是普通将军而非绝杀"
    assert w.board.fx is not None and w.board.fx["text"] == "将军"
    assert "check" in plays and "mate" not in plays
    w.board.cancel_effect()
    w.engine.stop() if w.engine else None


def test_drop_and_capture_sounds(qapp, monkeypatch):
    """普通走子播放落子声；吃子播放吃子声。"""
    from gui.main_window import MainWindow
    from core.board import iccs_to_move

    w = MainWindow()
    plays = []
    monkeypatch.setattr(w.sound, "play", lambda n: plays.append(n))
    w.game.reset("3kn4/9/9/9/4P4/9/9/9/4C4/4K4 w - - 0 1")
    w.board.set_game(w.game)
    w.board.resize(620, 700)
    # 炮吃黑马（有炮架？这里直接验证声音路径即可，选一个合法吃子）
    moves = [m for m in w.game.board.legal_moves()
             if w.game.board.piece_at(*m[1]) is not None]
    assert moves, "测试局面应存在吃子着法"
    w._play_move(moves[0], source="user")
    assert "capture" in plays
    plays.clear()
    w._undo()
    quiet = [m for m in w.game.board.legal_moves()
             if w.game.board.piece_at(*m[1]) is None]
    if quiet:
        w._play_move(quiet[0], source="user")
        assert "drop" in plays and "capture" not in plays
    w.engine.stop() if w.engine else None
