# -*- coding: utf-8 -*-
"""音效管理：程序化合成音效 + Windows TTS 语音播报。

设计约束：
- 零外部音频资源：所有音效用 numpy 合成 WAV 字节，启动时生成（<50ms），
  无需随包分发音频文件，也无需新增依赖。
- 播放用 winsound（Windows 自带，SND_ASYNC 不阻塞界面）。
- "绝杀"语音用 Windows 自带 SAPI TTS（经 PowerShell 调用，后台线程执行），
  中文系统自带中文语音；不可用时静默跳过，只保留音效。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import wave
from typing import Dict, Optional

import numpy as np

SR = 22050


def _wav_bytes(samples: np.ndarray) -> bytes:
    """float 采样（-1..1）-> 16bit 单声道 WAV 字节。"""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _env_exp(n: int, decay: float) -> np.ndarray:
    """指数衰减包络。"""
    t = np.arange(n) / SR
    return np.exp(-t * decay)


def _tone(freq: float, dur: float, decay: float, amp: float,
          sweep_to: Optional[float] = None) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    if sweep_to is None:
        wave_ = np.sin(2 * np.pi * freq * t)
    else:                                     # 频率线性下滑
        f = np.linspace(freq, sweep_to, n)
        wave_ = np.sin(2 * np.pi * f * t)
    return amp * wave_ * _env_exp(n, decay)


def _thump(freq: float = 165.0, dur: float = 0.13, amp: float = 0.62) -> np.ndarray:
    """木质敲击：低频正弦急衰减 + 短噪声瞬态。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * freq * t * (1 + 0.6 * np.exp(-t * 60)))
    noise = np.random.default_rng(7).standard_normal(n)
    noise = np.convolve(noise, np.ones(24) / 24, mode="same")   # 低通
    attack = np.exp(-t * 90)
    out = amp * (0.8 * body + 0.5 * noise) * _env_exp(n, 34) + 0.10 * noise * attack
    return out


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(dur * SR))


# ------------------------------------------------------------------ 音色定义
def synth_sound(name: str) -> bytes:
    """合成指定音效并返回 WAV 字节。"""
    if name == "pickup":
        # 抬子：短促高频轻响
        s = _tone(1900, 0.05, 70, 0.30, sweep_to=1350)
    elif name == "drop":
        # 落子：木质"啪"
        s = _thump(175, 0.13, 0.62)
    elif name == "capture":
        # 吃子：两声更沉的敲击
        s = np.concatenate([_thump(140, 0.12, 0.66), _silence(0.045),
                            _thump(120, 0.16, 0.72)])
    elif name == "check":
        # 将军：上行双音警示
        s = np.concatenate([_tone(640, 0.10, 22, 0.34),
                            _tone(854, 0.14, 18, 0.38)])
    elif name == "mate":
        # 绝杀：锣声（多泛音长衰减）
        n = int(1.7 * SR)
        t = np.arange(n) / SR
        s = np.zeros(n)
        for f, a, d in ((150, 0.50, 3.0), (401, 0.28, 4.5),
                        (663, 0.18, 6.0), (942, 0.10, 8.0)):
            s += a * np.sin(2 * np.pi * f * t * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))) \
                 * _env_exp(n, d)
        attack = np.random.default_rng(3).standard_normal(n)
        attack = np.convolve(attack, np.ones(16) / 16, mode="same")
        s += 0.22 * attack * np.exp(-t * 55)
        s *= 0.55
    else:
        s = _tone(880, 0.08, 30, 0.3)
    return _wav_bytes(np.asarray(s, dtype=np.float64))


class SoundManager:
    """音效播放 + 语音播报。所有方法线程安全、失败静默。

    优先播放 assets/sounds/ 下的真实素材（木质棋子实录音 + Edge 神经语音
    预生成的"绝杀/将军/困毙"人声，随包分发、运行时无需联网）；
    素材缺失时回退到程序合成音效 / 系统 TTS。
    """

    NAMES = ("pickup", "drop", "capture", "check", "mate")
    FILES = {"pickup": "pickup.wav", "drop": "drop.wav",
             "capture": "capture.wav", "check": "check.wav",
             "mate": "mate.wav"}
    VOICE_FILES = {"绝杀": "voice_js.wav", "将军": "voice_jj.wav",
                   "困毙": "voice_kb.wav"}

    def __init__(self, enabled: bool = True, voice: bool = True):
        self.enabled = enabled
        self.voice = voice
        self._cache: Dict[str, bytes] = {}
        self._winsound = None
        try:
            if sys.platform == "win32":
                import winsound
                self._winsound = winsound
        except Exception:
            self._winsound = None
        self.preload()

    @staticmethod
    def _sound_dir() -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "sounds")

    def _asset(self, filename: str) -> Optional[str]:
        if not filename:
            return None
        p = os.path.join(self._sound_dir(), filename)
        return p if os.path.isfile(p) else None

    def preload(self) -> None:
        """预合成兜底音效（仅当真实素材缺失时才会用到）。"""
        if self._cache:
            return
        for name in self.NAMES:
            try:
                self._cache[name] = synth_sound(name)
            except Exception:
                pass

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_voice(self, voice: bool) -> None:
        self.voice = voice

    # ---------------------------------------------------------------- 播放
    def play(self, name: str) -> None:
        """异步播放音效；未启用或平台不支持时静默。"""
        if not self.enabled or self._winsound is None:
            return
        path = self._asset(self.FILES.get(name, ""))
        try:
            if path:
                flags = self._winsound.SND_ASYNC | self._winsound.SND_FILENAME
                self._winsound.PlaySound(path, flags)
            else:
                data = self._cache.get(name)
                if not data:
                    return
                flags = self._winsound.SND_ASYNC | self._winsound.SND_MEMORY
                self._winsound.PlaySound(data, flags)
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """播报语音：优先预生成的自然语音文件，否则退回系统 TTS。"""
        if not self.voice or sys.platform != "win32":
            return
        vf = self._asset(self.VOICE_FILES.get(text, ""))
        if vf is not None and self._winsound is not None:
            try:
                flags = self._winsound.SND_ASYNC | self._winsound.SND_FILENAME
                self._winsound.PlaySound(vf, flags)
                return
            except Exception:
                pass

        def _run():
            try:
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                     "-Command",
                     "Add-Type -AssemblyName System.Speech;"
                     "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                     f"$s.Speak('{text}')"],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()
