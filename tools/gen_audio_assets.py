# -*- coding: utf-8 -*-
"""生成 assets/sounds/ 下的音频素材（一次性开发工具，产物随包分发）。

素材来源：
- 落子/吃子/抬子：lichess.org 开源仓库的木质棋子实录音（Move/Capture/Select），
  依据 GPL 项目的素材使用惯例附来源署名。
- 绝杀/将军/困毙人声：微软 Edge 神经语音（zh-CN-YunjianNeural）预生成，
  自然度远高于系统 SAPI；生成后随包分发，运行时无需联网。
- 绝杀垫底低音轰鸣：落子音降调处理。

用法：
    .venv/Scripts/python tools/gen_audio_assets.py
"""
import asyncio
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SOUND_DIR = os.path.join(ROOT, "assets", "sounds")
SR = 44100

LICHESS_BASE = ("https://raw.githubusercontent.com/lichess-org/lila/"
                "master/public/sound/standard")

# 目标音频: (输出名, 来源mp3, 峰值音量, 重采样倍率(>1 音调升高), 裁剪秒数)
PLAN = [
    ("pickup.wav",  "Select.mp3",  0.50, 1.0,  None),
    ("drop.wav",    "Move.mp3",    0.80, 1.0,  None),
    ("capture.wav", "Capture.mp3", 0.85, 1.0,  None),
    # 将军警示：落子音升调变亮，与普通落子区分
    ("check.wav",   "Move.mp3",    0.70, 1.5,  0.22),
    # 绝杀垫底：落子音降调 -> 低沉轰鸣
    ("mate.wav",    "Move.mp3",    0.95, 0.42, 1.4),
]

VOICES = [
    # (输出, 文本, 语气)
    ("voice_js.wav", "绝杀", "+18%"),
    ("voice_jj.wav", "将军", "+12%"),
    ("voice_kb.wav", "困毙", "+5%"),
]
VOICE = "zh-CN-YunjianNeural"


def download_missing():
    need = {src for _, src, *_ in PLAN}
    for name in list(need):
        p = os.path.join(SOUND_DIR, name)
        ok = os.path.exists(p) and os.path.getsize(p) > 500
        if not ok:
            import urllib.request
            url = f"{LICHESS_BASE}/{name}"
            print("下载", url)
            req = urllib.request.Request(url, headers={"User-Agent": "xq"})
            data = urllib.request.urlopen(req, timeout=60).read()
            with open(p, "wb") as f:
                f.write(data)


def decode_mp3(path: str) -> np.ndarray:
    """mp3 -> float32 单声道（-1..1），重采样到 SR。"""
    import miniaudio
    with open(path, "rb") as f:
        data = f.read()
    dec = miniaudio.decode(data)
    samples = np.asarray(dec.samples, dtype=np.float32)
    if samples.dtype.kind not in "fiu":
        samples = samples.astype(np.float32)
    if np.abs(samples).max() > 1.5:            # int16
        samples = samples / 32768.0
    if dec.nchannels > 1:
        samples = samples.reshape(-1, dec.nchannels).mean(axis=1)
    if dec.sample_rate != SR:
        n = int(len(samples) * SR / dec.sample_rate)
        samples = cv2.resize(samples.reshape(1, -1), (n, 1),
                             interpolation=cv2.INTER_LINEAR).ravel()
    return samples


def resample_pitch(samples: np.ndarray, factor: float) -> np.ndarray:
    """简单重采样变调（factor>1 音调升高、时长缩短）。"""
    n = max(1, int(len(samples) / factor))
    return cv2.resize(samples.reshape(1, -1), (n, 1),
                      interpolation=cv2.INTER_LINEAR).ravel()


def normalize(samples: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.abs(samples).max())
    if m < 1e-6:
        return samples
    return samples * (peak / m)


def save_wav(path: str, samples: np.ndarray) -> None:
    import wave
    pcm = (np.clip(samples, -1, 1) * 32000).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


async def gen_voice(text: str, out: str, rate: str) -> bool:
    """Edge TTS 生成人声（输出实为 MP3），解码转 WAV 并归一化。"""
    import edge_tts
    mp3 = os.path.join(SOUND_DIR, out + ".src.mp3")
    try:
        c = edge_tts.Communicate(text, VOICE, rate=rate)
        await c.save(mp3)
        if os.path.getsize(mp3) < 500:
            return False
        s = normalize(decode_mp3(mp3), 0.9)
        save_wav(os.path.join(SOUND_DIR, out), s)
        os.remove(mp3)
        return True
    except Exception as e:
        print("  语音生成失败", text, e)
        return False


def main():
    os.makedirs(SOUND_DIR, exist_ok=True)
    download_missing()

    for out, src, peak, factor, clip in PLAN:
        sp = os.path.join(SOUND_DIR, src)
        if not os.path.exists(sp) or os.path.getsize(sp) < 500:
            print(f"  跳过 {out}（源 {src} 缺失或无效）")
            continue
        s = decode_mp3(sp)
        if factor != 1.0:
            s = resample_pitch(s, factor)
        if clip:
            s = s[:int(clip * SR)]
        s = normalize(s, peak)
        save_wav(os.path.join(SOUND_DIR, out), s)
        print(f"  {out:14s} {len(s)/SR:.2f}s  peak {peak}")

    print("生成语音（Edge TTS）...")
    for out, text, rate in VOICES:
        ok = asyncio.run(gen_voice(text, out, rate))
        print(f"  {out:14s} {'OK' if ok else '失败'}")
    print("完成。产物在", SOUND_DIR)


if __name__ == "__main__":
    main()
