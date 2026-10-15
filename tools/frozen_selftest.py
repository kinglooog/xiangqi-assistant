# -*- coding: utf-8 -*-
"""打包产物自检脚本：在冻结环境里验证资源、引擎、识别模型。

用法（在打包目录内）：
    象棋接管助手.exe --selftest        （不可行：windowed exe 无控制台输出）

因此用 _internal 里的 Python 环境运行本脚本：把本脚本放到打包目录，
它会模拟 PyInstaller 的 _MEIPASS 资源布局做检查。
"""
import os
import sys


def main():
    # 模块从源码目录导入（打包机上有源码）：从脚本位置向上找含 recognition/ 的目录。
    # 资源（models/、engine/bin）必须从打包产物里加载——这才是要验证的东西。
    here = os.path.dirname(os.path.abspath(__file__))
    src_root = None
    probe = here
    for _ in range(4):
        if os.path.isdir(os.path.join(probe, "recognition")):
            src_root = probe
            break
        probe = os.path.dirname(probe)
    if src_root:
        sys.path.insert(0, src_root)
    candidates = [here,
                  os.path.join(here, "_internal"),
                  os.path.join(os.path.dirname(here), "_internal")]
    root = next((c for c in candidates if os.path.isdir(os.path.join(c, "models"))), None)
    if root is None:
        print("未找到资源目录（models/）")
        return 1
    print("资源根:", root)

    # 1) 资源齐全
    models = [f for f in os.listdir(os.path.join(root, "models")) if f.endswith(".onnx")]
    ebin = os.path.join(root, "engine", "bin")
    engine_exe = next((f for f in os.listdir(ebin) if f.endswith(".exe")), None)
    nnue = os.path.isfile(os.path.join(ebin, "pikafish.nnue"))
    print(f"模型 {len(models)} 个: {models}")
    print("引擎:", engine_exe, "| NNUE:", nnue)
    assert models and engine_exe and nnue

    # 2) 识别模型可加载并推理
    import cv2
    import numpy as np
    from recognition.yolo import YoloDetector
    from recognition.yolo_board import board_from_detections
    det = YoloDetector(os.path.join(root, "models", "中模型.onnx"), conf=0.4)
    # 用程序生成的棋盘测试（不依赖 samples）
    from core.board import Board
    from recognition.synth import render_board
    img = render_board([[p for p in row] for row in Board().grid])
    dets = det.detect(img)
    res = board_from_detections(dets)
    print(f"识别自检: 检出 {len(dets)} 个目标, 棋子 {res.n_pieces}, 冲突 {res.n_ambiguous}")
    assert res.n_pieces >= 30 and res.n_ambiguous == 0

    # 3) 引擎可启动并给出着法
    from engine.uci import UciEngine
    eng = UciEngine(os.path.join(ebin, engine_exe), os.path.join(ebin, "pikafish.nnue"))
    eng.start()
    eng.set_engine_options(threads=4, hash_mb=256)
    got = {}
    eng.on_bestmove = lambda m: got.setdefault("m", m)
    eng.position("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
    eng.go(depth=12)
    import time
    t0 = time.time()
    while "m" not in got and time.time() - t0 < 60:
        time.sleep(0.05)
    eng.quit()
    print("引擎自检: bestmove =", got.get("m"))
    assert got.get("m")
    print("\n打包自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
