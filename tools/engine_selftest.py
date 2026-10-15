# -*- coding: utf-8 -*-
"""引擎命令行自测：验证 Pikafish 可用并输出开局最佳着法。

用法：
    .venv/Scripts/python tools/engine_selftest.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.board import Board, move_to_chinese, iccs_to_move
from engine.uci import UciEngine, find_engine_exe, find_nnue_file

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def main():
    exe = find_engine_exe()
    if not exe:
        print("未找到引擎可执行文件，请先运行 tools/download_engine.py")
        return 1
    nnue = find_nnue_file()
    print(f"引擎: {exe}")
    print(f"权重: {nnue}")

    eng = UciEngine(exe, nnue)
    eng.start()
    print(f"标识: {eng.engine_id}")
    eng.set_engine_options(threads=max(1, os.cpu_count() - 1), hash_mb=256, multipv=3)
    eng.new_game()
    eng.position(INITIAL_FEN)

    best = {}
    done = threading_done = [False]

    def on_info(info):
        if info.get("multipv") == 1:
            print(f"  depth {info.get('depth')} score {info.get('score_str')} "
                  f"nps {info.get('nps', 0) // 1000}k pv {info.get('pv', [])[:5]}")

    def on_bestmove(move):
        best["move"] = move
        done[0] = True

    eng.on_info = on_info
    eng.on_bestmove = on_bestmove
    eng.go(depth=20)

    deadline = time.time() + 120
    while not done[0] and time.time() < deadline:
        time.sleep(0.1)

    move = best.get("move")
    if move:
        try:
            cn = move_to_chinese(Board(INITIAL_FEN), iccs_to_move(move))
        except Exception:
            cn = "?"
        print(f"最佳着法: {move}（{cn}）")
    else:
        print("引擎未返回着法")
    eng.quit()
    return 0 if move else 1


if __name__ == "__main__":
    sys.exit(main())
