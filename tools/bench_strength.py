# -*- coding: utf-8 -*-
"""速度/棋力等价基准：对比「对战模式默认（3s + 深度封顶）」与「纯深度搜索」。

目的：验证提速没有降低棋力——两者在测试局面上给出相同着法，
而对战模式在复杂局面快 2-3 倍。

每次搜索前清空置换表，保证两个条件可比（否则第二次搜索会复用
第一次填满的置换表而被不公平地加速）。

用法：
    .venv/Scripts/python tools/bench_strength.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.board import iccs_to_move
from core.game import Game
from engine.uci import UciEngine, find_engine_exe, find_nnue_file

BATTLE_MOVETIME_MS = 3000   # 对战模式默认每步用时
DEPTH_CAP = 24              # 深度上限（两种方式相同）


def build_positions():
    """构造若干局面：开局 + 数条真实走法序列后的中局。"""
    positions = []

    g = Game()
    positions.append(("开局", "红先", g.board.to_fen()))
    for u in ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "b0c2", "g6g5"]:
        g.play(iccs_to_move(u))
    positions.append(("中局1", "红先", g.board.to_fen()))
    for u in ["c3c4", "c6c5", "g3g4", "c5c4", "b2b6", "a9b9"]:
        g.play(iccs_to_move(u))
    positions.append(("中局2", "红先", g.board.to_fen()))

    g2 = Game()
    for u in ["h2e2", "h9g7", "b2e2", "b9c7", "b0c2", "a9b9",
              "a0b0", "b7b3", "c3c4", "h7h3"]:
        g2.play(iccs_to_move(u))
    positions.append(("中局3", "红先", g2.board.to_fen()))

    g3 = Game()
    for u in ["c3c4", "g6g5", "b0c2", "b9c7", "a0b0", "a9b9", "h0g2", "h9g7"]:
        g3.play(iccs_to_move(u))
    positions.append(("中局4", "红先", g3.board.to_fen()))
    return positions


def main():
    exe = find_engine_exe()
    if exe is None:
        print("未找到引擎，请先运行 tools/download_engine.py")
        return 1
    eng = UciEngine(exe, find_nnue_file())
    eng.start()
    threads = max(1, (os.cpu_count() or 4) - 1)
    eng.set_engine_options(threads=threads, hash_mb=1024)
    print(f"引擎: {eng.engine_id}  线程: {threads}\n")

    got = {}
    eng.on_bestmove = lambda m: got.setdefault("best", m)

    def search(fen, movetime=None):
        """每次搜索前清空置换表，保证条件可比。"""
        got.clear()
        eng.new_game()
        eng._send("setoption name Clear Hash")
        eng.position(fen, [])
        t0 = time.time()
        if movetime:
            eng.go(depth=DEPTH_CAP, movetime_ms=movetime)
        else:
            eng.go(depth=DEPTH_CAP)
        while "best" not in got and time.time() - t0 < 300:
            time.sleep(0.05)
        return got.get("best"), time.time() - t0

    print(f"{'局面':8s} {'对战模式(3s)':14s} {'用时':7s} "
          f"{'纯depth24':14s} {'用时':7s} 结果")
    print("-" * 70)
    same = 0
    positions = build_positions()
    for name, _side, fen in positions:
        m1, t1 = search(fen, movetime=BATTLE_MOVETIME_MS)
        m2, t2 = search(fen)
        ok = m1 == m2
        same += ok
        print(f"{name:8s} {str(m1):14s} {t1:6.1f}s {str(m2):14s} {t2:6.1f}s "
              f"{'一致' if ok else '不同'}")
    print("-" * 70)
    print(f"着法一致率: {same}/{len(positions)}")
    print("说明：一致即表示对战模式的限时不影响棋力（深度上限相同，")
    print("      时间到只是提前结束，着法与算满深度时相同）。")
    eng.quit()
    return 0 if same == len(positions) else 1


if __name__ == "__main__":
    sys.exit(main())
