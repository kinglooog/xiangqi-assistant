# -*- coding: utf-8 -*-
"""Pikafish（UCI 象棋引擎）子进程封装。

异步模型：go() 立即返回，搜索过程中通过 on_info 回调推送分析信息，
搜索结束时回调 on_bestmove。回调在引擎读取线程中执行，
GUI 层需自行做线程切换（如 Qt 信号）。

用法示例：
    eng = UciEngine("engine/bin/Pikafish-Windows-x86-64-universal.exe")
    eng.start()
    eng.set_engine_options(threads=8, hash_mb=512, multipv=3)
    eng.position(fen, moves)
    eng.go(depth=24, on_info=print, on_bestmove=lambda m: print("best", m))
    ...
    eng.quit()
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional


def find_engine_exe() -> Optional[str]:
    """在 engine/bin 下查找 Pikafish 可执行文件。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "bin")
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        if name.lower().endswith(".exe") and "pikafish" in name.lower():
            return os.path.join(base, name)
    # 非 Windows 或未知命名：任何 exe / 无后缀可执行文件都试一下
    for name in sorted(os.listdir(base)):
        if name.lower().endswith(".exe"):
            return os.path.join(base, name)
    return None


def find_nnue_file() -> Optional[str]:
    """在 engine/bin 下查找 NNUE 权重文件。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "bin")
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        if name.lower().endswith(".nnue"):
            return os.path.join(base, name)
    return None


class UciEngine:
    """UCI 引擎进程管理与分析控制。"""

    def __init__(self, exe_path: str, nnue_path: Optional[str] = None, name: str = "Pikafish"):
        self.exe_path = exe_path
        self.nnue_path = nnue_path
        self.name = name
        self.proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._quit = False
        self.on_info: Optional[Callable[[dict], None]] = None
        self.on_bestmove: Optional[Callable[[Optional[str]], None]] = None
        self.engine_id: str = ""
        self.options: Dict[str, str] = {}
        self._got_token = False

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        """启动引擎进程并完成 UCI 握手（阻塞直至 uciok）。"""
        if self.proc is not None:
            return
        self._quit = False
        # BELOW_NORMAL_PRIORITY_CLASS：引擎满线程搜索时不再把界面进程
        # 饿到动画卡顿；对引擎自身棋力/速度几乎无影响（其它核心空闲时照常跑满）
        below_normal = 0x4000
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(self.exe_path) or None,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | below_normal
            if sys.platform == "win32" else 0,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._got_token = False
        self._send("uci")
        if not self._wait_for("uciok", timeout=15):
            self.stop()
            raise RuntimeError("引擎未响应 uciok，可能不是 UCI 引擎")

    def stop(self) -> None:
        """终止引擎进程。"""
        self._quit = True
        if self.proc is not None:
            try:
                self._send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def quit(self) -> None:
        self.stop()

    # ------------------------------------------------------------------ 配置
    def set_engine_options(self, threads: int = None, hash_mb: int = None,
                           multipv: int = None, extra: dict = None) -> None:
        """设置常用引擎选项。"""
        if threads is not None:
            self.set_option("Threads", threads)
        if hash_mb is not None:
            self.set_option("Hash", hash_mb)
        if multipv is not None:
            self.set_option("MultiPV", multipv)
        if self.nnue_path:
            self.set_option("EvalFile", self.nnue_path)
        for k, v in (extra or {}).items():
            self.set_option(k, v)

    def set_option(self, name: str, value) -> None:
        self.options[name] = str(value)
        self._send(f"setoption name {name} value {value}")

    def new_game(self) -> None:
        self._send("ucinewgame")

    # ------------------------------------------------------------------ 搜索
    def position(self, fen: str, moves: Optional[List[str]] = None) -> None:
        """设置局面。moves 为 ICCS 着法列表（如 ['h2e2', ...]）。"""
        cmd = f"position fen {fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def go(self, depth: int = None, movetime_ms: int = None,
           nodes: int = None, infinite: bool = False,
           searchmoves: Optional[List[str]] = None) -> None:
        """开始搜索（异步）。先 stop_search() 再重新 go()。

        searchmoves 为 ICCS 着法白名单（如 ['h2e2']）：只在这些着法里
        选最优——用于犯规规避时排除重复着法后重新搜索。
        """
        cmd = "go"
        if depth is not None:
            cmd += f" depth {depth}"
        if movetime_ms is not None:
            cmd += f" movetime {movetime_ms}"
        if nodes is not None:
            cmd += f" nodes {nodes}"
        if infinite:
            cmd += " infinite"
        # searchmoves 必须放最后：Pikafish 会把其后的所有 token 都当作
        # 着法白名单（实测 `go searchmoves x y depth 6` 会吞掉 depth）
        if searchmoves:
            cmd += " searchmoves " + " ".join(searchmoves)
        self._send(cmd)

    def stop_search(self) -> None:
        self._send("stop")

    # ------------------------------------------------------------------ 内部
    def _send(self, cmd: str) -> None:
        if self.proc is None:
            return
        with self._write_lock:
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass

    def _wait_for(self, token: str, timeout: float) -> bool:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._got_token:
                return True
            time.sleep(0.05)
        return False

    def _read_loop(self) -> None:
        proc = self.proc
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line == "uciok":
                self._got_token = True
            if line.startswith("id name "):
                self.engine_id = line[8:]
            if line.startswith("info "):
                info = parse_info_line(line)
                if info and self.on_info:
                    try:
                        self.on_info(info)
                    except Exception:
                        pass
            elif line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 and parts[1] != "(none)" else None
                if self.on_bestmove:
                    try:
                        self.on_bestmove(move)
                    except Exception:
                        pass
        self._got_token = True


def parse_info_line(line: str) -> Optional[dict]:
    """解析 UCI info 行为字典。"""
    parts = line.split()
    info: dict = {}
    i = 1
    while i < len(parts):
        key = parts[i]
        if key == "score":
            if i + 2 < len(parts):
                kind, val = parts[i + 1], parts[i + 2]
                try:
                    info["score"] = int(val)
                except ValueError:
                    info["score"] = 0
                info["score_kind"] = kind  # cp / mate
                info["score_str"] = _fmt_score(kind, val)
                i += 3
                continue
            i += 1
        elif key == "pv":
            info["pv"] = parts[i + 1:]
            break
        elif key in ("depth", "seldepth", "multipv", "nodes", "nps", "hashfull",
                     "time", "tbhits"):
            if i + 1 < len(parts):
                try:
                    info[key] = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            i += 1
        elif key == "string":
            info["string"] = " ".join(parts[i + 1:])
            break
        else:
            i += 1
    info.setdefault("multipv", 1)
    return info if ("pv" in info or "score" in info) else None


def _fmt_score(kind: str, val: str) -> str:
    try:
        v = int(val)
    except ValueError:
        return val
    if kind == "mate":
        return f"#{v}" if v > 0 else f"#{v}"
    # 百分兵 -> 士/子 换算显示
    return f"{v / 100:+.2f}"
