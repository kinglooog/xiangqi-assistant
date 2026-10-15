# -*- coding: utf-8 -*-
"""对局窗口实时同步的状态机（一期·自动观察）。

输入是逐帧识别结果（10x9 棋子矩阵 + 方向），输出同步事件：
- baseline：首个稳定局面（作为同步基准）
- move    ：识别到一步合法形态的走子（含吃子）
- chaos   ：基准到当前的变化无法用一步走子解释（弹窗/误识别/跳变），
            需要人工处理，不做自动记录
- orientation_changed：截图方向（红方在上/在下）中途变化

设计要点：连续 stable_needed 帧识别一致才认定为"稳定"，
过滤走子动画/半透明弹窗造成的瞬时误识别。
"""
from __future__ import annotations

from typing import List, Optional


Grid = List[List[Optional[str]]]


class LiveObserver:
    def __init__(self, stable_needed: int = 2):
        self.stable_needed = max(1, stable_needed)
        self.last_stable: Optional[Grid] = None
        self.last_orientation: Optional[bool] = None
        self._pending: Optional[Grid] = None
        self._pending_count = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空全部状态（停止观察/重新连接时）。"""
        self.last_stable = None
        self.last_orientation = None
        self._pending = None
        self._pending_count = 0

    def force_baseline(self, grid: Grid, red_at_top: bool) -> None:
        """以当前画面强行重建基准（用户点"重新同步"时）。"""
        self.last_stable = [row[:] for row in grid]
        self.last_orientation = red_at_top
        self._pending = None
        self._pending_count = 0

    # ------------------------------------------------------------------
    def feed(self, grid: Optional[Grid], red_at_top: bool) -> List[dict]:
        """喂入一帧识别结果，返回事件列表（可为空）。"""
        if grid is None:
            self._pending = None
            self._pending_count = 0
            return []
        if self.last_stable is not None and red_at_top != self.last_orientation:
            # 方向变化：旧基准全部失效，等待重新建立
            self.last_stable = None
            self._pending = None
            self._pending_count = 0
            return [{"type": "orientation_changed"}]

        if self.last_stable is not None and grid == self.last_stable:
            self._pending = None
            self._pending_count = 0
            return []

        if self._pending is not None and grid == self._pending:
            self._pending_count += 1
        else:
            self._pending = [row[:] for row in grid]
            self._pending_count = 1

        if self._pending_count < self.stable_needed:
            return []

        stable = self._pending
        self._pending = None
        self._pending_count = 0

        if self.last_stable is None:
            self.last_stable = [row[:] for row in stable]
            self.last_orientation = red_at_top
            return [{"type": "baseline", "grid": [row[:] for row in stable]}]

        self.last_stable = [row[:] for row in stable]
        # 走子解释交给上层（用合法走法约束推断，棋盘感知），
        # 观察器只负责稳定性与基准管理
        return [{"type": "changed", "grid": [row[:] for row in stable]}]


