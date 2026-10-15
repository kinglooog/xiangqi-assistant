# -*- coding: utf-8 -*-
"""对局状态管理：走子历史、悔棋、引擎着法记录。"""
from __future__ import annotations

from typing import List, Optional, Tuple

from core.board import Board, Move, move_to_chinese, move_to_iccs

HistoryEntry = Tuple[Move, Optional[str], str]  # (走法, 被吃棋子, 走之前的行棋方)


class Game:
    """一局棋的完整状态。"""

    def __init__(self, fen: str = None):
        self.board = Board(fen) if fen else Board()
        self.start_fen = self.board.to_fen()   # 本局起点（回放/重复判定用）
        self.history: List[HistoryEntry] = []
        self.result: Optional[str] = None

    @property
    def side_to_move(self) -> str:
        return self.board.side_to_move

    def play(self, move: Move) -> bool:
        """执行走法。非法走法返回 False。"""
        if self.result is not None:
            return False
        mover = self.board.side_to_move
        try:
            captured = self.board.make_move(move)
        except ValueError:
            return False
        self.history.append((move, captured, mover))
        over, result = self.board.is_game_over()
        if over:
            self.result = result
        return True

    def undo(self) -> bool:
        """悔一步棋。"""
        if not self.history:
            return False
        move, captured, mover = self.history.pop()
        (fr, fc), (tr, tc) = move
        self.board.grid[fr][fc] = self.board.grid[tr][tc]
        self.board.grid[tr][tc] = captured
        self.board.side_to_move = mover
        self.result = None
        return True

    def reset(self, fen: str = None) -> None:
        self.board.set_fen(fen if fen else "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
        self.start_fen = self.board.to_fen()
        self.history.clear()
        self.result = None

    def move_list_iccs(self) -> List[str]:
        return [move_to_iccs(m) for m, _, _ in self.history]

    def move_list_chinese(self) -> List[str]:
        """逐着回放生成中文记法列表。"""
        probe = Board(self.board.to_fen())
        # 先回退全部历史到起点
        for move, captured, mover in reversed(self.history):
            (fr, fc), (tr, tc) = move
            probe.grid[fr][fc] = probe.grid[tr][tc]
            probe.grid[tr][tc] = captured
            probe.side_to_move = mover
        out = []
        for move, _, _ in self.history:
            out.append(move_to_chinese(probe, move))
            probe.make_move(move)
        return out
