# -*- coding: utf-8 -*-
"""重复局面与禁止着法判定（长将 / 长捉）。

中国象棋竞赛规则：长将、长捉、一将一捉属禁止着法，重复不变者判负；
长兑、长跟、长拦等允许着法不变作和。Pikafish 引擎没有相关设置
（UCI 选项里没有——实测确认），它只把局面重复当成和棋（评估 0.00），
处于劣势时会一直将军"求和"，在平台上直接判负。因此必须在软件层
替它把关：检测我方循环里是否全是将军/捉子着法，是则换着法脱出。

判定思路（务实近似，宁可少判不可误判）：
- 在我方该走棋时，若当前局面此前已出现过（重复循环），统计这一循环里
  我方每一着：全是将军 -> 长将；全是将军或白吃威胁 -> 长捉/一将一捉；
  只要有一着闲着（含兑、跟、拦）即视为允许着法，不下判定；
- 对方在同一循环里同样全是将军/捉子（双方都犯规，按和棋处理）时
  不干预——此时继续循环反而对我方有利或无害。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.board import Board, Move

# 兵种价值（近似值，用于"白吃威胁"的净收益估算）
PIECE_VALUE: Dict[str, int] = {
    "K": 10000, "R": 900, "C": 450, "N": 400, "B": 200, "A": 200, "P": 100,
}


def position_key(board: Board) -> str:
    """局面指纹：棋子布局 + 行棋方（重复判定不含回合计数）。"""
    parts = board.to_fen().split()
    return parts[0] + " " + parts[1]


def _apply(board: Board, move: Move) -> Board:
    b = Board(board.to_fen())
    b.make_move(move)
    return b


def position_keys(start_fen: str, moves: Iterable[Move]) -> List[str]:
    """从 start_fen 出发逐着回放，返回 [起点, 每步之后] 的局面指纹。"""
    b = Board(start_fen)
    keys = [position_key(b)]
    for m in moves:
        try:
            b.make_move(m)
        except ValueError:
            break
        keys.append(position_key(b))
    return keys


def is_check_move(board: Board, move: Move) -> bool:
    """走后是否将军（对方被将）。board 需轮到我方走。"""
    try:
        b = _apply(board, move)
    except ValueError:
        return False
    return b.in_check(b.side_to_move)


def _winning_targets(board: Board, side: str) -> Set[Tuple[int, int]]:
    """side 若轮走，能"白吃"（净赚至少一个兵）的目标格集合。

    两棋近似：吃子后对方能立即回吃就扣掉我方吃子子的价值；回吃本身
    若亏（吃掉的小于被吃的）则近似忽略——宁可少判不可误判。
    """
    b0 = Board(board.to_fen())
    b0.side_to_move = side
    out: Set[Tuple[int, int]] = set()
    for m in b0.legal_moves():
        (fr, fc), (tr, tc) = m
        victim = b0.grid[tr][tc]
        if victim is None:
            continue
        net = PIECE_VALUE[victim.upper()]
        b1 = _apply(b0, m)
        can_recapture = any((t, u) == (tr, tc)
                            for ((_f2, _c2), (t, u)) in b1.legal_moves())
        if can_recapture:
            net -= PIECE_VALUE[b0.grid[fr][fc].upper()]
        if net >= PIECE_VALUE["P"]:
            out.add((tr, tc))
    return out


def is_chase_move(board: Board, move: Move) -> bool:
    """走后是否形成"捉"：新增能白吃得子的目标（兑、跟、拦不算）。"""
    mover = board.side_to_move
    try:
        b_after = _apply(board, move)
    except ValueError:
        return False
    new = _winning_targets(b_after, mover) - _winning_targets(board, mover)
    return bool(new)


def _aggression(board: Board, move: Move) -> Optional[str]:
    """着法的攻击性：'将'（将军）/ '吃'（吃子）/ '捉'（白吃威胁）；闲着 None。"""
    (fr, fc), (tr, tc) = move
    if is_check_move(board, move):
        return "将"
    if board.grid[tr][tc] is not None:
        return "吃"
    if is_chase_move(board, move):
        return "捉"
    return None


def detect_our_foul(board: Board, start_fen: str, moves: List[Move],
                    our_side: str) -> Optional[str]:
    """当前是否已陷入"我方禁止着法"的重复循环。

    在我方该走棋时调用。返回 '长将' / '长捉' / '一将一捉'；
    未形成循环、循环中含允许着法、或双方都犯规（作和）时返回 None。
    """
    boards = [Board(start_fen)]
    for m in moves:
        try:
            boards.append(_apply(boards[-1], m))
        except ValueError:
            return None
    if boards[-1].side_to_move != our_side:
        return None      # 回放与当前行棋方不一致（如轮次自纠），不下判定
    keys = [position_key(b) for b in boards]
    occ = [i for i, k in enumerate(keys) if k == keys[-1]]
    if len(occ) < 2:
        return None      # 当前局面尚未重复
    j = occ[0]
    our_kinds: List[str] = []
    opp_aggressive: List[bool] = []
    for i in range(j, len(moves)):
        bd, m = boards[i], moves[i]
        agg = _aggression(bd, m)
        if bd.side_to_move == our_side:
            if agg is None:
                return None          # 有闲着：长兑/长跟等允许着法，不判
            our_kinds.append(agg)
        else:
            opp_aggressive.append(agg is not None)
    if not our_kinds:
        return None
    if opp_aggressive and all(opp_aggressive):
        return None                  # 双方都犯规：按规则作和，不干预
    if all(k == "将" for k in our_kinds):
        return "长将"
    if any(k == "将" for k in our_kinds):
        return "一将一捉"
    return "长捉"


def repeating_moves(board: Board, keys: Iterable[str]) -> Set[Move]:
    """当前局面上会落回已有局面（造成重复）的合法着法集合。"""
    seen = set(keys)
    out: Set[Move] = set()
    for m in board.legal_moves():
        try:
            if position_key(_apply(board, m)) in seen:
                out.add(m)
        except ValueError:
            continue
    return out
