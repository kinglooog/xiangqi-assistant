# -*- coding: utf-8 -*-
"""中国象棋棋盘与规则核心。

坐标系约定：
- row 0 为黑方底线（棋盘顶部），row 9 为红方底线（棋盘底部）；
- col 0 为 a 路（红方视角最左），col 8 为 i 路；
- 记谱纵线号（ICCS）：纵线 a-i 对应 col 0-8，横线号 = 9 - row（红方底线为 0）；
- FEN 从 row 0（黑方底线）写到 row 9（红方底线），大写红方、小写黑方。

棋子用单字符表示：K 帅/将、A 仕/士、B 相/象、N 马、R 车、C 炮、P 兵/卒，
大写为红方，小写为黑方，None 为空位。
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

RED, BLACK = "w", "b"
FILES = "abcdefghi"
COLS, ROWS = 9, 10

# 九宫范围
PALACE_COLS = (3, 4, 5)
PALACE_ROWS = {"w": (7, 8, 9), "b": (0, 1, 2)}

# 棋子名称（红/黑）
PIECE_NAMES = {
    "K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵",
    "k": "将", "a": "士", "b": "象", "n": "马", "r": "车", "c": "炮", "p": "卒",
}

Move = Tuple[Tuple[int, int], Tuple[int, int]]  # ((frm_row, frm_col), (to_row, to_col))


def side_of(piece: str) -> str:
    """返回棋子所属方：'w' 红 / 'b' 黑。"""
    return RED if piece.isupper() else BLACK


def in_board(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def in_palace(row: int, col: int, side: str) -> bool:
    return col in PALACE_COLS and PALACE_ROWS[side][0] <= row <= PALACE_ROWS[side][2]


def own_half(row: int, side: str) -> bool:
    """棋子是否在己方半场（象/士活动范围判断用）。红方 row>=5，黑方 row<=4。"""
    return row >= 5 if side == RED else row <= 4


def crossed_river(row: int, side: str) -> bool:
    """兵是否已过河。红方过河 row<=4，黑方过河 row>=5。"""
    return row <= 4 if side == RED else row >= 5


class Board:
    """棋盘状态与规则。"""

    def __init__(self, fen: str = INITIAL_FEN):
        self.set_fen(fen)

    # ------------------------------------------------------------------ FEN
    def set_fen(self, fen: str) -> None:
        fields = fen.strip().split()
        if len(fields) < 2:
            raise ValueError(f"非法 FEN: {fen!r}")
        rows = fields[0].split("/")
        if len(rows) != ROWS:
            raise ValueError(f"FEN 行数错误: {fen!r}")
        grid: List[List[Optional[str]]] = [[None] * COLS for _ in range(ROWS)]
        for r, row_str in enumerate(rows):
            c = 0
            for ch in row_str:
                if ch.isdigit():
                    c += int(ch)
                elif ch in PIECE_NAMES:
                    if c >= COLS:
                        raise ValueError(f"FEN 列越界: {fen!r}")
                    grid[r][c] = ch
                    c += 1
                else:
                    raise ValueError(f"FEN 含未知棋子 {ch!r}: {fen!r}")
            if c != COLS:
                raise ValueError(f"FEN 行长度错误: {fen!r}")
        if fields[1] not in (RED, BLACK):
            raise ValueError(f"FEN 行棋方错误: {fen!r}")
        self.grid = grid
        self.side_to_move = fields[1]

    def to_fen(self) -> str:
        rows = []
        for r in range(ROWS):
            row_str, empty = "", 0
            for c in range(COLS):
                p = self.grid[r][c]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row_str += str(empty)
                        empty = 0
                    row_str += p
            if empty:
                row_str += str(empty)
            rows.append(row_str)
        return f"{'/'.join(rows)} {self.side_to_move} - - 0 1"

    # ------------------------------------------------------------------ 基础查询
    def piece_at(self, row: int, col: int) -> Optional[str]:
        return self.grid[row][col]

    def find_king(self, side: str) -> Optional[Tuple[int, int]]:
        target = "K" if side == RED else "k"
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == target:
                    return (r, c)
        return None

    # ------------------------------------------------------------------ 走法生成
    def _pseudo_moves_from(self, row: int, col: int) -> Iterator[Move]:
        """生成 (row,col) 处棋子的所有伪合法走法（不含自将检查）。"""
        piece = self.grid[row][col]
        if piece is None:
            return
        side = side_of(piece)
        up = piece.upper()

        if up == "R":  # 车
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                while in_board(r, c):
                    if self.grid[r][c] is None:
                        yield ((row, col), (r, c))
                    else:
                        if side_of(self.grid[r][c]) != side:
                            yield ((row, col), (r, c))
                        break
                    r, c = r + dr, c + dc

        elif up == "C":  # 炮：平移不吃子，吃子需隔一个炮架
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                # 平移段
                while in_board(r, c) and self.grid[r][c] is None:
                    yield ((row, col), (r, c))
                    r, c = r + dr, c + dc
                if not in_board(r, c):
                    continue
                # 越过炮架找目标
                r, c = r + dr, c + dc
                while in_board(r, c):
                    target = self.grid[r][c]
                    if target is not None:
                        if side_of(target) != side:
                            yield ((row, col), (r, c))
                        break
                    r, c = r + dr, c + dc

        elif up == "N":  # 马：蹩马腿
            for dr, dc, lr, lc in ((-2, -1, -1, 0), (-2, 1, -1, 0),
                                   (2, -1, 1, 0), (2, 1, 1, 0),
                                   (-1, -2, 0, -1), (1, -2, 0, -1),
                                   (-1, 2, 0, 1), (1, 2, 0, 1)):
                r, c = row + dr, col + dc
                leg_r, leg_c = row + lr, col + lc
                if in_board(r, c) and self.grid[leg_r][leg_c] is None:
                    target = self.grid[r][c]
                    if target is None or side_of(target) != side:
                        yield ((row, col), (r, c))

        elif up == "B":  # 相/象：塞象眼，不过河
            for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                r, c = row + dr, col + dc
                eye_r, eye_c = row + dr // 2, col + dc // 2
                if (in_board(r, c) and own_half(r, side)
                        and self.grid[eye_r][eye_c] is None):
                    target = self.grid[r][c]
                    if target is None or side_of(target) != side:
                        yield ((row, col), (r, c))

        elif up == "A":  # 仕/士：九宫斜行
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                r, c = row + dr, col + dc
                if in_palace(r, c, side):
                    target = self.grid[r][c]
                    if target is None or side_of(target) != side:
                        yield ((row, col), (r, c))

        elif up == "K":  # 帅/将：九宫直行 + 飞将（白脸将作为吃子走法）
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if in_palace(r, c, side):
                    target = self.grid[r][c]
                    if target is None or side_of(target) != side:
                        yield ((row, col), (r, c))
            # 飞将：同线直接吃对方将帅
            enemy_k = self.find_king(BLACK if side == RED else RED)
            if enemy_k:
                er, ec = enemy_k
                if ec == col:
                    clear = all(self.grid[r][col] is None
                                for r in range(min(row, er) + 1, max(row, er)))
                    if clear:
                        yield ((row, col), (er, ec))

        elif up == "P":  # 兵/卒：过河前只进，过河后可横移
            fwd = -1 if side == RED else 1
            r = row + fwd
            if in_board(r, col):
                target = self.grid[r][col]
                if target is None or side_of(target) != side:
                    yield ((row, col), (r, col))
            if crossed_river(row, side):
                for dc in (-1, 1):
                    c = col + dc
                    if in_board(row, c):
                        target = self.grid[row][c]
                        if target is None or side_of(target) != side:
                            yield ((row, col), (row, c))

    def attacks_square(self, row: int, col: int, by_side: str) -> bool:
        """(row,col) 是否被 by_side 一方的棋子攻击（含将帅对脸的飞将攻击）。"""
        king = self.find_king(by_side)
        if king and king[1] == col:
            # 飞将：将帅之间无遮挡即互攻
            kr = king[0]
            between = [self.grid[r][col] for r in range(min(kr, row) + 1, max(kr, row))]
            if all(p is None for p in between):
                return True
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p is None or side_of(p) != by_side:
                    continue
                up = p.upper()
                if up == "P":
                    fwd = -1 if by_side == RED else 1
                    if r + fwd == row and c == col:
                        return True
                    if crossed_river(r, by_side) and r == row and abs(c - col) == 1:
                        return True
                elif up == "N":
                    if abs(r - row) == 2 and abs(c - col) == 1:
                        lr, lc = (r + (row - r) // 2), c
                    elif abs(r - row) == 1 and abs(c - col) == 2:
                        lr, lc = r, (c + (col - c) // 2)
                    else:
                        continue
                    if self.grid[lr][lc] is None:
                        return True
                elif up == "C":
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        # 找炮架
                        while in_board(rr, cc) and self.grid[rr][cc] is None:
                            rr, cc = rr + dr, cc + dc
                        if not in_board(rr, cc):
                            continue
                        rr, cc = rr + dr, cc + dc
                        while in_board(rr, cc):
                            if rr == row and cc == col:
                                return True
                            if self.grid[rr][cc] is not None:
                                break
                            rr, cc = rr + dr, cc + dc
                elif up == "R":
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        while in_board(rr, cc):
                            if rr == row and cc == col:
                                return True
                            if self.grid[rr][cc] is not None:
                                break
                            rr, cc = rr + dr, cc + dc
                # A/B 士象不攻击将（不可能过河吃到将），K 已单独处理飞将，
                # 士象在九宫内直接接触将的情况由走法生成时目标格判断覆盖。
        return False

    def in_check(self, side: str) -> bool:
        king = self.find_king(side)
        if king is None:
            return True
        return self.attacks_square(king[0], king[1], BLACK if side == RED else RED)

    def kings_face(self) -> bool:
        """将帅是否照面（同一路且中间无子）。这种局面在象棋中不合法。"""
        rk, bk = self.find_king(RED), self.find_king(BLACK)
        if rk is None or bk is None:
            return False
        if rk[1] != bk[1]:
            return False
        col = rk[1]
        return all(self.grid[r][col] is None
                   for r in range(min(rk[0], bk[0]) + 1, max(rk[0], bk[0])))

    def validate_position(self) -> Optional[str]:
        """检查局面是否为合法象棋局面。合法返回 None，否则返回问题描述。

        用于识别结果与手动摆局的校验——非法局面会让引擎拒绝工作
        （Pikafish 会直接退出）或给出无意义的分析。
        """
        from collections import Counter
        counts = Counter(p for row in self.grid for p in row if p)
        for up, name, lim in (("K", "帅", 1), ("A", "仕", 2), ("B", "相", 2),
                              ("N", "马", 2), ("R", "车", 2), ("C", "炮", 2),
                              ("P", "兵", 5)):
            for letter in (up, up.lower()):
                n = counts.get(letter, 0)
                if n > lim:
                    who = "红方" if letter.isupper() else "黑方"
                    return f"{who}{name}超过 {lim} 个（当前 {n} 个）"
        for side, letter, who in ((RED, "K", "红方"), (BLACK, "k", "黑方")):
            if counts.get(letter, 0) != 1:
                return f"{who}缺少将/帅"
        if self.kings_face():
            return "将帅照面（同一路无遮挡），不是合法局面"
        # 兵/卒不可能留在己方底线（引擎会拒收这类局面并直接退出）
        for c in range(COLS):
            if self.grid[ROWS - 1][c] == "P":
                return "红兵不可能在己方底线"
            if self.grid[0][c] == "p":
                return "黑卒不可能在己方底线"
        for side in (RED, BLACK):
            k = self.find_king(side)
            if k is not None and not in_palace(k[0], k[1], side):
                who = "红方" if side == RED else "黑方"
                return f"{who}将/帅不在九宫内"
        # 非行棋方被将军同样非法：上一手不能让自己的将处于被将状态。
        # Pikafish 会拒收这类局面并直接退出（实测确认），必须拦住。
        other = BLACK if self.side_to_move == RED else RED
        if self.in_check(other):
            who = "黑方" if other == BLACK else "红方"
            return f"{who}（非行棋方）正被将军，轮次疑似有误"
        return None

    def _make(self, move: Move) -> Optional[str]:
        (fr, fc), (tr, tc) = move
        captured = self.grid[tr][tc]
        self.grid[tr][tc] = self.grid[fr][fc]
        self.grid[fr][fc] = None
        return captured

    def _unmake(self, move: Move, captured: Optional[str]) -> None:
        (fr, fc), (tr, tc) = move
        self.grid[fr][fc] = self.grid[tr][tc]
        self.grid[tr][tc] = captured

    def legal_moves(self) -> List[Move]:
        """当前行棋方的所有合法走法。"""
        side = self.side_to_move
        result: List[Move] = []
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p is None or side_of(p) != side:
                    continue
                for move in self._pseudo_moves_from(r, c):
                    captured = self._make(move)
                    try:
                        king = self.find_king(side)
                        ok = king is not None and not self.in_check(side)
                    finally:
                        self._unmake(move, captured)
                    if ok:
                        result.append(move)
        return result

    def is_legal_move(self, move: Move) -> bool:
        return move in self.legal_moves()

    def make_move(self, move: Move) -> Optional[str]:
        """执行走法并切换行棋方，返回被吃掉的棋子。"""
        if not self.is_legal_move(move):
            raise ValueError(f"非法走法: {move_to_iccs(move)}")
        captured = self._make(move)
        self.side_to_move = BLACK if self.side_to_move == RED else RED
        return captured

    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """返回 (是否结束, 结果描述)。象棋中被困毙/被将死均判负。"""
        if self.find_king(RED) is None:
            return True, "黑方胜（红帅被吃）"
        if self.find_king(BLACK) is None:
            return True, "红方胜（黑将被吃）"
        if not self.legal_moves():
            loser = "红方" if self.side_to_move == RED else "黑方"
            check = self.in_check(self.side_to_move)
            reason = "被将死" if check else "被困毙"
            return True, f"{'黑方' if loser == '红方' else '红方'}胜（{loser}{reason}）"
        return False, None


# ------------------------------------------------------------------ 记法
def sq_iccs(row: int, col: int) -> str:
    """格子 -> ICCS 坐标（如 h2）。"""
    return f"{FILES[col]}{9 - row}"


def iccs_to_sq(s: str) -> Tuple[int, int]:
    """ICCS 坐标 -> 格子。"""
    return (9 - int(s[1]), FILES.index(s[0]))


def move_to_iccs(move: Move) -> str:
    return sq_iccs(*move[0]) + sq_iccs(*move[1])


def iccs_to_move(s: str) -> Move:
    return iccs_to_sq(s[:2]), iccs_to_sq(s[2:4])


_CN_DIGITS = "零一二三四五六七八九"


def _file_num(side: str, col: int) -> str:
    """纵线号码：红方用中文数字（从红方右侧数），黑方用阿拉伯数字（从黑方右侧数）。"""
    if side == RED:
        return _CN_DIGITS[9 - col]
    return str(col + 1)


def _step_num(side: str, n: int) -> str:
    return _CN_DIGITS[n] if side == RED else str(n)


def move_to_chinese(board: Board, move: Move) -> str:
    """把走法转成中文纵线记法（如 炮二平五、马8进7、前车进一）。

    需要在走法尚未执行（或已回退）的棋盘上调用。
    """
    (fr, fc), (tr, tc) = move
    piece = board.piece_at(fr, fc)
    if piece is None:
        return move_to_iccs(move)
    side = side_of(piece)
    name = PIECE_NAMES[piece]
    red_side = side == RED

    # 同列同类棋子 >=2 时用 前/中/后 前缀
    same_col = []
    for r in range(ROWS):
        p = board.piece_at(r, fc)
        if p is not None and p == piece:
            same_col.append(r)
    prefix = ""
    if len(same_col) >= 2:
        # 按前进方向排序（红方前进方向为 row 减小，row 小者为前）
        ordered = sorted(same_col) if red_side else sorted(same_col, reverse=True)
        idx = ordered.index(fr)
        if len(ordered) == 2:
            prefix = "前" if idx == 0 else "后"
        else:
            prefix = "前" if idx == 0 else ("后" if idx == len(ordered) - 1 else "中")
        frm = prefix + name
    else:
        frm = name + _file_num(side, fc)

    fwd = -1 if red_side else 1
    if tr == fr:  # 平
        action = "平" + _file_num(side, tc)
    elif (tr - fr) * fwd > 0:  # 进
        if piece.upper() in ("N", "B", "A"):  # 斜行走子用目标纵线
            action = "进" + _file_num(side, tc)
        else:
            action = "进" + _step_num(side, abs(tr - fr))
    else:  # 退
        if piece.upper() in ("N", "B", "A"):
            action = "退" + _file_num(side, tc)
        else:
            action = "退" + _step_num(side, abs(tr - fr))
    return frm + action

def infer_move_from_observation(board: "Board", observed_grid: List[List[Optional[str]]],
                                min_score: float = 0.985,
                                min_margin: float = 0.008
                                ) -> Tuple[Optional[Move], float]:
    """从合法走法中推断观察帧对应的着法（转移约束识别）。

    思路：从当前局面出发，合法走法只有几十种；对每个合法走法推演
    "走完后的局面"，与观察帧逐格比对打分：
    - 完全一致（含棋子种类）：+1
    - 同色不同子（典型误读形态，如 車→象/炮）：+0.75
    - 占用矛盾（有无棋子不符）或异色：0
    得分最高且与次优差距足够大的合法走法即为推断结果。

    返回 (move, best_score)；无可信匹配时 move 为 None。
    """
    legal = board.legal_moves()
    if not legal:
        return None, 0.0
    best_mv, best_s, second_s = None, -1.0, -1.0
    for mv in legal:
        (fr, fc), (tr, tc) = mv
        mover = board.grid[fr][fc]
        s = 0.0
        for r in range(ROWS):
            for c in range(COLS):
                h = board.grid[r][c]
                if (r, c) == (fr, fc):
                    h = None
                elif (r, c) == (tr, tc):
                    h = mover
                o = observed_grid[r][c]
                if h == o:
                    s += 1.0
                elif (h is None) != (o is None):
                    pass                       # 占用矛盾
                elif h is not None and side_of(h) == side_of(o):
                    s += 0.75                  # 同色异子：误读形态
        s /= (ROWS * COLS)
        if s > best_s:
            second_s = best_s
            best_mv, best_s = mv, s
        elif s > second_s:
            second_s = s
    if best_s >= min_score and (best_s - second_s) >= min_margin:
        return best_mv, best_s
    return None, best_s
