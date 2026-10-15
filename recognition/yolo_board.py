# -*- coding: utf-8 -*-
"""由 YOLOv5 检测结果还原棋盘局面。

算法照搬 VinXiangQi 的 GetBoardFromPrediction（作者 Vincentzyx，GPL-3.0），
用 Python 重写：

1. 取类别 "board" 的检测框作为棋盘区域；
2. 过滤棋子检测：宽高比需在 0.7~1.3（近似正方），且落在棋盘框内；
3. 若棋子中心在横竖方向的跨度足够（≥ 最小间距的 7.6 / 8.6 倍），
   用棋子中心的最小/最大范围重建棋盘框——比模型给的框更贴合交点；
4. 网格间距 gridWidth = board.Width / 8、gridHeight = board.Height / 9；
5. 棋子中心相对棋盘框的偏移除以网格间距并四舍五入即得格子坐标；
6. 方向：红帅落在图像上半部则说明截图红方在上，旋转 180° 得到 FEN 顺序。

相比透视变换方案，该方法只依赖模型的棋盘检测，对窗口位置、缩放、
有无无关 UI 元素都更鲁棒。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.board import Board, BLACK, RED, in_palace
from recognition.yolo import CLASS_TO_FEN, Detection
from recognition import skin

RATIO_MIN, RATIO_MAX = 0.7, 1.3   # 棋子框宽高比范围


@dataclass
class BoardResult:
    grid: List[List[Optional[str]]]   # FEN 顺序：row0 = 黑方底线
    red_at_top: bool                  # 截图是否红方在上（用户执黑）
    board_box: Tuple[float, float, float, float]
    n_pieces: int
    n_ambiguous: int                  # 多个棋子映射到同一格的冲突次数
    low_conf: List[str]               # 低置信度棋子描述（供提示用户校对）
    corrections: List[Tuple[int, int, str, str, float]] = None  # 皮肤模板修正
    n_corrected: int = 0              # 被皮肤模板纠正的棋子数

    def __post_init__(self):
        if self.corrections is None:
            self.corrections = []


def map_center_to_cell(cx: float, cy: float, board_box,
                       grid_w: float, grid_h: float
                       ) -> Optional[Tuple[int, int]]:
    """棋子中心 -> 棋盘格 (row, col)；越界返回 None。

    贴边行/列的棋子中心因棋盘框偏差可能算出 -1 / 10（只差半格内），
    回夹到边行/边列而不是丢弃——红帅黑将都在底线行，丢了整帧会被判成
    "残缺帧"作废（"缺少红帅/黑将"大量误报的根源）。

    识别（board_from_detections）与皮肤模板采集（skin.harvest）必须共用
    这一个实现：两边映射不一致时，会把邻格棋子当成本格类别采集，
    污染模板（实际发生过的隐患）。
    """
    bx1, by1, _bx2, _by2 = board_box
    col = int(round((cx - bx1) / grid_w))
    row = int(round((cy - by1) / grid_h))
    if row == -1:
        row = 0
    elif row == 10:
        row = 9
    if col == -1:
        col = 0
    elif col == 9:
        col = 8
    if not (0 <= col <= 8 and 0 <= row <= 9):
        return None
    return (row, col)


def board_from_detections(dets: List[Detection],
                          score_min: float = 0.5,
                          img: Optional[np.ndarray] = None,
                          skin_templates: Optional[dict] = None
                          ) -> Optional[BoardResult]:
    """检测结果 -> 棋盘局面。无法定位棋盘时返回 None。

    img 与 skin_templates 同时提供时，对每个棋子做皮肤模板二次校验：
    模板意见清晰且与模型不一致时以模板为准（修正 JJ 等皮肤上的
    系统性误判，如 車→象、車→炮）。
    """
    board = next((d for d in dets if d.label == "board"), None)
    pieces = [d for d in dets
              if d.label != "board" and d.label in CLASS_TO_FEN
              and d.score >= score_min
              and RATIO_MIN <= (d.width / d.height if d.height else 0) <= RATIO_MAX]
    if not pieces:
        return None

    if board is not None:
        bx1, by1, bx2, by2 = board.box
    else:
        # 没有 board 类：直接用棋子中心范围（VinXiangQi 的兜底路径）
        xs = [p.center[0] for p in pieces]
        ys = [p.center[1] for p in pieces]
        bx1, bx2, by1, by2 = min(xs), max(xs), min(ys), max(ys)

    # 只保留落在棋盘框内（含框外 0.6 格内）的棋子。board 框的边缘可能有
    # 偏差：实测某模型框左缘比棋子中心靠右 2.3px、某皮肤框底边比底行棋子
    # 中心短 32px（0.55 格）——按原始框过滤会把整条边列/边行的棋子丢掉。
    # 外扩取 0.6 格：够救边行（实测最大偏差 0.55 格），又能挡住棋盘边缘
    # 0.6~0.9 格处的 UI 装饰被回夹成"多出来的炮/相"（实际发生过的误检）
    est_gw = max((bx2 - bx1) / 8.0, 1.0)
    est_gh = max((by2 - by1) / 9.0, 1.0)
    mx, my = est_gw * 0.6, est_gh * 0.6
    inside = [p for p in pieces
              if bx1 - mx <= p.center[0] <= bx2 + mx
              and by1 - my <= p.center[1] <= by2 + my]
    if len(inside) < 2:
        inside = pieces

    # 用棋子中心范围细化棋盘框（跨度足够大时才采用）
    xs = sorted(p.center[0] for p in inside)
    ys = sorted(p.center[1] for p in inside)
    min_dist = _min_pair_distance(inside)
    span_x, span_y = xs[-1] - xs[0], ys[-1] - ys[0]
    if min_dist and span_y >= min_dist * 8.6 and span_x >= min_dist * 7.6:
        bx1, bx2, by1, by2 = xs[0], xs[-1], ys[0], ys[-1]

    width, height = bx2 - bx1, by2 - by1
    if width <= 0 or height <= 0:
        return None
    grid_w, grid_h = width / 8.0, height / 9.0

    # 方向：红帅在上半部 -> 截图红方在上（需在逐子处理前算好，
    # 皮肤模板修正的坐标要转换成 FEN 顺序）
    red_king = next((p for p in inside if p.label == "r_jiang"), None)
    red_at_top = False
    if red_king is not None:
        ky = int(round((red_king.center[1] - by1) / grid_h))
        red_at_top = ky < 5
    else:
        # 没有红帅则按红方棋子总数分布判断
        red_top = sum(1 for p in inside if p.label.startswith("r_")
                      and p.center[1] < (by1 + by2) / 2)
        red_bottom = sum(1 for p in inside if p.label.startswith("r_")
                         and p.center[1] >= (by1 + by2) / 2)
        red_at_top = red_top > red_bottom

    grid: List[List[Optional[str]]] = [[None] * 9 for _ in range(10)]
    # 每格保留得分最高的棋子；重复映射记入冲突数
    best_score: List[List[float]] = [[-1.0] * 9 for _ in range(10)]
    n_ambiguous = 0
    low_conf: List[str] = []
    n_pieces = 0
    _corr_internal = []   # (row, col, score, fen_row, fen_col, 旧, 新, 相关度)
    use_skin = img is not None and skin_templates

    for p in inside:
        cell = map_center_to_cell(p.center[0], p.center[1],
                                  (bx1, by1, bx2, by2), grid_w, grid_h)
        if cell is None:
            continue
        row, col = cell
        letter = CLASS_TO_FEN[p.label]
        # 皮肤模板二次校验：模板意见清晰且与模型不一致时以模板为准
        if use_skin:
            mask = skin.norm_crop(img, p.box)
            if mask is not None:
                pred, corr, margin = skin._match(mask, skin_templates)
                if (pred is not None and corr >= skin.MIN_CORR
                        and margin >= skin.MIN_MARGIN and pred != letter):
                    fen_r, fen_c = ((9 - row, 8 - col) if red_at_top
                                    else (row, col))
                    _corr_internal.append((row, col, p.score,
                                           fen_r, fen_c, letter, pred, corr))
                    letter = pred
        if best_score[row][col] >= 0:
            # 同格已有棋子：只有"两者分数接近且胜出方也不够自信"才算真正歧义，
            # 值得提示用户核对；胜出方明显更可信时按置信度取舍即可。
            if (abs(p.score - best_score[row][col]) < 0.2
                    and max(p.score, best_score[row][col]) < 0.85):
                n_ambiguous += 1
            if p.score <= best_score[row][col]:
                continue
        best_score[row][col] = p.score
        grid[row][col] = letter
        if p.score < 0.75:
            low_conf.append(f"{letter}({p.score:.2f})")

    n_pieces = sum(1 for r in range(10) for c in range(9) if grid[r][c])

    if red_at_top:
        grid = [row[:] for row in grid][::-1]
        for row in grid:
            row.reverse()

    # 只保留"赢得了所在格竞争"的修正：同格竞争落败的检测（例如重复框）
    # 即使被模板纠正也不影响最终棋盘，不应打扰用户
    effective, seen = [], set()
    for (row, col, score, fen_r, fen_c, old, new, corr) in _corr_internal:
        if not (0 <= fen_r < 10 and 0 <= fen_c < 9):
            continue
        if grid[fen_r][fen_c] != new or (fen_r, fen_c) in seen:
            continue
        if abs(best_score[row][col] - score) > 1e-6:
            continue                     # 该检测在同格竞争中落败
        seen.add((fen_r, fen_c))
        effective.append((fen_r, fen_c, old, new, corr))

    return BoardResult(grid=grid, red_at_top=red_at_top,
                       board_box=(bx1, by1, bx2, by2),
                       n_pieces=n_pieces, n_ambiguous=n_ambiguous,
                       low_conf=low_conf,
                       corrections=effective, n_corrected=len(effective))


def grid_plausibility(grid, stm: str = None) -> Tuple[bool, str]:
    """仅凭棋子矩阵的合理性校验（帧完整性门卫）。

    PrintWindow 对硬件渲染窗口偶发返回局部帧（如顶部整行丢失），
    这类残缺帧必须丢弃，绝不能进入同步或被采纳为棋盘。

    stm 为 None 时**不做"非行棋方被将军"检查**——帧本身不含行棋方信息，
    默认按红方判会让"我方正被将军"的正常局面被误判为非法并丢弃
    （实际发生过：对手将军的那一步永远同步不进来）。
    """
    board = Board()
    board.grid = [row[:] for row in grid]
    if stm is None:
        # 仅做与行棋方无关的检查（子力数量/将帅存在与位置/兵卒底线）
        return _grid_checks_without_turn(board)
    if board.find_king(RED) is None or board.find_king(BLACK) is None:
        return False, "缺少将/帅（疑似残缺帧）"
    problem = board.validate_position()
    if problem:
        return False, problem
    return True, ""


def _grid_checks_without_turn(board) -> Tuple[bool, str]:
    """与行棋方无关的局面检查（用于帧校验，帧不含行棋方信息）。"""
    from collections import Counter
    counts = Counter(p for row in board.grid for p in row if p)
    for up, name, lim in (("K", "帅", 1), ("A", "仕", 2), ("B", "相", 2),
                          ("N", "马", 2), ("R", "车", 2), ("C", "炮", 2),
                          ("P", "兵", 5)):
        for letter in (up, up.lower()):
            n = counts.get(letter, 0)
            if n > lim:
                who = "红方" if letter.isupper() else "黑方"
                return False, f"{who}{name}超过 {lim} 个（当前 {n} 个）"
    if counts.get("K", 0) != 1:
        return False, "缺少红帅（疑似残缺帧）"
    if counts.get("k", 0) != 1:
        return False, "缺少黑将（疑似残缺帧）"
    if board.kings_face():
        return False, "将帅照面（同一路无遮挡），不是合法局面"
    for side in (RED, BLACK):
        k = board.find_king(side)
        if k is not None and not in_palace(k[0], k[1], side):
            who = "红方" if side == RED else "黑方"
            return False, f"{who}将/帅不在九宫内"
    for c in range(9):
        if board.grid[9][c] == "P":
            return False, "红兵不可能在己方底线"
        if board.grid[0][c] == "p":
            return False, "黑卒不可能在己方底线"
    return True, ""


def board_plausibility(res: BoardResult) -> Tuple[bool, str]:
    """识别结果是否像一个真实象棋局面（自动找窗口时的防幻觉校验）。

    YOLO 在任意窗口内容上都可能幻觉出棋子，因此窗口探测阶段要求：
    棋盘框长宽比合理、双方将帅都在九宫内、子力数量合法。
    """
    bx1, by1, bx2, by2 = res.board_box
    w, h = bx2 - bx1, by2 - by1
    if h <= 0 or not (0.72 <= w / h <= 1.18):
        return False, f"棋盘框长宽比异常（{w / max(1, h):.2f}）"
    return grid_plausibility(res.grid)      # 帧校验不含行棋方，跳过轮次相关检查


def _min_pair_distance(pieces: List[Detection]) -> Optional[float]:
    """棋子中心之间的最小间距（VinXiangQi 用它判断棋盘跨度是否足够）。"""
    n = len(pieces)
    if n < 2:
        return None
    pts = [p.center for p in pieces]
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
            if best is None or d < best:
                best = d
    return best
