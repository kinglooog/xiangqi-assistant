# -*- coding: utf-8 -*-
"""基于真实截图的识别回归测试。

测试固件 samples/image-20220711145348990.png 等来自 VinXiangQi 项目仓库
（https://github.com/Vincentzyx/VinXiangQi，GPL-3.0）的 assets 目录，
是真实的「天天象棋」对局截图，其中 45348990 与 45437807 两图由逐子人工核对
确认了正确局面（32 子）。

这些测试保证 YOLO 检测流水线在真实截图上的行为不回退。
若模型文件缺失（models/*.onnx）或 onnxruntime 未安装，相关测试跳过。
"""
import os

import cv2
import pytest

from core.board import Board

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "samples")
# 经人工逐子核对确认的正确局面
VERIFIED_FEN = "2bakabnr/7c1/2n4c1/p1p1p1p1p/1r7/2P6/P3P1P1P/1CN1C1N2/9/1RBAKAB1R"
VERIFIED_IMAGES = ["image-20220711145348990.png", "image-20220711145437807.png"]


def _yolo_ready():
    from recognition.pipeline import yolo_available
    return yolo_available()


requires_model = pytest.mark.skipif(
    not _yolo_ready(), reason="检测模型或 onnxruntime 不可用")


@requires_model
@pytest.mark.parametrize("name", VERIFIED_IMAGES)
def test_real_screenshot_recognized_exactly(name):
    """真实截图应被逐子完全正确识别。"""
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, name)
    if not os.path.exists(path):
        pytest.skip(f"缺少测试截图 {name}")
    img = cv2.imread(path)
    assert img is not None
    res = recognize_screenshot(img)
    assert res.source == "yolo", f"应使用检测模型，实际 {res.source}"
    assert res.red_at_top is False
    board = Board()
    board.grid = [row[:] for row in res.grid]
    assert board.to_fen().split()[0] == VERIFIED_FEN


@requires_model
def test_real_screenshot_board_box_aligned():
    """棋盘定位框应贴合棋盘（宽高比接近 8:9 的格网比例）。"""
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, "image-20220711145348990.png")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    res = recognize_screenshot(cv2.imread(path))
    assert res.board_box is not None
    x1, y1, x2, y2 = res.board_box
    ratio = (x2 - x1) / (y2 - y1)
    assert 0.75 <= ratio <= 1.05, f"棋盘框宽高比异常: {ratio:.2f}"


@requires_model
def test_no_cell_conflicts_on_clean_screenshot():
    """干净截图上不应出现多子落同一格的冲突。"""
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, "image-20220711145348990.png")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    res = recognize_screenshot(cv2.imread(path))
    assert res.n_ambiguous == 0, f"出现 {res.n_ambiguous} 处同格冲突"
    assert res.n_pieces == 32, f"应识别出 32 子，实际 {res.n_pieces}"


@requires_model
def test_recognized_position_is_legal():
    """识别出的局面应通过合法性校验（否则引擎会拒绝工作）。"""
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, "image-20220711145348990.png")
    if not os.path.exists(path):
        pytest.skip("缺少测试截图")
    res = recognize_screenshot(cv2.imread(path))
    board = Board()
    board.grid = [row[:] for row in res.grid]
    assert board.validate_position() is None


# 用户实际使用的 JJ象棋 手机截图（3D 青铜棋子皮肤 + 大量 UI 装饰的残局局面）
# 经逐子人工核对确认：黑方底线 9 子、5 卒、5 兵、红 2 炮、2 俥、
# 底行"黑炮 相 仕 帥 仕 相 黑炮"共 30 子。
JJ_MOBILE_FEN = ("rnbakabnr/9/9/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/"
                 "R7R/1cBAKABc1 w - - 0 1")


@requires_model
def test_jj_mobile_screenshot_recognized_exactly():
    """JJ象棋手机截图（3D 立体棋子 + UI 装饰）应被完全正确识别。

    这是真实使用场景的关键用例：手机竖屏截图，棋盘外有大量
    头像/按钮/装饰元素，棋子是立体青铜风格。
    """
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少 JJ 手机截图固件")
    img = cv2.imread(path)
    assert img is not None
    res = recognize_screenshot(img)
    assert res.source == "yolo"
    assert res.red_at_top is False, "该截图为红方在下"
    assert res.n_pieces == 30
    assert res.n_ambiguous == 0, "不应有歧义检测"
    board = Board()
    board.grid = [row[:] for row in res.grid]
    assert board.to_fen() == JJ_MOBILE_FEN
    assert board.validate_position() is None


@requires_model
def test_jj_mobile_board_box_excludes_ui():
    """棋盘框应只覆盖棋盘，不因上方头像/下方按钮而外扩。"""
    from recognition.pipeline import recognize_screenshot
    path = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
    if not os.path.exists(path):
        pytest.skip("缺少 JJ 手机截图固件")
    img = cv2.imread(path)
    res = recognize_screenshot(img)
    x1, y1, x2, y2 = res.board_box
    h, w = img.shape[:2]
    # 棋盘高度应约为宽度的 1.1 倍（9 格 : 8 格），且不占满整图高度
    box_w, box_h = x2 - x1, y2 - y1
    assert 0.85 <= box_w / box_h <= 1.25, f"棋盘框比例异常 {box_w/box_h:.2f}"
    assert box_h < h * 0.75, "棋盘框过高，可能把 UI 也算进去了"


def test_class_mapping_covers_all_piece_types():
    from recognition.yolo import CLASS_NAMES, CLASS_TO_FEN
    letters = set(CLASS_TO_FEN.values())
    assert letters == {"K", "A", "B", "N", "R", "C", "P",
                       "k", "a", "b", "n", "r", "c", "p"}
    assert len(CLASS_NAMES) == 15 and CLASS_NAMES[-1] == "board"
    assert all(n in CLASS_NAMES for n in CLASS_TO_FEN)


def test_pipeline_falls_back_without_crash_on_blank():
    """空白图不应抛异常，应返回明确错误。"""
    import numpy as np
    from recognition.pipeline import recognize_screenshot
    blank = np.full((300, 300, 3), 255, np.uint8)
    res = recognize_screenshot(blank)
    assert res.n_pieces == 0
    assert res.error or res.source in ("none", "yolo", "geometric")


def test_board_box_edge_inside_piece_centers_keeps_edge_column():
    """board 框边缘压进最外列棋子中心内侧时，边列棋子不得被过滤丢弃。

    实测某检测模型的 board 框左缘比棋子中心靠右 2.3px；按原始框加
    2px 余量过滤会把整条边列的棋子全部丢掉（每条边 4 颗子）。
    """
    from recognition.yolo import CLASS_TO_FEN, Detection
    from recognition.yolo_board import board_from_detections

    fen_rows = ("rnbakabnr", ".........", ".c.....c.", "p.p.p.p.p",
                ".........", ".........", "P.P.P.P.P", ".C.....C.",
                ".........", "RNBAKABNR")
    label_of = {v: k for k, v in CLASS_TO_FEN.items()}
    step = 60.0
    x0, y0 = 80.0, 80.0
    dets = []
    for r, row in enumerate(fen_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            cx, cy = x0 + c * step, y0 + r * step
            dets.append(Detection(label=label_of[ch], score=0.95,
                                  box=(cx - 30, cy - 30, cx + 30, cy + 30)))
    # board 框左/右缘比最外列棋子中心 (80/560) 靠内侧 2.3px
    dets.append(Detection(label="board", score=0.85,
                          box=(82.3, 75.3, 564.5, 629.0)))

    br = board_from_detections(dets, img=None)
    assert br is not None
    assert br.n_pieces == 32, f"边列棋子被过滤丢弃，只识别到 {br.n_pieces} 子"
    assert br.grid[0][0] == "r" and br.grid[9][0] == "R"
    assert br.grid[3][0] == "p" and br.grid[3][8] == "p"
    assert br.red_at_top is False


def test_edge_row_pieces_clamped_not_dropped():
    """棋盘框边行偏差半格内时，贴边棋子回夹到边行而不是丢弃。

    实际发生过：某皮肤 board 框底边比底行棋子中心短 32px，round 出
    row=10 被丢弃——红帅就在底线上，丢了整帧被判"缺少红帅"作废、
    同步卡死（日志里累计 351 次"缺少红帅/黑将"的残缺帧）。
    """
    from recognition.yolo import CLASS_TO_FEN, Detection
    from recognition.yolo_board import board_from_detections, grid_plausibility

    label_of = {v: k for k, v in CLASS_TO_FEN.items()}

    def det(label, cx, cy):
        return Detection(label=label, score=0.95,
                         box=(cx - 30, cy - 30, cx + 30, cy + 30))

    # 棋子稀疏（跨度不足触发框细化）：黑将顶行 (0,3)、红帅底行 (9,4)
    pieces = [det(label_of["k"], 260, 80),     # (0,3)
              det(label_of["K"], 320, 620)]    # (9,4)

    # 底边比真实底行短 32px
    br = board_from_detections(
        pieces + [Detection(label="board", score=0.85,
                            box=(80.0, 80.0, 560.0, 588.0))], img=None)
    assert br is not None
    assert br.grid[0][3] == "k", "贴顶行的黑将不应被丢弃"
    assert br.grid[9][4] == "K", "贴底行的红帅不应被丢弃（round 到第 10 行）"
    assert grid_plausibility(br.grid)[0] is True

    # 顶边比真实顶行低 32px：同理
    br2 = board_from_detections(
        pieces + [Detection(label="board", score=0.85,
                            box=(80.0, 112.0, 560.0, 620.0))], img=None)
    assert br2 is not None
    assert br2.grid[0][3] == "k", "round 到第 -1 行的顶行棋子应回夹到第 0 行"
    assert br2.grid[9][4] == "K"


def test_crop_box_from_dets_math():
    """裁剪框估算：包含全部棋子、外扩约一格半、贴边裁剪。"""
    from recognition.pipeline import _crop_box_from_dets
    from recognition.yolo import Detection
    dets = [Detection("r_che", 0.9, (100, 200, 160, 260)),
            Detection("b_che", 0.9, (400, 200, 460, 260)),
            Detection("r_ma", 0.9, (100, 700, 160, 760)),
            Detection("b_ma", 0.9, (400, 700, 460, 760)),
            Detection("board", 0.9, (0, 0, 10, 10))]
    box = _crop_box_from_dets(dets, (800, 600, 3))
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 <= 0 or box[0] <= 100            # 包含全部棋子
    assert x1 <= 100 and y1 <= 200
    assert x2 >= 460 and y2 == 800             # 贴底边裁剪
    assert (x2 - x1) * (y2 - y1) < 600 * 800 * 0.97
    # 棋子太少或裁剪无意义时返回 None
    assert _crop_box_from_dets(dets[:2], (800, 600, 3)) is None
    full_cover = [Detection("r_che", 0.9, (0, 0, 600, 800)),
                  Detection("b_che", 0.9, (0, 0, 600, 800)),
                  Detection("r_ma", 0.9, (0, 0, 600, 800)),
                  Detection("b_ma", 0.9, (0, 0, 600, 800))]
    assert _crop_box_from_dets(full_cover, (800, 600, 3)) is None


def test_pipeline_crop_retry_uses_cropped_result(monkeypatch):
    """全窗口棋盘框校验不过（皮肤把边框算进去）时，自动裁剪重识别，
    并把坐标换算回原图（代点/采集依赖原图坐标）。"""
    import numpy as np
    import recognition.pipeline as P
    from core.board import Board
    from recognition.yolo import Detection
    from recognition.yolo_board import BoardResult

    full = np.full((900, 600, 3), 40, np.uint8)

    class _Det:
        def detect(self, img):
            # 裁剪后（尺寸不同）返回正常检测
            return [Detection("r_che", 0.9, (10, 10, 40, 40)),
                    Detection("b_che", 0.9, (50, 10, 80, 40)),
                    Detection("r_ma", 0.9, (10, 50, 40, 80)),
                    Detection("b_ma", 0.9, (50, 50, 80, 80))]

    def _fake_from_dets(dets, img=None, skin_templates=None):
        if img is not None and img.shape[:2] == full.shape[:2]:
            # 全窗口：框过高（长宽比异常），将帅缺失
            bad_grid = [[None] * 9 for _ in range(10)]
            bad_grid[0][0] = "r"          # 有棋子、但框长宽比异常
            return BoardResult(grid=bad_grid, red_at_top=False,
                               board_box=(10.0, 10.0, 300.0, 420.0),
                               n_pieces=16, n_ambiguous=0, low_conf=[],
                               corrections=[], n_corrected=0)
        return BoardResult(grid=[row[:] for row in Board().grid],
                           red_at_top=False,
                           board_box=(20.0, 20.0, 260.0, 290.0),
                           n_pieces=32, n_ambiguous=0, low_conf=[],
                           corrections=[], n_corrected=0)

    monkeypatch.setattr(P, "get_detector", lambda mp=None: _Det())
    monkeypatch.setattr(P, "get_skin_templates", lambda p=None: None)
    monkeypatch.setattr(P, "board_from_detections", _fake_from_dets)
    monkeypatch.setattr(P, "_crop_box_from_dets",
                        lambda dets, shape: (50, 60, 550, 800))

    res = P.recognize_screenshot(full)
    assert res.source == "yolo" and res.n_pieces == 32
    assert "裁剪" in (res.warning or ""), "应提示走了裁剪重识别路径"
    assert res.board_box == (70.0, 80.0, 310.0, 350.0), "棋盘框应换算回原图坐标"
    assert res.detections[0].box == (60, 70, 90, 100), "检测框也应换算回原图"


def test_crop_cache_fast_path_and_invalidation(monkeypatch):
    """裁剪成功后续帧走快路径（不再全窗检测）；裁剪失效时清缓存回退。"""
    import numpy as np
    import recognition.pipeline as P
    from core.board import Board
    from recognition.yolo import Detection
    from recognition.yolo_board import BoardResult

    full = np.full((900, 600, 3), 40, np.uint8)
    calls = {"full": 0, "crop": 0, "fail_crop": False}

    class _Det:
        def detect(self, img):
            if img.shape[:2] == full.shape[:2]:
                calls["full"] += 1
            else:
                calls["crop"] += 1
            return [Detection("r_che", 0.9, (10, 10, 40, 40)),
                    Detection("b_che", 0.9, (50, 10, 80, 40)),
                    Detection("r_ma", 0.9, (10, 50, 40, 80)),
                    Detection("b_ma", 0.9, (50, 50, 80, 80))]

    def _fake_from_dets(dets, img=None, skin_templates=None):
        if img is not None and img.shape[:2] == full.shape[:2]:
            grid = [[None] * 9 for _ in range(10)]
            grid[0][0] = "r"
            return BoardResult(grid=grid, red_at_top=False,
                               board_box=(10.0, 10.0, 300.0, 420.0),
                               n_pieces=16, n_ambiguous=0, low_conf=[],
                               corrections=[], n_corrected=0)
        if calls["fail_crop"]:
            return BoardResult(grid=[[None] * 9 for _ in range(10)],
                               red_at_top=False,
                               board_box=(5.0, 5.0, 60.0, 90.0),
                               n_pieces=1, n_ambiguous=0, low_conf=[],
                               corrections=[], n_corrected=0)
        return BoardResult(grid=[row[:] for row in Board().grid],
                           red_at_top=False,
                           board_box=(20.0, 20.0, 260.0, 290.0),
                           n_pieces=32, n_ambiguous=0, low_conf=[],
                           corrections=[], n_corrected=0)

    monkeypatch.setattr(P, "get_detector", lambda mp=None: _Det())
    monkeypatch.setattr(P, "get_skin_templates", lambda p=None: None)
    monkeypatch.setattr(P, "board_from_detections", _fake_from_dets)
    monkeypatch.setattr(P, "_crop_box_from_dets",
                        lambda dets, shape: (50, 60, 550, 800))
    P._crop_cache.update(shape=None, box=None)
    try:
        r1 = P.recognize_screenshot(full)
        assert r1.n_pieces == 32
        assert calls == {"full": 1, "crop": 1, "fail_crop": False}, "首帧：全窗+裁剪各一次"
        # 第二帧：命中缓存 -> 只裁剪检测一次
        r2 = P.recognize_screenshot(full)
        assert r2.n_pieces == 32 and calls["full"] == 1 and calls["crop"] == 2
        # 裁剪失效（画面变了）：清缓存并回退全窗流程
        calls["fail_crop"] = True
        P.recognize_screenshot(full)
        assert calls["full"] == 2, "缓存失效后应回退到全窗检测"
        assert P._crop_cache["box"] is None
    finally:
        P._crop_cache.update(shape=None, box=None)


def test_piece_far_outside_box_excluded():
    """框外 0.6~0.9 格的"棋子"（多半是 UI 装饰误检）不得回夹成边行棋子。

    "红方炮超过 2 个（当前 3 个）"这类误检的一个来源：棋盘边缘外的
    装饰被当棋子并回夹到边行——接纳范围从 0.9 格收紧到 0.6 格。
    """
    from recognition.yolo import CLASS_TO_FEN, Detection
    from recognition.yolo_board import board_from_detections

    label_of = {v: k for k, v in CLASS_TO_FEN.items()}
    box = Detection(label="board", score=0.85, box=(80.0, 80.0, 560.0, 588.0))
    # 底边 588；格高 (588-80)/9 ≈ 56.4 → 0.8 格 ≈ 45px
    far_junk = Detection(label=label_of["C"], score=0.9,
                         box=(290, 603, 350, 663))     # center y=633，框外 0.8 格
    real = Detection(label=label_of["K"], score=0.95,
                     box=(290, 80, 350, 140))          # 正常顶行棋子
    real2 = Detection(label=label_of["k"], score=0.95,
                      box=(170, 80, 230, 140))         # 第二颗正常棋子
    br = board_from_detections([far_junk, real, real2, box], img=None)
    assert br is not None
    assert br.n_pieces == 2, f"框外装饰不应被计入（实际 {br.n_pieces} 子）"
    assert all(br.grid[9][c] is None for c in range(9)), "底行不应出现回夹来的假棋子"


def test_crop_result_touching_edge_rejected(monkeypatch):
    """裁剪识别出的棋盘框贴着裁剪边（棋盘被裁掉一块）→ 不可采用。

    防止"裁剪切掉一列→映射拉伸→看似合法但错误的棋盘"被采用并进缓存。
    """
    import numpy as np
    import recognition.pipeline as P
    from recognition.yolo import Detection
    from recognition.yolo_board import BoardResult

    full = np.full((900, 600, 3), 40, np.uint8)

    class _Det:
        def detect(self, img):
            return [Detection("r_che", 0.9, (10, 10, 40, 40)),
                    Detection("b_che", 0.9, (50, 10, 80, 40)),
                    Detection("r_ma", 0.9, (10, 50, 40, 80)),
                    Detection("b_ma", 0.9, (50, 50, 80, 80))]

    calls = {"crop": 0}

    def _fake_from_dets(dets, img=None, skin_templates=None):
        if img is not None and img.shape[:2] == full.shape[:2]:
            grid = [[None] * 9 for _ in range(10)]
            grid[0][0] = "r"
            return BoardResult(grid=grid, red_at_top=False,
                               board_box=(10.0, 10.0, 300.0, 420.0),
                               n_pieces=16, n_ambiguous=0, low_conf=[],
                               corrections=[], n_corrected=0)
        calls["crop"] += 1
        # 裁剪结果：框贴着裁剪边缘（顶边 y=0）→ 应被拒绝
        from core.board import Board
        return BoardResult(grid=[row[:] for row in Board().grid],
                           red_at_top=False,
                           board_box=(20.0, 0.0, 260.0, 290.0),
                           n_pieces=32, n_ambiguous=0, low_conf=[],
                           corrections=[], n_corrected=0)

    monkeypatch.setattr(P, "get_detector", lambda mp=None: _Det())
    monkeypatch.setattr(P, "get_skin_templates", lambda p=None: None)
    monkeypatch.setattr(P, "board_from_detections", _fake_from_dets)
    monkeypatch.setattr(P, "_crop_box_from_dets",
                        lambda dets, shape: (50, 60, 550, 800))
    P._crop_cache.update(shape=None, box=None)
    try:
        res = P.recognize_screenshot(full)
        assert calls["crop"] == 1, "应尝试过裁剪"
        assert "裁剪" not in (res.warning or ""), "贴边裁剪结果不应被采用"
        assert P._crop_cache["box"] is None, "不可信的裁剪不得进缓存"
    finally:
        P._crop_cache.update(shape=None, box=None)


def test_always_recognizes_on_crop(monkeypatch):
    """统一"先裁剪再识别"：即使整窗结果可信，也优先返回裁剪图结果。"""
    import numpy as np
    import recognition.pipeline as P
    from core.board import Board
    from recognition.yolo import Detection
    from recognition.yolo_board import BoardResult

    full = np.full((900, 600, 3), 40, np.uint8)
    calls = {"full": 0, "crop": 0}

    class _Det:
        def detect(self, img):
            if img.shape[:2] == full.shape[:2]:
                calls["full"] += 1
            else:
                calls["crop"] += 1
            return [Detection("r_che", 0.9, (10, 10, 40, 40)),
                    Detection("b_che", 0.9, (50, 10, 80, 40)),
                    Detection("r_ma", 0.9, (10, 50, 40, 80)),
                    Detection("b_ma", 0.9, (50, 50, 80, 80))]

    marker = [row[:] for row in Board().grid]
    marker[9][0] = None                     # 裁剪结果与整窗结果故意不同

    def _fake_from_dets(dets, img=None, skin_templates=None):
        if img is not None and img.shape[:2] == full.shape[:2]:
            return BoardResult(grid=[row[:] for row in Board().grid],
                               red_at_top=False,
                               board_box=(20.0, 20.0, 500.0, 560.0),
                               n_pieces=32, n_ambiguous=0, low_conf=[],
                               corrections=[], n_corrected=0)
        return BoardResult(grid=[row[:] for row in marker], red_at_top=False,
                           board_box=(20.0, 20.0, 260.0, 290.0),
                           n_pieces=31, n_ambiguous=0, low_conf=[],
                           corrections=[], n_corrected=0)

    monkeypatch.setattr(P, "get_detector", lambda mp=None: _Det())
    monkeypatch.setattr(P, "get_skin_templates", lambda p=None: None)
    monkeypatch.setattr(P, "board_from_detections", _fake_from_dets)
    P._crop_cache.update(shape=None, box=None)
    try:
        res = P.recognize_screenshot(full)
        assert calls["full"] == 1 and calls["crop"] == 1
        assert "裁剪" in (res.warning or ""), "应走裁剪路径"
        assert res.n_pieces == 31, "应采用裁剪图上识别出的结果"
        assert res.grid[9][0] is None
        assert P._crop_cache["box"] is not None, "应记住紧凑裁剪框"
    finally:
        P._crop_cache.update(shape=None, box=None)
