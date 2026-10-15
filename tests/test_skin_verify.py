# -*- coding: utf-8 -*-
"""皮肤模板（自校准 + 二次校验）的回归测试。

场景：模型对某些棋子皮肤会有系统性误判（車→象、車→炮），
皮肤模板用"用户确认过的局面"采集参照，之后对每个检出棋子做
二次校验。测试覆盖：采集/存取往返、校验修正、跨皮肤安全、
同格落败检测的修正不产生误报。
"""
import os

import cv2
import numpy as np
import pytest

from core.board import Board

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "samples")
JJ = os.path.join(SAMPLES, "jj_endgame_mobile.jpg")
JJ_FEN = ("rnbakabnr/9/9/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/"
          "R7R/1cBAKABc1 w - - 0 1")
TT = os.path.join(SAMPLES, "image-20220711145348990.png")

requires_jj = pytest.mark.skipif(not os.path.exists(JJ),
                                 reason="缺少 JJ 截图固件")


@pytest.fixture
def jj_img(qapp):
    return cv2.imread(JJ)


@pytest.fixture
def jj_harvest(qapp, jj_img):
    """从 JJ 截图采集模板（标签用人工核对的真值局面）。"""
    from recognition import skin
    from recognition.yolo import YoloDetector
    from recognition.yolo_board import board_from_detections
    det = YoloDetector(det_path(), conf=0.35)
    dets = det.detect(jj_img)
    br = board_from_detections(dets)
    grid = [[p for p in row] for row in Board(JJ_FEN).grid]
    tpl = skin.harvest(jj_img, dets, br.board_box, br.red_at_top, grid)
    return tpl, det, dets, br


def det_path():
    from recognition.yolo import default_model_path
    return default_model_path()


# ---------------------------------------------------------------- 存取往返
def test_templates_save_load_roundtrip(qapp, jj_harvest, tmp_path):
    from recognition import skin
    tpl, *_ = jj_harvest
    p = str(tmp_path / "skin.npz")
    skin.save_templates(p, tpl)
    loaded = skin.load_templates(p)
    assert sorted(loaded) == sorted(tpl)
    for k in tpl:
        assert len(loaded[k]) == len(tpl[k])
        for a, b in zip(loaded[k], tpl[k]):
            assert np.array_equal(a, b)


def test_load_missing_returns_none(qapp, tmp_path):
    from recognition import skin
    assert skin.load_templates(str(tmp_path / "无.npz")) is None


def test_harvest_caps_per_class(qapp, jj_harvest):
    """每类模板数量有上限，避免无限膨胀。"""
    from recognition import skin
    tpl, *_ = jj_harvest
    assert all(len(v) <= skin.MAX_PER_CLASS for v in tpl.values())


# ---------------------------------------------------------------- 校验行为
@requires_jj
def test_verify_no_correction_when_templates_agree(qapp, jj_harvest, jj_img):
    """中模型在 JJ 截图上本就全对：模板校验不应产生任何修正。"""
    from recognition import skin
    from recognition.yolo_board import board_from_detections
    tpl, det, dets, br = jj_harvest
    r = board_from_detections(dets, img=jj_img, skin_templates=tpl)
    assert r.n_corrected == 0
    gt = [[p for p in row] for row in Board(JJ_FEN).grid]
    assert r.grid == gt


@requires_jj
def test_verify_fixes_confused_model(qapp, jj_harvest, jj_img):
    """用易混淆的小模型：模板校验应减少识别错误（复现并修复 車/炮 类混淆）。"""
    import os as _os
    from recognition import skin
    from recognition.yolo import YoloDetector
    from recognition.yolo_board import board_from_detections
    tpl, *_ = jj_harvest
    small = _os.path.join(_os.path.dirname(det_path()), "小模型.onnx")
    if not _os.path.exists(small):
        pytest.skip("缺少小模型")
    gt = [[p for p in row] for row in Board(JJ_FEN).grid]
    det = YoloDetector(small, conf=0.35)
    dets = det.detect(jj_img)
    r0 = board_from_detections(dets)
    r1 = board_from_detections(dets, img=jj_img, skin_templates=tpl)
    err0 = sum(1 for rr in range(10) for cc in range(9)
               if r0.grid[rr][cc] != gt[rr][cc])
    err1 = sum(1 for rr in range(10) for cc in range(9)
               if r1.grid[rr][cc] != gt[rr][cc])
    assert err1 < err0, f"模板校验应减少错误（{err0} -> {err1}）"
    assert r1.n_corrected >= 1


@requires_jj
def test_verify_safe_across_skins(qapp, jj_harvest):
    """JJ 模板用于天天象棋截图：不得修改任何棋子（跨皮肤安全）。"""
    from recognition import skin
    from recognition.yolo import YoloDetector
    from recognition.yolo_board import board_from_detections
    tpl, *_ = jj_harvest
    tt = cv2.imread(TT)
    if tt is None:
        pytest.skip("缺少天天象棋截图")
    TT_FEN = ("2bakabnr/7c1/2n4c1/p1p1p1p1p/1r7/2P6/P3P1P1P/"
              "1CN1C1N2/9/1RBAKAB1R w - - 0 1")
    gt = [[p for p in row] for row in Board(TT_FEN).grid]
    det = YoloDetector(det_path(), conf=0.35)
    r = board_from_detections(det.detect(tt), img=tt, skin_templates=tpl)
    assert r.n_corrected == 0, "跨皮肤时模板不应干预"
    assert r.grid == gt


# ---------------------------------------------------------------- 管线集成
@requires_jj
def test_pipeline_with_skin_templates(qapp, jj_harvest, tmp_path, monkeypatch):
    """recognize_screenshot 读取皮肤模板文件并完成校验（端到端）。"""
    from recognition import skin
    from recognition import pipeline
    tpl, *_ = jj_harvest
    p = str(tmp_path / "skin.npz")
    skin.save_templates(p, tpl)
    monkeypatch.setattr(pipeline, "get_skin_templates", lambda path=None: tpl)
    img = cv2.imread(JJ)
    res = pipeline.recognize_screenshot(img)
    assert res.source == "yolo"
    assert res.skin_classes == len(tpl)
    gt = [[x for x in row] for row in Board(JJ_FEN).grid]
    assert res.grid == gt


def test_norm_crop_basic(qapp):
    """norm_crop：正常棋子（浅色棋面 + 深色笔画）返回 64x64 掩码；极小框 None。"""
    from recognition import skin
    img = np.full((100, 100, 3), 60, np.uint8)          # 深色背景
    cv2.circle(img, (50, 50), 32, (210, 210, 210), -1)  # 浅色棋面
    # 深色"笔画"：十字 + 两点
    cv2.rectangle(img, (46, 30), (54, 70), (25, 25, 25), -1)
    cv2.rectangle(img, (30, 46), (70, 54), (25, 25, 25), -1)
    m = skin.norm_crop(img, (18, 18, 82, 82))
    assert m is not None and m.shape == (64, 64)
    assert m.sum() > 200, "掩码应包含深色笔画"
    assert skin.norm_crop(img, (49, 49, 50, 50)) is None


def test_harvest_uses_same_mapping_for_edge_rows():
    """模板采集与识别用同一套单元格映射：贴底行的棋子也应被采集。

    某皮肤 board 框底边比底行棋子中心短 32px，采集端此前直接按
    round 越界丢弃——红帅在底线上，模板永远学不到（还可能把邻格
    棋子采成本格类别）。修复后两边共用 map_center_to_cell。
    """
    from recognition.yolo import CLASS_TO_FEN, Detection
    from recognition import skin

    rng = np.random.default_rng(7)
    img = rng.integers(150, 250, (700, 640, 3), dtype=np.uint8)
    label_of = {v: k for k, v in CLASS_TO_FEN.items()}
    pieces = [
        Detection(label=label_of["K"], score=0.95,
                  box=(290, 590, 350, 650)),      # 红帅 (9,4) center=(320,620)
        Detection(label=label_of["k"], score=0.95,
                  box=(230, 50, 290, 110)),       # 黑将 (0,3) center=(260,80)
    ]
    grid = [[None] * 9 for _ in range(10)]
    grid[9][4] = "K"
    grid[0][3] = "k"
    # 框底边比真实底行短 32px（588 而非 620），顶边对齐
    tpl = skin.harvest(img, pieces, (80.0, 80.0, 560.0, 588.0), False, grid)
    assert "K" in tpl, "贴底行的红帅模板不应被丢弃"
    assert "k" in tpl, "贴顶行的黑将模板不应被丢弃"
