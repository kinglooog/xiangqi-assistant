# -*- coding: utf-8 -*-
"""皮肤模板校验的端到端评测。

1) 从人工核对过的 JJ 截图采集模板（模拟用户确认）；
2) 各模型在 JJ 扰动序列上的错误是否被模板修正；
3) 跨皮肤安全：JJ 模板用于天天象棋截图时不得乱改。
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.board import Board
from recognition import skin
from recognition.yolo import YoloDetector
from recognition.yolo_board import board_from_detections

JJ = os.path.join("samples", "jj_endgame_mobile.jpg")
TT = os.path.join("samples", "image-20220711145348990.png")
JJ_FEN = ("rnbakabnr/9/9/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/"
          "R7R/1cBAKABc1 w - - 0 1")
TT_FEN = ("2bakabnr/7c1/2n4c1/p1p1p1p1p/1r7/2P6/P3P1P1P/"
          "1CN1C1N2/9/1RBAKAB1R w - - 0 1")
MODELS = ["中模型.onnx", "万能带旋转.onnx", "小模型.onnx"]


def gt(fen):
    return [[p for p in row] for row in Board(fen).grid]


def cases(img):
    out = [("原图", img)]
    for s in (0.75, 0.55):
        out.append((f"缩放x{s}", cv2.resize(img, None, fx=s, fy=s,
                                            interpolation=cv2.INTER_AREA)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 45])
    out.append(("JPEGq45", cv2.imdecode(enc, cv2.IMREAD_COLOR)))
    out.append(("亮度65%", cv2.convertScaleAbs(img, alpha=0.65)))
    return out


def main():
    jj = cv2.imread(JJ)
    tt = cv2.imread(TT)
    assert jj is not None and tt is not None

    # 1) 采集模板（用真值模拟"用户确认过的局面"）
    det = YoloDetector(os.path.join("models", "中模型.onnx"), conf=0.35)
    dets = det.detect(jj)
    br = board_from_detections(dets)
    # 先关模板采集前的基础识别结果应与真值一致（此前已验证）
    assert br is not None and br.board_box
    tpl_path = os.path.join(tempfile.gettempdir(), "skin_eval.npz")
    tpl = skin.harvest(jj, dets, br.board_box, br.red_at_top, gt(JJ_FEN))
    skin.save_templates(tpl_path, tpl)
    print(f"模板采集: {len(tpl)} 类 "
          f"{sorted((k, len(v)) for k, v in tpl.items())}")

    # 2) 各模型 × JJ 扰动：无模板 vs 有模板
    jj_gt = gt(JJ_FEN)
    for model in MODELS:
        d = YoloDetector(os.path.join("models", model), conf=0.35)
        for use_tpl in (False, True):
            t = skin.load_templates(tpl_path) if use_tpl else None
            errs, fixes, total_corr = 0, 0, 0
            for name, im in cases(jj):
                r = board_from_detections(d.detect(im), img=im,
                                          skin_templates=t)
                assert r is not None
                total_corr += r.n_corrected
                for rr in range(10):
                    for cc in range(9):
                        if r.grid[rr][cc] != jj_gt[rr][cc]:
                            errs += 1
            tag = "有模板" if use_tpl else "无模板"
            print(f"[JJ/{model:10s}] {tag}: 错 {errs} 格, 模板修正 {total_corr} 处")

    # 3) 跨皮肤安全：JJ 模板 × 天天象棋截图
    tt_gt = gt(TT_FEN)
    d = YoloDetector(os.path.join("models", "中模型.onnx"), conf=0.35)
    t = skin.load_templates(tpl_path)
    for use_tpl in (False, True):
        r = board_from_detections(d.detect(tt), img=tt,
                                  skin_templates=(t if use_tpl else None))
        same = all(r.grid[rr][cc] == tt_gt[rr][cc]
                   for rr in range(10) for cc in range(9))
        print(f"[天天象棋/中模型] {'有JJ模板' if use_tpl else '无模板'}: "
              f"识别与真值一致={same}, 模板修正 {r.n_corrected} 处")

    os.remove(tpl_path)


if __name__ == "__main__":
    main()
