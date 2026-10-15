# -*- coding: utf-8 -*-
"""JJ 截图识别评测：三模型对比 + 扰动鲁棒性。

真值来自人工逐子核对的 samples/jj_endgame_mobile.jpg。

用法：
    .venv/Scripts/python tools/eval_jj.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.board import Board
from recognition.yolo import YoloDetector
from recognition.yolo_board import board_from_detections

SAMPLE = os.path.join("samples", "jj_endgame_mobile.jpg")
GT_FEN = ("rnbakabnr/9/9/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/"
          "R7R/1cBAKABc1 w - - 0 1")
MODELS = ["中模型.onnx", "万能带旋转.onnx", "小模型.onnx"]


def perturbations(img):
    """生成 (名称, 图像) 评测集：原图 + 常见截屏差异。"""
    out = [("原图", img)]
    for s in (0.75, 0.55):
        out.append((f"缩放x{s}", cv2.resize(img, None, fx=s, fy=s,
                                            interpolation=cv2.INTER_AREA)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 45])
    out.append(("JPEG q45", cv2.imdecode(enc, cv2.IMREAD_COLOR)))
    bright = cv2.convertScaleAbs(img, alpha=0.65, beta=0)   # 变暗
    out.append(("亮度65%", bright))
    return out


def gt_grid():
    b = Board(GT_FEN)
    return [[p for p in row] for row in b.grid]


def main():
    img = cv2.imread(SAMPLE)
    assert img is not None
    gt = gt_grid()
    cases = perturbations(img)

    for model_name in MODELS:
        path = os.path.join("models", model_name)
        det = YoloDetector(path, conf=0.35, iou=0.5)
        total_err = 0
        errs_by_class = {}
        t0 = time.time()
        for name, im in cases:
            res = board_from_detections(det.detect(im))
            if res is None:
                print(f"  [{model_name}] {name}: 未定位到棋盘")
                total_err += 32
                continue
            n_err = 0
            for r in range(10):
                for c in range(9):
                    if res.grid[r][c] != gt[r][c]:
                        n_err += 1
                        key = f"{gt[r][c]}->{res.grid[r][c]}"
                        errs_by_class[key] = errs_by_class.get(key, 0) + 1
            total_err += n_err
            print(f"  [{model_name}] {name:10s} 错 {n_err:2d} 格"
                  + (f"  {dict((k,v) for k,v in errs_by_class.items())}" if n_err else ""))
        dt = (time.time() - t0) / len(cases) * 1000
        print(f"== {model_name}: 6 例共错 {total_err} 格, 平均 {dt:.0f} ms/张\n")


if __name__ == "__main__":
    main()
