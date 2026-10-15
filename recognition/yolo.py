# -*- coding: utf-8 -*-
"""YOLOv5 ONNX 棋子检测（模型来自 VinXiangQi 项目）。

**模型与类别定义归属**：
- 模型权重与类别定义来自 VinXiangQi (https://github.com/Vincentzyx/VinXiangQi)，
  作者 Vincentzyx，以 GNU GPL v3.0 发布。
- 本文件为在 Python/onnxruntime 下重新实现的推理与解码（letterbox / 解码 / NMS），
  以及棋盘定位与格子映射逻辑。

模型规格（三个模型一致）：
- 输入  images [1, 3, 640, 640] float32，letterbox 填充值为 114/255
- 输出  output [1, 25200, 20]，25200 = (80²+40²+20²)×3，
  20 = 4 框参数 + 1 目标置信度 + 15 类别；锚框解码已在模型内部完成，
  坐标为 640×640 输入图上的像素值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 类别顺序（与 YoloXiangQiModel.cs 的 Labels 顺序一致，索引 0-14）
CLASS_NAMES = [
    "b_ma", "b_xiang", "b_shi", "b_jiang", "b_che", "b_pao", "b_bing",
    "r_che", "r_ma", "r_shi", "r_jiang", "r_xiang", "r_pao", "r_bing",
    "board",
]

# 类别 -> FEN 字母（大写红方、小写黑方）
CLASS_TO_FEN = {
    "b_ma": "n", "b_xiang": "b", "b_shi": "a", "b_jiang": "k",
    "b_che": "r", "b_pao": "c", "b_bing": "p",
    "r_ma": "N", "r_xiang": "B", "r_shi": "A", "r_jiang": "K",
    "r_che": "R", "r_pao": "C", "r_bing": "P",
}

INPUT_SIZE = 640
PAD_VALUE = 114


@dataclass
class Detection:
    label: str
    score: float
    # 原图坐标 (x1, y1, x2, y2)
    box: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def default_model_path() -> Optional[str]:
    """在 models/ 下挑选默认模型：优先中模型（精度/速度均衡），
    其次小模型、万能带旋转。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models")
    if not os.path.isdir(base):
        return None
    files = {n: os.path.join(base, n) for n in os.listdir(base)
             if n.endswith(".onnx")}
    if not files:
        return None
    for prefer in ("中模型.onnx", "小模型.onnx"):
        if prefer in files:
            return files[prefer]
    return sorted(files.values())[0]


def _letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    """等比缩放并填充到 size×size，返回 (图, 缩放比, 左填充, 上填充)。"""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """单类 NMS，返回保留的索引。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


class YoloDetector:
    """YOLOv5 ONNX 检测器。线程内复用，非线程安全。"""

    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.5,
                 intra_threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.model_path = model_path

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        """检测一张图，返回所有目标的检测结果（原图坐标）。"""
        lb, scale, dx, dy = _letterbox(img_bgr)
        blob = lb[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        out = self.session.run(None, {self.input_name: blob})[0]
        return self._decode(out[0], scale, dx, dy, img_bgr.shape)

    def _decode(self, pred: np.ndarray, scale: float, dx: int, dy: int,
                shape) -> List[Detection]:
        """pred: [25200, 20] -> Detection 列表（含 NMS）。

        目标置信度与类别分数是分开的两项，最终分数取两者乘积
        （标准 YOLOv5 导出行为）。
        """
        if pred.shape[1] < 6:
            return []
        boxes_xywh = pred[:, :4]
        obj = pred[:, 4]
        cls_scores = pred[:, 5:5 + len(CLASS_NAMES)]
        cls_idx = cls_scores.argmax(axis=1)
        cls_conf = cls_scores[np.arange(len(cls_idx)), cls_idx]
        scores = obj * cls_conf

        mask = scores >= self.conf
        if not mask.any():
            return []
        b = boxes_xywh[mask]
        s = scores[mask]
        c = cls_idx[mask]

        # 还原 letterbox：减去填充、除以缩放比
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        x2 = (cx + w / 2 - dx) / scale
        y2 = (cy + h / 2 - dy) / scale
        ih, iw = shape[:2]
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        results: List[Detection] = []
        for cls in np.unique(c):
            sel = np.where(c == cls)[0]
            keep = _nms(boxes[sel], s[sel], self.iou)
            name = CLASS_NAMES[int(cls)]
            for k in keep:
                idx = sel[k]
                bx = boxes[idx]
                bx = [float(np.clip(bx[0], 0, iw)), float(np.clip(bx[1], 0, ih)),
                      float(np.clip(bx[2], 0, iw)), float(np.clip(bx[3], 0, ih))]
                results.append(Detection(name, float(s[idx]), tuple(bx)))
        results.sort(key=lambda d: -d.score)
        return results
