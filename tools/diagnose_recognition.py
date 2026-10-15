# -*- coding: utf-8 -*-
"""识别诊断工具：把识别流水线各阶段导出成图片，定位失败环节。

用法：
    .venv/Scripts/python tools/diagnose_recognition.py <截图路径> [输出目录]

会输出：
  01_source.png     原始截图
  02_detections.png YOLO 检测结果叠加（棋盘框 + 每个检测框与类别）
  03_result.txt     识别结果（FEN、棋子数、冲突、低置信度项）
  04_grid.txt       识别矩阵的可视化
  （YOLO 不可用时回退几何方案，输出 02_lines/03_grid/04_warped/05_samples）

怎么看：
  - 02 里棋盘框（黄框）没套住棋盘 -> 定位问题
  - 02 里棋子框位置对但类别错 -> 模型对该皮肤识别不准
  - 03 里"棋子数"远少于实际 -> 模型漏检，考虑换模型或手动摆局
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.pipeline import recognize_screenshot, yolo_available


def _draw_detections(img, res, out_dir):
    vis = img.copy()
    if res.board_box:
        bx1, by1, bx2, by2 = [int(v) for v in res.board_box]
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
        gw, gh = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
        for r in range(10):
            for c in range(9):
                cv2.circle(vis, (int(bx1 + c * gw), int(by1 + r * gh)),
                           2, (255, 0, 255), -1)
    for d in res.detections:
        if d.label == "board":
            continue
        x1, y1, x2, y2 = [int(v) for v in d.box]
        color = (0, 0, 255) if d.label.startswith("r_") else (255, 60, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"{d.label} {d.score:.2f}", (x1, max(9, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
    cv2.imwrite(os.path.join(out_dir, "02_detections.png"), vis)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "diagnose_out"
    os.makedirs(out, exist_ok=True)

    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图片: {src}")
        return 1
    print(f"原图尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"检测模型可用: {yolo_available()}"
          f"{'' if yolo_available() else '（请运行 tools/download_models.py）'}")
    cv2.imwrite(os.path.join(out, "01_source.png"), img)

    res = recognize_screenshot(img)
    if res.source == "yolo":
        _draw_detections(img, res, out)

    lines = [
        f"识别来源: {res.source}",
        f"识别棋子数: {res.n_pieces}",
        f"同格冲突数: {res.n_ambiguous}",
        f"截图红方在上: {res.red_at_top}",
        f"棋盘框: {[round(v) for v in res.board_box] if res.board_box else None}",
        f"提示: {res.warning or '-'}",
        f"错误: {res.error or '-'}",
        "",
        "棋盘矩阵（row0 = 黑方底线，'.' = 空）:",
    ]
    for r in range(10):
        lines.append("  " + "".join(res.grid[r][c] or "." for c in range(9)))
    from recognition.classifier import grid_to_fen
    lines.append("")
    lines.append(f"FEN: {grid_to_fen(res.grid)}")
    text = "\n".join(lines)
    with open(os.path.join(out, "03_result.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n诊断产物已写入 {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

