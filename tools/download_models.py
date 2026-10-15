# -*- coding: utf-8 -*-
"""下载棋子检测模型（来自 VinXiangQi 项目）。

模型权重取自 VinXiangQi 的 Release 包（作者 Vincentzyx，GNU GPL v3.0）：
    https://github.com/Vincentzyx/VinXiangQi/releases

会从 Release zip 中解出 Models/ 下的 .onnx 到本项目的 models/ 目录。

用法：
    .venv/Scripts/python tools/download_models.py
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

REPO = "Vincentzyx/VinXiangQi"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models")
WANT = ("中模型.onnx", "小模型.onnx", "万能带旋转.onnx")


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "xiangqi-assistant"})
    return urllib.request.urlopen(req, timeout=600)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("查询 VinXiangQi 最新 Release ...")
    with http_get(f"https://api.github.com/repos/{REPO}/releases/latest") as r:
        rel = json.load(r)
    assets = rel.get("assets", [])
    zip_asset = next((a for a in assets if a["name"].lower().endswith(".zip")), None)
    if zip_asset is None:
        print("未找到 Release 压缩包:", [a["name"] for a in assets])
        return 1
    url = zip_asset["browser_download_url"]
    print(f"下载 {rel.get('tag_name')}: {url} "
          f"({zip_asset['size'] // 1024 // 1024} MB) ...")
    try:
        with http_get(url) as r:
            data = r.read()
    except Exception as e:
        print("下载失败:", e)
        print("可手动下载该 zip，把其中 Models/*.onnx 解压到 models/ 目录")
        return 1

    print("解出模型文件 ...")
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if name.startswith("Models/") and base.endswith(".onnx"):
                with z.open(name) as src, open(os.path.join(MODELS_DIR, base), "wb") as dst:
                    dst.write(src.read())
                print("  -", base, os.path.getsize(os.path.join(MODELS_DIR, base)) // 1024,
                      "KB")
                n += 1
    if n == 0:
        print("zip 内未找到 .onnx 模型")
        return 1
    print(f"已安装 {n} 个模型到 {MODELS_DIR}")
    print("注意：模型来自 VinXiangQi（GPL-3.0），使用时请遵守其许可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
