# -*- coding: utf-8 -*-
"""从 GitHub 下载最新版 Pikafish 引擎与 NNUE 权重到 engine/bin/。

用法：
    .venv/Scripts/python tools/download_engine.py

依赖 py7zr（pip install py7zr），用于解压官方 7z 发布包。
"""
import io
import json
import os
import sys
import urllib.request

REPO = "official-pikafish/Pikafish"
BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "bin")


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "xiangqi-assistant"})
    return urllib.request.urlopen(req, timeout=120)


def main():
    os.makedirs(BIN_DIR, exist_ok=True)
    print("查询最新 Release ...")
    with http_get(f"https://api.github.com/repos/{REPO}/releases/latest") as r:
        release = json.load(r)
    tag = release.get("tag_name", "?")
    assets = release.get("assets", [])
    archive = next((a for a in assets if a["name"].endswith(".7z")), None)
    if archive is None:
        print("Release 中未找到 7z 资源:", [a["name"] for a in assets])
        return 1
    url = archive["browser_download_url"]
    print(f"下载 {tag}: {url} ({archive['size'] // 1024 // 1024} MB) ...")
    try:
        with http_get(url) as r:
            data = r.read()
    except Exception as e:
        print("下载失败:", e)
        print("请手动下载后，将 Pikafish-Windows-x86-64-universal.exe 与 pikafish.nnue "
              "解压到 engine/bin/ 目录")
        return 1
    print("解压 ...")
    import py7zr
    with py7zr.SevenZipFile(io.BytesIO(data)) as z:
        names = z.getnames()
        targets = [n for n in names
                   if n == "pikafish.nnue" or
                   ("Windows" in n and n.endswith(".exe"))]
        z.extract(members=targets, path=BIN_DIR)
    print("已安装到", BIN_DIR)
    for n in targets:
        print(" -", os.path.join(BIN_DIR, os.path.basename(n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
