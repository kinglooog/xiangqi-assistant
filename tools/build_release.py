# -*- coding: utf-8 -*-
"""一键打包发布版（象棋接管助手）。

用法：
    .venv/Scripts/python tools/build_release.py

流程：
1. 跑全部单元测试（不通过则中止——不打包有问题的版本）
2. PyInstaller 按 spec 构建（含模型与引擎）
3. 把「使用说明.txt」放进包内
4. 在冻结环境跑自检（识别 + 引擎）
5. 生成 dist/象棋接管助手_发布包.zip
"""
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_NAME = "象棋接管助手"
SPEC = os.path.join(ROOT, f"{APP_NAME}.spec")
DIST = os.path.join(ROOT, "dist")
APP_DIR = os.path.join(DIST, APP_NAME)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, **kw)
    if r.returncode != 0:
        print(f"步骤失败（exit {r.returncode}），已中止", flush=True)
        sys.exit(r.returncode)


def step(n, title):
    print("=" * 60, flush=True)
    print(f"步骤 {n}/5: {title}", flush=True)


def main():
    # 1) 测试
    print("=" * 60)
    step(1, "单元测试")
    run([os.path.join(ROOT, ".venv/Scripts/python"), "-m", "pytest",
         "tests", "-q"])

    # 2) 构建
    print("=" * 60)
    step(2, "PyInstaller 构建（几分钟）")
    if os.path.isdir(APP_DIR):
        shutil.rmtree(APP_DIR)
    run([os.path.join(ROOT, ".venv/Scripts/python"), "-m", "PyInstaller",
         "--noconfirm", "--clean", SPEC])

    # 3) 说明文件
    print("=" * 60)
    step(3, "放入使用说明")
    manual = os.path.join(ROOT, "dist_extra", "使用说明.txt")
    if os.path.exists(manual):
        shutil.copy(manual, os.path.join(APP_DIR, "使用说明.txt"))

    # 4) 冻结环境自检
    print("=" * 60)
    step(4, "冻结环境自检")
    selftest = os.path.join(ROOT, "tools", "frozen_selftest.py")
    shutil.copy(selftest, os.path.join(APP_DIR, "_selftest.py"))
    r = subprocess.run([os.path.join(ROOT, ".venv/Scripts/python"),
                        os.path.join(APP_DIR, "_selftest.py")],
                       cwd=APP_DIR)
    os.remove(os.path.join(APP_DIR, "_selftest.py"))
    if r.returncode != 0:
        print("自检未通过，已中止（发布包保留在 dist/ 下供排查）", flush=True)
        sys.exit(r.returncode)

    # 5) 压缩
    print("=" * 60)
    step(5, "生成 zip")
    zip_base = os.path.join(DIST, f"{APP_NAME}_发布包")
    if os.path.exists(zip_base + ".zip"):
        os.remove(zip_base + ".zip")
    shutil.make_archive(zip_base, "zip", root_dir=DIST, base_dir=APP_NAME)
    size = os.path.getsize(zip_base + ".zip") // 1024 // 1024
    print("=" * 60)
    print(f"完成！发布包: {zip_base}.zip ({size} MB)", flush=True)
    print(f"解压后双击「{APP_NAME}.exe」即可使用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
