# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（象棋接管助手）。

构建：.venv/Scripts/python -m PyInstaller 象棋接管助手.spec
产物：dist/象棋接管助手/（免安装文件夹，整个拷给朋友即可用）
"""
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ---- 需要打进包里的数据 ----
datas = [
    # 识别模型（必需）
    ('models', 'models'),
    # Pikafish 引擎 + NNUE 权重（人机对战/分析必需）
    ('engine/bin', 'engine/bin'),
    # 音效与人声素材
    ('assets/sounds', 'assets/sounds'),
    # 结算界面按钮模板（自动续局用）
    ('assets/jj', 'assets/jj'),
]
binaries = []
hiddenimports = []

# onnxruntime 的 DLL/后端由 collect_all 兜住
for pkg in ('onnxruntime',):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'tests', 'tools', 'samples',
        'tkinter', 'matplotlib', 'IPython', 'jedi', 'pydoc_data',
    ],
    noarchive=False,
)

# ---- 说明：资源路径 ----
# 代码里统一用 os.path.dirname(os.path.abspath(__file__)) 向上找
# models/ 与 engine/bin。PyInstaller 打包后模块的 __file__ 指向
# sys._MEIPASS/...，与 --add-data 的落盘位置（_internal/models 等）一致，
# 因此无需额外运行时钩子。

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='象棋接管助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI 程序，不弹控制台
    disable_windowed_traceback=False,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='象棋接管助手',
)
