# -*- coding: utf-8 -*-
"""Win32 窗口枚举与矩形获取（对局窗口连接用）。

只依赖 ctypes 与系统 API，负责：
- 列出可见的顶层窗口（供用户选择 JJ 象棋对局所在窗口）
- 取窗口的物理像素矩形（与 mss 抓屏坐标系一致，含 DPI 处理）
- 最小化判断
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import time

import numpy as np
from typing import List, Optional, Tuple

DWMWA_EXTENDED_FRAME_BOUNDS = 9


def set_dpi_aware() -> None:
    """确保 GetWindowRect 返回物理像素（与 mss 一致）。

    Qt6 默认已设 per-monitor v2 DPI 感知，此时不要再设（重复设置会
    产生"拒绝访问"警告）；仅独立脚本（无 Qt）场景下补设。
    """
    if sys.platform != "win32":
        return
    try:
        awareness = ctypes.c_int(0)
        if ctypes.windll.shcore.GetProcessDpiAwareness(
                None, ctypes.byref(awareness)) == 0 and awareness.value != 0:
            return                      # 已有 DPI 感知
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def _user32():
    return ctypes.windll.user32


def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """窗口客户区外沿的物理像素矩形 (left, top, right, bottom)。"""
    u32 = _user32()
    # 优先用 DWM 扩展框（不含不可见的阴影边）
    rect = wt.RECT()
    try:
        dwm = ctypes.windll.dwmapi
        if dwm.DwmGetWindowAttribute(wt.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                     ctypes.byref(rect), ctypes.sizeof(rect)) == 0:
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass
    if not u32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def is_minimized(hwnd: int) -> bool:
    return bool(_user32().IsIconic(wt.HWND(hwnd)))


def is_visible(hwnd: int) -> bool:
    return bool(_user32().IsWindowVisible(wt.HWND(hwnd)))


def is_window_valid(hwnd: int) -> bool:
    """句柄是否仍指向存在的窗口（MuMu 重启/窗口重建后旧句柄会失效）。"""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        return bool(_user32().IsWindow(wt.HWND(hwnd)))
    except Exception:
        return False


def window_title(hwnd: int) -> str:
    """取窗口标题（诊断日志用）。"""
    if sys.platform != "win32" or not hwnd:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        _user32().GetWindowTextW(wt.HWND(hwnd), buf, 256)
        return buf.value
    except Exception:
        return ""


def restore_window(hwnd: int) -> bool:
    """恢复最小化的窗口（SW_RESTORE）。返回是否已不在最小化状态。"""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        u32 = _user32()
        u32.ShowWindow(wt.HWND(hwnd), 9)          # SW_RESTORE
        time.sleep(0.3)
        return not bool(u32.IsIconic(wt.HWND(hwnd)))
    except Exception:
        return False


def list_all_toplevel_windows() -> List[dict]:
    """诊断用：枚举全部顶层窗口（含不可见/工具窗口/无标题），不做任何过滤。"""
    results: List[dict] = []
    u32 = _user32()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def on_window(hwnd, _lparam):
        pid = wt.DWORD(0)
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True
        length = u32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(max(length + 1, 2))
        if length > 0:
            u32.GetWindowTextW(hwnd, buf, length + 1)
        rect = get_window_rect(hwnd) or (0, 0, 0, 0)
        cls = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(hwnd, cls, 256)
        ex = u32.GetWindowLongW(hwnd, -20)
        results.append({"hwnd": int(hwnd),
                        "title": buf.value.strip(),
                        "rect": (rect[0], rect[1], rect[2], rect[3]),
                        "visible": is_visible(hwnd),
                        "minimized": is_minimized(hwnd),
                        "toolwindow": bool(ex & 0x80),
                        "pid": int(pid.value),
                        "cls": cls.value})
        return True

    u32.EnumWindows(on_window, 0)
    return results


def list_windows() -> List[dict]:
    """枚举可见顶层窗口（含无标题窗口；模拟器主窗口可能没有标题）。"""
    results: List[dict] = []
    u32 = _user32()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def on_window(hwnd, _lparam):
        if not is_visible(hwnd):
            return True
        pid = wt.DWORD(0)
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        # 跳过不可见工具窗口（WS_EX_TOOLWINDOW）
        ex = u32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        if ex & 0x80:
            return True
        # 标题允许为空（部分模拟器主窗口标题为空，靠内容识别兜底）
        length = u32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(max(length + 1, 2))
        if length > 0:
            u32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        rect = get_window_rect(hwnd)
        if rect is None or rect[2] - rect[0] < 200 or rect[3] - rect[1] < 150:
            return True
        cls = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(hwnd, cls, 256)
        results.append({"hwnd": int(hwnd), "title": title, "rect": rect,
                        "minimized": is_minimized(hwnd), "pid": int(pid.value),
                        "cls": cls.value})
        return True

    u32.EnumWindows(on_window, 0)
    return results


def get_client_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """窗口客户区的物理像素矩形 (left, top, right, bottom)。

    比 DWM 扩展边界更贴合内容（不含边框/阴影，高 DPI 下不漏缝），
    抓屏与点击坐标都应基于它。
    """
    u32 = _user32()
    rc = wt.RECT()
    if not u32.GetClientRect(wt.HWND(hwnd), ctypes.byref(rc)):
        return None
    pt = wt.POINT(0, 0)
    if not u32.ClientToScreen(wt.HWND(hwnd), ctypes.byref(pt)):
        return None
    return (pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom)


GA_ROOT = 2


def window_root_at_point(x: int, y: int) -> int:
    """屏幕物理像素 (x, y) 处最顶层窗口的"根窗口"句柄。

    用 WindowFromPoint + GetAncestor(GA_ROOT)：即使目标是子窗口
    （MuMu 的渲染窗等），也返回其顶层祖先，便于与接管句柄比对。
    """
    pt = wt.POINT(int(x), int(y))
    h = _user32().WindowFromPoint(pt)
    if not h:
        return 0
    root = _user32().GetAncestor(h, GA_ROOT)
    return int(root or h or 0)


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def process_exe(hwnd: int) -> str:
    """窗口所属进程的 exe 完整路径（失败返回空串）。"""
    pid = wt.DWORD(0)
    _user32().GetWindowThreadProcessId(wt.HWND(hwnd), ctypes.byref(pid))
    if not pid.value:
        return ""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return ""
    try:
        size = wt.DWORD(512)
        buf = ctypes.create_unicode_buffer(size.value)
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        k32.CloseHandle(h)


def get_foreground_window() -> int:
    return int(_user32().GetForegroundWindow() or 0)


def set_foreground(hwnd: int) -> bool:
    """把窗口拉到前台；返回是否成功（GetForegroundWindow 校验）。"""
    u32 = _user32()
    u32.ShowWindow(wt.HWND(hwnd), 5)          # SW_SHOW
    u32.SetForegroundWindow(wt.HWND(hwnd))
    return get_foreground_window() == hwnd


PW_CLIENTONLY = 0x0001
PW_RENDERFULLCONTENT = 0x0002


def capture_window_client(hwnd: int) -> Optional[np.ndarray]:
    """用 PrintWindow 直接读取窗口客户区内容（BGR ndarray）。

    与抓屏幕区域不同：窗口被其它窗口遮挡、不在前台也能读到它自己的
    内容，不会被重叠窗口污染。MuMu 等硬件渲染窗口依赖
    PW_RENDERFULLCONTENT（Win8.1+）。失败或全黑时返回 None。
    """
    if sys.platform != "win32":
        return None
    rc = get_client_rect(hwnd)
    if rc is None:
        return None
    w, h = rc[2] - rc[0], rc[3] - rc[1]
    if w < 50 or h < 50:
        return None
    u32, gdi = _user32(), ctypes.windll.gdi32
    hdc_window = u32.GetDC(wt.HWND(hwnd))
    if not hdc_window:
        return None
    hdc_mem = gdi.CreateCompatibleDC(hdc_window)
    bmp = gdi.CreateCompatibleBitmap(hdc_window, w, h)
    old = gdi.SelectObject(hdc_mem, bmp)
    try:
        ok = u32.PrintWindow(wt.HWND(hwnd), hdc_mem,
                             PW_CLIENTONLY | PW_RENDERFULLCONTENT)
        class BMIHEADER(ctypes.Structure):
            _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                        ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                        ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                        ("biYPelsPerMeter", ctypes.c_int32),
                        ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]

        class BMI(ctypes.Structure):
            _fields_ = [("bmiHeader", BMIHEADER), ("bmiColors", ctypes.c_uint32 * 3)]

        bmi = BMI()
        bmi.bmiHeader.biSize = ctypes.sizeof(BMIHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h          # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0      # BI_RGB
        buf = np.empty((h, w, 4), np.uint8)
        got = gdi.GetDIBits(hdc_mem, bmp, 0, h,
                            buf.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(bmi), 0)   # DIB_RGB_COLORS
        if not ok or not got:
            return None
        img = buf[:, :, :3]                       # BGRA -> BGR
        if float(img.std()) < 2.0:                # 全黑/纯色：渲染失败
            return None
        return np.ascontiguousarray(img)
    finally:
        gdi.SelectObject(hdc_mem, old)
        gdi.DeleteObject(bmp)
        gdi.DeleteDC(hdc_mem)
        u32.ReleaseDC(wt.HWND(hwnd), hdc_window)


HWND_TOPMOST = -1
HWND_NOTMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001


def set_topmost(hwnd: int, on: bool) -> None:
    """把窗口置顶/取消置顶（接管时保证对局窗口可见、点击可命中）。"""
    if sys.platform != "win32":
        return
    _user32().SetWindowPos(wt.HWND(hwnd), wt.HWND(HWND_TOPMOST if on else HWND_NOTMOST),
                           0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)


# ------------------------------------------------------------------ 点击
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


# ---------------------------------------------------------------- 拟人化鼠标
MOUSEEVENTF_MOVE = 0x0001

# 由界面设置的拟人化参数（避免把参数层层传到各调用点）
HUMAN_INPUT = {"enabled": True, "method": "jerk", "speed": "normal"}
_SPEED_FACTOR = {"fast": 0.6, "normal": 1.0, "slow": 1.5}
_TARGET_SIZE = 50.0          # 目标（棋子）直径估计，用于菲茨定律


def cursor_pos() -> Tuple[int, int]:
    pt = wt.POINT()
    _user32().GetCursorPos(ctypes.byref(pt))
    return (int(pt.x), int(pt.y))


def _windmouse(x0: float, y0: float, x1: float, y1: float, rng,
               gravity: float = 9.0, wind: float = 3.0,
               max_step: float = 12.0, target_area: float = 8.0) -> List[Tuple[int, int]]:
    """WindMouse 物理轨迹：重力吸引 + 随机风扰 + 摩擦，产生弧线与过冲回修。"""
    import math
    sqrt3, sqrt5 = math.sqrt(3), math.sqrt(5)
    vx = vy = wx = wy = 0.0
    cx, cy = x0, y0
    pts: List[Tuple[int, int]] = [(int(round(cx)), int(round(cy)))]
    while math.hypot(x1 - cx, y1 - cy) > 1.0 and len(pts) < 400:
        d = max(1e-6, math.hypot(x1 - cx, y1 - cy))
        w = min(wind, d)
        if d >= target_area:
            wx = wx / sqrt3 + (w * 2 * rng.random() - w) / sqrt5
            wy = wy / sqrt3 + (w * 2 * rng.random() - w) / sqrt5
            vx += wx + gravity * (x1 - cx) / d
            vy += wy + gravity * (y1 - cy) / d
        else:
            vx += gravity * 3 * (x1 - cx) / d
            vy += gravity * 3 * (y1 - cy) / d
        v_mag = math.hypot(vx, vy)
        if v_mag > max_step:
            clip = max_step / 2 + (max_step / 2) * rng.random()
            vx = vx / v_mag * clip
            vy = vy / v_mag * clip
        cx += vx
        cy += vy
        pts.append((int(round(cx)), int(round(cy))))
    pts.append((int(round(x1)), int(round(y1))))
    return pts


def plan_path(frm: Point, to: Point, method: str = "jerk",
              speed: str = "normal", rng=None) -> List[Tuple[int, int]]:
    """规划拟人鼠标轨迹（纯计算，便于测试）。返回含起点与终点的坐标序列。

    method: 'jerk' 最小急动度（平滑、带微抖动、少过冲）
            'wind' WindMouse 物理模拟（弧线明显、带过冲回修）
    speed : 'fast' / 'normal' / 'slow'，时长按菲茨定律估算后按档位缩放
    """
    import math
    import random
    rng = rng or random.Random()
    x0, y0 = float(frm[0]), float(frm[1])
    x1, y1 = float(to[0]), float(to[1])
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 2:
        return [(int(round(x1)), int(round(y1)))]
    # 时长按菲茨定律：距离越远 / 目标越小越慢
    dur = 0.10 + 0.15 * math.log2(dist / _TARGET_SIZE + 1.0)
    dur = max(0.15, min(0.85, dur)) * _SPEED_FACTOR.get(speed, 1.0)

    if method == "wind":
        raw = _windmouse(x0, y0, x1, y1, rng)
        n = max(8, min(70, int(dur / 0.011)))      # 与最小急动度同一配速口径
        pts = _resample_polyline(raw, n)
    else:
        n = max(8, min(80, int(dur / 0.011)))          # ~90Hz 事件频率
        # 垂直方向的弧线偏置（真人轨迹不是直线）
        amp = min(dist * 0.06, 14.0) * rng.uniform(0.35, 1.0)
        if rng.random() < 0.5:
            amp = -amp
        ux, uy = (-dy / dist, dx / dist)
        pts = []
        for i in range(n + 1):
            t = i / n
            s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5   # 最小急动度位置曲线
            off = amp * math.sin(math.pi * t)            # 两端为 0、中间最大
            px, py = x0 + dx * s + ux * off, y0 + dy * s + uy * off
            if 0 < i < n:                                # 手部微颤
                px += rng.gauss(0, 0.9)
                py += rng.gauss(0, 0.9)
            pts.append((int(round(px)), int(round(py))))
    pts[-1] = (int(round(x1)), int(round(y1)))           # 精确落到目标
    return pts


def _resample_polyline(pts: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """沿折线按等弧长重采样为 n+1 个点（保留原路径的弯曲形状）。"""
    import math
    if len(pts) < 2 or n < 1:
        return list(pts)
    segs = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)]
    total = sum(segs)
    if total <= 0:
        return [pts[0], pts[-1]]
    out = [pts[0]]
    step = total / n
    acc = 0.0
    seg_i = 0
    seg_pos = 0.0
    for k in range(1, n):
        need = step
        while seg_i < len(segs) and seg_pos + need > segs[seg_i]:
            need -= segs[seg_i] - seg_pos
            seg_i += 1
            seg_pos = 0.0
        if seg_i >= len(segs):
            break
        seg_pos += need
        x0, y0 = pts[seg_i]
        x1, y1 = pts[seg_i + 1]
        t = seg_pos / segs[seg_i] if segs[seg_i] > 0 else 0.0
        out.append((int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))))
    out.append(pts[-1])
    return out


def _emit_move(dx: int, dy: int) -> None:
    """发出一个相对鼠标移动事件（真实输入流，非直接改坐标）。"""
    _user32().mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def human_move_to(x: int, y: int, method: str = "jerk",
                  speed: str = "normal", rng=None) -> None:
    """按拟人轨迹把光标移到 (x, y)。

    每步用相对移动事件，并按"当前实际位置"重算增量，
    因此自带的指针加速/缩放不会让路径跑偏。
    """
    import random
    import time
    import math
    rng = rng or random.Random()
    start = cursor_pos()
    pts = plan_path(start, (x, y), method, speed, rng)
    if len(pts) < 2:
        return
    # 总时长同样按菲茨定律，均匀铺到各步（含 20% 随机抖动），风速轨迹也随档位变化
    dist = math.hypot(x - start[0], y - start[1])
    dur = 0.10 + 0.15 * math.log2(dist / _TARGET_SIZE + 1.0)
    dur = max(0.15, min(0.85, dur)) * _SPEED_FACTOR.get(speed, 1.0)
    delay = max(0.004, min(0.030, dur / (len(pts) - 1)))
    for (px, py) in pts[1:]:
        cx, cy = cursor_pos()
        if px != cx or py != cy:
            _emit_move(px - cx, py - cy)
        time.sleep(delay * rng.uniform(0.75, 1.35))


def click_at(x: int, y: int) -> None:
    """把鼠标移到屏幕物理像素 (x, y) 并左键单击。

    默认走拟人轨迹（可在设置里关闭）：轨迹 -> 悬停 -> 按下 -> 按住 -> 抬起。
    速度档 "ultra"（极快）：不模拟轨迹，瞬时定位 + 直接按下抬起。
    """
    import random
    if sys.platform != "win32":
        return
    u32 = _user32()
    rng = random.Random()
    cfg = HUMAN_INPUT
    if cfg.get("enabled", True):
        if cfg.get("speed") == "ultra":
            # 极快：不模拟轨迹，光标瞬移到位直接点（两发点击即完成走子）
            u32.SetCursorPos(int(x), int(y))
            u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.02)
            u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return
        human_move_to(x, y, cfg.get("method", "jerk"), cfg.get("speed", "normal"), rng)
        time.sleep(rng.uniform(0.05, 0.15))             # 到位后短暂悬停
        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(rng.uniform(0.06, 0.13))             # 按住时长
        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return
    # 拟人化关闭：瞬时定位（旧行为）
    if not u32.SetCursorPos(int(x), int(y)):
        return
    u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05 + rng.random() * 0.06)
    u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
