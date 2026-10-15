# -*- coding: utf-8 -*-
"""拟人化鼠标轨迹测试。

测试只做纯计算与事件序列校验，**不会真正移动系统光标**
（所有低层发事件函数在测试中被替换）。
"""
import math
import random

import pytest

import capture_win
from capture_win import (_resample_polyline, click_at, human_move_to,
                         plan_path, HUMAN_INPUT)


# ---------------------------------------------------------------- 轨迹形状
@pytest.mark.parametrize("method", ["jerk", "wind"])
@pytest.mark.parametrize("speed", ["fast", "normal", "slow"])
def test_path_endpoints_exact(method, speed):
    """轨迹必须精确落在目标点（否则点击会偏）。"""
    pts = plan_path((100, 100), (873, 604), method, speed, random.Random(1))
    assert pts[0] == (100, 100)
    assert pts[-1] == (873, 604)


@pytest.mark.parametrize("method", ["jerk", "wind"])
def test_path_is_not_straight_line(method):
    """真人轨迹不是直线：路径弧长应大于直线距离，且中途点偏离直线。"""
    frm, to = (100, 100), (900, 620)
    pts = plan_path(frm, to, method, "normal", random.Random(3))
    straight = math.hypot(to[0] - frm[0], to[1] - frm[1])
    plen = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))
    assert plen > straight, "路径应略长于直线（弧线）"
    assert plen < straight * 1.35, "弧线不应夸张到乱飘"
    # 中途点与直线的最大偏离应可见但不夸张
    devs = []
    for (x, y) in pts[1:-1]:
        num = abs((to[0] - frm[0]) * (frm[1] - y) - (frm[0] - x) * (to[1] - frm[1]))
        devs.append(num / straight)
    assert max(devs) > 0.3, "轨迹过直，不像真人"
    assert max(devs) < 20, f"轨迹偏离过大（{max(devs):.1f} 像素）"


@pytest.mark.parametrize("method", ["jerk", "wind"])
def test_speed_affects_step_count(method):
    """速度档影响步数：快 < 中 < 慢（同距离同算法）。"""
    counts = {sp: len(plan_path((100, 100), (900, 620), method, sp,
                               random.Random(5)))
              for sp in ("fast", "normal", "slow")}
    assert counts["fast"] < counts["normal"] <= counts["slow"], counts


def test_steps_are_dense_enough():
    """步数不能太少（否则轨迹仍是跳跃式），也不能过多（否则操作过慢）。"""
    pts = plan_path((0, 0), (1200, 800), "jerk", "normal", random.Random(2))
    assert 8 <= len(pts) <= 90, len(pts)


def test_short_move_is_direct():
    """极短距离（<2px）直接返回目标点，不做无意义轨迹。"""
    assert plan_path((100, 100), (101, 100), "jerk", "normal",
                     random.Random(0)) == [(101, 100)]


def test_jitter_present():
    """相邻步长应不恒定（微抖动），但总体平滑推进。"""
    pts = plan_path((0, 0), (600, 400), "jerk", "normal", random.Random(11))
    steps = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
             for i in range(len(pts) - 1)]
    assert len(set(round(s, 2) for s in steps)) > len(steps) * 0.5, \
        "步长过于均匀，缺少自然抖动"


def test_progress_is_monotonic_toward_target():
    """到目标的距离应基本单调递减（允许末端微调），不会来回大跳。"""
    frm, to = (0, 0), (800, 600)
    pts = plan_path(frm, to, "jerk", "normal", random.Random(4))
    dists = [math.hypot(to[0] - x, to[1] - y) for (x, y) in pts]
    assert dists[-1] == 0
    drops = sum(1 for i in range(len(dists) - 1) if dists[i + 1] > dists[i] + 3)
    assert drops <= 2, f"出现 {drops} 次明显倒退"


def test_resample_polyline_keeps_shape_and_endpoints():
    """折线重采样保持端点与整体形状（中间点落在原路径上）。"""
    raw = [(0, 0), (50, 0), (50, 50), (100, 50)]
    out = _resample_polyline(raw, 10)
    assert out[0] == (0, 0) and out[-1] == (100, 50)
    assert len(out) == 11
    # 采样点应落在原折线附近
    for (x, y) in out:
        on_path = (y == 0 and 0 <= x <= 50) or (x == 50 and 0 <= y <= 50) \
            or (y == 50 and 50 <= x <= 100)
        assert on_path or abs(x - 50) <= 1, f"采样点偏离原路径: {(x, y)}"


# ---------------------------------------------------------------- 事件序列
def test_human_move_emits_relative_events(monkeypatch):
    """移动应由多步相对事件组成（真实输入流），而非一次瞬移。"""
    emits = []
    monkeypatch.setattr(capture_win, "cursor_pos", lambda: (100, 100))
    monkeypatch.setattr(capture_win, "_emit_move",
                        lambda dx, dy: emits.append((dx, dy)))
    monkeypatch.setattr(capture_win.time, "sleep", lambda *_a: None)
    human_move_to(500, 400, "jerk", "fast", random.Random(6))
    assert len(emits) > 5, f"应分多步移动，实际 {len(emits)} 步"


def test_click_sequence_has_hover_and_hold(monkeypatch):
    """点击序列：轨迹移动 -> 悬停 -> 按下 -> 按住 -> 抬起。"""
    order = []
    monkeypatch.setattr(capture_win, "human_move_to",
                        lambda *a, **k: order.append("move"))
    monkeypatch.setattr(capture_win, "cursor_pos", lambda: (0, 0))
    monkeypatch.setattr(capture_win.time, "sleep",
                        lambda *_a: order.append("sleep"))

    class FakeU32:
        def mouse_event(self, flag, *_a):
            order.append("down" if flag == capture_win.MOUSEEVENTF_LEFTDOWN
                         else "up")
    monkeypatch.setattr(capture_win, "_user32", lambda: FakeU32())
    old = dict(HUMAN_INPUT)
    HUMAN_INPUT.update({"enabled": True, "method": "jerk", "speed": "normal"})
    try:
        click_at(300, 200)
    finally:
        HUMAN_INPUT.update(old)
    assert order[0] == "move", order
    assert "sleep" in order[1:2], "到位后应悬停一下"
    assert order.count("down") == 1 and order.count("up") == 1
    assert order.index("down") < order.index("up")


def test_click_without_human_uses_teleport(monkeypatch):
    """关闭拟人化时应回到瞬时定位（不移动轨迹）。"""
    called = {"set": 0, "move": 0}
    monkeypatch.setattr(capture_win, "human_move_to",
                        lambda *a, **k: called.__setitem__("move", called["move"] + 1))
    monkeypatch.setattr(capture_win.time, "sleep", lambda *_a: None)

    class FakeU32:
        def SetCursorPos(self, x, y):
            called["set"] += 1
            return True

        def mouse_event(self, *_a):
            pass
    monkeypatch.setattr(capture_win, "_user32", lambda: FakeU32())
    old = dict(HUMAN_INPUT)
    HUMAN_INPUT["enabled"] = False
    try:
        click_at(10, 20)
    finally:
        HUMAN_INPUT.update(old)
    assert called["set"] == 1 and called["move"] == 0


def test_human_settings_applied_from_config(qapp, monkeypatch):
    """界面设置能同步到输入层（点击时即时生效）。"""
    from gui.main_window import MainWindow
    w = MainWindow()
    w.config.update({"human_mouse": False, "human_path": "wind",
                     "human_speed": "slow"})
    w._apply_human_mouse()
    assert capture_win.HUMAN_INPUT["enabled"] is False
    assert capture_win.HUMAN_INPUT["method"] == "wind"
    assert capture_win.HUMAN_INPUT["speed"] == "slow"
    w.engine.stop() if w.engine else None


def test_click_ultra_teleports_and_double_clicks(monkeypatch):
    """极快档：不模拟轨迹，瞬时定位 + 直接按下抬起（两次点击走子）。"""
    calls = []
    monkeypatch.setattr(capture_win, "human_move_to",
                        lambda *a, **k: calls.append("move"))
    monkeypatch.setattr(capture_win.time, "sleep", lambda *_a: None)

    class FakeU32:
        def SetCursorPos(self, x, y):
            calls.append(("pos", x, y))
            return True

        def mouse_event(self, flag, *_a):
            calls.append("down" if flag == capture_win.MOUSEEVENTF_LEFTDOWN
                         else "up")
    monkeypatch.setattr(capture_win, "_user32", lambda: FakeU32())
    old = dict(HUMAN_INPUT)
    HUMAN_INPUT.update({"enabled": True, "speed": "ultra"})
    try:
        click_at(321, 654)
    finally:
        HUMAN_INPUT.update(old)
    assert ("pos", 321, 654) in calls, "极快模式应瞬时定位"
    assert "move" not in calls, "极快模式不模拟轨迹"
    assert calls.count("down") == 1 and calls.count("up") == 1
