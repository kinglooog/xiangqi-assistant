# -*- coding: utf-8 -*-
"""主窗口：布局、引擎线程桥接、着法记录、半途接管流程控制。"""
from __future__ import annotations

import json
import os
import random
import threading
import time

import numpy as np
from typing import Optional

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QKeySequence, QAction, QFont, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QHBoxLayout, QLabel, QListWidgetItem,
                             QMainWindow, QMessageBox, QPushButton, QRadioButton,
                             QSpinBox, QSplitter, QVBoxLayout, QWidget)

from core.board import Board, Move, RED, BLACK, move_to_iccs, iccs_to_move, move_to_chinese
from core.game import Game
from engine.uci import UciEngine, find_engine_exe, find_nnue_file, parse_info_line
from gui.board_widget import BoardWidget
from gui.engine_panel import EnginePanel
from gui.import_dialog import ImportDialog
from gui.setup_dialog import SetupDialog
from gui.sound import SoundManager
from gui.takeover_dialog import TakeoverDialog
from live import LiveObserver

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".xq_assistant.json")

DEFAULT_CONFIG = {
    "threads": max(1, (os.cpu_count() or 4) - 1),
    "hash_mb": 1024,
    "multipv": 3,
    "depth": 24,              # 深度上限（分析模式用；对战模式作封顶）
    "movetime_ms": 0,         # 分析模式：>0 时按固定时间
    "battle_movetime_ms": 3000,  # 对战模式每步用时上限（0=不限，纯按深度）
    "model_name": "",         # 识别用的检测模型文件名（空=自动挑最优）
    "auto_continue": True,    # 一局结束后自动点叉→再来一局→开始下一局
    "opening": "",            # 选定开局走法（空=引擎选最优）
    "human_mouse": True,      # 拟人化鼠标轨迹（更接近真人操作）
    "human_path": "jerk",     # 轨迹算法：jerk 最小急动度 / wind 物理模拟
    "human_speed": "normal",  # 速度档：fast / normal / slow
    "sound": True,            # 音效（抬子/落子/吃子/将军/绝杀锣声）
    "voice": True,            # 绝杀/困毙时语音播报
}


def available_models() -> list:
    """models/ 下可用的检测模型文件名。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models")
    if not os.path.isdir(base):
        return []
    order = {"中模型.onnx": 0, "小模型.onnx": 1, "万能带旋转.onnx": 2}
    return sorted((n for n in os.listdir(base) if n.endswith(".onnx")),
                  key=lambda n: order.get(n, 9))


class EngineBridge(QObject):
    """把引擎后台线程的回调转成 Qt 信号（跨线程队列投递）。"""
    info = pyqtSignal(dict)
    bestmove = pyqtSignal(object)
    suggestion = pyqtSignal(object)


class MainWindow(QMainWindow):
    # 一键接管：工作线程 -> GUI 线程
    takeoverFound = pyqtSignal(dict)   # hwnd/res/my_color/my_turn/title/img
    takeoverStatus = pyqtSignal(str)
    # 预计算命中后：等深算收尾取回 bestmove 的宽限（毫秒）
    PONDER_HIT_GRACE_MS = 150

    def __init__(self):
        super().__init__()
        self.setWindowTitle("象棋接管助手 · Pikafish")
        self.resize(1280, 820)
        self.setAcceptDrops(True)     # 支持把截图直接拖进窗口

        self.config = self._load_config()
        self.game = Game()
        self.engine: Optional[UciEngine] = None
        # 启动默认静音（用户要求）：界面上的 🔊/🔇 按钮可随时开声
        self.config["sound"] = False
        self.config["voice"] = False
        self.sound = SoundManager(enabled=False, voice=False)
        self.bridge = EngineBridge()
        self.bridge.info.connect(self._on_engine_info)
        self.bridge.bestmove.connect(self._on_engine_bestmove)
        self.bridge.suggestion.connect(self._on_suggestion)
        self._search_gen = 0
        self._analysis_enabled = False
        self._search_in_flight = False
        self._pending_analysis = False
        self._search_started_at = 0.0
        self._last_info_time = 0.0
        self._last_pv_top = None      # info 流去重用的上一个 PV 首着
        self.playing = False          # 人机对战进行中
        self.engine_color = None      # 对战时引擎执子
        self._auto_timer_fen = None   # 引擎自动走子的定时器校验

        self._build_ui()
        self.board.status_message.connect(
            lambda s: self.statusBar().showMessage(s, 3000))
        self.board.on_piece_grabbed = lambda: self.sound.play("pickup")
        self._analysis_enabled = self.chk_analyze.isChecked()
        self._apply_human_mouse()
        self._start_engine()
        self._refresh_all()

    # ---------------------------------------------------------------- 配置
    def _load_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
        return cfg

    def _save_config(self) -> None:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------------------------------------------------------- 界面
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # 左：棋盘
        self.board = BoardWidget()
        self.board.set_game(self.game)
        self.board.move_attempted.connect(self._on_move_attempted)
        splitter.addWidget(self.board)

        # 右：面板
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.panel = EnginePanel()
        right_layout.addWidget(self.panel, 1)

        controls = QHBoxLayout()
        self.btn_import = QPushButton("导入截图 (Ctrl+V)")
        self.btn_setup = QPushButton("手动摆局")
        self.btn_new = QPushButton("新开局")
        self.btn_undo = QPushButton("悔棋")
        self.btn_flip = QPushButton("翻转视角")
        for b in (self.btn_import, self.btn_setup, self.btn_new,
                  self.btn_undo, self.btn_flip):
            controls.addWidget(b)
        right_layout.addLayout(controls)

        controls2 = QHBoxLayout()
        self.chk_analyze = QCheckBox("持续引擎分析")
        self.chk_analyze.setChecked(True)
        self.btn_settings = QPushButton("引擎设置...")
        # 静音开关（启动默认静音）
        self.btn_mute = QPushButton("🔇 已静音")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setChecked(True)
        self.btn_mute.setToolTip("静音开关：关闭/开启音效与绝杀语音播报"
                                 "（启动默认静音）")
        controls2.addWidget(self.chk_analyze)
        controls2.addWidget(self.btn_settings)
        controls2.addWidget(self.btn_mute)
        right_layout.addLayout(controls2)

        # 一键接管
        self.btn_takeover = QPushButton("🎯 一键接管对局（自动找窗口·自动识别·自动走子）")
        self.btn_takeover.setMinimumHeight(38)
        f_big = self.btn_takeover.font()
        f_big.setBold(True)
        f_big.setPointSize(11)
        self.btn_takeover.setFont(f_big)
        self.btn_takeover.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; border-radius: 6px; }"
            "QPushButton:hover { background: #1b5e20; }")
        right_layout.addWidget(self.btn_takeover)

        # 开局走法（接管时生效；每局都按此开局行棋）
        open_row = QHBoxLayout()
        open_row.addWidget(QLabel("开局走法:"))
        self.combo_opening = QComboBox()
        from core.openings import OPENINGS
        for name, tip, _moves in OPENINGS:
            label = name if not tip else f"{name}（{tip}）"
            self.combo_opening.addItem(label or "默认（引擎选最优）", name)
        # 启动默认"引擎最优"（用户要求：不要一开软件就是双铁滑车）。
        # 中途选的开局当次会话有效，重启后回到引擎最优。
        self.config["opening"] = ""
        idx0 = self.combo_opening.findData("")
        self.combo_opening.setCurrentIndex(idx0 if idx0 >= 0 else 0)
        self.combo_opening.setToolTip(
            "接管期间：轮到我方时优先走该开局的着法，走完（或不可行）后\n"
            "恢复引擎选最优。执黑时自动镜像。每局重新应用，"
            "直至停止接管。")
        open_row.addWidget(self.combo_opening, 1)
        right_layout.addLayout(open_row)

        # 对局窗口同步（高级）
        sync_title = QLabel("对局窗口同步（高级·手动选择窗口）")
        font3 = QFont()
        font3.setBold(True)
        sync_title.setFont(font3)
        right_layout.addWidget(sync_title)

        sync_row = QHBoxLayout()
        self.combo_window = QComboBox()
        self.combo_window.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_window.setMinimumContentsLength(12)
        sync_row.addWidget(self.combo_window, 1)
        self.btn_win_refresh = QPushButton("刷新")
        sync_row.addWidget(self.btn_win_refresh)
        right_layout.addLayout(sync_row)

        sync_row2 = QHBoxLayout()
        self.btn_sync = QPushButton("开始观察")
        self.btn_resync = QPushButton("重新同步")
        self.btn_resync.setEnabled(False)
        sync_row2.addWidget(self.btn_sync)
        sync_row2.addWidget(self.btn_resync)
        right_layout.addLayout(sync_row2)

        self.chk_continue = QCheckBox("自动续局（一局结束后自动点叉→再来一局）")
        self.chk_continue.setToolTip(
            "对局结束后自动关闭结算弹窗并点「再来一局」，识别新开局后\n"
            "按“你的棋子在下方”自动判定阵营：你执红则先走，你执黑则等对方。")
        self.chk_continue.setChecked(self.config.get("auto_continue", True))
        right_layout.addWidget(self.chk_continue)

        self.chk_auto = QCheckBox("自动代点（轮到我方时直接在对局窗口落子）")
        self.chk_auto.setToolTip(
            "需先开始观察。引擎算出着法后自动在对局窗口点击落子；"
            "落子是否成功以窗口画面回读为准，未确认会提醒。")
        right_layout.addWidget(self.chk_auto)

        self.lbl_sync = QLabel("未连接。先设置好局面，再选择对局窗口开始观察。")
        self.lbl_sync.setWordWrap(True)
        right_layout.addWidget(self.lbl_sync)

        # 人机对战区
        battle_box = QHBoxLayout()
        self.rb_mode_play = QRadioButton("人机对战")
        self.rb_mode_assist = QRadioButton("接管分析")
        self.rb_mode_assist.setChecked(True)
        battle_box.addWidget(self.rb_mode_play)
        battle_box.addWidget(self.rb_mode_assist)
        battle_box.addStretch(1)
        right_layout.addLayout(battle_box)

        battle_box2 = QHBoxLayout()
        battle_box2.addWidget(QLabel("我方执:"))
        self.side_combo = QComboBox()
        self.side_combo.addItem("红方", "w")
        self.side_combo.addItem("黑方", "b")
        battle_box2.addWidget(self.side_combo)
        self.btn_battle = QPushButton("开始对战")
        self.btn_battle.setEnabled(False)
        battle_box2.addWidget(self.btn_battle)
        right_layout.addLayout(battle_box2)

        splitter.addWidget(right)
        splitter.setSizes([760, 480])

        # 快捷键
        self._paste_sc = QShortcut(QKeySequence.StandardKey.Paste, self)
        self._paste_sc.activated.connect(self._open_import)
        self._import_sc = QShortcut(QKeySequence("Ctrl+O"), self)
        self._import_sc.activated.connect(self._open_import)

        # 菜单
        menu = self.menuBar().addMenu("文件(&F)")
        act_import = QAction("导入截图...", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.triggered.connect(self._open_import)
        menu.addAction(act_import)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        menu.addAction(act_quit)

        # 信号
        self.btn_import.clicked.connect(self._open_import)
        self.btn_setup.clicked.connect(self._open_setup)
        self.btn_new.clicked.connect(self._new_game)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_flip.clicked.connect(self._flip)
        self.btn_settings.clicked.connect(self._open_settings)
        self.btn_mute.toggled.connect(self._toggle_mute)
        self.chk_analyze.toggled.connect(self._toggle_analysis)
        self.rb_mode_play.toggled.connect(self._on_mode_changed)
        self.btn_battle.clicked.connect(self._toggle_battle)
        self.combo_opening.currentIndexChanged.connect(self._on_opening_changed)
        self.btn_win_refresh.clicked.connect(self._refresh_windows)
        self.btn_sync.clicked.connect(self._toggle_sync)
        self.btn_resync.clicked.connect(self._resync_baseline)
        self.btn_takeover.clicked.connect(self._toggle_takeover)
        self.takeoverFound.connect(self._on_takeover_found)
        self.takeoverStatus.connect(self._sync_status)
        self._refresh_mute_btn()          # 初始状态与配置一致（默认静音）

        self._sync_hwnd = None
        self._sct = None
        self._takeover = False
        self._takeover_hwnd = None
        self._last_sync_res = None      # 最近一帧识别结果（供代点坐标换算）
        self._clicking = False
        self._auto_click_pending = None  # {'move','at','warned'}
        self._auto_click_retries = 0
        self._last_auto_resync = 0.0
        self._sync_state = ""            # 同步循环状态（变化时写日志）
        self._last_mover_side = None     # 最近确认的走子方（推断轮次）
        self._pending_reply = None       # 两步合帧解释出的对手应着
        # 后台预计算（ponder）：对手思考的十几秒里提前算好我方应着
        self._pondering = False           # 预计算进行中
        self._ponder_hit_pending = False  # 命中且已 ponderhit，等深算结果
        self._ponder_discard = False      # 下一次 bestmove 属于被丢弃的预搜索
        self._ponder_expected = ""        # 预计对手会走的着法（ICCS）
        self._expected_reply = None       # 上一轮分析 PV 第二着（ICCS）
        self._search_root_fen = None      # 被搜索局面的 fen（PV 翻译的基准）
        self._info_mute_until = 0.0       # 过渡期静音截止时刻（吞在途旧 info）
        self._close_clicks = 0           # 本次结算的关弹窗点击次数（限次防误点）
        self._coord_clicks = 0           # 坐标兜底点击次数（限次）
        self._settling = False           # 接管时处于结算界面（棋盘不完整）
        self._opening_queue = []         # 本局待走的开局着法（我方视角 ICCS）
        self._opening_idx = 0            # 已走完的开局着法数
        self._last_board_box = None      # 最近一次识别到的棋盘框（坐标锚点）
        self._last_our_turn_ts = 0.0     # 我方回合计时（卡死看门狗）
        self._last_activity_ts = 0.0     # 最近同步活动时刻（静默自诊断）
        self._own_eval_before = None     # 我方走子前的评估（漏招审计）
        self._clicked_at = 0.0           # 上次代点时刻（拦截限时）
        self._await_new_game = False     # 已点"再来一局"，等待新开局棋盘
        self._await_since = 0.0
        self._no_board_since = 0.0       # 连续未识别到棋盘的起始时刻
        self._sync_last_npieces = 0      # 上一帧识别子数（骤降检测）
        self._dropped_frames = 0         # 残缺帧丢弃计数
        self.observer = LiveObserver()
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(700)
        self._sync_timer.timeout.connect(self._sync_tick)

        self.statusBar().showMessage("就绪。点击「导入截图」开始半途接管。")

    # ---------------------------------------------------------------- 引擎
    def _start_engine(self):
        exe = find_engine_exe()
        if exe is None:
            self.panel.set_engine_name("未找到 Pikafish（可用纯分析模式）")
            self.statusBar().showMessage(
                "未找到引擎：请运行 tools/download_engine.py 下载 Pikafish。识别与记录功能不受影响。")
            return
        try:
            self.engine = UciEngine(exe, find_nnue_file())
            self.engine.start()
            self.engine.on_info = self._engine_info_cb
            self.engine.on_bestmove = self._engine_bestmove_cb
            self.engine.set_engine_options(
                threads=self.config["threads"],
                hash_mb=self.config["hash_mb"],
                multipv=self.config["multipv"])
            self.engine.new_game()
            self.panel.set_engine_name(self.engine.engine_id or "Pikafish")
            self.statusBar().showMessage(f"引擎已启动: {self.engine.engine_id}")
        except Exception as e:
            self.engine = None
            QMessageBox.warning(self, "引擎", f"引擎启动失败: {e}")

    def _engine_info_cb(self, info: dict) -> None:
        """引擎线程回调。做最少的工作，其余投递到 GUI 线程。"""
        # 过渡期静音：预计算挂起/未命中时会有一批属于旧搜索的 info 在途，
        # 照单全收会短暂显示错误局面的评估与"建议着法"（箭头指错）
        if time.time() < getattr(self, "_info_mute_until", 0.0):
            return
        # 丢弃过期搜索的 info：局面已变时，上一步搜索仍会继续吐出若干条
        # info，若照单全收会让面板显示上一步局面的评估（红黑视角也会跟着错）。
        if getattr(self, "_search_fen", None) is not None and \
                self.game.board.to_fen() != self._search_fen:
            return
        # 行棋方必须取"被搜索局面"的行棋方，而不是当前局面（可能已轮转），
        # 否则评估的正负号会颠倒，把红黑搞混。
        stm = getattr(self, "_search_stm", None) or self.game.side_to_move
        # 实时建议着法：PV 首着即当前深度的最佳着法，直接在 info 流中取，
        # 无需等搜索结束（纯 depth 搜索可能要十几秒）。按着法字符串去重。
        pv = info.get("pv") or []
        if info.get("multipv", 1) == 1 and pv:
            top = pv[0]
            if top != self._last_pv_top:
                self._last_pv_top = top
                self.bridge.suggestion.emit(top)
        # 面板刷新节流：info 行密度可达每秒数十条，
        # 且 PV 中文翻译约 1ms/次，只在真正转发时才翻译。
        now = time.time()
        if now - self._last_info_time < 0.06:
            return
        self._last_info_time = now
        info = dict(info)
        info["stm"] = stm
        info["pv_cn"] = self._translate_pv(pv)
        self.panel.update_info(info)

    @pyqtSlot(object)
    def _on_suggestion(self, iccs: str):
        """GUI 线程：把 info 流中的最佳着法实时显示为箭头。"""
        if self.playing:
            return  # 对战模式由 bestmove 后自动落子，不显示箭头
        if getattr(self, "_search_fen", None) is not None and \
                self.game.board.to_fen() != self._search_fen:
            return
        try:
            mv = iccs_to_move(iccs)
        except Exception:
            return
        if not self.game.board.is_legal_move(mv):
            return
        stm = self.game.side_to_move
        if self.board.my_color is not None and stm != self.board.my_color:
            return  # 只在我方轮次给出箭头
        self.board.set_best_move(mv)
        score_str = ""
        if self.panel.pv_lines:
            score_str = self.panel.pv_lines[0].get("score_str", "")
        depth = self.panel.pv_lines[0].get("depth") if self.panel.pv_lines else None
        extra = f"（{score_str}" + (f", 深度 {depth}" if depth else "") + "）" \
            if score_str else ""
        self.statusBar().showMessage(
            f"建议着法: {move_to_chinese(self.game.board, mv)}{extra}"
            f" — 请在 JJ 象棋上照此走子")

    def _engine_bestmove_cb(self, move) -> None:
        # 引擎线程回调 -> 切换到 GUI 线程处理
        self.bridge.bestmove.emit(move)

    @pyqtSlot(object)
    def _on_engine_bestmove(self, move):
        """GUI 线程处理引擎最佳着法。"""
        # 预计算的守卫（放在最前）：预搜索/被中止搜索的 bestmove 不作数
        if getattr(self, "_ponder_discard", False):
            self._ponder_discard = False
            return                      # 主动丢弃的预搜索结果：忽略
        if getattr(self, "_pondering", False):
            return                      # 预计算进行中：旧搜索的 bestmove 一律忽略
        self._search_in_flight = False
        self._search_started_at = 0.0
        self._ponder_hit_pending = False    # 深算结果已到，命中流程结束
        if getattr(self, "_pending_analysis", False):
            self._pending_analysis = False
            self._trigger_analysis()
            return  # 该 bestmove 属于被中止的旧搜索，不作为建议展示
        if not move:
            return
        # 丢弃过期搜索的着法：局面指纹不一致直接忽略
        if getattr(self, "_search_fen", None) is not None and \
                self.game.board.to_fen() != self._search_fen:
            return
        try:
            mv = iccs_to_move(move)
        except Exception:
            return
        if not self.game.board.is_legal_move(mv):
            return
        self._audit_own_eval()
        # 指定开局：**只有轮到我方时**才走开局着法。
        # 接管中引擎在对手回合也会分析并给出 bestmove，若无条件应用开局，
        # 会在对手回合把"我方着法"判为不可行而永久跳过该着
        # （实际发生过：双铁滑车只进了一边的車）。
        me = self.board.my_color
        if me is not None and self.game.side_to_move == me:
            pending = self._pending_opening_move()
            if pending is not None:
                mv, cn_open = pending
                self._sync_log(f"按选定开局走子：{cn_open}")
                mv, cn = mv, cn_open
                self._trigger_opening_click(mv, cn)
                return
        stm = self.game.side_to_move
        cn = move_to_chinese(self.game.board, mv)
        score_str = ""
        if self.panel.pv_lines:
            top = self.panel.pv_lines[0]
            score_str = top.get("score_str", "")
        # 自动代点：观察中 -> 引擎的着法（对战）或我方的建议（接管）直接落子到窗口
        want_click = (
            (self.playing and stm == self.engine_color)
            or (self.board.my_color is not None and stm == self.board.my_color))
        if (self.chk_auto.isChecked() or self._takeover)                 and self._sync_hwnd and want_click:
            res = self._foul_avoid(mv, score_str)
            if res is None:
                return     # 已派出"排除重复着法"的限制搜索，等它带回脱出着法
            mv, score_str = res
            # 记录"预计对手应着"（PV 第二着），供对手回合的预计算使用
            self._expected_reply = None
            if self.panel.pv_lines:
                pv = self.panel.pv_lines[0].get("pv") or []
                if len(pv) >= 2 and pv[0] == move_to_iccs(mv):
                    self._expected_reply = pv[1]
            cn = move_to_chinese(self.game.board, mv)
            self._auto_click_move(mv, cn, score_str)
            return
        if self.playing and stm == self.engine_color:
            # 对战中轮到引擎 -> 短暂停顿后落子（停顿只为让用户看清引擎动了哪步）
            self._auto_timer_fen = self.game.board.to_fen()
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(120, lambda: self._engine_auto_move(mv, self._auto_timer_fen))
            self.statusBar().showMessage(f"引擎走了: {cn}（{score_str}）")
        elif self.board.my_color is not None and stm == self.board.my_color:
            self.board.set_best_move(mv)
            self.statusBar().showMessage(
                f"建议着法: {cn}（{score_str}）— 请在 JJ 象棋上照此走子")
        else:
            self.board.set_best_move(mv)
            self.statusBar().showMessage(f"对方最佳预测: {cn}（{score_str}）")

    # ---------------------------------------------------------------- 自动代点
    def _cell_screen_pos(self, fen_r: int, fen_c: int):
        """FEN 格子 -> 对局窗口屏幕物理像素坐标（含拟人化随机偏移）。"""
        res = self._last_sync_res
        if res is None or res.board_box is None:
            return None
        import capture_win
        rect = capture_win.get_client_rect(self._sync_hwnd)
        if rect is None or capture_win.is_minimized(self._sync_hwnd):
            return None
        bx1, by1, bx2, by2 = res.board_box
        gw, gh = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
        r, c = fen_r, fen_c
        if res.red_at_top:                 # 识别矩阵方向还原
            r, c = 9 - r, 8 - c
        x = rect[0] + bx1 + c * gw + random.uniform(-0.14, 0.14) * gw
        y = rect[1] + by1 + r * gh + random.uniform(-0.14, 0.14) * gh
        return (int(x), int(y))

    def _audit_own_eval(self) -> None:
        """漏招审计：对比我方走子前后的评估（均换算成我方视角）。

        走子后评估骤降（>1.5 兵）时告警——这通常意味着引擎分析的局面
        与真实局面不符（识别误差），便于及时发现而不是闷头输棋。
        """
        try:
            top = self.panel.pv_lines[0] if self.panel.pv_lines else None
            if top is None or top.get("score") is None or self.board.my_color is None:
                return
            raw = float(top["score"])
            mine = raw if top.get("stm", "w") == self.board.my_color else -raw
            if self.game.side_to_move == self.board.my_color:
                self._own_eval_before = mine              # 我方待走：记下走前评估
            elif self._own_eval_before is not None:
                drop = mine - self._own_eval_before
                if drop <= -150:                          # 下降超过 1.5 兵
                    self._sync_log(
                        f"⚠ 漏招审计：我方走后评估下降 {drop / 100:.2f}"
                        f"（{self._own_eval_before / 100:+.2f} → {mine / 100:+.2f}）"
                        f"——如与实际不符，可能是识别局面有误，请核对棋盘")
                self._own_eval_before = None
        except Exception:
            pass

    def _window_matches_board(self) -> bool:
        """最新观察帧是否与本地棋盘一致（点击前的关键校验）。

        观察到同步有 1~2 帧延迟（约 1.4 秒）。若对手刚走完但尚未同步，
        此时按旧局面算出的着法去点击，会"按旧坐标点新局面"——
        起点棋子已不在原位，可能触发完全不同的走法（真实漏招）。
        """
        obs = self._last_sync_res
        if obs is None:
            return True                      # 尚无观察数据，交给后续同步兜底
        return all(obs.grid[r][c] == self.game.board.grid[r][c]
                   for r in range(10) for c in range(9))

    def _arm_ponder(self) -> None:
        """对手回合：让引擎按"预计应着"预计算（后台提前算好我方下一着）。

        人类对手通常思考十几秒——这段时间引擎提前深算我方应着；
        对手真的走预计的着法时（ponderhit）立即取回深算结果，
        1.5 秒的出着等待几乎归零。
        """
        if self.engine is None or not self.engine.is_running():
            return
        if getattr(self, "_pondering", False) or \
                getattr(self, "_ponder_hit_pending", False):
            return                                   # 已在预计算/取着流程中
        if not (self.chk_auto.isChecked() or self._takeover):
            return
        if self.board.my_color is None or self.game.result is not None:
            return
        if self.game.side_to_move == self.board.my_color:
            return                                   # 不是对手回合
        expected = getattr(self, "_expected_reply", None)
        if not expected:
            return
        try:
            e_mv = iccs_to_move(expected)
        except Exception:
            return
        if not self.game.board.is_legal_move(e_mv):
            return                                   # 预测已过期（局面变了）
        iccs = move_to_iccs(e_mv)
        fen = self.game.board.to_fen()               # 我方走后的当前局面
        # 预计算搜索的根局面 = 当前局面 + 预计对手着法：PV 中文翻译、
        # 视角标记都以此为准（否则翻不出中文/评估正负号颠倒）
        try:
            _b2 = Board(fen)
            _b2.make_move(e_mv)
            self._search_root_fen = _b2.to_fen()
        except Exception:
            self._search_root_fen = fen
        self._ponder_discard = True                  # 先停掉残留搜索，其 bestmove 丢弃
        self._pondering = True                       # 守卫先立：旧 bestmove 一律忽略
        self._info_mute_until = time.time() + 0.25   # 在途旧 info 一并吞掉
        try:
            self.engine.stop_search()
            self.engine.position(fen, [iccs])        # 最后一步 = 预计对手着法
            self.engine.go(depth=self.config["depth"], ponder=True)
        except Exception:
            self._pondering = False
            self._ponder_discard = False
            return
        self._search_in_flight = False
        self._search_started_at = 0.0
        self._ponder_expected = iccs
        # 预计算搜索的根局面是"预计对手走后"（我方该走）：info 回调按
        # _search_stm 给评估贴视角——必须更新为我方，否则引擎以我方为正
        # 的分数会被当成对手的视角，胜率条整个翻面
        # （实际发生过：轮转后红90% 跳成黑90%）
        self._search_stm = self.board.my_color
        self._sync_log(f"预计算开始：预计对手走 {iccs}，"
                       f"引擎提前思考我方应着（对手想多久、引擎算多久）")

    def _resolve_ponder(self, actual: Move) -> None:
        """对手着法落定：预计算命中 -> 复用深算；未命中 -> 丢弃重算。

        必须在对手着法写入棋局之后、`_refresh_all()` 触发新分析之前调用。
        """
        if not getattr(self, "_pondering", False):
            return
        exp = self._ponder_expected or ""
        self._pondering = False
        if self.engine is None or not self.engine.is_running():
            return
        if move_to_iccs(actual) == exp:
            # 命中：引擎已在这条线上深算数秒——转正后短宽限取出着法
            self._ponder_hit_pending = True
            self._search_in_flight = True
            self._search_started_at = time.time()
            self._search_fen = self.game.board.to_fen()
            self._search_root_fen = self._search_fen
            self._last_pv_top = None
            self._sync_log(f"预计算命中 {exp}：引擎已提前算好，立即取应着")
            self._sync_status("预计算命中 ✓ 立即出着")
            try:
                self.engine.ponderhit()
            except Exception:
                pass
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(self.PONDER_HIT_GRACE_MS, self._finish_ponder_hit)
        else:
            self._sync_log(f"预计算未命中（实际 {move_to_iccs(actual)} ≠ "
                           f"预计 {exp or '无'}）——按常规重新分析")
            self._ponder_discard = True              # 丢弃该预搜索的 bestmove
            self._info_mute_until = time.time() + 0.25   # 在途旧 info 一并吞掉
            try:
                self.engine.stop_search()
            except Exception:
                self._ponder_discard = False
            self._search_in_flight = False
            self._pending_analysis = False

    def _finish_ponder_hit(self) -> None:
        """命中宽限到期：冻结深算结果，让 bestmove 回调把着法取回来。"""
        if not getattr(self, "_ponder_hit_pending", False):
            return
        try:
            self.engine.stop_search()
        except Exception:
            pass

    def _drop_ponder(self) -> None:
        """清掉预计算状态（停止接管/换局/重同步时调用）。"""
        active = (getattr(self, "_pondering", False)
                  or getattr(self, "_ponder_hit_pending", False))
        self._pondering = False
        self._ponder_hit_pending = False
        self._ponder_expected = ""
        self._expected_reply = None
        if active and self.engine is not None:
            self._ponder_discard = True
            self._info_mute_until = time.time() + 0.25   # 在途旧 info 一并吞掉
            try:
                self.engine.stop_search()
            except Exception:
                self._ponder_discard = False

    def _foul_avoid(self, mv: Move, score_str: str):
        """禁止着法规避（长将/长捉）：引擎着法若会继续犯规循环，换着法。

        Pikafish 没有中国规则设置（UCI 选项里没有，实测确认），只把局面
        重复当成和棋：劣势时它会一直将军"求个和棋"，在平台上直接判负。
        检测到"我方连续将军/捉子"的循环后：优先换用 MultiPV 里不会重复
        的候选着法；候选全都会重复时，用 searchmoves 排除重复着法重新
        搜索。返回 (着法, 评分串)；已派出限制搜索时返回 None。
        """
        try:
            from core import rules as _rules
            b = self.game.board
            start = self.game.start_fen
            moves = [m for m, _c, _s in self.game.history]
            kind = _rules.detect_our_foul(b, start, moves, b.side_to_move)
        except Exception as e:
            self._sync_log(f"犯规检测异常（忽略本次检测）：{e}")
            return mv, score_str
        if kind is None:
            self._foul_restrict_fen = None
            return mv, score_str
        keys = _rules.position_keys(start, moves)
        repeats = _rules.repeating_moves(b, keys)
        if mv not in repeats:
            self._foul_restrict_fen = None
            return mv, score_str          # 引擎已自行脱出循环
        # ① 优先换用 MultiPV 候选里不会重复的着法
        for line in self.panel.pv_lines:
            pv = line.get("pv") or []
            if not pv:
                continue
            try:
                cand = iccs_to_move(pv[0])
            except Exception:
                continue
            if cand in repeats or not b.is_legal_move(cand):
                continue
            try:
                cn = move_to_chinese(b, cand)
            except Exception:
                cn = move_to_iccs(cand)
            self._sync_log(f"⚠ 检测到{kind}循环——放弃 {move_to_iccs(mv)}，"
                           f"改用候选 {move_to_iccs(cand)}（{cn}）脱出"
                           f"（评分 {line.get('score_str', '')}）")
            self._sync_status(f"⚠ 检测到{kind}，已更换着法避免判负 ✓",
                              sticky=True)
            return cand, line.get("score_str", "") or score_str
        # ② 候选也全是重复着法：排除重复着法后重新搜索
        fresh = [m for m in b.legal_moves() if m not in repeats]
        fen = b.to_fen()
        if (fresh and self.engine is not None and self.engine.is_running()
                and getattr(self, "_foul_restrict_fen", None) != fen):
            self._foul_restrict_fen = fen
            self._search_restrict = fresh
            self._sync_log(f"⚠ 检测到{kind}循环——候选均会重复，"
                           f"排除 {len(repeats)} 个重复着法后重新搜索")
            self._trigger_analysis()
            return None                   # 等限制搜索带回脱出着法
        # ③ 引擎帮不上忙（未运行/限制搜索仍重复）：自己挑一个不重复的
        if fresh:
            cand = fresh[0]
            try:
                cn = move_to_chinese(b, cand)
            except Exception:
                cn = move_to_iccs(cand)
            self._sync_log(f"⚠ 检测到{kind}且搜索未给出脱出着法——"
                           f"直接改走 {move_to_iccs(cand)}（{cn}）")
            self._sync_status(f"⚠ 检测到{kind}，已强制更换着法避免判负 ✓",
                              sticky=True)
            return cand, score_str
        # ④ 无路可逃（极端局面）：照走并提醒人工
        self._sync_log(f"⚠ 检测到{kind}循环，但已无脱出着法")
        self._sync_status(f"⚠ 已陷入{kind}且无脱出着法，请手动走子！",
                          sticky=True)
        return mv, score_str

    def _auto_click_move(self, mv: Move, cn: str, score_str: str):
        self._last_activity_ts = time.time()
        # 点击前一致性校验：窗口局面与本地棋盘不同 -> 放弃本次点击，
        # 先让同步追上（对手的着法会被记录），随后重新分析再落子
        if not self._window_matches_board():
            self._sync_log(f"取消代点（{cn}）：窗口局面已变化但尚未同步——"
                           f"改为先同步对手着法，再重新分析")
            self._clicked_key = None
            self._auto_click_pending = None
            return
        """把着法在对局窗口里点出来（两次点击），拟人化延迟。

        不在本地棋盘落子——落子以窗口画面回读（观察同步）为准，
        未确认会提醒。同一局面同一着法只点一次。
        """
        key = (self.game.board.to_fen(), move_to_iccs(mv))
        if self._clicking:
            return
        # 同一着法只拦截 6 秒（防"点击未生效却永久拦住同一着法"导致卡死）
        if (getattr(self, "_clicked_key", None) == key
                and time.time() - getattr(self, "_clicked_at", 0.0) < 6):
            return
        self._clicked_key = key
        self._clicked_at = time.time()
        self._auto_click_pending = {"move": mv, "at": time.time(), "warned": False}
        # 重试计数按局面记：新局面清零，重试不清（否则会无限重试）
        if getattr(self, "_clicked_fen", "") != key[0]:
            self._auto_click_retries = 0
            self._clicked_fen = key[0]
        self._sync_log(f"代点开始：{cn}（{score_str}）局面={self.game.board.to_fen()}")
        self._last_mover_side = self.board.my_color or self.game.side_to_move
        self._sync_status(f"自动代点：{cn}（{score_str}）— 正在落子，等待画面确认…")
        threading.Thread(target=self._click_worker, args=(mv,), daemon=True).start()

    def _click_worker(self, mv: Move):
        import capture_win
        try:
            self._clicking = True
            # 极快模式：不模拟轨迹（click_at 内部直点），延迟压到最短
            ultra = self.config.get("human_speed") == "ultra"
            if ultra:
                time.sleep(0.04)
            else:
                time.sleep(0.25 + random.random() * 0.4)   # 拟人：先想一会儿
            # 尽力把对局窗口提到最前（Windows 前台锁可能拒绝，不作为拒绝条件）
            if capture_win.get_foreground_window() != self._sync_hwnd:
                capture_win.set_topmost(self._sync_hwnd, True)
                capture_win.set_foreground(self._sync_hwnd)
                time.sleep(0.2)
            for i, cell in enumerate(mv):
                pos = self._cell_screen_pos(*cell)
                if pos is None:
                    self._sync_log("代点放弃：坐标换算失败（无棋盘框或窗口不可用）")
                    self._sync_status("⚠ 代点失败：对局窗口不可用。", sticky=True)
                    # 清掉待确认代点：留着会让看门狗以为"有动作在跑"
                    # 而永不救援（实际发生过重启接管后静默不走棋）
                    self._auto_click_pending = None
                    return
                # 落点遮挡校验：该屏幕坐标最顶层的窗口必须是接管窗口本身
                # （MuMu 的渲染内容是其子窗口，比对根窗口即可）
                root = capture_win.window_root_at_point(*pos)
                if root != self._sync_hwnd:
                    capture_win.set_topmost(self._sync_hwnd, True)
                    capture_win.set_foreground(self._sync_hwnd)
                    time.sleep(0.3)
                    root = capture_win.window_root_at_point(*pos)
                    if root != self._sync_hwnd:
                        self._sync_log(f"落点 ({pos[0]},{pos[1]}) 被其它窗口遮挡"
                                       f"（root={root}），放弃本次落子")
                        self._sync_status(
                            "⚠ 落子位置被其它窗口遮挡，已放弃本次落子。"
                            "请把对局窗口移到无遮挡的位置后重试。", sticky=True)
                        self._auto_click_pending = None
                        return
                capture_win.click_at(*pos)
                if i == 0:
                    time.sleep(0.04 if ultra
                               else 0.16 + random.random() * 0.32)  # 两击间隔
        finally:
            self._clicking = False

    def _trigger_opening_click(self, mv, cn: str):
        """把开局着法走上去（接管模式同样走代点通道）。"""
        if (self.chk_auto.isChecked() or self._takeover) and self._sync_hwnd:
            self._auto_click_move(mv, cn, "开局")
        else:
            self.board.set_best_move(mv)
            self.statusBar().showMessage(f"开局走法：{cn}")

    # ---------------------------------------------------------------- 开局走法
    def _on_opening_changed(self, _idx: int):
        """切换开局走法：保存配置，接管中则即时重建本局队列。"""
        self.config["opening"] = self.combo_opening.currentData() or ""
        self._save_config()
        if self._takeover:
            self._reset_opening()

    def _near_initial(self) -> bool:
        """局面是否仍在开局阶段（与标准开局相比变动不超过 6 格 ≈ 3 步）。

        开局走法只应在开局阶段执行——中局接管时套用开局（如"双铁滑车"
        再进車）等于白送一步，实际发生过（评估直接掉到 -7.6）。
        """
        init = Board()
        moved = sum(1 for r in range(10) for c in range(9)
                    if self.game.board.grid[r][c] != init.grid[r][c])
        return moved <= 6

    def _reset_opening(self):
        """按当前选择与我方执子重建开局着法队列（每局开始时调用）。"""
        from core.openings import opening_moves
        name = self.combo_opening.currentData() if hasattr(self, "combo_opening") else ""
        side = self.board.my_color
        if name and side and not self._near_initial():
            self._opening_queue = []
            self._opening_idx = 0
            self._sync_log(f"局面已非开局阶段（差异超 3 步），本局不应用"
                           f"开局走法「{name}」，直接由引擎选最优")
            return
        self._opening_queue = opening_moves(name, side) if (name and side) else []
        self._opening_idx = 0
        if self._opening_queue:
            from core.board import iccs_to_move, move_to_chinese
            try:
                names = [move_to_chinese(self.game.board, iccs_to_move(m))
                         for m in self._opening_queue]
            except Exception:
                names = list(self._opening_queue)
            self._sync_log(f"本局开局走法：{name}（{('、'.join(names))}）")

    def _pending_opening_move(self):
        """取下一步待走的开局着法（含棋子与着法），不可行则跳过。

        返回 (move, 中文着法) 或 None。仅在接管期间生效。
        """
        if not self._takeover or not self._opening_queue:
            return None
        me = self.board.my_color
        if me is None or self.game.side_to_move != me:
            return None                       # 非我方回合：开局着法不能走
        from core.board import iccs_to_move, move_to_chinese
        while self._opening_idx < len(self._opening_queue):
            iccs = self._opening_queue[self._opening_idx]
            try:
                mv = iccs_to_move(iccs)
            except Exception:
                self._opening_idx += 1
                continue
            if self.game.board.is_legal_move(mv):
                try:
                    cn = move_to_chinese(self.game.board, mv)
                except Exception:
                    cn = iccs
                return (mv, cn)
            # 该着法在当前局面不可行（如对手变化导致）：跳过并记录
            self._sync_log(f"开局着法 {iccs} 在当前局面不可行，改由引擎选择")
            self._opening_idx += 1
        return None

    def _confirm_opening_move(self, mv):
        """我方（或引擎代点）确认走出开局着法后推进队列。"""
        if not self._opening_queue or self._opening_idx >= len(self._opening_queue):
            return
        from core.board import iccs_to_move, move_to_iccs
        if self._opening_queue[self._opening_idx] == move_to_iccs(mv):
            self._opening_idx += 1

    # ---------------------------------------------------------------- 自动续局
    def _click_window_point(self, img_x: int, img_y: int, what: str) -> bool:
        """点击窗口画面内的某点（换算为屏幕坐标 + 前台/遮挡校验）。"""
        import capture_win
        rect = capture_win.get_client_rect(self._sync_hwnd)
        if rect is None:
            return False
        sx, sy = rect[0] + int(img_x), rect[1] + int(img_y)
        if capture_win.get_foreground_window() != self._sync_hwnd:
            if not capture_win.set_foreground(self._sync_hwnd):
                self._sync_log(f"续局: 无法把窗口带到前台，跳过点击（{what}）")
                return False
            time.sleep(0.2)
        root = capture_win.window_root_at_point(sx, sy)
        if root != self._sync_hwnd:
            self._sync_log(f"续局: 落点被其它窗口遮挡，跳过点击（{what}）")
            return False
        capture_win.click_at(sx, sy)
        self._sync_log(f"续局: 已点击「{what}」@({sx},{sy})")
        return True

    def _auto_continue_tick(self, img) -> bool:
        """结算界面自动续局。

        顺序与安全约束（避免"误点应用自身的关闭按钮直接退游戏"）：
        1. **优先直接点「再来一局」**——该按钮特征鲜明（绿底四字），
           且只在结算界面出现，误检风险极低，通常无需先关弹窗；
        2. 仅当「再来一局」被弹窗挡住时才尝试关闭弹窗，且必须同时满足：
           对局已结束（有胜负结果）+ 只能在画面中部安全区域找到叉
           （X 字形过于通用，全图搜索会误匹配窗口/应用自身的关闭按钮）
           + 每次结算最多点 3 次。
        """
        if not getattr(self, "chk_continue", None) or                 not self.chk_continue.isChecked() or not self._sync_hwnd:
            return False
        if self._await_new_game:
            return False                    # 已点击，等待新棋盘出现
        from recognition.buttons import find_button, settle_button_point
        # ① 直接点「再来一局」（最优路径；以棋盘框锚定尺寸，适配任意分辨率）
        # 冷却：刚点过「再来一局」后 6 秒内不再点（防屏幕切换期间重复点击，
        # 第二个点击会落到新对局界面上——实际发生过）
        if time.time() - getattr(self, "_again_clicked_at", 0.0) < 6:
            return False
        hit = find_button(img, "again", board_box=self._last_board_box)
        if hit is not None:
            if self._click_window_point(hit[0], hit[1], "再来一局"):
                self._again_clicked_at = time.time()
                self._await_new_game = True
                self._await_since = time.time()
                self._sync_status("已点「再来一局」，等待新开局…")
                return True
            return False
        # ② 模板没匹配到：按画面锚点（按钮锚定在界面底部，标定自真实截图）
        #    用坐标直接点「再来一局」。仅在"对局已结束 + 棋盘被遮挡 3 秒以上"
        #    时启用，限 2 次。
        if ((self.game.result is not None or self._settling)
                and (self._no_board_since > 0 or self._settling)
                and self._coord_clicks < 2):
            pt = settle_button_point("again", img.shape)
            if pt is not None:
                self._coord_clicks += 1
                self._sync_log(f"模板未匹配，按棋盘锚点坐标定位「再来一局」"
                               f"（第 {self._coord_clicks} 次）")
                if self._click_window_point(pt[0], pt[1], "再来一局（坐标定位）"):
                    self._again_clicked_at = time.time()
                    self._await_new_game = True
                    self._await_since = time.time()
                    self._sync_status("已按坐标点「再来一局」，等待新开局…")
                    return True
            return False
        # ③ 「再来一局」不可见（被弹窗遮挡）时尝试关弹窗。
        #    安全条件：棋盘确实读不到 +（对局已结束 或 棋盘持续读不到 6 秒，
        #    覆盖对手认输这类本地无法判定胜负的结束方式）+ 限 3 次；
        #    且只能在画面中部安全区域匹配（边缘的应用关闭按钮被硬性排除）。
        if (self._no_board_since > 0
                and (self.game.result is not None
                     or time.time() - self._no_board_since > 6)
                and self._close_clicks < 3):
            hit = find_button(img, "close", board_box=self._last_board_box)
            if hit is not None:
                self._close_clicks += 1
                self._click_window_point(hit[0], hit[1], "关闭结算弹窗")
                return True
        return False

    def _adopt_new_game(self, res):
        """新对局采纳：阵营按"你的棋子在下方"推断，红先行。

        不受上一局设定影响——这是全新一局。
        """
        from recognition.classifier import grid_to_fen
        my_color = BLACK if res.red_at_top else RED
        stm = RED                      # 象棋红先行
        fen = grid_to_fen(res.grid, side_to_move=stm)
        self._drop_ponder()            # 新对局：旧预计算作废
        self.game.reset(fen)
        self._fix_turn_by_check()          # 同理：被将军方必是走棋方
        self.board.set_game(self.game)
        self.board.set_my_color(my_color)
        self.panel.set_my_color(my_color)
        self.panel.clear_analysis()
        self.observer.force_baseline(res.grid, res.red_at_top)
        self._await_new_game = False
        self._settling = False
        self._close_clicks = 0
        self._coord_clicks = 0
        self._pending_reply = None
        self._clicked_key = None
        self._auto_click_pending = None
        self._partial_key = None
        self._sync_last_npieces = res.n_pieces
        self._last_mover_side = None
        self._reset_opening()              # 每局重新应用选定开局
        side = "红方" if my_color == RED else "黑方"
        self._sync_log(f"新对局开始：推断我方执{side}，"
                       f"{'我方先走' if my_color == RED else '等对方先走'}")
        self._sync_status(f"新对局 ✓ 我方执{side}，"
                          + ("先走…" if my_color == RED else "等对方先走…"))
        self._refresh_all()

    def _stall_watchdog(self):
        """卡死看门狗：我方回合长时间无任何动作 -> 重新触发引擎分析。

        覆盖所有"分析未被触发/着法被拦截/点击静默失败"导致的停滞，
        是兜底保险（正常流程不会触发）。
        """
        if not (self._takeover or (self.chk_auto.isChecked() and self._sync_hwnd)):
            return
        me = self.board.my_color
        if me is None or self.game.side_to_move != me or self.game.result is not None:
            self._last_our_turn_ts = 0.0
            return
        now = time.time()
        # 引擎健康检查（优先于计时逻辑）：
        # ① 进程已退出 -> 立即重启并重新分析；
        # ② 搜索超过 15 秒没有 bestmove（引擎挂死）-> 重启；
        # 否则看门狗会一直"刷新计时"而永不救援（实际发生过的卡死）。
        if self.engine is not None and not self.engine.is_running():
            cur_fen = self.game.board.to_fen()
            if getattr(self, "_engine_kill_fen", None) == cur_fen:
                # 同一局面已杀死过引擎：不再重复送，避免"重启->再死"死循环
                # （实际发生过连续 152 次）。等局面变化或用户重新同步。
                self._search_in_flight = False
                self._pending_analysis = False
                self._search_started_at = 0.0
                if getattr(self, "_kill_logged", None) != cur_fen:
                    self._kill_logged = cur_fen
                    reason = self.game.board.validate_position() or "未知（局面看似合法）"
                    self._sync_log(f"引擎在该局面反复退出，暂停送分析：{reason}")
                    self._sync_status(
                        f"⚠ 引擎无法分析当前局面（{reason}）。"
                        f"请用「重新同步」或「手动摆局」修正局面。", sticky=True)
                self.engine = None
                self._start_engine()
                return
            self._sync_log("引擎进程已退出——自动重启并重新分析")
            self._engine_kill_fen = cur_fen     # 记住可疑局面
            self._search_in_flight = False
            self._pending_analysis = False
            self._search_started_at = 0.0
            self.engine = None
            self._start_engine()
            self._clicked_key = None
            self._trigger_analysis()
            return
        if (self._search_in_flight and self._search_started_at
                and now - self._search_started_at > 15):
            self._sync_log("引擎搜索超 15 秒无响应——重启引擎并重新分析")
            self._search_in_flight = False
            self._pending_analysis = False
            self._search_started_at = 0.0
            try:
                self.engine.stop() if self.engine else None
            except Exception:
                pass
            self.engine = None
            self._start_engine()
            self._clicked_key = None
            self._trigger_analysis()
            return
        if self._last_our_turn_ts == 0.0:
            self._last_our_turn_ts = now
            return
        if (self._search_in_flight or self._clicking
                or self._auto_click_pending is not None):
            self._last_our_turn_ts = now          # 有动作在跑，刷新计时
            return
        if now - self._last_our_turn_ts > 6:
            self._last_our_turn_ts = now
            self._sync_log("看门狗：我方回合 6 秒无动作——重新触发引擎分析")
            self._clicked_key = None              # 解除可能的旧拦截
            self._trigger_analysis()

    def _check_auto_click_confirm(self):
        """代点后超时未见画面确认：自动重试一次，再失败才提醒。"""
        p = self._auto_click_pending
        if p is None or p.get("warned"):
            return
        if time.time() - p["at"] > 8:
            if self._auto_click_retries < 1:
                self._auto_click_retries += 1
                try:
                    cn = move_to_chinese(self.game.board, p["move"])
                except Exception:
                    from core.board import move_to_iccs
                    cn = move_to_iccs(p["move"])
                self._sync_log(f"代点 8 秒未确认，自动重试（第 "
                               f"{self._auto_click_retries} 次）：{cn}")
                self._sync_status(f"落子未确认，自动重试：{cn}…")
                self._clicked_key = None
                self._auto_click_pending = None
                self._auto_click_move(p["move"], cn, "")
                return
            p["warned"] = True
            try:
                cn = move_to_chinese(self.game.board, p["move"])
            except Exception:
                from core.board import move_to_iccs
                cn = move_to_iccs(p["move"])
            self._sync_log(f"代点重试仍未确认：{cn}")
            self._sync_status(
                f"⚠ 代点的着法（{cn}）重试后仍未确认。"
                f"可能被弹窗/网络卡住，请手动检查 JJ。", sticky=True)
            # 放弃本次代点：pending 留着会把看门狗的"有动作在跑"判定
            # 永久喂住，卡死后无人救援（实际发生过静默不走棋）
            self._auto_click_pending = None

    def _engine_auto_move(self, move, expected_fen):
        """定时器回调：确认局面未变后替引擎落子。"""
        if (self.chk_auto.isChecked() or self._takeover) and self._sync_hwnd:
            return   # 代点模式：引擎着法走 JJ 窗口，经观察回读记录
        if not self.playing or self.game.result is not None:
            return
        if self.game.board.to_fen() != expected_fen:
            return
        if self.game.side_to_move != self.engine_color:
            return
        self._play_move(move, source="engine")

    # ---------------------------------------------------------------- 人机对战
    def _on_mode_changed(self, checked):
        self.btn_battle.setEnabled(self.rb_mode_play.isChecked())
        if self.rb_mode_assist.isChecked() and self.playing:
            self._stop_battle()

    def _toggle_battle(self):
        if self.playing:
            self._stop_battle()
        else:
            self._start_battle()

    def _start_battle(self):
        if self.engine is None:
            QMessageBox.warning(self, "人机对战", "引擎未启动，无法对战。")
            return
        if self.game.result is not None:
            self.game.reset()
        if not self._position_is_sane():
            QMessageBox.warning(self, "人机对战",
                                "当前局面不合法："
                                f"{getattr(self, '_position_problem', '未知')}。\n"
                                "请用「手动摆局」改正后再开始对战。")
            return
        my = self.side_combo.currentData()
        self.playing = True
        self.engine_color = "b" if my == "w" else "w"
        self.board.set_my_color(my)
        self.panel.set_my_color(my)
        self.btn_battle.setText("结束对战")
        self.side_combo.setEnabled(False)
        side = "红方" if my == "w" else "黑方"
        self.statusBar().showMessage(
            f"对战开始：你执{side}。"
            + ("请走子。" if self.game.side_to_move == my else "引擎先走..."))
        self._trigger_analysis()

    def _stop_battle(self):
        self.playing = False
        self.engine_color = None
        self.btn_battle.setText("开始对战")
        self.side_combo.setEnabled(True)
        try:
            self.engine.stop_search() if self.engine else None
        except Exception:
            pass
        self.statusBar().showMessage("对战已结束。")

    def _translate_pv(self, pv) -> list:
        """把 ICCS 着法序列翻成中文记法。

        必须从**被搜索的局面**回放（预计算搜索的是"预计对手走后"的局面，
        与当前局面不同）：用当前局面回放会在第一步就不合法而中断，
        面板只能回退显示 ICCS 编码串（实际发生过的 bug）。
        """
        if not pv:
            return []
        probe = Board(self._search_root_fen or self.game.board.to_fen())
        out = []
        for u in pv:
            try:
                mv = iccs_to_move(u)
            except Exception:
                break
            if not probe.is_legal_move(mv):
                break
            out.append(move_to_chinese(probe, mv))
            probe.make_move(mv)
        return out

    def _position_is_sane(self) -> bool:
        """局面合法性检查：识别校对失误或手动摆错会得到非法局面
        （子力超量、将帅缺失或照面），Pikafish 遇到非法局面会直接退出。"""
        problem = self.game.board.validate_position()
        if problem:
            self._position_problem = problem
            return False
        return True

    def _trigger_analysis(self):
        """对当前局面触发一次引擎搜索。"""
        # 犯规规避的限制搜索（_foul_avoid 设置）：在入口就取出清空，
        # 避免后续早退路径（局面不合法等）把它残留给下一次搜索
        restrict = getattr(self, "_search_restrict", None)
        self._search_restrict = None
        if self.engine is None:
            return
        # 预计算期间不打断：进行中等对手走子（异局面），命中待取着法时不重算
        if getattr(self, "_pondering", False) or \
                getattr(self, "_ponder_hit_pending", False):
            return
        if not (self.chk_analyze.isChecked() or self.playing or self._takeover):
            return
        if self.game.result is not None:
            return
        if not self._position_is_sane():
            problem = getattr(self, "_position_problem", "未知")
            self.statusBar().showMessage(
                f"局面不合法：{problem}。请用「手动摆局」或重新导入截图并校对。",
                8000)
            self._sync_log(f"拒绝送引擎分析：局面不合法（{problem}）")
            return
        if not self.engine.is_running():
            self.statusBar().showMessage("引擎进程异常退出，正在重启...", 5000)
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
            self._start_engine()
            if self.engine is None:
                return
        # 上一个搜索尚未结束：只发 stop，等它的 bestmove 到达后再启动新搜索，
        # 否则新旧 bestmove 无法区分归属
        if self._search_in_flight:
            self._pending_analysis = True
            try:
                self.engine.stop_search()
            except Exception:
                pass
            return
        self._search_gen += 1
        self._search_in_flight = True
        self._search_started_at = time.time()
        self._search_fen = self.game.board.to_fen()
        self._search_stm = self.game.side_to_move
        self._last_pv_top = None
        fen = self._search_fen
        self._search_root_fen = fen       # PV 中文翻译以被搜索局面为基准
        self.panel.set_stm(self.game.side_to_move)
        self.panel.clear_analysis()
        # 对战模式统一用「用时上限 + 深度封顶」双限制：时间先到即停，
        # 深度上限不变（默认 24），着法质量与纯深度搜索一致，
        # 但不必等它算完整个深度（实测中局纯 depth24 需 14s，
        # 3s 即达 depth20 且给出相同着法）。我方思考时同样限时，
        # 避免引擎占满 CPU 空转。
        if self.playing or self._takeover or                 (self.chk_auto.isChecked() and self._sync_hwnd):
            movetime = self.config.get("battle_movetime_ms", 0)  # 代点要求快速响应
        else:
            movetime = self.config.get("movetime_ms", 0)
        # 犯规规避的限制搜索（_foul_avoid 设置）：只在这些着法里选
        try:
            self.engine.stop_search()
            self.engine.position(fen, [])
            self.engine.go(depth=self.config["depth"],
                           movetime_ms=movetime if movetime > 0 else None,
                           searchmoves=([move_to_iccs(m) for m in restrict]
                                        if restrict else None))
        except Exception:
            pass

    @pyqtSlot(dict)
    def _on_engine_info(self, info):
        pass  # panel 内部已处理

    # ---------------------------------------------------------------- 对局操作
    def _on_move_attempted(self, move: Move):
        """棋盘上尝试走子（摆局 / 记录对方着法 / 对战中走我方棋子）。"""
        self._play_move(move, source="user")

    def _play_move(self, move: Move, source: str = "user"):
        if self.game.result is not None:
            return
        mover = self.game.side_to_move
        # 对战中：用户不能替引擎走子（引擎自己的着法 source='engine' 放行）
        if self.playing and source == "user" and mover == self.engine_color:
            self.statusBar().showMessage("对战中不能移动引擎的棋子", 3000)
            return
        # 动画需要走子前的棋子，先记下来
        moving_piece = self.game.board.piece_at(*move[0])
        captured = self.game.board.piece_at(*move[1])
        cn = move_to_chinese(self.game.board, move)
        if not self.game.play(move):
            self.statusBar().showMessage(f"非法着法: {move_to_iccs(move)}", 3000)
            return
        self.board.animate_move(move, moving_piece)
        self.board.set_best_move(None)
        # 音效：吃子用重音，普通落子用木声
        self.sound.play("capture" if captured is not None else "drop")
        self._refresh_all()
        self._after_move_effects(move, captured, mover)
        # 把中文着法也写进状态栏，配合动画看清这一步
        cap_txt = ""
        if captured is not None:
            from core.board import PIECE_NAMES
            cap_txt = f"，吃{'红' if captured.isupper() else '黑'}{PIECE_NAMES[captured]}"
        if self.playing:
            who = "你" if mover == (self.board.my_color or RED) else "引擎"
            self.statusBar().showMessage(f"{who}走了：{cn}{cap_txt}")
        else:
            who = "我方" if mover == (self.board.my_color or RED) else "对方"
            self.statusBar().showMessage(f"{who}着法已记录：{cn}{cap_txt}")

    def _after_move_effects(self, move: Move, captured, mover):
        """走子后的将军/绝杀音效与特效。"""
        board = self.game.board
        if self.game.result is not None:
            # 对局已结束：立刻叫停在跑的搜索，把 CPU 让给特效与界面
            if self.engine is not None and self.engine.is_running():
                try:
                    self.engine.stop_search()
                except Exception:
                    pass
            result = self.game.result
            # 输家 = 现在轮到走子的一方（无子可走/被吃帅将）
            loser = board.side_to_move
            king = board.find_king(loser) or move[1]
            if "将死" in result or "被吃" in result:
                text, spoken = "绝杀", "绝杀"
            else:
                text, spoken = "困毙", "困毙"
            self.board.start_effect(
                text, QColor(220, 40, 40), king=king, dur_ms=3200, sub=result)
            self.sound.play("mate")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(800, lambda: self.sound.speak(spoken))
            return
        # 未结束：若形成将军，给出提示音、语音与轻量特效
        if board.in_check(board.side_to_move):
            king = board.find_king(board.side_to_move)
            who = "红方被将军" if board.side_to_move == RED else "黑方被将军"
            self.board.start_effect("将军", QColor(240, 190, 40), king=king,
                                    dur_ms=1100, sub=who, scale=0.55)
            self.sound.play("check")
            self.sound.speak("将军")

    def _refresh_all(self):
        self.panel.set_my_color(self.board.my_color)
        self.panel.set_stm(self.game.side_to_move)
        self._refresh_move_list()
        over, result = self.game.board.is_game_over()
        if over:
            self.game.result = result
            if self.playing:
                self._stop_battle()
            self.statusBar().showMessage("对局结束: " + result)
        elif self._analysis_enabled or self.playing or self._takeover:
            self._trigger_analysis()

    def _refresh_move_list(self):
        self.panel.move_list.clear()
        moves = self.game.move_list_chinese()
        for i in range(0, len(moves), 2):
            red_mv = moves[i]
            black_mv = moves[i + 1] if i + 1 < len(moves) else ""
            item = QListWidgetItem(
                f"{i // 2 + 1}.  红 {red_mv:<10s} 黑 {black_mv}")
            self.panel.move_list.addItem(item)
        self.panel.move_list.scrollToBottom()

    def _new_game(self):
        if self.playing:
            self._stop_battle()
        self.game.reset()
        self.board.set_game(self.game)
        self.panel.clear_analysis()
        self._refresh_all()
        self.statusBar().showMessage("新开局（初始局面）。可通过「导入截图」设置任意局面。")

    def _undo(self):
        if self.game.undo():
            hist = self.game.history
            self.board.cancel_anim()          # 避免动画残留在已回退的局面上
            self.board.cancel_effect()
            self.board.set_last_move(hist[-1][0] if hist else None)
            self.board.set_best_move(None)
            self._refresh_all()
            self.statusBar().showMessage("已悔棋")

    def _flip(self):
        self.board.view_flip = not getattr(self.board, "view_flip", False)
        self.board.update()

    def _toggle_analysis(self, checked: bool):
        self._analysis_enabled = checked
        if checked:
            self._trigger_analysis()
        elif self.engine is not None:
            try:
                self.engine.stop_search()
            except Exception:
                pass

    # ---------------------------------------------------------------- 拖拽导入
    def dragEnterEvent(self, event):
        """把截图拖到窗口上即可导入。"""
        md = event.mimeData()
        if md.hasImage() or md.hasUrls():
            event.acceptProposedAction()
            self.statusBar().showMessage("松开鼠标即可导入该截图")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        from gui.import_dialog import image_from_mime
        img = image_from_mime(event.mimeData())
        if img is None:
            self.statusBar().showMessage("拖入的内容不是可识别的图片", 5000)
            event.ignore()
            return
        event.acceptProposedAction()
        self._open_import(preloaded=img)

    # ---------------------------------------------------------------- 一键接管
    def _toggle_takeover(self):
        if self._takeover:
            self._stop_sync()          # 统一停止（含取消置顶）
            return
        if self._sync_hwnd:            # 手动观察中 -> 先停
            self._stop_sync()
        self._reset_session_state()
        self._takeover = True
        self.btn_takeover.setText("⏹ 停止接管（再点一次）")
        self.takeoverStatus.emit("正在自动寻找含棋盘的窗口…")
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _reset_session_state(self):
        """开始一次新接管前，复位上一会话遗留的状态。

        上一局的会话标志若不清，重新接管后会以各种方式"静默不走棋"：
        - _engine_kill_fen：上一局引擎崩溃局面黑名单——重新接管常在
          同一局面，护栏直接禁止送分析（实际发生过）；
        - _await_new_game / _settling：上一局结算/续局流程残留，会让
          观察循环继续等"新开局"而不同步着法（还会用几何推断覆盖
          用户在接管对话框里选的阵营与轮次）；
        - _auto_click_pending：残留待确认代点会把看门狗喂成
          "有动作在跑"，永不救援。
        """
        self._await_new_game = False
        self._await_since = 0.0
        self._settling = False
        self._close_clicks = 0
        self._coord_clicks = 0
        self._engine_kill_fen = None
        self._kill_logged = None
        self._pending_reply = None
        self._clicked_key = None
        self._clicked_at = 0.0
        self._clicked_fen = ""
        self._auto_click_pending = None
        self._auto_click_retries = 0
        self._partial_key = None
        self._lift_key = None
        self._lift_since = 0.0
        self._lift_warned = False
        self._last_mover_side = None
        self._last_auto_resync = 0.0
        self._dropped_frames = 0
        self._no_board_since = 0.0
        self._sync_last_npieces = 0
        self._sync_state = ""
        self._last_our_turn_ts = 0.0
        self._again_clicked_at = 0.0
        self._search_restrict = None
        self._foul_restrict_fen = None
        self._no_board_saved = False
        self._no_board_hb = 0.0
        self._grab_heal_at = 0.0
        self._relocate_at = 0.0
        self._search_root_fen = None
        self._shift_guard_n = 0
        self._drop_ponder()
        self.observer.reset()

    def _probe_worker(self):
        """后台线程：枚举窗口并逐个探测是否有棋盘（两帧一致才算稳）。"""
        import capture_win
        log_path = os.path.join(os.path.expanduser("~"), "xq_assistant_takeover.log")

        def log(text: str):
            """探测全程写日志——任何失败路径都留痕。"""
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
            except Exception:
                pass

        try:
            _base = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "recognition")
            _fp = time.strftime("%m-%d %H:%M", time.localtime(max(
                os.path.getmtime(os.path.join(_base, f))
                for f in ("yolo_board.py", "pipeline.py", "skin.py"))))
        except Exception:
            _fp = "?"
        log(f"开始接管：自动寻找含棋盘的窗口（pid={os.getpid()}，"
            f"识别代码 {_fp}）")
        try:
            all_wins = capture_win.list_windows()
        except Exception as e:
            log(f"枚举窗口失败: {e}")
            self.takeoverStatus.emit(f"枚举窗口失败: {e}")
            return
        # 逐窗口记录 + 过滤（诊断：MuMu 为何没进候选一目了然）
        wins, rejected = [], []
        for w in all_wins:
            t = w["title"][:24] or f"(无标题·{w.get('cls', '')[:20]})"
            exe = capture_win.process_exe(w["hwnd"])
            reasons = []
            if w["minimized"]:
                # 模拟器窗口最小化：自动恢复后纳入候选（用户点接管就是要连它）
                if (self._looks_like_emulator(w["title"])
                        or (getattr(self, "_takeover_title", "")
                            and w["title"] == self._takeover_title)):
                    if capture_win.restore_window(w["hwnd"]):
                        log(f"已自动恢复最小化的窗口“{t}”")
                        w = dict(w, minimized=False)
                if w["minimized"]:
                    reasons.append("已最小化")
            if w["pid"] == os.getpid():
                reasons.append("本软件自身")
            if exe.lower().startswith(("c:" + os.sep + "windows" + os.sep,
                                       "c:/windows/")):
                reasons.append("系统进程(exe=" + os.path.basename(exe) + ")")
            if (w["rect"][2] - w["rect"][0]) < 400 or (w["rect"][3] - w["rect"][1]) < 300:
                reasons.append(f"尺寸过小({w['rect'][2]-w['rect'][0]}x{w['rect'][3]-w['rect'][1]})")
            if reasons:
                rejected.append(f"“{t}”[{os.path.basename(exe) or '未知'}"
                                f"|{w.get('cls', '')[:24]}]: " + ",".join(reasons))
            else:
                wins.append(w)
        if rejected:
            log("未入候选 " + str(len(rejected)) + " 个: " + " | ".join(rejected))
        # 探测优先级：模拟器（MuMu 等）先试——找不到棋盘再退回试浏览器
        # 等其它窗口（JJ 网页版等）；同组内大窗口优先（更可能是对局界面）
        wins.sort(key=lambda w: self._probe_priority(w["title"], w.get("rect")))
        log("进入候选 " + str(len(wins)) + " 个（探测顺序）: "
            + "; ".join(f"“{w['title'][:20]}”" for w in wins))
        self._probe_tried_emu = False
        if not wins:
            log("探测结束：没有可探测的窗口")
            minimized_n = sum(1 for r in rejected if "已最小化" in r)
            if minimized_n:
                self.takeoverStatus.emit(
                    f"检测到 {minimized_n} 个窗口处于最小化（如 MuMu 模拟器），"
                    f"请先还原窗口再点「一键接管」。")
            else:
                self.takeoverStatus.emit("没有可探测的窗口。请先打开 MuMu 并进入对局。")
            return
        from recognition.pipeline import recognize_screenshot
        fail_reasons = []
        try:
            for i, winfo in enumerate(wins):
                title = winfo["title"][:30]
                if self._looks_like_emulator(winfo["title"]):
                    self._probe_tried_emu = True
                elif (getattr(self, "_probe_tried_emu", False)
                        and not getattr(self, "_probe_emu_note", False)):
                    self._probe_emu_note = True
                    log("模拟器窗口未找到棋盘——继续尝试浏览器等其它窗口")
                    self.takeoverStatus.emit(
                        "模拟器里没找到棋盘，继续查找浏览器等其它窗口…")
                self.takeoverStatus.emit(
                    f"探测窗口 {i + 1}/{len(wins)}：{title} …")
                grids = []
                attempts = []
                # 最多取 3 帧：任意相邻两帧一致即认定稳定（避开动画）。
                # 皮肤/计时器抖动时两帧可能恰好不一致，多给一次机会。
                for _ in range(3):
                    img = capture_win.capture_window_client(winfo["hwnd"])
                    how = "PrintWindow"
                    if img is None:
                        # PrintWindow 失败（部分渲染引擎）-> 前台化后抓屏幕区域
                        try:
                            import ctypes as _ct
                            import ctypes.wintypes as _wt
                            _u32 = _ct.windll.user32
                            _u32.ShowWindow(_wt.HWND(winfo["hwnd"]), 5)   # SW_SHOW
                            _u32.SetForegroundWindow(_wt.HWND(winfo["hwnd"]))
                        except Exception:
                            pass
                        time.sleep(0.35)
                        if capture_win.get_foreground_window() != winfo["hwnd"]:
                            attempts.append("无法置前台(可能需管理员权限)")
                            break
                        rect = capture_win.get_client_rect(winfo["hwnd"])
                        if rect is None:
                            attempts.append("无客户区")
                            break
                        try:
                            import mss
                            with mss.mss() as sct:
                                raw = sct.grab({"left": rect[0], "top": rect[1],
                                                "width": rect[2] - rect[0],
                                                "height": rect[3] - rect[1]})
                            img = np.ascontiguousarray(np.asarray(raw)[:, :, :3])
                            how = "屏幕抓取"
                        except Exception as e:
                            attempts.append(f"抓屏失败:{e}")
                            break
                    res = recognize_screenshot(img, model_path=self._model_path())
                    if res.source != "yolo":
                        attempts.append(f"{how}:识别失败")
                        grids = []
                        break
                    # 棋子少不直接放弃：可能是结算界面（棋盘被弹窗遮挡、
                    # 将帅被挡），下面会检查「再来一局」按钮决定是否接受
                    attempts.append(f"{how}:{res.n_pieces}子")
                    grids.append((res.grid, res.red_at_top, res, img))
                    time.sleep(0.7)
                log(f"窗口“{title}”探测: {'; '.join(attempts) or '无尝试'}")
                from recognition.yolo_board import board_plausibility
                from recognition.buttons import find_button
                accepted = False
                # 相邻两帧一致 -> 校验并采纳（皮肤抖动时多帧提高命中率）
                for gi in range(len(grids) - 1):
                    if grids[gi][0] != grids[gi + 1][0]:
                        continue
                    grid, red_top, res, img2 = grids[gi + 1]
                    ok, reason = board_plausibility(res)
                    if ok:
                        log(f"采纳窗口“{title}”（{res.n_pieces} 子）")
                        self.takeoverFound.emit({
                            "hwnd": winfo["hwnd"], "res": res,
                            "title": winfo["title"], "img": img2})
                        accepted = True
                        break
                    # 结算界面识别：棋盘被结算弹窗遮挡 -> 棋子不全、将帅可能被挡，
                    # 合理性校验必然失败。但只要看到「再来一局」按钮，
                    # 就说明这是结算界面，应接受该窗口并交给续局逻辑开下一局。
                    hit = find_button(img2, "again", board_box=res.board_box)
                    if hit is not None:
                        log(f"窗口“{title}”识别为结算界面（含「再来一局」，"
                            f"分 {hit[2]:.2f}）——接受并交由续局逻辑")
                        self.takeoverFound.emit({
                            "hwnd": winfo["hwnd"], "res": res,
                            "title": winfo["title"], "img": img2,
                            "settling": True})
                        accepted = True
                        break
                    log(f"窗口“{title}”未通过合理性校验: {reason}")
                    self.takeoverStatus.emit(f"跳过窗口“{title}”：{reason}")
                    fail_reasons.append(f"{title}：{reason}")
                    # 未通过校验的画面存档，便于事后诊断
                    try:
                        import cv2 as _cv
                        _p = os.path.join(os.path.expanduser("~"),
                                          "xq_probe_fail.png")
                        _cv.imwrite(_p, img2)
                        log(f"未通过校验的画面已存档 {_p}")
                    except Exception:
                        pass
                    break
                if not accepted and grids:
                    # 帧间不一致或帧数不足：至少看是不是结算界面
                    grid, red_top, res, img2 = grids[-1]
                    hit = find_button(img2, "again", board_box=res.board_box)
                    if hit is not None:
                        log(f"窗口“{title}”识别为结算界面（含「再来一局」，"
                            f"分 {hit[2]:.2f}）——接受并交由续局逻辑")
                        self.takeoverFound.emit({
                            "hwnd": winfo["hwnd"], "res": res,
                            "title": winfo["title"], "img": img2,
                            "settling": True})
                        accepted = True
                if not accepted and not grids:
                    fail_reasons.append(f"{title}：画面识别失败")
                if accepted:
                    return           # 已采纳并发出 takeoverFound，结束探测
            if fail_reasons:
                self.takeoverStatus.emit(
                    f"未找到含棋盘的窗口（{fail_reasons[-1]}）。"
                    f"已先试模拟器、再试浏览器等其它窗口——"
                    f"请确认棋盘完整可见、无弹窗遮挡后重试。")
            else:
                self.takeoverStatus.emit(
                    "未找到含棋盘的窗口（已先试模拟器、再试浏览器等其它窗口）。"
                    "请确认已进入 JJ 对局界面（不是大厅），窗口未最小化，再重试。")
            log("探测结束：未找到含棋盘的窗口。原始窗口全量清单：")
            try:
                for w in capture_win.list_all_toplevel_windows():
                    exe = os.path.basename(capture_win.process_exe(w["hwnd"])) or "?"
                    if w["pid"] == os.getpid():
                        continue                       # 跳过软件自身窗口
                    flags = []
                    if not w["visible"]:
                        flags.append("不可见")
                    if w["minimized"]:
                        flags.append("最小化")
                    if w["toolwindow"]:
                        flags.append("工具窗")
                    log(f"  [{exe}] “{w['title'][:20] or '(无标题)'}”"
                        f" cls={w['cls'][:24]} rect={w['rect']}"
                        + (" flags=" + ",".join(flags) if flags else ""))
            except Exception as e:
                log(f"原始清单导出失败: {e}")
        finally:
            pass

    def _on_takeover_found(self, payload: dict):
        """探测成功（GUI 线程）：弹确认框（阵营自动推断）-> 采纳 -> 全自动。"""
        from recognition.classifier import grid_to_fen
        import capture_win
        self._sync_log("探测成功，进入采纳流程")
        hwnd = payload["hwnd"]
        res = payload["res"]
        title = payload["title"]
        img = payload["img"]
        if payload.get("settling"):
            # 结算界面：无需选阵营与核对局面，直接连上窗口由续局逻辑开下一局
            self._sync_hwnd = hwnd
            self._takeover_hwnd = hwnd
            self._takeover_title = title
            self._settling = True
            self._last_sync_res = res
            if res.board_box:
                self._last_board_box = res.board_box
            import capture_win
            capture_win.set_topmost(hwnd, True)
            try:
                import ctypes as _ct
                import ctypes.wintypes as _wt
                _ct.windll.user32.ShowWindow(_wt.HWND(hwnd), 5)
            except Exception:
                pass
            try:
                self._sct = __import__("mss").mss()
            except Exception:
                pass
            self.chk_continue.setChecked(True)
            self.chk_auto.setChecked(True)
            self._takeover = True
            self._sync_timer.start()
            self.btn_sync.setText("停止观察")
            self.btn_resync.setEnabled(True)
            self._reset_opening()
            self._sync_log("接管进入结算界面模式：等待自动续局")
            self._sync_status("检测到结算界面 ✓ 正在自动点「再来一局」开下一局…")
            return
        # 确认对话框：阵营按"你的棋子在下/上半屏"自动推断（可改），
        # 并让用户核对识别局面（核对过 = 干净皮肤模板，解决 車↔炮 类误差）
        import gui.takeover_dialog as _td
        dlg = _td.TakeoverDialog(self, red_at_top=res.red_at_top,
                                 window_title=title, n_pieces=res.n_pieces)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._takeover = False
            self.btn_takeover.setText("🎯 一键接管对局（自动找窗口·自动识别·自动走子）")
            self._sync_status("已取消接管。")
            return
        my_color = dlg.my_color
        my_turn = dlg.my_turn_now
        self._sync_log(f"确认完成：我执{'红' if my_color=='w' else '黑'}，"
                       f"轮到{'我方' if my_turn else '对方'}")
        # 局面核对（同框完成）：识别错处由用户修正 -> 干净皮肤模板
        vdlg = _td.VerifyBoardDialog(self, grid=res.grid, my_color=my_color)
        if vdlg.exec() != QDialog.DialogCode.Accepted or not vdlg.confirmed_grid:
            self._takeover = False
            self.btn_takeover.setText("🎯 一键接管对局（自动找窗口·自动识别·自动走子）")
            self._sync_status("已取消接管。", sticky=True)
            return
        verified_grid = vdlg.confirmed_grid
        # 核对界面里用户可能整盘翻转了方向（红黑上下识别反了）：网格已由
        # 编辑器翻转好，这里同步方向标记——点击坐标换算、同步基准、
        # 皮肤模板采集都以 red_at_top 为基准，必须保持一致
        if getattr(vdlg, "flipped", False):
            res.red_at_top = not res.red_at_top
            self._sync_log("核对界面翻转了棋盘方向（红黑上下识别反了）")
        res.grid = [row[:] for row in verified_grid]
        # 人工核对过的局面 = 可靠标签，采集皮肤模板供后续实时纠错
        try:
            from recognition import skin
            existing = skin.load_templates(skin.default_skin_path()) or {}
            tpl = skin.harvest(img, res.detections, res.board_box,
                               res.red_at_top, verified_grid, existing)
            skin.save_templates(skin.default_skin_path(), tpl)
            self._sync_log(f"皮肤模板已按核对局面校准（共 {len(tpl)} 类）")
        except Exception as e:
            self._sync_log(f"皮肤模板校准失败: {e}")
        self._sync_hwnd = hwnd
        self._takeover_hwnd = hwnd
        self._takeover_title = title          # 句柄失效时按标题重定位
        self._last_sync_res = res
        try:
            self._sct = __import__("mss").mss()
        except Exception as e:
            self._sync_status(f"初始化抓屏失败: {e}", sticky=True)
            return
        # 采纳局面；轮到谁由用户选择决定
        stm = my_color if my_turn else ("b" if my_color == RED else RED)
        fen = grid_to_fen(verified_grid, side_to_move=stm)
        self.game.reset(fen)
        self.board.set_game(self.game)
        self.board.set_my_color(my_color)
        self.panel.set_my_color(my_color)
        self.panel.clear_analysis()
        self.observer.force_baseline(res.grid, res.red_at_top)
        self.chk_auto.setChecked(True)
        self._takeover = True
        capture_win.set_topmost(hwnd, True)          # 置顶对局窗口
        try:
            import ctypes as _ct
            import ctypes.wintypes as _wt
            _u32 = _ct.windll.user32
            _u32.ShowWindow(_wt.HWND(hwnd), 5)       # SW_SHOW（防最小化）
            _u32.SetForegroundWindow(_wt.HWND(hwnd))  # 拉到前台
        except Exception:
            pass
        self.btn_sync.setText("停止观察")
        self.btn_resync.setEnabled(True)
        self.combo_window.blockSignals(True)
        # 同步窗口下拉里定位到该窗口（展示用）
        for i in range(self.combo_window.count()):
            if self.combo_window.itemData(i) == hwnd:
                self.combo_window.setCurrentIndex(i)
                break
        self.combo_window.blockSignals(False)
        self._sync_timer.start()                     # 启动观察循环
        self._reset_opening()
        self._last_activity_ts = time.time()
        side = "红方" if my_color == RED else "黑方"
        self._sync_status(
            f"接管成功 ✓ 已锁定 {side}（窗口「{title[:24]}」已置顶）。"
            + ("现在就轮到你，软件开始走子…"
               if my_turn else "等待对手走子，走完自动接管节奏。"))
        self._refresh_all()

    def _stop_takeover_cleanup(self):
        """接管收尾：取消置顶、复位按钮。"""
        if self._takeover_hwnd:
            try:
                capture_win.set_topmost(self._takeover_hwnd, False)
            except Exception:
                pass
        self._takeover = False
        self._takeover_hwnd = None
        self.btn_takeover.setText("🎯 一键接管对局（自动找窗口·自动识别·自动走子）")

    # ---------------------------------------------------------------- 窗口同步
    def _refresh_windows(self):
        """枚举可见窗口供选择（MuMu / JJ 客户端等）。"""
        try:
            import capture_win
            wins = capture_win.list_windows()
        except Exception as e:
            self.lbl_sync.setText(f"枚举窗口失败: {e}")
            return
        self.combo_window.clear()
        for winfo in wins:
            tag = "（已最小化）" if winfo["minimized"] else ""
            self.combo_window.addItem(f"{winfo['title'][:40]}{tag}",
                                      winfo["hwnd"])

    def _toggle_sync(self):
        if self._sync_hwnd:
            self._stop_sync()
            return
        hwnd = self.combo_window.currentData()
        if not hwnd:
            self._refresh_windows()
            hwnd = self.combo_window.currentData()
        if not hwnd:
            self.lbl_sync.setText("没有可选窗口。请先打开 MuMu/JJ 窗口再点「刷新」。")
            return
        try:
            import mss
            self._sct = mss.mss()
        except Exception as e:
            self.lbl_sync.setText(f"初始化抓屏失败: {e}")
            return
        self._sync_hwnd = hwnd
        self.observer.reset()
        self._sync_timer.start()
        self.btn_sync.setText("停止观察")
        self.btn_resync.setEnabled(True)
        self.lbl_sync.setText("观察中：等待画面稳定以建立基准…")

    def _stop_sync(self):
        self._sync_timer.stop()
        self._drop_ponder()
        self._sync_hwnd = None
        self._stop_takeover_cleanup()
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        self.btn_sync.setText("开始观察")
        self.btn_resync.setEnabled(False)
        self.lbl_sync.setText("已停止观察。")

    def _sync_log(self, text: str):
        """同步环节事件写日志（接管排查看这里）。"""
        try:
            p = os.path.join(os.path.expanduser("~"), "xq_assistant_takeover.log")
            with open(p, "a", encoding="utf-8") as f:
                f.write("[" + time.strftime("%H:%M:%S") + "] " + text + "\n")
        except Exception:
            pass

    def _sync_status(self, text: str, sticky: bool = False):
        """更新同步状态；sticky 的警告在数秒内不被常规刷新覆盖。"""
        if sticky:
            self._status_hold_until = time.time() + 6
        elif time.time() < getattr(self, "_status_hold_until", 0):
            return
        self.lbl_sync.setText(text)

    def _grab_window(self):
        """抓取当前同步窗口的画面，返回 BGR ndarray；失败返回 None。"""
        import capture_win
        rect = capture_win.get_client_rect(self._sync_hwnd)
        if rect is None:
            return None
        if capture_win.is_minimized(self._sync_hwnd):
            self._sync_status("对局窗口已最小化，暂停同步（恢复窗口后自动继续）")
            return None
        left, top, right, bottom = rect
        w, h = right - left, bottom - top
        if w < 200 or h < 150:
            return None
        img = capture_win.capture_window_client(self._sync_hwnd)
        if img is not None:
            return img
        import mss
        raw = self._sct.grab({"left": left, "top": top, "width": w, "height": h})
        arr = np.asarray(raw)[:, :, :3]        # BGRA -> BGR
        return np.ascontiguousarray(arr)

    @staticmethod
    def _looks_like_emulator(title: str) -> bool:
        """窗口标题像模拟器（用于最小化自动恢复与句柄失效重定位）。"""
        t = (title or "").lower()
        return any(k in t for k in (
            "mumu", "模拟器", "安卓设备", "雷电", "ldplayer", "bluestacks",
            "nox", "夜神", "memu", "逍遥"))

    @staticmethod
    def _probe_priority(title: str, rect=None):
        """一键接管的窗口探测顺序：(组, -面积)。

        组 0 = 模拟器（MuMu 等，优先——裁剪/点击逻辑不变）；
        组 1 = 其它窗口（浏览器里的 JJ 网页版等兜底）。
        同组内大窗口优先（更可能是对局界面）。
        """
        emu = 0 if MainWindow._looks_like_emulator(title) else 1
        if rect:
            area = max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])
        else:
            area = 0
        return (emu, -area)

    def _grab_fail_heal(self) -> None:
        """抓屏失败的自愈：最小化 -> 自动恢复；句柄失效 -> 静默重定位。

        窗口最小化时此前是静默停顿（无日志、不恢复），用户只能手动重启
        接管——这里自动处理并把每一步写进日志。
        """
        import capture_win
        now = time.time()
        if now - getattr(self, "_grab_heal_at", 0.0) < 30:
            return
        self._grab_heal_at = now
        try:
            valid = capture_win.is_window_valid(self._sync_hwnd)
            mini = capture_win.is_minimized(self._sync_hwnd)
        except Exception:
            return
        if not valid:
            self._sync_log("对局窗口句柄已失效（窗口被关闭/重建）——"
                           "尝试自动重新锁定…")
            if not self._relocate_window_silent():
                self._sync_status("⚠ 对局窗口已关闭。请重新打开 MuMu 并点「一键接管」。",
                                  sticky=True)
            return
        if mini:
            self._sync_log("对局窗口已最小化——自动恢复窗口")
            if capture_win.restore_window(self._sync_hwnd):
                self._sync_status("对局窗口最小化，已自动恢复 ✓")
            else:
                self._sync_status("⚠ 对局窗口已最小化，自动恢复失败，请手动还原。",
                                  sticky=True)

    def _no_board_heal(self, res, img) -> None:
        """识别不到棋盘持续时的自愈与取证（按段限频）。

        ① 15 秒：存档画面 + 详因日志（下次有真凭实据可查）；
        ② 20 秒且窗口句柄失效：自动在同标题/模拟器窗口里重新锁定；
        ③ 每 30 秒心跳日志：区分"循环还活着但读不到"与"循环停了"。
        """
        import capture_win
        now = time.time()
        elapsed = now - self._no_board_since
        if elapsed >= 15 and not getattr(self, "_no_board_saved", False):
            self._no_board_saved = True
            self._save_no_board_frame(res, img)
        if (elapsed >= 20
                and now - getattr(self, "_relocate_at", 0.0) > 30
                and not capture_win.is_window_valid(self._sync_hwnd)):
            self._relocate_at = now
            self._relocate_window_silent()
        if now - getattr(self, "_no_board_hb", 0.0) > 30:
            self._no_board_hb = now
            try:
                valid = capture_win.is_window_valid(self._sync_hwnd)
                mini = capture_win.is_minimized(self._sync_hwnd)
                title = capture_win.window_title(self._sync_hwnd)[:20]
            except Exception:
                valid, mini, title = None, None, "?"
            self._sync_log(f"无棋盘持续 {elapsed:.0f}s：窗口“{title}” 有效={valid} "
                           f"最小化={mini} 来源={res.source} 错误={res.error or '无'}")
            if elapsed > 30:
                self._sync_status(
                    f"识别不到棋盘已 {elapsed / 60:.1f} 分钟——正在持续重试；"
                    f"如画面正常可点「重新同步」，或重开 JJ 对局后重新接管。",
                    sticky=True)

    def _save_no_board_frame(self, res, img) -> None:
        """无棋盘持续 15 秒时存档画面，便于事后诊断（皮肤/遮挡/窗口问题）。"""
        try:
            import cv2
            p = os.path.join(os.path.expanduser("~"), "xq_sync_no_board.png")
            cv2.imwrite(p, img)
            self._sync_log(f"无棋盘已 15 秒，画面已存档 {p}"
                           f"（来源={res.source} 错误={res.error or '无'}）")
        except Exception as e:
            self._sync_log(f"存档无棋盘画面失败：{e}")

    def _relocate_window_silent(self) -> bool:
        """静默重新锁定对局窗口（句柄失效时）。

        MuMu 重启/窗口重建会产生新句柄，旧句柄抓屏永远失败——此前只能
        手动重启接管。这里按原窗口标题或模拟器特征找到新窗口，识别到
        合法棋盘后直接切换，并重建同步基准。
        """
        import capture_win
        if not self._takeover:
            return False
        old_title = getattr(self, "_takeover_title", "") or ""
        try:
            wins = capture_win.list_windows()
        except Exception as e:
            self._sync_log(f"自动重定位：枚举窗口失败 {e}")
            return False
        cands = [w for w in wins if not w["minimized"]
                 and (w["title"] == old_title or self._looks_like_emulator(w["title"]))]
        # 当前句柄若还活着也在候选里（可能只是内容暂时读不到）
        from recognition.pipeline import recognize_screenshot
        from recognition.yolo_board import board_plausibility
        for w in cands:
            if w["hwnd"] == self._sync_hwnd:
                continue
            img = capture_win.capture_window_client(w["hwnd"])
            if img is None:
                continue
            res = recognize_screenshot(img, model_path=self._model_path())
            if res.source != "yolo" or res.n_pieces < 4:
                continue
            if not board_plausibility(res)[0]:
                continue
            old = self._sync_hwnd
            self._sync_hwnd = w["hwnd"]
            self._takeover_hwnd = w["hwnd"]
            self._takeover_title = w["title"]
            self._last_sync_res = res
            if res.board_box:
                self._last_board_box = res.board_box
            self.observer.force_baseline(res.grid, res.red_at_top)
            capture_win.set_topmost(w["hwnd"], True)
            self._no_board_since = 0.0
            self._sync_log(f"自动重新锁定窗口 ✓「{w['title'][:24]}」（旧句柄 {old} "
                           f"→ 新句柄 {w['hwnd']}，{res.n_pieces} 子）")
            self._sync_status("对局窗口句柄已变化，已自动重新锁定 ✓", sticky=True)
            return True
        self._sync_log(f"自动重定位未找到可用的对局窗口（候选 {len(cands)} 个）")
        return False

    def _sync_tick(self):
        try:
            self._sync_tick_inner()
        except Exception as e:
            self._sync_status(f"同步出错（已继续）: {e}")

    def _sync_tick_inner(self):
        self._check_auto_click_confirm()
        self._stall_watchdog()
        from PyQt6.QtWidgets import QApplication
        if QApplication.activeModalWidget() is not None:
            return                              # 对话框打开时不同步
        if not self._sync_hwnd or self._sct is None:
            return
        img = self._grab_window()
        if img is None:
            # 抓屏失败/窗口最小化：也要走自愈限频（否则静默卡住、无日志可查）
            self._grab_fail_heal()
            return
        from recognition.pipeline import recognize_screenshot
        from recognition.yolo_board import grid_plausibility
        model_path = self._model_path()
        res = recognize_screenshot(img, model_path=model_path)
        board_unreadable = res.source != "yolo" or res.n_pieces < 4
        if board_unreadable:
            if self._no_board_since == 0.0:
                self._no_board_since = time.time()
        else:
            self._no_board_since = 0.0
            if res.board_box:
                self._last_board_box = res.board_box   # 坐标锚点（续局用）
        # 自动续局：每帧都检查「再来一局」按钮——**按钮的存在本身就是
        # 结算界面最可靠的标志**。不能依赖"对局已结束"（对手认输时本地
        # 判不出胜负）或"棋盘读不到"（结算界面棋盘往往完整可见）。
        # 为省 CPU 隔帧检查（约 1.4 秒一次）。
        self._continue_tick_n = getattr(self, "_continue_tick_n", 0) + 1
        if (self._continue_tick_n % 2 == 0
                or self.game.result is not None or board_unreadable):
            if self._auto_continue_tick(img):
                return
        if board_unreadable:
            if self._sync_state != "no_board":
                self._sync_state = "no_board"
                self._sync_log(f"未识别到棋盘（弹窗/未进对局），暂停同步——"
                               f"来源 {res.source}，错误 {res.error or '无'}")
            self._sync_status("未识别到棋盘（可能有弹窗/未进对局）。棋盘出现后自动恢复。")
            self._no_board_heal(res, img)
            self.observer.feed(None, False)
            return
        # 新开局采纳：await 模式下出现"完整合法棋盘"即为新对局
        # （结算弹窗关闭后；若棋盘仍被遮挡，上方 no_board 分支已先处理）
        if self._await_new_game:
            ok0, _r0 = grid_plausibility(res.grid)
            if ok0:
                self._adopt_new_game(res)
                return
        # 帧质量阶梯：标准阈值 -> 低阈值复核 -> 判定"棋盘不可用"
        # （避免长时间静默丢帧：实际发生过连续 562 次丢帧、几分钟无动作）
        def _frame_bad(r):
            ok_, why_ = grid_plausibility(r.grid)
            if not ok_:
                return why_
            if (self._sync_last_npieces
                    and r.n_pieces <= self._sync_last_npieces - 4):
                return "子数骤降"
            return None

        reason = _frame_bad(res)
        if reason is not None:
            # 低阈值复核：识别漏检（将帅被漏掉等）往往能靠降阈值找回
            low = self._recognize_low_conf(img)
            if low is not None and _frame_bad(low) is None:
                self._sync_log(f"标准阈值帧校验未过（{reason}）"
                               f"→ 低阈值复核通过，采用复核结果")
                res = low
                reason = None
        if reason is not None:
            self._dropped_frames = getattr(self, "_dropped_frames", 0) + 1
            if self._dropped_frames % 10 == 1:
                self._sync_log(f"丢弃疑似残缺帧（{reason}，"
                               f"第 {self._dropped_frames} 次）")
            # 取证：每个丢帧段的首帧就存档（限频 90 秒）。此前只每 30 帧
            # 存一次，间歇性误检（多一个炮/缺红帅）从不留下证据可查
            if (self._dropped_frames == 1
                    and time.time() - getattr(self, "_fail_save_at", 0.0) > 90):
                self._fail_save_at = time.time()
                self._save_fail_frame(img, reason)
            # 连续多帧不可用 -> 升级为"棋盘不可用"：允许续局逻辑介入
            # 并给出提示，而不是无声地一直丢帧
            if self._dropped_frames >= 5:
                if self._no_board_since == 0.0:
                    self._no_board_since = time.time()
                self._sync_status(f"画面持续无法识别（{reason}）——已按棋盘不可用处理")
                if self._auto_continue_tick(img):
                    return
                if self._dropped_frames % 30 == 1:
                    self._save_fail_frame(img, reason)
            else:
                time.sleep(0.2)
            return
        if self._dropped_frames:
            if self._dropped_frames >= 5:
                self._sync_log(f"画面恢复正常（连续异常 {self._dropped_frames} 帧后）")
            self._dropped_frames = 0
        self._sync_last_npieces = res.n_pieces
        # 自愈：本地误判"对局结束"会导致后续同步与引擎全部停摆（静默卡死）。
        # 若窗口局面与本地方案明显不同，说明"结束"判断有误 -> 清除并按 JJ 重新同步。
        if self.game.result is not None:
            diff = sum(1 for r in range(10) for c in range(9)
                       if self.game.board.grid[r][c] != res.grid[r][c])
            if diff >= 2:
                self._sync_log(f"本地判定已结束（{self.game.result}）但窗口局面"
                               f"不同（{diff} 格）——判定有误，清除结果并重新同步")
                self.game.result = None
                self._last_auto_resync = 0.0
                self._auto_resync_to_window(res, None)
                return
        # 静默自诊断：长时间无任何事件时记录完整状态，便于事后定位停滞原因
        now2 = time.time()
        if (self._last_activity_ts and now2 - self._last_activity_ts > 15
                and not self._search_in_flight and not self._clicking
                and self._auto_click_pending is None):
            self._last_activity_ts = now2
            try:
                eng = ("存活" if (self.engine and self.engine.is_running())
                       else ("未启动" if self.engine is None else "已退出"))
            except Exception:
                eng = "未知"
            self._sync_log(
                f"静默诊断：轮次={'红' if self.game.side_to_move == 'w' else '黑'} "
                f"我方={'红' if self.board.my_color == 'w' else ('黑' if self.board.my_color == 'b' else '未设定')} "
                f"结果={self.game.result or '无'} 分析中={self._search_in_flight} "
                f"引擎={eng} 帧={res.n_pieces}子 待确认代点={self._auto_click_pending is not None}")
        if self._sync_state == "no_board":
            self._sync_state = ""
            self._no_board_saved = False
            self._no_board_hb = 0.0
            self._sync_log("棋盘重新出现，恢复同步")
        events = self.observer.feed(res.grid, res.red_at_top)
        self._handle_sync_events(events, res, img)

    def _handle_sync_events(self, events, res, img=None):
        self._last_sync_res = res
        # 对局结束 -> 进入"等待新开局"模式：后续画面（结算弹窗遮挡的乱局）
        # 不再当棋盘同步/重同步——那是之前"垃圾帧重同步死循环"的根因。
        # 只等一张完整合法的新开局棋盘出现才采纳。
        if (self._await_new_game or (self.game.result is not None
                                     and self.chk_continue.isChecked())):
            if events and events[0]["type"] == "baseline":
                return
            from recognition.yolo_board import grid_plausibility
            ok0, _r0 = grid_plausibility(res.grid)
            n0 = res.n_pieces
            if events and events[-1].get("type") != "baseline":
                self._sync_status("对局结束，等待新开局（自动点「再来一局」）…")
            return
        self._last_sync_res = res
        if events:
            self._last_activity_ts = time.time()
        if not events:
            extra = f"，皮肤模板修正 {len(res.corrections)} 处" if res.corrections else ""
            st = f"同步中（{res.n_pieces} 子{extra}）"
            if st != self._sync_state:
                self._sync_state = st
                self._sync_log(f"同步正常（{res.n_pieces} 子"
                               + (f"，模板修正 {len(res.corrections)}" if res.corrections else "")
                               + "）")
            self._sync_status(st)
            return
        for ev in events:
            kind = ev["type"]
            if kind == "baseline":
                self._sync_log("锁定对局基准")
                self._sync_status("已锁定对局基准 ✓ 请确认左侧棋盘与 JJ 一致；"
                                  "之后双方着法将自动同步。")
            elif kind == "changed":
                new_grid = ev["grid"]
                same = all(new_grid[r][c] == self.game.board.grid[r][c]
                           for r in range(10) for c in range(9))
                if same:
                    self._sync_status("同步中（画面与棋盘一致）")
                    continue
                # 转移约束推断：从合法走法中解释观察帧（容错同色异子误读）
                from core.board import (infer_move_from_observation,
                                        move_to_chinese, move_to_iccs)
                mv, score = infer_move_from_observation(self.game.board, new_grid)
                if mv is None and self._auto_click_pending is not None:
                    # 我方代点与对手快速应着落进同一观察帧：两步假设
                    # （我方已知着法 + 对方合法应着）联合解释
                    first = self._auto_click_pending["move"]
                    try:
                        tmp = Board(self.game.board.to_fen())
                        tmp.make_move(first)
                        reply, score2 = infer_move_from_observation(tmp, new_grid)
                    except Exception:
                        reply = None
                    if reply is not None:
                        mv, score = first, 1.0
                        self._pending_reply = reply
                # 低阈值复核（多格路径）：检测抖动会让棋子"凭空消失"，
                # 降阈值重识别往往能找回，优先用复核结果解释变化
                if mv is None:
                    low = self._recognize_low_conf(img)
                    if low is not None:
                        from recognition.yolo_board import grid_plausibility as _gp
                        if _gp(low.grid)[0]:
                            mv_low, sc_low = infer_move_from_observation(
                                self.game.board, low.grid)
                            if mv_low is not None:
                                self._sync_log(
                                    f"标准阈值下无法解释，低阈值复核后可解释为"
                                    f" {move_to_iccs(mv_low)}——按复核结果同步")
                                mv, score, new_grid, res = (mv_low, sc_low,
                                                            low.grid, low)
                            elif low.n_pieces > res.n_pieces:
                                self._sync_log(
                                    f"低阈值复核更完整（{res.n_pieces}→"
                                    f"{low.n_pieces} 子），采用复核结果重新同步")
                                res = low
                if mv is None:
                    diff = [(r, c, self.game.board.grid[r][c], new_grid[r][c])
                            for r in range(10) for c in range(9)
                            if self.game.board.grid[r][c] != new_grid[r][c]]
                    # 类别自修复：模型对某颗棋子的一致性误读（如 马 被持续认成 車），
                    # 从走法几何反推真实兵种——起点重贴标签后走法立即可解释
                    if len(diff) == 2:
                        repaired = self._try_class_repair(diff, new_grid)
                        if repaired is not None:
                            mv, score = repaired, 0.99
                            self._sync_log(f"类别自修复：起点棋子重贴标签后"
                                           f"推断为 {move_to_iccs(mv)}")
                    # 单格差异分三类处理：
                    # A) 抬子思考（本地有子、观察无子）——对方拿起棋子在考虑，
                    #    只等待落子，绝不重同步（否则会把"少一颗子"的中间画面
                    #    采纳成棋盘，引擎对着缺子假局面思考甚至送非法局面崩溃）
                    # B) 落子完成（本地无、观察有，且与此前抬起的棋子吻合）——
                    #    与抬子格组合成走法，走正常记录流程
                    # C) 其它单格差异——持续 4 秒后按原逻辑复核/重同步
                    if mv is None and len(diff) == 1:
                        from core.board import side_of
                        (r, c, local_p, obs_p) = diff[0]
                        now = time.time()
                        lift = getattr(self, "_lift_key", None)

                        # B) 落子完成（含吃子：该格本地是我方被吃的子）
                        if (mv is None and obs_p is not None
                                and obs_p != local_p
                                and lift is not None and lift[2] == obs_p):
                            cand = ((lift[0], lift[1]), (r, c))
                            if self.game.board.is_legal_move(cand):
                                mv = cand
                                score = 0.99
                                self._lift_key = None
                                self._lift_since = 0.0
                                self._lift_warned = False
                                self._partial_key = None
                                self._sync_log("对方落子确认：结合此前抬子"
                                               f"推断走法 {move_to_iccs(mv)}")
                                # mv 已定，落到下方正常记录流程
                            else:
                                self._sync_status("对方落子中（落点暂不合法，"
                                                  "等待画面稳定）…")

                        # A) 抬子思考：等待落子，不重同步、不采纳本帧。
                        # 只有"当前行棋方自己的棋子消失"才可能是抬子考虑；
                        # 轮到我方时丢子、或非行棋方的棋子消失，只能是检测
                        # 失误/异常，落入 C 分支按时复核重同步。
                        if (mv is None and local_p is not None and obs_p is None
                                and side_of(local_p) == self.game.side_to_move):
                            if lift is None or lift[2] != local_p:
                                self._lift_key = (r, c, local_p)
                                self._lift_since = now
                                self._lift_warned = False
                                who = "红" if side_of(local_p) == "w" else "黑"
                                self._sync_log(f"检测到 {who}方棋子被抬起"
                                               f"（{local_p} 于 ({r},{c})）"
                                               f"——等待落子")
                            self._sync_status("对方思考中（棋子已抬起）"
                                              "——落子后自动继续…")
                            # 低阈值复核排除"检测漏检"（该格其实仍有子）
                            low = self._recognize_low_conf(img)
                            if low is not None:
                                from recognition.yolo_board import grid_plausibility
                                ok_low, _r = grid_plausibility(low.grid)
                                if ok_low and low.grid[r][c] is not None:
                                    self._sync_log("复核：该格棋子仍在"
                                                   "（检测漏检），忽略本帧")
                                    self._lift_key = None
                            elif (now - self._lift_since > 60
                                  and not getattr(self, "_lift_warned", False)):
                                self._lift_warned = True
                                self._sync_status("⚠ 对方长时间未落子，"
                                                  "请人工确认对局状态", sticky=True)
                            continue

                        # C) 其它单格差异：持续 4 秒后复核/重同步（原有逻辑）
                        if mv is None:
                            key = (r, c, local_p, obs_p)
                            if getattr(self, "_partial_key", None) != key:
                                self._partial_key = key
                                self._partial_since = now
                            if now - self._partial_since > 4:
                                self._partial_key = None
                                # 关键复核：单格差异持续存在时先排除"检测漏检"——
                                # 用低阈值重新识别该格，若棋子其实还在，则判定为
                                # 漏检，保持本地棋盘（否则会把漏检当成真丢子，
                                # 本地少一颗子、引擎误判局势，实际发生过）。
                                low = self._recognize_low_conf(img)
                                if low is not None:
                                    from recognition.yolo_board import grid_plausibility
                                    ok_low, _r = grid_plausibility(low.grid)
                                    if ok_low:
                                        fr_c, fc_c = diff[0][0], diff[0][1]
                                        if (diff[0][2] is not None
                                                and low.grid[fr_c][fc_c] is not None):
                                            self._sync_log(
                                                f"单格差异 {key} 复核为检测漏检"
                                                f"（低阈值下该格仍有"
                                                f"{low.grid[fr_c][fc_c]}）——保持本地棋盘")
                                            self._sync_status(
                                                "该格复核为识别漏检，已保持棋盘不变 ✓")
                                            continue
                                        res = low      # 采纳更完整的低阈值结果
                                mover = None
                                if diff[0][2] is not None:
                                    from core.board import side_of
                                    mover = side_of(diff[0][2])
                                elif diff[0][3] is not None:
                                    from core.board import side_of
                                    mover = side_of(diff[0][3])
                                self._sync_log(f"单格差异超 4 秒未补全 {key}"
                                               f"——自动以 JJ 局面为准重新同步"
                                               f"（走子方={'红' if mover == 'w' else '黑'}）")
                                self._auto_resync_to_window(res, mover)
                            else:
                                self._sync_status("检测到走子进行中/单格异常，等待稳定…"
                                                  "（也支持你直接在 JJ 里手动走子）")
                            continue
                    if mv is None:
                        # 多格无法解释时，同样优先从差异内容推走子方
                        mover = None
                        for (r, c, lp, op) in diff:
                            if lp is not None:
                                from core.board import side_of
                                mover = side_of(lp)
                                break
                        self._sync_log(f"观察帧无法用合法走法解释（best={score:.3f}，"
                                       f"差异 {diff[:6]} 共 {len(diff)} 格）"
                                       f"——自动以 JJ 局面为准重新同步")
                        self._auto_resync_to_window(res, mover)
                        continue
                try:
                    cn = move_to_chinese(self.game.board, mv)
                except Exception:
                    cn = move_to_iccs(mv)
                mover_before = self.game.side_to_move
                if self.game.play(mv):
                    if self._auto_click_pending and                             move_to_iccs(self._auto_click_pending["move"]) == move_to_iccs(mv):
                        self._auto_click_pending = None   # 代点已被窗口画面确认
                    # 局面已变：清掉旧建议箭头（否则整段对手回合都指着一个
                    # 已经走掉的着法、对手走完后坐标全变——看起来就是乱指）
                    self.board.set_best_move(None)
                    self.board.animate_move(mv,
                                            self.game.board.piece_at(*mv[1]))
                    self.sound.play("capture"
                                    if self.game.history and
                                    self.game.history[-1][1] else "drop")
                    # 记录"走子方"（非走后行棋方），供重同步推断轮次
                    self._last_mover_side = ("b" if self.game.side_to_move == "w"
                                             else "w")
                    self._confirm_opening_move(mv)     # 开局着法已确认走出
                    # 对手着法落定：先结清预计算（命中则直接复用深算，省掉整次重算）
                    if (self.board.my_color is not None
                            and mover_before != self.board.my_color):
                        self._resolve_ponder(mv)
                    self._refresh_all()
                    who = "红" if self.game.side_to_move == "b" else "黑"
                    self._sync_log(f"同步着法：{who}方 {cn}（推断分 {score:.3f}）")
                    reply = getattr(self, "_pending_reply", None)
                    if reply is not None:
                        self._pending_reply = None
                        if self.game.play(reply):
                            self.board.set_best_move(None)
                            self.board.animate_move(
                                reply, self.game.board.piece_at(*reply[1]))
                            if self.board.my_color is not None:
                                self._resolve_ponder(reply)
                            self._refresh_all()
                            self._last_mover_side = self.game.side_to_move
                            self._sync_log("同步着法：对手应着（两步合帧解释）")
                        else:
                            self._auto_resync_to_window(res, None)
                    self._sync_status(f"已自动同步着法：{cn}")
                    # 落定后若轮到对手：让引擎预计算我方应着——代替原本
                    # 送给"对手局面"的无用分析（对手是人类，不会照它走）
                    if (self.board.my_color is not None
                            and self.game.result is None
                            and self.game.side_to_move != self.board.my_color):
                        self._arm_ponder()
                else:
                    self._sync_log(f"推断着法({move_to_iccs(mv)})落子失败"
                                   f"——自动以 JJ 局面为准重新同步")
                    self._auto_resync_to_window(res, None)
            elif kind == "orientation_changed":
                self._sync_log("画面方向变化")
                self._sync_status(
                    "⚠ 画面方向变化（红上↔红下）。请重新同步或重新导入。",
                    sticky=True)

    def _try_class_repair(self, diff, new_grid):
        """类别自修复：本地某棋子被模型一致性误读时，从走法几何反推真实兵种。

        diff 恰好两格且字母相同（起点 P 消失、终点出现 P），但 P 的走法
        无法解释该位移（如"車"走出马步）——说明起点棋子真实类别不是 P。
        逐一同色兵种重贴标签试走，合法即确认：修正本地棋盘并返回走法。
        """
        if len(diff) != 2:
            return None
        # 起点/终点按"本地是否还有棋子"判别（diff 顺序是行优先，不保证先后）
        origin = next((c for c in diff if c[2] is not None), None)
        dest = next((c for c in diff if c[3] is not None), None)
        if origin is None or dest is None or origin == dest:
            return None
        fr, fc = origin[0], origin[1]
        tr, tc = dest[0], dest[1]
        pl, ql = origin[2], dest[3]
        if pl != ql:
            return None
        color = "w" if pl.isupper() else "b"
        letters = "KABNRCP" if color == "w" else "kabnrcp"
        for T in letters:
            if T == pl:
                continue
            trial = Board(self.game.board.to_fen())
            trial.grid[fr][fc] = T
            mv = ((fr, fc), (tr, tc))
            if trial.is_legal_move(mv):
                self.game.board.grid[fr][fc] = T     # 修复误读的类别
                self._sync_log(f"类别修复：({fr},{fc}) 的 {pl} 实为 {T}"
                               f"（由走法几何反推）")
                return mv
        return None

    def _fix_turn_by_check(self) -> bool:
        """若"非行棋方正被将军"，说明轮到谁猜错了 -> 自动翻转轮次。

        象棋规则：上一手不能让自己的将处于被将状态，因此"非行棋方被将军"
        的局面非法（Pikafish 会拒收并退出）。被将军的一方必然是该走棋的一方，
        据此可自动纠正轮次——比单纯拒绝更彻底。
        """
        board = self.game.board
        other = BLACK if board.side_to_move == RED else RED
        if not board.in_check(other):
            return False
        board.side_to_move = other
        self._sync_log(f"轮次自动纠正：{'红' if other == 'w' else '黑'}方正被将军，"
                       f"故应由该方走棋（原推断有误）")
        return True

    @staticmethod
    def _detect_systematic_shift(local_grid, observed, max_shift: int = 2):
        """观测帧是否只是本地棋盘的整体偏移（遮挡导致的映射伪影）。

        返回描述字符串（如 "(行+0, 列+1)"）；不疑似偏移返回 None。
        判据：某个非零偏移的吻合度明显高于原地吻合度。遮挡只会"少棋子"、
        不会凭空多出，因此无法解释的观测棋子额外扣分（权重 2）。
        """
        def fit(dr, dc):
            matched = extra = 0
            for r in range(10):
                for c in range(9):
                    obs_p = observed[r][c]
                    if obs_p is None:
                        continue
                    sr, sc = r - dr, c - dc
                    local_p = (local_grid[sr][sc]
                               if (0 <= sr < 10 and 0 <= sc < 9) else None)
                    if local_p == obs_p:
                        matched += 1
                    else:
                        extra += 1
            return matched - 2 * extra

        base = fit(0, 0)
        best_shift, best_score = None, base
        for dr in range(-max_shift, max_shift + 1):
            for dc in range(-max_shift, max_shift + 1):
                if dr == 0 and dc == 0:
                    continue
                sc_ = fit(dr, dc)
                if sc_ > best_score:
                    best_shift, best_score = (dr, dc), sc_
        if best_shift is not None and best_score >= base + 3 and best_score >= 6:
            return f"(行{best_shift[0]:+d}, 列{best_shift[1]:+d})"
        return None

    def _auto_resync_to_window(self, res, mover_side: Optional[str]):
        """本地棋盘与 JJ 脱节时，以 JJ 窗口局面为准自动重新同步（自愈）。

        JJ 是真实对局，永远是对的；本地只是镜像。限频防循环。
        """
        from recognition.classifier import grid_to_fen
        from recognition.yolo_board import grid_plausibility
        now = time.time()
        if now - self._last_auto_resync < 15:
            self._sync_status("检测到局面不同步，正在自动校准…")
            return
        ok, reason = grid_plausibility(res.grid)
        if not ok:
            self._sync_log(f"拒绝采纳残缺帧（{reason}）——等待完整帧")
            self._sync_status("捕获画面不完整，等待下一帧自动校准…")
            return
        # 系统性偏移守卫：弹窗/横幅遮住棋盘某条边时，模型会把棋盘框拟合到
        # "可见的那部分网格"上——整盘偏移一格却依然"看着合法"（比例和将帅
        # 位置都过得去）。这种帧绝不能当真相采纳（实际发生过：整体左移一格
        # 被采纳，本地棋盘全乱）。遮挡消失后画面自然一致，无需采纳。
        shift = self._detect_systematic_shift(self.game.board.grid, res.grid)
        if shift is not None:
            n_ref = getattr(self, "_shift_guard_n", 0) + 1
            self._shift_guard_n = n_ref
            if n_ref < 8:
                self._last_auto_resync = 0.0     # 不限频：下一帧继续判定
                self._sync_log(f"疑似遮挡导致识别整体偏移 {shift}——忽略本帧"
                               f"（第 {n_ref} 次）")
                self._sync_status("画面疑似被弹窗遮挡、识别发生整体偏移，"
                                  "等待画面恢复…", sticky=True)
                return
            self._sync_log(f"识别偏移已持续 {n_ref} 帧——不再等待，"
                           f"仍按 JJ 局面采纳")
        self._shift_guard_n = 0
        self._last_auto_resync = now
        # 重同步后，此前记录的"抬子等待"状态已失效
        self._lift_key = None
        self._lift_since = 0.0
        self._lift_warned = False
        # 轮次推断分两级，避免"该我方走却判成对手"而卡死：
        # ① 先用合法着法验证——谁的着法能解释"本地局面→观察帧"的变化，
        #    走子方就是谁（比只看棋子颜色/上一步记录可靠得多）；
        # ② 仍无法解释（大幅误读）时，按场景保守推断：我方点击在飞行中
        #    说明我方刚落子（轮对手），否则之前等的对手已走（轮我方）。
        from core.board import Board as _B, infer_move_from_observation
        old_fen = self.game.board.to_fen()
        order = ([mover_side] if mover_side in ("w", "b") else []) +                 [s for s in ("w", "b") if s != mover_side]
        explained = None
        for side in order:
            probe = _B(old_fen)
            probe.side_to_move = side
            mv2, sc2 = infer_move_from_observation(probe, res.grid)
            if mv2 is not None:
                explained = (side, mv2, sc2)
                break
        if explained is not None:
            side, mv2, sc2 = explained
            self._sync_log(f"轮次推断：{'红' if side == 'w' else '黑'}方的一步着法"
                           f"（{move_to_iccs(mv2)}，分 {sc2:.3f}）解释了这次变化")
            stm = "b" if side == "w" else "w"
            self._last_mover_side = side
        else:
            me = self.board.my_color
            if me and self._auto_click_pending is not None:
                stm = "b" if me == "w" else "w"      # 我方点击在飞行 -> 大概率我方刚落子
                why = "我方代点可能已落子"
            elif me:
                stm = me                              # 之前在等对手 -> 对手已走
                why = "此前在等对手走棋"
            else:
                stm = self.game.side_to_move
                why = "无从判断，沿用原轮次"
            self._sync_log(f"轮次保守推断：{why} -> 轮到"
                           f"{'红' if stm == 'w' else '黑'}方")
        self._sync_log(f"重新同步：局面取自 JJ，轮到 "
                       f"{'红' if stm == 'w' else '黑'}方")
        fen = grid_to_fen(res.grid, side_to_move=stm)
        self._drop_ponder()            # 局面以 JJ 为准重置：预计算作废
        self.game.reset(fen)
        self._fix_turn_by_check()          # 轮次猜错时按"被将军方该走"自动纠正
        self.board.set_game(self.game)
        self.observer.force_baseline(res.grid, res.red_at_top)
        self.board.set_best_move(None)
        # 清空代点拦截标记与待确认状态：局面已变，旧着法不应再被拦住
        self._clicked_key = None
        self._auto_click_pending = None
        self._pending_reply = None
        self._partial_key = None
        self._sync_last_npieces = res.n_pieces
        self._refresh_all()
        self._sync_status("检测到本地与 JJ 局面不一致，已自动以 JJ 为准重新同步 ✓"
                          "（如反复出现请检查识别模型/皮肤模板）", sticky=True)

    def _resync_baseline(self):
        """以当前对局窗口画面强行重建同步基准。"""
        if not self._sync_hwnd or self._sct is None:
            return
        img = self._grab_window()
        if img is None:
            self._sync_status("抓屏失败，无法重新同步。")
            return
        from recognition.pipeline import recognize_screenshot
        res = recognize_screenshot(img, model_path=self._model_path())
        if res.source != "yolo" or res.n_pieces < 4:
            self._sync_status("当前画面未识别到棋盘，无法重新同步。")
            return
        self.observer.force_baseline(res.grid, res.red_at_top)
        self._drop_ponder()
        # 手动重新同步 = 用户显式要求重试：解除引擎崩溃局面黑名单
        # （否则同一局面重新同步后依然"暂停送分析"，永远无着法）
        self._engine_kill_fen = None
        self._kill_logged = None
        self._clicked_key = None
        # 棋盘若与窗口不同步，提示用户（不自动改棋盘，避免误改）
        same = all(res.grid[r][c] == self.game.board.grid[r][c]
                   for r in range(10) for c in range(9))
        if same:
            self._sync_status("已重建同步基准 ✓ 棋盘与窗口一致，继续自动同步。")
        else:
            self._sync_status(
                "已重建同步基准。⚠ 注意：窗口局面与左侧棋盘不一致，"
                "请用「悔棋」/「手动摆局」或重新导入截图把棋盘对齐，"
                "否则同步的着法可能不合法。")

    def _save_fail_frame(self, img, reason: str) -> None:
        """连续无法识别时存档画面，便于事后诊断（皮肤/遮挡问题）。"""
        try:
            import cv2
            p = os.path.join(os.path.expanduser("~"), "xq_sync_fail.png")
            cv2.imwrite(p, img)
            self._sync_log(f"已存档无法识别的画面到 {p}（原因：{reason}）")
        except Exception:
            pass

    def _recognize_low_conf(self, img):
        """以更低的检测阈值重新识别（用于复核疑似漏检的格子）。

        漏检多与置信度临界有关；降低阈值能找回棋子，
        配合"本地棋盘合法性"判断可安全使用。
        """
        try:
            from recognition.pipeline import get_detector
            det = get_detector(self._model_path())
            if det is None:
                return None
            saved = det.conf
            try:
                det.conf = 0.22
                dets = det.detect(img)
            finally:
                det.conf = saved
            from recognition.yolo_board import board_from_detections
            from recognition import skin
            tpl = skin.load_templates(skin.default_skin_path())
            br = board_from_detections(dets, img=img, skin_templates=tpl)
            if br is None:
                return None

            class _R:
                pass
            r = _R()
            r.grid, r.red_at_top, r.n_pieces = br.grid, br.red_at_top, br.n_pieces
            r.board_box, r.detections = br.board_box, dets
            r.corrections, r.source, r.error = br.corrections, "yolo", None
            return r
        except Exception as e:
            self._sync_log(f"低阈值复核失败: {e}")
            return None

    def _model_path(self):
        name = self.config.get("model_name", "")
        if not name:
            return None
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", name)
        return p if os.path.exists(p) else None

    # ---------------------------------------------------------------- 手动摆局
    def _open_setup(self):
        """手动摆局：不依赖截图识别的可靠入口，可摆任意中局/残局。"""
        if self.playing:
            self._stop_battle()
        cur = self.game.board.to_fen()
        dlg = SetupDialog(self, fen=cur,
                          my_color=self.board.my_color or RED)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.fen:
            self.game.reset(dlg.fen)
            self.board.set_game(self.game)
            self.board.set_my_color(dlg.my_color)
            self.panel.set_my_color(dlg.my_color)
            self.panel.clear_analysis()
            self._refresh_all()
            side = "红方" if dlg.my_color == RED else "黑方"
            self.statusBar().showMessage(
                f"局面已摆好：我方执{side}。可直接「开始对战」，"
                f"或继续摆子后再开始。")

    # ---------------------------------------------------------------- 导入
    def _open_import(self, preloaded=None):
        """打开导入对话框。preloaded 为现成的 BGR 图像（来自拖拽）。"""
        if self.playing:
            self._stop_battle()
        dlg = ImportDialog(self, model_name=self.config.get("model_name", ""))
        try:
            if preloaded is not None:
                dlg._set_image(preloaded)
            else:
                # 未拖入图片时，自动尝试剪贴板里的截图，省一步操作
                from PyQt6.QtWidgets import QApplication
                from gui.import_dialog import image_from_mime
                mime = QApplication.clipboard().mimeData()
                img = image_from_mime(mime)
                if img is not None:
                    dlg._set_image(img)
        except Exception:
            pass
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.fen:
            self.game.reset(dlg.fen)
            self.board.set_game(self.game)
            self.board.set_my_color(dlg.my_color)
            self.panel.set_my_color(dlg.my_color)
            self.panel.clear_analysis()
            self._refresh_all()
            side = "红方" if dlg.my_color == RED else "黑方"
            self.statusBar().showMessage(
                f"已接管局面：我方执{side}。对方走子后请在棋盘上把对方棋子拖到目标位置。")

    # ---------------------------------------------------------------- 设置
    def _toggle_mute(self, muted: bool) -> None:
        """静音开关：静音时关闭音效与绝杀语音播报，开声时一并恢复。"""
        self.config["sound"] = not muted
        self.config["voice"] = not muted
        self.sound.set_enabled(not muted)
        self.sound.set_voice(not muted)
        self._refresh_mute_btn()
        self.statusBar().showMessage(
            "已静音（点「🔇 已静音」恢复声音）" if muted else "声音已开启",
            4000)

    def _refresh_mute_btn(self) -> None:
        """按当前配置刷新静音按钮的文字/勾选状态。"""
        if not hasattr(self, "btn_mute"):
            return
        muted = not self.config.get("sound", False)
        self.btn_mute.setChecked(muted)          # 状态相同则不发信号
        self.btn_mute.setText("🔇 已静音" if muted else "🔊 声音开")

    def _apply_human_mouse(self) -> None:
        """把拟人化鼠标设置应用到输入层（点击时即时生效）。"""
        try:
            import capture_win
            capture_win.HUMAN_INPUT.update({
                "enabled": bool(self.config.get("human_mouse", True)),
                "method": self.config.get("human_path", "jerk"),
                "speed": self.config.get("human_speed", "normal"),
            })
        except Exception:
            pass

    def _open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("引擎设置")
        form = QFormLayout(dlg)

        sp_threads = QSpinBox()
        sp_threads.setRange(1, 256)
        sp_threads.setValue(self.config["threads"])
        sp_hash = QSpinBox()
        sp_hash.setRange(16, 8192)
        sp_hash.setValue(self.config["hash_mb"])
        sp_hash.setSuffix(" MB")
        sp_multipv = QSpinBox()
        sp_multipv.setRange(1, 10)
        sp_multipv.setValue(self.config["multipv"])
        sp_depth = QSpinBox()
        sp_depth.setRange(1, 128)
        sp_depth.setValue(self.config["depth"])
        sp_movetime = QSpinBox()
        sp_movetime.setRange(0, 600000)
        sp_movetime.setSingleStep(1000)
        sp_movetime.setValue(self.config["movetime_ms"])
        sp_movetime.setSuffix(" ms")
        models = available_models()
        self.combo_model = QComboBox()
        self.combo_model.addItem("自动（优先中模型）", "")
        for m in models:
            self.combo_model.addItem(m, m)
        idx = self.combo_model.findData(self.config.get("model_name", ""))
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        else:
            self.combo_model.setCurrentIndex(0)

        sp_battle = QSpinBox()
        sp_battle.setRange(0, 600000)
        sp_battle.setSingleStep(500)
        sp_battle.setValue(self.config.get("battle_movetime_ms", 3000))
        sp_battle.setSuffix(" ms")
        note = QLabel("分析模式：每步用时 > 0 时按固定时间思考，否则按下面的深度上限。\n"
                      "对战模式：每步用时上限（深度仍封顶于「深度」值，时间先到即停）。\n"
                      "两者都是「时间 + 深度」双限制，时间到即出招，不会降低棋力上限。\n"
                      "线程/置换表修改后需重启软件生效。")
        note.setWordWrap(True)

        form.addRow("思考线程数", sp_threads)
        form.addRow("置换表大小", sp_hash)
        form.addRow("候选着法数 MultiPV", sp_multipv)
        form.addRow("深度上限", sp_depth)
        form.addRow("分析模式每步用时", sp_movetime)
        form.addRow("对战模式每步用时", sp_battle)
        form.addRow("截图识别模型", self.combo_model)

        self.chk_human = QCheckBox("拟人化鼠标（按真人轨迹移动，含加速/微抖/悬停）")
        self.chk_human.setChecked(self.config.get("human_mouse", True))
        self.combo_path = QComboBox()
        self.combo_path.addItem("最小急动度（平滑，推荐）", "jerk")
        self.combo_path.addItem("物理模拟 WindMouse（弧线更明显）", "wind")
        i_p = self.combo_path.findData(self.config.get("human_path", "jerk"))
        self.combo_path.setCurrentIndex(i_p if i_p >= 0 else 0)
        self.combo_speed = QComboBox()
        for label, key in (("极快（不模拟轨迹，直接两次点击）", "ultra"),
                           ("快", "fast"), ("中", "normal"), ("慢", "slow")):
            self.combo_speed.addItem(label, key)
        i_s = self.combo_speed.findData(self.config.get("human_speed", "normal"))
        self.combo_speed.setCurrentIndex(i_s if i_s >= 0 else 2)
        form.addRow(self.chk_human)
        form.addRow("轨迹算法", self.combo_path)
        form.addRow("操作速度", self.combo_speed)
        self.chk_sound = QCheckBox("音效（抬子/落子/吃子/将军/绝杀锣声）")
        self.chk_sound.setChecked(self.config.get("sound", True))
        self.chk_voice = QCheckBox("绝杀/困毙时语音播报")
        self.chk_voice.setChecked(self.config.get("voice", True))
        form.addRow(self.chk_sound)
        form.addRow(self.chk_voice)
        note2 = QLabel("识别模型：中模型精度最好（推荐，默认），"
                       "小模型更快但实测精度略低，"
                       "万能带旋转适合有旋转/透视的棋盘截图。")
        note2.setWordWrap(True)
        form.addRow(note2)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.config.update({
                "threads": sp_threads.value(),
                "hash_mb": sp_hash.value(),
                "multipv": sp_multipv.value(),
                "depth": sp_depth.value(),
                "movetime_ms": sp_movetime.value(),
                "battle_movetime_ms": sp_battle.value(),
                "model_name": self.combo_model.currentData(),
                "sound": self.chk_sound.isChecked(),
                "voice": self.chk_voice.isChecked(),
                # 拟人化三项此前漏存：选完不落盘，重启即还原（实际发生过的坑）
                "human_mouse": self.chk_human.isChecked(),
                "human_path": self.combo_path.currentData(),
                "human_speed": self.combo_speed.currentData(),
            })
            self._save_config()
            if self.engine is not None:
                self.engine.set_engine_options(multipv=self.config["multipv"])
            self.sound.set_enabled(self.config["sound"])
            self.sound.set_voice(self.config["voice"])
            self._refresh_mute_btn()
            self._apply_human_mouse()
            self.statusBar().showMessage("设置已保存（线程/置换表重启后生效）")

    # ---------------------------------------------------------------- 生命周期
    def closeEvent(self, event):
        if self._sync_hwnd:
            self._stop_sync()
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        super().closeEvent(event)
