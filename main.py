# -*- coding: utf-8 -*-
"""象棋接管助手入口。

用法：
    .venv/Scripts/python main.py
"""
import os
import sys
import traceback


def _install_excepthook():
    """全局异常兜底：把未捕获异常写日志并弹框，而不是让进程直接消失。

    PyQt 中槽函数里的未捕获异常默认会让整个进程 abort（用户看到的就是
    "闪退"），这里接管后至少能留下可读的错误信息与日志。
    """
    log_path = os.path.join(os.path.expanduser("~"), "xq_assistant_error.log")

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n" + "-" * 60 + "\n")
        except Exception:
            pass
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None, "出错了",
                    f"发生未预期的错误，操作未完成：\n\n"
                    f"{exc_type.__name__}: {exc}\n\n"
                    f"详细信息已写入：\n{log_path}")
        except Exception:
            pass

    sys.excepthook = hook


_install_excepthook()

from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("象棋接管助手")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
