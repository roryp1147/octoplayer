#!/usr/bin/env python3
"""
Octoplayer - entry point.

    python main.py [audio files...] [--source auto|local|loopback|synth|none]
                   [--on-top] [--controls] [--no-boot] [--quit-after N]

Env:
    OCTO_MUTE=1          never produce sound (tests)
    OCTO_SOURCE=synth    default visual source
    OCTO_NO_SETTINGS=1   don't read/write saved settings
    OCTO_NO_SMTC=1       don't touch the Windows media session
    OCTO_NO_BOOT=1       skip the wake-up sequence
    OCTO_FRAMELOG=path   write per-frame timing + brain position CSV on exit
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Octoplayer deep-sea media player")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--source", default=None)
    ap.add_argument("--on-top", action="store_true")
    ap.add_argument("--controls", action="store_true", help="control bar always out")
    ap.add_argument("--no-boot", action="store_true")
    ap.add_argument("--quit-after", type=float, default=0)
    args = ap.parse_args(argv)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from player import config
    from player.platform_win import set_app_user_model_id

    set_app_user_model_id("Octoplayer.MediaPlayer")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Octoplayer")
    app.setOrganizationName("Octoplayer")
    app.setQuitOnLastWindowClosed(True)
    icon = os.path.join(config.ASSETS, "octoplayer.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))

    from player.window import OctoWindow
    w = OctoWindow(source=args.source or config.SOURCE, files=args.files,
                   boot=False if args.no_boot else None)
    if args.on_top:
        w.set_on_top(True, save=False)
    if args.controls:
        w.controls_pinned = True
    w.show()
    if args.quit_after > 0:
        QTimer.singleShot(int(args.quit_after * 1000), w.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
