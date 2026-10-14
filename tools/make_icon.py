#!/usr/bin/env python3
"""Render assets/octoplayer.ico (teal palette, transparent) from the live renderer (offscreen)."""
import os
import sys

os.environ.update({"OCTO_MUTE": "1", "OCTO_NO_SMTC": "1", "OCTO_NO_SETTINGS": "1", "QT_QPA_PLATFORM": "offscreen"})
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from player import art  # noqa: E402
from player.window import OctoWindow  # noqa: E402

w = OctoWindow(source="none", use_settings=False, boot=False)
w.t0 = w._now() - art.DRIFT_LEG_S   # -> teal
w.rebuild_tint(force=True)
w.pupil_off = [0, 0]
w._render_dome(0, full=True)
img = QImage(w.size(), QImage.Format.Format_ARGB32_Premultiplied)
img.fill(0)
w.render(img, QPoint(0, 0))
ax = int(round(w.L["axis_x"]))
side = 300
crop = img.copy(QRect(ax - side // 2, 8, side, side))
out = os.path.join(APP, "assets", "octoplayer.png")
crop.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation).save(out)
from PIL import Image  # noqa: E402  (build-time only)
im = Image.open(out).convert("RGBA")
im.save(os.path.join(APP, "assets", "octoplayer.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icon written")
