"""Paths, environment switches and timing constants for Octoplayer."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
LAYERS = os.path.join(ASSETS, "layers")
SOUNDS = os.path.join(ASSETS, "sounds")

IS_WINDOWS = sys.platform == "win32"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


MUTE = env_flag("OCTO_MUTE")                     # never create any audio output
SOURCE = os.environ.get("OCTO_SOURCE", "auto").strip().lower()
BOOT = not env_flag("OCTO_NO_BOOT")
FRAMELOG = os.environ.get("OCTO_FRAMELOG", "")   # path: per-frame timing/brain log (CSV)

FPS = 60                 # awake: always 60
FPS_SLEEP = 30           # asleep: dim, slow bubbles, still smooth
SLEEP_AFTER_S = 45.0     # no audio this long -> sleep
CONTROLS_IDLE_S = 2.5    # control bar retracts after this long without hover
RECOLOR_HZ = 20           # colour field refresh (the motion is always 60)
TENT_HZ = float(os.environ.get("OCTO_TENT_HZ", "30"))   # tentacle layer re-render rate (motion ~0.1 px/frame)
