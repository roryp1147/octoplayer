"""UI sound bank: pre-rendered procedural WAVs played through QSoundEffect.
With OCTO_MUTE=1 no QSoundEffect is ever created (no audio device touched)."""
from __future__ import annotations

import os
import random

from PySide6.QtCore import QUrl

from . import config

# round 2: everything quiet by default (the WAVs themselves are already soft)
GAINS = {"sonar_ping": 0.3, "bubbles_sleep": 0.4, "sleep": 0.4, "wake": 0.4, "startle": 0.5,
         "tick_0": 0.5, "tick_1": 0.5}
MASTER = 0.35
GROUPS = {"squelch": ["squelch_0", "squelch_1", "squelch_2"],
          "bubble": ["bubble_0", "bubble_1", "bubble_2"],
          "pop": ["pop_0", "pop_1", "pop_2"],
          "tick": ["tick_0", "tick_1"]}


class SoundBank:
    def __init__(self, parent=None, enabled: bool = True):
        self.enabled = enabled and not config.MUTE
        self.muted = False
        self.master = MASTER
        self.fx = {}
        self.log: list[str] = []
        if not self.enabled:
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from .sfx_gen import SOUNDS, generate_all
            if any(not os.path.exists(os.path.join(config.SOUNDS, n + ".wav")) for n in SOUNDS):
                generate_all(config.SOUNDS, overwrite=False)
            for name in SOUNDS:
                pair = []
                for _ in range(2):
                    e = QSoundEffect(parent)
                    e.setSource(QUrl.fromLocalFile(os.path.join(config.SOUNDS, name + ".wav")))
                    e.setVolume(GAINS.get(name, 0.75) * self.master)
                    pair.append(e)
                self.fx[name] = pair
        except Exception as e:
            print(f"[octo] UI sounds disabled: {e}")
            self.fx = {}

    def play(self, name: str):
        if name in GROUPS:
            name = random.choice(GROUPS[name])
        self.log.append(name)
        if len(self.log) > 200:
            del self.log[:100]
        if not self.enabled or self.muted:
            return
        pair = self.fx.get(name)
        if pair:
            (pair[0] if not pair[0].isPlaying() else pair[1]).play()
