"""Octoplayer main window: a windowless, per-pixel-alpha deep-sea octopus (round 2).

Frame (60 fps, self-correcting single-shot timer):
  * colours: the flesh is B (+) M*T with T a live colour field - rings of hue around the
    head, a full colour wheel every ~24 s, a faster hue ripple, beat pulses, bass ripples
    running down each tentacle (art.py, palette.py). Recoloured at RECOLOR_HZ (~1-2 ms).
  * tentacles: ~150 radial strips drawn with per-strip rotation about the hub = travelling
    wave bending, independent per tentacle (tentacles.py).
  * dome: brain (sub-pixel drift), glass and spectrum dots composed offscreen; a soft-body
    mesh warps it when the head is poked.
  * controls live in a separate bioluminescent bar that slides out of the right ear
    (controlbar.py), so the octopus' own window never grows.
"""
from __future__ import annotations

import math
import os
import random
import time

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import (QColor, QCursor, QGuiApplication, QImage, QPainter, QPainterPath, QPen,
                           QPolygonF, QRadialGradient, QTransform)
from PySide6.QtWidgets import QFileDialog, QMenu, QWidget

from . import config
from .analysis import Analyzer
from .art import Art, gray_pixmap, mask_image, to_qimage
from .audioctl import AudioRouter, OwnTarget, WinAudioBackend
from .controlbar import ControlBar
from .media import MediaHub, make_provider
from .palette import shade, solid, wheel, wheel_many
from .softbody import SoftDome
from .sounds import SoundBank
from .sources import AUDIO_EXTS, AudioHub, LocalPlayer
from .tentacles import TentacleRig

CM = QPainter.CompositionMode
RIPPLE_SPEED = 520.0     # px/s: sucker flash travelling base -> tip
RIPPLE_W = 45.0
RIPPLE_LIFE = 1.0


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def qcol(c, a=1.0):
    return QColor.fromRgbF(float(min(1, max(0, c[0]))), float(min(1, max(0, c[1]))),
                           float(min(1, max(0, c[2]))), float(a))


class Knob:
    def __init__(self, name, lo, hi, value, default, steps, on_change, sounds):
        self.name, self.lo, self.hi, self.value, self.default = name, lo, hi, value, default
        self.steps, self.on_change, self.sounds = steps, on_change, sounds
        self.tag = ""
        self.shown_until = 0.0

    def set(self, v, now, sound=True):
        v = max(self.lo, min(self.hi, v))
        step = (self.hi - self.lo) / self.steps
        if round((v - self.lo) / step) != round((self.value - self.lo) / step) and sound:
            self.sounds.play("tick")
        self.value = v
        self.shown_until = now + 1.2
        self.on_change(v)

    def frac(self):
        return (self.value - self.lo) / (self.hi - self.lo)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "r", "life", "max", "kind", "wob")

    def __init__(self, x, y, vx, vy, r, life, kind):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, r
        self.life = self.max = life
        self.kind = kind
        self.wob = random.uniform(0, 6.28)


class OctoWindow(QWidget):
    def __init__(self, source: str = config.SOURCE, files=None, use_settings: bool = True,
                 media_provider=None, boot: bool | None = None, audio_backend=None, **_ignored):
        super().__init__(None)
        self.setWindowTitle("Octoplayer")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        self.art = Art()
        A = self.art
        L = A.L
        self.L = L
        self.setFixedSize(A.W, A.H)
        self.sounds = SoundBank(self)
        # ---- audio / media (Vexa architecture)
        self.local = LocalPlayer(self, audible=not config.MUTE)
        self.hub = AudioHub(self.local, source)
        self.audio_backend = audio_backend if audio_backend is not None else WinAudioBackend()
        self.own_target = OwnTarget(self.audio_backend, lambda: self.local.volume, self.local.set_volume)
        self.router = AudioRouter(self.audio_backend, self.own_target)
        self.provider = media_provider if media_provider is not None else make_provider()
        self.media = MediaHub(self.local, self.provider, clock=self._now)
        self.an = Analyzer(L["n_bands"])
        self.knobs = {
            "volume": Knob("volume", 0.0, 1.0, 0.6, 0.6, 20, self._on_volume, self.sounds),
            "balance": Knob("balance", -1.0, 1.0, 0.0, 0.0, 20, self._on_balance, self.sounds),
        }
        # ---- state
        self._virtual_clock = False
        self._last_t = time.monotonic()
        self.t0 = self._last_t
        self.ct = 0.0                 # colour clock (slows down when asleep)
        self.wt = 0.0                 # tentacle clock
        self.frame_count = 0
        self.frame_log = [] if config.FRAMELOG else None
        self.dt_hist = []
        self.paint_ms = []
        self.on_top = False
        self.controls_pinned = False
        self._drag = None
        self._press = None
        self.click_times = []
        self.flash_t0 = -1e9
        self.sleep_k = 1.0 if (config.BOOT if boot is None else boot) else 0.0
        self.sleeping = False
        self.last_active = self._now()
        self.boot_t0 = None
        self._boot = config.BOOT if boot is None else bool(boot)
        self.next_bubbles = 0.0
        self.particles: list[Particle] = []
        self.ripples = []
        self.beat = 0.0
        self._bass_prev = 0.0
        self._bass_s = 0.0
        self._last_ripple = -1e9
        self.pupil_off = [0.0, 0.0]
        self.level_s = 0.0
        self.next_blink = self._now() + 4.0
        self.blink_t0 = -1e9
        self.brain_pos = (0.0, 0.0)
        self.brain_rot = 0.0001
        self.brain_glow = 1.0
        self._rc_t = -1e9
        self._rc_part = None
        self._brain_t = -1e9
        self.hue = 0.0
        self.tint = wheel(0)
        self.eye_tint = wheel(180)
        # ---- tentacles, soft dome
        self.rig = TentacleRig(A)
        ty = max(0, int(self.rig.bbox[1]) - 90)
        self.tent_rect = QRect(0, ty, A.W, A.H - ty)
        self.tent_layer = QImage(A.W, A.H - ty, QImage.Format.Format_ARGB32_Premultiplied)
        self.tent_layer.fill(0)
        self.glow = 0.0
        self.throb = 1.0
        self.music = False
        self._applied_body = self._applied_tent = np.zeros(3, np.float32)
        r = A.dome_rect
        self.mesh = SoftDome(r.width(), r.height(), L["dome"]["cx"] - r.x(), L["dome"]["cy"] - r.y())
        self.dome_img = QImage(r.width(), r.height(), QImage.Format.Format_ARGB32_Premultiplied)
        self.brain_img = QImage(r.width(), r.height(), QImage.Format.Format_ARGB32_Premultiplied)
        self._build_static()
        self.recolor(force=True)

        self.settings = QSettings("Octoplayer", "Octoplayer") if use_settings and not config.env_flag("OCTO_NO_SETTINGS") else None
        self.bar = ControlBar(self)
        self._load_settings()
        self.local.set_volume(self.knobs["volume"].value)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._on_timer)
        self._next_deadline = None

        self.local.stateChanged.connect(self._on_local_state)
        self.local.trackChanged.connect(self._on_media_changed)
        self.media.changed.connect(self._on_media_changed)
        if files:
            self.local.set_files(files)
        print(f"[octo] {self.hub.describe()} mute={config.MUTE} "
              f"balance={'session' if self.audio_backend.ok else 'visual-only'} "
              f"smtc={'yes' if getattr(self.provider, 'available', False) else 'no'}"
              f"{'' if getattr(self.provider, 'available', False) else ' (' + str(getattr(self.provider, 'error', '')) + ')'}")

    # ================================================================ build
    def _build_static(self):
        A = self.art
        self.hit = (A.hit_alpha * 255).astype(np.uint8)
        self.dome_ellipse = (self.L["dome"]["cx"], self.L["dome"]["cy"], self.L["dome"]["rx"], self.L["dome"]["ry"])
        z = np.zeros_like(A.glass_dark)
        self.glass_dark_pm = to_qimage(np.dstack([z, z, z]), A.glass_dark)
        self.glass_hl_pm = gray_pixmap(A.glass_hl, (0.8, 0.95, 1.0)).toImage()
        self.glass_tint_mask = mask_image(np.clip(A.glass_tint * 3.0, 0, 1))
        self.flash_pm = gray_pixmap(A.flash_g * 0.9, (1.0, 0.95, 0.95))
        self.eye_masks = [mask_image(np.clip(m, 0, 1)) for m in A.eye_em]
        self.dot_masks = [mask_image(np.clip(gm, 0, 1)) for _, gm in A.dots]
        self.brain_em_mask = mask_image(np.clip(A.brain_em * 0.6, 0, 1))
        self.brain_img.fill(0)
        self.dome_img.fill(0)
        self.R_body = float(self.L["rmax"])
        self.hubx, self.huby = self.L["hub"]
        bl = self.L["brain"]
        self.brain_r = math.hypot(bl["cx"] - self.hubx, bl["cy"] - self.huby)

    # ================================================================ colour
    def _flash_k(self, now):
        u = (now - self.flash_t0) / 1.1
        if u < 0 or u > 1:
            return 0.0
        return math.sin(math.pi * min(1, u * 3)) if u < 0.17 else (1 - u) ** 2 / 0.69

    def target_colour(self, now=None):
        """The one body colour right now (sleep-dimmed, startle-flashed)."""
        now = self._now() if now is None else now
        c, h, fading = solid(self.ct)
        c = 1.0 - 0.92 * (1.0 - c)
        T = np.clip(c * (1.0 - 0.62 * self.sleep_k), 0, 1)
        fk = self._flash_k(now)
        if fk > 0:
            T = T * (1 - fk) + np.array([1.0, 0.25, 0.2], np.float32) * fk
        return T.astype(np.float32), h

    def recolor(self, force=False, part=None):
        """Round 3: ONE solid colour for the whole body. Only runs while the colour changes
        (cross-fades, sleep transitions, startle); part = 'body' / 'tent' splits the work over
        two frames. Beat brightness is NOT a recolour: it is an additive bloom of the cached
        glow planes (paintEvent)."""
        now = self._now()
        T, h = self.target_colour(now)
        self.hue = h
        self._rc_t = now
        q = qcol(T)

        def fill(p, rect):
            p.fillRect(0, 0, rect.width(), rect.height(), q)
        if part in (None, "tent"):
            self.tent_img = self.art.tent.recolor(fill)
            self.suck_pm = self._fill_mask(self.art.tent_s, np.clip(T * 0.5 + 0.5, 0, 1))
            self._tent_dirty = True
            self._applied_tent = T
            if part == "tent":
                return
        self._applied_body = T
        self.body_img = self.art.body.recolor(fill)
        self.dome_pm = self.art.dome.recolor(fill)
        self.tint = T
        # eyes: the contrasting (opposite) colour of the same sequence
        ec, _, _ = solid(self.ct, 180.0)
        e = shade(ec, 0.9, 1.0 - 0.55 * self.sleep_k, 0.08)
        fk = self._flash_k(now)
        if fk > 0:
            e = e * (1 - fk) + np.ones(3, np.float32) * fk
        self.eye_tint = e
        self.eye_pm = [self._fill_mask(m, e) for m in self.eye_masks]
        # dome dots follow the body colour, a little brighter
        dcol = np.clip(T * 0.85 + 0.2, 0, 1)
        self.dot_pm = [self._fill_mask(m, dcol) for m in self.dot_masks]
        dc = T
        self.glass_tint_pm = self._fill_mask(self.glass_tint_mask, dc * 0.35 + 0.05)
        self.brain_em_pm = self._fill_mask(self.brain_em_mask, np.clip(dc * 1.1 + 0.1, 0, 1))
        if force or now - self._brain_t > 0.12:
            A = self.art
            g = A.brain_gray[..., None] * dc * 0.8 + A.brain_rgb * 0.3
            self.brain_pm = to_qimage(g * (1 - 0.35 * self.sleep_k), A.brain_a)
            self._brain_t = now
        bar = getattr(self, "bar", None)
        if bar is not None:
            bar.colour = qcol(np.clip(self.tint * 1.1 + 0.12, 0, 1))
            bar.colour2 = qcol(np.clip(e * 1.05 + 0.1, 0, 1))
            bar.update()

    @staticmethod
    def _fill_mask(mask: QImage, colour) -> QImage:
        img = mask.copy()
        p = QPainter(img)
        p.setCompositionMode(CM.CompositionMode_SourceIn)
        p.fillRect(img.rect(), qcol(colour))
        p.end()
        return img

    # ================================================================ settings
    def _load_settings(self):
        s = self.settings
        if s is None:
            return
        try:
            self.knobs["volume"].value = float(s.value("volume", 0.6))
            self.knobs["balance"].value = float(s.value("balance", 0.0))
            self.local.set_volume(self.knobs["volume"].value)
            self.own_target.set_balance(self.knobs["balance"].value)
            self.controls_pinned = s.value("controls_pinned", "false") in (True, "true")
            self.sounds.muted = s.value("ui_sounds", "true") not in (True, "true")
            if s.value("on_top", "false") in (True, "true"):
                self.set_on_top(True, save=False)
            pos = s.value("pos")
            if isinstance(pos, QPoint) and QGuiApplication.screenAt(pos + QPoint(self.width() // 2, self.height() // 3)):
                self.move(pos)
        except Exception as e:
            print(f"[octo] settings: {e}")

    def _save_settings(self):
        s = self.settings
        if s is None:
            return
        s.setValue("volume", self.local.volume)
        s.setValue("balance", self.own_target.bal)
        s.setValue("controls_pinned", self.controls_pinned)
        s.setValue("ui_sounds", not self.sounds.muted)
        s.setValue("on_top", self.on_top)
        s.setValue("pos", self.pos())

    # ================================================================ timing
    def _now(self):
        return self._last_t if self._virtual_clock else time.monotonic()

    def showEvent(self, e):
        super().showEvent(e)
        if self.boot_t0 is None:
            self.boot_t0 = self._now()
            if not self._boot:
                self.sleep_k = 0.0
                self.recolor(force=True)
        if self.controls_pinned:
            self.bar.reveal(self._now())
        if not self._virtual_clock and not self.timer.isActive():
            self._next_deadline = time.monotonic()
            self._schedule()

    def _fps(self):
        busy = self.mesh.active or self.ripples or self.bar.busy() or self._flash_k(self._now()) > 0
        return config.FPS_SLEEP if (self.sleeping and self.sleep_k > 0.97 and not busy) else config.FPS

    def _schedule(self):
        period = 1.0 / self._fps()
        now = time.monotonic()
        self._next_deadline += period
        if self._next_deadline < now - period:      # fell behind (suspend etc.): resync
            self._next_deadline = now + period
        ms = max(0, int(round((self._next_deadline - now) * 1000)))
        self.timer.start(ms)

    def _on_timer(self):
        self.tick()
        if self.isVisible():
            self._schedule()

    def moveEvent(self, e):
        super().moveEvent(e)
        if hasattr(self, "bar"):
            self.bar.follow()

    # ================================================================ tick
    def tick(self, now=None):
        if now is not None:
            self._virtual_clock = True
            self._last_t = now
        else:
            now = time.monotonic()
        dt = max(1e-3, min(0.1, now - getattr(self, "_prev_tick", now - 1 / 60)))
        self._prev_tick = now
        self.frame_count += 1
        t = now - self.t0
        self.dt_hist.append(dt)
        if len(self.dt_hist) > 600:
            del self.dt_hist[:300]

        samples, sr = self.hub.latest(1024)
        self.an.process(samples, sr, dt)
        audio = not self.an.silent
        smtc = getattr(self.media, "smtc", None)
        smtc_playing = bool(smtc is not None and getattr(smtc, "playing", False))
        # wake on ANY sign of music: local playback, audible loopback (even quiet), SMTC 'playing'
        playing = (audio or self.an.audible_for > 0.15 or self.local.is_playing() or smtc_playing)
        self.music = playing

        # ---- boot / sleep
        if self.boot_t0 is not None and self._boot and now - self.boot_t0 < 2.6:
            self.sleep_k = 1 - ease((now - self.boot_t0) / 2.4)
        else:
            if playing:
                self.last_active = now
            want_sleep = now - self.last_active > config.SLEEP_AFTER_S     # sustained silence only
            if want_sleep and not self.sleeping:
                self.sleeping = True
            elif not want_sleep and self.sleeping:
                self.sleeping = False
                self.wake_t0 = now                       # smooth wake: stretch + bubbles
                self._spawn_bubbles(5)
            target = 1.0 if self.sleeping else 0.0
            if abs(self.sleep_k - target) > 1e-3:
                self.sleep_k += (target - self.sleep_k) * min(1, dt / (1.2 if target else 0.5))
                if abs(self.sleep_k - target) < 0.004:
                    self.sleep_k = target
        sk = self.sleep_k
        self.ct += dt * (1.0 - 0.7 * sk)
        self.wt += dt * (1.0 - 0.6 * sk)

        # ---- beat: onset detector -> (colour only) bloom / brain flare / dot flash / sucker ripple
        b = self.an.bass
        self._bass_s += (b - self._bass_s) * min(1, dt / (0.08 if b > self._bass_s else 0.35))
        if self.an.onset and sk < 0.8:
            strength = 0.75 + 0.25 * self.an.onset_strength
            self.beat = max(self.beat, strength)        # instant attack
            self.ripples.append([now, strength])
            self._last_ripple = now
        else:
            self.beat *= math.exp(-dt / 0.085)          # ~200 ms visible decay
        self.ripples = [r for r in self.ripples if now - r[0] < RIPPLE_LIFE]
        self.level_s += (min(1.0, self.an.level * 1.4) - self.level_s) * min(1, dt * 8)
        # bloom = beat flash + continuous bass glow between beats
        self.glow = min(1.0, 0.95 * self.beat + 0.25 * self._bass_s * (1 - sk))
        self.throb = 1.0            # round 4: beats never move or scale anything

        # ---- colour: recolour only while the colour actually changes
        if self._rc_part == "tent":
            self.recolor(part="tent")
            self._rc_part = None
        elif now - self._rc_t >= 1.0 / config.RECOLOR_HZ - 0.004:
            T, _ = self.target_colour(now)
            if (np.abs(T - self._applied_body).max() > 0.6 / 255
                    or np.abs(T - self._applied_tent).max() > 0.6 / 255):
                self.recolor(part="body")
                self._rc_part = "tent"

        # ---- soft body
        self.mesh.step(dt)

        # ---- particles
        if self.sleeping and sk > 0.5 and now > self.next_bubbles:
            self._spawn_bubbles(random.randint(2, 4), sleepy=True)
            self.next_bubbles = now + random.uniform(2.5, 5.0)
        for p in self.particles:
            p.life -= dt
            if p.kind == "bubble":
                p.vy -= 6 * dt
                p.x += (p.vx + 6 * math.sin(p.wob + t * 2.2)) * dt
                p.y += p.vy * dt
            else:
                p.vx *= (1 - 2.5 * dt)
                p.vy *= (1 - 2.5 * dt)
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.r += 14 * dt
        self.particles = [p for p in self.particles if p.life > 0 and p.y > -20]

        # ---- eyes: mouse tracking, blink, pupil
        try:
            gp = self.mapFromGlobal(QCursor.pos())
            mx, my = gp.x(), gp.y()
        except Exception:
            mx, my = self.width() / 2, self.height()
        cx = (self.L["eyes"][0][0] + self.L["eyes"][1][0]) / 2
        cy = self.L["eyes"][0][1]
        vx, vy = mx - cx, my - cy
        dd = math.hypot(vx, vy) + 1e-6
        mag = min(1.0, dd / 260.0) * self.L["eye_r"] * 0.16
        tx, ty = vx / dd * mag, vy / dd * mag
        k = min(1, dt * 5)
        self.pupil_off[0] += (tx - self.pupil_off[0]) * k
        self.pupil_off[1] += (ty - self.pupil_off[1]) * k
        if now > self.next_blink:
            self.blink_t0 = now
            self.next_blink = now + random.uniform(5, 11)

        # ---- brain drift (very slow, incommensurate; subpixel)
        ox = 0.75 * math.sin(2 * math.pi * t / 11.3) + 0.35 * math.sin(2 * math.pi * t / 7.9 + 1.3)
        oy = 0.6 * math.sin(2 * math.pi * t / 9.7 + 0.4) + 0.25 * math.sin(2 * math.pi * t / 13.1 + 2.1)
        self.brain_pos = (ox + self.mesh.brain[0], oy + self.mesh.brain[1])
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4.6)
        self.brain_glow = (0.87 + 0.26 * pulse + 0.3 * self._bass_s + 1.1 * self.beat) * (1 - 0.5 * sk)
        self.brain_rot = 0.35 * math.sin(2 * math.pi * t / 17.3)

        # ---- tentacles
        hz = config.TENT_HZ * (1.5 if self.ripples else 1.0)       # 45 Hz while a sucker ripple travels
        if (now - getattr(self, "_tent_t", -1e9) >= 1.0 / hz - 0.004
                or getattr(self, "_tent_dirty", True)):
            self._tent_t = now
            self.rig.step(self.wt, amp_k=1.0 - 0.35 * sk, bass=0.0)
            self._render_tentacles(now)
        if self.frame_log is not None:
            i = len(self.rig.disp) - 1
            self.frame_log.append((now, dt, self.brain_pos[0], self.brain_pos[1], self.brain_glow, self.rig.disp[i]))

        self._render_dome(now)
        self.bar.tick(now)
        self.update()

    # ================================================================ tentacle layer (cached)
    def _sucker_opacity(self, now):
        rip = [(now - t0, amp) for t0, amp in self.ripples]
        base = 0.3 * self._bass_s if self._bass_s > 0.15 else 0.0

        def op(smid, smax):
            o = base
            for age, amp in rip:
                x = (smid - age * RIPPLE_SPEED) / RIPPLE_W
                if -3 < x < 3:
                    o += amp * math.exp(-x * x) * max(0.0, 1 - age / RIPPLE_LIFE)
            return o
        return op

    def _render_tentacles(self, now):
        img = self.tent_layer
        r = self.tent_rect
        p = QPainter(img)
        p.setCompositionMode(CM.CompositionMode_Source)
        p.fillRect(img.rect(), Qt.GlobalColor.transparent)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        off = (r.x(), r.y())
        self.rig.draw(p, self.tent_img, off=off)
        if self.ripples or self._bass_s > 0.15:
            self.rig.draw(p, self.suck_pm, opacity_fn=self._sucker_opacity(now), off=off)
        p.end()
        self._tent_dirty = False

    # ================================================================ dome offscreen
    def _render_dome(self, now):
        A = self.art
        dr = A.dome_rect
        bl = self.L["brain"]
        bx = bl["cx"] - dr.x() + self.brain_pos[0]
        by = bl["cy"] - dr.y() + self.brain_pos[1]
        w, h = self.brain_pm.width(), self.brain_pm.height()
        box = QRect(int(math.floor(bx - w / 2)) - 3, int(math.floor(by - h / 2)) - 3, w + 7, h + 7)
        prev = getattr(self, "_brain_box", box)
        self._brain_box = box
        bbox = box.united(prev)

        img = self.brain_img
        p = QPainter(img)
        p.setClipRect(bbox)
        p.setCompositionMode(CM.CompositionMode_Source)
        p.fillRect(bbox, Qt.GlobalColor.transparent)
        p.setCompositionMode(CM.CompositionMode_SourceOver)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        tr = QTransform()
        tr.translate(bx, by)
        tr.rotate(self.brain_rot)
        p.setTransform(tr)
        # the rotation (never exactly 0) keeps the raster engine on its bilinear, sub-pixel path
        target = QRectF(-w / 2, -h / 2, w, h)
        p.drawImage(target, self.brain_pm)
        p.setCompositionMode(CM.CompositionMode_Plus)
        p.setOpacity(max(0.0, min(1.0, self.brain_glow)))
        p.drawImage(target, self.brain_em_pm)
        p.resetTransform()
        p.setOpacity(1.0)
        p.setCompositionMode(CM.CompositionMode_DestinationIn)
        p.drawImage(bbox, A.brain_clip, bbox)
        p.end()

        d = self.dome_img
        p = QPainter(d)
        p.setCompositionMode(CM.CompositionMode_Source)
        p.fillRect(d.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(CM.CompositionMode_SourceOver)
        ox, oy = A.dome_plane_offset
        p.drawImage(ox, oy, self.dome_pm)
        if self.glow > 0.01:
            p.setCompositionMode(CM.CompositionMode_Plus)
            p.setOpacity(min(1.0, self.glow))
            p.drawImage(ox, oy, self.art.dome._tmp)
            p.setOpacity(1.0)
            p.setCompositionMode(CM.CompositionMode_SourceOver)
        p.drawImage(bbox, img, bbox)
        self._draw_wave(p, dr)
        p.drawImage(0, 0, self.glass_tint_pm)
        p.drawImage(0, 0, self.glass_dark_pm)
        p.setCompositionMode(CM.CompositionMode_Plus)
        p.drawImage(0, 0, self.glass_hl_pm)
        # spectrum dots (always above the brain), each its own shimmering hue
        bands = self.an.bands
        t = now - self.t0
        boot_u = (now - self.boot_t0) / 2.2 if (self.boot_t0 is not None and self._boot) else 9
        for i, ((dot, _), pm) in enumerate(zip(A.dots, self.dot_pm)):
            bi = dot["band"]
            lv = float(bands[bi]) if bi < len(bands) else 0.0
            idle = 0.12 * math.sin(t * 0.9 + bi * 0.7) + 0.08 * math.sin(t * 5.3 + i * 2.1)
            o = 0.38 + idle + 0.62 * lv + 0.75 * self.beat
            if boot_u < 1.4:
                o *= ease((boot_u * 1.6 - bi / len(bands) * 0.8) * 3)
            o *= (1 - 0.6 * self.sleep_k)
            p.setOpacity(max(0.0, min(1.0, o)))
            p.drawImage(dot["x"] - dr.x(), dot["y"] - dr.y(), pm)
        p.end()

    def _draw_wave(self, p, dr):
        """Oscilloscope line through the brain: 128 points of the live waveform, smoothed,
        tapered at both ends, body-tinted, ~35 % opacity; flat and faint when silent."""
        an = self.an
        n = len(an.wave)
        if not hasattr(self, "_wave_s"):
            self._wave_s = np.zeros(n, np.float32)
            self._wave_x = np.linspace(-1, 1, n).astype(np.float32)
            self._wave_taper = (np.cos(self._wave_x * math.pi) * 0.5 + 0.5).astype(np.float32)
        live = 0.0 if an.silent else 1.0
        tgt = np.clip(an.wave, -1.5, 1.5) * live
        self._wave_s += (tgt - self._wave_s) * 0.45
        bl = self.L["brain"]
        cx = bl["cx"] - dr.x()
        cy = bl["cy"] - dr.y() + 8
        half = 62.0
        amp = 16.0 * self._wave_s * self._wave_taper
        xs = cx + self._wave_x * half
        ys = cy + amp
        flat = float(np.abs(self._wave_s).max()) < 0.01
        poly = None if flat else QPolygonF(list(map(QPointF, xs.tolist(), ys.tolist())))
        c = self.tint
        a = (0.16 + 0.22 * min(1.0, self.level_s * 1.5 + 0.3 * live)) * (1 - 0.6 * self.sleep_k)
        p.save()
        p.setCompositionMode(CM.CompositionMode_Plus)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qcol(np.clip(c * 0.8 + 0.2, 0, 1), a), 1.8))
        if flat:
            p.drawLine(QPointF(float(xs[0]), cy), QPointF(float(xs[-1]), cy))
        else:
            p.drawPolyline(poly)
        p.restore()

    def _draw_dome(self, p: QPainter):
        dr = self.art.dome_rect
        p.setCompositionMode(CM.CompositionMode_Plus)
        if not self.mesh.active:
            p.drawImage(dr.x(), dr.y(), self.dome_img)
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        for src, dst in self.mesh.quads():
            sq = QPolygonF([QPointF(x, y) for x, y in src])
            dq = QPolygonF([QPointF(x + dr.x(), y + dr.y()) for x, y in dst])
            T = QTransform()
            if not QTransform.quadToQuad(sq, dq, T):
                continue
            path = QPainterPath()
            c = dq.boundingRect().center()
            grown = QPolygonF([c + (pt - c) * 1.0 + QPointF(math.copysign(0.6, (pt - c).x()), math.copysign(0.6, (pt - c).y())) for pt in dq])
            path.addPolygon(grown)
            p.setClipPath(path)
            p.setTransform(T)
            sr = sq.boundingRect()
            p.drawImage(sr, self.dome_img, sr)
            p.resetTransform()
        p.restore()

    # ================================================================ paint
    def paintEvent(self, e):
        t_start = time.perf_counter()
        p = QPainter(self)
        p.setClipRegion(e.region())
        p.setCompositionMode(CM.CompositionMode_Source)
        p.fillRect(e.rect(), Qt.GlobalColor.transparent)
        A = self.art
        now = self._now()
        g = min(1.0, self.glow)
        p.setCompositionMode(CM.CompositionMode_Plus)
        tr = self.tent_rect.topLeft()
        p.drawImage(tr, self.tent_layer)
        if g > 0.01:
            p.setOpacity(g * 0.85)
            p.drawImage(tr, self.tent_layer)
            p.setOpacity(1.0)
        p.drawImage(A.body.rect.topLeft(), self.body_img)
        if g > 0.01:                          # bloom: the glow plane (T*M) added again
            p.setOpacity(g)
            p.drawImage(A.body.rect.topLeft(), A.body._tmp)
            p.setOpacity(1.0)
        for i, (r, pm) in enumerate(zip(A.eye_rects, self.eye_pm)):
            p.setCompositionMode(CM.CompositionMode_Plus)
            p.drawImage(r.topLeft(), pm)
            self._draw_pupil(p, i, now)
        self._draw_dome(p)
        fk = self._flash_k(now)
        if fk > 0:
            p.setCompositionMode(CM.CompositionMode_Plus)
            p.setOpacity(0.55 * fk)
            p.drawPixmap(0, 0, self.flash_pm)
            p.setOpacity(1)
        p.resetTransform()
        p.setCompositionMode(CM.CompositionMode_SourceOver)
        if self.particles:
            self._draw_particles(p)
        self._draw_bubble(p, now)
        p.end()
        self.paint_ms.append((time.perf_counter() - t_start) * 1000)
        if len(self.paint_ms) > 600:
            del self.paint_ms[:300]

    def bubble_centre(self, now=None):
        """The control bubble floating by his ear (window coords), gently bobbing."""
        now = self._now() if now is None else now
        t = now - self.t0
        ex, ey = self.L["ear"]
        x = ex + 16 + 1.2 * math.sin(t * 0.9 + 1.0)
        y = ey - 18 + 2.4 * math.sin(t * 1.55)
        if getattr(self, "bar", None) is not None and self.bar.side < 0:
            x = self.width() - x
        return x, y, 7.5 + 0.35 * math.sin(t * 2.3)

    def _draw_bubble(self, p, now):
        if self.bar.k > 0.0 or self.bar.isVisible():
            return
        x, y, r = self.bubble_centre(now)
        c = self.tint
        a = 1.0 - 0.5 * self.sleep_k
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        halo = QRadialGradient(QPointF(x, y), r * 2.2)
        halo.setColorAt(0.4, qcol(c, 0.22 * a))
        halo.setColorAt(1.0, qcol(c, 0.0))
        p.setBrush(halo)
        p.drawEllipse(QPointF(x, y), r * 2.2, r * 2.2)
        g = QRadialGradient(QPointF(x, y), r)
        g.setColorAt(0.0, qcol(c, 0.10 * a))
        g.setColorAt(0.75, qcol(c, 0.28 * a))
        g.setColorAt(1.0, qcol(np.clip(c * 0.6 + 0.4, 0, 1), 0.85 * a))
        p.setBrush(g)
        p.drawEllipse(QPointF(x, y), r, r)
        p.setBrush(QColor(255, 255, 255, int(170 * a)))
        p.drawEllipse(QPointF(x - r * 0.38, y - r * 0.4), r * 0.22, r * 0.16)

    def _draw_pupil(self, p, i, now):
        ex, ey = self.L["eyes"][i]
        er = self.L["eye_r"]
        p.setCompositionMode(CM.CompositionMode_SourceOver)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x = ex + self.pupil_off[0]
        y = ey + self.pupil_off[1]
        fk = self._flash_k(now)
        lvl = max(self.level_s, fk)
        pw = er * (0.22 - 0.13 * lvl) * (1 - 0.3 * self.sleep_k)
        ph = er * 0.62
        g = QRadialGradient(QPointF(x, y), ph)
        g.setColorAt(0.0, QColor(0, 0, 0, 235))
        g.setColorAt(0.75, QColor(0, 0, 0, 200))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.save()
        p.translate(x, y)
        p.scale(pw / ph, 1.0)
        p.translate(-x, -y)
        p.setBrush(g)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(x, y), ph, ph)
        p.restore()
        u = (now - self.blink_t0) / 0.22
        lid = math.sin(math.pi * u) if 0 <= u <= 1 else 0.0
        lid = max(lid, 0.55 * self.sleep_k)
        if lid > 0.01:
            p.save()
            path = QPainterPath()
            path.addEllipse(QPointF(ex, ey), er * 0.95, er * 0.8)
            p.setClipPath(path)
            c = QColor(int(20 + 30 * self.tint[0]), int(8 + 20 * self.tint[1]), int(10 + 20 * self.tint[2]), 245)
            p.fillRect(QRectF(ex - er, ey - er, 2 * er, 2 * er * lid * 0.95), c)
            p.restore()

    def _draw_particles(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tint = self.tint
        for q in self.particles:
            a = max(0.0, min(1.0, q.life / q.max)) * (min(1.0, (q.max - q.life) * 6))
            if q.kind == "bubble":
                c = QColor(int(120 + 135 * tint[0]), int(120 + 135 * tint[1]), int(120 + 135 * tint[2]), int(150 * a))
                p.setPen(QPen(c, 1.0))
                p.setBrush(QColor(c.red(), c.green(), c.blue(), int(25 * a)))
                p.drawEllipse(QPointF(q.x, q.y), q.r, q.r)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, int(170 * a)))
                p.drawEllipse(QPointF(q.x - q.r * 0.35, q.y - q.r * 0.35), q.r * 0.22, q.r * 0.22)
            elif q.kind == "ink":
                g = QRadialGradient(QPointF(q.x, q.y), q.r)
                g.setColorAt(0, QColor(6, 2, 10, int(150 * a)))
                g.setColorAt(1, QColor(6, 2, 10, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                p.drawEllipse(QPointF(q.x, q.y), q.r, q.r)
            else:
                c = QColor(int(255 * min(1, tint[0] + 0.3)), int(255 * min(1, tint[1] + 0.3)), int(255 * min(1, tint[2] + 0.3)), int(220 * a))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawEllipse(QPointF(q.x, q.y), 1.3, 1.3)

    # ================================================================ effects
    def _spawn_bubbles(self, n, sleepy=False, at=None):
        dcx, dcy = self.L["dome"]["cx"], self.L["dome"]["cy"]
        for _ in range(n):
            if at is None:
                x = dcx + random.uniform(-0.6, 0.6) * self.L["dome"]["rx"]
                y = dcy - self.L["dome"]["ry"] * random.uniform(0.6, 0.95)
            else:
                x, y = at[0] + random.uniform(-8, 8), at[1] + random.uniform(-6, 6)
            r = random.uniform(1.6, 4.2)
            self.particles.append(Particle(x, y, random.uniform(-4, 4), -random.uniform(10, 22) if sleepy else -random.uniform(25, 60),
                                           r, random.uniform(2.5, 4.5) if sleepy else random.uniform(1.2, 2.2), "bubble"))

    def squish(self, x, y):
        now = self._now()
        dr = self.art.dome_rect
        self.mesh.poke(x - dr.x(), y - dr.y(), 1.0)
        self.sounds.play("squelch")
        self._spawn_bubbles(5, at=(x, y - 10))
        for _ in range(6):
            a = random.uniform(0, 6.28)
            self.particles.append(Particle(x, y, math.cos(a) * 30, math.sin(a) * 20 + 10, random.uniform(5, 9), 1.1, "ink"))
        for _ in range(10):
            a = random.uniform(0, 6.28)
            sp = random.uniform(30, 80)
            self.particles.append(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp, 1, random.uniform(0.4, 0.9), "spark"))
        self.click_times = [c for c in self.click_times if now - c < 1.0] + [now]
        if len(self.click_times) >= 3 and now - self.flash_t0 > 1.2:
            self.flash_t0 = now
            self.sounds.play("startle")
            self.click_times = []
        self.wake()

    def wake(self):
        self.last_active = self._now()

    # ================================================================ input
    def _in_dome(self, x, y):
        cx, cy, rx, ry = self.dome_ellipse
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 0.95 and y < self.L["dome"]["seam_y"]

    def _hit(self, x, y):
        if 0 <= x < self.width() and 0 <= y < self.height():
            return self.hit[int(y), int(x)] > 40
        return False

    def _near_head(self, x, y):
        bx, by, br = self.bubble_centre()
        if (x - bx) ** 2 + (y - by) ** 2 < (br + 12) ** 2:
            return True
        return self._hit(x, y) and y < self.huby + 45

    def mousePressEvent(self, e):
        x, y = e.position().x(), e.position().y()
        self.wake()
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._press = (e.globalPosition().toPoint(), self.pos(), self._in_dome(x, y), (x, y))
        if self._in_dome(x, y):
            self.squish(x, y)

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            g0, p0, _, _ = self._press
            d = e.globalPosition().toPoint() - g0
            if self._drag or d.manhattanLength() > 4:
                self._drag = True
                self.move(p0 + d)
            return
        if self._near_head(x, y):
            self.bar.reveal(self._now())
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._in_dome(x, y) else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._save_settings()
        self._drag = None
        self._press = None

    def mouseDoubleClickEvent(self, e):
        x, y = e.position().x(), e.position().y()
        if self._in_dome(x, y):
            self.squish(x, y)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Space:
            self.button_action("play")
        elif k == Qt.Key.Key_Right:
            self.button_action("next")
        elif k == Qt.Key.Key_Left:
            self.button_action("prev")
        elif k == Qt.Key.Key_Escape:
            self.close()

    def contextMenuEvent(self, e):
        self.show_menu(e.globalPos())

    def show_menu(self, gpos):
        m = QMenu(self)
        m.setStyleSheet("QMenu{background:#07121a;color:#bff;border:1px solid #1a4a55;padding:4px}"
                        "QMenu::item:selected{background:#123a44}")
        m.addAction("Open files...", self.open_files)
        m.addAction("Open folder...", self.open_folder)
        m.addSeparator()
        a = m.addAction("Playlist")
        a.setCheckable(True)
        a.setChecked(self.bar.panel_open)
        a.triggered.connect(self.bar.set_panel)
        a = m.addAction("Controls always out")
        a.setCheckable(True)
        a.setChecked(self.controls_pinned)
        a.triggered.connect(self.set_controls_pinned)
        a = m.addAction("UI sounds")
        a.setCheckable(True)
        a.setChecked(not self.sounds.muted)
        a.triggered.connect(self.set_ui_sounds)
        a = m.addAction("Always on top")
        a.setCheckable(True)
        a.setChecked(self.on_top)
        a.triggered.connect(self.set_on_top)
        m.addSeparator()
        m.addAction("Quit", self.close)
        m.exec(gpos)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = self._expand([u.toLocalFile() for u in e.mimeData().urls()])
        if paths:
            self.local.set_files(paths)

    # ================================================================ actions
    def set_controls_pinned(self, on):
        self.controls_pinned = bool(on)
        if on:
            self.bar.reveal(self._now())
        self._save_settings()

    def set_ui_sounds(self, on):
        self.sounds.muted = not bool(on)
        self._save_settings()

    def set_on_top(self, on, save=True):
        self.on_top = bool(on)
        pos = self.pos()
        vis = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.on_top)
        if vis:
            self.move(pos)
            self.show()
        from .controlbar import set_topmost
        set_topmost(self, self.on_top)
        # every other top-level window follows (control panel / bubble / playlist live in the bar)
        if getattr(self, "bar", None) is not None:
            self.bar.set_on_top(self.on_top)
        if save:
            self._save_settings()

    def seek_fraction(self, u):
        np_ = self.media.now()
        if not np_.duration_ms:
            return
        ms = int(u * np_.duration_ms)
        if self.media.route() == "local":
            self.local.player.setPosition(ms)
        else:
            try:
                self.provider.command("seek", ms)
            except Exception as ex:
                print(f"[octo] seek: {ex}")
        self.media._poke()

    @staticmethod
    def _expand(paths):
        out = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, fs in os.walk(p):
                    out += [os.path.join(root, f) for f in sorted(fs) if f.lower().endswith(AUDIO_EXTS)]
            elif p.lower().endswith(AUDIO_EXTS):
                out.append(p)
        return out

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Octoplayer - open audio files", os.path.expanduser("~/Music"),
                                                "Audio (*.mp3 *.flac *.wav *.m4a *.ogg *.opus *.aac *.wma);;All files (*)")
        if files:
            self.local.set_files(files)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Octoplayer - open folder", os.path.expanduser("~/Music"))
        if d:
            fs = self._expand([d])
            if fs:
                self.local.set_files(fs)

    def button_action(self, name: str):
        route = self.media.route()
        if route == "none" and name == "play":
            self.open_files()
        else:
            self.media.control("toggle" if name == "play" else name)
        self.wake()
        self.update()

    def _on_media_changed(self, *_):
        self.bar.media_changed()
        self._retarget()

    # ---- volume / balance routing (same rules as Vexa: local -> own; SMTC app -> that app; else system)
    def _vol_route(self):
        r = self.media.route()
        if r == "local":
            return ("local",)
        if r == "smtc" and self.media.smtc is not None:
            return ("smtc", self.media.smtc.app_id or self.media.smtc.app or "")
        if self.hub.mode in ("auto", "loopback") and getattr(self.hub.loopback, "available", False):
            return ("system",)
        return ("local",)

    def _retarget(self, now=None, refresh=False):
        now = self._now() if now is None else now
        key = self._vol_route()
        if key == self.router.route and not refresh and not self.router.needs_refresh(now):
            return False
        self.router.resolve(key, now)
        tg = self.router.target
        self.knobs["volume"].tag = tg.tag
        if self.active_knob is None:
            v = tg.volume()
            if v is not None:
                self.knobs["volume"].value = max(0.0, min(1.0, float(v)))
            b = tg.balance()
            if b is not None:
                self.knobs["balance"].value = max(-1.0, min(1.0, float(b)))
        return True

    def _dial_target(self):
        now = self._now()
        r = self.router
        if r.route[0] != "local" and r.target.fallback and now - r.t_resolved > 1.0:
            self._retarget(now, refresh=True)
        else:
            self._retarget(now)
        return r.target

    def _on_volume(self, v):
        tg = self.router.target
        if not tg.set_volume(v) and tg.kind != "own":
            tg = self.router.resolve(self._vol_route(), self._now())
            self.knobs["volume"].tag = tg.tag
            tg.set_volume(v)
        if tg.kind == "own" and self.settings is not None:
            self.settings.setValue("volume", v)

    def _on_balance(self, v):
        tg = self.router.target
        if tg.kind == "own":
            tg.set_balance(v)
            if self.settings is not None:
                self.settings.setValue("balance", v)
        elif not tg.set_balance(v):
            tg = self.router.resolve(self._vol_route(), self._now())
            tg.set_balance(v)

    def _on_local_state(self, *_):
        if self.local.is_playing():
            self.wake()
            self.own_target.apply()
            for ms in (300, 1000, 2500):
                QTimer.singleShot(ms, lambda: self.own_target.applied or self.own_target.apply())
        self._retarget()
        self.bar.media_changed()

    # ================================================================ close
    def closeEvent(self, e):
        self._save_settings()
        self.timer.stop()
        if self.frame_log is not None:
            try:
                with open(config.FRAMELOG, "w") as f:
                    f.write("t,dt,brain_x,brain_y,glow,tip_disp\n")
                    for row in self.frame_log:
                        f.write(",".join(f"{v:.6f}" for v in row) + "\n")
            except Exception as ex:
                print(f"[octo] framelog: {ex}")
        self.bar.close()
        self.hub.shutdown()
        self.media.shutdown()
        super().closeEvent(e)
