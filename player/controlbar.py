"""The control bar: a bioluminescent membrane that slides out of the octopus' right ear fin.

Layout (bar coords, side = +1 grows to the right; mirrored stalk when it must grow left):
    [stalk]( art )  Title ..................................  [=]
                    artist                              1:23 / 4:56
                    ------------o---------------------------  seek
                    |<<   (>)   >>|     vol --o--     bal --o--
Hover near the head reveals it; it retracts CONTROLS_IDLE_S after the pointer leaves
(or stays out when 'Controls always out' is on). Drawing: ~0.3 ms, 60 fps only while
sliding, 30 fps otherwise, nothing while hidden.
"""
from __future__ import annotations

import math
import os
import time

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QGuiApplication, QImage, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient)
from PySide6.QtWidgets import QWidget

from . import config, tags

W, H0 = 360, 100
PANEL_H = 214
ROW_H = 18
STALK = 26          # px of stalk before the membrane body


def rpen(c, w):
    pen = QPen(c, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    return pen


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def fmt_ms(ms):
    s = int(max(0, ms) // 1000)
    if s >= 3600:
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def set_topmost(widget, on):
    """Win32: put a top-level window in / out of the TOPMOST band (keeps every other flag)."""
    if not config.IS_WINDOWS:
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        ctypes.windll.user32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(-1 if on else -2),
                                          0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)   # NOSIZE|NOMOVE|NOACTIVATE
    except Exception as ex:  # pragma: no cover
        print(f"[octo] topmost: {ex}")


class ControlBar(QWidget):
    def __init__(self, octo):
        super().__init__(octo)
        self.octo = octo
        self.setWindowTitle("Octoplayer controls")
        # Tool window OWNED by the octopus (parent) -> always stacked above him
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.on_top = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setFixedSize(W, H0)
        self.k = 0.0                # slide-out progress
        self.target = 0.0
        self.last_hover = -1e9
        self.side = 1
        self.panel_open = False
        self.scroll = 0
        self.hover = None
        self.press = None
        self._drag_knob = None
        self._last_paint = -1e9
        self._art_key = None
        self._art = None
        self.colour = QColor(80, 255, 220)
        self.colour2 = QColor(160, 120, 255)
        self.t0 = time.monotonic()

    # ------------------------------------------------------------ placement
    def anchor(self):
        o = self.octo
        ex, ey = o.L["ear"]
        x = o.x() + int(ex) - 12
        y = o.y() + int(ey) - H0 // 2 - 6
        scr = QGuiApplication.screenAt(o.pos() + QPoint(o.width() // 2, o.height() // 3))
        side = 1
        if scr is not None and x + W > scr.availableGeometry().right():
            side = -1
            x = o.x() + int(o.width() - ex) + 12 - W
        return QPoint(x, y), side

    def set_on_top(self, on):
        """Mirror the octopus' 'Always on top' (Qt flag + Win32 HWND_TOPMOST), keeping the
        frameless/tool/translucent setup."""
        self.on_top = bool(on)
        vis = self.isVisible()
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if self.on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if vis:
            self.show()
            self.apply_on_top()

    def apply_on_top(self):
        set_topmost(self, self.on_top)
        if self.on_top:
            self.raise_()

    def follow(self):
        pos, side = self.anchor()
        if side != self.side:
            self.side = side
        if self.isVisible():
            self.move(pos)

    # ------------------------------------------------------------ state
    def reveal(self, now):
        self.last_hover = now
        if self.target < 1:
            self.target = 1.0
            if not self.isVisible():
                self.follow()
                pos, _ = self.anchor()
                self.move(pos)
                self.show()
                self.apply_on_top()
                if self.octo.on_top:
                    self.raise_()
            self.octo.sounds.play("bubble")

    def set_panel(self, on):
        self.panel_open = bool(on)
        self.setFixedSize(W, H0 + (PANEL_H if self.panel_open else 0))
        if on:
            self.reveal(self.octo._now())
        self.update()

    def media_changed(self):
        self._art_key = None
        self.update()

    def busy(self):
        return 0.0 < self.k < 1.0

    def tick(self, now):
        o = self.octo
        if o.controls_pinned:
            self.target = 1.0
            self.last_hover = now
        elif (now - self.last_hover > config.CONTROLS_IDLE_S and self._drag_knob is None
              and not self.underMouse()):
            self.target = 0.0
        if self.target > 0 and not self.isVisible():
            self.follow()
            self.move(self.anchor()[0])
            self.show()
            self.apply_on_top()
        dt = 1 / 60
        if self.k != self.target:
            if self.target > self.k:
                self.k = min(1.0, self.k + dt / 0.5)
            else:
                self.k = max(0.0, self.k - dt / 0.45)
            if self.k <= 0.0 and self.target == 0.0:
                if self.panel_open:
                    self.set_panel(False)
                self.hide()
                return
            self.update()
            self._last_paint = now
        elif self.isVisible() and now - self._last_paint > (1 / 60 if self.k >= 1.0 and now - self.last_hover < 1.2 else 1 / 30):
            self.update()
            self._last_paint = now

    # ------------------------------------------------------------ geometry
    def _x(self, x):
        return x if self.side > 0 else W - x

    def _rects(self):
        """hit/draw geometry in bar coords (pre-mirroring of the stalk only)."""
        L = STALK + 6
        return {
            "art": (L + 26, H0 / 2 - 2, 24),
            "title": QRectF(L + 58, 11, W - L - 58 - 34, 17),
            "artist": QRectF(L + 58, 27, W - L - 58 - 34, 15),
            "list": QRectF(W - 34, 10, 24, 20),
            "seek": QRectF(L + 58, 44, W - L - 58 - 16, 12),
            "prev": (L + 74, 72, 10),
            "play": (L + 104, 72, 13),
            "next": (L + 134, 72, 10),
            "volume": QRectF(L + 160, 66, 58, 14),
            "balance": QRectF(L + 230, 66, 58, 14),
        }

    def _hit(self, x, y):
        R = self._rects()
        for n in ("prev", "play", "next"):
            cx, cy, r = R[n]
            if (x - cx) ** 2 + (y - cy) ** 2 < (r + 5) ** 2:
                return n
        for n in ("volume", "balance", "seek", "list"):
            if R[n].adjusted(-4, -5, 4, 5).contains(QPointF(x, y)):
                return n
        cx, cy, r = R["art"]
        if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
            return "art"
        if self.panel_open and y > H0:
            return "panel"
        return None

    # ------------------------------------------------------------ paint
    def _membrane(self, h, t):
        """organic blob: a stalk out of the ear that swells into a rounded body."""
        path = QPainterPath()
        top, bot = 8.0, h - 8.0
        mid = H0 / 2
        wob = 1.6 * math.sin(t * 1.3)
        x0, x1 = STALK, W - 4.0
        path.moveTo(self._x(0), mid - 7)
        path.cubicTo(self._x(STALK * 0.6), mid - 8, self._x(STALK * 0.5), top + wob, self._x(x0 + 18), top + wob * 0.5)
        path.cubicTo(self._x(x0 + 120), top - 3 + wob, self._x(x1 - 90), top + 2 - wob, self._x(x1 - 22), top)
        path.cubicTo(self._x(x1 + 2), top, self._x(x1 + 2), top + 18, self._x(x1), mid if h <= H0 + 1 else top + 30)
        if h > H0 + 1:
            path.lineTo(self._x(x1), bot - 22)
        path.cubicTo(self._x(x1 + 2), bot - 12, self._x(x1 - 6), bot, self._x(x1 - 26), bot)
        path.cubicTo(self._x(x1 - 120), bot + 3 - wob, self._x(x0 + 100), bot - 2 + wob, self._x(x0 + 18), bot - wob * 0.5)
        path.cubicTo(self._x(STALK * 0.5), bot, self._x(STALK * 0.6), mid + 8, self._x(0), mid + 7)
        path.closeSubpath()
        return path

    def _bubble_local(self):
        """the octopus' ear bubble in bar coordinates (morph start / end)."""
        o = self.octo
        bx, by, br = o.bubble_centre()
        return bx + o.x() - self.x(), by + o.y() - self.y(), br

    def _morph_path(self, h, t):
        """Bubble -> panel morph: a superellipse whose extent, squareness and wobble follow the
        progress k (springy overshoot, width leads height). Returns (path, content alpha)."""
        k = self.k
        opening = self.target > 0.5

        def out_cubic(u):
            u = max(0.0, min(1.0, u))
            return 1 - (1 - u) ** 3

        def out_back(u, c1=1.6):                 # springy overshoot (vertical only)
            u = max(0.0, min(1.0, u))
            return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2
        if opening:
            kx = out_cubic(k * 1.3)                  # stretches sideways first...
            ky = out_back(k * 1.35 - 0.35)           # ...then swells with a wobbly overshoot
        else:
            kx = ease(k * 1.3 - 0.3)                 # shrinks in height first, then pulls back in
            ky = ease(k * 1.4)
        bx, by, br = self._bubble_local()
        px0, py0, px1, py1 = STALK - 8.0, 8.0, W - 4.0, h - 8.0
        x0 = bx - br + (px0 - (bx - br)) * kx
        x1 = bx + br + (px1 - (bx + br)) * kx
        y0 = by - br + (py0 - (by - br)) * ky
        y1 = by + br + (py1 - (by + br)) * ky
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
        sq = min(max(kx, 0), max(ky, 0))
        n = 2.0 + 7.0 * sq
        wob = 3.2 * math.sin(math.pi * min(1.0, k)) + 0.7 * sq
        path = QPainterPath()
        N = 96
        for i in range(N + 1):
            th = 2 * math.pi * i / N
            ct, st = math.cos(th), math.sin(th)
            ex = a * math.copysign(abs(ct) ** (2 / n), ct)
            ey = b * math.copysign(abs(st) ** (2 / n), st)
            d = wob * math.sin(5 * th + t * 6.0) * (0.6 + 0.4 * math.sin(3 * th - t * 4.0))
            r = math.hypot(ex, ey) + 1e-6
            X = cx + ex + ex / r * d
            Y = cy + ey + ey / r * d
            if i == 0:
                path.moveTo(X, Y)
            else:
                path.lineTo(X, Y)
        path.closeSubpath()
        return path, ease((k - 0.62) / 0.38)

    def _c(self, c, a):
        q = QColor(c)
        q.setAlphaF(max(0.0, min(1.0, a)))
        return q

    def paintEvent(self, e):
        p = QPainter(self)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.k <= 0:
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        o = self.octo
        now = o._now()
        t = time.monotonic() - self.t0
        c, c2 = self.colour, self.colour2
        lvl = o.level_s
        h = self.height()
        path, content_a = self._morph_path(h, t)
        # frosted, glassy bioluminescent membrane tinted with the body colour (~65 % opacity)
        for wdt, a in ((10, 0.07), (5, 0.14)):
            p.setPen(QPen(self._c(c, a * (0.8 + 0.5 * lvl + 0.6 * o.glow)), wdt))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        br = path.boundingRect()
        g = QLinearGradient(0, br.top(), 0, br.bottom())
        g.setColorAt(0.0, QColor(int(c.red() * 0.30 + 10), int(c.green() * 0.30 + 12), int(c.blue() * 0.30 + 18), 172))
        g.setColorAt(0.55, QColor(int(c.red() * 0.16 + 6), int(c.green() * 0.16 + 8), int(c.blue() * 0.16 + 14), 160))
        g.setColorAt(1.0, QColor(int(c.red() * 0.22 + 8), int(c.green() * 0.22 + 8), int(c.blue() * 0.22 + 16), 168))
        p.setBrush(g)
        p.setPen(QPen(self._c(c, 0.8), 1.3))
        p.drawPath(path)
        # frost: soft sheen across the top + a specular glint
        sh = QLinearGradient(0, br.top(), 0, br.top() + min(34.0, br.height() * 0.5))
        sh.setColorAt(0, QColor(255, 255, 255, 34))
        sh.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(sh)
        p.drawPath(path)
        if content_a <= 0.01:
            # still a bubble / blob: a small specular glint near its top-left
            gr = min(4.0, max(1.2, br.height() * 0.12))
            p.setBrush(QColor(255, 255, 255, 140))
            p.drawEllipse(QPointF(br.left() + min(br.width() * 0.3, 14), br.top() + min(br.height() * 0.3, 9)), gr * 1.3, gr)
            p.end()
            return
        p.setOpacity(content_a)
        p.setClipPath(path)
        for i in range(9):
            x = self._x(STALK + 40 + i * (W - STALK - 80) / 8)
            yb = h - 12
            a = 0.3 + 0.3 * math.sin(t * 2.2 + i * 0.8) + 0.4 * lvl + 0.5 * o.glow
            rg = QRadialGradient(QPointF(x, yb), 3.0)
            rg.setColorAt(0, self._c(c2 if i % 2 else c, min(1, a)))
            rg.setColorAt(1, self._c(c, 0))
            p.setBrush(rg)
            p.drawEllipse(QPointF(x, yb), 3.0, 3.0)
        R = self._rects()
        self._draw_art(p, R, c)
        self._draw_text(p, R, c, t, now)
        self._draw_seek(p, R, c, c2, now)
        self._draw_buttons(p, R, c, t)
        self._draw_sliders(p, R, c, c2, now)
        if self.panel_open:
            self._draw_panel(p, c)
        p.end()

    def _draw_art(self, p, R, c):
        cx, cy, r = R["art"]
        np_ = self.octo.media.now()
        if np_.art_key != self._art_key:
            self._art_key = np_.art_key
            self._art = tags.thumbnail(np_.art_key, 2 * r, 2 * r) if np_.art_key else None
        path = QPainterPath()
        path.addEllipse(QPointF(cx, cy), r, r)
        p.save()
        p.setClipPath(path, Qt.ClipOperation.IntersectClip)
        if self._art is not None:
            p.drawImage(QRectF(cx - r, cy - r, 2 * r, 2 * r), self._art)
        else:
            g = QRadialGradient(QPointF(cx - 6, cy - 6), r * 1.2)
            g.setColorAt(0, self._c(c, 0.75))
            g.setColorAt(0.5, self._c(c, 0.22))
            g.setColorAt(1, QColor(0, 0, 0, 200))
            p.fillRect(QRectF(cx - r, cy - r, 2 * r, 2 * r), g)
        g = QRadialGradient(QPointF(cx, cy), r)
        g.setColorAt(0.72, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.fillRect(QRectF(cx - r, cy - r, 2 * r, 2 * r), g)
        p.restore()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(self._c(c, 0.9), 1.6))
        p.drawEllipse(QPointF(cx, cy), r + 1, r + 1)
        if self.hover == "art":
            p.setPen(QPen(self._c(c, 0.35), 4))
            p.drawEllipse(QPointF(cx, cy), r + 3, r + 3)

    def _draw_text(self, p, R, c, t, now):
        np_ = self.octo.media.now()
        if np_.source == "none" or not (np_.title or np_.artist):
            title, artist = "Octoplayer", "drop music on me, or play something"
        else:
            title = np_.title or "Unknown"
            artist = np_.artist or np_.app
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        r = R["title"]
        tw = fm.horizontalAdvance(title)
        p.save()
        p.setClipRect(r, Qt.ClipOperation.IntersectClip)
        p.setPen(QColor(235, 250, 255, 235))
        p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(title, Qt.TextElideMode.ElideRight, int(r.width())))
        p.restore()
        f2 = QFont("Segoe UI", 8)
        p.setFont(f2)
        p.setPen(self._c(c, 0.85))
        r2 = R["artist"]
        fm2 = QFontMetrics(f2)
        tm = ""
        if np_.duration_ms:
            tm = f"{fmt_ms(np_.pos_at(now))} / {fmt_ms(np_.duration_ms)}"
        # fixed-width time column (room for 'h:mm:ss / h:mm:ss'), right-aligned to the seek bar
        tmw = fm2.horizontalAdvance("0:00:00 / 0:00:00") + 6
        right = R["seek"].right()
        aw = max(10.0, right - tmw - 6 - r2.x())
        p.drawText(QRectF(r2.x(), r2.y(), aw, r2.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm2.elidedText(artist, Qt.TextElideMode.ElideRight, int(aw)))
        if tm:
            p.setPen(QColor(200, 220, 230, 190))
            p.drawText(QRectF(right - tmw, r2.y(), tmw, r2.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, tm)
        # playlist button
        lr = R["list"]
        p.setPen(rpen(self._c(c, 1.0 if (self.hover == "list" or self.panel_open) else 0.6), 1.6))
        for i in range(3):
            y = lr.y() + 5 + i * 5
            p.drawLine(QPointF(lr.x() + 5, y), QPointF(lr.right() - 5, y))

    def _draw_seek(self, p, R, c, c2, now):
        np_ = self.octo.media.now()
        r = R["seek"]
        y = r.center().y()
        u = 0.0
        if np_.duration_ms:
            u = max(0.0, min(1.0, np_.pos_at(now) / np_.duration_ms))
        if self._drag_knob == "seek":
            u = self._seek_u
        p.setPen(rpen(QColor(255, 255, 255, 40), 2))
        p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
        g = QLinearGradient(r.x(), 0, r.right(), 0)
        g.setColorAt(0, self._c(c2, 0.9))
        g.setColorAt(1, self._c(c, 0.95))
        pen = QPen(g, 2.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        x = r.x() + r.width() * u
        if u > 0:
            p.drawLine(QPointF(r.x(), y), QPointF(x, y))
        p.setPen(Qt.PenStyle.NoPen)
        rg = QRadialGradient(QPointF(x, y), 9)
        rg.setColorAt(0, self._c(c, 0.9))
        rg.setColorAt(0.35, self._c(c, 0.45))
        rg.setColorAt(1, self._c(c, 0))
        p.setBrush(rg)
        p.drawEllipse(QPointF(x, y), 9, 9)
        p.setBrush(QColor(255, 255, 255, 230))
        p.drawEllipse(QPointF(x, y), 2.4 if self.hover != "seek" else 3.2, 2.4 if self.hover != "seek" else 3.2)

    def _draw_buttons(self, p, R, c, t):
        playing = self.octo.media.now().playing
        for name in ("prev", "play", "next"):
            cx, cy, r = R[name]
            hov = self.hover == name
            if self.press == name:
                r *= 0.9
            # glowing pod
            g = QRadialGradient(QPointF(cx, cy), r * 1.9)
            g.setColorAt(0.4, self._c(c, 0.30 + (0.25 if hov else 0) + 0.08 * math.sin(t * 2 + cx)))
            g.setColorAt(1, self._c(c, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawEllipse(QPointF(cx, cy), r * 1.9, r * 1.9)
            g2 = QRadialGradient(QPointF(cx - r * 0.3, cy - r * 0.4), r * 1.3)
            g2.setColorAt(0, QColor(30, 40, 52, 250))
            g2.setColorAt(1, QColor(4, 6, 12, 250))
            p.setBrush(g2)
            p.setPen(QPen(self._c(c, 0.9 if hov else 0.6), 1.2))
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(235, 252, 255) if hov else self._c(c, 1.0).lighter(130))
            s = r * 0.42
            if name == "play":
                if playing:
                    p.drawRoundedRect(QRectF(cx - s * 0.85, cy - s, s * 0.6, 2 * s), 1, 1)
                    p.drawRoundedRect(QRectF(cx + s * 0.25, cy - s, s * 0.6, 2 * s), 1, 1)
                else:
                    p.drawPolygon(QPolygonF([QPointF(cx - s * 0.6, cy - s), QPointF(cx + s * 1.05, cy), QPointF(cx - s * 0.6, cy + s)]))
            else:
                d = 1 if name == "next" else -1
                for off in (-0.5, 0.45):
                    xx = cx + d * off * s
                    p.drawPolygon(QPolygonF([QPointF(xx - d * s * 0.5, cy - s * 0.75), QPointF(xx + d * s * 0.55, cy),
                                             QPointF(xx - d * s * 0.5, cy + s * 0.75)]))

    def _draw_sliders(self, p, R, c, c2, now):
        f = QFont("Segoe UI", 7)
        p.setFont(f)
        for name in ("volume", "balance"):
            r = R[name]
            kn = self.octo.knobs[name]
            y = r.center().y() + 3
            p.setPen(rpen(QColor(255, 255, 255, 45), 2))
            p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
            u = kn.frac()
            x = r.x() + r.width() * u
            pen = QPen(self._c(c, 0.9), 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            if name == "volume":
                p.drawLine(QPointF(r.x(), y), QPointF(x, y))
            else:
                xm = r.center().x()
                p.drawLine(QPointF(xm, y), QPointF(x, y))
                p.setPen(QPen(QColor(255, 255, 255, 90), 1))
                p.drawLine(QPointF(xm, y - 3), QPointF(xm, y + 3))
            p.setPen(Qt.PenStyle.NoPen)
            hov = self.hover == name or self._drag_knob == name
            rg = QRadialGradient(QPointF(x, y), 8)
            rg.setColorAt(0, self._c(c2 if name == "balance" else c, 0.95))
            rg.setColorAt(0.4, self._c(c, 0.4))
            rg.setColorAt(1, self._c(c, 0))
            p.setBrush(rg)
            p.drawEllipse(QPointF(x, y), 8, 8)
            p.setBrush(QColor(255, 255, 255, 235))
            p.drawEllipse(QPointF(x, y), 3.0 if hov else 2.3, 3.0 if hov else 2.3)
            if hov or now < kn.shown_until:
                txt = (f"{kn.tag or 'vol'} {int(round(kn.value * 100))}" if name == "volume"
                       else ("bal C" if abs(kn.value) < 0.025 else f"bal {'L' if kn.value < 0 else 'R'}{int(abs(kn.value) * 100)}"))
            else:
                txt = "vol" if name == "volume" else "bal"
            p.setPen(QColor(210, 230, 240, 170))
            p.drawText(QRectF(r.x(), r.y() - 9, r.width(), 10), Qt.AlignmentFlag.AlignCenter, txt)

    # ------------------------------------------------------------ playlist
    def _rows(self):
        lp = self.octo.local
        if lp.files:
            out = []
            for i, f in enumerate(lp.files):
                ti = tags.track_info(f)
                out.append((i, f"{ti.title or os.path.basename(f)}", i == lp.index))
            return out, "local"
        np_ = self.octo.media.now()
        if np_.source == "smtc":
            return [(None, f"{np_.app}: {np_.title}", True), (None, np_.artist, False), (None, np_.album, False)], "smtc"
        return [(None, "Right-click the octopus > Open files", False)], "none"

    def _panel_rect(self):
        return QRectF(STALK + 10, H0 + 2, W - STALK - 16, PANEL_H - 8)

    def _draw_panel(self, p, c):
        r = self._panel_rect()
        p.setPen(QPen(self._c(c, 0.55), 1.2))
        p.setBrush(QColor(4, 9, 16, 232))
        p.drawRoundedRect(r, 14, 14)
        rows, kind = self._rows()
        f = QFont("Segoe UI", 8)
        p.setFont(f)
        fm = QFontMetrics(f)
        n = int((r.height() - 12) // ROW_H)
        self.scroll = max(0, min(self.scroll, max(0, len(rows) - n)))
        p.save()
        p.setClipRect(r.adjusted(4, 4, -4, -4), Qt.ClipOperation.IntersectClip)
        for k, (idx, txt, cur) in enumerate(rows[self.scroll:self.scroll + n]):
            y = r.y() + 6 + k * ROW_H
            if cur:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self._c(c, 0.2))
                p.drawRoundedRect(QRectF(r.x() + 4, y, r.width() - 8, ROW_H - 2), 6, 6)
            p.setPen(QColor(240, 252, 255, 240) if cur else QColor(190, 215, 225, 200))
            label = (f"{idx + 1:>2}  " if idx is not None else "") + (txt or "")
            p.drawText(QRectF(r.x() + 10, y, r.width() - 20, ROW_H - 2),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(label, Qt.TextElideMode.ElideRight, int(r.width() - 20)))
        p.restore()

    def _row_at(self, y):
        r = self._panel_rect()
        k = int((y - r.y() - 6) // ROW_H)
        rows, kind = self._rows()
        i = self.scroll + k
        if kind == "local" and 0 <= k and 0 <= i < len(rows):
            return rows[i][0]
        return None

    # ------------------------------------------------------------ input
    def _knob_from_x(self, name, x):
        r = self._rects()[name]
        u = max(0.0, min(1.0, (x - r.x()) / r.width()))
        kn = self.octo.knobs[name]
        v = kn.lo + u * (kn.hi - kn.lo)
        if name == "balance" and abs(v) < 0.04:
            v = 0.0
        kn.set(v, self.octo._now())

    def _seek_from_x(self, x):
        r = self._rects()["seek"]
        self._seek_u = max(0.0, min(1.0, (x - r.x()) / r.width()))

    def enterEvent(self, e):
        self.last_hover = self.octo._now()

    def leaveEvent(self, e):
        self.hover = None
        self.last_hover = self.octo._now()

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        now = self.octo._now()
        self.last_hover = now
        self.octo.wake()
        if self._drag_knob in ("volume", "balance"):
            self._knob_from_x(self._drag_knob, x)
            return
        if self._drag_knob == "seek":
            self._seek_from_x(x)
            return
        h = self._hit(x, y)
        if h != self.hover:
            self.hover = h
            self.update()
        self.setCursor(Qt.CursorShape.PointingHandCursor if h and h != "panel" else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = e.position().x(), e.position().y()
        o = self.octo
        o.wake()
        h = self._hit(x, y)
        self.press = h
        if h in ("volume", "balance"):
            o._dial_target()
            self._drag_knob = h
            self._knob_from_x(h, x)
        elif h == "seek":
            self._drag_knob = "seek"
            self._seek_from_x(x)
        elif h == "panel":
            i = self._row_at(y)
            if i is not None:
                lp = o.local
                lp._load(i)
                lp.play()
                o.sounds.play("pop")
                self.media_changed()
        self.update()

    def mouseReleaseEvent(self, e):
        x, y = e.position().x(), e.position().y()
        o = self.octo
        h = self._hit(x, y)
        if self._drag_knob == "seek":
            o.seek_fraction(self._seek_u)
        elif self.press in ("prev", "play", "next") and h == self.press:
            o.sounds.play("pop")
            o.button_action(self.press)
        elif self.press == "art" and h == "art":
            o.sounds.play("pop")
            o.button_action("play")
        elif self.press == "list" and h == "list":
            o.sounds.play("bubble")
            self.set_panel(not self.panel_open)
        self._drag_knob = None
        self.press = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        x, y = e.position().x(), e.position().y()
        h = self._hit(x, y)
        if h in ("volume", "balance"):
            kn = self.octo.knobs[h]
            kn.set(kn.default, self.octo._now())
            self.octo.sounds.play("pop")

    def wheelEvent(self, e):
        x, y = e.position().x(), e.position().y()
        h = self._hit(x, y)
        self.last_hover = self.octo._now()
        if h in ("volume", "balance"):
            kn = self.octo.knobs[h]
            self.octo._dial_target()
            kn.set(kn.value + (e.angleDelta().y() / 120.0) * (kn.hi - kn.lo) / kn.steps, self.octo._now())
        elif self.panel_open and y > H0:
            self.scroll -= int(e.angleDelta().y() / 120 * 2)
        self.update()

    def contextMenuEvent(self, e):
        self.octo.show_menu(e.globalPos())
