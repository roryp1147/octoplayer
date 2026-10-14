"""Layer loading + live recolouring (round 2: fast, spatially varying colour).

Each colourable layer is stored as two premultiplied images built offline:
    B = (base rgb, alpha A - M)      dark, untinted flesh
    M = glow mask (rgb = alpha = M)
and re-combined as   out = B (+) M * T(x, y)
where T is a gradient brush (radial rings of hue around the head, or per-tentacle
vertical gradients in the strip atlas). Qt does it in three raster passes
(SourceIn gradient onto a buffer whose alpha is M, copy B, Plus), so a full recolour of every layer
costs ~1-2 ms and runs at RECOLOR_HZ while the strips/dome animate at 60 fps.
Because M <= A - max(base), out is valid premultiplied with alpha exactly A.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter, QPixmap

from . import config
from .qtutil import qimage_to_gray, qimage_to_rgba

CM = QPainter.CompositionMode


def _load(name):
    p = os.path.join(config.LAYERS, name + ".png")
    img = QImage(p)
    if img.isNull():
        raise FileNotFoundError(p)
    if img.isGrayscale():
        return qimage_to_gray(img)
    return qimage_to_rgba(img)[..., :3].copy()


def _argb(rgb8: np.ndarray, a8: np.ndarray) -> QImage:
    h, w = a8.shape
    arr = np.empty((h, w, 4), np.uint8)
    arr[..., 0] = rgb8[..., 2]
    arr[..., 1] = rgb8[..., 1]
    arr[..., 2] = rgb8[..., 0]
    arr[..., 3] = a8
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_ARGB32_Premultiplied).copy()


def to_qimage(rgb_pm: np.ndarray, a: np.ndarray) -> QImage:
    a8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    pm = (np.clip(np.minimum(rgb_pm, a[..., None]), 0, 1) * 255 + 0.5).astype(np.uint8)
    return _argb(pm, a8)


def to_pixmap(rgb_pm, a) -> QPixmap:
    return QPixmap.fromImage(to_qimage(rgb_pm, a))


def mask_image(g: np.ndarray) -> QImage:
    """white premultiplied mask (rgb = alpha = g): fill it with SourceIn to colour it."""
    g8 = (np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8)
    return _argb(np.dstack([g8] * 3), g8)


def gray_pixmap(g: np.ndarray, color, alpha_from_gray=True) -> QPixmap:
    c = np.asarray(color, np.float32)
    rgb = g[..., None] * c[None, None, :]
    a = np.clip(g * (max(c.max(), 1e-3) if alpha_from_gray else 1.0), 0, 1)
    return to_pixmap(np.minimum(rgb, a[..., None]), a)


class Colourable:
    """B (+) M*T layer cropped to its bounding box (see module doc)."""

    def __init__(self, prefix, crop=True):
        base = _load(prefix + "_base")
        ba = _load(prefix + "_ba")
        m = _load(prefix + "_m")
        a = ba + m
        if crop:
            ys, xs = np.where(a > 0.003)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            y0, x0, (y1, x1) = 0, 0, a.shape
        self.rect = QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        sl = (slice(y0, y1), slice(x0, x1))
        ba8 = (np.clip(ba[sl], 0, 1) * 255 + 0.5).astype(np.uint8)
        b8 = np.minimum((np.clip(base[sl], 0, 1) * 255 + 0.5).astype(np.uint8), ba8[..., None])
        self.B = _argb(b8, ba8)
        self.M = mask_image(m[sl])
        self.alpha = np.clip(a[sl], 0, 1)
        w, h = self.rect.width(), self.rect.height()
        self.out = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        # tmp keeps alpha = M forever: SourceIn replaces only its colour (T * M) each time
        self._tmp = self.M.copy()
        self.out.fill(0)

    def recolor(self, fill):
        """fill(painter, rect_in_local_coords, offset_qpoint) paints the colour field."""
        p = QPainter(self._tmp)
        p.setCompositionMode(CM.CompositionMode_SourceIn)
        fill(p, self.rect)
        p.end()
        p = QPainter(self.out)
        p.setCompositionMode(CM.CompositionMode_Source)
        p.drawImage(0, 0, self.B)
        p.setCompositionMode(CM.CompositionMode_Plus)
        p.drawImage(0, 0, self._tmp)
        p.end()
        return self.out


class Art:
    def __init__(self):
        with open(os.path.join(config.LAYERS, "layout.json")) as f:
            self.L = json.load(f)
        self.W, self.H = self.L["W"], self.L["H"]
        self.body = Colourable("body")
        self.dome = Colourable("dome")
        self.tent = Colourable("tent", crop=False)
        self.tent_s = mask_image(_load("tent_s"))          # sucker rings (beat ripple overlay)
        self.hit_alpha = np.clip(_load("hit_a"), 0, 1)
        dm = _load("dome_mask")
        ys, xs = np.where(dm > 0.003)
        pad = 6
        self.dome_rect = QRect(int(xs.min()) - pad, int(ys.min()) - pad,
                               int(xs.max() - xs.min()) + 1 + 2 * pad, int(ys.max() - ys.min()) + 1 + 2 * pad)
        r = self.dome_rect

        def crop(g):
            return np.ascontiguousarray(g[r.y():r.y() + r.height(), r.x():r.x() + r.width()])
        self.dome_plane_offset = (self.dome.rect.x() - r.x(), self.dome.rect.y() - r.y())
        self.glass_dark = crop(_load("glass_dark"))
        self.glass_tint = crop(_load("glass_tint"))
        self.glass_hl = crop(_load("glass_hl"))
        self.brain_clip = mask_image(crop(_load("brain_clip")))
        # brain planes
        self.brain_rgb = _load("brain_rgb")
        self.brain_gray = _load("brain_gray")
        self.brain_a = _load("brain_a")
        self.brain_em = _load("brain_em")
        # eyes
        eye = _load("eye_em")
        er = self.L["eye_r"] + 5
        self.eye_rects, self.eye_em = [], []
        for ex, ey in self.L["eyes"]:
            rr = QRect(int(ex - er), int(ey - er), int(2 * er) + 1, int(2 * er) + 1)
            self.eye_rects.append(rr)
            self.eye_em.append(eye[rr.y():rr.y() + rr.height(), rr.x():rr.x() + rr.width()].copy())
        self.dots = [(d, _load(d["name"])) for d in self.L["dots"]]
        self.flash_g = _load("flash")
