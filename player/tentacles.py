"""Round 3 tentacles: one continuous kinematic chain per tentacle.

tools/build_layers.py cuts every tentacle ACROSS its length into short segments (bands of geodesic
distance from the root, so curled legs are cut correctly) and records the rest spine (joint points).
At runtime a sideways displacement D(s, t) (slow sway + a sine travelling base -> tip + a smaller
faster wave, envelope 0 at the root growing to the tip) is turned into joint angles
alpha_j = dD/ds; the joints are re-chained (P'_{j+1} = P'_j + R(alpha_j)(P_{j+1} - P_j)), and every
segment is drawn rigidly about its own joint. Neighbouring segments share their joint point and
cross-fade over a few px, so the tentacle bends as one continuous strip (no seams, no shear).
The whole tentacle layer is rendered into a cached image at ~30 Hz (motion is ~0.1 px/frame) and
blitted every frame; it is re-rendered every frame only while a sucker ripple is travelling.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter, QTransform

CM = QPainter.CompositionMode


class TentacleRig:
    def __init__(self, art, seed=7):
        T = art.L["tentacles"]
        self.L = T["seg_l"]
        self.s0 = T.get("static_s", 0.0)
        rng = random.Random(seed)
        self.tents = []
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        for t in T["tents"]:
            pts = np.array(t["spine"], np.float64)
            vec = pts[1:] - pts[:-1]
            sj = np.arange(len(pts)) * self.L
            segs = []
            for g in t["segs"]:
                g["src"] = QRectF(g["ax"], g["ay"], g["w"], g["h"])
                g["dst"] = QRectF(g["lx"], g["ly"], g["w"], g["h"])
                g["smid"] = g["s0"] + self.L / 2
                segs.append(g)
            wide = t["t"] in (0, len(T["tents"]) + 1) or len(segs) and max(g["w"] for g in segs) > 100
            self.tents.append({
                "pts": pts, "vec": vec, "sj": sj, "smax": t["smax"], "segs": segs, "j0": t.get("j0", 0),
                "p1": rng.uniform(3.8, 5.4), "ph1": rng.uniform(0, 6.283),
                "p2": rng.uniform(2.2, 3.1), "ph2": rng.uniform(0, 6.283),
                "p3": rng.uniform(7.5, 11.0), "ph3": rng.uniform(0, 6.283),
                "lam": rng.uniform(380, 460) if wide else rng.uniform(270, 330),
                "amp": rng.uniform(4.0, 5.0) if wide else rng.uniform(5.0, 6.2),
            })
            x0, y0 = min(x0, pts[:, 0].min()), min(y0, pts[:, 1].min())
            x1, y1 = max(x1, pts[:, 0].max()), max(y1, pts[:, 1].max())
        self.bbox = (x0, y0, x1, y1)
        self.n = sum(len(t["segs"]) for t in self.tents)
        self.angle = [np.zeros(len(t["pts"])) for t in self.tents]
        self.joint = [t["pts"].copy() for t in self.tents]
        self.disp = [0.0] * len(self.tents)          # tip displacement (px) per tentacle

    def env(self, s, smax):
        u = np.clip((s - self.s0) / max(1.0, smax - self.s0), 0.0, 1.0)
        return (u * u * (3 - 2 * u)) ** 1.15

    def step(self, wt: float, amp_k: float = 1.0, bass: float = 0.0):
        tau = 2 * math.pi
        a_k = amp_k * (1.0 + 0.35 * bass)
        for i, T in enumerate(self.tents):
            w1 = tau * wt / T["p1"] + T["ph1"]
            w2 = tau * wt / T["p2"] + T["ph2"]
            sway = math.sin(tau * wt / T["p3"] + T["ph3"])
            kl = tau / T["lam"]
            s = T["sj"]
            x = s - self.s0
            D = T["amp"] * a_k * self.env(s, T["smax"]) * (
                0.55 * sway + 0.75 * np.sin(w1 - kl * x) + 0.28 * np.sin(w2 - 2.1 * kl * x))
            al = np.zeros(len(s))
            al[:-1] = (D[1:] - D[:-1]) / self.L
            c, sn = np.cos(al[:-1]), np.sin(al[:-1])
            v = T["vec"]
            rv = np.stack([c * v[:, 0] - sn * v[:, 1], sn * v[:, 0] + c * v[:, 1]], 1)
            J = np.empty_like(T["pts"])
            J[0] = T["pts"][0]
            J[1:] = T["pts"][0] + np.cumsum(rv, 0)
            self.angle[i] = np.degrees(al)
            self.joint[i] = J
            self.disp[i] = float(np.hypot(*(J[-1] - T["pts"][-1])))

    def draw(self, p: QPainter, img: QImage, opacity_fn=None, off=(0.0, 0.0)):
        """Draw every segment (Plus). opacity_fn(smid, smax) -> opacity, for the sucker overlay."""
        p.setCompositionMode(CM.CompositionMode_Plus)
        for i, T in enumerate(self.tents):
            J, A = self.joint[i], self.angle[i]
            for g in T["segs"]:
                j = g["j"]
                if opacity_fn is not None:
                    o = opacity_fn(g["smid"], T["smax"])
                    if o < 0.01:
                        continue
                    p.setOpacity(min(1.0, o))
                a = math.radians(g["phi"] + A[j])
                c, sn = math.cos(a), math.sin(a)
                p.setTransform(QTransform(c, sn, -sn, c, J[j, 0] - off[0], J[j, 1] - off[1]))
                p.drawImage(g["dst"], img, g["src"])
        p.resetTransform()
        p.setOpacity(1.0)
