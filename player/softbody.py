"""Spring-mesh soft body for the head dome (squish + jiggle) and the sloshing brain.

A coarse NX x NY node grid over the dome rect; structural + shear springs, a pull back
to rest (shape matching, so it always recovers), bottom row pinned to the mantle seam.
Semi-implicit Euler, 2 substeps per frame. Sleeps (zero cost) once it has settled.
"""
from __future__ import annotations

import math

import numpy as np


class SoftDome:
    def __init__(self, w: float, h: float, cx: float, cy: float, nx: int = 7, ny: int = 6):
        self.nx, self.ny, self.w, self.h = nx, ny, w, h
        xs = np.linspace(0, w, nx)
        ys = np.linspace(0, h, ny)
        self.rest = np.array([[x, y] for y in ys for x in xs], np.float64)
        self.p = self.rest.copy()
        self.v = np.zeros_like(self.p)
        self.c = np.array([cx, cy])
        self.pinned = np.zeros(len(self.p), bool)
        self.pinned[(ny - 1) * nx:] = True
        sp = []
        for j in range(ny):
            for i in range(nx):
                a = j * nx + i
                if i + 1 < nx:
                    sp.append((a, a + 1))
                if j + 1 < ny:
                    sp.append((a, a + nx))
                if i + 1 < nx and j + 1 < ny:
                    sp.append((a, a + nx + 1))
                    sp.append((a + 1, a + nx))
        self.sp = np.array(sp)
        self.L0 = np.linalg.norm(self.rest[self.sp[:, 0]] - self.rest[self.sp[:, 1]], axis=1)
        self.k_spring = 260.0
        self.k_rest = 55.0
        self.damp = 7.5
        self.active = False
        # brain slosh: offset spring following the dome centre displacement
        self.brain = np.zeros(2)
        self.brain_v = np.zeros(2)
        self._centre_nodes = [((ny - 1) // 2) * nx + nx // 2, ((ny - 1) // 2 + 1) * nx + nx // 2]

    def poke(self, x: float, y: float, strength: float = 1.0):
        """Click at (x, y) in dome-rect coords: push inward + squash; mirrored for symmetry."""
        strength /= 1.0 + 0.15 * float(np.abs(self.v).max()) / 40.0
        for mx in (x, self.w - x):
            d = self.p - np.array([mx, y])
            dist = np.linalg.norm(d, axis=1)
            fall = np.exp(-(dist / (0.38 * self.w)) ** 2)
            inward = self.c - self.p
            inward /= np.maximum(np.linalg.norm(inward, axis=1)[:, None], 1e-6)
            self.v += (inward * 420.0 * strength + np.array([0.0, 160.0 * strength])) * fall[:, None] * 0.5
        # global squash: wider + flatter
        rel = self.p - self.c
        self.v[:, 0] += rel[:, 0] * 2.2 * strength
        self.v[:, 1] += -rel[:, 1] * 1.4 * strength + 30 * strength
        self.v[self.pinned] = 0
        self.active = True

    def step(self, dt: float) -> bool:
        if not self.active:
            return False
        dt = min(dt, 1 / 30)
        for _ in range(2):
            h = dt / 2
            a, b = self.sp[:, 0], self.sp[:, 1]
            d = self.p[b] - self.p[a]
            L = np.linalg.norm(d, axis=1)
            f = (self.k_spring * (L - self.L0) / np.maximum(L, 1e-6))[:, None] * d
            F = np.zeros_like(self.p)
            np.add.at(F, a, f)
            np.add.at(F, b, -f)
            F += self.k_rest * (self.rest - self.p) - self.damp * self.v
            F[self.pinned] = 0
            self.v += F * h
            self.p += self.v * h
            # keep it a jiggle, not a crumple (rapid clicks stack impulses)
            d = self.p - self.rest
            n = np.linalg.norm(d, axis=1)
            over = n > 13.0
            if over.any():
                self.p[over] = self.rest[over] + d[over] / n[over, None] * 13.0
                self.v[over] *= 0.5
            # brain lags the dome centre on a softer spring -> slosh
            cdisp = (self.p[self._centre_nodes] - self.rest[self._centre_nodes]).mean(0)
            fb = 38.0 * (cdisp * 0.85 - self.brain) - 5.0 * self.brain_v
            self.brain_v += fb * h
            self.brain += self.brain_v * h
        e = float(np.abs(self.v).max() + np.abs(self.p - self.rest).max() + np.abs(self.brain_v).max()
                  + np.abs(self.brain).max())
        if e < 0.06:
            self.p[:] = self.rest
            self.v[:] = 0
            self.brain[:] = 0
            self.brain_v[:] = 0
            self.active = False
        return True

    def quads(self):
        """(src quad, dst quad) per cell, as 4 (x, y) tuples each."""
        nx = self.nx
        out = []
        for j in range(self.ny - 1):
            for i in range(nx - 1):
                idx = (j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
                out.append(([tuple(self.rest[k]) for k in idx], [tuple(self.p[k]) for k in idx]))
        return out

    def max_disp(self) -> float:
        return float(np.abs(self.p - self.rest).max())
