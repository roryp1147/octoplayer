"""Psychedelic but smooth colour field.

Hue positions (0..360) map to colours through a hand-keyed wheel interpolated in OKLab
(perceptually even, no muddy midpoints), normalised to full brightness. The violet key
is a true blue-violet; magenta/rose only get a short arc of the wheel (round-2 fix:
'violet turning magenta').

The field: hue(r, t) = H0(t) - r * SPREAD + wave(r, t)
  H0      a full trip round the wheel every CYCLE_S seconds
  SPREAD  ~150 deg of rainbow from the head to the tentacle tips (rings of hue)
  wave    a faster hue ripple travelling outward from the head
Beat pulses push brightness and briefly bleach toward white.
"""
from __future__ import annotations

import math

import numpy as np

CYCLE_S = 24.0           # a full wheel
SPREAD = 0.38            # deg per display px from the hub
WAVE_DEG, WAVE_LEN, WAVE_S = 34.0, 170.0, 3.4
EYE_CYCLE_S = 9.0

_KEYS = [  # (wheel position, linear-ish sRGB)
    (0, (1.00, 0.10, 0.07)),     # abyssal red
    (38, (1.00, 0.50, 0.06)),    # amber
    (78, (0.80, 1.00, 0.12)),    # chartreuse
    (118, (0.12, 1.00, 0.42)),   # sea green
    (160, (0.04, 0.95, 0.80)),   # teal
    (198, (0.08, 0.72, 1.00)),   # cyan
    (232, (0.14, 0.30, 1.00)),   # deep blue
    (268, (0.46, 0.20, 1.00)),   # violet (blue-violet, never pink)
    (304, (0.90, 0.16, 0.85)),   # magenta (short arc)
    (334, (1.00, 0.12, 0.38)),   # rose
    (360, (1.00, 0.10, 0.07)),
]


def _srgb_to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _to_oklab(rgb):
    l_ = _srgb_to_lin(rgb)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(l_ @ M1.T)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return lms @ M2.T


def _from_oklab(lab):
    M2i = np.array([[1, 0.3963377774, 0.2158037573],
                    [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480]])
    lms = (lab @ M2i.T) ** 3
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return _lin_to_srgb(lms @ M1i.T)


def _build_lut(n=720):
    pos = np.array([k[0] for k in _KEYS], np.float64)
    labs = _to_oklab(np.array([k[1] for k in _KEYS]))
    out = np.zeros((n, 3))
    for i in range(n):
        h = i * 360.0 / n
        j = int(np.searchsorted(pos, h, side="right") - 1)
        j = min(j, len(pos) - 2)
        u = (h - pos[j]) / (pos[j + 1] - pos[j])
        u = u * u * (3 - 2 * u) * 0.35 + u * 0.65          # linger a little on each key
        c = _from_oklab(labs[j] * (1 - u) + labs[j + 1] * u)
        c = np.clip(c, 0, None)
        out[i] = c / max(c.max(), 1e-6)
    return out.astype(np.float32)


LUT = _build_lut()
_N = len(LUT)


def wheel(h: float) -> np.ndarray:
    """Colour (max channel 1) at wheel position h (degrees), linearly sampled from the LUT."""
    x = (h % 360.0) * _N / 360.0
    i = int(x)
    f = x - i
    return LUT[i % _N] * (1 - f) + LUT[(i + 1) % _N] * f


def wheel_many(hs: np.ndarray) -> np.ndarray:
    x = (np.asarray(hs, np.float64) % 360.0) * _N / 360.0
    i = x.astype(np.int64)
    f = (x - i)[..., None]
    return LUT[i % _N] * (1 - f) + LUT[(i + 1) % _N] * f


def shade(c, sat=1.0, val=1.0, white=0.0):
    """Saturation (mix toward neutral of equal max), value scale, and a bleach toward white."""
    c = np.asarray(c, np.float32)
    c = 1.0 - sat * (1.0 - c) if sat < 1 else c
    c = c * (1 - white) + white
    return np.clip(c * val, 0, 1)


def base_hue(t: float) -> float:
    return 360.0 * t / CYCLE_S + 18.0 * math.sin(t / 7.3)


def hue_at(r, t: float, h0: float | None = None):
    """Hue (deg) at distance r (display px, scalar or array) from the tentacle hub."""
    h0 = base_hue(t) if h0 is None else h0
    return h0 - np.asarray(r) * SPREAD + WAVE_DEG * np.sin(2 * np.pi * (np.asarray(r) / WAVE_LEN - t / WAVE_S))


def eye_hue(t: float) -> float:
    """Eyes: their own, faster cycle, roughly opposite the body."""
    return 360.0 * t / EYE_CYCLE_S + 180.0 + 25.0 * math.sin(t / 3.1)


# ---------------------------------------------------------------- round 3: one solid colour
# The whole body is ONE colour at a time: hold HOLD_S, then a XFADE_S cross-fade to the next.
SEQ = [0.0, 175.0, 268.0, 130.0, 225.0, 28.0, 300.0, 190.0, 95.0, 345.0]   # red teal violet green blue orange pink cyan lime rose
HOLD_S = 4.0
XFADE_S = 2.0


def _mix(a, b, u):
    """cross-fade two wheel colours in OKLCh (lightness, chroma and hue interpolated along the
    short arc): stays vivid all the way, normalised to max channel 1."""
    la, lb = _to_oklab(np.asarray(a, np.float64)), _to_oklab(np.asarray(b, np.float64))
    ca, cb = math.hypot(la[1], la[2]), math.hypot(lb[1], lb[2])
    ha, hb = math.atan2(la[2], la[1]), math.atan2(lb[2], lb[1])
    dh = (hb - ha + math.pi) % (2 * math.pi) - math.pi
    h = ha + dh * u
    C = ca + (cb - ca) * u
    L = la[0] + (lb[0] - la[0]) * u
    c = np.clip(_from_oklab(np.array([L, C * math.cos(h), C * math.sin(h)])), 0, None)
    return (c / max(1e-6, float(c.max()))).astype(np.float32)


def solid(t: float, offset: float = 0.0):
    """(colour, hue, fading) of the body at colour-clock t. offset rotates the hue (eyes: 180)."""
    period = HOLD_S + XFADE_S
    n = len(SEQ)
    k = int(t // period)
    f = t - k * period
    h0, h1 = SEQ[k % n] + offset, SEQ[(k + 1) % n] + offset
    if f < HOLD_S:
        return wheel(h0), h0 % 360, False
    u = (f - HOLD_S) / XFADE_S
    u = u * u * (3 - 2 * u)
    return _mix(wheel(h0), wheel(h1), u), (h0 if u < 0.5 else h1) % 360, True
