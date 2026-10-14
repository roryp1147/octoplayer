"""Procedural deep-sea UI sounds (numpy only), round 2: soft, short and quiet.
generate_all() writes WAVs to assets/sounds.

squelch_*   head poke: a light, low, wet squish (~0.15 s)
bubble_*    soft water blips (gentle rising sine, low-passed)
pop_*       muted sucker pops (a low, rounded thump, no click)
tick_*      slider detents: barely-there blips
startle     rapid clicks: a faint shimmer
sonar_ping  very soft, short ping (no longer played at boot)
bubbles_sleep / sleep / wake: very soft (not used by default)
"""
from __future__ import annotations

import os
import wave

import numpy as np

SR = 44100


def _t(d):
    return np.arange(int(SR * d)) / SR


def _bp(x, lo, hi):
    n = len(x)
    nf = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
    if hi:
        g *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, nf)[:n]


def _reverb(x, dur=1.2, mix=0.35, seed=1, damp=3000):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    ir = rng.standard_normal(len(t)) * np.exp(-t / (dur / 5))
    ir = _bp(ir, 80, damp)
    ir /= np.abs(ir).sum() ** 0.5 * 8
    n = len(x) + len(ir)
    nf = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(ir, nf), nf)[:n]
    out = np.zeros(n)
    out[:len(x)] += x * (1 - mix)
    out += y * mix / max(np.abs(y).max(), 1e-9) * np.abs(x).max()
    return out


def _fin(x, peak=0.8):
    x = x - x.mean()
    f = min(len(x) // 4, int(0.004 * SR))
    if f:
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
    return x / max(np.abs(x).max(), 1e-9) * peak


def _chirp(f0, f1, d, tau):
    t = _t(d)
    f = f0 + (f1 - f0) * (1 - np.exp(-t / (d / 3)))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau)


def _lp(x, hi):
    return _bp(x, 0, hi)


def _env(n, att, tau):
    t = np.arange(n) / SR
    return (1 - np.exp(-t / max(att, 1e-4))) * np.exp(-t / tau)


def sonar_ping():
    t = _t(0.9)
    x = np.sin(2 * np.pi * 720 * t) * _env(len(t), 0.02, 0.18)
    x += 0.25 * np.sin(2 * np.pi * 1080 * t) * _env(len(t), 0.02, 0.1)
    return _fin(_reverb(_lp(x, 2500), 0.7, 0.3, damp=1800), 0.12)


def squelch(v=0):
    rng = np.random.default_rng(10 + v)
    d = 0.16
    t = _t(d)
    n = rng.standard_normal(len(t))
    wet = _bp(n, 260 + 30 * v, 900) * _env(len(t), 0.008, 0.045)
    body = _chirp(210 + 20 * v, 140, d, 0.05)
    x = _lp(wet * 0.6 + body * 0.8, 1600)
    return _fin(_reverb(x, 0.25, 0.12, seed=3, damp=1500), 0.30)


def bubble(v=0):
    rng = np.random.default_rng(30 + v)
    f0 = [380, 460, 540][v % 3] * rng.uniform(0.97, 1.03)
    d = 0.07
    t = _t(d)
    f = f0 * (1 + 0.6 * (1 - np.exp(-t / 0.02)))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.004, 0.018)
    return _fin(_reverb(_lp(x, 2000), 0.25, 0.12, seed=4, damp=2500), 0.26)


def bubbles_sleep():
    rng = np.random.default_rng(55)
    d = 1.2
    out = np.zeros(int(SR * d))
    for i in range(3):
        b = bubble(i) * rng.uniform(0.5, 0.9)
        a = int(rng.uniform(0, d - 0.4) * SR)
        out[a:a + len(b)] += b
    return _fin(out, 0.14)


def pop(v=0):
    d = 0.07
    t = _t(d)
    f = (150 + 15 * v) - 60 * (1 - np.exp(-t / 0.02))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.003, 0.016)
    return _fin(_lp(x, 700), 0.28)


def tick(v=0):
    d = 0.025
    t = _t(d)
    x = np.sin(2 * np.pi * (760 + 90 * v) * t) * _env(len(t), 0.002, 0.006)
    return _fin(_lp(x, 1800), 0.12)


def startle():
    t = _t(0.4)
    x = sum(np.sin(2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * 4 * t)) for f in (880, 1320))
    x *= _env(len(t), 0.03, 0.12)
    return _fin(_reverb(_lp(x, 3000), 0.5, 0.3), 0.16)


def glide(up):
    t = _t(0.9)
    f = (140 + 90 * (t / t[-1])) if up else (230 - 90 * (t / t[-1]))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 2
    return _fin(_lp(x, 900), 0.10)


SOUNDS = {
    "sonar_ping": sonar_ping,
    "squelch_0": lambda: squelch(0), "squelch_1": lambda: squelch(1), "squelch_2": lambda: squelch(2),
    "bubble_0": lambda: bubble(0), "bubble_1": lambda: bubble(1), "bubble_2": lambda: bubble(2),
    "bubbles_sleep": bubbles_sleep,
    "pop_0": lambda: pop(0), "pop_1": lambda: pop(1), "pop_2": lambda: pop(2),
    "tick_0": lambda: tick(0), "tick_1": lambda: tick(1),
    "startle": startle, "sleep": lambda: glide(False), "wake": lambda: glide(True),
}


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    st = np.stack([x, x], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())


def generate_all(folder, overwrite=True):
    os.makedirs(folder, exist_ok=True)
    for name, fn in SOUNDS.items():
        p = os.path.join(folder, name + ".wav")
        if overwrite or not os.path.exists(p):
            write_wav(p, fn())


if __name__ == "__main__":
    import sys
    generate_all(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "assets", "sounds"))
