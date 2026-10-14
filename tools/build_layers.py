#!/usr/bin/env python3
"""Build Octoplayer's runtime layers from the approved 2048 px art (brain v3).

Run on the dev box (needs numpy, pillow, opencv-python-headless, scipy):
    python tools/build_layers.py [--art ../art] [--height 640]

Outputs assets/layers/*.png (display resolution, premultiplied-ready planes) +
assets/layers/layout.json. The app itself only needs numpy + PySide6.

Refinement done here (the Stage-2 'gaps'):
  * matte: alpha from luminance, colour de-contaminated at the edge (no black fringe),
    premultiplied downscale (no dark halo when resampled)
  * tentacle seams: dynamic-programming cut along the darkest valley between tentacles
    (polar coordinates around the tentacle hub), mirrored -> exact symmetry
  * occlusion: moving tentacles fade into the static body at their root (exact re-composition
    at rest via additive premultiplied blend; no ghost when they swing out); dome is cut out
    with a soft seam so the squish mesh can warp it independently
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
OUT = os.path.join(APP, "assets", "layers")


def load(path, mode=None):
    im = Image.open(path)
    if mode:
        im = im.convert(mode)
    return np.asarray(im).astype(np.float32) / 255.0


def save_gray(name, a):
    Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8), "L").save(os.path.join(OUT, name + ".png"))


def save_rgb(name, a):
    Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8), "RGB").save(os.path.join(OUT, name + ".png"))


ROOT_A, ROOT_B = 96.0, 112.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.join(APP, "..", "art"))
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    art = args.art
    S = 2048
    k = S / 1264
    cut = load(os.path.join(art, "octoplayer_cutout_2048.png"), "RGBA")
    rgb, A = cut[..., :3].copy(), cut[..., 3].copy()
    em = load(os.path.join(art, "layers", "emissive_mask.png"), "L")
    spec = load(os.path.join(art, "layers", "spectrum_dot_mask.png"), "L")

    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, rx, ry = S / 2, 195 * k, 180 * k, 140 * k
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)

    # ---------------------------------------------------------------- matte
    A = np.clip((A - 0.04) / 0.96, 0, 1)
    A[A < 0.02] = 0
    # edge colour de-contamination: pull edge colours from the interior
    w = cv2.GaussianBlur(A, (0, 0), 3)
    inter = cv2.GaussianBlur(rgb * A[..., None], (0, 0), 3) / np.maximum(w[..., None], 1e-4)
    edge = np.clip((0.7 - A) / 0.7, 0, 1)[..., None]
    rgb = rgb * (1 - edge) + inter * edge
    # exact symmetry
    def sym(a):
        a = a.copy()
        a[:, S // 2:] = a[:, :S // 2][:, ::-1][:, :S - S // 2]
        return a
    rgb, A, em, spec = sym(rgb), sym(A), sym(em), sym(spec)

    # --------------------------------------------------------------- dome
    glass = np.clip((0.97 - d) / 0.08, 0, 1) * np.clip((335 * k - ys) / (45 * k), 0, 1)
    rgb = rgb * (1 - glass * 0.85)[..., None]           # darkened dome back (glassy)
    dome_m = np.clip((1.06 - d) / 0.08, 0, 1) * np.clip((345 * k - ys) / (55 * k), 0, 1)
    inner = np.clip((0.86 - d) / 0.14, 0, 1) * np.clip((300 * k - ys) / (40 * k), 0, 1)
    fres = np.clip((d - 0.6) / 0.4, 0, 1) ** 2 * glass
    hl = np.exp(-(((xs - (cx - 80 * k)) / (60 * k)) ** 2 + ((ys - (cy - 70 * k)) / (28 * k)) ** 2)) * 0.45 * glass
    # dot ring (spectrum lights)
    dr = np.sqrt(((xs - cx) / 270) ** 2 + ((ys - 420) / 235) ** 2)
    ringw = np.clip(spec * 1.6, 0, 1) * (np.abs(dr - 1) < 0.11) * (ys < 410)
    ringbin = ((spec > 0.4) & (np.abs(dr - 1) < 0.11) & (ys < 410)).astype(np.uint8)
    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(ringbin, 8)
    for c in range(1, _n):
        if _st[c, cv2.CC_STAT_AREA] < 40:
            ringbin[_lab == c] = 0
    # every ring pixel belongs to its nearest dot (Voronoi), within 16 px
    dist, vlab = cv2.distanceTransformWithLabels(1 - ringbin, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(ringbin, 8)
    # map voronoi labels -> component ids
    vmap = np.zeros(vlab.max() + 1, np.int32)
    vmap[vlab[ringbin > 0]] = _lab[ringbin > 0]
    lab_full = vmap[vlab] * ((dist < 16) & (ringw > 0.02))
    # remove the ring from the static emissive (it is drawn per-dot, live)
    em_noring = em * (1 - ringw)

    # --------------------------------------------------------------- tentacles
    hub = (cx, 470 * k)
    ang = np.degrees(np.arctan2(xs - hub[0], ys - hub[1]))        # 0 = straight down, <0 = viewer left
    rad = np.hypot(xs - hub[0], ys - hub[1])
    lum0 = cv2.GaussianBlur((rgb.mean(2) * A), (0, 0), 4)
    rmax = int(rad[A > 0.1].max())
    r0 = int(70 * k)
    rs = np.arange(r0, rmax, 2)

    def seam(theta0, span=11.0, step=0.25, smooth=0.6):
        ths = np.arange(theta0 - span, theta0 + span + 1e-6, step)
        R, T = np.meshgrid(rs, np.radians(ths), indexing="ij")
        px = hub[0] + R * np.sin(T)
        py = hub[1] + R * np.cos(T)
        cost = cv2.remap(lum0, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR)
        cost = cost + 0.02 * np.abs(ths - theta0)[None, :] / span   # weak pull to the prior
        n, m = cost.shape
        acc = cost.copy()
        back = np.zeros((n, m), np.int32)
        for i in range(1, n):
            best = np.full(m, np.inf)
            arg = np.zeros(m, np.int32)
            for dj in (-2, -1, 0, 1, 2):
                prev = np.roll(acc[i - 1], dj)
                if dj > 0:
                    prev[:dj] = np.inf
                elif dj < 0:
                    prev[dj:] = np.inf
                c = prev + smooth * 0.01 * abs(dj)
                better = c < best
                best[better] = c[better]
                arg[better] = (np.arange(m) - dj)[better]
            acc[i] = best + cost[i]
            back[i] = arg
        j = int(np.argmin(acc[-1]))
        path = np.zeros(n)
        for i in range(n - 1, -1, -1):
            path[i] = ths[j]
            j = back[i, j]
        # light smoothing of the cut
        path = np.convolve(np.pad(path, 6, mode="edge"), np.ones(13) / 13, mode="valid")
        return path

    left_prior = [-42.0, -24.0, -9.0]
    seams_l = [seam(t) for t in left_prior]
    ridx = np.clip(((rad - r0) / 2).astype(int), 0, len(rs) - 1)
    bounds = [np.full_like(ang, -180.0)]
    for p in seams_l:
        bounds.append(p[ridx])
    bounds.append(np.zeros_like(ang))
    for p in reversed(seams_l):
        bounds.append(-p[ridx])
    bounds.append(np.full_like(ang, 180.0))
    soft = 1.2   # degrees of seam feather
    tent_body = np.clip((ys - 430 * k) / 16.0 + 0.5, 0, 1)      # soft: both sides resample cleanly
    tents = []
    for i in range(8):
        lo, hi = bounds[i], bounds[i + 1]
        m = np.clip((ang - lo) / soft + 0.5, 0, 1) * np.clip((hi - ang) / soft + 0.5, 0, 1)
        if i == 0:
            m = np.clip((hi - ang) / soft + 0.5, 0, 1)
        if i == 7:
            m = np.clip((ang - lo) / soft + 0.5, 0, 1)
        tents.append(m * tent_body)
    # all 8 tentacles move (round 2): each fades in from the head (additive partition with the body)
    # round 3: the roots are baked into the static body; tentacles fade in at display r 96..112
    root_fade = np.clip((rad - ROOT_A / 0.366762) / ((ROOT_B - ROOT_A) / 0.366762), 0, 1)
    tm8 = [tents[i] * root_fade * A for i in range(8)]
    # round 3: the outer leg crosses the seam between sectors 0/1 (and 6/7) -> one moving unit
    tmask = [tm8[0] + tm8[1], tm8[2], tm8[3], tm8[4], tm8[5], tm8[6] + tm8[7]]
    hp = em - cv2.GaussianBlur(em, (0, 0), 8 * k)
    suck = np.clip((hp - 0.06) * 3.0, 0, 1) * np.clip(em * 1.6, 0, 1)

    # ------------------------------------------------- eye split (own colour cycle)
    eyem = np.zeros((S, S), np.float32)
    for sx in (-1, 1):
        e = (((xs - (cx + sx * 92 * k)) / (34 * k)) ** 2 + ((ys - 469 * k) / (31 * k)) ** 2)
        eyem = np.maximum(eyem, np.clip((1.0 - e) / 0.3, 0, 1))
    eyecore = np.zeros((S, S), np.float32)
    for sx in (-1, 1):
        e = (((xs - (cx + sx * 92 * k)) / (24 * k)) ** 2 + ((ys - 469 * k) / (22 * k)) ** 2)
        eyecore = np.maximum(eyecore, np.clip((1.0 - e) / 0.35, 0, 1))
    em_eye = np.clip(em * eyem + rgb.mean(2) * 0.6 * eyecore, 0, 1.5)
    em_body = em_noring * (1 - eyem)

    # ---------------------------------------------------- tint planes (premultiplied)
    # runtime: out = B + M * T(x, y)  with B = (base rgb, alpha A - M) and M = glow mask,
    # recombined by an additive (Plus) blend -> alpha = A exactly, rgb <= alpha for T <= 1.
    lum = rgb.mean(2) * (1 - 0.75 * eyecore)
    BASEK, LUMK, EMK = 0.14, 0.9, 1.5
    neutral = rgb.mean(2, keepdims=True) * 0.65 + rgb * 0.35      # mostly neutral: the hue comes from T
    base_full = neutral * BASEK * (1 - 0.6 * eyecore)[..., None]
    em_k = em_body * EMK * (1 - glass * 0.3)
    M_full = lum * LUMK + em_k

    # crop / scale
    yy, xx = np.where(A > 0.02)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    s = args.height / (y1 - y0)
    MX, MT, MB = 20, 68, 18
    EXT = 30                     # display rows above the art crop: the dome glow / glass fall-off stays round
    Wd = int(round((x1 - x0) * s)) + 2 * MX
    Wd += Wd % 2
    Hd = args.height + MT + MB
    OX = (Wd - int(round((x1 - x0) * s))) // 2

    def tod(a):
        """crop + premultiplied-safe downscale to the display canvas"""
        E = int(round(EXT / s))
        c = a[y0 - E:y1, x0:x1]
        w = int(round((x1 - x0) * s))
        r = cv2.resize(c, (w, args.height + EXT), interpolation=cv2.INTER_AREA)
        out = np.zeros((Hd, Wd) + r.shape[2:], np.float32)
        out[MT - EXT:MT + args.height, OX:OX + w] = r
        return out

    def P(x, y):
        """2048 coords -> display coords"""
        return ((x - x0) * s + OX, (y - y0) * s + MT)

    def split_bm(ad, bd, md):
        md = np.clip(np.minimum(md, ad - bd.max(2)), 0, None)
        return np.clip(ad - md, 0, 1), md

    def put(prefix, m):
        ad = tod(m)
        bd = tod(base_full * m[..., None])
        md = tod(M_full * m)
        ba, md = split_bm(ad, bd, md)
        save_rgb(prefix + "_base", bd)
        save_gray(prefix + "_ba", ba)
        save_gray(prefix + "_m", md)

    tent_info, static_extra = build_segments(tmask, base_full, M_full, suck, hub, s, x0, y0, OX, MT, P, tod)
    tm_total = sum(tmask[t["t"]] for t in tent_info["tents"]) - static_extra
    static_m = np.clip(A * (1 - dome_m) - tm_total, 0, 1)
    put("body", static_m)
    dm = A * dome_m
    put("dome", dm)
    # eye glow (own tint), lives on top of the body
    save_gray("eye_em", tod(em_eye * EMK * A))
    # glass pieces over the brain (inside the dome)
    save_gray("glass_dark", tod(fres * 0.55 * dome_m))
    save_gray("glass_tint", tod(glass * 0.12 * dome_m))
    save_gray("glass_hl", tod(hl * dome_m))
    save_gray("brain_clip", tod(inner))
    save_gray("dome_mask", tod(dome_m * A))
    save_gray("hit_a", tod(A))
    # startle flash: all emissive
    save_gray("flash", np.clip(tod(em_body * EMK * A) + tod(em_eye * EMK * A), 0, 1))

    # ---------------------------------------------------------- tentacle strips


    # ---------------------------------------------------------- dots
    ncomp, lab = _n, lab_full
    dots = []
    ringd = tod(ringw * em * EMK * 1.3 + ringw * 0.15)
    labd = cv2.resize(lab[y0:y1, x0:x1].astype(np.float32), (int(round((x1 - x0) * s)), args.height),
                      interpolation=cv2.INTER_NEAREST)
    labD = np.zeros((Hd, Wd), np.float32)
    ox = (Wd - labd.shape[1]) // 2
    labD[MT:MT + args.height, ox:ox + labd.shape[1]] = labd
    for c in range(1, ncomp):
        m = (labD == c)
        if m.sum() < 3:
            continue
        yy2, xx2 = np.where(m)
        bx0, bx1, by0, by1 = xx2.min() - 1, xx2.max() + 2, yy2.min() - 1, yy2.max() + 2
        g = (ringd * m)[by0:by1, bx0:bx1]
        if g.max() < 0.05:
            continue
        name = f"dot_{len(dots):02d}"
        save_gray(name, np.clip(g, 0, 1))
        dcx, dcy = float(xx2.mean()), float(yy2.mean())
        dots.append({"name": name, "x": int(bx0), "y": int(by0), "cx": dcx, "cy": dcy})
    # band index: per side, bottom (bass) -> crown (treble)
    axd = P(cx, 0)[0]
    left = sorted([d_ for d_ in dots if d_["cx"] < axd], key=lambda q: -q["cy"])
    right = sorted([d_ for d_ in dots if d_["cx"] >= axd], key=lambda q: -q["cy"])
    nb = max(len(left), len(right))
    for side in (left, right):
        for i, q in enumerate(side):
            q["band"] = int(round(i * (nb - 1) / max(1, len(side) - 1))) if len(side) > 1 else 0

    # ---------------------------------------------------------- brain
    br = Image.open(os.path.join(art, "layers", "brain.png")).convert("RGBA")
    bem = Image.open(os.path.join(art, "layers", "brain_emissive.png")).convert("L")
    bw = int(205 * k)
    bh = int(bw * br.height / br.width)
    dw, dh = int(round(bw * s)), int(round(bh * s))
    PAD = 3
    b = np.asarray(br.resize((bw, bh), Image.LANCZOS)).astype(np.float32) / 255
    be = np.asarray(bem.resize((bw, bh), Image.LANCZOS)).astype(np.float32) / 255
    # depth: blur + desaturate toward the dome edge (radial in brain space)
    by_, bx_ = np.mgrid[0:bh, 0:bw].astype(np.float32)
    bc = (cx, cy + 82)
    dd = np.sqrt(((bx_ + bc[0] - bw / 2 - cx) / rx) ** 2 + ((by_ + bc[1] - bh / 2 - cy) / ry) ** 2)
    bm = np.clip((dd - 0.45) / 0.4, 0, 1)[..., None]
    col = b[..., :3]
    blur = cv2.GaussianBlur(col, (0, 0), 3)
    col = col * (1 - bm) + blur * bm
    g = col.mean(2, keepdims=True)
    col = col * (1 - bm * 0.5) + g * bm * 0.5
    def small(a):
        r = cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA)
        out = np.zeros((dh + 2 * PAD, dw + 2 * PAD) + r.shape[2:], np.float32)
        out[PAD:PAD + dh, PAD:PAD + dw] = r
        return out
    ba = b[..., 3] * 0.9
    save_rgb("brain_rgb", small(col * ba[..., None]))           # premultiplied colour
    save_gray("brain_gray", small(col.mean(2) * ba))
    save_gray("brain_a", small(ba))
    save_gray("brain_em", small(be * ba))
    bcd = P(*bc)

    # ---------------------------------------------------------- layout
    def cen(m, rf):
        """point along tentacle m at radius fraction rf (2048 space)"""
        rr = r0 + (rmax - r0) * rf
        sel = (m > 0.5) & (np.abs(rad - rr) < 12) & (A > 0.4)
        yy3, xx3 = np.where(sel)
        return P(float(xx3.mean()), float(yy3.mean()))
    lay = {
        "W": Wd, "H": Hd, "scale": s, "axis_x": P(cx, 0)[0],
        "dome": {"cx": P(cx, cy)[0], "cy": P(cx, cy)[1], "rx": rx * s, "ry": ry * s,
                 "seam_y": P(0, 345 * k)[1]},
        "brain": {"cx": bcd[0], "cy": bcd[1], "w": dw + 2 * PAD, "h": dh + 2 * PAD},
        "eyes": [list(P(cx - 92 * k, 469 * k)), list(P(cx + 92 * k, 469 * k))],
        "eye_r": 24 * k * s,
        "hub": list(P(*hub)), "rmax": rmax * s,
        "tentacles": tent_info,
        "ear": ear_tip(tod(A), P(cx, cy)[1], P(*hub)[1]),
        "dots": dots, "n_bands": nb,
        "lumk": LUMK,
    }
    with open(os.path.join(OUT, "layout.json"), "w") as f:
        json.dump(lay, f, indent=1)
    print("canvas", Wd, Hd, "scale %.4f" % s, "dots", len(dots), "bands", nb)


def ear_tip(ad, y_top, y_hub):
    """Right ear (fin) tip in display coords: the right-most opaque pixel between the
    dome centre and the eyes. The control bar grows out of it."""
    band = ad[int(y_top):int(y_hub) - 10]
    ys, xs = np.where(band > 0.35)
    i = int(np.argmax(xs))
    return [float(xs[i]), float(ys[i] + int(y_top))]




SEG_L, SEG_O = 16.0, 5.0
STATIC_S = 32.0          # the first 30 px of every tentacle never move -> baked into the body
ATLAS_W = 512


def geodesic(mask, seeds):
    """8-neighbour geodesic distance (px) inside mask from the seed pixels."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.where(mask)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.41421), (1, -1, 1.41421)):
        a = idx[0:H - dy, max(0, -dx):W - max(0, dx)]
        b = idx[dy:H, max(0, dx):W - max(0, -dx)]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok]); cols.append(b[ok]); wts.append(np.full(ok.sum(), w))
    n = len(ys)
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    d = dijkstra(g, directed=False, indices=idx[seeds], min_only=True)
    out = np.full((H, W), np.inf)
    out[ys, xs] = d
    return out


def build_segments(tmask, base_full, M_full, suck, hub, s, x0, y0, OX, MT, P, tod):
    """Round 3 tentacles: each tentacle is cut ACROSS its length into short segments (bands of
    geodesic distance from the root, so curled tentacles are cut correctly too). Segments are
    joints of one kinematic chain: they share their joint points, so the tentacle deforms as one
    continuous strip. Neighbouring segments cross-fade over SEG_O px (partition of unity)."""
    from scipy import ndimage
    hubd = P(*hub)
    sig = 0.42 / s
    pieces = []
    tinfo = []
    static_extra = np.zeros_like(tmask[0])
    for ti, m in enumerate(tmask):
        md = tod(m)
        if md.max() < 0.05:
            continue
        Hd, Wd = md.shape
        gy, gx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
        rd = np.hypot(gx + 0.5 - hubd[0], gy + 0.5 - hubd[1])
        grid = md > 0.10
        seeds = grid & (rd <= ROOT_B + 1.5)
        if grid.sum() < 300 or seeds.sum() == 0:
            print("tentacle", ti, "too small -> body")
            continue
        dist = geodesic(grid, seeds)
        ok = grid & np.isfinite(dist)
        _, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
        sf = cv2.GaussianBlur(dist[iy, ix].astype(np.float32), (0, 0), 1.2)
        smax = float(sf[ok & (md > 0.3)].max())
        nb = max(1, int(np.ceil(smax / SEG_L)))
        wts = md * ok
        pts = []
        for j in range(nb + 1):
            band = np.abs(sf - j * SEG_L) < 1.5
            w = wts * band
            if w.sum() < 0.5:
                break
            pts.append([float(((gx + 0.5) * w).sum() / w.sum()), float(((gy + 0.5) * w).sum() / w.sum())])
        while len(pts) < nb + 1:
            a_, b_ = np.array(pts[-2]), np.array(pts[-1])
            pts.append((b_ + (b_ - a_)).tolist())
        # 2048 work box + geodesic field sampled there
        yy, xx = np.where(m > 0.003)
        Y0, Y1, X0, X1 = yy.min() - 6, yy.max() + 7, xx.min() - 6, xx.max() + 7
        Y0, X0 = max(Y0, 0), max(X0, 0)
        GY, GX = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        mapx = (GX + 0.5 - x0) * s + OX - 0.5
        mapy = (GY + 0.5 - y0) * s + MT - 0.5
        s2k = cv2.remap(sf, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mb = m[Y0:Y1, X0:X1]
        a_b = cv2.GaussianBlur(mb, (0, 0), sig)
        b_b = cv2.GaussianBlur(base_full[Y0:Y1, X0:X1] * mb[..., None], (0, 0), sig)
        m_b = cv2.GaussianBlur(M_full[Y0:Y1, X0:X1] * mb, (0, 0), sig)
        s_b = cv2.GaussianBlur(suck[Y0:Y1, X0:X1] * mb, (0, 0), sig)
        segs = []
        j0 = int(STATIC_S // SEG_L)
        wdyn = np.clip((s2k - STATIC_S) / SEG_O + 0.5, 0, 1)
        static_extra[Y0:Y1, X0:X1] += mb * (1 - wdyn)
        for j in range(j0, nb):
            e0, e1 = j * SEG_L, (j + 1) * SEG_L

            def wfun(sv):
                w = np.ones_like(sv)
                if j > 0:
                    w *= np.clip((sv - e0) / SEG_O + 0.5, 0, 1)
                if j < nb - 1:
                    w *= np.clip((e1 - sv) / SEG_O + 0.5, 0, 1)
                return w
            wd = wfun(sf) * md
            sel = wd > 0.004
            if sel.sum() < 4:
                continue
            Pj, Pn = np.array(pts[j]), np.array(pts[j + 1])
            dv = Pn - Pj
            phi = -math.atan2(dv[0], dv[1])
            c, sn = math.cos(phi), math.sin(phi)
            dxs = gx[sel] + 0.5 - Pj[0]
            dys = gy[sel] + 0.5 - Pj[1]
            u = c * dxs + sn * dys
            v = -sn * dxs + c * dys
            lx, ly = int(np.floor(u.min())) - 2, int(np.floor(v.min())) - 2
            pw, ph = int(np.ceil(u.max())) + 3 - lx, int(np.ceil(v.max())) + 3 - ly
            w2 = wfun(s2k)
            st4 = np.dstack([b_b * w2[..., None], a_b * w2]).astype(np.float32)
            st2 = np.dstack([m_b * w2, s_b * w2]).astype(np.float32)
            Mx = np.array([[c / s, -sn / s, 0], [sn / s, c / s, 0]], np.float64)
            Mx[0, 2] = (c * (lx + 0.5) - sn * (ly + 0.5) + Pj[0] - OX) / s + x0 - 0.5 - X0
            Mx[1, 2] = (sn * (lx + 0.5) + c * (ly + 0.5) + Pj[1] - MT) / s + y0 - 0.5 - Y0
            fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            p4 = cv2.warpAffine(st4, Mx, (pw, ph), flags=fl, borderMode=cv2.BORDER_CONSTANT)
            p2 = cv2.warpAffine(st2, Mx, (pw, ph), flags=fl, borderMode=cv2.BORDER_CONSTANT)
            pa = p4[..., 3]
            if pa.max() < 0.02:
                continue
            nz = np.where(pa > 0.004)
            cy0, cy1, cx0, cx1 = nz[0].min(), nz[0].max() + 1, nz[1].min(), nz[1].max() + 1
            seg = {"j": j, "s0": e0, "phi": math.degrees(phi), "lx": lx + int(cx0), "ly": ly + int(cy0),
                   "w": int(cx1 - cx0), "h": int(cy1 - cy0)}
            segs.append(seg)
            pieces.append((seg, pa[cy0:cy1, cx0:cx1], p4[cy0:cy1, cx0:cx1, :3], p2[cy0:cy1, cx0:cx1, 0],
                           p2[cy0:cy1, cx0:cx1, 1]))
        tinfo.append({"t": ti, "smax": smax, "spine": pts, "segs": segs, "j0": j0})
    # shelf packing (solid-colour recolour -> any layout works)
    PAD = 2
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i][0]["h"])
    x = y = rowh = 0
    for i in order:
        seg = pieces[i][0]
        if x + seg["w"] > ATLAS_W:
            x, y, rowh = 0, y + rowh + PAD, 0
        seg["ax"], seg["ay"] = x, y
        x += seg["w"] + PAD
        rowh = max(rowh, seg["h"])
    H = y + rowh
    W = ATLAS_W
    atl_a = np.zeros((H, W), np.float32)
    atl_b = np.zeros((H, W, 3), np.float32)
    atl_m = np.zeros((H, W), np.float32)
    atl_s = np.zeros((H, W), np.float32)
    for seg, pa, pb, pmm, ps in pieces:
        sl = (slice(seg["ay"], seg["ay"] + seg["h"]), slice(seg["ax"], seg["ax"] + seg["w"]))
        atl_a[sl], atl_b[sl], atl_m[sl], atl_s[sl] = pa, pb, pmm, ps
    md = np.clip(np.minimum(atl_m, atl_a - atl_b.max(2)), 0, None)
    save_rgb("tent_base", atl_b)
    save_gray("tent_ba", np.clip(atl_a - md, 0, 1))
    save_gray("tent_m", md)
    save_gray("tent_s", np.clip(np.minimum(atl_s * 1.4, atl_a), 0, 1))
    npx = sum(p[0]["w"] * p[0]["h"] for p in pieces)
    print("tentacle atlas", W, "x", H, "segments", len(pieces), "piece px", npx)
    return {"hub": list(hubd), "seg_l": SEG_L, "static_s": STATIC_S, "tents": tinfo}, static_extra


if __name__ == "__main__":
    main()
