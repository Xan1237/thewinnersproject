#!/usr/bin/env python3
"""Procedural seamless PBR textures for the hackathon stage room (Godot 4).

Outputs to assets/tex/:
  wood_floor_{albedo,normal,rough}.png   1024^2  tile = 1.4 m x 1.4 m
  brick_{albedo,normal,rough}.png        1024^2  tile = 1.0 m x 1.0 m (13 courses)
  wallpaper_{albedo,normal,rough}.png    1024^2  tile = 0.8 m x 0.8 m
  concrete_{albedo,normal,rough}.png     1024^2  tile = 2.0 m x 2.0 m
  rug_albedo.jpg / rug_normal.png        cropped + cleaned from ROOMFLOOR.jpg (not tiled)
Normal maps are OpenGL convention (+Y up), as Godot expects.

All tile-able maps are built from periodic (FFT) noise and integer-period patterns,
so they wrap perfectly.  Usage:  python tools/gen_textures.py [set ...]
Requires numpy, pillow, scipy, opencv-python(-headless).
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "tex")
RUG_SRC = "/Users/sukhov/Downloads/Telegram Desktop/ROOMFLOOR.jpg"
S = 1024


# ----------------------------------------------------------------- helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def grid(h=S, w=S):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return y + 0.5, x + 0.5


def fnoise(seed, beta=2.0, stretch=1.0, kmin=0.0, kmax=None, h=S, w=S):
    """Periodic noise, spectrum ~ r^-beta/2 amplitude. stretch<1 elongates features along x (x-frequencies are suppressed), stretch>1 along y.
    kmin/kmax in cycles/tile. Normalised to zero mean, unit std."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    r = np.sqrt((fx / stretch) ** 2 + fy ** 2)
    amp = 1.0 / np.maximum(r, 1.0) ** (beta / 2.0)
    amp[0, 0] = 0
    if kmax:
        amp = amp * np.exp(-(r / kmax) ** 2)
    if kmin:
        amp = amp * (1 - np.exp(-(r / kmin) ** 2))
    n = np.fft.ifft2(f * amp).real.astype(np.float32)
    n -= n.mean()
    return n / (n.std() + 1e-9)


def wrap_blur(a, sigma):
    return gaussian_filter(a, sigma, mode="wrap")


def sparse_dots(seed, count, sigma, h=S, w=S):
    """Wrapped gaussian dots with random amplitudes, peak ~1 for amp 1."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), np.float32)
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    np.add.at(a, (ys, xs), rng.uniform(0.3, 1.0, count).astype(np.float32))
    a = wrap_blur(a, sigma)
    return a * (2 * np.pi * sigma ** 2)


def normal_from_height(hgt, strength=1.0):
    """OpenGL normal (+Y up, image rows go down) from a periodic height map in pixel units."""
    gx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5 * strength
    gr = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5 * strength  # d/d(row)
    nx, ny, nz = -gx, gr, np.ones_like(gx)  # ny = -dh/dy_up = +dh/drow
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / ln, ny / ln, nz / ln], -1)
    return n


def save_rgb(path, arr01):
    Image.fromarray((np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)


def save_normal(path, n):
    save_rgb(path, n * 0.5 + 0.5)


def save_gray(path, a01):
    Image.fromarray((np.clip(a01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)


def rgb(r, g, b):
    return np.array([r, g, b], np.float32) / 255.0


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


# ----------------------------------------------------------------- wood floor
def gen_wood():
    rng = np.random.default_rng(11)
    Y, X = grid()
    NR = 10                      # planks across: 1.4 m / 0.14 m
    PW = S / NR
    row = np.floor(Y / PW).astype(int)
    v = (Y - row * PW) / PW
    # 1.4 m long planks (one joint per row, staggered) - tile length = plank length
    off = (np.arange(NR) * 0.37 + rng.uniform(-0.05, 0.05, NR)) % 1.0
    sx = np.round(off * S)
    xr = (X - sx[row]) % S
    d = np.minimum(np.minimum(v, 1 - v) * PW, np.minimum(xr, S - xr))
    gap_prof = smoothstep(0.0, 3.2, d)                       # 0 in the gap, 1 on plank
    gap_dark = 0.12 + 0.88 * smoothstep(0.5, 2.4, d)

    W1 = fnoise(101, 2.4, 5, kmax=1.4)
    W2 = fnoise(106, 2.0, 3, kmin=4, kmax=10)
    F2 = fnoise(102, 0.8, 0.05, kmin=10, kmax=110)       # fine fibres
    F2b = fnoise(107, 0.4, 0.08, kmin=40, kmax=230)
    F3 = fnoise(103, 0.0, 0.25, kmin=60, kmax=380)   # pore dashes
    F4 = fnoise(104, 2.2, 3, kmax=6)
    LF = fnoise(105, 3.0, 1, kmax=4)

    base_l, base_d = rgb(178, 122, 74), rgb(112, 68, 38)
    col = np.zeros((S, S, 3), np.float32)
    rough = np.zeros((S, S), np.float32)
    hgt = np.zeros((S, S), np.float32)
    for r in range(NR):
        m = row == r
        dy, dx = rng.integers(0, S, 2)
        sh = lambda F: np.roll(F, (dy, dx), (0, 1))
        w1, f2, f3, f4 = sh(W1), sh(F2), sh(F3), sh(F4)
        w2 = sh(W2)
        F2roll, F2broll = sh(F2), sh(F2b)
        K = rng.uniform(2.5, 6.0)
        K2 = rng.uniform(26, 44)
        A = rng.uniform(0.5, 1.1)
        ph = rng.uniform(0, 1)
        rings = 0.5 + 0.5 * np.sin(2 * np.pi * (v * K + A * w1 * 0.45 + 0.06 * w2 + ph))
        rings = rings ** 1.4
        # domain-warped streak fields -> irregular fibres that follow the grain flow
        warp = 5.0 * A * w1 + 2.0 * w2
        yy = (Y - 0.5 + warp).astype(np.float32)
        xx = (X - 0.5).astype(np.float32)
        wp = lambda F: map_coordinates(F, [yy, xx], order=1, mode="grid-wrap")
        f2 = wp(F2roll)
        f2b = wp(F2broll)
        lines = np.clip(0.5 + 0.28 * f2 + 0.2 * f2b, 0, 1)
        tone = rng.uniform(0.25, 0.65)                         # plank base brightness
        hue = rng.normal(0, 1, 3).astype(np.float32) * np.array([0.03, 0.02, 0.015], np.float32)
        base = lerp(base_d, base_l, tone) * (1 + hue)
        t = np.clip(0.42 + 0.26 * (rings - 0.5) * 1.6 + 0.45 * (lines - 0.5) + 0.07 * f4, 0, 1)
        light = base * 1.14
        dark = base * 0.72
        c = light[None, None, :] * (1 - t[..., None]) + dark[None, None, :] * t[..., None]
        pore = np.clip((f3 - 1.7) * 0.5, 0, 1)
        c = c * (1 - 0.38 * pore[..., None])
        c = c * (1 + 0.035 * LF[..., None])
        col[m] = c[m]
        rr = rng.uniform(0.42, 0.49) + 0.025 * f2 - 0.03 * (rings - 0.5) + 0.12 * pore
        rough[m] = rr[m]
        hh = 0.5 * (rings - 0.5) + 0.35 * f2 + 0.3 * f4 - 1.1 * pore + rng.uniform(-0.25, 0.25)
        hgt[m] = hh[m]
    lum = col.mean(-1, keepdims=True)
    col = lum + (col - lum) * 0.88          # slightly less orange
    col = col * gap_dark[..., None]
    rough = lerp(0.85, rough, gap_dark)
    hgt = hgt * 0.7 + 2.2 * gap_prof
    n = normal_from_height(hgt, 1.0)
    save_rgb(out("wood_floor_albedo.png"), col)
    save_normal(out("wood_floor_normal.png"), n)
    save_gray(out("wood_floor_rough.png"), rough)


# ----------------------------------------------------------------- brick
def gen_brick():
    rng = np.random.default_rng(22)
    Y, X = grid()
    NC, NB = 13, 4
    CH, BW = S / NC, S / NB
    MORT = 10.0
    # course offsets (fraction of a brick). 13 is odd, so the last course gets 0.25 to keep
    # the bond running across the vertical wrap.
    offs = np.array([0.0 if i % 2 == 0 else 0.5 for i in range(NC)], np.float32)
    offs[12] = 0.25
    c = np.floor(Y / CH).astype(int)
    xs = X - offs[c] * BW
    b = np.floor(xs / BW).astype(int)
    bm = b % NB
    u = xs - b * BW - BW / 2
    vv = Y - c * CH - CH / 2
    rj = rng.uniform(-1, 1, (NC, NB, 6)).astype(np.float32)   # per-brick randoms
    R = lambda k: rj[c, bm, k]
    # brick signed distance (rounded box), with wobbly hand-made edges
    hx = BW / 2 - MORT / 2 - 1.0 + 1.2 * R(0)
    hy = CH / 2 - MORT / 2 - 0.5 + 0.8 * R(1)
    cu, cv = u - 1.5 * R(2), vv - 1.0 * R(3)
    rad = 5.0
    qx, qy = np.abs(cu) - (hx - rad), np.abs(cv) - (hy - rad)
    sdf = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - rad
    edge_n = fnoise(201, 1.6, 1, kmin=20, kmax=130)
    sdf = sdf + 1.3 * edge_n
    inside = smoothstep(1.0, -1.0, sdf)            # 1 on brick
    bevel = smoothstep(0.0, -5.5, sdf)             # 0 at edge, 1 inside

    blotch = fnoise(202, 2.6, 1.0, kmax=22)
    blotch2 = fnoise(203, 2.0, 2.0, kmin=6, kmax=60)
    speck = fnoise(204, 0.0, 1.0, kmin=150, kmax=420)
    pits = np.clip(sparse_dots(205, 5000, 1.1) * 1.2, 0, 1)

    pal = np.stack([rgb(176, 66, 42), rgb(196, 98, 56), rgb(150, 54, 38), rgb(206, 120, 72), rgb(128, 48, 38)])
    wts = rng.dirichlet(np.ones(5) * 0.7, (NC, NB)).astype(np.float32)
    bcol = np.einsum("cbk,kj->cbj", wts, pal)
    bc = bcol[c, bm]
    bright = (1 + 0.08 * R(4))[..., None]
    brick = bc * bright
    brick = brick * (1 + 0.10 * blotch[..., None] + 0.07 * blotch2[..., None] + 0.04 * speck[..., None])
    brick = brick * (1 - 0.35 * pits[..., None])
    # burn/dark edges + occasional lighter sandy face
    brick = brick * (0.80 + 0.20 * bevel[..., None])
    brick = lerp(brick, brick * np.array([1.12, 1.04, 0.9], np.float32), (np.clip(blotch2 * 0.5, 0, 1))[..., None] * 0.35)

    sand = fnoise(206, 0.5, 1.0, kmin=80)
    mort = rgb(176, 172, 164)[None, None, :] * (1 + 0.08 * sand[..., None] + 0.05 * blotch[..., None])
    ao = smoothstep(-1.0, 10.0, sdf)             # darker near brick edges (recess)
    mort = mort * (0.70 + 0.30 * ao[..., None])
    bl_ = brick.mean(-1, keepdims=True)
    brick = (bl_ + (brick - bl_) * 0.88) * 0.93      # dusty, slightly darker
    col = lerp(mort, brick, inside[..., None])

    hgt = 6.5 * bevel + 0.9 * blotch2 * 0.5 + 0.35 * speck - 1.3 * pits
    hgt = hgt * inside + (1 - inside) * (0.45 * sand + 0.3 * blotch) + 0.0
    hgt = hgt + 0.0
    hgt = np.where(inside > 0.5, hgt, hgt - 0.0)
    n = normal_from_height(wrap_blur(hgt, 0.6), 1.0)
    rough = lerp(0.97 + 0.02 * sand, 0.84 + 0.05 * blotch2 + 0.04 * speck + 0.08 * pits, inside)
    save_rgb(out("brick_albedo.png"), col)
    save_normal(out("brick_normal.png"), n)
    save_gray(out("brick_rough.png"), rough)


# ----------------------------------------------------------------- wallpaper
def _leaf(x, y, L, w, aa, p=0.75):
    """Pointed leaf along y of half-length L and half-width w. returns soft mask."""
    t = np.clip(1 - np.abs(y) / L, 0, 1)
    val = w * t ** p - np.abs(x)
    val = np.where(np.abs(y) > L, -1.0, val)
    return smoothstep(-aa, aa, val)


def _ring(x, y, cx, cy, r, th, aa):
    dd = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
    return smoothstep(th + aa, th - aa, dd), np.arctan2(y - cy, x - cx)


def gen_wallpaper():
    rng = np.random.default_rng(33)
    Y, X = grid()
    SC = 2                      # pattern repeats 2x2 inside the 0.8 m tile -> 0.4 m repeat
    px = SC * X / S
    py = SC * Y / S
    aa = SC * 1.3 / S
    TH = 1.8                    # line thickness scale vs. a 1-repeat tile
    wrap = lambda a: a - np.floor(a + 0.5)

    def local(cx, cy):
        return wrap(px - cx), wrap(py - cy)

    mask = np.zeros((S, S), np.float32)
    line = np.zeros((S, S), np.float32)

    # beaded diamond trellis: lines x+y=0 and x-y=0.5 (mod 1) -> cross at the small motif spots
    for sgn, c0 in ((1, 0.0), (-1, 0.5)):
        s = (px + sgn * py - c0)
        dist = np.abs(wrap(s)) / np.sqrt(2)
        along = (px - sgn * py)
        beads = 0.5 + 0.5 * np.cos(2 * np.pi * along * 22)
        th = TH * (0.0017 + 0.0023 * beads ** 2)
        line = np.maximum(line, smoothstep(th + aa, th - aa, dist))
    # big medallions at (.25,.25),(.75,.75)
    for cx, cy in ((0.25, 0.25), (0.75, 0.75)):
        x, y = local(cx, cy)
        rr = np.sqrt(x * x + y * y)
        ang = np.arctan2(y, x)
        m = np.zeros_like(mask)
        m = np.maximum(m, _leaf(x, y, 0.23, 0.060, aa, 0.55))          # vertical leaf
        for k in range(4):                                              # diagonal petals
            a = np.pi / 4 + k * np.pi / 2
            ca, sa = np.cos(a), np.sin(a)
            xr_, yr_ = x * sa - y * ca, x * ca + y * sa                 # yr along axis
            pet = _leaf(xr_, yr_ - 0.09, 0.09, 0.05, aa, 0.5)
            m = np.maximum(m, pet)
        # scroll rings in the diagonals between petals
        for sx_ in (-1, 1):
            for sy_ in (-1, 1):
                r1, an = _ring(x, y, sx_ * 0.14, sy_ * 0.07, 0.04, 0.0075 * TH, aa)
                m = np.maximum(m, r1 * (np.abs(wrap((an - (np.pi / 2) * (1 if sx_ * sy_ > 0 else -1)) / (2 * np.pi))) < 0.38))
        # crown ring of dots
        for k in range(16):
            a = k * 2 * np.pi / 16
            dd = np.sqrt((x - 0.165 * np.cos(a)) ** 2 + (y - 0.165 * np.sin(a)) ** 2)
            m = np.maximum(m, smoothstep(0.011 + aa, 0.011 - aa, dd) * (np.abs(y) < 0.2))
        m = np.maximum(m, smoothstep(0.016 + aa, 0.016 - aa, rr))      # centre boss
        # thin ring
        r2, _ = _ring(x, y, 0, 0, 0.065, 0.0035 * TH, aa)
        m = np.maximum(m, r2)
        mask = np.maximum(mask, m)
    # small rosettes at (.75,.25),(.25,.75)
    for cx, cy in ((0.75, 0.25), (0.25, 0.75)):
        x, y = local(cx, cy)
        rr = np.sqrt(x * x + y * y)
        m = smoothstep(0.012 + aa, 0.012 - aa, rr)
        for k in range(8):
            a = k * np.pi / 4
            ca, sa = np.cos(a), np.sin(a)
            xr_, yr_ = x * sa - y * ca, x * ca + y * sa
            m = np.maximum(m, _leaf(xr_, yr_ - 0.034, 0.034, 0.016, aa, 0.55))
        r2, _ = _ring(x, y, 0, 0, 0.056, 0.003 * TH, aa)
        m = np.maximum(m, r2)
        mask = np.maximum(mask, m)
    mask = np.maximum(mask, line * 0.9)
    mask = wrap_blur(mask, 0.6)

    # woven structure: ground = horizontal-weft look, motif = satin (vertical) look
    P = 4  # px per thread -> 256 threads over 0.8 m
    thr_x = fnoise(301, 0.0, 1.0, kmin=60)   # per-thread irregularity
    jx = 0.5 + 0.5 * np.cos(2 * np.pi * X / P)
    jy = 0.5 + 0.5 * np.cos(2 * np.pi * Y / P)
    ground_w = 0.65 * jy * (1 + 0.15 * fnoise(302, 0.0, 40, kmin=30)) + 0.35 * jx * jy + 0.25 * fnoise(303, 0.3, 1, kmin=100)
    sat_w = 0.7 * jx * (1 + 0.15 * fnoise(304, 0.0, 1, kmin=0) * 0) + 0.2 * jy * jx + 0.25 * fnoise(305, 0.3, 1, kmin=100)
    weave = lerp(ground_w, sat_w, mask)
    fibre = fnoise(306, 1.0, 5, kmin=90)
    cloud = fnoise(307, 3.0, 1, kmax=6)

    ground = rgb(228, 213, 184)
    motif = rgb(205, 186, 150)
    edge = np.abs(mask - wrap_blur(mask, 2.0))
    col = lerp(ground, motif, mask[..., None])
    col = col * (1 + 0.025 * cloud[..., None] + 0.035 * (weave[..., None] - 0.5) + 0.02 * fibre[..., None])
    col = col * (1 - 0.7 * edge[..., None] * np.array([0.13, 0.19, 0.28], np.float32))
    hgt = 1.1 * weave + 0.35 * fibre + 1.4 * mask * 0.5 + 0.3 * wrap_blur(mask, 1.2)
    n = normal_from_height(hgt, 0.9)
    rough = lerp(0.82, 0.58, mask) + 0.04 * fibre + 0.03 * (weave - 0.5)
    save_rgb(out("wallpaper_albedo.png"), col)
    save_normal(out("wallpaper_normal.png"), n)
    save_gray(out("wallpaper_rough.png"), rough)


# ----------------------------------------------------------------- concrete
def gen_concrete():
    rng = np.random.default_rng(44)
    Y, X = grid()
    PPM = S / 2.0                     # px per metre (tile = 2 m)
    big = fnoise(401, 2.8, 1, kmax=5)
    mid = fnoise(402, 2.2, 1, kmin=4, kmax=40)
    fine = fnoise(403, 1.0, 1, kmin=40)
    vstr = fnoise(404, 2.4, 0.12, kmax=14)       # vertical run-off streaks (stretch along y)
    vstr = fnoise(404, 2.4, 1, kmax=14)
    # stretch along y: swap axes
    vstr = fnoise(404, 2.2, 0.08, kmax=24).T.copy()
    stain = np.clip((fnoise(405, 3.0, 1, kmax=6) - 0.5) * 0.9, 0, 1)
    stain2 = np.clip((fnoise(406, 3.2, 1, kmax=7) - 0.3) * 0.8, 0, 1)

    p1 = sparse_dots(410, 14000, 0.9)
    p2 = sparse_dots(411, 3000, 1.8)
    p3 = sparse_dots(412, 160, 3.4)
    pores = np.clip(p1 * 0.55 + p2 * 0.8 + p3 * 1.1, 0, 1)

    # formwork: 0.2 m boards running along x, plywood panel seams every 1 m, tie holes
    board = S / 10.0
    bi = np.floor(Y / board).astype(int)
    bv = Y - bi * board
    bd = np.minimum(bv, board - bv)                     # dist to board seam (px)
    brng = np.random.default_rng(45)
    b_off = brng.uniform(-1, 1, 10).astype(np.float32)
    b_tone = brng.uniform(-1, 1, 10).astype(np.float32)
    boardgrain = fnoise(407, 1.6, 0.03, kmin=8)
    bgr = np.zeros_like(boardgrain)
    for i in range(10):
        dy, dx = brng.integers(0, S, 2)
        mk = bi == i
        bgr[mk] = np.roll(boardgrain, (dy, dx), (0, 1))[mk]
    seam_b = smoothstep(2.2, 0.0, bd)
    panel = S / 2.0
    pdx = np.abs(((X + panel / 2) % panel) - panel / 2)       # distance to vertical panel seam at x=0,1 m
    pdy = np.abs(((Y + panel / 2) % panel) - panel / 2)
    seam_p = np.maximum(smoothstep(3.0, 0.4, pdx), smoothstep(3.0, 0.4, pdy))
    fin = np.maximum(smoothstep(8.0, 3.0, pdx) * smoothstep(1.0, 3.0, pdx),
                     smoothstep(8.0, 3.0, pdy) * smoothstep(1.0, 3.0, pdy))
    # tie holes at (0.25,0.5) and (0.75,0.5) of every 1 m panel
    tie = np.zeros((S, S), np.float32)
    rim = np.zeros((S, S), np.float32)
    for px_ in (0.25, 0.75, 1.25, 1.75):
        for py_ in (0.5, 1.5):
            dx = np.abs(X - px_ * PPM)
            dx = np.minimum(dx, S - dx)
            dy = np.abs(Y - py_ * PPM)
            dy = np.minimum(dy, S - dy)
            r = np.sqrt(dx * dx + dy * dy)
            tie = np.maximum(tie, smoothstep(7.5, 5.5, r))
            rim = np.maximum(rim, smoothstep(5.0, 10.0, r) * smoothstep(14.0, 10.0, r))

    base = rgb(142, 138, 131)
    tone = 1 + 0.05 * big + 0.05 * mid + 0.04 * fine
    col = base[None, None, :] * tone[..., None]
    # stains (warm brownish + dark damp) and vertical streaks
    col = lerp(col, col * np.array([0.84, 0.78, 0.70], np.float32), (stain * 0.5)[..., None])
    col = lerp(col, col * np.array([0.88, 0.89, 0.90], np.float32), (stain2 * 0.45)[..., None])
    streak = np.clip(vstr * 0.5 + 0.1, 0, 1) * np.clip(big * 0.7 + 0.6, 0, 1)
    col = col * (1 - 0.06 * streak[..., None])
    col = col * (1 + 0.025 * b_tone[bi][..., None] + 0.03 * bgr[..., None])
    col = col * (1 - 0.55 * pores[..., None] * 0.7)
    col = col * (1 - 0.22 * seam_b[..., None]) * (1 - 0.5 * seam_p[..., None])
    col = col * (1 - 0.72 * tie[..., None]) * (1 - 0.08 * rim[..., None])
    col = col * (1 + 0.05 * fin[..., None])

    hgt = (0.9 * mid + 0.35 * fine + 0.9 * big
           - 1.6 * pores - 1.1 * seam_b - 1.8 * seam_p + 1.2 * fin
           + 0.35 * b_off[bi] + 0.22 * bgr - 6.0 * tie + 0.8 * rim)
    n = normal_from_height(wrap_blur(hgt, 0.5), 1.0)
    rough = 0.88 + 0.04 * mid + 0.03 * fine - 0.06 * stain2 + 0.05 * pores - 0.03 * seam_b
    rough = np.clip(rough, 0.6, 1.0)
    save_rgb(out("concrete_albedo.png"), col)
    save_normal(out("concrete_normal.png"), n)
    save_gray(out("concrete_rough.png"), rough)


# ----------------------------------------------------------------- rug
def gen_rug():
    import cv2
    img = cv2.imread(RUG_SRC)
    # Clean, unoccluded part of the rug: right of the cable, below the feet, above the stage edge.
    x0, x1, y0, y1 = 215, 1038, 424, 716
    crop = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    crop = crop[..., ::-1].copy()  # RGB
    # Perspective: horizontals are ~parallel in the photo (no visible keystone); only vertical
    # foreshortening, so compensate with a mild vertical stretch instead of a full homography.
    VSTRETCH = 1.35
    W_OUT = 1648
    scale = W_OUT / crop.shape[1]
    H_OUT = int(round(crop.shape[0] * VSTRETCH * scale))
    big = cv2.resize(crop, (W_OUT, H_OUT), interpolation=cv2.INTER_CUBIC)
    # --- remove coloured lighting: divide by heavy low-pass per channel (keeps the pattern),
    #     then map to a neutral warm cream/grey tone.
    low = cv2.GaussianBlur(big, (0, 0), 90)
    lum_low = low.mean(-1, keepdims=True)
    # chroma flatten: strength 0.9 on colour, 0.85 on brightness
    flat = big / np.maximum(low, 1e-3)
    flat = flat * lum_low.mean()
    lum = flat.mean(-1, keepdims=True)
    # desaturate a little (remaining blue/pink casts), keep faint blue-grey motif colours
    flat = lum + (flat - lum) * 0.75
    # tone: warm cream/grey target mean
    target = np.array([0.70, 0.60, 0.49], np.float32)
    m = flat.reshape(-1, 3).mean(0)
    flat = flat * (target / m)
    # restore a bit of local contrast
    bl = cv2.GaussianBlur(flat, (0, 0), 3)
    flat = np.clip(flat + (flat - bl) * 0.6, 0, 1)
    # mild gamma/contrast curve
    flat = np.clip((flat - 0.5) * 1.12 + 0.5, 0, 1)
    save = (flat * 255 + 0.5).astype(np.uint8)
    Image.fromarray(save).save(out("rug_albedo.jpg"), quality=93, subsampling=0)

    # --- woven normal map (subtle) from procedural weave + design luminance
    h, w = flat.shape[:2]
    Y, X = grid(h, w)
    P = 4.0
    warp = np.cos(2 * np.pi * X / P)           # warp threads (vertical ridges)
    weft = np.cos(2 * np.pi * Y / (P * 1.25))
    weave = 0.6 * warp * (0.6 + 0.4 * weft) + 0.25 * weft
    irreg = fnoise(501, 0.4, 1.0, kmin=int(w * 0.06), h=h, w=w)
    fibre = fnoise(502, 0.8, 1.0, kmin=int(w * 0.05), h=h, w=w)
    lumg = cv2.cvtColor(save, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hp = lumg - cv2.GaussianBlur(lumg, (0, 0), 6)
    hgt = 0.55 * weave + 0.2 * irreg + 0.15 * fibre + 5.0 * hp
    n = normal_from_height(hgt, 0.55)
    save_normal(out("rug_normal.png"), n)


SETS = {"wood": gen_wood, "brick": gen_brick, "wallpaper": gen_wallpaper,
        "concrete": gen_concrete, "rug": gen_rug}

if __name__ == "__main__":
    which = sys.argv[1:] or list(SETS)
    for k in which:
        print("generating", k, flush=True)
        SETS[k]()
    print("done ->", OUT)
