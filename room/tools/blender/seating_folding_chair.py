"""folding_chair_grey.glb / folding_chair_white.glb - tubular-steel folding chair with moulded plastic seat + back.
Run: blender -b --python seating_folding_chair.py            (builds both variants)
     blender -b --python seating_folding_chair.py -- grey    (one variant)
Side view is the classic X: tube A (floor-rear -> seat front) crosses tube B (floor-front -> seat rear -> backrest post).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seating_common import *

VARIANTS = {
    # name: (plastic colour, plastic rough, frame colour, frame metallic, frame rough)
    "grey": ((0.30, 0.31, 0.33), 0.50, (0.07, 0.07, 0.075), 0.85, 0.38),
    "white": ((0.90, 0.90, 0.88), 0.42, (0.80, 0.81, 0.83), 1.0, 0.22),
}


def build(variant):
    reset()
    pc, pr, fc, fm, fr = VARIANTS[variant]
    plastic = make_material("plastic_" + variant, pc, pr)
    frame = make_material("steel_" + variant, fc, fr, fm)
    rubber = make_material("rubber_black", (0.02, 0.02, 0.02), 0.85)
    R = 0.0105
    parts = []
    SEAT_Z = 0.407          # centre of the seat-support rails
    for sx in (-1, 1):
        xa, xb = 0.196, 0.226
        # tube A: floor rear -> seat front rail
        parts.append(tube("A", frame, [(sx * xa, -0.268, 0.0), (sx * xa, 0.186, SEAT_Z)], [R, R], n=14,
                          flat_start=True, bevel=0.002))
        # tube B: floor front -> seat rear -> backrest post (rakes back)
        parts.append(tube("B", frame, [(sx * xb, 0.272, 0.0), (sx * xb, -0.186, SEAT_Z),
                                       (sx * xb, -0.222, 0.62), (sx * xb, -0.238, 0.835), (sx * xb, -0.2385, 0.847)],
                          [R, R, R, R, R * 0.6], n=14, flat_start=True, bevel=0.002))
        # side seat rail joining front/rear rails
        parts.append(tube("rail", frame, [(sx * (xa + xb) / 2, 0.186, SEAT_Z), (sx * (xa + xb) / 2, -0.186, SEAT_Z)],
                          [R * 0.85, R * 0.85], n=12, bevel=0.0015))
        # pivot bolt + washer head where the X crosses
        zc = 0.262
        parts.append(tube("bolt", frame, [(sx * 0.178, 0.0, zc), (sx * 0.244, 0.0, zc)], [0.0065, 0.0065], n=12, bevel=0.001))
        parts.append(tube("bolthead", frame, [(sx * 0.244, 0.0, zc), (sx * 0.2535, 0.0, zc)], [0.0125, 0.0125], n=14, bevel=0.0015))
        # rubber feet
        for yy, xx in ((-0.268, xa), (0.272, xb)):
            parts.append(tube("foot", rubber, [(sx * xx, yy, 0.0), (sx * xx, yy, 0.022)], [0.0135, 0.0135],
                              n=14, flat_start=True, bevel=0.003))
    # cross bars
    parts.append(tube("bar_seat_front", frame, [(-0.196, 0.186, SEAT_Z), (0.196, 0.186, SEAT_Z)], [R, R], n=12, bevel=0.0015))
    parts.append(tube("bar_seat_rear", frame, [(-0.226, -0.186, SEAT_Z), (0.226, -0.186, SEAT_Z)], [R, R], n=12, bevel=0.0015))
    parts.append(tube("bar_low_front", frame, [(-0.226, 0.158, 0.10), (0.226, 0.158, 0.10)], [R * 0.9, R * 0.9], n=12, bevel=0.0015))
    parts.append(tube("bar_low_rear", frame, [(-0.196, -0.154, 0.10), (0.196, -0.154, 0.10)], [R * 0.9, R * 0.9], n=12, bevel=0.0015))
    # moulded seat (top peak 0.45) - slight dish, rolled edge
    parts.append(pad("seat", plastic, 0.445, 0.425, (0, 0.0, 0.4415), rot=(math.pi / 2, 0, 0),
                     n=3.6, nu=22, nv=22, thick=0.020, bulge=0.008, bevel=0.0075))
    # backrest, curved around the sitter, raked to follow the posts
    rake = math.atan2(0.016, 0.215)
    parts.append(pad("back", plastic, 0.455, 0.235, (0, -0.2085, 0.722), rot=(rake, 0, 0),
                     n=3.4, nu=26, nv=14, thick=0.018, bulge=0.005, arc=0.42, bevel=0.007))
    finalize(parts, "folding_chair_" + variant)


args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
for v in (args or list(VARIANTS)):
    build(v)
