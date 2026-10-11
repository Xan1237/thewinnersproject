"""armchair_blue.glb - wide blue fabric tub armchair, rounded padded back, padded arms, tapered walnut legs.
Run: blender -b --python seating_armchair_blue.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seating_common import *

reset()
BLUE = (0.16, 0.20, 0.50)
WALNUT = (0.30, 0.16, 0.085)
fab = fabric_material("fabric_blue", BLUE, seed=21)
wood = wood_material("wood_walnut", WALNUT, seed=7)

parts = []
# upholstered base under the cushion
parts.append(superblock("base", fab, (0, 0.02, 0.31), (0.62, 0.54, 0.11), n=5.0, cuts=7))
# seat cushion (peak = 0.45)
parts.append(pad("seat", fab, 0.46, 0.50, (0, 0.05, 0.432), rot=(math.pi / 2, 0, 0),
                 n=3.4, nu=18, nv=18, thick=0.10, bulge=0.018, bevel=0.035))
# padded arms
for sx in (-1, 1):
    parts.append(superblock("arm", fab, (sx * 0.290, 0.0, 0.475), (0.10, 0.46, 0.27), n=3.2, cuts=7))
# tub back: wide wrap-around curved panel, egg-shaped, reclined ~10 deg
parts.append(pad("back", fab, 0.68, 0.70, (0, -0.205, 0.750), rot=(math.radians(10), 0, 0),
                 n=2.6, nu=28, nv=24, thick=0.10, bulge=0.028, arc=1.5, egg=0.05, bevel=0.04))
# legs
for sx in (-1, 1):
    for sy in (-1, 1):
        parts.append(tube("leg", wood,
                          [(sx * 0.300 + sx * 0.0, sy * 0.265 + 0.03 + sy * 0.0, 0.0),
                           (sx * 0.255, sy * 0.205 + 0.03, 0.30)],
                          [0.0135, 0.0265], n=16, flat_start=True, bevel=0.003, uv_scale=0.25))
finalize(parts, "armchair_blue")
