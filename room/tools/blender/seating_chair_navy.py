"""chair_navy.glb - armless event chair: navy padded seat + rounded curved back, tapered splayed walnut legs.
Run: blender -b --python seating_chair_navy.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seating_common import *

reset()
NAVY = (0.055, 0.09, 0.24)
WALNUT = (0.30, 0.16, 0.085)
fab = fabric_material("fabric_navy", NAVY, seed=11)
wood = wood_material("wood_walnut", WALNUT, seed=5)

parts = []
# --- wooden seat frame (apron) under the cushion
parts.append(bevel_box("apron", wood, (0, 0, 0.382), (0.385, 0.365, 0.048), bevel=0.008))
# --- seat cushion: top peak = 0.46 m
parts.append(pad("seat", fab, 0.455, 0.435, (0, 0.0, 0.447), rot=(math.pi / 2, 0, 0),
                 n=3.2, thick=0.052, bulge=0.013, bevel=0.018))

# --- legs (tapered, splayed, flat feet)
for sx in (-1, 1):
    # front
    parts.append(tube("legF", wood,
                      [(sx * 0.198, 0.222, 0.0), (sx * 0.172, 0.158, 0.38)], [0.0125, 0.0215],
                      n=16, flat_start=True, bevel=0.0025, uv_scale=0.25))
    # rear: floor -> seat -> rakes back up behind the back panel
    parts.append(tube("legR", wood,
                      [(sx * 0.205, -0.250, 0.0), (sx * 0.172, -0.158, 0.38),
                       (sx * 0.168, -0.180, 0.56), (sx * 0.168, -0.222, 0.675)],
                      [0.0125, 0.0215, 0.0195, 0.0155],
                      n=16, flat_start=True, bevel=0.0025, uv_scale=0.25))

# --- back panel: curved, egg-shaped, tilted back ~9 degrees, wraps toward sitter
tilt = math.radians(9)
parts.append(pad("back", fab, 0.40, 0.335, (0, -0.205, 0.690), rot=(tilt, 0, 0),
                 n=2.7, nu=28, nv=28, thick=0.05, bulge=0.012, arc=0.95, egg=0.05, bevel=0.017))

finalize(parts, "chair_navy")
