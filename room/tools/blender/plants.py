"""Generate potted rubber-fig plants as GLB.

Run:  blender -b --python tools/blender/plants.py
Outputs assets/models/plant_tall.glb and plant_medium.glb (Y-up, metres,
origin = floor centre under the pot).
"""
import math
import os
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

OUT_DIR = "/Users/sukhov/Projects/thewinnersproject/assets/models"
GOLDEN = math.radians(137.508)


# ---------------------------------------------------------------- materials
def make_mat(name, color, rough, spec=0.5, metallic=0.0, double_sided=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    for k in ("Specular IOR Level", "Specular"):
        if k in b.inputs:
            b.inputs[k].default_value = spec
            break
    m.use_backface_culling = not double_sided  # False -> glTF doubleSided
    return m


def build_materials():
    return {
        "leaf_dark": make_mat("leaf_dark_green", (0.012, 0.07, 0.025), 0.35, 0.6, double_sided=True),
        "leaf_deep": make_mat("leaf_deep_green", (0.02, 0.10, 0.035), 0.35, 0.6, double_sided=True),
        "leaf_bronze": make_mat("leaf_bronze_green", (0.06, 0.07, 0.03), 0.35, 0.6, double_sided=True),
        "leaf_burg": make_mat("leaf_burgundy", (0.07, 0.008, 0.028), 0.35, 0.6, double_sided=True),
        "leaf_purple": make_mat("leaf_dark_purple", (0.05, 0.01, 0.045), 0.35, 0.6, double_sided=True),
        "wood": make_mat("trunk_bark", (0.07, 0.055, 0.04), 0.7, 0.3),
        "stem_green": make_mat("petiole", (0.10, 0.12, 0.05), 0.5, 0.4),
        "pot": make_mat("pot_white_ceramic", (0.88, 0.88, 0.86), 0.5, 0.5),
        "soil": make_mat("soil", (0.035, 0.022, 0.014), 0.95, 0.1),
    }


# ----------------------------------------------------------------- helpers
def new_obj(name, bm, mat, smooth=True, parent=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if parent:
        ob.parent = parent
    return ob


def lathe(profile, segs=64):
    """profile: list of (r, z); returns bmesh surface of revolution."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        ring = []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for k in range(len(rings) - 1):
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def smooth_curve(pts, n):
    """Catmull-Rom through pts, n samples."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    segs = len(pts) - 1
    for s in range(n):
        u = s / (n - 1) * segs
        i = min(int(u), segs - 1)
        t = u - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                          (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return out


def add_tube(bm, pts, radii, sides=10, cap_end=True):
    """Tube along polyline pts (Vectors) with per-point radii."""
    rings = []
    prev_n = None
    for i, p in enumerate(pts):
        if i < len(pts) - 1:
            t = (pts[i + 1] - p).normalized()
        else:
            t = (p - pts[i - 1]).normalized()
        ref = Vector((1, 0, 0)) if abs(t.x) < 0.9 else Vector((0, 1, 0))
        if prev_n is not None:
            ref = prev_n
        n = (ref - t * ref.dot(t)).normalized()
        prev_n = n
        b = t.cross(n)
        ring = []
        for s in range(sides):
            a = 2 * math.pi * s / sides
            ring.append(bm.verts.new(p + (n * math.cos(a) + b * math.sin(a)) * radii[i]))
        rings.append(ring)
    for k in range(len(rings) - 1):
        for s in range(sides):
            q = (s + 1) % sides
            bm.faces.new((rings[k][s], rings[k][q], rings[k + 1][q], rings[k + 1][s]))
    if cap_end:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])


def make_leaf_bm(L, W, droop, base, rot, fold=0.12, thick=0.0025):
    """Rubber-plant leaf: ovate blade, midrib fold, tip droop, thin solid.
    Local +Y = length, +Z = upper surface normal. Result in world space."""
    nu, nv = 10, 4  # along length, across (each half = nv/2)
    bm = bmesh.new()
    top, bot = [], []
    for i in range(nu + 1):
        t = i / nu
        # widest around 40 %, pointed drip tip
        w = 0.5 * W * (math.sin(math.pi * t ** 0.8) ** 0.62)
        row_t, row_b = [], []
        for j in range(nv + 1):
            v = -1 + 2 * j / nv
            x = v * w
            y = t * L
            z = fold * abs(x) * (1 - 0.5 * t)          # V-fold along midrib
            z += 0.25 * w * (1 - v * v) * 0.0
            z -= droop * (t * L) ** 2 / max(L, 1e-3)   # tip droop
            z -= 0.10 * (x * x) / max(L, 1e-3)         # slight cross-curl down at edge
            pt = Vector((x, y, z))
            row_t.append(bm.verts.new(rot @ pt + base))
            row_b.append(bm.verts.new(rot @ (pt + Vector((0, 0, -thick))) + base))
        top.append(row_t)
        bot.append(row_b)
    for i in range(nu):
        for j in range(nv):
            bm.faces.new((top[i][j], top[i][j + 1], top[i + 1][j + 1], top[i + 1][j]))
            bm.faces.new((bot[i][j], bot[i + 1][j], bot[i + 1][j + 1], bot[i][j + 1]))
    # close the rim
    for i in range(nu):
        for j in (0, nv):
            a, b = top[i][j], top[i + 1][j]
            c, d = bot[i + 1][j], bot[i][j]
            if j == 0:
                bm.faces.new((a, d, c, b))
            else:
                bm.faces.new((a, b, c, d))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


# ------------------------------------------------------------------ pieces
def make_pot(mats, style, root):
    if style == "round":
        outer = [(0.0, 0.0), (0.14, 0.0), (0.17, 0.012), (0.215, 0.07), (0.255, 0.17),
                 (0.275, 0.30), (0.268, 0.42), (0.245, 0.52), (0.232, 0.575), (0.24, 0.60)]
    else:  # cylinder with rounded foot, slight taper
        outer = [(0.0, 0.0), (0.22, 0.0), (0.255, 0.012), (0.268, 0.04), (0.274, 0.15),
                 (0.276, 0.40), (0.275, 0.575), (0.275, 0.60)]
    t = 0.024  # wall thickness
    rim_r = outer[-1][0]
    # inner wall: outer profile offset inward, walked from the rim down to the floor
    inner = [(rim_r - t, outer[-1][1])]
    for r, z in reversed(outer[2:-1]):
        if z > 0.13:
            inner.append((r - t, z - 0.004))
    inner.append((0.17, 0.12))
    inner.append((0.0, 0.12))
    # smooth the outer/inner profiles so the lathe is round, not faceted
    def sm(pts):
        v = smooth_curve([Vector((r, z, 0)) for r, z in pts], 3 * len(pts))
        return [(max(q.x, 0.0), q.y) for q in v]
    outer = [(0.0, 0.0)] + sm(outer[1:])
    inner = sm(inner[:-2]) + inner[-2:]
    prof = outer + inner
    bm = lathe(prof, 72)
    ob = new_obj("pot", bm, mats["pot"], True, root)
    return ob


def make_soil(mats, root, z=0.50, r=0.235):
    rnd = random.Random(7)
    bm = bmesh.new()
    rings = [(0.0, 1)]
    nrings, segs = 7, 40
    verts = []
    centre = bm.verts.new((0, 0, z + 0.012))
    prev = None
    for k in range(1, nrings + 1):
        rr = r * k / nrings
        ring = []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            h = z + 0.012 + (rnd.uniform(-0.012, 0.012) if k < nrings else -0.01)
            if k == nrings:
                h = z + 0.0
            ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), h)))
        if prev is None:
            for i in range(segs):
                bm.faces.new((centre, ring[i], ring[(i + 1) % segs]))
        else:
            for i in range(segs):
                j = (i + 1) % segs
                bm.faces.new((prev[i], prev[j], ring[j], ring[i]))
        prev = ring
    bm.faces.new(list(reversed(prev)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = new_obj("soil", bm, mats["soil"], True, root)
    # a few pebbles/bark chips on top
    bm2 = bmesh.new()
    for _ in range(14):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(0.04, r - 0.03)
        s = rnd.uniform(0.012, 0.025)
        bmesh.ops.create_icosphere(bm2, subdivisions=1, radius=s,
                                   matrix=Matrix.Translation((rr * math.cos(a), rr * math.sin(a), z + 0.012 + s * 0.3))
                                   @ Matrix.Scale(0.6, 4, (0, 0, 1)))
    new_obj("soil_chips", bm2, mats["soil"], True, root)
    return ob


# ------------------------------------------------------------------- plant
def build_plant(name, cfg):
    rnd = random.Random(cfg["seed"])
    mats = build_materials()
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    make_pot(mats, cfg["pot"], root)
    make_soil(mats, root)

    wood = bmesh.new()
    petioles = bmesh.new()
    leaves = []  # (base, dir, size scale)

    def stem(points, r0, r1, n=18, leaf_from=0.0, leaf_count=0, branch_tip=True, size=1.0):
        pts = smooth_curve([Vector(p) for p in points], n)
        radii = [r0 + (r1 - r0) * (i / (n - 1)) ** 0.8 for i in range(n)]
        add_tube(wood, pts, radii, 10 if r0 > 0.02 else 7)
        # leaves
        for k in range(leaf_count):
            t = leaf_from + (1 - leaf_from) * ((k + rnd.uniform(0, 0.6)) / max(leaf_count, 1))
            t = min(t, 0.985)
            fi = t * (n - 1)
            i = min(int(fi), n - 2)
            p = pts[i].lerp(pts[i + 1], fi - i)
            tan = (pts[i + 1] - pts[i]).normalized()
            rr = radii[i]
            ang = k * GOLDEN + rnd.uniform(-0.3, 0.3) + cfg.get("phase", 0)
            out = Vector((math.cos(ang), math.sin(ang), 0))
            pitch = math.radians(rnd.uniform(-8, 34)) if t < 0.9 else math.radians(rnd.uniform(25, 55))
            d = out * math.cos(pitch) + Vector((0, 0, math.sin(pitch)))
            # near tip bias along stem direction
            w = max(0.0, (t - 0.88) / 0.12)
            d = d.lerp(tan, 0.5 * w).normalized()
            base = p + out * rr
            pet_len = rnd.uniform(0.035, 0.065)
            pd = (d + Vector((0, 0, 0.35))).normalized()
            tip = base + pd * pet_len
            add_tube(petioles, [base, tip], [0.0045, 0.0028], 5)
            leaves.append((tip, d, size * (1.0 - 0.35 * t) if branch_tip else size))
        return pts

    z0 = 0.50  # soil level
    # main trunk(s)
    for s in cfg["stems"]:
        stem(s["pts"], s["r0"], s["r1"], 22, s["leaf_from"], s["leaves"], size=s.get("size", 1.0))

    mat_cycle = ["leaf_dark", "leaf_dark", "leaf_deep", "leaf_deep", "leaf_bronze",
                 "leaf_burg", "leaf_purple", "leaf_dark"]
    new_obj("trunk", wood, mats["wood"], True, root)
    new_obj("petioles", petioles, mats["stem_green"], True, root)

    burg_frac = cfg["burgundy"]
    for idx, (base, d, size) in enumerate(leaves):
        L = rnd.uniform(0.32, 0.46) * size * cfg["leaf_scale"]
        W = L * rnd.uniform(0.50, 0.58)
        q = d.to_track_quat("Y", "Z")
        roll = Matrix.Rotation(rnd.uniform(-0.35, 0.35), 3, "Y")
        rot = q.to_matrix() @ roll
        bm = make_leaf_bm(L, W, droop=rnd.uniform(0.2, 0.6), base=base, rot=rot)
        r = rnd.random()
        if r < burg_frac:
            mk = rnd.choice(["leaf_burg", "leaf_purple", "leaf_burg"])
        else:
            mk = rnd.choice(["leaf_dark", "leaf_dark", "leaf_deep", "leaf_deep", "leaf_bronze"])
        new_obj("leaf_%03d" % idx, bm, mats[mk], True, root)
    return root, len(leaves)


def tall_cfg():
    # trunk with gentle S curve, main branches radiating, leaves from ~0.95 m upward
    stems = [
        dict(pts=[(0, 0, 0.50), (0.02, -0.01, 1.0), (-0.03, 0.02, 1.5), (0.02, 0.0, 1.86), (0.0, 0.0, 1.95)],
             r0=0.040, r1=0.009, leaf_from=0.12, leaves=29, size=1.0),
        dict(pts=[(0.0, 0.0, 1.05), (0.20, 0.06, 1.35), (0.40, 0.10, 1.55)],
             r0=0.014, r1=0.006, leaf_from=0.15, leaves=17, size=0.95),
        dict(pts=[(0.0, 0.0, 1.25), (-0.22, 0.10, 1.55), (-0.38, 0.12, 1.72)],
             r0=0.013, r1=0.006, leaf_from=0.15, leaves=17, size=0.95),
        dict(pts=[(0.0, 0.0, 0.92), (-0.06, -0.22, 1.2), (-0.12, -0.38, 1.40)],
             r0=0.012, r1=0.006, leaf_from=0.15, leaves=15, size=1.0),
        dict(pts=[(0.0, 0.0, 1.48), (0.10, -0.20, 1.66), (0.18, -0.32, 1.78)],
             r0=0.011, r1=0.005, leaf_from=0.15, leaves=13, size=0.85),
        dict(pts=[(0.0, 0.0, 0.78), (0.20, -0.10, 0.98), (0.36, -0.14, 1.10)],
             r0=0.013, r1=0.006, leaf_from=0.1, leaves=9, size=1.0),
        dict(pts=[(0.0, 0.0, 0.85), (-0.18, -0.12, 1.02), (-0.34, -0.10, 1.12)],
             r0=0.013, r1=0.006, leaf_from=0.1, leaves=9, size=1.0),
        dict(pts=[(0.0, 0.0, 1.62), (0.06, 0.20, 1.72), (0.0, 0.30, 1.84)],
             r0=0.010, r1=0.005, leaf_from=0.15, leaves=10, size=0.8),
    ]
    return dict(seed=11, pot="round", stems=stems, burgundy=0.22, leaf_scale=1.0, phase=0.0)


def medium_cfg():
    stems = [
        dict(pts=[(0.0, 0.0, 0.50), (0.03, 0.0, 0.9), (0.06, 0.02, 1.35), (0.04, 0.03, 1.52)],
             r0=0.024, r1=0.008, leaf_from=0.12, leaves=26, size=1.0),
        dict(pts=[(-0.06, 0.04, 0.50), (-0.14, 0.08, 0.85), (-0.22, 0.12, 1.20), (-0.27, 0.14, 1.36)],
             r0=0.020, r1=0.008, leaf_from=0.18, leaves=22, size=1.0),
        dict(pts=[(0.06, -0.05, 0.50), (0.14, -0.10, 0.85), (0.22, -0.12, 1.15), (0.28, -0.12, 1.32)],
             r0=0.020, r1=0.008, leaf_from=0.18, leaves=22, size=1.0),
        dict(pts=[(0.0, 0.07, 0.50), (-0.02, 0.15, 0.80), (-0.05, 0.22, 1.05), (-0.04, 0.28, 1.28)],
             r0=0.016, r1=0.007, leaf_from=0.2, leaves=16, size=0.95),
        dict(pts=[(0.0, -0.07, 0.50), (0.0, -0.15, 0.78), (0.04, -0.22, 1.02), (0.05, -0.28, 1.20)],
             r0=0.016, r1=0.007, leaf_from=0.2, leaves=16, size=0.95),
        dict(pts=[(0.02, 0.0, 1.0), (0.14, 0.14, 1.28), (0.24, 0.22, 1.38)],
             r0=0.010, r1=0.005, leaf_from=0.2, leaves=10, size=0.85),
    ]
    return dict(seed=23, pot="cyl", stems=stems, burgundy=0.18, leaf_scale=0.92, phase=1.0)


# ------------------------------------------------------------------ export
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def export(name):
    root = bpy.data.objects[name]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in bpy.data.objects:
        ob.select_set(True)
    path = os.path.join(OUT_DIR, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_yup=True,
                              export_apply=True, use_selection=True,
                              export_cameras=False, export_lights=False)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, cfg in (("plant_tall", tall_cfg()), ("plant_medium", medium_cfg())):
        reset()
        root, n = build_plant(name, cfg)
        tris = 0
        zmax = -1e9
        for ob in bpy.data.objects:
            if ob.type == "MESH":
                for p in ob.data.polygons:
                    tris += len(p.vertices) - 2
                for v in ob.data.vertices:
                    zmax = max(zmax, v.co.z)
        print("[plants] %s leaves=%d tris=%d height=%.3f" % (name, n, tris, zmax))
        print("[plants] wrote", export(name))


if __name__ == "__main__":
    main()
