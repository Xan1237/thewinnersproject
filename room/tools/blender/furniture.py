"""Procedural furniture / stage props for the hackathon room.

Run:  blender -b --python tools/blender/furniture.py -- [out_dir] [lectern table screen stage]

Blender is Z-up; the glTF exporter converts to Y-up (x, z, -y).  Conventions used:
  * "front" of everything is Godot -Z  ==  Blender +Y  (audience side).
  * Everything is modelled directly in final world coordinates (no object transforms).
"""
import bpy, bmesh, math, os, sys, random
from mathutils import Vector, Matrix

OUT_DIR = "/Users/sukhov/Projects/thewinnersproject/assets/models"
TEX_DIR = os.path.expanduser("~/.hermes/cache/scratch/furniture_tex")
V = Vector


# ----------------------------------------------------------------- materials
def make_material(name, color, rough=0.5, metal=0.0, cull=True, tex=None, spec=0.5):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = spec
    if tex:
        img = bpy.data.images.load(tex)
        img.pack()
        t = m.node_tree.nodes.new('ShaderNodeTexImage')
        t.image = img
        m.node_tree.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
    m.use_backface_culling = cull
    return m


# ----------------------------------------------------------------- builder
class Builder:
    def __init__(self):
        self.bms = {}      # material name -> bmesh
        self.mats = {}     # material name -> bpy material
        self.custom_uv = set()
        self.uv_scale = {}  # material name -> metres per uv tile (u, v)

    def add_mat(self, mat, custom_uv=False, uv_scale=(1.0, 1.0)):
        self.mats[mat.name] = mat
        bm = bmesh.new()
        bm.loops.layers.uv.verify()
        self.bms[mat.name] = bm
        if custom_uv:
            self.custom_uv.add(mat.name)
        self.uv_scale[mat.name] = uv_scale

    # -- merge a temp bmesh into the material bmesh
    def _merge(self, mat, src, bevel=0.0, seg=2, bevel_filter=None):
        if bevel > 0:
            edges = [e for e in src.edges if (bevel_filter is None or bevel_filter(e))]
            bmesh.ops.bevel(src, geom=edges, offset=bevel, segments=seg, profile=0.5,
                            affect='EDGES', clamp_overlap=True)
        dst = self.bms[mat]
        vm = {v: dst.verts.new(v.co) for v in src.verts}
        for f in src.faces:
            try:
                dst.faces.new([vm[v] for v in f.verts])
            except ValueError:
                pass
        src.free()

    def box(self, mat, center, size, bevel=0.004, seg=2):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = V((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2])) + V(center)
        self._merge(mat, bm, bevel, seg)

    def prism(self, mat, profile_yz, x0, x1, bevel=0.004, seg=2):
        """Extrude a (y,z) polygon (CCW seen from +X) along X from x0 to x1."""
        bm = bmesh.new()
        a = [bm.verts.new((x0, y, z)) for y, z in profile_yz]
        b = [bm.verts.new((x1, y, z)) for y, z in profile_yz]
        n = len(a)
        bm.faces.new(list(reversed(a)))
        bm.faces.new(b)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new([a[i], a[j], b[j], b[i]])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        self._merge(mat, bm, bevel, seg)

    def rounded_slab(self, mat, center, w, d, h, r, bevel=0.01, seg=3, corner_seg=8):
        bm = bmesh.new()
        pts = []
        for cx, cy, a0 in ((w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90),
                           (-w / 2 + r, -d / 2 + r, 180), (w / 2 - r, -d / 2 + r, 270)):
            for i in range(corner_seg + 1):
                a = math.radians(a0 + 90 * i / corner_seg)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        top = [bm.verts.new((x, y, h / 2)) for x, y in pts]
        bot = [bm.verts.new((x, y, -h / 2)) for x, y in pts]
        n = len(pts)
        bm.faces.new(top)
        bm.faces.new(list(reversed(bot)))
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new([bot[i], bot[j], top[j], top[i]])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        for v in bm.verts:
            v.co += V(center)
        # bevel only the horizontal rim edges
        self._merge(mat, bm, bevel, seg,
                    bevel_filter=lambda e: abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-6)

    def cyl(self, mat, p1, p2, r, seg=16, caps=True):
        p1, p2 = V(p1), V(p2)
        d = p2 - p1
        L = d.length
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=seg,
                              radius1=r, radius2=r, depth=L)
        rot = V((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
        bmesh.ops.transform(bm, matrix=Matrix.Translation((p1 + p2) / 2) @ rot, verts=bm.verts)
        self._merge(mat, bm)

    def sphere(self, mat, c, r, seg=12):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2 + 1, radius=r)
        for v in bm.verts:
            v.co += V(c)
        self._merge(mat, bm)

    def quad(self, mat, pts, uvs):
        bm = self.bms[mat]
        vs = [bm.verts.new(p) for p in pts]
        f = bm.faces.new(vs)
        uvl = bm.loops.layers.uv.active
        for l, uv in zip(f.loops, uvs):
            l[uvl].uv = uv

    def grid(self, mat, origin, u_dir, length, z_top, z_bot, normal, amp, period, cols_per_pleat=8):
        """Pleated hanging cloth strip along u_dir (unit), pleats offset along normal."""
        bm = self.bms[mat]
        o, u, n = V(origin), V(u_dir), V(normal)
        npl = max(1, round(length / period))
        cols = npl * cols_per_pleat
        top, bot = [], []
        for i in range(cols + 1):
            t = i / cols
            off = amp * math.sin(2 * math.pi * npl * t)
            p = o + u * (length * t) + n * off
            top.append(bm.verts.new((p.x, p.y, z_top)))
            bot.append(bm.verts.new((p.x, p.y, z_bot)))
        for i in range(cols):
            bm.faces.new([bot[i], bot[i + 1], top[i + 1], top[i]])

    # -- finalise into scene objects
    def finish(self, smooth_angle=35):
        objs = []
        for name, bm in self.bms.items():
            if not bm.faces:
                bm.free()
                continue
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:]) if name not in self.custom_uv else None
            uvl = bm.loops.layers.uv.active
            if name not in self.custom_uv:
                su, sv = self.uv_scale[name]
                for f in bm.faces:
                    n = f.normal
                    ax = max(range(3), key=lambda i: abs(n[i]))
                    for l in f.loops:
                        c = l.vert.co
                        if ax == 2:
                            uv = (c.x / su, c.y / sv)
                        elif ax == 1:
                            uv = (c.x / su, c.z / sv)
                        else:
                            uv = (c.y / su, c.z / sv)
                        l[uvl].uv = uv
            lim = math.radians(smooth_angle)
            for f in bm.faces:
                f.smooth = True
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    if e.calc_face_angle(0) > lim:
                        e.smooth = False
                else:
                    e.smooth = False
            me = bpy.data.meshes.new(name + "_mesh")
            bm.to_mesh(me)
            bm.free()
            me.materials.append(self.mats[name])
            ob = bpy.data.objects.new(name, me)
            bpy.context.scene.collection.objects.link(ob)
            objs.append(ob)
        return objs


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def export(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_yup=True,
                              export_apply=True, export_materials='EXPORT',
                              export_image_format='AUTO', use_selection=False)
    print("EXPORTED", path)


# ----------------------------------------------------------------- 1. lectern
def build_lectern():
    reset_scene()
    B = Builder()
    white = make_material("LecternWhite", (0.88, 0.88, 0.86), 0.32, 0.0)
    panel = make_material("LecternPanel", (0.80, 0.80, 0.79), 0.28, 0.0)
    black = make_material("LecternTrim", (0.015, 0.015, 0.017), 0.45, 0.2)
    for m in (white, panel, black):
        B.add_mat(m)

    slope = math.tan(math.radians(15))
    zs = lambda y: 1.088 - (0.25 - y) * slope        # slab top surface height at y
    # body (sides + back + front shell)
    by, bx, t = 0.235, 0.335, 0.035
    B.prism("LecternWhite", [(-by, 0.07), (by, 0.07), (by, zs(by) - t), (-by, zs(-by) - t)],
            -bx, bx, bevel=0.006, seg=2)
    # sloped top shelf
    B.prism("LecternWhite", [(-0.25, zs(-0.25) - t), (0.25, zs(0.25) - t), (0.25, zs(0.25)), (-0.25, zs(-0.25))],
            -0.35, 0.35, bevel=0.006, seg=3)
    # lip at the low (speaker) edge
    B.prism("LecternWhite", [(-0.25, zs(-0.25)), (-0.215, zs(-0.215)), (-0.215, zs(-0.215) + 0.04),
                              (-0.25, zs(-0.25) + 0.04)], -0.34, 0.34, bevel=0.004, seg=2)
    # stop strip at the high edge
    B.prism("LecternWhite", [(0.215, zs(0.215)), (0.25, zs(0.25)), (0.25, zs(0.25) + 0.012),
                              (0.215, zs(0.215) + 0.012)], -0.34, 0.34, bevel=0.003, seg=2)
    # black plinth
    B.box("LecternTrim", (0, 0, 0.035), (0.60, 0.42, 0.07), bevel=0.004)
    # front panel with black reveal
    B.box("LecternTrim", (0, by + 0.002, 0.50), (0.54, 0.004, 0.66), bevel=0.002, seg=1)
    B.box("LecternPanel", (0, by + 0.006, 0.50), (0.50, 0.008, 0.62), bevel=0.003, seg=2)
    # horizontal groove line across front above the panel
    B.box("LecternTrim", (0, by + 0.001, 0.86), (0.67, 0.003, 0.006), bevel=0.0)
    # side panels
    for sx in (-1, 1):
        B.box("LecternTrim", (sx * (bx + 0.002), 0, 0.50), (0.004, 0.40, 0.66), bevel=0.002, seg=1)
        B.box("LecternPanel", (sx * (bx + 0.006), 0, 0.50), (0.008, 0.36, 0.62), bevel=0.003, seg=2)
    # back: open-look shelf compartment (recessed dark panel + a shelf board)
    B.box("LecternTrim", (0, -by - 0.002, 0.45), (0.54, 0.004, 0.62), bevel=0.002, seg=1)
    B.box("LecternWhite", (0, -by - 0.006, 0.45), (0.50, 0.008, 0.58), bevel=0.003, seg=2)
    B.finish()
    export("lectern_white.glb", OUT_DIR)


# ----------------------------------------------------------------- 2. table
def build_table():
    reset_scene()
    B = Builder()
    plastic = make_material("TablePlastic", (0.90, 0.90, 0.89), 0.42, 0.0)
    metal = make_material("TableFrameBlack", (0.02, 0.02, 0.022), 0.38, 0.65)
    rubber = make_material("TableFeetRubber", (0.01, 0.01, 0.01), 0.85, 0.0)
    for m in (plastic, metal, rubber):
        B.add_mat(m)

    TH, T = 0.74, 0.03
    B.rounded_slab("TablePlastic", (0, 0, TH - T / 2), 1.8, 0.7, T, 0.04, bevel=0.01, seg=3)
    # underside apron
    zc = TH - T - 0.016
    B.box("TablePlastic", (0, 0.35 - 0.026, zc), (1.70, 0.012, 0.032), bevel=0.003, seg=1)
    B.box("TablePlastic", (0, -0.35 + 0.026, zc), (1.70, 0.012, 0.032), bevel=0.003, seg=1)
    for sx in (-1, 1):
        B.box("TablePlastic", (sx * (0.9 - 0.026), 0, zc), (0.012, 0.66, 0.032), bevel=0.003, seg=1)
    # underside ribs
    for x in (-0.45, 0.0, 0.45):
        B.box("TablePlastic", (x, 0, TH - T - 0.01), (0.012, 0.60, 0.02), bevel=0.002, seg=1)

    r = 0.0125
    top_z = TH - T - 0.02          # top rail height
    foot_z = 0.016
    mid_z = 0.22
    lx, ly = 0.78, 0.28
    for sx in (-1, 1):
        x = sx * lx
        for sy in (-1, 1):
            B.cyl("TableFrameBlack", (x, sy * ly, foot_z), (x, sy * ly, top_z), r, 16)
            B.cyl("TableFeetRubber", (x, sy * ly, 0.0), (x, sy * ly, 0.020), 0.0165, 16)
            B.box("TableFrameBlack", (x, sy * ly, top_z), (0.05, 0.05, 0.035), bevel=0.004, seg=1)
            B.sphere("TableFrameBlack", (x, sy * ly, foot_z), r)
        B.cyl("TableFrameBlack", (x, -ly, top_z), (x, ly, top_z), r * 0.8, 12)          # top rail
        B.cyl("TableFrameBlack", (x, -ly, foot_z), (x, ly, foot_z), r * 0.8, 12)        # foot rail
        B.cyl("TableFrameBlack", (x, -ly, mid_z), (x, ly, mid_z), 0.009, 12)            # mid rail
    # slim crossbar between the two end frames
    B.cyl("TableFrameBlack", (-lx, 0, mid_z), (lx, 0, mid_z), 0.009, 12)
    B.finish()
    export("table_folding.glb", OUT_DIR)


# ----------------------------------------------------------------- 3. projector screen
def build_screen():
    reset_scene()
    B = Builder()
    alu = make_material("ScreenCaseAluminium", (0.86, 0.86, 0.85), 0.38, 0.35)
    cap = make_material("ScreenCaseEndCap", (0.55, 0.56, 0.58), 0.35, 0.8)
    black = make_material("ScreenBorder", (0.012, 0.012, 0.012), 0.9, 0.0)
    back = make_material("ScreenBack", (0.02, 0.02, 0.02), 0.9, 0.0)
    surf = make_material("ScreenSurface", (0.86, 0.86, 0.84), 0.85, 0.0)
    for m in (alu, cap, black, back):
        B.add_mat(m)
    B.add_mat(surf, custom_uv=True)

    CW, CD, CH = 4.4, 0.22, 0.22
    # case
    B.box("ScreenCaseAluminium", (0, 0, -CH / 2), (CW - 0.024, CD, CH), bevel=0.05, seg=4)
    for sx in (-1, 1):
        B.box("ScreenCaseEndCap", (sx * (CW / 2 - 0.006), 0, -CH / 2), (0.012, CD, CH),
              bevel=0.008, seg=2)
    # dark slot where the fabric leaves the case
    B.box("ScreenBorder", (0, 0, -CH + 0.001), (CW - 0.1, 0.05, 0.004), bevel=0.0)

    SW, SH = 4.0, 2.25
    bt, bs, bb = 0.08, 0.12, 0.12     # top / side / bottom border
    th = 0.002
    z_top = -CH                       # fabric starts at the underside of the case
    z_s0 = z_top - bt                 # top of white surface
    z_s1 = z_s0 - SH                  # bottom of white surface
    z_b1 = z_s1 - bb                  # bottom of border
    tw = SW + 2 * bs
    # border strips (thin boxes, black)
    B.box("ScreenBorder", (0, 0, (z_top + z_s0) / 2), (tw, th, bt), bevel=0)
    B.box("ScreenBorder", (0, 0, (z_s1 + z_b1) / 2), (tw, th, bb), bevel=0)
    for sx in (-1, 1):
        B.box("ScreenBorder", (sx * (SW / 2 + bs / 2), 0, (z_s0 + z_s1) / 2), (bs, th, SH), bevel=0)
    # white surface quad, faces +Y (Godot -Z); viewer looks toward -Y, their right = -X
    y = th / 2
    x = SW / 2
    B.quad("ScreenSurface",
           [(x, y, z_s1), (-x, y, z_s1), (-x, y, z_s0), (x, y, z_s0)],
           [(0, 0), (1, 0), (1, 1), (0, 1)])
    # black backing, faces -Y
    yb = -th / 2
    B.quad("ScreenBack",
           [(-x, yb, z_s1), (x, yb, z_s1), (x, yb, z_s0), (-x, yb, z_s0)],
           [(0, 0), (1, 0), (1, 1), (0, 1)])
    # weight bar
    br = 0.022
    B.cyl("ScreenCaseAluminium", (-tw / 2, 0, z_b1 - br), (tw / 2, 0, z_b1 - br), br, 20)
    for sx in (-1, 1):
        B.sphere("ScreenCaseEndCap", (sx * tw / 2, 0, z_b1 - br), br * 1.05, 12)
    # bar pocket (fabric wrap)
    B.box("ScreenBorder", (0, 0, z_b1 - 0.002), (tw, 0.012, 0.01), bevel=0.0)
    B.finish()
    export("projector_screen.glb", OUT_DIR)


# ----------------------------------------------------------------- 4. stage deck
def make_plywood_texture():
    import numpy as np
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, "plywood_dark.png")
    W, H = 1024, 512
    rng = np.random.default_rng(7)
    u = np.arange(W)[None, :] / W
    v = np.arange(H)[:, None] / H
    # periodic grain: many thin streaks along u, gently bent
    val = np.zeros((H, W))
    for _ in range(40):
        f = rng.integers(8, 140)
        ph = rng.uniform(0, 2 * np.pi)
        bend = rng.integers(1, 3) * rng.uniform(0.002, 0.012)
        k = rng.integers(1, 3)
        val += rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * (f * v + bend * H * np.sin(2 * np.pi * k * u + ph)) + ph)
    val /= 12.0
    # fine fibre noise stretched along u
    n = rng.normal(0, 1, (H, W // 16))
    n = np.repeat(n, 16, axis=1)
    for _ in range(2):
        n = (n + np.roll(n, 1, 1) + np.roll(n, -1, 1)) / 3
    val += n * 0.10
    val = (val - val.mean()) / (val.std() + 1e-6)
    base = np.array([0.19, 0.125, 0.085])   # sRGB, dark stained birch ply
    img = base[None, None, :] * (1.0 + 0.22 * val[..., None])
    # faint sheet seam lines (texture tile = one 2440x1220 sheet)
    for ax_pos in (0, 1):
        img[(ax_pos * (H - 3)):(ax_pos * (H - 3)) + 3, :, :] *= 0.7
        img[:, (ax_pos * (W - 3)):(ax_pos * (W - 3)) + 3, :] *= 0.7
    img = np.clip(img, 0, 1)
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[..., :3] = img
    im = bpy.data.images.new("plywood_dark_tmp", W, H, alpha=False)
    im.pixels = rgba[::-1].reshape(-1).tolist()   # image rows bottom-up
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    return path


def build_stage():
    reset_scene()
    tex = None
    try:
        tex = make_plywood_texture()
    except Exception as e:
        print("plywood texture failed:", e)
    B = Builder()
    ply = make_material("StagePlywood", (0.19, 0.125, 0.085), 0.62, 0.0, tex=tex)
    alu = make_material("StageTrimAluminium", (0.78, 0.79, 0.80), 0.32, 0.9)
    steel = make_material("StageFrameSteel", (0.42, 0.43, 0.45), 0.45, 0.85)
    skirt = make_material("StageSkirt", (0.012, 0.012, 0.013), 0.95, 0.0, cull=False)
    rubber = make_material("StageFeet", (0.01, 0.01, 0.01), 0.85, 0.0)
    B.add_mat(ply, uv_scale=(2.44, 1.22))
    for m in (alu, steel, skirt, rubber):
        B.add_mat(m)

    L, W, H = 6.4, 3.8, 0.35
    hx, hy = L / 2, W / 2
    tr_t, tr_h = 0.015, 0.05
    # plywood deck
    B.box("StagePlywood", (0, 0, H - 0.0095), (L - 2 * tr_t, W - 2 * tr_t, 0.019), bevel=0.0015, seg=1)
    # aluminium edge trim (L-section: vertical face + small top lip)
    zt = H - tr_h / 2
    B.box("StageTrimAluminium", (0, hy - tr_t / 2, zt), (L, tr_t, tr_h), bevel=0.003, seg=2)
    B.box("StageTrimAluminium", (0, -hy + tr_t / 2, zt), (L, tr_t, tr_h), bevel=0.003, seg=2)
    for sx in (-1, 1):
        B.box("StageTrimAluminium", (sx * (hx - tr_t / 2), 0, zt), (tr_t, W - 2 * tr_t, tr_h), bevel=0.003, seg=2)
    # steel frame under the deck
    fz0, fz1 = 0.272, H - 0.019
    fh = fz1 - fz0
    fc = (fz0 + fz1) / 2
    ft = 0.04
    ox, oy = hx - 0.06, hy - 0.06
    B.box("StageFrameSteel", (0, oy - ft / 2, fc), (2 * ox, ft, fh), bevel=0.003, seg=1)
    B.box("StageFrameSteel", (0, -oy + ft / 2, fc), (2 * ox, ft, fh), bevel=0.003, seg=1)
    for sx in (-1, 1):
        B.box("StageFrameSteel", (sx * (ox - ft / 2), 0, fc), (ft, 2 * oy - 2 * ft, fh), bevel=0.003, seg=1)
    for x in (-2.4, -1.2, 0.0, 1.2, 2.4):
        B.box("StageFrameSteel", (x, 0, fc), (0.03, 2 * oy - 2 * ft, fh), bevel=0.002, seg=1)
    B.box("StageFrameSteel", (0, 0, fc), (2 * ox - 2 * ft, 0.03, fh), bevel=0.002, seg=1)
    # legs + adjustable feet
    lw = 0.05
    xo = ox - ft / 2
    for x in (-xo, -xo / 2, 0.0, xo / 2, xo):
        for y in (-oy + ft / 2, 0.0, oy - ft / 2):
            B.box("StageFrameSteel", (x, y, (0.012 + fz0) / 2), (lw, lw, fz0 - 0.012), bevel=0.004, seg=1)
            B.cyl("StageFeet", (x, y, 0.0), (x, y, 0.014), 0.034, 16)
    # black pleated skirt on front (+Y = Godot -Z) and the two short sides
    zs_top, zs_bot = H - tr_h, 0.014
    amp = 0.009
    base = tr_t + amp + 0.001
    B.grid("StageSkirt", (-(hx - base), hy - base, 0), (1, 0, 0), 2 * (hx - base), zs_top, zs_bot, (0, 1, 0), amp, 0.12)
    for sx in (-1, 1):
        B.grid("StageSkirt", (sx * (hx - base), -(hy - base), 0), (0, 1, 0), 2 * (hy - base), zs_top, zs_bot,
               (sx, 0, 0), amp, 0.12)
    B.finish()
    export("stage_deck.glb", OUT_DIR)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv and os.path.isdir(argv[0]) or (argv and argv[0].startswith("/")):
        OUT_DIR = argv[0]
        argv = argv[1:]
    which = argv or ["lectern", "table", "screen", "stage"]
    fn = dict(lectern=build_lectern, table=build_table, screen=build_screen, stage=build_stage)
    for w in which:
        fn[w]()
