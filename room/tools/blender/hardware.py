"""Stage / ceiling hardware GLB generator (Blender headless).

Usage:  blender -b --python tools/blender/hardware.py [-- name1 name2 ...]
Names: pendant_dome pa_speaker_hanging spot_can speaker_stand tripod_camera (default: all)

Conventions: Blender Z-up, real metres. Exported with export_yup=True, so
Blender +Z -> glTF/Godot +Y and Blender -Y (model "front") -> glTF/Godot +Z.
Each model is a single mesh object whose origin is at world (0,0,0) = the
attachment origin described in the task.
"""
import bpy, bmesh, math, os, sys
import numpy as np
from math import radians, cos, sin, pi, atan2, sqrt, hypot
from mathutils import Matrix, Vector

OUT = "/Users/sukhov/Projects/thewinnersproject/assets/models"
SHARP_ANGLE = radians(50)
I4 = Matrix.Identity(4)


# ---------------------------------------------------------------- transforms
def T(x, y=0.0, z=0.0):
    if isinstance(x, (Vector, tuple, list)):
        x, y, z = x
    return Matrix.Translation((x, y, z))

def Rx(a): return Matrix.Rotation(a, 4, 'X')
def Ry(a): return Matrix.Rotation(a, 4, 'Y')
def Rz(a): return Matrix.Rotation(a, 4, 'Z')
def S(x, y, z): return Matrix.Diagonal((x, y, z, 1.0))

def along(pos, direction):
    """Matrix placing local +Z along `direction` at `pos`."""
    d = Vector(direction).normalized()
    q = Vector((0, 0, 1)).rotation_difference(d)
    return T(pos) @ q.to_matrix().to_4x4()


# ---------------------------------------------------------------- materials
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def mat(name, base=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, coat=0.0,
        emit=None, emit_strength=0.0, ior=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    if ior:
        b.inputs["IOR"].default_value = ior
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = emit_strength
    return m

def grille_mat(name="grille_perforated"):
    """Perforated steel grille: hole pattern baked into base colour + normal map."""
    N = 256
    yy, xx = np.mgrid[0:N, 0:N]
    u = (xx + 0.5) / N
    v = (yy + 0.5) / N
    def dist(cx, cy):
        du = ((u - cx + 0.5) % 1.0) - 0.5
        dv = ((v - cy + 0.5) % 1.0) - 0.5
        return np.sqrt(du * du + dv * dv)
    d = np.minimum(dist(0.25, 0.25), dist(0.75, 0.75))
    rad = 0.19
    h = np.clip((d - rad + 0.03) / 0.06, 0.0, 1.0)
    h = h * h * (3 - 2 * h)                           # 0 = hole, 1 = plate
    grey = 0.004 + (0.22 - 0.004) * h
    base = np.stack([grey, grey, grey, np.ones_like(grey)], -1)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * 5.0
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * 5.0
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.concatenate([n * 0.5 + 0.5, np.ones_like(h)[..., None]], -1)

    def mkimg(nm, arr, nonc):
        im = bpy.data.images.new(nm, N, N, alpha=False)
        if nonc:
            im.colorspace_settings.name = 'Non-Color'
        im.pixels.foreach_set(arr.astype(np.float32).ravel())
        im.file_format = 'PNG'
        im.pack()
        return im
    ib = mkimg(name + "_base", base, False)
    inr = mkimg(name + "_normal", nrm, True)

    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.35
    bsdf.inputs["Roughness"].default_value = 0.5
    tb = nt.nodes.new("ShaderNodeTexImage"); tb.image = ib
    tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = inr
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0
    nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


# ------------------------------------------------------------------- builder
class Builder:
    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.mats = []
        self.uvs = []

    def _idx(self, m, uv_scale=1.0):
        if m not in self.mats:
            self.mats.append(m)
            self.uvs.append(uv_scale)
        return self.mats.index(m)

    def uv(self, m, scale):
        self.uvs[self._idx(m)] = scale

    def _commit(self, tmp, m, M, bevel=0.0, seg=2):
        if bevel > 0:
            edges = [e for e in tmp.edges
                     if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > radians(30)]
            if edges:
                bmesh.ops.bevel(tmp, geom=edges, offset=bevel, segments=seg,
                                affect='EDGES', clamp_overlap=True)
        idx = self._idx(m)
        flip = M.determinant() < 0
        vmap = {}
        for v in tmp.verts:
            vmap[v] = self.bm.verts.new(M @ v.co)
        for f in tmp.faces:
            vs = [vmap[v] for v in f.verts]
            if flip:
                vs.reverse()
            try:
                nf = self.bm.faces.new(vs)
                nf.material_index = idx
            except ValueError:
                pass
        tmp.free()

    # primitives -----------------------------------------------------------
    def box(self, m, size, M=I4, bevel=0.0, seg=2):
        tmp = bmesh.new()
        bmesh.ops.create_cube(tmp, size=1.0)
        bmesh.ops.transform(tmp, matrix=S(*size), verts=tmp.verts)
        self._commit(tmp, m, M, bevel, seg)

    def lathe(self, m, profile, M=I4, segs=32, closed=False, bevel=0.0, seg=2):
        """Revolve [(r, z), ...] about local Z. Open profiles are capped to the axis."""
        tmp = bmesh.new()
        pts = list(profile)
        if not closed:
            if pts[0][0] > 1e-9:
                pts.insert(0, (0.0, pts[0][1]))
            if pts[-1][0] > 1e-9:
                pts.append((0.0, pts[-1][1]))
        rings = []
        for r, z in pts:
            if r < 1e-9:
                rings.append([tmp.verts.new((0, 0, z))])
            else:
                rings.append([tmp.verts.new((r * cos(2 * pi * j / segs), r * sin(2 * pi * j / segs), z))
                              for j in range(segs)])
        n = len(pts)
        last = n if closed else n - 1
        for i in range(last):
            a = rings[i]; b = rings[(i + 1) % n]
            for j in range(segs):
                j2 = (j + 1) % segs
                vs = [a[j % len(a)], a[j2 % len(a)], b[j2 % len(b)], b[j % len(b)]]
                out = []
                for v in vs:
                    if not out or out[-1] is not v:
                        out.append(v)
                if len(out) > 1 and out[0] is out[-1]:
                    out.pop()
                if len(out) >= 3:
                    try:
                        tmp.faces.new(out)
                    except ValueError:
                        pass
        bmesh.ops.recalc_face_normals(tmp, faces=tmp.faces)
        self._commit(tmp, m, M, bevel, seg)

    def cyl(self, m, r, h, M=I4, segs=24, r2=None, bevel=0.0, seg=2):
        """Cylinder/cone with base at local origin extending along +Z."""
        r2 = r if r2 is None else r2
        self.lathe(m, [(r, 0.0), (r2, h)], M, segs, bevel=bevel, seg=seg)

    def torus(self, m, R, r, M=I4, segs=24, rsegs=8):
        prof = [(R + r * cos(2 * pi * k / rsegs), r * sin(2 * pi * k / rsegs)) for k in range(rsegs)]
        self.lathe(m, prof, M, segs, closed=True)

    def hull(self, m, pts, M=I4, bevel=0.0, seg=2):
        tmp = bmesh.new()
        for p in pts:
            tmp.verts.new(p)
        bmesh.ops.convex_hull(tmp, input=tmp.verts)
        bmesh.ops.dissolve_limit(tmp, angle_limit=radians(1.0), verts=tmp.verts, edges=tmp.edges)
        bmesh.ops.recalc_face_normals(tmp, faces=tmp.faces)
        self._commit(tmp, m, M, bevel, seg)

    def rod(self, m, p0, p1, r, segs=12, r2=None, bevel=0.0):
        p0 = Vector(p0); p1 = Vector(p1)
        self.cyl(m, r, (p1 - p0).length, along(p0, p1 - p0), segs, r2, bevel)

    # output ---------------------------------------------------------------
    def tri_count(self):
        return sum(len(f.verts) - 2 for f in self.bm.faces)

    def finish(self, filename):
        bm = self.bm
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        uvl = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            sc = self.uvs[f.material_index]
            n = f.normal
            ax = max(range(3), key=lambda i: abs(n[i]))
            for l in f.loops:
                c = l.vert.co
                if ax == 0:   uu, vv = c.y, c.z
                elif ax == 1: uu, vv = c.x, c.z
                else:         uu, vv = c.x, c.y
                l[uvl].uv = (uu / sc, vv / sc)
        for e in bm.edges:
            e.smooth = (len(e.link_faces) != 2) or (e.calc_face_angle(0.0) < SHARP_ANGLE)
        for f in bm.faces:
            f.smooth = True
        mesh = bpy.data.meshes.new(self.name)
        bm.to_mesh(mesh)
        bm.free()
        for m in self.mats:
            mesh.materials.append(m)
        obj = bpy.data.objects.new(self.name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, filename)
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_yup=True,
                                  export_apply=True, use_selection=True)
        tris = sum(len(p.vertices) - 2 for p in mesh.polygons)
        bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        mn = [min(v[i] for v in bb) for i in range(3)]
        mx = [max(v[i] for v in bb) for i in range(3)]
        print(f"[OK] {filename}: tris~{tris} blender-bbox min={tuple(round(x,3) for x in mn)} max={tuple(round(x,3) for x in mx)}")
        return path


# ------------------------------------------------------------- shared pieces
def speaker_cabinet(b, M_, wf, wb, d, z0, z1, grille_margin=0.018, bevel=0.012,
                    handles=True, handle_z=None):
    """Trapezoid (plan) loudspeaker cabinet; front (grille) faces -Y."""
    paint, grille, rubber, steel = M_['paint'], M_['grille'], M_['rubber'], M_['steel']
    pts = []
    for z in (z0, z1):
        pts += [(-wf / 2, -d / 2, z), (wf / 2, -d / 2, z), (wb / 2, d / 2, z), (-wb / 2, d / 2, z)]
    b.hull(paint, pts, bevel=bevel, seg=3)
    # perforated grille plate flush with the front face
    h = z1 - z0
    gz = (z0 + z1) / 2
    b.box(grille, (wf - 2 * grille_margin, 0.012, h - 2 * grille_margin),
          T(0, -d / 2 - 0.002, gz), bevel=0.002)
    # badge
    b.box(steel, (0.06, 0.003, 0.014), T(-wf / 2 + grille_margin + 0.05, -d / 2 - 0.0085, z0 + grille_margin + 0.02), bevel=0.0008)
    if handles:
        hz = gz if handle_z is None else handle_z
        L = hypot((wf - wb) / 2, d)
        for s in (-1, 1):
            dv = Vector((s * (wb - wf) / 2, d)) / L
            nv = Vector((s * d, (wf - wb) / 2)) / L
            mid = Vector((s * (wf + wb) / 4, 0.0))
            ang = atan2(dv.y, dv.x)
            R = Rz(ang)
            c = mid + nv * 0.026
            b.box(rubber, (0.13, 0.016, 0.022), T(c.x, c.y, hz) @ R, bevel=0.005)
            for k in (-1, 1):
                pc = mid + dv * (0.05 * k) + nv * 0.012
                b.box(steel, (0.016, 0.024, 0.026), T(pc.x, pc.y, hz) @ R, bevel=0.002)


def chain(b, m, z_from, z_to, x=0.0, y=0.0, pitch=0.0215):
    n = int((z_to - z_from) / pitch) + 1
    for i in range(n):
        z = z_from + i * pitch
        M = T(x, y, z) @ (Rz(pi / 2) if i % 2 else I4) @ Rx(pi / 2) @ S(1, 1.55, 1)
        b.torus(m, 0.0105, 0.0028, M, segs=16, rsegs=6)
    return n


def lerp(a, b_, t):
    return Vector(a) * (1 - t) + Vector(b_) * t


# ======================================================================
# 1. pendant_dome  -- origin: cord top attachment; hangs down (-Z blender / -Y godot)
# ======================================================================
def build_pendant_dome():
    reset()
    enamel = mat("enamel_white", (0.93, 0.93, 0.91), 0.0, 0.12, coat=1.0)
    steel = mat("brushed_steel", (0.75, 0.75, 0.77), 1.0, 0.32)
    cord_m = mat("fabric_cord_black", (0.012, 0.012, 0.012), 0.0, 0.95)
    ceramic = mat("socket_black", (0.02, 0.02, 0.02), 0.0, 0.4)
    emis = mat("bulb_emission", (1.0, 0.78, 0.4), 0.0, 0.4, emit=(1.0, 0.78, 0.4), emit_strength=3.0)
    b = Builder("pendant_dome")

    R, th = 0.396, 0.004
    apex = -1.03
    zc = apex - R
    # ceiling rose + cord
    b.cyl(enamel, 0.05, 0.026, T(0, 0, -0.026), segs=48, bevel=0.004, seg=3)
    b.cyl(cord_m, 0.0045, 1.0, T(0, 0, -1.0), segs=10)
    b.cyl(cord_m, 0.0075, 0.03, T(0, 0, -0.056), segs=10)   # strain relief at rose
    # collar on dome top
    b.lathe(steel, [(0.028, -1.03), (0.028, -0.995), (0.02, -0.972), (0.0075, -0.962)], segs=32)
    # hemispherical enamel shell (thick closed profile)
    arc = 18
    outer = [(R * sin(radians(90 * i / arc)), zc + R * cos(radians(90 * i / arc))) for i in range(arc + 1)]
    ri = R - th
    inner = [(ri * sin(radians(90 * i / arc)), zc + ri * cos(radians(90 * i / arc))) for i in range(arc, -1, -1)]
    b.lathe(enamel, outer + inner, segs=72)
    # thin metal rim band
    b.lathe(steel, [(0.389, zc - 0.007), (0.4, zc - 0.007), (0.4, zc + 0.009), (0.389, zc + 0.009)],
            segs=72, closed=True, bevel=0.0025, seg=2)
    # socket + bulb + diffuser disc
    b.cyl(ceramic, 0.03, 0.09, T(0, 0, apex + 0.0 - 0.09 + 0.004), segs=24, bevel=0.003)
    gc = apex - 0.2
    globe = [(0.058 * sin(radians(180 * i / 14)), gc + 0.058 * cos(radians(180 * i / 14))) for i in range(15)]
    b.lathe(emis, globe, segs=24)
    b.cyl(emis, 0.382, 0.004, T(0, 0, zc + 0.016), segs=72)   # warm inner bulb disc
    return b.finish("pendant_dome.glb")


# ======================================================================
# 2. pa_speaker_hanging -- origin: top of rigging; hangs down
# ======================================================================
def build_pa_speaker():
    reset()
    M_ = dict(
        paint=mat("pa_cabinet_paint", (0.014, 0.014, 0.016), 0.0, 0.62),
        grille=grille_mat(),
        rubber=mat("rubber_black", (0.02, 0.02, 0.02), 0.0, 0.85),
        steel=mat("zinc_steel", (0.55, 0.56, 0.58), 1.0, 0.38),
    )
    b = Builder("pa_speaker_hanging")
    b.uv(M_['grille'], 0.024)
    top = -0.40
    bottom = top - 0.65
    speaker_cabinet(b, M_, 0.40, 0.26, 0.40, bottom, top, bevel=0.012)
    steel = M_['steel']
    # rigging: top plate with two ears + pin, chain, top ring
    b.box(steel, (0.16, 0.10, 0.008), T(0, 0, top + 0.004), bevel=0.0015)
    for sx in (-1, 1):
        b.box(steel, (0.008, 0.06, 0.05), T(sx * 0.022, 0, top + 0.033), bevel=0.0015)
    b.rod(steel, (-0.04, 0, top + 0.040), (0.04, 0, top + 0.040), 0.0055, 16)
    b.rod(steel, (-0.045, 0, top + 0.040), (-0.035, 0, top + 0.040), 0.009, 16)
    n = chain(b, steel, top + 0.040, -0.062)
    b.torus(steel, 0.02, 0.0042, T(0, 0, -0.0242) @ Rx(pi / 2) @ S(1, 1.15, 1), segs=32, rsegs=8)
    # rubber feet on the bottom
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.cyl(M_['rubber'], 0.016, 0.01, T(sx * 0.11 * (1 if sy < 0 else 0.7), sy * 0.14, bottom - 0.008), segs=16)
    return b.finish("pa_speaker_hanging.glb")


# ======================================================================
# 3. spot_can -- origin at the clamp centre (pipe axis along X)
# ======================================================================
def build_spot_can():
    reset()
    body = mat("spot_black_powdercoat", (0.016, 0.016, 0.017), 0.35, 0.5)
    dark = mat("spot_dark_steel", (0.07, 0.07, 0.075), 0.9, 0.42)
    steel = mat("zinc_steel", (0.55, 0.56, 0.58), 1.0, 0.38)
    glass = mat("lens_glass", (0.08, 0.1, 0.12), 0.0, 0.03, coat=1.0, ior=1.5)
    rubber = mat("rubber_black", (0.02, 0.02, 0.02), 0.0, 0.85)
    b = Builder("spot_can")
    # clamp: collar around a 48 mm pipe whose axis is X, through the origin
    b.lathe(dark, [(0.0258, -0.022), (0.039, -0.022), (0.039, 0.022), (0.0258, 0.022)], Ry(pi / 2),
            segs=40, closed=True, bevel=0.003)
    b.box(dark, (0.06, 0.05, 0.034), T(0, 0, -0.045), bevel=0.004)
    b.rod(steel, (0, -0.06, -0.045), (0, 0.04, -0.045), 0.0065, 16)               # clamp bolt (Y)
    b.cyl(steel, 0.0125, 0.012, along((0, -0.062, -0.045), (0, -1, 0)), 6, bevel=0.0008)  # hex head
    b.cyl(rubber, 0.019, 0.022, along((0, 0.04, -0.045), (0, 1, 0)), 20, bevel=0.003)    # tightening knob
    b.cyl(dark, 0.012, 0.045, T(0, 0, -0.088), segs=20)                          # stem
    zb = -0.108                                                                   # yoke top bar bottom
    b.box(dark, (0.31, 0.04, 0.008), T(0, 0, zb + 0.004), bevel=0.0025)
    zp, tilt = -0.28, 30.0
    for sx in (-1, 1):
        b.box(dark, (0.01, 0.04, zb - zp), T(sx * 0.145, 0, (zb + zp) / 2), bevel=0.002)
        b.cyl(dark, 0.02, 0.01, along((sx * 0.145 - 0.005 * sx, 0, zp), (sx, 0, 0)), 24, bevel=0.002)
    # pivot knobs
    b.cyl(rubber, 0.022, 0.016, along((0.155, 0, zp), (1, 0, 0)), 8, bevel=0.003)
    b.cyl(steel, 0.008, 0.012, along((-0.15, 0, zp), (-1, 0, 0)), 6)
    # ---- can (local: axis +Z forward, z -0.2..0.2), then tilted down
    Mc = T(0, 0, zp) @ Rx(radians(90 + tilt))
    prof = [(0, -0.2), (0.10, -0.2), (0.118, -0.19), (0.125, -0.17), (0.125, 0.13), (0.117, 0.14),
            (0.117, 0.2), (0.108, 0.2), (0.108, 0.16), (0, 0.16)]
    b.lathe(body, prof, Mc, segs=48)
    for i in range(5):                                                            # cooling fins
        b.cyl(body, 0.129, 0.007, Mc @ T(0, 0, -0.155 + 0.026 * i), segs=48, bevel=0.0012)
    b.cyl(dark, 0.045, 0.018, Mc @ T(0, 0, -0.218), segs=24, bevel=0.003)         # rear cap
    for sx in (-1, 1):                                                            # trunnions
        b.cyl(dark, 0.02, 0.026, Mc @ along((sx * 0.124, 0, 0), (sx, 0, 0)), 20, bevel=0.002)
    # focus handle
    b.rod(dark, *[Mc @ Vector(p) for p in ((0, 0.125, 0.04), (0, 0.19, 0.04))], 0.005, 10)
    b.cyl(rubber, 0.012, 0.03, along(Mc @ Vector((0, 0.19, 0.04)), Mc.to_3x3() @ Vector((0, 1, 0))), 14, bevel=0.003)
    # lens
    lens = [(0, 0.19), (0.04, 0.186), (0.08, 0.172), (0.105, 0.16), (0, 0.16)]
    b.lathe(glass, lens, Mc, segs=40)
    # barn door ring + doors
    b.lathe(body, [(0.107, 0.2), (0.131, 0.2), (0.131, 0.214), (0.107, 0.214)], Mc, segs=48, closed=True, bevel=0.002)
    for k in range(4):
        Md = Mc @ Rz(k * pi / 2) @ T(0, 0.13, 0.208) @ Rx(radians(-52))
        b.box(body, (0.16, 0.003, 0.13), Md @ T(0, 0, 0.065), bevel=0.0012)
        b.rod(dark, *[Md @ Vector(p) for p in ((-0.08, 0, 0), (0.08, 0, 0))], 0.004, 8)
    return b.finish("spot_can.glb")


# ======================================================================
# 4. speaker_stand -- origin floor centre; total height 1.8 m
# ======================================================================
def build_speaker_stand():
    reset()
    M_ = dict(
        paint=mat("speaker_abs_black", (0.018, 0.018, 0.02), 0.0, 0.5),
        grille=grille_mat(),
        rubber=mat("rubber_black", (0.02, 0.02, 0.02), 0.0, 0.85),
        steel=mat("zinc_steel", (0.55, 0.56, 0.58), 1.0, 0.38),
    )
    tube = mat("stand_tube_black", (0.02, 0.02, 0.022), 0.85, 0.42)
    plastic = mat("plastic_black", (0.02, 0.02, 0.02), 0.0, 0.45)
    led = mat("led_green", (0.1, 1.0, 0.2), 0.0, 0.4, emit=(0.1, 1.0, 0.2), emit_strength=2.0)
    b = Builder("speaker_stand")
    b.uv(M_['grille'], 0.024)
    steel, rubber = M_['steel'], M_['rubber']
    hub_z = 0.78
    # centre pole: outer + telescoping inner, top at 1.30
    b.cyl(tube, 0.019, 0.55, T(0, 0, 0.40), segs=24)
    b.cyl(tube, 0.0155, 0.45, T(0, 0, 0.85), segs=24)
    b.cyl(plastic, 0.026, 0.045, T(0, 0, 0.88), segs=24, bevel=0.004)           # clamp collar
    b.rod(steel, (0.02, 0, 0.9), (0.05, 0, 0.9), 0.0045, 10)
    b.cyl(rubber, 0.017, 0.02, along((0.05, 0, 0.9), (1, 0, 0)), 16, bevel=0.003)
    b.cyl(plastic, 0.022, 0.014, T(0, 0, 1.286), segs=24)                        # pole cap
    # hub
    b.cyl(plastic, 0.038, 0.1, T(0, 0, hub_z - 0.05), segs=24, bevel=0.005)
    # legs (one back, two front)
    foot_r, foot_z = 0.46, 0.028
    cz = 0.22                                                                    # sliding collar height
    b.cyl(plastic, 0.027, 0.05, T(0, 0, cz - 0.025), segs=24, bevel=0.004)
    for ang in (90, 210, 330):
        a = radians(ang)
        dirv = Vector((cos(a), sin(a), 0))
        top = Vector((0.03 * cos(a), 0.03 * sin(a), hub_z + 0.015))
        foot = Vector((foot_r * cos(a), foot_r * sin(a), foot_z))
        b.rod(tube, top, foot, 0.0125, 16)
        b.cyl(rubber, 0.021, foot_z, T(foot.x, foot.y, 0), segs=16, bevel=0.004)
        # stay
        pl = lerp(top, foot, 0.55)
        b.rod(tube, (0.02 * cos(a), 0.02 * sin(a), cz + 0.01), pl, 0.0065, 10)
        b.rod(steel, pl - Vector((0, 0, 0.0)) , pl + dirv * 0.004, 0.011, 10)
    # ---- active speaker
    z0, z1 = 1.30, 1.80
    speaker_cabinet(b, M_, 0.35, 0.24, 0.35, z0, z1, grille_margin=0.015, bevel=0.012)
    y = 0.175
    # rear amp panel
    b.box(steel, (0.17, 0.006, 0.24), T(0, y - 0.001, z0 + 0.19), bevel=0.0015)
    for i in range(9):                                                           # heat-sink fins
        b.box(plastic, (0.15, 0.02, 0.0035), T(0, y + 0.012, z0 + 0.09 + i * 0.0075), bevel=0.0006)
    for i, x in enumerate((-0.045, 0.0, 0.045)):                                 # knobs
        b.cyl(plastic, 0.012, 0.016, along((x, y + 0.002, z0 + 0.265), (0, 1, 0)), 20, bevel=0.0018)
        b.box(steel, (0.002, 0.002, 0.008), T(x, y + 0.019, z0 + 0.268))
    for x in (-0.04, 0.04):                                                      # combo jacks
        b.lathe(steel, [(0.0105, 0), (0.0145, 0), (0.0145, 0.012), (0.0105, 0.012)],
                along((x, y + 0.002, z0 + 0.225), (0, 1, 0)), segs=24, closed=True)
        b.cyl(plastic, 0.0105, 0.004, along((x, y + 0.002, z0 + 0.225), (0, 1, 0)), 20)
    b.box(plastic, (0.034, 0.012, 0.022), T(0.045, y + 0.005, z0 + 0.145), bevel=0.002)   # IEC inlet
    b.cyl(led, 0.0035, 0.003, along((-0.055, y + 0.0035, z0 + 0.145), (0, 1, 0)), 10)
    # pole socket ring on the underside
    b.cyl(plastic, 0.03, 0.006, T(0, 0, z0 - 0.003), segs=24)
    return b.finish("speaker_stand.glb")


# ======================================================================
# 5. tripod_camera -- origin floor centre; tripod head plate at 1.40 m
# ======================================================================
def build_tripod_camera():
    reset()
    ali = mat("aluminium_black_anodised", (0.03, 0.03, 0.033), 0.9, 0.38)
    rubber = mat("rubber_black", (0.02, 0.02, 0.02), 0.0, 0.85)
    plastic = mat("plastic_black", (0.025, 0.025, 0.027), 0.0, 0.45)
    grey = mat("plastic_dark_grey", (0.09, 0.09, 0.1), 0.0, 0.5)
    steel = mat("zinc_steel", (0.55, 0.56, 0.58), 1.0, 0.38)
    glass = mat("lens_glass", (0.06, 0.08, 0.12), 0.0, 0.03, coat=1.0, ior=1.5)
    lcd = mat("lcd_screen", (0.01, 0.015, 0.02), 0.0, 0.08, coat=1.0)
    red = mat("rec_led", (1.0, 0.05, 0.03), 0.0, 0.4, emit=(1.0, 0.05, 0.03), emit_strength=2.0)
    foam = mat("mic_foam", (0.03, 0.03, 0.03), 0.0, 1.0)
    b = Builder("tripod_camera")
    hub_top = 1.40
    # hub + head plate
    b.cyl(plastic, 0.058, 0.085, T(0, 0, 1.30), segs=36, bevel=0.006)
    b.cyl(ali, 0.064, 0.016, T(0, 0, 1.384), segs=36, bevel=0.003)
    # legs: one front (-Y), two rear
    top_r, top_z = 0.05, 1.30
    foot_r, foot_z = 0.58, 0.02
    pts_mid = []
    for ang in (270, 30, 150):
        a = radians(ang)
        ux = Vector((cos(a), sin(a), 0))
        top = Vector((top_r * cos(a), top_r * sin(a), top_z))
        foot = Vector((foot_r * cos(a), foot_r * sin(a), foot_z))
        P = lambda t: lerp(top, foot, t)
        dirv = (foot - top).normalized()
        b.box(plastic, (0.03, 0.04, 0.05), T(top.x + ux.x * 0.012, top.y + ux.y * 0.012, top_z + 0.01) @ Rz(a), bevel=0.004)  # hinge lug
        b.rod(ali, P(0.0), P(0.46), 0.0165, 16)
        b.rod(ali, P(0.42), P(0.74), 0.0135, 16)
        b.rod(ali, P(0.70), P(1.0), 0.0108, 16)
        for t, r_ in ((0.46, 0.0215), (0.74, 0.018)):                              # lock collars
            b.rod(plastic, P(t) - dirv * 0.02, P(t) + dirv * 0.02, r_, 16, bevel=0.0025)
            lp = P(t) + ux * (r_ + 0.012)
            b.box(plastic, (0.028, 0.012, 0.01), T(lp) @ Rz(a), bevel=0.002)   # flip lever
        b.cyl(rubber, 0.0205, foot_z, T(foot.x, foot.y, 0), segs=16, bevel=0.004)
        pts_mid.append(P(0.70))
        b.cyl(plastic, 0.012, 0.03, along(P(0.70), (-ux.x, -ux.y, 0)), 12)  # spreader lug
    for i in range(3):                                                           # triangular spreader
        p0, p1 = pts_mid[i], pts_mid[(i + 1) % 3]
        b.rod(ali, p0, p1, 0.0065, 10)
    # fluid head
    b.cyl(plastic, 0.052, 0.028, T(0, 0, 1.40), segs=36, bevel=0.004)             # pan base
    b.cyl(grey, 0.0525, 0.004, T(0, 0, 1.4125), segs=36)                          # degree ring
    b.box(plastic, (0.075, 0.09, 0.06), T(0, 0, 1.458), bevel=0.01, seg=3)        # head body
    b.cyl(plastic, 0.03, 0.115, along((-0.0575, 0, 1.478), (1, 0, 0)), 24, bevel=0.003)  # tilt housing
    b.box(ali, (0.06, 0.14, 0.012), T(0, 0, 1.494), bevel=0.0015)                  # camera plate
    for sx in (-1, 1):
        b.cyl(rubber, 0.013, 0.018, along((sx * 0.058, 0.0, 1.478), (sx, 0, 0)), 14, bevel=0.0025)
    # pan handle (to the rear/down) + grip
    hd = Vector((0, 0.9, -0.3)).normalized()
    hp0 = Vector((0, 0.045, 1.45))
    b.cyl(plastic, 0.016, 0.03, along(hp0, hd), 16, bevel=0.003)
    b.rod(ali, hp0, hp0 + hd * 0.36, 0.0085, 14)
    b.rod(rubber, hp0 + hd * 0.22, hp0 + hd * 0.36, 0.0145, 16, bevel=0.003)
    # ---- video camera (front = -Y)
    zc = 1.5 + 0.05
    b.box(plastic, (0.085, 0.20, 0.10), T(0, 0.0, zc), bevel=0.009, seg=3)
    b.box(grey, (0.07, 0.07, 0.03), T(0, 0.115, zc - 0.03), bevel=0.004)           # battery
    b.box(ali, (0.07, 0.16, 0.006), T(0, -0.01, 1.503), bevel=0.001)               # mount plate
    # lens barrel
    ly = -0.10
    b.rod(plastic, (0, ly, zc + 0.005), (0, ly - 0.09, zc + 0.005), 0.036, 32, bevel=0.003)
    b.rod(rubber, (0, ly - 0.02, zc + 0.005), (0, ly - 0.06, zc + 0.005), 0.0385, 32, bevel=0.002)  # focus ring
    b.lathe(plastic, [(0.043, 0.0), (0.049, 0.0), (0.049, 0.04), (0.043, 0.04)],
            along((0, ly - 0.09, zc + 0.005), (0, -1, 0)), segs=32, closed=True, bevel=0.002)      # hood
    b.cyl(glass, 0.033, 0.006, along((0, ly - 0.093, zc + 0.005), (0, -1, 0)), 32, r2=0.03)
    b.cyl(glass, 0.02, 0.002, along((0, ly - 0.0995, zc + 0.005), (0, -1, 0)), 24)
    # top handle with mic
    for yy in (-0.06, 0.04):
        b.box(plastic, (0.016, 0.014, 0.035), T(0, yy, zc + 0.0675), bevel=0.003)
    b.box(plastic, (0.02, 0.13, 0.014), T(0, -0.01, zc + 0.092), bevel=0.005)
    b.rod(foam, (0, -0.15, zc + 0.108), (0, -0.04, zc + 0.108), 0.014, 16, bevel=0.003)
    b.cyl(plastic, 0.004, 0.003, along((0.0, -0.065, zc + 0.099), (0, 0, 1)), 10)
    b.cyl(red, 0.0045, 0.003, along((0.02, -0.095, zc + 0.05), (0, 0, 1)), 10)
    # viewfinder + LCD
    b.rod(plastic, (-0.015, 0.1, zc + 0.045), (-0.015, 0.17, zc + 0.045), 0.02, 20, bevel=0.002)
    b.rod(rubber, (-0.015, 0.16, zc + 0.045), (-0.015, 0.185, zc + 0.045), 0.0245, 20, bevel=0.003)
    b.box(plastic, (0.008, 0.095, 0.07), T(-0.0495, -0.0, zc + 0.0), bevel=0.003)   # LCD housing
    b.box(lcd, (0.002, 0.085, 0.06), T(-0.0545, 0.0, zc + 0.0), bevel=0.0004)
    for k in range(3):                                                           # side buttons
        b.cyl(grey, 0.005, 0.004, along((0.0425, -0.05 + k * 0.03, zc + 0.03), (1, 0, 0)), 10)
    return b.finish("tripod_camera.glb")


BUILDERS = {
    "pendant_dome": build_pendant_dome,
    "pa_speaker_hanging": build_pa_speaker,
    "spot_can": build_spot_can,
    "speaker_stand": build_speaker_stand,
    "tripod_camera": build_tripod_camera,
}

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for n in (argv or list(BUILDERS)):
        BUILDERS[n]()
