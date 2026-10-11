"""Shared helpers for the seating_*.py Blender scripts (run: blender -b --python seating_xxx.py).

Conventions: Blender Z-up, chair FRONT faces +Y.  glTF export (+Y up) maps Blender +Y -> glTF -Z,
which is Godot's forward (-Z).  Origin = floor centre of the footprint, units = metres.
"""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector, Matrix, Euler

OUT_DIR = "/Users/sukhov/Projects/thewinnersproject/assets/models"


# ----------------------------------------------------------------------------- scene / colour
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def s2l(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


# ----------------------------------------------------------------------------- textures
def _noise(n, rng, sx=1.0, sy=1.0, p=1.0):
    """Tileable filtered noise, unit std.  sx/sy scale the frequency per axis (small = long features)."""
    w = rng.standard_normal((n, n))
    F = np.fft.fft2(w)
    fy = np.fft.fftfreq(n)[:, None] * n * sy
    fx = np.fft.fftfreq(n)[None, :] * n * sx
    f = np.sqrt(fx ** 2 + fy ** 2) + 1.0
    F = F / f ** p
    r = np.real(np.fft.ifft2(F))
    return (r - r.mean()) / (r.std() + 1e-9)


def _make_image(name, arr, srgb=True):
    h, w, _ = arr.shape
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr, 0, 1)
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"   # must precede pixel upload
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def weave_height(n=512, k=64, seed=1):
    rng = np.random.default_rng(seed)
    u = (np.arange(n) / n)[None, :] * k
    v = (np.arange(n) / n)[:, None] * k
    chk = ((np.floor(u) + np.floor(v)) % 2) > 0
    hu = np.sin(np.pi * (u % 1.0)) ** 0.8
    hv = np.sin(np.pi * (v % 1.0)) ** 0.8
    h = np.where(chk, hu, hv) * (0.75 + 0.25 * np.cos(2 * np.pi * np.where(chk, v, u)))
    h = h + 0.10 * _noise(n, rng, 4, 4, 0.6)   # slub / irregularity
    return h


def fabric_images(name, color, seed=1, n=512):
    """albedo (colour baked in, weave-modulated) + weave normal map."""
    rng = np.random.default_rng(seed)
    h = weave_height(n, 64, seed)
    hn = (h - h.min()) / (h.max() - h.min())
    mott = 0.04 * _noise(n, rng, 6, 6, 1.0)
    val = 0.82 + 0.30 * (hn - 0.5) + mott
    alb = np.array(color)[None, None, :] * val[..., None]
    img_a = _make_image(name + "_albedo", alb, True)
    s = 3.0
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * n / 64.0
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * n / 64.0
    nx, ny, nz = -dx * s * 0.25, -dy * s * 0.25, np.ones_like(h)
    L = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nm = np.stack([nx / L, ny / L, nz / L], -1) * 0.5 + 0.5
    img_n = _make_image(name + "_normal", nm, False)
    return img_a, img_n


def wood_images(name, color, seed=3, n=512):
    """Wood albedo; grain runs along image V (up the leg)."""
    rng = np.random.default_rng(seed)
    streak = _noise(n, rng, 1.0, 0.04, 1.0)
    fine = _noise(n, rng, 1.0, 0.02, 0.4)
    warp = _noise(n, rng, 0.25, 0.03, 1.2)
    u = (np.arange(n) / n)[None, :]
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 7.0 + 0.35 * warp))
    pores = _noise(n, rng, 1.0, 0.3, 0.2)
    val = 0.80 + 0.10 * streak + 0.05 * fine + 0.12 * (rings - 0.5) + 0.02 * pores
    alb = np.array(color)[None, None, :] * val[..., None] * np.array([1.0, 0.96, 0.92])[None, None, :]
    return _make_image(name + "_albedo", alb, True)


# ----------------------------------------------------------------------------- materials
def make_material(name, color, rough, metal=0.0, albedo=None, normal=None, nstr=0.5, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*s2l(color), 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if sheen > 0:
        b.inputs["Sheen Weight"].default_value = sheen
        b.inputs["Sheen Tint"].default_value = (*s2l(tuple(min(1, c * 1.6 + 0.05) for c in color)), 1.0)
        b.inputs["Sheen Roughness"].default_value = 0.5
    if albedo is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = albedo
        t.location = (-600, 200)
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if normal is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = normal
        t.location = (-600, -150)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = nstr
        nm.location = (-300, -150)
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


def fabric_material(name, color, seed=1, rough=0.92):
    a, n = fabric_images(name, color, seed)
    return make_material(name, color, rough, 0.0, a, n, 0.6, sheen=0.12)


def wood_material(name, color, seed=3, rough=0.55):
    a = wood_images(name, color, seed)
    return make_material(name, color, rough, 0.0, a)


# ----------------------------------------------------------------------------- mesh helpers
def new_obj(name, bm, mat):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    return ob


def apply_mods(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    for md in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=md.name)


def add_bevel(ob, width, seg=3, angle=40):
    md = ob.modifiers.new("bevel", "BEVEL")
    md.width = width
    md.segments = seg
    md.limit_method = "ANGLE"
    md.angle_limit = math.radians(angle)
    return md


def box_uv(ob, scale=0.3):
    """World-space box projection UVs (1 UV unit = `scale` metres)."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    uv = bm.loops.layers.uv.verify()
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for l in f.loops:
            c = l.vert.co
            if ax == 0:
                u, v = c.y, c.z
            elif ax == 1:
                u, v = c.x, c.z
            else:
                u, v = c.x, c.y
            l[uv].uv = (u / scale, v / scale)
    bm.to_mesh(me)
    bm.free()


def tube(name, mat, pts, radii, n=16, flat_start=False, flat_end=False, bevel=0.0025, uv_scale=0.3):
    """Swept round tube along polyline `pts` with per-point radius; flat caps.
    flat_start/flat_end: end rings lie in a horizontal plane (so feet sit flat on the floor)."""
    pts = [Vector(p) for p in pts]
    m = len(pts)
    tans = []
    for i in range(m):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == m - 1:
            t = pts[-1] - pts[-2]
        else:
            t = (pts[i] - pts[i - 1]).normalized() + (pts[i + 1] - pts[i]).normalized()
        tans.append(t.normalized())
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    ref = Vector((1, 0, 0)) if abs(tans[0].x) < 0.9 else Vector((0, 1, 0))
    u = (ref - tans[0] * ref.dot(tans[0])).normalized()
    rings, frames, cum = [], [], [0.0]
    for i in range(m):
        t = tans[i]
        u = (u - t * u.dot(t)).normalized()
        v = t.cross(u)
        r = radii[i]
        if i > 0 and i < m - 1:
            cs = tans[i].dot((pts[i] - pts[i - 1]).normalized())
            r = r / max(cs, 0.6)
        uu, vv = u, v
        if (i == 0 and flat_start) or (i == m - 1 and flat_end):
            uu, vv = Vector((1, 0, 0)), Vector((0, 1, 0))
        ring = [bm.verts.new(pts[i] + (uu * math.cos(2 * math.pi * j / n) + vv * math.sin(2 * math.pi * j / n)) * r)
                for j in range(n)]
        rings.append(ring)
        if i:
            cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    for i in range(m - 1):
        for j in range(n):
            j2 = (j + 1) % n
            f = bm.faces.new([rings[i][j], rings[i][j2], rings[i + 1][j2], rings[i + 1][j]])
            circ = 2 * math.pi * radii[i]
            ua, ub = j / n * circ, (j + 1) / n * circ
            vals = [(ua, cum[i]), (ub, cum[i]), (ub, cum[i + 1]), (ua, cum[i + 1])]
            for l, (a, b) in zip(f.loops, vals):
                l[uvl].uv = (a / uv_scale, b / uv_scale)
    for ring in (rings[0], rings[-1]):
        f = bm.faces.new(ring if ring is rings[0] else ring[::-1])
        for l in f.loops:
            c = l.vert.co
            l[uvl].uv = (c.x / uv_scale, c.y / uv_scale)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = new_obj(name, bm, mat)
    if bevel:
        add_bevel(ob, bevel, 2, 40)
        apply_mods(ob)
    return ob


def superblock(name, mat, center, size, n=4.0, cuts=10, scale_uv=0.3, rot_z=0.0):
    """Pillowy block (superellipsoid mapped from a subdivided cube) -> soft upholstered volume."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    for v in bm.verts:
        p = v.co
        ch = max(abs(p.x), abs(p.y), abs(p.z))
        nn = (abs(p.x) ** n + abs(p.y) ** n + abs(p.z) ** n) ** (1.0 / n)
        s = ch / nn if nn > 1e-9 else 1.0
        v.co = Vector((p.x * s * size[0] / 2, p.y * s * size[1] / 2, p.z * s * size[2] / 2))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = new_obj(name, bm, mat)
    ob.location = center
    ob.rotation_euler = (0, 0, rot_z)
    apply_transform(ob)
    box_uv(ob, scale_uv)
    return ob


def bevel_box(name, mat, center, size, bevel=0.006, seg=3, rot=(0, 0, 0), scale_uv=0.3):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    ob = new_obj(name, bm, mat)
    ob.location = center
    ob.rotation_euler = rot
    apply_transform(ob)
    add_bevel(ob, bevel, seg, 40)
    apply_mods(ob)
    box_uv(ob, scale_uv)
    return ob


def apply_transform(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def pad(name, mat, w, h, loc, rot=(0, 0, 0), n=3.5, nu=26, nv=26, thick=0.05, bulge=0.012,
        arc=0.0, egg=0.0, bevel=None, scale_uv=0.3):
    """Upholstered pad.  Built in local XZ plane (u=X, v=Z), front normal +Y, puffed toward the front,
    optionally bent into a vertical arc (concave toward +Y), then solidified backwards + bevelled.
    `rot` (Euler XYZ) and `loc` place it in the world."""
    bm = bmesh.new()
    verts = {}
    R = w / arc if arc > 1e-6 else None
    for i in range(nu + 1):
        for j in range(nv + 1):
            gu, gv = -1 + 2 * i / nu, -1 + 2 * j / nv
            ch = max(abs(gu), abs(gv))
            nn = (abs(gu) ** n + abs(gv) ** n) ** (1 / n)
            s = ch / nn if nn > 1e-9 else 1.0
            x = gu * s * w / 2
            z = gv * s * h / 2
            x *= 1 + egg * (z / (h / 2))
            wv = bulge * (1 - ch ** 2) if bulge else 0.0
            if R:
                phi = x / R
                r = R - wv
                X, Y = r * math.sin(phi), R - r * math.cos(phi)
            else:
                X, Y = x, wv
            verts[(i, j)] = bm.verts.new((X, Y, z))
    for i in range(nu):
        for j in range(nv):
            bm.faces.new([verts[(i, j)], verts[(i, j + 1)], verts[(i + 1, j + 1)], verts[(i + 1, j)]])
    M = Matrix.Translation(loc) @ Euler(rot, "XYZ").to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=M, verts=bm.verts)
    ob = new_obj(name, bm, mat)
    sm = ob.modifiers.new("solid", "SOLIDIFY")
    sm.thickness = thick
    sm.offset = -1.0
    sm.use_even_offset = True
    add_bevel(ob, bevel if bevel else thick * 0.38, 3, 40)
    apply_mods(ob)
    box_uv(ob, scale_uv)
    return ob


# ----------------------------------------------------------------------------- finish / export
def finalize(objs, name, out_path=None, foot_band=0.02):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = objs[0]
    for o in objs:
        o.select_set(True)
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    zmin = min(v.co.z for v in bm.verts)
    feet = [v.co for v in bm.verts if v.co.z < zmin + foot_band]
    cx = (min(p.x for p in feet) + max(p.x for p in feet)) / 2
    cy = (min(p.y for p in feet) + max(p.y for p in feet)) / 2
    bmesh.ops.translate(bm, vec=(-cx, -cy, -zmin), verts=bm.verts)
    bm.to_mesh(ob.data)
    bm.free()
    ob.location = (0, 0, 0)
    ob.rotation_euler = (0, 0, 0)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(50))
    ob.data.update()
    tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    xs = [v.co.x for v in ob.data.vertices]
    ys = [v.co.y for v in ob.data.vertices]
    zs = [v.co.z for v in ob.data.vertices]
    print("MODEL", name, "tris", tris, "bbox_blender X %.3f Y %.3f Z %.3f" %
          (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    out_path = out_path or os.path.join(OUT_DIR, name + ".glb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB", export_yup=True, export_apply=True,
                              use_selection=True, export_image_format="AUTO")
    print("EXPORTED", out_path)
    return ob
