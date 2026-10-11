"""Architecture props for the hackathon stage room (visuals only, no collisions).

Run:  blender -b --python tools/blender/architecture.py [-- window_curtain curtain_panel exposed_beam cable_tray duct_round]
No args = build all. Outputs GLB (Y-up, metres, modifiers applied) to assets/models/.

Authoring space is Blender Z-up; glTF export converts to Y-up:  Godot (x, y, z) = Blender (x, z, -y).
  -> Blender -Y is Godot +Z (room interior side), Blender +Z is Godot +Y (up).
Long props (beam / tray / duct) run along X.
"""
import bpy, bmesh, math, os, sys, random
import numpy as np
from mathutils import Vector, Matrix, noise

OUT = '/Users/sukhov/Projects/thewinnersproject/assets/models'
TEX = '/Users/sukhov/.hermes/cache/scratch/arch_tex'
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEX, exist_ok=True)


# ----------------------------------------------------------------- helpers
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def srgb(*c):
    f = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return tuple(f(v) for v in c)


def hexc(h):
    h = h.lstrip('#')
    return srgb(*[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def fnoise(n, beta, seed, fx_cut=None, fy_cut=None):
    """Seamless 1/f^beta noise, unit std. Optional anisotropic band limit (cycles/px)."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    F *= f ** (-beta / 2)
    if fx_cut:
        F *= np.exp(-(fx / fx_cut) ** 2)
    if fy_cut:
        F *= np.exp(-(fy / fy_cut) ** 2)
    F[0, 0] = 0
    o = np.fft.ifft2(F).real
    return o / (o.std() + 1e-9)


def make_image(name, rgb, nonc=False):
    h, w, _ = rgb.shape
    a = np.ones((h, w, 4), np.float32)
    a[..., :3] = np.clip(rgb, 0, 1)
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.pixels.foreach_set(a.ravel())
    img.filepath_raw = os.path.join(TEX, name + '.png')
    img.file_format = 'PNG'
    img.save()
    if nonc:
        img.colorspace_settings.name = 'Non-Color'
    img.pack()
    return img


def normal_from_height(h, k):
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    n = np.stack([-gx * k, -gy * k, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def make_mat(name, color, rough=0.5, metal=0.0, sheen=0.0, albedo=None, normal=None, nstrength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if sheen > 0:
        b.inputs['Sheen Weight'].default_value = sheen
        b.inputs['Sheen Roughness'].default_value = 0.35
    if albedo is not None:
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = albedo
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    if normal is not None:
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = normal
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nm.inputs['Strength'].default_value = nstrength
        nt.links.new(t.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    return m


def autosmooth(bm, angle=35):
    lim = math.radians(angle)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0) > lim:
            e.smooth = False
        elif len(e.link_faces) != 2:
            e.smooth = False


def finish(bm, name, mat, angle=35, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    autosmooth(bm, angle)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def bake(ob):
    """Apply all modifiers of ob into its mesh."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)


def box(bm, c, s, bevel=0.0):
    m = Matrix.LocRotScale(Vector(c), None, Vector(s))
    r = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    bv = min(bevel, 0.4 * min(s))
    if bv > 0:
        edges = list({e for v in r['verts'] for e in v.link_edges})
        bmesh.ops.bevel(bm, geom=r['verts'] + edges, offset=bv, segments=1, affect='EDGES')


def cyl(bm, p0, p1, r, segs=16, r2=None):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    M = Matrix.Translation((p0 + p1) / 2) @ Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r,
                          radius2=r if r2 is None else r2, depth=d.length, matrix=M)


def sphere(bm, c, r, u=12, v=8):
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r, matrix=Matrix.Translation(c))


def tube(bm, pts, radius, sides=8, closed=False, normals=None, cap=True):
    pts = [Vector(p) for p in pts]
    n = len(pts)
    T = []
    for i in range(n):
        t = (pts[(i + 1) % n] - pts[i - 1]) if closed else (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
        T.append(t.normalized())
    if normals is None:
        a = Vector((0, 0, 1)) if abs(T[0].z) < 0.9 else Vector((1, 0, 0))
        Ns = [(a - T[0] * a.dot(T[0])).normalized()]
        for i in range(1, n):
            p = Ns[-1]
            Ns.append((p - T[i] * p.dot(T[i])).normalized())
    else:
        Ns = [(Vector(x) - T[i] * Vector(x).dot(T[i])).normalized() for i, x in enumerate(normals)]
    rings = []
    for i in range(n):
        N = Ns[i]
        B = T[i].cross(N)
        r = radius[i] if hasattr(radius, '__len__') else radius
        rings.append([bm.verts.new(pts[i] + r * (math.cos(2 * math.pi * k / sides) * N + math.sin(2 * math.pi * k / sides) * B))
                      for k in range(sides)])
    for i in range(n if closed else n - 1):
        a, b = rings[i], rings[(i + 1) % n]
        for k in range(sides):
            k2 = (k + 1) % sides
            bm.faces.new([a[k], b[k], b[k2], a[k2]])
    if cap and not closed:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])


def lathe_x(bm, profile, segs=64, uscale=1.0, vscale=1.0, closed=True):
    """Revolve (x, r) profile around the X axis. Closed profile -> watertight shell. UVs: u = x*uscale, v = angle*vscale."""
    uv = bm.loops.layers.uv.verify()
    rings = []
    for (x, r) in profile:
        rings.append([bm.verts.new((x, r * math.cos(2 * math.pi * k / segs), r * math.sin(2 * math.pi * k / segs)))
                      for k in range(segs)])
    n = len(profile)
    for i in range(n if closed else n - 1):
        i2 = (i + 1) % n
        for k in range(segs):
            k2 = (k + 1) % segs
            f = bm.faces.new([rings[i][k], rings[i2][k], rings[i2][k2], rings[i][k2]])
            uvs = [(profile[i][0] * uscale, k * vscale / segs), (profile[i2][0] * uscale, k * vscale / segs),
                   (profile[i2][0] * uscale, (k + 1) * vscale / segs), (profile[i][0] * uscale, (k + 1) * vscale / segs)]
            for l, u in zip(f.loops, uvs):
                l[uv].uv = u


def tri_count(obs):
    t = 0
    for o in obs:
        me = o.data
        me.calc_loop_triangles()
        t += len(me.loop_triangles)
    return t


def export(name, obs=None):
    for o in list(bpy.context.scene.objects):
        o.select_set(False)
    path = os.path.join(OUT, name + '.glb')
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_yup=True, export_apply=True,
                              export_materials='EXPORT', use_selection=False)
    allobs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    print(f'EXPORTED {path}: {len(allobs)} meshes, {tri_count(allobs)} tris')
    return path


# ----------------------------------------------------------------- materials
def mat_black_metal(name='black_metal'):
    return make_mat(name, srgb(0.035, 0.035, 0.04), rough=0.42, metal=0.7)


def mat_velvet(name='curtain_velvet'):
    return make_mat(name, srgb(0.035, 0.035, 0.04), rough=0.9, metal=0.0, sheen=0.3)


def mat_cyan_paint(name='reveal_cyan_paint'):
    return make_mat(name, hexc('#a6e3ea'), rough=0.62)


# ----------------------------------------------------------------- curtain
def curtain_bm(width, height, waves, amp, nx, nz, seed=1.0, header=0.07):
    """Pleated curtain surface. Hangs from z=0 (top) to z=-height, centred on x=0, folds displace in Y."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    V = []
    for j in range(nz + 1):
        s = j / nz
        row = []
        for i in range(nx + 1):
            u = i / nx
            ph = 2 * math.pi * waves * (u + 0.012 * math.sin(5 * s + 9 * u + seed))
            fold = math.sin(ph) + 0.22 * math.sin(2 * ph + 0.8 + seed)
            hdr = max(0.0, 1 - (s * height) / header)           # heading tape: regular, tight, flat top
            a = amp * (0.9 + 0.3 * s * s)                       # folds open up toward the hem
            a *= 1 + 0.18 * math.sin(u * 17.3 + 1.1 + seed) + 0.10 * math.sin(u * 41 + 2)
            a *= 1 - 0.25 * hdr
            x = (u - 0.5) * width + 0.012 * math.cos(ph) * (0.6 + s)
            y = a * fold + 0.0025 * noise.noise(Vector((u * 9, s * 5, seed)))
            z = -s * height
            # hem sags slightly between folds
            z -= 0.012 * s ** 4 * (0.5 + 0.5 * math.sin(ph + 1.7))
            row.append(bm.verts.new((x, y, z)))
        V.append(row)
    for j in range(nz):
        for i in range(nx):
            f = bm.faces.new([V[j][i], V[j][i + 1], V[j + 1][i + 1], V[j + 1][i]])
            for l, uv in zip(f.loops, [(i / nx, 1 - j / nz), ((i + 1) / nx, 1 - j / nz),
                                       ((i + 1) / nx, 1 - (j + 1) / nz), (i / nx, 1 - (j + 1) / nz)]):
                l[uvl].uv = uv
    return bm


def curtain_object(name, mat, width, height, waves, amp, nx, nz, offset=(0, 0, 0), seed=1.0, thick=0.007):
    bm = curtain_bm(width, height, waves, amp, nx, nz, seed)
    bmesh.ops.translate(bm, vec=offset, verts=bm.verts)
    # make faces face -Y (room side in window space)
    for f in bm.faces:
        if f.normal.y > 0:
            pass
    ob = finish(bm, name, mat, angle=60, recalc=False)
    mod = ob.modifiers.new('solid', 'SOLIDIFY')
    mod.thickness = thick
    mod.offset = 0
    mod.use_even_offset = True
    bake(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


# ----------------------------------------------------------------- 1. window + curtain
def build_window_curtain():
    reset()
    black = mat_black_metal('window_frame_black')
    paint = mat_cyan_paint()
    sillm = make_mat('sill_ledge', hexc('#c9d6d8'), rough=0.45, metal=0.0)
    velvet = mat_velvet()
    rail_m = make_mat('curtain_rail_black', srgb(0.03, 0.03, 0.035), rough=0.35, metal=0.85)

    OW, OH, RD = 2.8, 2.6, 0.25      # opening width, height, reveal depth
    hw = OW / 2

    # --- painted deep reveal (4 liner slabs outside the opening, y 0..+0.25) + wall-plane-flush
    bm = bmesh.new()
    t = 0.02
    box(bm, (-hw - t / 2, RD / 2, OH / 2 + t / 2), (t, RD, OH + t))              # left jamb
    box(bm, (hw + t / 2, RD / 2, OH / 2 + t / 2), (t, RD, OH + t))               # right jamb
    box(bm, (0, RD / 2, OH + t / 2), (OW + 2 * t, RD, t))                         # head
    box(bm, (0, RD / 2, -0.001), (OW, RD, 0.002))                                 # sill board under the ledge
    finish(bm, 'reveal', paint, angle=30)

    # --- ledge sill: projects 0.05 into the room, runs 0.1 past reveal each side
    bm = bmesh.new()
    box(bm, (0, (RD - 0.05) / 2, -0.02), (OW + 0.2, RD + 0.05, 0.04), bevel=0.004)
    finish(bm, 'sill', sillm, angle=30)

    # --- thin black metal frame inside the reveal; glass plane y=+0.145 (Godot z=-0.145)
    bm = bmesh.new()
    fy0, fd = 0.11, 0.07            # frame front face y, depth
    cy = fy0 + fd / 2
    fw = 0.05                        # face width of perimeter members
    mw = 0.04                        # mullion face width
    box(bm, (-hw + fw / 2, cy, OH / 2), (fw, fd, OH), 0.002)
    box(bm, (hw - fw / 2, cy, OH / 2), (fw, fd, OH), 0.002)
    box(bm, (0, cy, OH - fw / 2), (OW - 2 * fw, fd, fw), 0.002)
    box(bm, (0, cy, fw / 2), (OW - 2 * fw, fd, fw), 0.002)
    inner_w = OW - 2 * fw
    mx = [-inner_w / 6, inner_w / 6]
    for x in mx:
        box(bm, (x, cy, OH / 2), (mw, fd * 0.9, OH - 2 * fw), 0.0015)
    # glazing beads (raised rebate lip on the room side) around each of the 3 panes
    bead_w, bead_d = 0.012, 0.016
    xs = [-hw + fw] + [x - mw / 2 for x in mx[:1]] + [mx[0] + mw / 2, mx[1] - mw / 2, mx[1] + mw / 2, hw - fw]
    panes = [(xs[0], xs[1]), (xs[2], xs[3]), (xs[4], xs[5])]
    z0, z1 = fw, OH - fw
    by = fy0 - bead_d / 2 + 0.002
    for a, b in panes:
        box(bm, (a + bead_w / 2, by, (z0 + z1) / 2), (bead_w, bead_d, z1 - z0), 0.001)
        box(bm, (b - bead_w / 2, by, (z0 + z1) / 2), (bead_w, bead_d, z1 - z0), 0.001)
        box(bm, ((a + b) / 2, by, z0 + bead_w / 2), (b - a - 2 * bead_w, bead_d, bead_w), 0.001)
        box(bm, ((a + b) / 2, by, z1 - bead_w / 2), (b - a - 2 * bead_w, bead_d, bead_w), 0.001)
    finish(bm, 'frame', black, angle=30)

    # --- curtain rail (ceiling-height), brackets, finials
    RY, RZ = -0.14, 2.72
    rx0, rx1 = -1.85, 1.85
    bm = bmesh.new()
    cyl(bm, (rx0, RY, RZ), (rx1, RY, RZ), 0.0125, 16)
    sphere(bm, (rx0 - 0.02, RY, RZ), 0.024)
    sphere(bm, (rx1 + 0.02, RY, RZ), 0.024)
    for bx in (-1.7, -0.45, 0.8, 1.6):
        cyl(bm, (bx, 0.0, RZ), (bx, RY, RZ), 0.006, 8)
        box(bm, (bx, 0.004, RZ), (0.05, 0.008, 0.07), 0.001)
        box(bm, (bx, RY + 0.002, RZ - 0.012), (0.03, 0.008, 0.012))
    # hooks / rings on each fold of the left panel
    PX0, PW = -0.95, 1.6
    waves = 10
    for i in range(waves + 1):
        x = PX0 - PW / 2 + PW * i / waves
        R = 0.02
        pts = [(x, RY + R * math.cos(a), RZ + R * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]
        nrm = [(0, math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]
        tube(bm, pts, 0.0028, 5, closed=True, normals=nrm)
    finish(bm, 'curtain_rail', rail_m, angle=35)

    # --- velvet curtain: 1.6 wide, top z=2.70, bottom z=-0.20 (2.9 drape; floor is at z=-0.30 => 0.1 gap)
    curtain_object('curtain', velvet, PW, 2.9, waves, 0.042, 96, 34, offset=(PX0, RY, 2.70), seed=1.3)
    return export('window_curtain')


# ----------------------------------------------------------------- 2. curtain panel
def build_curtain_panel():
    reset()
    velvet = mat_velvet()
    curtain_object('curtain_panel', velvet, 1.2, 2.9, 8, 0.042, 72, 34, offset=(0, 0, 0), seed=0.4)
    return export('curtain_panel')


# ----------------------------------------------------------------- 3. concrete beam
def concrete_textures():
    n = 1024
    low = fnoise(n, 3.0, 11)
    mid = fnoise(n, 2.0, 12)
    fine = fnoise(n, 0.8, 13)
    grain = fnoise(n, 1.0, 14, fx_cut=0.004, fy_cut=0.12)     # long streaks along the length (image x)
    pore = fnoise(n, 0.2, 15)
    stain = fnoise(n, 3.6, 16)
    pores = np.clip((pore - 2.4) / 1.0, 0, 1)
    shade = 1 + 0.07 * low + 0.05 * mid + 0.035 * fine + 0.035 * grain - 0.09 * np.clip(stain, 0, None)
    shade *= 1 - 0.45 * pores
    base = np.array(hexc('#7d7b77'))
    alb = base[None, None, :] * shade[..., None]
    alb[..., 2] *= 0.985
    h = 1.0 * mid + 0.7 * fine + 0.5 * grain - 3.0 * pores
    nrm = normal_from_height(h, 0.55)
    return make_image('concrete_albedo', alb), make_image('concrete_normal', nrm, nonc=True)


def build_exposed_beam():
    reset()
    alb, nrm = concrete_textures()
    m = make_mat('rough_concrete', hexc('#7d7b77'), rough=0.92, albedo=alb, normal=nrm, nstrength=1.0)

    L, W, H = 4.0, 0.45, 0.6
    hw, hh = W / 2, H / 2
    c = 0.012
    ring = []   # dicts (y, z, ny, nz, groove)

    def side(p0, p1, nrm_, seams, step):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        ln = d.length
        dv = d / ln
        gw = 0.0032
        ts = {c, ln - c}
        for k in range(1, int((ln - 2 * c) / step)):
            t = c + k * step
            if all(abs(t - s) > 0.012 for s in seams):
                ts.add(round(t, 5))
        gr = {}
        for s in seams:
            for t, g in ((s - gw, 0.0), (s, -0.0028), (s + gw, 0.0)):
                ts.add(round(t, 5))
                gr[round(t, 5)] = g
        for t in sorted(ts):
            p = p0 + dv * t
            ring.append((p.x, p.y, nrm_[0], nrm_[1], gr.get(round(t, 5), 0.0)))

    side((-hw, -hh), (hw, -hh), (0, -1), [], 0.075)                       # soffit
    side((hw, -hh), (hw, hh), (1, 0), [0.15, 0.30, 0.45], 0.06)           # +Y face, boards 150mm
    side((hw, hh), (-hw, hh), (0, 1), [], 0.09)                           # top
    side((-hw, hh), (-hw, -hh), (-1, 0), [0.15, 0.30, 0.45], 0.06)        # -Y face
    # side() puts x=y-coord... values: (ring: y,z,...) ; ring tuple order is (y, z, ny, nz, groove)
    np_ = len(ring)
    arc = [0.0]
    for i in range(1, np_ + 1):
        a, b = ring[i - 1], ring[i % np_]
        arc.append(arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))

    xs = set(np.round(np.linspace(-L / 2, L / 2, 101), 5))
    joints = [-1.37, 1.07]
    jg = {}
    for xj in joints:
        for dx, g in ((-0.003, 0.0), (0.0, -0.0025), (0.003, 0.0)):
            xs.add(round(xj + dx, 5))
            jg[round(xj + dx, 5)] = g
    xs = sorted(xs)

    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    seed = Vector((3.1, 7.7, 1.9))
    V = []
    for x in xs:
        col = []
        for (y, z, ny, nz, g) in ring:
            d = g + jg.get(round(x, 5), 0.0)
            d += 0.0014 * noise.noise(Vector((x * 1.3, y * 3.0, z * 3.0)) + seed)
            d += 0.0007 * noise.noise(Vector((x * 7.0, y * 17.0, z * 17.0)) + seed * 2)
            d += 0.0025 * math.sin(x * 1.7 + (y + z) * 2.0)
            col.append(bm.verts.new((x, y + ny * d, z + nz * d)))
        V.append(col)
    for j in range(len(xs) - 1):
        for i in range(np_):
            i2 = (i + 1) % np_
            f = bm.faces.new([V[j][i], V[j + 1][i], V[j + 1][i2], V[j][i2]])
            u0, u1 = xs[j] / 2.0, xs[j + 1] / 2.0
            v0, v1 = arc[i] / 2.0, arc[i + 1] / 2.0
            for l, uv in zip(f.loops, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]):
                l[uvl].uv = uv
    for col in (V[0], V[-1]):
        f = bm.faces.new(col)
        for l in f.loops:
            l[uvl].uv = (l.vert.co.y * 0.5, l.vert.co.z * 0.5)
    ob = finish(bm, 'exposed_beam', m, angle=40)
    return export('exposed_beam')


# ----------------------------------------------------------------- 4. cable tray
def cable_path(x0, x1, y, z0, wob, ph, loop=None, n=110):
    """Cable lying in the tray, gentle sideways wobble, optional flat service loop (cx, R, r, dir)."""
    xs = np.linspace(x0, x1, n)
    pts = [Vector((x, y + wob * math.sin(x * 1.3 + ph) + 0.5 * wob * math.sin(x * 3.7 + 2 * ph), z0)) for x in xs]
    if loop:
        cx, R, r, dr = loop
        i0 = int(np.searchsorted(xs, cx))
        yc = pts[i0].y
        new = pts[:i0]
        for t in np.linspace(0, 2 * math.pi, 48):
            lift = 2.1 * r * (0.5 - 0.5 * math.cos(t / 2))      # rises so the cable crosses over itself
            new.append(Vector((cx + R * math.sin(t), yc + dr * (R - R * math.cos(t)), z0 + lift)))
        for p in pts[i0 + 1:]:
            q = p.copy()
            q.z = z0 + 2.1 * r * max(0.0, 1 - (p.x - cx) / 0.5)
            new.append(q)
        pts = new
    return pts


def build_cable_tray():
    reset()
    black = make_mat('tray_black_powdercoat', srgb(0.03, 0.03, 0.035), rough=0.5, metal=0.55)
    pvc = make_mat('cable_pvc_black', srgb(0.02, 0.02, 0.022), rough=0.42, metal=0.0)
    L, W, H, t = 4.0, 0.3, 0.1, 0.002
    zb = -H / 2

    # perforated base: plate + slot cutters (boolean applied)
    bm = bmesh.new()
    box(bm, (0, 0, zb + t / 2), (L, W - 2 * t, t))
    base = finish(bm, 'tray_base', black, angle=30)
    cm = bmesh.new()
    pitch = 0.075
    xs = np.arange(-L / 2 + 0.06, L / 2 - 0.05, pitch)
    for x in xs:
        for y in (-0.085, 0.0, 0.085):
            box(cm, (x, y, zb + t / 2), (0.055, 0.016, 0.02))
    cutter = finish(cm, 'cutter', black, recalc=False)
    mod = base.modifiers.new('perf', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter
    mod.solver = 'EXACT'
    bake(base)
    bpy.data.objects.remove(cutter)
    for p in base.data.polygons:
        p.use_smooth = False

    # side rails with return lips, splice plate + bolts
    bm = bmesh.new()
    for s in (-1, 1):
        box(bm, (0, s * (W / 2 - t / 2), 0), (L, t, H))
        box(bm, (0, s * (W / 2 - 0.009), H / 2 - t / 2), (L, 0.018, t))
    # joint cover at x = 0.3
    jx = 0.3
    box(bm, (jx, 0, zb - 0.0015), (0.20, W + 0.004, 0.003))
    for s in (-1, 1):
        box(bm, (jx, s * (W / 2 + 0.003), 0), (0.20, 0.003, H))
        for dx in (-0.07, 0.07):
            for dz in (-0.025, 0.025):
                cyl(bm, (jx + dx, s * (W / 2 + 0.0035), dz), (jx + dx, s * (W / 2 + 0.0065), dz), 0.0055, 6)
    finish(bm, 'tray_rails', black, angle=30)

    # cables
    bm = bmesh.new()
    specs = [
        (-1.96, 1.96, -0.105, 0.008, 0.006, 0.0, (-0.45, 0.032, 0.008, 1)),
        (-1.96, 1.96, -0.035, 0.0055, 0.006, 1.7, None),
        (-1.96, 1.96, 0.022, 0.006, 0.006, 3.1, None),
        (-1.96, 1.96, 0.095, 0.0095, 0.005, 4.4, (0.95, 0.032, 0.0095, -1)),
    ]
    for (x0, x1, y, r, wob, ph, loop) in specs:
        z0 = zb + t + r
        pts = cable_path(x0, x1, y, z0, wob, ph, loop)
        tube(bm, pts, r, 6)
    finish(bm, 'cables', pvc, angle=60)
    return export('cable_tray')


# ----------------------------------------------------------------- 5. HVAC duct
def galv_textures():
    n = 512
    rng = np.random.default_rng(7)
    warp = fnoise(n, 1.5, 21)
    yy, xx = np.mgrid[0:n, 0:n]
    gx = ((xx + warp * 14) / 8).astype(int) % 64
    gy = ((yy + fnoise(n, 1.5, 22) * 14) / 8).astype(int) % 64
    cells = rng.random((64, 64))
    spangle = cells[gy, gx]
    streak = fnoise(n, 1.0, 23, fx_cut=0.01, fy_cut=0.15)
    blotch = fnoise(n, 3.0, 24)
    v = 0.60 + 0.07 * (spangle - 0.5) + 0.02 * streak + 0.03 * blotch
    alb = np.stack([v * 0.97, v * 1.0, v * 1.03], -1)
    h = 0.4 * spangle + 0.1 * streak
    return make_image('galv_albedo', alb), make_image('galv_normal', normal_from_height(h * 6, 0.5), nonc=True)


def build_duct_round():
    reset()
    alb, nrm = galv_textures()
    galv = make_mat('galvanised_steel', (0.6, 0.62, 0.64), rough=0.38, metal=0.92, albedo=alb, normal=nrm, nstrength=0.35)
    R, L, th = 0.2, 4.0, 0.0012
    prof = []
    # outer surface from -L/2 to L/2 with slip-joint collars at x=-1,0,1
    prof.append((-L / 2, R))
    for xj in (-1.0, 0.0, 1.0):
        prof += [(xj - 0.095, R), (xj - 0.091, R + 0.004), (xj - 0.087, R),            # stiffening bead
                 (xj - 0.045, R), (xj - 0.040, R + 0.013), (xj + 0.040, R + 0.013), (xj + 0.045, R),  # collar
                 (xj + 0.087, R), (xj + 0.091, R + 0.004), (xj + 0.095, R)]
    prof.append((L / 2, R))
    prof += [(L / 2, R - th), (-L / 2, R - th)]
    bm = bmesh.new()
    lathe_x(bm, prof, 64, uscale=1.0, vscale=R * 2 * math.pi, closed=True)
    finish(bm, 'duct_body', galv, angle=30)

    # hanger straps
    strap = make_mat('hanger_strap_galv', (0.55, 0.57, 0.59), rough=0.45, metal=0.9)
    bm = bmesh.new()
    sw, st = 0.026, 0.0018
    for hx in (-1.5, 1.5):
        lathe_x(bm, [(hx - sw / 2, R + 0.0003), (hx - sw / 2, R + 0.0003 + st), (hx + sw / 2, R + 0.0003 + st), (hx + sw / 2, R + 0.0003)], 64, closed=True)
        rr = R + 0.0003 + st / 2
        top = R + 0.0003 + 0.30
        for s in (-1, 1):
            box(bm, (hx, s * rr, top / 2 - 0.0), (sw, st, top), 0.0005)
        box(bm, (hx, 0, top + st / 2), (sw, 2 * rr + st, st), 0.0005)           # cross strap at the top
        cyl(bm, (hx, 0, top), (hx, 0, top + 0.16), 0.0045, 8)                    # threaded rod stub
        cyl(bm, (hx, 0, top + 0.004), (hx, 0, top + 0.012), 0.009, 6)           # nut
        # bolt through both legs
        bz = 0.16
        cyl(bm, (hx, -rr - 0.012, bz), (hx, rr + 0.012, bz), 0.004, 8)
        for s in (-1, 1):
            cyl(bm, (hx, s * (rr + 0.003), bz), (hx, s * (rr + 0.011), bz), 0.0075, 6)
    finish(bm, 'hanger_straps', strap, angle=30)
    return export('duct_round')


BUILDERS = {
    'window_curtain': build_window_curtain,
    'curtain_panel': build_curtain_panel,
    'exposed_beam': build_exposed_beam,
    'cable_tray': build_cable_tray,
    'duct_round': build_duct_round,
}

if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    for k in (args or list(BUILDERS)):
        BUILDERS[k]()
