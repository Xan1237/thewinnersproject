"""Inspect exported GLBs without Blender: bbox, tri count, orientation (pure python).
Run: python3 seating_verify.py a.glb b.glb ...
Orientation: chairs must have their backrest on glTF +Z (front toward -Z = Godot forward).
"""
import json, struct, sys


def load(path):
    d = open(path, "rb").read()
    jl, _ = struct.unpack_from("<II", d, 12)
    js = json.loads(d[20:20 + jl])
    bo = 20 + jl
    bl, _ = struct.unpack_from("<II", d, bo)
    return js, d[bo + 8: bo + 8 + bl]


def verts(js, bin_, acc):
    a = js["accessors"][acc]
    bv = js["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 12)
    return [struct.unpack_from("<fff", bin_, off + i * stride) for i in range(a["count"])]


for p in sys.argv[1:]:
    js, b = load(p)
    pts, tris = [], 0
    for m in js["meshes"]:
        for pr in m["primitives"]:
            pts += verts(js, b, pr["attributes"]["POSITION"])
            tris += js["accessors"][pr["indices"]]["count"] // 3
    xs, ys, zs = zip(*pts)
    H = max(ys)
    top = [q for q in pts if q[1] > 0.6 * H]
    mz = sum(q[2] for q in top) / len(top)
    lo = [q for q in pts if q[1] < 0.02]
    print("%s: tris=%d  size(X,Y,Z)=%.3f x %.3f x %.3f  min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f)" %
          (p.split("/")[-1], tris, max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs),
           min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)))
    print("   upper-body mean Z=%.3f (%s), feet centre x=%.3f z=%.3f, materials=%s, images=%d" %
          (mz, "back at +Z => front faces -Z OK" if mz > 0 else "WRONG: back at -Z",
           (min(q[0] for q in lo) + max(q[0] for q in lo)) / 2, (min(q[2] for q in lo) + max(q[2] for q in lo)) / 2,
           [m["name"] for m in js.get("materials", [])], len(js.get("images", []))))
