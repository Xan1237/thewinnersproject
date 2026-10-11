#!/usr/bin/env python3
"""Convert official G1 link meshes to an articulated GLB without loading pickle.

Run with Python 3 + numpy. Geometry remains in native MuJoCo coordinates:
meters, X forward, Y left, Z up. XML body nodes are preserved, and official
Unitree articulated hands replace the XML rubber hand visuals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

KIMODO_COMMIT = "58e781898b3d7e328a676a75d3e338c45dce3ad9"
UNITREE_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
KIMODO_BASE = f"https://raw.githubusercontent.com/nv-tlabs/kimodo/{KIMODO_COMMIT}/"
UNITREE_BASE = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{UNITREE_COMMIT}/"
ASSET_BASE = "kimodo/assets/skeletons/g1skel34/"
OUT = Path(__file__).resolve().parents[1] / "public/assets/g1"
CACHE = Path("/tmp/g1-coffee-assets")


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def vec(value, default):
    return [float(x) for x in value.split()] if value else list(default)


def quat(value):
    w, x, y, z = vec(value, [1, 0, 0, 0])
    q = np.array([x, y, z, w], dtype=np.float64)
    return (q / np.linalg.norm(q)).tolist()


def rpy_quat(value):
    r, p, y = np.array(vec(value, [0, 0, 0])) / 2
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def matrix(position, quaternion):
    x, y, z, w = quaternion
    m = np.eye(4)
    m[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    m[:3, 3] = position
    return m


def read_stl(path):
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    positions, indices = np.unique(records["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    face_normals = np.cross(positions[faces[:, 1]]-positions[faces[:, 0]],
                            positions[faces[:, 2]]-positions[faces[:, 0]])
    valid = np.linalg.norm(face_normals, axis=1) > 1e-12
    faces, face_normals = faces[valid], face_normals[valid]
    normals = np.zeros_like(positions)
    for column in range(3): np.add.at(normals, faces[:, column], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return positions.astype("<f4"), normals.astype("<f4"), faces.astype("<u4")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = fetch(KIMODO_BASE+ASSET_BASE+"xml/g1.xml", OUT/"g1.xml")
    urdf_path = fetch(UNITREE_BASE+"robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf", OUT/"hands-source.urdf")
    fetch(KIMODO_BASE+"LICENSE", OUT/"LICENSE-KIMODO.txt")
    fetch(UNITREE_BASE+"LICENSE", OUT/"LICENSE-UNITREE.txt")
    xml, urdf = ET.parse(xml_path).getroot(), ET.parse(urdf_path).getroot()
    classes = {}
    def defaults(el, inherited):
        properties = {key: dict(value) for key, value in inherited.items()}
        for child in el:
            if child.tag != "default": properties.setdefault(child.tag, {}).update(child.attrib)
        if el.get("class"): classes[el.get("class")] = properties
        for child in el.findall("default"): defaults(child, properties)
    defaults(xml.find("default"), {})
    meshes = {e.get("name"): e.get("file") for e in xml.findall("asset/mesh")}
    bodies = []
    def parse_body(body, parent):
        name = body.get("name")
        record = {"name": name, "parent": parent,
                  "restPosition": vec(body.get("pos"), [0,0,0]),
                  "restQuaternion": quat(body.get("quat")), "geometries": [], "joints": []}
        for joint in body.findall("joint"):
            p = {**classes.get(joint.get("class"), {}).get("joint", {}), **joint.attrib}
            record["joints"].append({"name": p["name"], "axis": vec(p.get("axis"), [0,0,1]),
                                     "limits": vec(p.get("range"), [-3.14159,3.14159]),
                                     "position": vec(p.get("pos"), [0,0,0]), "type": p.get("type","hinge")})
        seen = set()
        for geom in body.findall("geom"):
            p = {**classes.get(geom.get("class"), {}).get("geom", {}), **geom.attrib}
            if p.get("type") != "mesh" or "mesh" not in p: continue
            if "rubber_hand" in p["mesh"]: continue
            key = (p["mesh"], p.get("pos"), p.get("quat"), p.get("material"))
            if key in seen: continue
            seen.add(key)
            record["geometries"].append({"name": p["mesh"], "file": meshes[p["mesh"]],
                                           "position": vec(p.get("pos"), [0,0,0]),
                                           "quaternion": quat(p.get("quat")), "material": p.get("material", "silver")})
        bodies.append(record)
        for child in body.findall("body"): parse_body(child, name)
    for body in xml.findall("worldbody/body"): parse_body(body, None)
    # Real optional 7-DOF dexterous hands from Unitree's supplied URDF.
    links = {link.get("name"): link for link in urdf.findall("link")}
    for joint in urdf.findall("joint"):
        name = joint.find("child").get("link")
        if "_hand_" not in name: continue
        origin = joint.find("origin"); limit = joint.find("limit"); visual = links[name].find("visual")
        mesh = visual.find("geometry/mesh").get("filename").split("/")[-1]
        vo = visual.find("origin")
        record = {"name": name, "parent": joint.find("parent").get("link"),
                  "restPosition": vec(origin.get("xyz"),[0,0,0]),
                  "restQuaternion": rpy_quat(origin.get("rpy")), "joints": [],
                  "geometries": [{"name": name, "file": mesh, "position": vec(vo.get("xyz"),[0,0,0]),
                                  "quaternion": rpy_quat(vo.get("rpy")), "material": "silver"}]}
        if joint.get("type") != "fixed":
            record["joints"].append({"name": joint.get("name"), "axis": vec(joint.find("axis").get("xyz"),[0,0,1]),
                                     "limits": [float(limit.get("lower")),float(limit.get("upper"))], "position": [0,0,0], "type": "hinge"})
        bodies.append(record)
    required = sorted({g["file"] for body in bodies for g in body["geometries"]})
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda file: fetch(KIMODO_BASE+ASSET_BASE+"meshes/g1/"+file, CACHE/KIMODO_COMMIT/file), required))
    gltf = {"asset": {"version":"2.0", "generator":"G1 coffee demo / build_g1.py"},
            "scene":0,"scenes":[{"nodes":[0]}],"nodes":[],"meshes":[],"accessors":[],"bufferViews":[],
            "materials":[{"name":"silver","pbrMetallicRoughness":{"baseColorFactor":[.65,.68,.71,1],"metallicFactor":.7,"roughnessFactor":.38}},
                         {"name":"black","pbrMetallicRoughness":{"baseColorFactor":[.045,.05,.06,1],"metallicFactor":.25,"roughnessFactor":.42}}]}
    buffer = bytearray(); total_triangles = 0; geometry_cache = {}; vertices_cache = {}
    def accessor(data, component, value_type, target):
        while len(buffer)%4: buffer.append(0)
        offset = len(buffer); content = data.tobytes(); buffer.extend(content)
        view = len(gltf["bufferViews"]); gltf["bufferViews"].append({"buffer":0,"byteOffset":offset,"byteLength":len(content),"target":target})
        a = {"bufferView":view,"componentType":component,"count":len(data),"type":value_type}
        if value_type == "VEC3": a.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
        index = len(gltf["accessors"]); gltf["accessors"].append(a); return index
    for file in required:
        positions, normals, faces = read_stl(CACHE/KIMODO_COMMIT/file)
        vertices_cache[file] = positions
        total_triangles += len(faces)
        geometry_cache[file] = {"attributes":{"POSITION":accessor(positions,5126,"VEC3",34962),"NORMAL":accessor(normals,5126,"VEC3",34962)},
                                "indices":accessor(faces.reshape(-1),5125,"SCALAR",34963)}
    name_to_node = {}
    for body in bodies:
        name_to_node[body["name"]] = len(gltf["nodes"])
        gltf["nodes"].append({"name":body["name"],"translation":body["restPosition"],"rotation":body["restQuaternion"],"children":[]})
    transforms = {}; all_world_positions = []
    for body in bodies:
        node = gltf["nodes"][name_to_node[body["name"]]]
        parent = body["parent"]
        if parent: gltf["nodes"][name_to_node[parent]]["children"].append(name_to_node[body["name"]])
        world = transforms.get(parent,np.eye(4)) @ matrix(body["restPosition"],body["restQuaternion"])
        transforms[body["name"]] = world
        body["restWorldMatrix"] = world.T.reshape(-1).tolist()
        for geom in body["geometries"]:
            mi = len(gltf["meshes"])
            primitive = {**geometry_cache[geom["file"]], "material": 1 if geom["material"] == "black" else 0}
            gltf["meshes"].append({"name":geom["name"],"primitives":[primitive]})
            node["children"].append(len(gltf["nodes"]))
            gltf["nodes"].append({"name":geom["name"]+"_visual","translation":geom["position"],"rotation":geom["quaternion"],"mesh":mi})
            gm = world @ matrix(geom["position"],geom["quaternion"])
            v = vertices_cache[geom["file"]]
            all_world_positions.append(v @ gm[:3,:3].T+gm[:3,3])
    bounds = np.vstack(all_world_positions)
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    assert 1.0 < hi[2]-lo[2] < 1.8, (lo,hi)
    gltf["buffers"] = [{"byteLength":len(buffer)}]
    content = json.dumps(gltf,separators=(",",":")).encode()
    content += b" "*((-len(content))%4)
    buffer.extend(b"\0"*((-len(buffer))%4))
    glb = struct.pack("<III",0x46546C67,2,12+8+len(content)+8+len(buffer))
    glb += struct.pack("<II",len(content),0x4E4F534A)+content+struct.pack("<II",len(buffer),0x004E4942)+buffer
    (OUT/"g1.glb").write_bytes(glb)
    metadata = {"schema":1,"model":"Unitree G1 29-DOF body + official 7-DOF hands",
                "coordinateSystem":"MuJoCo native: meters, X forward, Y left, Z up",
                "quaternionOrder":"xyzw","matrixOrder":"column-major",
                "nativeToThreeYUp":[0,0,1,1,0,0,0,1,0],
                "root":"pelvis","bodies":bodies,"joints":{j["name"]:{**j,"body":b["name"],"restQuaternion":b["restQuaternion"]} for b in bodies for j in b["joints"]},
                "bounds":{"min":lo.tolist(),"max":hi.tolist(),"size":(hi-lo).tolist()},
                "triangleCount":total_triangles,"glbBytes":len(glb),
                "graspSites":{"right":{"body":"right_hand_palm_link","localPosition":[.105,.025,0],"approximate":True},
                              "left":{"body":"left_hand_palm_link","localPosition":[.105,-.025,0],"approximate":True}},
                "source":{"kimodoCommit":KIMODO_COMMIT,"unitreeRosCommit":UNITREE_COMMIT,
                          "kimodoURL":"https://github.com/nv-tlabs/kimodo", "unitreeURL":"https://github.com/unitreerobotics/unitree_ros",
                          "meshFiles":[{"file":f,"sha256":hashlib.sha256((CACHE/KIMODO_COMMIT/f).read_bytes()).hexdigest()} for f in required]}}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (OUT/"PROVENANCE.md").write_text(f"""# Unitree G1 model provenance

Body hierarchy and all STL meshes: nv-tlabs/kimodo at `{KIMODO_COMMIT}`,
`kimodo/assets/skeletons/g1skel34`. Repository LICENSE copied alongside model.
Kimodo's repository license is Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES.

Articulated hand transforms/joint limits: unitreerobotics/unitree_ros at
`{UNITREE_COMMIT}`, `robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`.
Repository LICENSE copied alongside model. Meshes are from Kimodo's bundle.
Unitree's repository license is BSD-3-Clause, Copyright 2016-2022 HangZhou YuShu
TECHNOLOGY CO.,LTD. (Unitree Robotics).

Changes: binary STL geometry converted to indexed GLB with averaged vertex normals;
duplicate visual geoms removed; default rubber hand visuals replaced by the supplied
official palm/thumb/index/middle link geometry and its URDF joint hierarchy.
No newly invented robot mesh geometry. Mesh hashes in metadata.json identify inputs.

Model stays in MuJoCo coordinates (meters, X forward, Y left, Z up), root pelvis
translation 0,0,0.793. GLB quaternions use xyzw. Joint rotations compose as
restQuaternion * axisAngle(axis, q). Body geometry transforms remain separate.
`restWorldMatrix` is column-major. For Three Y-up use native (x,y,z) -> (y,z,x).

Grasp sites are explicitly approximate visualization targets; they are not measured
physical contact points. This is a visual animation asset, not a hardware controller.
""")
    print(json.dumps({"bodies":len(bodies),"joints":len(metadata["joints"]),"triangles":total_triangles,"bytes":len(glb),"bounds":metadata["bounds"]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
