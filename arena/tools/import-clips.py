#!/usr/bin/env python3
"""Pack clips generated with NVIDIA Kimodo (G1) into the compact format the viewer plays.

Input : clip JSONs written by convert_kimodo_to_viewer.py ({fps, frames:[{joints:{name:rad}, pelvis:{pos, quat}}]})
Output: public/animations/<name>.json  {name, fps, duration, joints:[names], frames:[[29 angles...]], root:[[x,y,z,qx,qy,qz,qw]...], lift:[m...]}
        public/animations/index.json   {<viewer clip name>: <file>}
`lift` is how far both soles are off the ground (jumps); the viewer's foot-contact logic would cancel it otherwise.

    python3 tools/import-clips.py <dir with sit_down.json jump.json wave.json> [--rename sit_down=sit]
Needs numpy + scipy. Model: nvidia/Kimodo-G1-RP-v1 (NVIDIA Open Model License).
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent.parent
meta = json.load(open(HERE / "public/g1/metadata.json"))
bodies = {b["name"]: b for b in meta["bodies"]}
children = {}
for b in meta["bodies"]:
    children.setdefault(b["parent"], []).append(b["name"])


def ankle_heights(joints, ppos, pquat):
    out = {}

    def rec(name, pos, rot):
        b = bodies[name]
        if b["parent"] is None:
            pos = np.array(ppos, float) + np.array(b["restPosition"])
            rot = R.from_quat(pquat) * R.from_quat(b["restQuaternion"])
        else:
            pos = pos + rot.apply(b["restPosition"])
            rot = rot * R.from_quat(b["restQuaternion"])
        for j in b.get("joints", []):
            rot = rot * R.from_rotvec(np.array(j["axis"]) * joints.get(j["name"], 0.0))
        if name.endswith("ankle_roll_link"):
            out[name] = pos[2]
        for c in children.get(name, []):
            rec(c, pos, rot)

    rec("pelvis", None, None)
    return min(out.values())


def main():
    src = Path(sys.argv[1])
    renames = dict(a.split("=") for a in sys.argv[2:] if "=" in a and not a.startswith("--"))
    outdir = HERE / "public/animations"
    outdir.mkdir(exist_ok=True)
    index = {}
    for f in sorted(src.glob("*.json")):
        if f.name == "manifest.json":
            continue
        c = json.load(open(f))
        names = list(c["frames"][0]["joints"])
        low = [ankle_heights(fr["joints"], fr["pelvis"]["pos"], fr["pelvis"]["quat"]) for fr in c["frames"]]
        base = min(low)
        lift = [round(max(h - base - 0.015, 0.0), 3) for h in low]
        name = renames.get(c.get("name", f.stem), c.get("name", f.stem))
        packed = {
            "name": name, "fps": c["fps"], "duration": round(len(c["frames"]) / c["fps"], 3), "joints": names,
            "frames": [[round(fr["joints"][n], 3) for n in names] for fr in c["frames"]],
            "root": [[*[round(v, 3) for v in fr["pelvis"]["pos"]], *[round(v, 4) for v in fr["pelvis"]["quat"]]] for fr in c["frames"]],
            "lift": lift,
        }
        (outdir / f"{name}.json").write_text(json.dumps(packed, separators=(",", ":")))
        index[name] = f"{name}.json"
        print(name, len(c["frames"]), "frames, max lift", max(lift))
    (outdir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
