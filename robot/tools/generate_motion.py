# /// script
# requires-python = ">=3.12"
# dependencies = ["modal==1.6.0", "numpy==1.26.4"]
# ///
"""One ephemeral, bounded Kimodo G1 generation using the existing Kimodo image.

Run: uv run tools/generate_motion.py
No deployment. G1 download happens in a CPU image-build layer before GPU start.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import time

import modal

BASE_IMAGE_ID = "im-xmXj0bi1K12m9XRdNzSury"
SOURCE_REVISION = "1aece8c124d73d255ceff5086d983b844c9f4e94"
MODEL_REVISION = "3020ad8c419c244e0429d360163730c63c4ed011"
MODEL_REPO = "nvidia/Kimodo-G1-RP-v1"
MODEL_KEY = "kimodo-g1-rp"
PROMPT = "a humanoid robot stands still, reaches forward with its right hand to pick up a coffee cup from a table, then gently lifts the cup towards its chest and holds it"
SEED = 42
FRAMES = 240
STEPS = 100
OUTPUT = Path(__file__).resolve().parents[1] / "public/assets/motion"

download_code = f'''import hashlib,json,urllib.request
from pathlib import Path
repo={MODEL_REPO!r}
rev={MODEL_REVISION!r}
dst=Path('/opt/kimodo-checkpoints/Kimodo-G1-RP-v1')
files=['config.yaml','model.safetensors','LICENSE','README.md','stats/motion/body/mean.npy','stats/motion/body/std.npy','stats/motion/global_root/mean.npy','stats/motion/global_root/std.npy','stats/motion/local_root/mean.npy','stats/motion/local_root/std.npy']
manifest={{'repository':repo,'revision':rev,'files':{{}}}}
for name in files:
    path=dst/name
    path.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(f'https://huggingface.co/{{repo}}/resolve/{{rev}}/{{name}}',path)
    manifest['files'][name]={{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}}
    print('downloaded',name,path.stat().st_size,flush=True)
(dst/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
'''

download_b64 = base64.b64encode(download_code.encode()).decode()
image = modal.Image.from_id(BASE_IMAGE_ID).run_commands("python -c " + shlex.quote("import base64;exec(base64.b64decode(" + repr(download_b64) + "))"))
app = modal.App("g1-coffee-demo-one-clip-20261011")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768,
              timeout=900, startup_timeout=900, scaledown_window=5,
              single_use_containers=True, max_containers=1)
def generate() -> dict:
    import tempfile
    import numpy as np
    import torch
    from kimodo import load_model
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34
    from kimodo.tools import seed_everything

    started = time.monotonic()
    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    seed_everything(SEED)
    model, resolved = load_model(MODEL_KEY, device="cuda:0", return_resolved_name=True)
    assert resolved == MODEL_KEY
    assert float(model.fps) == 30
    loaded = time.monotonic()
    output = model([PROMPT], [FRAMES], constraint_lst=[], num_denoising_steps=STEPS,
                   num_samples=1, multi_prompt=True, num_transition_frames=5,
                   post_processing=True, return_numpy=True)
    generated = time.monotonic()
    single = {k: v[0] if hasattr(v, "shape") and len(v.shape) and v.shape[0] == 1 else v
              for k, v in output.items()}
    assert single["posed_joints"].shape == (FRAMES, 34, 3)
    for value in single.values():
        if hasattr(value, "dtype"):
            assert np.isfinite(value).all()
    qpos = converter.dict_to_qpos(single, device="cpu", mujoco_rest_zero=False)
    if qpos.ndim == 3:
        qpos = qpos[0]
    import xml.etree.ElementTree as ET
    xml = ET.parse(converter.xml_path)
    mujoco_joint_names = [j.attrib["name"] for j in xml.findall(".//worldbody//joint") if j.attrib.get("type") != "free"]
    assert qpos.shape == (FRAMES, 36)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coffee.npz"
        save_kimodo_npz(str(path), single)
        data = path.read_bytes()
    skeleton_data = {
        "joint_names": skeleton.bone_order_names,
        "parent_indices": skeleton.joint_parents.tolist(),
        "neutral_joints": skeleton.neutral_joints.tolist(),
        "global_rot_offsets": skeleton.global_rot_offsets.tolist() if hasattr(skeleton, "global_rot_offsets") else None,
        "rest_pose_local_rot": skeleton.rest_pose_local_rot.tolist(),
        "mujoco_joint_names": mujoco_joint_names,
        "mujoco_qpos": qpos.tolist(),
    }
    receipt = {
        "status": "generated", "provider": "Kimodo on Modal", "model": MODEL_REPO,
        "modelRevision": MODEL_REVISION, "sourceRevision": SOURCE_REVISION,
        "baseImageId": BASE_IMAGE_ID, "modalImageId": os.environ.get("MODAL_IMAGE_ID"),
        "modalTaskId": os.environ.get("MODAL_TASK_ID"), "seed": SEED,
        "prompt": PROMPT, "numFrames": FRAMES, "fps": 30, "diffusionSteps": STEPS,
        "postProcessing": True, "loadSeconds": round(loaded-started, 3),
        "generationSeconds": round(generated-loaded, 3),
        "npzSha256": hashlib.sha256(data).hexdigest(), "npzBytes": len(data),
        "arrays": {k: list(v.shape) for k,v in single.items() if hasattr(v,"shape")},
        "cupMotion": "Not generated: Kimodo only generates robot body motion. Cup placement, attachment and any contact correction belong to the demo interaction layer.",
        "qposConvention": "Official Kimodo MujocoQposConverter, mujoco_rest_zero=False; z-up x-forward; root xyz then root quaternion wxyz then 29 hinge angles.",
        "checkpointDownload": json.loads(Path('/opt/kimodo-checkpoints/Kimodo-G1-RP-v1/download-manifest.json').read_text()),
        "gpu": torch.cuda.get_device_name(),
    }
    return {"npz": data, "skeleton": skeleton_data, "receipt": receipt}


def main():
    import numpy as np
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "coffee.npz").exists():
        raise SystemExit("A clip already exists; refusing another paid generation")
    with modal.enable_output():
        with app.run():
            result = generate.remote()
    data = result["npz"]
    (OUTPUT / "coffee.npz").write_bytes(data)
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        arrays = {k: npz[k].tolist() for k in npz.files}
    document = {
        "format": "kimodo-g1-browser-motion-v1", "fps": 30, "numFrames": FRAMES,
        "durationSeconds": FRAMES / 30, "frame_times": [i/30 for i in range(FRAMES)],
        "coordinateSystem": "Kimodo: right-handed y-up z-forward",
        **result["skeleton"], **arrays, "provenance": result["receipt"],
    }
    (OUTPUT / "coffee.json").write_text(json.dumps(document, separators=(",", ":")))
    (OUTPUT / "receipt.json").write_text(json.dumps(result["receipt"], indent=2) + "\n")
    print(json.dumps({k:v for k,v in result["receipt"].items() if k not in ("checkpointDownload", "arrays")}, indent=2))


if __name__ == "__main__":
    main()
