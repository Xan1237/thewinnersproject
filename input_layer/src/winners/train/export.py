"""Export a trained checkpoint as plain JSON so the browser can run the network (the "inside the
trained model" visual) and so the weights can be downloaded next to the labeled data."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import numpy as np


def checkpoint_to_json(ckpt: dict[str, Any], name: str = "learned-bc", digits: int = 4) -> dict[str, Any]:
    state = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    layers = []
    index = 0
    while f"{index}.weight" in state:  # nn.Sequential: Linear at even positions, ReLU between
        layers.append({"w": np.round(state[f"{index}.weight"], digits).tolist(), "b": np.round(state[f"{index}.bias"], digits).tolist()})
        index += 2
    sizes = [int(ckpt["in_dim"]), *[int(h) for h in ckpt["hidden"]], int(ckpt["out_dim"])]
    return {
        "name": name,
        "sizes": sizes,
        "params": int(sum(v.size for v in state.values())),
        "layers": layers,
        "x_mean": np.asarray(ckpt["x_mean"]).astype(float).tolist(),
        "x_std": np.asarray(ckpt["x_std"]).astype(float).tolist(),
        "y_mean": np.asarray(ckpt["y_mean"]).astype(float).tolist(),
        "y_std": np.asarray(ckpt["y_std"]).astype(float).tolist(),
        "inputs": ckpt.get("state_names"),
        "outputs": (ckpt.get("action_spec") or {}).get("names"),
    }


def post_model(url: str, model: dict[str, Any], timeout_s: float = 5.0) -> bool:
    """Best effort, like the training events: never raises."""
    try:
        req = urllib.request.Request(url, data=json.dumps(model).encode(), method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
        return True
    except Exception:  # noqa: BLE001
        return False
