"""The distilled policy: runs locally, no API calls, no per-step cost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from winners.schema import Action, ActionSpec, Observation, Step
from winners.train.bc import _mlp, load_checkpoint


class LearnedPolicy:
    def __init__(self, checkpoint: str | Path | dict[str, Any], name: str = "learned-bc"):
        import torch

        ckpt = load_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.name = name
        self._ckpt = ckpt
        self._model = _mlp(ckpt["in_dim"], ckpt["out_dim"], ckpt["hidden"])
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._torch = torch

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        import time

        started = time.perf_counter()
        c = self._ckpt
        x = (self._torch.tensor(obs.state, dtype=self._torch.float32).numpy() - c["x_mean"]) / c["x_std"]
        with self._torch.no_grad():
            y = self._model(self._torch.from_numpy(x[None, :]))[0].numpy()
        values = (y * c["y_std"] + c["y_mean"]).astype(float).tolist()
        return Action(
            values=values,
            label=_describe(values),
            meta={"latency_s": time.perf_counter() - started, "cost_usd": 0.0, "detail": {"policy": "bc-mlp", "tokens": 0}},
        )


def _describe(v: list[float]) -> str:
    """A human label for the dominant part of [forward, right, turn, interact]."""
    forward, right, turn, interact = (list(v) + [0, 0, 0, 0])[:4]
    if interact > 0.5:
        return "pick up"
    parts = []
    if forward > 0.25:
        parts.append("walk forward")
    elif forward < -0.25:
        parts.append("step back")
    if abs(right) > 0.25:
        parts.append("sidestep right" if right > 0 else "sidestep left")
    if abs(turn) > 0.2:
        parts.append("turn left" if turn > 0 else "turn right")
    return " + ".join(parts) or "hold"
