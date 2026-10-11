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
        c = self._ckpt
        x = (self._torch.tensor(obs.state, dtype=self._torch.float32).numpy() - c["x_mean"]) / c["x_std"]
        with self._torch.no_grad():
            y = self._model(self._torch.from_numpy(x[None, :]))[0].numpy()
        return Action(values=(y * c["y_std"] + c["y_mean"]).astype(float).tolist())
