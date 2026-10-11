"""Behavior cloning: fit a small MLP mapping robot state -> the action the teachers took."""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

from winners.dataset import BCDataset
from winners.schema import ActionSpec
from winners.train.events import TrainingEventSink


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]):
    import torch.nn as nn

    layers: list[Any] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_bc(
    ds: BCDataset,
    action_spec: ActionSpec,
    hidden: tuple[int, ...] = (128, 128),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    training_events_url: str | None = None,
    epoch_delay_s: float = 0.0,
    stop: threading.Event | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Returns (checkpoint, metrics). Checkpoint is self-contained (weights + normalization
    + action spec) so `LearnedPolicy` needs nothing else.

    If `training_events_url` is set, posts `{"type": "training", "phase": "epoch", "epoch",
    "loss", "val_loss"}` after every epoch (best effort: failures never affect training)."""
    import torch

    torch.manual_seed(seed)
    x = torch.from_numpy((ds.x_train - ds.x_mean) / ds.x_std)
    y = torch.from_numpy((ds.y_train - ds.y_mean) / ds.y_std)
    model = _mlp(x.shape[1], y.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    sink = TrainingEventSink(training_events_url)
    xv = yv = None
    if len(ds.x_val):
        xv = torch.from_numpy((ds.x_val - ds.x_mean) / ds.x_std)
        yv = torch.from_numpy((ds.y_val - ds.y_mean) / ds.y_std)

    for epoch in range(1, epochs + 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("training cancelled")
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if sink.active:
            val_loss = None
            if xv is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(xv), yv))
                model.train()
            sink.send(phase="epoch", epoch=epoch, loss=total / len(x), val_loss=val_loss)
            if epoch_delay_s:
                if stop is not None:
                    stop.wait(epoch_delay_s)
                else:
                    time.sleep(epoch_delay_s)  # slows training down so the live chart is watchable

    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        metrics["train_mse"] = float(loss_fn(model(x), y))
        if xv is not None:
            metrics["val_mse"] = float(loss_fn(model(xv), yv))

    checkpoint = {
        "state_dict": model.state_dict(),
        "in_dim": int(x.shape[1]),
        "out_dim": int(y.shape[1]),
        "hidden": tuple(hidden),
        "x_mean": ds.x_mean,
        "x_std": ds.x_std,
        "y_mean": ds.y_mean,
        "y_std": ds.y_std,
        "action_spec": action_spec.model_dump(),
    }
    return checkpoint, metrics


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    # weights_only=False: the checkpoint holds numpy arrays; only load files you trained.
    return torch.load(path, weights_only=False)


def predict(checkpoint: dict[str, Any], state: list[float]) -> list[float]:
    import torch

    model = _mlp(checkpoint["in_dim"], checkpoint["out_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = (np.asarray(state, dtype=np.float32) - checkpoint["x_mean"]) / checkpoint["x_std"]
    with torch.no_grad():
        y = model(torch.from_numpy(x[None, :]))[0].numpy()
    return (y * checkpoint["y_std"] + checkpoint["y_mean"]).astype(float).tolist()
