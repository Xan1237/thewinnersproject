"""Turns recorded episodes into (state -> action) arrays for behavior cloning."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from winners.schema import Episode

BAD_FLAGS = {"model_error", "no_action", "wrong_dim", "non_finite"}


@dataclass
class BCDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_train_episodes: int
    n_val_episodes: int


def select_episodes(
    episodes: list[Episode],
    only_success: bool = True,
    teachers: list[str] | None = None,
    keep_fastest_fraction: float = 1.0,
) -> list[Episode]:
    """Best-of-N filter: successful runs only, optionally from chosen teachers, and
    optionally just the fastest fraction of each teacher's successes."""
    picked = [
        e
        for e in episodes
        if (e.success or not only_success) and (teachers is None or e.model_id in teachers)
    ]
    if keep_fastest_fraction >= 1.0:
        return picked
    out: list[Episode] = []
    for model in {e.model_id for e in picked}:
        mine = sorted((e for e in picked if e.model_id == model), key=lambda e: e.length)
        out.extend(mine[: max(1, int(len(mine) * keep_fastest_fraction))])
    return out


def build_dataset(
    episodes: list[Episode],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> BCDataset:
    """Split by episode (never within one) so validation measures generalization."""
    if not episodes:
        raise ValueError("no episodes to build a dataset from")
    order = list(range(len(episodes)))
    random.Random(seed).shuffle(order)
    n_val = int(len(order) * val_fraction)
    if len(order) > 1:
        n_val = max(n_val, 1)
    val_idx = set(order[:n_val])

    def arrays(idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in idxs:
            for s in episodes[i].steps:
                if BAD_FLAGS.intersection(s.flags):
                    continue
                xs.append(s.state)
                ys.append(s.safe_action)
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    train_idx = [i for i in order if i not in val_idx]
    x_train, y_train = arrays(train_idx)
    x_val, y_val = arrays(sorted(val_idx))
    if len(x_train) == 0:
        raise ValueError("no usable training steps after filtering")

    eps = 1e-6
    return BCDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_mean=x_train.mean(0),
        x_std=x_train.std(0) + eps,
        y_mean=y_train.mean(0),
        y_std=y_train.std(0) + eps,
        n_train_episodes=len(train_idx),
        n_val_episodes=len(val_idx),
    )
