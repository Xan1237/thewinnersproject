"""Toy 2D reach task for developing and testing the pipeline without a robot."""

from __future__ import annotations

import math
import random
from typing import Any

from winners.schema import ActionSpec, Observation, ObservationSpec


class MockReachEnv:
    """A point must reach a goal. Action = (dx, dy) in [-1, 1], scaled by `step_size`."""

    def __init__(self, step_size: float = 0.1, tolerance: float = 0.1, max_steps: int = 60):
        self.step_size = step_size
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.task_description = (
            "Move the point to the goal position. Success when the point is within "
            f"{tolerance} of the goal."
        )
        self.action_spec = ActionSpec(
            names=["dx", "dy"], low=[-1.0, -1.0], high=[1.0, 1.0], units="fraction of max step"
        )
        self.observation_spec = ObservationSpec(
            names=["x", "y", "goal_x", "goal_y"],
            description="Point position followed by goal position, both in [0, 1].",
        )
        self._rng = random.Random(0)
        self._pos = [0.0, 0.0]
        self._goal = [0.0, 0.0]
        self._t = 0

    def _obs(self) -> Observation:
        return Observation(t=self._t, state=[*self._pos, *self._goal])

    def _dist(self) -> float:
        return math.dist(self._pos, self._goal)

    def reset(self, seed: int | None = None, *, model: str | None = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._pos = [self._rng.random(), self._rng.random()]
        self._goal = [self._rng.random(), self._rng.random()]
        self._t = 0
        return self._obs()

    def step(
        self, action: list[float], *, label: str | None = None, model: str | None = None
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        action = self.action_spec.clip(action)
        self._pos = [p + a * self.step_size for p, a in zip(self._pos, action)]
        self._t += 1
        dist = self._dist()
        success = dist <= self.tolerance
        done = success or self._t >= self.max_steps
        return self._obs(), -dist, done, {"success": success, "distance": dist}
