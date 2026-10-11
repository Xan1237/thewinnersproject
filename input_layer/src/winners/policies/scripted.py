"""Deterministic fake policies for tests and for bootstrapping data without network access."""

from __future__ import annotations

import random

from winners.schema import Action, ActionSpec, Observation, Step


class GreedyReachPolicy:
    """Heads straight for the goal of MockReachEnv (state = x, y, gx, gy), with optional noise."""

    def __init__(self, name: str = "scripted-greedy", noise: float = 0.0, seed: int = 0):
        self.name = name
        self.noise = noise
        self._rng = random.Random(seed)

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        x, y, gx, gy = obs.state
        dx, dy = gx - x, gy - y
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        values = [dx / norm, dy / norm]
        if self.noise:
            values = [v + self._rng.gauss(0.0, self.noise) for v in values]
        return Action(values=values)


class FixedPolicy:
    """Always returns the same raw values. Handy for testing safety handling."""

    def __init__(self, values: list[float], name: str = "scripted-fixed"):
        self.name = name
        self._values = values

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        return Action(values=list(self._values))
