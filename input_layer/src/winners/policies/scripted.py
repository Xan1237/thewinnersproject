"""Deterministic fake policies for tests and for bootstrapping data without network access."""

from __future__ import annotations

import math
import random

from winners.schema import Action, ActionSpec, Observation, Step



def _is_pickup(task: str) -> bool:
    """Tasks reach the policy as the sim's description ("... pick up the coffee cup ..."), or as the id."""
    t = task.lower()
    return any(w in t for w in ("fetch", "pick up", "coffee"))


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
        return Action(values=values, label="move toward goal")


class GoalSeekPolicy:
    """Scripted teacher for the G1 sim. Reads the robot-frame goal from the state vector
    `[x, z, heading, goal_forward, goal_right, goal_dist, holding, steps_left_frac]`
    and outputs `[forward, right, turn, interact]`: steer toward the goal, slow down while
    misaligned, and press `interact` once within `interact_dist` (for fetch tasks).

    Assumes positive `turn` increases the heading, i.e. turns left (counter-clockwise from
    above), so a goal on the right needs a negative turn.
    """

    def __init__(
        self,
        name: str = "scripted-goto",
        noise: float = 0.0,
        seed: int = 0,
        turn_gain: float = 1.5,
        interact_dist: float = 0.7,
    ):
        self.name = name
        self.noise = noise
        self.turn_gain = turn_gain
        self.interact_dist = interact_dist
        self._rng = random.Random(seed)

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        _x, _z, _heading, fwd, right, dist, _holding, _frac = obs.state[:8]
        if _is_pickup(task) and dist <= self.interact_dist:
            return Action(values=[0.0, 0.0, 0.0, 1.0], label="grab the coffee")
        angle = math.atan2(right, fwd)  # 0 = dead ahead, +pi/2 = goal on the right
        turn = -self.turn_gain * angle
        forward = max(0.0, math.cos(angle)) * min(1.0, dist)
        values = [forward, 0.0, turn, 0.0]
        if self.noise:
            for i in (0, 2):  # exploration noise on forward and turn only
                values[i] += self._rng.gauss(0.0, self.noise)
        label = "walk toward the goal" if abs(angle) < 0.4 else (
            "turn right toward the goal" if angle > 0 else "turn left toward the goal"
        )
        return Action(values=values, label=label)


class FixedPolicy:
    """Always returns the same raw values. Handy for testing safety handling."""

    def __init__(self, values: list[float], name: str = "scripted-fixed"):
        self.name = name
        self._values = values

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        return Action(values=list(self._values))
