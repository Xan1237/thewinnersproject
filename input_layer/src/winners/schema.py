"""Core data types shared by every module."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ActionSpec(BaseModel):
    """Describes the action vector a robot accepts."""

    names: list[str]
    low: list[float]
    high: list[float]
    units: str = ""

    @model_validator(mode="after")
    def _check_lengths(self) -> "ActionSpec":
        if not (len(self.names) == len(self.low) == len(self.high)):
            raise ValueError("names, low and high must have the same length")
        if any(lo > hi for lo, hi in zip(self.low, self.high)):
            raise ValueError("low must be <= high for every dimension")
        return self

    @property
    def dim(self) -> int:
        return len(self.names)

    def hold(self) -> list[float]:
        """The 'do nothing' action: zero, clipped into range."""
        return self.clip([0.0] * self.dim)

    def clip(self, values: list[float]) -> list[float]:
        return [min(max(v, lo), hi) for v, lo, hi in zip(values, self.low, self.high)]

    def contains(self, values: list[float]) -> bool:
        return len(values) == self.dim and all(
            math.isfinite(v) and lo <= v <= hi
            for v, lo, hi in zip(values, self.low, self.high)
        )


class ObservationSpec(BaseModel):
    names: list[str]
    description: str = ""


class Observation(BaseModel):
    """What the decision layer sees. `state` is the physics vector that gets
    recorded and later used for training; `text` is any extra context for LLMs."""

    t: int = 0
    state: list[float]
    text: str = ""


class Action(BaseModel):
    values: list[float]
    meta: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    t: int
    state: list[float]
    raw_action: list[float] | None
    safe_action: list[float]
    decided: bool  # True on ticks where the policy was actually queried
    reward: float
    source_model: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    flags: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    episode_id: str
    model_id: str
    task: str
    seed: int
    steps: list[Step]
    success: bool
    total_reward: float
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)
