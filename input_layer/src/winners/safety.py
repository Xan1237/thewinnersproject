"""Validates and clamps model output before it reaches the robot."""

from __future__ import annotations

import math

from winners.schema import ActionSpec


class SafetyFilter:
    """Turns whatever a model produced into an action that is always within limits.

    Returns (safe_action, flags). Flags record what was changed so the raw/safe
    difference is visible in recorded data.
    """

    def __init__(self, spec: ActionSpec):
        self.spec = spec

    def sanitize(self, raw: list[float] | None) -> tuple[list[float], list[str]]:
        if raw is None:
            return self.spec.hold(), ["no_action"]
        if len(raw) != self.spec.dim:
            return self.spec.hold(), ["wrong_dim"]
        if any(not math.isfinite(v) for v in raw):
            return self.spec.hold(), ["non_finite"]
        safe = self.spec.clip(raw)
        flags = ["clamped"] if safe != list(raw) else []
        return safe, flags
