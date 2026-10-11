"""The only seam between the decision layer and a robot/scene.

To wire in a real robot or simulator later, implement `RobotEnv`.
`step` must set `info["success"]` (bool) once the task outcome is known.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from winners.schema import ActionSpec, Observation, ObservationSpec


@runtime_checkable
class RobotEnv(Protocol):
    task_description: str
    action_spec: ActionSpec
    observation_spec: ObservationSpec

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: list[float]) -> tuple[Observation, float, bool, dict[str, Any]]: ...
