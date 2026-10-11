"""The only seam between the decision layer and a robot/scene.

To wire in a real robot or simulator later, implement `RobotEnv`.
`step` must set `info["success"]` (bool) once the task outcome is known, and may set
`info["joints"]` (dict of joint name -> angle) which the runner records next to the action.

`reset` and `step` also accept optional keyword arguments, `model` (the id of the policy
driving the robot) and, for `step`, `label` (a short description of the intent). Envs that
do not care can ignore them; the runner only passes them to envs whose signature accepts
them, so older envs without these keywords keep working.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from winners.schema import ActionSpec, Observation, ObservationSpec


@runtime_checkable
class RobotEnv(Protocol):
    task_description: str
    action_spec: ActionSpec
    observation_spec: ObservationSpec

    def reset(self, seed: int | None = None, *, model: str | None = None) -> Observation: ...

    def step(
        self, action: list[float], *, label: str | None = None, model: str | None = None
    ) -> tuple[Observation, float, bool, dict[str, Any]]: ...
