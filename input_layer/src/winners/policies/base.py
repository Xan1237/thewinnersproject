"""The single interface every decision-maker implements (LLMs, learned model)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from winners.schema import Action, ActionSpec, Observation, Step


@runtime_checkable
class Policy(Protocol):
    name: str

    def decide(
        self,
        obs: Observation,
        task: str,
        action_spec: ActionSpec,
        history: list[Step],
    ) -> Action:
        """Return an action. `Action.meta` may carry `latency_s`, `cost_usd`, `error`."""
        ...
