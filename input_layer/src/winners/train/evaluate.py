"""Roll a policy out on held-out seeds and report the success rate."""

from __future__ import annotations

from typing import Any

from winners.env.base import RobotEnv
from winners.policies.base import Policy
from winners.runner import run_episode
from winners.schema import Episode
from winners.train.events import TrainingEventSink


def evaluate_policy(
    env: RobotEnv,
    policy: Policy,
    seeds: list[int],
    max_steps: int = 500,
    decision_interval: int = 1,
    training_events_url: str | None = None,
) -> tuple[dict[str, Any], list[Episode]]:
    """Returns (`{"success_rate", "n"}`, episodes). If `training_events_url` is set, posts
    `{"type": "training", "phase": "eval", "success_rate", "n"}` (best effort)."""
    episodes = [run_episode(env, policy, s, decision_interval, max_steps) for s in seeds]
    n = len(episodes)
    result = {"success_rate": sum(e.success for e in episodes) / n if n else 0.0, "n": n}
    TrainingEventSink(training_events_url).send(phase="eval", **result)
    return result, episodes
