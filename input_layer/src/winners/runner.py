"""Runs one policy through one episode and records everything."""

from __future__ import annotations

import time
import uuid

from winners.env.base import RobotEnv
from winners.policies.base import Policy
from winners.safety import SafetyFilter
from winners.schema import Action, Episode, Step


def run_episode(
    env: RobotEnv,
    policy: Policy,
    seed: int,
    decision_interval: int = 1,
    max_steps: int = 500,
) -> Episode:
    """Roll out `policy` on `env`.

    The policy is queried every `decision_interval` ticks (LLMs are slow compared to
    control rate); in between, its last safe action is held. Every tick is recorded
    so the training data has full-rate physics, with `decided` marking real queries.
    """
    safety = SafetyFilter(env.action_spec)
    obs = env.reset(seed=seed)
    steps: list[Step] = []
    held: list[float] = env.action_spec.hold()
    held_raw: list[float] | None = None
    latency = cost = 0.0
    flags: list[str] = []
    info: dict = {}
    total_reward = 0.0

    for t in range(max_steps):
        decided = t % decision_interval == 0
        if decided:
            started = time.perf_counter()
            try:
                action = policy.decide(obs, env.task_description, env.action_spec, steps)
            except Exception as exc:  # a failing model must not crash the run
                action = Action(values=[], meta={"error": f"{type(exc).__name__}: {exc}"})
            wall = time.perf_counter() - started
            latency = float(action.meta.get("latency_s", wall))
            cost = float(action.meta.get("cost_usd", 0.0))
            held_raw = action.values or None
            held, flags = safety.sanitize(held_raw)
            if "error" in action.meta:
                flags = [*flags, "model_error"]
        else:
            latency = cost = 0.0
            flags = []

        state_before = obs.state
        obs, reward, done, info = env.step(held)
        total_reward += reward
        steps.append(
            Step(
                t=t,
                state=state_before,
                raw_action=held_raw,
                safe_action=held,
                decided=decided,
                reward=reward,
                source_model=policy.name,
                latency_s=latency,
                cost_usd=cost,
                flags=flags,
            )
        )
        if done:
            break

    return Episode(
        episode_id=uuid.uuid4().hex[:12],
        model_id=policy.name,
        task=env.task_description,
        seed=seed,
        steps=steps,
        success=bool(info.get("success", False)),
        total_reward=total_reward,
        meta={"decision_interval": decision_interval},
    )
