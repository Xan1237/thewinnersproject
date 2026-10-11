"""Runs one policy through one episode and records everything."""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, Callable

from winners.env.base import RobotEnv
from winners.policies.base import Policy
from winners.safety import SafetyFilter
from winners.schema import Action, Episode, Step


def _accepts(fn: Callable[..., Any], *names: str) -> set[str]:
    """Which of the optional keyword arguments `names` can `fn` take? (Backward compatibility
    with envs written before `label`/`model` existed.)"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return set(names)
    return {n for n in names if n in params}


def _joints(info: dict[str, Any]) -> dict[str, float] | None:
    raw = info.get("joints")
    if not isinstance(raw, dict) or not raw:
        return None
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def run_episode(
    env: RobotEnv,
    policy: Policy,
    seed: int,
    decision_interval: int = 1,
    max_steps: int = 500,
    *,
    resume: bool = False,
    close: bool = True,
    interrupt: Callable[[], str | None] | None = None,
) -> Episode:
    """Roll out `policy` on `env`.

    Live mode (voice orders): `resume=True` continues the env's current session instead of
    resetting it, `close=False` keeps the session (and robot) alive afterwards, and `interrupt`
    is polled before every tick; if it returns a task the episode ends early (meta["interrupted"])
    with the new task in meta["next_task"] for the caller to apply with `env.retask`.

    The policy is queried every `decision_interval` ticks (LLMs are slow compared to
    control rate); in between, its last safe action is held. Every tick is recorded
    so the training data has full-rate physics, with `decided` marking real queries.
    """
    reset_kw = {"model": policy.name} if _accepts(env.reset, "model") else {}
    step_ok = _accepts(env.step, "label", "model", "detail")
    obs = env.current_observation() if resume else env.reset(seed=seed, **reset_kw)
    interrupted: str | None = None
    # after reset: envs that discover their specs from a server only know them now
    safety = SafetyFilter(env.action_spec)
    steps: list[Step] = []
    held: list[float] = env.action_spec.hold()
    held_raw: list[float] | None = None
    held_label: str | None = None
    held_detail: dict[str, Any] | None = None
    latency = cost = 0.0
    flags: list[str] = []
    info: dict = {}
    total_reward = 0.0

    for t in range(max_steps):
        if interrupt is not None and (interrupted := interrupt()):
            break
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
            held_label = action.label
            held_detail = action.meta.get("detail") if decided else None
            held, flags = safety.sanitize(held_raw)
            if "error" in action.meta:
                flags = [*flags, "model_error"]
                held_label = held_label or "hold (model error)"
        else:
            latency = cost = 0.0
            flags = []

        state_before = obs.state
        step_kw: dict[str, Any] = {}
        if "label" in step_ok:
            step_kw["label"] = held_label
        if "model" in step_ok:
            step_kw["model"] = policy.name
        if "detail" in step_ok and decided and held_detail:
            step_kw["detail"] = held_detail
        obs, reward, done, info = env.step(held, **step_kw)
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
                label=held_label,
                joints=_joints(info),
            )
        )
        if done:
            break

    closer = getattr(env, "close", None)  # envs holding server-side sessions release them
    if close and callable(closer):
        closer()

    return Episode(
        episode_id=uuid.uuid4().hex[:12],
        model_id=policy.name,
        task=env.task_description,
        seed=seed,
        steps=steps,
        success=bool(info.get("success", False)),
        total_reward=total_reward,
        meta={"decision_interval": decision_interval, **({"interrupted": True, "next_task": interrupted} if interrupted else {})},
    )
