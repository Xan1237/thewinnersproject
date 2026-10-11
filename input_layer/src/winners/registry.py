"""Builds environments and policies from plain config dicts (see configs/default.yaml)."""

from __future__ import annotations

from typing import Any, Callable

from winners.env.base import RobotEnv
from winners.env.g1 import G1SimEnv
from winners.env.mock import MockReachEnv
from winners.policies.base import Policy
from winners.policies.claude import ClaudePolicy
from winners.policies.gemini import GeminiPolicy
from winners.policies.local import LocalPolicy
from winners.policies.openai import OpenAIPolicy
from winners.policies.openrouter import OpenRouterPolicy
from winners.policies.scripted import GoalSeekPolicy, GreedyReachPolicy

ENVS: dict[str, Callable[..., RobotEnv]] = {"mock-reach": MockReachEnv, "g1-sim": G1SimEnv}


def build_env_factory(cfg: dict[str, Any]) -> Callable[[], RobotEnv]:
    kind = cfg.get("type", "mock-reach")
    if kind not in ENVS:
        raise ValueError(f"unknown env type {kind!r}; known: {sorted(ENVS)}")
    params = {k: v for k, v in cfg.items() if k != "type"}
    return lambda: ENVS[kind](**params)


def build_policy(cfg: dict[str, Any]) -> Policy:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "claude":
        return ClaudePolicy(**cfg)
    if kind == "gemini":
        return GeminiPolicy(**cfg)
    if kind == "openai":
        return OpenAIPolicy(**cfg)
    if kind == "openrouter":
        return OpenRouterPolicy(**cfg)
    if kind == "local":
        return LocalPolicy(**cfg)
    if kind == "scripted-greedy":
        return GreedyReachPolicy(**cfg)
    if kind == "scripted-goto":
        return GoalSeekPolicy(**cfg)
    if kind == "learned":
        from winners.policies.learned import LearnedPolicy

        return LearnedPolicy(**cfg)
    raise ValueError(f"unknown policy type {kind!r}")
