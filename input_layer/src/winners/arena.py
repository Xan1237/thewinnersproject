"""Runs every policy on the same seeds and ranks them.

    python -m winners.arena --config configs/default.yaml --out runs/run1
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml

from winners.env.base import RobotEnv
from winners.policies.base import Policy
from winners.recorder import save_episodes
from winners.registry import build_env_factory, build_policy
from winners.runner import run_episode
from winners.schema import Episode


def _run_policy(
    env_factory: Callable[[], RobotEnv],
    policy: Policy,
    seeds: list[int],
    decision_interval: int,
    max_steps: int,
    stop: threading.Event | None = None,
) -> list[Episode]:
    env = env_factory()
    results = []
    try:
        for seed in seeds:
            if stop is not None and stop.is_set():
                break
            results.append(run_episode(env, policy, seed, decision_interval, max_steps,
                                       cancelled=stop.is_set if stop is not None else None))
        return results
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()


def run_arena(
    env_factory: Callable[[], RobotEnv],
    policies: list[Policy],
    episodes: int = 10,
    seed0: int = 0,
    decision_interval: int = 1,
    max_steps: int = 500,
    stop: threading.Event | None = None,
) -> list[Episode]:
    """Each policy runs the same seeds, sequentially, on its own thread (so each
    provider is naturally rate-limited to one in-flight request per model)."""
    seeds = list(range(seed0, seed0 + episodes))
    with ThreadPoolExecutor(max_workers=max(len(policies), 1)) as pool:
        futures = [
            pool.submit(_run_policy, env_factory, p, seeds, decision_interval, max_steps, stop)
            for p in policies
        ]
        return [ep for f in futures for ep in f.result()]


def leaderboard(episodes: list[Episode]) -> list[dict]:
    by_model: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_model.setdefault(ep.model_id, []).append(ep)
    rows = []
    for model, eps in by_model.items():
        wins = [e for e in eps if e.success]
        decisions = [s for e in eps for s in e.steps if s.decided]
        rows.append(
            {
                "model": model,
                "episodes": len(eps),
                "success_rate": len(wins) / len(eps),
                "mean_steps_on_success": sum(e.length for e in wins) / len(wins) if wins else None,
                "mean_decision_latency_s": (
                    sum(s.latency_s for s in decisions) / len(decisions) if decisions else 0.0
                ),
                "total_cost_usd": sum(e.total_cost_usd for e in eps),
                "model_errors": sum("model_error" in s.flags for e in eps for s in e.steps),
            }
        )
    return sorted(rows, key=lambda r: (-r["success_rate"], r["mean_steps_on_success"] or 1e9))


def format_leaderboard(rows: list[dict]) -> str:
    header = f"{'model':32} {'succ':>6} {'steps':>7} {'lat(s)':>8} {'cost($)':>9} {'errs':>5}"
    lines = [header, "-" * len(header)]
    for r in rows:
        steps = f"{r['mean_steps_on_success']:.1f}" if r["mean_steps_on_success"] is not None else "-"
        lines.append(
            f"{r['model'][:32]:32} {r['success_rate']:>6.0%} {steps:>7} "
            f"{r['mean_decision_latency_s']:>8.3f} {r['total_cost_usd']:>9.4f} {r['model_errors']:>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env_factory = build_env_factory(cfg.get("env", {}))
    policies = [build_policy(p) for p in cfg["policies"]]
    run = cfg.get("run", {})
    eps = run_arena(
        env_factory,
        policies,
        episodes=run.get("episodes", 10),
        seed0=run.get("seed0", 0),
        decision_interval=run.get("decision_interval", 1),
        max_steps=run.get("max_steps", 500),
    )
    out = Path(args.out)
    save_episodes(eps, out / "episodes.jsonl")
    rows = leaderboard(eps)
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(format_leaderboard(rows))


if __name__ == "__main__":
    main()
