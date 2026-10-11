"""Teachers -> arena -> best-of-N filter -> behaviour cloning -> held-out evaluation.

    python -m winners.pipeline --config configs/g1_offline.yaml --out runs/g1

Posts progress to the sim server (`<base_url>/sim/training`) so the live viewer can chart it.
The config is an arena config (see configs/g1_arena.yaml) plus an optional `train:` section
and `run.tasks` (a list of tasks; the arena runs once per task).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from winners.arena import format_leaderboard, leaderboard, run_arena
from winners.dataset import build_dataset, select_episodes
from winners.recorder import save_episodes
from winners.registry import build_env_factory, build_policy
from winners.train.events import TrainingEventSink


class _Paced:
    """Wraps a policy and sleeps before every decision, so a fast local policy plays back in
    real time in the viewer instead of finishing a whole episode in milliseconds."""

    def __init__(self, policy, pace_s: float):
        self._policy = policy
        self._pace_s = pace_s
        self.name = policy.name

    def decide(self, *args, **kwargs):
        time.sleep(self._pace_s)
        return self._policy.decide(*args, **kwargs)


def run_pipeline(cfg: dict, out: Path) -> dict:
    from winners.policies.learned import LearnedPolicy
    from winners.train.bc import save_checkpoint, train_bc
    from winners.train.evaluate import evaluate_policy

    env_cfg = cfg.get("env", {})
    run = cfg.get("run", {})
    train = cfg.get("train", {})
    tasks = run.get("tasks") or [env_cfg.get("task", "goto:armchair_blue")]
    base_url = env_cfg.get("base_url", "http://127.0.0.1:8818")
    events_url = f"{base_url}/sim/training"
    sink = TrainingEventSink(events_url)

    policies = [build_policy(p) for p in cfg["policies"]]
    episodes = []
    for task in tasks:
        factory = build_env_factory({**env_cfg, "task": task})
        episodes += run_arena(
            factory,
            policies,
            episodes=run.get("episodes", 10),
            seed0=run.get("seed0", 0),
            decision_interval=run.get("decision_interval", 1),
            max_steps=run.get("max_steps", 120),
        )
    out.mkdir(parents=True, exist_ok=True)
    save_episodes(episodes, out / "episodes.jsonl")
    rows = leaderboard(episodes)
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(format_leaderboard(rows))
    llm_episodes = len(episodes)

    # A handful of slow LLM runs is too little data for a network. Top it up with cheap scripted
    # runs on a separate sim server (the "data farm", default: the same one) so the viewer is
    # not flooded. The panel reports how many episodes came from where.
    extra = [build_policy(p) for p in train.get("extra_policies", [])]
    if extra:
        farm_env = {**env_cfg, "base_url": train.get("extra_base_url", base_url)}
        for task in tasks:
            episodes += run_arena(
                build_env_factory({**farm_env, "task": task}),
                extra,
                episodes=train.get("extra_episodes", 25),
                seed0=train.get("extra_seed0", 100),
                max_steps=run.get("max_steps", 120),
            )
        save_episodes(episodes, out / "episodes_all.jsonl")

    teachers = train.get("teachers") or None  # None: every model's successful runs
    kept = select_episodes(
        episodes,
        only_success=True,
        teachers=teachers,
        keep_fastest_fraction=train.get("keep_fastest_fraction", 1.0),
    )
    if not kept:
        raise SystemExit("no successful episodes to learn from")
    ds = build_dataset(kept, val_fraction=train.get("val_fraction", 0.2))
    n_samples = len(ds.x_train) + len(ds.x_val)
    print(f"training on {len(kept)} successful episodes, {n_samples} samples")
    from_llm = sum(e.model_id in {p.name for p in policies} for e in kept)
    sink.send(phase="start", episodes=len(kept), samples=n_samples, from_llm=from_llm, from_scripted=len(kept) - from_llm)

    env = build_env_factory({**env_cfg, "task": tasks[0]})()
    ckpt, metrics = train_bc(
        ds,
        env.action_spec,
        hidden=tuple(train.get("hidden", (128, 128))),
        epochs=train.get("epochs", 100),
        training_events_url=events_url,
        epoch_delay_s=train.get("epoch_delay_s", 0.0),
    )
    path = save_checkpoint(ckpt, out / "bc.pt")
    print("train metrics:", metrics)

    learned = LearnedPolicy(path)
    if train.get("eval_pace_s"):
        learned = _Paced(learned, train["eval_pace_s"])
    eval_seeds = list(range(train.get("eval_seed0", 1000), train.get("eval_seed0", 1000) + train.get("eval_episodes", 10)))
    results = {}
    for task in tasks:
        eval_env = build_env_factory({**env_cfg, "task": task})()
        result, eps = evaluate_policy(
            eval_env, learned, eval_seeds, max_steps=run.get("max_steps", 120), training_events_url=events_url
        )
        results[task] = result
        print(f"learned policy on {task}: {result['success_rate']:.0%} over {result['n']} episodes")
    sink.send(phase="done", checkpoint=str(path))
    return {"leaderboard": rows, "metrics": metrics, "eval": results, "checkpoint": str(path)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_pipeline(cfg, Path(args.out))


if __name__ == "__main__":
    main()
