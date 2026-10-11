"""The show: robots that never stop, orders that always work, a model that learns while you watch.

    python -m winners.show --config configs/g1_arena.yaml --out runs/g1-show

* Every frontier policy of the config gets a robot and works through little routines ("grab a
  coffee, then sit", "walk to the lectern and wave", ...) forever. Every decision is recorded.
* Any spoken or typed order (the viewer's mic button) interrupts all of them (see winners.live).
* Scripted teachers play the same routines on a second sim ("data farm") so training has data
  from the start; as soon as the frontier models have produced enough winning runs, a small
  network is trained on all winners, a gold robot with the trained model joins and plays the same
  routines, and it is retrained as more data arrives.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import yaml

from winners.arena import run_arena
from winners.dataset import build_dataset, select_episodes
from winners.live import agent_loop
from winners.policies.learned import LearnedPolicy
from winners.registry import build_env_factory, build_policy
from winners.schema import Episode
from winners.train.bc import save_checkpoint, train_bc
from winners.train.events import TrainingEventSink
from winners.train.export import checkpoint_to_json, post_model

TRAINED_NAME = "learned-bc"
DEFAULT_ROUTINES = ["make_coffee"]


class HotPolicy:
    """A policy whose weights can be swapped while its robot keeps playing."""

    def __init__(self, name: str):
        self.name = name
        self.inner: LearnedPolicy | None = None

    def decide(self, obs, task, action_spec, history):
        return self.inner.decide(obs, task, action_spec, history)  # type: ignore[union-attr]


def run_show(cfg: dict, out: Path, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    out.mkdir(parents=True, exist_ok=True)
    env_cfg, run_cfg, train_cfg = cfg.get("env", {}), cfg.get("run", {}), cfg.get("train", {})
    show_cfg = cfg.get("show", {})
    routines = show_cfg.get("routines") or DEFAULT_ROUTINES
    base_url = env_cfg.get("base_url", "http://127.0.0.1:8818")
    events_url = f"{base_url}/sim/training"
    sink_events = TrainingEventSink(events_url)

    lock = threading.Lock()
    episodes: list[Episode] = []      # everything the live robots did
    farm: list[Episode] = []          # scripted bootstrap runs
    path = out / "episodes_all.jsonl"
    names = set()

    def record(ep: Episode) -> None:
        with lock:
            episodes.append(ep)
            with path.open("a", encoding="utf-8") as f:
                f.write(ep.model_dump_json() + "\n")
        if ep.model_id == TRAINED_NAME:
            mine = [e for e in episodes if e.model_id == TRAINED_NAME][-12:]
            sink_events.send(phase="eval", success_rate=sum(e.success for e in mine) / len(mine), n=len(mine), task="routines")

    policies = [build_policy(p) for p in cfg["policies"]]
    names.update(p.name for p in policies)
    threads: list[threading.Thread] = []

    def spawn(policy, index: int) -> None:
        t = threading.Thread(
            target=agent_loop,
            args=(policy, env_cfg, run_cfg, index, stop, record, routines, show_cfg.get("pause_s", 3.0), show_cfg.get("hold_after_order_s", 12.0)),
            daemon=True,
        )
        t.start()
        threads.append(t)

    sink_events.send(phase="collect", total=0, models=len(policies))
    for i, p in enumerate(policies):
        spawn(p, i)
        time.sleep(0.4)

    # -- data farm: scripted teachers play the same routines on a second sim ---------------------
    def bootstrap() -> None:
        extra = [build_policy(p) for p in train_cfg.get("extra_policies", [])]
        if not extra:
            return
        farm_env = {**env_cfg, "base_url": train_cfg.get("extra_base_url", base_url)}
        sink_events.send(phase="topup", total=len(extra) * len(routines) * show_cfg.get("bootstrap_episodes", 6))
        for r in routines:
            if stop.is_set():
                return
            eps = run_arena(
                build_env_factory({**farm_env, "task": r}), extra,
                episodes=show_cfg.get("bootstrap_episodes", 6), seed0=100,
                max_steps=run_cfg.get("max_steps", 110),
            )
            with lock:
                farm.extend(eps)
                with (out / "episodes_farm.jsonl").open("a", encoding="utf-8") as f:
                    for e in eps:
                        f.write(e.model_dump_json() + "\n")

    farm_thread = threading.Thread(target=bootstrap, daemon=True)
    farm_thread.start()

    # -- trainer -----------------------------------------------------------------------------------
    hot = HotPolicy(TRAINED_NAME)
    trained_started = False
    wins_at_last_round = 0
    first_round_wins = show_cfg.get("first_round_wins", 10)
    every = show_cfg.get("retrain_every_wins", 25)
    env_for_spec = build_env_factory({**env_cfg, "task": "idle"})()
    round_no = 0

    while not stop.is_set():
        time.sleep(1.0)
        with lock:
            frontier_wins = sum(e.success for e in episodes if e.model_id in names)
            ready_farm = not farm_thread.is_alive()
        need = first_round_wins if round_no == 0 else every
        if frontier_wins - wins_at_last_round < need or not ready_farm:
            continue
        with lock:
            pool = [*episodes, *farm]
        kept = select_episodes([e for e in pool if e.model_id != TRAINED_NAME], only_success=True)
        if len(kept) < 8:
            continue
        round_no += 1
        wins_at_last_round = frontier_wins
        ds = build_dataset(kept, val_fraction=train_cfg.get("val_fraction", 0.2))
        n_samples = len(ds.x_train) + len(ds.x_val)
        from_llm = sum(e.model_id in names for e in kept)
        print(f"[train] round {round_no}: {len(kept)} winning runs ({from_llm} frontier), {n_samples} samples")
        sink_events.send(phase="start", round=round_no, epochs=train_cfg.get("epochs", 100), collected=len(pool), episodes=len(kept), samples=n_samples, from_llm=from_llm, from_scripted=len(kept) - from_llm)
        ckpt, metrics = train_bc(
            ds, env_for_spec.action_spec, hidden=tuple(train_cfg.get("hidden", (128, 128))), epochs=train_cfg.get("epochs", 100),
            training_events_url=events_url, epoch_delay_s=train_cfg.get("epoch_delay_s", 0.0),
        )
        ckpt_path = save_checkpoint(ckpt, out / "bc.pt")
        model_json = checkpoint_to_json({**ckpt, "state_names": list(getattr(env_for_spec.observation_spec, "names", []) or [])}, name=TRAINED_NAME)
        (out / "policy.json").write_text(json.dumps(model_json), encoding="utf-8")
        post_model(f"{base_url}/sim/model", model_json)
        hot.inner = LearnedPolicy(ckpt_path, name=TRAINED_NAME)
        print("[train] metrics:", metrics)
        sink_events.send(phase="done", checkpoint=str(ckpt_path), round=round_no)
        if not trained_started:
            trained_started = True
            spawn(hot, len(policies))

    for t in threads:
        t.join(timeout=5)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    stop = threading.Event()
    try:
        run_show(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), Path(args.out), stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
