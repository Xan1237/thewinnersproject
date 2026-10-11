"""Run robot routines, record attempts and train a local policy from successful episodes.

    python -m winners.show --config configs/g1_arena.yaml --out runs/g1-show --duration 240

* Every frontier policy of the config gets a robot and works through little routines ("grab a
  coffee, then sit", "walk to the lectern and wave", ...) forever. Every decision is recorded.
* Spoken or typed orders interrupt routines between decisions (see winners.live).
* Scripted teachers play the same routines on a second sim ("data farm") so training has data
  from the start; as soon as the frontier models have produced enough winning runs, a small
  network is trained on all winners, a gold robot with the trained model joins and plays the same
  routines, and it is retrained as more data arrives. Success is measured, not guaranteed.

The learned network consumes numeric simulator state, including the simulator-selected goal.
It does not parse natural language or infer goals from pixels.
"""

from __future__ import annotations

import argparse
import json
import signal
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
DEFAULT_ROUTINES = [
    "talk", "lectern_wave", "plant_coffee", "coffee_lectern",
    "armchair", "speaker_dance", "tripod_jump", "column_seat",
]


class HotPolicy:
    """A policy whose weights can be swapped while its robot keeps playing."""

    def __init__(self, name: str, pace_s: float = 0):
        self.name = name
        self.pace_s = pace_s
        self.inner: LearnedPolicy | None = None

    def decide(self, obs, task, action_spec, history):
        inner = self.inner
        if inner is None:
            raise RuntimeError("learned policy is not trained yet")
        if self.pace_s:
            time.sleep(self.pace_s)
        return inner.decide(obs, task, action_spec, history)


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
    errors: list[tuple[str, Exception]] = []

    def record(ep: Episode) -> None:
        with lock:
            episodes.append(ep)
            with path.open("a", encoding="utf-8") as f:
                f.write(ep.model_dump_json() + "\n")
            mine = [e for e in episodes if e.model_id == TRAINED_NAME][-12:]
        if ep.model_id == TRAINED_NAME:
            sink_events.send(phase="eval", success_rate=sum(e.success for e in mine) / len(mine), n=len(mine), task="routines")

    policies = [build_policy(p) for p in cfg["policies"]]
    if not policies:
        raise ValueError("show needs at least one collection policy")
    names.update(p.name for p in policies)
    threads: list[threading.Thread] = []

    def guarded(label, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:
            with lock:
                errors.append((label, exc))
            print(f"[{label}] worker failed: {type(exc).__name__}: {exc}", flush=True)
            stop.set()

    def spawn(policy, index: int) -> None:
        t = threading.Thread(
            target=guarded,
            args=(policy.name, agent_loop, policy, env_cfg, run_cfg, index, stop, record, routines, show_cfg.get("pause_s", 3.0), show_cfg.get("hold_after_order_s", 12.0)),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # The server supplies the specs on reset, not on construction. Release the probe immediately.
    env_for_spec = build_env_factory({**env_cfg, "task": "idle"})()
    try:
        env_for_spec.reset(seed=0, model="spec-probe")
    finally:
        env_for_spec.close()

    timer = None
    duration_s = show_cfg.get("duration_s")
    if duration_s is not None:
        if float(duration_s) <= 0:
            raise ValueError("show.duration_s must be positive")
        timer = threading.Timer(float(duration_s), stop.set)
        timer.daemon = True
        timer.start()

    sink_events.send(phase="collect", total=0, models=len(policies))
    for i, p in enumerate(policies):
        spawn(p, i)
        if stop.wait(0.4):
            break

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
                max_steps=run_cfg.get("max_steps", 110), stop=stop,
            )
            with lock:
                farm.extend(eps)
                with (out / "episodes_farm.jsonl").open("a", encoding="utf-8") as f:
                    for e in eps:
                        f.write(e.model_dump_json() + "\n")

    farm_thread = threading.Thread(target=guarded, args=("data-farm", bootstrap), daemon=True)
    farm_thread.start()

    # -- trainer -----------------------------------------------------------------------------------
    hot = HotPolicy(TRAINED_NAME, pace_s=max(0, float(show_cfg.get("local_pace_s", 0))))
    trained_started = False
    wins_at_last_round = 0
    first_round_wins = show_cfg.get("first_round_wins", 10)
    every = show_cfg.get("retrain_every_wins", 25)
    round_no = 0

    try:
        while not stop.is_set():
            if stop.wait(1.0):
                break
            with lock:
                frontier_wins = sum(e.success for e in episodes if e.model_id in names)
                ready_farm = not farm_thread.is_alive()
            need = first_round_wins if round_no == 0 else every
            if frontier_wins - wins_at_last_round < need or not ready_farm:
                continue
            with lock:
                pool = [*episodes, *farm]
            kept = select_episodes([e for e in pool if e.model_id != TRAINED_NAME], only_success=True)
            if len(kept) < show_cfg.get("min_train_episodes", 8):
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
                training_events_url=events_url, epoch_delay_s=train_cfg.get("epoch_delay_s", 0.0), stop=stop,
            )
            ckpt_path = save_checkpoint(ckpt, out / "bc.pt")
            model_json = checkpoint_to_json({**ckpt, "state_names": list(getattr(env_for_spec.observation_spec, "names", []) or [])}, name=TRAINED_NAME)
            (out / "policy.json").write_text(json.dumps(model_json), encoding="utf-8")
            if not post_model(f"{base_url}/sim/model", model_json):
                raise RuntimeError("trained policy saved, but the viewer rejected /sim/model")
            hot.inner = LearnedPolicy(ckpt_path, name=TRAINED_NAME)
            (out / f"training_round_{round_no}.json").write_text(json.dumps({
                "round": round_no, "episodes": len(kept), "from_collection_policies": from_llm,
                "from_scripted_bootstrap": len(kept) - from_llm, "samples": n_samples,
                "episode_ids": [e.episode_id for e in kept], "metrics": metrics,
            }, indent=2), encoding="utf-8")
            print("[train] metrics:", metrics)
            sink_events.send(phase="done", checkpoint=str(ckpt_path), round=round_no)
            if not trained_started:
                trained_started = True
                spawn(hot, len(policies))

    except InterruptedError:
        if not stop.is_set():
            raise
    finally:
        stop.set()
        if timer is not None:
            timer.cancel()
        # A provider HTTP request cannot be interrupted; it is bounded by its configured timeout.
        deadline = time.monotonic() + float(show_cfg.get("cleanup_timeout_s", 10))
        for t in [*threads, farm_thread]:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in [*threads, farm_thread] if t.is_alive()]
        if alive:
            print(f"[show] requests still finishing after cleanup timeout: {alive}", flush=True)
        sink_events.send(phase="stopped", rounds=round_no, collected=len(episodes))
    if errors:
        label, exc = errors[0]
        raise RuntimeError(f"show worker {label!r} failed") from exc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, help="stop collecting/training after this many seconds")
    args = ap.parse_args(argv)
    stop = threading.Event()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.duration is not None:
        cfg.setdefault("show", {})["duration_s"] = args.duration
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda _sig, _frame: stop.set())
        run_show(cfg, Path(args.out), stop)
    finally:
        stop.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
