"""Live mode: the robots stand in the room and obey spoken orders.

    python -m winners.live --config configs/g1_arena.yaml --out runs/g1-live-orders

Every policy of the config gets a robot. The operator talks into the viewer's mic button; the
sim server turns the speech into one task (POST /sim/voice -> /sim/command) and every agent
picks it up, re-tasks its robot where it stands and starts working. A new order interrupts
whatever the agents are doing. Every attempt is recorded like an arena episode.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from pathlib import Path

import yaml

from winners.registry import build_env_factory, build_policy
from winners.runner import run_episode
from winners.schema import Episode


class _Orders:
    """Polls the sim for the latest order, shared by the agent loop and its interrupt hook."""

    def __init__(self, env, poll_s: float = 0.4):
        self.env = env
        self.poll_s = poll_s
        self.seen = env.command().get("id", 0)  # orders issued before we started are history
        self._checked = 0.0
        self.text = ""  # the operator's words behind the latest order

    def new_task(self) -> str | None:
        """The task of an order we have not acted on yet (None if none, or not understood)."""
        now = time.monotonic()
        if now - self._checked < self.poll_s:
            return None
        self._checked = now
        try:
            cmd = self.env.command()
        except Exception:
            return None
        if cmd.get("id", 0) <= self.seen:
            return None
        self.seen = cmd["id"]
        self.text = str(cmd.get("text") or "")
        return cmd.get("task") or None


def agent_loop(
    policy,
    env_cfg: dict,
    run_cfg: dict,
    index: int,
    stop: threading.Event,
    sink,
    routines: list[str] | None = None,
    pause_s: float = 3.0,
    hold_after_order_s: float = 12.0,
    ready: threading.Event | None = None,
) -> None:
    """One robot, one agent. Spoken orders always win; between orders the robot works through
    `routines` (little tasks like "grab a coffee, then sit") so the room never goes quiet."""
    env = build_env_factory({**env_cfg, "task": "idle"})()
    env.reset(seed=index + 1, model=policy.name)  # the robot appears and waits
    orders = _Orders(env)
    rng = random.Random(index * 7919 + 5)
    last_routine = None
    seed = 1000 * (index + 1)
    task = None
    instruction = None
    from_order = False

    def pause(seconds: float) -> None:
        """Stand there for a moment (so the animation can be seen) unless a new order comes in."""
        nonlocal task, instruction, from_order
        end = time.monotonic() + seconds
        while not stop.is_set() and time.monotonic() < end:
            new = orders.new_task()
            if new:
                task, instruction, from_order = new, orders.text or None, True
                return
            time.sleep(0.2)

    if ready is not None:
        ready.wait()
    while not stop.is_set():
        if not task:
            new = orders.new_task()
            if new:
                task, instruction, from_order = new, orders.text or None, True
            elif routines:
                choices = [r for r in routines if r != last_routine] or routines
                task = last_routine = rng.choice(choices)
                instruction, from_order = None, False
        if not task:
            time.sleep(0.2)
            continue
        try:
            env.retask(task, instruction=instruction)
        except Exception as exc:  # e.g. no free chair left: report and carry on
            print(f"[{policy.name}] cannot do {task!r}: {exc}")
            task = None
            time.sleep(1.0)
            continue
        task = None
        seed += 1
        ep = run_episode(
            env,
            policy,
            seed,
            run_cfg.get("decision_interval", 1),
            run_cfg.get("max_steps", 60),
            resume=True,
            close=False,
            interrupt=orders.new_task,
        )
        if ep.steps:
            sink(ep)
        task = ep.meta.get("next_task")
        if task:
            instruction, from_order = orders.text or None, True
        verdict = "interrupted" if ep.meta.get("interrupted") else ("done" if ep.success else "failed")
        print(f"[{policy.name}] {ep.task[:50]!r}: {verdict} after {len(ep.steps)} steps")
        if not task:
            pause(hold_after_order_s if from_order else pause_s)
    env.close()


def run_live(cfg: dict, out: Path, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    path = out / "episodes.jsonl"

    def sink(ep: Episode) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(ep.model_dump_json() + "\n")

    env_cfg = cfg.get("env", {})
    policies = [build_policy(p) for p in cfg["policies"]]
    threads = [
        threading.Thread(target=agent_loop, args=(p, env_cfg, cfg.get("run", {}), i, stop, sink), daemon=True)
        for i, p in enumerate(policies)
    ]
    for t in threads:
        t.start()
    print(f"{len(threads)} agents waiting for orders (Ctrl+C to stop)")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(timeout=5)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_live(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), Path(args.out))


if __name__ == "__main__":
    main()
