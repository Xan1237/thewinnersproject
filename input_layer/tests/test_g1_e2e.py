"""Scripted teachers -> arena -> G1SimEnv (fake sim server) -> labelled episodes -> dataset
-> BC training -> LearnedPolicy driving the sim. No network beyond loopback."""

import pytest
from fake_sim import FakeSim

from winners.arena import leaderboard, run_arena
from winners.dataset import build_dataset, select_episodes
from winners.policies.scripted import FixedPolicy, GoalSeekPolicy
from winners.recorder import load_episodes, save_episodes
from winners.registry import build_env_factory
from winners.runner import run_episode


def test_g1_pipeline_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from winners.policies.learned import LearnedPolicy
    from winners.train.bc import save_checkpoint, train_bc
    from winners.train.evaluate import evaluate_policy

    with FakeSim() as sim:
        factory = build_env_factory(
            {"type": "g1-sim", "base_url": sim.url, "task": "goto:armchair_blue", "max_steps": 60}
        )
        teachers = [
            GoalSeekPolicy("teacher-a", noise=0.02, seed=1),
            GoalSeekPolicy("teacher-b", noise=0.08, seed=2),
            FixedPolicy([0.0, 0.0, 0.0, 0.0], "useless"),
        ]
        eps = run_arena(factory, teachers, episodes=30, max_steps=60)
        assert len(eps) == 90
        rows = {r["model"]: r for r in leaderboard(eps)}
        assert rows["teacher-a"]["success_rate"] == 1.0
        assert rows["useless"]["success_rate"] == 0.0

        # every recorded step of an LLM-like teacher carries label + joints
        good = [e for e in eps if e.model_id.startswith("teacher")]
        assert all(s.label and s.joints for e in good for s in e.steps)
        assert sim.step_events("teacher-a") and all(e["label"] for e in sim.step_events("teacher-a"))
        assert not sim.sessions

        # persistence keeps labels and joints
        path = save_episodes(good[:2], tmp_path / "episodes.jsonl")
        back = load_episodes(path)
        assert back[0].steps[0].label == good[0].steps[0].label
        assert back[0].steps[0].joints == good[0].steps[0].joints

        # dataset -> BC
        kept = select_episodes(eps, teachers=["teacher-a", "teacher-b"])
        ds = build_dataset(kept)
        assert ds.x_train.shape[1] == 8 and ds.y_train.shape[1] == 4
        env = factory()
        ckpt, metrics = train_bc(ds, env.action_spec, hidden=(64, 64), epochs=80)
        assert metrics["val_mse"] < 0.2

        # the free local policy drives the sim on held-out seeds
        learned = LearnedPolicy(save_checkpoint(ckpt, tmp_path / "bc.pt"))
        result, held_out = evaluate_policy(
            env, learned, seeds=list(range(1000, 1010)), max_steps=60,
            training_events_url=sim.url + "/sim/training",
        )
        assert result["n"] == 10 and result["success_rate"] >= 0.8
        assert {e["model"] for e in sim.step_events()} >= {"learned-bc", "teacher-a"}
        assert all(s.joints for s in held_out[0].steps)
        # train_bc posted nothing here (no url), but eval was broadcast
        assert sim.events[-1]["phase"] == "eval" and sim.events[-1]["n"] == 10
        # a learned episode is a normal Episode too
        assert run_episode(env, learned, seed=1, max_steps=60).steps[0].state
