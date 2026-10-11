"""Scripted teachers -> arena -> recorded data -> dataset -> BC training -> learned policy."""

import pytest

from winners.arena import format_leaderboard, leaderboard, run_arena
from winners.dataset import build_dataset, select_episodes
from winners.env.mock import MockReachEnv
from winners.policies.scripted import FixedPolicy, GreedyReachPolicy
from winners.recorder import load_episodes, save_episodes
from winners.runner import run_episode


def teachers():
    return [
        GreedyReachPolicy("teacher-a", noise=0.05, seed=1),
        GreedyReachPolicy("teacher-b", noise=0.15, seed=2),
        FixedPolicy([0.0, 0.0], "useless"),
    ]


def test_arena_and_leaderboard(tmp_path):
    eps = run_arena(MockReachEnv, teachers(), episodes=8, max_steps=150)
    assert len(eps) == 24
    rows = {r["model"]: r for r in leaderboard(eps)}
    assert rows["teacher-a"]["success_rate"] == 1.0
    assert rows["useless"]["success_rate"] == 0.0
    assert leaderboard(eps)[-1]["model"] == "useless"
    assert "teacher-a" in format_leaderboard(leaderboard(eps))
    [e] = load_episodes(save_episodes(eps[:1], tmp_path / "x.jsonl"))
    assert e == eps[0]


def test_best_of_n_excludes_failures_and_filters_teachers():
    eps = run_arena(MockReachEnv, teachers(), episodes=6, max_steps=150)
    kept = select_episodes(eps)
    assert kept and all(e.success for e in kept)
    assert {e.model_id for e in kept} == {"teacher-a", "teacher-b"}
    only_a = select_episodes(eps, teachers=["teacher-a"])
    assert {e.model_id for e in only_a} == {"teacher-a"}
    fastest = select_episodes(eps, keep_fastest_fraction=0.5)
    assert len(fastest) < len(kept)


def test_dataset_splits_by_episode():
    eps = select_episodes(run_arena(MockReachEnv, teachers(), episodes=10, max_steps=150))
    ds = build_dataset(eps, val_fraction=0.2)
    assert ds.n_val_episodes >= 1 and ds.n_train_episodes + ds.n_val_episodes == len(eps)
    assert ds.x_train.shape[1] == 4 and ds.y_train.shape[1] == 2


def test_distilled_policy_matches_teachers(tmp_path):
    pytest.importorskip("torch")
    from winners.policies.learned import LearnedPolicy
    from winners.train.bc import load_checkpoint, save_checkpoint, train_bc

    env = MockReachEnv()
    eps = select_episodes(
        run_arena(MockReachEnv, teachers(), episodes=40, max_steps=150), teachers=["teacher-a", "teacher-b"]
    )
    ds = build_dataset(eps)
    ckpt, metrics = train_bc(ds, env.action_spec, hidden=(64, 64), epochs=60)
    assert metrics["val_mse"] < 0.3

    path = save_checkpoint(ckpt, tmp_path / "bc.pt")
    policy = LearnedPolicy(path)
    assert load_checkpoint(path)["in_dim"] == 4

    # held-out seeds the teachers never saw in training
    wins = [run_episode(env, policy, seed=1000 + i, max_steps=150).success for i in range(20)]
    assert sum(wins) / len(wins) >= 0.8
