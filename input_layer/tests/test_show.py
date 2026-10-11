"""Lifecycle checks with in-memory environments; these are not demo episodes."""

import json
import threading

import pytest

from winners.env.mock import MockReachEnv
from winners.policies.scripted import FixedPolicy
from winners.runner import run_episode
from winners.schema import ActionSpec, Observation, ObservationSpec


class LiveEnv:
    def __init__(self, *, broken=False):
        self.broken = broken
        self.closed = False
        self.task_description = "test task"
        self.action_spec = ActionSpec(names=["forward", "right", "turn", "interact"], low=[-1]*4, high=[1]*4)
        self.observation_spec = ObservationSpec(names=[])

    def reset(self, seed=None, model=None):
        self.observation_spec = ObservationSpec(names=[f"state_{i}" for i in range(8)])
        return self.current_observation()

    def current_observation(self):
        return Observation(state=[0.0]*8)

    def command(self):
        return {"id": 0}

    def retask(self, task, instruction=None):
        self.task_description = task

    def step(self, action):
        if self.broken:
            raise RuntimeError("sim lost during step")
        return self.current_observation(), 1.0, True, {"success": True}

    def close(self):
        self.closed = True


class QuietEvents:
    active = False

    def __init__(self, *args, **kwargs):
        pass

    def send(self, **kwargs):
        return True


def setup_show(monkeypatch, *, broken=False):
    from winners import live, show
    from winners.train import bc

    envs = []

    def factory(cfg):
        def create():
            env = LiveEnv(broken=broken)
            envs.append(env)
            return env
        return create

    monkeypatch.setattr(live, "build_env_factory", factory)
    monkeypatch.setattr(show, "build_env_factory", factory)
    monkeypatch.setattr(show, "TrainingEventSink", QuietEvents)
    monkeypatch.setattr(bc, "TrainingEventSink", QuietEvents)
    monkeypatch.setattr(show, "post_model", lambda *args: True)
    return show, envs


def cfg(**show_options):
    return {
        "policies": [{"type": "scripted-goto", "name": "unit-test-policy"}],
        "show": {"duration_s": 0.6, "pause_s": 0.01, "routines": ["test"], **show_options},
        "train": {"epochs": 1, "hidden": [4]},
    }


def test_stopping_during_policy_call_sends_no_late_action():
    stop = threading.Event()

    class StopsDuringDecision(FixedPolicy):
        def decide(self, *args):
            stop.set()
            return super().decide(*args)

    ep = run_episode(MockReachEnv(), StopsDuringDecision([1, 1]), 0, cancelled=stop.is_set)
    assert not ep.steps
    assert ep.meta["cancelled"] and not ep.success


def test_duration_closes_every_robot_and_records_actual_attempts(monkeypatch, tmp_path):
    show, envs = setup_show(monkeypatch)
    show.run_show(cfg(), tmp_path)
    assert len(envs) >= 2 and all(e.closed for e in envs)
    episodes = [json.loads(line) for line in (tmp_path / "episodes_all.jsonl").read_text().splitlines()]
    assert episodes and all(ep["success"] for ep in episodes)


def test_worker_failure_is_reported_and_closes_session(monkeypatch, tmp_path):
    show, envs = setup_show(monkeypatch, broken=True)
    with pytest.raises(RuntimeError, match="show worker") as exc:
        show.run_show(cfg(duration_s=3), tmp_path)
    assert "sim lost during step" in str(exc.value.__cause__)
    assert all(e.closed for e in envs)


def test_training_exports_server_supplied_input_names_and_provenance(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    show, envs = setup_show(monkeypatch)
    show.run_show(cfg(duration_s=3, first_round_wins=2, min_train_episodes=2, retrain_every_wins=10000), tmp_path)
    model = json.loads((tmp_path / "policy.json").read_text())
    assert model["inputs"] == [f"state_{i}" for i in range(8)]
    provenance = json.loads((tmp_path / "training_round_1.json").read_text())
    assert provenance["from_scripted_bootstrap"] == 0
    assert provenance["from_collection_policies"] == len(provenance["episode_ids"])
    assert all(e.closed for e in envs)
