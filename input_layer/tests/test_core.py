import math

import pytest

from winners.env.mock import MockReachEnv
from winners.policies.scripted import FixedPolicy, GreedyReachPolicy
from winners.prompting import ActionParseError, build_prompt, parse_action
from winners.recorder import load_episodes, save_episodes
from winners.runner import run_episode
from winners.safety import SafetyFilter
from winners.schema import Action, ActionSpec, Observation

SPEC = ActionSpec(names=["a", "b"], low=[-1, -1], high=[1, 1])


def test_action_spec_validation_and_clip():
    with pytest.raises(ValueError):
        ActionSpec(names=["a"], low=[0, 0], high=[1])
    with pytest.raises(ValueError):
        ActionSpec(names=["a"], low=[2], high=[1])
    assert SPEC.clip([5, -5]) == [1, -1]
    assert SPEC.hold() == [0, 0]
    assert SPEC.contains([0.5, 0.5]) and not SPEC.contains([2, 0])


def test_safety_filter():
    f = SafetyFilter(SPEC)
    assert f.sanitize([0.2, 0.3]) == ([0.2, 0.3], [])
    assert f.sanitize([5, 0]) == ([1, 0], ["clamped"])
    assert f.sanitize([math.nan, 0]) == ([0, 0], ["non_finite"])
    assert f.sanitize([1]) == ([0, 0], ["wrong_dim"])
    assert f.sanitize(None) == ([0, 0], ["no_action"])


def test_parse_action_variants():
    assert parse_action('{"action": [0.1, -0.2]}', SPEC) == [0.1, -0.2]
    assert parse_action('Sure! {"action": [1, 0]} hope that helps', SPEC) == [1.0, 0.0]
    for bad in ["no json", '{"action": [1]}', '{"action": ["x", "y"]}', '{"action": [true, false]}']:
        with pytest.raises(ActionParseError):
            parse_action(bad, SPEC)


def test_build_prompt_mentions_task_limits_and_history():
    obs = Observation(t=3, state=[0.1, 0.2])
    prompt = build_prompt(obs, "reach it", SPEC, [], obs_names=["x", "y"])
    assert "reach it" in prompt and "x=0.1000" in prompt and "a in [-1.0, 1.0]" in prompt


def test_mock_env_success_and_timeout():
    env = MockReachEnv(max_steps=3)
    obs = env.reset(seed=1)
    for _ in range(3):
        obs, _, done, info = env.step([0.0, 0.0])
    assert done and info["success"] is False
    # same seed -> same scene
    assert MockReachEnv().reset(seed=7).state == MockReachEnv().reset(seed=7).state


def test_greedy_policy_solves_mock_env_and_records_full_rate():
    ep = run_episode(MockReachEnv(), GreedyReachPolicy(), seed=3, max_steps=200)
    assert ep.success and ep.length > 0
    assert all(s.decided for s in ep.steps)
    assert ep.model_id == "scripted-greedy"


def test_decision_interval_holds_actions_between_queries():
    ep = run_episode(MockReachEnv(), GreedyReachPolicy(), seed=2, decision_interval=4, max_steps=200)
    assert [s.decided for s in ep.steps[:8]] == [True, False, False, False, True, False, False, False]
    assert ep.steps[1].safe_action == ep.steps[0].safe_action


def test_out_of_range_action_is_clamped_and_logged():
    ep = run_episode(MockReachEnv(), FixedPolicy([9.0, -9.0]), seed=0, max_steps=3)
    assert ep.steps[0].raw_action == [9.0, -9.0]
    assert ep.steps[0].safe_action == [1.0, -1.0]
    assert "clamped" in ep.steps[0].flags


def test_crashing_policy_does_not_crash_episode():
    class Boom:
        name = "boom"

        def decide(self, *a, **k):
            raise RuntimeError("model fell over")

    ep = run_episode(MockReachEnv(), Boom(), seed=0, max_steps=3)
    assert not ep.success and ep.length == 3
    assert all("model_error" in s.flags for s in ep.steps)
    assert ep.steps[0].safe_action == [0.0, 0.0]


def test_episode_roundtrip(tmp_path):
    ep = run_episode(MockReachEnv(), GreedyReachPolicy(), seed=1, max_steps=50)
    path = save_episodes([ep], tmp_path / "e.jsonl")
    [back] = load_episodes(path)
    assert back == ep
