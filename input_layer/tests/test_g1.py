"""G1SimEnv against the fake sim, label/joints plumbing, provider extras, training hook."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from fake_sim import FakeSim

from winners.arena import run_arena
from winners.env.g1 import G1SimEnv, G1SimError
from winners.env.mock import MockReachEnv
from winners.policies.openai import OpenAIPolicy
from winners.policies.openrouter import OpenRouterPolicy
from winners.policies.scripted import FixedPolicy, GoalSeekPolicy, GreedyReachPolicy
from winners.prompting import ActionParseError, build_prompt, parse_action, parse_reply
from winners.registry import build_env_factory, build_policy
from winners.runner import run_episode
from winners.schema import Action, ActionSpec, Episode, Observation, Step

SPEC = ActionSpec(names=["dx", "dy"], low=[-1, -1], high=[1, 1])
OBS = Observation(t=0, state=[0.1, 0.2, 0.9, 0.8])


@pytest.fixture()
def sim():
    with FakeSim() as s:
        yield s


# -- env against the fake server ---------------------------------------------------------


def test_reset_builds_specs_and_creates_session(sim):
    env = G1SimEnv(base_url=sim.url, task="goto:armchair_blue", model="tag-a", max_steps=30)
    obs = env.reset(seed=3)
    assert env.action_spec.names == ["forward", "right", "turn", "interact"]
    assert env.observation_spec.names[3] == "goal_forward"
    assert len(obs.state) == len(env.observation_spec.names) == 8
    assert "armchair_blue" in env.task_description
    [(method, path, body)] = [r for r in sim.requests if r[0] == "POST"]
    assert (method, path) == ("POST", "/sim/sessions")
    assert body == {"task": "goto:armchair_blue", "seed": 3, "model": "tag-a", "max_steps": 30}
    assert env.manifest()["objects"][0]["id"] == "armchair_blue"


def test_step_maps_response_and_sends_label_and_model(sim):
    env = G1SimEnv(base_url=sim.url)
    env.reset(seed=1)
    obs, reward, done, info = env.step([1.0, 0.0, 0.0, 0.0], label="walk", model="m1")
    assert obs.t == 1 and len(obs.state) == 8 and "Goal" in obs.text
    assert isinstance(reward, float) and done is False
    assert info["success"] is False and info["joints"] and "position" in info
    ev = sim.step_events()[-1]
    assert ev["label"] == "walk" and ev["model"] == "m1" and ev["action"] == [1.0, 0.0, 0.0, 0.0]


def test_close_deletes_session_and_reset_replaces_it(sim):
    env = G1SimEnv(base_url=sim.url)
    env.reset(seed=0)
    first = env.session_id
    env.reset(seed=1)  # old session is deleted first
    assert first not in sim.sessions and env.session_id in sim.sessions
    env.close()
    assert not sim.sessions and env.session_id is None
    env.close()  # idempotent


def test_errors_are_explicit(sim):
    env = G1SimEnv(base_url=sim.url, task="goto:nonexistent")
    with pytest.raises(G1SimError) as e:
        env.reset(seed=0)
    assert e.value.status == 400
    with pytest.raises(G1SimError):
        G1SimEnv(base_url=sim.url).step([0, 0, 0, 0])  # before reset
    with pytest.raises(G1SimError):
        G1SimEnv(base_url="http://127.0.0.1:1").reset()  # unreachable


def test_fetch_coffee_needs_interact_near_station(sim):
    env = G1SimEnv(base_url=sim.url, task="fetch_coffee", max_steps=80)
    ep = run_episode(env, GoalSeekPolicy(), seed=2, max_steps=80)
    assert ep.success and ep.steps[-1].safe_action[3] == 1.0
    assert ep.steps[-1].label == "grab the coffee"


def test_runner_records_label_and_joints_and_passes_model(sim):
    env = G1SimEnv(base_url=sim.url, max_steps=60)
    ep = run_episode(env, GoalSeekPolicy(name="teacher"), seed=0, max_steps=60)
    assert ep.success
    assert all(s.label for s in ep.steps)
    assert all(s.joints and "hip_l" in s.joints for s in ep.steps)
    assert {e["model"] for e in sim.step_events()} == {"teacher"}
    assert all(e["label"] for e in sim.step_events())
    assert not sim.sessions  # runner closed the session
    again = Episode.model_validate_json(ep.model_dump_json())
    assert again.steps[0].label == ep.steps[0].label and again.steps[0].joints == ep.steps[0].joints


def test_runner_holds_label_between_decisions_and_labels_model_errors(sim):
    env = G1SimEnv(base_url=sim.url, max_steps=10)
    ep = run_episode(env, GoalSeekPolicy(), seed=0, decision_interval=3, max_steps=6)
    assert ep.steps[1].label == ep.steps[0].label and not ep.steps[1].decided

    class Boom:
        name = "boom"

        def decide(self, *a):
            raise RuntimeError("x")

    ep = run_episode(env, Boom(), seed=0, max_steps=2)
    assert ep.steps[0].label == "hold (model error)" and "model_error" in ep.steps[0].flags


def test_runner_still_works_with_envs_without_new_kwargs():
    class OldEnv(MockReachEnv):
        def reset(self, seed=None):  # old signature, no model kwarg
            return super().reset(seed)

        def step(self, action):  # old signature, no label/model
            return super().step(action)

    ep = run_episode(OldEnv(), FixedPolicy([0.0, 0.0]), seed=0, max_steps=3)
    assert ep.length == 3 and ep.steps[0].joints is None and ep.steps[0].label is None


def test_mock_env_accepts_new_kwargs():
    env = MockReachEnv()
    env.reset(seed=0, model="m")
    env.step([0.0, 0.0], label="x", model="m")


def test_registry_builds_g1_env_and_new_policies(sim):
    env = build_env_factory({"type": "g1-sim", "base_url": sim.url, "task": "fetch_coffee"})()
    assert isinstance(env, G1SimEnv) and env.task == "fetch_coffee"
    assert isinstance(build_policy({"type": "scripted-goto", "name": "g"}), GoalSeekPolicy)
    p = build_policy({"type": "openrouter", "model": "a/b", "extra_body": {"x": 1}})
    assert isinstance(p, OpenRouterPolicy) and p.name == "openrouter:a/b"


def test_arena_runs_parallel_policies_on_separate_sessions(sim):
    factory = build_env_factory({"type": "g1-sim", "base_url": sim.url, "max_steps": 60})
    pols = [GoalSeekPolicy("a"), GoalSeekPolicy("b", noise=0.05, seed=1)]
    eps = run_arena(factory, pols, episodes=3, max_steps=60)
    assert len(eps) == 6 and all(e.success for e in eps)
    assert {e["model"] for e in sim.step_events()} == {"a", "b"}


# -- prompting ---------------------------------------------------------------------------


def test_parse_reply_labels_and_tolerance():
    assert parse_reply('{"action": [0.1, -0.2], "label": "raise right arm"}', SPEC) == (
        [0.1, -0.2], "raise right arm")
    assert parse_reply('```json\n{"action": [1, 0], "label": "  go\\n  left "}\n```', SPEC) == (
        [1.0, 0.0], "go left")
    assert parse_reply('Sure: {"action": [1, 0]} ok', SPEC) == ([1.0, 0.0], None)
    assert parse_reply('{"label": "x {y}", "action": [0, 1]}', SPEC) == ([0.0, 1.0], "x {y}")
    assert parse_reply('{"action": {"dx": 1, "dy": 0.5}, "label": 5}', SPEC) == ([1.0, 0.5], None)
    assert parse_reply("[0.5, 0.5]", SPEC) == ([0.5, 0.5], None)
    assert len(parse_reply('{"action": [0,0], "label": "' + "x" * 500 + '"}', SPEC)[1]) == 120
    assert parse_action('{"action": [0.1, 0.2], "label": "z"}', SPEC) == [0.1, 0.2]
    with pytest.raises(ActionParseError):
        parse_reply('{"label": "only label"}', SPEC)


def test_prompt_asks_for_label_and_shows_history_labels():
    hist = [Step(t=0, state=[0, 0], raw_action=[1, 0], safe_action=[1, 0], decided=True,
                 reward=0.0, source_model="m", label="turn left")]
    prompt = build_prompt(OBS, "reach", SPEC, hist)
    assert '"label"' in prompt and "turn left" in prompt


# -- providers ---------------------------------------------------------------------------


def test_openai_extra_body_headers_and_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekret")
    seen = {}

    def handler(req):
        seen["headers"], seen["body"], seen["url"] = req.headers, json.loads(req.content), str(req.url)
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"action": [0, 1], "label": "go up"}'}}]})

    pin = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
    p = OpenRouterPolicy("qwen/x", extra_body=pin, extra_headers={"X-Title": "winners"},
                         api_key_env="MY_KEY", client=httpx.Client(transport=httpx.MockTransport(handler)))
    a = p.decide(OBS, "t", SPEC, [])
    assert a.values == [0, 1] and a.label == "go up"
    assert seen["body"]["provider"] == pin["provider"] and seen["body"]["model"] == "qwen/x"
    assert seen["headers"]["authorization"] == "Bearer sekret" and seen["headers"]["x-title"] == "winners"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"

    q = OpenAIPolicy("luna-id", base_url="http://luna.local/v1", api_key_env="MY_KEY",
                     extra_body={"top_p": 0.5}, client=httpx.Client(transport=httpx.MockTransport(handler)))
    q.decide(OBS, "t", SPEC, [])
    assert seen["body"]["top_p"] == 0.5 and seen["url"] == "http://luna.local/v1/chat/completions"


# -- training events hook ----------------------------------------------------------------


class _Collector:
    def __init__(self):
        self.events = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                outer.events.append(json.loads(self.rfile.read(n)))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/sim/training"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _tiny_dataset():
    from winners.dataset import build_dataset, select_episodes

    eps = select_episodes(run_arena(MockReachEnv, [GreedyReachPolicy()], episodes=6, max_steps=100))
    return build_dataset(eps)


def test_training_events_are_posted_per_epoch_and_after_eval():
    pytest.importorskip("torch")
    from winners.policies.learned import LearnedPolicy
    from winners.train.bc import train_bc
    from winners.train.evaluate import evaluate_policy

    ds = _tiny_dataset()
    col = _Collector()
    try:
        env = MockReachEnv()
        ckpt, _ = train_bc(ds, env.action_spec, hidden=(16,), epochs=4, training_events_url=col.url)
        result, _ = evaluate_policy(env, LearnedPolicy(ckpt), [100, 101], max_steps=20,
                                    training_events_url=col.url)
    finally:
        col.close()
    epochs = [e for e in col.events if e["phase"] == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2, 3, 4]
    assert all(e["type"] == "training" and e["loss"] >= 0 and e["val_loss"] >= 0 for e in epochs)
    assert col.events[-1] == {"type": "training", "phase": "eval", "n": 2,
                              "success_rate": result["success_rate"]}


def test_training_events_unreachable_url_never_fails_training():
    pytest.importorskip("torch")
    from winners.train.bc import train_bc
    from winners.train.evaluate import evaluate_policy

    ds = _tiny_dataset()
    env = MockReachEnv()
    ckpt, metrics = train_bc(ds, env.action_spec, hidden=(8,), epochs=5,
                             training_events_url="http://127.0.0.1:1/sim/training")
    assert "val_mse" in metrics
    from winners.policies.learned import LearnedPolicy

    result, _ = evaluate_policy(env, LearnedPolicy(ckpt), [1], max_steps=5,
                                training_events_url="http://127.0.0.1:1/x")
    assert result["n"] == 1


def test_g1_arena_config_builds_without_network():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).parent.parent / "configs" / "g1_arena.yaml").read_text())
    assert cfg["env"]["type"] == "g1-sim"
    assert isinstance(build_env_factory(cfg["env"])(), G1SimEnv)
    pols = {p["name"]: build_policy(p) for p in cfg["policies"]}
    from winners.policies.decision import JevOrchestratedPolicy

    assert len(pols) >= 5 and all(isinstance(p, JevOrchestratedPolicy) for p in pols.values())
    assert {p.jev_model for p in pols.values()} == {"typesafe/jev-1.13"}  # same decision model for all
    assert len({p.orchestrator for p in pols.values()}) == len(pols)  # different orchestrators
    assert all(p.api_key_env == "OPENROUTER_API_KEY" for p in pols.values())
    qwen = pols["qwen-3.8-27b"].extra_body["provider"]
    assert qwen == {"only": ["cerebras"], "allow_fallbacks": False}
