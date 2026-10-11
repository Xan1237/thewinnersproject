"""JevOrchestratedPolicy against a mocked OpenRouter (no network)."""

import json

import httpx
import pytest

from winners.policies.decision import MENU, JevOrchestratedPolicy
from winners.schema import ActionSpec, Observation, Step

SPEC = ActionSpec(names=["forward", "right", "turn", "interact"], low=[-1, -1, -1, 0], high=[1, 1, 1, 1])
OBS = Observation(state=[0, 0, 0, 5.0, 0.5, 5.0, 0, 1], text="You are at x=0.")


def _policy(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JevOrchestratedPolicy("some/orchestrator", "t", client=client, **kw)


def _step(t):
    return Step(t=t, state=OBS.state, raw_action=None, safe_action=[0] * 4, decided=True, reward=0, source_model="t")


def test_jev_picks_the_move_and_the_orchestrator_plans_on_schedule(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append((req.url.path, body))
        assert req.headers["authorization"] == "Bearer test-key"
        if req.url.path == "/api/alpha/decisions":
            q = body["questions"]["move"]
            assert set(q["criteria"]) == set(MENU)
            assert "Plan A" in q["instructions"]
            assert "bearing" in body["state"]["situation"]
            return httpx.Response(200, json={"answers": {"move": {"type": "choice", "choice": "veer_right",
                "probabilities": {"veer_right": 0.8, "forward": 0.2}, "confidence": 0.8}}, "usage": {"cost": 0.00002}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Plan A: head right."}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    p = _policy(handler, replan_every=3, price_in=10.0, price_out=50.0)
    a = p.decide(OBS, "task", SPEC, [])
    assert a.values == MENU["veer_right"][0] and a.label == "veer right (80%)"
    assert a.meta["detail"]["plan"] == "Plan A: head right." and a.meta["detail"]["replanned"]
    assert a.meta["cost_usd"] == pytest.approx(0.01 + 0.005 + 0.00002)
    for t in range(1, 3):  # plan reused for the next ticks
        assert not p.decide(OBS, "task", SPEC, [_step(i) for i in range(t)]).meta["detail"]["replanned"]
    assert p.decide(OBS, "task", SPEC, [_step(i) for i in range(3)]).meta["detail"]["replanned"]
    assert [path for path, _ in calls].count("/api/v1/chat/completions") == 2


def test_failing_jev_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    p = _policy(lambda req: httpx.Response(400, json={"error": "bad"}), max_retries=0)
    a = p.decide(OBS, "task", SPEC, [])
    assert a.values == [] and "error" in a.meta
