import json

import httpx
import pytest

from winners.policies.claude import ClaudePolicy
from winners.policies.gemini import GeminiPolicy
from winners.policies.llm import MissingApiKey
from winners.policies.local import LocalPolicy
from winners.policies.openai import OpenAIPolicy
from winners.schema import ActionSpec, Observation

SPEC = ActionSpec(names=["dx", "dy"], low=[-1, -1], high=[1, 1])
OBS = Observation(t=0, state=[0.1, 0.2, 0.9, 0.8])


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def decide(policy):
    return policy.decide(OBS, "reach the goal", SPEC, [])


def test_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def handler(req):
        seen["headers"], seen["body"] = req.headers, json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"action": [0.5, 0.5]}'}],
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    p = ClaudePolicy(client=client(handler), price_in=3.0, price_out=15.0)
    a = decide(p)
    assert a.values == [0.5, 0.5]
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["model"] == "claude-sonnet-5-5"
    assert a.meta["cost_usd"] == pytest.approx((100 * 3 + 10 * 15) / 1e6)


def test_openai_and_local(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    urls = []

    def handler(req):
        urls.append((str(req.url), req.headers.get("authorization")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"action": [0, 1]}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    assert decide(OpenAIPolicy(client=client(handler))).values == [0, 1]
    assert decide(LocalPolicy("llama3", client=client(handler))).values == [0, 1]
    assert urls[0] == ("https://api.openai.com/v1/chat/completions", "Bearer k")
    assert urls[1] == ("http://localhost:11434/v1/chat/completions", None)


def test_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")

    def handler(req):
        assert req.headers["x-goog-api-key"] == "g"
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"action": [1, -1]}'}]}}],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
            },
        )

    a = decide(GeminiPolicy(client=client(handler)))
    assert a.values == [1, -1] and a.meta["input_tokens"] == 7


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    replies = iter(["I think go right", '{"action": [0.3, 0.3]}'])

    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    a = decide(OpenAIPolicy(client=client(handler)))
    assert a.values == [0.3, 0.3] and a.meta["attempts"] == 2


def test_gives_up_with_error_meta(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})

    a = decide(OpenAIPolicy(client=client(handler), max_retries=1))
    assert a.values == [] and "no valid" in a.meta["error"] and a.meta["attempts"] == 2


def test_http_errors_are_contained(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    a = decide(OpenAIPolicy(client=client(lambda r: httpx.Response(500)), max_retries=0))
    assert a.values == [] and a.meta["error"].startswith("http:")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingApiKey):
        decide(ClaudePolicy(client=client(lambda r: httpx.Response(200))))
