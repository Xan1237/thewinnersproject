"""Anthropic Messages API adapter."""

from __future__ import annotations

from winners.policies.llm import LLMPolicy, require_env


class ClaudePolicy(LLMPolicy):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-sonnet-5-5", name: str | None = None, **kw):
        super().__init__(name or f"claude:{model}", model, **kw)

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        resp = self.client.post(
            self.URL,
            headers={
                "x-api-key": require_env("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 256,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
