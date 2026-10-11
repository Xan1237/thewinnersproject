"""OpenAI Chat Completions adapter. Also the base for any OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

from winners.policies.llm import LLMPolicy, require_env


class OpenAIPolicy(LLMPolicy):
    def __init__(
        self,
        model: str = "gpt-4o",
        name: str | None = None,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str | None = "OPENAI_API_KEY",
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        **kw,
    ):
        super().__init__(name or f"openai:{model}", model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        # merged into the request JSON, e.g. OpenRouter provider pinning:
        # {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        headers = dict(self.extra_headers)
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {require_env(self.api_key_env)}"
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
                **self.extra_body,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return text, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
