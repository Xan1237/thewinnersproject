"""Local / open models via any OpenAI-compatible server (Ollama, vLLM, llama.cpp, LM Studio).

This also covers other hosted providers that expose an OpenAI-compatible API: pass their
`base_url` and the name of the env var holding the key as `api_key_env`.
"""

from __future__ import annotations

from winners.policies.openai import OpenAIPolicy


class LocalPolicy(OpenAIPolicy):
    def __init__(
        self,
        model: str,
        name: str | None = None,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str | None = None,
        **kw,
    ):
        super().__init__(
            model, name or f"local:{model}", base_url=base_url, api_key_env=api_key_env, **kw
        )
