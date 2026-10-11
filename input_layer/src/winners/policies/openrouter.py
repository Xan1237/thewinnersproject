"""OpenRouter: an OpenAI-compatible gateway to many models.

Pin the upstream provider (so results are comparable and never silently rerouted) with
`extra_body={"provider": {"order": ["<Provider>"], "allow_fallbacks": False}}`.
"""

from __future__ import annotations

from winners.policies.openai import OpenAIPolicy


class OpenRouterPolicy(OpenAIPolicy):
    def __init__(
        self,
        model: str,
        name: str | None = None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str | None = "OPENROUTER_API_KEY",
        **kw,
    ):
        super().__init__(
            model, name or f"openrouter:{model}", base_url=base_url, api_key_env=api_key_env, **kw
        )
