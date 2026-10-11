"""Google Gemini generateContent adapter."""

from __future__ import annotations

from winners.policies.llm import LLMPolicy, require_env


class GeminiPolicy(LLMPolicy):
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str = "gemini-2.5-flash", name: str | None = None, **kw):
        super().__init__(name or f"gemini:{model}", model, **kw)

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        resp = self.client.post(
            f"{self.BASE}/{self.model}:generateContent",
            headers={"x-goog-api-key": require_env("GEMINI_API_KEY")},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return text, {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
        }
