"""Shared logic for LLM-backed policies. Provider subclasses only implement `_complete`."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from winners.prompting import SYSTEM_PROMPT, ActionParseError, build_prompt, parse_reply
from winners.schema import Action, ActionSpec, Observation, Step


class MissingApiKey(RuntimeError):
    pass


def require_env(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise MissingApiKey(f"environment variable {var} is not set")
    return value


class LLMPolicy(ABC):
    """Prompt -> reply -> parsed action, with retries on unparseable output.

    `price_in` / `price_out` are USD per million tokens (0 = untracked, e.g. local models).
    """

    def __init__(
        self,
        name: str,
        model: str,
        *,
        client: httpx.Client | None = None,
        max_retries: int = 2,
        timeout_s: float = 60.0,
        obs_names: list[str] | None = None,
        price_in: float = 0.0,
        price_out: float = 0.0,
    ):
        self.name = name
        self.model = model
        self.client = client or httpx.Client(timeout=timeout_s)
        self.max_retries = max_retries
        self.obs_names = obs_names
        self.price_in = price_in
        self.price_out = price_out

    @abstractmethod
    def _complete(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        """Return (reply_text, {"input_tokens": n, "output_tokens": m})."""

    def decide(
        self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]
    ) -> Action:
        prompt = build_prompt(obs, task, action_spec, history, self.obs_names)
        started = time.perf_counter()
        tokens_in = tokens_out = 0
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                text, usage = self._complete(SYSTEM_PROMPT, prompt)
            except MissingApiKey:
                raise
            except httpx.HTTPError as exc:
                last_error = f"http: {exc}"
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(0.6 * 2**attempt, 5.0))  # rate limited / upstream hiccup: back off
                continue
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
            try:
                values, label = parse_reply(text, action_spec)
            except ActionParseError as exc:
                last_error = str(exc)
                continue
            return Action(
                values=values,
                label=label,
                meta=self._meta(started, tokens_in, tokens_out, attempt),
            )
        meta = self._meta(started, tokens_in, tokens_out, self.max_retries)
        meta["error"] = last_error
        return Action(values=[], meta=meta)

    def _meta(self, started: float, tokens_in: int, tokens_out: int, attempt: int) -> dict[str, Any]:
        cost = (tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000
        return {
            "latency_s": time.perf_counter() - started,
            "cost_usd": cost,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "attempts": attempt + 1,
        }
