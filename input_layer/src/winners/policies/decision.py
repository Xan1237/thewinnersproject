"""A decision model (Jev) drives the robot; a normal LLM orchestrates it.

Jev (TypeSafe AI) is a "System One" model: it returns a typed answer with probabilities instead
of text. Here it picks one of a fixed menu of moves every tick (~0.2 s). The orchestrator is an
ordinary chat LLM that every few ticks writes a short plan, which is handed to Jev as the
instructions of the question. Same Jev for every robot, a different orchestrator per robot:
the orchestrator is what the arena compares.

Jev is called through OpenRouter's decisions endpoint (`POST <base>/alpha/decisions`), the
orchestrator through `POST <base>/v1/chat/completions`, both with the OpenRouter key.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from winners.policies.llm import MissingApiKey
from winners.schema import Action, ActionSpec, Observation, Step

# move -> (action vector over [forward, right, turn, interact], what Jev is told it means)
# Turn is positive = left, matching the sim contract (robot/docs/SIM_API.md).
MENU: dict[str, tuple[list[float], str]] = {
    "forward": ([1.0, 0.0, 0.0, 0.0], "Walk straight ahead at full speed. Use when the target is roughly in front of you."),
    "forward_slow": ([0.4, 0.0, 0.0, 0.0], "Walk straight ahead slowly. Use when close to the target or squeezing past an obstacle."),
    "veer_left": ([0.8, 0.0, 0.5, 0.0], "Walk ahead while curving to the left. Use when the target is ahead but somewhat to your left."),
    "veer_right": ([0.8, 0.0, -0.5, 0.0], "Walk ahead while curving to the right. Use when the target is ahead but somewhat to your right."),
    "turn_left": ([0.0, 0.0, 1.0, 0.0], "Rotate in place to the left without moving. Use when the target is to your left or behind you on the left."),
    "turn_right": ([0.0, 0.0, -1.0, 0.0], "Rotate in place to the right without moving. Use when the target is to your right or behind you on the right."),
    "sidestep_left": ([0.0, -0.8, 0.0, 0.0], "Step sideways to the left without turning. Use to get around an obstacle that is directly ahead."),
    "sidestep_right": ([0.0, 0.8, 0.0, 0.0], "Step sideways to the right without turning. Use to get around an obstacle that is directly ahead."),
    "back_off": ([-0.6, 0.0, 0.0, 0.0], "Walk backwards. Use when blocked or too close to something."),
    "grab": ([0.0, 0.0, 0.0, 1.0], "Reach out and pick up the object in front of you. Only useful when the task is to pick something up and you are within 0.9 m of it."),
    "stop": ([0.0, 0.0, 0.0, 0.0], "Stand still. Use when the task is done."),
}

def bearing_text(obs: Observation) -> str:
    """Target direction as an angle, from the contract's goal_forward / goal_right (state[3], [4])."""
    if len(obs.state) < 6:
        return ""
    fwd, right = obs.state[3], obs.state[4]
    deg = math.degrees(math.atan2(right, fwd))  # 0 = dead ahead, + = to the right, 180 = behind
    side = "right" if deg >= 0 else "left"
    mag = abs(deg)
    if mag < 12:
        return f"Target bearing: {mag:.0f} degrees {side} (essentially dead ahead: walk forward)."
    if mag > 150:
        return f"Target bearing: {mag:.0f} degrees {side} (almost directly behind you)."
    return f"Target bearing: {mag:.0f} degrees to your {side}."


ORCHESTRATOR_SYSTEM = """You are the planner ("orchestrator") of a walking humanoid robot in a room.
Every tick a fast decision model picks ONE move from this menu, using your plan as guidance:
{menu}
Directions are from the robot's own point of view. Write the guidance for the next few ticks:
one or two short sentences, plain text (no JSON, no lists). The plan stays in force for several
ticks while the robot moves, so write it as CONDITIONAL strategy, not as a fixed move: e.g. "Turn
toward the target until it is within about 15 degrees of dead ahead, then walk forward; pass the
plant on its left; stop within 1.2 m." Mention obstacles to avoid and when to stop or grab. If the
robot seems stuck, change the strategy."""


class JevOrchestratedPolicy:
    """Policy: orchestrator LLM (plans every `replan_every` ticks) + Jev (picks every tick)."""

    def __init__(
        self,
        orchestrator: str,
        name: str | None = None,
        *,
        jev_model: str = "typesafe/jev-1.13",
        base_url: str = "https://openrouter.ai/api",
        api_key_env: str = "OPENROUTER_API_KEY",
        replan_every: int = 5,
        low_confidence: float = 0.5,
        orchestrator_extra_body: dict[str, Any] | None = None,
        price_in: float = 0.0,  # orchestrator USD per 1M input tokens
        price_out: float = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ):
        self.orchestrator = orchestrator
        self.name = name or f"{orchestrator}+jev"
        self.jev_model = jev_model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.replan_every = max(1, replan_every)
        self.low_confidence = low_confidence
        self.extra_body = dict(orchestrator_extra_body or {})
        self.price_in = price_in
        self.price_out = price_out
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)
        self._plan = ""
        self._last_confidence = 1.0
        self._moves: list[str] = []
        self._plan_age = 0

    # -- helpers ---------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise MissingApiKey(f"{self.api_key_env} is not set")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(f"{self.base_url}{path}", json=body, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last = exc
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    break
            except httpx.HTTPError as exc:
                last = exc
            if attempt < self.max_retries:
                time.sleep(min(0.6 * 2**attempt, 5.0))
        assert last is not None
        raise last

    def _needs_plan(self, history: list[Step]) -> bool:
        return (
            not history
            or not self._plan
            or self._plan_age >= self.replan_every
            or self._last_confidence < self.low_confidence
        )

    def _plan_text(self, obs: Observation, task: str) -> tuple[str, float, int, int]:
        menu = "\n".join(f"- {k}: {v[1]}" for k, v in MENU.items())
        recent = ", ".join(self._moves[-6:]) or "none yet"
        user = (
            f"Task: {task}\n\nCurrent situation:\n{obs.text}\n{bearing_text(obs)}\n\nMoves chosen so far (latest last): {recent}\n"
            f"Previous plan: {self._plan or 'none'}\n\nWrite the guidance now."
        )
        data = self._post(
            "/v1/chat/completions",
            {
                "model": self.orchestrator,
                "messages": [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM.format(menu=menu)},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 700,
                **self.extra_body,
            },
        )
        text = (data["choices"][0]["message"].get("content") or "").strip().replace("\n", " ")
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        cost = (tin * self.price_in + tout * self.price_out) / 1_000_000
        return text[:400], cost, tin, tout

    def _jev_choice(self, obs: Observation, task: str) -> tuple[str, dict[str, float], float, float]:
        data = self._post(
            "/alpha/decisions",
            {
                "model": self.jev_model,
                "state": {
                    "task": task,
                    "situation": f"{obs.text} {bearing_text(obs)}".strip(),
                    "moves_so_far": self._moves[-6:],
                },
                "questions": {
                    "move": {
                        "type": "choice",
                        "instructions": (
                            "Pick the single best next move for the robot. Directions are from the robot's "
                            f"own point of view. Always trust the current situation and target bearing over the guidance if "
                            f"they disagree (the guidance may be a few ticks old). Orchestrator guidance: "
                            f"{self._plan or 'none, use your judgement'}"
                        ),
                        "criteria": {k: v[1] for k, v in MENU.items()},
                    }
                },
            },
        )
        ans = data["answers"]["move"]
        cost = float((data.get("usage") or {}).get("cost", 0.0))
        return ans["choice"], dict(ans.get("probabilities") or {}), float(ans.get("confidence", 0.0)), cost

    # -- Policy ----------------------------------------------------------------------------------

    def decide(self, obs: Observation, task: str, action_spec: ActionSpec, history: list[Step]) -> Action:
        started = time.perf_counter()
        if action_spec.dim != 4:
            return Action(values=[], meta={"error": f"move menu needs a 4-dim action, got {action_spec.dim}"})
        if not history:  # new episode
            self._plan, self._moves, self._plan_age, self._last_confidence = "", [], 0, 1.0
        cost = 0.0
        replanned = False
        try:
            if self._needs_plan(history):
                try:
                    self._plan, plan_cost, _, _ = self._plan_text(obs, task)
                    cost += plan_cost
                    self._plan_age = 0
                    replanned = True
                except MissingApiKey:
                    raise
                except Exception:  # a failed planner call is not fatal: Jev keeps the old plan
                    pass
            choice, probs, confidence, jev_cost = self._jev_choice(obs, task)
        except MissingApiKey:
            raise
        except Exception as exc:
            return Action(values=[], meta={"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - started})
        cost += jev_cost
        self._plan_age += 1
        self._last_confidence = confidence
        values, _ = MENU.get(choice, MENU["stop"])
        self._moves.append(choice)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        return Action(
            values=list(values),
            label=f"{choice.replace('_', ' ')} ({confidence:.0%})",
            meta={
                "latency_s": time.perf_counter() - started,
                "cost_usd": cost,
                "detail": {
                    "plan": self._plan,
                    "replanned": replanned,
                    "jev": {"choice": choice, "confidence": round(confidence, 3), "top": {k: round(v, 3) for k, v in top}},
                    "orchestrator": self.orchestrator,
                },
            },
        )
