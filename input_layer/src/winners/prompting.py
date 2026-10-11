"""Renders observations into prompts and parses model replies into action vectors."""

from __future__ import annotations

import json
import re

from winners.schema import ActionSpec, Observation, Step

SYSTEM_PROMPT = (
    "You are the decision-making controller of a robot. Each turn you receive the robot's "
    "state and must reply with the next action. Reply with ONLY a JSON object of the form "
    '{"action": [<numbers>]} and nothing else. Respect the stated action limits.'
)


class ActionParseError(ValueError):
    pass


def build_prompt(
    obs: Observation,
    task: str,
    spec: ActionSpec,
    history: list[Step],
    obs_names: list[str] | None = None,
    max_history: int = 5,
) -> str:
    state = (
        ", ".join(f"{n}={v:.4f}" for n, v in zip(obs_names, obs.state))
        if obs_names and len(obs_names) == len(obs.state)
        else str([round(v, 4) for v in obs.state])
    )
    limits = "; ".join(
        f"{n} in [{lo}, {hi}]" for n, lo, hi in zip(spec.names, spec.low, spec.high)
    )
    lines = [
        f"Task: {task}",
        f"Step: {obs.t}",
        f"State: {state}",
    ]
    if obs.text:
        lines.append(f"Notes: {obs.text}")
    lines.append(f"Action vector ({spec.dim} numbers, order: {', '.join(spec.names)}): {limits}")
    if spec.units:
        lines.append(f"Units: {spec.units}")
    recent = [s for s in history if s.decided][-max_history:]
    if recent:
        lines.append("Recent actions taken (oldest first):")
        for s in recent:
            lines.append(f"  t={s.t} action={[round(v, 4) for v in s.safe_action]} reward={s.reward:.4f}")
    lines.append('Reply with only: {"action": [...]}')
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_action(text: str, spec: ActionSpec) -> list[float]:
    """Extract `{"action": [...]}` from a reply. Raises ActionParseError if unusable."""
    candidates = [text.strip()] + _JSON_RE.findall(text)
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        values = data.get("action") if isinstance(data, dict) else data
        if (
            isinstance(values, list)
            and len(values) == spec.dim
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        ):
            return [float(v) for v in values]
    raise ActionParseError(f"no valid {spec.dim}-number action found in reply: {text[:200]!r}")
