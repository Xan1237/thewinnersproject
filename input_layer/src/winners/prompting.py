"""Renders observations into prompts and parses model replies into action vectors."""

from __future__ import annotations

import json
import re

from winners.schema import ActionSpec, Observation, Step

SYSTEM_PROMPT = (
    "You are the decision-making controller of a robot. Each turn you receive the robot's "
    "state and must reply with the next action. Reply with ONLY a JSON object of the form "
    '{"action": [<numbers>], "label": "<2-6 word description of your intent>"} and nothing else. '
    'The "label" is optional but encouraged (e.g. "walk toward the armchair", "turn left"). '
    "Respect the stated action limits."
)

MAX_LABEL_CHARS = 120


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
            said = f" label={s.label!r}" if s.label else ""
            lines.append(
                f"  t={s.t} action={[round(v, 4) for v in s.safe_action]}{said} reward={s.reward:.4f}"
            )
    lines.append('Reply with only: {"action": [...], "label": "<short intent>"}')
    return "\n".join(lines)


_DECODER = json.JSONDecoder()


def _json_objects(text: str) -> list[object]:
    """Every JSON value that can be decoded starting at a `{` (handles nesting, fences, prose)."""
    found: list[object] = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        found.append(obj)
    return found


def _numbers(values: object, spec: ActionSpec) -> list[float] | None:
    if isinstance(values, dict):  # {"forward": 1, "right": 0, ...} keyed by action names
        if set(values) >= set(spec.names):
            values = [values[n] for n in spec.names]
        else:
            return None
    if (
        isinstance(values, list)
        and len(values) == spec.dim
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    ):
        return [float(v) for v in values]
    return None


def _clean_label(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    label = " ".join(raw.split())
    return label[:MAX_LABEL_CHARS] or None


def parse_reply(text: str, spec: ActionSpec) -> tuple[list[float], str | None]:
    """Extract `{"action": [...], "label": "..."}` -> (values, label). Tolerant: accepts
    prose or code fences around the JSON, a bare list, or an action dict keyed by name.
    The label is optional (None when missing or not a string). Raises ActionParseError
    if no valid action is found."""
    candidates: list[object] = []
    try:
        candidates.append(json.loads(text.strip()))
    except json.JSONDecodeError:
        pass
    candidates += _json_objects(text)
    for data in candidates:
        if isinstance(data, dict):
            values = _numbers(data.get("action"), spec)
            if values is not None:
                label = data.get("label", data.get("intent"))
                return values, _clean_label(label)
        else:
            values = _numbers(data, spec)
            if values is not None:
                return values, None
    raise ActionParseError(f"no valid {spec.dim}-number action found in reply: {text[:200]!r}")


def parse_action(text: str, spec: ActionSpec) -> list[float]:
    """Extract `{"action": [...]}` from a reply. Raises ActionParseError if unusable."""
    return parse_reply(text, spec)[0]
