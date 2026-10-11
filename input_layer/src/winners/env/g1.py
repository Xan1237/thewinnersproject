"""Client of the headless G1 sim HTTP API (see robot/docs/SIM_API.md).

Pure standard library (urllib), so it works anywhere the package does. One `G1SimEnv`
owns at most one server-side session at a time: `reset()` creates it, `close()` deletes it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from winners.schema import ActionSpec, Observation, ObservationSpec

DEFAULT_BASE_URL = "http://127.0.0.1:8818"


class G1SimError(RuntimeError):
    """The sim server was unreachable or answered with an error status."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class G1SimEnv:
    """`RobotEnv` backed by the sim server. `model` is the tag shown in the live UI; the
    runner overrides it per episode with the policy's name (via `reset`/`step` kwargs)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        task: str = "goto:armchair_blue",
        model: str = "python",
        max_steps: int = 120,
        timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.task = task
        self.model = model
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        self.task_description = task
        # Contract defaults until reset() replaces them with what the server reports.
        self.action_spec = ActionSpec(
            names=["forward", "right", "turn", "interact"],
            low=[-1.0, -1.0, -1.0, 0.0],
            high=[1.0, 1.0, 1.0, 1.0],
        )
        self.observation_spec = ObservationSpec(names=[])
        self._session_model = model

    # -- HTTP ---------------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise G1SimError(f"{method} {path} -> HTTP {exc.code}: {detail}", exc.code) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise G1SimError(f"{method} {path} failed: {exc}") from exc
        return json.loads(raw) if raw else {}

    # -- RobotEnv -----------------------------------------------------------------------

    @staticmethod
    def _observation(data: dict[str, Any]) -> Observation:
        return Observation(
            t=int(data.get("t", 0)),
            state=[float(v) for v in data["state"]],
            text=str(data.get("text", "")),
        )

    def reset(self, seed: int | None = None, *, model: str | None = None) -> Observation:
        self.close()
        if model is not None:
            self.model = model
        data = self._request(
            "POST",
            "/sim/sessions",
            {
                "task": self.task,
                "seed": 0 if seed is None else int(seed),
                "model": self.model,
                "max_steps": self.max_steps,
            },
        )
        self.session_id = str(data["id"])
        self._session_model = self.model
        self.task_description = str(data.get("task_description") or self.task)
        self.action_spec = ActionSpec.model_validate(data["action_spec"])
        self.observation_spec = ObservationSpec.model_validate(data["observation_spec"])
        return self._observation(data["observation"])

    def step(
        self, action: list[float], *, label: str | None = None, model: str | None = None
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.session_id is None:
            raise G1SimError("step() called before reset()")
        body: dict[str, Any] = {
            "action": [float(a) for a in action],
            "model": model or self._session_model,
        }
        if label:
            body["label"] = label
        data = self._request("POST", f"/sim/sessions/{self.session_id}/step", body)
        info = dict(data.get("info") or {})
        info["success"] = bool(info.get("success", False))
        info.setdefault("joints", {})
        return (
            self._observation(data["observation"]),
            float(data.get("reward", 0.0)),
            bool(data.get("done", False)),
            info,
        )

    def close(self) -> None:
        """Delete the server-side session (best effort; a missing session is fine)."""
        sid, self.session_id = self.session_id, None
        if sid is None:
            return
        try:
            self._request("DELETE", f"/sim/sessions/{sid}")
        except G1SimError:
            pass  # cleanup must never mask the real result of an episode (404 = already gone)

    def manifest(self) -> dict[str, Any]:
        """The room: bounds and objects (useful as LLM context)."""
        return self._request("GET", "/sim/manifest")
