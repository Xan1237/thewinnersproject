"""Best-effort training-progress events for the live UI (`POST /sim/training`).

Never raises: an unreachable or misbehaving URL must not affect training or evaluation.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any


class TrainingEventSink:
    """POSTs JSON events to `url`. With `url=None` it is a no-op. After `max_failures`
    consecutive failures it disables itself so a dead server cannot slow training down."""

    def __init__(self, url: str | None, timeout_s: float = 1.0, max_failures: int = 3):
        self.url = url
        self.timeout_s = timeout_s
        self.max_failures = max_failures
        self.failures = 0

    @property
    def active(self) -> bool:
        return bool(self.url) and self.failures < self.max_failures

    def send(self, **event: Any) -> bool:
        """Send `{"type": "training", **event}`. Returns True when the server accepted it."""
        if not self.active:
            return False
        try:
            req = urllib.request.Request(
                self.url,  # type: ignore[arg-type]
                data=json.dumps({"type": "training", **event}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp.read()
        except Exception:  # noqa: BLE001 - best effort by design
            self.failures += 1
            return False
        self.failures = 0
        return True
