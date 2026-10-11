"""A stdlib fake of the headless sim server (robot/docs/SIM_API.md) for tests.

Simple 2D kinematics: position (x, z) and heading. At heading 0 the robot faces +Z and
its right is -X; positive `turn` increases the heading (turns left seen from above).
Implements the manifest, sessions (create / step / get / delete) and the training
endpoint. Instead of an SSE stream it records the events it would broadcast in
`FakeSim.events` so tests can assert on them.
"""

from __future__ import annotations

import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

BOUNDS = {"minX": -5.0, "maxX": 5.0, "minZ": -5.0, "maxZ": 5.0}
OBJECTS = [
    {"id": "armchair_blue", "name": "Blue armchair", "kind": "furniture", "x": 2.0, "z": 3.0,
     "sx": 0.9, "sz": 0.9, "height": 0.9, "solid": True},
    {"id": "coffee_station", "name": "Coffee station", "kind": "station", "x": -2.5, "z": 1.5,
     "sx": 0.6, "sz": 0.4, "height": 0.9, "solid": True},
]
ACTION_SPEC = {
    "names": ["forward", "right", "turn", "interact"],
    "low": [-1.0, -1.0, -1.0, 0.0],
    "high": [1.0, 1.0, 1.0, 1.0],
    "units": "fraction of 1.5 m/s, 1.5 m/s, 1.5 rad/s; interact > 0.5 grabs",
}
OBS_SPEC = {
    "names": ["x", "z", "heading", "goal_forward", "goal_right", "goal_dist", "holding",
              "steps_left_frac"],
    "description": "Robot pose, goal in the robot frame, holding flag, remaining time fraction.",
}
DT = 0.5
SUBSTEP = 0.02
MAX_SPEED = 1.5
MAX_TURN = 1.5


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


class Session:
    def __init__(self, sid: str, task: str, seed: int, model: str, max_steps: int):
        rng = random.Random(seed)
        self.id, self.task, self.seed, self.model, self.max_steps = sid, task, seed, model, max_steps
        self.x = rng.uniform(-4.0, 4.0)
        self.z = rng.uniform(-4.5, -1.0)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.t = 0
        self.holding = 0.0
        self.done = False
        self.success = False
        self.reason = "running"
        target_id = "coffee_station" if task == "fetch_coffee" else task.split(":", 1)[1]
        target = next(o for o in OBJECTS if o["id"] == target_id)
        self.goal = (target["x"], target["z"])
        self.goal_radius = 0.9 if task == "fetch_coffee" else 1.2
        self.prev_dist = self.goal_dist()

    def goal_dist(self) -> float:
        return math.hypot(self.goal[0] - self.x, self.goal[1] - self.z)

    def goal_robot_frame(self) -> tuple[float, float]:
        dx, dz = self.goal[0] - self.x, self.goal[1] - self.z
        s, c = math.sin(self.heading), math.cos(self.heading)
        return dx * s + dz * c, -dx * c + dz * s  # (forward, right)

    def observation(self) -> dict[str, Any]:
        fwd, right = self.goal_robot_frame()
        dist = self.goal_dist()
        state = [self.x, self.z, self.heading, fwd, right, dist, self.holding,
                 1.0 - self.t / self.max_steps]
        text = f"Goal is {fwd:.2f} m ahead and {right:.2f} m to the right ({dist:.2f} m away)."
        return {"t": self.t, "state": state, "text": text}

    def joints(self) -> dict[str, float]:
        phase = self.t * 0.8
        return {"hip_l": 0.4 * math.sin(phase), "hip_r": -0.4 * math.sin(phase),
                "knee_l": 0.3 * abs(math.sin(phase)), "shoulder_r": 0.2 * math.cos(phase)}

    def info(self, collided: bool = False) -> dict[str, Any]:
        return {"success": self.success, "reason": self.reason, "position": [self.x, self.z],
                "heading": self.heading, "joints": self.joints(), "collided": collided}

    def step(self, action: list[float]) -> tuple[float, bool, bool]:
        fwd, right, turn, interact = [
            _clip(a, lo, hi) for a, lo, hi in zip(action, ACTION_SPEC["low"], ACTION_SPEC["high"])
        ]
        collided = False
        for _ in range(int(round(DT / SUBSTEP))):
            self.heading += turn * MAX_TURN * SUBSTEP
            s, c = math.sin(self.heading), math.cos(self.heading)
            vf, vr = fwd * MAX_SPEED, right * MAX_SPEED
            nx = self.x + (vf * s - vr * c) * SUBSTEP  # right vector = (-cos h, sin h)
            nz = self.z + (vf * c + vr * s) * SUBSTEP
            cx, cz = _clip(nx, BOUNDS["minX"], BOUNDS["maxX"]), _clip(nz, BOUNDS["minZ"], BOUNDS["maxZ"])
            collided = collided or (cx, cz) != (nx, nz)
            self.x, self.z = cx, cz
        self.t += 1
        dist = self.goal_dist()
        reward = -0.01 + 0.1 * (self.prev_dist - dist)
        self.prev_dist = dist
        if self.task == "fetch_coffee":
            if interact > 0.5 and dist <= self.goal_radius:
                self.holding, self.success = 1.0, True
        elif dist <= self.goal_radius:
            self.success = True
        if self.success:
            reward += 1.0
            self.reason = "success"
        elif self.t >= self.max_steps:
            self.reason = "max_steps"
        self.done = self.success or self.t >= self.max_steps
        return reward, self.done, collided


class FakeSim:
    """Run with `with FakeSim() as sim:`; `sim.url` is the base URL (random free port)."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.events: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str, Any]] = []  # (method, path, body)
        self.lock = threading.Lock()
        self._next = 0
        sim = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # silence
                pass

            def _send(self, code: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> Any:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n)) if n else None

            def _route(self, method: str) -> None:
                body = self._body() if method == "POST" else None
                with sim.lock:
                    sim.requests.append((method, self.path, body))
                    code, payload = sim.handle(method, self.path, body)
                self._send(code, payload)

            def do_GET(self) -> None:
                self._route("GET")

            def do_POST(self) -> None:
                self._route("POST")

            def do_DELETE(self) -> None:
                self._route("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeSim":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- contract -----------------------------------------------------------------------

    def handle(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        parts = [p for p in path.split("?")[0].split("/") if p]
        if len(parts) < 2 or parts[0] != "sim":
            return 404, {"error": "not found"}
        if method == "GET" and parts[1:] == ["manifest"]:
            return 200, {"bounds": BOUNDS, "objects": OBJECTS}
        if method == "POST" and parts[1:] == ["training"]:
            self.events.append(body)
            return 200, {"ok": True}
        if parts[1] != "sessions":
            return 404, {"error": "not found"}
        if method == "POST" and len(parts) == 2:
            return self._create(body or {})
        if len(parts) < 3 or parts[2] not in self.sessions:
            return 404, {"error": "unknown session"}
        sess = self.sessions[parts[2]]
        if method == "DELETE" and len(parts) == 3:
            del self.sessions[sess.id]
            self.events.append({"type": "session_ended", "session": sess.id})
            return 200, {"ok": True}
        if method == "GET" and len(parts) == 3:
            return 200, {"observation": sess.observation(), "info": sess.info(),
                         "done": sess.done}
        if method == "POST" and parts[3:] == ["step"]:
            return self._step(sess, body or {})
        return 404, {"error": "not found"}

    def _create(self, body: dict[str, Any]) -> tuple[int, Any]:
        task = body.get("task", "")
        valid = task == "fetch_coffee" or (
            task.startswith("goto:") and any(o["id"] == task[5:] for o in OBJECTS)
        )
        if not valid:
            return 400, {"error": f"unknown task {task!r}"}
        self._next += 1
        sess = Session(f"s{self._next}", task, int(body.get("seed", 0)),
                       str(body.get("model", "")), int(body.get("max_steps", 120)))
        self.sessions[sess.id] = sess
        self.events.append({"type": "session_started", "session": sess.id, "model": sess.model,
                            "task": task})
        desc = (f"Walk to within 1.2 m of {task[5:]}." if task.startswith("goto:")
                else "Fetch the coffee from the station: get within 0.9 m and interact.")
        return 200, {"id": sess.id, "task_description": desc, "action_spec": ACTION_SPEC,
                     "observation_spec": OBS_SPEC, "observation": sess.observation()}

    def _step(self, sess: Session, body: dict[str, Any]) -> tuple[int, Any]:
        action = body.get("action")
        if not (isinstance(action, list) and len(action) == 4
                and all(isinstance(a, (int, float)) for a in action)):
            return 400, {"error": "action must be 4 numbers"}
        if sess.done:
            return 409, {"error": "session is done"}
        reward, done, collided = sess.step([float(a) for a in action])
        info = sess.info(collided)
        self.events.append({
            "session": sess.id, "model": body.get("model", sess.model), "task": sess.task,
            "t": sess.t, "action": action, "label": body.get("label"),
            "position": info["position"], "heading": sess.heading, "joints": info["joints"],
            "reward": reward, "done": done, "success": sess.success,
        })
        return 200, {"observation": sess.observation(), "reward": reward, "done": done,
                     "info": info}

    def step_events(self, model: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events if "action" in e and (model is None or e["model"] == model)]
