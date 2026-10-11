"""Live mode against the real Node sim server: one order reaches every agent."""

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[2] / "arena" / "server" / "server.js"
pytestmark = pytest.mark.skipif(not (shutil.which("node") and SERVER.exists()), reason="needs node + arena/")


def _call(base, path, body=None):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _wait(cond, timeout=40):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_voice_order_reaches_every_agent(tmp_path):
    from winners.live import run_live

    port = 18890
    proc = subprocess.Popen(
        ["node", str(SERVER)], env={**os.environ, "SIM_PORT": str(port), "ARENA_DATA_DIR": str(tmp_path / "data")}, stdout=subprocess.DEVNULL
    )
    base = f"http://127.0.0.1:{port}"
    stop = threading.Event()
    try:
        assert _wait(lambda: _alive(base), 10)
        cfg = {
            "env": {"type": "g1-sim", "base_url": base, "max_steps": 80},
            "run": {"max_steps": 80},
            "policies": [{"type": "scripted-goto", "name": f"bot{i}", "noise": 0.0} for i in range(3)],
        }
        t = threading.Thread(target=run_live, args=(cfg, tmp_path, stop), daemon=True)
        t.start()
        assert _wait(lambda: len(_call(base, "/sim/sessions")["sessions"]) == 3)  # robots spawned, idle

        _call(base, "/sim/command", {"task": "goto:lectern"})
        def seated():
            ss = _call(base, "/sim/sessions")["sessions"]
            return len(ss) == 3 and all(s["done"] and s["success"] for s in ss)
        assert _wait(seated), "all robots reached the lectern"
        assert _wait(lambda: (tmp_path / "episodes.jsonl").exists() and len((tmp_path / "episodes.jsonl").read_text().splitlines()) >= 3)
        tasks = {s["observation"]["text"] for s in _call(base, "/sim/sessions")["sessions"]}
        assert len(tasks) == 3

        _call(base, "/sim/command", {"task": "jump"})  # interrupts / follows: everyone jumps
        assert _wait(lambda: all("jump" in json.dumps(s["task"]) for s in _call(base, "/sim/sessions")["sessions"]))
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=5)


def _alive(base):
    try:
        _call(base, "/sim/manifest")
        return True
    except Exception:
        return False
