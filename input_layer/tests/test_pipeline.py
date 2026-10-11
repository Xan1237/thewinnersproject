"""winners.pipeline: arena -> scripted top-up -> BC -> paced evaluation, against the fake sim."""

import pytest
from fake_sim import FakeSim


def test_pipeline_with_scripted_top_up(tmp_path):
    pytest.importorskip("torch")
    from winners.pipeline import run_pipeline

    with FakeSim() as sim:
        cfg = {
            "env": {"type": "g1-sim", "base_url": sim.url, "max_steps": 60},
            "run": {"tasks": ["goto:armchair_blue"], "episodes": 3, "max_steps": 60},
            "train": {
                "extra_policies": [{"type": "scripted-goto", "name": "farm", "noise": 0.05}],
                "extra_episodes": 20,
                "epochs": 60,
                "eval_episodes": 5,
                "eval_pace_s": 0.001,
            },
            "policies": [{"type": "scripted-goto", "name": "llm-stand-in", "noise": 0.03}],
        }
        result = run_pipeline(cfg, tmp_path)
        assert result["eval"]["goto:armchair_blue"]["success_rate"] >= 0.6
        phases = [e.get("phase") for e in sim.events if e.get("type") == "training"]
        assert phases[0] == "start" and phases[-1] == "done"
        assert phases.count("epoch") == 60 and phases.count("eval") == 1
        start = next(e for e in sim.events if e.get("phase") == "start")
        assert start["from_llm"] > 0 and start["from_scripted"] > 0
        assert (tmp_path / "bc.pt").exists()
