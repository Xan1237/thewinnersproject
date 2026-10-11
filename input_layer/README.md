# winners

Multi-model robot decision layer. Several models (Claude, Gemini, GPT, local/open models) each try to complete a task. Successful rollouts are recorded and used to train a small local policy by behavior cloning, so the same task runs with no API cost.

```
Env ──► Arena ──► [Claude | Gemini | GPT | Local] ──► Safety filter ──► Env.step()
Recorder (state, raw action, safe action, reward, latency, cost) ──► successful episodes only
  ──► dataset ──► BC trainer ──► LearnedPolicy (same Policy interface, benchmarked in the arena)
```

## Setup
```
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,train]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m winners.arena --config configs/default.yaml --out runs/run1
```
API keys are read from environment variables only: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## Wiring in your robot or scene
Implement `winners.env.base.RobotEnv` — this is the only seam:

- `task_description`, `action_spec` (`ActionSpec`: names, low/high limits), `observation_spec`
- `reset(seed) -> Observation` where `Observation.state` is the physics vector that gets recorded and trained on
- `step(action) -> (Observation, reward, done, info)`, with `info["success"]` set when the outcome is known

Then register it in `winners/registry.py` (`ENVS`) and select it in the config (`env: {type: ...}`).
`run.decision_interval` controls how many control ticks one model decision is held for, since LLMs are slower than control rate; every tick is still recorded.

## Training the free policy
```python
from winners.recorder import load_episodes
from winners.dataset import select_episodes, build_dataset
from winners.train.bc import train_bc, save_checkpoint

eps = select_episodes(load_episodes("runs/run1/episodes.jsonl"))   # successes only
ckpt, metrics = train_bc(build_dataset(eps), env.action_spec)
save_checkpoint(ckpt, "checkpoints/bc.pt")
```
Use `{type: learned, checkpoint: checkpoints/bc.pt}` in the config to benchmark it next to its teachers.

## G1 sim arena
`env: {type: g1-sim, base_url: ..., task: "goto:armchair_blue" | "fetch_coffee"}` talks to the headless sim server (`robot/docs/SIM_API.md`, pure-stdlib client in `winners/env/g1.py`). See `configs/g1_arena.yaml`.

- Models reply `{"action": [...], "label": "<short intent>"}`; the label (and the env's `info["joints"]`) is recorded on every `Step` and forwarded to the sim with the policy name as the model tag.
- `RobotEnv.reset(seed, *, model=None)` and `step(action, *, label=None, model=None)` take the new keywords; the runner only passes them to envs that accept them.
- OpenAI-compatible policies (`openai`, `local`, `openrouter`) accept `extra_body` (merged into the request JSON, e.g. OpenRouter provider pinning), `extra_headers` and `api_key_env`.
- `train_bc(..., training_events_url=".../sim/training")` and `winners.train.evaluate.evaluate_policy(..., training_events_url=...)` POST per-epoch / eval events (best effort, never fail training).
- Tests run against a stdlib fake of the server: `tests/fake_sim.py`.

## Caveats
- Check each provider's terms on using model outputs to train another model.
- `load_checkpoint` uses `torch.load(weights_only=False)`; only load checkpoints you trained yourself.
