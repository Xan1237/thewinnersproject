# Multi-model robot decision layer → imitation-learned free policy

## Context
Goal: a multi-model "input layer" where several LLMs/VLMs (Claude, Gemini, GPT, local/open models) each try to control a robot to finish a task. We record the robot's physics data (states + the actions that worked), keep only the successful runs, and use them to train a cheap, local policy with traditional ML (behavior cloning). End result: the same task done with no API cost.

Scope (per user): only the **decision-making layer** now. The scene and robot get wired in later, so everything sits behind a small, robot-agnostic environment interface. A mock environment lets the whole pipeline be built and tested with no robot. Strategy: **compete, keep the best**: every model attempts the task, and only successful rollouts become training data.

`thewinnersproject/` is an empty git repo (branch `input`, no commits), so this is greenfield. Stack: Python 3.11+, pydantic, PyTorch, httpx/official SDKs, pytest.

## Architecture
```
Task + Env ──► Arena ──► [Claude | Gemini | GPT | Local] ──► Safety filter ──► Env.step()
                 │                                                              │
                 └────────── Recorder (obs, raw action, safe action, reward) ◄──┘
                                   │ successful episodes only
                                   ▼
                      Dataset builder ──► BC trainer ──► LearnedPolicy (same interface)
                                                              │
                                              Arena compares it against the LLMs
```
Key principle: **every decision-maker, LLM or learned, implements one `Policy` interface**, so the trained model drops into the same runner and is benchmarked against its teachers.

## Layout (`thewinnersproject/`)
- `pyproject.toml`, `configs/default.yaml` (models, task, episode count, limits)
- `src/winners/schema.py`: pydantic `ObservationSpec`, `ActionSpec` (name, dim, low/high, units), `Observation`, `Action`, `Step`, `Episode`
- `src/winners/env/base.py`: `RobotEnv` protocol: `reset() -> Observation`, `step(Action) -> (Observation, reward, done, info)`, `action_spec`, `observation_spec`, `task_description`. **This is the only seam to the future robot/scene.**
- `src/winners/env/mock.py`: toy 2D reach environment with simple dynamics and a success criterion. For tests/dev only.
- `src/winners/policies/base.py`: `Policy` protocol: `decide(obs, task, action_spec, history) -> Action`
- `src/winners/policies/{claude,gemini,openai,local}.py`: provider adapters. `local.py` talks to any OpenAI-compatible endpoint (Ollama/vLLM/llama.cpp), which also covers the "Other" models. `registry.py` builds them from config.
- `src/winners/policies/scripted.py`: deterministic fake policy for tests (no network).
- `src/winners/prompting.py`: renders the observation + `ActionSpec` into a prompt, and defines the strict JSON action schema (use each provider's structured-output/tool-call mode where available, plain-JSON parse otherwise).
- `src/winners/safety.py`: validate and clamp actions to `ActionSpec` limits, rate-limit, retry on malformed output, and fall back to a hold/zero action. Log both the raw and the safe action.
- `src/winners/runner.py`: runs one episode for one policy and returns an `Episode`.
- `src/winners/arena.py`: runs N episodes × each model (concurrently, with per-provider rate limits), then scores them (success rate, steps, latency, cost) and writes a leaderboard.
- `src/winners/recorder.py`: writes episodes to disk (parquet or jsonl): per-step observation, physics state, raw action, safe action, reward, plus model id, latency, token cost, and success flag.
- `src/winners/dataset.py`: loads successful episodes into `(state, action)` tensors, with a train/val split by episode and normalization stats saved with the dataset.
- `src/winners/train/bc.py`: behavior-cloning MLP (MSE on actions, optional small history window), checkpointing, metrics.
- `src/winners/policies/learned.py`: `LearnedPolicy` that loads a checkpoint and implements `Policy`.
- `tests/`: unit tests per module plus one end-to-end test on the mock env.

## Deferred: Jev (decision model)
Jev (TypeSafe AI) was removed for now. It is a decision model, not a text generator, so the intended design was an arbiter policy that picks among the other models' candidate actions. To add it back, implement a `Policy` that gathers candidates from the proposers and asks Jev to choose; nothing else in the pipeline needs to change. Verify TypeSafe's official API docs first.

## Design decisions to honor
1. **LLM latency vs. control rate.** LLMs act at about 1 Hz at best, and physics runs far faster. Models output a *high-level command* (target pose or short waypoint list) held or interpolated by the env adapter for several physics ticks. The recorder keeps the full-rate physics data between decisions, and the BC target is the interpolated low-level action, so the learned policy runs at control rate.
2. **Best-of-N filtering.** Keep only episodes where `info["success"]` is true, ranked by steps and smoothness. Failed episodes are logged for analysis but excluded from training.
3. **Model diversity matters.** Tag every step with the source model id so we can train on a single teacher or a mix and compare.
4. **Provider terms.** Check each provider's terms on using model outputs to train another model. Numeric control actions are likely fine, but verify before shipping.
5. **API keys** come from environment variables only, never from config files or commits (add `.gitignore` for `.env`, data, and checkpoints).
6. **Later improvement (out of scope now):** DAgger-style loops, where the learned policy runs and an LLM corrects its failures.

## Build order
1. Scaffold: `pyproject.toml`, `.gitignore`, package skeleton, `schema.py`.
2. `RobotEnv` protocol + `MockReachEnv`, with tests.
3. `Policy` protocol, `ScriptedPolicy`, `safety.py`, `runner.py`, `recorder.py`, with tests.
4. Provider adapters (Claude, Gemini, OpenAI, local) + `prompting.py`, with tests that mock HTTP (no real keys needed).
5. `arena.py` + leaderboard CLI (`python -m winners.arena --config configs/default.yaml`).
6. `dataset.py`, `train/bc.py`, `LearnedPolicy`, and the final comparison run.
7. Docs: short README on how to plug in a real robot by implementing `RobotEnv`.

## Verification
- `pytest` passes, including the end-to-end test: scripted policies on the mock env → arena → recorded episodes → dataset → BC training → `LearnedPolicy` reaches a success rate close to its teacher.
- Live smoke test (needs API keys): run the arena with one real model per provider on the mock env, 3 episodes each. Check the leaderboard prints and the episode files contain both raw and safe actions.
- Sanity check: the learned policy's success rate on the mock env is reported next to each LLM's in the same leaderboard, with its per-step cost at about zero.
- Confirm a malformed or out-of-range model output is clamped or falls back without crashing the episode.
