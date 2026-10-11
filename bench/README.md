# Winnersbench

Local, live version of the Summerbench layout (plot, leaderboard, docs) for the G1 arena results. No dependencies beyond Node 20+.

```sh
cd bench
npm start            # http://127.0.0.1:8830
npm test
```

## Data

Reads every run under `input_layer/runs` (`episodes_all.jsonl`, else `episodes.jsonl`; `tmp*` dirs are skipped).

- Scripted baselines load at once and sit in the free column.
- Model episodes are **replayed** one at a time (interleaved so all models advance together); the loop restarts after a pause. The status pill shows `Replay`.
- When new episodes appear in a run file while the server runs, the board settles to the full recorded history and streams only the new ones. The pill switches to `Live`.
- If the arena sim server is up (`ARENA_URL`, default `http://127.0.0.1:8818`), walking robots pulse on the plot and `POST /sim/training` events draw the free-policy training curve.

Score per episode: 0 on failure, else `60 + 40 * (1 - steps / 60)`; a model's score is the mean. Cost is the recorded provider spend per episode (log axis). Model episodes with no recorded spend (the first, pre-pricing run) and `teacher-*` episodes are left out.

## Environment

| Variable | Default |
| --- | --- |
| `BENCH_PORT` / `BENCH_HOST` | `8830` / `127.0.0.1` |
| `BENCH_RUNS` | `../input_layer/runs` |
| `BENCH_PACE_MS` | `900` (delay between replayed episodes) |
| `BENCH_LOOP_PAUSE_MS` | `12000` |
| `BENCH_LOOP` | `1` (`0` stops after one pass) |
| `BENCH_SKIP_MODELS` | `^teacher` |
| `ARENA_URL` | `http://127.0.0.1:8818` |
