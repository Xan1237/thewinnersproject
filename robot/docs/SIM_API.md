# Headless sim API (v1 contract)

Many independent simulated robots run in one Node process, with no browser needed. The browser app subscribes to the event stream and renders. The Python `G1Env` (in `input_layer`) is a client of this API.

Base URL: `http://127.0.0.1:8818` (override with `SIM_PORT`). Loopback only.

All numbers are SI units, radians, world Y up. At heading 0 the robot faces world +Z and its right is -X. World coordinates are identical to the room scene (`room/`) coordinates: +X east, -Z north (screen wall), Y up.

## Endpoints

### `GET /sim/manifest`
The room: `{"bounds": {"minX","maxX","minZ","maxZ"}, "objects": [{"id","name","kind","x","z","sx","sz","height","solid": bool}]}`. `x`,`z` is the centre, `sx`,`sz` the footprint size. LLM context and collision data come from here.

### `POST /sim/sessions`
Body: `{"task": "goto:armchair_blue" | "fetch_coffee", "seed": 0, "model": "claude-sonnet", "max_steps": 120}`.
Response: `{"id", "task_description", "action_spec": {"names","low","high","units"}, "observation_spec": {"names","description"}, "observation": {"t","state","text"}}`.

### `POST /sim/sessions/{id}/step`
Body: `{"action": [forward, right, turn, interact], "label": "walk toward the armchair", "model": "claude-sonnet"}`.
Action semantics (all clipped to the spec): `forward`/`right` in [-1,1] are speed fractions of 1.5 m/s relative to the heading; `turn` in [-1,1] is a fraction of 1.5 rad/s (positive = turn left, counter-clockwise seen from above; the heading increases); `interact` in [0,1], `> 0.5` means "grab / interact" (only valid next to the station). One step advances the sim by `dt = 0.5` s (internally sub-stepped at 20 ms).
Response: `{"observation": {"t","state","text"}, "reward": float, "done": bool, "info": {"success": bool, "reason": "...", "position": [x,z], "heading": float, "joints": {name: angle}, "collided": bool}}`.

`info.joints` holds the gait/arm joint angles at the end of the step. This is the "nested motor data" shown beside the label.

### `GET /sim/sessions/{id}` and `DELETE /sim/sessions/{id}`
State snapshot (same shape as a step response without the action) and cleanup.

### `GET /sim/events` (Server-Sent Events)
One JSON event per step for every session: `{"session","model","task","t","action","label","position","heading","joints","reward","done","success","ts"}`. Also `session_started` and `session_ended` events. The UI right panel is driven by this stream.

### Training events
`POST /sim/training` (body: any JSON) is rebroadcast on the same SSE stream as `{"type":"training", ...}`. Python posts `{"type":"training","phase":"epoch","epoch":3,"loss":0.12,"val_loss":0.14}` and `{"type":"training","phase":"eval","success_rate":0.7,"n":20}`. Both are optional and only used for the visualisation.

## Observation state vector
`["x","z","heading","goal_forward","goal_right","goal_dist","holding","steps_left_frac"]`
`goal_*` is the target position in the robot frame (forward along heading, right along its right-hand side). `text` is a short natural-language summary including nearby objects and what blocks the way.

## Tasks
- `goto:<object_id>`: success when the robot is within 1.2 m of the object's centre.
- `fetch_coffee`: success when `interact > 0.5` is sent within 1.0 m of the coffee station on the table, so that `holding == 1`.

Reward: `+1` on success, `-0.01` per step, plus a small shaping term of `0.1 * (previous_goal_dist - goal_dist)`.
