# Local robot API for an LLM

Each teammate runs their own simulator. No Modal account, token, GPU endpoint or deployment is needed. Download the free model once with `npm run assets`, then run `npm run dev` and open `http://127.0.0.1:8816/`. Expand **API для LLM** and click **Подключить API** in the browser. Keep that tab running: it owns the simulation and reports its state to the local server.

The API listens on `127.0.0.1:8817`; Vite also proxies the same routes under port 8816. It accepts one simulator, queues bounded commands, and tracks each command until completion or rejection. WASD input or a mode button can cancel an active API command. `POST /api/stop` clears queued commands and stops the active action.

## HTTP contract

| Method / path | Result |
| --- | --- |
| `GET /api/schema` | Command formats and bounds |
| `GET /api/state` | Connection status and latest browser state |
| `POST /api/commands` | Validate/queue a command; return its ID |
| `GET /api/commands/{id}` | Inspect queued/running/completed/rejected status |
| `POST /api/stop` | Stop and clear the queue |

Command examples:

```json
{"type":"move","forward":1,"right":0,"duration":0.4,"sprint":false}
{"type":"turn","radians":1.57,"duration":1}
{"type":"coffee"}
{"type":"sit"}
{"type":"stand"}
{"type":"joints","angles":{"right_elbow_joint":1.2},"duration":2}
{"type":"stop"}
```

World positions use meters and Y up. At heading zero, the robot faces world +Z and its right is -X. Positive heading turns it toward +X. Movement is relative to the robot heading at the start of the command; WASD remains camera-relative. `move` normalizes diagonal direction and uses 1.5 m/s (2.5 with sprint), with acceleration and collision against the table. An action completing means its bounded interval finished; read back `position` to see how far it actually moved if a collision blocked it.

Angles and headings are radians. `state.joints` and `state.limits` expose the available joint names and limits. Invalid joints or angles are rejected; a joint pose is held for the command duration and then returns to the locomotion pose. Arbitrary joint poses have no physics collision solver. `coffee` requires the root to be within 0.75 m of `[0,0,0]`, aligns it to the station, plays the approximately thirteen-second pickup/return, then returns control.

`sit` uses the repository's `armchair_blue.glb`. Move within 0.8 m of `state.chair.standingPosition` first; the command aligns to the contact pose and plays a 3.4-second descent, then holds the seated pose. `state.seated` becomes true only after the full clip. `stand` reverses the clip and restores WASD movement. Locomotion and coffee are rejected while seated. A stop during the descent freezes the current pose (`seatingPaused`, not `seated`); `stand` can recover from that pose. Seating is a deterministic local animation, independent of Modal, with feet held at fixed positions by the G1 geometry.

## Call it from a terminal or LLM tool handler

```sh
python3 tools/robot_api.py schema
python3 tools/robot_api.py state
python3 tools/robot_api.py command '{"type":"move","forward":-1,"right":0,"duration":0.4,"sprint":false}' --wait
python3 tools/robot_api.py command '{"type":"coffee"}' --wait
python3 tools/robot_api.py stop
```

`tools/robot_api.py` exposes the `RobotApi` class with `state`, `schema`, `command`, `wait` and `stop`. An LLM agent can use these as tool handlers. First fetch the schema and state, send one action, wait for the action's result, then read the resulting state before deciding on another action. Keep HTTP access and these bounded tools separate from shell execution.

## Credentials and Kimodo

The server binds only to loopback, rejects foreign browser origins and never calls Modal. It does not execute arbitrary code or accept paths or shell commands. Optional `ROBOT_API_TOKEN` comes from the operator's environment. If set, HTTP clients send `Authorization: Bearer <token>` and the browser's API panel accepts the same token in a password field. The token is not stored in Git or browser local storage. A loopback service without a token is accessible to other processes on the same computer; set a token when that distinction matters.

The development Mac has a generated Kimodo body clip cached locally. Other machines run the explicitly labeled procedural coffee sequence unless the owner shares that motion file separately. This does not affect HTTP controls. `npm run motion` is an optional owner-only inference step, with an existing private Modal configuration, and should not be part of teammate setup.

If text-to-motion generation is added later, put it behind a separate owner-run backend: it should accept a bounded prompt/job request and return animation data, while retaining Modal credentials internally. Teammates should receive a scoped credential for that backend, never a Modal workspace token. This repository currently contains the local simulation API only.
