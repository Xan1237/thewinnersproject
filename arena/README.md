# Winners Arena

Several LLMs drive simulated Unitree G1 robots inside the hackathon room. A side panel shows every decision as labeled data, a leaderboard, and the training of a small policy on the successful runs.

```
room/ (Godot)  -> room.glb + manifest.json -> arena/public/room/
arena/server   headless sim (Node, no browser): sessions, collisions, gait joints, SSE events
arena/src      three.js viewer: room, G1 robots, right panel (Live data / Arena / Training)
input_layer/   Python arena + policies + recorder + BC training; talks to the sim over HTTP
```

API contract: `../robot/docs/SIM_API.md`.

## Run

```
cd ../robot && npm ci && npm run assets      # builds the G1 model once
cd ../arena && npm install && npm run assets  # copies and decimates it (~160k triangles)
npm run dev                                   # sim on :8818, viewer on http://127.0.0.1:8820
node tools/demo-traffic.js --training         # fake models + fake training curve, to see the UI
npm test                                      # sim core + HTTP API tests
```

## Real models + training loop

```
export OPENROUTER_API_KEY=...                        # never commit it
(cd arena && SIM_PORT=8819 node server/server.js &)  # "data farm" for cheap scripted top-up runs
cd input_layer && uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e '.[dev,train]'
python -m winners.pipeline --config configs/g1_arena.yaml --out runs/g1-live
```

The pipeline runs Qwen3.8-27B (Cerebras), GPT-6 Luna and Claude Haiku 5.5 through OpenRouter on the same tasks (watch them in the viewer), tops the few LLM runs up with scripted runs, trains a small policy by behaviour cloning (live loss curve), then plays the learned policy back in the room. Offline version without keys: `configs/g1_offline.yaml`.

`?nogl` on the viewer URL skips 3D rendering (panel-only checks without a GPU).

## Rebuilding the room assets

```
cd room
godot --headless --path . -s tools/build_room.gd     # scenes/room.tscn + export/room_manifest.json
godot --headless --path . -s tools/export_web.gd     # export/room.glb (lights stripped)
cd export
npx @gltf-transform/cli@4 resize room.glb a.glb --width 1024 --height 1024
npx @gltf-transform/cli@4 webp a.glb b.glb && npx @gltf-transform/cli@4 meshopt b.glb room.opt.glb
cp room.opt.glb ../../arena/public/room/room.glb && cp room_manifest.json ../../arena/public/room/manifest.json
```

Do not use `gltf-transform optimize`: it joins meshes and drops the node names the viewer relies on (`Ceiling`, `WallSouth`).

## Notes

- The sim is kinematic (position, heading, procedural gait), not a physics engine. Treat the data as "level zero".
- The robot app in `../robot` is untouched; the arena reuses its G1 model and gait.

## Animations

Hand-made clips (sitting on a chair, ...) plug in through `src/animations.js`: `registerClip(name, {joints, pelvisDrop})`, then send `pose: "<name>"` on a step event. Without a registered clip the robot keeps standing, so it never vanishes while an animation is missing.
