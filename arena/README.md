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
