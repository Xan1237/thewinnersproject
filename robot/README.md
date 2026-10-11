# G1 coffee demo

A real Unitree G1 model takes a mug from a table, raises it, and returns it. Switch to **Управлять · WASD** to move the robot yourself, then press **E** near the table to take the coffee.

**Сесть в кресло** plays a 3.4-second animation fitted to `room/assets/models/armchair_blue.glb`. The feet stay planted while the pelvis lowers to the cushion; the seated pose holds afterward. **Встать** or **R** reverses the motion and returns manual control. The same animation is available in the room arena's **Sit in armchair** preview and `sit:armchair_blue` task.

## Run locally

Requirements: Node.js 22+, `uv`, and a browser with WebGL. Teammates do not need a Modal account.

```sh
cd robot
npm ci
npm run assets
npm run dev
```

Open <http://127.0.0.1:8816/>. On the development Mac the assets and generated motion are already present; `npm ci` and `npm run dev` are enough. Without the generated motion file, the demo still runs its procedural contact sequence and labels that fallback explicitly.

For LLM control, expand **API для LLM** in the viewer and click **Подключить API**. The local HTTP API runs at <http://127.0.0.1:8817/api/schema>. See [API examples and the Python tool adapter](docs/API.md). Each teammate runs their own instance; nothing in that workflow accesses Modal.

`npm run motion` uses the existing pinned Kimodo Modal image `im-xmXj0bi1K12m9XRdNzSury`, which belongs to the current experimental setup. A different Modal workspace must build/provide its own equivalent image; this is not a publicly deployable inference service. The generator downloads G1 weights before GPU allocation, generates exactly one eight-second clip, and stops the ephemeral app. It refuses to overwrite an existing clip. Modal GPU inference costs money; the G1 model assets are free.

## Controls

- **Демо:** automatic twelve-second pickup cycle; pause, replay, scrub, toggle looping.
- **Управлять · WASD:** camera-relative movement; **Shift** moves faster.
- **E** near the table: align to the coffee station, play the pickup, then return control.
- Drag the scene to orbit the camera; scroll to zoom. The camera follows the robot during manual movement.
- **Общий план / Рука и чашка:** preset camera views.

The table blocks locomotion. This is a local visual prototype, with no physical robot connection.

## What generates the motion

Kimodo-G1-RP-v1 generated the robot body clip (240 frames at 30 fps, seed 42). The demo applies the official MuJoCo joint-angle export to the articulated G1 hierarchy. A seven-axis arm IK layer corrects the reach for the actual cup position; finger closure, cup attachment, lifting/return choreography and WASD locomotion are procedural. Kimodo does not generate the cup interaction itself, and this scene is not a physics grasp simulation.

The G1 geometry is assembled from native rigid links; the hands use Unitree's official three-finger joint hierarchy. This preserves mechanical parts rather than deforming a human avatar mesh.

## Files

| File | Purpose |
| --- | --- |
| `src/main.js` | Scene, coffee interaction, arm IK, motion playback and UI |
| `src/controller.js` | Keyboard input, camera-relative locomotion, table collision and gait |
| `src/sitting-motion.js` | Shared geometry-fitted sitting clip and calibrated chair placement |
| `src/api-client.js` | Executes bounded API actions in the running browser |
| `server/api.js` | Local HTTP command queue and state relay |
| `tools/robot_api.py` | Standard-library Python adapter for LLM tools |
| `tools/build_g1.py` | Download pinned source assets and assemble articulated GLB |
| `tools/generate_motion.py` | One bounded ephemeral Modal inference run, source NPZ and browser JSON |
| `docs/ASSETS.md` | Source versions, licenses and generation evidence |
| `public/assets/` | Local regenerated model and motion; excluded from Git |

No API keys or credentials are embedded in the frontend or generator. Modal authentication is provided by the operator's existing external CLI configuration.
