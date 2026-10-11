# datanode walkthrough

The English walkthrough is four minutes: real robot attempts → planner/Jev decisions → recorded episode → training → local policy → limits. Vel and Tim alternate narration; there are no personal introductions. The spoken text and word-timed captions are in `narration.json`; `SCRIPT.md` is the readable copy.

The film uses current main's `winners.show`, with bounded execution, worker cleanup and training provenance. A clean `datanode-slide.svg` replaces the reference photograph in the room. The viewer's **Robot** selector locks the follow camera to one controller.

The footage is pinned to main `2acfe28` plus this branch's changes. Main subsequently advanced to `f09c518` during recording, introducing a seven-part coffee line; that later environment is not depicted in this export. Use this branch to reproduce the recorded setup, and review integration with the newer task separately.

## Run an experiment

Build the G1 assets and install the arena/input-layer dependencies using their READMEs. The recording config uses isolated loopback ports, so it can run beside an existing demo:

```sh
SIM_PORT=8848 ARENA_DATA_DIR=/outside/repo/live-data node arena/server/server.js
SIM_PORT=8849 ARENA_DATA_DIR=/outside/repo/farm-data node arena/server/server.js
VIEWER_PORT=8850 SIM_URL=http://127.0.0.1:8848 npm run viewer --prefix arena
python -m winners.show --config robot/video/pitch-run.yaml --out /outside/repo/run --duration 600
```

Set `OPENROUTER_API_KEY` in the process environment for the main server and show process. `run_with_env.py ENV_FILE COMMAND [ARGS...]` can inject the two provider keys from an operator-owned file outside the repository; it never prints or copies them. The farm server needs no provider credential.

The recorded experiment used Fable 5.1, GPT-6 Astra and Opus 5.5 as planners, with Jev 1.13 choosing actions. Scripted teachers bootstrap the training pool separately. Training rounds persist episode IDs, source counts and MSE in `training_round_N.json`, alongside `bc.pt` and `policy.json`. The gold robot's successes and failures are both retained.

## Capture and edit

Capture the actual browser tab at 1920×1080 through its CDP capability. `Page.startScreencast` yields `Page.screencastFrame` events; save the JPEGs with their metadata timestamps and acknowledge every frame. Convert the timestamped sequence to 30fps H.264. The camera and data windows are operated normally through browser controls; no page state is forged. A four-minute timeline needs at least four minutes of usable capture footage.

Map chapter IDs to actual capture ranges in a JSON manifest:

```json
{"01": [{"file": "overview.mp4", "start": 0, "duration": 31}]}
```

Generate independent voice clips, using `ELEVENLABS_API_KEY` only in the environment. Voice IDs can be overridden with `--vel-voice` and `--tim-voice`:

```sh
python robot/video/synthesize.py --script robot/video/narration.json --out /outside/repo/audio
python robot/video/render.py --script robot/video/narration.json \
  --media /outside/repo/captures.json --audio /outside/repo/audio/audio-manifest.json \
  --work /outside/repo/edit --out /outside/repo/datanode-four-minute.mp4
```

The renderer requires FFmpeg/ffprobe and Pillow. It overlays captions as transparent images, so FFmpeg does not need drawtext/libass. Captures play once at their original speed; voice is not sped up, clipped or looped. The output is 240 seconds, 1920×1080, 30fps, H.264/AAC. It runs a full decode and writes a verification/provenance report next to the MP4.

Independent local Whisper transcription checks narration omissions. Supply its JSON text via `synthesize.py --transcripts FILE` for a word-error check; never give the intended script as an ASR prompt. Caption offsets can be supplied directly in `narration.json` after transcription.

## What the evidence means

The footage contains real provider-controlled attempts and actual BC training, not `demo-traffic.js`. Some data views aggregate the two recorded sessions used during preparation; individual training-round files identify the exact episodes selected for that round. Training combines frontier successes and counted scripted bootstrap episodes. The learned policy receives numeric robot/goal state, not language or camera images. Success-rate curves are recent routine results, not a held-out-layout benchmark. Robot movement, cup attachment and prerecorded clips are kinematic; no physical balance/grasp claim is made.

Media, checkpoints, captures and credentials stay outside Git. Publication of the video or a public live service is separate from producing the local MP4.
