# datanode four-minute walkthrough implementation plan

> **For agentic workers:** implement this plan task-by-task, checking off each step. Independent runtime and slide work are delegated; recording, narration and final review remain with the parent agent.

**Goal:** A 240-second English walkthrough with Vel/Tim narration, real recorded interface activity, and evidence of the decision → data → local policy path.

**Architecture:** Use current main in an isolated checkout. Capture the browser through CDP with frame timestamps; encode at 1920×1080/30fps. Generate short independent ElevenLabs voice lines and edit to an eight-part timeline. Each claim must match the captured run or be clearly described as a limitation/next step.

**Tech stack:** Node sim/Vite/Three.js, Python/PyTorch/OpenRouter, ElevenLabs, CDP, FFmpeg/Pillow.

## Tasks

- [x] Inspect current main and identify the actual show script and reference-photo slide.
- [x] Replace the slide with fresh datanode vector branding, preserving the room geometry.
- [x] Audit and fix bounded show execution, cleanup, training provenance, and honest model roles.
- [x] Launch an isolated sim, farm and viewer; collect real model calls, successful episodes and a real checkpoint.
- [x] Write short spoken narration without introductions, invented statistics or broad marketing claims.
- [x] Capture room activity, planner/Jev detail, an actual episode, training and local policy playback.
- [ ] Generate Vel/Tim lines, verify wording, edit to 240 seconds and add readable captions.
- [ ] Inspect beginning/middle/end plus transitions; check audio, codecs, duration and full decode.
- [ ] Commit source/docs only; deliver the local MP4 and source script. Public upload is a separate action.

## Evidence rules

No fake training curve or synthetic model labels. Scripted bootstrap and real model episodes are reported separately. The trained policy uses numeric robot/goal state; it does not acquire language/vision understanding. This is kinematic simulation, including coffee attachment and prerecorded motion clips. If a run fails, preserve and describe it; do not convert it into a success in the edit.
