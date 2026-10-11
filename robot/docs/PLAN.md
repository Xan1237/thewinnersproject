# G1 coffee pickup implementation plan

> **For agentic workers:** implement this plan task-by-task, checking off each step as it lands. Use bounded independent asset and motion tasks, then review their output in the running browser.

**Goal:** Show the actual Unitree G1 taking a coffee mug from a small table in a locally viewable 3D demo.

**Architecture:** A standalone Three.js scene uses the official G1 mechanical hierarchy, a genuine Kimodo-G1 clip where available, and a local arm IK layer for precise mug contact. Cup attachment is controlled by the demo; motion provenance is stated explicitly. Existing Handshake VR remains a read-only reference.

**Tech stack:** Three.js, Vite, Python asset conversion, ephemeral Modal Kimodo inference.

## Tasks

- [x] Convert pinned official G1 XML/STL files into an articulated GLB with joint metadata. Verify link hierarchy, assembled scale, and attribution.
- [x] Generate one Kimodo-G1 coffee reach/lift clip and save source prompt, model, seed, and original output. Avoid persistent GPU deployment.
- [x] Build a focused coffee-table scene with replay, pause, timeline scrub, orbit camera, and a close view of the grasp.
- [x] Layer arm IK and cup attachment onto the clip; verify contact at pickup, cup following the hand, and release back onto the table.
- [x] Add camera-relative WASD movement, Shift acceleration, table collision and E interaction.
- [x] Add bounded local HTTP commands, browser state relay and a Python adapter for LLM tool handlers; teammates require no Modal access.
- [x] Run production build, inspect the full motion in the built-in browser, save preview evidence, document observed limits, and commit source only.

## Visual direction

One large 3D stage, cool blue-gray studio floor, warm espresso counter, white ceramic mug, and quiet navy UI. Palette: floor #b6c6ce, deep ink #172f3b, steel #738a96, porcelain #f1f1eb, coffee #462c1b. Display uses a restrained Georgia heading, body system sans, and monospace only for playback time. The signature is the robot physically wrapping its hand around the cup; controls stay subordinate to the scene.
