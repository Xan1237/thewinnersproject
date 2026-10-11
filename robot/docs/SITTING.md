# G1 sitting on the existing armchair

Source chair: `room/assets/models/armchair_blue.glb`, built by `room/tools/blender/seating_armchair_blue.py`. Meters, Y up, front local −Z. The seat center is at `(0, .45, -.02)`, with approximately .48 m clear width between the arms. The room stage places this chair at `(1.5, .35, -3.7)`, yaw 155 degrees; the shared clip uses that exact profile in the arena. The standalone robot viewer places the same model on the floor beside the coffee table.

`src/sitting-motion.js` exports `sampleSit(time, profile)` and `SIT_DURATION=3.4`. It produces joint angles, root translation, pelvis height, heading, phase, and support-point coordinates. The phases are balance, lowering, settling and seated. Position and height are derived from the pinned G1 leg forward kinematics rather than interpolated separately, keeping both feet planted. The final pelvis height is .604005 m above the chair's floor, accounting for the .152005 m pelvis underside and a 2 mm cushion gap. Because the chair is high relative to G1, the knees bend about 61 degrees rather than 90 degrees.

This is a procedural animation fitted to geometry, with no force/contact simulation or Kimodo inference. The reverse clip is used for standing in the standalone viewer. Other furniture requires a new fit; changing arbitrary seat heights without recalculating the leg solve is unsupported.

Entry points:

- Robot viewer: **Сесть в кресло**, **Встать**, pause/replay/timeline; HTTP `sit` and `stand` actions.
- Arena viewer: **Sit in armchair**; headless task `sit:armchair_blue`, front approach → interact → wait until `info.seated`.
- Arena's coarse obstacle box remains in use for navigation. The explicit sitting action aligns from a free approach into the calibrated standing pose over .4 s before playing the clip. This alignment is kinematic.

Verification: source geometry audited, five motion tests cover continuous lowering, stable support points, native joint limits, cushion clearance and rotated/elevated room placement. Browser measurements of actual foot mesh bounds at 1.7 and 3.4 seconds put both soles at floor level within floating-point precision. Arena tests retain existing coffee/goto behavior and require full seating completion. Live previews and screenshots are local ignored outputs under `robot/outputs/`.
