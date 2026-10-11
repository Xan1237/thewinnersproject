# Asset sources and motion evidence

## Robot

- Kimodo repository: <https://github.com/nv-tlabs/kimodo>, revision `58e781898b3d7e328a676a75d3e338c45dce3ad9`.
- Source folder: `kimodo/assets/skeletons/g1skel34/`: mechanical XML and original link STL meshes.
- Hand joint hierarchy: <https://github.com/unitreerobotics/unitree_ros>, revision `5994d4faef0a9cadd3287f8de0199a67eeb2a259`, `robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`.
- Kimodo code: Apache-2.0. Unitree source assets: BSD-3-Clause. License texts are retained in `licenses/` and copied beside rebuilt model assets. Preserve attribution in redistribution.
- Converted model: 46 rigid body groups, 43 hinge joints including 14 hand joints, 49 visual meshes. Height 1.323 m. Original geometry retained, about 629k triangles and 15.15 MB.
- Native model coordinate system: meters, X forward, Y left, Z up. Three.js conversion `(x,y,z) → (y,z,x)`.

## Generated clip, 2026-10-11

- Model: `nvidia/Kimodo-G1-RP-v1`; Hugging Face revision `3020ad8c419c244e0429d360163730c63c4ed011`.
- Kimodo source inside the reused inference image: `1aece8c124d73d255ceff5086d983b844c9f4e94`.
- Prompt: “a humanoid robot stands still, reaches forward with its right hand to pick up a coffee cup from a table, then gently lifts the cup towards its chest and holds it”.
- Seed 42, 240 frames, 30 fps, 100 denoising steps, post-processing enabled.
- One successful generated clip. Model loading measured 57.194 s; generation 2.856 s on L40S.
- Original NPZ SHA256: see local `public/assets/motion/receipt.json`; raw output and receipt remain alongside browser JSON.
- Verified finite arrays, 34 skeleton joints, official MuJoCo export shape `[240,36]` (root XYZ + quaternion WXYZ + 29 hinge angles), rotation orthogonality error below `4.3e-7`.
- Ephemeral Modal app stopped after generation; no deployment or persistent GPU service was created.

Downloaded meshes and generated NPZ/JSON/GLB are not committed. Scripts and these version pins make their origin inspectable; inference additionally depends on access to the existing Modal image.

## Browser and API verification, 2026-10-11

- Production build and Python/JavaScript syntax checks passed. Five server integration tests passed: validation, authentication/origin restrictions, FIFO/state acknowledgements, bounded queue/stop, and simulator replacement/disconnection.
- Real browser keyboard input moved the robot approximately 0.87 m with W held for 0.7 seconds; Shift+W moved approximately 1.36 m with the same hold. S movement was blocked at the table boundary. Timing includes acceleration and input delivery.
- Real HTTP actions in the connected browser completed a move, turn, joint pose, and full coffee sequence. A right elbow target of 1.2 radians was observed in state; 99 radians was rejected. The coffee action returned the root to `[0,0,0]`, returned the cup to `[-0.28,0.799,0.38]`, and restored manual mode.
- At 3.6 seconds, grasp-target error was about 0.19 mm; at 6 seconds, about 0.05 mm with the generated clip. The cup was raised to about 1.008 m. These are IK target errors, not proof of physical finger contacts.
- Removed the optional motion JSON temporarily for a separate browser run: the labeled procedural fallback loaded and lifted the cup, with about 0.14 mm target error. The original JSON was restored afterward. This verifies the teammate path without Modal.
- Local preview is `outputs/coffee-preview.jpg`; screenshots and rebuilt/generated assets are ignored by Git. Browser must remain running for the API simulator; a suspended browser cannot execute commands. The scene has no force/contact physics or hardware control.
