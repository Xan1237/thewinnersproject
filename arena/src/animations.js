/**
 * Animation clips (sit, jump, wave, ...) and the plug-in point for better ones.
 *
 * A step event carrying `pose: "<name>"` plays that clip on the robot instead of the walk cycle;
 * `t` is the time in seconds since the pose started. A clip is
 *
 *   { joints(t, k) -> { <joint name>: radians },   // merged over the standing pose
 *     pelvisDrop(t, k) -> metres }                 // optional, negative lifts the body (jumps)
 *
 * The built-in clips below are hand-written procedural motions (angles checked with forward
 * kinematics of arena/public/g1/metadata.json). To use captured or generated motion, register a
 * clip under the same name; it replaces the built-in one:
 *
 *   registerClip('sit', { joints: (t) => sampleFrames(frames, t) });
 *
 * G1 joint conventions (MuJoCo frame, X forward, Z up): negative hip pitch swings the leg
 * forward, positive knee bends the shin back, negative shoulder pitch raises the arm forward,
 * negative right shoulder roll raises the right arm out to the side.
 */
import { gaitJoints } from '../server/core.js';

const clips = new Map();

export function registerClip(name, clip) {
  if (typeof clip?.joints !== 'function') throw new Error(`clip "${name}" needs a joints(t, k) function`);
  clips.set(name, clip);
}

export const getClip = name => (name ? clips.get(name) : undefined);

const STAND = gaitJoints(0, 0);
const smooth = x => { const t = Math.min(Math.max(x, 0), 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const both = (name, v) => ({ [`left_${name}`]: v, [`right_${name}`]: v });
const pose = (base, over) => ({ ...base, ...over });

// Sit down on the chair under the robot and stay seated: thighs forward, shins down, hands in lap.
registerClip('sit', {
  joints(t) {
    const s = smooth(t / 0.9);
    return pose(STAND, {
      ...both('hip_pitch_joint', -1.5 * s),
      ...both('knee_joint', 1.5 * s),
      ...both('ankle_pitch_joint', 0),
      ...both('shoulder_pitch_joint', lerp(STAND.left_shoulder_pitch_joint, -0.35, s)),
      ...both('elbow_joint', lerp(STAND.left_elbow_joint, 1.25, s)),
      waist_pitch_joint: 0.08 * s,
    });
  },
  // The pelvis moves back onto the seat (the robot stands `poseArg` metres in front of the chair centre).
  root: (t, k) => ({ pos: [-(k.poseArg || 0.5) * smooth(t / 0.9), 0], quat: [0, 0, 0, 1] }),
});

// Crouch, spring up with the arms swinging overhead, tuck in the air, land softly. 1.4 s.
const jumpPhase = t => {
  if (t < 0.35) return { crouch: smooth(t / 0.35), lift: 0, arms: -smooth(t / 0.35) * 0.3 };
  if (t < 0.5) return { crouch: 1 - smooth((t - 0.35) / 0.15), lift: 0, arms: -lerp(0.3, 2.6, smooth((t - 0.35) / 0.15)) };
  if (t < 1.0) { const u = (t - 0.5) / 0.5; return { crouch: 0.25 * Math.sin(Math.PI * u), lift: 0.5 * Math.sin(Math.PI * u), arms: -2.6 }; }
  if (t < 1.25) return { crouch: smooth((t - 1.0) / 0.25) * 0.9, lift: 0, arms: -lerp(2.6, 0.3, smooth((t - 1.0) / 0.25)) };
  return { crouch: 0.9 * (1 - smooth((t - 1.25) / 0.15)), lift: 0, arms: -0.3 * (1 - smooth((t - 1.25) / 0.15)) };
};
registerClip('jump', {
  joints(t) {
    const p = jumpPhase(t);
    return pose(STAND, {
      ...both('hip_pitch_joint', -0.8 * p.crouch),
      ...both('knee_joint', 1.5 * p.crouch),
      ...both('ankle_pitch_joint', -0.7 * p.crouch),
      ...both('shoulder_pitch_joint', p.arms),
      ...both('elbow_joint', 0.2),
    });
  },
  pelvisDrop: t => -jumpPhase(t).lift,
});

// Right arm out and up, elbow and wrist waving. The left arm rests.
registerClip('wave', {
  joints(t) {
    const up = smooth(t / 0.4), wag = Math.sin(t * Math.PI * 2 * 2.2);
    return pose(STAND, {
      right_shoulder_pitch_joint: lerp(STAND.right_shoulder_pitch_joint, -0.45, up),
      right_shoulder_roll_joint: lerp(STAND.right_shoulder_roll_joint, -1.55, up),
      right_elbow_joint: lerp(STAND.right_elbow_joint, 1.15 + 0.5 * wag, up),
      right_wrist_yaw_joint: 0.5 * wag * up,
      waist_yaw_joint: -0.06 * up,
    });
  },
});

// Both arms up, pumping, with little bounces.
registerClip('cheer', {
  joints(t) {
    const up = smooth(t / 0.35), pump = Math.sin(t * Math.PI * 2 * 2.5);
    return pose(STAND, {
      ...both('shoulder_pitch_joint', lerp(STAND.left_shoulder_pitch_joint, -2.7 + 0.25 * pump, up)),
      left_shoulder_roll_joint: 0.25 * up,
      right_shoulder_roll_joint: -0.25 * up,
      ...both('elbow_joint', lerp(STAND.left_elbow_joint, 0.25 + 0.2 * pump, up)),
      ...both('knee_joint', 0.25 * Math.abs(pump) * up),
      ...both('hip_pitch_joint', -0.12 * Math.abs(pump) * up),
      ...both('ankle_pitch_joint', -0.13 * Math.abs(pump) * up),
    });
  },
  pelvisDrop: t => -0.05 * Math.max(Math.sin(t * Math.PI * 2 * 2.5), 0) * smooth(t / 0.35),
});

// Hip sway, bouncing knees and arms swinging in opposite directions.
registerClip('dance', {
  joints(t) {
    const w = Math.sin(t * Math.PI * 2 * 1.6), w2 = Math.sin(t * Math.PI * 2 * 3.2), on = smooth(t / 0.4);
    return pose(STAND, {
      left_shoulder_pitch_joint: lerp(STAND.left_shoulder_pitch_joint, -1.3 - 0.7 * w, on),
      right_shoulder_pitch_joint: lerp(STAND.right_shoulder_pitch_joint, -1.3 + 0.7 * w, on),
      left_shoulder_roll_joint: 0.3 * on,
      right_shoulder_roll_joint: -0.3 * on,
      ...both('elbow_joint', lerp(STAND.left_elbow_joint, 1.1, on)),
      waist_yaw_joint: 0.35 * w * on,
      ...both('knee_joint', (0.3 + 0.2 * w2) * on),
      ...both('hip_pitch_joint', -(0.15 + 0.1 * w2) * on),
      ...both('ankle_pitch_joint', -(0.15 + 0.1 * w2) * on),
    });
  },
});

/**
 * Clips generated with NVIDIA Kimodo (G1 text-to-motion), packed by tools/import-clips.py into
 * public/animations/. They replace the built-in clips of the same name once loaded; if the files
 * are missing the built-ins keep working.
 */
function frameClip(data, { rootScale } = {}) {
  const { fps, joints: names, frames, root, lift } = data, last = frames.length - 1;
  const at = t => { const f = Math.min(Math.max(t * fps, 0), last), i = Math.floor(f); return [i, Math.min(i + 1, last), f - i]; };
  const xy = root[last];
  const endDist = Math.hypot(xy[0], xy[1]);
  return {
    duration: data.duration,
    joints(t) {
      const [i, j, u] = at(t), out = { ...STAND };
      names.forEach((n, c) => { out[n] = lerp(frames[i][c], frames[j][c], u); });
      return out;
    },
    pelvisDrop(t) { const [i, j, u] = at(t); return -lerp(lift[i], lift[j], u); },
    root(t, k) {
      const [i, j, u] = at(t), a = root[i], b = root[j];
      const s = rootScale && endDist > 0.05 ? rootScale(k) / endDist : 1;
      const q = [3, 4, 5, 6].map(c => lerp(a[c], b[c], u)), n = Math.hypot(...q) || 1;
      return { pos: [lerp(a[0], b[0], u) * s, lerp(a[1], b[1], u) * s], quat: q.map(v => v / n) };
    },
  };
}

export async function loadGeneratedClips(base = '/animations') {
  try {
    const index = await (await fetch(`${base}/index.json`)).json();
    for (const [name, file] of Object.entries(index)) {
      const data = await (await fetch(`${base}/${file}`)).json();
      registerClip(name, frameClip(data, name === 'sit' ? { rootScale: k => k.poseArg || 0.5 } : {}));
    }
    return Object.keys(index);
  } catch (e) {
    console.warn('generated animation clips not loaded, using the built-in ones:', e.message);
    return [];
  }
}
