/**
 * Extension point for hand-made animations (sitting, waving, ...).
 *
 * A clip is registered by name. A step event that carries `pose: "<name>"` plays that clip on the
 * robot instead of the procedural walk cycle; with no clip registered for the name the robot just
 * stays standing, so nothing disappears while an animation is missing.
 *
 *   registerClip('sit', {
 *     // joint angles (radians, keyed by G1 joint name) for the moment `t` (seconds), key frame `k`
 *     joints: (t, k) => ({ left_knee_joint: 1.6, right_knee_joint: 1.6, ... }),
 *     // optional: lower the pelvis (metres) so the seated robot rests on the chair
 *     pelvisDrop: (t, k) => 0.35,
 *   });
 *
 * Nothing in the sim emits a pose yet; the clip only has to be registered once it exists.
 */
const clips = new Map();

export function registerClip(name, clip) {
  if (typeof clip?.joints !== 'function') throw new Error(`clip "${name}" needs a joints(t, k) function`);
  clips.set(name, clip);
}

export const getClip = name => (name ? clips.get(name) : undefined);
