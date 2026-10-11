/** Kinematic G1 sit, fitted to room/assets/models/armchair_blue.glb.
 * Meters, Three.js Y-up. Chair front is local -Z; robot front is +Z.
 * This is a geometric animation, not a Kimodo clip or a physics policy.
 */
export const SIT_DURATION = 3.4;
export const ARMCHAIR_SEAT = Object.freeze({
  seatHeight: .45,
  seatCenterZ: -.02,
  pelvisClearance: .15200547873973846,
  contactGap: .002,
  openingWidth: .48,
  seatFrontZ: -.266635,
  seatRearZ: .23,
  armrestHeight: .61,
  footAnchorZ: -.34,
  stagePlacement: Object.freeze({ x: 1.5, y: .35, z: -3.7, yaw: 155 * Math.PI / 180 }),
});
export const ARMCHAIR_ROOM_PLACEMENT = Object.freeze({
  chairX: ARMCHAIR_SEAT.stagePlacement.x,
  groundY: ARMCHAIR_SEAT.stagePlacement.y,
  chairZ: ARMCHAIR_SEAT.stagePlacement.z,
  chairYaw: ARMCHAIR_SEAT.stagePlacement.yaw,
});

// Exact translations/rest pitch from the pinned G1 metadata. Hip roll/yaw
// are zero; opposite +/-10-degree rest rotations cancel at the knee.
const REST_HIP_PITCH = -.17489998429394246;
const SOLE_DEPTH = .035409145057201385;
const rotateY = ([x, y, z], angle) => [Math.cos(angle) * x + Math.sin(angle) * z, y, -Math.sin(angle) * x + Math.cos(angle) * z];
const add = (a, b) => a.map((value, i) => value + b[i]);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const smooth = value => { const t = clamp(value, 0, 1); return t * t * t * (t * (t * 6 - 15) + 10); };

/** Native X-forward/Y-left/Z-up ankle-roll origin, relative to pelvis.
 * ankle pitch counter-rotates hip+knee to keep the complete sole flat.
 */
export function g1LegForwardKinematics(hip, knee, side = 'left') {
  const sign = side === 'right' ? -1 : 1;
  let ankle = [0, sign * .064452, -.1027];
  ankle = add(ankle, rotateY([0, sign * .052, -.030465], hip));
  ankle = add(ankle, rotateY([.025001, 0, -.12412], hip + REST_HIP_PITCH));
  ankle = add(ankle, rotateY([-.078273, sign * .0021489, -.17734], hip + REST_HIP_PITCH));
  ankle = add(ankle, rotateY([0, sign * -.000094445, -.30001], hip + knee));
  return add(ankle, [0, 0, -.017558]);
}

// Solve the final seated support condition once. The pelvis underside is
// just above the cushion, with ankle forward .32m and sole on the floor.
function solveSeatedLeg() {
  const height = ARMCHAIR_SEAT.seatHeight + ARMCHAIR_SEAT.pelvisClearance + ARMCHAIR_SEAT.contactGap;
  const target = [.32, SOLE_DEPTH - height];
  let hip = -1.12, knee = 1.06;
  for (let i = 0; i < 12; i++) {
    const p = g1LegForwardKinematics(hip, knee);
    const dh = g1LegForwardKinematics(hip + 1e-6, knee), dk = g1LegForwardKinematics(hip, knee + 1e-6);
    const a = (dh[0] - p[0]) / 1e-6, b = (dk[0] - p[0]) / 1e-6;
    const c = (dh[2] - p[2]) / 1e-6, d = (dk[2] - p[2]) / 1e-6;
    const ex = target[0] - p[0], ez = target[1] - p[2], determinant = a * d - b * c;
    hip += (d * ex - b * ez) / determinant;
    knee += (-c * ex + a * ez) / determinant;
  }
  return { hip, knee };
}
const SEATED_LEG = solveSeatedLeg();

const chairToWorld = (x, z, { chairX, chairZ, chairYaw }) => [
  chairX + Math.cos(chairYaw) * x + Math.sin(chairYaw) * z,
  chairZ - Math.sin(chairYaw) * x + Math.cos(chairYaw) * z,
];

/** Standing -> balance -> lower -> seated. Holds the seated pose afterward.
 * pelvisHeight is relative to groundY (assign native pelvis.position.z).
 * position is the Y-up worldRoot position. Feet metrics are world-space
 * sole reference points, useful for verification and simulation state.
 */
export function sampleSit(time, { chairX = 0, chairZ = 0, chairYaw = 0, groundY = 0 } = {}) {
  if (![time, chairX, chairZ, chairYaw, groundY].every(Number.isFinite)) throw new TypeError('Sit time and chair profile must be finite numbers');
  const t = clamp(time, 0, SIT_DURATION);
  const progress = smooth((t - .45) / 2.5);
  const hip = SEATED_LEG.hip * progress, knee = SEATED_LEG.knee * progress;
  const ankle = g1LegForwardKinematics(hip, knee);
  // Translation is derived from FK, never independently interpolated:
  // every frame cancels the moving leg's ankle displacement exactly.
  const pelvisHeight = SOLE_DEPTH - ankle[2];
  const localZ = ARMCHAIR_SEAT.footAnchorZ + ankle[0];
  const profile = { chairX, chairZ, chairYaw };
  const [x, z] = chairToWorld(0, localZ, profile);
  const balance = smooth(t / .45) * (1 - smooth((t - 2.4) / 1));
  const joints = { waist_yaw_joint: 0, waist_roll_joint: 0, waist_pitch_joint: .14 * balance };
  for (const side of ['left', 'right']) {
    Object.assign(joints, {
      [`${side}_hip_pitch_joint`]: hip,
      [`${side}_hip_roll_joint`]: 0,
      [`${side}_hip_yaw_joint`]: 0,
      [`${side}_knee_joint`]: knee,
      [`${side}_ankle_pitch_joint`]: -hip - knee,
      [`${side}_ankle_roll_joint`]: 0,
      [`${side}_shoulder_pitch_joint`]: -.30 * progress - .35 * balance,
      [`${side}_shoulder_roll_joint`]: (side === 'left' ? .12 : -.12) * progress,
      [`${side}_shoulder_yaw_joint`]: 0,
      [`${side}_elbow_joint`]: .45 * progress + .25 * balance,
      [`${side}_wrist_roll_joint`]: 0,
      [`${side}_wrist_pitch_joint`]: 0,
      [`${side}_wrist_yaw_joint`]: 0,
    });
  }
  const feet = Object.fromEntries(['left', 'right'].map(side => {
    const foot = g1LegForwardKinematics(hip, knee, side);
    const [footX, footZ] = chairToWorld(-foot[1], localZ - foot[0], profile);
    return [side, [footX, groundY + pelvisHeight + foot[2] - SOLE_DEPTH, footZ]];
  }));
  return {
    joints, pelvisHeight, position: [x, groundY, z], heading: chairYaw + Math.PI,
    progress, seated: t >= SIT_DURATION,
    phase: t < .45 ? 'balance' : t < 2.95 ? 'lower' : t < SIT_DURATION ? 'settle' : 'seated',
    feet,
  };
}
