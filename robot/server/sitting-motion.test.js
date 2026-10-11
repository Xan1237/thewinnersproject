import test from 'node:test';
import assert from 'node:assert/strict';
import { ARMCHAIR_SEAT, SIT_DURATION, sampleSit, g1LegForwardKinematics } from '../src/sitting-motion.js';

const near = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

test('actual G1 neutral foot geometry matches the source standing height', () => {
  const neutral = g1LegForwardKinematics(0, 0);
  // Independent golden value from model bounds: pelvis at .793 has soles
  // .0007271025 above ground, rather than assuming pelvis is .8m high.
  near(.793 + neutral[2] - .035409145057201385, .0007271025206238707, 2e-7);
  near(neutral[1], .118506455);
  near(sampleSit(0).pelvisHeight, .7922728974793761, 2e-7);
});

test('full sit plants both feet, lowers continuously, and respects G1 joint limits', () => {
  const start = sampleSit(0);
  let previousHeight = start.pelvisHeight;
  for (let frame = 0; frame <= 340; frame++) {
    const pose = sampleSit(frame / 100);
    assert.ok(pose.pelvisHeight <= previousHeight + 1e-10);
    previousHeight = pose.pelvisHeight;
    for (const side of ['left', 'right']) {
      pose.feet[side].forEach((value, axis) => near(value, start.feet[side][axis]));
      const limits = { hip_pitch: [-2.5307, 2.8798], knee: [-.087267, 2.8798], ankle_pitch: [-.87267, .5236], shoulder_pitch: [-3.0892, 2.6704], elbow: [-1.0472, 2.0944] };
      for (const [joint, [min, max]] of Object.entries(limits)) {
        const value = pose.joints[`${side}_${joint}_joint`];
        assert.ok(value >= min && value <= max, `${side} ${joint} outside model limit`);
      }
      near(pose.joints[`${side}_hip_pitch_joint`] + pose.joints[`${side}_knee_joint`] + pose.joints[`${side}_ankle_pitch_joint`], 0);
    }
  }
});

test('final pose is supported by the actual armchair cushion, with planted feet', () => {
  const end = sampleSit(SIT_DURATION);
  near(end.pelvisHeight - ARMCHAIR_SEAT.pelvisClearance, .452);
  near(end.position[2], -.02);
  assert.ok(end.joints.left_hip_pitch_joint < -1.1 && end.joints.left_hip_pitch_joint > -1.2);
  assert.ok(end.joints.left_knee_joint > 1 && end.joints.left_knee_joint < 1.15);
  assert.equal(end.seated, true);
  assert.equal(end.phase, 'seated');
  near(end.heading, Math.PI);
  assert.deepEqual(sampleSit(100), end);
  assert.deepEqual(sampleSit(-1), sampleSit(0));
});

test('same motion follows a rotated chair on the room stage without losing foot support', () => {
  const { x: chairX, y: groundY, z: chairZ, yaw: chairYaw } = ARMCHAIR_SEAT.stagePlacement;
  const profile = { chairX, groundY, chairZ, chairYaw };
  const start = sampleSit(0, profile);
  for (let frame = 0; frame <= 102; frame++) {
    const pose = sampleSit(frame / 30, profile);
    for (const side of ['left', 'right']) pose.feet[side].forEach((value, axis) => near(value, start.feet[side][axis]));
    near(pose.feet.left[1], .35);
  }
  const end = sampleSit(SIT_DURATION, profile);
  near(end.position[0], chairX + Math.sin(chairYaw) * -.02);
  near(end.position[2], chairZ + Math.cos(chairYaw) * -.02);
  near(end.position[1] + end.pelvisHeight - ARMCHAIR_SEAT.pelvisClearance, .802);
  near(end.heading, chairYaw + Math.PI);
});

test('rejects invalid time or chair coordinates before producing corrupt poses', () => {
  for (const time of [NaN, Infinity, undefined]) assert.throws(() => sampleSit(time), TypeError);
  assert.throws(() => sampleSit(1, { groundY: Infinity }), TypeError);
});
