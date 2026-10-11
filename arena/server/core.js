// Headless, deterministic simulation core for the G1 room arena. No browser or three.js needed.
// Contract: robot/docs/SIM_API.md. World: metres, Y up, heading 0 faces +Z, right = -X.

export const DT = 0.5;            // sim seconds per decision step
export const SUBSTEP = 0.02;      // integration step
export const ROBOT_RADIUS = 0.18;
export const WALK_SPEED = 1.5;
export const TURN_RATE = 1.5;     // rad/s at turn = 1
export const GOTO_RADIUS = 1.2;
export const COFFEE_RADIUS = 0.9;

export const ACTION_SPEC = {
  names: ['forward', 'right', 'turn', 'interact'],
  low: [-1, -1, -1, 0],
  high: [1, 1, 1, 1],
  units: 'forward/right: fraction of 1.5 m/s relative to heading; turn: fraction of 1.5 rad/s (positive = turn LEFT / counter-clockwise seen from above, negative = turn RIGHT); interact: >0.5 grabs the coffee cup when within 0.9 m',
};
export const OBSERVATION_SPEC = {
  names: ['x', 'z', 'heading', 'goal_forward', 'goal_right', 'goal_dist', 'holding', 'steps_left_frac'],
  description: 'x,z in metres (world), heading in radians (0 faces +Z). goal_forward/goal_right: target position in the robot frame (metres; goal_right < 0 means the target is on your left, so turn left = positive turn). holding: 1 once the coffee cup is in hand.',
};

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const round = (v, n = 4) => Math.round(v * 10 ** n) / 10 ** n;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Joint angles of the procedural gait, ported from robot/src/controller.js (updatePose). */
export function gaitJoints(phase, blend, stride = 1) {
  const j = {};
  for (const [side, offset, sign] of [['left', 0, 1], ['right', Math.PI, -1]]) {
    const p = phase + offset;
    const swing = Math.sin(p), lift = Math.max(swing, 0);
    const hip = -0.42 * swing * blend * stride;
    const knee = (0.12 + 0.67 * lift) * blend;
    j[`${side}_hip_pitch_joint`] = hip;
    j[`${side}_hip_roll_joint`] = sign * (0.045 + 0.015 * Math.cos(p) * blend);
    j[`${side}_hip_yaw_joint`] = 0.025 * swing * blend;
    j[`${side}_knee_joint`] = knee;
    j[`${side}_ankle_pitch_joint`] = -hip - knee + 0.08 * blend;
    j[`${side}_ankle_roll_joint`] = -sign * 0.045;
    j[`${side}_shoulder_pitch_joint`] = 0.1 + 0.33 * swing * blend;
    j[`${side}_shoulder_roll_joint`] = sign * 0.09;
    j[`${side}_shoulder_yaw_joint`] = 0;
    j[`${side}_elbow_joint`] = 0.95 + 0.15 * lift * blend;
    j[`${side}_wrist_pitch_joint`] = -0.12;
    j[`${side}_hand_thumb_0_joint`] = 0.2;
    j[`${side}_hand_thumb_1_joint`] = sign * 0.25;
    j[`${side}_hand_thumb_2_joint`] = sign * 0.3;
    for (const finger of ['index', 'middle']) {
      j[`${side}_hand_${finger}_0_joint`] = -sign * 0.25;
      j[`${side}_hand_${finger}_1_joint`] = -sign * 0.3;
    }
  }
  j.waist_yaw_joint = 0.025 * Math.sin(phase) * blend;
  j.waist_pitch_joint = 0.035 * blend;
  return j;
}

/** Right arm raised with the cup held in front of the chest (override on top of the gait). */
export const HOLD_POSE = {
  right_shoulder_pitch_joint: -0.9,
  right_shoulder_roll_joint: -0.15,
  right_elbow_joint: 1.35,
  right_wrist_pitch_joint: 0,
};

export class World {
  constructor(manifest) {
    this.manifest = manifest;
    this.bounds = manifest.bounds;
    this.objects = manifest.objects;
    this.byId = new Map(this.objects.map(o => [o.id, o]));
    this.solids = this.objects.filter(o => o.solid);
    this.platforms = this.objects.filter(o => o.kind === 'platform');
    this.cup = this.byId.get('coffee_cup') ?? null;
  }

  blocked(x, z) {
    const r = ROBOT_RADIUS, b = this.bounds;
    if (x < b.minX + r + 0.1 || x > b.maxX - r - 0.1 || z < b.minZ + r + 0.1 || z > b.maxZ - r - 0.1) return true;
    for (const o of this.solids) {
      if (x > o.x - o.sx / 2 - r && x < o.x + o.sx / 2 + r && z > o.z - o.sz / 2 - r && z < o.z + o.sz / 2 + r) return true;
    }
    return false;
  }

  groundHeight(x, z) {
    let h = 0;
    for (const p of this.platforms) {
      if (Math.abs(x - p.x) <= p.sx / 2 && Math.abs(z - p.z) <= p.sz / 2) h = Math.max(h, p.height);
    }
    return h;
  }

  /** Free start pose for a seed that is far enough from the goal. */
  startPose(rand, goal) {
    for (let i = 0; i < 400; i++) {
      const x = this.bounds.minX + 0.6 + rand() * (this.bounds.maxX - this.bounds.minX - 1.2);
      const z = this.bounds.minZ + 0.6 + rand() * (this.bounds.maxZ - this.bounds.minZ - 1.2);
      if (this.blocked(x, z)) continue;
      if (Math.hypot(goal.x - x, goal.z - z) < 2.5) continue;
      return { x, z, heading: (rand() * 2 - 1) * Math.PI };
    }
    throw new Error('no free start pose');
  }
}

export function resolveTask(world, task) {
  if (task === 'fetch_coffee') {
    const cup = world.cup;
    if (!cup) throw new Error('manifest has no coffee_cup');
    return { kind: 'fetch_coffee', target: cup, radius: COFFEE_RADIUS, description: 'Walk to the table on the right side of the room, pick up the coffee cup (set interact > 0.5 within 0.9 m of the cup) and hold it.' };
  }
  const m = /^goto:(.+)$/.exec(task);
  if (m) {
    const target = world.byId.get(m[1]);
    if (!target) throw new Error(`unknown object id: ${m[1]}`);
    return { kind: 'goto', target, radius: GOTO_RADIUS, description: `Walk to the ${target.name} and stop within ${GOTO_RADIUS} m of it.` };
  }
  throw new Error(`unknown task: ${task}`);
}

export class Session {
  constructor(world, { id, task = 'goto:armchair_blue', seed = 0, model = 'unknown', maxSteps = 120 } = {}) {
    this.world = world;
    this.id = id;
    this.seed = seed;
    this.model = model;
    this.maxSteps = maxSteps;
    this.task = task;
    this.spec = resolveTask(world, task);
    const rand = mulberry32(seed * 7919 + 13);
    const start = world.startPose(rand, this.spec.target);
    this.x = start.x;
    this.z = start.z;
    this.heading = start.heading;
    this.vx = 0;
    this.vz = 0;
    this.phase = 0;
    this.blend = 0;
    this.holding = false;
    this.t = 0;
    this.done = false;
    this.success = false;
    this.reason = '';
    this.collided = false;
    this.lastDist = this.goalDist();
  }

  goalDist() { return Math.hypot(this.spec.target.x - this.x, this.spec.target.z - this.z); }

  /** Target position expressed in the robot frame. */
  goalLocal(target = this.spec.target) {
    const dx = target.x - this.x, dz = target.z - this.z;
    const fx = Math.sin(this.heading), fz = Math.cos(this.heading);
    const rx = -Math.cos(this.heading), rz = Math.sin(this.heading);
    return { forward: dx * fx + dz * fz, right: dx * rx + dz * rz };
  }

  pose() {
    const joints = gaitJoints(this.phase, this.blend);
    if (this.holding) Object.assign(joints, HOLD_POSE);
    return Object.fromEntries(Object.entries(joints).map(([k, v]) => [k, round(v)]));
  }

  observation() {
    const g = this.goalLocal();
    const state = [this.x, this.z, this.heading, g.forward, g.right, this.goalDist(), this.holding ? 1 : 0, 1 - this.t / this.maxSteps].map(v => round(v));
    return { t: this.t, state, text: this.describe(g) };
  }

  describe(g = this.goalLocal()) {
    const lines = [];
    lines.push(`You are at x=${this.x.toFixed(2)}, z=${this.z.toFixed(2)}, facing ${(this.heading * 180 / Math.PI).toFixed(0)} degrees.`);
    lines.push(`Target "${this.spec.target.name}": ${Math.abs(g.forward).toFixed(2)} m ${g.forward >= 0 ? 'ahead' : 'behind you'}, ${Math.abs(g.right).toFixed(2)} m to your ${g.right >= 0 ? 'right' : 'left'}, distance ${this.goalDist().toFixed(2)} m.`);
    if (this.collided) lines.push('Your last move was blocked by an obstacle or wall.');
    if (this.holding) lines.push('You are holding the coffee cup.');
    const near = this.world.objects
      .filter(o => o.solid && o !== this.spec.target)
      .map(o => ({ o, d: Math.hypot(o.x - this.x, o.z - this.z) - Math.max(o.sx, o.sz) / 2 }))
      .filter(e => e.d < 2.5).sort((a, b) => a.d - b.d).slice(0, 5)
      .map(({ o, d }) => { const l = this.goalLocal(o); return `${o.name} (${Math.abs(l.forward).toFixed(1)} m ${l.forward >= 0 ? 'ahead' : 'behind'}, ${Math.abs(l.right).toFixed(1)} m ${l.right >= 0 ? 'right' : 'left'}, ${Math.max(d, 0).toFixed(1)} m away)`; });
    if (near.length) lines.push(`Nearby obstacles: ${near.join('; ')}.`);
    lines.push(`Room: x from ${this.world.bounds.minX} to ${this.world.bounds.maxX}, z from ${this.world.bounds.minZ} to ${this.world.bounds.maxZ}.`);
    return lines.join(' ');
  }

  /** Advance one decision step. Returns the response described in SIM_API.md plus the sampled path. */
  step(action) {
    if (this.done) throw new Error('session already finished');
    const a = ACTION_SPEC.names.map((_, i) => {
      const v = Number(action?.[i]);
      return Number.isFinite(v) ? clamp(v, ACTION_SPEC.low[i], ACTION_SPEC.high[i]) : 0;
    });
    const [fwd, rgt, turn, interact] = a;
    const path = [];
    this.collided = false;
    let moved = 0;
    const n = Math.round(DT / SUBSTEP);
    for (let i = 0; i < n; i++) {
      this.heading += turn * TURN_RATE * SUBSTEP;
      if (this.heading > Math.PI) this.heading -= 2 * Math.PI;
      if (this.heading < -Math.PI) this.heading += 2 * Math.PI;
      const fx = Math.sin(this.heading), fz = Math.cos(this.heading);
      const rx = -Math.cos(this.heading), rz = Math.sin(this.heading);
      let dx = fx * fwd + rx * rgt, dz = fz * fwd + rz * rgt;
      const len = Math.hypot(dx, dz);
      if (len > 1) { dx /= len; dz /= len; }
      const k = 1 - Math.exp(-SUBSTEP * (len > 0 ? 7 : 10));
      this.vx += (dx * WALK_SPEED - this.vx) * k;
      this.vz += (dz * WALK_SPEED - this.vz) * k;
      if (Math.hypot(this.vx, this.vz) < 0.003) { this.vx = 0; this.vz = 0; }
      const nx = this.x + this.vx * SUBSTEP, nz = this.z + this.vz * SUBSTEP;
      const bx = this.x, bz = this.z;
      if (!this.world.blocked(nx, this.z)) this.x = nx; else if (this.vx) { this.collided = true; this.vx = 0; }
      if (!this.world.blocked(this.x, nz)) this.z = nz; else if (this.vz) { this.collided = true; this.vz = 0; }
      moved += Math.hypot(this.x - bx, this.z - bz);
      const speed = Math.hypot(this.x - bx, this.z - bz) / SUBSTEP;
      const target = Math.min(speed / WALK_SPEED, 1);
      this.blend += (target - this.blend) * (1 - Math.exp(-SUBSTEP * 10));
      this.phase += SUBSTEP * 10.5 * target;
      if ((i + 1) % 5 === 0) path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), round(this.blend, 3)]);
    }
    this.t += 1;

    let reason = '';
    if (interact > 0.5 && this.spec.kind === 'fetch_coffee' && !this.holding) {
      if (this.goalDist() <= COFFEE_RADIUS) { this.holding = true; reason = 'picked up the coffee cup'; }
      else reason = `too far from the cup (${this.goalDist().toFixed(2)} m, need <= ${COFFEE_RADIUS} m)`;
    }
    const dist = this.goalDist();
    let reward = -0.01 + 0.1 * (this.lastDist - dist);
    this.lastDist = dist;
    if (this.spec.kind === 'goto' && dist <= GOTO_RADIUS) { this.success = true; reason = `reached the ${this.spec.target.name}`; }
    if (this.spec.kind === 'fetch_coffee' && this.holding) this.success = true;
    if (this.success) reward += 1;
    this.done = this.success || this.t >= this.maxSteps;
    if (this.done && !this.success) reason = reason || 'ran out of steps';
    this.reason = reason;
    return {
      observation: this.observation(),
      reward: round(reward),
      done: this.done,
      info: {
        success: this.success, reason, position: [round(this.x), round(this.z)], heading: round(this.heading),
        y: round(this.world.groundHeight(this.x, this.z)), joints: this.pose(), collided: this.collided, holding: this.holding,
        distance: round(dist), moved: round(moved),
      },
      path,
    };
  }
}
