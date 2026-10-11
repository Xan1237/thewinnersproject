import { ARMCHAIR_ROOM_PLACEMENT, SIT_DURATION, sampleSit } from '../../robot/src/sitting-motion.js';

// Headless, deterministic simulation core for the G1 room arena. No browser or three.js needed.
// Contract: robot/docs/SIM_API.md. World: metres, Y up, heading 0 faces +Z, right = -X.

export const DT = 0.5;            // sim seconds per decision step
export const SUBSTEP = 0.02;      // integration step
export const ROBOT_RADIUS = 0.18;
export const WALK_SPEED = 1.5;
export const TURN_RATE = 1.5;     // rad/s at turn = 1
export const GOTO_RADIUS = 1.2;
export const COFFEE_RADIUS = 0.9;
export const SIT_RADIUS = 0.95;
export const SEAT_RADIUS = 0.65;
export const SEAT_ALIGNMENT = 0.4;
/** Emotes: a one-step task that ends with the robot playing an animation clip (see arena/src/animations.js). */
export const EMOTES = { jump: 3.0, wave: 4.0, dance: 4.0, cheer: 2.4 };
/** How far in front of a chair's centre a robot stands when it sits down (its pelvis moves back onto the seat). */
export const seatStandOffset = seat => Math.max(0.5, Math.max(seat.sx, seat.sz) / 2 + ROBOT_RADIUS + 0.04);

export const ACTION_SPEC = {
  names: ['forward', 'right', 'turn', 'interact'],
  low: [-1, -1, -1, 0],
  high: [1, 1, 1, 1],
  units: 'forward/right: fraction of 1.5 m/s relative to heading; turn: fraction of 1.5 rad/s (positive = turn LEFT / counter-clockwise seen from above, negative = turn RIGHT); interact: >0.5 grabs coffee within 0.9 m or starts fitted armchair sitting within 0.65 m of its front approach',
};
export const OBSERVATION_SPEC = {
  names: ['x', 'z', 'heading', 'goal_forward', 'goal_right', 'goal_dist', 'holding', 'steps_left_frac'],
  description: 'x,z in metres (world), heading in radians (0 faces +Z). goal_forward/goal_right: target position in the robot frame (metres; goal_right < 0 means the target is on your left, so turn left = positive turn). holding: 1 once the coffee cup is in hand.',
};

const normAngle = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
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
    // Every chair can be sat on. `rot` is the Godot yaw of the model (its front is -Z at 0), so the
    // sim heading of a person sitting in it is rot + 180 degrees.
    this.seats = this.objects.filter(o => o.kind === 'chair' || o.kind === 'folding').map(o => ({ ...o, heading: normAngle(((o.rot ?? 0) + 180) * Math.PI / 180) }));
    this.peers = () => [];   // the server plugs in the other live sessions (robots are obstacles for each other)
  }

  /** Is another robot standing within body distance of (x, z)? */
  peerBlocked(x, z, self) {
    for (const p of this.peers()) {
      if (p === self || p.done) continue;
      if (Math.hypot(p.x - x, p.z - z) < ROBOT_RADIUS * 2 - 0.02) return true;
    }
    return false;
  }

  /** Steps-in-metres walking distance from (fx, fz) to every free cell of a 0.1 m grid (BFS, 4-neighbour). */
  walkGrid(fx, fz) {
    const cell = 0.1, b = this.bounds;
    const nx = Math.ceil((b.maxX - b.minX) / cell) + 1, nz = Math.ceil((b.maxZ - b.minZ) / cell) + 1;
    const dist = new Float32Array(nx * nz).fill(Infinity);
    const ix = Math.round((fx - b.minX) / cell), iz = Math.round((fz - b.minZ) / cell);
    const queue = [[ix, iz]];
    dist[iz * nx + ix] = 0;
    for (let q = 0; q < queue.length; q++) {
      const [x, z] = queue[q], d = dist[z * nx + x];
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const X = x + dx, Z = z + dz;
        if (X < 0 || Z < 0 || X >= nx || Z >= nz || dist[Z * nx + X] !== Infinity) continue;
        if (this.blocked(b.minX + X * cell, b.minZ + Z * cell)) continue;
        dist[Z * nx + X] = d + cell;
        queue.push([X, Z]);
      }
    }
    return { dist, nx, nz, cell, b };
  }

  /** Walking distance to get within `radius` of `obj` (Infinity if walled off). */
  walkDistanceTo(grid, obj, radius) {
    let best = Infinity;
    const { dist, nx, nz, cell, b } = grid, r = Math.ceil(radius / cell);
    const cx = Math.round((obj.x - b.minX) / cell), cz = Math.round((obj.z - b.minZ) / cell);
    for (let z = Math.max(cz - r, 0); z <= Math.min(cz + r, nz - 1); z++) {
      for (let x = Math.max(cx - r, 0); x <= Math.min(cx + r, nx - 1); x++) {
        if (Math.hypot((x - cx) * cell, (z - cz) * cell) <= radius - 0.05) best = Math.min(best, dist[z * nx + x]);
      }
    }
    return best;
  }

  /** Can a robot walk the straight line from (ax, az) to within `stop` of (bx, bz)? */
  lineClear(ax, az, bx, bz, stop = 0.9) {
    const len = Math.hypot(bx - ax, bz - az);
    for (let d = 0; d < len - stop; d += 0.1) {
      const t = d / len;
      if (this.blocked(ax + (bx - ax) * t, az + (bz - az) * t)) return false;
    }
    return true;
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
      if (this.blocked(x, z) || this.peerBlocked(x, z, null)) continue;
      if (Math.hypot(goal.x - x, goal.z - z) < 2.5) continue;
      return { x, z, heading: (rand() * 2 - 1) * Math.PI };
    }
    throw new Error('no free start pose');
  }
}

export const TASK_HELP = 'goto:<object id> | fetch_coffee | sit (any free chair) | sit:<chair id> | jump | wave | dance | cheer | idle';

/** `ctx.claimed`: seat ids other robots already sit in or walk to. `ctx.from`: where the robot is now. */
export function resolveTask(world, task, ctx = {}) {
  const claimed = ctx.claimed ?? new Set();
  if (task === 'idle') {
    const at = ctx.from ?? { x: 0, z: 0 };
    return { kind: 'idle', target: { id: 'nothing', name: 'nothing', x: at.x, z: at.z }, radius: 0, description: 'Stand still and wait for instructions.' };
  }
  if (task === 'fetch_coffee') {
    const cup = world.cup;
    if (!cup) throw new Error('manifest has no coffee_cup');
    return { kind: 'fetch_coffee', target: cup, radius: COFFEE_RADIUS, description: 'Walk to the table on the right side of the room, pick up the coffee cup (set interact > 0.5 within 0.9 m of the cup) and hold it.' };
  }
  if (EMOTES[task]) {
    const at = ctx.from ?? { x: 0, z: 0 };
    return { kind: 'emote', emote: task, target: { id: 'self', name: 'yourself', x: at.x, z: at.z }, radius: 0, description: `Perform the "${task}" animation right where you stand (any action completes it).` };
  }
  const sit = /^sit(?::(.+))?$/.exec(task);
  if (sit) {
    const at = ctx.from ?? { x: 0, z: 0 };
    let seat = sit[1] && sit[1] !== 'any' ? world.seats.find(s => s.id === sit[1]) : null;
    if (sit[1] && sit[1] !== 'any' && !seat) throw new Error(`unknown chair id: ${sit[1]}`);
    // One robot per chair: a taken chair falls back to the nearest free one.
    if (!seat || claimed.has(seat.id)) {
      // Nearest free chair by walking distance (chairs walled off by the table are skipped).
      const grid = world.walkGrid(at.x, at.z), ref = seat ?? at;
      // A straight, unobstructed walk is preferred (robots are not good at going around furniture).
      const cost = s => world.walkDistanceTo(grid, s, SIT_RADIUS) + 0.3 * Math.hypot(s.x - ref.x, s.z - ref.z) + (world.lineClear(at.x, at.z, s.x, s.z) ? 0 : 8);
      seat = world.seats.filter(s => !claimed.has(s.id)).map(s => ({ s, c: cost(s) })).filter(e => Number.isFinite(e.c)).sort((a, b) => a.c - b.c)[0]?.s;
      if (!seat) throw new Error('no free chair left');
    }
    if (seat.id === 'armchair_blue') {
      const placement = { ...ARMCHAIR_ROOM_PLACEMENT, chairX: seat.x, chairZ: seat.z, chairYaw: Number.isFinite(seat.rot) ? seat.rot * Math.PI / 180 : ARMCHAIR_ROOM_PLACEMENT.chairYaw, groundY: world.groundHeight(seat.x, seat.z) };
      const approach = Math.max(seat.sx, seat.sz) / 2 + ROBOT_RADIUS + 0.2;
      const target = { ...seat, name: `${seat.name} front approach`, x: seat.x - Math.sin(placement.chairYaw) * approach, z: seat.z - Math.cos(placement.chairYaw) * approach, solid: false };
      if (world.blocked(target.x, target.z)) throw new Error('chair front approach is obstructed');
      return { kind: 'sit', fitted: true, target, chair: seat, placement, radius: SEAT_RADIUS, description: `Walk to the front approach of the ${seat.name}, at x=${target.x.toFixed(2)}, z=${target.z.toFixed(2)}. Stop within ${SEAT_RADIUS} m on the free floor and set interact > 0.5 to sit. Then send [0,0,0,0] until seated; the animation takes ${(SEAT_ALIGNMENT + SIT_DURATION).toFixed(1)} simulated seconds. Reaching the chair alone does not finish this task.` };
    }
    return { kind: 'sit', target: seat, radius: SIT_RADIUS, description: `Walk to the ${seat.name} and sit down on it (get within ${SIT_RADIUS} m of it).` };
  }
  const m = /^goto:(.+)$/.exec(task);
  if (m) {
    const target = world.byId.get(m[1]);
    if (!target) throw new Error(`unknown object id: ${m[1]}`);
    return { kind: 'goto', target, radius: GOTO_RADIUS, description: `Walk to the ${target.name} and stop within ${GOTO_RADIUS} m of it.` };
  }
  throw new Error(`unknown task: ${task}. Supported: ${TASK_HELP}`);
}

export class Session {
  constructor(world, { id, task = 'goto:armchair_blue', seed = 0, model = 'unknown', maxSteps = 120, claimed, preview = false } = {}) {
    this.world = world;
    this.id = id;
    this.seed = seed;
    this.model = model;
    this.maxSteps = maxSteps;
    this.task = task;
    const rand = mulberry32(seed * 7919 + 13);
    const first = resolveTask(world, task, { claimed });
    if (preview && (task !== 'sit:armchair_blue' || !first.fitted)) throw new Error('preview requires an available sit:armchair_blue task');
    const start = preview ? { x: first.target.x, z: first.target.z, heading: normAngle(sampleSit(0, first.placement).heading) } : world.startPose(rand, first.target);
    // Tasks without a fixed target (sit on any chair, emotes, idle) depend on where the robot starts.
    this.spec = ['sit', 'emote', 'idle'].includes(first.kind) ? resolveTask(world, task, { claimed, from: start }) : first;
    this.seated = false;
    this.seating = false;
    this.seatTime = -1;
    this.seatElapsed = 0;
    this.seatFrom = null;
    this.poseOffset = null;
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

  blockedAt(x, z) { return this.world.blocked(x, z) || this.world.peerBlocked(x, z, this); }

  /** Give the robot a new task where it stands (voice commands). `claimed`: chairs other robots hold. */
  retask(task, claimed) {
    this.spec = resolveTask(this.world, task, { claimed, from: { x: this.x, z: this.z } });
    this.task = task;
    this.t = 0;
    this.done = false;
    this.success = false;
    this.reason = '';
    this.holding = false;
    this.seated = false;
    this.seating = false;
    this.seatTime = -1;
    this.seatElapsed = 0;
    this.seatFrom = null;
    this.poseOffset = null;
    this.collided = false;
    this.vx = 0;
    this.vz = 0;
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
    const joints = this.seatTime >= 0 ? sampleSit(this.seatTime, this.spec.placement).joints : gaitJoints(this.phase, this.blend);
    if (this.holding) Object.assign(joints, HOLD_POSE);
    return Object.fromEntries(Object.entries(joints).map(([k, v]) => [k, round(v)]));
  }

  observation() {
    const g = this.goalLocal();
    const state = [this.x, this.z, this.heading, g.forward, g.right, this.goalDist(), this.holding ? 1 : 0, 1 - this.t / this.maxSteps].map(v => round(v));
    return { t: this.t, state, text: this.describe(g), seating: this.seating, seated: this.seated, seat_time: round(this.seatTime) };
  }

  describe(g = this.goalLocal()) {
    const lines = [];
    lines.push(`You are at x=${this.x.toFixed(2)}, z=${this.z.toFixed(2)}, facing ${(this.heading * 180 / Math.PI).toFixed(0)} degrees.`);
    if (this.spec.kind === 'idle' || this.spec.kind === 'emote') lines.push(this.spec.description);
    else lines.push(`Target "${this.spec.target.name}": ${Math.abs(g.forward).toFixed(2)} m ${g.forward >= 0 ? 'ahead' : 'behind you'}, ${Math.abs(g.right).toFixed(2)} m to your ${g.right >= 0 ? 'right' : 'left'}, distance ${this.goalDist().toFixed(2)} m.`);
    if (this.collided) lines.push('Your last move was blocked by an obstacle or wall.');
    if (this.holding) lines.push('You are holding the coffee cup.');
    if (this.seated) lines.push('You are seated in a chair.');
    else if (this.seating) lines.push(`Sitting animation in progress (${this.seatTime.toFixed(2)} / ${SIT_DURATION.toFixed(2)} s). Send [0,0,0,0] to wait; movement is paused until seated.`);
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
    let reason = '';
    if (interact > 0.5 && this.spec.fitted && !this.seating && !this.seated) {
      if (this.goalDist() <= SEAT_RADIUS && !this.blockedAt(this.x, this.z)) {
        this.seating = true; this.seatTime = 0;
        this.seatFrom = { x: this.x, z: this.z, heading: this.heading };
        this.vx = 0; this.vz = 0;
        reason = 'started sitting in the armchair';
      } else reason = `approach the free front of the armchair (${this.goalDist().toFixed(2)} m, need <= ${SEAT_RADIUS} m)`;
    }
    const n = Math.round(DT / SUBSTEP);
    for (let i = 0; i < n; i++) {
      if (this.seatTime >= 0) {
        this.seatElapsed += SUBSTEP;
        this.seatTime = clamp(this.seatElapsed - SEAT_ALIGNMENT, 0, SIT_DURATION);
        const pose = sampleSit(this.seatTime, this.spec.placement);
        const u = clamp(this.seatElapsed / SEAT_ALIGNMENT, 0, 1);
        const ease = u * u * (3 - 2 * u);
        this.x = this.seatFrom.x + (pose.position[0] - this.seatFrom.x) * ease;
        this.z = this.seatFrom.z + (pose.position[2] - this.seatFrom.z) * ease;
        this.heading = normAngle(this.seatFrom.heading + normAngle(pose.heading - this.seatFrom.heading) * ease);
        this.blend = 0; this.seated = pose.seated; this.seating = !this.seated;
        if ((i + 1) % 5 === 0) path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), 0, round(this.seatTime, 3)]);
        continue;
      }
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
      if (!this.blockedAt(nx, this.z)) this.x = nx; else if (this.vx) { this.collided = true; this.vx = 0; }
      if (!this.blockedAt(this.x, nz)) this.z = nz; else if (this.vz) { this.collided = true; this.vz = 0; }
      moved += Math.hypot(this.x - bx, this.z - bz);
      const speed = Math.hypot(this.x - bx, this.z - bz) / SUBSTEP;
      const target = Math.min(speed / WALK_SPEED, 1);
      this.blend += (target - this.blend) * (1 - Math.exp(-SUBSTEP * 10));
      this.phase += SUBSTEP * 10.5 * target;
      if ((i + 1) % 5 === 0) path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), round(this.blend, 3), -1]);
    }
    this.t += 1;

    if (interact > 0.5 && this.spec.kind === 'fetch_coffee' && !this.holding) {
      if (this.goalDist() <= COFFEE_RADIUS) { this.holding = true; reason = 'picked up the coffee cup'; }
      else reason = `too far from the cup (${this.goalDist().toFixed(2)} m, need <= ${COFFEE_RADIUS} m)`;
    }
    const dist = this.goalDist();
    let reward = -0.01 + 0.1 * (this.lastDist - dist);
    this.lastDist = dist;
    let pose = null, poseS = 0;
    if (this.spec.kind === 'goto' && dist <= GOTO_RADIUS) { this.success = true; reason = `reached the ${this.spec.target.name}`; }
    if (this.spec.kind === 'sit' && !this.spec.fitted && dist <= SIT_RADIUS) {
      const seat = this.spec.target;
      this.success = true; this.seated = true; reason = `sat down on the ${seat.name}`;
      const off = seatStandOffset(seat);
      this.x = seat.x + Math.sin(seat.heading) * off; this.z = seat.z + Math.cos(seat.heading) * off; this.heading = seat.heading; this.vx = this.vz = 0; this.blend = 0; this.poseOffset = off;
      path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), 0]);
    }
    if (this.spec.kind === 'emote') { this.success = true; pose = this.spec.emote; poseS = EMOTES[pose]; reason = `played the ${pose} animation`; }
    if (this.seated && !this.spec.fitted) pose = 'sit';
    if (this.spec.fitted && this.seated) { this.success = true; reason = `seated in the ${this.spec.chair.name}`; }
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
        distance: round(dist), moved: round(moved), seating: this.seating, seated: this.seated, seat_time: round(this.seatTime), pelvis_height: this.seatTime >= 0 ? round(sampleSit(this.seatTime, this.spec.placement).pelvisHeight) : null, pose, pose_s: poseS, pose_offset: this.seated ? this.poseOffset : null, seat: this.seated ? this.spec.target.id : null,
      },
      path,
    };
  }
}
