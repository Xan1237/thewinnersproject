// Headless, deterministic simulation core for the G1 room arena. No browser or three.js needed.
// Contract: robot/docs/SIM_API.md. World: metres, Y up, heading 0 faces +Z, right = -X.

export const DT = 0.5;            // sim seconds per decision step
export const SUBSTEP = 0.02;      // integration step
export const ROBOT_RADIUS = 0.18;
export const WALK_SPEED = 1.5;
export const TURN_RATE = 1.5;     // rad/s at turn = 1
export const GOTO_RADIUS = 1.2;
export const COFFEE_RADIUS = 1.0;
export const SIT_RADIUS = 0.95;
export const STATION_RADIUS = 0.95;   // how close a robot must be to use a machine / counter spot
export const BREW_STEPS = 8;          // decision steps the machine needs to brew
/** The coffee line. Two long tables stand where the audience chairs used to be; every robot has its own numbered spot on both. */
export const BAR = { slots: 8, x0: -5.15, dx: 0.5, machineZ: 2.9, serveZ: 0.8, length: 3.6, depth: 0.7 };
export const barSlot = (slot, which) => ({ id: `${which}_${slot}`, kind: 'station', slot, x: BAR.x0 + BAR.dx * (slot % BAR.slots), z: which === 'machine' ? BAR.machineZ : BAR.serveZ,
  name: which === 'machine' ? 'your coffee spout' : 'your spot on the serving counter' });
const MACHINE_CODE = { empty: 0, cup: 1 / 3, brewing: 2 / 3, ready: 1 };
/** Emotes: a one-step task that ends with the robot playing an animation clip (see arena/src/animations.js). */
export const EMOTES = { jump: 3.0, wave: 4.0, dance: 4.0, cheer: 2.4 };
/** How far in front of a chair's centre a robot stands when it sits down (its pelvis moves back onto the seat). */
export const seatStandOffset = seat => Math.max(0.5, Math.max(seat.sx, seat.sz) / 2 + ROBOT_RADIUS + 0.04);

export const ACTION_SPEC = {
  names: ['forward', 'right', 'turn', 'interact'],
  low: [-1, -1, -1, 0],
  high: [1, 1, 1, 1],
  units: 'forward/right: fraction of 1.5 m/s relative to heading; turn: fraction of 1.5 rad/s (positive = turn LEFT / counter-clockwise seen from above, negative = turn RIGHT); interact: >0.5 uses what is in reach (picks up, puts down or presses a button) when within 1.0 m',
};
export const OBSERVATION_SPEC = {
  names: ['x', 'z', 'heading', 'goal_forward', 'goal_right', 'goal_dist', 'holding', 'steps_left_frac', 'task_progress', 'can_use', 'must_wait', 'machine'],
  description: 'task_progress: how far through the recipe (0..1). can_use: 1 when the current part needs the interact action. must_wait: 1 while the current part is waiting for the machine. machine: your coffee spout (0 empty, 0.33 cup in, 0.67 brewing, 1 ready). x,z in metres (world), heading in radians (0 faces +Z). goal_forward/goal_right: target position in the robot frame (metres; goal_right < 0 means the target is on your left, so turn left = positive turn). holding: 1 once the coffee cup is in hand.',
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
    this.bounds = manifest.bounds;
    // The audience chairs and the chairs at the cup table make room for the coffee line.
    this.objects = [
      ...manifest.objects.filter(o => !/^(fold_chair|table_chair)/.test(o.id)),
      { id: 'coffee_bar', kind: 'table', name: 'coffee machine bar', x: (2 * BAR.x0 + BAR.dx * (BAR.slots - 1)) / 2, z: BAR.machineZ, sx: BAR.length, sz: BAR.depth, rot: 0, height: 0.74, solid: true },
      { id: 'serve_bar', kind: 'table', name: 'serving counter', x: (2 * BAR.x0 + BAR.dx * (BAR.slots - 1)) / 2, z: BAR.serveZ, sx: BAR.length, sz: BAR.depth, rot: 0, height: 0.74, solid: true },
    ];
    this.manifest = { ...manifest, objects: this.objects, bar: BAR };
    this.byId = new Map(this.objects.map(o => [o.id, o]));
    this.solids = this.objects.filter(o => o.solid);
    this.platforms = this.objects.filter(o => o.kind === 'platform');
    this.cup = this.byId.get('coffee_cup') ?? null;
    // Every chair can be sat on. `rot` is the Godot yaw of the model (its front is -Z at 0), so the
    // sim heading of a person sitting in it is rot + 180 degrees.
    this.seats = this.objects.filter(o => o.kind === 'chair' || o.kind === 'folding').map(o => ({ ...o, heading: normAngle(((o.rot ?? 0) + 180) * Math.PI / 180) }));
    this.peers = () => [];   // the server plugs in the other live sessions (robots are obstacles for each other)
  }

  /** The lowest numbered spot on the coffee line that no other robot uses. */
  freeSlot(self) {
    const used = new Set([...this.peers()].filter(p => p !== self).map(p => p.slot));
    for (let i = 0; i < BAR.slots; i++) if (!used.has(i)) return i;
    return 0;
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

/** The one task: make a coffee and serve it. Seven parts, one instruction in plain words. */
export const COFFEE_INSTRUCTION = 'Make yourself a coffee and serve it. Take a cup from the cup table, put it under your own spout on the coffee machine bar, press the brew button, wait until the coffee is ready, take the full cup, put it down on your own spot on the serving counter, then cheer.';
export const SCENARIOS = {
  make_coffee: { plan: ['fetch_coffee', 'station:place', 'station:press', 'station:wait', 'station:take', 'station:serve', 'cheer'], instruction: COFFEE_INSTRUCTION },
};

/** What an object looks like to the robots. They never see ids, only these plain words. */
const LABELS = {
  chair_navy_left: 'navy wooden chair on the stage', chair_navy_middle: 'navy wooden chair on the stage', armchair_blue: 'big blue armchair on the stage',
  plant_tall: 'tall leafy plant', plant_medium: 'potted plant', blue_column: 'blue column', table: 'white cup table', coffee_bar: 'coffee machine bar (one spout per robot)', serve_bar: 'serving counter',
  coffee_cup: 'cups on the cup table (reachable from any side of the table)', speaker_stand: 'speaker on a stand', tripod: 'camera tripod', lectern: 'white lectern',
};
export function labelOf(o) {
  if (LABELS[o.id]) return LABELS[o.id];
  if (/^fold_chair/.test(o.id)) return 'grey folding chair in the audience';
  if (/^table_chair/.test(o.id)) return 'white folding chair by the table';
  return o.name;
}

export const TASK_HELP = `make_coffee (the full recipe) | goto:<object id> | fetch_coffee | sit (any free chair) | sit:<chair id> | jump | wave | dance | cheer | idle`;

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
    return { kind: 'fetch_coffee', target: cup, radius: COFFEE_RADIUS, description: 'Walk to the cup table on the right side of the room, pick up a cup (set interact > 0.5 within 1.0 m of the edge of the table) and hold it.' };
  }
  const st = /^station:(place|press|wait|take|serve)$/.exec(task);
  if (st) {
    const op = st[1], slot = ctx.slot ?? 0;
    const target = barSlot(slot, op === 'serve' ? 'serve' : 'machine');
    const text = { place: 'Put the cup under your spout (interact within 1 m of it while holding the cup).', press: 'Press the brew button (interact within 1 m of your spout, once the cup is in).', wait: 'Wait next to your spout until the coffee is ready.', take: 'Take the full cup (interact within 1 m of your spout).', serve: 'Put the cup down on your spot on the serving counter (interact within 1 m of it).' }[op];
    return { kind: 'station', op, target, radius: STATION_RADIUS, description: text };
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
      const cost = s => world.walkDistanceTo(grid, s, SIT_RADIUS) + 0.3 * Math.hypot(s.x - ref.x, s.z - ref.z) + (world.lineClear(at.x, at.z, s.x, s.z) ? 0 : 8) + (s.id.startsWith('table_chair') ? 6 : 0);
      seat = world.seats.filter(s => !claimed.has(s.id)).map(s => ({ s, c: cost(s) })).filter(e => Number.isFinite(e.c)).sort((a, b) => a.c - b.c)[0]?.s;
      if (!seat) throw new Error('no free chair left');
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
  constructor(world, { id, task = 'goto:armchair_blue', seed = 0, model = 'unknown', maxSteps = 120, claimed, instruction = null } = {}) {
    this.world = world;
    this.id = id;
    this.seed = seed;
    this.model = model;
    this.maxSteps = maxSteps;
    this.task = task;
    this.plan = SCENARIOS[task]?.plan ?? [task];
    this.stage = 0;
    this.slot = world.freeSlot(this);
    this.machine = { state: 'empty', left: 0 };
    this.served = false;
    // With an instruction the robot is "blind": it gets the words and a view of the room, not the target.
    this.instruction = instruction ?? SCENARIOS[task]?.instruction ?? null;
    const rand = mulberry32(seed * 7919 + 13);
    const first = resolveTask(world, this.plan[0], { claimed, slot: this.slot });
    const start = world.startPose(rand, first.target);
    // Tasks without a fixed target (sit on any chair, emotes, idle) depend on where the robot starts.
    this.spec = ['sit', 'emote', 'idle'].includes(first.kind) ? resolveTask(world, this.plan[0], { claimed, from: start, slot: this.slot }) : first;
    this.seated = false;
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
  retask(task, claimed, instruction = null) {
    this.plan = SCENARIOS[task]?.plan ?? [task];
    this.stage = 0;
    this.slot = this.world.freeSlot(this);
    this.machine = { state: 'empty', left: 0 };
    this.served = false;
    this.instruction = instruction ?? SCENARIOS[task]?.instruction ?? null;
    this.spec = resolveTask(this.world, this.plan[0], { claimed, from: { x: this.x, z: this.z }, slot: this.slot });
    this.task = task;
    this.t = 0;
    this.done = false;
    this.success = false;
    this.reason = '';
    this.holding = false;
    this.seated = false;
    this.collided = false;
    this.vx = 0;
    this.vz = 0;
    this.lastDist = this.goalDist();
  }

  /** What the agent is told the task is. */
  get taskText() { return this.instruction ?? this.spec.description; }

  goalDist() {
    const t = this.spec.target;
    if (this.spec.kind === 'fetch_coffee') {   // the cups stand somewhere on the cup table: you can reach them from any side of it
      const tb = this.world.byId.get('table');
      if (tb) return Math.hypot(Math.max(Math.abs(this.x - tb.x) - tb.sx / 2, 0), Math.max(Math.abs(this.z - tb.z) - tb.sz / 2, 0));
    }
    return Math.hypot(t.x - this.x, t.z - this.z);
  }

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

  /** Recipe progress and machine status: what a real controller would get from its task monitor and the machine's own sensor. */
  taskFeatures() {
    const k = this.spec.kind, op = this.spec.op;
    const canUse = (k === 'fetch_coffee' && !this.holding) || (k === 'station' && op !== 'wait');
    return [this.plan.length > 1 ? this.stage / (this.plan.length - 1) : 0, canUse ? 1 : 0, op === 'wait' ? 1 : 0, MACHINE_CODE[this.machine.state]];
  }

  /** Where the robot's own navigation should aim: the goal itself when the way is clear, otherwise a point ~1 m down a path around the furniture. */
  navTarget() {
    const t = this.spec.target;
    if (this.spec.kind === 'idle' || this.spec.kind === 'emote' || this.world.lineClear(this.x, this.z, t.x, t.z, Math.max(this.spec.radius - 0.1, 0.3))) return t;
    const reach = Math.max(this.spec.radius - 0.1, 0.3);
    const from = this.world.walkGrid(this.x, this.z);
    let best = null, bestD = Infinity;
    const { dist, nx, nz, cell, b } = from, r = Math.ceil(reach / cell);
    const cx = Math.round((t.x - b.minX) / cell), cz = Math.round((t.z - b.minZ) / cell);
    for (let z = Math.max(cz - r, 0); z <= Math.min(cz + r, nz - 1); z++) for (let x = Math.max(cx - r, 0); x <= Math.min(cx + r, nx - 1); x++) {
      if (Math.hypot((x - cx) * cell, (z - cz) * cell) <= reach && dist[z * nx + x] < bestD) { bestD = dist[z * nx + x]; best = [x, z]; }
    }
    if (!best) return t;
    const to = this.world.walkGrid(b.minX + best[0] * cell, b.minZ + best[1] * cell).dist;
    let x = Math.round((this.x - b.minX) / cell), z = Math.round((this.z - b.minZ) / cell);
    for (let i = 0; i < 10; i++) {
      let nxt = null, nd = to[z * nx + x];
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const X = x + dx, Z = z + dz;
        if (X >= 0 && Z >= 0 && X < nx && Z < nz && to[Z * nx + X] < nd) { nd = to[Z * nx + X]; nxt = [X, Z]; }
      }
      if (!nxt) break;
      [x, z] = nxt;
    }
    return { x: b.minX + x * cell, z: b.minZ + z * cell };
  }

  observation() {
    const real = this.goalLocal(), g = this.goalLocal(this.navTarget());
    const state = [this.x, this.z, this.heading, g.forward, g.right, this.goalDist(), this.holding ? 1 : 0, 1 - this.t / this.maxSteps, ...this.taskFeatures()].map(v => round(v));
    return { t: this.t, state, text: this.describe(real) };
  }

  describe(g = this.goalLocal()) {
    if (this.instruction) return this.describeScene();
    const lines = [];
    lines.push(`You are at x=${this.x.toFixed(2)}, z=${this.z.toFixed(2)}, facing ${(this.heading * 180 / Math.PI).toFixed(0)} degrees.`);
    if (this.spec.kind === 'idle' || this.spec.kind === 'emote') lines.push(this.spec.description);
    else lines.push(`Target "${this.spec.target.name}": ${Math.abs(g.forward).toFixed(2)} m ${g.forward >= 0 ? 'ahead' : 'behind you'}, ${Math.abs(g.right).toFixed(2)} m to your ${g.right >= 0 ? 'right' : 'left'}, distance ${this.goalDist().toFixed(2)} m.`);
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

  /** The agent's view when it only has words: an instruction, its own status and what it can see. No target. */
  describeScene() {
    const lines = [`Your instruction: "${this.instruction}"`];
    if (this.plan.length > 1) lines.push(`Progress: part ${this.stage + 1} of ${this.plan.length}${this.stage ? ' (the first part is done)' : ''}.`);
    lines.push(`You are at x=${this.x.toFixed(1)}, z=${this.z.toFixed(1)}, facing ${(this.heading * 180 / Math.PI).toFixed(0)} degrees. ${this.holding ? 'You are holding a coffee cup.' : 'Your hands are empty.'}${this.seated ? ' You are sitting down.' : ''}`);
    if (this.collided) lines.push('Your last move was blocked by an obstacle or wall.');
    if (this.plan.some(s => s.startsWith('station:'))) {
      const m = this.machine;
      lines.push(`Your coffee spout: ${{ empty: 'empty', cup: 'your cup is under it, not brewing yet', brewing: `brewing (about ${Math.max(m.left, 1)} more steps)`, ready: 'the coffee is ready' }[m.state]}.${this.served ? ' Your cup is on the serving counter.' : ''}`);
    }
    const taken = new Set([...this.world.peers()].filter(p => p !== this && p.seated).map(p => p.spec.target.id));
    const groups = new Map();
    for (const o of this.world.objects) {
      if (o.kind === 'platform' || (taken.has(o.id) && o.kind !== 'chair')) continue;
      const l = this.goalLocal(o), deg = Math.atan2(l.right, l.forward) * 180 / Math.PI;
      const label = labelOf(o) + (taken.has(o.id) ? ' (someone is sitting there)' : '');
      const d = Math.hypot(o.x - this.x, o.z - this.z);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push({ d, deg });
    }
    if (this.plan.some(s => s.startsWith('station:'))) {
      for (const [which, label] of [['machine', 'your coffee spout on the coffee machine bar'], ['serve', 'your own spot on the serving counter']]) {
        const t = barSlot(this.slot, which), l = this.goalLocal(t);
        groups.set(label, [{ d: Math.hypot(t.x - this.x, t.z - this.z), deg: Math.atan2(l.right, l.forward) * 180 / Math.PI }]);
      }
    }
    const where = ({ d, deg }) => {
      const a = Math.abs(deg), side = deg >= 0 ? 'right' : 'left';
      const dir = a < 10 ? 'straight ahead' : a > 150 ? 'directly behind you' : `${a.toFixed(0)}° to your ${side}${a > 100 ? ' (behind you)' : ''}`;
      return `${d.toFixed(1)} m away, ${dir}`;
    };
    const rows = [...groups.entries()].map(([label, list]) => ({ label, list: list.sort((x, y) => x.d - y.d) })).sort((x, y) => x.list[0].d - y.list[0].d);
    lines.push('You can use (pick up, put down, press) whatever is within 1 m of you, and you sit down by walking up to a free chair.');
    lines.push('You can see:');
    for (const { label, list } of rows) {
      const shown = list.slice(0, 2).map(where).join('; also ');
      lines.push(`- ${label}${list.length > 1 ? ` (${list.length} of them)` : ''}: nearest ${shown}`);
    }
    lines.push(`The room spans x from ${this.world.bounds.minX} to ${this.world.bounds.maxX} and z from ${this.world.bounds.minZ} to ${this.world.bounds.maxZ}.`);
    return lines.join('\n');
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
      if (!this.blockedAt(nx, this.z)) this.x = nx; else if (this.vx) { this.collided = true; this.vx = 0; }
      if (!this.blockedAt(this.x, nz)) this.z = nz; else if (this.vz) { this.collided = true; this.vz = 0; }
      moved += Math.hypot(this.x - bx, this.z - bz);
      const speed = Math.hypot(this.x - bx, this.z - bz) / SUBSTEP;
      const target = Math.min(speed / WALK_SPEED, 1);
      this.blend += (target - this.blend) * (1 - Math.exp(-SUBSTEP * 10));
      this.phase += SUBSTEP * 10.5 * target;
      if ((i + 1) % 5 === 0) path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), round(this.blend, 3)]);
    }
    this.t += 1;

    let reason = '';
    if (this.machine.state === 'brewing' && --this.machine.left <= 0) this.machine.state = 'ready';
    if (this.spec.kind === 'station') {
      const op = this.spec.op, d = this.goalDist(), m = this.machine, near = d <= STATION_RADIUS;
      if (op === 'wait') {
        if (m.state === 'ready' && d <= 1.6) { this.success = true; reason = 'the coffee is ready'; }
        else if (m.state !== 'brewing') reason = 'the machine is not brewing';
      } else if (interact > 0.5) {
        if (!near) reason = `too far from your spot (${d.toFixed(2)} m, need <= ${STATION_RADIUS} m)`;
        else if (op === 'place') { if (this.holding) { this.holding = false; m.state = 'cup'; this.success = true; reason = 'put the cup under the spout'; } else reason = 'you are not holding a cup'; }
        else if (op === 'press') { if (m.state === 'cup') { m.state = 'brewing'; m.left = BREW_STEPS; this.success = true; reason = 'pressed the brew button'; } else reason = 'there is no cup under the spout'; }
        else if (op === 'take') { if (m.state === 'ready') { this.holding = true; m.state = 'empty'; this.success = true; reason = 'took the full cup'; } else reason = 'the coffee is not ready yet'; }
        else if (op === 'serve') { if (this.holding) { this.holding = false; this.served = true; this.success = true; reason = 'served the coffee'; } else reason = 'you are not holding a cup'; }
      }
    }
    if (interact > 0.5 && this.spec.kind === 'fetch_coffee' && !this.holding) {
      if (this.goalDist() <= COFFEE_RADIUS) { this.holding = true; reason = 'picked up the coffee cup'; }
      else reason = `too far from the cup table (${this.goalDist().toFixed(2)} m from its edge, need <= ${COFFEE_RADIUS} m)`;
    }
    const dist = this.goalDist();
    let reward = -0.01 + 0.1 * (this.lastDist - dist);
    this.lastDist = dist;
    let pose = null, poseS = 0;
    if (this.spec.kind === 'goto' && dist <= GOTO_RADIUS) { this.success = true; reason = `reached the ${this.spec.target.name}`; }
    // Sitting counts on ANY free chair within reach, not only the one the sim would have picked.
    let seatHit = null;
    if (this.spec.kind === 'sit') {
      const taken = new Set([...this.world.peers()].filter(p => p !== this && p.seated).map(p => p.spec.target.id));
      seatHit = this.world.seats.filter(s => !taken.has(s.id) && Math.hypot(s.x - this.x, s.z - this.z) <= SIT_RADIUS).sort((a, b) => Math.hypot(a.x - this.x, a.z - this.z) - Math.hypot(b.x - this.x, b.z - this.z))[0] ?? null;
    }
    if (seatHit) {
      const seat = this.spec.target = seatHit;
      this.success = true; this.seated = true; reason = `sat down on the ${seat.name}`;
      const off = seatStandOffset(seat);
      this.x = seat.x + Math.sin(seat.heading) * off; this.z = seat.z + Math.cos(seat.heading) * off; this.heading = seat.heading; this.vx = this.vz = 0; this.blend = 0; this.poseOffset = off;
      path.push([round(this.x), round(this.z), round(this.heading), round(this.phase, 3), 0]);
    }
    if (this.spec.kind === 'emote') { this.success = true; pose = this.spec.emote; poseS = EMOTES[pose]; reason = `played the ${pose} animation`; }
    if (this.seated) pose = 'sit';
    if (this.spec.kind === 'fetch_coffee' && this.holding) this.success = true;
    let stageDone = null;
    if (this.success && this.stage < this.plan.length - 1) {   // part of a scenario: move on to the next part
      stageDone = this.plan[this.stage];
      this.stage += 1;
      this.success = false;
      this.spec = resolveTask(this.world, this.plan[this.stage], { claimed: new Set([...this.world.peers()].filter(p => p !== this && p.spec.kind === 'sit').map(p => p.spec.target.id)), from: { x: this.x, z: this.z }, slot: this.slot });
      this.lastDist = this.goalDist();
      reward += 0.5;
    }
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
        stage: this.stage, stages: this.plan.length, stage_done: stageDone, stage_name: this.plan[this.stage],
        bar: this.plan.some(s => s.startsWith('station:')) ? { slot: this.slot, machine: this.machine.state, left: this.machine.left, served: this.served } : null,
        distance: round(this.goalDist()), moved: round(moved), pose, pose_s: poseS, pose_offset: this.seated ? this.poseOffset : null, seat: this.seated ? this.spec.target.id : null,
      },
      path,
    };
  }
}
