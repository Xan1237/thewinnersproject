import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Session, World, ACTION_SPEC } from './core.js';
import { createServer } from './server.js';

const here = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(here, '..', 'public', 'room', 'manifest.json'), 'utf8'));
const world = new World(manifest);

// Simple greedy controller used to prove the tasks are solvable.
function greedy(s) {
  const g = s.goalLocal();
  const ang = Math.atan2(-g.right, g.forward); // positive turn = toward +X = toward robot's left
  const turn = Math.max(-1, Math.min(1, -ang * 1.5));
  const fwd = Math.abs(ang) < 0.6 ? 1 : 0;
  return [fwd, 0, -turn, s.spec.kind === 'fetch_coffee' && s.goalDist() < 0.85 ? 1 : 0];
}

test('forward motion follows heading (0 faces +Z)', () => {
  const s = new Session(world, { id: 't', task: 'goto:armchair_blue', seed: 1 });
  s.x = 0; s.z = 4; s.heading = Math.PI; // facing -Z
  for (let i = 0; i < 2; i++) s.step([1, 0, 0, 0]);
  assert.ok(s.z < 4 - 0.5, `moved north, z=${s.z}`);
  assert.ok(Math.abs(s.x) < 1e-6);
});

test('walls and solid objects block the robot', () => {
  const s = new Session(world, { id: 't', task: 'goto:armchair_blue', seed: 1 });
  s.x = 0; s.z = 5; s.heading = 0; // facing +Z toward the south wall
  for (let i = 0; i < 10; i++) s.step([1, 0, 0, 0]);
  assert.ok(s.z <= world.bounds.maxZ - 0.28 + 1e-6, `stopped at wall, z=${s.z}`);
  const t = world.byId.get('table');
  s.x = t.x; s.z = t.z + 3; s.heading = Math.PI;
  for (let i = 0; i < 12; i++) s.step([1, 0, 0, 0]);
  assert.ok(s.z >= t.z + t.sz / 2 + 0.18 - 1e-6, `stopped at table, z=${s.z}`);
});

test('deterministic given a seed and actions', () => {
  const run = () => { const s = new Session(world, { id: 'a', seed: 5 }); const out = []; for (let i = 0; i < 10 && !s.done; i++) out.push(s.step([1, 0.2, 0.3, 0]).observation.state); return JSON.stringify(out); };
  assert.equal(run(), run());
});

test('actions are clipped and NaN-safe', () => {
  const s = new Session(world, { id: 'a', seed: 2 });
  const r = s.step([99, NaN, -99, 7]);
  assert.ok(Number.isFinite(r.observation.state[0]));
  assert.equal(ACTION_SPEC.names.length, 4);
});

test('goto tasks are solvable with a greedy policy (open-field targets)', () => {
  let solved = 0;
  const ids = ['lectern', 'table', 'speaker_stand', 'plant_tall', 'plant_medium', 'armchair_blue'];
  for (const [i, id] of ids.entries()) {
    const s = new Session(world, { id: 'g', task: `goto:${id}`, seed: i, maxSteps: 120 });
    while (!s.done) s.step(greedy(s));
    if (s.success) solved++;
  }
  console.log(`greedy solved ${solved}/${ids.length}`);
  assert.ok(solved >= 5, `greedy solved ${solved}/${ids.length}`);
});

test('fetch_coffee: picking up requires being close', () => {
  const s = new Session(world, { id: 'c', task: 'fetch_coffee', seed: 3 });
  const far = s.step([0, 0, 0, 1]);
  assert.equal(far.info.success, false);
  assert.match(far.info.reason, /too far/);
  s.x = 4.3; s.z = 2.0; s.heading = Math.PI;
  const near = s.step([0, 0, 0, 1]);
  assert.equal(near.info.success, true);
  assert.equal(near.info.holding, true);
  assert.ok(near.info.joints.right_elbow_joint > 1.2);
});

test('HTTP API: session lifecycle, SSE, validation', async () => {
  const { server } = createServer({ dataDir: mkdtempSync(join(tmpdir(), 'arena-data-')) });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (p, b) => fetch(base + p, { method: 'POST', body: JSON.stringify(b) }).then(async r => [r.status, await r.json()]);
  const [st, created] = await post('/sim/sessions', { task: 'goto:lectern', seed: 1, model: 'test-model' });
  assert.equal(st, 200);
  assert.deepEqual(created.action_spec.names, ACTION_SPEC.names);
  const [st2, stepped] = await post(`/sim/sessions/${created.id}/step`, { action: [1, 0, 0, 0], label: 'walk ahead', model: 'test-model' });
  assert.equal(st2, 200);
  assert.ok(stepped.info.joints.left_knee_joint !== undefined);
  assert.equal((await post(`/sim/sessions/${created.id}/step`, { action: 'bad' }))[0], 400);
  assert.equal((await post('/sim/sessions/nope/step', { action: [0, 0, 0, 0] }))[0], 404);
  assert.equal((await post('/sim/sessions', { task: 'goto:nonexistent' }))[0], 400);
  const ac = new AbortController();
  const sse = await fetch(base + '/sim/events', { signal: ac.signal });
  const reader = sse.body.getReader();
  const { value } = await reader.read();
  let text = new TextDecoder().decode(value);
  while (!text.includes('"type":"step"')) text += new TextDecoder().decode((await reader.read()).value);
  assert.match(text, /"label":"walk ahead"/);
  ac.abort();
  server.close();
  server.closeAllConnections?.();
});

test('seat task targets the real armchair front, while goto keeps its centre target', () => {
  const seat = new Session(world, { id: 'seat', task: 'sit:armchair_blue', preview: true });
  const go = new Session(world, { id: 'go', task: 'goto:armchair_blue' });
  assert.equal(go.spec.target, world.byId.get('armchair_blue'));
  assert.equal(seat.spec.placement.chairX, 1.5);
  assert.equal(seat.spec.placement.chairZ, -3.7);
  assert.equal(seat.spec.placement.groundY, 0.35);
  assert.ok(Math.abs(seat.spec.placement.chairYaw - 155 * Math.PI / 180) < 1e-9);
  assert.ok(!world.blocked(seat.x, seat.z), 'front approach must be on free floor');
  const dx = seat.x - 1.5, dz = seat.z + 3.7;
  const localZ = dx * Math.sin(seat.spec.placement.chairYaw) + dz * Math.cos(seat.spec.placement.chairYaw);
  assert.ok(localZ < -0.5, `model front is local -Z, got ${localZ}`);
  assert.ok(seat.goalDist() < 1e-9);
  assert.match(seat.spec.description, /interact/);
});

test('seating requires front approach interaction and completes after the animated descent', () => {
  const far = new Session(world, { id: 'far', task: 'sit:armchair_blue', seed: 0 });
  const rejected = far.step([0, 0, 0, 1]);
  assert.equal(rejected.info.seating, false);
  assert.equal(rejected.info.seat_time, -1);
  assert.match(rejected.info.reason, /front/);
  const obstructed = new Session(world, { id: 'obstructed', task: 'sit:armchair_blue', preview: true });
  obstructed.x = world.byId.get('armchair_blue').x; obstructed.z = world.byId.get('armchair_blue').z;
  assert.equal(obstructed.step([0, 0, 0, 1]).info.seating, false, 'interacting from inside the chair collision does not start sitting');
  const s = new Session(world, { id: 's', task: 'sit:armchair_blue', preview: true });
  const idle = s.step([0, 0, 0, 0]);
  assert.equal(idle.done, false, 'being near the chair alone is not sitting');
  let result = s.step([0, 0, 0, 1]);
  assert.equal(result.done, false);
  assert.equal(result.info.seating, true);
  assert.equal(result.info.seated, false);
  assert.equal(result.observation.state.length, 8);
  assert.ok(Number.isFinite(result.info.pelvis_height));
  assert.equal(result.path[0].length, 6);
  const firstPelvis = result.info.pelvis_height;
  let waits = 0;
  while (!result.done && waits++ < 12) result = s.step([1, 1, 1, 0]);
  assert.equal(result.info.success, true);
  assert.equal(result.info.seated, true);
  assert.equal(result.info.seating, false);
  assert.ok(waits >= 5, `should animate for several steps, got ${waits}`);
  assert.ok(result.info.pelvis_height < firstPelvis, 'pelvis descends toward the seat');
  assert.match(result.info.reason, /seated/);
  assert.equal(result.observation.state.length, 8);
  assert.ok(result.info.joints.left_knee_joint > 1, 'legs remain bent in the seated pose');
  assert.throws(() => s.step([0, 0, 0, 0]), /finished/);
});

test('seat HTTP events include placement and continuous seating path samples', async () => {
  const { server } = createServer();
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (path, body) => fetch(base + path, { method: 'POST', body: JSON.stringify(body) }).then(r => r.json());
  const ac = new AbortController();
  try {
    const created = await post('/sim/sessions', { task: 'sit:armchair_blue', preview: true, model: 'seat-test' });
    assert.equal(created.observation.state.length, 8);
    let result = await post(`/sim/sessions/${created.id}/step`, { action: [0, 0, 0, 1] });
    assert.equal(result.info.seating, true);
    while (!result.done) result = await post(`/sim/sessions/${created.id}/step`, { action: [0, 0, 0, 0] });
    assert.equal(result.info.seated, true);
    const sse = await fetch(base + '/sim/events', { signal: ac.signal });
    const reader = sse.body.getReader();
    let text = '';
    while (!text.includes('"type":"session_finished"')) text += new TextDecoder().decode((await reader.read()).value);
    const events = text.split('\n').filter(line => line.startsWith('data: ')).map(line => JSON.parse(line.slice(6)));
    assert.equal(events[0].seat_placement.groundY, 0.35);
    const steps = events.filter(e => e.type === 'step');
    assert.ok(steps[0].path.every(p => p.length === 6 && p[5] >= 0));
    assert.equal(steps.at(-1).seated, true);
  } finally {
    ac.abort(); server.close(); server.closeAllConnections?.();
  }
});

test('retask interrupts fitted seating and can restart it without stale state', () => {
  const s = new Session(world, { id: 'retask-seat', task: 'sit:armchair_blue', preview: true });
  s.step([0, 0, 0, 1]);
  assert.equal(s.seating, true);
  s.retask('wave');
  assert.equal(s.seating, false);
  assert.equal(s.seatTime, -1);
  assert.equal(s.seatFrom, null);
  assert.equal(s.spec.placement, undefined);
  const wave = s.step([0, 0, 0, 0]);
  assert.equal(wave.info.pose, 'wave');
  assert.equal(wave.info.pelvis_height, null);
  s.retask('sit:armchair_blue');
  s.x = s.spec.target.x; s.z = s.spec.target.z;
  const restart = s.step([0, 0, 0, 1]);
  assert.equal(restart.info.seating, true);
  assert.ok(restart.info.seat_time < 0.2);
});

test('retasking releases a chair claim and preview only supports the fitted armchair', async () => {
  const { server } = createServer();
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = async (path, body) => { const r = await fetch(base + path, { method: 'POST', body: JSON.stringify(body) }); return { status: r.status, data: await r.json() }; };
  try {
    const first = await post('/sim/sessions', { task: 'sit:armchair_blue', preview: true });
    assert.equal(first.status, 200);
    assert.equal((await post('/sim/sessions', { task: 'sit:armchair_blue', preview: true })).status, 400, 'occupied chair does not silently preview another chair');
    assert.equal((await post(`/sim/sessions/${first.data.id}/task`, { task: 'idle' })).status, 200);
    const next = await post('/sim/sessions', { task: 'sit:armchair_blue', preview: true });
    assert.equal(next.status, 200, 'idle task released chair');
    assert.equal((await post('/sim/sessions', { task: 'sit:chair_navy_left', preview: true })).status, 400);
  } finally { server.close(); server.closeAllConnections?.(); }
});

test('generic sit choosing the blue armchair exposes the calibrated interaction contract', () => {
  const s = new Session(world, { id: 'generic-blue', task: 'sit:armchair_blue', preview: true });
  s.retask('sit');
  assert.equal(s.spec.target.id, 'armchair_blue');
  assert.equal(s.spec.fitted, true);
  assert.match(s.spec.description, /interact > 0.5/);
  assert.equal(s.step([0, 0, 0, 0]).done, false);
  assert.equal(s.step([0, 0, 0, 1]).info.seating, true);
});
