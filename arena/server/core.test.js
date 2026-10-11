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
