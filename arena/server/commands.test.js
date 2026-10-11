import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Session, World } from './core.js';
import { parseWithRules, interpret } from './commands.js';
import { createServer } from './server.js';

const here = dirname(fileURLToPath(import.meta.url));
const world = new World(JSON.parse(readFileSync(join(here, '..', 'public', 'room', 'manifest.json'), 'utf8')));

function drive(s, maxTicks = 80) {
  for (let i = 0; i < maxTicks && !s.done; i++) {
    const g = s.goalLocal();
    const ang = Math.atan2(-g.right, g.forward);
    s.step([Math.abs(ang) < 0.6 ? 1 : 0, 0, Math.max(-1, Math.min(1, ang * 1.5)), 0]);
  }
}

test('every chair has a facing and a heading', () => {
  assert.equal(world.seats.length, 34);
  const arm = world.seats.find(s => s.id === 'armchair_blue');
  assert.ok(Math.abs(arm.heading - ((155 + 180) * Math.PI / 180 - 2 * Math.PI)) < 1e-9);
});

test('sit: robots take different chairs and end up seated on them', () => {
  const sessions = new Map();
  world.peers = () => sessions.values();
  const claimed = () => new Set([...sessions.values()].map(s => s.spec.target.id));
  for (let i = 0; i < 4; i++) sessions.set(`s${i}`, new Session(world, { id: `s${i}`, task: 'sit:armchair_blue', seed: i + 1, claimed: claimed() }));
  const ids = [...sessions.values()].map(s => s.spec.target.id);
  assert.equal(new Set(ids).size, 4, `distinct seats: ${ids}`);
  assert.equal(ids[0], 'armchair_blue');
  for (const s of sessions.values()) drive(s);
  for (const s of sessions.values()) { assert.ok(s.success, `${s.id} seated (${s.reason})`); assert.equal(s.seated, true); }
  world.peers = () => [];
});

test('robots do not walk through each other', () => {
  const a = new Session(world, { id: 'a', task: 'goto:lectern', seed: 3 });
  const b = new Session(world, { id: 'b', task: 'goto:lectern', seed: 4 });
  world.peers = () => [a, b];
  b.x = a.x + 0.5; b.z = a.z; a.heading = Math.PI / 2; // a faces +X toward b
  for (let i = 0; i < 4; i++) a.step([1, 0, 0, 0]);
  assert.ok(Math.hypot(a.x - b.x, a.z - b.z) >= 0.33, 'kept the body distance');
  world.peers = () => [];
});

test('retask keeps the position, emotes finish in one step', () => {
  const s = new Session(world, { id: 'r', task: 'idle', seed: 5 });
  const [x, z] = [s.x, s.z];
  s.retask('jump');
  assert.equal(s.spec.kind, 'emote');
  const r = s.step([0, 0, 0, 0]);
  assert.ok(r.done && r.info.success && r.info.pose === 'jump');
  assert.equal(s.x, x); assert.equal(s.z, z);
  s.retask('goto:lectern');
  assert.equal(s.done, false);
});

test('spoken orders map to tasks (rule fallback, no API key)', async () => {
  assert.equal(parseWithRules('everybody sit down please', world).task, 'sit');
  assert.equal(parseWithRules('go to the blue armchair', world).task, 'goto:armchair_blue');
  assert.equal(parseWithRules('sit on the armchair blue', world).task, 'sit:armchair_blue');
  assert.equal(parseWithRules('bring me the coffee', world).task, 'fetch_coffee');
  assert.equal(parseWithRules('everyone jump', world).task, 'jump');
  assert.equal(parseWithRules('blah', world), null);
  assert.equal((await interpret('wave at them', world, '')).task, 'wave');
});

test('HTTP: command + task endpoints', async () => {
  const { server } = createServer({ dataDir: mkdtempSync(join(tmpdir(), 'arena-data-')) });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (p, b) => fetch(base + p, { method: 'POST', body: JSON.stringify(b) }).then(r => r.json());
  try {
    assert.equal((await (await fetch(`${base}/sim/command`)).json()).id, 0);
    const s1 = await post('/sim/sessions', { task: 'idle', seed: 1, model: 'a' });
    const s2 = await post('/sim/sessions', { task: 'idle', seed: 2, model: 'b' });
    const cmd = await post('/sim/command', { task: 'sit', say: 'sit' });
    assert.equal(cmd.id, 1);
    const t1 = await post(`/sim/sessions/${s1.id}/task`, { task: cmd.task });
    const t2 = await post(`/sim/sessions/${s2.id}/task`, { task: cmd.task });
    assert.notEqual(t1.task_description, t2.task_description, 'different chairs');
    const bad = await fetch(`${base}/sim/sessions/${s1.id}/task`, { method: 'POST', body: JSON.stringify({ task: 'fly' }) });
    assert.equal(bad.status, 400);
  } finally { server.close(); }
});

import { transcribe } from './voice.js';

test('transcribe sends base64 wav to the MAI model and returns the text', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, body: JSON.parse(init.body), auth: init.headers.authorization }; return { ok: true, json: async () => ({ text: ' Everybody sit down. ', usage: { cost: 0.0003 } }) }; };
  const r = await transcribe(Buffer.from('RIFFxxxx'), 'audio/wav', { apiKey: 'k', fetchImpl });
  assert.equal(r.text, 'Everybody sit down.');
  assert.equal(seen.body.model, 'microsoft/mai-transcribe-2');
  assert.equal(seen.body.input_audio.format, 'wav');
  assert.equal(Buffer.from(seen.body.input_audio.data, 'base64').toString(), 'RIFFxxxx');
  assert.match(seen.url, /audio\/transcriptions$/);
  await assert.rejects(transcribe(Buffer.from('x'), 'audio/webm', { apiKey: 'k', fetchImpl }), e => e.status === 415);
  await assert.rejects(transcribe(Buffer.from('x'), 'audio/wav', { apiKey: '', fetchImpl }), e => e.status === 501);
});
