import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from './server.js';

test('episodes are recorded per model folder and served as labeled data', async () => {
  const dataDir = mkdtempSync(join(tmpdir(), 'arena-data-'));
  const { server } = createServer({ dataDir });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (p, b) => fetch(base + p, { method: 'POST', body: JSON.stringify(b) }).then(r => r.json());
  try {
    const s = await post('/sim/sessions', { task: 'goto:armchair_blue', seed: 3, model: 'Opus 5.5 (test)', max_steps: 60 });
    let done = false, steps = 0;
    while (!done && steps < 60) {
      const r = await post(`/sim/sessions/${s.id}/step`, { action: [1, 0, 0.3, 0], label: 'forward', detail: { plan: 'go', latency_s: 1.2, cost_usd: 0.01, jev: { top: 'forward' } } });
      done = r.done; steps++;
    }
    await fetch(`${base}/sim/sessions/${s.id}`, { method: 'DELETE' });
    await post('/sim/model', { name: 'learned-bc', sizes: [8, 4, 4], params: 10, layers: [{ w: [[0]], b: [0] }] });

    assert.deepEqual(readdirSync(dataDir).sort(), ['learned-bc', 'opus-5.5-test']);
    const idx = await (await fetch(`${base}/sim/data`)).json();
    const opus = idx.models.find(m => m.folder === 'opus-5.5-test');
    assert.equal(opus.kind, 'frontier');
    assert.equal(opus.episodes, 1);
    assert.equal(opus.files[0].steps, steps);
    assert.equal(idx.models.find(m => m.folder === 'learned-bc').kind, 'trained');

    const jsonl = await (await fetch(`${base}/sim/data/opus-5.5-test.jsonl`)).text();
    const row = JSON.parse(jsonl.split('\n')[0]);
    assert.equal(row.model, 'Opus 5.5 (test)');
    assert.equal(row.state.length, 8);
    assert.equal(row.label, 'forward');
    assert.equal(row.plan, 'go');
    assert.equal(row.cost_usd, 0.01);
    assert.match(row.observation_text, /armchair/i);
    const ep = await (await fetch(`${base}/sim/data/opus-5.5-test/${opus.files[0].file}`)).json();
    assert.equal(ep.steps.length, steps);
    const all = await (await fetch(`${base}/sim/data/all.json`)).json();
    assert.equal(all.models.find(m => m.model === 'Opus 5.5 (test)').episodes.length, 1);
    assert.equal((await fetch(`${base}/sim/data/..%2F..%2Fetc.jsonl`)).status, 404);
    assert.ok(existsSync(join(dataDir, 'learned-bc', 'model.json')));
  } finally { server.close(); }
});

test('a viewer that joins late still gets every live robot (session events outlive step events)', async () => {
  const { server } = createServer({ dataDir: mkdtempSync(join(tmpdir(), 'arena-data-')) });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (p, b) => fetch(base + p, { method: 'POST', body: JSON.stringify(b) }).then(r => r.json());
  try {
    const s = await post('/sim/sessions', { task: 'idle', seed: 1, model: 'late-test', max_steps: 100000 });
    for (let i = 0; i < 2700; i++) await post(`/sim/sessions/${s.id}/step`, { action: [0, 0, 0, 0] });
    const ctrl = new AbortController();
    const res = await fetch(`${base}/sim/events`, { signal: ctrl.signal });
    const reader = res.body.getReader();
    let text = '';
    while (!text.includes('"type":"step"') || text.length < 4000) { const { value, done } = await reader.read(); if (done) break; text += new TextDecoder().decode(value); }
    ctrl.abort();
    assert.match(text, /"type":"session_started"/);
  } finally { server.closeAllConnections?.(); server.close(); }
});
