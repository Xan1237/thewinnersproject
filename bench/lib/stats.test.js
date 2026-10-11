import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Board, episodeScore, familyOf, interleave, summarize, taskKey } from './stats.js';

const step = (cost, decided = true, latency = 1) => ({ cost_usd: cost, decided, latency_s: latency });

test('episodeScore: failure is 0, success rewards fewer steps', () => {
  assert.equal(episodeScore(false, 10), 0);
  assert.equal(episodeScore(true, 60), 60);
  assert.equal(episodeScore(true, 30), 80);
  assert.equal(episodeScore(true, 0), 100);
});

test('summarize totals cost and averages latency over decided steps only', () => {
  const ep = summarize({
    episode_id: 'a',
    model_id: 'qwen-3.8-27b',
    task: 'Walk to the lectern and stop within 1.2 m of it.',
    success: true,
    steps: [step(0.01, true, 2), step(0.02, false, 99), step(0.03, true, 4)],
  });
  assert.equal(ep.task, 'lectern');
  assert.equal(ep.steps, 3);
  assert.ok(Math.abs(ep.cost - 0.06) < 1e-9);
  assert.equal(ep.latency, 3);
  assert.equal(ep.kind, 'llm');
});

test('classification helpers', () => {
  assert.equal(taskKey('pick up the coffee cup'), 'coffee');
  assert.equal(familyOf('opus-5.5'), 'claude');
  assert.equal(familyOf('scripted-precise'), 'baseline');
  assert.equal(familyOf('gpt-6-astra'), 'gpt');
});

test('Board keeps running means and a trail', () => {
  const board = new Board();
  const mk = (id, success, steps, cost) => ({ id, model: 'm', kind: 'llm', task: 'armchair', success, steps, cost, latency: 1, score: episodeScore(success, steps) });
  board.add(mk('1', true, 30, 0.1));
  const m = board.add(mk('2', false, 60, 0.3));
  assert.equal(m.episodes, 2);
  assert.equal(m.successRate, 0.5);
  assert.equal(m.score, 40);
  assert.ok(Math.abs(m.cost - 0.2) < 1e-9);
  assert.equal(m.trail.length, 2);
  assert.equal(m.byTask.armchair.n, 2);
  assert.equal(board.totals().episodes, 2);
});

test('interleave advances every model at the same relative pace', () => {
  const eps = [
    ...['a1', 'a2', 'a3', 'a4'].map(id => ({ id, model: 'a' })),
    ...['b1', 'b2'].map(id => ({ id, model: 'b' })),
  ];
  const order = interleave(eps).map(e => e.id);
  assert.deepEqual(order, ['a1', 'b1', 'a2', 'a3', 'b2', 'a4']);
});
