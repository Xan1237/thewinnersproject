// Fake "models" (scripted + noise) that drive the sim server over HTTP. For UI development and smoke tests.
// Usage: node tools/demo-traffic.js [--rounds 2] [--fast] [--training] [--task fetch_coffee]
const base = process.env.SIM_URL ?? 'http://127.0.0.1:8818';
const args = process.argv.slice(2);
const rounds = Number(args[args.indexOf('--rounds') + 1]) || 2;
const fast = args.includes('--fast');
const sleep = ms => new Promise(r => setTimeout(r, fast ? 5 : ms));
const post = async (p, b) => {
  const r = await fetch(base + p, { method: 'POST', body: JSON.stringify(b) });
  const j = await r.json();
  if (!r.ok) throw new Error(`${p}: ${j.error}`);
  return j;
};

const MODELS = [
  { name: 'qwen-27b', noise: 0.25, latency: 350, skill: 0.9 },
  { name: 'luna', noise: 0.15, latency: 800, skill: 1.0 },
  { name: 'gpt-fallback', noise: 0.4, latency: 1100, skill: 0.8 },
];
const only = args.includes('--task') ? args[args.indexOf('--task') + 1] : null;
const TASKS = only ? [only] : ['goto:armchair_blue', 'goto:lectern', 'fetch_coffee', 'goto:plant_tall'];

function decide(obs, task, m) {
  const [, , , gf, gr, dist, holding] = obs.state;
  const ang = Math.atan2(-gr, gf); // signed angle to the target; positive = target to the left
  const noisy = v => v + (Math.random() - 0.5) * m.noise;
  let turn = Math.max(-1, Math.min(1, -ang * 1.5));
  turn = Math.max(-1, Math.min(1, noisy(turn) * m.skill));
  const fwd = Math.abs(ang) < 0.6 ? Math.max(0, Math.min(1, noisy(1))) : 0;
  const grab = task === 'fetch_coffee' && !holding && dist < 0.85 ? 1 : 0;
  let label = 'hold position';
  if (grab) label = 'grab the coffee cup with the right hand';
  else if (Math.abs(ang) >= 0.6) label = ang > 0 ? 'turn left toward the target' : 'turn right toward the target';
  else if (fwd > 0.2) label = dist > 3 ? 'walk toward the target' : 'approach the target slowly';
  return { action: [fwd, 0, -turn, grab], label };
}

async function run(model, task, seed) {
  const s = await post('/sim/sessions', { task, seed, model: model.name, max_steps: 100 });
  let obs = s.observation, done = false;
  while (!done) {
    await sleep(model.latency * (0.6 + Math.random() * 0.8));
    const d = decide(obs, task, model);
    const r = await post(`/sim/sessions/${s.id}/step`, { action: d.action, label: d.label, model: model.name });
    obs = r.observation; done = r.done;
  }
  await fetch(`${base}/sim/sessions/${s.id}`, { method: 'DELETE' });
}

async function fakeTraining() {
  await post('/sim/training', { phase: 'start', episodes: 14, samples: 1200 });
  let loss = 0.9;
  for (let e = 1; e <= 40; e++) {
    loss *= 0.93 + Math.random() * 0.03;
    await post('/sim/training', { phase: 'epoch', epoch: e, loss, val_loss: loss * (1.1 + Math.random() * 0.1) });
    await sleep(150);
    if (e % 10 === 0) await post('/sim/training', { phase: 'eval', success_rate: Math.min(0.95, 0.2 + e * 0.018), n: 20 });
  }
  await post('/sim/training', { phase: 'done', checkpoint: 'checkpoints/bc.pt' });
}

await post('/sim/reset', {});
for (let r = 0; r < rounds; r++) {
  await Promise.all(MODELS.map((m, i) => run(m, TASKS[(r + i) % TASKS.length], r * 10 + i)));
}
if (args.includes('--training')) await fakeTraining();
console.log('demo traffic finished');
