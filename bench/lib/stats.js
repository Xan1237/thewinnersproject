// Scoring and aggregation for the benchmark. Pure functions, no I/O.

export const MAX_STEPS = 60; // the episode step budget used by every arena config
const TRAIL_LEN = 24;
const RECENT_LEN = 10;

const TASKS = [
  ['armchair', 'Armchair'],
  ['lectern', 'Lectern'],
  ['coffee', 'Coffee'],
];

const NAMES = {
  'fable-5.1': 'Fable 5.1',
  'gpt-6-astra': 'GPT-6 Astra',
  'gpt-6-luna': 'GPT-6 Luna',
  'opus-5.5': 'Opus 5.5',
  'grok-4.7': 'Grok 4.7',
  'gemini-3.1-pro': 'Gemini 3.1 Pro',
  'kimi-k3': 'Kimi K3',
  'qwen-3.8-27b': 'Qwen 3.8 27B',
  'qwen-27b': 'Qwen 27B',
  'claude-haiku': 'Claude Haiku',
  'scripted-precise': 'Scripted · precise',
  'scripted-sloppy': 'Scripted · sloppy',
  'teacher-precise': 'Teacher · precise',
  'teacher-sloppy': 'Teacher · sloppy',
};

export function taskKey(text = '') {
  const t = text.toLowerCase();
  for (const [key] of TASKS) if (t.includes(key)) return key;
  return 'other';
}

export const taskLabel = key => TASKS.find(([k]) => k === key)?.[1] ?? 'Other';

export const kindOf = id => (/^(scripted|teacher)/.test(id) ? 'baseline' : 'llm');

export function familyOf(id) {
  if (kindOf(id) === 'baseline') return 'baseline';
  if (/gpt/.test(id)) return 'gpt';
  if (/gemini/.test(id)) return 'gemini';
  if (/claude|fable|opus|haiku|sonnet/.test(id)) return 'claude';
  if (/grok/.test(id)) return 'grok';
  if (/kimi/.test(id)) return 'kimi';
  if (/qwen/.test(id)) return 'qwen';
  return 'other';
}

export const displayName = id => NAMES[id] ?? id.replace(/[-_]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

/** 0 for a failed episode; 60-100 for a success, more for fewer steps. */
export function episodeScore(success, steps) {
  if (!success) return 0;
  const efficiency = Math.min(1, Math.max(0, 1 - steps / MAX_STEPS));
  return 60 + 40 * efficiency;
}

/** Slim an arena episode record (drops the per-step payload). */
export function summarize(raw, run = '') {
  const steps = raw.steps ?? [];
  const decided = steps.filter(s => s.decided);
  const cost = steps.reduce((sum, s) => sum + (s.cost_usd ?? 0), 0);
  const latency = decided.length ? decided.reduce((sum, s) => sum + (s.latency_s ?? 0), 0) / decided.length : 0;
  const success = Boolean(raw.success);
  return {
    id: raw.episode_id,
    model: raw.model_id,
    kind: kindOf(raw.model_id),
    task: taskKey(raw.task),
    success,
    steps: steps.length,
    cost,
    latency,
    score: episodeScore(success, steps.length),
    run,
  };
}

/** Order episodes so every model advances at the same relative pace (a fair-looking race). */
export function interleave(episodes) {
  const perModel = new Map();
  for (const ep of episodes) {
    if (!perModel.has(ep.model)) perModel.set(ep.model, []);
    perModel.get(ep.model).push(ep);
  }
  const keyed = [];
  let m = 0;
  for (const list of perModel.values()) {
    list.forEach((ep, i) => keyed.push({ ep, key: (i + 0.5) / list.length + m * 1e-6 }));
    m += 1;
  }
  return keyed.sort((a, b) => a.key - b.key).map(k => k.ep);
}

export class Board {
  constructor() {
    this.models = new Map();
  }

  clear() {
    this.models.clear();
  }

  add(ep) {
    let m = this.models.get(ep.model);
    if (!m) {
      m = {
        id: ep.model,
        name: displayName(ep.model),
        family: familyOf(ep.model),
        kind: ep.kind,
        episodes: 0,
        successes: 0,
        scoreSum: 0,
        costSum: 0,
        stepsOnSuccess: 0,
        latencySum: 0,
        byTask: {},
        recent: [],
        trail: [],
      };
      this.models.set(ep.model, m);
    }
    m.episodes += 1;
    m.successes += ep.success ? 1 : 0;
    m.scoreSum += ep.score;
    m.costSum += ep.cost;
    m.stepsOnSuccess += ep.success ? ep.steps : 0;
    m.latencySum += ep.latency;
    const t = (m.byTask[ep.task] ??= { n: 0, successes: 0, scoreSum: 0 });
    t.n += 1;
    t.successes += ep.success ? 1 : 0;
    t.scoreSum += ep.score;
    m.recent.push(ep);
    if (m.recent.length > RECENT_LEN) m.recent.shift();
    m.trail.push({ cost: m.costSum / m.episodes, score: m.scoreSum / m.episodes });
    if (m.trail.length > TRAIL_LEN) m.trail.shift();
    return this.model(ep.model);
  }

  model(id) {
    const m = this.models.get(id);
    if (!m) return null;
    const byTask = {};
    for (const [key, t] of Object.entries(m.byTask)) {
      byTask[key] = { label: taskLabel(key), n: t.n, successRate: t.successes / t.n, score: t.scoreSum / t.n };
    }
    return {
      id: m.id,
      name: m.name,
      family: m.family,
      kind: m.kind,
      episodes: m.episodes,
      successes: m.successes,
      successRate: m.successes / m.episodes,
      score: m.scoreSum / m.episodes,
      cost: m.costSum / m.episodes,
      costTotal: m.costSum,
      steps: m.successes ? m.stepsOnSuccess / m.successes : null,
      latency: m.latencySum / m.episodes,
      byTask,
      recent: m.recent,
      trail: m.trail,
    };
  }

  snapshot() {
    return [...this.models.keys()].map(id => this.model(id));
  }

  totals() {
    let episodes = 0;
    let cost = 0;
    for (const m of this.models.values()) {
      episodes += m.episodes;
      cost += m.costSum;
    }
    return { episodes, models: this.models.size, cost };
  }
}
