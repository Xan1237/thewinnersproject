// Records every episode to disk as labeled data: data/<model>/<NNNN>-<task>-<result>.json
import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { kindOf } from '../src/kinds.js';

const slug = s => String(s).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'unknown';

export class Recorder {
  constructor(dir) {
    this.dir = dir;
    this.open = new Map();
    if (dir) mkdirSync(dir, { recursive: true });
  }

  start(s) {
    this.open.set(s.id, { model: s.model, task: s.task, task_description: s.taskText, seed: s.seed, started_at: new Date().toISOString(), steps: [] });
  }

  step(s, row) {
    const ep = this.open.get(s.id);
    if (ep) ep.steps.push(row);
  }

  /** Close the episode of a session; `outcome` is 'done' | 'interrupted' | 'abandoned'. Returns the written file. */
  finish(s, outcome, { success = false, reason = '' } = {}) {
    const ep = this.open.get(s.id);
    this.open.delete(s.id);
    if (!ep || !ep.steps.length || !this.dir) return null;
    const folder = join(this.dir, slug(ep.model));
    mkdirSync(folder, { recursive: true });
    const n = readdirSync(folder).filter(f => /^\d{4}-.*\.json$/.test(f)).length + 1;
    const result = outcome !== 'done' ? 'cut' : success ? 'success' : 'fail';
    const file = `${String(n).padStart(4, '0')}-${slug(ep.task.replace(':', '-'))}-${result}.json`;
    const record = {
      model: ep.model, kind: kindOf(ep.model), task: ep.task, task_description: ep.task_description, seed: ep.seed,
      outcome, success: outcome === 'done' && success, reason, started_at: ep.started_at, steps_count: ep.steps.length,
      total_cost_usd: Number(ep.steps.reduce((a, r) => a + (r.cost_usd ?? 0), 0).toFixed(6)),
      steps: ep.steps,
    };
    writeFileSync(join(folder, file), JSON.stringify(record, null, 1));
    return join(slug(ep.model), file);
  }

  /** Overview of everything on disk, grouped by model folder. */
  index() {
    const models = [];
    if (this.dir && existsSync(this.dir)) {
      for (const m of readdirSync(this.dir).sort()) {
        const folder = join(this.dir, m);
        if (!statSync(folder).isDirectory()) continue;
        const episodes = readdirSync(folder).filter(f => /^\d{4}-.*\.json$/.test(f)).sort().map(file => {
          const r = JSON.parse(readFileSync(join(folder, file), 'utf8'));
          return { file, task: r.task, success: r.success, outcome: r.outcome, steps: r.steps_count, cost_usd: r.total_cost_usd, seed: r.seed };
        });
        const hasModel = existsSync(join(folder, 'model.json'));
        if (!episodes.length && !hasModel) continue;
        const first = episodes[0] ? JSON.parse(readFileSync(join(folder, episodes[0].file), 'utf8')) : null;
        models.push({
          model: first?.model ?? m, folder: m, kind: first?.kind ?? kindOf(m), episodes: episodes.length,
          successes: episodes.filter(e => e.success).length, steps: episodes.reduce((a, e) => a + e.steps, 0), has_weights: hasModel, files: episodes,
        });
      }
    }
    return { dir: this.dir, models, totals: { episodes: models.reduce((a, m) => a + m.episodes, 0), steps: models.reduce((a, m) => a + m.steps, 0), successes: models.reduce((a, m) => a + m.successes, 0) } };
  }

  /** The labeled rows of one model: one per decision, with its episode metadata. */
  rows(folder) {
    const dir = join(this.dir, slug(folder));
    if (!existsSync(dir)) return null;
    const out = [];
    for (const file of readdirSync(dir).filter(f => /^\d{4}-.*\.json$/.test(f)).sort()) {
      const r = JSON.parse(readFileSync(join(dir, file), 'utf8'));
      for (const st of r.steps) out.push({ model: r.model, kind: r.kind, episode: file.replace(/\.json$/, ''), task: r.task, episode_success: r.success, ...st });
    }
    return out;
  }

  episode(folder, file) {
    if (!/^\d{4}-[a-z0-9._-]+\.json$/.test(file)) return null;
    const p = join(this.dir, slug(folder), file);
    return existsSync(p) ? readFileSync(p, 'utf8') : null;
  }
}
export { slug };
