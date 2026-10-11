// Training-tab visuals: the pipeline stepper, frontier-vs-trained head-to-head and the network view.
import { kindOf } from './kinds.js';

const $ = id => document.getElementById(id);
const GOLD = '#ffcc33', BLUE = '#7aa2ff';
const fmtMoney = v => (v === 0 ? '$0.00' : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(3)}`);
const fmtTime = s => (s == null ? '–' : s < 0.01 ? `${(s * 1000).toFixed(1)} ms` : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`);

export class Insight {
  constructor(panel) {
    this.panel = panel;
    this.net = null;          // weights of the trained policy (GET /sim/model)
    this.lastState = null;
    this.lastOut = null;
    this.reset();
    this.renderPipeline(); this.renderH2H(); this.renderDataset();
    fetch('/sim/model').then(r => (r.ok ? r.json() : null)).then(m => { if (m) { this.net = m; this.drawNet(); } }).catch(() => {});
  }

  reset() {
    this.p = { expected: 0, topup: false, topupTotal: 0, stage: 1, collected: 0, kept: 0, samples: 0, epochs: 0, epoch: 0, evalRate: null, evalN: 0, done: false, fromLlm: 0, fromScripted: 0 };
    this.lastState = null; this.lastOut = null;
  }

  onEvent(ev) {
    const p = this.p;
    if (ev.type === 'model') fetch('/sim/model').then(r => r.json()).then(m => { this.net = m; this.drawNet(); });
    if (ev.type === 'training') {
      if (ev.phase === 'collect') Object.assign(p, { stage: 1, expected: ev.total, topup: false });
      else if (ev.phase === 'topup') { p.topup = true; p.topupTotal = ev.total; }
      else if (ev.phase === 'start') Object.assign(p, { stage: 3, collected: ev.collected ?? ev.episodes, kept: ev.episodes, samples: ev.samples ?? 0, epochs: ev.epochs ?? 0, epoch: 0, fromLlm: ev.from_llm ?? 0, fromScripted: ev.from_scripted ?? 0, done: false, evalRate: null });
      else if (ev.phase === 'epoch') { p.stage = 3; p.epoch = ev.epoch; if (!p.epochs) p.epochs = Math.max(p.epochs, ev.epoch); }
      else if (ev.phase === 'eval') { p.stage = 4; p.evalRate = ev.success_rate; p.evalN = ev.n; }
      else if (ev.phase === 'done') { p.stage = 4; p.done = true; }
      this.renderPipeline();
    }
    if (ev.type === 'step' && kindOf(ev.model) === 'trained' && ev.state) { this.lastState = ev.state; this.lastEv = ev; this.drawNet(); }
    if (ev.type === 'step' || ev.type === 'session_finished') { this.scheduleDataset(); this.renderH2H(); if (p.stage === 1) this.renderPipeline(); }
  }

  /* ---------- pipeline stepper ---------- */
  renderPipeline() {
    const p = this.p, models = [...this.panel.models.entries()].filter(([m]) => kindOf(m) !== 'trained');
    const runs = models.reduce((n, [, s]) => n + s.runs, 0), wins = models.reduce((n, [, s]) => n + s.wins, 0);
    const trainPct = p.epochs ? Math.min(100, Math.round(100 * p.epoch / p.epochs)) : 0;
    const st = n => (p.stage > n || (p.done && n === 4) ? 'done' : p.stage === n ? 'active' : '');
    const cards = [
      [1, 'Collect', p.topup ? 'Scripted teachers top up the data' : 'Frontier models drive the robots', p.stage > 1 ? `<b>${p.collected}</b> runs recorded` : `<b>${runs}</b>${p.expected ? ` / ${p.expected}` : ''} runs${p.expected ? `<div class="bar"><i style="width:${Math.min(100, Math.round(100 * runs / p.expected))}%"></i></div>` : ''}`],
      [2, 'Filter', 'Keep only the winning runs', p.stage > 1 ? `<b>${p.kept}</b> of ${p.collected} kept` : `<b>${wins}</b> winners so far`],
      [3, 'Train', 'A small network copies the winners', p.stage >= 3 ? `epoch <b>${p.epoch}</b>${p.epochs ? ` / ${p.epochs}` : ''}<div class="bar"><i style="width:${trainPct}%"></i></div>` : `<b>${p.samples || '–'}</b> samples`],
      [4, 'Deploy', 'The trained model plays on its own', p.evalRate != null ? `<b>${Math.round(100 * p.evalRate)}%</b> success` : 'waiting'],
    ];
    $('pipeline').innerHTML = cards.map(([n, t, d, v]) => `<div class="stage ${st(n)}"><div class="n">${st(n) === 'done' ? '✓' : n}</div><div class="t">${t}</div><div class="d">${d}</div><div class="v">${v}</div></div>`).join('<div class="arrow">›</div>');
  }

  /* ---------- the dataset that is being built ---------- */
  scheduleDataset() {
    if (this._ds) return;
    this._ds = requestAnimationFrame(() => { this._ds = 0; this.renderDataset(); });
  }

  renderDataset() {
    const rows = [...this.panel.models.entries()];
    const wins = rows.reduce((n, [, s]) => n + s.wins, 0), runs = rows.reduce((n, [, s]) => n + s.runs, 0), steps = rows.reduce((n, [, s]) => n + (s.labeled || 0), 0);
    const max = Math.max(4, ...rows.map(([, s]) => s.runs + (s.now ? 1 : 0)));
    const bar = ([m, s]) => {
      const kind = kindOf(m), live = s.now ? 1 : 0, w = n => `${(100 * n / max).toFixed(1)}%`;
      return `<div class="drow ${kind}"><div class="dn">${kind === 'trained' ? '★ ' : ''}${m}</div>
        <div class="dbar"><i class="w" style="width:${w(s.wins)}"></i><i class="f" style="width:${w(s.runs - s.wins)}"></i><i class="l" style="width:${w(live)}"></i></div>
        <div class="dc">${s.wins}/${s.runs}</div></div>`;
    };
    $('dataset').innerHTML = `<div class="dtotals"><div><b>${steps.toLocaleString()}</b><span>labeled decisions</span></div><div><b>${runs}</b><span>runs played</span></div><div><b>${wins}</b><span>successful, kept for training</span></div></div>`
      + (rows.length ? rows.map(bar).join('') : '<div class="mute">Waiting for the first robot to move…</div>')
      + '<div class="dkey"><i class="w"></i> kept (success) <i class="f"></i> thrown away (failed) <i class="l"></i> playing now</div>';
  }

  /* ---------- frontier vs trained ---------- */
  aggregate(kind) {
    const a = { runs: 0, wins: 0, winSteps: 0, decisions: 0, latency: 0, cost: 0, n: 0 };
    for (const [m, s] of this.panel.models) {
      if (kindOf(m) !== kind) continue;
      a.n++; a.runs += s.runs; a.wins += s.wins; a.winSteps += s.winSteps; a.decisions += s.decisions || 0; a.latency += s.latency || 0; a.cost += s.cost || 0;
    }
    return a;
  }

  renderH2H() {
    let f = this.aggregate('frontier'), fLabel = 'FRONTIER MODELS';
    if (!f.n) { f = this.aggregate('scripted'); fLabel = 'SCRIPTED TEACHERS'; }
    const t = this.aggregate('trained');
    const have = t.runs > 0 || t.decisions > 0;
    const fm = { rate: f.runs ? f.wins / f.runs : null, steps: f.wins ? f.winSteps / f.wins : null, lat: f.decisions ? f.latency / f.decisions : null, cost: f.runs ? f.cost / f.runs : null };
    const tm = { rate: t.runs ? t.wins / t.runs : null, steps: t.wins ? t.winSteps / t.wins : null, lat: t.decisions ? t.latency / t.decisions : null, cost: t.decisions || t.runs ? t.cost / Math.max(t.runs, 1) : null };
    const pct = v => (v == null ? '–' : `${Math.round(100 * v)}%`);
    const row = (label, a, b, fmt, bar) => `<div class="h2row"><div class="lab">${label}</div>
      <div class="cell f">${fmt(a)}${bar ? `<div class="bar"><i style="width:${bar(a)}%;background:${BLUE}"></i></div>` : ''}</div>
      <div class="cell t">${have ? fmt(b) : '–'}${bar && have ? `<div class="bar"><i style="width:${bar(b)}%;background:${GOLD}"></i></div>` : ''}</div></div>`;
    const speed = fm.lat && tm.lat ? Math.round(fm.lat / tm.lat) : null;
    const headline = have && speed && fm.cost ? `<div class="headline"><b>${speed.toLocaleString()}×</b> faster decisions · <b>${fmtMoney(fm.cost)}</b> → <b>$0</b> per task</div>`
      : have && speed ? `<div class="headline"><b>${speed.toLocaleString()}×</b> faster decisions · <b>$0</b> per task</div>`
      : have ? `<div class="headline"><b>${fmtTime(tm.lat)}</b> per decision · <b>$0</b> per task · no tokens</div>` : '<div class="headline mute">The trained model has not played yet.</div>';
    $('h2h').innerHTML = `<div class="h2head"><div></div><div class="f">${fLabel}<small>${f.n} models · avg</small></div><div class="t">★ TRAINED MODEL<small>distilled · local</small></div></div>
      ${row('Success rate', fm.rate, tm.rate, pct, v => (v == null ? 0 : Math.round(100 * v)))}
      ${row('Steps to goal', fm.steps, tm.steps, v => (v == null ? '–' : v.toFixed(1)))}
      ${row('Time per decision', fm.lat, tm.lat, fmtTime)}
      ${row('Cost per task', fm.cost, tm.cost, v => (v == null ? '–' : fmtMoney(v)))}
      ${headline}`;
  }

  /* ---------- the network ---------- */
  forward(state) {
    const n = this.net;
    let x = state.map((v, i) => (v - n.x_mean[i]) / (n.x_std[i] || 1));
    const acts = [x];
    n.layers.forEach((L, li) => {
      const y = L.w.map((row, j) => { let s = L.b[j]; for (let i = 0; i < row.length; i++) s += row[i] * x[i]; return s; });
      x = li < n.layers.length - 1 ? y.map(v => Math.max(v, 0)) : y;
      acts.push(x);
    });
    return { acts, out: x.map((v, i) => v * n.y_std[i] + n.y_mean[i]) };
  }

  drawNet() {
    const cv = $('netviz');
    if (!cv) return;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    if (!this.net) { g.fillStyle = '#8b96a8'; g.font = '12px system-ui'; g.fillText('The network appears here once training has finished.', 24, H / 2); return; }
    const n = this.net, state = this.lastState ?? n.x_mean;
    const { acts, out } = this.forward(state);
    this.lastOut = out;
    const shown = n.sizes.map(s => Math.min(s, 10));
    const pick = (li, k) => (shown[li] === 1 ? 0 : Math.round(k * (n.sizes[li] - 1) / (shown[li] - 1)));
    const xs = n.sizes.map((_, i) => 76 + i * (W - 76 - 60) / (n.sizes.length - 1));
    const ys = li => Array.from({ length: shown[li] }, (_, k) => 20 + (shown[li] === 1 ? (H - 40) / 2 : k * (H - 40) / (shown[li] - 1)));
    const norm = acts.map(a => Math.max(...a.map(Math.abs), 1e-6));
    for (let li = 0; li < n.layers.length; li++) {          // edges
      const L = n.layers[li], ya = ys(li), yb = ys(li + 1);
      for (let b = 0; b < shown[li + 1]; b++) for (let a = 0; a < shown[li]; a++) {
        const ia = pick(li, a), ib = pick(li + 1, b);
        const flow = Math.abs(L.w[ib][ia] * acts[li][ia]) / norm[li];
        if (flow < 0.015) continue;
        g.strokeStyle = L.w[ib][ia] * acts[li][ia] >= 0 ? `rgba(255,204,51,${Math.min(flow * 1.6, 0.8)})` : `rgba(122,162,255,${Math.min(flow * 1.6, 0.8)})`;
        g.lineWidth = 0.6 + flow * 1.6;
        g.beginPath(); g.moveTo(xs[li], ya[a]); g.lineTo(xs[li + 1], yb[b]); g.stroke();
      }
    }
    for (let li = 0; li < n.sizes.length; li++) {           // neurons
      ys(li).forEach((y, k) => {
        const v = Math.abs(acts[li][pick(li, k)]) / norm[li];
        g.fillStyle = `rgba(255,204,51,${0.12 + 0.88 * Math.min(v, 1)})`;
        g.strokeStyle = '#3a4558'; g.lineWidth = 1;
        g.beginPath(); g.arc(xs[li], y, li === 0 || li === n.sizes.length - 1 ? 6 : 4.5, 0, 7); g.fill(); g.stroke();
      });
    }
    g.fillStyle = '#8b96a8'; g.font = '10px system-ui';
    const inNames = n.inputs?.length ? n.inputs : ['x', 'z', 'heading', 'goal fwd', 'goal right', 'goal dist', 'holding', 'time left'];
    ys(0).forEach((y, k) => g.fillText(String(inNames[pick(0, k)] ?? '').replace(/_/g, ' ').slice(0, 9), 2, y + 3));
    const outNames = ['forward', 'right', 'turn', 'grab'];
    ys(n.sizes.length - 1).forEach((y, k) => g.fillText(outNames[k] ?? '', W - 40, y + 3));
    g.fillStyle = '#c4cede'; g.font = '11px system-ui';
    g.fillText(`${n.sizes.join(' → ')} · ${n.params.toLocaleString()} weights`, 8, H - 4);
    this.renderReadout(out);
  }

  renderReadout(out) {
    const names = ['forward', 'right', 'turn', 'grab'], ev = this.lastEv;
    const bars = out.slice(0, 4).map((v, i) => { const c = Math.max(-1, Math.min(1, v)); return `<div class="obar"><span>${names[i]}</span><div><i style="left:${50 + Math.min(c, 0) * 50}%;width:${Math.abs(c) * 50}%"></i></div><em>${v.toFixed(2)}</em></div>`; }).join('');
    const lat = ev?.detail?.latency_s;
    $('netread').innerHTML = `${bars}<div class="meta">${ev ? `<b>${ev.label || ''}</b> · ` : ''}0 tokens · ${lat != null ? fmtTime(lat) : '–'} · $0.00</div>`;
  }
}
