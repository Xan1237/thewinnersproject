const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
import { kindOf, KIND_LABEL } from './kinds.js';
import { Insight } from './insight.js';
export const TRAINED_COLOR = '#ffcc33';
export const PALETTE = ['#ff5a8a', '#5ac8ff', '#7a8cff', '#8aff7a', '#c08aff', '#ff9a5a', '#5af0d0', '#e0e0e0'];
const KEY_JOINTS = ['right_shoulder_pitch_joint', 'right_elbow_joint', 'left_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'waist_yaw_joint'];

export class Panel {
  constructor() {
    this.colors = new Map();
    this.models = new Map();   // model -> stats
    this.sessions = new Map(); // session -> {model, task, steps, distance}
    this.feedItems = 0;
    this.manifest = null;
    this.trails = new Map();  // session -> {model, pts, target, status}
    fetch('/sim/manifest').then(r => r.json()).then(m => { this.manifest = m; this.drawMinimap(); }).catch(() => {});
    this.training = { epochs: [], evals: [], status: 'Waiting for training to start…', info: {} };
    this.insight?.reset(); this.insight?.renderPipeline(); this.insight?.renderH2H();
    document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => this.tab(b.dataset.tab));
    this.insight = new Insight(this);
  }

  tab(name) {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('selected', b.dataset.tab === name));
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('selected', t.id === `tab-${name}`));
    if (name === 'training') { this.drawCharts(); requestAnimationFrame(() => this.drawCharts()); }
  }

  color(model) {
    if (!this.colors.has(model)) this.colors.set(model, kindOf(model) === 'trained' ? TRAINED_COLOR : PALETTE[this.colors.size % PALETTE.length]);
    return this.colors.get(model);
  }

  stats(model) {
    if (!this.models.has(model)) this.models.set(model, { runs: 0, wins: 0, steps: 0, winSteps: 0, now: '', decisions: 0, latency: 0, cost: 0, labeled: 0, fails: 0 });
    return this.models.get(model);
  }

  reset() {
    this.trails.clear(); this.models.clear(); this.sessions.clear(); $('feed').innerHTML = ''; this.feedItems = 0;
    this.training = { epochs: [], evals: [], status: 'Waiting for training to start…', info: {} };
    this.insight?.reset(); this.insight?.renderPipeline(); this.insight?.renderH2H();
    this.renderBoard(); this.drawCharts();
  }

  onEvent(ev) {
    queueMicrotask(() => this.insight.onEvent(ev));
    if (ev.type === 'session_started') {
      this.sessions.set(ev.session, { model: ev.model, task: ev.task, steps: 0, distance: null, max: ev.max_steps });
      this.trails.set(ev.session, { model: ev.model, pts: [ev.position], target: ev.target, status: null });
      if (this.trails.size > 60) this.trails.delete(this.trails.keys().next().value);
      this.stats(ev.model).now = ev.task;
      this.addRow(ev.model, `<div class="label">Started <b>${esc(ev.task)}</b> · seed ${esc(ev.seed)}</div><div class="act">${esc(ev.task_description || '')}</div>`, '');
      this.renderBoard();
    } else if (ev.type === 'task_changed') {
      this.sessions.set(ev.session, { model: ev.model, task: ev.task, steps: 0, distance: null, max: ev.max_steps });
      this.trails.set(ev.session, { model: ev.model, pts: [ev.position], target: ev.target, status: null });
      this.stats(ev.model).now = ev.task;
      this.addRow(ev.model, `<div class="label">New order: <b>${esc(ev.task)}</b></div><div class="act">${esc(ev.task_description || '')}</div>`, '');
      this.renderBoard();
    } else if (ev.type === 'step') {
      const s = this.sessions.get(ev.session) || (this.sessions.set(ev.session, { model: ev.model, task: ev.task, steps: 0 }), this.sessions.get(ev.session));
      s.steps = ev.t; s.distance = ev.distance; this.stats(ev.model).labeled++;
      if (ev.detail?.latency_s != null) { const st = this.stats(ev.model); st.decisions++; st.latency += ev.detail.latency_s; st.cost += ev.detail.cost_usd || 0; }
      const tr = this.trails.get(ev.session); if (tr) for (const p of ev.path || [ev.position]) tr.pts.push([p[0], p[1]]);
      this.scheduleMinimap();
      this.addStep(ev);
      if (ev.stage_done) this.addRow(ev.model, `<div class="label">☕ Part ${ev.stage} done: got the coffee</div><div class="act">now looking for a free seat</div>`, 'ok');
      this.renderBoard();
    } else if (ev.type === 'session_finished') {
      const st = this.stats(ev.model);
      st.runs++; st.steps += ev.steps; if (ev.success) { st.wins++; st.winSteps += ev.steps; }
      st.now = '';
      const tr = this.trails.get(ev.session); if (tr) tr.status = ev.success ? 'ok' : 'fail';
      this.scheduleMinimap();
      this.addRow(ev.model, `<div class="label">${ev.success ? '✔ Success' : '✘ Failed'} after ${ev.steps} steps</div><div class="act">${esc(ev.reason)}</div>`, ev.success ? 'ok' : 'fail');
      this.sessions.delete(ev.session);
      this.renderBoard();
    } else if (ev.type === 'training') {
      this.onTraining(ev);
    } else if (ev.type === 'reset') {
      this.reset();
    }
  }

  addStep(ev) {
    const a = ev.action || [];
    const act = `fwd ${(a[0] ?? 0).toFixed(2)} · right ${(a[1] ?? 0).toFixed(2)} · turn ${(a[2] ?? 0).toFixed(2)}${(a[3] ?? 0) > 0.5 ? ' · GRAB' : ''}`;
    const joints = KEY_JOINTS.filter(k => ev.joints && k in ev.joints).map(k => `<span>${esc(k.replace('_joint', ''))}</span><span>${ev.joints[k].toFixed(3)}</span>`).join('');
    this.addRow(ev.model,
      `<div class="label">${esc(ev.label || '(no label)')}</div>${this.detailHtml(ev.detail)}<div class="act">t=${ev.t} · ${act} · dist ${ev.distance?.toFixed?.(2) ?? '?'} m · r ${ev.reward?.toFixed?.(2)}</div>` +
      `<details><summary>motor data</summary><div class="joints">${joints}</div></details>`, '', ev.t);
  }

  /** Extra decision context from a model: the orchestrator's plan and the decision model's odds. */
  detailHtml(d) {
    if (!d) return '';
    const plan = d.replanned && d.plan ? `<div class="plan"><b>${esc(d.orchestrator || 'orchestrator')}</b> plans: ${esc(d.plan)}</div>` : '';
    const top = d.jev?.top ? Object.entries(d.jev.top).map(([k, v]) => `<span class="odds"><i style="width:${Math.round(v * 100)}%"></i><em>${esc(k.replace(/_/g, ' '))} ${Math.round(v * 100)}%</em></span>`).join('') : '';
    return plan + (top ? `<div class="oddsbox"><small>Jev</small>${top}</div>` : '');
  }

  addRow(model, html, cls, t = '') {
    const feed = $('feed');
    const row = document.createElement('div');
    row.className = `row ${cls} ${kindOf(model)}`;
    row.style.setProperty('--c', this.color(model));
    row.innerHTML = `<div class="head"><span class="chip">${esc(model)}</span><span class="t">${t !== '' ? `step ${t}` : ''}</span></div>${html}`;
    feed.prepend(row);
    if (++this.feedItems > 150) { feed.lastChild.remove(); this.feedItems--; }
  }

  renderBoard() {
    const tb = document.querySelector('#board tbody');
    const rows = [...this.models.entries()].sort((a, b) => (b[1].wins / Math.max(b[1].runs, 1)) - (a[1].wins / Math.max(a[1].runs, 1)));
    tb.innerHTML = rows.map(([m, s]) => {
      const live = [...this.sessions.values()].find(x => x.model === m);
      const pct = s.runs ? Math.round(100 * s.wins / s.runs) : 0;
      const kind = kindOf(m), lat = s.decisions ? s.latency / s.decisions : null, cost = s.runs ? s.cost / s.runs : null;
      return `<tr class="${kind}" style="--c:${this.color(m)}"><td><span class="dot"></span>${esc(m)}<br><span class="kind ${kind}">${KIND_LABEL[kind]}</span></td><td>${s.runs}</td><td>${s.runs ? pct + '%' : '–'}<div class="bar"><i style="width:${pct}%"></i></div></td><td>${s.wins ? (s.winSteps / s.wins).toFixed(1) : '–'}</td><td>${lat == null ? '–' : lat < 0.01 ? (lat * 1000).toFixed(1) + ' ms' : lat < 1 ? Math.round(lat * 1000) + ' ms' : lat.toFixed(2) + ' s'}</td><td>${cost == null ? '–' : '$' + cost.toFixed(cost < 0.01 ? 4 : 3)}</td><td>${live ? esc(live.task) + ' · step ' + live.steps : '–'}</td></tr>`;
    }).join('');
    const ok = rows.reduce((n, [, s]) => n + s.wins, 0), all = rows.reduce((n, [, s]) => n + s.runs, 0);
    $('datastat').innerHTML = `<b>${ok}</b> successful episodes recorded as training data out of <b>${all}</b> runs.`;
  }

  onTraining(ev) {
    const t = this.training;
    if (ev.phase === 'collect') t.status = `Step 1 · ${ev.models} frontier models are playing ${ev.total} runs. Every decision is recorded as labeled data.`;
    else if (ev.phase === 'topup') t.status = `Step 1b · scripted teachers add ${ev.total} cheap runs on a second sim.`;
    else if (ev.phase === 'start') { t.epochs = []; t.evals = []; t.status = `Training started · ${ev.episodes ?? '?'} successful episodes (${ev.from_llm ?? '?'} from LLMs, ${ev.from_scripted ?? '?'} scripted), ${ev.samples ?? '?'} samples`; t.info = ev; }
    else if (ev.phase === 'epoch') { t.epochs.push(ev); t.status = `Training · epoch ${ev.epoch} · loss ${Number(ev.loss).toFixed(4)}`; }
    else if (ev.phase === 'eval') { t.evals.push(ev); t.status = `Evaluated learned policy: ${(100 * ev.success_rate).toFixed(0)}% success over ${ev.n} episodes`; }
    else if (ev.phase === 'done') { t.status = `Training finished · ${ev.checkpoint ?? ''}`; t.info = { ...t.info, ...ev }; }
    $('trainstatus').textContent = t.status;
    const teachers = [...this.models.entries()].filter(([m]) => kindOf(m) !== 'trained').map(([m, s]) => `${esc(m)} ${s.runs ? Math.round(100 * s.wins / s.runs) + '%' : '–'}`).join(' · ');
    $('trainstats').innerHTML = `<b>Policy:</b> small MLP, behaviour cloning on successful runs only<br><b>Cost per decision:</b> ~$0 (runs locally)<br><b>Teachers:</b> ${teachers || '–'}`;
    this.drawCharts();
  }

  scheduleMinimap() {
    if (this._mm) return;
    this._mm = requestAnimationFrame(() => { this._mm = 0; if (document.getElementById('tab-training').classList.contains('selected')) this.drawMinimap(); });
  }

  /** Top-down plan of the room with the path of every run; the learned policy is drawn on top. */
  drawMinimap() {
    const c = $('minimap'), g = c.getContext('2d'), W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    const m = this.manifest;
    if (!m) return;
    const b = m.bounds, pad = 8, sc = Math.min((W - 2 * pad) / (b.maxX - b.minX), (H - 2 * pad) / (b.maxZ - b.minZ));
    const ox = (W - sc * (b.maxX - b.minX)) / 2, oy = (H - sc * (b.maxZ - b.minZ)) / 2;
    const X = x => ox + (x - b.minX) * sc, Y = z => oy + (z - b.minZ) * sc;
    g.strokeStyle = '#3a4558'; g.lineWidth = 1.5; g.strokeRect(X(b.minX), Y(b.minZ), (b.maxX - b.minX) * sc, (b.maxZ - b.minZ) * sc);
    for (const o of m.objects) {
      if (o.kind === 'item') continue;
      g.fillStyle = o.kind === 'platform' ? '#1c2433' : '#2b3447';
      g.fillRect(X(o.x - o.sx / 2), Y(o.z - o.sz / 2), Math.max(o.sx * sc, 2), Math.max(o.sz * sc, 2));
    }
    const cup = m.objects.find(o => o.id === 'coffee_cup');
    if (cup) { g.fillStyle = '#c8a27a'; g.beginPath(); g.arc(X(cup.x), Y(cup.z), 3, 0, 7); g.fill(); }
    const learned = t => kindOf(t.model) === 'trained';
    const draw = tr => {
      if (tr.pts.length < 2) return;
      const lrn = learned(tr);
      g.lineWidth = lrn ? 2.4 : 1.2;
      g.strokeStyle = lrn ? (tr.status === 'fail' ? '#ff6b7a' : TRAINED_COLOR) : '#8b96a8';
      g.globalAlpha = lrn ? 0.95 : 0.28;
      g.beginPath(); tr.pts.forEach(([x, z], i) => i ? g.lineTo(X(x), Y(z)) : g.moveTo(X(x), Y(z))); g.stroke();
      g.globalAlpha = 1;
      if (tr.target) { g.strokeStyle = lrn ? '#e8ecf3' : '#5a6578'; g.lineWidth = 1; g.beginPath(); g.arc(X(tr.target[0]), Y(tr.target[1]), 5, 0, 7); g.stroke(); }
      if (lrn) { const [x, z] = tr.pts[tr.pts.length - 1]; g.fillStyle = tr.status === 'fail' ? '#ff6b7a' : TRAINED_COLOR; g.beginPath(); g.arc(X(x), Y(z), 4, 0, 7); g.fill(); }
    };
    const all = [...this.trails.values()];
    all.filter(t => !learned(t)).forEach(draw);
    all.filter(learned).forEach(draw);
  }

  drawCharts() {
    this.drawMinimap();
    const t = this.training;
    this.line($('losschart'), [{ pts: t.epochs.map(e => [e.epoch, e.loss]), color: '#7cc4ff' }, { pts: t.epochs.filter(e => e.val_loss != null).map(e => [e.epoch, e.val_loss]), color: '#ffd35a' }], { xlabel: 'epoch', log: true });
    const teacherRates = [...this.models.entries()].filter(([m, s]) => s.runs && kindOf(m) !== 'trained').map(([, s]) => s.wins / s.runs);
    const base = teacherRates.length ? Math.max(...teacherRates) : null;
    this.line($('evalchart'), [{ pts: t.evals.map((e, i) => [i + 1, e.success_rate]), color: TRAINED_COLOR, dots: true }], { xlabel: 'evaluation', min: 0, max: 1, hline: base == null ? null : { y: base, color: '#8b96a8' } });
  }

  line(canvas, series, { xlabel = '', log = false, min, max, hline } = {}) {
    const g = canvas.getContext('2d'), W = canvas.width, H = canvas.height, pad = { l: 38, r: 10, t: 10, b: 24 };
    g.clearRect(0, 0, W, H);
    const all = series.flatMap(s => s.pts);
    g.strokeStyle = '#252d3b'; g.fillStyle = '#8b96a8'; g.font = '10px system-ui'; g.lineWidth = 1;
    if (!all.length) { g.fillText('no data yet', W / 2 - 24, H / 2); return; }
    const f = v => log ? Math.log10(Math.max(v, 1e-6)) : v;
    let lo = min ?? Math.min(...all.map(p => f(p[1]))), hi = max ?? Math.max(...all.map(p => f(p[1])));
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const xs = all.map(p => p[0]), x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
    const X = x => pad.l + (x - x0) / (x1 - x0) * (W - pad.l - pad.r), Y = v => H - pad.b - (f(v) - lo) / (hi - lo) * (H - pad.t - pad.b);
    for (let i = 0; i <= 4; i++) { const y = pad.t + i * (H - pad.t - pad.b) / 4; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke(); const val = hi - i * (hi - lo) / 4; g.fillText((log ? 10 ** val : val).toFixed(log ? 3 : 2), 2, y + 3); }
    g.fillText(xlabel, W / 2 - 18, H - 6);
    if (hline) { g.strokeStyle = hline.color; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(pad.l, Y(hline.y)); g.lineTo(W - pad.r, Y(hline.y)); g.stroke(); g.setLineDash([]); }
    for (const s of series) {
      if (!s.pts.length) continue;
      g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = 2; g.beginPath();
      s.pts.forEach(([x, y], i) => i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)));
      g.stroke();
      if (s.dots) s.pts.forEach(([x, y]) => { g.beginPath(); g.arc(X(x), Y(y), 3.5, 0, 7); g.fill(); });
    }
  }
}
