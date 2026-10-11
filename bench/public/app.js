'use strict';

const SCORE_MIN = 40;
const SCORE_MAX = 100;
const PRICE_MIN = 0.001;
const PRICE_MAX = 0.25;
const PRICE_TICKS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
const MATURE_EPISODES = 6;
const PAD = { top: 24, right: 36, bottom: 56, left: 56 };
const FREE_GAP = 80; // pixels reserved on the left of the cost axis for the "free" column
const svgNS = 'http://www.w3.org/2000/svg';
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const plotRoot = document.getElementById('plot');
const boardBody = document.getElementById('board-body');
const dialog = document.getElementById('assessment');
const dialogContent = document.getElementById('assessment-content');
const statusEl = document.getElementById('status');

const S = {
  view: 'plot',
  models: new Map(), // id -> model snapshot from the server
  nodes: new Map(), // id -> plot node (dom + animated position)
  mode: 'replay',
  loops: 0,
  progress: { delivered: 0, planned: 0, queued: 0 },
  totals: { episodes: 0, models: 0, cost: 0 },
  running: new Set(),
  training: null,
  arena: false,
  connected: false,
  size: { width: 0, height: 0 },
  chart: null,
  boardDirty: true,
};

// ---------- helpers ----------

const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const money = cost => (cost === 0 ? 'Free' : cost < 0.01 ? `$${cost.toFixed(4)}` : `$${cost.toFixed(3)}`);
const priceTick = value => `$${value}`;
const pct = value => `${Math.round(value * 100)}%`;
const family = id => `var(--family-${S.models.get(id)?.family ?? 'other'})`;

function svg(name, attrs = {}, children = []) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, String(value));
  children.forEach(child => node.appendChild(child));
  return node;
}

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function next() {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function blobPath(seed, radius = 6.8) {
  const rand = mulberry32(hashString(seed));
  const count = 8;
  const points = [];
  for (let i = 0; i < count; i += 1) {
    const angle = (i / count) * Math.PI * 2 + (rand() - 0.5) * 0.22;
    const r = radius * (0.84 + rand() * 0.32);
    points.push({ x: Math.cos(angle) * r, y: Math.sin(angle) * r });
  }
  const parts = [];
  for (let i = 0; i < count; i += 1) {
    const p0 = points[(i - 1 + count) % count];
    const p1 = points[i];
    const p2 = points[(i + 1) % count];
    const p3 = points[(i + 2) % count];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    if (i === 0) parts.push(`M ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`);
    parts.push(`C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`);
  }
  parts.push('Z');
  return parts.join(' ');
}

// ---------- plot geometry ----------

function measurePlot() {
  return { width: plotRoot.clientWidth, height: Math.max(Math.round(window.innerHeight * 0.58), 520) };
}

function geometry() {
  const { width, height } = S.size;
  const innerH = height - PAD.top - PAD.bottom;
  const freeX = PAD.left + 28;
  const xMin = PAD.left + FREE_GAP;
  const xMax = width - PAD.right;
  const logMin = Math.log(PRICE_MIN);
  const logMax = Math.log(PRICE_MAX);
  const xOf = cost => {
    if (cost <= 0) return freeX;
    const t = (Math.log(Math.min(PRICE_MAX, Math.max(PRICE_MIN, cost))) - logMin) / (logMax - logMin);
    return xMin + t * (xMax - xMin);
  };
  const yOf = score => PAD.top + ((SCORE_MAX - Math.min(SCORE_MAX, Math.max(SCORE_MIN, score))) / (SCORE_MAX - SCORE_MIN)) * innerH;
  return { xOf, yOf, freeX, xMin, xMax };
}

const measureCtx = document.createElement('canvas').getContext('2d');

function placeLabels() {
  const { xOf, yOf } = geometry();
  const { width, height } = S.size;
  measureCtx.font = `12px ${getComputedStyle(plotRoot).fontFamily}`;
  const points = [...S.models.values()].map(m => ({ m, x: xOf(m.cost), y: yOf(m.score), w: measureCtx.measureText(m.name).width + 6 }));
  const overlap = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  const boxes = [];
  const dotBoxes = points.map(p => ({ left: p.x - 10, right: p.x + 10, top: p.y - 10, bottom: p.y + 10 }));
  for (const p of [...points].sort((a, b) => b.w - a.w)) {
    const node = S.nodes.get(p.m.id);
    const candidates = [];
    for (const dy of [0, -20, 20, -40, 40, -60, 60]) {
      for (const anchor of ['end', 'start']) {
        const edge = p.x + (anchor === 'end' ? -13 : 13);
        const left = anchor === 'end' ? edge - p.w : edge;
        const cy = p.y + dy;
        const box = { left, right: left + p.w, top: cy - 9, bottom: cy + 9 };
        if (box.left < PAD.left + 5 || box.right > width - 5 || box.top < PAD.top || box.bottom > height - PAD.bottom - 5) continue;
        const labelOverlap = boxes.reduce((sum, other) => sum + overlap(box, other), 0);
        const dotOverlap = dotBoxes.reduce((sum, other) => sum + overlap(box, other), 0);
        const sticky = node && node.choice && node.choice.dy === dy && node.choice.anchor === anchor ? -25 : 0;
        candidates.push({ left, cy, dy, anchor, box, penalty: labelOverlap * 100 + dotOverlap * 50 + Math.abs(dy) + (anchor === 'start' ? 1 : 0) + sticky });
      }
    }
    const best = candidates.sort((a, b) => a.penalty - b.penalty)[0];
    if (!node) continue;
    node.tx = p.x;
    node.ty = p.y;
    if (best) {
      boxes.push(best.box);
      node.choice = { dy: best.dy, anchor: best.anchor };
      node.ltx = best.left;
      node.lty = best.cy;
    } else {
      node.ltx = p.x + 13;
      node.lty = p.y;
    }
    node.lw = p.w;
    node.side = node.choice?.anchor ?? 'start';
  }
}

// ---------- plot nodes ----------

function createNode(m) {
  const color = `var(--family-${m.family})`;
  const ring = svg('circle', { class: 'plot-ring', r: 9, stroke: color });
  const runRing = svg('circle', { class: 'plot-ring', r: 9, stroke: color });
  const dot = svg('path', { class: 'plot-dot', d: blobPath(m.id), fill: color });
  const mark = svg('g', {}, [svg('circle', { class: 'plot-hit', r: 18 }), runRing, ring, dot]);
  const trail = svg('path', { class: 'plot-trail', stroke: color });
  const lead = svg('line', { class: 'plot-lead', stroke: color });
  const label = svg('text', { class: `plot-label${m.kind === 'baseline' ? ' is-baseline' : ''}`, fill: color, 'dominant-baseline': 'central' });
  label.textContent = m.name;
  const group = svg('g', { class: 'plot-row', role: 'button', tabindex: '0', 'data-id': m.id }, [trail, lead, mark, label]);
  const node = { id: m.id, group, mark, ring, runRing, trail, lead, label, x: null, y: null, tx: 0, ty: 0, lx: null, ly: null, ltx: 0, lty: 0, lw: 40, side: 'start', choice: null, phase: Math.random() * 6.28 };

  group.addEventListener('click', event => {
    event.stopPropagation();
    selectModel(m.id);
  });
  group.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectModel(m.id);
    }
  });
  group.addEventListener('pointerenter', event => {
    group.classList.add('is-hot');
    showTip(m.id, event);
  });
  group.addEventListener('pointermove', event => showTip(m.id, event));
  group.addEventListener('pointerleave', () => {
    group.classList.remove('is-hot');
    S.chart?.tip.classList.remove('is-on');
  });
  return node;
}

function syncNodes() {
  for (const [id, node] of S.nodes) {
    if (!S.models.has(id)) {
      node.group.remove();
      S.nodes.delete(id);
    }
  }
  for (const m of S.models.values()) {
    let node = S.nodes.get(m.id);
    if (!node) {
      node = createNode(m);
      S.nodes.set(m.id, node);
      S.chart?.rows.appendChild(node.group);
    }
    node.mark.style.opacity = String(0.55 + 0.45 * Math.min(1, m.episodes / MATURE_EPISODES));
    node.runRing.classList.toggle('is-run', S.running.has(m.id));
  }
  if (S.chart) placeLabels();
}

function pingNode(id, success) {
  const node = S.nodes.get(id);
  if (!node || reduceMotion) return;
  node.ring.setAttribute('stroke', success ? `var(--family-${S.models.get(id)?.family ?? 'other'})` : 'var(--fail)');
  node.ring.classList.remove('is-ping');
  void node.ring.getBoundingClientRect();
  node.ring.classList.add('is-ping');
}

function showTip(id, event) {
  const m = S.models.get(id);
  if (!m || !S.chart) return;
  const { tip } = S.chart;
  tip.querySelector('.tip-name').textContent = m.name;
  tip.querySelector('.tip-meta').textContent = `${money(m.cost)} / episode  ·  score ${m.score.toFixed(1)}  ·  ${m.episodes} episodes`;
  const bounds = plotRoot.getBoundingClientRect();
  tip.style.left = `${event.clientX - bounds.left}px`;
  tip.style.top = `${event.clientY - bounds.top}px`;
  tip.classList.add('is-on');
}

function buildPlot(intro) {
  const { width, height } = S.size;
  const { xOf, yOf, freeX, xMin } = geometry();
  plotRoot.querySelector('svg')?.remove();
  plotRoot.querySelector('.tip')?.remove();

  const axisY = height - PAD.bottom - PAD.top;
  const axisX = width - PAD.right + 20 - PAD.left;
  const board = svg('svg', { class: 'plot-board', role: 'img', 'aria-label': 'Scatter plot of model score by cost per episode', viewBox: `0 0 ${width} ${height}`, width: '100%', height: String(height) });
  board.appendChild(svg('defs', {}, [
    svg('filter', { id: 'dot-hover-glow', x: '-60%', y: '-60%', width: '220%', height: '220%' }, [
      svg('feGaussianBlur', { in: 'SourceGraphic', stdDeviation: '3.4', result: 'blur' }),
      svg('feComponentTransfer', { in: 'blur', result: 'dimBlur' }, [svg('feFuncA', { type: 'linear', slope: '0.55' })]),
      svg('feMerge', {}, [svg('feMergeNode', { in: 'dimBlur' }), svg('feMergeNode', { in: 'SourceGraphic' })]),
    ]),
    svg('linearGradient', { id: 'axis-fade-y', gradientUnits: 'userSpaceOnUse', x1: PAD.left, y1: height - PAD.bottom, x2: PAD.left, y2: PAD.top }, [
      svg('stop', { offset: '0%', 'stop-color': 'var(--bg)' }),
      svg('stop', { offset: '12%', 'stop-color': 'var(--axis)' }),
      svg('stop', { offset: '88%', 'stop-color': 'var(--axis)' }),
      svg('stop', { offset: '100%', 'stop-color': 'var(--bg)' }),
    ]),
    svg('linearGradient', { id: 'axis-fade-x', gradientUnits: 'userSpaceOnUse', x1: PAD.left, y1: height - PAD.bottom, x2: width - PAD.right + 20, y2: height - PAD.bottom }, [
      svg('stop', { offset: '0%', 'stop-color': 'var(--bg)' }),
      svg('stop', { offset: '8%', 'stop-color': 'var(--axis)' }),
      svg('stop', { offset: '92%', 'stop-color': 'var(--axis)' }),
      svg('stop', { offset: '100%', 'stop-color': 'var(--bg)' }),
    ]),
  ]));

  board.appendChild(svg('line', { class: 'plot-axis', style: `--len:${axisY}`, x1: PAD.left, y1: height - PAD.bottom, x2: PAD.left, y2: PAD.top, stroke: 'url(#axis-fade-y)', 'stroke-width': '1.25' }));
  board.appendChild(svg('line', { class: 'plot-axis', style: `--len:${axisX}`, x1: PAD.left, y1: height - PAD.bottom, x2: width - PAD.right + 20, y2: height - PAD.bottom, stroke: 'url(#axis-fade-x)', 'stroke-width': '1.25' }));
  board.appendChild(svg('line', { class: 'plot-divider', x1: xMin - 28, y1: PAD.top + 8, x2: xMin - 28, y2: height - PAD.bottom }));

  for (let value = SCORE_MIN; value <= SCORE_MAX; value += 10) {
    const label = svg('text', { class: 'plot-tick', x: PAD.left - 10, y: yOf(value), 'text-anchor': 'end', 'dominant-baseline': 'central' });
    label.textContent = String(value);
    board.appendChild(label);
  }
  const free = svg('text', { class: 'plot-tick', x: freeX, y: height - PAD.bottom + 18, 'text-anchor': 'middle' });
  free.textContent = 'Free';
  board.appendChild(free);
  for (const value of PRICE_TICKS) {
    const label = svg('text', { class: 'plot-tick', x: xOf(value), y: height - PAD.bottom + 18, 'text-anchor': 'middle' });
    label.textContent = priceTick(value);
    board.appendChild(label);
  }
  const caption = svg('text', { class: 'plot-caption', x: (PAD.left + width - PAD.right) / 2, y: height - 10, 'text-anchor': 'middle' });
  caption.textContent = 'Cost, $ / episode';
  board.appendChild(caption);

  const rows = svg('g', {});
  board.appendChild(rows);
  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.innerHTML = '<div class="tip-name"></div><div class="tip-meta"></div>';
  plotRoot.appendChild(board);
  plotRoot.appendChild(tip);
  S.chart = { board, rows, tip };

  for (const node of S.nodes.values()) {
    rows.appendChild(node.group);
    node.x = null; // snap to the new geometry
  }
  plotRoot.classList.toggle('is-intro', Boolean(intro) && !reduceMotion);
  if (intro) window.setTimeout(() => plotRoot.classList.remove('is-intro'), 2400);
  placeLabels();
}

function ensurePlot(intro = false) {
  const size = measurePlot();
  if (size.width < 80) return;
  if (S.chart && Math.abs(S.size.width - size.width) < 8 && Math.abs(S.size.height - size.height) < 8) return;
  S.size = size;
  buildPlot(intro);
}

// ---------- animation loop ----------

let lastFrame = 0;

function frame(now) {
  requestAnimationFrame(frame);
  if (S.view !== 'plot' || !S.chart || document.hidden) {
    lastFrame = now;
    return;
  }
  const dt = Math.min(0.05, (now - lastFrame) / 1000 || 0.016);
  lastFrame = now;
  const k = reduceMotion ? 1 : 1 - Math.exp(-dt * 4.5);
  const { xOf, yOf } = geometry();
  for (const node of S.nodes.values()) {
    const m = S.models.get(node.id);
    if (!m) continue;
    if (node.x === null) {
      node.x = node.tx;
      node.y = node.ty;
      node.lx = node.ltx;
      node.ly = node.lty;
    }
    node.x += (node.tx - node.x) * k;
    node.y += (node.ty - node.y) * k;
    node.lx += (node.ltx - node.lx) * k;
    node.ly += (node.lty - node.ly) * k;
    const wx = reduceMotion ? 0 : Math.sin(now / 1700 + node.phase) * 1.2;
    const wy = reduceMotion ? 0 : Math.cos(now / 2100 + node.phase) * 1.2;
    const px = node.x + wx;
    const py = node.y + wy;
    node.mark.setAttribute('transform', `translate(${px.toFixed(2)} ${py.toFixed(2)})`);
    node.label.setAttribute('x', node.lx.toFixed(2));
    node.label.setAttribute('y', node.ly.toFixed(2));

    const far = Math.abs(node.ly - py) > 6;
    node.lead.style.display = far ? '' : 'none';
    if (far) {
      const edge = node.side === 'end' ? node.lx + node.lw - 2 : node.lx - 4;
      node.lead.setAttribute('x1', px.toFixed(2));
      node.lead.setAttribute('y1', py.toFixed(2));
      node.lead.setAttribute('x2', edge.toFixed(2));
      node.lead.setAttribute('y2', node.ly.toFixed(2));
    }

    const trail = (m.kind === 'baseline' ? [] : m.trail.slice(-11, -1)).map(p => `${xOf(p.cost).toFixed(1)} ${yOf(p.score).toFixed(1)}`);
    node.trail.setAttribute('d', trail.length ? `M ${trail.join(' L ')} L ${px.toFixed(1)} ${py.toFixed(1)}` : '');
  }
}

// ---------- leaderboard ----------

const ranked = () => [...S.models.values()].sort((a, b) => b.score - a.score);

function rowHtml(m) {
  const sub = m.kind === 'baseline' ? 'baseline · no API cost' : `${m.steps == null ? '–' : m.steps.toFixed(1)} steps per success · ${m.latency.toFixed(2)}s / decision`;
  return `<td><button type="button" class="model-link" data-model="${escapeHtml(m.id)}" aria-haspopup="dialog"><span class="family-dot" style="--model-color:var(--family-${escapeHtml(m.family)})"></span>${escapeHtml(m.name)}<small>${escapeHtml(sub)}</small></button></td>
    <td data-k="score">${m.score.toFixed(1)}</td><td data-k="success">${pct(m.successRate)}</td><td class="cost-cell" data-k="cost">${money(m.cost)}</td><td data-k="n">${m.episodes}</td>`;
}

function renderBoard() {
  if (S.view !== 'board') {
    S.boardDirty = true;
    return;
  }
  S.boardDirty = false;
  const order = ranked();
  const rows = new Map([...boardBody.children].map(tr => [tr.dataset.id, tr]));
  const before = new Map([...rows].map(([id, tr]) => [id, tr.getBoundingClientRect().top]));
  for (const m of order) {
    let tr = rows.get(m.id);
    if (!tr) {
      tr = document.createElement('tr');
      tr.dataset.id = m.id;
      tr.innerHTML = rowHtml(m);
      tr.addEventListener('click', () => selectModel(m.id));
    } else {
      const prev = [...tr.querySelectorAll('[data-k]')].map(td => td.textContent);
      tr.innerHTML = rowHtml(m);
      tr.querySelectorAll('[data-k]').forEach((td, i) => {
        if (td.textContent !== prev[i]) td.classList.add('is-bumped');
      });
    }
    boardBody.appendChild(tr);
    rows.delete(m.id);
  }
  for (const tr of rows.values()) tr.remove();
  if (reduceMotion) return;
  for (const tr of boardBody.children) {
    const was = before.get(tr.dataset.id);
    if (was == null) continue;
    const dy = was - tr.getBoundingClientRect().top;
    if (Math.abs(dy) > 1) tr.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: 650, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' });
  }
}

boardBody.addEventListener('click', event => {
  const button = event.target.closest('[data-model]');
  if (button) selectModel(button.dataset.model);
});

// ---------- model dialog ----------

function selectModel(id) {
  window.location.hash = `model=${encodeURIComponent(id)}`;
}

function sparkline(trail, familyName) {
  const pts = trail.map(p => p.score);
  if (pts.length === 1) pts.push(pts[0]);
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const x = i => (i / (pts.length - 1)) * 100;
  const y = v => 54 - ((v - lo) / (hi - lo || 1)) * 52;
  const d = pts.map((v, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 100 56" preserveAspectRatio="none" aria-hidden="true"><path d="${d}" stroke="var(--family-${familyName})" /></svg>`;
}

function renderDialog() {
  const id = new URLSearchParams(window.location.hash.slice(1)).get('model');
  const m = S.models.get(id);
  if (!m) {
    if (dialog.open) dialog.close();
    return;
  }
  const order = ranked();
  const rank = 1 + order.findIndex(x => x.id === m.id);
  const tasks = Object.values(m.byTask);
  const recent = [...m.recent].reverse();
  dialogContent.innerHTML = `<header class="assessment-heading"><p class="eyebrow">Rank ${rank} of ${order.length}${m.kind === 'baseline' ? ' · baseline' : ''}</p>
    <h2 id="assessment-title">${escapeHtml(m.name)}</h2><div class="assessment-score">${m.score.toFixed(1)}<span>/100</span></div></header>
    <dl class="run-facts">
      <div><dt>Success</dt><dd>${pct(m.successRate)} (${m.successes}/${m.episodes})</dd></div>
      <div><dt>Cost / episode</dt><dd>${money(m.cost)}</dd></div>
      <div><dt>Total spend</dt><dd>${m.costTotal === 0 ? 'Free' : `$${m.costTotal.toFixed(3)}`}</dd></div>
      <div><dt>Steps per success</dt><dd>${m.steps == null ? '–' : m.steps.toFixed(1)}</dd></div>
      <div><dt>Latency / decision</dt><dd>${m.latency.toFixed(2)}s</dd></div>
    </dl>
    <p class="section-label">Score over episodes</p>${sparkline(m.trail, m.family)}
    <p class="section-label">By task</p>
    <table class="mini"><thead><tr><th>Task</th><th>Success</th><th>Score</th><th>Episodes</th></tr></thead><tbody>${tasks.map(t => `<tr><td>${escapeHtml(t.label)}</td><td>${pct(t.successRate)}</td><td>${t.score.toFixed(1)}</td><td>${t.n}</td></tr>`).join('')}</tbody></table>
    <p class="section-label">Latest episodes</p>
    <table class="mini"><thead><tr><th>Task</th><th>Result</th><th>Steps</th><th>Cost</th></tr></thead><tbody>${recent.map(e => `<tr><td>${escapeHtml(e.task)}</td><td class="${e.success ? 'ok' : 'bad'}">${e.success ? 'Reached' : 'Failed'}</td><td>${e.steps}</td><td class="dim">${money(e.cost)}</td></tr>`).join('')}</tbody></table>`;
  if (!dialog.open) {
    dialog.showModal();
    dialog.scrollTop = 0;
    document.activeElement?.blur(); // no focus ring on the close button at open
  }
  document.body.classList.add('has-assessment');
  plotRoot.classList.add('is-picking');
  S.nodes.get(m.id)?.group.classList.add('is-open');
}

function clearSelection() {
  document.body.classList.remove('has-assessment');
  plotRoot.classList.remove('is-picking');
  S.nodes.forEach(n => n.group.classList.remove('is-open'));
}

dialog.addEventListener('close', () => {
  clearSelection();
  if (new URLSearchParams(window.location.hash.slice(1)).has('model')) history.replaceState(null, '', window.location.pathname + window.location.search);
});
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
});
document.getElementById('close-assessment').addEventListener('click', () => dialog.close());
window.addEventListener('hashchange', renderDialog);

// ---------- status + training ----------

function renderStatus() {
  let cls = 'is-off';
  let text = 'Offline · retrying';
  if (S.connected) {
    if (S.mode === 'live') {
      cls = 'is-live';
      text = `Live · ${S.totals.episodes} episodes`;
    } else {
      cls = 'is-replay';
      text = `Replay · ${S.progress.delivered} / ${S.progress.planned}${S.loops ? ` · loop ${S.loops + 1}` : ''}`;
    }
    if (S.running.size) text += ` · ${S.running.size} running`;
  }
  statusEl.className = `status ${cls}`;
  statusEl.querySelector('span').textContent = text;
}

function renderTraining() {
  const el = document.getElementById('training');
  const t = S.training;
  if (!t || !t.epochs.length) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const last = t.epochs[t.epochs.length - 1];
  document.getElementById('training-title').textContent = t.phase === 'done' ? 'Free policy trained' : 'Training free policy';
  const parts = [`epoch ${last.epoch}`, `loss ${last.loss.toFixed(4)}`];
  if (last.val_loss != null) parts.push(`val ${last.val_loss.toFixed(4)}`);
  if (t.eval) parts.push(`eval ${pct(t.eval.success_rate)} of ${t.eval.n}`);
  document.getElementById('training-meta').textContent = parts.join(' · ');
  const curve = key => {
    const vals = t.epochs.map(e => e[key]).filter(v => v != null);
    if (vals.length < 2) return '';
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return vals.map((v, i) => `${i ? 'L' : 'M'} ${((i / (vals.length - 1)) * 400).toFixed(1)} ${(38 - ((v - lo) / (hi - lo || 1)) * 36).toFixed(1)}`).join(' ');
  };
  document.getElementById('training-loss').setAttribute('d', curve('loss'));
  document.getElementById('training-val').setAttribute('d', curve('val_loss'));
}

// ---------- data stream ----------

function applyState(s) {
  S.mode = s.mode;
  S.loops = s.loops;
  S.progress = s.progress;
  S.totals = s.totals;
  S.running = new Set(s.running);
  S.training = s.training;
  S.arena = s.arena;
  S.models = new Map(s.models.map(m => [m.id, m]));
  syncNodes();
  renderBoard();
  renderStatus();
  renderTraining();
  renderDialog();
}

function applyEpisode(e) {
  S.mode = e.mode;
  S.progress = e.progress;
  S.totals = e.totals;
  S.models.set(e.model.id, e.model);
  syncNodes();
  pingNode(e.model.id, e.episode.success);
  renderBoard();
  renderStatus();
  const open = new URLSearchParams(window.location.hash.slice(1)).get('model');
  if (open) renderDialog();
}

function connect() {
  const source = new EventSource('/api/stream');
  source.onopen = () => {
    S.connected = true;
    renderStatus();
  };
  source.onerror = () => {
    S.connected = false;
    renderStatus();
  };
  source.onmessage = event => {
    const e = JSON.parse(event.data);
    if (e.type === 'state') applyState(e);
    else if (e.type === 'episode') applyEpisode(e);
    else if (e.type === 'running') {
      S.running = new Set(e.running);
      syncNodes();
      renderStatus();
    } else if (e.type === 'training') {
      S.training = e.training;
      renderTraining();
    } else if (e.type === 'arena') {
      S.arena = e.up;
    }
  };
}

// ---------- views ----------

function setView(view) {
  S.view = view;
  document.querySelectorAll('[data-pane]').forEach(pane => {
    const on = pane.dataset.pane === view;
    pane.hidden = !on;
    pane.classList.toggle('is-on', on);
  });
  document.querySelectorAll('[data-view]').forEach(button => {
    const on = button.dataset.view === view;
    button.classList.toggle('is-on', on);
    if (on) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  if (view === 'plot') requestAnimationFrame(() => ensurePlot());
  if (view === 'board') renderBoard();
}

document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
new ResizeObserver(() => {
  if (S.view === 'plot') ensurePlot();
}).observe(plotRoot);

ensurePlot(true);
requestAnimationFrame(frame);
connect();
