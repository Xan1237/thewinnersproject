// "Labeled data" window: every recorded episode, grouped by model, viewable and downloadable.
import { kindOf, KIND_LABEL } from './kinds.js';

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const r2 = v => (typeof v === 'number' ? (Math.round(v * 100) / 100).toString() : v);

export function setupDataView() {
  const box = $('dataview'), body = $('databody');
  let timer = 0, selected = null, openFile = null;

  async function refresh() {
    let idx;
    try { idx = await (await fetch('/sim/data')).json(); } catch { body.innerHTML = '<p class="mute">Sim server not reachable.</p>'; return; }
    const t = idx.totals;
    const models = idx.models.filter(m => m.episodes || m.has_weights);
    $('datasummary').innerHTML = `<b>${t.episodes}</b> episodes · <b>${t.steps.toLocaleString()}</b> labeled decisions · <b>${t.successes}</b> successful (the training set) · saved in <code>arena/data/&lt;model&gt;/</code>`;
    $('dataall').href = '/sim/data/all.json';
    $('datalist').innerHTML = models.map(m => `
      <div class="dmodel ${m.kind} ${m.folder === selected ? 'sel' : ''}" data-folder="${esc(m.folder)}">
        <div class="dname">${esc(m.model)} <span class="kind ${m.kind}">${KIND_LABEL[m.kind]}</span></div>
        <div class="dstat">${m.episodes} episodes · ${m.successes} successful · ${m.steps} decisions${m.has_weights ? ' · weights' : ''}</div>
        <div class="dlinks"><a href="/sim/data/${encodeURIComponent(m.folder)}.jsonl" download>labeled steps (.jsonl)</a>${m.has_weights ? ' <a href="/sim/model" download="learned-bc-weights.json">weights (.json)</a>' : ''}</div>
      </div>`).join('') || '<p class="mute">Nothing recorded yet. Run the arena and come back.</p>';
    const cur = models.find(m => m.folder === selected);
    $('dataeps').innerHTML = cur ? cur.files.slice().reverse().map(f => `<button class="dep ${f.success ? 'ok' : f.outcome === 'done' ? 'bad' : 'cut'} ${f.file === openFile ? 'sel' : ''}" data-file="${esc(f.file)}">${esc(f.file.replace(/\.json$/, ''))}<span>${f.steps} steps${f.cost_usd ? ` · $${f.cost_usd}` : ''}</span></button>`).join('') : '<p class="mute">Pick a model.</p>';
  }

  async function show(folder, file) {
    openFile = file;
    const ep = await (await fetch(`/sim/data/${encodeURIComponent(folder)}/${encodeURIComponent(file)}`)).json();
    const rows = ep.steps.map(s => `<tr><td>${s.t}</td><td class="mono">${(s.state || []).map(r2).join(', ')}</td><td class="mono">${(s.action || []).map(r2).join(', ')}</td><td>${esc(s.label)}</td><td>${s.latency_s != null ? r2(s.latency_s) + ' s' : ''}</td><td>${s.cost_usd ? '$' + s.cost_usd : '$0'}</td></tr>`).join('');
    $('dataep').innerHTML = `<div class="dephead"><b>${esc(ep.task)}</b> · ${esc(ep.outcome)} · ${ep.success ? '✔ success' : '✘ not successful'} · <a href="/sim/data/${encodeURIComponent(folder)}/${encodeURIComponent(file)}" download>download this episode</a></div>
      <div class="mute">${esc(ep.reason)}</div>
      <table class="dtable"><thead><tr><th>t</th><th>state (x, z, heading, goal fwd, goal right, goal dist, holding, time left)</th><th>action (fwd, right, turn, grab)</th><th>label</th><th>decision time</th><th>cost</th></tr></thead><tbody>${rows}</tbody></table>`;
    refresh();
  }

  $('datalist').addEventListener('click', e => { const m = e.target.closest('.dmodel'); if (m && !e.target.closest('a')) { selected = m.dataset.folder; openFile = null; $('dataep').innerHTML = ''; refresh(); } });
  $('dataeps').addEventListener('click', e => { const b = e.target.closest('.dep'); if (b && selected) show(selected, b.dataset.file); });
  const open = () => { box.hidden = false; refresh(); timer = setInterval(refresh, 4000); };
  const close = () => { box.hidden = true; clearInterval(timer); };
  $('databtn').addEventListener('click', () => (box.hidden ? open() : close()));
  $('dataclose').addEventListener('click', close);
  box.addEventListener('click', e => { if (e.target === box) close(); });
  addEventListener('keydown', e => { if (e.key === 'Escape' && !box.hidden) close(); });
}
