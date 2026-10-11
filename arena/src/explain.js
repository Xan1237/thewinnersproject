// "What is this data?" window: the labeled data explained in plain words, with a live example.
import { STATE, ACTION, PARTS } from './recipe.js';
import { kindOf } from './kinds.js';

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const f = v => (typeof v === 'number' ? (Math.round(v * 100) / 100).toString() : '–');

export function setupExplain(panel) {
  const box = $('explain');
  let timer = 0;

  function render() {
    const ev = panel.lastStep;
    const rows = (spec, vals) => spec.map(([n, d], i) => `<tr><td class="n">${n}</td><td>${d}</td><td class="v">${vals ? f(vals[i]) : ''}</td></tr>`).join('');
    const models = [...panel.models.entries()];
    const front = models.filter(([m]) => kindOf(m) === 'frontier' && panel.models.get(m).decisions);
    const lat = front.length ? front.reduce((n, [, s]) => n + s.latency, 0) / front.reduce((n, [, s]) => n + s.decisions, 0) : null;
    const cost = front.length ? front.reduce((n, [, s]) => n + s.cost, 0) / Math.max(front.reduce((n, [, s]) => n + s.decisions, 0), 1) : null;
    const part = ev && ev.stages > 1 ? `part ${ev.stage + 1} of ${ev.stages}: ${PARTS[ev.stage_name] ?? ev.stage_name}` : '';
    $('explainbody').innerHTML = `
      <p class="lead">Every row of data is <b>one decision of one robot</b>: what it sensed, what it did, and in words why. This is exactly the shape of data that real robot policies are trained on (imitation learning).</p>
      ${ev ? `<div class="live"><span class="chip" style="--c:${panel.color(ev.model)}">${esc(ev.model)}</span> ${esc(part)} → <b>“${esc(ev.label || '')}”</b></div>` : '<p class="mute">Waiting for the first decision…</p>'}
      <div class="cols">
        <div><h4>1 · SENSE <small>12 numbers in</small></h4><table>${rows(STATE, ev?.state)}</table></div>
        <div><h4>2 · ACT <small>4 numbers out</small></h4><table>${rows(ACTION, ev?.action)}</table>
          <h4>3 · LABEL <small>in words</small></h4><p class="lab">${ev ? `“${esc(ev.label || '')}”` : ''}</p><p class="mute">The label says what the decision meant, so people (and models) can check the numbers against the story.</p></div>
      </div>
      <div class="why"><h4>Why this matters</h4>
        <ul>
          <li><b>Where it comes from:</b> 7 frontier AI models run the robots through API calls. Each one reads the room and decides, step by step, how to make the coffee.</li>
          <li><b>What we keep:</b> only the runs where the coffee was served. Those become the training set.</li>
          <li><b>What we train:</b> a tiny network that copies the winners: sensor numbers in, motor numbers out, no internet, no tokens.</li>
          <li><b>Cost of a decision:</b> frontier models ${lat != null ? `about ${lat.toFixed(1)} s and $${cost.toFixed(3)} each (measured live)` : 'take seconds and cost cents'}; the trained network ${panel.trainedLatency != null ? `${panel.trainedLatency}` : 'under a millisecond'} and $0. A real robot has to think many times a second, offline.</li>
          <li><b>Honest scope:</b> this is a simulation of one task. The point is the pipeline: the same loop, with real robot sensors, is how a robot learns any task.</li>
        </ul></div>`;
  }

  const open = () => { box.hidden = false; render(); timer = setInterval(render, 1500); };
  const close = () => { box.hidden = true; clearInterval(timer); };
  $('explainbtn').addEventListener('click', () => (box.hidden ? open() : close()));
  $('explainclose').addEventListener('click', close);
  box.addEventListener('click', e => { if (e.target === box) close(); });
  addEventListener('keydown', e => { if (e.key === 'Escape' && !box.hidden) close(); });
}
