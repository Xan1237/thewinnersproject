// Local benchmark server: serves the page and streams live results over SSE.
//
//   node server.js            -> http://127.0.0.1:8830
//
// Data: every run under input_layer/runs (episodes_all.jsonl, else episodes.jsonl).
// - Baselines (scripted/teacher) are loaded at once; model episodes are replayed one by one.
// - Episodes that appear in the run files while the server runs are streamed as live results,
//   and replay stops (the board is settled to the full recorded history first).
// - If the arena sim server is up, its session and training events are forwarded too.

import { createServer } from 'node:http';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Board, interleave, summarize } from './lib/stats.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(HERE, 'public');
const PORT = Number(process.env.BENCH_PORT ?? 8830);
const HOST = process.env.BENCH_HOST ?? '127.0.0.1';
const RUNS_DIR = resolve(process.env.BENCH_RUNS ?? join(HERE, '..', 'input_layer', 'runs'));
const ARENA_URL = process.env.ARENA_URL ?? 'http://127.0.0.1:8818';
const PACE_MS = Number(process.env.BENCH_PACE_MS ?? 900);
const LOOP_PAUSE_MS = Number(process.env.BENCH_LOOP_PAUSE_MS ?? 12000);
const LOOP = process.env.BENCH_LOOP !== '0';
// teacher-* episodes are synthetic data from the offline pipeline test, not a benchmark entrant.
const SKIP_MODELS = new RegExp(process.env.BENCH_SKIP_MODELS ?? '^teacher');

// ---------- data ----------

function runFiles() {
  if (!existsSync(RUNS_DIR)) return [];
  const files = [];
  for (const name of readdirSync(RUNS_DIR)) {
    if (name.startsWith('tmp') || name.startsWith('.')) continue;
    const dir = join(RUNS_DIR, name);
    if (!statSync(dir).isDirectory()) continue;
    for (const file of ['episodes_all.jsonl', 'episodes.jsonl']) {
      const path = join(dir, file);
      if (existsSync(path)) {
        files.push({ run: name, path, mtimeMs: statSync(path).mtimeMs });
        break;
      }
    }
  }
  return files.sort((a, b) => a.mtimeMs - b.mtimeMs);
}

function readEpisodes(file) {
  const out = [];
  for (const line of readFileSync(file.path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const ep = summarize(JSON.parse(line), file.run);
      // Model episodes with no recorded spend predate cost tracking; they would fake a free model.
      const unpriced = ep.kind === 'llm' && ep.cost === 0;
      if (!SKIP_MODELS.test(ep.model) && !unpriced) out.push(ep);
    } catch {
      /* a half-written line; the next poll picks it up */
    }
  }
  return out;
}

const board = new Board();
const seen = new Set();
const fileStamp = new Map(); // path -> "mtime:size"
let history = []; // every recorded episode at startup, in file order
let queue = []; // episodes waiting to be delivered to the board
let mode = 'replay'; // 'replay' | 'live'
let loops = 0;
let delivered = 0;
let planned = 0;
let loopTimer = null;
const running = new Map(); // arena session id -> model id
let training = null;

function stamp(file) {
  const s = statSync(file.path);
  return `${s.mtimeMs}:${s.size}`;
}

function loadHistory() {
  history = [];
  for (const file of runFiles()) {
    fileStamp.set(file.path, stamp(file));
    for (const ep of readEpisodes(file)) {
      if (seen.has(ep.id)) continue;
      seen.add(ep.id);
      history.push(ep);
    }
  }
}

function startReplay() {
  board.clear();
  for (const ep of history) if (ep.kind === 'baseline') board.add(ep);
  queue = interleave(history.filter(ep => ep.kind === 'llm'));
  delivered = 0;
  planned = queue.length;
}

// ---------- streaming ----------

const clients = new Set();

function send(res, event) {
  res.write(`data: ${JSON.stringify(event)}\n\n`);
}

function broadcast(event) {
  for (const res of clients) send(res, event);
}

function state() {
  return {
    type: 'state',
    mode,
    loops,
    progress: { delivered, planned, queued: queue.length },
    totals: board.totals(),
    models: board.snapshot(),
    running: [...new Set(running.values())],
    training,
    arena: arenaUp,
  };
}

function deliver() {
  const ep = queue.shift();
  if (!ep) return false;
  const model = board.add(ep);
  delivered += 1;
  broadcast({
    type: 'episode',
    episode: ep,
    model,
    mode,
    totals: board.totals(),
    progress: { delivered, planned, queued: queue.length },
  });
  return true;
}

function tick() {
  if (!deliver() && mode === 'replay' && LOOP && !loopTimer) {
    loopTimer = setTimeout(() => {
      loopTimer = null;
      if (mode !== 'replay') return;
      loops += 1;
      startReplay();
      broadcast(state());
    }, LOOP_PAUSE_MS);
  }
}

// Live ingestion: episodes appended to (or newly written into) run files after startup.
function pollRuns() {
  const fresh = [];
  for (const file of runFiles()) {
    let s;
    try {
      s = stamp(file);
    } catch {
      continue;
    }
    if (fileStamp.get(file.path) === s) continue;
    fileStamp.set(file.path, s);
    for (const ep of readEpisodes(file)) {
      if (seen.has(ep.id)) continue;
      seen.add(ep.id);
      fresh.push({ ...ep, live: true });
    }
  }
  if (!fresh.length) return;
  if (mode === 'replay') {
    // Settle the board to the full recorded history, then stream only what is new.
    mode = 'live';
    if (loopTimer) clearTimeout(loopTimer);
    loopTimer = null;
    board.clear();
    for (const ep of history) board.add(ep);
    queue = [];
    delivered = 0;
    planned = 0;
    broadcast(state());
  }
  history.push(...fresh);
  queue.push(...fresh.filter(ep => ep.kind === 'llm'));
  for (const ep of fresh) if (ep.kind === 'baseline') board.add(ep);
  planned += fresh.filter(ep => ep.kind === 'llm').length;
}

// ---------- arena sim (optional) ----------

let arenaUp = false;

function handleArena(e) {
  if (e.type === 'session_started') running.set(e.session, e.model);
  else if (e.type === 'session_finished' || e.type === 'session_ended') running.delete(e.session);
  else if (e.type === 'reset') running.clear();
  else if (e.type === 'training') {
    if (e.phase === 'start') training = { phase: 'start', epochs: [], info: e };
    else if (training && e.phase === 'epoch') {
      training.phase = 'epoch';
      training.epochs.push({ epoch: e.epoch, loss: e.loss, val_loss: e.val_loss });
      if (training.epochs.length > 400) training.epochs.shift();
    } else if (training && e.phase === 'eval') training.eval = { success_rate: e.success_rate, n: e.n };
    else if (training && e.phase === 'done') training.phase = 'done';
    broadcast({ type: 'training', training });
    return;
  } else return;
  broadcast({ type: 'running', running: [...new Set(running.values())] });
}

async function followArena() {
  for (;;) {
    try {
      const res = await fetch(`${ARENA_URL}/sim/events?replay=0`);
      if (!res.ok || !res.body) throw new Error(`arena ${res.status}`);
      if (!arenaUp) {
        arenaUp = true;
        broadcast({ type: 'arena', up: true });
      }
      const decoder = new TextDecoder();
      let buf = '';
      for await (const chunk of res.body) {
        buf += decoder.decode(chunk, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const line = block.split('\n').find(l => l.startsWith('data: '));
          if (!line) continue;
          try {
            handleArena(JSON.parse(line.slice(6)));
          } catch {
            /* ignore malformed event */
          }
        }
      }
    } catch {
      /* arena not running: retry below */
    }
    if (arenaUp) {
      arenaUp = false;
      running.clear();
      broadcast({ type: 'arena', up: false });
      broadcast({ type: 'running', running: [] });
    }
    await new Promise(r => setTimeout(r, 3000));
  }
}

// ---------- http ----------

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/api/stream') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
    send(res, state());
    clients.add(res);
    req.on('close', () => clients.delete(res));
    return;
  }
  if (url.pathname === '/api/state') {
    res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    res.end(JSON.stringify(state()));
    return;
  }
  const rel = normalize(decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
  const file = join(PUBLIC, rel);
  if (!file.startsWith(PUBLIC) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found');
    return;
  }
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(file));
});

loadHistory();
if (!history.length) console.warn(`No episodes found under ${RUNS_DIR}`);
startReplay();
setInterval(tick, PACE_MS);
setInterval(pollRuns, 1000);
setInterval(() => broadcast({ type: 'ping' }), 15000); // keeps idle SSE connections open
followArena();
server.listen(PORT, HOST, () => {
  console.log(`bench http://${HOST}:${PORT}  runs=${RUNS_DIR}  episodes=${history.length}  replay=${planned}`);
});
