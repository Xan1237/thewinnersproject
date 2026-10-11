// Headless sim server (contract: robot/docs/SIM_API.md). Loopback only.
import http from 'node:http';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ACTION_SPEC, OBSERVATION_SPEC, Session, World } from './core.js';
import { interpret } from './commands.js';
import { transcribe } from './voice.js';
import { Recorder, slug } from './recorder.js';

const here = dirname(fileURLToPath(import.meta.url));
const MAX_BODY = 64 * 1024;
const MAX_AUDIO = 8 * 1024 * 1024;
const MAX_SESSIONS = 64;
const HISTORY = 2000;

export function createServer({ manifestPath = join(here, '..', 'public', 'room', 'manifest.json'), dataDir = process.env.ARENA_DATA_DIR ?? join(here, '..', 'data') } = {}) {
  const rec = new Recorder(dataDir);
  let trainedModel = null;   // weights of the distilled policy (POST /sim/model), for the viewer's network visual
  const world = new World(JSON.parse(readFileSync(manifestPath, 'utf8')));
  const sessions = new Map();
  const clients = new Set();
  const history = [];
  let nextId = 1;
  let command = { id: 0 };   // latest operator order, polled by the agents (GET /sim/command)
  world.peers = () => sessions.values();
  /** Chairs other robots hold or walk to, so that one robot sits on one chair. */
  const claimedSeats = except => new Set([...sessions.values()].filter(s => s !== except && s.spec.kind === 'sit').map(s => s.spec.target.id));

  const broadcast = event => {
    event.ts = Date.now();
    history.push(event);
    if (history.length > HISTORY) history.shift();
    const data = `data: ${JSON.stringify(event)}\n\n`;
    for (const res of clients) res.write(data);
  };

  const send = (res, status, body) => {
    const text = JSON.stringify(body);
    res.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    res.end(text);
  };

  const sendRaw = (res, status, type, text, filename) => {
    res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store', ...(filename ? { 'content-disposition': `attachment; filename="${filename}"` } : {}) });
    res.end(text);
  };

  const readJson = req => new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', c => { size += c.length; if (size > MAX_BODY) { reject(Object.assign(new Error('body too large'), { status: 413 })); req.destroy(); } else chunks.push(c); });
    req.on('end', () => { try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); } catch { reject(Object.assign(new Error('invalid JSON'), { status: 400 })); } });
    req.on('error', reject);
  });

  const readBody = (req, limit) => new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', c => { size += c.length; if (size > limit) { reject(Object.assign(new Error('body too large'), { status: 413 })); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });

  /** Store + announce an order; the agents pick it up and re-task their robots. */
  const issueCommand = async body => {
    let task = body.task ?? null, say = body.say ?? '', via = 'explicit';
    if (!task && body.text) ({ task, say, via } = await interpret(String(body.text).slice(0, 500), world));
    command = { id: command.id + 1, text: body.text ?? '', task, say, via, stt: body.stt ?? null, ts: Date.now() };
    broadcast({ type: 'command', ...command });
    return command;
  };

  const sessionSnapshot = s => ({ id: s.id, task: s.task, model: s.model, seed: s.seed, t: s.t, done: s.done, success: s.success, observation: s.observation() });

  const handler = async (req, res) => {
    const origin = req.headers.origin;
    if (origin && !/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin)) return send(res, 403, { error: 'foreign origin' });
    const url = new URL(req.url, 'http://127.0.0.1');
    const path = url.pathname;
    try {
      if (req.method === 'GET' && path === '/sim/manifest') return send(res, 200, world.manifest);
      if (req.method === 'GET' && path === '/sim/events') {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
        res.write(': connected\n\n');
        const replay = url.searchParams.get('replay') !== '0';
        if (replay) for (const e of history) res.write(`data: ${JSON.stringify(e)}\n\n`);
        clients.add(res);
        req.on('close', () => clients.delete(res));
        return;
      }
      if (req.method === 'POST' && path === '/sim/training') {
        const body = await readJson(req);
        broadcast({ ...body, type: 'training' });
        return send(res, 200, { ok: true });
      }
      if (req.method === 'GET' && path === '/sim/data') return send(res, 200, rec.index());
      if (req.method === 'GET' && path === '/sim/data/all.json') {
        const idx = rec.index();
        const all = { exported_at: new Date().toISOString(), models: idx.models.map(m => ({ model: m.model, kind: m.kind, episodes: m.files.map(f => JSON.parse(rec.episode(m.folder, f.file))) })) };
        return sendRaw(res, 200, 'application/json', JSON.stringify(all), 'winners-labeled-data.json');
      }
      const dm = /^\/sim\/data\/([a-z0-9._-]+?)(?:\.jsonl|\/(\d{4}-[a-z0-9._-]+\.json))$/.exec(path);
      if (req.method === 'GET' && dm) {
        if (dm[2]) { const text = rec.episode(dm[1], dm[2]); return text ? sendRaw(res, 200, 'application/json', text, dm[2]) : send(res, 404, { error: 'no such episode' }); }
        const rows = rec.rows(dm[1]);
        return rows ? sendRaw(res, 200, 'application/x-ndjson', rows.map(r => JSON.stringify(r)).join('\n') + '\n', `${dm[1]}-labeled-steps.jsonl`) : send(res, 404, { error: 'no such model' });
      }
      if (path === '/sim/model') {
        if (req.method === 'GET') return trainedModel ? send(res, 200, trainedModel) : send(res, 404, { error: 'no trained model yet' });
        if (req.method === 'POST') {
          const body = JSON.parse((await readBody(req, 4 * 1024 * 1024)).toString('utf8') || '{}');
          if (!Array.isArray(body.layers) || !body.layers.length) return send(res, 400, { error: 'layers required' });
          trainedModel = body;
          if (dataDir) { const dir = join(dataDir, slug(body.name ?? 'learned-bc')); mkdirSync(dir, { recursive: true }); writeFileSync(join(dir, 'model.json'), JSON.stringify(body)); }
          broadcast({ type: 'model', name: body.name ?? 'learned-bc', sizes: body.sizes, params: body.params });
          return send(res, 200, { ok: true });
        }
      }
      if (req.method === 'POST' && path === '/sim/reset') {
        rec.open.clear();
        sessions.clear(); history.length = 0;
        broadcast({ type: 'reset' });
        return send(res, 200, { ok: true });
      }
      if (req.method === 'GET' && path === '/sim/command') return send(res, 200, command);
      if (req.method === 'POST' && path === '/sim/command') {
        const body = await readJson(req);
        return send(res, 200, await issueCommand(body));
      }
      if (req.method === 'POST' && path === '/sim/voice') {
        const audio = await readBody(req, MAX_AUDIO);
        if (!audio.length) return send(res, 400, { error: 'empty audio' });
        const heard = await transcribe(audio, req.headers['content-type'] ?? 'audio/webm');
        broadcast({ type: 'heard', text: heard.text, stt: heard.model, latency_s: heard.latency_s });
        if (!heard.text.trim()) return send(res, 200, { id: command.id, text: '', task: null, say: "I didn't hear anything." });
        return send(res, 200, await issueCommand({ text: heard.text, stt: heard.model }));
      }
      if (req.method === 'GET' && path === '/sim/sessions') return send(res, 200, { sessions: [...sessions.values()].map(sessionSnapshot) });
      if (req.method === 'POST' && path === '/sim/sessions') {
        if (sessions.size >= MAX_SESSIONS) {
          for (const [k, s] of sessions) if (s.done) sessions.delete(k);
          if (sessions.size >= MAX_SESSIONS) return send(res, 429, { error: 'too many sessions' });
        }
        const body = await readJson(req);
        const id = `s${nextId++}`;
        const s = new Session(world, { id, task: body.task ?? 'goto:armchair_blue', seed: Number.isInteger(body.seed) ? body.seed : 0, model: String(body.model ?? 'unknown').slice(0, 64), preview: body.preview === true, maxSteps: Number.isInteger(body.max_steps) ? Math.min(Math.max(body.max_steps, 1), 1000) : 120, claimed: claimedSeats(null) });
        sessions.set(id, s);
        rec.start(s);
        broadcast({ type: 'session_started', session: id, model: s.model, task: s.task, seed: s.seed, task_description: s.spec.description, target: [s.spec.target.x, s.spec.target.z], target_name: s.spec.target.name, position: [s.x, s.z], heading: s.heading, y: world.groundHeight(s.x, s.z), max_steps: s.maxSteps, seat_placement: s.spec.placement ?? null });
        return send(res, 200, { id, task_description: s.spec.description, action_spec: ACTION_SPEC, observation_spec: OBSERVATION_SPEC, observation: s.observation() });
      }
      const m = /^\/sim\/sessions\/([^/]+)(?:\/(step|task))?$/.exec(path);
      if (m) {
        const s = sessions.get(m[1]);
        if (!s) return send(res, 404, { error: 'unknown session' });
        if (req.method === 'GET' && !m[2]) return send(res, 200, sessionSnapshot(s));
        if (req.method === 'DELETE' && !m[2]) {
          sessions.delete(m[1]);
          rec.finish(s, 'abandoned');
          broadcast({ type: 'session_ended', session: s.id, model: s.model, task: s.task, success: s.success, steps: s.t, reason: s.reason });
          return send(res, 200, { ok: true });
        }
        if (req.method === 'POST' && m[2] === 'task') {
          const body = await readJson(req);
          rec.finish(s, 'interrupted');
          s.retask(String(body.task ?? ''), claimedSeats(s));
          rec.start(s);
          broadcast({ type: 'task_changed', session: s.id, model: s.model, task: s.task, task_description: s.spec.description, target: [s.spec.target.x, s.spec.target.z], target_name: s.spec.target.name, position: [s.x, s.z], heading: s.heading, y: world.groundHeight(s.x, s.z), max_steps: s.maxSteps, seat_placement: s.spec.placement ?? null });
          return send(res, 200, { id: s.id, task: s.task, task_description: s.spec.description, observation: s.observation() });
        }
        if (req.method === 'POST' && m[2] === 'step') {
          const body = await readJson(req);
          if (s.done) return send(res, 409, { error: 'session finished' });
          if (!Array.isArray(body.action)) return send(res, 400, { error: 'action must be an array' });
          const model = body.model ? String(body.model).slice(0, 64) : s.model;
          const label = body.label == null ? null : String(body.label).slice(0, 200);
          let detail = null;
          if (body.detail != null) { const j = JSON.stringify(body.detail); if (j.length <= 2000) detail = body.detail; }
          const before = s.observation();
          const r = s.step(body.action);
          broadcast({
            type: 'step', session: s.id, model, task: s.task, t: s.t, state: before.state, action: body.action.map(Number), label, detail, pose: r.info.pose ?? (typeof body.pose === 'string' ? body.pose.slice(0, 32) : null), pose_s: r.info.pose_s, pose_offset: r.info.pose_offset, seat: r.info.seat,
            position: r.info.position, heading: r.info.heading, y: r.info.y, path: r.path, joints: r.info.joints,
            holding: r.info.holding, collided: r.info.collided, distance: r.info.distance, reward: r.reward, done: r.done,
            success: r.info.success, reason: r.info.reason, seating: r.info.seating, seated: r.info.seated, seat_time: r.info.seat_time, pelvis_height: r.info.pelvis_height,
          });
          rec.step(s, {
            t: s.t - 1, state: before.state, observation_text: before.text, action: body.action.map(Number), label,
            plan: detail?.plan ?? null, jev: detail?.jev ?? null, latency_s: detail?.latency_s ?? null, cost_usd: detail?.cost_usd ?? 0,
            reward: r.reward, distance: r.info.distance, position: r.info.position, holding: r.info.holding, collided: r.info.collided,
            done: r.done, success: r.info.success, reason: r.info.reason, seating: r.info.seating, seated: r.info.seated, seat_time: r.info.seat_time, pelvis_height: r.info.pelvis_height,
          });
          if (r.done) rec.finish(s, 'done', { success: r.info.success, reason: r.info.reason });
          if (r.done) broadcast({ type: 'session_finished', session: s.id, model, task: s.task, success: r.info.success, steps: s.t, reason: r.info.reason });
          const { path, ...resp } = r;
          return send(res, 200, resp);
        }
      }
      return send(res, 404, { error: 'not found' });
    } catch (e) {
      return send(res, e.status ?? 400, { error: String(e.message ?? e) });
    }
  };

  return { server: http.createServer(handler), world, sessions };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.SIM_PORT ?? 8818);
  const { server } = createServer();
  server.listen(port, '127.0.0.1', () => console.log(`sim server on http://127.0.0.1:${port}`));
}
