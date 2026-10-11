import http from 'node:http';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { WebSocketServer, WebSocket } from 'ws';

export const JOINT_NAMES = Object.freeze([
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
  'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
  'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
  ...['left', 'right'].flatMap(side => [
    ...[0, 1, 2].map(i => `${side}_hand_thumb_${i}_joint`),
    ...[0, 1].map(i => `${side}_hand_index_${i}_joint`),
    ...[0, 1].map(i => `${side}_hand_middle_${i}_joint`),
  ]),
]);
const jointSet = new Set(JOINT_NAMES);
const terminal = new Set(['completed', 'rejected']);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const number = (value, min, max) => typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;

export function validateCommand(input) {
  if (!object(input)) throw new Error('Command must be a JSON object');
  if (!['move', 'turn', 'coffee', 'joints', 'stop'].includes(input.type)) throw new Error('Unknown command type');
  const fields = {
    move: ['type', 'forward', 'right', 'duration', 'sprint'],
    turn: ['type', 'radians', 'duration'], coffee: ['type'], joints: ['type', 'angles', 'duration'], stop: ['type'],
  }[input.type];
  if (!fields) throw new Error('Unknown command type');
  if (Object.keys(input).some(key => !fields.includes(key))) throw new Error('Unknown command field');
  if (input.type === 'move') {
    if (!number(input.forward, -1, 1) || !number(input.right, -1, 1)) throw new Error('forward and right must be finite numbers in [-1,1]');
    if (!number(input.duration, 0.1, 10)) throw new Error('duration must be in [0.1,10] seconds');
    if (input.sprint !== undefined && typeof input.sprint !== 'boolean') throw new Error('sprint must be boolean');
    return { ...input, sprint: input.sprint ?? false };
  }
  if (input.type === 'turn') {
    if (!number(input.radians, -Math.PI, Math.PI)) throw new Error('radians must be in [-pi,pi]');
    if (!number(input.duration, 0.1, 5)) throw new Error('duration must be in [0.1,5] seconds');
  }
  if (input.type === 'joints') {
    if (!object(input.angles) || Object.keys(input.angles).length === 0) throw new Error('angles must contain at least one G1 joint');
    for (const [name, angle] of Object.entries(input.angles)) {
      if (!jointSet.has(name)) throw new Error(`Unknown G1 joint: ${name}`);
      if (typeof angle !== 'number' || !Number.isFinite(angle)) throw new Error(`Joint angle must be finite: ${name}`);
    }
    if (!number(input.duration, 0.1, 10)) throw new Error('duration must be in [0.1,10] seconds');
  }
  return { ...input };
}

function finiteSnapshot(value, depth = 0, count = { value: 0 }) {
  if (++count.value > 8192 || depth > 20) return false;
  if (typeof value === 'number') return Number.isFinite(value);
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (Array.isArray(value)) return value.every(v => finiteSnapshot(v, depth + 1, count));
  if (object(value)) return Object.values(value).every(v => finiteSnapshot(v, depth + 1, count));
  return false;
}

export function createRobotApi({ port = 8817, token = process.env.ROBOT_API_TOKEN ?? '', vitePort = 8816 } = {}) {
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid API port');
  const records = new Map();
  let simulator = null, latestState = null, stateUpdatedAt = null, activeId = null, commandTimer = null;
  const waiting = [];
  const reply = (res, status, body) => {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    res.end(JSON.stringify(body));
  };
  const equalToken = supplied => {
    const a = Buffer.from(supplied ?? ''), b = Buffer.from(token);
    return a.length === b.length && timingSafeEqual(a, b);
  };
  const authenticated = req => !token || (req.headers.authorization?.startsWith('Bearer ') && equalToken(req.headers.authorization.slice(7)));
  const originAllowed = origin => {
    if (!origin) return true; // Native clients/curl do not supply browser Origin.
    try {
      const u = new URL(origin), actualPort = server.address()?.port ?? port;
      return u.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname)
        && [String(actualPort), String(vitePort)].includes(u.port);
    } catch { return false; }
  };
  const send = (socket, value) => { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(value)); };
  const finish = (record, status, message) => {
    record.status = status;
    record.finishedAt = new Date().toISOString();
    if (message) record.message = message.slice(0, 512);
    if (activeId === record.id) { activeId = null; clearTimeout(commandTimer); commandTimer = null; }
  };
  const dispatch = () => {
    if (!simulator || activeId || waiting.length === 0) return;
    const id = waiting.shift(), record = records.get(id);
    activeId = id;
    record.dispatchedAt = new Date().toISOString();
    send(simulator, { type: 'command', id, command: record.command });
    commandTimer = setTimeout(() => {
      if (activeId === id) {
        finish(record, 'rejected', 'Simulator did not complete command before timeout');
        send(simulator, { type: 'cancel', id });
        dispatch();
      }
    }, ((record.command.duration ?? 10) + 30) * 1000);
    commandTimer.unref();
  };
  const enqueue = command => {
    if (command.type === 'stop') {
      for (const id of waiting.splice(0)) finish(records.get(id), 'rejected', 'Canceled by stop');
      if (activeId) finish(records.get(activeId), 'rejected', 'Canceled by stop');
    } else if (waiting.length + (activeId ? 1 : 0) >= 32) {
      return null;
    }
    // Keep bounded status history without discarding any outstanding command.
    if (records.size >= 1024) {
      for (const [id, record] of records) if (terminal.has(record.status)) {
        records.delete(id);
        if (records.size < 1024) break;
      }
    }
    const record = { id: randomUUID(), command, status: 'queued', createdAt: new Date().toISOString() };
    records.set(record.id, record);
    waiting.push(record.id);
    dispatch();
    return record;
  };
  const schema = () => ({
    version: 1,
    transport: 'HTTP commands + one browser simulator WebSocket at /socket',
    authentication: token ? 'Bearer ROBOT_API_TOKEN required; WS query token required' : 'No token configured; loopback only',
    coordinates: {
      world: 'World X/Y/Z, Y-up. /api/state contains the latest browser observation, with its timestamp; inspect connected before acting.',
      move: 'Robot-relative, independent of camera: at heading=0, forward=1 is world +Z and right=1 is world -X. Negative values reverse. Diagonal input may be normalized by simulator.',
      turn: 'Signed radians about world +Y; positive heading turns initial +Z facing toward +X.',
      joints: 'Named native G1 hinge angles in radians. Current physical limits are in state.limits; the simulator rejects angles outside those limits. Not motor control of a real robot.',
      time: 'duration in seconds',
    },
    commands: {
      move: { type: 'move', forward: 'number [-1,1]', right: 'number [-1,1]', duration: 'number [0.1,10]', sprint: 'optional boolean, default false' },
      turn: { type: 'turn', radians: 'number [-pi,pi]', duration: 'number [0.1,5]' },
      coffee: { type: 'coffee' },
      joints: { type: 'joints', angles: 'object: G1 joint name -> finite radians', duration: 'number [0.1,10]', jointNames: JOINT_NAMES },
      stop: { type: 'stop', description: 'Interrupt active command and clear queue, then stop simulator' },
    },
    endpoints: { state: 'GET /api/state', submit: 'POST /api/commands', status: 'GET /api/commands/:id', stop: 'POST /api/stop', schema: 'GET /api/schema' },
    limits: { maxBodyBytes: 16384, maxOutstandingCommands: 32, retainedStatuses: 1024 },
    statuses: ['queued', 'running', 'completed', 'rejected'],
    statusMeaning: 'queued includes awaiting simulator acknowledgement; running/completed are confirmed by browser acknowledgements. Disconnect, stop and timeout can reject commands.',
    websocket: { hello: { type: 'hello', role: 'simulator' }, command: { type: 'command', id: 'uuid', command: 'validated command' },
      state: { type: 'state', state: 'JSON snapshot' }, ack: { type: 'ack', id: 'uuid', status: 'running|completed|rejected', message: 'optional text' },
      cancel: { type: 'cancel', id: 'uuid', description: 'Timeout; cease command before next dispatch' } },
  });
  const readBody = req => new Promise((accept, reject) => {
    let size = 0, rejected = false; const chunks = [];
    req.on('data', chunk => {
      size += chunk.length;
      if (size > 16384) { rejected = true; reject(Object.assign(new Error('Body exceeds 16KB'), { status: 413 })); }
      else if (!rejected) chunks.push(chunk);
    });
    req.on('end', () => {
      if (rejected) return;
      try { const data = Buffer.concat(chunks).toString('utf8'); accept(data ? JSON.parse(data) : null); }
      catch { reject(Object.assign(new Error('Malformed JSON'), { status: 400 })); }
    });
    req.on('error', reject);
  });
  const server = http.createServer(async (req, res) => {
    if (!originAllowed(req.headers.origin)) return reply(res, 403, { error: 'Origin is not allowed' });
    if (!authenticated(req)) return reply(res, 401, { error: 'Bearer authentication required' });
    let path;
    try { path = new URL(req.url, 'http://127.0.0.1').pathname; }
    catch { return reply(res, 400, { error: 'Invalid request URL' }); }
    if (req.method === 'GET' && path === '/api/state') return reply(res, 200, {
      connected: Boolean(simulator), state: latestState, stateUpdatedAt, activeCommandId: activeId, queuedCommands: waiting.length,
    });
    if (req.method === 'GET' && path === '/api/schema') return reply(res, 200, schema());
    if (req.method === 'GET' && path.startsWith('/api/commands/')) {
      const record = records.get(path.slice('/api/commands/'.length));
      return reply(res, record ? 200 : 404, record ?? { error: 'Unknown command id' });
    }
    if (req.method === 'POST' && (path === '/api/commands' || path === '/api/stop')) {
      try {
        const body = await readBody(req);
        if (path === '/api/stop' && body !== null && (!object(body) || Object.keys(body).length)) throw new Error('/api/stop accepts an empty body or {}');
        const command = validateCommand(path === '/api/stop' ? { type: 'stop' } : body);
        const record = enqueue(command);
        return reply(res, record ? 202 : 429, record ?? { error: 'Command queue is full' });
      } catch (error) { return reply(res, error.status ?? 400, { error: error.message }); }
    }
    reply(res, 404, { error: 'Unknown endpoint' });
  });
  const wss = new WebSocketServer({ noServer: true, maxPayload: 65536 });
  server.requestTimeout = 30000;
  server.headersTimeout = 10000;
  server.on('upgrade', (req, socket, head) => {
    let u;
    try { u = new URL(req.url, 'http://127.0.0.1'); }
    catch { socket.end('HTTP/1.1 400 Rejected\r\nConnection: close\r\n\r\n'); return; }
    const status = u.pathname !== '/socket' ? 404 : !originAllowed(req.headers.origin) ? 403
      : token && !equalToken(u.searchParams.get('token')) ? 401 : null;
    if (status) { socket.end(`HTTP/1.1 ${status} Rejected\r\nConnection: close\r\n\r\n`); return; }
    wss.handleUpgrade(req, socket, head, ws => wss.emit('connection', ws, req));
  });
  wss.on('connection', ws => {
    let ready = false;
    const helloTimer = setTimeout(() => ws.close(1008, 'Simulator hello required'), 5000);
    helloTimer.unref();
    ws.on('error', () => {});
    ws.on('message', raw => {
      let message;
      try { message = JSON.parse(raw.toString()); } catch { send(ws, { type: 'error', error: 'Malformed JSON' }); return; }
      if (!object(message)) { send(ws, { type: 'error', error: 'Object required' }); return; }
      if (!ready) {
        if (message.type !== 'hello' || message.role !== 'simulator' || simulator) { ws.close(1008, simulator ? 'Simulator already connected' : 'Simulator hello required'); return; }
        ready = true; simulator = ws; clearTimeout(helloTimer);
        send(ws, { type: 'welcome', role: 'simulator', protocolVersion: 1 });
        dispatch(); return;
      }
      if (message.type === 'state') {
        if (!object(message.state) || !finiteSnapshot(message.state)) { send(ws, { type: 'error', error: 'Invalid state snapshot' }); return; }
        latestState = message.state; stateUpdatedAt = new Date().toISOString(); return;
      }
      if (message.type === 'ack') {
        // The browser may acknowledge an interruption after stop/timeout has
        // already rejected that command. Such late acknowledgements are benign.
        if (message.id !== activeId && terminal.has(records.get(message.id)?.status)) return;
        if (message.id !== activeId || !['running', 'completed', 'rejected'].includes(message.status)) { send(ws, { type: 'error', error: 'Invalid active command acknowledgement' }); return; }
        if (message.message !== undefined && typeof message.message !== 'string') { send(ws, { type: 'error', error: 'Ack message must be text' }); return; }
        const record = records.get(activeId);
        if (terminal.has(message.status)) { finish(record, message.status, message.message); dispatch(); }
        else {
          record.status = 'running';
          record.startedAt ??= new Date().toISOString();
          if (message.message) record.message = message.message.slice(0, 512);
        }
        return;
      }
      send(ws, { type: 'error', error: 'Unknown simulator message' });
    });
    ws.on('close', () => {
      clearTimeout(helloTimer);
      if (simulator === ws) {
        simulator = null;
        if (activeId) finish(records.get(activeId), 'rejected', 'Simulator disconnected');
        for (const id of waiting.splice(0)) finish(records.get(id), 'rejected', 'Simulator disconnected before command began');
      }
    });
  });
  return {
    server,
    start: () => new Promise((accept, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', () => accept(server.address())); }),
    close: async () => {
      clearTimeout(commandTimer);
      for (const ws of wss.clients) ws.terminate();
      await new Promise(accept => wss.close(accept));
      server.closeIdleConnections();
      await new Promise((accept, reject) => server.close(error => error ? reject(error) : accept()));
    },
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const api = createRobotApi({ port: Number(process.env.ROBOT_API_PORT ?? 8817) });
  const address = await api.start();
  console.log(`Robot API listening at http://127.0.0.1:${address.port} (${process.env.ROBOT_API_TOKEN ? 'bearer token required' : 'loopback access'})`);
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, async () => { await api.close(); process.exit(0); });
}
