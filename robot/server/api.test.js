import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { WebSocket } from 'ws';
import { createRobotApi, validateCommand, JOINT_NAMES } from './api.js';

test('sit and stand are bounded actions without arbitrary overrides',()=>{
  assert.deepEqual(validateCommand({type:'sit'}),{type:'sit'});
  assert.deepEqual(validateCommand({type:'stand'}),{type:'stand'});
  assert.throws(()=>validateCommand({type:'sit',duration:Infinity}),/field/);
  assert.throws(()=>validateCommand({type:'stand',chair:'other'}),/field/);
});

async function setup(t, options = {}) {
  const api = createRobotApi({ port: 0, token: '', ...options });
  const { port } = await api.start();
  t.after(() => api.close());
  const url = `http://127.0.0.1:${port}`;
  const request = async (path, body, headers = {}) => {
    const response = await fetch(url + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
      body: body === undefined ? undefined : typeof body === 'string' ? body : JSON.stringify(body),
    });
    return { status: response.status, body: await response.json(), headers: response.headers };
  };
  return { api, url, port, request };
}

async function connect(port, options = {}) {
  const ws = new WebSocket(`ws://127.0.0.1:${port}/socket${options.token ? `?token=${encodeURIComponent(options.token)}` : ''}`, { origin: options.origin ?? 'http://127.0.0.1:8816' });
  const messages = [], waiters = [];
  ws.on('message', data => {
    const message = JSON.parse(data.toString());
    if (waiters.length) waiters.shift().resolve(message); else messages.push(message);
  });
  const next = () => new Promise((resolve, reject) => {
    if (messages.length) return resolve(messages.shift());
    const waiter = { resolve: value => { clearTimeout(timer); resolve(value); } };
    const timer = setTimeout(() => { const i = waiters.indexOf(waiter); if (i >= 0) waiters.splice(i, 1); reject(new Error('Timed out waiting for WS message')); }, 3000);
    waiters.push(waiter);
  });
  await once(ws, 'open');
  return { ws, next, send: value => ws.send(JSON.stringify(value)) };
}

test('validates finite motion inputs and exact native G1 joint names', () => {
  assert.equal(JOINT_NAMES.length, 43);
  assert.deepEqual(validateCommand({ type: 'move', forward: 1, right: -1, duration: 10 }), { type: 'move', forward: 1, right: -1, duration: 10, sprint: false });
  assert.deepEqual(validateCommand({ type: 'turn', radians: -Math.PI, duration: 0.1 }), { type: 'turn', radians: -Math.PI, duration: 0.1 });
  assert.deepEqual(validateCommand({ type: 'joints', angles: { right_elbow_joint: 0.5 }, duration: 1 }), { type: 'joints', angles: { right_elbow_joint: 0.5 }, duration: 1 });
  assert.deepEqual(validateCommand({ type: 'joints', angles: { left_hand_thumb_2_joint: 0.4, right_hand_middle_1_joint: 0.8 }, duration: 1 }), { type: 'joints', angles: { left_hand_thumb_2_joint: 0.4, right_hand_middle_1_joint: 0.8 }, duration: 1 });
  for (const command of [null, [], { type: 'fly' }, { type: 'coffee', hidden: 1 },
    { type: 'move', forward: NaN, right: 0, duration: 1 }, { type: 'move', forward: 0, right: 1.1, duration: 1 },
    { type: 'move', forward: 0, right: 0, duration: 0 }, { type: 'move', forward: 0, right: 0, duration: 1, sprint: 1 },
    { type: 'turn', radians: Infinity, duration: 1 }, { type: 'turn', radians: Math.PI + 0.01, duration: 1 },
    { type: 'joints', angles: { elbow: 0.5 }, duration: 1 }, { type: 'joints', angles: { right_elbow_joint: Infinity }, duration: 1 },
    { type: 'joints', angles: {}, duration: 1 }]) assert.throws(() => validateCommand(command));
});

test('HTTP auth, browser origin, JSON/body limits and schema', async t => {
  const { request, url } = await setup(t, { token: 'test-token' });
  assert.equal((await request('/api/state')).status, 401);
  assert.equal((await request('/api/state', undefined, { Authorization: 'test-token' })).status, 401);
  const auth = { Authorization: 'Bearer test-token' };
  assert.equal((await request('/api/state', undefined, auth)).status, 200);
  assert.equal((await request('/api/state', undefined, { ...auth, Origin: 'https://malicious.example' })).status, 403);
  assert.equal((await request('/api/state', undefined, { ...auth, Origin: 'http://localhost:8816' })).status, 200);
  const schema = await request('/api/schema', undefined, auth);
  assert.equal(schema.body.commands.joints.jointNames.length, 43);
  assert.equal(schema.headers.get('access-control-allow-origin'), null);
  assert.equal((await request('/api/commands', '{', auth)).status, 400);
  assert.equal((await request('/api/commands', '{"type":"turn","radians":1e309,"duration":1}', auth)).status, 400);
  assert.equal((await request('/api/commands', 'x'.repeat(16385), auth)).status, 413);
  assert.equal((await request('/api/commands', { type: 'joints', angles: { not_a_joint: 1 }, duration: 1 }, auth)).status, 400);
  assert.equal((await request('/api/stop', { ignored: true }, auth)).status, 400);
  const response = await fetch(url + '/api/state', { headers: auth });
  assert.equal(response.headers.get('cache-control'), 'no-store');
});

test('queues without simulator, dispatches FIFO, snapshots and acknowledgements reflect observed state', async t => {
  const { request, port } = await setup(t);
  const first = await request('/api/commands', { type: 'move', forward: 1, right: 0, duration: 1 });
  const second = await request('/api/commands', { type: 'coffee' });
  assert.equal(first.status, 202); assert.equal(first.body.status, 'queued');
  assert.equal((await request('/api/state')).body.connected, false);
  const browser = await connect(port);
  browser.send({ type: 'hello', role: 'simulator' });
  assert.equal((await browser.next()).type, 'welcome');
  const deliveredFirst = await browser.next();
  assert.equal(deliveredFirst.id, first.body.id);
  assert.equal(deliveredFirst.command.sprint, false);
  assert.equal((await request('/api/commands/' + first.body.id)).body.status, 'queued');
  browser.send({ type: 'ack', id: first.body.id, status: 'running' });
  browser.send({ type: 'invalid' }); await browser.next();
  assert.equal((await request('/api/commands/' + first.body.id)).body.status, 'running');
  assert.equal((await request('/api/commands/' + second.body.id)).body.status, 'queued');
  browser.send({ type: 'state', state: { position: [1, 0, 3], heading: 0.5, holdingCoffee: false } });
  browser.send({ type: 'ack', id: first.body.id, status: 'completed' });
  assert.equal((await browser.next()).id, second.body.id);
  const state = await request('/api/state');
  assert.equal(state.body.connected, true);
  assert.deepEqual(state.body.state.position, [1, 0, 3]);
  assert.ok(state.body.stateUpdatedAt);
  browser.send({ type: 'ack', id: second.body.id, status: 'rejected', message: 'Cup unreachable' });
  // Same-socket reply creates an ordering barrier for the rejection acknowledgement.
  browser.send({ type: 'invalid' });
  await browser.next();
  const status = await request('/api/commands/' + second.body.id);
  assert.equal(status.body.status, 'rejected'); assert.equal(status.body.message, 'Cup unreachable');
  assert.equal((await request('/api/commands/' + first.body.id)).body.status, 'completed');
  assert.equal((await request('/api/commands/no-such-id')).status, 404);
  browser.ws.send('{"type":"state","state":{"x":1e309}}');
  assert.equal((await browser.next()).type, 'error');
  assert.deepEqual((await request('/api/state')).body.state.position, [1, 0, 3]);
});

test('queue limit and stop interrupt active command and cancel waiting commands', async t => {
  const { request, port } = await setup(t);
  const queued = [];
  for (let i = 0; i < 32; i++) queued.push((await request('/api/commands', { type: 'coffee' })).body);
  assert.equal((await request('/api/commands', { type: 'coffee' })).status, 429);
  const browser = await connect(port);
  browser.send({ type: 'hello', role: 'simulator' });
  await browser.next();
  assert.equal((await browser.next()).id, queued[0].id);
  const stopped = await request('/api/stop', {});
  assert.equal(stopped.status, 202);
  const stopCommand = await browser.next();
  assert.equal(stopCommand.id, stopped.body.id); assert.equal(stopCommand.command.type, 'stop');
  for (const command of queued) assert.equal((await request('/api/commands/' + command.id)).body.status, 'rejected');
  assert.equal((await request('/api/state')).body.queuedCommands, 0);
  browser.send({ type: 'ack', id: queued[0].id, status: 'rejected', message: 'Stopped by browser' });
  browser.send({ type: 'ack', id: stopped.body.id, status: 'completed' });
  browser.send({ type: 'invalid' }); assert.equal((await browser.next()).error, 'Unknown simulator message');
  assert.equal((await request('/api/commands/' + stopped.body.id)).body.status, 'completed');
});

test('WS authenticates, rejects second simulator, and disconnection rejects in-flight work', async t => {
  const { request, port } = await setup(t, { token: 'test-token' });
  for (const options of [{ token: 'wrong', origin: 'http://localhost:8816' }, { token: 'test-token', origin: 'https://malicious.example' }]) {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/socket?token=${options.token}`, { origin: options.origin });
    const [error] = await once(ws, 'error');
    assert.match(error.message, /Unexpected server response: (401|403)/);
  }
  const browser = await connect(port, { token: 'test-token' });
  browser.send({ type: 'hello', role: 'simulator' }); await browser.next();
  const replacement = await connect(port, { token: 'test-token' });
  const closed = once(replacement.ws, 'close');
  replacement.send({ type: 'hello', role: 'simulator' });
  assert.equal((await closed)[0], 1008);
  const command = await request('/api/commands', { type: 'turn', radians: 1, duration: 1 }, { Authorization: 'Bearer test-token' });
  const pending = await request('/api/commands', { type: 'coffee' }, { Authorization: 'Bearer test-token' });
  assert.equal((await browser.next()).id, command.body.id);
  const browserClosed = once(browser.ws, 'close'); browser.ws.close(); await browserClosed;
  const status = await request('/api/commands/' + command.body.id, undefined, { Authorization: 'Bearer test-token' });
  assert.equal(status.body.status, 'rejected'); assert.equal(status.body.message, 'Simulator disconnected');
  assert.equal((await request('/api/commands/' + pending.body.id, undefined, { Authorization: 'Bearer test-token' })).body.status, 'rejected');
  assert.equal((await request('/api/state', undefined, { Authorization: 'Bearer test-token' })).body.connected, false);
});
