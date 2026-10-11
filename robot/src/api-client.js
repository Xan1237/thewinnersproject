/** Browser owns the simulation; the local API server only validates and queues commands. */
export class SimulationApi {
  constructor(simulation) {
    this.sim = simulation;
    this.socket = null;
    this.active = null;
    this.stateElapsed = 0;
  }

  connect(token = '') {
    if (this.socket) return;
    const url = new URL('/socket', location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (token) url.searchParams.set('token', token);
    const socket = this.socket = new WebSocket(url);
    socket.onopen = () => {
      socket.send(JSON.stringify({ type: 'hello', role: 'simulator' }));
      this.sim.status('API подключён');
      this.sendState();
    };
    socket.onmessage = event => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'command') this.start(message.id, message.command);
        else if (message.type === 'cancel' && this.active?.id === message.id) this.cancel('Команда превысила время ожидания');
        else if (message.type === 'error') this.sim.status(message.message || message.error || 'Ошибка API');
      } catch (error) { this.sim.status(error.message); }
    };
    socket.onclose = () => {
      this.cancel('Соединение API закрыто');
      this.socket = null;
      this.sim.status('API отключён');
    };
    socket.onerror = () => this.sim.status('Запусти npm run dev для API');
  }

  disconnect() { this.cancel('API отключён пользователем'); this.socket?.close(); }
  send(message) { if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message)); }
  sendState() { this.send({ type: 'state', state: this.sim.state() }); }
  ack(id, status, message) { this.send({ type: 'ack', id, status, ...(message ? { message } : {}) }); }

  cancel(message = 'Ручное управление отменило команду') {
    const active = this.active;
    this.active = null;
    this.sim.stop();
    if (active) this.ack(active.id, 'rejected', message);
  }

  start(id, command) {
    if (command.type === 'stop') {
      this.cancel('Остановлено через API');
      this.ack(id, 'completed');
      this.sendState();
      return;
    }
    if (this.active) { this.ack(id, 'rejected', 'Симулятор уже выполняет команду'); return; }
    try {
      this.sim.control(command.type);
      const active = { id, command, elapsed: 0 };
      if (command.type === 'coffee') this.sim.coffee();
      else if (command.type === 'sit') this.sim.sit();
      else if (command.type === 'stand') this.sim.stand();
      else if (command.type === 'turn') active.heading = this.sim.state().heading;
      else if (command.type === 'joints') active.angles = this.sim.validateAngles(command.angles);
      this.active = active;
      this.ack(id, 'running');
      this.sim.status(`API: ${command.type}`);
    } catch (error) { this.ack(id, 'rejected', error.message); }
  }

  beforeUpdate(dt) {
    const active = this.active;
    if (!active) return;
    active.elapsed += dt;
    const { command } = active;
    if (command.type === 'move') this.sim.drive(command.forward, command.right, command.sprint);
    else if (command.type === 'turn') {
      this.sim.turn(active.heading + command.radians * Math.min(active.elapsed / command.duration, 1));
    }
  }

  afterUpdate(dt) {
    const active = this.active;
    if (active) {
      if (active.angles) this.sim.setAngles(active.angles);
      const finished = active.command.type === 'coffee' ? !this.sim.state().coffeeActive : ['sit','stand'].includes(active.command.type) ? !this.sim.state().seatingActive : active.elapsed >= active.command.duration;
      if (finished) {
        this.active = null;
        this.sim.stopDrive();
        this.sendState();
        this.ack(active.id, 'completed');
        this.sim.status(`API: ${active.command.type} выполнено`);
      }
    }
    this.stateElapsed += dt;
    if (this.stateElapsed >= .1) { this.stateElapsed = 0; this.sendState(); }
  }
}
