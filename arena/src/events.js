// Server-Sent Events client for the sim server. Auto-reconnects, replays history on connect.
export function connectEvents(onEvent, onStatus) {
  let es;
  const open = () => {
    es = new EventSource('/sim/events');
    es.onopen = () => onStatus(true);
    es.onmessage = m => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.error('bad event', e); } };
    es.onerror = () => { onStatus(false); es.close(); setTimeout(open, 1500); };
  };
  open();
}
