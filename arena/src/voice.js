// One mic button: click to talk, click again to send. The server transcribes the speech, turns it
// into one task and every agent re-tasks its robot (see server/voice.js, server/commands.js).
const $ = id => document.getElementById(id);

export function setupVoice() {
  const btn = $('mic'), status = $('micstatus'), heard = $('heard');
  let rec = null, chunks = [], stream = null, raf = 0, audioCtx = null;

  const say = (text, cls = '') => { heard.textContent = text; heard.className = cls; heard.hidden = !text; };

  async function start() {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { say('Microphone blocked. Allow it for this page, or press / to type an order.', 'bad'); return; }
    chunks = [];
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(m => MediaRecorder.isTypeSupported(m));
    rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = () => send(new Blob(chunks, { type: rec.mimeType }));
    rec.start();
    btn.classList.add('live'); status.textContent = 'Listening… click to send'; say('');
    audioCtx = new AudioContext();
    const an = audioCtx.createAnalyser(); an.fftSize = 256;
    audioCtx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.frequencyBinCount);
    const tick = () => {
      an.getByteTimeDomainData(buf);
      let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
      btn.style.setProperty('--level', Math.min(peak / 60, 1).toFixed(2));
      raf = requestAnimationFrame(tick);
    };
    tick();
  }

  function stop() {
    cancelAnimationFrame(raf); btn.style.setProperty('--level', 0);
    rec?.state === 'recording' && rec.stop();
    stream?.getTracks().forEach(t => t.stop());
    audioCtx?.close();
    btn.classList.remove('live');
  }

  async function send(blob) {
    status.textContent = 'Understanding…'; btn.disabled = true;
    try {
      const r = await fetch('/sim/voice', { method: 'POST', headers: { 'content-type': blob.type || 'audio/webm' }, body: blob });
      show(await r.json(), r.ok);
    } catch (e) { say(`Could not reach the sim: ${e.message}`, 'bad'); }
    btn.disabled = false; status.textContent = 'Click to give an order';
  }

  function show(res, ok) {
    if (!ok || res.error) return say(res.error || 'Something went wrong', 'bad');
    const you = res.text ? `“${res.text}”` : '';
    say(`${you}${you && res.say ? '  →  ' : ''}${res.say || ''}`, res.task ? 'ok' : 'bad');
  }

  btn.addEventListener('click', () => (btn.classList.contains('live') ? stop() : start()));

  // Typing fallback (press /): same path as speech, minus the transcription.
  const box = $('typebox');
  addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement !== box) { e.preventDefault(); box.hidden = false; box.focus(); }
    if (e.key === 'Escape') { box.hidden = true; box.blur(); }
  });
  box.addEventListener('keydown', async e => {
    if (e.key !== 'Enter' || !box.value.trim()) return;
    const text = box.value.trim(); box.value = ''; box.hidden = true;
    const r = await fetch('/sim/command', { method: 'POST', body: JSON.stringify({ text }) });
    show(await r.json(), r.ok);
  });

  // Anything said (by anyone) shows up for everybody watching.
  return ev => {
    if (ev.type === 'heard' && ev.text) say(`“${ev.text}”`, '');
    if (ev.type === 'command') show(ev, true);
  };
}
