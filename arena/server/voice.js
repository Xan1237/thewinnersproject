// Speech to text: Microsoft MAI-Transcribe 2 through OpenRouter (POST /api/v1/audio/transcriptions).
// Verified on 2026-10-11: JSON body with base64 `input_audio`; the provider accepts WAV (and MP3) but
// answers 400 for WebM/Ogg/M4A, so the browser sends 16 kHz mono WAV (see src/voice.js).
const URL = 'https://openrouter.ai/api/v1/audio/transcriptions';
export const STT_MODEL = process.env.STT_MODEL ?? 'microsoft/mai-transcribe-2';

export async function transcribe(audio, contentType = 'audio/wav', { apiKey = process.env.OPENROUTER_API_KEY, fetchImpl = fetch, model = STT_MODEL } = {}) {
  if (!apiKey) throw Object.assign(new Error('OPENROUTER_API_KEY is not set for the sim server, so speech cannot be transcribed. Type the order instead (press /).'), { status: 501 });
  const format = /wav/i.test(contentType) ? 'wav' : /mpeg|mp3/i.test(contentType) ? 'mp3' : null;
  if (!format) throw Object.assign(new Error(`unsupported audio type ${contentType}; send audio/wav`), { status: 415 });
  const started = Date.now();
  const res = await fetchImpl(URL, {
    method: 'POST',
    headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model, input_audio: { data: Buffer.from(audio).toString('base64'), format }, language: 'en' }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw Object.assign(new Error(`speech-to-text failed (${res.status})`), { status: 502 });
  const body = await res.json();
  return { text: String(body.text ?? '').trim(), model, latency_s: (Date.now() - started) / 1000, cost_usd: body.usage?.cost ?? null };
}
