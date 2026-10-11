// Turn what the operator said (already transcribed) into one task for every robot.
import { EMOTES, TASK_HELP, resolveTask } from './core.js';

const OPENROUTER = 'https://openrouter.ai/api/v1/chat/completions';
export const INTERPRETER_MODEL = process.env.INTERPRETER_MODEL ?? 'qwen/qwen3.8-27b';

/** Objects worth naming to the interpreter (the 31 folding chairs are one group). */
function objectList(world) {
  return world.objects
    .filter(o => o.kind !== 'folding' && o.kind !== 'platform' && o.id !== 'coffee_cup')
    .map(o => `${o.id} (${o.name})`).join(', ');
}

export function parseWithRules(text, world) {
  const t = text.toLowerCase();
  for (const e of Object.keys(EMOTES)) if (new RegExp(`\\b${e}`).test(t)) return { task: e, say: `Everyone: ${e}!` };
  if (/\b(coffee|cup)\b/.test(t) && /\b(sit|seat|chair)\b/.test(t)) return { task: 'talk', say: 'Everyone: grab a coffee, then find a seat.' };
  if (/\b(coffee|cup)\b/.test(t)) return { task: 'fetch_coffee', say: 'Everyone: fetch the coffee.' };
  const named = world.objects
    .filter(o => o.kind !== 'folding' && o.kind !== 'platform' && o.id !== 'coffee_cup')
    .sort((x, y) => y.name.length - x.name.length)
    .find(o => o.name.split(' ').every(w => new RegExp(`\\b${w}`).test(t)));
  if (/\bsit\b|\bseat\b|\bsit down\b/.test(t)) {
    if (named && (named.kind === 'chair')) return { task: `sit:${named.id}`, say: `Everyone: sit on the ${named.name} (one robot per chair).` };
    return { task: 'sit', say: 'Everyone: sit down, one robot per chair.' };
  }
  if (named) return { task: `goto:${named.id}`, say: `Everyone: go to the ${named.name}.` };
  if (/\b(stand|stop|wait|idle|freeze)\b/.test(t)) return { task: 'idle', say: 'Everyone: stand by.' };
  return null;
}

export async function interpretWithLlm(text, world, apiKey) {
  const system = `You translate a spoken order to a team of humanoid robots into exactly one task.
Tasks (all robots get the same task): ${TASK_HELP}.
- "sit" makes every robot take a different free chair; "sit:<chair id>" prefers that chair, the rest take other chairs.
- jump, wave, dance, cheer are animations the robots play where they stand.
- idle means stand still.
Objects you may target with goto:<id>: ${objectList(world)}.
Reply with ONLY JSON: {"task": "<task>", "say": "<one short sentence confirming the order to the operator>"}.
If the order is not something the robots can do, reply {"task": null, "say": "<short explanation>"}.`;
  const res = await fetch(OPENROUTER, {
    method: 'POST',
    headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model: INTERPRETER_MODEL,
      messages: [{ role: 'system', content: system }, { role: 'user', content: text }],
      max_tokens: 200,
      temperature: 0,
      reasoning: { enabled: false },
      provider: { only: ['cerebras'], allow_fallbacks: false },
    }),
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) throw new Error(`interpreter HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const content = (await res.json()).choices?.[0]?.message?.content ?? '';
  const m = /\{[\s\S]*\}/.exec(content);
  if (!m) throw new Error(`interpreter returned no JSON: ${content.slice(0, 120)}`);
  const out = JSON.parse(m[0]);
  return { task: out.task ?? null, say: String(out.say ?? '').slice(0, 200) };
}

/** Text in, validated command out: { task, say, via } (task null = could not understand). */
export async function interpret(text, world, apiKey = process.env.OPENROUTER_API_KEY) {
  let result = null, via = 'rules';
  if (apiKey) {
    try { result = await interpretWithLlm(text, world, apiKey); via = INTERPRETER_MODEL; } catch (e) { result = null; via = `rules (LLM failed: ${String(e.message).slice(0, 80)})`; }
  }
  if (!result?.task && !(result && result.task === null && result.say)) result = parseWithRules(text, world);
  if (result?.task) {
    try { resolveTask(world, result.task); } catch { result = parseWithRules(text, world); }
  }
  if (!result) return { task: null, say: "Sorry, I didn't catch an order the robots can do.", via };
  return { ...result, via };
}
