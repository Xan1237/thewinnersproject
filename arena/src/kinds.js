/** What kind of decision-maker a model name is: a frontier LLM, a scripted teacher, or our trained policy. */
export const kindOf = model => (/learned|trained|bc-mlp/i.test(model) ? 'trained' : /^(teacher|scripted|expert)/i.test(model) ? 'scripted' : 'frontier');
export const KIND_LABEL = { frontier: 'FRONTIER', scripted: 'SCRIPTED TEACHER', trained: 'TRAINED MODEL' };
