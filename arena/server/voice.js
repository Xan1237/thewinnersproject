// Speech to text. Filled in once the model/endpoint is verified (see arena/README.md).
export async function transcribe() {
  throw Object.assign(new Error('speech-to-text is not configured'), { status: 501 });
}
