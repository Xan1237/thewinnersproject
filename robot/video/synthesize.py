#!/usr/bin/env python3
"""Generate individually reviewable Vel/Tim narration, with no embedded credentials.

python synthesize.py --script narration.json --out ../outputs/pitch/audio

Requires ffprobe and ELEVENLABS_API_KEY for uncached requests. Override the
existing clones with --vel-voice/--tim-voice or ELEVENLABS_VEL_VOICE_ID and
ELEVENLABS_TIM_VOICE_ID. A changed text, voice, model or format creates a new
cache entry. Optional --transcripts accepts independently obtained ASR text;
the script never supplies the intended words to a recognizer.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request


DEFAULT_VOICES = {"vel": "OIYncJRayP2q0OJ3GBdV", "tim": "hLWALlktcoTsKNlsjgZo"}


def read_segments(path):
    data = json.loads(Path(path).read_text())
    segments = data if isinstance(data, list) else data["segments"]
    if not segments:
        raise ValueError("Narration has no segments")
    seen = set()
    previous = 0.0
    for segment in segments:
        sid = str(segment["id"])
        if not re.fullmatch(r"[\w-]+", sid) or sid in seen:
            raise ValueError(f"Invalid or duplicate segment id: {sid!r}")
        seen.add(sid)
        start, end = float(segment["start"]), float(segment["end"])
        if start < previous - 0.001 or end <= start:
            raise ValueError(f"Invalid or overlapping timing for {sid}")
        previous = end
        if segment["speaker"].lower() not in DEFAULT_VOICES:
            raise ValueError(f"Unknown speaker in {sid}: {segment['speaker']}")
        if not isinstance(segment["text"], str) or not segment["text"].strip():
            raise ValueError(f"Empty narration text in {sid}")
    return segments


def duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def words(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?", text.lower())


def word_error_rate(expected, actual):
    """Levenshtein distance; independent transcript, without forced alignment."""
    source, target = words(expected), words(actual)
    row = list(range(len(target) + 1))
    for i, word in enumerate(source, 1):
        new = [i]
        for j, other in enumerate(target, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = new
    return row[-1] / max(1, len(source))


def synthesize(text, voice, model, output_format, api_key):
    payload = {"inputs": [{"text": text, "voice_id": voice}], "model_id": model}
    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/text-to-dialogue?output_format=" + output_format,
        data=json.dumps(payload).encode(), method="POST",
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
            content_type = response.headers.get("Content-Type", "")
            request_id = response.headers.get("request-id")
    except urllib.error.HTTPError as error:
        # Do not print request headers or credentials. Preserve a bounded API reason.
        reason = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"ElevenLabs HTTP {error.code}: {reason}") from None
    if not data or "json" in content_type:
        raise RuntimeError(f"ElevenLabs returned no audio ({content_type})")
    return data, request_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vel-voice", default=os.environ.get("ELEVENLABS_VEL_VOICE_ID", DEFAULT_VOICES["vel"]))
    parser.add_argument("--tim-voice", default=os.environ.get("ELEVENLABS_TIM_VOICE_ID", DEFAULT_VOICES["tim"]))
    parser.add_argument("--model", default="eleven_v4")
    parser.add_argument("--lead", type=float, default=0.3)
    parser.add_argument("--transcripts", type=Path, help="JSON mapping each segment id to independent ASR text")
    parser.add_argument("--max-word-error-rate", type=float, default=0.12)
    args = parser.parse_args()
    segments = read_segments(args.script)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out.resolve()
    transcripts = json.loads(args.transcripts.read_text()) if args.transcripts else None
    voices = {"vel": args.vel_voice, "tim": args.tim_voice}
    metadata = {"script": str(args.script.resolve()), "model": args.model, "lead": args.lead, "segments": {}}
    failures = []
    for segment in segments:
        sid, text = str(segment["id"]), segment["text"].strip()
        voice = voices[segment["speaker"].lower()]
        settings = {"text": text, "voice_id": voice, "model_id": args.model, "output_format": "mp3_44100_128"}
        cache_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        audio_path = output / (cache_key + ".mp3")
        sidecar = output / (cache_key + ".json")
        cached = audio_path.exists() and sidecar.exists()
        if not cached:
            key = os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                raise RuntimeError(f"ELEVENLABS_API_KEY is required for uncached segment {sid}")
            audio, request_id = synthesize(text, voice, args.model, settings["output_format"], key)
            temporary = audio_path.with_suffix(".partial.mp3")
            temporary.write_bytes(audio)
            try:
                seconds = duration(temporary)
                temporary.replace(audio_path)
            finally:
                temporary.unlink(missing_ok=True)
            sidecar.write_text(json.dumps({**settings, "duration": seconds, "request_id": request_id,
                                          "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + "\n")
        seconds = duration(audio_path)
        slot = float(segment["end"]) - float(segment["start"])
        entry = {"file": str(audio_path), "duration": seconds, "speaker": segment["speaker"],
                 "voice_id": voice, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "cache_key": cache_key, "slot": slot, "quality": "independent transcript not checked"}
        if seconds + args.lead > slot + 0.001:
            failures.append(f"{sid}: speech {seconds:.2f}s + lead {args.lead:.2f}s exceeds {slot:.2f}s slot")
        if transcripts is not None:
            transcript = transcripts.get(sid)
            if isinstance(transcript, dict):
                transcript = transcript.get("text")
            if not isinstance(transcript, str):
                failures.append(f"{sid}: missing independent transcript")
            else:
                rate = word_error_rate(text, transcript)
                entry.update(transcript=transcript, word_error_rate=rate, quality="independent transcript checked")
                if rate > args.max_word_error_rate:
                    failures.append(f"{sid}: transcript word error rate {rate:.1%} exceeds {args.max_word_error_rate:.1%}")
        metadata["segments"][sid] = entry
        (output / "audio-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{sid}: {seconds:.2f}s / {slot:.2f}s, {'cached' if cached else 'generated'}", flush=True)
    metadata["failures"] = failures
    metadata["timing_passed"] = not any("exceeds" in failure and "slot" in failure for failure in failures)
    (output / "audio-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Audio manifest: {output / 'audio-manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
