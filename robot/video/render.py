#!/usr/bin/env python3
"""Edit real capture clips into a captioned, fixed-length walkthrough.

python render.py --script narration.json --media captures.json \
  --audio ../outputs/pitch/audio/audio-manifest.json \
  --work ../outputs/pitch/edit --out ../outputs/pitch/final.mp4

Requires ffmpeg, ffprobe and Pillow. Captures JSON maps segment IDs to a list of
{file, start, duration}; relative paths resolve beside the manifest. Clips play
once at their recorded speed. Only the final frame may be held (at most three
seconds by default), and the verification report records that hold. Speech is
never sped up or truncated. Optional segment `title` adds a short chapter badge.
Optional `captions` use local segment offsets {text,start,end}; otherwise caption
chunks are approximately timed by word count, explicitly recorded as estimates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def media_duration(path):
    return float(probe(path)["format"]["duration"])


def resolve_file(value, manifest):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def load_font(path, size):
    choices = [path] if path else [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in choices:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise RuntimeError("No suitable font found; specify --font /path/to/font.ttf")


def wrap_text(text, draw, font, width):
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def caption_image(text, path, font, width, height):
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    # The right-hand telemetry panel stays uncovered. At 1080p this is 1120px wide.
    left, box_width, margin = round(width * 0.0292), round(width * 0.5833), round(width * 0.0167)
    lines = wrap_text(text, draw, font, box_width - 2 * margin)
    if len(lines) > 2:
        raise ValueError(f"Caption needs more than two lines; split it: {text}")
    line_height = round(font.size * 1.34)
    padding = round(height * 0.015)
    box_height = len(lines) * line_height + padding * 2
    top = height - round(height * 0.055) - box_height
    draw.rounded_rectangle((left, top, left + box_width, top + box_height), radius=round(height * 0.012),
                           fill=(11, 18, 27, 224))
    for index, line in enumerate(lines):
        draw.text((left + margin, top + padding + index * line_height), line, font=font, fill=(248, 250, 253, 255))
    image.save(path)


def title_image(text, path, font, width, height):
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    left, top, padding = round(width * 0.0292), round(height * 0.041), round(width * 0.0146)
    text_width = draw.textbbox((0, 0), text, font=font)[2]
    if text_width > width * 0.55:
        raise ValueError(f"Chapter title is too wide: {text}")
    box_height = round(font.size * 1.7)
    draw.rounded_rectangle((left, top, left + text_width + 2 * padding, top + box_height),
                           radius=round(height * 0.011), fill=(11, 18, 27, 228))
    draw.rectangle((left, top + 12, left + 4, top + box_height - 12), fill=(99, 211, 207, 255))
    draw.text((left + padding, top + round(font.size * 0.22)), text, font=font, fill=(248, 250, 253, 255))
    image.save(path)


def caption_chunks(text, lead, seconds):
    """Short script chunks; estimated timing is not presented as ASR alignment."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + 12, len(words))
        # Prefer sentence punctuation near the target length.
        for boundary in range(end, start + 5, -1):
            if words[boundary - 1].endswith((".", "?", "!", ";", ":")):
                end = boundary
                break
        chunks.append({"text": " ".join(words[start:end]),
                       "start": lead + seconds * start / len(words),
                       "end": lead + seconds * end / len(words)})
        start = end
    return chunks


def concat_list(paths, output):
    # ffconcat uses its own quoting; this is not shell quoting.
    output.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in paths))


def validate_script(segments, duration, fps):
    cursor, ids = 0.0, set()
    for segment in segments:
        sid = str(segment["id"])
        if sid in ids or not sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sid):
            raise ValueError(f"Invalid or duplicate segment id: {sid!r}")
        ids.add(sid)
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start + end) or abs(start - cursor) > 0.001 or end <= start:
            raise ValueError(f"Segments must cover the timeline without gaps or overlaps: {sid}")
        if abs(end * fps - round(end * fps)) > 0.001:
            raise ValueError(f"Segment {sid} end is not on a video frame boundary")
        cursor = end
    if not segments or abs(cursor - duration) > 0.001:
        raise ValueError(f"Narration must end at {duration:.3f}s; got {cursor:.3f}s")


def render_segment(segment, clips, audio_entry, args, fonts):
    sid = str(segment["id"])
    slot = float(segment["end"]) - float(segment["start"])
    audio_path = resolve_file(audio_entry["file"], args.audio)
    if not audio_path.is_file():
        raise ValueError(f"Missing narration audio for {sid}: {audio_path}")
    speech = media_duration(audio_path)
    expected_hash = hashlib.sha256(segment["text"].strip().encode()).hexdigest()
    if audio_entry.get("text_sha256") and audio_entry["text_sha256"] != expected_hash:
        raise ValueError(f"Narration audio for {sid} was synthesized from different text")
    if speech + args.lead > slot + 0.001:
        raise ValueError(f"{sid}: speech {speech:.2f}s + lead {args.lead:.2f}s exceeds {slot:.2f}s slot; shorten the line")
    selected, remaining = [], slot
    for clip in clips:
        path = resolve_file(clip["file"], args.media)
        total = media_duration(path)
        start = float(clip.get("start", 0))
        requested = float(clip.get("duration", total - start))
        if start < 0 or requested <= 0 or start + requested > total + 0.06:
            raise ValueError(f"{sid}: invalid capture range {start:.3f}+{requested:.3f}s in {path} ({total:.3f}s)")
        length = min(requested, remaining, total - start)
        if length <= 0:
            raise ValueError(f"{sid}: empty clip in {path}")
        selected.append({"file": str(path), "start": start, "duration": length})
        remaining -= length
        if remaining < 1 / args.fps / 2:
            remaining = 0
            break
    hold = max(0, remaining)
    if not selected:
        raise ValueError(f"No capture clips for {sid}")
    if hold > args.max_hold + 0.001:
        raise ValueError(f"{sid}: footage is {hold:.2f}s short; capture more or supply additional real clips (--max-hold {args.max_hold})")
    directory = args.work / sid
    directory.mkdir(parents=True, exist_ok=True)
    overlays = []
    if segment.get("title"):
        title_path = directory / "chapter.png"
        title_image(segment["title"], title_path, fonts[1], args.width, args.height)
        overlays.append((title_path, 0, min(4.0, slot)))
    explicit = "captions" in segment
    captions = segment.get("captions") if explicit else caption_chunks(segment["text"], args.lead, speech)
    previous = 0.0
    for index, caption in enumerate(captions):
        start, end = float(caption["start"]), float(caption["end"])
        if start < previous - 0.001 or end <= start or end > slot + 0.001:
            raise ValueError(f"{sid}: invalid caption timing")
        previous = end
        path = directory / f"caption-{index:03d}.png"
        caption_image(caption["text"], path, fonts[0], args.width, args.height)
        overlays.append((path, start, end))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    for clip in selected:
        command += ["-ss", f"{clip['start']:.6f}", "-t", f"{clip['duration']:.6f}", "-i", clip["file"]]
    for path, _, _ in overlays:
        command += ["-loop", "1", "-framerate", str(args.fps), "-i", str(path)]
    filters, labels = [], []
    for index, clip in enumerate(selected):
        tail = f",tpad=stop_mode=clone:stop_duration={hold:.6f}" if index == len(selected) - 1 and hold else ""
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
                       f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=0x0b121b,"
                       f"setsar=1,fps={args.fps},format=yuv420p,trim=duration={clip['duration']:.6f}{tail}[clip{index}]")
        labels.append(f"[clip{index}]")
    filters.append("".join(labels) + f"concat=n={len(selected)}:v=1:a=0[base]")
    source = "base"
    for index, (_, start, end) in enumerate(overlays):
        dest = f"overlay{index}"
        filters.append(f"[{source}][{len(selected) + index}:v]overlay=0:0:enable='gte(t,{start:.6f})*lt(t,{end:.6f})':"
                       f"eof_action=pass[{dest}]")
        source = dest
    video = directory / "video.mp4"
    command += ["-filter_complex", ";".join(filters), "-map", f"[{source}]", "-an", "-frames:v", str(round(slot * args.fps)),
                "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p", str(video)]
    print(f"Editing {sid}: {len(selected)} capture cuts, {hold:.2f}s final-frame hold", flush=True)
    run(command)
    wav = directory / "audio.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", audio_path, "-af",
         f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,adelay={round(args.lead * 1000)}:all=1,apad,atrim=duration={slot:.6f},asetpts=PTS-STARTPTS",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", wav])
    if abs(media_duration(video) - slot) > 1 / args.fps + 0.005:
        raise RuntimeError(f"{sid}: edited video duration does not match its slot")
    return video, wav, {"id": sid, "clips": selected, "final_frame_hold_seconds": hold, "speech_duration": speech,
                        "caption_timing": "provided local offsets" if explicit else "estimated by script word count", "captions": captions}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=240)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--lead", type=float, default=0.3)
    parser.add_argument("--max-hold", type=float, default=3)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()
    for name in ("script", "media", "audio", "out", "work"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.duration <= 0 or args.fps <= 0 or args.lead < 0 or args.max_hold < 0:
        raise ValueError("Duration/fps must be positive, lead/max-hold nonnegative")
    if args.width % 2 or args.height % 2 or args.width < 320 or args.height < 180:
        raise ValueError("Video dimensions must be even and at least 320x180")
    data = json.loads(args.script.read_text())
    segments = data if isinstance(data, list) else data["segments"]
    validate_script(segments, args.duration, args.fps)
    captures = json.loads(args.media.read_text())
    captures = captures.get("segments", captures)
    audio = json.loads(args.audio.read_text())
    audio = audio.get("segments", audio)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fonts = (load_font(args.font, max(15, round(args.height * 0.0352))),
             load_font(args.font, max(13, round(args.height * 0.0278))))
    videos, waves, provenance = [], [], []
    for segment in segments:
        sid = str(segment["id"])
        if sid not in captures or sid not in audio:
            raise ValueError(f"Missing capture/audio manifest entry for {sid}")
        video, wave, evidence = render_segment(segment, captures[sid], audio[sid], args, fonts)
        videos.append(video)
        waves.append(wave)
        provenance.append(evidence)
    video_list, audio_list = args.work / "video-concat.txt", args.work / "audio-concat.txt"
    concat_list(videos, video_list)
    concat_list(waves, audio_list)
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0", "-i", video_list,
         "-f", "concat", "-safe", "0", "-i", audio_list, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", str(args.duration), "-movflags", "+faststart", args.out])
    info = probe(args.out)
    video_stream = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    if (video_stream["codec_name"], video_stream["width"], video_stream["height"], video_stream["avg_frame_rate"], audio_stream["codec_name"]) != (
        "h264", args.width, args.height, f"{args.fps}/1", "aac"):
        raise RuntimeError("Output codecs, dimensions or frame rate failed validation")
    if abs(float(info["format"]["duration"]) - args.duration) > 0.06:
        raise RuntimeError("Output duration failed validation")
    decode = ["ffmpeg", "-v", "error", "-xerror", "-i", str(args.out), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"]
    run(decode)
    report = {"output": str(args.out), "duration": float(info["format"]["duration"]), "width": args.width, "height": args.height,
              "fps": args.fps, "video_codec": "h264", "audio_codec": "aac", "full_decode_passed": True,
              "decode_command": shlex.join(decode), "probe": info, "segments": provenance,
              "source_script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest()}
    report_path = args.out.with_suffix(".verification.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {args.out}: {args.duration:.3f}s, {args.width}x{args.height}, {args.fps}fps, H.264/AAC; full decode passed")
    print(f"Verification report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
