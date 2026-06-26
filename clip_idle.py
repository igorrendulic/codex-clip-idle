#!/usr/bin/env python3
"""Remove visually idle agent-waiting spans from screen recordings."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PRESETS = {
    "codex": {
        "threshold": 0.0035,
        "min_duration": 1.0,
        "padding": 0.1,
        "sample_fps": 2.0,
        "scale_width": 320,
        "crop": "full",
    },
    "claude": {
        "threshold": 0.0035,
        "min_duration": 1.0,
        "padding": 0.1,
        "sample_fps": 2.0,
        "scale_width": 320,
        "crop": "full",
    },
    "generic": {
        "threshold": 0.0025,
        "min_duration": 4.0,
        "padding": 0.2,
        "sample_fps": 2.0,
        "scale_width": 320,
        "crop": "full",
    },
}

OUTPUT_FORMATS = ("gif", "mp4")
DEFAULT_AGGRESSIVENESS = 1.0
DEFAULT_GIF_FPS = 12
DEFAULT_GIF_WIDTH = 960


@dataclass(frozen=True)
class Probe:
    timestamp: float
    difference: float
    quiet: bool


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    confidence: float
    mean_difference: float
    samples: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


class ProgressBar:
    def __init__(self, label: str, *, total: float, enabled: bool | None = None) -> None:
        self.label = label
        self.total = max(total, 1e-9)
        self.enabled = sys.stderr.isatty() if enabled is None else enabled
        self.value = 0.0
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0

    def update(self, value: float, *, force: bool = False) -> None:
        if not self.enabled:
            return
        self.value = max(0.0, min(value, self.total))
        now = time.monotonic()
        if not force and now - self.last_rendered_at < 0.1 and self.value < self.total:
            return
        self.last_rendered_at = now
        fraction = self.value / self.total
        width = 28
        filled = min(width, round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.started_at
        percent = round(fraction * 100)
        print(f"\r{self.label:<10} [{bar}] {percent:>3}% {elapsed:>5.1f}s", end="", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.update(self.total, force=True)
        print(file=sys.stderr, flush=True)


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe_video(path: Path) -> VideoInfo:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    duration = float(data.get("format", {}).get("duration") or 0.0)
    video_stream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return VideoInfo(
        duration=duration,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        has_audio=has_audio,
    )


def crop_filter(crop: str, width: int, height: int) -> str:
    if crop == "full":
        return ""
    if crop == "center":
        return f"crop={math.floor(width * 0.8)}:{math.floor(height * 0.8)}:{math.floor(width * 0.1)}:{math.floor(height * 0.1)},"
    if crop == "terminal":
        return f"crop={math.floor(width * 0.92)}:{math.floor(height * 0.86)}:{math.floor(width * 0.04)}:{math.floor(height * 0.10)},"
    raise ValueError(f"Unknown crop preset: {crop}")


def scaled_dimensions(width: int, height: int, scale_width: int) -> tuple[int, int]:
    if scale_width <= 0:
        raise ValueError("--scale-width must be positive")
    scale_height = max(2, round(height * (scale_width / width)))
    # yuv/gray rawvideo is simpler when dimensions are even.
    if scale_height % 2:
        scale_height += 1
    return scale_width, scale_height


def frame_differences(
    input_path: Path,
    *,
    info: VideoInfo,
    sample_fps: float,
    threshold: float,
    scale_width: int,
    crop: str,
    progress: ProgressBar | None = None,
) -> list[Probe]:
    scaled_width, scaled_height = scaled_dimensions(info.width, info.height, scale_width)
    vf = (
        f"fps={sample_fps},"
        f"{crop_filter(crop, info.width, info.height)}"
        f"scale={scaled_width}:{scaled_height},format=gray"
    )
    frame_size = scaled_width * scaled_height
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    probes: list[Probe] = []
    previous: bytes | None = None
    frame_index = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        while True:
            frame = process.stdout.read(frame_size)
            if not frame:
                break
            if len(frame) != frame_size:
                raise RuntimeError("ffmpeg returned a partial frame")
            timestamp = frame_index / sample_fps
            if previous is not None:
                difference = mean_abs_diff(previous, frame)
                probes.append(Probe(timestamp=timestamp, difference=difference, quiet=difference <= threshold))
            previous = frame
            frame_index += 1
            if progress:
                progress.update(frame_index)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("ffmpeg failed while sampling frames")
    if progress:
        progress.finish()
    return probes


def mean_abs_diff(left: bytes, right: bytes) -> float:
    if len(left) != len(right):
        raise ValueError("frames must have the same size")
    total = sum(abs(a - b) for a, b in zip(left, right))
    return total / (len(left) * 255.0)


def detect_idle_intervals(
    probes: Iterable[Probe],
    *,
    min_duration: float,
    padding: float,
    video_duration: float,
    threshold: float,
) -> list[Interval]:
    probe_list = list(probes)
    intervals: list[Interval] = []
    run: list[Probe] = []

    def flush() -> None:
        nonlocal run
        if not run:
            return
        raw_start = run[0].timestamp
        raw_end = run[-1].timestamp
        duration = raw_end - raw_start
        has_activity_before = any(p.timestamp < raw_start and not p.quiet for p in probe_list)
        has_activity_after = any(p.timestamp > raw_end and not p.quiet for p in probe_list)
        if duration >= min_duration and has_activity_before and has_activity_after:
            mean_difference = sum(p.difference for p in run) / len(run)
            # Confidence is intentionally conservative. Long, very quiet spans score higher.
            quiet_score = max(0.0, min(1.0, 1.0 - (mean_difference / max(threshold, 1e-9))))
            duration_score = max(0.0, min(1.0, duration / (min_duration * 3.0)))
            confidence = round((quiet_score * 0.65) + (duration_score * 0.35), 4)
            start = max(0.0, raw_start - padding)
            end = min(video_duration, raw_end + padding)
            if end > start:
                intervals.append(
                    Interval(
                        start=round(start, 3),
                        end=round(end, 3),
                        confidence=confidence,
                        mean_difference=round(mean_difference, 6),
                        samples=len(run),
                    )
                )
        run = []

    for probe in probe_list:
        if probe.quiet:
            run.append(probe)
        else:
            flush()
    flush()
    return merge_close_intervals(intervals, gap=padding * 2)


def merge_close_intervals(intervals: Iterable[Interval], *, gap: float) -> list[Interval]:
    sorted_intervals = sorted(intervals, key=lambda item: item.start)
    if not sorted_intervals:
        return []
    merged: list[Interval] = [sorted_intervals[0]]
    for interval in sorted_intervals[1:]:
        previous = merged[-1]
        if interval.start <= previous.end + gap:
            total_samples = previous.samples + interval.samples
            merged[-1] = Interval(
                start=previous.start,
                end=max(previous.end, interval.end),
                confidence=round(max(previous.confidence, interval.confidence), 4),
                mean_difference=round(
                    ((previous.mean_difference * previous.samples) + (interval.mean_difference * interval.samples))
                    / total_samples,
                    6,
                ),
                samples=total_samples,
            )
        else:
            merged.append(interval)
    return merged


def keep_intervals(cut_intervals: Iterable[Interval], *, duration: float) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for interval in sorted(cut_intervals, key=lambda item: item.start):
        if interval.start > cursor:
            keep.append((round(cursor, 3), round(interval.start, 3)))
        cursor = max(cursor, interval.end)
    if cursor < duration:
        keep.append((round(cursor, 3), round(duration, 3)))
    return [(start, end) for start, end in keep if end - start > 0.01]


def write_manifest(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    info: VideoInfo,
    intervals: list[Interval],
    args: argparse.Namespace,
) -> None:
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "duration": round(info.duration, 3),
        "removed_duration": round(sum(item.duration for item in intervals), 3),
        "preset": args.preset,
        "settings": {
            "format": args.output_format,
            "aggressiveness": args.aggressiveness,
            "threshold": args.threshold,
            "min_duration": args.min_duration,
            "padding": args.padding,
            "sample_fps": args.sample_fps,
            "scale_width": args.scale_width,
            "crop": args.crop,
            "gif_fps": getattr(args, "gif_fps", DEFAULT_GIF_FPS),
            "gif_width": getattr(args, "gif_width", DEFAULT_GIF_WIDTH),
        },
        "removed": [
            {
                "start": item.start,
                "end": item.end,
                "duration": round(item.duration, 3),
                "confidence": item.confidence,
                "mean_difference": item.mean_difference,
                "samples": item.samples,
            }
            for item in intervals
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def concat_filter_parts(intervals: list[Interval], info: VideoInfo, *, include_audio: bool) -> tuple[list[str], str, str | None]:
    keeps = keep_intervals(intervals, duration=info.duration)
    if not keeps:
        raise ValueError("All content would be removed; refusing to create an empty output")

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for index, (start, end) in enumerate(keeps):
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]")
        concat_inputs.append(f"[v{index}]")
        if include_audio:
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[a{index}]")
    video_label = "[outv]"
    audio_label = "[outa]" if include_audio else None
    filter_parts.append(
        "".join(concat_inputs)
        + f"concat=n={len(keeps)}:v=1:a={1 if include_audio else 0}{video_label}"
        + (audio_label or "")
    )
    return filter_parts, video_label, audio_label


def cut_video(
    input_path: Path,
    output_path: Path,
    intervals: list[Interval],
    info: VideoInfo,
    *,
    output_format: str,
    progress_enabled: bool,
    args: argparse.Namespace,
) -> None:
    if output_format == "mp4":
        cut_mp4(input_path, output_path, intervals, info, progress_enabled=progress_enabled)
        return
    if output_format == "gif":
        cut_gif(
            input_path,
            output_path,
            intervals,
            info,
            progress_enabled=progress_enabled,
            gif_fps=args.gif_fps,
            gif_width=args.gif_width,
        )
        return
    raise ValueError(f"Unknown output format: {output_format}")


def run_ffmpeg(command: list[str], *, progress: ProgressBar | None = None) -> None:
    if progress:
        command = [command[0], "-progress", "pipe:1", "-nostats", *command[1:]]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert process.stdout is not None
        assert process.stderr is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_ms":
                progress.update(int(value) / 1_000_000)
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
        progress.finish()
        return
    subprocess.run(command, check=True)


def cut_mp4(
    input_path: Path,
    output_path: Path,
    intervals: list[Interval],
    info: VideoInfo,
    *,
    progress_enabled: bool,
) -> None:
    if not intervals:
        if input_path.suffix.lower() == output_path.suffix.lower():
            shutil.copyfile(input_path, output_path)
            return
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
        ]
        if info.has_audio:
            command += ["-c:a", "aac", "-b:a", "160k"]
        command += [str(output_path)]
        run_ffmpeg(command, progress=ProgressBar("Encoding", total=info.duration, enabled=progress_enabled))
        return
    filter_parts, video_label, audio_label = concat_filter_parts(intervals, info, include_audio=info.has_audio)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        video_label,
    ]
    if audio_label:
        command += ["-map", audio_label]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ]
    if audio_label:
        command += ["-c:a", "aac", "-b:a", "160k"]
    command += [str(output_path)]
    run_ffmpeg(command, progress=ProgressBar("Encoding", total=info.duration, enabled=progress_enabled))


def cut_gif(
    input_path: Path,
    output_path: Path,
    intervals: list[Interval],
    info: VideoInfo,
    *,
    progress_enabled: bool,
    gif_fps: int,
    gif_width: int,
) -> None:
    filter_parts, video_label, _ = concat_filter_parts(intervals, info, include_audio=False)
    scale_width = "iw" if gif_width == 0 else str(gif_width)
    filter_parts.append(f"{video_label}fps={gif_fps},scale={scale_width}:-1:flags=lanczos,split[gif0][gif1]")
    filter_parts.append("[gif0]palettegen=stats_mode=full[p]")
    filter_parts.append("[gif1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        ";".join(filter_parts),
        "-loop",
        "0",
        str(output_path),
    ]
    run_ffmpeg(command, progress=ProgressBar("Encoding", total=info.duration, enabled=progress_enabled))


def output_default(input_path: Path, output_format: str) -> Path:
    suffix = ".gif" if output_format == "gif" else ".mp4"
    return input_path.with_name(f"{input_path.stem}.cleaned{suffix}")


def infer_output_format(output_path: Path | None, requested_format: str | None) -> str:
    if requested_format:
        return requested_format
    if output_path and output_path.suffix.lower() == ".gif":
        return "gif"
    return "mp4"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut visually idle Codex/Claude waiting spans from screen recordings."
    )
    parser.add_argument("input", type=Path, help="Input .mp4/.mkv recording")
    parser.add_argument("-o", "--output", type=Path, help="Cleaned output video path")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        help="Output format. Defaults to mp4, or gif when --output ends in .gif.",
    )
    parser.add_argument("--manifest", type=Path, help="Manifest JSON path")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="codex")
    parser.add_argument("--threshold", type=float, help="Mean normalized frame diff threshold for quiet frames")
    parser.add_argument("--min-duration", type=float, help="Minimum quiet span duration to cut, in seconds")
    parser.add_argument("--padding", type=float, help="Seconds to expand each cut on both sides")
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=DEFAULT_AGGRESSIVENESS,
        help=(
            "Detection aggressiveness multiplier. 1.0 keeps preset/custom settings; "
            "higher values raise the quiet threshold, shorten --min-duration, and widen --padding."
        ),
    )
    parser.add_argument("--sample-fps", type=float, help="Frame sampling rate for detection")
    parser.add_argument("--scale-width", type=int, help="Downscaled width used for diffing")
    parser.add_argument("--crop", choices=["full", "center", "terminal"], help="Region preset used for diffing")
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help=f"GIF frame rate. Defaults to {DEFAULT_GIF_FPS}.",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=DEFAULT_GIF_WIDTH,
        help="GIF output width in pixels. Defaults to 0, which keeps the source width.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest; do not create output video")
    parser.add_argument("--no-progress", action="store_true", help="Disable console progress bars")
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    for key, value in preset.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)
    if args.aggressiveness <= 0:
        parser.error("--aggressiveness must be positive")
    if args.gif_fps <= 0:
        parser.error("--gif-fps must be positive")
    if args.gif_width < 0:
        parser.error("--gif-width must be zero or positive")
    if args.aggressiveness != DEFAULT_AGGRESSIVENESS:
        args.threshold *= args.aggressiveness
        args.min_duration /= args.aggressiveness
        args.padding *= args.aggressiveness
    args.output_format = infer_output_format(args.output, args.output_format)
    if args.output is None:
        args.output = output_default(args.input, args.output_format)
    elif args.output_format == "gif" and args.output.suffix.lower() != ".gif":
        parser.error("--format gif requires an output path ending in .gif")
    elif args.output_format == "mp4" and args.output.suffix.lower() == ".gif":
        parser.error("--format mp4 cannot write to a .gif output path")
    if args.manifest is None:
        args.manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2
    if input_path == output_path:
        print("Output path must differ from input path", file=sys.stderr)
        return 2

    try:
        info = probe_video(input_path)
        progress_enabled = not args.no_progress and sys.stderr.isatty()
        probes = frame_differences(
            input_path,
            info=info,
            sample_fps=args.sample_fps,
            threshold=args.threshold,
            scale_width=args.scale_width,
            crop=args.crop,
            progress=ProgressBar(
                "Analyzing",
                total=max(1, math.ceil(info.duration * args.sample_fps)),
                enabled=progress_enabled,
            ),
        )
        intervals = detect_idle_intervals(
            probes,
            min_duration=args.min_duration,
            padding=args.padding,
            video_duration=info.duration,
            threshold=args.threshold,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(
            manifest_path,
            input_path=input_path,
            output_path=output_path,
            info=info,
            intervals=intervals,
            args=args,
        )
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="clip-idle-") as tmp:
                temporary_output = Path(tmp) / output_path.name
                cut_video(
                    input_path,
                    temporary_output,
                    intervals,
                    info,
                    output_format=args.output_format,
                    progress_enabled=progress_enabled,
                    args=args,
                )
                shutil.move(str(temporary_output), output_path)
        print(f"Detected {len(intervals)} removable span(s).")
        print(f"Manifest: {manifest_path}")
        if not args.dry_run:
            print(f"Output: {output_path}")
        return 0
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"clip-idle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
