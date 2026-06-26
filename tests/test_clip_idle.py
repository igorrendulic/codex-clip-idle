import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from clip_idle import (
    Interval,
    Probe,
    VideoInfo,
    cut_gif,
    detect_idle_intervals,
    keep_intervals,
    mean_abs_diff,
    parse_args,
    write_manifest,
)


class ClipIdleTests(unittest.TestCase):
    def test_mean_abs_diff_is_normalized(self):
        self.assertEqual(mean_abs_diff(bytes([0, 0]), bytes([0, 0])), 0)
        self.assertEqual(mean_abs_diff(bytes([0, 255]), bytes([255, 0])), 1)

    def test_detects_quiet_span_with_activity_on_both_sides(self):
        probes = [
            Probe(0.5, 0.02, False),
            Probe(1.0, 0.001, True),
            Probe(2.0, 0.001, True),
            Probe(3.0, 0.001, True),
            Probe(4.5, 0.02, False),
        ]

        intervals = detect_idle_intervals(
            probes,
            min_duration=2.0,
            padding=0.25,
            video_duration=5.0,
            threshold=0.005,
        )

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].start, 0.75)
        self.assertEqual(intervals[0].end, 3.25)
        self.assertGreater(intervals[0].confidence, 0.5)

    def test_detects_one_second_quiet_span_with_short_wait_settings(self):
        probes = [
            Probe(0.5, 0.02, False),
            Probe(1.0, 0.001, True),
            Probe(2.0, 0.001, True),
            Probe(2.5, 0.02, False),
        ]

        intervals = detect_idle_intervals(
            probes,
            min_duration=1.0,
            padding=0.1,
            video_duration=3.0,
            threshold=0.0035,
        )

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].start, 0.9)
        self.assertEqual(intervals[0].end, 2.1)

    def test_does_not_cut_static_tail_without_activity_after(self):
        probes = [
            Probe(0.5, 0.02, False),
            Probe(1.0, 0.001, True),
            Probe(2.0, 0.001, True),
            Probe(3.0, 0.001, True),
        ]

        intervals = detect_idle_intervals(
            probes,
            min_duration=2.0,
            padding=0.25,
            video_duration=4.0,
            threshold=0.005,
        )

        self.assertEqual(intervals, [])

    def test_keep_intervals_are_inverse_of_cuts(self):
        cuts = [Interval(1.0, 3.0, 0.9, 0.001, 4), Interval(5.0, 6.0, 0.8, 0.001, 2)]

        self.assertEqual(
            keep_intervals(cuts, duration=8.0),
            [(0.0, 1.0), (3.0, 5.0), (6.0, 8.0)],
        )

    def test_manifest_contains_removed_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            write_manifest(
                manifest,
                input_path=Path("input.mp4"),
                output_path=Path("output.mp4"),
                info=VideoInfo(duration=10.0, width=1920, height=1080, has_audio=True),
                intervals=[Interval(1.0, 3.5, 0.75, 0.001, 5)],
                args=Namespace(
                    preset="codex",
                    aggressiveness=1.0,
                    threshold=0.0035,
                    min_duration=3.0,
                    padding=0.25,
                    sample_fps=2.0,
                    scale_width=320,
                    crop="full",
                    output_format="mp4",
                ),
            )

            payload = json.loads(manifest.read_text())

        self.assertEqual(payload["removed_duration"], 2.5)
        self.assertEqual(payload["removed"][0]["start"], 1.0)
        self.assertEqual(payload["settings"]["crop"], "full")
        self.assertEqual(payload["settings"]["format"], "mp4")
        self.assertEqual(payload["settings"]["aggressiveness"], 1.0)

    def test_parse_args_defaults_to_mp4_output(self):
        args = parse_args(["recording.mkv"])

        self.assertEqual(args.output_format, "mp4")
        self.assertEqual(args.output, Path("recording.cleaned.mp4"))

    def test_parse_args_uses_gif_default_output_for_gif_format(self):
        args = parse_args(["recording.mkv", "--format", "gif"])

        self.assertEqual(args.output_format, "gif")
        self.assertEqual(args.output, Path("recording.cleaned.gif"))

    def test_parse_args_infers_gif_format_from_output_suffix(self):
        args = parse_args(["recording.mkv", "-o", "share.gif"])

        self.assertEqual(args.output_format, "gif")
        self.assertEqual(args.output, Path("share.gif"))

    def test_parse_args_applies_aggressiveness_multiplier(self):
        args = parse_args(["recording.mkv", "--aggressiveness", "2"])

        self.assertEqual(args.threshold, 0.007)
        self.assertEqual(args.min_duration, 0.5)
        self.assertEqual(args.padding, 0.2)

    def test_parse_args_rejects_non_positive_aggressiveness(self):
        with self.assertRaises(SystemExit):
            parse_args(["recording.mkv", "--aggressiveness", "0"])

    def test_parse_args_supports_disabling_progress(self):
        args = parse_args(["recording.mkv", "--no-progress"])

        self.assertTrue(args.no_progress)

    def test_parse_args_uses_high_quality_gif_defaults(self):
        args = parse_args(["recording.mkv", "--format", "gif"])

        self.assertEqual(args.gif_fps, 12)
        self.assertEqual(args.gif_width, 960)

    def test_parse_args_accepts_custom_gif_quality(self):
        args = parse_args(["recording.mkv", "--format", "gif", "--gif-fps", "24", "--gif-width", "1440"])

        self.assertEqual(args.gif_fps, 24)
        self.assertEqual(args.gif_width, 1440)

    def test_parse_args_rejects_invalid_gif_quality(self):
        with self.assertRaises(SystemExit):
            parse_args(["recording.mkv", "--gif-fps", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["recording.mkv", "--gif-width", "-1"])

    def test_gif_output_loops_forever(self):
        with patch("clip_idle.run_ffmpeg") as run_ffmpeg:
            cut_gif(
                Path("input.mp4"),
                Path("output.gif"),
                [],
                VideoInfo(duration=10.0, width=1920, height=1080, has_audio=False),
                progress_enabled=False,
                gif_fps=15,
                gif_width=0,
            )

        command = run_ffmpeg.call_args.args[0]
        loop_index = command.index("-loop")
        self.assertEqual(command[loop_index + 1], "0")


if __name__ == "__main__":
    unittest.main()
