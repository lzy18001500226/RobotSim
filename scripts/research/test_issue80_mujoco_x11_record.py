"""Offline tests for the passive, window-verified X11 recorder."""

from __future__ import annotations

import struct
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from issue80_mujoco_x11_record import parse_xwd_header, record_verified_no_motion_preflight


def xwd_fixture(*, version: int = 7, pixels: bytes = b"\x00" * 8) -> bytes:
    header = struct.pack(
        "<25I",
        101, version, 2, 24, 2, 1, 0, 0, 32, 0, 32, 32, 8, 4,
        0x00FF0000, 0x0000FF00, 0x000000FF, 8, 256, 0, 2, 1, 0, 0, 0,
    )
    return header + b"\x00" + pixels


class XwdHeaderTests(unittest.TestCase):
    def test_parses_24_bit_truecolor_image_with_32_bit_storage(self):
        result = parse_xwd_header(xwd_fixture())

        self.assertEqual(result["width"], 2)
        self.assertEqual(result["height"], 1)
        self.assertEqual(result["bits_per_pixel"], 32)
        self.assertEqual(result["bytes_per_line"], 8)
        self.assertEqual(result["pixel_data_offset"], 101)
        self.assertEqual(result["red_mask"], 0x00FF0000)

    def test_rejects_wrong_version_or_truncated_pixels(self):
        with self.assertRaisesRegex(ValueError, "version"):
            parse_xwd_header(xwd_fixture(version=6))
        with self.assertRaisesRegex(ValueError, "truncated XWD color table or pixel data"):
            parse_xwd_header(xwd_fixture(pixels=b"\x00" * 7))


class NoMotionVideoPreflightTests(unittest.TestCase):
    def test_preflight_records_verified_window_without_motion_or_input_events(self):
        class FakeImage:
            width = 640
            height = 480

            def copy(self):
                return self

            def resize(self, _size, _resample):
                return self

            def save(self, path, **_kwargs):
                Path(path).write_bytes(b"GIF89a-test-only")

        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(Resampling=types.SimpleNamespace(LANCZOS=1))
        with tempfile.TemporaryDirectory(prefix="issue80-no-motion-video-") as temporary:
            root = Path(temporary)
            (root / "reset_dispatch_event.json").write_text(json.dumps({
                "window": {"window_id": 10, "pid": 20, "title": "verified MuJoCo"},
            }), encoding="utf-8")
            with patch.dict(sys.modules, {"PIL": pil}), \
                    patch("issue80_mujoco_x11_record._verify_x11_window", return_value={
                        "xid": "0xa", "pid": 20, "title": "verified MuJoCo",
                    }) as verify, \
                    patch("issue80_mujoco_x11_record._capture_xwd", return_value=b"xwd"), \
                    patch("issue80_mujoco_x11_record._xwd_to_rgb_image", return_value=FakeImage()):
                result = record_verified_no_motion_preflight(
                    root, fps=2.0, duration_s=1.0,
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["classification"], "NO_MOTION_MUJOCO_GUI_RECORDING_PREFLIGHT")
            self.assertFalse(result["motion_command_observed"])
            self.assertFalse(result["input_events_sent"])
            self.assertGreaterEqual(result["frame_count"], 2)
            self.assertEqual(verify.call_args.args, (10, 20, "verified MuJoCo"))
            self.assertTrue((root / "mujoco_video_preflight.gif").is_file())
            self.assertTrue((root / "x11_video_preflight.json").is_file())

    def test_preflight_requires_reset_identity_and_short_bounded_duration(self):
        with tempfile.TemporaryDirectory(prefix="issue80-no-motion-video-") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "between 1 and 5"):
                record_verified_no_motion_preflight(root, duration_s=5.01)
            with self.assertRaisesRegex(FileNotFoundError, "Reset dispatch event"):
                record_verified_no_motion_preflight(root, duration_s=1.0)


if __name__ == "__main__":
    unittest.main()
