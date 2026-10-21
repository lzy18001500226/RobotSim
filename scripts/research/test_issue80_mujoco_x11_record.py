"""Offline tests for the passive, window-verified X11 recorder."""

from __future__ import annotations

import struct
import unittest

from issue80_mujoco_x11_record import parse_xwd_header


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


if __name__ == "__main__":
    unittest.main()
