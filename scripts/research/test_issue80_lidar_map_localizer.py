from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from issue80_lidar_map_localizer import MAGIC, best_candidate, load_database, raw_ranges
from issue80_lidar_raw_bridge import RAW_SCHEMA


class MapLocalizerTests(unittest.TestCase):
    def test_selects_static_signature_without_odom(self) -> None:
        ranges = np.array([[1.0, 2.0, np.nan, 4.0], [2.0, 3.0, np.nan, 5.0]], dtype=np.float32)
        candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1]], dtype=np.float32)
        index, score, matches = best_candidate(candidates, ranges, np.array([2.0, 3.0, np.nan, 5.0], dtype=np.float32), minimum_matches=2)
        self.assertEqual(index, 1)
        self.assertEqual(matches, 3)
        self.assertLess(score, 1e-6)

    def test_raw_frame_preserves_ray_slots(self) -> None:
        frame = {"schema": RAW_SCHEMA, "xyz": [[3.0, 4.0, 0.0], None, ["nan", 0.0, 0.0]]}
        scan = raw_ranges(frame, 3)
        self.assertAlmostEqual(float(scan[0]), 5.0)
        self.assertTrue(np.isnan(scan[1]))
        self.assertTrue(np.isnan(scan[2]))

    def test_binary_map_header_loads_exact_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.bin"
            values = np.array([[1.0, 2.0, 0.0, 3.0, 4.0]], dtype="<f4")
            with path.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 2))
                stream.write(values.tobytes())
            poses, ranges = load_database(path)
            self.assertEqual(poses.tolist(), [[1.0, 2.0, 0.0]])
            self.assertEqual(ranges.tolist(), [[3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
