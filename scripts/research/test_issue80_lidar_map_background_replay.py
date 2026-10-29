from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_map_localizer import MAGIC
from issue80_lidar_map_background_replay import replay


def capture(points: list[list[float]]) -> dict:
    return {
        "status": "PASS_NONEMPTY_POINTCLOUD",
        "last_message": {
            "frame_id": "map",
            "all_finite_xyz_retention_blocked": False,
            "all_finite_xyz": points,
        },
    }


class MapBackgroundReplayTests(unittest.TestCase):
    def test_repeat_is_clear_and_three_cell_chair_is_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "map.bin"
            values = [0.0, 0.0, 0.0] + [2.0] * 450
            with database.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 450))
                stream.write(struct.pack("<" + "f" * len(values), *values))
            background = root / "background.json"
            background.write_text(json.dumps(capture([[-0.75, -0.75, 0.5]])), encoding="utf-8")
            repeated = root / "repeated.json"
            repeated.write_text(json.dumps(capture([[-0.745, -0.75, 0.5]])), encoding="utf-8")
            clear = replay(
                background_capture=background, observed_capture=repeated, static_database=database,
                maximum_match_distance_m=0.02, minimum_cluster_cells=3,
            )
            self.assertEqual(clear["second_decision"], "CLEAR_DIRECT_PATH_STATIC_ONLY")
            chair = root / "chair.json"
            chair.write_text(json.dumps(capture([[0.02, -1.48, 0.5], [0.02, -1.38, 0.5], [0.12, -1.38, 0.5]])), encoding="utf-8")
            blocked = replay(
                background_capture=background, observed_capture=chair, static_database=database,
                maximum_match_distance_m=0.02, minimum_cluster_cells=3,
            )
            self.assertEqual(blocked["second_decision"], "DETOUR_READY_STATIC_ONLY")
            self.assertEqual(len(blocked["confirmed_cells"]), 3)
            self.assertFalse(blocked["velocity_publisher_created"])


if __name__ == "__main__":
    unittest.main()
