from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_standing_posture import PostureError, extract


class StandingPostureTests(unittest.TestCase):
    def test_selects_last_complete_current_generation_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [
                {"event": "model_metadata", "nq": 3},
                {"event": "post_step", "generation": 0, "qpos": [0, 0, 0], "pelvis": {"state": {"name": "pelvis"}}},
                {"event": "post_step", "generation": 1, "sim_time_s": 2.0, "host_monotonic_ns": 3, "qpos": [1, 2, 3], "pelvis": {"state": {"name": "pelvis"}}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "{", encoding="utf-8")
            result = extract(path)
            self.assertEqual(result["source_generation"], 1)
            self.assertEqual(result["qpos"], [1.0, 2.0, 3.0])

    def test_rejects_wrong_qpos_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps({"event": "model_metadata", "nq": 3}) + "\n" + json.dumps({"event": "post_step", "generation": 0, "qpos": [1], "pelvis": {"state": {"name": "pelvis"}}}) + "\n", encoding="utf-8")
            with self.assertRaises(PostureError):
                extract(path)
