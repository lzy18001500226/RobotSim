from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_observed_route_config import LidarRouteConfigError, build, smooth_initial_heading


class LidarObservedRouteConfigTests(unittest.TestCase):
    def test_initial_heading_smoothing_preserves_endpoints_and_starts_near_forward(self) -> None:
        points = ((-1.45, -1.45), (-0.45, 0.45), (0.35, 0.45), (0.95, -1.45))
        smoothed = smooth_initial_heading(points)
        self.assertEqual(smoothed[0], points[0])
        self.assertEqual(smoothed[-2:], points[-2:])
        self.assertGreater(len(smoothed), len(points))
        heading = math.atan2(smoothed[1][1] - smoothed[0][1], smoothed[1][0] - smoothed[0][0])
        self.assertLess(abs(heading), 0.25)

    def test_rejects_unqualified_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            replay, audit, occupancy = (root / "replay.json", root / "audit.json", root / "occupancy.json")
            replay.write_text(json.dumps({"schema": "robotsim.issue80.lidar-map-background-replay.v1", "status": "PASS_CAPTURE_REPLAYED", "second_decision": "CLEAR_DIRECT_PATH_STATIC_ONLY"}), encoding="utf-8")
            audit.write_text("{}", encoding="utf-8")
            occupancy.write_text("{}", encoding="utf-8")
            with self.assertRaises(LidarRouteConfigError):
                build(replay_path=replay, static_audit_path=audit, occupancy_path=occupancy)


if __name__ == "__main__":
    unittest.main()
