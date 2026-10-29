from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_observed_route_config import LidarRouteConfigError, build


class LidarObservedRouteConfigTests(unittest.TestCase):
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
