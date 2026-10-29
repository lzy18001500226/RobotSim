from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from issue80_raicom_observed_detour_static import ObservedDetourError, observed_route


class ObservedDetourStaticTests(unittest.TestCase):
    def write_replay(self, directory: Path, *, decision: str = "DETOUR_READY_STATIC_ONLY") -> Path:
        path = directory / "replay.json"
        path.write_text(json.dumps({
            "schema": "robotsim.issue80.lidar-map-background-replay.v1",
            "status": "PASS_CAPTURE_REPLAYED",
            "second_decision": decision,
            "detour_waypoints": [
                {"x_m": -1.45, "y_m": -1.45},
                {"x_m": -0.65, "y_m": -2.25},
                {"x_m": 0.95, "y_m": -1.45},
            ],
        }), encoding="utf-8")
        return path

    def test_loads_only_qualified_observed_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(len(observed_route(self.write_replay(Path(raw)))), 3)

    def test_rejects_non_detour_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ObservedDetourError):
                observed_route(self.write_replay(Path(raw), decision="CLEAR_DIRECT_PATH_STATIC_ONLY"))


if __name__ == "__main__":
    unittest.main()
