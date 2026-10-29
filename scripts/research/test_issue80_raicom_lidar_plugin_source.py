from __future__ import annotations

import unittest
from pathlib import Path


class RaycasterPluginSourceTests(unittest.TestCase):
    def test_raw_trace_keeps_local_and_explicit_world_coordinates(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_plugin.cc").read_text(encoding="utf-8")
        self.assertIn('"],\\"world_xyz\\":["', source)
        self.assertIn("mju_mulMatVec3(world_point, rotation, point)", source)
        self.assertIn("origin[0] + world_point[0]", source)
        self.assertIn("origin[2] + world_point[2]", source)


if __name__ == "__main__":
    unittest.main()
