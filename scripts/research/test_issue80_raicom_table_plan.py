"""Offline geometry sampling checks; no physics integration."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import mujoco

from issue80_raicom_table_plan import (
    _load_native_standing_qpos, sample_segment, build_raw_world_odom_route_config,
)
from issue80_local01_scene_route import RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD


class TablePlanTests(unittest.TestCase):
    def test_native_trace_loader_accepts_only_a_truncated_final_record(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
        )
        metadata = {
            "event": "model_metadata", "nq": model.nq, "nv": model.nv,
            "nu": model.nu, "nbody": model.nbody, "ngeom": model.ngeom,
            "mujoco_runtime_version": "3.3.7", "timestep_s": 0.002,
        }
        sample = {"event": "post_step", "generation": 1, "sim_time_s": 1.0,
                  "qpos": [0.0] * model.nq}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text(json.dumps(metadata) + "\n" + json.dumps(sample) + "\n{\"truncated")
            qpos, provenance = _load_native_standing_qpos(trace, model)
        self.assertEqual(qpos, [0.0] * model.nq)
        self.assertTrue(provenance["truncated_final_record_ignored"])

    def test_raw_odom_config_never_uses_planned_local_coordinates(self):
        poses = [
            {"name":"START","scene_world":{"xyz_m":[-1.5,-1.5,.68],"yaw_rad":math.pi/2},
             "local01_reset_local_odom":{"x_m":0,"y_m":0,"yaw_rad":0}},
            {"name":"WAYPOINT","scene_world":{"xyz_m":[-.25,-1.5,.68],"yaw_rad":0}},
            {"name":"TABLE_DOCKING","scene_world":{"xyz_m":[1,-1.5,.68],"yaw_rad":0}},
        ]
        config = build_raw_world_odom_route_config(poses)
        self.assertEqual(config["frame_id"],"odom")
        self.assertEqual(config["start"],{"x_m":-1.5,"y_m":-1.5,"yaw_rad":math.pi/2})
        self.assertEqual(config["table_docking"]["x_m"],1.)
        self.assertIn("OFFLINE-ONLY",config["route_id"])

    def test_translation_preserves_endpoints_and_spacing(self):
        a = (-1.5, -1.5, .68, 0.)
        b = (1., -1.5, .68, 0.)
        points = sample_segment(a, b)
        self.assertEqual(points[0], a)
        self.assertEqual(points[-1], b)
        self.assertTrue(all(math.dist(x[:3], y[:3]) <= .02500001
                            for x, y in zip(points, points[1:])))

    def test_heading_wrap_uses_short_branch(self):
        points = sample_segment((0, 0, .68, math.pi-.01),
                                (0, 0, .68, -math.pi+.01))
        self.assertAlmostEqual(points[-1][3]-points[0][3], .02)

    def test_initial_alignment_does_not_translate_base(self):
        points = sample_segment((-1.5, -1.5, .68, math.pi/2),
                                (-1.5, -1.5, .68, 0))
        self.assertGreater(len(points), 70)
        self.assertTrue(all(p[:3] == (-1.5, -1.5, .68) for p in points))

    def test_full_plan_uses_the_reviewed_runtime_start_yaw_envelope(self):
        self.assertAlmostEqual(RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD, .075)


if __name__ == "__main__":
    unittest.main()
