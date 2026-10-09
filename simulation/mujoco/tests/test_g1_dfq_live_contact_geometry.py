import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import g1_dfq_grasp_core as core


class LiveContactGeometryTests(unittest.TestCase):
    def test_cached_snapshot_does_not_read_stale_mujoco_contact_index(self):
        model = SimpleNamespace(nv=1, geom_bodyid=np.asarray([0, 2]))
        item = {
            "side": "right_hand",
            "digit": "index",
            "distance_m": -0.00001,
            "true_normal_force_n": 0.05,
            "contact_geom1_id": 0,
            "contact_geom2_id": 1,
            "contact_frame_world": np.eye(3).tolist(),
            "contact_position_world_m": [0.1, 0.2, 0.3],
            "contact_friction_axes": [1.4, 1.4],
            "contact_observed_time_s": 4.0,
            "mujoco_contact_index": 999,
            "other_geom": "index_geom",
            "other_body": "R_index_proximal",
            "bottle_geom": "bottle_body",
        }

        def jacobian(_model, _data, jacp, _jacr, _position, _body_id):
            jacp[:] = 0.0

        with patch.object(
            core.mujoco,
            "mj_id2name",
            side_effect=lambda _model, _kind, geom_id: (
                "bottle_body" if geom_id == 0 else "index_geom"
            ),
        ), patch.object(core.mujoco, "mj_jac", side_effect=jacobian):
            geometry = core.live_contact_allocation_geometry(
                model, object(), [item], {}, {}
            )

        self.assertTrue(geometry["success"])
        self.assertEqual(geometry["contacts"][0]["distance_m"], -0.00001)
        self.assertEqual(
            geometry["contacts"][0]["contact_position_world_m"], [0.1, 0.2, 0.3]
        )
        self.assertEqual(geometry["contacts"][0]["source_contact_time_s"], 4.0)


if __name__ == "__main__":
    unittest.main()
