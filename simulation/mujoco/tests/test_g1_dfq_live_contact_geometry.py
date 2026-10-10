from pathlib import Path
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

# This test exercises contact-geometry math only; rendering is not part of the
# headless smoke dependency set.
cv2_stub = ModuleType("cv2")
cv2_stub.__version__ = "unused-by-contact-geometry-test"
sys.modules.setdefault("cv2", cv2_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_dfq_grasp_core as core


class LiveContactGeometryTests(unittest.TestCase):
    def test_contact_sequence_waits_for_stable_opposition_before_releasing_thumb(self):
        self.assertEqual(core.contact_sequence_phase({"thumb"}), "OPPOSING_APPROACH")
        self.assertTrue(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "OPPOSING_APPROACH", "thumb", True
        ))
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "OPPOSING_APPROACH", "thumb", False
        ))

    def test_contact_sequence_holds_opposition_until_thumb_is_confirmed(self):
        self.assertEqual(core.contact_sequence_phase({"index", "ring"}), "THUMB_APPROACH")
        self.assertTrue(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "THUMB_APPROACH", "index", True
        ))
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "THUMB_APPROACH", "thumb", False
        ))
        self.assertEqual(
            core.contact_sequence_phase({"thumb", "index"}), "BILATERAL_CLOSE"
        )
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "BILATERAL_CLOSE", "thumb", True
        ))

    def test_simultaneous_close_profile_does_not_apply_sequence_holds(self):
        self.assertFalse(core.contact_sequence_holds_target(
            "simultaneous", "OPPOSING_APPROACH", "thumb", True
        ))

    def test_candidate_force_requirements_allow_digits_without_allocated_contacts(self):
        thumb, opposing = core.candidate_digit_normal_requirements({
            "thumb": {"normal_force_n": 2.1},
            "index": {"normal_force_n": 2.3},
            "ring": {"normal_force_n": 0.01},
        })
        self.assertAlmostEqual(thumb, 2.1)
        self.assertAlmostEqual(opposing, 2.31)

    def test_world_pitch_reduces_measured_thumb_index_height_offset(self):
        base = np.array([
            [np.cos(np.deg2rad(60.0)), -np.sin(np.deg2rad(60.0)), 0.0],
            [np.sin(np.deg2rad(60.0)), np.cos(np.deg2rad(60.0)), 0.0],
            [0.0, 0.0, 1.0],
        ])
        candidate = core.world_pitch_test_rotation(base)
        measured_contact_delta = np.array([0.0678, -0.0126, 0.0237])
        transformed_delta = candidate @ base.T @ measured_contact_delta
        expected_z = (
            -np.sin(np.deg2rad(15.0)) * measured_contact_delta[0]
            + np.cos(np.deg2rad(15.0)) * measured_contact_delta[2]
        )
        self.assertAlmostEqual(transformed_delta[2], expected_z, places=12)
        self.assertLess(abs(expected_z), abs(measured_contact_delta[2]))

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
