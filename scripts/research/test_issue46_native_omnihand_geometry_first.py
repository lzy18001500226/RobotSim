from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco

import issue46_native_omnihand_geometry_first as geometry
import issue46_native_omnihand_minimal_grasp as fixture


class GeometryFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="issue46-geometry-test-")
        cls.output = Path(cls.tempdir.name)
        (cls.output / "fixed").mkdir(parents=True)
        (cls.output / "lift").mkdir(parents=True)
        cls.candidate = geometry.CANDIDATES[1]
        cls.model, cls.details = fixture.build_model(
            "simulation-only",
            cls.output / "fixed",
            palm_position=cls.candidate["position"],
            palm_quaternion_wxyz=cls.candidate["quaternion_wxyz"],
        )
        cls.data = mujoco.MjData(cls.model)
        cls.groups = geometry.geometry_groups(cls.model, cls.details)
        geometry.assign_initial_hand_pose(
            cls.model,
            cls.data,
            cls.details,
            geometry.joint_targets(cls.details, {}),
        )
        mujoco.mj_forward(cls.model, cls.data)
        cls.envelope = geometry.scan_family_envelope(
            cls.model,
            cls.data,
            cls.details,
            cls.groups,
            str(cls.candidate["name"]),
            cls.output / "contact_envelope.csv",
        )
        cls.bilateral = geometry.find_bilateral_targets(
            cls.model,
            cls.data,
            cls.details,
            cls.groups,
            cls.envelope,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_contact_targets_stay_within_source_joint_ranges(self) -> None:
        self.assertEqual(self.bilateral["status"], "PASS")
        for joint_name, target in self.bilateral["source_joint_targets_rad"].items():
            low, high = self.details["joint_ranges"][joint_name]
            self.assertGreaterEqual(target, low)
            self.assertLessEqual(target, high)

    def test_open_pose_is_clear_and_bilateral_target_has_no_early_overlap(self) -> None:
        geometry.set_kinematic_pose(
            self.model,
            self.data,
            self.details,
            geometry.joint_targets(self.details, {}),
        )
        open_metrics = geometry.static_metrics(self.model, self.data, self.groups)
        self.assertGreater(open_metrics["palm_to_bottle"]["distance_m"], 0.005)
        self.assertGreaterEqual(open_metrics["hand_to_table"]["distance_m"], -geometry.GEOMETRY_TOLERANCE_M)

        self.assertEqual(self.bilateral["status"], "PASS")
        metrics = self.bilateral["metrics"]
        for family in ("thumb", "index"):
            self.assertLessEqual(
                abs(metrics["family_to_bottle"][family]["distance_m"]),
                geometry.GEOMETRY_TOLERANCE_M,
            )
        for family in ("middle", "ring", "pinky"):
            self.assertGreaterEqual(
                metrics["family_to_bottle"][family]["distance_m"],
                -geometry.GEOMETRY_TOLERANCE_M,
            )

    def test_vertical_lift_actuator_is_optional_and_bounded(self) -> None:
        model, details = fixture.build_model(
            "simulation-only",
            self.output / "lift",
            palm_position=self.candidate["position"],
            palm_quaternion_wxyz=self.candidate["quaternion_wxyz"],
            palm_slide_z=True,
        )
        self.assertTrue(details["palm_slide_z_enabled"])
        self.assertGreaterEqual(details["palm_slide_z_actuator_id"], 0)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fixture_palm_z")
        self.assertEqual(int(model.jnt_type[joint_id]), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.assertEqual(model.jnt_range[joint_id].tolist(), [0.0, 0.05])
        self.assertEqual(int(model.neq), 0)

    def test_bilateral_contact_metrics_distinguish_total_and_contiguous_contact(self) -> None:
        metrics = geometry.bilateral_contact_metrics(
            [0.002, 0.004, 0.006, 0.008, 0.010],
            {
                "thumb": [False, True, True, False, True],
                "index": [True, True, True, True, True],
            },
        )
        self.assertEqual(metrics["simultaneous_frames"], 3)
        self.assertAlmostEqual(metrics["simultaneous_fraction"], 0.6)
        self.assertAlmostEqual(metrics["simultaneous_active_duration_s"], 0.006)
        self.assertAlmostEqual(metrics["longest_contiguous_duration_s"], 0.004)
        self.assertEqual(metrics["first_simultaneous_time_s"], 0.004)
        self.assertEqual(metrics["last_simultaneous_time_s"], 0.010)


if __name__ == "__main__":
    unittest.main()
