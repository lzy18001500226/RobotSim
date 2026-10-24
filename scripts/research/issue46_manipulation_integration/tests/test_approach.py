import unittest

from scripts.research.issue46_manipulation_integration.approach import (
    IKWaypoint, ToolPose, make_linear_approach, validate_ik_waypoints,
)
from scripts.research.issue46_manipulation_integration.contracts import PoseSample, PoseSource, TimeDomain


class ApproachTests(unittest.TestCase):
    def setUp(self):
        self.object_pose = PoseSample("bottle", "world", (0.3, 0.04, 0.915), (0, 0, 0, 1),
                                      100, PoseSource.SIM_GROUND_TRUTH, TimeDomain.SIMULATION,
                                      model_identity="x2@sha", generation="reset-1", sequence=12)

    def test_builds_top_down_pregrasp_to_approach_and_intermediate_waypoints(self):
        plan = make_linear_approach(
            self.object_pose, tool_frame="R_omnipicker_base_link",
            target_point_world_m=(0.3, 0.04, 0.915), tool_orientation_xyzw=(0, 0, 0, 1),
            approach_direction_world=(0, 0, -1), pregrasp_standoff_m=0.14,
            interpolation_samples=3,
        )
        self.assertEqual([x.phase for x in plan.waypoints], ["PREGRASP", "APPROACH", "APPROACH", "APPROACH"])
        self.assertAlmostEqual(plan.waypoints[0].tool_pose.position_m[2], 1.055)
        self.assertAlmostEqual(plan.waypoints[-1].tool_pose.position_m[2], 0.915)
        self.assertEqual(plan.approach_direction_world, (0.0, 0.0, -1.0))

    def _waypoints(self, q1=0.2):
        limits = {"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)}
        return limits, (
            IKWaypoint("PREGRASP", 0.0, {"joint_a": 0.0, "joint_b": 0.0}, 0.0001, 0.001, 1.0),
            IKWaypoint("APPROACH", 1.0, {"joint_a": q1, "joint_b": -0.1}, 0.0002, 0.002, 0.8),
        )

    def test_ik_witness_checks_order_limits_residuals_and_margin(self):
        limits, points = self._waypoints()
        blockers = validate_ik_waypoints(
            points, expected_joint_names=("joint_a", "joint_b"),
            position_tolerance_m=0.002, orientation_tolerance_rad=0.01,
            minimum_limit_margin_rad=0.05, source_joint_limits_rad=limits,
            maximum_joint_step_rad=0.3,
        )
        self.assertEqual(blockers, ())

    def test_ik_witness_rejects_out_of_range_or_large_interpolation_step(self):
        limits, points = self._waypoints(1.1)
        blockers = validate_ik_waypoints(
            points, expected_joint_names=("joint_a", "joint_b"),
            position_tolerance_m=0.002, orientation_tolerance_rad=0.01,
            minimum_limit_margin_rad=0.05, source_joint_limits_rad=limits,
            maximum_joint_step_rad=0.3,
        )
        self.assertTrue(any(x.startswith("SOURCE_POSITION_LIMIT") for x in blockers))

    def test_ik_witness_rejects_name_order_mismatch(self):
        limits, points = self._waypoints()
        wrong = (points[0], IKWaypoint("APPROACH", 1.0, {"joint_b": -0.1, "joint_a": 0.2},
                                       0.0002, 0.002, 0.8))
        blockers = validate_ik_waypoints(
            wrong, expected_joint_names=("joint_a", "joint_b"),
            position_tolerance_m=0.002, orientation_tolerance_rad=0.01,
            minimum_limit_margin_rad=0.05, source_joint_limits_rad=limits,
        )
        self.assertIn("JOINT_NAME_ORDER:1", blockers)

    def test_ik_witness_rejects_negative_or_inconsistent_residuals(self):
        limits, points = self._waypoints()
        bad = (points[0], IKWaypoint("APPROACH", 1.0, {"joint_a": 0.2, "joint_b": -0.1},
                                     -0.001, 0.002, 0.8))
        blockers = validate_ik_waypoints(
            bad, expected_joint_names=("joint_a", "joint_b"),
            position_tolerance_m=0.002, orientation_tolerance_rad=0.01,
            minimum_limit_margin_rad=0.05, source_joint_limits_rad=limits,
        )
        self.assertIn("POSITION_IK_RESIDUAL:1", blockers)


if __name__ == "__main__":
    unittest.main()
