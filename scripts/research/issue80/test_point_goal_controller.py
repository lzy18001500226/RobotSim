"""Synthetic-only tests for the Issue #80 planar point-goal prototype."""

import math
import unittest

from point_goal_controller import (
    ControllerConfig,
    Goal2D,
    NavState,
    PointGoalController,
    Pose2D,
    goal_from_pose_fields,
    wrap_angle,
)


def pose(x=0.0, y=0.0, yaw=0.0, *, frame="odom", received=0.0):
    return Pose2D(x, y, yaw, frame, received)


class GoalValidationTests(unittest.TestCase):
    def test_pose_stamped_fields_validate_frame_and_planarity(self):
        goal = goal_from_pose_fields(
            frame_id="odom", expected_frame="odom",
            position_xyz=(1.0, -2.0, 0.0),
            orientation_xyzw=(0.0, 0.0, math.sin(0.4), math.cos(0.4)),
        )
        self.assertAlmostEqual(goal.yaw, 0.8)
        self.assertEqual(goal.frame_id, "odom")

    def test_goal_rejects_unknown_frame_zero_quaternion_and_nonfinite_pose(self):
        cases = (
            dict(frame_id="map", expected_frame="odom",
                 position_xyz=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1)),
            dict(frame_id="odom", expected_frame="odom",
                 position_xyz=(0, 0, 0), orientation_xyzw=(0, 0, 0, 0)),
            dict(frame_id="odom", expected_frame="odom",
                 position_xyz=(math.nan, 0, 0), orientation_xyzw=(0, 0, 0, 1)),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                goal_from_pose_fields(**kwargs)

    def test_goal_rejects_nonplanar_attitude(self):
        with self.assertRaises(ValueError):
            goal_from_pose_fields(
                frame_id="odom", expected_frame="odom",
                position_xyz=(0, 0, 0),
                orientation_xyzw=(math.sin(0.2), 0, 0, math.cos(0.2)),
                max_planar_tilt_rad=0.1,
            )

    def test_angle_wrap_handles_pi_boundary(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi / 2.0), -math.pi / 2.0)


class ControllerStateTests(unittest.TestCase):
    def setUp(self):
        self.config = ControllerConfig(
            xy_tolerance_m=0.1, yaw_tolerance_rad=0.1,
            align_tolerance_rad=0.2, pose_timeout_s=0.2,
            goal_timeout_s=2.0, arrival_dwell_s=0.25,
        )
        self.goal = Goal2D(1.0, 0.0, 0.0, "odom")
        self.controller = PointGoalController(self.config)

    def test_align_before_forward_when_goal_is_behind(self):
        out = self.controller.step(
            pose(), Goal2D(-1.0, 0.0, 0.0, "odom"), now_steady_s=0.0
        )
        self.assertEqual(out.state, NavState.ALIGN)
        self.assertEqual(out.linear_x, 0.0)
        self.assertLess(out.angular_z, 0.0)

    def test_forward_tracking_is_bounded_and_uses_measured_pose(self):
        out = self.controller.step(pose(), self.goal, now_steady_s=0.0)
        self.assertEqual(out.state, NavState.DRIVE)
        self.assertGreater(out.linear_x, 0.0)
        self.assertLessEqual(out.linear_x, self.config.max_forward_mps)
        self.assertLessEqual(abs(out.angular_z), self.config.max_angular_rps)

    def test_final_orientation_is_separate_zero_translation_phase(self):
        out = self.controller.step(
            pose(x=0.95), Goal2D(1.0, 0.0, 0.8, "odom"), now_steady_s=0.0
        )
        self.assertEqual(out.state, NavState.FINAL_ALIGN)
        self.assertEqual(out.linear_x, 0.0)
        self.assertGreater(out.angular_z, 0.0)

    def test_arrival_requires_position_and_yaw_dwell_then_latches_stop(self):
        reached = Goal2D(0.0, 0.0, 0.0, "odom")
        first = self.controller.step(
            pose(received=1.0), reached, now_steady_s=1.0
        )
        second = self.controller.step(
            pose(received=1.1), reached, now_steady_s=1.1
        )
        final = self.controller.step(
            pose(received=1.3), reached, now_steady_s=1.3
        )
        self.assertEqual(first.state, NavState.SETTLING)
        self.assertEqual(second.state, NavState.SETTLING)
        self.assertEqual(final.state, NavState.ARRIVED)
        self.assertEqual((final.linear_x, final.angular_z), (0.0, 0.0))
        self.assertEqual(
            self.controller.step(
                pose(received=1.4), reached, now_steady_s=1.4
            ).state,
            NavState.ARRIVED,
        )

    def test_stale_pose_frame_mismatch_cancel_and_timeout_all_stop(self):
        stale = self.controller.step(
            pose(received=0.0), self.goal, now_steady_s=0.21
        )
        self.assertEqual(stale.state, NavState.STALE_STATE)
        self.assertEqual((stale.linear_x, stale.angular_z), (0.0, 0.0))

        wrong_frame = PointGoalController(self.config).step(
            pose(frame="map"), self.goal, now_steady_s=0.0
        )
        self.assertEqual(wrong_frame.state, NavState.FRAME_MISMATCH)
        self.assertEqual((wrong_frame.linear_x, wrong_frame.angular_z), (0.0, 0.0))

        cancelled = PointGoalController(self.config).step(
            pose(), self.goal, now_steady_s=0.0, cancelled=True
        )
        self.assertEqual(cancelled.state, NavState.CANCELLED)
        self.assertEqual((cancelled.linear_x, cancelled.angular_z), (0.0, 0.0))

        timeout_controller = PointGoalController(self.config)
        timeout_controller.step(None, self.goal, now_steady_s=0.0)
        timeout = timeout_controller.step(None, self.goal, now_steady_s=2.1)
        self.assertEqual(timeout.state, NavState.TIMED_OUT)
        self.assertEqual((timeout.linear_x, timeout.angular_z), (0.0, 0.0))

    def test_backward_monotonic_clock_fails_closed(self):
        self.controller.step(pose(), self.goal, now_steady_s=1.0)
        out = self.controller.step(
            pose(received=0.9), self.goal, now_steady_s=0.9
        )
        self.assertEqual(out.state, NavState.CLOCK_DISCONTINUITY)
        self.assertEqual((out.linear_x, out.angular_z), (0.0, 0.0))


class SyntheticTrajectoryTests(unittest.TestCase):
    def test_unicycle_math_reaches_pose_goal_without_physics_or_ros(self):
        config = ControllerConfig(
            xy_tolerance_m=0.08, yaw_tolerance_rad=0.08,
            align_tolerance_rad=0.18, k_linear=0.7, k_angular=1.0,
            max_forward_mps=0.20, max_angular_rps=0.30,
            pose_timeout_s=0.15, goal_timeout_s=60.0,
            arrival_dwell_s=0.30,
        )
        goal = Goal2D(0.6, 0.3, 1.0, "odom")
        controller = PointGoalController(config)
        x = y = yaw = 0.0
        dt = 0.05
        output = None

        for index in range(1200):
            now = index * dt
            measured_pose = Pose2D(x, y, yaw, "odom", now)
            output = controller.step(measured_pose, goal, now_steady_s=now)
            if output.state == NavState.ARRIVED:
                break
            yaw = wrap_angle(yaw + output.angular_z * dt)
            x += output.linear_x * math.cos(yaw) * dt
            y += output.linear_x * math.sin(yaw) * dt

        self.assertIsNotNone(output)
        self.assertEqual(output.state, NavState.ARRIVED)
        self.assertLessEqual(math.hypot(goal.x - x, goal.y - y), config.xy_tolerance_m)
        self.assertLessEqual(abs(wrap_angle(goal.yaw - yaw)), config.yaw_tolerance_rad)
        self.assertLessEqual(abs(output.linear_x), config.max_forward_mps)
        self.assertLessEqual(abs(output.angular_z), config.max_angular_rps)


if __name__ == "__main__":
    unittest.main()