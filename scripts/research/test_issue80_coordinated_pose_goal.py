#!/usr/bin/env python3
"""Offline tests for the coordinated Issue 80 pose-goal path and terminal phase."""

from __future__ import annotations

import math
import unittest

from issue80_coordinated_pose_goal import (
    APPROACH,
    FAILED,
    TERMINAL_POSE_CONTROL,
    ZERO_STOP,
    POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD,
    POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD,
    POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD,
    PoseGoalPhaseController,
    active_locomotion_readback_confirmed,
    build_warmup_aware_pose_goal_path,
    build_warmup_aware_pose_goal_path,
    compensated_terminal_yaw_target_rad,
    fresh_yaw_rate_feedback,
    pose_goal_approach_path_reference,
    pose_goal_approach_target,
    pose_path_reference,
    sampled_pose_path_feasibility,
    pose_path_target_with_yaw_compensation,
    pure_pursuit_yaw_rate,
    terminal_yaw_rate_command,
)


class PosePathTests(unittest.TestCase):
    def make_goal(self, delta: float) -> tuple[dict, dict]:
        start = {"x": 0.2, "y": -0.1, "yaw": 0.3, "frame_id": "odom"}
        target_yaw = start["yaw"] + delta
        goal = {
            "x": start["x"] + 0.30 * math.cos(target_yaw),
            "y": start["y"] + 0.30 * math.sin(target_yaw),
            "yaw": target_yaw,
            "frame_id": "odom",
        }
        return start, goal

    def test_hermite_path_preserves_start_and_target_pose_tangents(self) -> None:
        start, goal = self.make_goal(-0.10)
        first = pose_path_reference(start, goal, 0.0)
        last = pose_path_reference(start, goal, 1.0)
        self.assertAlmostEqual(first["x_m"], start["x"], places=12)
        self.assertAlmostEqual(first["y_m"], start["y"], places=12)
        self.assertAlmostEqual(first["tangent_yaw_rad"], start["yaw"], places=12)
        self.assertAlmostEqual(last["x_m"], goal["x"], places=12)
        self.assertAlmostEqual(last["y_m"], goal["y"], places=12)
        self.assertAlmostEqual(last["tangent_yaw_rad"], goal["yaw"], places=12)
        self.assertTrue(math.isfinite(first["curvature_per_m"]))
        self.assertTrue(math.isfinite(last["curvature_per_m"]))

    def test_positive_and_negative_pose_paths_mirror_their_curvature(self) -> None:
        start = {"x": 0.2, "y": -0.1, "yaw": 0.0, "frame_id": "odom"}
        negative_goal = {"x": 0.30 * math.cos(-0.10) + start["x"], "y": 0.30 * math.sin(-0.10) + start["y"], "yaw": -0.10, "frame_id": "odom"}
        positive_goal = {"x": 0.30 * math.cos(0.10) + start["x"], "y": 0.30 * math.sin(0.10) + start["y"], "yaw": 0.10, "frame_id": "odom"}
        negative = pose_path_reference(start, negative_goal, 0.5)
        positive = pose_path_reference(start, positive_goal, 0.5)
        self.assertLess(negative["y_m"], start["y"])
        self.assertGreater(positive["y_m"], start["y"])
        self.assertLess(negative["curvature_per_m"], 0.0)
        self.assertGreater(positive["curvature_per_m"], 0.0)
        self.assertAlmostEqual(negative["curvature_per_m"], -positive["curvature_per_m"], places=10)

    def test_approach_path_endpoint_uses_drift_compensated_yaw_but_keeps_position(self) -> None:
        start, goal = self.make_goal(-0.10)
        compensated = pose_path_target_with_yaw_compensation(goal, 0.16)
        endpoint = pose_path_reference(start, compensated, 1.0)
        self.assertAlmostEqual(compensated["x"], goal["x"], places=12)
        self.assertAlmostEqual(compensated["y"], goal["y"], places=12)
        expected = math.atan2(math.sin(goal["yaw"] - 0.16), math.cos(goal["yaw"] - 0.16))
        self.assertAlmostEqual(compensated["yaw"], expected, places=12)
        self.assertAlmostEqual(endpoint["tangent_yaw_rad"], expected, places=12)

    def test_balanced_pre_stop_endpoint_is_position_bounded_and_rate_feasible(self) -> None:
        start = {"x": 0.0, "y": 0.0, "yaw": 0.05, "frame_id": "odom"}
        target_yaw = -0.05
        stop = {
            "x": 0.30 * math.cos(target_yaw),
            "y": 0.30 * math.sin(target_yaw),
            "frame_id": "odom",
        }
        endpoint = pose_goal_approach_target(start, stop, target_yaw, 0.05)
        self.assertAlmostEqual(endpoint["yaw"], -0.10, places=12)
        self.assertAlmostEqual(endpoint["approach_chord_yaw_rad"], -0.025, places=12)
        self.assertAlmostEqual(endpoint["approach_endpoint_offset_m"], 0.0075, places=4)
        self.assertLess(endpoint["approach_endpoint_offset_m"], 0.05)

        geometry = sampled_pose_path_feasibility(
            start,
            endpoint,
            reference_line_pose=stop,
            forward_command_m_s=0.14,
            maximum_angular_rad_s=0.075,
            zero_angular_warmup_s=0.5,
            warmup_forward_m_s=0.12,
            terminal_yaw_control_budget_s=2.6,
        )
        self.assertTrue(geometry["curvature_rate_feasible_at_command"])
        self.assertTrue(geometry["pure_pursuit_rate_feasible_at_command"])
        self.assertAlmostEqual(geometry["maximum_absolute_curvature_per_m"], 0.5023, places=3)
        self.assertAlmostEqual(geometry["required_angular_rate_at_forward_command_rad_s"], 0.0703, places=3)
        self.assertEqual(geometry["pure_pursuit_saturation_fraction"], 0.0)
        self.assertLess(geometry["maximum_deviation_from_stop_chord_m"], 0.011)
        self.assertLess(geometry["maximum_deviation_from_requested_stop_line_m"], 0.011)
        self.assertAlmostEqual(geometry["minimum_yaw_time_at_angular_cap_s"], 2.0, places=12)
        self.assertAlmostEqual(geometry["warmup_commanded_distance_m"], 0.06, places=12)
        self.assertAlmostEqual(geometry["remaining_yaw_correction_after_warmup_rad"], 0.15, places=12)
        self.assertFalse(geometry["remaining_yaw_feasible_after_warmup"])
        self.assertTrue(geometry["remaining_yaw_feasible_with_terminal_budget"])
        self.assertAlmostEqual(geometry["remaining_yaw_time_to_terminal_phase_at_cap_s"], 0.2837, places=3)

    def test_old_full_drift_endpoint_is_infeasible_above_forward_deadzone(self) -> None:
        start = {"x": 0.0, "y": 0.0, "yaw": 0.05, "frame_id": "odom"}
        target_yaw = -0.05
        exact_stop = {
            "x": 0.30 * math.cos(target_yaw),
            "y": 0.30 * math.sin(target_yaw),
            "yaw": target_yaw - 0.16,
            "frame_id": "odom",
        }
        geometry = sampled_pose_path_feasibility(
            start,
            exact_stop,
            forward_command_m_s=0.14,
            maximum_angular_rad_s=0.075,
        )
        self.assertFalse(geometry["curvature_rate_feasible_at_command"])
        self.assertFalse(geometry["pure_pursuit_rate_feasible_at_command"])
        self.assertGreater(geometry["required_angular_rate_at_forward_command_rad_s"], 0.20)
        self.assertLess(geometry["maximum_forward_speed_under_angular_cap_m_s"], 0.111)
        self.assertGreater(geometry["pure_pursuit_saturation_fraction"], 0.80)

    def test_warmup_aligned_path_has_bounded_position_and_angular_demand(self) -> None:
        start = {"frame_id": "odom", "x": 0.0183720733169952, "y": 0.003391006556233643, "yaw": 0.053050859283639784}
        target_yaw = -0.04694914071636022
        stop = {
            "frame_id": "odom",
            "x": start["x"] + 0.30 * math.cos(target_yaw),
            "y": start["y"] + 0.30 * math.sin(target_yaw),
        }
        path = build_warmup_aware_pose_goal_path(
            start, stop, target_yaw, 0.05,
            warmup_duration_s=0.5,
            warmup_forward_m_s=0.12,
            curve_forward_m_s=0.13,
            maximum_angular_rad_s=0.075,
            maximum_endpoint_offset_m=0.05,
            angular_utilization_limit=0.95,
        )
        gate = path["feasibility"]
        self.assertAlmostEqual(path["warmup_distance_m"], 0.06, places=12)
        self.assertAlmostEqual(path["curve_chord_length_m"], 0.275, places=12)
        self.assertLess(path["endpoint_position_offset_m"], 0.05)
        self.assertLess(gate["maximum_lateral_error_to_requested_stop_line_m"], 0.05)
        self.assertLessEqual(gate["maximum_ideal_or_warmup_transition_raw_angular_rate_rad_s"], 0.075 * 0.95)
        self.assertFalse(gate["warmup_transition_saturation_required"])
        self.assertLess(gate["terminal_entry_position_error_m"], 0.05)
        self.assertLess(gate["terminal_entry_lateral_error_m"], 0.05)
        self.assertTrue(gate["terminal_residual_and_dwell_fit"])
        self.assertLess(gate["terminal_entry_residual_to_pre_stop_target_rad"], 0.02)

        warmup_reference = pose_goal_approach_path_reference(
            start, path, path["warmup_distance_m"] / path["total_path_length_m"],
        )
        endpoint_reference = pose_goal_approach_path_reference(start, path, 1.0)
        self.assertAlmostEqual(warmup_reference["tangent_yaw_rad"], start["yaw"], places=12)
        self.assertAlmostEqual(endpoint_reference["tangent_yaw_rad"], path["curve_endpoint_pose"]["yaw"], places=12)

    def test_pure_pursuit_command_sign_and_cap_for_both_turn_directions(self) -> None:
        left = pure_pursuit_yaw_rate(
            current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
            carrot_x_m=1.0, carrot_y_m=1.0, forward_command_m_s=0.14,
            maximum_angular_rad_s=0.075,
        )
        right = pure_pursuit_yaw_rate(
            current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
            carrot_x_m=1.0, carrot_y_m=-1.0, forward_command_m_s=0.14,
            maximum_angular_rad_s=0.075,
        )
        self.assertGreater(left["command_yaw_rate_rad_s"], 0.0)
        self.assertLess(right["command_yaw_rate_rad_s"], 0.0)
        self.assertLessEqual(abs(left["command_yaw_rate_rad_s"]), 0.075)
        self.assertLessEqual(abs(right["command_yaw_rate_rad_s"]), 0.075)


class TerminalYawControlTests(unittest.TestCase):
    def test_empirical_drift_envelope_keeps_internal_target_inside_final_gate(self) -> None:
        final_tolerance = 0.17
        bias = POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD
        worst_predicted_final_error = max(
            abs(drift - bias) + POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD
            for drift in POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD
        )
        self.assertAlmostEqual(bias, 0.05)
        self.assertLess(worst_predicted_final_error, final_tolerance)
        self.assertAlmostEqual(final_tolerance - worst_predicted_final_error, 0.0213100865, places=6)

    def test_post_stop_drift_compensation_is_signed_and_wrap_safe(self) -> None:
        self.assertAlmostEqual(
            compensated_terminal_yaw_target_rad(-0.05, 0.16),
            -0.21,
        )
        wrapped = compensated_terminal_yaw_target_rad(math.pi - 0.05, 0.16)
        self.assertAlmostEqual(wrapped, math.atan2(math.sin(math.pi - 0.21), math.cos(math.pi - 0.21)))

    def test_terminal_entry_requires_confirmed_same_sample_mode_and_source(self) -> None:
        snapshot = {
            "get_mode_header_code": 0,
            "get_mode_desc": "LOCOMOTION_DEFAULT",
            "common_mode_desc": "LOCOMOTION_DEFAULT",
            "get_source_header_code": 0,
            "get_source_name": "node.locomotion_velocity",
            "common_source_name": "node.locomotion_velocity",
        }
        self.assertTrue(active_locomotion_readback_confirmed(
            snapshot,
            expected_source="node.locomotion_velocity",
            control_handshake_confirmed=True,
        ))
        self.assertFalse(active_locomotion_readback_confirmed(
            snapshot,
            expected_source="node.locomotion_velocity",
            control_handshake_confirmed=False,
        ))
        for key, value in (
            ("get_mode_desc", "STAND_DEFAULT"),
            ("common_source_name", "another_source"),
            ("get_source_header_code", 1),
        ):
            with self.subTest(key=key):
                invalid = dict(snapshot)
                invalid[key] = value
                self.assertFalse(active_locomotion_readback_confirmed(
                    invalid,
                    expected_source="node.locomotion_velocity",
                    control_handshake_confirmed=True,
                ))

    def test_terminal_yaw_command_wraps_and_tracks_both_signs(self) -> None:
        positive = terminal_yaw_rate_command(
            current_yaw_rad=0.0, target_yaw_rad=0.2,
            measured_yaw_rate_rad_s=0.0, maximum_angular_rad_s=0.075,
        )
        negative = terminal_yaw_rate_command(
            current_yaw_rad=0.0, target_yaw_rad=-0.2,
            measured_yaw_rate_rad_s=0.0, maximum_angular_rad_s=0.075,
        )
        wrapped = terminal_yaw_rate_command(
            current_yaw_rad=math.pi - 0.05, target_yaw_rad=-math.pi + 0.05,
            measured_yaw_rate_rad_s=0.0, maximum_angular_rad_s=0.075,
        )
        self.assertEqual(positive["command_yaw_rate_rad_s"], 0.075)
        self.assertEqual(negative["command_yaw_rate_rad_s"], -0.075)
        self.assertAlmostEqual(wrapped["yaw_error_rad"], 0.1, places=12)

    def test_terminal_yaw_rate_feedback_damps_overshoot(self) -> None:
        command = terminal_yaw_rate_command(
            current_yaw_rad=0.0,
            target_yaw_rad=0.1,
            measured_yaw_rate_rad_s=0.08,
            maximum_angular_rad_s=0.075,
        )
        self.assertEqual(command["command_yaw_rate_rad_s"], -0.03)
        self.assertLessEqual(abs(command["command_yaw_rate_rad_s"]), 0.075)

    def test_terminal_yaw_command_is_zero_only_when_angle_and_rate_are_both_settled(self) -> None:
        settled = terminal_yaw_rate_command(
            current_yaw_rad=0.0,
            target_yaw_rad=0.10,
            measured_yaw_rate_rad_s=0.01,
            maximum_angular_rad_s=0.075,
        )
        not_settled = terminal_yaw_rate_command(
            current_yaw_rad=0.0,
            target_yaw_rad=0.10,
            measured_yaw_rate_rad_s=0.03,
            maximum_angular_rad_s=0.075,
        )
        self.assertTrue(settled["within_yaw_and_rate_gates"])
        self.assertEqual(settled["command_yaw_rate_rad_s"], 0.0)
        self.assertFalse(not_settled["within_yaw_and_rate_gates"])
        self.assertNotEqual(not_settled["command_yaw_rate_rad_s"], 0.0)

    def test_freshness_gate_rejects_missing_stale_and_future_feedback(self) -> None:
        fresh = fresh_yaw_rate_feedback(
            {"signed_net_yaw_rate_rad_s": 0.02, "sample_monotonic_s": 10.0},
            sample_monotonic_s=10.1,
            maximum_age_s=0.15,
        )
        self.assertEqual(fresh, {"signed_yaw_rate_rad_s": 0.02, "age_s": 0.09999999999999964})
        for feedback, now in ((None, 10.1), ({"signed_net_yaw_rate_rad_s": 0.02, "sample_monotonic_s": 9.9}, 10.1), ({"signed_net_yaw_rate_rad_s": 0.02, "sample_monotonic_s": 10.2}, 10.1)):
            with self.subTest(feedback=feedback, now=now), self.assertRaisesRegex(ValueError, "unavailable|stale or from the future"):
                fresh_yaw_rate_feedback(feedback, sample_monotonic_s=now, maximum_age_s=0.15)


class PoseGoalPhaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = PoseGoalPhaseController()

    def update(self, **overrides):
        values = {
            "monotonic_s": 1.0,
            "position_stop_reason": None,
            "distance_to_goal_m": 0.20,
            "lateral_error_m": 0.0,
            "along_track_progress_m": 0.10,
            "yaw_error_rad": 0.0,
            "measured_yaw_rate_rad_s": 0.0,
        }
        values.update(overrides)
        return self.controller.update(**values)

    def test_position_before_yaw_enters_terminal_pose_control_not_zero_stop(self) -> None:
        decision = self.update(
            position_stop_reason="goal_approach_predicted_settle_within_tolerance",
            distance_to_goal_m=0.03,
            along_track_progress_m=0.331,
            yaw_error_rad=0.20,
            measured_yaw_rate_rad_s=0.0,
        )
        self.assertEqual(decision["phase"], TERMINAL_POSE_CONTROL)
        self.assertFalse(decision["complete"])
        self.assertFalse(decision["terminal_pose_ready"])

    def test_yaw_before_position_keeps_approach_active(self) -> None:
        decision = self.update(
            position_stop_reason=None,
            distance_to_goal_m=0.12,
            along_track_progress_m=0.18,
            yaw_error_rad=0.0,
            measured_yaw_rate_rad_s=0.0,
        )
        self.assertEqual(decision["phase"], APPROACH)
        self.assertFalse(decision["complete"])

    def test_terminal_requires_pose_rate_and_continuous_dwell_before_zero_stop(self) -> None:
        entered = self.update(
            monotonic_s=2.0,
            position_stop_reason="speed_scaled_stop_line_crossed",
            distance_to_goal_m=0.04,
            along_track_progress_m=0.335,
            yaw_error_rad=0.10,
            measured_yaw_rate_rad_s=0.0,
        )
        self.assertEqual(entered["phase"], TERMINAL_POSE_CONTROL)

        outside_yaw = self.update(
            monotonic_s=2.1, distance_to_goal_m=0.03,
            along_track_progress_m=0.335, yaw_error_rad=0.18,
            measured_yaw_rate_rad_s=0.0,
        )
        self.assertFalse(outside_yaw["complete"])

        first_ready = self.update(
            monotonic_s=2.2, distance_to_goal_m=0.03,
            along_track_progress_m=0.335, yaw_error_rad=0.10,
            measured_yaw_rate_rad_s=0.01,
        )
        self.assertEqual(first_ready["phase"], TERMINAL_POSE_CONTROL)
        self.assertFalse(first_ready["complete"])

        complete = self.update(
            monotonic_s=2.61, distance_to_goal_m=0.03,
            along_track_progress_m=0.335, yaw_error_rad=0.10,
            measured_yaw_rate_rad_s=0.01,
        )
        self.assertEqual(complete["phase"], ZERO_STOP)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["terminal_pose_ready"])

    def test_compensated_setpoint_must_settle_without_weakening_final_goal_gate(self) -> None:
        entered = self.update(
            monotonic_s=1.0,
            position_stop_reason="speed_scaled_stop_line_crossed",
            distance_to_goal_m=0.03,
            along_track_progress_m=0.30,
            yaw_error_rad=0.15,
            measured_yaw_rate_rad_s=0.0,
            terminal_setpoint_error_rad=0.15,
        )
        self.assertEqual(entered["phase"], TERMINAL_POSE_CONTROL)

        at_compensated_target = self.update(
            monotonic_s=1.1,
            distance_to_goal_m=0.03,
            along_track_progress_m=0.30,
            yaw_error_rad=0.15,
            measured_yaw_rate_rad_s=0.01,
            terminal_setpoint_error_rad=0.005,
        )
        self.assertFalse(at_compensated_target["complete"])
        self.assertEqual(at_compensated_target["phase"], TERMINAL_POSE_CONTROL)

        outside_final_goal = self.update(
            monotonic_s=1.2,
            distance_to_goal_m=0.03,
            along_track_progress_m=0.30,
            yaw_error_rad=0.171,
            measured_yaw_rate_rad_s=0.01,
            terminal_setpoint_error_rad=0.005,
        )
        self.assertFalse(outside_final_goal["complete"])

        first_ready = self.update(
            monotonic_s=1.61,
            distance_to_goal_m=0.03,
            along_track_progress_m=0.30,
            yaw_error_rad=0.15,
            measured_yaw_rate_rad_s=0.01,
            terminal_setpoint_error_rad=0.005,
        )
        self.assertFalse(first_ready["complete"])
        ready = self.update(
            monotonic_s=2.02,
            distance_to_goal_m=0.03,
            along_track_progress_m=0.30,
            yaw_error_rad=0.15,
            measured_yaw_rate_rad_s=0.01,
            terminal_setpoint_error_rad=0.005,
        )
        self.assertEqual(ready["phase"], ZERO_STOP)
        self.assertTrue(ready["terminal_pose_ready"])

    def test_stale_rate_position_escape_and_terminal_timeout_fail_closed(self) -> None:
        self.update(
            monotonic_s=1.0,
            position_stop_reason="speed_scaled_stop_line_crossed",
            distance_to_goal_m=0.03,
            along_track_progress_m=0.335,
        )
        stale = self.update(monotonic_s=1.1, distance_to_goal_m=0.03, measured_yaw_rate_rad_s=None)
        self.assertEqual(stale["phase"], FAILED)
        self.assertEqual(stale["failure_reason"], "pose_goal_terminal_yaw_rate_unavailable")

        self.controller = PoseGoalPhaseController()
        self.update(
            monotonic_s=1.0,
            position_stop_reason="speed_scaled_stop_line_crossed",
            distance_to_goal_m=0.03,
            along_track_progress_m=0.335,
        )
        outside = self.update(monotonic_s=1.1, distance_to_goal_m=0.051)
        self.assertEqual(outside["failure_reason"], "pose_goal_terminal_position_tolerance_exceeded")

        self.controller = PoseGoalPhaseController()
        self.update(
            monotonic_s=1.0,
            position_stop_reason="speed_scaled_stop_line_crossed",
            distance_to_goal_m=0.03,
            along_track_progress_m=0.335,
        )
        timeout = self.update(
            monotonic_s=4.0, distance_to_goal_m=0.03,
            along_track_progress_m=0.335, yaw_error_rad=0.2,
        )
        self.assertEqual(timeout["failure_reason"], "pose_goal_terminal_yaw_timeout")


if __name__ == "__main__":
    unittest.main()
