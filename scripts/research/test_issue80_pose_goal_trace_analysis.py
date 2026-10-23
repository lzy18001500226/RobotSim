#!/usr/bin/env python3
"""Offline regression tests for saved-pose Issue 80 controller replay."""

from __future__ import annotations

import math
import unittest

from issue80_pose_goal_trace_analysis import _counterfactual_pose_replay


class SavedPoseReplayTests(unittest.TestCase):
    def test_replay_uses_the_policy_recorded_for_each_historical_run(self) -> None:
        target_yaw = -0.10
        target = {
            "frame_id": "odom",
            "x": 0.30 * math.cos(target_yaw),
            "y": 0.30 * math.sin(target_yaw),
            "yaw": target_yaw,
        }
        start = {"frame_id": "odom", "x": 0.0, "y": 0.0, "yaw": 0.0}
        pose = {**start, "x": 0.10, "y": 0.05, "yaw": -0.05}
        state_rows = [{"phase": "activation", "snapshot": {"sample_monotonic_s": 10.2, "pose": pose}}]
        command = {
            "sample_monotonic_s": 10.2,
            "along_track_progress_m": 0.10,
            "activation_elapsed_s": 0.8,
            "angular_taper_factor": 1.0,
            "angular_velocity_rad_s": -0.075,
        }
        common = {
            "attempt_id": "policy-fixture",
            "start_pose": start,
            "target_pose": target,
            "dynamic_stop_target_pose_at_trigger": target,
            "dynamic_stop_progress_at_trigger_m": 0.30,
        }

        position_summary, position_rows, _ = _counterfactual_pose_replay(
            {
                **common,
                "command_profile_parameters": {
                    "heading_policy": "Run73_waypoint_bearing_to_current_ground_truth_goal",
                    "heading_gain": 0.8,
                    "maximum_angular_velocity_rad_s": 0.075,
                },
            }, state_rows, [10.2], [command], zero_start=None, stand_request_time=None,
        )
        direct_command = {**command, "angular_velocity_rad_s": -0.04}
        direct_summary, direct_rows, _ = _counterfactual_pose_replay(
            {
                **common,
                "command_profile_parameters": {
                    "heading_policy": "Run73 position-bearing for point goals; target-yaw tracking for heading-change goals",
                    "heading_gain": 0.8,
                    "maximum_angular_velocity_rad_s": 0.075,
                },
            }, state_rows, [10.2], [direct_command], zero_start=None, stand_request_time=None,
        )

        self.assertEqual(position_summary["legacy_policy_kind"], "RUN73_POSITION_BEARING")
        self.assertEqual(direct_summary["legacy_policy_kind"], "DIRECT_TARGET_YAW")
        self.assertAlmostEqual(position_rows[0]["legacy_policy_replayed_angular_command_rad_s"], -0.075)
        self.assertAlmostEqual(direct_rows[0]["legacy_policy_replayed_angular_command_rad_s"], -0.04)
        self.assertNotEqual(
            position_rows[0]["legacy_policy_replayed_angular_command_rad_s"],
            direct_rows[0]["legacy_policy_replayed_angular_command_rad_s"],
        )

    def test_replays_legacy_and_coordinated_commands_and_marks_missing_terminal_samples(self) -> None:
        target_yaw = -0.10
        target = {
            "frame_id": "odom",
            "x": 0.30 * math.cos(target_yaw),
            "y": 0.30 * math.sin(target_yaw),
            "yaw": target_yaw,
        }
        stop = {
            "frame_id": "odom",
            "x": target["x"] + 0.035 * math.cos(target_yaw),
            "y": target["y"] + 0.035 * math.sin(target_yaw),
        }
        start = {"frame_id": "odom", "x": 0.0, "y": 0.0, "yaw": 0.0}
        pose = {**start, "x": 0.14, "yaw": -0.01}
        command = {
            "sample_monotonic_s": 10.2,
            "along_track_progress_m": 0.14,
            "activation_elapsed_s": 0.8,
            "yaw_rate_feedback_rad_s": 0.02,
            "yaw_prediction_horizon_s": 1.0,
            "angular_velocity_rad_s": -0.075,
        }
        pg = {
            "attempt_id": "offline-fixture",
            "start_pose": start,
            "target_pose": target,
            "dynamic_stop_target_pose_at_trigger": stop,
            "dynamic_stop_progress_at_trigger_m": 0.335,
            "command_profile_parameters": {
                "heading_policy": "target-yaw tracking for heading-change goals",
                "heading_yaw_rate_prediction_horizon_s": 1.0,
                "heading_gain": 0.8,
                "maximum_angular_velocity_rad_s": 0.075,
            },
        }
        state_rows = [{"phase": "activation", "snapshot": {"sample_monotonic_s": 10.2, "pose": pose}}]

        summary, approach_rows, terminal_rows = _counterfactual_pose_replay(
            pg,
            state_rows,
            [10.2],
            [command],
            zero_start=11.0,
            stand_request_time=12.0,
        )

        self.assertEqual(summary["classification"], "COUNTERFACTUAL_SAVED_POSE_COMMAND_REPLAY_NOT_DYNAMICS_VALIDATION")
        self.assertEqual(summary["legacy_policy_kind"], "PREDICTED_TARGET_YAW")
        self.assertEqual(len(approach_rows), 1)
        self.assertAlmostEqual(
            approach_rows[0]["legacy_policy_replayed_angular_command_rad_s"],
            command["angular_velocity_rad_s"],
        )
        self.assertAlmostEqual(summary["legacy_replay_max_abs_command_error_rad_s"], 0.0)
        self.assertIsNotNone(approach_rows[0]["coordinated_curve_angular_command_rad_s"])
        self.assertEqual(summary["zero_to_stand_saved_state_sample_count"], 0)
        self.assertEqual(
            summary["terminal_replay_status"],
            "NOT_REPLAYED_NO_STATE_SAMPLES_IN_ZERO_TO_STAND_INTERVAL",
        )
        self.assertEqual(terminal_rows, [])


if __name__ == "__main__":
    unittest.main()
