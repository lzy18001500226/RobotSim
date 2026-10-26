"""Focused offline tests for the RAICOM standing-drift forensic replay."""
from __future__ import annotations

import unittest

from issue80_raicom_standing_drift_analysis import native_metrics, replay_hold


def common_state() -> dict:
    return {
        "action_info": {"action_desc": "STAND_DEFAULT", "status": {"value": 100}},
        "body_status": {"value": 1},
        "fsm_state": {"current_state": 4},
    }


def readback() -> dict:
    return {
        "header": {"code": 0},
        "info": {"action_desc": "STAND_DEFAULT", "status": {"value": 100}},
    }


def sample(index: int) -> dict:
    receipt = 1_000_000_000 + index * 100_000_000
    return {
        "event": "post_reset_odom_sample",
        "receipt_monotonic_ns": receipt,
        "source_stamp_ns": receipt,
        "fresh_all_required_topics": True,
        "raw_common_state": common_state(),
        "pose": {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.68,
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "linear_velocity_m_s": [0.0, 0.0, 0.0],
        },
        "leg_command": {
            "joints": [
                {
                    "name": "joint",
                    "position": 0.0,
                    "velocity": 0.0,
                    "effort": 0.0,
                    "stiffness": 1.0,
                    "damping": 1.0,
                }
            ]
        },
        "leg_state": {
            "joints": [
                {"name": "joint", "position": 0.0, "velocity": 0.0, "error_code": 0}
            ]
        },
    }


class StandingDriftAnalysisTests(unittest.TestCase):
    def test_replays_exact_continuous_hold_without_initial_reset_joint_noise(self):
        rows = [
            {"event": "JD_REQUEST"},
            {"event": "SD_REQUEST"},
            {
                "event": "NATIVE_TRANSITION",
                "phase": "SD",
                "state": "SD_READY",
                "receipt_monotonic_ns": 1_000_000_000,
            },
        ]
        for index in range(27):
            rows.append(
                {
                    "event": "GET_MC_ACTION",
                    "receipt_monotonic_ns": 1_000_000_000 + index * 100_000_000,
                    "response": readback(),
                }
            )
            rows.append(sample(index))
        result = replay_hold(
            rows,
            {"joint": {"lower_position": -1.0, "upper_position": 1.0, "max_velocity": 2.0}},
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["longest_continuous_qualified_window"]["samples"], 27)
        self.assertGreaterEqual(
            result["longest_continuous_qualified_window"]["duration_s"], 2.5
        )
        self.assertEqual(result["qualified_window_joint_limit_violations"], [])

    def test_native_metrics_keeps_bilateral_support_and_world_speed_separate(self):
        contact_left = {
            "geom1": {"name": "floor", "body_name": "world"},
            "geom2": {"name": "left_foot", "body_name": "left_ankle_roll_link"},
            "involves_selected_left_foot": True,
            "involves_selected_right_foot": False,
            "normal_force_n": 100.0,
            "tangential_force_norm_n": 2.0,
            "relative_tangential_speed_m_s": 0.01,
            "distance_m": -0.0002,
        }
        contact_right = {
            "geom1": {"name": "floor", "body_name": "world"},
            "geom2": {"name": "right_foot", "body_name": "right_ankle_roll_link"},
            "involves_selected_left_foot": False,
            "involves_selected_right_foot": True,
            "normal_force_n": 110.0,
            "tangential_force_norm_n": 3.0,
            "relative_tangential_speed_m_s": 0.02,
            "distance_m": -0.0003,
        }
        rows = []
        for index, velocity in enumerate(([0.01, 0.02], [0.02, 0.01])):
            rows.append(
                {
                    "event": "post_step",
                    "generation": 1,
                    "host_monotonic_ns": index + 1,
                    "sim_time_s": index * 0.01,
                    "pelvis": {
                        "state": {
                            "world_position_m": [index * 0.001, 0.0, 0.68],
                            "world_spatial_velocity_rot_lin": [0, 0, 0, velocity[0], velocity[1], 0],
                        }
                    },
                    "contacts": [contact_left, contact_right],
                    "actuator_force": [4.0],
                    "qfrc_actuator": [5.0],
                }
            )
        result = native_metrics(rows, 1, 2)
        self.assertTrue(result["bilateral_foot_floor_contact_every_native_sample"])
        self.assertAlmostEqual(result["max_world_planar_speed_m_s"], (0.02**2 + 0.01**2) ** 0.5)
        self.assertEqual(result["max_left_foot_floor_normal_force_n"], 100.0)
        self.assertEqual(result["max_right_foot_floor_normal_force_n"], 110.0)
        self.assertEqual(result["max_abs_actuator_force"], 4.0)


if __name__ == "__main__":
    unittest.main()
