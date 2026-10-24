from __future__ import annotations

import unittest

from issue80_m1_first_motion import MC_FSM_STABLE
from issue80_stand_baseline_analysis import (
    analyze_standing_trace,
    normalize_trace_samples,
    rolling_windows,
)


def sample(time_s: float, *, right_ankle_target: float = 0.0, pitch_rad: float = 0.0):
    names = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
    return {
        "sample_monotonic_s": time_s,
        "state_age_s": {key: 0.01 for key in ("odom", "imu", "common", "leg_state")},
        "pose": {
            "x": 0.0, "y": 0.0, "z": 0.65, "roll": 0.0, "pitch": pitch_rad,
            "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0,
        },
        "imu": {"angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "leg_states": [
            {"name": name, "position": 0.0, "velocity": 0.0, "error_code": 0}
            for name in names
        ],
        "mc_leg_command": {"joints": [
            {"name": "left_ankle_pitch_joint", "position": 0.0},
            {"name": "right_ankle_pitch_joint", "position": right_ankle_target},
        ]},
        "common_mode_desc": "STAND_DEFAULT",
        "common_mode_status_value": 100,
        "fsm_state": MC_FSM_STABLE,
        "body_status": 1,
        "input_source": {"name": "", "priority": 0},
    }


class StandBaselineAnalysisTests(unittest.TestCase):
    def test_stable_baseline_reports_rolling_and_final_two_second_windows(self):
        samples = [sample(index * 0.05) for index in range(161)]
        windows = rolling_windows(samples, expected_body_status=1)

        self.assertEqual(len(windows), 13)
        self.assertEqual(windows[0]["metrics"]["duration_s"], 2.0)
        self.assertEqual(windows[-1]["relative_start_s"], 6.0)
        self.assertEqual(windows[-1]["metrics"]["physical_state_gate"]["status"], "PASS")
        self.assertEqual(windows[-1]["metrics"]["controller_target_stability_gate"]["status"], "PASS")
        right_ankle = windows[-1]["metrics"]["per_joint_leg_motion"]["right_ankle_pitch_joint"]
        self.assertEqual(right_ankle["target_rad"]["range"], 0.0)
        self.assertEqual(right_ankle["actual_position_rad"]["range"], 0.0)
        self.assertEqual(right_ankle["actual_velocity_rad_s"]["max_abs"], 0.0)

    def test_target_drift_above_existing_limit_is_detected_without_locomotion(self):
        samples = [sample(index * 0.05, right_ankle_target=0.0121 * index * 0.05) for index in range(161)]
        windows = rolling_windows(samples, expected_body_status=1)

        self.assertTrue(all(
            window["metrics"]["controller_target_stability_gate"]["status"] == "FAIL"
            for window in windows
        ))
        self.assertAlmostEqual(
            windows[-1]["metrics"]["controller_target_stability_gate"]["max_target_range_rad"],
            0.0242,
        )

    def test_common_state_fsm_current_state_is_preserved_as_raw_value(self):
        stamp = 1_000_000_000
        rows = []
        for key, data in (
            ("odom", {"x": 0.0, "y": 0.0, "z": 0.65, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
                      "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}}),
            ("imu", {"orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}}),
            ("common", {"fsm_state": {"current_state": 4}, "body_status": {"value": 1},
                        "action_info": {"action_desc": "STAND_DEFAULT", "status": {"value": 100}}, "input_source": {"name": ""}}),
            ("leg_state", {"joints": [{"name": "right_ankle_pitch_joint", "position": 0.0, "velocity": 0.0}]}),
            ("leg_command", {"joints": [{"name": "right_ankle_pitch_joint", "position": 0.0}]}),
        ):
            rows.append({"event": "state", "topic_key": key, "wall_monotonic_ns": stamp, "data": data})

        samples, counts = normalize_trace_samples(rows, start_ns=stamp, end_ns=stamp)
        self.assertEqual(counts["aligned_sample_count"], 1)
        self.assertEqual(samples[0]["fsm_state"], 4)

    def test_observation_shorter_than_eight_seconds_is_incomplete(self):
        takeover = {"stand_default_mode_confirmed_monotonic_ns": 2_000_000_000}
        observer = {"standing_baseline_observation": {
            "complete": False,
            "observed_duration_s": 7.9,
            "observed_end_monotonic_ns": 9_900_000_000,
        }}
        result = analyze_standing_trace([], takeover, observer)

        self.assertEqual(result["classification"], "STAND_BASELINE_INCOMPLETE")
        self.assertFalse(result["minimum_duration_pass"])

    def test_raw_trace_analysis_separates_stable_stand_from_target_drift(self):
        def raw_trace(target_slope: float):
            rows = []
            for index in range(161):
                time_s = index * 0.05 + 1.0
                stamp = int(time_s * 1e9)
                common = {
                    "action_info": {"action_desc": "STAND_DEFAULT", "status": {"value": 100}},
                    "fsm_state": {"value": MC_FSM_STABLE},
                    "body_status": {"value": 1},
                    "input_source": {"name": "", "priority": 0},
                }
                rows.extend([
                    {"event": "state", "topic_key": "odom", "wall_monotonic_ns": stamp, "data": {
                        "x": 0.0, "y": 0.0, "z": 0.65, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
                        "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    }},
                    {"event": "state", "topic_key": "imu", "wall_monotonic_ns": stamp, "data": {
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    }},
                    {"event": "state", "topic_key": "common", "wall_monotonic_ns": stamp, "data": common},
                    {"event": "state", "topic_key": "leg_state", "wall_monotonic_ns": stamp, "data": {"joints": [
                        {"name": "left_ankle_pitch_joint", "position": 0.0, "velocity": 0.0, "error_code": 0},
                        {"name": "right_ankle_pitch_joint", "position": 0.0, "velocity": 0.0, "error_code": 0},
                    ]}},
                    {"event": "state", "topic_key": "leg_command", "wall_monotonic_ns": stamp, "data": {"joints": [
                        {"name": "left_ankle_pitch_joint", "position": 0.0},
                        {"name": "right_ankle_pitch_joint", "position": target_slope * (time_s - 1.0)},
                    ]}},
                ])
            return rows

        readback = {"transport_state": "RESPONSE_RECEIVED", "action_desc": "STAND_DEFAULT", "action_status_value": 100}
        takeover = {
            "stand_default_mode_confirmed_monotonic_ns": 1_000_000_000,
            "stand_default_set_mc_action": {},
            "joint_to_stand_gate": {"raw_body_status": 1},
            "get_mc_action_readbacks": [readback],
        }
        observer = {"standing_baseline_observation": {
            "anchor": "STAND_DEFAULT_GETMC_READBACK", "complete": True,
            "observed_duration_s": 8.0, "observed_end_monotonic_ns": 9_000_000_000,
            "first_unsafe_monotonic_ns": None,
        }, "final_stand_mode_readback": readback, "final_common_input_source": {"name": "", "priority": 0}}

        stable = analyze_standing_trace(raw_trace(0.0), takeover, observer)
        drifting = analyze_standing_trace(raw_trace(0.0121), takeover, observer)
        self.assertEqual(stable["classification"], "TARGET_STABLE_DURING_ZERO_COMMAND_STANDING_POST_LOCOMOTION_TRANSITION_SUSPECTED")
        self.assertEqual(len(stable["rolling_windows"]), 13)
        self.assertEqual(drifting["classification"], "TARGET_VARIATION_REPRODUCED_DURING_ZERO_COMMAND_STANDING")
        self.assertAlmostEqual(
            drifting["final_2s_window"]["controller_target_stability_gate"]["max_target_range_rad"],
            0.0242,
        )
        self.assertEqual(drifting["fsm_trace_reconciliation"]["trace_raw_current_state_values"], [MC_FSM_STABLE])
        self.assertEqual(drifting["fsm_trace_reconciliation"]["observer_summary_mismatch"], True)

    def test_complete_eight_second_observation_allows_fresh_command_samples_inside_edges(self):
        rows = []
        for index in range(161):
            time_s = index * 0.05 + 1.0
            stamp = int(time_s * 1e9)
            common = {
                "action_info": {"action_desc": "STAND_DEFAULT", "status": {"value": 100}},
                "fsm_state": {"current_state": MC_FSM_STABLE},
                "body_status": {"value": 1},
                "input_source": {"name": "", "priority": 0},
            }
            states = {
                "odom": {"x": 0.0, "y": 0.0, "z": 0.65, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
                         "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "imu": {"orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "common": common,
                "leg_state": {"joints": [{"name": "right_ankle_pitch_joint", "position": 0.0, "velocity": 0.0}]},
            }
            for key, data in states.items():
                rows.append({"event": "state", "topic_key": key, "wall_monotonic_ns": stamp, "data": data})
            if index not in (0, 160):
                rows.append({"event": "state", "topic_key": "leg_command", "wall_monotonic_ns": stamp, "data": {
                    "joints": [{"name": "right_ankle_pitch_joint", "position": 0.0}],
                }})

        readback = {"transport_state": "RESPONSE_RECEIVED", "action_desc": "STAND_DEFAULT", "action_status_value": 100}
        takeover = {
            "stand_default_mode_confirmed_monotonic_ns": 1_000_000_000,
            "stand_default_set_mc_action": {},
            "joint_to_stand_gate": {"raw_body_status": 1},
            "get_mc_action_readbacks": [readback],
        }
        observer = {"standing_baseline_observation": {
            "anchor": "STAND_DEFAULT_GETMC_READBACK", "complete": True,
            "observed_duration_s": 8.0, "observed_end_monotonic_ns": 9_000_000_000,
            "first_unsafe_monotonic_ns": None,
        }, "final_stand_mode_readback": readback}
        result = analyze_standing_trace(rows, takeover, observer)
        self.assertNotEqual(result["classification"], "STAND_BASELINE_INCOMPLETE")
        self.assertEqual(result["aligned_trace_edge_coverage"]["status"], "PASS")
        self.assertLess(result["aligned_target_trace_duration_s"], 8.0)

    def test_early_fsm_transition_is_reported_separately_from_final_two_second_pass(self):
        samples = [sample(index * 0.05) for index in range(161)]
        samples[0]["fsm_state"] = 1
        windows = rolling_windows(samples, expected_body_status=1)
        self.assertEqual(windows[0]["metrics"]["physical_state_gate"]["status"], "FAIL")
        self.assertEqual(windows[-1]["metrics"]["physical_state_gate"]["status"], "PASS")
        self.assertEqual(windows[-1]["metrics"]["controller_target_stability_gate"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
