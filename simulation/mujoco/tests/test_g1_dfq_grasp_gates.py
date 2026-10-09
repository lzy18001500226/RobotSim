import unittest

import numpy as np

from simulation.mujoco.g1_dfq_grasp_gates import (
    CHANNELS,
    GATE_NAMES,
    evaluate_gates,
    effort_handoff_references,
    is_contact_control_state,
    load_ready_summary,
    minimum_safe_brake_scale,
    progressive_load_transfer_summary,
    select_effort_feedforward,
    solve_live_contact_wrench_allocation,
)


def passing_result():
    return {
        "passed": True,
        "events": [
            {"event": "CLOSE_COMPLETE"},
            {
                "event": "HOLD_VALIDATED",
                "contact_fraction": 0.91,
                "simultaneous_contact_steps": 3640,
            },
            {"event": "UPWARD_HAND_SUPPORT_ESTABLISHED", "measured_hand_support_n": 5.4},
            {"event": "LOAD_READY"},
            {"event": "AIRBORNE_1MM_REACHED", "lift_m": 0.001, "table_supported": False},
            {"event": "AIRBORNE_5MM_REACHED", "lift_m": 0.005, "table_supported": False},
            {"event": "SHORT_LIFT_REACHED", "lift_m": 0.0304, "table_supported": False},
            {"event": "POST_LIFT_HOLD_COMPLETE", "duration_s": 1.0, "lift_m": 0.0301},
            {"event": "RELEASE_COMPLETE"},
        ],
        "measured_hand_motion": {
            "CLOSE": {
                "channels": {
                    channel: {"minimum_qpos_rad": 0.0, "maximum_qpos_rad": 0.1}
                    for channel in CHANNELS
                }
            }
        },
        "release": {
            "passed": True,
            "final_right_hand_bottle_contact": False,
            "final_table_supported": True,
            "fingers_opened_time_s": 4.0,
        },
        "contact_force_cap_gate": {"passed": True},
        "progressive_load_transfer": {"passed": True},
    }


class GraspGateTests(unittest.TestCase):
    def test_effort_mode_handoff_preserves_bounded_position_preload(self):
        targets = {channel: 0.0 for channel in CHANNELS}
        targets["index_proximal"] = 0.305
        references = effort_handoff_references(targets)
        self.assertEqual(references, targets)
        actual_qpos = 0.301
        self.assertAlmostEqual(references["index_proximal"] - actual_qpos, 0.004)

    def test_effort_mode_handoff_rejects_missing_or_nonfinite_target(self):
        with self.assertRaises(ValueError):
            effort_handoff_references({})
        targets = {channel: 0.0 for channel in CHANNELS}
        targets["thumb_proximal_pitch"] = float("nan")
        with self.assertRaises(ValueError):
            effort_handoff_references(targets)

    @staticmethod
    def opposing_contacts():
        return [
            np.array([[-1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]]),
        ]

    def allocate(self, *, friction=1.4, effort_limit=0.50445):
        bases = self.opposing_contacts()
        torque_maps = [np.zeros((1, 3)), np.zeros((1, 3))]
        torque_maps[0][0, 1] = 0.04
        torque_maps[1][0, 1] = 0.04
        return solve_live_contact_wrench_allocation(
            bases,
            [np.array([-0.03, 0.0, 0.0]), np.array([0.03, 0.0, 0.0])],
            [(friction, friction), (friction, friction)],
            torque_maps,
            np.zeros(3),
            5.59,
            np.zeros(1),
            driver_effort_limit_nm=effort_limit,
            total_normal_limit_n=5.4,
            single_contact_limit_n=4.5,
            total_contact_limit_n=9.0,
            horizontal_force_tolerance_n=0.559,
            torque_tolerance_nm=0.08,
            friction_utilization_limit=0.85,
        )

    def test_live_contact_solver_allocates_opposing_friction_with_effort_headroom(self):
        result = self.allocate()
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["predicted_wrench_on_bottle"]["force_n"][2], 5.59, places=7)
        self.assertLessEqual(result["total_normal_force_n"], 5.4 + 1e-9)
        self.assertLessEqual(result["total_conservative_resultant_bound_n"], 9.0 + 1e-9)
        self.assertTrue(all(item["actual_resultant_n"] <= 4.5 + 1e-9 for item in result["per_contact"]))
        self.assertLessEqual(result["maximum_friction_utilization"], 0.85 + 1e-9)
        self.assertLessEqual(abs(result["driver_effort_nm"][0]), 0.50445 + 1e-9)

    def test_live_contact_solver_rejects_insufficient_friction(self):
        result = self.allocate(friction=0.2)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "live_contact_wrench_infeasible")

    def test_live_contact_solver_rejects_insufficient_effort_authority(self):
        result = self.allocate(effort_limit=0.01)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "live_contact_wrench_infeasible")

    def test_progressive_transfer_requires_noise_aware_unload_and_hand_support(self):
        rows = [
            {"table_normal_force_n": 4.9, "hand_vertical_support_force_n": 0.7},
            {"table_normal_force_n": 4.8, "hand_vertical_support_force_n": 0.8},
        ]
        result = progressive_load_transfer_summary(5.59, 0.01, 0.0, rows, 5.59)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["minimum_detectable_change_n"], 0.0559)

    def test_progressive_transfer_rejects_table_drop_without_matching_hand_support(self):
        rows = [
            {"table_normal_force_n": 5.4, "hand_vertical_support_force_n": -0.01},
            {"table_normal_force_n": 5.3, "hand_vertical_support_force_n": 0.0},
        ]
        result = progressive_load_transfer_summary(5.59, 0.01, 0.0, rows, 5.59)
        self.assertFalse(result["passed"])

    def test_progressive_transfer_requires_stable_contacts_and_left_hand_clear(self):
        rows = [
            {
                "table_normal_force_n": 4.9,
                "hand_vertical_support_force_n": 0.7,
                "contact_window_valid": 1,
                "left_hand_contact": 0,
            },
            {
                "table_normal_force_n": 4.8,
                "hand_vertical_support_force_n": 0.8,
                "contact_window_valid": 0,
                "left_hand_contact": 0,
            },
        ]
        result = progressive_load_transfer_summary(5.59, 0.01, 0.0, rows, 5.59)
        self.assertFalse(result["passed"])
        self.assertFalse(result["contact_windows_valid"])

        rows[1]["contact_window_valid"] = 1
        rows[1]["left_hand_contact"] = 1
        result = progressive_load_transfer_summary(5.59, 0.01, 0.0, rows, 5.59)
        self.assertFalse(result["passed"])
        self.assertFalse(result["left_hand_clear"])

    def test_load_ready_requires_airborne_force_and_contact_gates(self):
        row = {
            "table_normal_force_n": 0.02, "hand_vertical_support_force_n": 5.56,
            "horizontal_hand_force_n": 0.1, "hand_torque_norm_nm": 0.01,
            "maximum_friction_utilization": 0.5, "maximum_hand_qvel_rad_s": 0.4,
            "maximum_applied_driver_effort_nm": 0.4, "maximum_mimic_error_rad": 0.001,
            "table_supported": False, "contact_window_valid": True,
            "left_hand_contact": False,
        }
        self.assertTrue(load_ready_summary([row] * 20, 5.59)["passed"])
        row["table_supported"] = True
        self.assertFalse(load_ready_summary([row] * 20, 5.59)["passed"])

    def test_contact_gate_accepts_intended_effort_mode_and_rejects_uncontrolled_states(self):
        self.assertTrue(is_contact_control_state("CONTACT_PRELOAD"))
        self.assertTrue(is_contact_control_state("SIMULATION_ONLY_TORQUE_IMPEDANCE"))
        self.assertFalse(is_contact_control_state("FREE_CLOSE"))
        self.assertFalse(is_contact_control_state(""))

    def test_load_phase_uses_reallocation_without_changing_preload_allocation(self):
        static = {"thumb": 0.42, "index": 0.02}
        load = {"thumb": 0.37, "index": 0.05}
        self.assertEqual(select_effort_feedforward("thumb", static, load, False), 0.42)
        self.assertEqual(select_effort_feedforward("thumb", static, load, True), 0.37)
        self.assertEqual(select_effort_feedforward("index", static, load, True), 0.05)

    def test_load_phase_reallocation_requires_each_driver_channel(self):
        with self.assertRaises(KeyError):
            select_effort_feedforward("thumb", {"thumb": 0.0}, {}, True)

    def test_velocity_guard_uses_only_minimum_braking_needed(self):
        peak = lambda scale: max(0.60 - 0.25 * scale, 0.0)
        scale = minimum_safe_brake_scale(peak, 0.48)
        self.assertAlmostEqual(scale, 0.48, places=6)
        self.assertLessEqual(peak(scale), 0.48)
        self.assertGreater(peak(scale - 1e-4), 0.48)

    def test_velocity_guard_reports_no_feasible_braking_scale(self):
        peak = lambda scale: 0.60 - 0.05 * scale
        self.assertIsNone(minimum_safe_brake_scale(peak, 0.48))

    def test_all_isolated_gates_require_measured_physics_evidence(self):
        gates = evaluate_gates(passing_result())
        self.assertEqual(tuple(gates), GATE_NAMES)
        self.assertEqual(
            [gates[name]["status"] for name in GATE_NAMES],
            ["PASS"] * 10 + ["NOT RUN"],
        )

    def test_direct_arm_lift_does_not_require_fixed_wrist_load_ready(self):
        result = passing_result()
        self.assertIn("LOAD_READY", [event["event"] for event in result["events"]])
        gates = evaluate_gates(result)
        self.assertEqual(gates["PHYSICAL_LIFT_30MM"]["status"], "PASS")

    def test_static_or_command_only_data_cannot_pass_physical_gates(self):
        gates = evaluate_gates({"passed": True, "events": []})
        for name in GATE_NAMES[:-1]:
            self.assertEqual(gates[name]["status"], "FAIL")
            self.assertFalse(gates[name]["verified"])
        self.assertEqual(gates["PR_CI"]["status"], "NOT RUN")

    def test_missing_driver_motion_blocks_hand_actuation(self):
        result = passing_result()
        result["measured_hand_motion"]["CLOSE"]["channels"][CHANNELS[-1]][
            "maximum_qpos_rad"
        ] = 0.0005
        gates = evaluate_gates(result)
        self.assertEqual(gates["HAND_ACTUATION"]["status"], "FAIL")
        self.assertIn(
            CHANNELS[-1],
            gates["HAND_ACTUATION"]["evidence"]["missing_or_insufficient_channels"],
        )

    def test_lift_requires_table_support_to_be_lost(self):
        result = passing_result()
        next(event for event in result["events"] if event["event"] == "SHORT_LIFT_REACHED")[
            "table_supported"
        ] = True
        self.assertEqual(
            evaluate_gates(result)["PHYSICAL_LIFT_30MM"]["status"],
            "FAIL",
        )

    def test_each_airborne_height_requires_its_own_unloaded_event(self):
        result = passing_result()
        result["events"] = [
            event for event in result["events"]
            if event["event"] != "AIRBORNE_1MM_REACHED"
        ]
        gates = evaluate_gates(result)
        self.assertEqual(gates["AIRBORNE_1MM"]["status"], "FAIL")
        self.assertEqual(gates["AIRBORNE_5MM"]["status"], "PASS")

        result = passing_result()
        result["events"] = [
            event for event in result["events"]
            if event["event"] != "AIRBORNE_5MM_REACHED"
        ]
        self.assertEqual(evaluate_gates(result)["AIRBORNE_5MM"]["status"], "FAIL")

    def test_release_requires_open_clear_and_table_settled(self):
        result = passing_result()
        result["release"]["final_right_hand_bottle_contact"] = True
        self.assertEqual(evaluate_gates(result)["PHYSICAL_RELEASE"]["status"], "FAIL")

    def test_pr_ci_requires_a_supported_status_value(self):
        result = passing_result()
        result["pr_ci"] = {"status": "green"}
        self.assertEqual(evaluate_gates(result)["PR_CI"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
