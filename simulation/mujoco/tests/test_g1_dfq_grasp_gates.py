import unittest

from simulation.mujoco.g1_dfq_grasp_gates import (
    CHANNELS,
    GATE_NAMES,
    evaluate_gates,
    is_contact_control_state,
    minimum_safe_brake_scale,
    select_effort_feedforward,
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
    }


class GraspGateTests(unittest.TestCase):
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
            ["PASS", "PASS", "PASS", "PASS", "PASS", "NOT RUN"],
        )

    def test_direct_arm_lift_does_not_require_fixed_wrist_load_ready(self):
        result = passing_result()
        self.assertNotIn("LOAD_READY", [event["event"] for event in result["events"]])
        gates = evaluate_gates(result)
        self.assertEqual(gates["ISOLATED_PHYSICAL_LIFT_30MM"]["status"], "PASS")

    def test_static_or_command_only_data_cannot_pass_physical_gates(self):
        gates = evaluate_gates({"passed": True, "events": []})
        for name in GATE_NAMES[:-1]:
            self.assertEqual(gates[name]["status"], "FAIL")
            self.assertFalse(gates[name]["verified"])
        self.assertEqual(gates["FULL_WORKCELL_PICKUP"]["status"], "NOT RUN")

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
        result["events"][2]["table_supported"] = True
        self.assertEqual(
            evaluate_gates(result)["ISOLATED_PHYSICAL_LIFT_30MM"]["status"],
            "FAIL",
        )

    def test_release_requires_open_clear_and_table_settled(self):
        result = passing_result()
        result["release"]["final_right_hand_bottle_contact"] = True
        self.assertEqual(evaluate_gates(result)["PHYSICAL_RELEASE"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
