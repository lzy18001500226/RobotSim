"""Synthetic-only tests for the PR #82 proposal-to-MC safety contract."""

from __future__ import annotations

import unittest

from issue80_m1_first_motion import MC_FSM_STABLE, VELOCITY_SOURCE_NAME
from issue80_navigation_bridge_gate import gate_navigation_twist_proposal


LIMITS = {
    "left_hip_pitch_joint": {
        "lower_position": -2.7,
        "upper_position": 2.5,
        "max_velocity": 11.936,
    }
}


def snapshot(**overrides):
    result = {
        "readback_fault": None,
        "state_age_s": {
            "odom": 0.01, "imu": 0.01, "common": 0.01,
            "leg_state": 0.01, "leg_command": 0.01,
            "get_mode": 0.01, "get_source": 0.01,
        },
        "get_mode_desc": "LOCOMOTION_DEFAULT",
        "common_mode_desc": "LOCOMOTION_DEFAULT",
        "get_mode_header_code": 0,
        "get_mode_status_value": 100,
        "common_mode_status_value": 100,
        "fsm_state": MC_FSM_STABLE,
        "body_status": 1,
        "pose": {"x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "forward_bounds": {"min_value": -1.0, "max_value": 0.8},
        "lateral_bounds": {"min_value": -1.0, "max_value": 1.0},
        "angular_bounds": {"min_value": -1.0, "max_value": 1.0},
        "velocity_bounds": {
            "forward": {"min_value": -1.0, "max_value": 0.8},
            "lateral": {"min_value": -1.0, "max_value": 1.0},
            "angular": {"min_value": -1.0, "max_value": 1.0},
        },
        "get_source_header_code": 0,
        "get_source_name": VELOCITY_SOURCE_NAME,
        "common_source_name": VELOCITY_SOURCE_NAME,
        "leg_states": [
            {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 0.0}
        ],
    }
    result.update(overrides)
    return result


def evaluate(
    proposal=None,
    *,
    status="DRIVE",
    status_received=9.95,
    received=10.0,
    now=10.05,
    state=None,
    registered=True,
    transport=True,
    handshake=True,
    cancelled=False,
):
    return gate_navigation_twist_proposal(
        proposal or {
            "linear_x": 0.15, "linear_y": 0.0, "linear_z": 0.0,
            "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.1,
        },
        nav_status=status,
        nav_status_received_steady_s=status_received,
        received_steady_s=received,
        now_steady_s=now,
        state_snapshot=state or snapshot(),
        joint_limits=LIMITS,
        source_registration_verified=registered,
        transport_preflight_verified=transport,
        same_sample_handshake_confirmed=handshake,
        cancelled=cancelled,
    )


class NavigationBridgeGateTests(unittest.TestCase):
    def test_fresh_bounded_proposal_passes_but_does_not_claim_motion(self):
        result = evaluate()
        self.assertEqual(result["decision"], "ALLOW_PROPOSAL")
        self.assertTrue(result["forward_to_mc"])
        self.assertEqual(result["linear_x"], 0.15)
        self.assertEqual(result["angular_z"], 0.1)
        self.assertEqual(result["physical_motion_status"], "NOT_MEASURED")

    def test_unverified_transport_never_routes_even_a_zero_to_mc(self):
        result = evaluate(transport=False)
        self.assertEqual(result["decision"], "BLOCKED")
        self.assertFalse(result["forward_to_mc"])
        self.assertEqual(result["linear_x"], 0.0)

    def test_unknown_unsafe_or_non_locomotion_modes_fail_closed_to_zero(self):
        cases = (
            snapshot(fsm_state=0),
            snapshot(fsm_state=4),
            snapshot(get_mode_desc="STAND_DEFAULT", common_mode_desc="STAND_DEFAULT"),
            snapshot(get_mode_desc="LOCOMOTION_DEFAULT", common_mode_desc="PASSIVE_DEFAULT"),
        )
        for state in cases:
            with self.subTest(state=state):
                result = evaluate(state=state)
                self.assertEqual(result["decision"], "ZERO")
                self.assertEqual(result["linear_x"], 0.0)
                self.assertEqual(result["angular_z"], 0.0)

    def test_stale_telemetry_or_unselected_source_routes_only_zero(self):
        stale = snapshot(state_age_s={"odom": 1.0})
        unselected = snapshot(get_source_name="", common_source_name="")
        for state in (stale, unselected):
            with self.subTest(state=state):
                result = evaluate(state=state)
                self.assertEqual(result["decision"], "ZERO")
                self.assertEqual(result["linear_x"], 0.0)
                self.assertEqual(result["angular_z"], 0.0)

    def test_source_registration_and_same_sample_handshake_are_required(self):
        self.assertEqual(evaluate(registered=False)["decision"], "ZERO")
        self.assertEqual(evaluate(handshake=False)["decision"], "ZERO")

    def test_stale_proposal_cancel_and_terminal_state_are_zeroed(self):
        stale_proposal = evaluate(now=10.21)
        cancelled = evaluate(cancelled=True)
        terminal = evaluate(status="STOPPED:CANCELLED")
        self.assertEqual(stale_proposal["reason"], "proposal_stale")
        self.assertEqual(cancelled["reason"], "cancelled")
        self.assertEqual(terminal["reason"], "proposal_status_not_active")
        for result in (stale_proposal, cancelled, terminal):
            with self.subTest(result=result):
                self.assertTrue(result["forward_to_mc"])
                self.assertEqual(result["linear_x"], 0.0)
                self.assertEqual(result["angular_z"], 0.0)
        stale_status = evaluate(status_received=0.0, received=30.95, now=31.0)
        self.assertEqual(stale_status["reason"], "nav_status_stale")
        self.assertTrue(stale_status["forward_to_mc"])
        self.assertEqual(stale_status["linear_x"], 0.0)

    def test_nonplanar_overbound_and_state_inconsistent_outputs_are_zeroed(self):
        nonplanar = {
            "linear_x": 0.1, "linear_y": 0.01, "linear_z": 0.0,
            "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,
        }
        too_fast = {
            "linear_x": 0.21, "linear_y": 0.0, "linear_z": 0.0,
            "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,
        }
        rotation_with_translation = {
            "linear_x": 0.01, "linear_y": 0.0, "linear_z": 0.0,
            "angular_x": 0.0, "angular_y": 0.0, "angular_z": 0.0,
        }
        self.assertEqual(evaluate(nonplanar)["reason"], "proposal_nonplanar")
        self.assertEqual(evaluate(too_fast)["reason"], "proposal_outside_adapter_bounds")
        self.assertEqual(evaluate(rotation_with_translation, status="ALIGN")["reason"], "rotation_state_has_forward_command")

    def test_live_speed_bounds_and_joint_safety_are_enforced(self):
        narrow = snapshot(velocity_bounds={
            "forward": {"min_value": 0.0, "max_value": 0.1},
            "lateral": {"min_value": 0.0, "max_value": 0.0},
            "angular": {"min_value": -0.05, "max_value": 0.05},
        })
        joint_violation = snapshot(leg_states=[
            {"name": "left_hip_pitch_joint", "position": 3.0, "velocity": 0.0}
        ])
        self.assertEqual(evaluate(state=narrow)["reason"], "proposal_outside_live_forward_bounds")
        self.assertTrue(evaluate(state=joint_violation)["reason"].startswith("unsafe_mc_snapshot:"))


if __name__ == "__main__":
    unittest.main()
