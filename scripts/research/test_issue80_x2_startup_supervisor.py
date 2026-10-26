"""Offline tests for the pure X2 simulator/real-robot lifecycle policy."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from issue80_x2_startup_supervisor import (
    MAX_PLANAR_SPEED_M_S,
    LOCOMOTION_VELOCITY_TOPIC,
    Operation,
    SET_MC_ACTION_SERVICE,
    SET_MC_INPUT_SOURCE_SERVICE,
    SUPPORTED_AIMDK_INTERFACES,
    StartupObservation,
    StartupPath,
    SupervisorState,
    X2StartupSupervisor,
    posture_failure,
)


def observation(**overrides):
    payload = dict(
        telemetry_fresh=True,
        source_stamps_strictly_advancing=True,
        root_z_m=.68,
        roll_rad=0.,
        pitch_rad=0.,
        planar_speed_m_s=0.,
        joint_state_ready=True,
        source_joint_limits_ok=True,
        controller_gains_active=False,
        feet_supported=True,
        support_evidence="contact_sensors",
        common_action="PASSIVE_DEFAULT",
        common_action_status=100,
        get_action="PASSIVE_DEFAULT",
        get_action_status=100,
        fsm_state=6,
        body_status=0,
    )
    payload.update(overrides)
    return StartupObservation(**payload)


class StartupSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = X2StartupSupervisor()

    def test_simulation_requires_verified_initial_configuration_not_gui_reset(self):
        initial = observation()
        decision = self.supervisor.decide(StartupPath.SIMULATION, initial)
        self.assertEqual(decision.state, SupervisorState.INITIAL_CONFIGURATION_REQUIRED)
        self.assertEqual(decision.allowed_operations, (
            Operation.INITIALIZE_VERIFIED_SIMULATION_STANDING_CONFIGURATION,
        ))
        verified = observation(
            simulation_initial_configuration_verified=True,
            simulation_initial_support_verified=True,
            simulation_generation_verified=True,
            support_evidence="verified_simulation_configuration",
        )
        decision = self.supervisor.decide(StartupPath.SIMULATION, verified)
        self.assertEqual(decision.state, SupervisorState.JD_REQUEST_ALLOWED)
        self.assertEqual(decision.allowed_operations, (Operation.REQUEST_JD,))

    def test_simulation_support_is_a_separate_required_evidence_gate(self):
        missing = observation(
            simulation_initial_configuration_verified=True,
            simulation_initial_support_verified=False,
            simulation_generation_verified=True,
            support_evidence="unknown",
            feet_supported=None,
        )
        decision = self.supervisor.decide(StartupPath.SIMULATION, missing)
        self.assertEqual(decision.state, SupervisorState.BLOCKED)
        self.assertEqual(decision.safety_failure, "simulation_support_not_verified_before_stepping")

    def test_real_robot_never_issues_jd_from_unknown_or_unassisted_posture(self):
        unknown = observation(feet_supported=None, support_evidence="unknown")
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, unknown)
        self.assertEqual(decision.state, SupervisorState.PHYSICAL_RECOVERY_REQUIRED)
        self.assertEqual(decision.allowed_operations, (
            Operation.REQUEST_OPERATOR_OR_SUPPORTED_PHYSICAL_RECOVERY,
        ))
        recovered_unknown_support = observation(
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
            feet_supported=None,
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, recovered_unknown_support)
        self.assertEqual(decision.state, SupervisorState.BLOCKED)
        self.assertEqual(decision.safety_failure, "feet_support_not_verified")

    def test_real_robot_allows_jd_only_after_recovery_and_support_proof(self):
        ready = observation(
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
            support_evidence="operator_confirmed",
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, ready)
        self.assertEqual(decision.state, SupervisorState.JD_REQUEST_ALLOWED)
        self.assertEqual(decision.allowed_operations, (Operation.REQUEST_JD,))

    def test_jd_ready_then_sd_transition_then_stable_hold(self):
        jd = observation(
            common_action="JOINT_DEFAULT",
            get_action="JOINT_DEFAULT",
            fsm_state=1,
            body_status=1,
            controller_gains_active=True,
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, jd)
        self.assertEqual(decision.state, SupervisorState.SD_REQUEST_ALLOWED)
        sd_transition = observation(
            common_action="STAND_DEFAULT",
            common_action_status=200,
            get_action="JOINT_DEFAULT",
            get_action_status=200,
            fsm_state=1,
            body_status=1,
            controller_gains_active=True,
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, sd_transition)
        self.assertEqual(decision.state, SupervisorState.SD_TRANSITION)
        sd_ready = observation(
            common_action="STAND_DEFAULT",
            get_action="STAND_DEFAULT",
            fsm_state=4,
            body_status=1,
            controller_gains_active=True,
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
            standing_hold_s=2.4,
            standing_unique_samples=25,
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, sd_ready)
        self.assertEqual(decision.state, SupervisorState.STANDING_HOLD)
        decision = self.supervisor.decide(
            StartupPath.REAL_ROBOT,
            observation(
                common_action="STAND_DEFAULT",
                get_action="STAND_DEFAULT",
                fsm_state=4,
                body_status=1,
                controller_gains_active=True,
                supported_physical_standup_completed=True,
                operator_recovery_confirmed=True,
                standing_hold_s=2.5,
                standing_unique_samples=25,
            ),
        )
        self.assertEqual(decision.state, SupervisorState.STABLE_IDLE)

    def test_saved_raicom_speed_failure_preserves_limit(self):
        failed = observation(
            common_action="STAND_DEFAULT",
            get_action="JOINT_DEFAULT",
            common_action_status=100,
            get_action_status=100,
            fsm_state=1,
            body_status=1,
            controller_gains_active=True,
            planar_speed_m_s=.20902621054935036,
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
        )
        self.assertEqual(posture_failure(failed), "planar_speed_limit")
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, failed)
        self.assertEqual(decision.state, SupervisorState.BLOCKED)
        self.assertEqual(decision.safety_failure, "planar_speed_limit")
        self.assertEqual(MAX_PLANAR_SPEED_M_S, .20)

    def test_navigation_requires_physical_motion_stop_and_source_release(self):
        stable = observation(
            common_action="STAND_DEFAULT",
            get_action="STAND_DEFAULT",
            fsm_state=4,
            body_status=1,
            controller_gains_active=True,
            standing_hold_s=2.5,
            standing_unique_samples=25,
        )
        decision = self.supervisor.decide(StartupPath.NAVIGATION, stable)
        self.assertEqual(decision.state, SupervisorState.NAVIGATION_SOURCE_REQUIRED)
        source = observation(
            common_action="STAND_DEFAULT", get_action="STAND_DEFAULT",
            fsm_state=4, body_status=1, controller_gains_active=True,
            standing_hold_s=2.5, standing_unique_samples=25,
            navigation_source_registered=True, navigation_source_selected=True,
        )
        self.assertEqual(
            self.supervisor.decide(StartupPath.NAVIGATION, source).state,
            SupervisorState.NAVIGATION_LD_PENDING,
        )
        moving = observation(
            common_action="STAND_DEFAULT", get_action="STAND_DEFAULT",
            fsm_state=4, body_status=1, controller_gains_active=True,
            standing_hold_s=2.5, standing_unique_samples=25,
            navigation_source_registered=True, navigation_source_selected=True,
            locomotion_mode_confirmed=True, valid_nonzero_navigation_command_seen=True,
            physical_movement_confirmed=True, zero_command_published=True,
        )
        decision = self.supervisor.decide(StartupPath.NAVIGATION, moving)
        self.assertEqual(decision.state, SupervisorState.NAVIGATION_STOP_VERIFY)
        self.assertEqual(decision.allowed_operations, (Operation.VERIFY_PHYSICAL_STOP,))
        released = observation(
            common_action="LOCOMOTION_DEFAULT",
            get_action="LOCOMOTION_DEFAULT",
            common_action_status=100,
            get_action_status=100,
            fsm_state=5,
            body_status=1,
            controller_gains_active=True,
            navigation_source_registered=True,
            navigation_source_selected=True,
            locomotion_mode_confirmed=True,
            valid_nonzero_navigation_command_seen=True,
            physical_movement_confirmed=True,
            zero_command_published=True,
            physical_stop_confirmed=True,
            navigation_source_released=True,
        )
        decision = self.supervisor.decide(StartupPath.NAVIGATION, released)
        self.assertEqual(decision.allowed_operations, (Operation.REQUEST_SD,))
        idle = observation(
            common_action="STAND_DEFAULT",
            get_action="STAND_DEFAULT",
            fsm_state=4,
            body_status=1,
            controller_gains_active=True,
            standing_hold_s=2.5,
            standing_unique_samples=25,
            navigation_source_released=True,
        )
        self.assertEqual(
            self.supervisor.decide(StartupPath.NAVIGATION, idle).state,
            SupervisorState.STABLE_IDLE,
        )

    def test_unknown_or_stale_evidence_is_never_promoted(self):
        stale = observation(telemetry_fresh=False)
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, stale)
        self.assertEqual(decision.state, SupervisorState.PHYSICAL_RECOVERY_REQUIRED)
        ready_then_stale = observation(
            supported_physical_standup_completed=True,
            operator_recovery_confirmed=True,
            telemetry_fresh=False,
        )
        decision = self.supervisor.decide(StartupPath.REAL_ROBOT, ready_then_stale)
        self.assertEqual(decision.state, SupervisorState.BLOCKED)
        self.assertEqual(decision.safety_failure, "stale_or_missing_telemetry")

    def test_supervisor_has_no_backend_or_robot_actuation_import(self):
        source = Path(__file__).with_name("issue80_x2_startup_supervisor.py").read_text()
        tree = ast.parse(source)
        imports = [alias.name for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names]
        self.assertFalse(any(name.startswith(("mujoco", "rclpy", "aimdk_msgs")) for name in imports))
        actions = {item.value for item in Operation}
        self.assertFalse(actions & {"reset", "write_qpos", "teleport_root", "physical_standup"})

    def test_supported_interface_contract_uses_installed_aimdk_names(self):
        self.assertEqual(
            SUPPORTED_AIMDK_INTERFACES[Operation.REQUEST_JD]["name"],
            SET_MC_ACTION_SERVICE,
        )
        self.assertEqual(
            SUPPORTED_AIMDK_INTERFACES[Operation.REQUEST_SD]["request"]["command.action_desc"],
            "STAND_DEFAULT",
        )
        self.assertEqual(
            SUPPORTED_AIMDK_INTERFACES[Operation.REQUEST_LD]["request"]["command.action_desc"],
            "LOCOMOTION_DEFAULT",
        )
        self.assertEqual(
            SUPPORTED_AIMDK_INTERFACES[Operation.REGISTER_NAVIGATION_SOURCE]["name"],
            SET_MC_INPUT_SOURCE_SERVICE,
        )
        self.assertEqual(
            SUPPORTED_AIMDK_INTERFACES[Operation.PUBLISH_ZERO_COMMAND]["name"],
            LOCOMOTION_VELOCITY_TOPIC,
        )

    def test_cli_returns_a_reviewable_non_actuating_decision(self):
        source = Path(__file__).with_name("issue80_x2_startup_supervisor.py")
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "observation.json"
            payload.write_text(json.dumps({
                "telemetry_fresh": True,
                "source_stamps_strictly_advancing": True,
                "root_z_m": .68,
                "roll_rad": 0.,
                "pitch_rad": 0.,
                "planar_speed_m_s": 0.,
                "joint_state_ready": True,
                "source_joint_limits_ok": True,
                "feet_supported": True,
                "support_evidence": "verified_simulation_configuration",
                "common_action": "PASSIVE_DEFAULT",
                "common_action_status": 100,
                "get_action": "PASSIVE_DEFAULT",
                "get_action_status": 100,
                "fsm_state": 6,
                "body_status": 0,
                "simulation_initial_configuration_verified": True,
                "simulation_initial_support_verified": True,
                "simulation_generation_verified": True,
            }))
            output = subprocess.check_output([
                sys.executable, str(source), "--path", "simulation",
                "--observation-json", str(payload),
            ], text=True)
        decision = json.loads(output)
        self.assertEqual(decision["state"], "jd_request_allowed")
        self.assertEqual(decision["allowed_operations"], ["request_joint_default"])


if __name__ == "__main__":
    unittest.main()
