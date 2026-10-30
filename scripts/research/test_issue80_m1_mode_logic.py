"""Offline tests for source and action response semantics in the Issue 80 M1 runner."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from issue80_m1_mode_logic import (
    LOCOMOTION_CONFIRMED,
    LOCOMOTION_REQUESTED,
    ACTIVATION_PROBE,
    ACTIVATION_PROBE_DURATION_S,
    ACTIVATION_PROBE_FORWARD_M_S,
    CPG_STEP_FORWARD_M_S,
    LATERAL_STEP_COMMAND_M_S,
    VENDOR_EXAMPLE_FORWARD_M_S,
    MOTION_ACTIVE,
    MODE_LD,
    MODE_LS,
    MODE_SD,
    OFFICIAL_VELOCITY_SOURCE,
    STAND_VERIFIED,
    ZERO_AND_STOP_REQUESTED,
    MCLocomotionAbortLifecycle,
    activation_probe_abort_is_expected,
    activation_probe_safe_stop_passes,
    assess_input_source_add,
    classify_locomotion_activation_readbacks,
    classify_ld_transition,
    classify_set_mc_action_response,
    is_valid_input_source_name,
    run_zero_only_locomotion_activation_observation,
    source_grounded_route_activation_speed_is_supported,
)


class FakeClock:
    def __init__(self, value: float = 20.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(0.0, duration)


def activation_snapshot(
    now: float,
    *,
    get_mode: str = MODE_SD,
    common_mode: str = MODE_SD,
    get_source: str = "",
    common_source: str = "",
    get_mode_age: float = 0.01,
    get_source_age: float = 0.01,
    common_age: float = 0.01,
    get_mode_status: int = 100,
    common_status: int = 100,
) -> dict[str, object]:
    return {
        "sample_monotonic_s": now,
        "readback_monotonic_s": {
            "get_mode": now - get_mode_age,
            "get_source": now - get_source_age,
        },
        "state_age_s": {"common": common_age},
        "get_mode_header_code": 0,
        "get_source_header_code": 0,
        "get_mode_status_value": get_mode_status,
        "common_mode_status_value": common_status,
        "get_mode_desc": get_mode,
        "common_mode_desc": common_mode,
        "get_source_name": get_source,
        "common_source_name": common_source,
        "fsm_state": 4 if get_mode == MODE_SD else 5,
        "body_status": 1,
    }


class SourceIdentityAndRegistrationTests(unittest.TestCase):
    def test_only_documented_velocity_source_is_live_identity(self) -> None:
        self.assertTrue(is_valid_input_source_name(OFFICIAL_VELOCITY_SOURCE))
        self.assertFalse(is_valid_input_source_name("node.set_mc_input_source"))
        self.assertFalse(is_valid_input_source_name("issue80_m1_fake_source"))
        self.assertFalse(is_valid_input_source_name(""))

    def test_documented_header_success_is_not_rejected_by_unknown_task_state(self) -> None:
        result = assess_input_source_add(0, task_id=1, state_value=0)
        self.assertEqual(result["assessment"], "ADD_ACCEPTED_TASK_NOT_TERMINAL")
        self.assertTrue(result["registration_operation_succeeded"])
        self.assertFalse(result["task_state_is_success"])
        self.assertFalse(result["enable_fallback_allowed"])
        self.assertEqual(result["state_name"], "UNKNOWN")

    def test_nonzero_add_header_requests_enable_recovery(self) -> None:
        result = assess_input_source_add(3, task_id=4, state_value=5)
        self.assertEqual(result["assessment"], "ADD_REJECTED")
        self.assertFalse(result["registration_operation_succeeded"])
        self.assertTrue(result["enable_fallback_allowed"])

    def test_add_task_terminal_failure_allows_one_enable_recovery(self) -> None:
        result = assess_input_source_add(0, task_id=7, state_value=2)
        self.assertEqual(result["assessment"], "ADD_TASK_TERMINAL_FAILURE")
        self.assertTrue(result["registration_operation_succeeded"])
        self.assertTrue(result["enable_fallback_allowed"])

    def test_add_not_ready_state_does_not_trigger_enable_replay(self) -> None:
        result = assess_input_source_add(0, task_id=8, state_value=100)
        self.assertEqual(result["assessment"], "ADD_TASK_NOT_READY")
        self.assertTrue(result["registration_operation_succeeded"])
        self.assertFalse(result["enable_fallback_allowed"])


class ActionResponseAndTransitionTests(unittest.TestCase):
    def observation(self, mode: str, code: int = 0) -> dict[str, object]:
        return {"action_desc": mode, "response_header_code": code}

    def test_set_action_diagnostics_use_code_status_and_message(self) -> None:
        success = classify_set_mc_action_response(0, 1, "success")
        self.assertTrue(success["accepted_by_service"])
        self.assertFalse(success["applied_mode_confirmed"])
        self.assertEqual(success["message"], "success")
        rejected = classify_set_mc_action_response(6, 1, "security protection")
        self.assertFalse(rejected["accepted_by_service"])
        self.assertEqual(rejected["response_header_code"], 6)

    def test_successful_ld_request_may_wait_for_first_locomotion_input(self) -> None:
        result = classify_ld_transition(
            {"response_header_code": 0, "response_status_value": 1},
            [self.observation(MODE_SD)],
            MODE_SD,
        )
        self.assertEqual(result, "ACCEPTED_WAITING_FOR_VALID_LOCOMOTION_INPUT")

    def test_detailed_header_rejection_wins_over_success_status(self) -> None:
        result = classify_ld_transition(
            {"response_header_code": 6, "response_status_value": 1},
            [self.observation(MODE_LD)],
            MODE_LD,
        )
        self.assertEqual(result, "REJECTED_HEADER_CODE_6")

    def test_ld_requires_actual_get_and_common_readbacks(self) -> None:
        result = classify_ld_transition(
            {"response_header_code": 0, "response_status_value": 1},
            [self.observation(MODE_SD), self.observation(MODE_LD)],
            MODE_LD,
        )
        self.assertEqual(result, "LD_CONFIRMED")

    def test_unrelated_mode_is_not_accepted_as_transition(self) -> None:
        result = classify_ld_transition(
            {"response_header_code": 0, "response_status_value": 1},
            [self.observation("PASSIVE_DEFAULT")],
            "PASSIVE_DEFAULT",
        )
        self.assertEqual(result, "UNRELATED_GET_MODE")

    def test_zero_only_pending_add_and_sd_readback_remain_waiting(self) -> None:
        add = assess_input_source_add(0, task_id=9, state_value=0)
        self.assertEqual(add["assessment"], "ADD_ACCEPTED_TASK_NOT_TERMINAL")
        self.assertFalse(add["task_state_is_success"])
        # An empty arbitration winner after zero-only output is not a source list.
        selected_source = ""
        self.assertEqual(selected_source, "")

        action = classify_set_mc_action_response(0, 1, "success")
        transition = classify_ld_transition(
            {
                "response_header_code": action["response_header_code"],
                "response_status_value": action["response_status_value"],
            },
            [self.observation(MODE_SD)],
            MODE_SD,
        )
        self.assertEqual(transition, "ACCEPTED_WAITING_FOR_VALID_LOCOMOTION_INPUT")
        self.assertFalse(action["applied_mode_confirmed"])

    def test_terminal_add_then_ld_with_both_readbacks_is_confirmed(self) -> None:
        add = assess_input_source_add(0, task_id=10, state_value=1)
        self.assertEqual(add["assessment"], "ADD_TASK_SUCCEEDED")
        action = classify_set_mc_action_response(0, 1, "success")
        self.assertTrue(add["task_state_is_success"])
        self.assertTrue(action["accepted_by_service"])
        self.assertEqual(
            classify_ld_transition(
                {
                    "response_header_code": action["response_header_code"],
                    "response_status_value": action["response_status_value"],
                },
                [self.observation(MODE_LD)],
                MODE_LD,
            ),
            "LD_CONFIRMED",
        )


class ActivationAbortLifecycleTests(unittest.TestCase):
    def test_vendor_example_speed_is_opt_in_and_keeps_the_activation_timeout(self) -> None:
        self.assertFalse(source_grounded_route_activation_speed_is_supported(
            VENDOR_EXAMPLE_FORWARD_M_S, allow_raicom_low_cruise=False,
        ))
        self.assertTrue(source_grounded_route_activation_speed_is_supported(
            VENDOR_EXAMPLE_FORWARD_M_S,
            allow_raicom_low_cruise=False,
            allow_vendor_example=True,
        ))
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=0.0)
        lifecycle.request_locomotion(accepted=True, now_s=0.1)
        lifecycle.begin_activation_probe(
            source_contract_supported=True,
            forward_command_m_s=VENDOR_EXAMPLE_FORWARD_M_S,
            duration_s=ACTIVATION_PROBE_DURATION_S,
            now_s=0.2,
            allow_vendor_example=True,
        )
        self.assertEqual(lifecycle.state, ACTIVATION_PROBE)

    def test_probe_pass_requires_exact_intentional_abort_marker_and_fresh_mode_confirmation(self) -> None:
        expected = {
            "activation_probe_only": True,
            "fault": "activation_probe_complete_abort_to_stand",
            "activation_probe": {
                "mode_source_confirmed": True,
                "terminal_reason": "fresh synchronized LOCO/source handshake observed; aborting probe to STAND",
            },
        }
        self.assertTrue(activation_probe_abort_is_expected(expected))

        safety_fault = {
            **expected,
            "fault": "measured_root_planar_speed_limit:0.250000>0.200000",
        }
        self.assertFalse(activation_probe_abort_is_expected(safety_fault))

        wrong_terminal_reason = {
            **expected,
            "activation_probe": {
                "mode_source_confirmed": True,
                "terminal_reason": "fall_detected_during_activation",
            },
        }
        self.assertFalse(activation_probe_abort_is_expected(wrong_terminal_reason))

        unconfirmed_mode = {
            **expected,
            "activation_probe": {
                "mode_source_confirmed": False,
                "terminal_reason": "fresh synchronized LOCO/source handshake observed; aborting probe to STAND",
            },
        }
        self.assertFalse(activation_probe_abort_is_expected(unconfirmed_mode))

    def test_probe_stop_pass_requires_intentional_abort_stand_and_physical_pass(self) -> None:
        probe_result = {
            "activation_probe_only": True,
            "fault": "activation_probe_complete_abort_to_stand",
            "activation_probe": {
                "mode_source_confirmed": True,
                "terminal_reason": "fresh synchronized LOCO/source handshake observed; aborting probe to STAND",
            },
            "after_zero_stop_action": {"success": True},
            "post_stop_fault": None,
            "nonzero_publish_calls_returned": 20,
            "zero_stop": {"zero_publish_calls_returned": 50},
        }
        physical_gate = {"status": "PASS"}
        stand_confirmation = {"stand_confirmed": True}
        lifecycle = {"state": STAND_VERIFIED}
        self.assertTrue(activation_probe_safe_stop_passes(
            probe_result, physical_gate, stand_confirmation, lifecycle,
        ))

        safety_abort = {**probe_result, "fault": "measured_root_planar_speed_limit"}
        self.assertFalse(activation_probe_safe_stop_passes(
            safety_abort, physical_gate, stand_confirmation, lifecycle,
        ))
        no_nonzero_probe = {**probe_result, "nonzero_publish_calls_returned": 0}
        self.assertFalse(activation_probe_safe_stop_passes(
            no_nonzero_probe, physical_gate, stand_confirmation, lifecycle,
        ))
        self.assertFalse(activation_probe_safe_stop_passes(
            probe_result, {"status": "FAIL"}, stand_confirmation, lifecycle,
        ))
        self.assertFalse(activation_probe_safe_stop_passes(
            probe_result, physical_gate, {"stand_confirmed": False}, lifecycle,
        ))

    def test_activation_probe_is_separate_and_fixed_to_the_supported_startup_envelope(self) -> None:
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=1.0)
        lifecycle.request_locomotion(accepted=True, now_s=1.1)
        lifecycle.begin_activation_probe(
            source_contract_supported=True,
            forward_command_m_s=ACTIVATION_PROBE_FORWARD_M_S,
            duration_s=ACTIVATION_PROBE_DURATION_S,
            now_s=1.2,
        )
        self.assertEqual(lifecycle.state, ACTIVATION_PROBE)
        self.assertFalse(lifecycle.nonzero_motion_allowed)
        self.assertTrue(lifecycle.activation_probe_command_allowed)
        invalid_profile = MCLocomotionAbortLifecycle()
        invalid_profile.verify_initial_stand(confirmed=True, now_s=0.0)
        invalid_profile.request_locomotion(accepted=True, now_s=0.1)
        with self.assertRaisesRegex(ValueError, "fixed at 0.12 m/s for at most 1.80 s"):
            invalid_profile.begin_activation_probe(
                source_contract_supported=True,
                forward_command_m_s=0.2,
                duration_s=0.5,
                now_s=0.0,
            )
        unsupported = MCLocomotionAbortLifecycle()
        unsupported.verify_initial_stand(confirmed=True, now_s=0.0)
        unsupported.request_locomotion(accepted=True, now_s=0.1)
        with self.assertRaisesRegex(ValueError, "source-supported startup contract"):
            unsupported.begin_activation_probe(
                source_contract_supported=False,
                forward_command_m_s=ACTIVATION_PROBE_FORWARD_M_S,
                duration_s=ACTIVATION_PROBE_DURATION_S,
                now_s=0.2,
            )

    def test_successful_set_action_does_not_advance_without_live_mode_and_source(self) -> None:
        lifecycle = MCLocomotionAbortLifecycle()
        self.assertTrue(lifecycle.verify_initial_stand(confirmed=True, now_s=20.0))
        self.assertEqual(lifecycle.state, STAND_VERIFIED)
        self.assertTrue(lifecycle.request_locomotion(accepted=True, now_s=20.1))
        self.assertEqual(lifecycle.state, LOCOMOTION_REQUESTED)
        pending = classify_locomotion_activation_readbacks(
            activation_snapshot(20.2), expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        self.assertEqual(pending["classification"], "PENDING_STAND_REMAINS_ACTIVE")
        lifecycle.observe_locomotion_readback(pending, now_s=20.2)
        self.assertEqual(lifecycle.state, LOCOMOTION_REQUESTED)
        self.assertFalse(lifecycle.nonzero_motion_allowed)

    def test_common_state_get_action_disagreement_never_confirms_activation(self) -> None:
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=0.0)
        lifecycle.request_locomotion(accepted=True, now_s=0.01)
        observation = classify_locomotion_activation_readbacks(
            activation_snapshot(
                0.4,
                get_mode=MODE_SD,
                common_mode=MODE_LD,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source=OFFICIAL_VELOCITY_SOURCE,
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        self.assertEqual(observation["classification"], "PENDING_MODE_DISAGREEMENT")
        lifecycle.observe_locomotion_readback(observation, now_s=0.4)
        self.assertEqual(lifecycle.state, LOCOMOTION_REQUESTED)
        with self.assertRaisesRegex(ValueError, "forbidden before LOCOMOTION_CONFIRMED"):
            lifecycle.begin_motion(now_s=0.41)

    def test_installed_activation_requires_both_fresh_modes_and_both_source_readbacks(self) -> None:
        stale = classify_locomotion_activation_readbacks(
            activation_snapshot(
                10.0,
                get_mode=MODE_LD,
                common_mode=MODE_LD,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source=OFFICIAL_VELOCITY_SOURCE,
                common_age=0.51,
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        self.assertEqual(stale["classification"], "PENDING_STALE_READBACK")
        get_only = classify_locomotion_activation_readbacks(
            activation_snapshot(
                10.0,
                get_mode=MODE_LD,
                common_mode=MODE_LD,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source="",
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        self.assertEqual(get_only["classification"], "PENDING_SOURCE_DISAGREEMENT")
        confirmed = classify_locomotion_activation_readbacks(
            activation_snapshot(
                10.0,
                get_mode=MODE_LD,
                common_mode=MODE_LD,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source=OFFICIAL_VELOCITY_SOURCE,
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        self.assertEqual(confirmed["classification"], "LOCOMOTION_CONFIRMED")
        self.assertTrue(confirmed["motion_allowed"])

    def test_step_action_requires_step_readbacks_and_installed_start_speed(self) -> None:
        default_mode = classify_locomotion_activation_readbacks(
            activation_snapshot(
                10.0,
                get_mode=MODE_LD,
                common_mode=MODE_LD,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source=OFFICIAL_VELOCITY_SOURCE,
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
            expected_locomotion_mode=MODE_LS,
        )
        self.assertEqual(default_mode["classification"], "ERROR_UNRELATED_MODE")
        step_mode = classify_locomotion_activation_readbacks(
            activation_snapshot(
                10.0,
                get_mode=MODE_LS,
                common_mode=MODE_LS,
                get_source=OFFICIAL_VELOCITY_SOURCE,
                common_source=OFFICIAL_VELOCITY_SOURCE,
            ),
            expected_source=OFFICIAL_VELOCITY_SOURCE,
            expected_locomotion_mode=MODE_LS,
        )
        self.assertEqual(step_mode["classification"], "LOCOMOTION_CONFIRMED")
        self.assertTrue(source_grounded_route_activation_speed_is_supported(
            CPG_STEP_FORWARD_M_S,
            allow_raicom_low_cruise=False,
            allow_cpg_step=True,
        ))
        self.assertFalse(source_grounded_route_activation_speed_is_supported(
            CPG_STEP_FORWARD_M_S,
            allow_raicom_low_cruise=False,
        ))

    def test_lateral_step_requires_the_documented_midrange_start_command(self) -> None:
        self.assertTrue(source_grounded_route_activation_speed_is_supported(
            0.0,
            lateral_command_m_s=LATERAL_STEP_COMMAND_M_S,
            allow_raicom_low_cruise=False,
            allow_lateral_step=True,
        ))
        self.assertFalse(source_grounded_route_activation_speed_is_supported(
            0.0,
            lateral_command_m_s=0.2,
            allow_raicom_low_cruise=False,
            allow_lateral_step=True,
        ))

    def test_zero_only_activation_observer_never_sends_nonzero_and_timeout_requires_abort(self) -> None:
        clock = FakeClock()
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=clock.now())
        lifecycle.request_locomotion(accepted=True, now_s=clock.now())
        zero_commands: list[float] = []

        def sample():
            return activation_snapshot(clock.now())

        result = run_zero_only_locomotion_activation_observation(
            sample=sample,
            publish_zero=lambda: zero_commands.append(0.0),
            lifecycle=lifecycle,
            safety_fault=lambda row: None,
            expected_source=OFFICIAL_VELOCITY_SOURCE,
            timeout_s=0.10,
            clock=clock.now,
            sleep=clock.sleep,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertGreaterEqual(result["zero_publish_calls_returned"], 1)
        self.assertTrue(all(value == 0.0 for value in zero_commands))
        self.assertEqual(lifecycle.state, ZERO_AND_STOP_REQUESTED)
        self.assertFalse(lifecycle.nonzero_motion_allowed)

    def test_zero_only_activation_can_confirm_then_enter_motion_only_after_confirmation(self) -> None:
        clock = FakeClock()
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=clock.now())
        lifecycle.request_locomotion(accepted=True, now_s=clock.now())
        zero_commands: list[float] = []
        sample_count = 0

        def sample():
            nonlocal sample_count
            sample_count += 1
            if sample_count >= 4:
                return activation_snapshot(
                    clock.now(),
                    get_mode=MODE_LD,
                    common_mode=MODE_LD,
                    get_source=OFFICIAL_VELOCITY_SOURCE,
                    common_source=OFFICIAL_VELOCITY_SOURCE,
                )
            return activation_snapshot(clock.now())

        result = run_zero_only_locomotion_activation_observation(
            sample=sample,
            publish_zero=lambda: zero_commands.append(0.0),
            lifecycle=lifecycle,
            safety_fault=lambda row: None,
            expected_source=OFFICIAL_VELOCITY_SOURCE,
            timeout_s=0.5,
            clock=clock.now,
            sleep=clock.sleep,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(lifecycle.state, LOCOMOTION_CONFIRMED)
        self.assertFalse(lifecycle.nonzero_motion_allowed)
        lifecycle.begin_motion(now_s=clock.now())
        self.assertEqual(lifecycle.state, MOTION_ACTIVE)
        self.assertTrue(lifecycle.nonzero_motion_allowed)
        self.assertTrue(all(value == 0.0 for value in zero_commands))

    def test_saved_curved_abort_trace_never_authorizes_motion_and_records_late_locomotion(self) -> None:
        fixture_path = Path(__file__).with_name("fixtures") / "issue80_curved_abort_transition_replay.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        command_timeline = fixture["command_timeline"]
        self.assertEqual(command_timeline["nonzero_publish_calls_returned"], 14)
        self.assertLess(
            command_timeline["last_nonzero_wall_utc"],
            command_timeline["first_zero_wall_utc"],
        )
        self.assertFalse(command_timeline["post_stop_stand_callback_invoked"])
        self.assertGreater(
            command_timeline["first_measured_speed_guard_failure"]["planar_speed_m_s"],
            command_timeline["first_measured_speed_guard_failure"]["limit_m_s"],
        )
        source_contract = fixture["installed_runtime_nonzero_activation_evidence"]
        self.assertEqual(command_timeline["first_nonzero_forward_velocity_m_s"], 0.12)
        self.assertEqual(command_timeline["last_nonzero_forward_velocity_m_s"], 0.12)
        self.assertEqual(source_contract["selected_source_get_service"], OFFICIAL_VELOCITY_SOURCE)
        self.assertEqual(source_contract["selected_source_common_state"], OFFICIAL_VELOCITY_SOURCE)
        self.assertLess(source_contract["common_state_locomotion_offset_s"], 0.5)
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=0.0)
        lifecycle.request_locomotion(accepted=True, now_s=0.0)
        observations = []
        for raw in fixture["observations"][:3]:
            now = float(raw["offset_s"])
            row = activation_snapshot(
                now,
                get_mode=raw["get_mode_desc"],
                common_mode=raw["common_mode_desc"],
                get_source=raw["get_source_name"],
                common_source=raw["common_source_name"],
                get_mode_age=raw["get_mode_age_s"],
                get_source_age=raw["get_source_age_s"],
                common_age=raw["common_age_s"],
                get_mode_status=raw.get("get_mode_status_value", 100),
                common_status=raw.get("common_mode_status_value", 100),
            )
            observation = classify_locomotion_activation_readbacks(
                row, expected_source=OFFICIAL_VELOCITY_SOURCE,
            )
            observations.append(observation["classification"])
            lifecycle.observe_locomotion_readback(observation, now_s=now)
        self.assertEqual(observations[0], "PENDING_STAND_REMAINS_ACTIVE")
        self.assertEqual(observations[1], "PENDING_MODE_DISAGREEMENT")
        self.assertEqual(observations[2], "PENDING_READBACK_SKEW")
        self.assertEqual(lifecycle.state, LOCOMOTION_REQUESTED)
        lifecycle.expire_activation(now_s=0.57, reason="0.5 s route handshake expired")
        late = fixture["observations"][3]
        late_row = activation_snapshot(
            float(late["offset_s"]),
            get_mode=late["get_mode_desc"],
            common_mode=late["common_mode_desc"],
            get_source=late["get_source_name"],
            common_source=late["common_source_name"],
            get_mode_age=late["get_mode_age_s"],
            get_source_age=late["get_source_age_s"],
            common_age=late["common_age_s"],
        )
        observation = classify_locomotion_activation_readbacks(
            late_row, expected_source=OFFICIAL_VELOCITY_SOURCE,
        )
        lifecycle.observe_locomotion_readback(observation, now_s=float(late["offset_s"]))
        self.assertEqual(observation["classification"], "LOCOMOTION_CONFIRMED")
        self.assertEqual(lifecycle.state, ZERO_AND_STOP_REQUESTED)
        self.assertTrue(lifecycle.late_locomotion_after_abort)
        self.assertFalse(lifecycle.nonzero_motion_allowed)
        with self.assertRaisesRegex(ValueError, "activation abort is unresolved"):
            lifecycle.begin_motion(now_s=float(late["offset_s"]))

    def test_stand_recovery_requires_a_single_request_and_composite_stop_pass(self) -> None:
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=0.0)
        lifecycle.request_locomotion(accepted=True, now_s=0.1)
        lifecycle.expire_activation(now_s=2.1, reason="zero-only activation timeout")
        self.assertTrue(lifecycle.begin_stand_request_once(now_s=2.1))
        self.assertFalse(lifecycle.begin_stand_request_once(now_s=2.2))
        self.assertFalse(lifecycle.confirm_stand_after_stop(
            stand_readback_pass=True, physical_stop_gate_pass=False, now_s=3.0,
        ))
        self.assertEqual(lifecycle.state, ZERO_AND_STOP_REQUESTED)
        self.assertTrue(lifecycle.confirm_stand_after_stop(
            stand_readback_pass=True, physical_stop_gate_pass=True, now_s=4.0,
        ))
        self.assertEqual(lifecycle.state, STAND_VERIFIED)
        with self.assertRaisesRegex(ValueError, "activation abort is unresolved"):
            lifecycle.request_locomotion(accepted=True, now_s=4.1)

    def test_speed_guard_error_aborts_to_zero_stop_state(self) -> None:
        clock = FakeClock()
        lifecycle = MCLocomotionAbortLifecycle()
        lifecycle.verify_initial_stand(confirmed=True, now_s=clock.now())
        lifecycle.request_locomotion(accepted=True, now_s=clock.now())
        calls = 0

        def safety(row):
            nonlocal calls
            calls += 1
            return "measured_root_planar_speed_limit" if calls >= 2 else None

        result = run_zero_only_locomotion_activation_observation(
            sample=lambda: activation_snapshot(clock.now()),
            publish_zero=lambda: None,
            lifecycle=lifecycle,
            safety_fault=safety,
            timeout_s=0.5,
            clock=clock.now,
            sleep=clock.sleep,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(lifecycle.state, ZERO_AND_STOP_REQUESTED)
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)


if __name__ == "__main__":
    unittest.main()
