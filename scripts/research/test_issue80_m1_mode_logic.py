"""Offline tests for source and action response semantics in the Issue 80 M1 runner."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from issue80_m1_mode_logic import (
    LOCOMOTION_CONFIRMED,
    LOCOMOTION_REQUESTED,
    MOTION_ACTIVE,
    MODE_LD,
    MODE_SD,
    OFFICIAL_VELOCITY_SOURCE,
    STAND_VERIFIED,
    ZERO_AND_STOP_REQUESTED,
    MCLocomotionAbortLifecycle,
    assess_input_source_add,
    classify_locomotion_activation_readbacks,
    classify_ld_transition,
    classify_set_mc_action_response,
    is_valid_input_source_name,
    run_zero_only_locomotion_activation_observation,
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
