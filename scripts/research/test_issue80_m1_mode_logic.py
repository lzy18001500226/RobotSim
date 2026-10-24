"""Offline tests for source and action response semantics in the Issue 80 M1 runner."""

from __future__ import annotations

import unittest

from issue80_m1_mode_logic import (
    MODE_LD,
    MODE_SD,
    OFFICIAL_VELOCITY_SOURCE,
    assess_input_source_add,
    classify_ld_transition,
    classify_set_mc_action_response,
    is_valid_input_source_name,
)


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


if __name__ == "__main__":
    unittest.main()
