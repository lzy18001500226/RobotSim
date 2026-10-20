"""Offline tests for the Issue 80 mode-transition acceptance gate."""

from __future__ import annotations

import unittest

from issue80_m1_mode_logic import classify_ld_transition, is_valid_input_source_name


class ClassifyLdTransitionTests(unittest.TestCase):
    @staticmethod
    def observation(mode: str, code: int = 0) -> dict[str, object]:
        return {"action_desc": mode, "response_header_code": code}

    def test_ack_without_live_ld_readback_is_not_pass(self) -> None:
        result = classify_ld_transition(
            {"response_status_value": 1},
            [self.observation("STAND_DEFAULT")],
            "STAND_DEFAULT",
        )
        self.assertEqual(result, "ACCEPTED_BUT_NOT_TRANSITIONED")

    def test_request_rejection_stops_before_readback_claim(self) -> None:
        result = classify_ld_transition(
            {"response_status_value": 2},
            [self.observation("LOCOMOTION_DEFAULT")],
            "LOCOMOTION_DEFAULT",
        )
        self.assertEqual(result, "REJECTED")

    def test_both_live_readbacks_must_confirm_ld(self) -> None:
        result = classify_ld_transition(
            {"response_status_value": 1},
            [self.observation("LOCOMOTION_DEFAULT")],
            "STAND_DEFAULT",
        )
        self.assertEqual(result, "READBACK_INTERFACE_INCONSISTENCY")

    def test_confirmed_ld_requires_service_and_common_state(self) -> None:
        result = classify_ld_transition(
            {"response_status_value": 1},
            [self.observation("STAND_DEFAULT"), self.observation("LOCOMOTION_DEFAULT")],
            "LOCOMOTION_DEFAULT",
        )
        self.assertEqual(result, "LD_CONFIRMED")

    def test_readback_error_blocks_transition_claim(self) -> None:
        result = classify_ld_transition(
            {"response_status_value": 1},
            [self.observation("LOCOMOTION_DEFAULT")],
            "LOCOMOTION_DEFAULT",
            readback_error=True,
        )
        self.assertEqual(result, "READBACK_INTERFACE_ERROR")


class InputSourceIdentityTests(unittest.TestCase):
    def test_accepts_exact_pinned_sdk_input_source_identity(self) -> None:
        self.assertTrue(is_valid_input_source_name("node.set_mc_input_source"))

    def test_accepts_unique_research_source(self) -> None:
        self.assertTrue(is_valid_input_source_name("issue80_m1_5c3914c1"))

    def test_rejects_unrelated_or_nonunique_source_names(self) -> None:
        for value in ("node.other", "rc", "issue80_m1_short", ""):
            with self.subTest(value=value):
                self.assertFalse(is_valid_input_source_name(value))


if __name__ == "__main__":
    unittest.main()
