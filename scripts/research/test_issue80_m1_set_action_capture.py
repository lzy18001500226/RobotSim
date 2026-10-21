"""Offline schema checks for the one-shot SetMcAction response recorder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue80_m1_set_action_capture import assess_pre_action_state, set_action_record


class FakeMessage:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def get_fields_and_field_types(self):
        return {name: "fake" for name in self.__dict__}


class SetActionCaptureTests(unittest.TestCase):
    def test_runner_has_one_non_retried_service_mutation_call(self) -> None:
        source = Path(__file__).with_name("issue80_m1_set_action_capture.py").read_text()
        self.assertEqual(source.count("node.client.call_async(request)"), 1)
        self.assertIn("Exactly one mutation request; never retry.", source)
        guard = source.index('if not precondition["eligible"]:')
        call = source.index("future = node.client.call_async(request)")
        self.assertLess(guard, call)
        self.assertLess(source.index("return 2", guard), call)

    def response(self, status: int, code: int, message: str):
        body = FakeMessage(
            header=FakeMessage(code=code),
            status=FakeMessage(value=status),
            message=message,
        )
        return FakeMessage(response=body)


    def test_current_fallen_posture_blocks_sd_even_if_reset_was_previously_upright(self) -> None:
        now_ns = 10_000_000_000
        latest = {
            "odom": {
                "wall_monotonic_ns": now_ns - 1_000_000,
                "data": {"z": 0.358, "roll_deg": -170.31, "pitch_deg": -88.13},
            },
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {"wall_monotonic_ns": now_ns - 3_000_000, "data": {"fsm_state": 6, "body_status": 6}},
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        result = assess_pre_action_state(
            latest,
            now_monotonic_ns=now_ns,
            upright_odom_samples=0,
            service_ready=True,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["failure_reason"], "live_posture_outside_existing_gate")
        self.assertEqual(result["raw_common_state"], {"fsm_state": 6, "body_status": 6})

    def test_fresh_upright_pre_action_snapshot_is_eligible_without_enum_interpretation(self) -> None:
        now_ns = 20_000_000_000
        latest = {
            "odom": {
                "wall_monotonic_ns": now_ns - 1_000_000,
                "data": {"z": 0.68, "roll_deg": 0.1, "pitch_deg": 0.2},
            },
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {"wall_monotonic_ns": now_ns - 3_000_000, "data": {"fsm_state": 6, "body_status": 2}},
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        result = assess_pre_action_state(
            latest,
            now_monotonic_ns=now_ns,
            upright_odom_samples=3,
            service_ready=True,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["raw_common_state"], {"fsm_state": 6, "body_status": 2})

    def test_full_nested_response_and_codes_are_preserved(self) -> None:
        record = set_action_record(
            FakeMessage(source="node.set_mc_action", action="STAND_DEFAULT"),
            self.response(0, 0, "accepted"),
            success_status=0,
            submitted_utc="2026-10-21T09:00:00+00:00",
            submitted_monotonic_ns=100,
            returned_utc="2026-10-21T09:00:00.010+00:00",
            returned_monotonic_ns=10_000_100,
        )
        self.assertEqual(record["response_status_value"], 0)
        self.assertEqual(record["response_header_code"], 0)
        self.assertEqual(record["response_message"], "accepted")
        self.assertTrue(record["service_success"])
        self.assertTrue(record["action_result_success"])
        self.assertEqual(record["response"], {"header": {"code": 0}, "status": {"value": 0}, "message": "accepted"})

    def test_service_refusal_and_action_result_code_remain_distinct(self) -> None:
        record = set_action_record(
            FakeMessage(source="node.set_mc_action", action="STAND_DEFAULT"),
            self.response(7, 4, "invalid posture"),
            success_status=0,
            submitted_utc="t0",
            submitted_monotonic_ns=1,
            returned_utc="t1",
            returned_monotonic_ns=2,
        )
        self.assertFalse(record["service_success"])
        self.assertFalse(record["action_result_success"])
        self.assertEqual(record["response_status_value"], 7)
        self.assertEqual(record["response_header_code"], 4)
        self.assertEqual(record["response_message"], "invalid posture")


if __name__ == "__main__":
    unittest.main()
