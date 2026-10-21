"""Offline schema checks for the one-shot SetMcAction response recorder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue80_m1_set_action_capture import set_action_record


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

    def response(self, status: int, code: int, message: str):
        body = FakeMessage(
            header=FakeMessage(code=code),
            status=FakeMessage(value=status),
            message=message,
        )
        return FakeMessage(response=body)

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
