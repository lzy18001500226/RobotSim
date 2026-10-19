"""Offline tests for reset delivery and stable-standing classification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from issue80_m1_reset_analyze import classify


EVENT_NS = 1_000_000_000
STEP_NS = 50_000_000


def state_row(key: str, at_ns: int, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "state",
        "topic_key": key,
        "wall_utc": "synthetic",
        "wall_monotonic_ns": at_ns,
        "data": data,
    }


def make_trace(*, stop_ns: int, fall_after_ns: int | None = None, expose_clock_reset: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for at_ns in range(500_000_000, stop_ns + 1, STEP_NS):
        common_at = at_ns - (at_ns % 100_000_000)
        body_status = 1 if common_at >= EVENT_NS + 250_000_000 else 2
        rows.append(
            state_row(
                "common",
                common_at,
                {
                    "action_info": {"action_desc": "STAND_DEFAULT"},
                    "body_status": {"value": body_status},
                },
            )
        )

        upright = at_ns >= EVENT_NS + 50_000_000
        if fall_after_ns is not None and at_ns >= fall_after_ns:
            upright = False
        pose = (
            {"z": 0.65, "roll_deg": 0.0, "pitch_deg": 0.0}
            if upright
            else {"z": 0.08, "roll_deg": 95.0, "pitch_deg": -80.0}
        )
        rows.append(state_row("leg_state", at_ns, {"joints": [{"name": "hip", "position": 0.0 if upright else 0.7}]}))
        rows.append(state_row("odom", at_ns, pose))

    if expose_clock_reset:
        rows.append({"event": "clock", "wall_utc": "synthetic", "wall_monotonic_ns": EVENT_NS - 10_000_000, "clock": {"sec": 4, "nanosec": 0}})
        rows.append({"event": "clock", "wall_utc": "synthetic", "wall_monotonic_ns": EVENT_NS + 10_000_000, "clock": {"sec": 0, "nanosec": 0}})
    return rows


class ResetAnalysisTests(unittest.TestCase):
    def run_analysis(self, rows: list[dict[str, Any]], *, dispatch_status: str = "KEY_DISPATCHED") -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace = root / "trace.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            dispatch = root / "dispatch.json"
            dispatch.write_text(
                json.dumps(
                    {
                        "window": {"window_id": 7},
                        "event": {
                            "status": dispatch_status,
                            "dispatch_monotonic_ns": EVENT_NS,
                            "dispatch_wall_utc": "synthetic",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return classify(trace, dispatch)

    def test_reset_pass_waits_for_pose_and_mc_stand_state_then_holds_2_5s(self) -> None:
        result = self.run_analysis(make_trace(stop_ns=4_000_000_000))
        self.assertEqual(result["classification"], "AUTO_RESET_PASS")
        self.assertTrue(result["stable_window_complete"])
        self.assertTrue(result["stable_window_pass"])
        self.assertEqual(result["stable_window_s"], 2.5)
        self.assertGreater(result["first_full_stand_ready_wall_monotonic_ns"], result["first_upright_wall_monotonic_ns"])

    def test_upright_reset_followed_by_fall_is_not_pass(self) -> None:
        result = self.run_analysis(make_trace(stop_ns=4_000_000_000, fall_after_ns=2_000_000_000))
        self.assertEqual(result["classification"], "RESET_APPLIED_THEN_FELL")
        self.assertFalse(result["stable_window_pass"])

    def test_key_without_state_discontinuity_is_not_delivered(self) -> None:
        result = self.run_analysis(make_trace(stop_ns=4_000_000_000, fall_after_ns=EVENT_NS))
        self.assertEqual(result["classification"], "RESET_NOT_DELIVERED")

    def test_clock_rollback_with_invalid_default_pose_is_distinguished(self) -> None:
        result = self.run_analysis(
            make_trace(stop_ns=4_000_000_000, fall_after_ns=EVENT_NS, expose_clock_reset=True)
        )
        self.assertEqual(result["classification"], "RESET_APPLIED_BUT_INVALID_INIT")

    def test_unverified_dispatch_is_rejected_before_state_classification(self) -> None:
        result = self.run_analysis(make_trace(stop_ns=4_000_000_000), dispatch_status="RESET_NOT_DELIVERED")
        self.assertEqual(result["classification"], "RESET_NOT_DELIVERED")

    def test_short_post_reset_trace_is_insufficient_not_pass(self) -> None:
        result = self.run_analysis(make_trace(stop_ns=2_000_000_000))
        self.assertEqual(result["classification"], "RESET_APPLIED_INSUFFICIENT_STABILITY_WINDOW")
        self.assertFalse(result["stable_window_complete"])


if __name__ == "__main__":
    unittest.main()
