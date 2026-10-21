"""Offline schema checks for the one-shot SetMcAction response recorder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue80_m1_set_action_capture import (
    assess_joint_to_stand_state,
    assess_pre_action_state,
    command_is_effective,
    installed_body_stand_value,
    set_action_record,
    summarize_takeover_timing,
)


class FakeMessage:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def get_fields_and_field_types(self):
        return {name: "fake" for name in self.__dict__}


class SetActionCaptureTests(unittest.TestCase):
    def test_runner_has_one_non_retried_service_mutation_call(self) -> None:
        source = Path(__file__).with_name("issue80_m1_set_action_capture.py").read_text()
        self.assertEqual(source.count("self.set_client.call_async(request)"), 1)
        self.assertIn("mutating SetMcAction retry blocked", source)
        joint = source.index('self.set_action_once("JOINT_DEFAULT"')
        joint_gate = source.index('if jd.get("transport_state") != "RESPONSE_RECEIVED"', joint)
        body_gate = source.index("stand_gate = self.wait_for_joint_stand", joint_gate)
        stand = source.index('self.set_action_once("STAND_DEFAULT"', body_gate)
        self.assertLess(joint, joint_gate)
        self.assertLess(joint_gate, body_gate)
        self.assertLess(body_gate, stand)

    def test_runner_prearms_single_ros_process_before_reset_and_does_not_spawn_client_afterward(self) -> None:
        runner = Path(__file__).with_name("issue80_m1_auto_reset_run.py").read_text()
        start = runner.index("def start_observer")
        dispatch = runner.index("def dispatch_reset")
        request = runner.index("def request_sd_and_verify")
        self.assertLess(start, dispatch)
        request_body = runner[request:runner.index("def _write_sd_record", request)]
        self.assertNotIn("self.docker_exec(", request_body)
        self.assertIn('"set_action_capture_result.json"', request_body)

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

    def test_pre_reset_samples_cannot_satisfy_the_post_reset_action_gate(self) -> None:
        reset_ns = 20_000_000_000
        latest = {
            key: {
                "wall_monotonic_ns": reset_ns - 1,
                "data": {"z": 0.68, "roll_deg": 0.0, "pitch_deg": 0.0},
            }
            for key in ("odom", "imu", "common", "leg_state", "leg_command")
        }
        result = assess_pre_action_state(
            latest,
            now_monotonic_ns=reset_ns + 1_000_000,
            upright_odom_samples=3,
            service_ready=True,
            reset_dispatch_monotonic_ns=reset_ns,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["failure_reason"], "required_post_reset_state_missing_or_stale")

    def test_sd_requires_confirmed_joint_default_and_fresh_mc_stand_classification(self) -> None:
        reset_ns = 30_000_000_000
        now_ns = reset_ns + 50_000_000
        latest = {
            "odom": {"wall_monotonic_ns": now_ns - 1_000_000, "data": {"z": 0.68, "roll_deg": 0.1, "pitch_deg": 0.2}},
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {
                "wall_monotonic_ns": now_ns - 3_000_000,
                "data": {"action_info": {"action_desc": "JOINT_DEFAULT", "status": {"value": 100}}, "body_status": {"value": 1}},
            },
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        ready = assess_joint_to_stand_state(
            latest,
            now_monotonic_ns=now_ns,
            reset_dispatch_monotonic_ns=reset_ns,
            upright_odom_samples=3,
            stand_common_samples=3,
            set_service_ready=True,
            get_service_ready=True,
            get_mode_desc="JOINT_DEFAULT",
            get_mode_status=100,
            body_stand_value=1,
        )
        self.assertTrue(ready["eligible"], ready)
        self.assertEqual(ready["raw_body_status"], 1)
        self.assertEqual(ready["raw_action_desc"], "JOINT_DEFAULT")

        latest["common"]["data"]["body_status"]["value"] = 2
        not_standing = assess_joint_to_stand_state(
            latest,
            now_monotonic_ns=now_ns,
            reset_dispatch_monotonic_ns=reset_ns,
            upright_odom_samples=3,
            stand_common_samples=3,
            set_service_ready=True,
            get_service_ready=True,
            get_mode_desc="JOINT_DEFAULT",
            get_mode_status=100,
            body_stand_value=1,
        )
        self.assertFalse(not_standing["eligible"])
        self.assertEqual(not_standing["failure_reason"], "MC_body_status_not_STAND")

    def test_joint_default_with_too_few_stand_samples_blocks_sd(self) -> None:
        reset_ns = 35_000_000_000
        now_ns = reset_ns + 60_000_000
        latest = {
            "odom": {"wall_monotonic_ns": now_ns - 1_000_000, "data": {"z": 0.68, "roll_deg": 0.0, "pitch_deg": 0.0}},
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {"wall_monotonic_ns": now_ns - 3_000_000, "data": {"action_info": {"action_desc": "JOINT_DEFAULT", "status": {"value": 100}}, "body_status": {"value": 1}}},
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        result = assess_joint_to_stand_state(
            latest,
            now_monotonic_ns=now_ns,
            reset_dispatch_monotonic_ns=reset_ns,
            upright_odom_samples=3,
            stand_common_samples=2,
            set_service_ready=True,
            get_service_ready=True,
            get_mode_desc="JOINT_DEFAULT",
            get_mode_status=100,
            body_stand_value=1,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["failure_reason"], "fresh_STAND_body_status_samples_insufficient")

    def test_stale_or_unsafe_joint_to_stand_state_blocks_sd(self) -> None:
        reset_ns = 40_000_000_000
        now_ns = reset_ns + 900_000_000
        latest = {
            "odom": {"wall_monotonic_ns": now_ns - 1_000_000, "data": {"z": 0.45, "roll_deg": 0.0, "pitch_deg": 0.0}},
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {"wall_monotonic_ns": now_ns - 3_000_000, "data": {"action_info": {"action_desc": "JOINT_DEFAULT", "status": {"value": 100}}, "body_status": {"value": 1}}},
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        result = assess_joint_to_stand_state(
            latest,
            now_monotonic_ns=now_ns,
            reset_dispatch_monotonic_ns=reset_ns,
            upright_odom_samples=3,
            stand_common_samples=3,
            set_service_ready=True,
            get_service_ready=True,
            get_mode_desc="JOINT_DEFAULT",
            get_mode_status=100,
            body_stand_value=1,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["failure_reason"], "live_posture_outside_existing_gate")

    def test_takeover_timing_reports_each_handoff_interval(self) -> None:
        summary = summarize_takeover_timing({
            "reset_dispatch_monotonic_ns": 1_000_000_000,
            "first_valid_post_reset_state_monotonic_ns": 1_010_000_000,
            "first_upright_post_reset_monotonic_ns": 1_020_000_000,
            "joint_default": {"wall_monotonic_ns_submitted": 1_030_000_000, "wall_monotonic_ns_returned": 1_032_000_000},
            "body_stand_ready_monotonic_ns": 1_200_000_000,
            "stand_default": {"wall_monotonic_ns_submitted": 1_210_000_000, "wall_monotonic_ns_returned": 1_212_000_000},
            "first_effective_leg_command_after_jd_monotonic_ns": 1_050_000_000,
            "first_effective_leg_command_after_sd_monotonic_ns": 1_220_000_000,
            "first_unsafe_post_reset_monotonic_ns": 1_500_000_000,
        })
        self.assertEqual(summary["reset_to_first_valid_state_ms"], 10.0)
        self.assertEqual(summary["reset_to_JD_request_ms"], 30.0)
        self.assertEqual(summary["JD_response_to_first_effective_leg_command_ms"], 18.0)
        self.assertEqual(summary["reset_to_SD_request_ms"], 210.0)
        self.assertEqual(summary["SD_response_to_first_effective_leg_command_ms"], 8.0)
        self.assertEqual(summary["reset_to_first_unsafe_posture_ms"], 500.0)

    def test_effective_leg_command_detector_ignores_zeroed_output(self) -> None:
        self.assertFalse(command_is_effective({"joints": [{"position": 0.0, "stiffness": 0.0}]}))
        self.assertTrue(command_is_effective({"joints": [{"position": 0.0, "stiffness": 100.0}]}))

    def test_body_stand_constant_is_read_from_generated_message_instance(self) -> None:
        class GeneratedBodyPoseStatus:
            @property
            def MC_BODY_POSE_STAND(self):
                return 1

        self.assertEqual(installed_body_stand_value(GeneratedBodyPoseStatus), 1)

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
