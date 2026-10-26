"""Offline schema checks for the one-shot SetMcAction response recorder."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue80_m1_set_action_capture import (
    assess_joint_to_stand_state,
    assess_pre_action_state,
    command_is_effective,
    native_foot_support,
    native_synchronized_keyframe_state,
    ros_leg_state_matches_synchronized_keyframe,
    read_complete_native_capture_rows,
    installed_body_stand_value,
    _raw_fsm_state_value,
    set_action_record,
    standing_observation_window,
    summarize_takeover_timing,
    write_complete_json,
)


class FakeMessage:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def get_fields_and_field_types(self):
        return {name: "fake" for name in self.__dict__}


class SetActionCaptureTests(unittest.TestCase):
    def test_native_support_requires_positive_contacts_on_both_selected_feet(self) -> None:
        sample = {
            "contact_count": 3,
            "captured_foot_floor_contact_count": 2,
            "contacts": [
                {"involves_selected_left_foot": True, "normal_force_n": 42.0},
                {"involves_selected_right_foot": True, "normal_force_n": 39.0},
                {"involves_selected_left_foot": True, "normal_force_n": -1.0},
            ],
        }
        support = native_foot_support(sample)
        self.assertTrue(support["supported"])
        self.assertFalse(support["safe_to_begin_ros_upright_gate"])
        self.assertEqual(support["left_contact_count"], 1)
        self.assertEqual(support["right_contact_count"], 1)
        self.assertEqual(support["left_normal_force_n"], 42.0)

        sample["contacts"] = sample["contacts"][:1]
        self.assertFalse(native_foot_support(sample)["supported"])

    def test_native_support_requires_a_settled_root_after_bilateral_contact(self) -> None:
        sample = {
            "pelvis": {"state": {
                "world_position_m": [0.0, 0.0, 0.68],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "world_spatial_velocity_rot_lin": [0, 0, 0, 0.03, 0.0, 0.001],
            }},
            "contacts": [
                {"involves_selected_left_foot": True, "normal_force_n": 100.0},
                {"involves_selected_right_foot": True, "normal_force_n": 100.0},
            ],
        }
        self.assertTrue(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])
        sample["pelvis"]["state"]["world_spatial_velocity_rot_lin"][5] = 0.035
        self.assertFalse(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])
        sample["pelvis"]["state"]["world_spatial_velocity_rot_lin"][5] = -0.004
        self.assertTrue(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])
        sample["pelvis"]["state"]["world_spatial_velocity_rot_lin"][3] = 0.21
        self.assertFalse(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])

    def test_native_support_requires_native_height_and_tilt_not_only_foot_force(self) -> None:
        sample = {
            "pelvis": {"state": {
                "world_position_m": [0.0, 0.0, 0.68],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "world_spatial_velocity_rot_lin": [0, 0, 0, 0.02, 0.0, 0.0],
            }},
            "contacts": [
                {"involves_selected_left_foot": True, "normal_force_n": 100.0},
                {"involves_selected_right_foot": True, "normal_force_n": 100.0},
            ],
        }
        support = native_foot_support(sample)
        self.assertTrue(support["safe_to_begin_ros_upright_gate"])
        self.assertEqual(support["pelvis_root_z_m"], 0.68)
        self.assertEqual(support["pelvis_tilt_deg"], 0.0)
        sample["pelvis"]["state"]["world_position_m"][2] = 0.51
        self.assertFalse(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])

    def test_synchronized_keyframe_requires_exact_static_qpos_zero_qvel_and_bilateral_tangent_contacts(self) -> None:
        sample = {
            "qpos": [1.0, 2.0, 3.0],
            "qvel": [0.0, 0.0],
            "pelvis": {"state": {
                "world_position_m": [0.0, 0.0, 0.68],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "world_spatial_velocity_rot_lin": [0, 0, 0, 0.0, 0.0, 0.0],
            }},
            "contacts": [
                {"involves_selected_left_foot": True, "distance_m": 0.0},
                {"involves_selected_right_foot": True, "distance_m": -1e-10},
            ],
        }
        expected = {"qpos": [1.0, 2.0, 3.0], "qvel": [0.0, 0.0]}
        accepted = native_synchronized_keyframe_state(sample, expected)
        self.assertTrue(accepted["eligible"])
        sample["qvel"][0] = 1e-3
        self.assertFalse(native_synchronized_keyframe_state(sample, expected)["eligible"])
        sample["qvel"][0] = 0.0
        sample["contacts"][1]["distance_m"] = -1e-3
        self.assertFalse(native_synchronized_keyframe_state(sample, expected)["eligible"])
        sample["pelvis"]["state"]["world_position_m"][2] = 0.68
        # 30 deg pitch: a valid quaternion but outside the unchanged 15 deg gate.
        sample["pelvis"]["state"]["world_quaternion_wxyz"] = [0.965925826, 0.0, 0.258819045, 0.0]
        self.assertFalse(native_foot_support(sample)["safe_to_begin_ros_upright_gate"])

    def test_synchronized_keyframe_requires_post_load_ros_leg_state_to_match_mc_nominal_legs(self) -> None:
        names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        values = [-0.24, 0.0, 0.0, 0.45, -0.21, 0.0] * 2
        expected = {
            "source_nominal_joint_names": names + ["waist_yaw_joint"],
            "source_nominal_joint_values_rad": values + [0.0],
        }
        state = {"joints": [
            {"name": name, "position": value} for name, value in zip(names, values)
        ]}
        accepted = ros_leg_state_matches_synchronized_keyframe(state, expected)
        self.assertTrue(accepted["eligible"])
        self.assertEqual(accepted["observed_matching_leg_joint_count"], 12)
        state["joints"][0]["position"] = -0.10
        self.assertFalse(ros_leg_state_matches_synchronized_keyframe(state, expected)["eligible"])
        state["joints"] = state["joints"][:-1]
        rejected = ros_leg_state_matches_synchronized_keyframe(state, expected)
        self.assertFalse(rejected["eligible"])
        self.assertEqual(rejected["missing_leg_joint_names"], ["right_ankle_roll_joint"])

    def test_native_capture_reader_defers_a_partial_final_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "native.jsonl"
            first = {"event": "post_step", "generation": 0, "host_monotonic_ns": 1}
            partial = '{"event":"post_step","generation":1'
            trace.write_text(json.dumps(first) + "\n" + partial, encoding="utf-8")
            offset, rows = read_complete_native_capture_rows(trace, 0)
            self.assertEqual(rows, [first])
            self.assertLess(offset, trace.stat().st_size)
            with trace.open("a", encoding="utf-8") as stream:
                stream.write(',"host_monotonic_ns":2}\n')
            offset, rows = read_complete_native_capture_rows(trace, offset)
            self.assertEqual(rows, [{"event": "post_step", "generation": 1, "host_monotonic_ns": 2}])
            self.assertEqual(offset, trace.stat().st_size)

    def test_raw_fsm_capture_reads_nested_current_state_without_interpreting_it(self) -> None:
        self.assertEqual(_raw_fsm_state_value({"fsm_state": {"current_state": 4}}), 4)
        self.assertEqual(_raw_fsm_state_value({"fsm_state": {"value": 4}}), 4)
        self.assertIsNone(_raw_fsm_state_value({"fsm_state": {"other": 4}}))

    def test_standing_baseline_duration_starts_at_positive_stand_readback(self) -> None:
        window = standing_observation_window(
            reset_dispatch_monotonic_ns=1_000_000_000,
            stand_default_confirmed_monotonic_ns=4_500_000_000,
            duration_s=8.0,
        )
        self.assertEqual(window["anchor"], "STAND_DEFAULT_GETMC_READBACK")
        self.assertEqual(window["anchor_monotonic_ns"], 4_500_000_000)
        self.assertEqual(window["deadline_monotonic_ns"], 12_500_000_000)

    def test_unconfirmed_stand_cannot_be_reported_as_an_eight_second_standing_window(self) -> None:
        window = standing_observation_window(
            reset_dispatch_monotonic_ns=1_000_000_000,
            stand_default_confirmed_monotonic_ns=None,
            duration_s=8.0,
        )
        self.assertEqual(window["anchor"], "RESET_FALLBACK_NO_CONFIRMED_STAND")
        self.assertEqual(window["deadline_monotonic_ns"], 9_000_000_000)

    def test_parent_cannot_observe_a_partially_written_result(self) -> None:
        original_dump = json.dump
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            def checked_dump(value, stream, **kwargs):
                stream.write(" ")
                stream.flush()
                self.assertFalse(target.exists())
                original_dump(value, stream, **kwargs)
                self.assertFalse(target.exists())
            with patch("issue80_m1_set_action_capture.json.dump", side_effect=checked_dump):
                write_complete_json(target, {"status": "COMPLETE", "trace": [1] * 10000})
            self.assertEqual(json.loads(target.read_text())["status"], "COMPLETE")
            with self.assertRaises(FileExistsError):
                write_complete_json(target, {"status": "OVERWRITE"})
            self.assertEqual(json.loads(target.read_text())["status"], "COMPLETE")

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

    def test_native_handoff_anchors_jd_to_mjdata_and_reanchors_ros_after_effective_command(self) -> None:
        source = Path(__file__).with_name("issue80_m1_set_action_capture.py").read_text()
        start = source.index("if self.require_native_reset_generation:", source.index("def wait_for_post_reset_upright"))
        native_gate = source[start:source.index("\n            while rclpy.ok()", start)]
        self.assertIn('"POST_RESET_NATIVE_SUPPORT_READY"', native_gate)
        self.assertIn("ros_transport_liveness", native_gate)
        self.assertIn("physical_state_authority", native_gate)
        self.assertIn("PENDING_POST_JD_EFFECTIVE_COMMAND", native_gate)
        self.assertIn("post_JD_effective_leg_command_epoch", source)
        self.assertIn("native_pre_JD_physical_safety_fault", source)
        self.assertIn("native_post_JD_physical_safety_fault", source)

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

    def test_native_generation_barrier_allows_one_fresh_ros_pose_only_after_supported_state(self) -> None:
        now_ns = 50_000_000_000
        latest = {
            "odom": {"wall_monotonic_ns": now_ns - 1_000_000, "data": {"z": 0.68, "roll_deg": 0.0, "pitch_deg": 0.0}},
            "imu": {"wall_monotonic_ns": now_ns - 2_000_000, "data": {}},
            "common": {"wall_monotonic_ns": now_ns - 3_000_000, "data": {}},
            "leg_state": {"wall_monotonic_ns": now_ns - 4_000_000, "data": {}},
            "leg_command": {"wall_monotonic_ns": now_ns - 5_000_000, "data": {}},
        }
        one_sample = assess_pre_action_state(
            latest,
            now_monotonic_ns=now_ns,
            upright_odom_samples=1,
            required_upright_odom_samples=1,
            service_ready=True,
            reset_dispatch_monotonic_ns=now_ns - 10_000_000,
        )
        self.assertTrue(one_sample["eligible"])
        legacy_three_sample = assess_pre_action_state(
            latest,
            now_monotonic_ns=now_ns,
            upright_odom_samples=1,
            service_ready=True,
            reset_dispatch_monotonic_ns=now_ns - 10_000_000,
        )
        self.assertFalse(legacy_three_sample["eligible"])

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
            "native_reset_generation_host_monotonic_ns": 1_015_000_000,
            "native_support_ready_host_monotonic_ns": 1_025_000_000,
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
        self.assertEqual(summary["reset_dispatch_to_native_generation_ms"], 15.0)
        self.assertEqual(summary["native_generation_to_bilateral_support_ms"], 10.0)
        self.assertEqual(summary["native_support_to_JD_request_ms"], 5.0)
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
