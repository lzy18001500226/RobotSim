"""Deterministic offline tests for Issue 80 first-motion safety and stop handling."""

from __future__ import annotations

import threading
import unittest
import json
import tempfile
import time
from pathlib import Path

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME,
    ActivationTracker,
    COMMAND_QOS_DEPTH,
    COMMAND_QOS_DURABILITY,
    COMMAND_QOS_RELIABILITY,
    FORWARD_COMMAND_M_S,
    PULSE_LIMIT_S,
    PUBLISH_HZ,
    SOURCE_TIMEOUT_MS,
    VELOCITY_SOURCE_NAME,
    ZERO_STOP_MIN_S,
    ZERO_STOP_MIN_MESSAGES,
    classify_source_registration,
    load_leg_joint_limits,
    run_first_motion_window,
    snapshot_fault,
    validate_first_motion_authorization,
    validate_local_sim_publishers,
    validate_runtime_scope,
)
from issue80_m1_mode_logic import classify_set_mc_action_response


LIMITS = {
    "left_hip_pitch_joint": {
        "lower_position": -2.7,
        "upper_position": 2.5,
        "max_velocity": 11.936,
    }
}


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.lock = threading.Lock()

    def now(self) -> float:
        with self.lock:
            return self.value

    def sleep(self, duration: float) -> None:
        with self.lock:
            self.value += max(0.0, duration)


class FakeTimer:
    def __init__(self, timeout: float, callback) -> None:
        self.timeout = timeout
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.thread = None

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if self.started and not self.cancelled:
            self.callback()

    def fire_async_and_wait_for_expiry(self) -> None:
        if not self.started or self.cancelled:
            return
        watchdog = getattr(self.callback, "__self__", None)
        self.thread = threading.Thread(target=self.callback, daemon=True)
        self.thread.start()
        deadline = threading.Event()
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            if getattr(watchdog, "expired", False):
                return
            deadline.wait(0.001)
        raise AssertionError("independent watchdog callback did not expire")


class FirstMotionSafetyTests(unittest.TestCase):
    def snapshot(self, **overrides):
        value = {
            "state_age_s": {
                "odom": 0.01,
                "imu": 0.01,
                "common": 0.01,
                "leg_state": 0.01,
                "leg_command": 0.01,
                "get_mode": 0.01,
                "get_source": 0.01,
            },
            "get_mode_desc": "STAND_DEFAULT",
            "common_mode_desc": "STAND_DEFAULT",
            "get_mode_header_code": 0,
            "get_mode_status_value": 0,
            "common_mode_status_value": 0,
            "fsm_state": 2,
            "body_status": 1,
            "pose": {"x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "forward_bounds": {"min_value": -1.0, "max_value": 1.0},
            "get_source_header_code": 0,
            "get_source_name": "",
            "common_source_name": "",
            "leg_states": [
                {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 0.0}
            ],
        }
        value.update(overrides)
        return value

    def run_protocol(
        self,
        sample,
        publish,
        *,
        registration=None,
        action=None,
        shutdown_event=None,
        timer_factory=None,
        fake_clock=None,
    ):
        clock = fake_clock or FakeClock()
        return run_first_motion_window(
            sample=sample,
            publish=publish,
            registration=registration or {"success": True, "selected_action": "ADD"},
            set_action=action or {
                "accepted_by_service": True,
                "header_code": 0,
                "status_value": 1,
                "message": "success",
            },
            joint_limits=LIMITS,
            authorized=True,
            shutdown_event=shutdown_event,
            state_timeout_s=0.5,
            min_root_z=0.52,
            max_tilt_deg=15.0,
            clock=clock.now,
            sleep=clock.sleep,
            timer_factory=timer_factory,
        )

    def test_leg_limits_are_loaded_from_the_installed_robot_model_yaml(self) -> None:
        names = [f"leg_joint_{index}" for index in range(12)]
        document = {
            "joints": [["LEG", [
                {"name": name, "limits": {
                    "lower_position": -1.0, "upper_position": 1.0, "max_velocity": 2.0,
                }}
                for name in names
            ]]]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "robot_model.yaml"
            source.write_text(json.dumps(document), encoding="utf-8")
            limits, digest = load_leg_joint_limits(source)
        self.assertEqual(set(limits), set(names))
        self.assertEqual(limits[names[0]]["max_velocity"], 2.0)
        self.assertEqual(len(digest), 64)

    def test_protocol_constants_are_hard_limited_to_vendor_example(self) -> None:
        self.assertEqual(VELOCITY_SOURCE_NAME, "node.locomotion_velocity")
        self.assertEqual(ACTION_SOURCE_NAME, "rc")
        self.assertEqual(FORWARD_COMMAND_M_S, 0.2)
        self.assertEqual(PULSE_LIMIT_S, 1.0)
        self.assertEqual(PUBLISH_HZ, 50.0)
        self.assertEqual(SOURCE_TIMEOUT_MS, 1000)
        self.assertEqual(ZERO_STOP_MIN_S, 1.0)
        self.assertEqual((COMMAND_QOS_RELIABILITY, COMMAND_QOS_DEPTH, COMMAND_QOS_DURABILITY),
                         ("reliable", 10, "volatile"))

    def test_live_entry_gate_requires_review_reset_and_bridge_before_ros_start(self) -> None:
        with self.assertRaisesRegex(PermissionError, "Reset"):
            validate_first_motion_authorization(
                reset_gate_verified=False, maintainer_authorized=True, network_mode="bridge"
            )
        with self.assertRaisesRegex(PermissionError, "maintainer authorization"):
            validate_first_motion_authorization(
                reset_gate_verified=True, maintainer_authorized=False, network_mode="bridge"
            )
        with self.assertRaisesRegex(ValueError, "bridge"):
            validate_first_motion_authorization(
                reset_gate_verified=True, maintainer_authorized=True, network_mode=None
            )
        validate_first_motion_authorization(
            reset_gate_verified=True, maintainer_authorized=True, network_mode="bridge"
        )

    def test_source_registration_accepts_documented_add_success_with_unknown_task_state(self) -> None:
        result = classify_source_registration(
            {"response_header_code": 0, "response_state": 0, "task_id": 17}
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["selected_action"], "ADD")
        self.assertEqual(result["add_response"]["response_state"], 0)
        self.assertTrue(result["accepted_by_service"])
        self.assertFalse(result["task_completion_confirmed"])
        self.assertFalse(result["enable_fallback_allowed"])

    def test_source_registration_falls_back_to_enable_only_after_add_rejection(self) -> None:
        result = classify_source_registration(
            {"response_header_code": 3, "response_state": 5, "task_id": 2},
            {"response_header_code": 0, "response_state": 1, "task_id": 3},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["selected_action"], "ENABLE")
        self.assertTrue(result["task_completion_confirmed"])

    def test_both_add_and_enable_rejection_blocks_motion(self) -> None:
        result = classify_source_registration(
            {"response_header_code": 3, "response_state": 5, "task_id": 2},
            {"response_header_code": 5, "response_state": 5, "task_id": 3},
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["enable_fallback_allowed"])

    def test_ambiguous_add_timeout_does_not_authorize_enable_replay(self) -> None:
        result = classify_source_registration(None)
        self.assertFalse(result["success"])
        self.assertFalse(result["accepted_by_service"])
        self.assertFalse(result["enable_fallback_allowed"])

    def test_terminal_task_failure_blocks_source_use_even_if_service_accepted(self) -> None:
        result = classify_source_registration({
            "response_header_code": 0, "response_state": 2, "task_id": 17,
        })
        self.assertTrue(result["accepted_by_service"])
        self.assertTrue(result["task_terminal_failure"])
        self.assertFalse(result["success"])
        self.assertTrue(result["enable_fallback_allowed"])

    def test_add_not_ready_state_blocks_motion_without_replaying_enable(self) -> None:
        result = classify_source_registration({
            "response_header_code": 0, "response_state": 100, "task_id": 17,
        })
        self.assertTrue(result["accepted_by_service"])
        self.assertTrue(result["task_state_unready"])
        self.assertFalse(result["success"])
        self.assertFalse(result["enable_fallback_allowed"])

    def test_enable_may_recover_a_terminal_add_task_failure_once(self) -> None:
        result = classify_source_registration(
            {"response_header_code": 0, "response_state": 2, "task_id": 17},
            {"response_header_code": 0, "response_state": 1, "task_id": 18},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["selected_action"], "ENABLE")
        self.assertEqual(result["enable_fallback_reason"], "ADD_TASK_TERMINAL_FAILURE")

    def test_source_registration_rejection_sends_no_nonzero_motion(self) -> None:
        commands = []
        result = self.run_protocol(
            lambda: self.snapshot(),
            commands.append,
            registration={"success": False, "selected_action": None},
        )
        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["fault"], "source_registration_rejected")
        self.assertEqual(result["nonzero_messages"], 0)
        self.assertGreaterEqual(result["zero_stop"]["actual_duration_s"], 1.0)

    def test_set_action_rejection_sends_no_nonzero_motion(self) -> None:
        commands = []
        result = self.run_protocol(
            lambda: self.snapshot(),
            commands.append,
            action={"accepted_by_service": False, "header_code": 6, "status_value": 2},
        )
        self.assertEqual(result["fault"], "set_mc_action_rejected")
        self.assertEqual(result["nonzero_messages"], 0)

    def test_action_response_preserves_header_status_and_message(self) -> None:
        result = classify_set_mc_action_response(6, 2, "security protection")
        self.assertFalse(result["accepted_by_service"])
        self.assertEqual(result["response_header_code"], 6)
        self.assertEqual(result["response_status_value"], 2)
        self.assertEqual(result["message"], "security protection")

    def test_transport_scope_requires_bridge_domain_and_one_mc_endpoint(self) -> None:
        self.assertIsNone(validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
        ))
        self.assertIn("network", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="host",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
        ))
        self.assertIn("publisher", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic", "other"],
            subscriber_names=["mc_ros2_node"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
        ))

    def test_transport_scope_rejects_rmw_or_domain_mismatch(self) -> None:
        common = dict(
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
        )
        self.assertIn("rmw", validate_runtime_scope(
            rmw_implementation="rmw_cyclonedds_cpp", ros_domain_id="20", **common
        ))
        self.assertIn("domain", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp", ros_domain_id="0", **common
        ))

    def test_transport_scope_rejects_best_effort_command_publisher(self) -> None:
        self.assertIn("publisher_qos", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="best_effort",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
        ))

    def test_transport_scope_rejects_extra_velocity_subscriber(self) -> None:
        self.assertIn("subscriber", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node", "external_controller"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort", "best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile", "volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last", "keep_last"],
            subscriber_depths=[10, 10],
        ))

    def test_local_state_and_mc_command_topics_require_single_publishers(self) -> None:
        topics = {
            "/aima/hal/odom/state": ["aima-sim"],
            "/aima/hal/imu/torso/state": ["aima-sim"],
            "/aima/hal/joint/leg/state": ["aima-sim"],
            "/aima/hal/joint/leg/command": ["mc_ros2_node"],
        }
        self.assertIsNone(validate_local_sim_publishers(topics))
        topics["/aima/hal/joint/leg/state"].append("external_robot")
        self.assertIn("unexpected_local_runtime_publishers", validate_local_sim_publishers(topics))

    def test_stale_telemetry_rejects_before_motion_and_zero_stops(self) -> None:
        sample = self.snapshot()
        sample["state_age_s"]["imu"] = 0.6
        commands = []
        result = self.run_protocol(lambda: sample, commands.append)
        self.assertEqual(result["fault"], "stale_or_missing_state:imu")
        self.assertEqual(result["nonzero_messages"], 0)
        self.assertGreaterEqual(result["zero_stop"]["actual_duration_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["zero_messages"], 49)

    def test_source_conflict_rejects_before_motion(self) -> None:
        sample = self.snapshot(get_source_name="rc")
        commands = []
        result = self.run_protocol(lambda: sample, commands.append)
        self.assertIn("input_source_conflict", result["fault"])
        self.assertEqual(result["nonzero_messages"], 0)

    def test_fall_height_and_tilt_guards_reject_before_motion(self) -> None:
        for pose, expected in (
            ({"x": 0.0, "y": 0.0, "z": 0.51, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, "fall_root_height"),
            ({"x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.3, "pitch": 0.0, "yaw": 0.0}, "fall_tilt"),
        ):
            with self.subTest(expected=expected):
                sample = self.snapshot(pose=pose)
                result = self.run_protocol(lambda: sample, lambda _v: None)
                self.assertEqual(result["fault"], expected)
                self.assertEqual(result["nonzero_messages"], 0)

    def test_safety_fsm_and_source_joint_velocity_limit_are_enforced(self) -> None:
        safe_fsm = self.snapshot(fsm_state=4)
        self.assertEqual(snapshot_fault(
            safe_fsm, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ), "mc_safety_protection_state")
        fast_joint = self.snapshot(leg_states=[
            {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 11.937}
        ])
        self.assertEqual(snapshot_fault(
            fast_joint, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ), "source_joint_velocity_limit:left_hip_pitch_joint")

    def test_expected_stand_locomotion_transition_is_allowed_but_other_modes_are_not(self) -> None:
        transitioning = self.snapshot(
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="STAND_DEFAULT",
            fsm_state=1,
        )
        self.assertIsNone(snapshot_fault(
            transitioning, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ))
        unrelated = self.snapshot(
            get_mode_desc="PASSIVE_DEFAULT",
            common_mode_desc="PASSIVE_DEFAULT",
        )
        self.assertIn("unrelated_or_missing_mode", snapshot_fault(
            unrelated, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ))

    def test_activation_evidence_reports_observed_odom_delta(self) -> None:
        tracker = ActivationTracker()
        tracker.observe(self.snapshot())
        tracker.observe(self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
            pose={"x": 0.12, "y": -0.02, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.1},
        ))
        delta = tracker.odometry_delta()
        self.assertEqual(delta["dx_m"], 0.12)
        self.assertEqual(delta["dy_m"], -0.02)
        self.assertEqual(delta["sample_count"], 2)

    def test_forward_command_must_be_inside_live_speed_bounds(self) -> None:
        sample = self.snapshot(forward_bounds={"min_value": -0.1, "max_value": 0.1})
        self.assertEqual(snapshot_fault(
            sample, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ), "activation_speed_outside_live_mc_bounds")

    def test_missing_mode_transition_is_reported_after_single_bounded_pulse(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["activation_classification"], "MISSING_MODE_TRANSITION")
        self.assertEqual(result["status"], "MISSING_MODE_TRANSITION")
        self.assertLessEqual(result["last_nonzero_command_offset_s"], PULSE_LIMIT_S)
        self.assertLessEqual(result["pulse_loop_elapsed_s"], 1.0)
        self.assertTrue(all(value in (0.0, 0.2) for value in commands))

    def test_publish_call_returning_after_hard_deadline_fails_closed(self) -> None:
        clock = FakeClock()
        sample = self.snapshot()

        def publish(value):
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
                with clock.lock:
                    clock.value += 1.01

        result = self.run_protocol(lambda: sample, publish, fake_clock=clock)
        self.assertEqual(result["fault"], "nonzero_publish_returned_after_hard_deadline")
        self.assertEqual(result["status"], "ABORTED")
        self.assertGreater(result["last_nonzero_command_offset_s"], PULSE_LIMIT_S)

    def test_success_requires_source_and_both_ld_readback_channels(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["status"], "ACTIVATION_CONFIRMED")
        self.assertGreater(result["nonzero_messages"], 0)
        self.assertTrue(result["source_selected_by_get_service"])
        self.assertTrue(result["source_selected_by_common_state"])
        self.assertGreaterEqual(result["zero_stop"]["actual_duration_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["published_span_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["zero_messages"], ZERO_STOP_MIN_MESSAGES)
        self.assertLess(result["zero_stop"]["max_inter_message_gap_s"], 1.0)
        self.assertLessEqual(sum(value == 0.2 for value in commands), 50)
        self.assertTrue(all(0.0 <= offset <= PULSE_LIMIT_S for offset in result["nonzero_command_attempt_offsets_s"]))
        self.assertGreaterEqual(sum(value == 0.0 for value in commands), 50)
        self.assertGreaterEqual(result["post_stop_sample_count"], 1)

    def test_source_conflict_during_activation_aborts_and_zero_stops(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["get_source_name"] = "rc"
                sample["common_source_name"] = "rc"

        result = self.run_protocol(lambda: sample, publish)
        self.assertIn("input_source_conflict", result["fault"])
        self.assertLessEqual(result["nonzero_messages"], 1)
        self.assertGreaterEqual(result["zero_stop"]["zero_messages"], 50)

    def test_fall_during_activation_aborts_and_zero_stops(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["pose"]["z"] = 0.51

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["fault"], "fall_root_height")
        self.assertLessEqual(result["nonzero_messages"], 1)
        self.assertGreaterEqual(result["zero_stop"]["published_span_s"], 1.0)

    def test_stale_telemetry_during_activation_aborts(self) -> None:
        sample = self.snapshot()
        count = 0
        commands = []

        def read():
            nonlocal count
            count += 1
            if count > 2:
                sample["state_age_s"]["odom"] = 0.6
            return sample

        result = self.run_protocol(read, commands.append)
        self.assertEqual(result["fault"], "stale_or_missing_state:odom")
        self.assertLessEqual(result["nonzero_messages"], 1)

    def test_post_pulse_zero_delivery_failure_is_reported(self) -> None:
        sample = self.snapshot()
        pulse_started = False

        def publish(value):
            nonlocal pulse_started
            if value == 0.2:
                pulse_started = True
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
                return
            if pulse_started:
                raise RuntimeError("injected zero publish failure")

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["status"], "ZERO_STOP_FAILED")
        self.assertTrue(result["zero_stop"]["publish_errors"])

    def test_fall_during_post_stop_observation_restarts_zero_stop(self) -> None:
        sample = self.snapshot()

        def publish(value):
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
            elif value == 0.0:
                sample["pose"]["z"] = 0.50

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["status"], "POST_STOP_SAFETY_FAILED")
        self.assertEqual(result["post_stop_fault"], "fall_root_height")
        self.assertGreaterEqual(result["post_stop_emergency_zero"]["actual_duration_s"], 1.0)

    def test_independent_watchdog_fires_and_forces_full_zero_window(self) -> None:
        sample = self.snapshot()
        commands = []
        timers = []

        def timer_factory(timeout, callback):
            timer = FakeTimer(timeout, callback)
            timers.append(timer)
            return timer

        def publish(value):
            commands.append(value)
            if value == 0.2 and len(timers) == 1:
                timers[0].fire_async_and_wait_for_expiry()

        result = self.run_protocol(lambda: sample, publish, timer_factory=timer_factory)
        self.assertEqual(result["fault"], "independent_watchdog_timeout")
        self.assertTrue(any(value == 0.0 for value in commands))
        self.assertGreaterEqual(result["zero_stop"]["actual_duration_s"], 1.0)
        self.assertEqual(timers[0].timeout, 1.0)
        self.assertIsNotNone(timers[0].thread)
        timers[0].thread.join(timeout=1.0)
        self.assertFalse(timers[0].thread.is_alive())

    def test_exception_during_sampling_forces_zero_timeout_window(self) -> None:
        sample = self.snapshot()
        count = 0
        commands = []

        def read():
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("injected observer exception")
            return sample

        result = self.run_protocol(read, commands.append)
        self.assertIn("exception:RuntimeError:injected observer exception", result["fault"])
        self.assertGreater(result["pulse_loop_elapsed_s"], 0.0)
        self.assertGreaterEqual(result["zero_stop"]["zero_messages"], 49)

    def test_exception_during_publish_forces_zero_timeout_window(self) -> None:
        sample = self.snapshot()
        calls = 0
        commands = []

        def publish(value):
            nonlocal calls
            calls += 1
            if value == 0.2:
                raise RuntimeError("injected publish exception")
            commands.append(value)

        result = self.run_protocol(lambda: sample, publish)
        self.assertIn("exception:RuntimeError:injected publish exception", result["fault"])
        self.assertEqual(result["nonzero_messages"], 0)
        self.assertGreaterEqual(result["zero_stop"]["zero_messages"], 49)

    def test_shutdown_during_activation_forces_zero_timeout_window(self) -> None:
        sample = self.snapshot()
        shutdown = threading.Event()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                shutdown.set()
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME

        result = self.run_protocol(lambda: sample, publish, shutdown_event=shutdown)
        self.assertEqual(result["fault"], "operator_shutdown")
        self.assertGreaterEqual(result["zero_stop"]["actual_duration_s"], 1.0)

    def test_unauthorized_call_emits_no_command(self) -> None:
        commands = []
        result = run_first_motion_window(
            sample=lambda: self.snapshot(),
            publish=commands.append,
            registration={"success": True},
            set_action={"accepted_by_service": True},
            joint_limits=LIMITS,
            authorized=False,
        )
        self.assertEqual(result["status"], "NOT_AUTHORIZED")
        self.assertEqual(commands, [])

    def test_zero_stop_rejects_short_successful_delivery_span(self) -> None:
        clock = FakeClock()

        def slow_zero(command):
            if command == 0.0:
                with clock.lock:
                    clock.value += 0.25

        result = self.run_protocol(
            lambda: self.snapshot(), slow_zero, fake_clock=clock
        )
        self.assertEqual(result["status"], "ZERO_STOP_DELIVERY_SPAN_TOO_SHORT")
        self.assertLess(result["zero_stop"]["published_span_s"], SOURCE_TIMEOUT_MS / 1000.0)


if __name__ == "__main__":
    unittest.main()
