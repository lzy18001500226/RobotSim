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
    BASELINE_DURATION_S,
    BASELINE_SAMPLE_PERIOD_S,
    BASELINE_WINDOW_COUNT,
    FORWARD_COMMAND_M_S,
    PULSE_LIMIT_S,
    PUBLISH_HZ,
    SOURCE_TIMEOUT_MS,
    VELOCITY_SOURCE_NAME,
    ZERO_STOP_MIN_S,
    ZERO_STOP_MIN_MESSAGES,
    classify_source_registration,
    classify_physical_motion,
    derive_stationary_odom_baseline,
    load_leg_joint_limits,
    load_mc_input_source_timeouts,
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
        raise AssertionError("process-local timer callback did not expire")


class FirstMotionSafetyTests(unittest.TestCase):
    def snapshot(self, **overrides):
        value = {
            "sample_monotonic_s": 0.0,
            "readback_monotonic_s": {"get_mode": 0.0, "get_source": 0.0},
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
        baseline=None,
    ):
        clock = fake_clock or FakeClock()

        def read_sample():
            row = dict(sample())
            now = clock.now()
            row["sample_monotonic_s"] = now
            row.setdefault("readback_monotonic_s", {"get_mode": now, "get_source": now})
            return row

        return run_first_motion_window(
            sample=read_sample,
            publish=publish,
            registration=registration or {"success": True, "selected_action": "ADD"},
            set_action=action or {
                "accepted_by_service": True,
                "header_code": 0,
                "status_value": 1,
                "message": "success",
            },
            joint_limits=LIMITS,
            baseline_forward_displacements_m=baseline if baseline is not None else [0.0, 0.0, 0.0],
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

    def test_installed_mc_timeout_config_is_recorded_as_configuration_not_runtime_proof(self) -> None:
        document = {"MotionControlModule": {"InputManager": {"input_sources": [
            {"name": "rc", "priority": 80, "timeout": 1000},
            {"name": "pnc", "priority": 40, "timeout": 1000},
        ]}}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mc_sim.yaml"
            source.write_text(json.dumps(document), encoding="utf-8")
            sources, digest = load_mc_input_source_timeouts(source)
        self.assertEqual(sources[0], {"name": "rc", "priority": 80, "timeout_ms": 1000})
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
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        self.assertFalse(result["zero_stop"]["mc_stop_confirmed"])

    def test_set_action_rejection_sends_no_nonzero_motion(self) -> None:
        commands = []
        result = self.run_protocol(
            lambda: self.snapshot(),
            commands.append,
            action={"accepted_by_service": False, "header_code": 6, "status_value": 2},
        )
        self.assertEqual(result["fault"], "set_mc_action_rejected")
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)

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
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], 49)

    def test_source_conflict_rejects_before_motion(self) -> None:
        sample = self.snapshot(get_source_name="rc")
        commands = []
        result = self.run_protocol(lambda: sample, commands.append)
        self.assertIn("input_source_conflict", result["fault"])
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)

    def test_fall_height_and_tilt_guards_reject_before_motion(self) -> None:
        for pose, expected in (
            ({"x": 0.0, "y": 0.0, "z": 0.51, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, "fall_root_height"),
            ({"x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.3, "pitch": 0.0, "yaw": 0.0}, "fall_tilt"),
        ):
            with self.subTest(expected=expected):
                sample = self.snapshot(pose=pose)
                result = self.run_protocol(lambda: sample, lambda _v: None)
                self.assertEqual(result["fault"], expected)
                self.assertEqual(result["nonzero_publish_calls_returned"], 0)

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
        tracker.observe(self.snapshot(), phase="pre-activation")
        tracker.observe(self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
            pose={"x": 0.12, "y": -0.02, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.1},
        ), phase="activation")
        delta = tracker.odometry_delta()
        self.assertEqual(delta["dx_m"], 0.12)
        self.assertEqual(delta["dy_m"], -0.02)
        self.assertGreater(delta["forward_projected_m"], 0.0)
        self.assertEqual(delta["sample_count"], 2)

    def test_source_and_ld_seen_at_different_samples_are_not_a_control_handshake(self) -> None:
        tracker = ActivationTracker()
        tracker.observe(self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
        ), phase="activation")
        tracker.observe(self.snapshot(
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
        ), phase="activation")
        self.assertTrue(tracker.source_acquired)
        self.assertTrue(tracker.ld_observed)
        self.assertFalse(tracker.control_handshake_confirmed)
        self.assertEqual(tracker.classify_control_handshake(), "CONTROL_HANDSHAKE_NOT_CONFIRMED")

    def test_same_sample_with_async_readback_skew_is_not_a_control_handshake(self) -> None:
        tracker = ActivationTracker()
        row = self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
        )
        row["readback_monotonic_s"] = {"get_mode": 1.0, "get_source": 1.21}
        tracker.observe(row, phase="activation")
        self.assertTrue(tracker.source_acquired)
        self.assertTrue(tracker.ld_observed)
        self.assertFalse(tracker.control_handshake_confirmed)
        self.assertGreater(tracker.samples[0]["mode_source_readback_skew_s"], 0.2)

    def test_stale_common_state_within_general_gate_is_not_contemporaneous_handshake(self) -> None:
        tracker = ActivationTracker()
        row = self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
        )
        row["state_age_s"]["common"] = 0.25
        tracker.observe(row, phase="activation")
        self.assertTrue(tracker.source_acquired)
        self.assertTrue(tracker.ld_observed)
        self.assertFalse(tracker.control_handshake_confirmed)
        self.assertGreater(tracker.samples[0]["mode_source_common_skew_max_s"], 0.2)

    def test_same_sample_and_aligned_readbacks_confirm_control_handshake_only(self) -> None:
        tracker = ActivationTracker()
        tracker.observe(self.snapshot(
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            fsm_state=3,
        ), phase="activation")
        self.assertTrue(tracker.control_handshake_confirmed)
        self.assertEqual(tracker.classify_control_handshake(), "CONTROL_HANDSHAKE_CONFIRMED")

    def test_physical_motion_requires_positive_displacement_above_empirical_noise(self) -> None:
        self.assertEqual(classify_physical_motion(
            control_handshake_confirmed=True,
            pulse_forward_displacement_m=0.0001,
            baseline_noise_bound_m=0.0,
        ), "BASELINE_NOISE_UNRESOLVED")
        self.assertEqual(classify_physical_motion(
            control_handshake_confirmed=True,
            pulse_forward_displacement_m=0.01,
            baseline_noise_bound_m=0.01,
        ), "CONTROL_HANDSHAKE_ONLY")
        self.assertEqual(classify_physical_motion(
            control_handshake_confirmed=True,
            pulse_forward_displacement_m=0.011,
            baseline_noise_bound_m=0.01,
        ), "PHYSICAL_MOTION_CONFIRMED")
        self.assertEqual(classify_physical_motion(
            control_handshake_confirmed=False,
            pulse_forward_displacement_m=0.02,
            baseline_noise_bound_m=0.01,
        ), "UNEXPECTED_MOTION_WITHOUT_CONTROL_HANDSHAKE")

    def test_stationary_noise_bound_uses_observed_nonzero_windows_not_a_vendor_threshold(self) -> None:
        samples = []
        for index in range(int(BASELINE_DURATION_S / BASELINE_SAMPLE_PERIOD_S) + 1):
            t = index * BASELINE_SAMPLE_PERIOD_S
            samples.append({
                "sample_monotonic_s": t,
                "pose": {"x": 0.001 * t, "y": 0.0, "yaw": 0.0},
            })
        result = derive_stationary_odom_baseline(samples)
        self.assertEqual(result["window_count"], BASELINE_WINDOW_COUNT)
        self.assertTrue(all(abs(row["forward_displacement_m"] - 0.001) < 1e-9 for row in result["windows"]))
        self.assertEqual(result["noise_bound_m"], 0.001)
        self.assertFalse(result["statistical_confidence_bound"])

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
        self.assertEqual(result["control_handshake_status"], "LD_NOT_OBSERVED")
        self.assertEqual(result["physical_motion_status"], "BASELINE_NOISE_UNRESOLVED")
        self.assertLessEqual(result["last_nonzero_command_offset_s"], PULSE_LIMIT_S)
        self.assertLessEqual(result["pulse_loop_elapsed_s"], 1.0)
        self.assertTrue(all(value in (0.0, 0.2) for value in commands))

    def test_partial_or_missing_baseline_blocks_nonzero_motion(self) -> None:
        commands = []
        result = self.run_protocol(
            lambda: self.snapshot(), commands.append, baseline=[0.001, -0.001]
        )
        self.assertEqual(result["fault"], "stationary_odom_baseline_unavailable")
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertEqual(result["physical_motion_status"], "BASELINE_UNAVAILABLE")

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
        motion_ticks = 0

        def publish(value):
            nonlocal motion_ticks
            commands.append(value)
            if value == 0.2:
                motion_ticks += 1
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
                sample["pose"]["x"] = motion_ticks * 0.002

        result = self.run_protocol(lambda: sample, publish, baseline=[0.002, -0.001, 0.001])
        self.assertEqual(result["status"], "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED")
        self.assertEqual(result["control_handshake_status"], "CONTROL_HANDSHAKE_CONFIRMED")
        self.assertEqual(result["physical_motion_status"], "PHYSICAL_MOTION_CONFIRMED")
        self.assertGreater(result["nonzero_publish_calls_returned"], 0)
        self.assertTrue(result["source_selected_by_get_service"])
        self.assertTrue(result["source_selected_by_common_state"])
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["publish_call_return_span_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], ZERO_STOP_MIN_MESSAGES)
        self.assertLess(result["zero_stop"]["max_gap_between_publish_call_returns_s"], 1.0)
        self.assertFalse(result["zero_stop"]["mc_receipt_confirmed"])
        self.assertFalse(result["zero_stop"]["mc_stop_confirmed"])
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
        self.assertLessEqual(result["nonzero_publish_calls_returned"], 1)
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], 50)

    def test_fall_during_activation_aborts_and_zero_stops(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["pose"]["z"] = 0.51

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["fault"], "fall_root_height")
        self.assertLessEqual(result["nonzero_publish_calls_returned"], 1)
        self.assertGreaterEqual(result["zero_stop"]["publish_call_return_span_s"], 1.0)

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
        self.assertLessEqual(result["nonzero_publish_calls_returned"], 1)

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
        self.assertEqual(result["status"], "ZERO_STOP_PUBLISH_EXCEPTION")
        self.assertTrue(result["zero_stop"]["publish_exceptions"])
        self.assertFalse(result["zero_stop"]["mc_receipt_confirmed"])

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
        self.assertGreaterEqual(result["post_stop_emergency_zero"]["loop_duration_s"], 1.0)
        self.assertFalse(result["post_stop_emergency_zero"]["mc_stop_confirmed"])

    def test_partial_post_stop_observation_does_not_confirm_mc_stop(self) -> None:
        sample = self.snapshot()
        stop_started = False

        def publish(value):
            nonlocal stop_started
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
            elif value == 0.0:
                stop_started = True

        def read():
            if stop_started:
                raise RuntimeError("injected post-stop telemetry loss")
            return sample

        result = self.run_protocol(read, publish, baseline=[0.001, -0.001, 0.001])
        self.assertEqual(result["status"], "POST_STOP_SAFETY_FAILED")
        self.assertEqual(result["post_stop_odometry"]["kinematic_status"], "POST_STOP_ODOMETRY_UNOBSERVED")
        self.assertFalse(result["zero_stop"]["mc_receipt_confirmed"])
        self.assertFalse(result["zero_stop"]["mc_stop_confirmed"])

    def test_process_local_timer_watchdog_fires_and_starts_full_zero_window(self) -> None:
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
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        self.assertEqual(
            result["watchdog_scope_limitation"],
            "threading.Timer runs in this Python process and can request zero while the process is alive; it cannot publish after this process crashes or is killed.",
        )
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
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], 49)

    def test_zero_stop_window_precedes_stalled_post_stop_sampling(self) -> None:
        sample = self.snapshot()
        clock = FakeClock()
        commands = []
        timers = []
        first_zero_watchdog_cancel_state = []
        zero_seen = threading.Event()
        post_stop_sample_entered = threading.Event()
        release_stalled_sample = threading.Event()
        injected_failure = False
        outcome = {}

        def timer_factory(timeout, callback):
            timer = FakeTimer(timeout, callback)
            timers.append(timer)
            return timer

        def publish(value):
            commands.append(value)
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = 3
            elif value == 0.0:
                if not zero_seen.is_set():
                    first_zero_watchdog_cancel_state.append(timers[0].cancelled)
                    zero_seen.set()

        def read():
            nonlocal injected_failure
            if zero_seen.is_set() and not injected_failure:
                post_stop_sample_entered.set()
                if not release_stalled_sample.wait(timeout=2.0):
                    raise AssertionError("test did not release the stalled post-stop sample")
                injected_failure = True
                raise RuntimeError("injected delayed post-stop telemetry failure")
            return sample

        def run():
            outcome["result"] = self.run_protocol(
                read,
                publish,
                timer_factory=timer_factory,
                fake_clock=clock,
            )

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.assertTrue(post_stop_sample_entered.wait(timeout=2.0))
        zero_commands_at_sample_entry = sum(value == 0.0 for value in commands)
        self.assertGreaterEqual(zero_commands_at_sample_entry, ZERO_STOP_MIN_MESSAGES)
        self.assertEqual(first_zero_watchdog_cancel_state, [False])
        release_stalled_sample.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive(), "bounded offline protocol did not finish")

        result = outcome["result"]
        self.assertEqual(result["status"], "POST_STOP_SAFETY_FAILED")
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        first_observation = result["post_stop_first_observation"]
        self.assertEqual(first_observation["timing_relation"], "sampled_after_zero_stop_window")
        self.assertTrue(first_observation["zero_window_complete_before_sample"])
        self.assertEqual(
            first_observation["safety_fault"],
            "post_stop_first_observation_exception:RuntimeError:injected delayed post-stop telemetry failure",
        )

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
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], 49)

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
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)

    def test_unauthorized_call_emits_no_command(self) -> None:
        commands = []
        result = run_first_motion_window(
            sample=lambda: self.snapshot(),
            publish=commands.append,
            registration={"success": True},
            set_action={"accepted_by_service": True},
            joint_limits=LIMITS,
            baseline_forward_displacements_m=[0.0, 0.0, 0.0],
            authorized=False,
        )
        self.assertEqual(result["status"], "NOT_AUTHORIZED")
        self.assertEqual(commands, [])

    def test_zero_stop_rejects_short_call_return_span_without_claiming_mc_receipt(self) -> None:
        clock = FakeClock()

        def slow_zero(command):
            if command == 0.0:
                with clock.lock:
                    clock.value += 0.25

        result = self.run_protocol(
            lambda: self.snapshot(), slow_zero, fake_clock=clock
        )
        self.assertEqual(result["status"], "ZERO_STOP_CALL_RETURN_SPAN_TOO_SHORT")
        self.assertLess(result["zero_stop"]["publish_call_return_span_s"], SOURCE_TIMEOUT_MS / 1000.0)
        self.assertFalse(result["zero_stop"]["mc_receipt_confirmed"])
        self.assertFalse(result["zero_stop"]["mc_stop_confirmed"])


if __name__ == "__main__":
    unittest.main()
