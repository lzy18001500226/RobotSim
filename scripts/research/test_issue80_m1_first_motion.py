"""Deterministic offline tests for Issue 80 first-motion safety and stop handling."""

from __future__ import annotations

import threading
import unittest
import json
import hashlib
import tempfile
import time
from types import SimpleNamespace
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
    MC_FSM_ACTIVE_STATES,
    MC_FSM_INTERFACE_SOURCE_PATH,
    MC_FSM_INTERFACE_SOURCE_SHA256,
    MC_FSM_SAFE,
    MC_FSM_STABLE,
    measure_post_stop_forward_tail,
    evaluate_physical_stop_gate,
    run_angular_turn_window,
    verify_mc_fsm_runtime_contract,
    MC_FSM_RUNTIME_LIBRARY_SHA256,
    FORWARD_COMMAND_M_S,
    PULSE_LIMIT_S,
    MAX_AUTHORIZED_PULSE_S,
    smoothstep_velocity,
    smoothstep_deceleration_velocity,
    smooth_turn_velocity,
    classify_turning_rotation,
    classify_turn_planar_motion,
    TURN_FORWARD_COMMAND_M_S,
    TURN_FORWARD_WARMUP_S,
    TURN_MAX_COMMAND_WINDOW_S,
    TURN_PULSE_S,
    TURN_MAX_SUSTAINED_YAW_RATE_RAD_S,
    TURN_YAW_RATE_WINDOW_S,
    SustainedYawRateWindow,
    startup_then_cruise_velocity,
    joint_command_array_snapshot,
    float64_multi_array_snapshot,
    mc_rl_debug_snapshot,
    PUBLISH_HZ,
    SOURCE_TIMEOUT_MS,
    VELOCITY_SOURCE_NAME,
    McFsmStateCode,
    ZERO_STOP_MIN_S,
    ZERO_STOP_MIN_MESSAGES,
    VerifiedMcRuntimeIdentity,
    capture_dds_qos_override_environment,
    classify_source_registration,
    classify_physical_motion,
    derive_stationary_odom_baseline,
    load_verified_mc_runtime_identity,
    load_leg_joint_limits,
    load_mc_input_source_timeouts,
    load_verified_mc_velocity_subscriber_provenance,
    mc_subscriber_history_depth_evidence,
    run_first_motion_window,
    snapshot_fault,
    validate_first_motion_authorization,
    validate_local_sim_publishers,
    validate_runtime_scope as _validate_runtime_scope,
)
from issue80_m1_mode_logic import classify_set_mc_action_response


LIMITS = {
    "left_hip_pitch_joint": {
        "lower_position": -2.7,
        "upper_position": 2.5,
        "max_velocity": 11.936,
    }
}


TEST_MC_IDENTITY = VerifiedMcRuntimeIdentity(
    configured_node_name="mc_ros2_node",
    process_pid=245,
    process_executable="/workspace/raicom2026/mc/bin/aima-mc-app-main",
    mc_config_sha256="a" * 64,
    startup_log_sha256="b" * 64,
)


class McDebugFloatArraySnapshotTests(unittest.TestCase):
    def test_preserves_layout_labels_values_and_receive_time_without_semantic_aliases(self):
        message = SimpleNamespace(
            layout=SimpleNamespace(
                data_offset=1,
                dim=[SimpleNamespace(label="raw_axis", size=2, stride=3)],
            ),
            data=[0.25, -0.5, 1.0],
        )

        result = float64_multi_array_snapshot(message, 12.5)

        self.assertEqual(result["received_monotonic_s"], 12.5)
        self.assertEqual(result["layout"]["data_offset"], 1)
        self.assertEqual(
            result["layout"]["dimensions"],
            [{"label": "raw_axis", "size": 2, "stride": 3}],
        )
        self.assertEqual(result["data"], [0.25, -0.5, 1.0])
        self.assertEqual(result["nonfinite_indexes"], [])
        self.assertIn("do not assign field semantics", result["interpretation"])

    def test_nonfinite_debug_values_remain_json_safe_and_are_indexed(self):
        message = SimpleNamespace(layout=SimpleNamespace(dim=[]), data=[float("nan"), 2.0])

        result = float64_multi_array_snapshot(message)

        self.assertEqual(result["data"], [None, 2.0])
        self.assertEqual(result["nonfinite_indexes"], [0])

    def test_absent_optional_debug_message_returns_none(self):
        self.assertIsNone(float64_multi_array_snapshot(None))


class McRlDebugSnapshotTests(unittest.TestCase):
    def test_preserves_version_matched_header_observation_and_action_vectors(self):
        message = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=12, nanosec=34),
                frame_id="mc",
                sequence=56,
                meas_stamp=SimpleNamespace(sec=11, nanosec=99),
            ),
            obs_vec=[0.105, 0.0, 0.0],
            action_vec=[0.1, -0.2],
        )

        result = mc_rl_debug_snapshot(message, 20.5)

        self.assertEqual(result["received_monotonic_s"], 20.5)
        self.assertEqual(result["header"]["stamp"], {"sec": 12, "nanosec": 34})
        self.assertEqual(result["header"]["meas_stamp"], {"sec": 11, "nanosec": 99})
        self.assertEqual(result["header"]["frame_id"], "mc")
        self.assertEqual(result["header"]["sequence"], 56)
        self.assertEqual(result["obs_vec"], [0.105, 0.0, 0.0])
        self.assertEqual(result["action_vec"], [0.1, -0.2])
        self.assertEqual(result["nonfinite_obs_indexes"], [])
        self.assertEqual(result["nonfinite_action_indexes"], [])

    def test_nonfinite_policy_values_are_json_safe_and_indexed(self):
        message = SimpleNamespace(header=None, obs_vec=[float("nan"), 0.1], action_vec=[float("inf")])

        result = mc_rl_debug_snapshot(message)

        self.assertEqual(result["obs_vec"], [None, 0.1])
        self.assertEqual(result["action_vec"], [None])
        self.assertEqual(result["nonfinite_obs_indexes"], [0])
        self.assertEqual(result["nonfinite_action_indexes"], [0])

    def test_absent_mcrldebug_message_returns_none(self):
        self.assertIsNone(mc_rl_debug_snapshot(None))


VALID_MC_SUBSCRIBER_PROVENANCE = {
    "subscriber_name": "locomotion_velocity",
    "node_name": "mc_ros2_node245",
    "topic_name": "/aima/mc/locomotion/velocity",
    "message_type": "aimdk_msgs/msg/McLocomotionVelocity",
    "subscriber_enabled": True,
    "ros2_route_enabled": True,
    "startup_subscriber_creation_confirmed": True,
    "startup_type_registration_confirmed": True,
    "config_namespace_option": None,
    "qos_override_present": False,
    "configured_history_depth": "not specified in audited subscriber configuration",
    "actual_reader_history_depth": "UNVERIFIED",
    "mc_config_sha256": TEST_MC_IDENTITY.mc_config_sha256,
    "startup_log_sha256": TEST_MC_IDENTITY.startup_log_sha256,
    "evidence_basis": "offline test fixture",
}

VALID_MC_SUBSCRIBER_ENDPOINT = {
    "node_name": "mc_ros2_node245",
    "node_namespace": "/",
    "queried_topic_name": "/aima/mc/locomotion/velocity",
    "topic_type": "aimdk_msgs/msg/McLocomotionVelocity",
}

NO_DDS_QOS_OVERRIDES = {
    "FASTDDS_DEFAULT_PROFILES_FILE": "unset",
    "FASTRTPS_DEFAULT_PROFILES_FILE": "unset",
    "RMW_FASTRTPS_USE_QOS_FROM_XML": "unset",
}


def validate_runtime_scope(**kwargs):
    """Add the verified base fixture; tests can override fields to exercise failures."""
    evidence = {
        "ros_distro": "humble",
        "subscriber_endpoint_metadata": [dict(VALID_MC_SUBSCRIBER_ENDPOINT)],
        "mc_subscriber_provenance": dict(VALID_MC_SUBSCRIBER_PROVENANCE),
        "dds_qos_override_environment": dict(NO_DDS_QOS_OVERRIDES),
    }
    evidence.update(kwargs)
    return _validate_runtime_scope(**evidence)


VALID_LOCAL_PUBLISHER_EVIDENCE = {
    "publisher_local_identity": {
        "node_name": "issue80_m1_focused_diagnostic",
        "topic_name": "/aima/mc/locomotion/velocity",
        "message_type": "McLocomotionVelocity",
        "is_expected_message_type": True,
        "is_owned_publisher": True,
        "node_topic_publisher_count": 1,
    },
    "publisher_configured_qos": {
        "history": "keep_last",
        "depth": COMMAND_QOS_DEPTH,
        "reliability": COMMAND_QOS_RELIABILITY,
        "durability": COMMAND_QOS_DURABILITY,
    },
    "publisher_api_qos": {
        "history": "keep_last",
        "depth": COMMAND_QOS_DEPTH,
        "reliability": COMMAND_QOS_RELIABILITY,
        "durability": COMMAND_QOS_DURABILITY,
    },
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
            "fsm_state": MC_FSM_STABLE,
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
        forward_command_m_s=0.2,
        pulse_duration_s=1.0,
        max_pulse_duration_s=1.0,
        ramp_start_speed_m_s=None,
        ramp_duration_s=0.0,
        deceleration_duration_s=0.0,
        deceleration_end_speed_m_s=0.0,
        startup_speed_m_s=None,
        startup_transition_displacement_m=None,
        startup_transition_timeout_s=0.0,
        max_observed_planar_speed_m_s=None,
        zero_until_stand_timeout_s=0.0,
        after_zero_stop=None,
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
            forward_command_m_s=forward_command_m_s,
            pulse_duration_s=pulse_duration_s,
            max_pulse_duration_s=max_pulse_duration_s,
            ramp_start_speed_m_s=ramp_start_speed_m_s,
            ramp_duration_s=ramp_duration_s,
            deceleration_duration_s=deceleration_duration_s,
            deceleration_end_speed_m_s=deceleration_end_speed_m_s,
            startup_speed_m_s=startup_speed_m_s,
            startup_transition_displacement_m=startup_transition_displacement_m,
            startup_transition_timeout_s=startup_transition_timeout_s,
            shutdown_event=shutdown_event,
            state_timeout_s=0.5,
            min_root_z=0.52,
            max_tilt_deg=15.0,
            max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
            zero_until_stand_timeout_s=zero_until_stand_timeout_s,
            after_zero_stop=after_zero_stop,
            clock=clock.now,
            sleep=clock.sleep,
            timer_factory=timer_factory,
        )

    def test_reduced_probe_speed_keeps_hard_cap_and_safety_gates(self):
        commands = []
        result = self.run_protocol(self.snapshot, commands.append, forward_command_m_s=0.1)
        self.assertIn(0.1, commands)
        self.assertNotIn(0.2, commands)
        self.assertEqual(result["command_forward_m_s"], 0.1)
        for invalid in (0.0, -0.1, 0.201, float("nan")):
            commands = []
            result = self.run_protocol(self.snapshot, commands.append, forward_command_m_s=invalid)
            self.assertEqual(result["fault"], "forward_command_outside_authorized_cap")
            self.assertEqual(result["nonzero_publish_calls_returned"], 0)
            self.assertTrue(all(v == 0.0 for v in commands))

    def test_shorter_pulse_cannot_extend_original_deadline(self):
        commands = []
        result = self.run_protocol(self.snapshot, commands.append, pulse_duration_s=0.6)
        self.assertLessEqual(result["last_nonzero_command_offset_s"], 0.6)
        self.assertEqual(result["pulse_duration_s_requested"], 0.6)
        self.assertEqual(result["pulse_limit_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        for invalid in (0.0, -0.1, 1.001, float("nan")):
            result = self.run_protocol(self.snapshot, lambda v: None, pulse_duration_s=invalid)
            self.assertEqual(result["fault"], "pulse_duration_outside_authorized_cap")
            self.assertEqual(result["nonzero_publish_calls_returned"], 0)

    def test_measured_root_planar_speed_guard_stops_before_overbound_motion(self):
        initial = self.snapshot(pose={
            "x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "vx": 0.0, "vy": 0.0,
        })
        fast = self.snapshot(pose={
            "x": 0.01, "y": 0.0, "z": 0.64, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "vx": 0.201, "vy": 0.0,
        })
        rows = iter((initial, fast))
        commands = []

        def sample():
            return next(rows, fast)

        result = self.run_protocol(
            sample, commands.append,
            forward_command_m_s=0.112,
            max_observed_planar_speed_m_s=0.2,
        )

        self.assertEqual(result["fault"], "measured_root_planar_speed_limit:0.201000>0.200000")
        self.assertEqual(result["nonzero_publish_calls_returned"], 0)
        self.assertTrue(commands)
        self.assertTrue(all(command == 0.0 for command in commands))

    def test_zero_stream_continues_until_stand_mode_is_actually_applied(self):
        clock = FakeClock()
        transition_at = 1.20

        def sample():
            if clock.now() < 0.01:
                return self.snapshot(
                    get_mode_status_value=100,
                    common_mode_status_value=100,
                )
            if clock.now() < transition_at:
                return self.snapshot(
                    get_mode_desc="LOCOMOTION_DEFAULT",
                    common_mode_desc="LOCOMOTION_DEFAULT",
                    get_mode_status_value=100,
                    common_mode_status_value=100,
                    fsm_state=5,
                    get_source_name=VELOCITY_SOURCE_NAME,
                    common_source_name=VELOCITY_SOURCE_NAME,
                )
            return self.snapshot(
                get_mode_desc="STAND_DEFAULT",
                common_mode_desc="STAND_DEFAULT",
                get_mode_status_value=100,
                common_mode_status_value=100,
                fsm_state=MC_FSM_STABLE,
                body_status=1,
            )

        commands = []
        result = self.run_protocol(
            sample,
            commands.append,
            fake_clock=clock,
            forward_command_m_s=0.112,
            pulse_duration_s=0.1,
            max_pulse_duration_s=0.1,
            zero_until_stand_timeout_s=3.5,
            after_zero_stop=lambda: {"success": True},
        )

        confirmation = result["stop_transition_confirmation"]
        extension = result["post_stop_zero_extension"]
        self.assertEqual(confirmation["status"], "STAND_READBACK_CONFIRMED")
        self.assertTrue(confirmation["stand_confirmed"])
        self.assertGreater(extension["zero_publish_calls_returned"], 0)
        self.assertEqual(extension["status"], "STAND_READBACK_CONFIRMED")

    def test_zero_stream_timeout_is_reported_and_zero_continues(self):
        commands = []
        loco = self.snapshot(
            get_mode_desc="LOCOMOTION_DEFAULT",
            common_mode_desc="LOCOMOTION_DEFAULT",
            get_mode_status_value=100,
            common_mode_status_value=100,
            fsm_state=5,
            get_source_name=VELOCITY_SOURCE_NAME,
            common_source_name=VELOCITY_SOURCE_NAME,
        )
        result = self.run_protocol(
            lambda: loco,
            commands.append,
            forward_command_m_s=0.112,
            pulse_duration_s=0.1,
            max_pulse_duration_s=0.1,
            zero_until_stand_timeout_s=0.2,
        )

        self.assertEqual(result["stop_transition_confirmation"]["status"], "AUTO_STAND_READBACK_TIMEOUT")
        self.assertIn("post_stop_auto_stand_transition_timeout", result["post_stop_fault"])
        self.assertIsNone(result["after_zero_stop_action"])
        self.assertEqual(result["post_stop_zero_extension"]["status"], "AUTO_STAND_READBACK_TIMEOUT_ZERO_CONTINUES")
        self.assertGreater(result["post_stop_zero_extension"]["zero_publish_calls_returned"], 1)
        self.assertFalse(result["stop_transition_confirmation"]["stand_confirmed"])

        commands = []
        invalid_cap = self.run_protocol(
            self.snapshot, commands.append,
            forward_command_m_s=0.112,
            max_observed_planar_speed_m_s=0.201,
        )
        self.assertEqual(invalid_cap["fault"], "observed_planar_speed_gate_outside_authorized_cap")
        self.assertEqual(invalid_cap["nonzero_publish_calls_returned"], 0)

    def test_stopping_tail_separates_braking_from_continued_motion(self):
        def row(t, x):
            return {"phase": "post-stop", "sample_monotonic_s": t,
                    "pose": {"x": x, "y": 0.0, "yaw": 0.0},
                    "safety_fault": None, "get_mode_desc": "STAND_DEFAULT",
                    "common_mode_desc": "STAND_DEFAULT", "fsm_state": MC_FSM_STABLE}
        samples = [row(0.0, 0.0), row(0.5, 0.1), row(1.0, 0.2),
                   row(1.5, 0.2), row(2.0, 0.2), row(2.5, 0.2)]
        result = measure_post_stop_forward_tail(samples, 0.001)
        self.assertEqual(result["status"], "WITHIN_BASELINE_ENVELOPE")
        self.assertEqual(result["max_forward_excursion_m"], 0.0)
        samples[-2]["pose"]["x"] += 0.01  # cancellation must not fake stopping
        self.assertEqual(measure_post_stop_forward_tail(samples, 0.001)["status"],
                         "ABOVE_BASELINE_ENVELOPE")
        samples[-2]["pose"]["x"] -= 0.01
        samples[-1]["safety_fault"] = "stale_or_missing_state:imu"
        self.assertNotEqual(measure_post_stop_forward_tail(samples, 0.001)["status"],
                            "WITHIN_BASELINE_ENVELOPE")
        self.assertEqual(measure_post_stop_forward_tail(samples[-1:], 0.001)["status"],
                         "INCOMPLETE_FINAL_WINDOW")

    def test_post_zero_stop_action_runs_only_after_handshake_and_zero_window(self):
        from issue80_m1_first_motion import run_first_motion_window
        clock = FakeClock()
        sample = self.snapshot()
        seen = []
        def publish(value):
            if value == 0.2:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
            seen.append((value, clock.now()))
        result = run_first_motion_window(
            sample=lambda: sample, publish=publish,
            registration={"success": True, "selected_action": "ADD"},
            set_action={"accepted_by_service": True, "header_code": 0, "status_value": 1},
            joint_limits=LIMITS, baseline_forward_displacements_m=[0.0, 0.0, 0.0],
            authorized=True, clock=clock.now, sleep=clock.sleep,
            state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0,
            after_zero_stop=lambda: {"success": True, "mode": "STAND_DEFAULT"},
        )
        action = result["after_zero_stop_action"]
        self.assertTrue(action["success"])
        self.assertEqual(action["mode"], "STAND_DEFAULT")
        self.assertTrue(any(value == 0.0 for value, _ in seen))
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)

    def test_reviewed_ten_second_pulse_remains_continuous_and_stops_at_deadline(self):
        commands = []
        result = self.run_protocol(
            self.snapshot, commands.append,
            pulse_duration_s=10.0,
            max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
        )
        nonzero = [value for value in commands if value > 0.0]
        self.assertEqual(len(nonzero), 500)
        self.assertEqual(result["pulse_duration_s_requested"], 10.0)
        self.assertEqual(result["pulse_limit_s"], 10.0)
        self.assertLessEqual(result["last_nonzero_command_offset_s"], 10.0)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)
        self.assertTrue(any(value == 0.0 for value in commands))

    def test_smoothstep_velocity_ramp_has_bounded_endpoints_and_slope(self):
        values=[smoothstep_velocity(0.105,0.15,t,2.0) for t in (0.0,0.5,1.0,1.5,2.0,3.0)]
        self.assertAlmostEqual(values[0],0.105)
        self.assertAlmostEqual(values[-2],0.15)
        self.assertAlmostEqual(values[-1],0.15)
        self.assertEqual(values,sorted(values))
        self.assertAlmostEqual(values[2],0.1275)
        with self.assertRaisesRegex(ValueError,"under the command cap"):
            smoothstep_velocity(0.105,0.21,1.0,2.0)

    def test_smoothstep_deceleration_reaches_zero_monotonically_within_one_second(self):
        values = [
            smoothstep_deceleration_velocity(0.14, t, 1.0)
            for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1)
        ]
        self.assertEqual(values[0], 0.14)
        self.assertEqual(values[2], 0.07)
        self.assertEqual(values[4:], [0.0, 0.0])
        self.assertTrue(all(0.0 <= right <= left <= 0.14 for left, right in zip(values, values[1:])))
        with self.assertRaisesRegex(ValueError, "in \(0, 1\] seconds"):
            smoothstep_deceleration_velocity(0.14, 0.5, 1.01)

    def test_smoothstep_deceleration_can_preserve_a_positive_minimum_command(self):
        values = [
            smoothstep_deceleration_velocity(0.14, t, 1.0, 0.112)
            for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1)
        ]
        self.assertEqual(values[0], 0.14)
        self.assertAlmostEqual(values[2], 0.126)
        self.assertEqual(values[4:], [0.112, 0.112])
        self.assertTrue(all(0.112 <= right <= left <= 0.14 for left, right in zip(values, values[1:])))

    def test_run_window_decelerates_only_in_final_second_then_uses_zero_stop(self):
        clock = FakeClock()
        commands = []
        result = self.run_protocol(
            self.snapshot,
            lambda value: commands.append((clock.now(), value)),
            fake_clock=clock,
            baseline=[0.001, 0.001, 0.001],
            forward_command_m_s=0.14,
            pulse_duration_s=2.0,
            max_pulse_duration_s=2.0,
            deceleration_duration_s=1.0,
        )
        pulse = [(t, value) for t, value in commands if value > 0.0]
        self.assertEqual(len(pulse), 100)
        self.assertTrue(all(abs(value - 0.14) < 1e-12 for _, value in pulse[:50]))
        decelerating = [value for _, value in pulse[50:]]
        self.assertTrue(all(right <= left for left, right in zip(decelerating, decelerating[1:])))
        self.assertGreater(decelerating[0], 0.13)
        self.assertLess(decelerating[-1], 0.001)
        self.assertEqual(result["velocity_profile"]["shape"], "constant-then-smoothstep-deceleration")
        self.assertEqual(result["velocity_profile"]["deceleration_start_s"], 1.0)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)

    def test_bounded_pulse_can_decelerate_to_threshold_margin(self):
        clock = FakeClock()
        commands = []
        result = self.run_protocol(
            self.snapshot,
            lambda value: commands.append((clock.now(), value)),
            fake_clock=clock,
            baseline=[0.001, 0.001, 0.001],
            forward_command_m_s=0.14,
            pulse_duration_s=2.0,
            max_pulse_duration_s=2.0,
            deceleration_duration_s=1.0,
            deceleration_end_speed_m_s=0.112,
        )
        pulse = [(t, value) for t, value in commands if value > 0.0]
        self.assertEqual(len(pulse), 100)
        self.assertTrue(all(abs(value - 0.14) < 1e-12 for _, value in pulse[:50]))
        decelerating = [value for _, value in pulse[50:]]
        self.assertTrue(all(right <= left for left, right in zip(decelerating, decelerating[1:])))
        self.assertGreater(decelerating[0], 0.13)
        self.assertGreaterEqual(min(decelerating), 0.112)
        self.assertAlmostEqual(decelerating[-1], 0.112, places=4)
        self.assertEqual(result["velocity_profile"]["shape"], "constant-then-smoothstep-deceleration")
        self.assertEqual(result["velocity_profile"]["deceleration_start_s"], 1.0)
        self.assertEqual(result["velocity_profile"]["end_speed_m_s"], 0.112)
        self.assertGreaterEqual(result["zero_stop"]["loop_duration_s"], 1.0)

    def test_motion_triggered_startup_uses_cruise_only_after_measured_displacement(self):
        self.assertEqual(startup_then_cruise_velocity(0.12, 0.08, 0.0199, 0.02), 0.12)
        self.assertEqual(startup_then_cruise_velocity(0.12, 0.08, 0.02, 0.02), 0.08)
        with self.assertRaisesRegex(ValueError, "under the command cap"):
            startup_then_cruise_velocity(0.21, 0.08, 0.02, 0.02)

    def test_joint_command_snapshot_preserves_mc_targets_and_timing(self):
        result = joint_command_array_snapshot({
            "header": {"stamp": {"sec": 123, "nanosec": 456}},
            "joints": [{
                "name": "left_knee_joint", "position": 0.5, "velocity": 1.25,
                "effort": 2.5, "stiffness": 100.0, "damping": 5.0,
            }],
        }, received_monotonic_s=10.25)
        self.assertEqual(result["header_stamp"], {"sec": 123, "nanosec": 456})
        self.assertEqual(result["received_monotonic_s"], 10.25)
        self.assertEqual(result["joints"][0]["name"], "left_knee_joint")
        self.assertEqual(result["joints"][0]["velocity"], 1.25)
        self.assertEqual(result["joints"][0]["stiffness"], 100.0)

    def test_motion_triggered_startup_transitions_once_and_keeps_hard_deadline(self):
        clock=FakeClock()
        state=self.snapshot()
        commands=[]
        def sample():
            state["pose"]["x"] = min(clock.now() * 0.12, 0.04)
            return state
        def publish(value):
            commands.append((clock.now(),value))
            if value > 0.0:
                state["get_source_name"]=VELOCITY_SOURCE_NAME
                state["common_source_name"]=VELOCITY_SOURCE_NAME
            if value > 0.1:
                state["get_mode_desc"]="LOCOMOTION_DEFAULT"
                state["common_mode_desc"]="LOCOMOTION_DEFAULT"
                state["fsm_state"]=int(McFsmStateCode.MOVING)
        result=self.run_protocol(
            sample,publish,fake_clock=clock,
            forward_command_m_s=0.08,
            pulse_duration_s=10.0,max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
            startup_speed_m_s=0.12,
            startup_transition_displacement_m=0.02,
            startup_transition_timeout_s=0.5,
        )
        nonzero=[value for _,value in commands if value>0.0]
        self.assertEqual(len(nonzero),500)
        self.assertEqual(nonzero[0],0.12)
        self.assertIn(0.08,nonzero)
        self.assertTrue(result["velocity_profile"]["transition_observed"])
        self.assertGreaterEqual(result["velocity_profile"]["transition_displacement_observed_m"],0.02)
        self.assertLessEqual(result["last_nonzero_command_offset_s"],10.0)
        self.assertTrue(any(value==0.0 for _,value in commands))

    def test_motion_triggered_startup_stops_if_displacement_gate_times_out(self):
        clock=FakeClock()
        state=self.snapshot()
        commands=[]
        def publish(value):
            commands.append((clock.now(),value))
            if value > 0.1:
                state["get_source_name"]=VELOCITY_SOURCE_NAME
                state["common_source_name"]=VELOCITY_SOURCE_NAME
                state["get_mode_desc"]="LOCOMOTION_DEFAULT"
                state["common_mode_desc"]="LOCOMOTION_DEFAULT"
                state["fsm_state"]=int(McFsmStateCode.MOVING)
        result=self.run_protocol(
            lambda:state,publish,fake_clock=clock,
            forward_command_m_s=0.08,
            pulse_duration_s=10.0,max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
            startup_speed_m_s=0.12,
            startup_transition_displacement_m=0.02,
            startup_transition_timeout_s=0.05,
        )
        self.assertEqual(result["fault"],"startup_displacement_not_reached_before_timeout")
        self.assertLess(result["nonzero_publish_calls_returned"],10)
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"],50)

    def test_ten_second_ramp_publishes_fifty_hz_then_stops_at_hard_deadline(self):
        clock=FakeClock()
        state=self.snapshot()
        commands=[]
        def publish(value):
            commands.append((clock.now(),value))
            if value > 0.1:
                state["get_source_name"]=VELOCITY_SOURCE_NAME
                state["common_source_name"]=VELOCITY_SOURCE_NAME
                state["get_mode_desc"]="LOCOMOTION_DEFAULT"
                state["common_mode_desc"]="LOCOMOTION_DEFAULT"
                state["fsm_state"]=int(McFsmStateCode.MOVING)
        result=self.run_protocol(
            lambda:state,publish,fake_clock=clock,
            forward_command_m_s=0.15,
            pulse_duration_s=10.0,max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
            ramp_start_speed_m_s=0.105,ramp_duration_s=2.0,
        )
        active=[value for _,value in commands if value>0]
        self.assertEqual(len(active),500)
        self.assertAlmostEqual(active[0],0.105)
        self.assertAlmostEqual(active[100],0.15,places=5)
        self.assertTrue(all(a<=b for a,b in zip(active,active[1:])))
        self.assertLessEqual(result["last_nonzero_command_offset_s"],10.0)
        self.assertTrue(any(value==0.0 for _,value in commands))

    def test_ramp_inputs_fail_closed(self):
        for start,duration in ((None,2.0),(0.0,2.0),(0.16,2.0),(0.105,11.0)):
            commands=[]
            result=self.run_protocol(
                self.snapshot,commands.append,forward_command_m_s=0.15,pulse_duration_s=10.0,
                max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
                ramp_start_speed_m_s=start,ramp_duration_s=duration,
            )
            self.assertEqual(result["nonzero_publish_calls_returned"],0)
            self.assertTrue(result["fault"] in {
                "velocity_ramp_duration_outside_pulse",
                "velocity_ramp_start_outside_command_range",
            })

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
        self.assertEqual(MAX_AUTHORIZED_PULSE_S, 10.0)
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

    def test_installed_mc_node_suffix_is_bound_to_config_and_live_process_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "mc_sim.yaml"
            config_path.write_text(json.dumps({"aimrt": {"plugin": {"plugins": [
                {"name": "ros2_plugin", "options": {"node_name": "mc_ros2_node"}},
            ]}}}), encoding="utf-8")
            startup_log_path = root / "mc-console.log"
            startup_log_path.write_text(
                "        node_name: mc_ros2_node\n"
                "| Executable Path | /workspace/raicom2026/mc/bin/aima-mc-app-main |\n"
                "| Process PID     | 245                                           |\n",
                encoding="utf-8",
            )
            fake_proc = root / "proc" / "245"
            fake_proc.mkdir(parents=True)
            fake_executable = root / "bin" / "aima-mc-app-main"
            fake_executable.parent.mkdir()
            fake_executable.write_bytes(b"offline process identity fixture")
            (fake_proc / "exe").symlink_to(fake_executable)
            (fake_proc / "cmdline").write_bytes(
                f"./aima-mc-app-main\0--cfg_file_path={config_path}\0".encode()
            )

            identity = load_verified_mc_runtime_identity(
                config_path, startup_log_path, proc_root=root / "proc"
            )

        self.assertEqual(identity.configured_node_name, "mc_ros2_node")
        self.assertEqual(identity.process_pid, 245)
        self.assertEqual(identity.node_name, "mc_ros2_node245")
        self.assertEqual(identity.process_executable, str(fake_executable))
        self.assertEqual(len(identity.mc_config_sha256), 64)
        self.assertEqual(len(identity.startup_log_sha256), 64)

    def test_mc_identity_loader_rejects_pid_or_config_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "mc_sim.yaml"
            config_path.write_text(json.dumps({"aimrt": {"plugin": {"plugins": [
                {"name": "ros2_plugin", "options": {"node_name": "mc_ros2_node"}},
            ]}}}), encoding="utf-8")
            startup_log_path = root / "mc-console.log"
            startup_log_path.write_text(
                "        node_name: mc_ros2_node\n"
                "| Executable Path | /runtime/aima-mc-app-main |\n"
                "| Process PID     | 245                         |\n",
                encoding="utf-8",
            )
            fake_proc = root / "proc" / "245"
            fake_proc.mkdir(parents=True)
            fake_executable = root / "bin" / "aima-mc-app-main"
            fake_executable.parent.mkdir()
            fake_executable.write_bytes(b"offline process identity fixture")
            (fake_proc / "exe").symlink_to(fake_executable)
            (fake_proc / "cmdline").write_bytes(
                b"./aima-mc-app-main\0--cfg_file_path=/different/mc_sim.yaml\0"
            )
            with self.assertRaisesRegex(ValueError, "verified MC config"):
                load_verified_mc_runtime_identity(
                    config_path, startup_log_path, proc_root=root / "proc"
                )
            (fake_proc / "cmdline").write_bytes(
                f"./aima-mc-app-main\0--cfg_file_path={config_path}\0".encode()
            )
            startup_log_path.write_text(
                "        node_name: mc_ros2_node_spoof\n"
                "| Executable Path | /runtime/aima-mc-app-main |\n"
                "| Process PID     | 245                         |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "startup log does not report"):
                load_verified_mc_runtime_identity(
                    config_path, startup_log_path, proc_root=root / "proc"
                )

    def test_transport_scope_requires_bridge_domain_and_provenance_bound_mc_endpoint(self) -> None:
        self.assertIsNone(validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))
        self.assertIn("network", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="host",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))
        self.assertIn("publisher", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic", "other"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))

    def test_transport_scope_rejects_rmw_or_domain_mismatch(self) -> None:
        common = dict(
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
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
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="best_effort",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last", publisher_depth=10,
            subscriber_histories=["keep_last"], subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))

    def test_transport_scope_rejects_graph_durability_mismatches(self) -> None:
        common = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="unknown",
            publisher_depth=0,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        self.assertIn("publisher_durability", validate_runtime_scope(
            **{**common, "publisher_durability": "transient_local"}
        ))
        self.assertIn("subscriber_durability", validate_runtime_scope(
            **{**common, "subscriber_durabilities": ["transient_local"]}
        ))

    def test_transport_scope_rejects_name_lookalikes_and_unproven_or_extra_nodes(self) -> None:
        common = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        for nodes in (
            ["mc_ros2_node"],
            ["mc_ros2_node246"],
            ["mc_ros2_node245_extra"],
            ["prefix_mc_ros2_node245"],
            ["mc_ros2_node245", "external_controller"],
        ):
            with self.subTest(nodes=nodes):
                self.assertIn("subscriber", validate_runtime_scope(
                    subscriber_names=nodes,
                    **common,
                ))
        self.assertEqual(validate_runtime_scope(
            subscriber_names=["mc_ros2_node245"],
            **{**common, "mc_identity": None},
        ), "mc_runtime_identity_unverified")

    def test_transport_scope_allows_fastrtps_unknown_zero_with_verified_local_qos(self) -> None:
        self.assertIsNone(validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="unknown",
            publisher_depth=0,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))

    def test_verified_mc_subscriber_allows_only_exact_unknown_zero_discovery_gap(self) -> None:
        fault = validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_distro="humble",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["unknown"],
            subscriber_depths=[0],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        self.assertIsNone(fault)
        qos_evidence = mc_subscriber_history_depth_evidence(
            "unknown", 0, rmw_implementation="rmw_fastrtps_cpp", ros_distro="humble"
        )
        self.assertEqual(qos_evidence["classification"], "FASTDDS_HUMBLE_DISCOVERY_HISTORY_DEPTH_GAP")
        self.assertEqual(qos_evidence["actual_mc_datareader_history_depth"], "UNVERIFIED")
        self.assertFalse(qos_evidence["command_delivery_proven"])

    def test_verified_mc_subscriber_known_keep_last_depth_passes(self) -> None:
        self.assertIsNone(validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_distro="humble",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))

    def test_verified_mc_subscriber_rejects_incompatible_reliability_and_durability(self) -> None:
        base = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_distro="humble",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["unknown"],
            subscriber_depths=[0],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        self.assertIn("publisher_qos_not_reliable", validate_runtime_scope(
            **{**base, "publisher_reliability": "best_effort", "subscriber_reliabilities": ["reliable"]}
        ))
        self.assertIn("subscriber_durability", validate_runtime_scope(
            **{**base, "subscriber_durabilities": ["transient_local"]}
        ))

    def test_verified_mc_subscriber_rejects_wrong_topic_type_or_namespace(self) -> None:
        for field, value, expected in (
            ("queried_topic_name", "/wrong/topic", "endpoint_topic_mismatch"),
            ("topic_type", "std_msgs/msg/String", "message_type_mismatch"),
            ("node_namespace", "/spoof", "namespace_mismatch"),
        ):
            with self.subTest(field=field):
                endpoint = {**VALID_MC_SUBSCRIBER_ENDPOINT, field: value}
                self.assertIn(expected, validate_runtime_scope(
                    rmw_implementation="rmw_fastrtps_cpp",
                    ros_distro="humble",
                    ros_domain_id="20",
                    docker_network_mode="bridge",
                    publisher_names=["issue80_m1_focused_diagnostic"],
                    subscriber_names=["mc_ros2_node245"],
                    own_node_name="issue80_m1_focused_diagnostic",
                    publisher_reliability="reliable",
                    subscriber_reliabilities=["best_effort"],
                    publisher_durability="volatile",
                    subscriber_durabilities=["volatile"],
                    publisher_history="keep_last",
                    publisher_depth=10,
                    subscriber_histories=["unknown"],
                    subscriber_depths=[0],
                    subscriber_endpoint_metadata=[endpoint],
                    mc_identity=TEST_MC_IDENTITY,
                    **VALID_LOCAL_PUBLISHER_EVIDENCE,
                ))

    def test_unknown_zero_subscriber_gap_requires_verified_config_and_no_xml_override(self) -> None:
        base = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_distro="humble",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["unknown"],
            subscriber_depths=[0],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        self.assertIn("configuration_provenance_missing", validate_runtime_scope(
            **{**base, "mc_subscriber_provenance": None}
        ))
        self.assertIn("dds_qos_xml_profile_override_active", validate_runtime_scope(
            **{
                **base,
                "dds_qos_override_environment": {
                    **NO_DDS_QOS_OVERRIDES,
                    "FASTDDS_DEFAULT_PROFILES_FILE": "set",
                },
            }
        ))
        self.assertIn("unexpected_rmw_implementation", validate_runtime_scope(
            **{**base, "rmw_implementation": "rmw_cyclonedds_cpp"}
        ))
        self.assertIn("unexpected_ros_distro", validate_runtime_scope(
            **{**base, "ros_distro": "jazzy"}
        ))

    def test_subscriber_history_depth_rejects_partial_unknown_and_known_unsafe_values(self) -> None:
        base = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_distro="humble",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        for history, depth in (("unknown", 10), ("keep_last", 0), ("keep_last", -1), ("keep_all", 10)):
            with self.subTest(history=history, depth=depth):
                self.assertIn("mc_subscriber_history_depth_unsupported", validate_runtime_scope(
                    **base,
                    subscriber_histories=[history],
                    subscriber_depths=[depth],
                ))

    def test_verified_mc_config_and_startup_log_prove_route_not_reader_depth(self) -> None:
        config = """aimrt:
  plugin:
    plugins:
      - name: ros2_plugin
        options:
          node_name: mc_ros2_node
  channel:
    backends:
      - type: ros2
        options: {}
    sub_topics_options:
      - topic_name: /aima/mc/locomotion/velocity
        enable_backends: [ros2]
MotionControlModule:
  SubscriberManager:
    subscriber_cfg_list:
      - name: locomotion_velocity
        topic: /aima/mc/locomotion/velocity
        enable: true
"""
        startup = """node_name: mc_ros2_node
| Executable Path | /runtime/aima-mc-app-main |
| Process PID | 245 |
Create subscriber: locomotion_velocity succeeded, topic: /aima/mc/locomotion/velocity
| /aima/mc/locomotion/velocity | ros2:aimdk_msgs/msg/McLocomotionVelocity | MotionControlModule | ros2 | monitor |
"""
        config_bytes, startup_bytes = config.encode(), startup.encode()
        identity = VerifiedMcRuntimeIdentity(
            configured_node_name="mc_ros2_node",
            process_pid=245,
            process_executable="/runtime/aima-mc-app-main",
            mc_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            startup_log_sha256=hashlib.sha256(startup_bytes).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, startup_path = root / "mc.yaml", root / "mc.log"
            config_path.write_bytes(config_bytes)
            startup_path.write_bytes(startup_bytes)
            provenance = load_verified_mc_velocity_subscriber_provenance(
                config_path, startup_path, identity
            )
        self.assertTrue(provenance["startup_type_registration_confirmed"])
        self.assertEqual(provenance["actual_reader_history_depth"], "UNVERIFIED")
        self.assertEqual(provenance["configured_history_depth"], "not specified in audited subscriber configuration")

    def test_dds_override_environment_capture_is_non_disclosing_and_fail_closed(self) -> None:
        self.assertEqual(capture_dds_qos_override_environment({}), NO_DDS_QOS_OVERRIDES)
        self.assertEqual(
            capture_dds_qos_override_environment({"RMW_FASTRTPS_USE_QOS_FROM_XML": "0"}),
            {**NO_DDS_QOS_OVERRIDES, "RMW_FASTRTPS_USE_QOS_FROM_XML": "false"},
        )

    def test_transport_scope_rejects_unknown_zero_without_local_publisher_evidence(self) -> None:
        self.assertIn("local_command_publisher_identity_unverified", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="unknown",
            publisher_depth=0,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
        ))

    def test_transport_scope_rejects_local_publisher_api_qos_mismatch(self) -> None:
        local_evidence = {
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
            "publisher_api_qos": {
                **VALID_LOCAL_PUBLISHER_EVIDENCE["publisher_api_qos"],
                "depth": 5,
            },
        }
        self.assertIn("local_command_publisher_publisher_api_qos_mismatch", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="unknown",
            publisher_depth=0,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **local_evidence,
        ))

        configured_mismatch = {
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
            "publisher_configured_qos": {
                **VALID_LOCAL_PUBLISHER_EVIDENCE["publisher_configured_qos"],
                "depth": 9,
            },
        }
        self.assertIn("local_command_publisher_configured_qos_mismatch", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            publisher_history="unknown",
            publisher_depth=0,
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **configured_mismatch,
        ))

    def test_transport_scope_rejects_partial_unknown_and_positive_graph_mismatch(self) -> None:
        common = dict(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile"],
            subscriber_histories=["keep_last"],
            subscriber_depths=[10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        )
        for history, depth in (("unknown", 10), ("keep_last", 0), ("keep_last", 9), ("keep_all", 10)):
            with self.subTest(history=history, depth=depth):
                self.assertIn("publisher_history_depth", validate_runtime_scope(
                    publisher_history=history,
                    publisher_depth=depth,
                    **common,
                ))

    def test_transport_scope_rejects_local_publisher_identity_mismatch(self) -> None:
        bad_identity_fields = (
            {"topic_name": "/other/topic"},
            {"is_expected_message_type": False},
            {"is_owned_publisher": False},
            {"node_topic_publisher_count": 2},
        )
        for updates in bad_identity_fields:
            with self.subTest(updates=updates):
                local_evidence = {
                    **VALID_LOCAL_PUBLISHER_EVIDENCE,
                    "publisher_local_identity": {
                        **VALID_LOCAL_PUBLISHER_EVIDENCE["publisher_local_identity"],
                        **updates,
                    },
                }
                self.assertIn("local_command_publisher_identity_mismatch", validate_runtime_scope(
                    rmw_implementation="rmw_fastrtps_cpp",
                    ros_domain_id="20",
                    docker_network_mode="bridge",
                    publisher_names=["issue80_m1_focused_diagnostic"],
                    subscriber_names=["mc_ros2_node245"],
                    own_node_name="issue80_m1_focused_diagnostic",
                    publisher_reliability="reliable",
                    subscriber_reliabilities=["best_effort"],
                    publisher_durability="volatile",
                    subscriber_durabilities=["volatile"],
                    publisher_history="unknown",
                    publisher_depth=0,
                    subscriber_histories=["keep_last"],
                    subscriber_depths=[10],
                    mc_identity=TEST_MC_IDENTITY,
                    **local_evidence,
                ))

    def test_transport_scope_rejects_extra_velocity_subscriber(self) -> None:
        self.assertIn("subscriber", validate_runtime_scope(
            rmw_implementation="rmw_fastrtps_cpp",
            ros_domain_id="20",
            docker_network_mode="bridge",
            publisher_names=["issue80_m1_focused_diagnostic"],
            subscriber_names=["mc_ros2_node245", "external_controller"],
            own_node_name="issue80_m1_focused_diagnostic",
            publisher_reliability="reliable",
            subscriber_reliabilities=["best_effort", "best_effort"],
            publisher_durability="volatile",
            subscriber_durabilities=["volatile", "volatile"],
            publisher_history="keep_last",
            publisher_depth=10,
            subscriber_histories=["keep_last", "keep_last"],
            subscriber_depths=[10, 10],
            mc_identity=TEST_MC_IDENTITY,
            **VALID_LOCAL_PUBLISHER_EVIDENCE,
        ))

    def test_local_state_and_mc_command_topics_require_single_publishers(self) -> None:
        topics = {
            "/aima/hal/odom/state": ["aima-sim"],
            "/aima/hal/imu/torso/state": ["aima-sim"],
            "/aima/hal/joint/leg/state": ["aima-sim"],
            "/aima/hal/joint/leg/command": ["mc_ros2_node245"],
        }
        self.assertIsNone(validate_local_sim_publishers(topics, mc_identity=TEST_MC_IDENTITY))
        topics["/aima/hal/joint/leg/state"].append("external_robot")
        self.assertIn(
            "unexpected_local_runtime_publishers",
            validate_local_sim_publishers(topics, mc_identity=TEST_MC_IDENTITY),
        )
        topics["/aima/hal/joint/leg/state"].pop()
        topics["/aima/hal/joint/leg/command"] = ["mc_ros2_node245_extra"]
        self.assertIn(
            "unexpected_mc_leg_command_publisher",
            validate_local_sim_publishers(topics, mc_identity=TEST_MC_IDENTITY),
        )

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

    def test_installed_fsm_contract_mapping_is_pinned_and_named(self) -> None:
        self.assertEqual(MC_FSM_INTERFACE_SOURCE_PATH,
                         "interface/robot/mc/data/msg/common/McFsmState.msg")
        self.assertEqual(MC_FSM_INTERFACE_SOURCE_SHA256,
                         "da8a0370cba82d5d2955004a9b68489ee150f557e4c520dcd085bf58600cf6cf")
        self.assertEqual(
            {state.name: int(state) for state in McFsmStateCode},
            {
                "UNKNOWN": 0,
                "PREPARE": 1,
                "STARTING": 2,
                "FINISH": 3,
                "STABLE": 4,
                "MOVING": 5,
                "SAFE": 6,
                "SPECIAL": 7,
                "TEST": 8,
            },
        )
        self.assertEqual(MC_FSM_ACTIVE_STATES, frozenset((2, 4, 5)))

    def test_fsm_decoder_rejects_unverified_binary_build(self) -> None:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            for name in MC_FSM_RUNTIME_LIBRARY_SHA256:
                (Path(tmp) / name).write_bytes(b"unverified build")
            with self.assertRaisesRegex(ValueError, "binary contract mismatch"):
                verify_mc_fsm_runtime_contract(tmp)

    def test_stable_standing_fsm_is_accepted_but_safe_and_unrelated_states_fail(self) -> None:
        stable_fsm = self.snapshot(fsm_state=MC_FSM_STABLE, body_status=1)
        self.assertIsNone(snapshot_fault(
            stable_fsm, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ))
        safe_fsm = self.snapshot(fsm_state=MC_FSM_SAFE, body_status=1)
        self.assertEqual(snapshot_fault(
            safe_fsm, LIMITS, state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0
        ), "mc_safety_protection_state")
        for state in (McFsmStateCode.UNKNOWN, McFsmStateCode.PREPARE, McFsmStateCode.FINISH,
                      McFsmStateCode.SPECIAL, McFsmStateCode.TEST, 99):
            with self.subTest(state=int(state)):
                self.assertIn("unrelated_or_unknown_fsm_state", snapshot_fault(
                    self.snapshot(fsm_state=int(state)), LIMITS,
                    state_timeout_s=0.5, min_root_z=0.52, max_tilt_deg=15.0,
                ))
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
            fsm_state=int(McFsmStateCode.STARTING),
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
            fsm_state=int(McFsmStateCode.MOVING),
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
            fsm_state=int(McFsmStateCode.MOVING),
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
            fsm_state=int(McFsmStateCode.MOVING),
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
            fsm_state=int(McFsmStateCode.MOVING),
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
            fsm_state=int(McFsmStateCode.MOVING),
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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
                sample["pose"]["x"] = motion_ticks * 0.002
            else:
                sample["get_mode_desc"] = sample["common_mode_desc"] = "STAND_DEFAULT"
                sample["fsm_state"] = MC_FSM_STABLE

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

    def test_safe_fsm_during_locomotion_aborts_and_zero_stops(self) -> None:
        sample = self.snapshot()
        commands = []

        def publish(value):
            commands.append(value)
            if value == FORWARD_COMMAND_M_S:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = MC_FSM_SAFE

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["fault"], "mc_safety_protection_state")
        self.assertLessEqual(result["nonzero_publish_calls_returned"], 1)
        self.assertGreaterEqual(result["zero_stop"]["publish_call_return_span_s"], 1.0)
        self.assertTrue(any(command == 0.0 for command in commands))

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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
            elif value == 0.0:
                sample["pose"]["z"] = 0.50

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["status"], "POST_STOP_SAFETY_FAILED")
        self.assertEqual(result["post_stop_fault"], "fall_root_height")
        self.assertGreaterEqual(result["post_stop_emergency_zero"]["loop_duration_s"], 1.0)
        self.assertFalse(result["post_stop_emergency_zero"]["mc_stop_confirmed"])

    def test_safe_fsm_during_post_stop_observation_restarts_zero_stop(self) -> None:
        sample = self.snapshot()

        def publish(value):
            if value == FORWARD_COMMAND_M_S:
                sample["get_source_name"] = VELOCITY_SOURCE_NAME
                sample["common_source_name"] = VELOCITY_SOURCE_NAME
                sample["get_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["common_mode_desc"] = "LOCOMOTION_DEFAULT"
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
            elif value == 0.0:
                sample["fsm_state"] = MC_FSM_SAFE

        result = self.run_protocol(lambda: sample, publish)
        self.assertEqual(result["status"], "POST_STOP_SAFETY_FAILED")
        self.assertEqual(result["post_stop_fault"], "mc_safety_protection_state")
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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
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
                sample["fsm_state"] = int(McFsmStateCode.MOVING)
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
        self.assertEqual(result["status"], "ZERO_STOP_CALL_RETURN_COUNT_TOO_LOW")
        self.assertLess(result["zero_stop"]["zero_publish_calls_returned"], ZERO_STOP_MIN_MESSAGES)
        # A publish call already in flight at the bounded deadline can only
        # be observed after its measured 0.25 s fake latency returns.
        self.assertLessEqual(result["zero_stop"]["loop_duration_s"], 1.35)
        self.assertFalse(result["zero_stop"]["mc_receipt_confirmed"])
        self.assertFalse(result["zero_stop"]["mc_stop_confirmed"])


class PhysicalStopGateTests(unittest.TestCase):
    JOINTS = ("left_knee_joint", "right_knee_joint")

    def _sample(self, t: float, *, x: float | None = None) -> dict:
        return {
            "phase": "post-stop",
            "sample_monotonic_s": t,
            "sample_wall_utc": "2026-10-22T00:00:00+00:00",
            "pose": {
                "x": 0.001 * (t - 5.0) if x is None else x,
                "y": 0.0,
                "z": 0.65,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "vx": 0.001,
                "vy": 0.0,
                "vz": 0.0,
                "frame_id": "odom",
                "child_frame_id": "pelvis",
            },
            "state_age_s": {
                "odom": 0.01,
                "imu": 0.01,
                "common": 0.01,
                "leg_state": 0.01,
                "leg_command": 0.01,
                "get_mode": 0.01,
            },
            "get_mode_desc": "STAND_DEFAULT",
            "get_mode_header_code": 0,
            "get_mode_status_value": 100,
            "common_mode_desc": "STAND_DEFAULT",
            "common_mode_status_value": 100,
            "fsm_state": 4,
            "body_status": 1,
            "readback_fault": None,
            "safety_check": "PASS",
            "safety_fault": None,
            "leg_states": [
                {"name": name, "position": 0.5, "velocity": 0.001, "error_code": 0}
                for name in self.JOINTS
            ],
            "mc_leg_command": {
                "received_monotonic_s": t,
                "joints": [
                    {"name": name, "position": 0.5, "velocity": 0.0}
                    for name in self.JOINTS
                ],
            },
        }

    def _fixture(self):
        rows = [{
            "phase": "activation",
            "sample_monotonic_s": 4.99,
            "pose": {
                "x": 0.0, "y": 0.0, "z": 0.65,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        }]
        rows.extend(self._sample(t / 20.0) for t in range(101, 161))
        return rows

    def test_physical_gate_can_pass_without_reclassifying_legacy_baseline_failure(self):
        rows = self._fixture()
        legacy = measure_post_stop_forward_tail(rows, 0.00005)
        physical = evaluate_physical_stop_gate(
            rows,
            zero_start_monotonic_s=5.0,
            stand_confirmed_monotonic_s=5.5,
            baseline_state_summary={"body_status": 1},
        )
        self.assertEqual(legacy["status"], "ABOVE_BASELINE_ENVELOPE")
        self.assertEqual(physical["status"], "PASS")
        self.assertFalse(physical["mc_stop_ack"])
        self.assertEqual(physical["legacy_baseline_stop_gate"], "unchanged; see post_stop_forward_tail.status")

    def test_physical_gate_fails_unsafe_mode_freshness_and_motion(self):
        cases = (
            ("get_mode_desc", "LOCOMOTION_DEFAULT", "get_mc_action_not_stand_throughout_window"),
            ("fsm_state", 5, "raw_fsm_not_stable_throughout_window"),
            ("body_status", 2, "raw_body_status_changed_from_standing_baseline"),
        )
        for field, value, expected in cases:
            with self.subTest(field=field):
                rows = self._fixture()
                for row in rows:
                    if row.get("phase") == "post-stop" and row["sample_monotonic_s"] >= 6.0:
                        row[field] = value
                result = evaluate_physical_stop_gate(
                    rows, zero_start_monotonic_s=5.0, stand_confirmed_monotonic_s=5.5,
                    baseline_state_summary={"body_status": 1},
                )
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(expected, result["final_window_violations"])

        rows = self._fixture()
        for row in rows:
            if row.get("phase") == "post-stop" and row["sample_monotonic_s"] >= 6.0:
                row["pose"]["vx"] = 0.02
                row["state_age_s"]["odom"] = 0.6
        result = evaluate_physical_stop_gate(
            rows, zero_start_monotonic_s=5.0, stand_confirmed_monotonic_s=5.5,
            baseline_state_summary={"body_status": 1},
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("planar_speed_exceeded", result["final_window_violations"])
        self.assertIn("state_stale", result["final_window_violations"])

    def test_physical_gate_requires_joint_settling_and_a_complete_final_hold(self):
        rows = self._fixture()
        for row in rows:
            if row.get("phase") == "post-stop" and row["sample_monotonic_s"] >= 6.0:
                row["leg_states"][0]["velocity"] = 0.1
                row["mc_leg_command"]["joints"][0]["position"] += 0.1 * (row["sample_monotonic_s"] - 6.0)
        result = evaluate_physical_stop_gate(
            rows, zero_start_monotonic_s=5.0, stand_confirmed_monotonic_s=5.5,
            baseline_state_summary={"body_status": 1},
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("joint_velocity_not_settled", result["final_window_violations"])
        self.assertIn("joint_target_not_settled", result["final_window_violations"])

        short = [row for row in rows if row.get("phase") == "post-stop" and row["sample_monotonic_s"] <= 6.3]
        result = evaluate_physical_stop_gate(
            short, zero_start_monotonic_s=5.0, stand_confirmed_monotonic_s=5.5,
            baseline_state_summary={"body_status": 1},
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("stable_window_too_short", result["final_window_violations"])


class BoundedTurningProfileTests(unittest.TestCase):
    def test_sustained_yaw_rate_gate_filters_single_twist_spike(self):
        monitor = SustainedYawRateWindow(TURN_YAW_RATE_WINDOW_S)
        measurement = None
        for index in range(6):
            t = index * 0.1
            measurement = monitor.update({
                "sample_monotonic_s": t,
                "pose": {"yaw": 0.0, "wz": 1.2 if index == 2 else 0.0},
            })
        self.assertEqual(measurement["status"], "MEASURED")
        self.assertEqual(measurement["absolute_net_yaw_rate_rad_s"], 0.0)
        self.assertEqual(monitor.max_abs_raw_twist_rad_s, 1.2)
        self.assertLess(monitor.max_abs_window_rate_rad_s, TURN_MAX_SUSTAINED_YAW_RATE_RAD_S)

    def test_sustained_yaw_rate_gate_detects_net_heading_rate_over_limit(self):
        monitor = SustainedYawRateWindow(TURN_YAW_RATE_WINDOW_S)
        measurement = None
        for index in range(7):
            t = index * 0.1
            measurement = monitor.update({
                "sample_monotonic_s": t,
                "pose": {"yaw": 0.4 * t, "wz": 0.4},
            })
        self.assertEqual(measurement["status"], "MEASURED")
        self.assertGreater(
            measurement["absolute_net_yaw_rate_rad_s"],
            TURN_MAX_SUSTAINED_YAW_RATE_RAD_S,
        )

    def test_source_stamp_regression_is_recorded_while_host_clock_remains_authoritative(self):
        monitor = SustainedYawRateWindow(TURN_YAW_RATE_WINDOW_S)

        def snapshot(receive_time, stamp, yaw):
            sec = int(stamp)
            nanosec = round((stamp - sec) * 1_000_000_000)
            return {
                "sample_monotonic_s": receive_time,
                "pose": {"stamp_sec": sec, "stamp_nanosec": nanosec, "yaw": yaw},
            }

        monitor.update(snapshot(0.0, 10.0, 0.0))
        duplicate = monitor.update(snapshot(0.0, 10.0, 0.5))
        self.assertEqual(duplicate["status"], "DUPLICATE_SAMPLE")
        measured = monitor.update(snapshot(0.4, 9.5, 0.1))
        self.assertEqual(measured["status"], "MEASURED")
        self.assertAlmostEqual(measured["absolute_net_yaw_rate_rad_s"], 0.25, places=6)
        self.assertEqual(monitor.source_stamp_regression_count, 1)

        backwards = monitor.update(snapshot(0.3, 9.6, 0.0))
        self.assertEqual(backwards["status"], "HOST_MONOTONIC_RESET")

    def test_turn_ramp_is_symmetric_signed_and_never_exceeds_point_zero_seven_rad_s(self):
        left = [smooth_turn_velocity(0.07, t, 2.0) for t in (0.0, 0.125, 0.25, 1.0, 1.75, 1.875, 2.0)]
        right = [smooth_turn_velocity(-0.07, t, 2.0) for t in (0.0, 0.125, 0.25, 1.0, 1.75, 1.875, 2.0)]
        self.assertEqual(left[0], 0.0)
        self.assertEqual(left[-1], 0.0)
        self.assertEqual(right, [-value for value in left])
        self.assertLessEqual(max(map(abs, left)), 0.07)
        self.assertEqual(left[2], 0.07)

    def test_turn_profile_rejects_command_or_duration_above_the_fixed_cap(self):
        with self.assertRaises(ValueError):
            smooth_turn_velocity(0.07001, 0.5, 2.0)
        with self.assertRaises(ValueError):
            smooth_turn_velocity(-0.1, 0.5, TURN_PULSE_S + 0.001)

    def test_left_right_turn_pair_stays_at_or_below_ten_active_seconds(self):
        self.assertLessEqual(
            2.0 * (TURN_FORWARD_WARMUP_S + TURN_PULSE_S),
            10.0,
        )
        self.assertEqual(TURN_FORWARD_WARMUP_S + TURN_PULSE_S, TURN_MAX_COMMAND_WINDOW_S)

    def test_turn_direction_requires_heading_change_beyond_stationary_noise(self):
        left = classify_turning_rotation(
            start_yaw_rad=0.0,
            end_yaw_rad=0.10,
            commanded_angular_velocity_rad_s=0.1,
            baseline_yaw_noise_rad=0.001,
        )
        self.assertEqual(left["status"], "TURN_DIRECTION_PASS")

        right = classify_turning_rotation(
            start_yaw_rad=0.0,
            end_yaw_rad=-0.10,
            commanded_angular_velocity_rad_s=-0.1,
            baseline_yaw_noise_rad=0.001,
        )
        self.assertEqual(right["status"], "TURN_DIRECTION_PASS")

        opposite = classify_turning_rotation(
            start_yaw_rad=0.0,
            end_yaw_rad=-0.10,
            commanded_angular_velocity_rad_s=0.1,
            baseline_yaw_noise_rad=0.001,
        )
        self.assertEqual(opposite["status"], "TURN_OPPOSITE_DIRECTION")

        too_small = classify_turning_rotation(
            start_yaw_rad=0.0,
            end_yaw_rad=0.01,
            commanded_angular_velocity_rad_s=0.1,
            baseline_yaw_noise_rad=0.001,
        )
        self.assertEqual(too_small["status"], "TURN_HEADING_CHANGE_TOO_SMALL")

    def test_turn_preflight_rejects_live_angular_bound_before_nonzero_publish(self):
        clock = FakeClock()
        base = FirstMotionSafetyTests().snapshot(
            get_mode_status_value=100,
            common_mode_status_value=100,
            pose={
                "x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0,
                "pitch": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "wz": 0.0, "frame_id": "odom", "child_frame_id": "base_link",
            },
            speed_status={
                "speed_mode": 1,
                "angular_bounds": {"min_value": -0.04, "max_value": 0.04},
            },
            leg_states=[
                {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 0.0, "error_code": 0}
            ],
            mc_leg_command={
                "joints": [{"name": "left_hip_pitch_joint", "position": 0.0}]
            },
        )
        turn_commands = []
        zero_commands = []

        def sample():
            row = dict(base)
            row["sample_monotonic_s"] = clock.now()
            row["readback_monotonic_s"] = {"get_mode": clock.now(), "get_source": clock.now()}
            row["mc_leg_command"] = {"joints": [{"name": "left_hip_pitch_joint", "position": 0.0}]}
            return row

        result = run_angular_turn_window(
            sample=sample,
            publish_turn=lambda _forward, _angular: turn_commands.append(0.0),
            publish_zero=lambda: zero_commands.append(0.0),
            registration={"success": True, "selected_action": "ADD"},
            set_action={"accepted_by_service": True},
            joint_limits=LIMITS,
            baseline_forward_displacements_m=[0.0, 0.0, 0.0],
            baseline_state_summary={"body_status": 1, "maxima": {"max_abs_yaw_excursion_rad": 0.001}},
            authorized=True,
            angular_velocity_rad_s=0.05,
            pulse_duration_s=2.0,
            extra_post_stop_observe_s=0.0,
            clock=clock.now,
            sleep=clock.sleep,
        )
        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["fault"], "turn_command_outside_live_angular_bounds")
        self.assertEqual(turn_commands, [])
        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], 49)
        self.assertFalse(result["mc_stop_ack"])

    def test_forward_arc_preflight_rejects_command_above_live_forward_bound(self):
        clock = FakeClock()
        base = FirstMotionSafetyTests().snapshot(
            get_mode_status_value=100,
            common_mode_status_value=100,
            pose={
                "x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0,
                "pitch": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "wz": 0.0, "frame_id": "odom", "child_frame_id": "base_link",
            },
            speed_status={
                "speed_mode": 1,
                "angular_bounds": {"min_value": -1.0, "max_value": 1.0},
                "forward_bounds": {"min_value": -0.2, "max_value": 0.1},
            },
            leg_states=[
                {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 0.0, "error_code": 0}
            ],
            mc_leg_command={
                "joints": [{"name": "left_hip_pitch_joint", "position": 0.0}]
            },
        )
        turn_commands = []

        def sample():
            row = dict(base)
            row["sample_monotonic_s"] = clock.now()
            row["readback_monotonic_s"] = {"get_mode": clock.now(), "get_source": clock.now()}
            return row

        result = run_angular_turn_window(
            sample=sample,
            publish_turn=lambda forward, angular: turn_commands.append((forward, angular)),
            publish_zero=lambda: None,
            registration={"success": True, "selected_action": "ADD"},
            set_action={"accepted_by_service": True},
            joint_limits=LIMITS,
            baseline_forward_displacements_m=[0.0, 0.0, 0.0],
            baseline_state_summary={"body_status": 1, "maxima": {"max_abs_yaw_excursion_rad": 0.001}},
            authorized=True,
            angular_velocity_rad_s=0.07,
            forward_velocity_m_s=0.12,
            pulse_duration_s=2.0,
            extra_post_stop_observe_s=0.0,
            clock=clock.now,
            sleep=clock.sleep,
        )

        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["fault"], "turn_forward_command_outside_live_forward_bounds")
        self.assertEqual(turn_commands, [])

    def test_turn_zero_stop_keeps_ros_readbacks_fresh_during_zero_stream(self):
        clock = FakeClock()
        base = FirstMotionSafetyTests().snapshot(
            get_mode_status_value=100,
            common_mode_status_value=100,
            pose={
                "x": 0.0, "y": 0.0, "z": 0.64, "roll": 0.0,
                "pitch": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "wz": 0.0, "frame_id": "odom", "child_frame_id": "base_link",
            },
            speed_status={
                "speed_mode": 1,
                "angular_bounds": {"min_value": -1.0, "max_value": 1.0},
                "forward_bounds": {"min_value": -1.0, "max_value": 0.8},
            },
            leg_states=[
                {"name": "left_hip_pitch_joint", "position": 0.0, "velocity": 0.0, "error_code": 0}
            ],
            mc_leg_command={
                "joints": [{"name": "left_hip_pitch_joint", "position": 0.0}]
            },
        )
        zero_commands = []
        turn_commands = []
        post_stop_samples = []

        def sample():
            now = clock.now()
            # Model the bounded executor work required to service the async
            # GetMcAction/GetCurrentInputSource futures during zero publication.
            with clock.lock:
                clock.value += 0.005
            row = dict(base)
            row["sample_monotonic_s"] = now
            row["readback_monotonic_s"] = {"get_mode": now, "get_source": now}
            row["state_age_s"] = {key: 0.0 for key in base["state_age_s"]}
            row["mc_leg_command"] = {"joints": [{"name": "left_hip_pitch_joint", "position": 0.0}]}
            return row

        def record_sample(phase, snapshot):
            if phase == "post-stop":
                post_stop_samples.append(snapshot)

        result = run_angular_turn_window(
            sample=sample,
            publish_turn=lambda forward, angular: turn_commands.append((forward, angular)),
            publish_zero=lambda: zero_commands.append(clock.now()),
            registration={"success": True, "selected_action": "ADD"},
            set_action={"accepted_by_service": True},
            joint_limits=LIMITS,
            baseline_forward_displacements_m=[0.0, 0.0, 0.0],
            baseline_state_summary={"body_status": 1, "maxima": {"max_abs_yaw_excursion_rad": 0.001}},
            authorized=True,
            angular_velocity_rad_s=0.07,
            forward_velocity_m_s=TURN_FORWARD_COMMAND_M_S,
            forward_warmup_s=TURN_FORWARD_WARMUP_S,
            pulse_duration_s=2.0,
            extra_post_stop_observe_s=0.0,
            clock=clock.now,
            sleep=clock.sleep,
            on_sample=record_sample,
        )

        self.assertGreaterEqual(result["zero_stop"]["zero_publish_calls_returned"], ZERO_STOP_MIN_MESSAGES)
        self.assertEqual(result["zero_stop"]["during_zero_callback_errors"], [])
        self.assertEqual(
            result["zero_stop"]["during_zero_callback_count"],
            result["zero_stop"]["zero_publish_calls_returned"],
        )
        self.assertGreaterEqual(result["zero_stop"]["publish_call_return_span_s"], ZERO_STOP_MIN_S)
        self.assertLessEqual(result["zero_stop"]["max_gap_between_publish_call_returns_s"], 0.020001)
        self.assertTrue(post_stop_samples)
        self.assertTrue(all(s["state_age_s"]["get_mode"] == 0.0 for s in post_stop_samples))
        self.assertEqual(result["fault"], "turn_control_handshake_not_confirmed_after_forward_warmup")
        self.assertGreater(len(turn_commands), 0)
        self.assertTrue(all(angular == 0.0 for _forward, angular in turn_commands))

    def test_forward_arc_planar_envelope_preserves_lateral_drift_limit(self):
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        expected_arc = classify_turn_planar_motion(
            start_pose=start,
            end_pose={"x": 0.24, "y": 0.01, "yaw": 0.14},
            forward_velocity_m_s=0.12,
            pulse_duration_s=2.0,
        )
        self.assertEqual(expected_arc["status"], "PASS")
        self.assertAlmostEqual(expected_arc["expected_forward_displacement_m"], 0.24)
        self.assertEqual(expected_arc["allowed_planar_displacement_m"], 0.29)

        excessive_lateral = classify_turn_planar_motion(
            start_pose=start,
            end_pose={"x": 0.20, "y": 0.051, "yaw": 0.14},
            forward_velocity_m_s=0.12,
            pulse_duration_s=2.0,
        )
        self.assertEqual(excessive_lateral["status"], "PLANAR_MOTION_OUTSIDE_COMMAND_ENVELOPE")

        yaw_only_drift = classify_turn_planar_motion(
            start_pose=start,
            end_pose={"x": 0.051, "y": 0.0, "yaw": 0.14},
            forward_velocity_m_s=0.0,
            pulse_duration_s=2.0,
        )
        self.assertEqual(yaw_only_drift["status"], "PLANAR_MOTION_OUTSIDE_COMMAND_ENVELOPE")


if __name__ == "__main__":
    unittest.main()
