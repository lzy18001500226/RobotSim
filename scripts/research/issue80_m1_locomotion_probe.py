#!/usr/bin/env python3
"""Bounded, simulation-only X2 locomotion authority and motion probe for Issue #80."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import (
    JointCommandArray,
    JointStateArray,
    McActionCommand,
    McCommonState,
    McLocomotionVelocity,
    MessageHeader,
    RequestHeader,
)
from aimdk_msgs.srv import SetMcAction, SetMcInputSource
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


VELOCITY_TOPIC = "/aima/mc/locomotion/velocity"
INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
MODE_LD = "LOCOMOTION_DEFAULT"
REQUIRED_TOPICS = ("odom", "imu", "common", "leg_state", "leg_command")


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def roll_pitch_from_quaternion(q: Any) -> tuple[float, float]:
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    roll = math.atan2(sinr, cosr)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _json_safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.get_fields_and_field_types()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


class M1Probe(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("issue80_m1_locomotion_probe")
        self.args = args
        self.source = args.source
        self.output_dir = Path(args.output_dir).resolve(strict=True)
        self.trace_path = self.output_dir / "m1_trace.jsonl"
        self.result_path = self.output_dir / "m1_result.json"
        if self.trace_path.exists() or self.result_path.exists():
            raise FileExistsError("refusing to overwrite an existing M1 evidence file")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        registration_path = Path(args.registration_result).resolve(strict=True)
        self.registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if (
            self.registration.get("source") != self.source
            or self.registration.get("response_code") != 0
        ):
            raise ValueError("registration result does not confirm this source")

        self.latest: dict[str, Any] = {}
        self.received_at: dict[str, float] = {}
        self.phase = "WAITING"
        self.phase_started = 0.0
        self.phase_samples: dict[str, dict[str, float]] = {}
        self.phase_plan: list[tuple[str, float, float, float]] = []
        self.phase_index = -1
        self.armed = False
        self.interrupted = False
        self.authority_zero_complete = False
        self.done = False
        self.fault: str | None = None
        self.max_abs_roll = 0.0
        self.max_abs_pitch = 0.0
        self.min_root_z = math.inf
        self.max_joint_speed = 0.0
        self.nonzero_command_count = 0
        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            McLocomotionVelocity, VELOCITY_TOPIC, self.state_qos
        )
        self.mode_client = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")
        self.mode_transition: dict[str, Any] | None = None

        self.create_subscription(
            Odometry, "/aima/hal/odom/state", self._odom_cb, self.state_qos
        )
        self.create_subscription(
            Imu, "/aima/hal/imu/torso/state", self._imu_cb, self.state_qos
        )
        self.create_subscription(
            McCommonState, "/aima/mc/common/state", self._common_cb, self.state_qos
        )
        self.create_subscription(
            JointStateArray, "/aima/hal/joint/leg/state", self._leg_state_cb, self.state_qos
        )
        self.create_subscription(
            JointCommandArray,
            "/aima/hal/joint/leg/command",
            self._leg_command_cb,
            self.state_qos,
        )
        self.timer = self.create_timer(0.05, self._tick)

    def _store(self, key: str, msg: Any) -> None:
        self.latest[key] = msg
        self.received_at[key] = time.monotonic()

    def _odom_cb(self, msg: Odometry) -> None:
        self._store("odom", msg)

    def _imu_cb(self, msg: Imu) -> None:
        self._store("imu", msg)

    def _common_cb(self, msg: McCommonState) -> None:
        self._store("common", msg)

    def _leg_state_cb(self, msg: JointStateArray) -> None:
        self._store("leg_state", msg)

    def _leg_command_cb(self, msg: JointCommandArray) -> None:
        self._store("leg_command", msg)

    def _mode(self) -> str:
        common = self.latest.get("common")
        return common.action_info.action_desc if common is not None else ""

    def _pose(self) -> dict[str, float] | None:
        odom = self.latest.get("odom")
        if odom is None:
            return None
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        roll, pitch = roll_pitch_from_quaternion(q)
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw_from_quaternion(q),
            "vx": float(odom.twist.twist.linear.x),
            "vy": float(odom.twist.twist.linear.y),
            "vz": float(odom.twist.twist.linear.z),
            "wx": float(odom.twist.twist.angular.x),
            "wy": float(odom.twist.twist.angular.y),
            "wz": float(odom.twist.twist.angular.z),
            "stamp_sec": int(odom.header.stamp.sec),
            "stamp_nanosec": int(odom.header.stamp.nanosec),
        }

    def _freshness(self) -> dict[str, float | None]:
        now = time.monotonic()
        return {
            key: now - self.received_at[key] if key in self.received_at else None
            for key in REQUIRED_TOPICS
        }

    def _safety_fault(self) -> str | None:
        ages = self._freshness()
        missing = [key for key, age in ages.items() if age is None or age > self.args.state_timeout]
        if missing:
            return "stale_or_missing_state:" + ",".join(missing)
        transitional_phase = self.phase in {"AUTHORITY_ZERO", "AUTHORITY_GATE", "MODE_SWITCH"}
        if transitional_phase:
            if self._mode() not in {"STAND_DEFAULT", MODE_LD}:
                return "MC_left_standing_or_locomotion_mode"
        elif self._mode() != MODE_LD:
            return "MC_left_LOCOMOTION_DEFAULT"
        pose = self._pose()
        assert pose is not None
        vals = [pose[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")]
        if not all(math.isfinite(value) for value in vals):
            return "nonfinite_odometry"
        if pose["z"] < self.args.min_root_z:
            return "root_height_below_safety_floor"
        if max(abs(pose["roll"]), abs(pose["pitch"])) > math.radians(self.args.max_tilt_deg):
            return "tilt_safety_stop"
        for joint in self.latest["leg_state"].joints:
            if not all(math.isfinite(float(v)) for v in (joint.position, joint.velocity)):
                return "nonfinite_leg_state"
        return None

    def _publish(self, forward: float = 0.0, lateral: float = 0.0, yaw_rate: float = 0.0) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source
        msg.forward_velocity = float(forward)
        msg.lateral_velocity = float(lateral)
        msg.angular_velocity = float(yaw_rate)
        self.publisher.publish(msg)
        if forward != 0.0 or lateral != 0.0 or yaw_rate != 0.0:
            self.nonzero_command_count += 1

    def _snapshot(self) -> dict[str, float] | None:
        pose = self._pose()
        if pose is None:
            return None
        return {key: pose[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")}

    def _set_phase(self, name: str, now: float) -> None:
        self.phase = name
        self.phase_started = now
        pose = self._snapshot()
        if pose is not None:
            self.phase_samples[name] = pose

    def _tick(self) -> None:
        if not self.armed:
            return
        now = time.monotonic()
        if self.interrupted:
            self.fault = "operator_signal_stop"
            self._set_phase("SAFE_STOP", now)
            self._publish()
            self._write_trace((0.0, 0.0, 0.0))
            self.done = True
            return
        fault = self._safety_fault()
        pose = self._pose()
        if pose is not None:
            self.max_abs_roll = max(self.max_abs_roll, abs(pose["roll"]))
            self.max_abs_pitch = max(self.max_abs_pitch, abs(pose["pitch"]))
            self.min_root_z = min(self.min_root_z, pose["z"])
        if fault:
            self.fault = fault
            self._set_phase("SAFE_STOP", now)
            self._publish()
            self._write_trace((0.0, 0.0, 0.0))
            self.done = True
            return

        command = (0.0, 0.0, 0.0)
        elapsed = now - self.phase_started
        if self.phase == "AUTHORITY_ZERO":
            if elapsed >= self.args.authority_zero_duration:
                self.authority_zero_complete = True
                self._set_phase("AUTHORITY_GATE", now)
        elif self.phase == "AUTHORITY_GATE":
            pass
        elif 0 <= self.phase_index < len(self.phase_plan):
            name, duration, forward, yaw_rate = self.phase_plan[self.phase_index]
            if elapsed >= duration:
                self.phase_index += 1
                if self.phase_index >= len(self.phase_plan):
                    self._set_phase("DONE", now)
                    self.done = True
                else:
                    name, _, _, _ = self.phase_plan[self.phase_index]
                    self._set_phase(name, now)
            else:
                command = (forward, 0.0, yaw_rate)

        self._publish(*command)
        self._write_trace(command)

    def _write_trace(self, command: tuple[float, float, float]) -> None:
        now = time.monotonic()
        pose = self._pose()
        leg_state = self.latest.get("leg_state")
        if leg_state is not None:
            self.max_joint_speed = max(
                self.max_joint_speed,
                max((abs(float(j.velocity)) for j in leg_state.joints), default=0.0),
            )
        common = self.latest.get("common")
        row = {
            "wall_monotonic_sec": now,
            "phase": self.phase,
            "mode": self._mode(),
            "fsm_state": int(common.fsm_state.current_state) if common else None,
            "body_status": int(common.body_status.value) if common else None,
            "active_input_source": common.input_source.name if common else None,
            "pose_odom": pose,
            "imu": _json_safe(self.latest.get("imu")),
            "state_age_sec": self._freshness(),
            "command": {
                "source": self.source,
                "forward_m_s": command[0],
                "lateral_m_s": command[1],
                "yaw_rate_rad_s": command[2],
            },
            "leg_state": _json_safe(leg_state),
            "leg_command": _json_safe(self.latest.get("leg_command")),
            "fault": self.fault,
        }
        self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def request_locomotion_mode(self) -> dict[str, Any]:
        if not self.mode_client.wait_for_service(timeout_sec=3.0):
            raise TimeoutError("SetMcAction service unavailable")
        mode_before = self._mode()
        common_source = self.latest["common"].input_source.name
        if common_source != self.source:
            raise RuntimeError(f"zero velocity did not activate the registered source: {common_source!r}")

        request = SetMcAction.Request()
        request.header = RequestHeader()
        request.header.stamp = self.get_clock().now().to_msg()
        request.source = self.source
        request.command = McActionCommand()
        request.command.action_desc = MODE_LD
        future = self.mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if self.interrupted:
            raise RuntimeError("operator_signal_stop")
        if not future.done() or future.result() is None:
            raise TimeoutError("SetMcAction LOCOMOTION_DEFAULT request timed out")
        response = future.result().response
        result = {
            "service": "/aimdk_5Fmsgs/srv/SetMcAction",
            "source": self.source,
            "requested_action": MODE_LD,
            "response_status": int(response.status.value),
            "response_message": response.message,
            "mode_before": mode_before,
        }
        if response.status.value != 1:
            self.mode_transition = result
            raise RuntimeError(f"MC rejected locomotion mode request: {result}")

        self._set_phase("MODE_SWITCH", time.monotonic())
        deadline = time.monotonic() + self.args.mode_timeout
        while rclpy.ok() and time.monotonic() < deadline and self._mode() != MODE_LD:
            rclpy.spin_once(self, timeout_sec=0.05)
            fault = self._safety_fault()
            if fault:
                self.mode_transition = result
                raise RuntimeError(fault)
            if self.latest["common"].input_source.name != self.source:
                self.mode_transition = result
                raise RuntimeError("active input source changed during mode transition")
        result["mode_after"] = self._mode()
        result["fsm_state_after"] = int(self.latest["common"].fsm_state.current_state)
        result["active_input_source_after"] = self.latest["common"].input_source.name
        self.mode_transition = result
        if result["mode_after"] != MODE_LD:
            raise RuntimeError(f"mode service acknowledged but readback stayed {result['mode_after']}")
        return result

    def wait_for_ready_state(self) -> None:
        deadline = time.monotonic() + self.args.ready_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.interrupted:
                raise RuntimeError("operator_signal_stop")
            rclpy.spin_once(self, timeout_sec=0.05)
            ages = self._freshness()
            if all(age is not None and age <= self.args.state_timeout for age in ages.values()):
                pose = self._pose()
                if pose and self._mode() == "STAND_DEFAULT":
                    if pose["z"] < self.args.min_root_z:
                        raise RuntimeError("initial root is below upright safety threshold")
                    if max(abs(pose["roll"]), abs(pose["pitch"])) > math.radians(self.args.max_tilt_deg):
                        raise RuntimeError("initial tilt exceeds M1 safety threshold")
                    return
        raise TimeoutError("live standing telemetry did not become fresh after Reset")

    def check_single_publisher(self) -> list[str]:
        deadline = time.monotonic() + 1.0
        names: list[str] = []
        while time.monotonic() < deadline:
            infos = self.get_publishers_info_by_topic(VELOCITY_TOPIC)
            names = [info.node_name for info in infos]
            if names:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        if names != [self.get_name()]:
            raise RuntimeError(f"conflicting velocity publishers: {names}")
        return names

    def start_authority_zero(self) -> None:
        self.phase_plan = [
            ("ZERO_BASELINE", self.args.zero_duration, 0.0, 0.0),
            ("FORWARD", self.args.forward_duration, self.args.forward_speed, 0.0),
            ("STOP_AFTER_FORWARD", self.args.interstage_stop, 0.0, 0.0),
            ("TURN", self.args.turn_duration, 0.0, self.args.turn_rate),
            ("FINAL_STOP", self.args.final_stop_duration, 0.0, 0.0),
        ]
        self.armed = True
        self.phase_index = -1
        self._set_phase("AUTHORITY_ZERO", time.monotonic())

    def start_motion(self) -> None:
        self.phase_index = 0
        name, _, _, _ = self.phase_plan[0]
        self._set_phase(name, time.monotonic())

    def wait_authority_zero(self) -> None:
        deadline = time.monotonic() + self.args.authority_zero_duration + 2.0
        while rclpy.ok() and not self.authority_zero_complete and time.monotonic() < deadline:
            if self.interrupted:
                raise RuntimeError("operator_signal_stop")
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.fault:
                raise RuntimeError(self.fault)
        if not self.authority_zero_complete:
            raise TimeoutError("zero-command authority preflight timed out")

    def execute(self) -> dict[str, Any]:
        self.wait_for_ready_state()
        publishers = self.check_single_publisher()
        registration = self.registration
        common = self.latest["common"]
        forward_bounds = common.speed_status.forward_bounds
        angular_bounds = common.speed_status.angular_bounds
        if not forward_bounds.min_value <= self.args.forward_speed <= forward_bounds.max_value:
            raise ValueError("forward probe speed is outside the live MC bounds")
        if not angular_bounds.min_value <= self.args.turn_rate <= angular_bounds.max_value:
            raise ValueError("turn probe rate is outside the live MC bounds")
        self.start_authority_zero()
        self.wait_authority_zero()
        if self.latest["common"].input_source.name != self.source:
            raise RuntimeError("bounded zero command did not activate the registered input source")
        mode_transition = self.request_locomotion_mode()
        self.start_motion()
        deadline = time.monotonic() + sum(item[1] for item in self.phase_plan) + 3.0
        while rclpy.ok() and not self.done and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not self.done:
            self.fault = self.fault or "probe_deadline_exceeded"
            self._set_phase("SAFE_STOP", time.monotonic())
        self.stop_now()
        return self.make_result(publishers, registration, mode_transition)

    def stop_now(self) -> None:
        for _ in range(10):
            self._publish()
            time.sleep(0.01)

    def make_result(
        self,
        publishers: list[str],
        registration: dict[str, Any],
        mode_transition: dict[str, Any],
    ) -> dict[str, Any]:
        p0 = self.phase_samples.get("ZERO_BASELINE") or self.phase_samples.get("AUTHORITY_ZERO")
        pf0 = self.phase_samples.get("FORWARD")
        pf1 = self.phase_samples.get("STOP_AFTER_FORWARD")
        pt0 = self.phase_samples.get("TURN")
        pt1 = self.phase_samples.get("FINAL_STOP") or self.phase_samples.get("DONE")
        if pf0 and pf1:
            dx = pf1["x"] - pf0["x"]
            dy = pf1["y"] - pf0["y"]
            forward_projection = dx * math.cos(pf0["yaw"]) + dy * math.sin(pf0["yaw"])
            forward_euclidean = math.hypot(dx, dy)
        else:
            forward_projection = None
            forward_euclidean = None
        yaw_change = wrap_angle(pt1["yaw"] - pt0["yaw"]) if pt0 and pt1 else None
        final_pose = self._snapshot()
        passed = (
            self.fault is None
            and forward_projection is not None
            and forward_projection >= self.args.min_forward_displacement
            and yaw_change is not None
            and abs(yaw_change) >= self.args.min_yaw_change
        )
        result = {
            "schema": "robotsim.issue80.m1.v1",
            "status": "PASS" if passed else "FAIL",
            "gate": "M1_STAND_TO_LOCOMOTION_MEASURED_MOTION",
            "simulation_only": True,
            "mode_required": MODE_LD,
            "active_velocity_topic": VELOCITY_TOPIC,
            "velocity_publishers": publishers,
            "registration": registration,
            "mode_transition": mode_transition,
            "active_source": mode_transition.get("active_input_source_after"),
            "commands": {
                "forward_m_s": self.args.forward_speed,
                "forward_duration_s": self.args.forward_duration,
                "yaw_rate_rad_s": self.args.turn_rate,
                "turn_duration_s": self.args.turn_duration,
                "stale_state_timeout_s": self.args.state_timeout,
                "safe_stop_max_tilt_deg": self.args.max_tilt_deg,
                "safe_stop_min_root_z_m": self.args.min_root_z,
            },
            "criteria": {
                "min_forward_projection_m": self.args.min_forward_displacement,
                "min_abs_yaw_change_rad": self.args.min_yaw_change,
                "these_are_robot_sim_experiment_thresholds_not_vendor_limits": True,
            },
            "measurements": {
                "start_pose": p0,
                "forward_start_pose": pf0,
                "forward_end_pose": pf1,
                "forward_projection_m": forward_projection,
                "forward_euclidean_displacement_m": forward_euclidean,
                "turn_start_pose": pt0,
                "turn_end_pose": pt1,
                "yaw_change_rad": yaw_change,
                "final_pose": final_pose,
                "max_abs_roll_deg": math.degrees(self.max_abs_roll),
                "max_abs_pitch_deg": math.degrees(self.max_abs_pitch),
                "min_root_z_m": self.min_root_z if math.isfinite(self.min_root_z) else None,
                "max_leg_joint_speed_rad_s": self.max_joint_speed,
                "nonzero_velocity_messages": self.nonzero_command_count,
                "final_mode": self._mode(),
                "final_fsm_state": int(self.latest["common"].fsm_state.current_state)
                if "common" in self.latest
                else None,
                "final_body_status": int(self.latest["common"].body_status.value)
                if "common" in self.latest
                else None,
            },
            "fault": self.fault,
            "limitations": [
                "Timed bounded probes establish locomotion, not autonomous point-goal navigation.",
                "The pre-transition authority phase publishes zero only; no nonzero command is sent before LD readback.",
                "M1 safe-stop thresholds are simulation experiment guards, not vendor limits.",
                "MC 2 ms worker missed-wakeup warnings and standing ankle target/state offsets remain monitored separately.",
            ],
        }
        with self.result_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result

    def close(self) -> None:
        try:
            self.stop_now()
        finally:
            self.trace.close()

    def write_blocked_result(self, fault: str) -> None:
        self.fault = fault
        if self.result_path.exists():
            return
        result = {
            "schema": "robotsim.issue80.m1.v1",
            "status": "BLOCKED",
            "gate": "M1_STAND_TO_LOCOMOTION_MEASURED_MOTION",
            "simulation_only": True,
            "source": self.source,
            "mode_observed": self._mode(),
            "mode_transition": self.mode_transition,
            "fault": fault,
            "trace_path": str(self.trace_path),
            "limitations": [
                "No nonzero velocity is sent unless LOCOMOTION_DEFAULT and the active input source are confirmed.",
                "Timed bounded probes are not autonomous point-goal navigation.",
            ],
        }
        with self.result_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


class SourceRegistrar(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("issue80_input_source_registrar")
        self.args = args
        self.client = self.create_client(SetMcInputSource, INPUT_SOURCE_SERVICE)

    def register(self) -> dict[str, Any]:
        if not self.client.wait_for_service(timeout_sec=3.0):
            raise TimeoutError("SetMcInputSource service unavailable")
        action_values = {"ADD": 1001, "ENABLE": 2001}
        if self.args.action not in action_values:
            raise ValueError(f"unsupported source action: {self.args.action}")
        request = SetMcInputSource.Request()
        request.request.header.stamp = self.get_clock().now().to_msg()
        request.action.value = action_values[self.args.action]
        request.input_source.name = self.args.source
        request.input_source.priority = self.args.priority
        request.input_source.timeout = self.args.source_timeout_ms
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if not future.done() or future.result() is None:
            raise TimeoutError("SetMcInputSource registration timed out")
        response = future.result().response
        result = {
            "service": INPUT_SOURCE_SERVICE,
            "action": int(request.action.value),
            "action_name": self.args.action,
            "source": self.args.source,
            "priority": int(request.input_source.priority),
            "timeout_ms": int(request.input_source.timeout),
            "response_code": int(response.header.code),
            "response_state": int(response.state.value),
            "task_id": int(response.task_id),
        }
        if response.header.code != 0:
            raise RuntimeError(f"input source registration failed: {result}")
        return result


def _register_command(args: argparse.Namespace) -> int:
    rclpy.init()
    node = SourceRegistrar(args)
    try:
        result = node.register()
        print(json.dumps(result, indent=2))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _probe_command(args: argparse.Namespace) -> int:
    rclpy.init()
    node = M1Probe(args)

    def request_stop(signum, frame):
        node.interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = node.execute()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    except KeyboardInterrupt:
        node.fault = "operator_interrupt"
        node.interrupted = True
        node.stop_now()
        node.write_blocked_result(node.fault)
        return 130
    except Exception as exc:
        node.fault = str(exc)
        node.stop_now()
        node.write_blocked_result(node.fault)
        print(json.dumps({"status": "BLOCKED", "fault": node.fault}, indent=2))
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default="issue80_nav")
    parser.add_argument("--priority", type=int, default=40)
    parser.add_argument("--source-timeout-ms", type=int, default=1000)
    parser.add_argument("--state-timeout", type=float, default=0.5)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--mode-timeout", type=float, default=5.0)
    parser.add_argument("--max-tilt-deg", type=float, default=15.0)
    parser.add_argument("--min-root-z", type=float, default=0.52)
    parser.add_argument("--authority-zero-duration", type=float, default=0.8)
    parser.add_argument("--zero-duration", type=float, default=0.5)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument("--forward-duration", type=float, default=1.0)
    parser.add_argument("--interstage-stop", type=float, default=0.75)
    parser.add_argument("--turn-rate", type=float, default=0.1)
    parser.add_argument("--turn-duration", type=float, default=1.0)
    parser.add_argument("--final-stop-duration", type=float, default=1.5)
    parser.add_argument("--min-forward-displacement", type=float, default=0.02)
    parser.add_argument("--min-yaw-change", type=float, default=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register", help="register a command source only")
    _add_common_arguments(register)
    register.add_argument("--action", choices=("ADD", "ENABLE"), default="ADD")
    probe = subparsers.add_parser("probe", help="run bounded M1 zero/forward/turn/stop probe")
    _add_common_arguments(probe)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--registration-result", required=True)
    args = parser.parse_args()
    try:
        if args.command == "register":
            return _register_command(args)
        return _probe_command(args)
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
