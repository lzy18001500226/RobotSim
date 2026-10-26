#!/usr/bin/env python3
"""Read-only ROS observer for one RAICOM Reset-only telemetry capture.

This process subscribes to state topics only. It has no publisher, service
client, route manager, or MuJoCo stepping interface. Reset is dispatched once
by the parent runner through the verified MuJoCo native Simulation/Reset UI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import rclpy
from aimdk_msgs.msg import JointCommandArray, JointStateArray, McCommonState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


TOPIC_TYPES = {
    "odom": (Odometry, "/aima/hal/odom/state"),
    "imu": (Imu, "/aima/hal/imu/torso/state"),
    "common": (McCommonState, "/aima/mc/common/state"),
    "leg_state": (JointStateArray, "/aima/hal/joint/leg/state"),
    "leg_command": (JointCommandArray, "/aima/hal/joint/leg/command"),
}
MAX_STATE_AGE_S = 0.5
MIN_ROOT_Z_M = 0.52
MAX_ABS_TILT_RAD = math.radians(15.0)
REQUIRED_UPRIGHT_ODOM_SAMPLES = 3
MAX_READY_WAIT_S = 5.0
MAX_RESET_EVENT_WAIT_S = 5.0
POST_RESET_OBSERVATION_S = 5.0
MAX_OBSERVER_TOTAL_S = 15.0
RESET_EFFECT_MIN_POSITION_DELTA_M = 0.001
RESET_EFFECT_MIN_YAW_DELTA_RAD = 0.005


def _safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {
            name: _safe(getattr(value, name))
            for name in value.get_fields_and_field_types()
        }
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec, nanosec = getattr(stamp, "sec", None), getattr(stamp, "nanosec", None)
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        return None
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def _odom_pose(message: Odometry) -> dict[str, Any]:
    position = message.pose.pose.position
    q = message.pose.pose.orientation
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    roll = math.atan2(sinr, cosr)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    twist = message.twist.twist
    return {
        "x_m": float(position.x), "y_m": float(position.y), "z_m": float(position.z),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "linear_velocity_m_s": [float(twist.linear.x), float(twist.linear.y), float(twist.linear.z)],
        "angular_velocity_rad_s": [float(twist.angular.x), float(twist.angular.y), float(twist.angular.z)],
        "frame_id": str(message.header.frame_id),
        "child_frame_id": str(message.child_frame_id),
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_snapshot_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise OSError("refusing to replace a snapshot symlink")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def classify_reset_effect(
    pre_reset_snapshot: Mapping[str, Any] | None,
    post_reset_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require a reset-generation or physical-pose discontinuity before PASS.

    A verified click alone does not prove the simulator consumed the native
    Reset. If initial and reset states are indistinguishable, this intentionally
    reports an unresolved result instead of treating upright telemetry as proof.
    """
    pre_pose = pre_reset_snapshot.get("odom_pose") if isinstance(pre_reset_snapshot, Mapping) else None
    pre_stamps = pre_reset_snapshot.get("latest_source_stamp_ns") if isinstance(pre_reset_snapshot, Mapping) else None
    pre_stamp = pre_stamps.get("odom") if isinstance(pre_stamps, Mapping) else None
    first_post = post_reset_rows[0] if post_reset_rows else None
    last_post = post_reset_rows[-1] if post_reset_rows else None
    first_stamp = first_post.get("source_stamp_ns") if isinstance(first_post, Mapping) else None
    post_pose = last_post.get("pose") if isinstance(last_post, Mapping) else None
    source_stamp_rebased = (
        isinstance(pre_stamp, int)
        and isinstance(first_stamp, int)
        and first_stamp < pre_stamp
    )
    source_stamp_unchanged = (
        isinstance(pre_stamp, int)
        and isinstance(first_stamp, int)
        and first_stamp == pre_stamp
    )
    source_stamp_advanced = (
        isinstance(pre_stamp, int)
        and isinstance(first_stamp, int)
        and first_stamp > pre_stamp
    )
    pose_delta: dict[str, float] | None = None
    pose_changed = False
    if isinstance(pre_pose, Mapping) and isinstance(post_pose, Mapping):
        try:
            dx = float(post_pose["x_m"]) - float(pre_pose["x_m"])
            dy = float(post_pose["y_m"]) - float(pre_pose["y_m"])
            dz = float(post_pose["z_m"]) - float(pre_pose["z_m"])
            dyaw = math.atan2(
                math.sin(float(post_pose["yaw_rad"]) - float(pre_pose["yaw_rad"])),
                math.cos(float(post_pose["yaw_rad"]) - float(pre_pose["yaw_rad"])),
            )
            position_delta = math.sqrt(dx * dx + dy * dy + dz * dz)
            pose_delta = {
                "x_m": dx,
                "y_m": dy,
                "z_m": dz,
                "yaw_rad": dyaw,
                "position_norm_m": position_delta,
            }
            pose_changed = (
                position_delta >= RESET_EFFECT_MIN_POSITION_DELTA_M
                or abs(dyaw) >= RESET_EFFECT_MIN_YAW_DELTA_RAD
            )
        except (KeyError, TypeError, ValueError):
            pose_delta = None
    observed = source_stamp_rebased or pose_changed
    return {
        "status": "RESET_EFFECT_OBSERVED" if observed else "RESET_EFFECT_UNCONFIRMED",
        "source_stamp_rebased": source_stamp_rebased,
        "source_stamp_unchanged_duplicate": source_stamp_unchanged,
        "source_stamp_advanced_without_generation_proof": source_stamp_advanced,
        "physical_pose_change_observed": pose_changed,
        "last_pre_reset_odom_stamp_ns": pre_stamp,
        "first_post_reset_odom_stamp_ns": first_stamp,
        "pre_to_last_post_pose_delta": pose_delta,
        "minimum_position_delta_m": RESET_EFFECT_MIN_POSITION_DELTA_M,
        "minimum_yaw_delta_rad": RESET_EFFECT_MIN_YAW_DELTA_RAD,
        "used_as_pass_gate": True,
    }


class ResetOnlyObserver(Node):
    def __init__(self, output_dir: Path, reset_event_path: Path):
        # This observer must have no implicit rosout publisher or parameter
        # service surface: it is subscriptions-only by construction.
        super().__init__(
            "issue80_raicom_reset_only_observer",
            enable_rosout=False,
            start_parameter_services=False,
        )
        # Humble rclpy still creates /parameter_events even when parameter
        # services are disabled. The observer has no parameters to announce,
        # so remove that framework publisher before creating subscriptions.
        parameter_events = getattr(self, "_parameter_event_publisher", None)
        if parameter_events is not None:
            self.destroy_publisher(parameter_events)
            self._parameter_event_publisher = None
        self.output_dir = output_dir
        self.reset_event_path = reset_event_path
        self.trace_path = output_dir / "reset_only_state_trace.jsonl"
        if self.trace_path.exists():
            raise FileExistsError("refusing to overwrite Reset-only state trace")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.latest: dict[str, dict[str, Any]] = {}
        self.latest_messages: dict[str, Any] = {}
        self.received_ns: dict[str, int] = {}
        self.last_odom_stamp_ns: int | None = None
        self.reset_event: dict[str, Any] | None = None
        self.reset_ns: int | None = None
        self.started_ns = time.monotonic_ns()
        self.ready_ns: int | None = None
        self.reset_event_deadline_ns: int | None = None
        self.post_reset_deadline_ns: int | None = None
        self.first_upright_ns: int | None = None
        self.upright_streak = 0
        self.upright_max_streak = 0
        self.unique_odom_post_reset = 0
        self.duplicate_odom_post_reset = 0
        self.regressing_odom_post_reset = 0
        self.failure: str | None = None
        self.complete = False
        self.post_reset_rows: list[dict[str, Any]] = []
        self.pre_reset_snapshot: dict[str, Any] | None = None
        self.subscription_handles = [
            self.create_subscription(message_type, topic, self._callback(key), self.qos)
            for key, (message_type, topic) in TOPIC_TYPES.items()
        ]

    def interface_counts(self) -> dict[str, Any]:
        """Expose the actual rclpy entity counts for the construction smoke test."""
        subscriptions = list(self.subscriptions)
        publishers = list(self.publishers)
        clients = list(self.clients)
        return {
            "subscriptions": len(self.subscription_handles),
            "node_subscriptions": len(subscriptions),
            "publishers": len(publishers),
            "service_clients": len(clients),
            "publisher_topics": [str(getattr(item, "topic_name", "")) for item in publishers],
            "service_client_names": [str(getattr(item, "srv_name", "")) for item in clients],
        }

    def _callback(self, key: str):
        def receive(message: Any) -> None:
            receipt_ns = time.monotonic_ns()
            stamp_ns = _stamp_ns(message)
            record = {
                "event": "state_message",
                "topic_key": key,
                "topic": TOPIC_TYPES[key][1],
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "receipt_monotonic_ns": receipt_ns,
                "source_stamp_ns": stamp_ns,
                "source_stamp_valid": stamp_ns is not None,
                "raw": _safe(message),
            }
            if key == "odom":
                record["derived_pose"] = _odom_pose(message)
            self._write(record)
            self.latest[key] = record
            self.latest_messages[key] = message
            self.received_ns[key] = receipt_ns
            if key == "odom":
                self._consider_odom(message, receipt_ns, stamp_ns)

        return receive

    def _write(self, record: Mapping[str, Any]) -> None:
        self.trace.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _ages(self, now_ns: int | None = None) -> dict[str, float | None]:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return {
            key: (now - self.received_ns[key]) / 1e9 if key in self.received_ns else None
            for key in TOPIC_TYPES
        }

    def _snapshot(self, now_ns: int | None = None) -> dict[str, Any]:
        now = time.monotonic_ns() if now_ns is None else now_ns
        common = self.latest_messages.get("common")
        odom = self.latest_messages.get("odom")
        raw_common = _safe(common) if common is not None else None
        action_info = getattr(common, "action_info", None) if common is not None else None
        fsm = getattr(common, "fsm_state", None) if common is not None else None
        body = getattr(common, "body_status", None) if common is not None else None
        return {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "receipt_monotonic_ns": now,
            "state_age_s": self._ages(now),
            "latest_source_stamp_ns": {
                key: item.get("source_stamp_ns") for key, item in self.latest.items()
            },
            "odom_pose": _odom_pose(odom) if odom is not None else None,
            "common_raw": raw_common,
            "action_info_raw": _safe(action_info) if action_info is not None else None,
            "fsm_state_raw": _safe(fsm) if fsm is not None else None,
            "body_status_raw": _safe(body) if body is not None else None,
            "imu_raw": _safe(self.latest_messages.get("imu")) if "imu" in self.latest_messages else None,
            "leg_state_raw": _safe(self.latest_messages.get("leg_state")) if "leg_state" in self.latest_messages else None,
            "leg_command_raw": _safe(self.latest_messages.get("leg_command")) if "leg_command" in self.latest_messages else None,
        }

    def _fresh(self, now_ns: int) -> bool:
        ages = self._ages(now_ns)
        return all(age is not None and 0.0 <= age <= MAX_STATE_AGE_S for age in ages.values())

    def _load_reset_event(self) -> None:
        if self.reset_event is not None or not self.reset_event_path.is_file():
            return
        try:
            payload = json.loads(self.reset_event_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        event = payload.get("event") if isinstance(payload, Mapping) else None
        if not isinstance(event, Mapping):
            self.failure = "RESET_EVENT_SCHEMA_INVALID"
            return
        self.reset_event = dict(payload)
        status = event.get("status")
        dispatch_ns = event.get("dispatch_monotonic_ns")
        if (
            status != "RESET_BUTTON_CLICKED"
            or not isinstance(dispatch_ns, int)
            or int(payload.get("reset_delivery_attempt_count", 0)) != 1
            or int(payload.get("reset_input_path_attempt_count", 0)) != 1
            or int(event.get("mouse_button_events_sent", 0)) != 2
            or int(event.get("key_events_sent", 0)) != 0
        ):
            self.failure = "RESET_EVENT_NOT_EXACTLY_ONE_VERIFIED_NATIVE_BUTTON_CLICK"
            return
        pre_path = self.output_dir / "reset_only_dispatch_pre_snapshot.json"
        try:
            pre_payload = json.loads(pre_path.read_text(encoding="utf-8"))
            pre_ns = int(pre_payload["snapshot_monotonic_ns"])
            pre_snapshot = pre_payload["snapshot"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.failure = "DISPATCH_PRE_RESET_SNAPSHOT_MISSING_OR_INVALID"
            return
        if pre_ns >= dispatch_ns or not isinstance(pre_snapshot, Mapping):
            self.failure = "DISPATCH_PRE_RESET_SNAPSHOT_NOT_BEFORE_RESET_EVENT"
            return
        self.pre_reset_snapshot = dict(pre_snapshot)
        self.reset_ns = dispatch_ns

    def _consider_odom(self, message: Odometry, receipt_ns: int, stamp_ns: int | None) -> None:
        if self.reset_ns is None or receipt_ns <= self.reset_ns:
            return
        event = self.latest.get("odom")
        pose = event.get("derived_pose") if isinstance(event, Mapping) else None
        if stamp_ns is None:
            self.failure = "POST_RESET_ODOM_SOURCE_STAMP_MISSING"
            return
        if self.last_odom_stamp_ns is not None:
            if stamp_ns == self.last_odom_stamp_ns:
                self.duplicate_odom_post_reset += 1
                self.upright_streak = 0
                return
            if stamp_ns < self.last_odom_stamp_ns:
                self.regressing_odom_post_reset += 1
                self.upright_streak = 0
                return
        self.last_odom_stamp_ns = stamp_ns
        self.unique_odom_post_reset += 1
        if not isinstance(pose, Mapping) or not self._fresh(receipt_ns):
            self.upright_streak = 0
            return
        tilt_ok = max(abs(float(pose["roll_rad"])), abs(float(pose["pitch_rad"]))) < MAX_ABS_TILT_RAD
        height_ok = float(pose["z_m"]) >= MIN_ROOT_Z_M
        safe = height_ok and tilt_ok
        if safe:
            self.upright_streak += 1
            self.upright_max_streak = max(self.upright_max_streak, self.upright_streak)
            if self.first_upright_ns is None and self.upright_streak >= REQUIRED_UPRIGHT_ODOM_SAMPLES:
                self.first_upright_ns = receipt_ns
        else:
            self.upright_streak = 0
            if self.first_upright_ns is not None:
                self.failure = "POST_RESET_UPRIGHT_POSTURE_LOST"
        if self.reset_ns is not None:
            row = {
                "receipt_monotonic_ns": receipt_ns,
                "source_stamp_ns": stamp_ns,
                "pose": dict(pose),
                "fresh_all_required_topics": self._fresh(receipt_ns),
                "upright": safe,
                "upright_streak": self.upright_streak,
                "raw_common_state": _safe(self.latest_messages.get("common")) if "common" in self.latest_messages else None,
                "imu": _safe(self.latest_messages.get("imu")) if "imu" in self.latest_messages else None,
                "leg_state": _safe(self.latest_messages.get("leg_state")) if "leg_state" in self.latest_messages else None,
                "leg_command": _safe(self.latest_messages.get("leg_command")) if "leg_command" in self.latest_messages else None,
            }
            self.post_reset_rows.append(row)
            self._write({"event": "post_reset_odom_sample", **row})

    def _finish(self, status: str, reason: str | None = None) -> None:
        if self.complete:
            return
        self.failure = reason or self.failure
        now_ns = time.monotonic_ns()
        reset_event = self.reset_event or {}
        event = reset_event.get("event") if isinstance(reset_event, Mapping) else None
        dispatch_ns = self.reset_ns
        post_rows = [row for row in self.post_reset_rows if dispatch_ns is not None and int(row["receipt_monotonic_ns"]) > dispatch_ns]
        pre = self.pre_reset_snapshot
        reset_effect = classify_reset_effect(pre, post_rows)
        last_pre_stamp = None
        if isinstance(pre, Mapping):
            stamps = pre.get("latest_source_stamp_ns")
            if isinstance(stamps, Mapping):
                last_pre_stamp = stamps.get("odom")
        first_post_stamp = post_rows[0].get("source_stamp_ns") if post_rows else None
        summary = {
            "schema": "robotsim.issue80.raicom-reset-only-observer.v1",
            "observer_status": status,
            "failure": self.failure,
            "started_monotonic_ns": self.started_ns,
            "ready_monotonic_ns": self.ready_ns,
            "reset_dispatch_monotonic_ns": dispatch_ns,
            "completed_monotonic_ns": now_ns,
            "observed_duration_s": (now_ns - self.started_ns) / 1e9,
            "post_reset_duration_s": (now_ns - dispatch_ns) / 1e9 if dispatch_ns else None,
            "post_reset_source_stamp_counts": {
                "unique": self.unique_odom_post_reset,
                "duplicate": self.duplicate_odom_post_reset,
                "regressing": self.regressing_odom_post_reset,
            },
            "odom_source_stamp_across_reset": {
                "last_pre_reset_stamp_ns": last_pre_stamp,
                "first_post_reset_stamp_ns": first_post_stamp,
                "first_post_stamp_rebased": (
                    isinstance(last_pre_stamp, int)
                    and isinstance(first_post_stamp, int)
                    and first_post_stamp < last_pre_stamp
                ),
                "first_post_stamp_unchanged_duplicate": (
                    isinstance(last_pre_stamp, int)
                    and isinstance(first_post_stamp, int)
                    and first_post_stamp == last_pre_stamp
                ),
                "first_post_stamp_advanced_without_generation_proof": (
                    isinstance(last_pre_stamp, int)
                    and isinstance(first_post_stamp, int)
                    and first_post_stamp > last_pre_stamp
                ),
                "classification": "DUPLICATES_DO_NOT_PROVE_RESET_GENERATION",
            },
            "post_reset_upright_status": (
                "PASS_UPRIGHT_SAMPLES_OBSERVED" if self.first_upright_ns is not None
                else "FAIL_NO_UPRIGHT_STREAK"
            ),
            "first_upright_monotonic_ns": self.first_upright_ns,
            "upright_latency_s": (
                (self.first_upright_ns - dispatch_ns) / 1e9
                if self.first_upright_ns is not None and dispatch_ns is not None else None
            ),
            "maximum_consecutive_upright_odom_samples": self.upright_max_streak,
            "pre_reset_snapshot": pre,
            "post_reset_first_odom": post_rows[0] if post_rows else None,
            "post_reset_first_fresh_upright_odom": next(
                (row for row in post_rows if row.get("upright")
                 and row.get("fresh_all_required_topics")), None
            ),
            "post_reset_last_odom": post_rows[-1] if post_rows else None,
            "reset_effect": reset_effect,
            "raw_reset_event": reset_event,
            "required_topics": {key: topic for key, (_, topic) in TOPIC_TYPES.items()},
            "state_freshness_limit_s": MAX_STATE_AGE_S,
            "upright_gate": {
                "minimum_root_z_m": MIN_ROOT_Z_M,
                "max_abs_roll_pitch_rad_exclusive": MAX_ABS_TILT_RAD,
                "consecutive_advancing_odom_samples": REQUIRED_UPRIGHT_ODOM_SAMPLES,
                "no_stand_or_locomotion_request": True,
            },
            "data_source_limitations": [
                "No separate MuJoCo qpos/body-state ROS endpoint is exposed to this observer; /aima/hal/odom/state is the simulator model-manager pelvis-body readout configured as odom.name=pelvis.",
                "No imu_1 sensor is synthesized; the observer records only the existing torso IMU topic.",
                "FSM, body status, and action status are preserved raw and are not reinterpreted as standing/safe enums.",
                "Reset dispatch is not treated as proof that MuJoCo processed it; state change is inferred only from time-aligned telemetry.",
            ],
        }
        summary_path = self.output_dir / "reset_only_observer_summary.json"
        _write_new_json(summary_path, summary)
        self.trace.flush()
        os.fsync(self.trace.fileno())
        self.complete = True

    def run_capture(self) -> int:
        ready_deadline = self.started_ns + int(MAX_READY_WAIT_S * 1e9)
        total_deadline = self.started_ns + int(MAX_OBSERVER_TOTAL_S * 1e9)
        while rclpy.ok() and not self.complete:
            rclpy.spin_once(self, timeout_sec=0.05)
            now_ns = time.monotonic_ns()
            if self.ready_ns is not None and self.reset_ns is None:
                latest_path = self.output_dir / "reset_only_latest_snapshot.json"
                _write_snapshot_replace(latest_path, {
                    "schema": "robotsim.issue80.raicom-reset-only-latest-snapshot.v1",
                    "snapshot": self._snapshot(now_ns),
                })
                stale_pre = [
                    key for key, age in self._ages(now_ns).items()
                    if age is None or age > MAX_STATE_AGE_S
                ]
                if stale_pre:
                    self._finish("BLOCKED", "PRE_RESET_REQUIRED_TELEMETRY_BECAME_STALE:" + ",".join(stale_pre))
                    break
            if self.ready_ns is None:
                if len(self.latest) == len(TOPIC_TYPES) and self._fresh(now_ns):
                    self.ready_ns = now_ns
                    self.pre_reset_snapshot = self._snapshot(now_ns)
                    _write_new_json(self.output_dir / "reset_watch_ready.json", {
                        "schema": "robotsim.issue80.raicom-reset-only-ready.v1",
                        "status": "READY",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "ready_monotonic_ns": now_ns,
                        "state_age_s": self._ages(now_ns),
                        "snapshot": self.pre_reset_snapshot,
                        "publishers_created": 0,
                        "service_clients_created": 0,
                    })
                    self.pre_reset_snapshot = self._snapshot(now_ns)
                    _write_new_json(self.output_dir / "reset_only_pre_reset_snapshot.json", {
                        "schema": "robotsim.issue80.raicom-reset-only-pre-reset.v1",
                        "snapshot_monotonic_ns": now_ns,
                        "snapshot": self.pre_reset_snapshot,
                    })
                elif now_ns >= ready_deadline:
                    self._finish("BLOCKED", "PRE_RESET_REQUIRED_TELEMETRY_NOT_FRESH")
                    break
            self._load_reset_event()
            if self.failure and self.reset_event is not None and self.reset_ns is None:
                self._finish("BLOCKED", self.failure)
                break
            if self.reset_ns is not None:
                if self.post_reset_deadline_ns is None:
                    self.post_reset_deadline_ns = self.reset_ns + int(POST_RESET_OBSERVATION_S * 1e9)
                ages = self._ages(now_ns)
                stale = [key for key, age in ages.items() if age is None or age > MAX_STATE_AGE_S]
                if stale:
                    self._finish("BLOCKED", "POST_RESET_STALE_TELEMETRY:" + ",".join(stale))
                    break
                if self.failure:
                    self._finish("BLOCKED", self.failure)
                    break
                if now_ns >= self.post_reset_deadline_ns:
                    if self.first_upright_ns is None:
                        self._finish("BLOCKED", "POST_RESET_UPRIGHT_NOT_CONFIRMED_WITHIN_BOUNDED_WINDOW")
                    elif classify_reset_effect(self.pre_reset_snapshot, self.post_reset_rows)["status"] != "RESET_EFFECT_OBSERVED":
                        self._finish("BLOCKED", "RESET_EFFECT_UNCONFIRMED")
                    else:
                        self._finish("CAPTURE_COMPLETE", None)
                    break
            elif self.ready_ns is not None and self.reset_event_deadline_ns is None:
                self.reset_event_deadline_ns = self.ready_ns + int(MAX_RESET_EVENT_WAIT_S * 1e9)
                if now_ns >= self.reset_event_deadline_ns:
                    self._finish("BLOCKED", "VERIFIED_RESET_EVENT_NOT_RECEIVED")
                    break
            if now_ns >= total_deadline and not self.complete:
                self._finish("BLOCKED", "OBSERVER_TOTAL_RUNTIME_LIMIT_REACHED")
                break
        if not self.complete:
            self._finish("BLOCKED", "ROS_CONTEXT_STOPPED_BEFORE_CAPTURE_COMPLETED")
        return 0 if self.failure is None and self.first_upright_ns is not None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reset-event-file", required=True, type=Path)
    parser.add_argument(
        "--construction-smoke",
        action="store_true",
        help="construct and tear down the subscriptions-only observer; do not capture or wait for Reset",
    )
    parser.add_argument(
        "--inject-construction-failure-for-test",
        action="store_true",
        help="test-only construction-smoke failure path; requires --construction-smoke",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    reset_event = args.reset_event_file.resolve(strict=False)
    if output.is_symlink() or not output.is_dir():
        parser.error("output directory must be an existing regular directory")
    if reset_event != output / "reset_dispatch_event.json":
        parser.error("reset event must be the single canonical event file in the evidence directory")
    if reset_event.exists():
        parser.error("refusing to start after a Reset event already exists")
    if args.inject_construction_failure_for_test and not args.construction_smoke:
        parser.error("--inject-construction-failure-for-test requires --construction-smoke")
    rclpy.init()
    observer: ResetOnlyObserver | None = None
    try:
        observer = ResetOnlyObserver(output, reset_event)
        if args.construction_smoke:
            if args.inject_construction_failure_for_test:
                raise RuntimeError("injected observer construction-smoke failure")
            counts = observer.interface_counts()
            expected_subscriptions = len(TOPIC_TYPES)
            counts_ok = (
                counts["subscriptions"] == expected_subscriptions
                and counts["node_subscriptions"] == expected_subscriptions
                and counts["publishers"] == 0
                and counts["service_clients"] == 0
            )
            status = "PASS_CONSTRUCTION_SMOKE" if counts_ok else "BLOCKED_CONSTRUCTION_INTERFACE_COUNTS"
            _write_new_json(output / "reset_only_observer_smoke.json", {
                "schema": "robotsim.issue80.raicom-reset-only-observer-smoke.v1",
                "status": status,
                "interface_counts": counts,
                "required_subscription_topics": {
                    key: topic for key, (_, topic) in TOPIC_TYPES.items()
                },
                "velocity_command_source_created": False,
                "service_action_or_input_source_created": False,
                "native_reset_dispatched": False,
                "locomotion_or_simulation_started": False,
            })
            return 0 if status == "PASS_CONSTRUCTION_SMOKE" else 2
        return observer.run_capture()
    except Exception as exc:
        try:
            _write_new_json(output / "reset_only_observer_startup_error.json", {
                "schema": "robotsim.issue80.raicom-reset-only-observer-error.v1",
                "status": "BLOCKED_OBSERVER_STARTUP",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": time.monotonic_ns(),
                "reset_dispatched_by_observer": False,
            })
        except Exception:
            pass
        return 2
    finally:
        if observer is not None:
            observer.trace.close()
            observer.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
