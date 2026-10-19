#!/usr/bin/env python3
"""Bounded read-only ROS state recorder around one automated MuJoCo Reset."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import JointCommandArray, JointStateArray, McCommonState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


TOPICS = (
    ("odom", "/aima/hal/odom/state", Odometry),
    ("imu", "/aima/hal/imu/torso/state", Imu),
    ("common", "/aima/mc/common/state", McCommonState),
    ("leg_state", "/aima/hal/joint/leg/state", JointStateArray),
    ("leg_command", "/aima/hal/joint/leg/command", JointCommandArray),
)


def json_safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {name: json_safe(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def pose_summary(msg: Odometry) -> dict[str, float | int]:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    sinp = 2 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return {
        "x": float(p.x), "y": float(p.y), "z": float(p.z),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),
        "linear_velocity": json_safe(msg.twist.twist.linear),
        "angular_velocity": json_safe(msg.twist.twist.angular),
        "stamp_sec": int(msg.header.stamp.sec),
        "stamp_nanosec": int(msg.header.stamp.nanosec),
    }


class ResetStateTrace(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("issue80_m1_reset_state_trace")
        if not 4.0 <= args.duration <= 10.0:
            raise ValueError("observer duration must be bounded to 4..10 seconds")
        self.args = args
        self.output_dir = Path(args.output_dir).resolve(strict=True)
        self.trace_path = self.output_dir / "reset_state_trace.jsonl"
        self.summary_path = self.output_dir / "reset_state_summary.json"
        self.ready_path = self.output_dir / "reset_watch_ready.json"
        if any(p.exists() for p in (self.trace_path, self.summary_path, self.ready_path)):
            raise FileExistsError("refusing to overwrite reset observer evidence")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        self.latest: dict[str, Any] = {}
        self.received: dict[str, float] = {}
        self.counts = {name: 0 for name, _, _ in TOPICS}
        self.clocks: list[dict[str, Any]] = []
        self.interrupted = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for name, topic, msg_type in TOPICS:
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, n=name: self._store(n, msg),
                qos,
            )
        self.create_subscription(Clock, "/clock", self._clock_cb, qos)

    def _store(self, name: str, msg: Any) -> None:
        now = time.monotonic()
        self.latest[name] = msg
        self.received[name] = now
        self.counts[name] += 1
        row = {
            "event": "state",
            "topic_key": name,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_ns": time.monotonic_ns(),
            "header_stamp": json_safe(msg.header.stamp) if hasattr(msg, "header") else None,
            "data": pose_summary(msg) if name == "odom" else json_safe(msg),
        }
        self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def _clock_cb(self, msg: Clock) -> None:
        row = {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_ns": time.monotonic_ns(),
            "clock": json_safe(msg.clock),
        }
        self.clocks.append(row)
        self.trace.write(json.dumps({"event": "clock", **row}, separators=(",", ":")) + "\n")

    def _fresh_keys(self) -> list[str]:
        now = time.monotonic()
        return [
            name for name, _, _ in TOPICS
            if name in self.received and now - self.received[name] <= 0.5
        ]

    def _ready_snapshot(self) -> dict[str, Any]:
        common = self.latest.get("common")
        odom = self.latest.get("odom")
        return {
            "schema": "robotsim.issue80.reset-watch-ready.v1",
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_ns": time.monotonic_ns(),
            "fresh_topics": self._fresh_keys(),
            "mode": common.action_info.action_desc if common else None,
            "action_info": json_safe(common.action_info) if common else None,
            "input_source": json_safe(common.input_source) if common else None,
            "fsm_state": int(common.fsm_state.current_state) if common else None,
            "body_status": int(common.body_status.value) if common else None,
            "odom": pose_summary(odom) if odom else None,
            "sample_counts": dict(self.counts),
        }

    def run(self) -> dict[str, Any]:
        start = time.monotonic()
        end = start + self.args.duration
        ready_written = False
        while rclpy.ok() and time.monotonic() < end and not self.interrupted:
            rclpy.spin_once(self, timeout_sec=0.005)
            if not ready_written and len(self._fresh_keys()) == len(TOPICS):
                ready = self._ready_snapshot()
                with self.ready_path.open("x", encoding="utf-8") as stream:
                    json.dump(ready, stream, indent=2)
                    stream.write("\n")
                print(json.dumps({"status": "TRACE_READY", "snapshot": ready}), flush=True)
                ready_written = True
        odom_samples = []
        # Re-read the just-written JSONL so the summary is based on the same durable trace.
        self.trace.flush()
        with self.trace_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "state" and row.get("topic_key") == "odom":
                    odom_samples.append(row)
        poses = [row["data"] for row in odom_samples]
        zs = [p["z"] for p in poses]
        summary = {
            "schema": "robotsim.issue80.reset-state-trace.v1",
            "observer_status": "TRACE_CAPTURED" if ready_written else "BLOCKED_NO_FRESH_GRAPH",
            "reset_classification": "NOT_CLASSIFIED_IN_OBSERVER",
            "trace_ready_observed": ready_written,
            "duration_limit_s": self.args.duration,
            "wall_elapsed_s": time.monotonic() - start,
            "sample_counts": dict(self.counts),
            "clock_messages": len(self.clocks),
            "clock_topic_exposed": bool(self.clocks),
            "odom_sample_count": len(poses),
            "odom_z_min_m": min(zs) if zs else None,
            "odom_z_max_m": max(zs) if zs else None,
            "odom_first": poses[0] if poses else None,
            "odom_last": poses[-1] if poses else None,
            "common_final": json_safe(self.latest.get("common")),
            "first_odom_wall_utc": odom_samples[0]["wall_utc"] if odom_samples else None,
            "last_odom_wall_utc": odom_samples[-1]["wall_utc"] if odom_samples else None,
            "limitations": [
                "Header timestamps are recorded as message fields; they are not assumed to be MuJoCo physics time.",
                "The trace observer is read-only and sends no reset key or robot command.",
                "The X11 key dispatch result must be correlated by wall and monotonic timestamps with this state trace.",
            ],
        }
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    rclpy.init()
    node = ResetStateTrace(args)
    previous = signal.signal(signal.SIGINT, lambda signum, frame: setattr(node, "interrupted", True))
    try:
        result = node.run()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    finally:
        signal.signal(signal.SIGINT, previous)
        node.trace.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
