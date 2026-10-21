#!/usr/bin/env python3
"""Read-only N2 telemetry recorder for the pinned MuJoCo + MC container."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import CommonRequest, JointCommandArray, JointStateArray, McCommonState
from aimdk_msgs.srv import GetCurrentInputSource, GetMcAction
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {name: safe(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def stamp(message: Any) -> dict[str, int] | None:
    header = getattr(message, "header", None)
    if header is None:
        return None
    return {"sec": int(header.stamp.sec), "nanosec": int(header.stamp.nanosec)}


def write_json_atomically_new(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite trace output: {path.name}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pose_data(message: Odometry) -> dict[str, Any]:
    p, q = message.pose.pose.position, message.pose.pose.orientation
    roll = math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y))
    sinp = 2 * (q.w*q.y - q.z*q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
    return safe({
        "x_m": float(p.x), "y_m": float(p.y), "z_m": float(p.z),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
        "linear_velocity_mps": safe(message.twist.twist.linear),
        "angular_velocity_radps": safe(message.twist.twist.angular),
        "header_stamp": stamp(message), "frame_id": str(message.header.frame_id),
    })


class RuntimeTrace(Node):
    def __init__(self, output_dir: Path, duration_s: float):
        super().__init__("issue80_n2_runtime_trace")
        self.output_dir = output_dir.resolve(strict=True)
        self.trace_path = self.output_dir / "n2_runtime_trace.jsonl"
        self.ready_path = self.output_dir / "n2_sim_ready.json"
        self.summary_path = self.output_dir / "n2_runtime_summary.json"
        if any(path.exists() for path in (self.trace_path, self.ready_path, self.summary_path,
                                          self.ready_path.with_name(self.ready_path.name + ".part"),
                                          self.summary_path.with_name(self.summary_path.name + ".part"))):
            raise FileExistsError("refusing to overwrite N2 trace files")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        self.start_mono = time.monotonic()
        self.end_mono = self.start_mono + duration_s
        self.latest: dict[str, dict[str, Any]] = {}
        self.counts = {key: 0 for key in ("odom", "imu", "leg_state", "leg_command", "common", "clock")}
        self.last_record_ns: dict[str, int] = {}
        self.period_ns = {"odom": 20_000_000, "imu": 50_000_000, "leg_state": 20_000_000,
                          "leg_command": 20_000_000, "common": 0, "clock": 20_000_000}
        self.fsm_transitions: list[dict[str, Any]] = []
        self.first_safe_state: dict[str, Any] | None = None
        self.first_moving_state: dict[str, Any] | None = None
        self.readonly_service_results: dict[str, Any] = {}
        self.readonly_service_status = {
            "get_mc_action": {"ready_observed": False, "request_sent": False, "response_received": False},
            "get_current_input_source": {"ready_observed": False, "request_sent": False, "response_received": False},
        }
        self.service_attempted: set[str] = set()
        self.pending: dict[str, Any] = {}
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Odometry, "/aima/hal/odom/state", lambda m: self.store("odom", m, pose_data(m)), qos)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", lambda m: self.store("imu", m, safe(m)), qos)
        self.create_subscription(JointStateArray, "/aima/hal/joint/leg/state", lambda m: self.store("leg_state", m, safe(m)), qos)
        self.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command", lambda m: self.store("leg_command", m, safe(m)), qos)
        self.create_subscription(McCommonState, "/aima/mc/common/state", self.on_common, qos)
        self.create_subscription(Clock, "/clock", self.on_clock, qos)
        self.get_mode = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
        self.get_source = self.create_client(GetCurrentInputSource, "/aimdk_5Fmsgs/srv/GetCurrentInputSource")
        self.create_timer(0.1, self.poll_readonly_services)

    def emit(self, row: dict[str, Any]) -> None:
        self.trace.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def store(self, key: str, message: Any, data: Any) -> None:
        now_ns = time.monotonic_ns()
        self.counts[key] += 1
        self.latest[key] = {"message": message, "data": data, "mono_ns": now_ns}
        period = self.period_ns[key]
        if now_ns - self.last_record_ns.get(key, 0) >= period:
            self.last_record_ns[key] = now_ns
            self.emit({"event": "state", "topic_key": key, "wall_utc": utc_now(),
                       "wall_monotonic_ns": now_ns, "header_stamp": stamp(message), "data": data})
        if not self.ready_path.exists() and all(key in self.latest for key in ("odom", "imu", "leg_state")):
            ready = {"schema": "robotsim.issue80.n2.simulator-ready.v1", "wall_utc": utc_now(),
                     "wall_monotonic_ns": now_ns, "fresh_sim_topics": ["odom", "imu", "leg_state"],
                     "odom": self.latest["odom"]["data"],
                     "imu_header_stamp": stamp(self.latest["imu"]["message"]),
                     "leg_state_header_stamp": stamp(self.latest["leg_state"]["message"]),
                     "sample_counts": dict(self.counts)}
            write_json_atomically_new(self.ready_path, ready)

    def on_common(self, message: McCommonState) -> None:
        previous = self.latest.get("common")
        fsm = int(message.fsm_state.current_state)
        mode = str(message.action_info.action_desc)
        data = {"fsm_state": fsm, "body_status": int(message.body_status.value),
                "action_desc": mode, "action_status": int(message.action_info.status.value),
                "input_source": str(message.input_source.name),
                "input_source_priority": int(message.input_source.priority),
                "input_source_timeout_ms": int(message.input_source.timeout),
                "speed_status": safe(message.speed_status), "full_message": safe(message)}
        now_ns = time.monotonic_ns()
        state_event = {"wall_utc": utc_now(), "wall_monotonic_ns": now_ns, "header_stamp": stamp(message),
                       "fsm_state": fsm, "body_status": int(message.body_status.value),
                       "action_desc": mode, "input_source": str(message.input_source.name)}
        if fsm == 4 and self.first_safe_state is None:
            self.first_safe_state = dict(state_event)
        if fsm == 3 and self.first_moving_state is None:
            self.first_moving_state = dict(state_event)
        if previous and (previous["data"]["fsm_state"] != fsm or previous["data"]["action_desc"] != mode):
            event = {"event": "state_transition", "wall_utc": utc_now(), "wall_monotonic_ns": time.monotonic_ns(),
                     "fsm_from": previous["data"]["fsm_state"], "fsm_to": fsm,
                     "mode_from": previous["data"]["action_desc"], "mode_to": mode}
            self.fsm_transitions.append(event)
            self.emit(event)
        self.store("common", message, data)

    def on_clock(self, message: Clock) -> None:
        self.counts["clock"] += 1
        now_ns = time.monotonic_ns()
        if now_ns - self.last_record_ns.get("clock", 0) < self.period_ns["clock"]:
            return
        self.last_record_ns["clock"] = now_ns
        clock_value = {"sec": int(message.clock.sec), "nanosec": int(message.clock.nanosec)}
        self.latest["clock"] = {"data": clock_value, "mono_ns": now_ns}
        self.emit({"event": "sim_clock", "wall_utc": utc_now(), "wall_monotonic_ns": now_ns, "clock": clock_value})

    def request_for(self, name: str, service_type: Any) -> Any:
        request = service_type.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        return request

    def poll_readonly_services(self) -> None:
        for name, client, service_type in (("get_mc_action", self.get_mode, GetMcAction),
                                           ("get_current_input_source", self.get_source, GetCurrentInputSource)):
            future = self.pending.get(name)
            if future is not None and future.done():
                try:
                    response = future.result()
                    data = safe(response)
                    if name == "get_mc_action":
                        data["interpreted"] = {"header_code": int(response.header.code),
                            "action_desc": str(response.info.action_desc), "action_status": int(response.info.status.value)}
                    else:
                        data["interpreted"] = {"header_code": int(response.response.header.code),
                            "task_state": int(response.response.state.value), "task_id": int(response.response.task_id),
                            "source_name": str(response.input_source.name), "source_priority": int(response.input_source.priority),
                            "source_timeout_ms": int(response.input_source.timeout)}
                    event = {"event": "readonly_service_response", "service": name, "wall_utc": utc_now(),
                             "wall_monotonic_ns": time.monotonic_ns(), "response": data}
                except Exception as exc:
                    event = {"event": "readonly_service_error", "service": name,
                             "error": f"{type(exc).__name__}: {exc}", "wall_utc": utc_now(),
                             "wall_monotonic_ns": time.monotonic_ns()}
                self.readonly_service_results[name] = event
                self.readonly_service_status[name]["response_received"] = event["event"] == "readonly_service_response"
                self.emit(event)
                self.pending.pop(name, None)
            ready = client.service_is_ready()
            self.readonly_service_status[name]["ready_observed"] |= bool(ready)
            if name not in self.service_attempted and ready:
                request = self.request_for(name, service_type)
                self.service_attempted.add(name)
                self.pending[name] = client.call_async(request)
                self.readonly_service_status[name]["request_sent"] = True
                self.emit({"event": "readonly_service_request", "service": name, "wall_utc": utc_now(),
                           "wall_monotonic_ns": time.monotonic_ns(), "request": safe(request)})

    def summarize(self) -> dict[str, Any]:
        self.trace.flush()
        first_common = None
        last_common = None
        first_odom = None
        last_odom = None
        first_joints = None
        last_joints = None
        clocks: list[float] = []
        with self.trace_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "sim_clock":
                    c = row["clock"]
                    clocks.append(c["sec"] + c["nanosec"] / 1e9)
                if row.get("event") != "state":
                    continue
                key = row.get("topic_key")
                if key == "common":
                    first_common = first_common or row
                    last_common = row
                elif key == "odom":
                    first_odom = first_odom or row
                    last_odom = row
                elif key == "leg_state":
                    first_joints = first_joints or row["data"]
                    last_joints = row["data"]
        stamp_delta = None
        if first_odom and last_odom and first_odom.get("header_stamp") and last_odom.get("header_stamp"):
            a, b = first_odom["header_stamp"], last_odom["header_stamp"]
            stamp_delta = (b["sec"] + b["nanosec"] / 1e9) - (a["sec"] + a["nanosec"] / 1e9)
        clock_delta = clocks[-1] - clocks[0] if len(clocks) >= 2 else None
        state_publish_duration = (
            (int(last_odom["wall_monotonic_ns"]) - int(first_odom["wall_monotonic_ns"])) / 1e9
            if first_odom and last_odom else None
        )
        return {"schema": "robotsim.issue80.n2.runtime-summary.v1", "wall_utc": utc_now(),
                "wall_observation_duration_s": time.monotonic() - self.start_mono,
                "message_callback_counts": dict(self.counts), "clock_sample_count": len(clocks),
                "sim_clock_first_s": clocks[0] if clocks else None, "sim_clock_last_s": clocks[-1] if clocks else None,
                "sim_clock_delta_s": clock_delta, "odom_header_stamp_delta_s": stamp_delta,
                "odom_header_stamp_time_domain": "not independently verified as MuJoCo time",
                "first_odom": first_odom.get("data") if first_odom else None,
                "last_odom": last_odom.get("data") if last_odom else None,
                "first_leg_state": first_joints, "last_leg_state": last_joints,
                "first_mc_common_state": first_common["data"] if first_common else None,
                "last_mc_common_state": last_common["data"] if last_common else None,
                "first_mc_common_state_event": first_common, "last_mc_common_state_event": last_common,
                "odom_state_publish_wall_span_s": state_publish_duration,
                "sim_clock_advancing_observed": clock_delta is not None and clock_delta > 0.0,
                "first_safe_state": self.first_safe_state, "first_moving_state": self.first_moving_state,
                "fsm_transitions": self.fsm_transitions,
                "readonly_service_results": self.readonly_service_results,
                "readonly_service_status": self.readonly_service_status,
                "limitations": ["No /clock is available in the previously inventoried pinned simulator graph; exact MuJoCo internal time/step count cannot be asserted unless messages appear in this run.",
                                "ROS header timestamp time domain is recorded but not equated with MuJoCo simulation time.",
                                "Observer is read-only and emits no MC command or mode/source mutation."]}

    def run(self) -> dict[str, Any]:
        while rclpy.ok() and time.monotonic() < self.end_mono:
            rclpy.spin_once(self, timeout_sec=0.01)
        summary = self.summarize()
        write_json_atomically_new(self.summary_path, summary)
        return summary


def trace_main(args: argparse.Namespace) -> int:
    if not 20 <= args.duration <= 55:
        raise ValueError("trace duration must be within 20..55 seconds")
    rclpy.init()
    node = RuntimeTrace(Path(args.output_dir), args.duration)
    try:
        summary = node.run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        node.trace.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=40.0)
    args = parser.parse_args()
    return trace_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
