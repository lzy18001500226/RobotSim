#!/usr/bin/env python3
"""Passive capture of version-matched MC observation/action diagnostics."""

from __future__ import annotations

import argparse
import json
import rclpy
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aimdk_msgs.msg import McRLDebug
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from issue80_m1_first_motion import mc_rl_debug_snapshot

MC_RL_DEBUG_TOPIC = "/aima/mc/rl/debug"
MC_RL_DEBUG_TYPE = "aimdk_msgs/msg/McRLDebug"


class McRlDebugCapture(Node):
    """Read-only sidecar; it owns no locomotion publisher or service client."""

    def __init__(self, output_path: Path, ready_path: Path, summary_path: Path, args: argparse.Namespace):
        super().__init__("issue80_mc_rl_debug_capture")
        self.output_path = output_path.resolve()
        self.ready_path = ready_path.resolve()
        self.summary_path = summary_path.resolve()
        for path in (self.output_path, self.ready_path, self.summary_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite MC debug evidence: {path.name}")
        self.trace = self.output_path.open("x", encoding="utf-8", buffering=1)
        self.stop_event = threading.Event()
        self.started_monotonic_s = time.monotonic()
        self.deadline_monotonic_s = self.started_monotonic_s + args.max_duration_s
        self.message_count = 0
        self.vector_lengths: dict[str, dict[int, int]] = {"obs_vec": {}, "action_vec": {}}
        self.first_receive_monotonic_s: float | None = None
        self.last_receive_monotonic_s: float | None = None
        self.endpoint_records: list[dict[str, Any]] = []
        self.ready_timeout_s = args.ready_timeout_s
        self.ready_status = "WAITING_FOR_SINGLE_MC_PUBLISHER"
        self.fault: str | None = None
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            McRLDebug,
            MC_RL_DEBUG_TOPIC,
            self._on_debug,
            self.qos,
        )

    def _debug_topic_publishers(self) -> list[dict[str, Any]]:
        endpoints = self.get_publishers_info_by_topic(MC_RL_DEBUG_TOPIC)
        records = []
        for endpoint in endpoints:
            qos = endpoint.qos_profile
            records.append({
                "node_name": str(endpoint.node_name),
                "node_namespace": str(endpoint.node_namespace),
                "topic_type": str(endpoint.topic_type),
                "reliability": int(qos.reliability),
                "durability": int(qos.durability),
                "history": int(qos.history),
                "depth": int(qos.depth),
            })
        self.endpoint_records = records
        return records

    def wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.ready_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            records = self._debug_topic_publishers()
            matching = [record for record in records if record["topic_type"] == MC_RL_DEBUG_TYPE]
            if len(records) == 1 and len(matching) == 1:
                self.ready_status = "READY"
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.ready_status != "READY":
            self.ready_status = "NO_SINGLE_VERSION_MATCHED_MC_RL_DEBUG_PUBLISHER"
        return self._write_ready_marker()

    def _write_ready_marker(self) -> dict[str, Any]:
        record = {
            "schema": "robotsim.issue80.mc-rl-debug-capture-ready.v1",
            "status": self.ready_status,
            "topic": MC_RL_DEBUG_TOPIC,
            "expected_type": MC_RL_DEBUG_TYPE,
            "publisher_count": len(self.endpoint_records),
            "publishers": self.endpoint_records,
            "observer_node": self.get_name(),
            "fault": self.fault,
            "ready_wall_utc": datetime.now(timezone.utc).isoformat(),
            "ready_monotonic_s": time.monotonic(),
        }
        with self.ready_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        return record

    def _on_debug(self, message: McRLDebug) -> None:
        received = time.monotonic()
        snapshot = mc_rl_debug_snapshot(message, received)
        if snapshot is None:
            return
        self.message_count += 1
        if self.first_receive_monotonic_s is None:
            self.first_receive_monotonic_s = received
        self.last_receive_monotonic_s = received
        for name in ("obs_vec", "action_vec"):
            length = len(snapshot[name])
            sizes = self.vector_lengths[name]
            sizes[length] = sizes.get(length, 0) + 1
        row = {
            "event": "mc_rl_debug",
            "topic": MC_RL_DEBUG_TOPIC,
            "type": MC_RL_DEBUG_TYPE,
            "payload": snapshot,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_sec": received,
        }
        self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def close(self) -> dict[str, Any]:
        if not self.trace.closed:
            self.trace.flush()
            self.trace.close()
        summary = {
            "schema": "robotsim.issue80.mc-rl-debug-capture-summary.v1",
            "status": "CAPTURED" if self.message_count else "NO_DEBUG_MESSAGES",
            "topic": MC_RL_DEBUG_TOPIC,
            "type": MC_RL_DEBUG_TYPE,
            "message_count": self.message_count,
            "vector_lengths": self.vector_lengths,
            "first_receive_monotonic_s": self.first_receive_monotonic_s,
            "last_receive_monotonic_s": self.last_receive_monotonic_s,
            "duration_s": time.monotonic() - self.started_monotonic_s,
            "ready_status": self.ready_status,
            "fault": self.fault,
            "trace_path": str(self.output_path),
            "interpretation_limit": (
                "Only the installed introspection field schema is asserted; observation/action index semantics "
                "must be mapped to the matching FoundationWalk configuration."
            ),
        }
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ready-marker", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ready-timeout-s", type=float, default=5.0)
    parser.add_argument("--max-duration-s", type=float, default=80.0)
    args = parser.parse_args()
    if not 1.0 <= args.ready_timeout_s <= 10.0:
        parser.error("ready timeout must be in 1..10 seconds")
    if not 5.0 <= args.max_duration_s <= 90.0:
        parser.error("capture duration must be in 5..90 seconds")
    rclpy.init()
    node = McRlDebugCapture(Path(args.output), Path(args.ready_marker), Path(args.summary), args)
    previous = {}

    def stop_handler(signum, frame):
        node.stop_event.set()

    previous[signal.SIGINT] = signal.signal(signal.SIGINT, stop_handler)
    previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, stop_handler)
    try:
        ready = node.wait_ready()
        print(json.dumps(ready, separators=(",", ":")), flush=True)
        if ready["status"] != "READY":
            return 2
        while rclpy.ok() and not node.stop_event.is_set() and time.monotonic() < node.deadline_monotonic_s:
            rclpy.spin_once(node, timeout_sec=0.05)
        return 0
    except Exception as exc:
        node.fault = f"{type(exc).__name__}: {exc}"
        node.ready_status = "OBSERVER_ERROR"
        if not node.ready_path.exists():
            node._write_ready_marker()
        print(node.fault, flush=True)
        return 3
    finally:
        summary = node.close()
        print(json.dumps(summary, separators=(",", ":")), flush=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
