#!/usr/bin/env python3
"""Bounded simulation-only Reset verification followed by the guarded Issue 80 M1 probe."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = REPO / "simulation/mujoco/raicom2026"
X11_HELPER = Path(__file__).with_name("issue80_mujoco_x11_reset.py")
ANALYZER = Path(__file__).with_name("issue80_m1_reset_analyze.py")
M1_RUNNER = Path(__file__).with_name("issue80_m1_locomotion_probe.py")
M1_MODE_LOGIC = Path(__file__).with_name("issue80_m1_mode_logic.py")
M1_FIRST_MOTION_LOGIC = Path(__file__).with_name("issue80_m1_first_motion.py")
SET_ACTION_CAPTURE = Path(__file__).with_name("issue80_m1_set_action_capture.py")
SIM_LAUNCH = Path(__file__).with_name("issue80_launch_sim.sh")
MC_LAUNCH = Path(__file__).with_name("issue80_launch_mc.sh")
IMAGE = "robotsim:humble-raicom-runtime-20261019"
IMAGE_DIGEST = "sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31"
RESET_OBSERVER_DURATION_S = 8.0
RESET_OBSERVER_COMPLETION_MARGIN_S = 3.0
POST_RESET_UPRIGHT_TIMEOUT_S = 2.0
TOPIC_PROBES = (
    "/aima/hal/odom/state",
    "/aima/hal/imu/torso/state",
    "/aima/hal/joint/leg/state",
)


class RunDeadline(TimeoutError):
    pass


M1_CHILD_RESULT_STATUSES = frozenset({"PASS", "BLOCKED"})
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def decode_m1_child_result(serialized: str) -> dict[str, Any]:
    """Decode one child result object, accepting one legacy string wrapper.

    The child currently writes a JSON string containing its Python mapping
    representation because its local _json_safe helper does not normalize
    Mapping instances. Prefer one extra JSON decode; retain a literal-only
    compatibility path for that observed representation. Never eval code.
    """
    try:
        value: Any = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"child result is not valid JSON: {exc}") from exc

    if isinstance(value, str):
        wrapped_value = value
        try:
            value = json.loads(wrapped_value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(wrapped_value)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise ValueError("child result string wrapper is not a mapping literal") from exc

    if not isinstance(value, Mapping):
        raise ValueError(f"child result must decode to a mapping, got {type(value).__name__}")
    result = dict(value)
    status = result.get("status")
    if not isinstance(status, str) or status not in M1_CHILD_RESULT_STATUSES:
        raise ValueError(f"child result has invalid status: {status!r}")
    return result


def classify_first_motion_outcome(activation: Mapping[str, Any] | None) -> str:
    """Keep measured motion distinct from an unverified post-stop gate."""
    if not isinstance(activation, Mapping):
        return "VELOCITY_COMMAND_NOT_EXECUTED"
    if int(activation.get("nonzero_publish_calls_returned", 0) or 0) <= 0:
        return "VELOCITY_COMMAND_NOT_EXECUTED"
    if activation.get("status") == "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED":
        return "FIRST_PHYSICAL_WALKING_PASS"
    if activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED":
        return "FIRST_PHYSICAL_WALKING_MOTION_CONFIRMED_STOP_UNVERIFIED"
    return "FIRST_PHYSICAL_WALKING_FAIL"


def assess_container_absence(
    *,
    inspect_returncode: int,
    list_returncode: int,
    list_stdout: str,
    container_name: str,
    container_id: str | None,
) -> dict[str, Any]:
    """Require inspect failure plus a successful exact name/ID inventory miss."""
    entries: list[tuple[str, str]] = []
    inventory_valid = bool(container_name)
    for line in list_stdout.splitlines():
        if not line:
            continue
        fields = line.split("\t", 1)
        if (
            len(fields) != 2
            or CONTAINER_ID_RE.fullmatch(fields[0]) is None
            or not fields[1]
        ):
            inventory_valid = False
            break
        entries.append((fields[0], fields[1]))
    if container_id is not None and CONTAINER_ID_RE.fullmatch(container_id) is None:
        inventory_valid = False

    name_present = any(name == container_name for _, name in entries)
    id_present = bool(container_id) and any(cid == container_id for cid, _ in entries)
    verified_absent = (
        inspect_returncode != 0
        and list_returncode == 0
        and inventory_valid
        and not name_present
        and not id_present
    )
    return {
        "container_absent": verified_absent,
        "inspect_returncode": inspect_returncode,
        "container_list_returncode": list_returncode,
        "container_list_valid": inventory_valid,
        "container_name_present": name_present,
        "container_id_present": id_present,
        "container_list_entry_count": len(entries),
    }


def first_post_action_fall(serialized_trace: str, *, action_monotonic_ns: int) -> dict[str, Any] | None:
    """Return the first post-SD odometry sample outside the existing height/tilt gate."""
    rows: list[dict[str, Any]] = []
    for line in serialized_trace.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("event") == "state"
            and row.get("topic_key") == "odom"
            and int(row.get("wall_monotonic_ns", 0)) >= action_monotonic_ns
        ):
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("wall_monotonic_ns", 0)))
    for row in rows:
        data = row.get("data", {})
        try:
            tilt = max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"])))
            unsafe = float(data["z"]) < 0.52 or tilt >= 15.0
        except (KeyError, TypeError, ValueError):
            unsafe = True
        if unsafe:
            return {
                "wall_utc": row.get("wall_utc"),
                "wall_monotonic_ns": int(row.get("wall_monotonic_ns", 0)),
                "odom": data,
                "reason": "post-action odometry crossed the unchanged root-height or tilt gate",
            }
    return None


class BoundedRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.evidence = Path(args.evidence_dir).resolve(strict=True)
        self.container = args.container
        self.source = "node.locomotion_velocity"
        self.started = time.monotonic()
        self.deadline = self.started + args.max_runtime_s
        self.step = 0
        self.steps: list[dict[str, Any]] = []
        self.container_created = False
        self.container_id: str | None = None
        self.network_mode: str | None = None
        self.cleanup_evidence: dict[str, Any] = {
            "attempted": False,
            "normal_path": "BoundedRun.execute() finally calls cleanup().",
            "process_crash_or_sigkill_cleanup_guaranteed": False,
            "operations": [],
        }
        self.reset_classification = "NOT_ATTEMPTED"
        self.m1_result: dict[str, Any] | None = None
        self.reset_analysis: dict[str, Any] | None = None
        self.before_reset_image: dict[str, Any] | None = None
        self.reset_dispatch_payload: dict[str, Any] | None = None
        self.immediate_reset_pose: dict[str, Any] | None = None
        self.sd_result: dict[str, Any] | None = None
        self.outcome_classification = "STANDING_CONTROL_NOT_READY"
        self.fault: str | None = None

    def remaining(self) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise RunDeadline("180-second wall-time budget expired")
        return value

    def save_command(self, name: str, command: list[str], result: subprocess.CompletedProcess[str] | None,
                     started: float, ended: float, error: str | None = None) -> None:
        self.step += 1
        record = {
            "step": self.step,
            "name": name,
            "command": command,
            "started_wall_utc": datetime.fromtimestamp(time.time() - (time.monotonic() - started), timezone.utc).isoformat(),
            "duration_s": ended - started,
            "returncode": result.returncode if result else None,
            "error": error,
        }
        stdout_path = self.evidence / f"step-{self.step:02d}-{name}.stdout.txt"
        stderr_path = self.evidence / f"step-{self.step:02d}-{name}.stderr.txt"
        with stdout_path.open("x", encoding="utf-8") as f:
            f.write(result.stdout if result else "")
        with stderr_path.open("x", encoding="utf-8") as f:
            f.write(result.stderr if result else error or "")
        record["stdout_path"] = str(stdout_path)
        record["stderr_path"] = str(stderr_path)
        self.steps.append(record)
        with (self.evidence / "execution.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def run(self, name: str, command: list[str], *, timeout_s: float = 8.0,
            check: bool = True) -> subprocess.CompletedProcess[str]:
        available = self.remaining()
        timeout = min(timeout_s, available)
        started = time.monotonic()
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            ended = time.monotonic()
            partial = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
            self.save_command(name, command, partial, started, ended, "subprocess timeout")
            raise RunDeadline(f"{name} timed out") from exc
        ended = time.monotonic()
        self.save_command(name, command, result, started, ended)
        if check and result.returncode != 0:
            raise RuntimeError(f"{name} returned {result.returncode}: {result.stderr[-500:]}")
        return result

    def docker_exec(self, name: str, shell: str, *, timeout_s: float = 8.0,
                    detached: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["docker", "exec"]
        if detached:
            cmd.append("-d")
        cmd += [self.container, "bash", "-lc", shell]
        return self.run(name, cmd, timeout_s=timeout_s, check=check)

    def stage_research_scripts(self) -> None:
        """Stage exact research code snapshots without overwriting evidence."""
        sources = (
            (Path(__file__), Path(__file__).name),
            (X11_HELPER, X11_HELPER.name),
            (ANALYZER, ANALYZER.name),
            (M1_RUNNER, M1_RUNNER.name),
            (M1_MODE_LOGIC, M1_MODE_LOGIC.name),
            (M1_FIRST_MOTION_LOGIC, M1_FIRST_MOTION_LOGIC.name),
            (SET_ACTION_CAPTURE, SET_ACTION_CAPTURE.name),
            (SIM_LAUNCH, "launch_sim.sh"),
            (MC_LAUNCH, "launch_mc.sh"),
        )
        records: list[dict[str, str]] = []
        for source, staged_name in sources:
            payload = source.read_bytes()
            target = self.evidence / staged_name
            digest = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"staged runtime input differs from current source: {target.name}")
                disposition = "PRESTAGED_IDENTICAL"
            else:
                with target.open("xb") as stream:
                    stream.write(payload)
                disposition = "COPIED"
            records.append({
                "name": source.name,
                "staged_name": staged_name,
                "sha256": digest,
                "disposition": disposition,
            })
        manifest = self.evidence / "staged_research_scripts.json"
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "robotsim.issue80.staged-research-scripts.v1", "scripts": records}, stream, indent=2)
            stream.write("\n")

    def start_container(self) -> None:
        cmd = [
            "docker", "run", "-d", "--pull=never", "--name", self.container,
            "--network=bridge", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770,size=64m",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026",
            "--mount", f"type=bind,source={self.evidence},target=/workspace/evidence",
            "--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly",
            "-e", "DISPLAY=:0", "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20",
            "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        self.run("docker_run", cmd, timeout_s=10.0)
        self.container_created = True
        inspect = self.run("container_inspect_start", ["docker", "inspect", self.container], timeout_s=5.0)
        payload = json.loads(inspect.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError("container inspect did not return one isolated runtime")
        container = payload[0]
        inspected_id = container.get("Id")
        if isinstance(inspected_id, str) and CONTAINER_ID_RE.fullmatch(inspected_id):
            self.container_id = inspected_id
        self.network_mode = str(container.get("HostConfig", {}).get("NetworkMode", ""))
        if self.network_mode != "bridge":
            raise RuntimeError(f"first-motion test requires bridge-isolated container; got {self.network_mode!r}")
        environment = set(container.get("Config", {}).get("Env", []))
        required_environment = {
            "ROS_DOMAIN_ID=20",
            "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        }
        if not required_environment.issubset(environment):
            raise RuntimeError("container ROS domain/RMW do not match the isolated M1 contract")
        with (self.evidence / "container-inspect-start.json").open("x", encoding="utf-8") as f:
            f.write(inspect.stdout)

    def start_sim_first(self) -> None:
        self.docker_exec(
            "launch_simulator",
            "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )
        # Do not use repeated short-lived `ros2 topic echo --once` processes as
        # startup readiness probes. On the pinned runtime they can time out
        # during ROS graph discovery (`!rclpy.ok()`) even after the simulator
        # has initialized and started its state-publisher threads. Wait for
        # the simulator's own module-ready log marker here; the persistent
        # read-only state observer below verifies actual live odom/IMU/leg
        # traffic after MC starts.
        self.docker_exec(
            "wait_simulator_module_ready",
            "timeout 25s bash -lc 'until grep -q \"MujocoSimModule Started Succeeded\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'",
            timeout_s=26.0,
        )
        self.docker_exec(
            "start_mc_after_sim_topics",
            "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )

    def wait_for_immediate_reset_pose(self) -> dict[str, Any]:
        event = (self.reset_dispatch_payload or {}).get("event", {})
        if event.get("status") != "KEY_DISPATCHED" or not event.get("dispatch_monotonic_ns"):
            raise RuntimeError("verified Reset dispatch timestamp is missing")
        dispatch_ns = int(event["dispatch_monotonic_ns"])
        trace = self.evidence / "reset_state_trace.jsonl"
        required = {"odom", "imu", "common", "leg_state", "leg_command"}
        deadline = min(self.deadline, time.monotonic() + POST_RESET_UPRIGHT_TIMEOUT_S)
        last = {"status": "POST_RESET_TRACE_NOT_AVAILABLE", "dispatch_monotonic_ns": dispatch_ns}
        while time.monotonic() < deadline:
            latest: dict[str, dict[str, Any]] = {}
            upright: list[dict[str, Any]] = []
            if trace.exists():
                for line in trace.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") != "state" or int(row.get("wall_monotonic_ns", 0)) <= dispatch_ns:
                        continue
                    key = row.get("topic_key")
                    if key in required:
                        latest[key] = row
                    if key == "odom":
                        data = row.get("data", {})
                        try:
                            ok = float(data["z"]) >= 0.52 and max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"]))) < 15.0
                        except (KeyError, TypeError, ValueError):
                            ok = False
                        if ok:
                            upright.append(row)
                        else:
                            upright.clear()
                now_ns = time.monotonic_ns()
                ages = {key: (now_ns - int(row["wall_monotonic_ns"])) / 1e9 for key, row in latest.items()}
                fresh = required.issubset(latest) and all(0.0 <= ages[key] <= 0.5 for key in required)
                last = {
                    "status": "POST_RESET_UPRIGHT_NOT_YET_CONFIRMED",
                    "dispatch_monotonic_ns": dispatch_ns,
                    "state_ages_s": ages,
                    "fresh_all_required_topics": fresh,
                    "consecutive_upright_odom_samples": len(upright),
                    "latest_odom": latest.get("odom"),
                    "latest_raw_common_state": latest.get("common"),
                }
                if fresh and len(upright) >= 3:
                    last.update({
                        "status": "POST_RESET_UPRIGHT_OBSERVED",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "dispatch_to_upright_ms": (int(upright[-3]["wall_monotonic_ns"]) - dispatch_ns) / 1e6,
                        "pose": upright[-1].get("data"),
                        "raw_common_state": latest.get("common", {}).get("data"),
                        "imu": latest.get("imu", {}).get("data"),
                        "leg_state": latest.get("leg_state", {}).get("data"),
                        "leg_command": latest.get("leg_command", {}).get("data"),
                    })
                    self.immediate_reset_pose = last
                    with (self.evidence / "reset_immediate_posture.json").open("x", encoding="utf-8") as f:
                        json.dump(last, f, indent=2)
                        f.write("\n")
                    return last
            time.sleep(0.02)
        last["status"] = "POST_RESET_UPRIGHT_TIMEOUT"
        last["timeout_s"] = POST_RESET_UPRIGHT_TIMEOUT_S
        self.immediate_reset_pose = last
        self.outcome_classification = "STANDING_CONTROL_NOT_READY"
        with (self.evidence / "reset_immediate_posture.json").open("x", encoding="utf-8") as f:
            json.dump(last, f, indent=2)
            f.write("\n")
        raise RuntimeError("Reset did not produce a fresh upright pose within 2 s; SD was not sent")

    def request_sd_and_verify(self) -> dict[str, Any]:
        # The ROS node, all state subscriptions, and both MC service clients
        # were created before Reset by start_observer(). Read the transaction
        # produced by that persistent process; do not spawn Python/DDS here.
        result_path = self.evidence / "set_action_capture_result.json"
        summary_path = self.evidence / "reset_state_summary.json"
        deadline = min(self.deadline, time.monotonic() + RESET_OBSERVER_DURATION_S + 2.0)
        while time.monotonic() < deadline and not result_path.exists():
            if summary_path.exists():
                break
            time.sleep(0.01)
        if result_path.exists():
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raw = {
                    "status": "BLOCKED_RESULT_READ",
                    "blocking_stage": "RESULT_READ",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            raw = {
                "status": "BLOCKED_RESULT_TIMEOUT",
                "blocking_stage": "RESULT_TIMEOUT",
                "error": "the pre-armed ROS process did not publish its result before the bounded capture window ended",
            }
        raw_sd = raw.get("stand_default_set_mc_action") or {}
        record: dict[str, Any] = {
            "wall_utc_observed": datetime.now(timezone.utc).isoformat(),
            "requested_action_desc": "STAND_DEFAULT",
            "request_source": "node.set_mc_action",
            "takeover_status": raw.get("status"),
            "blocking_stage": raw.get("blocking_stage"),
            "error": raw.get("error"),
            "pre_reset_get_mc_action": raw.get("pre_reset_get_mc_action"),
            "post_reset_upright_gate": raw.get("post_reset_upright_gate"),
            "joint_default_set_mc_action": raw.get("joint_default_set_mc_action"),
            "joint_to_stand_gate": raw.get("joint_to_stand_gate"),
            "raw_set_mc_action": raw_sd,
            "get_mc_action_readback": raw.get("get_mc_action_readbacks", []),
            "timing_ms": raw.get("timing_ms"),
            "limitations": raw.get("limitations", []),
            "response_status_value": raw_sd.get("response_status_value"),
            "response_header_code": raw_sd.get("response_header_code"),
            "response_message": raw_sd.get("response_message"),
        }
        self.sd_result = record
        if raw.get("status") != "TAKEOVER_SEQUENCE_COMPLETE":
            jd = raw.get("joint_default_set_mc_action") or {}
            sd = raw.get("stand_default_set_mc_action") or {}
            stage = raw.get("blocking_stage")
            detail = sd if sd else jd
            if (
                stage == "STAND_DEFAULT"
                and detail.get("transport_state") == "RESPONSE_RECEIVED"
                and (detail.get("service_success") is not True or detail.get("action_result_success") is not True)
            ):
                self.outcome_classification = "SD_REJECTED_WITH_EXACT_CODE"
            else:
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            self._write_sd_record(record)
            raise RuntimeError(
                "source-supported PD -> JD -> SD handoff did not complete: "
                f"stage={raw.get('blocking_stage')!r} status={detail.get('response_status_value')} "
                f"header.code={detail.get('response_header_code')} message={detail.get('response_message')!r} "
                f"reason={raw.get('error')!r}"
            )
        if raw_sd.get("service_success") is not True or raw_sd.get("action_result_success") is not True:
            self.outcome_classification = "SD_REJECTED_WITH_EXACT_CODE"
            self._write_sd_record(record)
            raise RuntimeError(
                "SetMcAction SD was rejected: "
                f"status={record['response_status_value']} header.code={record['response_header_code']} "
                f"message={record['response_message']!r}"
            )
        stand_readbacks = [
            item for item in record["get_mc_action_readback"]
            if item.get("action_desc") == "STAND_DEFAULT" and item.get("action_status_value") == 100
        ]
        if not stand_readbacks:
            self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            self._write_sd_record(record)
            raise RuntimeError("the persistent GetMcAction client did not confirm STAND_DEFAULT/status=100")
        self._write_sd_record(record)
        return record

    def _write_sd_record(self, record: dict[str, Any]) -> None:
        path = self.evidence / "set_mc_action_sd.json"
        if path.exists():
            return
        with path.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")

    def capture_service_topic_inventory(self) -> None:
        self.docker_exec(
            "service_inventory",
            "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && ros2 service list -t",
            timeout_s=5.0,
        )
        self.docker_exec(
            "topic_inventory",
            "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && ros2 topic list -t",
            timeout_s=5.0,
        )

    def start_observer(self) -> None:
        self.docker_exec(
            "start_prearmed_standing_takeover_observer",
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "timeout 45s python3 /workspace/evidence/issue80_m1_set_action_capture.py "
            "--action STAND_DEFAULT --source node.set_mc_action "
            "--output-dir /workspace/evidence "
            "--reset-event-file /workspace/evidence/reset_dispatch_event.json "
            "--ready-timeout-s 3 --reset-wait-timeout-s 30 --upright-timeout-s 2 "
            "--joint-stand-timeout-s 4 --set-timeout-s 2 --get-timeout-s 0.75 "
            f"--duration-s {RESET_OBSERVER_DURATION_S:.1f} "
            "> /workspace/evidence/takeover-capture-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )
        ready_path = self.evidence / "reset_watch_ready.json"
        # Child readiness has its own unchanged 3 s budget after Python/ROS
        # construction. Parent also allows bounded process bootstrap time.
        deadline = min(self.deadline, time.monotonic() + 5.0)
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.05)
        if not ready_path.exists():
            raise RuntimeError("Reset observer readiness result missing after 5 s including process bootstrap")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("status") != "READY":
            raise RuntimeError(f"pre-Reset ROS2 observer/client readiness failed: {ready}")
        if len(ready.get("fresh_topics", [])) != 5:
            raise RuntimeError(f"Reset precondition failed: live state streams are incomplete: {ready}")
        if ready.get("set_mc_action_service_ready") is not True or ready.get("get_mc_action_service_ready") is not True:
            raise RuntimeError(f"Reset precondition failed: persistent MC action services are not ready: {ready}")
        pre_mode = ready.get("pre_reset_get_mc_action", {})
        if pre_mode.get("action_desc") != "PASSIVE_DEFAULT" or pre_mode.get("action_status_value") != 100:
            raise RuntimeError(f"Reset precondition failed: expected PASSIVE_DEFAULT, got {pre_mode}")
        # Pre-SD mode/FSM values are retained raw and are not a standing gate.
        with (self.evidence / "pre-reset-telemetry-snapshot.json").open("x", encoding="utf-8") as f:
            json.dump(ready, f, indent=2)
            f.write("\n")

    def capture_reset_image(self, pid: int, title: str, window_id: int, phase: str) -> dict[str, Any]:
        """Capture images outside the live reset-observation interval when possible."""
        output = self.evidence / f"mujoco_{phase}_reset.png"
        command = [
            sys.executable, str(X11_HELPER), "capture",
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id), "--output", str(output),
        ]
        result = self.run(f"x11_capture_{phase}_reset", command, timeout_s=8.0, check=False)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "status": "NOT_CAPTURED",
                "reason": f"capture helper returned no JSON (exit={result.returncode})",
            }
        image = payload.get("image", {})
        if result.returncode != 0 or image.get("status") != "CAPTURED":
            return {
                "status": "NOT_CAPTURED",
                "reason": payload.get("reason") or payload.get("error") or result.stderr[-300:],
                "helper_result": payload,
            }
        return image

    def inspect_window(self) -> tuple[int, str, int]:
        pid_result = self.docker_exec(
            "sim_process_pid",
            "pgrep -x soc_sim",
            timeout_s=3.0,
        )
        container_pid = int(pid_result.stdout.strip().splitlines()[0])
        top = self.run("docker_top_processes", ["docker", "top", self.container, "-eo", "pid,comm,args"], timeout_s=4.0)
        host_pid = None
        for line in top.stdout.splitlines()[1:]:
            fields = line.strip().split(None, 2)
            if len(fields) >= 2 and fields[1] == "soc_sim":
                host_pid = int(fields[0])
                break
        pids = [container_pid] + ([host_pid] if host_pid else [])
        discoveries = []
        for pid in pids:
            result = self.run(
                f"x11_inspect_pid_{pid}",
                [sys.executable, str(X11_HELPER), "inspect", "--expected-pid", str(pid)],
                timeout_s=4.0,
                check=False,
            )
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
            discoveries.append(payload)
            if payload.get("windows"):
                windows = payload["windows"]
                if len(windows) != 1:
                    raise RuntimeError(f"ambiguous MuJoCo windows for PID {pid}: {windows}")
                window = windows[0]
                return int(pid), str(window["title"]), int(window["window_id"])
        with (self.evidence / "window-discovery.json").open("x", encoding="utf-8") as f:
            json.dump({"container_pid": container_pid, "host_pid": host_pid, "inspections": discoveries}, f, indent=2)
            f.write("\n")
        raise RuntimeError("no visible X11 window with exact aima-sim-app PID was found")

    def dispatch_reset(self, pid: int, title: str, window_id: int) -> None:
        event_path = self.evidence / "reset_dispatch_event.json"
        command = [
            sys.executable, str(X11_HELPER), "send-reset",
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id),
            "--dispatch-event-json", str(event_path),
        ]
        try:
            result = self.run("x11_backspace_dispatch", command, timeout_s=5.0, check=False)
        except RunDeadline as exc:
            # A dispatched key event is persisted before this timeout check.
            # Continue only when that exact event record exists; screenshots
            # are intentionally outside this key-delivery operation.
            if not event_path.exists():
                raise
            event_record = json.loads(event_path.read_text(encoding="utf-8"))
            event_record["helper_completion"] = f"TIMED_OUT_AFTER_DISPATCH: {exc}"
            self.reset_dispatch_payload = event_record
            if event_record.get("event", {}).get("status") != "KEY_DISPATCHED":
                self.reset_classification = "RESET_NOT_DELIVERED"
                raise RuntimeError(f"X11 helper did not dispatch BackSpace: {event_record}") from exc
            return
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            if event_path.exists():
                payload = json.loads(event_path.read_text(encoding="utf-8"))
                payload["helper_completion"] = "NO_COMPLETE_RESULT_JSON"
            else:
                raise RuntimeError(f"X11 helper did not return JSON: {result.stdout[-500:]}") from exc
        self.reset_dispatch_payload = payload
        if result.returncode != 0 or payload.get("event", {}).get("status") != "KEY_DISPATCHED":
            self.reset_classification = "RESET_NOT_DELIVERED"
            raise RuntimeError(f"X11 Backspace was not dispatched: {payload}")

    def finalize_reset_dispatch(self, pid: int, title: str, window_id: int) -> None:
        """Attach image evidence only after the continuous telemetry window ends."""
        event_path = self.evidence / "reset_dispatch_event.json"
        payload = self.reset_dispatch_payload
        if payload is None and event_path.exists():
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        if payload is None:
            raise RuntimeError("cannot finalize Reset evidence without a persisted key-event record")
        payload["before_image"] = self.before_reset_image or {"status": "NOT_CAPTURED"}
        payload["after_image"] = self.capture_reset_image(pid, title, window_id, "after")
        with (self.evidence / "reset_dispatch.json").open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        self.reset_dispatch_payload = payload

    def wait_and_classify_reset(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        trace = self.evidence / "reset_state_trace.jsonl"
        deadline = min(
            self.deadline,
            time.monotonic() + RESET_OBSERVER_DURATION_S + RESET_OBSERVER_COMPLETION_MARGIN_S,
        )
        while time.monotonic() < deadline and not trace.exists():
            time.sleep(0.05)
        if not trace.exists():
            raise RuntimeError("reset state trace was not produced")
        # The observer is capped at 8 s; wait for its atomic summary with the same bound.
        summary = self.evidence / "reset_state_summary.json"
        action_ns = int(
            ((self.sd_result or {}).get("raw_set_mc_action") or {}).get("wall_monotonic_ns_submitted", 0)
        )
        while time.monotonic() < deadline and not summary.exists():
            if trace.exists() and action_ns:
                fall = first_post_action_fall(
                    trace.read_text(encoding="utf-8"),
                    action_monotonic_ns=action_ns,
                )
                if fall:
                    self.reset_classification = "RESET_APPLIED_THEN_FELL"
                    self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                    self.reset_analysis = {
                        "classification": self.reset_classification,
                        "post_action_fall": fall,
                        "safety_window_completed": False,
                        "stop_reason": "stopped on first post-SD unsafe odometry sample",
                    }
                    raise RuntimeError(
                        "post-SD odometry crossed the existing standing safety gate; stopped observer and did not proceed to LD"
                    )
            time.sleep(0.02)
        if not summary.exists():
            raise RuntimeError("bounded Reset observer did not finish before its duration plus completion margin")

        # The state trace, not screenshot completion, owns the stability gate.
        self.finalize_reset_dispatch(pid, title, window_id)
        out = self.evidence / "reset_analysis.json"
        result = self.run(
            "classify_reset_state",
            [sys.executable, str(ANALYZER), "--trace", str(trace), "--dispatch", str(self.evidence / "reset_dispatch.json"), "--output", str(out)],
            timeout_s=4.0,
            check=False,
        )
        if out.exists():
            self.reset_analysis = json.loads(out.read_text(encoding="utf-8"))
        else:
            try:self.reset_analysis=json.loads(result.stdout)
            except json.JSONDecodeError:self.reset_analysis={"classification":"ANALYSIS_FAILED","stdout":result.stdout,"stderr":result.stderr}
        self.reset_classification = str(self.reset_analysis.get("classification","ANALYSIS_FAILED"))
        return self.reset_analysis

    def run_m1(self) -> None:
        if not self.args.maintainer_authorized_first_motion:
            self.m1_result = {
                "status": "NOT_AUTHORIZED",
                "reason": "first nonzero simulation-only command requires explicit maintainer authorization",
            }
            return
        if self.reset_classification != "AUTO_RESET_PASS":
            raise RuntimeError("first motion requires the unchanged post-Reset standing gate to pass")
        if self.remaining() < 35.0:
            raise RunDeadline("fewer than 35 seconds remain after AUTO_RESET_PASS; M1 was not started")
        if self.network_mode != "bridge":
            raise RuntimeError("verified bridge isolation is required before M1")
        command = [
            "docker", "exec", "-e", "ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", self.container, "bash", "-lc",
            "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && python3 /workspace/evidence/issue80_m1_locomotion_probe.py first-motion --maintainer-authorized --reset-gate-verified"
            + " --extra-stop-observation " + str(self.args.extra_stop_observation)
            + " --pulse-duration " + str(self.args.pulse_duration)
            + " --forward-speed " + str(self.args.forward_speed)
            + " --ramp-start-speed " + str(getattr(self.args, "ramp_start_speed", None) or self.args.forward_speed)
            + " --ramp-duration-s " + str(getattr(self.args, "ramp_duration_s", 0.0))
            + " --attempt-id " + self.args.attempt_id
            + " --robot-model-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/robot_model.yaml"
            + " --mc-sim-config-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"
            + " --state-timeout 0.5 --ready-timeout 4 --max-tilt-deg 15 --min-root-z 0.52"
            + " --output-dir /workspace/evidence"
        ]
        startup_speed = getattr(self.args, "startup_speed", None)
        if startup_speed is not None:
            command[-1] += (
                " --startup-speed " + str(startup_speed)
                + " --startup-transition-displacement "
                + str(getattr(self.args, "startup_transition_displacement", None))
                + " --startup-transition-timeout-s "
                + str(getattr(self.args, "startup_transition_timeout_s", 0.0))
            )
        result = self.run("m1_first_motion_diagnostic", command, timeout_s=min(45.0, self.remaining()), check=False)
        result_file = self.evidence / "m1_first_motion_result.json"
        if result_file.exists():
            try:
                self.m1_result = decode_m1_child_result(result_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.m1_result = {
                    "status": "CHILD_RESULT_INVALID",
                    "fault": f"child result could not be decoded: {type(exc).__name__}: {exc}",
                    "child_returncode": result.returncode,
                    "child_stdout_tail": result.stdout[-4000:],
                    "child_stderr_tail": result.stderr[-4000:],
                }
        else:
            self.m1_result = {"status":"NO_RESULT_FILE","returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}

    def cleanup(self) -> None:
        if not self.container_created:
            self.cleanup_evidence["attempted"] = False
            self.cleanup_evidence["reason"] = "container was not created"
            return
        self.cleanup_evidence["attempted"] = True
        self.cleanup_evidence["normal_path"] = (
            "Python finally copies audit files, then issues docker stop --time 2 and docker rm."
        )
        self.cleanup_evidence["process_crash_or_sigkill_cleanup_guaranteed"] = False
        try:
            copied = self.run(
                "copy_tmpfs_audit",
                ["docker", "exec", self.container, "bash", "-lc",
                 "cp -n /agibot/data/var/MotionControlModule/runtime.yaml /workspace/evidence/runtime_audit_mc.yaml 2>/dev/null || true; cp -n /agibot/data/var/mc/audit.log /workspace/evidence/runtime_audit_mc-audit.log 2>/dev/null || true"],
                timeout_s=3.0,
                check=False,
            )
            self.cleanup_evidence["operations"].append({
                "operation": "copy_tmpfs_audit",
                "returncode": copied.returncode,
            })
        except Exception:
            self.cleanup_evidence["operations"].append({"operation": "copy_tmpfs_audit", "error": "exception"})
        try:
            inspect = subprocess.run(
                ["docker", "inspect", "--type=container", self.container_id or self.container],
                text=True, capture_output=True, timeout=3.0,
            )
            if inspect.returncode == 0:
                with (self.evidence / "container-inspect-final.json").open("x", encoding="utf-8") as f:
                    f.write(inspect.stdout)
            self.cleanup_evidence["operations"].append({
                "operation": "inspect_before_stop",
                "returncode": inspect.returncode,
            })
        except Exception:
            self.cleanup_evidence["operations"].append({"operation": "inspect_before_stop", "error": "exception"})
        for operation, cmd in (
            ("docker_stop", ["docker", "stop", "--time", "2", self.container]),
            ("docker_rm", ["docker", "rm", self.container]),
        ):
            try:
                completed = subprocess.run(cmd, text=True, capture_output=True, timeout=4.0)
                self.cleanup_evidence["operations"].append({
                    "operation": operation,
                    "returncode": completed.returncode,
                })
            except Exception:
                self.cleanup_evidence["operations"].append({"operation": operation, "error": "exception"})
        verify: subprocess.CompletedProcess[str] | None = None
        listed: subprocess.CompletedProcess[str] | None = None
        verify_error = None
        list_error = None
        try:
            verify = subprocess.run(
                ["docker", "inspect", "--type=container", self.container_id or self.container],
                text=True, capture_output=True, timeout=3.0,
            )
        except Exception as exc:
            verify_error = f"{type(exc).__name__}: {exc}"
        try:
            listed = subprocess.run(
                ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
                text=True, capture_output=True, timeout=3.0,
            )
        except Exception as exc:
            list_error = f"{type(exc).__name__}: {exc}"
        absence = assess_container_absence(
            inspect_returncode=verify.returncode if verify else -1,
            list_returncode=listed.returncode if listed else -1,
            list_stdout=listed.stdout if listed else "",
            container_name=self.container,
            container_id=self.container_id,
        )
        self.cleanup_evidence["container_absent_after_cleanup"] = absence["container_absent"]
        self.cleanup_evidence["operations"].append({
            "operation": "verify_container_absent",
            **absence,
            "inspect_error": verify_error,
            "container_list_error": list_error,
        })

    def apply_container_teardown_gate(self, result: dict[str, Any]) -> None:
        result["container_teardown"] = self.cleanup_evidence
        if (
            self.cleanup_evidence.get("attempted")
            and not self.cleanup_evidence.get("container_absent_after_cleanup", False)
        ):
            result["status"] = "BLOCKED"
            result["fault"] = result.get("fault") or "container teardown was not verified after runtime"

    def classify_mc_shutdown(self) -> dict[str, Any]:
        candidates = (
            self.evidence / "mc-console.log",
            self.evidence / "mc_logs" / "mc.log",
        )
        inspected = []
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            inspected.append(str(path))
            if re.search(r"SIGSEGV|segmentation fault", text, flags=re.IGNORECASE):
                return {
                    "classification": "MC_SHUTDOWN_SIGSEGV",
                    "evidence_files": inspected,
                    "note": "shutdown crash is separate from the standing/walking experiment result",
                }
        return {
            "classification": "MC_SHUTDOWN_SIGSEGV_NOT_OBSERVED_IN_CAPTURED_LOGS",
            "evidence_files": inspected,
            "note": "absence from captured logs does not prove a clean MC shutdown",
        }

    def execute(self) -> dict[str, Any]:
        result = {
            "schema": "robotsim.issue80.m1.auto-reset-run.v4",
            "status": "NOT_AUTHORIZED",
            "attempt_id": self.args.attempt_id,
            "source": "node.locomotion_velocity",
            "container": self.container,
            "runtime_image_digest": IMAGE_DIGEST,
            "runtime_limit_s": self.args.max_runtime_s,
            "reset_classification": "NOT_ATTEMPTED",
            "m1_status": "NOT_RUN",
            "outcome_classification": "STANDING_CONTROL_NOT_READY",
            "fault": None,
        }
        standing_only = bool(getattr(self.args, "maintainer_authorized_standing_takeover", False))
        first_motion_authorized = bool(getattr(self.args, "maintainer_authorized_first_motion", False))
        if not standing_only and not first_motion_authorized:
            result["fault"] = "explicit maintainer standing-takeover or first-motion authorization is required before runtime"
            return result
        try:
            self.stage_research_scripts()
            self.start_container()
            self.start_sim_first()
            self.capture_service_topic_inventory()
            pid, title, window_id = self.inspect_window()
            self.before_reset_image = self.capture_reset_image(pid, title, window_id, "before")
            self.start_observer()
            self.dispatch_reset(pid, title, window_id)
            result["reset_immediate_posture"] = self.wait_for_immediate_reset_pose()
            # The pre-armed node sends the source-supported PD -> JD -> SD
            # handoff; SD is gated on fresh upright telemetry and MC STAND state.
            result["set_mc_action_sd"] = self.request_sd_and_verify()
            reset = self.wait_and_classify_reset(pid, title, window_id)
            result["reset_classification"] = reset.get("classification")
            result["reset_analysis"] = reset
            if reset.get("classification") != "AUTO_RESET_PASS":
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                raise RuntimeError(f"post-SD standing safety gate failed: {reset.get('classification')}")

            if standing_only and not first_motion_authorized:
                # This authorization is limited to the standing takeover gate.
                # It must not fall through to source registration, LD, or velocity.
                self.outcome_classification = "STANDING_TAKEOVER_PASS"
                result["m1_status"] = "NOT_RUN_STANDING_ONLY"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "PASS"
                result["next_motion_stage"] = "NOT_RUN_REQUIRES_SEPARATE_AUTHORIZATION"
                return result

            self.run_m1()
            result["m1_result"] = self.m1_result
            result["m1_status"] = (self.m1_result or {}).get("status", "UNKNOWN")
            child_fault = (self.m1_result or {}).get("fault")
            if isinstance(child_fault, str) and child_fault:
                result["fault"] = child_fault
            m1 = self.m1_result or {}
            fault_text = str(m1.get("fault") or "").lower()
            registration = m1.get("source_registration")
            ld_action = m1.get("set_mc_action_ld")
            activation = m1.get("activation")
            nonzero_calls = int(activation.get("nonzero_publish_calls_returned", 0) or 0) if isinstance(activation, Mapping) else 0
            safety_fault = any(token in fault_text for token in ("standing", "non_standing", "fall_", "stale_or_missing", "mc_safety"))
            if safety_fault and nonzero_calls > 0:
                self.outcome_classification = "FIRST_PHYSICAL_WALKING_FAIL"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "BLOCKED"
                return result
            if safety_fault:
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "BLOCKED"
                return result
            if isinstance(registration, Mapping) and registration.get("success") is not True:
                self.outcome_classification = "LOCOMOTION_SOURCE_REJECTED"
            elif isinstance(ld_action, Mapping) and (
                ld_action.get("accepted_by_service") is not True
                or ld_action.get("response_header_code") not in (0, None)
            ):
                self.outcome_classification = "LD_REJECTED_WITH_EXACT_CODE"
            elif isinstance(activation, Mapping):
                self.outcome_classification = classify_first_motion_outcome(activation)
            elif result["m1_status"] == "BLOCKED":
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            result["outcome_classification"] = self.outcome_classification
            result["status"] = (
                "PASS" if self.outcome_classification == "FIRST_PHYSICAL_WALKING_PASS" else "BLOCKED"
            )
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            result["fault"] = self.fault
            result["reset_classification"] = self.reset_classification
            result["set_mc_action_sd"] = self.sd_result
            result["m1_result"] = self.m1_result
            result["outcome_classification"] = self.outcome_classification
            result["status"] = "BLOCKED"
        finally:
            elapsed = time.monotonic() - self.started
            result["wall_elapsed_s"] = elapsed
            result["wall_budget_exhausted"] = elapsed > self.args.max_runtime_s
            if self.reset_analysis:
                result["reset_analysis"] = self.reset_analysis
            self.cleanup()
            result["shutdown_classification"] = self.classify_mc_shutdown()
            self.apply_container_teardown_gate(result)
            output = self.evidence / "auto-reset-run-result.json"
            with output.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
        return result



def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",required=True)
    parser.add_argument("--container",required=True)
    parser.add_argument("--attempt-id",required=True)
    parser.add_argument("--max-runtime-s",type=float,default=180.0)
    parser.add_argument("--extra-stop-observation", type=float, default=0.0)
    parser.add_argument("--pulse-duration", type=float, default=1.0)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument("--ramp-start-speed", type=float, default=None)
    parser.add_argument("--ramp-duration-s", type=float, default=0.0)
    parser.add_argument("--startup-speed", type=float, default=None)
    parser.add_argument("--startup-transition-displacement", type=float, default=None)
    parser.add_argument("--startup-transition-timeout-s", type=float, default=0.0)
    parser.add_argument(
        "--maintainer-authorized-first-motion",
        action="store_true",
        help="explicit human authorization for one bounded simulation-only first-motion run",
    )
    parser.add_argument(
        "--maintainer-authorized-standing-takeover",
        action="store_true",
        help="explicit human authorization for one simulation-only standing-takeover test; no LD or velocity command",
    )
    args=parser.parse_args()
    if not 0.0 <= args.extra_stop_observation <= 5.0:
        parser.error("extra zero-only observation must remain within 0..5 seconds")
    if not 0.0 < args.pulse_duration <= 10.0:
        parser.error("pulse duration must be positive and no greater than 10 seconds")
    if not 0.0 < args.forward_speed <= 0.2:
        parser.error("forward speed must be positive and no greater than 0.2 m/s")
    if not 0.0 <= args.ramp_duration_s <= args.pulse_duration:
        parser.error("velocity ramp duration must fit within the pulse duration")
    if args.ramp_duration_s > 0.0 and (
        args.ramp_start_speed is None
        or not 0.0 < args.ramp_start_speed <= args.forward_speed
    ):
        parser.error("ramp start must be positive and no faster than target speed")
    run=BoundedRun(args)
    result=run.execute()
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
