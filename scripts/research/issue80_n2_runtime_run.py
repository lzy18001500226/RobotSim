#!/usr/bin/env python3
"""One bounded N2 MuJoCo + MC startup diagnosis; no mode or velocity commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from issue80_m1_auto_reset_run import (
    BoundedRun,
    IMAGE,
    IMAGE_DIGEST,
    MC_LAUNCH,
    SIM_LAUNCH,
    X11_HELPER,
)

REPO = Path("/home/lzy18001500226/projects/RobotSim")
WORKTREE = Path("/home/lzy18001500226/robotsim-issue80-x2-navigation-20261019")
RUNTIME = REPO / "simulation/mujoco/raicom2026"
TRACE_SCRIPT = WORKTREE / "scripts/research/issue80_n2_runtime_trace.py"
EXPECTED_HEAD = "1a7f1104d41b72b662178bdb7bef2f05b4f77106"
EXPECTED_BRANCH = "research/issue80-x2-navigation-m1-diagnostic-20261019"
HASH_FILES = (
    "sim_mujoco/bin/aima-sim-app", "mc/bin/aima-mc-app-main", "mc/bin/em_run_wsl2.sh",
    "sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml",
    "sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml",
    "sim_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml",
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml",
    "aimdk_msgs_ws/src/aimdk_msgs/interface/robot/mc/data/msg/common/McFsmState.msg",
)
ROOT_Z_MIN_M = 0.52
MAX_TILT_DEG = 15.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return sha256(path)


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_jsonl_complete_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A live line-buffered writer may be in the middle of its final line.
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


class N2Run(BoundedRun):
    """Use the reviewed isolated-container/X11/cleanup implementation unchanged."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.branch = ""
        self.head = ""
        self.mc_started = False
        self.preflight_record: dict[str, Any] = {}
        self.pre_mc_record: dict[str, Any] | None = None
        self.pre_mc_init: dict[str, Any] | None = None
        self.runtime_summary: dict[str, Any] | None = None
        self.sim_ready_record: dict[str, Any] | None = None
        self.n2_summary: dict[str, Any] | None = None
        self.n2_simulator_window: dict[str, Any] | None = None
        self.runtime_hashes_before: dict[str, str] = {}
        self.runtime_hashes_after: dict[str, str] = {}
        self.runtime_hash_changes: dict[str, Any] = {}
        self.current_gate = "NOT_STARTED"

    def record_stage(self, stage: str, outcome: str, detail: str | None = None) -> None:
        path = self.evidence / "n2_stage_timeline.jsonl"
        record = {"stage": stage, "outcome": outcome, "wall_utc": utc_now(),
                  "wall_monotonic_ns": time.monotonic_ns(), "detail": detail}
        mode = "a" if path.exists() else "x"
        with path.open(mode, encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def preflight_n2(self) -> None:
        actual_root = self.run("n2_worktree_root", ["git", "-C", str(WORKTREE), "rev-parse", "--show-toplevel"]).stdout.strip()
        if Path(actual_root).resolve(strict=True) != WORKTREE.resolve(strict=True):
            raise RuntimeError(f"worktree root mismatch: expected {WORKTREE}, got {actual_root}")
        actual_runner = Path(__file__).resolve(strict=True)
        expected_runner = (WORKTREE / "scripts/research/issue80_n2_runtime_run.py").resolve(strict=True)
        if actual_runner != expected_runner:
            raise RuntimeError(f"runner is outside the verified worktree: {actual_runner}")
        self.branch = self.run("n2_branch", ["git", "-C", str(WORKTREE), "branch", "--show-current"]).stdout.strip()
        if self.branch != EXPECTED_BRANCH:
            raise RuntimeError(f"authorized branch mismatch: expected {EXPECTED_BRANCH}, got {self.branch}")
        self.head = self.run("n2_head", ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]).stdout.strip()
        if self.head != EXPECTED_HEAD:
            raise RuntimeError(f"authorized PR81 HEAD mismatch: expected {EXPECTED_HEAD}, got {self.head}")
        docker = self.run("n2_docker_info", ["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}"]).stdout.strip()
        image = self.run("n2_image_id", ["docker", "image", "inspect", "--format", "{{.Id}} {{.Os}}", IMAGE]).stdout.strip()
        if not image.startswith(IMAGE_DIGEST + " linux"):
            raise RuntimeError(f"pinned runtime image mismatch: {image}")
        if os.environ.get("DISPLAY") != ":0":
            raise RuntimeError(f"expected WSLg DISPLAY=:0, got {os.environ.get('DISPLAY')!r}")
        self.run("n2_x11_probe", ["xset", "-q"], timeout_s=5.0)
        inventory = self.run("n2_docker_inventory", ["docker", "ps", "-a", "--no-trunc", "--format",
            "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"])
        rows = inventory.stdout.splitlines()
        parsed_rows: list[dict[str, str]] = []
        for row in rows:
            if not row:
                continue
            fields = row.split("\t")
            if len(fields) != 4 or not fields[1]:
                raise RuntimeError(f"malformed Docker container inventory row: {row!r}")
            parsed_rows.append({"id": fields[0], "name": fields[1], "image": fields[2], "status": fields[3]})
        if any(row["name"] == self.container for row in parsed_rows):
            raise RuntimeError(f"one-shot container name already exists: {self.container}")
        conflicts = [row for row in parsed_rows if row["status"].startswith("Up ")
                     and ("raicom2026" in row["name"] or "humble-raicom-runtime" in row["image"])]
        if conflicts:
            raise RuntimeError(f"another RAICOM runtime container is active: {conflicts}")
        host_processes = {}
        for executable in ("aima-sim-app", "aima-mc-app-main", "soc_sim"):
            proc = self.run("n2_host_proc_" + executable, ["pgrep", "-a", "-x", executable], check=False)
            host_processes[executable] = proc.stdout.strip().splitlines()
            if proc.returncode == 0 and proc.stdout.strip():
                raise RuntimeError(f"conflicting host process: {proc.stdout.strip()}")
        ignore = self.run("n2_runtime_ignore", ["git", "-C", str(REPO), "check-ignore", "-v",
            "simulation/mujoco/raicom2026"]).stdout.strip()
        self.runtime_hashes_before = {relative: sha256(RUNTIME / relative) for relative in HASH_FILES}
        self.preflight_record = {
            "schema": "robotsim.issue80.n2.preflight.v1", "attempt_id": self.args.attempt_id,
            "worktree_root": actual_root, "runner_path": str(actual_runner),
            "branch": self.branch, "head": self.head, "docker_server": docker,
            "image_id": image.split()[0], "container_inventory": parsed_rows, "host_processes": host_processes,
            "display": os.environ["DISPLAY"], "runtime_git_ignore": ignore,
            "runtime_sha256_before": self.runtime_hashes_before,
            "fsm_codes": {"STABLE": 2, "MOVING": 3, "SAFE": 4},
            "prohibited_operations": ["SetMcInputSource", "SetMcAction", "nonzero velocity", "real robot"],
        }
        write_new(self.evidence / "n2_preflight.json", self.preflight_record)

    def stage_n2_inputs(self) -> None:
        base_module_path = Path(sys.modules[BoundedRun.__module__].__file__).resolve(strict=True)
        sources = (
            (base_module_path, base_module_path.name),
            (Path(X11_HELPER).resolve(strict=True), Path(X11_HELPER).name),
            (Path(SIM_LAUNCH).resolve(strict=True), "launch_sim.sh"),
            (Path(MC_LAUNCH).resolve(strict=True), "launch_mc.sh"),
            (Path(__file__).resolve(strict=True), "issue80_n2_runtime_run.py"),
            (TRACE_SCRIPT.resolve(strict=True), "issue80_n2_runtime_trace.py"),
        )
        records: list[dict[str, str]] = []
        for source, staged_name in sources:
            target = self.evidence / staged_name
            digest = sha256(source)
            if target.exists():
                if target.is_symlink() or not target.is_file() or sha256(target) != digest:
                    raise RuntimeError(f"refusing to overwrite non-identical N2 input: {staged_name}")
                disposition = "PRESTAGED_IDENTICAL"
            else:
                shutil.copy2(source, target)
                disposition = "COPIED"
            records.append({"name": source.name, "staged_name": staged_name,
                            "sha256": digest, "disposition": disposition})
        write_new(self.evidence / "n2_staged_inputs.json", {"schema": "robotsim.issue80.n2.inputs.v1", "files": records})
        write_new(self.evidence / "n2_trace_input_hash.json", {
            "runner_sha256": file_sha256(self.evidence / "issue80_n2_runtime_run.py"),
            "trace_path": "issue80_n2_runtime_trace.py",
            "trace_sha256": file_sha256(self.evidence / "issue80_n2_runtime_trace.py"),
        })

    def start_simulator_before_mc(self) -> None:
        self.docker_exec("n2_launch_simulator",
            "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1",
            timeout_s=4.0, detached=True)
        wait = "timeout 25s bash -c 'until grep -Fq \"MujocoSimModule Started Succeeded\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'"
        self.docker_exec("n2_wait_sim_ready", wait, timeout_s=27.0)
        self.record_stage("simulator_ready", "PASS", "MujocoSimModule Started Succeeded")
        observer = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "exec python3 /workspace/evidence/issue80_n2_runtime_trace.py "
            "--output-dir /workspace/evidence --duration 40"
        )
        self.docker_exec("n2_start_readonly_trace", observer, timeout_s=4.0, detached=True)
        ready_path = self.evidence / "n2_sim_ready.json"
        deadline = min(time.monotonic() + 8.0, self.deadline)
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.05)
        if not ready_path.exists():
            raise RuntimeError("simulator did not publish fresh odom/IMU/leg-state")
        self.sim_ready_record = json.loads(ready_path.read_text(encoding="utf-8"))

    def reset_and_verify_before_mc(self) -> None:
        pid, title, window_id = self.inspect_window()
        before_image = self.capture_reset_image(pid, title, window_id, "before")
        self.dispatch_reset(pid, title, window_id)
        dispatch = json.loads((self.evidence / "reset_dispatch_event.json").read_text(encoding="utf-8"))
        event = dispatch.get("event", {})
        if event.get("status") != "KEY_DISPATCHED":
            raise RuntimeError("Reset key was not dispatched to the process-verified simulator window")
        dispatch_ns = int(event["dispatch_monotonic_ns"])
        trace_path = self.evidence / "n2_runtime_trace.jsonl"
        deadline = time.monotonic() + 2.0
        trace_rows: list[dict[str, Any]] = []
        post_by_key: dict[str, list[dict[str, Any]]] = {}
        while time.monotonic() < deadline:
            trace_rows = read_jsonl_complete_rows(trace_path)
            post_by_key = {}
            for row in trace_rows:
                key = row.get("topic_key")
                if (row.get("event") == "state" and key in {"odom", "imu", "leg_state"}
                        and int(row.get("wall_monotonic_ns", 0)) >= dispatch_ns):
                    post_by_key.setdefault(key, []).append(row)
            if len(post_by_key.get("odom", [])) >= 3 and all(post_by_key.get(key) for key in ("imu", "leg_state")):
                break
            time.sleep(0.05)
        post = post_by_key.get("odom", [])
        if len(post) < 3:
            raise RuntimeError(f"only {len(post)} live post-Reset odom rows before MC startup")
        if not post_by_key.get("imu") or not post_by_key.get("leg_state"):
            raise RuntimeError("Reset was not followed by fresh IMU and leg-state samples before MC startup")
        first = post[0]["data"]
        before_by_key: dict[str, dict[str, Any]] = {}
        for row in trace_rows:
            key = row.get("topic_key")
            if row.get("event") == "state" and key in {"odom", "imu", "leg_state"}:
                if int(row.get("wall_monotonic_ns", 0)) < dispatch_ns:
                    before_by_key[key] = row
        # Capture outside the initial sample wait, then make a second, fresh live-state
        # check so a slow screenshot cannot hide a fall before MC startup.
        after_image = self.capture_reset_image(pid, title, window_id, "after")
        trace_rows = read_jsonl_complete_rows(trace_path)
        post_by_key = {}
        for row in trace_rows:
            key = row.get("topic_key")
            if (row.get("event") == "state" and key in {"odom", "imu", "leg_state"}
                    and int(row.get("wall_monotonic_ns", 0)) >= dispatch_ns):
                post_by_key.setdefault(key, []).append(row)
        post = post_by_key.get("odom", [])
        if not post or not post_by_key.get("imu") or not post_by_key.get("leg_state"):
            raise RuntimeError("fresh post-Reset odom/IMU/leg-state was unavailable immediately before MC startup")
        latest_odom_row = post[-1]
        latest_odom = latest_odom_row["data"]
        latest_age_s = (time.monotonic_ns() - int(latest_odom_row["wall_monotonic_ns"])) / 1e9
        upright = (latest_age_s <= 0.5 and float(latest_odom["z_m"]) >= ROOT_Z_MIN_M
                   and max(abs(float(latest_odom["roll_deg"])), abs(float(latest_odom["pitch_deg"]))) < MAX_TILT_DEG)
        last = latest_odom
        self.pre_mc_record = {
            "schema": "robotsim.issue80.n2.pre-mc-init.v1", "wall_utc": utc_now(),
            "reset_dispatch": dispatch, "before_image": before_image, "after_image": after_image,
            "post_reset_odom_count": len(post), "first_post_reset_odom": first,
            "last_pre_mc_odom": last, "last_pre_mc_odom_wall_monotonic_ns": latest_odom_row["wall_monotonic_ns"],
            "last_pre_mc_odom_age_s": latest_age_s, "upright_before_mc": upright,
            "pre_reset_latest_samples": before_by_key,
            "first_post_reset_samples": {key: values[0] for key, values in post_by_key.items() if values},
            "last_pre_mc_samples": {key: values[-1] for key, values in post_by_key.items() if values},
            "pre_mc_motion_measurements": {
                "post_reset_odometry_linear_velocity_mps": first.get("linear_velocity_mps"),
                "post_reset_odometry_angular_velocity_radps": first.get("angular_velocity_radps"),
                "post_reset_leg_state_full_message": post_by_key["leg_state"][0].get("data"),
                "interpretation": "Measured values only; no additional stationarity thresholds are imposed by this diagnostic.",
            },
            "thresholds": {"root_z_min_m": ROOT_Z_MIN_M, "max_tilt_deg": MAX_TILT_DEG},
        }
        self.pre_mc_init = self.pre_mc_record
        write_new(self.evidence / "n2_pre_mc_initialization.json", self.pre_mc_init)
        if not upright:
            raise RuntimeError("simulator was not upright in the fresh live sample immediately before MC startup")
        self.n2_simulator_window = {"pid": pid, "title": title, "window_id": window_id}
        self.record_stage("reset_and_pre_mc_upright", "PASS", "fresh odom/IMU/leg-state plus pose gate")

    def start_mc_after_reset_once(self) -> None:
        self.docker_exec("n2_launch_mc_once_after_reset",
            "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1",
            timeout_s=4.0, detached=True)
        wait = "timeout 15s bash -c 'until pgrep -x aima-mc-app-main >/dev/null; do sleep 0.2; done'"
        self.docker_exec("n2_wait_mc_process", wait, timeout_s=17.0)
        self.mc_started = True
        self.record_stage("mc_process_started", "STARTED", "aima-mc-app-main was observed in the container")

    def observe_runtime_once(self) -> None:
        summary_path = self.evidence / "n2_runtime_summary.json"
        deadline = min(time.monotonic() + 43.0, self.deadline)
        while time.monotonic() < deadline and not summary_path.exists():
            time.sleep(0.1)
        if not summary_path.exists():
            raise TimeoutError("N2 read-only ROS trace did not finish within its bound")
        self.n2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        window = getattr(self, "n2_simulator_window", None)
        if window:
            self.capture_reset_image(window["pid"], window["title"], window["window_id"], "after_mc")

    def stop_runtime_processes(self) -> None:
        if not self.container_created:
            return
        result = self.docker_exec(
            "n2_graceful_stop_processes",
            "for p in aima-mc-app-main soc_sim aima-sim-app; do pkill -TERM -x \"$p\" 2>/dev/null || true; done; sleep 1; "
            "printf 'remaining processes: '; (pgrep -a -x aima-mc-app-main; pgrep -a -x soc_sim; pgrep -a -x aima-sim-app) 2>/dev/null || true",
            timeout_s=4.0, check=False,
        )
        with (self.evidence / "n2_process_stop.txt").open("x", encoding="utf-8") as stream:
            stream.write(result.stdout + result.stderr)

    def run_n2(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": "robotsim.issue80.n2.runtime-result.v1",
                                  "attempt_id": self.args.attempt_id, "status": "BLOCKED_BEFORE_RUNTIME",
                                  "fault": None}
        try:
            self.current_gate = "preflight"
            self.preflight_n2()
            self.record_stage("preflight", "PASS", f"{self.branch}@{self.head}")
            result["preflight"] = self.preflight_record
            self.current_gate = "input_snapshot"
            self.stage_n2_inputs()
            self.record_stage("input_snapshot", "PASS")
            self.current_gate = "container_startup"
            self.start_container()
            self.record_stage("container_startup", "PASS", self.container_id)
            self.current_gate = "simulator_startup"
            self.start_simulator_before_mc()
            self.current_gate = "reset_before_mc"
            self.reset_and_verify_before_mc()
            self.current_gate = "mc_startup_once"
            self.start_mc_after_reset_once()
            self.current_gate = "runtime_observation"
            self.observe_runtime_once()
            self.record_stage("runtime_observation", "PASS", "bounded read-only trace completed")
            result["runtime_summary"] = self.n2_summary
            result["status"] = self.classify_n2(self.n2_summary or {})
            result["first_safe_context"] = self.first_safe_context(self.n2_summary or {})
        except Exception as exc:
            result["fault"] = f"{type(exc).__name__}: {exc}"
            result["first_failing_gate"] = self.current_gate
            try:
                self.record_stage(self.current_gate, "FAIL", result["fault"])
            except Exception:
                pass
            result["status"] = "RUNTIME_DIAGNOSTIC_INCOMPLETE" if self.mc_started else (
                "RUNTIME_STARTUP_BLOCKED" if self.container_created else "BLOCKED_BEFORE_RUNTIME"
            )
        finally:
            try:
                self.stop_runtime_processes()
            except Exception as exc:
                self.cleanup_evidence["operations"].append({
                    "operation": "n2_graceful_stop_processes",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                self.cleanup()
            result["container_teardown"] = self.cleanup_evidence
            try:
                self.record_stage("cleanup", "PASS" if self.cleanup_evidence.get("container_absent_after_cleanup") else (
                    "NOT_NEEDED" if not self.container_created else "BLOCKED"))
            except Exception:
                pass

        after = ({relative: file_sha256(RUNTIME / relative) for relative in HASH_FILES
                  if (RUNTIME / relative).is_file()} if self.container_created else {})
        self.runtime_hash_changes = ({
            relative: {"before": self.runtime_hashes_before.get(relative), "after": after.get(relative)}
            for relative in HASH_FILES if self.runtime_hashes_before.get(relative) != after.get(relative)
        } if self.runtime_hashes_before and self.container_created else {})
        result.update({
            "branch": self.branch, "head": self.head, "container": self.container,
            "container_id": self.container_id, "simulator_ready": self.sim_ready_record,
            "pre_mc_initialization": self.pre_mc_init, "mc_started": self.mc_started,
            "runtime_sha256_before": self.runtime_hashes_before, "runtime_sha256_after": after,
            "runtime_hash_changes": self.runtime_hash_changes,
            "forbidden_operations": {"SetMcInputSource": 0, "SetMcAction": 0, "velocity": 0, "real_robot": False},
            "run_command": (
                f"DISPLAY=:0 python3 {Path(__file__).resolve()} --maintainer-authorized-n2 "
                f"--attempt-id {self.args.attempt_id} --evidence-dir {self.evidence} "
                f"--container {self.container} --max-runtime-s {self.args.max_runtime_s:.0f}"
            ),
            "runner_sha256": file_sha256(Path(__file__).resolve(strict=True)),
            "runner_worktree_relative": "scripts/research/issue80_n2_runtime_run.py",
        })
        if self.runtime_hash_changes:
            result["status"] = "RUNTIME_SOURCE_INTEGRITY_FAILURE"
            result["fault"] = result.get("fault") or "pinned runtime input hashes changed"
        if self.container_created and not self.cleanup_evidence.get("container_absent_after_cleanup", False):
            result["status"] = "CLEANUP_BLOCKED"
            result["fault"] = result.get("fault") or "isolated container absence not verified"
        result["artifact_manifest"] = "artifact_paths.json"
        result_path = self.evidence / "n2_runtime_result.json"
        write_new(result_path, result)
        self.write_n2_report(result)
        sys.path.insert(0, str(REPO / "scripts/agent"))
        import path_governance
        path_governance.write_artifact_manifest(self.evidence)
        return result

    @staticmethod
    def classify_n2(summary: dict[str, Any]) -> str:
        first = summary.get("first_mc_common_state") or {}
        last = summary.get("last_mc_common_state") or {}
        if first.get("fsm_state") == 4 or any(e.get("fsm_to") == 4 for e in summary.get("fsm_transitions", [])):
            return "RUNTIME_SAFE_CONFIRMED"
        if first.get("fsm_state") == 2 and last.get("fsm_state") == 2 and last.get("body_status") == 1:
            return "RUNTIME_STABLE_CONFIRMED"
        if first.get("fsm_state") is None:
            return "RUNTIME_MC_STATE_NOT_OBSERVED"
        return "RUNTIME_STATE_UNRESOLVED"

    def first_safe_context(self, summary: dict[str, Any]) -> dict[str, Any]:
        safe = summary.get("first_safe_state")
        trace_path = self.evidence / "n2_runtime_trace.jsonl"
        if not isinstance(safe, dict) or not trace_path.is_file():
            return {"status": "NO_FSM_SAFE_SAMPLE", "detail": "FSM=4 was not observed, or the trace was unavailable."}
        safe_ns = int(safe.get("wall_monotonic_ns", 0))
        nearest: dict[str, tuple[int, dict[str, Any]]] = {}
        with trace_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("topic_key")
                if row.get("event") != "state" or key not in {"odom", "imu", "leg_state", "leg_command", "common"}:
                    continue
                sample_ns = int(row.get("wall_monotonic_ns", 0))
                skew_ns = abs(sample_ns - safe_ns)
                if key not in nearest or skew_ns < nearest[key][0]:
                    nearest[key] = (skew_ns, row)
        context = {
            "status": "FSM_SAFE_TRIGGER_CONTEXT_CAPTURED",
            "first_safe_state": safe,
            "nearest_stream_samples": {
                key: {"skew_s": skew / 1e9, "sample": row}
                for key, (skew, row) in nearest.items()
            },
            "diagnostic_logs": [],
        }
        log_paths = [self.evidence / "mc-console.log"]
        mc_log_dir = self.evidence / "mc_logs"
        if mc_log_dir.is_dir():
            log_paths.extend(sorted(path for path in mc_log_dir.rglob("*") if path.is_file()))
        for path in log_paths:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            matches = [f"{index + 1}:{line}" for index, line in enumerate(lines)
                       if re.search(r"safe|fall|fault|error|protect|warning|warn", line, flags=re.IGNORECASE)]
            context["diagnostic_logs"].append({"path": str(path.relative_to(self.evidence)),
                                               "sha256": file_sha256(path),
                                               "matched_line_count": len(matches),
                                               "matches": matches[:200]})
        write_new(self.evidence / "first_safe_context.json", context)
        return context

    def write_n2_report(self, result: dict[str, Any]) -> None:
        summary = result.get("runtime_summary") or {}
        first = summary.get("first_mc_common_state") or {}
        services = summary.get("readonly_service_results") or {}
        report = f"""# Issue #80 N2 Actual MuJoCo + MC Runtime Diagnosis

- Status: **{result.get('status')}**
- Attempt ID: `{self.args.attempt_id}`
- Branch / HEAD: `{self.branch}` / `{self.head}`
- Container: `{self.container}` (`{self.container_id}`)
- Reset upright before MC: `{(self.pre_mc_init or {}).get('upright_before_mc')}`
- First MC FSM/mode/body: `{first.get('fsm_state')}` / `{first.get('action_desc')}` / `{first.get('body_status')}`
- Initial selected source: `{first.get('input_source')}`
- Read-only GetMcAction/GetCurrentInputSource: `{json.dumps(services, ensure_ascii=False)}`
- `/clock` message count / sim-time delta: `{summary.get('clock_sample_count')}` / `{summary.get('sim_clock_delta_s')}`
- Odom header stamp delta (time domain unverified): `{summary.get('odom_header_stamp_delta_s')}`
- FSM transitions: `{json.dumps(summary.get('fsm_transitions') or [], ensure_ascii=False)}`
- Cleanup verified: `{self.cleanup_evidence.get('container_absent_after_cleanup')}`
- Runtime hash changes: `{json.dumps(self.runtime_hash_changes, ensure_ascii=False)}`
- Fault: `{result.get('fault')}`

## Reproduction

```bash
{result.get('run_command')}
```

No source mutation, SetMcAction request, velocity command, or real robot connection occurred. If `/clock` is not published, exact MuJoCo internal time and physics-step count remain unobserved.
"""
        with (self.evidence / "REPORT.md").open("x", encoding="utf-8") as stream:
            stream.write(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maintainer-authorized-n2", action="store_true")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--max-runtime-s", type=float, default=150.0)
    args = parser.parse_args()
    if not args.maintainer_authorized_n2:
        parser.error("N2 run requires --maintainer-authorized-n2")
    result = N2Run(args).run_n2()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"RUNTIME_SAFE_CONFIRMED", "RUNTIME_STABLE_CONFIRMED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
