#!/usr/bin/env python3
"""One bounded N3 reproduction of the previously used MuJoCo -> MC -> SD -> Reset sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import issue80_n2_runtime_run as n2
from issue80_m1_auto_reset_run import BoundedRun, IMAGE, IMAGE_DIGEST

REPO = Path("/home/lzy18001500226/projects/RobotSim")
WORKTREE = Path("/home/lzy18001500226/robotsim-issue80-x2-navigation-20261019")
TRACE_SCRIPT = WORKTREE / "scripts/research/issue80_n2_runtime_trace.py"
EXPECTED_BRANCH = "research/issue80-x2-navigation-m1-diagnostic-20261019"
EXPECTED_HEAD = "617ebaba0fa3e291f5fcea795df6c582b5f15535"
ROOT_Z_SCREENING_M = 0.52
MAX_TILT_SCREENING_DEG = 15.0
POST_RESET_OBSERVE_S = 30.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return n2.read_jsonl_complete_rows(path)


def qpos_map(row: dict[str, Any] | None) -> dict[str, float]:
    if not row:
        return {}
    return {str(j["name"]): float(j["position"]) for j in row.get("data", {}).get("joints", [])
            if j.get("name") is not None and isinstance(j.get("position"), (int, float))}


def max_joint_delta(first: dict[str, float], second: dict[str, float]) -> dict[str, Any]:
    common = sorted(first.keys() & second.keys())
    delta = {name: second[name] - first[name] for name in common}
    return {"common_joint_count": len(common), "max_abs_rad": max((abs(v) for v in delta.values()), default=None),
            "per_joint_rad": delta}


def imu_tilt_deg(row: dict[str, Any]) -> tuple[float, float] | None:
    q = row.get("data", {}).get("orientation")
    if not isinstance(q, dict) or not all(isinstance(q.get(k), (int, float)) for k in ("x", "y", "z", "w")):
        return None
    x, y, z, w = (float(q[k]) for k in ("x", "y", "z", "w"))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return math.degrees(roll), math.degrees(math.asin(sin_pitch))


class N3Run(BoundedRun):
    """Reuse the validated M1 container, X11 Reset, mode request, and cleanup helpers."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.branch = ""
        self.head = ""
        self.mc_started = False
        self.trace_started = False
        self.reset_event: dict[str, Any] | None = None
        self.reset_attempt_ns: int | None = None
        self.reset_analysis: dict[str, Any] | None = None
        self.summary: dict[str, Any] | None = None
        self.runtime_hashes_before: dict[str, str] = {}
        self.runtime_hashes_after: dict[str, str] = {}
        self.runtime_hash_changes: dict[str, Any] = {}
        self.mode_request: dict[str, Any] = {"status": "NOT_ATTEMPTED"}
        self.first_failing_gate: str | None = None
        self.simulator_window: dict[str, Any] | None = None
        self.post_observation_image: dict[str, Any] | None = None
        self.current_gate = "preflight"

    def record_stage(self, stage: str, outcome: str, detail: str | None = None) -> None:
        path = self.evidence / "n3_stage_timeline.jsonl"
        item = {"stage": stage, "outcome": outcome, "wall_utc": n2.utc_now(),
                "wall_monotonic_ns": time.monotonic_ns(), "detail": detail}
        with path.open("a" if path.exists() else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def record_stage(self, stage: str, outcome: str, detail: str | None = None) -> None:
        path = self.evidence / "n3_stage_timeline.jsonl"
        item = {"stage": stage, "outcome": outcome, "wall_utc": n2.utc_now(),
                "wall_monotonic_ns": time.monotonic_ns(), "detail": detail}
        with path.open("a" if path.exists() else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def preflight_n3(self) -> None:
        attempt_path = self.evidence / "attempt.json"
        attempt_record = json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt_record.get("attempt_id") != self.args.attempt_id or attempt_record.get("authorized_attempts") != 1:
            raise RuntimeError("evidence reservation does not match the one-shot N3 authorization")
        existing = sorted(p.name for p in self.evidence.iterdir() if p.name not in {
            ".robotsim-output-reservation.json", "attempt.json"})
        if existing:
            raise RuntimeError(f"N3 evidence directory was not fresh at preflight start: {existing}")
        actual_root = self.run("n3_worktree_root", ["git", "-C", str(WORKTREE), "rev-parse", "--show-toplevel"]).stdout.strip()
        if Path(actual_root).resolve(strict=True) != WORKTREE.resolve(strict=True):
            raise RuntimeError(f"worktree root mismatch: {actual_root}")
        actual_runner = Path(__file__).resolve(strict=True)
        if actual_runner != (WORKTREE / "scripts/research/issue80_n3_restore_m1_startup.py").resolve(strict=True):
            raise RuntimeError(f"N3 runner is not in the verified worktree: {actual_runner}")
        self.branch = self.run("n3_branch", ["git", "-C", str(WORKTREE), "branch", "--show-current"]).stdout.strip()
        self.head = self.run("n3_head", ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]).stdout.strip()
        if (self.branch, self.head) != (EXPECTED_BRANCH, EXPECTED_HEAD):
            raise RuntimeError(f"starting branch/HEAD mismatch: {self.branch}@{self.head}")
        docker = self.run("n3_docker_info", ["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}"]).stdout.strip()
        image = self.run("n3_image_id", ["docker", "image", "inspect", "--format", "{{.Id}} {{.Os}}", IMAGE]).stdout.strip()
        if not image.startswith(IMAGE_DIGEST + " linux"):
            raise RuntimeError(f"pinned image mismatch: {image}")
        if os.environ.get("DISPLAY") != ":0":
            raise RuntimeError(f"expected DISPLAY=:0, got {os.environ.get('DISPLAY')!r}")
        self.run("n3_x11_probe", ["xset", "-q"], timeout_s=5.0)
        inventory = self.run("n3_container_inventory", ["docker", "ps", "-a", "--no-trunc", "--format",
            "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"])
        containers = []
        for line in inventory.stdout.splitlines():
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise RuntimeError(f"malformed Docker inventory row: {line!r}")
            containers.append({"id": parts[0], "name": parts[1], "image": parts[2], "status": parts[3]})
        if any(c["name"] == self.container for c in containers):
            raise RuntimeError(f"N3 container name already exists: {self.container}")
        active = [c for c in containers if c["status"].startswith("Up ")
                  and ("raicom2026" in c["name"] or "humble-raicom-runtime" in c["image"])]
        if active:
            raise RuntimeError(f"conflicting RAICOM containers are active: {active}")
        processes = {}
        for process in ("aima-sim-app", "aima-mc-app-main", "soc_sim"):
            p = self.run("n3_host_process_" + process, ["pgrep", "-a", "-x", process], check=False)
            processes[process] = p.stdout.strip().splitlines()
            if p.returncode == 0 and p.stdout.strip():
                raise RuntimeError(f"conflicting host process: {p.stdout.strip()}")
        ignored = self.run("n3_runtime_ignore", ["git", "-C", str(REPO), "check-ignore", "-v",
            "simulation/mujoco/raicom2026"]).stdout.strip()
        self.runtime_hashes_before = {relative: sha256(n2.RUNTIME / relative) for relative in n2.HASH_FILES}
        self.preflight_record = {
            "schema": "robotsim.issue80.n3.preflight.v1", "attempt_id": self.args.attempt_id,
            "worktree_root": actual_root, "runner_path": str(actual_runner), "branch": self.branch,
            "head": self.head, "attempt_record": attempt_record, "docker_server": docker, "image_id": image.split()[0],
            "containers": containers, "host_processes": processes, "display": os.environ["DISPLAY"],
            "runtime_git_ignore": ignored, "runtime_sha256_before": self.runtime_hashes_before,
            "fsm_codes": {"STABLE": 2, "MOVING": 3, "SAFE": 4},
            "prohibited_operations": ["SetMcInputSource", "LD/LOCOMOTION_DEFAULT", "nonzero velocity", "qpos writes", "real robot"],
        }
        write_new(self.evidence / "n3_preflight.json", self.preflight_record)

    def stage_n3_inputs(self) -> None:
        n2.N2Run.stage_n2_inputs(self)
        source = Path(__file__).resolve(strict=True)
        target = self.evidence / "issue80_n3_restore_m1_startup.py"
        shutil.copy2(source, target)
        write_new(self.evidence / "n3_runner_sha256.json", {"path": target.name, "sha256": sha256(target)})

    def start_sim_and_observer(self) -> None:
        self.docker_exec("n3_launch_simulator",
            "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1",
            timeout_s=4.0, detached=True)
        wait = "timeout 25s bash -c 'until grep -Fq \"MujocoSimModule Started Succeeded\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'"
        self.docker_exec("n3_wait_simulator_ready", wait, timeout_s=27.0)
        self.record_stage("simulator_ready", "PASS", "MujocoSimModule Started Succeeded")
        observer = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "exec python3 /workspace/evidence/issue80_n2_runtime_trace.py "
            "--output-dir /workspace/evidence --duration 55 "
            "--stop-file /workspace/evidence/n3_observation_stop"
        )
        self.docker_exec("n3_start_early_readonly_trace", observer, timeout_s=4.0, detached=True)
        self.trace_started = True
        ready_path = self.evidence / "n2_sim_ready.json"
        deadline = min(self.deadline, time.monotonic() + 8.0)
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.05)
        if not ready_path.exists():
            raise RuntimeError("fresh simulator odom/IMU/leg-state was not observed")
        self.sim_ready_record = json.loads(ready_path.read_text(encoding="utf-8"))

    def start_mc_once(self) -> None:
        self.docker_exec("n3_launch_mc_after_sim_ready",
            "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1",
            timeout_s=4.0, detached=True)
        # The installed WSL launcher did start MotionControlModule, but this
        # distribution did not satisfy the assumed `pgrep -x aima-mc-app-main`
        # name. Gate on the module's own startup marker, then require live
        # McCommonState below before sending the single SD request.
        self.docker_exec("n3_wait_mc_module_started",
            "timeout 15s bash -c 'until grep -Fq \"MotionControlModule Started Succeeded\" /workspace/evidence/mc-console.log; do sleep 0.2; done'",
            timeout_s=17.0)
        self.mc_started = True
        self.record_stage("mc_module_started", "PASS", "MotionControlModule Started Succeeded in mc-console.log")
        trace_path = self.evidence / "n2_runtime_trace.jsonl"
        deadline = min(self.deadline, time.monotonic() + 5.0)
        while time.monotonic() < deadline:
            if any(r.get("event") == "state" and r.get("topic_key") == "common" for r in read_rows(trace_path)):
                return
            time.sleep(0.05)
        raise RuntimeError("MC process started but no fresh McCommonState was observed before SD request")

    def stop_trace_early(self, reason: str) -> None:
        if not self.trace_started:
            return
        marker = self.evidence / "n3_observation_stop"
        if not marker.exists():
            marker.write_text(reason + "\n", encoding="utf-8")

    def command_text(self, name: str) -> dict[str, Any] | None:
        step = next((item for item in reversed(self.steps) if item.get("name") == name), None)
        if not step:
            return None
        output = {}
        for label in ("stdout_path", "stderr_path"):
            path = Path(step[label])
            output[label.removesuffix("_path")] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        output["returncode"] = step.get("returncode")
        output["started_wall_utc"] = step.get("started_wall_utc")
        return output

    def wait_after_reset_30s(self, start_ns: int) -> tuple[bool, str | None]:
        target_ns = start_ns + int(POST_RESET_OBSERVE_S * 1e9)
        next_health_check = time.monotonic()
        while time.monotonic_ns() < target_ns and time.monotonic() < self.deadline:
            now = time.monotonic()
            if now >= next_health_check:
                live = self.docker_exec("n3_health_check", "pgrep -x aima-sim-app >/dev/null && pgrep -x aima-mc-app-main >/dev/null",
                                         timeout_s=3.0, check=False)
                if live.returncode != 0:
                    return False, "simulator or MC process exited during post-Reset observation"
                next_health_check = now + 5.0
            if (self.evidence / "n2_runtime_summary.json").exists():
                return False, "trace ended before 30 seconds elapsed after Reset"
            time.sleep(min(0.1, max(0.0, (target_ns - time.monotonic_ns()) / 1e9)))
        if time.monotonic_ns() < target_ns:
            return False, "150-second overall runtime deadline reached before 30-second post-Reset window completed"
        self.stop_trace_early("post-Reset 30-second observation complete")
        deadline = min(self.deadline, time.monotonic() + 4.0)
        summary_path = self.evidence / "n2_runtime_summary.json"
        while time.monotonic() < deadline and not summary_path.exists():
            time.sleep(0.05)
        if not summary_path.exists():
            return False, "read-only trace did not write its summary after the stop marker"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return True, None

    def analyze_reset_and_runtime(self) -> dict[str, Any]:
        trace_path = self.evidence / "n2_runtime_trace.jsonl"
        rows = read_rows(trace_path)
        dispatch_doc = self.reset_event or {}
        event = dispatch_doc.get("event", {}) if isinstance(dispatch_doc, dict) else {}
        reset_ns_value = event.get("dispatch_monotonic_ns") or self.reset_attempt_ns
        reset_ns = int(reset_ns_value) if reset_ns_value is not None else None
        state_rows = [r for r in rows if r.get("event") == "state"]
        pre: dict[str, dict[str, Any]] = {}
        post: dict[str, list[dict[str, Any]]] = {}
        for row in state_rows:
            key = row.get("topic_key")
            if key not in {"odom", "imu", "leg_state", "leg_command", "common"}:
                continue
            t = int(row.get("wall_monotonic_ns", 0))
            if reset_ns is None or t < reset_ns:
                pre[key] = row
            elif t >= reset_ns:
                post.setdefault(key, []).append(row)
        immediate = {key: values[0] for key, values in post.items() if values}
        last = {key: values[-1] for key, values in post.items() if values}
        pre_odom = pre.get("odom", {}).get("data", {})
        first_odom = immediate.get("odom", {}).get("data", {})
        last_odom = last.get("odom", {}).get("data", {})
        odom_delta = {key: first_odom.get(key) - pre_odom.get(key)
                      for key in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg")
                      if isinstance(first_odom.get(key), (int, float)) and isinstance(pre_odom.get(key), (int, float))}
        joint_delta = max_joint_delta(qpos_map(pre.get("leg_state")), qpos_map(immediate.get("leg_state")))
        fsm_rows = [r for r in post.get("common", []) if isinstance(r.get("data", {}).get("fsm_state"), int)]
        post_fsms = sorted({int(r["data"]["fsm_state"]) for r in fsm_rows})
        first_safe = (self.summary or {}).get("first_safe_state")
        first_stable = next((r for r in fsm_rows if r["data"].get("fsm_state") == 2), None)
        first_upright_odom = None
        matched_imu_tilt = None
        for odom_row in post.get("odom", []):
            odom_data = odom_row.get("data", {})
            if float(odom_data.get("z_m", -999.0)) < ROOT_Z_SCREENING_M:
                continue
            odom_ns = int(odom_row.get("wall_monotonic_ns", 0))
            imu_candidates = [(abs(int(i.get("wall_monotonic_ns", 0)) - odom_ns), i)
                              for i in post.get("imu", [])]
            if not imu_candidates:
                continue
            skew_ns, imu_row = min(imu_candidates, key=lambda item: item[0])
            tilt = imu_tilt_deg(imu_row)
            if (skew_ns <= 500_000_000 and tilt is not None
                    and max(abs(tilt[0]), abs(tilt[1])) < MAX_TILT_SCREENING_DEG):
                first_upright_odom = odom_row
                matched_imu_tilt = {"roll_deg": tilt[0], "pitch_deg": tilt[1], "odom_imu_skew_s": skew_ns / 1e9}
                break
        odom_times = [int(r["wall_monotonic_ns"]) for r in post.get("odom", [])]
        post_span = (odom_times[-1] - odom_times[0]) / 1e9 if len(odom_times) > 1 else 0.0
        post_from_reset = ((odom_times[-1] - reset_ns) / 1e9 if odom_times and reset_ns is not None else 0.0)
        post_z = [float(r["data"]["z_m"]) for r in post.get("odom", []) if isinstance(r.get("data", {}).get("z_m"), (int, float))]
        post_tilts = [max(abs(float(r["data"]["roll_deg"])), abs(float(r["data"]["pitch_deg"])))
                      for r in post.get("odom", []) if isinstance(r.get("data", {}).get("roll_deg"), (int, float))]
        common_last = fsm_rows[-1] if fsm_rows else None
        reset_key_delivered = event.get("status") == "KEY_DISPATCHED"
        reset_attempted = self.reset_attempt_ns is not None
        safe_ns = int(first_safe.get("wall_monotonic_ns", 0)) if isinstance(first_safe, dict) else None
        return {
            "schema": "robotsim.issue80.n3.analysis.v1",
            "first_mc_state_before_reset": (self.summary or {}).get("first_mc_common_state"),
            "first_safe_state": first_safe,
            "first_safe_observed_before_reset": bool(safe_ns is not None and reset_ns is not None and safe_ns < reset_ns),
            "first_safe_observed_after_reset": bool(safe_ns is not None and reset_ns is not None and safe_ns >= reset_ns),
            "fsm_transitions": (self.summary or {}).get("fsm_transitions", []),
            "fsm_values_observed_after_reset": post_fsms,
            "first_stable_state_after_reset": first_stable,
            "last_mc_common_state_after_reset": common_last,
            "reset_key_delivery_status": event.get("status"),
            "reset_attempted": reset_attempted,
            "reset_event": event,
            "reset_attempt_wall_monotonic_ns": self.reset_attempt_ns,
            "pre_reset_latest_samples": pre,
            "first_post_reset_samples": immediate,
            "last_post_reset_samples": last,
            "first_post_reset_odom_minus_pre_reset_latest": odom_delta,
            "first_post_reset_joint_position_delta": joint_delta,
            "first_post_reset_upright_odometry_sample": first_upright_odom,
            "first_post_reset_upright_odom_with_imu_tilt": ({"odom": first_upright_odom, "imu_tilt": matched_imu_tilt}
                                                             if first_upright_odom else None),
            "post_reset_observation_wall_span_s": post_span,
            "post_reset_observation_from_reset_event_s": post_from_reset,
            "post_reset_odom_z_reported_m_range": [min(post_z), max(post_z)] if post_z else None,
            "post_reset_odom_tilt_reported_deg_max": max(post_tilts) if post_tilts else None,
            "post_reset_30s_window_complete": post_from_reset >= POST_RESET_OBSERVE_S,
            "odom_semantics": {
                "topic_frame_id": (first_odom.get("frame_id") if first_odom else None),
                "source_mapping": "model_info/default.yaml maps the odom sensor to pelvis; x2.xml body-linear-pos is measured at imu_0, whose local pelvis offset is (0.0239465, -0.000228682, 0.04171) m.",
                "interpretation": "Reported odom z is retained as odometry data and is not asserted to equal raw MuJoCo pelvis-origin world z.",
                "screening_limits_used": {"odom_z_min_m": ROOT_Z_SCREENING_M, "max_tilt_deg": MAX_TILT_SCREENING_DEG},
            },
            "sim_clock_sample_count": (self.summary or {}).get("clock_sample_count"),
            "sim_clock_delta_s": (self.summary or {}).get("sim_clock_delta_s"),
            "odom_header_stamp_delta_s_time_domain_unverified": (self.summary or {}).get("odom_header_stamp_delta_s"),
            "recorded_stream_samples": {key: len(values) for key, values in post.items()},
            "readonly_service_results": (self.summary or {}).get("readonly_service_results", {}),
            "readonly_service_status": (self.summary or {}).get("readonly_service_status", {}),
            "reset_effect_conclusion": (
                "RESET_NOT_ATTEMPTED" if not reset_attempted else
                "RESET_KEY_NOT_DELIVERED" if not reset_key_delivered else
                "RESET_TO_UPRIGHT_OBSERVED" if first_upright_odom is not None else
                "RESET_EVENT_DELIVERED_BUT_UPRIGHT_POSTURE_NOT_OBSERVED"
            ),
        }

    def classify_n3(self, analysis: dict[str, Any]) -> str:
        summary = self.summary or {}
        all_common = [r for r in read_rows(self.evidence / "n2_runtime_trace.jsonl")
                      if r.get("event") == "state" and r.get("topic_key") == "common"]
        if not self.mc_started or not all_common:
            return "MC_STARTUP_FAILED"
        fsms = analysis.get("fsm_values_observed_after_reset", [])
        last_live = all_common[-1].get("data", {})
        if last_live.get("fsm_state") == 2 or any(r.get("data", {}).get("fsm_state") == 2 for r in all_common):
            return "MC_RUNTIME_OBSERVED_STABLE"
        if 4 in fsms or summary.get("first_safe_state") is not None or any(
                r.get("data", {}).get("fsm_state") == 4 for r in all_common):
            return "MC_RUNTIME_OBSERVED_SAFE"
        if not analysis.get("reset_attempted"):
            return "MC_STARTUP_FAILED"
        if analysis.get("reset_effect_conclusion") != "RESET_TO_UPRIGHT_OBSERVED":
            return "SIM_RESET_FAILED"
        return "MC_STARTUP_FAILED"

    def stop_processes_and_cleanup(self) -> None:
        try:
            if self.container_created:
                self.docker_exec("n3_graceful_process_stop",
                    "for p in aima-mc-app-main soc_sim aima-sim-app; do pkill -TERM -x \"$p\" 2>/dev/null || true; done; sleep 1; "
                    "printf 'remaining processes: '; (pgrep -a -x aima-mc-app-main; pgrep -a -x soc_sim; pgrep -a -x aima-sim-app) 2>/dev/null || true",
                    timeout_s=4.0, check=False)
        except Exception as exc:
            self.cleanup_evidence["operations"].append({"operation": "n3_graceful_process_stop", "error": f"{type(exc).__name__}: {exc}"})
        self.cleanup()

    def run_n3(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": "robotsim.issue80.n3.runtime-result.v1",
            "attempt_id": self.args.attempt_id, "status": "MC_STARTUP_FAILED", "fault": None}
        try:
            self.current_gate = "preflight"
            self.preflight_n3()
            self.record_stage("preflight", "PASS", f"{self.branch}@{self.head}")
            n2.N2Run.stage_n2_inputs(self)
            script_target = self.evidence / "issue80_n3_restore_m1_startup.py"
            shutil.copy2(Path(__file__).resolve(strict=True), script_target)
            write_new(self.evidence / "n3_runner_sha256.json", {"path": script_target.name, "sha256": sha256(script_target)})
            self.record_stage("input_snapshot", "PASS")
            self.current_gate = "container_startup"
            self.start_container()
            self.record_stage("container_startup", "PASS", self.container_id)
            self.current_gate = "simulator_startup_and_early_observer"
            self.start_sim_and_observer()
            self.current_gate = "mc_startup"
            self.start_mc_once()
            self.current_gate = "state_inventory"
            self.capture_service_topic_inventory()
            self.mode_request = {"status": "REQUESTING_SD", "interface": "/workspace/raicom2026/example/py/set_mode.py SD"}
            self.current_gate = "SD_STAND_DEFAULT"
            try:
                readback = self.request_sd_and_verify()
                self.mode_request = {"status": "STAND_DEFAULT_READBACK_CONFIRMED", "set_mode_example": self.command_text("request_sd"),
                                     "get_mode_output": readback}
                self.record_stage("sd_request_and_readback", "PASS", "STAND_DEFAULT")
            except Exception as exc:
                self.mode_request = {"status": "REJECTED_OR_NO_STAND_READBACK", "set_mode_example": self.command_text("request_sd"),
                                     "last_mode_readback": self.command_text("read_mode_stand"),
                                     "error": f"{type(exc).__name__}: {exc}"}
                self.record_stage("sd_request_and_readback", "FAIL", self.mode_request["error"])
                self.stop_trace_early("SD request rejected or STAND_DEFAULT readback failed; no Reset was sent")
                raise

            self.current_gate = "reset_window_and_before_image"
            pid, title, window_id = self.inspect_window()
            self.simulator_window = {"pid": pid, "title": title, "window_id": window_id}
            before_image = self.capture_reset_image(pid, title, window_id, "before")
            self.reset_attempt_ns = time.monotonic_ns()
            self.current_gate = "RESET_ONCE"
            try:
                self.dispatch_reset(pid, title, window_id)
            except Exception as exc:
                self.reset_event = json.loads((self.evidence / "reset_dispatch_event.json").read_text(encoding="utf-8")) if (self.evidence / "reset_dispatch_event.json").exists() else None
                self.mode_request["reset_dispatch_error"] = f"{type(exc).__name__}: {exc}"
            if self.reset_event is None and (self.evidence / "reset_dispatch_event.json").exists():
                self.reset_event = json.loads((self.evidence / "reset_dispatch_event.json").read_text(encoding="utf-8"))
            if self.reset_event is None:
                self.reset_event = json.loads((self.evidence / "reset_dispatch_event.json").read_text(encoding="utf-8"))
            dispatch_event = self.reset_event.get("event", {})
            reset_start_ns = int(dispatch_event.get("dispatch_monotonic_ns") or self.reset_attempt_ns)
            self.record_stage("reset_key_dispatch", str(dispatch_event.get("status", "UNKNOWN")),
                              "one process/window-verified BackSpace; state effect checked from ROS trace")
            self.wait_for_immediate_reset_samples(reset_start_ns)
            after_image = self.capture_reset_image(pid, title, window_id, "after")
            write_new(self.evidence / "n3_reset_images.json", {"before": before_image, "after": after_image})
            self.current_gate = "POST_RESET_OBSERVATION"
            complete, observation_fault = self.wait_after_reset_30s(reset_start_ns)
            result["observation_window_complete"] = complete
            result["observation_fault"] = observation_fault
            if not (self.evidence / "n2_runtime_summary.json").exists():
                self.stop_trace_early(observation_fault or "N3 observation ended")
                self.wait_trace_summary()
            else:
                self.summary = json.loads((self.evidence / "n2_runtime_summary.json").read_text(encoding="utf-8"))
            if self.simulator_window:
                self.post_observation_image = self.capture_reset_image(pid, title, window_id, "post_observation")
            analysis = self.analyze_reset_and_runtime()
            try:
                analysis["first_safe_context"] = n2.N2Run.first_safe_context(self, self.summary or {})
            except Exception as exc:
                analysis["first_safe_context"] = {"status": "ANALYSIS_ERROR", "error": f"{type(exc).__name__}: {exc}"}
            write_new(self.evidence / "n3_runtime_analysis.json", analysis)
            result["analysis"] = analysis
            result["status"] = self.classify_n3(analysis)
            result["mode_request"] = self.mode_request
            if observation_fault:
                result["fault"] = observation_fault
        except Exception as exc:
            self.first_failing_gate = self.current_gate
            result["fault"] = f"{type(exc).__name__}: {exc}"
            result["first_failing_gate"] = self.current_gate
            if self.trace_started and not (self.evidence / "n2_runtime_summary.json").exists():
                self.stop_trace_early(result["fault"])
                self.wait_trace_summary()
            result["mode_request"] = self.mode_request
            if self.mc_started and (self.evidence / "n2_runtime_summary.json").exists():
                self.summary = json.loads((self.evidence / "n2_runtime_summary.json").read_text(encoding="utf-8"))
                partial = self.analyze_reset_and_runtime()
                try:
                    partial["first_safe_context"] = n2.N2Run.first_safe_context(self, self.summary)
                except Exception as context_exc:
                    partial["first_safe_context"] = {"status": "ANALYSIS_ERROR",
                        "error": f"{type(context_exc).__name__}: {context_exc}"}
                result["analysis"] = partial
                result["status"] = self.classify_n3(partial)
            elif not self.mc_started:
                result["status"] = "MC_STARTUP_FAILED"
        finally:
            self.stop_processes_and_cleanup()
            result["cleanup"] = self.cleanup_evidence
        after = ({relative: n2.file_sha256(n2.RUNTIME / relative) for relative in n2.HASH_FILES
                  if (n2.RUNTIME / relative).is_file()} if self.container_created else {})
        changes = ({rel: {"before": self.runtime_hashes_before.get(rel), "after": after.get(rel)}
                    for rel in n2.HASH_FILES if self.runtime_hashes_before.get(rel) != after.get(rel)}
                   if self.runtime_hashes_before and self.container_created else {})
        result.update({
            "branch": self.branch, "head_before_commit": self.head, "container": self.container,
            "container_id": self.container_id, "simulator_ready": getattr(self, "sim_ready_record", None),
            "mc_started": self.mc_started, "runtime_sha256_before": self.runtime_hashes_before,
            "runtime_sha256_after": after, "runtime_hash_changes": changes,
            "container_teardown": self.cleanup_evidence,
            "post_observation_image": self.post_observation_image,
            "forbidden_operations": {"SetMcInputSource": 0, "LD": 0, "nonzero_velocity": 0,
                                      "qpos_writes": 0, "real_hardware": False},
            "run_command": (f"DISPLAY=:0 python3 {Path(__file__).resolve()} --maintainer-authorized-n3 "
                            f"--attempt-id {self.args.attempt_id} --evidence-dir {self.evidence} "
                            f"--container {self.container} --max-runtime-s {self.args.max_runtime_s:.0f}"),
            "runner_sha256": sha256(Path(__file__).resolve(strict=True)),
            "limitations": ["No locomotion command or source mutation was issued.",
                            "A missing /clock is reported as unavailable simulation-clock evidence, not treated as a simulation failure.",
                            "Odometry z is reported in its declared frame and is not assumed to equal raw pelvis-origin world z."],
        })
        if changes:
            result["status"] = "MC_STARTUP_FAILED" if result.get("status") not in {
                "MC_RUNTIME_OBSERVED_STABLE", "MC_RUNTIME_OBSERVED_SAFE"} else result["status"]
            result["fault"] = result.get("fault") or "pinned runtime inputs changed during the run"
        if self.container_created and not self.cleanup_evidence.get("container_absent_after_cleanup", False):
            result["fault"] = result.get("fault") or "container absence after cleanup was not verified"
        write_new(self.evidence / "n3_runtime_result.json", result)
        self.write_n3_report(result)
        sys.path.insert(0, str(REPO / "scripts/agent"))
        import path_governance
        path_governance.write_artifact_manifest(self.evidence)
        return result

    def wait_for_immediate_reset_samples(self, reset_ns: int) -> None:
        trace_path = self.evidence / "n2_runtime_trace.jsonl"
        deadline = min(self.deadline, time.monotonic() + 2.0)
        while time.monotonic() < deadline:
            rows = read_rows(trace_path)
            keys = {r.get("topic_key") for r in rows if r.get("event") == "state"
                    and int(r.get("wall_monotonic_ns", 0)) >= reset_ns}
            if {"odom", "imu", "leg_state", "common"}.issubset(keys):
                return
            time.sleep(0.05)

    def wait_after_reset_30s(self, start_ns: int) -> tuple[bool, str | None]:
        target = start_ns + int(POST_RESET_OBSERVE_S * 1e9)
        next_health = time.monotonic()
        while time.monotonic_ns() < target and time.monotonic() < self.deadline:
            if time.monotonic() >= next_health:
                check = self.docker_exec("n3_health_check", "pgrep -x aima-sim-app >/dev/null && pgrep -x aima-mc-app-main >/dev/null",
                                         timeout_s=3.0, check=False)
                if check.returncode != 0:
                    return False, "simulator or MC process exited during observation"
                next_health = time.monotonic() + 5.0
            if (self.evidence / "n2_runtime_summary.json").exists():
                return False, "trace ended before the 30-second post-Reset interval"
            time.sleep(min(0.1, max(0.0, (target - time.monotonic_ns()) / 1e9)))
        if time.monotonic_ns() < target:
            return False, "overall runtime bound expired before the 30-second post-Reset interval"
        self.stop_trace_early("30 seconds after the single Reset attempt")
        return self.wait_trace_summary()

    def wait_trace_summary(self) -> tuple[bool, str | None]:
        summary_path = self.evidence / "n2_runtime_summary.json"
        deadline = min(self.deadline, time.monotonic() + 4.0)
        while time.monotonic() < deadline and not summary_path.exists():
            time.sleep(0.05)
        if not summary_path.exists():
            return False, "read-only trace summary was not written within 4 seconds"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return True, None

    def write_n3_report(self, result: dict[str, Any]) -> None:
        analysis = result.get("analysis") or {}
        summary = self.summary or {}
        report = f"""# Issue #80 N3 Restore M1 Startup Sequence

- Outcome: **{result.get('status')}**
- Attempt: `{self.args.attempt_id}`
- Branch / pre-commit HEAD: `{self.branch}` / `{self.head}`
- Container: `{self.container}` (`{self.container_id}`)
- MC process started: `{self.mc_started}`
- SD procedure: `{json.dumps(result.get('mode_request'), ensure_ascii=False)}`
- Reset key status: `{(analysis.get('reset_key_delivery_status') if analysis else None)}`
- Reset effect conclusion: `{(analysis.get('reset_effect_conclusion') if analysis else None)}`
- First SAFE state: `{json.dumps(analysis.get('first_safe_state'), ensure_ascii=False)}`
- FSM values after Reset: `{analysis.get('fsm_values_observed_after_reset')}`
- Post-Reset observation span: `{analysis.get('post_reset_observation_from_reset_event_s')} s from Reset; complete={analysis.get('post_reset_30s_window_complete')}`
- Read-only services: `{json.dumps((summary.get('readonly_service_results') or {}), ensure_ascii=False)}`
- `/clock` sample count / delta: `{summary.get('clock_sample_count')} / {summary.get('sim_clock_delta_s')}`
- Cleanup verified: `{self.cleanup_evidence.get('container_absent_after_cleanup')}`
- Runtime hash changes: `{json.dumps(result.get('runtime_hash_changes'), ensure_ascii=False)}`
- Fault: `{result.get('fault')}`

## Reproduction

```bash
{result.get('run_command')}
```

No LD, nonzero velocity, input-source mutation, qpos write, or hardware connection was used. The odometry stream declares its frame and sensor mapping in the result; its z value is not represented as raw MuJoCo pelvis-origin height. No `/clock` message is interpreted as a MuJoCo time failure.
"""
        with (self.evidence / "REPORT.md").open("x", encoding="utf-8") as stream:
            stream.write(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maintainer-authorized-n3", action="store_true")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--max-runtime-s", type=float, default=150.0)
    args = parser.parse_args()
    if not args.maintainer_authorized_n3:
        parser.error("N3 live run requires --maintainer-authorized-n3")
    if args.max_runtime_s > 150.0 or args.max_runtime_s < 60.0:
        parser.error("N3 runtime bound must be 60..150 seconds")
    result = N3Run(args).run_n3()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"MC_RUNTIME_OBSERVED_STABLE", "MC_RUNTIME_OBSERVED_SAFE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
