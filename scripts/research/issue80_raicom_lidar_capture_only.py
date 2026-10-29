#!/usr/bin/env python3
"""One bounded, simulator-only capture of the RAICOM map-frame LiDAR topic.

This dedicated runner is the first live gate for the obstacle-navigation work.
It starts the LiDAR-enabled simulator and the unchanged vendor MC process only
because this delivery instantiates its MuJoCo model after that native startup
handshake. It captures one ``PointCloud2`` topic and tears the isolated
container down. It never sends Reset, JD/SD/LD, an input-source request, a
velocity command, or a route.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from issue80_local01_scene_route import load_local01_scene_inputs
from issue80_m1_auto_reset_run import (
    BoundedRun,
    MC_SIM_CONFIG_DUMP_CONTAINER_PATH,
    MC_SIM_GENERATED_DIRECTORY,
    REPO,
    RUNTIME,
    validate_local01_aligned_start_table_approach_mission_input,
)
from issue80_raicom_reset_only import _stage_mc_config_dump_writeback


WORKTREE = Path(__file__).resolve().parents[2]
POSES = WORKTREE / "simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json"
CAPTURE = Path(__file__).with_name("issue80_capture_lidar_pointcloud.py")
MAP_FIELD = Path(__file__).with_name("issue80_lidar_map_obstacle_field.py")
MAX_RUNTIME_S = 30.0
MAX_CAPTURE_S = 12.0
KNOWN_GOOD_MISSION = REPO / "reviews/issue-80/raicom-vendor-table-target-20261028T042026Z/route_config.json"
KNOWN_GOOD_STATIC_PLAN = REPO / "reviews/issue-80/raicom-vendor-table-target-20261028T042026Z/table_approach_static_plan.json"
RUNTIME_IMMUTABLE_RELATIVE_ROOTS = (
    Path("sim_mujoco/configuration/robot/lx2501_3_t2d5"),
    Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5"),
)
RUNTIME_IMMUTABLE_FILES = (
    Path("sim_mujoco/bin/aima-sim-app"),
    Path("mc/bin/aima-mc-app-main"),
)
RUNTIME_GENERATED_PREFIXES = (
    Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim"),
)

FORBIDDEN_NAMES = frozenset({
    "SetMcAction", "SetMcInputSource", "McLocomotionVelocity", "RouteMotionBridge",
    "dispatch_native_reset_button", "run_m1",
    "create_publisher", "create_client", "publish",
})
ALLOWED_RUNTIME_METHODS = frozenset({
    "run", "docker_exec", "start_container", "start_sim_first", "start_mc_after_sim_topics", "cleanup",
    "apply_container_teardown_gate", "remaining", "stage_research_scripts",
    "prepare_native_mjdata_capture", "finalize_native_mjdata_capture",
    "prepare_foundationwalk_diagnostic_overlay",
})


class CaptureOnlyError(RuntimeError):
    """The capture-only policy or its isolated runtime preflight failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_uuid(value: str) -> str:
    parsed = uuid.UUID(value)
    if str(parsed) != value:
        raise ValueError("attempt ID must use canonical UUID spelling")
    return str(parsed)


def audit_capture_only_source(path: Path) -> dict[str, Any]:
    """Prove the runner's direct surface cannot perform robot control."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden: list[str] = []
    runtime_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            forbidden.append(node.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            forbidden.append(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "runtime":
                runtime_calls.append(node.func.attr)
                if node.func.attr not in ALLOWED_RUNTIME_METHODS:
                    forbidden.append("runtime." + node.func.attr)
    if forbidden:
        raise CaptureOnlyError("capture-only runner contains forbidden control symbols: " + ", ".join(sorted(set(forbidden))))
    if not {"start_container", "start_sim_first", "docker_exec", "cleanup"}.issubset(runtime_calls):
        raise CaptureOnlyError("capture-only runner lacks the required isolated lifecycle calls")
    return {
        "status": "PASS_CAPTURE_ONLY_STATIC_CAPABILITY_AUDIT",
        "forbidden_control_symbols": [],
        "runtime_methods": runtime_calls,
        "set_mc_action_requests": 0,
        "set_mc_input_source_requests": 0,
        "velocity_publishers_created": 0,
        "native_reset_dispatches": 0,
        "mc_process_started": True,
        "mujoco_step_calls_by_runner": 0,
    }


def _host_preflight(container: str) -> dict[str, Any]:
    if os.environ.get("DISPLAY") != ":0":
        raise CaptureOnlyError("DISPLAY must be exactly :0")
    if subprocess.run(["docker", "info"], text=True, capture_output=True, timeout=8).returncode != 0:
        raise CaptureOnlyError("Docker daemon is unavailable")
    active = subprocess.run(["docker", "ps", "-q"], text=True, capture_output=True, timeout=5, check=True).stdout.split()
    if active:
        raise CaptureOnlyError("another Docker runtime is active; capture-only runner refuses to overlap experiments")
    if subprocess.run(["docker", "inspect", container], text=True, capture_output=True, timeout=5).returncode == 0:
        raise CaptureOnlyError("requested container name already exists")
    return {"status": "PASS_HOST_PREFLIGHT", "display": ":0", "active_container_count": 0, "container": container}


def _bounded_args(
    evidence: Path, container: str, lidar_overlay_source: Path, *, runtime_write_comparison: bool,
    known_good_startup_profile: bool, table_preflight: dict[str, Any] | None,
) -> argparse.Namespace:
    """Construct only the BoundedRun fields needed for a simulator-only launch."""
    return argparse.Namespace(
        evidence_dir=str(evidence), container=container, max_runtime_s=MAX_RUNTIME_S,
        local01_scene_world_poses=POSES if known_good_startup_profile else None,
        route_config=None,
        table_approach_mission=KNOWN_GOOD_MISSION if known_good_startup_profile else None,
        table_approach_static_plan=KNOWN_GOOD_STATIC_PLAN if known_good_startup_profile else None,
        configured_table_approach_preflight=table_preflight,
        aligned_start_table_approach=known_good_startup_profile,
        native_x2_timestep=known_good_startup_profile,
        synchronized_paused_initialization=False, static_nominal_start_initialization=False,
        mc_prearm_before_simulator=False, direct_stand_from_passive=False,
        raicom_lidar_overlay_source=str(lidar_overlay_source), raicom_lidar_static_map_source=None,
        runtime_readonly=not runtime_write_comparison, mc_sim_generated_tmpfs=True,
        # This is the already-validated read-only LD_PRELOAD capture used by
        # the successful LiDAR baseline. It does not produce control signals.
        native_mjdata_capture=True, native_mjdata_capture_interval_steps=10,
        forward_speed=0.0, pulse_duration=0.0,
    )


def snapshot_runtime_immutable_inputs() -> dict[str, Any]:
    """Hash all source/model/policy inputs before/after an opt-in RW comparison."""
    records: dict[str, str] = {}
    for relative_root in RUNTIME_IMMUTABLE_RELATIVE_ROOTS:
        root = RUNTIME / relative_root
        if root.is_symlink() or not root.is_dir():
            raise CaptureOnlyError(f"runtime immutable root missing or unsafe: {relative_root}")
        for path in sorted(root.rglob("*")):
            relative_path = path.relative_to(RUNTIME)
            if any(relative_path == generated or generated in relative_path.parents for generated in RUNTIME_GENERATED_PREFIXES):
                continue
            if path.is_symlink():
                raise CaptureOnlyError(f"runtime immutable root contains symlink: {path}")
            if path.is_file():
                records[str(relative_path)] = _sha256(path)
    for relative_path in RUNTIME_IMMUTABLE_FILES:
        path = RUNTIME / relative_path
        if path.is_symlink() or not path.is_file():
            raise CaptureOnlyError(f"runtime immutable file missing or unsafe: {relative_path}")
        records[str(relative_path)] = _sha256(path)
    return {
        "schema": "robotsim.issue80.runtime-write-comparison-inputs.v1",
        "runtime": str(RUNTIME),
        "immutable_file_sha256": records,
        "immutable_file_count": len(records),
        "permitted_generated_write_locations": [
            "sim_mujoco/bin/log",
            "sim_mujoco/bin/cfg/tmp",
            "mc/bin/log",
            "mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim",
        ],
    }


def stage_capture_tools(evidence: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for source in (CAPTURE, MAP_FIELD):
        if source.is_symlink() or not source.is_file():
            raise CaptureOnlyError(f"capture input is missing or unsafe: {source.name}")
        target = evidence / source.name
        if target.exists():
            raise CaptureOnlyError("capture input target already exists")
        shutil.copyfile(source, target)
        records[source.name] = _sha256(target)
    return records


def start_simulator_then_native_mc(runtime: BoundedRun) -> None:
    """Launch the installed simulator/MC model-init handshake without actions.

    The walking runner waits for a loaded model before it launches MC. The
    failed first capture established that this precompiled delivery's model
    creation itself needs the native MC/simulation startup handshake. Keep
    that dependency explicit, then wait for the same evidence after MC starts.
    """
    runtime.docker_exec(
        "launch_simulator_capture_only",
        "ROBOTSIM_LIDAR_RAW_TRACE=/workspace/evidence/lidar_raw_frames.jsonl "
        "ROBOTSIM_LIDAR_RAW_ENABLE_FILE=/workspace/evidence/lidar_raw_enable "
        "ROBOTSIM_SIMULATOR_CFG_OVERRIDE=/workspace/evidence/lidar_overlay/simulator_lidar_overlay.yaml "
        "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1",
        timeout_s=4.0,
        detached=True,
    )
    runtime.docker_exec(
        "wait_simulator_module_ready_capture_only",
        "timeout 12s bash -lc 'until grep -q \"MujocoSimModule Started Succeeded\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'",
        timeout_s=13.0,
    )
    runtime.start_mc_after_sim_topics()
    runtime.docker_exec(
        "wait_lidar_model_loaded_after_native_mc_startup",
        "timeout 12s bash -lc 'until grep -q \"Odom initialized: body_id=1, name=pelvis\" /workspace/evidence/mujoco-console.log && grep -q \"issue80_lidar_init ok\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'",
        timeout_s=13.0,
    )


def run_once(
    evidence: Path, container: str, lidar_overlay_source: Path, duration_s: float, *, runtime_write_comparison: bool,
    known_good_startup_profile: bool,
) -> dict[str, Any]:
    table_preflight = (
        validate_local01_aligned_start_table_approach_mission_input(POSES, KNOWN_GOOD_MISSION, KNOWN_GOOD_STATIC_PLAN)
        if known_good_startup_profile else None
    )
    runtime = BoundedRun(_bounded_args(
        evidence, container, lidar_overlay_source,
        runtime_write_comparison=runtime_write_comparison,
        known_good_startup_profile=known_good_startup_profile,
        table_preflight=table_preflight,
    ))
    source = table_preflight["source"] if table_preflight is not None else load_local01_scene_inputs(POSES, require_clean_commit=True)
    if not known_good_startup_profile:
        # BoundedRun needs this mount solely because the existing LiDAR overlay
        # references the pinned room by /workspace/raicom_scene_source. No route
        # configuration or motion adapter is staged or executed.
        runtime.local01_scene_bundle = {"classification": "CAPTURE_ONLY_REFERENCE_MOUNT"}
        runtime.local01_scene_source = source
        runtime.local01_scene_readonly_mounts = []
    staged_mc_dump, mc_writeback_manifest = _stage_mc_config_dump_writeback(evidence)
    runtime.runtime_writeback_mounts = [(staged_mc_dump, MC_SIM_CONFIG_DUMP_CONTAINER_PATH)]
    result: dict[str, Any] = {
        "schema": "robotsim.issue80.raicom-lidar-capture-only.v1",
        "classification": "SIMULATION_ONLY_READ_ONLY_MAP_FRAME_CAPTURE",
        "status": "BLOCKED",
        "lidar_overlay_source": str(lidar_overlay_source),
        "source_local01_commit": source["source_commit"],
        "duration_s": duration_s,
        "set_mc_action_requests": 0,
        "set_mc_input_source_requests": 0,
        "locomotion_velocity_publish_calls": 0,
        "native_reset_dispatches": 0,
        "waypoint_execution_calls": 0,
        "vendor_mc_process_started_for_simulator_model_handshake": True,
        "mc_runtime_writeback": mc_writeback_manifest,
        "runtime_write_comparison": runtime_write_comparison,
        "known_good_startup_profile": known_good_startup_profile,
    }
    before_runtime = snapshot_runtime_immutable_inputs() if runtime_write_comparison else None
    if before_runtime is not None:
        (evidence / "runtime_immutable_before.json").write_text(json.dumps(before_runtime, indent=2) + "\n", encoding="utf-8")
    try:
        runtime.stage_research_scripts()
        runtime.prepare_native_mjdata_capture()
        result["native_mjdata_capture_overlay"] = runtime.native_mjdata_overlay
        result["native_mjdata_capture_smoke"] = runtime.native_mjdata_smoke
        result["capture_tool_sha256"] = stage_capture_tools(evidence)
        runtime.start_container()
        runtime.docker_exec(
            "seed_isolated_generated_sim_config_capture_only",
            "cp /workspace/evidence/mc_runtime_writeback/sim_seed/*.yaml "
            + MC_SIM_GENERATED_DIRECTORY + "/",
            timeout_s=2.0,
        )
        if known_good_startup_profile:
            runtime.prepare_foundationwalk_diagnostic_overlay()
            runtime.start_sim_first(start_mc=False)
        else:
            start_simulator_then_native_mc(runtime)
        command = (
            "cd /workspace/evidence && timeout " + f"{duration_s + 3.0:g}s "
            "python3 issue80_capture_lidar_pointcloud.py "
            "--output /workspace/evidence/lidar_map_frame_capture.json "
            f"--duration-s {duration_s:g} --expected-frame-id map --save-all-finite-points"
        )
        captured = runtime.docker_exec("capture_map_frame_lidar", command, timeout_s=duration_s + 6.0, check=False)
        result["capture_returncode"] = captured.returncode
        output = evidence / "lidar_map_frame_capture.json"
        if output.is_file() and not output.is_symlink():
            result["capture"] = json.loads(output.read_text(encoding="utf-8"))
            result["status"] = "PASS" if result["capture"].get("status") == "PASS_NONEMPTY_POINTCLOUD" else "BLOCKED"
        else:
            result["failure"] = "capture process produced no result file"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["native_mjdata_capture"] = runtime.finalize_native_mjdata_capture()
        runtime.cleanup()
        runtime.apply_container_teardown_gate(result)
        result["cleanup"] = runtime.cleanup_evidence
        if before_runtime is not None:
            after_runtime = snapshot_runtime_immutable_inputs()
            changed = {
                path: {"before": before_runtime["immutable_file_sha256"].get(path), "after": after_runtime["immutable_file_sha256"].get(path)}
                for path in sorted(set(before_runtime["immutable_file_sha256"]) | set(after_runtime["immutable_file_sha256"]))
                if before_runtime["immutable_file_sha256"].get(path) != after_runtime["immutable_file_sha256"].get(path)
            }
            comparison = {"before": before_runtime, "after": after_runtime, "changed_immutable_files": changed}
            (evidence / "runtime_immutable_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
            result["runtime_immutable_comparison"] = {
                "status": "PASS_NO_IMMUTABLE_RUNTIME_CHANGE" if not changed else "FAIL_IMMUTABLE_RUNTIME_CHANGED",
                "changed_immutable_files": changed,
            }
            if changed:
                result["status"] = "BLOCKED"
                result["failure"] = "runtime write comparison changed immutable source/model/policy input"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--lidar-overlay-source", type=Path, required=True)
    parser.add_argument("--evidence-parent", type=Path, default=REPO / "reviews/issue-80")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument(
        "--allow-runtime-write-comparison", action="store_true",
        help="one logged full-runtime RW comparison; source/model/policy hashes must remain unchanged",
    )
    parser.add_argument(
        "--known-good-startup-profile", action="store_true",
        help="stage the same source-derived scene/FoundationWalk startup inputs as the passing LiDAR baseline; still no action or velocity",
    )
    args = parser.parse_args(argv)
    try:
        args.attempt_id = _parse_uuid(args.attempt_id)
        if not 1.0 <= args.duration_s <= MAX_CAPTURE_S:
            raise ValueError(f"duration must be in [1, {MAX_CAPTURE_S:g}] seconds")
        overlay_source = args.lidar_overlay_source.resolve(strict=True)
        overlay_source.relative_to(REPO / "reviews/issue-80")
        if (overlay_source / "lidar_overlay").is_symlink() or not (overlay_source / "lidar_overlay").is_dir():
            raise ValueError("LiDAR overlay source is incomplete or unsafe")
        source_audit = audit_capture_only_source(Path(__file__).resolve())
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
        from path_governance import create_unique_output_directory, write_artifact_manifest
        evidence = create_unique_output_directory(args.evidence_parent, prefix=f"raicom-lidar-mapframe-capture-{args.attempt_id[:8]}").native_path
        container = f"issue80-lidar-capture-{args.attempt_id}"
        host = _host_preflight(container)
        (evidence / "capture_only_preflight.json").write_text(json.dumps({
            "schema": "robotsim.issue80.raicom-lidar-capture-only-preflight.v1",
            "attempt_id": args.attempt_id, "source_audit": source_audit, "host": host,
        }, indent=2) + "\n", encoding="utf-8")
        result = run_once(
            evidence, container, overlay_source, args.duration_s,
            runtime_write_comparison=args.allow_runtime_write_comparison,
            known_good_startup_profile=args.known_good_startup_profile,
        )
        (evidence / "lidar_capture_only_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_artifact_manifest(evidence)
    except Exception as exc:
        parser.error(f"capture-only preflight failed: {type(exc).__name__}: {exc}")
    print(json.dumps({"status": result["status"], "attempt_id": args.attempt_id, "evidence_dir": str(evidence), "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
