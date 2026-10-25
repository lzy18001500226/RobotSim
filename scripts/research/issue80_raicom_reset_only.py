#!/usr/bin/env python3
"""One-shot, capture-only Reset telemetry diagnostic for the pinned RAICOM room.

The runner starts the existing simulator and MC, subscribes to state only, and
dispatches one verified native MuJoCo Simulation/Reset button click. It does
not create an action/velocity publisher, call an MC service, or execute a
waypoint route. The entire run is capped at 30 seconds; a host watchdog stops
the isolated container at 26 seconds if normal cleanup has not started.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from issue80_local01_scene_route import (
    LOCAL01_COMMIT,
    LOCAL01_SCENE_RELATIVE,
    build_short_route_config,
    load_local01_scene_inputs,
    prepare_local01_scene_runtime_bundle,
    validate_short_route_static_scene,
)
from issue80_m1_auto_reset_run import (
    BoundedRun,
    FOUNDATIONWALK_CONFIG_CONTAINER_PATH,
    FOUNDATIONWALK_CONFIG_RELATIVE,
    IMAGE,
    IMAGE_DIGEST,
    MC_LAUNCH,
    REPO,
    RUN38_ACTION_SETTING_SHA256,
    RUN38_FOUNDATIONWALK_CONFIG_SHA256,
    RUN38_FOUNDATIONWALK_ONNX_SHA256,
    RUN38_FOUNDATIONWALK_RUNNER_SHA256,
    RUN38_MC_SIM_CONFIG_SHA256,
    RUN38_ROBOT_MODEL_SHA256,
    RUNTIME,
    SIM_LAUNCH,
    X11_HELPER,
    build_runtime_bind_mount,
    build_run41_foundationwalk_min_run_overlay,
)


MAX_RUNTIME_S = 30.0
ACTIVE_CONTAINER_WATCHDOG_S = 26.0
OBSERVER_MAX_RUNTIME_S = 15.0
RESET_ONLY_SCHEMA = "robotsim.issue80.raicom-reset-only.v1"
RESET_OBSERVER = Path(__file__).with_name("issue80_raicom_reset_only_observer.py")
LOCAL01_POSES_DEFAULT = Path(
    "/home/lzy18001500226/robotsim-issue77-raicom-x2-reuse/"
    "simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json"
)
EVIDENCE_PARENT_DEFAULT = REPO / "reviews/issue-80"
FOUNDATIONWALK_INPUTS = {
    "foundationwalk_config": (
        FOUNDATIONWALK_CONFIG_RELATIVE,
        RUN38_FOUNDATIONWALK_CONFIG_SHA256,
    ),
    "foundationwalk_onnx": (
        Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl_models/foundationwalk_v1.onnx"),
        RUN38_FOUNDATIONWALK_ONNX_SHA256,
    ),
    "foundationwalk_runner": (
        Path("mc/lib/libmc_runner_rl_foundationwalk.so.0.0.0"),
        RUN38_FOUNDATIONWALK_RUNNER_SHA256,
    ),
    "action_setting": (
        Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/action_setting.yaml"),
        RUN38_ACTION_SETTING_SHA256,
    ),
    "robot_model": (
        Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/robot_model.yaml"),
        RUN38_ROBOT_MODEL_SHA256,
    ),
    "mc_sim_config": (
        Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"),
        RUN38_MC_SIM_CONFIG_SHA256,
    ),
}
SAFE_INFRA_METHODS = frozenset({
    "run", "docker_exec", "start_container", "start_sim_first",
    "capture_service_topic_inventory", "inspect_window",
    "dispatch_native_reset_button", "cleanup", "remaining",
    "apply_container_teardown_gate",
})
FORBIDDEN_AST_NAMES = frozenset({
    "SetMcAction", "SetMcInputSource", "McLocomotionVelocity",
    "WaypointRouteExecutor", "RouteMotionBridge", "create_publisher",
    "create_client", "publish", "set_mc_action", "set_mc_input_source",
    "register_locomotion_source", "execute_waypoint_route", "run_m1",
})
REQUIRED_STATE_KEYS = frozenset({"odom", "imu", "common", "leg_state", "leg_command"})


class ResetOnlyError(RuntimeError):
    """A Reset-only preflight or runtime gate failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_attempt_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("attempt ID must be a UUID") from exc
    if str(parsed) != value:
        raise ValueError("attempt ID must use canonical UUID spelling")
    return str(parsed)


def validate_policy(*, max_runtime_s: float, reset_dispatches: int = 1,
                    route_requested: bool = False, motion_requested: bool = False) -> None:
    if max_runtime_s <= 0 or max_runtime_s > MAX_RUNTIME_S:
        raise ValueError("Reset-only wall runtime must be in (0, 30] seconds")
    if reset_dispatches != 1:
        raise ValueError("Reset-only policy permits exactly one native Reset event")
    if route_requested or motion_requested:
        raise ValueError("Reset-only policy forbids waypoint routes and locomotion")


def required_state_ages_pass(ages: Mapping[str, Any] | None, *, max_age_s: float = 0.5) -> bool:
    if not isinstance(ages, Mapping) or set(ages) != REQUIRED_STATE_KEYS:
        return False
    return all(
        isinstance(age, (int, float)) and not isinstance(age, bool)
        and math.isfinite(float(age)) and 0.0 <= float(age) <= max_age_s
        for age in ages.values()
    )


def build_reset_only_runtime_args(evidence: Path, container: str) -> argparse.Namespace:
    return argparse.Namespace(
        evidence_dir=str(evidence),
        container=container,
        max_runtime_s=MAX_RUNTIME_S,
        local01_scene_world_poses=None,
        route_config=None,
        runtime_readonly=True,
    )


def audit_reset_only_source(runner_path: Path, observer_path: Path) -> dict[str, Any]:
    """Fail closed if either dedicated script grows an action or motion path."""
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    observer_tree = ast.parse(observer_path.read_text(encoding="utf-8"), filename=str(observer_path))
    infra_calls: list[str] = []
    reset_dispatch_calls = 0
    forbidden: list[str] = []

    def is_runtime_object(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name) and node.id == "runtime"
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "runtime"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    for tree, label in ((runner_tree, "runner"), (observer_tree, "observer")):
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_AST_NAMES:
                forbidden.append(f"{label}:name:{node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_AST_NAMES:
                    forbidden.append(f"{label}:attribute:{node.attr}")
                if (
                    label == "observer"
                    and node.attr == "subscriptions"
                    and isinstance(node.ctx, ast.Store)
                ):
                    forbidden.append("observer:reserved-rclpy-node-attribute:subscriptions")
            if label == "runner" and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if is_runtime_object(node.func.value):
                    if node.func.attr not in SAFE_INFRA_METHODS:
                        forbidden.append(f"runner:runtime-call:{node.func.attr}")
                    else:
                        infra_calls.append(node.func.attr)
                if node.func.attr == "dispatch_native_reset_button":
                    reset_dispatch_calls += 1
    if forbidden:
        raise ValueError("Reset-only static capability audit found forbidden control paths: " + ", ".join(sorted(set(forbidden))))
    if reset_dispatch_calls != 1:
        raise ValueError(f"Reset-only runner must contain exactly one Reset-dispatch call, found {reset_dispatch_calls}")
    if not infra_calls or not set(infra_calls).issubset(SAFE_INFRA_METHODS):
        raise ValueError("Reset-only runner uses an unreviewed runtime infrastructure method")
    return {
        "status": "PASS_RESET_ONLY_STATIC_CAPABILITY_AUDIT",
        "forbidden_motion_or_action_symbols": [],
        "reset_dispatch_call_sites": reset_dispatch_calls,
        "runtime_method_calls": infra_calls,
        "allowed_runtime_methods": sorted(SAFE_INFRA_METHODS),
        "observer_publishers": 0,
        "observer_service_clients": 0,
        "route_executor_imported": False,
        "mujoco_step_calls": 0,
        "assurance_scope": (
            "AST review of the dedicated runner/observer direct call surface; "
            "this is not an operating-system sandbox proof for imported infrastructure."
        ),
        "runtime_wall_cap_s": MAX_RUNTIME_S,
        "native_reset_clicks_max": 1,
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _command(command: list[str], *, timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def _x_display_check(display: str) -> dict[str, Any]:
    if display != ":0":
        return {"status": "BLOCKED", "display": display, "reason": "DISPLAY must be exactly :0"}
    library = ctypes.util.find_library("X11")
    if not library:
        return {"status": "BLOCKED", "display": display, "reason": "libX11 unavailable"}
    x11 = ctypes.CDLL(library)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int
    connection = x11.XOpenDisplay(display.encode())
    if not connection:
        return {"status": "BLOCKED", "display": display, "reason": "XOpenDisplay failed"}
    x11.XCloseDisplay(connection)
    return {"status": "PASS", "display": display, "identity_verified": False}


def _host_process_conflicts() -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if comm.casefold() in {
            "aima-sim-app", "soc_sim", "em_run", "motioncontrolmodule",
        } or "mujoco" in comm.casefold():
            conflicts.append({"pid": int(entry.name), "comm": comm})
    return conflicts


def _prepare_foundationwalk_overlay(evidence: Path) -> tuple[Path, dict[str, Any]]:
    import yaml

    hashes: dict[str, str] = {}
    for label, (relative, expected) in FOUNDATIONWALK_INPUTS.items():
        path = RUNTIME / relative
        if path.is_symlink() or not path.is_file():
            raise ResetOnlyError(f"verified FoundationWalk {label} input is absent or unsafe")
        actual = _sha256(path)
        if actual != expected:
            raise ResetOnlyError(f"FoundationWalk {label} hash mismatch: {actual}")
        hashes[label] = actual
    action_setting = yaml.safe_load((RUNTIME / FOUNDATIONWALK_INPUTS["action_setting"][0]).read_text(encoding="utf-8"))
    expected_runner = [{"name": "rl_foundationwalk", "cfg_path": "./rl/foundationwalk_config.yaml"}]
    if action_setting["transition_actions"]["LOCOMOTION_DEFAULT"][0]["runners"] != expected_runner:
        raise ResetOnlyError("LOCOMOTION_DEFAULT runner mapping differs from the pinned FoundationWalk profile")
    source = RUNTIME / FOUNDATIONWALK_CONFIG_RELATIVE
    overlay_bytes, manifest = build_run41_foundationwalk_min_run_overlay(source.read_bytes())
    target = evidence / "foundationwalk_knee019_minrun010.yaml"
    with target.open("xb") as stream:
        stream.write(overlay_bytes)
    manifest.update({
        "classification": "SIMULATION_ONLY_EXPERIMENTAL_FOUNDATIONWALK_OVERLAY",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "overlay_path": str(target),
        "verified_runtime_input_sha256": hashes,
        "runtime_image_digest": IMAGE_DIGEST,
        "runtime_mc_binary_modified": False,
        "runtime_onnx_modified": False,
        "source_limits_modified": False,
        "safety_gates_modified": False,
        "used_for_this_capture_only": True,
        "locomotion_source_registered": False,
        "locomotion_action_requested": False,
        "changed_values": [
            {"config_path": "BaseConfig.action_scale[9]", "vendor_source": 0.25, "simulation_only_overlay": 0.19},
            {"config_path": "BaseConfig.action_scale[10]", "vendor_source": 0.25, "simulation_only_overlay": 0.19},
            {"config_path": "FoundationWalkConfig.foundationwalk.min_run_velx", "vendor_source": 0.5, "simulation_only_overlay": 0.1, "semantics": "empirical field hypothesis; not vendor-default"},
        ],
    })
    manifest["overlay_sha256"] = hashlib.sha256(overlay_bytes).hexdigest()
    _write_json(evidence / "foundationwalk_config_overlay_manifest.json", manifest)
    return target, manifest


def _scene_expected_poses(source: Mapping[str, Any]) -> dict[str, Any]:
    config = source["scene_world_poses"]
    poses = config["poses"]
    transform = config["transforms"]["odom_from_scene_world"]
    raw_start = poses["START"]
    xyz = raw_start["xyz_m"]
    expected_scene = {
        "x_m": float(xyz[0]), "y_m": float(xyz[1]), "z_m": float(xyz[2]),
        "yaw_rad": float(raw_start["yaw_rad"]),
    }
    route = build_short_route_config(source)
    return {
        "scene_world_start": expected_scene,
        "reset_local_odom_anchor": route["reset_anchor_expected_odom"],
        "odom_from_scene_world": transform,
    }


def _pose_error(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, float]:
    dx = float(observed["x_m"]) - float(expected["x_m"])
    dy = float(observed["y_m"]) - float(expected["y_m"])
    dz = float(observed["z_m"]) - float(expected["z_m"])
    dyaw = math.atan2(
        math.sin(float(observed["yaw_rad"]) - float(expected["yaw_rad"])),
        math.cos(float(observed["yaw_rad"]) - float(expected["yaw_rad"])),
    )
    return {"xy_error_m": math.hypot(dx, dy), "z_error_m": abs(dz), "yaw_error_rad": abs(dyaw)}


def classify_odom_frame(observed: Mapping[str, Any] | None,
                        candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(observed, Mapping):
        return {"classification": "UNRESOLVED_NO_POST_RESET_ODOM", "candidate_errors": {}}
    errors: dict[str, Any] = {}
    for label, pose in candidates.items():
        try:
            errors[label] = _pose_error(observed, pose)
        except (KeyError, TypeError, ValueError):
            errors[label] = {"error": "candidate or observation missing pose fields"}
    matches = [
        label for label, value in errors.items()
        if isinstance(value, Mapping)
        and value.get("xy_error_m", float("inf")) <= 0.020
        and value.get("z_error_m", float("inf")) <= 0.020
        and value.get("yaw_error_rad", float("inf")) <= 0.020
    ]
    if len(matches) == 1:
        classification = matches[0]
    elif len(matches) > 1:
        classification = "AMBIGUOUS_MULTIPLE_CANDIDATES"
    else:
        classification = "NEITHER_CANDIDATE_OR_OUTSIDE_20MM_0P02RAD"
    return {
        "classification": classification,
        "candidate_errors": errors,
        "comparison_tolerance": {"xy_m": 0.020, "z_m": 0.020, "yaw_rad": 0.020},
        "used_as_motion_gate": False,
    }


def _static_preflight(args: argparse.Namespace, evidence: Path) -> dict[str, Any]:
    source = load_local01_scene_inputs(args.local01_scene_world_poses, require_clean_commit=True)
    if source["source_commit"] != LOCAL01_COMMIT:
        raise ResetOnlyError("Local01 scene source commit does not match the pinned revision")
    route_static_input = build_short_route_config(source)
    scene_root = Path(source["repository_root"]) / LOCAL01_SCENE_RELATIVE.parent
    scene_static = validate_short_route_static_scene(scene_root, route_static_input)
    if scene_static.get("status") != "PASS_STATIC_SCENE_CLEARANCE":
        raise ResetOnlyError("corrected RAICOM scene static/name/clearance preflight did not pass")
    contract = scene_static.get("native_runtime_name_contract", {})
    if contract.get("status") != "PASS_NATIVE_RUNTIME_NAMES":
        raise ResetOnlyError("corrected RAICOM scene does not resolve native X2 runtime names")
    if scene_static.get("native_timestep_s") != 0.001 or scene_static.get("combined_scene_timestep_s") != 0.002:
        raise ResetOnlyError("expected native 0.001 s versus RAICOM 0.002 s timesteps changed")
    bundle = prepare_local01_scene_runtime_bundle(source, evidence, runtime_root=RUNTIME)
    overlay_path, overlay = _prepare_foundationwalk_overlay(evidence)
    scene_dir = evidence / "local01_scene"
    for source_path, staged_name in ((SIM_LAUNCH, "launch_sim.sh"), (MC_LAUNCH, "launch_mc.sh"), (RESET_OBSERVER, RESET_OBSERVER.name)):
        if source_path.is_symlink() or not source_path.is_file():
            raise ResetOnlyError(f"required reset-only runtime input is missing: {source_path.name}")
        target = evidence / staged_name
        with target.open("xb") as stream:
            stream.write(source_path.read_bytes())
    _write_json(evidence / "local01_scene_static_preflight.json", scene_static)
    static_paths = [
        (scene_dir / "scene_workspace_runtime.xml", "/workspace/evidence/local01_scene/scene_workspace_runtime.xml"),
        (scene_dir / "mug_portable_runtime.xml", "/workspace/evidence/local01_scene/mug_portable_runtime.xml"),
        (scene_dir / "simulator_default_local01_scene.yaml", "/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml"),
        (scene_dir / "runtime_bundle_manifest.json", "/workspace/evidence/local01_scene/runtime_bundle_manifest.json"),
    ]
    return {
        "source": source,
        "static_scene": scene_static,
        "bundle": bundle,
        "overlay_path": overlay_path,
        "overlay_manifest": overlay,
        "scene_mounts": static_paths,
        "expected_poses": _scene_expected_poses(source),
    }


def _host_preflight(args: argparse.Namespace, container_name: str) -> dict[str, Any]:
    docker_info = _command(["docker", "info", "--format", "{{.ServerVersion}}"])
    if docker_info["status"] != "PASS":
        raise ResetOnlyError("Docker daemon is unavailable")
    server_version = docker_info["stdout"].strip()
    image_id = _command(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE])
    if image_id["status"] != "PASS" or image_id["stdout"].strip() != IMAGE_DIGEST:
        raise ResetOnlyError("the pinned isolated RAICOM runtime image is absent or has a different image ID")
    all_containers = _command(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if all_containers["status"] != "PASS":
        raise ResetOnlyError("unable to verify that the unique Reset-only container name is absent")
    container_names = all_containers["stdout"].splitlines()
    if container_name in container_names:
        raise ResetOnlyError("fresh Reset-only container name is already in use")
    running = _command(["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}"])
    if running["status"] != "PASS":
        raise ResetOnlyError("unable to verify absence of conflicting runtime containers")
    active = [line for line in running["stdout"].splitlines() if IMAGE in line or "robotsim-issue80" in line]
    if active:
        raise ResetOnlyError("a conflicting RobotSim/MC/MuJoCo container is active")
    host_conflicts = _host_process_conflicts()
    if host_conflicts:
        raise ResetOnlyError(f"conflicting host simulator/controller process detected: {host_conflicts}")
    display = _x_display_check(os.environ.get("DISPLAY", ""))
    if display["status"] != "PASS":
        raise ResetOnlyError("verified WSLg display is unavailable")
    return {
        "status": "PASS",
        "docker_server_version": server_version,
        "docker_image": IMAGE,
        "docker_image_id": image_id["stdout"].strip(),
        "expected_image_digest": IMAGE_DIGEST,
        "running_container_list": running["stdout"].splitlines(),
        "all_container_names_checked": container_names,
        "unique_container_absent_before_start": True,
        "conflicting_container_matches": active,
        "host_process_conflicts": host_conflicts,
        "display": display,
        "network_mode_required": "bridge",
        "container_unprivileged_required": True,
        "runtime_bind_mount": build_runtime_bind_mount(RUNTIME, readonly=True),
    }


def _verify_loaded_scene(runtime: BoundedRun, bundle: Mapping[str, Any]) -> dict[str, Any]:
    expected_config = "/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml"
    expected_scene = "/workspace/evidence/local01_scene/scene_workspace_runtime.xml"
    name_contract = bundle.get("generated_runtime_name_contract", {})
    required_joints = name_contract.get("configured_active_joints", [])
    if not isinstance(required_joints, list) or not required_joints or not all(
        isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+", name) for name in required_joints
    ):
        raise ResetOnlyError("native runtime joint lookup list is missing or malformed")
    missing_joint_patterns = " ".join(
        "-e " + shlex.quote("Can not find joint state: " + name)
        for name in required_joints
    )
    check = runtime.docker_exec(
        "verify_reset_only_scene_process_and_native_lookups",
        "set -e; pid=$(pgrep -x soc_sim | head -n1); test -n \"$pid\"; "
        "argv=$(tr '\\0' ' ' < /proc/$pid/cmdline); envs=$(tr '\\0' '\\n' < /proc/$pid/environ); "
        f"printf 'PID=%s\\nARGV=%s\\n' \"$pid\" \"$argv\"; "
        f"printf '%s\\n' \"$envs\" | grep -F 'ROBOTSIM_SIMULATOR_CFG_OVERRIDE={expected_config}'; "
        f"printf '%s\\n' \"$argv\" | grep -F -- '--cfg_file_path={expected_config}'; "
        "! grep -E -F \"Odom base body 'pelvis' not found\" /workspace/evidence/mujoco-console.log; "
        "grep -F 'Odom initialized:' /workspace/evidence/mujoco-console.log | grep -F 'name=pelvis'; "
        f"! grep -F {missing_joint_patterns} /workspace/evidence/mujoco-console.log; "
        "grep -F 'MujocoSimModule Started Succeeded' /workspace/evidence/mujoco-console.log",
        timeout_s=4.0,
    )
    wrapper_hash = bundle["runtime_files_sha256"]["scene_workspace_runtime.xml"]
    actual_hash = runtime.docker_exec(
        "reverify_reset_only_scene_wrapper_hash",
        "sha256sum " + expected_scene,
        timeout_s=3.0,
    ).stdout.split()[0]
    if actual_hash != wrapper_hash:
        raise ResetOnlyError("actual runtime scene wrapper differs from the statically compiled wrapper")
    return {
        "status": "PASS_RUNTIME_CONFIG_AND_NATIVE_LOOKUPS",
        "process_check_stdout": check.stdout,
        "runtime_scene_wrapper_path": expected_scene,
        "runtime_scene_wrapper_sha256_expected": wrapper_hash,
        "runtime_scene_wrapper_sha256_observed": actual_hash,
        "runtime_config_path": expected_config,
        "runtime_scene_identity_basis": [
            "simulator argv and inherited override name the verified generated wrapper",
            "wrapper asset model entries resolve the pinned X2, RAICOM room, and mug assets",
            "the simulator reported module ready",
            "native pelvis initialization is positively logged and every configured active-joint lookup error is absent",
            "a verified process/window screenshot is captured when the optional X11 capture succeeds",
        ],
        "configured_native_joint_names_checked_in_startup_log": required_joints,
        "configured_native_joint_name_count": len(required_joints),
        "raycaster_lidar_warning_is_not_a_gate": True,
    }


class ResetOnlyRun:
    def __init__(self, args: argparse.Namespace, evidence: Path, container: str):
        self.args = args
        self.evidence = evidence
        self.container = container
        self.runtime: BoundedRun | None = None
        self.reset_dispatch_started = False
        self.reset_dispatch_count = 0
        self.container_watchdog: threading.Timer | None = None
        self.container_watchdog_fired = threading.Event()
        self.container_watchdog_result: dict[str, Any] = {
            "armed": False,
            "fired": False,
            "limit_s": ACTIVE_CONTAINER_WATCHDOG_S,
            "stop_attempt": "NOT_RUN",
        }
        self.result: dict[str, Any] = {
            "schema": RESET_ONLY_SCHEMA,
            "status": "BLOCKED",
            "attempt_id": args.attempt_id,
            "classification": "RESET_ONLY_NOT_COMPLETED",
            "runtime_limit_s": MAX_RUNTIME_S,
            "motion_command_count": 0,
            "locomotion_source_registration_count": 0,
            "mc_action_request_count": 0,
            "waypoint_execution_count": 0,
            "native_reset_dispatch_count": 0,
            "scene_identity": None,
            "telemetry": None,
            "cleanup": None,
            "failure": None,
        }

    def _arm_container_stop_watchdog(self) -> None:
        assert self.runtime is not None
        def stop_only() -> None:
            self.container_watchdog_fired.set()
            self.container_watchdog_result["fired"] = True
            self.container_watchdog_result["fired_monotonic_ns"] = time.monotonic_ns()
            try:
                stopped = subprocess.run(
                    ["docker", "stop", "--time", "1", self.runtime.container],
                    text=True, capture_output=True, timeout=2.0,
                )
                self.container_watchdog_result.update({
                    "stop_attempt": "RETURNED",
                    "returncode": stopped.returncode,
                    "stdout": stopped.stdout,
                    "stderr": stopped.stderr,
                })
            except Exception as exc:
                self.container_watchdog_result.update({
                    "stop_attempt": "EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        self.container_watchdog = threading.Timer(ACTIVE_CONTAINER_WATCHDOG_S, stop_only)
        self.container_watchdog.daemon = True
        self.container_watchdog.start()
        self.container_watchdog_result["armed"] = True

    def _wait_for_file(
        self,
        path: Path,
        *,
        timeout_s: float,
        label: str,
        fail_if: Path | None = None,
    ) -> dict[str, Any]:
        assert self.runtime is not None
        deadline = min(time.monotonic() + timeout_s, self.runtime.deadline)
        while time.monotonic() < deadline:
            if fail_if is not None and fail_if.is_file() and not fail_if.is_symlink():
                try:
                    error = json.loads(fail_if.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    error = {"status": "ERROR_RECORD_INVALID"}
                raise ResetOnlyError(f"{label} failed: {error}")
            if path.is_file() and not path.is_symlink():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.025)
                    continue
                return payload
            if self.container_watchdog_fired.is_set():
                raise ResetOnlyError(
                    f"{ACTIVE_CONTAINER_WATCHDOG_S:g}-second isolated-container watchdog stopped the runtime"
                )
            time.sleep(0.05)
        raise ResetOnlyError(f"{label} did not complete within its bounded deadline")

    def _capture_optional_screenshot(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        assert self.runtime is not None
        output = self.evidence / "mujoco_raicom_before_reset.png"
        command = [
            sys.executable, str(X11_HELPER), "capture",
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id), "--output", str(output),
        ]
        try:
            result = self.runtime.run("optional_pre_reset_scene_screenshot", command, timeout_s=2.0, check=False)
        except Exception as exc:
            return {"status": "NOT_CAPTURED", "reason": f"{type(exc).__name__}: {exc}"}
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"status": "NOT_CAPTURED", "returncode": result.returncode, "stderr": result.stderr[-500:]}
        return {"status": value.get("status", "NOT_CAPTURED"), "image": value.get("image"), "path": str(output) if output.is_file() else None}

    def dispatch_reset_once(self, pid: int, title: str, window_id: int) -> None:
        """One-way reset capability: a failed dispatch is never retried."""
        if self.runtime is None:
            raise ResetOnlyError("runtime was not prepared before Reset dispatch")
        if self.reset_dispatch_started or self.reset_dispatch_count != 0:
            raise ResetOnlyError("Reset-only policy refuses a second native Reset dispatch")
        self.reset_dispatch_started = True
        self.reset_dispatch_count = 1
        self.runtime.dispatch_native_reset_button(pid, title, window_id)

    def run_once(self, preflight: Mapping[str, Any]) -> dict[str, Any]:
        audit = audit_reset_only_source(Path(__file__), RESET_OBSERVER)
        self.result["static_capability_audit"] = audit
        self.result["environment_preflight"] = dict(preflight)
        try:
            prepared = _static_preflight(self.args, self.evidence)
            source = prepared["source"]
            static = prepared["static_scene"]
            bundle = prepared["bundle"]
            overlay_path = prepared["overlay_path"]
            overlay_manifest = prepared["overlay_manifest"]
            scene_dir = self.evidence / "local01_scene"
            runtime_args = build_reset_only_runtime_args(self.evidence, self.container)
            self.runtime = BoundedRun(runtime_args)
            self.runtime.source = None
            self.runtime.local01_scene_source = source
            self.runtime.local01_scene_bundle = bundle
            self.runtime.local01_scene_readonly_mounts = prepared["scene_mounts"]
            self.runtime.foundationwalk_config_overlay = overlay_path
            self.runtime.foundationwalk_config_overlay_manifest = overlay_manifest
            self.runtime.start_container()
            self._arm_container_stop_watchdog()
            self.runtime.start_sim_first()
            velocity_graph = self.runtime.docker_exec(
                "inspect_velocity_endpoint_without_publishing",
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
                "ros2 topic info --verbose /aima/mc/locomotion/velocity",
                timeout_s=3.0,
                check=False,
            )
            velocity_publisher_counts = [
                int(value) for value in re.findall(r"Publisher count:\s*(\d+)", velocity_graph.stdout, flags=re.IGNORECASE)
            ]
            unknown_velocity_topic = "Unknown topic" in velocity_graph.stderr + velocity_graph.stdout
            if velocity_graph.returncode == 0 and velocity_publisher_counts and any(velocity_publisher_counts):
                raise ResetOnlyError("a velocity publisher was already visible in the isolated ROS graph")
            if velocity_graph.returncode == 0 and not velocity_publisher_counts and not unknown_velocity_topic:
                raise ResetOnlyError("velocity publisher count was not reported by ROS graph introspection")
            if velocity_graph.returncode != 0 and not unknown_velocity_topic:
                raise ResetOnlyError("could not safely inspect the velocity topic graph before Reset")
            self.result["velocity_topic_graph_pre_reset"] = {
                "command": "ros2 topic info --verbose /aima/mc/locomotion/velocity",
                "returncode": velocity_graph.returncode,
                "stdout": velocity_graph.stdout,
                "stderr": velocity_graph.stderr,
                "publisher_counts": velocity_publisher_counts,
                "no_command_published": True,
                "unknown_topic_means_no_visible_endpoint_not_a_publisher": unknown_velocity_topic,
            }
            scene_identity = _verify_loaded_scene(self.runtime, bundle)
            reference_hashes = {
                name: _sha256(Path(path))
                for name, path in source["reference_files"].items()
            }
            self.result["scene_identity"] = {
                **scene_identity,
                "local01_source_commit": source["source_commit"],
                "local01_source_hashes": source["source_hashes"],
                "reference_asset_hashes": reference_hashes,
                "scene_bundle": bundle,
                "scene_and_reset_frame_candidates": prepared["expected_poses"],
                "timestep_native_baseline_s": static["native_timestep_s"],
                "timestep_raicom_scene_s": static["combined_scene_timestep_s"],
                "foundationwalk_overlay": overlay_manifest,
                "imu_1_config_gap": static["native_runtime_name_contract"]["preexisting_configured_sensors_absent_from_native_model"],
            }

            observer_log = self.evidence / "reset_only_observer_console.log"
            self.runtime.docker_exec(
                "start_reset_only_readonly_observer",
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
                f"timeout {OBSERVER_MAX_RUNTIME_S}s python3 /workspace/evidence/{RESET_OBSERVER.name} "
                "--output-dir /workspace/evidence "
                "--reset-event-file /workspace/evidence/reset_dispatch_event.json "
                f"> /workspace/evidence/{observer_log.name} 2>&1",
                timeout_s=3.0,
                detached=True,
            )
            ready = self._wait_for_file(
                self.evidence / "reset_watch_ready.json",
                timeout_s=5.5,
                label="read-only ROS telemetry readiness",
                fail_if=self.evidence / "reset_only_observer_startup_error.json",
            )
            if ready.get("status") != "READY" or ready.get("publishers_created") != 0 or ready.get("service_clients_created") != 0:
                raise ResetOnlyError("read-only telemetry observer failed its pre-Reset setup gate")
            ages = ready.get("state_age_s", {})
            if not required_state_ages_pass(ages):
                raise ResetOnlyError("pre-Reset required state topics were not all fresh")
            self.result["pre_reset_telemetry_ready"] = ready
            pid, title, window_id = self.runtime.inspect_window()
            image = self._capture_optional_screenshot(pid, title, window_id)
            self.result["pre_reset_scene_screenshot"] = image
            pre_snapshot = self.evidence / "reset_only_pre_reset_snapshot.json"
            snapshot = self._wait_for_file(pre_snapshot, timeout_s=1.0, label="pre-Reset telemetry snapshot")
            self.result["pre_reset_snapshot"] = snapshot
            latest_snapshot = self._wait_for_file(
                self.evidence / "reset_only_latest_snapshot.json",
                timeout_s=1.0,
                label="fresh pre-Reset telemetry snapshot",
            )
            current_ages = latest_snapshot.get("snapshot", {}).get("state_age_s", {})
            if not required_state_ages_pass(current_ages):
                raise ResetOnlyError("telemetry became stale before Reset; no native Reset was dispatched")
            self.result["pre_reset_latest_snapshot"] = latest_snapshot
            dispatch_pre_snapshot_ns = int(
                latest_snapshot.get("snapshot", {}).get("receipt_monotonic_ns", 0)
            )
            if dispatch_pre_snapshot_ns <= 0 or dispatch_pre_snapshot_ns >= time.monotonic_ns():
                raise ResetOnlyError("pre-Reset telemetry snapshot timestamp is invalid")
            _write_json(self.evidence / "reset_only_dispatch_pre_snapshot.json", {
                "schema": "robotsim.issue80.raicom-reset-only-dispatch-pre-snapshot.v1",
                "snapshot_monotonic_ns": dispatch_pre_snapshot_ns,
                "snapshot": latest_snapshot.get("snapshot"),
                "captured_before_native_reset_dispatch": True,
            })
            # The only reset-dispatch call site in this file. No fallback or retry follows it.
            self.dispatch_reset_once(pid, title, window_id)
            self.result["native_reset_dispatch_count"] = self.reset_dispatch_count
            event = self.evidence / "reset_dispatch_event.json"
            self.result["reset_dispatch"] = self._wait_for_file(
                event, timeout_s=1.0, label="verified native Reset event record"
            )
            summary = self._wait_for_file(
                self.evidence / "reset_only_observer_summary.json",
                timeout_s=OBSERVER_MAX_RUNTIME_S + 0.5,
                label="bounded post-Reset telemetry observation",
                fail_if=self.evidence / "reset_only_observer_startup_error.json",
            )
            self.result["telemetry"] = summary
            self.result["odom_frame_assessment"] = classify_odom_frame(
                summary.get("post_reset_last_odom", {}).get("pose")
                if isinstance(summary.get("post_reset_last_odom"), Mapping) else None,
                _scene_expected_poses(source),
            )
            self.result["reset_effect"] = {
                "evidence": summary.get("reset_effect"),
                "source_stamp_transition": summary.get("odom_source_stamp_across_reset"),
                "reset_event_confirmed_as_one_native_button_click": (
                    summary.get("raw_reset_event", {}).get("event", {}).get("status") == "RESET_BUTTON_CLICKED"
                ),
                "state_change_claim": summary.get("reset_effect", {}).get("status"),
                "no_action_or_locomotion_request": True,
            }
            complete = (
                summary.get("observer_status") == "CAPTURE_COMPLETE"
                and summary.get("failure") is None
                and summary.get("post_reset_upright_status") == "PASS_UPRIGHT_SAMPLES_OBSERVED"
                and summary.get("raw_reset_event", {}).get("event", {}).get("status") == "RESET_BUTTON_CLICKED"
                and summary.get("reset_effect", {}).get("status") == "RESET_EFFECT_OBSERVED"
            )
            self.result["status"] = "PASS" if complete else "BLOCKED"
            self.result["classification"] = (
                "RESET_ONLY_TELEMETRY_PASS_NO_STANDING_OR_LOCOMOTION_CLAIM"
                if complete else (
                    "RESET_EFFECT_UNCONFIRMED"
                    if summary.get("reset_effect", {}).get("status") == "RESET_EFFECT_UNCONFIRMED"
                    else "POST_RESET_CAPTURE_BLOCKED"
                )
            )
            if not complete:
                self.result["failure"] = summary.get("failure") or summary.get("observer_status")
        except Exception as exc:
            self.result["status"] = "BLOCKED"
            self.result["classification"] = self._failure_classification()
            self.result["failure"] = f"{type(exc).__name__}: {exc}"
        finally:
            if self.container_watchdog is not None:
                self.container_watchdog.cancel()
                if not self.container_watchdog_fired.is_set():
                    self.container_watchdog_result["cancelled_before_fire"] = True
            if self.runtime is not None:
                self.runtime.cleanup()
                self.runtime.apply_container_teardown_gate(self.result)
                self.result["cleanup"] = self.runtime.cleanup_evidence
                if self.runtime.cleanup_evidence.get("attempted") and not self.runtime.cleanup_evidence.get("container_absent_after_cleanup"):
                    self.result["status"] = "BLOCKED"
                    self.result["failure"] = self.result.get("failure") or "container absence was not verified after cleanup"
        self.result["container_runtime_s"] = (
            time.monotonic() - self.runtime.started if self.runtime is not None else 0.0
        )
        self.result["container_watchdog"] = dict(self.container_watchdog_result)
        self.result["runtime_non_motion_invariants"] = {
            "set_mc_action_requests": 0,
            "set_mc_input_source_requests": 0,
            "locomotion_velocity_publish_calls": 0,
            "waypoint_route_execution_calls": 0,
            "native_reset_dispatch_calls": self.reset_dispatch_count,
            "runtime_wall_limit_s": MAX_RUNTIME_S,
            "active_container_watchdog_s": ACTIVE_CONTAINER_WATCHDOG_S,
        }
        return self.result

    def _failure_classification(self) -> str:
        if self.reset_dispatch_count == 0:
            return "PRE_RESET_SETUP_BLOCKED"
        event_path = self.evidence / "reset_dispatch_event.json"
        if not event_path.is_file():
            return "RESET_NOT_DISPATCHED"
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "RESET_NOT_DISPATCHED"
        if payload.get("event", {}).get("status") != "RESET_BUTTON_CLICKED":
            return "RESET_NOT_DISPATCHED"
        return "POST_RESET_CAPTURE_BLOCKED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--local01-scene-world-poses", type=Path, default=LOCAL01_POSES_DEFAULT)
    parser.add_argument("--evidence-parent", type=Path, default=EVIDENCE_PARENT_DEFAULT)
    args = parser.parse_args()
    try:
        args.attempt_id = parse_attempt_id(args.attempt_id)
        validate_policy(max_runtime_s=MAX_RUNTIME_S, reset_dispatches=1)
    except ValueError as exc:
        parser.error(str(exc))
    if os.environ.get("DISPLAY") != ":0":
        parser.error("run with DISPLAY=:0")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from path_governance import create_unique_output_directory, write_artifact_manifest

    output_parent = args.evidence_parent.resolve(strict=True)
    evidence = create_unique_output_directory(
        output_parent,
        prefix=f"raicom-reset-only-{args.attempt_id[:8]}",
    ).native_path
    container = f"robotsim-issue80-resetonly-{args.attempt_id}"
    result_path = evidence / "reset_only_result.json"
    runner_path = Path(__file__).resolve()
    observer_path = RESET_OBSERVER.resolve(strict=True)
    run: ResetOnlyRun | None = None
    try:
        static_capability = audit_reset_only_source(runner_path, observer_path)
        host = _host_preflight(args, container)
        _write_json(evidence / "reset_only_preflight.json", {
            "schema": "robotsim.issue80.raicom-reset-only-preflight.v1",
            "attempt_id": args.attempt_id,
            "status": "PASS_OFFLINE_AND_HOST_PREFLIGHT",
            "runner_sha256": _sha256(runner_path),
            "observer_sha256": _sha256(observer_path),
            "static_capability_audit": static_capability,
            "host_environment": host,
            "runtime_starts_after_preflight": True,
            "container": container,
        })
        run = ResetOnlyRun(args, evidence, container)
        result = run.run_once(host)
    except Exception as exc:
        result = {
            "schema": RESET_ONLY_SCHEMA,
            "status": "BLOCKED",
            "attempt_id": args.attempt_id,
            "classification": run._failure_classification() if run is not None else "PRE_RESET_SETUP_BLOCKED",
            "runtime_limit_s": MAX_RUNTIME_S,
            "native_reset_dispatch_count": run.reset_dispatch_count if run else 0,
            "motion_command_count": 0,
            "locomotion_source_registration_count": 0,
            "mc_action_request_count": 0,
            "waypoint_execution_count": 0,
            "failure": f"{type(exc).__name__}: {exc}",
            "cleanup": run.runtime.cleanup_evidence if run and run.runtime else {"attempted": False},
        }
        if run and run.runtime:
            run.runtime.cleanup()
            result["cleanup"] = run.runtime.cleanup_evidence
    _write_json(result_path, result)
    write_artifact_manifest(evidence)
    print(json.dumps({
        "status": result.get("status"),
        "classification": result.get("classification"),
        "attempt_id": args.attempt_id,
        "evidence_dir": str(evidence),
        "result": str(result_path),
        "failure": result.get("failure"),
    }, indent=2))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
