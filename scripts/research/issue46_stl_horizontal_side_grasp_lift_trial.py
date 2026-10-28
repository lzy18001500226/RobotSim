#!/usr/bin/env python3
"""One bounded physical lift trial from the accepted horizontal STL side grasp.

This runner has a fail-closed preflight and changes only actuator controls once
physics starts.  It neither rewrites object/hand qpos nor changes the frozen
model, assets, collision exclusions, or gripper coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import imageio_ffmpeg
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
OFFICIAL_URDF = EVIDENCE_REPO / "reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
DT = 0.001
MAX_STEPS = 20_000
MAX_SECONDS = 20.0
ROOT_TRACKING_GATE_RAD = 0.020
ARM_TRACKING_GATE_RAD = 0.080
EQUALITY_GATE_RAD = math.radians(1.0)
POSITION_MARGIN_RAD = 1e-5
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
BASE_STAGE_DURATIONS = (("INITIAL_SETTLE", 0.50), ("PRELOAD", 0.50), ("LIFT_50MM", 3.00), ("HOLD_0_5S", 0.50))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_at(time_s: float, stages: tuple[tuple[str, float], ...]) -> tuple[str, float, float]:
    elapsed = 0.0
    for name, duration in stages:
        if time_s < elapsed + duration:
            return name, (time_s - elapsed) / duration, duration
        elapsed += duration
    return "COMPLETE", 1.0, 0.0


def smoothstep(value: float) -> tuple[float, float]:
    value = min(1.0, max(0.0, value))
    return 3.0 * value**2 - 2.0 * value**3, 6.0 * value - 6.0 * value**2


def interpolate(start: np.ndarray, end: np.ndarray, phase: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    blend, derivative = smoothstep(phase)
    delta = end - start
    return start + blend * delta, derivative * delta / duration


def interpolate_waypoints(waypoints: list[tuple[float, np.ndarray]], phase: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly interpolate a verified contact-maintaining joint-space path."""
    if len(waypoints) < 2:
        raise RuntimeError("a lift path needs at least two waypoints")
    target_lift = min(1.0, max(0.0, phase)) * waypoints[-1][0]
    for index in range(1, len(waypoints)):
        lower_lift, lower_q = waypoints[index - 1]
        upper_lift, upper_q = waypoints[index]
        if target_lift <= upper_lift + 1e-12:
            fraction = (target_lift - lower_lift) / max(upper_lift - lower_lift, 1e-12)
            segment_duration = duration * (upper_lift - lower_lift) / waypoints[-1][0]
            return interpolate(lower_q, upper_q, fraction, segment_duration)
    return waypoints[-1][1].copy(), np.zeros_like(waypoints[-1][1])


def robot_table_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "raicom_table")
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if table not in (first, second):
            continue
        other = second if first == table else first
        if other == cylinder:
            continue
        records.append(
            {
                "body_names": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or f"body#{first}",
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or f"body#{second}",
                ),
                "distance_m": float(contact.dist),
            }
        )
    return records


def set_pose(static, force, model, data, state, names: tuple[str, ...], q: np.ndarray, command: float) -> None:
    motor_override = {name: float(value) for name, value in zip(names[:3], q[:3])}
    static.apply_prior_state(model, data, state, q[3:], motor_override)
    force.apply_positions(model, data, force.fit_positions(command))


def selected_candidate(
    scan: dict[str, object], *, height_offset_mm: float, approach_offset_mm: float, pinch_offset_mm: float,
    azimuth_deg: float, tool_pitch_deg: float, command_percent: int,
) -> dict[str, object]:
    """Select one exact, already static-screened contact state."""
    matches = [
        row
        for row in scan["valid_dual_distal_pad_candidates"]
        if row["height_offset_mm"] == height_offset_mm
        and row.get("approach_offset_mm", 0.0) == approach_offset_mm
        and row.get("pinch_offset_mm", 0.0) == pinch_offset_mm
        and row.get("azimuth_deg", 0.0) == azimuth_deg
        and row.get("tool_pitch_deg", 0.0) == tool_pitch_deg
        and row["command_percent"] == command_percent
    ]
    if len(matches) != 1:
        raise RuntimeError("requested static-screened contact candidate is not unique")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--contact-scan", required=True, type=Path)
    parser.add_argument("--height-offset-mm", type=float, default=100.0)
    parser.add_argument("--approach-offset-mm", type=float, default=0.0)
    parser.add_argument("--pinch-offset-mm", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--tool-pitch-deg", type=float, default=0.0)
    parser.add_argument("--command-percent", type=int, default=38)
    parser.add_argument("--pad-material-overlay", type=Path)
    parser.add_argument("--fixed-hold-preload-fraction", type=float, default=0.0)
    parser.add_argument("--close-command-percent", type=float)
    parser.add_argument("--close-duration-s", type=float, default=2.0)
    parser.add_argument("--force-close-torque-nm", type=float, default=0.0)
    parser.add_argument("--force-close-guard-command-percent", type=float)
    parser.add_argument("--force-close-duration-s", type=float, default=0.50)
    parser.add_argument("--contact-maintaining-path", type=Path)
    parser.add_argument("--contact-path-approach-bias-mm", type=float)
    parser.add_argument("--arm-gain-scale", type=float, default=1.0)
    parser.add_argument("--execute-dynamics", action="store_true")
    args = parser.parse_args()
    if not args.freeze_record.is_file() or not args.contact_scan.is_file() or not OFFICIAL_URDF.is_file():
        raise SystemExit("freeze record, contact scan, or pinned official URDF is missing")
    if not 0.0 <= args.fixed_hold_preload_fraction <= 1.0:
        raise SystemExit("fixed hold preload fraction must lie in [0, 1]")
    if args.close_command_percent is not None and not 0.0 <= args.close_command_percent <= 100.0:
        raise SystemExit("close command percent must lie in 0..100")
    if not 0.1 <= args.close_duration_s <= 10.0:
        raise SystemExit("close duration must lie in 0.1..10 seconds")
    if not 0.0 <= args.force_close_torque_nm <= 0.5 or not 0.1 <= args.force_close_duration_s <= 5.0:
        raise SystemExit("force-close torque or duration is outside the safe experimental envelope")
    if (args.force_close_torque_nm > 0.0) != (args.force_close_guard_command_percent is not None):
        raise SystemExit("force-close torque requires exactly one guard command")
    if args.force_close_guard_command_percent is not None and not 0.0 <= args.force_close_guard_command_percent <= 100.0:
        raise SystemExit("force-close guard command must lie in 0..100")
    if (args.contact_maintaining_path is None) != (args.contact_path_approach_bias_mm is None):
        raise SystemExit("contact-maintaining path and its selected approach bias must be supplied together")
    if not 1.0 <= args.arm_gain_scale <= 4.0:
        raise SystemExit("arm gain scale must lie in [1, 4]")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    scan = json.loads(args.contact_scan.read_text(encoding="utf-8"))
    if frozen.get("baseline_id") != "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1" or scan.get("classification") not in {"STATIC_IK_CONTACT_SCREENING", "STATIC_IK_AZIMUTH_CONTACT_SCREENING"}:
        raise SystemExit("inputs do not identify the accepted horizontal STL baseline and static contact scan")
    review_path = Path(frozen["accepted_review"]["path"])
    if not review_path.is_file() or sha256(review_path) != frozen["accepted_review"]["sha256"]:
        raise SystemExit("accepted review changed after freeze")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    scene = Path(review["scene"]["path"])
    if not scene.is_file() or sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted scene changed after freeze")
    experiment_scene = scene
    material_overlay = None
    if args.pad_material_overlay is not None:
        if not args.pad_material_overlay.is_file():
            raise SystemExit("pad-material overlay record is missing")
        material_overlay = json.loads(args.pad_material_overlay.read_text(encoding="utf-8"))
        required_gates = ("FROZEN_STL_MODEL", "SOURCE_SCENE", "NONPAD_GEOMETRY_UNCHANGED", "COLLISION_EXCLUSIONS_UNCHANGED", "PAD_MATERIAL_OVERLAY")
        if (
            material_overlay.get("classification") != "SIMULATION_ONLY_PAD_CONTACT_MATERIAL_OVERLAY"
            or material_overlay.get("model", {}).get("sha256") != frozen["model"]["sha256"]
            or material_overlay.get("source_scene", {}).get("sha256") != review["scene"]["sha256"]
            or any(material_overlay.get("gates", {}).get(gate) != "PASS" for gate in required_gates)
        ):
            raise SystemExit("pad-material overlay does not preserve the accepted frozen baseline")
        experiment_scene = Path(material_overlay["generated_scene"]["path"])
        if not experiment_scene.is_file() or sha256(experiment_scene) != material_overlay["generated_scene"]["sha256"]:
            raise SystemExit("pad-material scene is missing or changed")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"] or loaded.model_sha256 != scan["model"]["sha256"]:
        raise SystemExit("frozen STL model identity differs from input evidence")
    candidate = selected_candidate(
        scan,
        height_offset_mm=args.height_offset_mm,
        approach_offset_mm=args.approach_offset_mm,
        pinch_offset_mm=args.pinch_offset_mm,
        azimuth_deg=args.azimuth_deg,
        tool_pitch_deg=args.tool_pitch_deg,
        command_percent=args.command_percent,
    )
    close_command = float(candidate["command_percent"] if args.close_command_percent is None else args.close_command_percent)
    force_close_guard_command = close_command if args.force_close_guard_command_percent is None else float(args.force_close_guard_command_percent)
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    azimuth = load_module("stl_azimuth", REPO / "scripts/research/issue46_stl_horizontal_side_grasp_azimuth_scan.py")
    ik = load_module("stl_ik", REPO / "scripts/research/issue46_stl_horizontal_side_grasp_ik_scan.py")
    level = load_module("stl_level", REPO / "scripts/research/issue46_stl_level_side_grasp_search.py")
    orientation = load_module("stl_orientation", REPO / "scripts/research/issue46_stl_centered_orientation_search.py")
    runner = load_module("force_runner", REPO / "scripts/research/issue46_stl_force_control_lift.py")
    state = review["state"]
    names = ik.WAIST_JOINTS + static.ARM_JOINTS
    q_contact = np.asarray(
        [candidate["waist_q_rad"][name] for name in ik.WAIST_JOINTS]
        + [candidate["arm_q_rad"][name] for name in static.ARM_JOINTS],
        dtype=float,
    )
    source_model = mujoco.MjModel.from_xml_path(str(experiment_scene))
    ids = {
        key: ik.body_id(source_model, value)
        for key, value in {
            "elbow": "right_elbow_link",
            "wrist": "right_wrist_roll_link",
            "narrow": "R_hand_narrow3_Link",
            "wide": "R_hand_wide3_Link",
            "base": "R_omnipicker_base_link",
            "cylinder": "test_cylinder",
        }.items()
    }
    contact_command = float(candidate["command_percent"])
    reference = mujoco.MjData(source_model)
    set_pose(static, force, source_model, reference, state, names, q_contact, contact_command)
    mujoco.mj_forward(source_model, reference)
    target_center = orientation.jaw_center(source_model, reference) + np.array((0.0, 0.0, 0.050))
    target_rotation = reference.xmat[ids["base"]].reshape(3, 3).copy()
    limits = ik.joint_limits(source_model, names)
    q_lift, lift_metrics = azimuth.solve(
        ik=ik,
        static=static,
        force=force,
        orient=orientation,
        model=source_model,
        state=state,
        names=names,
        limits=limits,
        seed=q_contact,
        command=int(contact_command),
        target_center=target_center,
        target_rotation=target_rotation,
        ids=ids,
    )
    planned_lift_m = 0.050
    contact_path = None
    path_waypoints: list[tuple[float, np.ndarray]] | None = None
    if args.contact_maintaining_path is not None:
        if not args.contact_maintaining_path.is_file():
            raise SystemExit("contact-maintaining path record is missing")
        contact_path = json.loads(args.contact_maintaining_path.read_text(encoding="utf-8"))
        if (
            contact_path.get("classification") != "STATIC_CONTACT_MAINTAINING_IK_PATH"
            or contact_path.get("model", {}).get("sha256") != loaded.model_sha256
            or contact_path.get("freeze_record", {}).get("sha256") != sha256(args.freeze_record)
            or contact_path.get("azimuth_scan", {}).get("sha256") != sha256(args.contact_scan)
        ):
            raise SystemExit("contact-maintaining path does not match the selected frozen experiment")
        branches = [branch for branch in contact_path.get("valid_branches", []) if branch.get("approach_bias_mm") == args.contact_path_approach_bias_mm]
        if len(branches) != 1:
            raise SystemExit("selected contact-maintaining branch is not unique")
        branch = branches[0]
        path_waypoints = []
        for waypoint in branch["waypoints"]:
            q = np.asarray([waypoint["q"][name] for name in names], dtype=float)
            path_waypoints.append((float(waypoint["lift_mm"]) / 1000.0, q))
        path_waypoints.sort(key=lambda item: item[0])
        if abs(path_waypoints[0][0]) > 1e-12 or path_waypoints[-1][0] < 0.050 - 1e-9:
            raise SystemExit("contact-maintaining path does not span at least 50 mm")
        if float(np.max(np.abs(path_waypoints[0][1] - q_contact))) > 1e-3:
            raise SystemExit("contact-maintaining path does not start at the selected contact pose")
        q_lift = path_waypoints[-1][1].copy()
        lift_metrics = branch["waypoints"][-1]["metrics"]
        planned_lift_m = path_waypoints[-1][0]

    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-side-grasp-physical-trial")
    output = location.native_path
    overlay_scene = output / "existing_root_force_overlay.xml"
    overlay = force.build_force_control_scene(experiment_scene, overlay_scene, torque_cap_nm=0.50)
    model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    free_joint = runner.joint_id(model, "test_cylinder_free")
    free_dof = int(model.jnt_dofadr[free_joint])
    root_motor = runner.actuator_id(model, "motor_right_claw_stl_force")
    root_joint = runner.joint_id(model, "right_claw_joint")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    data = mujoco.MjData(model)
    open_command = 45.0
    set_pose(static, force, model, data, state, names, q_contact, open_command)
    endpoint_guard = runner.initialize_endpoint_guard(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_com = data.xipos[cylinder].copy()
    initial_base = data.xpos[base].copy()
    source_limits = runner.source_velocity_limits(OFFICIAL_URDF)
    warnings_before = np.asarray(data.warning.number, dtype=int).copy()
    contact_checks = level.actual_contacts(model, data, cylinder)
    close_check = mujoco.MjData(model)
    set_pose(static, force, model, close_check, state, names, q_contact, force_close_guard_command)
    mujoco.mj_forward(model, close_check)
    close_contacts = level.actual_contacts(model, close_check, cylinder)
    lift_check = mujoco.MjData(model)
    set_pose(static, force, model, lift_check, state, names, q_lift, contact_command)
    mujoco.mj_forward(model, lift_check)
    lift_robot_table = robot_table_contacts(model, lift_check)
    preflight_gates = {
        "FROZEN_MODEL": "PASS",
        "ACCEPTED_SCENE": "PASS",
        "OPEN_PREGRASP_NONPAD_CLEAR": "PASS" if not contact_checks["nonpad_bodies"] else "FAIL",
        "FORCE_CLOSE_GUARD_DUAL_PAD_ONLY": "PASS" if all(close_contacts["counts"][name] > 0 for name in PADS) and not close_contacts["nonpad_bodies"] else "FAIL",
        "LIFT_IK": "PASS" if lift_metrics["jaw_center_error_m"] <= 1e-4 and lift_metrics["base_orientation_error_rad"] <= 1e-3 else "FAIL",
        "LIFT_ROBOT_TABLE_CLEAR": "PASS" if not lift_robot_table else "FAIL",
        "DYNAMICS": "NOT_RUN",
    }
    preflight = {
        "candidate": candidate,
        "q_contact": {name: float(value) for name, value in zip(names, q_contact)},
        "q_lift": {name: float(value) for name, value in zip(names, q_lift)},
        "lift_metrics": lift_metrics,
        "open_pregrasp_contacts": contact_checks,
        "force_close_guard_command_percent": force_close_guard_command,
        "force_close_guard_contacts": close_contacts,
        "lift_robot_table_contacts": lift_robot_table,
        "gates": preflight_gates,
    }
    pg.write_new_text(output / "preflight.json", json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")
    if any(value != "PASS" for key, value in preflight_gates.items() if key != "DYNAMICS") or not args.execute_dynamics:
        preflight_gates["DYNAMICS"] = "BLOCKED_PREFLIGHT" if args.execute_dynamics else "NOT_RUN"
        pg.write_reserved_text(output, output / "preflight.json", json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", overwrite_existing=True)
        pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
        pg.write_artifact_manifest(output)
        print(json.dumps({"status": preflight_gates["DYNAMICS"], "output_wsl": str(output), "gates": preflight_gates, "mj_step_calls": 0}, ensure_ascii=False, indent=2))
        return

    motor_targets = {str(name): float(value) for name, value in state["motor_targets_rad"].items()}
    for name, values in endpoint_guard.items():
        motor_targets[name] = float(values["initialized_rad"])
    left_root = float(state["left_hand_qpos_rad"]["right_claw_joint"])
    reference_model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    reference_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    trace = (output / "trace.jsonl").open("x", encoding="utf-8")
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.43, -0.98, 0.80]
    camera.distance = 1.25
    camera.azimuth = 160.0
    camera.elevation = -8.0
    writer = imageio_ffmpeg.write_frames(
        str(output / "actual_mujoco_physics.mp4"), (640, 480), fps=25, quality=7, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", output_params=["-preset", "ultrafast", "-movflags", "+faststart"]
    )
    writer.send(None)
    preload_duration = args.force_close_duration_s if args.force_close_torque_nm > 0.0 else BASE_STAGE_DURATIONS[1][1]
    lift_duration = 3.0 * planned_lift_m / 0.050
    stage_plan = (("INITIAL_SETTLE", 0.50), ("CLOSE", args.close_duration_s), ("PRELOAD", preload_duration), ("LIFT_50MM", lift_duration), BASE_STAGE_DURATIONS[3])
    stages = {name: "NOT RUN" for name, _ in stage_plan}
    metrics = {"max_com_rise_m": 0.0, "max_base_rise_m": 0.0, "max_root_error_rad": 0.0, "max_equality_error_rad": 0.0, "max_arm_error_rad": 0.0, "max_pad_force_N": {name: 0.0 for name in PADS}, "first_bilateral_step": None, "first_table_clear_step": None}
    failure = None
    current_stage = None
    completed = False
    steps = 0
    try:
        for step in range(min(MAX_STEPS, int(MAX_SECONDS / DT))):
            stage, phase, duration = phase_at(float(data.time), stage_plan)
            if stage == "COMPLETE":
                completed = True
                break
            if stage != current_stage:
                if current_stage is not None and stages[current_stage] == "NOT RUN":
                    stages[current_stage] = "PASS"
                current_stage = stage
            if stage in {"INITIAL_SETTLE", "CLOSE", "PRELOAD"}:
                target = q_contact
                velocity = np.zeros(len(static.ARM_JOINTS))
            else:
                if stage == "LIFT_50MM" and path_waypoints is not None:
                    target, velocity = interpolate_waypoints(path_waypoints, phase, duration)
                else:
                    target, velocity = interpolate(q_contact, q_lift, phase if stage == "LIFT_50MM" else 1.0, duration if stage == "LIFT_50MM" else 1.0)
            command = open_command if stage == "INITIAL_SETTLE" else open_command + (close_command - open_command) * smoothstep(phase)[0] if stage == "CLOSE" else force_close_guard_command
            for name, value in zip(names, target):
                motor_targets[name] = float(value)
            feedforward = runner.reference_arm_feedforward(reference_model, data, motor_targets, velocity[3:], np.zeros(len(static.ARM_JOINTS)))
            contacts = runner.LEGACY.contact_data(model, data)
            root_target = float(force.fit_positions(command)["right_claw_joint"])
            current_root_q = float(data.qpos[model.jnt_qposadr[root_joint]])
            guard_root_target = float(force.fit_positions(force_close_guard_command)["right_claw_joint"])
            force_close_active = (
                stage in {"PRELOAD", "LIFT_50MM", "HOLD_0_5S"}
                and args.force_close_torque_nm > 0.0
                and current_root_q < guard_root_target
            )
            # A torque-controlled close has no position target other than the
            # measured current q.  This preserves the tracking gate's meaning
            # while the independent guard prevents further closure at q_guard.
            tracking_target = current_root_q if force_close_active else root_target
            fixed_preload = args.fixed_hold_preload_fraction if stage in {"PRELOAD", "LIFT_50MM", "HOLD_0_5S"} else 0.0
            saturation, requested, details = runner.set_controls(
                model, data, motor_targets, left_root, tracking_target, 1.0, 8.0, 0.10,
                stage in {"PRELOAD", "LIFT_50MM", "HOLD_0_5S"}, 0.20, fixed_preload, root_target, 0.04, args.arm_gain_scale, feedforward,
            )
            if force_close_active:
                data.ctrl[root_motor] = args.force_close_torque_nm
                requested["motor_right_claw_stl_force"] = args.force_close_torque_nm
                details["motor_right_claw_stl_force"].update({"mode": "GUARDED_FORCE_CLOSE", "guard_target_rad": guard_root_target, "applied_nm": args.force_close_torque_nm})
            root_q = float(data.qpos[model.jnt_qposadr[root_joint]])
            root_error = abs(root_q - tracking_target)
            equality = runner.equality_error(model, data)
            equality_max = max(equality.values())
            arm_error = max(abs(float(data.qpos[model.jnt_qposadr[runner.joint_id(model, name)]]) - float(value)) for name, value in zip(static.ARM_JOINTS, target[3:]))
            com_rise = float(data.xipos[cylinder][2] - initial_com[2])
            base_rise = float(data.xpos[base][2] - initial_base[2])
            metrics["max_com_rise_m"] = max(float(metrics["max_com_rise_m"]), com_rise)
            metrics["max_base_rise_m"] = max(float(metrics["max_base_rise_m"]), base_rise)
            metrics["max_root_error_rad"] = max(float(metrics["max_root_error_rad"]), root_error)
            metrics["max_equality_error_rad"] = max(float(metrics["max_equality_error_rad"]), equality_max)
            metrics["max_arm_error_rad"] = max(float(metrics["max_arm_error_rad"]), arm_error)
            for name, value in contacts["pad_force_norm_N"].items():
                metrics["max_pad_force_N"][name] = max(float(metrics["max_pad_force_N"][name]), float(value))
            if all(float(value) >= 0.03 for value in contacts["pad_force_norm_N"].values()) and metrics["first_bilateral_step"] is None:
                metrics["first_bilateral_step"] = step + 1
            if not contacts["table_contacts"] and metrics["first_table_clear_step"] is None:
                metrics["first_table_clear_step"] = step + 1
            trace.write(json.dumps({"step": step + 1, "time_s": float(data.time), "stage": stage, "command_percent": command, "root_q_rad": root_q, "root_error_rad": root_error, "force_close_active": force_close_active, "equality_error_rad": equality, "arm_error_rad": arm_error, "pad_force_norm_N": contacts["pad_force_norm_N"], "table_force_on_cylinder_world_N": contacts["table_force_on_cylinder_world_N"], "nonpad_cylinder_contacts": contacts["nonpad_cylinder_contacts"], "cylinder_com_world_m": data.xipos[cylinder].tolist(), "cylinder_com_rise_m": com_rise, "gripper_base_rise_m": base_rise, "right_root_control": details["motor_right_claw_stl_force"], "actuator_force": data.actuator_force.tolist()}, ensure_ascii=False) + "\n")
            if step % 40 == 0:
                renderer.update_scene(data, camera=camera)
                writer.send(np.ascontiguousarray(renderer.render(), dtype=np.uint8))
            if saturation:
                failure = {"gate": "ACTUATOR_SATURATION", "step": step + 1, "stage": stage, "details": saturation}
            elif contacts["nonpad_cylinder_contacts"]:
                failure = {"gate": "NONPAD_CYLINDER_CONTACT", "step": step + 1, "stage": stage, "details": contacts["nonpad_cylinder_contacts"]}
            elif robot_table_contacts(model, data):
                failure = {"gate": "ROBOT_TABLE_CONTACT", "step": step + 1, "stage": stage}
            elif root_error > ROOT_TRACKING_GATE_RAD:
                failure = {"gate": "ROOT_TRACKING_ERROR", "step": step + 1, "stage": stage, "error_rad": root_error}
            elif arm_error > ARM_TRACKING_GATE_RAD:
                failure = {"gate": "ARM_TRACKING_ERROR", "step": step + 1, "stage": stage, "error_rad": arm_error}
            elif equality_max > EQUALITY_GATE_RAD:
                failure = {"gate": "FOLLOWER_CONSTRAINT_ERROR", "step": step + 1, "stage": stage, "error_rad": equality_max}
            elif not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                failure = {"gate": "NONFINITE_STATE", "step": step + 1, "stage": stage}
            elif np.any(np.asarray(data.warning.number, dtype=int) > warnings_before):
                failure = {"gate": "MUJOCO_WARNING", "step": step + 1, "stage": stage, "warnings": data.warning.number.tolist()}
            else:
                for joint in range(model.njnt):
                    if not bool(model.jnt_limited[joint]):
                        continue
                    low, high = (float(value) for value in model.jnt_range[joint])
                    value = float(data.qpos[model.jnt_qposadr[joint]])
                    if value < low - POSITION_MARGIN_RAD or value > high + POSITION_MARGIN_RAD:
                        failure = {"gate": "JOINT_LIMIT", "step": step + 1, "stage": stage, "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), "qpos_rad": value, "range_rad": [low, high]}
                        break
                if failure is None:
                    for name, limit in source_limits.items():
                        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        if joint >= 0 and abs(float(data.qvel[model.jnt_dofadr[joint]])) > limit * 1.05 + 1e-9:
                            failure = {"gate": "SOURCE_VELOCITY_LIMIT", "step": step + 1, "stage": stage, "joint": name, "limit_rad_s": limit}
                            break
            if failure is not None:
                break
            mujoco.mj_step(model, data)
            steps += 1
        if failure is None and completed and stages[current_stage] == "NOT RUN":
            stages[current_stage] = "PASS"
    finally:
        trace.close()
        writer.close()
        renderer.close()
    successful_lift = completed and failure is None and metrics["max_com_rise_m"] >= 0.050 and metrics["first_table_clear_step"] is not None
    result = {
        "schema": "robotsim.issue46.stl.horizontal-side-grasp-physical-trial.v1",
        "classification": "EXPERIMENTAL_PHYSICAL_GRASP_PASS" if successful_lift else "EXPERIMENTAL_PHYSICAL_GRASP_FAIL",
        "mj_step_calls": steps,
        "simulated_time_s": float(data.time),
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "freeze_record": {"path": str(args.freeze_record), "sha256": sha256(args.freeze_record)},
        "contact_scan": {"path": str(args.contact_scan), "sha256": sha256(args.contact_scan)},
        "overlay": {**overlay, "path": str(overlay_scene), "sha256": sha256(overlay_scene)},
        "pad_material_overlay": None if material_overlay is None else {"path": str(args.pad_material_overlay), "sha256": sha256(args.pad_material_overlay), "id": material_overlay["overlay_id"], "material": material_overlay["material"]},
        "control_overlay": {"id": "STL-FixedHoldPreload-SIM-v1" if args.fixed_hold_preload_fraction else "NONE", "fixed_hold_preload_fraction": args.fixed_hold_preload_fraction, "fixed_hold_preload_cap_nm": 0.5 * args.fixed_hold_preload_fraction},
        "control_schedule": {"close_command_percent": close_command, "close_duration_s": args.close_duration_s, "force_close_torque_nm": args.force_close_torque_nm, "force_close_guard_command_percent": force_close_guard_command, "force_close_duration_s": args.force_close_duration_s},
        "arm_control_overlay": {"id": "STL-ArmPD-SIM-v1" if args.arm_gain_scale != 1.0 else "NONE", "gain_scale": args.arm_gain_scale},
        "contact_maintaining_path": None if contact_path is None else {"path": str(args.contact_maintaining_path), "sha256": sha256(args.contact_maintaining_path), "approach_bias_mm": args.contact_path_approach_bias_mm, "waypoint_count": len(path_waypoints or [])},
        "planned_gripper_lift_m": planned_lift_m,
        "preflight": preflight,
        "stages": stages,
        "metrics": metrics,
        "first_failure": failure,
        "object_qpos_writes_after_initialization": 0,
        "right_hand_qpos_writes_after_initialization": 0,
        "limitations": ["One bounded experimental trial only; no manufacturer-equivalence or grasp-success claim beyond measured result."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"classification": result["classification"], "steps": steps, "max_com_rise_mm": metrics["max_com_rise_m"] * 1000.0, "first_failure": failure, "output_wsl": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
