#!/usr/bin/env python3
"""One table-driven STL OmniPicker physical cylinder-lift experiment.

The robot and cylinder evolve through real MuJoCo mj_step integration. Only the
human-approved right STL hand is deliberately kinematic: its six recorded joint
coordinates are written from the source-STL table before each step. The cylinder
is never written after initialization.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


MODEL_ID = "X2-OmniPicker-STL"
ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
KINEMATIC_JOINTS = {
    "right_claw_joint",
    "R_hand_wide1_joint",
    "R_hand_narrow3_joint",
    "R_hand_narrow_loop_joint",
    "R_hand_wide3_joint",
    "R_hand_wide_loop_joint",
}
DT = 0.001
MAX_STEPS = 20000
MAX_SECONDS = 20.0
POSITION_MARGIN_RAD = 1e-5
ARM_TRACKING_GATE_RAD = 0.08
BILATERAL_FORCE_GATE_N = 0.03
SETTLE_LINEAR_SPEED_M_S = 0.02
SETTLE_ANGULAR_SPEED_RAD_S = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: float) -> tuple[float, float]:
    value = min(1.0, max(0.0, value))
    return 3.0 * value * value - 2.0 * value * value * value, 6.0 * value - 6.0 * value * value


def interpolate(start: np.ndarray, end: np.ndarray, phase: float) -> np.ndarray:
    blend, _ = smoothstep(phase)
    return start + blend * (end - start)


def interpolate_velocity(start: np.ndarray, end: np.ndarray, phase: float, duration_s: float) -> np.ndarray:
    _, derivative = smoothstep(phase)
    return derivative * (end - start) / duration_s


def interpolate_acceleration(start: np.ndarray, end: np.ndarray, phase: float, duration_s: float) -> np.ndarray:
    bounded = min(1.0, max(0.0, phase))
    return (6.0 - 12.0 * bounded) * (end - start) / (duration_s * duration_s)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise RuntimeError(f"missing joint: {name}")
    return value


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


def static_state(prior: dict[str, object], phase: str) -> dict[str, object]:
    for item in prior["static_states"]:
        if item.get("phase") == phase:
            return item
    raise RuntimeError(f"prior configuration lacks phase {phase}")


def left_name(right_name: str) -> str:
    return "left_claw_joint" if right_name == "right_claw_joint" else right_name.replace("R_", "L_", 1)


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float, velocity: float = 0.0) -> None:
    jid = joint_id(model, name)
    data.qpos[model.jnt_qposadr[jid]] = float(value)
    data.qvel[model.jnt_dofadr[jid]] = float(velocity)


def set_initial_state(model: mujoco.MjModel, data: mujoco.MjData, prior_state: dict[str, object]) -> tuple[dict[str, float], float]:
    targets = {name: float(value) for name, value in prior_state["motor_targets_rad"].items()}
    for name, value in targets.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, name, value)
    left_root = 0.0
    for right, value in prior_state["left_hand_qpos_rad"].items():
        name = left_name(right)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, name, float(value))
        if name == "left_claw_joint":
            left_root = float(value)
    return targets, left_root


def source_velocity_limits(urdf: Path) -> dict[str, float]:
    root = ET.parse(urdf).getroot()
    limits = {}
    for joint in root.findall("joint"):
        node = joint.find("limit")
        if node is None or node.get("velocity") is None:
            continue
        try:
            velocity = float(node.get("velocity", ""))
        except ValueError:
            continue
        # URDF velocity=0 means unspecified, not a literal zero-speed limit.
        if velocity > 0.0:
            limits[joint.get("name", "")] = velocity
    return limits


def motor_gains(name: str, arm_gain_scale: float) -> tuple[float, float]:
    if "head" in name:
        return 20.0, 2.0
    if any(token in name for token in ("shoulder", "elbow", "wrist")):
        return 30.0 * arm_gain_scale, 4.0 * arm_gain_scale
    return 40.0, 5.0


def inverse_static(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjData:
    inverse = mujoco.MjData(model)
    inverse.time = data.time
    inverse.qpos[:] = data.qpos
    inverse.qvel[:] = data.qvel
    inverse.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0.0
    mujoco.mj_inverse(model, inverse)
    return inverse


def reference_inverse_no_contact(
    reference_model: mujoco.MjModel,
    live_data: mujoco.MjData,
    motor_targets: dict[str, float],
    trajectory,
    command_percent: float,
    command_rate_percent_s: float,
    arm_velocity: np.ndarray,
    arm_acceleration: np.ndarray,
) -> dict[int, float]:
    """Compute desired arm motion feedforward in a separate contact-disabled model."""
    data = mujoco.MjData(reference_model)
    data.qpos[:] = live_data.qpos
    data.qvel[:] = 0.0
    for name, target in motor_targets.items():
        jid = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[reference_model.jnt_qposadr[jid]] = float(target)
    apply_stl_kinematic_pose(
        mujoco, reference_model, data, trajectory, command_percent,
        command_rate_percent_s=command_rate_percent_s,
    )
    for name, velocity, acceleration in zip(ARM_JOINTS, arm_velocity, arm_acceleration):
        jid = joint_id(reference_model, name)
        data.qvel[reference_model.jnt_dofadr[jid]] = float(velocity)
    mujoco.mj_forward(reference_model, data)
    data.qacc[:] = 0.0
    for name, acceleration in zip(ARM_JOINTS, arm_acceleration):
        jid = joint_id(reference_model, name)
        data.qacc[reference_model.jnt_dofadr[jid]] = float(acceleration)
    mujoco.mj_inverse(reference_model, data)
    values = {}
    for actuator in range(reference_model.nu):
        name = mujoco.mj_id2name(reference_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or ""
        if not name.startswith("motor_"):
            continue
        joint = int(reference_model.actuator_trnid[actuator, 0])
        dof = int(reference_model.jnt_dofadr[joint])
        gear = float(reference_model.actuator_gear[actuator, 0])
        values[actuator] = float(data.qfrc_inverse[dof] / gear)
    return values


def configure_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motor_targets: dict[str, float],
    left_root_target: float,
    feedforward_mode: str,
    arm_gain_scale: float,
    reference_feedforward: dict[int, float] | None,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, object]]:
    inverse = inverse_static(model, data) if feedforward_mode == "contact_coupled_inverse" else None
    data.ctrl[:] = 0.0
    saturation = []
    requested = {}
    components = {}
    for actuator in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or f"actuator#{actuator}"
        joint = int(model.actuator_trnid[actuator, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        dof = int(model.jnt_dofadr[joint])
        low, high = map(float, model.actuator_ctrlrange[actuator])
        if name.startswith("motor_"):
            gear = float(model.actuator_gear[actuator, 0])
            if abs(gear) < 1e-12:
                raise RuntimeError(f"zero gear for {name}")
            kp, kd = motor_gains(joint_name, arm_gain_scale)
            q = float(data.qpos[model.jnt_qposadr[joint]])
            qvel = float(data.qvel[dof])
            target = motor_targets.get(joint_name, q)
            feedforward = float(
                reference_feedforward[actuator]
                if reference_feedforward is not None
                else inverse.qfrc_inverse[dof] / gear
                if inverse is not None
                else data.qfrc_bias[dof] / gear
            )
            pd = float(kp * (target - q) - kd * qvel)
            value = float(feedforward + pd)
            requested[name] = value
            components[name] = {
                "joint": joint_name,
                "feedforward_ctrl": feedforward,
                "pd_ctrl": pd,
                "requested_ctrl": value,
                "qfrc_bias_Nm": float(data.qfrc_bias[dof]),
                "qfrc_constraint_Nm": float(data.qfrc_constraint[dof]),
                "feedforward_mode": feedforward_mode,
            }
            if value < low - 1e-12 or value > high + 1e-12:
                saturation.append({"actuator": name, "joint": joint_name, "requested": value, "range": [low, high]})
            data.ctrl[actuator] = float(np.clip(value, low, high))
        elif name == "servo_left_claw_joint":
            data.ctrl[actuator] = float(np.clip(left_root_target, low, high))
            requested[name] = left_root_target
    return saturation, requested, components


def contact_data(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if cylinder < 0:
        raise RuntimeError("missing test cylinder")
    records = []
    pad_vectors = {name: np.zeros(3) for name in PADS}
    table_contacts = []
    table_vector = np.zeros(3)
    nonpad = []
    robot_table = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        names = (body_name(model, body1), body_name(model, body2))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        world_force = frame.T @ wrench[:3]
        record = {
            "index": index,
            "body_names": names,
            "geom_names": (geom_name(model, geom1), geom_name(model, geom2)),
            "distance_m": float(contact.dist),
            "force_contact_frame_N_Nm": wrench.tolist(),
            "force_world_frame_N": world_force.tolist(),
        }
        records.append(record)
        if cylinder in (body1, body2):
            other = names[1] if body1 == cylinder else names[0]
            force_on_cylinder = world_force if body1 == cylinder else -world_force
            record["force_on_cylinder_world_N"] = force_on_cylinder.tolist()
            if other in PADS:
                pad_vectors[other] += force_on_cylinder
            elif other == "raicom_table":
                table_contacts.append(record)
                table_vector += force_on_cylinder
            elif other.startswith("R_hand_"):
                nonpad.append(record)
        if "raicom_table" in names and any(name.startswith(("R_hand_", "right_")) for name in names):
            robot_table.append(record)
    return {
        "records": records,
        "pad_forces_on_cylinder_world_N": {name: vector.tolist() for name, vector in pad_vectors.items()},
        "pad_force_norm_N": {name: float(np.linalg.norm(vector)) for name, vector in pad_vectors.items()},
        "table_force_on_cylinder_world_N": table_vector.tolist(),
        "table_contacts": table_contacts,
        "nonpad_cylinder_contacts": nonpad,
        "robot_table_contacts": robot_table,
    }


def phase_at(time_s: float, durations: dict[str, float]) -> tuple[str, float, float]:
    cursor = 0.0
    for name, duration in durations.items():
        if time_s < cursor + duration:
            return name, (time_s - cursor) / duration if duration else 1.0, duration
        cursor += duration
    return "COMPLETE", 1.0, 0.0


def write_video_frame(renderer, camera, data, writer) -> None:
    renderer.update_scene(data, camera=camera)
    writer.send(np.ascontiguousarray(renderer.render(), dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--full-urdf", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute-dynamics", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--close-duration-s", type=float, default=2.0)
    parser.add_argument("--feedforward-mode", choices=("contact_coupled_inverse", "gravity_bias", "reference_inverse_no_contact"), default="contact_coupled_inverse")
    parser.add_argument("--calibration-only", action="store_true", help="run the open-hand approach and measure gripper-base tracking without closing")
    parser.add_argument("--open-lift-calibration", action="store_true", help="run the open-hand 50 mm arm lift to validate physical arm tracking before contact")
    parser.add_argument("--arm-gain-scale", type=float, default=1.0, help="simulation-only multiplier for existing shoulder/elbow/wrist PD gains")
    parser.add_argument("--preload-percent", type=float, default=42.0, help="safe post-contact STL command used for physical pad preload")
    parser.add_argument("--adaptive-pad-centering", action="store_true", help="use measured single-pad contact to apply the source-geometry lateral arm correction before preload")
    args = parser.parse_args()
    if not np.isfinite(args.close_duration_s) or args.close_duration_s <= 0.0:
        raise SystemExit("--close-duration-s must be positive and finite")
    if args.calibration_only and args.open_lift_calibration:
        raise SystemExit("choose at most one calibration mode")
    if not np.isfinite(args.arm_gain_scale) or args.arm_gain_scale <= 0.0:
        raise SystemExit("--arm-gain-scale must be positive and finite")
    if not np.isfinite(args.preload_percent) or not 0.0 <= args.preload_percent <= 100.0:
        raise SystemExit("--preload-percent must be within 0..100")
    if not args.execute_dynamics:
        raise SystemExit("refusing to run physics without --execute-dynamics")
    static = json.loads(args.static_result.read_text(encoding="utf-8"))
    gates = static.get("gates", {})
    if gates.get("BILATERAL_PAD_ONLY_STATIC_CANDIDATE") != "PASS" or gates.get("LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY") != "PASS":
        raise SystemExit("static STL grasp/lift preflight is not approved for dynamics")
    scene = Path(static["generated_scene"]["path"])
    if not scene.is_file() or sha256(scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("static preflight scene hash mismatch")
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if static["model"]["sha256"] != loaded.model_sha256:
        raise SystemExit("selected STL model hash differs from static preflight")
    trajectory = load_registered_trajectory(loaded.package_path)
    location = (
        pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
        if args.output_dir is not None
        else pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-cylinder-lift")
    )
    output = location.native_path
    if any(path.name != ".robotsim-output-reservation.json" for path in output.iterdir()):
        raise SystemExit("output directory is not fresh")
    model = mujoco.MjModel.from_xml_path(str(scene))
    reference_model = None
    if args.feedforward_mode == "reference_inverse_no_contact":
        reference_model = mujoco.MjModel.from_xml_path(str(scene))
        reference_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if abs(float(model.opt.timestep) - DT) > 1e-12 or mujoco.__version__ != "3.3.6":
        raise SystemExit("unexpected MuJoCo model/runtime version")
    data = mujoco.MjData(model)
    pregrasp = static_state(prior, "HORIZONTAL_PREGRASP")
    approach = static_state(prior, "HORIZONTAL_APPROACH")
    motor_targets, left_root = set_initial_state(model, data, pregrasp)
    aligned_contact = static.get("aligned_contact_static")
    if not aligned_contact:
        raise SystemExit("static preflight lacks the required balanced pad-alignment candidate")
    selected_percent = float(aligned_contact["highest_safe_bilateral_percent"])
    if args.preload_percent > selected_percent:
        raise SystemExit("--preload-percent must close no less than the first bilateral-contact command")
    preload_static = next(
        (item for item in aligned_contact["closure_contact_scan"] if float(item["command_percent"]) == float(args.preload_percent)),
        None,
    )
    if preload_static is None or set(preload_static["pad_bodies"]) != PADS or preload_static["nonpad_right_contact_count"]:
        raise SystemExit("preload command is not a static dual-pad-only STL configuration")
    contact_target = static.get("controller_contact_target_static") or aligned_contact
    alignment_scan = static.get("lateral_alignment_scan", [])
    lift_target = static.get("controller_lift_target_static") or static["transport_end_static"]
    lift_arm = np.asarray([lift_target["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
    pregrasp_arm = np.asarray([pregrasp["motor_targets_rad"][name] for name in ARM_JOINTS], dtype=float)
    approach_arm = np.asarray([contact_target["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
    physical_alignment_reference = np.asarray(aligned_contact["gripper_base_world_m"], dtype=float)
    if args.adaptive_pad_centering:
        samples = {float(item["lateral_offset_mm"]): np.asarray([item["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float) for item in alignment_scan if "arm_q_rad" in item}
        if len(samples) < 2:
            raise SystemExit("static preflight lacks lateral arm samples for adaptive pad centering")
        static_offsets = sorted(samples)
        static_baseline_offset = float(aligned_contact["lateral_offset_mm"])
        static_baseline_arm = np.asarray([aligned_contact["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
    else:
        samples = {}
        static_offsets = []
        static_baseline_offset = 0.0
        static_baseline_arm = np.zeros(len(ARM_JOINTS))
    physical_lift_reference = np.asarray(static["transport_end_static"]["gripper_base_world_m"], dtype=float)
    source_limits = source_velocity_limits(args.full_urdf)
    cylinder_joint = joint_id(model, "test_cylinder_free")
    cylinder_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    gripper_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    cylinder_qadr = int(model.jnt_qposadr[cylinder_joint])
    cylinder_dadr = int(model.jnt_dofadr[cylinder_joint])
    if abs(float(model.body_mass[cylinder_body]) - 0.20) > 1e-12:
        raise SystemExit("cylinder mass differs from the frozen 0.20 kg scene")
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, 100.0)
    mujoco.mj_forward(model, data)
    initial_object_qpos = data.qpos[cylinder_qadr:cylinder_qadr + 7].copy()
    initial_com = data.xipos[cylinder_body].copy()
    durations = (
        {
            "INITIAL_SETTLE": 0.75,
            "HORIZONTAL_PREGRASP": 0.50,
            "APPROACH": 2.00,
            "CALIBRATION_HOLD": 0.50,
        }
        if args.calibration_only
        else {
            "INITIAL_SETTLE": 0.75,
            "HORIZONTAL_PREGRASP": 0.50,
            "APPROACH": 2.00,
            "OPEN_LIFT_CALIBRATION": 3.00,
            "CALIBRATION_HOLD": 0.50,
        }
        if args.open_lift_calibration
        else {
            "INITIAL_SETTLE": 0.75,
            "HORIZONTAL_PREGRASP": 0.50,
            "APPROACH": 2.00,
            "CLOSE": float(args.close_duration_s),
            "BILATERAL_PAD_CONTACT": 0.75,
            "PRELOAD": 1.00,
            "PRELOAD_HOLD": 0.75,
            "LIFT_50MM": 3.00,
            "HOLD_0_5S": 0.50,
        }
    )
    expected_duration = sum(durations.values())
    if expected_duration > MAX_SECONDS:
        raise SystemExit("planned sequence exceeds trial budget")
    stages = {name: "NOT RUN" for name in durations}
    metrics = {
        "max_cylinder_com_rise_m": 0.0,
        "max_arm_tracking_error_rad": 0.0,
        "max_table_contact_count": 0,
        "max_pad_force_norm_N": {name: 0.0 for name in PADS},
        "max_root_command_error_rad": 0.0,
        "max_gripper_base_position_error_m": 0.0,
        "max_physical_alignment_error_m": 0.0,
        "max_actuator_force": {},
        "max_requested_ctrl": {},
        "object_qpos_writes_after_initialization": 0,
        "kinematic_hand_qpos_writes": 0,
        "first_bilateral_step": None,
        "first_table_clear_step": None,
    }
    warnings_before = np.asarray(data.warning.number, dtype=int).copy()
    failure = None
    physics_steps = 0
    bilateral_streak = 0
    settle_streak = 0
    adaptive_offset_mm = static_baseline_offset
    last_adaptive_update_step = -1000
    adaptive_updates = []
    current_stage = None
    trace_path = output / "trace.jsonl"
    trace = trace_path.open("x", encoding="utf-8")
    video_writer = None
    renderer = None
    camera = None
    if args.record_video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [1.43, -0.92, 0.80]
        camera.distance = 1.45
        camera.azimuth = 165.0
        camera.elevation = -18.0
        video_writer = imageio_ffmpeg.write_frames(
            str(output / "actual_mujoco_physics.mp4"),
            (640, 480), fps=25, quality=7, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            output_params=["-preset", "ultrafast", "-movflags", "+faststart"],
        )
        video_writer.send(None)
    def static_arm_at_offset(offset_mm: float) -> np.ndarray:
        if offset_mm <= static_offsets[0]:
            return samples[static_offsets[0]]
        if offset_mm >= static_offsets[-1]:
            return samples[static_offsets[-1]]
        for low, high in zip(static_offsets, static_offsets[1:]):
            if low <= offset_mm <= high:
                fraction = (offset_mm - low) / (high - low)
                return samples[low] + fraction * (samples[high] - samples[low])
        raise RuntimeError("invalid adaptive lateral offset")

    try:
        for step in range(min(MAX_STEPS, int(MAX_SECONDS / DT))):
            time_s = float(data.time)
            stage, phase, duration = phase_at(time_s, durations)
            if stage == "COMPLETE":
                break
            if stage != current_stage:
                if current_stage is not None and stages[current_stage] == "NOT RUN":
                    stages[current_stage] = "PASS"
                current_stage = stage
            if stage in {"INITIAL_SETTLE", "HORIZONTAL_PREGRASP"}:
                arm_target = pregrasp_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                command, command_rate = 100.0, 0.0
            elif stage == "APPROACH":
                arm_target = interpolate(pregrasp_arm, approach_arm, phase)
                arm_velocity = interpolate_velocity(pregrasp_arm, approach_arm, phase, duration)
                arm_acceleration = interpolate_acceleration(pregrasp_arm, approach_arm, phase, duration)
                command, command_rate = 100.0, 0.0
            elif stage == "CLOSE":
                arm_target = approach_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                blend, deriv = smoothstep(phase)
                command = 100.0 + blend * (selected_percent - 100.0)
                command_rate = deriv * (selected_percent - 100.0) / duration
            elif stage == "BILATERAL_PAD_CONTACT":
                arm_target = approach_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                command, command_rate = selected_percent, 0.0
            elif stage == "PRELOAD":
                arm_target = approach_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                blend, deriv = smoothstep(phase)
                command = selected_percent + blend * (args.preload_percent - selected_percent)
                command_rate = deriv * (args.preload_percent - selected_percent) / duration
            elif stage == "PRELOAD_HOLD":
                arm_target = approach_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                command, command_rate = args.preload_percent, 0.0
            elif stage == "OPEN_LIFT_CALIBRATION":
                arm_target = interpolate(approach_arm, lift_arm, phase)
                arm_velocity = interpolate_velocity(approach_arm, lift_arm, phase, duration)
                arm_acceleration = interpolate_acceleration(approach_arm, lift_arm, phase, duration)
                command, command_rate = 100.0, 0.0
            elif stage == "LIFT_50MM":
                arm_target = interpolate(approach_arm, lift_arm, phase)
                arm_velocity = interpolate_velocity(approach_arm, lift_arm, phase, duration)
                arm_acceleration = interpolate_acceleration(approach_arm, lift_arm, phase, duration)
                command, command_rate = args.preload_percent, 0.0
            else:
                arm_target = lift_arm
                arm_velocity = np.zeros(len(ARM_JOINTS))
                arm_acceleration = np.zeros(len(ARM_JOINTS))
                command, command_rate = (100.0, 0.0) if args.open_lift_calibration else (args.preload_percent, 0.0)
            if args.adaptive_pad_centering and stage in {"BILATERAL_PAD_CONTACT", "PRELOAD", "PRELOAD_HOLD", "LIFT_50MM", "HOLD_0_5S"}:
                correction = static_arm_at_offset(adaptive_offset_mm) - static_baseline_arm
                arm_target = arm_target + correction
            for name, value in zip(ARM_JOINTS, arm_target):
                # Arm coordinates remain physical MuJoCo degrees of freedom. Only
                # their existing motor targets change; their qpos/qvel are never written here.
                motor_targets[name] = float(value)
            # The table writes only the six reviewed STL hand coordinates, never the object.
            pose = apply_stl_kinematic_pose(mujoco, model, data, trajectory, command, command_rate_percent_s=command_rate)
            metrics["kinematic_hand_qpos_writes"] += 1
            reference_feedforward = (
                reference_inverse_no_contact(reference_model, data, motor_targets, trajectory, command, command_rate, arm_velocity, arm_acceleration)
                if reference_model is not None
                else None
            )
            saturation, requested, control_components = configure_controls(model, data, motor_targets, left_root, args.feedforward_mode, args.arm_gain_scale, reference_feedforward)
            for name, value in requested.items():
                metrics["max_requested_ctrl"][name] = max(abs(value), metrics["max_requested_ctrl"].get(name, 0.0))
            contacts = contact_data(model, data)
            target_data = mujoco.MjData(model)
            target_data.qpos[:] = data.qpos
            target_data.qvel[:] = 0.0
            for name, value in zip(ARM_JOINTS, arm_target):
                target_data.qpos[model.jnt_qposadr[joint_id(model, name)]] = float(value)
            apply_stl_kinematic_pose(mujoco, model, target_data, trajectory, command)
            base_error = float(np.linalg.norm(data.xpos[gripper_base] - target_data.xpos[gripper_base]))
            active_reference = physical_lift_reference if stage in {"OPEN_LIFT_CALIBRATION", "LIFT_50MM", "HOLD_0_5S", "CALIBRATION_HOLD"} and (args.open_lift_calibration or stage in {"LIFT_50MM", "HOLD_0_5S"}) else physical_alignment_reference
            physical_alignment_error = float(np.linalg.norm(data.xpos[gripper_base] - active_reference))
            metrics["max_gripper_base_position_error_m"] = max(float(metrics["max_gripper_base_position_error_m"]), base_error)
            metrics["max_physical_alignment_error_m"] = max(float(metrics["max_physical_alignment_error_m"]), physical_alignment_error)
            com_rise = float(data.xipos[cylinder_body][2] - initial_com[2])
            metrics["max_cylinder_com_rise_m"] = max(float(metrics["max_cylinder_com_rise_m"]), com_rise)
            for name, value in contacts["pad_force_norm_N"].items():
                metrics["max_pad_force_norm_N"][name] = max(float(metrics["max_pad_force_norm_N"][name]), value)
            pad_ready = all(value >= BILATERAL_FORCE_GATE_N for value in contacts["pad_force_norm_N"].values())
            if args.adaptive_pad_centering and stage in {"BILATERAL_PAD_CONTACT", "PRELOAD", "PRELOAD_HOLD"} and step - last_adaptive_update_step >= 50:
                narrow_force = contacts["pad_force_norm_N"].get("R_hand_narrow3_Link", 0.0)
                wide_force = contacts["pad_force_norm_N"].get("R_hand_wide3_Link", 0.0)
                if narrow_force >= BILATERAL_FORCE_GATE_N and wide_force < BILATERAL_FORCE_GATE_N:
                    adaptive_offset_mm = max(static_offsets[0], adaptive_offset_mm - 0.10)
                    adaptive_updates.append({"step": step + 1, "direction": "toward_wide", "offset_mm": adaptive_offset_mm})
                    last_adaptive_update_step = step
                elif wide_force >= BILATERAL_FORCE_GATE_N and narrow_force < BILATERAL_FORCE_GATE_N:
                    adaptive_offset_mm = min(static_offsets[-1], adaptive_offset_mm + 0.10)
                    adaptive_updates.append({"step": step + 1, "direction": "toward_narrow", "offset_mm": adaptive_offset_mm})
                    last_adaptive_update_step = step
            bilateral_streak = bilateral_streak + 1 if pad_ready else 0
            if pad_ready and metrics["first_bilateral_step"] is None:
                metrics["first_bilateral_step"] = step + 1
            if not contacts["table_contacts"] and metrics["first_table_clear_step"] is None:
                metrics["first_table_clear_step"] = step + 1
            object_linear_speed = float(np.linalg.norm(data.qvel[cylinder_dadr:cylinder_dadr + 3]))
            object_angular_speed = float(np.linalg.norm(data.qvel[cylinder_dadr + 3:cylinder_dadr + 6]))
            root_q = float(data.qpos[model.jnt_qposadr[joint_id(model, "right_claw_joint")]])
            root_error = abs(root_q - pose.positions_rad["right_claw_joint"])
            metrics["max_root_command_error_rad"] = max(float(metrics["max_root_command_error_rad"]), root_error)
            arm_error = max(abs(float(data.qpos[model.jnt_qposadr[joint_id(model, name)]]) - float(target)) for name, target in zip(ARM_JOINTS, arm_target))
            metrics["max_arm_tracking_error_rad"] = max(float(metrics["max_arm_tracking_error_rad"]), arm_error)
            for actuator in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or f"actuator#{actuator}"
                metrics["max_actuator_force"][name] = max(abs(float(data.actuator_force[actuator])), metrics["max_actuator_force"].get(name, 0.0))
            if step < 150 or step % 10 == 0:
                trace.write(json.dumps({
                    "step": step + 1, "time_s": time_s, "stage": stage, "command_percent": command,
                    "cylinder_com_rise_m": com_rise, "cylinder_linear_speed_m_s": object_linear_speed,
                    "cylinder_angular_speed_rad_s": object_angular_speed,
                    "pad_force_norm_N": contacts["pad_force_norm_N"],
                    "pad_force_on_cylinder_world_N": contacts["pad_forces_on_cylinder_world_N"],
                    "table_force_on_cylinder_world_N": contacts["table_force_on_cylinder_world_N"],
                    "cylinder_com_world_m": data.xipos[cylinder_body].tolist(),
                    "table_contact_count": len(contacts["table_contacts"]),
                    "nonpad_cylinder_contacts": contacts["nonpad_cylinder_contacts"],
                    "arm_error_rad": arm_error, "root_error_rad": root_error,
                    "gripper_base_position_error_m": base_error,
                    "physical_alignment_error_m": physical_alignment_error,
                    "adaptive_lateral_offset_mm": adaptive_offset_mm,
                    "gripper_base_world_m": data.xpos[gripper_base].tolist(),
                    "target_gripper_base_world_m": target_data.xpos[gripper_base].tolist(),
                    "actuator_ctrl": data.ctrl.tolist(), "actuator_force": data.actuator_force.tolist(),
                    "right_wrist_roll_control": control_components.get("motor_right_wrist_roll_joint"),
                }, ensure_ascii=False) + "\n")
            if saturation:
                failure = {"gate": "ACTUATOR_SATURATION", "step": step + 1, "stage": stage, "details": saturation, "gripper_base_position_error_m": base_error, "pad_force_norm_N": contacts["pad_force_norm_N"], "right_wrist_roll_control": control_components.get("motor_right_wrist_roll_joint")}
            elif contacts["nonpad_cylinder_contacts"]:
                failure = {"gate": "NONPAD_CYLINDER_CONTACT", "step": step + 1, "stage": stage, "details": contacts["nonpad_cylinder_contacts"]}
            elif contacts["robot_table_contacts"]:
                failure = {"gate": "ROBOT_TABLE_CONTACT", "step": step + 1, "stage": stage, "details": contacts["robot_table_contacts"]}
            elif arm_error > ARM_TRACKING_GATE_RAD:
                failure = {"gate": "ARM_TRACKING_ERROR", "step": step + 1, "stage": stage, "error_rad": arm_error}
            else:
                for joint in range(model.njnt):
                    if not bool(model.jnt_limited[joint]):
                        continue
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
                    value = float(data.qpos[model.jnt_qposadr[joint]])
                    low, high = map(float, model.jnt_range[joint])
                    if value < low - POSITION_MARGIN_RAD or value > high + POSITION_MARGIN_RAD:
                        failure = {"gate": "JOINT_LIMIT", "step": step + 1, "stage": stage, "joint": name, "qpos_rad": value, "range_rad": [low, high]}
                        break
                if failure is None:
                    for name, limit in source_limits.items():
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
                            continue
                        velocity = abs(float(data.qvel[model.jnt_dofadr[joint_id(model, name)]]))
                        if velocity > limit * 1.05 + 1e-9:
                            failure = {"gate": "SOURCE_VELOCITY_LIMIT", "step": step + 1, "stage": stage, "joint": name, "qvel_rad_s": velocity, "limit_rad_s": limit}
                            break
                if failure is None and not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    failure = {"gate": "NONFINITE_STATE", "step": step + 1, "stage": stage}
                if failure is None and np.any(np.asarray(data.warning.number, dtype=int) > warnings_before):
                    failure = {"gate": "MUJOCO_WARNING", "step": step + 1, "stage": stage, "warnings": data.warning.number.tolist()}
            if failure is None and stage == "BILATERAL_PAD_CONTACT" and phase >= 0.999 and bilateral_streak < 100:
                failure = {"gate": "BILATERAL_PAD_CONTACT_NOT_SUSTAINED", "step": step + 1, "pad_force_norm_N": contacts["pad_force_norm_N"]}
            if failure is None and stage == "PRELOAD_HOLD" and phase >= 0.999 and bilateral_streak < 100:
                failure = {"gate": "PRELOAD_BILATERAL_CONTACT_NOT_SUSTAINED", "step": step + 1, "pad_force_norm_N": contacts["pad_force_norm_N"]}
            if args.record_video and step % 40 == 0:
                write_video_frame(renderer, camera, data, video_writer)
            if failure is not None:
                break
            mujoco.mj_step(model, data)
            physics_steps += 1
        if current_stage is not None and failure is None:
            stages[current_stage] = "PASS"
    finally:
        trace.close()
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()
    final_contacts = contact_data(model, data)
    final_rise = float(data.xipos[cylinder_body][2] - initial_com[2])
    complete = failure is None and physics_steps >= int(expected_duration / DT) - 1
    final_alignment_error = float(np.linalg.norm(data.xpos[gripper_base] - physical_alignment_reference))
    if complete:
        for phase in durations:
            if stages[phase] == "NOT RUN":
                stages[phase] = "PASS"
        if args.calibration_only or args.open_lift_calibration:
            calibration_reference = physical_lift_reference if args.open_lift_calibration else physical_alignment_reference
            final_alignment_error = float(np.linalg.norm(data.xpos[gripper_base] - calibration_reference))
            if final_alignment_error > 0.0005:
                failure = {"gate": "OPEN_HAND_ALIGNMENT_ERROR", "error_m": final_alignment_error, "limit_m": 0.0005}
            elif final_contacts["nonpad_cylinder_contacts"]:
                failure = {"gate": "UNEXPECTED_OPEN_HAND_CYLINDER_CONTACT", "contacts": final_contacts["nonpad_cylinder_contacts"]}
        elif not all(value >= BILATERAL_FORCE_GATE_N for value in final_contacts["pad_force_norm_N"].values()):
            failure = {"gate": "FINAL_BILATERAL_CONTACT_LOST", "pad_force_norm_N": final_contacts["pad_force_norm_N"]}
        elif final_contacts["table_contacts"]:
            failure = {"gate": "TABLE_SUPPORT_NOT_TRANSFERRED", "table_contacts": final_contacts["table_contacts"]}
        elif final_rise < 0.050 - 0.001:
            failure = {"gate": "LIFT_50MM_NOT_REACHED", "final_cylinder_com_rise_m": final_rise}
    status = "PASS" if failure is None and complete else "FAIL"
    result = {
        "schema": "robotsim.issue46.full-x2-stl-cylinder-lift.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "status": status,
        "first_failure": failure,
        "mj_step_calls": physics_steps,
        "simulated_time_s": float(data.time),
        "mujoco_version": mujoco.__version__,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "static_preflight": {"path": str(args.static_result), "sha256": sha256(args.static_result)},
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "sequence": {"durations_s": durations, "open_percent": 100, "contact_percent": selected_percent, "preload_percent": args.preload_percent, "lateral_alignment_mm": aligned_contact["lateral_offset_mm"], "kinematic_hand_qpos_driver": True, "calibration_only": args.calibration_only, "open_lift_calibration": args.open_lift_calibration},
        "adaptive_pad_centering": {"enabled": args.adaptive_pad_centering, "final_lateral_offset_mm": adaptive_offset_mm, "updates": adaptive_updates},
        "controller_overlay": {
            "id": "STL_GRAVITY_BIAS_PD_CONTROL_V1" if args.feedforward_mode == "gravity_bias" else "STL_REFERENCE_INVERSE_NO_CONTACT_PD_V1" if args.feedforward_mode == "reference_inverse_no_contact" else "CONTACT_COUPLED_INVERSE_PD_DIAGNOSTIC",
            "feedforward_mode": args.feedforward_mode,
            "motor_gains": {"head": [20.0, 2.0], "arm": [30.0 * args.arm_gain_scale, 4.0 * args.arm_gain_scale], "other": [40.0, 5.0]},
            "arm_gain_scale": args.arm_gain_scale,
            "motor_limits": "unchanged source model actuator ctrlrange",
        },
        "stages": stages,
        "metrics": metrics,
        "final": {
            "cylinder_com_rise_m": final_rise,
            "physical_alignment_error_m": final_alignment_error,
            "table_contact_count": len(final_contacts["table_contacts"]),
            "pad_force_norm_N": final_contacts["pad_force_norm_N"],
            "nonpad_cylinder_contacts": final_contacts["nonpad_cylinder_contacts"],
        },
        "qualification": {
            "physical_cylinder_dynamics": "PASS" if status == "PASS" else "FAIL",
            "manufacturer_transmission": "NOT_QUALIFIED",
            "runtime_object_qpos_writes": 0,
            "runtime_hand_qpos_writes": metrics["kinematic_hand_qpos_writes"],
        },
        "reproduction_command": "see source-controlled runner and result inputs; output directory must be fresh",
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": status, "steps": physics_steps, "stages": stages, "final_lift_mm": final_rise * 1000.0, "first_failure": failure, "output_wsl": location.wsl_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
