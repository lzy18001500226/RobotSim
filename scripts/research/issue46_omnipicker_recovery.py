#!/usr/bin/env python3
"""Pinned-source recovery validation for the X2 OmniPicker M0 prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
SDK_PIN = "c23801f03e396d8d667a307548a05b5d0aecf9aa"
X1_PIN = "9e0b818804d644fb9c9663e932dd33b03b24dfa4"
X2_URDF = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
X2_URDF_SHA = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
SDK_HEADER = Path("linux/x64/cpp/include/omnihand/omnipicker_2025.h")
SDK_HEADER_SHA = "7298b4f96d179ec62e99876b05641d9ac04b2788972ea626c1c81bc3ab890239"
X1_HEADER = Path("src/module/dcu_driver_module/xyber_controller/xyber_api/include/internal/omni_picker.h")
X1_HEADER_SHA = "22f418b2e0640c291b679c72ef0f96708d1e46cc2e866c3993ed4fd94edc25a6"
URDF_ROOT = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
SDK_ROOT = Path("/tmp/robotsim-issue46-agillink-c238")
X1_ROOT = Path("/tmp/robotsim-issue46-x1-infer-9e0")
STATIC_ROOT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready")
OUTPUT_ROOT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-dynamic/20261007-zero-metadata-recovery-v1")
TIMESTEP = 0.002
BASE_WORLD = np.array([0.38, 0.32, 0.68], dtype=float)
BASE_YAW = -math.pi / 2.0
LEFT_DRIVER_TARGET = -0.5
APERTURE_RATIO_OPEN = 1.0
APERTURE_RATIO_CLOSED = 0.0
TRAJECTORY_DURATION_S = 2.5
NATURAL_FREQUENCY_RAD_S = 10.0
DAMPING_RATIO = 1.0
TORQUE_CAP_NM = 0.08
EXPECTED_MAX_QVEL_RAD_S = 1.875 / TRAJECTORY_DURATION_S
EXPECTED_MAX_QACC_RAD_S2 = 5.773502691896258 / TRAJECTORY_DURATION_S**2
SIM_MAX_QVEL_RAD_S = 1.5 * EXPECTED_MAX_QVEL_RAD_S
SIM_MAX_QACC_RAD_S2 = 2.0 * EXPECTED_MAX_QACC_RAD_S2
MIMIC_DIAGNOSTIC_RAD = 0.003
LEFT_LOCK_SOLREF = [0.002, 1.0]
POSE_LOCK_SOLREF = [0.002, 1.0]
PICKERS = (
    ("left_claw_joint", "L_hand_wide1_joint", "left_picker_mimic"),
    ("right_claw_joint", "R_hand_wide1_joint", "right_picker_mimic"),
)
STATIC_COLLISION_EXCLUSIONS = (
    ("pelvis", "left_hip_pitch_link"),
    ("pelvis", "left_hip_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("pelvis", "right_hip_roll_link"),
    ("pelvis", "waist_yaw_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("waist_yaw_link", "torso_link"),
    ("torso_link", "head_pitch_link"),
    ("left_wrist_yaw_link", "left_wrist_roll_link"),
    ("right_wrist_yaw_link", "right_wrist_roll_link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_clean(root: Path) -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()


def verify_pin(root: Path, pin: str, relative_header: Path, header_sha: str) -> dict[str, object]:
    actual = git_head(root)
    clean = git_clean(root)
    header = root / relative_header
    actual_sha = sha256(header)
    if actual != pin or not clean or actual_sha != header_sha:
        raise RuntimeError(f"Pinned source identity failed for {root}: {actual}, clean={clean}, header_sha={actual_sha}")
    return {"root_label": root.name, "commit": actual, "expected_commit": pin, "working_tree_clean": clean,
            "header_path": str(relative_header), "header_sha256": actual_sha, "expected_header_sha256": header_sha}


def source_joint_audit(urdf_path: Path) -> tuple[dict[str, object], dict[str, ET.Element]]:
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    names = [name for pair in PICKERS for name in pair[:2]]
    rows = []
    for name in names:
        joint = joints[name]
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        rows.append({
            "name": name,
            "type": joint.get("type"),
            "axis": [float(x) for x in joint.find("axis").get("xyz", "0 0 1").split()],
            "position_range_rad": [float(limit.get("lower")), float(limit.get("upper"))],
            "source_velocity_metadata": float(limit.get("velocity")),
            "source_effort_metadata": float(limit.get("effort")),
            "dynamic_metadata_interpretation": "UNSPECIFIED_DYNAMIC_METADATA",
            "mimic": None if mimic is None else {
                "driver": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            },
        })
    mimic_pairs = {row["name"]: row["mimic"] for row in rows if row["mimic"]}
    if len(mimic_pairs) != 2:
        raise RuntimeError("Expected exactly two source picker mimic relations")
    return ({
        "repository": "https://github.com/AgibotTech/agibot_x2_urdf",
        "commit": git_head(URDF_ROOT),
        "urdf_path": str(X2_URDF),
        "urdf_sha256": sha256(urdf_path),
        "expected_urdf_sha256": X2_URDF_SHA,
        "license": "Mulan PSL v2",
        "joint_records": rows,
        "source_zero_metadata_reclassification": "Preserved as literal source values; treated as unspecified dynamic metadata for simulation, not as physical zero bounds.",
        "urdf_transmission_count": len(root.findall(".//transmission")),
    }, joints)


def dof_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Missing compiled joint {name}")
    return int(model.jnt_dofadr[jid])


def qpos_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Missing compiled joint {name}")
    return int(model.jnt_qposadr[jid])


def body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing compiled body {name}")
    return int(bid)


def spec_joint_map(spec: mujoco.MjSpec) -> dict[str, object]:
    return {str(joint.name): joint for joint in spec.joints if joint.name}


def apply_picker_and_lock_constraints(spec: mujoco.MjSpec, lock_rest: bool) -> None:
    joints = spec_joint_map(spec)
    for driver, follower, eq_name in PICKERS:
        driver_ref = float(joints[driver].ref)
        follower_ref = float(joints[follower].ref)
        multiplier = -1.0
        offset = 0.0
        a0 = multiplier * driver_ref + offset - follower_ref
        spec.add_equality(
            name=eq_name,
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=follower,
            name2=driver,
            data=[a0, multiplier, 0.0, 0.0, 0.0] + [0.0] * 6,
        )
    left_driver = "left_claw_joint"
    left_ref = float(joints[left_driver].ref)
    spec.add_equality(
        name="unused_left_picker_lock",
        type=mujoco.mjtEq.mjEQ_JOINT,
        name1=left_driver,
        data=[LEFT_DRIVER_TARGET - left_ref, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6,
        solref=LEFT_LOCK_SOLREF,
    )
    if lock_rest:
        skipped = {"left_claw_joint", "L_hand_wide1_joint", "right_claw_joint", "R_hand_wide1_joint"}
        for name, joint in joints.items():
            if name in skipped:
                continue
            spec.add_equality(
                name=f"m0_fixed_pose_{name}",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=name,
                data=[0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6,
                solref=POSE_LOCK_SOLREF,
            )


def add_robot_scene(spec: mujoco.MjSpec, include_environment: bool, include_table: bool, include_bottle: bool) -> None:
    if not include_environment:
        return
    spec.add_texture(
        name="issue46_floor_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300,
    )
    spec.add_material(name="issue46_floor", textures=["", "issue46_floor_texture"], texuniform=True,
                      texrepeat=[5.0, 5.0], reflectance=0.2)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
                            material="issue46_floor", group=1)
    spec.worldbody.add_light(name="issue46_key", pos=[0.0, 0.0, 1.5], dir=[0.0, 0.0, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if include_table or include_bottle:
        sys.path.insert(0, str(STATIC_ROOT))
        import canonical_manipulation_assets as canonical  # type: ignore[import-not-found]
        if include_table:
            canonical.add_g1_canonical_table(spec)
        if include_bottle:
            bottle = spec.worldbody.add_body(name="bottle", pos=list(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS))
            bottle.add_freejoint(name="bottle_free")
            geom_types = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
            for item in canonical.CANONICAL_X2_BOTTLE_GEOMS:
                bottle.add_geom(name=item["name"], type=geom_types[item["type"]], pos=list(item["pos"]),
                                size=list(item["size"]), mass=float(item["mass"]), rgba=list(item["rgba"]),
                                friction=[1.4, 0.02, 0.001], condim=4, group=1)


def build_spec(include_table: bool = False, include_bottle: bool = False, fixed_object: list[float] | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(URDF_ROOT / X2_URDF))
    spec.compiler.fusestatic = False
    spec.option.timestep = TIMESTEP
    spec.option.gravity = [0.0, 0.0, -9.81]
    pelvis = next((body for body in spec.bodies if body.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("Pinned URDF has no pelvis root")
    pelvis.pos = BASE_WORLD.tolist()
    pelvis.quat = [math.cos(BASE_YAW / 2.0), 0.0, 0.0, math.sin(BASE_YAW / 2.0)]
    for i, (body_a, body_b) in enumerate(STATIC_COLLISION_EXCLUSIONS):
        spec.add_exclude(name=f"accepted_static_same_chain_overlap_{i}", bodyname1=body_a, bodyname2=body_b)
    apply_picker_and_lock_constraints(spec, lock_rest=True)
    spec.add_actuator(
        name="right_picker_internal_motor", target="right_claw_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT,
        dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
        ctrllimited=True, ctrlrange=[-TORQUE_CAP_NM, TORQUE_CAP_NM],
        forcelimited=True, forcerange=[-TORQUE_CAP_NM, TORQUE_CAP_NM],
    )
    add_robot_scene(spec, include_table or include_bottle or fixed_object is not None, include_table, include_bottle)
    if fixed_object is not None:
        spec.worldbody.add_geom(name="fixed_contact_object", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                pos=fixed_object, size=[0.035, 0.10, 0.0], rgba=[0.92, 0.46, 0.08, 1.0],
                                friction=[1.4, 0.02, 0.001], condim=4, group=1)
    return spec


def initial_values(model: mujoco.MjModel) -> dict[str, float]:
    values = {"left_claw_joint": LEFT_DRIVER_TARGET, "L_hand_wide1_joint": -LEFT_DRIVER_TARGET,
              "right_claw_joint": -APERTURE_RATIO_OPEN, "R_hand_wide1_joint": APERTURE_RATIO_OPEN}
    return values


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    mujoco.mj_resetData(model, data)
    values = initial_values(model)
    for name, value in values.items():
        data.qpos[qpos_id(model, name)] = value
    data.qvel[:] = 0.0
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_picker_internal_motor")
    if actuator < 0 or model.nu != 1:
        raise RuntimeError("Expected exactly one internal right-picker motor and no follower actuator")
    data.ctrl[actuator] = 0.0
    mujoco.mj_forward(model, data)
    return int(actuator)


def min_jerk(t: float, duration: float, start: float, end: float) -> tuple[float, float, float]:
    if t <= 0.0:
        return start, 0.0, 0.0
    if t >= duration:
        return end, 0.0, 0.0
    s = t / duration
    p = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    dp = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    ddp = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    return start + (end - start) * p, (end - start) * dp, (end - start) * ddp


def effective_terms(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    d = dof_id(model, "right_claw_joint")
    f = dof_id(model, "R_hand_wide1_joint")
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    # Source mimic q_f = -q_d gives the tangent vector v=[1,-1].
    inertia = float(mass[d, d] + mass[f, f] - mass[d, f] - mass[f, d])
    bias = float(data.qfrc_bias[d] - data.qfrc_bias[f])
    return inertia, bias


def reference_for_phase(phase: str, phase_time: float) -> tuple[float, float, float, float]:
    if phase in {"OPEN_HOLD", "OPEN_HOLD_FINAL", "APPROACH_OPEN"}:
        ratio, ratio_dot, ratio_ddot = APERTURE_RATIO_OPEN, 0.0, 0.0
    elif phase in {"CLOSE", "APPROACH_CLOSE"}:
        ratio, ratio_dot, ratio_ddot = min_jerk(phase_time, TRAJECTORY_DURATION_S, 1.0, 0.0)
    elif phase in {"REOPEN"}:
        ratio, ratio_dot, ratio_ddot = min_jerk(phase_time, TRAJECTORY_DURATION_S, 0.0, 1.0)
    elif phase in {"CLOSED_HOLD", "CONTACT_HOLD", "BOTTLE_HOLD"}:
        ratio, ratio_dot, ratio_ddot = APERTURE_RATIO_CLOSED, 0.0, 0.0
    else:
        raise ValueError(f"Unknown picker phase: {phase}")
    return -ratio, -ratio_dot, -ratio_ddot, ratio


def schedule(stage: str) -> list[tuple[str, float]]:
    if stage == "no_contact":
        return [("OPEN_HOLD", 0.25), ("CLOSE", TRAJECTORY_DURATION_S), ("CLOSED_HOLD", 0.25),
                ("REOPEN", TRAJECTORY_DURATION_S), ("OPEN_HOLD_FINAL", 0.50)]
    if stage == "fixed_object":
        return [("OPEN_HOLD", 0.25), ("CLOSE", TRAJECTORY_DURATION_S), ("CONTACT_HOLD", 1.0),
                ("REOPEN", TRAJECTORY_DURATION_S), ("OPEN_HOLD_FINAL", 0.50)]
    raise ValueError(stage)


def contact_record(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    out = []
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        f = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, f)
        n1, n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1]))
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2]))
        out.append({"geom1": n1, "body1": b1, "geom2": n2, "body2": b2,
                    "distance_m": float(con.dist), "penetration_m": max(0.0, -float(con.dist)),
                    "force_contact_frame": f.tolist(), "normal_force_n": float(abs(f[0]))})
    return out


def source_limit_map(root: ET.Element) -> dict[str, tuple[float, float]]:
    out = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            out[joint.get("name", "")] = (float(limit.get("lower")), float(limit.get("upper")))
    return out


def run_stage(model: mujoco.MjModel, stage: str, out: Path, source_limits: dict[str, tuple[float, float]],
              expected_object_geom: str | None = None) -> dict[str, object]:
    data = mujoco.MjData(model)
    actuator = initialize(model, data)
    init_contacts = contact_record(model, data)
    initial_penetrations = [c for c in init_contacts if c["penetration_m"] > 1e-8]
    if initial_penetrations:
        return {"status": "FAIL", "first_failed_gate": "initial penetration", "initial_penetrations": initial_penetrations}
    eq_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
              for name in ["left_picker_mimic", "right_picker_mimic", "unused_left_picker_lock"]}
    trace_path = out / f"{stage}_trace.jsonl"
    max_eq = 0.0
    max_qvel = 0.0
    max_qacc = 0.0
    max_effort = 0.0
    max_bias = 0.0
    max_penetration = 0.0
    source_position_violations: list[dict[str, object]] = []
    nonfinite = False
    contacts_seen: dict[str, int] = {}
    contact_frames = 0
    max_contact_force = 0.0
    first_failure: dict[str, object] | None = None
    trace_count = 0
    final_phase = "NOT_STARTED"
    phase_start = 0.0
    schedule_rows = schedule(stage)
    total_steps = sum(max(1, round(duration / TIMESTEP)) for _, duration in schedule_rows)
    all_geom_ids = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    with trace_path.open("w", encoding="utf-8") as stream:
        global_step = 0
        for phase, duration in schedule_rows:
            phase_steps = max(1, round(duration / TIMESTEP))
            phase_start = float(data.time)
            for local_step in range(phase_steps):
                phase_time = local_step * TIMESTEP
                target_q, target_v, target_a, aperture_ratio = reference_for_phase(phase, phase_time)
                inertia, bias_eff = effective_terms(model, data)
                qd = float(data.qpos[qpos_id(model, "right_claw_joint")])
                vd = float(data.qvel[dof_id(model, "right_claw_joint")])
                tau_raw = bias_eff + inertia * (target_a + 2.0 * DAMPING_RATIO * NATURAL_FREQUENCY_RAD_S * (target_v - vd)
                                                + NATURAL_FREQUENCY_RAD_S**2 * (target_q - qd))
                tau = float(np.clip(tau_raw, -TORQUE_CAP_NM, TORQUE_CAP_NM))
                data.ctrl[actuator] = tau
                mujoco.mj_step(model, data)
                global_step += 1
                trace_count += 1
                qd = float(data.qpos[qpos_id(model, "right_claw_joint")])
                qf = float(data.qpos[qpos_id(model, "R_hand_wide1_joint")])
                vd = float(data.qvel[dof_id(model, "right_claw_joint")])
                vf = float(data.qvel[dof_id(model, "R_hand_wide1_joint")])
                ad = float(data.qacc[dof_id(model, "right_claw_joint")])
                af = float(data.qacc[dof_id(model, "R_hand_wide1_joint")])
                residual = qf + qd
                max_eq = max(max_eq, abs(residual))
                max_qvel = max(max_qvel, abs(vd), abs(vf))
                max_qacc = max(max_qacc, abs(ad), abs(af))
                max_effort = max(max_effort, abs(float(data.actuator_force[actuator])))
                max_bias = max(max_bias, abs(bias_eff))
                contacts = contact_record(model, data)
                if contacts:
                    contact_frames += 1
                for contact in contacts:
                    pair = "|".join(sorted((str(contact["geom1"]), str(contact["geom2"]))))
                    contacts_seen[pair] = contacts_seen.get(pair, 0) + 1
                    max_penetration = max(max_penetration, float(contact["penetration_m"]))
                    max_contact_force = max(max_contact_force, float(contact["normal_force_n"]))
                joint_rows = {}
                for name, (lower, upper) in source_limits.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid < 0:
                        continue
                    q = float(data.qpos[int(model.jnt_qposadr[jid])])
                    v = float(data.qvel[int(model.jnt_dofadr[jid])])
                    a = float(data.qacc[int(model.jnt_dofadr[jid])])
                    if q < lower - 1e-10 or q > upper + 1e-10:
                        source_position_violations.append({"step": global_step, "joint": name, "qpos": q, "range": [lower, upper]})
                    if not np.isfinite([q, v, a]).all():
                        nonfinite = True
                    joint_rows[name] = {"qpos_rad": q, "qvel_rad_s": v, "qacc_rad_s2": a,
                                        "source_range_rad": [lower, upper],
                                        "within_source_position_range": lower - 1e-10 <= q <= upper + 1e-10}
                max_pen = max((float(c["penetration_m"]) for c in contacts), default=0.0)
                sample = {
                    "step": global_step, "time_s": float(data.time), "phase": phase,
                    "aperture_ratio_target": aperture_ratio, "driver_target_rad": target_q,
                    "driver_target_velocity_rad_s": target_v, "driver_target_acceleration_rad_s2": target_a,
                    "driver_qpos_rad": qd, "driver_qvel_rad_s": vd, "driver_qacc_rad_s2": ad,
                    "follower_qpos_rad": qf, "follower_qvel_rad_s": vf, "follower_qacc_rad_s2": af,
                    "mimic_residual_rad": residual, "effective_inertia_kg_m2": inertia,
                    "effective_bias_nm": bias_eff, "torque_unclipped_nm": tau_raw,
                    "torque_command_nm": tau, "torque_applied_nm": float(data.actuator_force[actuator]),
                    "driver_qfrc_actuator_nm": float(data.qfrc_actuator[dof_id(model, "right_claw_joint")]),
                    "driver_qfrc_constraint_nm": float(data.qfrc_constraint[dof_id(model, "right_claw_joint")]),
                    "follower_qfrc_constraint_nm": float(data.qfrc_constraint[dof_id(model, "R_hand_wide1_joint")]),
                    "contact_count": int(data.ncon), "max_contact_penetration_m": max_pen,
                    "contacts": contacts, "source_joints": joint_rows,
                }
                stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                final_phase = phase
                if source_position_violations:
                    first_failure = {"gate": "source position limit", **source_position_violations[0]}
                elif nonfinite:
                    first_failure = {"gate": "finite state", "step": global_step}
                elif max_qvel > SIM_MAX_QVEL_RAD_S:
                    first_failure = {"gate": "simulation velocity bound", "step": global_step, "observed_rad_s": max_qvel}
                elif max_qacc > SIM_MAX_QACC_RAD_S2:
                    first_failure = {"gate": "simulation acceleration bound", "step": global_step, "observed_rad_s2": max_qacc}
                elif max_effort > TORQUE_CAP_NM + 1e-10:
                    first_failure = {"gate": "actuator effort bound", "step": global_step, "observed_nm": max_effort}
                elif max_eq > MIMIC_DIAGNOSTIC_RAD:
                    first_failure = {"gate": "native mimic equality diagnostic", "step": global_step, "observed_rad": max_eq}
                elif stage == "no_contact" and data.ncon > 0:
                    first_failure = {"gate": "no-contact free-space run", "step": global_step, "contacts": contacts}
                if first_failure:
                    break
            if first_failure:
                break

    right_object_pairs = [pair for pair in contacts_seen if expected_object_geom and expected_object_geom in pair]
    narrow_contact = any("R_hand_narrow" in pair and expected_object_geom in pair for pair in contacts_seen) if expected_object_geom else False
    wide_contact = any("R_hand_wide" in pair and expected_object_geom in pair for pair in contacts_seen) if expected_object_geom else False
    contact_loss_at_reopen = None
    if expected_object_geom and final_phase == "OPEN_HOLD_FINAL":
        last_sample = json.loads((out / f"{stage}_trace.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        contact_loss_at_reopen = not any(expected_object_geom in (str(c["geom1"]) + str(c["geom2"])) for c in last_sample["contacts"])
    status = "PASS" if first_failure is None else "FAIL"
    if stage == "fixed_object" and status == "PASS":
        if not narrow_contact or not wide_contact or contact_frames == 0 or not contact_loss_at_reopen:
            status = "FAIL"
            first_failure = {"gate": "fixed-object opposing-jaw contact/reopen", "narrow_contact": narrow_contact,
                             "wide_contact": wide_contact, "contact_loss_on_reopen": contact_loss_at_reopen}
    return {
        "status": status,
        "first_failed_gate": first_failure,
        "steps": trace_count,
        "final_phase": final_phase,
        "max_mimic_residual_rad": max_eq,
        "mimic_diagnostic_target_rad": MIMIC_DIAGNOSTIC_RAD,
        "max_abs_picker_qvel_rad_s": max_qvel,
        "max_abs_picker_qacc_rad_s2": max_qacc,
        "simulation_qvel_bound_rad_s": SIM_MAX_QVEL_RAD_S,
        "simulation_qacc_bound_rad_s2": SIM_MAX_QACC_RAD_S2,
        "max_applied_effort_nm": max_effort,
        "effort_cap_nm": TORQUE_CAP_NM,
        "max_abs_effective_bias_nm": max_bias,
        "source_position_violations": source_position_violations,
        "finite_state": not nonfinite,
        "max_contact_penetration_m": max_penetration,
        "max_contact_normal_force_n": max_contact_force,
        "contact_frames": contact_frames,
        "contact_pair_frame_counts": contacts_seen,
        "fixed_object_geom_pairs": right_object_pairs,
        "fixed_object_narrow_jaw_contact": narrow_contact,
        "fixed_object_wide_jaw_contact": wide_contact,
        "fixed_object_contact_loss_on_reopen": contact_loss_at_reopen,
        "initial_contacts": init_contacts,
        "trace": trace_path.name,
    }


def minimum_jerk_derivation(inertia: float, bias_peak: float) -> dict[str, object]:
    return {
        "classification": "SIMULATION_ONLY_M0 engineering parameters; not AgiBot ratings",
        "effective_inertia_method": "v^T M v for source mimic tangent v=[1,-1]",
        "effective_inertia_at_open_kg_m2": inertia,
        "timestep_s": TIMESTEP,
        "full_driver_stroke_rad": 1.0,
        "trajectory": "minimum jerk, 2.5 s, full 1 rad source stroke",
        "minimum_jerk_peak_velocity_factor": 1.875,
        "minimum_jerk_peak_acceleration_factor": 5.773502691896258,
        "expected_max_qvel_rad_s": EXPECTED_MAX_QVEL_RAD_S,
        "expected_max_qacc_rad_s2": EXPECTED_MAX_QACC_RAD_S2,
        "controller": "critically damped effective-inertia computed-torque plus PD tracking",
        "natural_frequency_rad_s": NATURAL_FREQUENCY_RAD_S,
        "damping_ratio": DAMPING_RATIO,
        "effective_kp_nm_per_rad_at_open": inertia * NATURAL_FREQUENCY_RAD_S**2,
        "effective_kv_nm_s_per_rad_at_open": 2.0 * DAMPING_RATIO * NATURAL_FREQUENCY_RAD_S * inertia,
        "effort_cap_nm": TORQUE_CAP_NM,
        "effort_cap_basis": "0.08 Nm bounds the measured gravity compensation (prior FK audit peak 0.03645 Nm) plus one similar load allowance; it is not a source/hardware rating.",
        "prior_effective_bias_peak_nm": bias_peak,
        "source_position_limits_unchanged": True,
        "source_zero_velocity_and_effort_values_preserved": True,
        "source_dynamic_metadata_interpretation": "UNSPECIFIED_DYNAMIC_METADATA",
        "public_command": "aperture_ratio in [0,1], 0 closed and 1 open; driver q = -aperture_ratio; follower q = -driver by native equality",
        "left_hand": "UNUSED_LEFT_PICKER_LOCKED_FOR_RIGHT_HAND_ONLY_M0 at driver -0.5 rad; follower +0.5 rad; no left actuator",
    }


def copy_accepted_static_images(out: Path) -> None:
    for name in ("omnipicker_overview.png", "omnipicker_front.png", "omnipicker_side.png",
                 "right_omnipicker_closeup.png", "right_omnipicker_open.png", "right_omnipicker_closed.png"):
        shutil.copy2(STATIC_ROOT / name, out / name)


def main() -> int:
    global URDF_ROOT, SDK_ROOT, X1_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=URDF_ROOT)
    parser.add_argument("--sdk-root", type=Path, default=SDK_ROOT)
    parser.add_argument("--x1-infer-root", type=Path, default=X1_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    URDF_ROOT, SDK_ROOT, X1_ROOT = args.source_root.resolve(), args.sdk_root.resolve(), args.x1_infer_root.resolve()
    urdf_path = URDF_ROOT / X2_URDF
    sdk_identity = verify_pin(SDK_ROOT, SDK_PIN, SDK_HEADER, SDK_HEADER_SHA)
    x1_identity = verify_pin(X1_ROOT, X1_PIN, X1_HEADER, X1_HEADER_SHA)
    x2_identity = {"repository": "https://github.com/AgibotTech/agibot_x2_urdf", "commit": git_head(URDF_ROOT),
                   "expected_commit": X2_PIN, "working_tree_clean": git_clean(URDF_ROOT),
                   "urdf_path": str(X2_URDF), "urdf_sha256": sha256(urdf_path), "expected_urdf_sha256": X2_URDF_SHA}
    if x2_identity["commit"] != X2_PIN or not x2_identity["working_tree_clean"] or x2_identity["urdf_sha256"] != X2_URDF_SHA:
        raise RuntimeError(f"Pinned X2 source identity failed: {x2_identity}")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source, _ = source_joint_audit(urdf_path)
    write_path = out / "source_audit.json"
    write_path.write_text(json.dumps({"X2": x2_identity, "OP1_SDK": sdk_identity, "X1_inference_API": x1_identity, "source": source}, indent=2) + "\n", encoding="utf-8")
    model = build_spec().compile()
    if model.nu != 1 or model.neq < 3:
        raise RuntimeError(f"Expected one public right picker motor, source equalities and static locks: nu={model.nu}, neq={model.neq}")
    data = mujoco.MjData(model)
    initialize(model, data)
    inertia, bias = effective_terms(model, data)
    derivation = minimum_jerk_derivation(inertia, abs(bias))
    (out / "controller_derivation.json").write_text(json.dumps(derivation, indent=2) + "\n", encoding="utf-8")
    source_limits = source_limit_map(ET.parse(urdf_path).getroot())
    stage3 = run_stage(model, "no_contact", out, source_limits)
    stages = {"stage_1_source_audit": "PASS", "stage_2_source_equalities_and_controller_build": "PASS",
              "stage_3_no_contact_open_close_reopen": stage3["status"],
              "stage_4_fixed_object_contact": "NOT_RUN", "stage_5_bottle_hold": "NOT_RUN", "stage_6_lifts": "NOT_RUN"}
    final_result = stage3["status"]
    stage4: dict[str, object] | None = None
    if stage3["status"] == "PASS":
        # Center a fixed, massless cylinder between the open source jaw-tip links.
        temp_data = mujoco.MjData(model)
        initialize(model, temp_data)
        mujoco.mj_forward(model, temp_data)
        narrow = temp_data.xpos[body_id(model, "R_hand_narrow3_Link")]
        wide = temp_data.xpos[body_id(model, "R_hand_wide3_Link")]
        center = ((narrow + wide) * 0.5).tolist()
        fixed_model = build_spec(fixed_object=center).compile()
        stage4 = run_stage(fixed_model, "fixed_object", out, source_limits, "fixed_contact_object")
        stages["stage_4_fixed_object_contact"] = stage4["status"]
        final_result = stage4["status"]

    result = {
        "result": final_result,
        "task": "Issue #46 X2 OmniPicker zero-metadata dynamic recovery",
        "source_identity": {"X2": x2_identity, "OP1_SDK": sdk_identity, "X1_inference_API": x1_identity},
        "runtime": {"python": platform.python_version(), "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(), "native_library": str(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "native_library_sha256": sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "timestep_s": TIMESTEP, "gravity_m_s2": [0.0, 0.0, -9.81],
                    "dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq}},
        "metadata": {"source_velocity_values_preserved": [0, 0, 0, 0], "source_effort_values_preserved": [0, 0, 0, 0],
                     "interpretation": "UNSPECIFIED_DYNAMIC_METADATA for simulation; not zero-speed or zero-effort hardware limits."},
        "controller": derivation,
        "support": {"left_picker": "UNUSED_LEFT_PICKER_LOCKED_FOR_RIGHT_HAND_ONLY_M0", "driver_q_rad": LEFT_DRIVER_TARGET,
                    "follower_q_rad": -LEFT_DRIVER_TARGET, "left_actuators": 0, "left_contact_assistance": False,
                    "other_robot_dofs": "held at the accepted static scene pose with native equality constraints for this picker-isolation experiment"},
        "public_aperture_map": {"ratio_min": 0.0, "ratio_max": 1.0, "closed_driver_target_rad": 0.0,
                                 "open_driver_target_rad": -1.0, "driver_target_equation": "q_driver = -aperture_ratio",
                                 "follower_target_equation": "q_follower = -q_driver via native source mimic equality", "follower_actuator": False},
        "stage_status": stages,
        "no_contact_stage": stage3,
        "fixed_object_stage": stage4,
        "forbidden_operations": {"active_rollout_follower_qpos_writes": 0, "active_rollout_object_qpos_writes": 0,
                                  "bottle_weld_or_equality": False, "mocap_or_follow_hand": False, "opposite_hand_assistance": False},
        "claim_boundary": "The X2 OmniPicker position topology and mimic relation are source-derived. The velocity/effort bounds used here are SIMULATION_ONLY_M0 engineering parameters because the pinned X2 URDF provides zero placeholders and an exact X2 hardware dynamic calibration has not been established.",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    copy_accepted_static_images(out)
    command = (f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} --source-root {URDF_ROOT} "
               f"--sdk-root {SDK_ROOT} --x1-infer-root {X1_ROOT} --output {out}")
    (out / "experiment_command.txt").write_text(command + "\n", encoding="utf-8")
    lines = [f"COMMAND={command}", f"X2_SHA={x2_identity['commit']}", f"OP1_SDK_SHA={sdk_identity['commit']}",
             f"X1_INFER_SHA={x1_identity['commit']}", f"MUJOCO={mujoco.__version__} native={mujoco.mj_versionString()}",
             f"MODEL=nq:{model.nq} nv:{model.nv} nu:{model.nu} neq:{model.neq}",
             f"J_EFF={inertia:.12g} kg*m^2 BIAS_EFFECTIVE={bias:.12g} Nm", f"STAGE3={stage3['status']}",
             f"STAGE4={stage4['status'] if stage4 else 'NOT_RUN'}", f"RESULT={final_result}"]
    (out / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if final_result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
