#!/usr/bin/env python3
"""Bounded X2 OmniPicker source audit and first no-contact dynamic gate."""

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


SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
URDF_RELATIVE = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
SOURCE_ROOT_DEFAULT = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
OUTPUT_DEFAULT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-dynamic/20261007-no-contact-first-failure"
)
STATIC_PACKET = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-static/20261007-ready"
)
BASE_WORLD = np.array([0.38, 0.32, 0.68], dtype=float)
BASE_YAW = -math.pi / 2.0
TIMESTEP = 0.002
COLLISION_EXCLUSIONS = (
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
PICKERS = (
    ("left_claw_joint", "L_hand_wide1_joint"),
    ("right_claw_joint", "R_hand_wide1_joint"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def source_audit(urdf_path: Path, source_root: Path) -> tuple[dict[str, object], dict[str, ET.Element]]:
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    records = []
    for driver_name, follower_name in PICKERS:
        for name in (driver_name, follower_name):
            joint = joints[name]
            limit = joint.find("limit")
            axis = joint.find("axis")
            origin = joint.find("origin")
            mimic = joint.find("mimic")
            records.append(
                {
                    "name": name,
                    "type": joint.get("type"),
                    "parent": joint.find("parent").get("link"),
                    "child": joint.find("child").get("link"),
                    "axis": [float(x) for x in axis.get("xyz", "0 0 1").split()],
                    "origin_xyz_m": [float(x) for x in origin.get("xyz", "0 0 0").split()],
                    "origin_rpy_rad": [float(x) for x in origin.get("rpy", "0 0 0").split()],
                    "position_range_rad": [float(limit.get("lower")), float(limit.get("upper"))],
                    "velocity_limit_source_value_rad_s": float(limit.get("velocity")),
                    "effort_limit_source_value_nm": float(limit.get("effort")),
                    "mimic": (
                        {
                            "driver": mimic.get("joint"),
                            "multiplier": float(mimic.get("multiplier", "1")),
                            "offset": float(mimic.get("offset", "0")),
                        }
                        if mimic is not None
                        else None
                    ),
                }
            )
    mimic_count = sum(joint.find("mimic") is not None for joint in joints.values())
    transmission_count = len(root.findall(".//transmission"))
    head = command_output("git", "rev-parse", "HEAD", cwd=source_root)
    dirty = bool(command_output("git", "status", "--porcelain", cwd=source_root))
    return (
        {
            "repository": "https://github.com/AgibotTech/agibot_x2_urdf",
            "commit": head,
            "expected_commit": SOURCE_PIN,
            "working_tree_clean": not dirty,
            "urdf_path": str(urdf_path),
            "urdf_sha256": sha256(urdf_path),
            "expected_urdf_sha256": URDF_SHA256,
            "license": "Mulan PSL v2",
            "joint_records": records,
            "mimic_relation_count": mimic_count,
            "transmission_count": transmission_count,
            "independent_command_dofs_per_picker": 1,
            "right_open": {"right_claw_joint": -1.0, "R_hand_wide1_joint": 1.0},
            "right_closed": {"right_claw_joint": 0.0, "R_hand_wide1_joint": 0.0},
            "literal_zero_velocity_limit_blocks_any_nonzero_velocity": True,
            "literal_zero_effort_limit_does_not_supply_a_positive_source_rating": True,
        },
        joints,
    )


def joint_index(model: mujoco.MjModel, name: str) -> int:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if index < 0:
        raise RuntimeError(f"Compiled model is missing source joint {name}")
    return int(index)


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_index(model, name)])


def dof_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_index(model, name)])


def build_model(urdf_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.compiler.fusestatic = False
    spec.option.timestep = TIMESTEP
    spec.option.gravity = [0.0, 0.0, -9.81]
    pelvis = next((body for body in spec.bodies if body.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("Expected pelvis root from the pinned X2 URDF")
    pelvis.pos = BASE_WORLD.tolist()
    pelvis.quat = [math.cos(BASE_YAW / 2.0), 0.0, 0.0, math.sin(BASE_YAW / 2.0)]
    for index, (body1, body2) in enumerate(COLLISION_EXCLUSIONS):
        spec.add_exclude(name=f"known_static_overlap_{index}", bodyname1=body1, bodyname2=body2)

    # URDF import omits mimic constraints. Add only the two source relations.
    for driver, follower in PICKERS:
        side = "left" if driver == "left_claw_joint" else "right"
        spec.add_equality(
            name=f"{side}_picker_mimic",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=driver,
            name2=follower,
            data=[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    # This is the sole externally commanded coordinate. Values are simulation-derived.
    spec.add_actuator(
        name="right_picker_driver_servo",
        target="right_claw_joint",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        dyntype=mujoco.mjtDyn.mjDYN_NONE,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[5.0] + [0.0] * 9,
        biasprm=[0.0, -5.0, -0.1] + [0.0] * 7,
        ctrllimited=True,
        ctrlrange=[-1.0, 0.0],
        forcelimited=True,
        forcerange=[-0.5, 0.5],
    )
    return spec.compile()


def eq_residual(data: mujoco.MjData, model: mujoco.MjModel, driver: str, follower: str) -> float:
    return float(data.qpos[qpos_address(model, follower)] + data.qpos[qpos_address(model, driver)])


def phase_command(step: int) -> tuple[str, float, float]:
    # OPEN hold 20 ms; then one smooth, bounded 500 ms close. Later phases run only
    # if the source velocity gate remains satisfiable.
    open_steps = 10
    close_steps = 250
    hold_steps = 10
    reopen_steps = 250
    final_open_steps = 10
    if step < open_steps:
        return "OPEN_HOLD", -1.0, 0.0
    cursor = step - open_steps
    if cursor < close_steps:
        x = (cursor + 1) / close_steps
        smooth = x * x * (3.0 - 2.0 * x)
        derivative_per_second = (6.0 * x * (1.0 - x)) / (close_steps * TIMESTEP)
        return "CLOSE", -1.0 * (1.0 - smooth), derivative_per_second
    cursor -= close_steps
    if cursor < hold_steps:
        return "CLOSED_HOLD", 0.0, 0.0
    cursor -= hold_steps
    if cursor < reopen_steps:
        x = (cursor + 1) / reopen_steps
        smooth = x * x * (3.0 - 2.0 * x)
        derivative_per_second = (6.0 * x * (1.0 - x)) / (reopen_steps * TIMESTEP)
        return "REOPEN", -smooth, -derivative_per_second
    cursor -= reopen_steps
    if cursor < final_open_steps:
        return "OPEN_HOLD_FINAL", -1.0, 0.0
    raise StopIteration


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    urdf_path = source_root / URDF_RELATIVE
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not urdf_path.is_file():
        raise SystemExit(f"Missing pinned source URDF: {urdf_path}")

    source, source_joints = source_audit(urdf_path, source_root)
    if (
        source["commit"] != SOURCE_PIN
        or source["urdf_sha256"] != URDF_SHA256
        or not source["working_tree_clean"]
        or source["mimic_relation_count"] != 2
        or source["transmission_count"] != 0
    ):
        raise SystemExit("Pinned source identity check failed; no simulation was run")
    model = build_model(urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    initial_values = {
        "left_claw_joint": 0.0,
        "L_hand_wide1_joint": 0.0,
        "right_claw_joint": -1.0,
        "R_hand_wide1_joint": 1.0,
    }
    for name, value in initial_values.items():
        data.qpos[qpos_address(model, name)] = value
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_picker_driver_servo")
    if actuator_id < 0 or model.nu != 1:
        raise RuntimeError("Expected exactly one externally commanded right-driver actuator")
    data.ctrl[actuator_id] = -1.0
    mujoco.mj_forward(model, data)

    eq_ids = {
        driver: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{'left' if driver.startswith('left') else 'right'}_picker_mimic")
        for driver, _ in PICKERS
    }
    eq_info = {
        driver: {
            "equality_id": int(eq_ids[driver]),
            "model_default_active": bool(model.eq_active0[eq_ids[driver]]),
            "runtime_active_after_forward": bool(data.eq_active[eq_ids[driver]]),
            "compiled_polycoef": [float(x) for x in model.eq_data[eq_ids[driver], :5]],
            "solref": [float(x) for x in model.eq_solref[eq_ids[driver]]],
            "solimp": [float(x) for x in model.eq_solimp[eq_ids[driver]]],
        }
        for driver, _ in PICKERS
    }

    source_limit_map = {
        name: float(joint.find("limit").get("velocity"))
        for _, follower in PICKERS
        for name in (_, follower)
        for joint in [source_joints[name]]
    }
    source_limit_map.update(
        {
            follower: float(source_joints[follower].find("limit").get("velocity"))
            for _, follower in PICKERS
        }
    )

    trace_path = out / "step_trace.jsonl"
    result_path = out / "result.json"
    log_path = out / "run.log"
    command = (
        f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
        f"--source-root {source_root} --output {out}"
    )
    (out / "experiment_command.txt").write_text(command + "\n", encoding="utf-8")
    trace: list[dict[str, object]] = []
    first_velocity_violation: dict[str, object] | None = None
    max_eq_error = 0.0
    max_abs_qvel = 0.0
    max_abs_qacc = 0.0
    position_violations: list[dict[str, object]] = []
    nonfinite = False
    final_phase = "NOT_STARTED"
    max_steps = 10 + 250 + 10 + 250 + 10

    for step in range(max_steps):
        phase, target, target_velocity = phase_command(step)
        data.ctrl[actuator_id] = target
        mujoco.mj_step(model, data)
        time_s = float(data.time)
        final_phase = phase
        joint_state = {}
        for driver, follower in PICKERS:
            for name in (driver, follower):
                qpos = float(data.qpos[qpos_address(model, name)])
                qvel = float(data.qvel[dof_address(model, name)])
                qacc = float(data.qacc[dof_address(model, name)])
                limit = source_joints[name].find("limit")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                velocity_limit = float(limit.get("velocity"))
                inside = lower <= qpos <= upper
                if not inside:
                    position_violations.append({"step": step + 1, "joint": name, "qpos": qpos, "range": [lower, upper]})
                if not np.isfinite([qpos, qvel, qacc]).all():
                    nonfinite = True
                max_abs_qvel = max(max_abs_qvel, abs(qvel))
                max_abs_qacc = max(max_abs_qacc, abs(qacc))
                joint_state[name] = {
                    "qpos_rad": qpos,
                    "qvel_rad_s": qvel,
                    "qacc_rad_s2": qacc,
                    "source_range_rad": [lower, upper],
                    "source_velocity_limit_rad_s": velocity_limit,
                    "within_source_position_range": inside,
                    "within_literal_source_velocity_limit": abs(qvel) <= velocity_limit,
                    "dof_address": dof_address(model, name),
                    "qfrc_actuator_nm": float(data.qfrc_actuator[dof_address(model, name)]),
                    "qfrc_constraint_nm": float(data.qfrc_constraint[dof_address(model, name)]),
                }
                if abs(qvel) > velocity_limit and first_velocity_violation is None:
                    first_velocity_violation = {
                        "step": step + 1,
                        "time_s": time_s,
                        "joint": name,
                        "qvel_rad_s": qvel,
                        "source_velocity_limit_rad_s": velocity_limit,
                        "reason": "nonzero motion exceeds the literal source velocity value 0",
                    }
        right_error = eq_residual(data, model, "right_claw_joint", "R_hand_wide1_joint")
        left_error = eq_residual(data, model, "left_claw_joint", "L_hand_wide1_joint")
        max_eq_error = max(max_eq_error, abs(right_error), abs(left_error))
        sample = {
            "step": step + 1,
            "time_s": time_s,
            "phase": phase,
            "right_driver_target_rad": target,
            "right_driver_target_velocity_rad_s": target_velocity,
            "actuator_force_nm": float(data.actuator_force[actuator_id]),
            "right_mimic_residual_rad": right_error,
            "left_mimic_residual_rad": left_error,
            "contacts": int(data.ncon),
            "joints": joint_state,
        }
        trace.append(sample)
        if first_velocity_violation is not None or position_violations or nonfinite:
            break

    with trace_path.open("w", encoding="utf-8") as stream:
        for sample in trace:
            stream.write(json.dumps(sample, separators=(",", ":")) + "\n")

    expected_actuators = ["right_picker_driver_servo"]
    actual_actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    result = {
        "result": "FAIL_SOURCE_VELOCITY_LIMIT_ZERO" if first_velocity_violation else "STAGE3_NOT_COMPLETED",
        "stage_status": {
            "stage_1_source_audit": "PASS",
            "stage_2_native_coupling": "PASS",
            "stage_3_no_contact_dynamic_validation": "FAIL" if first_velocity_violation else "NOT_COMPLETED",
            "stage_4_fixed_object_contact": "NOT_RUN",
            "stage_5_bottle_hold": "NOT_RUN",
            "stage_6_lift": "NOT_RUN",
        },
        "first_failed_gate": "source velocity limit" if first_velocity_violation else None,
        "source": source,
        "runtime": {
            "python": platform.python_version(),
            "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(),
            "native_library": str(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
            "native_library_sha256": sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
            "timestep_s": TIMESTEP,
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "model_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq},
            "mj_step_calls": len(trace),
        },
        "coupling": {
            "representation": "MuJoCo native joint equality, q_follower = -q_driver",
            "equality_rows": eq_info,
            "external_command_coordinates": ["right_claw_joint"],
            "compiled_actuators": actual_actuators,
            "expected_actuators": expected_actuators,
            "follower_actuators": [],
            "follower_qpos_writes_after_initialization": 0,
            "initial_qpos_values_before_first_step": initial_values,
            "max_mimic_residual_rad": max_eq_error,
            "actuator_profile": {
                "classification": "SIMULATION-DERIVED ONLY; not an AgiBot source rating",
                "kp_nm_per_rad": 5.0,
                "kv_nm_s_per_rad": 0.1,
                "force_bound_nm": 0.5,
            },
        },
        "no_contact_scope": {
            "table_in_model": False,
            "bottle_in_model": False,
            "mocap_bodies": 0,
            "bottle_equality_or_weld": False,
            "object_qpos_writes": 0,
            "contacts_observed": max((int(sample["contacts"]) for sample in trace), default=0),
            "note": "The accepted static scene was not altered; this free-space mechanism probe omits table and bottle entirely.",
        },
        "checks": {
            "first_source_velocity_violation": first_velocity_violation,
            "source_position_violations": position_violations,
            "nonfinite_state": nonfinite,
            "max_abs_qvel_rad_s": max_abs_qvel,
            "max_abs_qacc_rad_s2": max_abs_qacc,
            "final_phase": final_phase,
            "close_reopen_sequence_completed": final_phase == "OPEN_HOLD_FINAL",
            "source_velocity_limit_gate": "FAIL" if first_velocity_violation else "PASS",
        },
        "experiment_command": command,
        "trace_path": str(trace_path),
    }
    write_json(result_path, result)

    lines = [
        "EXPERIMENT=Issue #46 X2 OmniPicker first no-contact dynamic gate",
        f"COMMAND={command}",
        f"SOURCE_HEAD={source['commit']}",
        f"URDF_SHA256={source['urdf_sha256']}",
        f"MUJOCO={mujoco.__version__} native={mujoco.mj_versionString()}",
        f"MODEL=nq:{model.nq} nv:{model.nv} nu:{model.nu} neq:{model.neq}",
        f"MIMIC_EQ=max_error:{max_eq_error:.12g} follower_actuators:0 follower_qpos_writes_after_init:0",
        f"SOURCE_VELOCITY_LIMITS={json.dumps(source_limit_map, sort_keys=True)}",
        f"STEPS={len(trace)} FINAL_PHASE={final_phase}",
        f"MAX_ABS_QVEL={max_abs_qvel:.12g} MAX_ABS_QACC={max_abs_qacc:.12g}",
        f"FIRST_VELOCITY_VIOLATION={json.dumps(first_velocity_violation, sort_keys=True)}",
        f"RESULT={result['result']}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for image in (
        "omnipicker_overview.png",
        "omnipicker_front.png",
        "omnipicker_side.png",
        "right_omnipicker_closeup.png",
    ):
        shutil.copy2(STATIC_PACKET / image, out / image)
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
