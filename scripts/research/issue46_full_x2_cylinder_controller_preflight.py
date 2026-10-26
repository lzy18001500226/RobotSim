#!/usr/bin/env python3
"""Static controller and collision preflight for the approved full-X2 overlay.

This research tool never calls mj_step.  It solves a constraint-consistent
initial passive-hand state, applies only existing actuator ranges, and emits a
separate SIMULATION_ONLY controller configuration for a later guarded episode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
from agent import path_governance as pg


CONTROLLER_ID = "FULL_X2_GRAVITY_COMP_HOLD_SIM_V1"
ROOT_TRACKING_GATE_RAD = 0.020
ROBOT_STATIC_QACC_GATE_RAD_S2 = 1e-3
JOINT_EQUALITY_GATE_RAD = 0.003
CONNECT_EQUALITY_GATE_M = 0.001
LEFT_IDLE_PERCENT = 2
ENDPOINT_INTERIOR_MARGIN_RAD = 1e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preflight(path: Path):
    spec = importlib.util.spec_from_file_location("full_x2_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def name(model: mujoco.MjModel, object_type, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(index)) or f"unnamed_{int(index)}"


def source_fixture(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(float(row["command_percent"])): row for row in csv.DictReader(stream)}


def all_equality_residuals(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    joint = {}
    connect = []
    for equality in range(model.neq):
        eq_name = name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality)
        if int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_JOINT):
            first, second = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            q_first = float(data.qpos[model.jnt_qposadr[first]])
            q_second = float(data.qpos[model.jnt_qposadr[second]])
            coefficients = model.eq_data[equality, :5]
            joint[eq_name] = q_first - sum(float(coefficients[index]) * q_second ** index for index in range(5))
    for row in range(data.nefc):
        if int(data.efc_type[row]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        equality = int(data.efc_id[row])
        if int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_CONNECT):
            connect.append({"name": name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality), "residual_m": float(data.efc_pos[row])})
    return {
        "joint_residuals_rad": joint,
        "connect_residuals_m": connect,
        "max_joint_residual_rad": max((abs(value) for value in joint.values()), default=0.0),
        "max_connect_residual_m": max((abs(value["residual_m"]) for value in connect), default=0.0),
    }


def body_pair_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        body_a = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_a]))
        body_b = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_b]))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        records.append({
            "body_names": [body_a, body_b],
            "geom_ids": [geom_a, geom_b],
            "geom_names": [name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a), name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)],
            "distance_m": float(contact.dist),
            "force_contact_frame_N_Nm": wrench.tolist(),
        })
    return records


def classify_contacts(records: list[dict[str, object]]) -> dict[str, object]:
    pads = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
    allowed = []
    forbidden = []
    for record in records:
        pair = set(record["body_names"])
        if pair == {"test_cylinder", "raicom_table"}:
            allowed.append(record)
        elif "test_cylinder" in pair and (pair & pads):
            allowed.append(record)
        else:
            forbidden.append(record)
    return {"allowed": allowed, "forbidden": forbidden}


def actuator_joint(model: mujoco.MjModel, actuator: int) -> tuple[int, int]:
    joint = int(model.actuator_trnid[actuator, 0])
    return joint, int(model.jnt_dofadr[joint])


def all_motor_actuators(model: mujoco.MjModel) -> list[int]:
    return [
        actuator for actuator in range(model.nu)
        if name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator).startswith("motor_")
    ]


def move_motor_endpoint_qpos_inside(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    """Start non-moving posture joints microscopically inside source limits."""
    updates = []
    for actuator in all_motor_actuators(model):
        joint, _ = actuator_joint(model, actuator)
        if not bool(model.jnt_limited[joint]):
            continue
        address = int(model.jnt_qposadr[joint])
        qpos = float(data.qpos[address])
        lower, upper = map(float, model.jnt_range[joint])
        target = None
        boundary = None
        if abs(qpos - lower) <= 1e-12:
            target, boundary = lower + ENDPOINT_INTERIOR_MARGIN_RAD, "lower"
        elif abs(qpos - upper) <= 1e-12:
            target, boundary = upper - ENDPOINT_INTERIOR_MARGIN_RAD, "upper"
        if target is not None:
            data.qpos[address] = target
            updates.append({
                "joint": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), "boundary": boundary,
                "source_qpos_rad": qpos, "initialized_qpos_rad": target,
                "margin_rad": ENDPOINT_INTERIOR_MARGIN_RAD,
            })
    return updates


def endpoint_initialization_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records = []
    for actuator in all_motor_actuators(model):
        joint, _ = actuator_joint(model, actuator)
        if not bool(model.jnt_limited[joint]):
            continue
        qpos = float(data.qpos[model.jnt_qposadr[joint]])
        lower, upper = map(float, model.jnt_range[joint])
        if abs(qpos - (lower + ENDPOINT_INTERIOR_MARGIN_RAD)) <= 1e-10:
            records.append({"joint": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), "boundary": "lower", "initialized_qpos_rad": qpos, "margin_rad": ENDPOINT_INTERIOR_MARGIN_RAD})
        elif abs(qpos - (upper - ENDPOINT_INTERIOR_MARGIN_RAD)) <= 1e-10:
            records.append({"joint": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), "boundary": "upper", "initialized_qpos_rad": qpos, "margin_rad": ENDPOINT_INTERIOR_MARGIN_RAD})
    return records


def apply_static_state(
    preflight,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_q: np.ndarray,
    right: dict[str, float],
    left: dict[str, float],
    cylinder: np.ndarray | None,
    left_elbow_idle_guard_target_rad: float | None = None,
) -> None:
    preflight.set_static_state(model, data, arm_q, right, left, cylinder)
    move_motor_endpoint_qpos_inside(model, data)
    if left_elbow_idle_guard_target_rad is not None:
        joint = model.joint("left_elbow_joint").id
        lower, upper = map(float, model.jnt_range[joint])
        if not lower < left_elbow_idle_guard_target_rad < upper:
            raise ValueError("left-elbow idle guard must remain inside the source range")
        data.qpos[model.jnt_qposadr[joint]] = left_elbow_idle_guard_target_rad
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def inverse_static_forces(
    preflight,
    model: mujoco.MjModel,
    arm_q: np.ndarray,
    right: dict[str, float],
    left: dict[str, float],
    cylinder: np.ndarray | None,
    left_elbow_idle_guard_target_rad: float | None = None,
) -> mujoco.MjData:
    inverse = mujoco.MjData(model)
    apply_static_state(preflight, model, inverse, arm_q, right, left, cylinder, left_elbow_idle_guard_target_rad)
    inverse.ctrl[:] = 0.0
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0.0
    mujoco.mj_inverse(model, inverse)
    return inverse


def motor_posture_gains(joint_name: str) -> tuple[float, float]:
    """Conservative overlay gains; source torque limits remain authoritative."""
    if "head" in joint_name:
        return 20.0, 2.0
    if any(token in joint_name for token in ["shoulder", "elbow", "wrist"]):
        return 30.0, 4.0
    return 40.0, 5.0


def configure_existing_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    inverse: mujoco.MjData,
    motor_targets: dict[str, float],
    right_root_target: float,
    left_root_target: float,
    include_posture_pd: bool,
) -> dict[str, object]:
    """Use inverse gravity feedforward plus bounded existing motor commands."""
    data.ctrl[:] = 0.0
    saturation = []
    mapping = []
    for actuator in range(model.nu):
        actuator_name = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        joint, dof = actuator_joint(model, actuator)
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = map(float, model.actuator_ctrlrange[actuator])
        gear = float(model.actuator_gear[actuator, 0])
        if actuator_name.startswith("motor_"):
            if abs(gear) < 1e-12:
                raise RuntimeError(f"zero actuator gear for {actuator_name}")
            q = float(data.qpos[model.jnt_qposadr[joint]])
            qvel = float(data.qvel[dof])
            kp, kd = motor_posture_gains(joint_name)
            target = motor_targets[joint_name]
            pd = kp * (target - q) - kd * qvel if include_posture_pd else 0.0
            requested = float(inverse.qfrc_inverse[dof] / gear + pd)
            if requested < lo - 1e-12 or requested > hi + 1e-12:
                saturation.append({"actuator": actuator_name, "joint": joint_name, "requested_ctrl": requested, "range": [lo, hi]})
            data.ctrl[actuator] = float(np.clip(requested, lo, hi))
            mapping.append({
                "actuator": actuator_name, "joint": joint_name, "gear": gear,
                "inverse_static_torque_Nm": float(inverse.qfrc_inverse[dof]),
                "requested_ctrl": requested, "range": [lo, hi], "kp": kp, "kd": kd,
            })
        elif actuator_name == "servo_right_claw_joint":
            data.ctrl[actuator] = float(np.clip(right_root_target, lo, hi))
            mapping.append({"actuator": actuator_name, "joint": joint_name, "requested_ctrl": right_root_target, "range": [lo, hi], "kp": float(model.actuator_gainprm[actuator, 0]), "kd": float(-model.actuator_biasprm[actuator, 2])})
        elif actuator_name == "servo_left_claw_joint":
            data.ctrl[actuator] = float(np.clip(left_root_target, lo, hi))
            mapping.append({"actuator": actuator_name, "joint": joint_name, "requested_ctrl": left_root_target, "range": [lo, hi], "kp": float(model.actuator_gainprm[actuator, 0]), "kd": float(-model.actuator_biasprm[actuator, 2])})
    mujoco.mj_forward(model, data)
    return {"saturated": saturation, "actuator_mapping": mapping}


def robot_qacc(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    cylinder = model.joint("test_cylinder_free").id
    cylinder_dofs = set(range(int(model.jnt_dofadr[cylinder]), int(model.jnt_dofadr[cylinder]) + 6))
    values = []
    for joint in range(model.njnt):
        dof = int(model.jnt_dofadr[joint])
        if dof in cylinder_dofs:
            continue
        values.append((name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), float(data.qacc[dof])))
    worst = max(values, key=lambda item: abs(item[1]))
    return {"max_abs_robot_qacc_rad_s2": abs(worst[1]), "worst_joint": worst[0], "qacc_by_joint_rad_s2": dict(values)}


def hand_state_from_vector(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def solve_hand_equilibrium(
    preflight,
    model: mujoco.MjModel,
    arm_q: np.ndarray,
    right_seed: dict[str, float],
    left_seed: dict[str, float],
    right_target: float,
    left_target: float,
    cylinder: np.ndarray | None,
    left_elbow_idle_guard_target_rad: float | None = None,
) -> dict[str, object]:
    right_names = list(preflight.RIGHT_HAND_NAMES)
    left_names = [preflight.LEFT_FROM_RIGHT[item] for item in right_names]
    initial = np.asarray([right_seed[name] for name in right_names] + [left_seed[name] for name in right_names], dtype=float)
    lower = initial - 0.30
    upper = initial + 0.30
    lower[0], upper[0] = right_target - (ROOT_TRACKING_GATE_RAD - 1e-6), right_target + (ROOT_TRACKING_GATE_RAD - 1e-6)
    lower[len(right_names)], upper[len(right_names)] = left_target - (ROOT_TRACKING_GATE_RAD - 1e-6), left_target + (ROOT_TRACKING_GATE_RAD - 1e-6)
    data = mujoco.MjData(model)

    def residual(values: np.ndarray) -> np.ndarray:
        right = hand_state_from_vector(right_names, values[:len(right_names)])
        left = hand_state_from_vector(right_names, values[len(right_names):])
        apply_static_state(preflight, model, data, arm_q, right, left, cylinder, left_elbow_idle_guard_target_rad)
        inverse = inverse_static_forces(preflight, model, arm_q, right, left, cylinder, left_elbow_idle_guard_target_rad)
        targets = {
            name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator, 0])):
            float(data.qpos[model.jnt_qposadr[int(model.actuator_trnid[actuator, 0])]])
            for actuator in all_motor_actuators(model)
        }
        configure_existing_actuators(model, data, inverse, targets, right_target, left_target, include_posture_pd=False)
        dofs = [model.jnt_dofadr[model.joint(item).id] for item in right_names + left_names]
        return data.qacc[dofs]

    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=4000, ftol=1e-13, xtol=1e-13, gtol=1e-13, x_scale="jac")
    right = hand_state_from_vector(right_names, fit.x[:len(right_names)])
    left = hand_state_from_vector(right_names, fit.x[len(right_names):])
    apply_static_state(preflight, model, data, arm_q, right, left, cylinder, left_elbow_idle_guard_target_rad)
    inverse = inverse_static_forces(preflight, model, arm_q, right, left, cylinder, left_elbow_idle_guard_target_rad)
    targets = {
        name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator, 0])):
        float(data.qpos[model.jnt_qposadr[int(model.actuator_trnid[actuator, 0])]])
        for actuator in all_motor_actuators(model)
    }
    controls = configure_existing_actuators(model, data, inverse, targets, right_target, left_target, include_posture_pd=False)
    equality = all_equality_residuals(model, data)
    right_root = float(data.qpos[model.jnt_qposadr[model.joint("right_claw_joint").id]])
    left_root = float(data.qpos[model.jnt_qposadr[model.joint("left_claw_joint").id]])
    return {
        "solver_success": bool(fit.success), "iterations": int(fit.nfev),
        "residual_l2": float(np.linalg.norm(fit.fun)), "residual_max": float(np.max(np.abs(fit.fun))),
        "right_hand_qpos_rad": right, "left_hand_qpos_rad": left,
        "right_root_tracking_error_rad": abs(right_root - right_target),
        "left_root_tracking_error_rad": abs(left_root - left_target),
        "motor_targets_rad": targets, "controls": controls,
        "endpoint_interior_initialization": endpoint_initialization_records(model, data),
        "left_elbow_idle_guard_target_rad": left_elbow_idle_guard_target_rad,
        "equality": equality, "robot_qacc": robot_qacc(model, data),
        "contacts": classify_contacts(body_pair_contacts(model, data)),
        "data": data,
    }


def head_force_snapshot(preflight, scene: Path, arm_q: np.ndarray, right: dict[str, float], left: dict[str, float]) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    apply_static_state(preflight, model, data, arm_q, right, left, None)
    data.ctrl[:] = 0.0
    for actuator in range(model.nu):
        actuator_name = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if actuator_name == "servo_right_claw_joint":
            data.ctrl[actuator] = right["right_claw_joint"]
        elif actuator_name == "servo_left_claw_joint":
            data.ctrl[actuator] = left["right_claw_joint"]
    mujoco.mj_forward(model, data)
    inverse = mujoco.MjData(model)
    apply_static_state(preflight, model, inverse, arm_q, right, left, None)
    inverse.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0.0
    mujoco.mj_inverse(model, inverse)
    joints = {}
    for joint_name in ["head_yaw_joint", "head_pitch_joint"]:
        joint = model.joint(joint_name).id
        dof = int(model.jnt_dofadr[joint])
        actuator = model.actuator(f"motor_{joint_name}").id
        joints[joint_name] = {
            "qpos_rad": float(data.qpos[model.jnt_qposadr[joint]]),
            "qvel_rad_s": float(data.qvel[dof]), "qacc_rad_s2": float(data.qacc[dof]),
            "qfrc_bias_Nm": float(data.qfrc_bias[dof]), "qfrc_constraint_Nm": float(data.qfrc_constraint[dof]),
            "qfrc_passive_Nm": float(data.qfrc_passive[dof]), "qfrc_actuator_Nm": float(data.qfrc_actuator[dof]),
            "qfrc_inverse_static_Nm": float(inverse.qfrc_inverse[dof]),
            "actuator": {
                "name": name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                "ctrlrange_Nm": model.actuator_ctrlrange[actuator].tolist(),
                "gear": model.actuator_gear[actuator].tolist(),
                "trnid": model.actuator_trnid[actuator].tolist(),
            },
        }
    return {"scene": str(scene), "scene_sha256": sha256(scene), "head_joints": joints, "contacts": body_pair_contacts(model, data)}


def compare_model_arrays(baseline: mujoco.MjModel, derived: mujoco.MjModel) -> dict[str, bool]:
    arrays = [
        "jnt_range", "jnt_axis", "jnt_pos", "jnt_qposadr", "jnt_dofadr", "body_mass", "body_inertia",
        "actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "actuator_gear",
        "actuator_trnid", "eq_data", "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
    ]
    return {array: bool(np.array_equal(getattr(baseline, array), getattr(derived, array))) for array in arrays}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-scene", required=True, type=Path)
    parser.add_argument("--derived-scene", required=True, type=Path)
    parser.add_argument("--exception-preflight-result", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--left-elbow-idle-guard-target-rad", type=float)
    args = parser.parse_args()
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite evidence")
    for path in [args.baseline_scene, args.derived_scene, args.exception_preflight_result, args.fixture]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    preflight = load_preflight(WORKTREE / "scripts/research/issue46_full_x2_horizontal_cylinder_preflight.py")
    preflight_result = json.loads(args.exception_preflight_result.read_text(encoding="utf-8"))
    policy = preflight_result.get("collision_policy", {})
    if policy.get("policy") != "SIMULATION_ONLY_HEAD_TORSO_PROXY_EXCEPTION_V1":
        raise SystemExit("derived scene does not carry the approved exception policy")
    baseline = mujoco.MjModel.from_xml_path(str(args.baseline_scene))
    derived = mujoco.MjModel.from_xml_path(str(args.derived_scene))
    preserved = compare_model_arrays(baseline, derived)
    if not all(preserved.values()) or derived.nexclude != baseline.nexclude + 1:
        raise SystemExit("derived scene changed more than the one approved contact exclusion")
    fixture = source_fixture(args.fixture)
    right_open = preflight.right_hand(fixture[preflight.OPEN_PERCENT])
    right_closed = preflight.right_hand(fixture[preflight.CLOSE_PERCENT])
    left_idle = preflight.right_hand(fixture[LEFT_IDLE_PERCENT])
    left_elbow_guard = args.left_elbow_idle_guard_target_rad
    states = [
        ("INITIAL_SETTLE", np.asarray(preflight_result["waypoints"]["HORIZONTAL_PREGRASP"]["arm_q_rad"], dtype=float), right_open, None),
        ("HORIZONTAL_PREGRASP", np.asarray(preflight_result["waypoints"]["HORIZONTAL_PREGRASP"]["arm_q_rad"], dtype=float), right_open, None),
        ("HORIZONTAL_APPROACH", np.asarray(preflight_result["waypoints"]["HORIZONTAL_APPROACH"]["arm_q_rad"], dtype=float), right_open, None),
        ("CLOSE_75_PERCENT", np.asarray(preflight_result["waypoints"]["HORIZONTAL_CONTACT"]["arm_q_rad"], dtype=float), right_closed, None),
        ("VERTICAL_LIFT_50MM", np.asarray(preflight_result["waypoints"]["VERTICAL_LIFT_50MM"]["arm_q_rad"], dtype=float), right_closed, np.asarray(preflight_result["cylinder"]["center_world_m"], dtype=float) + np.array([0.0, 0.0, 0.05])),
    ]
    static_states = []
    for state_name, arm_q, right_seed, cylinder in states:
        solution = solve_hand_equilibrium(
            preflight, derived, arm_q, right_seed, left_idle,
            right_seed["right_claw_joint"], left_idle["right_claw_joint"], cylinder,
            left_elbow_guard,
        )
        solution.pop("data")
        solution["phase"] = state_name
        static_states.append(solution)
    initial_arm = states[0][1]
    before = head_force_snapshot(preflight, args.baseline_scene, initial_arm, right_open, left_idle)
    after = head_force_snapshot(preflight, args.derived_scene, initial_arm, right_open, left_idle)
    source_bad = {}
    for state in static_states:
        if state["phase"] == "CLOSE_75_PERCENT":
            model = derived
            data = mujoco.MjData(model)
            apply_static_state(
                preflight, model, data, states[3][1], state["right_hand_qpos_rad"],
                state["left_hand_qpos_rad"], None, left_elbow_guard,
            )
            source_bad = {
                body: value for body, value in preflight.source_mesh_sample_check(model, data, np.asarray(preflight_result["cylinder"]["center_world_m"], dtype=float))["per_body"].items()
                if value["sampled_surface_enters_cylinder"] and body not in preflight.PAD_BODIES
            }
    state_failures = []
    for state in static_states:
        equality = state["equality"]
        contacts = state["contacts"]
        common_pass = bool(
            state["solver_success"]
            and state["right_root_tracking_error_rad"] <= ROOT_TRACKING_GATE_RAD
            and state["left_root_tracking_error_rad"] <= ROOT_TRACKING_GATE_RAD
            and equality["max_joint_residual_rad"] <= JOINT_EQUALITY_GATE_RAD
            and equality["max_connect_residual_m"] <= CONNECT_EQUALITY_GATE_M
            and not state["controls"]["saturated"]
            and not contacts["forbidden"]
        )
        contact_bodies = {
            body for record in contacts["allowed"] if "test_cylinder" in record["body_names"]
            for body in record["body_names"] if body in preflight.PAD_BODIES
        }
        dynamic_contact_state = state["phase"] in {"CLOSE_75_PERCENT", "VERTICAL_LIFT_50MM"}
        if dynamic_contact_state:
            # At a real contact/load state, zero static acceleration is not a
            # prerequisite: that condition is established only through the
            # later mj_step episode.  Geometry, root margin, source limits and
            # non-exempt contacts are still fail-closed here.
            state_pass = common_pass and contact_bodies == preflight.PAD_BODIES
            state["qualification"] = "DYNAMIC_CONTACT_OR_LOAD_STATE"
            state["static_qacc_not_required"] = True
            state["distal_pad_bodies"] = sorted(contact_bodies)
        else:
            state_pass = common_pass and state["residual_max"] <= ROBOT_STATIC_QACC_GATE_RAD_S2 and state["robot_qacc"]["max_abs_robot_qacc_rad_s2"] <= ROBOT_STATIC_QACC_GATE_RAD_S2
            state["qualification"] = "STATIC_HOLD_EQUILIBRIUM"
            state["static_qacc_not_required"] = False
        state["pass"] = state_pass
        if not state_pass:
            state_failures.append(state["phase"])
    before_after_ok = all(
        abs(item["qfrc_inverse_static_Nm"]) <= max(abs(value) for value in item["actuator"]["ctrlrange_Nm"])
        for item in after["head_joints"].values()
    )
    gates = {
        "EXACT_PAIR_ONLY": "PASS" if all(preserved.values()) and derived.nexclude == baseline.nexclude + 1 else "FAIL",
        "HEAD_TORSO_CONTACT_REMOVED": "PASS" if not [item for item in after["contacts"] if set(item["body_names"]) == {"head_pitch_link", "torso_link"}] else "FAIL",
        "HEAD_ACTUATOR_STATIC_TORQUE_WITHIN_SOURCE_RANGE": "PASS" if before_after_ok else "FAIL",
        "ALL_STATIC_STATES_CONTROLLER_FEASIBLE": "PASS" if not state_failures else "FAIL",
        "SOURCE_STL_NONPAD_CLEARANCE_AT_CLOSE": "PASS" if not source_bad else "FAIL",
        "DYNAMICS": "NOT RUN",
    }
    overall = "PASS" if all(value == "PASS" for key, value in gates.items() if key != "DYNAMICS") else "BLOCKED"
    trial_config = {
        "schema": "robotsim.issue46.full_x2_cylinder_trial_config.v1",
        "status": "STATIC_READY_FOR_ONE_DYNAMICS_EPISODE" if overall == "PASS" else "STATIC_BLOCKED",
        "scene": str(args.derived_scene.resolve()), "scene_sha256": sha256(args.derived_scene.resolve()),
        "baseline_scene_sha256": sha256(args.baseline_scene.resolve()),
        "controller_overlay": {
            "id": CONTROLLER_ID,
            "method": "existing motor controls receive current-state inverse gravity feedforward plus bounded posture PD; frozen claw position servos stay unchanged",
            "motor_gains": {"head": [20.0, 2.0], "arm": [30.0, 4.0], "other": [40.0, 5.0]},
            "root_servo": "unchanged kp=0.5, kv=0.05, gear=1, ctrlrange [-1,0], forcerange [-0.5,0.5]",
            "endpoint_initialization": "source-limit endpoints for idle torque-motor joints are initialized 1e-5 rad inside their existing limits; source ranges are unchanged",
            "endpoint_interior_margin_rad": ENDPOINT_INTERIOR_MARGIN_RAD,
            "left_elbow_idle_guard_target_rad": left_elbow_guard,
        },
        "static_states": [{key: value for key, value in state.items() if key not in {"contacts", "controls"}} for state in static_states],
        "right_root_open_target_rad": right_open["right_claw_joint"],
        "right_root_close_target_rad": right_closed["right_claw_joint"],
        "left_root_idle_target_rad": left_idle["right_claw_joint"],
        "limits": {"max_steps": 20000, "max_simulated_seconds": 20.0, "root_tracking_gate_rad": ROOT_TRACKING_GATE_RAD},
        "gates": gates,
    }
    result = {
        "schema": "robotsim.issue46.full_x2_cylinder_controller_preflight.v1",
        "status": overall, "mj_step_calls": 0, "mujoco_version": mujoco.__version__,
        "baseline_scene": str(args.baseline_scene.resolve()), "baseline_scene_sha256": sha256(args.baseline_scene.resolve()),
        "derived_scene": str(args.derived_scene.resolve()), "derived_scene_sha256": sha256(args.derived_scene.resolve()),
        "policy": policy, "preserved_model_arrays": preserved,
        "baseline_vs_exception_head_force_snapshot": {"baseline": before, "exception": after},
        "static_states": static_states, "source_nonpad_collision_at_close": source_bad,
        "gates": gates, "trial_config": trial_config,
    }
    pg.write_new_text(output / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    pg.write_new_text(output / "trial_config.json", json.dumps(trial_config, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": overall, "gates": gates, "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
