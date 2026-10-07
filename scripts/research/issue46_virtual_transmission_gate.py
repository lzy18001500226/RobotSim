from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_x2_grasp as task  # noqa: E402
from issue46_virtual_transmission import VirtualTransmissionController  # noqa: E402

OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-virtual-transmission"))
ABORT_RAD = 0.010
# Simulation/control target margin derived from the measured one-step transient.
SOFT_LIMIT_MARGIN_RAD = 0.0015
LEFT_ARM = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.30,
    "left_elbow_joint": -0.15,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
}
RIGHT_ARM = {
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": -0.30,
    "right_elbow_joint": -0.35,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}
LEFT_HAND = {
    "L_thumb_abad_joint": 0.35,
    "L_index_abad_joint": 0.10,
    "L_ring_abad_joint": -0.10,
    "L_pinky_abad_joint": -0.16,
}
RIGHT_HAND = {
    "R_thumb_abad_joint": -0.72,
    "R_index_abad_joint": -0.10,
    "R_ring_abad_joint": 0.10,
    "R_pinky_abad_joint": 0.15,
}
INITIAL_POSE = {**LEFT_ARM, **RIGHT_ARM, **LEFT_HAND, **RIGHT_HAND}
IGNORED_GEOMS = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap", "floor", "m0_table_top"}
PROBE_GEOM = "virtual_transmission_fixed_contact_probe"


def source_map(model: mujoco.MjModel, details: dict[str, object]) -> tuple[dict[str, ET.Element], dict[str, dict[str, float | int]]]:
    root = ET.parse(task.URDF).getroot()
    nodes = {node.get("name"): node for node in root.findall("joint") if node.get("name")}
    joints: dict[str, dict[str, float | int]] = {}
    for joint_name, node in nodes.items():
        if node.get("type") != "revolute":
            continue
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        axis_node = node.find("axis")
        source_axis = np.fromstring("1 0 0" if axis_node is None else axis_node.get("xyz", "1 0 0"), sep=" ")
        compiled_axis = np.asarray(model.jnt_axis[jid], dtype=float)
        cosine = float(np.dot(source_axis, compiled_axis) / (np.linalg.norm(source_axis) * np.linalg.norm(compiled_axis)))
        if abs(abs(cosine) - 1.0) > 1e-6:
            raise ValueError(f"unresolved source/MuJoCo axis mapping for {joint_name}: dot={cosine}")
        limit = node.find("limit")
        joints[joint_name] = {
            "joint_id": jid,
            "qpos": int(model.jnt_qposadr[jid]),
            "dof": int(model.jnt_dofadr[jid]),
            "axis_sign": 1 if cosine >= 0.0 else -1,
            "reference": float(details["joint_refs"].get(joint_name, 0.0)),
            "lower": float(limit.get("lower")),
            "upper": float(limit.get("upper")),
            "velocity_limit": float(limit.get("velocity")),
            "effort_limit": float(limit.get("effort")),
        }
    return nodes, joints


def source_position(model: mujoco.MjModel, data: mujoco.MjData, joint: dict[str, float | int], raw: float | None = None) -> float:
    q = int(joint["qpos"])
    value = float(data.qpos[q] if raw is None else raw)
    return float(joint["reference"]) + int(joint["axis_sign"]) * (value - float(model.qpos0[q]))


def source_velocity(data: mujoco.MjData, joint: dict[str, float | int]) -> float:
    return int(joint["axis_sign"]) * float(data.qvel[int(joint["dof"])])


def set_initial_source_position(model: mujoco.MjModel, data: mujoco.MjData, joint: dict[str, float | int], value: float) -> None:
    q = int(joint["qpos"])
    data.qpos[q] = float(model.qpos0[q]) + int(joint["axis_sign"]) * (value - float(joint["reference"]))


def name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or f"unnamed_{idx}"


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    rows = []
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        n1, n2 = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        b1 = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1]))
        b2 = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2]))
        if n1 in IGNORED_GEOMS or n2 in IGNORED_GEOMS:
            continue
        if (b1 == "world" or b2 == "world") and PROBE_GEOM not in (n1, n2):
            continue
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, ci, wrench)
        rows.append({
            "geom_pair": [n1, n2],
            "body_pair": [b1, b2],
            "distance_m": float(contact.dist),
            "penetration_m": float(max(0.0, -contact.dist)),
            "force_components": wrench.tolist(),
        })
    return rows


def make_runtime(fixed_probe: bool = False) -> dict[str, object]:
    model, details = task.build_model(virtual_transmission=True, fixed_contact_probe=fixed_probe)
    data = details["data"]
    mujoco.mj_resetData(model, data)
    nodes, joints = source_map(model, details)
    follower_names = {str(relation["follower_joint"]) for relation in details["mimic_relations"]}
    if follower_names & set(INITIAL_POSE):
        raise AssertionError("initial pose may not assign follower coordinates")
    for joint_name, coordinate in INITIAL_POSE.items():
        set_initial_source_position(model, data, joints[joint_name], coordinate)
    actuator_by_joint = {int(model.actuator_trnid[aid, 0]): aid for aid in range(model.nu)}
    initial_state_active = {
        joint_name: source_position(model, data, joints[joint_name])
        for joint_name in details["active_hand_joint_names"]
    }
    initial_active = {
        joint_name: min(
            float(joints[joint_name]["upper"]) - SOFT_LIMIT_MARGIN_RAD,
            max(float(joints[joint_name]["lower"]) + SOFT_LIMIT_MARGIN_RAD, coordinate),
        )
        for joint_name, coordinate in initial_state_active.items()
    }
    initialization_qpos_write_count = len(INITIAL_POSE)
    for joint_name, target in initial_active.items():
        set_initial_source_position(model, data, joints[joint_name], target)
        initialization_qpos_write_count += 1
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        target = float(relation["multiplier"]) * initial_active[driver] + float(relation["offset"])
        low = float(joints[follower]["lower"]) + SOFT_LIMIT_MARGIN_RAD
        high = float(joints[follower]["upper"]) - SOFT_LIMIT_MARGIN_RAD
        if not low <= target <= high:
            raise ValueError(f"operational OPEN follower target outside soft range for {follower}: {target} not in [{low}, {high}]")
        set_initial_source_position(model, data, joints[follower], target)
        initialization_qpos_write_count += 1
    data.qvel[:] = 0.0
    hand_joint_names = set(details["active_hand_joint_names"]) | follower_names
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        data.ctrl[aid] = 0.0 if joint_name in hand_joint_names else float(data.qpos[int(model.jnt_qposadr[jid])])
    for geom_name in IGNORED_GEOMS - {"floor", "m0_table_top"}:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
    mujoco.mj_forward(model, data)

    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    controller_profile: dict[str, dict[str, float]] = {}
    for joint_name in sorted(hand_joint_names):
        joint = joints[joint_name]
        aid = actuator_by_joint[int(joint["joint_id"])]
        dof = int(joint["dof"])
        is_follower = joint_name in follower_names
        torque_cap = task.VIRTUAL_FOLLOWER_MAX_TORQUE_NM if is_follower else task.VIRTUAL_ACTIVE_MAX_TORQUE_NM
        bound = min(float(torque_cap), float(joint["effort_limit"]))
        bias = float(data.qfrc_bias[dof])
        pd_headroom = bound - abs(bias)
        if pd_headroom <= 0.0:
            raise ValueError(f"OPEN bias exhausts bounded actuator authority for {joint_name}: |bias|={abs(bias)} Nm, bound={bound} Nm")
        inverse_column = np.linalg.solve(mass, np.eye(model.nv, dtype=float)[:, dof])
        effective_inertia = 1.0 / float(inverse_column[dof])
        kp = pd_headroom / SOFT_LIMIT_MARGIN_RAD
        damping_ratio = 1.0
        kv = 2.0 * damping_ratio * float(np.sqrt(effective_inertia * kp))
        controller_profile[joint_name] = {
            "source_effort_limit_nm": float(joint["effort_limit"]),
            "simulation_effort_bound_nm": float(torque_cap),
            "effective_effort_bound_nm": bound,
            "open_qfrc_bias_nm": bias,
            "pd_torque_headroom_nm": pd_headroom,
            "operational_error_budget_rad": SOFT_LIMIT_MARGIN_RAD,
            "kp_nm_per_rad": kp,
            "effective_joint_inertia_kg_m2_equiv": effective_inertia,
            "damping_ratio": damping_ratio,
            "kv_nms_per_rad": kv,
        }

    transmission = VirtualTransmissionController(
        model,
        data,
        nodes,
        details["joint_refs"],
        details["mimic_relations"],
        actuator_by_joint,
        details["active_hand_joint_names"],
        kp_nm_per_rad=task.VIRTUAL_FOLLOWER_SERVO_KP_NM_PER_RAD,
        kv_nms_per_rad=task.VIRTUAL_FOLLOWER_SERVO_KV_NMS_PER_RAD,
        max_torque_nm=task.VIRTUAL_FOLLOWER_MAX_TORQUE_NM,
        soft_limit_margin_rad=SOFT_LIMIT_MARGIN_RAD,
        kp_by_joint={joint: row["kp_nm_per_rad"] for joint, row in controller_profile.items()},
        kv_by_joint={joint: row["kv_nms_per_rad"] for joint, row in controller_profile.items()},
        bias_feedforward=True,
    )
    transmission.command_active_sources(initial_active)
    transmission.write_internal_controls()
    mujoco.mj_forward(model, data)
    details["virtual_transmission_controller_profile"] = controller_profile
    for follower, spec in details["virtual_follower_actuator_specs"].items():
        spec["kp_nm_per_rad"] = controller_profile[follower]["kp_nm_per_rad"]
        spec["kv_nms_per_rad"] = controller_profile[follower]["kv_nms_per_rad"]
    return {
        "model": model,
        "data": data,
        "details": details,
        "joints": joints,
        "transmission": transmission,
        "actuator_by_joint": actuator_by_joint,
        "initial_active": initial_active,
        "initial_state_active": initial_state_active,
        "controller_profile": controller_profile,
        "initialization_qpos_write_count": initialization_qpos_write_count,
        "initial_follower_qpos": {
            f: float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f)])])
            for f in follower_names
        },
    }


def diagnostics(runtime: dict[str, object]) -> dict[str, object]:
    model = runtime["model"]
    data = runtime["data"]
    joints = runtime["joints"]
    details = runtime["details"]
    transmission = runtime["transmission"]
    actuator_by_joint = runtime["actuator_by_joint"]
    limits = []
    for joint_name, joint in joints.items():
        position = source_position(model, data, joint)
        velocity = source_velocity(data, joint)
        low, high = float(joint["lower"]), float(joint["upper"])
        speed_limit = float(joint["velocity_limit"])
        limits.append({
            "joint": joint_name,
            "position_rad": position,
            "lower_rad": low,
            "upper_rad": high,
            "lower_margin_rad": position - low,
            "upper_margin_rad": high - position,
            "velocity_rad_s": velocity,
            "velocity_limit_rad_s": speed_limit,
            "within_position_limits": low - 1e-9 <= position <= high + 1e-9,
            "within_velocity_limit": abs(velocity) <= speed_limit + 1e-9,
        })
    relations = transmission.measure()
    active_states = transmission.active_measure()
    contacts = contact_rows(model, data)
    self_contacts = [
        row for row in contacts
        if row["body_pair"][0] != "world"
        and row["body_pair"][1] != "world"
        and PROBE_GEOM not in row["geom_pair"]
    ]
    efforts = []
    for follower, spec in details["virtual_follower_actuator_specs"].items():
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower))
        aid = actuator_by_joint[jid]
        force = float(data.actuator_force[aid]) * float(model.actuator_gear[aid, 0])
        bound = float(spec["max_torque_nm"])
        efforts.append({"follower_joint": follower, "force_nm": force, "bound_nm": bound, "within_bound": abs(force) <= bound + 1e-10})
    active_efforts = []
    for joint_name, row in active_states.items():
        force = float(row["actual_actuator_force_nm"])
        bound = float(row["effort_limit_nm"])
        active_efforts.append({"active_joint": joint_name, "force_nm": force, "bound_nm": bound, "within_bound": abs(force) <= bound + 1e-10})
    active_target_error = max((abs(float(row["target_tracking_error_rad"])) for row in active_states.values()), default=0.0)
    follower_target_error = max((abs(float(row["target_tracking_error_rad"])) for row in relations.values()), default=0.0)
    target_error = max(active_target_error, follower_target_error)
    relation_error = max((abs(float(row["source_relation_error_rad"])) for row in relations.values()), default=0.0)
    return {
        "time_s": float(data.time),
        "mimic_relations": relations,
        "active_joint_states": active_states,
        "active_target_tracking_error_max_rad": active_target_error,
        "follower_target_tracking_error_max_rad": follower_target_error,
        "target_tracking_error_max_rad": target_error,
        "source_relation_error_max_rad": relation_error,
        "source_joint_limits": limits,
        "position_limit_violations": [row for row in limits if not row["within_position_limits"]],
        "velocity_limit_violations": [row for row in limits if not row["within_velocity_limit"]],
        "contacts": contacts,
        "robot_self_contacts": self_contacts,
        "max_self_penetration_m": max((float(row["penetration_m"]) for row in self_contacts), default=0.0),
        "follower_efforts": efforts,
        "active_efforts": active_efforts,
        "follower_effort_violation": any(not row["within_bound"] for row in efforts),
        "active_effort_violation": any(not row["within_bound"] for row in active_efforts),
        "actuator_effort_violation": any(not row["within_bound"] for row in efforts + active_efforts),
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()),
        "max_abs_qvel_rad_s": float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0,
        "max_abs_qacc_rad_s2": float(np.max(np.abs(data.qacc))) if data.qacc.size else 0.0,
    }


def feasible_driver_targets(runtime: dict[str, object]) -> dict[str, float]:
    joints = runtime["joints"]
    details = runtime["details"]
    transmission = runtime["transmission"]
    relations = details["mimic_relations"]
    bounds: dict[str, list[float]] = {}
    for relation in relations:
        driver = str(relation["driver_joint"])
        follower_name = str(relation["follower_joint"])
        multiplier, offset = float(relation["multiplier"]), float(relation["offset"])
        if multiplier <= 0.0:
            raise ValueError(f"unexpected nonpositive pinned mimic multiplier {multiplier}")
        driver_low, driver_high = transmission.operational_limits(driver)
        follower_low, follower_high = transmission.operational_limits(follower_name)
        interval = bounds.setdefault(driver, [driver_low, driver_high])
        interval[0] = max(interval[0], (follower_low - offset) / multiplier)
        interval[1] = min(interval[1], (follower_high - offset) / multiplier)
    result = {}
    for driver, (low, high) in bounds.items():
        current = float(runtime["initial_active"][driver])
        if low > high or not low - 1e-10 <= current <= high + 1e-10:
            raise ValueError(f"no valid source target interval for {driver}: {low} .. {high}; current={current}")
        target = low + 0.65 * (high - low) if abs(current - low) < abs(current - high) else high - 0.65 * (high - low)
        result[driver] = target
    return result


def smooth_blend(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    s = alpha * alpha * (3.0 - 2.0 * alpha)
    return {key: start[key] + s * (end[key] - start[key]) for key in start}


def arm_target_raw(runtime: dict[str, object], arm_targets: dict[str, float]) -> None:
    model, data, joints, actuator_by_joint = runtime["model"], runtime["data"], runtime["joints"], runtime["actuator_by_joint"]
    for joint_name, coordinate in arm_targets.items():
        joint = joints[joint_name]
        q = int(joint["qpos"])
        raw = float(model.qpos0[q]) + int(joint["axis_sign"]) * (coordinate - float(joint["reference"]))
        data.ctrl[actuator_by_joint[int(joint["joint_id"])]] = raw


def execute_trial(
    label: str,
    phases: list[tuple[str, int, dict[str, float], dict[str, float] | None]],
    trace_file,
    *,
    fixed_probe: bool = False,
) -> dict[str, object]:
    runtime = make_runtime(fixed_probe=fixed_probe)
    model, data, transmission = runtime["model"], runtime["data"], runtime["transmission"]
    initial = diagnostics(runtime)
    if initial["robot_self_contacts"] or initial["max_self_penetration_m"] > 0.0:
        return {"label": label, "passed": False, "first_failed_gate": "initial self-contact/penetration", "initial": initial, "steps": 0}
    if initial["position_limit_violations"] or initial["velocity_limit_violations"]:
        return {"label": label, "passed": False, "first_failed_gate": "initial source joint limits", "initial": initial, "steps": 0}
    if initial["source_relation_error_max_rad"] > 1e-9:
        return {"label": label, "passed": False, "first_failed_gate": "initial source mimic geometry", "initial": initial, "steps": 0}
    if model.neq != 0 or len(runtime["details"]["virtual_follower_actuator_specs"]) != 12:
        return {"label": label, "passed": False, "first_failed_gate": "virtual model topology", "initial": initial, "steps": 0}

    start = runtime["initial_active"]
    summary: dict[str, object] = {
        "label": label,
        "steps": 0,
        "target_tracking_error_max_rad": 0.0,
        "active_target_tracking_error_max_rad": 0.0,
        "follower_target_tracking_error_max_rad": 0.0,
        "source_relation_error_max_rad": 0.0,
        "diagnostic_0p003_exceedances": 0,
        "abort_0p010_exceedances": 0,
        "first_abort": None,
        "position_limit_violation_count": 0,
        "velocity_limit_violation_count": 0,
        "nan_seen": False,
        "max_self_penetration_m": 0.0,
        "max_follower_force_nm": 0.0,
        "max_active_force_nm": 0.0,
        "actuator_force_bound_violation": False,
        "max_abs_qvel_rad_s": 0.0,
        "max_abs_qacc_rad_s2": 0.0,
        "probe_contact_steps": 0,
        "probe_contact_pairs": {},
        "max_probe_normal_force_n": 0.0,
        "follower_positive_work_j": 0.0,
        "follower_signed_work_j": 0.0,
        "active_rollout_follower_qpos_writes": 0,
        "initial": initial,
    }
    step_index = 0
    for phase, duration, destination, arm_goal in phases:
        phase_start_hand = transmission.active_target_sources()
        if arm_goal:
            arm_start = {key: source_position(model, data, runtime["joints"][key]) for key in arm_goal}
        for k in range(duration):
            alpha = (k + 1) / duration
            hand_target = smooth_blend(phase_start_hand, destination, alpha)
            if arm_goal:
                arm_target = smooth_blend(arm_start, arm_goal, alpha)
                arm_target_raw(runtime, arm_target)
            transmission.command_active_sources(hand_target)
            transmission.write_internal_controls()
            mujoco.mj_step(model, data)
            work = transmission.record_step_work()
            step_index += 1
            state = diagnostics(runtime)
            summary["steps"] = step_index
            summary["target_tracking_error_max_rad"] = max(float(summary["target_tracking_error_max_rad"]), float(state["target_tracking_error_max_rad"]))
            summary["active_target_tracking_error_max_rad"] = max(float(summary["active_target_tracking_error_max_rad"]), float(state["active_target_tracking_error_max_rad"]))
            summary["follower_target_tracking_error_max_rad"] = max(float(summary["follower_target_tracking_error_max_rad"]), float(state["follower_target_tracking_error_max_rad"]))
            summary["source_relation_error_max_rad"] = max(float(summary["source_relation_error_max_rad"]), float(state["source_relation_error_max_rad"]))
            if state["target_tracking_error_max_rad"] > 0.003 or state["source_relation_error_max_rad"] > 0.003:
                summary["diagnostic_0p003_exceedances"] += 1
            relation_error = max(float(state["target_tracking_error_max_rad"]), float(state["source_relation_error_max_rad"]))
            if relation_error > ABORT_RAD:
                summary["abort_0p010_exceedances"] += 1
                if summary["first_abort"] is None:
                    offender = max(
                        state["mimic_relations"],
                        key=lambda f: max(
                            abs(float(state["mimic_relations"][f]["target_tracking_error_rad"])),
                            abs(float(state["mimic_relations"][f]["source_relation_error_rad"])),
                        ),
                    )
                    summary["first_abort"] = {"phase": phase, "step": step_index, "time_s": state["time_s"], "relation": offender, "error_rad": relation_error}
            summary["position_limit_violation_count"] += len(state["position_limit_violations"])
            summary["velocity_limit_violation_count"] += len(state["velocity_limit_violations"])
            summary["nan_seen"] = bool(summary["nan_seen"]) or not bool(state["finite"])
            summary["max_self_penetration_m"] = max(float(summary["max_self_penetration_m"]), float(state["max_self_penetration_m"]))
            summary["actuator_force_bound_violation"] = bool(summary["actuator_force_bound_violation"]) or bool(state["actuator_effort_violation"])
            summary["max_follower_force_nm"] = max(float(summary["max_follower_force_nm"]), max((abs(float(row["force_nm"])) for row in state["follower_efforts"]), default=0.0))
            summary["max_active_force_nm"] = max(float(summary["max_active_force_nm"]), max((abs(float(row["force_nm"])) for row in state["active_efforts"]), default=0.0))
            summary["max_abs_qvel_rad_s"] = max(float(summary["max_abs_qvel_rad_s"]), float(state["max_abs_qvel_rad_s"]))
            summary["max_abs_qacc_rad_s2"] = max(float(summary["max_abs_qacc_rad_s2"]), float(state["max_abs_qacc_rad_s2"]))
            summary["follower_positive_work_j"] += float(work["positive_work_step_j"])
            summary["follower_signed_work_j"] += float(work["signed_work_step_j"])
            probe_pairs = [row for row in state["contacts"] if PROBE_GEOM in row["geom_pair"]]
            if probe_pairs:
                summary["probe_contact_steps"] += 1
                for row in probe_pairs:
                    pair_key = " + ".join(sorted(row["body_pair"]))
                    summary["probe_contact_pairs"][pair_key] = int(summary["probe_contact_pairs"].get(pair_key, 0)) + 1
                    force_n = float(np.linalg.norm(np.asarray(row["force_components"][:3])))
                    summary["max_probe_normal_force_n"] = max(float(summary["max_probe_normal_force_n"]), force_n)
            trace_file.write(json.dumps({
                "trial": label,
                "step": step_index,
                "phase": phase,
                "simulation_time_s": float(data.time),
                "mimic_relations": state["mimic_relations"],
                "source_joint_limits": state["source_joint_limits"],
                "source_joint_limit_violations": state["position_limit_violations"],
                "source_joint_velocity_violations": state["velocity_limit_violations"],
                "follower_efforts": state["follower_efforts"],
                "active_efforts": state["active_efforts"],
                "active_joint_states": state["active_joint_states"],
                "contacts": state["contacts"],
                "max_self_penetration_m": state["max_self_penetration_m"],
                "qvel_max_abs_rad_s": state["max_abs_qvel_rad_s"],
                "qacc_max_abs_rad_s2": state["max_abs_qacc_rad_s2"],
                "follower_work": work,
                "active_rollout_follower_qpos_writes": 0,
            }, separators=(",", ":")) + "\n")
            trace_file.flush()
            if not state["finite"] or summary["first_abort"] or summary["position_limit_violation_count"] or summary["velocity_limit_violation_count"] or summary["actuator_force_bound_violation"]:
                break
        if summary["first_abort"] or summary["nan_seen"] or summary["position_limit_violation_count"] or summary["velocity_limit_violation_count"] or summary["actuator_force_bound_violation"]:
            break
    summary["passed"] = (
        summary["first_abort"] is None
        and summary["position_limit_violation_count"] == 0
        and summary["velocity_limit_violation_count"] == 0
        and not summary["nan_seen"]
        and not summary["actuator_force_bound_violation"]
        and float(summary["max_self_penetration_m"]) <= 1e-12
    )
    summary["first_failed_gate"] = None if summary["passed"] else (
        "mimic geometry exceeded 0.010 rad" if summary["first_abort"] else
        "source joint limit/velocity violation" if summary["position_limit_violation_count"] or summary["velocity_limit_violation_count"] else
        "numerical instability" if summary["nan_seen"] else
        "actuator effort bound" if summary["actuator_force_bound_violation"] else
        "self-penetration during trial"
    )
    if fixed_probe:
        summary["stable_external_contact_pass"] = summary["probe_contact_steps"] > 0 and summary["max_probe_normal_force_n"] > 0.0
        summary["passed"] = bool(summary["passed"]) and bool(summary["stable_external_contact_pass"])
        if not summary["stable_external_contact_pass"] and summary["first_failed_gate"] is None:
            summary["first_failed_gate"] = "fixed-object contact was not established"
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--preflight-only" in sys.argv:
        runtime = make_runtime()
        model, data, transmission = runtime["model"], runtime["data"], runtime["transmission"]
        before_qpos = data.qpos.copy()
        open_targets = dict(runtime["initial_active"])
        closed_targets = {**open_targets, **feasible_driver_targets(runtime)}
        phases = {
            "open": open_targets,
            "closed": closed_targets,
            "partial": smooth_blend(open_targets, closed_targets, 0.5),
        }
        target_checks = {}
        for phase_name, targets in phases.items():
            transmission.command_active_sources(targets)
            active = transmission.active_target_sources()
            followers = {}
            for relation in runtime["details"]["mimic_relations"]:
                follower = str(relation["follower_joint"])
                driver = str(relation["driver_joint"])
                target = float(relation["multiplier"]) * active[driver] + float(relation["offset"])
                low, high = transmission.operational_limits(follower)
                followers[follower] = {
                    "target_rad": target,
                    "operational_range_rad": [low, high],
                    "inside": low <= target <= high,
                }
            active_inside = all(
                transmission.operational_limits(joint)[0] <= value <= transmission.operational_limits(joint)[1]
                for joint, value in active.items()
            )
            target_checks[phase_name] = {
                "active_targets_inside": active_inside,
                "follower_targets": followers,
                "all_follower_targets_inside": all(row["inside"] for row in followers.values()),
                "active_targets": active,
            }
        initial = diagnostics(runtime)
        qpos_unchanged = bool(np.array_equal(before_qpos, data.qpos))
        passed = (
            all(row["active_targets_inside"] and row["all_follower_targets_inside"] for row in target_checks.values())
            and qpos_unchanged
            and not initial["robot_self_contacts"]
            and not initial["position_limit_violations"]
            and not initial["velocity_limit_violations"]
            and initial["source_relation_error_max_rad"] <= 1e-9
        )
        print(json.dumps({
            "status": "PASS" if passed else "FAIL",
            "mode": "static target preflight; no mj_step",
            "soft_limit_margin_rad": SOFT_LIMIT_MARGIN_RAD,
            "source_limits_modified": False,
            "qpos_written_or_changed": not qpos_unchanged,
            "initial_state": {
                "time_s": initial["time_s"],
                "self_contact_count": len(initial["robot_self_contacts"]),
                "max_self_penetration_m": initial["max_self_penetration_m"],
                "position_limit_violation_count": len(initial["position_limit_violations"]),
                "velocity_limit_violation_count": len(initial["velocity_limit_violations"]),
                "mimic_error_max_rad": initial["source_relation_error_max_rad"],
            },
            "target_checks": target_checks,
        }, indent=2))
        return 0 if passed else 2

    trace_path = OUT / "virtual_transmission_prevalidation_trace.jsonl"
    gate_results: list[dict[str, object]] = []
    root_runtime = make_runtime()
    initial_state = diagnostics(root_runtime)
    drivers = feasible_driver_targets(root_runtime)
    start = dict(root_runtime["initial_active"])
    closed = {**start, **drivers}
    # Gate execution is strictly staged. The first failure prevents later loads.
    with trace_path.open("w", encoding="utf-8") as trace:
        gate_results.append(execute_trial("open_close_both_hands", [
            ("open_hold", 200, start, None),
            ("close", 600, closed, None),
            ("partial", 400, smooth_blend(start, closed, 0.5), None),
            ("closed_hold", 300, closed, None),
            ("reopen", 600, start, None),
        ], trace))
        if gate_results[-1]["passed"]:
            gate_results.append(execute_trial("gravity_hold", [("gravity_hold", 1500, start, None)], trace))
        if gate_results[-1]["passed"]:
            runtime = make_runtime()
            arm_names = task.ARM
            arm_ids = [int(runtime["joints"][joint]["joint_id"]) for joint in arm_names]
            site_id = int(runtime["details"]["site_id"])
            arm_route = task.ik_plan(
                runtime["model"],
                runtime["data"].qpos.copy(),
                site_id,
                arm_ids,
                [task.APPROACH_PALM_POS.copy(), task.PREGRASP_PALM_POS.copy()],
                task.PALM_TARGET_ROTATION,
            )
            phases = []
            arm_current = {joint: source_position(runtime["model"], runtime["data"], runtime["joints"][joint]) for joint in arm_names}
            for index, raw_target in enumerate(arm_route):
                arm_goal = {}
                for i, joint_name in enumerate(arm_names):
                    joint = runtime["joints"][joint_name]
                    arm_goal[joint_name] = float(joint["reference"]) + int(joint["axis_sign"]) * (
                        float(raw_target[i]) - float(runtime["model"].qpos0[int(joint["qpos"])])
                    )
                phases.append((f"arm_to_approach_{index+1}", 600, start, arm_goal))
                arm_current = arm_goal
            gate_results.append(execute_trial("arm_motion_bottle_collision_disabled", phases, trace))
        if gate_results[-1]["passed"]:
            for driver in sorted(drivers):
                runtime = make_runtime()
                base = dict(runtime["initial_active"])
                low, high = runtime["transmission"].operational_limits(driver)
                center = float(base[driver])
                for direction in (-1.0, 1.0):
                    target = min(high, max(low, center + direction * 0.01))
                    if abs(target - center) <= 1e-10:
                        continue
                    perturb = dict(base)
                    perturb[driver] = target
                    gate_results.append(execute_trial(f"driver_perturb_{driver}_{direction:+.0f}x0p01", [
                        ("perturb", 100, perturb, None),
                        ("return", 100, base, None),
                    ], trace))
                    if not gate_results[-1]["passed"]:
                        break
                if not gate_results[-1]["passed"]:
                    break
        if gate_results[-1]["passed"]:
            # The fixed sphere is kept static; this checks external contact response only.
            right_start = {key: value for key, value in start.items()}
            right_close = {**right_start, **{key: value for key, value in drivers.items() if key.startswith("R_")}}
            gate_results.append(execute_trial("fixed_object_fingertip_contact", [
                ("close_to_fixed_probe", 900, right_close, None),
                ("fixed_probe_hold", 500, right_close, None),
                ("open_from_fixed_probe", 700, right_start, None),
            ], trace, fixed_probe=True))

    all_pass = bool(gate_results) and all(bool(result["passed"]) for result in gate_results)
    failed = next((result for result in gate_results if not result["passed"]), None)
    full_model, full_details = task.build_model(virtual_transmission=True)
    summary = {
        "status": "PASS" if all_pass else "FAIL",
        "first_failed_gate": None if failed is None else {"name": failed["label"], "condition": failed["first_failed_gate"]},
        "model_label": "CONTROL-LEVEL / ACTUATED MIMIC EMULATION",
        "public_interface": "Source active OmniHand coordinates only; follower coordinates are rejected by the command API.",
        "source_pin": task.SOURCE_PIN,
        "mujoco_version": mujoco.__version__,
        "timestep_s": task.DT,
        "dimensions": {"nq": full_model.nq, "nv": full_model.nv, "nu": full_model.nu, "neq": full_model.neq, "active_source_coordinates": len(full_details["active_hand_joint_names"]), "internal_follower_motors": len(full_details["virtual_follower_actuator_specs"]), "source_relations": len(full_details["mimic_relations"])},
        "simulation_derived_profiles": {
            "active_driver_kp_nm_per_rad": task.VIRTUAL_ACTIVE_SERVO_KP_NM_PER_RAD,
            "active_driver_kv_nms_per_rad": task.VIRTUAL_ACTIVE_SERVO_KV_NMS_PER_RAD,
            "active_driver_force_bound_nm": task.VIRTUAL_ACTIVE_MAX_TORQUE_NM,
            "follower_kp_nm_per_rad": task.VIRTUAL_FOLLOWER_SERVO_KP_NM_PER_RAD,
            "follower_kv_nms_per_rad": task.VIRTUAL_FOLLOWER_SERVO_KV_NMS_PER_RAD,
            "follower_force_bound_nm": task.VIRTUAL_FOLLOWER_MAX_TORQUE_NM,
            "operational_soft_limit_margin_rad": SOFT_LIMIT_MARGIN_RAD,
            "operational_soft_limit_note": "Targets only; source URDF/MuJoCo joint ranges and qpos are unchanged.",
            "passive_state_target_0p003_rad_is_architecture_blocker": False,
            "manipulation_abort_ceiling_rad": ABORT_RAD,
        },
        "initial_state": initial_state,
        "initial_active_source_state_rad": root_runtime["initial_state_active"],
        "initial_active_open_targets_rad": start,
        "prevalidation_trials": gate_results,
        "active_rollout_follower_qpos_writes": 0,
        "bottle_collision_disabled_for_prevalidation": True,
        "bottle_checkpoint_permitted": all_pass,
        "fixed_contact_probe_is_bottle": False,
        "trace_path": str(trace_path),
        "exact_command": "AGIBOT_X2_VENDOR_ROOT=<pinned checkout> ISSUE46_EVIDENCE_DIR=<evidence dir> <venv>/bin/python scripts/research/issue46_virtual_transmission_gate.py",
    }
    (OUT / "virtual_transmission_prevalidation_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
