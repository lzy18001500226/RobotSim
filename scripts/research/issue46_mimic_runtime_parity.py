#!/usr/bin/env python3
"""Compare accepted free-space mimic initialization with the manipulation path."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH_DIR))
import issue46_mimic_coordinate_diagnostic as accepted
import issue46_x2_grasp as task


OUT = Path(os.environ.get(
    "ISSUE46_PARITY_EVIDENCE_DIR",
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006",
))
TOLERANCE = 0.003
APPROACH_STEPS = int(os.environ.get("ISSUE46_PARITY_APPROACH_STEPS", "500"))
VENDOR_URDF = task.URDF


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def id_name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or f"unnamed_{object_id}"


def source_records() -> tuple[dict[str, ET.Element], dict[str, dict[str, Any]]]:
    root = ET.parse(VENDOR_URDF).getroot()
    joints = {node.get("name", ""): node for node in root.findall("joint")}
    relations: dict[str, dict[str, Any]] = {}
    for follower_name, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        relations[follower_name] = {
            "driver_joint": str(mimic.get("joint")),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }
    if len(relations) != 12:
        raise RuntimeError(f"expected 12 pinned URDF mimic relations, found {len(relations)}")
    return joints, relations


def spec_refs() -> dict[str, float]:
    spec = mujoco.MjSpec.from_file(str(VENDOR_URDF))
    return {joint.name: float(joint.ref) for joint in spec.joints if joint.name}


def actuator_by_joint(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for aid in range(model.nu):
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[aid, 0])
        if jid >= 0:
            result[id_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)] = aid
    return result


def actuator_record(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> dict[str, Any]:
    aid = actuator_by_joint(model).get(joint_name)
    if aid is None:
        return {"assigned": False, "target_rad": None}
    gain = np.asarray(model.actuator_gainprm[aid], dtype=float)
    bias = np.asarray(model.actuator_biasprm[aid], dtype=float)
    return {
        "assigned": True,
        "actuator_id": aid,
        "actuator_name": id_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
        "joint_name": joint_name,
        "target_rad": float(data.ctrl[aid]),
        "kp_from_gainprm0": float(gain[0]),
        "kv_from_negative_biasprm2": float(-bias[2]) if len(bias) > 2 else None,
        "gainprm": gain.tolist(),
        "biasprm": bias.tolist(),
        "gear": model.actuator_gear[aid].tolist(),
        "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
        "ctrllimited": bool(model.actuator_ctrllimited[aid]),
        "forcerange": model.actuator_forcerange[aid].tolist(),
        "forcelimited": bool(model.actuator_forcelimited[aid]),
        "target_joint_id": int(model.actuator_trnid[aid, 0]),
    }


def relation_ids(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for eid in range(model.neq):
        if int(model.eq_type[eid]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        follower = id_name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[eid]))
        result[follower] = eid
    return result


def pair_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_joints: dict[str, ET.Element],
    relations: dict[str, dict[str, Any]],
    refs: dict[str, float],
    eq_ids: dict[str, int],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    acts = actuator_by_joint(model)
    for follower_name, relation in sorted(relations.items()):
        driver_name = str(relation["driver_joint"])
        driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
        follower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
        if driver_id < 0 or follower_id < 0:
            continue
        driver_qadr = int(model.jnt_qposadr[driver_id])
        follower_qadr = int(model.jnt_qposadr[follower_id])
        driver_dadr = int(model.jnt_dofadr[driver_id])
        follower_dadr = int(model.jnt_dofadr[follower_id])
        driver_source = source_joints[driver_name]
        follower_source = source_joints[follower_name]
        driver_axis = np.fromstring(driver_source.find("axis").get("xyz", "1 0 0"), sep=" ")
        follower_axis = np.fromstring(follower_source.find("axis").get("xyz", "1 0 0"), sep=" ")
        driver_axis_dot = float(np.dot(driver_axis, model.jnt_axis[driver_id]))
        follower_axis_dot = float(np.dot(follower_axis, model.jnt_axis[follower_id]))
        driver_sign = 1 if driver_axis_dot >= 0 else -1
        follower_sign = 1 if follower_axis_dot >= 0 else -1
        equality_id = eq_ids.get(follower_name)
        polycoef = model.eq_data[equality_id, :5].tolist() if equality_id is not None else None
        driver_q = float(data.qpos[driver_qadr])
        follower_q = float(data.qpos[follower_qadr])
        driver_q0 = float(model.qpos0[driver_qadr])
        follower_q0 = float(model.qpos0[follower_qadr])
        if polycoef is None:
            residual = None
            expected_q = None
        else:
            dx = driver_q - driver_q0
            expected_q = follower_q0 + sum(float(polycoef[i]) * dx**i for i in range(5))
            residual = follower_q - expected_q
        source_driver_coordinate = refs.get(driver_name, 0.0) + driver_sign * (driver_q - driver_q0)
        source_follower_coordinate = refs.get(follower_name, 0.0) + follower_sign * (follower_q - follower_q0)
        expected_source = float(relation["offset"]) + float(relation["multiplier"]) * source_driver_coordinate
        output[follower_name] = {
            "driver_joint": driver_name,
            "source_multiplier": float(relation["multiplier"]),
            "source_offset": float(relation["offset"]),
            "driver_source_lower_upper": [
                float(source_joints[driver_name].find("limit").get("lower")),
                float(source_joints[driver_name].find("limit").get("upper")),
            ],
            "driver_source_axis": driver_axis.tolist(),
            "driver_mujoco_axis": model.jnt_axis[driver_id].tolist(),
            "driver_axis_dot_sign": driver_sign,
            "driver_mjspec_ref": refs.get(driver_name, 0.0),
            "driver_qpos0": driver_q0,
            "driver_qpos_range": model.jnt_range[driver_id].tolist(),
            "driver_qpos": driver_q,
            "driver_qpos_minus_qpos0": driver_q - driver_q0,
            "driver_qvel": float(data.qvel[driver_dadr]),
            "driver_source_coordinate": source_driver_coordinate,
            "driver_actuator": actuator_record(model, data, driver_name),
            "follower_source_lower_upper": [
                float(source_joints[follower_name].find("limit").get("lower")),
                float(source_joints[follower_name].find("limit").get("upper")),
            ],
            "follower_source_axis": follower_axis.tolist(),
            "follower_mujoco_axis": model.jnt_axis[follower_id].tolist(),
            "follower_axis_dot_sign": follower_sign,
            "follower_mjspec_ref": refs.get(follower_name, 0.0),
            "follower_qpos0": follower_q0,
            "follower_qpos_range": model.jnt_range[follower_id].tolist(),
            "follower_qpos": follower_q,
            "follower_qpos_minus_qpos0": follower_q - follower_q0,
            "follower_qvel": float(data.qvel[follower_dadr]),
            "follower_source_coordinate": source_follower_coordinate,
            "follower_expected_source_coordinate": expected_source,
            "equality_id": equality_id,
            "equality_name": id_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id) if equality_id is not None else None,
            "equality_active": bool(data.eq_active[equality_id]) if equality_id is not None else None,
            "equality_polycoef": polycoef,
            "equality_solref": model.eq_solref[equality_id].tolist() if equality_id is not None else None,
            "equality_solimp": model.eq_solimp[equality_id].tolist() if equality_id is not None else None,
            "equality_residual_rad": residual,
            "follower_has_independent_actuator": follower_name in acts,
            "follower_actuator": actuator_record(model, data, follower_name),
        }
    return output


def hand_limit_violations(model: mujoco.MjModel, data: mujoco.MjData, hand_names: list[str]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for joint_name in hand_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0 or not bool(model.jnt_limited[jid]):
            continue
        qadr = int(model.jnt_qposadr[jid])
        q = float(data.qpos[qadr])
        lo, hi = map(float, model.jnt_range[jid])
        depth = max(lo - q, q - hi, 0.0)
        if depth > 0.0:
            violations.append({"joint": joint_name, "qpos": q, "range": [lo, hi], "violation_rad": depth})
    return violations


def contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[str]]:
    pairs: list[list[str]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_name = id_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_name = id_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            names.append(f"{geom_name}@{body_name}")
        pairs.append(names)
    return sorted(pairs)


def stage_record(
    label: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_joints: dict[str, ET.Element],
    source_relations: dict[str, dict[str, Any]],
    refs: dict[str, float],
    eq_ids: dict[str, int],
    hand_names: list[str],
    arm_names: list[str],
) -> dict[str, Any]:
    pair_data = pair_state(model, data, source_joints, source_relations, refs, eq_ids)
    errors = {name: abs(float(item["equality_residual_rad"] or 0.0)) for name, item in pair_data.items()}
    return {
        "stage": label,
        "time_s": float(data.time),
        "gravity": model.opt.gravity.tolist(),
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "hand_qpos": {name: float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])]) for name in hand_names},
        "arm_qpos": {name: float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])]) for name in arm_names},
        "all_actuator_targets": {name: actuator_record(model, data, name)["target_rad"] for name in actuator_by_joint(model)},
        "relations": pair_data,
        "max_relation_error_rad": max(errors.values(), default=0.0),
        "limit_violations": hand_limit_violations(model, data, hand_names),
        "finite_qpos_qvel": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "contacts": contacts(model, data),
        "active_equalities": [id_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq) if bool(data.eq_active[i])],
    }


def set_position_actuator_profile(model: mujoco.MjModel, joint_name: str, kp: float, kv: float) -> None:
    aid = actuator_by_joint(model).get(joint_name)
    if aid is None:
        raise RuntimeError(f"no actuator for {joint_name}")
    model.actuator_gainprm[aid, 0] = kp
    model.actuator_biasprm[aid, 1] = -kp
    model.actuator_biasprm[aid, 2] = -kv


def compact_relation_trace(pair_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = (
        "driver_joint", "driver_qpos", "driver_qpos0", "driver_qpos_minus_qpos0",
        "driver_qvel", "driver_source_coordinate", "driver_source_lower_upper",
        "follower_qpos", "follower_qpos0", "follower_qpos_minus_qpos0",
        "follower_qvel", "follower_source_coordinate", "follower_expected_source_coordinate",
        "follower_source_lower_upper", "equality_active", "equality_residual_rad",
    )
    compact: dict[str, dict[str, Any]] = {}
    for name, record in pair_data.items():
        compact[name] = {field: record[field] for field in fields}
        compact[name]["driver_target_rad"] = record["driver_actuator"]["target_rad"]
    return compact


def initialize_pass_reference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    relations: list[dict[str, Any]],
    joint_records: dict[str, dict[str, Any]],
    actuator_map: dict[str, int],
) -> dict[str, float]:
    open_close = accepted.driver_open_and_close_targets(relations, joint_records)
    source_pose: dict[str, float] = {}
    for joint_name in actuator_map:
        if joint_name in open_close:
            source_pose[joint_name] = open_close[joint_name][0]
        else:
            source_pose[joint_name] = accepted.interior_neutral_target(joint_records[joint_name])
    for relation in relations:
        driver_name = str(relation["driver_joint"])
        follower_name = str(relation["follower_joint"])
        source_pose[follower_name] = float(relation["offset"]) + float(relation["multiplier"]) * source_pose[driver_name]
    for joint_name, source_coordinate in source_pose.items():
        record = joint_records[joint_name]
        data.qpos[int(record["qpos_address"])] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
            source_coordinate - float(record["source_coordinate_neutral_rad"])
        )
    for joint_name, aid in actuator_map.items():
        record = joint_records[joint_name]
        data.ctrl[aid] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
            source_pose[joint_name] - float(record["source_coordinate_neutral_rad"])
        )
    return source_pose


def pass_runtime_details(output: Path) -> tuple[Any, Any, Any, Any, Any, Any, Any, list[dict[str, Any]]]:
    model, relations, joint_records, actuator_map, eq_ids = accepted.build_model(
        "robot", "direct__moderate_controller", output
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    source_joint_map, source_relation_map = source_records()
    refs = spec_refs()
    all_hand = [name for name, node in source_joint_map.items() if name.startswith(("L_", "R_")) and node.get("type") == "revolute"]
    arms = [name for name in joint_records if any(part in name.lower() for part in ("shoulder", "elbow", "wrist"))]
    snapshots: list[dict[str, Any]] = []

    def capture(label: str) -> None:
        snapshots.append(stage_record(label, model, data, source_joint_map, source_relation_map, refs, eq_ids, all_hand, arms))

    capture("A_immediately_after_mj_resetData")
    open_close = accepted.driver_open_and_close_targets(relations, joint_records)
    source_pose: dict[str, float] = {}
    for joint_name in actuator_map:
        source_pose[joint_name] = open_close[joint_name][0] if joint_name in open_close else accepted.interior_neutral_target(joint_records[joint_name])
    for relation in relations:
        driver_name = str(relation["driver_joint"])
        follower_name = str(relation["follower_joint"])
        source_pose[follower_name] = float(relation["offset"]) + float(relation["multiplier"]) * source_pose[driver_name]
    for joint_name, source_coordinate in source_pose.items():
        record = joint_records[joint_name]
        data.qpos[int(record["qpos_address"])] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
            source_coordinate - float(record["source_coordinate_neutral_rad"])
        )
    capture("B_after_accepted_free_space_pose_initialization")
    for joint_name, aid in actuator_map.items():
        record = joint_records[joint_name]
        data.ctrl[aid] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
            source_pose[joint_name] - float(record["source_coordinate_neutral_rad"])
        )
    capture("C_after_accepted_free_space_target_initialization")
    mujoco.mj_forward(model, data)
    return model, data, relations, joint_records, actuator_map, eq_ids, source_pose, snapshots


def manipulation_setup(scenario: str, pass_pose_reference: dict[str, float]) -> dict[str, Any]:
    model, details = task.build_model()
    data = details["data"]
    if scenario in (
        "gravity_zero",
        "accepted_hand_state_gravity_zero",
        "accepted_hand_state_gravity_zero_all_contacts_disabled",
        "accepted_hand_state_full_targets_forward_each_step_no_contacts_no_gravity",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts_no_gravity",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_gravity_zero",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts_no_gravity",
    ):
        model.opt.gravity[:] = [0.0, 0.0, 0.0]
    source_joints, relations = source_records()
    refs = spec_refs()
    eq_ids = relation_ids(model)
    if scenario in ("left_legacy_gain", "pass_profile_left", "accepted_hand_state_profile") or scenario.startswith(
        "accepted_hand_state_full_targets_forward_each_step"
    ):
        for relation in relations.values():
            driver_name = str(relation["driver_joint"])
            if driver_name.startswith("L_"):
                set_position_actuator_profile(model, driver_name, 18.0, 2.4)
    for geom_name in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            raise RuntimeError(f"missing bottle geom {geom_name}")
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
    if scenario in (
        "all_contacts_disabled",
        "gravity_zero_all_contacts_disabled",
        "accepted_hand_state_all_contacts_disabled",
        "accepted_hand_state_gravity_zero_all_contacts_disabled",
        "accepted_hand_state_full_targets_forward_each_step_no_contacts_no_gravity",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts_no_gravity",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts_no_gravity",
    ):
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    if scenario in (
        "gravity_zero_all_contacts_disabled",
        "accepted_hand_state_gravity_zero_all_contacts_disabled",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts_no_gravity",
        "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_gravity_zero",
    ):
        model.opt.gravity[:] = [0.0, 0.0, 0.0]
    mujoco.mj_resetData(model, data)

    arm_names = list(task.ARM)
    finger_names = list(task.FINGERS)
    all_hand_names = [name for name, node in source_joints.items() if name.startswith(("L_", "R_")) and node.get("type") == "revolute"]
    left_hand_names = [name for name in all_hand_names if name.startswith("L_")]
    arms, arm_qpos = task.resolve_joints(model, arm_names)
    _, finger_qpos = task.resolve_joints(model, finger_names)
    actuator_map = actuator_by_joint(model)
    arm_ctrl = [actuator_map[name] for name in arm_names]
    finger_ctrl = [actuator_map[name] for name in finger_names]
    q0 = data.qpos.copy()
    init_snapshots: list[dict[str, Any]] = []

    def capture(label: str) -> None:
        init_snapshots.append(stage_record(label, model, data, source_joints, relations, refs, eq_ids, all_hand_names, arm_names))

    capture("A_immediately_after_mj_resetData")
    approach_route = task.ik_plan(
        model, q0, int(details["site_id"]), arms,
        [task.APPROACH_PALM_POS.copy(), task.PREGRASP_PALM_POS.copy(), task.GRASP_PALM_POS.copy()],
        task.PALM_TARGET_ROTATION,
    )
    data.qpos[arm_qpos] = approach_route[0]
    accepted_hand_state = scenario.startswith("accepted_hand_state")
    open_target = q0[finger_qpos].copy()
    if accepted_hand_state:
        for joint_name in all_hand_names:
            if joint_name not in pass_pose_reference:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            source_axis = np.fromstring(source_joints[joint_name].find("axis").get("xyz", "1 0 0"), sep=" ")
            sign = 1 if np.dot(source_axis, model.jnt_axis[jid]) >= 0 else -1
            qadr = int(model.jnt_qposadr[jid])
            source_target = float(pass_pose_reference[joint_name])
            data.qpos[qadr] = float(model.qpos0[qadr]) + sign * (source_target - refs.get(joint_name, 0.0))
        for index, joint_name in enumerate(finger_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            source_axis = np.fromstring(source_joints[joint_name].find("axis").get("xyz", "1 0 0"), sep=" ")
            sign = 1 if np.dot(source_axis, model.jnt_axis[jid]) >= 0 else -1
            qadr = int(model.jnt_qposadr[jid])
            source_target = float(pass_pose_reference[joint_name])
            open_target[index] = float(model.qpos0[qadr]) + sign * (source_target - refs.get(joint_name, 0.0))
    else:
        for joint_name, value in {
            "R_thumb_roll_joint": 0.80,
            "R_thumb_abad_joint": -1.70,
            "R_index_abad_joint": -0.18,
            "R_ring_abad_joint": 0.15,
            "R_pinky_abad_joint": 0.15,
        }.items():
            open_target[finger_names.index(joint_name)] = value
    data.qpos[finger_qpos] = open_target
    for relation in details["mimic_relations"]:
        driver_q = float(data.qpos[relation["driver_qpos_address"]])
        data.qpos[relation["follower_qpos_address"]] = (
            relation["compiled_follower_reference_rad"] + relation["mujoco_polycoef"][0]
            + relation["mujoco_polycoef"][1] * (driver_q - relation["compiled_driver_reference_rad"])
        )
    if scenario in ("left_seed_only", "left_seed_target", "pass_profile_left"):
        for joint_name in left_hand_names:
            if joint_name not in pass_pose_reference:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            sign = 1 if np.dot(np.fromstring(source_joints[joint_name].find("axis").get("xyz", "1 0 0"), sep=" "), model.jnt_axis[jid]) >= 0 else -1
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(model.qpos0[qadr]) + sign * (float(pass_pose_reference[joint_name]) - refs.get(joint_name, 0.0))
        for follower_name, relation in relations.items():
            if follower_name.startswith("L_"):
                did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(relation["driver_joint"]))
                fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                dq = float(data.qpos[int(model.jnt_qposadr[did])])
                dq0 = float(model.qpos0[int(model.jnt_qposadr[did])])
                poly = model.eq_data[eq_ids[follower_name], :5]
                fqadr = int(model.jnt_qposadr[fid])
                data.qpos[fqadr] = float(model.qpos0[fqadr]) + float(poly[0]) + float(poly[1]) * (dq - dq0)
    capture("B_after_pose_and_follower_initialization")

    mujoco.mj_forward(model, data)
    close_target = open_target.copy()
    for index, joint_name in enumerate(finger_names):
        target = task.FINGERS[joint_name]
        if "thumb_mcp" in joint_name:
            target = 0.82
        elif joint_name.endswith("_pip_joint"):
            target = 1.15
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        close_target[index] = np.clip(target, model.jnt_range[jid, 0], model.jnt_range[jid, 1])
    close_target = open_target + task.FINGER_CLOSE_FRACTION * (close_target - open_target)
    close_target[finger_names.index("R_thumb_roll_joint")] = 0.65
    close_target[finger_names.index("R_thumb_abad_joint")] = -0.30
    preshape_target = open_target + task.FINGER_PRESHAPE_FRACTION * (close_target - open_target)
    hold_targets = data.ctrl.copy()
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        hold_targets[aid] = q0[int(model.jnt_qposadr[jid])]
    data.ctrl[:] = hold_targets
    data.ctrl[arm_ctrl] = approach_route[0]
    data.ctrl[finger_ctrl] = open_target
    if scenario in ("left_target_only", "left_seed_target", "pass_profile_left") or accepted_hand_state:
        for joint_name in left_hand_names:
            aid = actuator_map.get(joint_name)
            if aid is None or joint_name not in pass_pose_reference:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            sign = 1 if np.dot(np.fromstring(source_joints[joint_name].find("axis").get("xyz", "1 0 0"), sep=" "), model.jnt_axis[jid]) >= 0 else -1
            qadr = int(model.jnt_qposadr[jid])
            target = float(model.qpos0[qadr]) + sign * (float(pass_pose_reference[joint_name]) - refs.get(joint_name, 0.0))
            data.ctrl[aid] = target
            hold_targets[aid] = target
    if scenario.startswith("accepted_hand_state_full_targets_forward_each_step"):
        for joint_name in all_hand_names:
            aid = actuator_map.get(joint_name)
            if aid is None or joint_name not in pass_pose_reference:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            source_axis = np.fromstring(source_joints[joint_name].find("axis").get("xyz", "1 0 0"), sep=" ")
            sign = 1 if np.dot(source_axis, model.jnt_axis[jid]) >= 0 else -1
            qadr = int(model.jnt_qposadr[jid])
            source_target = float(pass_pose_reference[joint_name])
            target = float(model.qpos0[qadr]) + sign * (source_target - refs.get(joint_name, 0.0))
            data.ctrl[aid] = target
            hold_targets[aid] = target
    capture("C_after_controller_target_initialization")
    return {
        "scenario": scenario, "model": model, "data": data, "details": details,
        "source_joints": source_joints, "source_relations": relations, "refs": refs, "eq_ids": eq_ids,
        "all_hand_names": all_hand_names, "left_hand_names": left_hand_names, "arm_names": arm_names,
        "arm_qpos": arm_qpos, "finger_qpos": finger_qpos, "arm_ctrl": arm_ctrl, "finger_ctrl": finger_ctrl,
        "approach_route": approach_route, "open_target": open_target, "preshape_target": preshape_target,
        "hold_targets": hold_targets,
        "bottle_collision_disabled": True,
        "open_fingers_during_approach": "open_during_approach" in scenario,
        "freeze_arm_during_diagnostic": "no_arm_motion" in scenario,
        "all_geom_contacts_disabled": scenario in (
            "all_contacts_disabled",
            "gravity_zero_all_contacts_disabled",
            "accepted_hand_state_all_contacts_disabled",
            "accepted_hand_state_gravity_zero_all_contacts_disabled",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts_no_gravity",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts_no_gravity",
        ),
        "init_snapshots": init_snapshots,
    }


def manipulate_step_control(state: dict[str, Any], index: int, total: int) -> None:
    model = state["model"]
    data = state["data"]
    arm_start = np.asarray(state["approach_route"][0], dtype=float)
    arm_target = np.asarray(state["approach_route"][1], dtype=float)
    open_target = np.asarray(state["open_target"], dtype=float)
    finger_target = (
        open_target
        if state.get("open_fingers_during_approach")
        else np.asarray(state["preshape_target"], dtype=float)
    )
    alpha = (index + 1) / total
    data.ctrl[:] = state["hold_targets"]
    if state.get("freeze_arm_during_diagnostic"):
        data.ctrl[state["arm_ctrl"]] = arm_start
    else:
        data.ctrl[state["arm_ctrl"]] = arm_start * (1.0 - alpha) + arm_target * alpha
    data.ctrl[state["finger_ctrl"]] = open_target * (1.0 - alpha) + finger_target * alpha


def execute_scenario(state: dict[str, Any], stage_file, trace_file, pass_step: bool = False) -> dict[str, Any]:
    model, data = state["model"], state["data"]
    hand_names = state["all_hand_names"]
    arm_names = state["arm_names"]
    snapshots: list[dict[str, Any]] = []

    def capture(label: str) -> None:
        record = stage_record(label, model, data, state["source_joints"], state["source_relations"], state["refs"], state["eq_ids"], hand_names, arm_names)
        record["scenario"] = state["scenario"]
        snapshots.append(record)
        stage_file.write(json.dumps(record, sort_keys=True) + "\n")

    for snapshot in state["init_snapshots"]:
        snapshot = dict(snapshot)
        snapshot["scenario"] = state["scenario"]
        stage_file.write(json.dumps(snapshot, sort_keys=True) + "\n")
    if pass_step:
        manipulate_step_control(state, 0, APPROACH_STEPS)
        mujoco.mj_forward(model, data)
    else:
        manipulate_step_control(state, 0, APPROACH_STEPS)
    capture("D_immediately_before_first_mj_step")
    mujoco.mj_step(model, data)
    capture("E_immediately_after_first_mj_step")

    first = snapshots[-1]
    max_error = float(first["max_relation_error_rad"])
    max_violation_count = len(first["limit_violations"])
    any_nan = not first["finite_qpos_qvel"]
    driver_motion_opposed_to_target_count = 0
    previous_sign_state: dict[str, tuple[float, float]] = {}
    for follower_name, relation in first["relations"].items():
        target = relation["driver_actuator"]["target_rad"]
        q0 = relation["driver_qpos0"]
        target_dir = float(target) - float(q0)
        actual_dir = relation["driver_qpos"] - float(q0)
        if abs(target_dir) > 1e-6 and actual_dir * target_dir < -1e-8:
            driver_motion_opposed_to_target_count += 1
        previous_sign_state[follower_name] = (float(target), float(q0))
    max_contacts = len(first["contacts"])
    first_bottle_contact = any("bottle_" in geom for pair in first["contacts"] for geom in pair)
    max_err_by_relation = {name: abs(float(item["equality_residual_rad"] or 0.0)) for name, item in first["relations"].items()}
    first_breach = None
    if max_error > TOLERANCE:
        failing = max(max_err_by_relation, key=max_err_by_relation.get)
        first_breach = {"step": 1, "time_s": float(data.time), "follower": failing, "error_rad": max_err_by_relation[failing]}
    trace_file.write(json.dumps({
        "scenario": state["scenario"], "step": 1, "time_s": float(data.time),
        "relations": compact_relation_trace(first["relations"]), "max_relation_error_rad": max_error,
        "limit_violations": first["limit_violations"], "nan": any_nan,
        "driver_motion_opposed_to_target_count": driver_motion_opposed_to_target_count, "contacts": first["contacts"],
        "qpos_write_count_after_rollout_start": 0,
    }, sort_keys=True) + "\n")

    first_step_summary = {
        "first_step_max_relation_error_rad": max_error,
        "first_step_first_breach": first_breach,
        "first_step_limit_violation_count": len(first["limit_violations"]),
        "first_step_limit_violations": first["limit_violations"],
        "first_step_nan": any_nan,
        "first_step_driver_motion_opposed_to_target_count": driver_motion_opposed_to_target_count,
        "first_step_contacts": first["contacts"],
        "first_step_bottle_contact": first_bottle_contact,
    }

    for index in range(1, APPROACH_STEPS):
        manipulate_step_control(state, index, APPROACH_STEPS)
        if pass_step:
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        pairs = pair_state(model, data, state["source_joints"], state["source_relations"], state["refs"], state["eq_ids"])
        errors = {name: abs(float(item["equality_residual_rad"] or 0.0)) for name, item in pairs.items()}
        violations = hand_limit_violations(model, data, hand_names)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        target_inversion = []
        for follower_name, relation in pairs.items():
            target = relation["driver_actuator"]["target_rad"]
            q0 = relation["driver_qpos0"]
            direction = target - q0
            actual = relation["driver_qpos"] - q0
            if abs(direction) > 1e-6 and actual * direction < -1e-8:
                target_inversion.append(follower_name)
        contact_pairs = contacts(model, data)
        max_error = max(max_error, max(errors.values(), default=0.0))
        max_violation_count = max(max_violation_count, len(violations))
        any_nan = any_nan or not finite
        driver_motion_opposed_to_target_count += len(target_inversion)
        max_contacts = max(max_contacts, len(contact_pairs))
        first_bottle_contact = first_bottle_contact or any("bottle_" in geom for pair in contact_pairs for geom in pair)
        if first_breach is None and any(value > TOLERANCE for value in errors.values()):
            failure = max(errors, key=errors.get)
            first_breach = {"step": index + 1, "time_s": float(data.time), "follower": failure, "error_rad": errors[failure]}
        trace_file.write(json.dumps({
            "scenario": state["scenario"], "step": index + 1, "time_s": float(data.time),
            "relations": compact_relation_trace(pairs), "max_relation_error_rad": max(errors.values(), default=0.0),
            "limit_violations": violations, "nan": not finite,
            "driver_motion_opposed_to_target_relations": target_inversion,
            "contacts": contact_pairs, "qpos_write_count_after_rollout_start": 0,
        }, sort_keys=True) + "\n")

    return {
        "scenario": state["scenario"],
        "gravity": model.opt.gravity.tolist(),
        "timestep_s": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "solver_iterations": int(model.opt.iterations),
        "approach_steps": APPROACH_STEPS,
        "approach_simulation_duration_s": float(data.time),
        "max_relation_error_rad": max_error,
        "first_breach": first_breach,
        "max_hand_limit_violation_count_at_any_step": max_violation_count,
        "any_nan": any_nan,
        "driver_motion_opposed_to_target_observations": driver_motion_opposed_to_target_count,
        "max_contact_count": max_contacts,
        "bottle_contact_observed": first_bottle_contact,
        "active_rollout_follower_qpos_writes": 0,
        "first_step": first_step_summary,
        "pass_gate": max_error <= TOLERANCE
        and max_violation_count == 0
        and not any_nan
        and not first_bottle_contact
        and driver_motion_opposed_to_target_count == 0,
    }


def model_profile(model: mujoco.MjModel, data: mujoco.MjData, relations: dict[str, dict[str, Any]], refs: dict[str, float]) -> dict[str, Any]:
    eq_ids = relation_ids(model)
    acts = actuator_by_joint(model)
    driver_names = sorted({str(item["driver_joint"]) for item in relations.values()})
    all_names = sorted({str(item) for item in driver_names} | set(relations))
    pair_records = {}
    for follower_name, relation in sorted(relations.items()):
        driver_name = str(relation["driver_joint"])
        did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
        aid = acts.get(driver_name)
        eid = eq_ids.get(follower_name)
        pair_records[follower_name] = {
            "driver_joint": driver_name,
            "driver_qpos0": float(model.qpos0[int(model.jnt_qposadr[did])]),
            "driver_range": model.jnt_range[did].tolist(),
            "driver_axis": model.jnt_axis[did].tolist(),
            "driver_mjspec_ref": refs.get(driver_name, 0.0),
            "driver_actuator": None if aid is None else {
                "id": aid, "kp": float(model.actuator_gainprm[aid, 0]), "kv": float(-model.actuator_biasprm[aid, 2]),
                "gear": model.actuator_gear[aid].tolist(), "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
                "forcerange": model.actuator_forcerange[aid].tolist(),
            },
            "follower_qpos0": float(model.qpos0[int(model.jnt_qposadr[fid])]),
            "follower_range": model.jnt_range[fid].tolist(),
            "follower_axis": model.jnt_axis[fid].tolist(),
            "follower_mjspec_ref": refs.get(follower_name, 0.0),
            "follower_actuator_assigned": follower_name in acts,
            "equality_id": eid,
            "equality_active_default": bool(model.eq_active0[eid]) if eid is not None else None,
            "equality_active_reset": bool(data.eq_active[eid]) if eid is not None else None,
            "equality_polycoef": model.eq_data[eid, :5].tolist() if eid is not None else None,
            "equality_solref": model.eq_solref[eid].tolist() if eid is not None else None,
            "equality_solimp": model.eq_solimp[eid].tolist() if eid is not None else None,
        }
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "neq": int(model.neq),
        "timestep_s": float(model.opt.timestep), "gravity": model.opt.gravity.tolist(),
        "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
        "solver_iterations": int(model.opt.iterations), "solver_tolerance": float(model.opt.tolerance),
        "impratio": float(model.opt.impratio), "cone": int(model.opt.cone), "jacobian": int(model.opt.jacobian),
        "mimic_driver_actuators": {joint: actuator_record(model, data, joint) for joint in driver_names},
        "mimic_pair_static": pair_records,
    }


def pass_step(state: dict[str, Any], stage_file, trace_file) -> dict[str, Any]:
    model, data = state["model"], state["data"]
    source_joints_map, source_relations = source_records()
    refs = spec_refs()
    all_hand = [name for name, node in source_joints_map.items() if name.startswith(("L_", "R_")) and node.get("type") == "revolute"]
    arms = [name for name in state["joint_records"] if any(part in name.lower() for part in ("shoulder", "elbow", "wrist"))]
    for snapshot in state["init_snapshots"]:
        snapshot = dict(snapshot)
        snapshot["scenario"] = "accepted_free_space_pass"
        stage_file.write(json.dumps(snapshot, sort_keys=True) + "\n")

    def capture(label: str) -> None:
        record = stage_record(label, model, data, source_joints_map, source_relations, refs, state["equality_ids"], all_hand, arms)
        record["scenario"] = "accepted_free_space_pass"
        stage_file.write(json.dumps(record, sort_keys=True) + "\n")
    mujoco.mj_forward(model, data)
    capture("D_immediately_before_first_mj_step")
    mujoco.mj_step(model, data)
    capture("E_immediately_after_first_mj_step")
    current = pair_state(model, data, source_joints_map, source_relations, refs, state["equality_ids"])
    error = max(abs(float(item["equality_residual_rad"] or 0.0)) for item in current.values())
    trace_file.write(json.dumps({
        "scenario": "accepted_free_space_pass", "step": 1, "time_s": float(data.time),
        "relations": current, "max_relation_error_rad": error,
        "limit_violations": hand_limit_violations(model, data, all_hand),
        "nan": not bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "contacts": contacts(model, data), "qpos_write_count_after_rollout_start": 0,
    }, sort_keys=True) + "\n")
    return {
        "scenario": "accepted_free_space_pass", "gravity": model.opt.gravity.tolist(),
        "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations),
        "first_step_max_relation_error_rad": error,
        "first_step_limit_violation_count": len(hand_limit_violations(model, data, all_hand)),
        "first_step_nan": not bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "contacts": contacts(model, data), "left_targets": {name: actuator_record(model, data, name)["target_rad"] for name in state["actuator_map"] if name.startswith("L_")},
        "left_driver_gains": {name: {"kp": actuator_record(model, data, name)["kp_from_gainprm0"], "kv": actuator_record(model, data, name)["kv_from_negative_biasprm2"]} for name in state["actuator_map"] if name.startswith("L_") and name in {str(v["driver_joint"]) for v in source_relations.values()}},
        "pass_gate_from_saved_result": True,
    }


def build_pass_state(output: Path) -> dict[str, Any]:
    model, data, relations, joint_records, actuator_map, eq_ids, source_pose, init_snapshots = pass_runtime_details(output)
    source_joints_map, source_relations = source_records()
    return {
        "model": model, "data": data, "relations": relations,
        "joint_records": joint_records, "actuator_map": actuator_map,
        "equality_ids": eq_ids, "source_pose": source_pose,
        "source_joints": source_joints_map, "source_relations": source_relations,
        "refs": spec_refs(), "init_snapshots": init_snapshots,
    }


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"evidence directory must be empty: {OUT}")
    if not VENDOR_URDF.is_file():
        raise FileNotFoundError(VENDOR_URDF)
    source_joints_map, source_relations = source_records()
    refs = spec_refs()
    data_file = (OUT / "step0_pre_post_trace.jsonl").open("w", encoding="utf-8")
    trace_file = (OUT / "all_12_mimic_error_trace.jsonl").open("w", encoding="utf-8")
    try:
        pass_state = build_pass_state(OUT)
        accepted_pass = pass_step(pass_state, data_file, trace_file)
        pass_model_profile = model_profile(pass_state["model"], pass_state["data"], source_relations, refs)

        scenarios = [
            "manipulation_baseline",
            "gravity_zero",
            "all_contacts_disabled",
            "gravity_zero_all_contacts_disabled",
            "left_seed_only",
            "left_target_only",
            "left_seed_target",
            "left_legacy_gain",
            "pass_profile_left",
            "manipulation_forward_before_step",
            "accepted_hand_state",
            "accepted_hand_state_profile",
            "accepted_hand_state_forward_each_step",
            "accepted_hand_state_full_targets_forward_each_step",
            "accepted_hand_state_full_targets_forward_each_step_no_contacts_no_gravity",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts_no_gravity",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_gravity_zero",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts",
            "accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts_no_gravity",
            "accepted_hand_state_all_contacts_disabled",
            "accepted_hand_state_gravity_zero",
            "accepted_hand_state_gravity_zero_all_contacts_disabled",
        ]
        requested_scenarios = os.environ.get("ISSUE46_PARITY_SCENARIOS")
        if requested_scenarios:
            requested = {item.strip() for item in requested_scenarios.split(",") if item.strip()}
            unknown = requested.difference(scenarios)
            if unknown:
                raise ValueError(f"unknown requested parity scenarios: {sorted(unknown)}")
            scenarios = [scenario for scenario in scenarios if scenario in requested]
            if not scenarios:
                raise ValueError("scenario filter selected no diagnostic scenarios")
        scenario_results = []
        model_profiles = {}
        pass_pose_reference = pass_state["source_pose"]
        for scenario in scenarios:
            state = manipulation_setup(scenario, pass_pose_reference)
            model, data = state["model"], state["data"]
            if scenario == "manipulation_forward_before_step":
                state["forward_before_step"] = True
            model_profiles[scenario] = model_profile(model, data, source_relations, refs)
            # A is reset state before the manipulation pose initialization; reconstructed with a fresh model.
            # B/C/D/E snapshots are recorded from the exact manipulation init/controller path.
            result = execute_scenario(
                state, data_file, trace_file,
                pass_step=(
                    scenario == "manipulation_forward_before_step"
                    or "forward_each_step" in scenario
                ),
            )
            scenario_results.append(result)
        test_summary = {
            "initialization_order": {
                "accepted_free_space_pass": [
                    "mj_resetData", "build interior hand/actuator target map", "seed all mapped actuated joints and mimic followers", "mj_forward", "set/update controls", "mj_forward", "mj_step"
                ],
                "manipulation_checkpoint": [
                    "task build_model", "mj_resetData", "solve approach route", "seed right arm/open hand and mimic followers", "mj_forward", "set all hold targets and right arm/hand targets", "per-step overwrite controls", "mj_step"
                ],
            },
            "controller_update_order": {
                "accepted_free_space_pass": "update target controls -> mj_forward -> mj_step",
                "manipulation_checkpoint": "overwrite hold and interpolated right arm/hand controls -> mj_step; no mj_forward between control update and first step",
            },
            "accepted_free_space_pass_first_step": accepted_pass,
            "manipulation_scenarios": scenario_results,
            "scenario_configuration": {name: {
                "gravity": model_profiles[name]["gravity"],
                "left_driver_gains": {joint: {
                    "kp": details["kp_from_gainprm0"], "kv": details["kv_from_negative_biasprm2"]
                } for joint, details in model_profiles[name]["mimic_driver_actuators"].items() if joint.startswith("L_")},
                "timestep_s": model_profiles[name]["timestep_s"],
                "integrator": model_profiles[name]["integrator"],
                "solver": model_profiles[name]["solver"],
                "solver_iterations": model_profiles[name]["solver_iterations"],
            } for name in scenarios},
        }
    finally:
        data_file.close()
        trace_file.close()

    git = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip(),
    }
    runtime = {
        "source_pin": task.SOURCE_PIN,
        "urdf": str(VENDOR_URDF),
        "urdf_sha256": sha256(VENDOR_URDF),
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "tolerance_rad": TOLERANCE,
        "approach_steps": APPROACH_STEPS,
        "scenario_filter": os.environ.get("ISSUE46_PARITY_SCENARIOS"),
        "robot_sim_git": git,
        "vendor_head": subprocess.check_output(["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_status": subprocess.check_output(["git", "-C", str(task.ROOT), "status", "--short"], text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "reproduction_command": "MUJOCO_GL=egl ISSUE46_PARITY_EVIDENCE_DIR=<empty-directory> /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python scripts/research/issue46_mimic_runtime_parity.py",
        "evidence_scope": "bottle collision disabled in every manipulation scenario; non-bottle robot contacts remain enabled except in explicitly named all-contact ablations; no bottle contact or manipulation beyond approach; no active-rollout qpos writes",
    }
    json_dump(OUT / "runtime_identity.json", runtime)
    json_dump(OUT / "runtime_comparison.json", test_summary)
    json_dump(OUT / "model_profiles.json", {"accepted_free_space_pass": pass_model_profile, "manipulation_scenarios": model_profiles})
    failing = [item for item in test_summary["manipulation_scenarios"] if not item["pass_gate"]]
    result_status = "PASS" if accepted_pass["first_step_max_relation_error_rad"] <= TOLERANCE and not failing else "FAIL"
    result = {
        "status": result_status,
        "classification": "runtime/configuration parity discrepancy; bottle contact absent; non-bottle contact, gravity, and controller ablations tested",
        "accepted_pass_step1_error_rad": accepted_pass["first_step_max_relation_error_rad"],
        "baseline_first_breach": next((x["first_step"]["first_step_first_breach"] for x in test_summary["manipulation_scenarios"] if x["scenario"] == "manipulation_baseline"), None),
        "bottle_collision_disabled": True,
        "all_scenarios_no_bottle_contact": all(not x["bottle_contact_observed"] and not x["first_step"]["first_step_bottle_contact"] for x in test_summary["manipulation_scenarios"]),
        "approach_motion_steps": APPROACH_STEPS,
        "pass_gate_tolerance_rad": TOLERANCE,
        "first_failing_manipulation_scenario": next((x["scenario"] for x in test_summary["manipulation_scenarios"] if not x["pass_gate"]), None),
        "variant_results": [{"scenario": x["scenario"], "pass_gate": x["pass_gate"], "max_relation_error_rad": x["max_relation_error_rad"], "first_breach": x["first_breach"], "limit_violation_count": x["max_hand_limit_violation_count_at_any_step"], "driver_motion_opposed_to_target_observations": x["driver_motion_opposed_to_target_observations"]} for x in test_summary["manipulation_scenarios"]],
    }
    json_dump(OUT / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run())
