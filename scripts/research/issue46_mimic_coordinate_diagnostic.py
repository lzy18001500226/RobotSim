#!/usr/bin/env python3
"""Isolate pinned X2 OmniHand mimic coordinates and dynamics in scratch models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
VENDOR_ROOT = Path(
    os.environ.get(
        "AGIBOT_X2_VENDOR_ROOT",
        "/tmp/robotsim-issue46-agibot-x2-urdf-575cc6b988f976c23550e0db85aa1e5475d3652d",
    )
)
URDF = VENDOR_ROOT / "X2_URDF-v1.4.0" / "X2-Ultra_omnihand.urdf"
OUT = Path(
    os.environ.get(
        "ISSUE46_EVIDENCE_DIR",
        "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/attempts/20261005-mimic-coordinate-diagnostic",
    )
)
DT = 0.002
RELATION_TOLERANCE = 0.003
PAIR = ("R_pinky_pip_joint", "R_pinky_dip_joint")
FULL_ROBOT_BASE_POS = [0.38, 0.32, 0.68]
FULL_ROBOT_BASE_YAW = -math.pi / 2.0
TEST_PROFILES = {
    "default__legacy_controller": {"solref": None, "kp": 18.0, "kv": 2.4, "effort_cap": 33.5},
    "direct__legacy_controller": {"solref": [-10000.0, -200.0], "kp": 18.0, "kv": 2.4, "effort_cap": 33.5},
    "default__inertia_scaled_controller": {"solref": None, "kp": 0.01, "kv": 0.001, "effort_cap": 33.5},
    "direct__inertia_scaled_controller": {"solref": [-10000.0, -200.0], "kp": 0.01, "kv": 0.001, "effort_cap": 33.5},
    "direct__moderate_controller": {"solref": [-10000.0, -200.0], "kp": 0.1, "kv": 0.003, "effort_cap": 33.5},
}
SWEEP_FRACTIONS = (0.0, 0.0, 0.4, 0.4, 1.0, 1.0, 0.0)
SWEEP_SEGMENT_SECONDS = (0.5, 0.5, 4.0, 0.5, 4.0, 0.5, 4.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source() -> tuple[ET.Element, dict[str, ET.Element], dict[str, ET.Element], dict[str, dict[str, object]]]:
    root = ET.parse(URDF).getroot()
    links = {item.get("name", ""): item for item in root.findall("link")}
    joints = {item.get("name", ""): item for item in root.findall("joint")}
    relations: dict[str, dict[str, object]] = {}
    for joint_name, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        limit = joint.find("limit")
        axis = joint.find("axis")
        relations[joint_name] = {
            "follower_joint": joint_name,
            "driver_joint": mimic.get("joint"),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
            "follower_axis_urdf": np.fromstring(axis.get("xyz", "1 0 0"), sep=" ").tolist(),
            "follower_limit_urdf": [float(limit.get("lower")), float(limit.get("upper"))],
        }
    return root, links, joints, relations


def set_absolute_mesh_paths(node: ET.Element) -> None:
    for mesh in node.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", str((URDF.parent / filename).resolve()))


def make_subset_urdf(kind: str, output: Path) -> tuple[list[str], dict[str, ET.Element]]:
    root, links, joints, relations = read_source()
    selected_links: set[str]
    selected_joints: set[str]
    mounts: dict[str, ET.Element] = {}
    if kind == "pair":
        selected_links = {"R_pinky_abad", "R_pinky_pip", "R_pinky_dip"}
        selected_joints = {"R_pinky_pip_joint", "R_pinky_dip_joint"}
    elif kind == "hands":
        selected_links = {"L_palm", "R_palm"}
        selected_joints = set()
        changed = True
        while changed:
            changed = False
            for joint_name, joint in joints.items():
                parent = joint.find("parent").get("link")
                child = joint.find("child").get("link")
                if parent in selected_links and joint_name not in selected_joints:
                    selected_joints.add(joint_name)
                    selected_links.add(child)
                    changed = True
        for side in ("L", "R"):
            source_mount = copy.deepcopy(joints[f"{side}_palm_joint"])
            origin = source_mount.find("origin")
            if origin is None:
                origin = ET.SubElement(source_mount, "origin")
            origin.set("xyz", "-0.20 0 0" if side == "L" else "0.20 0 0")
            source_rpy = joints[f"{side}_palm_joint"].find("origin")
            if source_rpy is not None and source_rpy.get("rpy"):
                origin.set("rpy", source_rpy.get("rpy"))
            else:
                origin.attrib.pop("rpy", None)
            source_mount.set("name", f"diagnostic_{side}_palm_mount")
            source_mount.set("type", "fixed")
            source_mount.find("parent").set("link", "diagnostic_world")
            mounts[side] = source_mount
    elif kind == "robot":
        selected_links = set(links)
        selected_joints = set(joints)
    else:
        raise ValueError(f"unknown subset kind: {kind}")

    generated = ET.Element("robot", {"name": f"issue46_{kind}_mimic_diagnostic"})
    if kind == "hands":
        ET.SubElement(generated, "link", {"name": "diagnostic_world"})
    for material in root.findall("material"):
        generated.append(copy.deepcopy(material))
    for link_name in sorted(selected_links):
        link = copy.deepcopy(links[link_name])
        set_absolute_mesh_paths(link)
        if kind == "pair":
            for child in list(link):
                if child.tag in ("visual", "collision"):
                    link.remove(child)
        generated.append(link)
    for side in mounts:
        generated.append(mounts[side])
    for joint_name in sorted(selected_joints):
        joint = copy.deepcopy(joints[joint_name])
        set_absolute_mesh_paths(joint)
        generated.append(joint)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(generated, space="  ")
    ET.ElementTree(generated).write(output, encoding="utf-8", xml_declaration=True)
    relevant_relations = {
        name: relation
        for name, relation in relations.items()
        if name in selected_joints and relation["driver_joint"] in selected_joints
    }
    return sorted(selected_joints), relevant_relations


def source_joint_records(spec: mujoco.MjSpec, model: mujoco.MjModel, selected_names: set[str]) -> dict[str, dict[str, object]]:
    _, _, source_joints, _ = read_source()
    spec_by_name = {joint.name: joint for joint in spec.joints if joint.name}
    records: dict[str, dict[str, object]] = {}
    for joint_name in sorted(selected_names):
        source = source_joints[joint_name]
        if source.get("type") != "revolute":
            continue
        source_axis = np.fromstring(source.find("axis").get("xyz", "1 0 0"), sep=" ")
        source_limit = source.find("limit")
        source_range = [float(source_limit.get("lower")), float(source_limit.get("upper"))]
        source_ref = 0.0
        spec_joint = spec_by_name[joint_name]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        compiled_axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
        axis_dot = float(np.dot(source_axis / np.linalg.norm(source_axis), compiled_axis / np.linalg.norm(compiled_axis)))
        sign = 1 if axis_dot >= 0 else -1
        expected_range = sorted(
            [float(model.qpos0[qpos_address] + sign * (source_range[0] - source_ref)),
             float(model.qpos0[qpos_address] + sign * (source_range[1] - source_ref))]
        )
        records[joint_name] = {
            "source_urdf_lower_upper_rad": source_range,
            "source_urdf_axis": source_axis.tolist(),
            "source_coordinate_neutral_rad": source_ref,
            "mjspec_joint_ref_rad": float(spec_joint.ref),
            "mujoco_joint_axis": compiled_axis.tolist(),
            "mujoco_joint_range_rad": model.jnt_range[joint_id].tolist(),
            "mujoco_joint_limit_solref": model.jnt_solref[joint_id].tolist(),
            "compiled_expected_range_from_mapping_rad": expected_range,
            "qpos_address": qpos_address,
            "qpos0_rad": float(model.qpos0[qpos_address]),
            "axis_dot": axis_dot,
            "source_from_qpos": "source_neutral + sign(axis_dot) * (qpos - qpos0)",
            "mapping_sign": sign,
        }
    return records


def name_for(model: mujoco.MjModel, kind: mujoco.mjtObj, obj_id: int) -> str:
    result = mujoco.mj_id2name(model, kind, obj_id)
    return result if result is not None else f"unnamed_{obj_id}"


def build_model(kind: str, profile: str, output: Path) -> tuple[mujoco.MjModel, list[dict[str, object]], dict[str, dict[str, object]], dict[str, int], dict[str, int]]:
    if kind == "pair":
        urdf_path = output / f"{kind}_only_generated.urdf"
        selected_joints, source_relations = make_subset_urdf(kind, urdf_path)
        spec = mujoco.MjSpec.from_file(str(urdf_path))
    elif kind in ("hands", "robot"):
        _, source_links, source_joints, source_relations = read_source()
        selected_joints = sorted(source_joints)
        spec = mujoco.MjSpec.from_file(str(URDF))
    else:
        raise ValueError(f"unknown model kind: {kind}")
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    if kind in ("hands", "robot"):
        spec.compiler.fusestatic = False
        for joint in spec.joints:
            if joint.name and joint.name.startswith(("L_", "R_")):
                joint.solref_limit = [-10000.0, -200.0]
    if kind == "robot":
        root_body = spec.worldbody.bodies[0]
        root_body.pos = FULL_ROBOT_BASE_POS
        root_body.quat = [
            math.cos(FULL_ROBOT_BASE_YAW / 2.0),
            0.0,
            0.0,
            math.sin(FULL_ROBOT_BASE_YAW / 2.0),
        ]
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0

    spec_joints = {joint.name: joint for joint in spec.joints if joint.name}
    if kind in ("hands", "robot"):
        _, source_links, source_joints, _ = read_source()
        selected_body_names = set(source_links)
        parent_by_child = {
            joint.find("child").get("link"): joint.find("parent").get("link")
            for joint in source_joints.values()
            if joint.find("parent") is not None and joint.find("child") is not None
        }
        exclusion_index = 0
        for descendant, parent in parent_by_child.items():
            if descendant not in selected_body_names:
                continue
            ancestor = parent
            while ancestor:
                if ancestor in selected_body_names:
                    spec.add_exclude(
                        name=f"diagnostic_self_chain_{exclusion_index}",
                        bodyname1=ancestor,
                        bodyname2=descendant,
                    )
                    exclusion_index += 1
                ancestor = parent_by_child.get(ancestor)
    follower_names = set(source_relations)
    relation_records: list[dict[str, object]] = []
    for follower_name, relation in sorted(source_relations.items()):
        driver_name = str(relation["driver_joint"])
        follower = spec_joints[follower_name]
        driver = spec_joints[driver_name]
        follower_axis = np.asarray(relation["follower_axis_urdf"], dtype=float)
        driver_source = read_source()[2][driver_name]
        driver_axis = np.fromstring(driver_source.find("axis").get("xyz", "1 0 0"), sep=" ")
        f_sign = 1 if float(np.dot(follower_axis, np.asarray(follower.axis))) >= 0 else -1
        d_sign = 1 if float(np.dot(driver_axis, np.asarray(driver.axis))) >= 0 else -1
        multiplier = float(relation["multiplier"])
        offset = float(relation["offset"])
        a0 = f_sign * offset
        a1 = f_sign * multiplier * d_sign
        polycoef = [a0, a1, 0.0, 0.0, 0.0]
        equality = spec.add_equality(
            name=f"source_mimic_{follower_name}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=follower_name,
            name2=driver_name,
            data=polycoef + [0.0] * 6,
        )
        profile_config = TEST_PROFILES[profile]
        profile_solref = profile_config["solref"]
        if profile_solref is not None:
            equality.solref = profile_solref
        relation_records.append(
            {
                **relation,
                "mujoco_equality_name": f"source_mimic_{follower_name}",
                "mujoco_joint1_is_follower": True,
                "joint1_name": follower_name,
                "joint2_name": driver_name,
                "polycoef": polycoef,
                "source_ref_adjusted_polycoef": polycoef,
                "driver_axis_sign": d_sign,
                "follower_axis_sign": f_sign,
                "solver_profile": profile,
                "solver_solref": profile_solref,
            }
        )

    driver_names = {str(relation["driver_joint"]) for relation in source_relations.values()}
    arm_joint_names = {
        joint_name
        for joint_name in spec_joints
        if any(part in joint_name.lower() for part in ("shoulder", "elbow", "wrist"))
    }
    if kind == "pair":
        actuator_joint_names = {PAIR[0]}
    elif kind in ("hands", "robot"):
        hand_joint_names = {
            joint_name
            for joint_name, joint in spec_joints.items()
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE
            and joint_name.startswith(("L_", "R_"))
        }
        actuator_joint_names = (driver_names | hand_joint_names | arm_joint_names) - follower_names
    else:
        actuator_joint_names = driver_names
    source_effort: dict[str, float] = {}
    for joint_name in selected_joints:
        source_joint = read_source()[2][joint_name]
        limit = source_joint.find("limit")
        if limit is not None and limit.get("effort") is not None:
            source_effort[joint_name] = float(limit.get("effort"))
    for joint_name in sorted(actuator_joint_names):
        if joint_name in follower_names:
            raise RuntimeError(f"source mimic follower must not get an independent actuator: {joint_name}")
        joint = spec_joints[joint_name]
        effort = source_effort.get(joint_name, 30.0)
        actuator = spec.add_actuator(
            name=f"diagnostic_position_{joint_name}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            ctrllimited=bool(joint.limited),
            ctrlrange=list(joint.range) if joint.limited else [0.0, 0.0],
            forcelimited=True,
            forcerange=[-min(effort, float(profile_config["effort_cap"])), min(effort, float(profile_config["effort_cap"]))],
        )
        if kind in ("hands", "robot") and joint_name in arm_joint_names:
            actuator.set_to_position(kp=150.0, kv=22.0)
        elif kind == "robot" and joint_name in driver_names and joint_name.startswith("L_"):
            actuator.set_to_position(kp=18.0, kv=2.4)
        elif kind in ("hands", "robot") and joint_name not in driver_names:
            actuator.set_to_position(kp=18.0, kv=2.4)
        else:
            actuator.set_to_position(kp=float(profile_config["kp"]), kv=float(profile_config["kv"]))

    model = spec.compile()
    if kind in ("hands", "robot"):
        model.vis.headlight.ambient[:] = [0.30, 0.30, 0.30]
        model.vis.headlight.diffuse[:] = [0.60, 0.60, 0.60]
        model.vis.headlight.specular[:] = [0.0, 0.0, 0.0]
        model.vis.rgba.haze[:] = [0.15, 0.25, 0.35, 1.0]
        vendor_visual_material_by_link: dict[str, str] = {}
        source_root, source_links, _, _ = read_source()
        for link_name, link in source_links.items():
            materials = {
                material.get("name")
                for visual in link.findall("visual")
                if (material := visual.find("material")) is not None and material.get("name")
            }
            if len(materials) == 1:
                vendor_visual_material_by_link[link_name] = next(iter(materials))
        material_rgba = {
            "silver": [0.76, 0.79, 0.82, 1.0],
            "blue": [0.14, 0.34, 0.76, 1.0],
            "brown": [0.46, 0.31, 0.20, 1.0],
            "white": [0.98, 0.98, 0.97, 1.0],
            "green": [0.16, 0.50, 0.30, 1.0],
            "orange": [0.92, 0.43, 0.14, 1.0],
        }
        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) != 1:
                continue
            body_name = name_for(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            material_name = vendor_visual_material_by_link.get(body_name)
            if material_name in material_rgba:
                model.geom_rgba[geom_id] = material_rgba[material_name]
    joint_records = source_joint_records(spec, model, set(selected_joints))
    actuator_by_joint: dict[str, int] = {}
    for joint_name in sorted(actuator_joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = next(
            (aid for aid in range(model.nu) if int(model.actuator_trnid[aid, 0]) == joint_id),
            -1,
        )
        if actuator_id < 0:
            raise RuntimeError(f"missing actuator for {joint_name}")
        actuator_by_joint[joint_name] = actuator_id
    return model, relation_records, joint_records, actuator_by_joint, {
        str(relation["follower_joint"]): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, str(relation["mujoco_equality_name"]))
        for relation in relation_records
    }


def equality_residual(model: mujoco.MjModel, data: mujoco.MjData, relation: dict[str, object]) -> tuple[float, float, float]:
    follower_name = str(relation["follower_joint"])
    driver_name = str(relation["driver_joint"])
    follower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
    driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
    follower_qadr = int(model.jnt_qposadr[follower_id])
    driver_qadr = int(model.jnt_qposadr[driver_id])
    y_delta = float(data.qpos[follower_qadr] - model.qpos0[follower_qadr])
    x_delta = float(data.qpos[driver_qadr] - model.qpos0[driver_qadr])
    a0, a1, a2, a3, a4 = relation["polycoef"]
    polynomial = a0 + a1 * x_delta + a2 * x_delta**2 + a3 * x_delta**3 + a4 * x_delta**4
    return y_delta - polynomial, float(data.qpos[follower_qadr]), float(data.qpos[driver_qadr])


def data_equality_row(model: mujoco.MjModel, data: mujoco.MjData, equality_id: int) -> float | None:
    for row in range(data.nefc):
        if (
            int(data.efc_type[row]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            and int(data.efc_id[row]) == equality_id
        ):
            return float(data.efc_pos[row])
    return None


def run_pair_profiles(output: Path) -> dict[str, dict[str, object]]:
    _, _, source_joints, source_relations = read_source()
    summary: dict[str, dict[str, object]] = {}
    trace_path = output / "single_pair_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for profile in TEST_PROFILES:
            model, relation_records, joint_records, actuator_by_joint, equality_ids = build_model("pair", profile, output)
            relation = next(item for item in relation_records if item["follower_joint"] == PAIR[1])
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            driver = source_joints[PAIR[0]]
            follower = source_joints[PAIR[1]]
            driver_range = joint_records[PAIR[0]]["source_urdf_lower_upper_rad"]
            follower_range = joint_records[PAIR[1]]["source_urdf_lower_upper_rad"]
            driver_qadr = int(joint_records[PAIR[0]]["qpos_address"])
            follower_qadr = int(joint_records[PAIR[1]]["qpos_address"])
            driver_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PAIR[0])
            follower_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PAIR[1])
            driver_vadr = int(model.jnt_dofadr[driver_jid])
            follower_vadr = int(model.jnt_dofadr[follower_jid])
            actuator_id = actuator_by_joint[PAIR[0]]
            eqid = equality_ids[PAIR[1]]
            initial_residual, _, _ = equality_residual(model, data, relation)
            initial_row = data_equality_row(model, data, eqid)
            if initial_row is None:
                raise RuntimeError("MuJoCo did not create the source mimic equality row at reset")
            trace.write(json.dumps({
                "record_type": "reset",
                "profile": profile,
                "time_s": float(data.time),
                "driver_joint": PAIR[0],
                "follower_joint": PAIR[1],
                "joint_records": joint_records,
                "equality": relation,
                "profile_config": TEST_PROFILES[profile],
                "equality_active": bool(model.eq_active0[eqid]),
                "driver_raw_qpos": float(data.qpos[driver_qadr]),
                "follower_raw_qpos": float(data.qpos[follower_qadr]),
                "driver_qpos0": float(model.qpos0[driver_qadr]),
                "follower_qpos0": float(model.qpos0[follower_qadr]),
                "equality_formula_residual_rad": initial_residual,
                "mujoco_efc_pos_rad": initial_row,
                "no_qpos_writes_after_reset": True,
            }, sort_keys=True) + "\n")

            max_abs_error = 0.0
            max_efc_formula_delta = 0.0
            max_limit_violation = 0.0
            min_source_driver = math.inf
            min_source_follower = math.inf
            max_source_driver = -math.inf
            max_source_follower = -math.inf
            nan_seen = False
            sign_inversion_seen = False
            max_abs_qvel = 0.0
            max_abs_actuator_force = 0.0
            step = 0
            segment_steps = [round(seconds / DT) for seconds in SWEEP_SEGMENT_SECONDS]
            previous_fraction = SWEEP_FRACTIONS[0]
            source_driver_neutral = 0.0
            source_follower_neutral = 0.0
            driver_sign = int(joint_records[PAIR[0]]["mapping_sign"])
            follower_sign = int(joint_records[PAIR[1]]["mapping_sign"])
            driver_q0 = float(model.qpos0[driver_qadr])
            follower_q0 = float(model.qpos0[follower_qadr])
            driver_lower, driver_upper = map(float, driver_range)
            follower_lower, follower_upper = map(float, follower_range)
            driver_source = driver.find("axis")
            follower_source = follower.find("axis")
            for segment_index, (end_fraction, count) in enumerate(zip(SWEEP_FRACTIONS, segment_steps)):
                start_fraction = previous_fraction
                for k in range(count):
                    ramp = (k + 1) / count
                    alpha = ramp * ramp * (3.0 - 2.0 * ramp)
                    fraction = start_fraction + (end_fraction - start_fraction) * alpha
                    source_target = driver_lower + fraction * (driver_upper - driver_lower)
                    ctrl_target = driver_q0 + driver_sign * (source_target - source_driver_neutral)
                    data.ctrl[actuator_id] = ctrl_target
                    mujoco.mj_forward(model, data)
                    pre_step_formula, _, _ = equality_residual(model, data, relation)
                    efc_before = data_equality_row(model, data, eqid)
                    if efc_before is not None:
                        max_efc_formula_delta = max(max_efc_formula_delta, abs(efc_before - pre_step_formula))
                    mujoco.mj_step(model, data)
                    step += 1

                    driver_source_coordinate = source_driver_neutral + driver_sign * (float(data.qpos[driver_qadr]) - driver_q0)
                    follower_source_coordinate = source_follower_neutral + follower_sign * (float(data.qpos[follower_qadr]) - follower_q0)
                    expected_follower_from_actual_driver = float(relation["offset"]) + float(relation["multiplier"]) * driver_source_coordinate
                    expected_follower_from_target = float(relation["offset"]) + float(relation["multiplier"]) * source_target
                    source_error = follower_source_coordinate - expected_follower_from_actual_driver
                    formula_error, raw_follower, raw_driver = equality_residual(model, data, relation)
                    max_abs_error = max(max_abs_error, abs(source_error), abs(formula_error))
                    min_source_driver = min(min_source_driver, driver_source_coordinate)
                    min_source_follower = min(min_source_follower, follower_source_coordinate)
                    max_source_driver = max(max_source_driver, driver_source_coordinate)
                    max_source_follower = max(max_source_follower, follower_source_coordinate)
                    driver_joint_range = model.jnt_range[driver_jid]
                    follower_joint_range = model.jnt_range[follower_jid]
                    qpos_violation = max(
                        float(driver_joint_range[0] - data.qpos[driver_qadr]),
                        float(data.qpos[driver_qadr] - driver_joint_range[1]),
                        float(follower_joint_range[0] - data.qpos[follower_qadr]),
                        float(data.qpos[follower_qadr] - follower_joint_range[1]),
                        0.0,
                    )
                    max_limit_violation = max(max_limit_violation, qpos_violation)
                    nan_seen = nan_seen or not bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                    sign_inversion_seen = sign_inversion_seen or (
                        fraction > 0.02
                        and (driver_source_coordinate < -1e-8 or follower_source_coordinate < -1e-8)
                    )
                    max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel))))
                    max_abs_actuator_force = max(max_abs_actuator_force, abs(float(data.actuator_force[actuator_id])))
                    trace.write(json.dumps({
                        "record_type": "step",
                        "profile": profile,
                        "step": step,
                        "time_s": float(data.time),
                        "phase": ("open_hold", "open_hold", "partial_close", "partial_hold", "full_close", "full_hold", "open")[segment_index],
                        "driver_source_target_rad": source_target,
                        "driver_actuator_target_coordinate": float(data.ctrl[actuator_id]),
                        "driver_actuator_force_nm": float(data.actuator_force[actuator_id]),
                        "driver_actuator_target_error_rad": float(data.ctrl[actuator_id] - data.qpos[driver_qadr]),
                        "driver_source_coordinate_from_qpos_rad": driver_source_coordinate,
                        "driver_raw_qpos": raw_driver,
                        "driver_qpos0": driver_q0,
                        "driver_physical_displacement_qpos_minus_qpos0": raw_driver - driver_q0,
                        "driver_source_lower_upper_rad": [driver_lower, driver_upper],
                        "driver_source_axis": np.fromstring(driver_source.get("xyz"), sep=" ").tolist(),
                        "driver_mujoco_joint_axis": model.jnt_axis[driver_jid].tolist(),
                        "driver_mujoco_joint_range_rad": model.jnt_range[driver_jid].tolist(),
                        "driver_mjspec_ref_rad": joint_records[PAIR[0]]["mjspec_joint_ref_rad"],
                        "expected_follower_source_from_driver_target_rad": expected_follower_from_target,
                        "follower_source_coordinate_from_qpos_rad": follower_source_coordinate,
                        "follower_raw_qpos": raw_follower,
                        "follower_qpos0": follower_q0,
                        "follower_physical_displacement_qpos_minus_qpos0": raw_follower - follower_q0,
                        "follower_source_lower_upper_rad": [follower_lower, follower_upper],
                        "follower_source_axis": np.fromstring(follower_source.get("xyz"), sep=" ").tolist(),
                        "follower_mujoco_joint_axis": model.jnt_axis[follower_jid].tolist(),
                        "follower_mujoco_joint_range_rad": model.jnt_range[follower_jid].tolist(),
                        "follower_mjspec_ref_rad": joint_records[PAIR[1]]["mjspec_joint_ref_rad"],
                        "expected_follower_source_from_actual_driver_rad": expected_follower_from_actual_driver,
                        "source_relation_signed_error_rad": source_error,
                        "mujoco_equality_formula_signed_residual_rad": formula_error,
                        "mujoco_efc_pos_pre_step_rad": efc_before,
                        "mujoco_formula_residual_pre_step_rad": pre_step_formula,
                        "joint_limit_violation_rad": qpos_violation,
                        "qvel_driver_rad_s": float(data.qvel[driver_vadr]),
                        "qvel_follower_rad_s": float(data.qvel[follower_vadr]),
                        "contacts": int(data.ncon),
                    }, sort_keys=True) + "\n")
                previous_fraction = end_fraction

            passes = (
                max_abs_error <= RELATION_TOLERANCE
                and max_limit_violation == 0.0
                and not nan_seen
                and not sign_inversion_seen
                and max_efc_formula_delta <= 1e-10
            )
            summary[profile] = {
                "driver_joint": PAIR[0],
                "follower_joint": PAIR[1],
                "max_runtime_relation_error_rad": max_abs_error,
                "gate_tolerance_rad": RELATION_TOLERANCE,
                "max_joint_limit_violation_rad": max_limit_violation,
                "nan_seen": nan_seen,
                "sign_inversion_seen": sign_inversion_seen,
                "max_abs_qvel_rad_s": max_abs_qvel,
                "max_abs_actuator_force_nm": max_abs_actuator_force,
                "max_abs_difference_manual_formula_to_mujoco_efc_rad": max_efc_formula_delta,
                "source_driver_min_max_rad": [min_source_driver, max_source_driver],
                "source_follower_min_max_rad": [min_source_follower, max_source_follower],
                "steps": step,
                "contacts": 0,
                "initial_equality_residual_rad": initial_residual,
                "initial_equality_efc_pos_rad": initial_row,
                "pass": passes,
                "joint_records": joint_records,
                "equality": relation,
            }
            print(json.dumps({"profile": profile, **summary[profile]}, sort_keys=True))
    return summary


def driver_open_and_close_targets(
    relations: list[dict[str, object]],
    joint_records: dict[str, dict[str, object]],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    driver_names = {str(item["driver_joint"]) for item in relations}
    for driver_name in sorted(driver_names):
        record = joint_records[driver_name]
        feasible_lower, feasible_upper = map(float, record["source_urdf_lower_upper_rad"])
        for relation in relations:
            if relation["driver_joint"] != driver_name:
                continue
            follower_record = joint_records[str(relation["follower_joint"])]
            follower_lower, follower_upper = map(float, follower_record["source_urdf_lower_upper_rad"])
            multiplier = float(relation["multiplier"])
            offset = float(relation["offset"])
            if multiplier > 0:
                feasible_lower = max(feasible_lower, (follower_lower - offset) / multiplier)
                feasible_upper = min(feasible_upper, (follower_upper - offset) / multiplier)
            elif multiplier < 0:
                feasible_lower = max(feasible_lower, (follower_upper - offset) / multiplier)
                feasible_upper = min(feasible_upper, (follower_lower - offset) / multiplier)
            elif not follower_lower <= offset <= follower_upper:
                raise ValueError(f"zero-multiplier follower offset is outside limits: {relation['follower_joint']}")
        neutral = min(max(0.0, feasible_lower), feasible_upper)
        if not feasible_lower - 1e-9 <= neutral <= feasible_upper + 1e-9:
            raise ValueError(f"source zero is outside the driver range: {driver_name}")
        close_endpoint = (
            feasible_upper
            if feasible_upper - neutral >= neutral - feasible_lower
            else feasible_lower
        )
        if math.isclose(close_endpoint, neutral, abs_tol=1e-9):
            raise ValueError(f"no usable source-coordinate close range for {driver_name}")
        result[driver_name] = (
            neutral + 0.01 * (close_endpoint - neutral),
            neutral + 0.90 * (close_endpoint - neutral),
        )
    return result


def interior_neutral_target(record: dict[str, object]) -> float:
    lower, upper = map(float, record["source_urdf_lower_upper_rad"])
    neutral = min(max(float(record["source_coordinate_neutral_rad"]), lower), upper)
    margin = 0.01 * (upper - lower)
    if math.isclose(neutral, lower, abs_tol=1e-9):
        return lower + margin
    if math.isclose(neutral, upper, abs_tol=1e-9):
        return upper - margin
    return neutral


def make_diagnostic_camera(model: mujoco.MjModel, data: mujoco.MjData, kind: str) -> mujoco.MjvCamera:
    selected = np.flatnonzero(model.geom_group == 2)
    geom_positions = np.asarray(data.geom_xpos[selected], dtype=float)
    if len(geom_positions):
        lookat = (geom_positions.min(axis=0) + geom_positions.max(axis=0)) / 2.0
        extent = float(np.max(np.ptp(geom_positions, axis=0)))
    else:
        lookat = np.zeros(3)
        extent = 0.0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = max(0.65, extent * 1.7) if kind == "hands" else 1.9
    camera.azimuth = 0
    camera.elevation = -12
    return camera


def run_relation_sweep(
    kind: str,
    profile: str,
    output: Path,
    trace_name: str,
    *,
    right_only: bool = False,
    video_name: str | None = None,
) -> dict[str, object]:
    model, relations, joint_records, actuator_by_joint, equality_ids = build_model(kind, profile, output)
    relation_by_follower = {str(relation["follower_joint"]): relation for relation in relations}
    open_close_targets = driver_open_and_close_targets(relations, joint_records)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model_reset_qpos = data.qpos.copy()
    initial_source_pose: dict[str, float] = {}
    for joint_name in actuator_by_joint:
        record = joint_records[joint_name]
        if joint_name in open_close_targets:
            initial_source_pose[joint_name] = open_close_targets[joint_name][0]
        else:
            initial_source_pose[joint_name] = interior_neutral_target(record)
    for relation in relations:
        driver_name = str(relation["driver_joint"])
        follower_name = str(relation["follower_joint"])
        initial_source_pose[follower_name] = float(relation["offset"]) + float(
            relation["multiplier"]
        ) * initial_source_pose[driver_name]
    for joint_name, source_coordinate in initial_source_pose.items():
        record = joint_records[joint_name]
        data.qpos[int(record["qpos_address"])] = float(record["qpos0_rad"]) + int(
            record["mapping_sign"]
        ) * (source_coordinate - float(record["source_coordinate_neutral_rad"]))
    for joint_name, actuator_id in actuator_by_joint.items():
        record = joint_records[joint_name]
        source_open = initial_source_pose[joint_name]
        data.ctrl[actuator_id] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
            source_open - float(record["source_coordinate_neutral_rad"])
        )
    mujoco.mj_forward(model, data)

    renderer = None
    writer = None
    render_camera = None
    render_option = None
    if video_name:
        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) != 1:
                continue
            body_name = name_for(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            if body_name.startswith(("L_", "R_")):
                model.geom_group[geom_id] = 2
        render_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(render_option)
        render_option.geomgroup[:] = 0
        render_option.geomgroup[2] = 1
        render_option.sitegroup[:] = 0
        render_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        renderer = mujoco.Renderer(model, height=720, width=1280)
        writer = imageio.get_writer(output / video_name, fps=15, codec="libx264", quality=8)
        render_camera = make_diagnostic_camera(model, data, kind)

    trace_path = output / trace_name
    max_error = 0.0
    max_efc_delta = 0.0
    max_limit_violation = 0.0
    max_limit_violation_details: dict[str, object] | None = None
    max_abs_force = 0.0
    first_gate_failure: dict[str, object] | None = None
    nan_seen = False
    sign_inversion_seen = False
    max_contacts = int(data.ncon)
    segment_steps = [round(seconds / DT) for seconds in SWEEP_SEGMENT_SECONDS]
    step = 0
    previous_fraction = SWEEP_FRACTIONS[0]
    driven_names = {
        name for name in open_close_targets if not right_only or name.startswith("R_")
    }
    with trace_path.open("w", encoding="utf-8") as trace:
        reset_relations: dict[str, dict[str, object]] = {}
        for follower_name, relation in relation_by_follower.items():
            error, follower_raw, driver_raw = equality_residual(model, data, relation)
            efc = data_equality_row(model, data, equality_ids[follower_name])
            driver_record = joint_records[str(relation["driver_joint"])]
            follower_record = joint_records[follower_name]
            driver_source = float(driver_record["source_coordinate_neutral_rad"]) + int(
                driver_record["mapping_sign"]
            ) * (driver_raw - float(driver_record["qpos0_rad"]))
            follower_source = float(follower_record["source_coordinate_neutral_rad"]) + int(
                follower_record["mapping_sign"]
            ) * (follower_raw - float(follower_record["qpos0_rad"]))
            reset_relations[follower_name] = {
                "driver_joint": relation["driver_joint"],
                "driver_raw_qpos_rad": driver_raw,
                "follower_raw_qpos_rad": follower_raw,
                "driver_source_coordinate_rad": driver_source,
                "follower_source_coordinate_rad": follower_source,
                "source_expected_follower_rad": float(relation["offset"])
                + float(relation["multiplier"]) * driver_source,
                "mujoco_formula_residual_rad": error,
                "mujoco_efc_pos_rad": efc,
            }
        trace.write(json.dumps({
            "record_type": "reset",
            "model_kind": kind,
            "profile": profile,
            "time_s": float(data.time),
            "relation_count": len(relations),
            "model_reset_qpos_before_initial_pose": model_reset_qpos.tolist(),
            "joint_records": joint_records,
            "relations": reset_relations,
            "initial_pose_qpos_seeded_before_rollout": True,
            "qpos_writes_after_rollout_start": 0,
            "geom_contacts_disabled": True,
        }, sort_keys=True) + "\n")
        if writer is not None and renderer is not None and render_camera is not None and render_option is not None:
            renderer.update_scene(data, camera=render_camera, scene_option=render_option)
            imageio.imwrite(output / "hand_only_open.png", renderer.render())
            writer.append_data(renderer.render())

        for segment_index, (end_fraction, count) in enumerate(zip(SWEEP_FRACTIONS, segment_steps)):
            start_fraction = previous_fraction
            for k in range(count):
                ramp = (k + 1) / count
                alpha = ramp * ramp * (3.0 - 2.0 * ramp)
                fraction = start_fraction + (end_fraction - start_fraction) * alpha
                source_targets: dict[str, float] = {}
                for joint_name, actuator_id in actuator_by_joint.items():
                    record = joint_records[joint_name]
                    if joint_name in open_close_targets:
                        source_open, source_close = open_close_targets[joint_name]
                        if segment_index < 2:
                            source_target = source_open
                        else:
                            driver_fraction = fraction if joint_name in driven_names else 0.0
                            source_target = source_open + driver_fraction * (source_close - source_open)
                    else:
                        source_target = initial_source_pose[joint_name]
                    source_targets[joint_name] = source_target
                    data.ctrl[actuator_id] = float(record["qpos0_rad"]) + int(record["mapping_sign"]) * (
                        source_target - float(record["source_coordinate_neutral_rad"])
                    )

                # Capture MuJoCo's equality constraint residual at this exact pre-step state.
                mujoco.mj_forward(model, data)
                pre_step_efc: dict[str, float | None] = {}
                pre_step_formula: dict[str, float] = {}
                for follower_name, relation in relation_by_follower.items():
                    pre_step_efc[follower_name] = data_equality_row(model, data, equality_ids[follower_name])
                    pre_step_formula[follower_name] = equality_residual(model, data, relation)[0]
                    if pre_step_efc[follower_name] is not None:
                        max_efc_delta = max(
                            max_efc_delta,
                            abs(float(pre_step_efc[follower_name]) - pre_step_formula[follower_name]),
                        )
                mujoco.mj_step(model, data)
                step += 1
                current_relations: dict[str, dict[str, object]] = {}
                for follower_name, relation in relation_by_follower.items():
                    driver_name = str(relation["driver_joint"])
                    error, follower_raw, driver_raw = equality_residual(model, data, relation)
                    driver_record = joint_records[driver_name]
                    follower_record = joint_records[follower_name]
                    driver_source = float(driver_record["source_coordinate_neutral_rad"]) + int(
                        driver_record["mapping_sign"]
                    ) * (driver_raw - float(driver_record["qpos0_rad"]))
                    follower_source = float(follower_record["source_coordinate_neutral_rad"]) + int(
                        follower_record["mapping_sign"]
                    ) * (follower_raw - float(follower_record["qpos0_rad"]))
                    expected_source = float(relation["offset"]) + float(relation["multiplier"]) * driver_source
                    source_error = follower_source - expected_source
                    max_error = max(max_error, abs(error), abs(source_error))
                    driver_source_target = source_targets[driver_name]
                    target_direction = driver_source_target - float(driver_record["source_coordinate_neutral_rad"])
                    actual_direction = driver_source - float(driver_record["source_coordinate_neutral_rad"])
                    if abs(target_direction) > 1e-6:
                        sign_inversion_seen = sign_inversion_seen or (
                            actual_direction * target_direction < -1e-8
                        )
                    current_relations[follower_name] = {
                        "driver_joint": driver_name,
                        "driver_source_coordinate_rad": driver_source,
                        "driver_source_target_rad": driver_source_target,
                        "driver_raw_qpos_rad": driver_raw,
                        "driver_qpos0_rad": float(driver_record["qpos0_rad"]),
                        "driver_qpos_minus_qpos0_rad": driver_raw - float(driver_record["qpos0_rad"]),
                        "driver_mjspec_ref_rad": float(driver_record["mjspec_joint_ref_rad"]),
                        "driver_urdf_range_rad": driver_record["source_urdf_lower_upper_rad"],
                        "driver_axis_urdf": driver_record["source_urdf_axis"],
                        "driver_axis_mujoco": driver_record["mujoco_joint_axis"],
                        "follower_source_coordinate_rad": follower_source,
                        "follower_expected_source_coordinate_rad": expected_source,
                        "follower_raw_qpos_rad": follower_raw,
                        "follower_qpos0_rad": float(follower_record["qpos0_rad"]),
                        "follower_qpos_minus_qpos0_rad": follower_raw - float(follower_record["qpos0_rad"]),
                        "follower_mjspec_ref_rad": float(follower_record["mjspec_joint_ref_rad"]),
                        "follower_urdf_range_rad": follower_record["source_urdf_lower_upper_rad"],
                        "follower_axis_urdf": follower_record["source_urdf_axis"],
                        "follower_axis_mujoco": follower_record["mujoco_joint_axis"],
                        "source_relation_error_rad": source_error,
                        "mujoco_equality_formula_residual_rad": error,
                        "mujoco_efc_pos_pre_step_rad": pre_step_efc[follower_name],
                    }
                    if abs(error) > RELATION_TOLERANCE and first_gate_failure is None:
                        first_gate_failure = {
                            "step": step,
                            "time_s": float(data.time),
                            "follower_joint": follower_name,
                            "error_rad": error,
                        }

                limit_violation = 0.0
                limit_violation_joint: dict[str, object] | None = None
                for joint_name, record in joint_records.items():
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    qadr = int(record["qpos_address"])
                    joint_range = model.jnt_range[joint_id]
                    qpos = float(data.qpos[qadr])
                    joint_violation = max(
                        float(joint_range[0] - qpos),
                        float(qpos - joint_range[1]),
                        0.0,
                    )
                    if joint_violation > limit_violation:
                        limit_violation = joint_violation
                        limit_violation_joint = {
                            "joint": joint_name,
                            "qpos_rad": qpos,
                            "range_rad": joint_range.tolist(),
                        }
                if limit_violation > max_limit_violation:
                    max_limit_violation = limit_violation
                    max_limit_violation_details = {
                        "step": step,
                        "time_s": float(data.time),
                        **(limit_violation_joint or {}),
                    }
                nan_seen = nan_seen or not bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                max_contacts = max(max_contacts, int(data.ncon))
                max_abs_force = max(
                    max_abs_force,
                    max((abs(float(data.actuator_force[aid])) for aid in actuator_by_joint.values()), default=0.0),
                )
                trace.write(json.dumps({
                    "record_type": "step",
                    "model_kind": kind,
                    "profile": profile,
                    "step": step,
                    "time_s": float(data.time),
                    "phase": ("open_hold", "open_hold", "partial_close", "partial_hold", "full_close", "full_hold", "open")[segment_index],
                    "global_close_fraction": fraction,
                    "relations": current_relations,
                    "max_relation_error_rad": max(abs(float(r["mujoco_equality_formula_residual_rad"])) for r in current_relations.values()),
                    "max_joint_limit_violation_rad": limit_violation,
                    "max_joint_limit_violation_joint": limit_violation_joint,
                    "nan_seen": nan_seen,
                    "sign_inversion_seen": sign_inversion_seen,
                    "max_abs_actuator_force_nm": max_abs_force,
                    "contacts": int(data.ncon),
                    "qpos_written_during_rollout": False,
                }, sort_keys=True) + "\n")
                if writer is not None and renderer is not None and render_camera is not None and render_option is not None and (step % 80 == 0 or (segment_index == 5 and k == 0)):
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=render_camera, scene_option=render_option)
                    frame = renderer.render()
                    writer.append_data(frame)
                    if segment_index == 5 and k == 0:
                        imageio.imwrite(output / "hand_only_closed.png", frame)
            previous_fraction = end_fraction
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()

    summary = {
        "model_kind": kind,
        "profile": profile,
        "relation_count": len(relations),
        "steps": step,
        "simulation_duration_s": step * DT,
        "max_runtime_relation_error_rad": max_error,
        "gate_tolerance_rad": RELATION_TOLERANCE,
        "max_mujoco_efc_pos_at_pre_step_states_rad": max_efc_delta,
        "max_joint_limit_violation_rad": max_limit_violation,
        "max_joint_limit_violation_details": max_limit_violation_details,
        "nan_seen": nan_seen,
        "sign_inversion_seen": sign_inversion_seen,
        "max_abs_actuator_force_nm": max_abs_force,
        "max_contacts": max_contacts,
        "geom_contacts_disabled": True,
        "initial_pose_qpos_seeded_before_rollout": True,
        "qpos_writes_after_rollout_start": 0,
        "driver_targets_source_rad": {
            name: list(targets) for name, targets in open_close_targets.items()
        },
        "held_nonmimic_hand_targets_source_rad": {
            name: interior_neutral_target(joint_records[name])
            for name in actuator_by_joint
            if name not in open_close_targets and name.startswith(("L_", "R_"))
        },
        "right_hand_only_actuated": right_only,
        "actuated_joint_names": sorted(actuator_by_joint),
        "nonmimic_hand_hold_gain": {"kp": 18.0, "kv": 2.4},
        "passive_left_driver_hold_gain": {"kp": 18.0, "kv": 2.4} if kind == "robot" else None,
        "joint_limit_solver_profile": "direct [-10000, -200] on source L_/R_ revolute joints",
        "first_gate_failure": first_gate_failure,
        "pass": (
            len(relations) == 12
            and max_error <= RELATION_TOLERANCE
            and max_limit_violation == 0.0
            and not nan_seen
            and not sign_inversion_seen
            and max_contacts == 0
        ),
        "video": str(output / video_name) if video_name else None,
        "trace": str(trace_path),
    }
    return summary


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"evidence directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    if not URDF.is_file():
        raise FileNotFoundError(URDF)
    git_metadata = {}
    try:
        git_metadata = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        }
    except (OSError, subprocess.CalledProcessError):
        git_metadata = {"head": "unavailable", "branch": "unavailable", "status": "unavailable"}
    runtime_identity = {
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "urdf": str(URDF),
        "urdf_sha256": sha256(URDF),
        "vendor_pin": SOURCE_PIN,
        "robot_sim_git": git_metadata,
        "timestep_s": DT,
        "contact_policy": "all diagnostic geom contacts disabled; no bottle or table in diagnostic models",
    }
    (OUT / "runtime_identity.json").write_text(json.dumps(runtime_identity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**runtime_identity, "output": str(OUT)}, sort_keys=True))

    pair_summaries = run_pair_profiles(OUT)
    (OUT / "pair_summary.json").write_text(json.dumps(pair_summaries, indent=2) + "\n", encoding="utf-8")
    pair_pass = any(item.get("pass") for item in pair_summaries.values())
    if not pair_pass:
        print(json.dumps({"stage": "single_pair", "pass": False, "summary": pair_summaries}, sort_keys=True))
        return 2

    selected_profile = "direct__moderate_controller"
    all_hand_summary = run_relation_sweep(
        "hands",
        selected_profile,
        OUT,
        "all_12_mimic_trace.jsonl",
        video_name="hand_only.mp4",
    )
    (OUT / "all_12_summary.json").write_text(json.dumps(all_hand_summary, indent=2) + "\n", encoding="utf-8")
    if not all_hand_summary["pass"]:
        print(json.dumps({"stage": "all_12_hand_only", "pass": False, "summary": all_hand_summary}, sort_keys=True))
        return 3

    integrated_summary = run_relation_sweep(
        "robot",
        selected_profile,
        OUT,
        "integrated_x2_free_space_trace.jsonl",
        right_only=True,
    )
    (OUT / "integrated_x2_summary.json").write_text(json.dumps(integrated_summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "experiment_status.json").write_text(json.dumps({
        "single_pair_any_profile_pass": pair_pass,
        "all_12_hand_only_pass": all_hand_summary["pass"],
        "integrated_x2_free_space_pass": integrated_summary["pass"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", "pair_pass": pair_pass, "all_12": all_hand_summary, "integrated_x2": integrated_summary}, sort_keys=True))
    return 0 if all_hand_summary["pass"] and integrated_summary["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
