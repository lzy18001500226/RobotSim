#!/usr/bin/env python3
"""Static source-STL grasp feasibility preflight for the horizontal X2 cylinder.

This tool builds a separate full-X2 scene that preserves the source table,
cylinder, robot, and approved STL hand package. It performs mj_forward scans
only; it never calls mj_step and never alters the frozen model assets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

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
RIGHT_HAND_BODIES = {
    "R_omnipicker_base_link",
    "R_hand_narrow1_Link",
    "R_hand_narrow2_Link",
    "R_hand_narrow3_Link",
    "R_hand_narrow4_Link",
    "R_hand_narrow_loop_Link",
    "R_hand_wide1_Link",
    "R_hand_wide2_Link",
    "R_hand_wide3_Link",
    "R_hand_wide4_Link",
    "R_hand_wide_loop_Link",
}
RIGHT_HAND_JOINTS = {
    "right_claw_joint",
    "R_hand_narrow2_joint",
    "R_hand_narrow3_joint",
    "R_hand_narrow_loop_joint",
    "R_hand_wide1_joint",
    "R_hand_wide2_joint",
    "R_hand_wide3_joint",
    "R_hand_wide_loop_joint",
}
STL_EXCLUSIONS = (
    ("R_hand_narrow1_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow2_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_wide1_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide2_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide3_Link", "R_hand_wide_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_wide3_Link"),
)
MESH_RENAMES = {
    "mesh_base_link": "base_link",
    "mesh_narrow1_Link": "narrow1_Link",
    "mesh_narrow2_Link": "narrow2_Link",
    "mesh_narrow3_Link": "narrow3_Link",
    "mesh_narrow_loop_Link": "narrow_loop_Link",
    "mesh_wide1_Link": "wide1_Link",
    "mesh_wide2_Link": "wide2_Link",
    "mesh_wide3_Link": "wide3_Link",
    "mesh_wide_loop_Link": "wide_loop_Link",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"missing body: {name}")


def remap_candidate_meshes(element: ET.Element) -> None:
    for geom in element.iter("geom"):
        mesh = geom.get("mesh")
        if mesh in MESH_RENAMES:
            geom.set("mesh", MESH_RENAMES[mesh])
        if "__visual" in geom.get("name", ""):
            geom.set("group", "2")
        elif "__collision" in geom.get("name", ""):
            geom.set("group", "3")


def references_right_hand(element: ET.Element) -> bool:
    values = " ".join(element.attrib.values())
    return "R_hand_" in values or "right_claw_joint" in values or values.startswith("candidate_")


def compose_stl_scene(
    source_scene: Path,
    stl_model: Path,
    destination: Path,
    *,
    horizontal_cylinder: bool = False,
    table_z_offset_m: float = 0.0,
    cylinder_axis_offset_y_m: float = 0.0,
) -> dict[str, object]:
    tree = ET.parse(source_scene)
    root = tree.getroot()
    stl = ET.parse(stl_model).getroot()
    target = find_body(root, "R_omnipicker_base_link")
    source = find_body(stl, "R_omnipicker_base_link")
    mount = {key: target.get(key) for key in ("name", "pos", "quat", "childclass")}
    for child in list(target):
        target.remove(child)
    for child in list(source):
        cloned = copy.deepcopy(child)
        remap_candidate_meshes(cloned)
        target.append(cloned)
    for key, value in mount.items():
        if value is not None:
            target.set(key, value)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for item in list(contact):
        if item.tag == "exclude" and item.get("body1") in RIGHT_HAND_BODIES and item.get("body2") in RIGHT_HAND_BODIES:
            contact.remove(item)
    for body1, body2 in STL_EXCLUSIONS:
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)

    for name in ("equality", "actuator", "sensor", "tendon"):
        node = root.find(name)
        if node is None:
            continue
        for child in list(node):
            if references_right_hand(child):
                node.remove(child)
        if not list(node):
            root.remove(node)
    cylinder_pose = None
    if horizontal_cylinder:
        table = find_body(root, "raicom_table")
        table_position = [float(value) for value in table.get("pos", "").split()]
        if len(table_position) != 3:
            raise RuntimeError("RAICOM table position is invalid")
        table_position[2] += float(table_z_offset_m)
        table.set("pos", " ".join(f"{value:.12g}" for value in table_position))
        cylinder = find_body(root, "test_cylinder")
        geom = next((node for node in cylinder.findall("geom") if node.get("name") == "test_cylinder_collision"), None)
        if geom is None or geom.get("type") != "cylinder":
            raise RuntimeError("source scene lacks the expected primitive test cylinder")
        radius, halfheight = (float(value) for value in geom.get("size", "").split())
        position = [float(value) for value in cylinder.get("pos", "").split()]
        if len(position) != 3:
            raise RuntimeError("test cylinder position is invalid")
        tabletop_z = position[2] - halfheight + float(table_z_offset_m)
        position[1] += float(cylinder_axis_offset_y_m)
        position[2] = tabletop_z + radius
        cylinder.set("pos", " ".join(f"{value:.12g}" for value in position))
        # Rotate source cylinder z-axis onto world y-axis; mesh, size, mass and friction stay unchanged.
        cylinder.set("quat", "0.7071067811865476 0.7071067811865475 0 0")
        cylinder_pose = {"axis_world": "-Y", "tabletop_z_m": tabletop_z, "center_world_m": position, "radius_m": radius, "half_length_m": halfheight, "table_z_offset_m": float(table_z_offset_m), "axis_offset_y_m": float(cylinder_axis_offset_y_m)}
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {"right_mount": mount, "right_exclusions": [list(item) for item in STL_EXCLUSIONS], "horizontal_cylinder": cylinder_pose}


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise RuntimeError(f"scene lacks required joint: {name}")
    data.qpos[model.jnt_qposadr[joint]] = float(value)
    data.qvel[model.jnt_dofadr[joint]] = 0.0


def left_name(right_name: str) -> str:
    if right_name == "right_claw_joint":
        return "left_claw_joint"
    return right_name.replace("R_", "L_", 1)


def apply_prior_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: dict[str, object],
    arm_override: np.ndarray | None = None,
    motor_override: dict[str, float] | None = None,
) -> None:
    targets = dict(state["motor_targets_rad"])
    if motor_override is not None:
        targets.update(motor_override)
    if arm_override is not None:
        targets.update(dict(zip(ARM_JOINTS, arm_override)))
    for name, value in targets.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, name, float(value))
    for right, value in state["left_hand_qpos_rad"].items():
        name = left_name(right)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, name, float(value))


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


def classify_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if cylinder < 0:
        raise RuntimeError("scene has no test_cylinder")
    pads = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
    pad_contacts = []
    nonpad_contacts = []
    table_contacts = []
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        names = (body_name(model, body1), body_name(model, body2))
        record = {
            "geom_names": (geom_name(model, geom1), geom_name(model, geom2)),
            "body_names": names,
            "distance_m": float(contact.dist),
        }
        records.append(record)
        if cylinder not in (body1, body2):
            continue
        other = names[1] if body1 == cylinder else names[0]
        if other in pads:
            pad_contacts.append(record)
        elif other.startswith("R_hand_"):
            nonpad_contacts.append(record)
        elif other == "raicom_table":
            table_contacts.append(record)
    return {
        "records": records,
        "pad_bodies": sorted({body for item in pad_contacts for body in item["body_names"] if body in pads}),
        "pad_contacts": pad_contacts,
        "nonpad_right_contacts": nonpad_contacts,
        "table_contacts": table_contacts,
    }


def state_by_phase(config: dict[str, object], phase: str) -> dict[str, object]:
    for state in config["static_states"]:
        if state.get("phase") == phase:
            return state
    raise RuntimeError(f"prior config lacks phase: {phase}")


def evaluate_pose(
    model,
    trajectory,
    state,
    percent: int,
    *,
    cylinder_lift_m: float = 0.0,
    arm_override: np.ndarray | None = None,
    motor_override: dict[str, float] | None = None,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    apply_prior_state(model, data, state, arm_override, motor_override)
    if cylinder_lift_m:
        cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
        if cylinder_joint < 0:
            raise RuntimeError("scene lacks free cylinder joint")
        address = int(model.jnt_qposadr[cylinder_joint])
        data.qpos[address + 2] += float(cylinder_lift_m)
    pose = apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(percent))
    contacts = classify_contacts(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    return {
        "command_percent": percent,
        "cylinder_lift_m": float(cylinder_lift_m),
        "gripper_base_world_m": data.xpos[base].tolist(),
        "gripper_base_world_xmat": data.xmat[base].tolist(),
        "right_qpos_rad": pose.positions_rad,
        "bilateral_pad_contact": set(contacts["pad_bodies"]) == {"R_hand_narrow3_Link", "R_hand_wide3_Link"},
        "pad_bodies": contacts["pad_bodies"],
        "pad_contact_count": len(contacts["pad_contacts"]),
        "nonpad_right_contact_count": len(contacts["nonpad_right_contacts"]),
        "table_contact_count": len(contacts["table_contacts"]),
        "nonpad_right_contacts": contacts["nonpad_right_contacts"],
        "pad_contacts": contacts["pad_contacts"],
        "all_contact_count": len(contacts["records"]),
    }


def rotation_error_world(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    return axis * (0.5 * angle / max(np.sin(angle), 1e-12))


def arm_vector(model: mujoco.MjModel, state: dict[str, object]) -> np.ndarray:
    values = []
    for joint in ARM_JOINTS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) < 0:
            raise RuntimeError(f"missing arm joint: {joint}")
        values.append(float(state["motor_targets_rad"][joint]))
    return np.asarray(values, dtype=float)


def solve_gripper_translation_ik(
    model,
    trajectory,
    state,
    percent: int,
    translation_world_m: np.ndarray,
    *,
    cylinder_lift_m: float,
    start_arm_override: np.ndarray | None = None,
    rotation_axis_world: np.ndarray | None = None,
    rotation_angle_rad: float = 0.0,
    rotation_matrix_world: np.ndarray | None = None,
    max_iterations: int = 400,
) -> dict[str, object]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    if body_id < 0:
        raise RuntimeError("scene lacks right OmniPicker base body")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    start = arm_vector(model, state) if start_arm_override is None else np.asarray(start_arm_override, dtype=float).copy()
    reference = mujoco.MjData(model)
    apply_prior_state(model, reference, state, start)
    apply_stl_kinematic_pose(mujoco, model, reference, trajectory, float(percent))
    target_pos = reference.xpos[body_id].copy() + np.asarray(translation_world_m, dtype=float)
    target_mat = reference.xmat[body_id].reshape(3, 3).copy()
    applied_rotation = np.eye(3)
    if rotation_axis_world is not None or abs(float(rotation_angle_rad)) > 0.0:
        if rotation_axis_world is None or not math.isfinite(float(rotation_angle_rad)):
            raise ValueError("a finite rotation axis and angle are required together")
        axis = np.asarray(rotation_axis_world, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or not np.isfinite(axis).all() or axis_norm <= 1e-12:
            raise ValueError("rotation axis must be a finite non-zero world vector")
        axis /= axis_norm
        cross = np.array(
            ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
            dtype=float,
        )
        applied_rotation = np.eye(3) + math.sin(rotation_angle_rad) * cross + (1.0 - math.cos(rotation_angle_rad)) * (cross @ cross)
    if rotation_matrix_world is not None:
        matrix = np.asarray(rotation_matrix_world, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8):
            raise ValueError("rotation matrix must be a finite proper orthonormal world rotation")
        applied_rotation = matrix @ applied_rotation
    target_mat = applied_rotation @ target_mat
    q = start.copy()
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in ARM_JOINTS]
    bounds = []
    for name in ARM_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not bool(model.jnt_limited[joint]):
            bounds.append((-np.inf, np.inf))
        else:
            bounds.append(tuple(float(x) for x in model.jnt_range[joint]))
    iterations = 0
    final_data = None
    position_error = float("inf")
    orientation_error = float("inf")
    for iterations in range(1, max_iterations + 1):
        data = mujoco.MjData(model)
        apply_prior_state(model, data, state, q)
        cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
        address = int(model.jnt_qposadr[cylinder_joint])
        data.qpos[address + 2] += cylinder_lift_m
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(percent))
        current_pos = data.xpos[body_id]
        current_mat = data.xmat[body_id].reshape(3, 3)
        pos_error = target_pos - current_pos
        rot_error = rotation_error_world(target_mat, current_mat)
        position_error = float(np.linalg.norm(pos_error))
        orientation_error = float(np.linalg.norm(rot_error))
        final_data = data
        if position_error <= 1e-5 and orientation_error <= 1e-4:
            break
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body_id)
        jacobian = np.vstack((jac_pos[:, dofs], 0.10 * jac_rot[:, dofs]))
        error = np.concatenate((pos_error, 0.10 * rot_error))
        damp = 1e-5
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damp * np.eye(6), error)
        delta = np.clip(delta, -0.05, 0.05)
        q += delta
        for index, (low, high) in enumerate(bounds):
            q[index] = float(np.clip(q[index], low + 1e-5, high - 1e-5))
    if final_data is None:
        raise RuntimeError("IK did not initialize")
    final = evaluate_pose(model, trajectory, state, percent, cylinder_lift_m=cylinder_lift_m, arm_override=q)
    final["arm_q_rad"] = {name: float(value) for name, value in zip(ARM_JOINTS, q)}
    final["iterations"] = iterations
    final["gripper_base_position_error_m"] = position_error
    final["gripper_base_orientation_error_rad"] = orientation_error
    final["source_arm_q_rad"] = {name: float(value) for name, value in zip(ARM_JOINTS, start)}
    final["target_rotation_axis_world"] = None if rotation_axis_world is None else axis.tolist()
    final["target_rotation_angle_rad"] = float(rotation_angle_rad)
    final["target_rotation_matrix_world"] = applied_rotation.tolist()
    final["pass"] = bool(
        position_error <= 1e-4
        and orientation_error <= 1e-3
        and final["bilateral_pad_contact"]
        and not final["nonpad_right_contact_count"]
        and not final["table_contact_count"]
    )
    return final


def static_holding_effort(model, trajectory, state, command_percent: int, arm_q: np.ndarray) -> dict[str, object]:
    data = mujoco.MjData(model)
    apply_prior_state(model, data, state, arm_q)
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command_percent))
    inverse = mujoco.MjData(model)
    inverse.qpos[:] = data.qpos
    inverse.qvel[:] = 0.0
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0.0
    mujoco.mj_inverse(model, inverse)
    values = {}
    for actuator in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or f"actuator#{actuator}"
        joint = int(model.actuator_trnid[actuator, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if not name.startswith("motor_"):
            continue
        dof = int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actuator, 0])
        required = float(inverse.qfrc_inverse[dof] / gear)
        values[name] = {
            "joint": joint_name,
            "required_ctrl": required,
            "range": [float(x) for x in model.actuator_ctrlrange[actuator]],
            "qfrc_bias_Nm": float(data.qfrc_bias[dof]),
            "qfrc_constraint_Nm": float(data.qfrc_constraint[dof]),
        }
    wrist = values.get("motor_right_wrist_roll_joint")
    if wrist is None:
        raise RuntimeError("right wrist roll actuator missing")
    return {"right_wrist_roll": wrist, "all_motor_required_ctrl": values}


def pad_distance_map(item: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for contact in item["pad_contacts"]:
        for body in contact["body_names"]:
            if body in {"R_hand_narrow3_Link", "R_hand_wide3_Link"}:
                values[body] = min(values.get(body, float("inf")), float(contact["distance_m"]))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lateral-offset-mm", type=float, help="measured pad-axis placement override for dynamic-compliance compensation")
    parser.add_argument("--controller-base-compensation-m", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="measured free-space approach error negated into the motor-target pose")
    parser.add_argument("--controller-lift-base-compensation-m", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="measured free-space lift error negated into the lift motor-target pose")
    args = parser.parse_args()
    if args.lateral_offset_mm is not None and not np.isfinite(args.lateral_offset_mm):
        raise SystemExit("--lateral-offset-mm must be finite")
    if args.controller_base_compensation_m is not None and not np.isfinite(args.controller_base_compensation_m).all():
        raise SystemExit("--controller-base-compensation-m must be finite")
    if args.controller_lift_base_compensation_m is not None and not np.isfinite(args.controller_lift_base_compensation_m).all():
        raise SystemExit("--controller-lift-base-compensation-m must be finite")
    if not args.source_scene.is_file() or not args.prior_config.is_file():
        raise SystemExit("source scene or prior config is missing")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    location = (
        pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
        if args.output_dir is not None
        else pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-cylinder-static")
    )
    output = location.native_path
    if any(path.name != ".robotsim-output-reservation.json" for path in output.iterdir()):
        raise SystemExit("output directory is not fresh")
    scene = output / "full_x2_stl_horizontal_cylinder.xml"
    composition = compose_stl_scene(args.source_scene, loaded.model_path, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv) == (0, 0):
        raise SystemExit("compiled scene is invalid")
    for absent in ("R_hand_narrow2_joint", "R_hand_wide2_joint"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, absent) >= 0:
            raise SystemExit(f"STL scene unexpectedly retains {absent}")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    scans = {}
    for phase in ("HORIZONTAL_APPROACH", "CLOSE_75_PERCENT"):
        state = state_by_phase(config, phase)
        scans[phase] = [evaluate_pose(model, trajectory, state, percent) for percent in range(101)]
    eligible = []
    for phase, rows in scans.items():
        for row in rows:
            if row["bilateral_pad_contact"] and not row["nonpad_right_contact_count"]:
                eligible.append({"phase": phase, **row})
    selected = max(eligible, key=lambda item: item["command_percent"]) if eligible else None
    aligned_contact = None
    alignment_scan = []
    transport_end = None
    transport_path = []
    if selected is not None:
        approach_state = state_by_phase(config, "HORIZONTAL_APPROACH")
        contact_data = mujoco.MjData(model)
        apply_prior_state(model, contact_data, approach_state)
        apply_stl_kinematic_pose(mujoco, model, contact_data, trajectory, float(selected["command_percent"]))
        narrow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_hand_narrow3_Link")
        wide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_hand_wide3_Link")
        pad_axis = contact_data.xpos[wide] - contact_data.xpos[narrow]
        pad_axis /= np.linalg.norm(pad_axis)
        offset_candidates = {float(value) for value in range(-4, 5)}
        if args.lateral_offset_mm is not None:
            offset_candidates.add(float(args.lateral_offset_mm))
        for offset_mm in sorted(offset_candidates):
            item = solve_gripper_translation_ik(
                model,
                trajectory,
                approach_state,
                int(selected["command_percent"]),
                pad_axis * (float(offset_mm) / 1000.0),
                cylinder_lift_m=0.0,
            )
            distances = pad_distance_map(item)
            closure_summary = []
            first_pad_percent = {"R_hand_narrow3_Link": None, "R_hand_wide3_Link": None}
            highest_safe_bilateral = None
            for command in range(100, -1, -1):
                sample = evaluate_pose(model, trajectory, approach_state, command, arm_override=np.asarray([item["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float))
                present = set(sample["pad_bodies"])
                for pad in first_pad_percent:
                    if pad in present and first_pad_percent[pad] is None:
                        first_pad_percent[pad] = command
                if sample["bilateral_pad_contact"] and not sample["nonpad_right_contact_count"] and highest_safe_bilateral is None:
                    highest_safe_bilateral = command
                closure_summary.append({
                    "command_percent": command,
                    "pad_bodies": sample["pad_bodies"],
                    "nonpad_right_contact_count": sample["nonpad_right_contact_count"],
                    "pad_distances_m": pad_distance_map(sample),
                })
            item["first_pad_contact_percent"] = first_pad_percent
            item["highest_safe_bilateral_percent"] = highest_safe_bilateral
            item["closure_contact_scan"] = closure_summary
            onset_narrow = first_pad_percent["R_hand_narrow3_Link"]
            onset_wide = first_pad_percent["R_hand_wide3_Link"]
            if highest_safe_bilateral is not None and onset_narrow is not None and onset_wide is not None and set(distances) == {"R_hand_narrow3_Link", "R_hand_wide3_Link"}:
                item["pad_onset_delta_percent"] = abs(onset_narrow - onset_wide)
                item["pad_balance_score_m"] = abs(distances["R_hand_narrow3_Link"] - distances["R_hand_wide3_Link"])
                item["selection_score"] = 0.01 * item["pad_onset_delta_percent"] + item["pad_balance_score_m"]
            else:
                item["pad_onset_delta_percent"] = float("inf")
                item["pad_balance_score_m"] = float("inf")
                item["selection_score"] = float("inf")
            item["lateral_offset_mm"] = offset_mm
            item["lateral_axis_world"] = pad_axis.tolist()
            alignment_scan.append(item)
        viable_alignment = [item for item in alignment_scan if np.isfinite(item["selection_score"])]
        if viable_alignment:
            if args.lateral_offset_mm is None:
                aligned_contact = min(viable_alignment, key=lambda item: (item["pad_onset_delta_percent"], item["pad_balance_score_m"]))
            else:
                aligned_contact = min(viable_alignment, key=lambda item: abs(item["lateral_offset_mm"] - args.lateral_offset_mm))
        if aligned_contact is not None:
            aligned_arm = np.asarray([aligned_contact["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
            transport_end = solve_gripper_translation_ik(
                model,
                trajectory,
                approach_state,
                int(aligned_contact["highest_safe_bilateral_percent"]),
                np.array([0.0, 0.0, 0.05]),
                cylinder_lift_m=0.05,
                start_arm_override=aligned_arm,
            )
            end_arm = np.asarray([transport_end["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
            for fraction in np.linspace(0.0, 1.0, 5):
                item = evaluate_pose(
                    model,
                    trajectory,
                    approach_state,
                    int(aligned_contact["highest_safe_bilateral_percent"]),
                    cylinder_lift_m=0.05 * float(fraction),
                    arm_override=aligned_arm + float(fraction) * (end_arm - aligned_arm),
                )
                item["fraction"] = float(fraction)
                transport_path.append(item)
    controller_contact_target = None
    controller_lift_target = None
    if aligned_contact is not None and args.controller_base_compensation_m is not None:
        controller_delta = np.asarray(args.controller_base_compensation_m, dtype=float)
        controller_lift_delta = np.asarray(args.controller_lift_base_compensation_m, dtype=float) if args.controller_lift_base_compensation_m is not None else controller_delta
        aligned_arm = np.asarray([aligned_contact["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
        controller_contact_target = solve_gripper_translation_ik(
            model, trajectory, approach_state, int(aligned_contact["highest_safe_bilateral_percent"]), controller_delta,
            cylinder_lift_m=0.0, start_arm_override=aligned_arm,
        )
        if transport_end is not None:
            lift_arm_static = np.asarray([transport_end["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float)
            controller_lift_target = solve_gripper_translation_ik(
                model, trajectory, approach_state, int(aligned_contact["highest_safe_bilateral_percent"]), controller_lift_delta,
                cylinder_lift_m=0.05, start_arm_override=lift_arm_static,
            )
    holding_effort = None
    if aligned_contact is not None:
        holding_effort = static_holding_effort(
            model,
            trajectory,
            approach_state,
            int(aligned_contact["highest_safe_bilateral_percent"]),
            np.asarray([aligned_contact["arm_q_rad"][name] for name in ARM_JOINTS], dtype=float),
        )
    transport_path_pass = bool(transport_path) and all(
        item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"] and (item["table_contact_count"] == 1 if item["fraction"] == 0.0 else item["table_contact_count"] == 0)
        for item in transport_path
    )
    result = {
        "schema": "robotsim.issue46.full-x2-stl-cylinder-static-preflight.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "mj_step_calls": 0,
        "mujoco_version": mujoco.__version__,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "source_scene": {"path": str(args.source_scene), "sha256": sha256(args.source_scene)},
        "generated_scene": {"path": str(scene), "sha256": sha256(scene)},
        "composition": composition,
        "lateral_offset_override_mm": args.lateral_offset_mm,
        "controller_base_compensation_world_m": args.controller_base_compensation_m,
        "controller_lift_base_compensation_world_m": args.controller_lift_base_compensation_m,
        "scan": scans,
        "selected_candidate": selected,
        "lateral_alignment_scan": alignment_scan,
        "aligned_contact_static": aligned_contact,
        "controller_contact_target_static": controller_contact_target,
        "transport_end_static": transport_end,
        "controller_lift_target_static": controller_lift_target,
        "transport_path_static": transport_path,
        "static_holding_effort": holding_effort,
        "gates": {
            "STL_HAND_COMPILED": "PASS",
            "EXACT_101_ROW_TABLE": "PASS",
            "BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS" if selected and aligned_contact else "BLOCKED",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS" if transport_end and transport_end["pass"] and transport_path_pass else "BLOCKED",
            "RIGHT_WRIST_STATIC_HOLD_WITHIN_CAP": "PASS" if holding_effort and abs(holding_effort["right_wrist_roll"]["required_ctrl"]) <= max(abs(value) for value in holding_effort["right_wrist_roll"]["range"]) else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The hand is table-driven kinematics, not a manufacturer-equivalent transmission.",
            "A static source-mesh collision result is a preflight gate, not a physical grasp result.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "selected_candidate": selected, "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
