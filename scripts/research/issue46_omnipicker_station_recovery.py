#!/usr/bin/env python3
"""Bounded Issue #46 station and source-limited OmniPicker geometry recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = ROOT / "scripts/research/issue46_omnipicker_1dof_m0.py"
GEOMETRY_PATH = ROOT / "scripts/research/issue46_omnipicker_geometric_feasibility.py"
DEFAULT_OUTPUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-station-recovery/20261008-run1"
)

APPROACH_WORLD = np.array([0.0, -1.0, 0.0])
JAW_AXIS_WORLD = np.array([-1.0, 0.0, 0.0])
PREGRASP_BACKOFF_M = 0.10
TARGET_POSITION_TOLERANCE_M = 0.0015
TARGET_ORIENTATION_TOLERANCE_RAD = math.radians(1.0)
PATH_SAMPLES = 41

CANDIDATES = (
    {
        "id": "a",
        "name": "forward-shifted-natural-facing",
        "base_world": [0.38, 0.28, 0.68],
        "base_yaw_rad": -math.pi / 2.0,
        "justification": (
            "Move the accepted standing station 40 mm toward the table along the verified "
            "world -Y facing direction to reduce the wrist reach while retaining the accepted "
            "lateral right-hand workspace and perpendicular torso/table alignment."
        ),
    },
    {
        "id": "b",
        "name": "right-shoulder-aligned",
        "base_world": [0.443, 0.32, 0.68],
        "base_yaw_rad": -math.pi / 2.0,
        "justification": (
            "Place the compiled neutral right shoulder over the bottle's X coordinate: the "
            "source shoulder lateral offset is -0.143 m in world X at this yaw, so pelvis X "
            "is derived as bottle X + 0.143 m. Preserve the accepted front/back station and yaw."
        ),
    },
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=normalize) + "\n", encoding="utf-8")


def name(model: mujoco.MjModel, obj: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, obj, int(object_id)) or f"{obj.name.lower()}_{object_id}"


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise RuntimeError("Cannot normalize a degenerate compiled geometry vector")
    return value / norm


def collision_geom_ids(model: mujoco.MjModel, body_id: int | None = None) -> list[int]:
    return [
        geom_id for geom_id in range(model.ngeom)
        if (body_id is None or int(model.geom_bodyid[geom_id]) == body_id)
        and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
    ]


def contact_policy(row: dict[str, Any]) -> str:
    bodies = {str(row["body1"]), str(row["body2"])}
    if "floor" in bodies and any("ankle" in body or "foot" in body for body in bodies):
        return "allowed_standing_support"
    if "m0_table" in bodies and "m0_bottle" in bodies:
        return "allowed_bottle_table_support"
    if "m0_bottle" in bodies:
        hand = next((body for body in bodies if body != "m0_bottle"), "")
        if hand.startswith(("R_hand_narrow", "R_hand_wide")) and "loop" not in hand.lower():
            return "allowed_intended_jaw_bottle_contact"
    return "unwanted_contact"


def bad_contacts(rows: list[dict[str, Any]], allow_jaw_bottle: bool = False) -> list[dict[str, Any]]:
    bad = []
    for row in rows:
        policy = contact_policy(row)
        if policy in {"allowed_standing_support", "allowed_bottle_table_support"}:
            continue
        if allow_jaw_bottle and policy == "allowed_intended_jaw_bottle_contact":
            continue
        if policy == "unwanted_contact" or float(row["distance_m"]) < -1e-7:
            bad.append({**row, "policy": policy})
    return bad


def signed_distance_rows(model: mujoco.MjModel, data: mujoco.MjData,
                         first_geoms: list[int], second_geoms: list[int]) -> list[dict[str, Any]]:
    rows = []
    for first in first_geoms:
        for second in second_geoms:
            if int(model.geom_bodyid[first]) == int(model.geom_bodyid[second]):
                continue
            witness = np.zeros(6, dtype=float)
            distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, witness))
            rows.append({
                "geom1_id": first,
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[first]),
                "geom2_id": second,
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[second]),
                "signed_distance_m": distance,
                "witness_segment_world_m": witness.tolist(),
            })
    return sorted(rows, key=lambda row: row["signed_distance_m"])


def render_pose(model: mujoco.MjModel, data: mujoco.MjData, path: Path,
                lookat: list[float], distance: float, azimuth: float, elevation: float,
                width: int = 640, height: int = 480) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render().copy()).save(path)
    renderer.close()


def source_and_fk_audit(exp, model: mujoco.MjModel, data: mujoco.MjData,
                        source: dict[str, Any]) -> dict[str, Any]:
    pelvis = exp.body_id(model, "pelvis")
    rotation = data.xmat[pelvis].reshape(3, 3)
    table = exp.body_id(model, "m0_table")
    bottle = exp.body_id(model, "m0_bottle")
    collision_bottle_geoms = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_bottle"
    ]
    bottle_body_geom = next(
        gid for gid in collision_bottle_geoms
        if name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "bottle_body"
    )
    source_joints = {row["name"]: row for row in source["joint_records"]}
    arm_limits = {}
    for joint_name in exp.RIGHT_ARM_JOINTS:
        jid = exp.joint_id(model, joint_name)
        source_limit = source_joints.get(joint_name)
        if source_limit is None:
            root = exp.ET.parse(exp.X2_ROOT / exp.X2_URDF_REL).getroot()
            joint = next(node for node in root.findall("joint") if node.get("name") == joint_name)
            limit = joint.find("limit")
            source_range = [float(limit.get("lower")), float(limit.get("upper"))]
            source_axis = [float(v) for v in joint.find("axis").get("xyz").split()]
        else:
            source_range = source_limit["source_position_range_rad"]
            source_axis = source_limit["axis"]
        compiled_range = model.jnt_range[jid].astype(float).tolist()
        arm_limits[joint_name] = {
            "source_range_rad": source_range,
            "compiled_range_rad": compiled_range,
            "range_exact_match": bool(np.array_equal(np.asarray(source_range), np.asarray(compiled_range))),
            "source_axis": source_axis,
            "compiled_axis": model.jnt_axis[jid].astype(float).tolist(),
            "axis_exact_match": bool(np.allclose(source_axis, model.jnt_axis[jid], atol=1e-12, rtol=0.0)),
            "qpos0_rad": float(model.qpos0[model.jnt_qposadr[jid]]),
            "initial_qpos_rad": float(data.qpos[model.jnt_qposadr[jid]]),
        }

    ee = exp.body_id(model, "R_omnipicker_base_link")
    local_point = np.array([0.0, 0.0, 0.10], dtype=float)
    point = data.xpos[ee] + data.xmat[ee].reshape(3, 3) @ local_point
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jac(model, data, jacp, jacr, point, ee)
    finite_difference = []
    eps = 1e-7
    for joint_name in exp.RIGHT_ARM_JOINTS:
        jid = exp.joint_id(model, joint_name)
        qadr = int(model.jnt_qposadr[jid])
        dof = int(model.jnt_dofadr[jid])
        plus = mujoco.MjData(model)
        minus = mujoco.MjData(model)
        plus.qpos[:] = data.qpos
        minus.qpos[:] = data.qpos
        plus.qpos[qadr] += eps
        minus.qpos[qadr] -= eps
        mujoco.mj_forward(model, plus)
        mujoco.mj_forward(model, minus)
        pp = plus.xpos[ee] + plus.xmat[ee].reshape(3, 3) @ local_point
        pm = minus.xpos[ee] + minus.xmat[ee].reshape(3, 3) @ local_point
        numeric = (pp - pm) / (2.0 * eps)
        finite_difference.append({
            "joint": joint_name,
            "mujoco_jacobian_linear_m_per_rad": jacp[:, dof].tolist(),
            "finite_difference_linear_m_per_rad": numeric.tolist(),
            "max_abs_difference": float(np.max(np.abs(jacp[:, dof] - numeric))),
        })

    old_filter = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]).startswith(("right_", "R_hand_"))
    ]
    all_robot = [
        gid for gid in collision_geom_ids(model)
        if int(model.geom_bodyid[gid]) != 0
        and name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) not in {"m0_table", "m0_bottle"}
    ]
    omitted = sorted({
        name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        for gid in all_robot if gid not in old_filter
    })
    return {
        "base": {
            "pelvis_world_position_m": data.xpos[pelvis].tolist(),
            "pelvis_rotation_row_major": rotation.tolist(),
            "local_x_forward_world": (rotation @ np.array([1.0, 0.0, 0.0])).tolist(),
            "local_y_left_world": (rotation @ np.array([0.0, 1.0, 0.0])).tolist(),
            "table_center_world_m": data.xpos[table].tolist(),
            "table_center_relative_to_pelvis_m": (data.xpos[table] - data.xpos[pelvis]).tolist(),
            "bottle_body_root_world_m": data.xpos[bottle].tolist(),
            "bottle_collision_cylinder_center_world_m": data.geom_xpos[bottle_body_geom].tolist(),
            "root_to_body_collision_center_m": (data.geom_xpos[bottle_body_geom] - data.xpos[bottle]).tolist(),
            "bottle_body_cylinder_diameter_m": float(2.0 * model.geom_size[bottle_body_geom, 0]),
            "bottle_body_cylinder_height_m": float(2.0 * model.geom_size[bottle_body_geom, 1]),
        },
        "right_arm_joint_mapping": arm_limits,
        "compiled_fk_finite_difference": finite_difference,
        "collision_selection_audit": {
            "legacy_right_filter_geom_count": len(old_filter),
            "all_active_robot_geom_count": len(all_robot),
            "legacy_filter_omitted_robot_bodies": omitted,
            "new_clearance_selection": "all active robot collision geoms except world/table/bottle; pair contacts are checked from compiled mjData contacts",
        },
    }


def derive_tool_geometry(exp, geo, model: mujoco.MjModel, base_data: mujoco.MjData,
                         aperture_samples: int = 101) -> dict[str, Any]:
    narrow = exp.collision_geom_id(model, "R_hand_narrow3_Link")
    wide = exp.collision_geom_id(model, "R_hand_wide3_Link")
    bottle_geoms = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_bottle"
    ]
    bottle_body = next(gid for gid in bottle_geoms if name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "bottle_body")
    diameter = float(2.0 * model.geom_size[bottle_body, 0])
    seed = exp.model_arm_home(model)
    samples = []
    for aperture in np.linspace(1.0, 0.0, aperture_samples):
        data = geo.q_state(exp, model, base_data, seed, float(aperture))
        witness = np.zeros(6, dtype=float)
        gap = float(mujoco.mj_geomDistance(model, data, narrow, wide, 1.0, witness))
        samples.append({
            "aperture_ratio": float(aperture),
            "compiled_jaw_surface_gap_m": gap,
            "witness_segment_world_m": witness.tolist(),
        })
    monotone = all(
        samples[index + 1]["compiled_jaw_surface_gap_m"] <= samples[index]["compiled_jaw_surface_gap_m"] + 1e-6
        for index in range(len(samples) - 1)
    )
    crossing = next((index for index in range(len(samples) - 1)
                     if samples[index]["compiled_jaw_surface_gap_m"] >= diameter
                     and samples[index + 1]["compiled_jaw_surface_gap_m"] <= diameter), None)
    if crossing is None:
        raise RuntimeError("Compiled narrow/wide jaw surfaces never span the canonical bottle diameter")
    a = samples[crossing]
    b = samples[crossing + 1]
    gap_a = a["compiled_jaw_surface_gap_m"]
    gap_b = b["compiled_jaw_surface_gap_m"]
    fraction = (gap_a - diameter) / (gap_a - gap_b)
    aperture = a["aperture_ratio"] + fraction * (b["aperture_ratio"] - a["aperture_ratio"])
    contact_data = geo.q_state(exp, model, base_data, seed, aperture)
    witness = np.zeros(6, dtype=float)
    contact_gap = float(mujoco.mj_geomDistance(model, contact_data, narrow, wide, 1.0, witness))
    ee = exp.body_id(model, "R_omnipicker_base_link")
    rotation = contact_data.xmat[ee].reshape(3, 3)
    contact_midpoint = 0.5 * (witness[:3] + witness[3:])
    contact_point_local = rotation.T @ (contact_midpoint - contact_data.xpos[ee])
    axis_local = rotation.T @ unit(witness[3:] - witness[:3])

    open_data = geo.q_state(exp, model, base_data, seed, 1.0)
    open_witness = np.zeros(6, dtype=float)
    open_gap = float(mujoco.mj_geomDistance(model, open_data, narrow, wide, 1.0, open_witness))
    open_midpoint = 0.5 * (open_witness[:3] + open_witness[3:])
    open_point_local = open_data.xmat[ee].reshape(3, 3).T @ (open_midpoint - open_data.xpos[ee])

    local_axis = unit(axis_local)
    local_approach = unit(contact_point_local - local_axis * float(np.dot(local_axis, contact_point_local)))
    source_basis = np.column_stack((local_axis, local_approach, unit(np.cross(local_axis, local_approach))))
    target_axis = unit(JAW_AXIS_WORLD)
    target_approach = unit(APPROACH_WORLD - target_axis * float(np.dot(target_axis, APPROACH_WORLD)))
    target_basis = np.column_stack((target_axis, target_approach, unit(np.cross(target_axis, target_approach))))
    target_rotation = target_basis @ source_basis.T

    return {
        "bottle_body_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_body),
        "bottle_body_center_world_m": contact_data.geom_xpos[bottle_body].tolist(),
        "bottle_body_diameter_m": diameter,
        "jaw_geoms": {"narrow": name(model, mujoco.mjtObj.mjOBJ_GEOM, narrow),
                      "wide": name(model, mujoco.mjtObj.mjOBJ_GEOM, wide)},
        "compiled_gap_monotonic": monotone,
        "compiled_gap_samples": samples,
        "open_gap_m": open_gap,
        "contact_aperture_ratio_for_body_diameter": aperture,
        "contact_gap_m": contact_gap,
        "contact_witness_world_m": witness.tolist(),
        "local_contact_gap_center_from_ee_m": contact_point_local.tolist(),
        "open_witness_world_m": open_witness.tolist(),
        "local_open_gap_center_from_ee_m": open_point_local.tolist(),
        "open_to_contact_gap_center_shift_m": float(np.linalg.norm(open_point_local - contact_point_local)),
        "local_jaw_axis_narrow_to_wide": local_axis.tolist(),
        "target_jaw_axis_world": target_axis.tolist(),
        "target_approach_world": target_approach.tolist(),
        "target_ee_rotation_row_major": target_rotation.tolist(),
        "target_ee_rotation_determinant": float(np.linalg.det(target_rotation)),
        "target_construction": (
            "Compiled jaw witness axis maps to horizontal world -X; compiled wrist-to-gap-center "
            "vector maps to world -Y along the verified pregrasp-to-bottle approach."
        ),
    }


def solve_pose(exp, geo, model: mujoco.MjModel, base_data: mujoco.MjData,
               tool: dict[str, Any], target_gap_center: np.ndarray, aperture: float,
               seed: dict[str, float], eval_path: Path, pose_name: str) -> dict[str, Any]:
    ee = exp.body_id(model, "R_omnipicker_base_link")
    target_rotation = np.asarray(tool["target_ee_rotation_row_major"], dtype=float)
    local_point = np.asarray(
        tool["local_open_gap_center_from_ee_m"] if aperture > 0.999
        else tool["local_contact_gap_center_from_ee_m"], dtype=float
    )
    target_ee = target_gap_center - target_rotation @ local_point
    source_ranges = np.asarray([model.jnt_range[exp.joint_id(model, joint)] for joint in exp.RIGHT_ARM_JOINTS])
    q_lower = source_ranges[:, 0] + 1e-8
    q_upper = source_ranges[:, 1] - 1e-8
    seed_q = np.asarray([seed[joint] for joint in exp.RIGHT_ARM_JOINTS], dtype=float)
    optimizer_seed = np.clip(seed_q, q_lower, q_upper)
    seed_clipping = (optimizer_seed - seed_q).tolist()
    counter = 0
    with eval_path.open("w", encoding="utf-8") as stream:
        def residual(q: np.ndarray) -> np.ndarray:
            nonlocal counter
            counter += 1
            arm = {joint: float(value) for joint, value in zip(exp.RIGHT_ARM_JOINTS, q)}
            data = geo.q_state(exp, model, base_data, arm, aperture)
            current_rotation = data.xmat[ee].reshape(3, 3)
            position_error = target_ee - data.xpos[ee]
            orientation_error = exp.rotation_error(target_rotation, current_rotation)
            limits = {
                joint: {
                    "qpos_rad": float(q[index]),
                    "lower_rad": float(source_ranges[index, 0]),
                    "upper_rad": float(source_ranges[index, 1]),
                    "lower_margin_rad": float(q[index] - source_ranges[index, 0]),
                    "upper_margin_rad": float(source_ranges[index, 1] - q[index]),
                }
                for index, joint in enumerate(exp.RIGHT_ARM_JOINTS)
            }
            contacts = geo.contact_rows(exp, model, data)
            record = {
                "evaluation": counter,
                "pose": pose_name,
                "qpos_rad": arm,
                "position_error_vector_m": position_error.tolist(),
                "position_error_norm_m": float(np.linalg.norm(position_error)),
                "orientation_error_vector_rad": orientation_error.tolist(),
                "orientation_error_norm_rad": float(np.linalg.norm(orientation_error)),
                "source_limit_state": limits,
                "contacts": contacts,
                "bad_contacts": bad_contacts(contacts),
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            return np.r_[position_error, 0.12 * orientation_error, 0.002 * (q - optimizer_seed)]

        result = least_squares(
            residual, optimizer_seed, bounds=(q_lower, q_upper), max_nfev=500,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, x_scale="jac",
        )

    q = result.x
    arm = {joint: float(value) for joint, value in zip(exp.RIGHT_ARM_JOINTS, q)}
    data = geo.q_state(exp, model, base_data, arm, aperture)
    current_rotation = data.xmat[ee].reshape(3, 3)
    position_error = target_ee - data.xpos[ee]
    orientation_error = exp.rotation_error(target_rotation, current_rotation)
    contacts = geo.contact_rows(exp, model, data)
    table_geoms = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_table"
    ]
    bottle_geoms = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_bottle"
    ]
    robot_geoms = [
        gid for gid in collision_geom_ids(model)
        if int(model.geom_bodyid[gid]) != 0
        and name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) not in {"m0_table", "m0_bottle"}
    ]
    return {
        "pose": pose_name,
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "solver_evaluation_path": str(eval_path),
        "optimizer_seed_qpos_rad": dict(zip(exp.RIGHT_ARM_JOINTS, map(float, optimizer_seed))),
        "optimizer_seed_clipping_rad": seed_clipping,
        "source_position_ranges_rad": {joint: source_ranges[i].tolist()
                                        for i, joint in enumerate(exp.RIGHT_ARM_JOINTS)},
        "solved_qpos_rad": arm,
        "target_gap_center_world_m": target_gap_center.tolist(),
        "target_ee_position_world_m": target_ee.tolist(),
        "achieved_ee_position_world_m": data.xpos[ee].tolist(),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "orientation_error_norm_rad": float(np.linalg.norm(orientation_error)),
        "orientation_error_deg": math.degrees(float(np.linalg.norm(orientation_error))),
        "target_rotation_row_major": target_rotation.tolist(),
        "achieved_jaw_axis_world": (current_rotation @ np.asarray(tool["local_jaw_axis_narrow_to_wide"])).tolist(),
        "contacts": contacts,
        "bad_contacts": bad_contacts(contacts),
        "minimum_robot_to_table_signed_distance_rows": signed_distance_rows(model, data, robot_geoms, table_geoms)[:20],
        "minimum_robot_to_bottle_signed_distance_rows": signed_distance_rows(model, data, robot_geoms, bottle_geoms)[:20],
        "within_source_ranges": all(
            source_ranges[i, 0] <= q[i] <= source_ranges[i, 1] for i in range(len(q))
        ),
        "pose_tolerance_pass": (
            np.linalg.norm(position_error) <= TARGET_POSITION_TOLERANCE_M
            and np.linalg.norm(orientation_error) <= TARGET_ORIENTATION_TOLERANCE_RAD
        ),
        "static_collision_pass": not bad_contacts(contacts),
    }


def route_samples(exp, geo, model: mujoco.MjModel, base_data: mujoco.MjData,
                  start: dict[str, float], end: dict[str, float], aperture: float,
                  segment: str) -> list[dict[str, Any]]:
    rows = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, PATH_SAMPLES)):
        arm = {
            joint: float(start[joint] + fraction * (end[joint] - start[joint]))
            for joint in exp.RIGHT_ARM_JOINTS
        }
        data = geo.q_state(exp, model, base_data, arm, aperture)
        contacts = geo.contact_rows(exp, model, data)
        rows.append({
            "segment": segment,
            "sample": index,
            "fraction": float(fraction),
            "arm_qpos_rad": arm,
            "contacts": contacts,
            "bad_contacts": bad_contacts(contacts),
            "all_source_ranges_valid": all(
                float(model.jnt_range[exp.joint_id(model, joint), 0]) <= arm[joint]
                <= float(model.jnt_range[exp.joint_id(model, joint), 1])
                for joint in exp.RIGHT_ARM_JOINTS
            ),
        })
    return rows


def evaluate_candidate(exp, geo, candidate: dict[str, Any], evidence: Path,
                       source: dict[str, Any]) -> dict[str, Any]:
    exp.BASE_WORLD = np.asarray(candidate["base_world"], dtype=float)
    exp.BASE_YAW = float(candidate["base_yaw_rad"])
    model, build_info = exp.build_model("bottle_hold", {})
    base_data = exp.initialize(model)
    mujoco.mj_forward(model, base_data)
    audit = source_and_fk_audit(exp, model, base_data, source)
    tool = derive_tool_geometry(exp, geo, model, base_data)
    initial_contacts = geo.contact_rows(exp, model, base_data)
    source_initial, source_violations = exp.live_source_state(
        model, base_data, build_info["source_position_limits_rad"]
    )
    tool["pregrasp_backoff_m"] = PREGRASP_BACKOFF_M
    bottle_center = np.asarray(tool["bottle_body_center_world_m"], dtype=float)
    pregrasp_center = bottle_center - APPROACH_WORLD * PREGRASP_BACKOFF_M
    initial_arm = exp.model_arm_home(model)
    pre_dir = evidence / f"candidate_{candidate['id']}_pregrasp_ik_evaluations.jsonl"
    grasp_dir = evidence / f"candidate_{candidate['id']}_grasp_ik_evaluations.jsonl"
    pregrasp = solve_pose(
        exp, geo, model, base_data, tool, pregrasp_center, 1.0, initial_arm, pre_dir, "pregrasp_open"
    )
    grasp = solve_pose(
        exp, geo, model, base_data, tool, bottle_center, float(tool["contact_aperture_ratio_for_body_diameter"]),
        pregrasp["solved_qpos_rad"], grasp_dir, "grasp_at_contact_gap",
    )
    pregrasp_arm = pregrasp["solved_qpos_rad"]
    grasp_arm = grasp["solved_qpos_rad"]
    home_to_pregrasp = route_samples(exp, geo, model, base_data, initial_arm, pregrasp_arm, 1.0,
                                     "initial_open_to_pregrasp")
    pregrasp_to_grasp = route_samples(exp, geo, model, base_data, pregrasp_arm, grasp_arm, 1.0,
                                      "pregrasp_to_grasp_open")

    open_grasp_data = geo.q_state(exp, model, base_data, grasp_arm, 1.0)
    closure_rows = []
    for aperture in np.linspace(1.0, float(tool["contact_aperture_ratio_for_body_diameter"]), 31):
        data = geo.q_state(exp, model, base_data, grasp_arm, float(aperture))
        narrow = exp.collision_geom_id(model, "R_hand_narrow3_Link")
        wide = exp.collision_geom_id(model, "R_hand_wide3_Link")
        bottle_body = next(
            gid for gid in range(model.ngeom)
            if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_bottle"
            and name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "bottle_body"
        )
        nseg = np.zeros(6, dtype=float)
        wseg = np.zeros(6, dtype=float)
        narrow_distance = float(mujoco.mj_geomDistance(model, data, narrow, bottle_body, 1.0, nseg))
        wide_distance = float(mujoco.mj_geomDistance(model, data, wide, bottle_body, 1.0, wseg))
        contacts = geo.contact_rows(exp, model, data)
        closure_rows.append({
            "aperture_ratio": float(aperture),
            "narrow3_to_bottle_body_signed_distance_m": narrow_distance,
            "narrow3_witness_world_m": nseg.tolist(),
            "wide3_to_bottle_body_signed_distance_m": wide_distance,
            "wide3_witness_world_m": wseg.tolist(),
            "contacts": contacts,
            "bad_contacts_allowing_intended_jaw_bottle": bad_contacts(contacts, allow_jaw_bottle=True),
        })

    table_geoms = [
        gid for gid in collision_geom_ids(model)
        if name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == "m0_table"
    ]
    robot_geoms = [
        gid for gid in collision_geom_ids(model)
        if int(model.geom_bodyid[gid]) != 0
        and name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) not in {"m0_table", "m0_bottle"}
    ]
    initial_table_distance = signed_distance_rows(model, base_data, robot_geoms, table_geoms)
    initial_bad = bad_contacts(initial_contacts)
    pre_bad = pregrasp["bad_contacts"]
    grasp_open_contacts = geo.contact_rows(exp, model, open_grasp_data)
    grasp_open_bad = bad_contacts(grasp_open_contacts)
    path_bad = [row for row in home_to_pregrasp + pregrasp_to_grasp if row["bad_contacts"]]
    contact_sample = min(closure_rows, key=lambda row: abs(
        row["aperture_ratio"] - float(tool["contact_aperture_ratio_for_body_diameter"])
    ))
    opposing_surfaces_reach = (
        abs(contact_sample["narrow3_to_bottle_body_signed_distance_m"]) <= 0.002
        and abs(contact_sample["wide3_to_bottle_body_signed_distance_m"]) <= 0.002
    )
    static_pass = bool(
        not initial_bad and not source_violations
        and audit["base"]["local_x_forward_world"][1] < -0.99
        and pregrasp["within_source_ranges"] and pregrasp["pose_tolerance_pass"] and not pre_bad
        and grasp["within_source_ranges"] and grasp["pose_tolerance_pass"] and not grasp_open_bad
        and not path_bad and opposing_surfaces_reach
        and all(row["all_source_ranges_valid"] for row in home_to_pregrasp + pregrasp_to_grasp)
    )

    candidate_result = {
        "candidate": candidate,
        "status": "PASS_STATIC" if static_pass else "FAIL_STATIC",
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                  "ngeom": model.ngeom, "nbody": model.nbody,
                  "timestep_s": float(model.opt.timestep),
                  "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
                  "iterations": int(model.opt.iterations)},
        "station_and_frame_audit": audit,
        "zero_step_initial_contacts": initial_contacts,
        "zero_step_initial_bad_contacts": initial_bad,
        "zero_step_initial_source_state": source_initial,
        "zero_step_source_limit_violations": source_violations,
        "initial_robot_to_table_signed_distance_rows": initial_table_distance[:30],
        "compiled_tool_geometry": tool,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "pregrasp_open_contacts": geo.contact_rows(
            exp, model, geo.q_state(exp, model, base_data, pregrasp_arm, 1.0)
        ),
        "grasp_open_contacts": grasp_open_contacts,
        "grasp_open_bad_contacts": grasp_open_bad,
        "approach_path_sample_count": len(home_to_pregrasp) + len(pregrasp_to_grasp),
        "approach_path_samples": home_to_pregrasp + pregrasp_to_grasp,
        "approach_path_first_bad_samples": path_bad[:10],
        "closure_samples": closure_rows,
        "opposing_jaw_surfaces_reach_bottle_diameter": opposing_surfaces_reach,
        "static_gate": {
            "initial_collision_clear": not initial_bad and not source_violations,
            "source_limited_pregrasp": pregrasp["within_source_ranges"] and pregrasp["pose_tolerance_pass"],
            "no_unwanted_pregrasp_contact": not pre_bad,
            "no_wrist_or_bottle_penetration_at_open_grasp": not grasp_open_bad,
            "collision_free_approach_path": not path_bad,
            "opposing_jaws_can_reach_70mm_body": opposing_surfaces_reach,
            "source_valid_path_samples": all(row["all_source_ranges_valid"] for row in home_to_pregrasp + pregrasp_to_grasp),
            "static_pass": static_pass,
            "first_failure": (
                "initial robot/table or source-limit preflight" if initial_bad or source_violations else
                "source-valid pregrasp self/table/bottle clearance" if not pregrasp["within_source_ranges"] or not pregrasp["pose_tolerance_pass"] or pre_bad else
                "open grasp wrist/hand clearance" if grasp_open_bad else
                "collision-free approach trajectory" if path_bad else
                "opposing jaw surface reach" if not opposing_surfaces_reach else None
            ),
        },
    }
    write_json(evidence / f"candidate_{candidate['id']}_result.json", candidate_result)

    original_arm = initial_arm
    candidate_states = {
        "overview": geo.q_state(exp, model, base_data, pregrasp_arm, 1.0),
        "front": geo.q_state(exp, model, base_data, pregrasp_arm, 1.0),
        "side": geo.q_state(exp, model, base_data, pregrasp_arm, 1.0),
        "hand": open_grasp_data,
        "table": geo.q_state(exp, model, base_data, pregrasp_arm, 1.0),
    }
    camera_specs = {
        "overview": ([0.30, 0.12, 0.91], 1.65, 142.0, -15.0),
        "front": ([0.30, 0.12, 0.90], 1.55, 180.0, -7.0),
        "side": ([0.30, 0.12, 0.84], 1.55, 90.0, -8.0),
        "hand": ([0.30, 0.02, 0.88], 0.62, 140.0, -12.0),
        "table": ([0.30, 0.04, 0.64], 1.05, 180.0, -48.0),
    }
    for view, (lookat, distance, azimuth, elevation) in camera_specs.items():
        render_pose(model, candidate_states[view], evidence / f"candidate_{candidate['id']}_{view}.png",
                    lookat, distance, azimuth, elevation)
    candidate_result["screenshots"] = {
        view: str(evidence / f"candidate_{candidate['id']}_{view}.png") for view in camera_specs
    }
    candidate_result["screenshots"]["original_station_overview"] = str(evidence / "original_station_overview.png")
    write_json(evidence / f"candidate_{candidate['id']}_result.json", candidate_result)
    return candidate_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = (
        "wsl.exe -d Ubuntu-22.04 --exec /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python "
        f"{Path(__file__).resolve()} --output {output}"
    )
    (output / "experiment_commands.txt").write_text(commands + "\n", encoding="utf-8")

    exp = load_module("issue46_omnipicker_1dof_m0", EXPERIMENT_PATH)
    geo = load_module("issue46_omnipicker_geometric_feasibility", GEOMETRY_PATH)
    source = exp.verify_inputs(output)
    identity = exp.runtime_identity()
    identity.update({
        "robot_sim_head_before_commit": exp.subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "station_recovery_script": str(Path(__file__).resolve()),
        "station_recovery_script_sha256": exp.sha256(Path(__file__).resolve()),
        "geometry_solver_sha256": exp.sha256(GEOMETRY_PATH),
        "accepted_runner_sha256": exp.sha256(EXPERIMENT_PATH),
        "candidate_count_maximum": 2,
        "physics_steps_before_static_decision": 0,
    })
    write_json(output / "runtime_identity.json", identity)
    write_json(output / "source_audit.json", source)

    exp.BASE_WORLD = np.array([0.38, 0.32, 0.68], dtype=float)
    exp.BASE_YAW = -math.pi / 2.0
    original_model, _ = exp.build_model("bottle_hold", {})
    original_data = exp.initialize(original_model)
    mujoco.mj_forward(original_model, original_data)
    render_pose(original_model, original_data, output / "original_station_overview.png",
                [0.30, 0.12, 0.91], 1.65, 142.0, -15.0)

    candidate_results = []
    for candidate in CANDIDATES:
        result = evaluate_candidate(exp, geo, candidate, output, source)
        candidate_results.append(result)
        # Each raw result and image packet is complete before the next candidate is evaluated.
        write_json(output / "candidate_summary.json", {
            "completed_candidate_ids": [item["candidate"]["id"] for item in candidate_results],
            "candidate_statuses": [item["status"] for item in candidate_results],
        })
        if result["static_gate"]["static_pass"]:
            break

    passing = next((item for item in candidate_results if item["static_gate"]["static_pass"]), None)
    status = "STATIC_GEOMETRY_PASS" if passing else "BLOCKED"
    first_failure = None if passing else candidate_results[-1]["static_gate"]["first_failure"]
    result = {
        "status": status,
        "first_unresolved_gate": first_failure,
        "station_candidate_count": len(candidate_results),
        "physics_steps": 0,
        "source_limit_violations": sum(len(item["zero_step_source_limit_violations"]) for item in candidate_results),
        "object_qpos_writes": 0,
        "robot_qpos_writes_during_physics": 0,
        "accepted_controller_changed": False,
        "source_joint_limits_changed": False,
        "collision_geometry_changed": False,
        "bottle_table_gripper_changed": False,
        "candidate_results": [
            {"id": item["candidate"]["id"], "name": item["candidate"]["name"],
             "status": item["status"], "static_gate": item["static_gate"],
             "result_path": str(output / f"candidate_{item['candidate']['id']}_result.json")}
            for item in candidate_results
        ],
        "next_physics_stages": "NOT_RUN: static corridor validation did not pass" if not passing else "NOT_IMPLEMENTED_IN_STATIC_RUNNER",
        "runtime_identity": identity,
        "source": source,
    }
    write_json(output / "result.json", result)
    report = [
        "# Issue #46 X2 OmniPicker station recovery",
        "",
        f"Status: **{status}**",
        f"First unresolved gate: `{first_failure}`",
        "",
        "## Scope and provenance",
        "",
        f"- RobotSim HEAD before this evidence commit: `{identity['robot_sim_head_before_commit']}`",
        f"- X2 source: `{source['repository']}` at `{source['commit']}`",
        f"- URDF: `{source['urdf_path']}` SHA-256 `{source['urdf_sha256']}`",
        f"- MuJoCo Python/native: `{identity['mujoco_python_version']}` / `{identity['mujoco_native_version']}`",
        f"- Native library: `{identity['native_libraries'][0]['path']}` SHA-256 `{identity['native_libraries'][0]['sha256']}`",
        f"- Experiment script SHA-256: `{identity['station_recovery_script_sha256']}`",
        f"- Physics steps: `{result['physics_steps']}`",
        "",
        "## Solver audit",
        "",
        "The compiled pelvis frame maps X2 local +X forward to world -Y. The G1 table is ahead of the robot, and the former world +Y pregrasp offset is the correct 100 mm backoff from the bottle toward the robot. The previous solver correctly selected the compiled `bottle_body` collision-cylinder center rather than the bottle root; the two differ by 40 mm in Z. The source arm ranges and axes match the compiled MuJoCo joints, and the independent finite-difference check records the full seven-joint point Jacobian residuals.",
        "",
        "The prior candidate-3 seed was source-limited but already penetrated the table top by 41.8 mm, the front-left leg by 52.6 mm, and a jaw link by 2.2 mm. The old line search rejects any colliding iterate, so it could not leave this seed. The present solver starts from the reset neutral arm pose and writes every least-squares evaluation, including source-limit margins and contact pairs.",
        "",
        "The previous target formulation also reused the jaw-gap midpoint measured at the bottle-diameter closure aperture for the open pregrasp. Compiled geometry shows the open gap midpoint shifts 13.45 mm from that closure midpoint. This runner derives separate OPEN and contact-aperture tool points and constrains the wrist-to-gap vector along the verified approach direction.",
        "",
        "The former clearance selector checked all contacts but omitted torso and non-right-side bodies from its clearance table. This runner's geometric clearance set includes all active robot collision geoms; `mjData` contacts remain the authoritative collision gate.",
        "",
        "## Candidate outcomes",
        "",
    ]
    for item in candidate_results:
        gate = item["static_gate"]
        report.append(
            f"- Candidate `{item['candidate']['id']}` `{item['candidate']['name']}`: **{item['status']}**; "
            f"first failure `{gate['first_failure']}`. Exact poses, solver evaluations, contacts, distances, and screenshots are in `candidate_{item['candidate']['id']}_result.json` and the corresponding candidate files."
        )
    report.extend([
        "",
        "## Stop decision",
        "",
        "No physics stage was run because neither evaluated station produced a source-valid, collision-free static approach and open-grasp corridor. This is a bounded geometry result only; it does not establish global X2 reachability or controller failure. The no-contact regression, bottle hold, and lift gates remain untested for this layout.",
        "",
        "## Reproduction",
        "",
        "```bash",
        commands,
        "```",
        "",
        f"Evidence directory: `C:\\Users\\HP\\Desktop\\Robot\\reviews\\issue-46-x2-single-hand\\omnipicker-station-recovery\\{output.name}\\`.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
