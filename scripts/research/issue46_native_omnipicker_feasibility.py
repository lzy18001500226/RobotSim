#!/usr/bin/env python3
"""Bounded full-X2 source-geometry feasibility check for the native OmniPicker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[2]
X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
URDF_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
CANONICAL_SHA256 = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"
RIGHT_ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
NARROW = "R_hand_narrow3_Link"
WIDE = "R_hand_wide3_Link"
EE = "R_omnipicker_base_link"
TABLE = "m0_table"
BOTTLE = "m0_bottle"
REFERENCE_APERTURE = 0.3335
GRASP_CENTER_Z = 0.92225
STANDOFF_M = 0.12
APPROACH_SAMPLES = 25
CLOSURE_SAMPLES = 101
IK_POSITION_TOL_M = 0.001
IK_ROTATION_TOL_RAD = math.radians(1.0)
MAX_INTENDED_PENETRATION_M = 0.002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def named_id(model, kind, name: str) -> int:
    ident = mujoco.mj_name2id(model, kind, name)
    if ident < 0:
        raise RuntimeError(f"Missing compiled {kind.name} {name}")
    return int(ident)


def body_id(model, name: str) -> int:
    return named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_qadr(model, name: str) -> int:
    jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def body_name(model, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"


def aperture_target(base, aperture: float) -> tuple[float, float]:
    targets = base.aperture_targets(float(aperture))
    return (float(targets["right_claw_joint_target_rad"]),
            float(targets["R_hand_wide1_joint_target_rad"]))


def set_aperture(base, model, data, aperture: float) -> None:
    driver, follower = aperture_target(base, aperture)
    data.qpos[joint_qadr(model, DRIVER)] = driver
    data.qpos[joint_qadr(model, FOLLOWER)] = follower


def collision_geoms(model, body_names: set[str] | None = None) -> list[int]:
    result = []
    for gid in range(model.ngeom):
        if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
            continue
        name = body_name(model, int(model.geom_bodyid[gid]))
        if body_names is None or name in body_names:
            result.append(gid)
    return result


def geoms_on_body(model, name: str) -> list[int]:
    bid = body_id(model, name)
    return [g for g in collision_geoms(model) if int(model.geom_bodyid[g]) == bid]


def geom_distance(model, data, g1: int, g2: int) -> dict[str, Any]:
    segment = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, int(g1), int(g2), 1.0, segment))
    return {
        "geom1": geom_name(model, g1), "body1": body_name(model, int(model.geom_bodyid[g1])),
        "geom2": geom_name(model, g2), "body2": body_name(model, int(model.geom_bodyid[g2])),
        "signed_distance_m": distance,
        "witness1_world_m": segment[:3].tolist(), "witness2_world_m": segment[3:].tolist(),
    }


def pair_minimum(model, data, first: list[int], second: list[int]) -> dict[str, Any] | None:
    rows = [geom_distance(model, data, a, b) for a in first for b in second]
    return min(rows, key=lambda row: row["signed_distance_m"]) if rows else None


def active_contacts(model, data) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        rows.append({
            "geom1": geom_name(model, g1), "body1": body_name(model, int(model.geom_bodyid[g1])),
            "geom2": geom_name(model, g2), "body2": body_name(model, int(model.geom_bodyid[g2])),
            "distance_m": float(contact.dist), "penetration_m": max(0.0, -float(contact.dist)),
        })
    return rows


def source_joint_limits(urdf: Path, model, data) -> dict[str, Any]:
    root = ET.parse(urdf).getroot()
    rows = []
    for source in root.findall("joint"):
        limit = source.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        name = source.get("name", "")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        compiled = model.jnt_range[jid].astype(float).tolist()
        rows.append({
            "joint": name, "source_range_rad": [lower, upper], "compiled_range_rad": compiled,
            "qpos_rad": q, "lower_margin_rad": q - lower, "upper_margin_rad": upper - q,
            "within_source_limits": lower - 1e-10 <= q <= upper + 1e-10,
            "compiled_range_matches_source": np.allclose(compiled, [lower, upper], atol=1e-12, rtol=0.0),
        })
    return {
        "all_within_source_limits": all(row["within_source_limits"] for row in rows),
        "all_compiled_ranges_match_source": all(row["compiled_range_matches_source"] for row in rows),
        "joint_rows": rows,
    }


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    return 0.5 * np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])


def solve_ik(base, model, seed_data, position: np.ndarray, rotation: np.ndarray,
             aperture: float, source_limits: dict[str, list[float]],
             label: str, max_iterations: int = 400) -> tuple[Any, dict[str, Any]]:
    data = mujoco.MjData(model)
    data.qpos[:] = seed_data.qpos
    data.qvel[:] = 0.0
    set_aperture(base, model, data, aperture)
    arm_qadrs = [joint_qadr(model, name) for name in RIGHT_ARM]
    arm_jids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM]
    dofs = [int(model.jnt_dofadr[jid]) for jid in arm_jids]
    end_id = body_id(model, EE)
    records = []
    scale = 0.15
    x0 = np.asarray([float(data.qpos[qadr]) for qadr in arm_qadrs])
    lower = np.asarray([source_limits[name][0] for name in RIGHT_ARM], dtype=float)
    upper = np.asarray([source_limits[name][1] for name in RIGHT_ARM], dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        data.qpos[:] = seed_data.qpos
        data.qvel[:] = 0.0
        for qadr, value in zip(arm_qadrs, q):
            data.qpos[qadr] = float(value)
        set_aperture(base, model, data, aperture)
        mujoco.mj_forward(model, data)
        current_p = data.xpos[end_id].copy()
        current_r = data.xmat[end_id].reshape(3, 3).copy()
        ep = position - current_p
        er = Rotation.from_matrix(rotation @ current_r.T).as_rotvec()
        records.append({
            "label": label, "evaluation": len(records),
            "position_error_m": float(np.linalg.norm(ep)),
            "rotation_error_rad": float(np.linalg.norm(er)),
            "arm_qpos_rad": {name: float(value) for name, value in zip(RIGHT_ARM, q)},
        })
        return np.concatenate((ep, scale * er))

    optimized = least_squares(
        residual, x0, bounds=(lower, upper), method="trf", jac="2-point",
        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=max_iterations,
    )
    data.qpos[:] = seed_data.qpos
    data.qvel[:] = 0.0
    for qadr, value in zip(arm_qadrs, optimized.x):
        data.qpos[qadr] = float(value)
    set_aperture(base, model, data, aperture)
    mujoco.mj_forward(model, data)
    actual_position = data.xpos[end_id].copy()
    actual_rotation = data.xmat[end_id].reshape(3, 3).copy()
    ep = position - actual_position
    er = Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec()
    accepted = float(np.linalg.norm(ep)) <= IK_POSITION_TOL_M and float(np.linalg.norm(er)) <= IK_ROTATION_TOL_RAD
    limit_margins = {
        name: {
            "qpos_rad": float(data.qpos[qadr]),
            "lower_margin_rad": float(data.qpos[qadr] - source_limits[name][0]),
            "upper_margin_rad": float(source_limits[name][1] - data.qpos[qadr]),
            "source_range_rad": source_limits[name],
        }
        for name, qadr in zip(RIGHT_ARM, arm_qadrs)
    }
    result = {
        "label": label, "converged": accepted,
        "position_error_m": float(np.linalg.norm(ep)),
        "position_error_vector_world_m": ep.astype(float).tolist(),
        "rotation_error_rad": float(np.linalg.norm(er)),
        "rotation_error_vector_world_rad": er.astype(float).tolist(),
        "target_ee_position_world_m": position.astype(float).tolist(),
        "actual_ee_position_world_m": actual_position.astype(float).tolist(),
        "target_ee_rotation_world": rotation.astype(float).tolist(),
        "actual_ee_rotation_world": actual_rotation.astype(float).tolist(),
        "arm_qpos_rad": {name: float(data.qpos[joint_qadr(model, name)]) for name in RIGHT_ARM},
        "source_limit_margins": limit_margins,
        "iterations": len(records),
        "optimizer_success": bool(optimized.success), "optimizer_status": int(optimized.status),
        "optimizer_message": str(optimized.message), "optimizer_cost": float(optimized.cost),
        "optimizer_optimality": float(optimized.optimality), "optimizer_nfev": int(optimized.nfev),
    }
    return data, {"result": result, "iterations": records}


def local_channel(base, model, aperture: float) -> dict[str, Any]:
    data = base.initialize(model)
    set_aperture(base, model, data, aperture)
    mujoco.mj_forward(model, data)
    narrow_geom = geoms_on_body(model, NARROW)[0]
    wide_geom = geoms_on_body(model, WIDE)[0]
    segment = np.zeros(6, dtype=float)
    gap = float(mujoco.mj_geomDistance(model, data, narrow_geom, wide_geom, 1.0, segment))
    ee_id = body_id(model, EE)
    ee_p = data.xpos[ee_id].copy()
    ee_r = data.xmat[ee_id].reshape(3, 3).copy()
    midpoint = 0.5 * (segment[:3] + segment[3:])
    jaw_world = data.xpos[body_id(model, WIDE)] - data.xpos[body_id(model, NARROW)]
    jaw_world /= np.linalg.norm(jaw_world)
    local_jaw = ee_r.T @ jaw_world
    local_midpoint = ee_r.T @ (midpoint - ee_p)
    local_tool = local_midpoint / np.linalg.norm(local_midpoint)
    local_jaw = local_jaw - np.dot(local_jaw, local_tool) * local_tool
    local_jaw /= np.linalg.norm(local_jaw)
    return {
        "aperture_ratio": aperture, "jaw_gap_m": gap,
        "witness_narrow_world_m": segment[:3].tolist(),
        "witness_wide_world_m": segment[3:].tolist(),
        "midpoint_local_to_ee_m": local_midpoint.tolist(),
        "jaw_axis_local_narrow_to_wide": local_jaw.tolist(),
        "tool_axis_local_ee_to_fingers": local_tool.tolist(),
        "ee_rotation_at_source_initial_pose": ee_r.tolist(),
    }


def candidate_rotation(channel: dict[str, Any], approach: np.ndarray,
                       jaw_world: np.ndarray) -> np.ndarray:
    tool_world = approach / np.linalg.norm(approach)
    jaw_world = jaw_world - np.dot(jaw_world, tool_world) * tool_world
    jaw_world /= np.linalg.norm(jaw_world)
    local_jaw = np.asarray(channel["jaw_axis_local_narrow_to_wide"], dtype=float)
    local_tool = np.asarray(channel["tool_axis_local_ee_to_fingers"], dtype=float)
    local_jaw -= np.dot(local_jaw, local_tool) * local_tool
    local_jaw /= np.linalg.norm(local_jaw)
    local_aux = np.cross(local_jaw, local_tool)
    world_aux = np.cross(jaw_world, tool_world)
    local_basis = np.column_stack((local_jaw, local_tool, local_aux))
    world_basis = np.column_stack((jaw_world, tool_world, world_aux))
    result = world_basis @ local_basis.T
    if np.linalg.det(result) < 0.999 or not np.allclose(result.T @ result, np.eye(3), atol=1e-10):
        raise RuntimeError("Candidate tool orientation is not a proper rotation")
    return result


def target_ee_position(base, model, rotation: np.ndarray, tcp_world: np.ndarray,
                       aperture: float, channel_cache: dict[float, dict[str, Any]],
                       jaw_world: np.ndarray, lateral_offset: float = 0.0) -> np.ndarray:
    if aperture not in channel_cache:
        channel_cache[aperture] = local_channel(base, model, aperture)
    local_mid = np.asarray(channel_cache[aperture]["midpoint_local_to_ee_m"], dtype=float)
    return tcp_world - rotation @ local_mid + lateral_offset * jaw_world


def pair_metrics(base, model, data) -> dict[str, Any]:
    robot = [g for g in collision_geoms(model)
             if body_name(model, int(model.geom_bodyid[g])) not in {"world", TABLE, BOTTLE}]
    bottle = geoms_on_body(model, BOTTLE)
    table = geoms_on_body(model, TABLE)
    pad_bodies = {NARROW, WIDE}
    robot_bottle_rows = [geom_distance(model, data, rg, bg) for rg in robot for bg in bottle]
    robot_table_rows = [geom_distance(model, data, rg, tg) for rg in robot for tg in table]
    nonpad_bottle = [row for row in robot_bottle_rows if row["body1"] not in pad_bodies]
    pads_body = {body: min((row for row in robot_bottle_rows
                           if row["body1"] == body and row["geom2"] == "bottle_body"),
                          key=lambda row: row["signed_distance_m"], default=None)
                 for body in pad_bodies}
    contacts = active_contacts(model, data)
    self_contacts = [row for row in contacts
                     if row["body1"] not in {"world", TABLE, BOTTLE}
                     and row["body2"] not in {"world", TABLE, BOTTLE}]
    table_contacts = [row for row in contacts if TABLE in {row["body1"], row["body2"]}]
    floor_contacts = [row for row in contacts if "floor" in {row["geom1"], row["geom2"]}]
    return {
        "minimum_robot_bottle": min(robot_bottle_rows, key=lambda row: row["signed_distance_m"]) if robot_bottle_rows else None,
        "minimum_nonpad_robot_bottle": min(nonpad_bottle, key=lambda row: row["signed_distance_m"]) if nonpad_bottle else None,
        "minimum_robot_table": min(robot_table_rows, key=lambda row: row["signed_distance_m"]) if robot_table_rows else None,
        "pad_to_bottle_body": pads_body,
        "active_contacts": contacts,
        "robot_self_contacts": self_contacts,
        "robot_table_contacts": table_contacts,
        "robot_floor_contacts": floor_contacts,
        "maximum_self_penetration_m": max((row["penetration_m"] for row in self_contacts), default=0.0),
        "maximum_table_penetration_m": max((row["penetration_m"] for row in table_contacts), default=0.0),
    }


def contact_normal_metrics(model, data, pad_geoms: dict[str, int], bottle_body_geom: int) -> dict[str, Any]:
    records = {}
    vectors = []
    for family, gid in pad_geoms.items():
        row = geom_distance(model, data, gid, bottle_body_geom)
        v = np.asarray(row["witness2_world_m"]) - np.asarray(row["witness1_world_m"])
        norm = float(np.linalg.norm(v))
        unit = (v / norm).tolist() if norm > 1e-9 else None
        records[family] = {**row, "unit_pad_to_bottle_normal": unit}
        if unit is not None:
            vectors.append(np.asarray(unit))
    angle = None
    resultant_fraction = None
    if len(vectors) == 2:
        dot = float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))
        angle = math.degrees(math.acos(dot))
        resultant_fraction = float(np.linalg.norm(vectors[0] + vectors[1]) / 2.0)
    return {"contacts": records, "normal_angle_deg": angle,
            "equal_normal_resultant_fraction_of_one_pad": resultant_fraction,
            "interpretation": "Closest-point vectors are geometric pre-contact normals; they are not measured contact forces."}


def limit_check(data, model, source_limits: dict[str, list[float]]) -> dict[str, Any]:
    rows = {}
    for name, interval in source_limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        rows[name] = {"qpos_rad": q, "source_range_rad": interval,
                      "within_source_limits": interval[0] - 1e-10 <= q <= interval[1] + 1e-10}
    return {"all_within_source_limits": all(row["within_source_limits"] for row in rows.values()),
            "joint_rows": rows}


def render(model, data, path: Path, view: str, target: np.ndarray) -> None:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    if view == "overview":
        camera.distance, camera.azimuth, camera.elevation = 2.5, 145.0, -14.0
        width, height = 1280, 800
    elif view == "side":
        camera.distance, camera.azimuth, camera.elevation = 2.0, 90.0, -8.0
        width, height = 1280, 800
    else:
        camera.distance, camera.azimuth, camera.elevation = 0.60, 145.0, -18.0
        width, height = 1280, 800
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render().copy()).save(path)
    renderer.close()


def candidate_static(base, model, source_limits, candidate_spec: dict[str, Any],
                     output: Path) -> dict[str, Any]:
    name = candidate_spec["name"]
    initial = base.initialize(model)
    elbow_seed = 0.5 * sum(source_limits["right_elbow_joint"])
    initial.qpos[joint_qadr(model, "right_elbow_joint")] = elbow_seed
    mujoco.mj_forward(model, initial)
    channel = local_channel(base, model, REFERENCE_APERTURE)
    approach = np.asarray(candidate_spec["approach_world"], dtype=float)
    jaw_world = np.asarray(candidate_spec["jaw_axis_world_narrow_to_wide"], dtype=float)
    rotation = candidate_rotation(channel, approach, jaw_world)
    tcp = np.array([0.300, 0.000, GRASP_CENTER_Z], dtype=float)
    cache = {1.0: local_channel(base, model, 1.0), REFERENCE_APERTURE: channel}
    trace = []
    ik_records = []
    seed = initial
    path_rows = []
    for index, progress in enumerate(np.linspace(0.0, 1.0, APPROACH_SAMPLES)):
        tcp_at = tcp - STANDOFF_M * (1.0 - float(progress)) * approach
        target = target_ee_position(base, model, rotation, tcp_at, 1.0, cache, jaw_world)
        data, ik = solve_ik(base, model, seed, target, rotation, 1.0, source_limits,
                            f"{name}:approach:{index:02d}")
        ik_records.extend(ik["iterations"])
        if not ik["result"]["converged"]:
            metrics = pair_metrics(base, model, data)
            limits = limit_check(data, model, source_limits)
            failed_pose = {
                "candidate": name, "phase": "NEAREST_FAILED_IK_PREGRASP",
                "target_tcp_world_m": tcp_at.tolist(), "ee_target_world_m": target.tolist(),
                "ik": ik["result"], "minimum_robot_bottle": metrics["minimum_robot_bottle"],
                "minimum_nonpad_robot_bottle": metrics["minimum_nonpad_robot_bottle"],
                "minimum_robot_table": metrics["minimum_robot_table"],
                "pad_to_bottle_body": metrics["pad_to_bottle_body"],
                "active_contacts": metrics["active_contacts"],
                "robot_self_contacts": metrics["robot_self_contacts"],
                "robot_table_contacts": metrics["robot_table_contacts"],
                "source_limits": limits,
                "interpretation": "Nearest bounded IK residual state; not a valid pregrasp or grasp.",
            }
            path_rows.append({"sample": index, "progress": float(progress), "ik": ik["result"],
                              "nearest_pose_geometry": failed_pose,
                              "passed": False, "failure": "IK_NOT_CONVERGED"})
            trace.append(failed_pose)
            for view, suffix, lookat in (
                ("overview", "overview", np.array([0.30, 0.10, 1.03])),
                ("side", "side", np.array([0.30, 0.05, 0.95])),
                ("closeup", "finger_contact_closeup", np.array([0.30, 0.0, GRASP_CENTER_Z])),
            ):
                render(model, data, output / f"{name}_nearest_failed_ik_{suffix}.png", view, lookat)
            break
        metrics = pair_metrics(base, model, data)
        limits = limit_check(data, model, source_limits)
        min_bottle = metrics["minimum_robot_bottle"]["signed_distance_m"]
        min_table = metrics["minimum_robot_table"]["signed_distance_m"]
        nonpad = metrics["minimum_nonpad_robot_bottle"]["signed_distance_m"]
        active_robot_bottle = [row for row in metrics["active_contacts"]
                               if BOTTLE in {row["body1"], row["body2"]}]
        passed = (limits["all_within_source_limits"] and min_bottle > 0.0 and
                  min_table > 0.0 and nonpad > 0.0 and not active_robot_bottle and
                  metrics["maximum_self_penetration_m"] <= 1e-8 and
                  metrics["maximum_table_penetration_m"] <= 1e-8)
        row = {
            "candidate": name, "phase": "OPEN_APPROACH", "sample": index,
            "progress_0_to_1": float(progress), "tcp_world_m": tcp_at.tolist(),
            "ee_target_world_m": target.tolist(), "ik": ik["result"],
            "minimum_robot_bottle": metrics["minimum_robot_bottle"],
            "minimum_nonpad_robot_bottle": metrics["minimum_nonpad_robot_bottle"],
            "minimum_robot_table": metrics["minimum_robot_table"],
            "robot_self_contacts": metrics["robot_self_contacts"],
            "robot_table_contacts": metrics["robot_table_contacts"],
            "source_limits": limits, "passed": passed,
        }
        path_rows.append(row)
        trace.append(row)
        if not passed:
            break
        seed = data

    approach_pass = len(path_rows) == APPROACH_SAMPLES and all(row.get("passed", False) for row in path_rows)
    if not approach_pass:
        write_jsonl(output / f"{name}_ik_iterations.jsonl", ik_records)
        write_jsonl(output / f"{name}_geometry_trace.jsonl", trace)
        return {"candidate": candidate_spec, "approach_pass": False,
                "approach_rows": path_rows, "contact_pass": False,
                "failure": path_rows[-1].get("failure", "APPROACH_COLLISION_OR_LIMIT") if path_rows else "NO_APPROACH_ROWS",
                "ik_iterations_file": f"{name}_ik_iterations.jsonl",
                "geometry_trace_file": f"{name}_geometry_trace.jsonl"}

    # Keep the single seeded IK branch and use its final OPEN arm pose for closure.
    open_end = seed
    contact_target = target_ee_position(base, model, rotation, tcp, REFERENCE_APERTURE, cache, jaw_world)
    contact_data, contact_ik = solve_ik(base, model, open_end, contact_target, rotation,
                                        REFERENCE_APERTURE, source_limits, f"{name}:contact_endpoint")
    ik_records.extend(contact_ik["iterations"])
    if not contact_ik["result"]["converged"]:
        write_jsonl(output / f"{name}_ik_iterations.jsonl", ik_records)
        write_jsonl(output / f"{name}_geometry_trace.jsonl", trace)
        return {"candidate": candidate_spec, "approach_pass": True, "contact_pass": False,
                "contact_ik": contact_ik["result"], "failure": "CONTACT_IK_NOT_CONVERGED",
                "ik_iterations_file": f"{name}_ik_iterations.jsonl",
                "geometry_trace_file": f"{name}_geometry_trace.jsonl"}

    # Recheck source frame with OPEN jaws, then close the source pair at fixed arm pose.
    closure = []
    closure_seed = contact_data
    for index, aperture in enumerate(np.linspace(1.0, REFERENCE_APERTURE, CLOSURE_SAMPLES)):
        data = mujoco.MjData(model)
        data.qpos[:] = closure_seed.qpos
        data.qvel[:] = 0.0
        set_aperture(base, model, data, float(aperture))
        mujoco.mj_forward(model, data)
        metrics = pair_metrics(base, model, data)
        limits = limit_check(data, model, source_limits)
        d_n = metrics["pad_to_bottle_body"].get(NARROW)
        d_w = metrics["pad_to_bottle_body"].get(WIDE)
        nonpad = metrics["minimum_nonpad_robot_bottle"]
        unexpected = []
        for row in metrics["active_contacts"]:
            if BOTTLE not in {row["body1"], row["body2"]}:
                continue
            robot_body = row["body1"] if row["body2"] == BOTTLE else row["body2"]
            bottle_geom = row["geom2"] if row["body2"] == BOTTLE else row["geom1"]
            if robot_body not in {NARROW, WIDE} or bottle_geom != "bottle_body":
                unexpected.append(row)
        closure.append({
            "candidate": name, "phase": "STATIC_CLOSURE", "sample": index,
            "aperture_ratio": float(aperture), "jaw_distances": {NARROW: d_n, WIDE: d_w},
            "minimum_nonpad_robot_bottle": nonpad,
            "minimum_robot_table": metrics["minimum_robot_table"],
            "unexpected_robot_bottle_contacts": unexpected,
            "robot_self_contacts": metrics["robot_self_contacts"],
            "robot_table_contacts": metrics["robot_table_contacts"],
            "source_limits": limits,
        })

    final = closure[-1]
    d_n = final["jaw_distances"][NARROW]
    d_w = final["jaw_distances"][WIDE]
    nonpad_rows = [row["minimum_nonpad_robot_bottle"] for row in closure]
    min_nonpad = min(nonpad_rows, key=lambda row: row["signed_distance_m"])
    min_table = min((row["minimum_robot_table"] for row in closure), key=lambda row: row["signed_distance_m"])
    unintended = [entry for row in closure for entry in row["unexpected_robot_bottle_contacts"]]
    pair_limits_pass = all(row["source_limits"]["all_within_source_limits"] for row in closure)
    intended_contact_pass = (
        d_n is not None and d_w is not None
        and d_n["signed_distance_m"] <= 0.0 and d_w["signed_distance_m"] <= 0.0
        and max(-d_n["signed_distance_m"], -d_w["signed_distance_m"]) <= MAX_INTENDED_PENETRATION_M
        and min_nonpad["signed_distance_m"] > 0.0 and not unintended
    )
    closure_monotone = all(
        closure[i + 1]["jaw_distances"][body]["signed_distance_m"]
        <= closure[i]["jaw_distances"][body]["signed_distance_m"] + 2e-5
        for body in (NARROW, WIDE) for i in range(len(closure) - 1)
        if closure[i]["jaw_distances"][body] is not None and closure[i + 1]["jaw_distances"][body] is not None
    )
    contact_normals = contact_normal_metrics(
        model, contact_data,
        {NARROW: geoms_on_body(model, NARROW)[0], WIDE: geoms_on_body(model, WIDE)[0]},
        geoms_on_body(model, BOTTLE)[0],
    )
    final_data = mujoco.MjData(model)
    final_data.qpos[:] = closure_seed.qpos
    final_data.qvel[:] = 0.0
    set_aperture(base, model, final_data, REFERENCE_APERTURE)
    mujoco.mj_forward(model, final_data)
    final_metrics = pair_metrics(base, model, final_data)
    gates = {
        "approach_ik_and_clearance": approach_pass,
        "source_limits_all_candidate_states": pair_limits_pass,
        "both_original_distal_collision_geoms_reach_bottle_body": intended_contact_pass,
        "closure_clear_except_intended_pad_body_contact": intended_contact_pass,
        "jaw_clearance_monotonic": closure_monotone,
        "wrist_table_clearance_positive": min_table["signed_distance_m"] > 0.0,
        "no_robot_self_penetration_at_grasp": final_metrics["maximum_self_penetration_m"] <= 1e-8,
        "no_robot_table_penetration_at_grasp": final_metrics["maximum_table_penetration_m"] <= 1e-8,
    }
    if contact_normals["normal_angle_deg"] is not None:
        mu = 1.4
        mg = 0.570 * 9.81
        n_each = 2.5
        vertical_capacity = mu * n_each * sum(
            math.sqrt(max(0.0, 1.0 - float(np.asarray(contact_normals["contacts"][body]["unit_pad_to_bottle_normal"])[2])**2))
            for body in (NARROW, WIDE)
        )
        contact_normals["idealized_capacity_screen"] = {
            "assumed_effective_friction": mu,
            "assumed_normal_force_per_pad_N": n_each,
            "idealized_vertical_friction_capacity_N": vertical_capacity,
            "bottle_weight_N": mg,
            "capacity_over_weight": vertical_capacity / mg,
            "warning": "Coulomb upper-bound screen only; not measured force or lift evidence.",
        }
    candidate_result = {
        "candidate": candidate_spec,
        "target_tcp_world_m": tcp.tolist(),
        "target_ee_rotation_world": rotation.tolist(),
        "target_ee_position_world_m": contact_target.tolist(),
        "contact_ik": contact_ik["result"],
        "pregrasp_arm_qpos_rad": path_rows[0]["ik"]["arm_qpos_rad"],
        "approach_end_arm_qpos_rad": path_rows[-1]["ik"]["arm_qpos_rad"],
        "contact_arm_qpos_rad": contact_ik["result"]["arm_qpos_rad"],
        "ik_initial_arm_seed_rad": {**{joint: 0.0 for joint in RIGHT_ARM},
                                     "right_elbow_joint": elbow_seed},
        "approach_pass": approach_pass,
        "approach_minimum_robot_bottle_m": min(
            row["minimum_robot_bottle"]["signed_distance_m"] for row in path_rows if row.get("minimum_robot_bottle")),
        "approach_minimum_robot_table_m": min(
            row["minimum_robot_table"]["signed_distance_m"] for row in path_rows if row.get("minimum_robot_table")),
        "closure_minimum_nonpad_robot_bottle_m": min_nonpad["signed_distance_m"],
        "closure_minimum_robot_table_m": min_table["signed_distance_m"],
        "final_pad_to_body_distances_m": {NARROW: d_n["signed_distance_m"], WIDE: d_w["signed_distance_m"]},
        "final_max_intended_pad_penetration_m": max(-d_n["signed_distance_m"], -d_w["signed_distance_m"]),
        "contact_normal_geometry": contact_normals,
        "source_joint_limits_at_grasp": limit_check(final_data, model, source_limits),
        "active_contacts_at_grasp": final_metrics["active_contacts"],
        "robot_self_contacts_at_grasp": final_metrics["robot_self_contacts"],
        "robot_table_contacts_at_grasp": final_metrics["robot_table_contacts"],
        "gates": gates,
        "static_geometry_pass": all(gates.values()),
        "physics_steps": 0,
        "rollout_object_qpos_writes": 0,
    }
    write_jsonl(output / f"{name}_ik_iterations.jsonl", ik_records)
    write_jsonl(output / f"{name}_geometry_trace.jsonl", trace + closure)
    return {**candidate_result,
            "ik_iterations_file": f"{name}_ik_iterations.jsonl",
            "geometry_trace_file": f"{name}_geometry_trace.jsonl",
            "_final_data": final_data,
            "_pregrasp_q": path_rows[0]["ik"]["arm_qpos_rad"],
            "_grasp_q": contact_ik["result"]["arm_qpos_rad"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, required=True)
    parser.add_argument("--canonical-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urdf = args.x2_root / URDF_REL
    baseline_path = REPO / "scripts/research/issue46_omnipicker_1dof_m0.py"
    if not urdf.is_file() or not args.canonical_helper.is_file():
        raise FileNotFoundError("Pinned source URDF or canonical scene helper is missing")
    vendor_sha = subprocess.check_output(["git", "-C", str(args.x2_root), "rev-parse", "HEAD"], text=True).strip()
    vendor_dirty = subprocess.check_output(["git", "-C", str(args.x2_root), "status", "--porcelain"], text=True).strip()
    if vendor_sha != X2_PIN or vendor_dirty:
        raise RuntimeError(f"Pinned source checkout mismatch: {vendor_sha=}, dirty={bool(vendor_dirty)}")
    if sha256(urdf) != URDF_SHA256 or sha256(args.canonical_helper) != CANONICAL_SHA256:
        raise RuntimeError("Pinned URDF or canonical scene helper hash mismatch")
    if mujoco.__version__ != "3.3.6" or mujoco.mj_versionString() != "3.3.6":
        raise RuntimeError("Expected MuJoCo Python/native 3.3.6")

    base = load_module("issue46_native_baseline", baseline_path)
    base.X2_ROOT = args.x2_root
    base.STATIC_ROOT = args.canonical_helper.parent
    model, build_info = base.build_model("bottle_hold")
    build_info["source_position_limits_rad"] = {name: list(value)
                                                 for name, value in build_info["source_position_limits_rad"].items()}
    initial = base.initialize(model)
    source_limits = build_info["source_position_limits_rad"]
    compiled_limits = source_joint_limits(urdf, model, initial)
    if not compiled_limits["all_compiled_ranges_match_source"]:
        raise RuntimeError("Compiled joint ranges do not preserve the source URDF ranges")
    bottle_body = body_id(model, BOTTLE)
    bottle_joint = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bq = int(model.jnt_qposadr[bottle_joint])
    bottle_qpos0 = model.qpos0[bq:bq + 7].astype(float).tolist()
    mujoco.mj_forward(model, initial)
    initial_scene = pair_metrics(base, model, initial)
    channel = local_channel(base, model, REFERENCE_APERTURE)

    candidates = [
        {"name": "A_front_approach_minus_y", "approach_world": [0.0, -1.0, 0.0],
         "jaw_axis_world_narrow_to_wide": [1.0, 0.0, 0.0],
         "derivation": "Reuse the prior source-mesh contact channel orientation: from the robot-facing table edge along -Y, with jaws spanning X."},
        {"name": "B_orthogonal_side_approach_minus_x", "approach_world": [-1.0, 0.0, 0.0],
         "jaw_axis_world_narrow_to_wide": [0.0, 1.0, 0.0],
         "derivation": "One orthogonal, mechanically distinct side insertion: tool axis along -X and jaw span along Y. This tests whether a changed insertion direction clears the wrist and changes pad normal alignment."},
    ]
    results = []
    for spec in candidates:
        result = candidate_static(base, model, source_limits, spec, args.output)
        if result.get("static_geometry_pass"):
            final_data = result.pop("_final_data")
            result.pop("_pregrasp_q", None)
            result.pop("_grasp_q", None)
            render(model, final_data, args.output / f"{spec['name']}_overview.png", "overview",
                   np.array([0.30, 0.10, 1.03]))
            render(model, final_data, args.output / f"{spec['name']}_side.png", "side",
                   np.array([0.30, 0.05, 0.95]))
            render(model, final_data, args.output / f"{spec['name']}_finger_contact_closeup.png", "closeup",
                   np.array([0.30, 0.0, GRASP_CENTER_Z]))
        else:
            result.pop("_final_data", None)
            result.pop("_pregrasp_q", None)
            result.pop("_grasp_q", None)
        results.append(result)

    result = {
        "task": "Issue #46 native AgiBot X2 OmniPicker source-geometry feasibility",
        "classification": "SOURCE_FAITHFUL_STATIC_GEOMETRY",
        "source": {
            "repository": "https://github.com/AgibotTech/agibot_x2_urdf",
            "commit": vendor_sha, "license": "Mulan PSL v2",
            "urdf": str(URDF_REL), "urdf_sha256": sha256(urdf),
            "canonical_scene_helper_sha256": sha256(args.canonical_helper),
            "narrow_collision_mesh_sha256": "0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0",
            "wide_collision_mesh_sha256": "a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb",
        },
        "runtime": {"python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(), "timestep_s": float(model.opt.timestep),
                    "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
                    "solver_iterations": int(model.opt.iterations), "gravity_m_s2": model.opt.gravity.tolist(),
                    "native_library_path": str(Path(mujoco.__file__).with_name("libmujoco.so.3.3.6")),
                    "native_library_sha256": sha256(Path(mujoco.__file__).with_name("libmujoco.so.3.3.6"))},
        "model": {"source_geometry_preserved": True,
                  "source_limits_preserved": compiled_limits["all_compiled_ranges_match_source"],
                  "source_joint_limit_rows_at_initialization": compiled_limits,
                  "historical_simulation_only_exclusions": list(base.STATIC_COLLISION_EXCLUSIONS),
                  "additional_collision_exclusions": [],
                  "control_classification": "SIMULATION_ONLY_M0 paired-jaw position-servo emulation; source geometry and joint limits remain unchanged.",
                  "ik_initialization": "Source-valid right-elbow midpoint seed; the URDF neutral elbow q=0 is exactly at its upper limit and blocked the first IK line-search step.",
                  "source_mimic": {"driver": DRIVER, "follower": FOLLOWER, "multiplier": -1.0, "offset_rad": 0.0},
                  "static_physics_steps": 0,
                  "initial_active_contacts": initial_scene["active_contacts"],
                  "initial_robot_self_contacts": initial_scene["robot_self_contacts"],
                  "initial_robot_table_contacts": initial_scene["robot_table_contacts"],
                  "initial_robot_floor_contacts": initial_scene["robot_floor_contacts"],
                  "local_channel_reference": channel},
        "canonical_scene": {"bottle_freejoint": True, "bottle_qpos0": bottle_qpos0,
                            "bottle_dimensions_m": [0.070, 0.2445], "bottle_mass_kg": 0.570,
                            "bottle_start_body_position_m": [0.300, 0.000, 0.9175],
                            "bottle_grasp_tcp_world_m": [0.300, 0.000, GRASP_CENTER_Z],
                            "bottle_components": ["bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"],
                            "table_center_xy_m": [0.300, -0.100], "table_half_extents_xy_m": [0.200, 0.200],
                            "table_top_z_m": 0.800, "bottle_friction": [1.4, 0.02, 0.001]},
        "candidate_count": len(results), "candidates": results,
        "source_faithful_static_grasp_pass": any(row.get("static_geometry_pass", False) for row in results),
        "physical_bilateral_contact": "NOT RUN",
        "physical_load_transfer": "NOT RUN",
        "next_action": "If a candidate passes all static gates, run one bounded native-model contact/load-transfer test with the existing SIMULATION_ONLY_M0 paired-jaw controller; otherwise stop and preserve the exact first geometric/IK failure.",
        "production_model_changed": False,
        "pr58_modified": False,
    }
    write_json(args.output / "result.json", result)
    return 0 if result["source_faithful_static_grasp_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
