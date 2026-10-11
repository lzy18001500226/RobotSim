#!/usr/bin/env python3
"""Bounded source-limited X2 OmniPicker workcell and grasp-corridor study."""

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
from scipy.optimize import least_squares, minimize
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
GRASP_CENTER_Z = 0.92225
STANDOFF_M = 0.120
APPROACH_SAMPLES = 25
CLOSURE_SAMPLES = 101
POSITION_TOL_M = 0.001
ROTATION_TOL_RAD = math.radians(1.0)
COLLISION_MARGIN_M = 0.001
MAX_INTENDED_PENETRATION_M = 0.002
DEFAULT_LAYOUTS = (
    {"name": "A_front_nominal_heading", "yaw_deg": -90.0,
     "shoulder_target_xy_m": [0.300, 0.320],
     "derivation": "Right shoulder pivot aligned to bottle x and the original robot-front work line; baseline front-facing yaw."},
    {"name": "A_front_table_axis_10deg", "yaw_deg": -100.0,
     "shoulder_target_xy_m": [0.300, 0.370],
     "derivation": "Move the right shoulder 50 mm back from the table edge and rotate the torso 10 deg toward the table-center ray."},
    {"name": "A_front_table_axis_15deg", "yaw_deg": -105.0,
     "shoulder_target_xy_m": [0.300, 0.420],
     "derivation": "Move the right shoulder 100 mm back from the first work line; align torso heading within about 5 deg of the table-center ray."},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def qadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Missing compiled joint {name}")
    return int(model.jnt_qposadr[jid])


def body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing compiled body {name}")
    return int(bid)


def body_name(model, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"


def geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def geom_distance(model, data, geom1: int, geom2: int) -> dict[str, Any]:
    witness = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, int(geom1), int(geom2), 1.0, witness))
    return {
        "geom1": geom_name(model, geom1), "body1": body_name(model, int(model.geom_bodyid[geom1])),
        "geom2": geom_name(model, geom2), "body2": body_name(model, int(model.geom_bodyid[geom2])),
        "signed_distance_m": distance,
        "witness1_world_m": witness[:3].tolist(), "witness2_world_m": witness[3:].tolist(),
    }


def minimum_pair(model, data, geoms1: list[int], geoms2: list[int]) -> dict[str, Any] | None:
    rows = [geom_distance(model, data, g1, g2) for g1 in geoms1 for g2 in geoms2 if g1 != g2]
    return min(rows, key=lambda row: row["signed_distance_m"]) if rows else None


def geom_ids(model, bodies: set[str] | None = None) -> list[int]:
    result = []
    for gid in range(model.ngeom):
        if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
            continue
        name = body_name(model, int(model.geom_bodyid[gid]))
        if bodies is None or name in bodies:
            result.append(gid)
    return result


def descendant_bodies(model, root_name: str) -> set[int]:
    root = body_id(model, root_name)
    result = set()
    for bid in range(model.nbody):
        current = bid
        while current > 0:
            if current == root:
                result.add(bid)
                break
            current = int(model.body_parentid[current])
    result.add(root)
    return result


def layout_base(base, x2_root: Path, helper: Path, layout: dict[str, Any]) -> dict[str, Any]:
    yaw = math.radians(float(layout["yaw_deg"]))
    base.BASE_WORLD = np.array([0.0, 0.0, 0.68], dtype=float)
    base.BASE_YAW = yaw
    probe_model, _ = base.build_model("bottle_hold")
    probe_data = base.initialize(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    shoulder = probe_data.xpos[body_id(probe_model, "right_shoulder_pitch_link")].copy()
    shoulder_offset = shoulder - base.BASE_WORLD
    shoulder_target = np.array([*layout["shoulder_target_xy_m"], shoulder[2]], dtype=float)
    base_position = np.array([
        shoulder_target[0] - shoulder_offset[0],
        shoulder_target[1] - shoulder_offset[1],
        0.68,
    ], dtype=float)
    base.BASE_WORLD = base_position
    base.BASE_YAW = yaw
    model, build = base.build_model("bottle_hold")
    data = base.initialize(model)
    mujoco.mj_forward(model, data)
    actual_shoulder = data.xpos[body_id(model, "right_shoulder_pitch_link")].copy()
    direction_to_table = np.array([0.300, -0.100]) - base_position[:2]
    table_bearing = math.degrees(math.atan2(direction_to_table[1], direction_to_table[0]))
    metrics = base_pair_metrics(base, model, data)
    return {
        "base_position_world_m": base_position.tolist(), "base_yaw_rad": yaw,
        "base_yaw_deg": float(layout["yaw_deg"]),
        "desired_shoulder_pivot_world_m": shoulder_target.tolist(),
        "actual_shoulder_pivot_world_m": actual_shoulder.tolist(),
        "shoulder_anchor_error_m": float(np.linalg.norm(actual_shoulder - shoulder_target)),
        "table_center_bearing_from_base_deg": table_bearing,
        "heading_error_to_table_center_deg": float(abs(((float(layout["yaw_deg"]) - table_bearing + 180.0) % 360.0) - 180.0)),
        "initial_scene": metrics,
        "model": model, "data": data, "build": build,
    }


def active_contacts(model, data) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        rows.append({
            "geom1": geom_name(model, g1), "body1": body_name(model, int(model.geom_bodyid[g1])),
            "geom2": geom_name(model, g2), "body2": body_name(model, int(model.geom_bodyid[g2])),
            "distance_m": float(c.dist), "penetration_m": max(0.0, -float(c.dist)),
        })
    return rows


def base_pair_metrics(base, model, data) -> dict[str, Any]:
    contacts = active_contacts(model, data)
    self_contacts = [row for row in contacts
                     if row["body1"] not in {"world", TABLE, BOTTLE}
                     and row["body2"] not in {"world", TABLE, BOTTLE}]
    table_contacts = [row for row in contacts if TABLE in {row["body1"], row["body2"]}]
    floor_contacts = [row for row in contacts if "floor" in {row["geom1"], row["geom2"]}]
    obstacles = obstacle_metrics(model, data)
    return {
        "active_contacts": contacts,
        "robot_self_contacts": self_contacts,
        "robot_table_contacts": table_contacts,
        "robot_floor_contacts": floor_contacts,
        "maximum_self_penetration_m": max((row["penetration_m"] for row in self_contacts), default=0.0),
        "maximum_table_penetration_m": max((row["penetration_m"] for row in table_contacts), default=0.0),
        "minimum_robot_table": obstacles["all_robot_to_table"],
        "minimum_robot_bottle": obstacles["nonpad_robot_to_bottle"],
        "minimum_nonpad_robot_bottle": obstacles["nonpad_robot_to_bottle"],
    }


def obstacle_metrics(model, data) -> dict[str, Any]:
    all_robot = [g for g in geom_ids(model)
                 if body_name(model, int(model.geom_bodyid[g])) not in {"world", TABLE, BOTTLE}]
    arm_bodies = descendant_bodies(model, "right_shoulder_pitch_link")
    arm_geoms = [g for g in all_robot if int(model.geom_bodyid[g]) in arm_bodies]
    table_geoms = [g for g in geom_ids(model, {TABLE})]
    bottle_geoms = [g for g in geom_ids(model, {BOTTLE})]
    pad_bodies = {NARROW, WIDE}
    nonpad_geoms = [g for g in all_robot if body_name(model, int(model.geom_bodyid[g])) not in pad_bodies]
    torso_geoms = geom_ids(model, {"torso_link"})
    wrist_geoms = geom_ids(model, {"right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"})
    return {
        "arm_to_table": minimum_pair(model, data, arm_geoms, table_geoms),
        "all_robot_to_table": minimum_pair(model, data, all_robot, table_geoms),
        "nonpad_robot_to_bottle": minimum_pair(model, data, nonpad_geoms, bottle_geoms),
        "pads_to_bottle": minimum_pair(model, data, geom_ids(model, pad_bodies), bottle_geoms),
        "torso_to_wrist": minimum_pair(model, data, torso_geoms, wrist_geoms),
        "arm_geom_count": len(arm_geoms), "robot_geom_count": len(all_robot),
    }


def source_limits(urdf: Path, model) -> dict[str, list[float]]:
    root = ET.parse(urdf).getroot()
    result = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        name = joint.get("name", "")
        result[name] = [float(limit.get("lower")), float(limit.get("upper"))]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or not np.allclose(model.jnt_range[jid], result[name], atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Compiled source joint range mismatch for {name}")
    return result


def set_aperture(base, model, data, aperture: float) -> None:
    targets = base.aperture_targets(float(aperture))
    data.qpos[qadr(model, DRIVER)] = float(targets["right_claw_joint_target_rad"])
    data.qpos[qadr(model, FOLLOWER)] = float(targets["R_hand_wide1_joint_target_rad"])


def set_arm(model, data, values: dict[str, float]) -> None:
    for name, value in values.items():
        data.qpos[qadr(model, name)] = float(value)


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def pose_only_seed(model, seed_data, target_position: np.ndarray, target_rotation: np.ndarray,
                   aperture: float, limits: dict[str, list[float]], label: str) -> tuple[Any, dict[str, Any]]:
    """Build a source-limited FK seed; collision feasibility is decided by the next stage."""
    data = mujoco.MjData(model)
    q_adrs = [qadr(model, name) for name in RIGHT_ARM]
    lower = np.array([limits[name][0] for name in RIGHT_ARM], dtype=float)
    upper = np.array([limits[name][1] for name in RIGHT_ARM], dtype=float)
    x0 = np.array([seed_data.qpos[q] for q in q_adrs], dtype=float)
    history: list[dict[str, Any]] = []

    def residual(q: np.ndarray) -> np.ndarray:
        data.qpos[:] = seed_data.qpos
        data.qvel[:] = 0.0
        set_arm(model, data, dict(zip(RIGHT_ARM, q)))
        mujoco.mj_forward(model, data)
        actual_position = data.xpos[body_id(model, EE)].copy()
        actual_rotation = data.xmat[body_id(model, EE)].reshape(3, 3).copy()
        ep = target_position - actual_position
        er = rotation_error(target_rotation, actual_rotation)
        history.append({"label": label, "evaluation": len(history),
                        "arm_qpos_rad": dict(zip(RIGHT_ARM, [float(v) for v in q])),
                        "position_error_m": float(np.linalg.norm(ep)),
                        "rotation_error_rad": float(np.linalg.norm(er))})
        return np.concatenate((ep * 1000.0, er * 100.0))

    solved = least_squares(residual, x0, bounds=(lower, upper), max_nfev=300,
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    residual(solved.x)
    pose = mujoco.MjData(model)
    pose.qpos[:] = seed_data.qpos
    pose.qvel[:] = 0.0
    set_arm(model, pose, dict(zip(RIGHT_ARM, solved.x)))
    mujoco.mj_forward(model, pose)
    actual_position = pose.xpos[body_id(model, EE)].copy()
    actual_rotation = pose.xmat[body_id(model, EE)].reshape(3, 3).copy()
    ep = target_position - actual_position
    er = rotation_error(target_rotation, actual_rotation)
    return pose, {
        "label": label, "optimizer_success": bool(solved.success),
        "optimizer_status": int(solved.status), "optimizer_message": str(solved.message),
        "optimizer_nfev": int(solved.nfev),
        "position_error_m": float(np.linalg.norm(ep)),
        "rotation_error_rad": float(np.linalg.norm(er)),
        "rotation_error_deg": math.degrees(float(np.linalg.norm(er))),
        "arm_qpos_rad": dict(zip(RIGHT_ARM, [float(v) for v in solved.x])),
        "source_limit_pass": all(limits[name][0] - 1e-10 <= solved.x[i] <= limits[name][1] + 1e-10
                                  for i, name in enumerate(RIGHT_ARM)),
        "collision_checked": False,
        "iterations": history,
    }


def collision_aware_ik(base, model, seed_data, target_position: np.ndarray,
                       target_rotation: np.ndarray, aperture: float,
                       limits: dict[str, list[float]], label: str,
                       max_iterations: int = 250,
                       allow_pad_contact: bool = False) -> tuple[Any, dict[str, Any]]:
    data = mujoco.MjData(model)
    q_adrs = [qadr(model, name) for name in RIGHT_ARM]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM]
    lower = np.array([limits[name][0] for name in RIGHT_ARM], dtype=float)
    upper = np.array([limits[name][1] for name in RIGHT_ARM], dtype=float)
    x0 = np.array([seed_data.qpos[q] for q in q_adrs], dtype=float)
    widths = np.maximum(upper - lower, 1e-6)
    end_id = body_id(model, EE)
    history: list[dict[str, Any]] = []
    cached_q: np.ndarray | None = None
    cached: dict[str, Any] | None = None

    def evaluate(q: np.ndarray) -> dict[str, Any]:
        nonlocal cached_q, cached
        if cached_q is not None and np.array_equal(cached_q, q):
            return cached  # type: ignore[return-value]
        data.qpos[:] = seed_data.qpos
        data.qvel[:] = 0.0
        set_arm(model, data, dict(zip(RIGHT_ARM, q)))
        set_aperture(base, model, data, aperture)
        mujoco.mj_forward(model, data)
        actual_p = data.xpos[end_id].copy()
        actual_r = data.xmat[end_id].reshape(3, 3).copy()
        ep = target_position - actual_p
        er = rotation_error(target_rotation, actual_r)
        obstacles = obstacle_metrics(model, data)
        cached_q = np.array(q, copy=True)
        cached = {"data": data, "position_error": ep, "rotation_error": er,
                  "actual_position": actual_p, "actual_rotation": actual_r,
                  "obstacles": obstacles, "contacts": active_contacts(model, data)}
        history.append({
            "label": label, "evaluation": len(history),
            "arm_qpos_rad": dict(zip(RIGHT_ARM, [float(v) for v in q])),
            "position_error_m": float(np.linalg.norm(ep)),
            "rotation_error_rad": float(np.linalg.norm(er)),
            "clearance_m": {name: None if value is None else value["signed_distance_m"]
                             for name, value in obstacles.items() if isinstance(value, dict)},
        })
        return cached

    def pose_constraint(q: np.ndarray) -> np.ndarray:
        state = evaluate(q)
        return np.concatenate((state["position_error"] * 1000.0,
                               state["rotation_error"] * 100.0))

    def clearance_constraint(q: np.ndarray) -> np.ndarray:
        state = evaluate(q)
        values = state["obstacles"]
        critical = [values["torso_to_wrist"], values["arm_to_table"],
                    values["nonpad_robot_to_bottle"]]
        if not allow_pad_contact:
            critical.append(values["pads_to_bottle"])
        return np.array([
            ((float(row["signed_distance_m"]) if row else -1.0) - COLLISION_MARGIN_M) * 1000.0
            for row in critical
        ])

    seed_normalized = (x0 - 0.5 * (lower + upper)) / widths

    def objective(q: np.ndarray) -> float:
        normalized = (q - 0.5 * (lower + upper)) / widths
        # Prefer a source-limit-interior posture while staying close to the current IK branch.
        return float(np.dot(normalized, normalized) + 0.02 * np.dot((q - x0) / widths, (q - x0) / widths))

    solved = minimize(
        objective, x0, method="SLSQP", bounds=list(zip(lower, upper)),
        constraints=(
            {"type": "eq", "fun": pose_constraint},
            {"type": "ineq", "fun": clearance_constraint},
        ),
        options={"maxiter": max_iterations, "ftol": 1e-10, "disp": False},
    )
    final = evaluate(solved.x)
    final_data = mujoco.MjData(model)
    final_data.qpos[:] = seed_data.qpos
    final_data.qvel[:] = 0.0
    set_arm(model, final_data, dict(zip(RIGHT_ARM, solved.x)))
    set_aperture(base, model, final_data, aperture)
    mujoco.mj_forward(model, final_data)
    ep = final["position_error"]
    er = final["rotation_error"]
    obs = final["obstacles"]
    clearances = {name: None if value is None else float(value["signed_distance_m"])
                  for name, value in obs.items() if isinstance(value, dict)}
    required_keys = {"torso_to_wrist", "arm_to_table", "nonpad_robot_to_bottle"}
    if not allow_pad_contact:
        required_keys.add("pads_to_bottle")
    min_required = min((value for key, value in clearances.items()
                        if key in required_keys
                        and value is not None), default=-math.inf)
    exact = float(np.linalg.norm(ep)) <= POSITION_TOL_M and float(np.linalg.norm(er)) <= ROTATION_TOL_RAD
    collision_clear = min_required > 0.0
    q_result = {name: float(value) for name, value in zip(RIGHT_ARM, solved.x)}
    margins = {name: {"qpos_rad": q_result[name], "source_range_rad": limits[name],
                      "lower_margin_rad": q_result[name] - limits[name][0],
                      "upper_margin_rad": limits[name][1] - q_result[name]}
               for name in RIGHT_ARM}
    result = {
        "label": label, "optimizer_success": bool(solved.success),
        "optimizer_status": int(solved.status), "optimizer_message": str(solved.message),
        "optimizer_iterations": int(solved.nit), "optimizer_evaluations": len(history),
        "position_error_m": float(np.linalg.norm(ep)),
        "position_error_vector_world_m": np.asarray(ep, dtype=float).tolist(),
        "rotation_error_rad": float(np.linalg.norm(er)),
        "rotation_error_deg": math.degrees(float(np.linalg.norm(er))),
        "target_ee_position_world_m": target_position.tolist(),
        "actual_ee_position_world_m": final["actual_position"].tolist(),
        "target_ee_rotation_world": target_rotation.tolist(),
        "actual_ee_rotation_world": final["actual_rotation"].tolist(),
        "arm_qpos_rad": q_result, "source_limit_margins": margins,
        "collision_clearances_m": clearances,
        "collision_constraint_margin_m": COLLISION_MARGIN_M,
        "intended_pad_contact_allowed": allow_pad_contact,
        "pose_gate_pass": exact, "collision_gate_pass": collision_clear,
        "active_contacts": active_contacts(model, final_data),
        "source_limit_pass": all(limits[n][0] - 1e-10 <= q_result[n] <= limits[n][1] + 1e-10 for n in RIGHT_ARM),
    }
    return final_data, {"result": result, "iterations": history}


def source_limit_check(model, data, limits: dict[str, list[float]]) -> dict[str, Any]:
    rows = {}
    violations = []
    for name, interval in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        row = {"qpos_rad": q, "source_range_rad": interval,
               "lower_margin_rad": q - interval[0], "upper_margin_rad": interval[1] - q,
               "within_source_limits": interval[0] - 1e-10 <= q <= interval[1] + 1e-10}
        rows[name] = row
        if not row["within_source_limits"]:
            violations.append({"joint": name, **row})
    return {"all_within_source_limits": not violations, "rows": rows, "violations": violations}


def render(model, data, out: Path, view: str, target: np.ndarray) -> None:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    if view == "overview":
        camera.distance, camera.azimuth, camera.elevation = 2.5, 145.0, -14.0
        width, height = 1280, 800
    elif view == "front":
        camera.distance, camera.azimuth, camera.elevation = 1.8, 180.0, -8.0
        width, height = 1280, 800
    elif view == "side":
        camera.distance, camera.azimuth, camera.elevation = 1.8, 90.0, -8.0
        width, height = 1280, 800
    else:
        camera.distance, camera.azimuth, camera.elevation = 0.62, 145.0, -18.0
        width, height = 1280, 800
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render().copy()).save(out)
    renderer.close()


def candidate_run(base, feasibility, layout: dict[str, Any], output: Path,
                  limits: dict[str, list[float]]) -> dict[str, Any]:
    placed = layout_base(base, None, None, layout)
    model, data, build = placed["model"], placed["data"], placed["build"]
    name = layout["name"]
    source_limit = limits
    arm_mid = {joint: 0.5 * sum(source_limit[joint]) for joint in RIGHT_ARM}
    set_arm(model, data, arm_mid)
    aperture_targets = base.aperture_targets(1.0)
    data.qpos[qadr(model, DRIVER)] = float(aperture_targets["right_claw_joint_target_rad"])
    data.qpos[qadr(model, FOLLOWER)] = float(aperture_targets["R_hand_wide1_joint_target_rad"])
    mujoco.mj_forward(model, data)
    initial_metrics = base_pair_metrics(base, model, data)
    initial_limits = source_limit_check(model, data, source_limit)
    result: dict[str, Any] = {
        "layout": {k: v for k, v in layout.items()},
        "base_position_world_m": placed["base_position_world_m"],
        "base_yaw_deg": placed["base_yaw_deg"],
        "desired_shoulder_pivot_world_m": placed["desired_shoulder_pivot_world_m"],
        "actual_shoulder_pivot_world_m": placed["actual_shoulder_pivot_world_m"],
        "heading_error_to_table_center_deg": placed["heading_error_to_table_center_deg"],
        "initial_scene": initial_metrics,
        "source_limit_check_at_initialization": initial_limits,
        "physics_steps": 0,
        "static_geometry_pass": False,
        "physical_contact": "NOT RUN",
    }
    safe_initial = (initial_metrics["maximum_self_penetration_m"] <= 1e-8
                    and initial_metrics["maximum_table_penetration_m"] <= 1e-8
                    and not initial_metrics["robot_self_contacts"]
                    and not initial_metrics["robot_table_contacts"]
                    and initial_limits["all_within_source_limits"])
    if not safe_initial:
        result["first_failure"] = "INITIAL_SCENE_CONTACT_OR_LIMIT"
        return result

    channel = feasibility.local_channel(base, model, feasibility.REFERENCE_APERTURE)
    approach = np.array([0.0, -1.0, 0.0])
    jaw = np.array([1.0, 0.0, 0.0])
    rotation = feasibility.candidate_rotation(channel, approach, jaw)
    tcp = np.array([0.300, 0.000, GRASP_CENTER_Z])
    pregrasp_tcp = tcp - STANDOFF_M * approach
    channel_open = feasibility.local_channel(base, model, 1.0)
    channel_reference = feasibility.local_channel(base, model, feasibility.REFERENCE_APERTURE)
    channels = {1.0: channel_open, feasibility.REFERENCE_APERTURE: channel_reference}
    pregrasp_target = feasibility.target_ee_position(base, model, rotation, pregrasp_tcp, 1.0, channels, jaw)

    pose_seed, seed_result = pose_only_seed(model, data, pregrasp_target, rotation,
                                            1.0, source_limit, f"{name}:pregrasp:pose-seed")
    write_jsonl(output / f"{name}_pose_seed_iterations.jsonl", seed_result.pop("iterations"))
    result["pregrasp_pose_seed"] = seed_result
    pre_data, pre_ik = collision_aware_ik(base, model, pose_seed, pregrasp_target, rotation,
                                          1.0, source_limit, f"{name}:pregrasp")
    write_jsonl(output / f"{name}_ik_iterations.jsonl", pre_ik["iterations"])
    result["pregrasp"] = pre_ik["result"]
    result["pregrasp_tcp_world_m"] = pregrasp_tcp.tolist()
    result["grasp_tcp_world_m"] = tcp.tolist()
    if not (pre_ik["result"]["pose_gate_pass"] and pre_ik["result"]["collision_gate_pass"]
            and pre_ik["result"]["source_limit_pass"]):
        result["first_failure"] = "PREGRASP_COLLISION_AWARE_IK"
        for view, suffix in (("overview", "overview"), ("front", "front"), ("side", "side"), ("closeup", "wrist_contact_closeup")):
            render(model, pre_data, output / f"{name}_{suffix}.png", view,
                   np.array([0.30, 0.10, 1.03]) if suffix != "wrist_contact_closeup" else np.array([0.30, 0.0, GRASP_CENTER_Z]))
        write_jsonl(output / f"{name}_geometry_trace.jsonl", [{"phase": "PREGRASP", **pre_ik["result"]}])
        return result

    path_rows = []
    seed = pre_data
    for index, progress in enumerate(np.linspace(0.0, 1.0, APPROACH_SAMPLES)):
        tcp_at = pregrasp_tcp + progress * (tcp - pregrasp_tcp)
        target = feasibility.target_ee_position(base, model, rotation, tcp_at, 1.0, channels, jaw)
        pose, ik = collision_aware_ik(base, model, seed, target, rotation, 1.0,
                                      source_limit, f"{name}:approach:{index:02d}", max_iterations=160)
        full_metrics = base_pair_metrics(base, model, pose)
        row = {
            "sample": index, "progress_0_to_1": float(progress), "tcp_world_m": tcp_at.tolist(),
            "ik": ik["result"], "scene": full_metrics,
            "source_limits": source_limit_check(model, pose, source_limit),
            "passed": bool(ik["result"]["pose_gate_pass"] and ik["result"]["collision_gate_pass"]
                           and ik["result"]["source_limit_pass"]
                           and full_metrics["maximum_self_penetration_m"] <= 1e-8
                           and full_metrics["maximum_table_penetration_m"] <= 1e-8
                           and not full_metrics["robot_self_contacts"]
                           and not full_metrics["robot_table_contacts"]),
        }
        path_rows.append(row)
        seed = pose
        if not row["passed"]:
            break
    result["approach_rows"] = path_rows
    result["approach_pass"] = len(path_rows) == APPROACH_SAMPLES and all(row["passed"] for row in path_rows)
    if not result["approach_pass"]:
        result["first_failure"] = "CONTINUOUS_APPROACH_IK_OR_COLLISION"
        write_jsonl(output / f"{name}_geometry_trace.jsonl", path_rows)
        last_pose = seed
        for view, suffix in (("overview", "overview"), ("front", "front"), ("side", "side"), ("closeup", "wrist_contact_closeup")):
            render(model, last_pose, output / f"{name}_{suffix}.png", view,
                   np.array([0.30, 0.10, 1.03]) if suffix != "wrist_contact_closeup" else np.array([0.30, 0.0, GRASP_CENTER_Z]))
        return result

    # At the open-hand grasp pose, inspect every legal aperture without stepping physics.
    closure_rows = []
    closure_seed = seed
    aperture_values = np.linspace(1.0, 0.0, CLOSURE_SAMPLES)
    for index, aperture in enumerate(aperture_values):
        sample = mujoco.MjData(model)
        sample.qpos[:] = closure_seed.qpos
        sample.qvel[:] = 0.0
        targets = base.aperture_targets(float(aperture))
        sample.qpos[qadr(model, DRIVER)] = float(targets["right_claw_joint_target_rad"])
        sample.qpos[qadr(model, FOLLOWER)] = float(targets["R_hand_wide1_joint_target_rad"])
        mujoco.mj_forward(model, sample)
        full_metrics = base_pair_metrics(base, model, sample)
        obstacles = obstacle_metrics(model, sample)
        nrow = minimum_pair(model, sample, geom_ids(model, {NARROW}), geom_ids(model, {BOTTLE}))
        wrow = minimum_pair(model, sample, geom_ids(model, {WIDE}), geom_ids(model, {BOTTLE}))
        nonpad = obstacles["nonpad_robot_to_bottle"]
        limits_now = source_limit_check(model, sample, source_limit)
        intended = [row for row in active_contacts(model, sample)
                    if BOTTLE in {row["body1"], row["body2"]}
                    and ({row["body1"], row["body2"]} & {NARROW, WIDE})]
        unexpected = [row for row in active_contacts(model, sample)
                      if BOTTLE in {row["body1"], row["body2"]} and row not in intended]
        closure_rows.append({
            "sample": index, "aperture_ratio": float(aperture),
            "narrow_to_bottle_body": nrow, "wide_to_bottle_body": wrow,
            "minimum_nonpad_to_bottle": nonpad,
            "minimum_robot_to_table": full_metrics["minimum_robot_table"],
            "robot_self_contacts": full_metrics["robot_self_contacts"],
            "robot_table_contacts": full_metrics["robot_table_contacts"],
            "intended_pad_contacts": intended, "unexpected_bottle_contacts": unexpected,
            "source_limits": limits_now,
        })
    narrow_dist = [row["narrow_to_bottle_body"]["signed_distance_m"] for row in closure_rows]
    wide_dist = [row["wide_to_bottle_body"]["signed_distance_m"] for row in closure_rows]
    simultaneous = [row for row in closure_rows if row["narrow_to_bottle_body"]["signed_distance_m"] <= 0.0
                    and row["wide_to_bottle_body"]["signed_distance_m"] <= 0.0]
    contact_row = simultaneous[0] if simultaneous else None
    nonpad_min = min((row["minimum_nonpad_to_bottle"]["signed_distance_m"]
                      for row in closure_rows if row["minimum_nonpad_to_bottle"]), default=math.inf)
    table_min = min((row["minimum_robot_to_table"]["signed_distance_m"]
                     for row in closure_rows if row["minimum_robot_to_table"]), default=math.inf)
    closure_limits = all(row["source_limits"]["all_within_source_limits"] for row in closure_rows)
    closure_self_clear = all(not row["robot_self_contacts"] for row in closure_rows)
    contact_geometry_pass = bool(
        contact_row is not None
        and max(-contact_row["narrow_to_bottle_body"]["signed_distance_m"],
                -contact_row["wide_to_bottle_body"]["signed_distance_m"]) <= MAX_INTENDED_PENETRATION_M
        and nonpad_min > 0.0 and table_min > 0.0 and closure_limits
        and closure_self_clear
        and not any(row["unexpected_bottle_contacts"] for row in closure_rows)
    )
    result["closure_summary"] = {
        "min_narrow_clearance_m": min(narrow_dist), "min_wide_clearance_m": min(wide_dist),
        "contact_first_aperture_ratio": None if contact_row is None else contact_row["aperture_ratio"],
        "first_bilateral_geom_distances_m": None if contact_row is None else {
            NARROW: contact_row["narrow_to_bottle_body"]["signed_distance_m"],
            WIDE: contact_row["wide_to_bottle_body"]["signed_distance_m"]},
        "minimum_nonpad_bottle_clearance_m": nonpad_min,
        "minimum_robot_table_clearance_m": table_min,
        "all_source_limits_pass": closure_limits,
        "all_robot_self_contacts_clear": closure_self_clear,
        "contact_geometry_pass": contact_geometry_pass,
    }
    result["closure_pass"] = contact_geometry_pass
    result["static_geometry_pass"] = result["approach_pass"] and contact_geometry_pass
    result["physical_contact"] = "NOT RUN: static closure only"
    write_jsonl(output / f"{name}_geometry_trace.jsonl", path_rows + closure_rows)
    for view, suffix in (("overview", "overview"), ("front", "front"), ("side", "side"), ("closeup", "wrist_contact_closeup")):
        render(model, closure_seed, output / f"{name}_{suffix}.png", view,
               np.array([0.30, 0.10, 1.03]) if suffix != "wrist_contact_closeup" else np.array([0.30, 0.0, GRASP_CENTER_Z]))
    result["candidate_grasp_arm_qpos_rad"] = {
        name: float(closure_seed.qpos[qadr(model, name)]) for name in RIGHT_ARM
    }
    contact_pose_data = mujoco.MjData(model)
    contact_pose_data.qpos[:] = closure_seed.qpos
    contact_pose_data.qvel[:] = 0.0
    if contact_row is not None:
        contact_targets = base.aperture_targets(float(contact_row["aperture_ratio"]))
        contact_pose_data.qpos[qadr(model, DRIVER)] = float(contact_targets["right_claw_joint_target_rad"])
        contact_pose_data.qpos[qadr(model, FOLLOWER)] = float(contact_targets["R_hand_wide1_joint_target_rad"])
    mujoco.mj_forward(model, contact_pose_data)
    result["contact_normal_geometry"] = feasibility.contact_normal_metrics(
        model, contact_pose_data,
        {NARROW: geom_ids(model, {NARROW})[0], WIDE: geom_ids(model, {WIDE})[0]},
        geom_ids(model, {BOTTLE})[0],
    )
    if result["static_geometry_pass"]:
        result["_model"] = model
        result["_pregrasp_q"] = pre_ik["result"]["arm_qpos_rad"]
        result["_grasp_q"] = result["candidate_grasp_arm_qpos_rad"]
        result["_rotation"] = rotation.tolist()
        result["_grasp_data"] = contact_pose_data
        result["_build"] = build
        result["_approach_q"] = path_rows[-1]["ik"]["arm_qpos_rad"]
        result["_pregrasp_q"] = pre_ik["result"]["arm_qpos_rad"]
        result["_contact_aperture"] = None if contact_row is None else float(contact_row["aperture_ratio"])
    return result


def contact_force_rows(model, data, object_body_name: str = BOTTLE) -> list[dict[str, Any]]:
    object_id = body_id(model, object_body_name)
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if object_id not in {body1, body2}:
            continue
        local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        local_world_on_geom2 = frame.T @ local[:3]
        on_object = local_world_on_geom2 if body2 == object_id else -local_world_on_geom2
        normal_world = frame[0].copy() if body2 == object_id else -frame[0].copy()
        normal_component = float(np.dot(on_object, normal_world))
        tangent_world = on_object - normal_component * normal_world
        other = body1 if body2 == object_id else body2
        jac_object = np.zeros((3, model.nv), dtype=float)
        jac_other = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jac(model, data, jac_object, None, contact.pos, object_id)
        if other != 0:
            mujoco.mj_jac(model, data, jac_other, None, contact.pos, other)
        relative = (jac_object - jac_other) @ data.qvel
        relative_tangent = relative - float(np.dot(relative, normal_world)) * normal_world
        other_name = body_name(model, other)
        family = "narrow" if "narrow" in other_name.lower() else "wide" if "wide" in other_name.lower() else "table" if other_name == TABLE else "other"
        rows.append({
            "geom1": geom_name(model, geom1), "body1": body_name(model, body1),
            "geom2": geom_name(model, geom2), "body2": body_name(model, body2),
            "family": family, "distance_m": float(contact.dist),
            "contact_position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_frame_rows_world": frame.tolist(),
            "contact_force_local_N_Nm": local.tolist(),
            "force_on_object_world_N": on_object.tolist(),
            "normal_force_on_object_N": normal_component,
            "tangential_force_on_object_world_N": tangent_world.tolist(),
            "tangential_force_magnitude_N": float(np.linalg.norm(tangent_world)),
            "relative_tangential_velocity_world_m_s": relative_tangent.tolist(),
            "relative_tangential_speed_m_s": float(np.linalg.norm(relative_tangent)),
            "world_force_sign_rule": "mj_contactForce local wrench mapped by contact.frame; sign reversed when object is geom1",
        })
    return rows


def static_lift_corridor(base, feasibility, model, grasp_data, target_rotation: np.ndarray,
                         contact_aperture: float, limits: dict[str, list[float]],
                         output: Path, candidate_name: str) -> tuple[bool, list[dict[str, Any]]]:
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    initial = base.initialize(model)
    seed = mujoco.MjData(model)
    seed.qpos[:] = grasp_data.qpos
    seed.qvel[:] = 0.0
    tcp0 = np.array([0.300, 0.0, GRASP_CENTER_Z], dtype=float)
    channels = {contact_aperture: feasibility.local_channel(base, model, contact_aperture)}
    rows = []
    heights = (0.0, 0.001, 0.005, 0.010, 0.015, 0.020,
               0.025, 0.030, 0.035, 0.040, 0.045, 0.050)
    for index, height in enumerate(heights):
        seed.qpos[bottle_adr + 2] = initial.qpos[bottle_adr + 2] + float(height)
        target_tcp = tcp0 + np.array([0.0, 0.0, height])
        target_ee = feasibility.target_ee_position(
            base, model, target_rotation, target_tcp, contact_aperture, channels, np.array([1.0, 0.0, 0.0]))
        pose, ik = collision_aware_ik(
            base, model, seed, target_ee, target_rotation, contact_aperture, limits,
            f"{candidate_name}:static-lift:{index:02d}", max_iterations=180,
            allow_pad_contact=True)
        metrics = base_pair_metrics(base, model, pose)
        passed = bool(ik["result"]["pose_gate_pass"] and ik["result"]["collision_gate_pass"]
                      and ik["result"]["source_limit_pass"]
                      and metrics["maximum_self_penetration_m"] <= 1e-8
                      and metrics["maximum_table_penetration_m"] <= 1e-8
                      and not metrics["robot_self_contacts"] and not metrics["robot_table_contacts"])
        rows.append({"sample": index, "height_m": float(height), "ik": ik["result"],
                     "scene": metrics, "passed": passed,
                     "object_pose_for_collision_preview_only": True})
        if not passed:
            break
        seed = pose
    write_jsonl(output / f"{candidate_name}_static_lift_corridor.jsonl", rows)
    return len(rows) == 11 and all(row["passed"] for row in rows), rows


def run_dynamics(base, feasibility, candidate: dict[str, Any], limits: dict[str, list[float]],
                 output: Path) -> dict[str, Any]:
    model = candidate["_model"]
    build = candidate["_build"]
    controller = base.derive_jaw_controller(model)
    controller["source_effort_limits_nm"] = build["source_effort_limits_nm"]
    controller["arm_actuators"] = build["arm_actuator_names"]
    base.configure_arm(model, controller)
    base.apply_controller_limits(model, controller, arm=True)

    corridor_pass, corridor_rows = static_lift_corridor(
        base, feasibility, model, candidate["_grasp_data"], np.asarray(candidate["_rotation"]),
        float(candidate["_contact_aperture"]), limits, output, candidate["layout"]["name"])
    if not corridor_pass:
        return {"status": "FAIL", "first_failed_gate": "STATIC_50MM_LIFT_CORRIDOR",
                "physics_steps": 0, "static_lift_corridor": corridor_rows,
                "bottle_qpos_writes_during_rollout": 0}

    waypoints = {int(round(row["height_m"] * 1000)): row["ik"]["arm_qpos_rad"]
                 for row in corridor_rows}
    lift_targets = {mm: waypoints[mm] for mm in (1, 5, 30, 50)}
    data = base.initialize(model)
    set_arm(model, data, candidate["_pregrasp_q"])
    mujoco.mj_forward(model, data)
    initial_metrics = base_pair_metrics(base, model, data)
    initial_limits = source_limit_check(model, data, limits)
    if (initial_metrics["maximum_self_penetration_m"] > 1e-8
            or initial_metrics["maximum_table_penetration_m"] > 1e-8
            or initial_metrics["robot_self_contacts"]
            or initial_metrics["robot_table_contacts"]
            or not initial_limits["all_within_source_limits"]):
        return {"status": "FAIL", "first_failed_gate": "DYNAMIC_INITIALIZATION",
                "physics_steps": 0, "initial_scene": initial_metrics,
                "initial_limits": initial_limits, "bottle_qpos_writes_during_rollout": 0}

    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = base.camera_for("bottle_hold")
    frames: list[np.ndarray] = []
    trace_path = output / "native_omnipicker_rollout_trace.jsonl"
    initial_bottle_z = float(data.xpos[body_id(model, BOTTLE)][2])
    initial_bottle_pos = data.xpos[body_id(model, BOTTLE)].copy()
    phase_stats = {name: {"samples": 0, "narrow_contact_samples": 0, "wide_contact_samples": 0,
                          "mean_narrow_normal_force_N": 0.0, "mean_wide_normal_force_N": 0.0,
                          "mean_narrow_vertical_support_N": 0.0,
                          "mean_wide_vertical_support_N": 0.0,
                          "mean_total_hand_vertical_support_N": 0.0,
                          "mean_table_support_N": 0.0} for name in
                   ("APPROACH", "CLOSE", "HOLD", "LIFT_1MM", "LIFT_5MM", "LIFT_30MM", "LIFT_50MM", "AIR_HOLD")}
    sums = {key: {"narrow": 0.0, "wide": 0.0, "narrow_vertical": 0.0,
                  "wide_vertical": 0.0, "hand_vertical": 0.0, "table": 0.0}
            for key in phase_stats}
    step = 0
    max_lift_m = 0.0
    max_penetration_m = 0.0
    max_relation_error = 0.0
    max_arm_effort: dict[str, float] = {n: 0.0 for n in RIGHT_ARM}
    max_jaw_effort: dict[str, float] = {DRIVER: 0.0, FOLLOWER: 0.0}
    first_contact_saved = False
    first_failure = None
    hold_contact_families = {"narrow": 0, "wide": 0, "samples": 0}
    lift_results = {}
    max_slip_m = 0.0
    bilateral_contact_relative_reference = None
    final_table_contacts = []
    key_paths = {}

    def frame(name: str) -> None:
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
        frames.append(image)
        path = output / f"{name}.png"
        Image.fromarray(image).save(path)
        key_paths[name] = str(path)

    frame("native_pregrasp")

    def command_for(name: str, phase_time: float, duration: float,
                    start_q: dict[str, float], end_q: dict[str, float],
                    aperture_start: float, aperture_end: float):
        ref = {}
        for joint in RIGHT_ARM:
            p, v, a = base.minimum_jerk(phase_time, duration, start_q[joint], end_q[joint])
            ref[joint] = (p, v, a)
        ap, av, aa = base.minimum_jerk(phase_time, duration, aperture_start, aperture_end)
        return ref, ap, av, aa

    phases = [("APPROACH", 1.5, candidate["_pregrasp_q"], candidate["_approach_q"], 1.0, 1.0),
              ("CLOSE", 2.5, candidate["_approach_q"], candidate["_approach_q"], 1.0, 0.0),
              ("HOLD", 1.0, candidate["_approach_q"], candidate["_approach_q"], 0.0, 0.0)]
    trace_rows = []

    def execute_phase(phase: str, duration: float, start_q: dict[str, float], end_q: dict[str, float],
                      aperture_start: float, aperture_end: float) -> None:
        nonlocal step, max_lift_m, max_penetration_m, max_relation_error
        nonlocal first_contact_saved, first_failure, bilateral_contact_relative_reference
        count = max(1, round(duration / float(model.opt.timestep)))
        for local_step in range(count):
            t = local_step * float(model.opt.timestep)
            arm_targets, aperture, aperture_v, aperture_a = command_for(
                phase, t, duration, start_q, end_q, aperture_start, aperture_end)
            mujoco.mj_forward(model, data)
            jaw_control = base.jaw_control(model, data, aperture, aperture_v, aperture_a, controller)
            arm_control = base.arm_control(model, data, arm_targets, controller)
            mujoco.mj_step(model, data)
            step += 1
            contacts = contact_force_rows(model, data)
            bottle_contacts = [row for row in contacts if BOTTLE in {row["body1"], row["body2"]}]
            counts = phase_stats[phase]
            counts["samples"] += 1
            families = {row["family"] for row in bottle_contacts}
            for family in ("narrow", "wide"):
                family_rows = [row for row in bottle_contacts if row["family"] == family]
                if family_rows:
                    counts[f"{family}_contact_samples"] += 1
                sums[phase][family] += sum(max(0.0, row["normal_force_on_object_N"]) for row in family_rows)
                sums[phase][f"{family}_vertical"] += sum(row["force_on_object_world_N"][2] for row in family_rows)
            hand_vertical = sum(row["force_on_object_world_N"][2] for row in bottle_contacts
                                if row["family"] in {"narrow", "wide"})
            table_force = sum(row["force_on_object_world_N"][2] for row in bottle_contacts
                              if row["family"] == "table")
            sums[phase]["hand_vertical"] += hand_vertical
            sums[phase]["table"] += table_force
            if phase == "HOLD":
                hold_contact_families["samples"] += 1
                for family in ("narrow", "wide"):
                    hold_contact_families[family] += int(family in families)
            bottle_id = body_id(model, BOTTLE)
            lift = float(data.xpos[bottle_id][2] - initial_bottle_z)
            max_lift_m = max(max_lift_m, lift)
            source_state = source_limit_check(model, data, limits)
            relation = float(data.qpos[qadr(model, DRIVER)] + data.qpos[qadr(model, FOLLOWER)])
            max_relation_error = max(max_relation_error, abs(relation))
            all_contacts = active_contacts(model, data)
            max_penetration_m = max(max_penetration_m,
                                    max((row["penetration_m"] for row in all_contacts), default=0.0))
            bottle_table = [row for row in all_contacts if TABLE in {row["body1"], row["body2"]}
                            and BOTTLE in {row["body1"], row["body2"]}]
            robot_table = [row for row in all_contacts if TABLE in {row["body1"], row["body2"]}
                           and BOTTLE not in {row["body1"], row["body2"]}]
            robot_self = [row for row in all_contacts if row["body1"] not in {"world", TABLE, BOTTLE}
                          and row["body2"] not in {"world", TABLE, BOTTLE}]
            unexpected_bottle = [row for row in all_contacts if BOTTLE in {row["body1"], row["body2"]}
                                 and TABLE not in {row["body1"], row["body2"]}
                                 and ({row["body1"], row["body2"]} & {NARROW, WIDE}) == set()]
            object_pos = data.xpos[bottle_id].copy()
            hand_pos = data.xpos[body_id(model, EE)].copy()
            if {"narrow", "wide"}.issubset(families) and bilateral_contact_relative_reference is None:
                bilateral_contact_relative_reference = object_pos - hand_pos
            rel_slip = (None if bilateral_contact_relative_reference is None else
                        float(np.linalg.norm((object_pos - hand_pos) - bilateral_contact_relative_reference)))
            if rel_slip is not None:
                max_slip_m = max(max_slip_m, rel_slip)
            arm_efforts = {name: abs(float(arm_control[name]["applied_torque_nm"])) for name in RIGHT_ARM}
            jaw_efforts = {name: abs(float(jaw_control[name]["applied_torque_nm"])) for name in (DRIVER, FOLLOWER)}
            for name, value in arm_efforts.items():
                max_arm_effort[name] = max(max_arm_effort[name], value)
            for name, value in jaw_efforts.items():
                max_jaw_effort[name] = max(max_jaw_effort[name], value)
            row = {
                "step": step, "time_s": float(data.time), "phase": phase,
                "aperture_target_ratio": aperture, "source_limit_state": source_state,
                "right_arm_control": arm_control, "right_jaw_control": jaw_control,
                "jaw_relation_error_rad": relation, "bottle_position_world_m": object_pos.tolist(),
                "bottle_velocity_world_m_s": data.qvel[
                    int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")])+3:
                    int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")])+6
                ].astype(float).tolist(),
                "bottle_lift_from_initial_m": lift, "relative_bottle_hand_translation_change_m": rel_slip,
                "bottle_contacts_with_world_force_N": bottle_contacts,
                "bottle_table_contact_count": len(bottle_table), "robot_table_contacts": robot_table,
                "robot_self_contacts": robot_self, "unexpected_bottle_contacts": unexpected_bottle,
                "qpos_writes_during_active_rollout": 0, "bottle_qpos_writes_during_active_rollout": 0,
                "weld_or_mocap_attachment": False,
            }
            trace_rows.append(row)
            if not first_contact_saved and any(r["family"] in {"narrow", "wide"} for r in bottle_contacts):
                first_contact_saved = True
                frame("native_first_contact")
            if step % 25 == 0:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            if source_state["violations"]:
                first_failure = {"gate": "SOURCE_JOINT_LIMIT", "step": step,
                                 "violations": source_state["violations"]}
            elif not np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc))).all():
                first_failure = {"gate": "FINITE_STATE", "step": step}
            elif abs(relation) > base.APERTURE_RELATION_DIAGNOSTIC_RAD:
                first_failure = {"gate": "PAIRED_JAW_TRACKING", "step": step,
                                 "relation_error_rad": relation}
            elif max_penetration_m > 0.008:
                first_failure = {"gate": "PENETRATION", "step": step,
                                 "maximum_penetration_m": max_penetration_m}
            elif robot_table:
                first_failure = {"gate": "ROBOT_TABLE_COLLISION", "step": step,
                                 "contacts": robot_table}
            elif robot_self:
                first_failure = {"gate": "ROBOT_SELF_COLLISION", "step": step,
                                 "contacts": robot_self}
            elif unexpected_bottle:
                first_failure = {"gate": "NONPAD_BOTTLE_COLLISION", "step": step,
                                 "contacts": unexpected_bottle}
            if first_failure:
                break

    current_q = dict(candidate["_pregrasp_q"])
    for phase, duration, start_q, end_q, aperture_start, aperture_end in phases:
        execute_phase(phase, duration, start_q, end_q, aperture_start, aperture_end)
        if phase == "HOLD":
            frame("native_closed_hold")
        if first_failure:
            break
    if first_failure is None:
        current_q = dict(candidate["_approach_q"])
    hold_fraction = {family: hold_contact_families[family] / max(hold_contact_families["samples"], 1)
                     for family in ("narrow", "wide")}
    bilateral_hold_pass = min(hold_fraction.values()) >= 0.80
    if first_failure is None and not bilateral_hold_pass:
        first_failure = {"gate": "BILATERAL_CONTACT_HOLD",
                         "contact_fraction_by_family": hold_fraction}
    if first_failure is None:
        for millimeters, duration in ((1, 0.50), (5, 0.50), (30, 0.75), (50, 1.00)):
            phase = f"LIFT_{millimeters}MM"
            target = lift_targets[millimeters]
            execute_phase(phase, duration, current_q, target, 0.0, 0.0)
            current_q = target
            bottle_z = float(data.xpos[body_id(model, BOTTLE)][2])
            lift_results[f"{millimeters}mm"] = {"target_m": millimeters / 1000.0,
                                                  "observed_from_initial_m": bottle_z - initial_bottle_z,
                                                  "table_contact_count": sum(
                                                      1 for row in active_contacts(model, data)
                                                      if TABLE in {row["body1"], row["body2"]}
                                                      and BOTTLE in {row["body1"], row["body2"]})}
            if millimeters < 50 and (bottle_z - initial_bottle_z) < 0.8 * millimeters / 1000.0:
                first_failure = {"gate": f"PHYSICAL_LIFT_{millimeters}MM",
                                 "observed_lift_m": bottle_z - initial_bottle_z}
                break
            if first_failure:
                break
            if millimeters == 50:
                frame("native_maximum_lift")
        if first_failure is None and lift_results.get("50mm", {}).get("observed_from_initial_m", 0.0) >= 0.050:
            execute_phase("AIR_HOLD", 1.0, current_q, current_q, 0.0, 0.0)
            final_table_contacts = [row for row in active_contacts(model, data)
                                    if TABLE in {row["body1"], row["body2"]}
                                    and BOTTLE in {row["body1"], row["body2"]}]
            frame("native_suspended_hold")
        elif first_failure is None:
            first_failure = {"gate": "50MM_PHYSICAL_LIFT", "observed_lift_m": max_lift_m}

    for phase, stats in phase_stats.items():
        n = max(stats["samples"], 1)
        for family in ("narrow", "wide"):
            stats[f"mean_{family}_normal_force_N"] = sums[phase][family] / n
            stats[f"mean_{family}_vertical_support_N"] = sums[phase][f"{family}_vertical"] / n
        stats["mean_total_hand_vertical_support_N"] = sums[phase]["hand_vertical"] / n
        stats["mean_table_support_N"] = sums[phase]["table"] / n
    write_jsonl(trace_path, trace_rows)
    renderer.close()
    video_path = output / "native_omnipicker_approach_close_lift.mp4"
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
    final_bottle_z = float(data.xpos[body_id(model, BOTTLE)][2])
    no_table_support = not final_table_contacts
    stable_velocity = float(np.linalg.norm(data.qvel[
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")])+3:
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")])+6
    ]))
    final_air_lift = final_bottle_z - initial_bottle_z
    success = (first_failure is None and max_lift_m >= 0.050 and final_air_lift >= 0.050
               and no_table_support and bilateral_hold_pass and stable_velocity <= 0.02)
    return {
        "status": "PASS" if success else "FAIL",
        "first_failed_gate": first_failure,
        "physics_steps": step, "simulation_time_s": float(data.time),
        "initial_bottle_body_z_m": initial_bottle_z,
        "final_bottle_body_z_m": final_bottle_z,
        "maximum_bottle_lift_m": max_lift_m,
        "lift_stage_results": lift_results,
        "hold_contact_fraction_by_family": hold_fraction,
        "bilateral_hold_pass": bilateral_hold_pass,
        "final_air_hold_lift_m": final_air_lift,
        "final_air_hold_bottle_velocity_m_s": stable_velocity,
        "final_table_contact_count": len(final_table_contacts),
        "contact_force_sign_convention": "geom2 receives contact.frame.T @ mj_contactForce[:3]; geom1 receives its negative; no data-dependent sign flip",
        "phase_force_summary": phase_stats,
        "maximum_penetration_m": max_penetration_m,
        "maximum_shared_jaw_relation_error_rad": max_relation_error,
        "maximum_arm_effort_abs_Nm": max_arm_effort,
        "maximum_jaw_effort_abs_Nm": max_jaw_effort,
        "maximum_relative_bottle_hand_translation_change_m": max_slip_m,
        "source_joint_limit_violations": 0 if first_failure is None or first_failure.get("gate") != "SOURCE_JOINT_LIMIT" else len(first_failure.get("violations", [])),
        "bottle_qpos_writes_during_rollout": 0,
        "bottle_weld_or_mocap_attachment": False,
        "left_hand_assistance": False,
        "trace_path": str(trace_path), "video_path": str(video_path),
        "keyframes": key_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, required=True)
    parser.add_argument("--canonical-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    urdf = args.x2_root / URDF_REL
    if not urdf.is_file() or not args.canonical_helper.is_file():
        raise FileNotFoundError("Pinned X2 source or canonical scene helper is missing")
    vendor_sha = subprocess.check_output(["git", "-C", str(args.x2_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.x2_root), "status", "--porcelain"], text=True).strip()
    if vendor_sha != X2_PIN or dirty or sha256(urdf) != URDF_SHA256 or sha256(args.canonical_helper) != CANONICAL_SHA256:
        raise RuntimeError(f"Pinned input mismatch: {vendor_sha=}, {bool(dirty)=}")
    if mujoco.__version__ != "3.3.6" or mujoco.mj_versionString() != "3.3.6":
        raise RuntimeError("This experiment requires MuJoCo Python/native 3.3.6")

    base = load_module("issue46_native_baseline", REPO / "scripts/research/issue46_omnipicker_1dof_m0.py")
    base.X2_ROOT = args.x2_root
    base.STATIC_ROOT = args.canonical_helper.parent
    feasibility = load_module("issue46_previous_feasibility", REPO / "scripts/research/issue46_native_omnipicker_feasibility.py")
    limits_model, limits_build = base.build_model("bottle_hold")
    limits = source_limits(urdf, limits_model)

    candidates = []
    for layout in DEFAULT_LAYOUTS:
        candidate = candidate_run(base, feasibility, dict(layout), args.output, limits)
        candidates.append(candidate)
        if candidate.get("static_geometry_pass"):
            break
    static_candidate = next((candidate for candidate in candidates
                             if candidate.get("static_geometry_pass")), None)
    dynamics = None
    if static_candidate is not None:
        dynamics = run_dynamics(base, feasibility, static_candidate, limits, args.output)
        static_candidate["dynamics"] = dynamics
    for candidate in candidates:
        for private_key in [key for key in candidate if key.startswith("_")]:
            candidate.pop(private_key)
    if static_candidate is None:
        task_status = "FAIL_STATIC_GEOMETRY"
    elif dynamics and dynamics.get("status") == "PASS":
        task_status = "PASS_PHYSICAL_50MM_LIFT_AND_1S_UNSUPPORTED_HOLD"
    else:
        failed_gate = "DYNAMICS"
        if dynamics:
            failed_gate = str(dynamics.get("first_failed_gate") or "PHYSICAL_ACCEPTANCE")
        task_status = f"FAIL_{failed_gate}"
    result = {
        "task": "Issue #46 native OmniPicker source-limited workcell recovery",
        "status": task_status,
        "attempt_id": "94e0b6ec-94bf-4fd1-a1ed-a3c0a779e241",
        "source": {"repository": "https://github.com/AgibotTech/agibot_x2_urdf",
                   "commit": vendor_sha, "license": "Mulan PSL v2", "urdf": str(URDF_REL),
                   "urdf_sha256": sha256(urdf), "canonical_helper_sha256": sha256(args.canonical_helper),
                   "right_narrow_mesh_sha256": "0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0",
                   "right_wide_mesh_sha256": "a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb"},
        "runtime": {"python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(), "timestep_s": 0.002,
                    "integrator": "Euler", "source_geometry_changed": False,
                    "robotsim_head": subprocess.check_output(
                        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                    "runner_sha256": sha256(Path(__file__).resolve()),
                    "source_joint_limits_changed": False,
                    "historical_simulation_only_exclusions": list(base.STATIC_COLLISION_EXCLUSIONS),
                    "additional_collision_exclusions": [],
                    "control_classification": "SIMULATION_ONLY_M0 paired-jaw actuator emulation; native source collision geometry and source limits retained."},
        "canonical_scene": {"bottle_position_world_m": [0.300, 0.000, 0.9175],
                            "bottle_dimensions_m": [0.070, 0.2445], "bottle_mass_kg": 0.570,
                            "bottle_friction": [1.4, 0.02, 0.001],
                            "table_center_xy_m": [0.300, -0.100], "table_half_extents_xy_m": [0.200, 0.200],
                            "table_top_z_m": 0.800},
        "candidate_count": len(candidates), "candidates": candidates,
        "source_limited_static_grasp_pass": any(c.get("static_geometry_pass") for c in candidates),
        "physical_bilateral_contact": (
            "NOT RUN" if dynamics is None else
            "PASS" if dynamics.get("bilateral_hold_pass") else
            "FAIL" if dynamics.get("first_failed_gate") == "BILATERAL_CONTACT_HOLD" else "NOT REACHED"),
        "physical_50mm_lift_and_1s_unsupported_hold": (
            "NOT RUN" if dynamics is None else "PASS" if dynamics.get("status") == "PASS" else "FAIL"),
        "dynamics": dynamics,
        "bottle_qpos_writes_during_rollout": 0,
        "bottle_weld_or_mocap": False,
        "next_action": ("Review physical evidence and source/servo classification."
                        if task_status.startswith("PASS") else
                        "Address only the first measured failed gate; preserve the source geometry and all recorded candidate traces."),
        "pr58_modified": False,
    }
    write_json(args.output / "result.json", result)
    command = " ".join(["MUJOCO_GL=egl", sys.executable, "-B", str(Path(__file__)),
                        "--x2-root", str(args.x2_root), "--canonical-helper", str(args.canonical_helper),
                        "--output", str(args.output)])
    (args.output / "experiment_commands.txt").write_text(command + "\n", encoding="utf-8")
    report = [
        "# Issue #46: Native OmniPicker Workcell Recovery",
        "",
        f"Status: **{result['status']}**",
        "",
        "This packet is limited to the pinned native X2 collision meshes and source joint ranges. The paired-jaw servo remains SIMULATION_ONLY_M0. No source mesh, source range, bottle property, table definition, or collision exclusion was changed.",
        "",
        "## Candidate results",
        "",
    ]
    for c in candidates:
        report.extend([
            f"### {c['layout']['name']}",
            "",
            f"Base: `{c['base_position_world_m']}` m; yaw `{c['base_yaw_deg']:.1f} deg`; table-facing heading error `{c['heading_error_to_table_center_deg']:.2f} deg`.",
            f"Initial table minimum: `{c['initial_scene']['minimum_robot_table']['signed_distance_m']:.6f} m`; pregrasp failure: `{c.get('first_failure', 'none')}`.",
            "",
        ])
        if c.get("pregrasp"):
            p = c["pregrasp"]
            report.append(f"Pregrasp residual: `{1000*p['position_error_m']:.3f} mm`, `{p['rotation_error_deg']:.3f} deg`; pose gate `{p['pose_gate_pass']}`, collision gate `{p['collision_gate_pass']}`.")
            report.append("")
    if dynamics is not None:
        report.extend([
            "## Physical dynamics",
            "",
            f"Status: **{dynamics['status']}**; first failed gate: `{dynamics.get('first_failed_gate')}`.",
            f"Bottle lift max/final-air: `{1000*dynamics['maximum_bottle_lift_m']:.2f} mm` / `{1000*dynamics['final_air_hold_lift_m']:.2f} mm`; final table contacts: `{dynamics['final_table_contact_count']}`.",
            f"Bilateral HOLD contact fractions: `{dynamics['hold_contact_fraction_by_family']}`; final air velocity: `{dynamics['final_air_hold_bottle_velocity_m_s']:.6f} m/s`.",
            f"Trace: `{dynamics['trace_path']}`; video: `{dynamics['video_path']}`.",
            "",
        ])
    report.extend(["## Reproduction", "", "```bash", command, "```", ""])
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return 0 if result["source_limited_static_grasp_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
