#!/usr/bin/env python3
"""Bounded source-limited X2 + Robotiq static reachability checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0
import issue46_x2_robotiq_mounted_smoke as smoke


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-x2-reachability-20261009"
)
STATION_CANDIDATES = (
    {"name": "right_side_established_clear_station", "base": [-0.08, 0.0, 0.68],
     "approach_yaw_deg": 90.0,
     "rationale": "Retain the previously measured outboard station that avoided table-leg overlap; approach the body from the robot's right-side workspace."},
    {"name": "natural_front_approach_clear_of_leg", "base": [0.0, 0.16, 0.68],
     "approach_yaw_deg": 0.0,
     "rationale": "Keep the torso facing the table and shift the base 80 mm laterally from the prior station to clear the measured right-coupler/table-leg overlap."},
    {"name": "right_side_lower_body_contact", "base": [-0.08, 0.0, 0.68],
     "approach_yaw_deg": 90.0, "grasp_height_offset_m": -0.025,
     "rationale": "Keep the measured clear right-side station and move the grasp point 25 mm lower within the canonical cylindrical body to test contact-height sensitivity."},
)
MAX_POSE_ERROR_M = 0.003
MAX_ORIENTATION_ERROR_RAD = 0.02
APPROACH_DISTANCE_M = 0.12
LIFT_DISTANCE_M = 0.03
COLLISION_CLEARANCE_M = 0.002
COLLISION_RESIDUAL_WEIGHT = 3.0
DISTANCE_MAX_M = 1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, qids: list[int]) -> None:
    data.qpos[qids] = q
    mujoco.mj_forward(model, data)


def relevant_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                      robot_bodies: set[str]) -> list[dict[str, Any]]:
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n1, n2 = name(model, mujoco.mjtObj.mjOBJ_BODY, b1), name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if n1 not in robot_bodies and n2 not in robot_bodies:
            continue
        gname1, gname2 = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        kind = "self" if n1 in robot_bodies and n2 in robot_bodies else (
            "bottle" if "m0_bottle" in (n1, n2) else (
                "table" if n1.startswith("m0_table") or n2.startswith("m0_table") else (
                    "floor" if "m0_floor" in (gname1, gname2) else "other")))
        # Planted ankle/floor contact is an intentional support contact.
        ankle_floor = kind == "floor" and ("ankle_roll_link" in n1 + n2)
        out.append({"kind": kind, "allowed_planted_floor_contact": ankle_floor,
                    "geom1_id": g1, "geom1": gname1, "body1": n1,
                    "geom2_id": g2, "geom2": gname2, "body2": n2,
                    "distance_m": float(c.dist), "position_world_m": c.pos.astype(float).tolist()})
    return out


def collision_cost(contacts: list[dict[str, Any]]) -> float:
    return float(sum(max(0.0, COLLISION_CLEARANCE_M - c["distance_m"])
                     for c in contacts if not c["allowed_planted_floor_contact"]))


def environment_pair_catalog(model: mujoco.MjModel, robot_bodies: set[str]) -> list[dict[str, Any]]:
    """Return collision-compatible robot/table and robot/bottle geom pairs."""
    robot_geoms = [g for g in range(model.ngeom)
                   if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) in robot_bodies
                   and (int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0)]
    environment_geoms = [g for g in range(model.ngeom)
                         if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])).startswith("m0_table")
                         or name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == "m0_bottle"]
    pairs = []
    for rg in robot_geoms:
        for eg in environment_geoms:
            if not ((int(model.geom_contype[rg]) & int(model.geom_conaffinity[eg])) or
                    (int(model.geom_contype[eg]) & int(model.geom_conaffinity[rg]))):
                continue
            env_body = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[eg]))
            pairs.append({
                "category": "robot_bottle" if env_body == "m0_bottle" else "robot_table",
                "robot_geom_id": rg,
                "robot_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, rg),
                "robot_body": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[rg])),
                "environment_geom_id": eg,
                "environment_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, eg),
                "environment_body": env_body,
            })
    return pairs


def robust_geom_distance(model: mujoco.MjModel, data: mujoco.MjData,
                         geom1: int, geom2: int) -> tuple[float, dict[str, Any]]:
    """Query signed distance, detecting the native-CCD zero/witness inconsistency."""
    segment = np.zeros(6, dtype=np.float64)
    native = float(mujoco.mj_geomDistance(model, data, geom1, geom2, DISTANCE_MAX_M, segment))
    witness_length = float(np.linalg.norm(segment[3:] - segment[:3]))
    diagnostic: dict[str, Any] = {"native_signed_distance_m": native,
                                  "native_witness_length_m": witness_length,
                                  "distance_method": "mujoco_mj_geomDistance"}
    pair_contacts = [float(data.contact[i].dist) for i in range(data.ncon)
                     if {int(data.contact[i].geom1), int(data.contact[i].geom2)} == {geom1, geom2}]
    if native != 0.0:
        return native, diagnostic
    if pair_contacts:
        distance = min(pair_contacts)
        diagnostic.update({"distance_method": "active_contact_distance_after_native_zero",
                           "active_contact_distances_m": pair_contacts,
                           "fallback_signed_distance_m": distance})
        return distance, diagnostic
    if witness_length <= 1e-9:
        diagnostic["distance_method"] = "native_zero_without_witness_or_active_contact"
        return 0.0, diagnostic

    # MuJoCo can return zero with a non-degenerate native-CCD witness for a
    # separated convex mesh pair. The legacy path is useful evidence, but can
    # also return a false negative. Cross-check that case against the actual
    # compiled collision pipeline at a temporary 2 mm query clearance.
    disable_bit = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    old_flags = int(model.opt.disableflags)
    try:
        model.opt.disableflags = old_flags | disable_bit
        fallback_segment = np.zeros(6, dtype=np.float64)
        fallback = float(mujoco.mj_geomDistance(model, data, geom1, geom2,
                                                DISTANCE_MAX_M, fallback_segment))
    finally:
        model.opt.disableflags = old_flags
    diagnostic.update({"distance_method": "legacy_ccd_zero_anomaly_crosscheck",
                       "fallback_signed_distance_m": fallback,
                       "fallback_witness_length_m": float(np.linalg.norm(
                           fallback_segment[3:] - fallback_segment[:3]))})
    old_margin1 = float(model.geom_margin[geom1])
    old_margin2 = float(model.geom_margin[geom2])
    query_pairs: list[dict[str, Any]] = []
    try:
        model.geom_margin[geom1] = old_margin1 + 0.5 * COLLISION_CLEARANCE_M
        model.geom_margin[geom2] = old_margin2 + 0.5 * COLLISION_CLEARANCE_M
        mujoco.mj_forward(model, data)
        query_pairs = [
            {"distance_m": float(data.contact[i].dist),
             "geom1": int(data.contact[i].geom1), "geom2": int(data.contact[i].geom2)}
            for i in range(data.ncon)
            if {int(data.contact[i].geom1), int(data.contact[i].geom2)} == {geom1, geom2}
        ]
    finally:
        model.geom_margin[geom1] = old_margin1
        model.geom_margin[geom2] = old_margin2
        mujoco.mj_forward(model, data)
    diagnostic["collision_pipeline_query_clearance_m"] = COLLISION_CLEARANCE_M
    diagnostic["collision_pipeline_query_contacts"] = query_pairs
    if query_pairs:
        diagnostic["distance_method"] = "collision_pipeline_contact_within_query_clearance"
        diagnostic["clearance_lower_bound_m"] = 0.0
        return 0.0, diagnostic
    if fallback > 0.0:
        diagnostic["distance_method"] = "legacy_ccd_positive_with_collision_pipeline_clearance_check"
        diagnostic["clearance_lower_bound_m"] = COLLISION_CLEARANCE_M
        return fallback, diagnostic
    diagnostic["distance_method"] = "collision_pipeline_clearance_lower_bound_after_distance_disagreement"
    diagnostic["clearance_lower_bound_m"] = COLLISION_CLEARANCE_M
    diagnostic["distance_is_lower_bound"] = True
    return COLLISION_CLEARANCE_M, diagnostic


def environment_distance_snapshot(model: mujoco.MjModel, data: mujoco.MjData,
                                  pair_catalog: list[dict[str, Any]],
                                  keep_all: bool = False) -> dict[str, Any]:
    rows = []
    for pair in pair_catalog:
        distance, diagnostic = robust_geom_distance(
            model, data, pair["robot_geom_id"], pair["environment_geom_id"])
        rows.append({**pair, "signed_distance_m": distance, **diagnostic})
    rows.sort(key=lambda row: row["signed_distance_m"])
    by_category = {}
    for category in ("robot_bottle", "robot_table"):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "minimum_signed_distance_m": selected[0]["signed_distance_m"] if selected else None,
            "near_pairs": [row for row in selected if row["signed_distance_m"] < 0.05][:12],
        }
    violating = [row for row in rows if row["signed_distance_m"] < COLLISION_CLEARANCE_M]
    return {"minimum_signed_distance_m": rows[0]["signed_distance_m"] if rows else None,
            "minimum_pair": rows[0] if rows else None,
            "by_category": by_category,
            "pairs_below_clearance": violating,
            "all_pairs": rows if keep_all else None}


def distance_residuals(snapshot: dict[str, Any], pair_count: int) -> np.ndarray:
    rows = snapshot.get("all_pairs")
    if rows is None:
        raise RuntimeError("Optimizer distance snapshot must retain all pair rows")
    residual = np.asarray([max(0.0, COLLISION_CLEARANCE_M - row["signed_distance_m"])
                           for row in rows], dtype=np.float64)
    if residual.size != pair_count:
        raise RuntimeError("Collision pair count changed during optimization")
    return residual / COLLISION_CLEARANCE_M * COLLISION_RESIDUAL_WEIGHT


def pose_errors(model: mujoco.MjModel, data: mujoco.MjData, site: int,
                target_pos: np.ndarray, target_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_err = data.site_xpos[site].copy() - target_pos
    rot = data.site_xmat[site].reshape(3, 3).copy()
    rot_err = Rotation.from_matrix(target_rot.T @ rot).as_rotvec()
    return pos_err, rot_err


def orientation_seed(model: mujoco.MjModel, data: mujoco.MjData, qids: list[int],
                     q0: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     target_rot: np.ndarray) -> dict[str, Any]:
    site = m0.obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    wrist_names = m0.ARM[-3:]
    wrist_ids = [m0.qpos_id(model, n) for n in wrist_names]
    seed_values = np.asarray([q0[m0.ARM.index(n)] for n in wrist_names])
    wrist_lower = np.asarray([lower[m0.ARM.index(n)] for n in wrist_names])
    wrist_upper = np.asarray([upper[m0.ARM.index(n)] for n in wrist_names])
    trace = []

    def residual(qw: np.ndarray) -> np.ndarray:
        q = q0.copy()
        for n, v in zip(wrist_names, qw):
            q[m0.ARM.index(n)] = v
        set_arm(model, data, q, qids)
        _, rot_err = pose_errors(model, data, site, data.site_xpos[site].copy(), target_rot)
        trace.append({"evaluation": len(trace), "wrist_q_rad": qw.tolist(),
                      "full_arm_q_rad": q.tolist(), "orientation_error_rad": rot_err.tolist()})
        return rot_err / 0.02

    opt = least_squares(residual, np.clip(seed_values, wrist_lower, wrist_upper),
                         bounds=(wrist_lower, wrist_upper), max_nfev=150,
                         xtol=1e-11, ftol=1e-11, gtol=1e-11)
    q = q0.copy()
    for n, v in zip(wrist_names, opt.x):
        q[m0.ARM.index(n)] = v
    set_arm(model, data, q, qids)
    rot_err = Rotation.from_matrix(target_rot.T @ data.site_xmat[site].reshape(3, 3)).as_rotvec()
    return {"success": bool(opt.success), "message": str(opt.message), "nfev": int(opt.nfev),
            "wrist_joint_names": wrist_names, "q_rad": q.tolist(),
            "final_orientation_error_rad": rot_err.tolist(), "final_orientation_error_norm_rad": float(np.linalg.norm(rot_err)),
            "trace": trace}


def solve_pose(model: mujoco.MjModel, data: mujoco.MjData, qids: list[int],
               lower: np.ndarray, upper: np.ndarray, seed: np.ndarray,
               target_pos: np.ndarray, target_rot: np.ndarray, robot_bodies: set[str],
               pair_catalog: list[dict[str, Any]], label: str) -> dict[str, Any]:
    site = m0.obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    trace = []

    def residual(q: np.ndarray) -> np.ndarray:
        set_arm(model, data, q, qids)
        pos_err, rot_err = pose_errors(model, data, site, target_pos, target_rot)
        pairs = relevant_contacts(model, data, robot_bodies)
        distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
        collision_rows = distances["all_pairs"]
        distance_residual = distance_residuals(distances, len(pair_catalog))
        coll_cost = float(sum(max(0.0, COLLISION_CLEARANCE_M - row["signed_distance_m"])
                              for row in collision_rows))
        trace.append({"evaluation": len(trace), "q_rad": q.tolist(),
                      "position_error_world_m": pos_err.tolist(),
                      "position_error_norm_m": float(np.linalg.norm(pos_err)),
                      "orientation_error_world_rad": rot_err.tolist(),
                      "orientation_error_norm_rad": float(np.linalg.norm(rot_err)),
                      "collision_cost_m": coll_cost,
                      "minimum_robot_bottle_distance_m": distances["by_category"]["robot_bottle"]["minimum_signed_distance_m"],
                      "minimum_robot_table_distance_m": distances["by_category"]["robot_table"]["minimum_signed_distance_m"],
                      "minimum_pair": distances["minimum_pair"],
                      "pairs_below_clearance": distances["pairs_below_clearance"],
                      "active_relevant_contacts": pairs})
        return np.concatenate((pos_err / 0.005, rot_err / 0.02, distance_residual))

    opt = least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                         bounds=(lower, upper), max_nfev=400, x_scale="jac",
                         xtol=1e-11, ftol=1e-11, gtol=1e-11)
    set_arm(model, data, opt.x, qids)
    pos_err, rot_err = pose_errors(model, data, site, target_pos, target_rot)
    pairs = relevant_contacts(model, data, robot_bodies)
    distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
    errors, velocity_errors = m0.all_limited_joint_checks(model, data, {})
    exact_collision_free = all(row["signed_distance_m"] > 0.0 for row in distances["all_pairs"])
    return {"label": label, "success": bool(opt.success), "message": str(opt.message),
            "nfev": int(opt.nfev), "joint_pose_rad": opt.x.tolist(),
            "target_position_world_m": target_pos.tolist(), "target_rotation_world": target_rot.tolist(),
            "position_error_world_m": pos_err.tolist(), "position_error_norm_m": float(np.linalg.norm(pos_err)),
            "orientation_error_world_rad": rot_err.tolist(), "orientation_error_norm_rad": float(np.linalg.norm(rot_err)),
            "active_relevant_contacts": pairs, "collision_cost_m": float(sum(
                max(0.0, COLLISION_CLEARANCE_M - row["signed_distance_m"])
                for row in distances["all_pairs"])),
            "exact_environment_distances": distances,
            "exact_environment_collision_free": exact_collision_free,
            "joint_limit_violations": errors, "velocity_limit_violations": velocity_errors,
            "pose_gate_pass": bool(np.linalg.norm(pos_err) <= MAX_POSE_ERROR_M and
                                    np.linalg.norm(rot_err) <= MAX_ORIENTATION_ERROR_RAD),
            "collision_gate_pass": exact_collision_free and not any(
                not c["allowed_planted_floor_contact"] and
                (c["distance_m"] < 0 or (c["kind"] == "bottle" and c["distance_m"] <= 0))
                for c in pairs),
            "trace": trace}


def solve_position_seed(model: mujoco.MjModel, data: mujoco.MjData, qids: list[int],
                       lower: np.ndarray, upper: np.ndarray, seed: np.ndarray,
                       target_pos: np.ndarray, robot_bodies: set[str],
                       pair_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Find a source-limited FK seed for the fixed target position only."""
    site = m0.obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    trace = []

    def residual(q: np.ndarray) -> np.ndarray:
        set_arm(model, data, q, qids)
        pos_err = data.site_xpos[site].copy() - target_pos
        pairs = relevant_contacts(model, data, robot_bodies)
        distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
        trace.append({"evaluation": len(trace), "q_rad": q.tolist(),
                      "position_error_world_m": pos_err.tolist(),
                      "position_error_norm_m": float(np.linalg.norm(pos_err)),
                      "minimum_robot_bottle_distance_m": distances["by_category"]["robot_bottle"]["minimum_signed_distance_m"],
                      "minimum_robot_table_distance_m": distances["by_category"]["robot_table"]["minimum_signed_distance_m"],
                      "minimum_pair": distances["minimum_pair"],
                      "pairs_below_clearance": distances["pairs_below_clearance"],
                      "active_relevant_contacts": pairs})
        return pos_err / 0.005

    opt = least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                         bounds=(lower, upper), max_nfev=400, x_scale="jac",
                         xtol=1e-11, ftol=1e-11, gtol=1e-11)
    set_arm(model, data, opt.x, qids)
    err = data.site_xpos[site].copy() - target_pos
    distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
    return {"success": bool(opt.success), "message": str(opt.message),
            "nfev": int(opt.nfev), "joint_pose_rad": opt.x.tolist(),
            "position_error_world_m": err.tolist(),
            "position_error_norm_m": float(np.linalg.norm(err)),
            "exact_environment_distances": distances,
            "collision_objective_used": False,
            "exact_environment_collision_free": all(row["signed_distance_m"] > 0.0
                                                      for row in distances["all_pairs"]),
            "active_relevant_contacts": relevant_contacts(model, data, robot_bodies),
            "trace": trace}


def state_metrics(model: mujoco.MjModel, data: mujoco.MjData, target_pos: np.ndarray,
                  target_rot: np.ndarray, robot_bodies: set[str]) -> dict[str, Any]:
    site = m0.obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    pads_left = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_left_pad1", "rq_left_pad2")]
    pads_right = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_right_pad1", "rq_right_pad2")]
    left = np.mean(data.geom_xpos[pads_left], axis=0)
    right = np.mean(data.geom_xpos[pads_right], axis=0)
    mid = (left + right) / 2.0
    rot = data.site_xmat[site].reshape(3, 3).copy()
    pos_err, rot_err = pose_errors(model, data, site, target_pos, target_rot)
    return {"tcp_position_world_m": data.site_xpos[site].astype(float).tolist(),
            "pad_midpoint_world_m": mid.astype(float).tolist(),
            "pad_center_position_error_to_bottle_m": (mid - target_pos).astype(float).tolist(),
            "pad_separation_open_m": float(np.linalg.norm(left - right)),
            "left_pad_center_world_m": left.astype(float).tolist(),
            "right_pad_center_world_m": right.astype(float).tolist(),
            "tcp_to_target_position_error_m": pos_err.astype(float).tolist(),
            "tcp_to_target_orientation_error_rad": rot_err.astype(float).tolist(),
            "insertion_axis_world": rot[:, 2].astype(float).tolist(),
            "jaw_opening_axis_world": (right - left).astype(float).tolist(),
            "active_relevant_contacts": relevant_contacts(model, data, robot_bodies)}


def exact_environment_distances(model: mujoco.MjModel, data: mujoco.MjData,
                                robot_bodies: set[str], limit: int = 24) -> dict[str, Any]:
    pair_catalog = environment_pair_catalog(model, robot_bodies)
    snapshot = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
    distances = snapshot["all_pairs"]
    summary = {}
    for category in ("robot_bottle", "robot_table"):
        items = [x for x in distances if x["category"] == category]
        summary[category] = {"minimum_signed_distance_m": items[0]["signed_distance_m"] if items else None,
                             "nearest_pairs": items[:limit],
                             "pairs_below_clearance": [x for x in items
                                                       if x["signed_distance_m"] < COLLISION_CLEARANCE_M]}
    pads = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
            ("rq_left_pad1", "rq_left_pad2", "rq_right_pad1", "rq_right_pad2")]
    bottle_geoms = [g for g in range(model.ngeom)
                    if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == "m0_bottle"]
    pad_distances = []
    for pg in pads:
        for bg in bottle_geoms:
            distance, diagnostic = robust_geom_distance(model, data, pg, bg)
            pad_distances.append({"pad_geom_id": pg, "pad_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, pg),
                                  "bottle_geom_id": bg, "bottle_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, bg),
                                  "signed_distance_m": distance, **diagnostic})
    pad_distances.sort(key=lambda row: row["signed_distance_m"])
    summary["pad_to_bottle"] = {"minimum_signed_distance_m": pad_distances[0]["signed_distance_m"] if pad_distances else None,
                                "all_pairs": pad_distances}
    return {**summary, "all_pairs": distances,
            "pairs_below_clearance": snapshot["pairs_below_clearance"]}


def candidate_xml(base_model: mujoco.MjModel, base_xml: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    variant = output_dir / "x2_robotiq_2f85_head_torso_exclude_simulation_only.xml"
    exception = smoke.create_exclusion_variant(base_xml, variant)
    return variant, exception


def interp_collision_check(model: mujoco.MjModel, data: mujoco.MjData, qids: list[int],
                           qa: np.ndarray, qb: np.ndarray, robot_bodies: set[str],
                           pair_catalog: list[dict[str, Any]], segment: str,
                           samples: int = 21) -> dict[str, Any]:
    points = []
    for alpha in np.linspace(0.0, 1.0, samples):
        q = qa + alpha * (qb - qa)
        set_arm(model, data, q, qids)
        pairs = relevant_contacts(model, data, robot_bodies)
        distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
        disallowed = [p for p in pairs if not p["allowed_planted_floor_contact"] and
                      (p["distance_m"] < 0.0 or (p["kind"] == "bottle" and p["distance_m"] <= 0.0))]
        points.append({"fraction": float(alpha), "q_rad": q.tolist(),
                       "active_relevant_contacts": pairs,
                       "minimum_robot_bottle_distance_m": distances["by_category"]["robot_bottle"]["minimum_signed_distance_m"],
                       "minimum_robot_table_distance_m": distances["by_category"]["robot_table"]["minimum_signed_distance_m"],
                       "minimum_pair": distances["minimum_pair"],
                       "pairs_below_clearance": distances["pairs_below_clearance"],
                       "penetrating_unintended_contacts": disallowed,
                       "distance_clear": all(row["signed_distance_m"] > 0.0
                                             for row in distances["all_pairs"])})
    return {"segment": segment, "samples": points,
            "pass": not any(p["penetrating_unintended_contacts"] or not p["distance_clear"]
                            for p in points)}


def evaluate_candidate(index: int, config: dict[str, Any], args, ident: dict[str, Any], canonical,
                       source_pose: np.ndarray) -> dict[str, Any]:
    out = args.output_dir.resolve() / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "static").mkdir(exist_ok=True)
    build_args = SimpleNamespace(x2_root=args.x2_root, menagerie_root=args.menagerie_root,
                                  station_base_pos=np.asarray(config["base"], dtype=float))
    base_model, adapter = m0.build_model(build_args, out, canonical)
    base_xml = Path(adapter["model_xml"])
    variant_path, exception = candidate_xml(base_model, base_xml, out)
    model = mujoco.MjModel.from_xml_path(str(variant_path))
    data = mujoco.MjData(model)
    smoke.set_mounted_neutral(model, data)
    data.ctrl[m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")] = 0.0
    mujoco.mj_forward(model, data)
    qids, jids, lower, upper = m0.arm_metadata(model, ident)
    q0 = data.qpos[qids].copy()
    robot_bodies = smoke.robot_body_names(model)
    pair_catalog = environment_pair_catalog(model, robot_bodies)
    initial_contacts = smoke.describe_contacts(model, data, robot_bodies)
    initial_distances = environment_distance_snapshot(model, data, pair_catalog, keep_all=True)
    initial_source_ranges, _ = m0.all_limited_joint_checks(model, data,
        {k: v["velocity"] for k, v in ident["urdf_joint_limits"].items()
         if math.isfinite(v["velocity"]) and v["velocity"] > 0})
    initial_robot_table = [c for c in initial_contacts["all_contacts"]
                           if (c["body1"] in robot_bodies and c["body2"].startswith("m0_table")) or
                           (c["body2"] in robot_bodies and c["body1"].startswith("m0_table"))]
    initial_environment_overlap = [row for row in initial_distances["all_pairs"]
                                   if row["signed_distance_m"] <= 0.0]
    if (initial_contacts["robot_self_contacts"] or initial_contacts["robot_bottle_contacts"] or
            initial_robot_table or initial_environment_overlap):
        return {"candidate_index": index, "configuration": config, "status": "FAIL_INITIAL_CLEARANCE",
                "model_path": str(variant_path), "model_sha256": sha256(variant_path),
                "initial_contacts": initial_contacts, "initial_robot_table_contacts": initial_robot_table,
                "initial_environment_overlaps": initial_environment_overlap,
                "initial_joint_limit_violations": initial_source_ranges,
                "exclusion": exception}

    bottle_geom = m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    bottle_target = data.geom_xpos[bottle_geom].copy()
    bottle_target[2] += float(config.get("grasp_height_offset_m", 0.0))
    r0 = np.asarray(adapter["desired_world_rotation_matrix"], dtype=float)
    yaw = math.radians(float(config["approach_yaw_deg"]))
    rz = Rotation.from_rotvec(np.array([0.0, 0.0, yaw])).as_matrix()
    target_rot = rz @ r0
    pitch = math.radians(float(config.get("pitch_about_jaw_axis_deg", 0.0)))
    jaw_axis_world = target_rot[:, 0].copy()
    target_rot = Rotation.from_rotvec(jaw_axis_world * pitch).as_matrix() @ target_rot
    insertion_axis = target_rot[:, 2]
    pre_target = bottle_target - insertion_axis * APPROACH_DISTANCE_M
    lift_target = bottle_target + np.array([0.0, 0.0, LIFT_DISTANCE_M])

    orient = orientation_seed(model, data, qids, q0, lower, upper, target_rot)
    # The wrist-only orientation solution is diagnostic: it can move the mounted
    # gripper through the torso/hip before the arm is repositioned. Preserve the
    # neutral-seed solve, then derive a position-only FK seed for a second,
    # collision-aware full-pose solve at the exact same target.
    pre_neutral = solve_pose(model, data, qids, lower, upper, q0.copy(),
                             pre_target, target_rot, robot_bodies, pair_catalog,
                             "OPEN_PREGRASP_FROM_CLEAR_NEUTRAL")
    position_seed = solve_position_seed(model, data, qids, lower, upper, q0.copy(),
                                        pre_target, robot_bodies, pair_catalog)
    pre_position = solve_pose(model, data, qids, lower, upper,
                              np.asarray(position_seed["joint_pose_rad"]),
                              pre_target, target_rot, robot_bodies, pair_catalog,
                              "OPEN_PREGRASP_FROM_POSITION_SEED")
    pre_orientation = solve_pose(model, data, qids, lower, upper,
                                 np.asarray(orient["q_rad"]),
                                 pre_target, target_rot, robot_bodies, pair_catalog,
                                 "OPEN_PREGRASP_FROM_ORIENTATION_SEED")
    source_seed_solutions = {}
    for fraction in (0.25, 0.75):
        seed = lower + fraction * (upper - lower)
        source_seed_solutions[f"source_range_{int(fraction * 100)}pct"] = solve_pose(
            model, data, qids, lower, upper, seed, pre_target, target_rot,
            robot_bodies, pair_catalog,
            f"OPEN_PREGRASP_FROM_SOURCE_RANGE_{int(fraction * 100)}PCT_SEED")
    sobol_points = qmc.Sobol(d=len(lower), scramble=False).random_base2(m=4)[1:9]
    for seed_index, sample in enumerate(sobol_points):
        seed = lower + sample * (upper - lower)
        seed = np.clip(seed, lower + 1e-8, upper - 1e-8)
        source_seed_solutions[f"sobol_source_valid_{seed_index:02d}"] = solve_pose(
            model, data, qids, lower, upper, seed, pre_target, target_rot,
            robot_bodies, pair_catalog,
            f"OPEN_PREGRASP_FROM_SOBOL_SOURCE_VALID_SEED_{seed_index:02d}")

    def pre_score(solution: dict[str, Any]) -> tuple[int, int, float]:
        return (not solution["pose_gate_pass"], not solution["collision_gate_pass"],
                solution["position_error_norm_m"] / MAX_POSE_ERROR_M +
                solution["orientation_error_norm_rad"] / MAX_ORIENTATION_ERROR_RAD +
                solution["collision_cost_m"] / COLLISION_CLEARANCE_M)

    pre = min((pre_neutral, pre_position, pre_orientation, *source_seed_solutions.values()),
              key=pre_score)
    grasp = None
    lift = None
    corridor = []
    pad_alignment = None
    if pre["pose_gate_pass"] and pre["collision_gate_pass"]:
        grasp = solve_pose(model, data, qids, lower, upper, np.asarray(pre["joint_pose_rad"]),
                           bottle_target, target_rot, robot_bodies, pair_catalog,
                           "OPEN_GRASP_CENTER")
    if grasp and grasp["pose_gate_pass"] and grasp["collision_gate_pass"]:
        pad_alignment = state_metrics(model, data, bottle_target, target_rot, robot_bodies)
        pad_alignment["exact_compiled_geom_distances"] = exact_environment_distances(model, data, robot_bodies)
        lift = solve_pose(model, data, qids, lower, upper, np.asarray(grasp["joint_pose_rad"]),
                          lift_target, target_rot, robot_bodies, pair_catalog,
                          "OPEN_LIFT_30MM")
    if grasp and grasp["pose_gate_pass"] and grasp["collision_gate_pass"] and lift and lift["pose_gate_pass"] and lift["collision_gate_pass"]:
        corridor = [
            interp_collision_check(model, data, qids, q0, np.asarray(pre["joint_pose_rad"]), robot_bodies,
                                   pair_catalog, "HOME_TO_PREGRASP_OPEN"),
            interp_collision_check(model, data, qids, np.asarray(pre["joint_pose_rad"]), np.asarray(grasp["joint_pose_rad"]),
                                   robot_bodies, pair_catalog, "PREGRASP_TO_GRASP_OPEN"),
            interp_collision_check(model, data, qids, np.asarray(grasp["joint_pose_rad"]), np.asarray(lift["joint_pose_rad"]),
                                   robot_bodies, pair_catalog, "GRASP_TO_LIFT_30MM_OPEN"),
        ]
        for segment in corridor:
            segment["minimum_environment_clearance_samples"] = []
            for sample in segment["samples"]:
                set_arm(model, data, np.asarray(sample["q_rad"]), qids)
                clearance = exact_environment_distances(model, data, robot_bodies, limit=1)
                segment["minimum_environment_clearance_samples"].append({
                    "fraction": sample["fraction"],
                    "robot_bottle_min_signed_distance_m": clearance["robot_bottle"]["minimum_signed_distance_m"],
                    "robot_table_min_signed_distance_m": clearance["robot_table"]["minimum_signed_distance_m"],
                    "pad_bottle_min_signed_distance_m": clearance["pad_to_bottle"]["minimum_signed_distance_m"],
                })

    summaries = {"pregrasp": pre, "pregrasp_from_clear_neutral": pre_neutral,
                 "position_only_fk_seed": position_seed,
                 "pregrasp_from_position_seed": pre_position,
                 "pregrasp_from_orientation_seed": pre_orientation,
                 "pregrasp_from_source_valid_multistart": source_seed_solutions,
                 "grasp": grasp, "lift_30mm": lift}
    path_pass = bool(corridor) and all(item["pass"] for item in corridor)
    gripper_range = model.jnt_range[m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rq_right_driver_joint")].tolist()
    open_gap = float(adapter["pad_center_separation_open_m"])
    # The previously executed mounted cycle supplies a dynamic travel interval;
    # here the static requirement is that the canonical diameter fits inside it.
    bottle_diameter = float(2.0 * model.geom_size[bottle_geom, 0])
    static_gate = bool(pre["pose_gate_pass"] and pre["collision_gate_pass"] and
                       grasp and grasp["pose_gate_pass"] and grasp["collision_gate_pass"] and
                       lift and lift["pose_gate_pass"] and lift["collision_gate_pass"] and path_pass and
                       pad_alignment and np.linalg.norm(pad_alignment["pad_center_position_error_to_bottle_m"]) <= MAX_POSE_ERROR_M and
                       open_gap > bottle_diameter)

    # Preserve endpoint frames as exact geometry evidence for this candidate.
    for label, solution in (("pregrasp", pre), ("grasp_open", grasp), ("lift30_open", lift)):
        if solution is None:
            continue
        set_arm(model, data, np.asarray(solution["joint_pose_rad"]), qids)
        m0.render(model, data, out / "static" / f"{label}.png", (0.25, 0.0, 0.9), 1.45, 135, -10)
    raw = {
        "candidate_index": index,
        "configuration": config,
        "status": "STATIC_PASS" if static_gate else "STATIC_FAIL",
        "model": {"base_xml": str(base_xml), "base_xml_sha256": sha256(base_xml),
                  "experimental_xml": str(variant_path), "experimental_xml_sha256": sha256(variant_path),
                  "exclusion": exception, "timestep_s": float(model.opt.timestep)},
        "initial_configuration": {"head_yaw_rad": float(data.qpos[m0.qpos_id(model, "head_yaw_joint")]),
                                   "head_pitch_rad": float(data.qpos[m0.qpos_id(model, "head_pitch_joint")]),
                                   "left_elbow_rad": float(data.qpos[m0.qpos_id(model, "left_elbow_joint")]),
                                   "right_elbow_rad": float(data.qpos[m0.qpos_id(model, "right_elbow_joint")]),
                                   "right_arm_joint_ranges_rad": {n: model.jnt_range[j].astype(float).tolist()
                                                                   for n, j in zip(m0.ARM, jids)},
                                   "source_joint_limit_violations": initial_source_ranges,
                                   "initial_contacts": initial_contacts,
                                   "initial_environment_distances": initial_distances},
        "grasp_frame_definition": {
            "tcp": "compiled Robotiq pad-center midpoint site rq_m0_tcp",
            "target_point_source": "compiled canonical bottle_body collision geom center",
            "bottle_target_position_world_m": bottle_target.astype(float).tolist(),
            "bottle_body_collision_geom_size_m": model.geom_size[bottle_geom].astype(float).tolist(),
            "bottle_diameter_m": bottle_diameter,
            "approach_distance_m": APPROACH_DISTANCE_M,
            "pregrasp_position_world_m": pre_target.astype(float).tolist(),
            "lift_target_world_m": lift_target.astype(float).tolist(),
            "approach_yaw_deg_about_bottle_vertical": float(config["approach_yaw_deg"]),
            "target_rotation_world": target_rot.tolist(),
            "insertion_axis_world": insertion_axis.tolist(),
            "open_pad_separation_m": open_gap,
            "measured_dynamic_closed_gap_reference_m": 0.014690997223188616,
            "open_gap_contains_bottle_diameter": bool(open_gap > bottle_diameter),
            "source_driver_range_rad": gripper_range,
            "pitch_about_jaw_axis_deg": float(config.get("pitch_about_jaw_axis_deg", 0.0)),
            "pitch_world_axis": jaw_axis_world.astype(float).tolist(),
            "pregrasp_vertical_offset_from_pitch_m": float(pre_target[2] - bottle_target[2]),
        },
        "orientation_only_reachable_fk_seed": orient,
        "position_only_fk_seed": position_seed,
        "full_pose_ik_seed": {
            "source": pre["label"],
            "joint_pose_rad": pre["trace"][0]["q_rad"],
            "initial_relevant_contacts": pre["trace"][0]["active_relevant_contacts"],
        },
        "ik": summaries,
        "pad_alignment_at_open_grasp": pad_alignment,
        "interpolated_collision_corridor": corridor,
        "gates": {
            "source_limited_pregrasp": bool(pre["pose_gate_pass"] and pre["collision_gate_pass"]),
            "open_grasp_alignment": bool(grasp and grasp["pose_gate_pass"] and grasp["collision_gate_pass"] and
                                          pad_alignment and np.linalg.norm(pad_alignment["pad_center_position_error_to_bottle_m"]) <= 0.003),
            "reachable_30mm_tcp_lift": bool(lift and lift["pose_gate_pass"] and lift["collision_gate_pass"]),
            "collision_free_approach_and_lift": path_pass,
            "static_corridor": static_gate,
        },
        "screenshots": sorted(str(p) for p in (out / "static").glob("*.png")),
    }
    write_json(out / "raw" / "candidate_trace.json", raw)
    return raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=m0.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        "experiment": "Bounded source-limited X2 + Robotiq static reachability with signed-distance-aware IK; no bottle dynamics",
        "status": "BLOCKED", "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
                                            "robot_sim_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                                            "command": sys.argv},
        "identity": {}, "candidates": [],
        "solver": {"collision_objective": "per-pair signed distance residual evaluated at every least_squares iteration",
                   "clearance_residual_margin_m": COLLISION_CLEARANCE_M,
                   "native_zero_witness_fallback": "cross-check only when native mj_geomDistance returns zero with a nondegenerate witness segment",
                   "final_clearance_gate": "every collision-compatible robot/table and robot/bottle pair has signed distance > 0 and no disallowed active contacts",
                   "additional_seed_set": "neutral, position-only, orientation-only, and 25/75 percent source-range joint seeds"},
        "budget": {"maximum_distinct_configurations": len(STATION_CANDIDATES), "configurations": STATION_CANDIDATES},
        "physics_run": False,
    }
    try:
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper changed from the accepted hash")
        ident = m0.identity(args, args.canonical_helper)
        result["identity"] = ident
        canonical = m0.load_module(args.canonical_helper)
        for index, candidate in enumerate(STATION_CANDIDATES, start=1):
            raw = evaluate_candidate(index, candidate, args, ident, canonical,
                                     np.array([0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0]))
            candidate_trace = out / candidate["name"] / "raw" / "candidate_trace.json"
            if not candidate_trace.exists():
                write_json(candidate_trace, raw)
            result["candidates"].append({k: raw.get(k) for k in
                ("candidate_index", "configuration", "status", "model", "grasp_frame_definition", "gates", "screenshots")})
            write_json(out / "result.json", result)
            if raw.get("status") == "STATIC_PASS":
                result["status"] = "STATIC_PASS"
                result["selected_candidate"] = raw["configuration"]["name"]
                break
        if result["status"] != "STATIC_PASS":
            result["status"] = "BLOCKED_STATIC_REACHABILITY"
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result_json": str(out / "result.json"),
                      "candidates": result["candidates"], "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "STATIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
