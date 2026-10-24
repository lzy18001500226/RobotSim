#!/usr/bin/env python3
"""Bounded zero-step DFQ grasp geometry search over three body-height families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_inspire_hand as hand  # noqa: E402
import run_g1_dfq_physical_grasp as physical  # noqa: E402
from g1_dfq_static_wrench import ContactWrench, pyramidal_wrench_rays, solve_static_wrench  # noqa: E402


ARM_NAMES = list(physical.runner.ARM_NAMES)
DRIVER_NAMES = [hand.channel_joint("R", suffix) for _, suffix in hand.CHANNELS]
BOTTLE_GEOMS = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}
CONTACT_GEOM = {
    "thumb": "dfq_collision_R_thumb_distal_0",
    "index": "dfq_collision_R_index_intermediate_0",
    "middle": "dfq_collision_R_middle_intermediate_0",
    "ring": "dfq_collision_R_ring_intermediate_0",
}
EFFORT_CAP_NM = 0.531
CONTACT_TARGET_M = -0.00003
MAX_PENETRATION_M = 0.0005
PENETRATION_GATE_TOLERANCE_M = 1e-7
MIN_OPEN_CLEARANCE_M = 0.0005
COLLISION_PENALTY_WEIGHT = 10000000.0
FORWARD_CALLS = 0


def name_id(model, kind, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"missing MuJoCo element: {name}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_joint(model, data, name: str, value: float) -> None:
    jid = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def bottle_body_ids(model) -> set[int]:
    return {name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in BOTTLE_GEOMS}


def set_mimics(model, data, mimic_map: dict) -> None:
    for _ in range(len(mimic_map) + 1):
        for follower, relation in mimic_map.items():
            parent_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, relation["parent"])
            q = float(data.qpos[int(model.jnt_qposadr[parent_id])])
            set_joint(model, data, follower,
                      float(relation["multiplier"]) * q + float(relation["offset"]))


def state_for_x(model, data, base_qpos, x, mimic_map) -> None:
    global FORWARD_CALLS
    data.qpos[:] = base_qpos
    for name, value in zip(ARM_NAMES + DRIVER_NAMES, x):
        set_joint(model, data, name, float(value))
    set_mimics(model, data, mimic_map)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    FORWARD_CALLS += 1


def geom_distance(model, data, bottle_gid: int, hand_gid: int) -> tuple[float, np.ndarray, np.ndarray]:
    fromto = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, bottle_gid, hand_gid, 0.25, fromto))
    return distance, fromto[:3].copy(), fromto[3:].copy()


def radial_normal(model, data, bottle_gid: int, point_world: np.ndarray) -> np.ndarray:
    center = np.asarray(data.geom_xpos[bottle_gid], dtype=float)
    rotation = np.asarray(data.geom_xmat[bottle_gid], dtype=float).reshape(3, 3)
    local = rotation.T @ (point_world - center)
    local[2] = 0.0
    norm = float(np.linalg.norm(local))
    return rotation @ (local / max(norm, 1e-12))


def robot_geom_ids(model) -> list[int]:
    pelvis = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    result = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body > 0 and body != pelvis:
            body = int(model.body_parentid[body])
        if body == pelvis:
            result.append(gid)
    return result


def collide_pair(model, first: int, second: int) -> bool:
    return bool((int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first])))


def open_path_clearance(model, data, base_qpos, x, pregrasp_arm, open_targets, mimics) -> dict:
    bottle_ids = bottle_body_ids(model)
    robot_ids = robot_geom_ids(model)
    best = {"distance_m": float("inf"), "robot_geom": None, "bottle_geom": None, "sample": None}
    path = np.linspace(np.asarray(pregrasp_arm), np.asarray(x[:7]), 81)
    for sample, arm_q in enumerate(path):
        q = np.asarray(x, dtype=float).copy()
        q[:7] = arm_q
        q[7:] = [float(open_targets[name]) for name in DRIVER_NAMES]
        state_for_x(model, data, base_qpos, q, mimics)
        if data.ncon:
            for i in range(data.ncon):
                con = data.contact[i]
                pair = {int(con.geom1), int(con.geom2)}
                if pair & bottle_ids:
                    hand_gid = int(con.geom2 if int(con.geom1) in bottle_ids else con.geom1)
                    hand_body = int(model.geom_bodyid[hand_gid])
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, hand_body) or ""
                    if body_name.startswith(("R_", "L_")) and float(con.dist) <= 0.0:
                        return {"passed": False, "reason": "active_open_path_contact", "sample": sample,
                                "distance_m": float(con.dist),
                                "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_gid),
                                "bottle_geom": mujoco.mj_id2name(
                                    model, mujoco.mjtObj.mjOBJ_GEOM,
                                    int(con.geom1 if int(con.geom1) in bottle_ids else con.geom2))}
        for rgid in robot_ids:
            for bgid in bottle_ids:
                if not collide_pair(model, rgid, bgid):
                    continue
                d, _, _ = geom_distance(model, data, bgid, rgid)
                if d < best["distance_m"]:
                    best = {"distance_m": d,
                            "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, rgid),
                            "bottle_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bgid),
                            "sample": sample}
    return {**best, "passed": best["distance_m"] >= MIN_OPEN_CLEARANCE_M}


def collect_contacts(model, data, com) -> tuple[list[ContactWrench], list[dict], list[dict]]:
    bottle_ids = bottle_body_ids(model)
    contacts: list[ContactWrench] = []
    rows = []
    unintended = []
    for index in range(data.ncon):
        con = data.contact[index]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in bottle_ids) == (g2 in bottle_ids) or float(con.dist) > 0.0:
            continue
        bottle_gid, hand_gid = (g1, g2) if g1 in bottle_ids else (g2, g1)
        body_id = int(model.geom_bodyid[hand_gid])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body.startswith("R_"):
            continue
        sign = 1.0 if bottle_gid == g2 else -1.0
        frame = np.asarray(con.frame, dtype=float).reshape(3, 3).copy()
        point = np.asarray(con.pos, dtype=float).copy()
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_gid) or ""
        bottle_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_gid) or ""
        digit = next((name for name in ("thumb", "index", "middle", "ring", "pinky")
                      if name in body.lower()), None)
        normal = frame[0] if bottle_gid == g1 else -frame[0]
        row = {"geom": geom_name, "body": body, "digit": digit, "bottle_geom": bottle_name,
               "distance_m": float(con.dist), "point_world_m": point.tolist(),
               "frame_world_rows": frame.tolist(), "normal_bottle_to_hand_world": normal.tolist(),
               "friction": np.asarray(con.friction).tolist(), "condim": int(con.dim),
               "force_on_bottle_sign": sign}
        if digit is None:
            unintended.append(row)
            continue
        contacts.append(ContactWrench(point, frame, np.asarray(con.friction, dtype=float).copy(),
                                      int(con.dim), sign, f"{body}:{bottle_name}", body_id))
        rows.append(row)
    return contacts, rows, unintended


def joint_driver_coefficients(joint_names: list[str], mimics: dict, driver_names: list[str]) -> dict[str, dict[str, float]]:
    cache: dict[str, dict[str, float]] = {}

    def resolve(name: str, visiting=()) -> dict[str, float]:
        if name in cache:
            return cache[name]
        if name in visiting:
            raise RuntimeError(f"mimic cycle at {name}")
        rel = mimics.get(name)
        if rel is None:
            result = {name: 1.0}
        else:
            result = {key: float(rel["multiplier"]) * val
                      for key, val in resolve(rel["parent"], (*visiting, name)).items()}
        cache[name] = result
        return result

    out = {driver: {} for driver in driver_names}
    for joint in joint_names:
        roots = resolve(joint)
        for driver in driver_names:
            coefficient = roots.get(driver, 0.0)
            if coefficient:
                out[driver][joint] = coefficient
    return out


def effort_map(model, data, contacts: list[ContactWrench], com: np.ndarray,
               mimic_map: dict) -> np.ndarray:
    ray_count = sum(2 * (c.condim - 1) for c in contacts)
    matrix = np.zeros((len(DRIVER_NAMES), ray_count))
    hand_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        for jid in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) or "").startswith("R_")
        and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    coeffs = joint_driver_coefficients(hand_joints, mimic_map, DRIVER_NAMES)
    column = 0
    for contact in contacts:
        rays = pyramidal_wrench_rays(contact, com)
        for ray_index in range(rays.shape[1]):
            wrench = rays[:, ray_index]
            force_on_bottle = wrench[:3]
            contact_torque_on_bottle = wrench[3:] - np.cross(
                contact.position_world_m - com, force_on_bottle)
            qfrc = np.zeros(model.nv)
            mujoco.mj_applyFT(model, data, -force_on_bottle, -contact_torque_on_bottle,
                              contact.position_world_m, int(contact.hand_body_id), qfrc)
            for i, driver in enumerate(DRIVER_NAMES):
                total = 0.0
                for joint, multiplier in coeffs[driver].items():
                    jid = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    dof = int(model.jnt_dofadr[jid])
                    total += multiplier * qfrc[dof]
                matrix[i, column] = total
            column += 1
    return matrix


def static_gates(model, data, x, base_qpos, pregrasp_arm, open_targets, mimic_map,
                 target_height, finger_geoms, bottle_body_gid, bottle_body_id,
                 bottle_mass, com, family_id) -> dict:
    state_for_x(model, data, base_qpos, x, mimic_map)
    pose_rows = []
    for digit, geom_name in finger_geoms.items():
        gid = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        dist, p_bottle, p_hand = geom_distance(model, data, bottle_body_gid, gid)
        pose_rows.append({"digit": digit, "hand_geom": geom_name, "distance_m": dist,
                          "bottle_point_world_m": p_bottle.tolist(),
                          "hand_point_world_m": p_hand.tolist(),
                          "contact_height_m": float(p_bottle[2]),
                          "radial_normal_world": radial_normal(model, data, bottle_body_gid, p_bottle).tolist()})
    contacts, contact_rows, unintended_contacts = collect_contacts(model, data, com)
    penetration = max([max(0.0, -c["distance_m"]) for c in contact_rows] or [0.0])
    digits_in_contact = sorted({row["digit"] for row in contact_rows})
    thumb_rows = [r for r in contact_rows if r["digit"] == "thumb"]
    opposing_rows = [r for r in contact_rows if r["digit"] != "thumb"]
    dots = [float(np.dot(np.asarray(t["normal_bottle_to_hand_world"]),
                         np.asarray(o["normal_bottle_to_hand_world"])))
            for t in thumb_rows for o in opposing_rows]
    best_dot = min(dots, default=None)
    path = open_path_clearance(model, data, base_qpos, x, pregrasp_arm, open_targets, mimic_map)
    state_for_x(model, data, base_qpos, x, mimic_map)
    gravity_wrench = np.r_[-bottle_mass * np.asarray(model.opt.gravity, dtype=float), np.zeros(3)]
    full = solve_static_wrench(contacts, com, gravity_wrench, include_moments=True)
    effort = effort_map(model, data, contacts, com, mimic_map) if contacts else None
    bounded = solve_static_wrench(
        contacts, com, gravity_wrench,
        include_moments=True, effort_map_nm_per_ray=effort,
        effort_limit_nm=EFFORT_CAP_NM if effort is not None else None,
    ) if contacts else {"feasible": False, "reason": "no_active_contacts"}
    if bounded.get("feasible") and effort is not None:
        allocated_effort = effort @ np.asarray(bounded["ray_coefficients_n"], dtype=float)
        bounded["estimated_driver_effort_nm"] = {
            name: float(value) for name, value in zip(DRIVER_NAMES, allocated_effort)
        }
        bounded["maximum_absolute_driver_effort_nm"] = float(np.max(np.abs(allocated_effort)))
        bounded["effort_cap_nm"] = EFFORT_CAP_NM
        bounded["effort_cap_pass"] = bool(np.all(np.abs(allocated_effort) <= EFFORT_CAP_NM + 1e-9))
    robustness = []
    if full.get("feasible"):
        for contact_index, contact in enumerate(contacts):
            frame = np.asarray(contact.frame_world_rows)
            for tangent_axis in (1, 2):
                direction = frame[tangent_axis]
                for sign in (-1.0, 1.0):
                    shifted = list(contacts)
                    shifted[contact_index] = ContactWrench(
                        contact.position_world_m + sign * 0.001 * direction,
                        contact.frame_world_rows, contact.friction, contact.condim,
                        contact.force_on_bottle_sign, contact.name, contact.hand_body_id,
                    )
                    shifted_effort = effort_map(model, data, shifted, com, mimic_map)
                    check = solve_static_wrench(
                        shifted, com, gravity_wrench,
                        include_moments=True, effort_map_nm_per_ray=shifted_effort,
                        effort_limit_nm=EFFORT_CAP_NM,
                    )
                    robustness.append({"contact": contact.name, "tangent_axis": tangent_axis,
                                       "sign": sign, "offset_m": 0.001,
                                       "feasible_with_effort_cap": check.get("feasible", False),
                                       "solver_status": check.get("solver_status")})
    right_joint_violations = []
    for jid in range(model.njnt):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) or ""
        if not body_name.startswith("R_") or not bool(model.jnt_limited[jid]):
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        lo, hi = map(float, model.jnt_range[jid])
        violation = max(lo - q, q - hi, 0.0)
        if violation > 1e-8:
            right_joint_violations.append({"joint": joint_name, "q_rad": q,
                                           "range_rad": [lo, hi], "violation_rad": violation})
    # Convert the existing driver path into an executable rate plan; no qvel is written.
    planned_duration_s = max(
        [abs(float(x[7 + i]) - float(open_targets[name])) / 0.5
         for i, name in enumerate(DRIVER_NAMES)] + [0.0]
    )
    velocity_plan = {name: {"delta_rad": float(x[7 + i]) - float(open_targets[name]),
                            "minimum_duration_at_0p5_rad_s": abs(float(x[7 + i]) - float(open_targets[name])) / 0.5,
                            "planned_peak_rad_s": (abs(float(x[7 + i]) - float(open_targets[name])) /
                                                   max(planned_duration_s, 1e-12))}
                     for i, name in enumerate(DRIVER_NAMES)}
    return {
        "candidate_id": family_id,
        "target_contact_height_m": target_height,
        "candidate_joint_values_rad": dict(zip(ARM_NAMES + DRIVER_NAMES, map(float, x))),
        "source_position_limits_pass": all(
            float(model.jnt_range[name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n), 0]) - 1e-8 <= v <=
            float(model.jnt_range[name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n), 1]) + 1e-8
            for n, v in zip(ARM_NAMES + DRIVER_NAMES, x)
        ),
        "right_hand_follower_limits_pass": not right_joint_violations,
        "right_hand_follower_limit_violations": right_joint_violations,
        "source_velocity_plan_pass": all(v["planned_peak_rad_s"] <= 0.5 + 1e-9
                                          for v in velocity_plan.values()),
        "source_velocity_plan": velocity_plan,
        "planned_close_duration_s_at_source_limit": planned_duration_s,
        "digit_surface_targets": pose_rows,
        "active_right_hand_bottle_contacts": contact_rows,
        "active_non_digit_right_hand_bottle_contacts": unintended_contacts,
        "contact_digits": digits_in_contact,
        "maximum_contact_penetration_m": penetration,
        "best_thumb_opposing_contact_normal_dot": best_dot,
        "open_approach": path,
        "full_6d_equilibrium": full,
        "effort_bounded_6d_equilibrium": bounded,
        "contact_point_perturbation_checks": robustness,
        "robust_perturbations_pass": bool(robustness) and all(
            r["feasible_with_effort_cap"] for r in robustness),
        "physics_steps": 0,
        "mj_step_calls": 0,
        "mj_forward_calls": FORWARD_CALLS,
    }


def main() -> int:
    global FORWARD_CALLS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-candidate", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path)
    parser.add_argument("--max-nfev", type=int, default=240)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    seed = json.loads(args.seed_candidate.read_text(encoding="utf-8"))
    base = json.loads(args.base_state.read_text(encoding="utf-8"))
    hand.URDF = args.urdf
    _, joint_limits, mimics, _ = hand.parse_urdf()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    base_state = __import__("audit_g1_dfq_candidate_wrench").load_candidate_state(
        model, seed, base, args.urdf
    )
    FORWARD_CALLS += 1
    base_qpos = base_state.qpos.copy()
    data.qpos[:] = base_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    FORWARD_CALLS += 1
    bottle_free = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    bottle_body = int(model.jnt_bodyid[bottle_free])
    mass = float(model.body_subtreemass[bottle_body])
    com = np.asarray(data.subtree_com[bottle_body], dtype=float).copy()
    body_gid = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    open_targets = seed["open_hand_approach"]["open_driver_targets_rad"]
    pregrasp_arm = np.asarray(seed["open_hand_approach"]["pregrasp_arm_q_rad"], dtype=float)
    x0 = np.asarray([seed["joint_configuration"][name]["q_rad"]
                     for name in ARM_NAMES + DRIVER_NAMES], dtype=float)
    lower = np.asarray([model.jnt_range[name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n), 0]
                        for n in ARM_NAMES + DRIVER_NAMES], dtype=float)
    upper = np.asarray([model.jnt_range[name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n), 1]
                        for n in ARM_NAMES + DRIVER_NAMES], dtype=float)
    # Contact heights come from the actual body span and the previously observed C3 vertical mismatch.
    families = [
        {"id": "C4_com_height_body_opposition", "height": float(com[2]), "digits": ("thumb", "index", "middle")},
        {"id": "C5_upper_body_opposition", "height": 0.93, "digits": ("thumb", "index", "middle")},
        {"id": "C6_lower_body_opposition", "height": 0.84, "digits": ("thumb", "middle", "ring")},
    ]
    prior = None
    if args.seed_result:
        prior_payload = json.loads(args.seed_result.read_text(encoding="utf-8"))
        prior = {row["candidate_id"]: row["candidate_joint_values_rad"]
                 for row in prior_payload["candidates"]}
    right_hand_collision_geoms = []
    for gid in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if body_name.startswith("R_") and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
            right_hand_collision_geoms.append((gid, body_name,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""))
    bottle_gids = sorted(bottle_body_ids(model))
    seed_scales = np.asarray([0.45] * 7 + [0.45] * 6, dtype=float)
    results = []
    for family in families:
        digit_geoms = {d: CONTACT_GEOM[d] for d in family["digits"]}
        geom_ids = {d: name_id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for d, g in digit_geoms.items()}
        family_seed = x0.copy()
        if prior and family["id"] in prior:
            family_seed = np.asarray([prior[family["id"]][name]
                                      for name in ARM_NAMES + DRIVER_NAMES], dtype=float)

        def residual(x):
            state_for_x(model, data, base_qpos, x, mimics)
            records = []
            for digit, gid in geom_ids.items():
                distance, point_bottle, point_hand = geom_distance(model, data, body_gid, gid)
                normal = radial_normal(model, data, body_gid, point_bottle)
                records.append((digit, distance, point_bottle, point_hand, normal))
            values = [1000.0 * (row[1] - CONTACT_TARGET_M) for row in records]
            values += [150.0 * (float(row[2][2]) - float(family["height"])) for row in records]
            values += [150.0 * (float(row[3][2]) - float(family["height"])) for row in records]
            thumb_normal = next(row[4] for row in records if row[0] == "thumb")
            values += [2.0 * max(0.0, float(np.dot(thumb_normal, row[4])) + 0.35)
                       for row in records if row[0] != "thumb"]
            selected_ids = set(geom_ids.values())
            for hand_gid, body_name, hand_name in right_hand_collision_geoms:
                for bottle_gid in bottle_gids:
                    if not collide_pair(model, hand_gid, bottle_gid):
                        continue
                    if hand_gid in selected_ids and bottle_gid == body_gid:
                        continue
                    distance, _, _ = geom_distance(model, data, bottle_gid, hand_gid)
                    clearance = MIN_OPEN_CLEARANCE_M if "hand_base" in body_name else -MAX_PENETRATION_M
                    values.append(COLLISION_PENALTY_WEIGHT * max(0.0, clearance - distance))
            values += (0.10 * (x - family_seed) / seed_scales).tolist()
            return np.asarray(values, dtype=float)

        optimized = least_squares(residual, np.clip(family_seed, lower, upper), bounds=(lower, upper),
                                  max_nfev=args.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                  diff_step=2e-5)
        row = static_gates(model, data, optimized.x, base_qpos, pregrasp_arm,
                           open_targets, mimics, family["height"], digit_geoms,
                           body_gid, bottle_body, mass, com, family["id"])
        row["optimizer"] = {"success": bool(optimized.success), "status": int(optimized.status),
                            "message": str(optimized.message), "nfev": int(optimized.nfev),
                            "cost": float(optimized.cost), "optimality": float(optimized.optimality)}
        row["geometry_family"] = family
        row["static_candidate_pass"] = bool(
            optimized.success and row["source_position_limits_pass"]
            and row["right_hand_follower_limits_pass"]
            and row["source_velocity_plan_pass"]
            and len(row["contact_digits"]) >= 2
            and "thumb" in row["contact_digits"]
            and any(d in row["contact_digits"] for d in ("index", "middle", "ring", "pinky"))
            and row["maximum_contact_penetration_m"] <= MAX_PENETRATION_M + PENETRATION_GATE_TOLERANCE_M
            and row["best_thumb_opposing_contact_normal_dot"] is not None
            and row["best_thumb_opposing_contact_normal_dot"] <= -0.1
            and not row["active_non_digit_right_hand_bottle_contacts"]
            and row["open_approach"].get("passed", False)
            and row["full_6d_equilibrium"].get("feasible", False)
            and row["effort_bounded_6d_equilibrium"].get("feasible", False)
            and row["effort_bounded_6d_equilibrium"].get("effort_cap_pass", False)
            and row["robust_perturbations_pass"]
        )
        results.append(row)

    payload = {
        "task": "Issue #43 bounded DFQ contact geometry search",
        "robotsim_head": subprocess.check_output(["git", "-C", str(HERE.parents[1]), "rev-parse", "HEAD"], text=True).strip(),
        "robotsim_dirty": bool(subprocess.check_output(["git", "-C", str(HERE.parents[1]), "status", "--porcelain"], text=True).strip()),
        "mujoco_version": mujoco.__version__,
        "model_sha256": sha256(args.model),
        "seed_candidate_sha256": sha256(args.seed_candidate),
        "seed_result_sha256": sha256(args.seed_result) if args.seed_result else None,
        "base_state_sha256": sha256(args.base_state),
        "urdf_sha256": sha256(args.urdf),
        "bottle_mass_kg": mass,
        "bottle_com_world_m": com.tolist(),
        "bottle_weight_n": mass * 9.81,
        "source_velocity_limit_rad_s": 0.5,
        "simulation_only_effort_cap_nm": EFFORT_CAP_NM,
        "max_contact_penetration_gate_m": MAX_PENETRATION_M,
        "penetration_gate_evaluation_tolerance_m": PENETRATION_GATE_TOLERANCE_M,
        "open_path_clearance_gate_m": MIN_OPEN_CLEARANCE_M,
        "optimizer_collision_penalty_weight": COLLISION_PENALTY_WEIGHT,
        "physics_steps": 0,
        "mj_step_calls": 0,
        "families_evaluated": len(results),
        "candidates": results,
        "selected_candidate": next((r["candidate_id"] for r in results if r["static_candidate_pass"]), None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_candidate": payload["selected_candidate"],
                      "families": [{"id": r["candidate_id"], "contacts": r["contact_digits"],
                                    "full_6d": r["full_6d_equilibrium"].get("feasible"),
                                    "effort": r["effort_bounded_6d_equilibrium"].get("feasible"),
                                    "robust": r["robust_perturbations_pass"],
                                    "passed": r["static_candidate_pass"]} for r in results],
                      "physics_steps": 0, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
