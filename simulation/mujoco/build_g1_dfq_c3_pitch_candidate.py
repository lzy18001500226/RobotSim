#!/usr/bin/env python3
"""Build one zero-step, evidence-derived DFQ C3 wrist-pitch candidate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(os.environ["ROBOTSIM_REPO_ROOT"])
OUT = Path(os.environ["ROBOTSIM_G1_DFQ_C3_DIR"])
FIXTURE = REPO / "simulation/mujoco/fixtures/g1_dfq_m0_c2"
sys.path.insert(0, str(REPO / "simulation/mujoco"))

import run_g1_dfq_physical_grasp as physical  # noqa: E402


core = physical.runner
hand = physical.hand
ARM_NAMES = list(core.ARM_NAMES)
PITCH_DEG = core.WORLD_PITCH_TEST_DEG
PATH_SAMPLES = 401
MIN_OPEN_CLEARANCE_M = 0.0005
MAX_CONTACT_PENETRATION_M = 0.0005
STATIC_CONTACT_ENGAGEMENT_M = 0.0001
MIN_OPPOSED_NORMAL_DOT = 0.0
CONTACT_ROOT_SCAN_SAMPLES = 65
CONTACT_ROOT_BISECTION_STEPS = 28
BOTTLE_GEOMS = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}
DIGIT_GEOMS = {
    "thumb": "dfq_collision_R_thumb_distal_0",
    "index": "dfq_collision_R_index_intermediate_0",
    "middle": "dfq_collision_R_middle_intermediate_0",
    "ring": "dfq_collision_R_ring_intermediate_0",
    "pinky": "dfq_collision_R_pinky_intermediate_0",
}


def element(model, kind, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"missing MuJoCo element: {name}")
    return value


def set_joint(model, data, name: str, value: float) -> None:
    jid = element(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(np.cross(current[:, axis], target[:, axis]) for axis in range(3))


def solve_arm_pose(model, data, site_id: int, target: np.ndarray,
                   rotation: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, dict]:
    qadr = core.arm_qpos_addresses(model)
    joint_ids = np.asarray([element(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES])
    dofs = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    data.qpos[qadr] = np.clip(seed, lower, upper)
    converged = False
    for iteration in range(800):
        mujoco.mj_forward(model, data)
        pos_error = target - data.site_xpos[site_id]
        current = data.site_xmat[site_id].reshape(3, 3)
        rot_error = rotation_error(current, rotation)
        if np.linalg.norm(pos_error) <= 0.0015 and np.linalg.norm(rot_error) <= 0.03:
            converged = True
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        weight, damping = 0.25, 0.04
        jac = np.vstack((jacp[:, dofs], jacr[:, dofs] * weight))
        err = np.r_[pos_error, rot_error * weight]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        norm = float(np.linalg.norm(dq))
        if norm > 0.015:
            dq *= 0.015 / norm
        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lower, upper)
    mujoco.mj_forward(model, data)
    achieved = data.site_xpos[site_id].copy()
    achieved_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    q = data.qpos[qadr].copy()
    return q, {
        "converged": converged,
        "iterations": iteration + 1,
        "requested_site_position_world_m": target.tolist(),
        "achieved_site_position_world_m": achieved.tolist(),
        "position_error_m": float(np.linalg.norm(target - achieved)),
        "requested_site_rotation_world": rotation.tolist(),
        "achieved_site_rotation_world": achieved_rot.tolist(),
        "rotation_error_rad": float(np.linalg.norm(rotation_error(achieved_rot, rotation))),
        "arm_q_rad": q.tolist(),
    }


def active_robot_bottle_contacts(model, data) -> list[dict]:
    rows = []
    for cid in range(data.ncon):
        contact = data.contact[cid]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if (n1 in BOTTLE_GEOMS) == (n2 in BOTTLE_GEOMS):
            continue
        bottle_geom, hand_geom = (g1, g2) if n1 in BOTTLE_GEOMS else (g2, g1)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[hand_geom])) or ""
        if not body.startswith("R_"):
            continue
        rows.append({
            "body": body,
            "hand_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom),
            "bottle_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom),
            "distance_m": float(contact.dist),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_frame_world": np.asarray(contact.frame, dtype=float).reshape(3, 3).tolist(),
        })
    return rows


def surface_distance(model, data, bottle_geom: int, hand_geom: int) -> dict:
    fromto = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, bottle_geom, hand_geom, 0.25, fromto))
    delta = fromto[3:] - fromto[:3]
    return {
        "distance_m": distance,
        "point_bottle_world_m": fromto[:3].tolist(),
        "point_hand_world_m": fromto[3:].tolist(),
        "normal_bottle_to_hand_world": (delta / max(np.linalg.norm(delta), 1e-12)).tolist(),
    }


def robot_geom_ids(model) -> list[int]:
    pelvis = element(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    result = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body > 0 and body != pelvis:
            body = int(model.body_parentid[body])
        if body == pelvis:
            result.append(gid)
    return result


def collision_pair(model, first: int, second: int) -> bool:
    return bool(
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    )


def contact_normal_bottle_to_hand(model, contact) -> np.ndarray:
    geom1, geom2 = int(contact.geom1), int(contact.geom2)
    name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
    name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
    normal = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
    return normal if name1 in BOTTLE_GEOMS else -normal


def set_arm_and_hand(model, data, arm_q, drivers, open_targets, mimics, *, closed: bool) -> None:
    data.qpos[core.arm_qpos_addresses(model)] = arm_q
    targets = drivers if closed else open_targets
    for name, value in targets.items():
        set_joint(model, data, name, float(value))
    physical.apply_mimics(model, data, mimics)
    mujoco.mj_forward(model, data)


def minimum_robot_bottle_clearance(model, data, robot_ids, bottle_ids) -> tuple[float, str | None, str | None]:
    closest = (0.25, -1, -1)
    for robot_geom in robot_ids:
        for bottle_geom in bottle_ids:
            if not collision_pair(model, robot_geom, bottle_geom):
                continue
            fromto = np.zeros(6, dtype=float)
            distance = float(mujoco.mj_geomDistance(
                model, data, robot_geom, bottle_geom, 0.25, fromto
            ))
            if distance < closest[0]:
                closest = (distance, robot_geom, bottle_geom)
    return (
        float(closest[0]),
        None if closest[1] < 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, closest[1]),
        None if closest[2] < 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, closest[2]),
    )


def contact_topology(model, data) -> dict:
    by_digit = {digit: [] for digit in DIGIT_GEOMS}
    non_digit = []
    for cid in range(data.ncon):
        contact = data.contact[cid]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        if (name1 in BOTTLE_GEOMS) == (name2 in BOTTLE_GEOMS):
            continue
        hand_geom = name2 if name1 in BOTTLE_GEOMS else name1
        body_id = int(model.geom_bodyid[int(contact.geom2 if name1 in BOTTLE_GEOMS else contact.geom1)])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body.startswith("R_"):
            continue
        digit = next((key for key in DIGIT_GEOMS if key in body.lower()), None)
        normal = contact_normal_bottle_to_hand(model, contact)
        row = {
            "body": body,
            "hand_geom": hand_geom,
            "bottle_geom": name1 if name1 in BOTTLE_GEOMS else name2,
            "distance_m": float(contact.dist),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "normal_bottle_to_hand_world": normal.tolist(),
        }
        if digit is None:
            non_digit.append(row)
        else:
            by_digit[digit].append(row)
    return {"by_digit": by_digit, "non_digit": non_digit}


def minimum_digit_surface_distance(model, data, hand_geom: int, bottle_ids: list[int]) -> tuple[float, int]:
    best_distance, best_geom = float("inf"), -1
    for bottle_geom in bottle_ids:
        distance = float(mujoco.mj_geomDistance(model, data, bottle_geom, hand_geom, 0.25, np.zeros(6)))
        if distance < best_distance:
            best_distance, best_geom = distance, bottle_geom
    return best_distance, best_geom


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compiled").mkdir(exist_ok=True)
    source_candidate = json.loads((FIXTURE / "candidate_c2.json").read_text(encoding="utf-8"))
    base = json.loads((FIXTURE / "canonical_ready_wrench_input.json").read_text(encoding="utf-8"))
    hand.URDF = core.URDF
    _, limits, mimics, _ = hand.parse_urdf()
    ranges = hand.derive_channel_ranges(limits, mimics)
    runtime_xml = physical.prepare_runtime_robot_xml(OUT)
    compiled = OUT / "compiled/g1_dfq_c3_pitch15_static_mj336.xml"
    build = core.build_scene(compiled, ranges, limits, mimics)
    model = build["model"]
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    for name, value in base["joint_state_qpos_rad"].items():
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid >= 0 and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            set_joint(model, data, name, float(value))
    for name in ARM_NAMES:
        set_joint(model, data, name, source_candidate["joint_configuration"][name]["q_rad"])
    close_drivers = {
        channel: float(source_candidate["joint_configuration"][hand.channel_joint("R", suffix)]["q_rad"])
        for channel, suffix in hand.CHANNELS
    }
    for channel, suffix in hand.CHANNELS:
        set_joint(model, data, hand.channel_joint("R", suffix), close_drivers[channel])
    physical.apply_mimics(model, data, mimics)
    data.qpos[core.bottle_qpos_addresses(model)] = np.r_[
        base["bottle_root_position_world_m"], base["bottle_root_quaternion_wxyz"]
    ]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    site_id = element(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    arm_qadr = core.arm_qpos_addresses(model)

    c2_contact_arm = np.asarray(source_candidate["open_hand_approach"]["optimized_contact_arm_q_rad"])
    data.qpos[arm_qadr] = c2_contact_arm
    mujoco.mj_forward(model, data)
    contact_pos = data.site_xpos[site_id].copy()
    contact_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    c3_contact_arm, contact_ik = solve_arm_pose(
        model, data, site_id, contact_pos,
        core.world_pitch_test_rotation(contact_rot, PITCH_DEG), c2_contact_arm
    )

    open_targets = source_candidate["open_hand_approach"]["open_driver_targets_rad"]
    pregrasp_seed = np.asarray(source_candidate["open_hand_approach"]["pregrasp_arm_q_rad"])
    data.qpos[arm_qadr] = pregrasp_seed
    for name, value in open_targets.items():
        set_joint(model, data, name, float(value))
    physical.apply_mimics(model, data, mimics)
    mujoco.mj_forward(model, data)
    pregrasp_pos = np.asarray(source_candidate["open_hand_approach"]["pregrasp_site_target_world_m"])
    pregrasp_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    c3_pregrasp_arm, pregrasp_ik = solve_arm_pose(
        model, data, site_id, pregrasp_pos,
        core.world_pitch_test_rotation(pregrasp_rot, PITCH_DEG), pregrasp_seed
    )

    candidate = json.loads(json.dumps(source_candidate))
    candidate["candidate"] = "C3_world_pitch15_first_bilateral_contact"
    candidate["run_id"] = f"dfq-c3-world-pitch15-static-{time.time_ns()}"
    candidate["physics_steps"] = 0
    candidate["mj_step_calls"] = 0
    candidate["source"].update({
        "robotsim_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "robotsim_dirty": bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()),
        "compiled_model_sha256": physical.normalized_model_sha256(compiled),
        "candidate_correction": {
            "type": "single evidence-derived world-Y wrist pitch rotation with source-valid per-digit contact preshape",
            "degrees": PITCH_DEG,
            "evidence": "measured HOLD thumb/index contact separation was +67.8 mm in X and +23.7 mm in Z; world-Y pitch is derived from that offset, while finger targets are subsequently localized to first contact along their existing source-valid close paths",
            "preserved": ["hand geometry", "bottle/table", "friction", "actuator parameters", "source limits", "mimic topology", "effort cap"],
        },
    })
    bottle_ids = [element(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in sorted(BOTTLE_GEOMS)]
    robot_ids = robot_geom_ids(model)
    driver_joint_names = {hand.channel_joint("R", suffix): float(close_drivers[channel])
                          for channel, suffix in hand.CHANNELS}
    clearance_samples = []
    closed_probes = []
    selected = None
    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, PATH_SAMPLES)):
        arm_q = (1.0 - alpha) * c3_pregrasp_arm + alpha * c3_contact_arm
        set_arm_and_hand(model, data, arm_q, driver_joint_names, open_targets, mimics, closed=False)
        open_clearance, open_geom, open_bottle = minimum_robot_bottle_clearance(
            model, data, robot_ids, bottle_ids
        )
        open_contacts = active_robot_bottle_contacts(model, data)
        row = {
            "sample": sample_index,
            "alpha": float(alpha),
            "open_minimum_collision_distance_m": open_clearance,
            "open_minimum_robot_geom": open_geom,
            "open_minimum_bottle_geom": open_bottle,
            "open_active_robot_bottle_contacts": open_contacts,
        }
        if open_clearance > MIN_OPEN_CLEARANCE_M and not open_contacts:
            set_arm_and_hand(model, data, arm_q, driver_joint_names, open_targets, mimics, closed=True)
            topology = contact_topology(model, data)
            all_digit_contacts = [item for contacts in topology["by_digit"].values() for item in contacts]
            shallow_contacts = [item for item in all_digit_contacts
                                if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
            thumb_contacts = [item for item in topology["by_digit"]["thumb"]
                              if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
            opposing = [(digit, item) for digit in ("index", "middle", "ring", "pinky")
                        for item in topology["by_digit"][digit]
                        if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
            normal_pairs = [(digit, thumb, item, float(np.dot(
                np.asarray(thumb["normal_bottle_to_hand_world"]),
                np.asarray(item["normal_bottle_to_hand_world"])
            ))) for thumb in thumb_contacts for digit, item in opposing]
            best_pair = min(normal_pairs, key=lambda pair: pair[3], default=None)
            if all_digit_contacts or topology["non_digit"]:
                digit_min_distances = {
                    digit: min((float(item["distance_m"]) for item in contacts), default=None)
                    for digit, contacts in topology["by_digit"].items()
                }
                closed_probes.append({
                    "sample": sample_index,
                    "alpha": float(alpha),
                    "open_clearance_m": float(open_clearance),
                    "digit_min_contact_distances_m": digit_min_distances,
                    "thumb_contact_count": len(topology["by_digit"]["thumb"]),
                    "opposing_digit_contact_counts": {
                        digit: len(topology["by_digit"][digit])
                        for digit in ("index", "middle", "ring", "pinky")
                    },
                    "maximum_digit_penetration_m": max(
                        [max(0.0, -float(item["distance_m"])) for item in all_digit_contacts] or [0.0]
                    ),
                    "best_thumb_opposing_normal_dot": None if best_pair is None else best_pair[3],
                    "best_opposing_digit": None if best_pair is None else best_pair[0],
                    "non_digit_contact_bodies": [item["body"] for item in topology["non_digit"]],
                    "contacts": all_digit_contacts + topology["non_digit"],
                })
            if (thumb_contacts and opposing and best_pair is not None
                    and best_pair[3] <= MIN_OPPOSED_NORMAL_DOT
                    and len(shallow_contacts) == len(all_digit_contacts)
                    and not topology["non_digit"]):
                selected = {
                    "sample": sample_index,
                    "alpha": float(alpha),
                    "arm_q_rad": arm_q.copy(),
                    "open_clearance_m": open_clearance,
                    "open_minimum_robot_geom": open_geom,
                    "open_minimum_bottle_geom": open_bottle,
                    "contacts": topology,
                    "shallow_contact_count": len(shallow_contacts),
                    "best_opposed_pair": {
                        "thumb_contact": best_pair[1],
                        "opposing_contact": best_pair[2],
                        "opposing_digit": best_pair[0],
                        "normal_dot": best_pair[3],
                    },
                }
                row["selected_contact_pose"] = True
                clearance_samples.append(row)
                break
        clearance_samples.append(row)

    contact_preshape = None
    if selected is None:
        safe_rows = [row for row in clearance_samples
                     if row["open_minimum_collision_distance_m"] > MIN_OPEN_CLEARANCE_M
                     and not row["open_active_robot_bottle_contacts"]]
        if safe_rows:
            safe_row = max(safe_rows, key=lambda row: row["alpha"])
            safe_alpha = float(safe_row["alpha"])
            clearance_samples = [row for row in clearance_samples
                                 if row["alpha"] <= safe_alpha + 1e-12]
            safe_arm_q = (1.0 - safe_alpha) * c3_pregrasp_arm + safe_alpha * c3_contact_arm
            digit_channel_names = {
                "thumb": [hand.channel_joint("R", "thumb_pitch"), hand.channel_joint("R", "thumb_yaw")],
                "index": [hand.channel_joint("R", "index_proximal")],
                "middle": [hand.channel_joint("R", "middle_proximal")],
                "ring": [hand.channel_joint("R", "ring_proximal")],
                "pinky": [hand.channel_joint("R", "pinky_proximal")],
            }
            digit_contact_targets = dict(open_targets)
            contact_root_records = {}
            for digit, joint_names in digit_channel_names.items():
                hand_geom = element(model, mujoco.mjtObj.mjOBJ_GEOM, DIGIT_GEOMS[digit])
                q_open = np.asarray([float(open_targets[name]) for name in joint_names])
                q_close = np.asarray([float(driver_joint_names[name]) for name in joint_names])
                fractions = np.linspace(0.0, 1.0, CONTACT_ROOT_SCAN_SAMPLES)
                distances = []
                for fraction in fractions:
                    probe_targets = dict(open_targets)
                    values = (1.0 - fraction) * q_open + fraction * q_close
                    probe_targets.update({name: float(value) for name, value in zip(joint_names, values)})
                    data.qpos[arm_qadr] = safe_arm_q
                    for name, value in probe_targets.items():
                        set_joint(model, data, name, value)
                    physical.apply_mimics(model, data, mimics)
                    mujoco.mj_forward(model, data)
                    distance, _ = minimum_digit_surface_distance(model, data, hand_geom, bottle_ids)
                    distances.append(distance)
                crossing = next((index for index, distance in enumerate(distances)
                                 if distance <= 0.0), None)
                if crossing is None:
                    contact_root_records[digit] = {
                        "reachable_contact": False,
                        "distance_at_open_m": float(distances[0]),
                        "distance_at_source_close_m": float(distances[-1]),
                        "joint_names": joint_names,
                    }
                    continue
                lo = float(fractions[max(0, crossing - 1)])
                hi = float(fractions[crossing])
                for _ in range(CONTACT_ROOT_BISECTION_STEPS):
                    mid = 0.5 * (lo + hi)
                    probe_targets = dict(open_targets)
                    values = (1.0 - mid) * q_open + mid * q_close
                    probe_targets.update({name: float(value) for name, value in zip(joint_names, values)})
                    data.qpos[arm_qadr] = safe_arm_q
                    for name, value in probe_targets.items():
                        set_joint(model, data, name, value)
                    physical.apply_mimics(model, data, mimics)
                    mujoco.mj_forward(model, data)
                    distance, _ = minimum_digit_surface_distance(model, data, hand_geom, bottle_ids)
                    if distance <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                root_fraction = hi
                engagement_crossing = next((index for index, distance in enumerate(distances)
                                            if distance <= -STATIC_CONTACT_ENGAGEMENT_M), None)
                if engagement_crossing is None:
                    contact_root_records[digit] = {
                        "reachable_contact": True,
                        "source_close_reaches_engagement_depth": False,
                        "first_contact_fraction_open_to_source_close": root_fraction,
                        "distance_at_source_close_m": float(distances[-1]),
                        "joint_names": joint_names,
                    }
                    continue
                engagement_lo = float(fractions[max(0, engagement_crossing - 1)])
                engagement_hi = float(fractions[engagement_crossing])
                for _ in range(CONTACT_ROOT_BISECTION_STEPS):
                    mid = 0.5 * (engagement_lo + engagement_hi)
                    probe_targets = dict(open_targets)
                    values = (1.0 - mid) * q_open + mid * q_close
                    probe_targets.update({name: float(value) for name, value in zip(joint_names, values)})
                    data.qpos[arm_qadr] = safe_arm_q
                    for name, value in probe_targets.items():
                        set_joint(model, data, name, value)
                    physical.apply_mimics(model, data, mimics)
                    mujoco.mj_forward(model, data)
                    distance, _ = minimum_digit_surface_distance(model, data, hand_geom, bottle_ids)
                    if distance <= -STATIC_CONTACT_ENGAGEMENT_M:
                        engagement_hi = mid
                    else:
                        engagement_lo = mid
                engagement_fraction = engagement_hi
                root_values = (1.0 - engagement_fraction) * q_open + engagement_fraction * q_close
                digit_contact_targets.update({name: float(value) for name, value in zip(joint_names, root_values)})
                contact_root_records[digit] = {
                    "reachable_contact": True,
                    "first_contact_fraction_open_to_source_close": root_fraction,
                    "static_engagement_fraction_open_to_source_close": engagement_fraction,
                    "static_engagement_depth_target_m": STATIC_CONTACT_ENGAGEMENT_M,
                    "joint_names": joint_names,
                    "joint_open_rad": q_open.tolist(),
                    "joint_source_close_rad": q_close.tolist(),
                    "joint_contact_target_rad": root_values.tolist(),
                    "source_range_respected": all(
                        float(model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0]) - 1e-9
                        <= float(value)
                        <= float(model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name), 1]) + 1e-9
                        for name, value in zip(joint_names, root_values)
                    ),
                }

            data.qpos[arm_qadr] = safe_arm_q
            for name, value in digit_contact_targets.items():
                set_joint(model, data, name, float(value))
            physical.apply_mimics(model, data, mimics)
            mujoco.mj_forward(model, data)
            topology = contact_topology(model, data)
            digit_rows = [item for rows in topology["by_digit"].values() for item in rows]
            thumb_contacts = [item for item in topology["by_digit"]["thumb"]
                              if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
            opposing = [(digit, item) for digit in ("index", "middle", "ring", "pinky")
                        for item in topology["by_digit"][digit]
                        if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
            normal_pairs = [(digit, thumb, item, float(np.dot(
                np.asarray(thumb["normal_bottle_to_hand_world"]),
                np.asarray(item["normal_bottle_to_hand_world"])
            ))) for thumb in thumb_contacts for digit, item in opposing]
            best_pair = min(normal_pairs, key=lambda pair: pair[3], default=None)
            final_min_distance = min(
                (float(item["distance_m"]) for item in digit_rows), default=float("inf")
            )
            final_max_penetration = max(
                [max(0.0, -float(item["distance_m"])) for item in digit_rows] or [0.0]
            )
            contact_preshape = {
                "method": "bounded per-digit first-contact root localization along existing open-to-source-close driver path",
                "open_waypoint_alpha": safe_alpha,
                "open_clearance_m": safe_row["open_minimum_collision_distance_m"],
                "root_scan_samples_per_digit": CONTACT_ROOT_SCAN_SAMPLES,
                "root_bisection_steps": CONTACT_ROOT_BISECTION_STEPS,
                "static_engagement_depth_target_m": STATIC_CONTACT_ENGAGEMENT_M,
                "digit_contact_roots": contact_root_records,
                "combined_digit_minimum_distance_m": None if not digit_rows else final_min_distance,
                "combined_digit_maximum_penetration_m": final_max_penetration,
                "combined_contact_topology": topology,
                "best_opposed_pair": None if best_pair is None else {
                    "thumb_contact": best_pair[1],
                    "opposing_contact": best_pair[2],
                    "opposing_digit": best_pair[0],
                    "normal_dot": best_pair[3],
                },
            }
            if (thumb_contacts and opposing and best_pair is not None
                    and best_pair[3] <= MIN_OPPOSED_NORMAL_DOT
                    and final_max_penetration <= MAX_CONTACT_PENETRATION_M
                    and not topology["non_digit"]):
                selected = {
                    "sample": safe_row["sample"],
                    "alpha": safe_alpha,
                    "arm_q_rad": safe_arm_q.copy(),
                    "open_clearance_m": safe_row["open_minimum_collision_distance_m"],
                    "open_minimum_robot_geom": safe_row["open_minimum_robot_geom"],
                    "open_minimum_bottle_geom": safe_row["open_minimum_bottle_geom"],
                    "contacts": topology,
                    "best_opposed_pair": contact_preshape["best_opposed_pair"],
                }
                driver_joint_names = {name: float(digit_contact_targets[name]) for name in driver_joint_names}

    if selected is None:
        candidate["static_gates"] = {"passed": False, "failure_reason": "no_clear_open_waypoint_with_shallow_opposed_digit_contact"}
        candidate["analysis"] = "zero-step first-contact corridor search found no collision-free open waypoint followed by shallow opposed thumb/digit contact; dynamics were not run"
        candidate["state_writes"] = "scratch MjData initial-state setup only; no active rollout, no runtime bottle qpos writes, no simulation steps"
        candidate["open_hand_approach"]["contact_preshape"] = contact_preshape
        path = OUT / "candidate_c3.json"
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        audit = {
            "candidate_path": str(path), "compiled_model_path": str(compiled),
            "compiled_model_sha256": physical.normalized_model_sha256(compiled),
            "physics_steps": 0, "mj_step_calls": 0, "pitch_deg": PITCH_DEG,
            "path_samples": len(clearance_samples), "minimum_open_approach_clearance_m": min(
                row["open_minimum_collision_distance_m"] for row in clearance_samples
            ),
            "closed_probe_count": len(closed_probes),
            "closed_probes": closed_probes,
            "contact_preshape": contact_preshape,
            "first_open_path_contact": next((row for row in clearance_samples
                if row["open_active_robot_bottle_contacts"] or row["open_minimum_collision_distance_m"] <= 0.0), None),
            "static_gates": candidate["static_gates"],
            "state_writes": candidate["state_writes"],
        }
        (OUT / "candidate_c3_static_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return 2

    c3_contact_arm = selected["arm_q_rad"]
    for name, value in zip(ARM_NAMES, c3_contact_arm):
        record = candidate["joint_configuration"][name]
        record["q_rad"] = float(value)
        lower, upper = map(float, model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        record.update({"lower_rad": lower, "upper_rad": upper,
                       "violation_rad": max(lower - float(value), float(value) - upper, 0.0)})
    for name, value in driver_joint_names.items():
        record = candidate["joint_configuration"][name]
        record["q_rad"] = float(value)
        lower, upper = map(float, model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        record.update({"lower_rad": lower, "upper_rad": upper,
                       "violation_rad": max(lower - float(value), float(value) - upper, 0.0)})
    candidate["open_hand_approach"].update({
        "optimized_contact_arm_q_rad": c3_contact_arm.tolist(),
        "pregrasp_arm_q_rad": c3_pregrasp_arm.tolist(),
        "pregrasp_ik": pregrasp_ik,
        "contact_pose_ik": contact_ik,
        "wrist_pitch_correction_deg": PITCH_DEG,
        "first_bilateral_contact_path_alpha": selected["alpha"],
        "open_contact_waypoint_clearance_m": selected["open_clearance_m"],
        "open_contact_waypoint_minimum_robot_geom": selected["open_minimum_robot_geom"],
        "open_contact_waypoint_minimum_bottle_geom": selected["open_minimum_bottle_geom"],
        "source_contact_pose_ik_target_position_world_m": contact_pos.tolist(),
        "source_contact_pose_ik_target_rotation_world": core.world_pitch_test_rotation(contact_rot, PITCH_DEG).tolist(),
        "selected_actual_site_position_world_m": data.site_xpos[site_id].tolist(),
        "selected_actual_site_rotation_world": data.site_xmat[site_id].reshape(3, 3).tolist(),
        "contact_driver_targets_rad": driver_joint_names,
        "contact_preshape": contact_preshape,
    })
    candidate["source"]["candidate_correction"]["finger_target_derivation"] = (
        "each channel target is the first zero-distance point on its official open-to-C2-close joint path; "
        "thumb pitch/yaw follow their existing shared closure path; no range, mimic, or actuator changes"
    )
    set_arm_and_hand(model, data, c3_contact_arm, driver_joint_names, open_targets, mimics, closed=True)
    digit_geometry = {}
    for digit, geom_name in DIGIT_GEOMS.items():
        hand_geom = element(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        surfaces = []
        for bottle_name in sorted(BOTTLE_GEOMS):
            bottle_geom = element(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_name)
            surfaces.append({"bottle_geom": bottle_name, **surface_distance(model, data, bottle_geom, hand_geom)})
        digit_geometry[digit] = {
            "hand_geom": geom_name,
            "closest_bottle_surface": min(surfaces, key=lambda row: abs(row["distance_m"])),
            "surfaces": surfaces,
        }
    close_contacts = active_robot_bottle_contacts(model, data)
    candidate["joint_contact_geometry"] = digit_geometry
    candidate["actual_contacts"] = close_contacts
    candidate["open_hand_approach"]["clearance_samples"] = clearance_samples
    candidate["open_hand_approach"]["minimum_path_collision_distance_m"] = min(
        row["open_minimum_collision_distance_m"] for row in clearance_samples
    )
    candidate["open_hand_approach"]["open_state_contacts"] = [
        c for row in clearance_samples for c in row["open_active_robot_bottle_contacts"]
    ]

    distances_by_digit = {
        digit: row["closest_bottle_surface"]["distance_m"]
        for digit, row in digit_geometry.items()
    }
    all_position_valid = all(
        rec["lower_rad"] - 1e-9 <= rec["q_rad"] <= rec["upper_rad"] + 1e-9
        for rec in candidate["joint_configuration"].values()
    )
    urdf_root = ET.parse(core.URDF).getroot()
    arm_velocity_limits = {}
    for joint in urdf_root.findall("joint"):
        if joint.get("name") in ARM_NAMES:
            arm_velocity_limits[joint.get("name")] = float(joint.find("limit").get("velocity"))
    arm_delta = {name: abs(float(q) - float(p)) for name, q, p in zip(ARM_NAMES, c3_contact_arm, c3_pregrasp_arm)}
    arm_rates = {name: value / core.APPROACH_SECONDS for name, value in arm_delta.items()}
    arm_rate_pass = all(arm_rates[name] <= arm_velocity_limits[name] for name in ARM_NAMES)
    min_clearance = min(row["open_minimum_collision_distance_m"] for row in clearance_samples)
    no_open_contacts = all(not row["open_active_robot_bottle_contacts"] for row in clearance_samples)
    selected_topology = selected["contacts"]["by_digit"]
    selected_thumb_contacts = [item for item in selected_topology["thumb"]
                               if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0]
    selected_opposing_contacts = [
        item for digit in ("index", "middle", "ring", "pinky") for item in selected_topology[digit]
        if -MAX_CONTACT_PENETRATION_M <= item["distance_m"] <= 0.0
    ]
    thumb_near = bool(selected_thumb_contacts)
    opposing_near = bool(selected_opposing_contacts)
    selected_penetration_m = max(
        [max(0.0, -float(item["distance_m"])) for item in close_contacts] or [0.0]
    )
    opposed_normal_dot = float(selected["best_opposed_pair"]["normal_dot"])
    pregrasp_position_valid = all(
        float(model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0]) - 1e-9
        <= float(value)
        <= float(model.jnt_range[element(model, mujoco.mjtObj.mjOBJ_JOINT, name), 1]) + 1e-9
        for name, value in zip(ARM_NAMES, c3_pregrasp_arm)
    )
    candidate["static_gates"] = {
        "source_position_limits": bool(all_position_valid and pregrasp_position_valid),
        "source_velocity_planned_command_rate": bool(arm_rate_pass and core.CLOSE_TARGET_SLEW_RAD_S <= core.MAX_HAND_QVEL_RAD_S),
        "thumb_has_actual_shallow_contact": bool(thumb_near),
        "opposing_digit_has_actual_shallow_contact": bool(opposing_near),
        "all_active_digit_contacts_within_penetration_bound": bool(selected_penetration_m <= MAX_CONTACT_PENETRATION_M),
        "maximum_static_contact_penetration_m": selected_penetration_m,
        "thumb_opposing_contact_normal_dot": opposed_normal_dot,
        "contact_normals_are_opposed": bool(opposed_normal_dot <= MIN_OPPOSED_NORMAL_DOT),
        "no_palm_wrist_arm_bottle_contact": not bool(selected["contacts"]["non_digit"]),
        "collision_free_open_approach": bool(no_open_contacts and min_clearance > MIN_OPEN_CLEARANCE_M),
        "open_approach_minimum_clearance_m": float(min_clearance),
        "new_fresh_contact_geometry_recorded": True,
        "thumb_active_at_static_pose": bool(selected_thumb_contacts),
        "opposing_active_at_static_pose": bool(selected_opposing_contacts),
        "active_contacts_are_not_claimed_as_load_bearing": True,
    }
    candidate["static_gates"]["passed"] = bool(
        candidate["static_gates"]["source_position_limits"]
        and candidate["static_gates"]["source_velocity_planned_command_rate"]
        and thumb_near and opposing_near
        and selected_penetration_m <= MAX_CONTACT_PENETRATION_M
        and opposed_normal_dot <= MIN_OPPOSED_NORMAL_DOT
        and not selected["contacts"]["non_digit"]
        and candidate["static_gates"]["collision_free_open_approach"]
    )
    candidate["analysis"] = "single evidence-derived C3 world-pitch correction; selected first collision-free open waypoint with shallow opposed thumb/finger contacts; zero-step geometry only, not a dynamic grasp result"
    candidate["state_writes"] = "scratch MjData initial-state setup only; no active rollout, no runtime bottle qpos writes, no simulation steps"
    path = OUT / "candidate_c3.json"
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    audit = {
        "candidate_path": str(path),
        "compiled_model_path": str(compiled),
        "compiled_model_sha256": physical.normalized_model_sha256(compiled),
        "physics_steps": 0,
        "mj_step_calls": 0,
        "pitch_deg": PITCH_DEG,
        "contact_ik": contact_ik,
        "pregrasp_ik": pregrasp_ik,
        "selected_path_alpha": selected["alpha"],
        "selected_arm_q_rad": c3_contact_arm.tolist(),
        "selected_open_clearance_m": selected["open_clearance_m"],
        "selected_open_clearance_pair": [selected["open_minimum_robot_geom"], selected["open_minimum_bottle_geom"]],
        "source_arm_velocity_limits_rad_s": arm_velocity_limits,
        "approach_duration_s": core.APPROACH_SECONDS,
        "arm_target_rates_during_approach_rad_s": arm_rates,
        "minimum_open_approach_clearance_m": min_clearance,
        "open_approach_has_active_contacts": not no_open_contacts,
        "contact_distances_by_digit_m": distances_by_digit,
        "contact_preshape": contact_preshape,
        "active_close_contacts": close_contacts,
        "contact_normal_pair": selected["best_opposed_pair"],
        "static_gates": candidate["static_gates"],
        "state_writes": candidate["state_writes"],
        "source_c2_candidate_sha256": core.sha256(FIXTURE / "candidate_c2.json"),
        "base_state_sha256": core.sha256(FIXTURE / "canonical_ready_wrench_input.json"),
        "runtime_robot_xml_sha256": physical.runner.sha256(runtime_xml),
    }
    (OUT / "candidate_c3_static_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if candidate["static_gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
