#!/usr/bin/env python3
"""Recheck a saved Robotiq X2 approach against the current static bottle geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import uuid

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REVIEWS = Path("/home/lzy18001500226/projects/RobotSim/reviews")
EXPECTED_DIAGNOSTIC_MODEL_SHA = "a5dcd589fedaea5645ccf29b708b560f2e0dc923c8955a7d5901a996df57be94"
EXPECTED_ROBOTIQ_SHA = "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de"
EXPECTED_X2_MJCF_SHA = "20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292"
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import path_governance as pg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geom_name(model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"unnamed_geom_{geom_id}"


def body_name(model, geom_id: int) -> str:
    body = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"unnamed_body_{body}"


def is_robot_body(name: str) -> bool:
    return name != "world" and not name.startswith(("m0_", "world"))


def geom_distance_m(model, data, geom1: int, geom2: int) -> float:
    return float(mujoco.mj_geomDistance(model, data, geom1, geom2, 2.0, np.zeros(6)))


def pair_can_collide(model, geom1: int, geom2: int) -> bool:
    return bool(
        (int(model.geom_contype[geom1]) & int(model.geom_conaffinity[geom2]))
        or (int(model.geom_contype[geom2]) & int(model.geom_conaffinity[geom1]))
    )


def world_aabb(model, data, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(model.geom_aabb[geom_id])
    center_local, half_local = bounds[:3], bounds[3:]
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    center_world = np.asarray(data.geom_xpos[geom_id]) + rotation @ center_local
    half_world = np.abs(rotation) @ half_local
    return center_world - half_world, center_world + half_world


def geom_pair_clearance(model, data, geom1: int, geom2: int) -> tuple[float, str]:
    min1, max1 = world_aabb(model, data, geom1)
    min2, max2 = world_aabb(model, data, geom2)
    separation = np.maximum(0.0, np.maximum(min1 - max2, min2 - max1))
    aabb_lower_bound = float(np.linalg.norm(separation))
    if aabb_lower_bound > 0.0:
        return aabb_lower_bound, "conservative_world_aabb_lower_bound"
    distance = geom_distance_m(model, data, geom1, geom2)
    return distance, "mj_geom_distance_with_overlapping_world_aabbs"


def bottle_table_footprint_margins(model, data, bottle_body_id: int, table_geom_id: int) -> tuple[float, float]:
    if int(model.geom_type[table_geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("table top must be a box to compute the documented footprint margin")
    bottle_geoms = [geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == bottle_body_id]
    radii = []
    for geom in bottle_geoms:
        geom_type = int(model.geom_type[geom])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radii.append(float(model.geom_size[geom, 0]))
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_SPHERE)):
            radii.append(float(max(model.geom_size[geom, 0], model.geom_size[geom, 1])))
        else:
            raise ValueError(f"unsupported bottle footprint geom type: {geom_type}")
    if not radii:
        raise ValueError("bottle has no collision geometry for footprint check")
    table_rotation = np.asarray(data.geom_xmat[table_geom_id]).reshape(3, 3)
    table_center = np.asarray(data.geom_xpos[table_geom_id])
    bottle_center = np.asarray(data.xpos[bottle_body_id])
    local_delta = table_rotation.T @ (bottle_center - table_center)
    half_sizes = np.asarray(model.geom_size[table_geom_id])[:2]
    radius = max(radii)
    return tuple(float(half_sizes[axis] - abs(local_delta[axis]) - radius) for axis in range(2))


def audit(model_path: Path, result_path: Path, samples_path: Path) -> dict:
    model_hash = sha256(model_path)
    samples_hash = sha256(samples_path)
    if model_hash != EXPECTED_DIAGNOSTIC_MODEL_SHA:
        raise ValueError("input Robotiq/X2 diagnostic model SHA mismatch")

    prior = json.loads(result_path.read_text(encoding="utf-8"))
    identity = prior["identity"]
    if prior.get("status") != "PASS" or prior.get("experiment") != "X2 + Robotiq 2F-85 — SIMULATION_ONLY":
        raise ValueError("historical result is not the expected mounted diagnostic episode")
    if identity.get("robotiq_xml_sha256") != EXPECTED_ROBOTIQ_SHA:
        raise ValueError("historical result Robotiq source identity mismatch")
    if identity.get("x2_mjcf_sha256") != EXPECTED_X2_MJCF_SHA:
        raise ValueError("historical result X2 source identity mismatch")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    if bottle_body_id < 0:
        raise ValueError("diagnostic model has no m0_bottle body")
    bottle_geom_ids = [geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == bottle_body_id]
    bottle_geom_names = tuple(geom_name(model, geom) for geom in bottle_geom_ids)
    if set(bottle_geom_names) != {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}:
        raise ValueError("unexpected historical bottle geom set")
    data = mujoco.MjData(model)

    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    if bottle_geom < 0 or bottle_joint < 0:
        raise ValueError("compiled canonical bottle geometry/free joint is missing")
    bottle_qadr = int(model.jnt_qposadr[bottle_joint])
    bottle_qpos0 = model.qpos0[bottle_qadr:bottle_qadr + 7].copy()
    bottle_mass = float(model.body_mass[bottle_body_id])

    table_geoms = [
        geom for geom in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("m0_table_")
        and (int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0)
    ]
    robot_geoms = [
        geom for geom in range(model.ngeom)
        if geom != bottle_geom
        and (int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0)
        and not body_name(model, geom).startswith(("m0_bottle", "m0_table"))
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "") != "m0_floor"
    ]
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM_JOINTS]
    if any(joint_id < 0 for joint_id in arm_joint_ids):
        raise ValueError("diagnostic model is missing a required right-arm source joint")

    rows = json.loads(samples_path.read_text(encoding="utf-8"))
    if len(rows) != 101 or [round(float(row.get("fraction", -1)), 2) for row in rows] != [i / 100 for i in range(101)]:
        raise ValueError("saved static approach input is not the expected 101-sample path")

    min_bottle = (float("inf"), None, None, None, None)
    min_table = (float("inf"), None, None, None, None, None)
    joint_limit_violations = []
    object_contacts = []
    robot_table_contacts = []
    robot_self_contacts = []
    forwards = 0
    for sample_index, row in enumerate(rows):
        arm_q = row.get("right_arm_qpos_rad")
        if not isinstance(arm_q, list) or len(arm_q) != len(arm_joint_ids):
            raise ValueError(f"invalid arm qpos sample {sample_index}")

        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        for name, joint_id, value in zip(RIGHT_ARM_JOINTS, arm_joint_ids, arm_q):
            q = float(value)
            qadr = int(model.jnt_qposadr[joint_id])
            data.qpos[qadr] = q
        for joint_id in range(model.njnt):
            if not model.jnt_limited[joint_id]:
                continue
            joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"unnamed_joint_{joint_id}"
            q = float(data.qpos[int(model.jnt_qposadr[joint_id])])
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
            if not (lower - 1e-9 <= q <= upper + 1e-9):
                joint_limit_violations.append({"sample": sample_index, "joint": joint, "q_rad": q, "range_rad": [lower, upper]})

        mujoco.mj_forward(model, data)
        forwards += 1

        for geom in robot_geoms:
            for bottle_component in bottle_geom_ids:
                if not pair_can_collide(model, geom, bottle_component):
                    continue
                dist, method = geom_pair_clearance(model, data, geom, bottle_component)
                if dist < min_bottle[0]:
                    min_bottle = (
                        dist,
                        sample_index,
                        body_name(model, geom),
                        f"{geom_name(model, geom)} <-> {geom_name(model, bottle_component)}",
                        method,
                    )
            for table_geom in table_geoms:
                if not pair_can_collide(model, geom, table_geom):
                    continue
                table_dist, method = geom_pair_clearance(model, data, geom, table_geom)
                if table_dist < min_table[0]:
                    min_table = (
                        table_dist,
                        sample_index,
                        body_name(model, geom),
                        geom_name(model, geom),
                        geom_name(model, table_geom),
                        method,
                    )

        for contact_index, contact in enumerate(data.contact[:data.ncon]):
            g0, g1 = int(contact.geom1), int(contact.geom2)
            b0, b1 = body_name(model, g0), body_name(model, g1)
            if g0 in bottle_geom_ids or g1 in bottle_geom_ids:
                bottle_component = g0 if g0 in bottle_geom_ids else g1
                other = g1 if g0 in bottle_geom_ids else g0
                if not body_name(model, other).startswith("m0_table"):
                    object_contacts.append({
                        "sample": sample_index,
                        "geom0": geom_name(model, g0),
                        "geom1": geom_name(model, g1),
                        "bottle_component": geom_name(model, bottle_component),
                        "body0": b0,
                        "body1": b1,
                        "distance_mm": float(contact.dist) * 1000.0,
                    })
            if b0.startswith("m0_table") and not b1.startswith(("m0_table", "m0_bottle")):
                robot_table_contacts.append({"sample": sample_index, "body": b1, "geom": geom_name(model, g1), "distance_mm": float(contact.dist) * 1000.0})
            if b1.startswith("m0_table") and not b0.startswith(("m0_table", "m0_bottle")):
                robot_table_contacts.append({"sample": sample_index, "body": b0, "geom": geom_name(model, g0), "distance_mm": float(contact.dist) * 1000.0})
            if is_robot_body(b0) and is_robot_body(b1):
                robot_self_contacts.append({"sample": sample_index, "body0": b0, "body1": b1, "distance_mm": float(contact.dist) * 1000.0})

    top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_table_top")
    bottle_table_gap = min(geom_distance_m(model, data, geom, top_id) for geom in bottle_geom_ids) if top_id >= 0 else None
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    edge_margins = bottle_table_footprint_margins(model, data, bottle_body_id, top_id) if top_id >= 0 else (float("-inf"), float("-inf"))
    bottle_geoms = [
        {
            "name": geom_name(model, geom),
            "type": int(model.geom_type[geom]),
            "local_position_m": model.geom_pos[geom].tolist(),
            "size_m": model.geom_size[geom].tolist(),
            "contype": int(model.geom_contype[geom]),
            "conaffinity": int(model.geom_conaffinity[geom]),
        }
        for geom in bottle_geom_ids
    ]
    table_top_z = float(data.geom_xpos[top_id, 2] + model.geom_size[top_id, 2]) if top_id >= 0 else None
    min_bottle_mm = min_bottle[0] * 1000.0
    min_table_mm = min_table[0] * 1000.0
    gates = {
        "input_model_identity": True,
        "saved_approach_witness": len(rows) == 101,
        "all_source_joint_limits": not joint_limit_violations,
        "current_bottle_on_table_geometry": bottle_table_gap is not None and abs(bottle_table_gap) <= 1e-4,
        "target_footprint_inside_table": min(edge_margins) > 0.0,
        "robot_bottle_clearance_ge_2mm": min_bottle_mm >= 2.0,
        "robot_table_collision_free": not robot_table_contacts and min_table_mm >= 2.0,
        "robot_self_contact_free": not robot_self_contacts,
        "no_unintended_bottle_contact_on_approach": not object_contacts,
    }
    status = "PASS_STATIC_APPROACH_DIAGNOSTIC" if all(gates.values()) else "BLOCKED_STATIC_APPROACH_DIAGNOSTIC"
    return {
        "schema": "robotsim.issue46.robotiq_current_bottle_static_audit.v1",
        "status": status,
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_STATIC_GEOMETRY",
        "physics_steps": 0,
        "mj_forward_calls": forwards,
        "input_model": {"path": str(model_path), "sha256": model_hash, "expected_sha256": EXPECTED_DIAGNOSTIC_MODEL_SHA},
        "input_historical_result": {"path": str(result_path), "sha256": sha256(result_path), "status": prior["status"]},
        "input_saved_approach": {"path": str(samples_path), "sha256": samples_hash, "sample_count": len(rows)},
        "source_identity": {
            "x2_commit": identity["x2_repository"],
            "x2_mjcf_sha256": identity["x2_mjcf_sha256"],
            "robotiq_commit": identity["menagerie_checkout"]["head"],
            "robotiq_xml_sha256": identity["robotiq_xml_sha256"],
            "robotiq_license": identity["robotiq_license"],
        },
        "canonical_bottle": {
            "shape": "four-geom source model: bottle_body, bottle_shoulder, bottle_neck, bottle_cap",
            "root_freejoint_qpos0": bottle_qpos0.tolist(),
            "mass_kg": bottle_mass,
            "geometry": bottle_geoms,
            "table_top_z_m": table_top_z,
            "footprint_edge_margins_xy_m": list(edge_margins),
            "geometry_replacement": "none; the exact historical multipart bottle and free joint are retained",
        },
        "geometry": {
            "robot_bottle_min_clearance_mm": min_bottle_mm,
            "robot_bottle_first_minimum": {
                "sample": min_bottle[1],
                "body": min_bottle[2],
                "geom_pair": min_bottle[3],
                "measurement": min_bottle[4],
            },
            "robot_table_min_clearance_mm": min_table_mm,
            "robot_table_first_minimum": {
                "sample": min_table[1],
                "body": min_table[2],
                "robot_geom": min_table[3],
                "table_geom": min_table[4],
                "measurement": min_table[5],
            },
            "bottle_table_signed_distance_mm": None if bottle_table_gap is None else bottle_table_gap * 1000.0,
            "robot_bottle_contact_count": len(object_contacts),
            "robot_table_contact_count": len(robot_table_contacts),
            "robot_self_contact_count": len(robot_self_contacts),
            "source_joint_limit_violations": joint_limit_violations,
            "first_robot_bottle_contacts": object_contacts[:12],
            "first_robot_table_contacts": robot_table_contacts[:12],
            "first_robot_self_contacts": robot_self_contacts[:12],
        },
        "gates": gates,
        "limitations": [
            "The input model is the historical SIMULATION_ONLY_DIAGNOSTIC X2+Robotiq model, including its recorded head_pitch_link/torso_link exclusion and 0.001 rad coupler activation margin.",
            "This audit checks the 101 saved arm-path samples only; it does not claim continuous swept-volume proof or rerun physical contact/load transfer.",
            "The canonical 0.570 kg four-geom bottle geometry and free-joint qpos0 are unchanged in all static samples.",
            "No collision geometry, source model, controller, or physical parameters were written to the input model.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic-model", required=True)
    ap.add_argument("--approach-samples", required=True)
    ap.add_argument("--historical-result", required=True)
    ap.add_argument("--output-parent", default=str(REVIEWS / "issue-46"))
    args = ap.parse_args()

    model_path = Path(args.diagnostic_model)
    samples_path = Path(args.approach_samples)
    result_path = Path(args.historical_result)
    for label, path in (("diagnostic model", model_path), ("saved approach", samples_path), ("historical result", result_path)):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    result = audit(model_path, result_path, samples_path)
    output_parent = pg.validate_output_path(
        args.output_parent,
        authorized_roots=(str(REVIEWS),),
        require_exists=True,
        expected_kind="directory",
        environment="wsl",
    )
    output = pg.create_unique_output_directory(
        output_parent.wsl_path,
        prefix="robotiq-current-bottle-static",
        authorized_roots=(str(REVIEWS),),
        environment="wsl",
    )
    result.update({
        "run_id": str(uuid.uuid4()),
        "output": output.as_dict(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    })
    pg.write_new_text(output.native_path / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", authorized_roots=(str(REVIEWS),), environment="wsl")
    report = [
        "# Existing X2 + Robotiq approach against the current static bottle",
        "",
        f"Run ID: {result['run_id']}",
        f"Decision: {result['status']}",
        f"Physics steps: {result['physics_steps']}",
        f"mj_forward calls: {result['mj_forward_calls']}",
        f"Minimum robot/bottle clearance: {result['geometry']['robot_bottle_min_clearance_mm']:.3f} mm",
        f"Minimum robot/table clearance: {result['geometry']['robot_table_min_clearance_mm']:.3f} mm",
        f"Canonical bottle mass: {result['canonical_bottle']['mass_kg']:.3f} kg",
        f"Bottle footprint margins: {result['canonical_bottle']['footprint_edge_margins_xy_m']} m",
        f"Closest approach pair: {result['geometry']['robot_bottle_first_minimum']}",
        f"Closest robot/table pair: {result['geometry']['robot_table_first_minimum']}",
        f"Unmet gates: {[name for name, passed in result['gates'].items() if not passed]}",
        "",
        "The checked model remains SIMULATION_ONLY_DIAGNOSTIC. This static check does not rerun the grasp episode, claim physical contact/load transfer, or certify continuous swept-volume clearance.",
    ]
    pg.write_new_text(output.native_path / "REPORT.md", "\n".join(report) + "\n", authorized_roots=(str(REVIEWS),), environment="wsl")
    model_arg = shlex.quote(str(model_path))
    sample_arg = shlex.quote(str(samples_path))
    result_arg = shlex.quote(str(result_path))
    command = (
        f"cd {shlex.quote(str(ROOT))}\n"
        f"PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python "
        f"scripts/research/issue46_manipulation_integration/audit_robotiq_current_bottle_static.py "
        f"--diagnostic-model {model_arg} --approach-samples {sample_arg} --historical-result {result_arg}\n"
    )
    pg.write_new_text(output.native_path / "REPRODUCE.md", command, authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(output.native_path, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "output": output.as_dict(), "manifest_count": len(manifest["artifacts"])}, indent=2))
    return 0 if result["status"] == "PASS_STATIC_APPROACH_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
