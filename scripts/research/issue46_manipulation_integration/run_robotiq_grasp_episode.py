#!/usr/bin/env python3
"""One guarded full-X2 Robotiq contact, load-transfer, lift, transfer and release episode."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

os.environ["MUJOCO_GL"] = "egl"

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
REVIEWS = Path("/home/lzy18001500226/projects/RobotSim/reviews")
MODEL = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/x2_robotiq_2f85_simulation_only.xml"
)
HISTORICAL = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/result.json"
)
STATIC = REVIEWS / "issue-46/robotiq-arm-approach-qualified-20261024T065752Z/result.json"
INTERNAL = REVIEWS / "issue-46/robotiq-internal-range-preflight-20261024T073206Z/result.json"
APPROACH_A = REVIEWS / "issue-46/robotiq-dynamic-approach-20261024T080312Z/result.json"
TRANSFER_PREFLIGHT = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/raw/live_loaded_transport_preflight.json"
)
LIFT_IK = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/raw/cartesian_lift_waypoint_ik.json"
)
EXPECTED_MODEL_SHA = "a5dcd589fedaea5645ccf29b708b560f2e0dc923c8955a7d5901a996df57be94"
EXPECTED_DT = 0.001
CLEARANCE_M = 0.002
GRAVITY = 9.81
ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)

sys.path.insert(0, str(ROOT / "scripts" / "agent"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "issue46_manipulation_integration"))
import path_governance as pg
from episode_gate import EpisodeCriteria, EpisodeFrame, ManipulationPhase, evaluate_episode_evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_criteria(target_com_xy: tuple[float, float]) -> EpisodeCriteria:
    return EpisodeCriteria(
        required_contact_groups=(("left_pad",), ("right_pad",)),
        minimum_contact_fraction=0.80,
        minimum_tool_upward_support_n=0.570 * GRAVITY,
        minimum_table_unloading_n=3.0,
        maximum_relative_slip_m=0.015,
        minimum_approach_clearance_m=0.002,
        minimum_lift_m=0.050,
        minimum_hold_duration_s=1.0,
        place_target_xy_world_m=target_com_xy,
        maximum_place_xy_error_m=0.020,
        maximum_upright_error_rad=math.radians(5.0),
        minimum_release_opening_m=0.0884,
        minimum_settle_duration_s=3.0,
        maximum_settle_linear_speed_m_s=0.020,
        maximum_settle_angular_speed_rad_s=0.050,
        maximum_settle_position_drift_m=0.020,
    )


def id_name(model, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or f"unnamed_{object_id}"


def allowed_contact(row: dict) -> bool:
    bodies, geoms = (row["body1"], row["body2"]), (row["geom1"], row["geom2"])
    bottle = next((body for body in bodies if body.startswith("m0_bottle")), None)
    if bottle is not None:
        other = bodies[1] if bodies[0] == bottle else bodies[0]
        return other.startswith("m0_table") or other in {"rq_left_pad", "rq_right_pad"}
    if "m0_floor" in geoms or "floor" in geoms:
        other = bodies[1] if geoms[0] in {"m0_floor", "floor"} else bodies[0]
        return other.startswith("m0_table") or "foot" in other.lower() or "ankle" in other.lower()
    return False


def detailed_contacts(model, data) -> dict:
    rows = []
    hand_z = 0.0
    table_z = 0.0
    pad_roles = set()
    any_pad = False
    table_contact = False
    left_hand_contact = False
    unexpected = []
    bottle_body = "m0_bottle"
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = id_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1]))
        body2 = id_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2]))
        row = {
            "geom1": id_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1), "body1": body1,
            "geom2": id_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2), "body2": body2,
            "distance_m": float(contact.dist), "contact_position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_normal_geom1_to_geom2_world": np.asarray(contact.frame, dtype=float).reshape(3, 3)[0].tolist(),
            "force_contact_frame_n": None,
            "contact_frame_to_world_n": None, "force_on_bottle_world_n": None,
            "bottle_contact_role": None, "normal_force_n": None, "tangential_force_on_bottle_world_n": None,
        }
        local = np.zeros(6)
        if int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, index, local)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_on_geom2_world = frame.T @ local[:3]
            row["force_contact_frame_n"] = local.tolist()
            row["contact_frame_to_world_n"] = force_on_geom2_world.tolist()
            row["normal_force_n"] = max(0.0, float(local[0]))
            if body1 == bottle_body or body2 == bottle_body:
                any_pad = True
                other_body = body2 if body1 == bottle_body else body1
                sign = 1.0 if body2 == bottle_body else -1.0
                force_on_bottle = sign * force_on_geom2_world
                row["force_on_bottle_world_n"] = force_on_bottle.tolist()
                normal_world = frame[0, :]
                normal_component = float(np.dot(force_on_bottle, normal_world))
                tangent = force_on_bottle - normal_component * normal_world
                row["tangential_force_on_bottle_world_n"] = tangent.tolist()
                if other_body.startswith("m0_table"):
                    table_contact = True
                    table_z += float(force_on_bottle[2])
                    row["bottle_contact_role"] = "table"
                elif other_body == "rq_left_pad":
                    if row["normal_force_n"] > 0.0:
                        pad_roles.add("left_pad")
                    hand_z += float(force_on_bottle[2])
                    row["bottle_contact_role"] = "left_pad"
                elif other_body == "rq_right_pad":
                    if row["normal_force_n"] > 0.0:
                        pad_roles.add("right_pad")
                    hand_z += float(force_on_bottle[2])
                    row["bottle_contact_role"] = "right_pad"
                else:
                    if other_body.startswith("left_") or "left_hand" in other_body or "left_wrist" in other_body:
                        left_hand_contact = True
                    row["bottle_contact_role"] = "unintended"
            if not allowed_contact(row):
                unexpected.append(row)
        rows.append(row)
    return {
        "contacts": rows, "tool_object_contact_active": any_pad,
        "force_bearing_contact_roles": sorted(pad_roles),
        "tool_upward_support_n": hand_z, "table_reaction_vertical_n": table_z,
        "table_contact": table_contact, "left_hand_contact": left_hand_contact,
        "unexpected_contacts": unexpected,
    }


def equality_audit(model, data) -> dict:
    equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = np.flatnonzero(np.asarray(data.efc_type[:data.nefc]) == equality_type)
    connect = []
    joint = []
    other = []
    for row in rows:
        equality_id = int(data.efc_id[row])
        eq_type = int(model.eq_type[equality_id])
        value = float(data.efc_pos[row])
        record = {"constraint_row": int(row), "equality_id": equality_id,
                  "equality_name": id_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id),
                  "equality_type": eq_type, "residual": value,
                  "units": "m" if eq_type == int(mujoco.mjtEq.mjEQ_CONNECT) else
                           "rad" if eq_type == int(mujoco.mjtEq.mjEQ_JOINT) else "model-specific"}
        if eq_type == int(mujoco.mjtEq.mjEQ_CONNECT):
            connect.append(record)
        elif eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
            joint.append(record)
        else:
            other.append(record)
    joint_values = np.asarray([r["residual"] for r in joint], dtype=float)
    connect_values = np.asarray([r["residual"] for r in connect], dtype=float)
    expected_joint_followers = sum(int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT) for i in range(model.neq))
    return {"constraint_rows": int(len(rows)),
            "connect_constraint_residuals_m": connect,
            "max_abs_connect_residual_m": float(np.max(np.abs(connect_values))) if len(connect_values) else 0.0,
            "joint_mimic_residuals_rad": joint,
            "max_abs_joint_mimic_residual_rad": float(np.max(np.abs(joint_values))) if len(joint_values) else 0.0,
            "other_equality_residuals": other,
            "joint_mimic_gate_rad": 1e-6,
            "pass": expected_joint_followers == 1 and len(joint_values) == 1
                    and bool(np.isfinite(joint_values).all())
                    and float(np.max(np.abs(joint_values))) <= 1e-6
                    and bool(np.isfinite(connect_values).all())}


def pad_opening(model, data) -> float:
    names = ("rq_left_pad1", "rq_left_pad2", "rq_right_pad1", "rq_right_pad2")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    left = np.mean(data.geom_xpos[ids[:2]], axis=0)
    right = np.mean(data.geom_xpos[ids[2:]], axis=0)
    return float(np.linalg.norm(left - right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", type=Path, default=REVIEWS / "issue-46")
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--prior-abort-steps", type=int, default=0,
                         help="steps spent on the ordinary mixed-unit follower-gate abort")
    parser.add_argument("--preflight-only", action="store_true",
                         help="compile and validate the exact B episode inputs without mj_step")
    args = parser.parse_args()
    if not math.isclose(args.max_duration_s, 30.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("the authorized B episode is fixed at a 30 s maximum")

    for path in (MODEL, HISTORICAL, STATIC, INTERNAL, APPROACH_A, TRANSFER_PREFLIGHT, LIFT_IK):
        if not path.is_file():
            raise SystemExit(f"required pinned input is missing: {path}")
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    internal = json.loads(INTERNAL.read_text(encoding="utf-8"))
    approach_a = json.loads(APPROACH_A.read_text(encoding="utf-8"))
    transfer_plan = json.loads(TRANSFER_PREFLIGHT.read_text(encoding="utf-8"))
    lift_plan = json.loads(LIFT_IK.read_text(encoding="utf-8"))
    if approach_a.get("status") != "DYNAMIC_APPROACH_PASS" or approach_a.get("physics_steps") != 5500:
        raise SystemExit("Experiment A did not pass the exact full-approach plus stop-hold gate")
    if approach_a.get("input_identity", {}).get("model_sha256") != EXPECTED_MODEL_SHA:
        raise SystemExit("Experiment A used a different full-X2 model")
    if approach_a.get("gates", {}).get("external_clearance_ge_2mm") is not True:
        raise SystemExit("Experiment A clearance gate is not PASS")
    if transfer_plan.get("status") != "PASS" or transfer_plan.get("runtime_bottle_pose_modified") is not False:
        raise SystemExit("historical loaded transfer preflight is not reusable as a reference")
    if len(lift_plan) != 16 or not all(row.get("gate_pass") for row in lift_plan):
        raise SystemExit("pinned source-valid lift waypoint set is incomplete")
    if historical.get("physical_integrity", {}).get("bottle_weld_or_equality") is not False:
        raise SystemExit("historical diagnostic model does not establish a free bottle")
    if historical.get("physical_integrity", {}).get("bottle_mocap_or_follow") is not False:
        raise SystemExit("historical model reports bottle mocap/follow support")
    if historical.get("physical_integrity", {}).get("hidden_support") is not False:
        raise SystemExit("historical model reports hidden support")
    if historical.get("physical_integrity", {}).get("left_hand_assistance") is not False:
        raise SystemExit("historical model reports left-hand assistance")
    model_sha = sha256(MODEL)
    if model_sha != EXPECTED_MODEL_SHA:
        raise SystemExit(f"exact diagnostic model SHA mismatch: {model_sha}")

    runner = ROOT / "scripts/research/issue46_manipulation_integration/run_robotiq_dynamic_approach.py"
    approach_module = load_module(runner, "issue46_robotiq_approach_helpers")
    controller_snapshot = Path(approach_a["input_identity"]["controller_source_snapshot_path"])
    if not controller_snapshot.is_file() or sha256(controller_snapshot) != approach_a["input_identity"]["controller_source_snapshot_sha256"]:
        raise SystemExit("the exact controller snapshot used by Experiment A is unavailable")
    legacy = load_module(controller_snapshot, "issue46_robotiq_controller_snapshot_b")

    source = pg.validate_output_path(args.output_parent, authorized_roots=(str(REVIEWS),),
                                     require_exists=True, expected_kind="directory", environment="wsl")
    output = pg.create_unique_output_directory(source.wsl_path, prefix="robotiq-full-x2-episode",
                                                authorized_roots=(str(REVIEWS),), environment="wsl")
    out = output.native_path
    raw = out / "raw"
    images = out / "images"
    inputs = out / "inputs"
    raw.mkdir()
    images.mkdir()
    inputs.mkdir()
    snapshot_path = inputs / "controller_source_snapshot.py"
    shutil.copyfile(controller_snapshot, snapshot_path)

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    if mujoco.__version__ != "3.3.6" or sys.version_info[:3] != (3, 10, 12):
        raise SystemExit("MuJoCo/Python runtime does not match the pinned benchmark runtime")
    if not math.isclose(float(model.opt.timestep), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("full-X2 model timestep is not 1 ms")
    if model.nu != 32 or model.ngeom != 137:
        raise SystemExit("full-X2 compiled model inventory changed")
    source_limits, velocity_limits = approach_module.source_limit_audit(model, historical)
    if not source_limits["pass"]:
        raise SystemExit(f"source position/velocity ranges are not verified: {source_limits}")
    refs = legacy.controller_config(model)
    if len(refs) != 31 or len(historical["controller"]["actuator_mapping"]) != 31:
        raise SystemExit("historical 31-actuator controller map is incomplete")
    for current, prior in zip(refs, historical["controller"]["actuator_mapping"]):
        for key in ("actuator_id", "actuator", "joint_id", "joint", "qpos_id", "dof_id", "gear",
                    "kp_simulation_derived", "kd_simulation_derived", "torque_min_nm", "torque_max_nm"):
            a_value, b_value = current.get(key), prior.get(key)
            equal = (abs(float(a_value) - float(b_value)) <= 1e-9
                     if isinstance(a_value, (int, float)) and isinstance(b_value, (int, float))
                     else a_value == b_value)
            if not equal:
                raise SystemExit(f"historical actuator semantics changed at {current['joint']} / {key}")

    candidate = static["candidate_25mm_retreat"]
    arm_names = candidate["right_arm_joint_names"]
    if tuple(arm_names) != ARM:
        raise SystemExit("static candidate right-arm joint order mismatch")
    qpre = np.asarray(candidate["joint_trajectory"][0]["right_arm_qpos_rad"], dtype=float)
    qapproach = np.asarray(candidate["joint_trajectory"][-1]["right_arm_qpos_rad"], dtype=float)
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
    arm_qpos_ids = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    arm_dof_ids = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
    bottle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    if min(arm_joint_ids + [grip_id, bottle_joint_id, bottle_body_id]) < 0:
        raise SystemExit("compiled model lacks a required full-X2, gripper, or free-bottle object")
    if int(model.jnt_type[bottle_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise SystemExit("canonical bottle is not an independent freejoint body")
    if int(model.body_mocapid[bottle_body_id]) >= 0:
        raise SystemExit("canonical bottle is a mocap body")
    for equality_id in range(model.neq):
        if int(model.eq_type[equality_id]) in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            if int(model.eq_obj1id[equality_id]) == bottle_body_id or int(model.eq_obj2id[equality_id]) == bottle_body_id:
                raise SystemExit("bottle is attached by a runtime equality")

    clearance_qualifier = load_module(
        ROOT / "scripts/research/issue46_manipulation_integration/qualify_robotiq_static_approach.py",
        "issue46_robotiq_geometry_qualifier_b",
    )
    exclusions = clearance_qualifier.load_exclusions(MODEL)
    active_geom_ids = [i for i in range(model.ngeom) if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])]
    mesh_cache = {}
    shapes = {i: clearance_qualifier.convex_shape(model, i, mesh_cache) for i in active_geom_ids}
    shapes = {i: value for i, value in shapes.items() if value is not None}
    pair_lists = clearance_qualifier.model_path_pairs(model, active_geom_ids, exclusions)
    pair_arrays = {k: np.asarray(v, dtype=np.int32).reshape((-1, 2)) for k, v in pair_lists.items()
                   if k in {"robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self"}}

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Single reset-time initialization. Bottle pose remains free and receives no later qpos write.
    data.qpos[arm_qpos_ids] = qpre
    data.ctrl[grip_id] = 0.0
    mujoco.mj_forward(model, data)
    initial_contacts = approach_module.contact_rows(model, data, legacy)
    if not legacy.static_contact_gate(initial_contacts).get("pass"):
        raise SystemExit("Episode B initial PREGRASP state is not nonpenetrating")
    initial_clearance = approach_module.clearance_scan(
        model, data, clearance_qualifier, shapes, pair_arrays,
        clearance_qualifier.all_world_shapes(model, data, shapes),
    )
    initial_clearance_pass = all(
        initial_clearance[group]["minimum_clearance_lower_bound_m"] is not None
        and initial_clearance[group]["minimum_clearance_lower_bound_m"] >= CLEARANCE_M
        for group in ("robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self")
    )
    if not initial_clearance_pass:
        raise SystemExit(f"Episode B initial PREGRASP does not satisfy the 2 mm external clearance gate: {initial_clearance}")
    bottle_qpos_id = int(model.jnt_qposadr[bottle_joint_id])
    bottle_dof_id = int(model.jnt_dofadr[bottle_joint_id])
    bottle_mass = float(model.body_mass[bottle_body_id])
    if not math.isclose(bottle_mass, 0.570, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"canonical bottle mass changed: {bottle_mass}")
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    if tcp_site_id < 0:
        raise SystemExit("compiled model is missing the pinned Robotiq TCP site")
    bottle_qpos_initial = data.qpos[bottle_qpos_id:bottle_qpos_id + 7].copy()
    bottle_root_initial = bottle_qpos_initial[:3].copy()
    bottle_com_initial = np.asarray(data.xipos[bottle_body_id], dtype=float).copy()
    com_offset_world = bottle_com_initial - bottle_root_initial
    target_root = np.asarray(historical["scene_configuration"]["target_bottle_root_position_world_m"], dtype=float)
    criteria = gate_criteria(tuple((target_root + com_offset_world)[:2]))
    if args.preflight_only:
        preflight = {
            "schema": "robotsim.issue46.full_x2_robotiq_episode_preflight.v1",
            "classification": "SIMULATION_ONLY_DIAGNOSTIC_ZERO_STEP_PREFLIGHT",
            "run_id": str(uuid.uuid4()), "physics_steps": 0,
            "mujoco_version": mujoco.__version__, "python": sys.version.split()[0],
            "model_path": str(MODEL), "model_sha256": model_sha,
            "approach_a_result_path": str(APPROACH_A), "approach_a_status": approach_a["status"],
            "lift_waypoint_count": len(lift_plan), "transfer_waypoint_count": len(transfer_plan["waypoints"]),
            "source_limit_audit": source_limits, "controller_mapping_count": len(refs),
            "initial_contacts": initial_contacts, "initial_clearances": initial_clearance,
            "initial_clearance_pass": initial_clearance_pass,
            "bottle_mass_kg": bottle_mass, "bottle_qpos_initial": data.qpos[bottle_qpos_id:bottle_qpos_id + 7].tolist(),
            "bottle_freejoint": "m0_bottle_free", "bottle_mocap_id": int(model.body_mocapid[bottle_body_id]),
            "bottle_runtime_equality_count": 0,
            "episode_gate_criteria": {"required_contact_groups": criteria.required_contact_groups,
                                       "minimum_contact_fraction": criteria.minimum_contact_fraction,
                                       "minimum_tool_upward_support_n": criteria.minimum_tool_upward_support_n,
                                       "minimum_table_unloading_n": criteria.minimum_table_unloading_n,
                                       "maximum_relative_slip_m": criteria.maximum_relative_slip_m,
                                       "minimum_lift_m": criteria.minimum_lift_m,
                                       "minimum_hold_duration_s": criteria.minimum_hold_duration_s},
            "output": output.as_dict(),
        }
        pg.write_new_text(out / "preflight.json", json.dumps(preflight, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          authorized_roots=(str(REVIEWS),), environment="wsl")
        manifest = pg.write_artifact_manifest(out, authorized_roots=(str(REVIEWS),), environment="wsl")
        pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
        print(json.dumps({"status": "PASS_ZERO_STEP_EPISODE_PREFLIGHT", "run_id": preflight["run_id"],
                          "physics_steps": 0, "initial_clearance_pass": initial_clearance_pass,
                          "output": output.as_dict(), "artifact_count": len(manifest["artifacts"])},
                         indent=2, ensure_ascii=False))
        return 0
    targets_base = {row["joint"]: float(data.qpos[row["qpos_id"]]) for row in refs if row["joint"] not in ARM}
    actuator_by_joint = {row["joint"]: int(row["actuator_id"]) for row in refs}
    target_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM]
    hinge_names = [id_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
                   if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    hinge_qpos_ids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in hinge_names]
    hinge_dof_ids = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in hinge_names]
    actuator_names = [id_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    lift_q = [np.asarray(row["q_rad"], dtype=float) for row in lift_plan]
    route_waypoints = transfer_plan["waypoints"]
    route_q = [np.asarray(row["arm_qpos_rad"], dtype=float) for row in route_waypoints]
    if len(route_q) != 28 or not all(row.get("gate_pass") for row in route_waypoints):
        raise SystemExit("saved loaded-transfer waypoint witness is not source-valid")

    stage_frames: list[EpisodeFrame] = []
    stage_rows: list[dict] = []
    episode_frames_path = raw / "episode_frames.jsonl"
    physics_trace_path = raw / "physics_trace.jsonl"
    contact_trace_path = raw / "contact_wrench_trace.jsonl"
    clearance_trace_path = raw / "clearance_trace.jsonl"
    video_path = out / "full_x2_robotiq_episode.mp4"
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.22, -0.05, 0.89]
    camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -14
    writer = imageio.get_writer(str(video_path), fps=25, codec="libx264", quality=7, macro_block_size=1)
    slip_reference_local = None
    previous_com = None
    previous_com_velocity = np.zeros(3)
    first_failure = None
    total_steps = 0
    total_wall_start = time.monotonic()
    q_target_current = qpre.copy()
    phase_records = []
    safety = {"finite_state": True, "source_position_limits": True, "source_velocity_limits": True,
              "actuator_limits": True, "unexpected_contact": True, "external_clearance": True,
              "internal_spring_pad_contact": True, "equality_constraints": True}
    minimum_clearance = {k: math.inf for k in ("robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self")}
    max_torque = {name: 0.0 for name in actuator_names}
    max_equality_residual = {"connect_m": 0.0, "joint_mimic_rad": 0.0}
    settle_support = []
    settle_support_rows = []
    settle_speed = []
    settle_balance_residual = []

    def save_frame(label: str) -> None:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(images / f"{label}.png")

    def current_opening() -> float:
        return pad_opening(model, data)

    def run_segment(operation: str, target_points: list[np.ndarray], duration: float,
                    grip_start: float, grip_end: float, gate_phase: ManipulationPhase | None = None,
                    approach_clearance: bool = False) -> dict:
        nonlocal total_steps, q_target_current, previous_com, previous_com_velocity, slip_reference_local, first_failure
        if args.prior_abort_steps + total_steps + int(round(duration / EXPECTED_DT)) > 30000:
            first_failure = {"gate": "30_second_physics_limit", "phase": operation}
            return {"operation": operation, "steps": 0, "completed": False, "failure": first_failure}
        steps = int(round(duration / EXPECTED_DT))
        phase_start_time = float(data.time)
        start_step = total_steps
        min_clear = {k: math.inf for k in minimum_clearance}
        contact_force_roles: dict[str, int] = {"left_pad": 0, "right_pad": 0}
        table_force_rows = []
        step_count_bilateral = 0
        for k in range(steps):
            u = (k + 1) / steps
            s, dsdu = legacy.quintic(u)
            location = s * (len(target_points) - 1)
            seg = min(int(location), len(target_points) - 2)
            frac = location - seg
            q0, q1 = target_points[seg], target_points[seg + 1]
            q_target = q0 + frac * (q1 - q0)
            qvel_target = (len(target_points) - 1) * (dsdu / duration) * (q1 - q0)
            q_target_current = q_target.copy()
            all_targets = legacy.set_target_dict(model, q_target, targets_base)
            target_vel = {name: float(value) for name, value in zip(ARM, qvel_target)}
            mujoco.mj_forward(model, data)
            command_rows = legacy.apply_controller(model, data, refs, all_targets, target_vel)
            grip_s, _ = legacy.quintic(u)
            data.ctrl[grip_id] = grip_start + grip_s * (grip_end - grip_start)
            mujoco.mj_step(model, data)
            total_steps += 1
            step = total_steps
            t = float(data.time)

            contacts = detailed_contacts(model, data)
            rows = contacts["contacts"]
            contact_trace_path_obj.write(json.dumps({"step": step, "time_s": t, "operation": operation,
                                                      "episode_phase": gate_phase.value if gate_phase else None,
                                                      "contacts": rows}, allow_nan=False) + "\n")
            contact_force_roles.update({role: contact_force_roles[role] + int(role in contacts["force_bearing_contact_roles"])
                                        for role in contact_force_roles})
            step_count_bilateral += int("left_pad" in contacts["force_bearing_contact_roles"]
                                        and "right_pad" in contacts["force_bearing_contact_roles"])
            table_force_rows.append(contacts["table_reaction_vertical_n"])

            range_errors, velocity_errors = legacy.all_limited_joint_checks(model, data, velocity_limits)
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                          and np.isfinite(data.qacc).all() and np.isfinite(data.ctrl).all()
                          and np.isfinite(data.actuator_force).all())
            equality = equality_audit(model, data)
            max_equality_residual["connect_m"] = max(max_equality_residual["connect_m"],
                                                      equality["max_abs_connect_residual_m"])
            max_equality_residual["joint_mimic_rad"] = max(max_equality_residual["joint_mimic_rad"],
                                                            equality["max_abs_joint_mimic_residual_rad"])
            internal = [row for row in rows if ("spring_link" in row["body1"] and "pad" in row["body2"])
                        or ("pad" in row["body1"] and "spring_link" in row["body2"])]
            clearance = approach_module.clearance_scan(
                model, data, clearance_qualifier, shapes, pair_arrays,
                clearance_qualifier.all_world_shapes(model, data, shapes),
            )
            for group, value in clearance.items():
                gap = value["minimum_clearance_lower_bound_m"]
                if gap is not None:
                    minimum_clearance[group] = min(minimum_clearance[group], float(gap))
                    min_clear[group] = min(min_clear[group], float(gap))
            if approach_clearance:
                failed_clear = [k for k in minimum_clearance if clearance[k]["minimum_clearance_lower_bound_m"] is None
                                or clearance[k]["minimum_clearance_lower_bound_m"] < CLEARANCE_M]
            else:
                failed_clear = [k for k in ("robot_bottle_nonpad", "robot_table", "robot_self")
                                if clearance[k]["minimum_clearance_lower_bound_m"] is None
                                or clearance[k]["minimum_clearance_lower_bound_m"] < CLEARANCE_M]
            unexpected = contacts["unexpected_contacts"]
            clearance_trace_path_obj.write(json.dumps({
                "step": step, "time_s": t, "operation": operation,
                "episode_phase": gate_phase.value if gate_phase else None,
                "approach_clearance_gate": bool(approach_clearance), "groups": clearance,
            }, allow_nan=False) + "\n")

            com = np.asarray(data.xipos[bottle_body_id], dtype=float).copy()
            if previous_com is None:
                com_velocity = np.zeros(3)
            else:
                com_velocity = (com - previous_com) / EXPECTED_DT
            com_acceleration_z = float((com_velocity[2] - previous_com_velocity[2]) / EXPECTED_DT)
            previous_com, previous_com_velocity = com, com_velocity
            balance_residual = (contacts["tool_upward_support_n"] + contacts["table_reaction_vertical_n"]
                                - bottle_mass * GRAVITY - bottle_mass * com_acceleration_z)
            qactual = np.asarray(data.qpos[arm_qpos_ids], dtype=float)
            qvel = np.asarray(data.qvel[arm_dof_ids], dtype=float)
            qerror = qactual - q_target
            tcp_pos = np.asarray(data.site_xpos[tcp_site_id], dtype=float).copy()
            tcp_rot = np.asarray(data.site_xmat[tcp_site_id], dtype=float).reshape(3, 3).copy()
            relative = tcp_rot.T @ (com - tcp_pos)
            if slip_reference_local is None and len(contacts["force_bearing_contact_roles"]) == 2:
                slip_reference_local = relative.copy()
            slip = None if slip_reference_local is None else float(np.linalg.norm(relative - slip_reference_local))
            bottle_quat_wxyz = np.asarray(data.xquat[bottle_body_id], dtype=float)
            bottle_quat_xyzw = (float(bottle_quat_wxyz[1]), float(bottle_quat_wxyz[2]),
                                float(bottle_quat_wxyz[3]), float(bottle_quat_wxyz[0]))
            linear_speed = float(np.linalg.norm(com_velocity))
            angular_speed = float(np.linalg.norm(data.qvel[bottle_dof_id + 3:bottle_dof_id + 6]))
            opening = current_opening()
            actuator_commands = []
            command_by_joint = {row["joint"]: row for row in command_rows}
            for index, name in enumerate(ARM):
                c = command_by_joint[name]
                aid = actuator_by_joint[name]
                actuator_commands.append({
                    "joint": name, "target_qpos_rad": float(q_target[index]),
                    "target_qvel_rad_s": float(qvel_target[index]),
                    "actual_qpos_rad": float(qactual[index]), "actual_qvel_rad_s": float(qvel[index]),
                    "qacc_rad_s2": float(data.qacc[arm_dof_ids[index]]),
                    "tracking_error_rad": float(qerror[index]),
                    "requested_torque_nm": float(c["requested_torque_nm"]),
                    "commanded_torque_nm": float(c["final_torque_nm"]),
                    "actuator_ctrl": float(data.ctrl[aid]), "actuator_force": float(data.actuator_force[aid]),
                    "generalized_actuator_force_nm": float(data.qfrc_actuator[arm_dof_ids[index]]),
                })
                max_torque[actuator_names[aid]] = max(max_torque[actuator_names[aid]], abs(float(data.actuator_force[aid])))
            physics_stream.write(json.dumps({
                "step": step, "time_s": t, "operation": operation,
                "episode_phase": gate_phase.value if gate_phase else None,
                "right_arm": actuator_commands,
                "all_hinge_qpos_rad": data.qpos[hinge_qpos_ids].astype(float).tolist(),
                "all_hinge_qvel_rad_s": data.qvel[hinge_dof_ids].astype(float).tolist(),
                "all_actuator_ctrl": data.ctrl.astype(float).tolist(),
                "all_actuator_force": data.actuator_force.astype(float).tolist(),
                "all_generalized_actuator_force": data.qfrc_actuator.astype(float).tolist(),
                "bottle_root_qpos": data.qpos[bottle_qpos_id:bottle_qpos_id + 7].astype(float).tolist(),
                "bottle_com_world_m": com.tolist(), "bottle_com_velocity_world_m_s": com_velocity.tolist(),
                "bottle_com_acceleration_z_m_s2": com_acceleration_z,
                "bottle_quat_world_xyzw": bottle_quat_xyzw,
                "tcp_world_m": tcp_pos.tolist(), "tcp_rotation_world": tcp_rot.tolist(),
                "tool_relative_bottle_local_m": relative.tolist(),
                "tool_upward_support_n": contacts["tool_upward_support_n"],
                "table_reaction_vertical_n": contacts["table_reaction_vertical_n"],
                "force_balance_residual_n": balance_residual,
                "pad_opening_m": opening, "max_relative_slip_m": slip,
                "follower_constraints": equality, "bottle_qpos_writes_after_start": 0,
            }, allow_nan=False) + "\n")

            if settle_start_time is not None and operation == "INITIAL_SETTLE" and t >= settle_start_time + 0.5:
                settle_support.append(bool(contacts["table_contact"]))
                settle_support_rows.append(float(contacts["table_reaction_vertical_n"]))
                settle_speed.append((linear_speed, angular_speed))
                settle_balance_residual.append(balance_residual)

            actuator_ctrl_bad, actuator_force_bad = [], []
            for aid in range(model.nu):
                if model.actuator_ctrllimited[aid]:
                    lo, hi = map(float, model.actuator_ctrlrange[aid])
                    val = float(data.ctrl[aid])
                    if val < lo - 1e-9 or val > hi + 1e-9:
                        actuator_ctrl_bad.append({"actuator": actuator_names[aid], "ctrl": val, "range": [lo, hi]})
                if model.actuator_forcelimited[aid]:
                    lo, hi = map(float, model.actuator_forcerange[aid])
                    val = float(data.actuator_force[aid])
                    if val < lo - 1e-6 or val > hi + 1e-6:
                        actuator_force_bad.append({"actuator": actuator_names[aid], "force": val, "range": [lo, hi]})
            safety["finite_state"] &= finite
            safety["source_position_limits"] &= not bool(range_errors)
            safety["source_velocity_limits"] &= not bool(velocity_errors)
            safety["actuator_limits"] &= not bool(actuator_ctrl_bad or actuator_force_bad)
            safety["unexpected_contact"] &= not bool(unexpected)
            safety["external_clearance"] &= not bool(failed_clear)
            safety["internal_spring_pad_contact"] &= not bool(internal)
            safety["equality_constraints"] &= equality["pass"]

            min_approach = None
            if gate_phase in (ManipulationPhase.PREGRASP, ManipulationPhase.APPROACH):
                min_approach = min(float(clearance[k]["minimum_clearance_lower_bound_m"])
                                   for k in ("robot_bottle", "robot_table", "robot_self"))
            target_com = target_root + com_offset_world
            frame = EpisodeFrame(
                phase=gate_phase or ManipulationPhase.PREGRASP,
                simulation_time_s=t, physics_step_index=step,
                model_id="X2-PR91-Robotiq-SIMULATION_ONLY_DIAGNOSTIC",
                model_sha256=model_sha,
                bottle_com_world_m=tuple(float(x) for x in com),
                bottle_quat_world_xyzw=bottle_quat_xyzw,
                bottle_linear_speed_m_s=linear_speed,
                bottle_angular_speed_rad_s=angular_speed,
                target_bottle_com_world_m=tuple(float(x) for x in target_com) if gate_phase == ManipulationPhase.PLACE else None,
                force_bearing_contact_roles=frozenset(contacts["force_bearing_contact_roles"]),
                tool_object_contact_active=contacts["tool_object_contact_active"],
                left_hand_contact=contacts["left_hand_contact"],
                table_contact=contacts["table_contact"],
                tool_upward_support_n=contacts["tool_upward_support_n"],
                table_reaction_vertical_n=contacts["table_reaction_vertical_n"],
                max_relative_slip_m=slip,
                minimum_approach_clearance_m=min_approach,
                tool_opening_m=opening,
                source_position_limits_pass=not bool(range_errors),
                source_velocity_limits_pass=not bool(velocity_errors),
                actuator_limits_pass=not bool(actuator_ctrl_bad or actuator_force_bad),
                follower_constraints_pass=equality["pass"],
                bottle_qpos_writes_after_start=0,
                bottle_attachment_active=False, mocap_carry_active=False, hidden_support_active=False,
            )
            if gate_phase is not None:
                stage_frames.append(frame)
                episode_frames_stream.write(json.dumps({
                    "phase": gate_phase.value, "simulation_time_s": t, "physics_step_index": step,
                    "model_sha256": model_sha, "bottle_com_world_m": list(frame.bottle_com_world_m),
                    "bottle_quat_world_xyzw": list(frame.bottle_quat_world_xyzw),
                    "bottle_linear_speed_m_s": linear_speed, "bottle_angular_speed_rad_s": angular_speed,
                    "force_bearing_contact_roles": sorted(frame.force_bearing_contact_roles),
                    "tool_object_contact_active": frame.tool_object_contact_active,
                    "table_contact": frame.table_contact,
                    "tool_upward_support_n": frame.tool_upward_support_n,
                    "table_reaction_vertical_n": frame.table_reaction_vertical_n,
                    "max_relative_slip_m": frame.max_relative_slip_m,
                    "minimum_approach_clearance_m": frame.minimum_approach_clearance_m,
                    "source_position_limits_pass": frame.source_position_limits_pass,
                    "source_velocity_limits_pass": frame.source_velocity_limits_pass,
                    "actuator_limits_pass": frame.actuator_limits_pass,
                    "follower_constraints_pass": frame.follower_constraints_pass,
                    "bottle_qpos_writes_after_start": 0,
                    "bottle_attachment_active": False, "mocap_carry_active": False, "hidden_support_active": False,
                }, allow_nan=False) + "\n")

            if step % 40 == 0 or k == steps - 1:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            safety_detail = None
            if not finite:
                safety_detail = {"gate": "finite_state", "step": step, "operation": operation}
            elif range_errors:
                safety_detail = {"gate": "source_position_limit", "step": step, "operation": operation,
                                 "detail": range_errors[0]}
            elif velocity_errors:
                safety_detail = {"gate": "source_velocity_limit", "step": step, "operation": operation,
                                 "detail": velocity_errors[0]}
            elif actuator_ctrl_bad or actuator_force_bad:
                safety_detail = {"gate": "actuator_limit", "step": step, "operation": operation,
                                 "detail": (actuator_ctrl_bad or actuator_force_bad)[0]}
            elif failed_clear:
                safety_detail = {"gate": "external_clearance_below_2mm", "step": step, "operation": operation,
                                 "groups": failed_clear, "clearance": clearance}
            elif internal:
                safety_detail = {"gate": "internal_spring_pad_contact", "step": step, "operation": operation,
                                 "contact": internal[0]}
            elif not equality["pass"]:
                safety_detail = {"gate": "follower_constraint_residual", "step": step,
                                 "operation": operation, "equality": equality}
            elif unexpected:
                safety_detail = {"gate": "unexpected_contact", "step": step, "operation": operation,
                                 "contact": unexpected[0]}
            if safety_detail is not None:
                first_failure = safety_detail
                break
        return {
            "operation": operation, "episode_phase": gate_phase.value if gate_phase else None,
            "start_time_s": phase_start_time, "duration_s": float(data.time - phase_start_time),
            "steps": total_steps - start_step, "completed": first_failure is None and total_steps - start_step == steps,
            "bilateral_force_bearing_fraction": step_count_bilateral / max(1, total_steps - start_step),
            "force_bearing_contact_steps": contact_force_roles,
            "mean_table_reaction_vertical_n": float(np.mean(table_force_rows)) if table_force_rows else None,
            "minimum_clearance_lower_bounds_m": {k: (None if not math.isfinite(v) else v) for k, v in min_clear.items()},
            "failure": first_failure,
        }

    physics_stream = (raw / "physics_trace.jsonl").open("w", encoding="utf-8", buffering=1)
    contact_trace_path_obj = (raw / "contact_wrench_trace.jsonl").open("w", encoding="utf-8", buffering=1)
    clearance_trace_path_obj = (raw / "clearance_trace.jsonl").open("w", encoding="utf-8", buffering=1)
    episode_frames_stream = (raw / "episode_frames.jsonl").open("w", encoding="utf-8", buffering=1)
    phases_done = []
    planned_cumulative = 0
    settle_start_time = float(data.time)
    try:
        # Initial deterministic bottle settle happens before PREGRASP; it remains a separate diagnostic gate.
        phase = run_segment("INITIAL_SETTLE", [qpre, qpre], 1.0, 0.0, 0.0,
                            approach_clearance=True)
        phases_done.append(phase)
        initial_support_fraction = sum(settle_support) / max(1, len(settle_support))
        settle_speeds = [x[0] for x in settle_speed]
        settle_angular = [x[1] for x in settle_speed]
        initial_settle_pass = bool(initial_support_fraction >= 0.95 and settle_speeds and settle_angular
                                   and max(settle_speeds) <= 0.020 and max(settle_angular) <= 0.050)
        initial_table_mean = float(np.mean(settle_support_rows)) if settle_support_rows else 0.0
        initial_support_sign_check = bool(abs(initial_table_mean - bottle_mass * GRAVITY) <= 1.0)
        if first_failure is None and not initial_settle_pass:
            first_failure = {"gate": "initial_bottle_settle", "support_fraction": initial_support_fraction,
                             "max_linear_speed_m_s": max(settle_speeds, default=None),
                             "max_angular_speed_rad_s": max(settle_angular, default=None)}
        if first_failure is None and not initial_support_sign_check:
            first_failure = {"gate": "contact_force_sign_balance", "mean_table_force_n": initial_table_mean,
                             "bottle_weight_n": bottle_mass * GRAVITY}
        if first_failure is None:
            save_frame("01_pregrasp_open")
            phases_done.append(run_segment("PREGRASP", [qpre, qpre], 0.2, 0.0, 0.0,
                                           ManipulationPhase.PREGRASP, approach_clearance=True))
        if first_failure is None:
            phases_done.append(run_segment("APPROACH", [qpre, qapproach], 4.0, 0.0, 0.0,
                                           ManipulationPhase.APPROACH, approach_clearance=True))
            save_frame("02_approach_open")
        if first_failure is None:
            phases_done.append(run_segment("CLOSE_ACQUISITION", [qapproach, qapproach], 0.75, 0.0, 255.0))
        if first_failure is None:
            phases_done.append(run_segment("CONTACT_HOLD", [qapproach, qapproach], 1.0, 255.0, 255.0,
                                           ManipulationPhase.CONTACT))
            save_frame("03_contact_hold")
            current_gate = evaluate_episode_evidence(stage_frames, criteria)
            if current_gate.stage_status[ManipulationPhase.CONTACT.value] != "PASS":
                first_failure = {"gate": "CONTACT", "episode_gate_status": current_gate.stage_status["CONTACT"],
                                 "blockers": list(current_gate.blockers)}
        # The first 5 mm is a bounded LOAD_TRANSFER probe. No further lift is attempted unless it passes.
        if first_failure is None:
            phases_done.append(run_segment("LOAD_TRANSFER_5MM", [qapproach, lift_q[0]], 0.5, 255.0, 255.0,
                                           ManipulationPhase.LOAD_TRANSFER))
            save_frame("04_load_transfer_5mm")
            current_gate = evaluate_episode_evidence(stage_frames, criteria)
            if current_gate.stage_status[ManipulationPhase.LOAD_TRANSFER.value] != "PASS":
                first_failure = {"gate": "LOAD_TRANSFER", "episode_gate_status": current_gate.stage_status["LOAD_TRANSFER"],
                                 "blockers": list(current_gate.blockers)}

        lift_started = False
        lift_complete = False
        last_lift_target = lift_q[0]
        if first_failure is None:
            previous_q = lift_q[0]
            for waypoint_index in range(1, len(lift_q)):
                # Keep measuring LOAD_TRANSFER until the bottle has actually lost table contact.
                in_air = not detailed_contacts(model, data)["table_contact"]
                gate_phase = ManipulationPhase.LIFT_50MM if in_air else ManipulationPhase.LOAD_TRANSFER
                lift_started |= in_air
                phase = run_segment(f"LIFT_WAYPOINT_{waypoint_index:02d}", [previous_q, lift_q[waypoint_index],
                                    ], 0.25, 255.0, 255.0, gate_phase)
                phases_done.append(phase)
                last_lift_target = lift_q[waypoint_index]
                previous_q = lift_q[waypoint_index]
                if first_failure is not None:
                    break
                current_gate = evaluate_episode_evidence(stage_frames, criteria)
                if current_gate.stage_status[ManipulationPhase.LOAD_TRANSFER.value] != "PASS":
                    first_failure = {"gate": "LOAD_TRANSFER", "episode_gate_status": current_gate.stage_status["LOAD_TRANSFER"],
                                     "blockers": list(current_gate.blockers)}
                    break
                last_frame = stage_frames[-1] if stage_frames else None
                if last_frame and last_frame.phase is ManipulationPhase.LIFT_50MM:
                    rise = last_frame.bottle_com_world_m[2] - stage_frames[0].bottle_com_world_m[2]
                    if rise >= criteria.minimum_lift_m and last_frame.table_contact is False:
                        lift_complete = True
                        break
            if first_failure is None:
                current_gate = evaluate_episode_evidence(stage_frames, criteria)
                if not lift_started or not lift_complete or current_gate.stage_status[ManipulationPhase.LIFT_50MM.value] != "PASS":
                    first_failure = {"gate": "LIFT_50MM", "episode_gate_status": current_gate.stage_status[ManipulationPhase.LIFT_50MM.value],
                                     "lift_started": lift_started, "lift_complete": lift_complete,
                                     "blockers": list(current_gate.blockers)}
        if first_failure is None:
            phases_done.append(run_segment("HOLD_1S", [last_lift_target, last_lift_target], 1.0, 255.0, 255.0,
                                           ManipulationPhase.HOLD))
            save_frame("05_lift_hold")
            current_gate = evaluate_episode_evidence(stage_frames, criteria)
            if current_gate.stage_status[ManipulationPhase.HOLD.value] != "PASS":
                first_failure = {"gate": "HOLD", "episode_gate_status": current_gate.stage_status["HOLD"],
                                 "blockers": list(current_gate.blockers)}

        if first_failure is None:
            # The pinned static route begins at an airborne pose. Execute its source-valid q waypoints
            # with real actuators; actual bottle pose remains entirely dynamic.
            transfer_points = [np.asarray(row["arm_qpos_rad"], dtype=float) for row in route_waypoints]
            phases_done.append(run_segment("TRANSFER_ENTRY", [last_lift_target, transfer_points[0]], 0.35,
                                           255.0, 255.0, ManipulationPhase.TRANSFER))
            previous_q = transfer_points[0]
            for index in range(1, 17):
                if first_failure is not None:
                    break
                phases_done.append(run_segment(f"TRANSFER_{index:02d}", [previous_q, transfer_points[index]], 0.20,
                                               255.0, 255.0, ManipulationPhase.TRANSFER))
                previous_q = transfer_points[index]
            if first_failure is None:
                current_gate = evaluate_episode_evidence(stage_frames, criteria)
                if current_gate.stage_status[ManipulationPhase.TRANSFER.value] != "PASS":
                    first_failure = {"gate": "TRANSFER", "episode_gate_status": current_gate.stage_status["TRANSFER"],
                                     "blockers": list(current_gate.blockers)}
            if first_failure is None:
                save_frame("06_transfer")
                for index in range(17, len(transfer_points)):
                    if first_failure is not None:
                        break
                    phases_done.append(run_segment(f"PLACE_LOWER_{index-16:02d}",
                                                   [previous_q, transfer_points[index]], 0.20,
                                                   255.0, 255.0, ManipulationPhase.PLACE))
                    previous_q = transfer_points[index]
                if first_failure is None:
                    current_gate = evaluate_episode_evidence(stage_frames, criteria)
                    if current_gate.stage_status[ManipulationPhase.PLACE.value] != "PASS":
                        first_failure = {"gate": "PLACE", "episode_gate_status": current_gate.stage_status["PLACE"],
                                         "blockers": list(current_gate.blockers)}
                    else:
                        save_frame("07_place_supported")
            if first_failure is None:
                phases_done.append(run_segment("OPENING", [previous_q, previous_q], 0.75, 255.0, 0.0))
                release_contacts = detailed_contacts(model, data)
                if release_contacts["tool_object_contact_active"] or pad_opening(model, data) < criteria.minimum_release_opening_m:
                    first_failure = {"gate": "RELEASE", "reason": "object contact persists or jaw opening below threshold",
                                     "tool_opening_m": pad_opening(model, data),
                                     "contacts": release_contacts["contacts"]}
                else:
                    phases_done.append(run_segment("RELEASED_FREE_HOLD", [previous_q, previous_q], 0.25,
                                                   0.0, 0.0, ManipulationPhase.RELEASE))
                    save_frame("08_released")
            if first_failure is None:
                phases_done.append(run_segment("FINAL_FREE_SETTLE", [previous_q, previous_q], 3.0,
                                               0.0, 0.0, ManipulationPhase.SETTLE))
                save_frame("09_settled")
    finally:
        writer.close()
        renderer.close()
        physics_stream.close()
        contact_trace_path_obj.close()
        clearance_trace_path_obj.close()
        episode_frames_stream.close()

    gate = evaluate_episode_evidence(stage_frames, criteria)
    all_physical_safety = all(safety.values())
    episode_pass = (gate.status == "PASS" and first_failure is None and all_physical_safety
                    and args.prior_abort_steps + total_steps <= 30000)
    phase_status = dict(gate.stage_status)
    if first_failure and first_failure.get("gate") in phase_status:
        if phase_status[first_failure["gate"]] == "NOT_RUN":
            phase_status[first_failure["gate"]] = "FAIL"
    result = {
        "schema": "robotsim.issue46.full_x2_robotiq_physical_episode.v1",
        "run_id": str(uuid.uuid4()),
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_PHYSICAL_CONTACT_EPISODE",
        "status": "PASS_FULL_EPISODE" if episode_pass else "FAIL_OR_BLOCKED",
        "physics_run_completed": first_failure is None,
        "physics_steps": total_steps, "prior_ordinary_runner_abort_steps": args.prior_abort_steps,
        "cumulative_episode_b_physics_steps": args.prior_abort_steps + total_steps,
        "physics_duration_s": float(data.time),
        "prior_aborted_attempts": [
            {"path": "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-full-x2-episode-20261024T082335Z",
             "physics_steps": 3,
             "reason": "mixed-unit equality residual diagnostic compared connect position meters and joint mimic radians to one scalar gate"},
            {"path": "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-full-x2-episode-20261024T082915Z",
             "physics_steps": 1000,
             "reason": "runner TypeError while summarizing scalar table-reaction samples after initial settle"},
        ],
        "first_failure": first_failure,
        "experiment_gate": {
            "episode_gate_status": gate.status,
            "stage_status": phase_status,
            "blockers": list(gate.blockers),
            "last_post_step_index": gate.last_physics_step_index,
            "all_dynamic_safety_gates_pass": all_physical_safety,
            "safety_gates": safety,
        },
        "phases": phases_done,
        "source_joint_limit_audit": source_limits,
        "model_identity": {
            "model_path": str(MODEL), "model_sha256": model_sha,
            "x2_source": historical["identity"]["x2_checkout"],
            "robotiq_source": historical["identity"]["menagerie_checkout"],
            "x2_mjcf_sha256": historical["identity"]["x2_mjcf_sha256"],
            "robotiq_mjcf_sha256": historical["identity"]["robotiq_xml_sha256"],
            "native_mujoco_library_sha256": historical["identity"]["native_library_sha256"],
            "controller_source_snapshot_path": str(snapshot_path),
            "controller_source_snapshot_sha256": sha256(snapshot_path),
            "historical_recorded_runner_sha256": historical["runner"]["sha256"],
            "historical_full_runner_hash_matches_available_source": internal["controller_semantics"]["controller_source_audit"]["runner_file_hash_matches_historical"],
            "controller_helpers_match_clean_source_head": internal["controller_semantics"]["controller_source_audit"]["controller_helpers_match_clean_head"],
            "official_wrist_mount": historical["identity"]["official_x2_tool_mount"],
            "simulation_only_adapter": historical["adapter"],
            "collision_exclusions": sorted([list(x) for x in exclusions]),
        },
        "physical_measurements": {
            "bottle_mass_kg": bottle_mass,
            "bottle_freejoint_qpos_initial": bottle_qpos_initial.tolist(),
            "bottle_qpos_writes_after_start": 0,
            "bottle_attachment_active": False, "mocap_carry_active": False, "hidden_support_active": False,
            "left_hand_assistance": False,
            "minimum_clearance_lower_bounds_m": {k: (None if not math.isfinite(v) else v)
                                                  for k, v in minimum_clearance.items()},
            "maximum_equality_residuals": max_equality_residual,
            "maximum_actuator_force_by_name": max_torque,
            "initial_settle_table_contact_fraction": sum(settle_support) / max(1, len(settle_support)),
            "initial_settle_mean_table_reaction_n": initial_table_mean,
            "initial_settle_mean_force_balance_residual_n": float(np.mean(settle_balance_residual)) if settle_balance_residual else None,
            "initial_settle_max_linear_speed_m_s": max(settle_speeds, default=None),
            "initial_settle_max_angular_speed_rad_s": max(settle_angular, default=None),
            "lift_path_waypoints": len(lift_plan),
            "transfer_path_waypoints": len(route_waypoints),
        },
        "gate_criteria": {
            "bilateral_force_bearing_contact_fraction": criteria.minimum_contact_fraction,
            "minimum_tool_upward_support_n": criteria.minimum_tool_upward_support_n,
            "minimum_table_unloading_n": criteria.minimum_table_unloading_n,
            "maximum_relative_slip_m": criteria.maximum_relative_slip_m,
            "minimum_external_approach_clearance_m": criteria.minimum_approach_clearance_m,
            "minimum_lift_m": criteria.minimum_lift_m,
            "minimum_hold_duration_s": criteria.minimum_hold_duration_s,
            "minimum_release_opening_m": criteria.minimum_release_opening_m,
            "minimum_settle_duration_s": criteria.minimum_settle_duration_s,
            "maximum_settle_linear_speed_m_s": criteria.maximum_settle_linear_speed_m_s,
            "maximum_settle_angular_speed_rad_s": criteria.maximum_settle_angular_speed_rad_s,
        },
        "input_hashes": {
            "approach_a_result_sha256": sha256(APPROACH_A), "static_approach_result_sha256": sha256(STATIC),
            "internal_preflight_result_sha256": sha256(INTERNAL), "historical_result_sha256": sha256(HISTORICAL),
            "lift_ik_sha256": sha256(LIFT_IK), "transfer_preflight_sha256": sha256(TRANSFER_PREFLIGHT),
        },
        "evidence": {
            "physics_trace": str(physics_trace_path), "contact_wrench_trace": str(contact_trace_path),
            "clearance_trace": str(clearance_trace_path), "episode_gate_frames": str(episode_frames_path),
            "episode_gate_result": str(out / "episode_gate.json"), "video": str(video_path),
            "images_dir": str(images), "report": str(out / "REPORT.md"),
        },
        "output": output.as_dict(),
        "duration_wall_s": time.monotonic() - total_wall_start,
    }
    pg.write_new_text(out / "episode_gate.json", json.dumps({"status": gate.status,
        "stage_status": phase_status, "blockers": list(gate.blockers),
        "last_physics_step_index": gate.last_physics_step_index,
        "criteria": result["gate_criteria"]}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        authorized_roots=(str(REVIEWS),), environment="wsl")
    report = [
        "# Full-X2 Robotiq physical grasp episode",
        "",
        f"Run ID: {result['run_id']}",
        f"Outcome: {result['status']}",
        f"EpisodeGate: {gate.status}",
        f"Physics: {total_steps} steps, {data.time:.3f} s at 1 ms",
        f"Cumulative Experiment B steps including the ordinary diagnostic abort: {args.prior_abort_steps + total_steps}/30000.",
        f"First failure: {first_failure}",
        "",
        "## Stage status",
        "",
        *[f"- {name}: {status}" for name, status in phase_status.items()],
        "",
        "## Physical measurement",
        "",
        f"- Bottle mass: {bottle_mass:.6f} kg; freejoint initialized once; post-start qpos writes: 0.",
        f"- Initial table-reaction mean: {initial_table_mean:.6f} N for a {bottle_mass * GRAVITY:.6f} N bottle.",
        f"- Initial force-balance residual mean: {float(np.mean(settle_balance_residual)) if settle_balance_residual else math.nan:.6f} N.",
        f"- Tool upward support, table unloading, slip, bottle COM rise: see per-step traces and EpisodeGate frames.",
        f"- Minimum external clearance lower bounds: {result['physical_measurements']['minimum_clearance_lower_bounds_m']}.",
        "- MuJoCo contact-frame force was rotated using the documented geom1-to-geom2 normal and equal/opposite body forces.",
        "- A dynamic safety or EpisodeGate failure stops the remaining phases; no retry is performed.",
        "",
        "## Reproduction",
        "",
        f"`cd {ROOT} && PYTHONDONTWRITEBYTECODE=1 {sys.executable} {Path(__file__).resolve()} --output-parent {args.output_parent} --prior-abort-steps {args.prior_abort_steps}`",
        "",
    ]
    pg.write_new_text(out / "REPORT.md", "\n".join(report), authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(out / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(out, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "episode_gate_status": gate.status, "stage_status": phase_status,
                      "first_failure": first_failure, "physics_steps": total_steps,
                      "physics_duration_s": float(data.time), "safety_gates": safety,
                      "minimum_clearance_m": result["physical_measurements"]["minimum_clearance_lower_bounds_m"],
                      "evidence_dir": output.as_dict(), "artifact_count": len(manifest["artifacts"])},
                     indent=2, ensure_ascii=False))
    return 0 if episode_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
