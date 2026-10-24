#!/usr/bin/env python3
"""Static-only source-model audit of the mounted 2F-85 internal linkage range."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
REVIEWS = Path("/home/lzy18001500226/projects/RobotSim/reviews")
EXPECTED_MODEL_SHA = "a5dcd589fedaea5645ccf29b708b560f2e0dc923c8955a7d5901a996df57be94"
PASSIVE_JOINTS = (
    "rq_right_coupler_joint", "rq_right_spring_link_joint", "rq_right_follower_joint",
    "rq_left_coupler_joint", "rq_left_spring_link_joint", "rq_left_follower_joint",
)
RIGHT_ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)

sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import path_governance as pg

if __package__:
    from . import qualify_robotiq_static_approach as geom
else:
    import qualify_robotiq_static_approach as geom


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(model, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or f"unnamed_{int(object_id)}"


def joint_ids(model, joint_names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in joint_names]
    if any(joint_id < 0 for joint_id in ids):
        raise ValueError("compiled full-X2 model is missing a required hand joint")
    qids = [int(model.jnt_qposadr[joint_id]) for joint_id in ids]
    return ids, qids


def compile_collision_shapes(model) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    shapes = {}
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
            shape = geom.convex_shape(model, geom_id, cache)
            if shape is not None:
                shapes[geom_id] = shape
    return shapes


def audit_historical_controller(model, historical: dict) -> dict:
    runner_path = Path(historical["runner"]["path"])
    if not runner_path.is_file():
        return {"available": False, "reason": "recorded runner source path is missing"}
    runner_sha = sha256(runner_path)
    runner_text = runner_path.read_text(encoding="utf-8")
    syntax = ast.parse(runner_text)
    names = {"obj_name", "controller_config", "apply_controller"}
    functions = [node for node in syntax.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    if {node.name for node in functions} != names:
        return {"available": False, "reason": "required controller functions are missing from the recorded source"}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    extracted = ast.Module(body=[future, *functions], type_ignores=[])
    namespace = {"mujoco": mujoco, "np": np, "math": math, "Any": Any,
                 "ARM": RIGHT_ARM, "__name__": "issue46_controller_audit"}
    exec(compile(ast.fix_missing_locations(extracted), str(runner_path), "exec"), namespace)
    reproduced = namespace["controller_config"](model)
    recorded = historical["controller"]["actuator_mapping"]
    keys = ("actuator_id", "actuator", "joint_id", "joint", "qpos_id", "dof_id", "gear",
            "kp_simulation_derived", "kd_simulation_derived", "torque_min_nm", "torque_max_nm")
    mismatches = []
    if len(reproduced) != len(recorded):
        mismatches.append({"field": "actuator_count", "current": len(reproduced), "recorded": len(recorded)})
    for index, (current, prior) in enumerate(zip(reproduced, recorded)):
        for key in keys:
            a, b = current.get(key), prior.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                equal = abs(float(a) - float(b)) <= 1e-9
            else:
                equal = a == b
            if not equal:
                mismatches.append({"index": index, "field": key, "current": a, "historical": b})
    apply_node = next(node for node in functions if node.name == "apply_controller")
    controller_source = ast.get_source_segment(runner_text, apply_node) or ""
    required_terms = (
        "data.qfrc_bias", "kp_simulation_derived", "kd_simulation_derived",
        "requested = bias + pd", "np.clip(requested",
    )
    equation_terms_present = all(term in controller_source for term in required_terms)
    source_root = runner_path.parents[2]
    relative_runner = runner_path.relative_to(source_root)
    committed_source = subprocess.check_output(
        ["git", "-C", str(source_root), "show", f"HEAD:{relative_runner.as_posix()}"], text=True
    )
    committed_ast = ast.parse(committed_source)
    helper_names = ("obj_name", "controller_config", "apply_controller", "set_target_dict", "quintic")

    def helper_hashes(source, tree):
        hashes = {}
        for helper_name in helper_names:
            node = next((item for item in tree.body
                         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == helper_name), None)
            if node is not None:
                function_text = ast.get_source_segment(source, node) or ""
                hashes[helper_name] = hashlib.sha256(function_text.encode("utf-8")).hexdigest()
        return hashes

    current_helper_hashes = helper_hashes(runner_text, syntax)
    committed_helper_hashes = helper_hashes(committed_source, committed_ast)
    source_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    helpers_match_clean_head = current_helper_hashes == committed_helper_hashes
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--", str(runner_path.relative_to(source_root))],
            text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        dirty = None
    return {
        "available": True,
        "recorded_runner_path": str(runner_path),
        "recorded_runner_sha256": historical["runner"]["sha256"],
        "current_runner_file_sha256": runner_sha,
        "runner_file_hash_matches_historical": runner_sha == historical["runner"]["sha256"],
        "source_checkout_dirty_for_runner_file": dirty,
        "current_source_checkout_head": source_commit,
        "controller_helper_hashes": current_helper_hashes,
        "controller_helpers_match_clean_head": helpers_match_clean_head,
        "historical_actuator_mapping_count": len(recorded),
        "current_actuator_mapping_count": len(reproduced),
        "historical_actuator_mapping_matches_compiled_model": not mismatches,
        "mapping_mismatch_count": len(mismatches), "mapping_mismatch_examples": mismatches[:20],
        "apply_controller_source_sha256": hashlib.sha256(controller_source.encode("utf-8")).hexdigest(),
        "bounded_pd_plus_bias_equation_terms_present": equation_terms_present,
        "current_controller_source_note": "The full historical runner file hash differs from the available file. The current helper functions are compared with the clean source checkout HEAD, and their mapped gains/limits are compared against recorded historical metadata. Record the selected implementation and helper hashes for any new experiment; retain SIMULATION_ONLY_DIAGNOSTIC classification.",
    }


def equality_residual(model, data) -> tuple[np.ndarray, list[dict]]:
    eq_value = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = np.flatnonzero(np.asarray(data.efc_type[:data.nefc]) == eq_value)
    residual = np.asarray(data.efc_pos[rows], dtype=float).copy()
    details = []
    for row in rows:
        constraint_id = int(data.efc_id[row])
        if not 0 <= constraint_id < model.neq:
            continue
        details.append({
            "efc_row": int(row), "equality_id": constraint_id,
            "equality_type": int(model.eq_type[constraint_id]),
            "equality_obj1_id": int(model.eq_obj1id[constraint_id]),
            "equality_obj2_id": int(model.eq_obj2id[constraint_id]),
            "residual": float(data.efc_pos[row]),
        })
    return residual, details


def solve_closed_state(model, data, driver_q: float, passive_joint_ids: list[int], passive_qids: list[int],
                       driver_joint_ids: list[int], driver_qids: list[int], seed: np.ndarray) -> dict:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[driver_qids] = driver_q
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in passive_joint_ids], dtype=float)
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in passive_joint_ids], dtype=float)
    margin = 1e-9
    low = np.where(model.jnt_limited[passive_joint_ids], lower + margin, -np.inf)
    high = np.where(model.jnt_limited[passive_joint_ids], upper - margin, np.inf)
    x0 = np.minimum(np.maximum(seed, low), high)
    calls = 0

    def residual(x):
        nonlocal calls
        calls += 1
        data.qpos[passive_qids] = x
        mujoco.mj_forward(model, data)
        values, _ = equality_residual(model, data)
        return values

    solved = least_squares(residual, x0, bounds=(low, high), xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=500, x_scale="jac")
    data.qpos[passive_qids] = solved.x
    data.qpos[driver_qids] = driver_q
    mujoco.mj_forward(model, data)
    residual_values, equality_details = equality_residual(model, data)
    all_equalities_satisfied = len(residual_values) == 7 and float(np.max(np.abs(residual_values))) <= 1e-6

    joint_state = {}
    position_violations = []
    for joint_id in driver_joint_ids + passive_joint_ids:
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        q = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        lower_j, upper_j = map(float, model.jnt_range[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        joint_state[joint_name] = {"qpos_rad": q, "source_compiled_range_rad": [lower_j, upper_j], "limited": limited}
        if limited and not lower_j - 1e-9 <= q <= upper_j + 1e-9:
            position_violations.append({"joint": joint_name, "qpos_rad": q, "range_rad": [lower_j, upper_j]})

    equality_names = []
    for equality_id in range(model.neq):
        obj1 = int(model.eq_obj1id[equality_id])
        obj2 = int(model.eq_obj2id[equality_id])
        type_id = int(model.eq_type[equality_id])
        if type_id == int(mujoco.mjtEq.mjEQ_CONNECT):
            obj_type = mujoco.mjtObj.mjOBJ_BODY
        elif type_id == int(mujoco.mjtEq.mjEQ_JOINT):
            obj_type = mujoco.mjtObj.mjOBJ_JOINT
        else:
            obj_type = mujoco.mjtObj.mjOBJ_BODY
        equality_names.append({"equality_id": equality_id, "type": type_id,
                               "obj1": name(model, obj_type, obj1), "obj2": name(model, obj_type, obj2)})

    return {
        "driver_q_rad": float(driver_q),
        "solver_success": bool(solved.success), "solver_message": str(solved.message),
        "solver_function_evaluations": int(solved.nfev), "mj_forward_calls": calls + 1,
        "equality_constraint_count": len(residual_values),
        "max_abs_equality_residual": float(np.max(np.abs(residual_values))) if residual_values.size else None,
        "equality_rows": equality_details, "equality_names": equality_names,
        "all_equalities_satisfied": all_equalities_satisfied,
        "source_joint_position_limits_pass": not position_violations,
        "position_violations": position_violations,
        "joint_state": joint_state, "passive_seed_rad": seed.tolist(),
        "passive_solution_rad": solved.x.tolist(),
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def run_range(model_path: Path, historical_result_path: Path, static_result_path: Path) -> dict:
    model_hash = sha256(model_path)
    if model_hash != EXPECTED_MODEL_SHA:
        raise ValueError("diagnostic full-X2 model SHA mismatch")
    history_bytes = historical_result_path.read_bytes()
    historical = json.loads(history_bytes)
    static_bytes = static_result_path.read_bytes()
    static = json.loads(static_bytes)
    if historical.get("status") != "PASS":
        raise ValueError("expected the exact historical runner's own recorded PASS")
    if static.get("input_identity", {}).get("diagnostic_model_sha256") != model_hash:
        raise ValueError("static candidate and internal-range audit model identities differ")
    if static.get("legacy_minimum_reconciliation") is None:
        raise ValueError("static candidate result lacks the sample-20 same-pose distance audit")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if abs(float(model.opt.timestep) - 0.001) > 1e-12:
        raise ValueError("compiled model timestep differs from the accepted 1 ms baseline")
    data = mujoco.MjData(model)
    source_range = historical["stages"]["mounted_open_close"]["motion"]
    observed_min = float(source_range["driver_qpos_min_rad"])
    observed_max = float(source_range["driver_qpos_max_rad"])
    driver_joint_names = ("rq_right_driver_joint", "rq_left_driver_joint")
    driver_joint_ids, driver_qids = joint_ids(model, driver_joint_names)
    passive_joint_ids, passive_qids = joint_ids(model, PASSIVE_JOINTS)
    arm_joint_ids, arm_qids = joint_ids(model, RIGHT_ARM)
    source_range_mismatches = []
    for joint_name, source in historical["identity"]["urdf_joint_limits"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            source_range_mismatches.append({"joint": joint_name, "reason": "missing_compiled_joint"})
            continue
        compiled = model.jnt_range[joint_id]
        if (abs(float(source["lower"]) - float(compiled[0])) > 1e-5
                or abs(float(source["upper"]) - float(compiled[1])) > 1e-5):
            source_range_mismatches.append({"joint": joint_name,
                                            "source_range_rad": [float(source["lower"]), float(source["upper"])],
                                            "compiled_range_rad": compiled.tolist()})
    source_ranges_match = not source_range_mismatches
    controller_audit = audit_historical_controller(model, historical)
    q_pre = np.asarray(static["candidate_25mm_retreat"]["ik"]["joint_positions_rad"], dtype=float)
    if q_pre.shape != (len(arm_qids),):
        raise ValueError("saved 25 mm pregrasp qpos has an invalid right-arm vector")

    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "rq_split")
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
    if min(tendon_id, actuator_id) < 0:
        raise ValueError("pinned model is missing the source Menagerie gripper tendon/actuator")
    sample_driver_q = np.linspace(0.0, observed_max, 9).tolist()
    if abs(sample_driver_q[0] - observed_min) > 5e-5:
        raise ValueError("observed historical open endpoint differs from zero beyond recorded tolerance")

    shapes = compile_collision_shapes(model)
    exclusions = geom.load_exclusions(model_path)
    pairs = geom.model_path_pairs(model, list(shapes), exclusions)
    internal_pairs = pairs["internal_mechanism"]
    if not internal_pairs:
        raise ValueError("no enabled spring-link/pad collision pairs were found")
    seed = np.asarray([model.qpos0[qid] for qid in passive_qids], dtype=float)
    q_range_rows = []
    per_pair_rows: dict[tuple[str, str, str, str], list[dict]] = {}
    total_forward = 0
    max_root_limit_violation = 0.0
    base_contacts = []
    for index, driver_q in enumerate(sample_driver_q):
        data.qpos[:] = model.qpos0
        data.qpos[arm_qids] = q_pre
        solved = solve_closed_state(model, data, float(driver_q), passive_joint_ids, passive_qids,
                                    driver_joint_ids, driver_qids, seed)
        total_forward += solved["mj_forward_calls"]
        seed = np.asarray(solved["passive_solution_rad"], dtype=float)
        root_hi = max(float(model.jnt_range[joint_id, 1]) for joint_id in driver_joint_ids)
        root_low = min(float(model.jnt_range[joint_id, 0]) for joint_id in driver_joint_ids)
        if not root_low - 1e-9 <= driver_q <= root_hi + 1e-9:
            max_root_limit_violation = max(max_root_limit_violation, max(root_low - driver_q, driver_q - root_hi))
        contact_map = geom.contact_map_for_data(data)
        if index == 0:
            base_contacts = [
                {"geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                 "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom1)])),
                 "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                 "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom2)])),
                 "distance_m": float(contact.dist)}
                for contact in data.contact[:data.ncon]
            ]
        pair_clearances = []
        world_shapes = geom.all_world_shapes(model, data, shapes)
        for first, second in internal_pairs:
            measured = geom.geom_pair_lower_bound(model, first, second, world_shapes, contact_map)
            row = {
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first), "body1": geom.body_name(model, first),
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second), "body2": geom.body_name(model, second),
                "clearance_lower_bound_m": float(measured["clearance_m"]),
                "clearance_lower_bound_mm": float(measured["clearance_m"] * 1000.0),
                "method": measured["method"], "active_contact": bool(contact_map.get(geom.pair_key(first, second))),
                "mujoco_geom_distance_m": float(mujoco.mj_geomDistance(model, data, first, second, 0.1, None)),
            }
            key = (row["body1"], row["geom1"], row["body2"], row["geom2"])
            per_pair_rows.setdefault(key, []).append(row)
            pair_clearances.append(row)
        q_range_rows.append({
            "sample_index": index, "driver_q_rad": float(driver_q),
            "menagerie_actuator_ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "menagerie_actuator_forcerange": model.actuator_forcerange[actuator_id].tolist(),
            "root_joint_ranges_rad": {joint: model.jnt_range[jid].tolist()
                                       for joint, jid in zip(driver_joint_names, driver_joint_ids)},
            "closure": solved,
            "internal_pairs": pair_clearances,
            "active_internal_contact_pairs": [row for row in pair_clearances if row["active_contact"]],
        })

    internal_summary = []
    for key, items in per_pair_rows.items():
        values = [float(item["clearance_lower_bound_m"]) for item in items]
        internal_summary.append({
            "body1": key[0], "geom1": key[1], "body2": key[2], "geom2": key[3],
            "minimum_clearance_lower_bound_m": min(values),
            "minimum_clearance_lower_bound_mm": min(values) * 1000.0,
            "maximum_clearance_lower_bound_m": max(values),
            "variation_m": max(values) - min(values),
            "active_contact_any_sample": any(item["active_contact"] for item in items),
            "mujoco_geom_distance_min_m": min(float(item["mujoco_geom_distance_m"]) for item in items),
            "mujoco_geom_distance_max_m": max(float(item["mujoco_geom_distance_m"]) for item in items),
            "measurement": items[0]["method"], "sample_count": len(items),
        })

    closure_pass = all(row["closure"]["solver_success"] and row["closure"]["all_equalities_satisfied"]
                       and row["closure"]["finite_state"] for row in q_range_rows)
    joint_limits_pass = all(row["closure"]["source_joint_position_limits_pass"] for row in q_range_rows)
    no_internal_penetration = all(
        row["clearance_lower_bound_m"] >= 0.0 and not row["active_contact"]
        for sample in q_range_rows for row in sample["internal_pairs"]
    )
    q_range_from_source = [float(model.jnt_range[jid, 0]) for jid in driver_joint_ids]
    q_range_to_source = [float(model.jnt_range[jid, 1]) for jid in driver_joint_ids]
    source_driver_range_pass = all(
        q_range_from_source[index] - 1e-9 <= q <= q_range_to_source[index] + 1e-9
        for q in sample_driver_q for index in range(len(driver_joint_ids))
    )
    return {
        "schema": "robotsim.issue46.robotiq_internal_range_preflight.v1",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_STATIC_LINKAGE_RANGE",
        "physics_steps": 0,
        "mj_forward_calls": total_forward,
        "input_identity": {
            "model_path": str(model_path), "model_sha256": model_hash,
            "historical_result_path": str(historical_result_path), "historical_result_sha256": hashlib.sha256(history_bytes).hexdigest(),
            "static_qualification_path": str(static_result_path), "static_qualification_sha256": hashlib.sha256(static_bytes).hexdigest(),
            "x2_commit": historical["identity"]["x2_checkout"]["head"],
            "x2_mjcf_sha256": historical["identity"]["x2_mjcf_sha256"],
            "robotiq_commit": historical["identity"]["menagerie_checkout"]["head"],
            "robotiq_mjcf_sha256": historical["identity"]["robotiq_xml_sha256"],
            "runner_path": historical["runner"]["path"],
            "runner_sha256": historical["runner"]["sha256"],
            "mujoco_version": mujoco.__version__,
        },
        "controller_semantics": {
            "controller_source": "historical pinned issue46_x2_robotiq_m0.py controller_config/apply_controller; not executed in this static preflight",
            "arm_controller": historical["controller"]["model"],
            "controller_source_audit": controller_audit,
            "arm_gains_simulation_derived": {row["joint"]: {"kp": row["kp_simulation_derived"], "kd": row["kd_simulation_derived"]}
                                              for row in historical["controller"]["actuator_mapping"]
                                              if row["joint"] in RIGHT_ARM},
            "gripper_actuator_name": "rq_fingers_actuator",
            "gripper_actuator_ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "gripper_actuator_forcerange": model.actuator_forcerange[actuator_id].tolist(),
            "tendon_name": name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id),
            "root_joint_names": list(driver_joint_names),
            "source_observed_root_q_range_rad": [observed_min, observed_max],
            "root_source_joint_ranges_rad": {joint: model.jnt_range[jid].tolist()
                                             for joint, jid in zip(driver_joint_names, driver_joint_ids)},
        },
        "static_pose": {
            "right_arm_qpos_rad": q_pre.tolist(),
            "bottle_qpos": "model qpos0; freejoint unchanged in each independent static MjData state",
            "new_collision_exclusions": [],
            "loaded_collision_exclusions": [list(pair) for pair in sorted(exclusions)],
        },
        "internal_range": {
            "root_q_samples_rad": sample_driver_q,
            "source_root_range_rad": {joint: model.jnt_range[jid].tolist()
                                       for joint, jid in zip(driver_joint_names, driver_joint_ids)},
            "passive_joints": list(PASSIVE_JOINTS),
            "passive_joint_ranges_rad": {joint: model.jnt_range[jid].tolist()
                                         for joint, jid in zip(PASSIVE_JOINTS, passive_joint_ids)},
            "closure_residual_gate_rad": 1e-6,
            "maximum_closure_residual_rad": max(float(row["closure"]["max_abs_equality_residual"] or 0.0)
                                                 for row in q_range_rows),
            "equality_constraint_names": q_range_rows[0]["closure"]["equality_names"],
            "per_sample": q_range_rows,
            "per_pair_summary": internal_summary,
            "minimum_source_observed_root_q_rad": observed_min,
            "maximum_source_observed_root_q_rad": observed_max,
            "source_limit_extents_rad": {"lower": q_range_from_source, "upper": q_range_to_source},
        },
        "gates": {
            "exact_model_identity": True,
            "x2_urdf_joint_ranges_match_compiled_model": source_ranges_match,
            "source_range_mismatches": source_range_mismatches,
            "historical_actuator_mapping_reproduced": bool(controller_audit.get("historical_actuator_mapping_matches_compiled_model")),
            "bounded_pd_feedforward_formula_identified": bool(controller_audit.get("bounded_pd_plus_bias_equation_terms_present")),
            "controller_helpers_match_clean_source_head": bool(controller_audit.get("controller_helpers_match_clean_head")),
            "robotiq_menagerie_model_pinned_by_compiled_mjcf_sha": True,
            "source_driver_range_covered_from_open_through_observed_max": (
                sample_driver_q[0] <= observed_min + 5e-5 and sample_driver_q[-1] >= observed_max - 1e-8
            ),
            "sampled_driver_positions_within_source_ranges": source_driver_range_pass,
            "closed_loop_static_solver_converged": closure_pass,
            "all_sampled_joint_position_limits_pass": joint_limits_pass,
            "no_active_or_negative_clearance_internal_collision_over_sampled_range": no_internal_penetration,
            "2mm_external_approach_static_gate": bool(static["gates"].get("candidate_robot_bottle_clearance_ge_2mm")
                                                       and static["gates"].get("candidate_nonpad_robot_bottle_clearance_ge_2mm")
                                                       and static["gates"].get("candidate_robot_table_clearance_ge_2mm")
                                                       and static["gates"].get("candidate_robot_self_clearance_ge_2mm")),
            "separate_internal_2mm_clearance_margin": "NOT_APPLIED; user task requires source/mechanism check before applying external margin",
        },
        "static_initial_contacts": base_contacts,
        "mechanical_scope": {
            "active_internal_collision_pairs_checked": len(internal_pairs),
            "source_spring_mesh": "/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie/robotiq_2f85/assets/spring_link.stl",
            "spring_mesh_sha256": historical["identity"]["robotiq_asset_sha256"]["spring_link.stl"],
            "pads_use_collision_boxes": True,
            "render_only_silicone_pad_is_not_used_as_contact_geometry": True,
            "model_visual_meshes_collision_masks_joints_couplings_and_actuator_limits_modified": False,
            "no_grasp_or_bottle_contact_simulation": True,
        },
        "status": "PASS_STATIC_INTERNAL_RANGE" if all((source_ranges_match,
                                                        controller_audit.get("historical_actuator_mapping_matches_compiled_model", False),
                                                        controller_audit.get("bounded_pd_plus_bias_equation_terms_present", False),
                                                        controller_audit.get("controller_helpers_match_clean_head", False),
                                                        source_driver_range_pass,
                                                        closure_pass, joint_limits_pass, no_internal_penetration))
                  else "BLOCKED_STATIC_INTERNAL_RANGE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-model", required=True)
    parser.add_argument("--historical-result", required=True)
    parser.add_argument("--static-result", required=True)
    parser.add_argument("--output-parent", default=str(REVIEWS / "issue-46"))
    args = parser.parse_args()
    model = Path(args.diagnostic_model)
    historical = Path(args.historical_result)
    static = Path(args.static_result)
    for path in (model, historical, static):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")
    result = run_range(model, historical, static)
    output_parent = pg.validate_output_path(args.output_parent, authorized_roots=(str(REVIEWS),),
                                            require_exists=True, expected_kind="directory", environment="wsl")
    output = pg.create_unique_output_directory(output_parent.wsl_path, prefix="robotiq-internal-range-preflight",
                                               authorized_roots=(str(REVIEWS),), environment="wsl")
    result["run_id"] = str(uuid.uuid4())
    result["runner_path"] = str(Path(__file__).resolve())
    result["runner_sha256"] = sha256(Path(__file__).resolve())
    result["output"] = output.as_dict()
    report = [
        "# Full-X2 Robotiq internal operating-range preflight",
        "",
        f"Run ID: {result['run_id']}",
        f"Decision: {result['status']}",
        "Physics steps: 0",
        f"Static mj_forward calls: {result['mj_forward_calls']}",
        f"Sampled root q range: {result['internal_range']['root_q_samples_rad'][0]:.6f} to {result['internal_range']['root_q_samples_rad'][-1]:.6f} rad",
        f"Maximum equality residual: {result['internal_range']['maximum_closure_residual_rad']:.3e} rad",
        f"Closed-loop state gate: {result['gates']['closed_loop_static_solver_converged']}",
        f"Joint-limit gate: {result['gates']['all_sampled_joint_position_limits_pass']}",
        f"Internal nonpenetration/contact gate: {result['gates']['no_active_or_negative_clearance_internal_collision_over_sampled_range']}",
        "",
        "## Internal spring/pad clearance",
        "",
    ]
    report.extend(
        f"- {row['body1']}/{row['geom1']} to {row['body2']}/{row['geom2']}: minimum conservative lower bound {row['minimum_clearance_lower_bound_mm']:.6f} mm; active contact={row['active_contact_any_sample']}; sampled states={row['sample_count']}."
        for row in result["internal_range"]["per_pair_summary"]
    )
    report.extend([
        "",
        "The 2 mm environmental clearance target is kept separate from internal mechanism clearance. This preflight reports the active compiled collision proxy distances, source joint ranges, static closure residuals, and active contacts across the recorded operating interval. It does not alter geometry, collision masks, equality relations, limits, or actuator parameters, and it does not simulate grasp or bottle contact.",
        "",
        "## Reproduction",
        "",
        "See REPRODUCE.md. This command runs only static MjData assignments and mj_forward; no mj_step is called.",
        "",
    ])
    reproduce = [
        f"cd {ROOT}",
        f"PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_robotiq_internal_range.py --diagnostic-model {model} --historical-result {historical} --static-result {static}",
    ]
    pg.write_new_text(output.native_path / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPORT.md", "\n".join(report), authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPRODUCE.md", "\n".join(reproduce) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(output.native_path, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "physics_steps": 0,
                      "gates": result["gates"], "output": output.as_dict(),
                      "manifest_count": len(manifest["artifacts"])}, indent=2))
    return 0 if result["status"] == "PASS_STATIC_INTERNAL_RANGE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
