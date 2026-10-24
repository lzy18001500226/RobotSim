#!/usr/bin/env python3
"""One bounded, guarded full-X2 Robotiq dynamic approach experiment (Issue 46)."""
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
DEFAULT_MODEL = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/"
    "x2_robotiq_2f85_simulation_only.xml"
)
DEFAULT_HISTORICAL = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/result.json"
)
DEFAULT_STATIC = REVIEWS / "issue-46/robotiq-arm-approach-qualified-20261024T065752Z/result.json"
DEFAULT_INTERNAL = REVIEWS / "issue-46/robotiq-internal-range-preflight-20261024T073206Z/result.json"
DEFAULT_CONTROLLER = Path(
    "/home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009/scripts/research/"
    "issue46_x2_robotiq_m0.py"
)
EXPECTED_MODEL_SHA = "a5dcd589fedaea5645ccf29b708b560f2e0dc923c8955a7d5901a996df57be94"
EXPECTED_DT = 0.001
CLEARANCE_M = 0.002
ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)

sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import path_governance as pg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned helper snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geom_name(model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"unnamed_{geom_id}"


def body_name(model, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[int(geom_id)])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def expected_contact(row: dict) -> bool:
    b1, b2 = row["body1"], row["body2"]
    g1, g2 = row["geom1"], row["geom2"]
    bodies = (b1, b2)
    geoms = (g1, g2)
    if any(body.startswith("m0_bottle") for body in bodies) and any(body.startswith("m0_table") for body in bodies):
        return True
    if ("m0_floor" in geoms or "floor" in geoms):
        other = b2 if g1 in {"m0_floor", "floor"} else b1
        return (other.startswith("m0_table") or "foot" in other.lower() or "ankle" in other.lower())
    return False


def source_limit_audit(model, historical: dict) -> tuple[dict, dict[str, float]]:
    source = historical["identity"]["urdf_joint_limits"]
    mismatches = []
    velocity_limits = {}
    checked = 0
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[jid]:
            continue
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        row = source.get(joint)
        if row is None:
            continue
        checked += 1
        lo, hi = map(float, model.jnt_range[jid])
        if abs(lo - float(row["lower"])) > 1e-5 or abs(hi - float(row["upper"])) > 1e-5:
            mismatches.append({"joint": joint, "compiled": [lo, hi],
                               "source": [float(row["lower"]), float(row["upper"]) ]})
        velocity = row.get("velocity")
        if velocity is not None and math.isfinite(float(velocity)) and float(velocity) > 0:
            velocity_limits[joint] = float(velocity)
    missing_arm_velocity = [joint for joint in ARM if joint not in velocity_limits]
    audit = {
        "source_limited_hinges_compared": checked,
        "source_position_range_mismatches": mismatches,
        "arm_source_velocity_limits_rad_s": {joint: velocity_limits.get(joint) for joint in ARM},
        "missing_arm_velocity_limits": missing_arm_velocity,
        "pass": not mismatches and not missing_arm_velocity,
    }
    return audit, velocity_limits


def prepare_geometry(model, data, qualifier, model_xml: Path):
    exclusions = qualifier.load_exclusions(model_xml)
    geom_ids = [i for i in range(model.ngeom) if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])]
    mesh_cache = {}
    shapes = {i: qualifier.convex_shape(model, i, mesh_cache) for i in geom_ids}
    shapes = {i: shape for i, shape in shapes.items() if shape is not None}
    pairs = qualifier.model_path_pairs(model, geom_ids, exclusions)
    pair_arrays = {
        group: np.asarray(values, dtype=np.int32).reshape((-1, 2))
        for group, values in pairs.items()
        if group in {"robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self"}
    }
    world_shapes = qualifier.all_world_shapes(model, data, shapes)
    return exclusions, shapes, pair_arrays, world_shapes


def clearance_scan(model, data, qualifier, shapes, pair_arrays, world_shapes) -> dict:
    # Compute conservative world AABB gaps for all enabled external pairs in vectorized form.
    world_shapes = qualifier.all_world_shapes(model, data, shapes)
    bounds = {}
    for geom_id, (points, _normals, _position, _rotation) in world_shapes.items():
        bounds[geom_id] = (points.min(axis=0), points.max(axis=0))
    contacts = qualifier.contact_map_for_data(data)
    groups = {}
    for group, pair_array in pair_arrays.items():
        if pair_array.size == 0:
            groups[group] = {"minimum_clearance_lower_bound_m": None, "witness": None, "pairs_checked": 0}
            continue
        first_min = np.asarray([bounds[int(g)][0] for g in pair_array[:, 0]])
        first_max = np.asarray([bounds[int(g)][1] for g in pair_array[:, 0]])
        second_min = np.asarray([bounds[int(g)][0] for g in pair_array[:, 1]])
        second_max = np.asarray([bounds[int(g)][1] for g in pair_array[:, 1]])
        delta = np.maximum(0.0, np.maximum(first_min - second_max, second_min - first_max))
        gaps = np.linalg.norm(delta, axis=1)
        # Use the same conservative AABB/SAT policy as the static qualification.
        for pair_index in np.flatnonzero(gaps < CLEARANCE_M):
            first, second = map(int, pair_array[pair_index])
            result = qualifier.geom_pair_lower_bound(model, first, second, world_shapes, contacts)
            gaps[pair_index] = float(result["clearance_m"])
        min_index = int(np.argmin(gaps))
        first, second = map(int, pair_array[min_index])
        groups[group] = {
            "minimum_clearance_lower_bound_m": float(gaps[min_index]),
            "witness": {"geom1": geom_name(model, first), "body1": body_name(model, first),
                        "geom2": geom_name(model, second), "body2": body_name(model, second)},
            "pairs_checked": int(len(pair_array)),
        }
    return groups


def contact_rows(model, data, legacy) -> list[dict]:
    rows = legacy.contacts(model, data, forces=True)
    for index, row in enumerate(rows):
        contact = data.contact[index]
        force_local = row.get("force_local_n")
        if force_local is None:
            row["force_contact_frame_n"] = None
            row["force_contact_frame_to_world_n"] = None
        else:
            local = np.asarray(force_local[:3], dtype=float)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            row["force_contact_frame_n"] = local.tolist()
            row["force_contact_frame_to_world_n"] = (frame.T @ local).tolist()
        row["expected_support_contact"] = expected_contact(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--historical-result", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--static-result", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--internal-preflight", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--controller-source", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--output-parent", type=Path, default=REVIEWS / "issue-46")
    parser.add_argument("--approach-duration-s", type=float, default=4.0)
    parser.add_argument("--settle-duration-s", type=float, default=1.0)
    parser.add_argument("--post-approach-stop-s", type=float, default=1.0)
    parser.add_argument("--prior-abort-steps", type=int, default=0,
                         help="steps already spent on ordinary telemetry-aborted Experiment A attempts")
    parser.add_argument("--preflight-only", action="store_true",
                         help="compile and check identity/initial safety without calling mj_step")
    args = parser.parse_args()

    for path in (args.model, args.historical_result, args.static_result, args.internal_preflight,
                 args.controller_source):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")
    model_sha = sha256(args.model)
    if model_sha != EXPECTED_MODEL_SHA:
        raise SystemExit(f"compiled model SHA mismatch: {model_sha}")
    if not math.isclose(args.approach_duration_s, 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("this bounded run is fixed at a 4.0 s approach; do not turn it into a sweep")
    if args.settle_duration_s + args.approach_duration_s + args.post_approach_stop_s > 9.0:
        raise SystemExit("planned physics duration must remain below the 10 s cap with margin")
    if args.prior_abort_steps < 0 or args.prior_abort_steps + int(round(
            (args.settle_duration_s + args.approach_duration_s + args.post_approach_stop_s) / EXPECTED_DT)) > 10000:
        raise SystemExit("cumulative Experiment A physics steps would exceed the 10,000-step cap")

    historical = json.loads(args.historical_result.read_text(encoding="utf-8"))
    static = json.loads(args.static_result.read_text(encoding="utf-8"))
    internal = json.loads(args.internal_preflight.read_text(encoding="utf-8"))
    if historical.get("status") != "PASS":
        raise SystemExit("historical source episode is not recorded PASS")
    if static.get("input_identity", {}).get("diagnostic_model_sha256") != model_sha:
        raise SystemExit("static approach result and experiment model do not match")
    if static.get("candidate_25mm_retreat", {}).get("path_sample_count") != 101:
        raise SystemExit("expected the exact saved 101-sample 25 mm candidate path")
    static_gates = static.get("gates", {})
    required_static_gates = (
        "candidate_pregrasp_ik", "candidate_all_source_position_limits",
        "candidate_robot_bottle_clearance_ge_2mm", "candidate_nonpad_robot_bottle_clearance_ge_2mm",
        "candidate_robot_table_clearance_ge_2mm", "candidate_robot_self_clearance_ge_2mm",
        "candidate_source_mesh_clearance_ge_2mm", "candidate_historical_final_grasp_preserved",
    )
    if not all(static_gates.get(key) for key in required_static_gates):
        raise SystemExit("static external approach qualification did not pass every required gate")
    if internal.get("status") != "PASS_STATIC_INTERNAL_RANGE" or internal.get("physics_steps") != 0:
        raise SystemExit("the zero-step internal linkage-range preflight is not PASS")
    for key in ("exact_model_identity", "x2_urdf_joint_ranges_match_compiled_model",
                "historical_actuator_mapping_reproduced", "bounded_pd_feedforward_formula_identified",
                "controller_helpers_match_clean_source_head", "closed_loop_static_solver_converged",
                "all_sampled_joint_position_limits_pass",
                "no_active_or_negative_clearance_internal_collision_over_sampled_range"):
        if not internal.get("gates", {}).get(key):
            raise SystemExit(f"static internal-range gate is missing or false: {key}")

    controller_audit = internal["controller_semantics"]["controller_source_audit"]
    if sha256(args.controller_source) != controller_audit.get("current_runner_file_sha256"):
        raise SystemExit("controller source differs from the exact file audited in the static preflight")
    identity = historical["identity"]
    native_library = Path(identity["native_library"])
    if not native_library.is_file() or sha256(native_library) != identity["native_library_sha256"]:
        raise SystemExit("pinned MuJoCo native library is missing or its SHA-256 changed")
    import subprocess
    for key in ("x2_checkout", "menagerie_checkout"):
        checkout = identity[key]
        source_root = Path(checkout["path"])
        if not source_root.is_dir():
            raise SystemExit(f"pinned source checkout is missing: {source_root}")
        source_head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        source_dirty = bool(subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip())
        if source_head != checkout["head"] or source_dirty != (not bool(checkout["clean"])):
            raise SystemExit(f"pinned source checkout identity/clean state changed: {key}")

    source = pg.validate_output_path(args.output_parent, authorized_roots=(str(REVIEWS),),
                                     require_exists=True, expected_kind="directory", environment="wsl")
    output = pg.create_unique_output_directory(source.wsl_path, prefix="robotiq-dynamic-approach",
                                                authorized_roots=(str(REVIEWS),), environment="wsl")
    out = output.native_path
    raw_dir = out / "raw"
    raw_dir.mkdir()
    snapshot_path = out / "inputs" / "controller_source_snapshot.py"
    snapshot_path.parent.mkdir()
    shutil.copyfile(args.controller_source, snapshot_path)
    controller_sha = sha256(snapshot_path)
    legacy = load_module(snapshot_path, "issue46_robotiq_controller_snapshot")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    if mujoco.__version__ != "3.3.6" or sys.version_info[:3] != (3, 10, 12):
        raise SystemExit(f"runtime mismatch: MuJoCo {mujoco.__version__}, Python {sys.version.split()[0]}")
    if not math.isclose(float(model.opt.timestep), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"model timestep mismatch: {model.opt.timestep}")
    if model.nu != 32 or model.ngeom != 137:
        raise SystemExit(f"unexpected compiled model inventory: nu={model.nu}, ngeom={model.ngeom}")
    source_limits, velocity_limits = source_limit_audit(model, historical)
    if not source_limits["pass"]:
        raise SystemExit(f"source joint-limit audit failed: {source_limits}")

    candidate = static["candidate_25mm_retreat"]
    trajectory = candidate["joint_trajectory"]
    q_waypoints = np.asarray([row["right_arm_qpos_rad"] for row in trajectory], dtype=float)
    if q_waypoints.shape != (101, len(ARM)):
        raise SystemExit(f"unexpected saved trajectory shape: {q_waypoints.shape}")
    q_start, q_goal = q_waypoints[0], q_waypoints[-1]
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM]
    if any(jid < 0 for jid in arm_joint_ids):
        raise SystemExit("compiled model is missing a right-arm joint")
    arm_qpos_ids = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    arm_dof_ids = [int(model.jnt_dofadr[jid]) for jid in arm_joint_ids]
    refs = legacy.controller_config(model)
    if len(refs) != 31:
        raise SystemExit(f"unexpected source motor map size: {len(refs)}")
    actuator_id_by_joint = {row["joint"]: int(row["actuator_id"]) for row in refs}
    historical_map = historical["controller"]["actuator_mapping"]
    if len(historical_map) != len(refs):
        raise SystemExit("historical controller mapping length mismatch")
    map_keys = ("actuator_id", "actuator", "joint_id", "joint", "qpos_id", "dof_id", "gear",
                "kp_simulation_derived", "kd_simulation_derived", "torque_min_nm", "torque_max_nm")
    for current, prior in zip(refs, historical_map):
        for key in map_keys:
            a, b = current.get(key), prior.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                equal = abs(float(a) - float(b)) <= 1e-9
            else:
                equal = a == b
            if not equal:
                raise SystemExit(f"controller map differs from historical record: {current['joint']} {key}")
    controller_helpers = internal["controller_semantics"]["controller_source_audit"]["controller_helper_hashes"]
    if not internal["controller_semantics"]["controller_source_audit"]["controller_helpers_match_clean_head"]:
        raise SystemExit("controller helper functions do not match their clean source HEAD")
    if not internal["controller_semantics"]["controller_source_audit"]["historical_actuator_mapping_matches_compiled_model"]:
        raise SystemExit("controller mapping was not reconciled to the historical record")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # One-time experiment initialization: place the actuated arm in the saved PREGRASP state.
    data.qpos[arm_qpos_ids] = q_start
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
    if grip_id < 0:
        raise SystemExit("compiled model is missing rq_fingers_actuator")
    data.ctrl[grip_id] = 0.0
    mujoco.mj_forward(model, data)
    initial_contacts = contact_rows(model, data, legacy)
    initial_contact_gate = legacy.static_contact_gate(initial_contacts)
    if not initial_contact_gate.get("pass"):
        raise SystemExit(f"initial state is not nonpenetrating: {initial_contact_gate}")
    if any(not expected_contact(row) for row in initial_contacts):
        raise SystemExit(f"initial state has an unexpected contact: {initial_contacts}")
    if data.time != 0.0:
        raise SystemExit("new data did not start at t=0")

    qualifier = load_module(
        ROOT / "scripts/research/issue46_manipulation_integration/qualify_robotiq_static_approach.py",
        "issue46_robotiq_geometry_qualifier",
    )
    exclusions, shapes, pair_arrays, world_shapes = prepare_geometry(model, data, qualifier, args.model)
    initial_clearance = clearance_scan(model, data, qualifier, shapes, pair_arrays, world_shapes)
    initial_clearance_pass = all(
        initial_clearance[group]["minimum_clearance_lower_bound_m"] is not None
        and initial_clearance[group]["minimum_clearance_lower_bound_m"] >= CLEARANCE_M
        for group in ("robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self")
    )
    if not initial_clearance_pass:
        raise SystemExit(f"initial state does not meet the 2 mm external clearance gate: {initial_clearance}")

    base_targets = {row["joint"]: float(data.qpos[row["qpos_id"]]) for row in refs if row["joint"] not in ARM}
    bottle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_qpos_id = int(model.jnt_qposadr[bottle_joint_id])
    bottle_dof_id = int(model.jnt_dofadr[bottle_joint_id])
    bottle_qpos_initial = data.qpos[bottle_qpos_id:bottle_qpos_id + 7].copy()
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
                   if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    joint_qpos_ids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                      for n in joint_names]
    joint_dof_ids = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in joint_names]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
                      for i in range(model.nu)]

    def compiled_geom_inventory(body_prefix: str) -> list[dict]:
        inventory = []
        for geom_id in range(model.ngeom):
            body = body_name(model, geom_id)
            if not body.startswith(body_prefix):
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            mesh = None if mesh_id < 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            inventory.append({
                "geom": geom_name(model, geom_id), "body": body,
                "type": int(model.geom_type[geom_id]), "mesh": mesh,
                "size": model.geom_size[geom_id].astype(float).tolist(),
                "local_position_m": model.geom_pos[geom_id].astype(float).tolist(),
                "local_quaternion_wxyz": model.geom_quat[geom_id].astype(float).tolist(),
                "friction": model.geom_friction[geom_id].astype(float).tolist(),
                "condim": int(model.geom_condim[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            })
        return inventory

    bottle_inventory = compiled_geom_inventory("m0_bottle")
    table_inventory = compiled_geom_inventory("m0_table")
    bottle_mass = float(model.body_mass[bottle_body_id])
    if not math.isclose(bottle_mass, 0.570, rel_tol=0.0, abs_tol=1e-6) or len(bottle_inventory) != 4:
        raise SystemExit(f"canonical bottle identity mismatch: mass={bottle_mass}, geoms={len(bottle_inventory)}")

    if args.preflight_only:
        preflight = {
            "schema": "robotsim.issue46.robotiq_dynamic_approach_preflight.v1",
            "classification": "SIMULATION_ONLY_DIAGNOSTIC_ZERO_STEP_PREFLIGHT",
            "run_id": str(uuid.uuid4()), "physics_steps": 0,
            "mujoco_version": mujoco.__version__, "python": sys.version.split()[0],
            "model_path": str(args.model), "model_sha256": model_sha,
            "timestep_s": float(model.opt.timestep), "nq": int(model.nq), "nv": int(model.nv),
            "nu": int(model.nu), "ngeom": int(model.ngeom),
            "controller_source_path": str(snapshot_path), "controller_source_sha256": controller_sha,
            "controller_mapping_count": len(refs), "source_joint_limit_audit": source_limits,
            "initial_contacts": initial_contacts, "initial_contact_gate": initial_contact_gate,
            "initial_external_clearances": initial_clearance,
            "initial_external_2mm_gate": initial_clearance_pass,
            "bottle": {"mass_kg": bottle_mass, "freejoint": "m0_bottle_free",
                       "freejoint_qpos": data.qpos[bottle_qpos_id:bottle_qpos_id + 7].astype(float).tolist(),
                       "compiled_geoms": bottle_inventory},
            "table_geoms": table_inventory,
            "official_wrist_mount": identity["official_x2_tool_mount"],
            "simulation_only_adapter": historical["adapter"],
            "existing_collision_exclusions": sorted([list(pair) for pair in exclusions]),
            "no_physics_stepping": True,
            "output": output.as_dict(),
        }
        pg.write_new_text(out / "preflight.json", json.dumps(preflight, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          authorized_roots=(str(REVIEWS),), environment="wsl")
        manifest = pg.write_artifact_manifest(out, authorized_roots=(str(REVIEWS),), environment="wsl")
        pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
        print(json.dumps({"status": "PASS_ZERO_STEP_PREFLIGHT", "run_id": preflight["run_id"],
                          "physics_steps": 0, "initial_clearances": initial_clearance,
                          "output": output.as_dict(), "artifact_count": len(manifest["artifacts"])},
                         indent=2, ensure_ascii=False))
        return 0

    trace_path = raw_dir / "joint_actuator_trace.jsonl"
    contact_path = raw_dir / "contact_trace.jsonl"
    clearance_path = raw_dir / "clearance_trace.jsonl"
    phase_rows = []
    failure = None
    observed_gates = {
        "finite_state": True, "source_position_limits": True, "source_velocity_limits": True,
        "actuator_ctrl_limits": True, "actuator_force_limits": True,
        "unexpected_contacts": True, "internal_spring_pad_contact": True,
        "external_clearance_ge_2mm": True,
    }
    maximums = {"right_arm_abs_tracking_error_rad": 0.0, "right_arm_abs_qvel_rad_s": 0.0,
                "abs_actuator_force_by_name": {name: 0.0 for name in actuator_names}}
    minimums = {"robot_bottle": math.inf, "robot_bottle_nonpad": math.inf,
                "robot_table": math.inf, "robot_self": math.inf}
    settle_contact_rows = []
    rows_written = 0
    render_path = out / "experiment_a_dynamic_approach.mp4"
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.22, -0.05, 0.89]
    camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -14
    writer = imageio.get_writer(str(render_path), fps=25, codec="libx264", quality=7, macro_block_size=1)

    start_pose = {
        "right_arm_qpos_rad": q_start.tolist(),
        "other_joint_positions": "compiled model qpos0; no other joint qpos is modified",
        "bottle_freejoint_qpos": bottle_qpos_initial.tolist(),
        "bottle_qpos_writes_after_rollout_start": 0,
    }
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(out / "start_pregrasp.png")
    run_start = time.monotonic()
    phases = (("pregrasp_settle", args.settle_duration_s, q_start, q_start),
              ("approach", args.approach_duration_s, q_start, q_goal),
              ("post_approach_stop", args.post_approach_stop_s, q_goal, q_goal))
    try:
        with trace_path.open("w", encoding="utf-8", buffering=1) as joint_stream, \
             contact_path.open("w", encoding="utf-8", buffering=1) as contact_stream, \
             clearance_path.open("w", encoding="utf-8", buffering=1) as clearance_stream:
            for phase, duration, phase_start_q, phase_end_q in phases:
                steps = int(round(duration / EXPECTED_DT))
                phase_start_step = int(round(data.time / EXPECTED_DT))
                phase_contact_count = 0
                phase_bilateral = 0
                phase_min_clearance = {key: math.inf for key in minimums}
                for k in range(steps):
                    u = (k + 1) / steps
                    s, dsdu = legacy.quintic(u)
                    qtarget = phase_start_q + s * (phase_end_q - phase_start_q)
                    qvel_target = (dsdu / duration) * (phase_end_q - phase_start_q)
                    targets = legacy.set_target_dict(model, qtarget, base_targets)
                    target_vel = {name: float(value) for name, value in zip(ARM, qvel_target)}
                    mujoco.mj_forward(model, data)
                    command_rows = legacy.apply_controller(model, data, refs, targets, target_vel)
                    data.ctrl[grip_id] = 0.0
                    mujoco.mj_step(model, data)
                    step = int(round(data.time / EXPECTED_DT))
                    contacts = contact_rows(model, data, legacy)
                    contact_stream.write(json.dumps({"step": step, "time_s": float(data.time),
                                                     "phase": phase, "contacts": contacts}, allow_nan=False) + "\n")
                    phase_contact_count += bool(contacts)
                    if phase == "pregrasp_settle":
                        settle_contact_rows.append(contacts)
                    range_errors, velocity_errors = legacy.all_limited_joint_checks(model, data, velocity_limits)
                    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                                  and np.isfinite(data.qacc).all() and np.isfinite(data.ctrl).all()
                                  and np.isfinite(data.actuator_force).all())
                    unexpected = [row for row in contacts if not expected_contact(row)]
                    internal_contacts = [row for row in contacts
                                         if ("spring_link" in row["body1"] and "pad" in row["body2"])
                                         or ("pad" in row["body1"] and "spring_link" in row["body2"])]
                    clearance = clearance_scan(model, data, qualifier, shapes, pair_arrays, world_shapes)
                    for group in minimums:
                        value = clearance[group]["minimum_clearance_lower_bound_m"]
                        if value is not None:
                            minimums[group] = min(minimums[group], value)
                            phase_min_clearance[group] = min(phase_min_clearance[group], value)
                    clearance_stream.write(json.dumps({"step": step, "time_s": float(data.time),
                                                       "phase": phase, "groups": clearance}, allow_nan=False) + "\n")
                    bilateral = any("rq_left_pad" in row["body1"] + row["body2"] for row in contacts) and \
                                any("rq_right_pad" in row["body1"] + row["body2"] for row in contacts)
                    phase_bilateral += bilateral

                    arm_actual = np.asarray(data.qpos[arm_qpos_ids], dtype=float)
                    arm_velocity = np.asarray(data.qvel[arm_dof_ids], dtype=float)
                    arm_error = arm_actual - qtarget
                    maximums["right_arm_abs_tracking_error_rad"] = max(
                        maximums["right_arm_abs_tracking_error_rad"], float(np.max(np.abs(arm_error)))
                    )
                    maximums["right_arm_abs_qvel_rad_s"] = max(
                        maximums["right_arm_abs_qvel_rad_s"], float(np.max(np.abs(arm_velocity)))
                    )
                    for aid, actuator_name in enumerate(actuator_names):
                        maximums["abs_actuator_force_by_name"][actuator_name] = max(
                            maximums["abs_actuator_force_by_name"][actuator_name],
                            abs(float(data.actuator_force[aid])))
                    command_by_joint = {row["joint"]: row for row in command_rows}
                    arm_controls = []
                    for index, name in enumerate(ARM):
                        command = command_by_joint[name]
                        aid = actuator_id_by_joint[name]
                        arm_controls.append({
                            "joint": name, "target_qpos_rad": float(qtarget[index]),
                            "target_qvel_rad_s": float(qvel_target[index]),
                            "actual_qpos_rad": float(arm_actual[index]),
                            "actual_qvel_rad_s": float(arm_velocity[index]),
                            "qacc_rad_s2": float(data.qacc[arm_dof_ids[index]]),
                            "tracking_error_rad": float(arm_error[index]),
                            "requested_torque_nm": float(command["requested_torque_nm"]),
                            "commanded_torque_nm": float(command["final_torque_nm"]),
                            "actuator_ctrl": float(data.ctrl[aid]),
                            "actuator_force": float(data.actuator_force[aid]),
                            "generalized_actuator_force_nm": float(data.qfrc_actuator[arm_dof_ids[index]]),
                        })
                    bottle_com = np.asarray(data.xipos[bottle_body_id], dtype=float)
                    bottle_v = np.asarray(data.qvel[bottle_dof_id:bottle_dof_id + 3], dtype=float)
                    bottle_w = np.asarray(data.qvel[bottle_dof_id + 3:bottle_dof_id + 6], dtype=float)
                    tcp_pos = np.asarray(data.site_xpos[tcp_site_id], dtype=float)
                    tcp_rot = np.asarray(data.site_xmat[tcp_site_id], dtype=float).reshape(3, 3)
                    joint_stream.write(json.dumps({
                        "step": step, "time_s": float(data.time), "phase": phase,
                        "right_arm": arm_controls,
                        "all_hinge_qpos_rad": data.qpos[joint_qpos_ids].astype(float).tolist(),
                        "all_hinge_qvel_rad_s": data.qvel[joint_dof_ids].astype(float).tolist(),
                        "all_actuator_ctrl": data.ctrl.astype(float).tolist(),
                        "all_actuator_force": data.actuator_force.astype(float).tolist(),
                        "all_generalized_actuator_force": data.qfrc_actuator.astype(float).tolist(),
                        "bottle_root_qpos": data.qpos[bottle_qpos_id:bottle_qpos_id + 7].astype(float).tolist(),
                        "bottle_com_world_m": bottle_com.tolist(),
                        "bottle_linear_velocity_world_m_s": bottle_v.tolist(),
                        "bottle_angular_velocity_world_rad_s": bottle_w.tolist(),
                        "tcp_world_m": tcp_pos.tolist(), "tcp_rotation_world": tcp_rot.tolist(),
                        "gripper_driver_qpos_rad": float(data.qpos[int(model.jnt_qposadr[
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rq_right_driver_joint")])]),
                        "bottle_qpos_writes_after_rollout_start": 0,
                    }, allow_nan=False) + "\n")
                    rows_written += 1

                    clearance_failures = [group for group in ("robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self")
                                          if clearance[group]["minimum_clearance_lower_bound_m"] is None
                                          or clearance[group]["minimum_clearance_lower_bound_m"] < CLEARANCE_M]
                    observed_gates["finite_state"] &= finite
                    observed_gates["source_position_limits"] &= not bool(range_errors)
                    observed_gates["source_velocity_limits"] &= not bool(velocity_errors)
                    observed_gates["unexpected_contacts"] &= not bool(unexpected)
                    observed_gates["internal_spring_pad_contact"] &= not bool(internal_contacts)
                    observed_gates["external_clearance_ge_2mm"] &= not bool(clearance_failures)
                    ctrl_violations, force_violations = [], []
                    for aid in range(model.nu):
                        if model.actuator_ctrllimited[aid]:
                            lo, hi = map(float, model.actuator_ctrlrange[aid])
                            value = float(data.ctrl[aid])
                            if value < lo - 1e-9 or value > hi + 1e-9:
                                ctrl_violations.append({"actuator": actuator_names[aid], "ctrl": value, "range": [lo, hi]})
                        if model.actuator_forcelimited[aid]:
                            lo, hi = map(float, model.actuator_forcerange[aid])
                            value = float(data.actuator_force[aid])
                            if value < lo - 1e-6 or value > hi + 1e-6:
                                force_violations.append({"actuator": actuator_names[aid], "force": value, "range": [lo, hi]})
                    observed_gates["actuator_ctrl_limits"] &= not bool(ctrl_violations)
                    observed_gates["actuator_force_limits"] &= not bool(force_violations)
                    if not finite:
                        failure = {"gate": "finite_state", "step": step, "phase": phase}
                    elif range_errors:
                        failure = {"gate": "source_joint_position_limit", "step": step, "phase": phase,
                                   "details": range_errors[0]}
                    elif velocity_errors:
                        failure = {"gate": "source_joint_velocity_limit", "step": step, "phase": phase,
                                   "details": velocity_errors[0]}
                    elif clearance_failures:
                        group = clearance_failures[0]
                        failure = {"gate": "external_clearance_below_2mm", "step": step, "phase": phase,
                                   "group": group, "measurement": clearance[group]}
                    elif unexpected:
                        failure = {"gate": "unexpected_active_contact", "step": step, "phase": phase,
                                   "contact": unexpected[0]}
                    elif internal_contacts:
                        failure = {"gate": "internal_spring_pad_contact", "step": step, "phase": phase,
                                   "contact": internal_contacts[0]}
                    elif ctrl_violations:
                        failure = {"gate": "actuator_control_limit", "step": step, "phase": phase,
                                   **ctrl_violations[0]}
                    elif force_violations:
                        failure = {"gate": "actuator_force_limit", "step": step, "phase": phase,
                                   **force_violations[0]}
                    if writer is not None and (step % 40 == 0 or rows_written == 1 or k == steps - 1):
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(renderer.render())
                    if failure is not None:
                        break
                phase_rows.append({"phase": phase, "planned_duration_s": duration,
                                   "actual_duration_s": float(data.time - phase_start_step * EXPECTED_DT),
                                   "steps": int(round(data.time / EXPECTED_DT)) - phase_start_step,
                                   "expected_contact_steps": phase_contact_count,
                                   "bilateral_pad_contact_steps": phase_bilateral,
                                   "minimum_clearance_lower_bounds_m": {
                                       key: (None if not math.isfinite(value) else value)
                                       for key, value in phase_min_clearance.items()},
                                   "completed": failure is None and rows_written >= sum(
                                       int(round(item[1] / EXPECTED_DT)) for item in phases[:len(phase_rows) + 1])})
                if failure is not None:
                    break
    finally:
        writer.close()
        renderer.close()

    final_renderer = mujoco.Renderer(model, height=720, width=1280)
    final_renderer.update_scene(data, camera=camera)
    Image.fromarray(final_renderer.render()).save(out / "final_state.png")
    final_renderer.close()
    actual_steps = int(round(data.time / EXPECTED_DT))
    planned_steps = int(round((args.settle_duration_s + args.approach_duration_s + args.post_approach_stop_s) / EXPECTED_DT))
    completed_all = failure is None and actual_steps == planned_steps
    bottle_qpos_write_count = 0
    gate_pass = bool(completed_all and source_limits["pass"] and initial_contact_gate.get("pass")
                     and initial_clearance_pass and all(
                         math.isfinite(minimums[group]) and minimums[group] >= CLEARANCE_M
                         for group in minimums))
    table_supported_steps = 0
    if settle_contact_rows:
        for step_contacts in settle_contact_rows:
            if any(any(body.startswith("m0_bottle") for body in (row["body1"], row["body2"]))
                   and any(body.startswith("m0_table") for body in (row["body1"], row["body2"]))
                   for row in step_contacts):
                table_supported_steps += 1
    result = {
        "schema": "robotsim.issue46.robotiq_dynamic_approach.v1",
        "run_id": str(uuid.uuid4()),
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_DYNAMIC_APPROACH",
        "status": "DYNAMIC_APPROACH_PASS" if gate_pass else "DYNAMIC_APPROACH_FAIL",
        "physics_run_completed": completed_all,
        "physics_steps": actual_steps,
        "prior_ordinary_runner_abort_steps": args.prior_abort_steps,
        "cumulative_experiment_a_physics_steps": args.prior_abort_steps + actual_steps,
        "physics_duration_s": float(data.time),
        "mj_step_calls": actual_steps,
        "timestep_s": float(model.opt.timestep),
        "first_failure": failure,
        "prior_aborted_attempts": [
            {"path": "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-dynamic-approach-20261024T075657Z",
             "steps": 1, "reason": "telemetry KeyError: controller command rows have no actuator_id field"},
            {"path": "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-dynamic-approach-20261024T075844Z",
             "steps": 4400, "reason": "stopped after detecting stale cached geom transforms in clearance monitor"},
        ],
        "gates": {
            "initial_nonpenetrating_state": bool(initial_contact_gate.get("pass")),
            "initial_external_clearance_ge_2mm": initial_clearance_pass,
            **observed_gates,
            "approach_and_post_stop_completed": completed_all,
            "bottle_qpos_writes_after_rollout_start": bottle_qpos_write_count,
        },
        "experiment": {
            "sequence": ["PREGRASP_SET_T=0", "OPEN_HOLD", "APPROACH_25MM", "POST_APPROACH_STOP_HOLD", "STOP"],
            "approach_duration_s": args.approach_duration_s,
            "settle_duration_s": args.settle_duration_s,
            "post_approach_stop_duration_s": args.post_approach_stop_s,
            "gripper_ctrl": 0.0,
            "no_grasp_or_lift": True,
            "one_time_initialization_only": True,
            "start_pose": start_pose,
            "start_arm_qpos_rad": q_start.tolist(),
            "goal_arm_qpos_rad": q_goal.tolist(),
            "maximums": maximums,
            "minimum_external_clearance_lower_bounds_m": minimums,
            "clearance_monitor": "refreshes world convex shapes from current post-step MjData every step; vectorized AABB for all enabled external pairs and SAT resolution below 2 mm",
            "phase_results": phase_rows,
            "initial_active_contacts": initial_contacts,
            "pregrasp_settle_bottle_table_support_steps": table_supported_steps,
            "pregrasp_settle_steps": int(round(args.settle_duration_s / EXPECTED_DT)),
            "collision_exclusions_from_model": sorted([list(pair) for pair in exclusions]),
        },
        "input_identity": {
            "model_path": str(args.model), "model_sha256": model_sha,
            "historical_result_path": str(args.historical_result), "historical_result_sha256": sha256(args.historical_result),
            "static_qualification_path": str(args.static_result), "static_qualification_sha256": sha256(args.static_result),
            "internal_range_preflight_path": str(args.internal_preflight), "internal_range_preflight_sha256": sha256(args.internal_preflight),
            "x2_source_commit": historical["identity"]["x2_checkout"]["head"],
            "x2_source_mjcf_sha256": historical["identity"]["x2_mjcf_sha256"],
            "robotiq_source_commit": historical["identity"]["menagerie_checkout"]["head"],
            "robotiq_source_mjcf_sha256": historical["identity"]["robotiq_xml_sha256"],
            "controller_source_original_path": str(args.controller_source),
            "controller_source_snapshot_path": str(snapshot_path),
            "controller_source_snapshot_sha256": controller_sha,
            "historical_recorded_runner_sha256": historical.get("runner", {}).get("sha256"),
            "historical_full_runner_hash_matches_available_source": bool(controller_audit.get("runner_file_hash_matches_historical")),
            "controller_helper_hashes": controller_helpers,
            "controller_helpers_match_clean_source_head": bool(controller_audit.get("controller_helpers_match_clean_head")),
            "historical_actuator_mapping_match": bool(controller_audit.get("historical_actuator_mapping_matches_compiled_model")),
            "controller_mapping": refs,
            "mujoco_version": mujoco.__version__, "python_executable": sys.executable,
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "ngeom": int(model.ngeom),
            "model_timestep_s": float(model.opt.timestep), "model_gravity_m_s2": model.opt.gravity.astype(float).tolist(),
            "model_solver": int(model.opt.solver), "model_solver_iterations": int(model.opt.iterations),
            "model_integrator": int(model.opt.integrator),
            "bottle": {"body": "m0_bottle", "mass_kg": bottle_mass, "freejoint": "m0_bottle_free",
                       "freejoint_qpos_initial": bottle_qpos_initial.tolist(), "compiled_geoms": bottle_inventory,
                       "identity_source": "compiled exact-SHA SIMULATION_ONLY model and historical integrity metadata"},
            "table": {"compiled_geoms": table_inventory,
                      "identity_source": "compiled exact-SHA SIMULATION_ONLY model; unchanged from historical episode"},
            "official_right_wrist_mount": identity["official_x2_tool_mount"],
            "simulation_only_adapter": historical["adapter"],
            "historical_physical_integrity_flags": historical.get("physical_integrity"),
            "x2_checkout": identity["x2_checkout"],
            "robotiq_checkout": identity["menagerie_checkout"],
            "robotiq_asset_sha256": identity["robotiq_asset_sha256"],
            "x2_mesh_sha256": identity["x2_mesh_sha256"],
            "source_joint_limit_audit": source_limits,
            "simulation_only_assumptions": [
                "the existing head_pitch/torso collision exclusion is retained",
                "the existing derived wrist adapter translation and rotation are retained",
                "the existing 0.001 rad coupler margin is retained",
                "no manufacturer-equivalent Robotiq mount or controller claim is made",
            ],
        },
        "evidence": {
            "joint_trace": str(trace_path), "contact_trace": str(contact_path),
            "clearance_trace": str(clearance_path), "mp4": str(render_path),
            "start_png": str(out / "start_pregrasp.png"), "final_png": str(out / "final_state.png"),
            "report": str(out / "REPORT.md"), "result": str(out / "result.json"),
        },
        "duration_wall_s": time.monotonic() - run_start,
    }
    result["output"] = output.as_dict()
    report = [
        "# Full-X2 Robotiq dynamic approach and tracking experiment",
        "",
        f"Run ID: {result['run_id']}",
        f"Outcome: {result['status']}",
        f"Physics: {actual_steps} steps at {model.opt.timestep * 1000:.3f} ms ({data.time:.3f} s)",
        f"Cumulative Experiment A steps including ordinary telemetry-aborted attempts: {args.prior_abort_steps + actual_steps}/10000.",
        f"First failure: {failure}",
        "",
        "## Measured result",
        "",
        f"- Approach and stop completed: {completed_all}.",
        f"- Maximum right-arm target error: {maximums['right_arm_abs_tracking_error_rad']:.9f} rad.",
        f"- Maximum right-arm speed: {maximums['right_arm_abs_qvel_rad_s']:.9f} rad/s.",
        f"- Minimum external conservative clearances (m): {minimums}.",
        f"- Bottle/table supporting-contact samples during settle: {table_supported_steps}/{int(round(args.settle_duration_s / EXPECTED_DT))}.",
        "- Gripper remained open; no contact/lift phase was run in Experiment A.",
        "- Bottle freejoint was initialized once and received zero qpos writes after rollout start.",
        "",
        "## Identity and interpretation",
        "",
        f"- Full-X2/Robotiq model SHA-256: `{model_sha}`.",
        f"- MuJoCo: `{mujoco.__version__}`; timestep `{model.opt.timestep}` s.",
        f"- Controller snapshot SHA-256: `{controller_sha}`. It is an immutable copy of the available source file used by this run.",
        f"- The historical result's full runner SHA is `{historical.get('runner', {}).get('sha256')}` and does not match the available full source file; verified controller helpers match clean source HEAD and the 31/31 actuator metadata mapping.",
        "- Existing SIMULATION_ONLY diagnostic wrist adapter, head/torso exclusion and 0.001 rad coupler margin are retained.",
        "- This experiment measures open-gripper dynamic approach tracking only. It is not a grasp or lift result.",
        "",
        "## Reproduction",
        "",
        f"`cd {ROOT} && PYTHONDONTWRITEBYTECODE=1 {sys.executable} {Path(__file__).resolve()} --model {args.model} --historical-result {args.historical_result} --static-result {args.static_result} --internal-preflight {args.internal_preflight} --controller-source {snapshot_path} --output-parent {args.output_parent}`",
        "",
    ]
    pg.write_new_text(out / "REPORT.md", "\n".join(report), authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(out / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(out, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "physics_steps": actual_steps,
                      "physics_duration_s": float(data.time), "first_failure": failure,
                      "maximums": maximums, "minimum_clearances_m": minimums,
                      "evidence_dir": output.as_dict(), "artifact_count": len(manifest["artifacts"])},
                     indent=2, ensure_ascii=False))
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
