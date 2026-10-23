#!/usr/bin/env python3
"""Audit one existing Candidate B full-X2 static approach witness; no rollout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "agent"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))
from issue46_manipulation_integration.approach import IKWaypoint, make_linear_approach, validate_ik_waypoints
from issue46_manipulation_integration.contracts import PoseSample, PoseSource, TimeDomain
import path_governance as pg
import mujoco


CANONICAL_REPO = Path("/home/lzy18001500226/projects/RobotSim")
AUTHORIZED_ROOT = CANONICAL_REPO / "reviews"
DEFAULT_SOURCE = AUTHORIZED_ROOT / "issue-46" / "candidate-b-centered-topdown-static-screen-20261023T104451Z"
HISTORICAL_PLAN = AUTHORIZED_ROOT / "issue-46" / "candidate-b-full-x2-midbody-side-grasp-20261023T063306Z" / "plan.json"
HISTORICAL_TRACE = AUTHORIZED_ROOT / "issue-46" / "candidate-b-full-x2-midbody-side-grasp-20261023T063306Z" / "dynamic-20261023T063317Z" / "physics_trace.csv"
SELECTED_CONFIG = "D70_H230_M550G_MID"
IK_JOINT_NAMES = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
EXPECTED_COMPILED_MODEL_SHA = "0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01"
EXPECTED_CANDIDATE_SOURCE_SHA = "b484595375381e4945d3685bf2b2346786b634c6a13b7433f7dfcfbd21252d22"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prototype_code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {str(path.relative_to(package)): sha256(path)
            for path in sorted(package.rglob("*.py"))}


def quaternion_from_matrix(matrix):
    """Convert a proper 3x3 rotation matrix to normalized xyzw."""
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = ((m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s,
             (m[1][0]-m[0][1])/s, 0.25*s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0]-m[1][1]-m[2][2]) * 2.0
        q = (0.25*s, (m[0][1]+m[1][0])/s, (m[0][2]+m[2][0])/s, (m[2][1]-m[1][2])/s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1]-m[0][0]-m[2][2]) * 2.0
        q = ((m[0][1]+m[1][0])/s, 0.25*s, (m[1][2]+m[2][1])/s, (m[0][2]-m[2][0])/s)
    else:
        s = math.sqrt(1.0 + m[2][2]-m[0][0]-m[1][1]) * 2.0
        q = ((m[0][2]+m[2][0])/s, (m[1][2]+m[2][1])/s, 0.25*s, (m[1][0]-m[0][1])/s)
    norm = math.sqrt(sum(x*x for x in q))
    return tuple(x/norm for x in q)


def git_value(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def read_rows(path: Path, config: str):
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["config"] == config]


def candidate_failure_summary(candidate):
    closure = candidate["closure"]
    interference = candidate["open_path"].get("first_interference")
    return {
        "id": candidate["id"],
        "object": candidate["object"],
        "height_offset_mm": candidate["pose"]["offset_from_midbody_mm"],
        "ik_all_pass": candidate["IK"]["all_pass"],
        "min_open_hand_object_clearance_mm": candidate["open_path"]["min_clearances_mm"]["hand_object"]["distance_mm"],
        "min_open_hand_scene_clearance_mm": candidate["open_path"]["min_clearances_mm"]["hand_scene"]["distance_mm"],
        "first_interference_sample": interference["sample"] if interference else None,
        "first_narrow_pad_cad": closure["first_narrow_pad_qCAD"],
        "first_wide_pad_cad": closure["first_wide_pad_qCAD"],
        "first_bilateral_pad_cad": closure["first_bilateral_pad_qCAD"],
        "first_linkage_cad": closure["first_loop_qCAD"],
        "pads_before_linkage": closure["pads_before_loop"],
        "bilateral_at_existing_close": closure["bilateral_at_CAD_0p30"],
        "linkage_at_existing_close": closure["loop_at_CAD_0p30"],
        "other_hand_at_existing_close": closure["other_hand_at_CAD_0p30"],
        "static_candidate_pass": candidate["static_candidate_pass"],
        "gates": candidate["gates"],
    }


def make_report(args) -> tuple[dict, str]:
    source = Path(args.source_run).resolve(strict=True)
    root = AUTHORIZED_ROOT.resolve(strict=True)
    if not source.is_relative_to(root):
        raise RuntimeError("source evidence must remain under canonical RobotSim reviews/")
    result_path = source / "result.json"
    manifest_path = source / "artifact_paths.json"
    report_path = source / "REPORT.md"
    open_csv = source / "open_insertion_samples.csv"
    closure_csv = source / "closure_samples.csv"
    for path in (result_path, manifest_path, report_path, open_csv, closure_csv):
        if not path.is_file():
            raise RuntimeError(f"required saved evidence is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = pg.verify_artifacts(manifest.get("artifacts"), authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = next((item for item in result["candidate_configurations"] if item.get("id") == SELECTED_CONFIG), None)
    if config is None:
        raise RuntimeError(f"saved source run lacks the one selected configuration {SELECTED_CONFIG}")
    model_path = Path(result["candidate_b_model"]["path"]).resolve(strict=True)
    candidate_source_path = Path(result["candidate_b_model"]["candidate_geometry_path"]).resolve(strict=True)
    model_sha = sha256(model_path)
    source_sha = sha256(candidate_source_path)
    if model_sha != EXPECTED_COMPILED_MODEL_SHA or model_sha != result["candidate_b_model"]["sha256"]:
        raise RuntimeError("compiled Candidate B full-X2 model SHA does not match the saved run")
    if source_sha != EXPECTED_CANDIDATE_SOURCE_SHA or source_sha != result["candidate_b_model"]["candidate_geometry_sha256"]:
        raise RuntimeError("Candidate B source model SHA does not match the saved run")
    if result.get("physics_steps") != 0 or result.get("forbidden_step_calls") != 0:
        raise RuntimeError("source static evidence is not zero-step")
    model_artifacts = {}
    for filename in ("runtime_model.xml", "runtime_model_geometry_selected.xml",
                     "runtime_model_head_torso_exclusion.xml", "runtime_model_fixed_base_diagnostic.xml",
                     "runtime_model_fixed_base_root_kp2_static_screen.xml"):
        path = source / filename
        if path.is_file():
            model_artifacts[filename] = {"path": str(path), "sha256": sha256(path)}
    expected_geometry_sha = result["current_model_build"]["geometry_selected_sha256"]
    expected_source_runtime_sha = result["current_model_build"]["source_build"]["sha256"]
    expected_fixed_sha = result["candidate_b_model"]["sha256"]
    if model_artifacts["runtime_model.xml"]["sha256"] != expected_source_runtime_sha:
        raise RuntimeError("source full-X2 runtime MJCF hash does not match source result")
    if model_artifacts["runtime_model_geometry_selected.xml"]["sha256"] != expected_geometry_sha:
        raise RuntimeError("integrated Candidate B geometry MJCF hash does not match source result")
    if model_artifacts["runtime_model_fixed_base_root_kp2_static_screen.xml"]["sha256"] != expected_fixed_sha:
        raise RuntimeError("fixed-base static diagnostic MJCF hash does not match source result")
    for artifact in model_artifacts.values():
        expected = result["input_sha256"].get(artifact["path"])
        if expected is not None and expected != artifact["sha256"]:
            raise RuntimeError("saved MJCF artifact differs from its source-run input hash")
    enabled_gripper_geoms = tuple(result["candidate_b_model"]["right_hand_enabled_collision_geoms"])
    geom_inventory_sha = hashlib.sha256(json.dumps(enabled_gripper_geoms, separators=(",", ":")).encode()).hexdigest()

    model = mujoco.MjModel.from_xml_path(str(model_path))
    counts = result["candidate_b_model"]["counts"]
    actual_counts = {"nbody": model.nbody, "njnt": model.njnt, "ngeom": model.ngeom,
                     "neq": model.neq, "nu": model.nu, "nq": model.nq, "nv": model.nv}
    if actual_counts != counts or mujoco.__version__ != result["candidate_b_model"]["mujoco_version"]:
        raise RuntimeError("compiled model inventory or MuJoCo version differs from the frozen evidence")

    plan_path = HISTORICAL_PLAN.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if tuple(plan.get("ik_joint_names", ())) != IK_JOINT_NAMES:
        raise RuntimeError("historical full-X2 plan joint ordering changed")
    limits = {}
    for name in IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"compiled model is missing arm joint {name}")
        limits[name] = tuple(float(x) for x in model.jnt_range[jid])

    ik_items = list(reversed(config["IK"]["waypoints"]))
    if len(ik_items) < 2 or not math.isclose(ik_items[0]["z_target_m"], config["pose"]["hover_target_site_world_m"][2], abs_tol=1e-8):
        raise RuntimeError("saved IK witness does not begin at the pregrasp hover pose")
    if not math.isclose(ik_items[-1]["z_target_m"], config["pose"]["grasp_target_site_world_m"][2], abs_tol=1e-8):
        raise RuntimeError("saved IK witness does not end at the approach target")
    ik = []
    for index, item in enumerate(ik_items):
        q = item["q_rad"]
        if len(q) != len(IK_JOINT_NAMES):
            raise RuntimeError("saved IK vector dimension differs from the full-X2 arm joint list")
        ik.append(IKWaypoint(
            phase="PREGRASP" if index == 0 else ("APPROACH" if index == len(ik_items)-1 else "APPROACH"),
            progress=index/(len(ik_items)-1),
            joint_positions_rad=dict(zip(IK_JOINT_NAMES, map(float, q))),
            position_error_m=float(item["position_error_mm"])/1000.0,
            orientation_error_rad=math.radians(float(item["orientation_error_deg"])),
            minimum_source_limit_margin_rad=float(item["minimum_source_margin_rad"]),
        ))
    ik_blockers = validate_ik_waypoints(
        ik, expected_joint_names=IK_JOINT_NAMES, position_tolerance_m=0.002,
        orientation_tolerance_rad=math.radians(1.0), minimum_limit_margin_rad=0.05,
        source_joint_limits_rad=limits,
    )
    if ik_blockers:
        raise RuntimeError("saved IK witness failed independent checks: " + ", ".join(ik_blockers))

    pose_spec = config["pose"]
    target = tuple(map(float, pose_spec["grasp_target_site_world_m"]))
    hover = tuple(map(float, pose_spec["hover_target_site_world_m"]))
    delta = tuple(target[i]-hover[i] for i in range(3))
    standoff = math.sqrt(sum(x*x for x in delta))
    direction = tuple(x/standoff for x in delta)
    matrix = pose_spec["orientation_world"]
    orientation = quaternion_from_matrix(matrix)
    object_pose = PoseSample("test_cylinder", "world", tuple(map(float, config["object"]["center_world_m"])),
                             (0.0, 0.0, 0.0, 1.0), 0, PoseSource.SIM_GROUND_TRUTH,
                             TimeDomain.SIMULATION, model_identity=model_sha)
    approach_plan = make_linear_approach(
        object_pose, tool_frame="R_omnipicker_base_link", target_point_world_m=target,
        tool_orientation_xyzw=orientation, approach_direction_world=direction,
        pregrasp_standoff_m=standoff, interpolation_samples=len(ik)-1,
    )
    if len(approach_plan.waypoints) != len(ik):
        raise RuntimeError("Cartesian approach and saved IK witness sample counts differ")

    path_rows = read_rows(open_csv, SELECTED_CONFIG)
    if len(path_rows) != config["open_path"]["sample_count"]:
        raise RuntimeError("saved collision-path CSV sample count differs from result.json")
    samples = sorted(path_rows, key=lambda row: int(row["sample"]))
    if [int(x["sample"]) for x in samples] != list(range(1, len(samples)+1)):
        raise RuntimeError("saved collision-path sample sequence is incomplete")
    path_arm_q = []
    min_arm_margins = []
    max_arm_step = 0.0
    max_xy_error = 0.0
    max_z_path_error = 0.0
    for index, row in enumerate(samples):
        q = tuple(float(x) for x in json.loads(row["q_arm_json"]))
        if len(q) != len(IK_JOINT_NAMES):
            raise RuntimeError("saved path arm state has an invalid joint dimension")
        margins = [min(value-limits[name][0], limits[name][1]-value)
                   for name, value in zip(IK_JOINT_NAMES, q)]
        if any(value < limits[name][0] or value > limits[name][1]
               for name, value in zip(IK_JOINT_NAMES, q)):
            raise RuntimeError("saved path interpolation violates a compiled joint range")
        if not math.isclose(min(margins), float(row["arm_margin_min_rad"]), abs_tol=1e-6):
            raise RuntimeError("saved path joint-margin column does not match compiled model limits")
        min_arm_margins.append(min(margins))
        path_arm_q.append(q)
        if index:
            max_arm_step = max(max_arm_step, max(abs(a-b) for a,b in zip(q, path_arm_q[index-1])))
        site = tuple(float(row[key]) for key in ("site_x_m", "site_y_m", "site_z_m"))
        max_xy_error = max(max_xy_error, math.hypot(site[0]-target[0], site[1]-target[1]))
        segment = int(row["segment"])
        alpha = float(row["alpha"])
        fraction = (segment+alpha)/(len(ik)-1)
        expected_z = hover[2] + (target[2]-hover[2])*fraction
        max_z_path_error = max(max_z_path_error, abs(site[2]-expected_z))
    if max_arm_step > 0.0200001 or max_xy_error > 0.0001 or max_z_path_error > 0.0001:
        raise RuntimeError("saved arm/Cartesian interpolation exceeds the frozen static path tolerances")
    min_hand_object = min(float(x["hand_object_clearance_mm"]) for x in samples)
    min_hand_scene = min(float(x["hand_scene_clearance_mm"]) for x in samples)
    min_robot_object = min(float(x["robot_object_clearance_mm"]) for x in samples)
    for got, expected in ((min_hand_object, config["open_path"]["min_clearances_mm"]["hand_object"]["distance_mm"]),
                          (min_hand_scene, config["open_path"]["min_clearances_mm"]["hand_scene"]["distance_mm"]),
                          (min_robot_object, config["open_path"]["min_clearances_mm"]["robot_object"]["distance_mm"])):
        if not math.isclose(got, float(expected), abs_tol=1e-5):
            raise RuntimeError("collision summary does not match its saved path CSV")
    first_row = samples[0]
    if int(first_row["object_contact_count"]) <= 0:
        raise RuntimeError("expected saved pregrasp loop contact is absent from first path sample")
    first_interference = config["open_path"]["first_interference"]
    if first_interference["sample"] != 1:
        raise RuntimeError("saved evidence no longer identifies pregrasp sample 1 as first interference")

    dynamic_pose = None
    trace_sha = None
    if HISTORICAL_TRACE.is_file():
        trace_sha = sha256(HISTORICAL_TRACE)
        with HISTORICAL_TRACE.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["step"]) == 9165:
                    targets = json.loads(row["source_joint_targets_rad"])
                    actual = json.loads(row["source_joint_positions_rad"])
                    velocities = json.loads(row["source_joint_velocities_rad_s"])
                    joint = "right_shoulder_pitch_joint"
                    dynamic_pose = {
                        "step": 9165, "time_s": float(row["time_s"]), "phase": row["stage"],
                        "target_rad": targets[joint], "actual_rad": actual[joint],
                        "target_error_rad": targets[joint]-actual[joint],
                        "measured_velocity_rad_s": velocities[joint],
                        "hand_vertical_support_N": float(row["hand_vertical_support_N"]),
                        "table_vertical_support_N": float(row["table_vertical_support_N"]),
                        "object_rise_mm": float(row["object_rise_mm"]),
                        "object_xy_drift_mm": float(row["object_xy_drift_mm"]),
                    }
                    break

    result_object = {
        "schema_version": "robotsim.issue46.manipulation-integration-audit.v1",
        "run_id": str(uuid.uuid4()),
        "task_scope": "OFFLINE_STATIC_AND_INTERFACE_VALIDATION_ONLY",
        "new_physics_steps": 0,
        "new_mj_step_calls": 0,
        "new_mj_forward_calls": 0,
        "mujoco_version": mujoco.__version__,
        "source_evidence": {
            "path": str(source), "run_id": result["run_id"],
            "artifact_manifest_verified_files": len(verified),
            "result_json_sha256": sha256(result_path),
            "open_insertion_samples_sha256": sha256(open_csv),
            "closure_samples_sha256": sha256(closure_csv),
            "original_input_sha256_map": result["input_sha256"],
            "compiled_full_x2_model": {"path": str(model_path), "sha256": model_sha, "counts": actual_counts},
            "candidate_b_source_model": {"path": str(candidate_source_path), "sha256": source_sha,
                                          "review_classification": result["current_candidate_context"]["classification"],
                                          "CAD_range": result["current_candidate_context"]["reviewed_CAD_range"]},
        },
        "candidate_b_configuration_contract": {
            "cad_range": result["candidate_b_model"]["CAD_range"],
            "open_cad": result["candidate_b_model"]["open_CAD"],
            "existing_close_target_cad": result["candidate_b_model"]["close_target_CAD"],
            "range_semantics": "UNVERIFIED_EXPERIMENTAL_SIMULATION_INPUT",
            "enabled_right_gripper_collision_proxy_count": len(enabled_gripper_geoms),
            "enabled_right_gripper_collision_proxy_names": list(enabled_gripper_geoms),
            "ordered_geom_name_list_sha256": geom_inventory_sha,
            "candidate_geometry_source_sha256": result["candidate_b_model"]["candidate_geometry_sha256"],
            "current_model_context": result["current_candidate_context"],
        },
        "candidate_b_model_artifacts": {
            "saved_mjcf_artifact_hashes": model_artifacts,
            "head_torso_and_adjacent_loop_collision_exclusions": result["current_model_build"]["head_torso_exclusion_pairs"],
            "collision_exclusion_configuration_sha256": result["current_model_build"]["head_torso_exclusion_sha256"],
            "local_contact_frame_audit": result["current_model_build"]["local_contact_audit"],
            "historical_dynamic_model_rebuild_comparison": result["historical_dynamic_model_rebuild_comparison"],
        },
        "prior_static_screen_candidate_failures": [candidate_failure_summary(item)
                                                   for item in result["candidate_configurations"]],
        "selected_configuration_only": SELECTED_CONFIG,
        "camera_sensor_runtime": {"compiled_camera_count": model.ncam, "compiled_sensor_count": model.nsensor,
                                  "rgbd_pose_source": "NOT RUN; no camera is compiled in the tested full-X2 model"},
        "navigation_handoff_compatibility": {
            "existing_issue80_adapter_path": "/home/lzy18001500226/robotsim-issue80-offline-pointgoal-20261020/ros2_ws/src/robotsim_navigation/",
            "message_contracts": ["nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped"],
            "proposal_topic": "/robotsim/navigation/cmd_vel_proposal",
            "generation_sequence_available_in_messages": False,
            "handoff_status": "BLOCKED_UNTIL_EXPLICIT_LIFECYCLE_METADATA_BRIDGE",
            "connected_to_live_navigation_or_mc": False,
        },
        "object": config["object"],
        "tool": {"frame": "R_omnipicker_base_link", "local_contact_midpoint_m": plan["local_contact"],
                 "target_site_world_m": target, "pregrasp_hover_world_m": hover,
                 "approach_direction_world": direction, "approach_standoff_m": standoff,
                 "wrist_rotation_world": matrix, "wrist_orientation_xyzw": orientation},
        "ik_witness": {
            "status": "PASS_STATIC_IK_WITNESS",
            "sample_count": len(ik), "joint_names": list(IK_JOINT_NAMES),
            "source_joint_limits_rad_from_compiled_model": {k: list(v) for k, v in limits.items()},
            "max_position_error_mm": max(x.position_error_m for x in ik)*1000.0,
            "max_orientation_error_deg": math.degrees(max(x.orientation_error_rad for x in ik)),
            "minimum_source_limit_margin_rad": min(x.minimum_source_limit_margin_rad for x in ik),
            "interpolated_cartesian_waypoints": len(approach_plan.waypoints),
            "dynamic_tracking": "NOT RUN IN THIS AUDIT",
        },
        "approach_collision_witness": {
            "status": "BLOCKED_STATIC_COLLISION",
            "sample_count": len(samples), "first_interference_sample": first_interference["sample"],
            "pregrasp_site_world_m": [float(first_row[x]) for x in ("site_x_m", "site_y_m", "site_z_m")],
            "pregrasp_hand_object_clearance_mm": float(first_row["hand_object_clearance_mm"]),
            "pregrasp_nearest_object_geom": first_row["hand_object_geom"],
            "pregrasp_object_contact_count": int(first_row["object_contact_count"]),
            "first_interference_contacts": first_interference["object_contacts"],
            "minimum_hand_object_clearance_mm": min_hand_object,
            "minimum_hand_scene_clearance_mm": min_hand_scene,
            "minimum_robot_object_clearance_mm": min_robot_object,
            "one_mm_path_clearance_gate": "FAIL",
            "collision_representations": "compiled enabled collision proxies; not visual meshes",
            "note": "Legacy saved path records minima/contacts, not a complete pairwise distance matrix; downstream feasibility API requires explicit pair-inventory coverage.",
        },
        "recorded_approach_trajectory_checks": {
            "status": "PASS_SAMPLED_SOURCE_LIMIT_AND_PATH_INTERPOLATION",
            "sample_count": len(samples),
            "maximum_adjacent_arm_joint_step_rad": max_arm_step,
            "minimum_computed_arm_joint_margin_rad": min(min_arm_margins),
            "maximum_tool_xy_error_to_target_m": max_xy_error,
            "maximum_tool_z_error_to_linear_path_m": max_z_path_error,
            "source_joint_ranges_rechecked": True,
            "continuous_collision_proof": False,
        },
        "closure_at_existing_close_target": {
            "cad": 0.30,
            "narrow_pad_signed_distance_mm": config["closure"]["state_CAD_0p30"]["narrow_pad_mm"],
            "wide_pad_signed_distance_mm": config["closure"]["state_CAD_0p30"]["wide_pad_mm"],
            "linkage_signed_distance_mm": config["closure"]["state_CAD_0p30"]["loop_mm"],
            "other_hand_signed_distance_mm": config["closure"]["state_CAD_0p30"]["other_hand_mm"],
            "first_bilateral_pad_cad": config["closure"]["first_bilateral_pad_qCAD"],
            "first_linkage_contact_cad": config["closure"]["first_loop_qCAD"],
            "pads_before_linkage": config["closure"]["pads_before_loop"],
            "status": "BLOCKED_LINKAGE_INTERFERENCE",
        },
        "historical_dynamic_evidence_read_only": {
            "trace_path": str(HISTORICAL_TRACE), "trace_sha256": trace_sha,
            "shoulder_pitch_sample": dynamic_pose,
            "not_reproduced_or_changed": True,
        },
        "physical_stages": {
            "PREGRASP_IK": "PASS_STATIC_ONLY",
            "PREGRASP_COLLISION_CLEARANCE": "BLOCKED",
            "APPROACH_COLLISION_CLEARANCE": "BLOCKED",
            "DYNAMIC_ARM_TRACKING": "NOT RUN",
            "GRASP": "NOT RUN",
            "LIFT": "NOT RUN IN THIS TASK; historical 1 mm attempt failed",
            "TRANSFER": "NOT RUN", "RELEASE": "NOT RUN", "PLACE": "NOT RUN", "SETTLE": "NOT RUN",
        },
        "checks": {
            "source_manifest": "PASS",
            "candidate_source_and_compiled_model_sha": "PASS",
            "compiled_model_inventory_and_mujoco_version": "PASS",
            "recorded_ik_target_limits_and_residuals": "PASS_STATIC_ONLY",
            "saved_approach_clearance": "BLOCKED",
            "physics_dynamics": "NOT RUN",
        },
    }
    lines = [
        "# Issue #46 — Candidate B approach / IK integration audit",
        "", "## Result", "",
        "This audit reads and checks one existing 70 mm full-X2 static witness. It does not rerun the earlier multi-size screen and performs no dynamics.",
        "", "- Full-X2 static IK witness: **PASS** (29 recorded arm targets; source limits and residuals rechecked against the compiled model).",
        "- PREGRASP collision clearance: **BLOCKED**. The first recorded hover sample already has loop-to-object collision.",
        "- APPROACH collision clearance: **BLOCKED**. The saved vertical path includes loop/body collisions and 0 mm minimum hand/workcell clearance.",
        "- Dynamic arm tracking: **NOT RUN** here. Historical sample has a shoulder-pitch lag; see below.",
        "- Grasp/lift/transfer/release/place/settle: **NOT RUN** in this audit.",
        "", "## Frozen Candidate B identity", "",
        f"- Candidate geometry source SHA-256: `{source_sha}` ({candidate_source_path}).",
        f"- Integrated/fixed-base compiled full-X2 MJCF SHA-256: `{model_sha}` ({model_path}).",
        f"- Integrated Candidate B geometry-selected MJCF SHA-256: `{expected_geometry_sha}`; model/exclusion asset paths and hashes are in `result.json`.",
        f"- MuJoCo `{mujoco.__version__}`; counts: `{actual_counts}`.",
        f"- CAD command range `{result['candidate_b_model']['CAD_range']}`, open `{result['candidate_b_model']['open_CAD']}`, existing close target `{result['candidate_b_model']['close_target_CAD']}`; evidence labels it `{result['current_candidate_context']['classification']}`.",
        f"- Enabled right-gripper collision proxies: {result['candidate_b_model']['right_hand_enabled_collision_geom_count']}; open-path and closure source artifacts are hashed in `result.json`.",
        f"- Frozen self-collision/exclusion entries: `{result['current_model_build']['head_torso_exclusion_pairs']}` (configuration SHA `{result['current_model_build']['head_torso_exclusion_sha256']}`).",
        "- Variant/model status is experiment-specific and remains unverified as a hardware command range; no geometry or transmission was changed.",
        "", "## Approach witness", "",
        f"- Object pose: center `{config['object']['center_world_m']} m`, diameter/height `{config['object']['diameter_mm']:.1f}/{config['object']['height_mm']:.1f} mm`, mass `{config['object']['mass_kg']:.3f} kg`.",
        f"- Tool frame `R_omnipicker_base_link`; local contact midpoint `{plan['local_contact']} m`.",
        f"- PREGRASP hover `{hover} m`; target `{target} m`; vertical approach direction `{direction}`; standoff `{standoff:.3f} m`.",
        f"- First collision is sample 1 at hover: {int(first_row['object_contact_count'])} active contacts; nearest recorded loop geom `{first_row['hand_object_geom']}` at `{float(first_row['hand_object_clearance_mm']):.3f} mm` signed clearance.",
        f"- Minimum sampled hand/object clearance `{min_hand_object:.3f} mm`; hand/workcell clearance `{min_hand_scene:.3f} mm`; full path has {len(samples)} saved samples.",
        f"- Saved full-X2 arm interpolation rechecked at all {len(samples)} samples: max adjacent joint step `{max_arm_step:.6f} rad`, minimum compiled range margin `{min(min_arm_margins):.6f} rad`, maximum tool x/y path error `{max_xy_error*1000:.4f} mm`, maximum z path error `{max_z_path_error*1000:.4f} mm`.",
        "- The recorded wrist pose IK passes the historical 2 mm / 1 degree / 0.05 rad margin gates; this proves sampled inverse kinematic feasibility only, not a collision-free route or dynamic shoulder tracking.",
        "", "## Linkage first-contact evidence", "",
        f"At the existing close target qCAD=0.30, pad clearances are `{config['closure']['state_CAD_0p30']['narrow_pad_mm']:.3f} mm` and `{config['closure']['state_CAD_0p30']['wide_pad_mm']:.3f} mm`, while a linkage proxy is at `{config['closure']['state_CAD_0p30']['loop_mm']:.3f} mm` and another hand geom at `{config['closure']['state_CAD_0p30']['other_hand_mm']:.3f} mm`.",
        f"Across the saved closure branch, first loop contact is qCAD `{config['closure']['first_loop_qCAD']}` and first bilateral pad proximity is qCAD `{config['closure']['first_bilateral_pad_qCAD']}`; loop collision precedes bilateral support: `{not config['closure']['pads_before_loop']}`.",
        "", "## Historical dynamic tracking note (read only)", "",
    ]
    if dynamic_pose:
        lines.append(f"At step {dynamic_pose['step']} ({dynamic_pose['time_s']:.3f} s, {dynamic_pose['phase']}), shoulder-pitch target was {dynamic_pose['target_rad']:.6f} rad and actual was {dynamic_pose['actual_rad']:.6f} rad (lag {dynamic_pose['target_error_rad']:.6f} rad); measured qvel {dynamic_pose['measured_velocity_rad_s']:.6f} rad/s. At that sample hand support was {dynamic_pose['hand_vertical_support_N']:.3f} N and table support {dynamic_pose['table_vertical_support_N']:.3f} N. This is historical evidence, not a new trajectory test.")
    else:
        lines.append("The historical dynamic trace row was not available; shoulder tracking is not inferred.")
    lines.extend([
        "", "## Previously tested Candidate B configurations (reused evidence; not rerun)", "",
        "| Candidate | D/H (mm) | Mass (kg) | Height offset (mm) | Open hand/object min (mm) | Bilateral pads at CAD .30 | Loop at CAD .30 | Static result |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ])
    for prior in result["candidate_configurations"]:
        lines.append(
            f"| {prior['id']} | {prior['object']['diameter_mm']:.1f}/{prior['object']['height_mm']:.1f} "
            f"| {prior['object']['mass_kg']:.3f} | {prior['pose']['offset_from_midbody_mm']:.1f} "
            f"| {prior['open_path']['min_clearances_mm']['hand_object']['distance_mm']:.3f} "
            f"| {prior['closure']['bilateral_at_CAD_0p30']} | {prior['closure']['loop_at_CAD_0p30']} "
            f"| {'PASS' if prior['static_candidate_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "", "## Integration interfaces and next unlock", "",
        "- Pose input: frame-stamped `PoseSample`; simulator ground truth carries exact model identity, RGB-D measurement carries camera frame, acquisition time domain, calibration ID, generation/sequence when available, and local monotonic receipt time for freshness. Transform every edge at acquisition time; no latest-TF fallback.",
        "- Docking handoff: navigation success plus fresh measured base pose, matching frame/generation/sequence, tolerance, low measured velocity, and stationary dwell. The existing Issue 80 `nav_msgs/Odometry`/`PoseStamped` messages have no generation/sequence fields; a lifecycle metadata bridge is still BLOCKED and none is synthesized here. No cmd_vel integration or Local03 connection.",
        f"- RGB-D object localization runtime: NOT RUN; this compiled model has {model.ncam} cameras and {model.nsensor} sensors. The pose/time/calibration/TF contract is prepared only.",
        "- Future gripper gate consumes full model identity, object-local dimensions and collision geoms, tool/robot poses, the actual open-to-close joint-state profile, pad and linkage geom names, explicit path-pair inventory, and contact-band interval. Incomplete collision coverage or loop-before-pad contact is BLOCKED.",
        "- Current screen output does not retain a complete collision pair matrix or object-local pad contact-band witness; the gate intentionally refuses to promote that evidence to a complete geometry PASS.",
        "- Resume physical grasp only after Local04 publishes the corrected, source-identified opening/closing model; rerun full-model static path and complete pairwise geometry gate, then perform dynamic arm tracking with no obstacles, then grasp/load transfer/lift under the existing physical rules.",
        "", "## Reproduction and provenance", "",
        f"Source screen: `{source}`; source artifact manifest verified ({len(verified)} files).",
        f"Static audit executable SHA-256: `{sha256(Path(__file__).resolve())}`.",
        "Reproduction command is listed in `REPRODUCE.md`. The new audit compiles the saved MJCF for identity/limit reading, but makes no `mj_forward` or `mj_step` calls.",
    ])
    return result_object, "\n".join(lines)+"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-parent", default=str(AUTHORIZED_ROOT / "issue-46"))
    args = parser.parse_args()
    parent = pg.validate_output_path(args.output_parent, authorized_roots=(str(AUTHORIZED_ROOT),),
                                     require_exists=True, expected_kind="directory", environment="wsl")
    summary, report = make_report(args)
    location = pg.create_unique_output_directory(parent.wsl_path, prefix="candidate-b-approach-ik-audit",
                                                authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
    out = Path(location.wsl_path)
    try:
        summary["run_id"] = str(uuid.uuid4())
        summary["output_path"] = location.as_dict()
        summary["implementation_branch"] = git_value(REPO_ROOT, "branch", "--show-current")
        summary["implementation_base_head"] = git_value(REPO_ROOT, "rev-parse", "HEAD")
        summary["audit_runner_sha256"] = sha256(Path(__file__).resolve())
        summary["prototype_code_sha256"] = prototype_code_hashes()
        command = (
            "env PYTHONDONTWRITEBYTECODE=1 "
            "/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python "
            f"{Path(__file__).resolve()} --source-run {args.source_run} --output-parent {args.output_parent}"
        )
        reproduction = ("WSL command (read-only saved evidence; static only):\n\n"+command+
                        "\n\nNo MuJoCo step or forward calls are made. Output is uniquely reserved under canonical reviews/.\n")
        pg.write_new_text(out / "result.json", json.dumps(summary, indent=2, ensure_ascii=False)+"\n",
                          authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
        pg.write_new_text(out / "REPORT.md", report, authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
        pg.write_new_text(out / "REPRODUCE.md", reproduction, authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
        manifest = pg.write_artifact_manifest(out, authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
        pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(AUTHORIZED_ROOT),), environment="wsl")
        print(json.dumps({"output": location.as_dict(), "run_id": summary["run_id"],
                          "ik_status": summary["ik_witness"]["status"],
                          "approach_status": summary["approach_collision_witness"]["status"],
                          "physics_steps": summary["new_physics_steps"],
                          "mj_forward_calls": summary["new_mj_forward_calls"],
                          "manifest_artifacts": len(manifest["artifacts"])}, indent=2))
        return 0
    except Exception:
        # Keep the reserved empty directory as a failed attempt; do not overwrite or reuse it.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
