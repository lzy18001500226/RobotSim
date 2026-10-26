#!/usr/bin/env python3
"""Static full-body RAICOM table-route audit using existing scene/frame helpers.

Offline evidence only: qpos placement and mj_forward; no integration, ROS or MC.
The geometric STOP_AND_ALIGN segment is explicitly not a qualified MC primitive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from issue80_local01_scene_route import (
    LOCAL01_SCENE_RELATIVE, _world_geom_aabb, load_local01_scene_inputs,
    native_name_scene_tree, transform_scene_pose_to_odom,
    RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,
    build_runtime_raw_odom_table_approach_mission,
    validate_runtime_raw_odom_table_approach_mission,
)
from issue80_raicom_reset_only import LOCAL01_POSES_DEFAULT, RUNTIME
from issue80_waypoint_route import WAYPOINT_DOCKING_POSE_SCHEMA_VERSION, load_waypoint_docking_pose_input


def sample_segment(a, b, spacing=0.025, yaw_spacing=0.02):
    delta = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
    count = max(2, math.ceil(math.dist(a[:3], b[:3])/spacing)+1,
                math.ceil(abs(delta)/yaw_spacing)+1)
    return [tuple(a[k]+(b[k]-a[k])*t/(count-1) for k in range(3))
            + (a[3]+delta*t/(count-1),) for t in range(count)]


def build_raw_world_odom_route_config(poses):
    """Keep executable-schema odom coordinates consistent with observed raw HAL.

    Local01's planned-local transform remains a separate report field. This
    output is still OFFLINE_ONLY: a Reset anchor is not an active-motion frame
    qualification, nor does this supply a qualified initial turning primitive.
    """
    by_name = {item["name"]: item["scene_world"] for item in poses}
    config = {"schema_version": WAYPOINT_DOCKING_POSE_SCHEMA_VERSION,
              "route_id": "raicom-source-table-raw-world-odom-OFFLINE-ONLY",
              "frame_id": "odom"}
    for dest, name in (("start", "START"), ("waypoint", "WAYPOINT"),
                       ("table_docking", "TABLE_DOCKING")):
        pose = by_name[name]
        config[dest] = {"x_m": float(pose["xyz_m"][0]),
                        "y_m": float(pose["xyz_m"][1]),
                        "yaw_rad": float(pose["yaw_rad"])}
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_native_standing_qpos(trace_path: Path, model) -> tuple[list[float], dict]:
    """Load one already-captured native MjData posture without stepping physics.

    The trace is evidence input, never a live state source.  A full compiled
    shape check prevents applying a qpos vector from another robot/scene.
    """
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("native standing trace must be a regular file")
    metadata = None
    candidate = None
    truncated_final_record = False
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    for index, raw in enumerate(lines):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                # Native capture records are append-only; preserve and label a
                # final interrupted write, while never silently accepting a
                # malformed interior record.
                truncated_final_record = True
                break
            raise ValueError(f"native standing trace has malformed record {index + 1}")
        if item.get("event") == "model_metadata":
            metadata = item
        if item.get("event") == "post_step" and item.get("generation") == 1:
            qpos = item.get("qpos")
            if isinstance(qpos, list):
                candidate = item
    if not isinstance(metadata, dict) or not isinstance(candidate, dict):
        raise ValueError("native standing trace lacks model metadata or generation-1 qpos")
    expected = {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nbody": int(model.nbody), "ngeom": int(model.ngeom),
    }
    actual = {name: metadata.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"native standing trace compiled shape mismatch: {actual} != {expected}")
    qpos = [float(value) for value in candidate["qpos"]]
    if len(qpos) != model.nq or not all(math.isfinite(value) for value in qpos):
        raise ValueError("native standing trace qpos is not a finite compiled-model vector")
    return qpos, {
        "source": str(trace_path),
        "sha256": _sha256(trace_path),
        "model_shape": expected,
        "native_mujoco_runtime_version": metadata.get("mujoco_runtime_version"),
        "sim_timestep_s": metadata.get("timestep_s"),
        "selected_sim_time_s": candidate.get("sim_time_s"),
        "selected_generation": candidate.get("generation"),
        "truncated_final_record_ignored": truncated_final_record,
    }


def _geom_descriptor(model, geom_id: int) -> dict:
    body_id = int(model.geom_bodyid[geom_id])
    return {
        "id": int(geom_id),
        "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body_id": body_id,
        "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "geom_type": int(model.geom_type[geom_id]),
        "mesh_dataid": int(model.geom_dataid[geom_id]),
    }


def _signed_distance_diagnostic(model, data, geom_a: int, geom_b: int, np) -> dict:
    """Record MuJoCo's signed-distance result without overclaiming mesh accuracy.

    MuJoCo 3.3.6's legacy mesh path can return a zero distance while the
    returned witness points are visibly separated.  A nonzero witness mismatch
    is therefore an *unresolved* narrow-phase result, never collision proof.
    """
    witness = np.zeros(6)
    signed = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, 2.0, witness))
    witness_gap = float(np.linalg.norm(witness[:3] - witness[3:]))
    if signed < 0.0:
        classification = "SIGNED_PENETRATION_REPORTED"
    elif signed == 0.0 and witness_gap > 1e-6:
        classification = "LEGACY_MESH_DISTANCE_UNRELIABLE_ZERO_WITH_SEPARATED_WITNESS"
    elif abs(signed - witness_gap) <= 1e-5:
        classification = "SIGNED_DISTANCE_WITNESS_CONSISTENT"
    elif signed >= 1.999999:
        classification = "NO_PAIR_WITHIN_DISTMAX"
    else:
        classification = "SIGNED_DISTANCE_WITNESS_MISMATCH_UNRESOLVED"
    return {
        "geom_a": _geom_descriptor(model, geom_a),
        "geom_b": _geom_descriptor(model, geom_b),
        "mujoco_signed_distance_m": signed,
        "witness_endpoint_separation_m": witness_gap,
        "witness_segment_world": witness.tolist(),
        "classification": classification,
        "mujoco_version": mujoco.__version__,
        "distmax_m": 2.0,
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-standing-qpos-trace", type=Path, default=None,
        help="read-only native MjData trace used as a third static posture",
    )
    args = parser.parse_args(argv)
    source = load_local01_scene_inputs(LOCAL01_POSES_DEFAULT)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
    from path_governance import create_unique_output_directory, write_artifact_manifest
    output = create_unique_output_directory(
        "/home/lzy18001500226/projects/RobotSim/reviews/issue-80", prefix="raicom-table-plan"
    ).native_path
    tree = native_name_scene_tree(Path(source["repository_root"])/LOCAL01_SCENE_RELATIVE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qadr = int(model.jnt_qposadr[root])
    bodies = {pelvis}
    for body in range(pelvis+1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    robot = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    environment = [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(("raicom_", "mug_"))
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
    cfg = yaml.safe_load((RUNTIME/"sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml").read_text())
    nominal = dict(zip([j for group in cfg["actual"]["active_joint_name"] for j in group],
                       [q for group in cfg["logical"]["nominal_configuration"] for q in group]))
    posture_specs: list[tuple[str, list[float] | None, dict]] = [
        ("source_qpos0", None, {"classification": "SOURCE_QPOS0"}),
        ("runtime_nominal", None, {"classification": "RUNTIME_DEFAULT_YAML_NOMINAL"}),
    ]
    if args.native_standing_qpos_trace is not None:
        qpos, trace_provenance = _load_native_standing_qpos(args.native_standing_qpos_trace, model)
        posture_specs.append(("native_standing_trace", qpos, {
            "classification": "READ_ONLY_NATIVE_MJDATA_STANDING_POSTURE",
            **trace_provenance,
        }))

    def configure(data, posture_name: str, posture_qpos: list[float] | None, p, yaw_offset: float) -> None:
        mujoco.mj_resetData(model, data)
        if posture_qpos is not None:
            data.qpos[:] = posture_qpos
        elif posture_name == "runtime_nominal":
            for name, q in nominal.items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[jid]] = q
        data.qpos[qadr:qadr+3] = p[:3]
        yaw = p[3] + yaw_offset
        data.qpos[qadr+3:qadr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        mujoco.mj_forward(model, data)

    points = source["scene_world_poses"]["poses"]
    pose = lambda name: tuple(points[name]["xyz_m"])+(float(points[name]["yaw_rad"]),)
    waypoints = [("START",pose("START")), ("START_ALIGN",pose("START_ALIGN")),
                 ("WAYPOINT",pose("WAYPOINT")), ("TABLE_APPROACH",(.90,-1.50,.68,0.0)),
                 ("TABLE_DOCKING",pose("TABLE_DOCKING"))]
    results = []
    rows = []
    segments = list(zip(waypoints,waypoints[1:]))
    segments.append((waypoints[-2], ("TABLE_REVIEW_CANDIDATE",(.95,-1.50,.68,0.0))))
    for posture, posture_qpos, posture_provenance in posture_specs:
        for index, ((aname,a),(bname,b)) in enumerate(segments):
            samples = sample_segment(a,b)
            contacts = []
            self_contacts = {}
            limit_violations = {}
            min_raw = math.inf
            min_inflated = math.inf
            limiting = None
            previous_boxes = None
            for step, p in enumerate(samples):
                sample_min_inflated = math.inf
                sample_limiting = None
                for yaw_offset in (-RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD, 0,
                                   RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD):
                    data = mujoco.MjData(model)
                    configure(data, posture, posture_qpos, p, yaw_offset)
                    for jid in range(model.njnt):
                        if model.jnt_limited[jid] and model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                            q = float(data.qpos[model.jnt_qposadr[jid]])
                            if q < model.jnt_range[jid,0]-1e-10 or q > model.jnt_range[jid,1]+1e-10:
                                limit_violations[jid] = q
                    for contact in data.contact[:data.ncon]:
                        g1,g2 = int(contact.geom1),int(contact.geom2)
                        if g1 in robot and g2 in robot and contact.dist < -1e-6:
                            pair = str(tuple(sorted((g1,g2))))
                            self_contacts[pair] = min(self_contacts.get(pair,0),float(contact.dist))
                        if (g1 in robot and g2 in environment) or (g2 in robot and g1 in environment):
                            contacts.append({"sample":step,"distance_m":float(contact.dist),
                                             "geom_ids":[g1,g2]})
                    boxes = {g:_world_geom_aabb(model,data,g,np) for g in robot}
                    envboxes = {g:_world_geom_aabb(model,data,g,np) for g in environment}
                    for g,(lo,hi) in boxes.items():
                        # Union adjacent translation samples; include a conservative
                        # yaw interpolation displacement bound for every mesh vertex.
                        if previous_boxes is not None:
                            lo = np.minimum(lo,previous_boxes[g][0]); hi=np.maximum(hi,previous_boxes[g][1])
                        radius = float(np.linalg.norm(data.geom_xpos[g]-data.xpos[pelvis])+model.geom_rbound[g])
                        yaw_pad = radius*RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD
                        ilo,ihi = lo.copy(),hi.copy()
                        ilo[:2] -= .05+yaw_pad; ihi[:2] += .05+yaw_pad
                        for eg,(elo,ehi) in envboxes.items():
                            if not ((model.geom_contype[g]&model.geom_conaffinity[eg]) or
                                    (model.geom_contype[eg]&model.geom_conaffinity[g])):
                                continue
                            raw = float(np.linalg.norm(np.maximum(0,np.maximum(elo-hi,lo-ehi))))
                            inflated=float(np.linalg.norm(np.maximum(0,np.maximum(elo-ihi,ilo-ehi))))
                            min_raw=min(min_raw,raw)
                            if inflated < sample_min_inflated:
                                sample_min_inflated=inflated
                                sample_limiting={"robot_geom_id":g,"environment_geom_id":eg,
                                    "robot":_geom_descriptor(model,g),
                                    "environment":_geom_descriptor(model,eg),
                                    "sample":step,"scene_world_pose":list(p),
                                    "yaw_offset_rad":yaw_offset}
                            if inflated < min_inflated:
                                min_inflated=inflated
                                limiting=sample_limiting
                    previous_boxes=boxes
                rows.append({"posture":posture,"segment":index,"sample":step,
                             "scene_world_pose":p,"minimum_inflated_clearance_m":sample_min_inflated,
                             "limiting_pair":sample_limiting})
            narrow = None
            if isinstance(limiting, dict):
                data = mujoco.MjData(model)
                configure(data, posture, posture_qpos, tuple(limiting["scene_world_pose"]),
                          float(limiting["yaw_offset_rad"]))
                narrow = _signed_distance_diagnostic(
                    model, data, int(limiting["robot_geom_id"]),
                    int(limiting["environment_geom_id"]), np,
                )
                narrow["active_solver_contacts_for_pair"] = [
                    {"distance_m": float(c.dist), "geom_ids": [int(c.geom1), int(c.geom2)]}
                    for c in data.contact[:data.ncon]
                    if {int(c.geom1), int(c.geom2)} == {
                        int(limiting["robot_geom_id"]), int(limiting["environment_geom_id"])
                    }
                ]
            first_below = next((row for row in rows if row["posture"] == posture
                                and row["segment"] == index
                                and row["minimum_inflated_clearance_m"] < .020), None)
            results.append({"posture":posture,"posture_provenance":posture_provenance,"from":aname,"to":bname,
                            "samples":len(samples)*3,"active_environment_contacts":contacts,
                            "active_robot_self_penetrations_m":self_contacts,
                            "source_limit_violations":limit_violations,
                            "raw_aabb_lower_bound_m":min_raw,
                            "tracking_inflated_aabb_lower_bound_m":min_inflated,
                            "limiting_pair":limiting,
                            "first_tracking_inflated_below_20mm":first_below,
                            "limiting_pair_narrow_phase":narrow,
                            "status":"PASS_STATIC_ONLY" if not contacts and not self_contacts
                            and not limit_violations and min_inflated>=.020 else "BLOCKED_STATIC_CLEARANCE"})
    poses = []
    for name,p in waypoints:
        world={"xyz_m":list(p[:3]),"yaw_rad":p[3]}
        poses.append({"name":name,"scene_world":world,
                      "local01_reset_local_odom":transform_scene_pose_to_odom(world,source["odom_from_scene"]),
                      "runtime_raw_odom_convention":"SCENE_WORLD_RESET_ANCHOR_OBSERVED_20261026; independent live GT unavailable"})
    # Reuse the existing route-config shape, but do not feed it into a live
    # runner: the complete route has different distance/turning budgets.
    config = build_raw_world_odom_route_config(poses)
    config_path=output/"table_route_raw_world_ODOM_OFFLINE_ONLY.json"
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    load_waypoint_docking_pose_input(config_path)
    mission = build_runtime_raw_odom_table_approach_mission(source)
    validate_runtime_raw_odom_table_approach_mission(mission)
    mission_path = output / "safe_table_approach_mission_raw_world_ODOM_OFFLINE_ONLY.json"
    mission_path.write_text(json.dumps(mission, indent=2)+"\n")
    length=sum(math.dist(a[:3],b[:3]) for (_,a),(_,b) in zip(waypoints,waypoints[1:]))
    minimum_turn_radius=.12/.075
    report={"schema":"robotsim.issue80.raicom-table-plan.v1",
            "status":"STATIC_GEOMETRY_REVIEW_ONLY_TABLE_APPROACH_NOT_DYNAMICALLY_QUALIFIED",
            "source_commit":source["source_commit"],"source_hashes":source["source_hashes"],
            "mujoco_version":mujoco.__version__,"mj_step_calls":0,
            "robot_collision_geoms":len(robot),"environment_collision_geoms":len(environment),
            "waypoints":poses,"segments":results,"translation_distance_m":length,
            "route_config_coordinate_contract":{
                "file":config_path.name,"published_frame_id":"odom",
                "coordinate_values":"scene_world; matches observed raw HAL Reset anchor",
                "planned_local_values":"report only, never silently relabeled raw odom",
                "active_motion_frame_qualification":"NOT_RUN",
                "live_execution_enabled":False},
            "safe_table_approach_mission": {
                "file": mission_path.name,
                "classification": mission["classification"],
                "execution_order": mission["execution_order"],
                "table_docking_reference_unchanged": mission["table_docking_reference"]["source_pose_unchanged"],
                "initial_turn_status": mission["initial_alignment"]["zero_forward_angular_mc_status"],
                "bounded_runtime_budget": mission["bounded_runtime_budget"],
            },
            "separate_review_candidate":{"scene_world_xyz_yaw":[.95,-1.50,.68,0.0],
                 "reason":"50 mm retreat from source docking to preserve nominal-posture uncertainty clearance",
                 "source_docking_unchanged":True,"manipulation_reach":"UNKNOWN_NOT_AN_ACCEPTED_DOCK"},
            "clearance_assumptions":{"xy_tracking_inflation_m":.05,
                                     "yaw_error_samples_rad":[-RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,0,RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD],
                                     "extra_rotation_interpolation_bound_rad":RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,"additional_clearance_gate_m":.02,
                                     "postures":[name for name,_,_ in posture_specs],
                                     "dynamic_gait_envelope":"NOT_PROVEN_BY_STATIC_POSTURES"},
            "narrow_phase_interpretation": {
                "active_solver_contact": "A contact in data.contact is direct compiled-model contact evidence.",
                "signed_distance": "Use only SIGNED_DISTANCE_WITNESS_CONSISTENT values as supported separation measurements.",
                "legacy_mesh_zero": "A zero signed-distance with separated witnesses is unresolved legacy mesh distance behavior, not collision proof.",
                "aabb": "Tracking-inflated AABB overlap is conservative route gating only; it does not establish a physical contact.",
            },
            "docking_requirement":{"position_tolerance_m":.025,"yaw_tolerance_rad":.02,
                                   "controller_accuracy":"NOT_YET_VALIDATED","manipulation_IK":"NOT_VALIDATED",
                                   "source_target_preserved":True},
            "turning":{"initial_heading_change_rad":-math.pi/2,"primitive":"STOP_AND_ALIGN_GEOMETRY_ONLY",
                       "in_place_rotation_MC_qualification":"NOT_VALIDATED",
                       "moving_arc_radius_at_caps_m":minimum_turn_radius,
                       "right_quarter_arc_endpoint_world":[-1.5+minimum_turn_radius,-1.5+minimum_turn_radius],
                       "right_quarter_arc_length_m":minimum_turn_radius*math.pi/2,
                       "moving_arc_does_not_join_straight_table_corridor":True},
            "future_execution_budget_PROPOSAL_NOT_AUTHORIZATION":{
                "ideal_translation_minimum_s":length/.12,
                "historical_effective_progress_m_s":.171312/4.9709,
                "empirical_translation_estimate_s":length/(.171312/4.9709),
                "max_nonzero_window_s":5,"approximate_transit_windows":math.ceil(length/.171312),
                "post_window_STAND_and_composite_hold":"unchanged full gates; up to 7 seconds observation",
                "the_old_15_second_budget_cannot_reach":length>.12*15,
                "total_budget_decision":"DEFERRED_UNTIL_90_DEG_TURN_AND_DOCK_ACCURACY_QUALIFIED"},
            "longest_verified_prefix":"STATIC_START_ALIGN_AND_CORRIDOR_ONLY; dynamic prefix requires live Reset/standing and turn qualification",
            "table_approach_live_execution":"NOT_RUN_UNTIL_INITIAL_TURN_AND_FULL_MISSION_RUNTIME_GATES_PASS",
            "full_route_live_execution":"NOT_RUN_NOT_AUTHORIZED"}
    (output/"table_plan.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"samples.json").write_text(json.dumps(rows)+"\n")
    # Compact SVG map drawn from the actual compiled collision AABBs.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 850">',
         '<rect width="1000" height="850" fill="white"/>',
         '<text x="35" y="30" font-size="19">Pinned RAICOM: static table approach; dynamic turn / docking NOT qualified</text>']
    project=lambda x,y:(100+180*(x+2),770-180*(y+2))
    for g in environment:
        lo,hi=_world_geom_aabb(model,data,g,np)
        x,y=project(lo[0],hi[1]); w=(hi[0]-lo[0])*180;h=(hi[1]-lo[1])*180
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#d4d7dc" stroke="#6c737c"/>')
    coordinates=[project(p[0],p[1]) for _,p in waypoints]
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in coordinates)+'" fill="none" stroke="#176da6" stroke-width="4"/>')
    for (name,p),(x,y) in zip(waypoints,coordinates):
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#b52222"/><text x="{x+8}" y="{y-12}" font-size="12">{name}</text>')
    svg.append('<text x="35" y="820" font-size="14">Scene world metres. Full body +50 mm XY uncertainty; source, nominal, and optional native standing postures. No physics stepping.</text></svg>')
    (output/"table_route.svg").write_text('\n'.join(svg))
    write_artifact_manifest(output)
    print(json.dumps({"output":str(output),"segments":[{k:s[k] for k in ("posture","from","to","status","tracking_inflated_aabb_lower_bound_m")} for s in results]},indent=2))


if __name__=="__main__":
    main()
