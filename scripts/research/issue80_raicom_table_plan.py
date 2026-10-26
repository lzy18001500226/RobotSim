#!/usr/bin/env python3
"""Static full-body RAICOM table-route audit using existing scene/frame helpers.

Offline evidence only: qpos placement and mj_forward; no integration, ROS or MC.
The geometric STOP_AND_ALIGN segment is explicitly not a qualified MC primitive.
"""
from __future__ import annotations

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


def main():
    source = load_local01_scene_inputs(LOCAL01_POSES_DEFAULT)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
    from path_governance import create_unique_output_directory, write_artifact_manifest
    output = create_unique_output_directory(
        "/home/lzy18001500226/projects/RobotSim/reviews/issue-80", prefix="raicom-table-plan"
    ).native_path
    tree = native_name_scene_tree(Path(source["repository_root"])/LOCAL01_SCENE_RELATIVE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
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
    points = source["scene_world_poses"]["poses"]
    pose = lambda name: tuple(points[name]["xyz_m"])+(float(points[name]["yaw_rad"]),)
    waypoints = [("START",pose("START")), ("START_ALIGN",pose("START_ALIGN")),
                 ("WAYPOINT",pose("WAYPOINT")), ("TABLE_APPROACH",(.90,-1.50,.68,0.0)),
                 ("TABLE_DOCKING",pose("TABLE_DOCKING"))]
    results = []
    rows = []
    segments = list(zip(waypoints,waypoints[1:]))
    segments.append((waypoints[-2], ("TABLE_REVIEW_CANDIDATE",(.95,-1.50,.68,0.0))))
    for posture in ("source_qpos0", "runtime_nominal"):
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
                for yaw_offset in (-.02,0,.02):
                    mujoco.mj_resetData(model,data)
                    if posture == "runtime_nominal":
                        for name,q in nominal.items():
                            jid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                            data.qpos[model.jnt_qposadr[jid]] = q
                    data.qpos[qadr:qadr+3] = p[:3]
                    yaw = p[3]+yaw_offset
                    data.qpos[qadr+3:qadr+7] = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
                    mujoco.mj_forward(model,data)
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
                        yaw_pad = radius*.02
                        ilo,ihi = lo.copy(),hi.copy()
                        ilo[:2] -= .05+yaw_pad; ihi[:2] += .05+yaw_pad
                        for eg,(elo,ehi) in envboxes.items():
                            if not ((model.geom_contype[g]&model.geom_conaffinity[eg]) or
                                    (model.geom_contype[eg]&model.geom_conaffinity[g])):
                                continue
                            raw = float(np.linalg.norm(np.maximum(0,np.maximum(elo-hi,lo-ehi))))
                            inflated=float(np.linalg.norm(np.maximum(0,np.maximum(elo-ihi,ilo-ehi))))
                            min_raw=min(min_raw,raw)
                            if inflated < min_inflated:
                                min_inflated=inflated
                                limiting={"robot_geom_id":g,"environment_geom_id":eg,
                                          "environment_body":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[eg]))}
                    previous_boxes=boxes
                rows.append({"posture":posture,"segment":index,"sample":step,
                             "scene_world_pose":p,"minimum_inflated_clearance_m":min_inflated})
            results.append({"posture":posture,"from":aname,"to":bname,
                            "samples":len(samples)*3,"active_environment_contacts":contacts,
                            "active_robot_self_penetrations_m":self_contacts,
                            "source_limit_violations":limit_violations,
                            "raw_aabb_lower_bound_m":min_raw,
                            "tracking_inflated_aabb_lower_bound_m":min_inflated,
                            "limiting_pair":limiting,
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
    length=sum(math.dist(a[:3],b[:3]) for (_,a),(_,b) in zip(waypoints,waypoints[1:]))
    minimum_turn_radius=.12/.075
    report={"schema":"robotsim.issue80.raicom-table-plan.v1",
            "status":"STATIC_GEOMETRY_REVIEW_ONLY_DYNAMIC_DOCKING_NOT_QUALIFIED",
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
            "separate_review_candidate":{"scene_world_xyz_yaw":[.95,-1.50,.68,0.0],
                 "reason":"50 mm retreat from source docking to preserve nominal-posture uncertainty clearance",
                 "source_docking_unchanged":True,"manipulation_reach":"UNKNOWN_NOT_AN_ACCEPTED_DOCK"},
            "clearance_assumptions":{"xy_tracking_inflation_m":.05,"yaw_error_samples_rad":[-.02,0,.02],
                                     "extra_rotation_interpolation_bound_rad":.02,"additional_clearance_gate_m":.02,
                                     "postures":["source_qpos0","runtime_nominal"],
                                     "dynamic_gait_envelope":"NOT_PROVEN_BY_TWO_STATIC_POSTURES"},
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
            "full_route_live_execution":"NOT_RUN_NOT_AUTHORIZED"}
    (output/"table_plan.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"samples.json").write_text(json.dumps(rows)+"\n")
    # Compact SVG map drawn from the actual compiled collision AABBs.
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
    svg.append('<text x="35" y="820" font-size="14">Scene world metres. Full body +50 mm XY uncertainty; source and nominal postures. No physics stepping.</text></svg>')
    (output/"table_route.svg").write_text('\n'.join(svg))
    write_artifact_manifest(output)
    print(json.dumps({"output":str(output),"segments":[{k:s[k] for k in ("posture","from","to","status","tracking_inflated_aabb_lower_bound_m")} for s in results]},indent=2))


if __name__=="__main__":
    main()
