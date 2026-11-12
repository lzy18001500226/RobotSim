"""Task-level RGB-D grasp-region selection using frozen IK and existing navigation.

Research consumer only. Source assets and physical limits are immutable. Pose
writes below affect offline planning MjData; the live bottle is never written.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from scripts.agent import path_governance as pg

ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")


def control_observation(estimated, native, *, now_ns=None):
    """Envelope a successful estimator return; keep its original freshness gate."""
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    now = time.monotonic_ns() if now_ns is None else now_ns
    record = dict(estimated, **{"pass": True}, estimated_center_world_m=[
        *estimated["center_xy_world_m"], estimated["top_world_z_m"]-.11],
        source_host_ns=native["host_monotonic_ns"], created_host_ns=now,
        generation=native["generation"], object_truth_used_by_estimator=False)
    vision.qualify_estimate(record, generation=native["generation"], now_ns=now)
    return record


def acquire_control_observation(read_native, capture, *, generation, after_source_ns, attempts, timeout_s=3.0):
    """Discard slow image results; preserve the unchanged 200ms source gate."""
    deadline = time.monotonic_ns()+int(timeout_s*1e9)
    while time.monotonic_ns() < deadline:
        native = read_native()
        stamp = native["host_monotonic_ns"]
        if native["generation"] != generation:
            raise RuntimeError("TASK_RGBD_GENERATION_CHANGED")
        if stamp < after_source_ns:
            raise RuntimeError("TASK_RGBD_SOURCE_REGRESSED")
        if stamp == after_source_ns:
            time.sleep(.005)
            continue
        began = time.monotonic_ns()
        rgb, estimated = capture(native)
        ended = time.monotonic_ns()
        record = {"source_host_ns": stamp, "generation": generation,
                  "completed_host_ns": ended, "processing_s": (ended-began)*1e-9,
                  "source_age_s": (ended-stamp)*1e-9, "accepted": False}
        attempts.append(record)
        if ended > deadline:
            record["failure"] = "TASK_RGBD_ACQUISITION_DEADLINE"
            raise RuntimeError(record["failure"])
        try:
            observation = control_observation(estimated, native, now_ns=ended)
        except RuntimeError as exc:
            record["failure"] = str(exc)
            if not str(exc).startswith("STALE_RGBD_SOURCE:"):
                raise
            # Never refresh the rejected image's stamp or use its estimated
            # target. Only a new source frame may qualify the next attempt.
            after_source_ns = stamp
            continue
        record["accepted"] = True
        return native, rgb, observation
    raise RuntimeError("TASK_RGBD_ACQUISITION_TIMEOUT")

def task_route_profile(profile):
    """Select the short checked turn; preserve every physical/control bound."""
    import copy
    result = copy.deepcopy(profile)
    if (result.get("schema") != "robotsim.x2.two-table-delivery.v1"
            or result.get("transit_root_cap_m_s") != .60
            or result.get("turn_translation_strategy") != "PURE_YAW"):
        raise ValueError("unexpected frozen task route profile")
    result["delivery_turn_direction"] = "SHORTEST"
    result["turn_translation_strategy"] = "FORWARD_ARC"
    # The previous 250mm early stop left the actual arm outside its placement
    # region. Reuse the existing measured-braking predictor; actual-pose
    # placement IK, 12deg tilt and all physical stop gates remain mandatory.
    result["stop_inside_functional_arrival_region"] = False

    result["task_turn_selection"] = {
        "classification": "TASK_COLLISION_CHECKED_FORWARD_ARC_NOT_MC_RETUNING",
        "evidence": "reviews/issue-46/x2-table-edge-task-20261111T145311Z/carry_radius333_arc_static.json",
        "live_measured_route_screen_still_required": True,
        "model_gains_limits_friction_unchanged": True}
    return result


def stage_runtime_posture(output, selected_path):
    """Merge only selected pose targets; preserve enriched startup safety data."""
    import copy
    base = json.loads((output/"standing_grasp_proposal.json").read_text())
    selected = json.loads(Path(selected_path).read_text())
    allowed = {"waist_target_rad", "torso_world_yaw_target_rad", "rgbd_posture_selection"}
    for key, value in selected.items():
        if key not in allowed and base.get(key) != value:
            raise ValueError("task posture may not change protected field:"+key)
    effective = copy.deepcopy(base)
    for key in allowed:
        if key in selected:
            effective[key] = selected[key]
    path = output/"task_effective_standing_grasp_proposal.json"
    pg.write_new_text(path, json.dumps(effective, indent=2)+"\n")
    return path


def reference_contract():
    return {"schema": "robotsim.rgbd-task-grasp-region.v3",
            "control_object_source": "FRESH_RGBD_ONLY",
            "existing_navigation_controller": "PointGoalController",
            "task_motion_mapping": "MC_BODY_XY_NO_PIVOT_SOURCE_CAP020",
            "maximum_stance_adjustments": 2,
            "posture_reference": str(Path(__file__).resolve().parents[4]/"simulation/mujoco/scenes/raicom2026/config/reviewed_h220_turn_forward_bend_pickup_functional.json"),
            "maximum_candidate_shift_m": .80, "maximum_navigation_excursion_m": .90,
            "translation_tube_xy_m": .05, "translation_tube_heading_rad": .20,
            "task_translation_position_tolerance_m": .06,
            "mechanical_changes": False}


def recover_clock_at_stop(node, baseline, generation, recoveries):
    """Reuse carry's zero/reacquire/composite-stop protocol, at most once here."""
    if (node.feedback_fault or {}).get("gate") != "CLOCK_STEP_DURING_MOTION":
        raise RuntimeError("TASK_CLOCK_RECOVERY_NOT_A_CLOCK_FAULT")
    if recoveries >= 1:
        raise RuntimeError("TASK_CLOCK_RECOVERY_BUDGET")
    node.deadline_zero()
    node.stop_started = True
    if node.moving:
        raise RuntimeError("TASK_CLOCK_RECOVERY_STILL_COMMANDING")
    node.feedback_fault = None
    node.pump()
    if node.latest_row["generation"] != generation:
        raise RuntimeError("TASK_CLOCK_RECOVERY_SIMULATION_RESET")
    # sample/observe still enforce 200ms, all source/physical gates, and a
    # complete new-generation stop window. Clearing the clock latch is not PASS.
    node.observe(1.05, baseline=baseline)
    node.stand_once()
    stop = node.observe(12.0, baseline=baseline, require_stop=True)
    if not stop["pass"] or node.latest_row["generation"] != generation:
        raise RuntimeError("TASK_CLOCK_RECOVERY_STOP_NOT_QUALIFIED")
    return {"stop": stop, "generation": generation, "time_monotonic_s": time.monotonic()}


def body_translation_velocity(pose, goal_xy, *, gain=.6, speed_cap=.20):
    """Map the existing point error to MC body-X/Y velocity, without pivoting."""
    values=[pose["x"],pose["y"],pose["yaw"],*goal_xy,gain,speed_cap]
    if not all(math.isfinite(v) for v in values) or gain<=0 or not 0<speed_cap<=.20:
        raise ValueError("invalid bounded MC translation input")
    dx,dy=goal_xy[0]-pose["x"],goal_xy[1]-pose["y"]
    c,s=math.cos(pose["yaw"]),math.sin(pose["yaw"])
    forward,lateral=gain*(c*dx+s*dy),gain*(-s*dx+c*dy)
    scale=min(1.,speed_cap/max(math.hypot(forward,lateral),1e-12))
    return forward*scale,lateral*scale,0.


def translation_corridor_error(pose, start_xy, goal_xy, heading):
    dx,dy=goal_xy[0]-start_xy[0],goal_xy[1]-start_xy[1]
    t=max(0.,min(1.,((pose["x"]-start_xy[0])*dx+(pose["y"]-start_xy[1])*dy)/max(dx*dx+dy*dy,1e-12)))
    distance=math.hypot(pose["x"]-start_xy[0]-t*dx,pose["y"]-start_xy[1]-t*dy)
    yaw_error=abs((pose["yaw"]-heading+math.pi)%(2*math.pi)-math.pi)
    return distance,yaw_error


def run_region_navigation(node, contract):
    """Existing PointGoalController and native Micro transport; no foot controller."""
    import sys
    import threading

    sys.path.insert(0, "/workspace/robotsim/ros2_ws/src/robotsim_navigation")
    from robotsim_navigation.point_goal_controller import (
        ControllerConfig, Goal2D, NavState, PointGoalController, Pose2D,
    )

    cfg = MICRO["task_region"]
    result = {"classification": "SIMULATION_ONLY_EXISTING_POINT_GOAL_TO_GRASP_REGION",
              "pass_result": False, "clock_recoveries": [], "goal": cfg["goal"],
              "failure": None,
              "motion_input_profile": "MC_BODY_XY_NO_PIVOT_SOURCE_CAP020",
              "maximum_command_norm_m_s": .20, "angular_command_rad_s": 0.}
    watchdog = None
    baseline = None
    try:
        node.start_feedback()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            node.publish()
            node.pump()
            try:
                row = node.sample()
                break
            except RuntimeError as exc:
                if not (str(exc) in ("CLOCK_GENERATION_PENDING", "NATIVE_FEEDBACK_STALE_OR_INVALID")
                        or str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:")):
                    raise
            time.sleep(.01)
        else:
            raise RuntimeError("TASK_REGION_READINESS_TIMEOUT")
        node.await_graph()
        baseline = row["body_status"]
        result["pre_motion_stop"] = node.observe(8.0, baseline=baseline, require_stop=True)
        origin = node.sample()["pose"]
        planned_start = cfg.get("path_start_xy_m") or [origin["x"],origin["y"]]
        if math.hypot(origin["x"]-planned_start[0],origin["y"]-planned_start[1])>.01:
            raise RuntimeError("TASK_TRANSLATION_PLAN_SOURCE_MOVED")
        generation = node.latest_row["generation"]
        goal = Goal2D(*cfg["goal"], "native_scene_world")
        controller = PointGoalController(ControllerConfig(
            xy_tolerance_m=.06, yaw_tolerance_rad=.25,
            max_forward_mps=.20, max_angular_rps=.40, goal_timeout_s=45.0))
        until = time.monotonic() + 45.0
        node.stand_once("LOCOMOTION_STEP")
        node.observe(.2, baseline=baseline)
        with node.lock:
            node.moving = True
        node.stop_started = False
        watchdog = threading.Timer(max(.01, until-time.monotonic()), node.deadline_zero)
        watchdog.daemon = True
        watchdog.start()
        last_progress = time.monotonic()
        best = math.inf
        last_nav_state = None
        source_acquired = False
        activation_started = time.monotonic()
        while time.monotonic() < until:
            tick = time.monotonic()
            try:
                node.pump()
                row = node.sample()
            except RuntimeError:
                if (node.feedback_fault or {}).get("gate") != "CLOCK_STEP_DURING_MOTION":
                    raise
                recovered = recover_clock_at_stop(node, baseline, generation, len(result["clock_recoveries"]))
                result["clock_recoveries"].append(recovered)
                if until-time.monotonic() < 5.0:
                    raise RuntimeError("TASK_CLOCK_RECOVERY_TASK_DEADLINE")
                controller.reset()
                node.await_graph()
                node.stand_once("LOCOMOTION_STEP")
                with node.lock:
                    node.moving = True
                node.stop_started = False
                last_progress = time.monotonic()
                best, last_nav_state = math.inf, None
                activation_started = time.monotonic()
                source_acquired = False
                continue
            if not node.moving or row["body_status"] != baseline:
                raise RuntimeError("TASK_REGION_WATCHDOG_OR_BODY_STATUS")
            pose = row["pose"]
            if math.hypot(pose["x"]-origin["x"], pose["y"]-origin["y"]) > cfg["maximum_excursion_m"]:
                raise RuntimeError("TASK_REGION_EXCURSION")
            for name, cap in cfg["effort_caps"].items():
                actual = node.latest_row["actuators"].get(name)
                if actual is None or not math.isfinite(actual["force"]) or abs(actual["force"]) > cap+1e-5:
                    raise RuntimeError("TASK_REGION_SOURCE_EFFORT:"+name)
            control_now = time.monotonic()
            output = controller.step(
                Pose2D(pose["x"], pose["y"], pose["yaw"], "native_scene_world", row["sample_monotonic_s"]),
                goal, now_steady_s=control_now)
            row["controller_evaluated_steady_s"] = control_now
            row["controller_pose_age_s"] = control_now-row["sample_monotonic_s"]
            row["controller_state"] = output.state.value
            node.trace.append(row)
            source = row.get("common_source",{}).get("name")
            source_acquired = source_acquired or (source=="robotsim_official_cmd_vel"
                and row.get("common_mode_desc")=="LOCOMOTION_STEP"
                and row.get("get_mode_desc")=="LOCOMOTION_STEP")
            if not source_acquired and time.monotonic()-activation_started>.75:
                raise RuntimeError("TASK_TRANSLATION_SOURCE_AND_MODE_NOT_OBSERVED")
            if output.state not in (NavState.ARRIVED,NavState.SETTLING,NavState.ALIGN,NavState.DRIVE,NavState.FINAL_ALIGN):
                raise RuntimeError("TASK_REGION_CONTROLLER:"+output.state.value+":pose_age="+str(row["controller_pose_age_s"]))
            cross, heading_error = translation_corridor_error(pose,planned_start,cfg["goal"][:2],cfg["goal"][2])
            row["translation_cross_track_m"],row["translation_heading_error_rad"]=cross,heading_error
            if cross>.05 or heading_error>.20:
                result["task_replan_pause"]={"reason":"LEFT_SAMPLED_TRANSLATION_TUBE",
                    "cross_track_m":cross,"heading_error_rad":heading_error}
                break
            if output.distance_m is not None and output.distance_m<=controller.config.xy_tolerance_m:
                result["position_region_entered"]=True
                result["legacy_heading_error_rad"]=output.yaw_error_rad
                break
            if output.state in (NavState.ARRIVED, NavState.SETTLING):
                node.publish()
                if output.state == NavState.ARRIVED:
                    break
            elif output.state in (NavState.ALIGN, NavState.DRIVE, NavState.FINAL_ALIGN):
                if output.state != last_nav_state:
                    best, last_progress = math.inf, tick
                    last_nav_state = output.state
                metric = output.distance_m
                if metric < best-.01:
                    best, last_progress = metric, tick
                if tick-last_progress > 5.0:
                    raise RuntimeError("TASK_REGION_NO_MEASURED_PROGRESS")
                command = body_translation_velocity(pose,cfg["goal"][:2],
                    gain=controller.config.k_linear,speed_cap=controller.config.max_forward_mps)
                node.publish(*command)
            else:
                raise RuntimeError("TASK_REGION_CONTROLLER:"+output.state.value+":pose_age="+str(row["controller_pose_age_s"]))
            time.sleep(max(0.0, tick+.02-time.monotonic()))
        else:
            raise RuntimeError("TASK_REGION_TIMEOUT")
        node.deadline_zero()
        node.stop_started = True
        node.observe(1.05, baseline=baseline)
        node.stand_once()
        result["post_stop_composite"] = node.observe(12.0, baseline=baseline, require_stop=True)
        result["end_pose"] = node.sample()["pose"]
        result["source_and_step_observed"] = source_acquired
        result["pass_result"] = bool(source_acquired and result["post_stop_composite"]["pass"])
        result["scope"] = "SAFE_UNLOADED_STOP_ONLY; FRESH_RGBD_AND_GRASP_REGION_MUST_BE_RECHECKED"
        result["point_heading_acceptance"] = "DEFERRED_TO_ACTUAL_GRASP_IK_NOT_REQUIRED_FOR_TRANSLATION"
    except Exception as exc:
        node.deadline_zero()
        node.stop_started = True
        result["failure"] = str(exc)
        try:
            node.observe(1.05, baseline=baseline)
            node.stand_once()
            result["abort_stop_composite"] = node.observe(12.0, baseline=baseline, require_stop=True)
        except Exception as stop_exc:
            result["abort_stop_failure"] = str(stop_exc)
    finally:
        node.deadline_zero()
        if watchdog:
            watchdog.cancel()
        node.stop_feedback()
        result.update(trace=node.trace, commands=node.command_log, events=node.events,
                      bridge_commands=node.bridge_commands)
        write_durable_milestone("/workspace/evidence/"+cfg["label"]+"_result.json", result)
        node.destroy_node()
        rclpy.shutdown()


def build_motion_script(goal, limits, bodies, caps, label, *, path_start_xy=None):
    from scripts.research.issue46.native_runtime import native_base_micro_probe as micro

    if not label.isidentifier():
        raise ValueError("invalid owned result label")
    code = micro.build_script("forward", .6, limits, command_speed=.20,
                              robot_bodies=bodies).rsplit("\nrun_probe()", 1)[0]
    marker = '    if MICRO.get("carry"):\n        return run_carry_navigation(n,contract)'
    if code.count(marker) != 1:
        raise ValueError("native dispatch changed")
    code = code.replace(marker,
        '    if MICRO.get("task_region"):\n        return run_region_navigation(n,contract)\n'+marker, 1)
    # Explicit task scope also works against the committed pre-visual Micro.
    mode_line = next(l for l in code.splitlines() if l.strip().startswith("if mode not in "))
    mode_part, rest = mode_line.split(" or status not in ", 1)
    previous = mode_part.split("if mode not in ", 1)[1]
    code = code.replace(mode_line, '                if mode not in (("STAND_DEFAULT","LOCOMOTION_STEP") if MICRO.get("task_region") else '+previous+') or status not in '+rest, 1)
    action_line = next(l for l in code.splitlines() if l.strip().startswith('if requested!="STAND_DEFAULT"'))
    code = code.replace(action_line,
        '            if MICRO.get("task_region") and requested not in ("STAND_DEFAULT","LOCOMOTION_STEP"):raise RuntimeError("TASK_ACTION_ALLOWLIST")\n'+
        action_line.replace("if requested!=", 'if not MICRO.get("task_region") and requested!='), 1)
    code = code.replace("                fault=physical_fault(",
        '                if MICRO.get("task_region"):limit=.60\n                fault=physical_fault(', 1)
    for function in (recover_clock_at_stop, body_translation_velocity, translation_corridor_error, run_region_navigation):
        code += "\n"+inspect.getsource(function)+"\n"
    code += "MICRO.update("+repr({"task_region": {
        "goal": list(goal), "effort_caps": caps, "label": label, "maximum_excursion_m": .90, "path_start_xy_m": path_start_xy},
        "graph_startup_timeout_s": 5.0})+")\nrun_probe()\n"
    compile(code, "task_region_navigation", "exec")
    return code


def sampled_navigation_clearance(study, native_row, estimated_center, chosen):
    """Sample a fixed-heading translation tube; actual gait remains a live gate."""
    current = study.frame(native_row)
    current[study.obj:study.obj+7] = np.r_[estimated_center,1.,0.,0.,0.]
    study.apply(current)
    anchor = study.foot_anchor().copy()
    body=native_row["bodies"]["pelvis"]
    yaw=math.atan2(body["rotation"][3],body["rotation"][0])
    target=np.asarray(chosen["anchor_xy_m"])
    count=0;minimum=.05
    offsets=((0.,0.),(.05,0.),(-.05,0.),(0.,.05),(0.,-.05))
    for u in np.linspace(0.,1.,17):
        for offset in offsets:
            for dyaw in (0.,-.20,.20):
                xy=anchor+u*(target-anchor)+np.asarray(offset)
                report=study.screen(study.relocate(current,xy,yaw+dyaw));count+=1
                minimum=min(minimum,report["minimum_external_clearance_m_capped_50mm"])
                if not report["pass"]:
                    return {"pass":False,"first_sample":count,"screen":report,
                            "scope":"SAMPLED_TRANSLATION_TUBE_NOT_DYNAMIC_GAIT"}
    return {"pass":True,"samples":count,"minimum_clearance_m_capped_50mm":minimum,
            "source_root_xy_m":body["position_m"][:2],
            "xy_envelope_m":.05,"heading_envelope_rad":.20,
            "scope":"SAMPLED_TRANSLATION_TUBE_NOT_DYNAMIC_GAIT"}


def torso_heading_for_target(center, root_xy, reference_root_xy, reference_object_xy,
                             reference_heading):
    center = np.asarray(center, dtype=float)
    root_xy = np.asarray(root_xy, dtype=float)
    a = np.asarray(reference_object_xy)-np.asarray(reference_root_xy)
    b = center[:2]-root_xy
    if not np.isfinite(np.r_[a, b, reference_heading]).all() or min(np.linalg.norm(a), np.linalg.norm(b)) < .1:
        raise ValueError("invalid camera/grasp bearing")
    correction = (math.atan2(b[1], b[0])-math.atan2(a[1], a[0])+math.pi)%(2*math.pi)-math.pi
    return float(reference_heading+correction), float(correction)


def configuration_visibility(study, q, model, data, renderer, camera):
    """Predict visibility from unchanged optics and offline planned robot pose."""
    import mujoco
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    scalar = {}
    for j in range(study.model.njnt):
        if study.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(study.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            scalar[name] = {"q": float(q[study.model.jnt_qposadr[j]]), "qvel": 0.}
    synthetic = {"joints": scalar, "bodies": {
        "pelvis": {"position_m": q[study.root:study.root+3].tolist(),
                   "quaternion_wxyz": q[study.root+3:study.root+7].tolist()},
        "test_cylinder": {"position_m": q[study.obj:study.obj+3].tolist(),
                         "quaternion_wxyz": q[study.obj+3:study.obj+7].tolist()}},
        "sim_time_s": 0.}
    rgb, depth, K, T = vision.render(model, data, renderer, camera, synthetic)
    try:
        estimate, _ = vision.estimate(rgb, depth, K, T)
        box = estimate["rgb_component"]["bbox_xyxy"]
        clear = box[0] > 0 and box[1] > 0 and box[2] < vision.WIDTH and box[3] < vision.HEIGHT
        return {"pass": bool(clear), "bbox_xyxy": box,
                "classification": "PREDICTED_CAMERA_VIEW_NOT_LIVE_SENSOR",
                "failure": None if clear else "PREDICTED_OBJECT_CLIPPED"}
    except Exception as exc:
        return {"pass": False, "classification": "PREDICTED_CAMERA_VIEW_NOT_LIVE_SENSOR",
                "failure": str(exc)}


def region_candidates(study, native_row, estimated_center, visibility_check=None,
                      reference_heading=.2449960245840995):
    """Co-plan torso viewing direction; current base before bounded relocation."""
    center = np.asarray(estimated_center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("finite RGBD target required")
    current = study.frame(native_row)
    current[study.obj:study.obj+7] = np.r_[center, 1., 0., 0., 0.]
    study.fixed_object = current[study.obj:study.obj+7].copy()
    study.object_center = center.copy()
    study.axis = np.array([0., 0., 1.])
    study.section = center + .01*study.axis
    study.apply(current)
    anchor = study.foot_anchor().copy()
    body = native_row["bodies"]["pelvis"]
    root_xy = np.asarray(body["position_m"][:2])
    yaw = math.atan2(body["rotation"][3], body["rotation"][0])
    candidates = [(anchor, "CURRENT_STANCE")]
    ref_body = study.frames["standing"]["bodies"]
    ref_root = np.asarray(ref_body["pelvis"]["position_m"][:2])
    ref_object = np.asarray(ref_body["test_cylinder"]["position_m"][:2])
    nominal = ref_root+(center[:2]-ref_object)+(anchor-root_xy)
    candidates.extend((nominal+np.array([0., dy]), "NEARBY_"+str(i)) for i, dy in enumerate((0., -.08, .08)))
    yaw_adr = int(study.qadr[0])
    prepared = study.templates["prepared"].copy()
    preparation = [q.copy() for q in study.preparation]
    rows = []
    try:
        for candidate_anchor, label in candidates:
            if np.linalg.norm(candidate_anchor-anchor) > .80:
                continue
            root_goal = candidate_anchor-(anchor-root_xy)
            heading, correction = torso_heading_for_target(center, root_goal, ref_root, ref_object, reference_heading)
            relative_change = heading-yaw-prepared[yaw_adr]
            if abs(relative_change) > .25:
                rows.append({"label": label, "stage": "OUTSIDE_EXISTING_POSTURE_EXCURSION",
                             "required_waist_change_rad": float(relative_change)})
                continue
            study.templates["prepared"] = prepared.copy()
            study.templates["prepared"][yaw_adr] += relative_change
            study.preparation = []
            for i, original in enumerate(preparation):
                q = original.copy()
                q[yaw_adr] += relative_change*i/max(1, len(preparation)-1)
                study.preparation.append(q)
            witness = study.candidate(candidate_anchor, yaw, "prepared")
            row = {"label": label, "anchor_xy_m": candidate_anchor.tolist(),
                   "root_goal_xy_m": root_goal.tolist(), "yaw_rad": yaw,
                   "torso_world_yaw_target_rad": heading, "object_bearing_change_rad": correction,
                   "stage": witness["stage"],
                   "minimum_clearance_m": witness["minimum_external_clearance_m_capped_50mm"],
                   "first_blocker": witness.get("first_blocker"), "witness": witness}
            rows.append(row)
            if witness["stage"] != "STATIC_WITNESS":
                continue
            if visibility_check is not None:
                final_preparation = next(p["qpos"] for p in reversed(witness["path"]) if p["phase"] == "PREPARATION")
                row["predicted_visibility"] = visibility_check(np.asarray(final_preparation))
                if not row["predicted_visibility"]["pass"]:
                    row["stage"] = "PREPARED_CAMERA_NOT_QUALIFIED"
                    continue
            if label != "CURRENT_STANCE":
                row["navigation_preflight"] = sampled_navigation_clearance(study, native_row, center, row)
                if not row["navigation_preflight"]["pass"]:
                    row["stage"] = "NAVIGATION_PATH_COLLISION"
                    continue
            return row, rows
        return None, rows
    finally:
        study.templates["prepared"] = prepared
        study.preparation = preparation


def execute(run, container, environment, output, scene, cache, reference, limits, bodies):
    import mujoco
    from PIL import Image
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    from scripts.research.issue46.grasp.stance_region_static_review import Study
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract

    if reference != reference_contract():
        raise ValueError("task grasp-region contract changed")
    study = Study()
    live_model = mujoco.MjModel.from_xml_path(str(scene))
    physical_arrays = ("body_mass", "body_inertia", "jnt_range", "jnt_axis",
                       "geom_type", "geom_size", "geom_contype", "geom_conaffinity",
                       "eq_data", "pair_solref", "pair_friction", "actuator_gainprm",
                       "actuator_forcerange")
    if any(not np.array_equal(getattr(live_model, key), getattr(study.model, key))
           for key in physical_arrays):
        raise ValueError("grasp region model does not match frozen physics")
    all_limits, _ = native_leg_safety_contract(include_upper_body=True)
    robot = mujoco.mj_name2id(live_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_bodies = [mujoco.mj_id2name(live_model, mujoco.mjtObj.mjOBJ_BODY, i)
                    for i in range(live_model.nbody) if live_model.body_rootid[i] == robot]
    caps = {mujoco.mj_id2name(live_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
            float(max(abs(live_model.actuator_forcerange[i])))
            for i in range(live_model.nu) if live_model.actuator_forcelimited[i]}
    model, data, renderer, camera = vision.build_sensor(scene, cache)
    events = []
    acquisition_audit = []
    began = time.monotonic()
    try:
        warm = read_fresh_native_row(output/"native_latest.json")
        rgb, depth, K, T = vision.render(model, data, renderer, camera, warm)
        vision.estimate(rgb, depth, K, T)  # discard cold setup, as existing sensor
        after_source_ns = warm["host_monotonic_ns"]
        def capture(native):
            rgb, depth, K, T = vision.render(model, data, renderer, camera, native)
            estimated, _ = vision.estimate(rgb, depth, K, T)
            return rgb, estimated
        for index in range(3):
            observations = []
            for label in ("a", "b"):
                audit = {"index": index, "label": label, "attempts": []}
                acquisition_audit.append(audit)
                native, rgb, obs = acquire_control_observation(
                    lambda: read_fresh_native_row(output/"native_latest.json"), capture,
                    generation=warm["generation"], after_source_ns=after_source_ns,
                    attempts=audit["attempts"])
                after_source_ns = native["host_monotonic_ns"]
                observations.append(obs)
                Image.fromarray(rgb).save(output/f"task_region_{index}_{label}.png")
                time.sleep(.03)
            snapshot = vision.make_stationary_snapshot(*observations, generation=native["generation"])
            chosen, rows = region_candidates(study, native, snapshot["estimated_center_world_m"],
                visibility_check=lambda q: configuration_visibility(study, q, model, data, renderer, camera))
            event = {"index": index, "estimated_object": snapshot, "candidates": rows,
                     "control_object_source": "FRESH_RGBD_ONLY", "chosen": None if chosen is None else chosen["label"]}
            pg.write_new_text(output/f"task_region_{index}.json", json.dumps(event, indent=2)+"\n")
            events.append({k: v for k, v in event.items() if k != "candidates"})
            if chosen is None:
                raise RuntimeError("NO_BOUNDED_COLLISION_CHECKED_GRASP_STANCE")
            if chosen["label"] == "CURRENT_STANCE":
                result = {"pass_result": True, "classification": "STATIC_REGION_SELECTION_NOT_GRASP_PASS",
                          "adjustments": index, "events": events, "elapsed_wall_s": time.monotonic()-began,
                          "fresh_rgbd_reobservation": True, "new_physics_by_host": False}
                import copy
                proposal = copy.deepcopy(json.loads(Path(reference["posture_reference"]).read_text()))
                proposal["torso_world_yaw_target_rad"] = chosen["torso_world_yaw_target_rad"]
                proposal["waist_target_rad"]["waist_yaw_joint"] = chosen["torso_world_yaw_target_rad"]-chosen["yaw_rad"]
                proposal["rgbd_posture_selection"] = {"target_source": "FRESH_RGBD_ONLY",
                    "estimated_center_world_m": snapshot["estimated_center_world_m"],
                    "object_bearing_change_rad": chosen["object_bearing_change_rad"],
                    "model_physics_or_optics_changed": False, "static_region": f"task_region_{index}.json"}
                posture_path = output/"task_rgbd_posture.json"
                pg.write_new_text(posture_path, json.dumps(proposal, indent=2)+"\n")
                result["posture_config"] = str(posture_path)
                pg.write_new_text(output/"visual_stance_result.json", json.dumps(result, indent=2)+"\n")
                return result
            if index == 2:
                raise RuntimeError("TASK_REGION_ADJUSTMENT_BUDGET")
            label = f"task_region_navigation_{index}"
            goal = [*chosen["root_goal_xy_m"], chosen["yaw_rad"]]
            code = build_motion_script(goal, all_limits, robot_bodies, caps, label,
                path_start_xy=chosen["navigation_preflight"]["source_root_xy_m"])
            pg.write_new_text(output/(label+".py"), code)
            process = run(["docker", "exec", container, "bash", "-lc",
                           environment+"timeout 85s python3 /workspace/evidence/"+label+".py"],
                          timeout_s=88., check=False)
            pg.write_new_text(output/(label+".stdout"), process.stdout)
            motion = json.loads((output/(label+"_result.json")).read_text())
            if process.returncode or not motion["pass_result"]:
                raise RuntimeError("TASK_REGION_NAVIGATION_FAILED:"+str(motion.get("failure")))
    except Exception as exc:
        pg.write_new_text(output/"visual_stance_result.json", json.dumps({
            "pass_result": False, "classification": "TASK_REGION_STOPPED_BEFORE_GRASP",
            "failure": str(exc), "events": events}, indent=2)+"\n")
        raise
    finally:
        renderer.close()
        pg.write_new_text(output/"task_rgbd_acquisition.json", json.dumps({
            "classification": "FRESH_RGBD_ACQUISITION_WITH_SLOW_FRAME_REJECTION",
            "maximum_source_age_s": .20, "timestamp_rewriting": False,
            "samples": acquisition_audit}, indent=2)+"\n")
