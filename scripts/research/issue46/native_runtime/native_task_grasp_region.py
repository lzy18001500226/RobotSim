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
FAR_CANDIDATE_SHIFT_M = .90  # Same task search extent as the existing navigation envelope.


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

def task_route_profile(profile, *, fast_task_timing=False):
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
    if fast_task_timing:
        # Existing supported stop sequence: retain active zero commands for
        # the full window before relinquishing our own input source.
        result["release_source_after_zero_window"]=True
        # Loaded CCW forward arcs at the measured south-wall start drifted
        # into the wall-side workspace. Reuse the physically successful
        # clockwise/pure-yaw prefix; actual-pose route screening remains required.
        result["delivery_turn_direction"]="CLOCKWISE"
        result["turn_translation_strategy"]="PURE_YAW"
        # Actual stopped pose with13.5deg nominal heading error passed the
        # original full-model placement IK and collision screen. Keep15cm XY;
        # handoff only after a new physical stop and actual-pose placement IK.
        result["stop_inside_functional_arrival_region"]=True
        result["approach_task_position_tolerance_m"]=.15
        result["approach_task_yaw_tolerance_rad"]=.30
        result["delivery_pose_acceptance"]="ACTUAL_PLACEMENT_IK"
        result["loaded_terminal_slowdown"]=True
        result["conservative_terminal_stop"]=True
        result["bounded_terminal_forward_reapproach"]=True
        # Saved actual near-table stop blocked30mm lowering at rightknee.
        # The smallest checked backoff80mm passed original lowering/placement.
        goal=list(result["base_goal_world_xy"])
        if goal!=[.015,.98]:raise ValueError("unexpected frozen second-table target")
        result["base_goal_world_xy"]=[goal[0],goal[1]-.08]
        # Measured transit stopped at31deg bearing error with positive forward
        # progress; use the existing simultaneous DRIVE steering inside43deg.
        # Larger errors still stop; actual environment/contact/velocity gates stay.
        result["delivery_align_reentry_rad"]=.75

    result["task_turn_selection"] = {
        "classification": "TASK_COLLISION_CHECKED_FORWARD_ARC_NOT_MC_RETUNING",
        "evidence": "reviews/issue-46/x2-table-edge-task-20261111T145311Z/carry_radius333_arc_static.json",
        "live_measured_route_screen_still_required": True,
        "model_gains_limits_friction_unchanged": True}
    if fast_task_timing:
        # Restore the shared0/10 physical baseline: retreat, turn, then walk
        # forward with heading feedback. No reverse delivery or final pivot.
        result.update(rearward_foundation_transit=False,
            delivery_motion_mode="LOCOMOTION_STEP",
            stand_transition_strategy="ZERO_WINDOW_THEN_SD",
            bounded_final_dock_alignment=False,require_forward_delivery=True,
            route_goal_scope="RETREAT_THEN_TURN_THEN_FORWARD_STEERING_TO_ACTUAL_PLACEMENT_REGION")
        result["task_turn_selection"].update(classification="RESTORED_SHARED_0_10_FORWARD_TRANSFER",evidence="reviews/issue-46/shared-defined-approach-all11-transfer-20261113T202351Z/transfer_matrix.json",runtime_validation_required=True)
    return result


def stage_runtime_posture(output, selected_path):
    """Merge only selected pose targets; preserve enriched startup safety data."""
    import copy
    base = json.loads((output/"standing_grasp_proposal.json").read_text())
    selected = json.loads(Path(selected_path).read_text())
    allowed = {"waist_target_rad", "torso_world_yaw_target_rad", "rgbd_posture_selection", "pre_raise_forearm_target_rad"}
    if "grasp_profile" in selected:
        from scripts.research.issue46.native_runtime.grasp_profiles import load_profiles
        if selected["grasp_profile"] not in load_profiles():
            raise ValueError("task posture requires an unchanged registered grasp profile")
        allowed.add("grasp_profile")
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


def reference_contract(*,far_observe=False):
    result = {"schema": "robotsim.rgbd-task-grasp-region.v3",
            "control_object_source": "FRESH_RGBD_ONLY",
            "existing_navigation_controller": "PointGoalController",
            "task_motion_mapping": "MC_BODY_XY_NO_PIVOT_SOURCE_CAP020",
            "maximum_stance_adjustments": 2,
            "posture_reference": str(Path(__file__).resolve().parents[4]/"simulation/mujoco/scenes/raicom2026/config/reviewed_h220_turn_forward_bend_pickup_functional.json"),
            "maximum_candidate_shift_m": .80, "maximum_navigation_excursion_m": .90,
            "translation_tube_xy_m": .05, "translation_tube_heading_rad": .20,
            "task_translation_position_tolerance_m": .06,
            "mechanical_changes": False}
    if far_observe:
        result.update(schema="robotsim.rgbd-task-grasp-region.v14",maximum_candidate_shift_m=FAR_CANDIDATE_SHIFT_M,early_forearm_clearance_recovery=True,compact_forearm_target_rad=-2.25,forward_bend_fallback_rad=.29,short_forward_request_m=.04,short_forward_max_nonzero_s=.20,forward_table_plane_braking=True,task_motion_mapping="FAR_RGBD_FORWARD_TURN_STEP",observation_standoff_m=.94,observation_right_offset_m=.12,rearward_balance_uncertainty_m=.03,maximum_stance_adjustments=6)
    return result


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


def body_translation_velocity(pose, goal_xy, *, gain=.6, speed_cap=.20, minimum_speed=0.):
    """Map the existing point error to MC body-X/Y velocity, without pivoting."""
    values=[pose["x"],pose["y"],pose["yaw"],*goal_xy,gain,speed_cap]
    if not all(math.isfinite(v) for v in values) or gain<=0 or not 0<speed_cap<=.20:
        raise ValueError("invalid bounded MC translation input")
    dx,dy=goal_xy[0]-pose["x"],goal_xy[1]-pose["y"]
    c,s=math.cos(pose["yaw"]),math.sin(pose["yaw"])
    forward,lateral=gain*(c*dx+s*dy),gain*(-s*dx+c*dy)
    scale=min(1.,speed_cap/max(math.hypot(forward,lateral),1e-12))
    forward,lateral=forward*scale,lateral*scale
    norm=math.hypot(forward,lateral)
    if not 0.<=minimum_speed<=speed_cap:raise ValueError("invalid activation speed")
    if 0.<norm<minimum_speed:
        forward,lateral=forward*minimum_speed/norm,lateral*minimum_speed/norm
    return forward,lateral,0.


def translation_corridor_error(pose, start_xy, goal_xy, heading):
    dx,dy=goal_xy[0]-start_xy[0],goal_xy[1]-start_xy[1]
    t=max(0.,min(1.,((pose["x"]-start_xy[0])*dx+(pose["y"]-start_xy[1])*dy)/max(dx*dx+dy*dy,1e-12)))
    distance=math.hypot(pose["x"]-start_xy[0]-t*dx,pose["y"]-start_xy[1]-t*dy)
    yaw_error=abs((pose["yaw"]-heading+math.pi)%(2*math.pi)-math.pi)
    return distance,yaw_error


def locomotion_source_observed(row, requested):
    """Require fresh MC state/readback, never just SetMcAction acknowledgement."""
    return (row.get("common_source",{}).get("name")=="robotsim_official_cmd_vel"
            and row.get("common_mode_desc")==requested and row.get("get_mode_desc")==requested
            and row.get("common_mode_status_value")==100 and row.get("get_mode_status_value")==100)


def locomotion_cycle_proof(trace, since_s, requested):
    """Readback may lag a short command; accept only this same gait cycle."""
    for row in trace:
        if row.get("sample_monotonic_s",-1.)>=since_s and locomotion_source_observed(row,requested):
            return {k:row.get(k) for k in ("sample_monotonic_s","sim_time_s","common_source","common_mode_desc","common_mode_status_value","get_mode_desc","get_mode_status_value")}
    return None


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
              "motion_input_profile": "VENDOR_DEFAULT_LOCOMOTION_BODY_XY_SOURCE_CAP020",
              "requested_mc_action":cfg.get("locomotion_action","LOCOMOTION_STEP"),
              "maximum_command_norm_m_s": .20, "angular_command_rad_s": 0.}
    if cfg.get("point_goal"):
        result["motion_input_profile"]="EXISTING_POINT_GOAL_OUTSIDE_TABLE"
        result["angular_command_rad_s"]="COARSE_LEFT_ALIGN_MAX0.60_RIGHTWARD_MAX_MINUS0.60_OTHERWISE0.40"
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
                if not (str(exc) in ("CLOCK_GENERATION_PENDING", "NATIVE_FEEDBACK_STALE_OR_INVALID","CRITICAL_NEW_CLOCK_GENERATION_TELEMETRY_MISSING")
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
        requested_span=math.hypot(cfg["goal"][0]-origin["x"],cfg["goal"][1]-origin["y"])
        short_forward=bool(cfg.get("point_goal") and requested_span<.06)
        first_active_command_at=None
        controller = PointGoalController(ControllerConfig(
            xy_tolerance_m=.01 if short_forward else .06, yaw_tolerance_rad=.25,
            max_forward_mps=.20, max_angular_rps=.60 if cfg.get("point_goal") else .40, goal_timeout_s=45.0))
        until = time.monotonic() + 45.0
        requested=cfg.get("locomotion_action","LOCOMOTION_STEP")
        mode_cycle_started=time.monotonic()
        node.stand_once(requested)
        node.observe(.2, baseline=baseline)
        # This delivery activates locomotion on valid nonzero input. The
        # running-state proof is post-trigger, never circular zero-only readiness.
        source_acquired=False
        result["mode_activation_protocol"]="ACK_THEN_BOUNDED_TRIGGER_THEN_APPLIED_STATE_PROOF"
        with node.lock:
            node.moving = True
        node.stop_started = False
        watchdog = threading.Timer(max(.01, until-time.monotonic()), node.deadline_zero)
        watchdog.daemon = True
        watchdog.start()
        last_progress = time.monotonic()
        best = math.inf
        last_nav_state = None
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
                node.stand_once(cfg.get("locomotion_action","LOCOMOTION_STEP"))
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
            source_acquired = source_acquired or locomotion_source_observed(row,cfg.get("locomotion_action","LOCOMOTION_STEP"))
            if not source_acquired and time.monotonic()-activation_started>.75:
                raise RuntimeError("TASK_TRANSLATION_SOURCE_AND_MODE_NOT_OBSERVED")
            if output.state not in (NavState.ARRIVED,NavState.SETTLING,NavState.ALIGN,NavState.DRIVE,NavState.FINAL_ALIGN):
                raise RuntimeError("TASK_REGION_CONTROLLER:"+output.state.value+":pose_age="+str(row["controller_pose_age_s"]))
            cross, heading_error = translation_corridor_error(pose,planned_start,cfg["goal"][:2],cfg["goal"][2])
            row["translation_cross_track_m"],row["translation_heading_error_rad"]=cross,heading_error
            bounds=cfg.get("outer_bounds")
            outside=(not (bounds["x"][0]<=pose["x"]<=bounds["x"][1] and bounds["y"][0]<=pose["y"]<=bounds["y"][1] and bounds["yaw"][0]<=pose["yaw"]<=bounds["yaw"][1])) if bounds else cross>.05 or heading_error>.20
            if outside:
                result["task_replan_pause"]={"reason":"LEFT_CHECKED_OUTER_BOX" if bounds else "LEFT_SAMPLED_TRANSLATION_TUBE",
                    "cross_track_m":cross,"heading_error_rad":heading_error}
                break
            if short_forward and first_active_command_at is not None and tick-first_active_command_at>=.20:
                result["short_forward_activation_pulse"]={"requested_span_m":requested_span,"maximum_nonzero_duration_s":.20,"actual_stopped_pose_must_requalify":True}
                break
            brake_pose=navigation_braking_pose(node.trace,pose) if cfg.get("point_goal") else pose
            if cfg.get("point_goal"):row["task_braking_velocity_world_m_s"]=[brake_pose["vx"],brake_pose["vy"]]
            if not short_forward and cfg.get("side_corridor_bounds") and side_corridor_boundary_reached(brake_pose,cfg["side_corridor_bounds"]):
                result["side_corridor_boundary_stop"]={"pose":pose,"bounds":cfg["side_corridor_bounds"],"claim":"PREDICTIVE_BOUNDARY_STOP_NOT_GRASP_READY"};break
            if not short_forward and not cfg.get("side_corridor_bounds") and cfg.get("point_goal") and forward_table_plane_reached(brake_pose,cfg["goal"],[origin["x"],origin["y"]]):
                result["functional_forward_plane_stop"]={"root_x_m":pose["x"],"velocity_x_m_s":pose["vx"],"checked_station_x_m":cfg["goal"][0],"lateral_error_m":cfg["goal"][1]-pose["y"],"claim":"PREDICTIVE_STOP_NOT_POSE_ARRIVAL"};break
            stop_distance=stopping_distance(brake_pose,cfg["goal"]) if cfg.get("point_goal") else controller.config.xy_tolerance_m
            if not short_forward and output.distance_m is not None and output.distance_m<=stop_distance:
                result["position_region_entered"]=True
                result["legacy_heading_error_rad"]=output.yaw_error_rad
                break
            if output.state in (NavState.ARRIVED, NavState.SETTLING):
                node.publish()
                if output.state == NavState.ARRIVED:
                    break
            elif output.state in (NavState.ALIGN, NavState.DRIVE, NavState.FINAL_ALIGN):
                command = (forward_turn_command(pose,cfg["goal"],output.linear_x,output.angular_z) if cfg.get("point_goal") else
                    body_translation_velocity(pose,cfg["goal"][:2],gain=controller.config.k_linear,speed_cap=controller.config.max_forward_mps,minimum_speed=.12 if cfg.get("locomotion_action")=="LOCOMOTION_DEFAULT" else 0.))
                coarse_heading=cfg.get("point_goal") and command[0]==0. and command[2]>0.
                phase="COARSE_HEADING" if coarse_heading else output.state
                row["motion_phase"]="COARSE_HEADING_WITH_MEASURED_DRIFT" if coarse_heading else output.state.value
                if phase != last_nav_state:
                    best, last_progress = math.inf, tick
                    last_nav_state = phase
                metric=abs(wrap(math.atan2(cfg["goal"][1]-pose["y"],cfg["goal"][0]-pose["x"])-pose["yaw"])) if coarse_heading else output.distance_m
                if metric < best-.01:
                    best, last_progress = metric, tick
                if tick-last_progress > 5.0:
                    raise RuntimeError("TASK_REGION_NO_MEASURED_PROGRESS")
                if first_active_command_at is None and any(command):first_active_command_at=time.monotonic()
                node.publish(*command)
            else:
                raise RuntimeError("TASK_REGION_CONTROLLER:"+output.state.value+":pose_age="+str(row["controller_pose_age_s"]))
            time.sleep(max(0.0, tick+.02-time.monotonic()))
        else:
            raise RuntimeError("TASK_REGION_TIMEOUT")
        node.deadline_zero()
        node.stop_started = True
        node.observe(1.05, baseline=baseline)
        cycle_proof=locomotion_cycle_proof(node.trace,mode_cycle_started,requested)
        source_acquired=source_acquired or cycle_proof is not None
        result["applied_mode_cycle_proof"]=cycle_proof
        node.stand_once()
        result["post_stop_composite"] = node.observe(12.0, baseline=baseline, require_stop=True)
        result["end_pose"] = node.sample()["pose"]
        result["source_and_requested_locomotion_observed"] = source_acquired
        result["nonzero_velocity_publish_calls"]=sum(any(c.get(k,0.) for k in ("forward","lateral","angular")) for c in node.command_log)
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


def build_motion_script(goal, limits, bodies, caps, label, *, path_start_xy=None, outer_bounds=None, side_corridor_bounds=None,point_goal=False,locomotion_action="LOCOMOTION_DEFAULT"):
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
    code = code.replace(mode_line, '                if mode not in (("STAND_DEFAULT","LOCOMOTION_STEP","LOCOMOTION_DEFAULT") if MICRO.get("task_region") else '+previous+') or status not in '+rest, 1)
    action_line = next(l for l in code.splitlines() if l.strip().startswith('if requested!="STAND_DEFAULT"'))
    code = code.replace(action_line,
        '            if MICRO.get("task_region") and requested not in ("STAND_DEFAULT","LOCOMOTION_STEP","LOCOMOTION_DEFAULT"):raise RuntimeError("TASK_ACTION_ALLOWLIST")\n'+
        action_line.replace("if requested!=", 'if not MICRO.get("task_region") and requested!='), 1)
    code = code.replace("                fault=physical_fault(",
        '                if MICRO.get("task_region"):limit=.60\n                fault=physical_fault(', 1)
    if point_goal:
        from scripts.research.issue46.native_runtime.far_observe_approach import forward_turn_command,stopping_distance,forward_table_plane_reached,side_corridor_boundary_reached,navigation_braking_pose,wrap
        for function in (wrap,forward_turn_command,stopping_distance,forward_table_plane_reached,side_corridor_boundary_reached,navigation_braking_pose):code+="\n"+inspect.getsource(function)+"\n"
    for function in (recover_clock_at_stop, body_translation_velocity, translation_corridor_error, locomotion_source_observed, locomotion_cycle_proof, run_region_navigation):
        code += "\n"+inspect.getsource(function)+"\n"
    code += "MICRO.update("+repr({"task_region": {
        "goal": list(goal), "effort_caps": caps, "label": label, "maximum_excursion_m": .90, "path_start_xy_m": path_start_xy, "outer_bounds":outer_bounds,"point_goal":point_goal,"locomotion_action":locomotion_action,"side_corridor_bounds":side_corridor_bounds},
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
        estimate, _ = vision.estimate(rgb, depth, K, T, supported_height_m=.22)
        box = estimate["rgb_component"]["bbox_xyxy"]
        clear = box[0] > 0 and box[1] > 0 and box[2] < vision.WIDTH and box[3] < vision.HEIGHT
        return {"pass": bool(clear), "bbox_xyxy": box,
                "classification": "PREDICTED_CAMERA_VIEW_NOT_LIVE_SENSOR",
                "failure": None if clear else "PREDICTED_OBJECT_CLIPPED"}
    except Exception as exc:
        return {"pass": False, "classification": "PREDICTED_CAMERA_VIEW_NOT_LIVE_SENSOR",
                "failure": str(exc)}


def forearm_clearance_recovery_needed(rows):
    """Only right-hand travel/preparation interference justifies compacting it."""
    for candidate in rows:
        violations=[]
        if candidate.get("stage")=="NAVIGATION_PATH_COLLISION":
            violations=candidate.get("navigation_preflight",{}).get("screen",{}).get("violations",[])
        elif candidate.get("stage")=="ACTUAL_PREPARATION_COLLISION":
            pre=candidate.get("actual_preparation_preflight",{})
            if pre.get("stage") in ("elbow","waist"):violations=[{"bodies":pre.get("pair",[])}]
        for violation in violations:
            names=set(violation.get("bodies",[]))
            if any(n.startswith("R_hand_") for n in names) and names.intersection({"raicom_table_table","test_cylinder"}):return True
    return False


def region_candidates(study, native_row, estimated_center, visibility_check=None,
                      reference_heading=.2449960245840995,far_observe=False,pre_raise_target_rad=-2.15):
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
    # Prefer a checked forward reach before depending on unqualified side-step response.
    candidates.extend((anchor+np.array([dx, 0.]), "FORWARD_ONLY_"+str(i))
                      for i, dx in enumerate((.08, .14, .20)))
    ref_body = study.frames["standing"]["bodies"]
    ref_root = np.asarray(ref_body["pelvis"]["position_m"][:2])
    ref_object = np.asarray(ref_body["test_cylinder"]["position_m"][:2])
    nominal = ref_root+(center[:2]-ref_object)+(anchor-root_xy)
    nearby=[(nominal+np.array([0.,dy]), "NEARBY_"+str(i)) for i,dy in enumerate((0.,-.08,.08,-.16,-.24))]
    candidates.extend(sorted(nearby,key=lambda item:np.linalg.norm(item[0]-anchor)))
    if far_observe:
        from scripts.research.issue46.native_runtime.far_observe_approach import stance_proposals
        candidates=stance_proposals(anchor,root_xy,yaw,center)
    else:
        candidates=[(a,label,yaw) for a,label in candidates]
    yaw_adr = int(study.qadr[0])
    prepared = study.templates["prepared"].copy()
    preparation = [q.copy() for q in study.preparation]
    if pre_raise_target_rad not in (-2.15,-2.25):raise ValueError("unsupported source-valid pregrasp pose")
    if far_observe and pre_raise_target_rad==-2.25:
        from scripts.research.issue46.grasp.stance_region_static_review import ACTIVE
        prepared[int(study.qadr[ACTIVE.index("right_elbow_joint")])]=pre_raise_target_rad
    if far_observe and abs(native_row["joints"]["right_elbow_joint"]["q"]-pre_raise_target_rad)<.025:
        from scripts.research.issue46.grasp.stance_region_static_review import ACTIVE
        elbow_adr=int(study.qadr[ACTIVE.index("right_elbow_joint")])
        # The earlier elbow action is already physically complete; retain all
        # remaining lower/waist collision checks without replaying that action.
        for q in preparation:q[elbow_adr]=prepared[elbow_adr]
    candidates=[(a,label,heading,None) for a,label,heading in candidates]
    if far_observe:
        # Bounded fallbacks only after the existing candidates fail.
        # Prefer an already source-valid current posture before any base move.
        # Saved left arrival passed this original pitch candidate, while a later
        # side-transit was obstructed. All original witness gates still apply.
        candidates.insert(1,(anchor,"CURRENT_STANCE_FORWARD_BEND_029",yaw,.29))
        candidates.extend([(anchor+np.array([.04,0.]),"FORWARD_SHORT_04_FORWARD_BEND_029",yaw,.29),
            (anchor+np.array([.04,0.]),"FORWARD_SHORT_04",yaw,None)])
    profile_by_label={}
    if far_observe:
        from scripts.research.issue46.native_runtime.grasp_profiles import select_profile,torso_world_heading
        profile=select_profile(center)
        label=profile['profile'];actual_label=label+"_ACTUAL_STANCE"
        profile_by_label[label]=profile;profile_by_label[actual_label]=profile
        # A reference is a posture seed, not a mandatory millimetric station.
        # Always test the actual stopped base before requesting another motion.
        # Only the complete IK/path/reach/visibility gates may accept it.
        preferred=profile.get('preferred_approach_label')
        if preferred:
            candidates.sort(key=lambda candidate:0 if candidate[1]==preferred else 1)
        from scripts.research.issue46.native_runtime.grasp_profiles import standing_stance_from_profile
        standing_label=label+"_STANDING_REFERENCE"
        profile_by_label[standing_label]=profile
        standing_anchor,standing_yaw=standing_stance_from_profile(profile,center,root_xy,anchor,yaw)
        # Retain the historical commanded approach targets ahead of the
        # measured-stop fallback. A stopped grasp witness is not a navigation
        # lookahead: using it first understeered the saved left-end approach.
        candidates.append((standing_anchor,standing_label,standing_yaw,profile['waist_pitch_rad']))
        candidates.insert(0,(anchor.copy(),actual_label,yaw,profile['waist_pitch_rad']))
        # Keep the demonstrated forward/turn navigation targets below. A saved
        # *achieved* stopped grasp pose is not the old commanded motion goal:
        # braking/coast made them differ by hundreds of millimetres in Y.
    rows = []
    try:
        for candidate_anchor, label,candidate_yaw,pitch_target in candidates:
            yaw=candidate_yaw
            if np.linalg.norm(candidate_anchor-anchor) > (FAR_CANDIDATE_SHIFT_M if far_observe else .80):
                continue
            from scipy.spatial.transform import Rotation
            oldyaw=math.atan2(body["rotation"][3],body["rotation"][0])
            root_goal=candidate_anchor+Rotation.from_euler("z",yaw-oldyaw).as_matrix()[:2,:2]@(root_xy-anchor)
            heading, correction = torso_heading_for_target(center, root_goal, ref_root, ref_object, reference_heading)
            if label in profile_by_label:
                heading=torso_world_heading(profile_by_label[label])
            relative_change = heading-yaw-prepared[yaw_adr]
            # Task search extent only; Study still enforces source bounds and path collisions.
            if abs(relative_change) > 1.05:
                rows.append({"label": label, "stage": "OUTSIDE_EXISTING_POSTURE_EXCURSION",
                             "required_waist_change_rad": float(relative_change)})
                continue
            study.templates["prepared"] = prepared.copy()
            study.templates["prepared"][yaw_adr] += relative_change
            pitch_adr=int(study.qadr[1])
            if pitch_target is not None:study.templates["prepared"][pitch_adr]=pitch_target
            study.preparation = []
            for i, original in enumerate(preparation):
                q = original.copy()
                q[yaw_adr] += relative_change*i/max(1, len(preparation)-1)
                if pitch_target is not None:
                    ratio=float(np.clip(original[pitch_adr]/prepared[pitch_adr],0.,1.))
                    q[pitch_adr]+=(pitch_target-prepared[pitch_adr])*ratio
                study.preparation.append(q)
            witness = study.candidate(candidate_anchor, yaw, "prepared",runtime_arm_only=True,source_arm_range=far_observe,arm_target_inset_rad=.03 if far_observe else 0.,root_xy=root_goal if far_observe else None,root_reference_label="standing" if far_observe else None)
            if not far_observe and witness["stage"] == "IK_UNRESOLVED":
                # A pre-raised elbow is not a physical reach limit. Reuse all
                # original arm ranges before forcing an unnecessary base move.
                original=witness
                witness=study.candidate(candidate_anchor,yaw,"prepared",source_arm_range=True,runtime_arm_only=True)
                witness["planning_neighborhood_fallback"]={"original_stage":original["stage"],
                    "source_arm_ranges_unchanged":True,"physical_tracking_limits_unchanged":True}
            row = {"label": label, "anchor_xy_m": candidate_anchor.tolist(),
                   "root_goal_xy_m": root_goal.tolist(), "yaw_rad": yaw,
                   "torso_world_yaw_target_rad": heading, "object_bearing_change_rad": correction,
                   "stage": witness["stage"],
                   "minimum_clearance_m": witness["minimum_external_clearance_m_capped_50mm"],
                   "first_blocker": witness.get("first_blocker"), "witness": witness}
            row["base_motion_required"]=not label.startswith("CURRENT_STANCE")
            if label in profile_by_label:
                row['grasp_profile']=profile_by_label[label]
                # Arrival is a region: qualify this actual stance and the same
                # profile posture before asking for another small base move.
                if np.linalg.norm(root_goal-root_xy)<=.08 and abs(math.atan2(math.sin(yaw-oldyaw),math.cos(yaw-oldyaw)))<=.20:
                    row['base_motion_required']=False
            row["pre_raise_forearm_target_rad"]=pre_raise_target_rad
            if pitch_target is not None:row["waist_pitch_target_rad"]=pitch_target
            rows.append(row)
            if witness["stage"] != "STATIC_WITNESS":
                continue
            if far_observe:
                # Real arm-extension runs displaced the pelvis rearward up to
                #21.26mm. Require a30mm rearward reach witness before crouching.
                balance=study.candidate(candidate_anchor-np.array([.03,0.]),yaw,"prepared",source_arm_range=True,runtime_arm_only=True,arm_target_inset_rad=.03,root_xy=root_goal-np.array([.03,0.]),root_reference_label="standing")
                row["rearward_balance_preflight"]={"uncertainty_m":.03,"stage":balance["stage"],"pass":balance["stage"]=="STATIC_WITNESS","witness":balance}
                if balance["stage"]!="STATIC_WITNESS":
                    row["stage"]="DYNAMIC_BALANCE_REACH_MARGIN_NOT_QUALIFIED"
                    continue
                from scripts.research.issue46.native_runtime.far_observe_approach import preparation_clearance
                row["actual_preparation_preflight"]=preparation_clearance(study,native_row,center,row)
                if not row["actual_preparation_preflight"]["pass"]:
                    row["stage"]="ACTUAL_PREPARATION_COLLISION"
                    continue
            if visibility_check is not None:
                final_preparation = next(p["qpos"] for p in reversed(witness["path"]) if p["phase"] == "PREPARATION")
                row["predicted_visibility"] = visibility_check(np.asarray(final_preparation))
                if not row["predicted_visibility"]["pass"]:
                    row["stage"] = "PREPARED_CAMERA_NOT_QUALIFIED"
                    continue
            if row["base_motion_required"]:
                if far_observe:
                    from scripts.research.issue46.native_runtime.far_observe_approach import sampled_approach_clearance
                    row["navigation_preflight"]=sampled_approach_clearance(study,native_row,center,row)
                else:
                    row["navigation_preflight"] = sampled_navigation_clearance(study, native_row, center, row)
                if not row["navigation_preflight"]["pass"]:
                    row["stage"] = "NAVIGATION_PATH_COLLISION"
                    continue
                if far_observe:
                    from scripts.research.issue46.native_runtime.far_observe_approach import side_corridor_waypoint,sampled_side_corridor_clearance
                    waypoint=side_corridor_waypoint(root_xy,root_goal)
                    if waypoint is not None:
                        row["side_corridor_preflight"]=sampled_side_corridor_clearance(study,native_row,center,waypoint)
                        if not row["side_corridor_preflight"]["pass"]:
                            row["stage"]="NAVIGATION_SIDE_CORRIDOR_COLLISION"
                            continue
            return row, rows
        return None, rows
    finally:
        study.templates["prepared"] = prepared
        study.preparation = preparation


def outer_corridor_preflight(study, native, center, bounds):
    """Full robot static pose envelope; actual gait retains existing live gates."""
    q=study.frame(native);q[study.obj:study.obj+7]=np.r_[center,1.,0.,0.,0.]
    study.apply(q);offset=study.foot_anchor()-np.asarray(native["bodies"]["pelvis"]["position_m"][:2])
    count=0;minimum=.05
    grids=[np.linspace(*bounds[k],max(2,int(math.ceil((bounds[k][1]-bounds[k][0])/(.06 if k!="yaw" else .15)))+1)) for k in ("x","y","yaw")]
    for x in grids[0]:
        for y in grids[1]:
            for yaw in grids[2]:
                report=study.screen(study.relocate(q,np.array([x,y])+offset,float(yaw)));count+=1
                minimum=min(minimum,report["minimum_external_clearance_m_capped_50mm"])
                if not report["pass"]:return dict(pass_result=False,samples=count,bounds=bounds,first_blocker=report)
    return dict(pass_result=True,samples=count,bounds=bounds,minimum_clearance_m_capped_50mm=minimum,scope="STATIC_ENVELOPE_NOT_DYNAMIC_GAIT_PASS",mj_step_calls=0)


def relocate_via_outer_corridor(run,container,environment,output,study,center,chosen,limits,bodies,caps,index):
    """Use existing forward/back velocity and point-goal turning outside the table."""
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    records=[]
    for ordinal in range(2):
        native=read_fresh_native_row(output/"native_latest.json");body=native["bodies"]["pelvis"];xy=np.array(body["position_m"][:2]);R=body["rotation"];yaw=math.atan2(R[3],R[0])
        goal=([max(.48,float(xy[0]-.27)),float(xy[1]),yaw] if ordinal==0 else [*chosen["root_goal_xy_m"],yaw])
        if ordinal==0:
            bounds={"x":[goal[0]-.05,float(xy[0]+.05)],"y":[float(xy[1]-.10),float(xy[1]+.10)],"yaw":[yaw-.20,yaw+.20]}
        else:
            heading=math.atan2(goal[1]-xy[1],goal[0]-xy[0]);bounds={"x":[float(min(xy[0],goal[0])-.06),float(max(xy[0],goal[0])+.05)],"y":[float(min(xy[1],goal[1])-.06),float(max(xy[1],goal[1])+.06)],"yaw":[min(yaw,heading)-.35,max(yaw,heading)+.35]}
        label=f"task_outer_{index}_{ordinal}";proof=outer_corridor_preflight(study,native,center,bounds)
        pg.write_new_text(output/(label+"_preflight.json"),json.dumps(proof,indent=2)+"\n")
        if not proof["pass_result"]:raise RuntimeError("OUTER_CORRIDOR_STATIC_COLLISION:"+label)
        if ordinal==1 and xy[0]>.78:raise RuntimeError("OUTER_TURN_STANDOFF_NOT_REACHED")
        code=build_motion_script(goal,limits,bodies,caps,label,path_start_xy=xy.tolist(),outer_bounds=bounds,point_goal=ordinal==1)
        pg.write_new_text(output/(label+".py"),code)
        process=run(["docker","exec",container,"bash","-lc",environment+"timeout 85s python3 /workspace/evidence/"+label+".py"],timeout_s=88.,check=False)
        pg.write_new_text(output/(label+".stdout"),process.stdout)
        motion=json.loads((output/(label+"_result.json")).read_text());records.append({"label":label,"goal":goal,"motion":{k:v for k,v in motion.items() if k not in ("trace","commands","bridge_commands","events")}})
        if process.returncode or not motion["pass_result"]:raise RuntimeError("OUTER_CORRIDOR_MOTION_FAILED:"+str(motion.get("failure")))
    return records


def execute(run, container, environment, output, scene, cache, reference, limits, bodies):
    import mujoco
    from PIL import Image
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    from scripts.research.issue46.grasp.stance_region_static_review import Study
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract

    far_observe=reference.get("task_motion_mapping")=="FAR_RGBD_FORWARD_TURN_STEP"
    if reference != reference_contract(far_observe=far_observe):
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
        vision.estimate(rgb, depth, K, T, supported_height_m=.22)  # discard cold setup, as existing sensor
        after_source_ns = warm["host_monotonic_ns"]
        def capture(native):
            rgb, depth, K, T = vision.render(model, data, renderer, camera, native)
            try:
                estimated, _ = vision.estimate(rgb, depth, K, T, supported_height_m=.22)
            except ValueError:
                # Preserve the exact failed sensor input before safe shutdown.
                Image.fromarray(rgb).save(output/"failed_control_rgb.png")
                np.savez_compressed(output/"failed_control_rgbd.npz",rgb=rgb,depth=depth,
                                    K=json.dumps(K),T=T)
                raise
            return rgb, estimated
        adjustment_budget=reference["maximum_stance_adjustments"]
        for index in range(adjustment_budget+1):
            observations = []
            captured_images=[]
            for label in ("a", "b"):
                audit = {"index": index, "label": label, "attempts": []}
                acquisition_audit.append(audit)
                native, rgb, obs = acquire_control_observation(
                    lambda: read_fresh_native_row(output/"native_latest.json"), capture,
                    generation=warm["generation"], after_source_ns=after_source_ns,
                    attempts=audit["attempts"])
                after_source_ns = native["host_monotonic_ns"]
                observations.append(obs)
                captured_images.append((label,rgb))
            # Qualify/freeze acquired observations before diagnostic PNG I/O.
            # No timestamp changes: only valid fresh frames can create a snapshot.
            snapshot = vision.make_stationary_snapshot(*observations, generation=native["generation"])
            for label,picture in captured_images:
                Image.fromarray(picture).save(output/f"task_region_{index}_{label}.png")
            early_target=-2.15
            if (output/"pre_raise_forearm_result.json").exists():
                prior=json.loads((output/"pre_raise_forearm_result.json").read_text())
                if not prior.get("pass_result") or prior["request"].get("generation")!=native["generation"] or prior["request"].get("process_pid")!=native["process_pid"]:raise RuntimeError("TASK_EARLY_ARM_NOT_SAME_EPISODE")
                early_target=prior["request"]["target_rad"]
            chosen, rows = region_candidates(study, native, snapshot["estimated_center_world_m"],
                visibility_check=lambda q: configuration_visibility(study, q, model, data, renderer, camera),far_observe=far_observe,pre_raise_target_rad=early_target)
            event = {"index": index, "estimated_object": snapshot, "candidates": rows,
                     "control_object_source": "FRESH_RGBD_ONLY", "chosen": None if chosen is None else chosen["label"]}
            pg.write_new_text(output/f"task_region_{index}.json", json.dumps(event, indent=2)+"\n")
            events.append({k: v for k, v in event.items() if k != "candidates"})
            if chosen is None:
                if far_observe and index<adjustment_budget and not (output/"pre_raise_forearm_result.json").exists() and forearm_clearance_recovery_needed(rows):
                    import xml.etree.ElementTree as ET
                    from scripts.research.issue46.native_runtime.standing_waist_preposition import pre_raise_forearm
                    _,source=native_leg_safety_contract(include_upper_body=True)
                    joint=next(j for j in ET.parse(source["urdf"]).getroot().findall("joint") if j.get("name")=="right_elbow_joint")
                    proposal=json.loads((output/"standing_grasp_proposal.json").read_text())
                    proposal["pre_raise_forearm_target_rad"]=-2.25
                    recovery=pre_raise_forearm(output,proposal,all_limits,float(joint.find("limit").get("effort")),elbow_only=True)
                    pg.write_new_text(output/"task_early_forearm_clearance.json",json.dumps({"reason":"HANGING_HAND_BLOCKED_TRAVEL","source_static_screen":f"task_region_{index}.json","actual_source_valid_arm_motion":recovery,"model_or_masks_changed":False},indent=2)+"\n")
                    events[-1]["recovery_action"]="SOURCE_VALID_ELBOW_PRE_RAISE_BEFORE_UNLOADED_TRAVEL"
                    continue
                raise RuntimeError("NO_BOUNDED_COLLISION_CHECKED_GRASP_STANCE")
            if not chosen.get("base_motion_required",chosen["label"]!="CURRENT_STANCE"):
                result = {"pass_result": True, "classification": "STATIC_REGION_SELECTION_NOT_GRASP_PASS",
                          "adjustments": index, "events": events, "elapsed_wall_s": time.monotonic()-began,
                          "fresh_rgbd_reobservation": True, "new_physics_by_host": False}
                import copy
                proposal = copy.deepcopy(json.loads(Path(reference["posture_reference"]).read_text()))
                proposal["torso_world_yaw_target_rad"] = chosen["torso_world_yaw_target_rad"]
                proposal["waist_target_rad"]["waist_yaw_joint"] = chosen["torso_world_yaw_target_rad"]-chosen["yaw_rad"]
                if "waist_pitch_target_rad" in chosen:proposal["waist_target_rad"]["waist_pitch_joint"]=chosen["waist_pitch_target_rad"]
                proposal["pre_raise_forearm_target_rad"]=chosen.get("pre_raise_forearm_target_rad",proposal["pre_raise_forearm_target_rad"])
                if 'grasp_profile' in chosen:proposal['grasp_profile']=chosen['grasp_profile']
                proposal["rgbd_posture_selection"] = {"target_source": "FRESH_RGBD_ONLY",
                    "estimated_center_world_m": snapshot["estimated_center_world_m"],
                    "object_bearing_change_rad": chosen["object_bearing_change_rad"],
                    "model_physics_or_optics_changed": False, "static_region": f"task_region_{index}.json"}
                posture_path = output/"task_rgbd_posture.json"
                pg.write_new_text(posture_path, json.dumps(proposal, indent=2)+"\n")
                result["posture_config"] = str(posture_path)
                pg.write_new_text(output/"visual_stance_result.json", json.dumps(result, indent=2)+"\n")
                return result
            side_waypoint=None
            side_preflight=None
            if far_observe:
                from scripts.research.issue46.native_runtime.far_observe_approach import side_corridor_waypoint,side_corridor_bounds,sampled_side_corridor_clearance
                current_root=np.asarray(native["bodies"]["pelvis"]["position_m"][:2]);terminal=np.asarray(chosen["root_goal_xy_m"])
                side_waypoint=side_corridor_waypoint(current_root,terminal)
                if side_waypoint is not None:
                    side_preflight=sampled_side_corridor_clearance(study,native,snapshot["estimated_center_world_m"],side_waypoint)
                    if not side_preflight["pass"]:raise RuntimeError("SIDE_CORRIDOR_STATIC_COLLISION:"+json.dumps(side_preflight.get("screen")))
            if index == adjustment_budget:
                raise RuntimeError("TASK_REGION_ADJUSTMENT_BUDGET")
            if not far_observe and chosen["label"].startswith("NEARBY_") and abs(chosen["root_goal_xy_m"][1]-native["bodies"]["pelvis"]["position_m"][1])>.10:
                event["outer_relocation"]=relocate_via_outer_corridor(run,container,environment,output,study,snapshot["estimated_center_world_m"],chosen,all_limits,robot_bodies,caps,index)
                continue
            label = f"task_region_navigation_{index}"
            motion_target=chosen
            if side_waypoint is not None:
                study.apply(study.frame(native));anchor=study.foot_anchor().copy();rotation=native["bodies"]["pelvis"]["rotation"];heading=math.atan2(rotation[3],rotation[0])
                motion_target={"label":"SIDE_CORRIDOR_TRANSIT_NOT_GRASP_READY","root_goal_xy_m":side_waypoint.tolist(),"anchor_xy_m":(anchor+side_waypoint-np.asarray(native["bodies"]["pelvis"]["position_m"][:2])).tolist(),"yaw_rad":heading,"side_corridor_bounds":side_corridor_bounds(np.asarray(native["bodies"]["pelvis"]["position_m"][:2]),side_waypoint),"side_corridor_preflight":side_preflight}
                origin=np.asarray(native["bodies"]["pelvis"]["position_m"][:2]);travel=heading+(math.atan2(side_waypoint[1]-origin[1],side_waypoint[0]-origin[0])-heading+math.pi)%(2*math.pi)-math.pi
                motion_target["navigation_preflight"]={**side_preflight,"outer_bounds":{"x":[float(origin[0]-.16),float(origin[0]+.16)],"y":[float(min(origin[1],side_waypoint[1])-.16),float(max(origin[1],side_waypoint[1])+.16)],"yaw":[float(min(heading,travel)-.35),float(max(heading,travel)+.35)]}}
            elif far_observe:
                from scripts.research.issue46.native_runtime.far_observe_approach import intermediate_forward_waypoint,sampled_approach_clearance
                origin=np.asarray(native["bodies"]["pelvis"]["position_m"][:2])
                transit=intermediate_forward_waypoint(origin,chosen["root_goal_xy_m"])
                if transit is not None:
                    study.apply(study.frame(native));anchor=study.foot_anchor().copy()
                    rotation=native["bodies"]["pelvis"]["rotation"]
                    heading=math.atan2(rotation[3],rotation[0])
                    motion_target={"label":"INTERMEDIATE_FORWARD_TRANSIT_NOT_GRASP_READY",
                        "root_goal_xy_m":transit.tolist(),"anchor_xy_m":(anchor+transit-origin).tolist(),"yaw_rad":heading}
                    motion_target["navigation_preflight"]=sampled_approach_clearance(study,native,snapshot["estimated_center_world_m"],motion_target)
                    if not motion_target["navigation_preflight"]["pass"]:raise RuntimeError("INTERMEDIATE_TRANSIT_PATH_COLLISION")
            pg.write_new_text(output/(label+"_plan.json"),json.dumps({"terminal_grasp_witness":f"task_region_{index}.json",
                "terminal_label":chosen["label"],"terminal_root_goal_xy_m":chosen["root_goal_xy_m"],
                "motion_label":motion_target["label"],"motion_root_goal_xy_m":motion_target["root_goal_xy_m"],
                "transit_is_grasp_ready":False,"side_corridor_preflight":motion_target.get("side_corridor_preflight"),"navigation_preflight":motion_target.get("navigation_preflight")},indent=2)+"\n")
            goal = [*motion_target["root_goal_xy_m"], motion_target["yaw_rad"]]
            code = build_motion_script(goal, all_limits, robot_bodies, caps, label,
                path_start_xy=motion_target["navigation_preflight"]["source_root_xy_m"] if motion_target.get("navigation_preflight") else np.asarray(native["bodies"]["pelvis"]["position_m"][:2]).tolist(),
                outer_bounds=motion_target["navigation_preflight"].get("outer_bounds") if motion_target.get("navigation_preflight") else None,side_corridor_bounds=motion_target.get("side_corridor_bounds"),point_goal=far_observe,
                locomotion_action="LOCOMOTION_STEP" if far_observe else "LOCOMOTION_DEFAULT")
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
