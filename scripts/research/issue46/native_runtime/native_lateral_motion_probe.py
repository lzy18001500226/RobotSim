"""Bounded real-MC lateral probe after actual START docking; no grasp or model edits."""
from __future__ import annotations
import inspect,json,math,time
from pathlib import Path
from scripts.agent import path_governance as pg


def phases(direction,profile="reference"):
    if direction not in ("left","right"):raise ValueError("left/right direction required")
    side=.12 if direction=="left" else -.12
    if profile=="side-start":
        sign=1 if direction=="left" else -1
        return [("SIDE_START_PULSE",.25,(0.,sign*.50,0.)),("SIDE_MAINTAIN",1.,(0.,side,0.))]
    if profile!="reference":raise ValueError("unknown diagnostic profile")
    return [("PURE_SIDE",.60,(0.,side,0.)),("COMBINED_START",2.00,(.20,0.,.40 if side>0 else -.40)),("SIDE_MAINTAIN",1.00,(0.,side,0.))]


def side_start_bridge_command(command):
    """Override only the named diagnostic's existing gateway side-input cap."""
    import re
    pattern=r"max_lateral_velocity:=0\.20?(?=\s|$)"
    matches=list(re.finditer(pattern,command))
    if len(matches)!=1:raise ValueError("SIDE_START_BRIDGE_PROFILE_NOT_IDENTIFIED")
    return re.sub(pattern,"max_lateral_velocity:=0.50",command)


def validate_velocity(velocity,profile):
    x,y,w=velocity;cap=.50 if profile=="side-start" else .20
    if not all(math.isfinite(v) for v in velocity) or math.hypot(x,y)>cap+1e-9 or abs(w)>.400000001:raise RuntimeError("LATERAL_PROFILE_COMMAND_BOUND")
    if profile=="side-start" and (x or w or abs(y)>.500000001):raise RuntimeError("SIDE_START_PROFILE_REQUIRES_PURE_SIDE")


def run_lateral_probe(node,contract):
    import threading
    cfg=MICRO["lateral_diagnostic"]
    result={"classification":"REAL_FROM_START_NATIVE_MC_LATERAL_DIAGNOSTIC","direction":cfg["direction"],
            "safe_stop_pass":False,"behavior_pass":False,"failure":None,"phases":[],
            "requested_mode":"LOCOMOTION_DEFAULT","profile":cfg.get("profile","reference"),"selected_translation_input_cap_m_s":.50 if cfg.get("profile")=="side-start" else .20,"table_normal_bounds_m":cfg["bounds"]["x"],
            "gains_model_source_limits_unchanged":True,"new_grasp_or_lift":"NOT_RUN"}
    watchdog=None;baseline=None
    try:
        node.start_feedback();until=time.monotonic()+5.
        while time.monotonic()<until:
            node.publish();node.pump()
            try:row=node.sample();break
            except RuntimeError as exc:
                if not (str(exc) in ("CLOCK_GENERATION_PENDING","NATIVE_FEEDBACK_STALE_OR_INVALID") or str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:")):raise
            time.sleep(.01)
        else:raise RuntimeError("LATERAL_FEEDBACK_READINESS")
        node.await_graph();baseline=row["body_status"]
        result["pre_motion_stop"]=node.observe(8.,baseline=baseline,require_stop=True)
        row=node.sample();origin=row["pose"];result["start_pose"]=origin
        if cfg.get("profile")=="side-start":
            bounds=row.get("raw_common",{}).get("speed_status",{}).get("lateral_bounds",{})
            if bounds.get("min_value",0.)>-.50 or bounds.get("max_value",0.)<.50:raise RuntimeError("SDK_LATERAL_BOUNDS_DO_NOT_ALLOW_PROFILE")
            result["observed_sdk_lateral_bounds"]=bounds
        generation=node.latest_row["generation"]
        node.stand_once("LOCOMOTION_DEFAULT");node.observe(.2,baseline=baseline)
        with node.lock:node.moving=True
        node.stop_started=False;begin=time.monotonic();watchdog=threading.Timer(10.,node.deadline_zero);watchdog.start()
        source_seen=False;mode_seen=False;finish_early=False
        for label,duration,velocity in cfg["phases"]:
            phase_start=node.sample();sim_start=phase_start["sim_time_s"];t0=time.monotonic()
            phase={"name":label,"requested_body_velocity":velocity,"start_pose":phase_start["pose"],"start_sim_s":sim_start,"source_host_start_ns":node.latest_row["host_monotonic_ns"]}
            result["phases"].append(phase)
            while True:
                node.pump();row=node.sample();pose=row["pose"]
                if node.latest_row["generation"]!=generation:raise RuntimeError("LATERAL_GENERATION_CHANGED")
                if not node.moving:raise RuntimeError("LATERAL_WATCHDOG_STOPPED")
                bounds=cfg["bounds"]
                if not (bounds["x"][0]<=pose["x"]<=bounds["x"][1] and bounds["y"][0]<=pose["y"]<=bounds["y"][1] and bounds["yaw"][0]<=pose["yaw"]<=bounds["yaw"][1]):
                    result["early_stop"]="LEFT_CHECKED_FREE_SPACE_ENVELOPE";finish_early=True;break
                source=row.get("common_source",{}).get("name")
                source_seen=source_seen or source=="robotsim_official_cmd_vel"
                mode_seen=mode_seen or row.get("get_mode_desc")=="LOCOMOTION_DEFAULT" and row.get("common_mode_desc")=="LOCOMOTION_DEFAULT"
                if not source_seen and time.monotonic()-begin>1.:raise RuntimeError("LATERAL_SOURCE_NOT_SELECTED")
                row["lateral_probe_phase"]=label;node.trace.append(row)
                signed=(pose["y"]-origin["y"])*(1 if cfg["direction"]=="left" else -1)
                if signed>=.10:result["early_stop"]="USEFUL_SIDE_REGION_REACHED";finish_early=True;break
                if label=="SIDE_START_PULSE" and row["sim_time_s"]-sim_start>.30:raise RuntimeError("SIDE_START_PULSE_DURATION_BOUND")
                if row["sim_time_s"]-sim_start>=duration:break
                if time.monotonic()-begin>9.:raise RuntimeError("LATERAL_WALL_BUDGET")
                validate_velocity(velocity,cfg.get("profile","reference"));node.publish(*velocity);time.sleep(.02)
            phase.update(end_pose=pose,end_sim_s=row["sim_time_s"],source_host_end_ns=node.latest_row["host_monotonic_ns"],wall_s=time.monotonic()-t0)
            if finish_early:break
        node.deadline_zero();node.stop_started=True
        node.observe(1.05,baseline=baseline);node.stand_once()
        result["post_stop_composite"]=node.observe(12.,baseline=baseline,require_stop=True)
        end=node.sample()["pose"];result["end_pose"]=end
        dx,dy=end["x"]-origin["x"],end["y"]-origin["y"];c,s=math.cos(origin["yaw"]),math.sin(origin["yaw"])
        result["measured"]={"table_normal_m":dx,"table_parallel_left_m":dy,"body_forward_m":c*dx+s*dy,"body_left_m":-s*dx+c*dy,"yaw_change_rad":(end["yaw"]-origin["yaw"]+math.pi)%(2*math.pi)-math.pi}
        signed=dy*(1 if cfg["direction"]=="left" else -1)
        result["safe_stop_pass"]=result["post_stop_composite"]["pass"]
        result["behavior_pass"]=bool(source_seen and mode_seen and signed>=.02 and abs(dx)<=.10 and result["safe_stop_pass"])
        result.update(selected_source_observed=source_seen,requested_mode_observed=mode_seen,pass_result=result["behavior_pass"])
    except Exception as exc:
        node.deadline_zero();node.stop_started=True;result["failure"]=str(exc)
        try:
            node.observe(1.05,baseline=baseline);node.stand_once();result["abort_stop"]=node.observe(12.,baseline=baseline,require_stop=True)
        except Exception as stop:result["abort_stop_failure"]=str(stop)
    finally:
        node.deadline_zero()
        if watchdog:watchdog.cancel()
        node.stop_feedback()
        result.update(trace=node.trace,commands=node.command_log,bridge_commands=node.bridge_commands,events=node.events)
        write_durable_milestone("/workspace/evidence/lateral_motion_probe_result.json",result)
        node.destroy_node();rclpy.shutdown()


def build_script(direction,limits,bodies,caps,bounds,profile="reference"):
    from scripts.research.issue46.native_runtime.native_task_grasp_region import build_motion_script
    code=build_motion_script([0.,0.,0.],limits,bodies,caps,"lateral_motion_probe")
    code=code.replace('("STAND_DEFAULT","LOCOMOTION_STEP")','("STAND_DEFAULT","LOCOMOTION_STEP","LOCOMOTION_DEFAULT")')
    old="return run_region_navigation(n,contract)"
    if code.count(old)!=1:raise ValueError("existing native dispatcher changed")
    code=code.replace(old,"return run_lateral_probe(n,contract)",1)
    # The installed native transport and all physical gates are reused.
    code=code.rsplit("\nrun_probe()",1)[0]+"\n"+inspect.getsource(validate_velocity)+"\n"+inspect.getsource(run_lateral_probe)+"\nMICRO['lateral_diagnostic']="+repr({"direction":direction,"profile":profile,"phases":phases(direction,profile),"bounds":bounds})+"\nrun_probe()\n"
    compile(code,"native_lateral_motion_probe","exec");return code


def simulator_leg_receipt(text):
    """Pinned delivered simulator Handle log, not subscriber creation or MC ACK."""
    return any("leg_joint_command_subscriber.cc" in line and "@Handle]" in line and "cmd size: 12" in line for line in text.splitlines())


def wait_simulator_leg_receipt(output,timeout_s=4.):
    log=output/"mujoco-console.log";begin=time.monotonic();deadline=begin+timeout_s
    while time.monotonic()<deadline:
        text=log.read_text(errors="replace")
        if simulator_leg_receipt(text):
            proof=dict(pass_result=True,evidence="SIMULATOR_OWN_LEG_COMMAND_HANDLE_12_OBSERVED",wait_wall_s=time.monotonic()-begin,source_log=str(log),not_a_mode_ack=True)
            pg.write_new_text(output/"initial_sdk_leg_receipt.json",json.dumps(proof,indent=2)+"\n");return proof
        time.sleep(.05)
    proof=dict(pass_result=False,evidence="NO_SIMULATOR_LEG_COMMAND_HANDLE_OBSERVED",wait_wall_s=time.monotonic()-begin,source_log=str(log))
    pg.write_new_text(output/"initial_sdk_leg_receipt.json",json.dumps(proof,indent=2)+"\n")
    raise RuntimeError("SIMULATOR_LEG_COMMAND_RECEIVER_NOT_READY")


def native_standing_snapshot(row):
    """Reuse existing posture/support thresholds on actual physics, not mode ACK."""
    from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose
    pose=native_pose(row);feet={}
    for side in ("left","right"):
        feet[side]=sum(max(0.,float(c["local_force_torque"][0])) for c in row.get("contacts",[]) if c.get("efc_address",-1)>=0 and side+"_ankle_roll_link" in (c["body1"],c["body2"]) and "world" in (c["body1"],c["body2"]))
    reasons=[]
    if pose["z"]<.45:reasons.append("HEIGHT")
    if math.hypot(pose["roll"],pose["pitch"])>math.radians(15):reasons.append("TILT")
    if min(feet.values())<5.:reasons.append("BILATERAL_SUPPORT")
    return dict(ready=not reasons,reasons=reasons,height_m=pose["z"],feet_normal_n=feet,source_host_ns=row["host_monotonic_ns"],generation=row["generation"],sim_time_s=row["sim_time_s"])


def confirm_physical_standing(output):
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    until=time.monotonic()+3.;first=None;rows=[]
    while time.monotonic()<until:
        row=read_fresh_native_row(output/"native_latest.json");value=native_standing_snapshot(row);rows.append(value)
        if not value["ready"]:
            pg.write_new_text(output/"initial_native_standing_check.json",json.dumps(dict(pass_result=False,samples=rows),indent=2)+"\n")
            raise RuntimeError("LATERAL_INITIAL_PHYSICAL_STAND_FAILED:"+str(value["reasons"]))
        if first is None:first=value
        if value["generation"]!=first["generation"]:raise RuntimeError("INITIAL_STAND_GENERATION_CHANGED")
        if value["sim_time_s"]-first["sim_time_s"]>=1.:
            result=dict(pass_result=True,samples=rows,scope="REAL_PHYSICS_HEIGHT_TILT_AND_BILATERAL_SUPPORT",thresholds_changed=False)
            pg.write_new_text(output/"initial_native_standing_check.json",json.dumps(result,indent=2)+"\n");return result
        time.sleep(.02)
    raise RuntimeError("INITIAL_PHYSICAL_STANDING_WINDOW_INCOMPLETE")


def execute(run,container,environment,output,direction,scene,profile="reference"):
    import mujoco,numpy as np
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract
    from scripts.research.issue46.grasp.stance_region_static_review import Study
    native=read_fresh_native_row(output/"native_latest.json");body=native["bodies"]["pelvis"];xy=body["position_m"][:2];R=body["rotation"];yaw=math.atan2(R[3],R[0])
    bounds={"x":[xy[0]-.10,xy[0]+.10],"y":[xy[1]-.25,xy[1]+.25],"yaw":[yaw-.40,yaw+.40]}
    study=Study();model=mujoco.MjModel.from_xml_path(str(scene))
    for key in ("body_mass","body_inertia","jnt_range","geom_type","geom_size","geom_contype","geom_conaffinity","eq_data","actuator_gainprm","actuator_forcerange"):
        if not np.array_equal(getattr(model,key),getattr(study.model,key)):raise RuntimeError("LATERAL_MODEL_IDENTITY_CHANGED:"+key)
    center=native["bodies"]["test_cylinder"]["position_m"]
    q=study.frame(native);q[study.obj:study.obj+7]=np.r_[center,1.,0.,0.,0.];study.apply(q)
    offset=study.foot_anchor()-np.asarray(xy);count=0;minimum=.05;violation=None
    for x in np.linspace(*bounds["x"],5):
        for y in np.linspace(*bounds["y"],7):
            for angle in np.linspace(*bounds["yaw"],5):
                screen=study.screen(study.relocate(q,np.array([x,y])+offset,float(angle)));count+=1
                minimum=min(minimum,screen["minimum_external_clearance_m_capped_50mm"])
                if not screen["pass"] and violation is None:violation=screen
    proof=dict(pass_result=violation is None,bounds=bounds,samples=count,first_blocker=violation,minimum_clearance_m_capped_50mm=minimum,mj_step_calls=0)
    # Object truth is only collision/safety evaluation, never a motion target.
    proof.update(object_pose_use="SAFETY_ONLY_NOT_CONTROL_TARGET",arrival_pose=body,table_normal_reference_x_m=1.3213823326277394)
    pg.write_new_text(output/"lateral_motion_preflight.json",json.dumps(proof,indent=2)+"\n")
    if not proof["pass_result"]:raise RuntimeError("LATERAL_START_CLEARANCE_PREFLIGHT")
    limits,_=native_leg_safety_contract(include_upper_body=True);root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis");bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody) if model.body_rootid[i]==root]
    caps={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i):float(max(abs(model.actuator_forcerange[i]))) for i in range(model.nu) if model.actuator_forcelimited[i]}
    code=build_script(direction,limits,bodies,caps,bounds,profile);pg.write_new_text(output/"lateral_motion_probe.py",code)
    p=run(["docker","exec",container,"bash","-lc",environment+"timeout 45s python3 /workspace/evidence/lateral_motion_probe.py"],timeout_s=48.,check=False)
    pg.write_new_text(output/"lateral_motion_probe.stdout",p.stdout)
    path=output/"lateral_motion_probe_result.json"
    if not path.exists():raise RuntimeError("LATERAL_PROBE_NO_RESULT:"+p.stdout[-500:])
    result=json.loads(path.read_text());result["child_returncode"]=p.returncode
    return result
