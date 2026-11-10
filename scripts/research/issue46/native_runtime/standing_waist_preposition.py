"""Pre-position yaw/pitch through the existing exclusive MC-waist relay."""
from pathlib import Path
import inspect,json,math,time,os
WAIST=("waist_yaw_joint","waist_pitch_joint","waist_roll_joint")

def load_proposal(path, limits):
    value=json.loads(Path(path).read_text())
    if value.get("schema")!="robotsim.standing-grasp-posture.v1" or value.get("roll_mode")!="RETAIN_MEASURED":raise ValueError("posture proposal schema/roll policy")
    targets=value["waist_target_rad"]
    if set(targets)!={"waist_yaw_joint","waist_pitch_joint"}:raise ValueError("only torso turn and forward bend may be proposed")
    for name,x in targets.items():
        lo,hi=limits[name][:2]
        if not math.isfinite(x) or not lo+.015<=x<=hi-.015:raise ValueError("posture source margin:"+name)
    if value.get("body_drop_m")!=.10 or value.get("grasp_height_above_center_m")!=.01 or value.get("grasp_height_from_bottom_m")!=.12:raise ValueError("only qualified H220 modest-crouch task")
    return value

def position_mode_setpoint(desired, actual, velocity, bias, kp, kd, limits, effort_cap):
    lo,hi,speed=limits
    values=(desired,actual,velocity,bias,kp,kd,lo,hi,speed,effort_cap)
    if not all(math.isfinite(x) for x in values) or kp<=0. or kd<=0.:raise RuntimeError("WAIST_POSITION_MODE_INVALID")
    if not lo+.001<=desired<=hi-.001 or abs(velocity)>speed:raise RuntimeError("WAIST_POSITION_MODE_SOURCE_LIMIT")
    command=desired+bias/kp
    torque=kp*(desired-actual)-kd*velocity+bias
    if not lo+.001<=command<=hi-.001 or abs(torque)>effort_cap or abs(bias)>effort_cap:raise RuntimeError("WAIST_POSITION_MODE_EFFORT_OR_SETPOINT_LIMIT")
    return command

def override_waist(self, message):
    path="/workspace/evidence/standing_waist_request.json"
    if getattr(self,"waist_request",None) is None:
        if not os.path.exists(path):return message
        self.waist_request=json.load(open(path));self.waist_native=None;self.waist_read_at=0.;self.waist_report_at=0.
    request=self.waist_request;now=time.monotonic()
    if request.get("schema")!="robotsim.waist-preposition.v1" or request["duration_s"]!=4. or not 0.<=(time.monotonic_ns()-request["created_host_ns"])*1e-9<=60.:raise RuntimeError("WAIST_POSTURE_REQUEST_EXPIRED_OR_INVALID")
    if now-self.waist_read_at>=.02:
        incoming=json.load(open(path))
        if incoming["created_host_ns"]>request["created_host_ns"]:
            if incoming["generation"]!=request["generation"]:raise RuntimeError("WAIST_POSTURE_REQUEST_GENERATION_CHANGED")
            self.waist_request=incoming;request=incoming
        self.waist_native=json.load(open("/workspace/evidence/native_latest.json"));self.waist_read_at=now
    row=self.waist_native
    if row["generation"]!=request["generation"] or not 0.<=(time.monotonic_ns()-row["host_monotonic_ns"])*1e-9<=.20:raise RuntimeError("WAIST_POSTURE_NATIVE_STATE_NOT_FRESH")
    u=min(1.,max(0.,(time.monotonic_ns()-request["created_host_ns"])*1e-9/request["duration_s"]));u=u*u*(3.-2.*u)
    actual={};published={};source_gains={j.name:{"stiffness":float(j.stiffness),"damping":float(j.damping)} for j in message.joints}
    mc_source_stamp_ns=int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec)
    for joint in message.joints:
        name=joint.name;lo,hi,speed=request["source_limits"][name]
        target=float(request["target_rad"][name]);start=float(request["start_rad"][name])
        if not lo+.001<=target<=hi-.001 or joint.stiffness<=0. or joint.damping<=0.:raise RuntimeError("WAIST_POSTURE_SOURCE_CONTRACT:"+name)
        bias=float(row["joints"][name]["bias_force"])
        if not math.isfinite(bias) or abs(bias)>request["effort_caps_nm"][name]:raise RuntimeError("WAIST_POSTURE_GRAVITY_EFFORT:"+name)
        desired=start+u*(target-start)
        native=row["joints"][name]
        joint.position=position_mode_setpoint(desired,float(native["q"]),float(native["qvel"]),bias,float(joint.stiffness),float(joint.damping),(lo,hi,speed),request["effort_caps_nm"][name]);joint.velocity=0.;joint.effort=0.
        actual[name]=float(row["joints"][name]["q"]);published[name]=float(joint.position)
    message.header.stamp=self.get_clock().now().to_msg()
    if now-self.waist_report_at>=.10:
        info=dict(owner_pid=os.getpid(),generation=row["generation"],source_host_ns=row["host_monotonic_ns"],host_monotonic_ns=time.monotonic_ns(),actual_rad=actual,published_rad=published,roll_policy="RETAIN_MEASURED",gains="UNCHANGED_MC",gravity_term="ACTUAL_NATIVE_QFRC_BIAS_DIVIDED_BY_UNCHANGED_KP_POSITION_MODE",fraction=u,request_created_host_ns=request["created_host_ns"],unchanged_mc_source_gains=source_gains,mc_source_stamp_ns=mc_source_stamp_ns)
        with open("/workspace/evidence/standing_waist_feedback.tmp","w") as f:json.dump(info,f)
        os.replace("/workspace/evidence/standing_waist_feedback.tmp","/workspace/evidence/standing_waist_feedback.json");self.waist_report_at=now
    return message

def extend_bootstrap(script):
    if script.count(" def on_waist(self,m):")!=1:raise ValueError("MC waist relay method changed")
    script=script.replace("class Hold(Node):",inspect.getsource(position_mode_setpoint)+"\n"+inspect.getsource(override_waist)+"\nclass Hold(Node):",1)
    old="  self.waist_pub.publish(m);self.waist_last_receipt=time.monotonic();self.waist_relay_count+=1"
    if script.count(old)!=1:raise ValueError("single waist relay publication changed")
    script=script.replace(old,"  m=override_waist(self,m)\n"+old,1)
    compile(script,"existing_waist_relay_preposition","exec");return script

def read_native(path):
    from scripts.research.issue46.native_runtime.native_grasp_audit import require_fresh_sample
    checked=require_fresh_sample(path)
    row=json.loads(Path(path).read_text())
    if (row["generation"],row["process_pid"],row["engine_version"])!=(checked["generation"],checked["process_pid"],337) or not 0.<=(time.monotonic_ns()-row["host_monotonic_ns"])*1e-9<=.20:raise RuntimeError("WAIST_NATIVE_READ_NOT_FRESH")
    return row

def request_and_wait(output, proposal, limits, effort_caps, *, retain_current=False):
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import require_fresh_sample
    from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose,physical_fault
    row=read_native(output/"native_latest.json");target={n:row["joints"][n]["q"] for n in WAIST} if retain_current else {**proposal["waist_target_rad"],"waist_roll_joint":row["joints"]["waist_roll_joint"]["q"]}
    request=dict(schema="robotsim.waist-preposition.v1",generation=row["generation"],created_host_ns=time.monotonic_ns(),duration_s=4.,start_rad={n:row["joints"][n]["q"] for n in WAIST},target_rad=target,source_limits={n:limits[n] for n in WAIST},effort_caps_nm=effort_caps)
    pending=output/"standing_waist_request.tmp"
    pg.write_new_text(pending,json.dumps(request,indent=2)+"\n")
    os.replace(pending,output/"standing_waist_request.json")
    deadline=time.monotonic()+10.
    while time.monotonic()<deadline:
        actual=read_native(output/"native_latest.json")
        fault=physical_fault(actual,native_pose(actual),limits,robot_bodies=proposal["robot_bodies"],review_incidental_contacts=True)
        if fault:raise RuntimeError("WAIST_PREPOSITION_PHYSICAL:"+fault)
        feet=[sum(max(0.,c["local_force_torque"][0]) for c in actual["contacts"] if c.get("efc_address",-1)>=0 and side+"_ankle_roll_link" in (c["body1"],c["body2"]) and "world" in (c["body1"],c["body2"])) for side in ("left","right")]
        if min(feet)<5.:raise RuntimeError("WAIST_PREPOSITION_SUPPORT_LOST")
        u=min(1.,max(0.,(time.monotonic_ns()-request["created_host_ns"])*1e-9/4.));u=u*u*(3.-2.*u)
        if any(abs(actual["joints"][n]["q"]-(request["start_rad"][n]+u*(target[n]-request["start_rad"][n])))>.05 for n in WAIST):raise RuntimeError("WAIST_PREPOSITION_TRACKING_ERROR")
        feedback=output/"standing_waist_feedback.json"
        if feedback.exists():
            f=json.loads(feedback.read_text())
            if f.get("request_created_host_ns")==request["created_host_ns"] and f["generation"]==row["generation"] and 0.<=(time.monotonic_ns()-f["source_host_ns"])*1e-9<=.20 and f["fraction"]>=1. and max(abs(actual["joints"][n]["q"]-target[n]) for n in WAIST)<.025 and max(abs(actual["joints"][n]["qvel"]) for n in WAIST)<.05:
                return dict(request=request,feedback=f,actual_source_host_ns=actual["host_monotonic_ns"],position_error_rad={n:actual["joints"][n]["q"]-target[n] for n in WAIST})
        time.sleep(.02)
    raise RuntimeError("WAIST_PREPOSITION_NOT_REACHED")

def execute_preposition(run,container,environment,output,proposal_path):
    import xml.etree.ElementTree as ET
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract
    proposal=json.loads((output/"standing_grasp_proposal.json").read_text())
    import yaml
    config=Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml")
    native=yaml.safe_load(config.read_text())["actual"]
    waist=native["active_joint_frame_type"].index("waist")
    if native["active_joint_name"][waist]!=list(WAIST) or native["active_joint_control_type"][waist]!=["position"]*3:raise RuntimeError("INSTALLED_WAIST_NOT_VERIFIED_POSITION_MODE")
    limits,provenance=native_leg_safety_contract(include_upper_body=True)
    joints={j.attrib["name"]:j for j in ET.parse(provenance["urdf"]).getroot().findall("joint")}
    caps={n:float(joints[n].find("limit").attrib["effort"]) for n in WAIST}
    initial_waist_hold=request_and_wait(output,proposal,limits,caps,retain_current=True)
    pg.write_new_text(output/"standing_waist_initial_hold.json",json.dumps(initial_waist_hold,indent=2)+"\n")
    for label in ("body_crouch_pickup",):
        child=run(["docker","exec",container,"bash","-lc",environment+"timeout 35s python3 /workspace/evidence/"+label+".py"],timeout_s=38.,check=False)
        pg.write_new_text(output/(label+".stdout"),child.stdout)
        path=output/(label+"_result.json")
        result=json.loads(path.read_text()) if path.exists() else {}
        if child.returncode or not result.get("pass_result"):raise RuntimeError("BODY_CROUCH_PICKUP_NOT_QUALIFIED:"+str(result.get("failure",child.stdout[-1000:])))
    hold=request_and_wait(output,proposal,limits,caps)
    pg.write_new_text(output/"standing_waist_result.json",json.dumps(hold,indent=2)+"\n")
    child=run(["docker","exec",container,"bash","-lc",environment+"timeout 15s python3 /workspace/evidence/body_posture_hold.py"],timeout_s=18.,check=False)
    pg.write_new_text(output/"body_posture_hold.stdout",child.stdout)
    record=output/"body_posture_hold_result.json"
    observed=json.loads(record.read_text()) if record.exists() else {}
    if child.returncode or not observed.get("pass_result"):raise RuntimeError("TURN_FORWARD_BEND_COMPOSITE_HOLD_NOT_QUALIFIED:"+str(observed.get("failure",child.stdout[-1000:])))
    return dict(classification="SIMULATION_ONLY_NATIVE_CROUCH_TORSO_TURN_FORWARD_BEND",roll_policy="RETAIN_MEASURED_NO_ACTIVE_SIDE_BEND",
                initial_waist_hold=initial_waist_hold,body_crouch=result,waist=hold,posture_stop=observed,proposal_path=str(proposal_path))
