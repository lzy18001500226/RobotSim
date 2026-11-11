"""Pre-position yaw/pitch through the existing exclusive MC-waist relay."""
from pathlib import Path
import inspect,json,math,time,os
WAIST=("waist_yaw_joint","waist_pitch_joint","waist_roll_joint")

def waist_position_ready(actual, target, roll_tolerance=.025):
    if not math.isfinite(roll_tolerance) or not .025<=roll_tolerance<=.04:raise ValueError('bounded task roll readiness tolerance')
    tolerance={n:(roll_tolerance if n=='waist_roll_joint' else .025) for n in WAIST}
    return all(math.isfinite(actual[n]) and abs(actual[n]-target[n])<tolerance[n] for n in WAIST)

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

def clearance_posture_error(row, local_center, pelvis_relative_target):
    hand=row["bodies"]["R_omnipicker_base_link"];base=row["bodies"]["pelvis"]
    H=hand["rotation"];B=base["rotation"]
    actual=[hand["position_m"][i]+sum(H[3*i+j]*local_center[j] for j in range(3)) for i in range(3)]
    expected=[base["position_m"][i]+sum(B[3*i+j]*pelvis_relative_target[j] for j in range(3)) for i in range(3)]
    return actual,expected,math.dist(actual,expected)

def position_mode_setpoint(desired, actual, velocity, bias, kp, kd, limits, effort_cap, *, bound_effort=False):
    lo,hi,speed=limits
    values=(desired,actual,velocity,bias,kp,kd,lo,hi,speed,effort_cap)
    if not all(math.isfinite(x) for x in values) or kp<=0. or kd<=0.:raise RuntimeError("WAIST_POSITION_MODE_INVALID")
    if not lo+.001<=desired<=hi-.001 or abs(velocity)>speed:raise RuntimeError("WAIST_POSITION_MODE_SOURCE_LIMIT")
    command=desired+bias/kp
    torque=kp*(desired-actual)-kd*velocity+bias
    if bound_effort:
        reserve=1e-6
        lower=max(lo+.001,actual+(-effort_cap+reserve+kd*velocity)/kp)
        upper=min(hi-.001,actual+(effort_cap-reserve+kd*velocity)/kp)
        if lower>upper:raise RuntimeError('NO_SOURCE_VALID_BOUNDED_POSITION_COMMAND')
        command=min(upper,max(lower,command))
        torque=kp*(command-actual)-kd*velocity
    if not lo+.001<=command<=hi-.001 or abs(torque)>effort_cap or (not bound_effort and abs(bias)>effort_cap):raise RuntimeError("WAIST_POSITION_MODE_EFFORT_OR_SETPOINT_LIMIT")
    return command

def bounded_logging_overlay(config):
    """Reduce only Info logging; retain warnings/errors and all control fields."""
    import copy
    value=copy.deepcopy(config);log=value['aimrt']['log']
    if log.get('core_lvl')!='Info' or log.get('default_module_lvl')!='Info':raise ValueError('unexpected delivered MC log profile')
    log['default_module_lvl']='Warn'
    return value


def refresh_bootstrap_waist_io(self):
    """Main-loop I/O only; the MC subscription callback uses validated RAM state."""
    import os,time,json
    now=time.monotonic()
    if now-getattr(self,'waist_input_read_at',0.)<.02:return
    path='/workspace/evidence/standing_waist_request.json'
    if os.path.exists(path):
        incoming=json.load(open(path));current=getattr(self,'waist_request',None)
        if current is not None and incoming['generation']!=current['generation']:raise RuntimeError('WAIST_POSTURE_REQUEST_GENERATION_CHANGED')
        row=json.load(open('/workspace/evidence/native_latest.json'))
        self.waist_native=row
        if current is None:
            import copy
            template=getattr(self,'waist_source_template',None)
            if template is None or time.monotonic()-self.waist_last_receipt>.20:return
            if any(j.stiffness<=0. or j.damping<=0. for j in template.joints):raise RuntimeError('MANUAL_WAIST_REQUIRES_FRESH_ACTIVE_SOURCE_GAIN_CAPTURE')
            self.waist_manual_template=copy.deepcopy(template);self.waist_report_at=0.
        self.waist_request=incoming
    self.waist_input_read_at=now
    feedback=getattr(self,'waist_pending_feedback',None)
    if feedback is not None and feedback['host_monotonic_ns']>getattr(self,'waist_saved_feedback_ns',0):
        with open('/workspace/evidence/standing_waist_feedback.tmp','w') as f:json.dump(feedback,f)
        os.replace('/workspace/evidence/standing_waist_feedback.tmp','/workspace/evidence/standing_waist_feedback.json')
        self.waist_saved_feedback_ns=feedback['host_monotonic_ns']


def bootstrap_unowned_waist_ready(self):
    """Before manipulation, pause publication on late MC input; never use it."""
    import time,json
    if getattr(self,'waist_request',None) is not None:return True
    last=getattr(self,'waist_last_receipt',None)
    if last is None:return True
    age=time.monotonic()-last
    if age<=.20:
        if getattr(self,'waist_relay_paused',False):print(json.dumps({'waist_relay_reacquired':True,'accepted_age_s':age,'new_motion_targets_during_pause':0}),flush=True)
        self.waist_relay_paused=False
        return True
    if getattr(self,'forearm_request',None) is not None or age>1.:raise RuntimeError('STALE_MC_WAIST_RELAY')
    if not getattr(self,'waist_relay_paused',False):print(json.dumps({'waist_relay_paused':True,'input_age_s':age,'new_motion_targets_during_pause':0,'silence_deadline_s':last+1.}),flush=True)
    self.waist_relay_paused=True
    return False


def emit_manual_waist(self):
    """The current owner drives manual position control on its own cycle."""
    import copy
    if getattr(self,'waist_request',None) is None:return
    message=override_waist(self,copy.deepcopy(self.waist_manual_template))
    self.waist_pub.publish(message)


def override_waist(self, message):
    if getattr(self,"waist_request",None) is None:return message
    request=self.waist_request;now=time.monotonic()
    if request.get("schema")!="robotsim.waist-preposition.v1" or request["duration_s"]!=4. or not 0.<=(time.monotonic_ns()-request["created_host_ns"])*1e-9<=float(request.get("lease_s",60.)):raise RuntimeError("WAIST_POSTURE_REQUEST_EXPIRED_OR_INVALID")
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
        info=dict(owner_pid=os.getpid(),generation=row["generation"],source_host_ns=row["host_monotonic_ns"],host_monotonic_ns=time.monotonic_ns(),actual_rad=actual,published_rad=published,controller_mode="MANUAL_SOURCE_RANGE_POSITION_CONTROL",roll_policy="RETAIN_MEASURED",gains="UNCHANGED_MC_CAPTURED_CONFIGURATION",gravity_term="ACTUAL_NATIVE_QFRC_BIAS_DIVIDED_BY_UNCHANGED_KP_POSITION_MODE",fraction=u,request_created_host_ns=request["created_host_ns"],unchanged_mc_source_gains=source_gains,mc_source_stamp_ns=mc_source_stamp_ns)
        self.waist_pending_feedback=info;self.waist_report_at=now
    return message

def override_forearm(self,target,name):
    if name!='right_elbow_joint':return target
    path='/workspace/evidence/pre_raise_forearm_request.json'
    if getattr(self,'forearm_request',None) is None:
        if not os.path.exists(path):return target
        self.forearm_request=json.load(open(path));self.forearm_read_at=0.;self.forearm_report_at=0.
    req=self.forearm_request;now=time.monotonic()
    if req.get('schema')!='robotsim.forearm-pre-raise.v1' or req.get('duration_s')!=3. or not 0.<=(time.monotonic_ns()-req['created_host_ns'])*1e-9<=300.:raise RuntimeError('PRE_RAISE_REQUEST_INVALID')
    if now-self.forearm_read_at>=.02:
        self.forearm_row=json.load(open('/workspace/evidence/native_latest.json'));self.forearm_read_at=now
    row=self.forearm_row
    if row['generation']!=req['generation'] or not 0.<=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('PRE_RAISE_FEEDBACK_STALE')
    u=min(1.,max(0.,(time.monotonic_ns()-req['created_host_ns'])*1e-9/3.));u=u*u*(3.-2.*u)
    desired=req['start_rad']+u*(req['target_rad']-req['start_rad']);j=row['joints'][name];kp,kd=GAINS[name]
    command=position_mode_setpoint(desired,j['q'],j['qvel'],j['bias_force'],kp,kd,req['source_limits'],req['effort_cap_nm'])
    if now-self.forearm_report_at>=.10:
        value=dict(owner_pid=os.getpid(),generation=row['generation'],source_host_ns=row['host_monotonic_ns'],fraction=u,actual_rad=j['q'],desired_rad=desired,published_rad=command,only_joint_command_changed=name)
        with open('/workspace/evidence/pre_raise_forearm_feedback.tmp','w') as f:json.dump(value,f)
        os.replace('/workspace/evidence/pre_raise_forearm_feedback.tmp','/workspace/evidence/pre_raise_forearm_feedback.json');self.forearm_report_at=now
    return command

def extend_bootstrap(script):
    if script.count(" def on_waist(self,m):")!=1:raise ValueError("MC waist relay method changed")
    script=script.replace("class Hold(Node):",inspect.getsource(position_mode_setpoint)+"\n"+inspect.getsource(override_waist)+"\n"+inspect.getsource(override_forearm)+"\n"+inspect.getsource(refresh_bootstrap_waist_io)+"\n"+inspect.getsource(bootstrap_unowned_waist_ready)+"\n"+inspect.getsource(emit_manual_waist)+"\nclass Hold(Node):",1)
    old="  self.waist_pub.publish(m);self.waist_last_receipt=time.monotonic();self.waist_relay_count+=1"
    if script.count(old)!=1:raise ValueError("single waist relay publication changed")
    script=script.replace(old,"  self.waist_source_template=m;self.waist_last_receipt=time.monotonic();self.waist_relay_count+=1\n  if getattr(self,'waist_request',None) is None:self.waist_pub.publish(m)",1)
    stale="  if self.waist_last_receipt is not None and time.monotonic()-self.waist_last_receipt>.20:raise RuntimeError('STALE_MC_WAIST_RELAY')\n"
    if script.count(stale)!=1:raise ValueError('waist source-phase watchdog anchor')
    script=script.replace(stale,"",1)
    anchor="   x=JointCommand();x.name=name;x.position=target;"
    if script.count(anchor)!=1:raise ValueError("forearm command owner anchor")
    script=script.replace(anchor,"   target=override_forearm(self,target,name)\n"+anchor,1)
    effort="x.effort=float(INITIAL_ARM_EFFORT.get(name,0.))"
    if script.count(effort)!=1:raise ValueError("forearm effort owner anchor")
    script=script.replace(effort,"x.effort=0. if name==\'right_elbow_joint\' and getattr(self,\'forearm_request\',None) is not None else float(INITIAL_ARM_EFFORT.get(name,0.))",1)
    # Reuse the native micro observer's dedicated callback executor pattern.
    # The control loop must not let DDS callbacks queue behind its file I/O.
    startup="rclpy.init();n=Hold()"
    replacement="""rclpy.init();n=Hold()
from rclpy.executors import SingleThreadedExecutor
import threading
feedback_executor=SingleThreadedExecutor();feedback_executor.add_node(n)
feedback_errors=[]
def spin_feedback():
 try:feedback_executor.spin()
 except Exception as exc:feedback_errors.append(repr(exc))
feedback_thread=threading.Thread(target=spin_feedback,name='bootstrap-mc-feedback',daemon=True)
feedback_thread.start()"""
    if script.count(startup)!=1:raise ValueError('bootstrap executor startup anchor')
    script=script.replace(startup,replacement,1)
    old_tick="  n.send();rclpy.spin_once(n,timeout_sec=.001)"
    new_tick="  if feedback_errors or not feedback_thread.is_alive():raise RuntimeError('BOOTSTRAP_FEEDBACK_EXECUTOR_FAILED:'+str(feedback_errors))\n  refresh_bootstrap_waist_io(n)\n  if not bootstrap_unowned_waist_ready(n):time.sleep(.005);continue\n  emit_manual_waist(n)\n  n.send()"
    if script.count(old_tick)!=1:raise ValueError('bootstrap executor tick anchor')
    script=script.replace(old_tick,new_tick,1)
    idle="  while rclpy.ok() and time.monotonic()<next_tick:rclpy.spin_once(n,timeout_sec=.001)"
    if script.count(idle)!=1:raise ValueError('bootstrap executor idle anchor')
    script=script.replace(idle,"  time.sleep(max(0.,next_tick-time.monotonic()))",1)
    old_end="finally:n.destroy_node();rclpy.shutdown()"
    new_end="finally:\n feedback_executor.shutdown(timeout_sec=1.);feedback_thread.join(timeout=1.)\n if feedback_thread.is_alive():raise RuntimeError('BOOTSTRAP_FEEDBACK_SHUTDOWN_UNCONFIRMED')\n n.destroy_node();rclpy.shutdown()"
    if script.count(old_end)!=1:raise ValueError('bootstrap executor cleanup anchor')
    script=script.replace(old_end,new_end,1)
    compile(script,"existing_waist_relay_preposition","exec");return script

from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row as read_native

def request_and_wait(output, proposal, limits, effort_caps, *, retain_current=False):
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import require_fresh_sample
    from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose,physical_fault
    row=read_native(output/"native_latest.json");target={n:row["joints"][n]["q"] for n in WAIST} if retain_current else {**proposal["waist_target_rad"],"waist_roll_joint":row["joints"]["waist_roll_joint"]["q"]}
    if not retain_current and proposal.get('torso_world_yaw_target_rad') is not None:
        R=row['bodies']['pelvis']['rotation'];root_yaw=math.atan2(R[3],R[0]);desired=float(proposal['torso_world_yaw_target_rad'])
        if not math.isfinite(desired):raise ValueError('torso world task heading')
        target['waist_yaw_joint']=(desired-root_yaw+math.pi)%(2*math.pi)-math.pi
        for name,value in target.items():
            if not limits[name][0]+.015<=value<=limits[name][1]-.015:raise RuntimeError('WORLD_TASK_WAIST_TARGET_SOURCE_RANGE:'+name)
        # A new navigation heading must not silently become a new wrist/model.
        import mujoco
        from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
        scene=json.loads((output/'live_view_scene.json').read_text())['scene'];model=mujoco.MjModel.from_xml_path(scene['path']);data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row)
        root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');robots={i for i in range(model.nbody) if model.body_rootid[i]==root};ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in WAIST};bad=[]
        for index in range(21):
            fraction=index/20.
            for name,j in ids.items():data.qpos[model.jnt_qposadr[j]]=row['joints'][name]['q']+fraction*(target[name]-row['joints'][name]['q'])
            mujoco.mj_forward(model,data)
            for c in data.contact:
                if c.dist>=-.001:continue
                a,b=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,j) for j in (a,b)]
                if not (a in robots or b in robots) or 'world' in names:continue
                if set(names)=={'pelvis','left_hip_roll_link'} and c.dist>=-.005:continue
                bad.append(dict(fraction=fraction,pair=names,distance_m=float(c.dist)))
        if bad:raise RuntimeError('WORLD_FRAME_WAIST_PATH_COLLISION:'+str(bad[:2]))
        proof=dict(classification='ACTUAL_STATE_WORLD_TORSO_FRAME_STATIC_PREFLIGHT',mj_step_calls=0,pelvis_yaw_rad=root_yaw,torso_world_yaw_target_rad=desired,waist_target_rad=target,samples=21,violations=bad)
        pg.write_new_text(output/'world_torso_waist_static_preflight.json',json.dumps(proof,indent=2)+'\n')
        row=read_native(output/'native_latest.json')
    request=dict(schema="robotsim.waist-preposition.v1",generation=row["generation"],created_host_ns=time.monotonic_ns(),duration_s=4.,lease_s=300.,start_rad={n:row["joints"][n]["q"] for n in WAIST},target_rad=target,source_limits={n:limits[n] for n in WAIST},effort_caps_nm=effort_caps)
    pending=output/"standing_waist_request.tmp"
    pg.write_new_text(pending,json.dumps(request,indent=2)+"\n")
    os.replace(pending,output/"standing_waist_request.json")
    deadline=time.monotonic()+30.
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
            if f.get("request_created_host_ns")==request["created_host_ns"] and f["generation"]==row["generation"] and 0.<=(time.monotonic_ns()-f["source_host_ns"])*1e-9<=.20 and f["fraction"]>=1. and waist_position_ready({n:actual["joints"][n]["q"] for n in WAIST},target,float(proposal.get("task_roll_readiness_tolerance_rad",.025))) and max(abs(actual["joints"][n]["qvel"]) for n in WAIST)<.05:
                return dict(request=request,feedback=f,actual_source_host_ns=actual["host_monotonic_ns"],task_roll_readiness_tolerance_rad=float(proposal.get("task_roll_readiness_tolerance_rad",.025)),strict_original_position_025_pass=waist_position_ready({n:actual["joints"][n]["q"] for n in WAIST},target),position_error_rad={n:actual["joints"][n]["q"]-target[n] for n in WAIST})
        time.sleep(.02)
    raise RuntimeError("WAIST_PREPOSITION_NOT_REACHED")

def pre_raise_forearm(output, proposal, limits, effort_cap):
    """Source-bounded same-owner elbow raise before torso lowering/turning."""
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_base_micro_probe import physical_fault,native_pose
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
    import mujoco
    row=read_native(output/'native_latest.json');name='right_elbow_joint';target=float(proposal['pre_raise_forearm_target_rad']);lo,hi,speed=limits[name]
    if not lo+.03<=target<=hi-.03:raise ValueError('PRE_RAISE_SOURCE_MARGIN')
    scene=json.loads((output/'live_view_scene.json').read_text())['scene'];model=mujoco.MjModel.from_xml_path(scene['path']);data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row)
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');robots={i for i in range(model.nbody) if model.body_rootid[i]==root};ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in row['joints']};rootadr=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);lower=json.loads(Path(proposal['native_lowering_reference']['path']).read_text());legs=[n for n in ids if any(x in n for x in ('hip_','knee_','ankle_'))];feet={'left_ankle_pitch_link','left_ankle_roll_link','right_ankle_pitch_link','right_ankle_roll_link'}
    R=row['bodies']['pelvis']['rotation'];yaw=math.atan2(R[3],R[0]);goals={**proposal['waist_target_rad'],'waist_roll_joint':row['joints']['waist_roll_joint']['q']};goals['waist_yaw_joint']=(proposal['torso_world_yaw_target_rad']-yaw+math.pi)%(2*math.pi)-math.pi;bad=[]
    for stage in ('elbow','lower','waist'):
        for index in range(21):
            f=index/20.;apply_recorded_native_frame(model,data,row)
            data.qpos[model.jnt_qposadr[ids[name]]]=row['joints'][name]['q']+(f if stage=='elbow' else 1.)*(target-row['joints'][name]['q'])
            if stage in ('lower','waist'):
                u=f if stage=='lower' else 1.;data.qpos[rootadr+2]-=.10*u
                for n in legs:data.qpos[model.jnt_qposadr[ids[n]]]=row['joints'][n]['q']+u*(lower['joints'][n]['q']-row['joints'][n]['q'])
            if stage=='waist':
                for n,goal in goals.items():data.qpos[model.jnt_qposadr[ids[n]]]=row['joints'][n]['q']+f*(goal-row['joints'][n]['q'])
            mujoco.mj_forward(model,data)
            for c in data.contact:
                if c.dist>=-.001:continue
                x,y=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,j) for j in (x,y)]
                if not (x in robots or y in robots):continue
                if 'world' in names and any(n in feet for n in names):continue
                if set(names)=={'pelvis','left_hip_roll_link'} and c.dist>=-.005:continue
                bad.append(dict(stage=stage,fraction=f,pair=names,distance_m=float(c.dist)))
    proof=dict(classification='ACTUAL_STANDING_STATE_ELBOW_LOWER_WAIST_STATIC_PREFLIGHT',mj_step_calls=0,samples=63,elbow_target_rad=target,waist_target_rad=goals,violations=bad)
    pg.write_new_text(output/'pre_raise_full_sequence_static.json',json.dumps(proof,indent=2)+'\n')
    if bad:raise RuntimeError('PRE_RAISE_FULL_SEQUENCE_COLLISION:'+str(bad[:2]))
    row=read_native(output/'native_latest.json');request=dict(schema='robotsim.forearm-pre-raise.v1',generation=row['generation'],created_host_ns=time.monotonic_ns(),duration_s=3.,start_rad=row['joints'][name]['q'],target_rad=target,source_limits=limits[name],effort_cap_nm=effort_cap)
    temp=output/'pre_raise_forearm_request.tmp';pg.write_new_text(temp,json.dumps(request)+'\n');os.replace(temp,output/'pre_raise_forearm_request.json')
    deadline=time.monotonic()+12.
    while time.monotonic()<deadline:
        actual=read_native(output/'native_latest.json');fault=physical_fault(actual,native_pose(actual),limits,robot_bodies=proposal['robot_bodies'],review_incidental_contacts=True)
        if fault:raise RuntimeError('PRE_RAISE_PHYSICAL:'+fault)
        for c in actual['contacts']:
            if 'test_cylinder' in (c['body1'],c['body2']) and any(n in proposal['robot_bodies'] for n in (c['body1'],c['body2'])) and c['local_force_torque'][0]>.1:raise RuntimeError('PRE_RAISE_UNINTENDED_OBJECT_CONTACT')
        f=output/'pre_raise_forearm_feedback.json'
        if f.exists():
            value=json.loads(f.read_text())
            if value['generation']==row['generation'] and value['fraction']>=1. and 0.<=(time.monotonic_ns()-value['source_host_ns'])*1e-9<=.20 and abs(actual['joints'][name]['q']-target)<.025 and abs(actual['joints'][name]['qvel'])<.05:
                result=dict(pass_result=True,request=request,feedback=value,only_joint_command_changed=name);pg.write_new_text(output/'pre_raise_forearm_result.json',json.dumps(result,indent=2)+'\n');return result
        time.sleep(.02)
    raise RuntimeError('PRE_RAISE_ACTUAL_ELBOW_NOT_REACHED')

def load_runtime_proposal(output, runtime_proposal_path=None):
    """Select the exact staged task proposal without changing its source limits."""
    effective_path=Path(runtime_proposal_path) if runtime_proposal_path is not None else output/"standing_grasp_proposal.json"
    return effective_path,json.loads(effective_path.read_text())


def execute_preposition(run,container,environment,output,proposal_path, *, runtime_proposal_path=None):
    import xml.etree.ElementTree as ET
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract
    effective_path,proposal=load_runtime_proposal(output,runtime_proposal_path)
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
    pre_raise=None
    if proposal.get("pre_raise_forearm_target_rad") is not None:pre_raise=pre_raise_forearm(output,proposal,limits,float(joints["right_elbow_joint"].find("limit").attrib["effort"]))
    for label in ("body_crouch_pickup",):
        child=run(["docker","exec",container,"bash","-lc",environment+"timeout 65s python3 /workspace/evidence/"+label+".py"],timeout_s=68.,check=False)
        pg.write_new_text(output/(label+".stdout"),child.stdout)
        path=output/(label+"_result.json")
        result=json.loads(path.read_text()) if path.exists() else {}
        if child.returncode or not result.get("pass_result"):raise RuntimeError("BODY_CROUCH_PICKUP_NOT_QUALIFIED:"+str(result.get("failure",child.stdout[-1000:])))
    hold=request_and_wait(output,proposal,limits,caps)
    pg.write_new_text(output/"standing_waist_result.json",json.dumps(hold,indent=2)+"\n")
    child=run(["docker","exec",container,"bash","-lc",environment+"timeout 30s python3 /workspace/evidence/body_posture_hold.py"],timeout_s=33.,check=False)
    pg.write_new_text(output/"body_posture_hold.stdout",child.stdout)
    record=output/"body_posture_hold_result.json"
    observed=json.loads(record.read_text()) if record.exists() else {}
    if child.returncode or not observed.get("pass_result"):raise RuntimeError("TURN_FORWARD_BEND_COMPOSITE_HOLD_NOT_QUALIFIED:"+str(observed.get("failure",child.stdout[-1000:])))
    return dict(classification="SIMULATION_ONLY_NATIVE_CROUCH_TORSO_TURN_FORWARD_BEND",roll_policy="RETAIN_MEASURED_NO_ACTIVE_SIDE_BEND",
                pre_raise_forearm=pre_raise,initial_waist_hold=initial_waist_hold,body_crouch=result,waist=hold,posture_stop=observed,proposal_path=str(proposal_path),effective_runtime_proposal_path=str(effective_path))
