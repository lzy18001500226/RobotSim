"""Restore native standing height/waist ownership while the existing owner holds."""
from pathlib import Path
import inspect,json,math,time
WAIST=("waist_yaw_joint","waist_pitch_joint","waist_roll_joint")
from scripts.research.issue46.native_runtime.native_base_micro_probe import observed_clock_step

def start_owner_feedback(owner):
    """Reuse the micro/bootstrap executor pattern; no new command publisher."""
    import threading
    owner.owner_waist_lock = threading.RLock()
    if not USE_NATIVE_FEEDBACK:
        return
    from rclpy.executors import SingleThreadedExecutor
    owner.owner_feedback_errors = []
    owner.owner_feedback_executor = SingleThreadedExecutor()
    owner.owner_feedback_executor.add_node(owner)

    def spin():
        try:
            owner.owner_feedback_executor.spin()
        except Exception as exc:
            owner.owner_feedback_errors.append(repr(exc))

    owner.owner_feedback_thread = threading.Thread(
        target=spin, name="carry-owner-feedback", daemon=True)
    owner.owner_feedback_thread.start()


def check_owner_feedback(owner):
    if USE_NATIVE_FEEDBACK and (owner.owner_feedback_errors
            or not owner.owner_feedback_thread.is_alive()):
        raise RuntimeError("CARRY_FEEDBACK_EXECUTOR_FAILED:"
                           + str(owner.owner_feedback_errors))


def wait_owner_feedback(owner, seconds):
    """Control-loop wait; exactly one independent executor owns ROS callbacks."""
    end = time.monotonic() + seconds
    if not USE_NATIVE_FEEDBACK:
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(owner, timeout_sec=.002)
        return
    check_owner_feedback(owner)
    while rclpy.ok() and time.monotonic() < end:
        time.sleep(min(.005, max(0., end - time.monotonic())))
        check_owner_feedback(owner)


def stop_owner_feedback(owner):
    if not USE_NATIVE_FEEDBACK:
        return
    owner.owner_feedback_executor.shutdown(timeout_sec=1.)
    owner.owner_feedback_thread.join(timeout=1.)
    if owner.owner_feedback_thread.is_alive():
        raise RuntimeError("CARRY_FEEDBACK_SHUTDOWN_UNCONFIRMED")


def receive_owner_waist(owner, message):
    # Judge the packet being received, not a past scheduling interruption.
    # receive_native_waist still rejects stale/repeated stamps and invalid or
    # out-of-range source commands. Record recovered gaps without killing the
    # independent executor before it can deliver the fresh packet.
    with owner.owner_waist_lock:
        previous=getattr(owner,"mc_waist_receipt",None)
        receive_native_waist(owner, message)
        current=getattr(owner,"mc_waist_receipt",None)
        if previous is not None and current is not None and current-previous>.25:
            owner.mc_waist_last_recovered_gap_s=current-previous
            owner.mc_waist_recovered_gap_count=getattr(owner,"mc_waist_recovered_gap_count",0)+1


def await_fresh_waist_relay(owner):
    """Existing200ms source age, with one50ms wait for a real new callback."""
    started=time.monotonic();initial=owner.mc_waist_receipt
    if started-initial<=.20:return
    deadline=started+.05
    while True:
        now=time.monotonic()
        if owner.mc_waist_receipt>initial and 0.<=now-owner.mc_waist_receipt<=.20:
            owner.mc_waist_recovery=dict(previous_receipt_s=initial,
                new_receipt_s=owner.mc_waist_receipt,wait_s=now-started,
                initial_age_s=started-initial,source_age_limit_s=.20)
            return
        if now>=deadline:
            raise RuntimeError("MC_WAIST_RELAY_STALE:"+json.dumps(dict(
                age_s=now-owner.mc_waist_receipt,reread_wait_s=now-started,
                reread_budget_s=.05,last_callback_timing_ns=getattr(owner,"mc_waist_callback_timing",None))))
        check_owner_feedback(owner)
        time.sleep(min(.002,deadline-now))


def freeze_owner_waist(owner):
    if USE_NATIVE_FEEDBACK:
        with owner.owner_waist_lock:
            owner.mc_waist_abort_frozen = True
    else:
        owner.mc_waist_abort_frozen = True


def receive_native_waist(self,message):
    import copy,os,time,json,math
    callback_started_ns=time.monotonic_ns()
    if len(message.joints)!=3 or {j.name for j in message.joints}!=set(WAIST):raise RuntimeError("MC_WAIST_GROUP_CHANGED")
    if any(not all(math.isfinite(float(getattr(j,k))) for k in ("position","velocity","effort","stiffness","damping")) or j.stiffness<=0. or j.damping<=0. for j in message.joints):raise RuntimeError("MC_WAIST_SOURCE_INVALID")
    stamp=int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec)
    before=time.monotonic_ns();wall=self.get_clock().now().nanoseconds;after=time.monotonic_ns()
    self.mc_waist_clock_offset,clock_step=observed_clock_step(getattr(self,'mc_waist_clock_offset',None),before,wall,after)
    if clock_step is not None:
        self.mc_waist_source_clock_generation=getattr(self,'mc_waist_source_clock_generation',0)+1
        self.mc_waist_source_stamp_ns=-1
    source_age=(wall-stamp)*1e-9
    if not 0.<=source_age<=.20 or stamp<=getattr(self,"mc_waist_source_stamp_ns",-1):return
    self.mc_waist_source_stamp_ns=stamp
    self.mc_waist_message=copy.deepcopy(message);self.mc_waist_receipt=time.monotonic()
    if (getattr(self,"mc_waist_blend_started",None) is None or getattr(self,"mc_waist_abort_frozen",False) or getattr(self,"stationary_delivery_waist_target",None) is not None):return
    source={j.name:float(j.position) for j in message.joints}
    fraction=min(1.,(time.monotonic()-self.mc_waist_blend_started)/4.)
    smooth=fraction*fraction*(3.-2.*fraction)
    for joint in message.joints:
        lo,hi,speed=LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][joint.name]
        raw_position=float(joint.position)
        # At completed handoff forward the exact source endpoint. The prior
        # start+1*(target-start) introduced a5e-17rad false range violation for
        # the legal MC endpoint-0.314. No source range or physical qpos changes.
        start=self.mc_waist_blend_from[joint.name]
        joint.position=raw_position if smooth==1. else start if smooth==0. else (1.-smooth)*start+smooth*raw_position
        if not lo<=joint.position<=hi:raise RuntimeError('MC_WAIST_PUBLISHED_TARGET_RANGE:'+json.dumps(dict(joint=joint.name,source_position_rad=raw_position,published_position_rad=float(joint.position),source_range_rad=[lo,hi],fraction=fraction)))
    before_publish_ns=time.monotonic_ns()
    self.waist_pub.publish(message);self.last_waist_message=message;self.mc_waist_relay_count=getattr(self,"mc_waist_relay_count",0)+1
    after_publish_ns=time.monotonic_ns()
    self.mc_waist_callback_timing={"callback_started_ns":callback_started_ns,"before_publish_ns":before_publish_ns,"after_publish_ns":after_publish_ns,"source_receipt_s":self.mc_waist_receipt}
    now=time.monotonic()
    if now-getattr(self,"mc_waist_report_at",0.)>=.10:
        value=dict(schema="robotsim.native-waist-handoff-feedback.v1",owner_pid=os.getpid(),generation=self.native_generation,host_monotonic_ns=time.monotonic_ns(),source_receipt_monotonic_s=self.mc_waist_receipt,source_stamp_ns=int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec),source_clock_generation=getattr(self,'mc_waist_source_clock_generation',0),source_positions_rad=source,published_positions_rad={j.name:float(j.position) for j in message.joints},callback_timing_ns=dict(self.mc_waist_callback_timing),fraction=fraction,relay_count=self.mc_waist_relay_count,gains="UNCHANGED_CURRENT_MC",bounded_relay_recovery=getattr(self,"mc_waist_recovery",None))
        with open("/workspace/evidence/native_waist_handoff_feedback.tmp","w") as f:json.dump(value,f)
        os.replace("/workspace/evidence/native_waist_handoff_feedback.tmp","/workspace/evidence/native_waist_handoff_feedback.json");self.mc_waist_report_at=now
        self.mc_waist_callback_timing["report_io_finished_ns"]=time.monotonic_ns()

def torso_carry_reference(row, center, local_axis):
    """Re-express the current measured tool pose; do not move any body."""
    hand=row['bodies']['R_omnipicker_base_link'];body=row['bodies']['torso_link']
    H=hand['rotation'];B=body['rotation']
    point=[hand['position_m'][i]+sum(H[3*i+j]*center[j] for j in range(3)) for i in range(3)]
    axis=[sum(H[3*i+j]*local_axis[j] for j in range(3)) for i in range(3)]
    delta=[point[i]-body['position_m'][i] for i in range(3)]
    return dict(carry_pose_frame='TORSO_RIGID',carry_rigid_offset_base_m=[sum(B[3*i+j]*delta[i] for i in range(3)) for j in range(3)],carry_rigid_axis_base=[sum(B[3*i+j]*axis[i] for i in range(3)) for j in range(3)])


def request_torso_carry_reference(self):
    import os,json,time
    path='/workspace/evidence/carry_torso_reference_request.json'
    if getattr(self,'carry_torso_reference_done',False) or not os.path.exists(path):return
    req=json.load(open(path));row=self.native_row
    if req.get('owner_pid')!=os.getpid() or req.get('generation')!=self.native_generation or not 0.<=(time.monotonic_ns()-req['created_host_ns'])*1e-9<=.20:raise RuntimeError('TORSO_CARRY_REFERENCE_REQUEST_NOT_FRESH')
    if getattr(self,'mc_waist_blend_started',None) is None or time.monotonic()-self.mc_waist_blend_started<4.:raise RuntimeError('TORSO_CARRY_REFERENCE_BEFORE_NATIVE_WAIST_HANDOFF')
    if not 0.<=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('TORSO_CARRY_REFERENCE_NATIVE_STALE')
    reference=torso_carry_reference(row,CIRCLE_CONTRACT['circle_center_local_m'],CIRCLE_CONTRACT['local_bottle_axis_in_hand'])
    if req.get('standing_restore_pass') is not True:raise RuntimeError('TORSO_REFERENCE_REQUIRES_QUALIFIED_STANDING')
    self.lift_phase.update(reference,carry_nominal_right_arm_rad={name:float(row['joints'][name]['q']) for name in RIGHT},postured_restore_torso_reference=dict(schema='robotsim.postured-carry-frame.v1',standing_restore_pass=True,owner_pid=os.getpid(),generation=self.native_generation,source_host_ns=row['host_monotonic_ns']))
    self.write_phase(self.lift_phase);self.carry_torso_reference_done=True
    record=dict(schema='robotsim.carry-torso-reference.v1',owner_pid=os.getpid(),generation=self.native_generation,source_host_ns=row['host_monotonic_ns'],host_monotonic_ns=time.monotonic_ns(),reference=reference,scope='CURRENT_MEASURED_TOOL_POSE_REEXPRESSION_AFTER_STANDING_NOT_NEW_MOTION',extra_wrist_stow=False)
    with open('/workspace/evidence/carry_torso_reference_feedback.tmp','x') as f:json.dump(record,f)
    os.replace('/workspace/evidence/carry_torso_reference_feedback.tmp','/workspace/evidence/carry_torso_reference_feedback.json')



def request_idle_left_parking(self,target):
    """Move only the unloaded shoulder, using this existing arm owner."""
    import os,json,time
    path="/workspace/evidence/idle_left_parking_request.json"
    joint="left_shoulder_roll_joint"
    if getattr(self,"idle_left_parking_started",None) is None:
        if not os.path.exists(path):return target
        req=json.load(open(path));now=time.monotonic_ns()
        if (req.get("schema")!="robotsim.idle-left-parking.v1"
                or req.get("owner_pid")!=os.getpid()
                or req.get("generation")!=self.native_generation
                or req.get("pass_static") is not True
                or not getattr(self,"carry_torso_reference_done",False)
                or not 0.<=(now-req["created_host_ns"])*1e-9<=.20):
            raise RuntimeError("IDLE_LEFT_PARKING_REQUEST_UNQUALIFIED")
        lo,hi,speed=LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][joint]
        end=float(req["target_rad"])
        if end!=.35 or not lo+.01<=end<=hi-.01 or abs(end-target[joint])*1.5/4.>speed:
            raise RuntimeError("IDLE_LEFT_PARKING_SOURCE_LIMIT")
        self.idle_left_parking_started=time.monotonic()
        self.idle_left_parking_from=float(target[joint])
    fraction=min(1.,(time.monotonic()-self.idle_left_parking_started)/4.)
    u=fraction*fraction*(3.-2.*fraction)
    target=dict(target);target[joint]=(1.-u)*self.idle_left_parking_from+u*.35
    if time.monotonic()-getattr(self,"idle_left_parking_report",0.)>=.10:
        self.idle_left_parking_report=time.monotonic()
        value=dict(owner_pid=os.getpid(),generation=self.native_generation,
                   host_monotonic_ns=time.monotonic_ns(),fraction=fraction,
                   command_rad=target[joint],actual_rad=self.native_row["joints"][joint]["q"],
                   right_arm_target_changed=False,source_limits_changed=False)
        with open("/workspace/evidence/idle_left_parking_feedback.tmp","w") as f:json.dump(value,f)
        os.replace("/workspace/evidence/idle_left_parking_feedback.tmp","/workspace/evidence/idle_left_parking_feedback.json")
    return target


def request_carry_waist_return(self,target):
    import os,json,time
    path='/workspace/evidence/carry_waist_return_request.json'
    if getattr(self,'carry_waist_return_started',None) is None and os.path.exists(path):
        req=json.load(open(path));baseline=json.load(open('/workspace/evidence/standing_waist_initial_hold.json'))['request']
        if (req.get('owner_pid')!=os.getpid() or req.get('generation')!=self.native_generation
            or baseline.get('generation')!=self.native_generation or req.get('target_rad')!=baseline['target_rad']
            or not 0.<=(time.monotonic_ns()-req['created_host_ns'])*1e-9<=.20):raise RuntimeError('CARRY_WAIST_RETURN_REQUEST_UNQUALIFIED')
        self.carry_waist_return_started=time.monotonic();self.carry_waist_return_from={name:float(target[name]) for name in WAIST};self.carry_waist_return_target=dict(req['target_rad'])
    if getattr(self,'carry_waist_return_started',None) is None:return target
    fraction=min(1.,(time.monotonic()-self.carry_waist_return_started)/4.);u=fraction*fraction*(3.-2.*fraction)
    target=dict(target)
    for name in WAIST:
        lo,hi=LIFT_CONTRACT['retract_preflight']['source_motion_limits'][name][:2]
        start=self.carry_waist_return_from[name];end=self.carry_waist_return_target[name]
        target[name]=end if u==1. else (1.-u)*start+u*end
        if not lo<=target[name]<=hi:raise RuntimeError('CARRY_WAIST_RETURN_SOURCE_LIMIT:'+name)
    if time.monotonic()-getattr(self,'carry_waist_return_report',0.)>=.10:
        value=dict(owner_pid=os.getpid(),generation=self.native_generation,fraction=fraction,target_rad={name:target[name] for name in WAIST},host_monotonic_ns=time.monotonic_ns(),gains_and_gravity_conversion='UNCHANGED_EXISTING_OWNER')
        with open('/workspace/evidence/carry_waist_return_feedback.tmp','w') as stream:json.dump(value,stream)
        os.replace('/workspace/evidence/carry_waist_return_feedback.tmp','/workspace/evidence/carry_waist_return_feedback.json');self.carry_waist_return_report=time.monotonic()
    return target


def request_stationary_delivery_waist(self, target):
    """Reuse the existing bias-compensated waist owner after a real route stop."""
    import os,json,time,math
    path="/workspace/evidence/stationary_delivery_waist_request.json"
    if getattr(self,"stationary_delivery_waist_target",None) is None and os.path.exists(path):
        req=json.load(open(path));row=self.native_row
        if (req.get("schema")!="robotsim.stationary-delivery-waist.v1"
                or req.get("owner_pid")!=os.getpid() or req.get("generation")!=self.native_generation
                or req.get("physical_stop_pass") is not True
                or not 0.<=(time.monotonic_ns()-req["created_host_ns"])*1e-9<=.20
                or not 0.<=(time.monotonic_ns()-row["host_monotonic_ns"])*1e-9<=.20):
            raise RuntimeError("STATIONARY_DELIVERY_WAIST_REQUEST_UNQUALIFIED")
        if row.get("generation")!=self.native_generation:raise RuntimeError("STATIONARY_DELIVERY_WAIST_GENERATION")
        with self.owner_waist_lock:
            hold={name:float(row["joints"][name]["q"]) for name in WAIST}
            checks={}
            for name in WAIST:
                j=row["joints"][name];kp,kd=GAINS[name]
                limits=LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][name]
                if abs(j["qvel"])>.05:raise RuntimeError("STATIONARY_DELIVERY_WAIST_NOT_STOPPED")
                command=position_mode_setpoint(hold[name],j["q"],j["qvel"],j["bias_force"],kp,kd,limits,120. if name=="waist_yaw_joint" else 36.,bound_effort=True)
                checks[name]=dict(desired_rad=hold[name],equivalent_position_command_rad=command,kp=kp,kd=kd,source_limits=limits)
            self.stationary_delivery_waist_target=hold
            value=dict(schema=req["schema"],owner_pid=os.getpid(),generation=self.native_generation,host_monotonic_ns=time.monotonic_ns(),target_rad=hold,source_checks=checks,policy="EXISTING_POSITION_MODE_GRAVITY_COMPENSATION_UNCHANGED_GAINS")
            with open("/workspace/evidence/stationary_delivery_waist_feedback.tmp","x") as f:json.dump(value,f)
            os.replace("/workspace/evidence/stationary_delivery_waist_feedback.tmp","/workspace/evidence/stationary_delivery_waist_feedback.json")
    hold=getattr(self,"stationary_delivery_waist_target",None)
    return {**target,**hold} if hold is not None else target


def request_native_waist_handoff(self):
    import os,json,time
    path="/workspace/evidence/native_waist_handoff_request.json"
    if getattr(self,"mc_waist_blend_started",None) is None and os.path.exists(path):
        request=json.load(open(path))
        if request.get("schema")!="robotsim.native-waist-handoff.v1" or request.get("owner_pid")!=os.getpid() or request.get("generation")!=self.native_generation or not 0.<=(time.monotonic_ns()-request["created_host_ns"])*1e-9<=.20:raise RuntimeError("MC_WAIST_HANDOFF_REQUEST_UNQUALIFIED")
        publishers=self.get_publishers_info_by_topic("/robotsim/issue46/mc/waist/command")
        public=self.get_publishers_info_by_topic("/aima/hal/joint/waist/command")
        if len(publishers)!=1 or publishers[0].node_name!=request["expected_mc_node"] or publishers[0].node_namespace!="/" or len(public)!=1 or public[0].node_name!=self.get_name() or public[0].node_namespace!="/":raise RuntimeError("MC_WAIST_HANDOFF_OWNER_GRAPH")
        if getattr(self,"mc_waist_receipt",None) is None or time.monotonic()-self.mc_waist_receipt>.20 or not hasattr(self,"last_waist_message"):raise RuntimeError("MC_WAIST_SOURCE_NOT_FRESH_FOR_HANDOFF")
        initial=dict(schema='robotsim.native-waist-source-before-handoff.v1',owner_pid=os.getpid(),generation=self.native_generation,source_stamp_ns=self.mc_waist_source_stamp_ns,source_receipt_monotonic_s=self.mc_waist_receipt,source_joints={j.name:{k:float(getattr(j,k)) for k in ('position','velocity','effort','stiffness','damping')} for j in self.mc_waist_message.joints},last_published_positions_rad={j.name:float(j.position) for j in self.last_waist_message.joints})
        with open('/workspace/evidence/native_waist_source_before_handoff.json','x') as f:json.dump(initial,f)
        self.mc_waist_blend_from={j.name:float(j.position) for j in self.last_waist_message.joints}
        self.mc_waist_blend_started=time.monotonic()
    # Stationary delivery has explicitly reclaimed waist command ownership.
    # Its source-checked native-state position hold no longer consumes MC
    # commands; keep relay freshness mandatory only while that relay is active.
    if (getattr(self,"mc_waist_blend_started",None) is not None
            and getattr(self,"stationary_delivery_waist_target",None) is None):
        await_fresh_waist_relay(self)

def extend_existing_owner(script):
    """Same arm/hand/waist publishers; only the waist source changes after proof."""
    def replace(old,new):
        nonlocal script
        if script.count(old)!=1:raise ValueError("postured restoration owner anchor:"+old[:60])
        script=script.replace(old,new)
    helper="\n".join(inspect.getsource(f) for f in (observed_clock_step,start_owner_feedback,check_owner_feedback,wait_owner_feedback,stop_owner_feedback,receive_owner_waist,await_fresh_waist_relay,freeze_owner_waist,receive_native_waist,torso_carry_reference,request_torso_carry_reference,request_idle_left_parking,request_carry_waist_return,request_stationary_delivery_waist,request_native_waist_handoff))
    replace("class Plan(Node):",helper+"\nclass Plan(Node):")
    anchor="self.waist_pub=self.create_publisher("
    lines=script.splitlines();index=next(i for i,l in enumerate(lines) if l.strip().startswith(anchor))
    lines.insert(index+1,"   self.create_subscription(JointCommandArray,\'/robotsim/issue46/mc/waist/command\',lambda m:receive_owner_waist(self,m),QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))")
    script="\n".join(lines)+"\n"
    replace("   if WAIST:\n    w=JointCommandArray();w.header=m.header","   if stage==\'CARRY_HOLD\':target=request_carry_waist_return(self,target);request_native_waist_handoff(self);request_torso_carry_reference(self);target=request_stationary_delivery_waist(self,target)\n   if WAIST and (getattr(self,\'mc_waist_blend_started\',None) is None or getattr(self,\'stationary_delivery_waist_target\',None) is not None):\n    w=JointCommandArray();w.header=m.header")
    replace("    self.waist_pub.publish(w)","    self.waist_pub.publish(w);self.last_waist_message=w")
    replace('   published_positions=dict(target)', "   if stage=='CARRY_HOLD':target=request_idle_left_parking(self,target)\n   published_positions=dict(target)")
    replace("self.last_published_arm_target={name:float(target[name]) for name in LEFT+RIGHT}","self.last_published_arm_target={name:float(target[name]) for name in CONTROL}")
    replace(" def send(self,target,seconds,stage,hand_right=None):", " def send(self,target,seconds,stage,hand_right=None):\n  if WAIST and stage in (\'CARRY_STOW\',\'CARRY_HOLD\'):target={**self.last_published_arm_target,**target}")
    replace("def carry_abort_last_command_hold(n, carry_error):","def carry_abort_last_command_hold(n, carry_error):"+chr(10)+" freeze_owner_waist(n)")
    replace("   n.pub.publish(n.last_arm_message);n.hand_pub.publish(n.last_hand_message)","   n.pub.publish(n.last_arm_message);n.hand_pub.publish(n.last_hand_message)\n   if getattr(n,\'last_waist_message\',None) is not None:n.waist_pub.publish(n.last_waist_message)")
    replace("rclpy.init();n=Plan()", "rclpy.init();n=Plan();start_owner_feedback(n)")
    replace("  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.002)",
            "  wait_owner_feedback(self,s)")
    replace("   rclpy.spin_once(self,timeout_sec=.001)",
            "   wait_owner_feedback(self,.001)")
    replace("  if not USE_NATIVE_FEEDBACK:return\n  path='/workspace/evidence/native_latest.json'",
            "  if not USE_NATIVE_FEEDBACK:return\n  check_owner_feedback(self)\n  path='/workspace/evidence/native_latest.json'")
    replace("finally:\n print(json.dumps({'execution_exception_context'",
            "finally:\n stop_owner_feedback(n)\n print(json.dumps({'execution_exception_context'")
    compile(script,"existing_owner_postured_stand_restore","exec")
    return script

def held_waist_return_ready(rows, baseline):
    """Require a continuous measured neutral-waist window before body rise."""
    if len(rows)<2:return False
    last=rows[-1];window=[r for r in rows if 0.<=last['sim_time_s']-r['sim_time_s']<=.25]
    if len(window)<2 or window[-1]['sim_time_s']-window[0]['sim_time_s']<.20:return False
    for before,after in zip(window,window[1:]):
        if (before['generation']!=after['generation'] or after['host_monotonic_ns']<=before['host_monotonic_ns']
            or not 0.<after['sim_time_s']-before['sim_time_s']<=.05):return False
    return all(all(math.isfinite(float(row['joints'][name][key])) for key in ('q','qvel'))
        and abs(row['joints'][name]['q']-baseline[name])<=.10
        and abs(row['joints'][name]['qvel'])<=.05 for row in window for name in WAIST)


def restore_postured_hold(run,container,environment,output,ready,result):
    """Return waist then height, then native MC ownership; the arm owner holds the tool."""
    import os
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    proof=json.loads((output/"body_crouch_pickup_result.json").read_text())
    expected_mc=proof["body_subscriber"]["name"]
    if not proof.get("pass_result") or proof.get("generation")!=ready["generation"]:raise RuntimeError("RESTORE_NEEDS_SAME_EPISODE_QUALIFIED_LOWERING")
    # Return the existing gravity-compensated waist target first. The native
    # MC crouch reference remains forward-leaning, so its relay stays deferred
    # until body height has returned. Arm IK holds the pelvis-relative tool pose.
    baseline=json.loads((output/'standing_waist_initial_hold.json').read_text())['request']['target_rad']
    request=dict(owner_pid=ready['pid'],generation=ready['generation'],created_host_ns=time.monotonic_ns(),target_rad=baseline)
    pg.write_new_text(output/'carry_waist_return_request.tmp',json.dumps(request)+'\n')
    os.replace(output/'carry_waist_return_request.tmp',output/'carry_waist_return_request.json')
    waist_rows=[];deadline=time.monotonic()+10.
    while time.monotonic()<deadline:
        row=read_fresh_native_row(output/'native_latest.json')
        health=json.loads((output/'carry_heartbeat.json').read_text())
        if health.get('healthy') is not True or health.get('pid')!=ready['pid'] or health.get('generation')!=row['generation'] or not 0.<=(time.monotonic_ns()-health['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('RESTORE_WAIST_OWNER_NOT_FRESH:'+json.dumps({k:health.get(k) for k in ('mode','healthy','failure','pid','generation','host_monotonic_ns')}))
        if not waist_rows or row['host_monotonic_ns']>waist_rows[-1]['host_monotonic_ns']:waist_rows.append(row)
        path=output/'carry_waist_return_feedback.json'
        if path.exists():
            returned=json.loads(path.read_text())
            if returned.get('owner_pid')==ready['pid'] and returned.get('generation')==ready['generation'] and returned.get('fraction')==1. and 0.<=(time.monotonic_ns()-returned['host_monotonic_ns'])*1e-9<=.20 and held_waist_return_ready(waist_rows,baseline):break
        time.sleep(.01)
    else:raise RuntimeError('HELD_WAIST_RETURN_BEFORE_RISE_NOT_QUALIFIED')
    waist_proof=dict(classification='MEASURED_EXISTING_OWNER_WAIST_RETURN_BEFORE_BODY_RISE',pass_result=True,baseline_rad=baseline,
        source_host_ns=row['host_monotonic_ns'],source_sim_time_s=row['sim_time_s'],actual_rad={name:row['joints'][name]['q'] for name in WAIST},
        actual_velocity_rad_s={name:row['joints'][name]['qvel'] for name in WAIST},window_sim_s=.20,position_tolerance_rad=.10,velocity_limit_rad_s=.05,
        joints_and_gains_changed=False,held_tool_reference='PELVIS_RIGID')
    pg.write_new_text(output/'held_waist_before_body_rise.json',json.dumps(waist_proof,indent=2)+'\n')
    result['held_waist_before_body_rise']=waist_proof
    observed=run(["docker","exec",container,"bash","-lc",environment+"timeout 65s python3 /workspace/evidence/body_restore_carry.py"],timeout_s=68.,check=False)
    pg.write_new_text(output/"body_restore_carry.stdout",observed.stdout)
    body_path=output/"body_restore_carry_result.json"
    body=json.loads(body_path.read_text()) if body_path.exists() else {}
    result["body_restore_carry"]=body
    if observed.returncode or not body.get("pass_result"):raise RuntimeError("HELD_BODY_RESTORE_NOT_QUALIFIED:"+str(body.get("failure",observed.stdout[-1000:])))
    row=read_fresh_native_row(output/"native_latest.json")
    request=dict(schema="robotsim.native-waist-handoff.v1",owner_pid=ready["pid"],generation=row["generation"],expected_mc_node=expected_mc,created_host_ns=time.monotonic_ns())
    pg.write_new_text(output/"native_waist_handoff_request.tmp",json.dumps(request)+"\n")
    os.replace(output/"native_waist_handoff_request.tmp",output/"native_waist_handoff_request.json")
    deadline=time.monotonic()+8.;feedback=None
    while time.monotonic()<deadline:
        read_fresh_native_row(output/"native_latest.json")
        health=json.loads((output/"carry_heartbeat.json").read_text())
        if not health.get("healthy") or health.get("pid")!=ready["pid"]:
            raise RuntimeError("RESTORE_GRIP_OWNER_NOT_FRESH:"+json.dumps({k:health.get(k) for k in ("mode","healthy","failure","pid","generation")}))
        if not 0.<=(time.monotonic_ns()-health["host_monotonic_ns"])*1e-9<=.20:
            # No new action: wait within the existing handoff deadline for a
            # current healthy owner heartbeat instead of rejecting one old read.
            time.sleep(.02)
            continue
        path=output/"native_waist_handoff_feedback.json"
        if path.exists():
            feedback=json.loads(path.read_text())
            if feedback.get("owner_pid")==ready["pid"] and feedback.get("generation")==ready["generation"] and feedback["fraction"]>=1. and feedback["relay_count"]>=2 and 0.<=(time.monotonic_ns()-feedback["host_monotonic_ns"])*1e-9<=.20:break
        time.sleep(.02)
    else:raise RuntimeError("NATIVE_MC_WAIST_HANDOFF_UNCONFIRMED")
    observed=run(["docker","exec",container,"bash","-lc",environment+"timeout 30s python3 /workspace/evidence/body_after_restore_observer.py"],timeout_s=33.,check=False)
    pg.write_new_text(output/"body_after_restore_observer.stdout",observed.stdout)
    path=output/"body_restore_final_result.json";stop=json.loads(path.read_text()) if path.exists() else {}
    if observed.returncode or not stop.get("pass_result"):raise RuntimeError("RESTORED_NATIVE_STAND_COMPOSITE_NOT_QUALIFIED:"+str(stop.get("failure",observed.stdout[-1000:])))
    request=dict(owner_pid=ready['pid'],generation=ready['generation'],created_host_ns=time.monotonic_ns(),standing_restore_pass=True)
    pg.write_new_text(output/'carry_torso_reference_request.tmp',json.dumps(request)+'\n')
    os.replace(output/'carry_torso_reference_request.tmp',output/'carry_torso_reference_request.json')
    until=time.monotonic()+3.;path=output/'carry_torso_reference_feedback.json'
    while time.monotonic()<until:
        read_fresh_native_row(output/'native_latest.json')
        if path.exists():
            frame=json.loads(path.read_text())
            if frame['owner_pid']==ready['pid'] and frame['generation']==ready['generation'] and 0.<=(time.monotonic_ns()-frame['host_monotonic_ns'])*1e-9<=.20:break
        time.sleep(.01)
    else:raise RuntimeError('RESTORED_TORSO_CARRY_REFERENCE_NOT_ACKNOWLEDGED')
    idle_left=park_idle_left_after_restore(output,ready)
    value=dict(classification="SIMULATION_ONLY_REAL_GRASP_NATIVE_HEIGHT_AND_WAIST_RESTORED",pass_result=True,body_restore=body,native_waist_handoff=feedback,standing_stop=stop,walking_carry_reference=frame,idle_left_parking=idle_left,velocity_publishers_created=0,locomotion_actions_sent=0,walking_qualification="NOT_RUN")
    pg.write_new_text(output/"postured_standing_restore_result.json",json.dumps(value,indent=2)+"\n")
    return value



def park_idle_left_after_restore(output,ready):
    """Statically screen the current parked-arm path before a bounded command."""
    import os
    import mujoco
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row,apply_recorded_native_frame
    from scripts.research.issue46.native_runtime.native_carry_posture import screen_native_stow
    row=read_fresh_native_row(output/"native_latest.json")
    manifest=json.loads((output/"live_review_model_manifest.json").read_text())
    from scripts.research.issue46.native_runtime.native_grasp_audit import digest
    if digest(Path(manifest["path"]))!=manifest["sha256"]:raise RuntimeError("IDLE_LEFT_PARKING_MODEL_HASH")
    model=mujoco.MjModel.from_binary_path(manifest["path"]);data=mujoco.MjData(model)
    apply_recorded_native_frame(model,data,row)
    names=[n for n in row["joints"] if n.startswith(("left_shoulder_","left_elbow_","left_wrist_","right_shoulder_","right_elbow_","right_wrist_"))]
    if len(names)!=14:raise RuntimeError("IDLE_LEFT_PARKING_ARM_INVENTORY")
    target={n:float(row["joints"][n]["q"]) for n in names}
    target["left_shoulder_roll_joint"]=.35
    proof=screen_native_stow(model,data,dict(arm_target_rad=target,duration_s=4.))
    pg.write_new_text(output/"idle_left_parking_static.json",json.dumps(proof,indent=2)+"\n")
    if not proof["pass_static"]:raise RuntimeError("IDLE_LEFT_PARKING_STATIC_COLLISION")
    row=read_fresh_native_row(output/"native_latest.json")
    if row["generation"]!=ready["generation"]:raise RuntimeError("IDLE_LEFT_PARKING_GENERATION")
    request=dict(schema="robotsim.idle-left-parking.v1",owner_pid=ready["pid"],
                 generation=ready["generation"],created_host_ns=time.monotonic_ns(),
                 pass_static=True,target_rad=.35)
    pg.write_new_text(output/"idle_left_parking_request.tmp",json.dumps(request)+"\n")
    os.replace(output/"idle_left_parking_request.tmp",output/"idle_left_parking_request.json")
    deadline=time.monotonic()+8.;stable_start=None
    while time.monotonic()<deadline:
        row=read_fresh_native_row(output/"native_latest.json")
        j=row["joints"]["left_shoulder_roll_joint"]
        path=output/"idle_left_parking_feedback.json"
        if path.exists():
            feedback=json.loads(path.read_text())
            qualified=(feedback["owner_pid"]==ready["pid"]
                       and feedback["generation"]==ready["generation"]
                       and 0.<=(time.monotonic_ns()-feedback["host_monotonic_ns"])*1e-9<=.20
                       and feedback["fraction"]==1. and .25<=j["q"]<=.45
                       and abs(j["qvel"])<=.05)
            if qualified:
                if stable_start is None:stable_start=row["sim_time_s"]
                if row["sim_time_s"]-stable_start>=.20:
                    result=dict(pass_result=True,actual_rad=j["q"],actual_velocity_rad_s=j["qvel"],
                                feedback=feedback,mj_step_calls_in_static_screen=0)
                    pg.write_new_text(output/"idle_left_parking_result.json",json.dumps(result,indent=2)+"\n")
                    return result
            else:stable_start=None
        time.sleep(.01)
    raise RuntimeError("IDLE_LEFT_PARKING_ACTUAL_TRACKING_UNQUALIFIED")


def postured_carry_requested(args):
    """Only the explicit H220 two-table route may request native restoration."""
    if not getattr(args,'restore_postured_carry',False):return False
    if not (args.standing_grasp_posture and args.physical_lift_test and args.fixed_reviewed_checkpoint and args.carry_delivery_config and args.carry_away_1m and args.reviewed_foundationwalk_profile):
        raise ValueError('postured carry requires H220 restore/transfer route')
    profile=json.loads(Path(args.carry_delivery_config).read_text())
    if (profile.get('bottle_height_m'),profile.get('bottle_radius_m'),profile.get('bottle_mass_kg'))!=(.22,.035,.2):raise ValueError('postured carry object identity')
    return True


def retire_owned_zero_bridge(run, container, output, phase):
    """After an observed stop, retire only this episode's completed bridge."""
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    row=read_fresh_native_row(output/'native_latest.json')
    velocity=row['bodies']['pelvis']['origin_velocity_world_angular_linear']
    if math.hypot(velocity[3],velocity[4])>.02 or max(abs(x) for x in velocity[:3])>.05:raise RuntimeError('BRIDGE_RETIRE_REQUIRES_ACTUAL_STOP')
    code="""import os,signal,time,json
from pathlib import Path
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(bytes([0]))
 except (OSError,PermissionError):continue
 if len(a)>1 and a[1]==b'/workspace/evidence/steady_cmd_vel_bridge.py':owned.append(int(p.name))
if len(owned)>1:raise RuntimeError('AMBIGUOUS_OWNED_BRIDGE')
for pid in owned:os.kill(pid,signal.SIGINT)
until=time.monotonic()+4.
while time.monotonic()<until and any(Path('/proc/'+str(pid)).exists() for pid in owned):time.sleep(.02)
if any(Path('/proc/'+str(pid)).exists() for pid in owned):raise RuntimeError('OWNED_BRIDGE_DID_NOT_EXIT')
print(json.dumps(dict(pass_result=True,retired_pids=owned,signal='SIGINT',scope='THIS_EPISODE_STEADY_BRIDGE_ONLY')))
"""
    completed=run(['docker','exec',container,'python3','-c',code],timeout_s=6.,check=False)
    if completed.returncode:raise RuntimeError('OWNED_BRIDGE_RETIRE_FAILED:'+completed.stdout[-1000:])
    value=json.loads(completed.stdout.strip().splitlines()[-1]);pg.write_new_text(output/('bridge_retired_'+phase+'.json'),json.dumps(value,indent=2)+'\n');return value


def acquire_stationary_delivery_waist(output, ready, navigation):
    """No new publisher; require qualified stop and fresh same-owner receipt."""
    import os
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    stop=navigation.get("functional_stop_composite",navigation.get("post_stop_composite",{}))
    if not navigation.get("pass_result") or stop.get("pass") is not True:raise RuntimeError("DELIVERY_WAIST_REQUIRES_PHYSICAL_STOP")
    row=read_fresh_native_row(output/"native_latest.json")
    if row["generation"]!=ready["generation"]:raise RuntimeError("DELIVERY_WAIST_GENERATION")
    req=dict(schema="robotsim.stationary-delivery-waist.v1",owner_pid=ready["pid"],generation=ready["generation"],created_host_ns=time.monotonic_ns(),physical_stop_pass=True)
    pg.write_new_text(output/"stationary_delivery_waist_request.tmp",json.dumps(req)+"\n")
    os.replace(output/"stationary_delivery_waist_request.tmp",output/"stationary_delivery_waist_request.json")
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:
        path=output/"stationary_delivery_waist_feedback.json"
        if path.exists():
            feedback=json.loads(path.read_text())
            if (feedback.get("owner_pid")==ready["pid"] and feedback.get("generation")==ready["generation"]
                    and 0.<=(time.monotonic_ns()-feedback["host_monotonic_ns"])*1e-9<=.20):return feedback
            raise RuntimeError("DELIVERY_WAIST_RECEIPT_UNQUALIFIED")
        time.sleep(.01)
    raise RuntimeError("DELIVERY_WAIST_RECEIPT_TIMEOUT")


def run_body_stage(run, container, environment, output, label):
    from scripts.agent import path_governance as pg
    if label not in ('body_crouch_delivery','body_restore_after_delivery'):raise ValueError('body delivery stage')
    completed=run(['docker','exec',container,'bash','-lc',environment+'timeout 65s python3 /workspace/evidence/'+label+'.py'],timeout_s=68.,check=False)
    pg.write_new_text(output/(label+'.stdout'),completed.stdout)
    path=output/(label+'_result.json');value=json.loads(path.read_text()) if path.exists() else {}
    if completed.returncode or not value.get('pass_result'):raise RuntimeError('DELIVERY_BODY_STAGE_FAILED:'+label+':'+str(value.get('failure',completed.stdout[-700:])))
    return value


def screen_measured_route(args, output, replan_record, label, *, already_lowered=False):
    """Run the existing full-model screen against current measured carry state."""
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row,digest
    from scripts.research.issue46.native_runtime.native_carry_placement import prepare_delivery_scene
    cfg=json.loads(args.carry_delivery_config.read_text());cfg.pop('require_restored_route_preflight',None);cfg.pop('carry_native_neutral_stow',None)
    if already_lowered:cfg.pop('delivery_body_lowering',None)
    row=read_fresh_native_row(output/'native_latest.json');snapshot=output/label;snapshot.mkdir()
    pg.write_new_text(snapshot/'native_latest.json',json.dumps(row)+'\n');pg.write_new_bytes(snapshot/'native_hal_bilateral_arm_pregrasp.py',(output/'native_hal_bilateral_arm_pregrasp.py').read_bytes())
    base=row['bodies']['pelvis'];R=base['rotation']
    if label!='restored_route_snapshot':cfg.update(base_goal_world_xy=base['position_m'][:2],base_goal_world_yaw_rad=math.atan2(R[3],R[0]))
    cfg['route_reference']={'path':str(snapshot),'native_latest_sha256':digest(snapshot/'native_latest.json')}
    pg.write_new_text(snapshot/'config.json',json.dumps(cfg,indent=2)+'\n')
    cache=replan_record.get('host_model_cache');cache=cache.get('path') if isinstance(cache,dict) else cache
    result=prepare_delivery_scene(args.claw_control_overlay_result,snapshot/'config.json',host_model_cache=cache)
    after=read_fresh_native_row(output/'native_latest.json')
    if after['generation']!=row['generation'] or math.dist(after['bodies']['pelvis']['position_m'],base['position_m'])>.01:raise RuntimeError('STATIC_ROUTE_SOURCE_POSE_CHANGED')
    return dict(result=str(result),sha256=digest(result),source_snapshot=str(snapshot/'native_latest.json'),classification='MEASURED_CURRENT_STATE_STATIC_SCREEN_NOT_DYNAMIC_PASS')
