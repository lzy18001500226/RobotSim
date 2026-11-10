"""Restore native standing height/waist ownership while the existing owner holds."""
from pathlib import Path
import inspect,json,math,time

def receive_native_waist(self,message):
    import copy,os,time,json,math
    if len(message.joints)!=3 or {j.name for j in message.joints}!=set(WAIST):raise RuntimeError("MC_WAIST_GROUP_CHANGED")
    if any(not all(math.isfinite(float(getattr(j,k))) for k in ("position","velocity","effort","stiffness","damping")) or j.stiffness<=0. or j.damping<=0. for j in message.joints):raise RuntimeError("MC_WAIST_SOURCE_INVALID")
    self.mc_waist_message=copy.deepcopy(message);self.mc_waist_receipt=time.monotonic()
    if getattr(self,"mc_waist_blend_started",None) is None or getattr(self,"mc_waist_abort_frozen",False):return
    source={j.name:float(j.position) for j in message.joints}
    fraction=min(1.,(time.monotonic()-self.mc_waist_blend_started)/4.)
    smooth=fraction*fraction*(3.-2.*fraction)
    for joint in message.joints:
        lo,hi,speed=LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][joint.name]
        if not lo+.001<=joint.position<=hi-.001:raise RuntimeError("MC_WAIST_SOURCE_TARGET_RANGE")
        joint.position=self.mc_waist_blend_from[joint.name]+smooth*(joint.position-self.mc_waist_blend_from[joint.name])
    self.waist_pub.publish(message);self.last_waist_message=message;self.mc_waist_relay_count=getattr(self,"mc_waist_relay_count",0)+1
    now=time.monotonic()
    if now-getattr(self,"mc_waist_report_at",0.)>=.10:
        value=dict(schema="robotsim.native-waist-handoff-feedback.v1",owner_pid=os.getpid(),generation=self.native_generation,host_monotonic_ns=time.monotonic_ns(),source_receipt_monotonic_s=self.mc_waist_receipt,source_stamp_ns=int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec),source_positions_rad=source,published_positions_rad={j.name:float(j.position) for j in message.joints},fraction=fraction,relay_count=self.mc_waist_relay_count,gains="UNCHANGED_CURRENT_MC")
        with open("/workspace/evidence/native_waist_handoff_feedback.tmp","w") as f:json.dump(value,f)
        os.replace("/workspace/evidence/native_waist_handoff_feedback.tmp","/workspace/evidence/native_waist_handoff_feedback.json");self.mc_waist_report_at=now

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
        self.mc_waist_blend_from={j.name:float(j.position) for j in self.last_waist_message.joints}
        self.mc_waist_blend_started=time.monotonic()
    if getattr(self,"mc_waist_blend_started",None) is not None and time.monotonic()-self.mc_waist_receipt>.20:raise RuntimeError("MC_WAIST_RELAY_STALE")

def extend_existing_owner(script):
    """Same arm/hand/waist publishers; only the waist source changes after proof."""
    def replace(old,new):
        nonlocal script
        if script.count(old)!=1:raise ValueError("postured restoration owner anchor:"+old[:60])
        script=script.replace(old,new)
    helper=inspect.getsource(receive_native_waist)+"\n"+inspect.getsource(request_native_waist_handoff)
    replace("class Plan(Node):",helper+"\nclass Plan(Node):")
    anchor="self.waist_pub=self.create_publisher("
    lines=script.splitlines();index=next(i for i,l in enumerate(lines) if l.strip().startswith(anchor))
    lines.insert(index+1,"   self.create_subscription(JointCommandArray,\'/robotsim/issue46/mc/waist/command\',lambda m:receive_native_waist(self,m),q)")
    script="\n".join(lines)+"\n"
    replace("   if WAIST:\n    w=JointCommandArray();w.header=m.header","   if stage==\'CARRY_HOLD\':request_native_waist_handoff(self)\n   if WAIST and getattr(self,\'mc_waist_blend_started\',None) is None:\n    w=JointCommandArray();w.header=m.header")
    replace("    self.waist_pub.publish(w)","    self.waist_pub.publish(w);self.last_waist_message=w")
    replace("self.last_published_arm_target={name:float(target[name]) for name in LEFT+RIGHT}","self.last_published_arm_target={name:float(target[name]) for name in CONTROL}")
    replace("def carry_abort_last_command_hold(n, carry_error):","def carry_abort_last_command_hold(n, carry_error):"+chr(10)+" n.mc_waist_abort_frozen=True")
    replace("   n.pub.publish(n.last_arm_message);n.hand_pub.publish(n.last_hand_message)","   n.pub.publish(n.last_arm_message);n.hand_pub.publish(n.last_hand_message)\n   if getattr(n,\'last_waist_message\',None) is not None:n.waist_pub.publish(n.last_waist_message)")
    compile(script,"existing_owner_postured_stand_restore","exec")
    return script

def restore_postured_hold(run,container,environment,output,ready,result):
    """No locomotion publisher/action; restore through the installed body interface."""
    import os
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    proof=json.loads((output/"body_crouch_pickup_result.json").read_text())
    expected_mc=proof["body_subscriber"]["name"]
    if not proof.get("pass_result") or proof.get("generation")!=ready["generation"]:raise RuntimeError("RESTORE_NEEDS_SAME_EPISODE_QUALIFIED_LOWERING")
    observed=run(["docker","exec",container,"bash","-lc",environment+"timeout 35s python3 /workspace/evidence/body_restore_carry.py"],timeout_s=38.,check=False)
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
        if not health.get("healthy") or health.get("pid")!=ready["pid"] or not 0.<=(time.monotonic_ns()-health["host_monotonic_ns"])*1e-9<=.20:raise RuntimeError("RESTORE_GRIP_OWNER_NOT_FRESH:"+json.dumps({k:health.get(k) for k in ("mode","healthy","failure","pid","generation")}))
        path=output/"native_waist_handoff_feedback.json"
        if path.exists():
            feedback=json.loads(path.read_text())
            if feedback.get("owner_pid")==ready["pid"] and feedback.get("generation")==ready["generation"] and feedback["fraction"]>=1. and feedback["relay_count"]>=2 and 0.<=(time.monotonic_ns()-feedback["host_monotonic_ns"])*1e-9<=.20:break
        time.sleep(.02)
    else:raise RuntimeError("NATIVE_MC_WAIST_HANDOFF_UNCONFIRMED")
    observed=run(["docker","exec",container,"bash","-lc",environment+"timeout 15s python3 /workspace/evidence/body_after_restore_observer.py"],timeout_s=18.,check=False)
    pg.write_new_text(output/"body_after_restore_observer.stdout",observed.stdout)
    path=output/"body_restore_final_result.json";stop=json.loads(path.read_text()) if path.exists() else {}
    if observed.returncode or not stop.get("pass_result"):raise RuntimeError("RESTORED_NATIVE_STAND_COMPOSITE_NOT_QUALIFIED:"+str(stop.get("failure",observed.stdout[-1000:])))
    value=dict(classification="SIMULATION_ONLY_REAL_GRASP_NATIVE_HEIGHT_AND_WAIST_RESTORED",pass_result=True,body_restore=body,native_waist_handoff=feedback,standing_stop=stop,velocity_publishers_created=0,locomotion_actions_sent=0,walking_qualification="NOT_RUN")
    pg.write_new_text(output/"postured_standing_restore_result.json",json.dumps(value,indent=2)+"\n")
    return value
