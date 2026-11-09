"""Opt-in unloaded-arm following in the existing grip owner.

The MC remains the producer of native arm targets. There is still exactly one
public arm publisher. The right grip/IK and every physical guard are unchanged.
"""
import math
import json
import time
import os
from scripts.research.issue46.native_runtime.native_base_micro_probe import observed_clock_step


def validate_native_arm_packet(packet, limits, now_ns, wall_ns):
    if not 0 <= now_ns-packet["received_ns"] <= 200000000 or abs(wall_ns-packet["stamp_ns"])>200000000:
        raise RuntimeError("STALE_NATIVE_ARM_POLICY_PACKET")
    joints=packet["joints"]
    if len(joints)!=14 or set(joints)!=set(limits):
        raise RuntimeError("INCOMPLETE_NATIVE_ARM_POLICY_GROUP")
    active=all(j["stiffness"]>0 and j["damping"]>0 for j in joints.values())
    inactive=all(j["stiffness"]==0 and j["damping"]==0 and j["effort"]==0 for j in joints.values())
    if not (active or inactive):raise RuntimeError("MIXED_NATIVE_ARM_POLICY_MODE")
    for name,j in joints.items():
        if not all(math.isfinite(float(v)) for v in j.values()):
            raise RuntimeError("NONFINITE_NATIVE_ARM_POLICY")
        lo,hi,speed=limits[name]
        if j["stiffness"]<0 or j["damping"]<0 or active and (not lo-.001<=j["position"]<=hi+.001 or abs(j["velocity"])>speed):
            raise RuntimeError("NATIVE_ARM_POLICY_SOURCE_LIMIT:"+name)
    return active


def on_native_carry_arm(self, message):
    before=time.monotonic_ns();wall=time.time_ns();after=time.monotonic_ns()
    offset,step=observed_clock_step(getattr(self,"native_policy_offset",None),before,wall,after)
    self.native_policy_offset=offset
    if step is not None:
        self.native_policy_stamp=0;self.native_policy_packet=None
        self.native_left_last=None;self.native_left_idle=None
        self.native_policy_epoch=getattr(self,"native_policy_epoch",0)+1
        print(json.dumps({"native_arm_source_clock_epoch":self.native_policy_epoch,"offset_delta_ns":step,"old_packet_discarded":True}),flush=True)
    stamp=int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec)
    if stamp<=getattr(self,"native_policy_stamp",0):return
    if len(message.joints)!=14 or len({j.name for j in message.joints})!=14:
        raise RuntimeError("DUPLICATE_NATIVE_ARM_POLICY_NAMES")
    packet=dict(stamp_ns=stamp,received_ns=time.monotonic_ns(),
                joints={j.name:{f:float(getattr(j,f)) for f in ("position","velocity","effort","stiffness","damping")} for j in message.joints})
    limits={n:LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][n] for n in LEFT+RIGHT}
    try:validate_native_arm_packet(packet,limits,time.monotonic_ns(),time.time_ns())
    except RuntimeError as exc:
        if str(exc)=="STALE_NATIVE_ARM_POLICY_PACKET":return
        raise
    self.native_policy_packet=packet;self.native_policy_stamp=stamp


def select_native_carry_left(self, stage):
    from pathlib import Path
    marker_path=Path("/workspace/evidence/carry_unloaded_arm_follow.json")
    if stage!="CARRY_HOLD" or self.lift_phase.get("placement_active") or not marker_path.exists():return None
    marker=json.loads(marker_path.read_text())
    if (marker.get("owner_pid")!=os.getpid() or marker.get("generation")!=self.native_generation or
        marker.get("turn_and_composite_stop_pass") is not True or
        not marker["created_ns"]<=time.monotonic_ns()<marker["deadline_ns"]):
        raise RuntimeError("UNQUALIFIED_NATIVE_LEFT_HANDOFF")
    private=self.get_publishers_info_by_topic("/robotsim/issue46/mc/arm/command")
    public=self.get_publishers_info_by_topic("/aima/hal/joint/arm/command")
    source=json.loads(Path("/workspace/evidence/enable_native_arm_relay").read_text())
    if not (len(private)==1 and private[0].node_name==source["expected_mc_node"] and
            private[0].node_namespace=="/" and private[0].topic_type=="aimdk_msgs/msg/JointCommandArray" and
            len(public)==1 and public[0].node_name==self.get_name() and public[0].node_namespace=="/" and
            public[0].topic_type=="aimdk_msgs/msg/JointCommandArray"):
        raise RuntimeError("NATIVE_LEFT_SOURCE_IDENTITY")
    packet=getattr(self,"native_policy_packet",None)
    if packet is None:raise RuntimeError("MISSING_NATIVE_ARM_POLICY_PACKET")
    limits={n:LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][n] for n in LEFT+RIGHT}
    active=validate_native_arm_packet(packet,limits,time.monotonic_ns(),time.time_ns())
    if active:
        self.native_left_idle=None
        selected={}
        previous=getattr(self,"native_left_last",None) or {n:float(self.arm[n]) for n in LEFT}
        for name in LEFT:
            selected[name]=dict(packet["joints"][name])
            step=.02*min(.5,float(limits[name][2]))
            selected[name]["position"]=previous[name]+max(-step,min(step,selected[name]["position"]-previous[name]))
        self.native_left_last={n:j["position"] for n,j in selected.items()}
    else:
        if getattr(self,"native_left_idle",None) is None:
            self.native_left_idle={n:float(self.arm[n]) for n in LEFT}
        self.native_left_last=None
        selected={n:dict(position=self.native_left_idle[n],velocity=0.,effort=float(self.native_row["joints"][n]["bias_force"]),stiffness=GAINS[n][0],damping=GAINS[n][1]) for n in LEFT}
    if self.last_trace_at is None or time.monotonic()-self.last_trace_at>=.05:
        print(json.dumps({"native_unloaded_arm_follow":dict(active_native_gains=active,packet=packet,selected=selected,maximum_position_slew_rad_s=.5,right_arm_ik_unchanged=True)}),flush=True)
    return selected


def extend_unloaded_arm_follow(script):
    import inspect
    helpers="\n".join(inspect.getsource(f) for f in (observed_clock_step,validate_native_arm_packet,on_native_carry_arm,select_native_carry_left))
    anchor="class Plan(Node):"
    if script.count(anchor)!=1:raise ValueError("native left owner anchor")
    script=script.replace(anchor,helpers+"\n"+anchor)
    anchor="  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,q)"
    if script.count(anchor)!=1:raise ValueError("native left subscription anchor")
    script=script.replace(anchor,anchor+"\n  self.create_subscription(JointCommandArray,'/robotsim/issue46/mc/arm/command',lambda m:on_native_carry_arm(self,m),q)")
    lines=script.splitlines();out=[];in_send=False;replaced=0
    for line in lines:
        if line.strip().startswith("def send("):in_send=True
        if line.startswith("rclpy.init") or line.startswith(" import gc"):in_send=False
        if in_send and line.strip()=="for name in LEFT+RIGHT:":
            out.append(" "* (len(line)-len(line.lstrip()))+"native_left=select_native_carry_left(self,stage)")
        if in_send and line.strip().startswith("x=JointCommand();x.name=name;") and line.rstrip().endswith(";m.joints.append(x)"):
            indent=" "*(len(line)-len(line.lstrip()))
            line=line.rsplit(";m.joints.append(x)",1)[0]
            out.append(line)
            out.extend([indent+"if native_left and name in LEFT:",
                        indent+" for field,value in native_left[name].items():setattr(x,field,float(value))",
                        indent+" target[name]=float(x.position)",
                        indent+"m.joints.append(x)"])
            replaced+=1
        else:out.append(line)
    if replaced!=1:raise ValueError("native left command ownership anchor:"+str(replaced))
    result="\n".join(out)+"\n";compile(result,"existing_owner_native_unloaded_arm","exec")
    return result
