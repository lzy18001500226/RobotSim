"""Bounded body-frame micro-step using the existing native Dock transport.

One pulse only. Unchanged Issue-80 composite stop contract; no rearm on failure.
Official cmd_vel bridge owns the single MC velocity publisher. Native ground
truth is explicit diagnostic feedback, never described as LiDAR localization.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def advancing_stamp(store, name, stamp, now):
    """Duplicates/regressions never refresh the age of a qualified sample."""
    old = store.get(name)
    if stamp <= 0 or old and stamp <= old[0]:
        return False
    store[name] = (stamp, now)
    return True


def drain_ready_callbacks(node, rclpy_module, *, budget_s=.004, max_callbacks=128):
    """Drain queued state callbacks within a bounded CPU slice, without sleeping."""
    began=time.monotonic();count=0
    while count<max_callbacks and time.monotonic()-began<budget_s:
        rclpy_module.spin_once(node,timeout_sec=0.)
        count+=1
    return count

def native_pose(row):
    body = row["bodies"]["pelvis"]
    x, y, z = body["position_m"]
    R = body["rotation"]
    velocity = body["origin_velocity_world_angular_linear"]
    values = [x, y, z, *R, *velocity]
    if len(velocity) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid native body-origin velocity or pose")
    return dict(x=x, y=y, z=z, roll=math.atan2(R[7], R[8]),
                pitch=math.asin(max(-1., min(1., -R[6]))),
                yaw=math.atan2(R[3], R[0]), vx=velocity[3],
                vy=velocity[4], vz=velocity[5],
                frame_id="native_scene_world", child_frame_id="pelvis_origin")


def physical_fault(row, pose, limits, start=None, direction=None):
    if pose["z"] < .45 or math.hypot(pose["roll"], pose["pitch"]) > math.radians(15):
        return "POSTURE"
    if math.hypot(pose["vx"], pose["vy"]) > .20:
        return "PLANAR_SPEED"
    for name, (lo, hi, speed) in limits.items():
        j = row["joints"][name]
        if not lo-.001 <= j["q"] <= hi+.001:
            return "JOINT_LIMIT:" + name
        if not math.isfinite(j["qvel"]) or abs(j["qvel"]) > speed or j.get("max_abs_qvel_since_previous_sample",abs(j["qvel"]))>speed:
            return "SOURCE_JOINT_SPEED:" + name
    if start is not None:
        dx, dy = pose["x"]-start["x"], pose["y"]-start["y"]
        if math.hypot(dx, dy) > .18:
            return "MICRO_WORKSPACE"
        c, s = math.cos(start["yaw"]), math.sin(start["yaw"])
        forward, lateral = c*dx+s*dy, -s*dx+c*dy
        if abs(lateral if direction in ("forward", "backward") else forward) > .05:
            return "MICRO_CROSS_TRACK"
        if abs((pose["yaw"]-start["yaw"]+math.pi)%(2*math.pi)-math.pi) > .12:
            return "MICRO_YAW_DRIFT"
    return None


def build_steady_bridge_source(source):
    """Three hash-bound scheduling edits; preserve ROS headers and all MC fields."""
    import hashlib
    if hashlib.sha256(source.encode()).hexdigest() != "9996c19790ef3de720c4d0f9fa2a8a75949f95a2d5800db50cfd3c7587562cc5":
        raise ValueError("unsupported official bridge source hash")
    changes=(
        ("self.timer = self.create_timer(timer_period, self.publish_velocity)",
         "self.timer = self.create_timer(timer_period, self.publish_velocity, clock=_robotsim_Clock(clock_type=_robotsim_ClockType.STEADY_TIME))"),
        ("self.last_cmd_time = self.get_clock().now()",
         "self.last_cmd_time = _robotsim_steady_time.monotonic()"),
        ("age = (now - self.last_cmd_time).nanoseconds / 1e9",
         "age = _robotsim_steady_time.monotonic() - self.last_cmd_time"),
    )
    for old,new in changes:
        if source.count(old)!=1:raise ValueError("bridge timer adapter pattern mismatch")
        source=source.replace(old,new)
    return ("import time as _robotsim_steady_time\n"
            "from rclpy.clock import Clock as _robotsim_Clock, ClockType as _robotsim_ClockType\n"
            + source)


def observed_clock_step(previous_offset_ns, before_ns, wall_ns, after_ns):
    """Qualify an actual host clock discontinuity, not a late DDS timestamp."""
    if after_ns<before_ns or after_ns-before_ns>1000000:
        return previous_offset_ns, None
    offset=wall_ns-(before_ns+after_ns)//2
    delta=None if previous_offset_ns is None else offset-previous_offset_ns
    return offset, delta if delta is not None and abs(delta)>10000000 else None


def heading_hold_command(current_yaw, initial_yaw):
    """Existing Dock .8 heading feedback, within the qualified .075 rad/s cap."""
    if not all(math.isfinite(x) for x in (current_yaw, initial_yaw)):
        raise ValueError("invalid heading feedback")
    error=(initial_yaw-current_yaw+math.pi)%(2*math.pi)-math.pi
    # The delivered bridge raises tiny nonzero angular inputs to its minimum.
    # Use an explicit 0.02rad hold deadband rather than chattering near zero.
    return 0. if abs(error)<=.02 else max(-.075,min(.075,.8*error))


def micro_displacement(pose,start,direction):
    dx,dy=pose["x"]-start["x"],pose["y"]-start["y"]
    c,s=math.cos(start["yaw"]),math.sin(start["yaw"])
    forward,lateral=c*dx+s*dy,-s*dx+c*dy
    progress={"forward":forward,"backward":-forward,"left":lateral,"right":-lateral}[direction]
    cross=lateral if direction in ("forward","backward") else forward
    return forward,lateral,progress,cross


def build_script(direction, duration_s, joint_limits, *, command_speed=.2,native_arm_relay=False,heading_hold=False,target_distance_m=None):
    if direction not in ("forward", "backward", "left", "right"):
        raise ValueError("unknown body-frame micro direction")
    maximum=1. if target_distance_m is None else 2.
    if target_distance_m is not None and (not math.isfinite(target_distance_m) or not .02<=target_distance_m<=.05 or command_speed not in (.12,.14) or command_speed==.14 and direction!="forward"):
        raise ValueError("targeted cycle requires .02..05m and .12, or the reviewed Run81 forward .14 profile")
    if not .4 <= duration_s <= maximum:
        raise ValueError("micro pulse outside its explicit .4..1s or targeted <=2s profile")
    if not math.isfinite(command_speed) or not .12<=command_speed<=.2:raise ValueError("micro input outside previously validated .12..0.20 envelope")
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import locomotion_dock_script
    base = locomotion_dock_script(world_start_xy=[-1.5, -1.5],
        world_start_yaw_rad=0., dock_world_xy=[-1.5, -1.5],
        dock_world_yaw_rad=0., dock_position_only=True,
        refine_world_xy=None, refine_world_yaw_rad=None,
        refine_position_only=False, dock_position_tolerance_m=.10,
        refine_position_tolerance_m=.15, timeout_s=30.,
        use_cmd_vel_bridge=True, native_pose_feedback=True,native_arm_relay=native_arm_relay)
    base = base.split("rclpy.init();n=Dock();registered=False")[0]
    plan = dict(direction=direction, duration_s=duration_s, joint_limits=joint_limits, command_speed=command_speed, heading_hold=bool(heading_hold),target_distance_m=target_distance_m)
    return base + "\n" + Path(__file__).read_text().replace("from __future__ import annotations","") + "\nMICRO=" + repr(plan) + "\nrun_probe()\n"


def run_probe(smoke_only=False):
    import os
    import re
    import threading
    from rosidl_runtime_py.convert import message_to_ordereddict as raw
    from aimdk_msgs.msg import McCommonState, JointCommandArray
    from aimdk_msgs.srv import GetMcAction
    import native_micro_stop_contract as contract

    class Micro(Dock):
        def __init__(self):
            super().__init__()
            self.unique = {}
            self.messages = {}
            self.pending = {}
            self.readbacks = {}
            self.events = []
            self.trace = []
            self.latest_row = None
            self.stop_started = False
            self.lock = threading.Lock()
            self.moving = False
            self.command_log = []
            self.bridge_commands=[]
            self.feedback_executor = None
            self.feedback_thread = None
            self.feedback_executor_error = None
            self.receipt_diagnostics = {}
            self.telemetry_lock = threading.RLock()
            self.clock_offset_ns = None
            self.source_epoch = 0
            self.pumped_epoch = 0
            self.epoch_changed_at = None
            self.create_subscription(McLocomotionVelocity,"/aima/mc/locomotion/velocity",lambda m:self.bridge_commands.append({"t":time.monotonic(),"raw":raw(m),"evidence":"DDS_OBSERVED_BRIDGE_OUTPUT_NOT_MC_RECEIPT_ACK"}),QOS)
            self.create_subscription(McCommonState, "/aima/mc/common/state",
                                     lambda m: self.store("common", m), QOS)
            self.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command",
                                     lambda m: self.store("leg_command", m), QOS)
            self.get_mode = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")

        def store(self, name, m):
            stamp = int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
            before=time.monotonic_ns();wall=time.time_ns();after=time.monotonic_ns()
            now=after*1e-9
            with self.telemetry_lock:
                offset,step=observed_clock_step(self.clock_offset_ns,before,wall,after)
                self.clock_offset_ns=offset
                if step is not None:
                    self.source_epoch+=1;self.epoch_changed_at=now
                    self.unique.clear();self.messages.clear()
                    moving=self.moving
                    self.events.append(dict(event="HOST_CLOCK_GENERATION_CHANGE",t=now,
                        source_epoch=self.source_epoch,offset_delta_ns=step,while_moving=moving))
                    if moving:
                        self.deadline_zero()
                        self.feedback_fault=self.feedback_fault or dict(gate="CLOCK_STEP_DURING_MOTION")
                diag=self.receipt_diagnostics.setdefault(name,dict(callbacks=0,duplicates=0,
                    regressions=0,rejected_wrong_epoch=0,last_callback_monotonic_s=None,last_received_stamp_ns=None))
                previous=self.unique.get(name)
                diag["callbacks"]+=1;diag["last_callback_monotonic_s"]=now
                diag["last_received_stamp_ns"]=stamp
                # Old queued packets cannot seed a new clock generation.
                if abs(wall-stamp)>200000000:
                    diag["rejected_wrong_epoch"]+=1
                    return
                if previous and stamp<=previous[0]:
                    diag["duplicates" if stamp==previous[0] else "regressions"]+=1
                if advancing_stamp(self.unique,name,stamp,now):
                    self.messages[name]=raw(m)

        def on_leg(self, m):
            super().on_leg(m)
            self.store("leg_state", m)

        def on_imu(self, m):
            super().on_imu(m)
            self.store("imu", m)

        def publish(self, fwd=0., lat=0., ang=0.):
            with self.lock:
                if not self.moving:
                    fwd=lat=ang=0.
                super().publish(fwd, lat, ang)
                self.command_log.append(dict(t=time.monotonic(), forward=fwd, lateral=lat, angular=ang,
                                             evidence="PYTHON_PUBLISH_CALL_ONLY_NO_MC_ACK"))

        def deadline_zero(self):
            with self.lock:
                self.moving=False
            self.publish()
            self.events.append(dict(event="DEADLINE_ZERO_ATTEMPT", t=time.monotonic()))

        def start_feedback(self):
            # This one executor exclusively owns callbacks for the probe node.
            # A 4ms/20ms polling slice can leave DDS/service work unprocessed
            # during the controller's zero/transition phase. Do not relax age
            # checks or change source-stamp qualification to hide that backlog.
            from rclpy.executors import SingleThreadedExecutor
            self.feedback_executor = SingleThreadedExecutor()
            self.feedback_executor.add_node(self)
            def spin():
                try:
                    self.feedback_executor.spin()
                except Exception as exc:
                    self.feedback_executor_error = repr(exc)
            self.feedback_thread = threading.Thread(target=spin,
                name="micro-feedback-executor", daemon=True)
            self.feedback_thread.start()

        def stop_feedback(self):
            if self.feedback_executor is not None:
                self.feedback_executor.shutdown(timeout_sec=1.)
            if self.feedback_thread is not None:
                self.feedback_thread.join(timeout=1.)
                if self.feedback_thread.is_alive():
                    raise RuntimeError("FEEDBACK_EXECUTOR_SHUTDOWN_UNCONFIRMED")
            self.feedback_executor = None

        def pump(self):
            if getattr(self, "feedback_executor", None) is None:
                drain_ready_callbacks(self,rclpy)
            elif self.feedback_executor_error is not None or not self.feedback_thread.is_alive():
                raise RuntimeError("FEEDBACK_EXECUTOR_FAILURE:"+str(self.feedback_executor_error))
            self.refresh_native_pose()
            self.latest_row=json.loads(Path("/workspace/evidence/native_latest.json").read_text())
            now=time.monotonic()
            with self.telemetry_lock:
                if self.pumped_epoch!=self.source_epoch:
                    for future,_ in self.pending.values():future.cancel()
                    self.pending.clear();self.readbacks.clear()
                    self.pumped_epoch=self.source_epoch
            for name, client, req in (
                ("get_mode", self.get_mode, GetMcAction.Request),
                ("source", self.current, GetCurrentInputSource.Request),
            ):
                pending=self.pending.get(name)
                if pending:
                    f, started=pending
                    if f.done():
                        if f.result() is not None:
                            self.readbacks[name]=(now, raw(f.result()))
                        self.pending.pop(name, None)
                    elif now-started>.25:
                        f.cancel()
                        self.events.append(dict(event="READBACK_TIMEOUT", name=name, t=now))
                        self.pending.pop(name, None)
                elif now-self.readbacks.get(name, (0,))[0]>.10 and client.service_is_ready():
                    q=req()
                    q.request.header.stamp=self.get_clock().now().to_msg()
                    self.pending[name]=(client.call_async(q),now)

        def sample(self, start=None):
            with self.telemetry_lock:
                now=time.monotonic(); r=self.latest_row
                if self.epoch_changed_at is not None and not all(k in self.messages for k in ("common","imu","leg_state","leg_command")):
                    if not self.moving and now-self.epoch_changed_at<=.20:
                        raise RuntimeError("CLOCK_GENERATION_PENDING")
                    raise RuntimeError("CRITICAL_NEW_CLOCK_GENERATION_TELEMETRY_MISSING")
                if r is None or now-r["host_monotonic_ns"]*1e-9>.20 or self.feedback_fault:
                    raise RuntimeError("NATIVE_FEEDBACK_STALE_OR_INVALID")
                ages={name:now-self.unique.get(name,(0,0))[1]
                      for name in ("common","imu","leg_state","leg_command")}
                if any(age>.20 for age in ages.values()):
                    raise RuntimeError("CRITICAL_ROS_TELEMETRY_STALE:"+str(ages))
                p=native_pose(r)
                fault=physical_fault(r,p,MICRO["joint_limits"],start,MICRO["direction"])
                if fault: raise RuntimeError(fault)
                cm=self.messages["common"]
                mode=cm["action_info"]["action_desc"];status=int(cm["action_info"]["status"]["value"])
                fsm=int(cm["fsm_state"]["current_state"]);body=int(cm["body_status"]["value"])
                if mode not in ("STAND_DEFAULT","LOCOMOTION_DEFAULT") or status not in (100,200,300) or fsm not in contract.MC_FSM_ACTIVE_STATES:
                    raise RuntimeError("MC_PROTECTION_OR_UNEXPECTED_STATE:"+str((mode,status,fsm,body)))
                leg=self.messages["leg_state"]["joints"]
                targets=self.messages["leg_command"]["joints"]
                names=set(LEG)
                if {j["name"] for j in leg}!=names or {j["name"] for j in targets}!=names:
                    raise RuntimeError("INCOMPLETE_LEG_COVERAGE")
                if any(int(j["error_code"]) for j in leg):
                    raise RuntimeError("LEG_ERROR_CODE")
                gm_time,gm=self.readbacks.get("get_mode",(0,{}))
                ages.update(odom=now-r["host_monotonic_ns"]*1e-9,get_mode=now-gm_time)
                out=dict(sample_monotonic_s=r["host_monotonic_ns"]*1e-9, sim_time_s=r["sim_time_s"],
                         pose=p,leg_states=leg,mc_leg_command=dict(joints=targets),
                         state_age_s=ages, get_mode_desc=gm.get("info",{}).get("action_desc"),
                         get_mode_status_value=gm.get("info",{}).get("status",{}).get("value"),
                         get_mode_header_code=gm.get("header",{}).get("code"),
                         common_mode_desc=mode,common_mode_status_value=status,fsm_state=fsm,
                         body_status=body,common_source=cm["input_source"],
                         source_readback=self.readbacks.get("source"),raw_common=cm,
                         native_generation=r["generation"],source_clock_generation=self.source_epoch)
                return out

        def graph(self):
            pubs=self.get_publishers_info_by_topic("/aima/mc/locomotion/velocity")
            subs=self.get_subscriptions_info_by_topic("/aima/mc/locomotion/velocity")
            rawpubs=self.get_publishers_info_by_topic("/cmd_vel")
            self.events.append(dict(event="GRAPH_SNAPSHOT",t=time.monotonic(),publishers=[dict(name=p.node_name,namespace=p.node_namespace,type=p.topic_type,reliability=int(p.qos_profile.reliability),durability=int(p.qos_profile.durability)) for p in pubs],subscribers=[dict(name=p.node_name,namespace=p.node_namespace,type=p.topic_type,reliability=int(p.qos_profile.reliability),durability=int(p.qos_profile.durability)) for p in subs],raw_publishers=[p.node_name for p in rawpubs]))
            if len(pubs)>1 or len(subs)>2 or len(rawpubs)>1:
                raise RuntimeError("CONFLICTING_VELOCITY_GRAPH")
            if not pubs or len(subs)<2 or not rawpubs:
                raise RuntimeError("GRAPH_DISCOVERY_PENDING")
            if any(q.node_name in ("_NODE_NAME_UNKNOWN_","") or q.node_namespace=="_NODE_NAMESPACE_UNKNOWN_" for q in [*pubs,*subs,*rawpubs]):
                raise RuntimeError("GRAPH_DISCOVERY_PENDING")
            if pubs[0].node_name!="cmd_vel_bridge" or rawpubs[0].node_name!=self.get_name():
                raise RuntimeError("VELOCITY_GRAPH_NOT_UNIQUE:"+str(([p.node_name for p in pubs],[s.node_name for s in subs],[p.node_name for p in rawpubs])))
            # Exact configured-name + process PID contract from reviewed Issue80.
            candidates=[]
            for d in Path("/proc").iterdir():
                if d.name.isdigit():
                    try:
                        exe=os.readlink(d/"exe")
                        cmd=(d/"cmdline").read_bytes().replace(bytes([0]),b" ").decode()
                    except (OSError,UnicodeError):continue
                    if Path(exe).name=="aima-mc-app-main":
                        candidates.append((int(d.name),exe,cmd))
            if len(candidates)!=1:
                raise RuntimeError("MC_PROCESS_NOT_UNIQUE")
            mcsubs=[q for q in subs if q.node_name=="mc_ros2_node"+str(candidates[0][0])]
            ownsubs=[q for q in subs if q.node_name==self.get_name()]
            if len(mcsubs)!=1 or len(ownsubs)!=1:
                raise RuntimeError("MC_SUBSCRIBER_PROCESS_IDENTITY:"+str((candidates,[(q.node_name,q.node_namespace) for q in subs],self.get_name())))
            subs=mcsubs
            if subs[0].node_namespace!="/" or pubs[0].node_namespace!="/":
                raise RuntimeError("MC_SUBSCRIBER_PROCESS_IDENTITY:"+str((candidates,[(q.node_name,q.node_namespace) for q in subs],self.get_name())))
            if "--cfg_file_path=/workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml" not in candidates[0][2]:
                raise RuntimeError("MC_CONFIG_ARGV")
            if candidates[0][1]!="/workspace/raicom2026/mc/bin/aima-mc-app-main":
                raise RuntimeError("MC_EXECUTABLE_PATH")
            if pubs[0].topic_type!="aimdk_msgs/msg/McLocomotionVelocity" or subs[0].topic_type!=pubs[0].topic_type:
                raise RuntimeError("MC_TOPIC_TYPE")
            if pubs[0].qos_profile.reliability!=ReliabilityPolicy.RELIABLE or pubs[0].qos_profile.durability!=DurabilityPolicy.VOLATILE:
                raise RuntimeError("BRIDGE_QOS")
            if subs[0].qos_profile.durability!=DurabilityPolicy.VOLATILE or subs[0].qos_profile.reliability not in (ReliabilityPolicy.BEST_EFFORT,ReliabilityPolicy.RELIABLE):
                raise RuntimeError("MC_SUBSCRIBER_QOS")
            self.events.append(dict(event="UNIQUE_PROVENANCE_GRAPH",t=time.monotonic(),mc=candidates[0],
                                    publisher=pubs[0].node_name,subscriber=subs[0].node_name,
                                    configured_history_depth="KEEP_LAST/10; FastDDS graph history may be UNKNOWN/0"))

        def await_graph(self):
            deadline=time.monotonic()+2.
            while time.monotonic()<deadline:
                self.publish();self.pump()
                try:self.graph()
                except RuntimeError as exc:
                    if str(exc)!="GRAPH_DISCOVERY_PENDING":raise
                else:
                    if self.bridge_commands and time.monotonic()-self.bridge_commands[-1]["t"]<=.20:
                        return
                time.sleep(.01)
            raise RuntimeError("POSITIVE_BRIDGE_GRAPH_AND_FRESH_OUTPUT_NOT_OBSERVED")

        def observe(self, seconds, start=None, baseline=None, require_stop=False):
            until=time.monotonic()+seconds;last=None;rows=[];qualified=None
            while rclpy.ok() and time.monotonic()<until:
                tick=time.monotonic();self.publish();self.pump()
                try:row=self.sample(start)
                except RuntimeError as exc:
                    if str(exc)!="CLOCK_GENERATION_PENDING":raise
                    rows=[];last=None
                    time.sleep(max(0.,tick+.02-time.monotonic()))
                    continue
                if rows and row.get("source_clock_generation",0)!=rows[-1].get("source_clock_generation",0):
                    rows=[];last=None
                if baseline is not None and row["body_status"]!=baseline:
                    raise RuntimeError("BODY_STATUS_CHANGED")
                stamp=row["sample_monotonic_s"]
                if stamp!=last:
                    rows.append(row);self.trace.append(row);last=stamp
                if len(rows)>1:
                    tail=[r for r in rows if stamp-r["sample_monotonic_s"]<=2.000001]
                    if len(tail)>1 and tail[-1]["sample_monotonic_s"]-tail[0]["sample_monotonic_s"]>=1.90:
                        trial=contract.evaluate_stop_window(tail,baseline_body_status=baseline)
                        qualified=trial
                        if require_stop and trial["pass"]:
                            return trial
                time.sleep(max(0.,tick+.02-time.monotonic()))
            if require_stop:raise RuntimeError("COMPOSITE_STOP_NOT_QUALIFIED:"+json.dumps(qualified))
            return qualified

        def stand_once(self):
            self.publish()
            q=SetMcAction.Request();q.header.stamp=self.get_clock().now().to_msg()
            q.source=SOURCE;q.command.action_desc="STAND_DEFAULT"
            if not self.mode.service_is_ready():raise RuntimeError("STAND_SERVICE_NOT_READY")
            f=self.mode.call_async(q);until=time.monotonic()+.75
            self.events.append(dict(event="STAND_REQUEST",t=time.monotonic()))
            while not f.done() and time.monotonic()<until:
                self.publish();self.pump();time.sleep(.015)
            response=raw(f.result()) if f.done() and f.result() is not None else None
            self.events.append(dict(event="STAND_RESPONSE",t=time.monotonic(),raw=response))
            if response is None or response["response"]["header"]["code"]!=0 or response["response"]["status"]["value"]!=1:
                raise RuntimeError("STAND_REQUEST_REJECTED_OR_TIMEOUT")

    contract.verify_mc_fsm_runtime_contract("/workspace/raicom2026/mc/lib")
    rclpy.init();n=Micro();result=dict(classification="SIMULATION_ONLY_TARGETED_MICRO_CYCLE_DIAGNOSTIC" if MICRO.get("target_distance_m") is not None else "SIMULATION_ONLY_NATIVE_MICRO_CAPABILITY",
        direction=MICRO["direction"],duration_s=MICRO["duration_s"],heading_hold=MICRO.get("heading_hold",False),zero_receipt_ack="UNAVAILABLE",
        rearm_permitted=False,pass_result=False)
    if smoke_only:
        n.start_feedback()
        time.sleep(.05)
        healthy = n.feedback_thread.is_alive() and n.feedback_executor_error is None
        n.stop_feedback()
        print(json.dumps({"constructed":True,"publisher_topics":[p.topic_name for p in n.publishers],"subscription_topics":[s.topic_name for s in n.subscriptions],"client_services":[c.srv_name for c in n.clients],"motion_publish_calls":len(n.command_log),"dedicated_feedback_executor_healthy":healthy,"feedback_executor_stopped":not n.feedback_thread.is_alive()}),flush=True)
        n.destroy_node();rclpy.shutdown();return
    watchdog=None;baseline=None;motion_started=False
    try:
        n.start_feedback()
        deadline=time.monotonic()+3.
        while time.monotonic()<deadline:
            n.publish();n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:
                break
        n.await_graph();n.observe(.12);n.graph()
        baseline=n.sample()["body_status"]
        result["pre_motion_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        start=n.sample()["pose"];result["start_pose"]=start
        n.await_graph()
        v=MICRO["command_speed"]
        velocity={"forward":(v,0.),"backward":(-v,0.),"left":(0.,v),"right":(0.,-v)}[MICRO["direction"]]
        bounds=n.messages["common"]["speed_status"]
        requested={"forward":velocity[0],"lateral":velocity[1],"angular":0.}
        if any(not bounds[k+"_bounds"]["min_value"]<=value<=bounds[k+"_bounds"]["max_value"] for k,value in requested.items()):raise RuntimeError("MC_REPORTED_INPUT_BOUNDS")
        if MICRO.get("heading_hold") and not (bounds["angular_bounds"]["min_value"]<=-.075 and bounds["angular_bounds"]["max_value"]>=.075):
            raise RuntimeError("MC_REPORTED_HEADING_INPUT_BOUNDS")
        result["command_input_speed_m_s"]=v
        result["mc_reported_speed_bounds"]=bounds
        with n.lock:n.moving=True
        result["target_distance_m"]=MICRO.get("target_distance_m")
        result["goal_triggered_stop"]=False
        began=time.monotonic();motion_started=True
        n.events.append(dict(event="NONZERO_BEGIN",t=began))
        watchdog=threading.Timer(MICRO["duration_s"],n.deadline_zero);watchdog.daemon=True;watchdog.start()
        while time.monotonic()-began<MICRO["duration_s"] and n.moving:
            tick=time.monotonic();n.pump();row=n.sample(start);n.trace.append(row)
            if row["body_status"]!=baseline:raise RuntimeError("BODY_STATUS_CHANGED")
            if MICRO.get("target_distance_m") is not None:
                _,_,progress,_=micro_displacement(row["pose"],start,MICRO["direction"])
                if progress>=MICRO["target_distance_m"]:
                    result["goal_triggered_stop"]=True
                    result["trigger_progress_m"]=progress
                    result["trigger_elapsed_s"]=time.monotonic()-began
                    break
            angular=heading_hold_command(row["pose"]["yaw"],start["yaw"]) if MICRO.get("heading_hold") else 0.
            command_velocity=velocity
            if MICRO["direction"]=="forward" and v==.14 and angular!=0.:
                command_velocity=(.12,0.)  # exact Run81 straight/arc input envelope
            n.publish(*command_velocity,angular);time.sleep(max(0.,tick+.02-time.monotonic()))
        # Initiate zero BEFORE any blocking readback or post-pulse inspection.
        n.deadline_zero();n.stop_started=True
        if watchdog:watchdog.cancel()
        n.observe(1.05,start=start,baseline=baseline)
        n.stand_once()
        result["post_stop_composite"]=n.observe(8.,start=start,baseline=baseline,require_stop=True)
        end=n.sample(start)["pose"];result["end_pose"]=end
        forward,lateral,progress,cross=micro_displacement(end,start,MICRO["direction"])
        result["signed_axis_progress_m"]=progress
        result["cross_axis_displacement_m"]=cross
        result.update(body_forward_m=forward,body_lateral_m=lateral,
                      yaw_change_rad=(end["yaw"]-start["yaw"]+math.pi)%(2*math.pi)-math.pi)
        acquired=any(r["common_source"]["name"]=="robotsim_official_cmd_vel" and
                     r["common_mode_desc"]=="LOCOMOTION_DEFAULT" for r in n.trace)
        result["source_and_ld_observed"]=acquired
        result["pass_result"]=result["post_stop_composite"]["pass"] and progress>=.005 and acquired
        result["failure"]=None if result["pass_result"] else ("SOURCE_AND_LD_NOT_OBSERVED" if not acquired else "NO_CORRECT_SIGNED_MICRO_MOVEMENT")
        if MICRO.get("target_distance_m") is not None:
            error=math.hypot(progress-MICRO["target_distance_m"],cross)
            result["final_position_error_m"]=error
            result["position_acceptance_tolerance_m"]=.020
            result["pass_result"]=result["pass_result"] and result["goal_triggered_stop"] and error<=.020
            if not result["pass_result"] and result["failure"] is None:
                result["failure"]="TARGETED_MICRO_POSITION_OR_GOAL_STOP_NOT_QUALIFIED"
    except Exception as exc:
        n.deadline_zero();result["abort_native_sample"]=n.latest_row;result["failure"]=str(exc);result["available_state_topics"]=list(n.messages);result["last_unique_receipts"]=n.unique
    finally:
        # Failed readback never suppresses zeros. Native bridge also has an
        # independent 0.20s stale-input timeout. Timer is same-process only.
        n.deadline_zero()
        if watchdog:watchdog.cancel()
        end=time.monotonic()+1.05
        while rclpy.ok() and time.monotonic()<end:
            n.publish()
            try:n.pump()
            except Exception as e:n.events.append(dict(event="CLEANUP_OBSERVE_ERROR",detail=str(e)))
            time.sleep(.015)
        try:n.stop_feedback()
        except Exception as exc:
            result["feedback_cleanup_failure"] = str(exc)
            result["pass_result"] = False
        result["source_receipt_diagnostics"] = n.receipt_diagnostics
        result["source_clock_generations"] = n.source_epoch
        result.update(events=n.events,command_log=n.command_log,bridge_commands=n.bridge_commands,trace=n.trace,
                      nonzero_started=motion_started,watchdog_limitation="IN_PROCESS_TIMER_PLUS_BRIDGE_STALE_INPUT_WATCHDOG",
                      stop_contract_revision=contract.__doc__)
        Path("/workspace/evidence/native_micro_probe_result.json").write_text(json.dumps(result,indent=2)+"\n")
        n.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in result.items() if k not in ("trace","command_log","bridge_commands","abort_native_sample")}),flush=True)
