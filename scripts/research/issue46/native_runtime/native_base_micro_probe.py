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
        if not math.isfinite(j["qvel"]) or abs(j["qvel"]) > speed:
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


def build_script(direction, duration_s, joint_limits):
    if direction not in ("forward", "backward", "left", "right"):
        raise ValueError("unknown body-frame micro direction")
    if not .4 <= duration_s <= 1.:
        raise ValueError("micro pulse must be .4..1 seconds")
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import locomotion_dock_script
    base = locomotion_dock_script(world_start_xy=[-1.5, -1.5],
        world_start_yaw_rad=0., dock_world_xy=[-1.5, -1.5],
        dock_world_yaw_rad=0., dock_position_only=True,
        refine_world_xy=None, refine_world_yaw_rad=None,
        refine_position_only=False, dock_position_tolerance_m=.10,
        refine_position_tolerance_m=.15, timeout_s=30.,
        use_cmd_vel_bridge=True, native_pose_feedback=True)
    base = base.split("rclpy.init();n=Dock();registered=False")[0]
    plan = dict(direction=direction, duration_s=duration_s, joint_limits=joint_limits)
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
            self.create_subscription(McLocomotionVelocity,"/aima/mc/locomotion/velocity",lambda m:self.bridge_commands.append({"t":time.monotonic(),"raw":raw(m),"evidence":"DDS_OBSERVED_BRIDGE_OUTPUT_NOT_MC_RECEIPT_ACK"}),QOS)
            self.create_subscription(McCommonState, "/aima/mc/common/state",
                                     lambda m: self.store("common", m), QOS)
            self.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command",
                                     lambda m: self.store("leg_command", m), QOS)
            self.get_mode = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")

        def store(self, name, m):
            stamp = int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
            if advancing_stamp(self.unique, name, stamp, time.monotonic()):
                self.messages[name] = raw(m)

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

        def pump(self):
            rclpy.spin_once(self, timeout_sec=.001)
            self.refresh_native_pose()
            self.latest_row=json.loads(Path("/workspace/evidence/native_latest.json").read_text())
            now=time.monotonic()
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
            now=time.monotonic(); r=self.latest_row
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
                     native_generation=r["generation"])
            return out

        def graph(self):
            pubs=self.get_publishers_info_by_topic("/aima/mc/locomotion/velocity")
            subs=self.get_subscriptions_info_by_topic("/aima/mc/locomotion/velocity")
            rawpubs=self.get_publishers_info_by_topic("/cmd_vel")
            if len(pubs)!=1 or pubs[0].node_name!="cmd_vel_bridge" or len(subs)!=2 or len(rawpubs)!=1 or rawpubs[0].node_name!=self.get_name():
                raise RuntimeError("VELOCITY_GRAPH_NOT_UNIQUE:"+str(([p.node_name for p in pubs],[s.node_name for s in subs],[p.node_name for p in rawpubs])))
            # Exact configured-name + process PID contract from reviewed Issue80.
            candidates=[]
            for d in Path("/proc").iterdir():
                if d.name.isdigit():
                    try:
                        exe=os.readlink(d/"exe")
                        cmd=(d/"cmdline").read_bytes().replace(b"\\0",b" ").decode()
                    except (OSError,UnicodeError):continue
                    if Path(exe).name=="aima-mc-app-main":
                        candidates.append((int(d.name),exe,cmd))
            if len(candidates)!=1:
                raise RuntimeError("MC_PROCESS_NOT_UNIQUE")
            mcsubs=[q for q in subs if q.node_name=="mc_ros2_node"+str(candidates[0][0])]
            ownsubs=[q for q in subs if q.node_name==self.get_name()]
            if len(mcsubs)!=1 or len(ownsubs)!=1:
                raise RuntimeError("MC_SUBSCRIBER_PROCESS_IDENTITY")
            subs=mcsubs
            if subs[0].node_namespace!="/" or pubs[0].node_namespace!="/":
                raise RuntimeError("MC_SUBSCRIBER_PROCESS_IDENTITY")
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

        def observe(self, seconds, start=None, baseline=None, require_stop=False):
            until=time.monotonic()+seconds;last=None;rows=[];qualified=None
            while rclpy.ok() and time.monotonic()<until:
                tick=time.monotonic();self.publish();self.pump()
                row=self.sample(start)
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
    rclpy.init();n=Micro();result=dict(classification="SIMULATION_ONLY_NATIVE_MICRO_CAPABILITY",
        direction=MICRO["direction"],duration_s=MICRO["duration_s"],zero_receipt_ack="UNAVAILABLE",
        rearm_permitted=False,pass_result=False)
    if smoke_only:
        print(json.dumps({"constructed":True,"publisher_topics":[p.topic_name for p in n.publishers],"subscription_topics":[s.topic_name for s in n.subscriptions],"client_services":[c.srv_name for c in n.clients],"motion_publish_calls":len(n.command_log)}),flush=True)
        n.destroy_node();rclpy.shutdown();return
    watchdog=None;baseline=None;motion_started=False
    try:
        deadline=time.monotonic()+3.
        while time.monotonic()<deadline:
            n.publish();n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:
                break
        n.graph();n.observe(.12);n.graph()
        baseline=n.sample()["body_status"]
        result["pre_motion_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        start=n.sample()["pose"];result["start_pose"]=start
        n.graph()
        velocity={"forward":(.2,0.),"backward":(-.2,0.),"left":(0.,.2),"right":(0.,-.2)}[MICRO["direction"]]
        with n.lock:n.moving=True
        began=time.monotonic();motion_started=True
        n.events.append(dict(event="NONZERO_BEGIN",t=began))
        watchdog=threading.Timer(MICRO["duration_s"],n.deadline_zero);watchdog.daemon=True;watchdog.start()
        while time.monotonic()-began<MICRO["duration_s"] and n.moving:
            tick=time.monotonic();n.pump();row=n.sample(start);n.trace.append(row)
            if row["body_status"]!=baseline:raise RuntimeError("BODY_STATUS_CHANGED")
            n.publish(*velocity);time.sleep(max(0.,tick+.02-time.monotonic()))
        # Initiate zero BEFORE any blocking readback or post-pulse inspection.
        n.deadline_zero();n.stop_started=True
        if watchdog:watchdog.cancel()
        n.observe(1.05,start=start,baseline=baseline)
        n.stand_once()
        result["post_stop_composite"]=n.observe(8.,start=start,baseline=baseline,require_stop=True)
        end=n.sample(start)["pose"];result["end_pose"]=end
        dx,dy=end["x"]-start["x"],end["y"]-start["y"]
        c,s=math.cos(start["yaw"]),math.sin(start["yaw"])
        forward,lateral=c*dx+s*dy,-s*dx+c*dy
        progress={"forward":forward,"backward":-forward,"left":lateral,"right":-lateral}[MICRO["direction"]]
        result.update(body_forward_m=forward,body_lateral_m=lateral,
                      yaw_change_rad=(end["yaw"]-start["yaw"]+math.pi)%(2*math.pi)-math.pi)
        acquired=any(r["common_source"]["name"]=="robotsim_official_cmd_vel" and
                     r["common_mode_desc"]=="LOCOMOTION_DEFAULT" for r in n.trace)
        result["source_and_ld_observed"]=acquired
        result["pass_result"]=result["post_stop_composite"]["pass"] and progress>=.005 and acquired
        result["failure"]=None if result["pass_result"] else "NO_VERIFIED_SIGNED_MICRO_MOVEMENT_OR_HANDSHAKE"
    except Exception as exc:
        n.deadline_zero();result["failure"]=str(exc)
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
        result.update(events=n.events,command_log=n.command_log,bridge_commands=n.bridge_commands,trace=n.trace,
                      nonzero_started=motion_started,watchdog_limitation="IN_PROCESS_TIMER_PLUS_BRIDGE_STALE_INPUT_WATCHDOG",
                      stop_contract_revision=contract.__doc__)
        Path("/workspace/evidence/native_micro_probe_result.json").write_text(json.dumps(result,indent=2)+"\n")
        n.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in result.items() if k not in ("trace","command_log","bridge_commands")}),flush=True)
