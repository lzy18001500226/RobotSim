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


def physical_fault(row, pose, limits, start=None, direction=None,robot_bodies=None,allowed_contact_pairs=None,planar_speed_limit=.20):
    if planar_speed_limit not in (.20,.50):raise ValueError("unsupported explicit physical speed profile")
    if pose["z"] < .45 or math.hypot(pose["roll"], pose["pitch"]) > math.radians(15):
        return "POSTURE"
    if math.hypot(pose["vx"], pose["vy"]) > planar_speed_limit:
        return "PLANAR_SPEED"
    for name, (lo, hi, speed) in limits.items():
        j = row["joints"][name]
        if not lo-.001 <= j["q"] <= hi+.001:
            return "JOINT_LIMIT:" + name
        if not math.isfinite(j["qvel"]) or abs(j["qvel"]) > speed or j.get("max_abs_qvel_since_previous_sample",abs(j["qvel"]))>speed:
            return "SOURCE_JOINT_SPEED:" + name
    allowed={frozenset(p) for p in (allowed_contact_pairs or [])}
    approved={frozenset(('test_cylinder',p)) for p in ('R_hand_narrow3_Link','R_hand_wide3_Link')}
    if not allowed.issubset(approved):raise ValueError('only intended distal-pad bottle contact may be allowed')
    if robot_bodies:
        if (row.get("contact_capture_scope")!="ALL_COMPILED_GEOM_CONTACTS" or
                not isinstance(row.get("contacts"),list) or
                len(row["contacts"])!=row.get("active_contact_count")):
            return "INCOMPLETE_NATIVE_CONTACT_CAPTURE"
        for contact in row["contacts"]:
            if contact["efc_address"]<0 or contact["distance_m"]>=-.001 or contact["local_force_torque"][0]<=.1:continue
            a,b=contact["body1"],contact["body2"];ra,rb=a in robot_bodies,b in robot_bodies
            if frozenset((a,b)) in allowed:continue
            if ra and rb:return "ACTIVE_ROBOT_SELF_COLLISION:"+a+":"+b
            if ra or rb:
                robot,other=(a,b) if ra else (b,a)
                if other=="world" and robot in ("left_ankle_pitch_link","left_ankle_roll_link","right_ankle_pitch_link","right_ankle_roll_link"):continue
                return "ROBOT_ENVIRONMENT_COLLISION:"+robot+":"+other
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


def carry_transit_speed_limit(record, stamp_ns, generation, owner_pid):
    """Approved phase-only limit; timestamps outside the permit retain .20."""
    if record is None:return .20
    if (record.get("schema")!="robotsim.carry-transit050.v1" or record.get("owner_pid")!=owner_pid
            or record.get("generation")!=generation):
        raise ValueError("CARRY_TRANSIT_PERMIT_IDENTITY")
    names=("created_ns","command_deadline_ns","absolute_deadline_ns","route_pid","route_start_ticks")
    if any(type(record.get(k)) is not int or record[k]<=0 for k in names):
        raise ValueError("CARRY_TRANSIT_PERMIT_FIELDS")
    created,command,end=(record[k] for k in names[:3])
    brake,revoked=record.get("braking_ns"),record.get("revoked_ns")
    if (not 5_000_000_000<=command-created<=90_000_000_000 or end!=command+15_000_000_000
            or brake is not None and (type(brake) is not int or not created<=brake<=end)
            or revoked is not None and (type(revoked) is not int or revoked<created
                                       or brake is not None and revoked<brake)):
        raise ValueError("CARRY_TRANSIT_PERMIT_BOUND")
    if stamp_ns<created or revoked is not None and stamp_ns>=revoked:return .20
    expiry=min(end,brake+15_000_000_000) if brake is not None and stamp_ns>=brake else command
    return .50 if stamp_ns<=expiry else .20


def current_carry_transit_speed_limit(evidence, row, owner_pid):
    """Only the live PID/start-token verified navigation process can permit .50."""
    import os,json,time
    from pathlib import Path
    path=Path(evidence)/"carry_transit_permit.json"
    if not path.exists():return .20
    record=json.loads(path.read_text())
    limit=min(carry_transit_speed_limit(record,time.monotonic_ns(),row["generation"],owner_pid),
              carry_transit_speed_limit(record,row["host_monotonic_ns"],row["generation"],owner_pid))
    if limit==.20:return limit
    proc=Path("/proc")/str(record["route_pid"])
    argv=(proc/"cmdline").read_bytes().split(bytes([0]))
    ticks=int((proc/"stat").read_text().rsplit(") ",1)[1].split()[19])
    if (b"/workspace/evidence/native_carry_route.py" not in argv
            or proc.stat().st_uid!=os.getuid() or ticks!=record["route_start_ticks"]):
        raise ValueError("CARRY_TRANSIT_ROUTE_PROCESS_IDENTITY")
    return limit


def carry_transit_phase(record, phase, cfg, native_row, evidence="/workspace/evidence"):
    """One permit, one bounded braking interval, and irreversible revocation."""
    import os,json,time
    from pathlib import Path
    path=Path(evidence)/"carry_transit_permit.json";now=time.monotonic_ns()
    if phase=="TRANSIT":
        if record is not None or path.exists():raise RuntimeError("CARRY_TRANSIT_REARM_FORBIDDEN")
        seconds=float(cfg["timeout_s"])
        if not 5.<=seconds<=90.:raise ValueError("CARRY_TRANSIT_TIMEOUT")
        record=dict(schema="robotsim.carry-transit050.v1",owner_pid=cfg["pid"],
                    route_pid=os.getpid(),route_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19]),
                    generation=native_row["generation"],created_ns=now,
                    command_deadline_ns=now+int(seconds*1e9),absolute_deadline_ns=now+int(seconds*1e9)+15_000_000_000,
                    braking_ns=None,revoked_ns=None)
    elif record is None:return None
    else:
        record=dict(record)
        if phase=="BRAKING":
            if record["revoked_ns"] is not None or record["braking_ns"] is not None:return record
            record["braking_ns"]=now
        elif phase=="REVOKED":
            if record["revoked_ns"] is not None:return record
            record["revoked_ns"]=now
        else:raise ValueError("CARRY_TRANSIT_PHASE")
    carry_transit_speed_limit(record,now,record["generation"],cfg["pid"])
    temporary=path.with_name(path.name+".committing")
    with temporary.open("x") as f:
        json.dump(record,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    return record


def prepare_reviewed_rl_debug_capture(output,runtime):
    """Reuse the old read-only collector and matched ROS overlay, not another controller."""
    import ast,hashlib
    from scripts.agent import path_governance as pg
    base=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-80/foundationwalk-final-left-right-turns-run65-20261022T073901Z")
    library=Path(runtime)/"mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so"
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(library)!="935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931":
        raise ValueError("MC debug ABI is not the reviewed build")
    schema=json.loads((base/"mc_rl_debug_schema.json").read_text())
    mc=next(x for x in schema["results"] if "/mc/lib/" in x["library"])
    if mc["sha256"]!=sha(library) or [x["name"] for x in mc["fields"]]!=["header","obs_vec","action_vec"]:
        raise ValueError("MC debug field contract mismatch")
    setup=base/"mc_debug_overlay/install/setup.bash"
    if not setup.is_file():raise FileNotFoundError(setup)
    capture=base/"issue80_mc_rl_debug_capture.py"
    original=base/"issue80_m1_first_motion.py"
    text=original.read_text();tree=ast.parse(text)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="mc_rl_debug_snapshot")
    stub="from __future__ import annotations\nimport math\nfrom typing import Any, Mapping\n"+ast.get_source_segment(text,helper)+"\n"
    capture_text=capture.read_text()
    old_ready="if len(records) == 1 and len(matching) == 1:"
    new_ready='if len(records) == 1 and len(matching) == 1 and matching[0]["node_name"] == _EXPECTED_MC_NODE and matching[0]["node_namespace"] == "/":'
    if capture_text.count(old_ready)!=1:raise ValueError("reviewed debug readiness pattern differs")
    capture_text=capture_text.replace("from pathlib import Path","from pathlib import Path\nimport os\n_EXPECTED_MC_NODE=os.environ.get('ROBOTSIM_MC_DEBUG_EXPECTED_NODE')",1).replace(old_ready,new_ready,1)
    capture_text=capture_text.replace('"depth": int(qos.depth),','"depth": int(qos.depth), "endpoint_gid": list(endpoint.endpoint_gid),',1)
    pg.write_new_text(Path(output)/capture.name,capture_text)
    pg.write_new_text(Path(output)/original.name,stub)
    return dict(classification="REUSED_READ_ONLY_ISSUE80_MC_RL_DEBUG_CAPTURE",
        source_collector=str(capture),source_collector_sha256=sha(capture),
        staged_collector_sha256=hashlib.sha256(capture_text.encode()).hexdigest(),
        readiness_only_delta="wait for exact MC PID-bound node identity, not initial FastDDS UNKNOWN metadata",
        source_snapshot_helper=str(original),source_snapshot_helper_sha256=sha(original),
        extracted_helper_sha256=hashlib.sha256(stub.encode()).hexdigest(),
        overlay_setup=str(setup),mc_introspection_sha256=sha(library),
        fields=["header","obs_vec","action_vec"],velocity_publishers=0,action_clients=0)


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


def write_durable_milestone(path, value):
    """Atomic, fsynced one-shot milestones; never used in the 50Hz loop."""
    import json,os
    from pathlib import Path
    destination=Path(path)
    if destination.exists():raise FileExistsError('refusing existing milestone:'+str(destination))
    temporary=destination.with_name(destination.name+'.committing')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,allow_nan=False);stream.write(chr(10));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,destination)
    directory=os.open(str(destination.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


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


def build_script(direction, duration_s, joint_limits, *, command_speed=.2,native_arm_relay=False,heading_hold=False,target_distance_m=None,observe_native_arm_policy=False,robot_bodies=None):
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
    plan = dict(direction=direction, duration_s=duration_s, joint_limits=joint_limits, command_speed=command_speed, heading_hold=bool(heading_hold),target_distance_m=target_distance_m,observe_native_arm_policy=bool(observe_native_arm_policy),robot_bodies=robot_bodies)
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
            self.graph_startup_timeout_s=MICRO.get("graph_startup_timeout_s",2.)
            self.lock = threading.Lock()
            self.moving = False
            self.command_log = []
            self.bridge_commands=[]
            self.arm_policy_trace=[]
            self.arm_policy_last_log=0.
            if MICRO.get("observe_native_arm_policy"):
                self.create_subscription(JointCommandArray,"/robotsim/issue46/mc/arm/command",self.on_arm_policy,QOS)
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

        def on_arm_policy(self,m):
            now=time.monotonic()
            if now-self.arm_policy_last_log<.02:return
            if len(m.joints)!=14 or {j.name for j in m.joints}!=set(ARM):
                raise RuntimeError("INVALID_OBSERVED_MC_ARM_GROUP")
            self.arm_policy_last_log=now
            self.arm_policy_trace.append(dict(received_monotonic_s=now,raw=raw(m),
                classification="READ_ONLY_PRIVATE_MC_ARM_OUTPUT_NOT_APPLIED"))

        def on_odom(self,m):
            super().on_odom(m)
            self.store("ros_odom",m)

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
                if self.feedback_fault:
                    raise RuntimeError("NATIVE_FEEDBACK_INVALID:"+json.dumps(self.feedback_fault))
                if r is None or now-r["host_monotonic_ns"]*1e-9>.20:
                    raise RuntimeError("NATIVE_FEEDBACK_STALE_OR_INVALID")
                ages={name:now-self.unique.get(name,(0,0))[1]
                      for name in ("common","imu","leg_state","leg_command")}
                if any(age>.20 for age in ages.values()):
                    raise RuntimeError("CRITICAL_ROS_TELEMETRY_STALE:"+str(ages))
                if MICRO.get("carry"):
                    health=json.loads(Path("/workspace/evidence/carry_heartbeat.json").read_text())
                    if health.get("mode")!="MEASURED_LOAD_BEARING_CARRY_HOLD" or health.get("healthy") is not True or health.get("pid")!=MICRO["carry"]["pid"] or health.get("generation")!=r["generation"] or not 0.<=(time.monotonic_ns()-health["host_monotonic_ns"])*1e-9<=.20 or not 0.<=(time.monotonic_ns()-health["native_source_host_ns"])*1e-9<=.20:
                        raise RuntimeError("CARRY_OWNER_OR_PHYSICAL_HEARTBEAT_UNQUALIFIED:"+json.dumps({k:health.get(k) for k in ("mode","healthy","failure","pid","generation","host_monotonic_ns","native_source_host_ns")}))
                if MICRO.get("carry"):
                    for c in r.get("contacts",[]):
                        pair={c.get("body1"),c.get("body2")}
                        if "test_cylinder" in pair and not pair.intersection({"R_hand_narrow3_Link","R_hand_wide3_Link"}) and float(c["local_force_torque"][0])>.1:
                            raise RuntimeError("CARRY_OBJECT_EXTERNAL_SUPPORT")
                p=native_pose(r)
                limit=current_carry_transit_speed_limit("/workspace/evidence",r,MICRO["carry"]["pid"]) if MICRO.get("carry",{}).get("transit_policy_050") else .20
                fault=physical_fault(r,p,MICRO["joint_limits"],start,MICRO["direction"],robot_bodies=MICRO.get("robot_bodies"),allowed_contact_pairs=MICRO.get("allowed_contact_pairs"),planar_speed_limit=limit)
                if fault: raise RuntimeError(fault)
                imu=self.messages["imu"]
                q=imu["orientation"]
                norm2=sum(float(q[k])**2 for k in ("w","x","y","z"))
                if not math.isfinite(norm2) or abs(norm2-1.)>.02:
                    raise RuntimeError("INVALID_IMU_QUATERNION")
                cm=self.messages["common"]
                mode=cm["action_info"]["action_desc"];status=int(cm["action_info"]["status"]["value"])
                fsm=int(cm["fsm_state"]["current_state"]);body=int(cm["body_status"]["value"])
                if mode not in ("STAND_DEFAULT",MICRO.get("carry",{}).get("motion_mode","LOCOMOTION_DEFAULT")) or status not in (100,200,300) or fsm not in contract.MC_FSM_ACTIVE_STATES:
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
                source_arm_health=json.loads(Path("/workspace/evidence/native_arm_relay_heartbeat.json").read_text()) if PLAN.get("native_arm_relay") else None
                out=dict(sample_monotonic_s=r["host_monotonic_ns"]*1e-9, sim_time_s=r["sim_time_s"],
                         native_arm_source_health=source_arm_health,
                         pose=p,leg_states=leg,mc_leg_command=dict(header=self.messages["leg_command"]["header"],joints=targets),
                         raw_imu=imu,raw_ros_odom=self.messages.get("ros_odom"),
                         imu_quaternion_norm_squared=norm2,
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
            if MICRO.get("observe_native_arm_policy"):
                arm_pubs=self.get_publishers_info_by_topic("/robotsim/issue46/mc/arm/command")
                if not arm_pubs or any(p.node_name=="_NODE_NAME_UNKNOWN_" for p in arm_pubs):
                    raise RuntimeError("GRAPH_DISCOVERY_PENDING")
                if len(arm_pubs)!=1 or arm_pubs[0].node_name!=mcsubs[0].node_name or arm_pubs[0].node_namespace!="/" or arm_pubs[0].topic_type!="aimdk_msgs/msg/JointCommandArray":
                    raise RuntimeError("MC_ARM_POLICY_PRODUCER_NOT_VERIFIED")
                if not self.arm_policy_trace or time.monotonic()-self.arm_policy_trace[-1]["received_monotonic_s"]>.20:
                    raise RuntimeError("GRAPH_DISCOVERY_PENDING")
            if MICRO.get("carry"):
                pid=MICRO["carry"]["pid"]
                argv=Path("/proc",str(pid),"cmdline").read_bytes().split(bytes([0]))
                if b"/workspace/evidence/native_hal_bilateral_arm_pregrasp.py" not in argv:raise RuntimeError("CARRY_PROCESS_IDENTITY")
                for topic,typ in (("/aima/hal/joint/arm/command","aimdk_msgs/msg/JointCommandArray"),("/aima/hal/joint/hand/command","aimdk_msgs/msg/HandCommandArray")):
                    owners=self.get_publishers_info_by_topic(topic)
                    records=[dict(name=x.node_name,namespace=x.node_namespace,type=x.topic_type,gid=list(x.endpoint_gid)) for x in owners]
                    self.events.append(dict(event="CARRY_OWNER_GRAPH",topic=topic,t=time.monotonic(),owners=records))
                    if not owners or any(x.node_name in ("_NODE_NAME_UNKNOWN_","") or x.node_namespace in ("_NODE_NAMESPACE_UNKNOWN_","") or not x.topic_type for x in owners):raise RuntimeError("GRAPH_DISCOVERY_PENDING")
                    if len(owners)!=1 or owners[0].node_name!="robotsim_hal_bilateral_pregrasp" or owners[0].node_namespace!="/" or owners[0].topic_type!=typ:raise RuntimeError("CARRY_PUBLISHER_NOT_UNIQUE:"+json.dumps(dict(topic=topic,owners=records)))
            self.events.append(dict(event="UNIQUE_PROVENANCE_GRAPH",t=time.monotonic(),mc=candidates[0],
                                    publisher=pubs[0].node_name,subscriber=subs[0].node_name,
                                    configured_history_depth="KEEP_LAST/10; FastDDS graph history may be UNKNOWN/0"))

        def await_graph(self):
            deadline=time.monotonic()+getattr(self,"graph_startup_timeout_s",2.)
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

        def stand_once(self, requested="STAND_DEFAULT"):
            if requested!="STAND_DEFAULT" and not (requested=="LOCOMOTION_STEP" and MICRO.get("carry",{}).get("motion_mode")==requested):raise RuntimeError("ACTION_NOT_ALLOWLISTED")
            self.publish()
            q=SetMcAction.Request();q.header.stamp=self.get_clock().now().to_msg()
            q.source=SOURCE;q.command.action_desc=requested
            if not self.mode.service_is_ready():raise RuntimeError("STAND_SERVICE_NOT_READY")
            f=self.mode.call_async(q);until=time.monotonic()+.75
            self.events.append(dict(event="STAND_REQUEST" if requested=="STAND_DEFAULT" else "NATIVE_STEP_REQUEST",t=time.monotonic()))
            while not f.done() and time.monotonic()<until:
                self.publish();self.pump();time.sleep(.015)
            response=raw(f.result()) if f.done() and f.result() is not None else None
            self.events.append(dict(event="STAND_RESPONSE" if requested=="STAND_DEFAULT" else "NATIVE_STEP_RESPONSE",t=time.monotonic(),raw=response))
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
    if MICRO.get("carry"):
        return run_carry_navigation(n,contract)
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
            # Run81/82 raw command traces retain .14 even with nonzero
            # steering. Their prose description of an arc downgrade disagrees.
            command_velocity=velocity
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
                     r["common_mode_desc"]=="LOCOMOTION_DEFAULT" and
                     (not PLAN.get("native_arm_relay") or r.get("native_arm_source_health",{}).get("last_gain_positive")) for r in n.trace)
        result["source_and_motion_mode_observed"]=acquired
        if cfg.get("motion_mode","LOCOMOTION_DEFAULT")=="LOCOMOTION_DEFAULT":result["source_and_ld_observed"]=acquired
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
        result["abort_raw_imu"]=n.messages.get("imu");result["abort_raw_ros_odom"]=n.messages.get("ros_odom")
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
        result["mc_arm_policy_trace"] = n.arm_policy_trace
        result.update(events=n.events,command_log=n.command_log,bridge_commands=n.bridge_commands,trace=n.trace,
                      nonzero_started=motion_started,watchdog_limitation="IN_PROCESS_TIMER_PLUS_BRIDGE_STALE_INPUT_WATCHDOG",
                      stop_contract_revision=contract.__doc__)
        Path("/workspace/evidence/native_micro_probe_result.json").write_text(json.dumps(result,indent=2)+"\n")
        n.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in result.items() if k not in ("trace","command_log","bridge_commands","abort_native_sample")}),flush=True)

def build_carry_route_script(goal_xy, joint_limits, robot_bodies, *, carry_pid, timeout_s=90., angular_command_cap=.075, native_action_smoothing=False, native_step=False, transit_policy_050=False):
    """Reuse Micro transport/stop guards and the existing point-goal controller."""
    if angular_command_cap not in (.075,.10):raise ValueError('carry angular command must use existing cap or pinned SDK minimum')
    if len(goal_xy)!=2 or not all(math.isfinite(x) and -1.65<=x<=1.65 for x in goal_xy) or not 5.<=timeout_s<=90. or not isinstance(carry_pid,int) or carry_pid<=0:
        raise ValueError("invalid bounded carry route")
    if transit_policy_050 and not native_step:raise ValueError("TRANSIT050 requires isolated native STEP carry")
    script=build_script("forward",1.,joint_limits,command_speed=.12,robot_bodies=robot_bodies)
    script=script.rsplit("\nrun_probe()",1)[0]
    options=dict(carry=dict(goal_xy_m=list(goal_xy),pid=carry_pid,timeout_s=timeout_s,angular_command_cap=angular_command_cap,native_action_smoothing=bool(native_action_smoothing),motion_mode="LOCOMOTION_STEP" if native_step else "LOCOMOTION_DEFAULT",transit_policy_050=bool(transit_policy_050)),graph_startup_timeout_s=5.,
                 allowed_contact_pairs=[["test_cylinder",p] for p in ("R_hand_narrow3_Link","R_hand_wide3_Link")])
    return script+"\nMICRO.update("+repr(options)+")\nrun_probe()\n"



def recoverable_carry_clock_fault(fault, error, recovery_count):
    return (str(error).startswith("NATIVE_FEEDBACK_INVALID:") and
            isinstance(fault,dict) and fault.get("gate")=="CLOCK_STEP_DURING_MOTION" and
            0<=recovery_count<4)


def native_step_activation_ready(state, baseline):
    # Actual installed evidence: successful LS action selects STEP speed mode,
    # but zero locomotion retains SD. This is prepared, never motion success.
    return (state['body_status']==baseline and
            state['get_mode_header_code']==0 and state['state_age_s']['get_mode']<=.20 and
            state['common_mode_desc']==state['get_mode_desc'] and
            state['common_mode_desc'] in ('STAND_DEFAULT','LOCOMOTION_STEP') and
            state['common_mode_status_value']==100 and
            state['raw_common']['speed_status']['speed_mode']['value']==4)


def carry_stop_evidence(result):
    """Describe saved stop evidence without issuing actions or qualifying a stop."""
    commands=result.get("command_log",[])
    nonzero=[c for c in commands if any(c.get(k,0.) for k in ("forward","lateral","angular"))]
    last_nonzero=nonzero[-1]["t"] if nonzero else None
    zeros=[c for c in commands if not any(c.get(k,0.) for k in ("forward","lateral","angular"))
           and last_nonzero is not None and c["t"]>=last_nonzero]
    first_zero=zeros[0]["t"] if zeros else None
    events=result.get("events",[])
    requests=[e for e in events if e.get("event")=="STAND_REQUEST"]
    responses=[e for e in events if e.get("event")=="STAND_RESPONSE"]
    stand_samples=[]
    for row in result.get("trace",[]):
        ages=row.get("state_age_s",{})
        required=("odom","common","imu","leg_state","leg_command","get_mode")
        if (first_zero is not None and row.get("sample_monotonic_s",-1.)>=first_zero
                and (not stand_samples or row["sample_monotonic_s"]>stand_samples[-1])
                and all(k in ages and math.isfinite(ages[k]) and 0.<=ages[k]<=.20 for k in required)
                and row.get("get_mode_header_code")==0
                and row.get("get_mode_desc")=="STAND_DEFAULT"
                and row.get("common_mode_desc")=="STAND_DEFAULT"
                and row.get("get_mode_status_value")==100
                and row.get("common_mode_status_value")==100):
            stand_samples.append(row["sample_monotonic_s"])
    gate=result.get("post_stop_composite")
    gate_status=("PASS" if gate.get("pass") is True else "FAIL" if gate.get("pass") is False
                 else "UNCONFIRMED") if isinstance(gate,dict) else "NOT_RUN"
    return dict(
        classification="OBSERVATION_ONLY_NOT_AN_ADDITIONAL_STOP_ACCEPTANCE",
        command_publication_evidence="PYTHON_PUBLISH_CALL_ONLY_NO_MC_ACK",
        zero_receipt_ack="UNAVAILABLE",
        last_nonzero_publish_s=last_nonzero,first_zero_after_nonzero_publish_s=first_zero,
        zero_publish_count_after_nonzero=len(zeros),
        zero_publication_span_s=zeros[-1]["t"]-first_zero if zeros else None,
        zero_publication_max_gap_s=max((b["t"]-a["t"] for a,b in zip(zeros,zeros[1:])),default=None),
        stand_request_count=len(requests),
        stand_service_responses=[dict(t=e.get("t"),raw=e.get("raw")) for e in responses],
        fresh_agreeing_stand_after_zero_sample_count=len(stand_samples),
        first_fresh_agreeing_stand_after_zero_s=stand_samples[0] if stand_samples else None,
        independent_composite_stop=gate_status,
        route_acceptance_recorded=bool(gate_status=="PASS" and result.get("pass_result") is True
                                      and not result.get("failure")),
        rearm_authorized=False)


def run_carry_navigation(n,contract):
    """One bounded measured-pose task, with zero/qualified-stop clock recovery."""
    import threading
    from point_goal_controller import PointGoalController,ControllerConfig,Pose2D,Goal2D,NavState
    cfg=MICRO["carry"];result=dict(classification="SIMULATION_ONLY_NATIVE_LOCOMOTION_STEP_CARRY" if cfg.get("motion_mode")=="LOCOMOTION_STEP" else "SIMULATION_ONLY_FETCH_CARRY_ROUTE",pass_result=False,
        goal_xy_m=cfg["goal_xy_m"],zero_receipt_ack="UNAVAILABLE",clock_recoveries=[],route_samples=[])
    if cfg.get("transit_policy_050"):result["classification"]="SIMULATION_ONLY_NATIVE_STEP_CARRY_TRANSIT_050"
    watchdog=None;began=None;baseline=None;transit_permit=None
    try:
        n.start_feedback();until=time.monotonic()+5.
        while time.monotonic()<until:
            n.publish();n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:break
            time.sleep(.01)
        n.await_graph();baseline=n.sample()["body_status"]
        result["pre_motion_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        start=n.sample()["pose"];result["start_pose"]=start
        goal=cfg["goal_xy_m"];bearing=math.atan2(goal[1]-start["y"],goal[0]-start["x"])
        if cfg.get("motion_mode")=="LOCOMOTION_STEP":
            n.stand_once("LOCOMOTION_STEP")
            step_until=time.monotonic()+.75;step_ready=False
            while time.monotonic()<step_until:
                n.publish();n.pump();state=n.sample()
                result.setdefault("native_step_transition",[]).append({k:state[k] for k in ("sample_monotonic_s","sim_time_s","get_mode_desc","common_mode_desc","common_mode_status_value","pose")})
                if state["body_status"]!=baseline:raise RuntimeError("STEP_BODY_STATUS_CHANGED")
                if native_step_activation_ready(state,baseline):
                    step_ready=True;result["native_step_prepared_not_motion_pass"]=True;break
                time.sleep(.02)
            if not step_ready:raise RuntimeError("NATIVE_STEP_ACTIVATION_PRECONDITIONS_UNCONFIRMED")
        controller=PointGoalController(ControllerConfig(xy_tolerance_m=.15,yaw_tolerance_rad=.12,max_forward_mps=.12,max_angular_rps=cfg.get("angular_command_cap",.075),k_angular=.8,goal_timeout_s=cfg["timeout_s"]))
        target=Goal2D(*goal,bearing,"native_scene_world")
        began=time.monotonic();until=began+cfg["timeout_s"];last_graph=began;acquired=False;activation_started=began;progress_state=None;progress_best=None;progress_at=began
        if cfg.get("transit_policy_050"):transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
        with n.lock:n.moving=True
        watchdog=threading.Timer(cfg["timeout_s"],n.deadline_zero);watchdog.daemon=True;watchdog.start()
        while time.monotonic()<until:
            tick=time.monotonic()
            if not n.moving and (n.feedback_fault or {}).get("gate")!="CLOCK_STEP_DURING_MOTION":break
            try:
                n.pump();row=n.sample()
            except RuntimeError as exc:
                fault=n.feedback_fault or {}
                # Only a qualified source-clock epoch fault is recoverable.
                if cfg.get("motion_mode")=="LOCOMOTION_STEP" or not recoverable_carry_clock_fault(fault,exc,len(result["clock_recoveries"])):raise
                n.deadline_zero();n.stop_started=True
                if n.moving:raise RuntimeError("CLOCK_RECOVERY_STILL_MOVING")
                n.feedback_fault=None
                n.observe(1.05,baseline=baseline);n.stand_once()
                stopped=n.observe(8.,baseline=baseline,require_stop=True)
                result["clock_recoveries"].append(dict(fault=fault,stop=stopped,t=time.monotonic()))
                controller.reset();n.await_graph()
                n.stop_started=False
                with n.lock:n.moving=True
                activation_started=time.monotonic();continue
            if row["body_status"]!=baseline:raise RuntimeError("BODY_STATUS_CHANGED")
            p=row["pose"]
            if not -1.65<=p["x"]<=1.65 or not -1.75<=p["y"]<=1.75:raise RuntimeError("CARRY_ROOM_WORKSPACE")
            control_now=time.monotonic()
            output=controller.step(Pose2D(p["x"],p["y"],p["yaw"],"native_scene_world",row["sample_monotonic_s"]),target,now_steady_s=control_now)
            if output.state in (NavState.ALIGN,NavState.DRIVE,NavState.FINAL_ALIGN):
                metric=output.distance_m if output.state==NavState.DRIVE else abs(output.yaw_error_rad)
                threshold=.01 if output.state==NavState.DRIVE else .02
                if progress_state!=output.state or progress_best is None or metric<progress_best-threshold:
                    progress_state=output.state;progress_best=metric;progress_at=tick
                elif tick-progress_at>3.:
                    raise RuntimeError("CARRY_COMMAND_WITHOUT_MEASURED_PROGRESS")
                # This is the existing point-goal controller, not a qualified
                # straight/curved-reference tracking run. Keep path error visible.
                dx,dy=goal[0]-start["x"],goal[1]-start["y"];length=math.hypot(dx,dy)
                cross=abs(dx*(p["y"]-start["y"])-dy*(p["x"]-start["x"]))/length
                result["max_cross_track_to_direct_segment_m"]=max(result.get("max_cross_track_to_direct_segment_m",0.),cross)
                if output.state==NavState.ALIGN and abs(output.yaw_error_rad)>abs((bearing-start["yaw"]+math.pi)%(2*math.pi)-math.pi)+.10:
                    raise RuntimeError("CARRY_HEADING_RESPONSE_DIVERGED")
                forward=.12 if output.linear_x>0 else 0.
                angular=output.angular_z
                bounds=row["raw_common"]["speed_status"]
                if not bounds["forward_bounds"]["min_value"]<=forward<=bounds["forward_bounds"]["max_value"] or not bounds["angular_bounds"]["min_value"]<=angular<=bounds["angular_bounds"]["max_value"]:raise RuntimeError("MC_REPORTED_INPUT_BOUNDS")
                if row["common_mode_status_value"]==300 and not (cfg.get("native_action_smoothing") and not acquired and 0<=tick-activation_started<=.75):raise RuntimeError("MC_TRANSITION_DURING_CARRY_COMMAND")
                handshake=(row["common_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_header_code"]==0 and row["state_age_s"]["get_mode"]<=.20 and row["common_source"]["name"]=="robotsim_official_cmd_vel")
                acquired=acquired or handshake
                if tick-activation_started>.75 and not handshake:raise RuntimeError("CARRY_SOURCE_LD_HANDSHAKE_NOT_CONFIRMED")
                n.publish(forward,0.,angular)
            elif output.state==NavState.SETTLING:n.publish()
            elif output.state==NavState.ARRIVED:
                result["goal_triggered_stop"]=True;break
            else:
                result["controller_rejection"]={"state":output.state.value,"reason":output.reason,"control_now_s":control_now,"native_source_s":row["sample_monotonic_s"]}
                raise RuntimeError("CARRY_CONTROLLER_"+output.state.value+":"+output.reason)
            result["route_samples"].append(dict(t=tick,state=output.state.value,pose=p,distance_m=output.distance_m,yaw_error_rad=output.yaw_error_rad,linear=n.command_log[-1]["forward"],angular=n.command_log[-1]["angular"],controller_linear=output.linear_x,controller_angular=output.angular_z))
            n.trace.append(row)
            if tick-last_graph>=1.:n.graph();last_graph=tick
            time.sleep(max(0.,tick+.02-time.monotonic()))
        n.deadline_zero();n.stop_started=True
        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
        if watchdog:watchdog.cancel()
        n.observe(1.05,baseline=baseline);n.stand_once()
        result["post_stop_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        end=n.sample()["pose"];result["end_pose"]=end
        result["final_distance_m"]=math.hypot(goal[0]-end["x"],goal[1]-end["y"])
        result["measured_distance_m"]=math.hypot(end["x"]-start["x"],end["y"]-start["y"])
        result["source_and_motion_mode_observed"]=acquired
        if cfg.get("motion_mode","LOCOMOTION_DEFAULT")=="LOCOMOTION_DEFAULT":result["source_and_ld_observed"]=acquired
        result["pass_result"]=bool(result.get("goal_triggered_stop") and acquired and result["post_stop_composite"]["pass"] and result["final_distance_m"]<=.15 and result["measured_distance_m"]>.005)
        result["failure"]=None if result["pass_result"] else "CARRY_GOAL_OR_STOP_NOT_QUALIFIED"
    except Exception as exc:
        n.deadline_zero();result["failure"]=str(exc);result["abort_native_sample"]=n.latest_row
    finally:
        n.deadline_zero()
        try:transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
        except Exception as exc:result["transit_policy_error"]=str(exc);result["pass_result"]=False
        if watchdog:watchdog.cancel()
        end=time.monotonic()+1.05
        while time.monotonic()<end:
            n.publish()
            try:n.pump()
            except Exception:pass
            time.sleep(.015)
        try:transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
        except Exception as exc:result["transit_policy_error"]=str(exc);result["pass_result"]=False
        result["transit_policy_record"]=transit_permit
        try:n.stop_feedback()
        except Exception as exc:result["feedback_cleanup_failure"]=str(exc);result["pass_result"]=False
        result.update(events=n.events,command_log=n.command_log,bridge_commands=n.bridge_commands,trace=n.trace)
        try:result["stop_evidence"]=carry_stop_evidence(result)
        except Exception as exc:
            result["stop_evidence_error"]=str(exc);result["pass_result"]=False
        write_durable_milestone("/workspace/evidence/native_carry_route_result.json",result)
        n.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in result.items() if k not in ("events","command_log","bridge_commands","trace","abort_native_sample","route_samples")}),flush=True)
