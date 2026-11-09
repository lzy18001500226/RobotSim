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


def physical_fault(row, pose, limits, start=None, direction=None,robot_bodies=None,allowed_contact_pairs=None,planar_speed_limit=.20,review_incidental_contacts=False):
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
            incidental=(frozenset((a,b))==frozenset(("pelvis","left_hip_roll_link")) or "test_cylinder" in (a,b) and (ra or rb))
            if review_incidental_contacts and incidental and contact["distance_m"]>=-.005:continue
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


def carry_policy_speed_limit(records,stamp_ns,generation,owner_pid):
    """At most two non-overlapping one-use intervals for the away/1m task."""
    if records is None:return .20
    if isinstance(records,dict):records=[records]
    if not isinstance(records,list):raise ValueError("CARRY_POLICY_INTERVALS")
    budget=7 if records and all(r.get("task_interval_budget")==7 for r in records) else 6 if records and all(r.get("task_interval_budget")==6 for r in records) else 2
    if len(records)>budget:raise ValueError("CARRY_POLICY_INTERVALS")
    if budget in (6,7) and len({r["command_deadline_ns"] for r in records})>1:raise ValueError("CARRY_POLICY_DEADLINE_RESET")
    ordered=sorted(records,key=lambda r:r["created_ns"])
    for first,second in zip(ordered,ordered[1:]):
        if first.get("revoked_ns") is None or first["revoked_ns"]>second["created_ns"]:
            raise ValueError("CARRY_POLICY_OVERLAP_OR_UNQUALIFIED_REARM")
    return max([.20]+[carry_transit_speed_limit(r,stamp_ns,generation,owner_pid) for r in ordered])


def current_carry_transit_speed_limit(evidence, row, owner_pid):
    """Only a live PID/start-token verified process can permit .50."""
    import os,json,time
    from pathlib import Path
    paths=[Path(evidence)/name for name in ("carry_transit_permit.json","carry_transit_forward_permit.json","carry_transit_delivery_permit.json",*[f"carry_transit_delivery_rearm_{i}.json" for i in range(1,5)],*[f"carry_transit_clock_recovery_{i}.json" for i in range(1,5)])]
    records=[json.loads(path.read_text()) for path in paths if path.exists()]
    now=time.monotonic_ns()
    limit=min(carry_policy_speed_limit(records,now,row["generation"],owner_pid),
              carry_policy_speed_limit(records,row["host_monotonic_ns"],row["generation"],owner_pid))
    if limit==.20:return limit
    live=[r for r in records if carry_transit_speed_limit(r,now,row["generation"],owner_pid)==.50]
    if len(live)!=1:raise ValueError("CARRY_TRANSIT_PROCESS_NOT_UNIQUE")
    record=live[0];proc=Path("/proc")/str(record["route_pid"])
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
    name=cfg.get("permit_filename","carry_transit_permit.json")
    if name not in ("carry_transit_permit.json","carry_transit_forward_permit.json","carry_transit_delivery_permit.json",*[f"carry_transit_delivery_rearm_{i}.json" for i in range(1,5)],*[f"carry_transit_clock_recovery_{i}.json" for i in range(1,5)]):raise ValueError("CARRY_PERMIT_PATH")
    path=Path(evidence)/name;now=time.monotonic_ns()
    if phase=="TRANSIT":
        if record is not None or path.exists():raise RuntimeError("CARRY_TRANSIT_REARM_FORBIDDEN")
        seconds=float(cfg["timeout_s"])
        if not 5.<=seconds<=90.:raise ValueError("CARRY_TRANSIT_TIMEOUT")
        if name=="carry_transit_delivery_permit.json":
            previous=json.loads((Path(evidence)/cfg["previous_permit_filename"]).read_text())
            if not cfg.get("delivery") or cfg.get("delivery_turn_stop_pass") is not True or previous.get("revoked_ns") is None or previous["owner_pid"]!=cfg["pid"] or previous["generation"]!=native_row["generation"]:
                raise RuntimeError("DELIVERY_REARM_NOT_QUALIFIED")
        if name in [f"carry_transit_delivery_rearm_{i}.json" for i in range(1,5)]:
            previous=json.loads((Path(evidence)/cfg["previous_permit_filename"]).read_text())
            if not cfg.get("delivery") or (cfg.get("delivery_alignment_stop_pass") is not True and cfg.get("egress_stop_pass") is not True) or previous.get("revoked_ns") is None or previous["owner_pid"]!=cfg["pid"] or previous["generation"]!=native_row["generation"] or cfg.get("absolute_command_deadline_ns")!=previous["command_deadline_ns"]:
                raise RuntimeError("DELIVERY_ALIGNMENT_REARM_NOT_QUALIFIED")
        if name.startswith("carry_transit_clock_recovery_"):
            previous=json.loads((Path(evidence)/cfg["previous_permit_filename"]).read_text())
            if not cfg.get("compact_turn") or cfg.get("clock_stop_pass") is not True or previous.get("revoked_ns") is None or previous["owner_pid"]!=cfg["pid"] or previous["generation"]!=native_row["generation"]:
                raise RuntimeError("CLOCK_RECOVERY_REARM_NOT_QUALIFIED")
        if name=="carry_transit_forward_permit.json":
            previous=json.loads((Path(evidence)/"carry_transit_permit.json").read_text())
            if cfg.get("turn_stop_pass") is not True or previous.get("revoked_ns") is None or previous["owner_pid"]!=cfg["pid"] or previous["generation"]!=native_row["generation"]:
                raise RuntimeError("CARRY_FORWARD_REARM_NOT_QUALIFIED")
        deadline=min(now+int(seconds*1e9),int(cfg.get("absolute_command_deadline_ns",now+int(seconds*1e9))))
        record=dict(schema="robotsim.carry-transit050.v1",owner_pid=cfg["pid"],
                    route_pid=os.getpid(),route_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19]),
                    generation=native_row["generation"],created_ns=now,
                    command_deadline_ns=deadline,absolute_deadline_ns=deadline+15_000_000_000,
                    braking_ns=None,revoked_ns=None)
        if cfg.get("compact_turn"):record["task_interval_budget"]=7 if cfg.get("delivery") else 6
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
                if MICRO.get("carry",{}).get("inplace_centre_hold"):
                    limit=(.20 if MICRO["carry"].get("compact_centre_feedback") else 0. if MICRO["carry"].get("compact_turn") else MICRO["carry"].get("turn_recovery_speed",.12)) if MICRO["carry"].get("active_mission_phase")=="TURN" else MICRO["carry"].get("transit_command_speed",.12)
                    if math.hypot(fwd,lat)>limit+1e-8 or abs(ang)>MICRO["carry"].get("angular_command_cap",.1)+1e-8:
                        raise RuntimeError("INPLACE_COMMAND_BOUND")
                    if not MICRO["carry"].get("compact_centre_feedback") and MICRO["carry"].get("active_mission_phase")=="TURN" and math.hypot(fwd,lat)>.12000001 and (not MICRO["carry"].get("active_centre_recovery") or abs(ang)>1e-9):
                        raise RuntimeError("RECOVERY020_OUTSIDE_POSITION_ONLY_PHASE")
                    if MICRO["carry"].get("compact_centre_feedback") and lat:raise RuntimeError("UNQUALIFIED_LATERAL_COMPACT_COMMAND")
                    if lat and MICRO["carry"].get("active_mission_phase")!="TURN":
                        raise RuntimeError("LATERAL_COMMAND_OUTSIDE_INPLACE_TURN")
                carry=MICRO.get("carry",{})
                if carry.get("delivery"):
                    positive=float(carry["delivery"].get("delivery_forward_command_m_s",.20)) if carry.get("active_mission_phase") in ("EGRESS","DELIVERY") else .20
                    if fwd>positive+1e-8 or fwd<-.20000001 or lat:raise RuntimeError("TWO_TABLE_PHASE_COMMAND_BOUND")
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
                            if MICRO["carry"].get("incidental_contact_diagnostic") and c["distance_m"]>=-.005 and pair.intersection(set(MICRO.get("robot_bodies",[]))):continue
                            raise RuntimeError("CARRY_OBJECT_EXTERNAL_SUPPORT")
                if (MICRO.get("carry",{}).get("inplace_centre_hold") or MICRO.get("carry",{}).get("compact_turn")) and self.bridge_commands:
                    command=self.bridge_commands[-1]["raw"]
                    if MICRO["carry"].get("compact_centre_feedback") and command["lateral_velocity"]:raise RuntimeError("OBSERVED_UNQUALIFIED_LATERAL_COMPACT_COMMAND")
                    limit=(.20 if MICRO["carry"].get("compact_centre_feedback") else 0. if MICRO["carry"].get("compact_turn") else MICRO["carry"].get("turn_recovery_speed",.12)) if MICRO["carry"].get("active_mission_phase")=="TURN" else MICRO["carry"].get("transit_command_speed",.12)
                    if math.hypot(command["forward_velocity"],command["lateral_velocity"])>limit+1e-8 or abs(command["angular_velocity"])>MICRO["carry"].get("angular_command_cap",.1)+1e-8:
                        raise RuntimeError("OBSERVED_BRIDGE_COMMAND_BOUND")
                p=native_pose(r)
                limit=current_carry_transit_speed_limit("/workspace/evidence",r,MICRO["carry"]["pid"]) if MICRO.get("carry",{}).get("transit_policy_050") else .20
                fault=physical_fault(r,p,MICRO["joint_limits"],start,MICRO["direction"],robot_bodies=MICRO.get("robot_bodies"),allowed_contact_pairs=MICRO.get("allowed_contact_pairs"),planar_speed_limit=limit,review_incidental_contacts=bool(MICRO.get("carry",{}).get("incidental_contact_diagnostic")))
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
                    # Initial zero-only readiness may reacquire telemetry; any
                    # already-commanded motion or native/physical fault stays fail-closed.
                    initial_zero_only=bool(MICRO.get("carry")) and not self.moving and not any(
                        any(float(c.get(k,0.)) for k in ("forward","lateral","angular")) for c in self.command_log)
                    if (initial_zero_only and str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:")
                            and time.monotonic()<until-seconds+min(3.,seconds)):
                        self.events.append(dict(event="PREMOTION_FRESHNESS_REACQUIRE",t=time.monotonic(),failure=str(exc)))
                        rows=[];last=None;qualified=None
                        time.sleep(max(0.,tick+.02-time.monotonic()))
                        continue
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

        def source_enabled(self, enabled):
            if not MICRO.get("carry",{}).get("release_source_on_stop"):return
            if getattr(self,"source_requested_enabled",True)==enabled:return
            q=SetMcInputSource.Request();q.request.header.stamp=self.get_clock().now().to_msg()
            q.action.value=McInputAction.INPUTACTION_ENABLE if enabled else McInputAction.INPUTACTION_DISABLE
            q.input_source.name="robotsim_official_cmd_vel";q.input_source.priority=40;q.input_source.timeout=1000
            if not self.src.service_is_ready():raise RuntimeError("CARRY_SOURCE_ADMIN_UNAVAILABLE")
            f=self.src.call_async(q);deadline=time.monotonic()+.5
            while not f.done() and time.monotonic()<deadline:
                self.publish();self.pump();time.sleep(.01)
            response=raw(f.result()) if f.done() and f.result() is not None else None
            self.events.append(dict(event="SOURCE_ENABLE_ACK" if enabled else "SOURCE_DISABLE_ACK",t=time.monotonic(),request=raw(q),raw=response,meaning="SERVICE_ACCEPTANCE_NOT_TASK_COMPLETION_OR_ZERO_RECEIPT"))
            if response is None or response["response"]["header"]["code"]!=0:raise RuntimeError("CARRY_SOURCE_ADMIN_REJECTED_OR_TIMEOUT")
            self.source_requested_enabled=enabled

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
        if cfg.get("release_source_on_stop"):
            try:n.source_enabled(False)
            except Exception as exc:result["source_disable_cleanup_failure"]=str(exc)
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

def native_foot_loads(row):
    values={"left":0.,"right":0.}
    for c in row["contacts"]:
        if c.get("efc_address",-1)<0 or "world" not in (c.get("body1"),c.get("body2")):continue
        for side in values:
            if any(c.get(k) in (side+"_ankle_pitch_link",side+"_ankle_roll_link") for k in ("body1","body2")):
                values[side]+=max(0.,float(c["local_force_torque"][0]))
    total=sum(values.values())
    dominant=max(values,key=values.get) if total>50. and max(values.values())>.75*total else None
    return values,dominant


def bounded_start_scale(elapsed_s):
    """Half-second command ramp; no increase to command or joint limits."""
    if not math.isfinite(elapsed_s) or elapsed_s<0:raise RuntimeError("INVALID_START_CLOCK")
    u=min(1.,elapsed_s/.5)
    return u*u*(3.-2.*u)


def inplace_recovery_command(pose, origin, speed=.12):
    """Position-first correction above the observed forward deadzone."""
    if speed not in (.12,.20):raise ValueError("unreviewed turn recovery command")
    values=[pose[k] for k in ("x","y","yaw")]+[origin[k] for k in ("x","y")]
    if not all(math.isfinite(v) for v in values):raise RuntimeError("INVALID_TURN_CENTRE_FEEDBACK")
    ex=origin["x"]-pose["x"];ey=origin["y"]-pose["y"]
    c=math.cos(pose["yaw"]);si=math.sin(pose["yaw"])
    f=c*ex+si*ey;l=-si*ex+c*ey
    if abs(f)>=abs(l):return math.copysign(speed,f),0.
    return 0.,math.copysign(.20 if speed==.20 else .05,l)


def inplace_centre_command(pose, origin, max_speed=.12, forward_only=False):
    """Existing centre feedback; explicit combined-turn diagnostic may use .20."""
    if max_speed not in (.12,.20):raise ValueError("centre feedback cap")
    values=[pose[k] for k in ("x","y","yaw")]+[origin[k] for k in ("x","y")]
    if not all(math.isfinite(v) for v in values):raise RuntimeError("INVALID_TURN_CENTRE_FEEDBACK")
    ex=origin["x"]-pose["x"];ey=origin["y"]-pose["y"]
    radius=math.hypot(ex,ey)
    if radius<=.01:return 0.,0.
    gain=.12/.05
    c=math.cos(pose["yaw"]);si=math.sin(pose["yaw"])
    if forward_only:
        error=c*ex+si*ey
        return (max(-max_speed,min(max_speed,gain*error)) if abs(error)>.01 else 0.),0.
    vx=gain*ex;vy=gain*ey
    scale=min(1.,max_speed/math.hypot(vx,vy));vx*=scale;vy*=scale
    return c*vx+si*vy,-si*vx+c*vy



def table_terminal_braking_footprint(poses, origin, target_yaw, baseline_radius, extension=None):
    """Task clearance only; never substitutes for the composite physical stop."""
    peak=0.;outside=[];used=False
    for pose in poses:
        dx=pose["x"]-origin["x"];dy=pose["y"]-origin["y"];radius=math.hypot(dx,dy);peak=max(peak,radius)
        if radius<=baseline_radius:continue
        yaw_error=abs((pose["yaw"]-target_yaw+math.pi)%(2*math.pi)-math.pi)
        allowed=(extension is not None and extension.get("frame")=="native_scene_world" and
                 0.<=dx and abs(dy)<=extension["lateral_m"] and radius<=extension["maximum_radius_m"] and
                 yaw_error<=extension["terminal_yaw_error_rad"])
        if allowed:used=True
        else:outside.append(dict(dx_m=dx,dy_m=dy,radius_m=radius,yaw_error_rad=yaw_error))
    return dict(pass_clearance=not outside,peak_radius_m=peak,used_aisle_reserve=used,
                original_radial_criterion_pass=peak<=baseline_radius,violations=outside)



def table_turn_target(pose, goal_xy):
    bearing=math.atan2(goal_xy[1]-pose["y"],goal_xy[0]-pose["x"])
    delta=(bearing-pose["yaw"]+math.pi)%(2*math.pi)-math.pi
    return bearing,pose["yaw"]+delta


def pure_forward_activation_output(output):
    from dataclasses import replace
    return replace(output,angular_z=0.)


def delivery_direction_probe(probe, pose, now):
    """One short pure-forward activation; physical displacement decides direction."""
    if probe is None:
        probe=dict(started_s=now,origin=dict(pose),qualified=False)
    origin=probe["origin"]
    dx=pose["x"]-origin["x"];dy=pose["y"]-origin["y"]
    forward=math.cos(origin["yaw"])*dx+math.sin(origin["yaw"])*dy
    lateral=-math.sin(origin["yaw"])*dx+math.cos(origin["yaw"])*dy
    probe=dict(probe,elapsed_s=now-probe["started_s"],forward_m=forward,lateral_m=lateral)
    if forward<-.03:probe["failure"]="DELIVERY_FORWARD_RESPONSE_REVERSED_30MM"
    elif abs(lateral)>.05:probe["failure"]="DELIVERY_DIRECTION_PROBE_LATERAL_50MM"
    elif forward>=.005:probe["qualified"]=True
    elif now-probe["started_s"]>=1.:probe["failure"]="DELIVERY_FORWARD_RESPONSE_UNCONFIRMED_1S"
    return probe


def delivery_forward_progress(state, distance, now, active, epoch):
    """Accumulate the existing three-second no-progress budget across ALIGN."""
    if state is None:
        if not active:return None
        state=dict(best_distance_m=distance,active_without_progress_s=0.,last_time=now,last_active=False,epoch=epoch)
    result=dict(state)
    if now<result["last_time"]:raise RuntimeError("DELIVERY_PROGRESS_CLOCK")
    if epoch==result["epoch"] and result["last_active"]:
        # The existing command bridge expires each request after 0.20s.
        result["active_without_progress_s"]+=min(.20,now-result["last_time"])
    if distance<result["best_distance_m"]-.01:
        result["best_distance_m"]=distance;result["active_without_progress_s"]=0.
    result.update(last_time=now,last_active=bool(active),epoch=epoch,
                  no_progress=result["active_without_progress_s"]>3.,
                  timer_scope="PUBLISHED_REQUEST_LEASE_NOT_MC_ACK")
    return result


def build_carry_route_script(goal_xy, joint_limits, robot_bodies, *, carry_pid, timeout_s=90., angular_command_cap=.075, native_action_smoothing=False, native_step=False, transit_policy_050=False, reverse_moving_arc=False, away_1m=False, allow_carried_bottle_tilt=False, straight_retreat_turn=False, inplace_centre_hold=False, turn_brake_delay_s=0., retreat_brake_delay_s=0.,turn_recovery_speed=.12, transit_command_speed=.12, compact_turn=False, gait_prepare=False, compact_angular_cap=.30, release_source_on_stop=False, compact_centre_feedback=False, delivery=None, vendor_transition_table=False):
    """Reuse Micro transport/stop guards and the existing point-goal controller."""
    step_all=bool(delivery and delivery.get("transit_motion_mode")=="LOCOMOTION_STEP")
    if step_all and (not native_step or not vendor_transition_table):raise ValueError("native STEP transit requires original table and explicit source mode")
    if vendor_transition_table and (native_action_smoothing or native_step and not step_all or not delivery):raise ValueError("vendor action-table comparison scope")
    if delivery is not None:
        if delivery.get("transit_motion_mode","LOCOMOTION_DEFAULT") not in ("LOCOMOTION_DEFAULT","LOCOMOTION_STEP"):raise ValueError("unsupported native transit mode")
        if float(delivery.get("delivery_forward_command_m_s",.20)) not in (.20,.30):raise ValueError("delivery forward response profile bound")
        if delivery.get("stand_transition_strategy","ZERO_WINDOW_THEN_SD") not in ("ZERO_WINDOW_THEN_SD","EARLY_SD_CONTINUOUS_ZERO_DIAGNOSTIC"):raise ValueError("stand transition diagnostic scope")
        direct_retreat=delivery.get("activation_protocol")=="VENDOR_DIRECT_RETREAT_NONZERO"
        if not (compact_turn and (gait_prepare or direct_retreat and transit_command_speed==.20) and release_source_on_stop and compact_centre_feedback and (not native_step or step_all)):raise ValueError("delivery requires the measured FoundationWalk compact/stop or explicit vendor-direct retreat profile")
        if not .30<=float(delivery.get("turn_braking_footprint_m",.30))<=.35:raise ValueError("task turn braking footprint bound")
        extension=delivery.get("terminal_braking_aisle_reserve")
        if extension is not None:
            if extension.get("classification")!="SIMULATION_ONLY_TERMINAL_AISLE_RESERVE" or extension.get("frame")!="native_scene_world" or extension.get("maximum_radius_m")!=.50 or extension.get("lateral_m")!=.20 or extension.get("terminal_yaw_error_rad")!=.40:raise ValueError("terminal aisle reserve scope")
            import hashlib
            proof=Path(extension["static_clearance_path"])
            if hashlib.sha256(proof.read_bytes()).hexdigest()!=extension["static_clearance_sha256"] or json.loads(proof.read_text()).get("pass_clearance") is not True:raise ValueError("terminal aisle clearance provenance")

        if not .05<=float(delivery.get("retreat_cross_track_limit_m",.05))<=.15:raise ValueError("retreat task corridor bound")
        if not 0.<=float(delivery.get("intermediate_turn_stop_reserve_rad",0.))<=.40:raise ValueError("intermediate stopping reserve bound")
        if delivery.get("gait_prepare_required_load_states",2) not in (1,2):raise ValueError("gait readiness scope")
        if delivery.get("gait_prepare_angular_rad_s",.10) not in (-.10,.10):raise ValueError("gait preparation direction")
        if not .12<=float(delivery.get("intermediate_turn_heading_tolerance_rad",.12))<=.30:raise ValueError("intermediate task heading bound")
        if delivery.get("schema")!="robotsim.x2.two-table-delivery.v1" or delivery.get("frame")!="native_scene_world":raise ValueError("delivery scene contract")
        if len(delivery["base_goal_world_xy"])!=2 or not all(math.isfinite(x) and -1.65<=x<=1.65 for x in delivery["base_goal_world_xy"]):raise ValueError("delivery goal workspace")
    if compact_centre_feedback and not compact_turn:raise ValueError("combined centre feedback requires compact carry")
    if gait_prepare and not compact_turn:raise ValueError("gait preparation is scoped to the explicit compact carry experiment")
    if compact_turn and not (straight_retreat_turn and transit_policy_050 and not inplace_centre_hold):raise ValueError('compact turn requires pure-yaw bounded carry profile')
    if compact_angular_cap not in (.30,.60) or (compact_angular_cap!=.30 or release_source_on_stop) and not compact_turn:raise ValueError("compact angular/source protocol scope")
    if compact_turn:angular_command_cap=compact_angular_cap
    if angular_command_cap not in (.075,.10,.30,.60) or angular_command_cap>.10 and not compact_turn:raise ValueError('carry angular command must use existing cap or pinned SDK minimum')
    if (not away_1m and (goal_xy is None or len(goal_xy)!=2 or not all(math.isfinite(x) and -1.65<=x<=1.65 for x in goal_xy))) or not 5.<=timeout_s<=90. or not isinstance(carry_pid,int) or carry_pid<=0:
        raise ValueError("invalid bounded carry route")
    if away_1m and (not transit_policy_050 or (native_step and not straight_retreat_turn) or reverse_moving_arc):raise ValueError("away1m requires distinct bounded FoundationWalk profile")
    if transit_command_speed not in (.12,.20) or transit_command_speed>.12 and not (straight_retreat_turn and transit_policy_050):
        raise ValueError('effective020 transit is scoped to the explicit simulation carry task')
    if turn_recovery_speed not in (.12,.20) or turn_recovery_speed>.12 and not (straight_retreat_turn and inplace_centre_hold):
        raise ValueError("turn-recovery020 is scoped to explicit in-place centre correction")
    if not math.isfinite(retreat_brake_delay_s) or not 0.<=retreat_brake_delay_s<=3.:raise ValueError("retreat braking bound")
    if not math.isfinite(turn_brake_delay_s) or not 0.<=turn_brake_delay_s<=3. or (inplace_centre_hold or turn_brake_delay_s) and not straight_retreat_turn:
        raise ValueError("turn-centre/braking profile requires the explicit in-place task")
    if straight_retreat_turn and (not away_1m or not (native_action_smoothing or vendor_transition_table)):
        raise ValueError("retreat/in-place prefix requires the explicit bounded carry transition path")
    if reverse_moving_arc and (not transit_policy_050 or native_step):raise ValueError("reverse moving arc requires explicit FoundationWalk carry transit profile")
    script=build_script("forward",1.,joint_limits,command_speed=.12,robot_bodies=robot_bodies)
    script=script.rsplit("\nrun_probe()",1)[0]
    options=dict(carry=dict(goal_xy_m=None if away_1m else list(goal_xy),pid=carry_pid,timeout_s=timeout_s,angular_command_cap=angular_command_cap,native_action_smoothing=bool(native_action_smoothing),vendor_transition_table=bool(vendor_transition_table),review_prefix_only=bool((delivery or {}).get("review_prefix_only")),incidental_contact_diagnostic=bool((delivery or {}).get("incidental_contact_diagnostic") or (delivery or {}).get("review_prefix_only")),native_step_all_transit=step_all,motion_mode="LOCOMOTION_STEP" if step_all or native_step and not straight_retreat_turn else "LOCOMOTION_DEFAULT",straight_retreat_turn=bool(straight_retreat_turn),compact_turn=bool(compact_turn),delivery=delivery,compact_centre_feedback=bool(compact_centre_feedback),release_source_on_stop=bool(release_source_on_stop),gait_prepare=bool(gait_prepare),inplace_centre_hold=bool(inplace_centre_hold),turn_recovery_speed=float(turn_recovery_speed),transit_command_speed=float(transit_command_speed),turn_brake_delay_s=float(turn_brake_delay_s),retreat_brake_delay_s=float(retreat_brake_delay_s),inplace_motion_mode="LOCOMOTION_STEP" if native_step else "LOCOMOTION_DEFAULT",transit_policy_050=bool(transit_policy_050),reverse_moving_arc=bool(reverse_moving_arc),away_1m=bool(away_1m),allow_carried_bottle_tilt=bool(allow_carried_bottle_tilt)),graph_startup_timeout_s=5.,
                 allowed_contact_pairs=[["test_cylinder",p] for p in ("R_hand_narrow3_Link","R_hand_wide3_Link")])
    return script+"\nMICRO.update("+repr(options)+")\nrun_probe()\n"



def recovered_turn_stop(mission, row, commands, now, heading_tolerance):
    """Reuse a measured recovery stop only before any new nonzero command."""
    cached=mission.get("clock_recovery_turn_stop")
    if not cached or not mission.get("turn_source_and_mode_observed"):return None
    if cached["stop"].get("pass") is not True or row["native_generation"]!=cached["generation"]:return None
    if not 0.<=now-cached["completed_s"]<=.20:return None
    if any(any(c.get(k,0.) for k in ("forward","lateral","angular"))
           for c in commands[cached["command_index"]:]):return None
    if (row["common_mode_desc"]!="STAND_DEFAULT" or row["get_mode_desc"]!="STAND_DEFAULT"
        or row["get_mode_header_code"]!=0 or row["common_mode_status_value"]!=100
        or row["get_mode_status_value"]!=100 or max(row["state_age_s"].values())>.20):return None
    if abs(mission["target_unwrapped_yaw"]-mission["unwrapped_yaw"])>heading_tolerance:return None
    return cached["stop"]


def recoverable_carry_clock_fault(fault, error, recovery_count):
    return ((str(error).startswith("NATIVE_FEEDBACK_INVALID:") or str(error)=="CLOCK_GENERATION_PENDING") and
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
    gate=result.get("post_stop_composite");scope="FINAL"
    turn_gate=result.get("turn_stop_composite")
    if gate is None and isinstance(turn_gate,dict) and turn_gate.get("pass") is True:
        gate_end=turn_gate.get("metrics",{}).get("window_end_monotonic_s")
        if isinstance(gate_end,(int,float)) and (last_nonzero is None or last_nonzero<=gate_end):
            gate=turn_gate;scope="TURN_ONLY"

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
        independent_composite_stop=gate_status,composite_stop_scope=scope,
        route_acceptance_recorded=bool(gate_status=="PASS" and result.get("pass_result") is True
                                      and not result.get("failure")),
        rearm_authorized=False)


def start_carry_graph_check(n):
    """Keep the same readonly graph verification off the command scheduler."""
    import threading,time
    job=dict(started=time.monotonic(),completed=None,error=None)
    def worker():
        try:n.graph()
        except Exception as exc:
            job["error"]=str(exc)
            n.deadline_zero()
        finally:job["completed"]=time.monotonic()
    job["thread"]=threading.Thread(target=worker,daemon=True)
    job["thread"].start()
    return job


def poll_carry_graph_check(n,job,now):
    if job is None:return None
    end=job["completed"]
    if job["error"] is not None:
        n.deadline_zero();raise RuntimeError("CARRY_GRAPH_CHECK_FAILED:"+job["error"])
    if (end if end is not None else now)-job["started"]>.20:
        n.deadline_zero();raise RuntimeError("CARRY_GRAPH_CHECK_DEADLINE")
    if end is not None:
        n.events.append(dict(event="ASYNC_GRAPH_CHECK_COMPLETE",started=job["started"],completed=end,
                             duration_s=end-job["started"]))
        return None
    return job


def run_carry_navigation(n,contract):
    """One bounded measured-pose task, with zero/qualified-stop clock recovery."""
    import threading
    from types import SimpleNamespace
    from point_goal_controller import PointGoalController,ControllerConfig,Pose2D,Goal2D,NavState
    cfg=MICRO["carry"];result=dict(classification="SIMULATION_ONLY_NATIVE_LOCOMOTION_STEP_CARRY" if cfg.get("motion_mode")=="LOCOMOTION_STEP" else "SIMULATION_ONLY_FETCH_CARRY_ROUTE",pass_result=False,
        goal_xy_m=cfg["goal_xy_m"],empty_bottle_tilt_allowed=bool(cfg.get("allow_carried_bottle_tilt",False)),zero_receipt_ack="UNAVAILABLE",clock_recoveries=[],route_samples=[])
    if cfg.get("transit_policy_050"):result["classification"]="SIMULATION_ONLY_NATIVE_STEP_CARRY_TRANSIT_050" if cfg.get("motion_mode")=="LOCOMOTION_STEP" else "SIMULATION_ONLY_FOUNDATIONWALK_CARRY_TRANSIT_050"
    if cfg.get("reverse_moving_arc"):result.update(classification="SIMULATION_ONLY_REVERSE_MOVING_ARC_CARRY",goal_yaw_requirement="NOT_REQUIRED_FOR_KNOWN_CHAIR_AREA")
    if cfg.get("away_1m"):result["classification"]="SIMULATION_ONLY_LEAVE_TURN_FORWARD_ONE_METRE"
    if cfg.get("straight_retreat_turn"):result["classification"]="SIMULATION_ONLY_STRAIGHT_RETREAT_AND_INPLACE_TURN_PREFIX"
    result["activation_protocol"]="YAW_PRESTEP_EXPERIMENT" if cfg.get("gait_prepare") else "VENDOR_DIRECT_RETREAT_NONZERO"
    watchdog=None;began=None;baseline=None;transit_permit=None;graph_job=None;closed_permits=[];mission=None
    def stand_transition():
        strategy=(cfg.get("delivery") or {}).get("stand_transition_strategy","ZERO_WINDOW_THEN_SD")
        result["stand_transition_strategy"]=strategy
        if strategy=="EARLY_SD_CONTINUOUS_ZERO_DIAGNOSTIC":
            # Caller has already issued deadline-zero and disabled our source.
            # SD acceptance is NOT stop confirmation. Retain >=1.05s of zero
            # publication afterwards and the independent composite observation.
            n.stand_once();n.observe(1.05,baseline=baseline)
        else:
            n.observe(1.05,baseline=baseline);n.stand_once()

    try:
        n.start_feedback();until=time.monotonic()+5.
        while time.monotonic()<until:
            n.publish();n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:break
            time.sleep(.01)
        n.await_graph();n.pump();baseline=n.sample()["body_status"]
        result["pre_motion_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        start=n.sample()["pose"];result["start_pose"]=start
        goal=cfg["goal_xy_m"]
        if cfg.get("away_1m"):
            away=(start["yaw"]-math.pi+math.pi)%(2*math.pi)-math.pi
            mission=dict(phase="PRESTEP" if cfg.get("gait_prepare") else "RETREAT" if cfg.get("straight_retreat_turn") else "TURN",away_heading_rad=away,unwrapped_yaw=start["yaw"],
                         target_unwrapped_yaw=start["yaw"]-math.pi,previous_yaw=start["yaw"],
                         previous_source_s=None,turn_source_and_mode_observed=False)
            result["away_mission"]=mission
            bearing=away
        else:bearing=math.atan2(goal[1]-start["y"],goal[0]-start["x"])
        if cfg.get("motion_mode")=="LOCOMOTION_STEP" and not cfg.get("native_step_all_transit"):
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
        controller=PointGoalController(ControllerConfig(xy_tolerance_m=.01 if cfg.get("away_1m") else .10 if cfg.get("reverse_moving_arc") else .15,yaw_tolerance_rad=.12,max_forward_mps=.12,max_angular_rps=cfg.get("angular_command_cap",.075),k_angular=.8,goal_timeout_s=cfg["timeout_s"]))
        target=None if cfg.get("away_1m") else Goal2D(*goal,bearing,"native_scene_world")
        began=time.monotonic();until=began+cfg["timeout_s"];last_graph=began;acquired=False;activation_started=began;progress_state=None;progress_best=None;progress_at=began
        if cfg.get("away_1m"):cfg["absolute_command_deadline_ns"]=int(until*1e9)
        if cfg.get("transit_policy_050"):transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
        with n.lock:n.moving=True
        watchdog=threading.Timer(max(.001,until-time.monotonic()),n.deadline_zero);watchdog.daemon=True;watchdog.start()
        while time.monotonic()<until:
            if Path("/workspace/evidence/human_review_stop").exists():raise RuntimeError("HUMAN_REVIEW_STOP")
            tick=time.monotonic()
            if not n.moving and (n.feedback_fault or {}).get("gate")!="CLOCK_STEP_DURING_MOTION":break
            graph_job=poll_carry_graph_check(n,graph_job,tick)
            try:
                n.pump();row=n.sample()
            except RuntimeError as exc:
                fault=n.feedback_fault or {}
                # Only a qualified source-clock epoch fault is recoverable.
                if (cfg.get("transit_policy_050") and not cfg.get("compact_turn")) or (cfg.get("motion_mode")=="LOCOMOTION_STEP" and not cfg.get("compact_turn")) or not recoverable_carry_clock_fault(fault,exc,len(result["clock_recoveries"])):raise
                n.deadline_zero();n.stop_started=True
                if cfg.get("compact_turn"):transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
                if n.moving:raise RuntimeError("CLOCK_RECOVERY_STILL_MOVING")
                n.feedback_fault=None
                n.source_enabled(False) if cfg.get("release_source_on_stop") else None
                stand_transition()
                stopped=n.observe(8.,baseline=baseline,require_stop=True)
                result["clock_recoveries"].append(dict(fault=fault,stop=stopped,t=time.monotonic()))
                controller.reset();n.await_graph();n.pump()
                if cfg.get("compact_turn"):
                    transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                    closed_permits.append(transit_permit);transit_permit=None
                    cfg["previous_permit_filename"]=cfg.get("permit_filename","carry_transit_permit.json")
                    cfg["permit_filename"]="carry_transit_clock_recovery_"+str(len(result["clock_recoveries"]))+".json"
                    cfg["clock_stop_pass"]=bool(stopped["pass"])
                    if cfg.get("motion_mode")=="LOCOMOTION_STEP" and not cfg.get("native_step_all_transit"):
                        n.stand_once("LOCOMOTION_STEP")
                        prepared=False;prepare_until=time.monotonic()+.75
                        while time.monotonic()<prepare_until:
                            n.publish();n.pump();state=n.sample()
                            if native_step_activation_ready(state,baseline):prepared=True;break
                            time.sleep(.02)
                        if not prepared:raise RuntimeError("CLOCK_RECOVERY_STEP_PREPARATION_FAILED")
                    if until-time.monotonic()<5.:raise RuntimeError("CLOCK_RECOVERY_TASK_DEADLINE_INSUFFICIENT")
                    transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                    if mission is not None and mission["phase"]=="PRESTEP":
                        pose=n.sample()["pose"]
                        result["clock_recoveries"][-1]["prestep_restart_from_measured_stop"]=pose
                        result["clock_recoveries"][-1]["previous_prestep_load_sequence"]=mission.get("prestep_load_sequence",[])
                        mission["prestep_origin_pose"]=pose
                        mission["prestep_started_s"]=time.monotonic()
                        mission["prestep_load_sequence"]=[]
                    if mission is not None and mission["phase"]=="TURN":
                        pose=n.sample()["pose"]
                        delta=(pose["yaw"]-mission["previous_yaw"]+math.pi)%(2*math.pi)-math.pi
                        mission["unwrapped_yaw"]+=delta
                        mission["previous_yaw"]=pose["yaw"];mission["previous_source_s"]=None
                        mission["clock_recovery_turn_stop"]=dict(stop=stopped,completed_s=time.monotonic(),
                            command_index=len(n.command_log),generation=n.latest_row["generation"])
                    if cfg.get("release_source_on_stop"):n.source_enabled(True)
                    acquired=False
                n.stop_started=False
                with n.lock:n.moving=True
                activation_started=time.monotonic()
                progress_state=None;progress_best=None;progress_at=activation_started
                continue
            if row["body_status"]!=baseline:raise RuntimeError("BODY_STATUS_CHANGED")
            if cfg.get("native_step_all_transit") and cfg.get("step_prepared_permit")!=cfg.get("permit_filename","carry_transit_permit.json"):
                n.stand_once("LOCOMOTION_STEP")
                prepared=False;prepare_until=min(until,time.monotonic()+.75)
                while time.monotonic()<prepare_until:
                    n.publish();n.pump();prepared_state=n.sample()
                    result.setdefault("native_step_transitions",[]).append({k:prepared_state[k] for k in ("sample_monotonic_s","get_mode_desc","common_mode_desc","get_mode_status_value","common_mode_status_value","body_status","raw_common")})
                    if native_step_activation_ready(prepared_state,baseline):prepared=True;break
                    time.sleep(.02)
                if not prepared:raise RuntimeError("NATIVE_STEP_TRANSIT_PREPARATION_FAILED")
                cfg["step_prepared_permit"]=cfg.get("permit_filename","carry_transit_permit.json")
                activation_started=time.monotonic()
                continue
            p=row["pose"]
            if not -1.65<=p["x"]<=1.65 or not -1.75<=p["y"]<=1.75:raise RuntimeError("CARRY_ROOM_WORKSPACE")
            if mission is not None:
                if cfg.get("gait_prepare"):
                    foot_forces,dominant=native_foot_loads(n.latest_row)
                    observed=mission.setdefault("observed_support_changes",[])
                    if dominant and (not observed or observed[-1]["side"]!=dominant):
                        observed.append(dict(side=dominant,t=tick,source_host_ns=n.latest_row["host_monotonic_ns"],force_n=foot_forces))
                if mission["phase"]=="PRESTEP":
                    prep_origin=mission.get("prestep_origin_pose",start)
                    distance=math.hypot(p["x"]-prep_origin["x"],p["y"]-prep_origin["y"])
                    yaw_delta=(p["yaw"]-prep_origin["yaw"]+math.pi)%(2*math.pi)-math.pi
                    if distance>.15 or abs(yaw_delta)>.20:raise RuntimeError("GAIT_PREPARATION_POSE_BOUND")
                    if tick-mission.get("prestep_started_s",began)>2.:raise RuntimeError("GAIT_PREPARATION_NOT_OBSERVED")
                    forces,dominant=native_foot_loads(n.latest_row)
                    supports=mission.setdefault("prestep_load_sequence",[])
                    if dominant and (not supports or supports[-1]["side"]!=dominant):
                        supports.append(dict(side=dominant,t=tick,source_host_ns=n.latest_row["host_monotonic_ns"],force_n=forces))
                    required=(cfg.get("delivery") or {}).get("gait_prepare_required_load_states",2)
                    knee_motion=max(abs(n.latest_row["joints"][name]["qvel"]) for name in ("left_knee_joint","right_knee_joint"))
                    mission["prestep_actual_knee_motion_rad_s"]=knee_motion
                    if len(supports)>=required and acquired and (required==2 or knee_motion>.20):
                        mission["prestep_end_pose"]=p;mission["retreat_origin_pose"]=p
                        mission["phase"]="RETREAT";progress_state=None;progress_best=None;progress_at=tick
                        continue
                    prep_angular=(cfg.get("delivery") or {}).get("gait_prepare_angular_rad_s",.10)
                    output=SimpleNamespace(state=NavState.ALIGN,distance_m=None,yaw_error_rad=prep_angular,linear_x=0.,angular_z=prep_angular)
                elif mission["phase"]=="RETREAT":
                    origin=mission.get("retreat_origin_pose",start)
                    dx=p["x"]-origin["x"];dy=p["y"]-origin["y"]
                    progress=-math.cos(origin["yaw"])*dx-math.sin(origin["yaw"])*dy
                    cross=-math.sin(origin["yaw"])*dx+math.cos(origin["yaw"])*dy
                    mission["retreat_progress_m"]=progress
                    mission["retreat_max_cross_track_m"]=max(mission.get("retreat_max_cross_track_m",0.),abs(cross))
                    corridor=float((cfg.get("delivery") or {}).get("retreat_cross_track_limit_m",.05))
                    mission["retreat_task_corridor_m"]=corridor
                    mission["legacy_instantaneous_50mm_cross_track_pass"]=mission["retreat_max_cross_track_m"]<=.05
                    if abs(cross)>corridor:raise RuntimeError("RETREAT_CROSS_TRACK")
                    error=(origin["yaw"]-p["yaw"]+math.pi)%(2*math.pi)-math.pi
                    history=mission.setdefault("retreat_progress_history",[])
                    if not history or row["sample_monotonic_s"]>history[-1][0]:
                        history.append([row["sample_monotonic_s"],progress])
                    history[:]=[x for x in history if row["sample_monotonic_s"]-x[0]<=.4]
                    velocity=0.
                    if len(history)>2 and history[-1][0]-history[0][0]>=.25:
                        xs=[x[0]-history[0][0] for x in history];ys=[x[1] for x in history]
                        mx=sum(xs)/len(xs);my=sum(ys)/len(ys);var=sum((x-mx)**2 for x in xs)
                        velocity=max(0.,sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var) if var else 0.
                    predicted=progress+velocity*cfg.get("retreat_brake_delay_s",0.)
                    mission["retreat_predicted_settled_progress_m"]=predicted
                    retreat_goal=.50 if cfg.get("compact_turn") else .30
                    if progress<retreat_goal and not (progress>=.03 and predicted>=retreat_goal):
                        output=SimpleNamespace(state=NavState.DRIVE,distance_m=retreat_goal-progress,
                                               yaw_error_rad=error,linear_x=-.12,
                                               angular_z=(0. if abs(error)<=.02 else math.copysign(min(.30 if cfg.get("compact_turn") else .1,cfg["angular_command_cap"]),error)))
                    else:
                        if not acquired:raise RuntimeError("RETREAT_SOURCE_NOT_CONFIRMED")
                        n.deadline_zero();n.stop_started=True
                        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
                        n.source_enabled(False) if cfg.get("release_source_on_stop") else None
                        stand_transition()
                        result["retreat_stop_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
                        n.await_graph();n.pump();stopped=n.sample()
                        mission["retreat_end_pose"]=stopped["pose"]
                        transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                        closed_permits.append(transit_permit);transit_permit=None
                        cfg["motion_mode"]=cfg.get("inplace_motion_mode","LOCOMOTION_STEP")
                        state=stopped;prepared=cfg["motion_mode"]=="LOCOMOTION_DEFAULT" or bool(cfg.get("native_step_all_transit"))
                        mission["turn_mode"]=cfg["motion_mode"]
                        if cfg["motion_mode"]=="LOCOMOTION_STEP":
                            n.stand_once("LOCOMOTION_STEP")
                            ready_until=time.monotonic()+.75
                            while time.monotonic()<ready_until:
                                n.publish();n.pump();state=n.sample()
                                result.setdefault("inplace_mode_preparation",[]).append({k:state[k] for k in ("sample_monotonic_s","get_mode_desc","common_mode_desc","common_mode_status_value","pose")})
                                if native_step_activation_ready(state,baseline):prepared=True;break
                                time.sleep(.02)
                        if not prepared:raise RuntimeError("INPLACE_STEP_PREPARATION_FAILED")
                        mission["phase"]="TURN";mission["turn_origin_pose"]=state["pose"]
                        # The stopped retreat pose defines a new physical turn;
                        # do not interpret intervening braking yaw as a timestamp jump.
                        mission["unwrapped_yaw"]=state["pose"]["yaw"]
                        mission["previous_yaw"]=state["pose"]["yaw"]
                        mission["previous_source_s"]=state["sample_monotonic_s"]
                        mission["target_unwrapped_yaw"]=state["pose"]["yaw"]-math.pi
                        mission["turn_start_source_s"]=state["sample_monotonic_s"]
                        mission["away_heading_rad"]=(state["pose"]["yaw"]-math.pi+math.pi)%(2*math.pi)-math.pi
                        if (cfg.get("delivery") or {}).get("turn_to_delivery_bearing"):
                            bearing,mission["target_unwrapped_yaw"]=table_turn_target(state["pose"],cfg["delivery"]["base_goal_world_xy"])
                            mission["away_heading_rad"]=bearing;mission["turn_goal"]="ACTUAL_SECOND_TABLE_BEARING"
                        bearing=mission["away_heading_rad"]
                        origin=mission.get("retreat_origin_pose",start)
                        ex=state["pose"]["x"]-origin["x"];ey=state["pose"]["y"]-origin["y"]
                        mission["retreat_settled_progress_m"]=-math.cos(origin["yaw"])*ex-math.sin(origin["yaw"])*ey
                        mission["retreat_settled_cross_track_m"]=-math.sin(origin["yaw"])*ex+math.cos(origin["yaw"])*ey

                        cfg["permit_filename"]="carry_transit_forward_permit.json";cfg["turn_stop_pass"]=True
                        transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                        if cfg.get("release_source_on_stop"):n.source_enabled(True)
                        acquired=False;activation_started=time.monotonic()
                        progress_state=None;progress_best=None;progress_at=activation_started
                        n.stop_started=False
                        with n.lock:n.moving=True
                        continue
                elif mission["phase"]=="TURN":
                    source=row["sample_monotonic_s"]
                    if mission["previous_source_s"] is None or source>mission["previous_source_s"]:
                        delta=(p["yaw"]-mission["previous_yaw"]+math.pi)%(2*math.pi)-math.pi
                        if abs(delta)>.20:raise RuntimeError("AWAY_TURN_YAW_GENERATION_DISCONTINUITY")
                        mission["unwrapped_yaw"]+=delta
                        mission["previous_yaw"]=p["yaw"];mission["previous_source_s"]=source
                        history=mission.setdefault("turn_yaw_history",[])
                        history.append([source,mission["unwrapped_yaw"]])
                        history[:]=[x for x in history if source-x[0]<=1.5]
                    if (cfg.get("delivery") or {}).get("turn_to_delivery_bearing"):
                        bearing,_=table_turn_target(p,cfg["delivery"]["base_goal_world_xy"])
                        mission["target_unwrapped_yaw"]=mission["unwrapped_yaw"]+(bearing-p["yaw"]+math.pi)%(2*math.pi)-math.pi
                        mission["away_heading_rad"]=bearing
                    turn_error=mission["target_unwrapped_yaw"]-mission["unwrapped_yaw"]
                    history=mission.get("turn_yaw_history",[])
                    reserve=0.;rate=0.
                    if len(history)>2 and history[-1][0]-history[0][0]>=1.:
                        xs=[x[0]-history[0][0] for x in history];ys=[x[1] for x in history]
                        mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
                        variance=sum((x-mx)**2 for x in xs)
                        rate=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/variance if variance else 0.
                        if rate*turn_error>0:
                            reserve=min(.75,abs(rate)*cfg.get("turn_brake_delay_s",0.))
                    profile_reserve=float((cfg.get("delivery") or {}).get("intermediate_turn_stop_reserve_rad",0.))
                    reserve=max(reserve,profile_reserve)
                    mission["measured_yaw_rate_rad_s"]=rate
                    mission["profile_observed_stop_reserve_rad"]=profile_reserve
                    mission["predicted_post_zero_yaw_rad"]=math.copysign(reserve,turn_error if profile_reserve else rate)
                    mission["turn_max_center_excursion_m"]=max(mission.get("turn_max_center_excursion_m",0.),math.hypot(p["x"]-mission.get("turn_origin_pose",start)["x"],p["y"]-mission.get("turn_origin_pose",start)["y"]))
                    if cfg.get("compact_turn") and not cfg.get("incidental_contact_diagnostic") and mission["turn_max_center_excursion_m"]>.30:raise RuntimeError("COMPACT_TURN_RADIUS_300MM")
                    centre_origin=mission.get("turn_origin_pose",start)
                    centre_error=math.hypot(p["x"]-centre_origin["x"],p["y"]-centre_origin["y"])
                    if cfg.get("inplace_centre_hold"):
                        if centre_error>=.025:mission["turn_centre_recovery"]=True
                        elif centre_error<=.01:mission["turn_centre_recovery"]=False
                    recovering=cfg.get("inplace_centre_hold") and mission.get("turn_centre_recovery",False)
                    if abs(turn_error)>max(.08,reserve) or recovering:
                        if abs(turn_error)>math.pi+.10:raise RuntimeError("CARRY_HEADING_RESPONSE_DIVERGED")
                        output=SimpleNamespace(state=NavState.ALIGN,distance_m=None,yaw_error_rad=turn_error,
                                               linear_x=0. if cfg.get("straight_retreat_turn") else -.12,angular_z=math.copysign(min(.30,cfg["angular_command_cap"]) if cfg.get("compact_turn") and abs(turn_error)<float((cfg.get("delivery") or {}).get("turn_terminal_angular_taper_angle_rad",.60)) else cfg["angular_command_cap"],turn_error) if cfg.get("straight_retreat_turn") else max(-cfg["angular_command_cap"],min(cfg["angular_command_cap"],.8*turn_error)))
                        if cfg.get("compact_centre_feedback") and not (cfg.get("delivery") or {}).get("turn_translation_strategy")=="PURE_YAW":
                            terminal_angle=float((cfg.get("delivery") or {}).get("turn_terminal_translation_free_angle_rad",0.))
                            if abs(turn_error)<terminal_angle:
                                output.linear_x=0.;output.lateral_y=0.
                                mission["terminal_turn_translation_free_ticks"]=mission.get("terminal_turn_translation_free_ticks",0)+1
                            else:
                                output.linear_x,output.lateral_y=inplace_centre_command(p,mission["turn_origin_pose"],.20,forward_only=True)
                                mission["combined_xy_feedback_ticks"]=mission.get("combined_xy_feedback_ticks",0)+1
                        if cfg.get("inplace_centre_hold"):
                            output.linear_x,output.lateral_y=(0.,0.) if cfg.get("turn_recovery_speed",.12)>.12 else inplace_centre_command(p,mission["turn_origin_pose"])
                            if recovering:
                                output.linear_x,output.lateral_y=inplace_recovery_command(p,mission["turn_origin_pose"],cfg.get("turn_recovery_speed",.12))
                                output.angular_z=0.;output.state=NavState.DRIVE;output.distance_m=centre_error
                                mission["centre_recovery_ticks"]=mission.get("centre_recovery_ticks",0)+1
                    else:
                        recovery_stop=recovered_turn_stop(mission,row,n.command_log,time.monotonic(),
                            float((cfg.get("delivery") or {}).get("intermediate_turn_heading_tolerance_rad",.12)))
                        if not acquired and recovery_stop is None:raise RuntimeError("AWAY_TURN_SOURCE_NOT_CONFIRMED")
                        n.deadline_zero();n.stop_started=True
                        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
                        n.source_enabled(False) if cfg.get("release_source_on_stop") else None
                        if recovery_stop is not None:
                            result["turn_stop_composite"]=recovery_stop
                            mission["turn_completion_used_measured_clock_recovery_stop"]=True
                        else:
                            stand_transition()
                            result["turn_stop_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
                        if result["turn_stop_composite"].get("pass") is not True:raise RuntimeError("AWAY_TURN_STOP_NOT_QUALIFIED")
                        if cfg.get("review_prefix_only"):
                            n.pump();review_end=n.sample()["pose"]
                            result.update(classification="SIMULATION_ONLY_LIVE_RETREAT_TURN_REVIEW_NOT_TRANSFER_ACCEPTANCE",review_capture_complete=True,pass_result=True,failure=None,end_pose=review_end,post_stop_composite=result["turn_stop_composite"])
                            mission["turn_center_final_drift_m"]=math.hypot(review_end["x"]-mission["turn_origin_pose"]["x"],review_end["y"]-mission["turn_origin_pose"]["y"])
                            mission["turn_stop_heading_error_rad"]=(bearing-review_end["yaw"]+math.pi)%(2*math.pi)-math.pi
                            return result
                        n.await_graph();n.pump();stopped=n.sample()
                        heading_error=(bearing-stopped["pose"]["yaw"]+math.pi)%(2*math.pi)-math.pi
                        mission["turn_stop_heading_error_rad"]=heading_error
                        mission["turn_end_pose"]=stopped["pose"]
                        heading_tolerance=float(cfg["delivery"].get("intermediate_turn_heading_tolerance_rad",.12)) if cfg.get("delivery") else .12
                        mission["intermediate_heading_tolerance_rad"]=heading_tolerance
                        mission["legacy_strict_heading012_pass"]=abs(heading_error)<=.12
                        if abs(heading_error)>heading_tolerance:raise RuntimeError("AWAY_TURN_STOP_HEADING_NOT_QUALIFIED")
                        mission["turn_active_unwrapped_change_rad"]=mission["unwrapped_yaw"]-(mission["turn_origin_pose"]["yaw"] if cfg.get("straight_retreat_turn") else start["yaw"])
                        tail_delta=(stopped["pose"]["yaw"]-mission["previous_yaw"]+math.pi)%(2*math.pi)-math.pi
                        mission["turn_unwrapped_change_rad"]=mission["turn_active_unwrapped_change_rad"]+tail_delta
                        mission["turn_stop_heading_error_rad"]=heading_error
                        if cfg.get("straight_retreat_turn"):
                            origin=mission["turn_origin_pose"]
                            drift=math.hypot(stopped["pose"]["x"]-origin["x"],stopped["pose"]["y"]-origin["y"])
                            mission["turn_center_final_drift_m"]=drift
                            result["end_pose"]=stopped["pose"]
                            result["post_stop_composite"]=result["turn_stop_composite"]
                            result["source_and_motion_mode_observed"]=mission["turn_source_and_mode_observed"]
                            result["forward_one_metre"]="NOT_RUN"
                            result["delivery_table_placement"]="NOT_RUN"
                            bound=.20 if cfg.get("compact_turn") else .05
                            if cfg.get("delivery"):bound=float(cfg["delivery"].get("turn_braking_footprint_m",.30))
                            mission["final_centre_tolerance_m"]=bound
                            turn_rows=[r for r in n.trace if r["sample_monotonic_s"]>=mission.get("turn_start_source_s",0.)]
                            peak=max([drift]+[math.hypot(r["pose"]["x"]-origin["x"],r["pose"]["y"]-origin["y"]) for r in turn_rows])
                            mission["turn_and_braking_peak_radius_m"]=peak
                            extension=(cfg.get("delivery") or {}).get("terminal_braking_aisle_reserve")
                            radial_braking_bound=float(cfg["delivery"].get("turn_braking_footprint_m",.30)) if cfg.get("delivery") else .30
                            footprint=table_terminal_braking_footprint([r["pose"] for r in turn_rows],origin,bearing,radial_braking_bound,extension)
                            mission["terminal_braking_footprint"]=footprint
                            if cfg.get("compact_turn") and not footprint["pass_clearance"]:raise RuntimeError("COMPACT_TURN_BRAKING_RADIUS_TASK_LIMIT")
                            if not (extension and footprint["pass_clearance"]):
                                if drift>bound:raise RuntimeError("INPLACE_TURN_CENTER_TRANSLATION:"+str(drift))
                            result["retreat_turn_prefix_pass"]=True
                            if cfg.get("delivery"):
                                transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                                closed_permits.append(transit_permit);transit_permit=None
                                cfg["previous_permit_filename"]=cfg["permit_filename"]
                                cfg["permit_filename"]="carry_transit_delivery_permit.json";cfg["delivery_turn_stop_pass"]=True
                                egress=float(cfg["delivery"].get("post_turn_egress_m",0.))
                                if egress:
                                    goal=[stopped["pose"]["x"]+(egress+.10)*math.cos(bearing),stopped["pose"]["y"]+(egress+.10)*math.sin(bearing)]
                                    target=Goal2D(*goal,bearing,"native_scene_world");mission["phase"]="EGRESS"
                                    mission["egress_origin_pose"]=stopped["pose"];mission["egress_heading_rad"]=bearing
                                else:
                                    goal=list(cfg["delivery"]["base_goal_world_xy"])
                                    target=Goal2D(*goal,float(cfg["delivery"]["base_goal_world_yaw_rad"]),"native_scene_world");mission["phase"]="DELIVERY"
                                result.pop("post_stop_composite",None);result.pop("end_pose",None)
                                controller=PointGoalController(ControllerConfig(xy_tolerance_m=.01 if egress else .10,yaw_tolerance_rad=.12,max_forward_mps=float(cfg["delivery"].get("delivery_forward_command_m_s",.20)),max_angular_rps=.10 if egress else float(cfg["delivery"].get("delivery_tracking_angular_cap_rad_s",cfg["angular_command_cap"])),align_tolerance_rad=float(cfg["delivery"].get("intermediate_turn_heading_tolerance_rad",.20)),k_angular=.8,goal_timeout_s=max(5.,until-time.monotonic())))
                                mission["delivery_origin_pose"]=stopped["pose"]
                                mission["delivery_target_world_xy"]=goal;cfg["transit_command_speed"]=float(cfg["delivery"].get("delivery_forward_command_m_s",.20))
                                if (cfg["delivery"].get("carry_native_neutral_stow") or {}).get("follow_native_unloaded_arm_after_turn"):
                                    write_durable_milestone("/workspace/evidence/carry_unloaded_arm_follow.json",dict(owner_pid=cfg["pid"],generation=n.latest_row["generation"],created_ns=time.monotonic_ns(),deadline_ns=int((until+15.)*1e9),turn_and_composite_stop_pass=True))
                                transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                                n.source_enabled(True);n.stop_started=False;acquired=False;activation_started=time.monotonic()
                                progress_state=None;progress_best=None;progress_at=activation_started
                                with n.lock:n.moving=True
                                continue
                            result["pass_result"]=True;result["failure"]=None
                            return result
                        transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                        closed_permits.append(transit_permit);transit_permit=None
                        if time.monotonic()>=until-5.:raise RuntimeError("AWAY_FORWARD_AGGREGATE_DEADLINE_INSUFFICIENT")
                        graph_job=poll_carry_graph_check(n,graph_job,time.monotonic())
                        mission["phase"]="FORWARD";origin=stopped["pose"]
                        mission["forward_origin_pose"]=origin
                        mission["forward_acceptance_xy_m"]=[origin["x"]+math.cos(bearing),origin["y"]+math.sin(bearing)]
                        goal=[origin["x"]+1.10*math.cos(bearing),origin["y"]+1.10*math.sin(bearing)]
                        result["forward_controller_target_xy_m"]=goal
                        target=Goal2D(*goal,bearing,"native_scene_world");controller.reset()
                        cfg["permit_filename"]="carry_transit_forward_permit.json";cfg["turn_stop_pass"]=True
                        transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                        n.stop_started=False;acquired=False;activation_started=time.monotonic()
                        progress_state=None;progress_best=None;progress_at=activation_started
                        with n.lock:n.moving=True
                        continue
                elif mission["phase"]=="EGRESS":
                    origin=mission["egress_origin_pose"];heading=mission["egress_heading_rad"]
                    dx=p["x"]-origin["x"];dy=p["y"]-origin["y"]
                    along=math.cos(heading)*dx+math.sin(heading)*dy
                    cross=-math.sin(heading)*dx+math.cos(heading)*dy
                    mission["egress_measured_forward_m"]=along
                    mission["egress_max_cross_track_m"]=max(mission.get("egress_max_cross_track_m",0.),abs(cross))
                    if abs(cross)>.05:raise RuntimeError("OUTWARD_EGRESS_CROSS_TRACK")
                    if along>=float(cfg["delivery"]["post_turn_egress_m"]):
                        n.deadline_zero();n.stop_started=True
                        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
                        n.source_enabled(False) if cfg.get("release_source_on_stop") else None
                        stand_transition()
                        result["egress_stop_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
                        if result["egress_stop_composite"].get("pass") is not True:raise RuntimeError("OUTWARD_EGRESS_STOP_NOT_QUALIFIED")
                        n.await_graph();n.pump();stopped=n.sample()
                        result["egress_stop_pose"]=stopped["pose"]
                        transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                        closed_permits.append(transit_permit);transit_permit=None
                        cfg["previous_permit_filename"]=cfg["permit_filename"];cfg["egress_stop_pass"]=True
                        cfg["delivery_rearm_count"]=cfg.get("delivery_rearm_count",0)+1
                        cfg["permit_filename"]="carry_transit_delivery_rearm_"+str(cfg["delivery_rearm_count"])+".json"
                        goal=list(cfg["delivery"]["base_goal_world_xy"])
                        target=Goal2D(*goal,float(cfg["delivery"]["base_goal_world_yaw_rad"]),"native_scene_world")
                        controller=PointGoalController(ControllerConfig(xy_tolerance_m=.10,yaw_tolerance_rad=.12,max_forward_mps=float(cfg["delivery"].get("delivery_forward_command_m_s",.20)),max_angular_rps=float(cfg["delivery"].get("delivery_tracking_angular_cap_rad_s",cfg["angular_command_cap"])),align_tolerance_rad=float(cfg["delivery"].get("intermediate_turn_heading_tolerance_rad",.20)),k_angular=.8,goal_timeout_s=max(5.,until-time.monotonic())))
                        mission["phase"]="DELIVERY";mission["delivery_origin_pose"]=stopped["pose"]
                        mission["delivery_target_world_xy"]=goal;mission["delivery_alignment_state"]=None
                        transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                        n.source_enabled(True);n.stop_started=False;acquired=False;activation_started=time.monotonic()
                        progress_state=None;progress_best=None;progress_at=activation_started
                        with n.lock:n.moving=True
                        continue
                    output=controller.step(Pose2D(p["x"],p["y"],p["yaw"],"native_scene_world",row["sample_monotonic_s"]),target,now_steady_s=time.monotonic())
                elif mission["phase"]=="DELIVERY":
                    distance=math.hypot(goal[0]-p["x"],goal[1]-p["y"])
                    final_yaw_error=(target.yaw-p["yaw"]+math.pi)%(2*math.pi)-math.pi
                    if distance<=.10 and abs(final_yaw_error)<=.12:
                        result["goal_triggered_stop"]=True;mission["delivery_stop_trigger_pose"]=p;break
                    origin=mission["delivery_origin_pose"];dx=goal[0]-origin["x"];dy=goal[1]-origin["y"];length=math.hypot(dx,dy)
                    along=((p["x"]-origin["x"])*dx+(p["y"]-origin["y"])*dy)/length
                    if along>length+.15:raise RuntimeError("DELIVERY_GOAL_CROSSED_WITHOUT_ARRIVAL")
                    output=controller.step(Pose2D(p["x"],p["y"],p["yaw"],"native_scene_world",row["sample_monotonic_s"]),target,now_steady_s=time.monotonic())
                    if output.state==NavState.DRIVE and mission.get("delivery_alignment_state") is not None:
                        # A turn is physically stopped before forward reactivation.
                        # Do not carry the pivot's transient base motion into DRIVE.
                        n.deadline_zero();n.stop_started=True
                        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
                        n.source_enabled(False) if cfg.get("release_source_on_stop") else None
                        stand_transition()
                        alignment_stop=n.observe(8.,baseline=baseline,require_stop=True)
                        result.setdefault("delivery_alignment_stop_composites",[]).append(alignment_stop)
                        if alignment_stop.get("pass") is not True:raise RuntimeError("DELIVERY_ALIGNMENT_STOP_NOT_QUALIFIED")
                        n.await_graph();n.pump();stopped=n.sample()
                        result.setdefault("delivery_alignment_stopped_poses",[]).append(stopped["pose"])
                        if time.monotonic()>=until-5.:raise RuntimeError("DELIVERY_REARM_DEADLINE_INSUFFICIENT")
                        transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
                        closed_permits.append(transit_permit);transit_permit=None
                        cfg["previous_permit_filename"]=cfg["permit_filename"]
                        cfg["delivery_alignment_stop_pass"]=True
                        cfg["delivery_rearm_count"]=cfg.get("delivery_rearm_count",0)+1
                        count=cfg["delivery_rearm_count"]
                        cfg["permit_filename"]="carry_transit_delivery_rearm_"+str(count)+".json"
                        transit_permit=carry_transit_phase(None,"TRANSIT",cfg,n.latest_row)
                        mission["delivery_alignment_state"]=None;controller.reset()
                        n.source_enabled(True);n.stop_started=False;acquired=False;activation_started=time.monotonic()
                        progress_state=None;progress_best=None;progress_at=activation_started
                        with n.lock:n.moving=True
                        continue

                else:
                    origin=mission["forward_origin_pose"];dx=p["x"]-origin["x"];dy=p["y"]-origin["y"]
                    along=math.cos(bearing)*dx+math.sin(bearing)*dy
                    cross=-math.sin(bearing)*dx+math.cos(bearing)*dy
                    mission["forward_projection_m"]=along
                    mission["max_forward_cross_track_m"]=max(mission.get("max_forward_cross_track_m",0.),abs(cross))
                    if abs(cross)>.05:raise RuntimeError("AWAY_FORWARD_CROSS_TRACK")
                    if along>=1.:
                        result["goal_triggered_stop"]=True;mission["forward_trigger_pose"]=p
                        mission["forward_trigger_projection_m"]=along
                        break
                    output=controller.step(Pose2D(p["x"],p["y"],p["yaw"],"native_scene_world",row["sample_monotonic_s"]),target,now_steady_s=time.monotonic())
            if cfg.get("reverse_moving_arc") and math.hypot(goal[0]-p["x"],goal[1]-p["y"])<=.10:
                result["goal_triggered_stop"]=True;result["position_only_chair_area"]=True;break
            control_now=time.monotonic()
            steering_yaw=(p["yaw"]+2*math.pi)%(2*math.pi)-math.pi if cfg.get("reverse_moving_arc") else p["yaw"]
            if mission is None:output=controller.step(Pose2D(p["x"],p["y"],steering_yaw,"native_scene_world",row["sample_monotonic_s"]),target,now_steady_s=control_now)
            if mission is not None and mission["phase"] in ("EGRESS","DELIVERY") and output.distance_m is not None:
                monitor=delivery_forward_progress(mission.get("delivery_forward_progress"),output.distance_m,tick,output.state==NavState.DRIVE and output.linear_x>0.,cfg["permit_filename"])
                mission["delivery_forward_progress"]=monitor
                if monitor is not None and monitor["no_progress"]:raise RuntimeError("DELIVERY_CUMULATIVE_FORWARD_WITHOUT_PROGRESS")
            if output.state in (NavState.ALIGN,NavState.DRIVE,NavState.FINAL_ALIGN):
                metric=output.distance_m if output.state==NavState.DRIVE else abs(output.yaw_error_rad)
                threshold=.01 if output.state==NavState.DRIVE else .02
                if progress_state!=output.state or progress_best is None or metric<progress_best-threshold:
                    progress_state=output.state;progress_best=metric;progress_at=tick
                elif tick-progress_at>3.:
                    raise RuntimeError("CARRY_COMMAND_WITHOUT_MEASURED_PROGRESS")
                # This is the existing point-goal controller, not a qualified
                # straight/curved-reference tracking run. Keep path error visible.
                if mission is None:
                    dx,dy=goal[0]-start["x"],goal[1]-start["y"];length=math.hypot(dx,dy)
                    cross=abs(dx*(p["y"]-start["y"])-dy*(p["x"]-start["x"]))/length
                    result["max_cross_track_to_direct_segment_m"]=max(result.get("max_cross_track_to_direct_segment_m",0.),cross)
                    initial_steering_yaw=(start["yaw"]+2*math.pi)%(2*math.pi)-math.pi if cfg.get("reverse_moving_arc") else start["yaw"]
                    if output.state==NavState.ALIGN and abs(output.yaw_error_rad)>abs((bearing-initial_steering_yaw+math.pi)%(2*math.pi)-math.pi)+.10:
                        raise RuntimeError("CARRY_HEADING_RESPONSE_DIVERGED")
                if cfg.get("reverse_moving_arc") and output.state==NavState.FINAL_ALIGN:raise RuntimeError("REVERSE_ARC_UNSUPPORTED_TERMINAL_ALIGNMENT")
                if mission is not None:
                    cfg["active_mission_phase"]=mission["phase"]
                    cfg["active_centre_recovery"]=mission["phase"]=="TURN" and bool(mission.get("turn_centre_recovery"))
                forward=(output.linear_x if (cfg.get("inplace_centre_hold") or cfg.get("compact_centre_feedback")) and mission["phase"]=="TURN" else 0. if cfg.get("straight_retreat_turn") and mission["phase"]=="TURN" else -cfg.get("transit_command_speed",.12) if mission["phase"] in ("RETREAT","TURN") else cfg.get("transit_command_speed",.12)) if mission is not None else -.12 if cfg.get("reverse_moving_arc") else .12 if output.linear_x>0 else 0.
                if cfg.get("inplace_centre_hold") and not cfg.get("gait_prepare") and mission is not None and mission["phase"]=="RETREAT" and cfg.get("transit_command_speed",.12)==.12:
                    forward*=bounded_start_scale(max(0.,tick-activation_started))
                    mission["retreat_input_ramp_s"]=.5
                if mission is not None and mission["phase"] in ("EGRESS","DELIVERY"):
                    forward=float(cfg["delivery"].get("delivery_forward_command_m_s",.20)) if output.linear_x>0. else 0.
                    if output.state in (NavState.ALIGN,NavState.FINAL_ALIGN):
                        if mission.get("delivery_alignment_state")!=output.state.value:
                            mission["delivery_alignment_origin"]=p;mission["delivery_alignment_state"]=output.state.value
                        pivot=mission["delivery_alignment_origin"]
                        if math.hypot(p["x"]-pivot["x"],p["y"]-pivot["y"])>.30:raise RuntimeError("DELIVERY_ALIGNMENT_RADIUS_300MM")
                        if cfg["delivery"].get("delivery_alignment_pure_yaw"):
                            forward=0.
                        else:
                            forward,_=inplace_centre_command(p,pivot,.20,forward_only=True)
                    else:mission["delivery_alignment_state"]=None
                    if mission["phase"]=="DELIVERY" and output.state==NavState.DRIVE and cfg["delivery"].get("pure_forward_direction_probe") and not mission.get("direction_probe",{}).get("qualified"):
                        probe=delivery_direction_probe(mission.get("direction_probe"),p,tick)
                        mission["direction_probe"]=probe
                        if probe.get("failure"):raise RuntimeError(probe["failure"])
                        if not probe["qualified"]:output=pure_forward_activation_output(output)
                if mission is not None and mission["phase"]=="PRESTEP":forward=0.
                lateral=getattr(output,"lateral_y",0.)
                angular=0. if cfg.get("reverse_moving_arc") and abs(output.yaw_error_rad)<=.02 else output.angular_z
                bounds=row["raw_common"]["speed_status"]
                if not bounds["forward_bounds"]["min_value"]<=forward<=bounds["forward_bounds"]["max_value"] or not bounds["lateral_bounds"]["min_value"]<=lateral<=bounds["lateral_bounds"]["max_value"] or not bounds["angular_bounds"]["min_value"]<=angular<=bounds["angular_bounds"]["max_value"]:raise RuntimeError("MC_REPORTED_INPUT_BOUNDS")
                if row["common_mode_status_value"]==300 and not ((cfg.get("native_action_smoothing") or cfg.get("vendor_transition_table")) and not acquired and 0<=tick-activation_started<=.75):raise RuntimeError("MC_TRANSITION_DURING_CARRY_COMMAND")
                handshake=(row["common_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_header_code"]==0 and row["state_age_s"]["get_mode"]<=.20 and row["common_source"]["name"]=="robotsim_official_cmd_vel")
                if handshake and (cfg.get("delivery",{}).get("carry_native_neutral_stow") or {}).get("native_arm_follow_scope")=="FULL_SOURCE" and not Path("/workspace/evidence/carry_unloaded_arm_follow.json").exists():
                    write_durable_milestone("/workspace/evidence/carry_unloaded_arm_follow.json",dict(owner_pid=cfg["pid"],generation=n.latest_row["generation"],created_ns=time.monotonic_ns(),deadline_ns=int((until+15.)*1e9),turn_and_composite_stop_pass=False,standing_and_ld_source_handoff_pass=bool(result["pre_motion_composite"]["pass"] and handshake),arm_scope="FULL_SOURCE"))
                acquired=acquired or handshake
                if handshake and mission is not None and mission["phase"]=="TURN":
                    mission["turn_source_and_mode_observed"]=True
                if tick-activation_started>.75 and not handshake:raise RuntimeError("CARRY_SOURCE_LD_HANDSHAKE_NOT_CONFIRMED")
                n.publish(forward,lateral,angular)
            elif output.state==NavState.SETTLING:n.publish()
            elif output.state==NavState.ARRIVED:
                result["goal_triggered_stop"]=True;break
            else:
                result["controller_rejection"]={"state":output.state.value,"reason":output.reason,"control_now_s":control_now,"native_source_s":row["sample_monotonic_s"]}
                raise RuntimeError("CARRY_CONTROLLER_"+output.state.value+":"+output.reason)
            result["route_samples"].append(dict(t=tick,mission_phase=mission["phase"] if mission else None,state=output.state.value,pose=p,distance_m=output.distance_m,yaw_error_rad=output.yaw_error_rad,linear=n.command_log[-1]["forward"],lateral=n.command_log[-1]["lateral"],angular=n.command_log[-1]["angular"],controller_linear=output.linear_x,controller_angular=output.angular_z))
            n.trace.append(row)
            if tick-last_graph>=1. and graph_job is None:
                graph_job=start_carry_graph_check(n);last_graph=tick
            time.sleep(max(0.,tick+.02-time.monotonic()))
        n.deadline_zero();n.stop_started=True
        transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
        if watchdog:watchdog.cancel()
        n.source_enabled(False) if cfg.get("release_source_on_stop") else None
        stand_transition()
        result["post_stop_composite"]=n.observe(8.,baseline=baseline,require_stop=True)
        end=n.sample()["pose"];result["end_pose"]=end
        acceptance=mission["forward_acceptance_xy_m"] if mission is not None and mission["phase"]=="FORWARD" else goal
        result["final_distance_m"]=math.hypot(acceptance[0]-end["x"],acceptance[1]-end["y"]) if acceptance else None
        result["measured_distance_m"]=math.hypot(end["x"]-start["x"],end["y"]-start["y"])
        result["measured_yaw_change_rad"]=(end["yaw"]-start["yaw"]+math.pi)%(2*math.pi)-math.pi
        result["source_and_motion_mode_observed"]=acquired
        if cfg.get("motion_mode","LOCOMOTION_DEFAULT")=="LOCOMOTION_DEFAULT":result["source_and_ld_observed"]=acquired
        result["pass_result"]=bool(result.get("goal_triggered_stop") and acquired and result["post_stop_composite"]["pass"] and result["final_distance_m"]<=.15 and result["measured_distance_m"]>.005)
        if mission is not None:
            if mission["phase"]=="DELIVERY":
                yaw_error=(cfg["delivery"]["base_goal_world_yaw_rad"]-end["yaw"]+math.pi)%(2*math.pi)-math.pi
                result["delivery_final_yaw_error_rad"]=yaw_error
                result["pass_result"]=bool(result["pass_result"] and abs(yaw_error)<=.12 and result.get("retreat_turn_prefix_pass") and result["retreat_stop_composite"]["pass"] and result["turn_stop_composite"]["pass"])
                mission["bilateral_support_alternation_observed"]={x["side"] for x in mission.get("observed_support_changes",[])}=={"left","right"}
                result["pass_result"]=bool(result["pass_result"] and mission["bilateral_support_alternation_observed"])
                if cfg["delivery"].get("post_turn_egress_m"):
                    result["pass_result"]=bool(result["pass_result"] and result.get("egress_stop_composite",{}).get("pass") and mission.get("egress_measured_forward_m",0.)>=cfg["delivery"]["post_turn_egress_m"])
                result["delivery_base_arrival_pass"]=result["pass_result"]
                result["classification"]="SIMULATION_ONLY_TWO_TABLE_APPROACH_NOT_PLACEMENT"
            elif mission["phase"]!="FORWARD":result["pass_result"]=False
            else:
                origin=mission["forward_origin_pose"];dx=end["x"]-origin["x"];dy=end["y"]-origin["y"]
                mission["final_forward_projection_m"]=math.cos(bearing)*dx+math.sin(bearing)*dy
                mission["final_forward_cross_track_m"]=-math.sin(bearing)*dx+math.cos(bearing)*dy
                mission["final_heading_error_rad"]=(bearing-end["yaw"]+math.pi)%(2*math.pi)-math.pi
                result["pass_result"]=bool(result["pass_result"] and mission["turn_source_and_mode_observed"]
                    and result["turn_stop_composite"]["pass"] and mission["final_forward_projection_m"]>=1.
                    and mission.get("max_forward_cross_track_m",float("inf"))<=.05
                    and abs(mission["final_forward_cross_track_m"])<=.05 and abs(mission["final_heading_error_rad"])<=.12)
        result["failure"]=None if result["pass_result"] else "CARRY_GOAL_OR_STOP_NOT_QUALIFIED"
    except Exception as exc:
        n.deadline_zero();result["failure"]=str(exc);result["abort_native_sample"]=n.latest_row
    finally:
        n.deadline_zero()
        try:transit_permit=carry_transit_phase(transit_permit,"BRAKING",cfg,n.latest_row)
        except Exception as exc:result["transit_policy_error"]=str(exc);result["pass_result"]=False
        if watchdog:watchdog.cancel()
        if cfg.get("release_source_on_stop"):
            try:n.source_enabled(False)
            except Exception as exc:result["source_disable_cleanup_failure"]=str(exc)
        end=time.monotonic()+1.05
        while time.monotonic()<end:
            n.publish()
            try:n.pump()
            except Exception:pass
            time.sleep(.015)
        if cfg.get("delivery") and not result.get("pass_result") and result.get("failure"):
            try:
                n.pump();safe=n.sample()
                if baseline is not None and safe["body_status"]==baseline:
                    last_nonzero=max((c["t"] for c in n.command_log if any(c[k] for k in ("forward","lateral","angular"))),default=0.)
                    already_requested=any(e.get("event")=="STAND_REQUEST" and e.get("t",0.)>=last_nonzero for e in n.events)
                    if not already_requested:n.stand_once()
                    brake_end=min(transit_permit["absolute_deadline_ns"],transit_permit["braking_ns"]+15000000000)*1e-9 if transit_permit else time.monotonic()+8.
                    remaining=min(8.,brake_end-time.monotonic())
                    if remaining>=2.:result["abort_post_stop_composite"]=n.observe(remaining,baseline=baseline,require_stop=True)
            except Exception as exc:result["abort_stop_observation_failure"]=str(exc)
        try:transit_permit=carry_transit_phase(transit_permit,"REVOKED",cfg,n.latest_row)
        except Exception as exc:result["transit_policy_error"]=str(exc);result["pass_result"]=False
        result["transit_policy_record"]=transit_permit
        result["transit_policy_records"]=closed_permits+([transit_permit] if transit_permit is not None else [])
        if graph_job is not None:
            graph_job["thread"].join(.20)
            if graph_job["thread"].is_alive():
                result["graph_cleanup_failure"]="GRAPH_THREAD_NOT_JOINED";result["pass_result"]=False
        try:n.stop_feedback()
        except Exception as exc:result["feedback_cleanup_failure"]=str(exc);result["pass_result"]=False
        result.update(events=n.events,command_log=n.command_log,bridge_commands=n.bridge_commands,trace=n.trace,receipt_diagnostics=getattr(n,"receipt_diagnostics",{}))
        try:result["stop_evidence"]=carry_stop_evidence(result)
        except Exception as exc:
            result["stop_evidence_error"]=str(exc);result["pass_result"]=False
        write_durable_milestone("/workspace/evidence/native_carry_route_result.json",result)
        n.destroy_node();rclpy.shutdown()
        print(json.dumps({k:v for k,v in result.items() if k not in ("events","command_log","bridge_commands","trace","abort_native_sample","route_samples")}),flush=True)
