"""Bounded native McBodyPose compatibility probe using the existing observer."""
import ast,inspect,json
from scripts.research.issue46.native_runtime import native_base_micro_probe as micro


def prepare_publisher():
    """Compile once in the existing no-network image, with no robot processes."""
    from pathlib import Path
    import hashlib,json,subprocess
    from scripts.agent import path_governance as pg
    root=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
    source=Path(__file__).with_name("native_body_pose_publisher.c")
    mc=Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026/mc")
    library=mc/"lib/libaimdk_msgs__rosidl_typesupport_introspection_c.so"
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    image="robotsim:humble-raicom-runtime-20261019"
    image_id=subprocess.check_output(["docker","image","inspect",image,"--format","{{.Id}}"],text=True).strip()
    for path in sorted(root.glob("native-body-pose-no-physics-build-*/binary_manifest.json"),reverse=True):
        record=json.loads(path.read_text())
        if record["source"]["sha256"]==sha(source) and record["introspection_library"]["sha256"]==sha(library) and record["image_id"]==image_id and record.get("compiled_no_physics") and record.get("marker_rejection_pass") and sha(record["binary"]["path"])==record["binary"]["sha256"]:
            return path
    if subprocess.check_output(["docker","ps","-q"],text=True).strip():raise RuntimeError("compile native body tool only before simulation startup")
    out=pg.create_unique_output_directory(root,prefix="native-body-pose-no-physics-build").native_path
    pg.write_new_bytes(out/source.name,source.read_bytes())
    script=r"""set -eo pipefail
source /opt/ros/humble/setup.bash
include_dirs=()
for directory in /opt/ros/humble/include/*; do
 [[ -d "$directory" ]] || continue
 label=$(basename "$directory")
 case "$label" in dds|idl) continue;; esac
 include_dirs+=("-I$directory")
done
gcc -std=c11 -O2 -Wall -Wextra "@DOLLAR@{include_dirs[@]}" /workspace/evidence/native_body_pose_publisher.c -L/workspace/raicom2026/mc/lib -L/opt/ros/humble/lib -Wl,-rpath,/workspace/raicom2026/mc/lib -Wl,-rpath,/opt/ros/humble/lib -laimdk_msgs__rosidl_generator_c -laimdk_msgs__rosidl_typesupport_c -laimdk_msgs__rosidl_typesupport_introspection_c -lrosidl_runtime_c -lrcl -lrcutils -lm -o /workspace/evidence/native_body_pose_publisher
set +e
/workspace/evidence/native_body_pose_publisher robotsim_body_height 2 > /workspace/evidence/marker_rejection.stdout 2>&1
code=$?
test "$code" = 2
""".replace("@DOLLAR@",chr(36))
    command=["docker","run","--rm","--network","none","--cap-drop","ALL","--security-opt","no-new-privileges","--user","1000:1000","--tmpfs","/tmp:rw,nosuid,nodev,size=64m","--mount",f"type=bind,src={out},dst=/workspace/evidence","--mount",f"type=bind,src={mc},dst=/workspace/raicom2026/mc,readonly",image,"bash","-lc",script]
    result=subprocess.run(command,text=True,capture_output=True,timeout=40)
    pg.write_new_text(out/"build.stdout",result.stdout+result.stderr)
    if result.returncode:raise RuntimeError("native body publisher compilation/marker gate failed:"+str(out))
    record=dict(schema="robotsim.native-body-pose-build.v1",classification="SIMULATION_ONLY_INSTALLED_NATIVE_MCBODYPOSE",
        binary=dict(path=str(out/"native_body_pose_publisher"),sha256=sha(out/"native_body_pose_publisher")),
        source=dict(path=str(source),sha256=sha(source)),introspection_library=dict(path=str(library),sha256=sha(library)),
        image_id=image_id,compiled_no_physics=True,marker_rejection_pass=True,command=command)
    path=out/"binary_manifest.json";pg.write_new_text(path,json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(out)
    return path

def build_script(limits, robot_bodies, drop_m=.03, *, mode="roundtrip", label="native_body_height", reference_result=None):
    if drop_m not in (.03,.10,.13):raise ValueError("only measured30mm or bounded task100/130mm native-height probes are supported")
    if mode not in ("roundtrip","lower_hold","restore","observe") or label not in ("native_body_height","body_crouch_pickup","body_restore_carry","body_crouch_delivery","body_restore_final","body_posture_hold"):raise ValueError("body task mode/label")
    if mode=="restore" and reference_result is None:raise ValueError("restore requires same-episode qualified lowering proof")
    code=micro.build_script("forward",.6,limits,robot_bodies=robot_bodies)
    code=code.rsplit("\nrun_probe()",1)[0]
    tree=ast.parse(code);lines=code.splitlines(keepends=True);edits=[]
    dock=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Dock")
    init=next(n for n in dock.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    assignment=next(n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=="pub" for t in n.targets))
    edits.append((assignment.lineno,assignment.end_lineno,"  self.pub=None\n"))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
    cls=next(n for n in ast.walk(run) if isinstance(n,ast.ClassDef) and n.name=="Micro")
    publish=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="publish")
    edits.append((publish.lineno,publish.end_lineno,'        def publish(self,fwd=0.,lat=0.,ang=0.):\n            if any((fwd,lat,ang)):raise RuntimeError("BODY_OBSERVER_VELOCITY_FORBIDDEN")\n'))
    for first,last,value in sorted(edits,reverse=True):lines[first-1:last]=[value]
    code="".join(lines)
    anchor='    if MICRO.get("carry"):\n        return run_carry_navigation(n,contract)'
    if code.count(anchor)!=1:raise ValueError("existing observer routing changed")
    code=code.replace(anchor,'    if MICRO.get("body_height_probe"):\n        return run_height_probe(n,contract)\n'+anchor,1)
    code+="\n"+inspect.getsource(run_height_probe)
    code+='\nSOURCE="robotsim_body_height"\nMICRO.update('+repr(dict(body_height_probe=dict(drop_m=drop_m,max_runtime_s=30.,height_command_rate=.03,mode=mode,label=label,reference_result=reference_result),robot_bodies=robot_bodies))+')\nrun_probe()\n'
    compile(code,"existing_observer_native_body_height","exec")
    return code

def run_height_probe(n, contract):
    import os,subprocess,time,json,math
    from pathlib import Path
    from aimdk_msgs.srv import SetMcInputSource
    from rosidl_runtime_py.convert import message_to_ordereddict as raw
    import rclpy
    result=dict(classification="SIMULATION_ONLY_INSTALLED_NATIVE_BODY_POSE_COMPATIBILITY",
                requested_drop_m=MICRO["body_height_probe"]["drop_m"],velocity_publishers_created=sum(p.topic_name in ("/cmd_vel","/aima/mc/locomotion/velocity") for p in n.publishers),node_publisher_topics=[p.topic_name for p in n.publishers],motion_action_requests_sent=0,
                mc_receipt_ack="UNAVAILABLE",pass_result=False,commands=[],source_request_attempts=[],source_responses=[])
    process=None;source_mutation_attempted=False;started=time.monotonic();deadline=started+30.
    config=MICRO["body_height_probe"];rate=float(config.get("height_command_rate",.03))
    if rate not in (.03,.1):raise ValueError("BODY_RATE_PROFILE")
    result["height_command_rate"]=rate;mode=config.get("mode","roundtrip");label=config.get("label","native_body_height");result["mode"]=mode
    def request_source(action):
        nonlocal source_mutation_attempted
        q=SetMcInputSource.Request();q.request.header.stamp=n.get_clock().now().to_msg()
        q.action.value=action;q.input_source.name=SOURCE;q.input_source.priority=40;q.input_source.timeout=1000
        result["source_request_attempts"].append(dict(action=action,request=raw(q),t=time.monotonic()))
        if action in (1001,2001):source_mutation_attempted=True
        f=n.src.call_async(q);until=min(deadline,time.monotonic()+.75)
        while not f.done() and time.monotonic()<until:n.pump();time.sleep(.01)
        reply=raw(f.result()) if f.done() and f.result() is not None else None
        result["source_responses"].append(dict(action=action,raw=reply,t=time.monotonic()))
        if reply is None or reply["response"]["header"]["code"]!=0:raise RuntimeError("BODY_SOURCE_REQUEST_NOT_ACCEPTED")
    def command(value):
        if value not in (-rate,0.,rate):raise RuntimeError("BODY_RATE_BOUND")
        if process is None or process.poll() is not None:raise RuntimeError("BODY_PUBLISHER_NOT_ALIVE")
        process.stdin.write(str(value)+"\n");process.stdin.flush()
        result["commands"].append(dict(t=time.monotonic(),stand_height=value,evidence="PIPE_WRITE_NOT_MC_RECEIPT"))
    def sample():
        if time.monotonic()>=deadline:raise RuntimeError("BODY_HEIGHT_WATCHDOG")
        n.pump();row=n.sample()
        if row["get_mode_desc"]!="STAND_DEFAULT" or row["common_mode_desc"]!="STAND_DEFAULT" or row["get_mode_status_value"]!=100 or row["common_mode_status_value"]!=100 or row["get_mode_header_code"]!=0:raise RuntimeError("BODY_HEIGHT_MODE_NOT_STAND")
        if row["fsm_state"] not in contract.MC_FSM_ACTIVE_STATES or row["body_status"] not in (1,4,5):raise RuntimeError("BODY_HEIGHT_POSTURE_OR_PROTECTION")
        feet={}
        for side in ("left","right"):
            feet[side]=sum(max(0.,c["local_force_torque"][0]) for c in n.latest_row["contacts"] if c.get("efc_address",-1)>=0 and side+"_ankle_roll_link" in (c["body1"],c["body2"]) and "world" in (c["body1"],c["body2"]))
        if min(feet.values())<5.:raise RuntimeError("BODY_HEIGHT_BILATERAL_SUPPORT_LOST")
        if not n.trace or row["sample_monotonic_s"]>n.trace[-1]["sample_monotonic_s"]:n.trace.append(row)
        return row
    try:
        if result["velocity_publishers_created"]!=0 or any(t!="/parameter_events" for t in result["node_publisher_topics"]):raise RuntimeError("BODY_OBSERVER_MUST_CREATE_ZERO_ACTUATION_PUBLISHERS")
        n.start_feedback();until=min(deadline,time.monotonic()+5.);first=None
        while time.monotonic()<until:
            n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:
                try:first=sample();break
                except RuntimeError as exc:
                    if str(exc)!="CLOCK_GENERATION_PENDING":raise
            time.sleep(.01)
        if first is None:raise RuntimeError("BODY_INITIAL_FRESH_STATE_UNCONFIRMED")
        result["initial_stop"]=n.observe(5.,baseline=first["body_status"],require_stop=True)
        if not result["initial_stop"].get("pass"):raise RuntimeError("BODY_INITIAL_COMPOSITE_STOP_FAILED")
        initial=sample();height=initial["pose"]["z"];result["initial_height_m"]=height
        if mode=="observe":
            result["pass_result"]=True
            return result
        initial_reference=float(initial["raw_common"]["servo_status"]["squat_offset"]);result["initial_squat_offset"]=initial_reference
        result["generation"]=initial["native_generation"];result["native_process_pid"]=n.latest_row["process_pid"]
        if mode=="restore":
            proof=json.loads(Path(config["reference_result"]).read_text())
            if not proof.get("pass_result") or proof.get("mode")!="lower_hold" or proof.get("generation")!=initial["native_generation"] or proof.get("native_process_pid")!=n.latest_row["process_pid"]:raise RuntimeError("BODY_RESTORE_PROOF_NOT_SAME_QUALIFIED_EPISODE")
            height=float(proof["initial_height_m"]);initial_reference=float(proof["initial_squat_offset"]);result["restore_target_height_m"]=height
        if n.get_publishers_info_by_topic("/aima/mc/locomotion/velocity"):raise RuntimeError("BODY_STANDALONE_VELOCITY_PUBLISHER_PRESENT")
        common=n.get_publishers_info_by_topic("/aima/mc/common/state");subs=n.get_subscriptions_info_by_topic("/aima/mc/body_pose")
        if len(common)!=1 or len(subs)!=1 or (common[0].node_name,common[0].node_namespace)!=(subs[0].node_name,subs[0].node_namespace) or subs[0].topic_type!="aimdk_msgs/msg/McBodyPose":raise RuntimeError("BODY_NATIVE_MC_SUBSCRIBER_UNVERIFIED")
        result["body_subscriber"]=dict(name=subs[0].node_name,namespace=subs[0].node_namespace,type=subs[0].topic_type,gid=list(subs[0].endpoint_gid))
        if mode!="restore":request_source(1001)
        request_source(2001)
        log=Path("/workspace/evidence/"+label+"_publisher.stdout").open("w")
        process=subprocess.Popen(["/workspace/evidence/native_body_pose_publisher",SOURCE,str(min(27.,deadline-time.monotonic()-1.1))],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,env={**os.environ,"LD_LIBRARY_PATH":"/workspace/raicom2026/mc/lib:"+os.environ.get("LD_LIBRARY_PATH","")})
        wait=min(deadline,time.monotonic()+2.)
        while time.monotonic()<wait:
            command(0.);n.pump()
            pubs=n.get_publishers_info_by_topic("/aima/mc/body_pose")
            if len(pubs)==1 and pubs[0].node_name=="robotsim_native_body_pose_publisher" and pubs[0].topic_type=="aimdk_msgs/msg/McBodyPose":break
            time.sleep(.02)
        else:raise RuntimeError("BODY_PUBLISHER_GRAPH_UNCONFIRMED")
        if mode!="restore":
            selected=False;lower_deadline=min(deadline-8.,time.monotonic()+8.)
            while time.monotonic()<lower_deadline:
                row=sample();z=row["pose"]["z"]
                if z>height+.02 or z<height-config["drop_m"]-.03:raise RuntimeError("BODY_HEIGHT_UNEXPECTED_DIRECTION_OR_EXCURSION")
                selected=selected or row["common_source"].get("name")==SOURCE
                reference=float(row["raw_common"]["servo_status"]["squat_offset"])
                if not math.isfinite(reference) or not -.5<=reference<=.005:raise RuntimeError("BODY_REFERENCE_INVALID")
                if reference<=initial_reference-config["drop_m"]+.01:
                    result["down_zero_trigger"]=dict(reference_offset_m=reference,actual_root_z_m=z,source_common_age_s=row["state_age_s"]["common"],basis="MC_REFERENCE_LEAD_PHYSICAL_ROOT_NOT_YET_SETTLED")
                    break
                command(-rate);time.sleep(.02)
            else:raise RuntimeError("BODY_DOWN_RESPONSE_NOT_OBSERVED")
            command(0.);result["source_selected_observed"]=selected
            result["low_stop"]=n.observe(5.,baseline=row["body_status"],require_stop=True)
            low=sample();result["low_height_m"]=low["pose"]["z"];result["low_squat_offset"]=low["raw_common"]["servo_status"]["squat_offset"]
            if not selected or not result["low_stop"].get("pass") or not config["drop_m"]-.01<=height-low["pose"]["z"]<=config["drop_m"]+.03:raise RuntimeError("BODY_LOWERED_STATE_NOT_CONFIRMED")
            if mode=="lower_hold":
                result["pass_result"]=True
                return result
        raise_deadline=min(deadline-4.,time.monotonic()+8.)
        while time.monotonic()<raise_deadline:
            row=sample();z=row["pose"]["z"]
            if z>height+.02:raise RuntimeError("BODY_RETURN_OVERSHOOT")
            if float(row["raw_common"]["servo_status"]["squat_offset"])>=initial_reference-.003:break
            command(rate);time.sleep(.02)
        else:raise RuntimeError("BODY_RETURN_RESPONSE_NOT_OBSERVED")
        command(0.);result["restored_stop"]=n.observe(5.,baseline=row["body_status"],require_stop=True)
        final=sample();result["final_height_m"]=final["pose"]["z"]
        result["pass_result"]=bool(result["restored_stop"].get("pass") and abs(final["pose"]["z"]-height)<=.01 and final["body_status"]==1)
        if not result["pass_result"]:raise RuntimeError("BODY_RESTORED_STOP_NOT_QUALIFIED")
    except Exception as exc:result["failure"]=str(exc)
    finally:
        if process is not None:
            try:
                command(0.);process.stdin.close()
                process.terminate();process.wait(timeout=3.)
            except Exception:
                process.kill();process.wait(timeout=2.)
            result["publisher_returncode"]=process.returncode
        if source_mutation_attempted:
            try:request_source(2002)
            except Exception as exc:result["source_disable_failure"]=str(exc);result["pass_result"]=False
        if mode=="lower_hold" and result.get("pass_result"):
            try:
                current=sample();result["released_source_low_hold"]=n.observe(5.,baseline=current["body_status"],require_stop=True)
                current=sample();result["final_height_m"]=current["pose"]["z"]
                if not result["released_source_low_hold"].get("pass") or abs(current["pose"]["z"]-result["low_height_m"])>.01:raise RuntimeError("LOWERED_BODY_DID_NOT_HOLD_AFTER_SOURCE_RELEASE")
            except Exception as exc:result["failure"]=str(exc);result["pass_result"]=False
        try:n.stop_feedback()
        except Exception as exc:result["observer_cleanup_failure"]=str(exc);result["pass_result"]=False
        result["elapsed_s"]=time.monotonic()-started
        write_durable_milestone("/workspace/evidence/"+label+"_trace.json",dict(trace=n.trace,events=n.events))
        write_durable_milestone("/workspace/evidence/"+label+"_result.json",result)
        n.destroy_node();rclpy.shutdown()
        print(json.dumps(result),flush=True)
    return result
