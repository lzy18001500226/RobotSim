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

def same_episode_registered_body_source(proof, native_row):
    return bool(proof.get('pass_result') is True and proof.get('mode')=='lower_hold'
        and proof.get('generation')==native_row['generation']
        and proof.get('native_process_pid')==native_row['process_pid']
        and any(x.get('action')==1001 and (x.get('raw') or {}).get('response',{}).get('header',{}).get('code')==0
                for x in proof.get('source_responses',[])))


def body_pause_zero_required(publisher_running, source_disabled, low_stop_pass):
    """A confirmed released source is read-only; never write its closed pipe."""
    if publisher_running:return True
    if source_disabled and low_stop_pass:return False
    raise RuntimeError("BODY_PUBLISHER_NOT_ALIVE")


def build_script(limits, robot_bodies, drop_m=.03, *, mode="roundtrip", label="native_body_height", reference_result=None, task_shortfall_tolerance_m=.01):
    if not .01<=task_shortfall_tolerance_m<=.015 or task_shortfall_tolerance_m>.01 and label not in ("body_crouch_pickup","body_crouch_delivery"):raise ValueError("bounded task lowering tolerance only")
    if drop_m not in (.03,.10,.13):raise ValueError("only measured30mm or bounded task100/130mm native-height probes are supported")
    if mode not in ("roundtrip","lower_hold","restore","observe") or label not in ("native_body_height","body_crouch_pickup","body_restore_carry","body_crouch_delivery","body_restore_final","body_restore_after_delivery","body_posture_hold"):raise ValueError("body task mode/label")
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
    code+="\n"+inspect.getsource(same_episode_registered_body_source)+"\n"+inspect.getsource(body_pause_zero_required)+"\n"+inspect.getsource(run_height_probe)
    code+='\nSOURCE="robotsim_body_height"\nMICRO.update('+repr(dict(body_height_probe=dict(drop_m=drop_m,max_runtime_s=60.,held_grip=(label in ("body_restore_carry","body_crouch_delivery") or mode=="observe" and reference_result is not None),height_command_rate=.03,mode=mode,label=label,reference_result=reference_result,task_shortfall_tolerance_m=task_shortfall_tolerance_m),robot_bodies=robot_bodies))+')\nrun_probe()\n'
    compile(code,"existing_observer_native_body_height","exec")
    return code

def run_height_probe(n, contract):
    def settle_native_body(n, baseline, deadline, result, label):
        """Wait for an unchanged complete stop contract; retain failed windows."""
        import json,time
        until=min(deadline,time.monotonic()+15.)
        while time.monotonic()<until:
            try:return n.observe(min(5.,until-time.monotonic()),baseline=baseline,require_stop=True)
            except RuntimeError as exc:
                prefix='COMPOSITE_STOP_NOT_QUALIFIED:'
                if not str(exc).startswith(prefix):raise
                measured=json.loads(str(exc)[len(prefix):])
                if not set(measured.get('violations',[])).issubset({'planar_speed_exceeded','joint_target_not_settled','joint_position_not_settled'}):raise
                result.setdefault('settling_windows',[]).append(dict(label=label,measurement=measured))
                if measured.get('metrics',{}).get('safety_faults'):raise
        raise RuntimeError('BODY_STABLE_WINDOW_TIMEOUT:'+label)
    import os,subprocess,time,json,math
    from pathlib import Path
    from aimdk_msgs.srv import SetMcInputSource
    from rosidl_runtime_py.convert import message_to_ordereddict as raw
    import rclpy
    result=dict(classification="SIMULATION_ONLY_INSTALLED_NATIVE_BODY_POSE_COMPATIBILITY",
                requested_drop_m=MICRO["body_height_probe"]["drop_m"],velocity_publishers_created=sum(p.topic_name in ("/cmd_vel","/aima/mc/locomotion/velocity") for p in n.publishers),node_publisher_topics=[p.topic_name for p in n.publishers],motion_action_requests_sent=0,
                mc_receipt_ack="UNAVAILABLE",pass_result=False,commands=[],source_request_attempts=[],source_responses=[])
    process=None;source_mutation_attempted=False;started=time.monotonic();deadline=started+float(MICRO["body_height_probe"]["max_runtime_s"])
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
        if action==2002:result["source_disable_confirmed"]=True
    def command(value):
        if value not in (-rate,0.,rate):raise RuntimeError("BODY_RATE_BOUND")
        if process is None or process.poll() is not None:raise RuntimeError("BODY_PUBLISHER_NOT_ALIVE")
        process.stdin.write(str(value)+"\n");process.stdin.flush()
        result["commands"].append(dict(t=time.monotonic(),stand_height=value,evidence="PIPE_WRITE_NOT_MC_RECEIPT"))
    def fresh_mode_sample():
        row=n.sample()
        if any(row.get(k) is None for k in ('get_mode_desc','get_mode_header_code','get_mode_status_value')):
            # Clock-epoch cache invalidation is missing evidence, not a
            # confirmed non-STAND result. The caller must zero and reacquire.
            raise RuntimeError('CRITICAL_ROS_TELEMETRY_STALE:get_mode')
        return row
    def sample():
        if time.monotonic()>=deadline:raise RuntimeError("BODY_HEIGHT_WATCHDOG")
        n.pump()
        try:row=fresh_mode_sample()
        except RuntimeError as exc:
            transient=str(exc)=="CLOCK_GENERATION_PENDING" or str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:")
            if not transient or process is None or result.get('clock_pauses'):raise
            need_zero=body_pause_zero_required(process.poll() is None,result.get("source_disable_confirmed",False),result.get("low_stop",{}).get("pass",False))
            if need_zero:command(0.)
            else:result["post_release_readonly_reacquire"]=True
            pause_started=time.monotonic();identity=(n.latest_row['generation'],n.latest_row['process_pid'])
            fresh_until=min(deadline,pause_started+1.)
            while time.monotonic()<fresh_until:
                n.pump()
                try:row=fresh_mode_sample();break
                except RuntimeError as pending:
                    if str(pending)!="CLOCK_GENERATION_PENDING" and not str(pending).startswith("CRITICAL_ROS_TELEMETRY_STALE:"):raise
                time.sleep(.005)
            else:raise RuntimeError('BODY_NEW_CLOCK_GENERATION_NOT_FRESH')
            if identity!=(n.latest_row['generation'],n.latest_row['process_pid']):raise RuntimeError('BODY_NATIVE_PROCESS_CHANGED_DURING_CLOCK_PAUSE')
            stopped=settle_native_body(n,None,deadline,result,'clock_pause_stop')
            if not stopped.get('pass'):raise RuntimeError('BODY_CLOCK_PAUSE_NOT_STOPPED')
            result['clock_pauses']=[dict(start_monotonic_s=pause_started,end_monotonic_s=time.monotonic(),cause=str(exc),zero_before_reacquire=True,stop=stopped)]
            row=fresh_mode_sample()
        if row["get_mode_desc"]!="STAND_DEFAULT" or row["common_mode_desc"]!="STAND_DEFAULT" or row["get_mode_status_value"]!=100 or row["common_mode_status_value"]!=100 or row["get_mode_header_code"]!=0:
            result['rejected_body_mode']={k:row.get(k) for k in ('sample_monotonic_s','sim_time_s','get_mode_desc','common_mode_desc','get_mode_status_value','common_mode_status_value','get_mode_header_code','fsm_state','body_status','source_clock_generation','state_age_s')}
            raise RuntimeError("BODY_HEIGHT_MODE_NOT_STAND:"+json.dumps(result['rejected_body_mode']))
        if row["fsm_state"] not in contract.MC_FSM_ACTIVE_STATES or row["body_status"] not in (1,4,5):raise RuntimeError("BODY_HEIGHT_POSTURE_OR_PROTECTION")
        feet={}
        for side in ("left","right"):
            feet[side]=sum(max(0.,c["local_force_torque"][0]) for c in n.latest_row["contacts"] if c.get("efc_address",-1)>=0 and side+"_ankle_roll_link" in (c["body1"],c["body2"]) and "world" in (c["body1"],c["body2"]))
        if min(feet.values())<5.:raise RuntimeError("BODY_HEIGHT_BILATERAL_SUPPORT_LOST")
        if not n.trace or row["sample_monotonic_s"]>n.trace[-1]["sample_monotonic_s"]:n.trace.append(row)
        return row
    def sample_after_qualified_stop(key):
        try:return sample()
        except RuntimeError as exc:
            if str(exc)!='CLOCK_GENERATION_PENDING' or result.get(key,{}).get('pass') is not True:raise
            identity=(n.latest_row['generation'],n.latest_row['process_pid'])
            if process is None:
                if key!="initial_stop" or result.get("commands"):raise RuntimeError("BODY_READONLY_RECHECK_NOT_INITIAL_ZERO_PHASE")
                result["initial_readonly_clock_recheck"]=True
            elif body_pause_zero_required(process.poll() is None,result.get("source_disable_confirmed",False),result.get("low_stop",{}).get("pass",False)):command(0.)
            fresh_stop=settle_native_body(n,None,deadline,result,key+'_new_epoch')
            if not fresh_stop.get('pass'):raise RuntimeError('BODY_STOP_EPOCH_NOT_REQUALIFIED')
            row=sample()
            if identity!=(n.latest_row['generation'],n.latest_row['process_pid']):raise RuntimeError('BODY_NATIVE_IDENTITY_CHANGED_AT_COMPLETED_STOP')
            result.setdefault('stationary_clock_rechecks',[]).append(dict(cause=str(exc),motion_commands_issued=False,zero_before_recheck=True,stop=fresh_stop))
            result[key]=fresh_stop
            return row

    if config.get('held_grip'):
        ready=json.loads(Path('/workspace/evidence/carry_ready.json').read_text())
        if not ready.get('physical_lift_pass') or not ready.get('post_lift_retract_pass'):raise RuntimeError('BODY_RESTORE_REQUIRES_MEASURED_GRIP')
        MICRO['allowed_contact_pairs']=[['test_cylinder',p] for p in ('R_hand_narrow3_Link','R_hand_wide3_Link')]
        original_sample=n.sample
        def held_sample(*args,**kwargs):
            value=original_sample(*args,**kwargs);health=json.loads(Path('/workspace/evidence/carry_heartbeat.json').read_text())
            if health.get('healthy') is not True or health.get('pid')!=ready['pid'] or health.get('generation')!=n.latest_row['generation'] or not 0.<=(time.monotonic_ns()-health['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('BODY_HELD_OWNER_NOT_FRESH')
            return value
        n.sample=held_sample
    try:
        if result["velocity_publishers_created"]!=0 or any(t!="/parameter_events" for t in result["node_publisher_topics"]):raise RuntimeError("BODY_OBSERVER_MUST_CREATE_ZERO_ACTUATION_PUBLISHERS")
        n.start_feedback();until=min(deadline,time.monotonic()+5.);first=None
        while time.monotonic()<until:
            n.pump()
            if all(k in n.messages for k in ("common","imu","leg_state","leg_command")) and "get_mode" in n.readbacks:
                try:first=sample();break
                except RuntimeError as exc:
                    if str(exc) not in ("CLOCK_GENERATION_PENDING","NATIVE_FEEDBACK_STALE_OR_INVALID") and not str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:"):raise
                    result.setdefault('initial_readiness_waits',[]).append(dict(cause=str(exc),t=time.monotonic(),motion_publishers_started=False))
            time.sleep(.01)
        if first is None:raise RuntimeError("BODY_INITIAL_FRESH_STATE_UNCONFIRMED")
        # This fresh observer has no height publisher yet. Reuse the existing
        # zero-only telemetry acquisition protocol, not an active-motion retry.
        if process is not None:raise RuntimeError("BODY_READONLY_PHASE_HAS_HEIGHT_PUBLISHER")
        MICRO["body_readonly_zero_phase"]=True
        try:result["initial_stop"]=settle_native_body(n,first["body_status"],deadline,result,"initial_stop")
        finally:MICRO.pop("body_readonly_zero_phase",None)
        if not result["initial_stop"].get("pass"):raise RuntimeError("BODY_INITIAL_COMPOSITE_STOP_FAILED")
        initial=sample_after_qualified_stop("initial_stop");height=initial["pose"]["z"];result["initial_height_m"]=height
        if mode=="observe":
            if config.get('reference_result'):
                proof=json.loads(Path(config['reference_result']).read_text())
                original_height=float(proof['initial_height_m']);height_error=initial['pose']['z']-original_height
                baseline=json.loads(Path('/workspace/evidence/standing_waist_initial_hold.json').read_text())['request']['target_rad']
                waist_error={name:n.latest_row['joints'][name]['q']-baseline[name] for name in baseline}
                waist_velocity=max(abs(n.latest_row['joints'][name]['qvel']) for name in baseline)
                result.update(qualification_scope='FINAL_HELD_NATIVE_STANDING_HEIGHT_AND_WAIST',original_height_m=original_height,actual_height_error_m=height_error,waist_return_error_rad=waist_error,max_waist_velocity_rad_s=waist_velocity)
                if proof.get('generation')!=initial['native_generation'] or abs(height_error)>.01 or max(abs(v) for v in waist_error.values())>.10 or waist_velocity>.05:raise RuntimeError('FINAL_HELD_STANDING_STATE_NOT_RESTORED')
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
        if label=="body_crouch_delivery":
            proof=json.loads(Path('/workspace/evidence/body_crouch_pickup_result.json').read_text())
            if not same_episode_registered_body_source(proof,n.latest_row):raise RuntimeError('DELIVERY_BODY_SOURCE_REGISTRATION_PROOF_INVALID')
            result['source_registration']='REUSE_SAME_MC_EPISODE_REGISTERED_DISABLED_SOURCE'
        elif mode!="restore":request_source(1001)
        request_source(2001)
        log=Path("/workspace/evidence/"+label+"_publisher.stdout").open("w")
        process=subprocess.Popen(["/workspace/evidence/native_body_pose_publisher",SOURCE,str(min(27.,deadline-time.monotonic()-1.1))],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,env={**os.environ,"LD_LIBRARY_PATH":"/workspace/raicom2026/mc/lib:"+os.environ.get("LD_LIBRARY_PATH","")})
        wait=min(deadline,time.monotonic()+5.);last_graph=None
        while time.monotonic()<wait:
            command(0.);n.pump()
            pubs=n.get_publishers_info_by_topic("/aima/mc/body_pose")
            graph=[dict(name=p.node_name,namespace=p.node_namespace,type=p.topic_type) for p in pubs]
            if graph!=last_graph:
                result.setdefault('publisher_graph_observations',[]).append(dict(t=time.monotonic(),publishers=graph))
                last_graph=graph
            if len(pubs)==1 and pubs[0].node_name=="robotsim_native_body_pose_publisher" and pubs[0].topic_type=="aimdk_msgs/msg/McBodyPose":break
            time.sleep(.02)
        else:raise RuntimeError("BODY_PUBLISHER_GRAPH_UNCONFIRMED")
        if mode!="restore":
            selected=False;lower_deadline=min(deadline-8.,time.monotonic()+8.)
            while time.monotonic()<lower_deadline+sum(p["end_monotonic_s"]-p["start_monotonic_s"] for p in result.get("clock_pauses",[])):
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
            result["low_stop"]=settle_native_body(n,row["body_status"],deadline,result,"low_stop")
            low=sample_after_qualified_stop("low_stop");result["low_height_m"]=low["pose"]["z"];result["low_squat_offset"]=low["raw_common"]["servo_status"]["squat_offset"]
            result["lowering_target_acceptance"]=dict(actual_drop_m=height-low["pose"]["z"],legacy_shortfall010_pass=height-low["pose"]["z"]>=config["drop_m"]-.01,task_shortfall_tolerance_m=config.get("task_shortfall_tolerance_m",.01))
            if not selected or not result["low_stop"].get("pass") or not config["drop_m"]-config.get("task_shortfall_tolerance_m",.01)<=height-low["pose"]["z"]<=config["drop_m"]+.03:raise RuntimeError("BODY_LOWERED_STATE_NOT_CONFIRMED")
            if mode=="lower_hold":
                result["pass_result"]=True
                return result
        raise_deadline=min(deadline-4.,time.monotonic()+8.)
        while time.monotonic()<raise_deadline+sum(p["end_monotonic_s"]-p["start_monotonic_s"] for p in result.get("clock_pauses",[])):
            row=sample();z=row["pose"]["z"]
            if z>height+.02:raise RuntimeError("BODY_RETURN_OVERSHOOT")
            if float(row["raw_common"]["servo_status"]["squat_offset"])>=initial_reference-.003:break
            command(rate);time.sleep(.02)
        else:raise RuntimeError("BODY_RETURN_RESPONSE_NOT_OBSERVED")
        command(0.);result["restored_stop"]=settle_native_body(n,row["body_status"],deadline,result,"restored_stop")
        final=sample_after_qualified_stop("restored_stop");result["final_height_m"]=final["pose"]["z"]
        result['original_height_match']=abs(final['pose']['z']-height)<=.01
        result['native_height_reference_restored']=abs(float(final['raw_common']['servo_status']['squat_offset'])-initial_reference)<=.003
        if label=='body_restore_carry':
            result['qualification_scope']='INTERMEDIATE_NATIVE_HEIGHT_REFERENCE_WAIST_HANDOFF_STILL_REQUIRED'
            result['pass_result']=bool(result['restored_stop'].get('pass') and result['native_height_reference_restored'] and final['body_status']==1)
        else:result['pass_result']=bool(result['restored_stop'].get('pass') and result['original_height_match'] and final['body_status']==1)
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
                current=sample();result["released_source_low_hold"]=settle_native_body(n,current["body_status"],deadline,result,"released_source_low_hold")
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
