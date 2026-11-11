"""One-shot live native WSL retreat/turn review; no history playback."""
from pathlib import Path
import argparse,subprocess,sys,os,json,time,uuid
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
ROOT=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
CACHE=ROOT/"human-internal-contact-review-20261109T091125Z/review_binary_manifest.json"
WINPY=Path("/mnt/d/Dev/Anaconda/python.exe")
CMD=['bash', 'scripts/research/issue46/grasp/run_live_rgbd_grasp_review.sh', '--post-lift-retract-world-m', '-.24', '0', '0', '--reviewed-foundationwalk-profile', '--carry-timeout-s', '90', '--carry-sdk-minimum-turn', '--capture-mc-rl-debug', '--carry-grip-force-margin', '15.0', '--approved-pad-friction-reference', '--carry-transit-speed-policy-050', '--carry-away-1m', '--grasp-height-above-center-m', '0.10', '--allow-carried-bottle-tilt', '--right-claw-kp', '4', '--carry-straight-retreat-turn', '--carry-retention-only', '--approved-two-table-layout', '--carry-turn-brake-delay-s', '0.30', '--carry-retreat-brake-delay-s', '2.23', '--carry-compact-turn', '--carry-gait-prepare', '--carry-transit-command-speed', '0.12', '--carry-compact-angular-cap', '0.60', '--carry-release-source-on-stop', '--carry-compact-centre-feedback', '--carry-vendor-transition-table', '--carry-delivery-config', '/home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030/simulation/mujoco/scenes/raicom2026/config/retreat_turn_human_review.json', '--native-background-xvfb-tools', '/home/lzy18001500226/projects/RobotSim/reviews/issue-46/portable-xvfb-background-native-tools-20261108T230445Z/manifest.json']

def matching_review_model(manifest, episode):
    """Use the already verified camera-only cache for this exact live scene."""
    import hashlib
    frozen=json.loads(CACHE.read_text())
    if manifest["scene"]["sha256"]==frozen["source_scene"]["sha256"]:return CACHE
    for path in sorted(ROOT.glob("native-rgbd-host-cache-*/result.json"),reverse=True):
        cache=json.loads(path.read_text())
        if cache.get("source_sha256")!=manifest["scene"]["sha256"] or cache.get("engine")!=336:continue
        required={"body_pos","body_mass","jnt_range","geom_size","geom_pos","actuator_gainprm","actuator_forcerange","key_qpos"}
        if not required.issubset(cache.get("unchanged_physical_arrays",[])) or cache.get("mj_step_calls")!=0:raise RuntimeError("unverified review cache")
        binary=Path(cache["binary"])
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=cache["binary_sha256"]:raise RuntimeError("review cache binary changed")
        out=episode/"live_review_model_manifest.json"
        pg.write_new_text(out,json.dumps(dict(path=str(binary),sha256=cache["binary_sha256"],source_scene=manifest["scene"],provenance=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),scope="READ_ONLY_ACTUAL_NATIVE_STATE_CAMERA_ONLY_MODEL_CACHE"),indent=2)+"\n")
        return out
    raise RuntimeError("no exact-scene verified review model cache")



def copy_readonly_view_state(episode):
    """Separate Windows rendering IO from the native controller's state file."""
    if episode is None:return
    try:
        raw=(episode/'native_latest.json').read_bytes();json.loads(raw)
        temp=episode/'live_view_latest.tmp';temp.write_bytes(raw);os.replace(temp,episode/'live_view_latest.json')
    except (OSError,ValueError):return


def review_source_ready(episode):
    if episode is None:return False
    try:
        row=json.loads((episode/'live_view_latest.json').read_text())
        return isinstance(row,dict) and row.get('generation',0)>=1
    except (OSError,ValueError,TypeError):return False



def launch_first_person_observer(episode):
    """Start a disposable read-only RGB stream, never another physics owner."""
    invocation=json.loads((episode/"invocation.json").read_text())["argv"]
    flag="--task-region-host-cache"
    if flag not in invocation:raise RuntimeError("exact task camera cache absent")
    cache=Path(invocation[invocation.index(flag)+1]);record=json.loads(cache.read_text())
    destination=episode/"first_person_video";destination.mkdir()
    script=REPO/"scripts/research/issue46/perception/native_rgb_video_stream.py"
    command=[sys.executable,"-B",str(script),"--scene",record["source"],"--cache",str(cache),
             "--run",str(episode),"--output-dir",str(destination),"--port","8793"]
    with (destination/"observer.stdout").open("x") as log:
        observer=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,"MUJOCO_GL":"egl","MESA_D3D12_DEFAULT_ADAPTER_NAME":"NVIDIA",
                 "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
    (destination/"observer_process.json").write_text(json.dumps({
        "pid":observer.pid,"command":command,"role":"READ_ONLY_SAME_NATIVE_STATE_H264",
        "url":"http://localhost:8793/","physics_owner":False},indent=2))
    return observer

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--higher-carry-clearance",action="store_true",help="H300 source-bounded arm-only +60mm carry task pose");parser.add_argument("--multiccd",action="store_true",help="explicit numerical native337 manifold comparison, unchanged geometry/gains");parser.add_argument("--from-start",action="store_true",help="complete aligned START navigation before pickup");parser.add_argument("--bottle-height-m",type=float,choices=(.30,.22),default=.30);parser.add_argument("--check-only",action="store_true");parser.add_argument("--to-delivery-table",action="store_true");parser.add_argument("--delivery-mode",choices=("default","step"),default="step");parser.add_argument("--carry-angular-cap",type=float,choices=(.30,.60),default=.60,help="bounded command cap; source joint speed gates remain unchanged");parser.add_argument("--right-claw-kp",type=float,choices=(4.,6.),default=None,help="SIMULATION_ONLY grip gain; original 2Nm torque cap is unchanged");parser.add_argument("--no-viewer",action="store_true",help="autonomous real-physics run without the extra Windows read-only renderer");parser.add_argument("--task-grasp-region",action="store_true");parser.add_argument("--task-bottle-offset-xy-m",type=float,nargs=2,default=(0.,0.));args=parser.parse_args()
    if any(args.task_bottle_offset_xy_m) and not args.task_grasp_region:
        raise ValueError("task initialization offset needs RGBD task scope")
    if args.task_grasp_region:
        if not args.from_start or not args.to_delivery_table or args.bottle_height_m!=.22 or args.carry_angular_cap!=.60:
            raise ValueError("task region is the complete H220 START transfer")
        CMD.append("--task-grasp-region")
        if any(args.task_bottle_offset_xy_m):
            CMD.extend(["--task-bottle-offset-xy-m",*map(str,args.task_bottle_offset_xy_m)])
    host_threads={name:"1" for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}
    os.environ.update(host_threads)
    os.chdir(REPO)
    if args.right_claw_kp is None:args.right_claw_kp=6. if args.to_delivery_table else 4.
    if Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip()).resolve()!=REPO:raise RuntimeError("wrong worktree")
    if not args.no_viewer and (not CACHE.is_file() or not WINPY.is_file()):raise RuntimeError("review renderer dependency missing")
    installed=Path("/mnt/c/Users/HP/.codex/visualizations/2026/10/03/01a1014f-2b35-7fe2-b6f6-a934b993405b/mujoco336-review-runtime/packages/mujoco")
    if not args.no_viewer and not installed.is_dir():raise RuntimeError("isolated Windows renderer missing")
    profile=REPO/"simulation/mujoco/scenes/raicom2026/config"/(("reviewed_carry_step_forward.json" if args.delivery_mode=="step" else "reviewed_carry_table_delivery.json") if args.to_delivery_table else "retreat_turn_human_review.json")
    if args.higher_carry_clearance:
        if not args.to_delivery_table or args.bottle_height_m!=.30:raise ValueError("higher carry pose scope is H300 transfer")
        profile=profile.with_name("reviewed_carry_step_high_clearance.json")
    if args.bottle_height_m==.22:
        if not args.to_delivery_table or args.delivery_mode!="step":raise ValueError("H220 scope requires complete STEP-forward transfer")
        profile=profile.with_name("reviewed_carry_step_h220_forward_no_roll.json")
        CMD.extend(["--bottle-height-m","0.22"])
        CMD.remove("--carry-gait-prepare")
        CMD[CMD.index("--carry-transit-command-speed")+1]="0.20"
        CMD.extend(["--standing-grasp-posture",str(REPO/"simulation/mujoco/scenes/raicom2026/config/reviewed_h220_turn_forward_bend_pickup_functional.json"),"--restore-postured-carry"])
        CMD[CMD.index("--grasp-height-above-center-m")+1]="0.01"
    if args.from_start:
        if not args.to_delivery_table:raise ValueError("from START requires complete delivery mission")
        CMD.extend(["--execute-pickup-navigation","--dock-command-speed-cap","0.20","--dock-braking-delay-s","0.80","--dock-position-tolerance-m",("0.25" if args.bottle_height_m==.22 else "0.06"),"--refine-position-tolerance-m",("0.25" if args.bottle_height_m==.22 else "0.06")])
    if args.task_grasp_region:
        from scripts.research.issue46.native_runtime.native_task_grasp_region import task_route_profile
        task_profile=pg.create_unique_output_directory(ROOT,prefix="rgbd-task-short-turn-route").native_path
        pg.write_new_text(task_profile/"config.json",json.dumps(task_route_profile(json.loads(profile.read_text())),indent=2)+"\n")
        profile=task_profile/"config.json"
    CMD[CMD.index("--carry-delivery-config")+1]=str(profile)
    CMD[CMD.index("--right-claw-kp")+1]=str(args.right_claw_kp)
    if args.to_delivery_table:
        CMD[CMD.index("--carry-compact-angular-cap")+1]=str(args.carry_angular_cap)
        CMD[CMD.index("--post-lift-retract-world-m")+1]="-0.16"
    if args.multiccd:
        CMD.append("--native-multiccd")
        cycle=ROOT/"raicom-stl-xvfb-native-reset-stand-20261110T053739Z"
        if args.to_delivery_table and (cycle/"result.json").exists():CMD.extend(["--reuse-verified-claw-cycle",str(cycle)])
    if args.to_delivery_table and args.bottle_height_m==.30 and not args.multiccd:
        prior="raicom-stl-xvfb-native-reset-stand-20261109T150617Z" if args.right_claw_kp==4. else "raicom-stl-xvfb-native-reset-stand-20261109T201208Z"
        if args.from_start:prior="raicom-stl-xvfb-native-reset-stand-20261109T221003Z"
        CMD.extend(["--reuse-verified-grasp-plan",str(ROOT/prior)])
    if bool(json.loads(profile.read_text()).get("review_prefix_only"))==args.to_delivery_table:raise RuntimeError("review/transfer scope changed")
    if args.to_delivery_table:
        from scripts.research.issue46.native_runtime.native_base_micro_probe import build_carry_route_script
        route_check=build_carry_route_script(None,{},[],carry_pid=1,timeout_s=90.,transit_policy_050=True,away_1m=True,straight_retreat_turn=True,compact_turn=True,gait_prepare=("--carry-gait-prepare" in CMD),transit_command_speed=float(CMD[CMD.index("--carry-transit-command-speed")+1]),compact_centre_feedback=True,release_source_on_stop=True,compact_angular_cap=args.carry_angular_cap,delivery=json.loads(profile.read_text()),vendor_transition_table=True)
        compile(route_check,'prephysics_route_configuration','exec')
    if args.check_only:
        print(json.dumps(dict(status="READY",worktree=str(REPO),command=CMD,physics_started=False)));return
    status=subprocess.run(["docker","info","--format","{{.ServerVersion}}"],text=True,capture_output=True,timeout=8)
    if status.returncode:raise RuntimeError("Docker is not ready:"+status.stderr)
    if subprocess.check_output(["docker","ps","-q"],text=True).strip():raise RuntimeError("another container is running; did not launch")
    out=pg.create_unique_output_directory(ROOT,prefix="live-retreat-turn-human-review-command").native_path
    attempt=str(uuid.uuid4());old=set(ROOT.glob("raicom-stl-xvfb-native-reset-stand-*"));episode=None;native=None;viewer_started=False;closed=False;camera_started=False
    (out/"command.json").write_text(json.dumps(dict(attempt_id=attempt,route_configuration_validated_before_physics=args.to_delivery_table,host_numeric_thread_limits=host_threads,head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),command=CMD,scope="LIVE_PHYSICS_SECOND_TABLE_PLACEMENT" if args.to_delivery_table else "LIVE_PHYSICS_RETREAT_TURN_ONLY_NO_TRANSFER_QUALIFICATION"),indent=2))
    print("ATTEMPT",attempt,"LOG",out,flush=True)
    try:
        with (out/"runtime.stdout").open("x") as log:
            native=subprocess.Popen(CMD,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"DISPLAY":":0","MUJOCO_GL":"glfw"})
            deadline=time.monotonic()+(650. if args.bottle_height_m==.22 else 500.)
            while native.poll() is None:
                if time.monotonic()>deadline:raise RuntimeError("bounded full-episode runtime ended")
                owned=set(ROOT.glob("raicom-stl-xvfb-native-reset-stand-*"))-old
                if len(owned)>1:raise RuntimeError("ambiguous live episode")
                if owned:episode=next(iter(owned))
                copy_readonly_view_state(episode)
                if args.task_grasp_region and not args.no_viewer and episode and not camera_started and (episode/"invocation.json").exists() and review_source_ready(episode):
                    camera_started=True
                    try:
                        launch_first_person_observer(episode)
                        print("FIRST_PERSON_RGB_VIDEO http://localhost:8793/",flush=True)
                    except Exception as camera_error:
                        # Observer failure cannot command or abort the physical owner.
                        (episode/"first_person_observer_start_failure.json").write_text(json.dumps({"failure":str(camera_error),"physical_controller_unchanged":True}))
                        print("FIRST_PERSON_OBSERVER_FAILED",str(camera_error),flush=True)
                if not args.no_viewer and episode and not viewer_started and (episode/"live_view_scene.json").exists() and review_source_ready(episode):
                    manifest=json.loads((episode/"live_view_scene.json").read_text());review_cache=matching_review_model(manifest,episode)
                    def win(path):return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
                    viewer=REPO/"scripts/research/issue46/grasp/native_live_review_windows.py"
                    argv=[r"D:\Dev\Anaconda\python.exe","-I","-S",win(viewer),win(episode),win(review_cache),win(REPO)]
                    code="import subprocess,json;from pathlib import Path;root=Path("+repr(win(episode))+");log=(root/'live_viewer.stdout').open('x');p=subprocess.Popen("+repr(argv)+",stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);print(json.dumps(dict(viewer_pid=p.pid)))"
                    launched=subprocess.run([str(WINPY),"-c",code],text=True,capture_output=True,timeout=12)
                    if launched.returncode:raise RuntimeError("live viewer launch failed:"+launched.stderr)
                    (out/"viewer_process.json").write_text(launched.stdout);viewer_started=True;view_deadline=time.monotonic()+25.
                    print("LIVE_WINDOW_STARTING",episode,flush=True)
                if viewer_started and not (episode/"live_view_ready.json").exists() and time.monotonic()>view_deadline:raise RuntimeError("live viewer did not become ready")
                if episode and (episode/"live_view_closed.json").exists() and not closed:
                    (episode/"human_review_stop").write_text("viewer closed; existing controller must zero/stand/cleanup")
                    closed=True;print("REVIEW_CLOSED_REQUESTED_SAFE_STOP",flush=True)
                time.sleep(.05)
    finally:
        if native and native.poll() is None:
            native.terminate()
            try:native.wait(timeout=6)
            except subprocess.TimeoutExpired:native.kill();native.wait()
        owned=set(ROOT.glob("raicom-stl-xvfb-native-reset-stand-*"))-old
        for name in subprocess.check_output(["docker","ps","--format","{{.Names}}"],text=True).splitlines():
            obj=json.loads(subprocess.check_output(["docker","inspect",name],text=True))[0]
            mounts=[m for m in obj["Mounts"] if m.get("Destination")=="/workspace/evidence" and Path(m["Source"]) in owned]
            if mounts:
                dest=Path(mounts[0]["Source"])/"emergency_mc_var"
                subprocess.run(["docker","cp",name+":/agibot/data/var",str(dest)],capture_output=True,timeout=10)
                subprocess.run(["docker","rm","-f",name],check=True,capture_output=True,timeout=10)
        for d in owned:
            copy_readonly_view_state(d)
            if not (d/"live_review_finished.json").exists():(d/"live_review_finished.json").write_text(json.dumps(dict(physics_ended=True,review_only=not args.to_delivery_table)))
        (out/"execution.json").write_text(json.dumps(dict(attempt_id=attempt,returncode=None if native is None else native.returncode,episodes=[str(x) for x in owned],viewer_started=viewer_started,review_only=not args.to_delivery_table)))
    if episode and (episode/"result.json").exists() and (episode/"native_latest.json").exists():
        capture_script=REPO/"scripts/research/issue46/grasp/capture_native_run_end.py"
        try:
            captured=subprocess.run([sys.executable,str(capture_script),"--run",str(episode)],cwd=REPO,text=True,capture_output=True,timeout=30.,env={**os.environ,"MUJOCO_GL":"egl","MESA_D3D12_DEFAULT_ADAPTER_NAME":"NVIDIA"})
            pg.write_new_text(out/"end_capture.stdout",captured.stdout+captured.stderr)
            if captured.returncode==0:print("END_CAPTURE "+captured.stdout.strip().splitlines()[-1],flush=True)
            else:print("END_CAPTURE_FAILED "+captured.stderr[-600:],flush=True)
        except subprocess.TimeoutExpired:print("END_CAPTURE_TIMEOUT; physics result preserved",flush=True)
    if native is not None and native.returncode!=0:
        print("STARTUP_OR_RUNTIME_FAILED",native.returncode,flush=True)
        print((out/"runtime.stdout").read_text()[-6000:],flush=True)
    print("EVIDENCE",out,flush=True)
    if native is not None and native.returncode!=0:raise SystemExit(native.returncode)
    if episode and (episode/"result.json").exists():
        result=json.loads((episode/"result.json").read_text())
        failure=result.get("failure")
        if isinstance(failure,dict) and isinstance(failure.get("detail"),dict):
            failure=dict(gate=failure.get("gate"),cause=failure["detail"].get("failure",failure["detail"].get("error",failure["detail"].get("gate"))))
        keys=("HAL_PHYSICAL_LIFT","POST_LIFT_RETRACT","LEAVE_TABLE_AND_TURN","SECOND_TABLE_BASE_ARRIVAL_AND_STOP","FINAL_COMPOSITE_STOP","PHYSICAL_TABLE_PLACEMENT","TWO_TABLE_TRANSFER_EXECUTION")
        print(json.dumps(dict(gates={k:result.get("gates",{}).get(k,"NOT_RUN") for k in keys},failure=failure,cleanup=result.get("cleanup"),full_result=str(episode/"result.json")),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
