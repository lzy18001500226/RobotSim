"""One-shot live native WSL retreat/turn review; no history playback."""
from pathlib import Path
import argparse,subprocess,sys,os,json,time,uuid
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
ROOT=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
CACHE=ROOT/"human-internal-contact-review-20261109T091125Z/review_binary_manifest.json"
WINPY=Path("/mnt/d/Dev/Anaconda/python.exe")
CMD=['bash', 'scripts/research/issue46/grasp/run_live_rgbd_grasp_review.sh', '--post-lift-retract-world-m', '-.24', '0', '0', '--reviewed-foundationwalk-profile', '--carry-timeout-s', '90', '--carry-sdk-minimum-turn', '--capture-mc-rl-debug', '--carry-grip-force-margin', '15.0', '--approved-pad-friction-reference', '--carry-transit-speed-policy-050', '--carry-away-1m', '--grasp-height-above-center-m', '0.10', '--allow-carried-bottle-tilt', '--right-claw-kp', '4', '--carry-straight-retreat-turn', '--carry-retention-only', '--approved-two-table-layout', '--carry-turn-brake-delay-s', '0.30', '--carry-retreat-brake-delay-s', '2.23', '--carry-compact-turn', '--carry-gait-prepare', '--carry-transit-command-speed', '0.12', '--carry-compact-angular-cap', '0.60', '--carry-release-source-on-stop', '--carry-compact-centre-feedback', '--carry-vendor-transition-table', '--carry-delivery-config', '/home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030/simulation/mujoco/scenes/raicom2026/config/retreat_turn_human_review.json', '--native-background-xvfb-tools', '/home/lzy18001500226/projects/RobotSim/reviews/issue-46/portable-xvfb-background-native-tools-20261108T230445Z/manifest.json']
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check-only",action="store_true");parser.add_argument("--to-delivery-table",action="store_true");args=parser.parse_args()
    os.chdir(REPO)
    if Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip()).resolve()!=REPO:raise RuntimeError("wrong worktree")
    if not CACHE.is_file() or not WINPY.is_file():raise RuntimeError("review renderer dependency missing")
    installed=Path("/mnt/c/Users/HP/.codex/visualizations/2026/10/03/01a1014f-2b35-7fe2-b6f6-a934b993405b/mujoco336-review-runtime/packages/mujoco")
    if not installed.is_dir():raise RuntimeError("isolated Windows renderer missing")
    status=subprocess.run(["docker","info","--format","{{.ServerVersion}}"],text=True,capture_output=True,timeout=8)
    if status.returncode:raise RuntimeError("Docker is not ready:"+status.stderr)
    if subprocess.check_output(["docker","ps","-q"],text=True).strip():raise RuntimeError("another container is running; did not launch")
    profile=REPO/"simulation/mujoco/scenes/raicom2026/config"/("reviewed_carry_table_delivery.json" if args.to_delivery_table else "retreat_turn_human_review.json")
    CMD[CMD.index("--carry-delivery-config")+1]=str(profile)
    if args.to_delivery_table:
        CMD[CMD.index("--carry-compact-angular-cap")+1]="0.30"
        CMD.extend(["--reuse-verified-grasp-plan",str(ROOT/"raicom-stl-xvfb-native-reset-stand-20261109T150617Z")])
    if bool(json.loads(profile.read_text()).get("review_prefix_only"))==args.to_delivery_table:raise RuntimeError("review/transfer scope changed")
    if args.check_only:
        print(json.dumps(dict(status="READY",worktree=str(REPO),command=CMD,physics_started=False)));return
    out=pg.create_unique_output_directory(ROOT,prefix="live-retreat-turn-human-review-command").native_path
    attempt=str(uuid.uuid4());old=set(ROOT.glob("raicom-stl-xvfb-native-reset-stand-*"));episode=None;native=None;viewer_started=False;closed=False
    (out/"command.json").write_text(json.dumps(dict(attempt_id=attempt,head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),command=CMD,scope="LIVE_PHYSICS_SECOND_TABLE_PLACEMENT" if args.to_delivery_table else "LIVE_PHYSICS_RETREAT_TURN_ONLY_NO_TRANSFER_QUALIFICATION"),indent=2))
    print("ATTEMPT",attempt,"LOG",out,flush=True)
    try:
        with (out/"runtime.stdout").open("x") as log:
            native=subprocess.Popen(CMD,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"DISPLAY":":0","MUJOCO_GL":"glfw"})
            deadline=time.monotonic()+500.
            while native.poll() is None:
                if time.monotonic()>deadline:raise RuntimeError("bounded500s runtime ended")
                owned=set(ROOT.glob("raicom-stl-xvfb-native-reset-stand-*"))-old
                if len(owned)>1:raise RuntimeError("ambiguous live episode")
                if owned:episode=next(iter(owned))
                if episode and not viewer_started and (episode/"live_view_scene.json").exists() and (episode/"native_latest.json").exists():
                    manifest=json.loads((episode/"live_view_scene.json").read_text());cache=json.loads(CACHE.read_text())
                    if manifest["scene"]["sha256"]!=cache["source_scene"]["sha256"]:raise RuntimeError("live scene and review model differ")
                    def win(path):return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
                    viewer=REPO/"scripts/research/issue46/grasp/native_live_review_windows.py"
                    argv=[r"D:\Dev\Anaconda\python.exe","-I","-S",win(viewer),win(episode),win(CACHE),win(REPO)]
                    code="import subprocess,json;from pathlib import Path;root=Path("+repr(win(episode))+");log=(root/'live_viewer.stdout').open('x');p=subprocess.Popen("+repr(argv)+",stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);print(json.dumps(dict(viewer_pid=p.pid)))"
                    launched=subprocess.run([str(WINPY),"-c",code],text=True,capture_output=True,timeout=12)
                    if launched.returncode:raise RuntimeError("live viewer launch failed:"+launched.stderr)
                    (out/"viewer_process.json").write_text(launched.stdout);viewer_started=True;view_deadline=time.monotonic()+25.
                    print("LIVE_WINDOW_STARTING",episode,flush=True)
                if viewer_started and not (episode/"live_view_ready.json").exists() and time.monotonic()>view_deadline:raise RuntimeError("live viewer did not become ready")
                if episode and (episode/"live_view_closed.json").exists() and not closed:
                    (episode/"human_review_stop").write_text("viewer closed; existing controller must zero/stand/cleanup")
                    closed=True;print("REVIEW_CLOSED_REQUESTED_SAFE_STOP",flush=True)
                time.sleep(.25)
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
            if not (d/"live_review_finished.json").exists():(d/"live_review_finished.json").write_text(json.dumps(dict(physics_ended=True,review_only=not args.to_delivery_table)))
        (out/"execution.json").write_text(json.dumps(dict(attempt_id=attempt,returncode=None if native is None else native.returncode,episodes=[str(x) for x in owned],viewer_started=viewer_started,review_only=not args.to_delivery_table)))
    if native is not None and native.returncode!=0:
        print("STARTUP_OR_RUNTIME_FAILED",native.returncode,flush=True)
        print((out/"runtime.stdout").read_text()[-6000:],flush=True)
    print("EVIDENCE",out,flush=True)
    if native is not None and native.returncode!=0:raise SystemExit(native.returncode)
    if episode and (episode/"result.json").exists():
        result=json.loads((episode/"result.json").read_text());print(json.dumps(dict(review=result.get("gates",{}).get("LIVE_RETREAT_TURN_REVIEW"),failure=result.get("failure"),cleanup=result.get("cleanup")),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
