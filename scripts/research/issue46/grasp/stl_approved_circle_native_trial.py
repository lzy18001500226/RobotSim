"""Bounded adapter to the existing native standing/docking/pregrasp runner.

Only the static replan subprocess is replaced by approved-circle IK. Native
startup, ownership, contact-model overlay, stopping and teardown are unchanged.
This executes pregrasp/closure first; no lift is attempted after a failed gate.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
import math

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as native
from scripts.agent import path_governance as pg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--circle-review",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--fixed-reviewed-checkpoint",type=Path,help="initialize native episode with reviewed ungrasped keyframe; no walking claimed")
    p.add_argument("--checkpoint-base-offset-xy-m",type=float,nargs=2,default=(0.,0.),help="screened <=20mm per-axis episode-boundary stance diagnostic only")
    p.add_argument("--review-captured-run",type=Path)
    p.add_argument("--review-dry-run",action="store_true")
    p.add_argument("--review-full-flow",action="store_true",help="play captured post-Reset physics from START, stopping at the qualified controlled terminal frame")
    p.add_argument("--execute-simulation",action="store_true")
    p.add_argument("--startup-opening-diagnostic",action="store_true")
    p.add_argument("--both-claws-zero-internal-friction",action="store_true")
    p.add_argument("--right-claw-zero-internal-friction",action="store_true")
    p.add_argument("--right-claw-damping",type=float)
    p.add_argument("--noslip-solver-diagnostic",action="store_true",help="separate SIMULATION_ONLY two-iteration numerical postsolver; no contact-material/geometry changes")
    p.add_argument("--elliptic-grip-solver-diagnostic",action="store_true",help="Menagerie-style numerical elliptic cone/impratio=10, unchanged actual friction/condim/geometry")
    p.add_argument("--physical-lift-test",action="store_true")
    p.add_argument("--source-pad-coacd-comparison",action="store_true",help="reuse the existing hash-locked source-STL proxy pieces for only the two right distal pads; separate SIMULATION_ONLY physics")
    p.add_argument("--open-hand-transit-park",action="store_true")
    p.add_argument("--initial-hands-fully-open",action="store_true")
    p.add_argument("--pause-for-human-review",action="store_true")
    p.add_argument("--native-physics-audit",action="store_true",help="read-only actual-engine pose/contact/force observer; ABI must match installed library")
    p.add_argument("--target-source",choices=("truth-diagnostic","historical-rgbd-replay","fresh-rgbd"),default="truth-diagnostic",help="explicit target provenance; fresh-rgbd uses the calibrated virtual head sensor")
    p.add_argument("--rgbd-render-adapter",help="explicit WSLg/Mesa sensor-process GPU selection, e.g. NVIDIA; actual renderer identity must agree")
    p.add_argument("--grasp-height-above-center-m",type=float,default=.075,help="grasp section within the canonical 300mm bottle body")
    p.add_argument("--grasp-axis-tilt-rad",type=float,default=0.)
    p.add_argument("--inverted-bottle-axis",action="store_true")
    p.add_argument("--grasp-axis-calibration",type=Path)
    p.add_argument("--use-measured-load-frame",action="store_true")
    p.add_argument("--coordinated-waist-pitch-offset-rad",type=float,default=0.)
    p.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument("--dock-world-xy",type=float,nargs=2,default=(.85,-1.44))
    p.add_argument("--dock-position-tolerance-m",type=float,default=.10)
    p.add_argument("--dock-braking-delay-s",type=float)
    p.add_argument("--arm-plan-duration-s",type=float,default=12.0)
    p.add_argument("--arm-plan-hold-s",type=float,default=5.0)
    args=p.parse_args()
    if args.review_captured_run:
        from scripts.research.issue46.native_runtime.native_grasp_audit import review_native_capture
        review_native_capture(args.review_captured_run,dry_run=args.review_dry_run,full_flow=args.review_full_flow)
        return
    if not args.execute_simulation:raise ValueError("explicit simulation execution flag required")
    if any(not math.isfinite(v) or abs(v)>.02000001 for v in args.checkpoint_base_offset_xy_m) or any(args.checkpoint_base_offset_xy_m) and not args.fixed_reviewed_checkpoint:raise ValueError('stance offset requires a fixed reviewed episode and <=20mm bounds')
    if args.use_measured_load_frame and (args.grasp_axis_calibration is None or not args.elliptic_grip_solver_diagnostic):raise ValueError('measured load frame requires its evidence and matched numerical profile')
    baseline=json.loads((args.baseline/"result.json").read_text())
    if not args.native_physics_audit:
        raise ValueError("approved-circle execution requires --native-physics-audit")
    original=native.run
    if original(["docker","ps","-q"],timeout_s=5).stdout.strip():raise RuntimeError("conflicting container")
    tracking=None
    tracking_output=None
    perception_process=None
    perception_output=None
    host_model_cache=None
    def run(command,**kwargs):
        nonlocal tracking,tracking_output,perception_process,perception_output
        # The native parent now builds, launches and checks its required observer.
        # Keep one owner: a wrapper must not inject a second LD_PRELOAD launch.
        if any(str(v).endswith("/runtime_actual_grasp_replan.py") for v in command):
            state=command[command.index("--dock-state")+1]
            # Preserve ROS capture; use one fresh native snapshot for all FK inputs.
            from scripts.research.issue46.native_runtime import native_grasp_audit
            state=Path(state)
            physics=state.parent/"native_physics.jsonl"
            native_grasp_audit.require_fresh_sample(physics)
            with physics.open('rb') as stream:
                stream.seek(0,2);stream.seek(max(0,stream.tell()-1048576));raw=stream.read()
            lines=raw.splitlines()
            if not raw.endswith(bytes([10])):lines=lines[:-1]
            for line in reversed(lines):
                try:row=json.loads(line);break
                except (ValueError,UnicodeDecodeError):continue
            synchronized=native_grasp_audit.planning_state_from_native(row,json.loads(state.read_text()))
            if args.target_source=='fresh-rgbd':
                from scripts.research.issue46.perception.native_rgbd_grasp import qualify_estimate
                perception_output=state.parent/'fresh_rgbd'
                scene=json.loads(args.candidate.read_text())['scene']['path']
                sensor_command=['/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python',str(REPO/'scripts/research/issue46/perception/native_rgbd_grasp.py'),'--stream-scene',scene,'--native-latest',str(state.parent/'native_latest.json'),'--output-dir',str(perception_output)]
                if host_model_cache:sensor_command.extend(['--host-model-cache',str(host_model_cache)])
                pg.write_new_text(state.parent/'fresh_rgbd_command.json',json.dumps(sensor_command)+'\n')
                import os
                environment=dict(os.environ);environment['MUJOCO_GL']='egl'
                if args.rgbd_render_adapter:environment['MESA_D3D12_DEFAULT_ADAPTER_NAME']=args.rgbd_render_adapter
                with (state.parent/'fresh_rgbd.stdout').open('x') as log:perception_process=subprocess.Popen(sensor_command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,env=environment)
                deadline=time.monotonic()+15.
                while time.monotonic()<deadline:
                    native_grasp_audit.require_fresh_sample(state.parent/'native_latest.json')
                    if perception_process.poll() is not None:raise RuntimeError('RGBD_SENSOR_STARTUP_FAILED')
                    latest=perception_output/'latest.json'
                    if latest.exists():
                        observation=json.loads(latest.read_text())
                        if observation.get('pass') is not True:raise RuntimeError('RGBD_SENSOR_NOT_QUALIFIED:'+str(observation))
                        pose=qualify_estimate(observation,generation=row['generation']);break
                    time.sleep(.02)
                else:raise RuntimeError('RGBD_SENSOR_READINESS_TIMEOUT')
                synchronized['object_pose_world']={'xyz_m':pose['position_m'],'quaternion_wxyz':pose['quaternion_wxyz']}
                synchronized['perception']={'latest_path':str(latest),'source':'FRESH_SIMULATED_CALIBRATED_HEAD_RGBD','truth_not_control_target':True}
            state=state.parent/"native_synchronized_dock_state.json"
            pg.write_new_text(state,json.dumps(synchronized,indent=2)+"\n")
            state=str(state)
            # Navigation already applied the requested docking offset. The new
            # state is MEASURED; adding that offset again invents a root pose.
            command=["/home/lzy18001500226/.local/bin/uv","run","--offline","--with","mujoco==3.3.6","--with","numpy==2.2.6","--with","scipy==1.15.3","python3",str(Path(__file__).with_name("stl_approved_circle_ik.py")),"--circle-review",str(args.circle_review),"--candidate",str(args.candidate),"--standing-state",state,"--grasp-height-above-center-m","0.075","--runtime-handoff"]
            command.extend(["--target-source",args.target_source,"--free-bottle-axis-roll"])
            if host_model_cache:command.extend(['--host-model-cache',str(host_model_cache)])
            if args.physical_lift_test:command.append('--physical-lift-test')
            command[command.index("--grasp-height-above-center-m")+1]=str(args.grasp_height_above_center_m)
            command.extend(['--grasp-axis-tilt-rad',str(args.grasp_axis_tilt_rad)])
            if args.inverted_bottle_axis:command.append('--inverted-bottle-axis')
            if args.grasp_axis_calibration:command.extend(['--grasp-axis-calibration',str(args.grasp_axis_calibration)])
            if args.use_measured_load_frame:command.append('--use-measured-load-frame')
            if args.coordinated_waist_pitch_offset_rad:
                command.extend(["--waist-pitch-offset-rad",str(args.coordinated_waist_pitch_offset_rad)])
            kwargs["timeout_s"]=45.
        result=original(command,**kwargs)
        if any(str(v).endswith('/stl_approved_circle_ik.py') for v in command) and '--runtime-handoff' in command and result.returncode==0 and json.loads(result.stdout.strip().splitlines()[-1]).get('gate')=='PASS':
            reference=Path(json.loads(result.stdout.strip().splitlines()[-1])['output_wsl'])/'result.json'
            evidence=Path(command[command.index('--standing-state')+1]).parent
            tracking_output=evidence/'native_tracking_target.json'
            track_command=['/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python',str(Path(__file__).with_name('stl_approved_circle_ik.py')),'--track-native',str(reference),str(evidence/'native_latest.json'),str(tracking_output)]
            pg.write_new_text(evidence/'native_tracking_command.json',json.dumps(track_command)+'\n')
            log=(evidence/'native_tracking.stdout').open('x')
            tracking=subprocess.Popen(track_command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO);log.close()
            readiness_started=time.monotonic();deadline=readiness_started+25.;ready=False;initial_position=None
            while time.monotonic()<deadline and tracking.poll() is None:
                native_grasp_audit.require_fresh_sample(evidence/'native_latest.json')
                current=json.loads((evidence/'native_latest.json').read_text());body=current['bodies']['pelvis'];xyz=body['position_m'];R=body['rotation']
                initial_position=initial_position or xyz[:2]
                if not .5<=xyz[2]<=.82 or max(abs(math.asin(max(-1.,min(1.,-R[6])))),abs(math.atan2(R[7],R[8])))>math.radians(15) or math.dist(xyz[:2],initial_position)>.05:raise RuntimeError('UNSAFE_POSTURE_DURING_TRACKER_INITIALIZATION')
                if tracking_output.exists():
                    target=json.loads(tracking_output.read_text())
                    if not target.get('pass'):raise RuntimeError('native circle tracker initialization failed:'+str(target))
                    age=(time.monotonic_ns()-target['source_host_ns'])*1e-9
                    if 0.<=age<=.20:ready=True;break
                time.sleep(.02)
            pg.write_new_text(evidence/'native_tracking_readiness.json',json.dumps({'ready':ready,'elapsed_s':time.monotonic()-readiness_started,'loading_budget_s':25.,'active_source_max_age_s':.20,'child_returncode':tracking.poll()},indent=2)+'\n')
            if not ready:raise RuntimeError('native circle tracker did not qualify within bounded model-loading readiness')
        return result
    native.run=run
    table=Path(baseline["table_overlay"]["path"])
    claw=Path(baseline["claw_control_overlay"]["path"])
    if args.source_pad_coacd_comparison:
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import reuse_source_pad_proxy
        claw=reuse_source_pad_proxy(claw)
    if args.right_claw_zero_internal_friction and args.both_claws_zero_internal_friction:raise ValueError("select only one friction scope")
    if args.right_claw_zero_internal_friction or args.both_claws_zero_internal_friction:
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import build
        claw=build(claw,both_hands=args.both_claws_zero_internal_friction,right_damping=args.right_claw_damping,noslip_iterations=2 if args.noslip_solver_diagnostic else 0,elliptic_grip_solver=args.elliptic_grip_solver_diagnostic)
    elif args.right_claw_damping is not None or args.noslip_solver_diagnostic or args.elliptic_grip_solver_diagnostic:raise ValueError('parameter/solver diagnostics require an explicit existing zero-internal-friction profile')
    if args.fixed_reviewed_checkpoint:
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import add_reviewed_checkpoint
        claw=add_reviewed_checkpoint(claw,args.fixed_reviewed_checkpoint,base_offset_xy=args.checkpoint_base_offset_xy_m)
    if args.source_pad_coacd_comparison:
        # The static collider must match the actual selected runtime variant;
        # retain the original candidate and source scene as immutable inputs.
        original_candidate=args.candidate;candidate=json.loads(original_candidate.read_text())
        candidate['scene']=dict(json.loads(claw.read_text())['generated']['host_scene'])
        candidate['source_pad_collision_comparison']={'source_candidate':str(original_candidate),'source_candidate_sha256':native.sha256(original_candidate),'overlay':str(claw),'overlay_sha256':native.sha256(claw),'classification':'SIMULATION_ONLY_SOURCE_DERIVED_COLLISION_COMPARISON'}
        candidate_dir=pg.create_unique_output_directory(Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46'),prefix='source-pad-coacd-ik-input').native_path
        pg.write_new_text(candidate_dir/'candidate.json',json.dumps(candidate,indent=2)+'\n');pg.write_artifact_manifest(candidate_dir);args.candidate=candidate_dir/'candidate.json'
    if args.target_source=='fresh-rgbd':
        from scripts.research.issue46.perception.native_rgbd_grasp import prepare_host_cache
        host_model_cache=prepare_host_cache(Path(json.loads(args.candidate.read_text())['scene']['path']))
    helper=REPO/"scripts/research/issue46/native_runtime/native_mujoco_reset.py"
    display=__import__("os").environ.get("DISPLAY")
    if not display:raise RuntimeError("DISPLAY is unset; run this command from the WSLg terminal")
    dock_xy=args.dock_world_xy or (0.74+args.base_offset_xyz_m[0],-1.44+args.base_offset_xyz_m[1])
    argv=["native", "--table-overlay-result",str(table),"--table-overlay-result-sha256",native.sha256(table),"--claw-control-overlay-result",str(claw),"--claw-control-overlay-result-sha256",native.sha256(claw),"--reset-helper",str(helper),"--reset-helper-sha256",native.sha256(helper),"--interactive-x11-display",display,"--hal-arm-candidate",str(args.candidate),"--hal-arm-candidate-sha256",native.sha256(args.candidate),"--disable-mc-arm-publisher","--disable-mc-hand-publisher","--hal-arm-plan-mode","pregrasp-hold","--hal-right-hand-command","0.0","--hal-hand-close-hold-s","2","--dock-world-xy",str(dock_xy[0]),str(dock_xy[1]),"--dock-yaw-rad","-0.07591850724454846","--dock-position-only","--use-official-cmd-vel-bridge","--dock-timeout-s","30"]
    if args.fixed_reviewed_checkpoint:argv.append("--fixed-reviewed-checkpoint")
    if args.physical_lift_test:
        argv.append('--physical-lift-test')
        argv[argv.index('--hal-hand-close-hold-s')+1]='5'
    argv.extend(["--dock-position-tolerance-m",str(args.dock_position_tolerance_m)])
    if args.dock_braking_delay_s is not None:
        if not .0<args.dock_braking_delay_s<=1. or args.target_source not in ('truth-diagnostic','fresh-rgbd'):raise ValueError('measured braking profile requires bounded delay and native feedback')
        argv.extend(['--dock-braking-delay-s',str(args.dock_braking_delay_s)])
    argv.extend(["--arm-plan-duration-s",str(args.arm_plan_duration_s),"--arm-plan-hold-s",str(args.arm_plan_hold_s)])
    if args.target_source in ("truth-diagnostic","fresh-rgbd"):argv.extend(["--native-dock-truth-feedback","--native-circle-tracking"])
    if args.startup_opening_diagnostic:argv.append("--startup-opening-diagnostic")
    if args.initial_hands_fully_open:argv.append("--initial-hands-fully-open")
    if args.open_hand_transit_park:argv.append("--open-hand-transit-park")
    if args.pause_for_human_review:argv.append("--pause-on-claw-failure")
    if args.coordinated_waist_pitch_offset_rad and not args.native_physics_audit:raise ValueError("coordinated circle execution requires native TCP observation")
    if args.coordinated_waist_pitch_offset_rad:argv.extend(["--coordinated-circle-waist","--disable-mc-waist-publisher"])
    sys.argv=argv
    try:native.main()
    finally:
        if perception_output is not None and perception_output.exists() and not (perception_output/'stop').exists():(perception_output/'stop').write_text('stop')
        if perception_process is not None:
            try:perception_process.wait(timeout=3.)
            except subprocess.TimeoutExpired:perception_process.terminate();perception_process.wait(timeout=3.)
        if tracking_output is not None and not tracking_output.with_suffix('.stop').exists():tracking_output.with_suffix('.stop').write_text('stop')
        if tracking is not None:
            try:tracking.wait(timeout=3.)
            except subprocess.TimeoutExpired:tracking.terminate();tracking.wait(timeout=3.)


if __name__=="__main__":main()
