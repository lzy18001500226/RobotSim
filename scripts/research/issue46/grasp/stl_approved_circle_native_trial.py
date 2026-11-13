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



def configure_native_background_x11(argv, manifest_path, fixed_checkpoint):
    """Existing Xvfb backend, same native simulator/physics, no visible WM."""
    import hashlib
    manifest_path=Path(manifest_path).resolve()
    if not fixed_checkpoint:raise ValueError("background X11 requires verified episode-boundary initialization")
    tools=manifest_path.parent;record=json.loads(manifest_path.read_text())
    for relative,key in (("root/usr/bin/Xvfb","xvfb_sha256"),("root/usr/bin/xkbcomp","xkbcomp_sha256")):
        source=tools/relative
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest()!=record[key]:
            raise ValueError("background X11 tool provenance:"+relative)
    for original,digest in record["sources"].items():
        target=tools/"libs"/Path(original).name
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError("background X11 library provenance")
    result=list(argv)
    if "--interactive-x11-display" not in result:raise ValueError("expected explicit original X11 configuration")
    index=result.index("--interactive-x11-display");del result[index:index+2]
    result.extend(["--xvfb-root",str(tools/"root"),"--xvfb-libs",str(tools/"libs"),"--xkb-root","/usr/share/X11/xkb"])
    return result


def verified_grasp_reuse(episode):
    import hashlib
    episode=Path(episode).resolve();result=json.loads((episode/"result.json").read_text())
    final_gates=all(result["gates"].get(k)=="PASS" for k in ("HAL_PHYSICAL_LIFT","POST_LIFT_RETRACT","HAL_RIGHT_HAND_CLOSE"))
    ready=result.get("carry_ready",{})
    qualified_handoff=(ready.get("physical_lift_pass") is True and ready.get("post_lift_retract_pass") is True
        and result.get("carry_static_hold",{}).get("pass") is True
        and result.get("carry_native_motion_audit",{}).get("pass") is True)
    if not (final_gates or qualified_handoff):
        raise ValueError("reuse requires physically verified grasp/lift/retract")
    plan=Path(result["hal_arm_runtime_replan"]["path"])
    if hashlib.sha256(plan.read_bytes()).hexdigest()!=result["hal_arm_runtime_replan"]["sha256"]:raise ValueError("cached grasp plan changed")
    cycle=episode/"right_claw_gain_no_contact_cycle.json";proof=json.loads(cycle.read_text())
    if not (proof.get("pass") and proof.get("actual_native_physics") and proof.get("zero_object_contact") and proof.get("torque_cap_nm")==2.):
        raise ValueError("prior actual gain cycle is not verified")
    return dict(episode=str(episode),plan=str(plan),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                cycle=str(cycle),cycle_sha256=hashlib.sha256(cycle.read_bytes()).hexdigest(),
                scene_sha256=json.loads(plan.read_text())["scene"]["sha256"])



def verified_claw_cycle_reuse(episode, current_overlay):
    """Reuse only actual same-control/same-algorithm no-contact calibration."""
    proof=verified_grasp_reuse(episode)
    prior_result=json.loads((Path(episode)/"result.json").read_text())
    prior=json.loads(Path(prior_result["claw_control_overlay"]["path"]).read_text())
    current=json.loads(Path(current_overlay).read_text())
    def flag(record):
        return int(record.get("numeric_contact_manifold",{}).get("new_enableflags",0))
    if flag(current)!=flag(prior):raise ValueError("changed contact algorithm needs its own verified gain-cycle combination")
    if flag(prior) and prior_result.get("native_solver_options",{}).get("enableflags")!=flag(prior):raise ValueError("prior algorithm was not observed in native physics")
    def read_parent(ref):
        path=Path(ref["path"])
        if native.sha256(path)!=ref["sha256"]:raise ValueError("fixture parent changed")
        return json.loads(path.read_text())
    def physical_source(record):
        if record.get("numeric_contact_manifold"):
            numeric=record["numeric_contact_manifold"]
            required={"body_mass","body_inertia","jnt_range","jnt_axis","geom_size","actuator_forcerange","actuator_gainprm"}
            if (numeric.get("old_enableflags"),numeric.get("new_enableflags"))!=(0,16) or not required.issubset(numeric.get("geometry_material_mass_gain_limit_arrays_unchanged",[])):raise ValueError("solver-only equivalence proof missing")
            record=read_parent(numeric["parent"])
        episode_start=record.get("episode_start_navigation")
        if episode_start:
            if episode_start.get("keyframe_only_before_stepping") is not True or episode_start.get("physical_arrays_unchanged") is not True:raise ValueError("episode relocation proof missing")
            record=read_parent(episode_start["parent_overlay"])
        initial=record.get('initial_object_placement')
        if initial:
            if initial.get('physical_properties_and_robot_unchanged') is not True or initial.get('mj_step_calls')!=0 or initial.get('controller_knows_object_pose') is not False:raise ValueError('initial free-object pose proof missing')
            record=read_parent(initial['parent'])
        return record
    current=physical_source(current);prior=physical_source(prior)
    if current["generated"]["host_scene"]["sha256"]!=prior["generated"]["host_scene"]["sha256"]:
        fixture=current.get("fixture_height_delta",{})
        if not fixture.get("robot_and_environment_unchanged") or fixture.get("mj_step_calls")!=0 or (fixture.get("source_height_m"),fixture.get("height_m"),fixture.get("diameter_m"),fixture.get("mass_kg"))!=(.30,.22,.07,.2):
            raise ValueError("cycle reuse requires exact source or verified bottle-only H220 fixture")
        original=read_parent(fixture["parent"])["generated"]["host_scene"]
        if original["sha256"]!=prior["generated"]["host_scene"]["sha256"] or native.sha256(Path(original["path"]))!=original["sha256"]:raise ValueError("fixture original robot/control differs from verified gain cycle")
    return {**proof,"scope":"ACTUAL_UNCHANGED_CLAW_GAIN_CYCLE_ONLY_NOT_IK_OR_GRASP_ACCEPTANCE","active_native_enableflags":flag(json.loads(Path(current_overlay).read_text()))}


def runtime_plan_result(result, effective_dir):
    """Make parent and tracker consume the identical effective plan artifact."""
    payload=json.loads(result.stdout.strip().splitlines()[-1])
    payload.update(output=str(effective_dir),output_wsl=str(effective_dir))
    previous=result.stdout.rstrip().rsplit("\n",1)
    prefix=previous[0]+"\n" if len(previous)>1 else ""
    return subprocess.CompletedProcess(result.args,result.returncode,prefix+json.dumps(payload)+"\n",result.stderr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visual-stance-adjustment',action='store_true')
    p.add_argument('--initial-bottle-offset-xy-m',type=float,nargs=2,default=(0.,0.))
    p.add_argument("--task-grasp-region",action="store_true")
    p.add_argument("--far-observe-approach",action="store_true")
    p.add_argument("--lateral-motion-direction",choices=("left","right"));p.add_argument("--lateral-motion-profile",choices=("reference","side-start"),default="reference")
    p.add_argument("--grasp-only",action="store_true")
    p.add_argument("--task-bottle-offset-xy-m",type=float,nargs=2,default=(0.,0.))
    p.add_argument("--native-multiccd",action="store_true",help="separate SIMULATION_ONLY native337 contact-manifold comparison; no geometry/material/gain change")
    p.add_argument("--body-height-probe-drop-m",type=float,choices=(.03,.10,.13))
    p.add_argument("--body-height-native-build",type=Path)
    p.add_argument("--standing-grasp-posture",type=Path,help="bounded native lowering then torso yaw/pitch; measured roll retained")
    p.add_argument("--restore-postured-carry",action="store_true",help="H220 only: gate walking on native height and waist restoration")
    p.add_argument("--execute-pickup-navigation",action="store_true",help="initialize aligned START keyframe then execute existing real table docking")
    p.add_argument("--bottle-height-m",type=float,choices=(.30,.22),default=.30)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--circle-review",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--reuse-verified-grasp-plan",type=Path,help="prior actual grasp episode; reuse IK seed and verified unchanged gain cycle")
    p.add_argument("--reuse-verified-claw-cycle",type=Path,help="verified unchanged gripper cycle only; never reuse its bottle pose or IK")
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
    p.add_argument("--right-claw-kp",type=float,choices=(2.,4.,6.),help="isolated C.1 gain diagnostic; actual no-contact close/open required before approach")
    p.add_argument("--global-contact-impratio-one",action="store_true",help="SIMULATION_ONLY global impedance10->1 comparison; elliptic/local22 retained")
    p.add_argument("--approved-pad-friction-reference",action="store_true",help="2026-11-08 approved22 pad/bottle pairs solreffriction .002s, NoSlip0; unchanged geometry/mu/normal/limits")
    p.add_argument("--noslip-solver-diagnostic",action="store_true",help="separate SIMULATION_ONLY two-iteration numerical postsolver; no contact-material/geometry changes")
    p.add_argument("--elliptic-grip-solver-diagnostic",action="store_true",help="Menagerie-style numerical elliptic cone/impratio=10, unchanged actual friction/condim/geometry")
    p.add_argument("--micro-base-probe",choices=("forward","backward","left","right"))
    p.add_argument("--micro-probe-duration-s",type=float,default=.6)
    p.add_argument("--micro-command-speed",type=float,default=.2)
    p.add_argument("--micro-heading-hold",action="store_true")
    p.add_argument("--micro-target-distance-m",type=float)
    p.add_argument("--observe-native-arm-policy",action="store_true")
    p.add_argument("--capture-mc-rl-debug",action="store_true",help="reuse the reviewed read-only MC debug sidecar; no actuator ownership change")
    p.add_argument("--reviewed-foundationwalk-profile",action="store_true",help="explicitly reuse frozen Issue80 SIMULATION_ONLY diagnostic; no new tuning")
    p.add_argument("--physical-lift-test",action="store_true")
    p.add_argument("--source-pad-coacd-comparison",action="store_true",help="reuse the existing hash-locked source-STL proxy pieces for only the two right distal pads; separate SIMULATION_ONLY physics")
    p.add_argument("--native-arm-policy-relay",action="store_true")
    p.add_argument("--open-hand-transit-park",action="store_true")
    p.add_argument("--initial-hands-fully-open",action="store_true")
    p.add_argument("--pause-for-human-review",action="store_true")
    p.add_argument("--native-physics-audit",action="store_true",help="read-only actual-engine pose/contact/force observer; ABI must match installed library")
    p.add_argument("--target-source",choices=("truth-diagnostic","historical-rgbd-replay","fresh-rgbd","fresh-rgbd-snapshot"),default="truth-diagnostic",help="explicit provenance; snapshot mode reobserves before a bounded stationary-target execution")
    p.add_argument("--rgbd-render-adapter",help="explicit WSLg/Mesa sensor-process GPU selection, e.g. NVIDIA; actual renderer identity must agree")
    p.add_argument("--verified-native-mouse-controls",action="store_true")
    p.add_argument("--native-reset-input",choices=("backspace","button"),default="backspace")
    p.add_argument("--native-background-xvfb-tools",type=Path,help="explicit provenance manifest for existing isolated Xvfb backend; RGB-D host rendering remains separate")
    p.add_argument("--native-render-adapter",help="explicit process-only GPU selection for the native simulator, with actual GL identity verification")
    p.add_argument("--grasp-height-above-center-m",type=float,default=.075,help="grasp section within the canonical 300mm bottle body")
    p.add_argument("--carry-retention-only",action="store_true",help="user-approved empty-bottle carry: record wobble/slip, stop on actual loss/drop")
    p.add_argument("--approved-two-table-layout",action="store_true",help="reviewed scene: delivery90deg/Y1.6, pickup and fixed stance advanced10cm")
    p.add_argument("--carry-compact-angular-cap",type=float,choices=(.30,.60),default=.30)
    p.add_argument("--carry-release-source-on-stop",action="store_true",help="DISABLE only our locomotion source after ZERO; ENABLE after full stop before rearm")
    p.add_argument("--carry-compact-centre-feedback",action="store_true",help="existing world-centre P feedback, <=.20m/s norm, during compact turn only")
    p.add_argument("--carry-vendor-transition-table",action="store_true",help="comparison with unchanged delivered SD/LD action table; no extra500ms rules")
    p.add_argument("--carry-delivery-config",type=Path,help="explicit approved two-table transfer task; not the legacy one-metre test")
    p.add_argument("--carry-gait-prepare",action="store_true",help="bounded native yaw gait wakeup; require measured alternating foot loads before retreat")
    p.add_argument("--carry-inplace-centre-hold",action="store_true")
    p.add_argument("--carry-compact-turn",action="store_true",help="SIMULATION_ONLY .30rad/s pure yaw; .30m active radius / .20m final centre bound")
    p.add_argument("--carry-turn-recovery-speed",type=float,choices=(.12,.20),default=.12)
    p.add_argument("--carry-transit-command-speed",type=float,choices=(.12,.20),default=.12,help="explicit SIMULATION_ONLY effective SDK020 transit; no MC/model changes")
    p.add_argument("--carry-turn-brake-delay-s",type=float,default=0.)
    p.add_argument("--carry-retreat-brake-delay-s",type=float,default=0.)
    p.add_argument("--carry-straight-retreat-turn",action="store_true",help="actual30cm retreat then native in-place180, each with composite stop")
    p.add_argument("--allow-carried-bottle-tilt",action="store_true",help="empty bottle: world tilt is recorded, not a carry failure; retention and robot guards stay active")
    p.add_argument("--carry-away-1m",action="store_true",help="leave table, turn to opposite heading, then carry forward >=1m with measured stops")
    p.add_argument("--carry-goal-world-xy",type=float,nargs=2,help="explicit known-map delivery goal after physical grasp/retract")
    p.add_argument("--carry-timeout-s",type=float,default=90.)
    p.add_argument("--carry-grip-force-margin",type=float,choices=(1.5,3.0,6.0,7.5,15.0),default=1.5)
    p.add_argument("--carry-reverse-moving-arc",action="store_true")
    p.add_argument("--carry-transit-speed-policy-050",action="store_true")
    p.add_argument("--carry-native-step-mode",action="store_true",help="explicit source-pinned native LOCOMOTION_STEP alternative; no policy/config/gain edits")
    p.add_argument("--carry-native-action-smoothing",action="store_true",help="separate native ActionSelector SD/LD 500ms transition diagnostic; no runner/gain/policy edits")
    p.add_argument("--carry-sdk-minimum-turn",action="store_true",help="explicit .10 rad/s SDK-minimum turn diagnostic; physical root/joint caps unchanged")
    p.add_argument("--post-lift-retract-world-m",type=float,nargs=3,help="bounded <=250mm post-lift retreat with unchanged physical load gate")
    p.add_argument("--grasp-axis-tilt-rad",type=float,default=0.)
    p.add_argument("--inverted-bottle-axis",action="store_true")
    p.add_argument("--grasp-axis-calibration",type=Path)
    p.add_argument("--use-measured-load-frame",action="store_true")
    p.add_argument("--coordinated-waist-pitch-offset-rad",type=float,default=0.)
    p.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument("--dock-world-xy",type=float,nargs=2,default=(.85,-1.44))
    p.add_argument("--dock-position-tolerance-m",type=float,default=.10)
    p.add_argument("--refine-position-tolerance-m",type=float,default=.15,help="position-only docking terminal tolerance")
    p.add_argument("--dock-angular-speed-cap",type=float)
    p.add_argument("--dock-command-speed-cap",type=float)
    p.add_argument("--dock-braking-delay-s",type=float)
    p.add_argument("--arm-plan-duration-s",type=float,default=12.0)
    p.add_argument("--arm-plan-hold-s",type=float,default=5.0)
    args=p.parse_args()
    if args.right_claw_kp is not None and not args.physical_lift_test:
        raise ValueError("right gain comparison requires the instrumented physical-lift/carry path")
    if (args.carry_compact_angular_cap!=.30 or args.carry_release_source_on_stop) and not args.carry_compact_turn:raise ValueError("compact source/cap scope")
    source_step_all=bool(args.carry_delivery_config and json.loads(args.carry_delivery_config.read_text()).get("transit_motion_mode")=="LOCOMOTION_STEP")
    if args.carry_vendor_transition_table and (args.carry_delivery_config is None or args.carry_native_action_smoothing or args.carry_native_step_mode and not source_step_all):raise ValueError("vendor transition comparison scope")
    if args.carry_delivery_config is not None:
        if any((json.loads(args.carry_delivery_config.read_text()).get("carry_native_neutral_stow") or {}).get(k) for k in ("follow_native_unloaded_arm_after_turn","native_arm_follow_scope")) and not args.observe_native_arm_policy:raise ValueError("native unloaded arm following requires verified private MC arm producer")
        direct_retreat=json.loads(args.carry_delivery_config.read_text()).get("activation_protocol")=="VENDOR_DIRECT_RETREAT_NONZERO" and args.carry_transit_command_speed==.20
        if not (args.carry_compact_turn and (args.carry_gait_prepare or direct_retreat) and args.carry_release_source_on_stop and args.carry_compact_centre_feedback and (not args.carry_native_step_mode or source_step_all)):raise ValueError("delivery profile requires qualified compact controls or explicit vendor-direct retreat")
    if args.carry_compact_centre_feedback and not args.carry_compact_turn:raise ValueError("compact centre feedback scope")
    if args.carry_gait_prepare and not args.carry_compact_turn:raise ValueError("gait preparation requires compact carry")
    if args.carry_compact_turn and not (args.carry_straight_retreat_turn and args.carry_transit_speed_policy_050 and not args.carry_inplace_centre_hold):raise ValueError("compact pure-turn scope")
    if args.carry_transit_command_speed>.12 and not (args.carry_straight_retreat_turn and args.carry_transit_speed_policy_050):
        raise ValueError("effective020 transit requires explicit bounded carry task")
    if (args.carry_inplace_centre_hold or args.carry_turn_recovery_speed>.12 or args.carry_turn_brake_delay_s) and not args.carry_straight_retreat_turn:
        raise ValueError("in-place controls require their explicit bounded task")
    if args.allow_carried_bottle_tilt and args.carry_goal_world_xy is None and not args.carry_away_1m:raise ValueError("empty-bottle tilt policy is scoped to verified carry tasks")
    if args.carry_away_1m and (args.carry_goal_world_xy is not None or (args.carry_native_step_mode and not args.carry_straight_retreat_turn) or args.carry_reverse_moving_arc or not args.carry_transit_speed_policy_050):
        raise ValueError("away1m requires its distinct bounded FoundationWalk transit mission")

    if args.review_captured_run:
        from scripts.research.issue46.native_runtime.native_grasp_audit import review_native_capture
        review_native_capture(args.review_captured_run,dry_run=args.review_dry_run,full_flow=args.review_full_flow)
        return
    if not args.execute_simulation:raise ValueError("explicit simulation execution flag required")
    if any(not math.isfinite(v) or abs(v)>.02000001 for v in args.checkpoint_base_offset_xy_m) or any(args.checkpoint_base_offset_xy_m) and not args.fixed_reviewed_checkpoint:raise ValueError('stance offset requires a fixed reviewed episode and <=20mm bounds')
    if args.use_measured_load_frame and (args.grasp_axis_calibration is None or not args.elliptic_grip_solver_diagnostic):raise ValueError('measured load frame requires its evidence and matched numerical profile')
    if args.post_lift_retract_world_m is not None:
        v=args.post_lift_retract_world_m
        if not args.physical_lift_test or any(not math.isfinite(x) for x in v) or math.sqrt(sum(x*x for x in v))>.25 or not -.02<=v[2]<=.05:
            raise ValueError('bounded retract requires physical lift and <=250mm translation')
    from scripts.research.issue46.native_runtime.native_grasp_stand_restore import postured_carry_requested
    restoring_carry=postured_carry_requested(args)
    if (args.carry_goal_world_xy is not None or getattr(args,"carry_away_1m",False)) and (not args.physical_lift_test or not args.reviewed_foundationwalk_profile or args.post_lift_retract_world_m is None or args.coordinated_waist_pitch_offset_rad or not 5.<=args.carry_timeout_s<=90. or any(not math.isfinite(x) or abs(x)>1.65 for x in (args.carry_goal_world_xy or []))):
        raise ValueError('carry needs screened arm-only physical lift/retract and bounded room goal/deadline')
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
    gain_cycle_reuse=None
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
            if args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot'):
                from scripts.research.issue46.perception.native_rgbd_grasp import await_initial_estimate
                perception_output=state.parent/'fresh_rgbd'
                scene=json.loads(args.candidate.read_text())['scene']['path']
                sensor_command=['/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python',str(REPO/'scripts/research/issue46/perception/native_rgbd_grasp.py'),'--stream-scene',scene,'--native-latest',str(state.parent/'native_latest.json'),'--output-dir',str(perception_output),'--half-height-m',str(args.bottle_height_m/2.)]
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
                        observation,pose=await_initial_estimate(latest,generation=row['generation']);break
                    time.sleep(.02)
                else:raise RuntimeError('RGBD_SENSOR_READINESS_TIMEOUT')
                synchronized['object_pose_world']={'xyz_m':pose['position_m'],'quaternion_wxyz':pose['quaternion_wxyz']}
                synchronized['perception']={'latest_path':str(latest),'source':'FRESH_SIMULATED_CALIBRATED_HEAD_RGBD','truth_not_control_target':True}
            if args.physical_lift_test or (args.carry_goal_world_xy is not None or getattr(args,"carry_away_1m",False)):
                carry_source_limits,carry_source_provenance=native.native_leg_safety_contract(include_upper_body=True)
                synchronized["runtime_source_motion_limits"]=carry_source_limits
                synchronized["runtime_source_motion_provenance"]=carry_source_provenance
            if gain_cycle_reuse is not None:
                seed_plan=json.loads(Path(gain_cycle_reuse["plan"]).read_text())
                names=("right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint")
                synchronized["runtime_initial_arm_guess_rad"]={n:float(seed_plan["rows"][0]["q_rad"][n]) for n in names}
                synchronized["runtime_initial_arm_guess_provenance"]={"source_plan":gain_cycle_reuse["plan"],"sha256":gain_cycle_reuse["plan_sha256"],"scope":"SEVEN_JOINT_INITIAL_GUESS_ONLY_CURRENT_CAMERA_ROBOT_STATE_AND_ALL_GATES_RECOMPUTED"}
            if args.standing_grasp_posture and args.carry_delivery_config:
                synchronized["forearm_first_approach"]=json.loads(args.carry_delivery_config.read_text()).get("approach_posture_policy")=="FOREARM_FIRST_TABLE_OUTSIDE"
                synchronized["forearm_already_raised"]=json.loads(args.carry_delivery_config.read_text()).get("approach_posture_policy")=="PRE_RAISED_FOREARM_THEN_REACH"
            state=state.parent/"native_synchronized_dock_state.json"
            pg.write_new_text(state,json.dumps(synchronized,indent=2)+"\n")
            state=str(state)
            # Navigation already applied the requested docking offset. The new
            # state is MEASURED; adding that offset again invents a root pose.
            command=["/home/lzy18001500226/.local/bin/uv","run","--offline","--with","mujoco==3.3.6","--with","numpy==2.2.6","--with","scipy==1.15.3","python3",str(Path(__file__).with_name("stl_approved_circle_ik.py")),"--circle-review",str(args.circle_review),"--candidate",str(args.candidate),"--standing-state",state,"--grasp-height-above-center-m","0.075","--runtime-handoff"]
            command.extend(["--target-source",args.target_source,"--free-bottle-axis-roll"])
            if args.grasp_only or args.far_observe_approach:command.append("--source-arm-planning-range")
            if host_model_cache:command.extend(['--host-model-cache',str(host_model_cache)])
            if args.reuse_verified_grasp_plan:
                reuse=verified_grasp_reuse(args.reuse_verified_grasp_plan)
                if reuse["scene_sha256"]!=json.loads(args.candidate.read_text())["scene"]["sha256"]:raise ValueError("reuse requires unchanged compiled scene/control configuration")
                command.extend(["--warm-start-plan",reuse["plan"]])
            if args.physical_lift_test:command.append('--physical-lift-test')
            if args.approved_pad_friction_reference:command.extend(['--lift-grip-force-margin',str(args.carry_grip_force_margin)])
            if args.post_lift_retract_world_m is not None:command.extend(['--post-lift-retract-world-m',*map(str,args.post_lift_retract_world_m)])
            if (args.carry_goal_world_xy is not None or getattr(args,"carry_away_1m",False)):
                carry_source_limits,_=native.native_leg_safety_contract(include_upper_body=True)
                command.extend(['--carry-left-roll-source-range',*map(str,carry_source_limits['left_shoulder_roll_joint'][:2])])

            command[command.index("--grasp-height-above-center-m")+1]=str(args.grasp_height_above_center_m)
            command.extend(['--grasp-axis-tilt-rad',str(args.grasp_axis_tilt_rad)])
            if args.inverted_bottle_axis:command.append('--inverted-bottle-axis')
            if args.grasp_axis_calibration:command.extend(['--grasp-axis-calibration',str(args.grasp_axis_calibration)])
            if args.use_measured_load_frame:command.append('--use-measured-load-frame')
            if args.coordinated_waist_pitch_offset_rad:
                command.extend(["--waist-pitch-offset-rad",str(args.coordinated_waist_pitch_offset_rad)])
            if args.standing_grasp_posture:command.extend(["--right-arm-seed",str(args.standing_grasp_posture)])
            kwargs["timeout_s"]=45.
        result=original(command,**kwargs)
        if (any(str(v).endswith('/stl_approved_circle_ik.py') for v in command) and '--runtime-handoff' in command
            and '--warm-start-plan' in command and result.returncode==0
            and json.loads(result.stdout.strip().splitlines()[-1]).get('gate')!='PASS'):
            # A reused initial guess is not acceptance. Fall back once to the
            # existing full branch search, still before any grasp command.
            evidence=Path(command[command.index('--standing-state')+1]).parent
            pg.write_new_text(evidence/'cached_grasp_branch_rejected.stdout',result.stdout)
            cold=list(command);i=cold.index('--warm-start-plan');del cold[i:i+2]
            old_state=json.loads(Path(cold[cold.index('--standing-state')+1]).read_text())
            native_grasp_audit.require_fresh_sample(evidence/'native_latest.json')
            fresh=native_grasp_audit.planning_state_from_native(json.loads((evidence/'native_latest.json').read_text()),old_state)
            state_path=evidence/'native_synchronized_fallback_state.json'
            pg.write_new_text(state_path,json.dumps(fresh,indent=2)+'\n')
            cold[cold.index('--standing-state')+1]=str(state_path)
            pg.write_new_text(evidence/'cached_grasp_fallback_command.json',json.dumps(cold,indent=2)+'\n')
            result=original(cold,**kwargs)
        if any(str(v).endswith('/stl_approved_circle_ik.py') for v in command) and '--runtime-handoff' in command and result.returncode==0 and json.loads(result.stdout.strip().splitlines()[-1]).get('gate')=='PASS':
            reference=Path(json.loads(result.stdout.strip().splitlines()[-1])['output_wsl'])/'result.json'
            evidence=Path(command[command.index('--standing-state')+1]).parent
            if args.right_claw_kp is not None:
                extended=json.loads(reference.read_text())
                if args.reuse_verified_claw_cycle:extended["reused_gain_cycle"]=gain_cycle_reuse
                elif args.reuse_verified_grasp_plan:extended["reused_gain_cycle"]=verified_grasp_reuse(args.reuse_verified_grasp_plan)
                else:extended['native_no_contact_gain_cycle_budget_s']=9.
                reference=evidence/'native_gain_cycle_tracking_reference.json'
                pg.write_new_text(reference,json.dumps(extended,indent=2)+'\n')
            if args.target_source=='fresh-rgbd-snapshot':
                from scripts.research.issue46.perception.native_rgbd_grasp import await_initial_estimate,make_stationary_snapshot
                plan=json.loads(reference.read_text());planned=plan['perception']['observation_at_plan']
                verification,_=await_initial_estimate(perception_output/'latest.json',generation=planned['generation'],after_source_ns=planned['source_host_ns'])
                snapshot=make_stationary_snapshot(planned,verification,generation=planned['generation'])
                snapshot_path=evidence/'rgbd_grasp_snapshot.json';pg.write_new_text(snapshot_path,json.dumps(snapshot,indent=2)+'\n')
                plan['static_replan_source']={'path':str(reference),'sha256':native.sha256(reference)}
                plan['perception']['snapshot']={'path':str(snapshot_path),'sha256':native.sha256(snapshot_path)}
                reference=evidence/'rgbd_snapshot_tracking_reference.json';pg.write_new_text(reference,json.dumps(plan,indent=2)+'\n')
                pg.write_new_text(perception_output/'stop','snapshot acquired and reverified before arm execution\n')
                perception_process.wait(timeout=3.)
                if perception_process.returncode:raise RuntimeError('RGBD_SNAPSHOT_SENSOR_TEARDOWN_FAILED')
            if args.reuse_verified_grasp_plan or args.standing_grasp_posture or args.reuse_verified_claw_cycle:
                effective=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="approved-circle-cached-runtime-contract").native_path
                pg.write_new_text(effective/"result.json",reference.read_text())
                reference=effective/"result.json"
                result=runtime_plan_result(result,effective)
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
        claw=build(claw,both_hands=args.both_claws_zero_internal_friction,right_damping=args.right_claw_damping,right_kp=args.right_claw_kp,noslip_iterations=2 if args.noslip_solver_diagnostic else 0,elliptic_grip_solver=args.elliptic_grip_solver_diagnostic)
    elif args.right_claw_kp is not None or args.right_claw_damping is not None or args.noslip_solver_diagnostic or args.elliptic_grip_solver_diagnostic:raise ValueError('parameter/solver diagnostics require an explicit existing zero-internal-friction profile')
    if args.approved_pad_friction_reference:
        if args.noslip_solver_diagnostic or not args.source_pad_coacd_comparison or not args.elliptic_grip_solver_diagnostic:raise ValueError('approved local tangent reference requires existing22 source pieces/elliptic10 and NoSlip0')
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import add_approved_pad_friction_reference
        claw=add_approved_pad_friction_reference(claw)
    if args.global_contact_impratio_one:
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import compare_global_impratio_one
        claw=compare_global_impratio_one(claw)
    if args.fixed_reviewed_checkpoint:
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import add_reviewed_checkpoint
        claw=add_reviewed_checkpoint(claw,args.fixed_reviewed_checkpoint,base_offset_xy=args.checkpoint_base_offset_xy_m)
    if args.approved_two_table_layout:
        if not args.fixed_reviewed_checkpoint:raise ValueError("two-table fixed-stance diagnostic requires reviewed episode initialization")
        from scripts.research.issue46.scene.delivery_table_overlay import build as scene_build
        layout=scene_build(claw,90.,.10,.10,.10)
        claw=Path(json.loads(Path(layout).read_text())["native_overlay_result"]["path"])
    if args.bottle_height_m==.22:
        if abs(args.grasp_height_above_center_m-.01)>1e-9:raise ValueError("H220 task requires section120mm above bottom")
        from scripts.research.issue46.scene.bottle_height_overlay import build as height_build
        claw=height_build(claw,.22)
    if any(args.task_bottle_offset_xy_m):
        if not args.task_grasp_region:raise ValueError("task object initialization requires sensor-directed task")
        from scripts.research.issue46.scene.bottle_initial_pose_overlay import build as initial_pose_build
        claw=initial_pose_build(claw,args.task_bottle_offset_xy_m)
    if any(args.initial_bottle_offset_xy_m):
        if not args.visual_stance_adjustment:raise ValueError('object placement is an unknown initial scenario only')
        from scripts.research.issue46.scene.bottle_initial_pose_overlay import build as initial_pose_build
        claw=initial_pose_build(claw,args.initial_bottle_offset_xy_m)
    if args.body_height_probe_drop_m is not None:
        if args.execute_pickup_navigation or args.carry_away_1m or args.carry_goal_world_xy is not None or args.micro_base_probe or args.carry_delivery_config:
            raise ValueError("native height compatibility probe cannot execute a locomotion route or grasp")
        if args.body_height_native_build is None:
            from scripts.research.issue46.native_runtime.native_body_height import prepare_publisher
            args.body_height_native_build=prepare_publisher()
    if args.execute_pickup_navigation:
        if not args.fixed_reviewed_checkpoint or not args.approved_two_table_layout:raise ValueError("START execution requires approved native episode keyframe/room")
        from scripts.research.issue46.scene.episode_start_overlay import build as start_build
        claw=start_build(claw,REPO/"simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json")
        args.dock_world_xy=tuple(json.loads(claw.read_text())["episode_start_navigation"]["pickup_goal_world_xyz_m"][:2])
        if args.task_grasp_region:
            import mujoco
            from scripts.research.issue46.scene.table_edge_grid import table_bounds, observation_goal
            record=json.loads(claw.read_text())
            m=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
            d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
            task_observation=observation_goal(table_bounds(m,d))
            if args.far_observe_approach:
                task_observation["xy_m"][0]-=.30
                task_observation["xy_m"][1]-=.12
            args.dock_world_xy=tuple(task_observation["xy_m"])

    if args.grasp_only and not args.execute_pickup_navigation:
        if not args.task_grasp_region:raise ValueError("grasp-only task initialization scope")
        from scripts.research.issue46.scene.episode_start_overlay import build as observation_build
        claw=observation_build(claw,REPO/"simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json",observation_only=True)
    if args.native_multiccd:
        from scripts.research.issue46.scene.multiccd_solver_overlay import build as multiccd_build
        claw=multiccd_build(claw)
    if args.carry_delivery_config is not None:
        if not args.approved_two_table_layout:raise ValueError("delivery requires the exact approved two-table scene")
        from scripts.research.issue46.native_runtime.native_carry_placement import prepare_delivery_scene
        claw=prepare_delivery_scene(claw,args.carry_delivery_config)
    if args.carry_retention_only and (not args.allow_carried_bottle_tilt or not args.carry_away_1m):
        raise ValueError("retention-only is explicitly scoped to qualified empty-bottle carry")
    if args.source_pad_coacd_comparison:
        # The static collider must match the actual selected runtime variant;
        # retain the original candidate and source scene as immutable inputs.
        original_candidate=args.candidate;candidate=json.loads(original_candidate.read_text())
        candidate['scene']=dict(json.loads(claw.read_text())['generated']['host_scene'])
        candidate['source_pad_collision_comparison']={'source_candidate':str(original_candidate),'source_candidate_sha256':native.sha256(original_candidate),'overlay':str(claw),'overlay_sha256':native.sha256(claw),'classification':'SIMULATION_ONLY_SOURCE_DERIVED_COLLISION_COMPARISON'}
        candidate_dir=pg.create_unique_output_directory(Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46'),prefix='source-pad-coacd-ik-input').native_path
        pg.write_new_text(candidate_dir/'candidate.json',json.dumps(candidate,indent=2)+'\n');pg.write_artifact_manifest(candidate_dir);args.candidate=candidate_dir/'candidate.json'
    if args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot') and args.body_height_probe_drop_m is None:
        from scripts.research.issue46.perception.native_rgbd_grasp import prepare_host_cache
        host_model_cache=prepare_host_cache(Path(json.loads(args.candidate.read_text())['scene']['path']))
    if args.reuse_verified_claw_cycle:gain_cycle_reuse=verified_claw_cycle_reuse(args.reuse_verified_claw_cycle,claw)
    helper=REPO/"scripts/research/issue46/native_runtime/native_mujoco_reset.py"
    display=__import__("os").environ.get("DISPLAY")
    if not display:raise RuntimeError("DISPLAY is unset; run this command from the WSLg terminal")
    dock_xy=args.dock_world_xy or (0.74+args.base_offset_xyz_m[0],-1.44+args.base_offset_xyz_m[1])
    argv=["native", "--table-overlay-result",str(table),"--table-overlay-result-sha256",native.sha256(table),"--claw-control-overlay-result",str(claw),"--claw-control-overlay-result-sha256",native.sha256(claw),"--reset-helper",str(helper),"--reset-helper-sha256",native.sha256(helper),"--interactive-x11-display",display,"--hal-arm-candidate",str(args.candidate),"--hal-arm-candidate-sha256",native.sha256(args.candidate),"--disable-mc-arm-publisher","--disable-mc-hand-publisher","--hal-arm-plan-mode","pregrasp-hold","--hal-right-hand-command","0.0","--hal-hand-close-hold-s","2","--dock-world-xy",str(dock_xy[0]),str(dock_xy[1]),"--dock-yaw-rad","-0.07591850724454846","--dock-position-only","--use-official-cmd-vel-bridge","--dock-timeout-s","30"]
    if args.task_grasp_region:
        argv[argv.index("--dock-yaw-rad")+1]="0.0"
    if args.micro_base_probe:
        if args.target_source!="truth-diagnostic" or args.physical_lift_test or args.pause_for_human_review or args.fixed_reviewed_checkpoint or not args.initial_hands_fully_open:
            raise ValueError("isolated micro capability diagnostic cannot execute grasp")
        start=argv.index("--dock-world-xy")
        argv=argv[:start]+["--use-official-cmd-vel-bridge","--micro-base-probe",args.micro_base_probe,"--micro-probe-duration-s",str(args.micro_probe_duration_s),"--micro-command-speed",str(args.micro_command_speed)]
    if args.micro_target_distance_m is not None:argv.extend(["--micro-target-distance-m",str(args.micro_target_distance_m)])
    if args.micro_heading_hold:
        if not args.micro_base_probe:raise ValueError("micro heading hold requires isolated micro probe")
        argv.append("--micro-heading-hold")
    if args.observe_native_arm_policy:argv.append("--observe-native-arm-policy")
    if args.capture_mc_rl_debug:argv.append("--capture-mc-rl-debug")
    if args.reviewed_foundationwalk_profile:argv.append("--reviewed-foundationwalk-profile")
    if args.fixed_reviewed_checkpoint:argv.append("--fixed-reviewed-checkpoint")
    if args.execute_pickup_navigation:argv.append("--execute-pickup-navigation")
    if args.lateral_motion_direction:argv.extend(["--lateral-motion-direction",args.lateral_motion_direction,"--lateral-motion-profile",args.lateral_motion_profile])
    if args.grasp_only:argv.append("--grasp-only")
    if args.far_observe_approach:argv.append("--far-observe-approach")
    if args.body_height_probe_drop_m is not None:
        argv.extend(["--body-height-probe-drop-m",str(args.body_height_probe_drop_m),"--body-height-native-build",str(args.body_height_native_build),"--body-height-native-build-sha256",native.sha256(args.body_height_native_build)])
    if args.dock_angular_speed_cap is not None:argv.extend(["--dock-angular-speed-cap",str(args.dock_angular_speed_cap)])
    if args.dock_command_speed_cap is not None:argv.extend(["--dock-command-speed-cap",str(args.dock_command_speed_cap)])
    if args.verified_native_mouse_controls and not args.native_background_xvfb_tools:argv.append("--verified-native-mouse-controls")
    argv.extend(["--native-reset-input",args.native_reset_input])
    if args.native_background_xvfb_tools:
        argv=configure_native_background_x11(argv,args.native_background_xvfb_tools,args.fixed_reviewed_checkpoint)
    elif args.native_render_adapter:argv.extend(['--native-render-adapter',args.native_render_adapter])
    if args.carry_reverse_moving_arc and ((args.carry_goal_world_xy is None and not getattr(args,"carry_away_1m",False)) or not args.carry_transit_speed_policy_050 or args.carry_native_step_mode):raise ValueError("reverse arc requires bounded FoundationWalk carry, with approved transit policy")
    if args.carry_transit_speed_policy_050 and not ((args.carry_goal_world_xy is not None or getattr(args,"carry_away_1m",False)) and args.approved_pad_friction_reference):raise ValueError("TRANSIT050 requires isolated carry and approved local contact profile")
    if args.carry_native_step_mode and not args.carry_native_action_smoothing and not (source_step_all and args.carry_vendor_transition_table):raise ValueError('native step diagnostic requires original validated table or explicit transition diagnostic')
    if (args.carry_sdk_minimum_turn or args.carry_native_action_smoothing or args.carry_native_step_mode) and (args.carry_goal_world_xy is None and not getattr(args,"carry_away_1m",False)):raise ValueError('SDK turn diagnostic requires a carry goal')
    if (args.carry_goal_world_xy is not None or getattr(args,"carry_away_1m",False)):
        argv.extend(['--carry-timeout-s',str(args.carry_timeout_s),'--carry-grip-force-margin',str(args.carry_grip_force_margin)])
        if args.allow_carried_bottle_tilt:argv.append('--allow-carried-bottle-tilt')
        if args.carry_retention_only:argv.append('--carry-retention-only')
        if args.carry_straight_retreat_turn:argv.append('--carry-straight-retreat-turn')
        if args.carry_inplace_centre_hold:argv.append('--carry-inplace-centre-hold')
        if args.carry_compact_turn:argv.append('--carry-compact-turn')
        if args.carry_gait_prepare:argv.append('--carry-gait-prepare')
        if args.carry_vendor_transition_table:argv.append('--carry-vendor-transition-table')
        if args.carry_delivery_config is not None:argv.extend(['--carry-delivery-config',str(args.carry_delivery_config.resolve())])
        if args.carry_compact_centre_feedback:argv.append('--carry-compact-centre-feedback')
        if args.carry_compact_angular_cap!=.30:argv.extend(['--carry-compact-angular-cap',str(args.carry_compact_angular_cap)])
        if args.carry_release_source_on_stop:argv.append('--carry-release-source-on-stop')
        if args.carry_transit_command_speed!=.12:argv.extend(['--carry-transit-command-speed',str(args.carry_transit_command_speed)])
        if args.carry_turn_recovery_speed!=.12:argv.extend(['--carry-turn-recovery-speed',str(args.carry_turn_recovery_speed)])
        if args.carry_turn_brake_delay_s:argv.extend(['--carry-turn-brake-delay-s',str(args.carry_turn_brake_delay_s)])
        if args.carry_retreat_brake_delay_s:argv.extend(['--carry-retreat-brake-delay-s',str(args.carry_retreat_brake_delay_s)])
        if args.carry_away_1m:argv.append('--carry-away-1m')
        else:argv.extend(['--carry-goal-world-xy',*map(str,args.carry_goal_world_xy)])
        if args.carry_sdk_minimum_turn:argv.append('--carry-sdk-minimum-turn')
        if args.carry_native_action_smoothing:argv.append('--carry-native-action-smoothing')
        if args.carry_native_step_mode:argv.append('--carry-native-step-mode')
        if args.carry_transit_speed_policy_050:argv.append('--carry-transit-speed-policy-050')
        if args.carry_reverse_moving_arc:argv.append('--carry-reverse-moving-arc')
    if args.physical_lift_test:
        argv.append('--physical-lift-test')
        argv[argv.index('--hal-hand-close-hold-s')+1]='5'
    if args.visual_stance_adjustment:
        if not args.execute_pickup_navigation or host_model_cache is None or args.target_source!='fresh-rgbd-snapshot':raise ValueError('sensor-only stance requires current RGBD/cache and START navigation')
        from scripts.research.issue46.native_runtime.native_visual_stance import reference_contract
        reference_dir=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='rgbd-stance-control-reference').native_path
        pg.write_new_text(reference_dir/'reference.json',json.dumps(reference_contract(),indent=2)+'\n');pg.write_artifact_manifest(reference_dir)
        argv.extend(['--visual-stance-adjustment','--visual-stance-host-cache',str(host_model_cache),'--visual-stance-reference',str(reference_dir/'reference.json')])
    if args.task_grasp_region:
        if (not args.execute_pickup_navigation and not args.grasp_only) or host_model_cache is None or args.target_source!="fresh-rgbd-snapshot":
            raise ValueError("task region requires START, fresh RGBD and matched camera cache")
        from scripts.research.issue46.native_runtime.native_task_grasp_region import reference_contract
        reference_dir=pg.create_unique_output_directory("/home/lzy18001500226/projects/RobotSim/reviews/issue-46",prefix="rgbd-task-grasp-control-reference").native_path
        pg.write_new_text(reference_dir/"reference.json",json.dumps(reference_contract(far_observe=args.far_observe_approach),indent=2)+"\n")
        pg.write_artifact_manifest(reference_dir)
        argv.extend(["--task-grasp-region","--task-region-host-cache",str(host_model_cache),"--task-region-reference",str(reference_dir/"reference.json")])
    argv.extend(["--dock-position-tolerance-m",str(args.dock_position_tolerance_m)])
    if not .03<=args.refine_position_tolerance_m<=(.25 if restoring_carry else .15):raise ValueError("bounded position-only docking tolerance")
    argv.extend(["--refine-position-tolerance-m",str(args.refine_position_tolerance_m)])
    if args.dock_braking_delay_s is not None:
        if not .0<args.dock_braking_delay_s<=1. or args.target_source not in ('truth-diagnostic','fresh-rgbd','fresh-rgbd-snapshot'):raise ValueError('measured braking profile requires bounded delay and native feedback')
        argv.extend(['--dock-braking-delay-s',str(args.dock_braking_delay_s)])
    argv.extend(["--arm-plan-duration-s",str(args.arm_plan_duration_s),"--arm-plan-hold-s",str(args.arm_plan_hold_s)])
    if args.target_source in ("truth-diagnostic","fresh-rgbd","fresh-rgbd-snapshot"):argv.extend(["--native-dock-truth-feedback","--native-circle-tracking"])
    if args.startup_opening_diagnostic:argv.append("--startup-opening-diagnostic")
    if args.initial_hands_fully_open:argv.append("--initial-hands-fully-open")
    if args.native_arm_policy_relay:argv.append("--native-arm-policy-relay")
    if args.open_hand_transit_park:argv.append("--open-hand-transit-park")
    if args.pause_for_human_review and not args.native_background_xvfb_tools:argv.append("--pause-on-claw-failure")
    if args.coordinated_waist_pitch_offset_rad and not args.native_physics_audit:raise ValueError("coordinated circle execution requires native TCP observation")
    if args.coordinated_waist_pitch_offset_rad or args.standing_grasp_posture:argv.extend(["--coordinated-circle-waist","--disable-mc-waist-publisher"])
    if args.standing_grasp_posture:
        if not args.fixed_reviewed_checkpoint or (args.carry_delivery_config and not restoring_carry) or not args.physical_lift_test:raise ValueError("postured carry requires verified native standing restoration")
        if args.body_height_native_build is None:
            from scripts.research.issue46.native_runtime.native_body_height import prepare_publisher
            args.body_height_native_build=prepare_publisher()
        argv.extend(["--standing-grasp-posture",str(args.standing_grasp_posture.resolve()),"--body-height-native-build",str(args.body_height_native_build),"--body-height-native-build-sha256",native.sha256(args.body_height_native_build)])
    if restoring_carry:argv.append("--restore-postured-carry")
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
