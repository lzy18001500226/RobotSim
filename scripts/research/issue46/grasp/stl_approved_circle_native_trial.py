"""Bounded adapter to the existing native standing/docking/pregrasp runner.

Only the static replan subprocess is replaced by approved-circle IK. Native
startup, ownership, contact-model overlay, stopping and teardown are unchanged.
This executes pregrasp/closure first; no lift is attempted after a failed gate.
"""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as native
from scripts.agent import path_governance as pg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--circle-review",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--execute-simulation",action="store_true")
    p.add_argument("--initial-hands-fully-open",action="store_true")
    p.add_argument("--pause-for-human-review",action="store_true")
    p.add_argument("--native-physics-audit",action="store_true",help="read-only actual-engine pose/contact/force observer; ABI must match installed library")
    p.add_argument("--target-source",choices=("truth-diagnostic","historical-rgbd-replay"),default="truth-diagnostic",help="truth baseline or explicitly historical RGB-D replay; not fresh camera capture")
    p.add_argument("--grasp-height-above-center-m",type=float,default=.075,help="grasp section within the canonical 300mm bottle body")
    p.add_argument("--coordinated-waist-pitch-offset-rad",type=float,default=0.)
    p.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument("--dock-world-xy",type=float,nargs=2,default=(.89,-1.44))
    p.add_argument("--arm-plan-duration-s",type=float,default=12.0)
    p.add_argument("--arm-plan-hold-s",type=float,default=5.0)
    args=p.parse_args()
    if not args.execute_simulation:raise ValueError("explicit simulation execution flag required")
    baseline=json.loads((args.baseline/"result.json").read_text())
    if not args.native_physics_audit:
        raise ValueError("approved-circle execution requires --native-physics-audit")
    original=native.run
    if original(["docker","ps","-q"],timeout_s=5).stdout.strip():raise RuntimeError("conflicting container")
    def run(command,**kwargs):
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
            state=state.parent/"native_synchronized_dock_state.json"
            pg.write_new_text(state,json.dumps(synchronized,indent=2)+"\n")
            state=str(state)
            # Navigation already applied the requested docking offset. The new
            # state is MEASURED; adding that offset again invents a root pose.
            command=["/home/lzy18001500226/.local/bin/uv","run","--offline","--with","mujoco==3.3.6","--with","numpy==2.2.6","--with","scipy==1.15.3","python3",str(Path(__file__).with_name("stl_approved_circle_ik.py")),"--circle-review",str(args.circle_review),"--candidate",str(args.candidate),"--standing-state",state,"--grasp-height-above-center-m","0.075","--runtime-handoff"]
            command.extend(["--target-source",args.target_source,"--free-bottle-axis-roll"])
            command[command.index("--grasp-height-above-center-m")+1]=str(args.grasp_height_above_center_m)
            if args.coordinated_waist_pitch_offset_rad:
                command.extend(["--waist-pitch-offset-rad",str(args.coordinated_waist_pitch_offset_rad)])
            kwargs["timeout_s"]=45.
        return original(command,**kwargs)
    native.run=run
    table=Path(baseline["table_overlay"]["path"])
    claw=Path(baseline["claw_control_overlay"]["path"])
    helper=REPO/"scripts/research/issue46/native_runtime/native_mujoco_reset.py"
    display=__import__("os").environ.get("DISPLAY")
    if not display:raise RuntimeError("DISPLAY is unset; run this command from the WSLg terminal")
    dock_xy=args.dock_world_xy or (0.74+args.base_offset_xyz_m[0],-1.44+args.base_offset_xyz_m[1])
    argv=["native", "--table-overlay-result",str(table),"--table-overlay-result-sha256",native.sha256(table),"--claw-control-overlay-result",str(claw),"--claw-control-overlay-result-sha256",native.sha256(claw),"--reset-helper",str(helper),"--reset-helper-sha256",native.sha256(helper),"--xvfb-root","/tmp/robotsim-xvfb-pkg/root","--xvfb-libs","/tmp/robotsim-xvfb-pkg/libs","--xkb-root","/tmp/robotsim-xvfb-pkg/xkb","--interactive-x11-display",display,"--hal-arm-candidate",str(args.candidate),"--hal-arm-candidate-sha256",native.sha256(args.candidate),"--disable-mc-arm-publisher","--disable-mc-hand-publisher","--hal-arm-plan-mode","pregrasp-hold","--hal-right-hand-command","0.0","--hal-hand-close-hold-s","2","--dock-world-xy",str(dock_xy[0]),str(dock_xy[1]),"--dock-yaw-rad","-0.07591850724454846","--dock-position-only","--use-official-cmd-vel-bridge","--dock-timeout-s","30"]
    argv.extend(["--arm-plan-duration-s",str(args.arm_plan_duration_s),"--arm-plan-hold-s",str(args.arm_plan_hold_s)])
    if args.initial_hands_fully_open:argv.append("--initial-hands-fully-open")
    if args.pause_for_human_review:argv.append("--pause-on-claw-failure")
    if args.coordinated_waist_pitch_offset_rad and not args.native_physics_audit:raise ValueError("coordinated circle execution requires native TCP observation")
    if args.coordinated_waist_pitch_offset_rad:argv.extend(["--coordinated-circle-waist","--disable-mc-waist-publisher"])
    sys.argv=argv
    native.main()


if __name__=="__main__":main()
