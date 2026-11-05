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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--circle-review",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--execute-simulation",action="store_true")
    p.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    args=p.parse_args()
    if not args.execute_simulation:raise ValueError("explicit simulation execution flag required")
    baseline=json.loads((args.baseline/"result.json").read_text())
    original=native.run
    if original(["docker","ps","-q"],timeout_s=5).stdout.strip():raise RuntimeError("conflicting container")
    def run(command,**kwargs):
        if any(str(v).endswith("/runtime_actual_grasp_replan.py") for v in command):
            state=command[command.index("--dock-state")+1]
            # Navigation already applied the requested docking offset. The new
            # state is MEASURED; adding that offset again invents a root pose.
            command=["/home/lzy18001500226/.local/bin/uv","run","--offline","--with","mujoco==3.3.6","--with","numpy==2.2.6","--with","scipy==1.15.3","python3",str(Path(__file__).with_name("stl_approved_circle_ik.py")),"--circle-review",str(args.circle_review),"--candidate",str(args.candidate),"--standing-state",state,"--grasp-height-above-center-m","0.075","--runtime-handoff"]
            kwargs["timeout_s"]=45.
        return original(command,**kwargs)
    native.run=run
    table=Path(baseline["table_overlay"]["path"])
    claw=Path(baseline["claw_control_overlay"]["path"])
    helper=args.baseline/"native_mujoco_reset.py"
    display=__import__("os").environ.get("DISPLAY")
    if not display:raise RuntimeError("DISPLAY is unset; run this command from the WSLg terminal")
    argv=["native", "--table-overlay-result",str(table),"--table-overlay-result-sha256",native.sha256(table),"--claw-control-overlay-result",str(claw),"--claw-control-overlay-result-sha256",native.sha256(claw),"--reset-helper",str(helper),"--reset-helper-sha256",native.sha256(helper),"--xvfb-root","/tmp/robotsim-xvfb-pkg/root","--xvfb-libs","/tmp/robotsim-xvfb-pkg/libs","--xkb-root","/tmp/robotsim-xvfb-pkg/xkb","--interactive-x11-display",display,"--pause-on-claw-failure","--hal-arm-candidate",str(args.candidate),"--hal-arm-candidate-sha256",native.sha256(args.candidate),"--disable-mc-arm-publisher","--disable-mc-hand-publisher","--hal-arm-plan-mode","pregrasp-hold","--hal-right-hand-command","0.0","--hal-hand-close-hold-s","2","--dock-world-xy",str(0.74+args.base_offset_xyz_m[0]),str(-1.44+args.base_offset_xyz_m[1]),"--dock-yaw-rad","-0.07591850724454846","--dock-position-only","--use-official-cmd-vel-bridge","--dock-timeout-s","30"]
    sys.argv=argv
    native.main()


if __name__=="__main__":main()
