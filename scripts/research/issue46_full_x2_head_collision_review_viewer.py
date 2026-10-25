#!/usr/bin/env python3
"""Open a zero-step interactive MuJoCo view of a full-X2 collision candidate."""
from __future__ import annotations
import argparse,csv,time
from pathlib import Path
import mujoco
import mujoco.viewer

DEFAULT_FIXTURE=Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/tests/fixtures/pr90_static_samples.csv")
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--scene",type=Path,required=True)
 parser.add_argument("--fixture",type=Path,default=DEFAULT_FIXTURE)
 parser.add_argument("--command-percent",type=float,default=2.0)
 args=parser.parse_args()
 model=mujoco.MjModel.from_xml_path(str(args.scene));data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
 with args.fixture.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
 row=next(x for x in rows if float(x["command_percent"])==args.command_percent)
 root=float(row["right_claw_joint_rad"]);n2=float(row["narrow2_rad"]);n3=float(row["narrow3_rad"]);loop=float(row["narrow_loop_rad"])
 for side,prefix in (("right","R"),("left","L")):
  values={side+"_claw_joint":root,prefix+"_hand_wide1_joint":-root,prefix+"_hand_narrow2_joint":n2,prefix+"_hand_wide2_joint":-n2,prefix+"_hand_narrow3_joint":n3,prefix+"_hand_wide3_joint":-n3,prefix+"_hand_narrow_loop_joint":loop,prefix+"_hand_wide_loop_joint":-loop}
  for name,q in values.items():
   jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
   if jid<0:raise SystemExit("Missing joint: "+name)
   data.qpos[int(model.jnt_qposadr[jid])]=q
 mujoco.mj_forward(model,data)
 camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[.30,.12,.90];camera.distance=2.15;camera.azimuth=135;camera.elevation=-18
 with mujoco.viewer.launch_passive(model,data,show_left_ui=True,show_right_ui=True) as viewer:
  viewer.cam.type=camera.type;viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance;viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
  while viewer.is_running():
   viewer.sync()
   time.sleep(1.0/60.0)
if __name__=="__main__":main()
