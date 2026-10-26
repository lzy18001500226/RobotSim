#!/usr/bin/env python3
"""Bounded vertical hand-offset search for the source-preserving horizontal cylinder."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import sys
import mujoco
import numpy as np

REPO=Path(__file__).resolve().parents[2]
EVIDENCE_REPO=Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory

MODEL_ID="X2-OmniPicker-STL"

def load_pre():
 p=REPO/"scripts/research/issue46_stl_cylinder_static_preflight.py"; spec=importlib.util.spec_from_file_location("pre",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--horizontal-result",required=True,type=Path);ap.add_argument("--prior-config",required=True,type=Path);args=ap.parse_args()
 pre=load_pre();h=json.loads(args.horizontal_result.read_text());prior=json.loads(args.prior_config.read_text());loaded=load_x2_omnipicker(MODEL_ID,repo_root=REPO);tr=load_registered_trajectory(loaded.package_path);model=mujoco.MjModel.from_xml_path(h["generated_scene"]["path"]);state=pre.state_by_phase(prior,"HORIZONTAL_APPROACH")
 # Preserve the previously source-validated vertical-cylinder arm seed.
 vertical=json.load(open("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/full-x2-stl-cylinder-static-20261026T180513Z/result.json"))
 seed=np.asarray([vertical["aligned_contact_static"]["arm_q_rad"][n] for n in pre.ARM_JOINTS]);command=44
 rows=[]
 for dz in (-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05):
  pose=pre.solve_gripper_translation_ik(model,tr,state,command,np.array([0.0,0.0,dz]),cylinder_lift_m=0.0,start_arm_override=seed)
  arm=np.asarray([pose["arm_q_rad"][n] for n in pre.ARM_JOINTS]); scan=[pre.evaluate_pose(model,tr,state,p,arm_override=arm) for p in range(100,-1,-1)]; eligible=[x for x in scan if x["bilateral_pad_contact"] and not x["nonpad_right_contact_count"]]; selected=max(eligible,key=lambda x:x["command_percent"]) if eligible else None
  rows.append({"base_z_offset_m":dz,"pose":pose,"selected":selected})
 viable=[x for x in rows if x["selected"]];best=max(viable,key=lambda x:x["selected"]["command_percent"]) if viable else None
 loc=pg.create_unique_output_directory(EVIDENCE_REPO/"reviews/issue-46",prefix="full-x2-stl-horizontal-z-search");out=loc.native_path
 result={"schema":"robotsim.issue46.horizontal-cylinder-z-search.v1","classification":"KINEMATIC_SIMULATION_ONLY","mj_step_calls":0,"horizontal_source":str(args.horizontal_result),"results":rows,"best":best,"gates":{"SOURCE_STL_DUAL_PAD_WORKCELL_PLACEMENT":"PASS" if best else "BLOCKED","DYNAMICS":"NOT_RUN"}}
 pg.write_new_text(out/"result.json",json.dumps(result,ensure_ascii=False,indent=2)+"\n");pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes());pg.write_artifact_manifest(out);print(json.dumps({"output_wsl":loc.wsl_path,"best":best,"gates":result["gates"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
