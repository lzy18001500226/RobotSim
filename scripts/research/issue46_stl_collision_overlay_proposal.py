#!/usr/bin/env python3
"""Prepare the exact review package for a source-STL CoACD collision overlay."""
from __future__ import annotations
import argparse, importlib.util, json, shutil
from pathlib import Path
import sys
import mujoco
import numpy as np

REPO=Path(__file__).resolve().parents[2]
EVIDENCE_REPO=Path("/home/lzy18001500226/projects/RobotSim")
AUDIT_PACKAGE=Path("/home/lzy18001500226/robotsim-issue46-stl-mainline-audit/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1")
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory

def load_pre():
 p=REPO/"scripts/research/issue46_stl_cylinder_static_preflight.py";s=importlib.util.spec_from_file_location("pre",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):
 import hashlib
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--vertical-static-result",required=True,type=Path);ap.add_argument("--prior-config",required=True,type=Path);ap.add_argument("--comparison-result",required=True,type=Path);args=ap.parse_args()
 pre=load_pre();base=json.load(open(args.vertical_static_result));prior=json.load(open(args.prior_config));comparison=json.load(open(args.comparison_result));loaded=load_x2_omnipicker("X2-OmniPicker-STL",repo_root=REPO);tr=load_registered_trajectory(loaded.package_path);proxy_scene=Path(comparison["proxy_scene"]);model=mujoco.MjModel.from_xml_path(str(proxy_scene));state=pre.state_by_phase(prior,"HORIZONTAL_APPROACH");arm=np.asarray([base["aligned_contact_static"]["arm_q_rad"][n] for n in pre.ARM_JOINTS]);lift=np.asarray([base["transport_end_static"]["arm_q_rad"][n] for n in pre.ARM_JOINTS]);
 samples=[]
 for fraction in np.linspace(0,1,5):
  item=pre.evaluate_pose(model,tr,state,42,cylinder_lift_m=.05*float(fraction),arm_override=arm+float(fraction)*(lift-arm));item["fraction"]=float(fraction);samples.append(item)
 path_pass=all(x["bilateral_pad_contact"] and not x["nonpad_right_contact_count"] and (x["table_contact_count"]==1 if x["fraction"]==0 else x["table_contact_count"]==0) for x in samples)
 manifest=json.load(open(AUDIT_PACKAGE/"collision_proxy_manifest.json"));mapping={Path(x["source_mesh"]).stem:len(x["parts"]) for x in manifest["mesh_decompositions"]}
 loc=pg.create_unique_output_directory(EVIDENCE_REPO/"reviews/issue-46",prefix="stl-collision-overlay-proposal");out=loc.native_path
 pg.write_new_bytes(out/"full_x2_stl_proxy_contact.xml",proxy_scene.read_bytes())
 payload={"schema":"robotsim.issue46.stl-collision-overlay-proposal.v1","classification":"PROPOSAL_REQUIRES_MODEL_INTEGRITY_APPROVAL","mj_step_calls":0,"current_stl_model_sha256":loaded.model_sha256,"trajectory_sha256":loaded.metadata["trajectory_sha256"],"proxy_manifest_sha256":sha(AUDIT_PACKAGE/"collision_proxy_manifest.json"),"source_proxy_piece_count":sum(mapping.values()),"proxy_piece_count_by_source_mesh":mapping,"current_raw_collision_evidence":comparison["comparison"],"proposed_proxy_static_path_samples":samples,"proposed_proxy_static_path_pass":path_pass,"frozen_invariants":["visual STL mesh bytes","101-row STL trajectory","all joints, axes, limits and fixed intermediate links","body inertias and masses","friction coefficients","seven approved internal collision exclusions","cylinder/table/robot geometry"],"proposed_delta":"Replace only right OmniPicker raw whole-mesh collision geoms with the manifest-locked source-STL CoACD pieces; visual geoms stay byte-for-byte source meshes. No new links, joints, actuators, transmissions, limits or contact exclusions.","not_authorized_or_executed":["selected model collision replacement","new physical dynamics with proxy overlay","grasp/lift claim"]}
 pg.write_new_text(out/"proposal.json",json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
 md=f'''# STL collision overlay proposal\n\nCurrent model: `X2-OmniPicker-STL` SHA-256 `{loaded.model_sha256}`.\n\nThe raw whole-mesh collision representation reports two pad contacts at the reviewed 44% pose. The same pose with the existing source-STL CoACD pieces reports **no gripper contact**. The raw contacts are therefore convex-hull artifacts, not enough evidence for a physical lift.\n\nThe proposal uses `{sum(mapping.values())}` pre-existing, hash-locked pieces from manifest `{payload["proxy_manifest_sha256"]}`. It changes collision representation only. Visual meshes, joint table, links, axes, limits, inertias, friction and exclusions remain unchanged.\n\nAt 42%, the proposed proxy scene has bilateral pad contact, no non-pad contact, and passes all five static 0–50 mm lift samples.\n\n**Approval required before implementation:** activate this collision overlay as a separate `SIMULATION_ONLY` contact candidate and run a bounded real MuJoCo lift test.\n'''
 pg.write_new_text(out/"PROPOSAL.md",md);pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes());pg.write_artifact_manifest(out);print(json.dumps({"output_wsl":loc.wsl_path,"path_pass":path_pass,"model":loaded.model_sha256,"manifest":payload["proxy_manifest_sha256"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
