#!/usr/bin/env python3
"""Compare raw-MuJoCo STL hull contacts with pinned source-derived CoACD proxies."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

REPO=Path(__file__).resolve().parents[2]
EVIDENCE_REPO=Path("/home/lzy18001500226/projects/RobotSim")
AUDIT_PACKAGE=Path("/home/lzy18001500226/robotsim-issue46-stl-mainline-audit/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1")
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory,apply_stl_kinematic_pose

PADS={"R_hand_narrow3_Link","R_hand_wide3_Link"}

def digest(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def load_pre():
 p=REPO/"scripts/research/issue46_stl_cylinder_static_preflight.py";s=importlib.util.spec_from_file_location("pre",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def contacts(model,data):
 cyl=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder");out=[]
 for c in data.contact:
  g=(int(c.geom1),int(c.geom2));b=(int(model.geom_bodyid[g[0]]),int(model.geom_bodyid[g[1]]))
  names=tuple(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,x) or "" for x in b)
  if cyl in b:
   other=names[1] if b[0]==cyl else names[0]
   out.append({"body_names":names,"other":other,"distance_m":float(c.dist),"position_world_m":c.pos.tolist(),"geom_names":tuple(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,x) or "" for x in g)})
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--vertical-static-result",required=True,type=Path);ap.add_argument("--prior-config",required=True,type=Path);args=ap.parse_args();pre=load_pre();st=json.load(open(args.vertical_static_result));prior=json.load(open(args.prior_config));loaded=load_x2_omnipicker("X2-OmniPicker-STL",repo_root=REPO);tr=load_registered_trajectory(loaded.package_path)
 scene=Path(st["generated_scene"]["path"]);tree=ET.parse(scene);root=tree.getroot();manifest=json.load(open(AUDIT_PACKAGE/"collision_proxy_manifest.json"));proxy_dir=AUDIT_PACKAGE/"collision_generated";asset=root.find("asset")
 partmap={Path(x["source_mesh"]).stem:[part["filename"] for part in x["parts"]] for x in manifest["mesh_decompositions"]}
 for parts in partmap.values():
  for f in parts:
   ET.SubElement(asset,"mesh",name="proxy_"+f,file=str(proxy_dir/f))
 for parent in root.iter():
  for geom in list(parent):
   name=geom.get("name","")
   if not (name.startswith("R_hand_") and "__collision_0" in name): continue
   stem=Path(geom.get("mesh","")).stem
   if stem not in partmap: continue
   idx=list(parent).index(geom); parent.remove(geom)
   for part in partmap[stem]:
    new=copy.deepcopy(geom);new.set("name",name+"__"+part);new.set("mesh","proxy_"+part);parent.insert(idx,new);idx+=1
 loc=pg.create_unique_output_directory(EVIDENCE_REPO/"reviews/issue-46",prefix="stl-raw-vs-proxy-contact");out=loc.native_path;proxy_scene=out/"full_x2_stl_proxy_contact.xml";ET.indent(tree,space="  ");pg.write_new_bytes(proxy_scene,ET.tostring(root,encoding="utf-8",xml_declaration=True))
 state=pre.state_by_phase(prior,"HORIZONTAL_APPROACH");arm=np.asarray([st["aligned_contact_static"]["arm_q_rad"][n] for n in pre.ARM_JOINTS]);cmd=int(st["aligned_contact_static"]["highest_safe_bilateral_percent"])
 comparison={}
 for label,path in (("raw",scene),("proxy",proxy_scene)):
  m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);pre.apply_prior_state(m,d,state,arm);apply_stl_kinematic_pose(mujoco,m,d,tr,cmd);c=contacts(m,d);comparison[label]={"all_cylinder_contacts":c,"pad_contacts":[x for x in c if x["other"] in PADS],"nonpad_contacts":[x for x in c if x["other"].startswith("R_hand_") and x["other"] not in PADS]}
 result={"schema":"robotsim.issue46.stl-raw-vs-proxy-contact.v1","classification":"READ_ONLY_COLLISION_COMPARISON","mj_step_calls":0,"model_sha256":loaded.model_sha256,"raw_scene":str(scene),"raw_scene_sha256":digest(scene),"proxy_scene":str(proxy_scene),"proxy_scene_sha256":digest(proxy_scene),"proxy_manifest_sha256":digest(AUDIT_PACKAGE/"collision_proxy_manifest.json"),"comparison":comparison,"proposal_boundary":"No proxy collision geometry is selected or used for dynamics. This output is evidence for a separate approval decision."}
 pg.write_new_text(out/"result.json",json.dumps(result,ensure_ascii=False,indent=2)+"\n");pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes());pg.write_artifact_manifest(out);print(json.dumps({"output_wsl":loc.wsl_path,"raw":comparison["raw"],"proxy":comparison["proxy"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
