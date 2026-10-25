#!/usr/bin/env python3
"""Build a source-derived head/torso collision overlay and run zero-step checks.

Research-only. It changes derived collision proxies only; vendor meshes, mass,
inertia, links, joints, actuators, collision flags, and contact parameters stay fixed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,struct,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

VENDOR_REV="575cc6b988f976c23550e0db85aa1e5475d3652d"
SCENE_SHA="a137a6efc793a16df808e23d764594eba4f3449664fbc5ceca554a210742b10a"
VENDOR_ROOT=Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/reference/AgibotTech/agibot_x2_urdf")
BASE_SCENE=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/official-wrist-variant-static-audit-20261024T162630Z/verified/full_x2_d_right_ext_collision_diagnostic.xml")
FIXTURE=Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/tests/fixtures/pr90_static_samples.csv")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def git(path:Path,*args:str)->str:return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
def load_stl(path:Path):
 b=path.read_bytes();n=struct.unpack_from("<I",b,80)[0]
 dt=np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attribute","<u2")])
 tri=np.frombuffer(b,dtype=dt,offset=84,count=n)["vertices"]
 return n,np.unique(tri.reshape(-1,3),axis=0)
def generate_parts(source:Path,dest:Path,threshold:float,max_hulls:int,coacd_site:Path):
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True)
 sys.path.insert(0,str(coacd_site))
 import coacd
 from importlib.metadata import version
 if version("coacd")!="1.0.14":raise RuntimeError("CoACD 1.0.14 is required")
 count,vertices=load_stl(source)
 payload=source.read_bytes();dt=np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attribute","<u2")])
 triangles=np.frombuffer(payload,dtype=dt,offset=84,count=count)["vertices"]
 _,inverse=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True);faces=inverse.reshape(-1,3).astype(np.int32)
 result=coacd.run_coacd(coacd.Mesh(vertices.astype(np.float64),faces),threshold=threshold,max_convex_hull=max_hulls,preprocess_mode="auto",preprocess_resolution=50,resolution=2000,mcts_nodes=10,mcts_iterations=60,mcts_max_depth=2,seed=42,real_metric=True)
 records=[]
 for i,(v0,f0) in enumerate(result):
  v=np.asarray(v0,dtype=float);f=np.asarray(f0,dtype=int);target=dest/f"{source.stem}_coacd_{i:02d}.obj"
  with target.open("x",encoding="ascii",newline="\n") as stream:
   stream.write("# CoACD 1.0.14 convex part, source link-local meters\n")
   for x,y,z in v:stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
   for a,c,d in f:stream.write(f"f {a+1} {c+1} {d+1}\n")
  records.append({"file":target.name,"vertices":len(v),"triangles":len(f),"sha256":sha(target)})
 manifest={"tool":"CoACD","version":"1.0.14","source":str(source),"source_sha256":sha(source),"source_triangles":count,"source_unique_vertices":len(vertices),"threshold_m":threshold,"real_metric":True,"max_convex_hulls":max_hulls,"preprocess_mode":"auto","preprocess_resolution":50,"resolution":2000,"mcts_nodes":10,"mcts_iterations":60,"mcts_max_depth":2,"seed":42,"parts":records}
 (dest/"parts_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 return sorted(dest.glob("*.obj")),manifest
def fixture_row():
 with FIXTURE.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
 return next(row for row in rows if float(row["command_percent"])==2.0)
def contacts(model,data):
 result=[];bottle=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"canonical_bottle")
 for i in range(data.ncon):
  c=data.contact[i];gids=[int(c.geom1),int(c.geom2)];bids=[int(model.geom_bodyid[g]) for g in gids];bn=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in bids];gn=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in gids]
  wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench);R=np.asarray(c.frame).reshape(3,3);force2=R.T@wrench[:3]
  fb=force2 if bids[1]==bottle else (-force2 if bids[0]==bottle else None)
  neck="head_pitch_link" in bn and "torso_link" in bn;table="canonical_bottle" in bn and "m0_table" in bn;floor=("floor" in gn and any("ankle" in str(x) or "foot" in str(x) for x in bn))
  result.append({"geom_names":gn,"body_names":bn,"distance_m":float(c.dist),"wrench_contact_frame_N_Nm":wrench.tolist(),"force_on_bottle_world_N":None if fb is None else fb.tolist(),"allowed":bool(table or floor),"category":"head_torso" if neck else ("bottle_table_support" if table else ("foot_floor_support" if floor else "unexpected"))})
 return result
def set_q2(model,data,row):
 root=float(row["right_claw_joint_rad"]);n2=float(row["narrow2_rad"]);n3=float(row["narrow3_rad"]);loop=float(row["narrow_loop_rad"])
 for side,prefix in (("right","R"),("left","L")):
  values={side+"_claw_joint":root,prefix+"_hand_wide1_joint":-root,prefix+"_hand_narrow2_joint":n2,prefix+"_hand_wide2_joint":-n2,prefix+"_hand_narrow3_joint":n3,prefix+"_hand_wide3_joint":-n3,prefix+"_hand_narrow_loop_joint":loop,prefix+"_hand_wide_loop_joint":-loop}
  for name,q in values.items():
   jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
   if jid<0:raise RuntimeError("Missing fixture joint "+name)
   data.qpos[int(model.jnt_qposadr[jid])]=q
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--output-dir",type=Path,required=True)
 ap.add_argument("--coacd-site",type=Path,required=True)
 ap.add_argument("--reuse-head-parts",type=Path)
 ap.add_argument("--reuse-torso-parts",type=Path)
 ap.add_argument("--head-manifest",type=Path)
 ap.add_argument("--torso-manifest",type=Path)
 ap.add_argument("--head-threshold",type=float,default=.002)
 ap.add_argument("--head-max-hulls",type=int,default=24)
 ap.add_argument("--torso-threshold",type=float,default=.002)
 ap.add_argument("--torso-max-hulls",type=int,default=24)
 args=ap.parse_args()
 out=args.output_dir.resolve();review_root=Path("/home/lzy18001500226/projects/RobotSim/reviews").resolve()
 if not out.is_relative_to(review_root):raise RuntimeError("Output must be under governed reviews")
 if out.exists():raise FileExistsError(out)
 if git(VENDOR_ROOT,"rev-parse","HEAD")!=VENDOR_REV or git(VENDOR_ROOT,"status","--porcelain"):raise RuntimeError("Pinned vendor checkout mismatch or dirty")
 if sha(BASE_SCENE)!=SCENE_SHA:raise RuntimeError("Unexpected base scene hash")
 urdf=VENDOR_ROOT/"X2_URDF-v1.4.0/X2-Ultra.urdf";root=ET.parse(urdf).getroot();source_links={}
 for link_name in ("head_pitch_link","torso_link"):
  link=next(x for x in root.findall("link") if x.get("name")==link_name);mesh=link.find("./collision/geometry/mesh");origin=link.find("./collision/origin")
  if mesh is None or Path(mesh.get("filename","")).name!=link_name+".stl":raise RuntimeError("Unexpected source collision mesh for "+link_name)
  if origin is None or origin.get("xyz","0 0 0")!="0 0 0" or origin.get("rpy","0 0 0")!="0 0 0":raise RuntimeError("Unexpected source collision transform for "+link_name)
  source_links[link_name]=(urdf.parent/mesh.get("filename")).resolve()
 out.mkdir(parents=True,exist_ok=False)
 if args.reuse_head_parts and args.reuse_torso_parts:
  if args.head_manifest is None or args.torso_manifest is None:raise RuntimeError("Reusing CoACD parts requires both source manifests")
  head_manifest=json.loads(args.head_manifest.read_text());torso_manifest=json.loads(args.torso_manifest.read_text())
  if head_manifest.get("version",head_manifest.get("coacd_version"))!="1.0.14" or torso_manifest.get("version",torso_manifest.get("coacd_version"))!="1.0.14":raise RuntimeError("Unexpected CoACD version in reused manifests")
  if head_manifest.get("source_stl_sha256",head_manifest.get("source_sha256"))!=sha(source_links["head_pitch_link"]):raise RuntimeError("Head CoACD manifest source hash mismatch")
  if torso_manifest.get("source_stl_sha256",torso_manifest.get("source_sha256"))!=sha(source_links["torso_link"]):raise RuntimeError("Torso CoACD manifest source hash mismatch")
  head_parts=sorted(args.reuse_head_parts.glob("*.obj"));torso_parts=sorted(args.reuse_torso_parts.glob("*.obj"))
  if not head_parts or not torso_parts:raise RuntimeError("Reused parts directories must contain OBJ convex pieces")
  for label,paths,manifest in (("head",head_parts,head_manifest),("torso",torso_parts,torso_manifest)):
   expected={item["file"]:item["sha256"] for item in manifest.get("parts",[])}
   if set(expected)!={path.name for path in paths}:raise RuntimeError(label+" CoACD part file set mismatch")
   if any(sha(path)!=expected[path.name] for path in paths):raise RuntimeError(label+" CoACD part hash mismatch")
 else:
  head_parts,head_manifest=generate_parts(source_links["head_pitch_link"],out/"head_pitch_parts",args.head_threshold,args.head_max_hulls,args.coacd_site)
  torso_parts,torso_manifest=generate_parts(source_links["torso_link"],out/"torso_parts",args.torso_threshold,args.torso_max_hulls,args.coacd_site)
 tree=ET.parse(BASE_SCENE);xml=tree.getroot();asset=xml.find("asset");parts_manifest={}
 def replace_collision(body_name,old_pred,paths,prefix):
  body=next(x for x in xml.findall(".//body") if x.get("name")==body_name);old=[x for x in body.findall("geom") if old_pred(x)]
  if len(old)!=1:raise RuntimeError("Expected one derived collision geom for "+body_name)
  index=list(body).index(old[0]);body.remove(old[0]);records=[]
  for i,path in enumerate(paths):
   name=f"{prefix}_{i:02d}";ET.SubElement(asset,"mesh",{"name":name,"file":str(path.resolve())});body.insert(index+i,ET.Element("geom",{"name":name,"class":"collision","type":"mesh","mesh":name,"density":"0"}));records.append({"name":name,"file":str(path.resolve()),"sha256":sha(path)})
  return records
 head_records=replace_collision("head_pitch_link",lambda g:g.get("class")=="collision" and g.get("type")=="cylinder",head_parts,"head_pitch_collision")
 torso_records=replace_collision("torso_link",lambda g:g.get("class")=="collision" and g.get("type")=="mesh" and g.get("mesh")=="torso_link",torso_parts,"torso_link_collision")
 scene=out/"full_x2_source_derived_head_torso_collision.xml";tree.write(scene,encoding="unicode",xml_declaration=True)
 baseline=mujoco.MjModel.from_xml_path(str(BASE_SCENE));candidate=mujoco.MjModel.from_xml_path(str(scene));bd=mujoco.MjData(baseline);cd=mujoco.MjData(candidate);mujoco.mj_resetData(baseline,bd);mujoco.mj_resetData(candidate,cd);set_q2(baseline,bd,fixture_row());set_q2(candidate,cd,fixture_row());mujoco.mj_forward(baseline,bd);mujoco.mj_forward(candidate,cd)
 arrays=("jnt_range","jnt_axis","jnt_pos","jnt_qposadr","jnt_dofadr","actuator_ctrlrange","actuator_trnid","body_mass","body_inertia","eq_data")
 for name in arrays:
  if not np.array_equal(getattr(baseline,name),getattr(candidate,name)):raise RuntimeError("Non-collision model array changed: "+name)
 base_contacts=contacts(baseline,bd);candidate_contacts=contacts(candidate,cd);bad=[c for c in candidate_contacts if not c["allowed"]];headtorso=[c for c in candidate_contacts if c["category"]=="head_torso"]
 result={"schema":"robotsim.issue46.head_torso_collision_pair_static_preflight.v1","mujoco_version":mujoco.__version__,"mj_step_calls":0,"base_scene":str(BASE_SCENE),"base_scene_sha256":sha(BASE_SCENE),"vendor_revision":VENDOR_REV,"source_urdf":str(urdf),"source_urdf_sha256":sha(urdf),"source_meshes":{name:{"path":str(path),"sha256":sha(path)} for name,path in source_links.items()},"head_coacd_manifest":head_manifest,"torso_coacd_manifest":torso_manifest,"head_reuse_manifest":str(args.head_manifest) if args.head_manifest else None,"torso_reuse_manifest":str(args.torso_manifest) if args.torso_manifest else None,"head_collision_parts":head_records,"torso_collision_parts":torso_records,"candidate_scene":str(scene),"candidate_scene_sha256":sha(scene),"preserved_non_collision_model_arrays":list(arrays),"preserved_counts":{"nq":baseline.nq,"nv":baseline.nv,"nu":baseline.nu,"neq":baseline.neq,"nbody":baseline.nbody,"njnt":baseline.njnt},"candidate_ngeom":candidate.ngeom,"baseline_contacts":base_contacts,"candidate_contacts":candidate_contacts,"head_torso_contacts":headtorso,"unexpected_contacts":bad,"head_collision_gate":"PASS" if not headtorso and not bad else "FAIL"}
 (out/"zero_step_preflight.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(json.dumps({"gate":result["head_collision_gate"],"head_torso_contact_count":len(headtorso),"unexpected_contact_count":len(bad),"candidate_scene":str(scene)},indent=2))
if __name__=="__main__":main()
