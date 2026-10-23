#!/usr/bin/env python3
"""Seven bounded PR90 lateral-pose refinements using the frozen full-X2 model; static only."""
from __future__ import annotations
import csv,hashlib,json,sys,uuid,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"/"research"))
import issue46_manipulation_integration.audit_pr90_full_x2_static as base
from issue46_manipulation_integration.audit_pr90_bounded_24_candidate_search import (
 Evaluator,source_visual_state,group_gap,REV,PR90_HEAD,PR90_MODEL_SHA,OLD_MODEL_SHA,
 CLEARANCE_MM,CONTACT_TOL_MM,CONTACT_BAND_HALF_MM,ARM,HAND,SOURCE_BODIES,sha,gn,bn,jid,gid)
import issue46_manipulation_integration.audit_pr90_bounded_pose_variants as bounded
pg=base.pg

def main():
 out_parent=pg.validate_output_path(str(REV/"issue-46"),authorized_roots=(str(REV),),require_exists=True,
     expected_kind="directory",environment="wsl")
 coarse=REV/"issue-46/pr90-full-x2-bounded-24-static-20261023T180937Z"
 coarse_result=json.loads((coarse/"result.json").read_text())
 pg.verify_artifacts(json.loads((coarse/"artifact_paths.json").read_text())["artifacts"],
     authorized_roots=(str(REV),),environment="wsl")
 if coarse_result["run_id"]!="52882211-82a8-4e67-a291-1321ca842d0c":raise RuntimeError("coarse-screen identity mismatch")
 if base.git(base.WT,"rev-parse","HEAD")!=PR90_HEAD or base.git(base.WT,"status","--porcelain"):
  raise RuntimeError("PR90 is not exact clean read-only input")
 if sha(base.NX)!=PR90_MODEL_SHA or sha(base.OX)!=OLD_MODEL_SHA:raise RuntimeError("frozen source model hash mismatch")
 ri=json.loads(base.NR.read_text());profile=base.profile();integrity=ri["frozen_candidate_b"]["source_integrity_record"]
 urdf=Path(base.pr90.p.UPSTREAM)/"X2-Ultra_omnipicker.urdf"
 if sha(urdf)!=integrity["URDF_sha256"]:raise RuntimeError("official URDF hash mismatch")
 source_paths={b:Path(next(p for p in integrity["STL_sha256"] if Path(p).name==fn)) for b,fn in SOURCE_BODIES.items()}
 for b,p in source_paths.items():
  expected=next(v for k,v in integrity["STL_sha256"].items() if Path(k).name==SOURCE_BODIES[b])
  if not p.is_file() or sha(p)!=expected:raise RuntimeError("vendor STL missing or changed "+str(p))
 xml=base.make_overlay_xml(base.OX.read_text(),ri);root=ET.fromstring(xml);oldroot=ET.fromstring(base.OX.read_text())
 if base.excl(root)!=base.excl(oldroot):raise RuntimeError("collision exclusions changed")
 m=mujoco.MjModel.from_xml_string(xml);om=mujoco.MjModel.from_xml_path(str(base.OX))
 for name in ("geom_contype","geom_conaffinity","geom_bodyid","geom_dataid","geom_pos","geom_quat",
              "body_pos","body_quat","jnt_range","jnt_axis","jnt_pos","actuator_trnid"):
  if not np.array_equal(getattr(om,name),getattr(m,name)):raise RuntimeError("geometry/config overlay changed "+name)
 d=mujoco.MjData(m);obj=gid(m,base.OBJ);ob=int(m.geom_bodyid[obj]);oq=int(m.jnt_qposadr[jid(m,base.OJ)])
 inertia=[base.MASS*(3*(base.D/2)**2+base.H**2)/12]*2+[.5*base.MASS*(base.D/2)**2]
 if not np.allclose(m.geom_size[obj,:2],[base.D/2,base.H/2],atol=1e-12):raise RuntimeError("bottle size drift")
 if abs(m.body_mass[ob]-base.MASS)>1e-12 or not np.allclose(m.body_inertia[ob],inertia,atol=1e-12):raise RuntimeError("bottle mass/inertia drift")
 friction=m.geom_friction[obj].copy();aq=tuple(int(m.jnt_qposadr[jid(m,n)]) for n in ARM);ad=tuple(int(m.jnt_dofadr[jid(m,n)]) for n in ARM)
 limits={n:tuple(map(float,m.jnt_range[jid(m,n)])) for n in ARM};openq=profile[-1]["q"]
 side=json.loads((REV/"issue-46/candidate-b-full-x2-midbody-side-grasp-20261023T063306Z/plan.json").read_text())
 home=list(map(float,side["approach_q"][0]))
 cfg=json.loads((base.OLD/"result.json").read_text());cfg=next(x for x in cfg["candidate_configurations"] if x["id"]=="D70_H230_M550G_MID")
 with base.OC.open(newline="",encoding="utf-8") as f:rows=[x for x in csv.DictReader(f) if x["config"]==cfg["id"]]
 rows.sort(key=lambda x:int(x["sample"]));seed=list(map(float,json.loads(rows[-1]["q_arm_json"])))
 baseid=int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link"))
 base.state(m,d,seed,openq,oq,aq,obj,ob,friction)
 ref={"position":d.xpos[baseid].astype(float).tolist(),"rotation":d.xmat[baseid].reshape(3,3).astype(float).tolist()}
 e=Evaluator(m,d,root,obj,ob,oq,aq,ad,limits,friction)
 start=e.pose(home,openq,details=True)
 if not start["pass"]:raise RuntimeError("starting arm pose fails static collision clearance")
 offsets=[(0,.002),(0,.004),(0,.006),(0,.008),(+.002,.004),(-.002,.004),(0,.010)]
 candidates=[]
 origf=mujoco.mj_forward;origsteps=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2);steps=0;fw=0
 def forward(*a,**k):
  nonlocal fw;fw+=1;return origf(*a,**k)
 def forbidden(*a,**k):
  nonlocal steps;steps+=1;raise RuntimeError("no physics stepping in static refinement")
 mujoco.mj_forward=forward;mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
 try:
  for i,(dx,dy) in enumerate(offsets,1):
   c={"id":f"SIDE_Y{dy*1000:.0f}_X{dx*1000:+.0f}","route":"SIDE_ENTRY","height_offset_m":0.,
      "yaw_deg":0.,"tilt_deg":0.,"offset_xy_m":[dx,dy],"ik_seed_arm":seed}
   row=e.path_candidate(c,home,openq,ref,limits)
   endq=row.get("q_approach_arm")
   if endq is None:
    endpoint={"pass":False,"minimum_clearance_mm":None,"worst_pair":None}
   else:endpoint=e.pose(endq,openq,details=True)
   source=None
   if endq is not None:
    e.apply(endq,openq);source=source_visual_state(e,source_paths,integrity)
   closure=None
   if endq is not None and endpoint["pass"]:
    seq=list(reversed(profile));first_nonpad=None;first_narrow=None;first_wide=None;first_bilateral=None;timeline=[]
    nonpad=[]
    for p in e.gr_obj:
     other=p[1] if p[0]==obj else p[0]
     if bn(m,int(m.geom_bodyid[other])) not in {"R_hand_narrow3_Link","R_hand_wide3_Link"}:nonpad.append(p)
    for step,entry in enumerate(seq):
     h=entry["q"];ng,nc=group_gap(e,endq,h,{"R_hand_narrow3_Link"});wg,wc=group_gap(e,endq,h,{"R_hand_wide3_Link"})
     og=e.eval_current(endq,h,nonpad,None,True)
     nt=[x for x in nc if x["distance_mm"]<=CONTACT_TOL_MM];wt=[x for x in wc if x["distance_mm"]<=CONTACT_TOL_MM]
     if nt and first_narrow is None:first_narrow={"step":step,"percent":entry["percent"],"contacts":nt}
     if wt and first_wide is None:first_wide={"step":step,"percent":entry["percent"],"contacts":wt}
     if og["minimum_clearance_mm"] is not None and og["minimum_clearance_mm"]<=CONTACT_TOL_MM and first_nonpad is None:
      first_nonpad={"step":step,"percent":entry["percent"],"clearance_mm":og["minimum_clearance_mm"],"pair":og["worst_pair"]}
     inband=lambda cs:bool(cs) and all(abs(x["point_object_local_m"][2]*1000)<=CONTACT_BAND_HALF_MM for x in cs)
     if inband(nt) and inband(wt):
      first_bilateral={"step":step,"percent":entry["percent"],"contacts":nt+wt,
       "pad_clearance_mm":{"narrow":ng["minimum_clearance_mm"],"wide":wg["minimum_clearance_mm"],"nonpad":og["minimum_clearance_mm"]}}
      break
     timeline.append({"step":step,"percent":entry["percent"],"pad_gaps_mm":{"narrow":ng["minimum_clearance_mm"],"wide":wg["minimum_clearance_mm"]},
       "nonpad_clearance_mm":og["minimum_clearance_mm"],"nonpad_pair":og["worst_pair"],"pad_touch":{"narrow":nt,"wide":wt}})
     if first_nonpad:break
    closure={"first_narrow":first_narrow,"first_wide":first_wide,"first_nonpad":first_nonpad,
      "first_bilateral":first_bilateral,"pad_first":bool(first_bilateral and (first_nonpad is None or first_bilateral["step"]<first_nonpad["step"])),
      "timeline":timeline}
   candidates.append({"candidate":c,**row,"endpoint_open":endpoint,"source_meshes_open":source,"closure":closure})
   print("REFINE",c["id"],row.get("ik_status"),row.get("path_status"),row.get("minimum_path_clearance_mm"),
         "open",endpoint.get("pass"),endpoint.get("minimum_clearance_mm"),flush=True)
 finally:
  mujoco.mj_forward=origf;mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=origsteps
 out=pg.create_unique_output_directory(out_parent.wsl_path,prefix="pr90-full-x2-lateral-7-refine-static",
  authorized_roots=(str(REV),),environment="wsl")
 result={"schema":"robotsim.issue46.pr90_full_x2_lateral_refinement.v1","run_id":str(uuid.uuid4()),
  "classification":"SIMULATION_ONLY_STATIC_GEOMETRY_AND_IK","physics_steps":steps,"mj_forward_calls":fw,
  "Mujoco":mujoco.__version__,"coarse_evidence_path":str(coarse),"coarse_run_id":coarse_result["run_id"],
  "coarse_result_sha256":sha(coarse/"result.json"),"PR90_head":base.git(base.WT,"rev-parse","HEAD"),
  "PR90_model_sha256":sha(base.NX),"old_full_X2_sha256":sha(base.OX),"generated_full_X2_xml_sha256":hashlib.sha256((xml+"\n").encode()).hexdigest(),
  "official_URDF_path":str(urdf),"official_URDF_sha256":sha(urdf),
  "official_STLs":[{"path":str(p),"sha256":sha(p)} for p in source_paths.values()],
  "object":{"D_m":base.D,"H_m":base.H,"mass_kg":base.MASS,"center_world_m":list(base.CENTER),"friction":m.geom_friction[obj].tolist()},
  "refinement_basis":{"source_plan":str(REV/"issue-46/candidate-b-full-x2-midbody-side-grasp-20261023T063306Z/plan.json"),
    "x2_pose_seed":seed,"offsets_xy_m":[[x,y] for x,y in offsets],"side_pregrasp_standoff_m":.040,
    "reason":"source contact points show loop collisions around x=+-20..34mm, y=+8..25mm in object frame; small +Y offsets test outward radial clearance while retaining pad reach"},
  "candidates":candidates,"stage_status":{"PREGRASP_IK":"STATIC_ONLY","APPROACH":"STATIC_ONLY","GRASP":"NOT RUN","LOAD_TRANSFER":"NOT RUN","LIFT_50MM":"NOT RUN"}}
 result["output"]=out.as_dict();result["runner_path"]=str(Path(__file__).resolve());result["runner_sha256"]=sha(Path(__file__).resolve())
 pg.write_new_text(out.native_path/"result.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
  authorized_roots=(str(REV),),environment="wsl")
 lines=["# PR90 full-X2 lateral pose refinement","",f"Run {result['run_id']}",f"Output {out.wsl_path}",f"UNC {out.windows_unc}",
  "",f"Static only: physics_steps={steps}, mj_forward_calls={fw}. No object pose or model geometry was changed.",
  "Seven bounded refinements use the exact saved q_goal as IK seed and side-entry pregrasp derived from the saved plan (39.85 mm, rounded to 40 mm). Y shifts 2-10 mm test whether measured loop intrusion can be cleared while pad contact remains reachable; x adjustments are at most 2 mm.",
  "Model: exact PR90 overlay on full-X2 source model; original source URDF/STLs/mount, collision masks/exclusions, bottle dimensions/mass/friction are unchanged.",
  "","## Candidate results"]
 for x in candidates:
  line=f"- {x['candidate']['id']}: IK={x.get('ik_status')}; path={x.get('path_status')}; min={x.get('minimum_path_clearance_mm')}; endpoint clear={x['endpoint_open'].get('pass')}; endpoint min={x['endpoint_open'].get('minimum_clearance_mm')}."
  if x.get("closure"):line+=f" closure pad-first={x['closure']['pad_first']}; first non-pad={x['closure']['first_nonpad']}."
  lines.append(line)
 lines.extend(["","No dynamic trial was run. A static pass would still not demonstrate grip force or lifting."])
 pg.write_new_text(out.native_path/"REPORT.md","\n".join(lines)+"\n",authorized_roots=(str(REV),),environment="wsl")
 command=f"cd {ROOT}\nenv PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python {Path(__file__).resolve()}\n"
 pg.write_new_text(out.native_path/"REPRODUCE.md",command,authorized_roots=(str(REV),),environment="wsl")
 mf=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl");pg.verify_artifacts(mf["artifacts"],authorized_roots=(str(REV),),environment="wsl")
 print(json.dumps({"run_id":result["run_id"],"output":out.as_dict(),"steps":steps,"forwards":fw,
  "refined":[{"id":c["candidate"]["id"],"ik":c.get("ik_status"),"path":c.get("path_status"),
   "open":c["endpoint_open"].get("pass"),"closure":None if not c.get("closure") else c["closure"]["pad_first"]} for c in candidates],
  "manifest":len(mf["artifacts"])},indent=2))
 return 0

if __name__=="__main__":raise SystemExit(main())
