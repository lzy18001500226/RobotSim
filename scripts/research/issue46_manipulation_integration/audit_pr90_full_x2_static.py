#!/usr/bin/env python3
"""PR90 quartic full-X2 static preflight; no physics stepping."""
import argparse,csv,hashlib,json,math,subprocess,sys,time,uuid,xml.etree.ElementTree as ET,io
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/"scripts/agent"),str(ROOT/"scripts/research")]
sys.path.insert(0, '/home/lzy18001500226/robotsim-issue46-full-open-static-20261023/scripts/research')
import path_governance as pg
import issue46_candidate_b_clearance_coupling as pr90
from issue46_manipulation_integration.approach import ToolPose
from issue46_manipulation_integration.contracts import PoseSample,PoseSource,TimeDomain
from issue46_manipulation_integration.feasibility import ApproachObservation,GeometryPair,CollisionInventory,collision_inventory_digest,GripperContract,JointCommandProfile,ClosureObservation,GripperFeasibilityRequest,ObjectGeometry,ContactBand,FeasibilityThresholds,evaluate_gripper_feasibility
REV=Path("/home/lzy18001500226/projects/RobotSim/reviews");WT=Path("/home/lzy18001500226/robotsim-issue46-full-open-static-20261023")
OLD=REV/"issue-46/candidate-b-centered-topdown-static-screen-20261023T104451Z";NEW=REV/"issue-46/candidate-b-full-useful-open-static-20261023T105521Z"
OX=OLD/"runtime_model_fixed_base_root_kp2_static_screen.xml";OR=OLD/"result.json";OC=OLD/"open_insertion_samples.csv"
NX=NEW/"candidate_b_full_useful_range_static.xml";NR=NEW/"static_range.json";NP=NEW/"normalized_static_samples.csv"
SIDE=REV/"issue-46/candidate-b-full-x2-midbody-side-grasp-20261023T063306Z/plan.json"
ARM=("right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint")
HAND=("right_claw_joint","R_hand_wide1_joint","R_hand_narrow2_joint","R_hand_wide2_joint","R_hand_narrow3_joint","R_hand_wide3_joint","R_hand_narrow_loop_joint","R_hand_wide_loop_joint")
OBJ="test_cylinder_collision";OJ="test_cylinder_free";TABLE="m0_table_top";CENTER=(.300,.040,.915);D=.070;H=.230;MASS=.550;TOL=.0001
DER={"R_hand_wide1_joint":"official_root_mimic","R_hand_narrow2_joint":"PR90 quartic","R_hand_wide2_joint":"PR90 quartic","R_hand_narrow3_joint":"PR90 D state","R_hand_wide3_joint":"PR90 D state","R_hand_narrow_loop_joint":"PR90 D state","R_hand_wide_loop_joint":"PR90 D state"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(p,*a):return subprocess.check_output(["git","-C",str(p),*a],text=True).strip()
def nm(m,t,i):return mujoco.mj_id2name(m,t,int(i)) or f"{t}_{i}"
def id_by_name(m,t,n):
 i=int(mujoco.mj_name2id(m,t,n))
 if i<0:raise ValueError("missing "+n)
 return i
def jid(m,n):return int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n))
def gid(m,n):return int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n))
def body(m,i):return nm(m,mujoco.mjtObj.mjOBJ_BODY,i)
def geom(m,i):return nm(m,mujoco.mjtObj.mjOBJ_GEOM,i)
def excl(root):return {tuple(sorted((x.get("body1",""),x.get("body2","")))) for x in root.findall("./contact/exclude")}
def dist(m,d,a,b):return float(mujoco.mj_geomDistance(m,d,int(a),int(b),.5,None))
def coll(m,a,b,e):
 ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
 return a!=b and m.body_weldid[ba]!=m.body_weldid[bb] and tuple(sorted((body(m,ba),body(m,bb)))) not in e and m.body_parentid[ba]!=bb and m.body_parentid[bb]!=ba and bool((int(m.geom_contype[a])&int(m.geom_conaffinity[b])) or (int(m.geom_contype[b])&int(m.geom_conaffinity[a])))
def profile():
 rows=list(csv.DictReader(NP.open(newline="",encoding="utf-8")))
 if len(rows)!=101:raise ValueError("PR90 exact 101-state profile missing")
 out=[]
 for x in rows:
  u=float(x["command_percent"])/100;root=-.6841259890668666*u
  q={"right_claw_joint":float(x["right_claw_joint_rad"]),"R_hand_wide1_joint":-float(x["right_claw_joint_rad"]),
  "R_hand_narrow2_joint":float(x["narrow2_rad"]),"R_hand_wide2_joint":-float(x["narrow2_rad"]),
  "R_hand_narrow3_joint":float(x["narrow3_rad"]),"R_hand_wide3_joint":-float(x["narrow3_rad"]),
  "R_hand_narrow_loop_joint":float(x["narrow_loop_rad"]),"R_hand_wide_loop_joint":-float(x["narrow_loop_rad"])}
  if abs(q["right_claw_joint"]-root)>1e-9 or abs(float(x["qCAD"])-(-0.13412598906686654+0.6841259890668666*u))>1e-9:raise ValueError("PR90 normalized command mapping mismatch")
  out.append({"percent":float(x["command_percent"]),"u":u,"fraction":u,"qCAD":float(x["qCAD"]),"q":q})
 info=json.loads(NR.read_text())
 for point in info["static_full_useful_range"]["five_command_points"]:
  q=out[int(point["command_percent"])]["q"]
  for joint,value in point["joint_positions_urdf_rad"].items():
   if abs(q[joint]-float(value))>1e-8:raise ValueError("profile differs from PR90 saved command point")
 return out
def state(m,d,qarm,qhand,oq,aq,obj,ob,fr):
 mujoco.mj_resetData(m,d)
 for n,v in (("left_knee_joint",.01),("right_knee_joint",.01),("head_pitch_joint",0)):d.qpos[int(m.jnt_qposadr[jid(m,n)])]=v
 for a,v in zip(aq,qarm):d.qpos[a]=v
 for n,v in qhand.items():d.qpos[int(m.jnt_qposadr[jid(m,n)])]=v
 d.qpos[oq:oq+7]=[*CENTER,1,0,0,0]
 if not np.array_equal(m.geom_friction[obj],fr):raise ValueError("bottle friction changed")
 mujoco.mj_forward(m,d)
def mesh_inventory(m,root):
 md=Path(root.find("compiler").get("meshdir","."));files={x.get("name"):x.get("file") for x in root.findall("./asset/mesh")};out=[]
 for g in range(m.ngeom):
  b=body(m,int(m.geom_bodyid[g]))
  if not b.startswith(("R_hand_","R_omnipicker")):continue
  mi=int(m.geom_dataid[g])
  if mi<0:continue
  a=nm(m,mujoco.mjtObj.mjOBJ_MESH,mi);p=Path(files[a]);p=(md/p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
  out.append({"geom":geom(m,g),"body":b,"mesh":a,"path":str(p),"sha256":sha(p),"collision":bool(m.geom_contype[g] or m.geom_conaffinity[g])})
 return out
def make_overlay_xml(base_xml, info):
 root=ET.fromstring(base_xml);root.set("model","SIMULATION_ONLY_PR90_QUARTIC_FULL_X2_STATIC")
 eq=root.find("equality");old={"geometry_narrow_coupling","geometry_wide_coupling"}
 if {x.get("name") for x in eq if x.get("name") in old}!=old:raise ValueError("old q2 laws missing")
 for x in list(eq):
  if x.get("name") in old:eq.remove(x)
 c=info["coupling"]
 for n,j1,j2,coef in (("candidate_full_open_narrow_q2",c["narrow_joint1"],c["narrow_joint2"],c["narrow_polycoef"]),("candidate_full_open_wide_q2",c["wide_joint1"],c["wide_joint2"],c["wide_polycoef"])):
  ET.SubElement(eq,"joint",name=n,joint1=j1,joint2=j2,polycoef=" ".join(f"{float(v):.16g}" for v in coef),solref="0.002 1")
 return ET.tostring(root,encoding="unicode")

def name(m,t,i):
 return nm(m,t,i)
def id_by_name(m,t,n):
 i=int(mujoco.mj_name2id(m,t,n))
 if i<0:raise ValueError("missing "+n)
 return i
def collision_enabled(m,a,b,e):
 return coll(m,a,b,e)
def parse_exclusions(root):
 return excl(root)
def min_pair_distance(m,d,aids,bids,e,*,maxdist=.5,points=False):
 best=None
 for a in aids:
  for b in bids:
   if not coll(m,a,b,e):continue
   line=np.zeros(6) if points else None
   x=float(mujoco.mj_geomDistance(m,d,int(a),int(b),maxdist,line))
   z={"distance_m":x,"geom1":geom(m,a),"geom2":geom(m,b),"body1":body(m,int(m.geom_bodyid[a])),"body2":body(m,int(m.geom_bodyid[b]))}
   if points:z["point1_world_m"]=line[:3].tolist();z["point2_world_m"]=line[3:].tolist()
   if best is None or x<best["distance_m"]:best=z
 return best
def contact_list_for_object(m,d,obj):
 out=[]
 for c in d.contact[:d.ncon]:
  a,b=int(c.geom1),int(c.geom2)
  if obj not in (a,b):continue
  other=b if a==obj else a
  out.append({"geom":geom(m,other),"body":body(m,int(m.geom_bodyid[other])),"distance_mm":float(c.dist)*1000,"position_world_m":np.asarray(c.pos).tolist()})
 return out
def candidate_mjcf_mesh_inventory(m,root):
 md=Path(root.find("compiler").get("meshdir","."));files={x.get("name"):x.get("file") for x in root.findall("./asset/mesh")};out=[]
 for g in range(m.ngeom):
  bn=body(m,int(m.geom_bodyid[g]))
  if not bn.startswith(("R_hand_","R_omnipicker")):continue
  mid=int(m.geom_dataid[g])
  if mid<0:continue
  mn=nm(m,mujoco.mjtObj.mjOBJ_MESH,mid);p=Path(files[mn]);p=(md/p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
  out.append({"geom":geom(m,g),"body":bn,"mesh_file":str(p),"mesh_sha256":sha(p),"collision_enabled":bool(m.geom_contype[g] or m.geom_conaffinity[g])})
 return out
def body_mount(m):
 a=id_by_name(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_official_mount");p=int(m.body_parentid[a])
 b=id_by_name(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link")
 return {"mount":"R_omnipicker_official_mount","parent":body(m,p),"xyz_m":m.body_pos[a].tolist(),"quat_wxyz":m.body_quat[a].tolist(),"base_parent":body(m,int(m.body_parentid[b]))}

def evaluate():
 of=mujoco.mj_forward;os=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2);fw=0;steps=0
 def forward(*a,**k):
  nonlocal fw;fw+=1;return of(*a,**k)
 def stop(*a,**k):
  nonlocal steps;steps+=1;raise RuntimeError("physics stepping forbidden")
 mujoco.mj_forward=forward;mujoco.mj_step=stop;mujoco.mj_step1=stop;mujoco.mj_step2=stop
 try:
  pin="facdaa0d6e5f8152168d2e812d864b660eac9f96"
  if git(WT,"rev-parse","HEAD")!=pin or git(WT,"status","--porcelain"):raise ValueError("PR90 worktree not exact clean input")
  if sha(NX)!="32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45":raise ValueError("PR90 source MJCF hash changed")
  if sha(OX)!="0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01":raise ValueError("old full-X2 model hash changed")
  for f in (OLD,NEW):
   pg.verify_artifacts(json.loads((f/"artifact_paths.json").read_text())["artifacts"],authorized_roots=(str(REV),),environment="wsl")
  src=json.loads(NR.read_text());prof=profile();oldres=json.loads(OR.read_text())
  cfg=next(x for x in oldres["candidate_configurations"] if x["id"]=="D70_H230_M550G_MID")
  oldrows=[x for x in csv.DictReader(OC.open(newline="",encoding="utf-8")) if x["config"]==cfg["id"]];oldrows.sort(key=lambda x:int(x["sample"]))
  root=ET.fromstring(OX.read_text());newroot=ET.fromstring(make_overlay_xml(OX.read_text(),src))
  ex=excl(root)
  if ex!=excl(newroot):raise ValueError("collision exclusions changed")
  om=mujoco.MjModel.from_xml_string(OX.read_text());m=mujoco.MjModel.from_xml_string(ET.tostring(newroot,encoding="unicode"))
  counts={k:getattr(m,k) for k in ("nbody","njnt","ngeom","nmesh","neq","nu","nq","nv")}
  if counts!={k:getattr(om,k) for k in counts}:raise ValueError("compiled topology changed")
  for a in ("geom_contype","geom_conaffinity","geom_bodyid","geom_dataid","geom_pos","geom_quat","body_pos","body_quat","jnt_range","jnt_axis","jnt_pos","actuator_trnid"):
   if not np.array_equal(getattr(om,a),getattr(m,a)):raise ValueError("overlay changed "+a)
  data=mujoco.MjData(m);obj=gid(m,OBJ);ob=int(m.geom_bodyid[obj]);oq=int(m.jnt_qposadr[jid(m,OJ)]);table=gid(m,TABLE)
  r=D/2;m.geom_size[obj,:2]=[r,H/2];m.body_mass[ob]=MASS;m.body_inertia[ob]=[MASS*(3*r*r+H*H)/12]*2+[.5*MASS*r*r];fr=m.geom_friction[obj].copy()
  aq=tuple(int(m.jnt_qposadr[jid(m,j)]) for j in ARM);limits={j:tuple(map(float,m.jnt_range[jid(m,j)])) for j in ARM}
  gr=[g for g in range(m.ngeom) if body(m,int(m.geom_bodyid[g])).startswith(("R_hand_","R_omnipicker")) and (m.geom_contype[g] or m.geom_conaffinity[g])]
  expected=set(oldres["candidate_b_model"]["right_hand_enabled_collision_geoms"])
  if {geom(m,g) for g in gr}!=expected:raise ValueError("gripper collision geom inventory differs from frozen old screen")
  groups={"narrow_pad":[g for g in gr if body(m,int(m.geom_bodyid[g]))=="R_hand_narrow3_Link"],"wide_pad":[g for g in gr if body(m,int(m.geom_bodyid[g]))=="R_hand_wide3_Link"],"linkage_loop":[g for g in gr if "loop_Link" in body(m,int(m.geom_bodyid[g]))]}
  groups["other_gripper"]=[g for g in gr if all(g not in z for z in groups.values())]
  robot=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in (0,ob) and not body(m,int(m.geom_bodyid[g])).startswith("m0_table") and (m.geom_contype[g] or m.geom_conaffinity[g])]
  moving=[g for g in robot if body(m,int(m.geom_bodyid[g])).startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker"))]
  scene=[g for g in range(m.ngeom) if int(m.geom_bodyid[g])==0 or body(m,int(m.geom_bodyid[g])).startswith("m0_table")]
  pairs=set()
  for g in robot:
   if coll(m,g,obj,ex):pairs.add(tuple(sorted((geom(m,g),geom(m,obj)))))
  for g in moving:
   for h in scene+robot:
    if coll(m,g,h,ex):pairs.add(tuple(sorted((geom(m,g),geom(m,h)))))
  pairs=tuple(sorted(pairs));gids={geom(m,g):g for g in range(m.ngeom)}
  ident="full-x2-pr90-static:"+hashlib.sha256(ET.tostring(newroot)).hexdigest()
  names=tuple(sorted({v for p in pairs for v in p}));inv=CollisionInventory(ident,names,pairs,collision_inventory_digest(ident,names,pairs))
  now=time.monotonic_ns();pose=PoseSample("test_cylinder","world",CENTER,(0,0,0,1),0,PoseSource.SIM_GROUND_TRUTH,TimeDomain.SIMULATION,model_identity=ident,received_steady_ns=now)
  def path_run(qrows,label):
   metrics=[];obs=[]
   for i,row in enumerate(qrows):
    q=row["q"];state(m,data,q,prof[-1]["q"],oq,aq,obj,ob,fr)
    pd=[]
    for a,b in pairs:
     x=float(mujoco.mj_geomDistance(m,data,gids[a],gids[b],.5,None));pd.append(GeometryPair(a,b,x,"collision" if x<=0 else "clearance"))
    near=min(pd,key=lambda x:x.signed_distance_m);cs=[]
    for c in data.contact[:data.ncon]:
     a,b=int(c.geom1),int(c.geom2);cs.append({"geom1":geom(m,a),"geom2":geom(m,b),"body1":body(m,int(m.geom_bodyid[a])),"body2":body(m,int(m.geom_bodyid[b])),"dist_mm":float(c.dist)*1000})
    object_c=[x for x in cs if (x["geom1"]==OBJ or x["geom2"]==OBJ) and not x["body1"].startswith("m0_table") and not x["body2"].startswith("m0_table")]
    self_c=[x for x in cs if x["geom1"]!=OBJ and x["geom2"]!=OBJ and (x["body1"].startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker")) or x["body2"].startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker")))]
    group_clear={k:(None if min_pair_distance(m,data,v,[obj],ex) is None else min_pair_distance(m,data,v,[obj],ex)["distance_m"]*1000) for k,v in groups.items()}
    table_ids=[g for g in scene if body(m,int(m.geom_bodyid[g])).startswith("m0_table") or geom(m,g).startswith("m0_table")]
    table_min=min_pair_distance(m,data,gr,table_ids,ex)
    metrics.append({"sample":row["sample"],"route":label,"q_arm_rad":list(q),"minimum_clearance_mm":near.signed_distance_m*1000,"nearest_pair":[near.geom_a,near.geom_b],"gripper_object_clearance_by_group_mm":group_clear,"gripper_table_clearance_mm":None if table_min is None else table_min["distance_m"]*1000,"gripper_table_nearest_pair":None if table_min is None else [table_min["geom1"],table_min["geom2"]],"robot_object_contacts":object_c,"moving_self_contacts":self_c})
    base=id_by_name(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link");qbase=data.xquat[base]
    obs.append(ApproachObservation(i,"PREGRASP" if i==0 else "APPROACH",ident,ToolPose("world",tuple(map(float,data.xpos[base])),(float(qbase[1]),float(qbase[2]),float(qbase[3]),float(qbase[0]))),pose,dict(zip(ARM,q)),tuple(pd),tuple(sorted(tuple(sorted((x["geom1"],x["geom2"]))) for x in cs if tuple(sorted((x["geom1"],x["geom2"]))) in pairs))))
   return metrics,obs
  base_rows=[{"sample":int(x["sample"]),"q":tuple(map(float,json.loads(x["q_arm_json"])))} for x in oldrows];toprows=[]
  for i,x in enumerate(base_rows):
   if i:toprows.append({"sample":f"{i}-mid","q":tuple((a+b)/2 for a,b in zip(base_rows[i-1]["q"],x["q"]))})
   toprows.append(x)
  top,to=path_run(toprows,"TOP_DOWN_VERTICAL")
  clear=lambda x:bool(x) and all(y["minimum_clearance_mm"]>=1 and not y["robot_object_contacts"] and not y["moving_self_contacts"] for y in x)
  topok=clear(top);side=None;sideok=False;sl=None;chosen="TOP_DOWN_VERTICAL";selected=top;so=to;finalq=base_rows[-1]["q"]
  if not topok:
   plan=json.loads(SIDE.read_text());sq=[tuple(map(float,x)) for x in plan["approach_q"]]+[tuple(map(float,plan["q_pregrasp"])),tuple(map(float,plan["q_grasp"]))]
   sl=all(all(limits[j][0]<=v<=limits[j][1] for j,v in zip(ARM,q)) for q in sq)
   if sl:
    uq=[]
    for q in sq:
     if not uq or max(abs(a-b) for a,b in zip(q,uq[-1]))>1e-9:uq.append(q)
    sr=[]
    for i,q in enumerate(uq):
     if i:sr.append({"sample":f"side{i}-mid","q":tuple((a+b)/2 for a,b in zip(uq[i-1],q))})
     sr.append({"sample":f"side{i}","q":q})
    side,ss=path_run(sr,"EXISTING_SIDE_OBLIQUE");sideok=clear(side)
    if sideok:chosen="EXISTING_SIDE_OBLIQUE";selected=side;so=ss;finalq=tuple(plan["q_grasp"])
  closure=[];maxD=0;maxpen=0;limok=True
  for i,e in enumerate(reversed(prof)):
   state(m,data,finalq,e["q"],oq,aq,obj,ob,fr)
   valid=all(not m.jnt_limited[jid(m,j)] or m.jnt_range[jid(m,j),0]<=v<=m.jnt_range[jid(m,j),1] for j,v in e["q"].items());limok &= valid
   gaps=[]
   for s in ("narrow","wide"):
    a=id_by_name(m,mujoco.mjtObj.mjOBJ_SITE,f"candidate_{s}_D_jaw");b=id_by_name(m,mujoco.mjtObj.mjOBJ_SITE,f"candidate_{s}_D_loop");gaps.extend((data.site_xpos[a]-data.site_xpos[b])*1000)
   dg=float(np.max(np.abs(gaps)));maxD=max(maxD,dg);active=[];internal=[]
   for c in data.contact[:data.ncon]:
    a,b=int(c.geom1),int(c.geom2);z={"geom1":geom(m,a),"geom2":geom(m,b),"body1":body(m,int(m.geom_bodyid[a])),"body2":body(m,int(m.geom_bodyid[b])),"dist_mm":float(c.dist)*1000}
    if obj in (a,b):active.append(z)
    elif z["body1"].startswith(("R_hand_","R_omnipicker")) and z["body2"].startswith(("R_hand_","R_omnipicker")):internal.append(z)
   maxpen=max(maxpen,max((max(0,-x["dist_mm"]) for x in internal),default=0))
   mins={k:min_pair_distance(m,data,v,[obj],ex,points=k.endswith("pad")) for k,v in groups.items()}
   wrist_ids=[g for g in robot if "wrist" in body(m,int(m.geom_bodyid[g])) or body(m,int(m.geom_bodyid[g]))=="R_omnipicker_official_mount"]
   mins["wrist_arm"]=min_pair_distance(m,data,wrist_ids,[obj],ex)
   table_ids=[g for g in scene if body(m,int(m.geom_bodyid[g])).startswith("m0_table") or geom(m,g).startswith("m0_table")]
   mins["table"]=min_pair_distance(m,data,gr,table_ids,ex)
   pts={};geoms={}
   for s in ("narrow","wide"):
    x=mins[s+"_pad"]
    if x and x["distance_m"]<=TOL:
     p=(np.asarray(x["point2_world_m"])-data.xpos[ob])@data.xmat[ob].reshape(3,3);pts[s]=(tuple(map(float,p)),);geoms[s]=(x["geom1"],)
    else:pts[s]=();geoms[s]=()
   closure.append({"step_index":i,"percent":e["percent"],"fraction":e["fraction"],"qCAD":e["qCAD"],"q":e["q"],"limits_pass":bool(valid),"D_gap_mm":dg,"wrist_arm_mm":None if mins["wrist_arm"] is None else mins["wrist_arm"]["distance_m"]*1000,"gripper_table_mm":None if mins["table"] is None else mins["table"]["distance_m"]*1000,
    **{k:(None if mins[k] is None else mins[k]["distance_m"]*1000) for k in groups},"pad_points":pts,"pad_geoms":geoms,"object_contacts":active,"internal_contacts":internal})
  first=lambda k:next((x for x in closure if x[k] is not None and x[k]<=.1),None)
  forward_total=fw
  fn,first_wide,fl,fo=[first(k) for k in ("narrow_pad","wide_pad","linkage_loop","other_gripper")]
  bi=next((x for x in closure if x["narrow_pad"] is not None and x["wide_pad"] is not None and x["narrow_pad"]<=.1 and x["wide_pad"]<=.1 and all(x["pad_points"][s] and abs(x["pad_points"][s][0][2])<=.005 for s in ("narrow","wide"))),None)
  clean=[x for x in closure if x["narrow_pad"] is not None and x["wide_pad"] is not None and x["narrow_pad"]<=.1 and x["wide_pad"]<=.1 and (x["linkage_loop"] is None or x["linkage_loop"]>.1)]
  padfirst=bool(bi and (fl is None or bi["step_index"]<fl["step_index"]));rangeok=limok and maxD<=.01 and maxpen<=.01
  n=tuple(geom(m,g) for g in groups["narrow_pad"]);w=tuple(geom(m,g) for g in groups["wide_pad"]);l=tuple(geom(m,g) for g in groups["linkage_loop"]);o=tuple(geom(m,g) for g in groups["other_gripper"])
  limited={j:tuple(map(float,m.jnt_range[jid(m,j)])) for j in HAND if m.jnt_limited[jid(m,j)]}
  contract=GripperContract(ident,"open_fraction",0,1,1,0,HAND,("right_claw_joint",),DER,limited,prof[-1]["q"],tuple(JointCommandProfile(i,x["fraction"],x["q"]) for i,x in enumerate(reversed(prof))),str(NR),rangeok,{"narrow":n,"wide":w},l,o)
  clpairs=tuple(tuple(sorted((g,OBJ))) for g in contract.all_collision_geom_names);clobs=[]
  for x in closure:
   lp=()
   op=()
   clobs.append(ClosureObservation(x["step_index"],x["fraction"],{"narrow":math.inf if x["narrow_pad"] is None else x["narrow_pad"]/1000,"wide":math.inf if x["wide_pad"] is None else x["wide_pad"]/1000},x["pad_points"],x["pad_geoms"],math.inf if x["linkage_loop"] is None else x["linkage_loop"]/1000,clpairs,lp,op))
  now=time.monotonic_ns();po=PoseSample("test_cylinder","world",CENTER,(0,0,0,1),0,PoseSource.SIM_GROUND_TRUTH,TimeDomain.SIMULATION,model_identity=ident,received_steady_ns=now)
  og=ObjectGeometry("test_cylinder","cylinder",(D,H),"test_cylinder",(OBJ,),MASS);band=ContactBand("test_cylinder",(0,0,1),-.005,.005,"midbody")
  margin=min(float(x) for x in cfg["object"]["table_edge_margins"].values())/1000
  gate=evaluate_gripper_feasibility(GripperFeasibilityRequest(ident,og,band,contract,limits,inv,now,3600000000000,tuple(so),tuple(clobs),margin,FeasibilityThresholds(.001,TOL,.015,.05)))
  lag_ref=None
  lag_trace=SIDE.parent/"dynamic-20261023T063317Z/physics_trace.csv"
  if lag_trace.is_file():
   for rr in csv.DictReader(lag_trace.open(newline="",encoding="utf-8")):
    if int(rr.get("step",-1))==9165:
     tt=json.loads(rr["source_joint_targets_rad"]);aa=json.loads(rr["source_joint_positions_rad"]);jj="right_shoulder_pitch_joint"
     lag_ref={"step":9165,"time_s":float(rr["time_s"]),"target_rad":tt[jj],"actual_rad":aa[jj],"actual_minus_target_rad":aa[jj]-tt[jj],"source":str(lag_trace)}
     break
  rootj=jid(m,"right_claw_joint");act=id_by_name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,"servo_right_claw_joint")
  root_mimic=next(x for x in newroot.findall("./equality/joint") if x.get("name")=="official_root_mimic")
  integrity=src["frozen_candidate_b"]["source_integrity_record"];urdf=pr90.p.UPSTREAM/"X2-Ultra_omnipicker.urdf"
  if sha(urdf)!=integrity["URDF_sha256"] or git(pr90.p.UPSTREAM.parents[0],"rev-parse","HEAD")!=pr90.p.PIN:raise ValueError("official URDF source identity mismatch")
  official_stls=[]
  for path,digest in integrity["STL_sha256"].items():
   if sha(Path(path))!=digest:raise ValueError("official STL SHA mismatch: "+Path(path).name)
   official_stls.append({"path":path,"sha256":digest})
  old_mesh=mesh_inventory(om,root);full_mesh=mesh_inventory(m,newroot)
  official_stl_hashes={x["sha256"] for x in official_stls}
  full_visual_hashes={x["sha256"] for x in full_mesh if x["geom"].endswith("__visual")}
  official_visuals_present=official_stl_hashes<=full_visual_hashes
  isolated=mujoco.MjModel.from_xml_path(str(NX));isolated_mesh=mesh_inventory(isolated,ET.parse(NX).getroot())
  multiset=lambda arr:sorted((x["sha256"],Path(x["path"]).name) for x in arr if x["collision"])
  old_hash,full_hash,iso_hash=multiset(old_mesh),multiset(full_mesh),multiset(isolated_mesh)

  eqs={x.get("name"):x for x in ET.fromstring(OX.read_text()).findall("./equality/*")}
  oldcoef=list(map(float,eqs["geometry_narrow_coupling"].get("polycoef").split()))
  oldq=float(np.polynomial.polynomial.polyval(prof[-1]["q"]["right_claw_joint"],oldcoef));newq=prof[-1]["q"]["R_hand_narrow2_joint"]
  return {"run_id":str(uuid.uuid4()),"physics_steps":0,"mj_step_calls":steps,"mj_forward_calls":forward_total,"mujoco":mujoco.__version__,
   "PR90_head":git(WT,"rev-parse","HEAD"),"PR90_model_sha256":sha(NX),"PR90_range_sha256":sha(NR),"PR90_profile_sha256":sha(NP),"PR90_state_count":len(prof),"PR90_normalized_command_mapping":src["normalized_command_mapping"],"PR90_quartic_coefficients":src["coupling"],
   "old_full_X2_sha256":sha(OX),"old_candidate_geometry_sha256":oldres["candidate_b_model"]["candidate_geometry_sha256"],
   "old_q2_at_qCAD_055":oldq,"new_PR90_q2_at_qCAD_055":newq,"same_qCAD_delta":newq-oldq,"model_sha256":hashlib.sha256((ET.tostring(newroot,encoding="unicode")+"\n").encode()).hexdigest(),
   "counts":counts,"right_wrist_mount":body_mount(m),"root_range_rad":m.jnt_range[rootj].tolist(),
   "root_actuator_ctrlrange":m.actuator_ctrlrange[act].tolist(),"root_actuator":{"name":"servo_right_claw_joint","joint":"right_claw_joint","trnid":m.actuator_trnid[act].tolist()},"official_root_mimic":{k:root_mimic.get(k) for k in ("name","joint1","joint2","polycoef")},"gripper_collision_geom_count":len(gr),"collision_exclusion_count":len(ex),"official_visual_STLs_present_in_full_compiled_inventory":official_visuals_present,"official_URDF_path":str(urdf),"official_URDF_sha256":sha(urdf),"official_URDF_revision":pr90.p.PIN,"official_source_STLs":official_stls,"collision_mesh_inventory":{"old_full_equals_new_full":old_hash==full_hash,"PR90_isolated_equals_full":iso_hash==full_hash,"old_count":len(old_hash),"full_count":len(full_hash),"isolated_count":len(iso_hash),"full_assets":full_mesh},"historical_shoulder_lag":lag_ref,
   "object":{"diameter_m":D,"height_m":H,"mass_kg":MASS,"center_m":list(CENTER),"friction":m.geom_friction[obj].tolist(),"table_distance_mm":dist(m,data,obj,table)*1000},
   "IK":{"status":"PASS_STATIC_IK_WITNESS" if cfg["IK"]["all_pass"] else "FAIL","samples":cfg["IK"]["sample_count"],"max_position_error_mm":cfg["IK"]["max_position_error_mm"],"max_orientation_error_deg":cfg["IK"]["max_orientation_error_deg"],"min_source_margin_rad":cfg["IK"]["min_source_margin_rad"],"dynamic_tracking":False},
   "top_down":{"status":"PASS_STATIC" if topok else "BLOCKED_STATIC","min_clearance_mm":min(x["minimum_clearance_mm"] for x in top),"first_obstruction":next((x for x in top if x["minimum_clearance_mm"]<1 or x["robot_object_contacts"] or x["moving_self_contacts"]),None),"timeline":top},
   "side_route":{"status":"PASS_STATIC" if sideok else ("BLOCKED_STATIC" if side is not None else "NOT_RUN"),"joint_limits":sl,"timeline":side},
   "selected_route":{"name":chosen,"status":"PASS_STATIC" if clear(selected) else "BLOCKED_STATIC","minimum_mm":min(x["minimum_clearance_mm"] for x in selected)},
   "closure":{"status":"PASS_STATIC_PAD_FIRST" if rangeok and bi and padfirst and clean else "BLOCKED_STATIC","limits_pass":limok,"max_D_gap_mm":maxD,"max_internal_penetration_mm":maxpen,"static_range_verified":rangeok,"first_narrow":fn,"first_wide":first_wide,"first_bilateral_midbody":bi,"first_loop":fl,"first_other":fo,"pads_before_loop":padfirst,"bilateral_no_loop_count":len(clean),"timeline":closure},
   "feasibility_interface":{"status":gate.status.value,"blockers":list(gate.blockers),"first_obstruction":None if gate.first_obstruction is None else [gate.first_obstruction.geom_a,gate.first_obstruction.geom_b,gate.first_obstruction.signed_distance_m*1000]},
   "stage_status":{"PREGRASP":"STATIC_ONLY","APPROACH":"STATIC_ONLY","GRASP":"NOT RUN","LIFT":"NOT RUN","TRANSFER":"NOT RUN","RELEASE":"NOT RUN","PLACE":"NOT RUN","SETTLE":"NOT RUN"},
   "isolated_PR90_unloaded_validation":{"status":"PASS_REPORTED_BY_MAINTAINER","mujoco_version":"3.3.6","physics_steps":17500,"duration_s":17.5,"sequence":"2% HOLD -> 100% OPEN -> HOLD -> 2% CLOSE -> HOLD","minimum_five_section_aperture_mm":{"2_percent":12.831,"100_percent":90.999,"returned_2_percent":12.831},"max_root_tracking_error_rad":0.015576,"unchanged_tracking_limit_rad":0.02,"scope":"isolated unloaded gripper only; not full-X2 bottle contact validation"},
   "evidence_boundary":"This full-X2 result is static collision-proxy geometry only; it makes no physical force/hold/lift claim."},ET.tostring(newroot,encoding="unicode")+"\n"
 finally:mujoco.mj_forward=of;mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=os
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-parent",default=str(REV/"issue-46"));a=p.parse_args()
 parent=pg.validate_output_path(a.output_parent,authorized_roots=(str(REV),),require_exists=True,expected_kind="directory",environment="wsl")
 r,xml=evaluate();out=pg.create_unique_output_directory(parent.wsl_path,prefix="pr90-quartic-full-x2-static",authorized_roots=(str(REV),),environment="wsl")
 r["output"]=out.as_dict();r["runner_sha256"]=sha(Path(__file__).resolve())
 pg.write_new_text(out.native_path/"full_x2_pr90_quartic_static.xml",xml,authorized_roots=(str(REV),),environment="wsl")
 pg.write_new_text(out.native_path/"result.json",json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n",authorized_roots=(str(REV),),environment="wsl")
 topfirst=r["top_down"]["first_obstruction"]
 sidefirst=None if not r["side_route"]["timeline"] else next((x for x in r["side_route"]["timeline"] if x["minimum_clearance_mm"]<1 or x["robot_object_contacts"] or x["moving_self_contacts"]),None)
 c=r["closure"];openrow=c["timeline"][0]
 p59=next((x for x in c["timeline"] if x["percent"]==59.0),None)
 p60=next((x for x in c["timeline"] if x["percent"]==60.0),None)
 worst_internal=min((y for x in c["timeline"] for y in x["internal_contacts"]),key=lambda x:x["dist_mm"],default=None)
 table_mins=[x.get("gripper_table_clearance_mm") for x in r["top_down"]["timeline"] if x.get("gripper_table_clearance_mm") is not None]
 table_best=min(r["top_down"]["timeline"],key=lambda x:x.get("gripper_table_clearance_mm") if x.get("gripper_table_clearance_mm") is not None else float("inf"))
 report="\n".join([
  "# PR90 quartic candidate in the full-X2 static preflight","",
  f"Run ID: {r['run_id']}. Model SHA-256: {r['model_sha256']}.",
  f"Evidence: {out.wsl_path}",
  f"Windows UNC: {out.windows_unc}","",
  "## Result","",
  f"- PR #90 input: HEAD {r['PR90_head']}; isolated static MJCF SHA {r['PR90_model_sha256']}; source classification remains SIMULATION_ONLY.",
  f"- Full-X2 model uses the exact PR #90 101-state profile and the unchanged official right wrist mount under {r['right_wrist_mount']['parent']} at xyz {r['right_wrist_mount']['xyz_m']} m, quat wxyz {r['right_wrist_mount']['quat_wxyz']}.",
  f"- Root actuator mapping: {r['root_actuator']}; official root mimic: {r['official_root_mimic']}; PR #90 normalized mapping: {r['PR90_normalized_command_mapping']}.",
  f"- Static path: top-down {r['top_down']['status']} (85 samples; minimum {r['top_down']['min_clearance_mm']:.3f} mm); existing side/oblique route {r['side_route']['status']}.",
  f"- Closure: {c['status']}; first bilateral pad proximity within the specified 10 mm mid-body band: {c['first_bilateral_midbody']}.",
  f"- Feasibility interface: {r['feasibility_interface']['status']} with blockers {r['feasibility_interface']['blockers']}.",
  f"- Runtime accounting: physics steps {r['mj_step_calls']}; forbidden step calls {r['mj_step_calls']}; mj_forward calls {r['mj_forward_calls']}.",
  "- No grasp, load transfer, lift, transfer, release, place, or settle is claimed.","",
  "## Old Candidate B versus PR #90",
  f"- Old Candidate B geometry source SHA: {r['old_candidate_geometry_sha256']}; old full-X2 MJCF SHA: {r['old_full_X2_sha256']}.",
  f"- At qCAD 0.55 the old quadratic gives narrow2={r['old_q2_at_qCAD_055']:.9f} rad; PR #90's recorded state is {r['new_PR90_q2_at_qCAD_055']:.9f} rad (delta {r['same_qCAD_delta']:.9f} rad). Same qCAD is not the same passive state.",
  "- The generated model changes only the two q2 equality laws in a research copy. The robot geometry, right mount, actuator, joint limits, collision masks, exclusions, and compiled topology stay unchanged.",
  f"- Collision mesh hashes: old full X2 equals new full X2={r['collision_mesh_inventory']['old_full_equals_new_full']}; PR #90 isolated fixture equals assembled full X2={r['collision_mesh_inventory']['PR90_isolated_equals_full']}; each inventory contains {r['collision_mesh_inventory']['full_count']} collision mesh records.",
  f"- All nine pinned official source visual STL hashes occur in the compiled full-X2 gripper visual inventory: {r['official_visual_STLs_present_in_full_compiled_inventory']}.",
  f"- The original X2 URDF SHA is {r['official_URDF_sha256']} at revision {r['official_URDF_revision']}; verified source STL hashes are recorded in result.json.","",
  "## 70 mm bottle insertion and first obstruction",
  f"- Object: upright free cylinder, diameter {D*1000:.0f} mm, height {H*1000:.0f} mm, mass {MASS:.3f} kg, center {CENTER} m; original friction retained.",
  f"- First top-down sample is already blocked: {topfirst}. This is the wide-side linkage-loop collision with the bottle at PREGRASP; minimum over the sampled path is {r['top_down']['min_clearance_mm']:.3f} mm.",
  f"- Side/oblique alternative uses the saved full-X2 arm witness and is also blocked. Its first obstruction is {sidefirst}.",
  f"- Minimum gripper/table clearance on the sampled top-down route: {min(table_mins):.3f} mm at pair {table_best.get('gripper_table_nearest_pair')}." if table_mins else "- Gripper/table clearance: unavailable in the sampled collision inventory.",
  "- The path reuses the prior 43 IK samples and 42 adjacent joint-space midpoints. This is a sampled static witness, not continuous collision detection.",
  "- The unchanged saved IK witness passes its static joint-limit and residual checks. It does not prove shoulder tracking; the historical trace recorded shoulder-pitch lag of "
   f"{r['historical_shoulder_lag']['actual_minus_target_rad']:.6f} rad at step 9165, so no positive practical approach clearance is established here.","",
  "## Closure ordering and full-height interference",
  f"- PR #90's isolated static evidence reports minimum five-section open width 90.992 mm at 100 percent. That cross-section does not guarantee bottle clearance over the full gripper height.",
  f"- At the selected full-X2 grasp pose with 100 percent open, pad clearances are {openrow['narrow_pad']:.3f}/{openrow['wide_pad']:.3f} mm while linkage loop and other gripper geometry are already at {openrow['linkage_loop']:.3f}/{openrow['other_gripper']:.3f} mm.",
  f"- First wide pad proximity occurs at {c['first_wide']['percent']:.0f}% (qCAD {c['first_wide']['qCAD']:.6f}); first narrow pad at {c['first_narrow']['percent']:.0f}% (qCAD {c['first_narrow']['qCAD']:.6f}). Linkage loop proximity already exists at the open endpoint {c['first_loop']['percent']:.0f}% ({c['first_loop']['linkage_loop']:.3f} mm).",
  f"- At 59 percent both pad proxies overlap, but the nearest candidate pad points are outside the accepted mid-body band: narrow z={p59['pad_points']['narrow'][0][2]*1000:.2f} mm and wide z={p59['pad_points']['wide'][0][2]*1000:.2f} mm relative to bottle center. Thus no bilateral mid-body pair is accepted.",
  f"- Worst internal proxy interference during the 101-state profile is {c['max_internal_penetration_mm']:.3f} mm at {worst_internal}; max D-site component gap is {c['max_D_gap_mm']:.8g} mm. Joint limits pass, but the full static range gate fails because of this internal collision.",
  "- Detailed collision and closure timelines, exact geom identities, signed distances, transforms, and hashes are in result.json; CSV exports are provided alongside.","",
  "## Local04 handoff",
  "Do not tune grasp or alter this model in Local02. Review whether the isolated PR #90 root-frame convention corresponds to the unchanged mounted full-X2 frame; the collision proxy mesh hashes match, but the assembled full-X2 model retains a 1.752 mm narrow3/wide3 interference at the closed endpoint and the bottle is struck by linkage loops while open. The static quartic alone therefore does not solve the full-height insertion corridor.",
  "Physical stages GRASP/LIFT/TRANSFER/RELEASE/PLACE/SETTLE remain NOT RUN.",""])
 fields=["sample","route","minimum_clearance_mm","nearest_pair","gripper_object_clearance_by_group_mm","gripper_table_clearance_mm","robot_object_contacts","moving_self_contacts"]
 def csv_text(rows,cols):
  stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=cols,extrasaction="ignore");writer.writeheader()
  for row in rows:
   item={}
   for key in cols:
    value=row.get(key,"")
    item[key]=json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
   writer.writerow(item)
  return stream.getvalue()
 path_rows=r["top_down"]["timeline"]+(r["side_route"]["timeline"] or [])
 closure_cols=["step_index","percent","fraction","qCAD","limits_pass","narrow_pad","wide_pad","linkage_loop","other_gripper","wrist_arm_mm","gripper_table_mm","D_gap_mm","pad_points","pad_geoms","object_contacts","internal_contacts"]
 pg.write_new_text(out.native_path/"collision_timeline.csv",csv_text(path_rows,fields),authorized_roots=(str(REV),),environment="wsl")
 pg.write_new_text(out.native_path/"closure_timeline.csv",csv_text(r["closure"]["timeline"],closure_cols),authorized_roots=(str(REV),),environment="wsl")
 handoff="\n".join(["# Local04 handoff: PR #90 in full X2","",
  f"Static-only result from run {r['run_id']} at full-X2 model SHA {r['model_sha256']}.",
  f"PR #90 exact HEAD {r['PR90_head']}; isolated MJCF SHA {r['PR90_model_sha256']}. No source/model from PR #90 was changed.",
  f"Open vertical insertion first hits {topfirst['nearest_pair']} at {topfirst['minimum_clearance_mm']:.3f} mm; that is a wide loop proxy against the bottle at the saved hover.",
  f"At 100 percent open and the selected grasp pose, loop clearance is {openrow['linkage_loop']:.3f} mm and other gripper clearance {openrow['other_gripper']:.3f} mm.",
  f"During closure, loop interference is already present before pad proximity; bilateral pads inside the 10 mm mid-body band: {c['first_bilateral_midbody']}.",
  f"At 0 percent, max full-X2 internal interference is {c['max_internal_penetration_mm']:.3f} mm: {worst_internal}.",
  f"Collision mesh hashes match old full X2 and PR #90 isolated fixture: {r['collision_mesh_inventory']['old_full_equals_new_full']} / {r['collision_mesh_inventory']['PR90_isolated_equals_full']}.",
  "Recommended next check: compare the isolated fixture root frame and fixed-link/body transforms against the official mounted full-X2 assembly at the exact saved PR90 states. The evidence points to persistent full-height loop/object collision and a closed-end pad self-interference; it does not identify a unique correction to the quartic coefficients.",
  "No full-X2 bottle/contact dynamic validation was performed; the isolated unloaded PR90 cycle is separate maintainer-reported evidence. No geometry/pose change is authorized by this handoff.",""])
 pg.write_new_text(out.native_path/"LOCAL04_HANDOFF.md",handoff,authorized_roots=(str(REV),),environment="wsl")
 pg.write_new_text(out.native_path/"REPORT.md",report,authorized_roots=(str(REV),),environment="wsl")
 cmd=("cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023\n"
  "env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python "
  f"{Path(__file__).resolve()} --output-parent {parent.wsl_path}\n")
 pg.write_new_text(out.native_path/"REPRODUCE.md",cmd,authorized_roots=(str(REV),),environment="wsl")
 man=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
 pg.verify_artifacts(man["artifacts"],authorized_roots=(str(REV),),environment="wsl")
 print(json.dumps({"output":out.as_dict(),"run_id":r["run_id"],"top_down":r["top_down"]["status"],"side":r["side_route"]["status"],"closure":r["closure"]["status"],"feasibility":r["feasibility_interface"]["status"],"steps":r["mj_step_calls"],"forwards":r["mj_forward_calls"]},indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
