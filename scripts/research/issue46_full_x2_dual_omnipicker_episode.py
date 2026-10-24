#!/usr/bin/env python3
"""One bounded full-X2 Option-D dual OmniPicker physical regression, MuJoCo 3.3.6."""
from __future__ import annotations
import csv,gzip,hashlib,json,math,os,platform,subprocess,sys,xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
import numpy as np
import mujoco
import imageio_ffmpeg

SCENE=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/official-wrist-variant-static-audit-20261024T162630Z/verified/full_x2_d_right_ext_collision_diagnostic.xml")
FIX=Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/tests/fixtures/pr90_static_samples.csv")
MODEL=Path("/home/lzy18001500226/robotsim-issue46-full-x2-dual-omnipicker-20261024/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/model.xml")
VENDOR=Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/reference/AgibotTech/agibot_x2_urdf")
URDF_ROBOT=VENDOR/"X2_URDF-v1.4.0/X2-Ultra.urdf";URDF_HAND=VENDOR/"X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
OUT=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/full-x2-omnipicker-physical-grasp-20261024T181317Z")
RUN_OUT=OUT/"stage2_run01"
DT=.001;MAX_STEPS=30000;SETTLE_MAX=2000;WINDOW=1000;LIN_MAX=.020;ANG_MAX=.050;ROOT_ERR_MAX=.020;MIMIC_MAX=.003
SUPPORT_MIN=.80;SUPPORT_MAX=1.20;NECK_BASE=-.007112461424559693;NECK_EXTRA=.0005
OPEN_STEPS=CLOSE_STEPS=4000;HOLD_STEPS=250;FINAL_HOLD=1000;VIDEO_FPS=20;FRAME_EVERY=50

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(m,obj,i):return mujoco.mj_id2name(m,obj,int(i)) or f"unnamed_{int(i)}"
def git(path,*args):
 try:return subprocess.check_output(["git","-C",str(path),*args],text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return f"UNAVAILABLE: {e}"
def source_limits(path):
 out={};root=ET.parse(path).getroot()
 for j in root.findall("joint"):
  x=j.find("limit")
  if x is None:continue
  def num(k):return None if x.get(k) is None else float(x.get(k))
  vel=num("velocity");eff=num("effort")
  out[j.get("name")]={"lower":num("lower"),"upper":num("upper"),"velocity":vel if vel and vel>0 else None,"velocity_source_value":vel,"effort":eff if eff and eff>0 else None,"effort_source_value":eff}
 return out
def set_hand(m,d,row,side):
 root=float(row["right_claw_joint_rad"]);n2=float(row["narrow2_rad"]);n3=float(row["narrow3_rad"]);nl=float(row["narrow_loop_rad"]);p="R" if side=="right" else "L"
 values={side+"_claw_joint":root,p+"_hand_wide1_joint":-root,p+"_hand_narrow2_joint":n2,p+"_hand_wide2_joint":-n2,p+"_hand_narrow3_joint":n3,p+"_hand_wide3_joint":-n3,p+"_hand_narrow_loop_joint":nl,p+"_hand_wide_loop_joint":-nl}
 for n,q in values.items():
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
  if j<0:raise RuntimeError("Missing fixture joint "+n)
  d.qpos[int(m.jnt_qposadr[j])]=q
def collect_contacts(m,d):
 out=[];bottle=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"canonical_bottle")
 for k in range(d.ncon):
  c=d.contact[k];g1=int(c.geom1);g2=int(c.geom2);b1=int(m.geom_bodyid[g1]);b2=int(m.geom_bodyid[g2]);n1=name(m,mujoco.mjtObj.mjOBJ_BODY,b1);n2=name(m,mujoco.mjtObj.mjOBJ_BODY,b2);gn1=name(m,mujoco.mjtObj.mjOBJ_GEOM,g1);gn2=name(m,mujoco.mjtObj.mjOBJ_GEOM,g2);pair={n1,n2}
  neck=("head_pitch_link" in pair and "torso_link" in pair);table=("canonical_bottle" in pair and "m0_table" in pair);floor=((gn1=="floor" or gn2=="floor") and any(t in n1+n2 for t in ["ankle","foot"]))
  local=np.zeros(6);mujoco.mj_contactForce(m,d,k,local);R=np.asarray(c.frame,dtype=float).reshape(3,3);f2=R.T@local[:3]
  fb=(f2 if b2==bottle else (-f2 if b1==bottle else None))
  out.append({"geom1":gn1,"geom2":gn2,"body1":n1,"body2":n2,"distance_m":float(c.dist),"position_world_m":np.asarray(c.pos,dtype=float).tolist(),"wrench_contact_frame_N_Nm":local.tolist(),"force_on_geom2_world_N":f2.tolist(),"force_on_bottle_world_N":None if fb is None else fb.tolist(),"allowed":bool(neck or table or floor),"category":"neck_baseline" if neck else ("bottle_table_support" if table else ("foot_floor_support" if floor else "unexpected"))})
 return out
def object_state(m,d,bid):
 v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,v,0);q=np.zeros(4);mujoco.mju_mat2Quat(q,np.asarray(d.xmat[bid]).reshape(-1))
 return {"com_world_m":np.asarray(d.xipos[bid]).tolist(),"body_origin_world_m":np.asarray(d.xpos[bid]).tolist(),"quat_wxyz":q.tolist(),"angular_velocity_world_rad_s":v[:3].tolist(),"linear_velocity_world_m_s":v[3:].tolist(),"angular_speed_rad_s":float(np.linalg.norm(v[:3])),"linear_speed_m_s":float(np.linalg.norm(v[3:]))}
def equalities(m,d):
 joint={};connect=[]
 for e in range(m.neq):
  if int(m.eq_type[e])==int(mujoco.mjtEq.mjEQ_JOINT):
   j1=int(m.eq_obj1id[e]);j2=int(m.eq_obj2id[e]);a=float(d.qpos[int(m.jnt_qposadr[j1])]);b=float(d.qpos[int(m.jnt_qposadr[j2])]);co=np.asarray(m.eq_data[e,:5]);joint[name(m,mujoco.mjtObj.mjOBJ_EQUALITY,e)]=a-sum(float(co[k])*b**k for k in range(5))
 for r in range(d.nefc):
  if int(d.efc_type[r])==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
   e=int(d.efc_id[r])
   if int(m.eq_type[e])==int(mujoco.mjtEq.mjEQ_CONNECT):connect.append({"equality":name(m,mujoco.mjtObj.mjOBJ_EQUALITY,e),"residual_m":float(d.efc_pos[r])})
 return joint,connect
def limit_check(m,d,src):
 badp=[];badv=[];worst={"ratio":0.0,"joint":None,"qvel":0.0,"limit":None};margin=float("inf");margin_joint=None
 for j in range(m.njnt):
  n=name(m,mujoco.mjtObj.mjOBJ_JOINT,j);s=src.get(n)
  if s and s["lower"] is not None and s["upper"] is not None and s["upper"]>s["lower"]:
   lo=float(s["lower"]);hi=float(s["upper"]);q=float(d.qpos[int(m.jnt_qposadr[j])]);marg=min(q-lo,hi-q)
   if marg<margin:margin=marg;margin_joint=n
   if q<lo or q>hi:badp.append({"joint":n,"qpos":q,"range":[lo,hi]})
  elif bool(m.jnt_limited[j]):
   lo,hi=map(float,m.jnt_range[j]);q=float(d.qpos[int(m.jnt_qposadr[j])]);marg=min(q-lo,hi-q)
   if marg<margin:margin=marg;margin_joint=n
   if q<lo or q>hi:badp.append({"joint":n,"qpos":q,"range":[lo,hi]})
  if s and s["velocity"] is not None:
   v=float(d.qvel[int(m.jnt_dofadr[j])]);lim=float(s["velocity"]);ratio=abs(v)/lim
   if ratio>worst["ratio"]:worst={"ratio":ratio,"joint":n,"qvel":v,"limit":lim}
   if abs(v)>lim:badv.append({"joint":n,"qvel":v,"limit":lim})
 return badp,badv,{"margin_rad":None if math.isinf(margin) else margin,"joint":margin_joint},worst
def asset_hashes(scene):
 root=ET.parse(scene).getroot();out=[]
 for e in root.findall("./asset/*"):
  f=e.get("file")
  if f:
   p=Path(f);out.append({"type":e.tag,"name":e.get("name"),"path":str(p),"sha256":sha(p) if p.is_file() else None,"size_bytes":p.stat().st_size if p.is_file() else None})
 return out

def main():
 RUN_OUT.mkdir(parents=True,exist_ok=True);result_path=RUN_OUT/"stage2_result.json"
 if result_path.exists() and "--preflight-only" not in sys.argv:raise SystemExit(f"Refusing repeat episode: {result_path} exists")
 if mujoco.__version__!="3.3.6":raise SystemExit(f"Expected MuJoCo 3.3.6, got {mujoco.__version__}")
 for p in [SCENE,FIX,MODEL,URDF_ROBOT,URDF_HAND]:
  if not p.is_file():raise SystemExit(f"Missing required input: {p}")
 m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 if abs(float(m.opt.timestep)-DT)>1e-12:raise SystemExit(f"Unexpected model timestep {m.opt.timestep}")
 samples=list(csv.DictReader(FIX.open(newline="")));r2=next(r for r in samples if float(r["command_percent"])==2.0);r100=next(r for r in samples if float(r["command_percent"])==100.0)
 mujoco.mj_resetData(m,d);set_hand(m,d,r2,"left");set_hand(m,d,r2,"right");d.qvel[:]=0;mujoco.mj_forward(m,d)
 bottle=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"canonical_bottle");pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
 if bottle<0 or mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"m0_table")<0:raise SystemExit("Complete table/bottle scene is missing")
 if int(m.body_mocapid[bottle])>=0:raise SystemExit("Bottle is mocap-controlled")
 for e in range(m.neq):
  en=name(m,mujoco.mjtObj.mjOBJ_EQUALITY,e)
  if "bottle" in en.lower() or int(m.eq_obj1id[e])==bottle or int(m.eq_obj2id[e])==bottle:raise SystemExit("Bottle attachment/equality found")
 act_names=[name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a) for a in range(m.nu)];followers=[];servo={}
 for a,an in enumerate(act_names):
  if an in ("servo_right_claw_joint","servo_left_claw_joint"):servo["right" if "right" in an else "left"]=a
  j=int(m.actuator_trnid[a,0]);jn=name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
  if (jn.startswith("R_hand_") or jn.startswith("L_hand_")) and jn not in ("right_claw_joint","left_claw_joint"):followers.append({"actuator":an,"joint":jn})
 if set(servo)!={"left","right"} or followers:raise SystemExit(f"Unexpected hand actuator map: servos={servo}, followers={followers}")
 asset_root=ET.parse(SCENE).getroot();mesh_files={x.get("name"):x.get("file") for x in asset_root.findall("./asset/mesh")}
 rw=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"right_wrist_roll_link");d_mesh=[]
 for g in range(m.ngeom):
  if int(m.geom_bodyid[g])==rw and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])):
   mid=int(m.geom_dataid[g]);mn=name(m,mujoco.mjtObj.mjOBJ_MESH,mid) if mid>=0 else None;d_mesh.append({"mesh_asset":mn,"file":mesh_files.get(mn)})
 if not any(x["file"] and Path(x["file"]).name=="right_wrist_roll_extend_link.stl" for x in d_mesh):raise SystemExit(f"Option-D collision STL absent: {d_mesh}")
 src={**source_limits(URDF_ROBOT),**source_limits(URDF_HAND)};c0=collect_contacts(m,d);bad0=[x for x in c0 if not x["allowed"]]
 if bad0:raise SystemExit(f"Unexpected initial active contacts: {bad0}")
 je0,ce0=equalities(m,d)
 if max([abs(x) for x in je0.values()] or [0])>MIMIC_MAX:raise SystemExit(f"Initial mimic residual too large: {je0}")
 if abs(float(m.body_mass[bottle])-.570)>1e-6:raise SystemExit(f"Canonical bottle mass changed: {m.body_mass[bottle]}")
 if "--preflight-only" in sys.argv:
  vh=git(VENDOR,"rev-parse","HEAD");vd=git(VENDOR,"status","--porcelain");repo=Path("/home/lzy18001500226/robotsim-issue46-full-x2-dual-omnipicker-20261024");rh=git(repo,"rev-parse","HEAD");md=git(repo,"status","--porcelain","--",str(MODEL.relative_to(repo)))
  if vh!="575cc6b988f976c23550e0db85aa1e5475d3652d" or vd:raise SystemExit(f"Vendor source identity/cleanliness failed: {vh}, {vd}")
  if rh!="5e2ed1dcbfac2e357fd19191733b377034ea09f5" or md:raise SystemExit(f"PR92 model baseline failed: {rh}, {md}")
  print(json.dumps({"preflight_only":True,"physics_steps":0,"MuJoCo":mujoco.__version__,"scene_sha256":sha(SCENE),"canonical_model_sha256":sha(MODEL),"vendor_head":vh,"initial_contacts":c0,"joint_eq_residuals_rad":je0,"connect_eq_residuals_m":ce0,"Option_D_wrist_collision_meshes":d_mesh,"follower_actuators":followers,"bottle_mass_kg":float(m.body_mass[bottle])},indent=2,allow_nan=False));return
 # Renderer and MP4 encoder are fully initialized before the only authorized physics episode.
 os.environ.setdefault("MUJOCO_GL","egl")
 renderer=mujoco.Renderer(m,height=480,width=480)
 fullcam=mujoco.MjvCamera();fullcam.type=mujoco.mjtCamera.mjCAMERA_FREE;fullcam.lookat[:]=[.30,.16,.82];fullcam.distance=2.2;fullcam.azimuth=135;fullcam.elevation=-18
 closecam=mujoco.MjvCamera();closecam.type=mujoco.mjtCamera.mjCAMERA_FREE;closecam.distance=.42;closecam.azimuth=135;closecam.elevation=-8
 def frame(stage):
  renderer.update_scene(d,camera=fullcam);whole=renderer.render()
  side="left" if stage.startswith("LEFT") or stage.startswith("INITIAL") else "right"
  bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"L_omnipicker_base_link" if side=="left" else "R_omnipicker_base_link")
  closecam.lookat[:]=d.xpos[bid];renderer.update_scene(d,camera=closecam);hand=renderer.render()
  return np.ascontiguousarray(np.concatenate([whole,hand],axis=1),dtype=np.uint8)
 video=RUN_OUT/"full_x2_dual_omnipicker_stage2_actual_mujoco.mp4"
 vgen=imageio_ffmpeg.write_frames(str(video),(960,480),fps=VIDEO_FPS,quality=7,codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",output_params=["-preset","ultrafast","-movflags","+faststart"]);vgen.send(None);vgen.send(frame("INITIAL"))
 motors=[a for a,n in enumerate(act_names) if n not in ("servo_right_claw_joint","servo_left_claw_joint")]
 qtarget={int(m.actuator_trnid[a,0]):float(d.qpos[int(m.jnt_qposadr[int(m.actuator_trnid[a,0])])]) for a in motors}
 gains={}
 for a in motors:
  j=int(m.actuator_trnid[a,0]);jn=name(m,mujoco.mjtObj.mjOBJ_JOINT,j);gains[a]=(150.,22.) if any(k in jn for k in ["shoulder","elbow","wrist"]) else (250.,30.)
  if not bool(m.actuator_ctrllimited[a]):raise SystemExit(f"Existing motor ctrl limit missing: {act_names[a]}")
 qheaders=[];vheaders=[]
 for j in range(m.njnt):
  n=name(m,mujoco.mjtObj.mjOBJ_JOINT,j);jt=int(m.jnt_type[j]);nq=7 if jt==int(mujoco.mjtJoint.mjJNT_FREE) else (4 if jt==int(mujoco.mjtJoint.mjJNT_BALL) else 1);nv=6 if jt==int(mujoco.mjtJoint.mjJNT_FREE) else (3 if jt==int(mujoco.mjtJoint.mjJNT_BALL) else 1)
  qheaders += [f"qpos_{n}_{i}" for i in range(nq)];vheaders += [f"qvel_{n}_{i}" for i in range(nv)]
 joint_headers=qheaders+vheaders
 statef=gzip.open(RUN_OUT/"joint_actuator_trace.csv.gz","wt",newline="",encoding="utf8");sw=csv.writer(statef);sw.writerow(["step","time_s","stage"]+joint_headers+[f"ctrl_{n}" for n in act_names]+[f"force_{n}" for n in act_names])
 contactf=gzip.open(RUN_OUT/"contact_physics_trace.jsonl.gz","wt",encoding="utf8")
 bottlef=gzip.open(RUN_OUT/"bottle_support_trace.csv.gz","wt",newline="",encoding="utf8");bw=csv.writer(bottlef);bw.writerow(["step","time_s","stage","com_x_m","com_y_m","com_z_m","linear_speed_m_s","angular_speed_rad_s","table_contact_count","table_support_on_bottle_z_N","hand_support_on_bottle_z_N","neck_contact_min_m","max_joint_eq_resid_rad","max_connect_eq_resid_m","max_root_tracking_error_rad","source_velocity_ratio"])
 assets=asset_hashes(SCENE);vend_head=git(VENDOR,"rev-parse","HEAD")
 if vend_head!="575cc6b988f976c23550e0db85aa1e5475d3652d":raise SystemExit(f"Pinned vendor revision mismatch: {vend_head}")
 vendor_dirty=git(VENDOR,"status","--porcelain")
 if vendor_dirty:raise SystemExit(f"Pinned vendor checkout is dirty: {vendor_dirty}")
 repo=Path("/home/lzy18001500226/robotsim-issue46-full-x2-dual-omnipicker-20261024");repo_head=git(repo,"rev-parse","HEAD")
 model_dirty=git(repo,"status","--porcelain","--",str(MODEL.relative_to(repo)))
 if repo_head!="5e2ed1dcbfac2e357fd19191733b377034ea09f5" or model_dirty:raise SystemExit(f"Integration baseline/model mismatch: HEAD={repo_head}, model_status={model_dirty}")
 manifest={"schema":"robotsim.issue46.full_x2_dual_omnipicker_stage2_manifest.v1","attempt_id":"a57b153e-aa31-43f0-a598-4a02457c17b1","mujoco_version":mujoco.__version__,"python":sys.version,"platform":platform.platform(),"scene_path":str(SCENE),"scene_sha256":sha(SCENE),"canonical_model_path":str(MODEL),"canonical_model_sha256":sha(MODEL),"pinned_vendor_head":vend_head,"vendor_clean":not bool(vendor_dirty),"vendor_expected":"575cc6b988f976c23550e0db85aa1e5475d3652d","integration_worktree_head":repo_head,"canonical_model_worktree_clean":not bool(model_dirty),"robot_urdf_sha256":sha(URDF_ROBOT),"hand_urdf_sha256":sha(URDF_HAND),"fixture_sha256":sha(FIX),"runner_sha256":sha(Path(__file__).resolve()),"assets":assets,"option_d_right_wrist_collision_meshes":d_mesh,"fixed_base_world_pos_m":np.asarray(m.body_pos[pelvis]).tolist(),"model_counts":{"nq":m.nq,"nv":m.nv,"nu":m.nu,"neq":m.neq,"ngeom":m.ngeom,"nbody":m.nbody},"timestep_s":float(m.opt.timestep),"physics_budget":{"stage2_max":MAX_STEPS,"planned_max":28250,"stage3_5_max":50000,"combined_max":80000,"previous_steps":0},"initial_contacts":c0,"initial_joint_eq_residual_rad":je0,"initial_connect_eq_residual_m":ce0,"controller":{"arm_kp_kd":[150,22],"body_kp_kd":[250,30],"motor_ctrlrange_unchanged":True,"native_gripper_actuators_unchanged":True,"gripper_force_cap_Nm":.5,"trajectory":"quintic 2%-100%-2%, 4000 steps per ramp"},"limits_note":"URDF gripper velocity/effort fields are zero placeholders and are reported as unspecified; positive source velocity limits and all position ranges are enforced."}
 (RUN_OUT/"stage2_run_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
 physics_steps=0;stage="INITIAL_SETTLE";first_failure=None;warn0=np.asarray(d.warning.number,dtype=int).copy();initial_com=np.asarray(d.xipos[bottle],dtype=float).copy();mass=float(m.body_mass[bottle]);weight=mass*9.81
 if abs(mass-.570)>1e-6:raise SystemExit(f"Bottle mass mismatch {mass}")
 claw={"left":float(r2["right_claw_joint_rad"]),"right":float(r2["right_claw_joint_rad"])};flags=deque(maxlen=WINDOW);stable_window=None
 metrics={"max_bottle_com_rise_m":0.0,"max_bottle_linear_speed_m_s":0.0,"max_bottle_angular_speed_rad_s":0.0,"max_joint_eq_residual_rad":0.0,"max_connect_eq_residual_m":0.0,"max_root_tracking_error_rad":0.0,"max_abs_joint_qvel_rad_s":{},"max_abs_actuator_force":{},"max_source_velocity_ratio":{"ratio":0.0,"joint":None,"qvel":0.0,"limit":None},"min_neck_contact_distance_m":None,"table_support_min_N":None,"table_support_max_N":None}
 phases={"initial_settle":"NOT RUN","left_gripper_dynamic":"NOT RUN","right_gripper_dynamic":"NOT RUN","dual_gripper_dynamic":"NOT RUN"}
 def sample(step,stage,draw=True):
  nonlocal first_failure
  cs=collect_contacts(m,d);bad=[c for c in cs if not c["allowed"]]
  if bad and first_failure is None:first_failure={"step":step,"time_s":float(d.time),"stage":stage,"gate":"UNEXPECTED_ACTIVE_CONTACT","contacts":bad}
  neck=[c["distance_m"] for c in cs if c["category"]=="neck_baseline"];nmin=min(neck) if neck else None
  if nmin is not None:
   metrics["min_neck_contact_distance_m"]=nmin if metrics["min_neck_contact_distance_m"] is None else min(metrics["min_neck_contact_distance_m"],nmin)
   if nmin<NECK_BASE-NECK_EXTRA and first_failure is None:first_failure={"step":step,"time_s":float(d.time),"stage":stage,"gate":"NECK_BASELINE_PENETRATION_WORSENED","distance_m":nmin,"baseline_m":NECK_BASE}
  support=np.zeros(3);hand=np.zeros(3);table_count=0
  for c in cs:
   f=c["force_on_bottle_world_N"]
   if f is None:continue
   if c["category"]=="bottle_table_support":support+=np.asarray(f);table_count+=1
   elif any(t in c["body1"]+c["body2"] for t in ["R_hand_","L_hand_","omnipicker"]):hand+=np.asarray(f)
  bs=object_state(m,d,bottle);rise=bs["com_world_m"][2]-initial_com[2]
  metrics["max_bottle_com_rise_m"]=max(metrics["max_bottle_com_rise_m"],rise);metrics["max_bottle_linear_speed_m_s"]=max(metrics["max_bottle_linear_speed_m_s"],bs["linear_speed_m_s"]);metrics["max_bottle_angular_speed_rad_s"]=max(metrics["max_bottle_angular_speed_rad_s"],bs["angular_speed_rad_s"])
  fz=float(support[2]);metrics["table_support_min_N"]=fz if metrics["table_support_min_N"] is None else min(metrics["table_support_min_N"],fz);metrics["table_support_max_N"]=fz if metrics["table_support_max_N"] is None else max(metrics["table_support_max_N"],fz)
  je,ce=equalities(m,d);jmax=max([abs(v) for v in je.values()] or [0]);cmax=max([abs(x["residual_m"]) for x in ce] or [0]);metrics["max_joint_eq_residual_rad"]=max(metrics["max_joint_eq_residual_rad"],jmax);metrics["max_connect_eq_residual_m"]=max(metrics["max_connect_eq_residual_m"],cmax)
  if jmax>MIMIC_MAX and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"MIMIC_EQUALITY_RESIDUAL","max_rad":jmax,"limit_rad":MIMIC_MAX}
  badp,badv,margin,worst=limit_check(m,d,src)
  if badp and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"SOURCE_POSITION_LIMIT","violations":badp}
  if badv and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"SOURCE_VELOCITY_LIMIT","violations":badv}
  if worst["ratio"]>metrics["max_source_velocity_ratio"]["ratio"]:metrics["max_source_velocity_ratio"]=worst
  for j in range(m.njnt):
   n=name(m,mujoco.mjtObj.mjOBJ_JOINT,j);v=abs(float(d.qvel[int(m.jnt_dofadr[j])]));metrics["max_abs_joint_qvel_rad_s"][n]=max(v,metrics["max_abs_joint_qvel_rad_s"].get(n,0.0))
  rooterr={}
  for side,a in servo.items():
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,side+"_claw_joint");rooterr[side]=abs(float(d.qpos[int(m.jnt_qposadr[j])])-float(d.ctrl[a]))
  rerr=max(rooterr.values());metrics["max_root_tracking_error_rad"]=max(metrics["max_root_tracking_error_rad"],rerr)
  if rerr>ROOT_ERR_MAX and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"ROOT_TRACKING_ERROR","error_rad":rerr,"limit_rad":ROOT_ERR_MAX,"by_hand":rooterr}
  if not all(np.all(np.isfinite(x)) for x in [d.qpos,d.qvel,d.qacc,d.ctrl,d.actuator_force]) and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"NONFINITE_STATE_OR_CONTROL"}
  newwarn=np.asarray(d.warning.number,dtype=int)-warn0
  if np.any(newwarn>0) and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"MUJOCO_WARNING","increments":newwarn.tolist()}
  for a,n in enumerate(act_names):
   force=abs(float(d.actuator_force[a]));metrics["max_abs_actuator_force"][n]=max(force,metrics["max_abs_actuator_force"].get(n,0.0))
   if bool(m.actuator_forcelimited[a]):
    lim=max(abs(float(m.actuator_forcerange[a,0])),abs(float(m.actuator_forcerange[a,1])))
    if force>lim and first_failure is None:first_failure={"step":step,"stage":stage,"gate":"ACTUATOR_FORCE_LIMIT","actuator":n,"force":force,"limit":lim}
  qflat=[];vflat=[]
  for j in range(m.njnt):
   jt=int(m.jnt_type[j]);nq=7 if jt==int(mujoco.mjtJoint.mjJNT_FREE) else (4 if jt==int(mujoco.mjtJoint.mjJNT_BALL) else 1);nv=6 if jt==int(mujoco.mjtJoint.mjJNT_FREE) else (3 if jt==int(mujoco.mjtJoint.mjJNT_BALL) else 1);qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j]);qflat.extend(d.qpos[qa:qa+nq]);vflat.extend(d.qvel[va:va+nv])
  sw.writerow([step,float(d.time),stage]+qflat+vflat+list(d.ctrl)+list(d.actuator_force))
  contactf.write(json.dumps({"step":step,"time_s":float(d.time),"stage":stage,"contacts":cs,"table_support_on_bottle_z_N":fz,"table_contact_count":table_count,"hand_support_on_bottle_world_N":hand.tolist(),"bottle":bs,"neck_contact_min_m":nmin,"joint_equalities_rad":je,"connect_equalities_m":ce,"first_failure":first_failure},allow_nan=False)+"\n")
  bw.writerow([step,float(d.time),stage,*bs["com_world_m"],bs["linear_speed_m_s"],bs["angular_speed_rad_s"],table_count,fz,float(hand[2]),"" if nmin is None else nmin,jmax,cmax,rerr,worst["ratio"]])
  if draw and step>0 and step%FRAME_EVERY==0:vgen.send(frame(stage))
  if step>0 and step%2500==0:print(f"step={step} t={d.time:.3f} stage={stage} bottle_v={bs['linear_speed_m_s']:.4f} bottle_w={bs['angular_speed_rad_s']:.4f} tableFz={fz:.3f} rooterr={rerr:.4f} mimic={jmax:.2g}",flush=True)
  return {"contacts":cs,"bottle":bs,"table_force_z":fz,"table_count":table_count}
 def apply():
  for a in motors:
   j=int(m.actuator_trnid[a,0]);kp,kd=gains[a];q=float(d.qpos[int(m.jnt_qposadr[j])]);v=float(d.qvel[int(m.jnt_dofadr[j])]);lo,hi=map(float,m.actuator_ctrlrange[a]);d.ctrl[a]=float(np.clip(kp*(qtarget[j]-q)-kd*v,lo,hi))
  for side,a in servo.items():d.ctrl[a]=float(claw[side])
 def step(stage):
  nonlocal physics_steps
  if physics_steps>=MAX_STEPS:
   if first_failure is None:first_failure={"step":physics_steps,"stage":stage,"gate":"STEP_BUDGET_EXHAUSTED"}
   return None
  apply();mujoco.mj_step(m,d);physics_steps+=1
  return sample(physics_steps,stage)
 def hold(stage,n):
  for _ in range(n):
   if step(stage) is None or first_failure is not None:return False
  return True
 def ramp_one(stage,side,a,b,n):
  for s in range(1,n+1):
   u=s/n;x=10*u**3-15*u**4+6*u**5;claw[side]=a+(b-a)*x
   if step(stage) is None or first_failure is not None:return False
  return True
 def ramp_dual(stage,a,b,n):
  for s in range(1,n+1):
   u=s/n;x=10*u**3-15*u**4+6*u**5;claw["left"]=claw["right"]=a+(b-a)*x
   if step(stage) is None or first_failure is not None:return False
  return True
 physics_steps=0;warning0=np.asarray(d.warning.number,dtype=int).copy();initial_com=np.asarray(d.xipos[bottle],dtype=float).copy();mass=float(m.body_mass[bottle]);weight=mass*9.81
 if abs(mass-.570)>1e-6:raise SystemExit(f"Canonical bottle mass mismatch {mass}")
 q2=float(r2["right_claw_joint_rad"]);q100=float(r100["right_claw_joint_rad"]);claw={"left":q2,"right":q2}
 flags=deque(maxlen=WINDOW);window=None;first_failure=None
 phases={"initial_settle":"NOT RUN","left_gripper_dynamic":"NOT RUN","right_gripper_dynamic":"NOT RUN","dual_gripper_dynamic":"NOT RUN"}
 metrics={"max_bottle_com_rise_m":0.0,"max_bottle_linear_speed_m_s":0.0,"max_bottle_angular_speed_rad_s":0.0,"max_joint_eq_residual_rad":0.0,"max_connect_eq_residual_m":0.0,"max_root_tracking_error_rad":0.0,"max_abs_joint_qvel_rad_s":{},"max_abs_actuator_force":{},"max_source_velocity_ratio":{"ratio":0.0,"joint":None,"qvel":0.0,"limit":None},"min_neck_contact_distance_m":None,"table_support_min_N":None,"table_support_max_N":None}
 # Record the initialized state; it is not a physics step and occurs before the sole episode starts.
 sample(0,"INITIAL",False)
 for local in range(1,SETTLE_MAX+1):
  rec=step("INITIAL_SETTLE")
  if rec is None or first_failure is not None:break
  bs=rec["bottle"];fz=rec["table_force_z"];table_ok=rec["table_count"]>0 and SUPPORT_MIN*weight<=fz<=SUPPORT_MAX*weight
  flags.append(bool(table_ok and bs["linear_speed_m_s"]<=LIN_MAX and bs["angular_speed_rad_s"]<=ANG_MAX))
  if len(flags)==WINDOW and all(flags):
   s0=physics_steps-WINDOW+1;window={"start_step":s0,"end_step":physics_steps,"start_time_s":(s0-1)*DT,"end_time_s":physics_steps*DT,"consecutive_samples":WINDOW,"table_contact_continuous":True,"linear_speed_threshold_m_s":LIN_MAX,"angular_speed_threshold_rad_s":ANG_MAX,"support_force_window_N":[SUPPORT_MIN*weight,SUPPORT_MAX*weight]};break
 if first_failure is None and window is not None:
  phases["initial_settle"]="PASS";print("INITIAL_SETTLE PASS",window,flush=True)
 elif first_failure is None:
  phases["initial_settle"]="FAIL";first_failure={"step":physics_steps,"stage":"INITIAL_SETTLE","gate":"NO_1000_CONSECUTIVE_STABLE_TABLE_SUPPORTED_SAMPLES_WITHIN_2000_STEPS","criteria":{"continuous_table_contact":True,"support_force_N":[SUPPORT_MIN*weight,SUPPORT_MAX*weight],"linear_speed_max_m_s":LIN_MAX,"angular_speed_max_rad_s":ANG_MAX}};print("INITIAL_SETTLE FAIL; no gripper motion started",flush=True)
 else:
  phases["initial_settle"]="FAIL"
 if phases["initial_settle"]=="PASS":
  for side,lab in [("left","LEFT_GRIPPER"),("right","RIGHT_GRIPPER")]:
   ok=ramp_one(lab+"_OPEN",side,q2,q100,OPEN_STEPS) and hold(lab+"_OPEN_HOLD",HOLD_STEPS)
   if ok:ok=ramp_one(lab+"_CLOSE",side,q100,q2,CLOSE_STEPS) and hold(lab+"_CLOSED_HOLD",HOLD_STEPS)
   phases["left_gripper_dynamic" if side=="left" else "right_gripper_dynamic"]="PASS" if ok else "FAIL"
   if not ok:break
  if first_failure is None and phases["left_gripper_dynamic"]=="PASS" and phases["right_gripper_dynamic"]=="PASS":
   ok=ramp_dual("DUAL_GRIPPER_OPEN",q2,q100,OPEN_STEPS) and hold("DUAL_GRIPPER_OPEN_HOLD",HOLD_STEPS)
   if ok:ok=ramp_dual("DUAL_GRIPPER_CLOSE",q100,q2,CLOSE_STEPS) and hold("DUAL_GRIPPER_FINAL_HOLD",FINAL_HOLD)
   phases["dual_gripper_dynamic"]="PASS" if ok else "FAIL"
 # Close artifacts even after a hard gate.
 try:vgen.send(frame("FINAL"));vgen.close()
 finally:
  try:renderer.close()
  except Exception:pass
  for f in [statef,contactf,bottlef]:
   try:f.close()
   except Exception:pass
 final_bottle=object_state(m,d,bottle);final_contacts=collect_contacts(m,d);final_je,final_ce=equalities(m,d)
 report={"schema":"robotsim.issue46.full_x2_dual_omnipicker_stage2_result.v1","attempt_id":"a57b153e-aa31-43f0-a598-4a02457c17b1","run_status":"FAIL" if first_failure else "PASS","phase_results":phases,"first_failure":first_failure,"physics_steps":physics_steps,"physics_time_s":float(d.time),"max_steps":MAX_STEPS,"initial_settle_window":window,"full_body_scope":"fixed-base diagnostic; no balance/locomotion claim","option_d":"D_RIGHT_EXT_COLLISION_DIAGNOSTIC","bottle_mass_kg":mass,"bottle_weight_N":weight,"final_bottle_state":final_bottle,"final_contacts":final_contacts,"final_joint_equality_residual_rad":final_je,"final_connect_equality_residual_m":final_ce,"metrics":metrics,"video":{"path":str(video),"windows_unc":"\\\\wsl.localhost\\Ubuntu-22.04\\home\\lzy18001500226\\projects\\RobotSim\\reviews\\issue-46\\full-x2-omnipicker-physical-grasp-20261024T181317Z\\stage2_run01\\full_x2_dual_omnipicker_stage2_actual_mujoco.mp4","exists":video.is_file(),"size_bytes":video.stat().st_size if video.is_file() else 0,"fps":VIDEO_FPS,"sample_every_steps":FRAME_EVERY,"render":"full X2/table/bottle left pane; active gripper close-up right pane; sampled from post-mj_step state"},"safety":{"bottle_qpos_writes_after_first_step":0,"bottle_equality_or_weld":False,"mocap_carry":False,"hidden_support":False,"follower_actuators":followers,"collision_suppression":False,"robot_root_fixed":True,"motor_ctrlrange_unchanged":True,"gripper_native_actuator_and_force_cap_unchanged":True},"step_budget":{"this_episode":physics_steps,"stage2_max":MAX_STEPS,"stage3_5_max":50000,"combined_max":80000,"retries":0},"trace_files":["joint_actuator_trace.csv.gz","contact_physics_trace.jsonl.gz","bottle_support_trace.csv.gz"],"reproduction_command":"cd /home/lzy18001500226/robotsim-issue46-full-x2-dual-omnipicker-20261024 && source /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/activate && MUJOCO_GL=egl python scripts/research/issue46_full_x2_dual_omnipicker_episode.py","human_visual_review":"PENDING"}
 (RUN_OUT/"stage2_result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 (RUN_OUT/"STAGE2_REPORT.md").write_text("# Full-X2 Dual OmniPicker Stage 2\n\nStatus: **"+report["run_status"]+"**\n\nPhysics steps: "+str(physics_steps)+"\n\nFirst failure: `"+str(first_failure)+"`\n\nActual MuJoCo MP4: `"+str(video)+"`\n\nThis is fixed-base dynamics only. The video uses post-`mj_step` frames.\n")
 print(json.dumps({"run_status":report["run_status"],"phase_results":phases,"physics_steps":physics_steps,"first_failure":first_failure,"settle_window":window,"video":report["video"],"metrics":metrics},indent=2,allow_nan=False))

if __name__=="__main__":main()
