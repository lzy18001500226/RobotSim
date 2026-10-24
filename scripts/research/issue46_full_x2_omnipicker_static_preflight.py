#!/usr/bin/env python3
"""Static full-X2 pad-center IK and sampled collision preflight; no physics steps."""
import csv, json, hashlib, math
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SCENE = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/official-wrist-variant-static-audit-20261024T162630Z/verified/full_x2_d_right_ext_collision_diagnostic.xml")
FIXTURE = Path("/home/lzy18001500226/robotsim-issue46-canonical-model-integration-20261024/simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/tests/fixtures/pr90_static_samples.csv")
OUT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/full-x2-omnipicker-physical-grasp-20261024T181317Z")
ARM_NAMES = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(m, obj, i): return mujoco.mj_id2name(m,obj,int(i)) or f"unnamed_{i}"
def set_gripper_row(m,d,row):
    root=float(row["right_claw_joint_rad"]);n2=float(row["narrow2_rad"]);n3=float(row["narrow3_rad"]);nl=float(row["narrow_loop_rad"])
    for side,prefix in (("right","R"),("left","L")):
        vals={side+"_claw_joint":root,prefix+"_hand_wide1_joint":-root,prefix+"_hand_narrow2_joint":n2,prefix+"_hand_wide2_joint":-n2,prefix+"_hand_narrow3_joint":n3,prefix+"_hand_wide3_joint":-n3,prefix+"_hand_narrow_loop_joint":nl,prefix+"_hand_wide_loop_joint":-nl}
        for jn,q in vals.items():
            jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,jn);d.qpos[m.jnt_qposadr[jid]]=q

def contact_rows(m,d):
    rows=[]
    for k in range(d.ncon):
        c=d.contact[k]; g1=int(c.geom1);g2=int(c.geom2)
        b1=int(m.geom_bodyid[g1]);b2=int(m.geom_bodyid[g2])
        n1=name(m,mujoco.mjtObj.mjOBJ_BODY,b1);n2=name(m,mujoco.mjtObj.mjOBJ_BODY,b2)
        pair={n1,n2}
        allowed=("head_pitch_link" in pair and "torso_link" in pair) or ("canonical_bottle" in pair and "m0_table" in pair) or ("floor" in pair and ("left_ankle" in n1+n2 or "right_ankle" in n1+n2))
        rows.append({"geom1":name(m,mujoco.mjtObj.mjOBJ_GEOM,g1),"geom2":name(m,mujoco.mjtObj.mjOBJ_GEOM,g2),"body1":n1,"body2":n2,"distance_m":float(c.dist),"allowed_baseline":allowed})
    return rows

def min_env_distance(m,d,robot_geoms,env_geoms):
    best={"distance_m":1.0,"geom1":None,"geom2":None,"fromto":None}
    for a in robot_geoms:
        for b in env_geoms:
            ft=np.zeros(6,dtype=np.float64)
            dist=float(mujoco.mj_geomDistance(m,d,int(a),int(b),1.0,ft))
            if dist<best["distance_m"]:
                best={"distance_m":dist,"geom1":name(m,mujoco.mjtObj.mjOBJ_GEOM,a),"geom2":name(m,mujoco.mjtObj.mjOBJ_GEOM,b),"fromto":ft.tolist()}
    return best

m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
rows=list(csv.DictReader(FIXTURE.open(newline="")));q100=rows[-1]
set_gripper_row(m,d,q100)
# Option D full assembly base pose for the grasp approach: fixed diagnostic pelvis is shifted 30 mm toward the workcell.
pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
base_pos_original=m.body_pos[pelvis].copy();m.body_pos[pelvis,1]-=0.030
mujoco.mj_forward(m,d)
base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link")
Rwb=d.xmat[base].reshape(3,3);pwb=d.xpos[base].copy()
narrow=[i for i in range(m.ngeom) if (name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith("R_hand_narrow3_Link__collision"))]
wide=[i for i in range(m.ngeom) if (name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith("R_hand_wide3_Link__collision"))]
best=(math.inf,None,None,None)
for a in narrow:
    for b in wide:
        ft=np.zeros(6,dtype=np.float64);dist=float(mujoco.mj_geomDistance(m,d,a,b,1.0,ft))
        if dist<best[0]:best=(dist,a,b,ft.copy())
span,ga,gb,ft=best
pad_local=Rwb.T@((ft[:3]+ft[3:])/2-pwb)
gap_local=Rwb.T@(ft[3:]-ft[:3]);gap_local/=np.linalg.norm(gap_local)
finger_local=pad_local/np.linalg.norm(pad_local)
palm_local=np.cross(gap_local,finger_local);palm_local/=np.linalg.norm(palm_local)
R_base_pad=np.column_stack((palm_local,gap_local,finger_local))
# Right-hand horizontal side pinch: palm axis +Z, jaw gap axis -X, finger insertion axis -Y.
R_pad_target=np.column_stack((np.array([0.,0.,1.]),np.array([-1.,0.,0.]),np.array([0.,-1.,0.])))
center=np.array([0.300,0.000,0.915])
# Correct target is the measured pad center; PREGRASP is above and in front, approach descends then inserts.
waypoints=[("PREGRASP",np.array([0.300,0.200,1.015])),("FRONT_ALIGN",np.array([0.300,0.200,0.915])),("APPROACH",center.copy())]
qadr=[];lo=[];hi=[];jids=[]
for jn in ARM_NAMES:
    jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,jn);jids.append(jid);qadr.append(int(m.jnt_qposadr[jid]));lo.append(float(m.jnt_range[jid,0]));hi.append(float(m.jnt_range[jid,1]))
qadr=np.asarray(qadr);lo=np.asarray(lo);hi=np.asarray(hi);margin=0.020;lb=lo+margin;ub=hi-margin
# Disable contact generation only inside the numerical IK solve; restore immediately for collision qualification.
ct=m.geom_contype.copy();ca=m.geom_conaffinity.copy();m.geom_contype[:]=0;m.geom_conaffinity[:]=0

def solve_pad(target,seed):
    R_base_target=R_pad_target@R_base_pad.T
    def residual(q):
        d.qpos[qadr]=q;mujoco.mj_forward(m,d)
        Rb=d.xmat[base].reshape(3,3);p=d.xpos[base]+Rb@pad_local;Ra=Rb@R_base_pad
        return np.r_[(p-target)*100.,(Rotation.from_matrix(R_base_target).inv()*Rotation.from_matrix(Ra)).as_rotvec()]
    ans=least_squares(residual,np.clip(seed,lb+1e-8,ub-1e-8),bounds=(lb,ub),max_nfev=240,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    v=residual(ans.x)
    return ans.x,float(np.linalg.norm(v[:3])/100),float(np.linalg.norm(v[3:])),float(np.min(np.minimum(ans.x-lo,hi-ans.x))),int(ans.nfev)

seed=np.array([-0.40,-0.50,0.80,-1.60,-1.00,0.20,0.00])
solved=[];last=seed
# 10 mm Cartesian subdivisions, exact target rotation and measured pad center objective.
path_points=[]
prev=None
for stage,target in waypoints:
    if prev is None: samples=[target]
    else:
        n=max(1,int(math.ceil(np.linalg.norm(target-prev)/0.010)))
        samples=[prev+(target-prev)*(i/n) for i in range(1,n+1)]
    for idx,pad_target in enumerate(samples):
        last,pe,ae,jmargin,nfev=solve_pad(pad_target,last)
        rec={"stage":stage,"sample_index":idx,"pad_target_world_m":pad_target.tolist(),"arm_q_rad":last.tolist(),"position_error_m":pe,"orientation_error_rad":ae,"min_source_joint_margin_rad":jmargin,"ik_nfev":nfev}
        solved.append(rec);path_points.append((pad_target.copy(),last.copy(),stage))
    prev=target.copy()
# Restore collision masks and check the complete home-to-pregrasp joint segment plus Cartesian path.
m.geom_contype[:]=ct;m.geom_conaffinity[:]=ca
# Begin at the model's original arm configuration, preserving only the qualified 100% gripper initialization.
d.qpos[:]=m.qpos0;set_gripper_row(m,d,q100);mujoco.mj_forward(m,d)
qhome=d.qpos[qadr].copy()
# Use a bounded 100-sample joint-space HOME->PREGRASP witness, then the 10 mm Cartesian IK path.
pre_q=solved[0]["arm_q_rad"]
path_states=[]
for i in range(101):path_states.append((qhome+(pre_q-qhome)*i/100.0,"HOME_TO_PREGRASP",i,None))
for i,(pad_target,q,stage) in enumerate(path_points):path_states.append((q,stage,i,pad_target))
robot_geoms=[i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and not name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith(("bottle_","m0_table_","floor"))]
table_geoms=[i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith("m0_table_") and (m.geom_contype[i] or m.geom_conaffinity[i])]
bottle_geoms=[i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith("bottle_") and (m.geom_contype[i] or m.geom_conaffinity[i])]
path_report=[];first_bad=None;min_table={"distance_m":1.0};min_bottle={"distance_m":1.0};min_q_margin=math.inf
for idx,(q,stage,k,pad_target) in enumerate(path_states):
    d.qpos[qadr]=q;mujoco.mj_forward(m,d)
    contacts=contact_rows(m,d);unexpected=[c for c in contacts if not c["allowed_baseline"]]
    qmargin=float(np.min(np.minimum(q-lo,hi-q)));min_q_margin=min(min_q_margin,qmargin)
    dc_table=min_env_distance(m,d,robot_geoms,table_geoms)
    dc_bottle=min_env_distance(m,d,robot_geoms,bottle_geoms)
    if dc_table["distance_m"]<min_table["distance_m"]:min_table=dc_table.copy();min_table["path_index"]=idx;min_table["stage"]=stage
    if dc_bottle["distance_m"]<min_bottle["distance_m"]:min_bottle=dc_bottle.copy();min_bottle["path_index"]=idx;min_bottle["stage"]=stage
    # Bottle proximity is intentionally allowed only at final APPROACH with fully open jaws; static no-contact is still required.
    rec={"path_index":idx,"stage":stage,"waypoint_index":k,"target_pad_center_world_m":None if pad_target is None else pad_target.tolist(),"arm_q_rad":q.tolist(),"source_joint_margin_rad":qmargin,"contacts":contacts,"unexpected_contacts":unexpected,"min_table_distance":dc_table,"min_bottle_distance":dc_bottle}
    path_report.append(rec)
    if unexpected and first_bad is None:first_bad={"path_index":idx,"stage":stage,"contacts":unexpected}

# Re-evaluate only final open approach and 2% closing endpoint as a static geometry/contact preflight, not dynamics.
final_q=solved[-1]["arm_q_rad"]
def set_row_all(row):set_gripper_row(m,d,row);d.qvel[:]=0
endpoints={}
for label,row in (("open100",rows[-1]),("closed2",rows[2])):
    d.qpos[:]=m.qpos0;m.body_pos[pelvis,1]=base_pos_original[1]-0.030;set_row_all(row);d.qpos[qadr]=final_q;mujoco.mj_forward(m,d)
    c=contact_rows(m,d)
    endpoints[label]={"contacts":c,"unexpected_robot_contacts":[x for x in c if not x["allowed_baseline"]],"qpos_arm":list(final_q)}

# Static closing sweep with the compiled complete geometry; this writes qpos only in this zero-step diagnostic.
close_sweep=[];first_contact=None;first_nonpad=None;first_bilateral=None
left_open=rows[-1]
for row in reversed(rows[2:]):
    d.qpos[:]=m.qpos0;m.body_pos[pelvis,1]=base_pos_original[1]-0.030
    set_gripper_row(m,d,left_open)
    root=float(row["right_claw_joint_rad"]);n2=float(row["narrow2_rad"]);n3=float(row["narrow3_rad"]);nl=float(row["narrow_loop_rad"])
    vals={"right_claw_joint":root,"R_hand_wide1_joint":-root,"R_hand_narrow2_joint":n2,"R_hand_wide2_joint":-n2,"R_hand_narrow3_joint":n3,"R_hand_wide3_joint":-n3,"R_hand_narrow_loop_joint":nl,"R_hand_wide_loop_joint":-nl}
    for jn,q in vals.items():jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,jn);d.qpos[m.jnt_qposadr[jid]]=q
    d.qpos[qadr]=final_q;mujoco.mj_forward(m,d)
    contacts=contact_rows(m,d);bp=[x for x in contacts if "canonical_bottle" in (x["body1"],x["body2"]) and (x["body1"].startswith("R_hand_") or x["body2"].startswith("R_hand_"))]
    bodies={x["body1"] if x["body1"]!="canonical_bottle" else x["body2"] for x in bp}
    pad_n="R_hand_narrow3_Link" in bodies;pad_w="R_hand_wide3_Link" in bodies
    nonpad=[x for x in bp if not ((x["body1"]=="R_hand_narrow3_Link" and x["body2"]=="canonical_bottle") or (x["body2"]=="R_hand_narrow3_Link" and x["body1"]=="canonical_bottle") or (x["body1"]=="R_hand_wide3_Link" and x["body2"]=="canonical_bottle") or (x["body2"]=="R_hand_wide3_Link" and x["body1"]=="canonical_bottle"))]
    pc={"command_percent":float(row["command_percent"]),"root_q_rad":root,"bottle_contacts":bp,"nonpad_bottle_contacts":nonpad,"narrow3_contact":pad_n,"wide3_contact":pad_w,"all_contacts":contacts}
    close_sweep.append(pc)
    if bp and first_contact is None:first_contact=pc
    if nonpad and first_nonpad is None:first_nonpad=pc
    if pad_n and pad_w and first_bilateral is None:first_bilateral=pc
out={"schema":"robotsim.issue46.full_x2_omnipicker.correct_pad_center_static_preflight.v2","mujoco_version":mujoco.__version__,"physics_steps":0,"mj_forward_calls_counted":False,"mj_forward_calls_note":"least-squares internal mj_forward calls are not instrumented; all operations are static and no mj_step occurs","scene_path":str(SCENE),"scene_sha256":sha(SCENE),"static_fixture_path":str(FIXTURE),"fixture_sha256":sha(FIXTURE),"pelvis_world_pos_original_m":base_pos_original.tolist(),"pelvis_world_pos_candidate_m":[float(base_pos_original[0]),float(base_pos_original[1]-0.030),float(base_pos_original[2])],"wrist_option":"D_RIGHT_EXT_COLLISION_DIAGNOSTIC","right_pad_center_local_from_q100_m":pad_local.tolist(),"pad_gap_axis_local":gap_local.tolist(),"finger_insertion_axis_local":finger_local.tolist(),"pad_frame_local_columns_palm_gap_finger":R_base_pad.tolist(),"nominal_target_pad_frame_world_columns_palm_gap_finger":R_pad_target.tolist(),"target_bottle_body_center_world_m":center.tolist(),"target_grasp_band_z_m":float(center[2]),"approach_waypoints":[{"name":n,"pad_center_world_m":p.tolist()} for n,p in waypoints],"ik_solutions":solved,"path_sample_count":len(path_report),"first_unexpected_contact":first_bad,"min_table_clearance":min_table,"min_robot_bottle_clearance":min_bottle,"min_source_arm_joint_margin_rad":min_q_margin,"endpoint_states":endpoints,"closing_sweep":close_sweep,"first_bottle_contact":first_contact,"first_nonpad_linkage_contact":first_nonpad,"first_bilateral_pad_contact":first_bilateral,"distance_note":"mj_geomDistance was used for pairwise proximity; exact-zero legacy-CCD distances are not treated as positive-clearance certification. Active contact identities and exact source STL results are reported separately.","collision_masks_restored":True}
(OUT/"static_pad_center_preflight_v2.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
with (OUT/"static_pad_center_path_v2.csv").open("w",newline="") as f:
    import csv as _csv
    w=_csv.DictWriter(f,fieldnames=["path_index","stage","waypoint_index","x_pad_m","y_pad_m","z_pad_m","joint_margin_rad","table_clearance_m","table_geom1","table_geom2","bottle_clearance_m","bottle_geom1","bottle_geom2","unexpected_contact_count"]);w.writeheader()
    for x,(q,stage,k,pad_target) in zip(path_report,path_states):
        w.writerow({"path_index":x["path_index"],"stage":stage,"waypoint_index":k,"x_pad_m":"" if pad_target is None else float(pad_target[0]),"y_pad_m":"" if pad_target is None else float(pad_target[1]),"z_pad_m":"" if pad_target is None else float(pad_target[2]),"joint_margin_rad":x["source_joint_margin_rad"],"table_clearance_m":x["min_table_distance"]["distance_m"],"table_geom1":x["min_table_distance"]["geom1"],"table_geom2":x["min_table_distance"]["geom2"],"bottle_clearance_m":x["min_bottle_distance"]["distance_m"],"bottle_geom1":x["min_bottle_distance"]["geom1"],"bottle_geom2":x["min_bottle_distance"]["geom2"],"unexpected_contact_count":len(x["unexpected_contacts"])})
print(json.dumps({"physics_steps":0,"samples":len(path_report),"first_unexpected_contact":first_bad,"min_table_clearance":min_table,"min_robot_bottle_clearance":min_bottle,"min_joint_margin":min_q_margin,"endpoints":endpoints},indent=2))
