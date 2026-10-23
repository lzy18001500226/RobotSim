#!/usr/bin/env python3
"""Bounded static IK and contact-band screen for PR90 in full X2; no physics steps."""
import argparse,csv,hashlib,json,math,subprocess,sys,time,uuid,xml.etree.ElementTree as ET,io
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/research"))
import issue46_manipulation_integration.audit_pr90_full_x2_static as base
pg=base.pg
GeometryPair=base.GeometryPair
REV=base.REV
PR90_HEAD="facdaa0d6e5f8152168d2e812d864b660eac9f96"
PR90_MODEL_SHA="32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45"
OLD_MODEL_SHA="0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01"
PR90_WT=base.WT
OLD=base.OLD
NEW=base.NEW
OLDXML=base.OX
OLDRES=base.OR
OLDCSV=base.OC
SIDE=base.SIDE
NEWXML=base.NX
RANGE=base.NR
def sha(p):return base.sha(Path(p))
def git(p,*a):return base.git(p,*a)
def objname(m,t,i):return base.nm(m,t,i)
def body(m,i):return base.body(m,i)
def geom(m,i):return base.geom(m,i)
def jid(m,n):return base.jid(m,n)
def gid(m,n):return base.gid(m,n)
def excluded(root):return base.excl(root)

def dls_pose(model,data,arm_q,hand_q,object_q,arm_qpos,obj,body_id,friction,base_id,tip_pos,target_pos,target_R,dofs,limits,max_iter=80):
    q=np.asarray(arm_q,dtype=float).copy()
    for iteration in range(max_iter):
        base.state(model,data,tuple(q),hand_q,object_q,arm_qpos,obj,body_id,friction)
        current_pos=np.asarray(data.xpos[base_id]).copy()
        current_R=np.asarray(data.xmat[base_id]).reshape(3,3).copy()
        pos_err=np.asarray(target_pos)-current_pos
        rot_err=Rotation.from_matrix(np.asarray(target_R)@current_R.T).as_rotvec()
        pos_norm=float(np.linalg.norm(pos_err));rot_norm=float(np.linalg.norm(rot_err))
        if pos_norm<=0.0005 and rot_norm<=math.radians(.25):
            return q.tolist(),{"converged":True,"iterations":iteration,"position_error_mm":pos_norm*1000,
                               "orientation_error_deg":math.degrees(rot_norm),"joint_limits_pass":True}
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,jr,current_pos,base_id)
        J=np.vstack((jp[:,dofs],0.2*jr[:,dofs]))
        e=np.concatenate((pos_err,0.2*rot_err))
        try: dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),e)
        except np.linalg.LinAlgError: return None,{"converged":False,"reason":"SINGULAR_JACOBIAN","iterations":iteration}
        mag=float(np.linalg.norm(dq))
        if mag>.06:dq*=.06/mag
        accepted=False
        scale=1.0
        for _ in range(8):
            candidate=q+scale*dq
            if all(limits[j][0] <= v <= limits[j][1] for j,v in zip(base.ARM,candidate)):
                q=candidate;accepted=True;break
            scale*=.5
        if not accepted:return None,{"converged":False,"reason":"SOURCE_LIMIT_BLOCK","iterations":iteration,
                                    "position_error_mm":pos_norm*1000,"orientation_error_deg":math.degrees(rot_norm)}
    return None,{"converged":False,"reason":"IK_ITERATION_LIMIT","iterations":max_iter,
                 "position_error_mm":pos_norm*1000,"orientation_error_deg":math.degrees(rot_norm)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-parent",default=str(REV/"issue-46"))
    args=parser.parse_args()
    parent=pg.validate_output_path(args.output_parent,authorized_roots=(str(REV),),require_exists=True,
                                   expected_kind="directory",environment="wsl")
    original_forward=mujoco.mj_forward;original_steps=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2)
    forward_count=0;step_count=0
    def forward(*a,**k):
        nonlocal forward_count
        forward_count+=1;return original_forward(*a,**k)
    def forbidden(*a,**k):
        nonlocal step_count
        step_count+=1;raise RuntimeError("mj_step prohibited in static candidate screen")
    mujoco.mj_forward=forward;mujoco.mj_step=forbidden;mujoco.mj_step1=forbidden;mujoco.mj_step2=forbidden
    try:
        if git(PR90_WT,"rev-parse","HEAD")!=PR90_HEAD or git(PR90_WT,"status","--porcelain"):
            raise RuntimeError("PR90 exact static input is not a clean worktree")
        if sha(NEWXML)!=PR90_MODEL_SHA or sha(OLDXML)!=OLD_MODEL_SHA:
            raise RuntimeError("frozen source model hash mismatch")
        pr90_manifest=json.loads((NEW/"artifact_paths.json").read_text())
        old_manifest=json.loads((OLD/"artifact_paths.json").read_text())
        pg.verify_artifacts(pr90_manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
        pg.verify_artifacts(old_manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
        src=json.loads(RANGE.read_text());profile=base.profile()
        oldres=json.loads(OLDRES.read_text())
        config=next(x for x in oldres["candidate_configurations"] if x["id"]=="D70_H230_M550G_MID")
        with OLDCSV.open(newline="",encoding="utf-8") as stream:
            oldrows=[x for x in csv.DictReader(stream) if x["config"]==config["id"]]
        oldrows.sort(key=lambda x:int(x["sample"]))
        root=ET.fromstring(OLDXML.read_text());xml=base.make_overlay_xml(OLDXML.read_text(),src);newroot=ET.fromstring(xml)
        if excluded(root)!=excluded(newroot):raise RuntimeError("collision exclusions changed")
        oldm=mujoco.MjModel.from_xml_string(OLDXML.read_text());model=mujoco.MjModel.from_xml_string(xml)
        counts={k:getattr(model,k) for k in ("nbody","njnt","ngeom","nmesh","neq","nu","nq","nv")}
        if counts!={k:getattr(oldm,k) for k in counts}:raise RuntimeError("compiled full-X2 topology changed")
        for a in ("geom_contype","geom_conaffinity","geom_bodyid","geom_dataid","geom_pos","geom_quat",
                  "body_pos","body_quat","jnt_range","jnt_axis","jnt_pos","actuator_trnid"):
            if not np.array_equal(getattr(oldm,a),getattr(model,a)):raise RuntimeError("overlay changed "+a)
        data=mujoco.MjData(model);obj=gid(model,base.OBJ);objbody=int(model.geom_bodyid[obj])
        objq=int(model.jnt_qposadr[jid(model,base.OJ)]);table=gid(model,base.TABLE)
        radius=base.D/2;height=base.H;mass=base.MASS
        model.geom_size[obj,:2]=[radius,height/2];model.body_mass[objbody]=mass
        model.body_inertia[objbody]=[mass*(3*radius**2+height**2)/12]*2+[.5*mass*radius**2]
        friction=np.asarray(model.geom_friction[obj]).copy()
        arm_qpos=tuple(int(model.jnt_qposadr[jid(model,j)]) for j in base.ARM)
        arm_dofs=tuple(int(model.jnt_dofadr[jid(model,j)]) for j in base.ARM)
        limits={j:tuple(map(float,model.jnt_range[jid(model,j)])) for j in base.ARM}
        hand_geoms=[g for g in range(model.ngeom) if body(model,int(model.geom_bodyid[g])).startswith(("R_hand_","R_omnipicker"))
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
        groups={"narrow_pad":[g for g in hand_geoms if body(model,int(model.geom_bodyid[g]))=="R_hand_narrow3_Link"],
                "wide_pad":[g for g in hand_geoms if body(model,int(model.geom_bodyid[g]))=="R_hand_wide3_Link"],
                "linkage_loop":[g for g in hand_geoms if "loop_Link" in body(model,int(model.geom_bodyid[g]))]}
        groups["other_gripper"]=[g for g in hand_geoms if all(g not in z for z in groups.values())]
        robot=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) not in (0,objbody)
               and not body(model,int(model.geom_bodyid[g])).startswith("m0_table")
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
        moving=[g for g in robot if body(model,int(model.geom_bodyid[g])).startswith(
            ("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker"))]
        scene=[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==0
               or body(model,int(model.geom_bodyid[g])).startswith("m0_table")]
        exc=excluded(newroot);pairset=set()
        for g in robot:
            if base.coll(model,g,obj,exc):pairset.add(tuple(sorted((geom(model,g),geom(model,obj)))))
        for g in moving:
            for h in scene+robot:
                if base.coll(model,g,h,exc):pairset.add(tuple(sorted((geom(model,g),geom(model,h)))))
        pairs=tuple(sorted(pairset));geom_ids={geom(model,g):g for g in range(model.ngeom)}
        identity="full-x2-pr90-static-variants:"+hashlib.sha256(xml.encode()).hexdigest()
        now=time.monotonic_ns()
        basepose=base.PoseSample("test_cylinder","world",base.CENTER,(0,0,0,1),0,base.PoseSource.SIM_GROUND_TRUTH,
                                 base.TimeDomain.SIMULATION,model_identity=identity,received_steady_ns=now)
        # Ground-truth pose and source-valid arm states from the original accepted static path.
        with OLDCSV.open(newline="",encoding="utf-8") as stream:
            qrows=[x for x in csv.DictReader(stream) if x["config"]==config["id"]]
        qrows.sort(key=lambda x:int(x["sample"]))
        qpath=[tuple(map(float,json.loads(x["q_arm_json"]))) for x in qrows]
        q_start=qpath[0];q_goal=qpath[-1];open_state=profile[-1]["q"]
        base_id=base.id_by_name(model,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link")
        def set_arm(q):
            base.state(model,data,tuple(q),open_state,objq,arm_qpos,obj,objbody,friction)
        set_arm(q_goal)
        p_goal=np.asarray(data.xpos[base_id]).copy()
        R_goal=np.asarray(data.xmat[base_id]).reshape(3,3).copy()
        candidates=[
            {"id":"HIGHER_HOVER_50MM","dz_m":0.0,"yaw_deg":0.0,"standoff_m":0.190},
            {"id":"CONTACT_BAND_UP_20MM","dz_m":0.020,"yaw_deg":0.0,"standoff_m":0.140},
            {"id":"CONTACT_BAND_DOWN_20MM","dz_m":-0.020,"yaw_deg":0.0,"standoff_m":0.140},
            {"id":"WRIST_YAW_PLUS_15DEG","dz_m":0.0,"yaw_deg":15.0,"standoff_m":0.140},
            {"id":"WRIST_YAW_MINUS_15DEG","dz_m":0.0,"yaw_deg":-15.0,"standoff_m":0.140},
        ]
        results=[];all_path=[];all_closure=[]
        for candidate in candidates:
            angle=math.radians(candidate["yaw_deg"])
            yaw=Rotation.from_euler("z",angle).as_matrix()
            Rtarget=yaw@R_goal
            ptarget=p_goal+np.array([0,0,candidate["dz_m"]])
            height=candidate["standoff_m"]
            q_seed=list(q_start);waypoints=[];ik_rows=[];failed=None
            for k in range(11):
                alpha=k/10.0
                target_pos=ptarget+np.array([0,0,height*(1-alpha)])
                q,ik=dls_pose(model,data,q_seed,open_state,objq,arm_qpos,obj,objbody,friction,base_id,
                              ptarget,target_pos,Rtarget,arm_dofs,limits)
                ik_rows.append({"index":k,"alpha":alpha,"target_position_world_m":target_pos.tolist(),
                                "ik":ik,"q_arm_rad":None if q is None else q})
                if q is None:
                    failed=ik;break
                q_seed=q;waypoints.append({"sample":k,"q":tuple(q)})
            if failed:
                results.append({"candidate":candidate,"IK":"BLOCKED_IK","failure":failed,
                                "path_status":"NOT_RUN","closure_status":"NOT_RUN","ik_timeline":ik_rows})
                continue
            path_metrics=[]
            for i,row in enumerate(waypoints):
                q=row["q"]
                base.state(model,data,q,open_state,objq,arm_qpos,obj,objbody,friction)
                distances=[]
                for a,b in pairs:
                    value=float(mujoco.mj_geomDistance(model,data,geom_ids[a],geom_ids[b],.5,None))
                    distances.append(GeometryPair(a,b,value,"collision" if value<=0 else "clearance"))
                near=min(distances,key=lambda z:z.signed_distance_m)
                active=[]
                for c in data.contact[:data.ncon]:
                    a,b=int(c.geom1),int(c.geom2)
                    active.append({"geom1":geom(model,a),"geom2":geom(model,b),
                                   "body1":body(model,int(model.geom_bodyid[a])),"body2":body(model,int(model.geom_bodyid[b])),
                                   "distance_mm":float(c.dist)*1000})
                obj_contact=[x for x in active if (x["geom1"]==base.OBJ or x["geom2"]==base.OBJ)
                             and not x["body1"].startswith("m0_table") and not x["body2"].startswith("m0_table")]
                self_contact=[x for x in active if x["geom1"]!=base.OBJ and x["geom2"]!=base.OBJ
                    and (x["body1"].startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker"))
                         or x["body2"].startswith(("right_shoulder","right_elbow","right_wrist","R_hand_","R_omnipicker")))]
                table_ids=[g for g in scene if body(model,int(model.geom_bodyid[g])).startswith("m0_table")
                           or geom(model,g).startswith("m0_table")]
                table_min=base.min_pair_distance(model,data,hand_geoms,table_ids,exc)
                path_metrics.append({"candidate":candidate["id"],"sample":i,"alpha":i/10.0,
                    "min_collision_clearance_mm":near.signed_distance_m*1000,
                    "nearest_pair":[near.geom_a,near.geom_b],
                    "gripper_table_clearance_mm":None if table_min is None else table_min["distance_m"]*1000,
                    "robot_object_contacts":obj_contact,"moving_self_contacts":self_contact})
            path_ok=all(x["min_collision_clearance_mm"]>=1.0 and not x["robot_object_contacts"] and not x["moving_self_contacts"]
                        for x in path_metrics)
            # Full 101-state PR90 closure at the candidate's IK endpoint.
            final_q=waypoints[-1]["q"];closure=[];maxD=0.0;maxSelf=0.0
            for idx,entry in enumerate(reversed(profile)):
                base.state(model,data,final_q,entry["q"],objq,arm_qpos,obj,objbody,friction)
                gaps=[]
                for side in ("narrow","wide"):
                    a=base.id_by_name(model,mujoco.mjtObj.mjOBJ_SITE,f"candidate_{side}_D_jaw")
                    b=base.id_by_name(model,mujoco.mjtObj.mjOBJ_SITE,f"candidate_{side}_D_loop")
                    gaps.extend((data.site_xpos[a]-data.site_xpos[b])*1000.0)
                dgap=float(np.max(np.abs(gaps)));maxD=max(maxD,dgap)
                active=[]
                for c in data.contact[:data.ncon]:
                    a,b=int(c.geom1),int(c.geom2)
                    if obj in (a,b):
                        active.append({"geom1":geom(model,a),"geom2":geom(model,b),
                            "body1":body(model,int(model.geom_bodyid[a])),"body2":body(model,int(model.geom_bodyid[b])),
                            "distance_mm":float(c.dist)*1000})
                min_group={k:base.min_pair_distance(model,data,v,[obj],exc,points=k.endswith("pad")) for k,v in groups.items()}
                points={};geoms={}
                for side in ("narrow","wide"):
                    item=min_group[side+"_pad"]
                    if item is None or item["distance_m"]>base.TOL:
                        points[side]=();geoms[side]=()
                    else:
                        local=(np.asarray(item["point2_world_m"])-data.xpos[objbody])@data.xmat[objbody].reshape(3,3)
                        points[side]=(tuple(map(float,local)),);geoms[side]=(item["geom1"],)
                self_contacts=[]
                for c in data.contact[:data.ncon]:
                    a,b=int(c.geom1),int(c.geom2)
                    if obj in (a,b):continue
                    b1,b2=body(model,int(model.geom_bodyid[a])),body(model,int(model.geom_bodyid[b]))
                    if b1.startswith(("R_hand_","R_omnipicker")) and b2.startswith(("R_hand_","R_omnipicker")):
                        self_contacts.append({"geom1":geom(model,a),"geom2":geom(model,b),"body1":b1,"body2":b2,"distance_mm":float(c.dist)*1000})
                maxSelf=max(maxSelf,max((max(0.0,-x["distance_mm"]) for x in self_contacts),default=0.0))
                closure.append({"candidate":candidate["id"],"step":idx,"percent":entry["percent"],"qCAD":entry["qCAD"],
                    "pad_clearance_mm":{k:None if min_group[k+"_pad"] is None else min_group[k+"_pad"]["distance_m"]*1000 for k in ("narrow","wide")},
                    "linkage_loop_mm":None if min_group["linkage_loop"] is None else min_group["linkage_loop"]["distance_m"]*1000,
                    "other_gripper_mm":None if min_group["other_gripper"] is None else min_group["other_gripper"]["distance_m"]*1000,
                    "pad_points_object_local_m":points,"pad_geom_names":geoms,"D_gap_component_mm":dgap,
                    "object_contacts":active,"internal_contacts":self_contacts})
            first_pad=lambda side:next((x for x in closure if x["pad_clearance_mm"][side] is not None
                                        and x["pad_clearance_mm"][side]<=0.1),None)
            first_loop=next((x for x in closure if x["linkage_loop_mm"] is not None and x["linkage_loop_mm"]<=0.1),None)
            def in_band_pair(x):
                return all(x["pad_clearance_mm"][side] is not None and x["pad_clearance_mm"][side]<=0.1
                    and x["pad_points_object_local_m"][side]
                    and abs(x["pad_points_object_local_m"][side][0][2]-candidate["dz_m"])<=.005
                    for side in ("narrow","wide"))
            first_bilateral=next((x for x in closure if in_band_pair(x)),None)
            clean=[x for x in closure if all(x["pad_clearance_mm"][s] is not None and x["pad_clearance_mm"][s]<=0.1
                and x["pad_points_object_local_m"][s] and abs(x["pad_points_object_local_m"][s][0][2]-candidate["dz_m"])<=.005
                for s in ("narrow","wide")) and (x["linkage_loop_mm"] is None or x["linkage_loop_mm"]>0.1)]
            worst_ik=max((x["ik"]["position_error_mm"] for x in ik_rows),default=0.0)
            worst_rot=max((x["ik"]["orientation_error_deg"] for x in ik_rows),default=0.0)
            results.append({"candidate":candidate,"IK":"PASS_STATIC_IK" if all(x["ik"]["converged"] for x in ik_rows) else "FAIL",
                "waypoint_count":len(ik_rows),"max_position_residual_mm":worst_ik,"max_orientation_residual_deg":worst_rot,
                "arm_limits_pass":all(all(limits[j][0]<=q<=limits[j][1] for j,q in zip(base.ARM,x["q_arm_rad"])) for x in ik_rows),
                "path_status":"PASS_STATIC_SAMPLED_PATH" if path_ok else "BLOCKED_STATIC_PATH",
                "path_min_clearance_mm":min(x["min_collision_clearance_mm"] for x in path_metrics),
                "path_first_blocker":next((x for x in path_metrics if x["min_collision_clearance_mm"]<1 or x["robot_object_contacts"] or x["moving_self_contacts"]),None),
                "closure_status":"PASS_STATIC_PAD_FIRST" if first_bilateral and clean and (first_loop is None or first_bilateral["step"]<first_loop["step"]) else "BLOCKED_STATIC_CLOSURE",
                "first_narrow_pad":first_pad("narrow"),"first_wide_pad":first_pad("wide"),
                "first_bilateral_contact_in_band":first_bilateral,"first_loop":first_loop,
                "bilateral_clean_count":len(clean),"max_D_gap_component_mm":maxD,"max_internal_penetration_mm":maxSelf,
                "ik_timeline":ik_rows,"path_timeline":path_metrics,"closure_timeline":closure})
            all_path.extend(path_metrics);all_closure.extend(closure)
        result={"run_id":str(uuid.uuid4()),"classification":"SIMULATION_ONLY_STATIC_BOUNDED_POSE_SCREEN",
            "physics_steps":0,"mj_step_calls":step_count,"mj_forward_calls":forward_count,
            "PR90_head":PR90_HEAD,"PR90_model_sha256":base.sha(NEWXML),
            "full_x2_model_sha256":hashlib.sha256((base.make_overlay_xml(OLDXML.read_text(),json.loads(RANGE.read_text()))).encode()).hexdigest(),
            "accepted_scene_object":{"diameter_m":base.D,"height_m":base.H,"mass_kg":base.MASS,"center_world_m":list(base.CENTER)},
            "source_joint_limits_rad":{j:list(v) for j,v in limits.items()},
            "fixed_geometry_notice":"No meshes, scales, mounts, collision masks, exclusions, or quartic coefficients changed.",
            "candidates":results}
    finally:
        mujoco.mj_forward=original_forward
        mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original_steps
    out=pg.create_unique_output_directory(parent.wsl_path,prefix="pr90-bounded-pose-static",
        authorized_roots=(str(REV),),environment="wsl")
    result["output"]=out.as_dict()
    result["runner_path"]=str(Path(__file__).resolve())
    result["runner_sha256"]=sha(Path(__file__).resolve())
    pg.write_new_text(out.native_path/"result.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                      authorized_roots=(str(REV),),environment="wsl")
    report=["# PR90 bounded alternative-pose static screen","",
      f"Run ID {result['run_id']}; output {out.wsl_path}; UNC {out.windows_unc}.",
      f"Static only: mj_step={step_count}; mj_forward={forward_count}.",
      "Candidates: +50 mm hover; contact band +/-20 mm; world wrist yaw +/-15 degrees. Original model, object, source limits, PR90 joint states, and collision rules are unchanged.",
      ""]
    for x in results:
        report.append(f"- {x['candidate']['id']}: IK {x['IK']}; path {x['path_status']} (min {x.get('path_min_clearance_mm')}); closure {x['closure_status']}; first bilateral in band {x.get('first_bilateral_contact_in_band')}.")
    report.extend(["","No candidate is a physical grasp result. Static collision proximity only; no dynamics or lifting was run."])
    pg.write_new_text(out.native_path/"REPORT.md","\n".join(report)+"\n",authorized_roots=(str(REV),),environment="wsl")
    def write_csv(path,rows,fields):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(dict,list)) else row.get(k,"") for k in fields})
        pg.write_new_text(out.native_path/path,stream.getvalue(),authorized_roots=(str(REV),),environment="wsl")
    write_csv("candidate_paths.csv",all_path,["candidate","sample","alpha","min_collision_clearance_mm","nearest_pair","gripper_table_clearance_mm","robot_object_contacts","moving_self_contacts"])
    write_csv("candidate_closures.csv",all_closure,["candidate","step","percent","qCAD","pad_clearance_mm","linkage_loop_mm","other_gripper_mm","pad_points_object_local_m","pad_geom_names","D_gap_component_mm","object_contacts","internal_contacts"])
    command=("cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023\n"
        "env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python "
        f"{Path(__file__).resolve()}\n")
    pg.write_new_text(out.native_path/"REPRODUCE.md",command,authorized_roots=(str(REV),),environment="wsl")
    manifest=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
    pg.verify_artifacts(manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
    print(json.dumps({"output":out.as_dict(),"run_id":result["run_id"],
        "candidates":[{"id":x["candidate"]["id"],"IK":x["IK"],"path":x["path_status"],"closure":x["closure_status"]} for x in results],
        "steps":step_count,"forwards":forward_count,"manifest":len(manifest["artifacts"])},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
