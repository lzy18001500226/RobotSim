"""Table placement extension for the existing grip owner; no motion publisher.

Targets are the known table frame plus the reviewed circle calibration.
Native object truth is used only for contact/support/drop acceptance.
No source model, qpos write, attachment or alternate controller is introduced.
"""
from __future__ import annotations
import json
import math
import time

def placement_contact_state(row, request):
    body=row["bodies"]["test_cylinder"];pos=body["position_m"];R=body["rotation"]
    axis=[R[2],R[5],R[8]];velocity=body["origin_velocity_world_angular_linear"]
    radius=float(request["bottle_radius_m"]);half=float(request["bottle_height_m"])/2.
    extent=[radius*math.sqrt(max(0.,1.-v*v))+half*abs(v) for v in axis]
    lo=request["table_bounds_min_world_m"];hi=request["table_bounds_max_world_m"]
    inside=all(lo[i]+.005<=pos[i]-extent[i] and pos[i]+extent[i]<=hi[i]-.005 for i in (0,1))
    vertical=0.;pads=0.;forbidden=[]
    for c in row["contacts"]:
        if c["efc_address"]<0 or "test_cylinder" not in (c["body1"],c["body2"]):continue
        other=c["body2"] if c["body1"]=="test_cylinder" else c["body1"]
        force=max(0.,float(c["local_force_torque"][0]))
        if other=="raicom_delivery_table":vertical+=float(c["signed_bottle_world_z_force_n"])
        elif other in ("R_hand_narrow3_Link","R_hand_wide3_Link"):pads+=force
        elif force>.1:forbidden.append(other)
    values=[*pos,*axis,*velocity,vertical,pads]
    if not all(math.isfinite(v) for v in values):raise RuntimeError("NONFINITE_PLACEMENT_STATE")
    return dict(host_ns=row["host_monotonic_ns"],sim_time_s=row["sim_time_s"],generation=row["generation"],
                position_world_m=pos,axis_world=axis,table_vertical_force_n=vertical,pad_normal_force_n=pads,
                inside_table=inside,bottom_world_z_m=pos[2]-extent[2],forbidden=forbidden,
                planar_speed_m_s=math.hypot(velocity[3],velocity[4]),vertical_speed_m_s=abs(velocity[5]),
                angular_speed_rad_s=math.sqrt(sum(v*v for v in velocity[:3])))

def evaluate_placed_hold(rows, request):
    tail=[];last=None;needed=.75*float(request["bottle_mass_kg"])*9.81
    for row in rows:
        item=placement_contact_state(row,request)
        valid=(item["generation"]==request["generation"] and item["inside_table"] and
               item["table_vertical_force_n"]>=needed and item["pad_normal_force_n"]<=.1 and
               item["bottom_world_z_m"]>=request["tabletop_world_z_m"]-.005 and not item["forbidden"] and
               item["planar_speed_m_s"]<=.02 and item["vertical_speed_m_s"]<=.02 and item["angular_speed_rad_s"]<=.1)
        if last is not None:
            dt=(item["host_ns"]-last["host_ns"])*1e-9
            if not 0.<dt<=.20 or not 0.<item["sim_time_s"]-last["sim_time_s"]<=.20:valid=False
        if valid:tail.append(item)
        else:tail=[]
        if tail:
            while len(tail)>1 and item["sim_time_s"]-tail[1]["sim_time_s"]>=2.:tail.pop(0)
        last=item
    span=tail[-1]["sim_time_s"]-tail[0]["sim_time_s"] if len(tail)>1 else 0.
    return dict(pass_result=len(tail)>=25 and span>=2.,measured_hold_sim_s=span,unique_samples=len(tail),
                classification="RELEASED_BOTTLE_SUPPORTED_BY_DELIVERY_TABLE_NOT_GRIP",
                last=last,minimum_table_vertical_force_n=min((x["table_vertical_force_n"] for x in tail),default=None),
                maximum_pad_normal_force_n=max((x["pad_normal_force_n"] for x in tail),default=None))

def observe_released_support(tick,point,request,rows,*,deadline_s,clock=time.monotonic):
    """Collect measured continuous support inside the existing placement budget."""
    result=None
    while clock()<deadline_s:
        item=tick(point,[0.,0.,1.],HAND_OPEN)
        if not item["inside_table"] or item["bottom_world_z_m"]<request["tabletop_world_z_m"]-.005 or item["forbidden"]:
            raise RuntimeError("RELEASED_BOTTLE_LEFT_SAFE_TABLE_SUPPORT:"+json.dumps(item))
        result=evaluate_placed_hold(rows[-175:],request)
        if result["pass_result"]:return result
    return result or dict(pass_result=False,classification="RELEASED_SUPPORT_OBSERVATION_INCOMPLETE",measured_hold_sim_s=0.,unique_samples=0)

def perform_native_placement(n, request):
    # This function is injected into the existing owner process. It uses that
    # process's original send/tracker/force/feedback path, never a second owner.
    import os
    if (request.get("schema")!="robotsim.x2.table-place-request.v1" or
        request.get("owner_pid")!=os.getpid() or request.get("generation")!=n.native_generation or
        request.get("navigation_and_composite_stop_pass") is not True or
        request.get("confirmed_stand") is not True or
        not 0.<=(time.monotonic_ns()-int(request["created_host_ns"]))*1e-9<=.20):
        raise RuntimeError("PLACEMENT_HANDOFF_NOT_QUALIFIED")
    began=time.monotonic()
    n.refresh_native();n.placement_root_start=list(n.native_row["bodies"]["pelvis"]["position_m"])
    if request.get("preplacement_arm_settings"):
        n.placement_request=request
        perform_native_neutral_stow(n,request["preplacement_arm_settings"],result_name="native_preplacement_stow_result.json")
    n.refresh_native();start_row=n.native_row;start_pose=n.placement_root_start
    hand=start_row["bodies"]["R_omnipicker_base_link"];H=hand["rotation"];local=CIRCLE_CONTRACT["circle_center_local_m"]
    initial=[hand["position_m"][i]+sum(H[3*i+j]*local[j] for j in range(3)) for i in range(3)]
    local_axis=CIRCLE_CONTRACT["local_bottle_axis_in_hand"]
    initial_axis=[sum(H[3*i+j]*local_axis[j] for j in range(3)) for i in range(3)]
    desired=list(request["section_target_world_m"]);clearance=desired.copy();clearance[2]+=.06
    if math.dist(initial,clearance)>.45:raise RuntimeError("PLACEMENT_CARTESIAN_EXCURSION")
    target=dict(n.last_published_arm_target);n.placement_request=request;n.placement_release=False
    n.lift_phase.update(placement_active=True,placement_start_world_m=initial,placement_request=request)
    rows=[];n.placement_rows=rows
    def tick(point,axis,hand_command):
        if time.monotonic()-began>35.:raise RuntimeError("PLACEMENT_WATCHDOG")
        n.lift_phase.update(placement_target_world_m=list(point),placement_axis_world=list(axis))
        n.write_phase(n.lift_phase)
        n.send(target,.02,"CARRY_HOLD",hand_command)
        n.refresh_native()
        if math.dist(start_pose,n.native_row["bodies"]["pelvis"]["position_m"])>.08:raise RuntimeError("PLACEMENT_BASE_MOVED")
        if not rows or n.native_row["host_monotonic_ns"]>rows[-1]["host_monotonic_ns"]:rows.append(n.native_row)
        return placement_contact_state(n.native_row,request)
    high=initial.copy();high[2]=max(initial[2],clearance[2])
    for index in range(150):
        u=(index+1)/150.;v=u*u*(3.-2.*u)
        point=[a+v*(b-a) for a,b in zip(initial,high)]
        axis=[a+v*(b-a) for a,b in zip(initial_axis,[0.,0.,1.])]
        size=math.sqrt(sum(x*x for x in axis));axis=[x/size for x in axis]
        tick(point,axis,HAND_RIGHT)
    for index in range(200):
        u=(index+1)/200.;v=u*u*(3.-2.*u)
        tick([a+v*(b-a) for a,b in zip(high,clearance)],[0.,0.,1.],HAND_RIGHT)
    for _ in range(25):tick(clearance,[0.,0.,1.],HAND_RIGHT)
    if n.tracking_metrics.get("actual_position_error_m",1.)>.005:raise RuntimeError("PLACEMENT_APPROACH_NOT_REACHED")
    # Approach the support surface slowly. A positive actual vertical table
    # reaction gates release; a planned geometric height alone never suffices.
    point=clearance.copy();contact=None
    for _ in range(400):
        point[2]-=.0002
        contact=tick(point,[0.,0.,1.],HAND_RIGHT)
        if contact["forbidden"]:raise RuntimeError("PLACEMENT_FORBIDDEN_CONTACT")
        if contact["table_vertical_force_n"]>10.*request["bottle_mass_kg"]*9.81:raise RuntimeError("PLACEMENT_EXCESS_TABLE_FORCE")
        if contact["table_vertical_force_n"]>.2:break
    else:raise RuntimeError("PLACEMENT_TABLE_CONTACT_NOT_OBSERVED")
    n.placement_release=True
    start_command=float(n.grip_command)
    for index in range(150):
        u=(index+1)/150.;u=u*u*(3.-2.*u)
        tick(point,[0.,0.,1.],start_command+u*(HAND_OPEN-start_command))
    n.placement_released=True
    # Reserve the unchanged 100 withdrawal ticks and 175 final hold ticks.
    first_hold_deadline=began+35.-(100+175)*.02
    result=observe_released_support(tick,point,request,rows,deadline_s=first_hold_deadline)
    if not result or not result["pass_result"]:raise RuntimeError("RELEASED_BOTTLE_NOT_STABLY_TABLE_SUPPORTED:"+json.dumps(result))
    # Clear the real open claw, then require the same independent released
    # support again. No geometric cage is counted as an independent placement.
    n.refresh_native()
    if -float(n.native_row["joints"]["R_claw_joint"]["q"])<.95:raise RuntimeError("PLACEMENT_HAND_NOT_ACTUALLY_OPEN")
    hand=n.native_row["bodies"]["R_omnipicker_base_link"];H=hand["rotation"];local=CIRCLE_CONTRACT["circle_center_local_m"]
    outward=[sum(H[3*i+j]*local[j] for j in range(3)) for i in range(3)];length=math.sqrt(sum(x*x for x in outward))
    retreat=[point[i]-.08*outward[i]/length for i in range(3)]
    start_point=point.copy()
    for index in range(100):
        u=(index+1)/100.;v=u*u*(3.-2.*u)
        tick([a+v*(b-a) for a,b in zip(start_point,retreat)],[0.,0.,1.],HAND_OPEN)
    result=observe_released_support(tick,retreat,request,rows,deadline_s=began+35.)
    if not result["pass_result"]:raise RuntimeError("PLACEMENT_NOT_STABLE_AFTER_WITHDRAWAL:"+json.dumps(result))
    result["open_claw_withdrawal_m"]=.08
    result.update(owner_pid=os.getpid(),generation=n.native_generation,command_path="EXISTING_HAL_ARM_AND_HAND_OWNER",table_body="raicom_delivery_table",elapsed_wall_s=time.monotonic()-began,request=request)
    write_durable_milestone("/workspace/evidence/native_table_place_result.json",result)
    n.placement_completed=True
    return result


def extend_existing_owner(script):
    """Opt-in extension of the existing grip-owner source, not a new owner."""
    import inspect
    helper="\n".join(inspect.getsource(f) for f in (placement_contact_state,evaluate_placed_hold,observe_released_support,perform_native_placement))
    if script.count("class Plan(Node):")!=1:raise ValueError("placement owner class anchor")
    script=script.replace("class Plan(Node):",helper+"\nclass Plan(Node):")
    lines=script.splitlines();output=[]
    for line in lines:
        text=line.strip()
        if text=="n.send(carry_target,.05,'CARRY_HOLD',HAND_RIGHT)" and len(line)-len(line.lstrip())==5:
            output.extend([
                "     if os.path.exists('/workspace/evidence/carry_place_request.json') and not getattr(n,'placement_completed',False):",
                "      try:",
                "       with open('/workspace/evidence/carry_place_request.json') as stream:request=json.load(stream)",
                "       perform_native_placement(n,request)",
                "      except Exception as exc:",
                "       if not os.path.exists('/workspace/evidence/native_table_place_result.json'):write_durable_milestone('/workspace/evidence/native_table_place_result.json',{'pass_result':False,'failure':str(exc),'owner_pid':os.getpid(),'time_ns':time.monotonic_ns()})",
                "       raise",
            ])
        if "n.send(carry_target," in line and "'CARRY_HOLD',HAND_RIGHT)" in line:
            line=line.replace("'CARRY_HOLD',HAND_RIGHT)","'CARRY_HOLD',HAND_OPEN if getattr(n,'placement_released',False) else HAND_RIGHT)")
        if "if 'test_cylinder' in pair and not pair." in line:
            line=line.replace("and not pair.","and not (getattr(self,'placement_request',None) and 'raicom_delivery_table' in pair) and not pair.")
        if "if (sum(forces.values()) if stage" in line:
            line=line[:-1]+" and not (getattr(self,'placement_release',False) and placement_contact_state(self.native_row,self.placement_request)['table_vertical_force_n']>.1):"
        if "if LIFT_CONTRACT is not None and stage in " in line:
            line=line[:-1]+" and not getattr(self,'placement_release',False):"
        if text.startswith("contact_result=native_bilateral_contact("):
            line=" contact_result=dict(n.initial_physical_bilateral_contact,scope='PRE_LIFT_VERIFIED_BEFORE_CARRY') if getattr(n,'placement_completed',False) else "+text.split("=",1)[1]
        if "'classification':'GRIP_HOLD_AND_RELEASE_NOT_ROUTE_ACCEPTANCE'" in line:
            line=line.replace("'classification':'GRIP_HOLD_AND_RELEASE_NOT_ROUTE_ACCEPTANCE'","'classification':('MEASURED_TABLE_PLACEMENT_WITH_RELEASED_HAND' if getattr(n,'placement_completed',False) else 'GRIP_HOLD_AND_RELEASE_NOT_ROUTE_ACCEPTANCE')")
        if "'mode':'MEASURED_LOAD_BEARING_CARRY_HOLD'" in line:
            line=line.replace("'mode':'MEASURED_LOAD_BEARING_CARRY_HOLD'","'mode':('PLACEMENT_TABLE_SUPPORTED' if getattr(self,'placement_release',False) else 'MEASURED_LOAD_BEARING_CARRY_HOLD')")
        if "'right_hand_command':HAND_RIGHT" in line:
            line=line.replace("'right_hand_command':HAND_RIGHT","'right_hand_command':(HAND_OPEN if getattr(n,'placement_completed',False) else HAND_RIGHT)")
        output.append(line)
        if "'prelift_bilateral_contact':contact" in text:
            output.append("   n.initial_physical_bilateral_contact=dict(contact)")
        if text=="n.send(carry_target,1.5,'CARRY_HOLD',HAND_RIGHT)":
            output.extend([
                "    if getattr(n,'placement_completed',False):",
                "     final_table=evaluate_placed_hold(n.native_lift_rows,n.placement_request)",
                "     if not final_table['pass_result']:raise RuntimeError('TABLE_SUPPORT_LOST_BEFORE_CLOSEOUT:'+json.dumps(final_table))",
            ])
        if text.startswith("self.arm.update(values);self.native_row=row;"):
            output.extend([
                "  if getattr(self,'placement_request',None):",
                "   fault=physical_fault(row,native_pose(row),LIFT_CONTRACT['retract_preflight']['source_motion_limits'],robot_bodies=LIFT_CONTRACT['retract_preflight']['robot_bodies'],allowed_contact_pairs=[['test_cylinder',p] for p in ('R_hand_narrow3_Link','R_hand_wide3_Link')],review_incidental_contacts=bool(LIFT_CONTRACT['retract_preflight'].get('incidental_contact_diagnostic')))",
                "   if fault:raise RuntimeError('PLACEMENT_PHYSICAL_SAFETY:'+fault)",
                '   if getattr(self,"placement_root_start",None) and math.dist(self.placement_root_start,self.native_row["bodies"]["pelvis"]["position_m"])>.08:raise RuntimeError("PLACEMENT_BASE_MOVED")',
            ])
    result="\n".join(output)+"\n";compile(result,"existing_owner_with_table_placement","exec")
    return result



def verify_right_claw_pd_only(source, current, source_sha256, *, allow_episode_start=False):
    """Reuse a geometry-only screen only after exact compiled-model comparison."""
    import hashlib,mujoco,numpy as np
    from pathlib import Path
    if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=source_sha256:raise ValueError("aisle reference hash mismatch")
    a=mujoco.MjModel.from_xml_path(str(source));b=mujoco.MjModel.from_xml_path(str(current))
    i=mujoco.mj_name2id(a,mujoco.mjtObj.mjOBJ_ACTUATOR,"motor_R_claw_joint")
    if i<0 or (a.nq,a.nv,a.nu,a.nbody,a.ngeom)!=(b.nq,b.nv,b.nu,b.nbody,b.ngeom):raise ValueError("aisle topology changed")
    allowed={}
    for name in ("actuator_gainprm","actuator_biasprm"):
        x=getattr(a,name).copy();y=getattr(b,name)
        x[i]=y[i];allowed[name]=x
    if b.actuator_gainprm[i,0] not in (2.,4.,6.) or not np.array_equal(b.actuator_forcerange[i],[-2.,2.]):raise ValueError("unsupported claw control delta")
    expected_gain=a.actuator_gainprm[i].copy();expected_gain[0]=b.actuator_gainprm[i,0]
    expected_bias=a.actuator_biasprm[i].copy();expected_bias[1]=-b.actuator_gainprm[i,0]
    if not np.array_equal(expected_gain,b.actuator_gainprm[i]) or not np.array_equal(expected_bias,b.actuator_biasprm[i]):raise ValueError("delta is not right-claw kp only")
    if allow_episode_start:
        joint=mujoco.mj_name2id(a,mujoco.mjtObj.mjOBJ_JOINT,"floating_base_joint");adr=int(a.jnt_qposadr[joint])
        expected=a.key_qpos.copy();expected[:,adr:adr+2]=b.key_qpos[:,adr:adr+2];expected[:,adr+3:adr+7]=b.key_qpos[:,adr+3:adr+7]
        allowed["key_qpos"]=expected
    checked=[]
    for name in dir(a):
        x=getattr(a,name)
        if not isinstance(x,np.ndarray):continue
        y=getattr(b,name)
        if not np.array_equal(allowed.get(name,x),y,equal_nan=True):raise ValueError("aisle non-PD model delta:"+name)
        checked.append(name)
    return dict(scope="RIGHT_CLAW_KP_ONLY_GEOMETRY_CLEARANCE_REUSE_NOT_DYNAMICS",source=str(source),source_sha256=source_sha256,current=str(current),old_kp=float(a.actuator_gainprm[i,0]),new_kp=float(b.actuator_gainprm[i,0]),torque_cap_nm=2.,checked_array_count=len(checked),mj_step_calls=0)


def prepare_delivery_scene(parent, config_path, *, host_model_cache=None):
    """One actual-scene static screen; no mj_step and no source model edits."""
    import ast,copy,hashlib
    from pathlib import Path
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scripts.agent import path_governance as pg
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
    from scripts.research.issue46.grasp.stl_approved_circle_ik import axis_grasp_error,circle_task_jacobian
    root=Path("/home/lzy18001500226/projects/RobotSim")
    record=json.loads(Path(parent).read_text());cfg=json.loads(Path(config_path).read_text())
    if cfg.get("schema")!="robotsim.x2.two-table-delivery.v1" or not record.get("approved_two_table_layout"):raise ValueError("two-table scene provenance")
    if float(cfg.get("delivery_forward_command_m_s",.20)) not in (.20,.30,.40,5./9.):raise ValueError("delivery forward response profile bound")
    if cfg.get("terminal_braking_aisle_reserve") and cfg["terminal_braking_aisle_reserve"]["scene_sha256"]!=record["generated"]["host_scene"]["sha256"]:
        ref=cfg["terminal_braking_aisle_reserve"]
        record["terminal_aisle_control_equivalence"]=verify_right_claw_pd_only(ref["scene_path"],record["generated"]["host_scene"]["path"],ref["scene_sha256"],allow_episode_start=bool(record.get("episode_start_navigation")))
    if host_model_cache is not None:
        from scripts.research.issue46.perception.native_rgbd_grasp import load_cached_model
        model=load_cached_model(Path(record["generated"]["host_scene"]["path"]),Path(host_model_cache))
    else:model=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"raicom_delivery_table")
    points=[]
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=table or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mid=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        points.extend(v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
    points=np.asarray(points)
    if not np.allclose(points.min(axis=0),cfg["table_bounds_min_world_m"],atol=1e-6) or not np.allclose(points.max(axis=0),cfg["table_bounds_max_world_m"],atol=1e-6):raise ValueError("actual compiled delivery table differs from config")
    geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"test_cylinder_collision")
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder")
    if not np.allclose(model.geom_size[geom,:2],[cfg["bottle_radius_m"],cfg["bottle_height_m"]/2.]) or abs(model.body_mass[body]-cfg["bottle_mass_kg"])>1e-9:raise ValueError("actual bottle differs from profile")
    source=Path(cfg.get("route_reference",{}).get("path",str(root/"reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261108T190527Z")))
    if cfg.get("route_reference") and hashlib.sha256((source/"native_latest.json").read_bytes()).hexdigest()!=cfg["route_reference"]["native_latest_sha256"]:
        raise ValueError("accepted route start reference changed")
    route_reference=json.loads((source/"native_latest.json").read_text())
    row=route_reference
    if cfg.get("carry_native_neutral_stow"):
        ready=json.loads((source/"carry_ready.json").read_text())
        for line in (source/"native_physics.jsonl").open():
            candidate=json.loads(line)
            if candidate["host_monotonic_ns"]>=ready["host_monotonic_ns"]:
                row=candidate;break
    if row["engine_version"]!=337:raise ValueError("reference ABI")
    constants={}
    for n in ast.parse((source/"native_hal_bilateral_arm_pregrasp.py").read_text()).body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in ("CIRCLE_CONTRACT","RIGHT","LIFT_CONTRACT"):constants[target.id]=ast.literal_eval(n.value)
    apply_recorded_native_frame(model,data,row)
    if record.get("fixture_height_delta"):
        # Static planning assumption only. Live pickup uses fresh RGB-D and
        # actual measured contact; no runtime object-state write is generated.
        hand_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link")
        object_adr=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"test_cylinder_free")])
        rotation=data.xmat[hand_id].reshape(3,3)
        local=np.asarray(constants["CIRCLE_CONTRACT"]["circle_center_local_m"])
        axis_local=np.asarray(constants["CIRCLE_CONTRACT"]["local_bottle_axis_in_hand"])
        data.qpos[object_adr:object_adr+3]=data.xpos[hand_id]+rotation@(local-float(cfg["grasp_height_above_center_m"])*axis_local)
        mujoco.mj_forward(model,data)
    stow_plan=None
    if cfg.get("carry_native_neutral_stow"):
        from scripts.research.issue46.native_runtime.native_carry_posture import native_stand_arm_reference,screen_native_stow
        actual_settings=native_stand_arm_reference(profile=cfg["carry_native_neutral_stow"].get("reference_profile","FULL_SOURCE"))
        if actual_settings!=cfg["carry_native_neutral_stow"]:raise ValueError("native stow source changed")
        stow_plan=screen_native_stow(model,data,actual_settings)
        if not stow_plan["pass_static"]:raise ValueError("native stow collision:"+str([r for r in stow_plan["rows"] if r["collisions"]][:1]))
        stow_plan["actual_reference_source_ns"]=row["host_monotonic_ns"]
        oldhand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link");oldobj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder")
        H=data.xmat[oldhand].reshape(3,3).copy();held_local=H.T@(data.xpos[oldobj]-data.xpos[oldhand]);held_rotation=H.T@data.xmat[oldobj].reshape(3,3)
        row=route_reference;apply_recorded_native_frame(model,data,row)
        for name,q in actual_settings["arm_target_rad"].items():
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]]=q
        mujoco.mj_forward(model,data);H=data.xmat[oldhand].reshape(3,3)
        oa=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"test_cylinder_free")])
        data.qpos[oa:oa+3]=data.xpos[oldhand]+H@held_local
        quat=Rotation.from_matrix(H@held_rotation).as_quat();data.qpos[oa+3:oa+7]=quat[[3,0,1,2]]
        mujoco.mj_forward(model,data)
    if cfg.get("review_prefix_only"):
        out=pg.create_unique_output_directory(root/"reviews/issue-46",prefix="retreat-turn-review-scene-check").native_path
        plan=dict(classification="PREFIX_REVIEW_SCENE_AND_STOW_ONLY_NOT_DYNAMIC_VALIDATION",config=cfg,config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),host_scene=record["generated"]["host_scene"],mj_step_calls=0,stow_plan=stow_plan,scene=record["generated"]["host_scene"],placement_and_route_to_second_table="NOT_RUN")
        pg.write_new_text(out/"plan.json",json.dumps(plan,indent=2))
        record["delivery_static_plan"]={"path":str(out/"plan.json"),"sha256":hashlib.sha256((out/"plan.json").read_bytes()).hexdigest()}
        pg.write_new_text(out/"result.json",json.dumps(record,indent=2))
        return out/"result.json"
    old=np.asarray(row["bodies"]["pelvis"]["position_m"]);oldR=np.asarray(row["bodies"]["pelvis"]["rotation"]).reshape(3,3)
    oldyaw=math.atan2(oldR[1,0],oldR[0,0]);goal=np.asarray(cfg["base_goal_world_xy"])
    hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link");bottle=body
    rel=oldR.T@(data.xpos[bottle]-old);object_rotation_local=oldR.T@data.xmat[bottle].reshape(3,3)
    free=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"floating_base_joint")])
    object_free=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"test_cylinder_free")])
    robot_root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    robot={i for i in range(model.nbody) if model.body_rootid[i]==robot_root}
    def collision_screen(check_object=True):
        bad=[]
        for c in data.contact:
            if c.dist>=-.001:continue
            a,b=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            ra,rb=a in robot,b in robot
            if check_object and body in (a,b):
                other=b if a==body else a
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other)
                if name not in ("R_hand_narrow3_Link","R_hand_wide3_Link"):
                    bad.append(["test_cylinder",name,float(c.dist)])
                continue
            if ra==rb:continue
            other=b if ra else a
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other)
            if name in ("world","test_cylinder"):continue
            bad.append([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,a),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b),float(c.dist)])
        return bad
    def set_root(xy,yaw):
        R=Rotation.from_euler("z",yaw-oldyaw).as_matrix()@oldR
        data.qpos[free:free+2]=xy;quat=Rotation.from_matrix(R).as_quat();data.qpos[free+3:free+7]=quat[[3,0,1,2]]
        data.qpos[object_free:object_free+3]=np.r_[xy,old[2]]+R@rel
        quat=Rotation.from_matrix(R@object_rotation_local).as_quat();data.qpos[object_free+3:object_free+7]=quat[[3,0,1,2]]
        mujoco.mj_forward(model,data)
    egress=float(cfg.get("post_turn_egress_m",0.))
    if egress not in (0.,1.):raise ValueError("unsupported outward egress length")
    waypoints=[old[:2]]
    if egress:waypoints.append(old[:2]+(egress+.10)*np.array([math.cos(oldyaw),math.sin(oldyaw)]))
    waypoints.append(goal);poses=[];previous_yaw=oldyaw
    for a,b in zip(waypoints,waypoints[1:]):
        bearing=math.atan2(b[1]-a[1],b[0]-a[0])
        poses.extend((a,previous_yaw+u*((bearing-previous_yaw+math.pi)%(2*math.pi)-math.pi)) for u in np.linspace(0,1,9))
        poses.extend((a+u*(b-a),bearing) for u in np.linspace(0,1,17))
        previous_yaw=bearing
    poses.extend((goal,bearing+u*((cfg["base_goal_world_yaw_rad"]-bearing+math.pi)%(2*math.pi)-math.pi)) for u in np.linspace(0,1,9))
    for xy,yaw in poses:
        set_root(xy,yaw)
        bad=collision_screen()
        if bad:raise ValueError("nominal actual-scene route collision:"+str(bad[:3]))
    preplacement=cfg.get("preplacement_arm_settings")
    if preplacement:
        pin=cfg["preplacement_reference"];reference_path=Path(pin["path"])
        if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=pin["sha256"] or json.loads(reference_path.read_text())["settings"]!=preplacement:
            raise ValueError("preplacement operational source changed")
        from scripts.research.issue46.native_runtime.native_carry_posture import screen_native_stow
        preplace_screen=screen_native_stow(model,data,preplacement)
        if not preplace_screen["pass_static"]:raise ValueError("preplacement arm path collision:"+str([x for x in preplace_screen["rows"] if x["collisions"]][:1]))
    names=constants["RIGHT"];ids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names])
    qa=model.jnt_qposadr[ids];dof=model.jnt_dofadr[ids];qseed=data.qpos[qa].copy()
    bounds=np.asarray([constants["LIFT_CONTRACT"]["retract_preflight"]["source_motion_limits"][n][:2] for n in names])
    centre=np.asarray(constants["CIRCLE_CONTRACT"]["circle_center_local_m"]);axis=np.asarray(constants["CIRCLE_CONTRACT"]["local_bottle_axis_in_hand"])
    section=np.asarray(cfg["placement_center_world_m"])+[0.,0.,float(cfg.get("grasp_height_above_center_m",.10))]
    start=data.xpos[hand]+data.xmat[hand].reshape(3,3)@centre
    start_axis=data.xmat[hand].reshape(3,3)@axis
    high=start.copy();high[2]=max(start[2],section[2]+.06)
    targets=[(start+u*(high-start),(1-u)*start_axis+u*np.array([0.,0.,1.])) for u in np.linspace(0,1,9)]
    targets += [(high+u*(section+[0,0,.06]-high),np.array([0.,0.,1.])) for u in np.linspace(0,1,13)]
    targets += [(section+[0.,0.,z],np.array([0.,0.,1.])) for z in np.linspace(.06,0,7)]
    checks=[]
    for point,worldaxis in targets:
        worldaxis=worldaxis/np.linalg.norm(worldaxis)
        def residual(q):
            data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),centre,point,worldaxis,axis)
        def jac(q):
            residual(q);return circle_task_jacobian(model,data,hand,centre,axis,dof)
        fit=least_squares(residual,np.clip(qseed,bounds[:,0]+.01,bounds[:,1]-.01),jac=jac,bounds=(bounds[:,0]+.01,bounds[:,1]-.01),max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        qseed=fit.x;error=residual(qseed)
        mujoco.mj_forward(model,data);bad=collision_screen(check_object=False)
        checks.append(dict(point=point.tolist(),position_error_m=float(np.linalg.norm(error[:3])),axis_residual=float(np.linalg.norm(error[3:])),robot_environment_contacts=bad,right_arm_rad=qseed.tolist()))
        if np.linalg.norm(error[:3])>.001 or np.linalg.norm(error[3:])>.001 or bad:raise ValueError("placement IK/collision screen:"+str(checks[-1]))
    out=pg.create_unique_output_directory(root/"reviews/issue-46",prefix="two-table-delivery-static-plan").native_path
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=dict(stow_plan=stow_plan,reference_native_source_ns=row["host_monotonic_ns"],classification="STATIC_REFERENCE_POSTURE_ROUTE_AND_ARM_SCREEN_NOT_DYNAMIC_PASS",mj_step_calls=0,config=cfg,config_path=str(Path(config_path).resolve()),config_sha256=digest(config_path),host_scene=record["generated"]["host_scene"],reference_native_frame=str(source/"native_latest.json"),reference_native_frame_sha256=digest(source/"native_latest.json"),route_pose_samples=len(poses),arm_samples=checks,limitations="fixed recorded carry posture; actual gait, support/release and arrival remain live gates")
    pg.write_new_text(out/"plan.json",json.dumps(plan,indent=2)+"\n")
    record["delivery_static_plan"]={"path":str(out/"plan.json"),"sha256":digest(out/"plan.json")}
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(out)
    return out/"result.json"


def placement_mode_observer_script():
    return "\n".join([
        "import json,time,rclpy",
        "from rclpy.node import Node",
        "from rclpy.qos import QoSProfile,ReliabilityPolicy",
        "from aimdk_msgs.msg import McCommonState",
        "from aimdk_msgs.srv import GetMcAction",
        "from rosidl_runtime_py.convert import message_to_ordereddict",
        "rclpy.init();n=Node('robotsim_table_placement_readback');state={}",
        "def on_common(m):",
        " age=(time.time_ns()-int(m.header.stamp.sec)*1000000000-int(m.header.stamp.nanosec))*1e-9",
        " if 0.<=age<=.20:state.update(raw=message_to_ordereddict(m),received_ns=time.monotonic_ns())",
        "sub=n.create_subscription(McCommonState,'/aima/mc/common/state',on_common,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))",
        "client=n.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction')",
        "try:",
        " discovery_started=time.monotonic();discovery_deadline=discovery_started+5.",
        " while time.monotonic()<discovery_deadline:",
        "  rclpy.spin_once(n,timeout_sec=.02)",
        "  if client.service_is_ready() and state and (time.monotonic_ns()-state['received_ns'])*1e-9<=.20:break",
        " if not client.service_is_ready():raise RuntimeError('PLACEMENT_MODE_SERVICE_UNAVAILABLE:'+json.dumps(n.get_service_names_and_types()))",
        " if not state or (time.monotonic_ns()-state['received_ns'])*1e-9>.20:raise RuntimeError('PLACEMENT_COMMON_NOT_FRESH_DURING_DISCOVERY')",
        " q=GetMcAction.Request();q.request.header.stamp=n.get_clock().now().to_msg()",
        " future=client.call_async(q);deadline=time.monotonic()+1.",
        " while time.monotonic()<deadline:",
        "  rclpy.spin_once(n,timeout_sec=.01)",
        "  if future.done() and state and (time.monotonic_ns()-state['received_ns'])*1e-9<=.20:break",
        " if not future.done() or future.result() is None or not state:raise RuntimeError('PLACEMENT_FRESH_MODE_UNCONFIRMED')",
        " reply=message_to_ordereddict(future.result());common=state['raw']",
        " passed=reply['header']['code']==0 and reply['info']['action_desc']=='STAND_DEFAULT' and reply['info']['status']['value']==100 and common['action_info']['action_desc']=='STAND_DEFAULT' and common['action_info']['status']['value']==100 and common['body_status']['value']==1 and (time.monotonic_ns()-state['received_ns'])*1e-9<=.20",
        " print(json.dumps(dict(pass_result=passed,get_mode=reply,common=common,completed_host_ns=time.monotonic_ns(),discovery_elapsed_s=time.monotonic()-discovery_started,publishers_created=0,action_requests_sent=0)),flush=True)",
        " if not passed:raise RuntimeError('PLACEMENT_MODE_NOT_STAND')",
        "finally:n.destroy_node();rclpy.shutdown()",
    ])+"\n"
