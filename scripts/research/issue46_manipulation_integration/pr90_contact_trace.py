"""Read-only PR90 bottle contact tracing helpers. Never steps or writes model/data state."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import numpy as np

RIGHT_PAD_BODIES={"R_hand_narrow3_Link","R_hand_wide3_Link"}

def contact_force_world(local_force_torque:Sequence[float],frame:Sequence[float])->np.ndarray:
    f=np.asarray(local_force_torque,dtype=np.float64)
    R=np.asarray(frame,dtype=np.float64).reshape(3,3)
    if f.size<3: raise ValueError("contact force needs 3 components")
    return R.T@f[:3]

def force_on_object(force_world_on_geom2:Sequence[float],object_is_geom2:bool)->np.ndarray:
    f=np.asarray(force_world_on_geom2,dtype=np.float64)
    if f.shape!=(3,): raise ValueError("force must be a 3-vector")
    return f.copy() if object_is_geom2 else -f

def tangential_velocity_components(relative_velocity_world:Sequence[float],frame:Sequence[float])->tuple[float,float]:
    v=np.asarray(relative_velocity_world,dtype=np.float64)
    R=np.asarray(frame,dtype=np.float64).reshape(3,3)
    return float(v@R[1]),float(v@R[2])

def _name(model,objtype,ident):
    import mujoco
    x=mujoco.mj_id2name(model,objtype,int(ident))
    return x or f"unnamed_{objtype}_{ident}"

def _role(body):
    if body=="R_hand_narrow3_Link": return "narrow_pad"
    if body=="R_hand_wide3_Link": return "wide_pad"
    if body.startswith(("R_hand_","R_omnipicker")): return "right_gripper_other"
    if body.startswith(("L_hand_","L_omnipicker")): return "left_gripper"
    if body.startswith("m0_table"): return "table"
    return "other"

def _geom_point_velocity(model,data,geom,point):
    import mujoco
    body=int(model.geom_bodyid[geom])
    if body==0:return np.zeros(3)
    vel=np.zeros(6,dtype=np.float64)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,int(geom),vel,0)
    return vel[3:]+np.cross(vel[:3],np.asarray(point)-data.geom_xpos[geom])

def collect_pr90_contact_frame(model,data,object_geom_name="test_cylinder_collision"):
    """Read one solved data frame; forces are equal/opposite on the two geoms."""
    import mujoco
    obj=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,object_geom_name))
    if obj<0: raise ValueError(f"missing object geom {object_geom_name}")
    ob=int(model.geom_bodyid[obj]);rows=[]
    right=np.zeros(3);left=np.zeros(3);table=np.zeros(3);other=np.zeros(3)
    local=np.zeros(6,dtype=np.float64)
    for i,c in enumerate(data.contact[:data.ncon]):
        g0,g1=int(c.geom1),int(c.geom2)
        if obj not in (g0,g1):continue
        other_geom=g1 if g0==obj else g0
        other_body=int(model.geom_bodyid[other_geom])
        body=_name(model,mujoco.mjtObj.mjOBJ_BODY,other_body)
        active=int(c.efc_address)>=0
        local.fill(0.0)
        if active:mujoco.mj_contactForce(model,data,i,local)
        frame=np.asarray(c.frame,dtype=np.float64)
        f_geom2=contact_force_world(local,frame)
        object_is_geom2=(g1==obj)
        f_obj=force_on_object(f_geom2,object_is_geom2)
        n01=frame[:3]
        n_out=(-n01 if object_is_geom2 else n01)
        force_direction=-n_out
        tangential=f_obj-float(local[0])*force_direction
        pos=np.asarray(c.pos,dtype=np.float64)
        local_pos=(pos-data.geom_xpos[obj])@data.geom_xmat[obj].reshape(3,3)
        v0=_geom_point_velocity(model,data,g0,pos);v1=_geom_point_velocity(model,data,g1,pos)
        slip=tangential_velocity_components(v1-v0,frame)
        role=_role(body)
        row={"geom0":_name(model,mujoco.mjtObj.mjOBJ_GEOM,g0),
             "geom1":_name(model,mujoco.mjtObj.mjOBJ_GEOM,g1),
             "body0":_name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g0])),
             "body1":_name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g1])),
             "role":role,"active":active,"distance_m":float(c.dist),"point_world_m":pos.tolist(),
             "point_object_local_m":local_pos.tolist(),"normal_geom0_to_geom1_world":n01.tolist(),
             "normal_outward_object_world":n_out.tolist(),"condim":int(c.dim),"contact_friction":np.asarray(c.friction,dtype=float).tolist(),"force_on_geom2_contact_frame_N":local[:3].tolist(),
             "force_on_object_world_N":f_obj.tolist(),"normal_force_N":float(local[0]) if active else 0.0,
             "tangential_force_on_object_world_N":tangential.tolist(),
             "tangential_force_magnitude_N":float(np.linalg.norm(tangential)),
             "relative_tangent_velocity_m_s":list(slip),
             "relative_slip_speed_m_s":float(math.hypot(*slip))}
        rows.append(row)
        if role in ("narrow_pad","wide_pad","right_gripper_other"):right+=f_obj
        elif role=="left_gripper":left+=f_obj
        elif role=="table":table+=f_obj
        else:other+=f_obj
    vel=np.zeros(6,dtype=np.float64)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,ob,vel,0)
    return {"simulation_time_s":float(data.time),"contact_count":len(rows),
        "object_pose_world":{"position_m":data.xpos[ob].astype(float).tolist(),
                             "quaternion_wxyz":data.xquat[ob].astype(float).tolist()},
        "object_velocity_world":{"angular_rad_s":vel[:3].tolist(),"linear_m_s":vel[3:].tolist()},
        "right_hand_support_world_N":right.tolist(),"right_hand_vertical_support_N":float(right[2]),
        "left_hand_support_world_N":left.tolist(),"left_hand_contact":any(x["role"]=="left_gripper" for x in rows),
        "table_reaction_on_object_world_N":table.tolist(),"table_reaction_vertical_N":float(table[2]),
        "other_contact_force_world_N":other.tolist(),"contacts":rows}

def summarize_pr90_contact_window(frames:Sequence[Mapping[str,Any]])->dict[str,Any]:
    """Report measured window statistics; this function imposes no acceptance thresholds."""
    n=len(frames)
    if not n:return {"frame_count":0,"bilateral_pad_contact_fraction":0.0}
    bilateral=0;max_slip=0.0;rz=[];tz=[];left=0
    start=np.asarray(frames[0]["object_pose_world"]["position_m"],dtype=float);max_xy=0.0;max_z=0.0
    for row in frames:
        active={x["role"] for x in row["contacts"] if x["active"] and x["normal_force_N"]>0.0}
        bilateral+=int({"narrow_pad","wide_pad"}<=active)
        max_slip=max(max_slip,max((float(x["relative_slip_speed_m_s"]) for x in row["contacts"]
                                   if x["role"] in ("narrow_pad","wide_pad")),default=0.0))
        rz.append(float(row["right_hand_vertical_support_N"]));tz.append(float(row["table_reaction_vertical_N"]))
        left+=int(bool(row["left_hand_contact"]))
        delta=np.asarray(row["object_pose_world"]["position_m"],dtype=float)-start
        max_xy=max(max_xy,float(np.linalg.norm(delta[:2])));max_z=max(max_z,float(delta[2]))
    return {"frame_count":n,"bilateral_pad_contact_fraction":bilateral/n,"left_hand_contact_frames":left,
        "peak_right_hand_vertical_support_N":max(rz),"minimum_right_hand_vertical_support_N":min(rz),
        "table_reaction_vertical_range_N":[min(tz),max(tz)],"max_pad_tangential_slip_speed_m_s":max_slip,
        "max_object_horizontal_drift_m":max_xy,"max_object_vertical_displacement_m":max_z,
        "claim_boundary":"Measured trace summary only; acceptance thresholds are defined by the task contract."}
