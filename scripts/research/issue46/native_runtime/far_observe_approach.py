"""Fresh-RGBD far observation and forward/turn approach, frozen mechanics.

Research prototype: PORT-TO-C++ after physical coverage. All qpos below belong
only to independent static planning MjData; no live object or passive writes.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


def stance_proposals(anchor, root, yaw, center):
    """Five bounded task poses from sensor estimate, never scenario index/truth."""
    anchor,root,center=map(np.asarray,(anchor,root,center))
    if not np.isfinite(np.r_[anchor,root,yaw,center]).all():
        raise ValueError("finite observed pose required")
    result=[(anchor.copy(),"CURRENT_STANCE",float(yaw))]
    # Reuse the already demonstrated short forward reach correction.
    result.extend((anchor+np.array([dx,0.]),"FORWARD_ONLY_"+str(i),float(yaw))
                  for i,dx in enumerate((.08,.12)))
    # The successful current-grasp region used approximately 0.46–0.53m root-to-object
    # forward separation. Preserve its source reach, not millimetric base pose.
    poses=((.52,0.),(.48,0.),(.46,.10),(.52,-.05),(.48,.05))
    if center[1]-root[1]<-.10:
        poses=((.48,.16),(.46,.20),(.48,.10),(.52,0.))
    for i,(dx,dy) in enumerate(poses):
        target=center[:2]+np.array([-dx,dy])
        delta=target-root
        heading=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>.06 else yaw
        heading=yaw+wrap(heading-yaw)
        a=target+Rotation.from_euler("z",heading-yaw).as_matrix()[:2,:2]@(anchor-root)
        result.append((a,"FAR_DIAGONAL_"+str(i),float(heading)))
    # A travel bearing is not a required final grasp heading. Saved left-end
    # arrival rejected all old poses despite a feasible current-heading grasp.
    # Try the same bounded root targets at current yaw only as a fallback.
    for a,label,heading in list(result):
        if not label.startswith("FAR_DIAGONAL_"):continue
        target=a-Rotation.from_euler("z",heading-yaw).as_matrix()[:2,:2]@(anchor-root)
        result.append((target+anchor-root,label+"_CURRENT_HEADING",float(yaw)))
    return result


def sampled_approach_clearance(study,native,center,chosen):
    """Sample full robot along translation/rotation and include pose uncertainty.

    This is a static screening witness, not a certification of footstep sweeps.
    Actual all-body contacts, source limits and stop/reobserve remain live gates.
    """
    q=study.frame(native);q[study.obj:study.obj+7]=np.r_[center,1.,0.,0.,0.]
    study.apply(q);start=study.foot_anchor().copy()
    root=np.array(native["bodies"]["pelvis"]["position_m"][:2])
    R=native["bodies"]["pelvis"]["rotation"]
    yaw=math.atan2(R[3],R[0]);end=np.asarray(chosen["anchor_xy_m"])
    dyaw=wrap(chosen["yaw_rad"]-yaw);minimum=.05;count=0
    for u in np.linspace(0.,1.,25):
        for offset in ((0.,0.),(.05,0.),(-.05,0.),(0.,.05),(0.,-.05)):
            for error in (0.,-.15,.15):
                xy=start+u*(end-start)+offset
                screen=study.screen(study.relocate(q,xy,yaw+u*dyaw+error));count+=1
                minimum=min(minimum,screen["minimum_external_clearance_m_capped_50mm"])
                if not screen["pass"]:
                    return {"pass":False,"first_sample":count,"screen":screen,
                            "scope":"SAMPLED_FULLBODY_FORWARD_TURN_NOT_DYNAMIC_GAIT"}
    target=np.asarray(chosen["root_goal_xy_m"])
    heading_error=wrap(math.atan2(target[1]-root[1],target[0]-root[0])-yaw)
    drift_samples=0
    if target[1]-root[1]>.10 and heading_error>.35:
        # Check the source-Y strip too: coarse heading acquisition is allowed to
        # drift forward, rather than presuming translation follows the diagonal.
        for x in np.linspace(root[0],target[0],9):
            for heading in np.linspace(yaw,yaw+heading_error,9):
                for offset in ((0.,0.),(.05,0.),(-.05,0.),(0.,.05),(0.,-.05)):
                    xy=start+np.array([x-root[0],0.])+offset
                    screen=study.screen(study.relocate(q,xy,heading));count+=1;drift_samples+=1
                    minimum=min(minimum,screen["minimum_external_clearance_m_capped_50mm"])
                    if not screen["pass"]:
                        return {"pass":False,"first_sample":count,"screen":screen,"scope":"COARSE_HEADING_FORWARD_DRIFT_STRIP_NOT_DYNAMIC_GAIT"}
    return {"pass":True,"samples":count,"coarse_heading_drift_samples":drift_samples,"minimum_clearance_m_capped_50mm":minimum,
            "source_root_xy_m":root.tolist(),"scope":"SAMPLED_FULLBODY_FORWARD_TURN_NOT_DYNAMIC_GAIT",
            "outer_bounds":{"x":[float(min(root[0],target[0])-.16),float(max(root[0],target[0])+.12)],
                            "y":[float(min(root[1],target[1])-.16),float(max(root[1],target[1])+.16)],
                            "yaw":[float(min(yaw,yaw+dyaw)-.35),float(max(yaw,yaw+dyaw)+.35)]},
            "xy_uncertainty_samples_m":.05,"heading_uncertainty_samples_rad":.15}



def intermediate_forward_waypoint(origin, terminal):
    """A transit waypoint is not a grasp witness; reobserve after the real stop.

    Saved STEP starts traveled 190mm for a120mm request near the table. Split
    these forward-only corrections into the existing bounded40mm pulse. The
    full target's grasp witness is retained separately, never assigned here.
    """
    origin,terminal=map(lambda v:np.asarray(v,dtype=float),(origin,terminal))
    if origin.shape!=(2,) or terminal.shape!=(2,) or not np.isfinite(np.r_[origin,terminal]).all():
        raise ValueError("finite XY transit and terminal required")
    dx,dy=terminal-origin
    if .06 < dx <= .16 and abs(dy)<.08:
        return origin+np.array([.04,0.])
    return None


def forward_turn_command(pose,goal,linear,angular):
    """Compose existing heading controller with documented forward+turn input."""
    # No unloaded zero-forward pivot near the table: STEP pivots in the saved
    # run translated forward 0.48m. Keep forward demand explicit and bounded.
    dx,dy=goal[0]-pose["x"],goal[1]-pose["y"]
    heading=wrap(math.atan2(dy,dx)-pose["yaw"])
    # Right-side endpoint needs heading acquisition before forward space is used.
    # Reuse the already exercised .60rad/s command envelope only when the
    # sensed task target is >10cm to the right; no source/physical limit change.
    right_cap=.60 if dy<-.10 else .40
    if dy>.10 and heading>.35:
        # Reuse PointGoalController's coarse heading-first behavior in the open
        # approach. A STEP pivot can drift; the checked drift strip and existing
        # physical/predictive-stop gates remain mandatory. Never claim a fixed
        # rotation centre, and do not require exact final grasp yaw.
        return (0.,0.,min(.60,.9*heading))
    return (max(.12,min(.20,linear)),0.,max(-right_cap,min(.40,.9*heading)))


def forward_table_plane_reached(pose,goal,origin,*,delay_s=.80):
    """Do not chase lateral error by driving beyond the checked forward station."""
    if abs(goal[1]-origin[1])<=.15:return False
    return pose["x"]+delay_s*max(0.,pose["vx"])>=goal[0]-.02


def stopping_distance(pose,goal):
    """Existing 0.8s forward-braking estimate; actual stop must requalify IK."""
    dx,dy=goal[0]-pose["x"],goal[1]-pose["y"]
    norm=math.hypot(dx,dy)
    speed=0. if norm<1e-8 else max(0.,(pose["vx"]*dx+pose["vy"]*dy)/norm)
    return .06+.80*speed


def preparation_clearance(study,native,center,chosen):
    """Mirror actual elbow/lower/waist sequence, not old prepared screenshots."""
    import json,mujoco
    from pathlib import Path
    from scripts.research.issue46.native_runtime.native_task_grasp_region import reference_contract
    proposal=json.loads(Path(reference_contract()["posture_reference"]).read_text())
    proposal["pre_raise_forearm_target_rad"]=chosen.get("pre_raise_forearm_target_rad",proposal["pre_raise_forearm_target_rad"])
    lower=json.loads(Path(proposal["native_lowering_reference"]["path"]).read_text())
    q=study.frame(native);q[study.obj:study.obj+7]=np.r_[center,1.,0.,0.,0.]
    study.apply(q);current_yaw=math.atan2(native["bodies"]["pelvis"]["rotation"][3],native["bodies"]["pelvis"]["rotation"][0])
    base=study.relocate(q,np.asarray(chosen["anchor_xy_m"]),chosen["yaw_rad"])
    base[study.obj:study.obj+7]=q[study.obj:study.obj+7]
    m=study.model;ids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in native["joints"]}
    elbow="right_elbow_joint";legs=[n for n in ids if any(k in n for k in ("hip_","knee_","ankle_"))]
    waist=dict(proposal["waist_target_rad"],waist_roll_joint=native["joints"]["waist_roll_joint"]["q"])
    waist["waist_yaw_joint"]=wrap(chosen["torso_world_yaw_target_rad"]-chosen["yaw_rad"])
    waist["waist_pitch_joint"]=chosen.get("waist_pitch_target_rad",waist["waist_pitch_joint"])
    feet={"left_ankle_pitch_link","left_ankle_roll_link","right_ankle_pitch_link","right_ankle_roll_link"}
    minimum=.05
    for stage in ("elbow","lower","waist"):
        for u in np.linspace(0.,1.,21):
            v=base.copy();a=int(m.jnt_qposadr[ids[elbow]])
            v[a]+=(u if stage=="elbow" else 1.)*(proposal["pre_raise_forearm_target_rad"]-v[a])
            if stage in ("lower","waist"):
                t=u if stage=="lower" else 1.;v[study.root+2]-=.10*t
                for n in legs:
                    a=int(m.jnt_qposadr[ids[n]]);v[a]+=t*(lower["joints"][n]["q"]-v[a])
            if stage=="waist":
                for n,target in waist.items():
                    a=int(m.jnt_qposadr[ids[n]]);v[a]+=u*(target-v[a])
            study.apply(v)
            for c in study.data.contact:
                if c.dist>=-.001:continue
                ba,bb=[int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)];names=[study.names[ba],study.names[bb]]
                if not (ba in study.robot or bb in study.robot):continue
                if "world" in names and any(n in feet for n in names):continue
                if set(names)=={"pelvis","left_hip_roll_link"} and c.dist>=-.005:continue
                return {"pass":False,"stage":stage,"fraction":float(u),"pair":names,"distance_m":float(c.dist),"scope":"ACTUAL_ARM_PREPARATION_STATIC"}
    return {"pass":True,"samples":63,"scope":"ACTUAL_ARM_PREPARATION_STATIC"}
