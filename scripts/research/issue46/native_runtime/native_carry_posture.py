"""Optional native-reference arm stow in the existing HAL grip owner.

This changes an operational arm pose, not the model, policy, gains or force cap.
"""
import math
import json
import time

def native_stand_arm_reference():
    from pathlib import Path
    import hashlib,yaml
    source=Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl/cpgtelecon_config.yaml")
    other=yaml.safe_load(source.read_text())["CPGTeleConConfig"]["other"]
    targets={}
    for side in ("left","right"):
        for joint,value in (("shoulder_pitch",other["default_shoulder_pitch_pos"]),("shoulder_roll",other["default_shoulder_roll_pos"]),("shoulder_yaw",0.),("elbow",other["default_elbow_pos"]),("wrist_yaw",0.),("wrist_pitch",0.),("wrist_roll",0.)):
            targets[side+"_"+joint+"_joint"]=float(value)
    walk_source=source.with_name("foundationwalk_config.yaml")
    walk=yaml.safe_load(walk_source.read_text())
    walking_neutral=dict(zip(walk["BaseConfig"]["seq"],walk["ACConfig"]["robot"]["default_dof_pos"]))
    for name in targets:
        if name.startswith("left_"):targets[name]=float(walking_neutral[name])
    # Fixed operational carry pose solved against the actual source meshes;
    # left arm remains the delivered neutral reference. It is not a policy edit.
    task_path=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/native-upright-high-stow-task-reference-20261108T233416Z/result.json")
    if hashlib.sha256(task_path.read_bytes()).hexdigest()!="f138e29388046309fce187d72ca1f10f4af040d758a52c20cc39077a4da22a42":raise ValueError("upright carry task source changed")
    task=json.loads(task_path.read_text())
    if not task["static_screen"]["pass_static"] or max(abs(x) for x in task["task_residual"])>.0001:raise ValueError("upright task geometry unqualified")
    if set(task["right_arm_rad"])!={n for n in targets if n.startswith("right_")}:raise ValueError("upright task scope")
    targets.update(task["right_arm_rad"])
    return dict(classification="NATIVE_FOUNDATIONWALK_LEFT_REFERENCE_AND_IK_UPRIGHT_RIGHT_PAYLOAD_NOT_VENDOR_CARRY_QUALIFICATION",
                source=str(walk_source),source_sha256=hashlib.sha256(walk_source.read_bytes()).hexdigest(),
                historical_stand_reference=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                operational_right_arm_ik=dict(path=str(task_path),sha256="f138e29388046309fce187d72ca1f10f4af040d758a52c20cc39077a4da22a42",source_native_frame=task["source_native_frame"],scene_sha256=task["scene_sha256"]),
                arm_target_rad=targets,duration_s=6.,carry_reference_frame="PELVIS_RIGID",physical_validation="NOT_RUN")

def perform_native_neutral_stow(n, settings):
    import os
    target=settings["arm_target_rad"]
    if set(target)!=set(LEFT+RIGHT) or settings["duration_s"]!=6.:raise RuntimeError("STOW_SCOPE")
    for name,value in target.items():
        lo,hi=LIFT_CONTRACT["retract_preflight"]["source_motion_limits"][name][:2]
        if not math.isfinite(value) or not lo+.01<=value<=hi-.01:raise RuntimeError("STOW_SOURCE_LIMIT:"+name)
    n.refresh_native();start={name:float(n.arm[name]) for name in LEFT+RIGHT};began=time.monotonic()
    for index in range(300):
        u=(index+1)/300.;u=u*u*(3.-2.*u)
        q={name:start[name]+u*(target[name]-start[name]) for name in LEFT+RIGHT}
        n.send(q,.02,"CARRY_STOW",HAND_RIGHT)
    n.send(target,1.,"CARRY_STOW",HAND_RIGHT);n.refresh_native()
    error=max(abs(n.arm[name]-target[name]) for name in LEFT+RIGHT)
    if error>.08:raise RuntimeError("NATIVE_STOW_NOT_REACHED:"+str(error))
    hand=n.native_row["bodies"]["R_omnipicker_base_link"];base=n.native_row["bodies"]["pelvis"]
    H=hand["rotation"];centre=CIRCLE_CONTRACT["circle_center_local_m"];local_axis=CIRCLE_CONTRACT["local_bottle_axis_in_hand"]
    tcp=[hand["position_m"][i]+sum(H[3*i+j]*centre[j] for j in range(3)) for i in range(3)]
    axis=[sum(H[3*i+j]*local_axis[j] for j in range(3)) for i in range(3)]
    frame=settings.get("carry_reference_frame","TORSO_RIGID")
    if frame not in ("TORSO_RIGID","PELVIS_RIGID"):raise RuntimeError("STOW_CARRY_REFERENCE_FRAME")
    torso=n.native_row["bodies"]["pelvis" if frame=="PELVIS_RIGID" else "torso_link"];T=torso["rotation"];delta=[tcp[i]-torso["position_m"][i] for i in range(3)]
    yaw=math.atan2(base["rotation"][3],base["rotation"][0]);dx=tcp[0]-base["position_m"][0];dy=tcp[1]-base["position_m"][1]
    n.lift_phase.update(carry_heading_xy_m=[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy],carry_world_z_m=tcp[2],carry_pose_frame=frame,carry_rigid_offset_base_m=[sum(T[3*i+j]*delta[i] for i in range(3)) for j in range(3)],carry_rigid_axis_base=[sum(T[3*i+j]*axis[i] for i in range(3)) for j in range(3)],carry_pose_revision=int(n.seq))
    n.ff_offset={name:0. for name in RIGHT};n.write_phase(n.lift_phase)
    deadline=time.monotonic()+1.
    while time.monotonic()<deadline:
        n.send(target,.02,"CARRY_STOW",HAND_RIGHT);n.tracking_target()
        if getattr(n,"tracking_revision",None)==n.lift_phase["carry_pose_revision"]:break
    else:raise RuntimeError("STOW_TRACKER_REVISION_UNCONFIRMED")
    result=dict(pass_result=True,classification="MEASURED_JOINT_STOW_NOT_NEW_GRASP_OR_LIFT",settings=settings,elapsed_wall_s=time.monotonic()-began,max_actual_arm_error_rad=error,generation=n.native_generation,owner_pid=os.getpid(),end_base_world_m=n.native_row["bodies"]["pelvis"]["position_m"],end_bottle_world_m=n.native_row["bodies"]["test_cylinder"]["position_m"],object_pose_commands_sent=0,force_cap_nm_unchanged=2.)
    write_durable_milestone("/workspace/evidence/native_neutral_stow_result.json",result)
    return dict(target)

def extend_native_stow(script, settings):
    import inspect
    script=script.replace("class Plan(Node):",inspect.getsource(perform_native_neutral_stow)+"\nNATIVE_STOW_SETTINGS="+repr(settings)+"\nclass Plan(Node):")
    lines=script.splitlines();out=[];in_send=False
    for i,line in enumerate(lines):
        text=line.strip()
        if text.startswith("def send("):in_send=True
        if line.startswith("rclpy.init()"):in_send=False
        if in_send:
            line=line.replace("'CARRY_HOLD')","'CARRY_HOLD','CARRY_STOW')")
            line=line.replace("sum(forces.values()) if stage=='CARRY_HOLD'","sum(forces.values()) if stage in ('CARRY_HOLD','CARRY_STOW')")
            line=line.replace("if stage=='CARRY_HOLD':","if stage in ('CARRY_HOLD','CARRY_STOW'):")
            line=line.replace("if stage=='CARRY_HOLD' and USE_NATIVE_BIAS","if stage in ('CARRY_HOLD','CARRY_STOW') and USE_NATIVE_BIAS")
            line=line.replace("'CARRY_ABORT_HOLD')","'CARRY_ABORT_HOLD','CARRY_STOW')")
        if text=="try:" and i and lines[i-1].strip()=="n.write_phase(n.lift_phase)" and i+1<len(lines) and "n.send(carry_target,.05" in lines[i+1]:
            out.append("   carry_target=perform_native_neutral_stow(n,NATIVE_STOW_SETTINGS)")
        out.append(line)
        if in_send and text=="self.refresh_native()":
            indent=" "*(len(line)-len(line.lstrip()))
            out.extend([indent+"if stage=='CARRY_STOW':",
                        indent+" fault=physical_fault(self.native_row,native_pose(self.native_row),LIFT_CONTRACT['retract_preflight']['source_motion_limits'],robot_bodies=LIFT_CONTRACT['retract_preflight']['robot_bodies'],allowed_contact_pairs=[['test_cylinder',p] for p in ('R_hand_narrow3_Link','R_hand_wide3_Link')])",
                        indent+" if fault:raise RuntimeError('STOW_PHYSICAL_SAFETY:'+fault)"])
        if "self.tracking_metrics=r.get('metrics',{})" in text:
            out.append("  self.tracking_revision=r.get('phase_revision')")
    result="\n".join(out)+"\n";compile(result,"native_reference_stow_owner","exec");return result


def screen_native_stow(model, data, settings):
    """Preserve every source geom; rigid-held transform is a planning assumption."""
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation
    ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in settings["arm_target_rad"]}
    start={n:float(data.qpos[model.jnt_qposadr[j]]) for n,j in ids.items()}
    hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"R_omnipicker_base_link")
    obj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder")
    H=data.xmat[hand].reshape(3,3).copy();rel=H.T@(data.xpos[obj]-data.xpos[hand]);relR=H.T@data.xmat[obj].reshape(3,3)
    free=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"test_cylinder_free")])
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    robot={i for i in range(model.nbody) if model.body_rootid[i]==root};rows=[]
    for fraction in np.linspace(0.,1.,21):
        for name,j in ids.items():
            q=start[name]+fraction*(settings["arm_target_rad"][name]-start[name])
            if not model.jnt_range[j,0]+.01<=q<=model.jnt_range[j,1]-.01:raise ValueError("stow source bound")
            data.qpos[model.jnt_qposadr[j]]=q
        mujoco.mj_forward(model,data);R=data.xmat[hand].reshape(3,3)
        data.qpos[free:free+3]=data.xpos[hand]+R@rel
        quat=Rotation.from_matrix(R@relR).as_quat();data.qpos[free+3:free+7]=quat[[3,0,1,2]]
        mujoco.mj_forward(model,data);bad=[]
        for c in data.contact:
            if c.dist>=-.001:continue
            a,b=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            an,bn=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in (a,b)]
            pair={an,bn}
            if pair.intersection({"test_cylinder"}) and pair.intersection({"R_hand_narrow3_Link","R_hand_wide3_Link"}):continue
            if "world" in pair:
                nonworld=an if bn=="world" else bn
                if nonworld in ("left_ankle_pitch_link","left_ankle_roll_link","right_ankle_pitch_link","right_ankle_roll_link"):continue
            if a in robot or b in robot or "test_cylinder" in pair:bad.append([an,bn,float(c.dist)])
        rows.append(dict(fraction=float(fraction),bottle_world_m=data.xpos[obj].tolist(),collisions=bad))
    return dict(pass_static=not any(r["collisions"] for r in rows),classification="SOURCE_ARM_JOINT_PATH_STATIC_ONLY",mj_step_calls=0,settings=settings,rows=rows,assumption="rigid held transform for clearance only; no runtime object pose command")
