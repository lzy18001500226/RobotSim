"""Source-preserving approved support-circle adapter for existing standing IK.

No model edits or dynamics. Frozen standing root/legs/waist are retained.
The resulting gate must pass before handing any target to live arm control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime import standing_side_grasp_static_search as ik


def circle_target(object_center, orientation, circle_center):
    return np.asarray(object_center) - np.asarray(orientation) @ np.asarray(circle_center)


def axis_grasp_error(hand_position, hand_rotation, local_circle_center, section_center, bottle_axis):
    """Five independent task constraints: centre XYZ and directed axis tilt.

    Roll around the bottle axis is free. Use directed vector subtraction,
    rather than a cross product that also vanishes for an inverted axis.
    """
    return np.r_[np.asarray(hand_position)+np.asarray(hand_rotation)@np.asarray(local_circle_center)-section_center,
                 .1*(np.asarray(hand_rotation)[:,0]-np.asarray(bottle_axis))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circle-review", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--standing-state", type=Path, required=True)
    parser.add_argument("--grasp-height-above-center-m",type=float,default=0.0)
    parser.add_argument("--runtime-handoff",action="store_true")
    parser.add_argument("--free-bottle-axis-roll",action="store_true",help="constrain reviewed bottle axis and circle centre, letting IK choose axial roll")
    parser.add_argument("--target-source",choices=("historical-rgbd-replay","truth-diagnostic"),default="historical-rgbd-replay",help="explicit target provenance; neither mode is fresh runtime RGB-D")
    parser.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument("--waist-pitch-offset-rad",type=float,default=0.,help="offline source-limited forward-bend candidate; nonzero requires a separately supported waist control handoff")
    args = parser.parse_args()
    scene = json.loads(args.circle_review.read_text())
    candidate = json.loads(args.candidate.read_text())
    state = json.loads(args.standing_state.read_text())
    base_offset=np.asarray(args.base_offset_xyz_m,dtype=float)
    if base_offset.shape!=(3,) or not np.isfinite(base_offset).all(): raise ValueError("base offset must be finite")
    state=json.loads(json.dumps(state))
    state["base_pose_world"]["xyz_m"]=(np.asarray(state["base_pose_world"]["xyz_m"])+base_offset).tolist()
    samples = scene["circle_review"]["samples"]
    if scene["circle_review"]["classification"] != "STATIC_VARIABLE_DIAMETER_SUPPORT_CIRCLE_REVIEW_ONLY":
        raise ValueError("expected approved variable-diameter support-circle record")
    source = Path(candidate["scene"]["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != candidate["scene"]["sha256"]:
        raise ValueError("source scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(source))
    q = np.array([state["waist_positions_rad"][n] for n in ("waist_yaw_joint","waist_pitch_joint")]+[state["arm_positions_rad"][n] for n in ik.ARM])
    approach_start=q.copy()
    if not np.isfinite(args.waist_pitch_offset_rad):raise ValueError("waist offset must be finite")
    q[1]+=args.waist_pitch_offset_rad
    limits=model.jnt_range[ik.native.jid(model,"waist_pitch_joint")]
    if not limits[0]+1e-3<=q[1]<=limits[1]-1e-3:raise ValueError("planned waist pitch violates unchanged source joint limits")
    if args.runtime_handoff and args.waist_pitch_offset_rad!=0:
        raise ValueError("nonzero waist bend is offline-only until exclusive waist-control handoff is implemented")
    perception = candidate["perception"]
    object_center = np.array(perception["estimated_center_world_m"])
    if args.target_source == "truth-diagnostic":
        source_data=mujoco.MjData(model)
        mujoco.mj_forward(model,source_data)
        object_center=source_data.xpos[ik.native.bid(model,"test_cylinder")].copy()
        perception={"source":"MUJOCO_TRUTH_DIAGNOSTIC_ONLY","object_center_world_m":object_center.tolist(),"scene_sha256":candidate["scene"]["sha256"]}
    else:
        perception={**perception,"source":"HISTORICAL_RGBD_REPLAY_NOT_CURRENT_EPISODE","fresh_capture":False}
    # Canonical bottle half-height=150mm; the wide pad spans roughly +/-30mm
    # along the bottle axis. A 100mm section offset retains >=20mm top margin.
    # This is a task grasp-height choice, not a joint/model/safety-limit edit.
    if not 0<=args.grasp_height_above_center_m<=.10:
        raise ValueError("grasp section height must be within the canonical bottle body")
    object_center[2]+=args.grasp_height_above_center_m
    # The canonical existing fixture is 70mm diameter. Match its circumference
    # with the nearest approved opening, then inspect adjacent closure commands.
    match = min(samples,key=lambda s:abs(s["diameter_mm"]-70))
    command = match["opening_percent"]
    if match["status"] != "FOUR_BANDS_NEAR_0_5MM":
        raise ValueError("canonical diameter has no qualified support distribution")
    center = np.array(match["center_m"])
    # Bottle frame +Z is its upright axis. Set gripper +X exactly to it,
    # preserving the old reference roll as closely as possible.
    x = np.array([0.,0.,1.])
    y = ik.REFERENCE_HAND_ROTATION[:,1].copy(); y-=x*np.dot(x,y); y/=np.linalg.norm(y)
    rotation = np.column_stack([x,y,np.cross(x,y)])
    ik.REFERENCE_HAND_ROTATION = rotation
    ik.REFERENCE_HAND_POSITION = -rotation @ center
    rows=[]
    # Reuse the previously converged arm seed, not only the neutral standing
    # arm. Test three rolls about the bottle axis without moving the base.
    old_q=np.array([candidate["search"]["candidate"]["upper_q_rad"][n] for n in ik.ACTIVE])
    old_q[:2]=q[:2]
    for azimuth in (float(candidate["search"]["candidate"]["grasp_azimuth_rad"]),float(candidate["search"]["candidate"]["grasp_azimuth_rad"])-0.8,float(candidate["search"]["candidate"]["grasp_azimuth_rad"])+0.8):
        attempts=[ik.solve(model,state,np.zeros(3),0.,command,seed.copy(),np.zeros(3),azimuth,np.zeros(3),0.,0.,None,object_center) for seed in (q,old_q)]
        solved,metrics=min(attempts,key=lambda item:item[1]["hand_position_error_m"]+item[1]["hand_orientation_error_rad"]*.1)
        # Independently verify a stuck Jacobian solution using bounded nonlinear
        # least squares and the full SO(3) rotation error (no 180deg ambiguity).
        data=mujoco.MjData(model)
        hand=ik.native.bid(model,"R_omnipicker_base_link")
        cylinder=ik.native.bid(model,"test_cylinder")
        bounds=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ARM])
        def residual(arm):
            trial=np.r_[q[:2],arm]
            ik.set_pose(model,data,state,trial,np.zeros(3),0.,command,np.zeros(3),0.,None)
            c,s=np.cos(azimuth),np.sin(azimuth)
            around=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            desired=data.xmat[cylinder].reshape(3,3)@around@rotation
            target=circle_target(object_center,desired,center)
            if args.free_bottle_axis_roll:
                return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,object_center,data.xmat[cylinder].reshape(3,3)[:,2])
            return np.r_[data.xpos[hand]-target,.1*Rotation.from_matrix(desired@data.xmat[hand].reshape(3,3).T).as_rotvec()]
        for seed in (solved,old_q):
            opt=least_squares(residual,np.clip(seed[2:],bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10,diff_step=1e-5)
            e=residual(opt.x)
            if np.linalg.norm(e[:3])+.1*np.linalg.norm(e[3:]) < metrics["hand_position_error_m"]+.01*metrics["hand_orientation_error_rad"]:
                solved=np.r_[q[:2],opt.x]
                angle=float(2*np.arcsin(np.clip(np.linalg.norm(e[3:])/.2,0,1))) if args.free_bottle_axis_roll else float(np.linalg.norm(e[3:])/.1)
                metrics={**metrics,"solver":"bounded_least_squares_directed_axis" if args.free_bottle_axis_roll else "bounded_least_squares_SO3","hand_position_error_m":float(np.linalg.norm(e[:3])),"hand_orientation_error_rad":angle,"orientation_constraint":"directed_bottle_axis_free_axial_roll" if args.free_bottle_axis_roll else "full_rotation","minimum_joint_margin_rad":float(np.min(np.minimum(opt.x-bounds[:,0],bounds[:,1]-opt.x)))}
        checked={}
        for opening in (min(command+10,100),command,max(command-1,0)):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,solved,np.zeros(3),0.,opening,np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            checked[str(opening)]={"pad_bodies":pads,"forbidden_object_bodies":bad,"robot_table_bodies":robot_table,"table_support":table}
        final_check=checked[str(command)]
        metrics.update({"pad_bodies":final_check["pad_bodies"],"nonpad_cylinder_bodies":final_check["forbidden_object_bodies"],"robot_table_bodies":final_check["robot_table_bodies"],"table_support":final_check["table_support"]})
        metrics["mc_owned_waist_fixed"]=args.waist_pitch_offset_rad==0
        position_ok=metrics["hand_position_error_m"]<=1e-4 and metrics["hand_orientation_error_rad"]<=1e-3 and metrics["minimum_joint_margin_rad"]>=1e-3
        clear=all(not c["forbidden_object_bodies"] and not c["robot_table_bodies"] for c in checked.values())
        open_clear=not checked[str(min(command+10,100))]["pad_bodies"]
        bilateral=any(set(c["pad_bodies"])==ik.DISTALS for k,c in checked.items() if int(k)<=command)
        sweep=[]
        for fraction in np.linspace(0,1,31):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,approach_start+(solved-approach_start)*fraction,np.zeros(3),0.,min(command+10,100),np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            if pads or bad or robot_table:sweep.append({"fraction":float(fraction),"pads":pads,"forbidden_object":bad,"robot_table":robot_table})
        row_gates={"RUNTIME_STATIC_IK":"PASS" if position_ok else "FAIL","LEFT_ARM_PARK_CLEAR":"PASS" if all(not any(n.startswith('L_') or n.startswith('left_') for n in c["robot_table_bodies"]+c["forbidden_object_bodies"]) for c in checked.values()) else "FAIL","OPEN_PREGRASP_CLEAR":"PASS" if open_clear and not checked[str(min(command+10,100))]["forbidden_object_bodies"] and not checked[str(min(command+10,100))]["robot_table_bodies"] else "FAIL","CLOSED_CONTACT_CLEAR":"PASS" if bilateral and clear else "FAIL","APPROACH_SWEEP":"PASS" if not sweep else "FAIL"}
        rows.append({"azimuth_rad":azimuth,"metrics":metrics,"contacts":checked,"gates":row_gates,"approach_sweep_violations":sweep,"q_rad":dict(zip(ik.ACTIVE,solved.tolist())),"pass":bool(position_ok and clear and open_clear and bilateral and not sweep)})
    # Among truly passing candidates, prefer source-limit margin rather than
    # nanometre differences in numerical convergence residuals.
    rows.sort(key=lambda r:(not r["pass"],-r["metrics"]["minimum_joint_margin_rad"] if r["pass"] else r["metrics"]["hand_position_error_m"],r["metrics"]["hand_position_error_m"]))
    out=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="approved-circle-standing-ik").native_path
    record={"classification":"SIMULATION_ONLY_APPROVED_CIRCLE_STANDING_IK","mj_step_calls":0,"grasp_section_center_world_m":object_center.tolist(),"grasp_height_above_center_m":args.grasp_height_above_center_m,"circle_review":str(args.circle_review),"circle_review_sha256":hashlib.sha256(args.circle_review.read_bytes()).hexdigest(),"circle":match,"standing_state":str(args.standing_state),"scene":candidate["scene"],"perception":perception,"root_legs_waist_frozen":True,"commands":{"open":min(command+10,100),"match":command,"closure_probe":max(command-1,0)},"rows":rows,"gate":"PASS" if rows[0]["pass"] else "FAIL","dynamics":"NOT_RUN"}
    record["target_source"]=args.target_source
    record["free_bottle_axis_roll"]=args.free_bottle_axis_roll
    record["waist_pitch_offset_rad"]=args.waist_pitch_offset_rad
    record["waist_actuation_required"]=args.waist_pitch_offset_rad!=0
    record["root_legs_waist_frozen"]=args.waist_pitch_offset_rad==0
    record["root_and_legs_frozen"]=True
    record["fresh_runtime_rgbd_verified"]=False
    if args.runtime_handoff:
        best=rows[0];target_state=json.loads(json.dumps(state))
        target_state["arm_positions_rad"].update({n:best["q_rad"][n] for n in ik.ARM})
        record.update({"classification":"SIMULATION_ONLY_RUNTIME_CAPTURED_STATIC_REPLAN","initial_state":target_state,"gates":{"DOCK_CAPTURE":"PASS" if state.get("motion_stability",{}).get("pass") is True else "FAIL",**best["gates"]}})
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n")
    pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes())
    pg.write_artifact_manifest(out)
    print(json.dumps({"output":str(out),"output_wsl":str(out),"gate":record["gate"],"rows":rows}))


if __name__=="__main__":
    main()
