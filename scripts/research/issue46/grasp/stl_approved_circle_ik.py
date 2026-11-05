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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circle-review", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--standing-state", type=Path, required=True)
    parser.add_argument("--grasp-height-above-center-m",type=float,default=0.0)
    parser.add_argument("--runtime-handoff",action="store_true")
    parser.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
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
    perception = candidate["perception"]
    object_center = np.array(perception["estimated_center_world_m"])
    if not 0<=args.grasp_height_above_center_m<=.075:
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
            return np.r_[data.xpos[hand]-target,.1*Rotation.from_matrix(desired@data.xmat[hand].reshape(3,3).T).as_rotvec()]
        for seed in (solved,old_q):
            opt=least_squares(residual,np.clip(seed[2:],bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10,diff_step=1e-5)
            e=residual(opt.x)
            if np.linalg.norm(e[:3])+.1*np.linalg.norm(e[3:]) < metrics["hand_position_error_m"]+.01*metrics["hand_orientation_error_rad"]:
                solved=np.r_[q[:2],opt.x]
                metrics={**metrics,"solver":"bounded_least_squares_SO3","hand_position_error_m":float(np.linalg.norm(e[:3])),"hand_orientation_error_rad":float(np.linalg.norm(e[3:])/.1),"minimum_joint_margin_rad":float(np.min(np.minimum(opt.x-bounds[:,0],bounds[:,1]-opt.x)))}
        checked={}
        for opening in (min(command+10,100),command,max(command-1,0)):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,solved,np.zeros(3),0.,opening,np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            checked[str(opening)]={"pad_bodies":pads,"forbidden_object_bodies":bad,"robot_table_bodies":robot_table,"table_support":table}
        position_ok=metrics["hand_position_error_m"]<=1e-4 and metrics["hand_orientation_error_rad"]<=1e-3 and metrics["minimum_joint_margin_rad"]>=1e-3
        clear=all(not c["forbidden_object_bodies"] and not c["robot_table_bodies"] for c in checked.values())
        open_clear=not checked[str(min(command+10,100))]["pad_bodies"]
        bilateral=any(set(c["pad_bodies"])==ik.DISTALS for k,c in checked.items() if int(k)<=command)
        sweep=[]
        for fraction in np.linspace(0,1,31):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,q+(solved-q)*fraction,np.zeros(3),0.,min(command+10,100),np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            if pads or bad or robot_table:sweep.append({"fraction":float(fraction),"pads":pads,"forbidden_object":bad,"robot_table":robot_table})
        rows.append({"azimuth_rad":azimuth,"metrics":metrics,"contacts":checked,"approach_sweep_violations":sweep,"q_rad":dict(zip(ik.ACTIVE,solved.tolist())),"pass":bool(position_ok and clear and open_clear and bilateral and not sweep)})
    rows.sort(key=lambda r:(not r["pass"],r["metrics"]["hand_position_error_m"]))
    out=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="approved-circle-standing-ik").native_path
    record={"classification":"SIMULATION_ONLY_APPROVED_CIRCLE_STANDING_IK","mj_step_calls":0,"grasp_section_center_world_m":object_center.tolist(),"grasp_height_above_center_m":args.grasp_height_above_center_m,"circle_review":str(args.circle_review),"circle_review_sha256":hashlib.sha256(args.circle_review.read_bytes()).hexdigest(),"circle":match,"standing_state":str(args.standing_state),"scene":candidate["scene"],"perception":perception,"root_legs_waist_frozen":True,"commands":{"open":min(command+10,100),"match":command,"closure_probe":max(command-1,0)},"rows":rows,"gate":"PASS" if rows[0]["pass"] else "FAIL","dynamics":"NOT_RUN"}
    if args.runtime_handoff:
        best=rows[0];target_state=json.loads(json.dumps(state))
        target_state["arm_positions_rad"].update({n:best["q_rad"][n] for n in ik.ARM})
        record.update({"classification":"SIMULATION_ONLY_RUNTIME_CAPTURED_STATIC_REPLAN","initial_state":target_state,"gates":{name:record["gate"] for name in ("DOCK_CAPTURE","RUNTIME_STATIC_IK","LEFT_ARM_PARK_CLEAR","OPEN_PREGRASP_CLEAR","CLOSED_CONTACT_CLEAR")}})
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n")
    pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes())
    pg.write_artifact_manifest(out)
    print(json.dumps({"output":str(out),"output_wsl":str(out),"gate":record["gate"],"rows":rows}))


if __name__=="__main__":
    main()
