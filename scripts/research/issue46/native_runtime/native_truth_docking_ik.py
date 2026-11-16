#!/usr/bin/env python3
"""Solve a truth-pose STL pregrasp from an observed native standing state."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory  # noqa: E402


ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
UPPER = WAIST + ARM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jid(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise ValueError(f"scene lacks joint {name}")
    return value


def bid(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise ValueError(f"scene lacks body {name}")
    return value


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = jid(model, name)
    data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[model.jnt_dofadr[joint]] = 0.0


def set_native_state(model: mujoco.MjModel, data: mujoco.MjData, state: dict[str, object], upper: np.ndarray, command: float, base_offset: np.ndarray, base_yaw_offset_rad: float, base_rpy_override: np.ndarray | None) -> None:
    free = jid(model, "floating_base_joint")
    base = state["base_pose_world"]
    assert isinstance(base, dict)
    qadr = model.jnt_qposadr[free]
    data.qpos[qadr:qadr + 3] = np.asarray(base["xyz_m"], dtype=float) + base_offset
    quaternion = np.asarray(base["quaternion_wxyz"], dtype=float)
    half = 0.5 * base_yaw_offset_rad
    yaw = np.array((np.cos(half), 0.0, 0.0, np.sin(half)))
    w1, x1, y1, z1 = yaw
    w2, x2, y2, z2 = quaternion / np.linalg.norm(quaternion)
    if base_rpy_override is None:
        data.qpos[qadr + 3:qadr + 7] = (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
    else:
        override_quaternion = np.empty(4)
        mujoco.mju_euler2Quat(override_quaternion, base_rpy_override, "XYZ")
        data.qpos[qadr + 3:qadr + 7] = override_quaternion
    for group_name in ("leg_positions_rad", "waist_positions_rad", "arm_positions_rad"):
        group = state[group_name]
        assert isinstance(group, dict)
        for name, value in group.items():
            set_joint(model, data, name, float(value))
    for name, value in zip(UPPER, upper):
        set_joint(model, data, name, float(value))
    trajectory = load_registered_trajectory(REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic")
    right_root = "right_claw_joint" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint") >= 0 else "R_claw_joint"
    left_root = "left_claw_joint" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_claw_joint") >= 0 else "L_claw_joint"
    pose = trajectory.pose(command)
    for side in ("right", "left"):
        for name, value in pose.positions_rad.items():
            target = (right_root if side == "right" else left_root) if name == "right_claw_joint" else name if side == "right" else "L_" + name[2:]
            set_joint(model, data, target, value)
    mujoco.mj_forward(model, data)


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body: int, dofs: np.ndarray) -> np.ndarray:
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, linear, angular, body)
    return linear[:, dofs]


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    rows=[]
    for contact in data.contact:
        bodies=tuple(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "world" for g in (contact.geom1,contact.geom2))
        rows.append({"bodies":bodies,"distance_m":float(contact.dist)})
    return rows


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small-angle world-frame error that rotates ``current`` toward ``target``."""
    return 0.5 * (np.cross(current[:, 0], target[:, 0]) + np.cross(current[:, 1], target[:, 1]) + np.cross(current[:, 2], target[:, 2]))


def accepted_stand_state(payload: dict[str, object]) -> tuple[dict[str, object], str]:
    """Accept either the vendor-MC capture or a measured FoundationWalk settle.

    The latter is deliberately accepted only after a real stepped standing
    result passed its full-contact gate.  It prevents a static arm solve from
    silently using a different waist/base pose than the physical controller.
    """
    docked = payload.get("native_dock_state_capture")
    gates = payload.get("gates")
    if (
        isinstance(docked, dict)
        and isinstance(gates, dict)
        and all(gates.get(name) == "PASS" for name in (
            "NATIVE_TABLE_DOCK_ARRIVAL",
            "NATIVE_TABLE_DOCK_STOP_HOLD",
            "NATIVE_TABLE_DOCK_STATE_HANDOFF",
        ))
    ):
        return docked, "NATIVE_MC_TABLE_DOCK_CAPTURE"
    native = payload.get("native_arm_state_capture")
    if isinstance(native, dict):
        return native, "NATIVE_MC_STAND_CAPTURE"
    stabilized = payload.get("stabilized_state")
    if isinstance(stabilized, dict) and isinstance(gates, dict) and gates.get("FULL_CONTACT_STAND") == "PASS":
        return stabilized, "FOUNDATIONWALK_STEPPED_STAND_CAPTURE"
    raise ValueError("stand result lacks an accepted native or stepped FoundationWalk state")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-overlay",required=True,type=Path)
    parser.add_argument("--native-stand",required=True,type=Path)
    parser.add_argument("--open-command-percent",type=int,default=100)
    parser.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.0,0.0,0.0))
    parser.add_argument("--base-yaw-offset-rad",type=float,default=0.0)
    parser.add_argument("--base-rpy-override-rad",type=float,nargs=3)
    parser.add_argument("--target-offset-xyz-m",type=float,nargs=3,default=(0.0,0.0,0.0))
    parser.add_argument("--lock-waist-roll",action="store_true")
    parser.add_argument("--lock-waist",action="store_true")
    parser.add_argument("--max-waist-deviation-rad",type=float)
    parser.add_argument("--reference-hand-relative-pos-m",type=float,nargs=3)
    parser.add_argument("--reference-hand-relative-rot",type=float,nargs=9)
    parser.add_argument("--reference-hand-yaw-offset-rad",type=float,default=0.0)
    args=parser.parse_args()
    table=json.loads(args.table_overlay.read_text(encoding="utf-8"))
    native=json.loads(args.native_stand.read_text(encoding="utf-8"))
    if table.get("gates",{}).get("TABLE_SCENE_STATIC_COMPILE")!="PASS":
        raise ValueError("table evidence is not passing")
    state, stand_state_source = accepted_stand_state(native)
    scene=Path(table["generated"]["host_scene"]["path"])
    if not scene.is_file() or sha256(scene)!=table["generated"]["host_scene"]["sha256"]:
        raise ValueError("table host scene hash mismatch")
    orient=load_module("stl_orientation",REPO / "scripts/research/issue46/grasp/centered_orientation_search.py")
    base_offset=np.asarray(args.base_offset_xyz_m,dtype=float)
    if base_offset.shape!=(3,) or not np.isfinite(base_offset).all():
        raise ValueError("base offset must be a finite three-vector")
    if not np.isfinite(args.base_yaw_offset_rad):
        raise ValueError("base yaw offset must be finite")
    base_rpy_override=None if args.base_rpy_override_rad is None else np.asarray(args.base_rpy_override_rad,dtype=float)
    if base_rpy_override is not None and (base_rpy_override.shape!=(3,) or not np.isfinite(base_rpy_override).all()):
        raise ValueError("base RPY override must be finite")
    if args.max_waist_deviation_rad is not None and (not np.isfinite(args.max_waist_deviation_rad) or not 0.0 <= args.max_waist_deviation_rad <= 1.0):
        raise ValueError("max waist deviation must lie in 0..1 rad")
    target_offset=np.asarray(args.target_offset_xyz_m,dtype=float)
    if target_offset.shape!=(3,) or not np.isfinite(target_offset).all():
        raise ValueError("target offset must be a finite three-vector")
    if (args.reference_hand_relative_pos_m is None) != (args.reference_hand_relative_rot is None):
        raise ValueError("reference hand position and rotation must be supplied together")
    reference_hand = None
    if args.reference_hand_relative_pos_m is not None:
        reference_pos = np.asarray(args.reference_hand_relative_pos_m, dtype=float)
        reference_rot = np.asarray(args.reference_hand_relative_rot, dtype=float).reshape(3, 3)
        if not np.isfinite(reference_pos).all() or not np.isfinite(reference_rot).all() or not np.allclose(reference_rot.T @ reference_rot, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(reference_rot), 1.0, atol=1e-6):
            raise ValueError("reference hand transform is not a finite right-handed rotation")
        if not np.isfinite(args.reference_hand_yaw_offset_rad):
            raise ValueError("reference hand yaw offset must be finite")
        cosine, sine = np.cos(args.reference_hand_yaw_offset_rad), np.sin(args.reference_hand_yaw_offset_rad)
        around = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
        reference_hand = (around @ reference_pos, around @ reference_rot)
    model=mujoco.MjModel.from_xml_path(str(scene))
    if args.lock_waist:
        active_indices=np.asarray(list(range(len(WAIST),len(UPPER))),dtype=int)
    elif args.lock_waist_roll:
        active_indices=np.asarray([0,1,*range(3,len(UPPER))],dtype=int)
    else:
        active_indices=np.asarray(list(range(len(UPPER))),dtype=int)
    active_names=tuple(UPPER[index] for index in active_indices)
    dofs=np.asarray([model.jnt_dofadr[jid(model,n)] for n in active_names],dtype=int)
    bounds=np.asarray([model.jnt_range[jid(model,n)] for n in active_names],dtype=float)
    source_arm=state["arm_positions_rad"]
    source_waist=state["waist_positions_rad"]
    assert isinstance(source_arm,dict) and isinstance(source_waist,dict)
    q=np.asarray([float(source_waist[n]) for n in WAIST]+[float(source_arm[n]) for n in ARM],dtype=float)
    seed=q.copy()
    if args.max_waist_deviation_rad is not None:
        for local_index, name in enumerate(WAIST):
            joint_bounds = model.jnt_range[jid(model, name)]
            lower = max(float(joint_bounds[0]), q[local_index] - args.max_waist_deviation_rad)
            upper = min(float(joint_bounds[1]), q[local_index] + args.max_waist_deviation_rad)
            active_row = int(np.where(active_indices == local_index)[0][0]) if local_index in active_indices else None
            if active_row is not None:
                bounds[active_row] = (lower, upper)
    elbow,wrist,narrow,wide,cylinder,hand=(bid(model,n) for n in ("right_elbow_link","right_wrist_roll_link","R_hand_narrow3_Link","R_hand_wide3_Link","test_cylinder","R_omnipicker_base_link"))
    for iteration in range(1,401):
        data=mujoco.MjData(model);set_native_state(model,data,state,q,args.open_command_percent,base_offset,args.base_yaw_offset_rad,base_rpy_override)
        if reference_hand is None:
            centre_error=data.xpos[cylinder]+target_offset-orient.jaw_center(model,data)
            forearm_delta=float(data.xpos[wrist,2]-data.xpos[elbow,2])
            distal_delta=float(data.xpos[narrow,2]-data.xpos[wide,2])
            error=np.r_[centre_error,-forearm_delta,-distal_delta]
            elbow_jac=point_jacobian(model,data,elbow,dofs);wrist_jac=point_jacobian(model,data,wrist,dofs);narrow_jac=point_jacobian(model,data,narrow,dofs);wide_jac=point_jacobian(model,data,wide,dofs)
            jacobian=np.vstack((orient.jaw_jacobian(model,data)[:,dofs],wrist_jac[2]-elbow_jac[2],narrow_jac[2]-wide_jac[2]))
        else:
            rel_pos, rel_rot = reference_hand
            cylinder_rot = data.xmat[cylinder].reshape(3, 3)
            desired_pos = data.xpos[cylinder] + cylinder_rot @ rel_pos + target_offset
            desired_rot = cylinder_rot @ rel_rot
            current_rot = data.xmat[hand].reshape(3, 3)
            centre_error = desired_pos - data.xpos[hand]
            orientation_error = rotation_error(current_rot, desired_rot)
            forearm_delta=0.0; distal_delta=0.0
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, linear, angular, hand)
            error=np.r_[centre_error,orientation_error]
            jacobian=np.vstack((linear[:,dofs], angular[:,dofs]))
        margin=np.min(np.minimum(q[active_indices]-bounds[:,0],bounds[:,1]-q[active_indices]))
        if np.linalg.norm(centre_error)<=1e-4 and (reference_hand is not None or (abs(forearm_delta)<=1e-4 and abs(distal_delta)<=1e-4)) and margin>=1e-3 and (reference_hand is None or np.linalg.norm(orientation_error)<=1e-3):
            break
        inverse=np.linalg.solve(jacobian@jacobian.T+1e-5*np.eye(len(error)),np.eye(len(error)))
        delta=jacobian.T@inverse@error
        delta-=2e-4*(q[active_indices]-seed[active_indices])
        # Use redundant joint motion to retreat from range endpoints after
        # satisfying the Cartesian constraints, rather than accepting a
        # visually plausible pose that cannot tolerate physical tracking.
        low_gap=q[active_indices]-bounds[:,0]
        high_gap=bounds[:,1]-q[active_indices]
        endpoint_bias=0.001*(1.0/(low_gap+0.02)-1.0/(high_gap+0.02))
        nullspace=np.eye(len(active_indices))-jacobian.T@inverse@jacobian
        delta+=nullspace@np.clip(endpoint_bias,-0.02,0.02)
        q[active_indices]+=np.clip(delta,-0.025,0.025)
        q[active_indices]=np.clip(q[active_indices],bounds[:,0]+1e-5,bounds[:,1]-1e-5)
    data=mujoco.MjData(model);set_native_state(model,data,state,q,args.open_command_percent,base_offset,args.base_yaw_offset_rad,base_rpy_override)
    all_bounds=np.asarray([model.jnt_range[jid(model,n)] for n in UPPER],dtype=float)
    if reference_hand is None:
        final_position_error=float(np.linalg.norm(data.xpos[cylinder]+target_offset-orient.jaw_center(model,data)))
        final_orientation_error=None
        final_forearm=float(data.xpos[wrist,2]-data.xpos[elbow,2]); final_distal=float(data.xpos[narrow,2]-data.xpos[wide,2])
    else:
        rel_pos, rel_rot=reference_hand; cylinder_rot=data.xmat[cylinder].reshape(3,3); desired_pos=data.xpos[cylinder]+cylinder_rot@rel_pos+target_offset; desired_rot=cylinder_rot@rel_rot
        final_position_error=float(np.linalg.norm(desired_pos-data.xpos[hand])); final_orientation_error=float(np.linalg.norm(rotation_error(data.xmat[hand].reshape(3,3),desired_rot))); final_forearm=None; final_distal=None
    metrics={"iterations":iteration,"jaw_center_error_m":final_position_error,"hand_orientation_error_rad":final_orientation_error,"forearm_height_delta_m":final_forearm,"distal_height_delta_m":final_distal,"min_joint_margin_rad":float(np.min(np.minimum(q-all_bounds[:,0],all_bounds[:,1]-q)))}
    contacts=contact_records(model,data)
    bad_table=[row for row in contacts if "raicom_table_table" in row["bodies"] and "test_cylinder" not in row["bodies"]]
    cylinder_nonpad=[row for row in contacts if "test_cylinder" in row["bodies"] and not any(name in {"R_hand_narrow3_Link","R_hand_wide3_Link","raicom_table_table"} for name in row["bodies"])]
    aligned = metrics["jaw_center_error_m"]<=1e-4 and (reference_hand is not None and metrics["hand_orientation_error_rad"] is not None and metrics["hand_orientation_error_rad"]<=1e-3 or reference_hand is None and metrics["forearm_height_delta_m"] is not None and abs(metrics["forearm_height_delta_m"])<=1e-4 and metrics["distal_height_delta_m"] is not None and abs(metrics["distal_height_delta_m"])<=1e-4)
    gates={"NATIVE_STAND_INPUT":"PASS","TRUTH_IK_ALIGNMENT":"PASS" if aligned else "FAIL","JOINT_LIMIT_MARGIN":"PASS" if metrics["min_joint_margin_rad"]>=1e-3 else "FAIL","TABLE_CLEARANCE":"PASS" if not bad_table else "FAIL","OPEN_PREGRASP_NONPAD_CLEAR":"PASS" if not cylinder_nonpad else "FAIL","DYNAMICS":"NOT_RUN"}
    output=pg.create_unique_output_directory(EVIDENCE/"reviews/issue-46",prefix="raicom-native-truth-docking-ik").native_path
    result={"schema":"robotsim.issue46.raicom-native-truth-docking-ik.v1","classification":"SIMULATION_ONLY_STATIC_NATIVE_STAND_TRUTH_IK","mj_step_calls":0,"table_overlay":{"path":str(args.table_overlay),"sha256":sha256(args.table_overlay)},"native_stand":{"path":str(args.native_stand),"sha256":sha256(args.native_stand),"state_source":stand_state_source},"scene":{"path":str(scene),"sha256":sha256(scene)},"base_offset_xyz_m":base_offset.tolist(),"base_yaw_offset_rad":args.base_yaw_offset_rad,"base_rpy_override_rad":None if base_rpy_override is None else base_rpy_override.tolist(),"target_offset_xyz_m":target_offset.tolist(),"reference_hand_relative_to_cylinder":None if reference_hand is None else {"position_m":reference_hand[0].tolist(),"rotation_matrix":reference_hand[1].tolist()},"reference_hand_yaw_offset_rad":args.reference_hand_yaw_offset_rad,"lock_waist_roll":args.lock_waist_roll,"lock_waist":args.lock_waist,"max_waist_deviation_rad":args.max_waist_deviation_rad,"waist_q_rad":{n:float(v) for n,v in zip(WAIST,q[:len(WAIST)])},"right_arm_q_rad":{n:float(v) for n,v in zip(ARM,q[len(WAIST):])},"open_command_percent":args.open_command_percent,"metrics":metrics,"contacts":contacts,"gates":gates,"limitations":["Static truth-pose IK only.","No arm command, closure or object physics executed."]}
    pg.write_new_text(output/"result.json",json.dumps(result,ensure_ascii=False,indent=2)+"\n");pg.write_new_bytes(output/Path(__file__).name,Path(__file__).read_bytes());pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl":str(output),"gates":gates,"metrics":metrics,"right_arm_q_rad":result["right_arm_q_rad"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
