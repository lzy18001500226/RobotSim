from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(os.environ.get("AGIBOT_X2_VENDOR_ROOT", "/tmp/robotsim-issue46-agibot-x2-urdf-575cc6b988f976c23550e0db85aa1e5475d3652d"))
URDF = ROOT / "X2_URDF-v1.4.0" / "X2-Ultra_omnihand.urdf"
OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", str(Path(__file__).parent)))
SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
DT = 0.002
SOURCE_TABLE_XY = np.array([0.30, 0.0])
SOURCE_TABLE_HALF_EXTENTS = np.array([0.20, 0.12])
TARGET_TABLE_BODY_XY = np.array([0.30, -0.24])
TARGET_TABLE_HALF_EXTENTS = np.array([0.20, 0.10])
TARGET_TABLE_SITE_OFFSET_Y = 0.04
TABLE_TOP_Z = 0.80
START = np.array([0.30, 0.0, 0.878])
GOAL = np.array([0.30, -0.20, 0.878])
BOTTLE_RADIUS = 0.0325
BOTTLE_TOP_Z = float(START[2] + 0.153)
ROBOT_BASE_WORLD = np.array([0.443, 0.323, 0.68])
ROBOT_BASE_YAW = -math.pi / 2.0
ROBOT_FORWARD_WORLD = np.array([math.cos(ROBOT_BASE_YAW), math.sin(ROBOT_BASE_YAW), 0.0])
MANIPULATION_APPROACH_WORLD = np.array([0.0, -1.0, 0.0])
PALM_APPROACH_STANDOFF = 0.140
PREGRASP_STANDOFF = 0.240
GRASP_PALM_POS = np.array(
    [START[0], START[1] - MANIPULATION_APPROACH_WORLD[1] * PALM_APPROACH_STANDOFF, START[2]]
)
PREGRASP_PALM_POS = np.array(
    [START[0], START[1] - MANIPULATION_APPROACH_WORLD[1] * PREGRASP_STANDOFF, START[2]]
)
APPROACH_PALM_POS = (PREGRASP_PALM_POS + GRASP_PALM_POS) / 2.0
LIFT_PALM_Z = float(GRASP_PALM_POS[2] + 0.080)
PALM_NORMAL_WORLD = MANIPULATION_APPROACH_WORLD.copy()
FINGER_AXIS_WORLD = MANIPULATION_APPROACH_WORLD.copy()
PALM_SPREAD_WORLD = np.array([0.0, 0.0, 1.0])
PALM_SIDE_WORLD = np.cross(PALM_SPREAD_WORLD, PALM_NORMAL_WORLD)
PALM_TARGET_ROTATION = np.column_stack(
    (PALM_SIDE_WORLD, PALM_SPREAD_WORLD, PALM_NORMAL_WORLD)
)
PLACE_PALM_POS = np.array([GOAL[0], GOAL[1], START[2]]) - PALM_NORMAL_WORLD * PALM_APPROACH_STANDOFF
FINGER_CLOSE_FRACTION = float(os.environ.get("ISSUE46_FINGER_CLOSE_FRACTION", "0.45"))
ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
FINGERS = {
    "R_thumb_roll_joint": 0.32,
    "R_thumb_abad_joint": -0.72,
    "R_thumb_mcp_joint": 0.55,
    "R_thumb_pip_joint": 0.75,
    "R_thumb_dip_joint": 0.82,
    "R_index_abad_joint": -0.10,
    "R_index_pip_joint": 0.70,
    "R_index_dip_joint": 0.82,
    "R_middle_pip_joint": 0.70,
    "R_middle_dip_joint": 0.82,
    "R_ring_abad_joint": 0.10,
    "R_ring_pip_joint": 0.70,
    "R_ring_dip_joint": 0.82,
    "R_pinky_abad_joint": 0.14,
    "R_pinky_pip_joint": 0.70,
    "R_pinky_dip_joint": 0.82,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or f"unnamed_{idx}"


def body_ancestors(model: mujoco.MjModel, body_id: int) -> list[str]:
    names: list[str] = []
    current = body_id
    while current > 0:
        names.append(name(model, mujoco.mjtObj.mjOBJ_BODY, current))
        current = int(model.body_parentid[current])
    return names


def build_model() -> tuple[mujoco.MjModel, dict[str, object]]:
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    base = spec.worldbody.bodies[0]
    base.pos = ROBOT_BASE_WORLD.tolist()
    base.quat = [math.cos(ROBOT_BASE_YAW / 2.0), 0.0, 0.0, math.sin(ROBOT_BASE_YAW / 2.0)]
    base_name = base.name
    base_pos = list(base.pos)
    base_quat = list(base.quat)
    source_xml = ET.parse(URDF).getroot()
    wrist = next(b for b in spec.bodies if b.name == "right_wrist_roll_link")
    palm_joint = next(j for j in source_xml.findall("joint") if j.get("name") == "R_palm_joint")
    palm_origin = palm_joint.find("origin")
    palm_local_pos = np.fromstring(palm_origin.get("xyz", "0 0 0"), sep=" ")
    palm_local_rpy = np.fromstring(palm_origin.get("rpy", "0 0 0"), sep=" ")
    palm_local_quat = np.zeros(4)
    mujoco.mju_euler2Quat(palm_local_quat, palm_local_rpy, "XYZ")
    palm_local_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(palm_local_rotation, palm_local_quat)
    palm_local_rotation = palm_local_rotation.reshape(3, 3)

    def quat_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    wrist.add_site(
        name="right_palm_frame",
        pos=palm_local_pos.tolist(),
        quat=palm_local_quat.tolist(),
        size=[0.008, 0.008, 0.008],
        rgba=[0.95, 0.10, 0.72, 1.0],
    )
    axis_half_length = 0.045
    axis_sites = (
        ("palm_axis_side", np.array([1.0, 0.0, 0.0]), [1.0, 0.1, 0.1, 1.0], np.array([1.0, 0.0, 0.0, 0.0])),
        ("palm_axis_spread", np.array([0.0, 1.0, 0.0]), [0.1, 0.8, 0.2, 1.0], np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])),
        ("palm_axis_normal", np.array([0.0, 0.0, 1.0]), [0.1, 0.35, 1.0, 1.0], np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])),
    )
    for axis_name, palm_axis, rgba, local_rotation in axis_sites:
        wrist.add_site(
            name=axis_name,
            pos=(palm_local_pos + palm_local_rotation @ (palm_axis * axis_half_length)).tolist(),
            quat=quat_product(palm_local_quat, local_rotation).tolist(),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[axis_half_length, 0.0025, 0.0025],
            rgba=rgba,
        )

    spec.add_texture(
        name="issue46_sky_gradient",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.88, 0.92, 0.96],
        rgb2=[0.70, 0.78, 0.86],
        width=512,
        height=3072,
    )
    floor_texture = spec.add_texture(
        name="issue46_floor_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.96, 0.97, 0.98],
        rgb2=[0.70, 0.75, 0.79],
        width=512,
        height=512,
    )
    floor_material = spec.add_material(name="issue46_floor_material", texrepeat=[2.0, 2.0], reflectance=0.0)
    floor_material.textures = [floor_texture.name]
    floor_material.texuniform = True

    spec.worldbody.add_geom(
        name="experiment_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.1],
        material=floor_material.name,
        rgba=[0.88, 0.90, 0.92, 1.0],
    )
    for index, coordinate in enumerate(np.arange(-0.7, 1.11, 0.1)):
        spec.worldbody.add_geom(
            name=f"issue46_floor_grid_x_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(coordinate), 0.0, 0.0005],
            size=[0.0007, 0.6, 0.0005],
            rgba=[0.49, 0.56, 0.62, 1.0],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_geom(
            name=f"issue46_floor_grid_y_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.2, float(coordinate), 0.0005],
            size=[0.9, 0.0007, 0.0005],
            rgba=[0.49, 0.56, 0.62, 1.0],
            contype=0,
            conaffinity=0,
        )
    spec.worldbody.add_geom(
        name="experiment_source_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(SOURCE_TABLE_XY[0]), float(SOURCE_TABLE_XY[1]), TABLE_TOP_Z / 2.0],
        size=[float(SOURCE_TABLE_HALF_EXTENTS[0]), float(SOURCE_TABLE_HALF_EXTENTS[1]), TABLE_TOP_Z / 2.0],
        rgba=[0.24, 0.34, 0.42, 1.0],
        friction=[3.0, 0.01, 0.001],
    )
    spec.worldbody.add_geom(
        name="experiment_target_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(TARGET_TABLE_BODY_XY[0]), float(TARGET_TABLE_BODY_XY[1]), TABLE_TOP_Z / 2.0],
        size=[float(TARGET_TABLE_HALF_EXTENTS[0]), float(TARGET_TABLE_HALF_EXTENTS[1]), TABLE_TOP_Z / 2.0],
        rgba=[0.24, 0.34, 0.42, 1.0],
        friction=[3.0, 0.01, 0.001],
    )
    spec.worldbody.add_geom(
        name="experiment_target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(GOAL[0]), float(GOAL[1]), TABLE_TOP_Z + 0.003],
        size=[0.055, 0.002, 0.0],
        rgba=[0.90, 0.66, 0.16, 1.0],
        contype=0,
        conaffinity=0,
    )
    bottle = spec.worldbody.add_body(name="bottle", pos=START.tolist())
    bottle.add_freejoint(name="bottle_free")
    bottle.add_geom(
        name="m0_bottle_body",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.0],
        size=[0.0325, 0.078, 0.0],
        mass=0.470,
        rgba=[0.64, 0.84, 0.90, 0.78],
        friction=[1.4, 0.02, 0.001],
        condim=4,
    )
    bottle.add_geom(
        name="m0_bottle_shoulder",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.084],
        size=[0.029, 0.020, 0.0],
        mass=0.015,
        rgba=[0.64, 0.84, 0.90, 0.78],
        friction=[1.4, 0.02, 0.001],
        condim=4,
    )
    bottle.add_geom(
        name="m0_bottle_neck",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.118],
        size=[0.015, 0.019, 0.0],
        mass=0.010,
        rgba=[0.68, 0.86, 0.91, 0.85],
        friction=[1.4, 0.02, 0.001],
        condim=4,
    )
    bottle.add_geom(
        name="m0_bottle_cap",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.145],
        size=[0.018, 0.008, 0.0],
        mass=0.005,
        rgba=[0.94, 0.95, 0.91, 1.0],
        friction=[1.4, 0.02, 0.001],
        condim=4,
    )

    effort = {}
    for joint in source_xml.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort") is not None:
            effort[joint.get("name")] = float(limit.get("effort"))

    servo_counts = {"arm": 0, "finger": 0, "other": 0}
    for joint in list(spec.joints):
        joint_name = joint.name
        if joint_name is None or joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        group = "finger" if joint_name.startswith(("R_", "L_")) else "arm" if "shoulder" in joint_name or "elbow" in joint_name or "wrist" in joint_name else "other"
        kp, kv = (18.0, 2.4) if group == "finger" else (150.0, 22.0) if group == "arm" else (250.0, 30.0)
        max_force = effort.get(joint_name, 30.0)
        actuator = spec.add_actuator(
            name=f"servo_{joint_name}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            ctrllimited=bool(joint.limited),
            ctrlrange=list(joint.range) if joint.limited else [0.0, 0.0],
            forcelimited=True,
            forcerange=[-max_force, max_force],
        )
        actuator.set_to_position(kp=kp, kv=kv)
        servo_counts[group] += 1

    model = spec.compile()
    model.opt.timestep = DT
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = [0.32, 0.32, 0.32]
    model.vis.headlight.diffuse[:] = [0.58, 0.58, 0.58]
    model.vis.headlight.specular[:] = [0.08, 0.08, 0.08]
    model.vis.rgba.haze[:] = [0.91, 0.93, 0.95, 1.0]
    model.vis.map.fogstart = 3.0
    model.vis.map.fogend = 8.0
    for geom_name in (
        "experiment_source_table",
        "experiment_target_table",
        "m0_bottle_body",
        "m0_bottle_shoulder",
        "m0_bottle_neck",
        "m0_bottle_cap",
    ):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_solref[geom_id] = [0.008, 1.0]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, {
        "data": data,
        "base_body": base_name,
        "base_pos": base_pos,
        "base_quat_wxyz": base_quat,
        "site_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm_frame"),
        "palm_local_pos": palm_local_pos.tolist(),
        "palm_local_rpy": palm_local_rpy.tolist(),
        "palm_local_quat": palm_local_quat.tolist(),
        "palm_local_rotation": palm_local_rotation.tolist(),
        "source_table_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "experiment_source_table"),
        "target_table_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "experiment_target_table"),
        "servo_counts": servo_counts,
        "effort_count": len(effort),
    }


def resolve_joints(model: mujoco.MjModel, joint_names: list[str]) -> tuple[list[int], list[int]]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
    if any(idx < 0 for idx in ids):
        raise RuntimeError(f"Missing joints: {[n for n, idx in zip(joint_names, ids) if idx < 0]}")
    return ids, [int(model.jnt_qposadr[idx]) for idx in ids]


def ik_plan(
    model: mujoco.MjModel,
    seed: np.ndarray,
    site_id: int,
    arm_ids: list[int],
    targets: list[np.ndarray],
    target_rotation: np.ndarray | None = None,
) -> list[np.ndarray]:
    plan_data = mujoco.MjData(model)
    plan_data.qpos[:] = seed
    mujoco.mj_forward(model, plan_data)
    qpos = [int(model.jnt_qposadr[idx]) for idx in arm_ids]
    lower = model.jnt_range[arm_ids, 0] + 1e-6
    upper = model.jnt_range[arm_ids, 1] - 1e-6
    target_quat = np.zeros(4)
    if target_rotation is not None:
        mujoco.mju_mat2Quat(target_quat, np.asarray(target_rotation, dtype=float).reshape(9))
    outputs = []
    for target in targets:
        base_qpos = plan_data.qpos.copy()
        seed_arm = np.clip(plan_data.qpos[qpos], lower, upper)

        def residual(arm_qpos: np.ndarray) -> np.ndarray:
            plan_data.qpos[:] = base_qpos
            plan_data.qpos[qpos] = arm_qpos
            mujoco.mj_forward(model, plan_data)
            position_error = 100.0 * (target - plan_data.site_xpos[site_id])
            errors = position_error
            if target_rotation is not None:
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, plan_data.site_xmat[site_id])
                rotation_error = np.zeros(3)
                mujoco.mju_subQuat(rotation_error, target_quat, current_quat)
                rotation_error = plan_data.site_xmat[site_id].reshape(3, 3) @ rotation_error
                errors = np.concatenate((position_error, rotation_error))
            return errors

        fit = least_squares(
            residual,
            seed_arm,
            bounds=(lower, upper),
            max_nfev=500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        plan_data.qpos[qpos] = fit.x
        mujoco.mj_forward(model, plan_data)
        residual = float(np.linalg.norm(target - plan_data.site_xpos[site_id]))
        rotation_residual = 0.0
        if target_rotation is not None:
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, plan_data.site_xmat[site_id])
            rotation_error = np.zeros(3)
            mujoco.mju_subQuat(rotation_error, target_quat, current_quat)
            rotation_residual = float(np.linalg.norm(plan_data.site_xmat[site_id].reshape(3, 3) @ rotation_error))
        outputs.append(plan_data.qpos[qpos].copy())
        print(f"IK target={target.tolist()} achieved={plan_data.site_xpos[site_id].round(5).tolist()} pos_residual={residual:.6f} rot_residual={rotation_residual:.6f}")
        if residual > 0.003 or rotation_residual > 0.04:
            raise RuntimeError(f"Right palm target is unreachable: target={target.tolist()} pos={residual:.6f} rot={rotation_residual:.6f}")
    return outputs


def referenced_meshes() -> dict[str, str]:
    root = ET.parse(URDF).getroot()
    result = {}
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if not filename:
            continue
        asset = (URDF.parent / filename).resolve()
        result[str(asset.relative_to(ROOT))] = sha256(asset)
    return result


def main() -> int:
    wall_start = time.time()
    if not 0.0 < FINGER_CLOSE_FRACTION <= 1.0:
        raise ValueError("ISSUE46_FINGER_CLOSE_FRACTION must be in (0, 1]")
    OUT.mkdir(parents=True, exist_ok=True)
    print("EXPERIMENT=RobotSim Issue #46 scratch OmniHand physical grasp")
    print(f"COMMAND=MUJOCO_GL=egl {sys.executable} {Path(__file__)}")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    robot_sim_root = Path(__file__).resolve().parents[2]
    robot_sim_commit = subprocess.check_output(["git", "-C", str(robot_sim_root), "rev-parse", "HEAD"], text=True).strip()
    robot_sim_branch = subprocess.check_output(["git", "-C", str(robot_sim_root), "branch", "--show-current"], text=True).strip()
    robot_sim_dirty = subprocess.check_output(["git", "-C", str(robot_sim_root), "status", "--porcelain"], text=True).strip()
    if commit != SOURCE_PIN or dirty:
        raise RuntimeError(f"Pinned vendor checkout identity mismatch: head={commit}, dirty={bool(dirty)}")
    model, details = build_model()
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "source_url": "https://github.com/AgibotTech/agibot_x2_urdf",
        "source_pin": SOURCE_PIN,
        "source_head": commit,
        "source_dirty": bool(dirty),
        "source_license": "Mulan PSL v2",
        "license_sha256": sha256(ROOT / "LICENSE"),
        "urdf": str(URDF),
        "urdf_sha256": sha256(URDF),
        "mesh_sha256": referenced_meshes(),
        "robot_sim_commit": robot_sim_commit,
        "robot_sim_branch": robot_sim_branch,
        "robot_sim_worktree_dirty": bool(robot_sim_dirty),
        "end_effector_frame": {
            "site": "right_palm_frame",
            "derived_from_urdf_joint": "R_palm_joint",
            "position_in_right_wrist_roll_link_m": details["palm_local_pos"],
            "rpy_in_right_wrist_roll_link_rad": details["palm_local_rpy"],
            "quaternion_wxyz_in_right_wrist_roll_link": details["palm_local_quat"],
            "palm_axes_visualization_sites": [
                "palm_axis_side",
                "palm_axis_spread",
                "palm_axis_normal",
            ],
            "target_semantics": {
                "palm_side_world": PALM_SIDE_WORLD.tolist(),
                "palm_normal_world": PALM_NORMAL_WORLD.tolist(),
                "finger_axis_world": FINGER_AXIS_WORLD.tolist(),
                "spread_axis_world": PALM_SPREAD_WORLD.tolist(),
                "palm_local_z_is_palm_normal_and_finger_extension": True,
                "palm_local_y_is_digit_row_spread": True,
                "palm_local_x_is_side_and_curl_axis": True,
                "urdf_local_axis_mapping": {
                    "palm_local_x_to_world": PALM_SIDE_WORLD.tolist(),
                    "palm_local_y_to_world": PALM_SPREAD_WORLD.tolist(),
                    "palm_local_z_to_world": PALM_NORMAL_WORLD.tolist(),
                },
                "rotation_matrix_columns": PALM_TARGET_ROTATION.tolist(),
            },
        },
        "benchmark_station": {
            "robot_base_position_world_m": ROBOT_BASE_WORLD.tolist(),
            "robot_base_yaw_world_rad": ROBOT_BASE_YAW,
            "robot_forward_axis_world": ROBOT_FORWARD_WORLD.tolist(),
            "manipulation_approach_axis_world": MANIPULATION_APPROACH_WORLD.tolist(),
            "source_table_xy_m": SOURCE_TABLE_XY.tolist(),
            "target_table_body_xy_m": TARGET_TABLE_BODY_XY.tolist(),
            "source_table_half_extents_m": SOURCE_TABLE_HALF_EXTENTS.tolist(),
            "target_table_half_extents_m": TARGET_TABLE_HALF_EXTENTS.tolist(),
            "table_top_z_m": TABLE_TOP_Z,
        },
        "grasp_parameters": {
            "grasp_palm_position_m": GRASP_PALM_POS.tolist(),
            "pregrasp_palm_position_m": PREGRASP_PALM_POS.tolist(),
            "place_palm_position_m": PLACE_PALM_POS.tolist(),
            "lift_palm_z_m": LIFT_PALM_Z,
            "finger_close_fraction": FINGER_CLOSE_FRACTION,
            "thumb_roll_rad": 0.25,
            "thumb_abduction_rad": -0.30,
            "thumb_mcp_rad": 0.82,
        },
    }
    data = details["data"]
    arm_ids, arm_qpos = resolve_joints(model, ARM)
    finger_ids, finger_qpos = resolve_joints(model, list(FINGERS))
    actuator_by_joint = {}
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        actuator_by_joint[joint_id] = aid
    arm_ctrl = [actuator_by_joint[idx] for idx in arm_ids]
    finger_ctrl = [actuator_by_joint[idx] for idx in finger_ids]
    site_id = int(details["site_id"])
    if site_id < 0 or model.neq != 0:
        raise RuntimeError(f"Invalid palm site or nonzero equality constraints: site={site_id} neq={model.neq}")

    q0 = data.qpos.copy()
    initial_bottle_qpos = q0[-7:].copy()
    target_rotation = PALM_TARGET_ROTATION.copy()
    approach_positions = [
        PREGRASP_PALM_POS.copy(),
        APPROACH_PALM_POS.copy(),
        GRASP_PALM_POS.copy(),
    ]
    pregrasp_route = ik_plan(
        model,
        q0,
        site_id,
        arm_ids,
        approach_positions,
        target_rotation,
    )
    lift_seed = q0.copy()
    lift_seed[arm_qpos] = pregrasp_route[-1]
    lift_target = ik_plan(
        model,
        lift_seed,
        site_id,
        arm_ids,
        [np.array([GRASP_PALM_POS[0], GRASP_PALM_POS[1], LIFT_PALM_Z])],
        target_rotation,
    )[0]
    transfer_seed = q0.copy()
    transfer_seed[arm_qpos] = lift_target
    transfer_targets = ik_plan(
        model,
        transfer_seed,
        site_id,
        arm_ids,
        [
            np.array([PLACE_PALM_POS[0], y, LIFT_PALM_Z])
            for y in np.linspace(GRASP_PALM_POS[1], PLACE_PALM_POS[1], 5)[1:]
        ],
        target_rotation,
    )
    lower_target = ik_plan(
        model,
        transfer_seed,
        site_id,
        arm_ids,
        [PLACE_PALM_POS],
        target_rotation,
    )[0]
    place_high_target = transfer_targets[-1]
    data.qpos[arm_qpos] = pregrasp_route[0]

    open_target = q0[finger_qpos].copy()
    for joint_name, value in {
        "R_thumb_roll_joint": 0.80,
        "R_thumb_abad_joint": -1.70,
        "R_index_abad_joint": -0.18,
        "R_ring_abad_joint": 0.15,
        "R_pinky_abad_joint": 0.15,
    }.items():
        open_target[list(FINGERS).index(joint_name)] = value
    data.qpos[finger_qpos] = open_target
    mujoco.mj_forward(model, data)
    initial_robot_arm_pose = data.qpos[arm_qpos].copy()
    close_target = open_target.copy()
    for i, joint_name in enumerate(FINGERS):
        joint_id = finger_ids[i]
        target = FINGERS[joint_name]
        if "thumb_mcp" in joint_name:
            target = 0.82
        elif "thumb_pip" in joint_name:
            target = 1.15
        elif "thumb_dip" in joint_name:
            target = 1.0
        elif joint_name.endswith("_pip_joint"):
            target = 1.15
        elif joint_name.endswith("_dip_joint"):
            target = 1.30
        close_target[i] = np.clip(target, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
    close_target = open_target + FINGER_CLOSE_FRACTION * (close_target - open_target)
    close_target[list(FINGERS).index("R_thumb_roll_joint")] = 0.25
    close_target[list(FINGERS).index("R_thumb_abad_joint")] = -0.30
    thumb_target = open_target.copy()
    for i, joint_name in enumerate(FINGERS):
        if joint_name.startswith("R_thumb_"):
            thumb_target[i] = close_target[i]
    hold_targets = data.ctrl.copy()
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        hold_targets[aid] = q0[qadr]
    data.ctrl[:] = hold_targets
    data.ctrl[arm_ctrl] = initial_robot_arm_pose
    data.ctrl[finger_ctrl] = open_target

    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    bottle_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in ("m0_bottle_body", "m0_bottle_shoulder", "m0_bottle_neck", "m0_bottle_cap")
    }
    source_table_geom = int(details["source_table_geom_id"])
    target_table_geom = int(details["target_table_geom_id"])
    table_geoms = {source_table_geom, target_table_geom}
    bottle_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")])
    bottle_dadr = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")])
    bottle_mass = float(model.body_mass[bottle_body])

    def apply_targets(arm_target: np.ndarray, finger_target: np.ndarray) -> None:
        data.ctrl[:] = hold_targets
        data.ctrl[arm_ctrl] = arm_target
        data.ctrl[finger_ctrl] = finger_target

    def contact_state() -> tuple[set[str], set[str], bool, bool]:
        hand_bodies: set[str] = set()
        left_bodies: set[str] = set()
        table = False
        target_table = False
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if bottle_geoms.intersection((g1, g2)):
                bottle_geom = g1 if g1 in bottle_geoms else g2
                other = g2 if g1 == bottle_geom else g1
                other_body = int(model.geom_bodyid[other])
                ancestors = body_ancestors(model, other_body)
                if other in table_geoms:
                    table = True
                if other == target_table_geom:
                    target_table = True
                if any(n.startswith(("R_", "right_")) for n in ancestors):
                    hand_bodies.add(ancestors[0])
                if any(n.startswith(("L_", "left_")) for n in ancestors):
                    left_bodies.add(ancestors[0])
        return hand_bodies, left_bodies, table, target_table

    initial_right_contacts, initial_left_contacts, initial_table_contact, initial_target_table_contact = contact_state()
    pregrasp_palm_position = data.site_xpos[site_id].copy()
    pregrasp_palm_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    if initial_right_contacts or initial_left_contacts:
        raise RuntimeError(
            "Initial robot staging pose contacts the bottle: "
            f"right={sorted(initial_right_contacts)} left={sorted(initial_left_contacts)}"
        )

    phases: list[dict[str, object]] = []
    contact_events: list[dict[str, object]] = []
    previous_contact: tuple[tuple[str, ...], bool, bool] | None = None
    qpos_assignments_during_rollout = 0
    step_count = 0
    max_bottle_translation_step = 0.0
    max_bottle_rotation_step = 0.0
    max_penetration = 0.0
    maximum_bottle_height = float(data.qpos[bottle_qadr + 2])
    first_right_contact = None
    first_left_contact = None
    first_table_contact = None
    first_target_table_contact = None
    carry_stats = {
        label: {"steps": 0, "contact_steps": 0, "finger_contact_steps": 0, "right_bodies": set(), "left_bodies": set()}
        for label in ("lift", "transfer")
    }

    import imageio.v2 as imageio

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.28, 0.02, 0.91]
    camera.distance = 2.35
    camera.azimuth = 135
    camera.elevation = -25
    closeup_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(closeup_camera)
    closeup_camera.lookat[:] = [START[0], START[1] + 0.10, START[2]]
    closeup_camera.distance = 0.82
    closeup_camera.azimuth = 135
    closeup_camera.elevation = -12
    renderer = mujoco.Renderer(model, height=720, width=1280)
    video_writer = imageio.get_writer(OUT / "m0_x2_grasp.mp4", fps=25, codec="libx264", quality=8)
    trace_file = (OUT / "physics_contact_trace.jsonl").open("w", encoding="utf-8")

    def save_snapshot(filename: str, snapshot_camera: mujoco.MjvCamera | None = None) -> None:
        renderer.update_scene(data, camera=snapshot_camera or camera)
        imageio.imwrite(OUT / filename, renderer.render())

    save_snapshot("before_overview.png")
    save_snapshot("pregrasp.png")
    save_snapshot("pregrasp_closeup.png", closeup_camera)

    def run_phase(label: str, steps: int, arm_target: np.ndarray, finger_target: np.ndarray) -> None:
        nonlocal step_count, max_bottle_translation_step, max_bottle_rotation_step, max_penetration
        nonlocal maximum_bottle_height, first_right_contact, first_left_contact, first_table_contact
        nonlocal first_target_table_contact, previous_contact
        phase_start = float(data.time)
        phase_robot_bodies: set[str] = set()
        arm_start = data.ctrl[arm_ctrl].copy()
        fingers_start = data.ctrl[finger_ctrl].copy()
        for k in range(steps):
            alpha = (k + 1) / steps
            current_arm = arm_start * (1.0 - alpha) + arm_target * alpha
            current_fingers = fingers_start * (1.0 - alpha) + finger_target * alpha
            apply_targets(current_arm, current_fingers)
            old_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
            old_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
            mujoco.mj_step(model, data)
            step_count += 1
            new_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
            new_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
            max_bottle_translation_step = max(max_bottle_translation_step, float(np.linalg.norm(new_position - old_position)))
            dot = min(1.0, abs(float(np.dot(old_quat / np.linalg.norm(old_quat), new_quat / np.linalg.norm(new_quat)))))
            max_bottle_rotation_step = max(max_bottle_rotation_step, 2.0 * math.acos(dot))
            maximum_bottle_height = max(maximum_bottle_height, float(new_position[2]))
            right_contacts, left_contacts, table_contact, target_table_contact = contact_state()
            contact_pairs = []
            step_penetration = 0.0
            for ci in range(data.ncon):
                contact = data.contact[ci]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if not bottle_geoms.intersection((g1, g2)):
                    continue
                bottle_geom = g1 if g1 in bottle_geoms else g2
                other = g2 if g1 == bottle_geom else g1
                other_body = int(model.geom_bodyid[other])
                if other_body != 0:
                    phase_robot_bodies.add(name(model, mujoco.mjtObj.mjOBJ_BODY, other_body))
                step_penetration = max(step_penetration, float(max(0.0, -contact.dist)))
                contact_pairs.append(
                    {
                        "bottle_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom),
                        "other_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        "other_body": name(model, mujoco.mjtObj.mjOBJ_BODY, other_body),
                        "distance_m": float(contact.dist),
                        "position_m": contact.pos.tolist(),
                    }
                )
            if right_contacts and first_right_contact is None:
                first_right_contact = float(data.time)
            if left_contacts and first_left_contact is None:
                first_left_contact = float(data.time)
            if table_contact and first_table_contact is None:
                first_table_contact = float(data.time)
            if target_table_contact and first_target_table_contact is None:
                first_target_table_contact = float(data.time)
            if label in carry_stats:
                stats = carry_stats[label]
                stats["steps"] += 1
                if right_contacts:
                    stats["contact_steps"] += 1
                if any(body.startswith("R_") for body in right_contacts):
                    stats["finger_contact_steps"] += 1
                stats["right_bodies"].update(right_contacts)
                stats["left_bodies"].update(left_contacts)
            state = (tuple(sorted(right_contacts)), bool(table_contact), bool(target_table_contact))
            if state != previous_contact:
                contact_events.append(
                    {
                        "time": float(data.time),
                        "step": step_count,
                        "phase": label,
                        "right_hand_geoms": list(state[0]),
                        "table_contact": state[1],
                        "target_table_contact": state[2],
                        "left_hand_geoms": sorted(left_contacts),
                    }
                )
                previous_contact = state
            trace_file.write(
                json.dumps(
                    {
                        "step": step_count,
                        "simulation_time_s": float(data.time),
                        "phase": label,
                        "bottle_position_m": new_position.tolist(),
                        "bottle_quaternion_wxyz": new_quat.tolist(),
                        "bottle_linear_velocity_m_s": data.qvel[bottle_dadr : bottle_dadr + 3].tolist(),
                        "bottle_angular_velocity_rad_s": data.qvel[bottle_dadr + 3 : bottle_dadr + 6].tolist(),
                        "right_contact_bodies": sorted(right_contacts),
                        "opposite_hand_contact_bodies": sorted(left_contacts),
                        "table_contact": table_contact,
                        "target_table_contact": target_table_contact,
                        "bottle_contact_pairs": contact_pairs,
                        "max_penetration_m": step_penetration,
                        "bottle_qpos_assignments_this_step": 0,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if step_count % 20 == 0:
                renderer.update_scene(data, camera=camera)
                video_writer.append_data(renderer.render())
            for ci in range(data.ncon):
                contact = data.contact[ci]
                if bottle_geoms.intersection((int(contact.geom1), int(contact.geom2))):
                    max_penetration = max(max_penetration, float(max(0.0, -contact.dist)))
        qvel = data.qvel[bottle_dadr : bottle_dadr + 6].copy()
        arm_actual = data.qpos[arm_qpos].copy()
        arm_error = arm_actual - arm_target
        right_now, left_now, table_now, target_table_now = contact_state()
        phases.append(
            {
                "name": label,
                "start_time": phase_start,
                "end_time": float(data.time),
                "duration_seconds": float(data.time) - phase_start,
                "bottle_position": data.qpos[bottle_qadr : bottle_qadr + 3].tolist(),
                "bottle_linear_velocity": qvel[:3].tolist(),
                "bottle_angular_velocity": qvel[3:].tolist(),
                "right_hand_contact_geoms": sorted(right_now),
                "left_hand_contact_geoms": sorted(left_now),
                "robot_contact_bodies": sorted(phase_robot_bodies),
                "arm_joint_target_rad": arm_target.tolist(),
                "arm_joint_actual_rad": arm_actual.tolist(),
                "arm_tracking_error_l2_rad": float(np.linalg.norm(arm_error)),
                "arm_tracking_error_max_abs_rad": float(np.max(np.abs(arm_error))),
                "right_finger_target_rad": finger_target.tolist(),
                "right_finger_actual_rad": data.qpos[finger_qpos].tolist(),
                "table_contact": table_now,
                "target_table_contact": target_table_now,
                "palm_frame_position_m": data.site_xpos[site_id].tolist(),
                "palm_frame_rotation_matrix": data.site_xmat[site_id].reshape(3, 3).tolist(),
            }
        )
        print(
            f"PHASE {label} t={data.time:.6f} bottle={data.qpos[bottle_qadr:bottle_qadr+3].round(5).tolist()} "
            f"palm={data.site_xpos[site_id].round(5).tolist()} ncon={data.ncon} right={sorted(right_now)} "
            f"table={table_now} target_table={target_table_now}"
        )

    # Only initialization and offline IK assign generalized position state.
    # From this point through settling, bottle qpos is advanced only by mj_step.
    for waypoint, arm_target in enumerate(pregrasp_route[1:], start=1):
        run_phase(f"approach_{waypoint}", 500, arm_target, open_target)
    save_snapshot("open_approach.png")
    save_snapshot("open_approach_closeup.png", closeup_camera)
    run_phase("thumb_pregrasp", 500, pregrasp_route[-1], thumb_target)
    run_phase("finger_wrap", 1100, pregrasp_route[-1], close_target)
    run_phase("grasp_hold", 500, pregrasp_route[-1], close_target)
    grip_contacts = contact_state()[0]
    print(f"GRASP_GATE right_hand_geoms={sorted(grip_contacts)}")
    save_snapshot("grasp.png")
    save_snapshot("grasp_closeup.png", closeup_camera)
    run_phase("lift", 1000, lift_target, close_target)
    save_snapshot("lift.png")
    save_snapshot("lift_closeup.png", closeup_camera)
    for arm_target in transfer_targets:
        run_phase("transfer", 700, arm_target, close_target)
    run_phase("lower_to_place", 1000, lower_target, close_target)
    release_time = float(data.time)
    run_phase("finger_open_release", 500, lower_target, open_target)
    run_phase("retreat_after_release", 600, place_high_target, open_target)
    settle_start = float(data.time)
    run_phase("free_physics_settle", 1500, place_high_target, open_target)
    settle_duration = float(data.time) - settle_start
    save_snapshot("release_settle.png")
    save_snapshot("after_overview.png")
    save_snapshot("release_settle_closeup.png", closeup_camera)

    final_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
    final_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
    final_velocity = data.qvel[bottle_dadr : bottle_dadr + 6].copy()
    final_contacts, final_left_contacts, final_table_contact, final_target_table_contact = contact_state()
    bottle_delta = final_position - START
    bottle_lift = maximum_bottle_height - float(START[2])
    bottle_target_error = float(np.linalg.norm(final_position[:2] - GOAL[:2]))
    joint_contact_bodies = sorted(grip_contacts)
    contact_finger_families = {
        "thumb": any("thumb" in n.lower() for n in joint_contact_bodies),
        "index": any("index" in n.lower() for n in joint_contact_bodies),
        "middle": any("middle" in n.lower() for n in joint_contact_bodies),
        "ring": any("ring" in n.lower() for n in joint_contact_bodies),
        "pinky": any("pinky" in n.lower() for n in joint_contact_bodies),
    }
    active_families = sum(contact_finger_families.values())
    carry_contact_report = {}
    for phase_name, stats in carry_stats.items():
        bodies = sorted(stats["right_bodies"])
        families = {
            finger: any(finger in body.lower() for body in bodies)
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        }
        carry_contact_report[phase_name] = {
            "steps": stats["steps"],
            "contact_steps": stats["contact_steps"],
            "contact_fraction": stats["contact_steps"] / stats["steps"] if stats["steps"] else 0.0,
            "finger_contact_steps": stats["finger_contact_steps"],
            "finger_contact_fraction": stats["finger_contact_steps"] / stats["steps"] if stats["steps"] else 0.0,
            "right_hand_contact_bodies": bodies,
            "left_hand_contact_bodies": sorted(stats["left_bodies"]),
            "finger_families": families,
            "active_finger_family_count": sum(families.values()),
        }
    carry_valid = all(
        item["finger_contact_fraction"] >= 0.90
        and item["finger_families"]["thumb"]
        and item["active_finger_family_count"] >= 3
        and not item["left_hand_contact_bodies"]
        for item in carry_contact_report.values()
    )
    final_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(final_rotation, final_quat)
    final_tilt = math.acos(max(-1.0, min(1.0, float(final_rotation.reshape(3, 3)[2, 2]))))
    table_min = TARGET_TABLE_BODY_XY - TARGET_TABLE_HALF_EXTENTS
    table_max = TARGET_TABLE_BODY_XY + TARGET_TABLE_HALF_EXTENTS
    final_footprint_inside = bool(
        np.all(final_position[:2] - BOTTLE_RADIUS >= table_min)
        and np.all(final_position[:2] + BOTTLE_RADIUS <= table_max)
    )
    pregrasp_position = np.asarray(pregrasp_palm_position, dtype=float)
    pregrasp_rotation = np.asarray(pregrasp_palm_rotation, dtype=float)
    relative_rotation = PALM_TARGET_ROTATION.T @ pregrasp_rotation
    pregrasp_rotation_error = math.acos(
        max(-1.0, min(1.0, float((np.trace(relative_rotation) - 1.0) / 2.0)))
    )
    pregrasp_diagnostics = {
        "site": "right_palm_frame",
        "target_position_m": PREGRASP_PALM_POS.tolist(),
        "actual_position_m": pregrasp_position.tolist(),
        "position_error_m": float(np.linalg.norm(pregrasp_position - PREGRASP_PALM_POS)),
        "target_rotation_matrix": PALM_TARGET_ROTATION.tolist(),
        "actual_rotation_matrix": pregrasp_rotation.tolist(),
        "rotation_error_rad": pregrasp_rotation_error,
        "actual_palm_side_world": pregrasp_rotation[:, 0].tolist(),
        "actual_spread_axis_world": pregrasp_rotation[:, 1].tolist(),
        "actual_palm_normal_world": pregrasp_rotation[:, 2].tolist(),
        "actual_finger_axis_world": pregrasp_rotation[:, 2].tolist(),
        "opposing_hand_contact_bodies": sorted(initial_left_contacts),
        "active_hand_contact_bodies_before_closure": sorted(initial_right_contacts),
        "arm_joint_order": ARM,
        "arm_joint_actual_rad": initial_robot_arm_pose.tolist(),
        "elbow_angle_rad": float(initial_robot_arm_pose[ARM.index("right_elbow_joint")]),
    }
    pregrasp_valid = bool(
        pregrasp_diagnostics["position_error_m"] <= 0.03
        and pregrasp_rotation_error <= 0.10
        and not pregrasp_diagnostics["active_hand_contact_bodies_before_closure"]
        and not pregrasp_diagnostics["opposing_hand_contact_bodies"]
    )
    (OUT / "frame_diagnostics.json").write_text(json.dumps(pregrasp_diagnostics, indent=2) + "\n")
    gates = {
        "plausible_collision_free_pregrasp": pregrasp_valid,
        "thumb_plus_at_least_two_other_digit_families_at_grasp": contact_finger_families["thumb"] and active_families >= 3,
        "at_least_three_digit_families_contact_during_lift_and_transfer": carry_valid,
        "opposite_hand_never_assists": first_left_contact is None and not final_left_contacts,
        "lift_at_least_0_05_m": bottle_lift >= 0.05,
        "bottle_final_footprint_inside_table": final_footprint_inside,
        "final_xy_error_at_most_0_05_m": bottle_target_error <= 0.05,
        "free_physics_target_table_supported_settle": not final_contacts and not final_left_contacts and final_target_table_contact,
        "no_runtime_weld_or_equality": model.neq == 0,
        "no_bottle_qpos_writes_during_rollout": qpos_assignments_during_rollout == 0,
    }
    passed = all(
        [
            pregrasp_valid,
            len(grip_contacts) >= 2,
            contact_finger_families["thumb"],
            active_families >= 3,
            carry_valid,
            first_left_contact is None,
            bottle_lift >= 0.05,
            not final_contacts,
            not final_left_contacts,
            final_target_table_contact,
            bottle_target_error <= 0.05,
            float(np.linalg.norm(final_velocity[:3])) <= 0.02,
            float(np.linalg.norm(final_velocity[3:])) <= 0.05,
            final_tilt <= 0.20,
            max_penetration <= 0.02,
            model.neq == 0,
            qpos_assignments_during_rollout == 0,
        ]
    )

    identity.update(
        {
            "native_timestep_seconds": float(model.opt.timestep),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "njnt": model.njnt,
            "ngeom": model.ngeom,
            "nmesh": model.nmesh,
            "neq": model.neq,
            "servo_counts": details["servo_counts"],
            "effort_limited_joint_count": details["effort_count"],
            "bottle_mass_kg": bottle_mass,
            "bottle_dimensions_m": {
                "body_diameter": 0.065,
                "overall_height": 0.231,
                "total_mass_kg": 0.500,
                "source": "same collision-geometry specification as G1 M0",
            },
            "table_top_z_m": TABLE_TOP_Z,
            "table_geometry": {
                "source_half_extents_xy_m": SOURCE_TABLE_HALF_EXTENTS.tolist(),
                "target_half_extents_xy_m": TARGET_TABLE_HALF_EXTENTS.tolist(),
                "source_center_xy_m": SOURCE_TABLE_XY.tolist(),
                "target_body_center_xy_m": TARGET_TABLE_BODY_XY.tolist(),
                "target_site_offset_y_m": TARGET_TABLE_SITE_OFFSET_Y,
            },
            "control_model": "URDF joint-effort-limited position servos; right arm waypoints from bounded least-squares palm-site IK",
            "runtime_gl": os.environ.get("MUJOCO_GL", "default"),
        }
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "diagnosis": {
            "station_root_cause": "The previous fixed-base station kept the X2 torso facing world +X while the canonical source-to-target axis ran along world -Y, so the active arm reached laterally across its body. The scratch station now puts the source table on the robot's forward -Y axis and aligns the torso with that axis.",
            "wrist_palm_root_cause": "The official URDF R_palm_joint transform is present and correctly composed by the converter. The previous task frame treated the wrist-roll origin as the palm frame and added an undocumented 180 degree yaw, so the requested hand normal and finger-extension directions were wrong. The corrected task frame is attached at the URDF palm transform and maps its documented local axes explicitly.",
            "frame_verification": "The compiled right_palm_frame site is attached to right_wrist_roll_link; local position error against the URDF R_palm_joint origin is 0 m and quaternion absolute-dot agreement is 1.0.",
            "corrected_layer": "Scratch fixed-base scene/station and task-level end-effector frame convention. No URDF, vendor MJCF, joint conversion, or shared controller code was changed.",
            "canonical_assets": "The scratch scene uses MuJoCo primitives matching the canonical G1 M0 collision specification in simulation/mujoco/m0_pick_place.py: table top at 0.80 m, source half-extents 0.20 by 0.12 m, target half-extents 0.20 by 0.10 m, and a free 65 mm by 231 mm, 0.50 kg bottle. No shared standalone table or bottle asset was found in the inspected M0 scene source.",
        },
        "identity": identity,
        "scene": {
            "fixed_base_body": details["base_body"],
            "fixed_base_position": details["base_pos"],
            "fixed_base_quaternion_wxyz": details["base_quat_wxyz"],
            "robot_forward_axis_world": ROBOT_FORWARD_WORLD.tolist(),
            "manipulation_approach_axis_world": MANIPULATION_APPROACH_WORLD.tolist(),
            "source_table_center_xy": SOURCE_TABLE_XY.tolist(),
            "target_table_body_center_xy": TARGET_TABLE_BODY_XY.tolist(),
            "table_top_z": TABLE_TOP_Z,
            "bottle_initial_qpos": initial_bottle_qpos.tolist(),
            "bottle_start_position": START.tolist(),
            "bottle_goal_position": GOAL.tolist(),
            "grasp_palm_target_position": GRASP_PALM_POS.tolist(),
            "pregrasp_palm_position": PREGRASP_PALM_POS.tolist(),
            "place_palm_target_position": PLACE_PALM_POS.tolist(),
            "grasp_palm_target_rotation_matrix": PALM_TARGET_ROTATION.tolist(),
            "robot_initialization_arm_qpos": initial_robot_arm_pose.tolist(),
            "initial_robot_bottle_contacts": sorted(initial_right_contacts | initial_left_contacts),
            "initial_table_contact": initial_table_contact,
            "initial_target_table_contact": initial_target_table_contact,
            "model_free_bottle_joint": True,
            "runtime_weld": False,
            "equality_constraint_count": model.neq,
            "opposite_hand_contact": first_left_contact is not None,
            "rollout_bottle_qpos_assignments": qpos_assignments_during_rollout,
            "target_footprint_inside_table": final_footprint_inside,
        },
        "pregrasp": {
            "valid": pregrasp_valid,
            **pregrasp_diagnostics,
        },
        "events": {
            "first_right_hand_contact_time": first_right_contact,
            "first_opposite_hand_contact_time": first_left_contact,
            "first_table_contact_time": first_table_contact,
            "first_target_table_contact_time": first_target_table_contact,
            "release_command_time": release_time,
            "contact_sequence": contact_events,
        },
        "grasp_gate": {
            "right_hand_contact_bodies": joint_contact_bodies,
            "finger_families": contact_finger_families,
            "active_finger_family_count": active_families,
            "carry_contact": carry_contact_report,
        },
        "acceptance_gates": gates,
        "evidence_files": {
            "result_json": str(OUT / "result.json"),
            "runtime_identity_json": str(OUT / "runtime_identity.json"),
            "physics_contact_trace_jsonl": str(OUT / "physics_contact_trace.jsonl"),
            "video_mp4": str(OUT / "m0_x2_grasp.mp4"),
            "before_overview_png": str(OUT / "before_overview.png"),
            "after_overview_png": str(OUT / "after_overview.png"),
            "pregrasp_png": str(OUT / "pregrasp.png"),
            "pregrasp_closeup_png": str(OUT / "pregrasp_closeup.png"),
            "open_approach_png": str(OUT / "open_approach.png"),
            "open_approach_closeup_png": str(OUT / "open_approach_closeup.png"),
            "grasp_png": str(OUT / "grasp.png"),
            "grasp_closeup_png": str(OUT / "grasp_closeup.png"),
            "lift_png": str(OUT / "lift.png"),
            "lift_closeup_png": str(OUT / "lift_closeup.png"),
            "release_settle_png": str(OUT / "release_settle.png"),
            "release_settle_closeup_png": str(OUT / "release_settle_closeup.png"),
            "frame_diagnostics_json": str(OUT / "frame_diagnostics.json"),
            "final_screenshot_png": str(OUT / "final.png"),
        },
        "result": {
            "maximum_lift_m": bottle_lift,
            "initial_position": START.tolist(),
            "final_position": final_position.tolist(),
            "final_quaternion_wxyz": final_quat.tolist(),
            "final_velocity_linear_m_s": final_velocity[:3].tolist(),
            "final_velocity_angular_rad_s": final_velocity[3:].tolist(),
            "target_xy_error_m": bottle_target_error,
            "final_upright_tilt_rad": final_tilt,
            "final_table_contact": final_table_contact,
            "final_target_table_contact": final_target_table_contact,
            "final_right_hand_contact": sorted(final_contacts),
            "final_opposite_hand_contact": sorted(final_left_contacts),
            "settle_duration_seconds": settle_duration,
            "max_penetration_m": max_penetration,
            "max_object_translation_per_step_m": max_bottle_translation_step,
            "max_object_rotation_per_step_rad": max_bottle_rotation_step,
            "object_initial_to_final_translation_m": float(np.linalg.norm(bottle_delta)),
        },
        "phases": phases,
        "wall_runtime_seconds": time.time() - wall_start,
    }
    (OUT / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    trace_file.close()
    renderer.update_scene(data, camera=camera)
    import imageio.v3 as iio

    iio.imwrite(OUT / "final.png", renderer.render())
    video_writer.close()
    renderer.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
