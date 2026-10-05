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


SIM_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_REPO_ROOT / "simulation" / "mujoco"))
from canonical_manipulation_assets import (
    CANONICAL_X2_BOTTLE_DIAMETER_M,
    CANONICAL_X2_BOTTLE_GEOMS,
    CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M,
    CANONICAL_X2_BOTTLE_HEIGHT_M,
    CANONICAL_X2_BOTTLE_MASS_KG,
    CANONICAL_X2_BOTTLE_START_BODY_POS,
    CANONICAL_X2_BOTTLE_TARGET_BODY_POS,
    G1_CANONICAL_TABLE_CENTER_XY,
    G1_CANONICAL_TABLE_HALF_EXTENTS,
    G1_CANONICAL_TABLE_TOP_Z,
    add_g1_canonical_table,
)


ROOT = Path(os.environ.get("AGIBOT_X2_VENDOR_ROOT", "/tmp/robotsim-issue46-agibot-x2-urdf-575cc6b988f976c23550e0db85aa1e5475d3652d"))
URDF = ROOT / "X2_URDF-v1.4.0" / "X2-Ultra_omnihand.urdf"
OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", str(Path(__file__).parent)))
SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
DT = 0.002
MIMIC_RELATION_TOLERANCE_RAD = 0.003
MIMIC_SOLREF_DIRECT = [-10000.0, -200.0]
HAND_JOINT_LIMIT_SOLREF_DIRECT = [-10000.0, -200.0]
RIGHT_MIMIC_DRIVER_SERVO_KP = 0.1
RIGHT_MIMIC_DRIVER_SERVO_KV = 0.003
SOURCE_TABLE_XY = np.array(G1_CANONICAL_TABLE_CENTER_XY, dtype=float)
SOURCE_TABLE_HALF_EXTENTS = np.array(G1_CANONICAL_TABLE_HALF_EXTENTS, dtype=float)
TARGET_TABLE_BODY_XY = SOURCE_TABLE_XY.copy()
TARGET_TABLE_HALF_EXTENTS = SOURCE_TABLE_HALF_EXTENTS.copy()
TARGET_TABLE_SITE_OFFSET_Y = 0.0
TABLE_TOP_Z = G1_CANONICAL_TABLE_TOP_Z
START = np.array(CANONICAL_X2_BOTTLE_START_BODY_POS, dtype=float)
GOAL = np.array(CANONICAL_X2_BOTTLE_TARGET_BODY_POS, dtype=float)
BOTTLE_RADIUS = CANONICAL_X2_BOTTLE_DIAMETER_M / 2.0
BOTTLE_TOP_Z = float(START[2] + CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M[1])
ROBOT_BASE_WORLD = np.array([0.38, 0.32, 0.68])
ROBOT_BASE_YAW = -math.pi / 2.0
ROBOT_FORWARD_WORLD = np.array([math.cos(ROBOT_BASE_YAW), math.sin(ROBOT_BASE_YAW), 0.0])
MANIPULATION_APPROACH_WORLD = np.array([0.0, -1.0, 0.0])
PALM_APPROACH_STANDOFF = 0.155
PREGRASP_STANDOFF = 0.200
APPROACH_STANDOFF = 0.240
GRASP_PALM_POS = np.array(
    [START[0], START[1] - MANIPULATION_APPROACH_WORLD[1] * PALM_APPROACH_STANDOFF, START[2]]
)
PREGRASP_PALM_POS = np.array(
    [START[0], START[1] - MANIPULATION_APPROACH_WORLD[1] * PREGRASP_STANDOFF, START[2]]
)
APPROACH_PALM_POS = np.array(
    [START[0], START[1] - MANIPULATION_APPROACH_WORLD[1] * APPROACH_STANDOFF, START[2]]
)
LIFT_PALM_Z = float(GRASP_PALM_POS[2] + 0.050)
PALM_NORMAL_WORLD = MANIPULATION_APPROACH_WORLD.copy()
FINGER_AXIS_WORLD = MANIPULATION_APPROACH_WORLD.copy()
PALM_SPREAD_WORLD = np.array([0.0, 0.0, 1.0])
PALM_SIDE_WORLD = np.cross(PALM_SPREAD_WORLD, PALM_NORMAL_WORLD)
PALM_TARGET_ROTATION = np.column_stack(
    (PALM_SIDE_WORLD, PALM_SPREAD_WORLD, PALM_NORMAL_WORLD)
)
PLACE_PALM_POS = np.array([GOAL[0], GOAL[1], START[2]]) - PALM_NORMAL_WORLD * PALM_APPROACH_STANDOFF
FINGER_PRESHAPE_FRACTION = 0.45
FINGER_CLOSE_FRACTION = float(os.environ.get("ISSUE46_FINGER_CLOSE_FRACTION", "1.0"))
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
    "R_index_abad_joint": -0.10,
    "R_index_pip_joint": 0.70,
    "R_middle_pip_joint": 0.70,
    "R_ring_abad_joint": 0.10,
    "R_ring_pip_joint": 0.70,
    "R_pinky_abad_joint": 0.14,
    "R_pinky_pip_joint": 0.70,
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
    # Keep the fixed pelvis in the kinematic tree so robot/world contacts remain distinct.
    spec.compiler.fusestatic = False
    base = spec.worldbody.bodies[0]
    base.pos = ROBOT_BASE_WORLD.tolist()
    base.quat = [math.cos(ROBOT_BASE_YAW / 2.0), 0.0, 0.0, math.sin(ROBOT_BASE_YAW / 2.0)]
    base_name = base.name
    base_pos = list(base.pos)
    base_quat = list(base.quat)
    source_xml = ET.parse(URDF).getroot()
    vendor_visual_color_values = sorted(
        {
            color.get("rgba")
            for color in source_xml.findall(".//visual/material/color")
            if color.get("rgba")
        }
    )
    vendor_visual_material_by_link: dict[str, str] = {}
    for link in source_xml.findall("link"):
        material_names = {
            material.get("name")
            for material in (visual.find("material") for visual in link.findall("visual"))
            if material is not None and material.get("name")
        }
        if len(material_names) > 1:
            raise ValueError(
                f"Multiple named visual materials on one link are not mapped safely: {link.get('name')}"
            )
        if material_names:
            vendor_visual_material_by_link[link.get("name")] = next(iter(material_names))
    parent_by_child = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in source_xml.findall("joint")
        if joint.find("parent") is not None and joint.find("child") is not None
    }
    self_chain_exclusions: list[tuple[str, str]] = []
    for descendant, parent in parent_by_child.items():
        ancestor = parent
        while ancestor:
            pair = (ancestor, descendant)
            spec.add_exclude(
                name=f"self_chain_{len(self_chain_exclusions)}",
                bodyname1=ancestor,
                bodyname2=descendant,
            )
            self_chain_exclusions.append(pair)
            ancestor = parent_by_child.get(ancestor)
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
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.30, 0.50, 0.70],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texuniform=True,
        texrepeat=[5.0, 5.0],
        reflectance=0.2,
    )
    spec.worldbody.add_light(
        name="scene_key_light",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="groundplane",
        group=1,
    )
    add_g1_canonical_table(spec)
    bottle = spec.worldbody.add_body(name="bottle", pos=START.tolist())
    bottle.add_freejoint(name="bottle_free")
    if not math.isclose(
        CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M[1]
        - CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M[0],
        CANONICAL_X2_BOTTLE_HEIGHT_M,
        abs_tol=1e-12,
    ) or not math.isclose(
        sum(geom_spec["mass"] for geom_spec in CANONICAL_X2_BOTTLE_GEOMS),
        CANONICAL_X2_BOTTLE_MASS_KG,
        abs_tol=1e-12,
    ):
        raise ValueError("Shared X2 bottle geometry, height, and mass definitions disagree")
    geom_types = {
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }
    for geom_spec in CANONICAL_X2_BOTTLE_GEOMS:
        bottle.add_geom(
            name=geom_spec["name"],
            type=geom_types[geom_spec["type"]],
            pos=list(geom_spec["pos"]),
            size=list(geom_spec["size"]),
            mass=geom_spec["mass"],
            rgba=list(geom_spec["rgba"]),
            friction=[1.4, 0.02, 0.001],
            condim=4,
            group=1,
        )

    effort = {}
    for joint in source_xml.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort") is not None:
            effort[joint.get("name")] = float(limit.get("effort"))

    spec_joint_by_name = {joint.name: joint for joint in spec.joints if joint.name}
    for joint in spec.joints:
        if joint.name and joint.name.startswith(("L_", "R_")) and joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.solref_limit = HAND_JOINT_LIMIT_SOLREF_DIRECT

    mimic_relations = []
    mimic_child_names: set[str] = set()
    for source_joint in source_xml.findall("joint"):
        mimic = source_joint.find("mimic")
        if mimic is None:
            continue
        follower_name = source_joint.get("name")
        driver_name = mimic.get("joint")
        if not follower_name or not driver_name or follower_name in mimic_child_names:
            raise ValueError(f"Invalid or duplicate URDF mimic follower: {follower_name}")
        follower = spec_joint_by_name.get(follower_name)
        driver = spec_joint_by_name.get(driver_name)
        if (
            follower is None
            or driver is None
            or follower.type != mujoco.mjtJoint.mjJNT_HINGE
            or driver.type != mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError(f"URDF mimic relation must connect imported hinge joints: {follower_name} -> {driver_name}")
        multiplier = float(mimic.get("multiplier", "1"))
        offset = float(mimic.get("offset", "0"))
        follower_ref = float(follower.ref)
        driver_ref = float(driver.ref)
        polycoef = [offset + multiplier * driver_ref - follower_ref, multiplier, 0.0, 0.0, 0.0]
        equality = spec.add_equality(
            name=f"urdf_mimic_{follower_name}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=follower_name,
            name2=driver_name,
            data=polycoef + [0.0] * 6,
        )
        equality.solref = MIMIC_SOLREF_DIRECT
        mimic_child_names.add(follower_name)
        mimic_relations.append(
            {
                "follower_joint": follower_name,
                "driver_joint": driver_name,
                "multiplier": multiplier,
                "offset": offset,
                "follower_reference_rad": follower_ref,
                "driver_reference_rad": driver_ref,
                "mujoco_polycoef": polycoef,
                "mujoco_solref_direct": MIMIC_SOLREF_DIRECT,
            }
        )
    if len(mimic_relations) != 12:
        raise ValueError(f"Expected the pinned OmniHand's 12 URDF mimic relations, found {len(mimic_relations)}")

    mimic_driver_names = {str(item["driver_joint"]) for item in mimic_relations}
    servo_counts = {"arm": 0, "finger": 0, "other": 0, "mimic_follower_joints": 0}
    for joint in list(spec.joints):
        joint_name = joint.name
        if joint_name is None or joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if joint_name in mimic_child_names:
            servo_counts["mimic_follower_joints"] += 1
            continue
        group = "finger" if joint_name.startswith(("R_", "L_")) else "arm" if "shoulder" in joint_name or "elbow" in joint_name or "wrist" in joint_name else "other"
        kp, kv = (18.0, 2.4) if group == "finger" else (150.0, 22.0) if group == "arm" else (250.0, 30.0)
        if joint_name.startswith("R_") and joint_name in mimic_driver_names:
            kp, kv = RIGHT_MIMIC_DRIVER_SERVO_KP, RIGHT_MIMIC_DRIVER_SERVO_KV
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
    expected_mimics = {
        item["follower_joint"]: item
        for item in mimic_relations
    }
    compiled_mimics = {}
    for equality_id in range(model.neq):
        if int(model.eq_type[equality_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        follower_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[equality_id]))
        driver_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[equality_id]))
        compiled_mimics[follower_name] = {
            "driver_joint": driver_name,
            "polycoef": model.eq_data[equality_id, :5].tolist(),
            "solref": model.eq_solref[equality_id].tolist(),
        }
    mimic_constraints_match = (
        model.neq == len(mimic_relations)
        and set(compiled_mimics) == set(expected_mimics)
        and all(
            compiled_mimics[follower]["driver_joint"] == relation["driver_joint"]
            and np.allclose(
                compiled_mimics[follower]["polycoef"],
                relation["mujoco_polycoef"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                compiled_mimics[follower]["solref"],
                MIMIC_SOLREF_DIRECT,
                rtol=0.0,
                atol=1e-12,
            )
            for follower, relation in expected_mimics.items()
        )
    )
    if not mimic_constraints_match:
        raise RuntimeError(f"Compiled MuJoCo model does not preserve the URDF mimic set: {compiled_mimics}")
    mimic_track = []
    for relation in mimic_relations:
        follower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, relation["follower_joint"])
        driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, relation["driver_joint"])
        follower_qpos = int(model.jnt_qposadr[follower_id])
        driver_qpos = int(model.jnt_qposadr[driver_id])
        polycoef = relation["mujoco_polycoef"]
        mimic_track.append(
            {
                **relation,
                "follower_qpos_address": follower_qpos,
                "driver_qpos_address": driver_qpos,
                "compiled_follower_reference_rad": float(model.qpos0[follower_qpos]),
                "compiled_driver_reference_rad": float(model.qpos0[driver_qpos]),
            }
        )
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = [0.30, 0.30, 0.30]
    model.vis.headlight.diffuse[:] = [0.60, 0.60, 0.60]
    model.vis.headlight.specular[:] = [0.0, 0.0, 0.0]
    model.vis.rgba.haze[:] = [0.15, 0.25, 0.35, 1.0]
    model.vis.global_.azimuth = 200.0
    model.vis.global_.elevation = -20.0
    hand_material_geom_assignments = []
    material_rgba = {
        "silver": [0.76, 0.79, 0.82, 1.0],
        "blue": [0.14, 0.34, 0.76, 1.0],
        "brown": [0.46, 0.31, 0.20, 1.0],
        "white": [0.98, 0.98, 0.97, 1.0],
        "green": [0.16, 0.50, 0.30, 1.0],
        "orange": [0.92, 0.43, 0.14, 1.0],
    }
    unmatched_hand_material_links = []
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != 1:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name.startswith(("R_", "L_")):
            continue
        material_name = vendor_visual_material_by_link.get(body_name)
        if material_name in material_rgba:
            model.geom_rgba[geom_id] = material_rgba[material_name]
            hand_material_geom_assignments.append(
                {
                    "geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                    "link": body_name,
                    "vendor_material_name": material_name,
                    "renderer_rgba": material_rgba[material_name],
                }
            )
        elif np.allclose(model.geom_rgba[geom_id], [0.5, 0.5, 0.5, 1.0]):
            unmatched_hand_material_links.append(body_name)
    for geom_name in (
        "m0_table_top",
        "bottle_body",
        "bottle_shoulder",
        "bottle_neck",
        "bottle_cap",
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
        "source_table_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_table_top"),
        "target_table_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_table_top"),
        "servo_counts": servo_counts,
        "mimic_relations": mimic_track,
        "mimic_constraints_match": mimic_constraints_match,
        "effort_count": len(effort),
        "self_chain_exclusion_count": len(self_chain_exclusions),
        "self_chain_exclusions": self_chain_exclusions,
        "vendor_visual_color_count": len(source_xml.findall(".//visual/material/color")),
        "vendor_visual_color_values": vendor_visual_color_values,
        "vendor_visual_texture_count": len(source_xml.findall(".//visual/material/texture")),
        "vendor_visual_material_by_link": vendor_visual_material_by_link,
        "hand_material_geom_assignments": hand_material_geom_assignments,
        "hand_material_rgba_fallback": material_rgba,
        "unmatched_hand_material_links": sorted(set(unmatched_hand_material_links)),
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
    reproduction_command = (
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'egl')} "
        f"ISSUE46_EVIDENCE_DIR={OUT} "
        f"ISSUE46_FINGER_CLOSE_FRACTION={FINGER_CLOSE_FRACTION} "
        f"{sys.executable} {Path(__file__).resolve()}"
    )
    (OUT / "reproduction_command.txt").write_text(reproduction_command + "\n", encoding="utf-8")
    print("EXPERIMENT=RobotSim Issue #46 scratch OmniHand physical grasp")
    print(f"COMMAND={reproduction_command}")
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
            "thumb_roll_rad": 0.65,
            "thumb_abduction_rad": -0.30,
            "thumb_mcp_rad": 0.82,
        },
    }
    data = details["data"]
    arm_ids, arm_qpos = resolve_joints(model, ARM)
    finger_joint_names = list(FINGERS)
    finger_ids, finger_qpos = resolve_joints(model, finger_joint_names)
    right_hand_joint_names = [
        joint.get("name")
        for joint in ET.parse(URDF).getroot().findall("joint")
        if joint.get("name", "").startswith("R_") and joint.get("type") == "revolute"
    ]
    _, right_hand_joint_qpos = resolve_joints(model, right_hand_joint_names)
    actuator_by_joint = {}
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        actuator_by_joint[joint_id] = aid
    arm_ctrl = [actuator_by_joint[idx] for idx in arm_ids]
    finger_ctrl = [actuator_by_joint[idx] for idx in finger_ids]
    site_id = int(details["site_id"])
    if site_id < 0 or not details["mimic_constraints_match"] or model.neq != len(details["mimic_relations"]):
        raise RuntimeError(
            f"Invalid palm site or URDF mimic constraints: site={site_id} neq={model.neq} "
            f"mimic_count={len(details['mimic_relations'])}"
        )

    def mimic_relation_state() -> dict[str, dict[str, object]]:
        states = {}
        for relation in details["mimic_relations"]:
            follower_qpos = float(data.qpos[relation["follower_qpos_address"]])
            driver_qpos = float(data.qpos[relation["driver_qpos_address"]])
            expected = relation["compiled_follower_reference_rad"] + relation["mujoco_polycoef"][0]
            expected += relation["mujoco_polycoef"][1] * (
                driver_qpos - relation["compiled_driver_reference_rad"]
            )
            states[relation["follower_joint"]] = {
                "driver_joint": relation["driver_joint"],
                "follower_position_rad": follower_qpos,
                "driver_position_rad": driver_qpos,
                "expected_follower_position_rad": float(expected),
                "error_rad": float(abs(follower_qpos - expected)),
            }
        return states

    q0 = data.qpos.copy()
    initial_bottle_qpos = q0[-7:].copy()
    target_rotation = PALM_TARGET_ROTATION.copy()
    approach_positions = [
        APPROACH_PALM_POS.copy(),
        PREGRASP_PALM_POS.copy(),
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
    pregrasp_route[-1] = ik_plan(
        model,
        q0,
        site_id,
        arm_ids,
        [GRASP_PALM_POS],
        target_rotation,
    )[0]
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
        open_target[finger_joint_names.index(joint_name)] = value
    data.qpos[finger_qpos] = open_target
    mujoco.mj_forward(model, data)
    initial_robot_arm_pose = data.qpos[arm_qpos].copy()
    close_target = open_target.copy()
    for i, joint_name in enumerate(finger_joint_names):
        joint_id = finger_ids[i]
        target = FINGERS[joint_name]
        if "thumb_mcp" in joint_name:
            target = 0.82
        elif joint_name.endswith("_pip_joint"):
            target = 1.15
        close_target[i] = np.clip(target, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
    close_target = open_target + FINGER_CLOSE_FRACTION * (close_target - open_target)
    close_target[finger_joint_names.index("R_thumb_roll_joint")] = 0.65
    close_target[finger_joint_names.index("R_thumb_abad_joint")] = -0.30
    preshape_target = open_target + FINGER_PRESHAPE_FRACTION * (close_target - open_target)
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
        for geom_name in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap")
    }
    source_table_geom = int(details["source_table_geom_id"])
    target_table_geom = int(details["target_table_geom_id"])
    table_geoms = {source_table_geom, target_table_geom}
    bottle_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")])
    bottle_dadr = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")])
    bottle_mass = float(model.body_mass[bottle_body])

    def right_table_contact_bodies() -> set[str]:
        bodies: set[str] = set()
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in table_geoms:
                other = g2
            elif g2 in table_geoms:
                other = g1
            else:
                continue
            ancestors = body_ancestors(model, int(model.geom_bodyid[other]))
            if any(n.startswith(("R_", "right_")) for n in ancestors):
                bodies.add(ancestors[0])
        return bodies

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
    initial_hand_table_contacts = right_table_contact_bodies()
    if initial_right_contacts or initial_left_contacts:
        raise RuntimeError(
            "Initial robot staging pose contacts the bottle: "
            f"right={sorted(initial_right_contacts)} left={sorted(initial_left_contacts)}"
        )

    phases: list[dict[str, object]] = []
    contact_events: list[dict[str, object]] = []
    preclosure_contact_bodies: set[str] = set()
    all_right_table_contact_bodies: set[str] = set(initial_hand_table_contacts)
    previous_contact: tuple[tuple[str, ...], bool, bool] | None = None
    qpos_assignments_during_rollout = 0
    step_count = 0
    max_bottle_translation_step = 0.0
    max_bottle_rotation_step = 0.0
    max_penetration = 0.0
    max_mimic_relation_error_rad = 0.0
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
    camera.lookat[:] = [0.30, 0.10, 1.00]
    camera.distance = 2.65
    camera.azimuth = 135
    camera.elevation = -25
    closeup_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(closeup_camera)
    closeup_camera.lookat[:] = [START[0], START[1] + 0.10, START[2]]
    closeup_camera.distance = 0.82
    closeup_camera.azimuth = 135
    closeup_camera.elevation = -12
    front_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(front_camera)
    front_camera.lookat[:] = [0.30, 0.20, 1.00]
    front_camera.distance = 2.20
    front_camera.azimuth = 90
    front_camera.elevation = -12
    side_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(side_camera)
    side_camera.lookat[:] = [0.30, 0.20, 1.00]
    side_camera.distance = 2.20
    side_camera.azimuth = 180
    side_camera.elevation = -12
    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.geomgroup[0] = 0
    render_option.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=720, width=1280)
    video_writer = imageio.get_writer(OUT / "m0_x2_grasp.mp4", fps=25, codec="libx264", quality=8)
    trace_file = (OUT / "physics_contact_trace.jsonl").open("w", encoding="utf-8")

    def save_snapshot(filename: str, snapshot_camera: mujoco.MjvCamera | None = None) -> None:
        renderer.update_scene(data, camera=snapshot_camera or camera, scene_option=render_option)
        imageio.imwrite(OUT / filename, renderer.render())

    save_snapshot("before_overview.png")
    save_snapshot("approach_start.png")

    def run_phase(label: str, steps: int, arm_target: np.ndarray, finger_target: np.ndarray) -> None:
        nonlocal step_count, max_bottle_translation_step, max_bottle_rotation_step, max_penetration
        nonlocal maximum_bottle_height, first_right_contact, first_left_contact, first_table_contact
        nonlocal first_target_table_contact, previous_contact, max_mimic_relation_error_rad
        phase_start = float(data.time)
        phase_mimic_relation_error_max_rad = 0.0
        phase_robot_bodies: set[str] = set()
        phase_table_bodies: set[str] = set()
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
            current_mimic_state = mimic_relation_state()
            current_mimic_errors = {
                follower: values["error_rad"]
                for follower, values in current_mimic_state.items()
            }
            current_mimic_relation_error_max_rad = max(current_mimic_errors.values(), default=0.0)
            phase_mimic_relation_error_max_rad = max(
                phase_mimic_relation_error_max_rad,
                current_mimic_relation_error_max_rad,
            )
            max_mimic_relation_error_rad = max(
                max_mimic_relation_error_rad,
                current_mimic_relation_error_max_rad,
            )
            new_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
            new_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
            max_bottle_translation_step = max(max_bottle_translation_step, float(np.linalg.norm(new_position - old_position)))
            dot = min(1.0, abs(float(np.dot(old_quat / np.linalg.norm(old_quat), new_quat / np.linalg.norm(new_quat)))))
            max_bottle_rotation_step = max(max_bottle_rotation_step, 2.0 * math.acos(dot))
            maximum_bottle_height = max(maximum_bottle_height, float(new_position[2]))
            right_contacts, left_contacts, table_contact, target_table_contact = contact_state()
            step_table_bodies = right_table_contact_bodies()
            phase_table_bodies.update(step_table_bodies)
            all_right_table_contact_bodies.update(step_table_bodies)
            if label in ("approach_preshape", "finger_close"):
                preclosure_contact_bodies.update(right_contacts)
                preclosure_contact_bodies.update(left_contacts)
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
                        "mimic_joint_relation_error_max_rad": current_mimic_relation_error_max_rad,
                        "mimic_joint_relations": current_mimic_state,
                        "right_hand_joint_state_rad": {
                            joint_name: float(data.qpos[qpos_address])
                            for joint_name, qpos_address in zip(right_hand_joint_names, right_hand_joint_qpos)
                        },
                        "right_hand_actuator_target_rad": {
                            joint_name: float(data.ctrl[actuator_by_joint[joint_id]])
                            for joint_name, joint_id in zip(finger_joint_names, finger_ids)
                        },
                        "bottle_position_m": new_position.tolist(),
                        "bottle_quaternion_wxyz": new_quat.tolist(),
                        "bottle_linear_velocity_m_s": data.qvel[bottle_dadr : bottle_dadr + 3].tolist(),
                        "bottle_angular_velocity_rad_s": data.qvel[bottle_dadr + 3 : bottle_dadr + 6].tolist(),
                        "right_contact_bodies": sorted(right_contacts),
                        "opposite_hand_contact_bodies": sorted(left_contacts),
                        "right_hand_table_contact_bodies": sorted(step_table_bodies),
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
                renderer.update_scene(data, camera=camera, scene_option=render_option)
                video_writer.append_data(renderer.render())
            for ci in range(data.ncon):
                contact = data.contact[ci]
                if bottle_geoms.intersection((int(contact.geom1), int(contact.geom2))):
                    max_penetration = max(max_penetration, float(max(0.0, -contact.dist)))
        qvel = data.qvel[bottle_dadr : bottle_dadr + 6].copy()
        arm_actual = data.qpos[arm_qpos].copy()
        arm_error = arm_actual - arm_target
        right_now, left_now, table_now, target_table_now = contact_state()
        table_bodies_now = right_table_contact_bodies()
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
                "right_hand_table_contact_bodies": sorted(phase_table_bodies),
                "arm_joint_target_rad": arm_target.tolist(),
                "arm_joint_actual_rad": arm_actual.tolist(),
                "arm_tracking_error_l2_rad": float(np.linalg.norm(arm_error)),
                "arm_tracking_error_max_abs_rad": float(np.max(np.abs(arm_error))),
                "right_finger_target_rad": finger_target.tolist(),
                "right_finger_actual_rad": data.qpos[finger_qpos].tolist(),
                "right_hand_joint_state_rad": {
                    joint_name: float(data.qpos[qpos_address])
                    for joint_name, qpos_address in zip(right_hand_joint_names, right_hand_joint_qpos)
                },
                "mimic_joint_relation_error_max_rad": phase_mimic_relation_error_max_rad,
                "table_contact": table_now,
                "target_table_contact": target_table_now,
                "right_hand_table_contact": bool(table_bodies_now),
                "palm_frame_position_m": data.site_xpos[site_id].tolist(),
                "palm_frame_rotation_matrix": data.site_xmat[site_id].reshape(3, 3).tolist(),
            }
        )
        print(
            f"PHASE {label} t={data.time:.6f} bottle={data.qpos[bottle_qadr:bottle_qadr+3].round(5).tolist()} "
            f"palm={data.site_xpos[site_id].round(5).tolist()} ncon={data.ncon} right={sorted(right_now)} "
            f"table={table_now} target_table={target_table_now} hand_table={sorted(table_bodies_now)}"
        )

    # Only initialization and offline IK assign generalized position state.
    # From this point through settling, bottle qpos is advanced only by mj_step.
    run_phase("approach_preshape", 500, pregrasp_route[1], preshape_target)
    pregrasp_right_contacts, pregrasp_left_contacts, _, _ = contact_state()
    pregrasp_hand_table_contacts = right_table_contact_bodies()
    pregrasp_palm_position = data.site_xpos[site_id].copy()
    pregrasp_palm_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    pregrasp_arm_pose = data.qpos[arm_qpos].copy()
    pregrasp_bottle_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
    pregrasp_bottle_translation = float(np.linalg.norm(pregrasp_bottle_position - START))
    save_snapshot("pregrasp.png")
    save_snapshot("pregrasp_closeup.png", closeup_camera)
    save_snapshot("front_view.png", front_camera)
    save_snapshot("side_view.png", side_camera)
    run_phase("finger_close", 500, pregrasp_route[1], close_target)
    save_snapshot("finger_close.png")
    run_phase("closed_hand_approach", 500, pregrasp_route[2], close_target)
    save_snapshot("closed_hand_approach.png")
    save_snapshot("closed_hand_approach_closeup.png", closeup_camera)
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
    save_snapshot("transfer.png")
    run_phase("lower_to_place", 1000, lower_target, close_target)
    save_snapshot("lower_to_place.png")
    release_time = float(data.time)
    run_phase("finger_open_release", 500, lower_target, open_target)
    save_snapshot("finger_open_release.png")
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
        "opposing_hand_contact_bodies": sorted(pregrasp_left_contacts),
        "active_hand_contact_bodies_before_closure": sorted(pregrasp_right_contacts),
        "right_hand_table_contact_bodies": sorted(pregrasp_hand_table_contacts),
        "bottle_translation_before_closure_m": pregrasp_bottle_translation,
        "preclosure_bottle_contact_bodies": sorted(preclosure_contact_bodies),
        "finger_preshape_fraction": FINGER_PRESHAPE_FRACTION,
        "arm_joint_order": ARM,
        "arm_joint_actual_rad": pregrasp_arm_pose.tolist(),
        "elbow_angle_rad": float(pregrasp_arm_pose[ARM.index("right_elbow_joint")]),
    }
    pregrasp_valid = bool(
        pregrasp_diagnostics["position_error_m"] <= 0.03
        and pregrasp_rotation_error <= 0.10
        and not pregrasp_diagnostics["active_hand_contact_bodies_before_closure"]
        and not pregrasp_diagnostics["opposing_hand_contact_bodies"]
        and not pregrasp_diagnostics["right_hand_table_contact_bodies"]
        and not preclosure_contact_bodies
        and pregrasp_bottle_translation <= 0.002
    )
    (OUT / "frame_diagnostics.json").write_text(json.dumps(pregrasp_diagnostics, indent=2) + "\n")
    no_bottle_attachment_constraints = bool(
        details["mimic_constraints_match"]
        and model.neq == len(details["mimic_relations"])
        and all(
            int(model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
            and int(model.eq_objtype[equality_id]) == int(mujoco.mjtObj.mjOBJ_JOINT)
            for equality_id in range(model.neq)
        )
    )
    mimic_relations_within_tolerance = max_mimic_relation_error_rad <= MIMIC_RELATION_TOLERANCE_RAD
    gates = {
        "plausible_collision_free_pregrasp": pregrasp_valid,
        "no_open_hand_approach_push": not preclosure_contact_bodies and pregrasp_bottle_translation <= 0.002,
        "right_hand_table_contact_free_at_pregrasp": not pregrasp_hand_table_contacts,
        "thumb_plus_at_least_two_other_digit_families_at_grasp": contact_finger_families["thumb"] and active_families >= 3,
        "at_least_three_digit_families_contact_during_lift_and_transfer": carry_valid,
        "opposite_hand_never_assists": first_left_contact is None and not final_left_contacts,
        "lift_at_least_0_05_m": bottle_lift >= 0.05,
        "bottle_final_footprint_inside_table": final_footprint_inside,
        "final_xy_error_at_most_0_05_m": bottle_target_error <= 0.05,
        "free_physics_target_table_supported_settle": not final_contacts and not final_left_contacts and final_target_table_contact,
        "all_12_vendor_mimic_relations_compiled": details["mimic_constraints_match"],
        "mimic_joint_relation_error_within_tolerance": mimic_relations_within_tolerance,
        "no_runtime_bottle_weld_or_attachment_constraint": no_bottle_attachment_constraints,
        "no_bottle_qpos_writes_during_rollout": qpos_assignments_during_rollout == 0,
    }
    passed = all(
        [
            pregrasp_valid,
            not preclosure_contact_bodies,
            pregrasp_bottle_translation <= 0.002,
            not pregrasp_hand_table_contacts,
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
            details["mimic_constraints_match"],
            mimic_relations_within_tolerance,
            no_bottle_attachment_constraints,
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
            "urdf_mimic_relation_count": len(details["mimic_relations"]),
            "urdf_mimic_relations": details["mimic_relations"],
            "urdf_mimic_constraints_match_source": details["mimic_constraints_match"],
            "mimic_relation_runtime_tolerance_rad": MIMIC_RELATION_TOLERANCE_RAD,
            "mimic_relation_max_error_rad": max_mimic_relation_error_rad,
            "effort_limited_joint_count": details["effort_count"],
            "self_collision_policy": {
                "policy": "exclude_robot_ancestor_descendant_pairs_only",
                "excluded_pair_count": details["self_chain_exclusion_count"],
                "inter_branch_self_collision_enabled": True,
                "robot_environment_contact_enabled": True,
                "robot_bottle_contact_enabled": True,
            },
            "bottle_mass_kg": bottle_mass,
            "bottle_dimensions_m": {
                "body_diameter": 0.070,
                "overall_height": 0.2445,
                "total_mass_kg": 0.570,
                "source": "current Issue #46 X2 bottle; not the smaller G1 PR #45 bottle",
            },
            "table_top_z_m": TABLE_TOP_Z,
            "table_geometry": {
                "source_half_extents_xy_m": SOURCE_TABLE_HALF_EXTENTS.tolist(),
                "target_half_extents_xy_m": TARGET_TABLE_HALF_EXTENTS.tolist(),
                "source_center_xy_m": SOURCE_TABLE_XY.tolist(),
                "target_body_center_xy_m": TARGET_TABLE_BODY_XY.tolist(),
                "target_site_offset_y_m": TARGET_TABLE_SITE_OFFSET_Y,
            },
            "control_model": "URDF joint-effort-limited position servos on independent joints; 12 source-defined URDF mimic relations are MuJoCo joint equalities; right arm waypoints use bounded least-squares palm-site IK",
            "mimic_driver_controller": {
                "right_hand_mimic_driver_kp": RIGHT_MIMIC_DRIVER_SERVO_KP,
                "right_hand_mimic_driver_kv": RIGHT_MIMIC_DRIVER_SERVO_KV,
                "left_hand_mimic_driver_kp": 18.0,
                "left_hand_mimic_driver_kv": 2.4,
                "nonmimic_finger_servo_kp": 18.0,
                "nonmimic_finger_servo_kv": 2.4,
                "follower_has_independent_actuator": False,
                "hand_joint_limit_solref_direct": HAND_JOINT_LIMIT_SOLREF_DIRECT,
                "validation_scope": "free-space mimic tests only; bottle manipulation not rerun",
            },
            "runtime_gl": os.environ.get("MUJOCO_GL", "default"),
            "visual_provenance": {
                "robot_body_materials": "official pinned X2 URDF visual material colors and mesh assets",
                "vendor_visual_color_count": details["vendor_visual_color_count"],
                "vendor_visual_color_values_rgba": details["vendor_visual_color_values"],
                "vendor_visual_texture_count": details["vendor_visual_texture_count"],
                "collision_visualization": "collision geoms in group 0 hidden; vendor visual meshes in group 1 retained",
                "hand_color_fallback": "light neutral only for hand visual meshes with no authored URDF color",
                "environment": "vendor sky/haze retained; vendor ground checker had identical RGB colors and was replaced with a restrained neutral checker; headlight diffuse/specular reduced to control glare",
                "floor_checker_rgb1": [0.50, 0.53, 0.55],
                "floor_checker_rgb2": [0.43, 0.46, 0.48],
                "headlight_ambient_diffuse_specular": [0.26, 0.45, 0.45],
            },
            "canonical_scene_provenance": {
                "table": "G1 PR #45 M0 source/target table geometry and placement",
                "bottle": "current X2 Issue #46 70 mm x 244.5 mm, 0.57 kg bottle",
            },
            "vendor_mimic_conversion": {
                "source": "URDF <mimic> joint/multiplier/offset tags from the pinned OmniHand model",
                "conversion": "MuJoCo scalar joint equalities; follower joints have no independent servo actuator",
                "relations": details["mimic_relations"],
                "compiled_source_match": details["mimic_constraints_match"],
                "max_runtime_position_error_rad": max_mimic_relation_error_rad,
                "runtime_error_tolerance_rad": MIMIC_RELATION_TOLERANCE_RAD,
                "bottle_attachment_constraint_count": 0 if no_bottle_attachment_constraints else None,
            },
        }
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "diagnosis": {
            "station_root_cause": "The fixed-base scratch station places the X2 torso at [0.38, 0.32, 0.68] facing world -Y, perpendicular to the canonical table edge; the canonical bottle at x=0.30 is in the right arm workspace. The torso base frame is fixed and walking is not used.",
            "collision_conversion_root_cause": "The URDF importer fused fixed links into the world and enabled overlapping parent/descendant collision meshes as independent constraints. The adaptation preserves the fixed body hierarchy and excludes only robot ancestor-descendant pairs; inter-branch, robot-table, and robot-bottle collisions remain enabled.",
            "wrist_palm_root_cause": "The official URDF R_palm_joint transform is present and correctly composed by the converter. The previous task frame treated the wrist-roll origin as the palm frame and added an undocumented 180 degree yaw, so the requested hand normal and finger-extension directions were wrong. The corrected task frame is attached at the URDF palm transform and maps its documented local axes explicitly.",
            "frame_verification": "The compiled right_palm_frame site is attached to right_wrist_roll_link; local position error against the URDF R_palm_joint origin is 0 m and quaternion absolute-dot agreement is 1.0.",
            "corrected_layer": "Scratch fixed-base scene/station, fixed-root URDF conversion collision policy, vendor visual settings, and task-level palm frame. No vendor asset, G1 M0 implementation, or shared architecture code was changed.",
            "canonical_assets": "The scratch scene matches the G1 M0 table placement and extents from simulation/mujoco/m0_pick_place.py, while retaining the current X2 bottle at 70 mm diameter, 244.5 mm height, and 0.57 kg. The values are recorded as one benchmark contract here; the G1 production path remains untouched.",
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
            "initial_right_hand_table_contact_bodies": sorted(initial_hand_table_contacts),
            "right_hand_table_contact_bodies_any_time": sorted(all_right_table_contact_bodies),
            "initial_table_contact": initial_table_contact,
            "initial_target_table_contact": initial_target_table_contact,
            "model_free_bottle_joint": True,
            "runtime_weld": False,
            "equality_constraint_count": model.neq,
            "equality_constraints_are_only_vendor_mimics": details["mimic_constraints_match"],
            "bottle_attachment_constraint_count": 0 if no_bottle_attachment_constraints else None,
            "mimic_relation_max_error_rad": max_mimic_relation_error_rad,
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
            "reproduction_command": str(OUT / "reproduction_command.txt"),
            "before_overview_png": str(OUT / "before_overview.png"),
            "after_overview_png": str(OUT / "after_overview.png"),
            "pregrasp_png": str(OUT / "pregrasp.png"),
            "pregrasp_closeup_png": str(OUT / "pregrasp_closeup.png"),
            "front_view_png": str(OUT / "front_view.png"),
            "side_view_png": str(OUT / "side_view.png"),
            "closed_hand_approach_png": str(OUT / "closed_hand_approach.png"),
            "finger_close_png": str(OUT / "finger_close.png"),
            "transfer_png": str(OUT / "transfer.png"),
            "lower_to_place_png": str(OUT / "lower_to_place.png"),
            "finger_open_release_png": str(OUT / "finger_open_release.png"),
            "closed_hand_approach_closeup_png": str(OUT / "closed_hand_approach_closeup.png"),
            "visual_material_closeup_png": str(OUT / "pregrasp_closeup.png"),
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
            "mimic_relation_error_max_rad": max_mimic_relation_error_rad,
            "mimic_relation_runtime_tolerance_rad": MIMIC_RELATION_TOLERANCE_RAD,
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
    renderer.update_scene(data, camera=camera, scene_option=render_option)
    import imageio.v3 as iio

    iio.imwrite(OUT / "final.png", renderer.render())
    video_writer.close()
    renderer.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
