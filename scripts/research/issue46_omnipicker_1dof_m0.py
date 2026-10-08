#!/usr/bin/env python3
"""Issue #46 SIMULATION_ONLY_M0 paired-jaw OmniPicker experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_REPO = "https://github.com/AgibotTech/agibot_x2_urdf"
X2_ROOT = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
X2_URDF_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
X2_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
STATIC_ROOT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready")
CANONICAL_MODULE_SHA256 = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"
OUTPUT_DEFAULT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-1dof-m0/20261007-run1")

TIMESTEP = 0.002
BASE_WORLD = np.array([0.38, 0.32, 0.68], dtype=float)
BASE_YAW = -math.pi / 2.0
LEFT_DRIVER_HOLD = -0.5
ENDPOINT_MARGIN_RAD = 0.001
APERTURE_RELATION_DIAGNOSTIC_RAD = 0.02
CONTACT_FORCE_BEARING_MIN_N = 1e-6
CONTACT_ACCELERATION_BOUND_RAD_S2 = 2.0 * 75.6624308578453
JAW_NATURAL_FREQUENCY_RAD_S = 8.0
DAMPING_RATIO = 1.0
MOTION_DURATION_S = 2.5
ARM_NATURAL_FREQUENCY_RAD_S = 4.0
ARM_MOTION_DURATION_S = 2.5
CONTACT_FRICTION_ASSUMPTION = 1.4
CONTACT_NORMAL_DESIGN_FACTOR = 2.0
GRAVITY_M_S2 = 9.81
POSE_LOCK_SOLREF = [0.002, 1.0]
FIXTURE_TARGET_BODIES = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
FIXTURE_RADIUS_M = 0.020
FIXTURE_HALF_HEIGHT_M = 0.025

RIGHT_DRIVER = "right_claw_joint"
RIGHT_FOLLOWER = "R_hand_wide1_joint"
LEFT_DRIVER = "left_claw_joint"
LEFT_FOLLOWER = "L_hand_wide1_joint"
PICKER_JOINTS = {RIGHT_DRIVER, RIGHT_FOLLOWER, LEFT_DRIVER, LEFT_FOLLOWER}
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
STATIC_COLLISION_EXCLUSIONS = (
    ("pelvis", "left_hip_pitch_link"),
    ("pelvis", "left_hip_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("pelvis", "right_hip_roll_link"),
    ("pelvis", "waist_yaw_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("waist_yaw_link", "torso_link"),
    ("torso_link", "head_pitch_link"),
    ("left_wrist_yaw_link", "left_wrist_roll_link"),
    ("right_wrist_yaw_link", "right_wrist_roll_link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise RuntimeError(f"Compiled model is missing joint {name}")
    return int(value)


def qpos_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, name)])


def dof_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, name)])


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"Compiled model is missing body {name}")
    return int(value)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if value < 0:
        raise RuntimeError(f"Compiled model is missing actuator {name}")
    return int(value)


def verify_inputs(out: Path) -> dict[str, Any]:
    urdf = X2_ROOT / X2_URDF_REL
    canonical_module = STATIC_ROOT / "canonical_manipulation_assets.py"
    if not urdf.is_file() or not canonical_module.is_file():
        raise FileNotFoundError("Pinned X2 URDF or accepted canonical scene helper is unavailable")
    commit = subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(X2_ROOT), "status", "--porcelain"], text=True).strip()
    actual_urdf_sha = sha256(urdf)
    canonical_sha = sha256(canonical_module)
    if commit != X2_PIN or dirty:
        raise RuntimeError(f"Pinned X2 checkout mismatch: commit={commit}, dirty={bool(dirty)}")
    if actual_urdf_sha != X2_URDF_SHA256 or canonical_sha != CANONICAL_MODULE_SHA256:
        raise RuntimeError("Pinned URDF or accepted canonical scene helper hash mismatch")
    root = ET.parse(urdf).getroot()
    joints = {j.get("name", ""): j for j in root.findall("joint")}
    joint_rows: list[dict[str, Any]] = []
    for name in (RIGHT_DRIVER, RIGHT_FOLLOWER, LEFT_DRIVER, LEFT_FOLLOWER):
        joint = joints[name]
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        joint_rows.append({
            "name": name,
            "type": joint.get("type"),
            "axis": [float(v) for v in joint.find("axis").get("xyz", "0 0 1").split()],
            "source_position_range_rad": [float(limit.get("lower")), float(limit.get("upper"))],
            "source_effort_metadata": float(limit.get("effort", "0")),
            "source_velocity_metadata": float(limit.get("velocity", "0")),
            "source_zero_effort_velocity_interpretation": "UNSPECIFIED_DYNAMIC_METADATA",
            "mimic": None if mimic is None else {
                "driver": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            },
        })
    right_mimic = joints[RIGHT_FOLLOWER].find("mimic")
    if right_mimic is None or right_mimic.get("joint") != RIGHT_DRIVER:
        raise RuntimeError("Pinned right OmniPicker source mimic relation is missing")
    multiplier = float(right_mimic.get("multiplier", "1"))
    offset = float(right_mimic.get("offset", "0"))
    if multiplier != -1.0 or offset != 0.0:
        raise RuntimeError(f"Unexpected pinned source OmniPicker mapping {multiplier=}, {offset=}")
    return {
        "repository": X2_REPO,
        "commit": commit,
        "urdf_path": str(X2_URDF_REL),
        "urdf_sha256": actual_urdf_sha,
        "license": "Mulan PSL v2",
        "working_tree_clean": not bool(dirty),
        "right_source_mapping": {"driver": RIGHT_DRIVER, "follower": RIGHT_FOLLOWER,
                                  "multiplier": multiplier, "offset": offset},
        "joint_records": joint_rows,
        "urdf_transmission_count": len(root.findall(".//transmission")),
        "canonical_scene_helper": str(canonical_module),
        "canonical_scene_helper_sha256": canonical_sha,
    }


def aperture_targets(aperture_ratio: float, aperture_velocity: float = 0.0,
                     aperture_acceleration: float = 0.0) -> dict[str, float]:
    if not 0.0 <= aperture_ratio <= 1.0:
        raise ValueError("aperture_ratio must be in [0, 1]")
    span = 1.0 - 2.0 * ENDPOINT_MARGIN_RAD
    driver = -(ENDPOINT_MARGIN_RAD + span * aperture_ratio)
    driver_v = -span * aperture_velocity
    driver_a = -span * aperture_acceleration
    return {
        "aperture_ratio": aperture_ratio,
        "right_claw_joint_target_rad": driver,
        "R_hand_wide1_joint_target_rad": -driver,
        "right_claw_joint_target_velocity_rad_s": driver_v,
        "R_hand_wide1_joint_target_velocity_rad_s": -driver_v,
        "right_claw_joint_target_acceleration_rad_s2": driver_a,
        "R_hand_wide1_joint_target_acceleration_rad_s2": -driver_a,
    }


def minimum_jerk(t: float, duration: float, start: float, end: float) -> tuple[float, float, float]:
    if t <= 0.0:
        return start, 0.0, 0.0
    if t >= duration:
        return end, 0.0, 0.0
    s = t / duration
    p = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    dp = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    ddp = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    return start + (end - start) * p, (end - start) * dp, (end - start) * ddp


def add_floor(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="issue46_floor_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
        width=300, height=300,
    )
    spec.add_material(name="issue46_floor", textures=["", "issue46_floor_texture"],
                      texuniform=True, texrepeat=[5.0, 5.0], reflectance=0.2)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.0, 0.0, 0.05], material="issue46_floor", group=1,
                            friction=[1.0, 0.01, 0.001], condim=4)
    spec.worldbody.add_light(name="issue46_key", pos=[0.0, 0.0, 1.5], dir=[0.0, 0.0, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


def add_canonical_scene(spec: mujoco.MjSpec, bottle: bool) -> None:
    sys.path.insert(0, str(STATIC_ROOT))
    import canonical_manipulation_assets as canonical

    add_floor(spec)
    canonical.add_g1_canonical_table(spec)
    if bottle:
        bottle_body = spec.worldbody.add_body(name="m0_bottle", pos=list(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS))
        bottle_body.add_freejoint(name="m0_bottle_free")
        geom_types = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                      "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
        for item in canonical.CANONICAL_X2_BOTTLE_GEOMS:
            bottle_body.add_geom(
                name=item["name"], type=geom_types[item["type"]], pos=list(item["pos"]),
                size=list(item["size"]), mass=float(item["mass"]), rgba=list(item["rgba"]),
                friction=[1.4, 0.02, 0.001], condim=4, group=1,
            )


def collision_geom_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = body_id(model, body_name)
    matches = [gid for gid in range(model.ngeom)
               if int(model.geom_bodyid[gid]) == bid
               and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one compiled collision geom on {body_name}, found {matches}")
    return matches[0]


def geom_mesh_world_vertices(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise RuntimeError("Pinned wrist collision geometry is expected to be a mesh")
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    local = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return local @ rotation.T + data.geom_xpos[geom_id]


def derive_fixture_geometry() -> dict[str, Any]:
    model, _ = build_model("open_hold")
    data = initialize(model)
    narrow_id = collision_geom_id(model, FIXTURE_TARGET_BODIES[0])
    wide_id = collision_geom_id(model, FIXTURE_TARGET_BODIES[1])
    segment = np.zeros(6, dtype=float)
    jaw_gap = float(mujoco.mj_geomDistance(model, data, narrow_id, wide_id, 1.0, segment))
    if jaw_gap <= 2.0 * FIXTURE_RADIUS_M:
        raise RuntimeError(f"Open jaw surface gap cannot contain the fixed cylinder: {jaw_gap=}")
    narrow_surface = segment[:3].copy()
    wide_surface = segment[3:].copy()
    jaw_normal = wide_surface - narrow_surface
    jaw_normal /= float(np.linalg.norm(jaw_normal))
    jaw_midpoint = 0.5 * (narrow_surface + wide_surface)

    wrist_records = []
    for body_name in ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"):
        geom_id = collision_geom_id(model, body_name)
        vertices = geom_mesh_world_vertices(model, data, geom_id)
        index = int(np.argmin(vertices[:, 2]))
        wrist_records.append({
            "body": body_name,
            "geom_id": geom_id,
            "mesh": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                       int(model.geom_dataid[geom_id])),
            "lowest_world_vertex_m": vertices[index].tolist(),
        })
    center = jaw_midpoint.tolist()

    urdf = ET.parse(X2_ROOT / X2_URDF_REL).getroot()
    source_links = {link.get("name", ""): link for link in urdf.findall("link")}
    surfaces = []
    for body_name, geom_id, point, inward_normal in (
            (FIXTURE_TARGET_BODIES[0], narrow_id, narrow_surface, jaw_normal),
            (FIXTURE_TARGET_BODIES[1], wide_id, wide_surface, -jaw_normal)):
        collision = source_links[body_name].find("collision")
        mesh = collision.find("geometry/mesh") if collision is not None else None
        surfaces.append({
            "body": body_name,
            "geom_id": geom_id,
            "compiled_mesh": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                                int(model.geom_dataid[geom_id])),
            "source_collision_mesh": None if mesh is None else mesh.get("filename"),
            "surface_world_m": point.tolist(),
            "inward_contact_normal_world": inward_normal.tolist(),
        })
    return {
        "source_of_geometry": "compiled pinned X2 collision meshes at accepted OPEN qpos; no body-origin reference",
        "target_surfaces": surfaces,
        "closest_open_surface_gap_m": jaw_gap,
        "narrow_to_wide_corridor_normal_world": jaw_normal.tolist(),
        "jaw_surface_midpoint_world_m": jaw_midpoint.tolist(),
        "wrist_collision_meshes": wrist_records,
        "wrist_clearance_rule": "wrist collision geometry is checked by the zero-step compiled-geom preflight; it does not set fixture height",
        "cylinder": {"type": "vertical fixed cylinder", "radius_m": FIXTURE_RADIUS_M,
                     "half_height_m": FIXTURE_HALF_HEIGHT_M, "center_world_m": center,
                     "position_rule": "XYZ center is the midpoint of the two intended compiled jaw-surface witnesses"},
    }


def fixture_geometry_preflight(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    fixture_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_fixed_contact_cylinder")
    if fixture_id < 0:
        raise RuntimeError("Fixed fixture geom is missing from compiled model")
    rows = []
    target_by_body = {body: role for body, role in zip(FIXTURE_TARGET_BODIES,
                                                         ("intended_narrow_jaw", "intended_wide_jaw"))}
    for geom_id in range(model.ngeom):
        if geom_id == fixture_id:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[geom_id])) or "world"
        if body == "world":
            continue
        collision_enabled = bool(
            (int(model.geom_contype[fixture_id]) & int(model.geom_conaffinity[geom_id]))
            or (int(model.geom_contype[geom_id]) & int(model.geom_conaffinity[fixture_id]))
        )
        segment = np.zeros(6, dtype=float)
        distance = float(mujoco.mj_geomDistance(model, data, fixture_id, geom_id, 1.0, segment))
        mesh_id = int(model.geom_dataid[geom_id]) if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
        role = target_by_body.get(body)
        if role is None and "wrist" in body.lower():
            role = "wrist"
        elif role is None and "loop" in body.lower():
            role = "jaw_loop"
        elif role is None:
            role = "other_right_hand_or_arm"
        rows.append({
            "body": body,
            "geom_id": geom_id,
            "compiled_mesh": None if mesh_id < 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
            "collision_enabled_with_fixture": collision_enabled,
            "role": role,
            "signed_distance_m": distance,
            "penetration_m": max(0.0, -distance),
            "witness_segment_world_m": segment.tolist(),
        })
    collidable = [row for row in rows if row["collision_enabled_with_fixture"]]
    target = [row for row in collidable if row["role"].startswith("intended_")]
    failures = [row for row in collidable if row["signed_distance_m"] <= 0.0]
    fixture_contacts = [row for row in contact_rows(model, data)
                        if contact_has_object(row, "m0_fixed_contact_cylinder")]
    return {
        "status": "PASS" if not failures and len(target) == len(FIXTURE_TARGET_BODIES) else "FAIL",
        "physics_steps_before_check": 0,
        "fixture_robot_contact_count": len(fixture_contacts),
        "fixture_robot_contacts": fixture_contacts,
        "distances": rows,
        "minimum_collidable_clearance_m": min((row["signed_distance_m"] for row in collidable), default=None),
        "intended_jaw_clearances_m": {row["body"]: row["signed_distance_m"] for row in target},
        "penetrating_or_touching_collidable_geoms": failures,
    }


def fixture_closure_geometry_check(model: mujoco.MjModel) -> dict[str, Any]:
    fixture_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_fixed_contact_cylinder")
    target_ids = {name: collision_geom_id(model, name) for name in FIXTURE_TARGET_BODIES}
    probe = initialize(model)
    samples = []
    monotonic_tolerance_m = 1e-6
    for aperture in np.linspace(1.0, 0.0, 51):
        targets = aperture_targets(float(aperture))
        probe.qpos[qpos_id(model, RIGHT_DRIVER)] = targets["right_claw_joint_target_rad"]
        probe.qpos[qpos_id(model, RIGHT_FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
        probe.qvel[:] = 0.0
        mujoco.mj_forward(model, probe)
        distances = {}
        for name, geom_id in target_ids.items():
            segment = np.zeros(6, dtype=float)
            distances[name] = float(mujoco.mj_geomDistance(model, probe, fixture_id, geom_id, 1.0, segment))
        samples.append({"aperture_ratio": float(aperture), "clearances_m": distances})

    monotonic = {
        name: all(samples[i + 1]["clearances_m"][name] <=
                  samples[i]["clearances_m"][name] + monotonic_tolerance_m
                  for i in range(len(samples) - 1))
        for name in target_ids
    }
    contact_aperture = {
        name: next((sample["aperture_ratio"] for sample in samples
                    if sample["clearances_m"][name] <= 0.0), None)
        for name in target_ids
    }
    return {
        "status": "PASS" if all(monotonic.values()) and all(v is not None for v in contact_aperture.values()) else "FAIL",
        "physics_steps": 0,
        "aperture_samples": len(samples),
        "monotonic_tolerance_m": monotonic_tolerance_m,
        "clearances_monotonically_decrease": monotonic,
        "first_geometric_contact_aperture_ratio": contact_aperture,
        "samples": samples,
    }


def build_model(stage: str, params: dict[str, Any] | None = None) -> tuple[mujoco.MjModel, dict[str, Any]]:
    spec = mujoco.MjSpec.from_file(str(X2_ROOT / X2_URDF_REL))
    spec.compiler.fusestatic = False
    spec.option.timestep = TIMESTEP
    spec.option.gravity = [0.0, 0.0, -GRAVITY_M_S2]
    pelvis = next((body for body in spec.bodies if body.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("Pinned X2 URDF has no pelvis root")
    pelvis.pos = BASE_WORLD.tolist()
    pelvis.quat = [math.cos(BASE_YAW / 2.0), 0.0, 0.0, math.sin(BASE_YAW / 2.0)]
    for i, (body1, body2) in enumerate(STATIC_COLLISION_EXCLUSIONS):
        spec.add_exclude(name=f"accepted_static_overlap_{i}", bodyname1=body1, bodyname2=body2)

    active_arm = stage in {"bottle_hold", "lift"}
    arm_set = set(RIGHT_ARM_JOINTS) if active_arm else set()
    source_limits: dict[str, tuple[float, float]] = {}
    source_efforts: dict[str, float] = {}
    root = ET.parse(X2_ROOT / X2_URDF_REL).getroot()
    source_joints = {j.get("name", ""): j for j in root.findall("joint")}
    for name, joint_xml in source_joints.items():
        limit = joint_xml.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            source_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
            source_efforts[name] = float(limit.get("effort", "0"))

    equalities: list[str] = []
    joints = {str(joint.name): joint for joint in spec.joints if joint.name}
    for name, joint in joints.items():
        if name in {RIGHT_DRIVER, RIGHT_FOLLOWER} or name in arm_set:
            continue
        if name == LEFT_DRIVER:
            target = LEFT_DRIVER_HOLD
        elif name == LEFT_FOLLOWER:
            target = -LEFT_DRIVER_HOLD
        else:
            target = float(joint.ref)
        spec.add_equality(
            name=f"m0_fixed_pose_{name}", type=mujoco.mjtEq.mjEQ_JOINT,
            name1=name, data=[target - float(joint.ref), 0.0, 0.0, 0.0, 0.0] + [0.0] * 6,
            solref=POSE_LOCK_SOLREF,
        )
        equalities.append(name)

    provisional_cap = 1.0
    for name, joint_name in (("m0_right_driver_servo", RIGHT_DRIVER), ("m0_right_follower_servo", RIGHT_FOLLOWER)):
        spec.add_actuator(
            name=name, target=joint_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
            ctrllimited=True, ctrlrange=[-provisional_cap, provisional_cap],
            forcelimited=True, forcerange=[-provisional_cap, provisional_cap],
        )

    arm_actuators: dict[str, str] = {}
    if active_arm:
        for joint_name in RIGHT_ARM_JOINTS:
            effort = source_efforts.get(joint_name, 0.0)
            if effort <= 0.0:
                raise RuntimeError(f"Right arm joint has no positive source effort limit: {joint_name}")
            act_name = f"m0_arm_{joint_name}"
            spec.add_actuator(
                name=act_name, target=joint_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
                ctrllimited=True, ctrlrange=[-effort, effort],
                forcelimited=True, forcerange=[-effort, effort],
            )
            arm_actuators[joint_name] = act_name

    fixture_derivation = None
    if stage == "fixed_object":
        fixture_derivation = derive_fixture_geometry()
        spec.worldbody.add_geom(
            name="m0_fixed_contact_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=fixture_derivation["cylinder"]["center_world_m"],
            size=[FIXTURE_RADIUS_M, FIXTURE_HALF_HEIGHT_M, 0.0], rgba=[0.92, 0.46, 0.08, 1.0],
            friction=[1.4, 0.02, 0.001], condim=4, group=1,
        )
    if stage in {"bottle_hold", "lift"}:
        add_canonical_scene(spec, bottle=True)

    model = spec.compile()
    if not hasattr(model, "eq_obj1id"):
        raise RuntimeError("Unexpected MuJoCo model equality metadata")
    equality_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "" for i in range(model.neq)]
    for name in equality_names:
        if RIGHT_DRIVER in name or RIGHT_FOLLOWER in name:
            raise RuntimeError(f"Right jaw equality must not exist in the dynamic path: {name}")
    if model.nu < 2 or actuator_id(model, "m0_right_driver_servo") < 0 or actuator_id(model, "m0_right_follower_servo") < 0:
        raise RuntimeError("Expected two internal right jaw actuators")
    for name, target in (("m0_right_driver_servo", RIGHT_DRIVER), ("m0_right_follower_servo", RIGHT_FOLLOWER)):
        aid = actuator_id(model, name)
        if int(model.actuator_trnid[aid, 0]) != joint_id(model, target):
            raise RuntimeError(f"Internal actuator target mismatch: {name}")
    if stage == "fixed_object" and "m0_fixed_contact_cylinder" not in (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m0_fixed_contact_cylinder")) or ""):
        raise RuntimeError("Fixed-object diagnostic geom missing")
    params = dict(params or {})
    params["source_position_limits_rad"] = {k: list(v) for k, v in source_limits.items()}
    params["source_effort_limits_nm"] = source_efforts
    params["equality_names"] = equality_names
    params["arm_actuator_names"] = arm_actuators
    if fixture_derivation is not None:
        params["fixture_geometry_derivation"] = fixture_derivation
    return model, params


def initialize(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    open_targets = aperture_targets(1.0)
    data.qpos[qpos_id(model, RIGHT_DRIVER)] = open_targets["right_claw_joint_target_rad"]
    data.qpos[qpos_id(model, RIGHT_FOLLOWER)] = open_targets["R_hand_wide1_joint_target_rad"]
    data.qpos[qpos_id(model, LEFT_DRIVER)] = LEFT_DRIVER_HOLD
    data.qpos[qpos_id(model, LEFT_FOLLOWER)] = -LEFT_DRIVER_HOLD
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, full, data.qM)
    return full


def derive_jaw_controller(model: mujoco.MjModel) -> dict[str, Any]:
    data = initialize(model)
    sample_count = 9
    rows: dict[str, dict[str, float]] = {
        RIGHT_DRIVER: {"max_mass_diagonal_kg_m2": 0.0, "max_abs_bias_nm": 0.0},
        RIGHT_FOLLOWER: {"max_mass_diagonal_kg_m2": 0.0, "max_abs_bias_nm": 0.0},
    }
    lever_arms = {RIGHT_DRIVER: 0.0, RIGHT_FOLLOWER: 0.0}
    for aperture in np.linspace(0.0, 1.0, sample_count):
        targets = aperture_targets(float(aperture))
        data.qpos[qpos_id(model, RIGHT_DRIVER)] = targets["right_claw_joint_target_rad"]
        data.qpos[qpos_id(model, RIGHT_FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
        mujoco.mj_forward(model, data)
        full = mass_matrix(model, data)
        narrow_id = body_id(model, "R_hand_narrow3_Link")
        wide_id = body_id(model, "R_hand_wide3_Link")
        narrow = data.xpos[narrow_id].copy()
        wide = data.xpos[wide_id].copy()
        normal = narrow - wide
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        for name in rows:
            dof = dof_id(model, name)
            rows[name]["max_mass_diagonal_kg_m2"] = max(rows[name]["max_mass_diagonal_kg_m2"], float(full[dof, dof]))
            rows[name]["max_abs_bias_nm"] = max(rows[name]["max_abs_bias_nm"], abs(float(data.qfrc_bias[dof])))
            jacp = np.zeros((3, model.nv), dtype=float)
            jacr = np.zeros((3, model.nv), dtype=float)
            body = narrow_id if name == RIGHT_DRIVER else wide_id
            mujoco.mj_jacBody(model, data, jacp, jacr, body)
            lever_arms[name] = max(lever_arms[name], abs(float(np.dot(jacp[:, dof], normal))))

    span = 1.0 - 2.0 * ENDPOINT_MARGIN_RAD
    max_velocity = span * 1.875 / MOTION_DURATION_S
    max_acceleration = span * 5.773502691896258 / (MOTION_DURATION_S**2)
    cap: dict[str, float] = {}
    gains: dict[str, dict[str, float]] = {}
    import sys as _sys
    _sys.path.insert(0, str(STATIC_ROOT))
    import canonical_manipulation_assets as canonical
    mass_bottle = float(canonical.CANONICAL_X2_BOTTLE_MASS_KG)
    design_normal_force = mass_bottle * GRAVITY_M_S2 / (2.0 * CONTACT_FRICTION_ASSUMPTION)
    per_jaw_normal = CONTACT_NORMAL_DESIGN_FACTOR * design_normal_force
    for name, row in rows.items():
        inertia = row["max_mass_diagonal_kg_m2"]
        kp = inertia * JAW_NATURAL_FREQUENCY_RAD_S**2
        kd = 2.0 * DAMPING_RATIO * JAW_NATURAL_FREQUENCY_RAD_S * inertia
        motion_torque = inertia * max_acceleration
        required_free = row["max_abs_bias_nm"] + motion_torque
        contact_torque = per_jaw_normal * lever_arms[name] + row["max_abs_bias_nm"] + motion_torque
        cap[name] = max(2.0 * required_free, 1.15 * contact_torque)
        gains[name] = {"kp_nm_per_rad": kp, "kd_nm_s_per_rad": kd,
                       "compiled_inertia_used_kg_m2": inertia,
                       "max_abs_bias_sampled_nm": row["max_abs_bias_nm"],
                       "lever_arm_projection_m": lever_arms[name],
                       "effort_cap_nm": cap[name],
                       "cap_derivation": "max(2x sampled free-space bias+reference acceleration, 1.15x planned bottle pinch torque)"}
    controller = {
        "label": "SIMULATION_ONLY_M0 paired-jaw servo abstraction",
        "public_command": {"name": "aperture_ratio", "range": [0.0, 1.0], "closed": 0.0, "open": 1.0},
        "endpoint_margin_rad": ENDPOINT_MARGIN_RAD,
        "aperture_mapping": "q_driver=-(margin+(1-2*margin)*aperture_ratio); q_follower=-q_driver",
        "open_targets_rad": aperture_targets(1.0),
        "closed_targets_rad": aperture_targets(0.0),
        "jaw_natural_frequency_rad_s": JAW_NATURAL_FREQUENCY_RAD_S,
        "damping_ratio": DAMPING_RATIO,
        "motion_reference": "minimum jerk",
        "motion_duration_s": MOTION_DURATION_S,
        "predicted_peak_joint_velocity_rad_s": max_velocity,
        "predicted_peak_joint_acceleration_rad_s2": max_acceleration,
        "design_bottle_mass_kg": mass_bottle,
        "design_friction_assumption": CONTACT_FRICTION_ASSUMPTION,
        "design_normal_force_per_jaw_n": per_jaw_normal,
        "per_joint": gains,
        "relation_diagnostic_bound_rad": APERTURE_RELATION_DIAGNOSTIC_RAD,
        "parameters_are": "SIMULATION_ONLY_M0 engineering parameters; not hardware ratings or transmission dynamics",
    }
    return controller


def apply_controller_limits(model: mujoco.MjModel, controller: dict[str, Any], arm: bool = False) -> None:
    for joint_name, record in controller["per_joint"].items():
        act_name = "m0_right_driver_servo" if joint_name == RIGHT_DRIVER else "m0_right_follower_servo"
        aid = actuator_id(model, act_name)
        bound = float(record["effort_cap_nm"])
        model.actuator_ctrlrange[aid] = [-bound, bound]
        model.actuator_forcerange[aid] = [-bound, bound]
    if arm:
        for joint_name, act_name in controller["arm_actuators"].items():
            aid = actuator_id(model, act_name)
            source_effort = float(controller["source_effort_limits_nm"][joint_name])
            model.actuator_ctrlrange[aid] = [-source_effort, source_effort]
            model.actuator_forcerange[aid] = [-source_effort, source_effort]


def jaw_control(model: mujoco.MjModel, data: mujoco.MjData, aperture: float, aperture_v: float,
                aperture_a: float, controller: dict[str, Any]) -> dict[str, Any]:
    targets = aperture_targets(aperture, aperture_v, aperture_a)
    full = mass_matrix(model, data)
    outputs: dict[str, Any] = {}
    for joint_name, target_key, vel_key, acc_key, actuator_name in (
        (RIGHT_DRIVER, "right_claw_joint_target_rad", "right_claw_joint_target_velocity_rad_s",
         "right_claw_joint_target_acceleration_rad_s2", "m0_right_driver_servo"),
        (RIGHT_FOLLOWER, "R_hand_wide1_joint_target_rad", "R_hand_wide1_joint_target_velocity_rad_s",
         "R_hand_wide1_joint_target_acceleration_rad_s2", "m0_right_follower_servo"),
    ):
        dof = dof_id(model, joint_name)
        q = float(data.qpos[qpos_id(model, joint_name)])
        v = float(data.qvel[dof])
        record = controller["per_joint"][joint_name]
        error = targets[target_key] - q
        velocity_error = targets[vel_key] - v
        tau_pd = record["kp_nm_per_rad"] * error + record["kd_nm_s_per_rad"] * velocity_error
        tau_bias = float(data.qfrc_bias[dof])
        tau_ff = float(full[dof, dof]) * targets[acc_key]
        raw = tau_bias + tau_pd + tau_ff
        bound = record["effort_cap_nm"]
        applied = float(np.clip(raw, -bound, bound))
        data.ctrl[actuator_id(model, actuator_name)] = applied
        outputs[joint_name] = {
            "target_q_rad": targets[target_key], "target_qvel_rad_s": targets[vel_key],
            "target_qacc_rad_s2": targets[acc_key], "position_error_rad": error,
            "velocity_error_rad_s": velocity_error, "pd_torque_nm": tau_pd,
            "bias_feedforward_nm": tau_bias, "reference_acceleration_torque_nm": tau_ff,
            "requested_torque_nm": raw, "applied_torque_nm": applied,
            "qfrc_bias_nm": tau_bias, "mass_diagonal_kg_m2": float(full[dof, dof]),
            "qpos_rad": q, "qvel_rad_s": v, "qacc_rad_s2": float(data.qacc[dof]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
        }
    return outputs


def arm_control(model: mujoco.MjModel, data: mujoco.MjData, target: dict[str, tuple[float, float, float]],
                controller: dict[str, Any]) -> dict[str, Any]:
    full = mass_matrix(model, data)
    outputs: dict[str, Any] = {}
    for name in RIGHT_ARM_JOINTS:
        dof = dof_id(model, name)
        q = float(data.qpos[qpos_id(model, name)])
        v = float(data.qvel[dof])
        qref, vref, aref = target[name]
        inertia = float(full[dof, dof])
        kp = inertia * ARM_NATURAL_FREQUENCY_RAD_S**2
        kd = 2.0 * ARM_NATURAL_FREQUENCY_RAD_S * inertia
        pd = kp * (qref - q) + kd * (vref - v)
        bias = float(data.qfrc_bias[dof])
        ff = inertia * aref
        raw = bias + pd + ff
        aid = actuator_id(model, controller["arm_actuators"][name])
        cap = float(controller["source_effort_limits_nm"][name])
        applied = float(np.clip(raw, -cap, cap))
        data.ctrl[aid] = applied
        outputs[name] = {"target_q_rad": qref, "target_qvel_rad_s": vref,
                         "qpos_rad": q, "qvel_rad_s": v, "qacc_rad_s2": float(data.qacc[dof]),
                         "kp_nm_per_rad": kp, "kd_nm_s_per_rad": kd,
                         "bias_feedforward_nm": bias, "pd_torque_nm": pd,
                         "requested_torque_nm": raw, "applied_torque_nm": applied,
                         "source_effort_bound_nm": cap}
    return outputs


def source_limit_rows(model: mujoco.MjModel, limits: dict[str, list[float]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for name, interval in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        q = float(model.qpos0[int(model.jnt_qposadr[jid])])
        # Live values are filled by update_source_state after every step.
        rows[name] = {"range_rad": interval, "qpos_rad": q}
    return rows, violations


def live_source_state(model: mujoco.MjModel, data: mujoco.MjData,
                      limits: dict[str, list[float]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for name, interval in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        q, v, a = float(data.qpos[qadr]), float(data.qvel[dadr]), float(data.qacc[dadr])
        lower, upper = interval
        good = lower - 1e-10 <= q <= upper + 1e-10
        row = {"qpos_rad": q, "qvel_rad_s": v, "qacc_rad_s2": a,
               "source_range_rad": interval, "within_source_position_range": good}
        state[name] = row
        if not good:
            violations.append({"joint": name, **row})
    return state, violations


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        rows.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
            "contact_force_frame_n_and_nm": force.tolist(),
            "normal_force_n": float(abs(force[0])),
        })
    return rows


def is_jaw_contact_body(body_name: str, family: str) -> bool:
    return body_name.startswith(f"R_hand_{family}") and "loop" not in body_name.lower()


def contact_has_object(contact: dict[str, Any], object_name: str) -> bool:
    return object_name in {str(contact.get("body1")), str(contact.get("body2")),
                           str(contact.get("geom1")), str(contact.get("geom2"))}


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    angle = math.acos(float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    if angle < 1e-7:
        return 0.5 * skew
    return angle * skew / (2.0 * math.sin(angle))


def solve_hand_pose(model: mujoco.MjModel, start_data: mujoco.MjData,
                    target_position: np.ndarray, target_rotation: np.ndarray,
                    seed_joints: dict[str, float] | None = None) -> dict[str, float]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    home = initialize(model)
    data.qpos[:] = home.qpos
    for name, value in (seed_joints or {}).items():
        data.qpos[qpos_id(model, name)] = value
    joint_ids = [joint_id(model, name) for name in RIGHT_ARM_JOINTS]
    dofs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    qpos_adrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    ranges = [model.jnt_range[jid].copy() for jid in joint_ids]
    end_body = body_id(model, "R_omnipicker_base_link")
    for _ in range(300):
        mujoco.mj_forward(model, data)
        position = data.xpos[end_body].copy()
        rotation = data.xmat[end_body].reshape(3, 3).copy()
        ep = target_position - position
        er = rotation_error(target_rotation, rotation)
        if np.linalg.norm(ep) < 0.0015 and np.linalg.norm(er) < math.radians(1.0):
            break
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, jacp, jacr, end_body)
        scale = 0.15
        jac = np.vstack((jacp[:, dofs], scale * jacr[:, dofs]))
        error = np.concatenate((ep, scale * er))
        damping = 0.025
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > 0.12:
            delta *= 0.12 / max_delta
        factor = 1.0
        while factor > 1e-4:
            candidate = [float(data.qpos[qadr] + factor * dq) for qadr, dq in zip(qpos_adrs, delta)]
            if all(float(limits[0]) + 0.005 < q < float(limits[1]) - 0.005
                   for q, limits in zip(candidate, ranges)):
                break
            factor *= 0.5
        if factor <= 1e-4:
            break
        for qadr, dq in zip(qpos_adrs, delta):
            data.qpos[qadr] += factor * dq
    mujoco.mj_forward(model, data)
    position = data.xpos[end_body].copy()
    rotation = data.xmat[end_body].reshape(3, 3).copy()
    error_position = float(np.linalg.norm(target_position - position))
    error_orientation = float(np.linalg.norm(rotation_error(target_rotation, rotation)))
    if error_position > 0.005 or error_orientation > math.radians(3.0):
        raise RuntimeError(f"Pre-rollout arm IK failed: pos_error={error_position:.6f} m, "
                           f"orientation_error={math.degrees(error_orientation):.3f} deg")
    return {name: float(data.qpos[qpos_id(model, name)]) for name in RIGHT_ARM_JOINTS}


def model_arm_home(model: mujoco.MjModel) -> dict[str, float]:
    data = initialize(model)
    return {name: float(data.qpos[qpos_id(model, name)]) for name in RIGHT_ARM_JOINTS}


def interpolate_arm(start: dict[str, float], end: dict[str, float], t: float,
                    duration: float) -> dict[str, tuple[float, float, float]]:
    p, v, a = minimum_jerk(t, duration, 0.0, 1.0)
    dp, da = (v, a)
    out = {}
    for name in RIGHT_ARM_JOINTS:
        delta = end[name] - start[name]
        out[name] = (start[name] + delta * p, delta * dp / duration, delta * da / duration**2)
    return out


def configure_arm(model: mujoco.MjModel, controller: dict[str, Any]) -> None:
    controller["arm_actuators"] = {name: f"m0_arm_{name}" for name in RIGHT_ARM_JOINTS}
    for name in RIGHT_ARM_JOINTS:
        aid = actuator_id(model, controller["arm_actuators"][name])
        effort = float(controller["source_effort_limits_nm"][name])
        model.actuator_ctrlrange[aid] = [-effort, effort]
        model.actuator_forcerange[aid] = [-effort, effort]


def camera_for(stage: str) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if stage in {"bottle_hold", "lift"}:
        camera.lookat[:] = [0.30, 0.02, 0.86]
        camera.distance = 1.45
        camera.azimuth = 145
        camera.elevation = -18
    else:
        camera.lookat[:] = [0.25, 0.20, 0.70]
        camera.distance = 1.35
        camera.azimuth = 145
        camera.elevation = -12
    return camera


def save_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
               out: Path, name: str, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    frame = renderer.render().copy()
    Image.fromarray(frame).save(out / name)
    return frame


def stage_plan(stage: str) -> list[tuple[str, float, float, float]]:
    if stage == "open_hold":
        return [("OPEN_HOLD", 2.0, 1.0, 1.0)]
    if stage == "no_contact_motion":
        return [("OPEN_HOLD", 0.25, 1.0, 1.0),
                ("CLOSE", MOTION_DURATION_S, 1.0, 0.0),
                ("CLOSED_HOLD", 0.5, 0.0, 0.0),
                ("REOPEN", MOTION_DURATION_S, 0.0, 1.0),
                ("OPEN_HOLD_FINAL", 0.5, 1.0, 1.0)]
    if stage == "fixed_object":
        return [("OPEN_HOLD", 0.25, 1.0, 1.0),
                ("CLOSE", MOTION_DURATION_S, 1.0, 0.0),
                ("CONTACT_HOLD", 1.0, 0.0, 0.0),
                ("REOPEN", MOTION_DURATION_S, 0.0, 1.0),
                ("OPEN_HOLD_FINAL", 0.5, 1.0, 1.0)]
    if stage == "bottle_hold":
        return [("SETTLE", 0.5, 1.0, 1.0),
                ("PREGRASP", ARM_MOTION_DURATION_S, 1.0, 1.0),
                ("APPROACH", 1.5, 1.0, 1.0),
                ("CLOSE", MOTION_DURATION_S, 1.0, 0.0),
                ("HOLD", 1.0, 0.0, 0.0)]
    raise ValueError(stage)


def phase_aperture(phase: str, phase_time: float, start: float, end: float, duration: float) -> tuple[float, float, float]:
    if phase in {"CLOSE", "REOPEN"}:
        return minimum_jerk(phase_time, duration, start, end)
    return start, 0.0, 0.0


def collect_failure_diagnosis(model: mujoco.MjModel, data: mujoco.MjData,
                              controller: dict[str, Any], control_terms: dict[str, Any],
                              gate: str) -> dict[str, Any]:
    full = mass_matrix(model, data)
    qstate: dict[str, Any] = {}
    for name in (RIGHT_DRIVER, RIGHT_FOLLOWER):
        dof = dof_id(model, name)
        qstate[name] = {
            "qpos_rad": float(data.qpos[qpos_id(model, name)]),
            "qvel_rad_s": float(data.qvel[dof]),
            "qacc_rad_s2": float(data.qacc[dof]),
            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
            "mass_diagonal_kg_m2": float(full[dof, dof]),
            "control_terms": control_terms.get(name),
            "gain_and_bound": controller["per_joint"][name],
        }
    return {"diagnosis_type": "single bounded no-step state decomposition",
            "gate": gate, "time_s": float(data.time), "qpos_write_during_diagnosis": False,
            "jaw_states": qstate, "contacts": contact_rows(model, data)}


def apply_arm_targets(stage: str, phase: str, phase_time: float, arm_poses: dict[str, dict[str, float]],
                      home: dict[str, float]) -> dict[str, tuple[float, float, float]] | None:
    if stage != "bottle_hold":
        return None
    if phase == "SETTLE":
        return {name: (home[name], 0.0, 0.0) for name in RIGHT_ARM_JOINTS}
    if phase == "PREGRASP":
        return interpolate_arm(home, arm_poses["pregrasp"], phase_time, ARM_MOTION_DURATION_S)
    if phase == "APPROACH":
        return interpolate_arm(arm_poses["pregrasp"], arm_poses["grasp"], phase_time, 1.5)
    return {name: (arm_poses["grasp"][name], 0.0, 0.0) for name in RIGHT_ARM_JOINTS}


def run_stage(stage: str, evidence: Path, controller: dict[str, Any], source: dict[str, Any],
              arm_poses: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    model, build_info = build_model(stage, controller)
    apply_controller_limits(model, controller, arm=stage in {"bottle_hold", "lift"})
    data = initialize(model)
    if stage in {"bottle_hold", "lift"}:
        configure_arm(model, controller)
    arm_home = model_arm_home(model) if stage == "bottle_hold" else {}
    init_contacts = contact_rows(model, data)
    initial_penetrations = [row for row in init_contacts if row["penetration_m"] > 1e-8]
    renderer: mujoco.Renderer | None = None
    camera = camera_for(stage)
    frame_samples: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height=360, width=480)
    trace_path = evidence / f"{stage}_trace.jsonl"
    limits = build_info["source_position_limits_rad"]
    time_by_phase: dict[str, float] = {}
    phase_frames: dict[str, int] = {}
    object_contact_frames_by_phase: dict[str, int] = {}
    bilateral_contact_frames_by_phase: dict[str, int] = {}
    contact_pair_frames: dict[str, int] = {}
    object_contact_frames = 0
    object_contact_seen = False
    expected_object = "m0_fixed_contact_cylinder" if stage == "fixed_object" else "m0_bottle"
    jaw_contact_families: set[str] = set()
    max_penetration = 0.0
    max_force = 0.0
    max_qvel = 0.0
    max_qacc = 0.0
    max_relation = 0.0
    max_precontact_relation = 0.0
    max_contact_relation = 0.0
    contact_hold_samples: list[dict[str, Any]] = []
    max_effort: dict[str, float] = {RIGHT_DRIVER: 0.0, RIGHT_FOLLOWER: 0.0}
    saturation_steps = {RIGHT_DRIVER: 0, RIGHT_FOLLOWER: 0}
    hard_limit_violations: list[dict[str, Any]] = []
    nonfinite = False
    first_gate_failure: dict[str, Any] | None = None
    row_count = 0
    phase_begin = 0.0
    last_control: dict[str, Any] = {}
    initial_bottle_z: float | None = None
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    if bottle_body_id >= 0:
        initial_bottle_z = float(data.xpos[bottle_body_id][2])
    fixture_preflight = None
    fixture_closure = None
    fixture_initial_contacts = []
    if stage == "fixed_object":
        fixture_preflight = fixture_geometry_preflight(model, data)
        fixture_closure = fixture_closure_geometry_check(model)
        fixture_initial_contacts = [row for row in init_contacts
                                    if contact_has_object(row, "m0_fixed_contact_cylinder")]
        if fixture_preflight["status"] != "PASS":
            first_gate_failure = {"gate": "zero-step fixture geometry preflight",
                                  "preflight": fixture_preflight}
        elif fixture_initial_contacts:
            first_gate_failure = {"gate": "zero-step fixture initial contact",
                                  "contacts": fixture_initial_contacts}
        elif fixture_closure["status"] != "PASS":
            first_gate_failure = {"gate": "zero-step jaw closure geometry",
                                  "closure_geometry": fixture_closure}
        else:
            fixture_pose = build_info["fixture_geometry_derivation"]["cylinder"]["center_world_m"]
            camera.lookat[:] = fixture_pose
            camera.distance = 0.34
            camera.azimuth = 145
            camera.elevation = -18
    elif initial_penetrations:
        first_gate_failure = {"gate": "initial penetration", "contacts": initial_penetrations}

    with trace_path.open("w", encoding="utf-8") as stream:
        if stage == "fixed_object" and fixture_preflight and fixture_preflight["status"] == "PASS":
            save_frame(renderer, model, data, evidence, "fixed_object_open_fixture.png", camera)
        if stage in {"bottle_hold", "lift"}:
            save_frame(renderer, model, data, evidence, "bottle_pregrasp.png", camera)
        step = 0
        for phase, duration, start, end in stage_plan(stage):
            if first_gate_failure:
                break
            phase_steps = max(1, round(duration / TIMESTEP))
            phase_begin = float(data.time)
            time_by_phase[phase] = phase_begin
            for local_step in range(phase_steps):
                phase_time = local_step * TIMESTEP
                aperture, aperture_v, aperture_a = phase_aperture(phase, phase_time, start, end, duration)
                mujoco.mj_forward(model, data)
                last_control = jaw_control(model, data, aperture, aperture_v, aperture_a, controller)
                arm_targets = apply_arm_targets(stage, phase, phase_time, arm_poses or {}, arm_home)
                arm_terms = arm_control(model, data, arm_targets, controller) if arm_targets is not None else {}
                mujoco.mj_step(model, data)
                step += 1
                row_count += 1
                contacts = contact_rows(model, data)
                phase_frames[phase] = phase_frames.get(phase, 0) + 1
                source_state, violations = live_source_state(model, data, limits)
                hard_limit_violations.extend({"step": step, "time_s": float(data.time), **x} for x in violations)
                qd = float(data.qpos[qpos_id(model, RIGHT_DRIVER)])
                qf = float(data.qpos[qpos_id(model, RIGHT_FOLLOWER)])
                relation = qd + qf
                max_relation = max(max_relation, abs(relation))
                max_qvel = max(max_qvel, abs(float(data.qvel[dof_id(model, RIGHT_DRIVER)])),
                               abs(float(data.qvel[dof_id(model, RIGHT_FOLLOWER)])))
                max_qacc = max(max_qacc, abs(float(data.qacc[dof_id(model, RIGHT_DRIVER)])),
                               abs(float(data.qacc[dof_id(model, RIGHT_FOLLOWER)])))
                for name, record in last_control.items():
                    max_effort[name] = max(max_effort[name], abs(float(record["applied_torque_nm"])))
                    if abs(float(record["requested_torque_nm"])) > controller["per_joint"][name]["effort_cap_nm"] * 0.999:
                        saturation_steps[name] += 1
                object_contacts = []
                frame_jaw_families: set[str] = set()
                for contact in contacts:
                    pair = "|".join(sorted(str(v) for v in (contact["body1"], contact["body2"])))
                    contact_pair_frames[pair] = contact_pair_frames.get(pair, 0) + 1
                    max_penetration = max(max_penetration, float(contact["penetration_m"]))
                    max_force = max(max_force, float(contact["normal_force_n"]))
                    bodies = {str(contact["body1"]), str(contact["body2"])}
                    if contact_has_object(contact, expected_object):
                        object_contacts.append(contact)
                        force_bearing = float(contact["normal_force_n"]) > CONTACT_FORCE_BEARING_MIN_N
                        if force_bearing and any(is_jaw_contact_body(body, "narrow") for body in bodies):
                            jaw_contact_families.add("narrow")
                            frame_jaw_families.add("narrow")
                        if force_bearing and any(is_jaw_contact_body(body, "wide") for body in bodies):
                            jaw_contact_families.add("wide")
                            frame_jaw_families.add("wide")
                if object_contacts:
                    object_contact_seen = True
                    object_contact_frames += 1
                    object_contact_frames_by_phase[phase] = object_contact_frames_by_phase.get(phase, 0) + 1
                    max_contact_relation = max(max_contact_relation, abs(relation))
                elif not object_contact_seen:
                    max_precontact_relation = max(max_precontact_relation, abs(relation))
                if {"narrow", "wide"}.issubset(frame_jaw_families):
                    bilateral_contact_frames_by_phase[phase] = bilateral_contact_frames_by_phase.get(phase, 0) + 1
                if stage == "fixed_object" and phase == "CONTACT_HOLD":
                    contact_hold_samples.append({
                        "step": step,
                        "driver_qvel_rad_s": float(data.qvel[dof_id(model, RIGHT_DRIVER)]),
                        "follower_qvel_rad_s": float(data.qvel[dof_id(model, RIGHT_FOLLOWER)]),
                        "driver_qacc_rad_s2": float(data.qacc[dof_id(model, RIGHT_DRIVER)]),
                        "follower_qacc_rad_s2": float(data.qacc[dof_id(model, RIGHT_FOLLOWER)]),
                        "relation_error_rad": relation,
                        "penetration_m": max((float(c["penetration_m"]) for c in object_contacts), default=0.0),
                        "bilateral": {"narrow", "wide"}.issubset(frame_jaw_families),
                    })
                finite_values = np.concatenate((data.qpos, data.qvel, data.qacc))
                nonfinite = nonfinite or not bool(np.isfinite(finite_values).all())
                bottle_z = float(data.xpos[bottle_body_id][2]) if bottle_body_id >= 0 else None
                row = {
                    "step": step, "time_s": float(data.time), "phase": phase,
                    "aperture_command": aperture_targets(aperture, aperture_v, aperture_a),
                    "driver_follower_relation_error_rad": relation,
                    "right_jaw_control": last_control,
                    "arm_control": arm_terms,
                    "source_joint_state": source_state,
                    "contacts": contacts,
                    "object_contact_count": len(object_contacts),
                    "bottle_body_z_m": bottle_z,
                    "bottle_lift_from_initial_m": None if bottle_z is None or initial_bottle_z is None else bottle_z - initial_bottle_z,
                    "qpos_writes_during_active_rollout": 0,
                    "bottle_qpos_writes_during_active_rollout": 0,
                    "follower_qpos_writes_during_active_rollout": 0,
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

                if renderer is not None and ((stage in {"bottle_hold", "lift"} and
                                               (step % 40 == 0 or local_step == 0)) or
                                              (stage not in {"bottle_hold", "lift"} and local_step == 0)):
                    frame = save_frame(renderer, model, data, evidence,
                                       f"{stage}_{phase.lower()}_{step:06d}.png", camera)
                    if stage in {"bottle_hold", "lift"}:
                        frame_samples.append(frame)

                if hard_limit_violations:
                    first_gate_failure = {"gate": "source position limit", **hard_limit_violations[0]}
                elif not np.isfinite(finite_values).all():
                    first_gate_failure = {"gate": "finite state", "step": step}
                elif max_relation > APERTURE_RELATION_DIAGNOSTIC_RAD and (
                        stage != "fixed_object" or not object_contact_seen):
                    first_gate_failure = {"gate": "shared aperture relation tracking", "step": step,
                                          "observed_rad": max_relation,
                                          "bound_rad": APERTURE_RELATION_DIAGNOSTIC_RAD,
                                          "scope": "free motion before first fixed-object contact" if stage == "fixed_object" else "no-contact motion"}
                if stage in {"open_hold", "no_contact_motion"} and contacts:
                    first_gate_failure = {"gate": "no-contact dynamics", "step": step, "contacts": contacts}
                if stage in {"open_hold", "no_contact_motion"}:
                    qvel_bound = 1.5 * controller["predicted_peak_joint_velocity_rad_s"] + 0.05
                    if max_qvel > qvel_bound:
                        first_gate_failure = {"gate": "jaw velocity bound", "step": step,
                                              "observed_rad_s": max_qvel, "bound_rad_s": qvel_bound}
                if stage == "fixed_object" and max_penetration > 0.006:
                    first_gate_failure = {"gate": "fixed-object penetration bound", "step": step,
                                          "observed_m": max_penetration, "bound_m": 0.006}
                if stage == "fixed_object":
                    qvel_bound = 1.5 * controller["predicted_peak_joint_velocity_rad_s"] + 0.05
                    if max_qvel > qvel_bound:
                        first_gate_failure = {"gate": "fixed-object jaw velocity bound", "step": step,
                                              "observed_rad_s": max_qvel, "bound_rad_s": qvel_bound}
                    elif max_qacc > CONTACT_ACCELERATION_BOUND_RAD_S2:
                        first_gate_failure = {"gate": "fixed-object jaw acceleration bound", "step": step,
                                              "observed_rad_s2": max_qacc,
                                              "bound_rad_s2": CONTACT_ACCELERATION_BOUND_RAD_S2}
                if stage in {"bottle_hold", "lift"} and max_penetration > 0.008:
                    first_gate_failure = {"gate": "bottle/contact penetration bound", "step": step,
                                          "observed_m": max_penetration, "bound_m": 0.008}
                if first_gate_failure:
                    break
            if first_gate_failure:
                break

    if renderer is not None:
        renderer.close()
    video_path = None
    if stage in {"bottle_hold", "lift"} and frame_samples:
        video_path = evidence / f"{stage}.mp4"
        try:
            imageio.mimsave(video_path, frame_samples, fps=20, macro_block_size=1)
        except Exception as exc:
            video_path = None
            build_info["video_error"] = str(exc)

    if stage == "open_hold" and first_gate_failure is None:
        all_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        tail_rows = all_rows[-100:]
        tail_vel = max((abs(row["source_joint_state"][name]["qvel_rad_s"])
                        for row in tail_rows for name in (RIGHT_DRIVER, RIGHT_FOLLOWER)), default=0.0)
        tail_err = max((abs(row["driver_follower_relation_error_rad"]) for row in tail_rows), default=0.0)
        acceleration_peaks = []
        for start in range(0, len(all_rows), 100):
            chunk = all_rows[start:start + 100]
            acceleration_peaks.append(max((abs(row["source_joint_state"][name]["qacc_rad_s2"])
                                           for row in chunk for name in (RIGHT_DRIVER, RIGHT_FOLLOWER)), default=0.0))
        if tail_vel > 0.02:
            first_gate_failure = {"gate": "OPEN hold settling", "tail_max_abs_qvel_rad_s": tail_vel,
                                  "bound_rad_s": 0.02}
        elif max_relation > APERTURE_RELATION_DIAGNOSTIC_RAD:
            first_gate_failure = {"gate": "OPEN hold relation", "max_error_rad": max_relation}
        elif len(acceleration_peaks) > 3 and acceleration_peaks[-1] > 2.0 * max(acceleration_peaks[-4:-1], default=0.0) + 0.5:
            first_gate_failure = {"gate": "OPEN hold acceleration growth",
                                  "recent_window_peak_rad_s2": acceleration_peaks[-1],
                                  "preceding_window_peaks_rad_s2": acceleration_peaks[-4:-1]}
        build_info["tail_max_abs_qvel_rad_s"] = tail_vel
        build_info["tail_relation_error_rad"] = tail_err
        build_info["acceleration_window_peaks_rad_s2"] = acceleration_peaks

    if stage == "no_contact_motion" and first_gate_failure is None:
        tail_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()[-100:]]
        tail_vel = max((abs(row["source_joint_state"][name]["qvel_rad_s"])
                        for row in tail_rows for name in (RIGHT_DRIVER, RIGHT_FOLLOWER)), default=0.0)
        if tail_vel > 0.02:
            first_gate_failure = {"gate": "reopen final settling", "tail_max_abs_qvel_rad_s": tail_vel,
                                  "bound_rad_s": 0.02}

    contact_loss = None
    contact_hold_windows: list[dict[str, Any]] = []
    contact_hold_growing_oscillation = False
    if stage == "fixed_object" and contact_hold_samples:
        for start in range(0, len(contact_hold_samples), 100):
            window = contact_hold_samples[start:start + 100]
            if not window:
                continue
            contact_hold_windows.append({
                "step_start": window[0]["step"], "step_end": window[-1]["step"],
                "peak_abs_qvel_rad_s": max(max(abs(x["driver_qvel_rad_s"]), abs(x["follower_qvel_rad_s"])) for x in window),
                "peak_abs_qacc_rad_s2": max(max(abs(x["driver_qacc_rad_s2"]), abs(x["follower_qacc_rad_s2"])) for x in window),
                "peak_abs_relation_error_rad": max(abs(x["relation_error_rad"]) for x in window),
                "peak_penetration_m": max(x["penetration_m"] for x in window),
                "bilateral_contact_fraction": sum(bool(x["bilateral"]) for x in window) / len(window),
            })
        if len(contact_hold_windows) >= 3:
            tail = contact_hold_windows[-3:]
            contact_hold_growing_oscillation = any(
                all(tail[i + 1][key] > tail[i][key] for i in range(2))
                for key in ("peak_abs_qvel_rad_s", "peak_abs_qacc_rad_s2",
                            "peak_abs_relation_error_rad", "peak_penetration_m")
            )
    if stage == "fixed_object" and first_gate_failure is None:
        final = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        contact_loss = not any(contact_has_object(c, expected_object) for c in final["contacts"])
        if not {"narrow", "wide"}.issubset(jaw_contact_families):
            first_gate_failure = {"gate": "opposing jaw contact", "families": sorted(jaw_contact_families)}
        elif bilateral_contact_frames_by_phase.get("CONTACT_HOLD", 0) < phase_frames.get("CONTACT_HOLD", 1):
            first_gate_failure = {"gate": "fixed-object bilateral contact persistence",
                                  "fraction": bilateral_contact_frames_by_phase.get("CONTACT_HOLD", 0) /
                                             max(phase_frames.get("CONTACT_HOLD", 1), 1),
                                  "required": 1.0}
        elif contact_hold_growing_oscillation:
            first_gate_failure = {"gate": "fixed-object contact-hold growing oscillation",
                                  "windows": contact_hold_windows[-3:]}
        elif not contact_loss:
            first_gate_failure = {"gate": "reopen contact loss", "final_contacts": final["contacts"]}

    if stage == "bottle_hold" and first_gate_failure is None:
        contact_fraction = object_contact_frames_by_phase.get("HOLD", 0) / max(phase_frames.get("HOLD", 1), 1)
        if not {"narrow", "wide"}.issubset(jaw_contact_families):
            first_gate_failure = {"gate": "bottle opposing jaw contact", "families": sorted(jaw_contact_families)}
        elif contact_fraction < 0.80:
            first_gate_failure = {"gate": "bottle hold contact persistence", "fraction": contact_fraction,
                                  "required": 0.80}

    diagnosis = None
    if first_gate_failure is not None:
        mujoco.mj_forward(model, data)
        diagnosis = collect_failure_diagnosis(model, data, controller, last_control,
                                              str(first_gate_failure.get("gate", "unknown")))
        (evidence / f"{stage}_one_step_diagnosis.json").write_text(
            json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")

    result = {
        "stage": stage,
        "status": "PASS" if first_gate_failure is None else "FAIL",
        "first_failed_gate": first_gate_failure,
        "steps": row_count,
        "sim_time_s": float(data.time),
        "phase_start_time_s": time_by_phase,
        "source_position_violations": hard_limit_violations,
        "finite_state": not nonfinite,
        "max_abs_jaw_qvel_rad_s": max_qvel,
        "max_abs_jaw_qacc_rad_s2": max_qacc,
        "max_shared_aperture_relation_error_rad": max_relation,
        "shared_aperture_relation_diagnostic_bound_rad": APERTURE_RELATION_DIAGNOSTIC_RAD,
        "shared_aperture_relation_diagnostic_scope": "hard synchronization gate before first object contact; diagnostic-only after contact, when bilateral physical-contact stability gates apply",
        "contact_force_bearing_minimum_n": CONTACT_FORCE_BEARING_MIN_N,
        "contact_acceleration_bound_rad_s2": CONTACT_ACCELERATION_BOUND_RAD_S2,
        "max_precontact_shared_aperture_relation_error_rad": max_precontact_relation,
        "max_contact_shared_aperture_relation_error_rad": max_contact_relation,
        "max_jaw_effort_nm": max_effort,
        "effort_caps_nm": {n: controller["per_joint"][n]["effort_cap_nm"] for n in controller["per_joint"]},
        "jaw_actuator_saturation_steps": saturation_steps,
        "initial_contacts": init_contacts,
        "fixture_geometry_derivation": build_info.get("fixture_geometry_derivation"),
        "fixture_open_preflight": fixture_preflight,
        "fixture_closure_geometry": fixture_closure,
        "fixture_initial_contacts": fixture_initial_contacts,
        "contact_pair_frame_counts": contact_pair_frames,
        "jaw_contact_families": sorted(jaw_contact_families),
        "object_contact_frames_by_phase": object_contact_frames_by_phase,
        "bilateral_contact_frames_by_phase": bilateral_contact_frames_by_phase,
        "contact_hold_stability_windows": contact_hold_windows,
        "contact_hold_growing_oscillation": contact_hold_growing_oscillation,
        "phase_frame_counts": phase_frames,
        "object_contact_fraction": object_contact_frames / max(row_count, 1),
        "hold_contact_fraction": object_contact_frames_by_phase.get("HOLD", 0) /
                                 max(phase_frames.get("HOLD", 1), 1),
        "max_penetration_m": max_penetration,
        "max_contact_normal_force_n": max_force,
        "fixed_object_contact_lost_after_reopen": contact_loss,
        "bottle_initial_body_z_m": initial_bottle_z,
        "bottle_final_body_z_m": float(data.xpos[bottle_body_id][2]) if bottle_body_id >= 0 else None,
        "bottle_lift_m": (float(data.xpos[bottle_body_id][2]) - initial_bottle_z)
                         if bottle_body_id >= 0 and initial_bottle_z is not None else None,
        "qpos_writes_during_active_rollout": 0,
        "bottle_qpos_writes_during_active_rollout": 0,
        "follower_qpos_writes_during_active_rollout": 0,
        "right_jaw_equalities": [n for n in build_info["equality_names"] if RIGHT_DRIVER in n or RIGHT_FOLLOWER in n],
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq,
                  "timestep_s": float(model.opt.timestep),
                  "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
                  "iterations": int(model.opt.iterations)},
        "trace_path": str(trace_path),
        "video_path": None if video_path is None else str(video_path),
        "one_step_diagnosis_path": None if diagnosis is None else str(evidence / f"{stage}_one_step_diagnosis.json"),
    }
    (evidence / f"{stage}_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def prepare_arm_poses(model: mujoco.MjModel, controller: dict[str, Any]) -> dict[str, dict[str, float]]:
    home = model_arm_home(model)
    initial = initialize(model)
    end_body = body_id(model, "R_omnipicker_base_link")
    mujoco.mj_forward(model, initial)
    current_base = initial.xpos[end_body].copy()
    narrow = initial.xpos[body_id(model, "R_hand_narrow3_Link")].copy()
    wide = initial.xpos[body_id(model, "R_hand_wide3_Link")].copy()
    current_center = 0.5 * (narrow + wide)
    offset = current_base - current_center
    bottle_center = np.array([0.300, 0.000, 0.8775], dtype=float)
    grasp_base_target = bottle_center + offset
    pregrasp_base_target = grasp_base_target + np.array([0.0, 0.10, 0.0])
    target_rotation = initial.xmat[end_body].reshape(3, 3).copy()
    pre = solve_hand_pose(model, initial, pregrasp_base_target, target_rotation)
    grasp = solve_hand_pose(model, initial, grasp_base_target, target_rotation)
    return {"home": home, "pregrasp": pre, "grasp": grasp}


def run_lift(evidence: Path, controller: dict[str, Any], source: dict[str, Any],
             arm_poses: dict[str, dict[str, float]]) -> dict[str, Any]:
    model, build_info = build_model("lift", controller)
    apply_controller_limits(model, controller, arm=True)
    configure_arm(model, controller)
    data = initialize(model)
    bottle_id = body_id(model, "m0_bottle")
    initial_bottle_z = float(data.xpos[bottle_id][2])
    home = arm_poses["grasp"]
    target_qs: dict[float, dict[str, float]] = {}
    end_body = body_id(model, "R_omnipicker_base_link")
    ik_data = initialize(model)
    for name in RIGHT_ARM_JOINTS:
        ik_data.qpos[qpos_id(model, name)] = home[name]
    mujoco.mj_forward(model, ik_data)
    base_target = ik_data.xpos[end_body].copy()
    target_rotation = ik_data.xmat[end_body].reshape(3, 3).copy()
    for height in (0.001, 0.005, 0.030):
        target_qs[height] = solve_hand_pose(model, ik_data, base_target + np.array([0.0, 0.0, height]),
                                            target_rotation, seed_joints=home)

    trace_path = evidence / "lift_trace.jsonl"
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = camera_for("lift")
    frames: list[np.ndarray] = []
    step = 0
    max_lift = 0.0
    max_pen = 0.0
    max_relation = 0.0
    family: set[str] = set()
    object_frames = 0
    first_failure = None
    checkpoints: dict[str, Any] = {}
    with trace_path.open("w", encoding="utf-8") as stream:
        # Repeat the accepted hold path from a clean reset before the lift path.
        prefix = [("SETTLE", 0.5, 1.0, 1.0), ("PREGRASP", ARM_MOTION_DURATION_S, 1.0, 1.0),
                  ("APPROACH", 1.5, 1.0, 1.0), ("CLOSE", MOTION_DURATION_S, 1.0, 0.0),
                  ("HOLD", 1.0, 0.0, 0.0)]
        home_start = model_arm_home(model)
        for phase, duration, aperture_start, aperture_end in prefix:
            for local_step in range(round(duration / TIMESTEP)):
                t = local_step * TIMESTEP
                if phase == "SETTLE":
                    arm_targets = {name: (home_start[name], 0.0, 0.0) for name in RIGHT_ARM_JOINTS}
                elif phase == "PREGRASP":
                    arm_targets = interpolate_arm(home_start, arm_poses["pregrasp"], t, duration)
                elif phase == "APPROACH":
                    arm_targets = interpolate_arm(arm_poses["pregrasp"], arm_poses["grasp"], t, duration)
                else:
                    arm_targets = {name: (home[name], 0.0, 0.0) for name in RIGHT_ARM_JOINTS}
                aperture, av, aa = phase_aperture(phase, t, aperture_start, aperture_end, duration)
                mujoco.mj_forward(model, data)
                jaw_terms = jaw_control(model, data, aperture, av, aa, controller)
                arm_terms = arm_control(model, data, arm_targets, controller)
                mujoco.mj_step(model, data)
                step += 1
                if step % 50 == 0:
                    frames.append(save_frame(renderer, model, data, evidence,
                                             f"lift_prefix_{step:06d}.png", camera))
        for height, target_arm in target_qs.items():
            segment_start = {name: float(data.qpos[qpos_id(model, name)]) for name in RIGHT_ARM_JOINTS}
            duration = 0.5 if height < 0.01 else 1.0
            phase = f"LIFT_{int(height * 1000)}MM"
            for local_step in range(round(duration / TIMESTEP)):
                t = local_step * TIMESTEP
                aperture, av, aa = 0.0, 0.0, 0.0
                arm_targets = interpolate_arm(segment_start, target_arm, t, duration)
                mujoco.mj_forward(model, data)
                jaw_terms = jaw_control(model, data, aperture, av, aa, controller)
                arm_terms = arm_control(model, data, arm_targets, controller)
                mujoco.mj_step(model, data)
                step += 1
                contacts = contact_rows(model, data)
                bottle_z = float(data.xpos[bottle_id][2])
                lift = bottle_z - initial_bottle_z
                max_lift = max(max_lift, lift)
                max_relation = max(max_relation, abs(float(data.qpos[qpos_id(model, RIGHT_DRIVER)] +
                                                              data.qpos[qpos_id(model, RIGHT_FOLLOWER)])))
                object_contacts = []
                for contact in contacts:
                    if contact_has_object(contact, "m0_bottle"):
                        object_contacts.append(contact)
                        bodies = {str(contact["body1"]), str(contact["body2"])}
                        if any("R_hand_narrow" in x for x in bodies):
                            family.add("narrow")
                        if any("R_hand_wide" in x for x in bodies):
                            family.add("wide")
                    max_pen = max(max_pen, float(contact["penetration_m"]))
                object_frames += bool(object_contacts)
                state, violations = live_source_state(model, data, build_info["source_position_limits_rad"])
                finite = bool(np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc))).all())
                row = {"step": step, "time_s": float(data.time), "phase": phase,
                       "target_lift_m": height, "bottle_body_z_m": bottle_z,
                       "bottle_lift_from_initial_m": lift,
                       "driver_follower_relation_error_rad": float(data.qpos[qpos_id(model, RIGHT_DRIVER)] +
                                                                      data.qpos[qpos_id(model, RIGHT_FOLLOWER)]),
                       "jaw_control": jaw_terms, "arm_control": arm_terms,
                       "source_joint_state": state, "source_limit_violations": violations,
                       "contacts": contacts, "object_contact_count": len(object_contacts),
                       "qpos_writes_during_active_rollout": 0,
                       "bottle_qpos_writes_during_active_rollout": 0,
                       "follower_qpos_writes_during_active_rollout": 0}
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if step % 40 == 0 or local_step == 0:
                    frames.append(save_frame(renderer, model, data, evidence,
                                             f"lift_{int(height * 1000)}mm_{step:06d}.png", camera))
                if violations:
                    first_failure = {"gate": "source position limit", "step": step, "violations": violations}
                elif not finite:
                    first_failure = {"gate": "finite state", "step": step}
                elif max_relation > APERTURE_RELATION_DIAGNOSTIC_RAD:
                    first_failure = {"gate": "paired-jaw tracking", "step": step, "max_error_rad": max_relation}
                elif max_pen > 0.008:
                    first_failure = {"gate": "bottle contact penetration", "step": step, "max_penetration_m": max_pen}
                if first_failure:
                    break
            if first_failure:
                break
            achieved = max_lift
            checkpoints[f"{int(height * 1000)}mm"] = {"requested_lift_m": height,
                                                       "observed_max_lift_m": achieved,
                                                       "airborne": achieved >= max(height * 0.8, 0.0005),
                                                       "right_jaw_families": sorted(family)}
            if not {"narrow", "wide"}.issubset(family):
                first_failure = {"gate": "bottle opposing jaw contact during lift", "families": sorted(family)}
                break
            if achieved < max(height * 0.8, 0.0005):
                first_failure = {"gate": f"{int(height * 1000)} mm lift", "observed_lift_m": achieved,
                                 "required_m": max(height * 0.8, 0.0005)}
                break
    renderer.close()
    video_path = evidence / "lift.mp4"
    if frames:
        try:
            imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
        except Exception:
            video_path = None
    result = {
        "stage": "lift", "status": "PASS" if first_failure is None else "FAIL",
        "first_failed_gate": first_failure, "steps": step, "sim_time_s": float(data.time),
        "initial_bottle_body_z_m": initial_bottle_z,
        "final_bottle_body_z_m": float(data.xpos[bottle_id][2]),
        "maximum_bottle_lift_m": max_lift,
        "checkpoints": checkpoints, "jaw_contact_families": sorted(family),
        "object_contact_frame_fraction": object_frames / max(step, 1),
        "max_shared_aperture_relation_error_rad": max_relation,
        "max_penetration_m": max_pen,
        "qpos_writes_during_active_rollout": 0,
        "bottle_qpos_writes_during_active_rollout": 0,
        "follower_qpos_writes_during_active_rollout": 0,
        "trace_path": str(trace_path), "video_path": None if video_path is None else str(video_path),
    }
    (evidence / "lift_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def runtime_identity() -> dict[str, Any]:
    package_dir = Path(mujoco.__file__).resolve().parent
    libraries = sorted(package_dir.glob("libmujoco.so*"))
    if not libraries:
        libraries = sorted(package_dir.glob("**/libmujoco.so*"))
    script_path = Path(__file__).resolve()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "mujoco_module_path": str(Path(mujoco.__file__).resolve()),
        "native_libraries": [{"path": str(path), "sha256": sha256(path)} for path in libraries],
        "experiment_script": str(script_path),
        "experiment_script_sha256": sha256(script_path),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "numpy": np.__version__,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--stage-only", choices=("open_hold", "no_contact_motion", "fixed_object", "bottle_hold"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="compile the fixed-object scene and run zero-step geometry/contact checks only")
    args = parser.parse_args()
    if args.preflight_only and args.stage_only not in (None, "fixed_object"):
        parser.error("--preflight-only can only be combined with --stage-only fixed_object")
    evidence = args.output
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "experiment_commands.txt").write_text(
        "wsl.exe -d Ubuntu-22.04 -- bash -lc 'source /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/activate && python3 "
        f"{Path(__file__)} --output {evidence}"
        f"{' --stage-only ' + args.stage_only if args.stage_only else ''}"
        f"{' --preflight-only' if args.preflight_only else ''}'\n", encoding="utf-8")
    source = verify_inputs(evidence)
    identity = runtime_identity()
    (evidence / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    write_json(evidence / "source_audit.json", source)
    if args.preflight_only:
        model, build = build_model("fixed_object")
        data = initialize(model)
        preflight = fixture_geometry_preflight(model, data)
        result = {"status": preflight["status"], "physics_steps": 0,
                  "fixture_geometry_derivation": build["fixture_geometry_derivation"],
                  "fixture_open_preflight": preflight,
                  "fixture_closure_geometry": fixture_closure_geometry_check(model),
                  "source": source, "runtime": identity,
                  "controller_parameters_changed": False}
        result["status"] = "PASS" if result["fixture_open_preflight"]["status"] == "PASS" and result["fixture_closure_geometry"]["status"] == "PASS" else "FAIL"
        write_json(evidence / "fixture_preflight.json", result)
        (evidence / "REPORT.md").write_text(
            "# Issue #46 fixed-object zero-step preflight\n\n"
            f"Status: **{result['status']}**\n\n"
            "No `mj_step` calls were made. The compiled fixture and robot were checked from a clean reset with `mj_forward` only.\n",
            encoding="utf-8")
        return 0 if preflight["status"] == "PASS" else 2
    if not np.allclose(aperture_targets(1.0)["R_hand_wide1_joint_target_rad"],
                       -aperture_targets(1.0)["right_claw_joint_target_rad"]):
        raise RuntimeError("Shared aperture mapping does not preserve the source ratio")

    model, build = build_model("open_hold")
    controller = derive_jaw_controller(model)
    controller["source_effort_limits_nm"] = build["source_effort_limits_nm"]
    controller["left_picker_state"] = {LEFT_DRIVER: LEFT_DRIVER_HOLD,
                                        LEFT_FOLLOWER: -LEFT_DRIVER_HOLD,
                                        "contact_allowed": False, "runtime_qpos_writes": 0}
    controller["no_right_mimic_equality"] = True
    apply_controller_limits(model, controller)
    write_json(evidence / "controller_derivation.json", controller)
    stages: list[dict[str, Any]] = []
    stage_sequence = [args.stage_only] if args.stage_only else ["open_hold", "no_contact_motion", "fixed_object", "bottle_hold"]
    for stage in stage_sequence:
        if args.stage_only:
            stage_model, stage_build = build_model(stage, controller)
            controller["source_effort_limits_nm"] = stage_build["source_effort_limits_nm"]
            controller["arm_actuators"] = stage_build["arm_actuator_names"]
            apply_controller_limits(stage_model, controller, arm=stage in {"bottle_hold", "lift"})
            arm_poses = prepare_arm_poses(stage_model, controller) if stage == "bottle_hold" else None
            if arm_poses is not None:
                write_json(evidence / "arm_approach_ik.json", arm_poses)
            result = run_stage(stage, evidence, controller, source, arm_poses)
            stages.append(result)
            write_json(evidence / "result.json", {"status": result["status"], "last_completed_stage": stage,
                                                   "stages": stages, "source": source,
                                                   "runtime": identity, "controller": controller})
            break
        if stage == "no_contact_motion" and stages[-1]["status"] != "PASS":
            break
        if stage == "fixed_object" and stages[-1]["status"] != "PASS":
            break
        if stage == "bottle_hold" and stages[-1]["status"] != "PASS":
            break
        stage_model, stage_build = build_model(stage, controller)
        controller["source_effort_limits_nm"] = stage_build["source_effort_limits_nm"]
        controller["arm_actuators"] = stage_build["arm_actuator_names"]
        apply_controller_limits(stage_model, controller, arm=stage in {"bottle_hold", "lift"})
        if stage == "bottle_hold":
            arm_poses = prepare_arm_poses(stage_model, controller)
            write_json(evidence / "arm_approach_ik.json", arm_poses)
        else:
            arm_poses = None
        # run_stage builds a clean model/data for each independent stage.
        result = run_stage(stage, evidence, controller, source, arm_poses)
        stages.append(result)
        write_json(evidence / "result.json", {"status": result["status"], "last_completed_stage": stage,
                                               "stages": stages, "source": source,
                                               "runtime": identity, "controller": controller})
        if result["status"] != "PASS":
            break
        if stage == "bottle_hold":
            lift = run_lift(evidence, controller, source, arm_poses or {})
            stages.append(lift)
            write_json(evidence / "result.json", {"status": lift["status"], "last_completed_stage": "lift",
                                                   "stages": stages, "source": source,
                                                   "runtime": identity, "controller": controller})
            break

    final_status = stages[-1]["status"] if stages else "FAIL"
    result = {"status": final_status, "last_completed_stage": stages[-1]["stage"] if stages else None,
              "stages": stages, "source": source, "runtime": identity, "controller": controller,
              "forbidden_mechanisms": {"right_jaw_equality": False, "right_follower_external_command": False,
                                       "follower_qpos_writes": 0, "bottle_qpos_writes": 0,
                                       "bottle_weld_or_mocap": False, "left_hand_assistance": False}}
    write_json(evidence / "result.json", result)
    report_lines = [
        "# Issue #46: X2 OmniPicker 1-DOF SIMULATION_ONLY_M0",
        "",
        f"Status: **{final_status}**",
        f"Last completed stage: `{result['last_completed_stage']}`",
        "",
        "## Claim boundary",
        "",
        "The X2 OmniPicker source geometry, joint ranges, and mimic mapping are source-derived. The paired-jaw actuator model is a SIMULATION_ONLY_M0 engineering abstraction and is not a validated model of the real X2 OmniPicker transmission or hardware dynamics.",
        "",
        "## Artifacts",
        "",
        "- `source_audit.json`: pinned source and joint audit.",
        "- `runtime_identity.json`: Python, MuJoCo native/runtime versions and library hash.",
        "- `controller_derivation.json`: shared aperture mapping and fixed controller parameters.",
        "- Per-stage `*_result.json`, `*_trace.jsonl`, and one bounded diagnosis for the first failed stage.",
        "- PNG/MP4 visual evidence when rendering and later gates are reached.",
        "",
        "## Stage results",
        "",
    ]
    for item in stages:
        report_lines.append(f"- `{item['stage']}`: **{item['status']}**; first gate: `{item.get('first_failed_gate')}`")
    (evidence / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0 if final_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
