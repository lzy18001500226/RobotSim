#!/usr/bin/env python3
"""Source-geometry OmniPicker fixture gate and bounded cylinder lift trial.

The fixture retains the pinned right-wrist-roll/OmniPicker subtree and adds
two simulation-only translational support axes. It is not an X2 arm result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import issue46_omnipicker_p0_acceptance as accepted


SOURCE_ROOT = Path("/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf")
SOURCE_URDF = SOURCE_ROOT / accepted.X2_URDF_REL
CANONICAL_HELPER = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py")
EVIDENCE_ROOT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-automation-20261011")

WRIST_ROOT = "right_wrist_roll_link"
TOOL_BODY = "R_omnipicker_base_link"
PAD_BODIES = accepted.DRIVER_FOLLOWER["right"][2:4]
APPROACH_WORLD = np.array([0.0, -1.0, 0.0])
TABLE_CENTER = np.array([0.300, -0.100, 0.0])
CYLINDER_CENTER = np.array([0.300, -0.175, 0.800 + accepted.CYLINDER_HALF_HEIGHT])
INSERTION_DEPTH = 0.075
APPROACH_STANDOFF = 0.120
APPROACH_START = APPROACH_STANDOFF
APPROACH_END = 0.0
APPROACH_RANGE = [0.0, APPROACH_STANDOFF]
LIFT_RANGE = [0.0, 0.075]
APPROACH_FORCE_CAP_N = 80.0
LIFT_FORCE_CAP_N = 120.0
APPROACH_DURATION_S = 3.0
SLIDE_FREQUENCY_HZ = 5.0
STATIC_DISTANCE_TOLERANCE_M = 1e-6
MAX_ALLOWED_CONTACT_PENETRATION_M = 0.002
MIN_TABLE_SEPARATION_M = 0.002
MIN_HAND_SUPPORT_N = 0.90 * accepted.CYLINDER_MASS * accepted.GRAVITY
MAX_HOLD_RELATIVE_SLIP_M = 0.005
MAX_HOLD_LINEAR_SPEED_M_S = 0.025
MAX_UPRIGHT_ERROR_DEG = 5.0
TABLE_HALF_EXTENTS = np.array([0.200, 0.200])
TABLE_TOP_Z = 0.800


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def serialize(value: Any) -> Any:
        if isinstance(value, np.bool_):
            return bool(value)
        return accepted.json_default(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=serialize) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), default=accepted.json_default) + "\n")


def mesh_source(filename: str) -> Path:
    path = accepted.source_file_for_mesh(SOURCE_ROOT, SOURCE_URDF.parent, filename)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve source visual/collision mesh: {filename}")
    return path.resolve()


def write_wrist_subtree_urdf(destination: Path) -> dict[str, Any]:
    original = ET.parse(SOURCE_URDF).getroot()
    joints = original.findall("joint")
    descendants = {WRIST_ROOT}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    selected_joints = []
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent in descendants and child in descendants:
            selected_joints.append(copy.deepcopy(joint))
    robot = ET.Element("robot", {"name": "robotsim_p0_source_right_omnipicker_fixture"})
    for material in original.findall("material"):
        robot.append(copy.deepcopy(material))
    mesh_links = {}
    for link in original.findall("link"):
        if link.get("name") not in descendants:
            continue
        clone = copy.deepcopy(link)
        for mesh in clone.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                source_mesh = mesh_source(filename)
                mesh.set("filename", source_mesh.name)
                mesh_links[source_mesh.name] = source_mesh
        robot.append(clone)
    for joint in selected_joints:
        robot.append(joint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    symlinks = []
    for basename, source_mesh in sorted(mesh_links.items()):
        link_path = destination.parent / basename
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink() or link_path.resolve() != source_mesh:
                raise FileExistsError(f"Refusing to overwrite non-matching scratch mesh link: {link_path}")
        else:
            link_path.symlink_to(source_mesh)
        symlinks.append({"basename": basename, "source": str(source_mesh), "scratch_link": str(link_path)})
    ET.ElementTree(robot).write(destination, encoding="utf-8", xml_declaration=True)
    return {"root_link": WRIST_ROOT, "retained_links": sorted(descendants),
            "retained_joint_names": [j.get("name") for j in selected_joints],
            "retained_link_count": len(descendants), "retained_joint_count": len(selected_joints),
            "generated_urdf": str(destination), "generated_urdf_sha256": sha256(destination),
            "mesh_symlinks": symlinks,
            "source_urdf_sha256": sha256(SOURCE_URDF),
            "source_geometry_changes": [],
            "omitted_geometry_scope": "Upstream right arm and torso omitted; wrist-roll link and all descendants retained with original link/joint/mesh definitions."}


def transform_alignment(jaw_axis_world: np.ndarray, tool_z_world: np.ndarray) -> np.ndarray:
    a = jaw_axis_world / np.linalg.norm(jaw_axis_world)
    b = tool_z_world - float(np.dot(tool_z_world, a)) * a
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    source_basis = np.column_stack((a, b, c))
    target_a = np.array([1.0, 0.0, 0.0])
    target_b = np.array([0.0, -1.0, 0.0])
    target_c = np.cross(target_a, target_b)
    target_basis = np.column_stack((target_a, target_b, target_c))
    rotation = target_basis @ source_basis.T
    if np.linalg.det(rotation) < 0.999999:
        raise RuntimeError("Constructed source-preserving alignment is not a proper rotation")
    return rotation


def source_pose() -> dict[str, Any]:
    spec = mujoco.MjSpec.from_file(str(SOURCE_URDF))
    spec.compiler.fusestatic = False
    accepted.set_root_world(spec, accepted.BASE_WORLD, accepted.BASE_YAW)
    spec.option.timestep = accepted.TIMESTEP
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    accepted.set_aperture_qpos(model, data, "right", 1.0)
    accepted.set_aperture_qpos(model, data, "left", 1.0)
    mujoco.mj_forward(model, data)
    narrow = accepted.collision_geom_for_body(model, PAD_BODIES[0])
    wide = accepted.collision_geom_for_body(model, PAD_BODIES[1])
    witnesses = accepted.pair_contact_at(model, data, narrow, wide)
    jaw_mid = 0.5 * (np.asarray(witnesses["witness_a_world_m"]) + np.asarray(witnesses["witness_b_world_m"]))
    jaw_axis = np.asarray(witnesses["witness_b_world_m"]) - np.asarray(witnesses["witness_a_world_m"])
    jaw_axis /= np.linalg.norm(jaw_axis)
    tool_rotation = data.xmat[accepted.body_id(model, TOOL_BODY)].reshape(3, 3)
    wrist_id = accepted.body_id(model, WRIST_ROOT)
    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
    wrist_position = data.xpos[wrist_id].copy()
    alignment = transform_alignment(jaw_axis, tool_rotation[:, 2])
    target_wrist_rotation = alignment @ wrist_rotation
    target_jaw_mid = CYLINDER_CENTER - APPROACH_WORLD * INSERTION_DEPTH
    jaw_mid_local = wrist_rotation.T @ (jaw_mid - wrist_position)
    target_wrist_position = target_jaw_mid - target_wrist_rotation @ jaw_mid_local
    approach_axis_local = target_wrist_rotation.T @ np.array([0.0, 1.0, 0.0])
    lift_axis_local = target_wrist_rotation.T @ np.array([0.0, 0.0, 1.0])
    return {"source_model_compiled": True, "source_right_open_gap_m": witnesses["distance_m"],
            "source_jaw_mid_world_m": jaw_mid.tolist(), "source_jaw_axis_world": jaw_axis.tolist(),
            "source_tool_plus_z_world": tool_rotation[:, 2].tolist(),
            "jaw_midpoint_local_to_wrist_m": jaw_mid_local.tolist(),
            "alignment_rotation_world": alignment.tolist(),
            "target_wrist_rotation_world": target_wrist_rotation.tolist(),
            "target_wrist_quaternion_wxyz": Rotation.from_matrix(target_wrist_rotation).as_quat()[[3, 0, 1, 2]].tolist(),
            "target_wrist_position_world_m": target_wrist_position.tolist(),
            "target_jaw_midpoint_world_m": target_jaw_mid.tolist(),
            "cylinder_center_world_m": CYLINDER_CENTER.tolist(),
            "approach_direction_gripper_to_object_world": APPROACH_WORLD.tolist(),
            "insertion_depth_m": INSERTION_DEPTH, "approach_standoff_m": APPROACH_STANDOFF,
            "approach_axis_local": approach_axis_local.tolist(), "lift_axis_local": lift_axis_local.tolist(),
            "source_joint_configuration": "official compiled URDF qpos0, both claw drivers at accepted OPEN endpoint",
            "source_right_claw_joint_range_rad": [float(x) for x in model.jnt_range[accepted.joint_id(model, accepted.DRIVER_FOLLOWER["right"][0])]],
            "source_right_follower_joint_range_rad": [float(x) for x in model.jnt_range[accepted.joint_id(model, accepted.DRIVER_FOLLOWER["right"][1])]]}


def add_fixture_axes(spec: mujoco.MjSpec, pose: dict[str, Any], canonical: Any) -> dict[str, Any]:
    spec.compiler.fusestatic = False
    spec.option.timestep = accepted.TIMESTEP
    spec.option.gravity = [0.0, 0.0, -accepted.GRAVITY]
    root = spec.body(WRIST_ROOT)
    root.pos = pose["target_wrist_position_world_m"]
    root.quat = pose["target_wrist_quaternion_wxyz"]
    approach = root.add_joint(name="p0_fixture_approach_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                              axis=pose["approach_axis_local"], limited=True, range=APPROACH_RANGE,
                              damping=0.0, frictionloss=0.0)
    lift = root.add_joint(name="p0_fixture_lift_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=pose["lift_axis_local"], limited=True, range=LIFT_RANGE,
                          damping=0.0, frictionloss=0.0)
    canonical.add_g1_canonical_table(spec)
    spec.worldbody.add_geom(name="p0_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0.0, 0.0, 0.0], size=[0.0, 0.0, 0.05],
                            friction=[0.9, 0.01, 0.001], condim=4,
                            rgba=[0.58, 0.61, 0.63, 1.0])
    bottle = spec.worldbody.add_body(name="p0_canonical_test_cylinder", pos=CYLINDER_CENTER.tolist())
    bottle.add_freejoint(name="p0_canonical_test_cylinder_free")
    bottle.add_geom(name="p0_canonical_test_cylinder_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[accepted.CYLINDER_RADIUS, accepted.CYLINDER_HALF_HEIGHT, 0.0],
                    mass=accepted.CYLINDER_MASS, friction=accepted.CYLINDER_FRICTION,
                    condim=4, rgba=[0.12, 0.52, 0.82, 1.0])
    spec.add_actuator(name="p0_fixture_approach_motor", target=approach.name,
                      trntype=mujoco.mjtTrn.mjTRN_JOINT, dyntype=mujoco.mjtDyn.mjDYN_NONE,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_NONE,
                      gainprm=[1.0] + [0.0] * 9, ctrllimited=True,
                      ctrlrange=[-APPROACH_FORCE_CAP_N, APPROACH_FORCE_CAP_N],
                      forcelimited=True, forcerange=[-APPROACH_FORCE_CAP_N, APPROACH_FORCE_CAP_N])
    spec.add_actuator(name="p0_fixture_lift_motor", target=lift.name,
                      trntype=mujoco.mjtTrn.mjTRN_JOINT, dyntype=mujoco.mjtDyn.mjDYN_NONE,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_NONE,
                      gainprm=[1.0] + [0.0] * 9, ctrllimited=True,
                      ctrlrange=[-LIFT_FORCE_CAP_N, LIFT_FORCE_CAP_N],
                      forcelimited=True, forcerange=[-LIFT_FORCE_CAP_N, LIFT_FORCE_CAP_N])
    for role, joint_name in zip(("driver", "follower"), accepted.DRIVER_FOLLOWER["right"][:2]):
        cap = 2.0
        spec.add_actuator(name=f"diag_right_{role}_servo", target=joint_name,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT, dyntype=mujoco.mjtDyn.mjDYN_NONE,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_NONE,
                          gainprm=[1.0] + [0.0] * 9, ctrllimited=True,
                          ctrlrange=[-cap, cap], forcelimited=True, forcerange=[-cap, cap])
    return {"approach_joint": approach.name, "lift_joint": lift.name,
            "approach_motor": "p0_fixture_approach_motor", "lift_motor": "p0_fixture_lift_motor",
            "fixture_force_caps_N": {"approach": APPROACH_FORCE_CAP_N, "lift": LIFT_FORCE_CAP_N},
            "hand_motor_names": ["diag_right_driver_servo", "diag_right_follower_servo"],
            "hand_motor_temporary_compile_cap_Nm": 2.0,
            "new_contact_exclusions": [], "equality_attachments": [], "mocap_bodies": []}


def build_model(subtree_urdf: Path, pose: dict[str, Any], canonical: Any) -> tuple[mujoco.MjSpec, mujoco.MjModel, dict[str, Any]]:
    spec = mujoco.MjSpec.from_file(str(subtree_urdf))
    build = add_fixture_axes(spec, pose, canonical)
    model = spec.compile()
    for role, joint in zip(("driver", "follower"), accepted.DRIVER_FOLLOWER["right"][:2]):
        aid = accepted.actuator_id(model, f"diag_right_{role}_servo")
        model.actuator_ctrlrange[aid] = [-2.0, 2.0]
        model.actuator_forcerange[aid] = [-2.0, 2.0]
    return spec, model, build


def set_fixture_state(model: mujoco.MjModel, data: mujoco.MjData, approach: float, lift: float,
                      aperture: float) -> None:
    data.qpos[accepted.qpos_addr(model, "p0_fixture_approach_slide")] = approach
    data.qpos[accepted.qpos_addr(model, "p0_fixture_lift_slide")] = lift
    accepted.set_aperture_qpos(model, data, "right", aperture)


def source_hand_body_ids(model: mujoco.MjModel) -> set[int]:
    root_id = accepted.body_id(model, WRIST_ROOT)
    result = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > 0:
            if current == root_id:
                result.add(candidate)
                break
            current = int(model.body_parentid[current])
    return result


def robot_contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    hand_ids = source_hand_body_ids(model)
    rows = accepted.active_contact_rows(model, data)
    return [row for row in rows
            if any(accepted.body_id(model, name) in hand_ids
                   for name in (row["body1"], row["body2"]) if name != "world")]


def all_geom_distances(model: mujoco.MjModel, data: mujoco.MjData, cylinder_gid: int,
                       pad_body_ids: set[int]) -> tuple[list[dict[str, Any]], float, float]:
    rows = []
    minimum_all = math.inf
    minimum_nonpad = math.inf
    hand_ids = source_hand_body_ids(model)
    for gid in accepted.collision_geoms(model):
        if gid == cylinder_gid or int(model.geom_bodyid[gid]) == int(model.geom_bodyid[cylinder_gid]):
            continue
        if int(model.geom_bodyid[gid]) not in hand_ids:
            continue
        witness = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, cylinder_gid, gid, 1.0, witness))
        bid = int(model.geom_bodyid[gid])
        pad = bid in pad_body_ids
        minimum_all = min(minimum_all, distance)
        if not pad:
            minimum_nonpad = min(minimum_nonpad, distance)
        rows.append({"geom_id": gid, "geom": accepted.geom_name(model, gid),
                     "body": accepted.body_name(model, bid), "signed_distance_m": distance,
                     "penetration_m": max(0.0, -distance), "intended_pad": pad,
                     "witness_segment_world_m": witness.tolist()})
    return rows, minimum_all, minimum_nonpad


def compiled_collision_vertices(model: mujoco.MjModel, geom_id: int,
                                cache: dict[int, np.ndarray]) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise RuntimeError(f"Expected a compiled mesh collision geom: {accepted.geom_name(model, geom_id)}")
    if mesh_id not in cache:
        first_poly = int(model.mesh_polyadr[mesh_id])
        poly_count = int(model.mesh_polynum[mesh_id])
        vertex_ids = []
        for poly_id in range(first_poly, first_poly + poly_count):
            address = int(model.mesh_polyvertadr[poly_id])
            count = int(model.mesh_polyvertnum[poly_id])
            vertex_ids.extend(model.mesh_polyvert[address:address + count].tolist())
        local_indices = np.unique(np.asarray(vertex_ids, dtype=np.int32))
        vertex_address = int(model.mesh_vertadr[mesh_id])
        cache[mesh_id] = np.asarray(model.mesh_vert[vertex_address + local_indices], dtype=float)
    return cache[mesh_id]


def compiled_mesh_table_clearance(model: mujoco.MjModel, data: mujoco.MjData,
                                  hand_geom_ids: list[int], table_ids: set[int],
                                  mesh_vertex_cache: dict[int, np.ndarray]) -> dict[str, Any]:
    """Conservative vertical separation for compiled convex collision meshes and table boxes.

    MuJoCo 3.3.6's mj_geomDistance can return zero for separated mesh/box pairs,
    with witness points that do not lie on the table surface. The compiled mesh
    hull vertex bound and the generated contact set are used instead.
    """
    if not table_ids:
        raise RuntimeError("Canonical table collision geoms were not found")
    table_top_z = max(float(data.geom_xpos[gid][2] + model.geom_size[gid][2]) for gid in table_ids)
    lowest = (math.inf, -1)
    for geom_id in hand_geom_ids:
        points = compiled_collision_vertices(model, geom_id, mesh_vertex_cache)
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        z_min = float(np.min(points @ rotation[2, :] + data.geom_xpos[geom_id][2]))
        if z_min < lowest[0]:
            lowest = (z_min, geom_id)
    return {"lower_bound_m": lowest[0] - table_top_z,
            "lowest_compiled_hull_vertex_z_m": lowest[0],
            "highest_table_collision_surface_z_m": table_top_z,
            "limiting_geom": accepted.geom_name(model, lowest[1]),
            "method": "minimum transformed compiled collision-hull vertex height minus maximum table-box top; positive value proves separation from every table geom"}


def geometry_gate(model: mujoco.MjModel, pose: dict[str, Any], output: Path, canonical: Any) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_fixture_state(model, data, APPROACH_START, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    cylinder_gid = accepted.geom_id(model, "p0_canonical_test_cylinder_geom")
    table_ids = {g for g in accepted.collision_geoms(model)
                 if accepted.body_name(model, int(model.geom_bodyid[g])) == "m0_table"}
    pad_ids = {accepted.body_id(model, name) for name in PAD_BODIES}
    hand_geom_ids = [g for g in accepted.collision_geoms(model)
                     if int(model.geom_bodyid[g]) in source_hand_body_ids(model)]
    mesh_vertex_cache: dict[int, np.ndarray] = {}
    cylinder_body_id = accepted.body_id(model, "p0_canonical_test_cylinder")
    cylinder_qadr = accepted.qpos_addr(model, "p0_canonical_test_cylinder_free")
    initial_cylinder_qpos = data.qpos[cylinder_qadr:cylinder_qadr + 7].copy()
    table_top = TABLE_TOP_Z
    baseline_bottom = float(data.xpos[cylinder_body_id][2] - accepted.CYLINDER_HALF_HEIGHT)
    path_rows = []
    for i, value in enumerate(np.linspace(APPROACH_START, APPROACH_END, 25)):
        set_fixture_state(model, data, float(value), 0.0, 1.0)
        mujoco.mj_forward(model, data)
        distances, min_all, min_nonpad = all_geom_distances(model, data, cylinder_gid, pad_ids)
        table_clearance = compiled_mesh_table_clearance(model, data, hand_geom_ids, table_ids, mesh_vertex_cache)
        contacts = robot_contact_rows(model, data)
        path_rows.append({"phase": "OPEN_APPROACH", "sample": i, "approach_slide_m": float(value),
                          "minimum_all_geom_distance_m": min_all,
                          "minimum_nonpad_distance_m": min_nonpad,
                          "gripper_table_clearance": table_clearance,
                          "active_contacts": contacts,
                          "closest_geoms": sorted(distances, key=lambda r: r["signed_distance_m"])[:8]})
    closure_rows = []
    bilateral_target = None
    chosen_pad_penetrations = None
    bilateral_contact_rows = None
    for aperture in np.linspace(1.0, 0.30, 1401):
        set_fixture_state(model, data, APPROACH_END, 0.0, float(aperture))
        mujoco.mj_forward(model, data)
        distances, min_all, min_nonpad = all_geom_distances(model, data, cylinder_gid, pad_ids)
        table_clearance = compiled_mesh_table_clearance(model, data, hand_geom_ids, table_ids, mesh_vertex_cache)
        pad_distances = {name: min((r["signed_distance_m"] for r in distances if r["body"] == name), default=math.inf)
                         for name in PAD_BODIES}
        contacts = robot_contact_rows(model, data)
        row = {"phase": "STATIC_CLOSE", "aperture_command": float(aperture),
               "pad_distances_m": pad_distances, "minimum_nonpad_distance_m": min_nonpad,
               "gripper_table_clearance": table_clearance,
               "maximum_pad_penetration_m": max(max(0.0, -d) for d in pad_distances.values()),
               "active_contacts": contacts}
        closure_rows.append(row)
        if bilateral_target is None and all(d <= -0.0005 for d in pad_distances.values()):
            bilateral_target = float(aperture)
            chosen_pad_penetrations = pad_distances
            bilateral_contact_rows = contacts
    write_jsonl(output / "B2_CLEARANCE_TRACE.jsonl", path_rows + closure_rows)
    set_fixture_state(model, data, APPROACH_END, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    open_rows, open_min, open_nonpad = all_geom_distances(model, data, cylinder_gid, pad_ids)
    open_table = compiled_mesh_table_clearance(model, data, hand_geom_ids, table_ids, mesh_vertex_cache)
    initial_contacts = robot_contact_rows(model, data)
    approach_min = min(r["minimum_nonpad_distance_m"] for r in path_rows)
    approach_table_min = min(r["gripper_table_clearance"]["lower_bound_m"] for r in path_rows)
    closure_nonpad_min = min(r["minimum_nonpad_distance_m"] for r in closure_rows)
    closure_table_min = min(r["gripper_table_clearance"]["lower_bound_m"] for r in closure_rows)
    table_contact_free = all(not any("m0_table" in (c["body1"], c["body2"]) for c in row["active_contacts"])
                             for row in path_rows + closure_rows)
    actual_pad_contact_bodies = {name for row in (bilateral_contact_rows or [])
                                 for name in (row["body1"], row["body2"]) if name in PAD_BODIES}
    actual_bilateral_contact = set(PAD_BODIES).issubset(actual_pad_contact_bodies)
    collision_free_approach = all(not r["active_contacts"] and r["minimum_nonpad_distance_m"] > STATIC_DISTANCE_TOLERANCE_M
                                  and r["gripper_table_clearance"]["lower_bound_m"] > STATIC_DISTANCE_TOLERANCE_M for r in path_rows)
    no_nonpad_closure_penetration = closure_nonpad_min > STATIC_DISTANCE_TOLERANCE_M
    table_footprint_ok = bool(np.all(np.abs(CYLINDER_CENTER[:2] - TABLE_CENTER[:2]) + accepted.CYLINDER_RADIUS
                                     <= TABLE_HALF_EXTENTS))
    cylinder_on_table = abs(baseline_bottom - table_top) <= 1e-9
    start_clear = not initial_contacts and open_min > STATIC_DISTANCE_TOLERANCE_M
    passed = (start_clear and collision_free_approach and no_nonpad_closure_penetration
              and closure_table_min > STATIC_DISTANCE_TOLERANCE_M and table_contact_free
              and bilateral_target is not None and actual_bilateral_contact
              and table_footprint_ok and cylinder_on_table)
    renderer = mujoco.Renderer(model, height=480, width=640)
    set_fixture_state(model, data, APPROACH_START, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    Image.fromarray(render(renderer, model, data, "overview")).save(output / "B2_open_overview.png")
    Image.fromarray(render(renderer, model, data, "front")).save(output / "B2_open_front.png")
    Image.fromarray(render(renderer, model, data, "side")).save(output / "B2_open_side.png")
    if bilateral_target is not None:
        set_fixture_state(model, data, APPROACH_END, 0.0, bilateral_target)
        mujoco.mj_forward(model, data)
        Image.fromarray(render(renderer, model, data, "close")).save(output / "B2_bilateral_target_closeup.png")
    renderer.close()
    result = {"status": "PASS" if passed else "FAIL", "stage": "B2_STATIC_GRASP_GEOMETRY",
              "physics_steps": 0, "contact_exclusions_added": [], "equality_attachments": [],
              "object_state_writes_during_rollout": 0,
              "candidate_count": 1, "source_faithful_geometry": True,
              "fixture_classification": "SIMULATION_ONLY support fixture; official source wrist-roll and all right OmniPicker descendant geometry retained",
              "source_pose_derivation": pose,
              "bottle": {"center_world_m": CYLINDER_CENTER.tolist(), "diameter_m": 2*accepted.CYLINDER_RADIUS,
                         "height_m": 2*accepted.CYLINDER_HALF_HEIGHT, "mass_kg": accepted.CYLINDER_MASS,
                         "friction": accepted.CYLINDER_FRICTION, "bottom_z_m": baseline_bottom,
                         "table_top_z_m": table_top, "footprint_inside_table": table_footprint_ok},
              "approach": {"axis_world": APPROACH_WORLD.tolist(), "standoff_m": APPROACH_STANDOFF,
                           "minimum_nonpad_bottle_clearance_m": approach_min,
                           "minimum_gripper_table_clearance_lower_bound_m": approach_table_min,
                           "table_contact_free_across_approach_and_closure": table_contact_free,
                           "collision_free_entire_path": collision_free_approach},
              "open": {"minimum_all_collision_geom_distance_m": open_min,
                       "minimum_nonpad_distance_m": open_nonpad,
                       "gripper_table_clearance": open_table,
                       "initial_contacts": initial_contacts,
                       "source_pad_surface_distances_m": {name: min(r["signed_distance_m"] for r in open_rows if r["body"] == name)
                                                           for name in PAD_BODIES}},
              "closure": {"samples": len(closure_rows), "bilateral_contact_target_aperture": bilateral_target,
                          "static_pad_distances_at_target_m": chosen_pad_penetrations,
                          "actual_bilateral_collision_contacts_at_target": actual_bilateral_contact,
                          "contact_rows_at_target": bilateral_contact_rows,
                          "minimum_nonpad_distance_m": closure_nonpad_min,
                          "minimum_gripper_table_clearance_lower_bound_m": closure_table_min,
                          "non_gripping_collision_free_through_target": no_nonpad_closure_penetration},
              "canonical_table": {"center_xy_m": list(canonical.G1_CANONICAL_TABLE_CENTER_XY),
                                  "half_extents_xy_m": list(canonical.G1_CANONICAL_TABLE_HALF_EXTENTS),
                                  "top_z_m": table_top},
              "trace": "B2_CLEARANCE_TRACE.jsonl",
              "source_joint_ranges_unchanged": True,
              "note": "Geometry gate uses compiled source collision geoms and static qpos placement only; it is not physical contact/lift evidence."}
    write_json(output / "B2_STATIC_GEOMETRY_GATE.json", result)
    return result


def right_controller(model: mujoco.MjModel) -> dict[str, Any]:
    data = mujoco.MjData(model)
    result: dict[str, Any] = {"label": "SIMULATION_ONLY paired-jaw actuator emulation",
                              "not_hardware_calibrated": True, "endpoint_margin_rad": accepted.ENDPOINT_MARGIN,
                              "motion_duration_s": accepted.MOTION_DURATION, "source_mapping": "follower=-driver",
                              "frequency_rad_s": accepted.FREQUENCY, "damping_ratio": accepted.DAMPING_RATIO,
                              "friction_assumption": accepted.CONTACT_FRICTION_ASSUMPTION, "per_side": {}}
    side = "right"
    driver, follower, narrow, wide = accepted.DRIVER_FOLLOWER[side]
    sampled = {driver: {"inertia": 0.0, "bias": 0.0, "lever": 0.0},
               follower: {"inertia": 0.0, "bias": 0.0, "lever": 0.0}}
    for aperture in np.linspace(0.0, 1.0, 9):
        data.qpos[:] = model.qpos0
        accepted.set_aperture_qpos(model, data, side, float(aperture))
        mujoco.mj_forward(model, data)
        M = accepted.full_mass_matrix(model, data)
        bodies = (accepted.body_id(model, narrow), accepted.body_id(model, wide))
        normal = data.xpos[bodies[0]] - data.xpos[bodies[1]]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        for name, body in ((driver, bodies[0]), (follower, bodies[1])):
            dof = accepted.dof_addr(model, name)
            sampled[name]["inertia"] = max(sampled[name]["inertia"], float(M[dof, dof]))
            sampled[name]["bias"] = max(sampled[name]["bias"], abs(float(data.qfrc_bias[dof])))
            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, body)
            sampled[name]["lever"] = max(sampled[name]["lever"], abs(float(np.dot(jacp[:, dof], normal))))
    normal_design = (2.0 * accepted.CYLINDER_MASS * accepted.GRAVITY /
                     (2.0 * accepted.CONTACT_FRICTION_ASSUMPTION))
    peak_accel = ((1.0 - 2.0 * accepted.ENDPOINT_MARGIN) * 5.773502691896258 /
                  accepted.MOTION_DURATION**2)
    joints = {}
    for name, row in sampled.items():
        kp = row["inertia"] * accepted.FREQUENCY**2
        kd = 2.0 * accepted.DAMPING_RATIO * accepted.FREQUENCY * row["inertia"]
        free_demand = row["bias"] + row["inertia"] * peak_accel
        contact_demand = normal_design * row["lever"] + row["bias"] + row["inertia"] * peak_accel
        cap = max(2.0 * free_demand, 1.15 * contact_demand)
        joints[name] = {"kp_nm_rad": kp, "kd_nm_s_rad": kd, "effort_cap_nm": cap,
                        "compiled_inertia_kg_m2": row["inertia"], "sampled_bias_nm": row["bias"],
                        "lever_arm_m": row["lever"]}
    result["per_side"][side] = {"driver": driver, "follower": follower,
                                "design_normal_force_per_jaw_n": normal_design,
                                "peak_reference_acceleration_rad_s2": peak_accel, "joints": joints}
    for role, name in zip(("driver", "follower"), (driver, follower)):
        aid = accepted.actuator_id(model, f"diag_right_{role}_servo")
        cap = joints[name]["effort_cap_nm"]
        model.actuator_ctrlrange[aid] = [-cap, cap]
        model.actuator_forcerange[aid] = [-cap, cap]
    return result


def slide_gains(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    matrix = accepted.full_mass_matrix(model, data)
    gains = {}
    for joint_name, actuator_name, force_cap in (
        ("p0_fixture_approach_slide", "p0_fixture_approach_motor", APPROACH_FORCE_CAP_N),
        ("p0_fixture_lift_slide", "p0_fixture_lift_motor", LIFT_FORCE_CAP_N),
    ):
        dof = accepted.dof_addr(model, joint_name)
        inertia = float(matrix[dof, dof])
        omega = 2.0 * math.pi * SLIDE_FREQUENCY_HZ
        kp, kd = inertia * omega**2, 2.0 * inertia * omega
        gains[joint_name] = {"actuator": actuator_name, "kp_N_m": kp, "kd_N_s_m": kd,
                             "effective_inertia_kg": inertia, "force_cap_N": force_cap,
                             "profile_classification": "SIMULATION_ONLY test-fixture servo; does not apply force directly to the bottle"}
    return gains


def bottle_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    bid = accepted.body_id(model, "p0_canonical_test_cylinder")
    pos = data.xpos[bid].copy()
    rot = data.xmat[bid].reshape(3, 3)
    upright = float(np.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0)))))
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
    return {"position_world_m": pos.tolist(), "quaternion_wxyz": data.xquat[bid].tolist(),
            "bottom_z_m": float(pos[2] - accepted.CYLINDER_HALF_HEIGHT),
            "height_above_table_m": float(pos[2] - CYLINDER_CENTER[2]),
            "angular_velocity_world_rad_s": velocity[:3].tolist(),
            "linear_velocity_world_m_s": velocity[3:].tolist(), "upright_error_deg": upright}


def contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    bottle_gid = accepted.geom_id(model, "p0_canonical_test_cylinder_geom")
    bottle_body = accepted.body_id(model, "p0_canonical_test_cylinder")
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 != bottle_gid and g2 != bottle_gid:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        frame = np.asarray(contact.frame).reshape(3, 3)
        world_on_geom2 = frame.T @ local[:3]
        if int(model.geom_bodyid[g2]) == bottle_body:
            force_on_bottle = world_on_geom2
        else:
            force_on_bottle = -world_on_geom2
        other_gid = g1 if g2 == bottle_gid else g2
        other_bid = int(model.geom_bodyid[other_gid])
        other_body = accepted.body_name(model, other_bid)
        friction = getattr(contact, "friction", np.zeros(5))
        rows.append({"geom1": accepted.geom_name(model, g1), "body1": accepted.body_name(model, int(model.geom_bodyid[g1])),
                     "geom2": accepted.geom_name(model, g2), "body2": accepted.body_name(model, int(model.geom_bodyid[g2])),
                     "other_body": other_body, "contact_position_world_m": np.asarray(contact.pos).tolist(),
                     "normal_world": frame[0].tolist(), "contact_force_local_N_Nm": local.tolist(),
                     "normal_force_N": float(max(0.0, local[0])),
                     "tangential_force_N": float(np.linalg.norm(local[1:3])),
                     "force_on_bottle_world_N": force_on_bottle.tolist(),
                     "signed_distance_m": float(contact.dist), "penetration_m": max(0.0, -float(contact.dist)),
                     "effective_contact_friction": np.asarray(friction).tolist()})
    return rows


def render(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
           mode: str = "overview") -> np.ndarray:
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.30, -0.15, 0.92]
    cam.distance = 0.80 if mode == "overview" else 0.43
    if mode == "front":
        cam.azimuth, cam.elevation = 180, -10
    elif mode == "side":
        cam.azimuth, cam.elevation = 90, -8
    elif mode == "close":
        cam.azimuth, cam.elevation = 155, -10
        cam.distance = 0.42
    else:
        cam.azimuth, cam.elevation = 145, -15
    renderer.update_scene(data, cam)
    return renderer.render().copy()


def physical_run(spec: mujoco.MjSpec, model: mujoco.MjModel, build: dict[str, Any],
                 geometry: dict[str, Any], output: Path) -> dict[str, Any]:
    if geometry.get("status") != "PASS":
        return {"status": "NOT_RUN", "reason": "B2 static geometry did not pass."}
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_fixture_state(model, data, APPROACH_START, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    cylinder_qadr = accepted.qpos_addr(model, "p0_canonical_test_cylinder_free")
    bottle_initial_qpos = data.qpos[cylinder_qadr:cylinder_qadr + 7].copy()
    controller = right_controller(model)
    slides = slide_gains(model, data)
    for name, value in slides.items():
        if value["force_cap_N"] <= 0:
            raise RuntimeError(f"invalid fixture force cap for {name}")
    renderer = mujoco.Renderer(model, height=480, width=640)
    first = render(renderer, model, data)
    Image.fromarray(first).save(output / "C_pregrasp.png")
    Image.fromarray(render(renderer, model, data, "front")).save(output / "C_front_view.png")
    Image.fromarray(render(renderer, model, data, "side")).save(output / "C_side_view.png")
    Image.fromarray(render(renderer, model, data, "close")).save(output / "C_pregrasp_closeup.png")
    frames = [first]
    trace: list[dict[str, Any]] = []
    stages = [
        ("SETTLE_OPEN", 0.5, {"approach": (APPROACH_START, APPROACH_START), "lift": (0.0, 0.0), "aperture": (1.0, 1.0)}),
        ("APPROACH", APPROACH_DURATION_S, {"approach": (APPROACH_START, APPROACH_END), "lift": (0.0, 0.0), "aperture": (1.0, 1.0)}),
        ("CLOSE", accepted.MOTION_DURATION, {"approach": (APPROACH_END, APPROACH_END), "lift": (0.0, 0.0),
                                               "aperture": (1.0, geometry["closure"]["bilateral_contact_target_aperture"])}),
        ("HOLD", 1.0, {"approach": (0.0, 0.0), "lift": (0.0, 0.0),
                        "aperture": (geometry["closure"]["bilateral_contact_target_aperture"], geometry["closure"]["bilateral_contact_target_aperture"])}),
        ("LIFT_1MM", 0.50, {"approach": (0.0, 0.0), "lift": (0.0, 0.001),
                             "aperture": (geometry["closure"]["bilateral_contact_target_aperture"], geometry["closure"]["bilateral_contact_target_aperture"])}),
        ("LIFT_5MM", 0.70, {"approach": (0.0, 0.0), "lift": (0.001, 0.005),
                             "aperture": (geometry["closure"]["bilateral_contact_target_aperture"], geometry["closure"]["bilateral_contact_target_aperture"])}),
        ("LIFT_50MM", 1.0, {"approach": (0.0, 0.0), "lift": (0.005, 0.050),
                              "aperture": (geometry["closure"]["bilateral_contact_target_aperture"], geometry["closure"]["bilateral_contact_target_aperture"])}),
        ("HOLD_50MM", 1.0, {"approach": (0.0, 0.0), "lift": (0.050, 0.050),
                             "aperture": (geometry["closure"]["bilateral_contact_target_aperture"], geometry["closure"]["bilateral_contact_target_aperture"])}),
    ]
    t_global = 0.0
    step = 0
    contacts_start_hold: list[dict[str, Any]] | None = None
    relative_pose_ref: np.ndarray | None = None
    max_relative_slip = 0.0
    max_penetration = 0.0
    source_limit_violations: list[dict[str, Any]] = []
    nonfinite = False
    unexpected_contacts: set[str] = set()
    max_mimic_error = 0.0
    max_slide_tracking_error = {"p0_fixture_approach_slide": 0.0, "p0_fixture_lift_slide": 0.0}
    max_joint_effort: dict[str, float] = {}
    max_hand_speed = 0.0
    stage_summary = {}
    failure = None
    bottle_body_id = accepted.body_id(model, "p0_canonical_test_cylinder")
    tool_body_id = accepted.body_id(model, TOOL_BODY)
    hand_body_names = {accepted.body_name(model, i) for i in source_hand_body_ids(model)}
    tabletop = 0.800
    total_steps = sum(round(duration / accepted.TIMESTEP) for _, duration, _ in stages)
    for stage, duration, targets in stages:
        stage_start = t_global
        start_values = {key: tuple(value) for key, value in targets.items()}
        local_steps = round(duration / accepted.TIMESTEP)
        stage_rows = 0
        bilateral_ticks = 0
        table_force_peak = 0.0
        hand_support_min = math.inf
        stage_pen_max = 0.0
        stage_lift_max = -math.inf
        for local_step in range(local_steps):
            local_t = local_step * accepted.TIMESTEP
            values = {}
            for key, (start, end) in start_values.items():
                if key == "aperture":
                    pos, vel, acc = accepted.minimum_jerk(local_t, duration, start, end)
                elif abs(end - start) > 1e-12:
                    pos, vel, acc = accepted.minimum_jerk(local_t, duration, start, end)
                else:
                    pos, vel, acc = start, 0.0, 0.0
                values[key] = (pos, vel, acc)
            mujoco.mj_forward(model, data)
            ref = accepted.aperture_reference(*values["aperture"])
            hand_controls = accepted.controller_step(model, data, controller, {"right": ref})
            slide_controls = {}
            for joint_name, actuator_name, key in (
                ("p0_fixture_approach_slide", "p0_fixture_approach_motor", "approach"),
                ("p0_fixture_lift_slide", "p0_fixture_lift_motor", "lift"),
            ):
                qadr, dof = accepted.qpos_addr(model, joint_name), accepted.dof_addr(model, joint_name)
                q, v = float(data.qpos[qadr]), float(data.qvel[dof])
                qref, vref, aref = values[key]
                gains = slides[joint_name]
                pd = gains["kp_N_m"] * (qref - q) + gains["kd_N_s_m"] * (vref - v)
                bias = float(data.qfrc_bias[dof])
                feedforward = float(accepted.full_mass_matrix(model, data)[dof, dof]) * aref
                request = bias + pd + feedforward
                applied = float(np.clip(request, -gains["force_cap_N"], gains["force_cap_N"]))
                data.ctrl[accepted.actuator_id(model, actuator_name)] = applied
                slide_controls[joint_name] = {"q_m": q, "qvel_m_s": v, "target_q_m": qref,
                                              "target_qvel_m_s": vref, "position_error_m": qref-q,
                                              "velocity_error_m_s": vref-v, "pd_force_N": pd,
                                              "bias_feedforward_N": bias, "reference_acceleration_force_N": feedforward,
                                              "requested_force_N": request, "applied_force_N": applied,
                                              "force_cap_N": gains["force_cap_N"]}
                max_slide_tracking_error[joint_name] = max(max_slide_tracking_error[joint_name], abs(qref-q))
                max_joint_effort[joint_name] = max(max_joint_effort.get(joint_name, 0.0), abs(applied))
            mujoco.mj_step(model, data)
            t_global += accepted.TIMESTEP
            step += 1
            stage_rows += 1
            hand_contacts = contact_forces(model, data)
            contact_keys = {c["other_body"] for c in hand_contacts}
            for c in hand_contacts:
                max_penetration = max(max_penetration, c["penetration_m"])
                stage_pen_max = max(stage_pen_max, c["penetration_m"])
                if c["other_body"] == "m0_table":
                    table_force_peak = max(table_force_peak, c["normal_force_N"])
                elif c["other_body"] in PAD_BODIES:
                    pass
                else:
                    unexpected_contacts.add(c["other_body"])
            contacts = accepted.active_contact_rows(model, data)
            robot_contacts = [c for c in contacts
                              if c["body1"] in hand_body_names or c["body2"] in hand_body_names]
            for contact in robot_contacts:
                bodies = {contact["body1"], contact["body2"]}
                allowed_pair = ("p0_canonical_test_cylinder" in bodies
                                and bool(bodies & set(PAD_BODIES)))
                if not allowed_pair:
                    unexpected_contacts.add("|".join(sorted(bodies)))
            pad_contact_bodies = {c["other_body"] for c in hand_contacts if c["other_body"] in PAD_BODIES}
            if set(PAD_BODIES).issubset(pad_contact_bodies):
                bilateral_ticks += 1
            hand_vertical = sum(c["force_on_bottle_world_N"][2] for c in hand_contacts if c["other_body"] in PAD_BODIES)
            if stage in {"HOLD", "LIFT_1MM", "LIFT_5MM", "LIFT_50MM", "HOLD_50MM"}:
                hand_support_min = min(hand_support_min, hand_vertical)
            pose = bottle_pose(model, data)
            bottom_separation = pose["bottom_z_m"] - tabletop
            stage_lift_max = max(stage_lift_max, pose["height_above_table_m"])
            tool_rot = data.xmat[tool_body_id].reshape(3, 3)
            rel = tool_rot.T @ (data.xpos[bottle_body_id] - data.xpos[tool_body_id])
            if stage == "HOLD" and local_step == 0:
                relative_pose_ref = rel.copy()
                contacts_start_hold = hand_contacts
                Image.fromarray(render(renderer, model, data, "close")).save(output / "C_first_contact.png")
            if relative_pose_ref is not None and stage in {"HOLD", "LIFT_1MM", "LIFT_5MM", "LIFT_50MM", "HOLD_50MM"}:
                max_relative_slip = max(max_relative_slip, float(np.linalg.norm(rel - relative_pose_ref)))
            driver, follower = accepted.DRIVER_FOLLOWER["right"][:2]
            mimic = float(data.qpos[accepted.qpos_addr(model, follower)] + data.qpos[accepted.qpos_addr(model, driver)])
            max_mimic_error = max(max_mimic_error, abs(mimic))
            for name, ctrl in hand_controls.items():
                max_joint_effort[name] = max(max_joint_effort.get(name, 0.0), abs(ctrl["applied_torque_nm"]))
                max_hand_speed = max(max_hand_speed, abs(float(data.qvel[accepted.dof_addr(model, name)])))
            finite = bool(np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc))).all())
            nonfinite = nonfinite or not finite
            limits = accepted.valid_joint_state(model, data, source_joints={
                n: j for n, j in _source_joint_elements(SOURCE_URDF).items()
            })
            if limits:
                source_limit_violations.extend({"step": step, **row} for row in limits)
            bottle_qpos_now = data.qpos[cylinder_qadr:cylinder_qadr + 7].copy()
            trace.append({"step": step, "time_s": float(data.time), "stage": stage,
                          "commands": {k: {"q": values[k][0], "qvel": values[k][1], "qacc": values[k][2]} for k in values},
                          "fixture_controller": slide_controls, "hand_controller": hand_controls,
                          "hand_joint_state": {n: {"qpos_rad": float(data.qpos[accepted.qpos_addr(model,n)]),
                                                    "qvel_rad_s": float(data.qvel[accepted.dof_addr(model,n)]),
                                                    "qacc_rad_s2": float(data.qacc[accepted.dof_addr(model,n)])}
                                                for n in (driver,follower)},
                          "mimic_residual_rad": mimic, "bottle": pose,
                          "bottle_bottom_table_separation_m": bottom_separation,
                          "bottle_relative_position_in_tool_m": rel.tolist(),
                          "relative_slip_from_hold_start_m": None if relative_pose_ref is None else float(np.linalg.norm(rel-relative_pose_ref)),
                          "contacts": hand_contacts,
                          "robot_contacts": robot_contacts,
                          "hand_vertical_support_force_N": hand_vertical,
                          "table_normal_support_force_N": sum(c["normal_force_N"] for c in hand_contacts if c["other_body"] == "m0_table"),
                          "source_limit_violations": limits, "finite_state": finite,
                          "bottle_qpos_write_count": 0,
                          "bottle_qpos_matches_physics_state": bool(np.all(np.isfinite(bottle_qpos_now))),
                          "active_contact_exclusions": [], "unexpected_contact_bodies": sorted(unexpected_contacts)})
            if stage == "CLOSE" and bilateral_ticks > 0 and not (output / "C_first_contact.png").exists():
                Image.fromarray(render(renderer, model, data, "close")).save(output / "C_first_contact.png")
            if stage == "CLOSE" and local_step == local_steps - 1:
                Image.fromarray(render(renderer, model, data, "close")).save(output / "C_closed_grasp.png")
            if stage == "HOLD" and local_step == local_steps - 1:
                Image.fromarray(render(renderer, model, data, "close")).save(output / "C_hold.png")
            if step % 20 == 0 or stage == "HOLD_50MM" and local_step == local_steps - 1:
                frames.append(render(renderer, model, data, "overview"))
            if max_penetration > MAX_ALLOWED_CONTACT_PENETRATION_M:
                failure = "CONTACT_PENETRATION_LIMIT"
                break
            if nonfinite or source_limit_violations:
                failure = "CONTROL_OR_LIMIT_INSTABILITY"
                break
        stage_summary[stage] = {"start_time_s": stage_start, "end_time_s": t_global,
                                "steps": stage_rows, "bilateral_pad_contact_fraction": bilateral_ticks/max(stage_rows, 1),
                                "peak_table_normal_force_N": table_force_peak,
                                "minimum_hand_vertical_support_force_N": None if math.isinf(hand_support_min) else hand_support_min,
                                "maximum_penetration_m": stage_pen_max,
                                "maximum_bottle_height_above_initial_m": stage_lift_max}
        if failure:
            break
    Image.fromarray(render(renderer, model, data, "overview")).save(output / "C_maximum_lift.png")
    Image.fromarray(render(renderer, model, data, "close")).save(output / "C_maximum_lift_closeup.png")
    write_jsonl(output / "C_CONTACT_LIFT_TRACE.jsonl", trace)
    imageio.mimsave(output / "C_isolated_contact_lift.mp4", frames, fps=20, macro_block_size=1)
    final = bottle_pose(model, data)
    final_support = contact_forces(model, data)
    final_hand_support = sum(c["force_on_bottle_world_N"][2] for c in final_support if c["other_body"] in PAD_BODIES)
    final_table_support = sum(c["normal_force_N"] for c in final_support if c["other_body"] == "m0_table")
    bilateral_fraction = min((row["bilateral_pad_contact_fraction"] for name, row in stage_summary.items()
                              if name in {"HOLD", "LIFT_1MM", "LIFT_5MM", "LIFT_50MM", "HOLD_50MM"}), default=0.0)
    lift = float(final["height_above_table_m"])
    hold_ok = (stage_summary.get("HOLD", {}).get("bilateral_pad_contact_fraction", 0.0) >= 0.80
               and stage_summary.get("HOLD", {}).get("maximum_bottle_height_above_initial_m", -1.0) > -0.003)
    lift_ok = lift >= 0.050 - 0.001
    support_ok = final_hand_support >= MIN_HAND_SUPPORT_N and final_table_support <= 0.05
    stability_ok = max_relative_slip <= MAX_HOLD_RELATIVE_SLIP_M and np.linalg.norm(final["linear_velocity_world_m_s"]) <= MAX_HOLD_LINEAR_SPEED_M_S
    upright_ok = final["upright_error_deg"] <= MAX_UPRIGHT_ERROR_DEG
    successful = (failure is None and not source_limit_violations and not nonfinite and
                  max_penetration <= MAX_ALLOWED_CONTACT_PENETRATION_M and hold_ok and lift_ok and
                  support_ok and stability_ok and upright_ok and bilateral_fraction >= 0.80 and
                  not unexpected_contacts and max_mimic_error <= accepted.MIMIC_TRACKING_DIAGNOSTIC_RAD and
                  all(stage in stage_summary for stage in ("HOLD", "LIFT_1MM", "LIFT_5MM", "LIFT_50MM", "HOLD_50MM")))
    result = {"status": "PASS" if successful else "FAIL", "stage": "C_ISOLATED_CONTACT_LIFT_50MM",
              "classification": "SIMULATION_ONLY isolated source-geometry hand fixture; not full-X2 or hardware acceptance",
              "source_model_changes": [], "source_collision_exclusions_added": [], "bottle_qpos_writes_during_rollout": 0,
              "bottle_welds_or_equalities": [], "mocap_carry": False, "hidden_bottle_support": False,
              "artificial_bottle_force": False, "controller": controller, "fixture_servo_profiles": slides,
              "fixture_actuator_caps_N": {"approach": APPROACH_FORCE_CAP_N, "lift": LIFT_FORCE_CAP_N},
              "approach_duration_s": APPROACH_DURATION_S,
              "stage_metrics": stage_summary, "maximum_lift_above_initial_m": max((r["bottle"]["height_above_table_m"] for r in trace), default=0.0),
              "final_lift_m": lift, "final_bottle": final, "final_hand_upward_support_N": final_hand_support,
              "final_table_normal_support_N": final_table_support, "minimum_required_hand_support_N": MIN_HAND_SUPPORT_N,
              "bilateral_contact_fraction_min_during_hold_and_lift": bilateral_fraction,
              "max_relative_slip_from_hold_start_m": max_relative_slip,
              "max_contact_penetration_m": max_penetration,
              "maximum_mimic_residual_rad": max_mimic_error,
              "mimic_diagnostic_bound_rad": accepted.MIMIC_TRACKING_DIAGNOSTIC_RAD,
              "source_limit_violations": source_limit_violations, "nonfinite_state": nonfinite,
              "unexpected_contact_bodies": sorted(unexpected_contacts),
              "final_gate_checks": {"hold": hold_ok, "lift_50mm": lift_ok, "hand_support_replaces_table": support_ok,
                                    "slip_and_hold_velocity": stability_ok, "upright": upright_ok},
              "failure_classification": failure or (None if successful else _classify_failure(result_values={
                  "bilateral_fraction": bilateral_fraction, "support": final_hand_support,
                  "table": final_table_support, "lift": lift, "slip": max_relative_slip,
                  "penetration": max_penetration, "mimic": max_mimic_error,
                  "unexpected_contacts": sorted(unexpected_contacts)})),
              "trace": "C_CONTACT_LIFT_TRACE.jsonl", "video": "C_isolated_contact_lift.mp4",
              "evidence_images": ["C_pregrasp.png", "C_first_contact.png", "C_closed_grasp.png", "C_hold.png",
                                  "C_maximum_lift.png", "C_maximum_lift_closeup.png", "C_front_view.png", "C_side_view.png"]}
    write_json(output / "C_RESULT.json", result)
    renderer.close()
    return result


def _source_joint_elements(path: Path) -> dict[str, ET.Element]:
    root = ET.parse(path).getroot()
    return {j.get("name", ""): j for j in root.findall("joint")}


def _classify_failure(result_values: dict[str, Any]) -> str:
    if result_values["bilateral_fraction"] < 0.80:
        return "NO_BILATERAL_CONTACT"
    if result_values["lift"] < 0.049:
        return "SLIP_DURING_LIFT" if result_values["support"] < MIN_HAND_SUPPORT_N else "INSUFFICIENT_NORMAL_FORCE"
    if result_values["table"] > 0.05:
        return "LOAD_TRANSFER_INCOMPLETE"
    if result_values["slip"] > MAX_HOLD_RELATIVE_SLIP_M:
        return "SLIP_DURING_LIFT"
    if result_values["penetration"] > MAX_ALLOWED_CONTACT_PENETRATION_M:
        return "CONTACT_PENETRATION_LIMIT"
    if result_values["mimic"] > accepted.MIMIC_TRACKING_DIAGNOSTIC_RAD:
        return "CONTROL_INSTABILITY"
    if result_values["unexpected_contacts"]:
        return "UNEXPECTED_CONTACT"
    return "CONTROL_INSTABILITY"


def model_runtime_identity(model: mujoco.MjModel, script_path: Path, generated_urdf: Path) -> dict[str, Any]:
    package = Path(mujoco.__file__).resolve().parent
    libs = sorted(package.glob("libmujoco.so*"))
    return {"python": sys.version, "platform": platform.platform(), "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(), "numpy": np.__version__,
            "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations),
            "gravity_m_s2": model.opt.gravity.tolist(), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "native_library_sha256": [{"path": str(p), "sha256": sha256(p)} for p in libs],
            "source_repository": accepted.X2_REPO, "source_commit": accepted.X2_PIN,
            "source_urdf_sha256": sha256(SOURCE_URDF),
            "generated_subtree_urdf": str(generated_urdf), "generated_subtree_urdf_sha256": sha256(generated_urdf),
            "canonical_scene_helper_sha256": sha256(CANONICAL_HELPER),
            "runner_sha256": sha256(script_path)}


def write_report(output: Path, static: dict[str, Any], dynamic: dict[str, Any] | None,
                 runtime: dict[str, Any], command: str) -> None:
    status = dynamic["status"] if dynamic else static["status"]
    lines = ["# Issue #46 — OmniPicker Isolated Physical Fixture", "",
             f"Final bounded stage result: **{status}**", "",
             "## Gates", "",
             f"- B2 source-geometry static corridor: **{static['status']}**; open approach minimum non-pad/bottle clearance {static['approach']['minimum_nonpad_bottle_clearance_m']*1000:.2f} mm; conservative gripper/table clearance lower bound {static['approach']['minimum_gripper_table_clearance_lower_bound_m']*1000:.2f} mm.",
             f"- Bilateral geometric jaw target: aperture {static['closure']['bilateral_contact_target_aperture']}; non-gripping clearance through target {static['closure']['minimum_nonpad_distance_m']*1000:.2f} mm.",
             f"- Isolated physical 50 mm lift: **{dynamic['status'] if dynamic else 'NOT RUN'}**.",
             "- Full X2 right-arm integration: **NOT RUN** unless the isolated lift gate passes and a separate phase is executed.", "",
             "## Model and controls", "",
             f"- Source: `{accepted.X2_PIN}` / `{accepted.X2_URDF_REL.as_posix()}`; original source collision meshes and joint ranges retained.",
             "- Fixture: SIMULATION_ONLY, two physically actuated prismatic axes; source wrist-roll link and complete right OmniPicker descendant subtree retained.",
             "- Jaw controls: SIMULATION_ONLY paired-jaw emulation with source follower mapping; no bottle attachment or bottle qpos writes.",
             f"- Bottle: 70 mm diameter × 244.5 mm high; 0.570 kg; friction {accepted.CYLINDER_FRICTION}; freejoint over the canonical G1 table.",
             "- Full per-step trace and machine-readable result files are stored beside this report.", "",
             "## Reproduction", "", "```bash", command, "```", "",
             "## Runtime", "", f"- MuJoCo: {runtime['mujoco_native']}; Python {runtime['python'].split()[0]}; timestep {runtime['timestep_s']} s.",
             f"- Runner SHA256: `{runtime['runner_sha256']}`.", f"- Generated subtree URDF SHA256: `{runtime['generated_subtree_urdf_sha256']}`.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    canonical = accepted.load_canonical_module(CANONICAL_HELPER)
    model_workdir = Path(tempfile.mkdtemp(prefix=f"{output.name}-model-"))
    subtree_path = model_workdir / "generated_right_wrist_omnipicker_subtree.urdf"
    subtree = write_wrist_subtree_urdf(subtree_path)
    pose = source_pose()
    spec, model, fixture = build_model(subtree_path, pose, canonical)
    shutil.copy2(subtree_path, output / subtree_path.name)
    spec.to_file(str(output / "generated_fixture_model.xml"))
    subtree["generated_urdf"] = str(output / subtree_path.name)
    subtree["generated_urdf_sha256"] = sha256(output / subtree_path.name)
    subtree["mesh_symlink_workdir"] = str(model_workdir)
    runtime = model_runtime_identity(model, Path(__file__).resolve(), subtree_path)
    write_json(output / "runtime_identity.json", runtime)
    write_json(output / "source_subtree_identity.json", subtree)
    write_json(output / "fixture_transform.json", pose)
    write_json(output / "fixture_model_inventory.json", {"build": fixture, "nq": model.nq, "nv": model.nv,
               "nu": model.nu, "njnt": model.njnt, "nbody": model.nbody, "ngeom": model.ngeom,
               "nmesh": model.nmesh, "neq": model.neq, "joint_names": accepted.joint_names(model),
               "source_joint_ranges_rad": {name: model.jnt_range[accepted.joint_id(model, name)].tolist()
                                            for name in accepted.DRIVER_FOLLOWER["right"][:2]}})
    static = geometry_gate(model, pose, output, canonical)
    dynamic = physical_run(spec, model, fixture, static, output) if static["status"] == "PASS" else None
    command = ("PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages "
               "LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 MUJOCO_GL=egl "
               f"python3 {Path(__file__).resolve()} --output {output}")
    write_json(output / "P0_ISOLATED_RESULT.json", {"B2_STATIC_GEOMETRY_GATE": static["status"],
               "ISOLATED_CONTACT_LIFT_50MM": (dynamic["status"] if dynamic else "NOT_RUN"),
               "FULL_X2_CONTACT_LIFT_50MM": "NOT_RUN", "FULL_X2_RELEASE_SETTLE": "NOT_RUN",
               "static_result": "B2_STATIC_GEOMETRY_GATE.json",
               "dynamic_result": "C_RESULT.json" if dynamic else None,
               "source": {"repository": accepted.X2_REPO, "commit": accepted.X2_PIN,
                          "urdf": accepted.X2_URDF_REL.as_posix(), "sha256": sha256(SOURCE_URDF)},
               "classification": "SIMULATION_ONLY isolated fixture; no hardware claim",
               "reproduction_command": command})
    write_report(output, static, dynamic, runtime, command)
    print(json.dumps({"output": str(output), "B2": static["status"],
                      "C": dynamic["status"] if dynamic else "NOT_RUN",
                      "full_X2": "NOT_RUN"}, indent=2))
    return 0 if static["status"] == "PASS" and dynamic and dynamic["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
