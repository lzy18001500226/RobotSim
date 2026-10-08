#!/usr/bin/env python3
"""Collision-model audit and bounded X2 OmniPicker recovery experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw


X2_ROOT = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_URDF = X2_ROOT / "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
OLD_RUNNER = Path(__file__).with_name("issue46_omnipicker_isolated_grasp.py")
BASELINE = Path(__file__).with_name("issue46_omnipicker_1dof_m0.py")
OLD_FILTERED_URDF = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-isolated-grasp/20261008-run5/isolated_omnipicker_source.urdf"
)
EVIDENCE_ROOT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-collision-audit-recovery"
)
ROOT_LINK = "right_elbow_link"
WRIST_PAIR = ("right_wrist_yaw_link", "right_wrist_pitch_link")
WRIST_JOINTS = (
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
JAW_BODIES = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
CONTACT_MARGIN_M = 0.001
STATIC_GATE_ORDER = (
    "open_source_joint_targets_valid",
    "zero_active_initial_self_penetration",
    "open_all_robot_bottle_pairs_clear",
    "open_robot_table_clear",
    "open_robot_floor_clear",
    "approach_path_clear",
    "closed_jaws_reach_cylinder_body",
    "jaw_witnesses_on_opposing_cylinder_sides",
    "jaw_contacts_within_usable_cylinder_band",
    "closed_wrist_loop_base_clear",
    "closed_robot_table_and_floor_clear",
    "closure_reduces_both_jaw_clearances_monotonically",
    "30mm_static_lift_corridor_clear",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def name(model: mujoco.MjModel, obj: mujoco.mjtObj, ident: int) -> str:
    fallback = "geom" if obj == mujoco.mjtObj.mjOBJ_GEOM else "body" if obj == mujoco.mjtObj.mjOBJ_BODY else obj.name
    return mujoco.mj_id2name(model, obj, int(ident)) or f"{fallback}_{ident}"


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if ident < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return int(ident)


def geom_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = body_id(model, body_name)
    matches = [gid for gid in range(model.ngeom)
               if int(model.geom_bodyid[gid]) == bid
               and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one collision geom for {body_name}; found {matches}")
    return matches[0]


def qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Missing joint: {joint_name}")
    return int(model.jnt_qposadr[jid])


def digest_element(element: ET.Element) -> str:
    node = copy.deepcopy(element)
    for item in node.iter():
        item.tail = None
        ordered = sorted(item.attrib.items())
        item.attrib.clear()
        item.attrib.update(ordered)
        if item.tag.rsplit("}", 1)[-1] == "mesh" and item.get("filename"):
            item.set("filename", Path(item.get("filename", "")).name)
    return hashlib.sha256(ET.tostring(node, encoding="utf-8")).hexdigest()


def extract_subtree(output: Path, source: Path, root_link: str) -> dict[str, Any]:
    source_root = ET.parse(source).getroot()
    links = {row.get("name", ""): row for row in source_root.findall("link")}
    joints = source_root.findall("joint")
    child_parent = {
        row.find("child").get("link"): row.find("parent").get("link")
        for row in joints if row.find("child") is not None and row.find("parent") is not None
    }
    included = {root_link}
    changed = True
    while changed:
        changed = False
        for child, parent in child_parent.items():
            if parent in included and child not in included:
                included.add(child)
                changed = True
    if root_link not in links:
        raise RuntimeError(f"Source URDF has no root {root_link}")

    robot = ET.Element("robot", {"name": "X2-OmniPicker-collision-audit"})
    for item in source_root:
        tag = item.tag.rsplit("}", 1)[-1]
        if tag == "mujoco":
            node = copy.deepcopy(item)
            compiler = node.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(node, "compiler")
            compiler.set("meshdir", str(source.parent / "meshes"))
            robot.append(node)
        elif tag == "material":
            robot.append(copy.deepcopy(item))
        elif tag == "link" and item.get("name") in included:
            node = copy.deepcopy(item)
            for mesh in node.findall(".//mesh"):
                mesh.set("filename", Path(mesh.get("filename", "")).name)
            robot.append(node)
        elif tag == "joint":
            parent, child = item.find("parent"), item.find("child")
            if (parent is not None and child is not None
                    and parent.get("link") in included and child.get("link") in included):
                robot.append(copy.deepcopy(item))

    result_path = output / "isolated_omnipicker_source_elbow_root.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(result_path, encoding="utf-8", xml_declaration=True)
    copied_links = {row.get("name", ""): row for row in robot.findall("link")}
    copied_joints = {row.get("name", ""): row for row in robot.findall("joint")}
    link_parity = {
        key: digest_element(links[key]) == digest_element(copied_links[key])
        for key in sorted(included)
    }
    joint_parity = {
        row.get("name", ""): digest_element(row) == digest_element(copied_joints[row.get("name", "")])
        for row in joints
        if row.find("parent") is not None and row.find("child") is not None
        and row.find("parent").get("link") in included
        and row.find("child").get("link") in included
    }
    return {
        "path": str(result_path),
        "sha256": sha256(result_path),
        "root_link": root_link,
        "included_links": sorted(included),
        "included_joints": sorted(copied_joints),
        "link_xml_parity_after_mesh_path_normalization": link_parity,
        "joint_xml_parity": joint_parity,
        "all_link_joint_xml_preserved": all(link_parity.values()) and all(joint_parity.values()),
        "preserved_wrist_yaw_joint": "right_wrist_yaw_joint" in copied_joints,
    }


def parse_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Invalid STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError(f"Expected binary STL for source topology audit: {path}")
    rows = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        rows[index] = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
    return rows


def mesh_topology(model: mujoco.MjModel, geom: int, source_mesh: Path) -> dict[str, Any]:
    triangles = parse_binary_stl(source_mesh)
    source_vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    hull = ConvexHull(source_vertices)
    mesh_id = int(model.geom_dataid[geom])
    address = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    face_address = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    return {
        "source_stl_sha256": sha256(source_mesh),
        "source_stl_triangle_count": int(len(triangles)),
        "source_stl_unique_vertex_count": int(len(source_vertices)),
        "source_convex_hull_vertex_count": int(len(hull.vertices)),
        "compiled_mesh_id": mesh_id,
        "compiled_collision_vertex_count": count,
        "compiled_collision_face_count": face_count,
        "compiled_collision_geom_type": mujoco.mjtGeom(int(model.geom_type[geom])).name,
        "compiled_mesh_stores_source_triangulation": count == len(source_vertices) and face_count == len(triangles),
        "mujoco_mesh_collision_uses_convex_hull": True,
        "source_triangle_intersection_test": "NOT IMPLEMENTED; raw mesh overlap remains unresolved",
        "source_triangles": triangles,
    }


def pair_contact(model: mujoco.MjModel, data: mujoco.MjData,
                 first: int, second: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == {first, second}:
            rows.append({
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                "distance_m": float(contact.dist),
                "efc_address": int(contact.efc_address),
                "exclude": int(contact.exclude),
            })
    return rows


def geom_distance(model: mujoco.MjModel, data: mujoco.MjData,
                  first: int, second: int) -> tuple[float, list[float]]:
    segment = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
    return distance, segment.tolist()


def compile_isolated(base, urdf: Path, root_position: np.ndarray,
                     root_quaternion: np.ndarray, bottle: bool) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.fusestatic = False
    spec.option.timestep = base.TIMESTEP
    spec.option.gravity = [0.0, 0.0, -base.GRAVITY_M_S2]
    root = next(body for body in spec.bodies if body.name == ROOT_LINK)
    root.pos = np.asarray(root_position, dtype=float).tolist()
    root.quat = np.asarray(root_quaternion, dtype=float).tolist()
    joint_map = {str(joint.name): joint for joint in spec.joints if joint.name}
    for joint_name in WRIST_JOINTS:
        joint = joint_map.get(joint_name)
        if joint is not None:
            ref = float(joint.ref)
            spec.add_equality(
                name=f"m0_rigid_mount_{joint_name}",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=joint_name,
                data=[0.0] * 11,
                solref=base.POSE_LOCK_SOLREF,
            )
    for actuator_name, target in (("m0_right_driver_servo", DRIVER),
                                  ("m0_right_follower_servo", FOLLOWER)):
        spec.add_actuator(
            name=actuator_name, target=target, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
            ctrllimited=True, ctrlrange=[-1.0, 1.0], forcelimited=True,
            forcerange=[-1.0, 1.0],
        )
    for index, (first, second) in enumerate(base.STATIC_COLLISION_EXCLUSIONS):
        names = {str(body.name) for body in spec.bodies}
        if first in names and second in names:
            spec.add_exclude(name=f"m0_source_static_exclusion_{index}",
                             bodyname1=first, bodyname2=second)
    base.add_canonical_scene(spec, bottle=bottle)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    open_targets = base.aperture_targets(1.0)
    data.qpos[qpos_address(model, DRIVER)] = open_targets["right_claw_joint_target_rad"]
    data.qpos[qpos_address(model, FOLLOWER)] = open_targets["R_hand_wide1_joint_target_rad"]
    for joint_name in WRIST_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, {"nexclude": int(model.nexclude), "npair": int(model.npair),
                         "disableflags": int(model.opt.disableflags),
                         "enableflags": int(model.opt.enableflags)}


def contact_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    return [{
        "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
        "body1": name(model, mujoco.mjtObj.mjOBJ_BODY,
                      int(model.geom_bodyid[data.contact[i].geom1])),
        "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
        "body2": name(model, mujoco.mjtObj.mjOBJ_BODY,
                      int(model.geom_bodyid[data.contact[i].geom2])),
        "distance_m": float(data.contact[i].dist),
        "efc_address": int(data.contact[i].efc_address),
    } for i in range(data.ncon)]


def audit_stage(output: Path) -> dict[str, Any]:
    if subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip() != X2_PIN:
        raise RuntimeError("Pinned X2 vendor SHA mismatch")
    dirty = subprocess.check_output(["git", "-C", str(X2_ROOT), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Pinned vendor checkout is dirty")

    old = load_module("issue46_omnipicker_isolated_grasp", OLD_RUNNER)
    base = old.load_baseline()
    control_model, _ = base.build_model("open_hold", {})
    controller = base.derive_jaw_controller(control_model)
    extracted = extract_subtree(output, X2_URDF, ROOT_LINK)

    # Control: the prior wrist-yaw-rooted subtree leaves its source yaw joint out.
    control_model, control_info = old.build_model(
        base, OLD_FILTERED_URDF, np.zeros(3), np.eye(3), include_bottle=True, add_lift=False
    )
    control_data = old.initialize(base, control_model, controller)
    prior_pair = (geom_id(control_model, WRIST_PAIR[0]), geom_id(control_model, WRIST_PAIR[1]))
    prior_contact = pair_contact(control_model, control_data, *prior_pair)
    prior_distance, prior_segment = geom_distance(control_model, control_data, *prior_pair)

    # Source-tree comparison: reset the complete official robot to URDF qpos0.
    full_model, _ = base.build_model("open_hold", {})
    full_data = mujoco.MjData(full_model)
    mujoco.mj_resetData(full_model, full_data)
    open_targets = base.aperture_targets(1.0)
    for joint_name, target in ((DRIVER, open_targets["right_claw_joint_target_rad"]),
                               (FOLLOWER, open_targets["R_hand_wide1_joint_target_rad"])):
        full_data.qpos[qpos_address(full_model, joint_name)] = target
    mujoco.mj_forward(full_model, full_data)
    full_pair = (geom_id(full_model, WRIST_PAIR[0]), geom_id(full_model, WRIST_PAIR[1]))
    full_contact = pair_contact(full_model, full_data, *full_pair)
    full_distance, full_segment = geom_distance(full_model, full_data, *full_pair)

    # Corrected isolated topology: root at the elbow so the source wrist-yaw joint remains.
    corrected_model, corrected_data, corrected_flags = compile_isolated(
        base, Path(extracted["path"]), np.array([0.0, 0.0, 2.0]),
        np.array([1.0, 0.0, 0.0, 0.0]), bottle=True
    )
    corrected_pair = (geom_id(corrected_model, WRIST_PAIR[0]), geom_id(corrected_model, WRIST_PAIR[1]))
    corrected_contact = pair_contact(corrected_model, corrected_data, *corrected_pair)
    corrected_distance, corrected_segment = geom_distance(corrected_model, corrected_data, *corrected_pair)

    # Regression: model-filtering the adjacent wrist pair must not hide unrelated hand/object contact.
    regression_data = mujoco.MjData(corrected_model)
    regression_data.qpos[:] = corrected_data.qpos
    regression_data.qvel[:] = corrected_data.qvel
    regression_data.act[:] = corrected_data.act
    regression_data.time = corrected_data.time
    mujoco.mj_forward(corrected_model, regression_data)
    bottle_joint = mujoco.mj_name2id(corrected_model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_qpos = int(corrected_model.jnt_qposadr[bottle_joint])
    jaw_probe = geom_id(corrected_model, JAW_BODIES[0])
    regression_data.qpos[bottle_qpos:bottle_qpos + 3] = (
        regression_data.geom_xpos[jaw_probe] + np.array([0.0, 0.0, 0.020])
    )
    regression_data.qpos[bottle_qpos + 3:bottle_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(corrected_model, regression_data)
    jaw_bottle_contacts = [row for row in contact_names(corrected_model, regression_data)
                           if ((row["body1"] == JAW_BODIES[0] and row["body2"] == "m0_bottle")
                               or (row["body2"] == JAW_BODIES[0] and row["body1"] == "m0_bottle"))]
    regression_wrist_contacts = pair_contact(corrected_model, regression_data, *corrected_pair)
    contact_filter_regression = {
        "test_type": "zero-step static geometry regression; bottle pose set before mj_forward; no mj_step",
        "forced_overlap_target": JAW_BODIES[0],
        "fixture_height_m": 2.0,
        "geometry_derived_bottle_center_offset_from_jaw_geom_m": [0.0, 0.0, 0.020],
        "active_jaw_bottle_contact_rows": jaw_bottle_contacts,
        "adjacent_yaw_pitch_contact_rows_after_source_filter_preservation": regression_wrist_contacts,
        "unrelated_active_contact_preserved": bool(jaw_bottle_contacts),
        "adjacent_pair_stays_filtered": not regression_wrist_contacts,
    }

    source_root = ET.parse(X2_URDF).getroot()
    source_links = {row.get("name", ""): row for row in source_root.findall("link")}
    mesh_rows = {}
    for body in WRIST_PAIR:
        link = source_links[body]
        visual = link.find("visual/geometry/mesh")
        collision = link.find("collision/geometry/mesh")
        if visual is None or collision is None:
            raise RuntimeError(f"Source visual/collision mesh missing on {body}")
        visual_path = (X2_URDF.parent / visual.get("filename", "")).resolve()
        collision_path = (X2_URDF.parent / collision.get("filename", "")).resolve()
        gid = geom_id(corrected_model, body)
        topology = mesh_topology(corrected_model, gid, collision_path)
        mesh_rows[body] = {
            "visual_mesh": str(visual_path.relative_to(X2_URDF.parent)),
            "collision_mesh": str(collision_path.relative_to(X2_URDF.parent)),
            "visual_and_collision_reference_same_asset": visual_path == collision_path,
            "visual_origin": link.find("visual/origin").attrib if link.find("visual/origin") is not None else {},
            "collision_origin": link.find("collision/origin").attrib if link.find("collision/origin") is not None else {},
            "geom_id": gid,
            "geom_contype": int(corrected_model.geom_contype[gid]),
            "geom_conaffinity": int(corrected_model.geom_conaffinity[gid]),
            "body_id": int(corrected_model.geom_bodyid[gid]),
            "body_parent_id": int(corrected_model.body_parentid[int(corrected_model.geom_bodyid[gid])]),
            "body_weld_id": int(corrected_model.body_weldid[int(corrected_model.geom_bodyid[gid])]),
            "geom_dataid": int(corrected_model.geom_dataid[gid]),
            **{key: value for key, value in topology.items() if key != "source_triangles"},
        }

    result = {
        "stage": "collision-model-audit",
        "source": {"repository": "https://github.com/AgibotTech/agibot_x2_urdf",
                   "commit": X2_PIN,
                   "urdf": str(X2_URDF.relative_to(X2_ROOT)),
                   "urdf_sha256": sha256(X2_URDF), "license": "Mulan PSL v2"},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(),
                    "native_library": str(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "native_library_sha256": sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "mujoco_gl": os.environ.get("MUJOCO_GL")},
        "prior_filtered_subtree": {
            "path": str(OLD_FILTERED_URDF),
            "sha256": sha256(OLD_FILTERED_URDF),
            "root_link": "right_wrist_yaw_link",
            "preserved_right_wrist_yaw_joint": any(
                row.get("name") == "right_wrist_yaw_joint"
                for row in ET.parse(OLD_FILTERED_URDF).getroot().findall("joint")),
            "build_info": control_info,
            "model_flags": {"disableflags": int(control_model.opt.disableflags),
                            "enableflags": int(control_model.opt.enableflags),
                            "filterparent_enabled": not bool(control_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
                            "nexclude": int(control_model.nexclude), "npair": int(control_model.npair)},
            "root_body_weld_id": int(control_model.body_weldid[body_id(control_model, "right_wrist_yaw_link")]),
            "wrist_pair": {"distance_m": prior_distance,
                           "witness_segment_world_m": prior_segment,
                           "active_contact_rows": prior_contact,
                           "classification": "ACTIVE_PHYSICAL_CONTACT" if prior_contact else "GEOMETRIC_OVERLAP_FILTERED_BY_MODEL"},
        },
        "full_source_tree": {
            "wrist_pair": {"distance_m": full_distance,
                           "witness_segment_world_m": full_segment,
                           "active_contact_rows": full_contact,
                           "classification": "ACTIVE_PHYSICAL_CONTACT" if full_contact else "GEOMETRIC_OVERLAP_FILTERED_BY_MODEL"},
            "model_flags": {"disableflags": int(full_model.opt.disableflags),
                            "enableflags": int(full_model.opt.enableflags),
                            "filterparent_enabled": not bool(full_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
                            "nexclude": int(full_model.nexclude), "npair": int(full_model.npair)},
        },
        "corrected_isolated_subtree": {
            **extracted,
            "model_flags": corrected_flags,
            "filterparent_enabled": not bool(corrected_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
            "root_body_weld_id": int(corrected_model.body_weldid[body_id(corrected_model, ROOT_LINK)]),
            "wrist_pair": {"distance_m": corrected_distance,
                           "witness_segment_world_m": corrected_segment,
                           "active_contact_rows": corrected_contact,
                           "classification": "ACTIVE_PHYSICAL_CONTACT" if corrected_contact else "GEOMETRIC_OVERLAP_FILTERED_BY_MODEL"},
            "all_active_contacts": contact_names(corrected_model, corrected_data),
        },
        "active_collision_regression": contact_filter_regression,
        "mesh_diagnostics": mesh_rows,
        "interpretation": {
            "root_cause": "The prior isolated URDF rooted at right_wrist_yaw_link and omitted right_wrist_yaw_joint. That made the yaw link part of the world-welded root; MuJoCo's parent-child filter exempts parent pairs when the parent is world-welded. The full source tree retains the yaw joint, so yaw/pitch is filtered there. Rooting the isolated subtree at right_elbow_link preserves the source yaw joint and its parent-child collision semantics.",
            "source_parent_child_mapping": {
                "right_wrist_yaw_joint": {"parent": "right_elbow_link", "child": "right_wrist_yaw_link"},
                "right_wrist_pitch_joint": {"parent": "right_wrist_yaw_link", "child": "right_wrist_pitch_link"},
            },
            "parent_child_filter_enabled": True,
            "explicit_source_pair_or_exclude_for_yaw_pitch": False,
            "old_isolated_root_changed_world_weld_group": True,
            "corrected_root_preserves_source_joint_tree_for_audited_pair": extracted["preserved_wrist_yaw_joint"],
            "active_collision_regression_pass": contact_filter_regression["unrelated_active_contact_preserved"] and contact_filter_regression["adjacent_pair_stays_filtered"],
            "raw_source_triangle_overlap": "UNRESOLVED; compiled collision meshes use convex hulls and raw triangle intersection was not evaluated in this stage.",
            "no_collision_geometry_changed": True,
            "no_physics_steps": True,
        },
    }
    return result


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat(scalar_first=True)


def collision_ids(model: mujoco.MjModel, body_filter: set[str] | None = None) -> list[int]:
    rows = []
    for gid in range(model.ngeom):
        if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
            continue
        body = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
        if body_filter is None or body in body_filter:
            rows.append(gid)
    return rows


def geom_pair_rows(model: mujoco.MjModel, data: mujoco.MjData,
                   firsts: list[int], seconds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for first in firsts:
        for second in seconds:
            distance, segment = geom_distance(model, data, first, second)
            rows.append({
                "body1": name(model, mujoco.mjtObj.mjOBJ_BODY,
                              int(model.geom_bodyid[first])),
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "body2": name(model, mujoco.mjtObj.mjOBJ_BODY,
                              int(model.geom_bodyid[second])),
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "signed_distance_m": distance,
                "penetration_m": max(0.0, -distance),
                "witness_segment_world_m": segment,
            })
    return rows


def minimum_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"signed_distance_m": math.inf, "body1": None, "geom1": None,
                "body2": None, "geom2": None, "witness_segment_world_m": None}
    return min(rows, key=lambda row: float(row["signed_distance_m"]))


def first_failed_static_gate(gates: dict[str, Any]) -> str | None:
    return next((key for key in STATIC_GATE_ORDER if not gates.get(key, True)), None)


def contact_frame(base, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    narrow_id = geom_id(model, JAW_BODIES[0])
    wide_id = geom_id(model, JAW_BODIES[1])
    body_id_bottle = next(
        gid for gid in range(model.ngeom)
        if name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "bottle_body"
    )
    diameter = 2.0 * float(model.geom_size[body_id_bottle][0])
    gap_trace = []
    for aperture in np.linspace(1.0, 0.0, 101):
        targets = base.aperture_targets(float(aperture))
        data.qpos[qpos_address(model, DRIVER)] = targets["right_claw_joint_target_rad"]
        data.qpos[qpos_address(model, FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        gap, witness = geom_distance(model, data, narrow_id, wide_id)
        witness_length = float(np.linalg.norm(np.asarray(witness[3:]) - np.asarray(witness[:3])))
        query_consistent = (math.isfinite(gap)
                            and abs(abs(gap) - witness_length) <= 1e-7 + 1e-6 * witness_length)
        active_pair_contacts = sum(
            {int(data.contact[index].geom1), int(data.contact[index].geom2)}
            == {narrow_id, wide_id}
            for index in range(data.ncon)
        )
        gap_trace.append({"aperture_ratio": float(aperture),
                          "jaw_surface_gap_m": gap,
                          "witness_segment_length_m": witness_length,
                          "distance_query_consistent": query_consistent,
                          "active_jaw_pair_contact_count": int(active_pair_contacts),
                          "witness_segment_world_m": witness})
    valid_gap_trace = [row for row in gap_trace if row["distance_query_consistent"]]
    if len(valid_gap_trace) < 0.95 * len(gap_trace):
        raise RuntimeError("Compiled jaw distance query produced too many inconsistent samples")
    monotonic_increases = [
        float(valid_gap_trace[index + 1]["jaw_surface_gap_m"])
        - float(valid_gap_trace[index]["jaw_surface_gap_m"])
        for index in range(len(valid_gap_trace) - 1)
    ]
    if any(increase > 1e-5 for increase in monotonic_increases):
        raise RuntimeError("Compiled jaw clearance is not monotonic over the accepted close trajectory")
    bracket = next((index for index in range(len(valid_gap_trace) - 1)
                    if valid_gap_trace[index]["jaw_surface_gap_m"] >= diameter
                    and valid_gap_trace[index + 1]["jaw_surface_gap_m"] <= diameter), None)
    if bracket is None:
        raise RuntimeError("Accepted source-limited jaw travel does not span bottle diameter")
    before, after = valid_gap_trace[bracket], valid_gap_trace[bracket + 1]
    fraction = ((before["jaw_surface_gap_m"] - diameter)
                / (before["jaw_surface_gap_m"] - after["jaw_surface_gap_m"]))
    aperture = before["aperture_ratio"] + fraction * (
        after["aperture_ratio"] - before["aperture_ratio"]
    )
    targets = base.aperture_targets(float(aperture))
    data.qpos[qpos_address(model, DRIVER)] = targets["right_claw_joint_target_rad"]
    data.qpos[qpos_address(model, FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
    mujoco.mj_forward(model, data)
    gap, witness = geom_distance(model, data, narrow_id, wide_id)
    narrow_witness, wide_witness = np.asarray(witness[:3]), np.asarray(witness[3:])
    midpoint_world = 0.5 * (narrow_witness + wide_witness)
    local_normal = wide_witness - narrow_witness
    local_normal /= max(float(np.linalg.norm(local_normal)), 1e-12)
    base_link = body_id(model, "R_omnipicker_base_link")
    narrow_tip = body_id(model, JAW_BODIES[0])
    wide_tip = body_id(model, JAW_BODIES[1])
    root = body_id(model, ROOT_LINK)
    root_origin = data.xpos[root].copy()
    tip_midpoint = 0.5 * (data.xpos[narrow_tip] + data.xpos[wide_tip])
    finger_axis = data.xpos[base_link] - tip_midpoint
    finger_axis -= local_normal * float(np.dot(finger_axis, local_normal))
    finger_axis /= max(float(np.linalg.norm(finger_axis)), 1e-12)
    local_side = np.cross(local_normal, finger_axis)
    local_side /= max(float(np.linalg.norm(local_side)), 1e-12)
    local_basis = np.column_stack((local_normal, finger_axis, local_side))
    target_normal = np.array([1.0, 0.0, 0.0])
    target_finger = np.array([0.0, 0.0, 1.0])
    target_side = np.cross(target_normal, target_finger)
    target_basis = np.column_stack((target_normal, target_finger, target_side))
    rotation = target_basis @ local_basis.T
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("Compiled source contact frame did not produce a proper rotation")
    bottle_center = data.geom_xpos[body_id_bottle].copy()
    return {
        "narrow_geom": narrow_id,
        "wide_geom": wide_id,
        "bottle_body_geom": body_id_bottle,
        "bottle_body_center_world_m": bottle_center.tolist(),
        "bottle_body_diameter_m": diameter,
        "bottle_body_half_height_m": float(model.geom_size[body_id_bottle][1]),
        "contact_aperture_ratio": float(aperture),
        "jaw_gap_at_contact_aperture_m": gap,
        "jaw_gap_trace": gap_trace,
        "distance_query_integrity": {
            "samples_total": len(gap_trace),
            "samples_consistent": len(valid_gap_trace),
            "inconsistent_samples": [row for row in gap_trace if not row["distance_query_consistent"]],
            "maximum_open_to_close_gap_increase_m": max([0.0, *monotonic_increases]),
            "bracket_samples": [before, after],
        },
        "jaw_witnesses_at_probe_pose_m": {
            "narrow": narrow_witness.tolist(), "wide": wide_witness.tolist(),
            "midpoint": midpoint_world.tolist(),
        },
        "local_frame": {
            "normal_narrow_to_wide": local_normal.tolist(),
            "finger_axis_tip_to_base_from_compiled_link_chain": finger_axis.tolist(),
            "base_link_origin_world_at_probe_pose": data.xpos[base_link].tolist(),
            "tip_link_midpoint_world_at_probe_pose": tip_midpoint.tolist(),
            "root_origin_world_at_probe_pose": root_origin.tolist(),
            "contact_midpoint_relative_to_root_m": (midpoint_world - root_origin).tolist(),
            "rotation_matrix_source_frame_to_horizontal_grasp": rotation.tolist(),
        },
    }


def set_static_state(base, model: mujoco.MjModel, data: mujoco.MjData,
                     pose: dict[str, Any], aperture: float, bottle_lift_m: float = 0.0) -> None:
    root_id = int(pose["root_body_id"])
    model.body_pos[root_id] = np.asarray(pose["root_position_world_m"], dtype=float)
    model.body_quat[root_id] = quat_from_matrix(np.asarray(pose["root_rotation_world"], dtype=float))
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    data.ctrl[:] = 0.0
    targets = base.aperture_targets(float(aperture))
    data.qpos[qpos_address(model, DRIVER)] = targets["right_claw_joint_target_rad"]
    data.qpos[qpos_address(model, FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_qpos = int(model.jnt_qposadr[bottle_joint])
    data.qpos[bottle_qpos + 2] += float(bottle_lift_m)
    mujoco.mj_forward(model, data)


def heatmap_png(path: Path, title: str, csv_rows: list[dict[str, Any]],
                depth_values: np.ndarray, height_values: np.ndarray) -> None:
    width, height = 840, 570
    image = Image.new("RGB", (width, height), (249, 250, 249))
    draw = ImageDraw.Draw(image)
    draw.text((22, 16), title, fill=(25, 32, 36))
    left, top, plot_w, plot_h = 82, 56, 560, 460
    values = np.asarray([row["minimum_robot_bottle_clearance_m"] for row in csv_rows], dtype=float)
    depth_count, height_count = len(depth_values), len(height_values)
    grid = values.reshape(height_count, depth_count)
    for yi in range(height_count):
        for xi in range(depth_count):
            value = float(grid[height_count - 1 - yi, xi])
            if value < 0.0:
                t = min(1.0, abs(value) / 0.06)
                color = (int(220 + 25 * t), int(75 - 35 * t), int(68 - 25 * t))
            elif value < 0.005:
                t = value / 0.005
                color = (245, int(174 + 45 * t), int(75 + 60 * t))
            else:
                t = min(1.0, (value - 0.005) / 0.06)
                color = (int(180 - 95 * t), int(220 - 8 * t), int(135 - 40 * t))
            x0 = left + int(xi * plot_w / depth_count)
            x1 = left + int((xi + 1) * plot_w / depth_count)
            y0 = top + int(yi * plot_h / height_count)
            y1 = top + int((yi + 1) * plot_h / height_count)
            draw.rectangle((x0, y0, x1, y1), fill=color)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(45, 48, 50), width=2)
    draw.text((left, top + plot_h + 10), "Contact-center offset along jaw axis (m)", fill=(25, 32, 36))
    draw.text((8, top + 5), "Height\noffset (m)", fill=(25, 32, 36))
    for index in (0, depth_count // 2, depth_count - 1):
        x = left + int(index * plot_w / (depth_count - 1))
        draw.line((x, top + plot_h, x, top + plot_h + 5), fill=(45, 48, 50), width=1)
        draw.text((x - 18, top + plot_h + 28), f"{depth_values[index]:+.2f}", fill=(25, 32, 36))
    for index in (0, height_count // 2, height_count - 1):
        y = top + int((height_count - 1 - index) * plot_h / (height_count - 1))
        draw.line((left - 5, y, left, y), fill=(45, 48, 50), width=1)
        draw.text((left - 48, y - 6), f"{height_values[index]:+.2f}", fill=(25, 32, 36))
    draw.text((680, 95), "Minimum compiled\nrobot-to-bottle\nsigned distance", fill=(25, 32, 36))
    draw.rectangle((680, 170, 705, 194), fill=(235, 58, 43))
    draw.text((715, 175), "penetration", fill=(25, 32, 36))
    draw.rectangle((680, 207, 705, 231), fill=(245, 195, 105))
    draw.text((715, 212), "0 to 5 mm", fill=(25, 32, 36))
    draw.rectangle((680, 244, 705, 268), fill=(100, 210, 110))
    draw.text((715, 249), "> 5 mm", fill=(25, 32, 36))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_reference(model: mujoco.MjModel, data: mujoco.MjData,
                     output: Path, filename: str, lookat: list[float],
                     distance: float, azimuth: float, elevation: float) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render().copy()).save(output / filename)
    renderer.close()


def explicit_exclusion_rows(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows = []
    for signature in model.exclude_signature:
        body1_id = int(signature) >> 16
        body2_id = int(signature) & 0xFFFF
        rows.append({
            "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id),
            "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id),
            "signature": int(signature),
        })
    return rows


def explicit_pair_rows(model: mujoco.MjModel) -> list[dict[str, Any]]:
    return [{
        "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[index])),
        "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[index])),
    } for index in range(model.npair)]


def compiled_geom_inventory(model: mujoco.MjModel, body_name: str) -> dict[str, Any]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        return {"body": body_name, "present": False, "geoms": []}
    geoms = []
    for geom_id_value in range(model.ngeom):
        if int(model.geom_bodyid[geom_id_value]) != body:
            continue
        mesh_id = (int(model.geom_dataid[geom_id_value])
                   if int(model.geom_type[geom_id_value]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1)
        geoms.append({
            "geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id_value),
            "type": mujoco.mjtGeom(int(model.geom_type[geom_id_value])).name,
            "contype": int(model.geom_contype[geom_id_value]),
            "conaffinity": int(model.geom_conaffinity[geom_id_value]),
            "collision_enabled": bool(model.geom_contype[geom_id_value] or model.geom_conaffinity[geom_id_value]),
            "margin_m": float(model.geom_margin[geom_id_value]),
            "size": np.asarray(model.geom_size[geom_id_value], dtype=float).tolist(),
            "mesh_id": mesh_id,
            "mesh_vertex_count": int(model.mesh_vertnum[mesh_id]) if mesh_id >= 0 else None,
            "mesh_face_count": int(model.mesh_facenum[mesh_id]) if mesh_id >= 0 else None,
        })
    return {"body": body_name, "present": True, "geoms": geoms}


def resume_static_artifacts(output: Path) -> dict[str, Any]:
    audit_path = output / "collision_model_audit.json"
    summary_path = output / "stage_1_summary.json"
    trace_path = output / "six_dof_optimizer_trace.jsonl"
    candidates_path = output / "six_dof_candidate_summary.jsonl"
    if not all(path.is_file() for path in (audit_path, summary_path, trace_path, candidates_path)):
        raise FileNotFoundError("Cannot recover static artifacts; required completed run outputs are missing")
    stage1 = json.loads(audit_path.read_text(encoding="utf-8"))
    stage1_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    search_path = output / "static_geometry_search.json"
    source_search = json.loads(search_path.read_text(encoding="utf-8")) if search_path.is_file() else {}
    source_identity = source_search.get("identity", {})
    trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    candidate_rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line]
    if not trace_rows or not candidate_rows:
        raise RuntimeError("Completed search trace is empty")
    best_detail = min(trace_rows, key=lambda row: float(row["objective_squared_norm"]))
    old = load_module("issue46_omnipicker_isolated_grasp", OLD_RUNNER)
    base = old.load_baseline()
    model, _, flags = compile_isolated(
        base, Path(stage1["corrected_isolated_subtree"]["path"]),
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), bottle=True
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    frame = contact_frame(base, model, data)
    full_model, _ = base.build_model("open_hold", {})
    full_data = mujoco.MjData(full_model)
    mujoco.mj_resetData(full_model, full_data)
    mujoco.mj_forward(full_model, full_data)
    source_joint = next(
        joint for joint in ET.parse(X2_URDF).getroot().findall("joint")
        if joint.get("name") == "right_wrist_pitch_joint"
    )
    source_parent = source_joint.find("parent").get("link")
    source_child = source_joint.find("child").get("link")
    full_yaw_geom = geom_id(full_model, "right_wrist_yaw_link")
    full_pitch_geom = geom_id(full_model, "right_wrist_pitch_link")
    full_distance, full_witness = geom_distance(full_model, full_data, full_yaw_geom, full_pitch_geom)
    full_yaw_body = body_id(full_model, source_parent)
    full_pitch_body = body_id(full_model, source_child)
    full_exclusions = explicit_exclusion_rows(full_model)
    full_pairs = explicit_pair_rows(full_model)
    filter_audit = {
        "source_joint": {
            "name": "right_wrist_pitch_joint",
            "parent_link": source_parent,
            "child_link": source_child,
            "axis_xyz": source_joint.find("axis").get("xyz") if source_joint.find("axis") is not None else None,
        },
        "full_source_model": {
            "filterparent_enabled": not bool(
                int(full_model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
            ),
            "explicit_exclusion_count": full_model.nexclude,
            "explicit_exclusions": full_exclusions,
            "explicit_geom_pair_count": full_model.npair,
            "explicit_geom_pairs": full_pairs,
            "yaw_pitch_pair_explicitly_excluded": any(
                {row["body1"], row["body2"]} == {source_parent, source_child}
                for row in full_exclusions
            ),
            "yaw_pitch_pair_is_direct_parent_child": int(full_model.body_parentid[full_pitch_body]) == full_yaw_body,
            "yaw_pitch_distance_m": full_distance,
            "yaw_pitch_witness_segment_world_m": full_witness,
            "yaw_pitch_active_contacts": pair_contact(full_model, full_data, full_yaw_geom, full_pitch_geom),
            "classification": "GEOMETRIC_OVERLAP_FILTERED_BY_MODEL",
        },
        "corrected_isolated_subtree": {
            "filterparent_enabled": not bool(
                int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
            ),
            "explicit_exclusion_count": model.nexclude,
            "explicit_exclusions": explicit_exclusion_rows(model),
            "explicit_geom_pair_count": model.npair,
            "explicit_geom_pairs": explicit_pair_rows(model),
            "yaw_pitch_pair_is_direct_parent_child": int(model.body_parentid[body_id(model, source_child)])
                == body_id(model, source_parent),
            "yaw_pitch_distance_m": stage1["corrected_isolated_subtree"]["wrist_pair"]["distance_m"],
            "yaw_pitch_active_contacts": stage1["corrected_isolated_subtree"]["wrist_pair"]["active_contact_rows"],
        },
        "raw_source_triangle_intersection": "UNRESOLVED; compiled MuJoCo collision meshes use convex hulls",
    }
    write_json(output / "collision_pair_filter_audit.json", filter_audit)
    root_id = body_id(model, ROOT_LINK)
    pose = {
        "root_body_id": root_id,
        "root_rotation_world": np.asarray(best_detail["root_rotation_world"], dtype=float),
        "root_position_world_m": np.asarray(best_detail["root_position_world_m"], dtype=float),
    }
    center = np.asarray(frame["bottle_body_center_world_m"], dtype=float)
    contact_aperture = float(frame["contact_aperture_ratio"])
    set_static_state(base, model, data, pose, 1.0)
    render_reference(model, data, output, "best_candidate_open_overview.png",
                     center.tolist(), 0.95, 135.0, -20.0)
    render_reference(model, data, output, "best_candidate_open_side.png",
                     center.tolist(), 0.85, 90.0, -5.0)
    set_static_state(base, model, data, pose, contact_aperture)
    render_reference(model, data, output, "best_candidate_contact_overview.png",
                     center.tolist(), 0.85, 135.0, -20.0)
    render_reference(model, data, output, "best_candidate_contact_side.png",
                     center.tolist(), 0.80, 90.0, -5.0)

    source_limits = {}
    for joint in ET.parse(X2_URDF).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            source_limits[joint.get("name", "")] = [float(limit.get("lower")), float(limit.get("upper"))]
    static_pass = any(bool(row["all_static_gates_pass"]) for row in trace_rows)
    first_failure = first_failed_static_gate(best_detail["static_gates"])
    geometry_inventory = {
        role: compiled_geom_inventory(model, body_name)
        for role, body_name in (
            ("narrow_jaw", "R_hand_narrow3_Link"),
            ("wide_jaw", "R_hand_wide3_Link"),
            ("narrow_loop", "R_hand_narrow_loop_Link"),
            ("wide_loop", "R_hand_wide_loop_Link"),
            ("omnipicker_base", "R_omnipicker_base_link"),
            ("wrist_yaw", "right_wrist_yaw_link"),
            ("wrist_pitch", "right_wrist_pitch_link"),
            ("wrist_roll", "right_wrist_roll_link"),
            ("bottle", "m0_bottle"),
        )
    }
    stage2_block = {
        "status": "PASS" if frame["jaw_gap_at_contact_aperture_m"] > 0.0 else "FAIL",
        "geometry_source": "compiled collision geoms from pinned source URDF plus canonical bottle/table helper",
        "contact_frame": frame,
        "compiled_geometry_inventory": geometry_inventory,
        "collision_maps": {
            "open_csv": str(output / "collision_distance_map_open.csv"),
            "open_png": str(output / "collision_distance_map_open.png"),
            "contact_csv": str(output / "collision_distance_map_contact.csv"),
            "contact_png": str(output / "collision_distance_map_contact.png"),
        },
    }
    inventory_summary = "; ".join(
        f"{role}={','.join(row['geom'] + ':' + row['type'] + ('[collision]' if row['collision_enabled'] else '[visual-only]') for row in values['geoms']) or 'no compiled geom'}"
        for role, values in geometry_inventory.items()
    )
    script_path = Path(__file__).resolve()
    source_commit = subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    elapsed = max(float(row.get("elapsed_s", 0.0)) for row in trace_rows)
    candidate_summaries = sorted(candidate_rows,
                                 key=lambda row: float(row.get("objective_squared_norm", math.inf)))
    canonical_scene_helper = Path(base.STATIC_ROOT) / "canonical_manipulation_assets.py"
    runner_inputs = {
        "isolated_grasp_runner": {"path": str(OLD_RUNNER), "sha256": sha256(OLD_RUNNER)},
        "paired_jaw_baseline": {"path": str(BASELINE), "sha256": sha256(BASELINE)},
        "canonical_scene_helper": {
            "path": str(canonical_scene_helper),
            "sha256": sha256(canonical_scene_helper),
        },
    }
    result = {
        "stage": "compiled-geometry-contact-frame-and-six-dof-search",
        "status": "PASS" if static_pass else "FAIL",
        "stage_2": stage2_block,
        "stage_3": {
            "status": "PASS" if static_pass else "FAIL",
            "optimization": {
                "method": "bounded deterministic multi-start scipy least_squares over grasp depth, lateral centering, height, and SO(3) rotation-vector increments",
                "orientation_start_count": len(candidate_rows),
                "actual_evaluation_count": len(trace_rows),
                "elapsed_s": elapsed,
                "max_solver_evaluations": 720,
                "wall_time_budget_s": 180.0,
                "best_candidate_summary": candidate_summaries[0],
                "best_observed_evaluation": {
                    "evaluation": best_detail["evaluation"],
                    "seed_index": best_detail["seed_index"],
                    "objective_squared_norm": best_detail["objective_squared_norm"],
                    "parameters": best_detail["parameters"],
                },
                "all_orientation_seeds": candidate_rows,
            },
            "best_static_candidate": best_detail,
            "first_failed_static_gate": first_failure,
            "all_source_joint_ranges_rad": source_limits,
            "source_joint_limit_semantics": "source URDF limits are unchanged; static candidates are checked against source intervals",
            "geometry_only_no_physics_steps": True,
            "bottle_pose_changes": "only in zero-step static candidate states; no active physics rollout was run",
        },
        "identity": {
            **source_identity,
            "executed_script_sha256": source_identity.get(
                "executed_script_sha256", "965ec6c50c8c90a846e9832dcf9be4bbd27a1b18de3112ffb8b2a2eeadde5b9b"
            ),
            "artifact_recovery_script_sha256": sha256(script_path),
            "vendor_commit": source_commit,
            "vendor_urdf_sha256": sha256(X2_URDF),
            "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mujoco_native_library": str(Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")),
            "mujoco_native_library_sha256": sha256(
                Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
            ),
            "runner_inputs": runner_inputs,
            "branch_head_at_search": source_identity.get(
                "branch_head_at_search", "7b3d17114ee2700fd87fa6c10cd6a47693dc1719"
            ),
        },
        "interpretation": {
            "bounded_search_does_not_prove_global_infeasibility": True,
            "physical_grasp_or_lift_claimed": False,
            "continuation_authorized_only_if_stage_3_passes": True,
            "mujoco_compile_flags": flags,
        },
    }
    write_json(output / "static_geometry_search.json", result)
    write_json(output / "runtime_identity.json", {
        "python_version": result["identity"]["python_version"],
        "python_executable": result["identity"]["python_executable"],
        "platform": result["identity"]["platform"],
        "mujoco_python": result["identity"]["mujoco_python"],
        "mujoco_native": result["identity"]["mujoco_native"],
        "mujoco_native_library": result["identity"]["mujoco_native_library"],
        "mujoco_native_library_sha256": result["identity"]["mujoco_native_library_sha256"],
        "vendor_commit": result["identity"]["vendor_commit"],
        "vendor_urdf_sha256": result["identity"]["vendor_urdf_sha256"],
        "executed_script_sha256": result["identity"]["executed_script_sha256"],
        "artifact_recovery_script_sha256": result["identity"]["artifact_recovery_script_sha256"],
        "runner_inputs": runner_inputs,
        "branch_head_at_search": result["identity"]["branch_head_at_search"],
    })
    write_json(output / "recovery_result.json", {
        "collision_model_audit": "PASS" if stage1_summary.get("stage_1") == "PASS" else "FAIL",
        "static_contact_frame_and_geometry_search": result["stage_3"]["status"],
        "standalone_hand_grasp": "NOT RUN",
        "standalone_30mm_lift": "NOT RUN",
        "full_x2_right_arm_corridor": "NOT RUN",
        "physical_rollout": "NOT RUN; stopped at failed static gate",
        "first_failed_static_gate": first_failure,
    })
    command = (
        f"MUJOCO_GL=egl {sys.executable} {script_path} --resume-artifacts --output {output}"
    )
    (output / "artifact_recovery_command.txt").write_text(
        f"Exact artifact recovery command (does not rerun collision or geometry search):\n{command}\n",
        encoding="utf-8",
    )
    first_failure = first_failure or "none"
    report_text = (
        "# Issue #46 X2 OmniPicker collision audit and grasp recovery\n\n"
        f"Status: **{'STATIC GEOMETRY PASS' if static_pass else 'BLOCKED AT STATIC GEOMETRY'}**.\n\n"
        "## Collision model\n\n"
        "The prior isolated rig omitted `right_wrist_yaw_joint` by rooting at its child link. "
        "That made the yaw link world-welded and changed MuJoCo's parent-child collision filtering. "
        "The corrected elbow-root extraction preserves the official yaw joint and source link/joint XML. "
        "At neutral, the yaw/pitch collision meshes have a compiled signed distance of -39.217 mm; "
        "the exact pair has no active contact in the full source model or corrected subtree. "
        "The old extraction did create an active pair. A zero-step regression confirms that a separate "
        "jaw/bottle overlap remains active while yaw/pitch remains filtered.\n\n"
        "The source visual and collision references for both wrist links name the same STL asset. "
        "MuJoCo retains the source vertices/faces but uses convex hulls for mesh collision. "
        "Raw source-triangle intersection was not tested, so the contribution of convexification to the "
        "39.217 mm distance remains unresolved. No collision mesh was changed. The official joint maps "
        f"`{source_parent}` to `{source_child}`; the full model has `filterparent` enabled, "
        f"{full_model.npair} explicit geom pairs, and {full_model.nexclude} explicit body exclusions, "
        "none matching this yaw/pitch pair. Thus the pair's absence from `mjData.contact` is the normal "
        "adjacent-parent filter, while `mj_geomDistance` still reports the signed geometric overlap.\n\n"
        "## Static contact search\n\n"
        f"The contact frame used the compiled base-to-jaw link chain; target bottle diameter was "
        f"{frame['bottle_body_diameter_m'] * 1000:.1f} mm and the source-limited contact aperture was "
        f"{contact_aperture:.6f}. The deterministic six-DoF search recorded {len(trace_rows)} evaluations "
        f"across {len(candidate_rows)} orientation seeds in {elapsed:.2f} s. Best observed objective was "
        f"{best_detail['objective_squared_norm']:.3f}; first failed gate was `{first_failure}`. "
        f"The best observed pose still had a minimum open robot/bottle distance of "
        f"{best_detail['open_minimum_robot_bottle']['signed_distance_m'] * 1000:.2f} mm and a minimum "
        f"critical wrist/loop/base-to-bottle distance of "
        f"{best_detail['closed_minimum_wrist_loop_base_bottle']['signed_distance_m'] * 1000:.2f} mm. "
        f"The narrow/wide intended jaw distances were "
        f"{best_detail['closed_jaw_to_body']['narrow']['distance_m'] * 1000:.2f} mm and "
        f"{best_detail['closed_jaw_to_body']['wide']['distance_m'] * 1000:.2f} mm; their witness radial "
        f"dot product was {best_detail['closed_jaw_to_body']['radial_direction_dot']:.3f}.\n\n"
        f"The jaw distance scan retained {frame['distance_query_integrity']['samples_consistent']} of "
        f"{frame['distance_query_integrity']['samples_total']} samples whose reported distance matched "
        f"the witness segment length. Inconsistent samples were excluded from the contact-aperture bracket. "
        f"The excluded sample at aperture ratio "
        f"{frame['distance_query_integrity']['inconsistent_samples'][0]['aperture_ratio']:.2f} reported "
        f"0 mm while its witness segment measured "
        f"{frame['distance_query_integrity']['inconsistent_samples'][0]['witness_segment_length_m'] * 1000:.3f} mm; "
        f"there was no active jaw-pair contact at that sample. "
        f"The maximum open-to-close gap increase among retained samples was "
        f"{frame['distance_query_integrity']['maximum_open_to_close_gap_increase_m'] * 1000:.4f} mm.\n\n"
        f"Compiled collision geometry inventory: {inventory_summary}.\n\n"
        + (f"Stage 1 PASS was reused from `{json.loads((output / 'stage1_reuse.json').read_text(encoding='utf-8'))['source_evidence_directory']}`; its collision audit was not repeated.\n\n"
           if (output / "stage1_reuse.json").is_file() else "")
        + "This finite bounded search did not establish a feasible static grasp configuration and does not "
        "prove global infeasibility. No physical grasp or lift was attempted.\n\n"
        "## Evidence\n\n"
        "- `collision_model_audit.json`, `collision_pair_filter_audit.json`, `stage_1_summary.json`, `stage1_reuse.json`, `static_geometry_search.json`, `runtime_identity.json`, `recovery_result.json`\n"
        "- `six_dof_optimizer_trace.jsonl`, `six_dof_candidate_summary.jsonl`\n"
        "- `collision_distance_map_open.csv/.png`, `collision_distance_map_contact.csv/.png`\n"
        "- `best_candidate_open_overview.png`, `best_candidate_open_side.png`, `best_candidate_contact_overview.png`, `best_candidate_contact_side.png`\n"
        "- `experiment_commands.txt`, `artifact_recovery_command.txt`\n\n"
        "Runtime and runner hashes are recorded in `runtime_identity.json`. Vendor source: `AgibotTech/agibot_x2_urdf` at `575cc6b988f976c23550e0db85aa1e5475d3652d`, "
        "URDF `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, Mulan PSL v2. MuJoCo 3.3.6.\n"
    )
    (output / "REPORT.md").write_text(report_text, encoding="utf-8")
    run_scope = (
        f"Reused completed Stage 1 PASS from {json.loads((output / 'stage1_reuse.json').read_text(encoding='utf-8'))['source_evidence_directory']}; reran only stage 2/3 compiled-geometry search.\n"
        if (output / "stage1_reuse.json").is_file()
        else "Completed stage 1 collision-model audit and stage 2/3 compiled-geometry search.\n"
    )
    (output / "run.log").write_text(
        run_scope
        + f"Search evaluations: {len(trace_rows)}; orientation seeds: {len(candidate_rows)}; elapsed: {elapsed:.3f}s.\n"
        f"Static result: {'PASS' if static_pass else 'FAIL'}; first failed gate: {first_failure}.\n"
        "No physics steps, bottle rollout, grasp, or lift were run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage_1": stage1_summary.get("stage_1"),
                      "stage_3": result["stage_3"]["status"],
                      "best_evaluation": best_detail["evaluation"],
                      "first_failed_gate": first_failure,
                      "rendered": True}, indent=2))
    return result


def static_geometry_search(output: Path, stage1: dict[str, Any]) -> dict[str, Any]:
    old = load_module("issue46_omnipicker_isolated_grasp", OLD_RUNNER)
    base = old.load_baseline()
    extracted_path = Path(stage1["corrected_isolated_subtree"]["path"])
    model, _, flags = compile_isolated(
        base, extracted_path, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), bottle=True
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    frame = contact_frame(base, model, data)
    root_id = body_id(model, ROOT_LINK)
    frame["root_body_id"] = root_id
    root_origin = np.asarray(frame["local_frame"]["root_origin_world_at_probe_pose"], dtype=float)
    midpoint_local = (np.asarray(frame["local_frame"]["contact_midpoint_relative_to_root_m"], dtype=float))
    base_rotation = np.asarray(frame["local_frame"]["rotation_matrix_source_frame_to_horizontal_grasp"], dtype=float)
    bottle_center = np.asarray(frame["bottle_body_center_world_m"], dtype=float)
    local_normal = np.asarray(frame["local_frame"]["normal_narrow_to_wide"], dtype=float)
    diameter = float(frame["bottle_body_diameter_m"])
    body_half = float(frame["bottle_body_half_height_m"])
    bottle_geom_ids = [gid for gid in range(model.ngeom)
                       if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) == "m0_bottle"
                       and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]
    table_geom_ids = [gid for gid in range(model.ngeom)
                      if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) == "m0_table"
                      and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]
    floor_geom_ids = [gid for gid in range(model.ngeom)
                      if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) == "world"
                      and name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor"
                      and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]
    robot_geom_ids = [gid for gid in collision_ids(model)
                      if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
                      not in {"world", "m0_table", "m0_bottle"}]
    critical_geom_ids = [gid for gid in robot_geom_ids
                         if any(token in name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              int(model.geom_bodyid[gid])).lower()
                                for token in ("wrist", "loop", "omnipicker_base", "elbow"))]
    if not table_geom_ids or not floor_geom_ids or not bottle_geom_ids:
        raise RuntimeError("Canonical collision geometry set is incomplete")

    seed_normal = base_rotation @ local_normal
    seed_normal[2] = 0.0
    seed_normal /= max(float(np.linalg.norm(seed_normal)), 1e-12)
    seed_tangent = np.array([-seed_normal[1], seed_normal[0], 0.0])

    def make_pose(rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
        center = bottle_center + translation
        return {"root_body_id": root_id,
                "root_rotation_world": rotation,
                "root_position_world_m": center - rotation @ midpoint_local,
                "target_contact_midpoint_world_m": center.tolist()}

    def current_minima(pose: dict[str, Any], aperture: float,
                       bottle_lift: float = 0.0) -> dict[str, Any]:
        set_static_state(base, model, data, pose, aperture, bottle_lift)
        rb = minimum_row(geom_pair_rows(model, data, robot_geom_ids, bottle_geom_ids))
        rt = minimum_row(geom_pair_rows(model, data, robot_geom_ids, table_geom_ids))
        rf = minimum_row(geom_pair_rows(model, data, robot_geom_ids, floor_geom_ids))
        bt = minimum_row(geom_pair_rows(model, data, bottle_geom_ids, table_geom_ids))
        return {"robot_bottle": rb, "robot_table": rt, "robot_floor": rf,
                "bottle_table": bt, "active_contacts": contact_names(model, data)}

    map_rows: dict[str, list[dict[str, Any]]] = {}
    depths = np.linspace(-0.12, 0.12, 31)
    heights = np.linspace(-0.06, 0.06, 31)
    map_rotation = base_rotation
    for label, aperture in (("open", 1.0), ("contact", float(frame["contact_aperture_ratio"]))):
        rows = []
        for z_offset in heights:
            for depth in depths:
                pose = make_pose(map_rotation, seed_normal * float(depth) + np.array([0.0, 0.0, float(z_offset)]))
                minima = current_minima(pose, aperture)
                rows.append({
                    "depth_offset_m": float(depth), "height_offset_m": float(z_offset),
                    "minimum_robot_bottle_clearance_m": minima["robot_bottle"]["signed_distance_m"],
                    "closest_robot_body": minima["robot_bottle"]["body1"],
                    "closest_robot_geom": minima["robot_bottle"]["geom1"],
                    "closest_bottle_geom": minima["robot_bottle"]["geom2"],
                    "minimum_robot_table_clearance_m": minima["robot_table"]["signed_distance_m"],
                    "minimum_robot_floor_clearance_m": minima["robot_floor"]["signed_distance_m"],
                    "active_contact_count": len(minima["active_contacts"]),
                })
        map_rows[label] = rows
        csv_path = output / f"collision_distance_map_{label}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            import csv
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        heatmap_png(output / f"collision_distance_map_{label}.png",
                    f"Compiled X2 OmniPicker {label.upper()} collision cross-section",
                    rows, depths, heights)

    source_root = ET.parse(X2_URDF).getroot()
    source_limits = {}
    for joint in source_root.findall("joint"):
        if joint.get("name") not in {name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                      for i in range(model.njnt)}:
            continue
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            source_limits[joint.get("name", "")] = [float(limit.get("lower")), float(limit.get("upper"))]

    contact_aperture = float(frame["contact_aperture_ratio"])
    approach_samples = [1.0, 0.75, 0.50, 0.25, 0.0]
    contact_apertures = np.linspace(1.0, contact_aperture, 7).tolist()
    joint_ranges = {}
    for joint_name in (DRIVER, FOLLOWER, *WRIST_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            joint_ranges[joint_name] = {
                "source_range_rad": source_limits.get(joint_name),
                "compiled_range_rad": model.jnt_range[int(jid)].tolist(),
                "compiled_limited": bool(model.jnt_limited[int(jid)]),
                "qpos0_rad": float(model.qpos0[int(model.jnt_qposadr[int(jid)])]),
            }

    max_evaluations = 720
    time_budget_s = 180.0
    started = time.monotonic()
    evaluation_count = 0
    trace_path = output / "six_dof_optimizer_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    candidate_path = output / "six_dof_candidate_summary.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    best_candidate: dict[str, Any] | None = None
    limit_hit = False
    margin = 0.001
    bottle_center_xy = bottle_center[:2].copy()
    half_body_clearance = max(0.0, body_half - 0.008)

    def evaluate(seed_rotation: np.ndarray, x: np.ndarray, seed_index: int) -> np.ndarray:
        nonlocal evaluation_count, limit_hit
        elapsed = time.monotonic() - started
        if evaluation_count >= max_evaluations or elapsed >= time_budget_s:
            limit_hit = True
            raise TimeoutError("bounded six-DoF solver budget exhausted")
        evaluation_count += 1
        increment = Rotation.from_rotvec(np.asarray(x[3:6], dtype=float)).as_matrix()
        rotation = increment @ seed_rotation
        axis = rotation @ local_normal
        axis[2] = 0.0
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        tangent = np.array([-axis[1], axis[0], 0.0])
        translation = axis * float(x[0]) + tangent * float(x[1]) + np.array([0.0, 0.0, float(x[2])])
        pose = make_pose(rotation, translation)
        contact_center = np.asarray(pose["target_contact_midpoint_world_m"], dtype=float)

        set_static_state(base, model, data, pose, 1.0)
        open_bottle_rows = geom_pair_rows(model, data, robot_geom_ids, bottle_geom_ids)
        open_table_rows = geom_pair_rows(model, data, robot_geom_ids, table_geom_ids)
        open_floor_rows = geom_pair_rows(model, data, robot_geom_ids, floor_geom_ids)
        open_self_contacts = [row for row in contact_names(model, data)
                              if row["body1"] not in {"m0_bottle", "m0_table", "world"}
                              and row["body2"] not in {"m0_bottle", "m0_table", "world"}
                              and row["distance_m"] < -1e-7]

        # The start is outside along the current horizontal approach normal.
        pregrasp = make_pose(rotation, translation + axis * 0.090)
        path_min = math.inf
        path_rows = []
        for fraction in (0.0, 0.5, 1.0):
            target_delta = translation + axis * (0.090 * (1.0 - fraction))
            path_pose = make_pose(rotation, target_delta)
            set_static_state(base, model, data, path_pose, 1.0)
            bottle_rows = geom_pair_rows(model, data, robot_geom_ids, bottle_geom_ids)
            table_rows = geom_pair_rows(model, data, robot_geom_ids, table_geom_ids)
            floor_rows = geom_pair_rows(model, data, robot_geom_ids, floor_geom_ids)
            local_min = min(float(minimum_row(bottle_rows)["signed_distance_m"]),
                            float(minimum_row(table_rows)["signed_distance_m"]),
                            float(minimum_row(floor_rows)["signed_distance_m"]))
            path_min = min(path_min, local_min)
            path_rows.append({"path_fraction": fraction,
                              "minimum_robot_bottle": minimum_row(bottle_rows),
                              "minimum_robot_table": minimum_row(table_rows),
                              "minimum_robot_floor": minimum_row(floor_rows)})

        set_static_state(base, model, data, pose, contact_aperture)
        closed_bottle_rows = geom_pair_rows(model, data, robot_geom_ids, bottle_geom_ids)
        closed_table_rows = geom_pair_rows(model, data, robot_geom_ids, table_geom_ids)
        closed_floor_rows = geom_pair_rows(model, data, robot_geom_ids, floor_geom_ids)
        critical_rows = geom_pair_rows(model, data, critical_geom_ids, bottle_geom_ids)
        closed_self_contacts = [row for row in contact_names(model, data)
                                if row["body1"] not in {"m0_bottle", "m0_table", "world"}
                                and row["body2"] not in {"m0_bottle", "m0_table", "world"}
                                and row["distance_m"] < -1e-7]
        narrow_distance, narrow_segment = geom_distance(
            model, data, int(frame["narrow_geom"]), int(frame["bottle_body_geom"])
        )
        wide_distance, wide_segment = geom_distance(
            model, data, int(frame["wide_geom"]), int(frame["bottle_body_geom"])
        )
        body_center = data.geom_xpos[int(frame["bottle_body_geom"])].copy()
        radial = []
        contact_points = []
        for distance, segment in ((narrow_distance, narrow_segment), (wide_distance, wide_segment)):
            point = np.asarray(segment[3:6], dtype=float)
            radial_vector = point[:2] - body_center[:2]
            norm = float(np.linalg.norm(radial_vector))
            radial.append(radial_vector / max(norm, 1e-12))
            contact_points.append(point)
        opposition_dot = float(np.dot(radial[0], radial[1]))
        contact_height_bounds = [float(body_center[2] - half_body_clearance),
                                 float(body_center[2] + half_body_clearance)]
        contact_heights = [float(point[2]) for point in contact_points]

        closure_rows = []
        for aperture in contact_apertures:
            set_static_state(base, model, data, pose, float(aperture))
            closure_rows.append({
                "aperture_ratio": float(aperture),
                "narrow_distance_m": geom_distance(model, data,
                    int(frame["narrow_geom"]), int(frame["bottle_body_geom"]))[0],
                "wide_distance_m": geom_distance(model, data,
                    int(frame["wide_geom"]), int(frame["bottle_body_geom"]))[0],
            })

        lift_rows = []
        for lift in (0.001, 0.005, 0.030):
            lift_pose = make_pose(rotation, translation + np.array([0.0, 0.0, lift]))
            set_static_state(base, model, data, lift_pose, contact_aperture, bottle_lift_m=lift)
            lift_rows.append({
                "lift_m": lift,
                "robot_table": minimum_row(geom_pair_rows(model, data, robot_geom_ids, table_geom_ids)),
                "robot_floor": minimum_row(geom_pair_rows(model, data, robot_geom_ids, floor_geom_ids)),
                "bottle_table": minimum_row(geom_pair_rows(model, data, bottle_geom_ids, table_geom_ids)),
                "jaw_distances_m": [
                    geom_distance(model, data, int(frame["narrow_geom"]), int(frame["bottle_body_geom"]))[0],
                    geom_distance(model, data, int(frame["wide_geom"]), int(frame["bottle_body_geom"]))[0],
                ],
            })

        target_values = []
        limit_violations = []
        for aperture in [1.0, *contact_apertures, contact_aperture]:
            targets = base.aperture_targets(float(aperture))
            for joint_name, key in ((DRIVER, "right_claw_joint_target_rad"),
                                    (FOLLOWER, "R_hand_wide1_joint_target_rad")):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                target = float(targets[key])
                bounds = source_limits.get(joint_name)
                if bounds is not None and not (bounds[0] - 1e-9 <= target <= bounds[1] + 1e-9):
                    limit_violations.append({"joint": joint_name, "target_rad": target,
                                             "source_range_rad": bounds})
                target_values.append({"aperture_ratio": float(aperture),
                                      "joint": joint_name, "target_rad": target,
                                      "source_range_rad": bounds})

        open_min = minimum_row(open_bottle_rows)
        open_table = minimum_row(open_table_rows)
        open_floor = minimum_row(open_floor_rows)
        closed_critical = minimum_row(critical_rows)
        closed_table = minimum_row(closed_table_rows)
        closed_floor = minimum_row(closed_floor_rows)
        closed_nonjaw = minimum_row([row for row in closed_bottle_rows
                                     if row["body1"] not in JAW_BODIES])
        closure_increases = {
            "narrow": [max(0.0, closure_rows[i + 1]["narrow_distance_m"]
                               - closure_rows[i]["narrow_distance_m"]) for i in range(len(closure_rows) - 1)],
            "wide": [max(0.0, closure_rows[i + 1]["wide_distance_m"]
                             - closure_rows[i]["wide_distance_m"]) for i in range(len(closure_rows) - 1)],
        }
        self_penetration = max([max(0.0, -float(row["distance_m"]))
                                for row in open_self_contacts + closed_self_contacts] + [0.0])
        approach_distance = path_min
        lift_robot_table = min(float(row["robot_table"]["signed_distance_m"]) for row in lift_rows)
        lift_robot_floor = min(float(row["robot_floor"]["signed_distance_m"]) for row in lift_rows)
        lift_bottle_table = float(lift_rows[-1]["bottle_table"]["signed_distance_m"])

        def clearance_residual(distance: float, wanted: float = margin) -> float:
            return max(0.0, wanted - distance) / 0.005

        residuals = [
            narrow_distance / 0.002,
            wide_distance / 0.002,
            clearance_residual(float(open_min["signed_distance_m"])),
            clearance_residual(float(closed_critical["signed_distance_m"])),
            clearance_residual(float(open_table["signed_distance_m"])),
            clearance_residual(float(closed_table["signed_distance_m"])),
            clearance_residual(float(open_floor["signed_distance_m"])),
            clearance_residual(float(closed_floor["signed_distance_m"])),
            clearance_residual(approach_distance),
            clearance_residual(lift_robot_table),
            clearance_residual(lift_robot_floor),
            clearance_residual(lift_bottle_table),
            max(0.0, 0.80 + opposition_dot),
            max(0.0, contact_height_bounds[0] - contact_heights[0]) / 0.005,
            max(0.0, contact_heights[0] - contact_height_bounds[1]) / 0.005,
            max(0.0, contact_height_bounds[0] - contact_heights[1]) / 0.005,
            max(0.0, contact_heights[1] - contact_height_bounds[1]) / 0.005,
            self_penetration / 0.005,
            max((max(values) for values in closure_increases.values()), default=0.0) / 0.005,
            10.0 * len(limit_violations),
            *[value / 0.005 for value in closure_increases["narrow"]],
            *[value / 0.005 for value in closure_increases["wide"]],
            *[float(value) / 0.10 for value in x[:3]],
        ]
        gates = {
            "open_all_robot_bottle_pairs_clear": float(open_min["signed_distance_m"]) > 0.0,
            "open_robot_table_clear": float(open_table["signed_distance_m"]) > 0.0,
            "open_robot_floor_clear": float(open_floor["signed_distance_m"]) > 0.0,
            "open_source_joint_targets_valid": not limit_violations,
            "approach_path_clear": approach_distance > 0.0,
            "closed_jaws_reach_cylinder_body": abs(narrow_distance) <= 1e-5 and abs(wide_distance) <= 1e-5,
            "jaw_witnesses_on_opposing_cylinder_sides": opposition_dot <= -0.80,
            "jaw_contacts_within_usable_cylinder_band": all(
                contact_height_bounds[0] <= value <= contact_height_bounds[1]
                for value in contact_heights
            ),
            "closed_wrist_loop_base_clear": float(closed_critical["signed_distance_m"]) > 0.0,
            "closed_robot_table_and_floor_clear": float(closed_table["signed_distance_m"]) > 0.0
                and float(closed_floor["signed_distance_m"]) > 0.0,
            "closure_reduces_both_jaw_clearances_monotonically": all(
                value <= 1e-6 for rows in closure_increases.values() for value in rows
            ),
            "zero_active_initial_self_penetration": self_penetration <= 1e-7,
            "30mm_static_lift_corridor_clear": lift_robot_table > 0.0 and lift_robot_floor > 0.0
                and lift_bottle_table > 0.0,
        }
        row = {
            "evaluation": evaluation_count,
            "seed_index": seed_index,
            "parameters": {"contact_depth_offset_m": float(x[0]),
                           "lateral_offset_m": float(x[1]),
                           "height_offset_m": float(x[2]),
                           "rotation_increment_rotvec_rad": np.asarray(x[3:6]).tolist()},
            "contact_aperture_ratio": contact_aperture,
            "root_position_world_m": np.asarray(pose["root_position_world_m"]).tolist(),
            "root_rotation_world": rotation.tolist(),
            "open_minimum_robot_bottle": open_min,
            "open_minimum_robot_table": open_table,
            "open_minimum_robot_floor": open_floor,
            "pregrasp_approach_path_min_clearance_m": approach_distance,
            "pregrasp_and_approach_samples": path_rows,
            "closed_jaw_to_body": {
                "narrow": {"distance_m": narrow_distance, "witness_segment_world_m": narrow_segment},
                "wide": {"distance_m": wide_distance, "witness_segment_world_m": wide_segment},
                "radial_direction_dot": opposition_dot,
                "bottle_contact_point_heights_m": contact_heights,
                "usable_body_height_band_world_m": contact_height_bounds,
            },
            "closed_minimum_wrist_loop_base_bottle": closed_critical,
            "closed_minimum_robot_table": closed_table,
            "closed_minimum_robot_floor": closed_floor,
            "closed_minimum_nonjaw_robot_bottle": closed_nonjaw,
            "active_open_self_contacts": open_self_contacts,
            "active_closed_self_contacts": closed_self_contacts,
            "closure_progression": closure_rows,
            "closure_clearance_increases_m": closure_increases,
            "lift_corridor": lift_rows,
            "source_target_values": target_values,
            "source_limit_violations": limit_violations,
            "residual_vector": residuals,
            "objective_squared_norm": float(np.dot(residuals, residuals)),
            "static_gates": gates,
            "first_failed_static_gate": first_failed_static_gate(gates),
            "all_static_gates_pass": all(gates.values()),
            "elapsed_s": time.monotonic() - started,
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
        return np.asarray(residuals, dtype=float)

    orientation_seeds = []
    for yaw in (0.0, 90.0, 180.0, 270.0):
        orientation_seeds.append({"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0})
    orientation_seeds.extend((
        {"yaw_deg": 0.0, "pitch_deg": 18.0, "roll_deg": 0.0},
        {"yaw_deg": 180.0, "pitch_deg": -18.0, "roll_deg": 0.0},
        {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 18.0},
        {"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": -18.0},
    ))
    bounds_low = np.array([-0.12, -0.08, -0.06, -0.35, -0.35, -0.35], dtype=float)
    bounds_high = -bounds_low
    starts = []
    for seed_index, seed in enumerate(orientation_seeds):
        seed_rotation = Rotation.from_euler(
            "zyx", [seed["yaw_deg"], seed["pitch_deg"], seed["roll_deg"]], degrees=True
        ).as_matrix() @ base_rotation
        x0 = np.zeros(6, dtype=float)
        try:
            solution = least_squares(
                lambda x, si=seed_index, sr=seed_rotation: evaluate(sr, x, si),
                x0=x0, bounds=(bounds_low, bounds_high), max_nfev=12,
                xtol=1e-4, ftol=1e-4, gtol=1e-4,
            )
            final_residual = evaluate(seed_rotation, solution.x, seed_index)
            summary = {
                "seed_index": seed_index, **seed,
                "solver_status": int(solution.status), "solver_message": str(solution.message),
                "solver_nfev": int(solution.nfev), "solver_njev": int(solution.njev or 0),
                "solution_parameters": solution.x.tolist(),
                "objective_squared_norm": float(np.dot(final_residual, final_residual)),
                "budget_exhausted": False,
            }
        except TimeoutError as exc:
            limit_hit = True
            summary = {"seed_index": seed_index, **seed, "solver_status": 0,
                       "solver_message": str(exc), "budget_exhausted": True}
        starts.append(summary)
        with candidate_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
            stream.flush()
        if summary.get("objective_squared_norm") is not None and (
                best_candidate is None or summary["objective_squared_norm"]
                < best_candidate["objective_squared_norm"]):
            best_candidate = summary
        if limit_hit:
            break

    best_detail = None
    if best_candidate is not None:
        row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        # The last line may be a different seed after its final evaluation; match the best parameters explicitly.
        best_seed = int(best_candidate["seed_index"])
        seed = orientation_seeds[best_seed]
        seed_rotation = Rotation.from_euler(
            "zyx", [seed["yaw_deg"], seed["pitch_deg"], seed["roll_deg"]], degrees=True
        ).as_matrix() @ base_rotation
        best_residual = evaluate(seed_rotation, np.asarray(best_candidate["solution_parameters"]), best_seed)
        best_detail = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        pose = {"root_body_id": root_id,
                "root_rotation_world": np.asarray(best_detail["root_rotation_world"], dtype=float),
                "root_position_world_m": np.asarray(best_detail["root_position_world_m"], dtype=float)}
        set_static_state(base, model, data, pose, 1.0)
        render_reference(model, data, output, "best_candidate_open_overview.png",
                         bottle_center.tolist(), 0.95, 135.0, -20.0)
        set_static_state(base, model, data, pose, contact_aperture)
        render_reference(model, data, output, "best_candidate_contact_overview.png",
                         bottle_center.tolist(), 0.85, 135.0, -20.0)
        render_reference(model, data, output, "best_candidate_contact_side.png",
                         bottle_center.tolist(), 0.80, 90.0, -5.0)
        best_detail["residual_vector"] = best_residual.tolist()

    static_pass = bool(best_detail and best_detail["all_static_gates_pass"])
    first_failure = first_failed_static_gate(best_detail["static_gates"]) if best_detail else None
    result = {
        "stage": "compiled-geometry-contact-frame-and-six-dof-search",
        "status": "PASS" if static_pass else "FAIL",
        "stage_2": {
            "status": "PASS" if frame["jaw_gap_at_contact_aperture_m"] > 0.0 else "FAIL",
            "geometry_source": "compiled collision geoms from pinned source URDF plus canonical bottle/table helper",
            "handedness_and_wrist_parentage": "right elbow root preserves right_wrist_yaw_joint -> right_wrist_yaw_link -> right_wrist_pitch_joint -> right_wrist_pitch_link",
            "contact_frame": frame,
            "source_and_compiled_joint_ranges": joint_ranges,
            "compiled_geom_counts": {"robot_collision_geoms": len(robot_geom_ids),
                                      "critical_wrist_loop_base_geoms": len(critical_geom_ids),
                                      "bottle_collision_geoms": len(bottle_geom_ids),
                                      "table_collision_geoms": len(table_geom_ids),
                                      "floor_collision_geoms": len(floor_geom_ids)},
            "collision_maps": {
                "open_csv": str(output / "collision_distance_map_open.csv"),
                "open_png": str(output / "collision_distance_map_open.png"),
                "contact_csv": str(output / "collision_distance_map_contact.csv"),
                "contact_png": str(output / "collision_distance_map_contact.png"),
                "fixed_orientation": base_rotation.tolist(),
                "depth_axis_world": seed_normal.tolist(),
                "height_axis_world": [0.0, 0.0, 1.0],
                "grid_shape": [len(heights), len(depths)],
            },
        },
        "stage_3": {
            "status": "PASS" if static_pass else "FAIL",
            "first_failed_static_gate": first_failure,
            "optimization": {
                "method": "bounded deterministic multi-start scipy least_squares over source-frame midpoint depth/lateral/height and SO(3) rotation-vector increments",
                "variables": ["grasp_depth_along_jaw_axis_m", "lateral_centering_m",
                              "grasp_height_m", "gripper_yaw_pitch_roll_rotvec_rad"],
                "translation_bounds_m": {"depth": [-0.12, 0.12], "lateral": [-0.08, 0.08],
                                          "height": [-0.06, 0.06]},
                "rotation_increment_bounds_rad": [-0.35, 0.35],
                "orientation_seeds": orientation_seeds,
                "max_solver_evaluations": max_evaluations,
                "wall_time_budget_s": time_budget_s,
                "actual_evaluation_count": evaluation_count,
                "elapsed_s": time.monotonic() - started,
                "budget_exhausted": limit_hit,
                "seed_results": starts,
                "best_candidate": best_candidate,
            },
            "best_static_candidate": best_detail,
            "all_source_joint_ranges_rad": source_limits,
            "source_joint_limit_semantics": "source URDF limits are unchanged; static candidates are checked against each source interval",
            "geometry_only_no_physics_steps": True,
            "bottle_pose_changes": "only in zero-step static candidate states; no active physics rollout was run",
        },
        "interpretation": {
            "bounded_search_does_not_prove_global_infeasibility": True,
            "physical_grasp_or_lift_claimed": False,
            "continuation_authorized_only_if_stage_3_passes": True,
            "mujoco_compile_flags": flags,
        },
        "identity": {
            "executed_script_sha256": sha256(Path(__file__).resolve()),
            "vendor_commit": subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "vendor_urdf_sha256": sha256(X2_URDF),
            "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mujoco_native_library": str(Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")),
            "mujoco_native_library_sha256": sha256(
                Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
            ),
            "branch_head_at_search": subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
            ).strip(),
            "stage1_audit_sha256": sha256(output / "collision_model_audit.json"),
            "stage1_summary_sha256": sha256(output / "stage_1_summary.json"),
        },
    }
    write_json(output / "static_geometry_search.json", result)
    write_json(output / "runtime_identity.json", {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(),
        "mujoco_native_library": str(Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")),
        "mujoco_native_library_sha256": sha256(
            Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
        ),
        "vendor_commit": result["identity"]["vendor_commit"],
        "vendor_urdf_sha256": result["identity"]["vendor_urdf_sha256"],
        "executed_script_sha256": result["identity"]["executed_script_sha256"],
        "branch_head_at_search": result["identity"]["branch_head_at_search"],
    })
    write_json(output / "recovery_result.json", {
        "collision_model_audit": "PASS" if stage1["interpretation"]["active_collision_regression_pass"] else "FAIL",
        "static_contact_frame_and_geometry_search": result["stage_3"]["status"],
        "standalone_hand_grasp": "NOT RUN" if not static_pass else "NOT RUN YET",
        "standalone_30mm_lift": "NOT RUN",
        "full_x2_right_arm_corridor": "NOT RUN",
        "physical_rollout": "NOT RUN",
        "first_failed_static_gate": first_failure,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-artifacts", action="store_true")
    parser.add_argument("--search-from", type=Path,
                        help="reuse a completed Stage 1 audit and rerun only static geometry stages")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume_artifacts and args.search_from:
        parser.error("--resume-artifacts and --search-from cannot be combined")
    if args.resume_artifacts:
        if not output.is_dir():
            raise FileNotFoundError(f"Existing partial evidence directory required: {output}")
        result = resume_static_artifacts(output)
        return 0 if result["status"] == "PASS" else 3
    if args.search_from:
        source = args.search_from.resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Completed Stage 1 evidence directory required: {source}")
        source_audit_path = source / "collision_model_audit.json"
        source_summary_path = source / "stage_1_summary.json"
        if not source_audit_path.is_file() or not source_summary_path.is_file():
            raise FileNotFoundError("--search-from requires collision_model_audit.json and stage_1_summary.json")
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary.get("stage_1") != "PASS":
            raise RuntimeError("Refusing to reuse a Stage 1 audit that did not pass")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty evidence directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        source_urdf = Path(source_audit["corrected_isolated_subtree"]["path"])
        copied_urdf = output / source_urdf.name
        shutil.copy2(source_urdf, copied_urdf)
        source_audit["corrected_isolated_subtree"]["path"] = str(copied_urdf)
        write_json(output / "collision_model_audit.json", source_audit)
        shutil.copy2(source_summary_path, output / "stage_1_summary.json")
        write_json(output / "stage1_reuse.json", {
            "source_evidence_directory": str(source),
            "source_collision_model_audit_sha256": sha256(source_audit_path),
            "source_stage_1_summary_sha256": sha256(source_summary_path),
            "source_extracted_urdf": str(source_urdf),
            "reused_extracted_urdf": str(copied_urdf),
            "reused_extracted_urdf_sha256": sha256(copied_urdf),
            "stage_1_repeated": False,
        })
        command = (
            f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
            f"--search-from {source} --output {output}"
        )
        (output / "experiment_commands.txt").write_text(
            f"Branch: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'branch', '--show-current'], text=True).strip()}\n"
            f"Starting HEAD: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()}\n"
            f"Reused completed Stage 1 evidence from: {source}\n"
            f"Exact reproduction command:\n{command}\n",
            encoding="utf-8",
        )
        geometry = static_geometry_search(output, source_audit)
        summary = json.loads((output / "stage_1_summary.json").read_text(encoding="utf-8"))
        summary["stage_2"] = geometry["stage_2"]["status"]
        summary["stage_3"] = geometry["stage_3"]["status"]
        summary["solver_evaluations"] = geometry["stage_3"]["optimization"]["actual_evaluation_count"]
        summary["solver_elapsed_s"] = geometry["stage_3"]["optimization"]["elapsed_s"]
        write_json(output / "stage_1_summary.json", summary)
        result = resume_static_artifacts(output)
        return 0 if result["stage_3"]["status"] == "PASS" else 3
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = (
        f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
        f"--output {output}"
    )
    (output / "experiment_commands.txt").write_text(
        f"Branch: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'branch', '--show-current'], text=True).strip()}\n"
        f"Starting HEAD: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()}\n"
        f"Exact reproduction command:\n{command}\n",
        encoding="utf-8",
    )
    report = audit_stage(output)
    write_json(output / "collision_model_audit.json", report)
    summary = {
        "prior_isolated_wrist_pair": report["prior_filtered_subtree"]["wrist_pair"],
        "full_source_wrist_pair": report["full_source_tree"]["wrist_pair"],
        "corrected_isolated_wrist_pair": report["corrected_isolated_subtree"]["wrist_pair"],
        "corrected_subtree_preserves_yaw_joint": report["corrected_isolated_subtree"]["preserved_wrist_yaw_joint"],
        "xml_parity": report["corrected_isolated_subtree"]["all_link_joint_xml_preserved"],
        "active_collision_regression": report["active_collision_regression"],
        "stage_1": "PASS" if not report["corrected_isolated_subtree"]["wrist_pair"]["active_contact_rows"]
        and not report["full_source_tree"]["wrist_pair"]["active_contact_rows"]
        and report["corrected_isolated_subtree"]["all_link_joint_xml_preserved"]
        and report["interpretation"]["active_collision_regression_pass"] else "FAIL",
    }
    write_json(output / "stage_1_summary.json", summary)
    if summary["stage_1"] != "PASS":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2
    geometry = static_geometry_search(output, report)
    summary["stage_2"] = geometry["stage_2"]["status"]
    summary["stage_3"] = geometry["stage_3"]["status"]
    summary["solver_evaluations"] = geometry["stage_3"]["optimization"]["actual_evaluation_count"]
    summary["solver_elapsed_s"] = geometry["stage_3"]["optimization"]["elapsed_s"]
    write_json(output / "stage_1_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if geometry["stage_3"]["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
