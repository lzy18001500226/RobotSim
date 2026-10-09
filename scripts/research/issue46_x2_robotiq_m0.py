#!/usr/bin/env python3
"""Isolated X2 + Robotiq 2F-85 SIMULATION_ONLY pick-and-place experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
MENAGERIE_PIN = "0059d4335f8156206f63a35662313385f7ad6d74"
CANONICAL_SHA = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"
SOURCE_CACHE = Path("/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources")
X2_DEFAULT = SOURCE_CACHE / "agibot_x2_urdf"
MENAGERIE_DEFAULT = SOURCE_CACHE / "mujoco_menagerie"
CANONICAL_DEFAULT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py")
OUT_DEFAULT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-simulation-only-m0-20261009")
X2_MJCF = Path("X2_URDF-v1.4.0/X2-Ultra.xml")
X2_URDF = Path("X2_URDF-v1.4.0/X2-Ultra.urdf")
X2_TOOL_URDF = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
MENAGERIE_MJCF = Path("robotiq_2f85/2f85.xml")
ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
ARM_ACTUATORS = {f"motor_{joint}" for joint in ARM}
TOOL_POS = np.array([0.00196, 0.00035, -0.062], dtype=float)
STATION_BASE_POS = np.array([0.000, 0.080, 0.680], dtype=float)
DT = 0.001  # Retain the official X2 MJCF timestep.
GRAVITY = 9.81
REQUIRED_LIFT_M = 0.050
LIFT_COMMAND_M = 0.055
PREGRASP_RETREAT_M = 0.080
PREFIX = "rq_"
REFERENCE_GRIPPER_MASS_KG = 1.0526083388427392


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("issue46_canonical_assets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical scene helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_checkout(root: Path, expected: str) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if head != expected or dirty:
        raise RuntimeError(f"Pinned source checkout mismatch: {root}, HEAD={head}, dirty={bool(dirty)}")
    return {"path": str(root), "head": head, "clean": True}


def obj_id(model: mujoco.MjModel, kind: mujoco.mjtObj, value: str) -> int:
    ident = int(mujoco.mj_name2id(model, kind, value))
    if ident < 0:
        raise RuntimeError(f"Missing MuJoCo {kind.name}: {value}")
    return ident


def obj_name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def qpos_id(model: mujoco.MjModel, joint_name: str) -> int:
    jid = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def dof_id(model: mujoco.MjModel, joint_name: str) -> int:
    jid = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[jid])


def robotiq_axes(menagerie_xml: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(menagerie_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")
    root_rotation = data.xmat[root_id].reshape(3, 3)
    root_position = data.xpos[root_id].copy()
    pads = [obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
            ("right_pad1", "right_pad2", "left_pad1", "left_pad2")]
    right = np.mean(data.geom_xpos[pads[:2]], axis=0)
    left = np.mean(data.geom_xpos[pads[2:]], axis=0)
    center = (right + left) / 2.0
    insertion = center - root_position
    insertion /= np.linalg.norm(insertion)
    opening = left - right
    opening /= np.linalg.norm(opening)
    opening -= insertion * float(np.dot(opening, insertion))
    opening /= np.linalg.norm(opening)
    # The imported root frame maps its insertion axis to robot-forward (+X),
    # and the opposing pad axis to world-lateral (+Y).
    local_basis = np.column_stack((np.cross(insertion, opening), insertion, opening))
    world_basis = np.column_stack((np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])))
    mount_rotation = world_basis @ local_basis.T
    mount_quat_xyzw = Rotation.from_matrix(mount_rotation).as_quat()
    mount_quat_wxyz = np.array([mount_quat_xyzw[3], *mount_quat_xyzw[:3]])
    center_local = root_rotation.T @ (center - root_position)
    pinch_id = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    source_pinch_local = root_rotation.T @ (data.site_xpos[pinch_id] - root_position)
    return {
        "mount_quat_wxyz": mount_quat_wxyz.tolist(),
        "mount_rotation_matrix": mount_rotation.tolist(),
        "tcp_pad_midpoint_local_m": center_local.tolist(),
        "source_pinch_site_local_m": source_pinch_local.tolist(),
        "insertion_axis_local": insertion.tolist(),
        "pad_opening_axis_local": opening.tolist(),
        "right_pad_center_local_m": (root_rotation.T @ (right - root_position)).tolist(),
        "left_pad_center_local_m": (root_rotation.T @ (left - root_position)).tolist(),
        "pad_center_separation_open_m": float(np.linalg.norm(left - right)),
    }


def namespace_menagerie(root: ET.Element, men: ET.Element, men_root: Path, axes: dict[str, Any]) -> ET.Element:
    classes = {e.get("class") for e in men.iter("default") if e.get("class")}
    names = {e.get("name") for e in men.iter() if e.get("name")}
    asset_meshes = men.findall("./asset/mesh")
    stems = {Path(e.get("file", "")).stem for e in asset_meshes if e.get("file")}
    mapping = {n: PREFIX + n for n in classes | names | stems}
    refs = {
        "class", "childclass", "mesh", "material", "texture", "joint", "joint1", "joint2",
        "body", "body1", "body2", "site", "geom", "tendon", "actuator", "target", "jointinparent",
    }
    for element in men.iter():
        for key, value in list(element.attrib.items()):
            if key == "name" or key in refs:
                if value in mapping:
                    element.set(key, mapping[value])
        if element.tag == "mesh" and element.get("file"):
            if element.get("name") is None:
                element.set("name", mapping[Path(element.get("file", "")).stem])
            element.set("file", str((men_root / "assets" / element.get("file", "")).resolve()))

    for element in men.findall("./asset/*"):
        root.find("asset").append(copy.deepcopy(element))
    for element in men.findall("./default/*"):
        root.find("default").append(copy.deepcopy(element))
    mount = copy.deepcopy(men.find("./worldbody/body"))
    if mount is None:
        raise RuntimeError("Menagerie model has no base_mount body")
    mount.set("pos", " ".join(f"{v:.12g}" for v in TOOL_POS))
    adapter_quat = axes["mount_quat_wxyz"]
    mount.set("quat", " ".join(f"{v:.12g}" for v in adapter_quat))
    ET.SubElement(mount, "site", {
        "name": "rq_m0_tcp", "pos": " ".join(f"{v:.12g}" for v in axes["tcp_pad_midpoint_local_m"]),
        "size": "0.006", "rgba": "1 0.2 0.1 1", "group": "5",
    })
    wrist = root.find('.//body[@name="right_wrist_roll_link"]')
    if wrist is None:
        raise RuntimeError("X2 MJCF has no right_wrist_roll_link")
    wrist.append(mount)
    for section in ("contact", "tendon", "equality"):
        source = men.find(section)
        if source is None:
            continue
        target = root.find(section)
        if target is None:
            target = ET.SubElement(root, section)
        for element in source:
            target.append(copy.deepcopy(element))
    actuator = root.find("actuator")
    for element in men.findall("./actuator/*"):
        actuator.append(copy.deepcopy(element))
    return mount


def add_scene(root: ET.Element, canonical, table_center_xy=None, bottle_root_pos=None) -> None:
    asset = root.find("asset")
    ET.SubElement(asset, "texture", {
        "name": "m0_checker", "type": "2d", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
        "width": "300", "height": "300",
    })
    ET.SubElement(asset, "texture", {
        "name": "m0_sky", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "material", {
        "name": "m0_floor_mat", "texture": "m0_checker", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2",
    })
    world = root.find("worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 1"})
    ET.SubElement(world, "light", {"pos": "0 -0.2 1", "dir": "0 0.2 -0.8", "directional": "true"})
    ET.SubElement(world, "geom", {
        "name": "m0_floor", "type": "plane", "size": "0 0 0.05", "material": "m0_floor_mat",
        "friction": "1 0.01 0.001", "condim": "4",
    })
    cx, cy = (canonical.G1_CANONICAL_TABLE_CENTER_XY if table_center_xy is None
              else table_center_xy)
    table = ET.SubElement(world, "body", {"name": "m0_table", "pos": f"{cx} {cy} 0"})
    ET.SubElement(table, "geom", {
        "name": "m0_table_top", "type": "box", "pos": "0 0 0.775", "size": "0.2 0.2 0.025",
        "rgba": "0.6 0.4 0.2 1", "friction": "1.2 0.01 0.001", "condim": "4",
    })
    for name, x, y in (
        ("m0_table_leg_front_left", -0.175, 0.175), ("m0_table_leg_front_right", 0.175, 0.175),
        ("m0_table_leg_back_left", -0.175, -0.175), ("m0_table_leg_back_right", 0.175, -0.175),
    ):
        ET.SubElement(table, "geom", {
            "name": name, "type": "box", "pos": f"{x} {y} 0.375", "size": "0.025 0.025 0.375",
            "rgba": "0.6 0.4 0.2 1", "friction": "0.9 0.01 0.001", "condim": "4",
        })
    bottle_pos = (canonical.CANONICAL_X2_BOTTLE_START_BODY_POS if bottle_root_pos is None
                  else bottle_root_pos)
    bottle = ET.SubElement(world, "body", {
        "name": "m0_bottle", "pos": " ".join(map(str, bottle_pos)),
    })
    ET.SubElement(bottle, "freejoint", {"name": "m0_bottle_free"})
    for geom in canonical.CANONICAL_X2_BOTTLE_GEOMS:
        ET.SubElement(bottle, "geom", {
            "name": geom["name"], "type": geom["type"], "pos": " ".join(map(str, geom["pos"])),
            "size": " ".join(map(str, geom["size"])), "mass": str(geom["mass"]),
            "rgba": " ".join(map(str, geom["rgba"])), "friction": "1.4 0.02 0.001", "condim": "4",
        })
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "720")
    ET.SubElement(visual, "headlight", {
        "diffuse": "0.75 0.75 0.75", "ambient": "0.30 0.30 0.30", "specular": "0.15 0.15 0.15",
    })


def identity(args, helper: Path) -> dict[str, Any]:
    x2_repo = checked_checkout(args.x2_root, X2_PIN)
    men_repo = checked_checkout(args.menagerie_root, MENAGERIE_PIN)
    x2_path = args.x2_root / X2_MJCF
    x2_urdf = args.x2_root / X2_URDF
    tool_urdf = args.x2_root / X2_TOOL_URDF
    men_path = args.menagerie_root / MENAGERIE_MJCF
    native = Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
    xroot = ET.parse(x2_urdf).getroot()
    mesh_names = sorted({Path(node.get("filename", "")).name for node in xroot.findall(".//mesh")})
    men_assets = sorted((args.menagerie_root / "robotiq_2f85/assets").glob("*.stl"))
    mount_joint = None
    urdf_tool = ET.parse(tool_urdf).getroot()
    for joint in urdf_tool.findall("joint"):
        if joint.get("name") == "R_omnipicker_joint":
            origin = joint.find("origin")
            mount_joint = {"name": joint.get("name"), "parent": joint.find("parent").get("link"),
                           "child": joint.find("child").get("link"), "xyz": origin.get("xyz"), "rpy": origin.get("rpy")}
            break
    if mount_joint is None or mount_joint["parent"] != "right_wrist_roll_link":
        raise RuntimeError("Could not verify official right tool mounting frame")
    urdf_joints = {}
    for joint in xroot.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            urdf_joints[joint.get("name")] = {
                "type": joint.get("type"), "lower": float(limit.get("lower", "nan")),
                "upper": float(limit.get("upper", "nan")), "velocity": float(limit.get("velocity", "nan")),
                "effort": float(limit.get("effort", "nan")),
            }
    return {
        "python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(), "native_library": str(native),
        "native_library_sha256": sha256(native), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "x2_repository": "https://github.com/AgibotTech/agibot_x2_urdf", "x2_checkout": x2_repo,
        "x2_mjcf": str(X2_MJCF), "x2_mjcf_sha256": sha256(x2_path),
        "x2_urdf": str(X2_URDF), "x2_urdf_sha256": sha256(x2_urdf),
        "x2_tool_variant_urdf": str(X2_TOOL_URDF), "x2_tool_variant_urdf_sha256": sha256(tool_urdf),
        "x2_license": "Mulan PSL v2", "x2_mesh_sha256": {n: sha256(args.x2_root / X2_MJCF.parent / "meshes" / n) for n in mesh_names},
        "official_x2_tool_mount": mount_joint,
        "robotiq_repository": "https://github.com/google-deepmind/mujoco_menagerie", "menagerie_checkout": men_repo,
        "robotiq_model": str(MENAGERIE_MJCF), "robotiq_xml_sha256": sha256(men_path),
        "robotiq_license": "BSD-2-Clause", "robotiq_asset_sha256": {p.name: sha256(p) for p in men_assets},
        "canonical_helper": str(helper), "canonical_helper_sha256": sha256(helper),
        "urdf_joint_limits": urdf_joints,
    }


def build_model(args, out: Path, canonical) -> tuple[mujoco.MjModel, dict[str, Any]]:
    x2_path = args.x2_root / X2_MJCF
    men_path = args.menagerie_root / MENAGERIE_MJCF
    station_base_yaw_deg = float(getattr(args, "station_base_yaw_deg", 0.0))
    base_yaw_rotation = Rotation.from_euler("z", station_base_yaw_deg, degrees=True).as_matrix()
    root = ET.parse(x2_path).getroot()
    neutral_x2 = mujoco.MjModel.from_xml_path(str(x2_path))
    neutral_data = mujoco.MjData(neutral_x2)
    mujoco.mj_resetData(neutral_x2, neutral_data)
    mujoco.mj_forward(neutral_x2, neutral_data)
    source_head = obj_id(neutral_x2, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")
    source_torso = obj_id(neutral_x2, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    source_head_geoms = [i for i in range(neutral_x2.ngeom)
                         if int(neutral_x2.geom_bodyid[i]) == source_head
                         and int(neutral_x2.geom_contype[i]) != 0]
    source_torso_geoms = [i for i in range(neutral_x2.ngeom)
                          if int(neutral_x2.geom_bodyid[i]) == source_torso
                          and int(neutral_x2.geom_contype[i]) != 0]
    source_pair_distances = []
    for head_geom in source_head_geoms:
        for torso_geom in source_torso_geoms:
            distance = float(mujoco.mj_geomDistance(
                neutral_x2, neutral_data, head_geom, torso_geom, 0.5, np.zeros(6)))
            source_pair_distances.append({"head_geom_id": head_geom, "torso_geom_id": torso_geom,
                                          "distance_m": distance})
    neutral_wrist_id = obj_id(neutral_x2, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    neutral_wrist_rotation = (base_yaw_rotation @
                              neutral_data.xmat[neutral_wrist_id].reshape(3, 3).copy())
    root.find("compiler").set("meshdir", str((x2_path.parent / "meshes").resolve()))
    root.find("compiler").set("autolimits", "true")
    pelvis = root.find('./worldbody/body[@name="pelvis"]')
    for element in list(pelvis):
        if element.tag == "freejoint" and element.get("name") == "floating_base_joint":
            pelvis.remove(element)
    if pelvis.find("freejoint") is not None:
        raise RuntimeError("X2 base was not fixed for the M0 fixture")
    station_base_pos = np.asarray(args.station_base_pos, dtype=float)
    pelvis.set("pos", " ".join(f"{v:.9g}" for v in station_base_pos))
    original_quat_wxyz = np.asarray([float(v) for v in pelvis.get("quat", "1 0 0 0").split()])
    original_rot = Rotation.from_quat(original_quat_wxyz[[1, 2, 3, 0]])
    yaw_rot = Rotation.from_euler("z", station_base_yaw_deg, degrees=True)
    pelvis_rot = yaw_rot * original_rot
    pelvis_quat_xyzw = pelvis_rot.as_quat()
    pelvis.set("quat", " ".join(f"{v:.12g}" for v in
                                  [pelvis_quat_xyzw[3], *pelvis_quat_xyzw[:3]]))
    pelvis.attrib.pop("euler", None)
    option = root.find("option")
    option.set("timestep", str(DT))
    option.set("gravity", f"0 0 {-GRAVITY}")
    men = ET.parse(men_path).getroot()
    axes = robotiq_axes(men_path)
    source_forward_rotation = np.asarray(axes["mount_rotation_matrix"], dtype=float)
    desired_world_rotation = base_yaw_rotation @ source_forward_rotation
    mount_rotation_in_wrist = neutral_wrist_rotation.T @ desired_world_rotation
    mount_quat_xyzw = Rotation.from_matrix(mount_rotation_in_wrist).as_quat()
    axes["desired_world_rotation_matrix"] = desired_world_rotation.tolist()
    axes["source_world_forward_rotation_matrix"] = source_forward_rotation.tolist()
    axes["isolated_lift_grasp_offset_tool_local_m"] = (
        source_forward_rotation.T @ np.array([-0.007, 0.0, 0.015])).tolist()
    axes["station_base_yaw_deg"] = station_base_yaw_deg
    axes["official_source_head_torso_neutral_geometry_distances"] = source_pair_distances
    axes["neutral_wrist_rotation_world"] = neutral_wrist_rotation.tolist()
    axes["mount_rotation_in_right_wrist_matrix"] = mount_rotation_in_wrist.tolist()
    axes["mount_rotation_matrix"] = mount_rotation_in_wrist.tolist()
    axes["mount_quat_wxyz"] = np.array([mount_quat_xyzw[3], *mount_quat_xyzw[:3]]).tolist()
    namespace_menagerie(root, men, args.menagerie_root / "robotiq_2f85", axes)
    coupler_margin = float(getattr(args, "coupler_limit_activation_margin_rad", 0.0))
    if not 0.0 <= coupler_margin <= 0.001:
        raise ValueError("SIMULATION_ONLY coupler activation margin must be in [0, 0.001] rad")
    if coupler_margin > 0.0:
        joint_nodes = {node.get("name"): node for node in root.iter("joint") if node.get("name")}
        for name in ("rq_left_coupler_joint", "rq_right_coupler_joint"):
            if name not in joint_nodes:
                raise RuntimeError(f"Mounted Menagerie model has no {name}")
            joint_nodes[name].set("margin", f"{coupler_margin:.12g}")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        if any({item.get("body1"), item.get("body2")} == {"head_pitch_link", "torso_link"}
               for item in contact.findall("exclude")):
            raise RuntimeError("The exact head/torso diagnostic exclusion already exists in the input")
        ET.SubElement(contact, "exclude", {"body1": "head_pitch_link", "body2": "torso_link"})
        axes["simulation_only_contact_exclusions"] = [{
            "body1": "head_pitch_link", "body2": "torso_link",
            "reason": "Verified internal source self-contact also present in original X2; excluded only in SIMULATION_ONLY diagnostic model",
        }]
    else:
        axes["simulation_only_contact_exclusions"] = []
    table_center = getattr(args, "table_center_xy", None)
    bottle_root = getattr(args, "bottle_root_pos", None)
    add_scene(root, canonical, table_center, bottle_root)
    xml_path = out / "x2_robotiq_2f85_simulation_only.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    expected_mount_mass = float(model.body_subtreemass[obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")])
    if abs(expected_mount_mass - REFERENCE_GRIPPER_MASS_KG) > 1e-5:
        raise RuntimeError(f"Mounted Robotiq mass mismatch: {expected_mount_mass:.9f} kg")
    axes["adapter_translation_in_right_wrist_frame_m"] = TOOL_POS.tolist()
    axes["adapter_quat_wxyz_in_right_wrist_frame"] = np.asarray(axes.pop("mount_quat_wxyz")).tolist()
    axes["fixed_base_station_position_world_m"] = station_base_pos.tolist()
    axes["table_center_xy_world_m"] = list(canonical.G1_CANONICAL_TABLE_CENTER_XY
                                           if table_center is None else table_center)
    axes["bottle_root_position_world_m"] = list(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS
                                                 if bottle_root is None else bottle_root)
    axes["coupler_limit_activation_margin_simulation_derived_rad"] = coupler_margin
    axes["source_joint_ranges_unchanged"] = True
    axes["source_faithful_coupler_baseline_status"] = "FAIL: reset spring torque exceeds upper stop on first step"
    axes["model_configuration"] = "SIMULATION_ONLY_DIAGNOSTIC" if coupler_margin else "SOURCE_FAITHFUL"
    axes["fixed_base_source_position_world_m"] = [0.0, 0.0, 0.68]
    axes["gripper_subtree_mass_kg"] = expected_mount_mass
    axes["model_xml"] = str(xml_path)
    axes["model_xml_sha256"] = sha256(xml_path)
    return model, axes


def contacts(model: mujoco.MjModel, data: mujoco.MjData, forces: bool = False) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        force = np.zeros(6)
        if forces and int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, index, force)
        rows.append({
            "geom1": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), "body1": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
            "geom2": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2), "body2": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
            "distance_m": float(contact.dist), "normal_force_n": float(force[0]) if forces and contact.efc_address >= 0 else None,
            "force_local_n": force.tolist() if forces and contact.efc_address >= 0 else None,
        })
    return rows


def render(model: mujoco.MjModel, data: mujoco.MjData, path: Path, lookat, distance, azimuth, elevation, size=(1280, 720)) -> None:
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def arm_metadata(model: mujoco.MjModel, ident: dict[str, Any]) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    qids = [qpos_id(model, name) for name in ARM]
    jids = [obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM]
    lower = np.array([model.jnt_range[jid, 0] for jid in jids])
    upper = np.array([model.jnt_range[jid, 1] for jid in jids])
    for name, jid in zip(ARM, jids):
        source = ident["urdf_joint_limits"].get(name)
        if not source:
            raise RuntimeError(f"Source URDF limit missing for {name}")
        if abs(source["lower"] - model.jnt_range[jid, 0]) > 1e-5 or abs(source["upper"] - model.jnt_range[jid, 1]) > 1e-5:
            raise RuntimeError(f"Compiled X2 range differs from source URDF for {name}")
    return qids, jids, lower, upper


def solve_tcp_ik(model: mujoco.MjModel, data: mujoco.MjData, target_pos: np.ndarray,
                 target_rot: np.ndarray, seed: np.ndarray, qids: list[int], lower: np.ndarray,
                 upper: np.ndarray) -> dict[str, Any]:
    site = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    evals: list[dict[str, Any]] = []

    def residual(q):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[site].copy()
        rot = data.site_xmat[site].reshape(3, 3).copy()
        pos_err = pos - target_pos
        rot_err = Rotation.from_matrix(target_rot.T @ rot).as_rotvec()
        evals.append({"evaluation": len(evals), "q_rad": q.tolist(), "position_error_m": pos_err.tolist(),
                      "orientation_error_rad": rot_err.tolist()})
        return np.concatenate((pos_err / 0.005, rot_err / 0.05))

    result = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7), bounds=(lower, upper),
                           xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=250, x_scale="jac")
    residual(result.x)
    pos_error = float(np.linalg.norm(data.site_xpos[site] - target_pos))
    rot_error = float(np.linalg.norm(Rotation.from_matrix(target_rot.T @ data.site_xmat[site].reshape(3, 3)).as_rotvec()))
    return {"success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev),
            "q_rad": result.x.tolist(), "target_position_m": target_pos.tolist(),
            "position_error_m": pos_error, "orientation_error_rad": rot_error,
            "evaluations": evals, "gate_pass": bool(pos_error <= 0.003 and rot_error <= 0.02)}


def solve_grasp_first_ik(model: mujoco.MjModel, data: mujoco.MjData, bottle_center: np.ndarray,
                         tool_offset_local: np.ndarray, insertion_local: np.ndarray,
                         vertical_local: np.ndarray, world_height_offset: float,
                         retreat_distance_m: float,
                         seed: np.ndarray, qids: list[int], lower: np.ndarray,
                         upper: np.ndarray) -> dict[str, Any]:
    """Solve a grasp from pad-center geometry and the mounted gripper root frame."""
    site = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    gripper_root = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
    evals: list[dict[str, Any]] = []

    def metrics(q):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        position = data.site_xpos[site].copy()
        rotation = data.site_xmat[site].reshape(3, 3).copy()
        insertion = rotation @ insertion_local
        target_position = (bottle_center + rotation @ tool_offset_local
                           + np.array([0., 0., world_height_offset])
                           - insertion * retreat_distance_m)
        vertical = rotation @ vertical_local
        desired_approach_xy = bottle_center - data.xpos[gripper_root]
        desired_approach_xy[2] = 0.0
        desired_approach_xy = desired_approach_xy[:2]
        desired_approach_xy /= max(float(np.linalg.norm(desired_approach_xy)), 1e-12)
        insertion_xy = insertion[:2]
        insertion_xy /= max(float(np.linalg.norm(insertion_xy)), 1e-12)
        yaw_error = math.atan2(
            desired_approach_xy[0] * insertion_xy[1] - desired_approach_xy[1] * insertion_xy[0],
            float(np.dot(desired_approach_xy, insertion_xy)))
        vertical_error = float(math.acos(np.clip(np.dot(vertical, np.array([0., 0., 1.])), -1.0, 1.0)))
        position_error = position - target_position
        return position, rotation, target_position, insertion, vertical, yaw_error, vertical_error, position_error

    def residual(q):
        position, rotation, target_position, insertion, vertical, yaw_error, vertical_error, position_error = metrics(q)
        evals.append({"evaluation": len(evals), "q_rad": q.tolist(),
                      "tcp_world_m": position.tolist(), "target_tcp_world_m": target_position.tolist(),
                      "position_error_m": position_error.tolist(), "insertion_axis_world": insertion.tolist(),
                      "vertical_axis_world": vertical.tolist(), "yaw_error_rad": yaw_error,
                      "vertical_error_rad": vertical_error})
        vertical_vector_error = vertical - np.array([0., 0., 1.])
        return np.concatenate((position_error / 0.005, vertical_vector_error / 0.04,
                               np.array([yaw_error / 0.30])))

    result = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                           bounds=(lower, upper), xtol=1e-11, ftol=1e-11,
                           gtol=1e-11, max_nfev=300, x_scale="jac")
    position, rotation, target_position, insertion, vertical, yaw_error, vertical_error, position_error = metrics(result.x)
    position_norm = float(np.linalg.norm(position_error))
    orientation_error = float(math.hypot(vertical_error, yaw_error))
    return {"success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev),
            "q_rad": result.x.tolist(), "target_position_m": target_position.tolist(),
            "position_error_m": position_norm, "orientation_error_rad": orientation_error,
            "vertical_error_rad": vertical_error, "yaw_error_rad": yaw_error,
            "target_rotation_matrix": rotation.tolist(), "insertion_axis_world": insertion.tolist(),
            "evaluations": evals,
            "gate_pass": bool(position_norm <= 0.003 and vertical_error <= 0.03
                               and abs(yaw_error) <= math.radians(25.0))}


def all_limited_joint_checks(model: mujoco.MjModel, data: mujoco.MjData, velocity_limits: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    range_errors, velocity_errors = [], []
    for jid in range(model.njnt):
        if int(model.jnt_limited[jid]) == 0 or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        lo, hi = map(float, model.jnt_range[jid])
        q = float(data.qpos[qadr])
        if q < lo - 1e-7 or q > hi + 1e-7:
            range_errors.append({"joint": name, "qpos": q, "range": [lo, hi]})
        vlim = velocity_limits.get(name)
        if vlim is not None and abs(float(data.qvel[dadr])) > vlim + 1e-4:
            velocity_errors.append({"joint": name, "qvel": float(data.qvel[dadr]), "source_velocity_limit": vlim})
    return range_errors, velocity_errors


def controller_config(model: mujoco.MjModel) -> list[dict[str, Any]]:
    items = []
    for aid in range(model.nu):
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[aid, 0])
        jname = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        aname = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        gear = float(model.actuator_gear[aid, 0])
        if "wrist_" in jname:
            kp, kd = 30.0, 5.0
        elif jname in ARM:
            kp, kd = 110.0, 14.0
        else:
            kp, kd = 80.0, 12.0
        lo, hi = map(float, model.jnt_actfrcrange[jid])
        if abs(lo) < 1e-9 and abs(hi) < 1e-9:
            if int(model.actuator_ctrllimited[aid]):
                lo, hi = map(float, model.actuator_ctrlrange[aid])
            else:
                lo, hi = -float("inf"), float("inf")
        if int(model.actuator_ctrllimited[aid]):
            ctrl_lo, ctrl_hi = map(float, model.actuator_ctrlrange[aid])
            lo = max(lo, min(ctrl_lo * gear, ctrl_hi * gear))
            hi = min(hi, max(ctrl_lo * gear, ctrl_hi * gear))
        items.append({"actuator_id": aid, "actuator": aname, "joint_id": jid, "joint": jname,
                      "qpos_id": int(model.jnt_qposadr[jid]), "dof_id": int(model.jnt_dofadr[jid]),
                      "gear": gear, "kp_simulation_derived": kp, "kd_simulation_derived": kd,
                      "torque_min_nm": lo, "torque_max_nm": hi})
    return items


def set_target_dict(model: mujoco.MjModel, q_arm: np.ndarray, base_targets: dict[str, float]) -> dict[str, float]:
    targets = dict(base_targets)
    targets.update({name: float(value) for name, value in zip(ARM, q_arm)})
    return targets


def apply_controller(model: mujoco.MjModel, data: mujoco.MjData, refs: list[dict[str, Any]], targets: dict[str, float], target_vel: dict[str, float]) -> list[dict[str, Any]]:
    commands = []
    for ref in refs:
        q = float(data.qpos[ref["qpos_id"]])
        v = float(data.qvel[ref["dof_id"]])
        qtarget = float(targets.get(ref["joint"], q))
        vtarget = float(target_vel.get(ref["joint"], 0.0))
        error, velocity_error = qtarget - q, vtarget - v
        bias = float(data.qfrc_bias[ref["dof_id"]])
        pd = ref["kp_simulation_derived"] * error + ref["kd_simulation_derived"] * velocity_error
        requested = bias + pd
        torque = float(np.clip(requested, ref["torque_min_nm"], ref["torque_max_nm"]))
        if abs(ref["gear"]) < 1e-12:
            raise RuntimeError(f"Zero motor gear on {ref['actuator']}")
        data.ctrl[ref["actuator_id"]] = torque / ref["gear"]
        commands.append({"joint": ref["joint"], "q_target": qtarget, "qvel_target": vtarget,
                         "qpos": q, "qvel": v, "position_error": error, "velocity_error": velocity_error,
                         "pd_torque_nm": pd, "bias_feedforward_nm": bias, "requested_torque_nm": requested,
                         "final_torque_nm": torque, "gear": ref["gear"]})
    return commands


def contact_sums(contact_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bottle = [c for c in contact_rows if "bottle" in c["body1"] or "bottle" in c["body2"]]
    pad_side = {"right": [], "left": []}
    table_force = 0.0
    unintended = []
    for c in bottle:
        other_geom = c["geom1"] if "bottle" in c["body2"] else c["geom2"]
        other_body = c["body1"] if "bottle" in c["body2"] else c["body2"]
        force = c["normal_force_n"] or 0.0
        if "rq_right_pad" in other_geom:
            pad_side["right"].append(c)
        elif "rq_left_pad" in other_geom:
            pad_side["left"].append(c)
        elif other_body.startswith("m0_table"):
            table_force += force
        else:
            unintended.append(c)
    return {"bottle_contacts": bottle, "right_pad_contacts": pad_side["right"],
            "left_pad_contacts": pad_side["left"], "table_normal_force_n": table_force,
            "unintended_bottle_contacts": unintended}


def quintic(u: float) -> tuple[float, float]:
    u = min(1.0, max(0.0, u))
    return 10*u**3 - 15*u**4 + 6*u**5, (30*u**2 - 60*u**3 + 30*u**4)


def run_segment(model, data, segment_name, qstart, qend, duration, grip_start, grip_end, targets,
                refs, writer, trace, contact_trace, velocity_limits, max_values, save_frame,
                q_waypoints: np.ndarray | None = None) -> dict[str, Any]:
    steps = max(1, int(round(duration / float(model.opt.timestep))))
    grip_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
    segment_start_time = float(data.time)
    first_contact_time = None
    bilateral_steps = 0
    any_pad_steps = 0
    segment_table_force = []
    for k in range(steps):
        u = (k + 1) / steps
        s, dsdu = quintic(u)
        if q_waypoints is None:
            qtarget = qstart + s * (qend - qstart)
            qdtarget = (dsdu / max(duration, 1e-12)) * (qend - qstart)
        else:
            path_position = s * (len(q_waypoints) - 1)
            segment = min(int(path_position), len(q_waypoints) - 2)
            segment_fraction = path_position - segment
            segment_delta = q_waypoints[segment + 1] - q_waypoints[segment]
            qtarget = q_waypoints[segment] + segment_fraction * segment_delta
            qdtarget = ((len(q_waypoints) - 1) * dsdu / max(duration, 1e-12)) * segment_delta
        arm_targets = set_target_dict(model, qtarget, targets)
        arm_vel = {name: float(v) for name, v in zip(ARM, qdtarget)}
        mujoco.mj_forward(model, data)
        command_rows = apply_controller(model, data, refs, arm_targets, arm_vel)
        grip_s, _ = quintic(u)
        data.ctrl[grip_id] = grip_start + grip_s * (grip_end - grip_start)
        mujoco.mj_step(model, data)
        range_errors, velocity_errors = all_limited_joint_checks(model, data, velocity_limits)
        if range_errors:
            raise RuntimeError(f"Source position-limit violation during {segment_name}: {range_errors[0]}")
        if velocity_errors:
            raise RuntimeError(f"Source velocity-limit violation during {segment_name}: {velocity_errors[0]}")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all():
            raise RuntimeError(f"Non-finite MuJoCo state during {segment_name}")
        rows = contacts(model, data, forces=True)
        sums = contact_sums(rows)
        if sums["bottle_contacts"] and first_contact_time is None:
            first_contact_time = float(data.time)
        has_right = bool(sums["right_pad_contacts"])
        has_left = bool(sums["left_pad_contacts"])
        if has_right and has_left:
            bilateral_steps += 1
        if has_right or has_left:
            any_pad_steps += 1
        segment_table_force.append(sums["table_normal_force_n"])
        if sums["unintended_bottle_contacts"] and segment_name not in {"close", "hold", "lift", "high_lift", "transfer", "lower", "release", "settle"}:
            raise RuntimeError(f"Bottle contacted a non-pad robot surface during {segment_name}: {sums['unintended_bottle_contacts'][0]}")
        if sums["unintended_bottle_contacts"] and any("right_wrist" in c["body1"] + c["body2"] for c in sums["unintended_bottle_contacts"]):
            raise RuntimeError(f"Wrist/bottle contact during {segment_name}")

        bottle_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
        bottle_com = data.xipos[bottle_id].copy()
        bottle_quat = data.xquat[bottle_id].copy()
        bottle_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
        bottle_dof = int(model.jnt_dofadr[bottle_joint])
        bottle_speed = float(np.linalg.norm(data.qvel[bottle_dof:bottle_dof+3]))
        bottle_spin = float(np.linalg.norm(data.qvel[bottle_dof+3:bottle_dof+6]))
        arm_error = max(abs(float(data.qpos[qpos_id(model, n)]) - float(v)) for n, v in zip(ARM, qtarget))
        pad_force = sum((c["normal_force_n"] or 0.0) for c in sums["right_pad_contacts"] + sums["left_pad_contacts"])
        row = {
            "step": int(data.time / model.opt.timestep), "time_s": float(data.time), "phase": segment_name,
            "gripper_ctrl": float(data.ctrl[grip_id]), "bottle_com_x_m": float(bottle_com[0]),
            "bottle_com_y_m": float(bottle_com[1]), "bottle_com_z_m": float(bottle_com[2]),
            "bottle_quat_wxyz": json.dumps(bottle_quat.tolist()), "bottle_linear_speed_m_s": bottle_speed,
            "bottle_angular_speed_rad_s": bottle_spin, "right_pad_contact_count": len(sums["right_pad_contacts"]),
            "left_pad_contact_count": len(sums["left_pad_contacts"]), "table_normal_force_n": sums["table_normal_force_n"],
            "pad_normal_force_sum_n": pad_force, "bottle_contacts": json.dumps(sums["bottle_contacts"], separators=(",", ":")),
            "right_arm_max_tracking_error_rad": arm_error, "qpos_writes_after_rollout_start": 0,
            "right_driver_qpos_rad": float(data.qpos[qpos_id(model, "rq_right_driver_joint")]),
            "left_driver_qpos_rad": float(data.qpos[qpos_id(model, "rq_left_driver_joint")]),
            "gripper_actuator_force": float(data.actuator_force[grip_id]),
            "pad_gap_m": float(np.linalg.norm(
                np.mean(data.geom_xpos[[obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_left_pad1", "rq_left_pad2")]], axis=0) -
                np.mean(data.geom_xpos[[obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_right_pad1", "rq_right_pad2")]], axis=0))),
            "right_arm_qpos_rad": json.dumps([float(data.qpos[qpos_id(model, n)]) for n in ARM]),
            "right_arm_target_rad": json.dumps(qtarget.tolist()),
            "right_arm_qvel_rad_s": json.dumps([float(data.qvel[dof_id(model, n)]) for n in ARM]),
        }
        trace.append(row)
        for c in rows:
            if "bottle" in c["body1"] or "bottle" in c["body2"]:
                contact_trace.append({"step": row["step"], "time_s": row["time_s"], "phase": segment_name, **c})
                if c["distance_m"] < max_values["minimum_bottle_contact_distance_m"]:
                    max_values["minimum_bottle_contact_distance_m"] = c["distance_m"]
        for c in command_rows:
            if abs(c["final_torque_nm"]) >= abs(max_values["maximum_motor_torque_nm"].get(c["joint"], 0.0)):
                max_values["maximum_motor_torque_nm"][c["joint"]] = c["final_torque_nm"]
        if writer is not None and (row["step"] % 20 == 0 or k == steps - 1):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.22, -0.05, 0.89]
            camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -14
            save_frame(data, camera)
        if k == steps // 2 or k == steps - 1:
            max_values["max_arm_tracking_error_rad"] = max(max_values["max_arm_tracking_error_rad"], arm_error)
    return {
        "segment": segment_name, "duration_s": float(data.time - segment_start_time), "steps": steps,
        "first_bottle_contact_time_s": first_contact_time, "bilateral_pad_contact_fraction": bilateral_steps / steps,
        "any_pad_contact_fraction": any_pad_steps / steps, "mean_table_normal_force_n": float(np.mean(segment_table_force)),
        "max_table_normal_force_n": float(np.max(segment_table_force)),
    }


def static_contact_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unwanted = []
    for row in rows:
        if row["distance_m"] > 0:
            continue
        bodies = {row["body1"], row["body2"]}
        if "m0_bottle" in bodies and any(name.startswith("m0_table") for name in bodies):
            continue
        if "m0_floor" in {row["geom1"], row["geom2"]} and any("ankle_roll_link" in name for name in bodies):
            continue
        unwanted.append(row)
    return {"all_contacts": rows, "penetrating_contacts": unwanted, "pass": not unwanted}


def static_loaded_corridor(model: mujoco.MjModel, source: mujoco.MjData,
                            arm_segments: list[tuple[str, np.ndarray, np.ndarray, bool]],
                            arm_qpos_ids: np.ndarray, bottle_joint_id: int,
                            tcp_site_id: int, bottle_body_id: int,
                            sample_count: int = 101) -> dict[str, Any]:
    """Check a scratch-only rigid bottle envelope from an observed loaded grasp."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = source.qpos
    scratch.ctrl[:] = source.ctrl
    scratch.time = source.time
    mujoco.mj_forward(model, scratch)

    bottle_qpos = int(model.jnt_qposadr[bottle_joint_id])
    bottle_root = source.qpos[bottle_qpos:bottle_qpos + 3].copy()
    bottle_quat = source.qpos[bottle_qpos + 3:bottle_qpos + 7].copy()
    bottle_rotation = Rotation.from_quat(bottle_quat[[1, 2, 3, 0]]).as_matrix()
    tcp_position = source.site_xpos[tcp_site_id].copy()
    tcp_rotation = source.site_xmat[tcp_site_id].reshape(3, 3).copy()
    relative_position = tcp_rotation.T @ (bottle_root - tcp_position)
    relative_rotation = tcp_rotation.T @ bottle_rotation
    baseline_com_z = float(source.xipos[bottle_body_id, 2])
    samples = []

    for segment_name, q_start, q_end, placement_segment in arm_segments:
        for fraction in np.linspace(0.0, 1.0, sample_count):
            scratch.qpos[:] = source.qpos
            scratch.ctrl[:] = source.ctrl
            scratch.qpos[arm_qpos_ids] = q_start + fraction * (q_end - q_start)
            scratch.qpos[bottle_qpos:bottle_qpos + 3] = bottle_root
            scratch.qpos[bottle_qpos + 3:bottle_qpos + 7] = bottle_quat
            mujoco.mj_forward(model, scratch)
            current_tcp_position = scratch.site_xpos[tcp_site_id].copy()
            current_tcp_rotation = scratch.site_xmat[tcp_site_id].reshape(3, 3).copy()
            virtual_root = current_tcp_position + current_tcp_rotation @ relative_position
            virtual_rotation = current_tcp_rotation @ relative_rotation
            virtual_quat_xyzw = Rotation.from_matrix(virtual_rotation).as_quat()
            virtual_quat_wxyz = np.array([virtual_quat_xyzw[3], *virtual_quat_xyzw[:3]])
            scratch.qpos[bottle_qpos:bottle_qpos + 3] = virtual_root
            scratch.qpos[bottle_qpos + 3:bottle_qpos + 7] = virtual_quat_wxyz
            mujoco.mj_forward(model, scratch)
            rows = contacts(model, scratch)
            bottle_rows = [row for row in rows
                           if "m0_bottle" in (row["body1"], row["body2"])]
            pad_rows = [row for row in bottle_rows
                        if "rq_left_pad" in row["geom1"] + row["geom2"]
                        or "rq_right_pad" in row["geom1"] + row["geom2"]]
            left_pad_rows = [row for row in pad_rows
                             if "rq_left_pad" in row["geom1"] + row["geom2"]]
            right_pad_rows = [row for row in pad_rows
                              if "rq_right_pad" in row["geom1"] + row["geom2"]]
            table_rows = [row for row in bottle_rows
                          if any(name.startswith("m0_table")
                                 for name in (row["body1"], row["body2"]))]
            nonpad_rows = [row for row in rows if row not in pad_rows]
            gate = static_contact_gate(nonpad_rows)
            com_z = float(scratch.xipos[bottle_body_id, 2])
            table_clear = placement_segment or not (com_z > baseline_com_z + 0.001 and table_rows)
            bilateral = bool(left_pad_rows and right_pad_rows)
            samples.append({
                "segment": segment_name,
                "fraction": float(fraction),
                "right_arm_qpos_rad": scratch.qpos[arm_qpos_ids].astype(float).tolist(),
                "tcp_world_m": current_tcp_position.tolist(),
                "virtual_bottle_root_world_m": virtual_root.tolist(),
                "virtual_bottle_com_world_m": scratch.xipos[bottle_body_id].astype(float).tolist(),
                "pad_contact_pairs": pad_rows,
                "nonpad_robot_contacts": gate["penetrating_contacts"],
                "table_bottle_contacts": table_rows,
                "bilateral_pad_geometry_present": bilateral,
                "table_clear_after_lift": table_clear,
                "placement_segment": placement_segment,
                "pass": gate["pass"] and bilateral and table_clear,
            })

    by_segment = {}
    for segment_name, _, _, _ in arm_segments:
        segment_samples = [row for row in samples if row["segment"] == segment_name]
        by_segment[segment_name] = {
            "status": "PASS" if segment_samples and all(row["pass"] for row in segment_samples) else "FAIL",
            "sample_count": len(segment_samples),
            "bilateral_sample_count": sum(row["bilateral_pad_geometry_present"] for row in segment_samples),
            "table_contact_sample_count": sum(bool(row["table_bottle_contacts"]) for row in segment_samples),
        }
    return {
        "status": "PASS" if samples and all(row["pass"] for row in samples) else "FAIL",
        "model": "scratch static collision replay using measured loaded-grasp bottle-to-TCP pose",
        "active_rollout_state_modified": False,
        "bottle_pose_method": "rigid pose propagated only in disposable preflight MjData; no mj_step",
        "measured_bottle_root_to_tcp_position_m": relative_position.tolist(),
        "measured_bottle_orientation_in_tcp_wxyz": np.array([
            Rotation.from_matrix(relative_rotation).as_quat()[3],
            *Rotation.from_matrix(relative_rotation).as_quat()[:3],
        ]).tolist(),
        "segments": by_segment,
        "samples": samples,
    }


def audit_head_torso_clearance(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    head_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")
    torso_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    yaw_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw_joint")
    pitch_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch_joint")
    yaw_qadr, pitch_qadr = int(model.jnt_qposadr[yaw_id]), int(model.jnt_qposadr[pitch_id])
    yaw_range = model.jnt_range[yaw_id].copy()
    pitch_range = model.jnt_range[pitch_id].copy()
    head_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == head_body and model.geom_contype[i] != 0]
    torso_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == torso_body and model.geom_contype[i] != 0]
    samples = []
    best = None
    safe = []
    for yaw in np.linspace(yaw_range[0], yaw_range[1], 101):
        for pitch in np.linspace(pitch_range[0], pitch_range[1], 101):
            data.qpos[yaw_qadr] = yaw
            data.qpos[pitch_qadr] = pitch
            mujoco.mj_forward(model, data)
            pairs = []
            for head_geom in head_geoms:
                for torso_geom in torso_geoms:
                    distance = float(mujoco.mj_geomDistance(
                        model, data, head_geom, torso_geom, 0.5, np.zeros(6)))
                    pairs.append({"geom1_id": head_geom, "geom2_id": torso_geom,
                                  "distance_m": distance})
            clearance = min((p["distance_m"] for p in pairs), default=math.inf)
            item = {"head_yaw_rad": float(yaw), "head_pitch_rad": float(pitch),
                    "pair_contacts": pairs, "minimum_pair_distance_m": clearance,
                    "separated_beyond_contact_margin": clearance >= 0.0,
                    "pair_collision_excluded_in_simulation_model": True}
            samples.append(item)
            if clearance >= 0.0:
                safe.append(item)
            elif best is None or clearance > best["minimum_pair_distance_m"]:
                best = item
    neutral = min(samples, key=lambda row: row["head_yaw_rad"] ** 2 + row["head_pitch_rad"] ** 2)
    selected = {"head_yaw_rad": 0.0, "head_pitch_rad": 0.0,
                "minimum_pair_distance_m": neutral["minimum_pair_distance_m"],
                "separated_beyond_contact_margin": neutral["separated_beyond_contact_margin"],
                "pair_geometric_distances": neutral["pair_contacts"],
                "pair_collision_excluded_in_simulation_model": True,
                "selection": "official neutral head configuration; not optimized to reduce source overlap"}
    return {
        "source_joint_ranges_rad": {"head_yaw_joint": yaw_range.tolist(), "head_pitch_joint": pitch_range.tolist()},
        "collision_geom_ids": {"head_pitch_link": head_geoms, "torso_link": torso_geoms},
        "grid_shape": [101, 101], "sample_count": len(samples), "geometrically_clear_samples": len(safe),
        "simulated_pair_exclusion": "head_pitch_link <-> torso_link, exact pair only",
        "best_sample": best, "selected_initial_pose": selected, "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=STATION_BASE_POS.tolist(),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--station-base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--table-center-xy", type=float, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--bottle-root-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-bottle-root-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--grasp-height-offset-m", type=float, default=0.0,
                        help="World-vertical shift from the source-derived bottle-centered pad pose")
    parser.add_argument("--pregrasp-retreat-m", type=float, default=PREGRASP_RETREAT_M,
                        help="Distance to retreat along the measured gripper insertion axis")
    parser.add_argument("--lift-command-m", type=float, default=LIFT_COMMAND_M,
                        help="Vertical TCP lift command; source-limited IK validates the endpoint and path")
    parser.add_argument("--coupler-limit-activation-margin-rad", type=float, default=0.0,
                        help="SIMULATION_ONLY solver activation margin; source joint ranges remain unchanged")
    args = parser.parse_args()
    lift_command_m = float(args.lift_command_m)
    if not math.isfinite(lift_command_m) or not 0.0 < lift_command_m <= 0.15:
        parser.error("--lift-command-m must be finite and in (0, 0.15] m")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    static_dir = out / "static"
    raw_dir = out / "raw"
    static_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    result: dict[str, Any] = {"experiment": "X2 + Robotiq 2F-85 — SIMULATION_ONLY",
                              "status": "BLOCKED", "highest_gate": "NONE", "stages": {}}
    trace: list[dict[str, Any]] = []
    contact_trace: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    max_values = {"minimum_bottle_contact_distance_m": 0.0,
                  "maximum_motor_torque_nm": {}, "max_arm_tracking_error_rad": 0.0}
    result["runner"] = {
        "path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
        "station_base_position_world_m": list(args.station_base_pos),
        "station_base_yaw_deg": float(args.station_base_yaw_deg),
    }
    try:
        if sha256(args.canonical_helper) != CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash differs from the accepted immutable helper")
        canonical = load_module(args.canonical_helper)
        ident = identity(args, args.canonical_helper)
        result["identity"] = ident
        model, adapter = build_model(args, out, canonical)
        bottle_start_root = np.asarray(adapter["bottle_root_position_world_m"], dtype=float)
        bottle_target_root = np.asarray(
            args.target_bottle_root_pos if args.target_bottle_root_pos is not None
            else canonical.CANONICAL_X2_BOTTLE_TARGET_BODY_POS, dtype=float)
        bottle_transfer_delta = bottle_target_root - bottle_start_root
        if np.linalg.norm(bottle_transfer_delta[:2]) < 0.01:
            raise RuntimeError("Scene-first layout must include a meaningful supported placement transfer")
        result["adapter"] = adapter
        result["scene_configuration"] = {
            "table_center_xy_world_m": adapter["table_center_xy_world_m"],
            "bottle_root_position_world_m": adapter["bottle_root_position_world_m"],
            "station_base_position_world_m": adapter["fixed_base_station_position_world_m"],
            "station_base_yaw_deg": adapter["station_base_yaw_deg"],
            "target_bottle_root_position_world_m": bottle_target_root.tolist(),
            "grasp_height_offset_world_m": float(args.grasp_height_offset_m),
            "pregrasp_retreat_m": float(args.pregrasp_retreat_m),
            "lift_command_m": lift_command_m,
            "bottle_transfer_delta_world_m": bottle_transfer_delta.tolist(),
            "model_configuration": adapter["model_configuration"],
            "coupler_limit_activation_margin_simulation_derived_rad": adapter[
                "coupler_limit_activation_margin_simulation_derived_rad"],
        }
        result["highest_gate"] = "PINNED SOURCE IDENTITY + MODEL COMPILED"
        if abs(float(model.opt.timestep) - DT) > 1e-12 or model.opt.gravity[2] != -GRAVITY:
            raise RuntimeError("Compiled timestep/gravity does not match the recorded X2 M0 settings")
        result["model"] = {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "nbody": int(model.nbody),
            "ngeom": int(model.ngeom), "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
            "solver_iterations": int(model.opt.iterations), "gripper_actuator": "rq_fingers_actuator",
            "gripper_actuator_ctrlrange": model.actuator_ctrlrange[obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")].tolist(),
            "gripper_actuator_forcerange": model.actuator_forcerange[obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")].tolist(),
            "gripper_equalities": [obj_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)],
            "gripper_tendons": [obj_name(model, mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(model.ntendon)],
        }
        qids, jids, lower, upper = arm_metadata(model, ident)
        velocity_limits = {name: value["velocity"] for name, value in ident["urdf_joint_limits"].items()
                           if math.isfinite(value["velocity"]) and value["velocity"] > 0}
        refs = controller_config(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        head_audit = audit_head_torso_clearance(model, data)
        result["head_torso_clearance_audit"] = head_audit
        head_initial = head_audit["selected_initial_pose"]
        data.qpos[qpos_id(model, "head_yaw_joint")] = head_initial["head_yaw_rad"]
        data.qpos[qpos_id(model, "head_pitch_joint")] = head_initial["head_pitch_rad"]
        left_elbow_qpos = qpos_id(model, "left_elbow_joint")
        data.qpos[left_elbow_qpos] = -0.20
        data.ctrl[obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")] = 0.0
        mujoco.mj_forward(model, data)
        initial_contacts = contacts(model, data, forces=True)
        result["stages"]["mounted_static"] = {
            "status": "PASS" if static_contact_gate(initial_contacts)["pass"] else "FAIL",
            "base_fixed": True, "source_arm_limits_match": True, "initial_contacts": initial_contacts,
            "self_contact_gate": static_contact_gate(initial_contacts),
        }
        if result["stages"]["mounted_static"]["status"] == "PASS":
            result["highest_gate"] = "MOUNTED MODEL INITIAL CLEARANCE"
        else:
            result["stages"]["mounted_open_close"] = {
                "status": "NOT RUN", "reason": "Initial collision-clearance gate failed"
            }
        render(model, data, static_dir / "overview_home.png", (0.12, -0.05, 0.84), 2.6, 135, -12)
        render(model, data, static_dir / "front_home.png", (0.12, -0.05, 0.9), 2.0, 90, -8)
        render(model, data, static_dir / "side_home.png", (0.12, -0.05, 0.9), 2.0, 180, -8)
        render(model, data, static_dir / "robotiq_mount_closeup.png", (0.0, -0.12, 0.6), 0.8, 135, -12)

        # Convert the wrist-local adapter rotation into the world frame at the
        # source-neutral arm pose before using it as a world-space IK target.
        wrist_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        wrist_rot_world = data.xmat[wrist_id].reshape(3, 3).copy()
        target_rot = wrist_rot_world @ np.asarray(adapter["mount_rotation_matrix"], dtype=float)
        adapter["neutral_wrist_rotation_world"] = wrist_rot_world.tolist()
        desired_rot = np.asarray(adapter["desired_world_rotation_matrix"], dtype=float)
        adapter["compiled_neutral_gripper_rotation_world"] = target_rot.tolist()
        adapter["mount_orientation_error_rad"] = float(np.linalg.norm(
            Rotation.from_matrix(desired_rot.T @ target_rot).as_rotvec()))
        if adapter["mount_orientation_error_rad"] > 1e-8:
            raise RuntimeError("Compiled Robotiq mount does not match the intended world jaw basis")
        adapter["neutral_gripper_rotation_world"] = desired_rot.tolist()
        target_rot = desired_rot
        tcp_site_id = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
        bottle_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
        bottle_center = data.geom_xpos[bottle_geom_id].copy()
        grasp_offset_local = np.asarray(adapter["isolated_lift_grasp_offset_tool_local_m"], dtype=float)
        insertion_local = np.asarray(adapter["insertion_axis_local"], dtype=float)
        vertical_local = np.cross(insertion_local, np.asarray(adapter["pad_opening_axis_local"], dtype=float))
        vertical_local /= np.linalg.norm(vertical_local)
        q_approach = solve_grasp_first_ik(
            model, data, bottle_center, grasp_offset_local, insertion_local, vertical_local,
            float(args.grasp_height_offset_m), 0.0, data.qpos[qids].copy(), qids, lower, upper)
        ik_results = {"grasp_first": q_approach}
        if not q_approach["gate_pass"]:
            result["ik"] = ik_results
            result["stages"]["source_limited_grasp_ik"] = {
                "status": "FAIL", "position_error_m": q_approach["position_error_m"],
                "orientation_error_rad": q_approach["orientation_error_rad"],
                "vertical_error_rad": q_approach["vertical_error_rad"],
                "yaw_error_rad": q_approach["yaw_error_rad"],
                "joint_pose_rad": q_approach["q_rad"],
                "collision_validation_follows_ik": True,
            }
            raise RuntimeError("Reachable bottle-centered GRASP configuration did not meet position/axis/yaw gates")
        target_rot = np.asarray(q_approach["target_rotation_matrix"], dtype=float)
        grasp_target = np.asarray(q_approach["target_position_m"], dtype=float)
        insertion_axis = target_rot @ insertion_local
        result["stages"]["source_limited_grasp_ik"] = {
            "status": "PASS" if q_approach["gate_pass"] else "FAIL",
            "position_error_m": q_approach["position_error_m"],
            "orientation_error_rad": q_approach["orientation_error_rad"],
            "vertical_error_rad": q_approach["vertical_error_rad"],
            "approach_yaw_error_rad": q_approach["yaw_error_rad"],
            "joint_pose_rad": q_approach["q_rad"], "target_position_m": q_approach["target_position_m"],
            "collision_validation_follows_ik": True,
        }
        q_pre = solve_grasp_first_ik(
            model, data, bottle_center, grasp_offset_local, insertion_local, vertical_local,
            float(args.grasp_height_offset_m), float(args.pregrasp_retreat_m),
            np.asarray(q_approach["q_rad"]), qids, lower, upper)
        ik_results["pregrasp_retreat_from_grasp"] = q_pre
        result["ik"] = ik_results
        result["stages"]["source_limited_pregrasp_ik"] = {
            "status": "PASS" if q_pre["gate_pass"] else "FAIL",
            "position_error_m": q_pre["position_error_m"],
            "orientation_error_rad": q_pre["orientation_error_rad"],
            "vertical_error_rad": q_pre["vertical_error_rad"],
            "approach_yaw_error_rad": q_pre["yaw_error_rad"],
            "joint_pose_rad": q_pre["q_rad"], "target_position_m": q_pre["target_position_m"],
            "retreat_distance_along_insertion_m": float(args.pregrasp_retreat_m),
        }
        if q_pre["gate_pass"]:
            result["highest_gate"] = "MOUNTED MODEL + GRASP-FIRST SOURCE-LIMITED IK"
            result["stages"]["source_limited_pregrasp_ik"]["derived_from_validated_grasp_pose"] = True
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        if not q_pre["gate_pass"]:
            raise RuntimeError(f"Source-limited PREGRASP retreat IK failed: {q_pre['position_error_m']:.6f} m, {q_pre['orientation_error_rad']:.6f} rad")
        ik_results["approach_same_as_grasp_configuration"] = q_approach
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        if not q_approach["gate_pass"]:
            raise RuntimeError(f"Source-limited APPROACH IK failed: {q_approach['position_error_m']:.6f} m, {q_approach['orientation_error_rad']:.6f} rad")
        q_lift = solve_tcp_ik(model, data, grasp_target + np.array([0., 0., lift_command_m]), target_rot,
                              np.asarray(q_approach["q_rad"]), qids, lower, upper)
        ik_results["lift_55mm_command"] = q_lift
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        result["stages"]["source_limited_short_lift_ik"] = {
            "status": "PASS" if q_lift["gate_pass"] else "FAIL",
            "position_error_m": q_lift["position_error_m"],
            "orientation_error_rad": q_lift["orientation_error_rad"],
            "joint_pose_rad": q_lift["q_rad"],
            "reason": None if q_lift["gate_pass"] else "No source-limited 55 mm TCP endpoint",
        }
        lift_waypoint_iks = []
        lift_waypoint_q = [np.asarray(q_approach["q_rad"], dtype=float)]
        lift_waypoint_count = max(1, math.ceil(lift_command_m / 0.005))
        for waypoint_index in range(1, lift_waypoint_count + 1):
            waypoint_target = grasp_target + np.array([
                0., 0., lift_command_m * waypoint_index / lift_waypoint_count,
            ])
            waypoint = solve_tcp_ik(
                model, data, waypoint_target, target_rot, lift_waypoint_q[-1],
                qids, lower, upper,
            )
            waypoint["waypoint_index"] = waypoint_index
            waypoint["vertical_fraction"] = waypoint_index / lift_waypoint_count
            lift_waypoint_iks.append(waypoint)
            if not waypoint["gate_pass"]:
                break
            lift_waypoint_q.append(np.asarray(waypoint["q_rad"], dtype=float))
        lift_waypoint_path_pass = (
            len(lift_waypoint_iks) == lift_waypoint_count
            and all(item["gate_pass"] for item in lift_waypoint_iks)
        )
        result["stages"]["source_limited_cartesian_lift_path_ik"] = {
            "status": "PASS" if lift_waypoint_path_pass else "FAIL",
            "method": "sequential source-limited IK waypoints along a straight vertical TCP path",
            "waypoint_spacing_m": lift_command_m / lift_waypoint_count,
            "planned_waypoint_count": lift_waypoint_count,
            "solved_waypoint_count": len(lift_waypoint_iks),
            "waypoints": [
                {key: item[key] for key in (
                    "waypoint_index", "vertical_fraction", "target_position_m",
                    "position_error_m", "orientation_error_rad", "q_rad", "gate_pass",
                )}
                for item in lift_waypoint_iks
            ],
        }
        ik_results["cartesian_lift_waypoints"] = lift_waypoint_iks
        json_write(raw_dir / "cartesian_lift_waypoint_ik.json", lift_waypoint_iks)
        if lift_waypoint_path_pass:
            q_lift = lift_waypoint_iks[-1]
        high_target = grasp_target + np.array([0., 0., lift_command_m + 0.020])
        transfer_target = high_target + bottle_transfer_delta
        place_target = grasp_target + bottle_transfer_delta
        q_high = solve_tcp_ik(model, data, high_target, target_rot,
                              np.asarray(q_lift["q_rad"]), qids, lower, upper)
        ik_results["high_lift"] = q_high
        q_transfer = solve_tcp_ik(model, data, transfer_target, target_rot,
                                  np.asarray(q_high["q_rad"]), qids, lower, upper)
        ik_results["transfer"] = q_transfer
        q_place = solve_tcp_ik(model, data, place_target, target_rot,
                               np.asarray(q_transfer["q_rad"]), qids, lower, upper)
        ik_results["place"] = q_place
        q_retract = solve_tcp_ik(model, data, place_target - insertion_axis * 0.12, target_rot,
                                 np.asarray(q_place["q_rad"]), qids, lower, upper)
        ik_results["retract"] = q_retract
        for name, item in (("high_lift", q_high), ("transfer", q_transfer),
                           ("place", q_place), ("retract", q_retract)):
            result["stages"][f"source_limited_{name}_ik"] = {
                "status": "PASS" if item["gate_pass"] else "FAIL",
                "position_error_m": item["position_error_m"],
                "orientation_error_rad": item["orientation_error_rad"],
                "joint_pose_rad": item["q_rad"],
            }
        result["ik"] = ik_results
        json_write(raw_dir / "ik_solver_trace.json", ik_results)

        # Static collision checks at the no-contact path endpoints. The full
        # interpolation is rechecked immediately before the first rollout.
        path_checks = []
        bottle_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
        for label, q in (("pregrasp", np.asarray(q_pre["q_rad"])),
                         ("grasp", np.asarray(q_approach["q_rad"]))):
            data.qpos[qids] = q
            data.ctrl[obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")] = 0.0
            mujoco.mj_forward(model, data)
            rows = contacts(model, data)
            gate = static_contact_gate(rows)
            path_checks.append({"pose": label, "contacts": rows,
                                "unwanted_penetrations": gate["penetrating_contacts"], "pass": gate["pass"]})
        result["stages"]["static_ik_and_clearance"] = {
            "status": "PASS" if all(c["pass"] for c in path_checks) else "FAIL",
            "target_bottle_center_m": bottle_center.tolist(), "pregrasp_target_m": q_pre["target_position_m"],
            "approach_target_m": grasp_target.tolist(), "path_checks": path_checks,
        }
        result["highest_gate"] = "MOUNTED STATIC + SOURCE-LIMITED IK" if result["stages"]["static_ik_and_clearance"]["status"] == "PASS" else "MOUNTED STATIC"
        data.qpos[qids] = np.asarray(q_pre["q_rad"])
        mujoco.mj_forward(model, data)
        render(model, data, static_dir / "open_pregrasp.png", (0.22, -0.05, 0.9), 1.8, 135, -8)
        data.qpos[qids] = np.asarray(q_approach["q_rad"])
        mujoco.mj_forward(model, data)
        render(model, data, static_dir / "open_approach.png", (0.28, -0.02, 0.89), 0.85, 135, -10)

        # Stage 1: isolated OPEN/CLOSE/REOPEN with no object contact.
        if result["stages"]["mounted_static"]["status"] != "PASS" or result["stages"]["static_ik_and_clearance"]["status"] != "PASS":
            result["highest_gate"] = "MOUNTED STATIC; CLEARANCE/IK GATE FAILED"
            raise RuntimeError("Static initial or source-limited corridor clearance gate failed; no physics rollout started")

        dry = mujoco.MjData(model)
        mujoco.mj_resetData(model, dry)
        dry.qpos[qpos_id(model, "head_yaw_joint")] = head_initial["head_yaw_rad"]
        dry.qpos[qpos_id(model, "head_pitch_joint")] = head_initial["head_pitch_rad"]
        dry.qpos[left_elbow_qpos] = -0.20
        dry.qpos[qids] = np.asarray(q_pre["q_rad"])
        dry_bottle_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
        dry_bottle_qadr = int(model.jnt_qposadr[dry_bottle_joint])
        dry.qpos[dry_bottle_qadr:dry_bottle_qadr + 3] = np.array([2.0, 2.0, 12.0])
        mujoco.mj_forward(model, dry)
        base_targets = {obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[r["actuator_id"], 0])): 0.0
                        for r in refs if r["joint"] not in ARM}
        grip_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
        open_q = dry.qpos[qids].copy()
        stage1_rows = []
        stage1_contact_rows = []
        stage1_max = {"minimum_bottle_contact_distance_m": 0.0, "maximum_motor_torque_nm": {}, "max_arm_tracking_error_rad": 0.0}
        dry_writer = imageio.get_writer(str(out / "stage1_open_close.mp4"), fps=25, codec="libx264", quality=7)
        def dry_frame(d, camera):
            renderer = mujoco.Renderer(model, height=720, width=1280)
            renderer.update_scene(d, camera=camera)
            dry_writer.append_data(renderer.render())
            renderer.close()
        dry_stages = []
        try:
            for name, gs, ge, duration in (("dry_open", 0., 0., 0.25), ("dry_close", 0., 255., 0.5), ("dry_reopen", 255., 0., 0.5)):
                dry_stages.append(run_segment(model, dry, name, open_q, open_q, duration, gs, ge, base_targets,
                                              refs, dry_writer, stage1_rows, stage1_contact_rows, velocity_limits,
                                              stage1_max, dry_frame))
        finally:
            dry_writer.close()
        driver_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rq_right_driver_joint")
        follower_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rq_left_driver_joint")
        driver_qpos, follower_qpos = int(model.jnt_qposadr[driver_id]), int(model.jnt_qposadr[follower_id])
        driver_values = [r["right_driver_qpos_rad"] for r in stage1_rows]
        follower_values = [r["left_driver_qpos_rad"] for r in stage1_rows]
        gap_values = [r["pad_gap_m"] for r in stage1_rows]
        gripper_motion = {
            "driver_qpos_initial_open_rad": float(driver_values[0]),
            "driver_qpos_min_rad": float(min(driver_values)), "driver_qpos_max_rad": float(max(driver_values)),
            "follower_qpos_min_rad": float(min(follower_values)), "follower_qpos_max_rad": float(max(follower_values)),
            "pad_gap_min_m": float(min(gap_values)), "pad_gap_max_m": float(max(gap_values)),
            "stages": dry_stages,
            "final_driver_qpos_rad": float(dry.qpos[driver_qpos]),
            "final_follower_qpos_rad": float(dry.qpos[follower_qpos]),
            "final_open_pad_separation_m": float(np.linalg.norm(
                np.mean(dry.geom_xpos[[obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_left_pad1", "rq_left_pad2")]], axis=0) -
                np.mean(dry.geom_xpos[[obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_right_pad1", "rq_right_pad2")]], axis=0))),
            "bottle_contact_during_dry_run": any("bottle" in c["body1"] + c["body2"] for c in stage1_contact_rows),
            "qpos_writes_after_start": 0,
        }
        json_write(raw_dir / "stage1_open_close_trace.json", stage1_rows)
        with (raw_dir / "stage1_open_close_contacts.jsonl").open("w", encoding="utf-8") as stream:
            for item in stage1_contact_rows:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
        result["stages"]["mounted_open_close"] = {
            "status": "PASS" if gripper_motion["final_open_pad_separation_m"] > 0.02 and not gripper_motion["bottle_contact_during_dry_run"] else "FAIL",
            "diagnostic_arm_hold_pose": {"right_arm_joint_rad": open_q.astype(float).tolist(),
                                          "pose_source": "validated source-limited pregrasp, set before rollout"},
            "diagnostic_bottle": {"pose_set_before_diagnostic_steps_m": [2.0, 2.0, 12.0],
                                  "separate_reset_MjData": True,
                                  "active_manipulation_rollout_uses_canonical_bottle_pose": True},
            "motion": gripper_motion,
        }
        result["highest_gate"] = "MOUNTED OPEN/CLOSE + STATIC IK"
        render(model, dry, static_dir / "open_close_reopen.png", (0.1, -0.14, 0.65), 0.8, 135, -10)

        if result["stages"]["static_ik_and_clearance"]["status"] != "PASS":
            raise RuntimeError("Static source-limited IK or collision clearance gate failed")
        if result["stages"]["mounted_open_close"]["status"] != "PASS":
            raise RuntimeError("Mounted gripper OPEN/CLOSE validation failed")

        # The open approach is the pre-contact collision gate. Carry clearance is
        # checked later from the measured contact-loaded bottle-to-TCP transform.
        approach_path_samples = []
        bottle_freejoint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
        bottle_freejoint_qpos = int(model.jnt_qposadr[bottle_freejoint_id])
        initial_bottle_root_qpos = data.qpos[bottle_freejoint_qpos:bottle_freejoint_qpos + 3].copy()
        initial_bottle_quat = data.qpos[bottle_freejoint_qpos + 3:bottle_freejoint_qpos + 7].copy()
        for fraction in np.linspace(0.0, 1.0, 101):
            data.qpos[qids] = np.asarray(q_pre["q_rad"]) + fraction * (
                np.asarray(q_approach["q_rad"]) - np.asarray(q_pre["q_rad"]))
            data.qpos[bottle_freejoint_qpos:bottle_freejoint_qpos + 3] = initial_bottle_root_qpos
            data.qpos[bottle_freejoint_qpos + 3:bottle_freejoint_qpos + 7] = initial_bottle_quat
            data.ctrl[grip_id] = 0.0
            mujoco.mj_forward(model, data)
            rows = contacts(model, data)
            gate = static_contact_gate(rows)
            approach_path_samples.append({
                "fraction": float(fraction),
                "right_arm_qpos_rad": data.qpos[qids].astype(float).tolist(),
                "contacts": rows,
                "unwanted_penetrations": gate["penetrating_contacts"],
                "pass": gate["pass"],
            })
        approach_pass = bool(approach_path_samples) and all(row["pass"] for row in approach_path_samples)
        result["stages"]["static_approach_path"] = {
            "status": "PASS" if approach_pass else "FAIL",
            "sample_count": len(approach_path_samples),
            "configuration": "open gripper, original free bottle at supported reset pose",
            "samples": approach_path_samples,
        }
        json_write(raw_dir / "static_approach_path_samples.json", approach_path_samples)
        if not approach_pass:
            raise RuntimeError("Open PREGRASP-to-GRASP approach contains an unintended collision")
        result["highest_gate"] = "SOURCE-LIMITED GRASP + OPEN APPROACH CORRIDOR"

        # From here onward, qpos is initialized once, before the first task step.
        run = mujoco.MjData(model)
        mujoco.mj_resetData(model, run)
        run.qpos[qpos_id(model, "head_yaw_joint")] = head_initial["head_yaw_rad"]
        run.qpos[qpos_id(model, "head_pitch_joint")] = head_initial["head_pitch_rad"]
        run.qpos[left_elbow_qpos] = -0.20
        home = np.zeros(len(ARM))
        run.qpos[qids] = np.asarray(q_pre["q_rad"])
        run.ctrl[grip_id] = 0.0
        mujoco.mj_forward(model, run)
        initial_contacts = contacts(model, run, forces=True)
        initial_gate = static_contact_gate(initial_contacts)
        if not initial_gate["pass"]:
            result["stages"]["rollout_initialization"] = {"status": "FAIL", **initial_gate}
            raise RuntimeError("Initial robot/gripper/bottle/table state contains penetration")
        initial_com = run.xipos[bottle_id].copy()
        initial_bottle_quat = run.xquat[bottle_id].copy()
        result["initial_arm_pose"] = {"label": "source-limited pregrasp initialized before rollout",
                                       "right_arm_joint_rad": run.qpos[qids].astype(float).tolist(),
                                       "unused_left_elbow_joint_rad": float(run.qpos[left_elbow_qpos]),
                                       "unused_left_elbow_source_range_rad": model.jnt_range[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")].astype(float).tolist()}
        result["initial_bottle_com_m"] = initial_com.tolist()
        result["initial_bottle_quaternion_wxyz"] = initial_bottle_quat.tolist()
        writer = imageio.get_writer(str(out / "x2_robotiq_m0_physics.mp4"), fps=25, codec="libx264", quality=7)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.22, -0.05, 0.89]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -14
        def append_frame(d, cam):
            renderer.update_scene(d, camera=cam)
            writer.append_data(renderer.render())
        base_targets = {}
        for ref in refs:
            if ref["joint"] not in ARM:
                base_targets[ref["joint"]] = float(run.qpos[ref["qpos_id"]])
        q_home = np.asarray(q_pre["q_rad"])
        q_pre_v, q_app_v = np.asarray(q_pre["q_rad"]), np.asarray(q_approach["q_rad"])
        q_lift_v, q_high_v = np.asarray(q_lift["q_rad"]), np.asarray(q_high["q_rad"])
        q_transfer_v, q_place_v = np.asarray(q_transfer["q_rad"]), np.asarray(q_place["q_rad"])
        try:
            # Initial open settle, followed by the staged task.
            phases.append(run_segment(model, run, "pregrasp_settle", q_home, q_home, 0.75, 0., 0., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            phases.append(run_segment(model, run, "approach", q_pre_v, q_app_v, 0.75, 0., 0., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            phases.append(run_segment(model, run, "close", q_app_v, q_app_v, 0.75, 0., 255., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            phases.append(run_segment(model, run, "hold", q_app_v, q_app_v, 1.0, 255., 255., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            hold_rows = [r for r in trace if r["phase"] == "hold"]
            bilateral_fraction = sum(r["right_pad_contact_count"] > 0 and r["left_pad_contact_count"] > 0 for r in hold_rows) / max(1, len(hold_rows))
            hold_pos = np.array([[r["bottle_com_x_m"], r["bottle_com_y_m"], r["bottle_com_z_m"]] for r in hold_rows])
            hold_slip = float(np.max(np.linalg.norm(hold_pos - hold_pos[0], axis=1))) if len(hold_pos) else math.inf
            result["stages"]["bilateral_grasp_hold"] = {
                "status": "PASS" if bilateral_fraction >= 0.80 and hold_slip <= 0.005 else "FAIL",
                "bilateral_pad_contact_fraction": bilateral_fraction, "bottle_hold_slip_m": hold_slip,
                "hold_duration_s": phases[-1]["duration_s"],
            }
            if result["stages"]["bilateral_grasp_hold"]["status"] != "PASS":
                raise RuntimeError("Bilateral pad contact or stable one-second hold gate failed")
            result["highest_gate"] = "BILATERAL CONTACT + 1 S HOLD"
            render(model, run, out / "x2_robotiq_grasp_hold.png", (0.28, -0.02, 0.89), 0.75, 135, -10)

            if not q_lift["gate_pass"]:
                result["stages"]["physical_50mm_lift"] = {
                    "status": "NOT RUN", "reason": "Source-limited short-lift IK failed",
                }
                result["stages"]["transfer_place_settle"] = {"status": "NOT RUN"}
                result["status"] = "FAIL"
                raise RuntimeError("Contact hold passed, but no source-limited short-lift endpoint exists")
            if not lift_waypoint_path_pass:
                result["stages"]["physical_50mm_lift"] = {
                    "status": "NOT RUN", "reason": "Cartesian vertical lift path did not have source-limited IK at every waypoint",
                }
                result["stages"]["transfer_place_settle"] = {"status": "NOT RUN"}
                result["status"] = "FAIL"
                raise RuntimeError("Contact hold passed, but the Cartesian vertical lift path is not source-valid")

            lift_arm_segments = [
                (f"short_lift_cartesian_{i:02d}", lift_waypoint_q[i], lift_waypoint_q[i + 1], False)
                for i in range(len(lift_waypoint_q) - 1)
            ]
            lift_corridor = static_loaded_corridor(
                model, run,
                lift_arm_segments,
                np.asarray(qids, dtype=int), bottle_freejoint_id, tcp_site_id,
                bottle_id,
            )
            json_write(raw_dir / "static_loaded_lift_corridor.json", lift_corridor)
            result["stages"]["static_loaded_lift_corridor"] = {
                "status": lift_corridor["status"],
                "sample_count": len(lift_corridor["samples"]),
                "failing_samples": [sample for sample in lift_corridor["samples"] if not sample["pass"]],
                "evidence": str(raw_dir / "static_loaded_lift_corridor.json"),
            }
            if lift_corridor["status"] != "PASS":
                result["stages"]["physical_50mm_lift"] = {
                    "status": "NOT RUN", "reason": "Measured grasp envelope has no clear static lift corridor",
                }
                result["stages"]["transfer_place_settle"] = {"status": "NOT RUN"}
                result["status"] = "FAIL"
                raise RuntimeError("Measured contact-loaded bottle envelope blocks the short-lift corridor")

            lift_phase = f"lift_{round(lift_command_m * 1000):03d}mm_cartesian_path"
            phases.append(run_segment(model, run, lift_phase, q_app_v, q_lift_v, 1.25, 255., 255., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame,
                                      q_waypoints=np.asarray(lift_waypoint_q)))
            lift_rows = [r for r in trace if r["phase"] == lift_phase]
            max_com = max(lift_rows, key=lambda row: row["bottle_com_z_m"])
            max_lift = float(max_com["bottle_com_z_m"] - initial_com[2])
            table_free_steps = sum(r["table_normal_force_n"] <= 1e-4 for r in lift_rows if r["bottle_com_z_m"] - initial_com[2] >= 0.01)
            lift_contact_fraction = sum(r["right_pad_contact_count"] > 0 and r["left_pad_contact_count"] > 0 for r in lift_rows) / max(1, len(lift_rows))
            result["stages"]["physical_50mm_lift"] = {
                "status": "PASS" if max_lift >= REQUIRED_LIFT_M and table_free_steps > 0 and lift_contact_fraction >= 0.80 else "FAIL",
                "required_bottle_com_lift_m": REQUIRED_LIFT_M,
                "commanded_tcp_lift_m": lift_command_m,
                "maximum_bottle_com_lift_m": max_lift,
                "maximum_lift_com_m": [max_com["bottle_com_x_m"], max_com["bottle_com_y_m"], max_com["bottle_com_z_m"]],
                "table_unsupported_lift_samples": int(table_free_steps),
                "bilateral_contact_fraction_during_lift": lift_contact_fraction,
            }
            result["stages"]["physical_30mm_mounted_lift"] = {
                "status": "PASS" if max_lift >= 0.030 and table_free_steps > 0 and lift_contact_fraction >= 0.80 else "FAIL",
                "diagnostic_threshold_m": 0.030,
                "maximum_bottle_com_lift_m": max_lift,
                "table_unsupported_lift_samples": int(table_free_steps),
                "bilateral_contact_fraction_during_lift": lift_contact_fraction,
            }
            render(model, run, out / "x2_robotiq_mounted_lift_peak.png", (0.28, -0.02, 0.92), 0.75, 135, -10)
            if result["stages"]["physical_30mm_mounted_lift"]["status"] != "PASS":
                raise RuntimeError("Bottle did not achieve the diagnostic 30 mm airborne lift with bilateral contact")

            phases.append(run_segment(model, run, "mounted_lift_hold_1s", q_lift_v, q_lift_v, 1.0, 255., 255., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            hold_lift_rows = [r for r in trace if r["phase"] == "mounted_lift_hold_1s"]
            hold_positions = np.asarray([
                [float(r["bottle_com_x_m"]), float(r["bottle_com_y_m"]), float(r["bottle_com_z_m"])]
                for r in hold_lift_rows
            ])
            hold_slip = float(np.max(np.linalg.norm(hold_positions - hold_positions[0], axis=1)))
            hold_bilateral_fraction = sum(
                int(r["right_pad_contact_count"]) > 0 and int(r["left_pad_contact_count"]) > 0
                for r in hold_lift_rows
            ) / max(1, len(hold_lift_rows))
            hold_min_lift = float(np.min(hold_positions[:, 2]) - initial_com[2])
            hold_table_clear = all(
                float(r["table_normal_force_n"]) <= 1e-4
                for r in hold_lift_rows if float(r["bottle_com_z_m"]) - initial_com[2] >= 0.01
            )
            lift_hold_pass = (
                hold_min_lift >= 0.030 and hold_bilateral_fraction >= 0.80
                and hold_table_clear and hold_slip <= 0.005
            )
            result["stages"]["mounted_lift_hold"] = {
                "status": "PASS" if lift_hold_pass else "FAIL",
                "duration_s": phases[-1]["duration_s"],
                "minimum_bottle_com_lift_m": hold_min_lift,
                "bottle_slip_during_hold_m": hold_slip,
                "bilateral_contact_fraction": hold_bilateral_fraction,
                "table_clear_after_lift": hold_table_clear,
            }
            result["highest_gate"] = (
                "PHYSICAL 50 MM LIFT + 1 S HOLD" if result["stages"]["physical_50mm_lift"]["status"] == "PASS"
                else "MOUNTED 30 MM LIFT + 1 S HOLD; 50 MM GATE NOT MET"
            )

            transfer_keys = ("high_lift", "transfer", "place", "retract")
            transfer_ik_pass = all(ik_results[key]["gate_pass"] for key in transfer_keys)
            if transfer_ik_pass:
                loaded_transport = static_loaded_corridor(
                    model, run,
                    [("high_lift", q_lift_v, q_high_v, False),
                     ("transfer", q_high_v, q_transfer_v, False),
                     ("lower_to_place", q_transfer_v, q_place_v, True)],
                    np.asarray(qids, dtype=int), bottle_freejoint_id, tcp_site_id,
                    bottle_id,
                )
            else:
                loaded_transport = {"status": "FAIL", "samples": [],
                                    "reason": "One or more source-limited high-lift/transfer/place/retract IK endpoints failed"}
            json_write(raw_dir / "static_loaded_transport_corridor.json", loaded_transport)
            result["stages"]["static_loaded_transport_corridor"] = {
                "status": loaded_transport["status"],
                "reason": loaded_transport.get("reason"),
                "sample_count": len(loaded_transport["samples"]),
                "failing_samples": [sample for sample in loaded_transport["samples"] if not sample["pass"]],
                "evidence": str(raw_dir / "static_loaded_transport_corridor.json"),
            }
            if loaded_transport["status"] != "PASS":
                phases.append(run_segment(model, run, "return_to_start_cartesian_path", q_lift_v, q_app_v, 1.25, 255., 255., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame,
                                          q_waypoints=np.asarray(lift_waypoint_q[::-1])))
                phases.append(run_segment(model, run, "release_at_start", q_app_v, q_app_v, 0.50, 255., 0., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "settle_at_start", q_app_v, q_app_v, 1.00, 0., 0., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                release_rows = [r for r in trace if r["phase"] == "release_at_start"]
                release_clear = bool(release_rows) and all(
                    row["right_pad_contact_count"] == 0 and row["left_pad_contact_count"] == 0
                    for row in release_rows[-max(1, len(release_rows) // 5):])
                final_com = run.xipos[bottle_id].copy()
                result["stages"]["safe_return_release"] = {
                    "status": "PASS" if release_clear else "FAIL",
                    "release_contact_loss": release_clear,
                    "final_bottle_com_m": final_com.tolist(),
                    "placement_target_reached": False,
                    "final_linear_speed_m_s": float(np.linalg.norm(run.qvel[int(model.jnt_dofadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")]):][:3])),
                }
                result["stages"]["transfer_place_settle"] = {
                    "status": "NOT RUN", "reason": "Loaded transport corridor or source-limited endpoint unavailable",
                }
                if result["stages"]["physical_50mm_lift"]["status"] == "PASS":
                    result["highest_gate"] = "PHYSICAL 50 MM LIFT + HOLD + RETURN/RELEASE; TRANSFER CORRIDOR BLOCKED"
                else:
                    result["highest_gate"] = "MOUNTED 30 MM LIFT + HOLD + RETURN/RELEASE; 50 MM AND TRANSFER NOT MET"
                result["status"] = (
                    "PARTIAL_PASS"
                    if result["stages"]["physical_30mm_mounted_lift"]["status"] == "PASS"
                    and result["stages"]["mounted_lift_hold"]["status"] == "PASS"
                    and release_clear
                    else "FAIL"
                )
            else:
                phases.append(run_segment(model, run, "high_lift", q_lift_v, q_high_v, 1.0, 255., 255., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "transfer", q_high_v, q_transfer_v, 1.5, 255., 255., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "lower", q_transfer_v, q_place_v, 1.25, 255., 255., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "release", q_place_v, q_place_v, 0.5, 255., 0., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "retract", q_place_v, np.asarray(q_retract["q_rad"]), 0.75, 0., 0., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                phases.append(run_segment(model, run, "settle", np.asarray(q_retract["q_rad"]), np.asarray(q_retract["q_rad"]), 1.5, 0., 0., base_targets,
                                          refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
                final_com = run.xipos[bottle_id].copy()
                final_q = run.xquat[bottle_id].copy()
                final_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
                final_dof = int(model.jnt_dofadr[final_joint])
                final_speed = float(np.linalg.norm(run.qvel[final_dof:final_dof+3]))
                final_spin = float(np.linalg.norm(run.qvel[final_dof+3:final_dof+6]))
                target_com_xy = (bottle_target_root + np.array([0., 0., -0.0246]))[:2]
                place_error = float(np.linalg.norm(final_com[:2] - target_com_xy))
                release_contacts = [r for r in trace if r["phase"] == "release"]
                release_clear = all(r["right_pad_contact_count"] == 0 and r["left_pad_contact_count"] == 0 for r in release_contacts[-max(1, len(release_contacts)//5):])
                rotation = Rotation.from_quat([final_q[1], final_q[2], final_q[3], final_q[0]])
                upright_error = float(np.linalg.norm(rotation.as_rotvec()[:2]))
                result["stages"]["transfer_place_settle"] = {
                    "status": "PASS" if release_clear and place_error <= 0.02 and upright_error <= math.radians(5) and final_speed <= 0.02 and final_spin <= 0.05 else "FAIL",
                    "release_contact_loss": release_clear, "final_bottle_com_m": final_com.tolist(),
                    "final_xy_error_m": place_error, "final_upright_error_rad": upright_error,
                    "final_linear_speed_m_s": final_speed, "final_angular_speed_rad_s": final_spin,
                    "settle_duration_s": phases[-1]["duration_s"],
                }
                if result["stages"]["transfer_place_settle"]["status"] == "PASS":
                    result["highest_gate"] = "TRANSFER, RELEASE, AND FREE SETTLE"
                    result["status"] = "PASS"
                else:
                    raise RuntimeError("Transfer/place/release/free-settle gate failed")
        finally:
            writer.close()
            renderer.close()
        result["stages"]["rollout_initialization"] = {"status": "PASS", "initial_contacts": initial_contacts,
                                                         "bottle_qpos_writes_after_initialization": 0,
                                                         "bottle_freejoint": True, "bottle_equalities": [], "bottle_mocap": False}
        result["phases"] = phases
        result["controller"] = {
            "model": "source X2 motor actuators controlled by simulation-derived bounded torque PD + qfrc_bias feedforward",
            "active_arm_joint_names": list(ARM), "actuator_mapping": refs,
            "maximum_motor_torque_nm": max_values["maximum_motor_torque_nm"],
            "maximum_arm_tracking_error_rad": max_values["max_arm_tracking_error_rad"],
            "qfrc_bias_feedforward_sign": "positive qfrc_bias compensation per M*qacc + qfrc_bias = qfrc_applied + qfrc_actuator + qfrc_constraint",
        }
        result["physical_integrity"] = {
            "bottle_dynamic_freejoint": True, "bottle_qpos_writes_after_initialization": 0,
            "bottle_weld_or_equality": False, "bottle_mocap_or_follow": False,
            "hidden_support": False, "left_hand_assistance": False,
            "bottle_geometry_mass_friction_unchanged": True,
        }
        result["maxima"] = max_values
        with (raw_dir / "physics_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(trace[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(trace)
        with (raw_dir / "contact_trace.jsonl").open("w", encoding="utf-8") as stream:
            for item in contact_trace:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
        result["evidence"] = {"output_dir": str(out), "model_xml": str(out / "x2_robotiq_2f85_simulation_only.xml"),
                               "physics_trace_csv": str(raw_dir / "physics_trace.csv"),
                               "contact_trace_jsonl": str(raw_dir / "contact_trace.jsonl"),
                               "video": str(out / "x2_robotiq_m0_physics.mp4"),
                               "static_images": sorted(str(p) for p in static_dir.glob("*.png")),
                               "result_json": str(out / "result.json")}
    except Exception as exc:
        result["status"] = "PARTIAL_PASS" if result.get("stages", {}).get("physical_50mm_lift", {}).get("status") == "PASS" else "FAIL"
        result["failure"] = {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()}
    if trace:
        with (raw_dir / "physics_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(trace[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(trace)
    if contact_trace:
        with (raw_dir / "contact_trace.jsonl").open("w", encoding="utf-8") as stream:
            for item in contact_trace:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    if phases:
        result["phases"] = phases
    if trace:
        result["maxima"] = max_values
        result["evidence"] = {"output_dir": str(out), "result_json": str(out / "result.json"),
                              "physics_trace_csv": str(raw_dir / "physics_trace.csv"),
                              "contact_trace_jsonl": str(raw_dir / "contact_trace.jsonl"),
                              "video": str(out / "x2_robotiq_m0_physics.mp4"),
                              "static_images": sorted(str(p) for p in static_dir.glob("*.png"))}
    json_write(out / "result.json", result)
    if result.get("status") not in {"PASS", "PARTIAL_PASS"}:
        print(json.dumps({"status": result.get("status"), "highest_gate": result.get("highest_gate"),
                          "failure": result.get("failure"), "stages": result.get("stages")}, indent=2))
        return 1
    print(json.dumps({"status": result["status"], "highest_gate": result["highest_gate"],
                      "result_json": str(out / "result.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
