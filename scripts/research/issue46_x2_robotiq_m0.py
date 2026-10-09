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
MAX_LIFT_M = 0.030
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


def add_scene(root: ET.Element, canonical) -> None:
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
    cx, cy = canonical.G1_CANONICAL_TABLE_CENTER_XY
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
    bottle = ET.SubElement(world, "body", {
        "name": "m0_bottle", "pos": " ".join(map(str, canonical.CANONICAL_X2_BOTTLE_START_BODY_POS)),
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
    root = ET.parse(x2_path).getroot()
    neutral_x2 = mujoco.MjModel.from_xml_path(str(x2_path))
    neutral_data = mujoco.MjData(neutral_x2)
    mujoco.mj_resetData(neutral_x2, neutral_data)
    mujoco.mj_forward(neutral_x2, neutral_data)
    neutral_wrist_id = obj_id(neutral_x2, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    neutral_wrist_rotation = neutral_data.xmat[neutral_wrist_id].reshape(3, 3).copy()
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
    option = root.find("option")
    option.set("timestep", str(DT))
    option.set("gravity", f"0 0 {-GRAVITY}")
    men = ET.parse(men_path).getroot()
    axes = robotiq_axes(men_path)
    desired_world_rotation = np.asarray(axes["mount_rotation_matrix"], dtype=float)
    mount_rotation_in_wrist = neutral_wrist_rotation.T @ desired_world_rotation
    mount_quat_xyzw = Rotation.from_matrix(mount_rotation_in_wrist).as_quat()
    axes["desired_world_rotation_matrix"] = desired_world_rotation.tolist()
    axes["neutral_wrist_rotation_world"] = neutral_wrist_rotation.tolist()
    axes["mount_rotation_in_right_wrist_matrix"] = mount_rotation_in_wrist.tolist()
    axes["mount_rotation_matrix"] = mount_rotation_in_wrist.tolist()
    axes["mount_quat_wxyz"] = np.array([mount_quat_xyzw[3], *mount_quat_xyzw[:3]]).tolist()
    namespace_menagerie(root, men, args.menagerie_root / "robotiq_2f85", axes)
    add_scene(root, canonical)
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
                refs, writer, trace, contact_trace, velocity_limits, max_values, save_frame) -> dict[str, Any]:
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
        qtarget = qstart + s * (qend - qstart)
        qdtarget = (dsdu / max(duration, 1e-12)) * (qend - qstart)
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
            for i in range(data.ncon):
                contact = data.contact[i]
                b1 = int(model.geom_bodyid[int(contact.geom1)])
                b2 = int(model.geom_bodyid[int(contact.geom2)])
                if {b1, b2} == {head_body, torso_body}:
                    pairs.append({"geom1_id": int(contact.geom1), "geom2_id": int(contact.geom2),
                                  "distance_m": float(contact.dist)})
            clearance = min((p["distance_m"] for p in pairs), default=None)
            item = {"head_yaw_rad": float(yaw), "head_pitch_rad": float(pitch),
                    "pair_contacts": pairs, "minimum_pair_distance_m": clearance,
                    "separated_beyond_contact_margin": not pairs}
            samples.append(item)
            if not pairs:
                safe.append(item)
            elif best is None or clearance > best["minimum_pair_distance_m"]:
                best = item
    if safe:
        selected = min(safe, key=lambda s: s["head_yaw_rad"]**2 + s["head_pitch_rad"]**2)
    elif best is not None:
        selected = best
    else:
        selected = {"head_yaw_rad": 0.0, "head_pitch_rad": 0.0, "minimum_pair_distance_m": None,
                    "separated_beyond_contact_margin": True}
    return {
        "source_joint_ranges_rad": {"head_yaw_joint": yaw_range.tolist(), "head_pitch_joint": pitch_range.tolist()},
        "collision_geom_ids": {"head_pitch_link": head_geoms, "torso_link": torso_geoms},
        "grid_shape": [101, 101], "sample_count": len(samples), "collision_free_samples": len(safe),
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
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    static_dir = out / "static"
    raw_dir = out / "raw"
    static_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    result: dict[str, Any] = {"experiment": "X2 + Robotiq 2F-85 — SIMULATION_ONLY",
                              "status": "BLOCKED", "highest_gate": "NONE", "stages": {}}
    result["runner"] = {
        "path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
        "station_base_position_world_m": list(args.station_base_pos),
    }
    try:
        if sha256(args.canonical_helper) != CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash differs from the accepted immutable helper")
        canonical = load_module(args.canonical_helper)
        ident = identity(args, args.canonical_helper)
        result["identity"] = ident
        model, adapter = build_model(args, out, canonical)
        result["adapter"] = adapter
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
        bottle_body_pos = np.asarray(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS, dtype=float)
        bottle_center = bottle_body_pos + np.array([0., 0., -0.04])
        pre_target = bottle_center + np.array([-0.12, 0., 0.])
        grasp_target = bottle_center.copy()
        arm_seed = data.qpos[qids].copy()
        ik_results = {}
        q_pre = solve_tcp_ik(model, data, pre_target, target_rot, arm_seed, qids, lower, upper)
        ik_results["pregrasp"] = q_pre
        result["ik"] = ik_results
        result["stages"]["source_limited_pregrasp_ik"] = {
            "status": "PASS" if q_pre["gate_pass"] else "FAIL",
            "position_error_m": q_pre["position_error_m"],
            "orientation_error_rad": q_pre["orientation_error_rad"],
            "joint_pose_rad": q_pre["q_rad"], "target_position_m": q_pre["target_position_m"],
        }
        if q_pre["gate_pass"]:
            result["highest_gate"] = "MOUNTED MODEL + SOURCE-LIMITED PREGRASP IK"
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        if not q_pre["gate_pass"]:
            raise RuntimeError(f"Source-limited PREGRASP IK failed: {q_pre['position_error_m']:.6f} m, {q_pre['orientation_error_rad']:.6f} rad")
        q_approach = solve_tcp_ik(model, data, grasp_target, target_rot, np.asarray(q_pre["q_rad"]), qids, lower, upper)
        ik_results["approach"] = q_approach
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        if not q_approach["gate_pass"]:
            raise RuntimeError(f"Source-limited APPROACH IK failed: {q_approach['position_error_m']:.6f} m, {q_approach['orientation_error_rad']:.6f} rad")
        q_lift = solve_tcp_ik(model, data, grasp_target + np.array([0., 0., MAX_LIFT_M]), target_rot,
                              np.asarray(q_approach["q_rad"]), qids, lower, upper)
        ik_results["lift_30mm"] = q_lift
        json_write(raw_dir / "ik_solver_trace.json", ik_results)
        if not q_lift["gate_pass"]:
            raise RuntimeError(f"Source-limited 30 mm lift IK failed: {q_lift['position_error_m']:.6f} m, {q_lift['orientation_error_rad']:.6f} rad")
        q_high = solve_tcp_ik(model, data, grasp_target + np.array([0., 0., 0.12]), target_rot,
                              np.asarray(q_lift["q_rad"]), qids, lower, upper)
        ik_results["high_lift"] = q_high
        q_transfer = solve_tcp_ik(model, data, grasp_target + np.array([0., -0.20, 0.12]), target_rot,
                                  np.asarray(q_high["q_rad"]), qids, lower, upper)
        ik_results["transfer"] = q_transfer
        q_place = solve_tcp_ik(model, data, grasp_target + np.array([0., -0.20, 0.]), target_rot,
                               np.asarray(q_transfer["q_rad"]), qids, lower, upper)
        ik_results["place"] = q_place
        q_retract = solve_tcp_ik(model, data, grasp_target + np.array([-0.12, -0.20, 0.]), target_rot,
                                 np.asarray(q_place["q_rad"]), qids, lower, upper)
        ik_results["retract"] = q_retract
        result["ik"] = ik_results
        json_write(raw_dir / "ik_solver_trace.json", ik_results)

        # Static collision checks at the no-contact path endpoints. The full
        # interpolation is rechecked immediately before the first rollout.
        path_checks = []
        bottle_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
        for label, q in (("home", np.zeros(len(ARM))), ("pregrasp", np.asarray(q_pre["q_rad"])),
                         ("approach", np.asarray(q_approach["q_rad"]))):
            data.qpos[qids] = q
            data.ctrl[obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")] = 0.0
            mujoco.mj_forward(model, data)
            rows = contacts(model, data)
            gate = static_contact_gate(rows)
            path_checks.append({"pose": label, "contacts": rows,
                                "unwanted_penetrations": gate["penetrating_contacts"], "pass": gate["pass"]})
        result["stages"]["static_ik_and_clearance"] = {
            "status": "PASS" if all(c["pass"] for c in path_checks) else "FAIL",
            "target_bottle_center_m": bottle_center.tolist(), "pregrasp_target_m": pre_target.tolist(),
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
            "motion": gripper_motion,
        }
        result["highest_gate"] = "MOUNTED OPEN/CLOSE + STATIC IK"
        render(model, dry, static_dir / "open_close_reopen.png", (0.1, -0.14, 0.65), 0.8, 135, -10)

        if result["stages"]["static_ik_and_clearance"]["status"] != "PASS":
            raise RuntimeError("Static source-limited IK or collision clearance gate failed")
        if result["stages"]["mounted_open_close"]["status"] != "PASS":
            raise RuntimeError("Mounted gripper OPEN/CLOSE validation failed")

        # Validate smooth robot path samples in scratch data before any physics rollout.
        path_samples = []
        for start_name, end_name, qa, qb in (
            ("home", "pregrasp", np.zeros(len(ARM)), np.asarray(q_pre["q_rad"])),
            ("pregrasp", "approach", np.asarray(q_pre["q_rad"]), np.asarray(q_approach["q_rad"])),
            ("approach", "lift30", np.asarray(q_approach["q_rad"]), np.asarray(q_lift["q_rad"])),
        ):
            for s in np.linspace(0.0, 1.0, 21):
                q = qa + s * (qb - qa)
                data.qpos[qids] = q
                data.ctrl[grip_id] = 0.0
                mujoco.mj_forward(model, data)
                rows = contacts(model, data)
                gate = static_contact_gate(rows)
                path_samples.append({"segment": f"{start_name}_to_{end_name}", "fraction": float(s), "contacts": rows,
                                     "unwanted_penetrations": gate["penetrating_contacts"], "pass": gate["pass"]})
        result["stages"]["static_path"] = {"status": "PASS" if all(s["pass"] for s in path_samples) else "FAIL",
                                               "sample_count": len(path_samples), "samples": path_samples}
        json_write(raw_dir / "static_path_samples.json", path_samples)
        if result["stages"]["static_path"]["status"] != "PASS":
            raise RuntimeError("Static approach/lift corridor contains an unintended collision")
        result["highest_gate"] = "STATIC APPROACH AND LIFT CORRIDOR"

        # From here onward, qpos is initialized once, before the first task step.
        run = mujoco.MjData(model)
        mujoco.mj_resetData(model, run)
        run.qpos[qpos_id(model, "head_yaw_joint")] = head_initial["head_yaw_rad"]
        run.qpos[qpos_id(model, "head_pitch_joint")] = head_initial["head_pitch_rad"]
        home = np.zeros(len(ARM))
        run.qpos[qids] = home
        run.ctrl[grip_id] = 0.0
        mujoco.mj_forward(model, run)
        initial_contacts = contacts(model, run, forces=True)
        initial_gate = static_contact_gate(initial_contacts)
        if not initial_gate["pass"]:
            result["stages"]["rollout_initialization"] = {"status": "FAIL", **initial_gate}
            raise RuntimeError("Initial robot/gripper/bottle/table state contains penetration")
        initial_com = run.xipos[bottle_id].copy()
        initial_bottle_quat = run.xquat[bottle_id].copy()
        result["initial_bottle_com_m"] = initial_com.tolist()
        result["initial_bottle_quaternion_wxyz"] = initial_bottle_quat.tolist()
        trace: list[dict[str, Any]] = []
        contact_trace: list[dict[str, Any]] = []
        max_values = {"minimum_bottle_contact_distance_m": 0.0, "maximum_motor_torque_nm": {}, "max_arm_tracking_error_rad": 0.0}
        phases = []
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
        q_home = home
        q_pre_v, q_app_v = np.asarray(q_pre["q_rad"]), np.asarray(q_approach["q_rad"])
        q_lift_v, q_high_v = np.asarray(q_lift["q_rad"]), np.asarray(q_high["q_rad"])
        q_transfer_v, q_place_v = np.asarray(q_transfer["q_rad"]), np.asarray(q_place["q_rad"])
        if not all(ik_results[k]["gate_pass"] for k in ("high_lift", "transfer", "place")):
            raise RuntimeError("Transfer/placement source-limited IK is not feasible")
        try:
            # Initial open settle, followed by the staged task.
            phases.append(run_segment(model, run, "settle", q_home, q_home, 0.75, 0., 0., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            phases.append(run_segment(model, run, "home_to_pregrasp", q_home, q_pre_v, 1.0, 0., 0., base_targets,
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
            phases.append(run_segment(model, run, "lift_30mm", q_app_v, q_lift_v, 0.75, 255., 255., base_targets,
                                      refs, writer, trace, contact_trace, velocity_limits, max_values, append_frame))
            max_com = run.xipos[bottle_id].copy()
            max_lift = float(max_com[2] - initial_com[2])
            table_free_steps = sum(r["table_normal_force_n"] <= 1e-4 for r in trace if r["phase"] == "lift_30mm" and r["bottle_com_z_m"] - initial_com[2] >= 0.01)
            result["stages"]["physical_30mm_lift"] = {
                "status": "PASS" if max_lift >= MAX_LIFT_M and table_free_steps > 0 else "FAIL",
                "maximum_bottle_com_lift_m": max_lift, "maximum_lift_com_m": max_com.tolist(),
                "table_unsupported_lift_samples": int(table_free_steps),
                "bilateral_contact_fraction_during_lift": sum(r["right_pad_contact_count"] > 0 and r["left_pad_contact_count"] > 0 for r in trace if r["phase"] == "lift_30mm") / max(1, sum(r["phase"] == "lift_30mm" for r in trace)),
            }
            render(model, run, out / "x2_robotiq_30mm_lift.png", (0.28, -0.02, 0.92), 0.75, 135, -10)
            if result["stages"]["physical_30mm_lift"]["status"] != "PASS":
                raise RuntimeError("Bottle did not achieve an airborne physical 30 mm COM lift")
            result["highest_gate"] = "PHYSICAL 30 MM LIFT"
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
            target_root = np.asarray(canonical.CANONICAL_X2_BOTTLE_TARGET_BODY_POS, dtype=float)
            place_error = float(np.linalg.norm(final_com[:2] - (target_root + np.array([0., 0., -0.0246]))[:2]))
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
        result["status"] = "FAIL"
        result["failure"] = {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()}
    json_write(out / "result.json", result)
    if result.get("status") != "PASS":
        print(json.dumps({"status": result.get("status"), "highest_gate": result.get("highest_gate"),
                          "failure": result.get("failure"), "stages": result.get("stages")}, indent=2))
        return 1
    print(json.dumps({"status": result["status"], "highest_gate": result["highest_gate"],
                      "result_json": str(out / "result.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
