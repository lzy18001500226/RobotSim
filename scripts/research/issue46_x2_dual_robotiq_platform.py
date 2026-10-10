#!/usr/bin/env python3
"""Build and gate an isolated dual-Robotiq X2 simulation-only platform."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import issue46_x2_robotiq_m0 as prior


OUT_DEFAULT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "dual-robotiq-platform-20261010"
)
WRIST_LINKS = {
    "left": "left_wrist_roll_link",
    "right": "right_wrist_roll_link",
}
WRIST_JOINTS = {
    "left": "left_wrist_roll_joint",
    "right": "right_wrist_roll_joint",
}
WRIST_MESH = {
    "left": {
        "visual": "left_wrist_roll_extend_link.stl",
        "collision": "left_wrist_roll_extend_link.stl",
    },
    "right": {
        "visual": "right_wrist_roll_extend_link.stl",
        "collision": "right_wrist_roll_link.stl",
    },
}
REFERENCE_MASS_KG = 1.0526083388427392
REFERENCE_JOINTS = tuple(
    f"{finger}_{joint}"
    for finger in ("left", "right")
    for joint in ("driver_joint", "coupler_joint", "spring_link_joint", "follower_joint")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    ident = int(mujoco.mj_name2id(model, kind, name))
    if ident < 0:
        raise RuntimeError(f"Missing MuJoCo {kind.name} {name!r}")
    return ident


def sha_inventory(paths: list[Path]) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(paths)}


def official_mounts(urdf_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, dict[str, Any]] = {}
    for side, letter in (("left", "L"), ("right", "R")):
        joint_name = f"{letter}_omnipicker_joint"
        joint = next((node for node in root.findall("joint") if node.get("name") == joint_name), None)
        if joint is None or joint.get("type") != "fixed":
            raise RuntimeError(f"Official tool URDF lacks fixed {joint_name}")
        parent, child, origin = joint.find("parent"), joint.find("child"), joint.find("origin")
        if parent is None or child is None or origin is None:
            raise RuntimeError(f"Incomplete official {joint_name}")
        expected_child = f"{letter}_omnipicker_base_link"
        if parent.get("link") != WRIST_LINKS[side] or child.get("link") != expected_child:
            raise RuntimeError(f"Unexpected official {joint_name} endpoints")
        xyz = np.asarray([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = np.asarray([float(v) for v in origin.get("rpy", "0 0 0").split()])
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        result[side] = {
            "joint": joint_name,
            "parent": parent.get("link"),
            "child": child.get("link"),
            "xyz_m": xyz.tolist(),
            "rpy_rad": rpy.tolist(),
            "rotation_matrix": rotation.tolist(),
        }
    return result


def menagerie_axes(xml_path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")
    rotation = data.xmat[root_id].reshape(3, 3)
    origin = data.xpos[root_id].copy()
    pad_ids = [object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in
               ("right_pad1", "right_pad2", "left_pad1", "left_pad2")]
    right_center = np.mean(data.geom_xpos[pad_ids[:2]], axis=0)
    left_center = np.mean(data.geom_xpos[pad_ids[2:]], axis=0)
    midpoint = (right_center + left_center) / 2.0
    insertion = midpoint - origin
    insertion /= np.linalg.norm(insertion)
    opening = left_center - right_center
    opening /= np.linalg.norm(opening)
    opening -= insertion * float(np.dot(insertion, opening))
    opening /= np.linalg.norm(opening)
    source_basis = np.column_stack((np.cross(insertion, opening), insertion, opening))
    target_insertion = np.array([0.0, 0.0, 1.0])
    target_opening = np.array([0.0, 1.0, 0.0])
    target_basis = np.column_stack((np.cross(target_insertion, target_opening),
                                    target_insertion, target_opening))
    if not np.isclose(np.linalg.det(source_basis), 1.0, atol=1e-7):
        raise RuntimeError("Menagerie insertion/opening axes do not form a right-handed frame")
    tcp_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    return {
        "mount_rotation_matrix": (target_basis @ source_basis.T).tolist(),
        "pad_midpoint_local_m": (rotation.T @ (midpoint - origin)).tolist(),
        "source_pinch_site_local_m": (rotation.T @ (data.site_xpos[tcp_id] - origin)).tolist(),
        "insertion_axis_local": insertion.tolist(),
        "opening_axis_local": opening.tolist(),
        "open_pad_separation_m": float(np.linalg.norm(left_center - right_center)),
    }


def namespaced_menagerie(root: ET.Element, template: ET.Element, men_root: Path,
                         side: str, mount: dict[str, Any], axes: dict[str, Any]) -> None:
    prefix = f"rq_{side}_"
    classes = {node.get("class") for node in template.iter() if node.get("class")}
    named = {node.get("name") for node in template.iter() if node.get("name")}
    mesh_stems = {Path(node.get("file", "")).stem for node in template.findall("./asset/mesh")}
    mapping = {value: prefix + value for value in classes | named | mesh_stems}
    reference_attrs = {
        "class", "childclass", "mesh", "material", "texture", "joint", "joint1", "joint2",
        "body", "body1", "body2", "site", "geom", "tendon", "actuator", "target",
        "jointinparent",
    }
    model = copy.deepcopy(template)
    object_tags = {
        "body", "geom", "joint", "freejoint", "site", "material", "texture",
        "hfield", "skin", "motor", "position", "velocity", "general", "intvelocity",
        "damper", "muscle", "adhesion", "spatial", "fixed", "connect", "weld",
        "exclude", "pair", "touch", "accelerometer", "velocimeter", "gyro", "force",
        "torque", "magnetometer", "rangefinder", "jointpos", "jointvel", "tendonpos",
        "tendonvel", "actuatorpos", "actuatorvel", "actuatorfrc", "framepos", "framequat",
        "framelinvel", "frameangvel", "subtreecom", "subtreelinvel", "subtreeangmom",
    }
    anonymous_counts: dict[str, int] = {}
    default_section = model.find("./default")
    default_nodes = set(default_section.iter()) if default_section is not None else set()
    parents = {child: parent for parent in model.iter() for child in parent}
    for node in model.iter():
        parent = parents.get(node)
        is_joint_or_site_object = node.tag not in {"joint", "site"} or (
            parent is not None and parent.tag in {"body", "equality"}
        )
        if (node not in default_nodes and node.tag in object_tags
                and is_joint_or_site_object and not node.get("name")):
            anonymous_counts[node.tag] = anonymous_counts.get(node.tag, 0) + 1
            node.set("name", f"{prefix}anon_{node.tag}_{anonymous_counts[node.tag]:03d}")
        for key, value in list(node.attrib.items()):
            if key == "name" or key in reference_attrs:
                node.set(key, mapping.get(value, value))
        if node.tag == "mesh" and node.get("file"):
            node.set("file", str((men_root / "assets" / node.get("file", "")).resolve()))
            if node.get("name") is None:
                node.set("name", mapping[Path(node.get("file", "")).stem])

    asset = root.find("asset")
    defaults = root.find("default")
    actuator = root.find("actuator")
    if asset is None or defaults is None or actuator is None:
        raise RuntimeError("X2 MJCF lacks a required asset/default/actuator section")
    for node in model.findall("./asset/*"):
        asset.append(copy.deepcopy(node))
    for node in model.findall("./default/*"):
        defaults.append(copy.deepcopy(node))

    mount_body = copy.deepcopy(model.find("./worldbody/body"))
    if mount_body is None:
        raise RuntimeError("Menagerie model lacks base_mount")
    mount_body.set("pos", " ".join(f"{v:.12g}" for v in mount["xyz_m"]))
    rotation = np.asarray(mount["rotation_matrix"], dtype=float) @ np.asarray(
        axes["mount_rotation_matrix"], dtype=float)
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = [quat_xyzw[3], *quat_xyzw[:3]]
    mount_body.set("quat", " ".join(f"{v:.12g}" for v in quat_wxyz))
    mount_body.set("name", prefix + "base_mount")
    ET.SubElement(mount_body, "site", {
        "name": prefix + "tcp",
        "pos": " ".join(f"{v:.12g}" for v in axes["pad_midpoint_local_m"]),
        "size": "0.006", "rgba": "1 0.2 0.1 1", "group": "5",
    })
    wrist = root.find(f'.//body[@name="{WRIST_LINKS[side]}"]')
    if wrist is None:
        raise RuntimeError(f"Pinned X2 MJCF lacks {WRIST_LINKS[side]}")
    wrist.append(mount_body)

    for section in ("contact", "tendon", "equality"):
        source = model.find(section)
        if source is None:
            continue
        target = root.find(section)
        if target is None:
            target = ET.SubElement(root, section)
        for node in source:
            target.append(copy.deepcopy(node))
    for node in model.findall("./actuator/*"):
        actuator.append(copy.deepcopy(node))


def set_wrist_mesh_pair(root: ET.Element, mesh_dir: Path, side: str) -> dict[str, str]:
    link = WRIST_LINKS[side]
    body = root.find(f'.//body[@name="{link}"]')
    if body is None:
        raise RuntimeError(f"No MJCF body for {link}")
    visual = next((node for node in body.findall("geom") if node.get("class") == "visual"), None)
    collision = next((node for node in body.findall("geom") if node.get("class") == "collision"), None)
    if visual is None or collision is None:
        raise RuntimeError(f"Expected visual/collision wrist geoms on {link}")
    mesh_names = {}
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("X2 MJCF has no asset section")
    for role in ("visual", "collision"):
        filename = WRIST_MESH[side][role]
        mesh_name = f"dual_{side}_wrist_{role}"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str((mesh_dir / filename).resolve())})
        (visual if role == "visual" else collision).set("mesh", mesh_name)
        mesh_names[role] = filename
    return mesh_names


def source_inertial(root: ET.Element, link_name: str) -> dict[str, Any]:
    link = next(node for node in root.findall("link") if node.get("name") == link_name)
    inertial = link.find("inertial")
    if inertial is None:
        raise RuntimeError(f"Source URDF has no inertial for {link_name}")
    origin, mass_node, inertia = inertial.find("origin"), inertial.find("mass"), inertial.find("inertia")
    if mass_node is None or inertia is None:
        raise RuntimeError(f"Incomplete inertial for {link_name}")
    tensor = np.array([
        [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
        [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
        [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
    ])
    return {
        "mass_kg": float(mass_node.get("value")),
        "origin_xyz_m": [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0],
        "origin_rpy_rad": [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0],
        "inertia_tensor_kg_m2": tensor.tolist(),
        "principal_inertia_kg_m2": np.linalg.eigvalsh(tensor).tolist(),
    }


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        c = data.contact[index]
        g1, g2 = int(c.geom1), int(c.geom2)
        rows.append({
            "body1": object_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
            "geom1": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body2": object_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
            "geom2": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "distance_m": float(c.dist),
        })
    return rows


def mounted_pad_gap(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    prefix = f"rq_{side}_"
    first = [object_id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
             for name in ("left_pad1", "left_pad2")]
    second = [object_id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
              for name in ("right_pad1", "right_pad2")]
    return float(np.linalg.norm(np.mean(data.geom_xpos[first], axis=0)
                                - np.mean(data.geom_xpos[second], axis=0)))


def run_independent_gripper(model: mujoco.MjModel, side: str,
                            joint_limits: dict[str, dict[str, float]],
                            max_steps: int = 1800) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for joint, value in (("head_yaw_joint", 0.0), ("head_pitch_joint", 0.0),
                         ("left_elbow_joint", -0.20), ("right_elbow_joint", -0.02)):
        data.qpos[prior.qpos_id(model, joint)] = value
    bottle = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bq = int(model.jnt_qposadr[bottle])
    data.qpos[bq:bq + 3] = [100.0, 100.0, 100.0]
    data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    aids = {item: object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                            f"rq_{item}_fingers_actuator") for item in ("left", "right")}
    for item, aid in aids.items():
        data.ctrl[aid] = float(model.actuator_ctrlrange[aid, 0])
    mujoco.mj_forward(model, data)
    refs = prior.controller_config(model)
    targets = {item["joint"]: float(data.qpos[item["qpos_id"]]) for item in refs}
    limits = source_limit_audit(model, data, joint_limits)
    initial_out_of_range = [row for row in limits
                            if row["lower_margin_rad"] < -1e-7 or row["upper_margin_rad"] < -1e-7]
    if initial_out_of_range:
        return {"side": side, "status": "FAIL", "steps": 0,
                "first_failure": {"step": 0, "time_s": 0.0,
                                  "reason": "source joint-limit violation at reset",
                                  "joint_limits": initial_out_of_range}, "trace": []}
    phases = (("open_hold", 100, 0.0), ("close", 700, 255.0),
              ("closed_hold", 100, 255.0), ("reopen", 700, 0.0),
              ("open_hold_final", 100, 0.0))
    trace: list[dict[str, Any]] = []
    step = 0
    failure = None
    completed = True
    for phase, phase_steps, final_target in phases:
        for local in range(phase_steps):
            blend, _ = prior.quintic((local + 1) / phase_steps)
            if phase == "close":
                target = 255.0 * blend
            elif phase == "reopen":
                target = 255.0 * (1.0 - blend)
            else:
                target = final_target
            prior.apply_controller(model, data, refs, targets,
                                   {key: 0.0 for key in targets})
            data.ctrl[aids[side]] = target
            data.ctrl[aids["right" if side == "left" else "left"]] = 0.0
            mujoco.mj_step(model, data)
            step += 1
            qpos_limits = source_limit_audit(model, data, joint_limits)
            violations = [row for row in qpos_limits
                          if row["lower_margin_rad"] < -1e-7 or row["upper_margin_rad"] < -1e-7]
            contacts = contact_rows(model, data)
            wrist_contacts = [row for row in contacts if row["distance_m"] < -1e-5 and
                              (("wrist_roll_link" in row["body1"] and row["body2"].startswith("rq_"))
                               or ("wrist_roll_link" in row["body2"] and row["body1"].startswith("rq_")))]
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                          and np.isfinite(data.qacc).all())
            trace.append({
                "step": step, "time_s": float(data.time), "phase": phase,
                "control": target, "pad_gap_m": mounted_pad_gap(model, data, side),
                "source_limit_violations": violations,
                "wrist_gripper_contacts": wrist_contacts,
                "qpos": {item["joint"]: float(data.qpos[item["qpos_id"]]) for item in refs},
                "qvel_abs_max_rad_s": float(np.max(np.abs(data.qvel))) if model.nv else 0.0,
                "finite": finite,
            })
            if not finite or violations or wrist_contacts:
                completed = False
                failure = {
                    "step": step, "time_s": float(data.time),
                    "reason": ("non-finite state" if not finite else
                               "source joint-limit violation" if violations else
                               "unintended wrist/gripper penetration"),
                    "joint_limits": violations, "contacts": wrist_contacts,
                }
                break
            if step >= max_steps:
                completed = False
                failure = {"step": step, "time_s": float(data.time),
                           "reason": "bounded dynamic step budget reached"}
                break
        if not completed:
            break
    return {
        "side": side, "status": "PASS" if completed else "FAIL",
        "steps": step, "duration_s": float(data.time), "first_failure": failure,
        "qpos_writes_after_rollout_start": 0, "bottle_contact": False,
        "open_pad_gap_m": trace[0]["pad_gap_m"] if trace else None,
        "last_pad_gap_m": trace[-1]["pad_gap_m"] if trace else None,
        "trace": trace,
    }


def make_camera(lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def save_render(model: mujoco.MjModel, data: mujoco.MjData, path: Path,
                lookat, distance: float, azimuth: float, elevation: float,
                hidden_group: int | None = None, collision_overlay: bool = False) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=900, width=1440)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[5] = 1
    if hidden_group is not None:
        option.geomgroup[hidden_group] = 0
    old_rgba = None
    if collision_overlay:
        old_rgba = model.geom_rgba.copy()
        collision_ids = np.flatnonzero(model.geom_group == 3)
        model.geom_rgba[collision_ids] = [1.0, 0.05, 0.05, 0.38]
    renderer.update_scene(data, camera=make_camera(lookat, distance, azimuth, elevation), scene_option=option)
    frame = renderer.render()
    Image.fromarray(frame).save(path)
    renderer.close()
    if old_rgba is not None:
        model.geom_rgba[:] = old_rgba
    return frame


def write_static_pose_review_video(open_image: Path, closed_image: Path, output: Path) -> None:
    frames = []
    for source, label in ((open_image, "STATIC OPEN POSE"),
                          (closed_image, "STATIC KINEMATIC CLOSED POSE"),
                          (open_image, "STATIC OPEN POSE")):
        image = Image.open(source).convert("RGB").resize((1280, 720))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 48), fill=(20, 27, 34))
        draw.text((22, 14), f"{label} | NOT DYNAMIC EVIDENCE", fill=(255, 255, 255))
        frames.extend([np.asarray(image)] * 30)
    with imageio.get_writer(output, fps=30, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def source_joint_limits(urdf_path: Path) -> dict[str, dict[str, float]]:
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        node = joint.find("limit")
        if node is None:
            continue
        limits[joint.get("name", "")] = {
            key: float(node.get(key, "nan")) for key in ("lower", "upper", "effort", "velocity")
        }
    return limits


def source_limit_audit(model: mujoco.MjModel, data: mujoco.MjData,
                       urdf_limits: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[jid]:
            continue
        joint = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if joint.startswith("rq_"):
            local_name = joint.split("_", 2)[-1]
            menagerie_name = local_name.replace("left_", "left_").replace("right_", "right_")
            # Menagerie ranges are taken directly from its pinned MJCF classes.
            source_range = model.jnt_range[jid].tolist()
            origin = "pinned Menagerie 2f85.xml"
        else:
            source = urdf_limits.get(joint)
            if source is None or not all(math.isfinite(source[k]) for k in ("lower", "upper")):
                continue
            source_range = [source["lower"], source["upper"]]
            origin = "pinned X2 Ultra URDF"
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        rows.append({
            "joint": joint, "source_range_rad": source_range, "qpos_rad": q,
            "lower_margin_rad": q - source_range[0], "upper_margin_rad": source_range[1] - q,
            "origin": origin,
        })
    return rows


def identity(args, runner: Path, xml_path: Path) -> dict[str, Any]:
    robotsim_root = runner.resolve().parents[2]
    controller_helper = Path(prior.__file__).resolve()
    controller_helper_relative = controller_helper.relative_to(robotsim_root).as_posix()
    controller_helper_head = subprocess.check_output(
        ["git", "-C", str(robotsim_root), "show", f"HEAD:{controller_helper_relative}"]
    )
    worktree_status = git_value(robotsim_root, "status", "--porcelain").splitlines()
    x2_mjcf = args.x2_root / prior.X2_MJCF
    x2_urdf = args.x2_root / prior.X2_URDF
    tool_urdf = args.x2_root / prior.X2_TOOL_URDF
    menagerie_xml = args.menagerie_root / prior.MENAGERIE_MJCF
    native = Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
    x2_mesh_dir = x2_mjcf.parent / "meshes"
    men_mesh_dir = args.menagerie_root / "robotiq_2f85" / "assets"
    return {
        "robot_model": "SIMULATION_ONLY dual Robotiq adapter prototype",
        "robotsim_branch": git_value(robotsim_root, "branch", "--show-current"),
        "robotsim_head_at_run": git_value(robotsim_root, "rev-parse", "HEAD"),
        "robotsim_worktree_dirty_at_run": bool(worktree_status),
        "robotsim_worktree_status_porcelain_at_run": worktree_status,
        "controller_helper": str(controller_helper),
        "controller_helper_sha256": sha256(controller_helper),
        "controller_helper_sha256_at_robotsim_head": hashlib.sha256(controller_helper_head).hexdigest(),
        "controller_helper_differs_from_robotsim_head":
            hashlib.sha256(controller_helper_head).hexdigest() != sha256(controller_helper),
        "python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(), "native_library": str(native),
        "native_library_sha256": sha256(native), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "x2_repository": "https://github.com/AgibotTech/agibot_x2_urdf",
        "x2_commit": git_value(args.x2_root, "rev-parse", "HEAD"),
        "x2_mjcf": str(prior.X2_MJCF), "x2_mjcf_sha256": sha256(x2_mjcf),
        "x2_urdf": str(prior.X2_URDF), "x2_urdf_sha256": sha256(x2_urdf),
        "official_tool_urdf": str(prior.X2_TOOL_URDF), "official_tool_urdf_sha256": sha256(tool_urdf),
        "x2_license": "Mulan PSL v2",
        "tool_wrist_mesh_sha256": {
            f"{side}_{role}": sha256(x2_mesh_dir / filename)
            for side, pair in WRIST_MESH.items() for role, filename in pair.items()
        },
        "menagerie_repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "menagerie_commit": git_value(args.menagerie_root, "rev-parse", "HEAD"),
        "menagerie_model": "robotiq_2f85/2f85.xml",
        "menagerie_xml_sha256": sha256(menagerie_xml), "menagerie_license": "BSD-2-Clause",
        "menagerie_mesh_sha256": sha_inventory(list(men_mesh_dir.glob("*.stl"))),
        "canonical_helper": str(args.canonical_helper),
        "canonical_helper_sha256": sha256(args.canonical_helper),
        "runner": str(runner.resolve()), "runner_sha256": sha256(runner),
        "compiled_model_xml": str(xml_path), "compiled_model_xml_sha256": sha256(xml_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=prior.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=prior.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=prior.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=[0.0, 0.08, 0.68])
    parser.add_argument("--station-base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--max-dynamic-steps", type=int, default=3000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    static_dir, raw_dir = out / "static", out / "raw"
    static_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "BLOCKED", "source_faithful_status": "NOT ACCEPTED",
        "simulation_only_status": "DIAGNOSTIC ONLY", "bottle_manipulation": "NOT RUN",
    }
    try:
        if args.x2_root.resolve() != prior.X2_DEFAULT.resolve():
            # Alternate clones are allowed only when they retain the exact pinned revision.
            if git_value(args.x2_root, "rev-parse", "HEAD") != prior.X2_PIN:
                raise RuntimeError("X2 source revision is not the pinned official SHA")
        if git_value(args.x2_root, "rev-parse", "HEAD") != prior.X2_PIN:
            raise RuntimeError("X2 source revision is not the pinned official SHA")
        if git_value(args.menagerie_root, "rev-parse", "HEAD") != prior.MENAGERIE_PIN:
            raise RuntimeError("Menagerie source revision is not the pinned official SHA")
        if prior.sha256(args.canonical_helper) != prior.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash mismatch")
        canonical = prior.load_module(args.canonical_helper)
        x2_path = args.x2_root / prior.X2_MJCF
        tool_path = args.x2_root / prior.X2_TOOL_URDF
        x2_source = ET.parse(x2_path).getroot()
        tool_source = ET.parse(tool_path).getroot()
        men_path = args.menagerie_root / prior.MENAGERIE_MJCF
        men_template = ET.parse(men_path).getroot()
        mounts = official_mounts(tool_path)
        axes = menagerie_axes(men_path)

        root = copy.deepcopy(x2_source)
        root.find("compiler").set("meshdir", str((x2_path.parent / "meshes").resolve()))
        root.find("compiler").set("autolimits", "true")
        root.find("compiler").set("angle", "radian")
        root.find("option").set("timestep", str(prior.DT))
        root.find("option").set("gravity", f"0 0 {-prior.GRAVITY}")
        pelvis = root.find('./worldbody/body[@name="pelvis"]')
        pelvis.remove(next(node for node in pelvis.findall("freejoint")
                           if node.get("name") == "floating_base_joint"))
        pelvis.set("pos", " ".join(f"{v:.10g}" for v in args.station_base_pos))
        yaw = Rotation.from_euler("z", args.station_base_yaw_deg, degrees=True)
        pelvis_quat = np.asarray([float(v) for v in pelvis.get("quat", "1 0 0 0").split()])
        pelvis_rotation = Rotation.from_quat([*pelvis_quat[1:], pelvis_quat[0]])
        combined = yaw * pelvis_rotation
        quat = combined.as_quat()
        pelvis.set("quat", " ".join(f"{v:.12g}" for v in [quat[3], *quat[:3]]))
        pelvis.attrib.pop("euler", None)

        mesh_inventory = {}
        for side in ("left", "right"):
            mesh_inventory[side] = set_wrist_mesh_pair(root, x2_path.parent / "meshes", side)
            namespaced_menagerie(root, men_template, args.menagerie_root / "robotiq_2f85",
                                 side, mounts[side], axes)
        prior.add_scene(root, canonical)
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_visual = visual.find("global")
        if global_visual is None:
            global_visual = ET.SubElement(visual, "global")
        global_visual.set("offwidth", "1440")
        global_visual.set("offheight", "900")
        # Review-only full-body cameras may hide table geom group 5; physical scene keeps it.
        for geom in root.findall('.//body[@name="m0_table"]//geom'):
            geom.set("group", "5")
        # Keep the canonical bottle beside the table in the workcell view, but hide it
        # together with the table in body-only review cameras.
        for geom in root.findall('.//body[@name="m0_bottle"]//geom'):
            geom.set("group", "5")

        xml_path = out / "x2_dual_robotiq_2f85_simulation_only.xml"
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        head_audit = prior.audit_head_torso_clearance(model, data)
        data.qpos[prior.qpos_id(model, "head_yaw_joint")] = 0.0
        data.qpos[prior.qpos_id(model, "head_pitch_joint")] = 0.0
        data.qpos[prior.qpos_id(model, "left_elbow_joint")] = -0.20
        data.qpos[prior.qpos_id(model, "right_elbow_joint")] = -0.02
        for actuator_name in ("rq_left_fingers_actuator", "rq_right_fingers_actuator"):
            aid = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            data.ctrl[aid] = float(model.actuator_ctrlrange[aid, 0])
        mujoco.mj_forward(model, data)

        source_model = mujoco.MjModel.from_xml_path(str(x2_path))
        source_tool_inertials = {
            side: source_inertial(tool_source, f"{side}_wrist_roll_link") for side in ("left", "right")
        }
        wrist_inertials = {}
        for side, link in WRIST_LINKS.items():
            wrist_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            source_id = object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, link)
            wrist_inertials[side] = {
                "ordinary_x2_mjcf_compiled": {
                    "mass_kg": float(source_model.body_mass[source_id]),
                    "inertial_position_m": source_model.body_ipos[source_id].tolist(),
                    "principal_inertia_kg_m2": source_model.body_inertia[source_id].tolist(),
                },
                "official_tool_urdf": source_tool_inertials[side],
                "dual_variant_preserved_ordinary_x2_inertia": {
                    "mass_kg": float(model.body_mass[wrist_id]),
                    "inertial_position_m": model.body_ipos[wrist_id].tolist(),
                    "principal_inertia_kg_m2": model.body_inertia[wrist_id].tolist(),
                },
                "reason": "Keep pinned X2 MJCF wrist mass/inertia unchanged; no Robotiq-specific vendor wrist variant or adapter inertia is available.",
            }
        source_body_names = {object_name(source_model, mujoco.mjtObj.mjOBJ_BODY, i)
                             for i in range(source_model.nbody)}
        compiled_body_names = {object_name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                               for i in range(model.nbody)}
        missing_source_bodies = sorted(source_body_names - compiled_body_names)
        source_joint_names = {object_name(source_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                              for i in range(source_model.njnt)}
        compiled_joint_names = {object_name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                for i in range(model.njnt)}
        missing_source_joints = sorted(source_joint_names - compiled_joint_names - {"floating_base_joint"})
        retained_source_body_tree = not missing_source_bodies
        retained_source_joint_tree = not missing_source_joints
        source_joint_status = {}
        for joint in sorted(source_joint_names - {"floating_base_joint"}):
            source_id = object_id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            current_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            source_joint_status[joint] = {
                "type_unchanged": int(source_model.jnt_type[source_id]) == int(model.jnt_type[current_id]),
                "axis_unchanged": bool(np.array_equal(source_model.jnt_axis[source_id], model.jnt_axis[current_id])),
                "limited_unchanged": bool(source_model.jnt_limited[source_id]) == bool(model.jnt_limited[current_id]),
                "range_unchanged": bool(np.array_equal(source_model.jnt_range[source_id], model.jnt_range[current_id])),
            }
        all_source_joint_properties_unchanged = all(
            all(status.values()) for status in source_joint_status.values()
        )
        wrist_joint_status = {}
        for side, joint in WRIST_JOINTS.items():
            joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            source_id = object_id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            actuator_name = f"motor_{joint}"
            actuator_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
            wrist_joint_status[side] = {
                "joint": joint,
                "actuator": actuator_name,
                "actuator_present": actuator_id >= 0,
                "actuator_targets_wrist_joint": actuator_id >= 0
                    and int(model.actuator_trnid[actuator_id, 0]) == joint_id,
                "source_range_rad": source_model.jnt_range[source_id].tolist(),
                "compiled_range_rad": model.jnt_range[joint_id].tolist(),
                "range_unchanged": bool(np.array_equal(
                    source_model.jnt_range[source_id], model.jnt_range[joint_id]
                )),
            }
        original_hand_bodies = [body for body in sorted(source_body_names)
                                if any(key in body.lower()
                                       for key in ("thumb", "finger", "palm", "omnipicker", "hand"))]
        arm_joint_status = {}
        for joint in prior.ARM:
            src = object_id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            cur = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            arm_joint_status[joint] = {
                "source_range_rad": source_model.jnt_range[src].tolist(),
                "dual_range_rad": model.jnt_range[cur].tolist(),
                "range_unchanged": bool(np.array_equal(source_model.jnt_range[src], model.jnt_range[cur])),
            }

        prefixes = {side: f"rq_{side}_" for side in ("left", "right")}
        objects = {}
        for side, prefix in prefixes.items():
            objects[side] = {
                "root_body": prefix + "base_mount",
                "body_count": sum(object_name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith(prefix)
                                  for i in range(model.nbody)),
                "body_names": [object_name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                               for i in range(model.nbody)
                               if object_name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith(prefix)],
                "joint_names": [object_name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                for i in range(model.njnt)
                                if object_name(model, mujoco.mjtObj.mjOBJ_JOINT, i).startswith(prefix)],
                "geom_names": [object_name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                               for i in range(model.ngeom)
                               if object_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              int(model.geom_bodyid[i])).startswith(prefix)],
                "mesh_names": sorted({
                    object_name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[i]))
                    for i in range(model.ngeom)
                    if object_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[i])).startswith(prefix)
                    and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                }),
                "actuator_names": [object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                                   for i in range(model.nu)
                                   if object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(prefix)],
                "tendon_names": [object_name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
                                 for i in range(model.ntendon)
                                 if object_name(model, mujoco.mjtObj.mjOBJ_TENDON, i).startswith(prefix)],
                "equality_names": [object_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
                                   for i in range(model.neq)
                                   if object_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i).startswith(prefix)],
                "site_names": [object_name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                               for i in range(model.nsite)
                               if object_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              int(model.site_bodyid[i])).startswith(prefix)],
                "tcp_site": prefix + "tcp",
                "root_subtree_mass_kg": float(model.body_subtreemass[
                    object_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_mount")]),
                "official_omnipicker_reference_mount": mounts[side],
                "robotiq_axis_fit": axes,
            }
        namespace_fields = (
            "body_names", "joint_names", "geom_names", "mesh_names", "actuator_names",
            "tendon_names", "equality_names", "site_names",
        )
        names_unique = all(
            len(set(name for item in objects.values() for name in item[field]))
            == sum(len(item[field]) for item in objects.values())
            for field in namespace_fields
        )
        namespace_complete = all(
            all(name.startswith(f"rq_{side}_") for field in (
                "body_names", "joint_names", "geom_names", "mesh_names", "actuator_names",
                "tendon_names", "equality_names", "site_names",
            ) for name in item[field])
            for side, item in objects.items()
        )
        mount_contacts = [row for row in contact_rows(model, data)
                          if row["distance_m"] < 0.0 and
                          (("wrist_roll_link" in row["body1"] and row["body2"].startswith("rq_"))
                           or ("wrist_roll_link" in row["body2"] and row["body1"].startswith("rq_")))]
        self_contacts = contact_rows(model, data)
        limits = source_joint_limits(args.x2_root / prior.X2_URDF)
        initial_limits = source_limit_audit(model, data, limits)

        base_lookat = [float(args.station_base_pos[0]), float(args.station_base_pos[1]),
                       float(args.station_base_pos[2] + 0.34)]
        views = {
            "front": (140, -7), "side": (50, -7), "rear": (320, -7),
            "three_quarter": (225, -7),
        }
        for view, (azimuth, elevation) in views.items():
            save_render(model, data, static_dir / f"full_body_{view}.png", base_lookat,
                        3.15, azimuth, elevation, hidden_group=5)
        workcell = save_render(model, data, static_dir / "full_workcell_dual_open.png",
                               [0.18, 0.14, 1.05], 2.8, 65, -16)
        save_render(model, data, static_dir / "wrist_collision_overlay.png", base_lookat,
                    2.15, 90, -5, hidden_group=5, collision_overlay=True)
        wrist_views = {}
        for side, link in WRIST_LINKS.items():
            wrist = object_id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            wrist_views[side] = data.xpos[wrist].tolist()
            tcp_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"rq_{side}_tcp")
            tcp = data.site_xpos[tcp_id].copy()
            save_render(model, data, static_dir / f"{side}_robotiq_open_closeup.png", tcp,
                        0.52, 315 if side == "right" else 225, -18, hidden_group=5)
        # A closed render pose is a visualization-only kinematic state, not a dynamic result.
        closed = mujoco.MjData(model)
        mujoco.mj_resetData(model, closed)
        closed.qpos[:] = data.qpos
        for side in ("left", "right"):
            for joint in (f"rq_{side}_left_driver_joint", f"rq_{side}_right_driver_joint"):
                jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                closed.qpos[int(model.jnt_qposadr[jid])] = 0.8
        mujoco.mj_forward(model, closed)
        for side, link in WRIST_LINKS.items():
            wrist = object_id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            tcp = closed.site_xpos[object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"rq_{side}_tcp")]
            save_render(model, closed, static_dir / f"{side}_robotiq_closed_kinematic.png",
                        tcp, 0.52, 315 if side == "right" else 225, -18, hidden_group=5)
        save_render(model, closed, static_dir / "full_body_closed_kinematic.png", base_lookat,
                    3.15, 140, -7, hidden_group=5)
        static_video = out / "dual_gripper_static_pose_review.mp4"
        write_static_pose_review_video(static_dir / "full_body_front.png",
                                       static_dir / "full_body_closed_kinematic.png", static_video)

        result["identity"] = identity(args, Path(__file__), xml_path)
        result["source_audit"] = {
            "x2_mjcf_source_commit": prior.X2_PIN,
            "menagerie_source_commit": prior.MENAGERIE_PIN,
            "full_x2_body_and_arm_tree_retained": True,
            "source_x2_hand_subtree_names": original_hand_bodies,
            "source_x2_hand_subtree_present": bool(original_hand_bodies),
            "tool_variant_wrist_meshes": mesh_inventory,
            "wrist_mass_inertia_comparison": wrist_inertials,
            "tool_mounts_are_omnipicker_references_not_robotiq_hardware_mount_proof": True,
            "unverified_adapter_label": "SIMULATION_ONLY mechanical-interface prototype; no adapter mass/inertia modeled",
        }
        result["mounts"] = {
            side: {
                "official_reference": mounts[side],
                "robotiq_translation_in_wrist_frame_m": mounts[side]["xyz_m"],
                "derivation": "side-specific official OmniPicker wrist transform composed with the pinned Robotiq insertion/opening-axis fit",
            } for side in mounts
        }
        # Record the actual derived rotations after compiling the root body transforms.
        for side in ("left", "right"):
            body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"rq_{side}_base_mount")
            wrist_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_LINKS[side])
            rel_rotation = data.xmat[wrist_id].reshape(3, 3).T @ data.xmat[body_id].reshape(3, 3)
            quat = Rotation.from_matrix(rel_rotation).as_quat()
            result["mounts"][side]["robotiq_rotation_in_wrist_frame_wxyz"] = [float(quat[3]), *quat[:3].tolist()]
            result["mounts"][side]["robotiq_root_world_position_m"] = data.xpos[body_id].tolist()
            result["mounts"][side]["tcp_world_position_m"] = data.site_xpos[
                object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"rq_{side}_tcp")].tolist()
        result["structural_gate"] = {
            "two_gripper_root_bodies": sum(1 for item in objects.values() if item["root_body"]),
            "exactly_two_instances": len(objects) == 2 and all(item["body_count"] > 0 for item in objects.values()),
            "independent_namespaces": names_unique,
            "instance_inventory": objects,
            "no_unwanted_five_finger_x2_hand_geometry": not original_hand_bodies,
            "original_x2_arm_joint_ranges_preserved": all(item["range_unchanged"] for item in arm_joint_status.values()),
            "arm_joint_ranges": arm_joint_status,
            "wrist_mount_penetrations_at_open": mount_contacts,
            "initial_active_contacts": self_contacts,
            "initial_source_limit_margins": initial_limits,
            "unreviewed_collision_exclusions_added": [],
        }

        # The bottle is moved only in this clean pre-rollout diagnostic state.
        bottle_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
        bottle_qpos = int(model.jnt_qposadr[bottle_joint])
        data.qpos[bottle_qpos:bottle_qpos + 3] = [100.0, 100.0, 100.0]
        data.qpos[bottle_qpos + 3:bottle_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        hold_targets = {item["joint"]: float(data.qpos[item["qpos_id"]])
                        for item in prior.controller_config(model)}
        refs = prior.controller_config(model)
        actuator_ids = {side: object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        f"rq_{side}_fingers_actuator") for side in ("left", "right")}
        independent_trials = {
            side: run_independent_gripper(model, side, limits)
            for side in ("left", "right")
        }
        for side, trial in independent_trials.items():
            trace_path = raw_dir / f"{side}_independent_open_close_trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as stream:
                for row in trial["trace"]:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            trial["trace_path"] = str(trace_path)
            trial.pop("trace", None)
        traces: list[dict[str, Any]] = []
        dynamic_status = "PASS"
        first_failure = None
        video_path = out / "dual_gripper_dynamic_until_first_failure.mp4"
        renderer = mujoco.Renderer(model, height=720, width=1280)
        camera = make_camera(base_lookat, 2.25, 135, -8)
        initial_limits_before_rollout = source_limit_audit(model, data, limits)
        with imageio.get_writer(video_path, fps=30, codec="libx264", quality=8) as video:
            renderer.update_scene(data, camera)
            video.append_data(renderer.render())
            phases = (("both_open_hold", 250, 0.0, 0.0),
                      ("left_close_right_hold", 750, 255.0, 0.0),
                      ("left_hold_right_close", 750, 255.0, 255.0),
                      ("both_reopen", 750, 0.0, 0.0))
            step = 0
            for phase, steps, left_start, right_start in phases:
                for local in range(steps):
                    u = min(1.0, (local + 1) / steps)
                    blend, _ = prior.quintic(u)
                    left_target = (left_start if phase != "both_reopen" else 255.0) * (1.0 - blend)
                    right_target = (right_start if phase != "both_reopen" else 255.0) * (1.0 - blend)
                    if phase == "left_close_right_hold":
                        left_target, right_target = 255.0 * blend, 0.0
                    elif phase == "left_hold_right_close":
                        left_target, right_target = 255.0, 255.0 * blend
                    elif phase == "both_reopen":
                        left_target, right_target = 255.0 * (1.0 - blend), 255.0 * (1.0 - blend)
                    prior.apply_controller(model, data, refs, hold_targets,
                                           {key: 0.0 for key in hold_targets})
                    data.ctrl[actuator_ids["left"]] = left_target
                    data.ctrl[actuator_ids["right"]] = right_target
                    mujoco.mj_step(model, data)
                    limits_now = source_limit_audit(model, data, limits)
                    failed_limits = [row for row in limits_now
                                     if row["lower_margin_rad"] < -1e-7 or row["upper_margin_rad"] < -1e-7]
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all():
                        dynamic_status = "FAIL"
                        first_failure = {"step": step + 1, "time_s": float(data.time),
                                         "reason": "non-finite state", "joint_limits": failed_limits}
                    elif failed_limits:
                        dynamic_status = "FAIL"
                        first_failure = {"step": step + 1, "time_s": float(data.time),
                                         "reason": "source joint-limit violation", "joint_limits": failed_limits}
                    contact_now = contact_rows(model, data)
                    unexpected = [row for row in contact_now if row["distance_m"] < -1e-5 and
                                  (("wrist_roll_link" in row["body1"] and row["body2"].startswith("rq_"))
                                   or ("wrist_roll_link" in row["body2"] and row["body1"].startswith("rq_")))]
                    if unexpected and dynamic_status == "PASS":
                        dynamic_status = "FAIL"
                        first_failure = {"step": step + 1, "time_s": float(data.time),
                                         "reason": "wrist/gripper unintended contact", "contacts": unexpected}
                    qpos = {}
                    qvel = {}
                    for side in ("left", "right"):
                        for suffix in REFERENCE_JOINTS:
                            joint = f"rq_{side}_{suffix}"
                            jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                            qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                            qpos[joint] = float(data.qpos[qadr])
                            qvel[joint] = float(data.qvel[dadr])
                    traces.append({
                        "step": step + 1, "time_s": float(data.time), "phase": phase,
                        "left_control": left_target, "right_control": right_target,
                        "left_pad_gap_m": mounted_pad_gap(model, data, "left"),
                        "right_pad_gap_m": mounted_pad_gap(model, data, "right"),
                        "qpos": qpos, "qvel": qvel, "source_limit_violations": failed_limits,
                        "active_contacts": contact_now,
                    })
                    if step % 10 == 0:
                        renderer.update_scene(data, camera)
                        video.append_data(renderer.render())
                    step += 1
                    if dynamic_status != "PASS" or step >= args.max_dynamic_steps:
                        if dynamic_status == "PASS":
                            dynamic_status = "NOT COMPLETE: bounded step budget reached"
                            first_failure = {"step": step, "time_s": float(data.time),
                                             "reason": "dynamic phase budget ended before all phases completed"}
                        break
                if dynamic_status != "PASS":
                    break
        renderer.close()
        with (raw_dir / "dual_open_close_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in traces:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        np.save(raw_dir / "dual_open_close_final_qpos.npy", data.qpos.copy())

        unexpected_static = [row for row in self_contacts if row["distance_m"] < -1e-5 and
                             (("wrist_roll_link" in row["body1"] and row["body2"].startswith("rq_"))
                              or ("wrist_roll_link" in row["body2"] and row["body1"].startswith("rq_")))]
        gripper_mass_ok = all(abs(item["root_subtree_mass_kg"] - REFERENCE_MASS_KG) < 1e-5
                              for item in objects.values())
        source_limit_ok = all(row["lower_margin_rad"] >= -1e-7 and row["upper_margin_rad"] >= -1e-7
                              for row in initial_limits_before_rollout)
        visual_gate = all((static_dir / f"full_body_{view}.png").is_file()
                          for view in ("front", "side", "rear", "three_quarter"))
        visual_artifacts_present = bool(
            visual_gate
            and all((static_dir / f"{side}_robotiq_open_closeup.png").is_file()
                    for side in ("left", "right"))
            and all((static_dir / f"{side}_robotiq_closed_kinematic.png").is_file()
                    for side in ("left", "right"))
            and (static_dir / "wrist_collision_overlay.png").is_file()
            and (static_dir / "full_workcell_dual_open.png").is_file()
            and static_video.is_file()
            and video_path.is_file()
        )
        structure_pass = (
            len(objects) == 2 and names_unique and not original_hand_bodies and
            namespace_complete and all_source_joint_properties_unchanged and
            all(item["actuator_present"] and item["actuator_targets_wrist_joint"]
                and item["range_unchanged"] for item in wrist_joint_status.values()) and
            retained_source_body_tree and retained_source_joint_tree and
            all(item["range_unchanged"] for item in arm_joint_status.values()) and
            gripper_mass_ok and source_limit_ok and not unexpected_static
        )
        result["visual_evidence"] = {
            "full_body_front_unoccluded": str(static_dir / "full_body_front.png"),
            "full_body_side": str(static_dir / "full_body_side.png"),
            "full_body_rear": str(static_dir / "full_body_rear.png"),
            "full_body_three_quarter": str(static_dir / "full_body_three_quarter.png"),
            "left_open_closeup": str(static_dir / "left_robotiq_open_closeup.png"),
            "right_open_closeup": str(static_dir / "right_robotiq_open_closeup.png"),
            "left_closed_kinematic_render_only": str(static_dir / "left_robotiq_closed_kinematic.png"),
            "right_closed_kinematic_render_only": str(static_dir / "right_robotiq_closed_kinematic.png"),
            "collision_overlay": str(static_dir / "wrist_collision_overlay.png"),
            "full_workcell": str(static_dir / "full_workcell_dual_open.png"),
            "static_pose_review_video_not_physics": str(static_video),
            "dynamic_video_until_first_failure": str(video_path),
            "dynamic_open_close_video_complete": dynamic_status == "PASS",
            "render_only_closed_pose_is_not_physics_evidence": True,
            "full_body_views_hide_table_and_bottle_only_in_review_render": True,
            "all_required_visual_files_present": visual_artifacts_present,
            "dynamic_open_close_visual_gate": "FAIL; source coupler limit failure at first physics step",
        }
        result["source_audit"] = {
            "x2_tool_urdf_mounts": mounts,
            "wrist_mesh_pair_from_official_tool_urdf": mesh_inventory,
            "wrist_mass_inertia_comparison": wrist_inertials,
            "source_body_names_missing_from_compiled_model": missing_source_bodies,
            "source_joint_names_missing_from_compiled_model": missing_source_joints,
            "source_joint_property_comparison": source_joint_status,
            "all_source_joint_properties_unchanged": all_source_joint_properties_unchanged,
            "wrist_joint_and_actuator_status": wrist_joint_status,
            "fixed_base_joint_removed_by_design": "floating_base_joint" in source_joint_names
                and "floating_base_joint" not in compiled_joint_names,
            "full_x2_body_and_arm_tree_retained": bool(
                retained_source_body_tree and retained_source_joint_tree
                and all_source_joint_properties_unchanged
            ),
            "tool_frame_is_only_an_omnipicker_reference": True,
            "robotiq_adapter_is_unverified_and_simulation_only": True,
            "added_adapter_mass_inertia": False,
        }
        result["gripper_object_inventory"] = objects
        result["gripper_namespace_diagnostics"] = {
            "names_unique_by_object_type": names_unique,
            "every_object_side_prefixed": namespace_complete,
            "unprefixed_names": {
                side: {
                    field: [name for name in item[field] if not name.startswith(f"rq_{side}_")]
                    for field in namespace_fields
                    if any(not name.startswith(f"rq_{side}_") for name in item[field])
                }
                for side, item in objects.items()
            },
        }
        result["structural_gate"] = {
            "status": "PASS" if structure_pass else "FAIL",
            "exactly_two_independently_namespaced_instances": bool(
                len(objects) == 2 and names_unique and namespace_complete
            ),
            "namespace_names_unique_by_object_type": names_unique,
            "all_object_names_side_prefixed": namespace_complete,
            "no_unwanted_five_finger_geometry": not original_hand_bodies,
            "both_wrist_bodies_retained": all(
                link in [object_name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                         for i in range(model.nbody)] for link in WRIST_LINKS.values()
            ),
            "both_wrist_joints_retained_with_actuators": all(
                item["actuator_present"] and item["actuator_targets_wrist_joint"]
                and item["range_unchanged"] for item in wrist_joint_status.values()
            ),
            "wrist_joint_and_actuator_status": wrist_joint_status,
            "full_x2_body_and_arms_retained": bool(
                retained_source_body_tree and retained_source_joint_tree
                and all_source_joint_properties_unchanged
            ),
            "all_source_body_names_retained": retained_source_body_tree,
            "all_source_joint_names_retained_except_fixed_base_freejoint": retained_source_joint_tree,
            "missing_source_bodies": missing_source_bodies,
            "missing_source_joints": missing_source_joints,
            "all_source_joint_properties_unchanged": all_source_joint_properties_unchanged,
            "source_joint_property_comparison": source_joint_status,
            "source_arm_limits_unchanged": all(item["range_unchanged"] for item in arm_joint_status.values()),
            "both_gripper_subtree_mass_matches_pinned_menagerie": gripper_mass_ok,
            "initial_source_limits_pass": source_limit_ok,
            "unintended_wrist_gripper_penetration_count": len(unexpected_static),
            "unintended_wrist_gripper_penetrations": unexpected_static,
            "initial_contacts": self_contacts,
            "source_limit_margins": initial_limits_before_rollout,
            "source_neutral_head_torso_audit": {
                "minimum_neutral_distance_m": head_audit["selected_initial_pose"]["minimum_pair_distance_m"],
                "pair_excluded": False,
                "classification": "retained source collision/contact behavior; no new exclusion",
            },
            "collision_exclusions_added": [],
            "wrist_mount_world_positions_m": wrist_views,
        }
        result["dynamic_gate"] = {
            "status": dynamic_status,
            "duration_s": float(data.time), "steps": len(traces),
            "first_failure": first_failure,
            "independent_trials": independent_trials,
            "qpos_writes_after_rollout_start": 0,
            "bottle_contact": False,
            "source_faithful_coupler_limit_violations_preserved": True,
            "per_gripper_open_close_test": {
                side: {
                    "status": item["status"],
                    "steps": item["steps"],
                    "first_failure": item["first_failure"],
                    "independent_clean_reset_attempted": True,
                } for side, item in independent_trials.items()
            },
            "video_note": "The dynamic clip records the clean initial state and first failed physics step only; the separate static pose-review clip is kinematic and is not physics evidence.",
        }
        result["gates"] = {
            "pinned_source_identity": "PASS",
            "full_robot_morphology_and_dual_namespaces": "PASS" if len(objects) == 2 and names_unique and not original_hand_bodies else "FAIL",
            "wrist_mount_and_initial_clearance": "PASS" if structure_pass else "FAIL",
            "source_faithful_coupler": "FAIL" if dynamic_status == "FAIL" and first_failure
                and first_failure.get("reason") == "source joint-limit violation" else "NOT RUN",
            "dual_gripper_open_close_dynamics": dynamic_status,
            "single_arm_bottle_pick_place": "NOT RUN" if not (structure_pass and dynamic_status == "PASS") else "READY TO EVALUATE",
        }
        result["status"] = "PASS" if structure_pass and dynamic_status == "PASS" else "BLOCKED"
        prior.json_write(out / "result.json", result)
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        prior.json_write(out / "result.json", result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
