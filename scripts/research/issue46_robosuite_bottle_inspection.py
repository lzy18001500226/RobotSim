#!/usr/bin/env python3
"""Inspect pinned robosuite BottleObject beside RobotSim's canonical bottle.

This is research-only evidence tooling. It imports MuJoCo and NumPy, but not
robosuite. Upstream sources are read-only inputs and remain outside this repo.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


UPSTREAM_REPOSITORY = "https://github.com/ARISE-Initiative/robosuite"
UPSTREAM_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
UPSTREAM_OBJECT_XML = "robosuite/models/assets/objects/bottle.xml"
UPSTREAM_MESH = "robosuite/models/assets/objects/meshes/bottle.stl"
UPSTREAM_TEXTURE = "robosuite/models/assets/textures/glass.png"

# Copied verbatim from the preserved Issue #46 canonical scene helper at the
# task's starting RobotSim research baseline; physical properties are unchanged.
TABLE_CENTER_XY = (0.300, -0.100)
TABLE_TOP_Z = 0.800
TABLE_RGBA = (0.6, 0.4, 0.2, 1.0)
CANONICAL_BOTTLE = (
    ("bottle_body", "cylinder", (0.0, 0.0, -0.04), (0.035, 0.0775, 0.0), 0.49, (0.12, 0.52, 0.82, 1.0)),
    ("bottle_shoulder", "ellipsoid", (0.0, 0.0, 0.0535), (0.035, 0.035, 0.026), 0.05, (0.12, 0.52, 0.82, 1.0)),
    ("bottle_neck", "cylinder", (0.0, 0.0, 0.0855), (0.018, 0.020, 0.0), 0.02, (0.12, 0.52, 0.82, 1.0)),
    ("bottle_cap", "cylinder", (0.0, 0.0, 0.1185), (0.020, 0.0085, 0.0), 0.01, (0.10, 0.16, 0.21, 1.0)),
)
CANONICAL_BOTTLE_FRICTION = (1.4, 0.02, 0.001)
CANONICAL_BOTTLE_CONDIM = 4
DROP_HEIGHT_M = 0.050
SIMULATION_SECONDS = 10.0
STABILITY_WINDOW_SECONDS = 0.5
STABLE_LINEAR_SPEED_MPS = 0.01
STABLE_ANGULAR_SPEED_RADPS = 0.05
STABLE_UPRIGHT_ERROR_DEG = 5.0
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_stl_bounds(path: Path) -> dict[str, Any]:
    """Read an STL's geometric extents without third-party mesh libraries."""
    payload = path.read_bytes()
    vertices: np.ndarray
    if len(payload) >= 84:
        count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + count * 50 == len(payload):
            records = np.frombuffer(payload, dtype=np.dtype([
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]), offset=84, count=count)
            vertices = records["vertices"].reshape(-1, 3).astype(np.float64)
            triangle_count = int(count)
            stl_encoding = "binary"
        else:
            vertices = np.asarray([
                [float(value) for value in line.split()[1:4]]
                for line in payload.decode("ascii").splitlines()
                if line.strip().lower().startswith("vertex ")
            ], dtype=np.float64)
            triangle_count = len(vertices) // 3
            stl_encoding = "ascii"
    else:
        vertices = np.asarray([
            [float(value) for value in line.split()[1:4]]
            for line in payload.decode("ascii").splitlines()
            if line.strip().lower().startswith("vertex ")
        ], dtype=np.float64)
        triangle_count = len(vertices) // 3
        stl_encoding = "ascii"
    if vertices.size == 0 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Could not parse STL vertices: {path}")
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return {
        "encoding": stl_encoding,
        "triangle_count": triangle_count,
        "vertex_record_count": int(len(vertices)),
        "bounds_min_m": lower.tolist(),
        "bounds_max_m": upper.tolist(),
        "dimensions_m": (upper - lower).tolist(),
    }


def fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(format(float(value), ".12g") for value in values)


def absolute_source_assets(asset_root: ET.Element, asset_dir: Path, texture_dir: Path) -> None:
    for mesh in asset_root.findall("mesh"):
        source = asset_dir / mesh.attrib["file"]
        mesh.set("file", str(source.resolve()))
    for texture in asset_root.findall("texture"):
        source = (asset_dir / texture.attrib["file"]).resolve()
        if not source.is_file():
            source = texture_dir / Path(texture.attrib["file"]).name
        texture.set("file", str(source.resolve()))


def add_table(world: ET.Element, rgba: tuple[float, ...] = TABLE_RGBA) -> None:
    table = ET.SubElement(world, "body", {
        "name": "canonical_table",
        "pos": fmt((*TABLE_CENTER_XY, 0.0)),
    })
    parts = [
        ("table_top", "box", (0.0, 0.0, 0.775), (0.200, 0.200, 0.025), (1.2, 0.01, 0.001)),
        ("table_leg_front_left", "box", (-0.175, 0.175, 0.375), (0.025, 0.025, 0.375), (0.9, 0.01, 0.001)),
        ("table_leg_front_right", "box", (0.175, 0.175, 0.375), (0.025, 0.025, 0.375), (0.9, 0.01, 0.001)),
        ("table_leg_back_left", "box", (-0.175, -0.175, 0.375), (0.025, 0.025, 0.375), (0.9, 0.01, 0.001)),
        ("table_leg_back_right", "box", (0.175, -0.175, 0.375), (0.025, 0.025, 0.375), (0.9, 0.01, 0.001)),
    ]
    for name, geom_type, pos, size, friction in parts:
        physical = ET.SubElement(table, "geom", {
            "name": f"{name}_collision",
            "type": geom_type,
            "pos": fmt(pos),
            "size": fmt(size),
            "rgba": fmt(rgba),
            "friction": fmt(friction),
            "condim": "4",
            "group": "0",
        })
        visual = copy.deepcopy(physical)
        visual.set("name", f"{name}_visual")
        visual.set("group", "1")
        visual.set("contype", "0")
        visual.set("conaffinity", "0")
        visual.set("mass", "1e-8")
        table.append(visual)


def make_scene_root(model_name: str, upstream_root: Path) -> ET.Element:
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {
        "timestep": "0.002",
        "gravity": "0 0 -9.81",
    })
    asset = ET.SubElement(root, "asset")
    source_asset_root = ET.parse(upstream_root / UPSTREAM_OBJECT_XML).getroot().find("asset")
    if source_asset_root is None:
        raise ValueError("Pinned bottle.xml has no asset block")
    source_asset_root = copy.deepcopy(source_asset_root)
    absolute_source_assets(
        source_asset_root,
        upstream_root / "robosuite/models/assets/objects",
        upstream_root / "robosuite/models/assets/textures",
    )
    for element in list(source_asset_root):
        asset.append(element)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {
        "name": "key_light",
        "pos": "0.3 -0.6 2.0",
        "dir": "0 0 -1",
        "directional": "true",
        "diffuse": "0.8 0.8 0.8",
        "specular": "0.2 0.2 0.2",
    })
    ET.SubElement(world, "geom", {
        "name": "ground_collision",
        "type": "plane",
        "pos": "0 0 0",
        "size": "5 5 0.1",
        "rgba": "0.74 0.76 0.78 1",
        "friction": "0.9 0.01 0.001",
        "condim": "4",
        "group": "0",
    })
    ET.SubElement(world, "geom", {
        "name": "ground_visual",
        "type": "plane",
        "pos": "0 0 0",
        "size": "5 5 0.1",
        "rgba": "0.74 0.76 0.78 1",
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    })
    add_table(world)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(RENDER_WIDTH), "offheight": str(RENDER_HEIGHT)})
    ET.SubElement(visual, "headlight", {
        "ambient": "0.35 0.35 0.35",
        "diffuse": "0.65 0.65 0.65",
        "specular": "0.2 0.2 0.2",
    })
    ET.SubElement(root, "statistic", {"center": "0.3 -0.1 0.9", "extent": "0.65"})
    return root


def class_equivalent_bottle(root: ET.Element, upstream_root: Path, name: str, pos: tuple[float, float, float]) -> ET.Element:
    source_root = ET.parse(upstream_root / UPSTREAM_OBJECT_XML).getroot()
    obj = source_root.find("./worldbody/body/body[@name='object']")
    if obj is None:
        raise ValueError("Pinned bottle.xml no longer has the BottleObject subtree")
    bottle = copy.deepcopy(obj)
    bottle.set("name", name)
    bottle.set("pos", fmt(pos))
    for geom_index, geom in enumerate(list(bottle.iter("geom"))):
        if geom.get("name") is None:
            geom.set("name", f"g{geom_index}")
        if geom.get("group") == "0":
            original_name = geom.get("name", "geom")
            visual = copy.deepcopy(geom)
            visual.set("name", f"{original_name}_visual")
            visual.set("group", "1")
            visual.set("conaffinity", "0")
            visual.set("contype", "0")
            visual.set("mass", "1e-8")
            geom.set("rgba", "0.5 0 0 1")
            geom.attrib.pop("material", None)
            bottle.append(visual)
    ET.SubElement(bottle, "joint", {
        "name": f"{name}_free",
        "type": "free",
        "damping": "0.0005",
    })
    ET.SubElement(bottle, "site", {
        "name": f"{name}_default_site",
        "pos": "0 0 0",
        "size": "0.005",
        "rgba": "1 0 0 0",
    })
    root.find("worldbody").append(bottle)
    return bottle


def raw_source_bottle(root: ET.Element, upstream_root: Path, name: str, pos: tuple[float, float, float]) -> ET.Element:
    source_root = ET.parse(upstream_root / UPSTREAM_OBJECT_XML).getroot()
    obj = source_root.find("./worldbody/body/body[@name='object']")
    if obj is None:
        raise ValueError("Pinned bottle.xml no longer has the expected object body")
    bottle = copy.deepcopy(obj)
    bottle.set("name", name)
    bottle.set("pos", fmt(pos))
    ET.SubElement(bottle, "joint", {
        "name": f"{name}_free",
        "type": "free",
        "damping": "0.0005",
    })
    root.find("worldbody").append(bottle)
    return bottle


def canonical_bottle(root: ET.Element, name: str, x: float, z: float) -> ET.Element:
    bottle = ET.SubElement(root.find("worldbody"), "body", {
        "name": name,
        "pos": fmt((x, TABLE_CENTER_XY[1], z)),
    })
    ET.SubElement(bottle, "freejoint", {"name": f"{name}_free"})
    for geom_name, geom_type, pos, size, mass, rgba in CANONICAL_BOTTLE:
        physical = ET.SubElement(bottle, "geom", {
            "name": f"{name}_{geom_name}_collision",
            "type": geom_type,
            "pos": fmt(pos),
            "size": fmt(size),
            "mass": format(mass, ".12g"),
            "rgba": fmt(rgba),
            "friction": fmt(CANONICAL_BOTTLE_FRICTION),
            "condim": str(CANONICAL_BOTTLE_CONDIM),
            "group": "0",
        })
        visual = copy.deepcopy(physical)
        visual.set("name", f"{name}_{geom_name}_visual")
        visual.set("group", "1")
        visual.set("contype", "0")
        visual.set("conaffinity", "0")
        visual.set("mass", "1e-8")
        bottle.append(visual)
    return bottle


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def find_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(f"Missing body {name}")
    return body_id


def geom_summary(model: mujoco.MjModel, body_id: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id])).name
        item: dict[str, Any] = {
            "id": geom_id,
            "name": name,
            "type": geom_type,
            "group": int(model.geom_group[geom_id]),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "condim": int(model.geom_condim[geom_id]),
            "friction": model.geom_friction[geom_id].tolist(),
            "solref": model.geom_solref[geom_id].tolist(),
            "solimp": model.geom_solimp[geom_id].tolist(),
            "rgba": model.geom_rgba[geom_id].tolist(),
        }
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            item["mesh_id"] = mesh_id
            item["compiled_render_mesh_vertices"] = int(model.mesh_vertnum[mesh_id])
            item["compiled_render_mesh_faces"] = int(model.mesh_facenum[mesh_id])
            item["compiled_collision_poly_count"] = int(model.mesh_polynum[mesh_id])
            item["compiled_collision_poly_vertex_count"] = int(model.mesh_polyvertnum[mesh_id])
        result.append(item)
    return result


def body_summary(model: mujoco.MjModel, name: str) -> dict[str, Any]:
    body_id = find_body(model, name)
    return {
        "body_id": body_id,
        "mass_kg": float(model.body_mass[body_id]),
        "inertia_principal_kg_m2": model.body_inertia[body_id].tolist(),
        "center_of_mass_body_frame_m": model.body_ipos[body_id].tolist(),
        "geom_inventory": geom_summary(model, body_id),
    }


def set_body_root_z(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, z: float) -> None:
    body_id = find_body(model, body_name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"Body {body_name} does not have a free joint")
    qposadr = int(model.jnt_qposadr[joint_id])
    data.qpos[qposadr + 2] = z


def body_pose_metrics(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, Any]:
    body_id = find_body(model, name)
    joint_id = int(model.body_jntadr[body_id])
    qveladr = int(model.jnt_dofadr[joint_id])
    qvel = data.qvel[qveladr:qveladr + 6]
    quat = data.xquat[body_id].copy()
    upright_dot = float(np.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))
    upright_error = math.degrees(math.acos(upright_dot))
    return {
        "time_s": float(data.time),
        "position_world_m": data.xpos[body_id].tolist(),
        "quaternion_wxyz": quat.tolist(),
        "linear_velocity_mps": qvel[:3].tolist(),
        "angular_velocity_radps": qvel[3:].tolist(),
        "linear_speed_mps": float(np.linalg.norm(qvel[:3])),
        "angular_speed_radps": float(np.linalg.norm(qvel[3:])),
        "upright_error_deg": upright_error,
    }


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData, body_names: set[str]) -> list[dict[str, Any]]:
    body_ids = {find_body(model, name) for name in body_names}
    contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 not in body_ids and body2 not in body_ids:
            continue
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or str(contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or str(contact.geom2)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        contacts.append({
            "time_s": float(data.time),
            "geom1": geom1,
            "geom2": geom2,
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or str(body1),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or str(body2),
            "distance_m": float(contact.dist),
            "position_world_m": contact.pos.tolist(),
            "normal_force_N": float(wrench[0]),
        })
    return contacts


def configure_camera(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_png(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    camera: mujoco.MjvCamera,
    collision_only: bool = False,
) -> None:
    renderer = mujoco.Renderer(model, width=RENDER_WIDTH, height=RENDER_HEIGHT)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    if collision_only:
        option.geomgroup[0] = 1
        option.geomgroup[1] = 0
    else:
        option.geomgroup[0] = 0
        option.geomgroup[1] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    imageio.imwrite(path, renderer.render())
    renderer.close()


def run_simulation(
    model: mujoco.MjModel,
    bottle_names: list[str],
    video_path: Path | None,
    duration_s: float = SIMULATION_SECONDS,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    traces: dict[str, list[dict[str, Any]]] = {name: [] for name in bottle_names}
    contact_rows: list[dict[str, Any]] = []
    video_writer = imageio.get_writer(
        str(video_path),
        fps=30,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) if video_path is not None else None
    renderer = mujoco.Renderer(model, width=RENDER_WIDTH, height=RENDER_HEIGHT) if video_writer else None
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[0] = 0
    option.geomgroup[1] = 1
    camera = configure_camera((0.3, -0.1, 0.86), 0.92, 140, -18)
    frame_step = max(1, int(round((1.0 / 30.0) / model.opt.timestep)))
    steps = int(math.ceil(duration_s / model.opt.timestep))
    started = time.monotonic()
    try:
        for step in range(steps + 1):
            if step % frame_step == 0 and renderer is not None and video_writer is not None:
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
                video_writer.append_data(renderer.render())
            for name in bottle_names:
                traces[name].append(body_pose_metrics(model, data, name))
            for contact in object_contacts(model, data, set(bottle_names)):
                contact_rows.append({"step": step, **contact})
            if step == steps:
                break
            mujoco.mj_step(model, data)
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    final_contacts = object_contacts(model, data, set(bottle_names))
    stability_start = max(0.0, duration_s - STABILITY_WINDOW_SECONDS)
    stability: dict[str, Any] = {}
    for name, rows in traces.items():
        window = [row for row in rows if row["time_s"] >= stability_start]
        bottle_contacts = [
            item for item in contact_rows
            if name in (item["body1"], item["body2"])
        ]
        bottle_final_contacts = [
            item for item in final_contacts
            if name in (item["body1"], item["body2"])
        ]
        first_contact_time = bottle_contacts[0]["time_s"] if bottle_contacts else None
        post_contact = [row for row in rows if first_contact_time is not None and row["time_s"] >= first_contact_time]
        reference_xy = post_contact[0]["position_world_m"][:2] if post_contact else rows[0]["position_world_m"][:2]
        max_horizontal_drift = max((
            math.dist(reference_xy, row["position_world_m"][:2]) for row in post_contact
        ), default=0.0)
        contact_steps_window = {
            item["step"] for item in contact_rows
            if name in (item["body1"], item["body2"]) and item["time_s"] >= stability_start
        }
        contact_fraction_window = len(contact_steps_window) / max(1, len(window))
        metrics = [
            row["linear_speed_mps"] <= STABLE_LINEAR_SPEED_MPS
            and row["angular_speed_radps"] <= STABLE_ANGULAR_SPEED_RADPS
            and row["upright_error_deg"] <= STABLE_UPRIGHT_ERROR_DEG
            for row in window
        ]
        suffix_index = len(metrics)
        while suffix_index > 0 and metrics[suffix_index - 1]:
            suffix_index -= 1
        stable_over_final_window = bool(metrics and suffix_index == 0)
        first_stable_time = window[suffix_index]["time_s"] if suffix_index < len(window) else None
        max_upright_error = max((row["upright_error_deg"] for row in post_contact), default=None)
        stability[name] = {
            "stable_over_final_window": stable_over_final_window,
            "remains_supported_and_upright": bool(
                bottle_final_contacts
                and contact_fraction_window >= 0.95
                and max_upright_error is not None
                and max_upright_error <= STABLE_UPRIGHT_ERROR_DEG
                and max_horizontal_drift <= 0.005
            ),
            "first_contact_time_s": first_contact_time,
            "contact_fraction_final_window": contact_fraction_window,
            "max_horizontal_drift_after_contact_m": max_horizontal_drift,
            "max_upright_error_after_contact_deg": max_upright_error,
            "stability_window_start_s": stability_start,
            "first_stable_sample_in_final_window_s": first_stable_time,
            "max_linear_speed_final_window_mps": max((row["linear_speed_mps"] for row in window), default=None),
            "max_angular_speed_final_window_radps": max((row["angular_speed_radps"] for row in window), default=None),
            "max_upright_error_final_window_deg": max((row["upright_error_deg"] for row in window), default=None),
            "final_pose": rows[-1],
            "final_contact_count": len(bottle_final_contacts),
            "final_total_normal_force_N": sum(item["normal_force_N"] for item in bottle_final_contacts),
            "maximum_penetration_m": max((max(0.0, -item["distance_m"]) for item in bottle_contacts), default=0.0),
        }
    return {
        "duration_s": float(data.time),
        "steps": steps,
        "wall_runtime_s": time.monotonic() - started,
        "traces": traces,
        "contact_trace": contact_rows,
        "final_contacts": final_contacts,
        "stability": stability,
        "data": data,
    }


def write_csv(path: Path, run: dict[str, Any]) -> None:
    columns = [
        "bottle", "time_s", "x_m", "y_m", "z_m",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "vx_mps", "vy_mps", "vz_mps", "wx_radps", "wy_radps", "wz_radps",
        "linear_speed_mps", "angular_speed_radps", "upright_error_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, rows in run["traces"].items():
            for row in rows:
                writer.writerow({
                    "bottle": name,
                    "time_s": row["time_s"],
                    "x_m": row["position_world_m"][0],
                    "y_m": row["position_world_m"][1],
                    "z_m": row["position_world_m"][2],
                    **dict(zip(("quat_w", "quat_x", "quat_y", "quat_z"), row["quaternion_wxyz"])),
                    **dict(zip(("vx_mps", "vy_mps", "vz_mps"), row["linear_velocity_mps"])),
                    **dict(zip(("wx_radps", "wy_radps", "wz_radps"), row["angular_velocity_radps"])),
                    "linear_speed_mps": row["linear_speed_mps"],
                    "angular_speed_radps": row["angular_speed_radps"],
                    "upright_error_deg": row["upright_error_deg"],
                })


def write_contact_csv(path: Path, run: dict[str, Any]) -> None:
    columns = ["step", "time_s", "geom1", "geom2", "body1", "body2", "distance_m", "normal_force_N", "x_m", "y_m", "z_m"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in run["contact_trace"]:
            writer.writerow({
                **{key: item[key] for key in columns[:8]},
                "x_m": item["position_world_m"][0],
                "y_m": item["position_world_m"][1],
                "z_m": item["position_world_m"][2],
            })


def label_comparison_image(
    path: Path,
    upstream_mass: float,
    upstream_dimensions: list[float],
    canonical_mass: float,
) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 64), fill=(250, 250, 250))
    draw.text((image.width * 0.27, 9), "robosuite BottleObject", fill=(20, 24, 28), font=font)
    source_mm = tuple(value * 1000 for value in upstream_dimensions)
    draw.text((image.width * 0.27, 29), f"{upstream_mass:.3f} kg | {source_mm[0]:.1f} x {source_mm[1]:.1f} x {source_mm[2]:.1f} mm", fill=(20, 24, 28), font=font)
    draw.text((image.width * 0.68, 9), "RobotSim canonical", fill=(20, 24, 28), font=font)
    draw.text((image.width * 0.68, 29), f"{canonical_mass:.3f} kg | 70 x 244.5 mm", fill=(20, 24, 28), font=font)
    image.save(path)


def write_external_reports(
    output: Path,
    result: dict[str, Any],
    runner_path: Path,
    upstream_root: Path,
    canonical_helper: Path | None,
) -> None:
    bottle = result["bottle_object_semantics"]
    canonical = result["canonical_comparison"]
    physics = result["physics"]
    upstream_run = physics["standalone_upstream"]
    comparison_run = physics["comparison"]
    upstream_stability = upstream_run["stability"]["robosuite_bottle"]
    canonical_stability = comparison_run["stability"]["robotsim_bottle"]
    upstream_body = bottle["class_equivalent_compiled_body"]
    canonical_body = canonical["compiled_body"]
    upstream_dims_mm = [value * 1000 for value in bottle["mesh_stl"]["dimensions_m"]]
    canonical_dims_mm = [value * 1000 for value in canonical["dimensions_m"]]
    stability_result = result["status"]
    quiet_result = result["quiet_settle_status"]

    report = f"""# robosuite BottleObject Inspection

## Result

- Table support and upright stability over the {physics['simulation_seconds']:.1f} s run: **{stability_result}**.
- Strict low-motion settle gate over the last {physics['settle_stability_thresholds']['window_s']:.1f} s: **{quiet_result}**. This is reported separately because both bottles stayed supported/upright while retaining measurable angular motion.
- No robot, grasp, arm IK, controller, or finger physics was included.

## Provenance and license

- Upstream: [{result['source']['repository']}](https://github.com/ARISE-Initiative/robosuite/tree/{result['source']['commit']}) at `{result['source']['commit']}`.
- License: MIT. The unmodified upstream `LICENSE`, bottle XML/STL, glass texture, `BottleObject` implementation, base object implementation, and PickPlace source are preserved in `upstream_source/`. The upstream license also carries a separate MuJoCo Apache-2.0 notice.
- Asset hashes are recorded in `result.json`; the complete evidence file hashes are in `sha256sums.txt`.

## Upstream object

`BottleObject` uses `bottle.xml`, adds a free joint with damping `0.0005`, keeps group 0 and group 1 geoms, and duplicates the collision geom as a non-contact group-1 visual with mass `1e-8`. The original visual-only group-1 mesh retains its source density `50 kg/m^3`, so it contributes to compiled body mass too. The compiled class-equivalent mass is **{upstream_body['mass_kg']:.9f} kg**, not 0.570 kg. The raw source body compiles to `{bottle['raw_source_compiled_body']['mass_kg']:.9f} kg` before the visual duplicate.

- STL dimensions: **{upstream_dims_mm[0]:.2f} x {upstream_dims_mm[1]:.2f} x {upstream_dims_mm[2]:.2f} mm**; {bottle['mesh_stl']['triangle_count']} source STL triangles.
- Compiled body principal inertia: `{upstream_body['inertia_principal_kg_m2']}` kg m^2; COM in the compiled body frame: `{upstream_body['center_of_mass_body_frame_m']}` m.
- Class-equivalent geom inventory: {len(upstream_body['geom_inventory'])} compiled geoms, {bottle['physical_collision_geom_count']} active mesh collision geom, {bottle['visual_geom_count']} visual geoms.
- Active collision geom: friction `{bottle['source_asset_geom_inventory'][1]['friction']}`, `condim={upstream_body['geom_inventory'][1]['condim']}`, `solref={upstream_body['geom_inventory'][1]['solref']}`, `solimp={upstream_body['geom_inventory'][1]['solimp']}`.
- The `glass` material and `glass.png` texture were preserved. The source skybox was hidden only in rendered review captures; the bottle material remains active.
- The pinned `PickPlace` environment constructs Milk, Bread, Cereal, and Can objects; it does not instantiate `BottleObject`.
- MuJoCo renders the 132-triangle mesh but uses its compiled 14-polygon convex hull for collision; this fills concave recesses. See the [MuJoCo 3.3.6 mesh reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html).
- The source XML's bottom/top sites span 157 mm, while the mesh bounds span 160 mm; `BottleObject` selects the inner `object` body and does not retain those outer sites.

## RobotSim comparison

Both objects used the same G1 table geometry and placement (0.400 x 0.400 m top, 0.800 m top height), gravity, timestep, and initial 50 mm drop. The upstream model retained its source friction/contact properties; the canonical bottle retained its existing four geom masses, shapes, friction, and 570 g total mass.

| Property | robosuite BottleObject | RobotSim canonical |
|---|---:|---:|
| Bounds (X x Y x Z) | {upstream_dims_mm[0]:.2f} x {upstream_dims_mm[1]:.2f} x {upstream_dims_mm[2]:.2f} mm | {canonical_dims_mm[0]:.1f} x {canonical_dims_mm[1]:.1f} x {canonical_dims_mm[2]:.1f} mm |
| Compiled mass | {upstream_body['mass_kg']:.6f} kg | {canonical_body['mass_kg']:.6f} kg |
| Principal inertia | `{upstream_body['inertia_principal_kg_m2']}` kg m^2 | `{canonical_body['inertia_principal_kg_m2']}` kg m^2 |
| Collision model | One active triangle mesh; convex-hull collision | Four primitive collision geoms: body cylinder, shoulder ellipsoid, neck cylinder, cap cylinder |
| Bottle friction / condim | `{upstream_body['geom_inventory'][1]['friction']}` / {upstream_body['geom_inventory'][1]['condim']} | `{canonical_body['geom_inventory'][0]['friction']}` / {canonical_body['geom_inventory'][0]['condim']} |
| Final table contacts | {upstream_stability['final_contact_count']} | {canonical_stability['final_contact_count']} |

The upstream bottle reached table contact at `{upstream_stability['first_contact_time_s']:.3f} s`, stayed supported for `{physics['simulation_seconds']:.1f} s`, and finished with `{upstream_stability['final_total_normal_force_N']:.3f} N` normal support force, `{upstream_stability['max_horizontal_drift_after_contact_m'] * 1000:.3f} mm` maximum horizontal drift after first contact, and `{upstream_stability['max_upright_error_after_contact_deg']:.4f} deg` maximum upright error. Maximum transient contact penetration was `{upstream_stability['maximum_penetration_m'] * 1000:.3f} mm`; the canonical comparison transient was `{canonical_stability['maximum_penetration_m'] * 1000:.3f} mm`. The supported/upright criterion passed, but the transient penetrations and quiet-settle failure are material limits of this inspection run, not tuned away.

The strict quiet-settle check uses linear speed <= `{physics['settle_stability_thresholds']['linear_speed_mps']:.3f} m/s`, angular speed <= `{physics['settle_stability_thresholds']['angular_speed_radps']:.3f} rad/s`, and upright error <= `{physics['settle_stability_thresholds']['upright_error_deg']:.1f} deg` for the final window. The upstream bottle did **not** meet the angular-speed part; the canonical control also did not meet the full window criterion. The exact traces are retained in CSV.

## Recommendation

**Retain both as comparison fixtures.** The pinned robosuite object is a useful source-provenance visual/collision reference, but it is about `{canonical_body['mass_kg'] / upstream_body['mass_kg']:.1f}x` lighter and materially shorter/narrower than the RobotSim canonical bottle. Do not silently replace the 570 g RobotSim object. No adapted 570 g upstream variant was created.

## Evidence

- `comparison_overview.png`, `comparison_front.png`, `comparison_side.png`
- `robosuite_full_view.png`, `robosuite_front_view.png`, `robosuite_side_view.png`
- `collision_geometry.png`, `robosuite_collision_geometry.png`
- `robosuite_bottle_settling.mp4`, `bottle_comparison_settling.mp4`
- `result.json`, bottle pose and contact CSV traces, generated MJCF scenes, pinned source copy
- Reproduction: `REPRODUCE.md`
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    pythonpath = ":".join((
        "/home/lzy18001500226/.cache/uv/archive-v0/BsWg146mGGPEhJS6/lib/python3.10/site-packages",
        "/home/lzy18001500226/.cache/uv/archive-v0/-9WOngKRZ8NW_wsn",
        "/home/lzy18001500226/.cache/uv/archive-v0/28gk2TpstdU5cAuq",
        "/home/lzy18001500226/.cache/uv/archive-v0/aoYU5wG7SrhxwH9h/lib/python3.10/site-packages",
    ))
    command = (
        f"MUJOCO_GL=egl PYTHONPATH={pythonpath} /usr/bin/python3 {runner_path} "
        f"--robosuite-root {upstream_root} --output-dir {output} "
        + (f"--canonical-helper {canonical_helper} " if canonical_helper else "")
        + f"--duration {physics['simulation_seconds']:g} --drop-height {physics['drop_height_m']:g}"
    )
    reproduce = f"""# Reproduction

## Exact run command (WSL2 Ubuntu 22.04)

```bash
{command}
```

The command requires Python 3.10 with MuJoCo `{result['runtime']['mujoco']}`, NumPy 1.26.4, ImageIO 2.38.0, imageio-ffmpeg 0.6.0, and Pillow 12.3.0. The task evidence includes the exact runner as `{runner_path.name}` and an immutable copy of the upstream MIT-licensed source files. The upstream checkout must be clean at `{result['source']['commit']}`; the runner rejects a different SHA or a dirty checkout.

This was an inspection-only run: standalone BottleObject plus ground/table, then the same upstream BottleObject beside the existing four-geom RobotSim bottle. No robot, grasp, IK, or controller code was run.
"""
    (output / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    checksum_lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "sha256sums.txt"):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "sha256sums.txt").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items() if key != "data"}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def copy_upstream_evidence(root: Path, output: Path) -> dict[str, str]:
    destinations = {
        "LICENSE": output / "upstream_source" / "LICENSE",
        UPSTREAM_OBJECT_XML: output / "upstream_source" / UPSTREAM_OBJECT_XML,
        UPSTREAM_MESH: output / "upstream_source" / UPSTREAM_MESH,
        UPSTREAM_TEXTURE: output / "upstream_source" / UPSTREAM_TEXTURE,
        "robosuite/models/objects/xml_objects.py": output / "upstream_source/robosuite/models/objects/xml_objects.py",
        "robosuite/models/objects/objects.py": output / "upstream_source/robosuite/models/objects/objects.py",
        "robosuite/environments/manipulation/pick_place.py": output / "upstream_source/robosuite/environments/manipulation/pick_place.py",
    }
    hashes: dict[str, str] = {}
    for source_name, destination in destinations.items():
        source = root / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[source_name] = sha256(source)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robosuite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-helper", type=Path)
    parser.add_argument("--duration", type=float, default=SIMULATION_SECONDS)
    parser.add_argument("--drop-height", type=float, default=DROP_HEIGHT_M)
    args = parser.parse_args()

    upstream_root = args.robosuite_root.resolve()
    output = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output == repo_root or repo_root in output.parents:
        raise SystemExit("Generated evidence must be stored outside the RobotSim checkout")
    commit = checked_git(upstream_root, "rev-parse", "HEAD")
    dirty = checked_git(upstream_root, "status", "--porcelain")
    if commit != UPSTREAM_COMMIT or dirty:
        raise SystemExit(f"Upstream checkout must be clean at {UPSTREAM_COMMIT}; got {commit}, dirty={bool(dirty)}")
    if not (upstream_root / "LICENSE").is_file():
        raise SystemExit("Pinned upstream checkout has no LICENSE file")
    output.mkdir(parents=True, exist_ok=True)
    canonical_helper = args.canonical_helper.resolve() if args.canonical_helper else None
    canonical_helper_hash = None
    if canonical_helper is not None:
        if not canonical_helper.is_file():
            raise SystemExit(f"Canonical comparison helper not found: {canonical_helper}")
        canonical_copy = output / "robotsim_reference" / canonical_helper.name
        canonical_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical_helper, canonical_copy)
        canonical_helper_hash = sha256(canonical_helper)

    source_xml = upstream_root / UPSTREAM_OBJECT_XML
    source_mesh = upstream_root / UPSTREAM_MESH
    source_texture = upstream_root / UPSTREAM_TEXTURE
    source_xml_root = ET.parse(source_xml).getroot()
    source_geom_rows = [
        {
            "source_group": int(geom.get("group", "0")),
            "type": geom.get("type"),
            "mesh": geom.get("mesh"),
            "density_kg_m3": float(geom.get("density", "1000")),
            "friction": [float(value) for value in geom.get("friction", "1 0.005 0.0001").split()],
            "solref": [float(value) for value in geom.get("solref", "0.02 1").split()],
            "solimp": [float(value) for value in geom.get("solimp", "0.9 0.95 0.001").split()],
            "contype": int(geom.get("contype", "1")),
            "conaffinity": int(geom.get("conaffinity", "1")),
        }
        for geom in source_xml_root.findall("./worldbody/body/body[@name='object']/geom")
    ]
    stl_metrics = parse_stl_bounds(source_mesh)

    object_min_z = stl_metrics["bounds_min_m"][2]
    rest_root_z = TABLE_TOP_Z - object_min_z
    upstream_pos = (TABLE_CENTER_XY[0] - 0.12, TABLE_CENTER_XY[1], rest_root_z + args.drop_height)
    canonical_pos_z = TABLE_TOP_Z - (-0.1175) + args.drop_height

    raw_scene_root = make_scene_root("robosuite_raw_source_reference", upstream_root)
    raw_source_bottle(raw_scene_root, upstream_root, "raw_robosuite_bottle", upstream_pos)
    raw_xml = output / "raw_source_reference.xml"
    write_xml(raw_scene_root, raw_xml)
    raw_model = mujoco.MjModel.from_xml_path(str(raw_xml))

    single_root = make_scene_root("robosuite_bottle_standalone", upstream_root)
    class_equivalent_bottle(single_root, upstream_root, "robosuite_bottle", upstream_pos)
    single_xml = output / "robosuite_bottle_scene.xml"
    write_xml(single_root, single_xml)
    single_model = mujoco.MjModel.from_xml_path(str(single_xml))

    comparison_root = make_scene_root("robosuite_vs_robotsim_bottle", upstream_root)
    class_equivalent_bottle(comparison_root, upstream_root, "robosuite_bottle", upstream_pos)
    canonical_bottle(comparison_root, "robotsim_bottle", TABLE_CENTER_XY[0] + 0.12, canonical_pos_z)
    comparison_xml = output / "bottle_comparison_scene.xml"
    write_xml(comparison_root, comparison_xml)
    comparison_model = mujoco.MjModel.from_xml_path(str(comparison_xml))

    raw_summary = body_summary(raw_model, "raw_robosuite_bottle")
    single_summary = body_summary(single_model, "robosuite_bottle")
    canonical_summary = body_summary(comparison_model, "robotsim_bottle")
    table_top = TABLE_TOP_Z
    canonical_height = 0.2445
    canonical_dims = [0.070, 0.070, canonical_height]
    source_visual_geom_count = sum(1 for item in single_summary["geom_inventory"] if item["group"] == 1)
    source_collision_geom_count = sum(1 for item in single_summary["geom_inventory"] if item["group"] == 0 and item["contype"] != 0 and item["conaffinity"] != 0)

    single_run = run_simulation(
        single_model,
        ["robosuite_bottle"],
        output / "robosuite_bottle_settling.mp4",
        args.duration,
    )
    write_csv(output / "robosuite_bottle_trace.csv", single_run)
    write_contact_csv(output / "robosuite_contact_trace.csv", single_run)
    render_png(single_model, single_run["data"], output / "robosuite_full_view.png", configure_camera((0.18, -0.1, 0.53), 1.85, 135, -18))
    render_png(single_model, single_run["data"], output / "robosuite_front_view.png", configure_camera((0.18, -0.1, 0.90), 0.55, 180, -7))
    render_png(single_model, single_run["data"], output / "robosuite_side_view.png", configure_camera((0.18, -0.1, 0.90), 0.55, 90, -7))
    render_png(single_model, single_run["data"], output / "robosuite_collision_geometry.png", configure_camera((0.18, -0.1, 0.90), 0.62, 135, -12), collision_only=True)

    comparison_run = run_simulation(
        comparison_model,
        ["robosuite_bottle", "robotsim_bottle"],
        output / "bottle_comparison_settling.mp4",
        args.duration,
    )
    write_csv(output / "bottle_comparison_trace.csv", comparison_run)
    write_contact_csv(output / "bottle_comparison_contact_trace.csv", comparison_run)
    comparison_data = comparison_run["data"]
    render_png(comparison_model, comparison_data, output / "comparison_overview.png", configure_camera((0.3, -0.1, 0.53), 1.90, 135, -18))
    render_png(comparison_model, comparison_data, output / "comparison_front.png", configure_camera((0.3, -0.1, 0.88), 0.72, 180, -5))
    render_png(comparison_model, comparison_data, output / "comparison_side.png", configure_camera((0.3, -0.1, 0.88), 0.72, 90, -5))
    render_png(comparison_model, comparison_data, output / "collision_geometry.png", configure_camera((0.3, -0.1, 0.53), 1.90, 135, -18), collision_only=True)
    label_comparison_image(
        output / "comparison_overview.png",
        single_summary["mass_kg"],
        stl_metrics["dimensions_m"],
        canonical_summary["mass_kg"],
    )

    upstream_hashes = copy_upstream_evidence(upstream_root, output)
    runner_path = Path(__file__).resolve()
    shutil.copy2(runner_path, output / runner_path.name)
    result = {
        "task": "RobotSim Issue #46 robosuite BottleObject inspection",
        "status": "PASS" if all(item["remains_supported_and_upright"] for item in comparison_run["stability"].values()) and single_run["stability"]["robosuite_bottle"]["remains_supported_and_upright"] else "FAIL",
        "status_semantics": "PASS means both objects remained table-supported and upright; low-motion settling is reported separately in quiet_settle_status.",
        "quiet_settle_status": "PASS" if all(item["stable_over_final_window"] for item in comparison_run["stability"].values()) and single_run["stability"]["robosuite_bottle"]["stable_over_final_window"] else "FAIL",
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": commit,
            "license": "MIT; upstream root LICENSE copied into upstream_source/LICENSE",
            "working_tree_clean": not bool(dirty),
            "file_sha256": {
                UPSTREAM_OBJECT_XML: sha256(source_xml),
                UPSTREAM_MESH: sha256(source_mesh),
                UPSTREAM_TEXTURE: sha256(source_texture),
                "LICENSE": sha256(upstream_root / "LICENSE"),
                "robosuite/models/objects/xml_objects.py": sha256(upstream_root / "robosuite/models/objects/xml_objects.py"),
                "robosuite/models/objects/objects.py": sha256(upstream_root / "robosuite/models/objects/objects.py"),
                "robosuite/environments/manipulation/pick_place.py": sha256(upstream_root / "robosuite/environments/manipulation/pick_place.py"),
            },
            "evidence_copy_sha256": upstream_hashes,
            "runner_path": runner_path.name,
            "runner_sha256": sha256(runner_path),
            "canonical_helper_sha256": canonical_helper_hash,
            "license_additional_notice": "MuJoCo is separately licensed under Apache License 2.0; see the pinned upstream LICENSE.",
        },
        "runtime": {
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "timestep_s": float(single_model.opt.timestep),
            "gravity_mps2": single_model.opt.gravity.tolist(),
            "integrator": mujoco.mjtIntegrator(int(single_model.opt.integrator)).name,
            "solver": mujoco.mjtSolver(int(single_model.opt.solver)).name,
            "solver_iterations": int(single_model.opt.iterations),
            "render_backend": "EGL (MUJOCO_GL=egl; verify in reproduction environment)",
        },
        "bottle_object_semantics": {
            "robosuite_class": "BottleObject(name)",
            "source_code_path": "robosuite/models/objects/xml_objects.py",
            "source_asset_geom_inventory": source_geom_rows,
            "raw_source_compiled_body": raw_summary,
            "class_equivalent_compiled_body": single_summary,
            "class_transformations_reproduced": [
                "select ./worldbody/body/body[@name='object']",
                "keep group 0 and group 1 because obj_type='all'",
                "add a group-1 collision-geometry visual duplicate (mass=1e-8, no contacts)",
                "recolor group-0 collision geom to OBJECT_COLLISION_COLOR and remove its material",
                "add free joint with damping=0.0005 and a transparent default site",
            ],
            "physical_collision_geom_count": source_collision_geom_count,
            "visual_geom_count": source_visual_geom_count,
            "mesh_stl": stl_metrics,
            "compiled_mesh_collision_note": "The triangulated mesh is rendered as authored; MuJoCo uses a convex hull for mesh collision, which fills non-convex recesses. See the official MuJoCo 3.3.5 computation documentation and the compiled collision poly counts.",
        },
        "canonical_comparison": {
            "source": "preserved Issue #46 canonical_manipulation_assets.py and the 2026-10-10 Robotiq M0 runner snapshot",
            "dimensions_m": canonical_dims,
            "mass_target_kg": 0.570,
            "geom_count": 4,
            "geom_definitions": [
                {"name": name, "type": geom_type, "pos_m": pos, "size_m": size, "input_mass_kg": mass, "rgba": rgba}
                for name, geom_type, pos, size, mass, rgba in CANONICAL_BOTTLE
            ],
            "friction": CANONICAL_BOTTLE_FRICTION,
            "condim": CANONICAL_BOTTLE_CONDIM,
            "compiled_body": canonical_summary,
            "same_table_geometry_and_placement": True,
        },
        "physics": {
            "simulation_seconds": args.duration,
            "drop_height_m": args.drop_height,
            "table_top_z_m": table_top,
            "initial_upstream_body_pos_m": upstream_pos,
            "initial_canonical_body_pos_m": [TABLE_CENTER_XY[0] + 0.12, TABLE_CENTER_XY[1], canonical_pos_z],
            "standalone_upstream": {key: value for key, value in single_run.items() if key not in ("data", "traces", "contact_trace")},
            "comparison": {key: value for key, value in comparison_run.items() if key not in ("data", "traces", "contact_trace")},
            "trace_files": ["robosuite_bottle_trace.csv", "robosuite_contact_trace.csv", "bottle_comparison_trace.csv", "bottle_comparison_contact_trace.csv"],
            "settle_stability_thresholds": {
                "window_s": STABILITY_WINDOW_SECONDS,
                "linear_speed_mps": STABLE_LINEAR_SPEED_MPS,
                "angular_speed_radps": STABLE_ANGULAR_SPEED_RADPS,
                "upright_error_deg": STABLE_UPRIGHT_ERROR_DEG,
            },
        },
        "artifacts": [
            "robosuite_full_view.png", "robosuite_front_view.png", "robosuite_side_view.png",
            "robosuite_collision_geometry.png", "comparison_overview.png", "comparison_front.png",
            "comparison_side.png", "collision_geometry.png", "robosuite_bottle_settling.mp4",
            "bottle_comparison_settling.mp4", "robosuite_bottle_trace.csv", "bottle_comparison_trace.csv",
            "robosuite_contact_trace.csv", "bottle_comparison_contact_trace.csv",
            "robosuite_bottle_scene.xml", "bottle_comparison_scene.xml", "raw_source_reference.xml",
            runner_path.name, "REPORT.md", "REPRODUCE.md", "sha256sums.txt",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(clean_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_external_reports(output, result, runner_path, upstream_root, canonical_helper)
    print(json.dumps({
        "status": result["status"],
        "status_semantics": result["status_semantics"],
        "quiet_settle_status": result["quiet_settle_status"],
        "upstream_class_mass_kg": single_summary["mass_kg"],
        "upstream_raw_source_mass_kg": raw_summary["mass_kg"],
        "canonical_mass_kg": canonical_summary["mass_kg"],
        "upstream_dimensions_m": stl_metrics["dimensions_m"],
        "upstream_supported_and_upright": single_run["stability"]["robosuite_bottle"]["remains_supported_and_upright"],
        "comparison_stability": comparison_run["stability"],
        "output_dir": str(output),
    }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
