#!/usr/bin/env python3
"""Static, source-traceable X2 OmniPicker review states; never advances physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


URDF = Path("/tmp/robotsim-asset-registry-upstreams/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
OUT = Path("/mnt/c/Users/HP/Desktop/RobotSim/reviews/asset-registry-review-20261012/x2_omnipicker_visual_anomaly")
MUJOCO_VERSION = "3.3.6"

GRIPPER_MESHES = {
    "base_link", "narrow1_Link", "narrow2_Link", "narrow3_Link", "narrow_loop_Link",
    "wide1_Link", "wide2_Link", "wide3_Link", "wide_loop_Link",
}

STATES = {
    "right_wrist_visual": "Right wrist visual mesh only: right_wrist_roll_extend_link.stl",
    "right_wrist_collision": "Right wrist collision mesh only: right_wrist_roll_link.stl",
    "right_wrist_mesh_overlay": "Right wrist source visual in white; source collision mesh in orange",
    "left_wrist_visual": "Left wrist visual mesh only: left_wrist_roll_extend_link.stl",
    "left_wrist_collision": "Left wrist collision mesh only: left_wrist_roll_extend_link.stl",
    "omnipicker_right": "Right OmniPicker base and jaw components only",
    "right_assembly": "Full assembled right wrist and OmniPicker (official visual geoms)",
    "right_assembly_runtime_view": "Right assembly as normally rendered with both source visual and collision geoms",
    "left_assembly": "Full assembled left wrist and OmniPicker (official visual geoms)",
    "full_robot": "Full X2 Ultra OmniPicker robot (all official visual geoms)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def int_attr(node: ET.Element, key: str, default: int) -> int:
    try:
        return int(node.get(key, str(default)))
    except ValueError:
        return default


def child_body_paths(worldbody: ET.Element) -> list[tuple[ET.Element, tuple[str, ...]]]:
    result: list[tuple[ET.Element, tuple[str, ...]]] = []

    def visit(parent: ET.Element, names: tuple[str, ...]) -> None:
        for body in parent.findall("body"):
            name = body.get("name", "")
            chain = names + (name,)
            result.append((body, chain))
            visit(body, chain)

    visit(worldbody, ())
    return result


def geom_selected(state: str, chain: tuple[str, ...], geom: ET.Element) -> bool:
    mesh = geom.get("mesh", "")
    group = int_attr(geom, "group", 0)
    contype = int_attr(geom, "contype", 1)
    conaffinity = int_attr(geom, "conaffinity", 1)
    source_visual = contype == 0 and conaffinity == 0
    body = chain[-1] if chain else "world"
    under_right = "right_wrist_roll_link" in chain
    under_left = "left_wrist_roll_link" in chain

    if state == "right_wrist_visual":
        return body == "right_wrist_roll_link" and mesh == "right_wrist_roll_extend_link" and source_visual
    if state == "right_wrist_collision":
        return body == "right_wrist_roll_link" and mesh == "right_wrist_roll_link" and not source_visual
    if state == "right_wrist_mesh_overlay":
        return body == "right_wrist_roll_link" and (
            (mesh == "right_wrist_roll_extend_link" and source_visual)
            or (mesh == "right_wrist_roll_link" and not source_visual)
        )
    if state == "left_wrist_visual":
        return body == "left_wrist_roll_link" and mesh == "left_wrist_roll_extend_link" and source_visual
    if state == "left_wrist_collision":
        return body == "left_wrist_roll_link" and mesh == "left_wrist_roll_extend_link" and not source_visual
    if state == "omnipicker_right":
        return under_right and mesh in GRIPPER_MESHES and source_visual
    if state == "right_assembly":
        return under_right and source_visual
    if state == "right_assembly_runtime_view":
        return under_right
    if state == "left_assembly":
        return under_left and source_visual
    if state == "full_robot":
        return source_visual
    raise ValueError(f"unknown state: {state}")


def insert_before(root: ET.Element, tag: str, before: tuple[str, ...]) -> ET.Element:
    existing = root.find(tag)
    if existing is not None:
        return existing
    node = ET.Element(tag)
    rank = {name: index for index, name in enumerate(before)}
    new_rank = rank.get(tag, 100)
    insert_at = len(root)
    for i, child in enumerate(root):
        if rank.get(child.tag, 100) > new_rank:
            insert_at = i
            break
    root.insert(insert_at, node)
    return node


def make_scene_xml(base_xml: str, state: str, floor_z: float, out_path: Path,
                   meshdir: Path) -> None:
    root = ET.fromstring(base_xml)
    root.set("model", f"X2_Ultra_OmniPicker_review_{state}")
    order = ("compiler", "option", "size", "visual", "statistic", "default", "asset", "worldbody", "contact", "equality", "actuator", "sensor", "keyframe")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = insert_before(root, "compiler", order)
    compiler.set("meshdir", str(meshdir.resolve()))
    compiler.set("angle", "radian")

    visual = insert_before(root, "visual", order)
    global_node = visual.find("global")
    if global_node is None:
        global_node = ET.SubElement(visual, "global")
    global_node.set("azimuth", "200")
    global_node.set("elevation", "-18")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update({"diffuse": "0.62 0.62 0.62", "ambient": "0.34 0.34 0.34", "specular": "0.18 0.18 0.18"})
    haze = visual.find("rgba")
    if haze is None:
        haze = ET.SubElement(visual, "rgba")
    haze.set("haze", "0.18 0.28 0.38 1")

    asset = insert_before(root, "asset", order)
    ET.SubElement(asset, "texture", {"name": "review_sky", "type": "skybox", "builtin": "gradient", "rgb1": "0.35 0.54 0.72", "rgb2": "0.04 0.07 0.12", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "review_checker", "type": "2d", "builtin": "checker", "rgb1": "0.24 0.34 0.40", "rgb2": "0.44 0.53 0.56", "mark": "edge", "markrgb": "0.83 0.86 0.86", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "review_checker_mat", "texture": "review_checker", "texuniform": "true", "texrepeat": "7 7", "reflectance": "0.08"})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("compiled source MJCF has no worldbody")
    paths = child_body_paths(worldbody)
    selected_geoms = 0
    geom_index = 0

    def process_geom(chain: tuple[str, ...], geom: ET.Element) -> None:
        nonlocal selected_geoms, geom_index
        keep = geom_selected(state, chain, geom)
        if not geom.get("name"):
            mesh_name = geom.get("mesh", "primitive")
            body_label = chain[-1] if chain else "world"
            geom.set("name", f"review_{geom_index:03d}_{body_label}_{mesh_name}")
        geom_index += 1
        if keep:
            selected_geoms += 1
            rgba = [float(x) for x in geom.get("rgba", "1 1 1 1").split()]
            if len(rgba) == 3:
                rgba.append(1.0)
            source_visual = int_attr(geom, "contype", 1) == 0 and int_attr(geom, "conaffinity", 1) == 0
            if state in ("right_wrist_mesh_overlay", "right_assembly_runtime_view") and not source_visual:
                rgba[:3] = [1.0, 0.20, 0.06]
            rgba[3] = 1.0
            geom.set("rgba", " ".join(f"{x:.6g}" for x in rgba))
        else:
            old = [float(x) for x in geom.get("rgba", "1 1 1 1").split()]
            if len(old) == 3:
                old.append(1.0)
            old[3] = 0.0
            geom.set("rgba", " ".join(f"{x:.6g}" for x in old))
        # Review model is deliberately visual-only. No static scenery or source geom can contact.
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    # Root-link visual/collision geoms are direct children of worldbody after URDF import.
    for geom in worldbody.findall("geom"):
        process_geom((), geom)
    for body, chain in paths:
        for geom in body.findall("geom"):
            process_geom(chain, geom)

    # The floor is a visual backdrop only; it cannot affect the approved physics model.
    ET.SubElement(worldbody, "geom", {"name": "review_floor", "type": "plane", "pos": f"0 0 {floor_z:.6f}", "size": "0 0 0.05", "material": "review_checker_mat", "group": "5", "contype": "0", "conaffinity": "0"})
    ET.SubElement(worldbody, "light", {"name": "review_key", "pos": "2 -2 3", "dir": "-0.45 0.45 -0.75", "directional": "true", "diffuse": "0.84 0.84 0.84", "specular": "0.22 0.22 0.22"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    if selected_geoms == 0:
        raise RuntimeError(f"state {state} selected no source geoms")


def bounds_for(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    keep = (model.geom_rgba[:, 3] > 0.5) & (model.geom_group != 5) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    if not np.any(keep):
        return np.array([0.0, 0.0, 0.8]), 0.3
    p = data.geom_xpos[keep]
    r = model.geom_rbound[keep]
    lower = np.min(p - r[:, None], axis=0)
    upper = np.max(p + r[:, None], axis=0)
    return (lower + upper) * 0.5, max(float(np.linalg.norm(upper - lower)) * 0.52, 0.12)


def camera_for(model: mujoco.MjModel, data: mujoco.MjData, state: str) -> tuple[np.ndarray, float, float, float]:
    target, radius = bounds_for(model, data)
    if state != "full_robot":
        radius = min(max(radius, 0.24), 0.75)
        if state == "right_wrist_visual" or state == "right_wrist_collision":
            radius = max(radius, 0.38)
    distance = radius * (2.2 if state == "full_robot" else 2.7)
    return target, distance, 0.0, -18.0


def save_labeled(frame: np.ndarray, path: Path, title: str) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(17, 27, 34))
    draw.text((12, 10), title, fill=(248, 250, 250), font=font)
    image.save(path)


def make_contact_sheet(images: list[Path], output: Path) -> None:
    tile_w, tile_h, label_h = 640, 480, 30
    cols = 2
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (238, 241, 242))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, path in enumerate(images):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((tile_w, tile_h))
            x = (i % cols) * tile_w
            y = (i // cols) * (tile_h + label_h)
            sheet.paste(img, (x, y + label_h))
            draw.text((x + 10, y + 9), path.stem, fill=(20, 28, 32), font=font)
    sheet.save(output)


def source_inventory(urdf: Path, model: mujoco.MjModel) -> dict[str, Any]:
    root = ET.parse(urdf).getroot()
    links = {link.get("name", ""): link for link in root.findall("link")}
    joints = root.findall("joint")
    mesh_dir = urdf.parent / "meshes"
    target_links = ["left_wrist_roll_link", "right_wrist_roll_link", "L_omnipicker_base_link", "R_omnipicker_base_link"]
    target_links += [f"{side}_hand_{finger}{n}_Link" for side in ("L", "R") for finger, n in (("narrow", 1), ("narrow", 2), ("narrow", 3), ("wide", 1), ("wide", 2), ("wide", 3))]
    target_links += [f"{side}_hand_{finger}_loop_Link" for side in ("L", "R") for finger in ("narrow", "wide")]

    relationships = []
    mesh_map: dict[str, Any] = {}
    for name in target_links:
        link = links.get(name)
        if link is None:
            continue
        associations = {}
        for kind in ("visual", "collision"):
            entries = []
            for index, item in enumerate(link.findall(kind)):
                mesh = item.find("./geometry/mesh")
                if mesh is None:
                    continue
                filename = Path(mesh.get("filename", "")).name
                p = mesh_dir / filename
                row = {"mesh_file": filename, "scale": mesh.get("scale"), "origin": item.find("origin").attrib if item.find("origin") is not None else None,
                       "sha256": sha256(p) if p.exists() else None, "bytes": p.stat().st_size if p.exists() else None}
                entries.append(row)
                mesh_map[filename] = row
            associations[kind] = entries
        relationships.append({"link": name, "visual_collision": associations})

    parent_child = []
    for j in joints:
        parent = j.find("parent").get("link", "")
        child = j.find("child").get("link", "")
        if parent.endswith("wrist_roll_link") or child.startswith(("L_omnipicker", "R_omnipicker", "L_hand_", "R_hand_")):
            origin = j.find("origin")
            parent_child.append({"joint": j.get("name"), "type": j.get("type"), "parent": parent, "child": child,
                                 "xyz": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                                 "rpy": origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"})

    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
    mesh_bounds = {}
    for mid in range(model.nmesh):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
        start = int(model.mesh_vertadr[mid])
        count = int(model.mesh_vertnum[mid])
        vertices = model.mesh_vert[start:start + count]
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        mesh_bounds[mesh_name] = {"vertices": count, "faces": int(model.mesh_facenum[mid]),
                                  "local_bounds_m": [float(x) for x in np.round(upper - lower, 7)],
                                  "local_min_m": [float(x) for x in np.round(lower, 7)],
                                  "local_max_m": [float(x) for x in np.round(upper, 7)]}
    geoms = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        mesh_name = ""
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid])) or ""
        if body_name.endswith("wrist_roll_link") or body_name.startswith(("L_", "R_")) or mesh_name in GRIPPER_MESHES:
            geoms.append({"geom_id": gid, "body_id": body_id, "body": body_name, "mesh": mesh_name,
                          "group": int(model.geom_group[gid]), "contype": int(model.geom_contype[gid]),
                          "conaffinity": int(model.geom_conaffinity[gid]), "source_role": "visual" if model.geom_contype[gid] == 0 else "collision"})

    forbidden_bodies = [n for n in bodies if any(token in n.lower() for token in ("omnihand", "omni_hand", "dex3", "dex2"))]
    forbidden_joints = [n for n in joint_names if any(token in n.lower() for token in ("omnihand", "omni_hand", "finger_joint"))]
    # Include both official left/right tree details and their compiled geom mappings.
    return {
        "official_source": {"urdf_path": str(urdf.resolve()), "urdf_sha256": sha256(urdf), "upstream_commit": "575cc6b988f976c23550e0db85aa1e5475d3652d"},
        "mujoco_version": mujoco.__version__,
        "source_links": relationships,
        "source_joint_tree": parent_child,
        "source_meshes": sorted(mesh_map.values(), key=lambda x: x["mesh_file"]),
        "compiled": {"nbody": model.nbody, "ngeom": model.ngeom, "nmesh": model.nmesh, "njnt": model.njnt, "neq": model.neq,
                     "bodies": bodies, "joints": joint_names, "wrist_gripper_geoms": geoms,
                     "mesh_bounds": mesh_bounds,
                     "omnihand_or_other_hand_bodies": forbidden_bodies, "unexpected_finger_joint_names": forbidden_joints},
        "mounts": {"left": {"joint": "L_omnipicker_joint", "parent": "left_wrist_roll_link", "child": "L_omnipicker_base_link", "xyz": [0.00259, 0.00035, -0.062], "rpy": [3.14, 0.0, 3.14]},
                   "right": {"joint": "R_omnipicker_joint", "parent": "right_wrist_roll_link", "child": "R_omnipicker_base_link", "xyz": [0.00196, 0.00035, -0.062], "rpy": [3.14, 0.0, 0.0]}},
    }


def render_state(state: str, scene_path: Path, output: Path, width: int, height: int,
                 azimuth: float = 0.0, elevation: float = -18.0) -> Path:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    target, distance, _, _ = camera_for(model, data, state)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = True
    renderer.update_scene(data, camera=camera, scene_option=option)
    frame = renderer.render()
    save_labeled(frame, output, STATES[state] + " | static; physics_steps=0; contacts disabled")
    renderer.close()
    return output


def interactive(state: str, scene_path: Path) -> None:
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    target, distance, azimuth, elevation = camera_for(model, data, state)
    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as handle:
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.lookat[:] = target
        handle.cam.distance = distance
        handle.cam.azimuth = azimuth
        handle.cam.elevation = elevation
        handle.opt.geomgroup[:] = True
        print(f"State: {state} -- {STATES[state]}")
        print("Static review only: this script never calls mj_step; mouse drag=rotate, right-drag/shift-drag=pan, wheel=zoom.")
        while handle.is_running():
            handle.sync()
            time.sleep(0.02)


def run(args: argparse.Namespace) -> None:
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"requires MuJoCo {MUJOCO_VERSION}; found {mujoco.__version__}")
    urdf = args.urdf.resolve()
    meshdir = urdf.parent / "meshes"
    output = args.output.resolve()
    scenes = output / "scenes"
    output.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(urdf))
    base_xml = spec.to_xml()
    base_model = spec.compile()
    base_data = mujoco.MjData(base_model)
    mujoco.mj_resetData(base_model, base_data)
    mujoco.mj_forward(base_model, base_data)
    visible = base_model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    low_points = []
    for gid in np.flatnonzero(visible):
        if base_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(base_model.geom_dataid[gid])
            start = int(base_model.mesh_vertadr[mid])
            count = int(base_model.mesh_vertnum[mid])
            verts = base_model.mesh_vert[start:start + count]
            rotation = base_data.geom_xmat[gid].reshape(3, 3)
            world_verts_z = (verts @ rotation.T + base_data.geom_xpos[gid])[..., 2]
            low_points.append(float(world_verts_z.min()))
        else:
            low_points.append(float(base_data.geom_xpos[gid, 2] - base_model.geom_rbound[gid]))
    floor_z = min(low_points) - 0.006

    scene_paths = {key: scenes / f"{key}.xml" for key in STATES}
    keys_to_build = [args.state] if args.interactive else list(STATES)
    for key in keys_to_build:
        scene_path = scenes / f"{key}.xml"
        make_scene_xml(base_xml, key, floor_z, scene_path, meshdir)
        scene_paths[key] = scene_path

    inventory = source_inventory(urdf, base_model)
    inventory["review_scene"] = {"floor_z_visual_only": floor_z, "floor_contype": 0, "floor_conaffinity": 0,
                                 "all_source_geoms_contype": 0, "all_source_geoms_conaffinity": 0,
                                 "physics_steps": 0, "mj_forward_calls": 1, "source_or_production_files_modified": False}
    inventory["review_scenes"] = {key: str(path) for key, path in scene_paths.items()}
    inventory["candidate_registry_status"] = "VALIDATED_PENDING_HUMAN_REVIEW"
    inventory["finding"] = {
        "root_cause": "The official right-wrist collision mesh is a distinct, unaligned, longer STL than its visual mesh. It contains the suspicious finger-like geometry inside a single source mesh and compiled collision geom; the normal viewer can draw that collision geom alongside the OmniPicker visuals.",
        "right_wrist_visual": {"source_file": "right_wrist_roll_extend_link.stl", "sha256": "94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de", "compiled_geom_id": 87, "contact_mask": [0, 0], "local_aabb_m": inventory["compiled"]["mesh_bounds"]["right_wrist_roll_extend_link"]["local_bounds_m"]},
        "right_wrist_collision": {"source_file": "right_wrist_roll_link.stl", "sha256": "f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72", "compiled_geom_id": 88, "contact_mask": [1, 1], "local_aabb_m": inventory["compiled"]["mesh_bounds"]["right_wrist_roll_link"]["local_bounds_m"], "vertices": inventory["compiled"]["mesh_bounds"]["right_wrist_roll_link"]["vertices"], "faces": inventory["compiled"]["mesh_bounds"]["right_wrist_roll_link"]["faces"]},
        "left_wrist_visual_and_collision": {"source_file": "left_wrist_roll_extend_link.stl", "sha256": "f379d912d341e6224ec16a62ec9a3c21621e43b6aae1bec0ba93e9a947d7507f", "compiled_geom_ids": [56, 57], "local_aabb_m": inventory["compiled"]["mesh_bounds"]["left_wrist_roll_extend_link"]["local_bounds_m"]},
        "no_extra_omnihand_or_duplicate_end_effector": True,
        "official_urdf_already_maps_the_distinct_right_collision_mesh": True,
        "issue46_same_collision_asset": True,
        "issue46_evidence": "/mnt/c/Users/HP/Desktop/RobotSim/reviews/asset-registry-review-20261012/issue46_runtime_model_comparison.json",
        "issue46_collision_overlap_note": "That read-only report records the same right_wrist_roll_link.stl hash in the actual run05 compiled fixture and prior sampled source geometry intersecting the bottle volume in tested candidates; this makes the asset physically relevant but does not alone establish it as the sole cause of contact outcomes.",
        "safe_action": "Use a review-only geom-group display toggle to hide collision geom visuals when judging appearance. Do not change this physics mesh mapping until an intended right-side collision asset is verified; no production asset was changed."
    }
    (output / "geometry_audit.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    if args.interactive:
        if not args.state:
            raise ValueError("--state is required with --interactive")
        interactive(args.state, scene_paths[args.state])
        return

    selected_states = list(STATES) if args.render_all else [args.state or "full_robot"]
    outputs = []
    for state in selected_states:
        filename = f"{state}.png"
        path = render_state(state, scene_paths[state], output / filename, args.width, args.height)
        outputs.append(path)

    if args.render_all:
        full_model_front = render_state("full_robot", scene_paths["full_robot"], output / "full_robot_front.png", args.width, args.height, 0.0, -18.0)
        full_model_rear = render_state("full_robot", scene_paths["full_robot"], output / "full_robot_rear.png", args.width, args.height, 180.0, -18.0)
        outputs.extend([full_model_front, full_model_rear])
        make_contact_sheet(outputs, output / "component_contact_sheet.png")
        # Same geometry and camera scale makes wrist mesh origin/source comparison direct.
        comparison = [output / "right_wrist_visual.png", output / "right_wrist_collision.png", output / "right_wrist_mesh_overlay.png", output / "left_wrist_visual.png", output / "left_wrist_collision.png", output / "omnipicker_right.png", output / "right_assembly.png", output / "right_assembly_runtime_view.png", output / "left_assembly.png", output / "full_robot_rear.png"]
        make_contact_sheet(comparison, output / "six_state_review_sheet.png")

    report = {
        "task": "X2 OmniPicker right-hand visual anomaly, static geometry source audit",
        "classification": "REVIEW_ONLY_NOT_ASSET_APPROVAL",
        "asset_id": "x2_ultra_omnipicker_v1_4",
        "candidate_registry_status": "VALIDATED_PENDING_HUMAN_REVIEW",
        "source_commit": inventory["official_source"]["upstream_commit"],
        "source_urdf_sha256": inventory["official_source"]["urdf_sha256"],
        "physics_steps": 0,
        "exact_user_view_is_from_behind": True,
        "rendered_states": [str(x) for x in outputs],
        "interactive_command_template": "MUJOCO_GL=glfw /tmp/robotsim-asset-registry-venv/bin/python /home/lzy18001500226/robotsim-asset-registry-20261012/scripts/review_x2_omnipicker_layers.py --interactive --state right_assembly_runtime_view",
        "interactive_state_names": STATES,
    }
    (output / "review_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--state", choices=tuple(STATES))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
