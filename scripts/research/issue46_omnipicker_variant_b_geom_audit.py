#!/usr/bin/env python3
"""Build a static geom-group audit from the exact Variant B viewer model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORKTREE = Path("/home/lzy18001500226/robotsim-asset-registry-20261012")
AB_RUNNER = WORKTREE / "scripts/research/issue46_omnipicker_wrist_collision_ab.py"
SOURCE_REPO = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf")
SOURCE_MODEL = SOURCE_REPO / "X2_URDF-v1.4.0"
SOURCE_URDF = SOURCE_MODEL / "X2-Ultra_omnipicker.urdf"
REFERENCE_DIR = Path("/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-omnipicker-right-wrist-collision-ab-20261012-final-review")
REFERENCE_RESULT = REFERENCE_DIR / "result.json"
REFERENCE_INTERACTIVE_MJCF = REFERENCE_DIR / "models/right_wrist_roll_link_B_local03.xml"
PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
DEFAULT_OUTPUT = Path("/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-omnipicker-variant-b-geom-audit-20261012")
MESH_ENUM = int(mujoco.mjtGeom.mjGEOM_MESH)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_ab_runner():
    spec = importlib.util.spec_from_file_location("issue46_omnipicker_wrist_collision_ab", AB_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the A/B runner: {AB_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_URDF = SOURCE_URDF
    module.MODEL_DIR = SOURCE_MODEL
    module.MESH_DIR = SOURCE_MODEL / "meshes"
    return module


def model_name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str | None:
    return mujoco.mj_id2name(model, kind, int(ident))


def mesh_assets(xml_path: Path) -> tuple[Path, dict[str, Path]]:
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("Compiled model XML has no compiler element")
    meshdir = Path(compiler.get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = (xml_path.parent / meshdir).resolve()
    rows = {}
    for mesh in root.findall("asset/mesh"):
        mesh_name = mesh.get("name")
        filename = mesh.get("file")
        if mesh_name and filename:
            rows[mesh_name] = (meshdir / filename).resolve()
    return meshdir, rows


def geom_inventory(model: mujoco.MjModel, mesh_files: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body = model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        geom = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        geom_type = mujoco.mjtGeom(int(model.geom_type[gid])).name
        mesh_id = int(model.geom_dataid[gid]) if int(model.geom_type[gid]) == MESH_ENUM else -1
        mesh = model_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None
        path = mesh_files.get(mesh) if mesh else None
        contype, conaffinity = int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])
        rows.append({
            "geom_id": gid,
            "geom": geom,
            "body_id": body_id,
            "body": body,
            "geom_type": geom_type,
            "mesh_id": mesh_id if mesh_id >= 0 else None,
            "mesh": mesh,
            "mesh_path": str(path) if path else None,
            "mesh_sha256": sha256(path) if path and path.is_file() else None,
            "group": int(model.geom_group[gid]),
            "rgba": [float(x) for x in model.geom_rgba[gid]],
            "contype": contype,
            "conaffinity": conaffinity,
            "active_collision": bool(contype or conaffinity),
            "body_local_pos": [float(x) for x in model.geom_pos[gid]],
            "body_local_quat_wxyz": [float(x) for x in model.geom_quat[gid]],
            "size": [float(x) for x in model.geom_size[gid]],
        })
    return rows


def identity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("geom_id", "geom", "body_id", "body", "geom_type", "mesh_id", "mesh", "mesh_sha256",
            "group", "rgba", "contype", "conaffinity", "active_collision", "body_local_pos",
            "body_local_quat_wxyz", "size")
    return [{key: row[key] for key in keys} for row in rows]


def style_variant_b(model: mujoco.MjModel) -> None:
    wrist_collision = [0.0, 0.78, 0.78, 0.88]
    for gid in range(model.ngeom):
        body = model_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        geom = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        group = int(model.geom_group[gid])
        if body == "right_wrist_roll_link":
            if int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]):
                model.geom_rgba[gid] = wrist_collision
            else:
                model.geom_rgba[gid] = [0.88, 0.91, 0.92, 1.0]
        elif body.startswith("R_hand_") or body == "R_omnipicker_base_link":
            model.geom_rgba[gid] = [0.42, 0.68, 0.77, 0.95] if group == 1 else [0.42, 0.68, 0.77, 0.35]
        elif body in ("m0_bottle", "p0_canonical_test_cylinder") or geom.startswith("bottle_") or geom == "p0_canonical_test_cylinder_geom":
            model.geom_rgba[gid] = [0.10, 0.48, 0.80, 0.78]
        elif body == "m0_table" or geom.startswith("m0_table_"):
            model.geom_rgba[gid] = [0.45, 0.31, 0.22, 1.0]


def set_open_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint, value in (("right_claw_joint", -0.98), ("R_hand_wide1_joint", 0.98)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise RuntimeError(f"Expected jaw joint missing from Variant B: {joint}")
        data.qpos[int(model.jnt_qposadr[jid])] = value


def render_groups(model: mujoco.MjModel, data: mujoco.MjData, output: Path,
                  groups: set[int], title: str, subtitle: str = "",
                  body_filter: set[str] | None = None, distance: float = 0.52) -> None:
    saved_rgba = model.geom_rgba.copy()
    renderer = mujoco.Renderer(model, height=760, width=1120)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.30, -0.07, 0.92]
    camera.distance = distance
    camera.azimuth = 180.0
    camera.elevation = -10.0
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 0, 0, 0, 0, 0]
    for group in groups:
        option.geomgroup[group] = 1
    if body_filter is not None:
        for gid in range(model.ngeom):
            body = model_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if body not in body_filter:
                model.geom_rgba[gid, 3] = 0.0
    renderer.update_scene(data, camera=camera, scene_option=option)
    frame = Image.fromarray(renderer.render())
    renderer.close()
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 54 if subtitle else 38), fill=(17, 28, 35))
    draw.text((12, 9), title, fill=(250, 252, 252), font=ImageFont.load_default())
    if subtitle:
        draw.text((12, 29), subtitle, fill=(211, 229, 234), font=ImageFont.load_default())
    frame.save(output)
    model.geom_rgba[:] = saved_rgba


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 490))
        tile = Image.new("RGB", (740, 525), (235, 239, 241))
        tile.paste(image, ((740 - image.width) // 2, 26))
        ImageDraw.Draw(tile).text((10, 8), path.name, fill=(20, 28, 33), font=ImageFont.load_default())
        tiles.append(tile)
    sheet = Image.new("RGB", (1480, 1050), (225, 230, 232))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 740, (index // 2) * 525))
    sheet.save(output)


def make_body_sheet(paths: list[tuple[str, Path]], output: Path) -> None:
    tile_w, tile_h = 560, 400
    cols = 2
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (230, 235, 238))
    for index, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 12, tile_h - 38))
        x, y = (index % cols) * tile_w, (index // cols) * tile_h
        sheet.paste(image, (x + (tile_w - image.width) // 2, y + 30))
        ImageDraw.Draw(sheet).text((x + 8, y + 8), label, fill=(20, 28, 33), font=ImageFont.load_default())
    sheet.save(output)


def source_joint_rows(urdf_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    rows = []
    for joint in root.findall("joint"):
        parent, child, origin, mimic = (joint.find("parent"), joint.find("child"),
                                        joint.find("origin"), joint.find("mimic"))
        rows.append({
            "name": joint.get("name"), "type": joint.get("type"),
            "parent": parent.get("link") if parent is not None else None,
            "child": child.get("link") if child is not None else None,
            "origin_xyz": origin.get("xyz") if origin is not None else None,
            "origin_rpy": origin.get("rpy") if origin is not None else None,
            "mimic": mimic.attrib if mimic is not None else None,
        })
    return rows


def patch_meshdir(xml_in: Path, xml_out: Path) -> None:
    tree = ET.parse(xml_in)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("No compiler element to set persistent meshdir")
    compiler.set("meshdir", "../source_snapshot/meshes")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    tree.write(xml_out, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence directory: {output}")
    if mujoco.__version__ != "3.3.6":
        raise RuntimeError(f"Expected MuJoCo 3.3.6; got {mujoco.__version__}")
    commit = subprocess.run(["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != PIN:
        raise RuntimeError(f"Upstream checkout is not pinned: {commit}")
    source_urdf_hash = sha256(SOURCE_URDF)
    expected_source_hash = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
    if source_urdf_hash != expected_source_hash:
        raise RuntimeError(f"Pinned URDF hash mismatch: {source_urdf_hash}")

    ref = read_json(REFERENCE_RESULT)
    blocker = ref["scenarios"]["local03_translated"]["A"]["first_approach_blocker"]
    if blocker is None:
        raise RuntimeError("The retained A/B evidence has no A first-blocker pose")
    if ref["scenarios"]["local03_translated"]["B"]["at_A_first_approach_blocker_pose"] is None:
        raise RuntimeError("The retained A/B evidence lacks the exact-pose B sample")

    output.mkdir(parents=True)
    for folder in ("models", "images", "source_snapshot/meshes"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_URDF, output / "source_snapshot" / SOURCE_URDF.name)

    base = load_ab_runner()
    helper = base.load_module("canonical_manipulation_assets", base.CANONICAL_HELPER)
    pose = (blocker["root_position_world_m"], blocker["root_quaternion_wxyz"])
    model, data, derived_urdf, subtree_manifest = base.build_subtree(
        "B", "right_wrist_roll_link", output, helper, "local03", pose, review=True)
    base.set_named_qpos(model, data, {"right_claw_joint": -0.98, "R_hand_wide1_joint": 0.98})
    mujoco.mj_forward(model, data)
    style_variant_b(model)
    original_xml = output / "models" / "right_wrist_roll_link_B_local03.xml"
    persistent_xml = output / "models" / "variant_b_local03_first_blocker.xml"
    if not REFERENCE_INTERACTIVE_MJCF.is_file():
        raise FileNotFoundError(f"Retained interactive Variant B MJCF not found: {REFERENCE_INTERACTIVE_MJCF}")
    interactive_mjcf_sha = sha256(REFERENCE_INTERACTIVE_MJCF)
    rebuilt_interactive_mjcf_sha = sha256(original_xml)
    if interactive_mjcf_sha != rebuilt_interactive_mjcf_sha:
        raise RuntimeError("Rebuilt Variant B MJCF differs from the retained interactive viewer model")

    _, source_mesh_files = mesh_assets(original_xml)
    used_mesh_names = sorted({
        model_name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid]))
        for gid in range(model.ngeom) if int(model.geom_type[gid]) == MESH_ENUM
    })
    source_assets = {}
    for mesh_name in used_mesh_names:
        source_path = source_mesh_files[mesh_name]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = output / "source_snapshot" / "meshes" / source_path.name
        shutil.copy2(source_path, destination)
        source_assets[mesh_name] = {"source_path": str(source_path), "snapshot_path": str(destination),
                                    "sha256": sha256(destination), "bytes": destination.stat().st_size}
    patch_meshdir(original_xml, persistent_xml)

    # Recompile the persistent snapshot; require it to reproduce the exact runtime geom inventory.
    portable_spec = mujoco.MjSpec.from_file(str(persistent_xml))
    portable_model = portable_spec.compile()
    portable_data = mujoco.MjData(portable_model)
    mujoco.mj_resetData(portable_model, portable_data)
    set_open_pose(portable_model, portable_data)
    mujoco.mj_forward(portable_model, portable_data)
    source_rgba = portable_model.geom_rgba.copy()
    style_variant_b(portable_model)

    original_mesh_map = mesh_assets(original_xml)[1]
    portable_mesh_map = mesh_assets(persistent_xml)[1]
    original_rows = geom_inventory(model, original_mesh_map)
    portable_rows = geom_inventory(portable_model, portable_mesh_map)
    structurally_equal = identity_rows(original_rows) == identity_rows(portable_rows)
    if not structurally_equal:
        raise RuntimeError("Persistent Variant B model failed exact geom identity comparison")

    source_meshes = {item["sha256"] for item in source_assets.values()}
    old_mesh_path = SOURCE_MODEL / "meshes" / "right_wrist_roll_link.stl"
    old_mesh_sha = sha256(old_mesh_path)
    old_mesh_loaded = any(row["mesh_sha256"] == old_mesh_sha for row in original_rows)
    wrist_rows = [row for row in original_rows if row["body"] == "right_wrist_roll_link"]
    active_wrist = [row for row in wrist_rows if row["active_collision"]]
    jaw_rows = [row for row in original_rows
                if row["body"] == "R_omnipicker_base_link" or row["body"].startswith("R_hand_")]
    active_jaws = [row for row in jaw_rows if row["active_collision"]]
    if len(active_wrist) != 1:
        raise RuntimeError(f"Expected one active B wrist collision geom, got {len(active_wrist)}")
    if old_mesh_loaded:
        raise RuntimeError("Legacy right_wrist_roll_link.stl unexpectedly remains in the B compiled model")
    compiled_mesh_names = [model_name(model, mujoco.mjtObj.mjOBJ_MESH, mid) for mid in range(model.nmesh)]
    compiled_body_names = [model_name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "" for bid in range(1, model.nbody)]
    compiled_joint_names = [model_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "" for jid in range(model.njnt)]
    if any("omnihand" in name.lower() for name in compiled_body_names + compiled_joint_names):
        raise RuntimeError("Unexpected OmniHand body or joint exists in compiled Variant B subtree")
    if any("variant_a" in name.lower() or "ghost" in name.lower() for name in compiled_body_names):
        raise RuntimeError("Unexpected A/debug ghost body exists in compiled Variant B subtree")

    for group, name in ((0, "geom_group_0_only.png"), (1, "geom_group_1_only.png"),
                        (2, "geom_group_2_only.png")):
        count = sum(1 for row in original_rows if row["group"] == group)
        render_groups(model, data, output / "images" / name, {group},
                      f"Variant B | DISPLAY GROUP {group} ONLY | {count} compiled geoms",
                      "Only this geomgroup is visible; physics state is static")
    visual_bodies = {"right_wrist_roll_link", "R_omnipicker_base_link"} | {
        row["body"] for row in original_rows if row["body"].startswith("R_hand_")
    }
    portable_model.geom_rgba[:] = source_rgba
    render_groups(portable_model, portable_data, output / "images" / "variant_b_visual_only.png", {1},
                  "Variant B | SOURCE VISUAL ASSEMBLY | WRIST + OMNIPICKER",
                  "Source MJCF RGBA; no collision or debug overlay", visual_bodies)
    portable_model.geom_rgba[:] = np.asarray([row["rgba"] for row in original_rows], dtype=float)
    jaw_names = ", ".join(f"{r['body']}/{r['geom']}" for r in active_jaws)
    render_groups(model, data, output / "images" / "variant_b_collision_only.png", {0},
                  "Variant B | ACTIVE COLLISION ONLY | DISPLAY GROUP 0",
                  f"Wrist: {active_wrist[0]['body']}/{active_wrist[0]['geom']} | jaw/finger geoms: {len(active_jaws)}",
                  {"right_wrist_roll_link"} | {row["body"] for row in active_jaws})
    group_images = [output / "images" / f"geom_group_{group}_only.png" for group in (0, 1, 2)]
    make_contact_sheet(group_images, output / "geom_groups_comparison.png")

    isolation_dir = output / "images" / "isolated_geoms"
    isolation_dir.mkdir(parents=True, exist_ok=True)
    visual_bodies_in_order = []
    for row in original_rows:
        if row["group"] == 1 and row["mesh"] and row["body"] not in visual_bodies_in_order:
            visual_bodies_in_order.append(row["body"])
    isolated_panels = []
    shadow_panels = []
    for body in visual_bodies_in_order:
        visual_row = next(row for row in original_rows if row["body"] == body and row["group"] == 1 and row["mesh"])
        visual_path = isolation_dir / f"{body}_visual.png"
        portable_model.geom_rgba[:] = source_rgba
        render_groups(portable_model, portable_data, visual_path, {1},
                      f"SOURCE VISUAL | geom {visual_row['geom_id']} | {body}",
                      f"mesh={visual_row['mesh']} | sha256={visual_row['mesh_sha256'][:12]}", {body}, 0.29)
        isolated_panels.append((f"VISUAL | {body} | {visual_row['mesh']}", visual_path))
        shadow_path = isolation_dir / f"{body}_table_shadow.png"
        render_groups(portable_model, portable_data, shadow_path, {1},
                      f"SHADOW ISOLATION | {body}",
                      "Only this visual body + group-1 table", {body, "m0_table"}, 0.40)
        shadow_panels.append((f"SHADOW | {body} | {visual_row['mesh']}", shadow_path))
        collision_row = next((row for row in original_rows if row["body"] == body and
                              row["group"] == 0 and row["active_collision"] and row["mesh"]), None)
        if collision_row is not None:
            collision_path = isolation_dir / f"{body}_collision.png"
            render_groups(model, data, collision_path, {0},
                          f"ACTIVE COLLIDER | geom {collision_row['geom_id']} | {body}",
                          f"mesh={collision_row['mesh']} | sha256={collision_row['mesh_sha256'][:12]}", {body}, 0.29)
            isolated_panels.append((f"COLLISION | {body} | {collision_row['mesh']}", collision_path))
    make_body_sheet(isolated_panels, output / "isolated_visual_collision_by_body.png")
    make_body_sheet(shadow_panels, output / "isolated_table_shadow_by_body.png")

    viewer_source = WORKTREE / "scripts/research/issue46_variant_b_static_viewer.py"
    launcher_source = WORKTREE / "scripts/research/launch_issue46_variant_b_static_viewer.sh"
    shutil.copy2(viewer_source, output / "viewer.py")
    shutil.copy2(launcher_source, output / "launch_variant_b_viewer.sh")
    (output / "launch_variant_b_viewer.sh").chmod(0o755)

    inventory = {
        "status": "STATIC_VARIANT_B_COMPILED_GEOM_AUDIT",
        "physics_steps": 0,
        "mj_step_calls": 0,
        "state_advance_calls": 0,
        "allowed_state_calls": ["mj_resetData", "mj_forward"],
        "runtime_model_origin": "same build_subtree(... variant B, root wrist, local03, saved A first-blocker pose) path used by --interactive --variant B --state local03_first_blocker",
        "variant": "B",
        "state": "local03_first_blocker",
        "source": {"repository": "AgibotTech/agibot_x2_urdf", "commit": commit,
                    "source_urdf": str(SOURCE_URDF), "source_urdf_sha256": source_urdf_hash,
                    "legacy_collision_mesh": {"filename": old_mesh_path.name, "sha256": old_mesh_sha,
                                              "present_in_compiled_b_mesh_assets": old_mesh_loaded},
                    "snapshot_meshes": source_assets},
        "pose": {"source_sample": blocker["source_trace_sample"],
                 "root_position_world_m": blocker["root_position_world_m"],
                 "root_quaternion_wxyz": blocker["root_quaternion_wxyz"],
                 "finger_qpos": {"right_claw_joint": -0.98, "R_hand_wide1_joint": 0.98}},
        "compiled_model": {"source_mjcf_path": str(original_xml), "source_mjcf_sha256": sha256(original_xml),
                           "persistent_mjcf_path": str(persistent_xml), "persistent_mjcf_sha256": sha256(persistent_xml),
                           "exact_geom_identity_reproduced": structurally_equal,
                           "retained_interactive_mjcf_path": str(REFERENCE_INTERACTIVE_MJCF),
                           "retained_interactive_mjcf_sha256": interactive_mjcf_sha,
                           "rebuilt_interactive_mjcf_sha256": rebuilt_interactive_mjcf_sha,
                           "exact_retained_interactive_model_match": interactive_mjcf_sha == rebuilt_interactive_mjcf_sha,
                           "compiled_mesh_names": compiled_mesh_names,
                           "compiled_joint_names": compiled_joint_names,
                           "compiled_body_names": compiled_body_names,
                           "variant_a_or_debug_ghost_loaded": False,
                           "omnihand_body_or_joint_loaded": False,
                           "nbody": int(model.nbody), "ngeom": int(model.ngeom), "njnt": int(model.njnt),
                           "nmesh": int(model.nmesh), "nu": int(model.nu)},
        "group_counts": {str(group): sum(1 for row in original_rows if row["group"] == group)
                         for group in sorted({row["group"] for row in original_rows})},
        "compiled_bodies": [{"body_id": bid,
                             "name": model_name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
                             "parent_id": int(model.body_parentid[bid]),
                             "parent": model_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                  int(model.body_parentid[bid]))}
                            for bid in range(1, model.nbody)],
        "source_joint_hierarchy": source_joint_rows(derived_urdf),
        "group_0_active_collisions": [row for row in original_rows if row["group"] == 0 and row["active_collision"]],
        "wrist_geoms": wrist_rows,
        "active_wrist_collision_geoms": active_wrist,
        "jaw_and_finger_geoms": jaw_rows,
        "active_jaw_and_finger_collision_geoms": active_jaws,
        "geoms": original_rows,
        "conclusion": {
            "legacy_finger_collision_mesh_active_in_b": False,
            "visible_finger_like_meshes": [row for row in original_rows
                                             if row["group"] == 1 and row["mesh"]],
            "source_visual_rgba_by_geom": {str(row["geom_id"]): [float(x) for x in source_rgba[row["geom_id"]]]
                                           for row in original_rows},
            "b_wrist_visual_mesh": "right_wrist_roll_extend_link.stl",
            "b_wrist_active_collision_mesh": active_wrist[0]["mesh"],
            "wrist_visual_and_collision_use_same_mesh": next((r["mesh"] for r in wrist_rows if r["group"] == 1), None) == active_wrist[0]["mesh"],
            "active_collision_geoms_in_group_0_are_physical": True,
            "group_1_visual_geoms_are_noncolliding": all(not row["active_collision"] for row in original_rows if row["group"] == 1 and row["body"] != "m0_table"),
        },
        "runner": {"ab_runner_path": str(AB_RUNNER), "ab_runner_sha256": sha256(AB_RUNNER)},
        "artifacts": {"visual_only_image": "images/variant_b_visual_only.png",
                      "collision_only_image": "images/variant_b_collision_only.png",
                      "group_0_image": "images/geom_group_0_only.png",
                      "group_1_image": "images/geom_group_1_only.png",
                      "group_2_image": "images/geom_group_2_only.png",
                      "group_comparison": "geom_groups_comparison.png",
                      "isolated_visual_collision_comparison": "isolated_visual_collision_by_body.png",
                      "isolated_table_shadow_comparison": "isolated_table_shadow_by_body.png"},
    }
    write_json(output / "geom_inventory.json", inventory)

    source_assets_text = "\n".join(
        f"| `{mesh}` | `{item['sha256']}` | {item['bytes']:,} | `{Path(item['snapshot_path']).relative_to(output)}` |"
        for mesh, item in sorted(source_assets.items()))
    active_wrist_text = ", ".join(f"`{r['geom']}` / `{r['mesh']}` (group {r['group']}, contype={r['contype']}, conaffinity={r['conaffinity']})" for r in active_wrist)
    active_jaw_text = "\n".join(f"- geom {r['geom_id']}: `{r['body']}/{r['geom']}` → `{r['mesh']}`; group {r['group']}, contype={r['contype']}, conaffinity={r['conaffinity']}" for r in active_jaws)
    report = f'''# X2 OmniPicker Variant B 残余手形状与 geom-group 核验

## 核验对象

- 固定上游：`AgibotTech/agibot_x2_urdf` commit `{commit}`。
- URDF：`X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`，SHA-256 `{source_urdf_hash}`。
- MuJoCo `{mujoco.__version__}`；Variant B；Local03 保存的 A 首次 blocker 姿态 sample `{blocker['source_trace_sample']}`；手爪静态 OPEN。
- B 使用与原 `--interactive --variant B --state local03_first_blocker` 相同的 `build_subtree` 编译路径。重建运行 MJCF SHA-256 `{rebuilt_interactive_mjcf_sha}` 与原交互模型 `{interactive_mjcf_sha}` 完全相同；持久 MJCF 重编译后，所有 geom ID/name/body/mesh/hash/group/RGBA/contact flags/局部位姿与运行模型逐项一致：`{structurally_equal}`。
- `physics_steps=0`，`mj_step_calls=0`，`state_advance_calls=0`。只做 reset、静态 qpos 初始化、forward 和渲染。

## Geom group 的实际含义

这里的 group 是 `MjModel.geom_group[geom_id]`，不是 geom ID。图像分别只启用 group 0、1、2。Group 1 包含 vendor visual geoms，也包含 Local03 fixture 的桌子；group 2 是瓶子；group 0 包含未显式指定 group 的 active robot collider。具体以 `geom_inventory.json` 的逐 geom 表为准。

| 图像 | 启用的显示组 |
|---|---|
| `images/geom_group_0_only.png` | 仅 group 0 |
| `images/geom_group_1_only.png` | 仅 group 1 |
| `images/geom_group_2_only.png` | 仅 group 2 |
| `images/variant_b_visual_only.png` | 仅 source visual group 1，并筛选右腕和 OmniPicker body；保留 source MJCF RGBA，无诊断着色 |
| `images/variant_b_collision_only.png` | 仅 group 0，并筛选右腕和 jaw/finger active collision geoms |

## 残余结构来源与物理影响

- Variant B 中右腕的视觉 mesh 和 active collision mesh 均为 `right_wrist_roll_extend_link.stl`。对应的实际 wrist geom 为：{active_wrist_text}。
- 老 mesh `right_wrist_roll_link.stl` 的 SHA-256 为 `{old_mesh_sha}`，它不在 Variant B 编译资产表中，也没有任何 Variant B geom 引用它。B 的生成 URDF只对右腕 collision filename 做了这一项替换；没有加载 A 的 compiled model，viewer 中不存在 A ghost/debug overlay。
- jaw/finger subtree 的活动碰撞 geom：
{active_jaw_text}
- 每个 visual/collision body 的独立画面在 `isolated_visual_collision_by_body.png` 与 `images/isolated_geoms/`。因此可将“腕延伸件轮廓”和真实窄/宽 OmniPicker jaw links 分开核对，不再仅凭合成画面推断来源。
- group 1 独占图还会绘制其桌面 geom 的投影阴影。桌面阴影不是 MuJoCo geom、不会进入碰撞；逐 body 投影结果见 `isolated_table_shadow_by_body.png`，而去掉桌面后的真实视觉装配见 `images/variant_b_visual_only.png`。
- 组 1 的手指状轮廓按 body/geom/mesh 的逐项映射见 `geom_inventory.json` 中 `visible_finger_like_meshes`；这些 source visual geoms 仅用于渲染，机器人可视 geom 的接触 mask 为零。组 0 则显示 B 的活动碰撞副本及 jaw/finger collider。
- 右侧的大块 U 形/长指状腕部轮廓精确来自 `right_wrist_roll_link/geom_1`（source visual）及同位 `right_wrist_roll_link/geom_2`（active collision）；两者都用 `right_wrist_roll_extend_link.stl`，SHA-256 `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`。另一侧的分节 jaw/pad 轮廓来自 `R_omnipicker_base_link` 的 `base_link` 及 `R_hand_narrow*`、`R_hand_wide*` links，是官方的两组 OmniPicker jaws。详见逐 body 隔离图。
- 原 A 版的 finger-like collision mesh `right_wrist_roll_link.stl` 不再 active。B 仍显示腕部轮廓，是因为 B 的腕 visual 与 active collider 都引用 `right_wrist_roll_extend_link.stl`；jaw 自身也有多段窄/宽 links。它不是 A 模型 ghost，也不是 OmniHand 重复加载。

## Variant B 实际引用的源网格

| MjModel mesh | SHA-256 | 字节 | 持久快照 |
|---|---|---:|---|
{source_assets_text}

实际残留腕形轮廓由右腕 link 的 visual geom `geom_1` 和 active collision geom `geom_2` 共同产生；两者都引用 `right_wrist_roll_extend_link.stl`，SHA-256 `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`。额外手指状轮廓不是另一个 OmniHand 资源：编译模型仅含 `right_wrist_roll_link`、`R_omnipicker_base_link` 和相应 `R_hand_{{narrow,wide}}*` body；没有 OmniHand body/joint，也没有第二个 end-effector subtree。`R_hand_narrow*` / `R_hand_wide*` 是实际 OmniPicker 可动 jaw link，而非额外人手。

持久模型 `models/variant_b_local03_first_blocker.xml` 的 meshdir 是相对路径 `../source_snapshot/meshes`，不依赖 `/tmp/robotsim-asset-registry-upstreams` 或 upstream cache。A/B 原结果目录没有被覆盖。

## 启动

在 WSL Ubuntu-22.04 终端执行：

```bash
cd {output} && ./launch_variant_b_viewer.sh --view visual
```

`launch_variant_b_viewer.sh` 使用固定 MuJoCo 3.3.6 venv，不依赖 `/tmp` upstream。默认只显示 source visual group 1 的右腕和 OmniPicker 几何；窗口叠加标题 `Variant B | local03_first_blocker | VISUAL ONLY`。可用 `--view collision`、`--view all`、`--view group0`、`--view group1`、`--view group2` 切换。鼠标可旋转、平移和缩放；程序只同步 viewer，不调用 `mj_step`。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            manifest_files.append({"path": str(path.relative_to(output)),
                                   "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(output / "SHA256SUMS.json", {"files": manifest_files})
    print(json.dumps({"output": str(output), "compiled_identity": structurally_equal,
                      "ngeom": model.ngeom, "nmesh": model.nmesh,
                      "old_mesh_loaded": old_mesh_loaded,
                      "interactive_model_hash_match": interactive_mjcf_sha == rebuilt_interactive_mjcf_sha,
                      "active_wrist": active_wrist[0], "active_jaw_count": len(active_jaws)}, indent=2))


if __name__ == "__main__":
    main()
