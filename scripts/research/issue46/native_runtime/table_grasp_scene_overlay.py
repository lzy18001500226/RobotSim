#!/usr/bin/env python3
"""Attach the RAICOM table room and approved true-pose cylinder to a native STL-hand overlay.

This is an evidence-local, source-preserving composition. It retains the
native RAICOM X2 body needed by the delivered MC, the frozen bilateral STL
hand overlay, and the approved upright cylinder. It never edits vendor files,
the frozen gripper package, or the approved review fixture. No dynamics run in
this builder; successful output is only a static table-side preflight.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
CONTAINER_EVIDENCE = "/workspace/evidence"
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hashed_file(path: Path, expected: str, *, label: str) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return path.resolve()


def exact_hand_overlay_inputs(result_path: Path) -> dict[str, Path]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("classification") not in {
        "SIMULATION_ONLY_NATIVE_MC_HAND_BRIDGE_PREPARED",
    }:
        raise ValueError("not a prepared native STL hand overlay")
    gates = payload.get("gates", {})
    required = ("FROZEN_STL_MOUNT_INPUT", "HOST_STATIC_MODEL_COMPILE", "NATIVE_HAND_CHANNEL_CONFIGURATION")
    if any(gates.get(name) != "PASS" for name in required):
        raise ValueError("hand overlay prerequisites did not pass")
    generated = payload.get("generated")
    if not isinstance(generated, dict):
        raise ValueError("hand overlay has no generated manifest")
    result: dict[str, Path] = {}
    for name in ("host_scene", "runtime_scene", "simulator_config", "robot_config"):
        item = generated.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"hand overlay lacks {name}")
        result[name] = require_hashed_file(Path(str(item.get("path", ""))), str(item.get("sha256", "")), label=f"hand overlay {name}")
    return result


def find_body(root: ET.Element, name: str) -> ET.Element:
    matches = [body for body in root.findall(".//body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one body named {name}, found {len(matches)}")
    return matches[0]


def vector(text: str | None, *, label: str) -> np.ndarray:
    if not text:
        raise ValueError(f"missing {label}")
    values = np.fromstring(text, sep=" ")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"invalid {label}: {text!r}")
    return values


ROOM_PREFIX = "raicom_table_"


def room_asset_path(room_source: Path, raw: str, *, runtime: bool) -> tuple[str, dict[str, str]]:
    """Return one safe room asset path and its immutable provenance record."""
    source = (room_source.parent / raw).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"room asset is missing or unsafe: {source}")
    try:
        relative = source.relative_to(EVIDENCE_REPO)
    except ValueError as exc:
        raise ValueError(f"room asset lies outside approved runtime mount: {source}") from exc
    target = "/workspace/robotsim/" + relative.as_posix() if runtime else str(source)
    return target, {"from": raw, "to": target, "sha256": sha256(source)}


def inline_room(
    scene_source: Path,
    room_source: Path,
    room_frame: np.ndarray,
    cylinder: ET.Element | None,
    *,
    runtime: bool,
) -> tuple[bytes, list[dict[str, str]]]:
    """Inline the source room without MuJoCo ``attach`` model instances.

    The delivered native RAICOM simulator publishes empty state arrays when a
    scene contains an ``asset/model`` attachment.  This function preserves the
    manufacturer room geometry while copying its static assets and worldbody
    nodes into the already working native X2 scene.  It deliberately introduces
    no room joints or degrees of freedom.
    """
    scene_tree = ET.parse(scene_source)
    room_tree = ET.parse(room_source)
    root, room_root = scene_tree.getroot(), room_tree.getroot()
    asset, worldbody = root.find("asset"), root.find("worldbody")
    room_asset, room_worldbody = room_root.find("asset"), room_root.find("worldbody")
    if asset is None or worldbody is None or room_asset is None or room_worldbody is None:
        raise ValueError("scene or room source lacks asset/worldbody")
    if root.find(".//body[@name='raicom_table_room_frame']") is not None:
        raise ValueError("hand overlay already contains an inlined competition room")
    if cylinder is not None and root.find(".//body[@name='test_cylinder']") is not None:
        raise ValueError("hand overlay already contains test_cylinder")

    rebindings: list[dict[str, str]] = []
    asset_names: dict[str, str] = {}
    for source_asset in room_asset:
        clone = copy.deepcopy(source_asset)
        source_name = source_asset.get("name")
        if source_name:
            clone.set("name", ROOM_PREFIX + source_name)
            asset_names[source_name] = ROOM_PREFIX + source_name
        raw = source_asset.get("file")
        if raw:
            if Path(raw).is_absolute():
                raise ValueError("room asset paths must be source-relative")
            target, record = room_asset_path(room_source, raw, runtime=runtime)
            clone.set("file", target)
            record["name"] = clone.get("name", "")
            rebindings.append(record)
        asset.append(clone)
    if not rebindings:
        raise ValueError("room source has no relative assets to rebind")
    for material in asset.findall("material"):
        texture = material.get("texture")
        if texture in asset_names:
            material.set("texture", asset_names[texture])

    def remap_world_names(element: ET.Element) -> None:
        name = element.get("name")
        if name:
            element.set("name", ROOM_PREFIX + name)
        for attribute in ("material", "mesh", "texture"):
            value = element.get(attribute)
            if value in asset_names:
                element.set(attribute, asset_names[value])
        for child in element:
            remap_world_names(child)

    frame = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "raicom_table_room_frame",
            "pos": " ".join(f"{value:.17g}" for value in room_frame),
            "quat": "1 0 0 0",
        },
    )
    for source_child in room_worldbody:
        clone = copy.deepcopy(source_child)
        remap_world_names(clone)
        frame.append(clone)
    if cylinder is not None:
        worldbody.append(cylinder)
    ET.indent(scene_tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), rebindings


def inline_source_floor(scene_bytes: bytes, floor_source: Path) -> tuple[bytes, dict[str, str]]:
    """Copy the vendor floor texture/material/geom into a composed physics scene."""
    tree = ET.ElementTree(ET.fromstring(scene_bytes))
    source_tree = ET.parse(floor_source)
    root, source_root = tree.getroot(), source_tree.getroot()
    asset, worldbody = root.find("asset"), root.find("worldbody")
    source_asset, source_worldbody = source_root.find("asset"), source_root.find("worldbody")
    if asset is None or worldbody is None or source_asset is None or source_worldbody is None:
        raise ValueError("scene or floor source lacks asset/worldbody")
    source_floor = source_worldbody.find("geom[@name='floor']")
    if source_floor is None or root.find(".//geom[@name='raicom_floor']") is not None:
        raise ValueError("source floor is absent or already inlined")
    material_name = source_floor.get("material")
    source_material = source_asset.find(f"material[@name='{material_name}']") if material_name else None
    if source_material is None:
        raise ValueError("source floor material is absent")
    texture_name = source_material.get("texture")
    source_texture = source_asset.find(f"texture[@name='{texture_name}']") if texture_name else None
    if source_texture is None:
        raise ValueError("source floor texture is absent")
    texture = copy.deepcopy(source_texture)
    texture.set("name", "raicom_floor_" + texture_name)
    material = copy.deepcopy(source_material)
    material.set("name", "raicom_floor_" + material_name)
    material.set("texture", texture.get("name", ""))
    floor = copy.deepcopy(source_floor)
    floor.set("name", "raicom_floor")
    floor.set("material", material.get("name", ""))
    asset.extend((texture, material))
    worldbody.append(floor)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), {"floor_geom": "raicom_floor", "source_material": material_name, "source_texture": texture_name}


def append_diagnostic_primitive_table(scene_source: Path, table_world: np.ndarray) -> bytes:
    """Append one static box only to isolate native MC environment loading.

    This source-independent construction is diagnostic-only; it is never a
    grasp scene and intentionally contains no added degrees of freedom.
    """
    tree = ET.parse(scene_source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("hand overlay scene lacks worldbody")
    if root.find(".//geom[@name='diagnostic_primitive_table_top']") is not None:
        raise ValueError("hand overlay already contains diagnostic primitive table")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "diagnostic_primitive_table_top",
            "type": "box",
            "size": "0.3 0.3 0.025",
            "pos": " ".join(f"{value:.17g}" for value in table_world + np.array([0.0, 0.0, 0.025])),
            "rgba": "0.5 0.5 0.5 1",
        },
    )
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def table_frame_and_cylinder(
    approved_scene: Path,
    host_scene: Path,
    room_source: Path,
    *,
    table_world_override: np.ndarray | None = None,
) -> tuple[np.ndarray, ET.Element, dict[str, object]]:
    approved_root = ET.parse(approved_scene).getroot()
    host_root = ET.parse(host_scene).getroot()
    room_root = ET.parse(room_source).getroot()
    approved_pelvis = vector(find_body(approved_root, "pelvis").get("pos"), label="approved pelvis position")
    native_pelvis = vector(find_body(host_root, "pelvis").get("pos"), label="native pelvis position")
    approved_table = vector(find_body(approved_root, "raicom_table").get("pos"), label="approved table position")
    room_table = vector(find_body(room_root, "table").get("pos"), label="room table position")
    cylinder = copy.deepcopy(find_body(approved_root, "test_cylinder"))
    approved_cylinder = vector(cylinder.get("pos"), label="approved cylinder position")
    # Match the approved table-to-pelvis and cylinder-to-pelvis transforms in
    # the native MC X2 world. The room source table becomes the approved table.
    source_aligned_table = native_pelvis + (approved_table - approved_pelvis)
    desired_table = np.asarray(table_world_override, dtype=float) if table_world_override is not None else source_aligned_table
    if desired_table.shape != (3,) or not np.isfinite(desired_table).all():
        raise ValueError("table world override must be a finite three-vector")
    room_frame = desired_table - room_table
    source_aligned_cylinder = native_pelvis + (approved_cylinder - approved_pelvis)
    cylinder_world = source_aligned_cylinder + (desired_table - source_aligned_table)
    cylinder.set("pos", " ".join(f"{value:.17g}" for value in cylinder_world))
    return room_frame, cylinder, {
        "approved_pelvis_xyz_m": approved_pelvis.tolist(),
        "native_pelvis_xyz_m": native_pelvis.tolist(),
        "approved_table_xyz_m": approved_table.tolist(),
        "room_source_table_xyz_m": room_table.tolist(),
        "source_aligned_table_xyz_m": source_aligned_table.tolist(),
        "room_frame_xyz_m": room_frame.tolist(),
        "source_aligned_cylinder_xyz_m": source_aligned_cylinder.tolist(),
        "cylinder_xyz_m": cylinder_world.tolist(),
        "cylinder_table_translation_xyz_m": (desired_table - source_aligned_table).tolist(),
        "table_world_override": None if table_world_override is None else desired_table.tolist(),
    }


def compile_static(scene: Path, *, require_cylinder: bool, require_source_table: bool) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.xpos).all():
        raise ValueError("table scene initializes with non-finite state")
    if require_cylinder and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder") < 0:
        raise ValueError("table scene lacks the true-pose test cylinder")
    if require_source_table and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "raicom_table_table") < 0:
        raise ValueError("table scene lacks the inlined RAICOM table")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "raicom_floor") < 0:
        raise ValueError("table scene lacks the source RAICOM floor")
    return {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "neq": int(model.neq), "ncon": int(data.ncon)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-overlay-result", required=True, type=Path)
    parser.add_argument("--approved-object-scene", required=True, type=Path)
    parser.add_argument("--approved-object-scene-sha256", required=True)
    parser.add_argument("--room-source", required=True, type=Path)
    parser.add_argument("--room-source-sha256", required=True)
    parser.add_argument("--floor-source", required=True, type=Path)
    parser.add_argument("--floor-source-sha256", required=True)
    parser.add_argument("--table-world-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"), help="optional collision-free table position from static placement search")
    parser.add_argument(
        "--without-cylinder",
        action="store_true",
        help="diagnostic-only: compose the table room without the free cylinder to isolate native MC model-loading behavior",
    )
    parser.add_argument(
        "--diagnostic-primitive-table",
        action="store_true",
        help="diagnostic-only: replace the source room with one static box to isolate native MC environment compatibility",
    )
    args = parser.parse_args()
    inputs = exact_hand_overlay_inputs(args.hand_overlay_result)
    approved_scene = require_hashed_file(args.approved_object_scene, args.approved_object_scene_sha256, label="approved object scene")
    room_source = require_hashed_file(args.room_source, args.room_source_sha256, label="RAICOM room source")
    floor_source = require_hashed_file(args.floor_source, args.floor_source_sha256, label="RAICOM floor source")
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-standing-table-overlay").native_path
    override = None if args.table_world_xyz is None else np.asarray(args.table_world_xyz, dtype=float)
    room_frame, cylinder, transforms = table_frame_and_cylinder(approved_scene, inputs["host_scene"], room_source, table_world_override=override)
    host_scene = output / "scene_stl_native_hand_table_host.xml"
    runtime_scene = output / "scene_stl_native_hand_table_runtime.xml"
    composed_cylinder = None if args.without_cylinder else cylinder
    table_world = np.asarray(transforms["table_world_override"], dtype=float) if transforms["table_world_override"] is not None else room_frame + vector(find_body(ET.parse(room_source).getroot(), "table").get("pos"), label="room table position")
    if args.diagnostic_primitive_table:
        if composed_cylinder is not None:
            raise ValueError("diagnostic primitive table requires --without-cylinder")
        host_scene_bytes = append_diagnostic_primitive_table(inputs["host_scene"], table_world)
        runtime_scene_bytes = append_diagnostic_primitive_table(inputs["runtime_scene"], table_world)
        host_rebindings, runtime_rebindings = [], []
        composition_method = "diagnostic_primitive_static_box"
    else:
        host_scene_bytes, host_rebindings = inline_room(inputs["host_scene"], room_source, room_frame, composed_cylinder, runtime=False)
        runtime_scene_bytes, runtime_rebindings = inline_room(inputs["runtime_scene"], room_source, room_frame, composed_cylinder, runtime=True)
        composition_method = "inline_static_source_room_no_model_attach"
    host_scene_bytes, floor_host = inline_source_floor(host_scene_bytes, floor_source)
    runtime_scene_bytes, floor_runtime = inline_source_floor(runtime_scene_bytes, floor_source)
    pg.write_new_bytes(host_scene, host_scene_bytes)
    pg.write_new_bytes(runtime_scene, runtime_scene_bytes)
    model = compile_static(
        host_scene,
        require_cylinder=not args.without_cylinder,
        require_source_table=not args.diagnostic_primitive_table,
    )
    simulator = yaml.safe_load(inputs["simulator_config"].read_text(encoding="utf-8"))
    module = simulator.get("MujocoSimModule") if isinstance(simulator, dict) else None
    if not isinstance(module, dict):
        raise ValueError("hand overlay simulator config lacks MujocoSimModule")
    module["simulation_model"] = f"{CONTAINER_EVIDENCE}/{runtime_scene.name}"
    simulator_output = output / "simulator_stl_native_hand_table.yaml"
    robot_output = output / "robot_stl_native_hand_table.yaml"
    pg.write_new_text(simulator_output, yaml.safe_dump(simulator, allow_unicode=True, sort_keys=False))
    pg.write_new_bytes(robot_output, inputs["robot_config"].read_bytes())
    result = {
        "schema": "robotsim.issue46.raicom-stl-standing-table-overlay.v1",
        "classification": "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED",
        "mj_step_calls": 0,
        "hand_overlay_result": {"path": str(args.hand_overlay_result), "sha256": sha256(args.hand_overlay_result)},
        "approved_object_scene": {"path": str(approved_scene), "sha256": sha256(approved_scene)},
        "room_source": {"path": str(room_source), "sha256": sha256(room_source)},
        "floor_source": {"path": str(floor_source), "sha256": sha256(floor_source), "host": floor_host, "runtime": floor_runtime},
        "room_composition": {
            "method": composition_method,
            "host_asset_rebindings": host_rebindings,
            "runtime_asset_rebindings": runtime_rebindings,
        },
        "transforms": transforms,
        "cylinder": {
            "included": not args.without_cylinder,
            "diagnostic_only": args.without_cylinder,
            "source_pose_xyz_m": transforms["cylinder_xyz_m"],
        },
        "generated": {
            "host_scene": {"path": str(host_scene), "sha256": sha256(host_scene)},
            "runtime_scene": {"path": str(runtime_scene), "sha256": sha256(runtime_scene)},
            "simulator_config": {"path": str(simulator_output), "sha256": sha256(simulator_output)},
            "robot_config": {"path": str(robot_output), "sha256": sha256(robot_output)},
        },
        "model": model,
        "gates": {
            "FROZEN_STL_MOUNT_INPUT": "PASS",
            "HOST_STATIC_MODEL_COMPILE": "PASS",
            "NATIVE_HAND_CHANNEL_CONFIGURATION": "PASS",
            "TABLE_SCENE_STATIC_COMPILE": "PASS",
            "SOURCE_ROOM_INLINED": "NOT_RUN" if args.diagnostic_primitive_table else "PASS",
            "TRUE_POSE_CYLINDER_PRESENT": "NOT_RUN" if args.without_cylinder else "PASS",
            "NATIVE_MC_FULL_ENVIRONMENT": "FAIL",
            "NATIVE_STANDING_TABLE_GRASP": "NOT_RUN",
            "CONTACT_DRIVEN_LIFT": "NOT_RUN",
        },
        "limitations": [
            "The native RAICOM X2 body is retained for delivered MC compatibility; this evidence-local scene is not the human-approved visual fixture.",
            "The table/cylinder transforms are source-derived from the approved static review and require physical standing/arm validation before any grasp claim.",
            "No vendor or frozen STL model file is modified.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "model": model}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
