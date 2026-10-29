#!/usr/bin/env python3
"""Create deterministic source-preserving RAICOM unknown-obstacle scenarios.

The static map scene retains the known room/table.  The runtime scene adds
only evidence-local source-chair instances and primitive static barricades;
the latter are deliberately omitted from the static map and must be detected
by LiDAR at runtime.  X2 geometry, MC files, and source scene assets are
never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from issue80_local01_scene_route import (
    _resolve_raicom_reference_root,
    _resolve_runtime_x2_path,
    native_name_scene_tree,
)
from issue80_raicom_chair_detour_static import static_mug_wrapper


WORKTREE = Path(__file__).resolve().parents[2]
SCENE_SOURCE = WORKTREE / "simulation/mujoco/scenes/raicom2026/scene_workspace.xml"
CHAIR_RELATIVE = Path("robot/lx2501_3_t2d5/model_info/object/mesh/chair.stl")
ROOM_RELATIVE = Path("robot/lx2501_3_t2d5/model_info/scene/room/room_with_chair_review.xml")
START_XY = (-1.5, -1.5)
TABLE_APPROACH_XY = (0.9, -1.5)
# The generated scene is later fed to a 0.10 m occupancy grid which inflates
# novel returns by 0.465657 m.  The largest source object in this fixture is
# the 0.05 x 0.45 m barricade: its planar half diagonal is 0.45277 m.  The
# 1.05 m endpoint exclusion therefore leaves a small grid-quantisation
# margin after object extent and route-envelope inflation.  Earlier 0.75 m
# centre-only checks allowed an inflated barricade to cover the start cell;
# that made a correctly fail-closed planner look like a sensing failure.
ENDPOINT_EXCLUSION_M = 1.05
OBJECT_SEPARATION_M = 1.10
SAMPLE_X_RANGE_M = (-1.40, 0.30)
SAMPLE_Y_RANGE_M = (-2.50, -0.50)
ALIGNED_START_QUAT_WXYZ = "1 0 0 0"


class ScenarioError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chair_model_xml(chair_mesh: Path) -> ET.Element:
    root = ET.Element("mujoco", {"model": "sim_only_unknown_chair"})
    ET.SubElement(root, "compiler", {"meshdir": str(chair_mesh.parent), "angle": "degree"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "chair_mesh", "file": chair_mesh.name, "scale": "0.2 0.2 0.2"})
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "chair"})
    ET.SubElement(body, "geom", {
        "name": "chair_collision", "type": "mesh", "mesh": "chair_mesh",
        "contype": "1", "conaffinity": "1",
    })
    return root


def _safe(candidate: tuple[float, float], accepted: list[tuple[float, float]]) -> bool:
    protected = [START_XY, TABLE_APPROACH_XY, *accepted]
    return all(
        math.dist(candidate, item) >= (
            OBJECT_SEPARATION_M if item in accepted else ENDPOINT_EXCLUSION_M
        )
        for item in protected
    )


def sample_obstacles(*, seed: int, chair_count: int, barrier_count: int) -> tuple[list[dict], list[dict]]:
    if chair_count < 0 or barrier_count < 0 or chair_count + barrier_count > 8:
        raise ScenarioError("obstacle counts must be in [0, 8] in total")
    rng = random.Random(seed)
    accepted: list[tuple[float, float]] = []
    chairs, barriers = [], []
    # Candidate region stays inside the known room and preserves a direct
    # start/table safety buffer.  A later map planner may still choose any
    # collision-free detour; this generator does not manufacture a route.
    for index in range(chair_count + barrier_count):
        chosen = None
        for _ in range(500):
            candidate = (
                round(rng.uniform(*SAMPLE_X_RANGE_M), 2),
                round(rng.uniform(*SAMPLE_Y_RANGE_M), 2),
            )
            if _safe(candidate, accepted):
                chosen = candidate
                break
        if chosen is None:
            raise ScenarioError("could not sample a separated unknown obstacle layout")
        accepted.append(chosen)
        yaw = round(rng.choice((0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)), 6)
        if index < chair_count:
            chairs.append({"id": f"UNKNOWN_CHAIR_{index:02d}", "x_m": chosen[0], "y_m": chosen[1], "yaw_rad": yaw})
        else:
            barriers.append({"id": f"UNKNOWN_BARRIER_{index-chair_count:02d}", "x_m": chosen[0], "y_m": chosen[1], "yaw_rad": yaw, "size_m": [0.05, 0.45, 0.55]})
    return chairs, barriers


def dynamic_scene(*, chair_mesh: Path, reference_root: Path, chairs: list[dict], barriers: list[dict], output: Path) -> None:
    tree = native_name_scene_tree(SCENE_SOURCE)
    asset = tree.find("./asset")
    worldbody = tree.find("./worldbody")
    if asset is None or worldbody is None:
        raise ScenarioError("source scene lacks asset/worldbody")
    start_frame = worldbody.find("./frame[@name='x2_start_frame']")
    if start_frame is None:
        raise ScenarioError("source scene lacks x2_start_frame")
    # This is the same reviewed pre-physics aligned-start wrapper used by the
    # TABLE_APPROACH and LiDAR background runs. It avoids treating a 90-degree
    # initial pose rotation as an unknown environment change.
    start_frame.set("quat", ALIGNED_START_QUAT_WXYZ)
    x2_model = _resolve_runtime_x2_path(WORKTREE)
    room_model = (reference_root / ROOM_RELATIVE).resolve(strict=True)
    mug_source = SCENE_SOURCE.parent / "mug_portable.xml"
    mug_wrapper = output.with_name("mug_portable_static_runtime.xml")
    if not mug_wrapper.exists():
        static_mug_wrapper(
            mug_source,
            reference_root / "robot/lx2501_3_t2d5/model_info/object/model/mug",
            mug_wrapper,
        )
    includes = {item.get("name"): item for item in asset.findall("model")}
    if set(("x2_mc_runtime", "raicom_room", "raicom_mug")) - set(includes):
        raise ScenarioError("source scene lacks one required model include")
    includes["x2_mc_runtime"].set("file", str(x2_model))
    includes["raicom_room"].set("file", str(room_model))
    includes["raicom_mug"].set("file", str(mug_wrapper))
    model_path = output.with_name("sim_only_unknown_chair.xml")
    chair_tree = ET.ElementTree(chair_model_xml(chair_mesh))
    ET.indent(chair_tree, space="  ")
    chair_tree.write(model_path, encoding="utf-8", xml_declaration=True)
    ET.SubElement(asset, "model", {"name": "sim_only_unknown_chair", "file": str(model_path)})
    for chair in chairs:
        frame = ET.SubElement(worldbody, "frame", {
            "name": chair["id"],
            "pos": f"{chair['x_m']} {chair['y_m']} 0",
            "quat": f"{math.cos(chair['yaw_rad']/2):.17g} 0 0 {math.sin(chair['yaw_rad']/2):.17g}",
        })
        ET.SubElement(frame, "attach", {"model": "sim_only_unknown_chair", "prefix": chair["id"].lower() + "_"})
    for barrier in barriers:
        ET.SubElement(worldbody, "geom", {
            "name": barrier["id"].lower(), "type": "box",
            "pos": f"{barrier['x_m']} {barrier['y_m']} {barrier['size_m'][2]}",
            "euler": f"0 0 {barrier['yaw_rad']}",
            "size": " ".join(str(value) for value in barrier["size_m"]),
            "contype": "1", "conaffinity": "1", "rgba": "0.8 0.2 0.1 1",
        })
    ET.indent(tree, space="  ")
    ET.ElementTree(tree).write(output, encoding="utf-8", xml_declaration=True)


def generate(output: Path, *, seed: int, chair_count: int, barrier_count: int) -> dict:
    allowed = {".robotsim-output-reservation.json"}
    if output.is_symlink() or any(item.name not in allowed for item in output.iterdir()):
        raise ScenarioError("output directory must contain only its governed reservation")
    scene = SCENE_SOURCE.resolve(strict=True)
    reference_root = _resolve_raicom_reference_root(WORKTREE)
    chair_mesh = (reference_root / CHAIR_RELATIVE).resolve(strict=True)
    if chair_mesh.is_symlink() or not chair_mesh.is_file():
        raise ScenarioError("source chair mesh is missing or unsafe")
    chairs, barriers = sample_obstacles(seed=seed, chair_count=chair_count, barrier_count=barrier_count)
    static_scene = output / "scene_static_known_map.xml"
    dynamic_scene(chair_mesh=chair_mesh, reference_root=reference_root, chairs=[], barriers=[], output=static_scene)
    runtime_scene = output / "scene_runtime_unknown_obstacles.xml"
    dynamic_scene(chair_mesh=chair_mesh, reference_root=reference_root, chairs=chairs, barriers=barriers, output=runtime_scene)
    return {
        "schema": "robotsim.issue80.raicom-random-obstacle-scenario.v1",
        "classification": "SIMULATION_ONLY_KNOWN_STATIC_MAP_UNKNOWN_RUNTIME_OBSTACLES",
        "status": "PASS_SCENARIO_GENERATED",
        "seed": seed,
        "known_static": {
            "scene": str(static_scene),
            "scene_sha256": sha256(static_scene),
            "table_approach_target_odom": {"x_m": TABLE_APPROACH_XY[0], "y_m": TABLE_APPROACH_XY[1], "yaw_rad": 0.0},
        },
        "derived_start": {
            "classification": "SIMULATION_ONLY_ALIGNED_START_PRE_PHYSICS",
            "x2_start_frame_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source_start_yaw_rad": math.pi / 2.0,
            "active_qpos_write": False,
        },
        "sampling_contract": {
            "endpoint_exclusion_m": ENDPOINT_EXCLUSION_M,
            "object_separation_m": OBJECT_SEPARATION_M,
            "sample_x_range_m": list(SAMPLE_X_RANGE_M),
            "sample_y_range_m": list(SAMPLE_Y_RANGE_M),
            "meaning": "centre-distance screening for source object footprint plus route-envelope inflation; it does not replace full-body clearance validation",
        },
        "runtime_unknown_obstacles": {
            "scene": str(runtime_scene),
            "scene_sha256": sha256(runtime_scene),
            "chairs": chairs,
            "barriers": barriers,
        },
        "source_assets": {"chair_mesh": str(chair_mesh), "chair_mesh_sha256": sha256(chair_mesh)},
        "limits": [
            "Unknown obstacles are absent from scene_static_known_map.xml.",
            "The known table target remains the source TABLE_APPROACH pose in this scenario version.",
            "No X2/MC/source asset is modified.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chair-count", type=int, default=2)
    parser.add_argument("--barrier-count", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = generate(args.output_dir.resolve(strict=True), seed=args.seed, chair_count=args.chair_count, barrier_count=args.barrier_count)
    except (OSError, ScenarioError) as exc:
        parser.error(str(exc))
    result_path = args.output_dir / "scenario.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
