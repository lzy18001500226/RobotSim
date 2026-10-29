#!/usr/bin/env python3
"""Generate and statically audit a source-mesh chair detour in the RAICOM room.

This is an evidence-only ``SIMULATION_ONLY`` scenario generator.  It creates
an extra chair *scene wrapper* from the same immutable chair STL already used
by the RAICOM room, compiles it with MuJoCo, and calls ``mj_forward`` only.
It never changes vendor assets, robot mechanics, MC configuration, controls,
or a running simulator.  The generated detour remains an offline plan until a
separately authorized runtime episode proves sensing, re-planning, walking and
the existing physical-stop contract together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import mujoco
import numpy as np

from issue80_lidar_obstacle_layer import (
    GridCell,
    GridSpec,
    ObstacleLayerError,
    path_world_points,
    plan_a_star,
    simplify_grid_path,
)
from issue80_local01_scene_route import (
    _resolve_raicom_reference_root,
    _resolve_runtime_x2_path,
    _world_geom_aabb,
    assert_native_robot_equivalence,
    native_name_scene_tree,
)


# The task worktree is authoritative for scripts and source scene inputs.  The
# shared main checkout may intentionally lack a task branch's materialized
# RAICOM scene, while the helper below resolves immutable reference assets via
# the registered Git worktree list.
WORKTREE = Path(__file__).resolve().parents[2]
SCENE_SOURCE = WORKTREE / "simulation/mujoco/scenes/raicom2026/scene_workspace.xml"
CHAIR_RELATIVE = Path("robot/lx2501_3_t2d5/model_info/object/mesh/chair.stl")
GRID_RESOLUTION_M = 0.10
TRACKING_ENVELOPE_M = 0.05
ADDITIONAL_CLEARANCE_GATE_M = 0.020
ROBOT_ROOT_Z_M = 0.68


class ChairDetourError(RuntimeError):
    """Static scenario generation or full-body clearance check failed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chair_model_xml(chair_mesh: Path) -> ET.Element:
    root = ET.Element("mujoco", {"model": "sim_only_obstacle_chair"})
    ET.SubElement(root, "compiler", {"meshdir": str(chair_mesh.parent), "angle": "degree"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "chair_mesh", "file": chair_mesh.name, "scale": "0.2 0.2 0.2"})
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "chair"})
    ET.SubElement(body, "geom", {
        "name": "chair_collision", "type": "mesh", "mesh": "chair_mesh", "contype": "1", "conaffinity": "1",
    })
    return root


def scene_with_extra_chair(
    scene_source: Path,
    chair_model_path: Path,
    *,
    chair_x_m: float,
    chair_y_m: float,
    chair_yaw_rad: float,
) -> ET.Element:
    tree = native_name_scene_tree(scene_source)
    asset = tree.find("./asset")
    worldbody = tree.find("./worldbody")
    if asset is None or worldbody is None:
        raise ChairDetourError("source scene lacks asset or worldbody")
    if any(item.get("name") == "sim_only_obstacle_chair" for item in asset.findall("model")):
        raise ChairDetourError("source scene unexpectedly already has a dynamic-chair model")
    ET.SubElement(asset, "model", {"name": "sim_only_obstacle_chair", "file": str(chair_model_path)})
    frame = ET.SubElement(worldbody, "frame", {
        "name": "sim_only_obstacle_chair_frame",
        "pos": f"{chair_x_m:.9g} {chair_y_m:.9g} 0",
        "quat": f"{math.cos(chair_yaw_rad / 2.0):.17g} 0 0 {math.sin(chair_yaw_rad / 2.0):.17g}",
    })
    ET.SubElement(frame, "attach", {"model": "sim_only_obstacle_chair", "prefix": "simobs_"})
    return tree


def static_mug_wrapper(mug_source: Path, mug_assets: Path, output_path: Path) -> None:
    """Resolve only the immutable mug's relative asset directories for compilation."""
    tree = ET.parse(mug_source)
    compiler = tree.getroot().find("./compiler")
    if compiler is None:
        raise ChairDetourError("source mug model lacks a compiler asset declaration")
    compiler.set("meshdir", str(mug_assets))
    compiler.set("texturedir", str(mug_assets))
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def robot_body_ids(model: mujoco.MjModel) -> set[int]:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ChairDetourError("compiled scene has no native pelvis body")
    result = {pelvis}
    for body_id in range(pelvis + 1, model.nbody):
        if int(model.body_parentid[body_id]) in result:
            result.add(body_id)
    return result


def collision_geom_ids(model: mujoco.MjModel, robot_ids: set[int]) -> tuple[list[int], list[int]]:
    robot, environment = [], []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in robot_ids:
            robot.append(geom_id)
        elif (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith(("raicom_", "simobs_")):
            environment.append(geom_id)
    if not robot or not environment:
        raise ChairDetourError("compiled scene lacks robot or static-environment collision geoms")
    return robot, environment


def footprint_radius(model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: Iterable[int], root_body: int) -> float:
    root_xy = data.xpos[root_body][:2]
    return max(
        float(np.linalg.norm(data.geom_xpos[geom_id][:2] - root_xy)) + float(model.geom_rbound[geom_id])
        for geom_id in robot_geoms
    )


def chair_aabb(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    matches = [geom_id for geom_id in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or "").startswith("simobs_")]
    if len(matches) != 1:
        raise ChairDetourError(f"expected one attached dynamic chair geom, found {len(matches)}")
    return _world_geom_aabb(model, data, matches[0], np)


def aabb_blocked_cells(grid: GridSpec, lower: np.ndarray, upper: np.ndarray, *, inflation_m: float) -> frozenset[GridCell]:
    if not math.isfinite(inflation_m) or inflation_m < 0:
        raise ChairDetourError("planner inflation must be finite and non-negative")
    result: set[GridCell] = set()
    for ix in range(grid.width):
        for iy in range(grid.height):
            x, y = grid.cell_center(GridCell(ix, iy))
            if lower[0] - inflation_m <= x <= upper[0] + inflation_m and lower[1] - inflation_m <= y <= upper[1] + inflation_m:
                result.add(GridCell(ix, iy))
    return frozenset(result)


def heading_points(points: tuple[tuple[float, float], ...], final_yaw_rad: float) -> tuple[tuple[float, float, float], ...]:
    if len(points) < 2:
        raise ChairDetourError("detour needs at least start and goal")
    output: list[tuple[float, float, float]] = []
    for index, (x, y) in enumerate(points):
        if index + 1 < len(points):
            next_x, next_y = points[index + 1]
            yaw = math.atan2(next_y - y, next_x - x)
        else:
            yaw = final_yaw_rad
        output.append((x, y, yaw))
    return tuple(output)


def segment_samples(a: tuple[float, float, float], b: tuple[float, float, float]) -> Iterable[tuple[float, float, float]]:
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    yaw_delta = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
    count = max(2, math.ceil(distance / 0.025) + 1, math.ceil(abs(yaw_delta) / 0.02) + 1)
    for index in range(count):
        fraction = index / (count - 1)
        yield (
            a[0] + (b[0] - a[0]) * fraction,
            a[1] + (b[1] - a[1]) * fraction,
            a[2] + yaw_delta * fraction,
        )


def static_clearance_audit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    poses: tuple[tuple[float, float, float], ...],
    robot_geoms: list[int],
    environment_geoms: list[int],
    root_joint: int,
) -> dict[str, Any]:
    qposadr = int(model.jnt_qposadr[root_joint])
    environment_boxes: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
    minimum = math.inf
    limiting: dict[str, Any] | None = None
    contacts: list[dict[str, Any]] = []
    sample_count = 0
    for segment_index, (start, end) in enumerate(zip(poses, poses[1:])):
        for x, y, yaw in segment_samples(start, end):
            for yaw_offset in (-0.075, 0.0, 0.075):
                actual_yaw = yaw + yaw_offset
                mujoco.mj_resetData(model, data)
                data.qpos[qposadr:qposadr + 3] = [x, y, ROBOT_ROOT_Z_M]
                data.qpos[qposadr + 3:qposadr + 7] = [math.cos(actual_yaw / 2.0), 0.0, 0.0, math.sin(actual_yaw / 2.0)]
                mujoco.mj_forward(model, data)
                sample_count += 1
                if environment_boxes is None:
                    environment_boxes = {geom: _world_geom_aabb(model, data, geom, np) for geom in environment_geoms}
                for contact in data.contact[:data.ncon]:
                    a, b = int(contact.geom1), int(contact.geom2)
                    if (a in robot_geoms and b in environment_geoms) or (b in robot_geoms and a in environment_geoms):
                        contacts.append({"segment_index": segment_index, "pose": [x, y, actual_yaw], "geom1": a, "geom2": b, "distance_m": float(contact.dist)})
                for robot_geom in robot_geoms:
                    robot_lower, robot_upper = _world_geom_aabb(model, data, robot_geom, np)
                    robot_lower[:2] -= TRACKING_ENVELOPE_M
                    robot_upper[:2] += TRACKING_ENVELOPE_M
                    for environment_geom, (env_lower, env_upper) in environment_boxes.items():
                        if not ((int(model.geom_contype[robot_geom]) & int(model.geom_conaffinity[environment_geom])) or (int(model.geom_contype[environment_geom]) & int(model.geom_conaffinity[robot_geom]))):
                            continue
                        gap = np.maximum(0.0, np.maximum(env_lower - robot_upper, robot_lower - env_upper))
                        clearance = float(np.linalg.norm(gap))
                        if clearance < minimum:
                            minimum = clearance
                            limiting = {"segment_index": segment_index, "robot_geom": robot_geom, "environment_geom": environment_geom, "pose": [x, y, actual_yaw]}
    if environment_boxes is None:
        raise ChairDetourError("static audit did not sample a route")
    return {
        "status": "PASS_STATIC_FULLBODY_DETOUR_CLEARANCE" if not contacts and minimum >= ADDITIONAL_CLEARANCE_GATE_M else "BLOCKED_STATIC_FULLBODY_DETOUR_CLEARANCE",
        "mj_step_calls": 0,
        "samples": sample_count,
        "tracking_envelope_m": TRACKING_ENVELOPE_M,
        "additional_clearance_gate_m": ADDITIONAL_CLEARANCE_GATE_M,
        "minimum_tracking_inflated_aabb_clearance_m": minimum,
        "limiting_pair": limiting,
        "active_robot_environment_contacts": contacts,
        "static_only_limitations": [
            "No gait envelope, actuator dynamics, or MC mode transition is represented.",
            "Heading changes are static poses only; this does not qualify a moving turn primitive.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, already governed evidence directory")
    parser.add_argument("--chair-x-m", type=float, default=-0.25)
    parser.add_argument("--chair-y-m", type=float, default=-1.50)
    parser.add_argument("--chair-yaw-rad", type=float, default=0.0)
    parser.add_argument("--start-x-m", type=float, default=-1.50)
    parser.add_argument("--start-y-m", type=float, default=-1.50)
    parser.add_argument("--goal-x-m", type=float, default=0.90)
    parser.add_argument("--goal-y-m", type=float, default=-1.50)
    parser.add_argument("--goal-yaw-rad", type=float, default=0.0)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=True)
    allowed_existing = {".robotsim-output-reservation.json"}
    if output.is_symlink() or any(item.name not in allowed_existing for item in output.iterdir()):
        parser.error("--output-dir must contain only its governed output reservation")
    if mujoco.__version__ != "3.3.6":
        parser.error(f"requires MuJoCo 3.3.6, got {mujoco.__version__}")
    values = (args.chair_x_m, args.chair_y_m, args.chair_yaw_rad, args.start_x_m, args.start_y_m, args.goal_x_m, args.goal_y_m, args.goal_yaw_rad)
    if not all(math.isfinite(value) for value in values):
        parser.error("scenario coordinates must be finite")
    try:
        scene = SCENE_SOURCE.resolve(strict=True)
        reference_root = _resolve_raicom_reference_root(WORKTREE)
        chair_mesh = (reference_root / CHAIR_RELATIVE).resolve(strict=True)
        if chair_mesh.is_symlink() or not chair_mesh.is_file():
            raise ChairDetourError("source chair mesh is missing or unsafe")
        chair_model = output / "sim_only_dynamic_chair.xml"
        chair_tree = ET.ElementTree(chair_model_xml(chair_mesh))
        ET.indent(chair_tree, space="  ")
        chair_tree.write(chair_model, encoding="utf-8", xml_declaration=True)
        mug_model = output / "mug_portable_static_runtime.xml"
        static_mug_wrapper(
            scene.parent / "mug_portable.xml",
            reference_root / "robot/lx2501_3_t2d5/model_info/object/model/mug",
            mug_model,
        )
        scene_tree = scene_with_extra_chair(
            scene, chair_model, chair_x_m=args.chair_x_m, chair_y_m=args.chair_y_m, chair_yaw_rad=args.chair_yaw_rad,
        )
        mug_include = [item for item in scene_tree.findall("./asset/model") if item.get("name") == "raicom_mug"]
        if len(mug_include) != 1:
            raise ChairDetourError("native scene has no unique mug model include")
        mug_include[0].set("file", str(mug_model))
        scene_output = output / "scene_with_sim_only_chair.xml"
        ET.indent(ET.ElementTree(scene_tree), space="  ")
        ET.ElementTree(scene_tree).write(scene_output, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene_tree, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        native_scene = _resolve_runtime_x2_path(WORKTREE).with_name("scene.xml")
        native = mujoco.MjModel.from_xml_path(str(native_scene))
        robot_equivalence = assert_native_robot_equivalence(native, model)
        root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        robot_ids = robot_body_ids(model)
        robot_geoms, environment_geoms = collision_geom_ids(model, robot_ids)
        radius = footprint_radius(model, data, robot_geoms, root_body)
        chair_lower, chair_upper = chair_aabb(model, data)
        grid = GridSpec(-1.85, -1.85, GRID_RESOLUTION_M, 38, 38)
        blocked = aabb_blocked_cells(grid, chair_lower, chair_upper, inflation_m=radius + TRACKING_ENVELOPE_M)
        start = grid.world_to_cell(args.start_x_m, args.start_y_m)
        goal = grid.world_to_cell(args.goal_x_m, args.goal_y_m)
        raw_path = plan_a_star(start, goal, grid=grid, blocked=blocked)
        path = simplify_grid_path(raw_path, blocked=blocked)
        pose_path = heading_points(path_world_points(path, grid), args.goal_yaw_rad)
        clearance = static_clearance_audit(
            model, data, poses=pose_path, robot_geoms=robot_geoms, environment_geoms=environment_geoms, root_joint=root_joint,
        )
    except (OSError, ValueError, ObstacleLayerError, ChairDetourError, mujoco.FatalError) as exc:
        parser.error(f"static chair-detour audit failed: {type(exc).__name__}: {exc}")
    result = {
        "schema": "robotsim.issue80.raicom-chair-detour-static.v1",
        "classification": "SIMULATION_ONLY_SOURCE_MESH_CHAIR_STATIC_FULLBODY_AUDIT",
        "status": clearance["status"],
        "source": {"scene": str(scene), "scene_sha256": sha256(scene), "chair_mesh": str(chair_mesh), "chair_mesh_sha256": sha256(chair_mesh)},
        "generated": {"scene": str(scene_output), "chair_model": str(chair_model), "mug_wrapper": str(mug_model)},
        "chair_pose": {"x_m": args.chair_x_m, "y_m": args.chair_y_m, "yaw_rad": args.chair_yaw_rad, "aabb": {"lower": chair_lower.tolist(), "upper": chair_upper.tolist()}},
        "native_robot_equivalence": robot_equivalence,
        "planner": {"algorithm": "A_STAR_8_CONNECTED", "grid_resolution_m": GRID_RESOLUTION_M, "robot_radial_upper_bound_m": radius, "tracking_envelope_m": TRACKING_ENVELOPE_M, "inflated_chair_blocked_cell_count": len(blocked), "raw_path_cell_count": len(raw_path), "simplified_path_cell_count": len(path), "map_waypoints": [{"x_m": x, "y_m": y, "yaw_rad": yaw} for x, y, yaw in pose_path]},
        "static_fullbody_clearance": clearance,
        "velocity_publisher_created": False,
        "mc_action_or_source_client_created": False,
    }
    (output / "chair_detour_static_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if clearance["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
