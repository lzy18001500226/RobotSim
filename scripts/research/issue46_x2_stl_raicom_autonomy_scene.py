#!/usr/bin/env python3
"""Build a static RAICOM table scene with bilateral reviewed STL OmniPickers.

The delivered RAICOM scene attaches its unchanged ``x2.xml`` directly and
therefore retains its historical hand geometry.  This research builder instead
creates an evidence-local X2 XML with the two reviewed STL hands mounted at the
official wrist origins, then attaches that generated X2 model to the preserved
RAICOM room and mug.  It calls :func:`mujoco.mj_forward` only: no physics step,
navigation command, arm command, or object pose write happens after setup.

The generated scene is a review/integration gate.  It establishes that the
navigation scene selects the same ``X2-OmniPicker-STL`` model the maintainer
accepted; it does not establish navigation, standing balance, perception, or
physical grasp success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
MODEL_ID = "X2-OmniPicker-STL"
CONFIG_DEFAULT = REPO / "simulation/mujoco/scenes/raicom2026/config/x2_stl_autonomy_scene.json"

sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from scripts.research import issue46_raicom_stl_mount_audit as mount_audit  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    LEFT_JOINT_MAPPING,
    load_registered_trajectory,
)


COLLISION_RENDER_GROUPS = (3, mount_audit.STL_COLLISION_RENDER_GROUP)


PREFIXED_INHERITED_EXCLUSIONS = (
    ("x2_left_wrist_roll_link", "x2_L_hand_narrow3_Link"),
    ("x2_right_wrist_roll_link", "x2_R_hand_narrow3_Link"),
    *(tuple("x2_" + name for name in pair) for pair in mount_audit.RIGHT_EXCLUSIONS),
    *(tuple("x2_" + mount_audit.right_to_left(name) for name in pair) for pair in mount_audit.RIGHT_EXCLUSIONS),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_source(repo_root: Path, record: dict[str, object], *, label: str) -> Path:
    raw_path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"{label} source record is incomplete")
    path = (repo_root / raw_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} source is missing or not a regular file: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} source hash mismatch: expected {expected}, got {observed}")
    return path


def load_scene_config(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"scene config is missing or unsafe: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded.get("schema") != "robotsim.raicom2026.x2-stl-autonomy-scene.v1":
        raise ValueError("scene config has an unknown schema")
    if loaded.get("classification") != "SIMULATION_ONLY_STATIC_AUTONOMY_SCENE_INPUT":
        raise ValueError("scene config has an unexpected qualification")
    if not isinstance(loaded.get("sources"), dict) or not isinstance(loaded.get("poses"), dict):
        raise ValueError("scene config lacks sources or poses")
    return loaded


def yaw_euler(yaw_rad: float) -> str:
    if not math.isfinite(yaw_rad):
        raise ValueError("scene yaw must be finite")
    return f"0 0 {yaw_rad:.17g}"


def vec3(record: object, *, label: str) -> tuple[float, float, float]:
    if not isinstance(record, list) or len(record) != 3:
        raise ValueError(f"{label} must be a three-element vector")
    values = tuple(float(value) for value in record)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
    return values


def add_stl_hands(
    *,
    raicom_model: Path,
    official_urdf: Path,
    stl_model: Path,
    output: Path,
) -> dict[str, object]:
    """Build a bilateral STL hand overlay around the unchanged RAICOM X2 body."""
    mounts = mount_audit.official_mounts(official_urdf)
    tree = ET.parse(raicom_model)
    root = tree.getroot()
    removed_hand_roots = remove_existing_hand_roots(root, mounts)
    asset_rebindings = mount_audit.remap_mesh_paths(root, raicom_model)
    stl_root = ET.parse(stl_model).getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("RAICOM X2 source lacks an asset section")
    for side in ("right", "left"):
        for mesh in mount_audit.stl_assets(stl_root, stl_model, side):
            asset.append(mesh)
        mount_audit.add_hand(root, stl_root, side=side, mount=mounts[side])
    mount_audit.add_internal_exclusions(root)
    ET.indent(tree, space="  ")
    pg.write_new_bytes(output, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {
        "path": str(output),
        "sha256": sha256(output),
        "official_wrist_mounts": mounts,
        "preexisting_hand_roots_removed": removed_hand_roots,
        "raicom_asset_rebindings": asset_rebindings,
    }


def remove_existing_hand_roots(root: ET.Element, mounts: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Detach only an exact old OmniPicker root before adding its STL replacement.

    The delivered RAICOM input currently ends at each wrist and therefore
    returns an empty list.  Keeping this check here prevents a future scene
    input that already embeds the standard OmniPicker roots from rendering or
    colliding with a second copy of the reviewed STL hand.  Wrist geometry is
    deliberately not touched.
    """
    removed: list[dict[str, str]] = []
    for side, root_name in (("right", "R_omnipicker_base_link"), ("left", "L_omnipicker_base_link")):
        mount = mounts.get(side)
        if not isinstance(mount, dict):
            raise ValueError(f"missing official {side} wrist mount")
        wrist = mount_audit.find_body(root, str(mount["parent_link"]))
        for child in list(wrist.findall("body")):
            if child.get("name") == root_name:
                wrist.remove(child)
                removed.append({"side": side, "root": root_name, "parent": str(mount["parent_link"])})
    return removed


def scene_xml(*, x2_overlay: Path, room: Path, mug: Path, poses: dict[str, object]) -> bytes:
    start = poses.get("robot_start")
    mug_pose = poses.get("mug_source")
    if not isinstance(start, dict) or not isinstance(mug_pose, dict):
        raise ValueError("scene config lacks robot_start or mug_source pose")
    start_xyz = vec3(start.get("xyz_m"), label="robot_start.xyz_m")
    mug_xyz = vec3(mug_pose.get("xyz_m"), label="mug_source.xyz_m")
    start_yaw = float(start.get("yaw_rad"))
    mug_yaw = float(mug_pose.get("yaw_rad"))
    if not math.isfinite(start_yaw) or not math.isfinite(mug_yaw):
        raise ValueError("scene pose yaw must be finite")

    root = ET.Element("mujoco", {"model": "RAICOM X2 STL autonomy static review"})
    ET.SubElement(root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
    ET.SubElement(root, "option", {"timestep": "0.001"})
    statistic = ET.SubElement(root, "statistic")
    statistic.set("center", "0 0 0.6")
    statistic.set("extent", "3.2")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.5 0.5 0.5", "ambient": "0.2 0.2 0.2", "specular": "0.9 0.9 0.9"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "200", "elevation": "-20"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "model", {"name": "x2_stl_runtime", "file": str(x2_overlay)})
    ET.SubElement(asset, "model", {"name": "raicom_room", "file": str(room)})
    ET.SubElement(asset, "model", {"name": "raicom_mug", "file": str(mug)})
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.3 0.52 0.63", "rgb2": "0.3 0.52 0.63", "markrgb": "1 1 1", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
    # MuJoCo attaches child model bodies but does not retain their contact
    # exclusions in the parent contact table.  Carry forward exactly the
    # scoped internal exclusions already present in the reviewed STL overlay.
    contact = ET.SubElement(root, "contact")
    for body1, body2 in PREFIXED_INHERITED_EXCLUSIONS:
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
    worldbody = ET.SubElement(root, "worldbody")
    x2_frame = ET.SubElement(worldbody, "frame", {"name": "x2_start_frame", "pos": " ".join(f"{value:.17g}" for value in start_xyz), "euler": yaw_euler(start_yaw)})
    ET.SubElement(x2_frame, "attach", {"model": "x2_stl_runtime", "prefix": "x2_"})
    room_frame = ET.SubElement(worldbody, "frame", {"name": "raicom_room_frame", "pos": "0 0 0", "quat": "1 0 0 0"})
    ET.SubElement(room_frame, "attach", {"model": "raicom_room", "prefix": "raicom_"})
    mug_frame = ET.SubElement(worldbody, "frame", {"name": "mug_frame", "pos": " ".join(f"{value:.17g}" for value in mug_xyz), "euler": yaw_euler(mug_yaw)})
    ET.SubElement(mug_frame, "attach", {"model": "raicom_mug", "prefix": "mug_"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def set_prefixed_hand_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    trajectory,
    command_percent: float,
    side: str,
) -> dict[str, float]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    pose = trajectory.pose(command_percent)
    mapping = pose.positions_rad if side == "right" else {
        LEFT_JOINT_MAPPING[name]: value for name, value in pose.positions_rad.items()
    }
    result: dict[str, float] = {}
    for joint, value in mapping.items():
        prefixed = "x2_" + joint
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
        if joint_id < 0:
            raise ValueError(f"assembled scene lacks STL joint {prefixed}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0
        result[prefixed] = value
    return result


def contacts_after_forward(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    """Return exact static contact identities; never treat a count as a collision verdict."""
    records: list[dict[str, object]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        records.append({
            "geom_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom#{geom1}",
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom#{geom2}",
            ],
            "body_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or f"body#{body1}",
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or f"body#{body2}",
            ],
            "distance_m": float(contact.dist),
        })
    return records


def stl_hand_geometry_inventory(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """Prove visual and collision meshes are paired on the same STL hand bodies."""
    inventory: dict[str, dict[str, int]] = {}
    for side in ("R", "L"):
        prefix = f"x2_{side}_"
        visual = collision = unexpected = 0
        bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
            in {prefix + "omnipicker_base_link", *(prefix + "hand_" + suffix for suffix in (
                "narrow1_Link", "narrow2_Link", "narrow3_Link", "narrow4_Link", "narrow_loop_Link",
                "wide1_Link", "wide2_Link", "wide3_Link", "wide4_Link", "wide_loop_Link",
            ))}
        }
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if not (body == prefix + "omnipicker_base_link" or body.startswith(prefix + "hand_")):
                continue
            group = int(model.geom_group[geom_id])
            contype = int(model.geom_contype[geom_id])
            conaffinity = int(model.geom_conaffinity[geom_id])
            if group == 0 and contype == 0 and conaffinity == 0:
                visual += 1
            elif group == mount_audit.STL_COLLISION_RENDER_GROUP and contype == 1 and conaffinity == 1:
                collision += 1
            else:
                unexpected += 1
        if len(bodies) != 11 or visual != 9 or collision != 8 or unexpected:
            raise RuntimeError(
                f"unexpected {side} STL hand geometry inventory: bodies={len(bodies)}, "
                f"visual={visual}, collision={collision}, unexpected={unexpected}"
            )
        inventory[side] = {
            "source_hand_bodies": len(bodies),
            "visual_geoms_group_0": visual,
            "collision_geoms_group_4": collision,
            "unexpected_hand_geoms": unexpected,
        }
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--left-command-percent", type=float, default=0.0)
    parser.add_argument("--right-command-percent", type=float, default=0.0)
    parser.add_argument("--live-viewer", action="store_true", help="open the generated static scene in the native MuJoCo GLFW viewer")
    parser.add_argument("--show-collision-proxies", action="store_true", help="render collision groups 1 and 3; collision physics is always retained")
    args = parser.parse_args()
    if not all(0.0 <= value <= 100.0 and math.isfinite(value) for value in (args.left_command_percent, args.right_command_percent)):
        raise SystemExit("each STL command must be finite and lie in 0..100 percent")
    config = load_scene_config(args.scene_config)
    sources = config["sources"]
    assert isinstance(sources, dict)
    raicom_model = require_source(REPO, sources["raicom_x2_model"], label="RAICOM X2 model")
    official_urdf = require_source(REPO, sources["official_x2_urdf"], label="official X2 URDF")
    room = require_source(REPO, sources["raicom_room"], label="RAICOM room")
    mug = require_source(REPO, sources["raicom_mug"], label="RAICOM mug")

    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    selected = config["selected_gripper"]
    assert isinstance(selected, dict)
    if loaded.model_sha256 != selected.get("model_sha256") or loaded.metadata.get("trajectory_sha256") != selected.get("trajectory_sha256"):
        raise SystemExit("registered STL model identity differs from the autonomy scene contract")
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-stl-raicom-autonomy-static").native_path
    x2_overlay = output / "raicom_x2_bilateral_stl_overlay.xml"
    overlay = add_stl_hands(raicom_model=raicom_model, official_urdf=official_urdf, stl_model=loaded.package_path / "model.xml", output=x2_overlay)
    scene_path = output / "x2_stl_raicom_autonomy_scene.xml"
    poses = config["poses"]
    assert isinstance(poses, dict)
    pg.write_new_bytes(scene_path, scene_xml(x2_overlay=x2_overlay, room=room, mug=mug, poses=poses))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    trajectory = load_registered_trajectory(loaded.package_path)
    left = set_prefixed_hand_pose(model, data, trajectory=trajectory, command_percent=args.left_command_percent, side="left")
    right = set_prefixed_hand_pose(model, data, trajectory=trajectory, command_percent=args.right_command_percent, side="right")
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.xpos).all():
        raise RuntimeError("assembled scene has non-finite static state")
    contacts = contacts_after_forward(model, data)
    geometry_inventory = stl_hand_geometry_inventory(model)
    result = {
        "schema": "robotsim.issue46.x2-stl-raicom-autonomy-static.v1",
        "classification": "SIMULATION_ONLY_STATIC_AUTONOMY_SCENE",
        "mj_step_calls": 0,
        "scene_config": {"path": str(args.scene_config.resolve()), "sha256": sha256(args.scene_config)},
        "selected_gripper": {"model_id": loaded.model_id, "model_sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "generated": {"x2_stl_overlay": overlay, "scene": {"path": str(scene_path), "sha256": sha256(scene_path)}},
        "inherited_internal_collision_exclusions": [list(pair) for pair in PREFIXED_INHERITED_EXCLUSIONS],
        "hand_geometry_inventory": geometry_inventory,
        "viewer": {
            "collision_proxies_visible": args.show_collision_proxies,
            "hidden_groups_by_default": [] if args.show_collision_proxies else list(COLLISION_RENDER_GROUPS),
            "note": "Render-group changes do not modify contype, conaffinity, or any physics collision pair.",
        },
        "model": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "neq": int(model.neq),
            "ncon_after_mj_forward": int(data.ncon), "contacts_after_mj_forward": contacts,
        },
        "hand_commands": {"left_percent": args.left_command_percent, "right_percent": args.right_command_percent, "left_qpos_rad": left, "right_qpos_rad": right},
        "gates": {
            "PINNED_RAICOM_AND_VENDOR_SOURCES": "PASS",
            "DEFAULT_STL_GRIPPER_IDENTITY": "PASS",
            "BILATERAL_STL_STATIC_MJ_FORWARD": "PASS",
            "NAVIGATION": "NOT_RUN",
            "BALANCED_ARM_APPROACH": "NOT_RUN",
            "CONTACT_DRIVEN_LIFT": "NOT_RUN"
        },
        "limitations": [
            "The RAICOM base body, room, and mug are source inputs; only the generated evidence-local hand overlay changes the selected gripper representation.",
            "The reviewed STL trajectory is a SIMULATION_ONLY table-driven kinematic mapping, not a manufacturer transmission claim.",
            "This static assembly must not be presented as autonomous navigation, perception, standing balance, or physical grasp evidence."
        ]
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "scene_wsl": str(scene_path), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"]}, ensure_ascii=False, indent=2))
    if args.live_viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            if not args.show_collision_proxies:
                for group in COLLISION_RENDER_GROUPS:
                    viewer.opt.geomgroup[group] = 0
            viewer.cam.lookat[:] = [0.3, -1.1, 0.85]
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 130.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
