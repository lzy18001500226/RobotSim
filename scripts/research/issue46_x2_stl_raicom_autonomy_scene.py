#!/usr/bin/env python3
"""Review the approved full X2-STL assembly inside the RAICOM competition room.

The human-approved full-X2 STL scene is the sole robot source. This tool never
imports the separate RAICOM runtime ``x2.xml``: that model has a different body
presentation and must not become a second robot definition. It retains the
approved X2 table-front pose and cylinder, removes only that source scene's
duplicate table, then attaches the preserved RAICOM room/table/chair model.
It calls ``mj_forward`` only.
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
STL_VISUAL_RENDER_GROUP = 2
STL_COLLISION_RENDER_GROUP = 3

sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    LEFT_JOINT_MAPPING,
    load_registered_trajectory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_source(repo_root: Path, record: dict[str, object], *, label: str) -> Path:
    raw_path, expected = record.get("path"), record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"{label} source record is incomplete")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} source path must be repository-relative")
    # Human-approved review scenes are governed under the canonical main
    # checkout's reviews/ root, while source/configuration inputs belong to
    # the current code checkout.  This is an explicit source boundary, not a
    # fallback to another model definition.
    source_root = EVIDENCE_REPO if relative.parts and relative.parts[0] == "reviews" else repo_root
    path = (source_root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} source is missing or unsafe: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} source hash mismatch: expected {expected}, got {actual}")
    return path


def load_scene_config(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"scene config is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "robotsim.raicom2026.x2-stl-autonomy-scene.v3":
        raise ValueError("scene config schema is not the explicit-review-input variant")
    if payload.get("classification") != "SIMULATION_ONLY_STATIC_AUTONOMY_SCENE_INPUT":
        raise ValueError("scene config qualification is unexpected")
    if not isinstance(payload.get("sources"), dict) or not isinstance(payload.get("environment"), dict):
        raise ValueError("scene config lacks source or environment records")
    return payload


def require_explicit_file(path: Path, expected_sha256: str, *, label: str) -> Path:
    """Validate a review-only CLI input without making it a runtime config dependency."""
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256.lower()):
        raise ValueError(f"{label} expected SHA-256 is malformed")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} hash mismatch: expected {expected_sha256}, got {actual}")
    return path.resolve()


def body_names(root: ET.Element) -> list[str]:
    return [body.get("name", "") for body in root.findall(".//body")]


def verify_collision_proxy_recovery(
    parts: Path,
    manifest: Path,
    manifest_hash: str,
    expected_count: int,
) -> dict[str, object]:
    if not parts.is_absolute() or not parts.is_dir() or parts.is_symlink():
        raise ValueError(f"collision proxy directory is missing or unsafe: {parts}")
    manifest = require_explicit_file(manifest, manifest_hash, label="collision proxy manifest")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        str(part["filename"]): str(part["sha256"])
        for decomposition in payload.get("mesh_decompositions", [])
        for part in decomposition.get("parts", [])
    }
    if len(expected) != expected_count:
        raise ValueError(f"collision manifest expected {expected_count} pieces, found {len(expected)}")
    actual = {path.name for path in parts.iterdir() if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".stl"}
    if actual != set(expected):
        raise ValueError(f"collision recovery filenames differ: missing={sorted(set(expected) - actual)[:3]}, extra={sorted(actual - set(expected))[:3]}")
    for filename, expected_hash in expected.items():
        actual_hash = sha256(parts / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"collision proxy hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}")
    return {"parts_directory": parts.resolve(), "manifest": manifest, "manifest_sha256": manifest_hash, "expected_hashes": expected}


def rebind_collision_proxy_meshes(root: ET.Element, recovery: dict[str, object]) -> list[dict[str, str]]:
    """Replace deleted-worktree proxy paths only in the generated scene copy."""
    parts = recovery["parts_directory"]
    expected = recovery["expected_hashes"]
    if not isinstance(parts, Path) or not isinstance(expected, dict):
        raise ValueError("collision recovery input is malformed")
    rewrites: list[dict[str, str]] = []
    referenced: set[str] = set()
    for mesh in root.findall("./asset/mesh"):
        raw = mesh.get("file")
        if not raw:
            continue
        filename = Path(raw).name
        if filename not in expected:
            continue
        if "collision_generated" not in raw:
            raise ValueError(f"proxy-named mesh does not identify the frozen collision source: {raw}")
        target = parts / filename
        mesh.set("file", str(target))
        referenced.add(filename)
        rewrites.append({"mesh": str(mesh.get("name", "")), "from": raw, "to": str(target), "sha256": str(expected[filename])})
    if not referenced:
        raise ValueError("approved scene does not reference any verified collision proxy")
    return rewrites


def rebind_room_assets(room: Path, output: Path) -> dict[str, object]:
    """Write an evidence-local room XML with only relative assets made absolute."""
    tree = ET.parse(room)
    root = tree.getroot()
    rewrites: list[dict[str, str]] = []
    for element in root.findall(".//asset/*"):
        raw = element.get("file")
        if not raw or Path(raw).is_absolute():
            continue
        source = (room.parent / raw).resolve()
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"RAICOM room asset is missing or unsafe: {source}")
        element.set("file", str(source))
        rewrites.append({"tag": element.tag, "name": str(element.get("name", "")), "from": raw, "to": str(source), "sha256": sha256(source)})
    if not rewrites:
        raise ValueError("RAICOM room has no relative assets to rebind")
    ET.indent(tree, space="  ")
    pg.write_new_bytes(output, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {"path": str(output), "sha256": sha256(output), "asset_rebindings": rewrites}


def compose_scene(
    approved_scene: Path,
    room: Path,
    environment: dict[str, object],
    recovery: dict[str, object] | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Keep the approved robot scene, replacing only its duplicate table."""
    tree = ET.parse(approved_scene)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None or asset is None:
        raise ValueError("approved full-X2 scene lacks worldbody or asset")
    proxy_rebindings = [] if recovery is None else rebind_collision_proxy_meshes(root, recovery)
    names = body_names(root)
    required = {"pelvis", "R_omnipicker_base_link", "L_omnipicker_base_link"}
    if not required.issubset(names):
        raise ValueError("approved scene does not contain the reviewed complete X2-STL robot")
    retain_body = environment.get("retain_body")
    if not isinstance(retain_body, str) or retain_body not in names:
        raise ValueError("approved scene does not retain the declared grasp object")
    remove_names = environment.get("remove_from_approved_scene")
    if not isinstance(remove_names, list) or not all(isinstance(name, str) for name in remove_names):
        raise ValueError("environment removal list is invalid")
    removed: list[str] = []
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") in remove_names:
            removed.append(str(child.get("name")))
            worldbody.remove(child)
    if set(removed) != set(remove_names):
        raise ValueError(f"approved scene environment mismatch: expected {remove_names}, removed {removed}")
    model_name = environment.get("attach_model_name")
    prefix = environment.get("attach_prefix")
    table_after_attach = environment.get("table_body_after_attach")
    if not all(isinstance(value, str) for value in (model_name, prefix, table_after_attach)):
        raise ValueError("environment attach contract is invalid")
    if any(item.get("name") == model_name for item in asset.findall("model")):
        raise ValueError(f"approved scene already has model asset {model_name}")
    ET.SubElement(asset, "model", {"name": model_name, "file": str(room)})
    frame = ET.SubElement(worldbody, "frame", {"name": "raicom_competition_room", "pos": "0 0 0", "quat": "1 0 0 0"})
    ET.SubElement(frame, "attach", {"model": model_name, "prefix": prefix})
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), {
        "approved_scene_environment_bodies_removed": removed,
        "approved_robot_body": "pelvis",
        "approved_hands": ["L_omnipicker_base_link", "R_omnipicker_base_link"],
        "retained_grasp_object": retain_body,
        "attached_competition_room": {"model": model_name, "prefix": prefix, "table_body": table_after_attach},
        "collision_proxy_rebindings": proxy_rebindings,
    }


def apply_hand_pose(model: mujoco.MjModel, data: mujoco.MjData, trajectory, command_percent: float, side: str) -> dict[str, float]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    pose = trajectory.pose(command_percent)
    positions = pose.positions_rad if side == "right" else {
        LEFT_JOINT_MAPPING[name]: value for name, value in pose.positions_rad.items()
    }
    for joint, value in positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            raise ValueError(f"approved full-X2 scene lacks {side} STL joint {joint}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0
    return positions


def stl_hand_inventory(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """Assert exactly one reviewed visual/collision assembly per hand side."""
    result: dict[str, dict[str, int]] = {}
    for side in ("R", "L"):
        prefix = side + "_"
        expected_bodies = {
            prefix + "omnipicker_base_link",
            *(prefix + "hand_" + suffix for suffix in (
                "narrow1_Link", "narrow2_Link", "narrow3_Link", "narrow4_Link", "narrow_loop_Link",
                "wide1_Link", "wide2_Link", "wide3_Link", "wide4_Link", "wide_loop_Link",
            )),
        }
        observed_bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "") in expected_bodies
        }
        visual = collision = unexpected = 0
        for geom_id in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
            if body not in expected_bodies:
                continue
            group = int(model.geom_group[geom_id])
            contype, conaffinity = int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])
            if group == STL_VISUAL_RENDER_GROUP and contype == 0 and conaffinity == 0:
                visual += 1
            elif group == STL_COLLISION_RENDER_GROUP and contype == 1 and conaffinity == 1:
                collision += 1
            else:
                unexpected += 1
        if observed_bodies != expected_bodies or visual != 9 or collision != 8 or unexpected:
            raise RuntimeError(
                f"{side} hand is not the approved single STL assembly: bodies={sorted(observed_bodies)}, "
                f"visual={visual}, collision={collision}, unexpected={unexpected}"
            )
        result[side] = {
            "source_hand_bodies": len(observed_bodies),
            "visual_geoms_group_2": visual,
            "collision_geoms_group_3": collision,
            "unexpected_hand_geoms": unexpected,
        }
    return result


def contacts_after_forward(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--approved-full-x2-scene", required=True, type=Path, help="absolute human-approved full X2-STL review scene")
    parser.add_argument("--approved-full-x2-scene-sha256", required=True, help="SHA-256 paired with --approved-full-x2-scene")
    parser.add_argument("--collision-proxy-parts", required=True, type=Path, help="absolute governed directory containing the reviewed CoACD proxy pieces")
    parser.add_argument("--collision-proxy-manifest", required=True, type=Path, help="absolute frozen collision proxy manifest")
    parser.add_argument("--collision-proxy-manifest-sha256", required=True, help="SHA-256 paired with --collision-proxy-manifest")
    parser.add_argument("--left-command-percent", type=float, default=0.0)
    parser.add_argument("--right-command-percent", type=float, default=0.0)
    parser.add_argument("--live-viewer", action="store_true", help="open the generated static scene in native MuJoCo GLFW")
    parser.add_argument("--show-collision-proxies", action="store_true", help="render group 3 collision proxies; physics is unchanged")
    args = parser.parse_args()
    if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in (args.left_command_percent, args.right_command_percent)):
        raise SystemExit("STL commands must lie in 0..100 percent")
    config = load_scene_config(args.scene_config)
    sources = config["sources"]
    assert isinstance(sources, dict)
    approved_scene = require_explicit_file(args.approved_full_x2_scene, args.approved_full_x2_scene_sha256, label="approved full X2-STL scene")
    room = require_source(REPO, sources["raicom_room"], label="RAICOM competition room")
    recovery = verify_collision_proxy_recovery(
        args.collision_proxy_parts,
        args.collision_proxy_manifest,
        args.collision_proxy_manifest_sha256,
        161,
    )
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    selected = config["selected_gripper"]
    assert isinstance(selected, dict)
    if loaded.model_sha256 != selected.get("model_sha256") or loaded.metadata.get("trajectory_sha256") != selected.get("trajectory_sha256"):
        raise SystemExit("registered STL model identity differs from the scene contract")
    environment = config["environment"]
    assert isinstance(environment, dict)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-stl-raicom-autonomy-static").native_path
    room_rebound_path = output / "raicom_room_absolute_assets.xml"
    room_rebound = rebind_room_assets(room, room_rebound_path)
    scene_path = output / "x2_stl_raicom_autonomy_scene.xml"
    payload, composition = compose_scene(approved_scene, room_rebound_path, environment, recovery)
    pg.write_new_bytes(scene_path, payload)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    trajectory = load_registered_trajectory(loaded.package_path)
    left = apply_hand_pose(model, data, trajectory, args.left_command_percent, "left")
    right = apply_hand_pose(model, data, trajectory, args.right_command_percent, "right")
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.xpos).all():
        raise RuntimeError("combined approved-X2/RAICOM scene has non-finite state")
    result = {
        "schema": "robotsim.issue46.x2-stl-raicom-autonomy-static.v2",
        "classification": "SIMULATION_ONLY_STATIC_APPROVED_FULL_X2_IN_RAICOM_ROOM",
        "mj_step_calls": 0,
        "scene_config": {"path": str(args.scene_config.resolve()), "sha256": sha256(args.scene_config)},
        "sources": {
            "approved_full_x2_stl_scene": {"path": str(approved_scene), "sha256": sha256(approved_scene)},
            "raicom_room": {"path": str(room), "sha256": sha256(room)},
            "raicom_room_rebound": room_rebound,
            "collision_proxy_recovery": {"parts_directory": str(recovery["parts_directory"]), "manifest": str(recovery["manifest"]), "manifest_sha256": str(recovery["manifest_sha256"]), "piece_count": len(recovery["expected_hashes"])},
        },
        "selected_gripper": {"model_id": loaded.model_id, "model_sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "generated_scene": {"path": str(scene_path), "sha256": sha256(scene_path)},
        "composition": composition,
        "hand_geometry_inventory": stl_hand_inventory(model),
        "viewer": {"collision_proxies_visible": args.show_collision_proxies, "hidden_groups_by_default": [] if args.show_collision_proxies else [STL_COLLISION_RENDER_GROUP]},
        "model": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "neq": int(model.neq),
            "ncon_after_mj_forward": int(data.ncon), "contacts_after_mj_forward": contacts_after_forward(model, data),
        },
        "hand_commands": {"left_percent": args.left_command_percent, "right_percent": args.right_command_percent, "left_qpos_rad": left, "right_qpos_rad": right},
        "gates": {
            "APPROVED_FULL_X2_SOURCE": "PASS", "RAICOM_ROOM_SOURCE": "PASS", "SINGLE_BILATERAL_STL_HAND_ASSEMBLY": "PASS", "STATIC_MJ_FORWARD": "PASS",
            "NAVIGATION": "NOT_RUN", "BALANCED_ARM_APPROACH": "NOT_RUN", "CONTACT_DRIVEN_LIFT": "NOT_RUN",
        },
        "limitations": [
            "The robot is the approved static full-X2 STL scene, not a native MC runtime body.",
            "The room/table/chair are static source inputs; no navigation, balance, perception, or grasp physics ran.",
            "The reviewed STL trajectory remains SIMULATION_ONLY kinematics, not a manufacturer transmission claim.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "scene_wsl": str(scene_path), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"]}, ensure_ascii=False, indent=2))
    if args.live_viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            if not args.show_collision_proxies:
                viewer.opt.geomgroup[STL_COLLISION_RENDER_GROUP] = 0
            viewer.cam.lookat[:] = [1.45, -1.15, 0.88]
            viewer.cam.distance = 2.1
            viewer.cam.azimuth = 155.0
            viewer.cam.elevation = -14.0
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
