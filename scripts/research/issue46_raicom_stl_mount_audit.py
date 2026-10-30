#!/usr/bin/env python3
"""Build a source-grounded RAICOM X2 + bilateral STL-hand static mount audit.

The fixed wrist-to-OmniPicker origins come from the pinned v1.4 vendor URDF.
This creates a separate XML overlay, applies only the reviewed STL table poses,
and uses ``mj_forward``.  It is not a dynamics, balance, or grasp result.
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


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
MESH_NAMES = (
    "mesh_base_link",
    "mesh_narrow1_Link",
    "mesh_narrow2_Link",
    "mesh_narrow3_Link",
    "mesh_narrow_loop_Link",
    "mesh_wide1_Link",
    "mesh_wide2_Link",
    "mesh_wide3_Link",
    "mesh_wide_loop_Link",
)
RIGHT_EXCLUSIONS = (
    ("R_hand_narrow1_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow2_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_wide1_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide2_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide3_Link", "R_hand_wide_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_wide3_Link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"body is missing: {name}")


def right_to_left(value: str) -> str:
    if value == "right_claw_joint":
        return "left_claw_joint"
    if value.startswith("R_"):
        return "L_" + value[2:]
    return value


def official_mounts(urdf: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(urdf).getroot()
    mounts: dict[str, dict[str, str]] = {}
    for name, side in (("R_omnipicker_joint", "right"), ("L_omnipicker_joint", "left")):
        joint = next((node for node in root.findall("joint") if node.get("name") == name), None)
        if joint is None or joint.get("type") != "fixed":
            raise RuntimeError(f"vendor URDF lacks fixed mount: {name}")
        origin = joint.find("origin")
        parent = joint.find("parent")
        child = joint.find("child")
        if origin is None or parent is None or child is None:
            raise RuntimeError(f"vendor mount is incomplete: {name}")
        expected_parent = f"{side}_wrist_roll_link"
        if parent.get("link") != expected_parent:
            raise RuntimeError(f"vendor mount parent differs from expected wrist: {name}")
        mounts[side] = {"joint": name, "parent_link": parent.get("link"), "child_link": child.get("link"), "xyz": origin.get("xyz", ""), "rpy": origin.get("rpy", "")}
    return mounts


def remap_mesh_paths(root: ET.Element, source: Path) -> list[dict[str, str]]:
    """Resolve RAICOM's relative meshdir files in the generated overlay only."""
    compiler = root.find("compiler")
    meshdir = source.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    changes = []
    for mesh in root.findall("./asset/mesh"):
        raw = mesh.get("file")
        if raw is None or Path(raw).is_absolute():
            continue
        resolved = (meshdir / raw).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"RAICOM source mesh is missing: {resolved}")
        mesh.set("file", str(resolved))
        changes.append({"mesh": mesh.get("name", ""), "from": raw, "to": str(resolved)})
    return changes


def stl_assets(stl_root: ET.Element, stl_model: Path, side: str) -> list[ET.Element]:
    assets = {node.get("name"): node for node in stl_root.findall("./asset/mesh")}
    compiler = stl_root.find("compiler")
    meshdir = stl_model.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    prefix = "stl_R_" if side == "right" else "stl_L_"
    result = []
    for name in MESH_NAMES:
        source = assets.get(name)
        if source is None:
            raise RuntimeError(f"STL model asset is missing: {name}")
        clone = copy.deepcopy(source)
        clone.set("name", prefix + name)
        raw_path = clone.get("file")
        if raw_path is None:
            raise RuntimeError(f"STL source mesh path is missing: {name}")
        resolved = (meshdir / raw_path).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"STL source mesh is missing: {resolved}")
        clone.set("file", str(resolved))
        result.append(clone)
    return result


def remap_hand(element: ET.Element, side: str) -> None:
    prefix = "stl_R_" if side == "right" else "stl_L_"
    for node in element.iter():
        for key, value in list(node.attrib.items()):
            if side == "left":
                value = right_to_left(value)
            if key == "mesh" and value in MESH_NAMES:
                value = prefix + value
            node.set(key, value)


def add_hand(root: ET.Element, stl_root: ET.Element, *, side: str, mount: dict[str, str]) -> None:
    wrist = find_body(root, mount["parent_link"])
    source = find_body(stl_root, "R_omnipicker_base_link")
    hand = copy.deepcopy(source)
    remap_hand(hand, side)
    hand.set("pos", mount["xyz"])
    hand.set("euler", mount["rpy"])
    hand.attrib.pop("quat", None)
    wrist.append(hand)


def add_internal_exclusions(root: ET.Element) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side in ("R", "L"):
        for first, second in RIGHT_EXCLUSIONS:
            if side == "L":
                first, second = right_to_left(first), right_to_left(second)
            ET.SubElement(contact, "exclude", body1=first, body2=second)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raicom-model", required=True, type=Path)
    parser.add_argument("--official-urdf", required=True, type=Path)
    args = parser.parse_args()
    if not args.raicom_model.is_file() or not args.official_urdf.is_file():
        raise SystemExit("RAICOM model or pinned official URDF is missing")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    stl_model = loaded.package_path / "model.xml"
    if not stl_model.is_file():
        raise SystemExit("registered STL model source is missing")
    mounts = official_mounts(args.official_urdf)
    tree = ET.parse(args.raicom_model)
    root = tree.getroot()
    paths = remap_mesh_paths(root, args.raicom_model)
    stl_root = ET.parse(stl_model).getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("RAICOM model lacks an asset section")
    for side in ("right", "left"):
        for mesh in stl_assets(stl_root, stl_model, side):
            asset.append(mesh)
        add_hand(root, stl_root, side=side, mount=mounts[side])
    add_internal_exclusions(root)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-mount-audit").native_path
    overlay = output / "raicom_x2_bilateral_stl_mount_overlay.xml"
    ET.indent(tree, space="  ")
    pg.write_new_bytes(overlay, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(overlay))
    trajectory = load_registered_trajectory(loaded.package_path)
    checks = []
    for command in (0, 50, 100):
        data = mujoco.MjData(model)
        left = apply_stl_kinematic_pose(mujoco, model, data, trajectory, command, side="left", forward=True)
        right = apply_stl_kinematic_pose(mujoco, model, data, trajectory, command, side="right", forward=True)
        mujoco.mj_forward(model, data)
        checks.append({"command_percent": command, "left_joint_count": len(left.positions_rad), "right_joint_count": len(right.positions_rad), "finite_qpos": bool(np.isfinite(data.qpos).all()), "finite_xpos": bool(np.isfinite(data.xpos).all())})
    if not all(item["finite_qpos"] and item["finite_xpos"] for item in checks):
        raise RuntimeError("static bilateral STL pose produced a non-finite model state")
    result = {
        "schema": "robotsim.issue46.raicom-stl-mount-audit.v1",
        "classification": "SIMULATION_ONLY_OFFICIAL_WRIST_MOUNT_WITH_REVIEWED_STL_TRAJECTORY_OVERLAY",
        "mj_step_calls": 0,
        "raicom_model": {"path": str(args.raicom_model.resolve()), "sha256": sha256(args.raicom_model)},
        "official_urdf": {"path": str(args.official_urdf.resolve()), "sha256": sha256(args.official_urdf)},
        "stl_model": {"id": loaded.model_id, "model_sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"], "path": str(stl_model)},
        "official_mounts": mounts,
        "raicom_asset_path_rebindings": paths,
        "generated_overlay": {"path": str(overlay), "sha256": sha256(overlay)},
        "static_checks": checks,
        "gates": {"OFFICIAL_WRIST_PARENT": "PASS", "OFFICIAL_FIXED_MOUNT_ORIGINS": "PASS", "BILATERAL_STL_STATIC_MJ_FORWARD": "PASS", "DYNAMICS": "NOT_RUN", "BALANCE": "NOT_RUN", "EXTERNAL_CONTACT": "NOT_RUN"},
        "limitations": ["The RAICOM model's wrist inertial/mesh representation differs from the v1.4 source wrist extension visual used in prior fixed-base reviews.", "Only the fixed wrist-to-base mount origins are vendor-sourced; the STL trajectory remains the human-reviewed SIMULATION_ONLY kinematic mapping.", "This overlay has not received a new RAICOM full-body visual review.", "No dynamics, collision qualification, navigation, or grasping ran."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "mounts": mounts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
