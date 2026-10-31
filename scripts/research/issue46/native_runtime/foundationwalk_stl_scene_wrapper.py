#!/usr/bin/env python3
"""Build a static FoundationWalk-room wrapper using the frozen bilateral STL hands.

The source FoundationWalk room wrapper contains an attached native X2 model.
MuJoCo cannot add children to that attached submodel from the outer document,
so this tool copies the already-audited, wrist-mounted STL X2 *overlay* as a
separate immutable submodel and points an otherwise unchanged room wrapper at
that copy.  It performs only ``mj_forward``; no controller, hand command, or
physics step is run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


REQUIRED_HAND_BODIES = (
    "R_omnipicker_base_link",
    "L_omnipicker_base_link",
    "R_hand_narrow3_Link",
    "L_hand_narrow3_Link",
    "R_hand_wide3_Link",
    "L_hand_wide3_Link",
)
REQUIRED_HAND_JOINTS = ("right_claw_joint", "left_claw_joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")


def model_asset(root: ET.Element, name: str) -> ET.Element:
    node = root.find(f"./asset/model[@name='{name}']")
    if node is None:
        raise ValueError(f"source wrapper has no model asset named {name}")
    return node


def rebind_wrapper(
    source: Path,
    *,
    x2_path: str,
    room_path: str,
    mug_path: str,
) -> ET.ElementTree:
    tree = ET.parse(source)
    root = tree.getroot()
    model_asset(root, "x2_mc_runtime").set("file", x2_path)
    model_asset(root, "raicom_room").set("file", room_path)
    model_asset(root, "raicom_mug").set("file", mug_path)
    return tree


def write_host_mug(source: Path, destination: Path, room_model: Path) -> Path:
    """Copy the pinned mug XML and rebase only its host mesh directories.

    The preserved runtime mug XML intentionally points at the container mount
    ``/workspace/raicom_scene_source``.  That path is invalid for the local
    static ``mj_forward`` check, so the generated host-only copy is rebased to
    the same immutable RAICOM source tree as the pinned room model.
    """
    scene_root = room_model.parents[5]
    mug_assets = scene_root / "robot/lx2501_3_t2d5/model_info/object/model/mug"
    if not (mug_assets / "mug.obj").is_file() or not (mug_assets / "mug.png").is_file():
        raise ValueError(f"pinned RAICOM mug assets are incomplete: {mug_assets}")
    tree = ET.parse(source)
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        raise ValueError("pinned mug model has no compiler element")
    compiler.set("meshdir", str(mug_assets))
    compiler.set("texturedir", str(mug_assets))
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True))
    return mug_assets


def runtime_simulator_config_text(source: Path) -> str:
    """Return a copied native simulator config pointed at the generated scene."""
    text = source.read_text(encoding="utf-8")
    target = "  simulation_model: /workspace/evidence/scene_foundationwalk_stl_runtime.xml"
    rewritten, substitutions = re.subn(
        r"(?m)^  simulation_model: .*?$",
        target,
        text,
        count=1,
    )
    if substitutions != 1:
        raise ValueError("native simulator config has no unique simulation_model entry")
    return rewritten


def write_runtime_simulator_config(source: Path, destination: Path) -> None:
    """Publish the generated runtime simulator config under governed evidence."""
    rewritten = runtime_simulator_config_text(source)
    pg.write_new_text(destination, rewritten)


def compiled_summary(scene: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    missing_bodies = [
        name
        for name in REQUIRED_HAND_BODIES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
    ]
    missing_joints = [
        name
        for name in REQUIRED_HAND_JOINTS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
    ]
    if missing_bodies or missing_joints:
        raise ValueError(
            "compiled FoundationWalk STL wrapper is missing "
            f"bodies={missing_bodies}, joints={missing_joints}"
        )
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("compiled FoundationWalk STL wrapper has non-finite state")
    contacts: list[list[str]] = []
    for contact in data.contact:
        first = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1])
        ) or "<world>"
        second = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2])
        ) or "<world>"
        contacts.append([first, second])
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "neq": int(model.neq),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "initial_contacts": int(data.ncon),
        "initial_contact_pairs": contacts,
        "required_hand_bodies": list(REQUIRED_HAND_BODIES),
        "required_hand_joints": list(REQUIRED_HAND_JOINTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wrapper", required=True, type=Path)
    parser.add_argument("--source-wrapper-sha256", required=True)
    parser.add_argument("--x2-stl-overlay", required=True, type=Path)
    parser.add_argument("--x2-stl-overlay-sha256", required=True)
    parser.add_argument("--room-model", required=True, type=Path)
    parser.add_argument("--room-model-sha256", required=True)
    parser.add_argument("--mug-model", required=True, type=Path)
    parser.add_argument("--mug-model-sha256", required=True)
    parser.add_argument("--simulator-config", required=True, type=Path)
    parser.add_argument("--simulator-config-sha256", required=True)
    args = parser.parse_args()

    check_source(args.source_wrapper, args.source_wrapper_sha256, "source FoundationWalk wrapper")
    check_source(args.x2_stl_overlay, args.x2_stl_overlay_sha256, "audited X2 STL overlay")
    check_source(args.room_model, args.room_model_sha256, "room source model")
    check_source(args.mug_model, args.mug_model_sha256, "mug source model")
    check_source(args.simulator_config, args.simulator_config_sha256, "native simulator config")

    output = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46",
        prefix="foundationwalk-stl-scene-wrapper",
    ).native_path
    x2_host = output / "x2_stl_native_model.xml"
    mug_host = output / "mug_portable_runtime.xml"
    pg.write_new_bytes(x2_host, args.x2_stl_overlay.read_bytes())
    mug_asset_root = write_host_mug(args.mug_model, mug_host, args.room_model)

    host_tree = rebind_wrapper(
        args.source_wrapper,
        x2_path=str(x2_host),
        room_path=str(args.room_model),
        mug_path=str(mug_host),
    )
    ET.indent(host_tree, space="  ")
    host_scene = output / "scene_foundationwalk_stl_host.xml"
    pg.write_new_bytes(host_scene, ET.tostring(host_tree.getroot(), encoding="utf-8", xml_declaration=True))
    summary = compiled_summary(host_scene)

    runtime_tree = rebind_wrapper(
        args.source_wrapper,
        x2_path="/workspace/evidence/x2_stl_native_model.xml",
        room_path="/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/scene/room/room_with_chair_review.xml",
        mug_path="/workspace/evidence/mug_portable_runtime.xml",
    )
    ET.indent(runtime_tree, space="  ")
    runtime_scene = output / "scene_foundationwalk_stl_runtime.xml"
    pg.write_new_bytes(runtime_scene, ET.tostring(runtime_tree.getroot(), encoding="utf-8", xml_declaration=True))
    runtime_simulator = output / "simulator_foundationwalk_stl.yaml"
    write_runtime_simulator_config(args.simulator_config, runtime_simulator)

    result = {
        "schema": "robotsim.issue46.foundationwalk-stl-scene-wrapper.v1",
        "classification": "SIMULATION_ONLY_FOUNDATIONWALK_STL_STATIC_WRAPPER",
        "mj_step_calls": 0,
        "sources": {
            "foundationwalk_wrapper": {"path": str(args.source_wrapper), "sha256": sha256(args.source_wrapper)},
            "x2_stl_overlay": {"path": str(args.x2_stl_overlay), "sha256": sha256(args.x2_stl_overlay)},
            "room_model": {"path": str(args.room_model), "sha256": sha256(args.room_model)},
            "mug_model": {"path": str(args.mug_model), "sha256": sha256(args.mug_model)},
            "simulator_config": {"path": str(args.simulator_config), "sha256": sha256(args.simulator_config)},
        },
        "host_only_asset_rebindings": {
            "mug_mesh_and_texture_directory": str(mug_asset_root),
        },
        "generated": {
            "x2_host": {"path": str(x2_host), "sha256": sha256(x2_host)},
            "mug_host": {"path": str(mug_host), "sha256": sha256(mug_host)},
            "host_scene": {"path": str(host_scene), "sha256": sha256(host_scene)},
            "runtime_scene": {"path": str(runtime_scene), "sha256": sha256(runtime_scene)},
            "runtime_simulator_config": {"path": str(runtime_simulator), "sha256": sha256(runtime_simulator)},
        },
        "compiled_model": summary,
        "gates": {
            "PINNED_SOURCE_HASHES": "PASS",
            "FROZEN_STL_OVERLAY_REUSED": "PASS",
            "FULL_FOUNDATIONWALK_ROOM_STATIC_MJ_FORWARD": "PASS",
            "NATIVE_SIMULATOR_CONFIG_REPOINTED": "PASS",
            "NATIVE_MC_STARTUP_WITH_STL": "NOT_RUN",
            "FOUNDATIONWALK_WITH_STL": "NOT_RUN",
            "BALANCED_ARM_GRASP": "NOT_RUN",
        },
        "limitations": [
            "The runtime wrapper is a generated SIMULATION_ONLY overlay; it does not modify the RAICOM runtime, vendor X2 model, room, mug, or frozen STL package.",
            "This check establishes only static compilation and finite forward kinematics. It does not establish native MC compatibility, walking, balance, arm control, collision safety, or grasping.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": result["gates"], "compiled_model": summary, "artifact_manifest": manifest["manifest"]["WSL_PATH"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
