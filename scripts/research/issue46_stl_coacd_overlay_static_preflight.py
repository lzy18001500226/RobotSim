#!/usr/bin/env python3
"""Build and validate the approved source-STL CoACD collision overlay.

This creates a separate review-scene XML.  It preserves the frozen
``X2-OmniPicker-STL`` visual meshes and kinematic table byte-for-byte, while
replacing only the eight right-hand raw collision geoms present in the full-X2
scene with the corresponding hash-locked CoACD source-STL pieces.

The tool runs ``mj_forward`` only.  It never calls ``mj_step`` and never edits
the frozen model, vendor assets, source trajectory, or base scene.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
PACKAGE = REPO / "simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1"
PROXY_MANIFEST = PACKAGE / "collision_proxy_manifest.json"
PROXY_DIRECTORY = PACKAGE / "collision_generated"
SOURCE_ASSETS = PACKAGE / "source_assets.json"
EXPECTED_PROXY_MANIFEST_SHA256 = "7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a"
EXPECTED_STL_MODEL_SHA256 = "088c1f44963773b6a3cebe0ca3e428b409fa16077bdb930e7b142c5c75981857"
EXPECTED_TRAJECTORY_SHA256 = "117bc4e81edbef8b341341c1cb81baba5a7bde61b590eaea93553ec77c4b540a"
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
OVERLAY_STEMS = {
    "narrow1_Link",
    "narrow2_Link",
    "narrow3_Link",
    "narrow_loop_Link",
    "wide1_Link",
    "wide2_Link",
    "wide3_Link",
    "wide_loop_Link",
}

sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    load_registered_trajectory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preflight_module():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("stl_static_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load static-preflight helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_geoms(element: ET.Element, body_name: str = ""):
    if element.tag == "body":
        body_name = element.get("name", body_name)
    for child in list(element):
        if child.tag == "geom":
            yield body_name, child
        yield from iter_geoms(child, body_name)


def visual_fingerprint(root: ET.Element) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    records = []
    for body, geom in iter_geoms(root):
        if geom.get("contype") != "0" or geom.get("conaffinity") != "0":
            continue
        records.append((body, tuple(sorted(geom.attrib.items()))))
    return sorted(records)


def contact_exclude_fingerprint(root: ET.Element) -> list[tuple[tuple[str, str], ...]]:
    contact = root.find("contact")
    if contact is None:
        return []
    return sorted(tuple(sorted(item.attrib.items())) for item in contact)


def load_locked_proxy_parts() -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    if sha256(PROXY_MANIFEST) != EXPECTED_PROXY_MANIFEST_SHA256:
        raise RuntimeError("collision proxy manifest SHA-256 differs from approved overlay proposal")
    manifest = json.loads(PROXY_MANIFEST.read_text(encoding="utf-8"))
    source_assets = json.loads(SOURCE_ASSETS.read_text(encoding="utf-8"))
    source_root = REPO / source_assets["repository_path"]
    if not source_root.is_dir():
        raise RuntimeError("pinned vendor source checkout is missing")
    expected: dict[str, str] = {}
    by_stem: dict[str, list[dict[str, object]]] = {}
    for mesh in manifest["mesh_decompositions"]:
        source_mesh = source_root / mesh["source_mesh"]
        if not source_mesh.is_file() or sha256(source_mesh) != mesh["source_sha256"]:
            raise RuntimeError(f"source mesh hash mismatch: {mesh['source_mesh']}")
        stem = Path(mesh["source_mesh"]).stem
        by_stem[stem] = list(mesh["parts"])
        for part in mesh["parts"]:
            filename = str(part["filename"])
            if filename in expected:
                raise RuntimeError(f"duplicate proxy filename in manifest: {filename}")
            expected[filename] = str(part["sha256"])
    if len(expected) != 161:
        raise RuntimeError(f"proxy manifest has {len(expected)} pieces, expected 161")
    actual = {
        path.name: sha256(path)
        for path in PROXY_DIRECTORY.glob("*.stl")
        if path.is_file() and not path.is_symlink()
    }
    if actual.keys() != expected.keys():
        raise RuntimeError("provisioned proxy file set differs from the locked manifest")
    mismatched = sorted(name for name, digest in expected.items() if actual[name] != digest)
    if mismatched:
        raise RuntimeError("provisioned proxy hash mismatch: " + ", ".join(mismatched[:5]))
    return by_stem, {
        "manifest_sha256": EXPECTED_PROXY_MANIFEST_SHA256,
        "manifest_piece_count": len(expected),
        "provisioned_piece_count": len(actual),
        "source_commit": source_assets["commit"],
        "source_urdf_sha256": source_assets["required_assets"]["urdf"]["sha256"],
    }


def build_overlay(raw_scene: Path, destination: Path) -> dict[str, object]:
    by_stem, proxy_lock = load_locked_proxy_parts()
    tree = ET.parse(raw_scene)
    root = tree.getroot()
    visual_before = visual_fingerprint(root)
    excludes_before = contact_exclude_fingerprint(root)
    assets = root.find("asset")
    if assets is None:
        raise RuntimeError("base scene has no asset block")
    raw_collision_stems = {
        Path(geom.get("mesh", "")).stem
        for parent in root.iter()
        for geom in list(parent)
        if geom.get("name", "").startswith("R_hand_")
        and geom.get("name", "").endswith("__collision_0")
    }
    if raw_collision_stems != OVERLAY_STEMS:
        raise RuntimeError(
            f"raw right-hand collision mesh set differs: {sorted(raw_collision_stems)}"
        )
    existing_asset_names = {item.get("name") for item in assets.findall("mesh")}
    for stem in sorted(raw_collision_stems):
        parts = by_stem[stem]
        for part in parts:
            name = f"stl_coacd_{part['filename']}"
            if name in existing_asset_names:
                raise RuntimeError(f"collision overlay asset name already exists: {name}")
            ET.SubElement(
                assets,
                "mesh",
                name=name,
                file=str(PROXY_DIRECTORY / str(part["filename"])),
            )
            existing_asset_names.add(name)

    replacements: dict[str, int] = {}
    for parent in root.iter():
        for geom in list(parent):
            name = geom.get("name", "")
            if not (name.startswith("R_hand_") and name.endswith("__collision_0")):
                continue
            source_stem = Path(geom.get("mesh", "")).stem
            parts = by_stem.get(source_stem)
            if parts is None:
                raise RuntimeError(f"missing proxy mapping for collision mesh: {source_stem}")
            index = list(parent).index(geom)
            parent.remove(geom)
            for part in parts:
                replacement = copy.deepcopy(geom)
                replacement.set("name", f"{name}__{part['filename']}")
                replacement.set("mesh", f"stl_coacd_{part['filename']}")
                parent.insert(index, replacement)
                index += 1
            replacements[source_stem] = len(parts)
    if set(replacements) != OVERLAY_STEMS:
        raise RuntimeError(
            f"right-hand collision replacement set differs: {sorted(replacements)}"
        )
    if sum(replacements.values()) != 137:
        raise RuntimeError("overlay must select exactly 137 pieces for the eight scene collision geoms")
    if visual_fingerprint(root) != visual_before:
        raise RuntimeError("visual geoms changed while applying collision overlay")
    if contact_exclude_fingerprint(root) != excludes_before:
        raise RuntimeError("collision exclusions changed while applying collision overlay")
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {
        **proxy_lock,
        "selected_collision_piece_count": sum(replacements.values()),
        "selected_piece_count_by_source_mesh": replacements,
        "raw_right_hand_collision_geom_count": len(replacements),
        "visual_fingerprint_unchanged": True,
        "contact_exclusions_unchanged": True,
    }


def validate_compiled_overlay(model: mujoco.MjModel) -> dict[str, object]:
    collision_names = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "__collision_0__" in name and name.startswith("R_hand_"):
            collision_names.append(name)
    if len(collision_names) != 137:
        raise RuntimeError(f"compiled overlay has {len(collision_names)} right-hand proxy geoms, expected 137")
    if any("_coacd_" not in name for name in collision_names):
        raise RuntimeError("compiled overlay retained an unexpanded right-hand collision geom")
    return {"compiled_right_hand_proxy_geom_count": len(collision_names)}


def valid_contact(item: dict[str, object]) -> bool:
    return bool(item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-static-result", required=True, type=Path)
    parser.add_argument("--truth-scan-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--command-percent", type=int, default=42)
    parser.add_argument("--alignment-scan-result", type=Path)
    parser.add_argument("--alignment-offset-mm", type=float)
    args = parser.parse_args()
    if not 0 <= args.command_percent <= 100:
        raise SystemExit("command percent must be within 0..100")
    for path in (args.baseline_static_result, args.truth_scan_result, args.prior_config):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    if args.alignment_scan_result is not None and not args.alignment_scan_result.is_file():
        raise SystemExit("alignment scan result is missing")

    baseline = json.loads(args.baseline_static_result.read_text(encoding="utf-8"))
    truth = json.loads(args.truth_scan_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    if truth["gates"].get("RAW_PROXY_COMMON_STATIC_ARM_TRANSPORT_50MM") != "PASS":
        raise SystemExit("prior raw/proxy comparison did not pass the static arm-transport gate")
    raw_scene = Path(baseline["generated_scene"]["path"])
    if not raw_scene.is_file() or sha256(raw_scene) != baseline["generated_scene"]["sha256"]:
        raise SystemExit("baseline raw scene is missing or hash changed")

    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    if loaded.model_sha256 != EXPECTED_STL_MODEL_SHA256:
        raise SystemExit("frozen STL model SHA-256 changed")
    if loaded.metadata["trajectory_sha256"] != EXPECTED_TRAJECTORY_SHA256:
        raise SystemExit("frozen STL trajectory SHA-256 changed")
    trajectory = load_registered_trajectory(loaded.package_path)
    pre = load_preflight_module()
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    alignment = None
    if args.alignment_scan_result is not None:
        alignment = json.loads(args.alignment_scan_result.read_text(encoding="utf-8"))
        if int(alignment["command_percent"]) != args.command_percent:
            raise SystemExit("alignment scan command does not match requested command")
        alignment_gates = alignment["gates"]
        if (
            alignment_gates.get("LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY") != "PASS"
            and alignment_gates.get("RAW_PROXY_COMMON_ARM_TRANSPORT_50MM") != "PASS"
        ):
            raise SystemExit("alignment scan did not pass static transport validation")

        def alignment_contact(item: dict[str, object]) -> dict[str, object]:
            contact = item.get("contact") or item.get("proxy_contact")
            if not isinstance(contact, dict):
                raise RuntimeError("alignment scan candidate lacks a contact record")
            return contact

        eligible = [
            item for item in alignment.get("candidates", [])
            if item.get("arm_transport_path_pass")
            and alignment_contact(item)["bilateral_pad_contact"]
            and not alignment_contact(item)["nonpad_right_contact_count"]
        ]
        if args.alignment_offset_mm is None:
            selected_alignment = alignment.get("best_candidate")
        else:
            selected_alignment = next(
                (
                    item for item in eligible
                    if abs(float(item["lateral_offset_mm"]) - args.alignment_offset_mm) <= 1e-9
                ),
                None,
            )
        if selected_alignment is None or selected_alignment not in eligible:
            raise SystemExit("requested alignment offset is not a valid CoACD static candidate")
        arm_source = selected_alignment["arm_q_rad"]
    else:
        selected_alignment = None
        arm_source = truth["arm_q_rad"]
    arm_start = np.asarray([arm_source[name] for name in pre.ARM_JOINTS], dtype=float)

    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-coacd-overlay-static"
    )
    output = location.native_path
    overlay_scene = output / "full_x2_stl_coacd_collision_overlay.xml"
    overlay = build_overlay(raw_scene, overlay_scene)
    model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    compiled = validate_compiled_overlay(model)

    contact = pre.evaluate_pose(model, trajectory, state, args.command_percent, arm_override=arm_start)
    contact["arm_q_rad"] = {
        name: float(value) for name, value in zip(pre.ARM_JOINTS, arm_start)
    }
    if not valid_contact(contact) or contact["table_contact_count"] != 1:
        raise SystemExit("initial CoACD overlay pose is not a bilateral table-supported pad contact")
    contact["highest_safe_bilateral_percent"] = args.command_percent
    contact["lateral_offset_mm"] = (
        0.0 if selected_alignment is None else float(selected_alignment["lateral_offset_mm"])
    )
    contact["selection_basis"] = (
        "approved CoACD source-STL collision overlay"
        if selected_alignment is None
        else "approved CoACD source-STL collision overlay with static lateral alignment"
    )
    closure_scan = []
    for percent in range(100, -1, -1):
        item = pre.evaluate_pose(model, trajectory, state, percent, arm_override=arm_start)
        closure_scan.append(
            {
                "command_percent": percent,
                "pad_bodies": item["pad_bodies"],
                "nonpad_right_contact_count": item["nonpad_right_contact_count"],
                "pad_distances_m": pre.pad_distance_map(item),
            }
        )
    contact["closure_contact_scan"] = closure_scan

    transport = []
    for fraction in np.linspace(0.0, 1.0, 5):
        lift_m = float(0.05 * fraction)
        item = pre.solve_gripper_translation_ik(
            model,
            trajectory,
            state,
            args.command_percent,
            np.asarray([0.0, 0.0, lift_m]),
            cylinder_lift_m=lift_m,
            start_arm_override=arm_start,
        )
        item["fraction"] = float(fraction)
        transport.append(item)
    transport_pass = all(
        valid_contact(item)
        and item["gripper_base_position_error_m"] <= 1e-4
        and item["gripper_base_orientation_error_rad"] <= 1e-3
        and (item["fraction"] == 0.0 or item["table_contact_count"] == 0)
        for item in transport
    )
    if not transport_pass:
        raise SystemExit("CoACD overlay did not preserve the 50 mm static pad-only transport path")

    result = {
        "schema": "robotsim.issue46.full-x2-stl-coacd-overlay-static.v1",
        "classification": "SIMULATION_ONLY_SOURCE_STL_COACD_COLLISION_OVERLAY",
        "mj_step_calls": 0,
        "mujoco_version": mujoco.__version__,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "baseline_raw_scene": {"path": str(raw_scene), "sha256": sha256(raw_scene)},
        "generated_scene": {"path": str(overlay_scene), "sha256": sha256(overlay_scene)},
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "truth_scan": {"path": str(args.truth_scan_result), "sha256": sha256(args.truth_scan_result)},
        "alignment_scan": (
            None
            if args.alignment_scan_result is None
            else {"path": str(args.alignment_scan_result), "sha256": sha256(args.alignment_scan_result)}
        ),
        "collision_overlay": overlay | compiled,
        "selected_command_percent": args.command_percent,
        "aligned_contact_static": contact,
        "controller_contact_target_static": contact,
        "transport_end_static": transport[-1],
        "controller_lift_target_static": transport[-1],
        "transport_path_static": transport,
        "lateral_alignment_scan": [
            {"lateral_offset_mm": 0.0, "arm_q_rad": contact["arm_q_rad"]}
        ],
        "gates": {
            "FROZEN_STL_MODEL_HASH": "PASS",
            "FROZEN_STL_TRAJECTORY_HASH": "PASS",
            "LOCKED_COACD_PROXY_HASHES": "PASS",
            "OVERLAY_VISUAL_GEOMS_UNCHANGED": "PASS",
            "OVERLAY_COLLISION_EXCLUSIONS_UNCHANGED": "PASS",
            "OVERLAY_COMPILED": "PASS",
            "BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The overlay is SIMULATION_ONLY and source-derived, not a manufacturer-provided collision model.",
            "The visual geometry and table-driven STL hand trajectory remain frozen.",
            "Static contact geometry does not prove force closure, load transfer, or a physical lift.",
        ],
        "reproduction": {
            "provision": "uv run --offline --with coacd==1.0.14 --with numpy==2.2.6 python3 simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/provision_collision.py",
            "preflight": "see source-controlled issue46_stl_coacd_overlay_static_preflight.py inputs recorded above",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "scene_sha256": result["generated_scene"]["sha256"],
                "selected_proxy_pieces": overlay["selected_collision_piece_count"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
