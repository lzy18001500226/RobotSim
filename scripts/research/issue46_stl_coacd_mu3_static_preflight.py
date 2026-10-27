#!/usr/bin/env python3
"""Restore the existing SIMULATION_ONLY mu=3 cylinder contact overlay.

The approved STL hand, its CoACD collision overlay, table, and cylinder
geometry remain untouched.  This creates a distinct review-scene XML whose
only parameter difference is ``test_cylinder_collision`` sliding friction
``1 -> 3``.  It matches the existing maximum-friction cylinder experiment
contract and executes ``mj_forward`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


MU3 = [3.0, 0.005, 0.0001]
DEFAULT = [1.0, 0.005, 0.0001]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_fingerprint(root: ET.Element):
    return sorted(
        tuple(sorted(geom.attrib.items()))
        for geom in root.findall(".//geom")
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0"
    )


def contact_excludes(root: ET.Element):
    contact = root.find("contact")
    return [] if contact is None else sorted(tuple(sorted(item.attrib.items())) for item in contact)


def hand_proxy_fingerprint(root: ET.Element):
    return sorted(
        tuple(sorted(geom.attrib.items()))
        for geom in root.findall(".//geom")
        if geom.get("name", "").startswith("R_hand_")
        and "__collision_0__" in geom.get("name", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-static-result", required=True, type=Path)
    parser.add_argument("--prior-max-friction-config", required=True, type=Path)
    args = parser.parse_args()
    if not args.overlay_static_result.is_file() or not args.prior_max_friction_config.is_file():
        raise SystemExit("overlay static result or prior maximum-friction config is missing")
    static = json.loads(args.overlay_static_result.read_text(encoding="utf-8"))
    required = ("LOCKED_COACD_PROXY_HASHES", "OVERLAY_VISUAL_GEOMS_UNCHANGED", "OVERLAY_COMPILED")
    if any(static["gates"].get(gate) != "PASS" for gate in required):
        raise SystemExit("source CoACD overlay integrity preflight has not passed")
    source_scene = Path(static["generated_scene"]["path"])
    if not source_scene.is_file() or sha256(source_scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("source CoACD overlay scene is missing or changed")
    prior_config = json.loads(args.prior_max_friction_config.read_text(encoding="utf-8"))
    prior_override = prior_config.get("simulation_only_friction_override", {})
    if prior_override.get("derived_friction") != MU3:
        raise SystemExit("prior config does not prove the existing mu=3 simulation-only overlay")

    tree = ET.parse(source_scene)
    root = tree.getroot()
    visuals_before = visual_fingerprint(root)
    excludes_before = contact_excludes(root)
    proxies_before = hand_proxy_fingerprint(root)
    cylinder = next(
        (geom for geom in root.findall(".//geom") if geom.get("name") == "test_cylinder_collision"),
        None,
    )
    if cylinder is None:
        raise SystemExit("source overlay scene lacks test_cylinder_collision")
    source_friction_text = cylinder.get("friction")
    if source_friction_text != "1 0.005 0.0001":
        raise SystemExit(f"unexpected source cylinder friction: {source_friction_text}")
    cylinder.set("friction", "3 0.005 0.0001")
    if visual_fingerprint(root) != visuals_before:
        raise RuntimeError("visual geom changed during friction overlay")
    if contact_excludes(root) != excludes_before:
        raise RuntimeError("contact exclusions changed during friction overlay")
    if hand_proxy_fingerprint(root) != proxies_before:
        raise RuntimeError("hand collision proxy changed during friction overlay")

    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-coacd-mu3-static"
    )
    output = location.native_path
    destination = output / "full_x2_stl_coacd_mu3_collision_overlay.xml"
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(destination))
    cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_cylinder_collision")
    if cylinder_id < 0:
        raise RuntimeError("compiled mu3 scene lacks cylinder geom")
    actual_friction = model.geom_friction[cylinder_id].tolist()
    if actual_friction != MU3:
        raise RuntimeError(f"compiled cylinder friction differs: {actual_friction}")
    hand_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "R_hand_narrow3_Link__collision_0__narrow3_Link_coacd_00.stl",
    )
    if hand_id < 0 or model.geom_friction[hand_id].tolist() != DEFAULT:
        raise RuntimeError("hand collision friction unexpectedly changed")

    result = dict(static)
    result.update(
        {
            "schema": "robotsim.issue46.full-x2-stl-coacd-mu3-static.v1",
            "classification": "SIMULATION_ONLY_SOURCE_STL_COACD_MU3_CONTACT_OVERLAY",
            "mj_step_calls": 0,
            "source_overlay_static_result": {
                "path": str(args.overlay_static_result),
                "sha256": sha256(args.overlay_static_result),
            },
            "prior_max_friction_config": {
                "path": str(args.prior_max_friction_config),
                "sha256": sha256(args.prior_max_friction_config),
            },
            "generated_scene": {"path": str(destination), "sha256": sha256(destination)},
            "contact_parameter_overlay": {
                "changed_geom": "test_cylinder_collision",
                "source_friction": DEFAULT,
                "derived_friction": MU3,
                "compiled_cylinder_friction": actual_friction,
                "compiled_hand_friction": model.geom_friction[hand_id].tolist(),
                "only_xml_parameter_changed": "test_cylinder_collision@friction",
                "visual_geoms_unchanged": True,
                "hand_proxy_geoms_unchanged": True,
                "contact_exclusions_unchanged": True,
            },
        }
    )
    result["gates"] = dict(static["gates"]) | {
        "SIMULATION_ONLY_MU3_CONTACT_OVERLAY": "PASS",
        "DYNAMICS": "NOT_RUN",
    }
    result["limitations"] = list(static.get("limitations", [])) + [
        "Cylinder mu=3 is an existing SIMULATION_ONLY contact-parameter overlay, not a vendor friction claim.",
    ]
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "scene_sha256": result["generated_scene"]["sha256"],
                "friction": actual_friction,
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
