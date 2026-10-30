#!/usr/bin/env python3
"""Build a reversible pad-only material overlay for the frozen STL baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
PAD_PREFIXES = ("R_hand_narrow3_Link__collision_0__", "R_hand_wide3_Link__collision_0__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(root: ET.Element, selected: bool) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    rows = []
    for geom in root.findall(".//geom"):
        match = geom.get("name", "").startswith(PAD_PREFIXES)
        if match == selected:
            rows.append((geom.get("name", ""), tuple(sorted(geom.attrib.items()))))
    return sorted(rows)


def contact_fingerprint(root: ET.Element) -> list[tuple[tuple[str, str], ...]]:
    contact = root.find("contact")
    return [] if contact is None else sorted(tuple(sorted(item.attrib.items())) for item in contact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--sliding-friction", type=float, default=3.0)
    parser.add_argument("--torsional-friction", type=float, default=0.02)
    parser.add_argument("--rolling-friction", type=float, default=0.001)
    parser.add_argument("--condim", type=int, choices=(3, 4, 6), default=4)
    parser.add_argument("--solref", nargs=2, type=float, default=(0.02, 1.0), metavar=("TIME", "DAMP"))
    parser.add_argument("--solimp", nargs=5, type=float, default=(0.9, 0.95, 0.001, 0.5, 2.0), metavar=("D0", "D_WIDTH", "WIDTH", "MID", "POWER"))
    parser.add_argument("--pad-priority", type=int, choices=(0, 1), default=0)
    parser.add_argument("--overlay-id", default="STL-PadMaterial-SIM-v1")
    args = parser.parse_args()
    if not args.freeze_record.is_file() or min(args.sliding_friction, args.torsional_friction, args.rolling_friction, *args.solref, *args.solimp) <= 0.0:
        raise SystemExit("freeze record and strictly positive contact parameters are required")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    if frozen.get("baseline_id") != "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1":
        raise SystemExit("freeze record is not the accepted horizontal STL baseline")
    review_path = Path(frozen["accepted_review"]["path"])
    if not review_path.is_file() or sha256(review_path) != frozen["accepted_review"]["sha256"]:
        raise SystemExit("accepted review changed after freeze")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    scene = Path(review["scene"]["path"])
    if not scene.is_file() or sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted scene changed after freeze")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"]:
        raise SystemExit("frozen model hash differs from freeze record")
    tree = ET.parse(scene)
    root = tree.getroot()
    nonpad_before = fingerprint(root, selected=False)
    pad_before = fingerprint(root, selected=True)
    contact_before = contact_fingerprint(root)
    targets = [geom for geom in root.findall(".//geom") if geom.get("name", "").startswith(PAD_PREFIXES)]
    if len(targets) != 22:
        raise RuntimeError(f"expected 22 distal-pad CoACD geoms, found {len(targets)}")
    friction = f"{args.sliding_friction:.17g} {args.torsional_friction:.17g} {args.rolling_friction:.17g}"
    for geom in targets:
        geom.set("friction", friction)
        geom.set("condim", str(args.condim))
        geom.set("solref", " ".join(f"{value:.17g}" for value in args.solref))
        geom.set("solimp", " ".join(f"{value:.17g}" for value in args.solimp))
        geom.set("priority", str(args.pad_priority))
    if fingerprint(root, selected=False) != nonpad_before:
        raise RuntimeError("non-pad geometry changed in a pad-material overlay")
    if contact_fingerprint(root) != contact_before:
        raise RuntimeError("collision exclusions or explicit pairs changed in a pad-material overlay")
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-pad-material-overlay")
    output = location.native_path
    generated = output / "full_x2_stl_horizontal_pad_material_overlay.xml"
    ET.indent(tree, space="  ")
    pg.write_new_bytes(generated, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    result = {
        "schema": "robotsim.issue46.stl.horizontal-pad-material-overlay.v1",
        "classification": "SIMULATION_ONLY_PAD_CONTACT_MATERIAL_OVERLAY",
        "mj_step_calls": 0,
        "overlay_id": args.overlay_id,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "freeze_record": {"path": str(args.freeze_record), "sha256": sha256(args.freeze_record)},
        "source_scene": {"path": str(scene), "sha256": sha256(scene)},
        "generated_scene": {"path": str(generated), "sha256": sha256(generated)},
        "changed_pad_collision_geom_count": len(targets),
        "material": {"sliding_friction": args.sliding_friction, "torsional_friction": args.torsional_friction, "rolling_friction": args.rolling_friction, "condim": args.condim, "solref": args.solref, "solimp": args.solimp, "pad_priority": args.pad_priority},
        "nonpad_geom_fingerprint_sha256": hashlib.sha256(repr(nonpad_before).encode()).hexdigest(),
        "pad_geom_fingerprint_before_sha256": hashlib.sha256(repr(pad_before).encode()).hexdigest(),
        "gates": {"FROZEN_STL_MODEL": "PASS", "SOURCE_SCENE": "PASS", "NONPAD_GEOMETRY_UNCHANGED": "PASS", "COLLISION_EXCLUSIONS_UNCHANGED": "PASS", "PAD_MATERIAL_OVERLAY": "PASS", "DYNAMICS": "NOT_RUN"},
        "limitations": ["This is a contact-material experiment only. It is not an official material specification or manufacturer-equivalent model."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "overlay_id": result["overlay_id"], "material": result["material"], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
