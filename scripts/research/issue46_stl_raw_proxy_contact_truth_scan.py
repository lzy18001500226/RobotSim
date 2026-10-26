#!/usr/bin/env python3
"""Find source-STL pad contacts that agree in raw and convex-decomposed scenes.

This is a read-only static gate.  It writes qpos then calls ``mj_forward``;
it never calls ``mj_step`` and does not alter the registered model, trajectory,
robot, cylinder, or table.
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
AUDIT_PACKAGE = Path(
    "/home/lzy18001500226/robotsim-issue46-stl-mainline-audit/"
    "simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1"
)
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_preflight_module():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("stl_static_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load static preflight helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_proxy_scene(raw_scene: Path, destination: Path) -> dict[str, object]:
    """Replace only right-hand collision geoms in a review-only XML copy."""
    tree = ET.parse(raw_scene)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("raw scene has no asset section")
    manifest_path = AUDIT_PACKAGE / "collision_proxy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proxy_dir = AUDIT_PACKAGE / "collision_generated"
    partmap = {
        Path(item["source_mesh"]).stem: [part["filename"] for part in item["parts"]]
        for item in manifest["mesh_decompositions"]
    }
    for parts in partmap.values():
        for filename in parts:
            ET.SubElement(
                asset,
                "mesh",
                name=f"proxy_{filename}",
                file=str(proxy_dir / filename),
            )
    replaced = 0
    replaced_source_stems: set[str] = set()
    for parent in root.iter():
        for geom in list(parent):
            name = geom.get("name", "")
            if not (name.startswith("R_hand_") and "__collision_0" in name):
                continue
            source_stem = Path(geom.get("mesh", "")).stem
            if source_stem not in partmap:
                continue
            index = list(parent).index(geom)
            parent.remove(geom)
            for filename in partmap[source_stem]:
                replacement = copy.deepcopy(geom)
                replacement.set("name", f"{name}__{filename}")
                replacement.set("mesh", f"proxy_{filename}")
                parent.insert(index, replacement)
                index += 1
            replaced += 1
            replaced_source_stems.add(source_stem)
    # The raw full-X2 scene has collision geoms for eight right-hand link
    # meshes.  The base mesh appears only as a visual geom in this scene, so
    # requiring every manifest source would incorrectly reject a faithful copy.
    if not replaced_source_stems:
        raise RuntimeError("raw scene has no replaceable right-hand collision geoms")
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {
        "manifest_sha256": digest(manifest_path),
        "piece_count": sum(len(parts) for parts in partmap.values()),
        "replaced_raw_collision_geom_count": replaced,
        "replaced_source_meshes": sorted(replaced_source_stems),
    }


def evaluate(pre, model, trajectory, state, arm_q_rad: np.ndarray, percent: int, lift_m: float):
    return pre.evaluate_pose(
        model,
        trajectory,
        state,
        percent,
        cylinder_lift_m=lift_m,
        arm_override=arm_q_rad,
    )


def is_pad_only_contact(item: dict[str, object]) -> bool:
    return bool(
        item["bilateral_pad_contact"]
        and not item["nonpad_right_contact_count"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    args = parser.parse_args()
    if not args.baseline_static_result.is_file() or not args.prior_config.is_file():
        raise SystemExit("baseline result or prior configuration is missing")

    baseline = json.loads(args.baseline_static_result.read_text(encoding="utf-8"))
    raw_scene = Path(baseline["generated_scene"]["path"])
    if not raw_scene.is_file():
        raise SystemExit("baseline raw scene is missing")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    pre = load_preflight_module()
    state = pre.state_by_phase(config, "HORIZONTAL_APPROACH")
    aligned = baseline["aligned_contact_static"]
    arm_q_rad = np.asarray([aligned["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)

    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46",
        prefix="stl-raw-proxy-contact-truth-scan",
    )
    output = location.native_path
    proxy_scene = output / "full_x2_stl_proxy_contact.xml"
    proxy_info = build_proxy_scene(raw_scene, proxy_scene)
    raw_model = mujoco.MjModel.from_xml_path(str(raw_scene))
    proxy_model = mujoco.MjModel.from_xml_path(str(proxy_scene))

    rows = []
    viable = []
    for percent in range(101):
        raw = evaluate(pre, raw_model, trajectory, state, arm_q_rad, percent, 0.0)
        proxy = evaluate(pre, proxy_model, trajectory, state, arm_q_rad, percent, 0.0)
        row = {
            "command_percent": percent,
            "raw": raw,
            "proxy": proxy,
            "raw_pad_only": is_pad_only_contact(raw),
            "proxy_pad_only": is_pad_only_contact(proxy),
        }
        row["common_pad_only"] = bool(row["raw_pad_only"] and row["proxy_pad_only"])
        rows.append(row)
        if not row["common_pad_only"]:
            continue
        fixed_hand_lift_samples = []
        for fraction in np.linspace(0.0, 1.0, 5):
            lift_m = float(0.05 * fraction)
            raw_lift = evaluate(pre, raw_model, trajectory, state, arm_q_rad, percent, lift_m)
            proxy_lift = evaluate(pre, proxy_model, trajectory, state, arm_q_rad, percent, lift_m)
            fixed_hand_lift_samples.append(
                {
                    "lift_m": lift_m,
                    "raw": raw_lift,
                    "proxy": proxy_lift,
                    "common_pad_only": bool(
                        is_pad_only_contact(raw_lift) and is_pad_only_contact(proxy_lift)
                    ),
                    "table_detached_after_start": bool(
                        fraction == 0.0
                        or (
                            raw_lift["table_contact_count"] == 0
                            and proxy_lift["table_contact_count"] == 0
                        )
                    ),
                }
            )
        fixed_hand_path_pass = all(
            sample["common_pad_only"] and sample["table_detached_after_start"]
            for sample in fixed_hand_lift_samples
        )
        arm_transport_samples = []
        for fraction in np.linspace(0.0, 1.0, 5):
            lift_m = float(0.05 * fraction)
            raw_transport = pre.solve_gripper_translation_ik(
                raw_model,
                trajectory,
                state,
                percent,
                np.asarray([0.0, 0.0, lift_m]),
                cylinder_lift_m=lift_m,
                start_arm_override=arm_q_rad,
            )
            transport_arm = np.asarray(
                [raw_transport["arm_q_rad"][name] for name in pre.ARM_JOINTS],
                dtype=float,
            )
            proxy_transport = evaluate(
                pre,
                proxy_model,
                trajectory,
                state,
                transport_arm,
                percent,
                lift_m,
            )
            arm_transport_samples.append(
                {
                    "lift_m": lift_m,
                    "raw": raw_transport,
                    "proxy": proxy_transport,
                    "common_pad_only": bool(
                        is_pad_only_contact(raw_transport)
                        and is_pad_only_contact(proxy_transport)
                    ),
                    "table_detached_after_start": bool(
                        fraction == 0.0
                        or (
                            raw_transport["table_contact_count"] == 0
                            and proxy_transport["table_contact_count"] == 0
                        )
                    ),
                }
            )
        arm_transport_path_pass = all(
            sample["common_pad_only"] and sample["table_detached_after_start"]
            for sample in arm_transport_samples
        )
        viable.append(
            {
                "command_percent": percent,
                "fixed_hand_lift_path_pass": fixed_hand_path_pass,
                "fixed_hand_lift_samples": fixed_hand_lift_samples,
                "arm_transport_path_pass": arm_transport_path_pass,
                "arm_transport_samples": arm_transport_samples,
            }
        )

    passing = [row for row in viable if row["arm_transport_path_pass"]]
    result = {
        "schema": "robotsim.issue46.stl-raw-proxy-contact-truth-scan.v2",
        "classification": "READ_ONLY_STATIC_CONTACT_VALIDATION",
        "mj_step_calls": 0,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "baseline_raw_scene": {"path": str(raw_scene), "sha256": digest(raw_scene)},
        "proxy_scene": {"path": str(proxy_scene), "sha256": digest(proxy_scene), **proxy_info},
        "prior_config": {"path": str(args.prior_config), "sha256": digest(args.prior_config)},
        "arm_q_rad": {name: float(value) for name, value in zip(pre.ARM_JOINTS, arm_q_rad)},
        "contact_rows": rows,
        "common_contact_candidates": viable,
        "lift_50mm_candidates": passing,
        "gates": {
            "RAW_PROXY_COMMON_PAD_CONTACT": "PASS" if viable else "BLOCKED",
            "RAW_PROXY_COMMON_STATIC_ARM_TRANSPORT_50MM": "PASS" if passing else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "Both models share the same visual STL, joints, trajectory, table, cylinder and arm state.",
            "This compares collision representations under mj_forward only; it does not establish force closure or a physical lift.",
            "A fixed-hand object-offset scan is reported separately from the real arm-transport path and is not a lift claim.",
            "The proxy XML is an evidence-only copy and is not selected as a runtime model.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "common_percentages": [item["command_percent"] for item in viable],
                "arm_transport_50mm_percentages": [item["command_percent"] for item in passing],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
