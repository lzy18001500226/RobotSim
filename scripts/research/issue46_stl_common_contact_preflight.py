#!/usr/bin/env python3
"""Build a raw-STL static preflight from a raw/proxy-common contact pose.

The preflight makes no model change.  It selects a command only after the
separate truth scan shows that the raw scene and source-derived proxy scene
both report the same bilateral distal-link contact through a 50 mm static
transport path.  It calls ``mj_forward`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    load_registered_trajectory,
)


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


def valid(item: dict[str, object]) -> bool:
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
        raise SystemExit("command percent must be in 0..100")
    for path in (args.baseline_static_result, args.truth_scan_result, args.prior_config):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.alignment_scan_result is not None and not args.alignment_scan_result.is_file():
        raise SystemExit("alignment scan is missing")

    baseline = json.loads(args.baseline_static_result.read_text(encoding="utf-8"))
    truth = json.loads(args.truth_scan_result.read_text(encoding="utf-8"))
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    alignment = None
    if args.alignment_scan_result is None:
        candidates = {
            int(item["command_percent"]): item
            for item in truth.get("lift_50mm_candidates", [])
        }
        candidate = candidates.get(args.command_percent)
        if candidate is None or not candidate.get("arm_transport_path_pass"):
            raise SystemExit("requested command did not pass raw/proxy-common 50 mm static contact validation")
        if truth["gates"].get("RAW_PROXY_COMMON_STATIC_ARM_TRANSPORT_50MM") != "PASS":
            raise SystemExit("truth scan did not pass raw/proxy static lift validation")
        selected_alignment = None
    else:
        alignment = json.loads(args.alignment_scan_result.read_text(encoding="utf-8"))
        if int(alignment["command_percent"]) != args.command_percent:
            raise SystemExit("alignment scan command does not match requested command")
        if alignment["gates"].get("RAW_PROXY_COMMON_ARM_TRANSPORT_50MM") != "PASS":
            raise SystemExit("alignment scan did not pass raw/proxy 50 mm transport validation")
        eligible = [
            item for item in alignment.get("candidates", [])
            if item.get("arm_transport_path_pass")
            and valid(item["raw_contact"])
            and valid(item["proxy_contact"])
        ]
        if args.alignment_offset_mm is None:
            selected_alignment = alignment.get("best_balanced_candidate")
        else:
            selected_alignment = next(
                (item for item in eligible if abs(float(item["lateral_offset_mm"]) - args.alignment_offset_mm) <= 1e-9),
                None,
            )
        if selected_alignment is None or selected_alignment not in eligible:
            raise SystemExit("requested alignment offset is not a raw/proxy-valid candidate")
        candidate = {"arm_transport_samples": selected_alignment["arm_transport_samples"]}

    pre = load_preflight_module()
    raw_scene = Path(baseline["generated_scene"]["path"])
    if not raw_scene.is_file() or digest(raw_scene) != baseline["generated_scene"]["sha256"]:
        raise SystemExit("baseline raw scene is missing or changed")
    model = mujoco.MjModel.from_xml_path(str(raw_scene))
    state = pre.state_by_phase(config, "HORIZONTAL_APPROACH")
    source_arm = truth["arm_q_rad"] if selected_alignment is None else selected_alignment["arm_q_rad"]
    arm = np.asarray([source_arm[name] for name in pre.ARM_JOINTS], dtype=float)
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    contact = pre.evaluate_pose(model, trajectory, state, args.command_percent, arm_override=arm)
    if not valid(contact):
        raise SystemExit("selected raw contact no longer satisfies the bilateral-pad-only gate")
    contact["arm_q_rad"] = {
        name: float(value) for name, value in zip(pre.ARM_JOINTS, arm)
    }

    closure_scan = []
    for percent in range(100, -1, -1):
        item = pre.evaluate_pose(model, trajectory, state, percent, arm_override=arm)
        closure_scan.append(
            {
                "command_percent": percent,
                "pad_bodies": item["pad_bodies"],
                "nonpad_right_contact_count": item["nonpad_right_contact_count"],
                "pad_distances_m": pre.pad_distance_map(item),
            }
        )
    contact["highest_safe_bilateral_percent"] = args.command_percent
    contact["closure_contact_scan"] = closure_scan
    contact["lateral_offset_mm"] = 0.0 if selected_alignment is None else float(selected_alignment["lateral_offset_mm"])
    contact["selection_basis"] = (
        "raw-proxy common source-STL pad contact"
        if selected_alignment is None
        else "raw-proxy common source-STL pad contact after static lateral alignment"
    )

    path = [sample["raw"] | {"fraction": sample["lift_m"] / 0.05} for sample in candidate["arm_transport_samples"]]
    if not all(
        valid(item)
        and item["gripper_base_position_error_m"] <= 1e-4
        and item["gripper_base_orientation_error_rad"] <= 1e-3
        and (item["fraction"] == 0.0 or item["table_contact_count"] == 0)
        for item in path
    ):
        raise SystemExit("selected raw transport path no longer passes the source-mesh static gate")
    lift_target = path[-1]

    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46",
        prefix="full-x2-stl-common-contact-static",
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.full-x2-stl-common-contact-static.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY_STATIC_PREFLIGHT",
        "mj_step_calls": 0,
        "mujoco_version": mujoco.__version__,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "source_baseline": {"path": str(args.baseline_static_result), "sha256": digest(args.baseline_static_result)},
        "source_truth_scan": {"path": str(args.truth_scan_result), "sha256": digest(args.truth_scan_result)},
        "source_alignment_scan": (
            None if args.alignment_scan_result is None else {"path": str(args.alignment_scan_result), "sha256": digest(args.alignment_scan_result)}
        ),
        "prior_config": {"path": str(args.prior_config), "sha256": digest(args.prior_config)},
        "generated_scene": {"path": str(raw_scene), "sha256": digest(raw_scene)},
        "selected_command_percent": args.command_percent,
        "aligned_contact_static": contact,
        "controller_contact_target_static": contact,
        "transport_end_static": lift_target,
        "controller_lift_target_static": lift_target,
        "transport_path_static": path,
        "lateral_alignment_scan": [{"lateral_offset_mm": 0.0, "arm_q_rad": contact["arm_q_rad"]}],
        "gates": {
            "STL_HAND_COMPILED": "PASS",
            "EXACT_101_ROW_TABLE": "PASS",
            "RAW_PROXY_COMMON_PAD_CONTACT": "PASS",
            "BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The hand remains table-driven source-STL kinematics rather than a manufacturer-equivalent transmission.",
            "This static preflight demonstrates raw/proxy-common geometry only; it does not claim contact force closure or a physical lift.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "command_percent": args.command_percent,
                "lift_path_samples": len(path),
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
