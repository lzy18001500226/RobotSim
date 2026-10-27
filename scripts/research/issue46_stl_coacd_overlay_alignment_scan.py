#!/usr/bin/env python3
"""Find a balanced source-STL CoACD contact pose around one hand command.

The candidate scene must already have passed the CoACD overlay integrity
preflight.  This tool only calls ``mj_forward`` while scanning a bounded
lateral registration range and validating the same five static lift samples.
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
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


PADS = ("R_hand_narrow3_Link", "R_hand_wide3_Link")


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


def valid_contact(item: dict[str, object]) -> bool:
    return bool(item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"])


def depths(item: dict[str, object]) -> dict[str, float | None]:
    values: dict[str, float | None] = {name: None for name in PADS}
    for contact in item["pad_contacts"]:
        for name in contact["body_names"]:
            if name not in values:
                continue
            depth = float(contact["distance_m"])
            previous = values[name]
            values[name] = depth if previous is None else min(previous, depth)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--command-percent", type=int, default=42)
    parser.add_argument("--offset-limit-mm", type=float, default=3.0)
    parser.add_argument("--offset-step-mm", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.command_percent <= 100:
        raise SystemExit("command percent must be in 0..100")
    if args.offset_limit_mm <= 0 or args.offset_step_mm <= 0:
        raise SystemExit("offset limit and step must be positive")
    if not args.overlay_static_result.is_file() or not args.prior_config.is_file():
        raise SystemExit("overlay static result or prior config is missing")

    static = json.loads(args.overlay_static_result.read_text(encoding="utf-8"))
    if (
        static["gates"].get("LOCKED_COACD_PROXY_HASHES") != "PASS"
        and static["gates"].get("COACD_OVERLAY_INTEGRITY") != "PASS"
    ):
        raise SystemExit("CoACD overlay proxy hashes have not passed")
    scene = Path(static["generated_scene"]["path"])
    if not scene.is_file() or sha256(scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("CoACD overlay scene is missing or hash changed")
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    pre = load_preflight_module()
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    arm_start = np.asarray(
        [static["aligned_contact_static"]["arm_q_rad"][name] for name in pre.ARM_JOINTS],
        dtype=float,
    )
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    reference = mujoco.MjData(model)
    pre.apply_prior_state(model, reference, state, arm_start)
    apply_stl_kinematic_pose(mujoco, model, reference, trajectory, args.command_percent)
    narrow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADS[0])
    wide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADS[1])
    axis = reference.xpos[wide] - reference.xpos[narrow]
    axis /= np.linalg.norm(axis)

    offsets = np.arange(
        -args.offset_limit_mm,
        args.offset_limit_mm + 0.5 * args.offset_step_mm,
        args.offset_step_mm,
    )
    candidates = []
    for offset_mm in offsets:
        offset_m = axis * (float(offset_mm) / 1000.0)
        contact = pre.solve_gripper_translation_ik(
            model,
            trajectory,
            state,
            args.command_percent,
            offset_m,
            cylinder_lift_m=0.0,
            start_arm_override=arm_start,
        )
        arm = np.asarray([contact["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
        transport = []
        for fraction in np.linspace(0.0, 1.0, 5):
            lift_m = float(fraction * 0.05)
            item = pre.solve_gripper_translation_ik(
                model,
                trajectory,
                state,
                args.command_percent,
                offset_m + np.asarray([0.0, 0.0, lift_m]),
                cylinder_lift_m=lift_m,
                start_arm_override=arm,
            )
            item["fraction"] = float(fraction)
            transport.append(item)
        contact_depth = depths(contact)
        both_depths = [contact_depth[name] for name in PADS]
        balance_mm = (
            abs(float(both_depths[0]) - float(both_depths[1])) * 1000.0
            if all(value is not None for value in both_depths)
            else None
        )
        shallow_contact_mm = (
            min(abs(float(value)) for value in both_depths) * 1000.0
            if all(value is not None for value in both_depths)
            else 0.0
        )
        path_pass = all(
            valid_contact(item)
            and item["gripper_base_position_error_m"] <= 1e-4
            and item["gripper_base_orientation_error_rad"] <= 1e-3
            and (item["fraction"] == 0.0 or item["table_contact_count"] == 0)
            for item in transport
        )
        candidates.append(
            {
                "lateral_offset_mm": float(offset_mm),
                "arm_q_rad": contact["arm_q_rad"],
                "contact": contact,
                "pad_depth_m": contact_depth,
                "pad_balance_mm": balance_mm,
                "shallow_pad_penetration_mm": shallow_contact_mm,
                "arm_transport_samples": transport,
                "arm_transport_path_pass": path_pass,
            }
        )
    eligible = [
        item
        for item in candidates
        if valid_contact(item["contact"])
        and item["contact"]["table_contact_count"] == 1
        and item["arm_transport_path_pass"]
        and item["pad_balance_mm"] is not None
    ]
    # Prefer the largest weaker-side engagement, then the closest two-side depth.
    best = max(
        eligible,
        key=lambda item: (
            float(item["shallow_pad_penetration_mm"]),
            -float(item["pad_balance_mm"]),
        ),
        default=None,
    )
    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="stl-coacd-overlay-alignment-scan"
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl-coacd-overlay-alignment-scan.v1",
        "classification": "SIMULATION_ONLY_STATIC_CONTACT_ALIGNMENT",
        "mj_step_calls": 0,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "overlay_static_result": {
            "path": str(args.overlay_static_result),
            "sha256": sha256(args.overlay_static_result),
        },
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "command_percent": args.command_percent,
        "pad_axis_world": axis.tolist(),
        "offset_limit_mm": args.offset_limit_mm,
        "offset_step_mm": args.offset_step_mm,
        "candidates": candidates,
        "best_candidate": best,
        "gates": {
            "COACD_OVERLAY_INTEGRITY": "PASS",
            "BILATERAL_PAD_ONLY_STATIC_ALIGNMENT": "PASS" if eligible else "BLOCKED",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS" if eligible else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "This chooses a static source-STL CoACD contact registration only.",
            "Static penetration depth is not a physical normal-force or lift guarantee.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "eligible_offsets_mm": [item["lateral_offset_mm"] for item in eligible],
                "best_offset_mm": None if best is None else best["lateral_offset_mm"],
                "best_shallow_contact_mm": None if best is None else best["shallow_pad_penetration_mm"],
                "best_balance_mm": None if best is None else best["pad_balance_mm"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
