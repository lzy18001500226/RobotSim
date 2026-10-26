#!/usr/bin/env python3
"""Map source-supported p38 lateral approach offsets without stepping physics.

Every candidate is evaluated both in the raw whole-mesh scene and a read-only
source-STL CoACD comparison scene.  This tool changes neither scene input nor
the registered hand kinematics; it calls ``mj_forward`` only.
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


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_module(name: str, filename: str):
    path = REPO / "scripts/research" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad_depths(item: dict[str, object]) -> dict[str, float | None]:
    depths: dict[str, float | None] = {pad: None for pad in PADS}
    for contact in item["pad_contacts"]:
        for body in contact["body_names"]:
            if body in depths:
                value = float(contact["distance_m"])
                current = depths[body]
                depths[body] = value if current is None else min(current, value)
    return depths


def valid(item: dict[str, object]) -> bool:
    return bool(item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-static-result", required=True, type=Path)
    parser.add_argument("--truth-scan-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--command-percent", type=int, default=38)
    args = parser.parse_args()
    if not 0 <= args.command_percent <= 100:
        raise SystemExit("command percent must be in 0..100")
    baseline = json.loads(args.baseline_static_result.read_text(encoding="utf-8"))
    truth = json.loads(args.truth_scan_result.read_text(encoding="utf-8"))
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    # The seed p38..p42 range has already passed the truth scan.  Adjacent
    # command values are allowed here only to test whether a small source-pose
    # lateral registration clears a non-pad contact; the raw/proxy gate below
    # remains mandatory for every output candidate.

    pre = load_module("stl_static_preflight", "issue46_stl_cylinder_static_preflight.py")
    comparison = load_module("stl_raw_proxy_compare", "issue46_stl_raw_proxy_contact_truth_scan.py")
    raw_scene = Path(baseline["generated_scene"]["path"])
    if not raw_scene.is_file() or digest(raw_scene) != baseline["generated_scene"]["sha256"]:
        raise SystemExit("baseline raw scene missing or changed")
    state = pre.state_by_phase(config, "HORIZONTAL_APPROACH")
    arm0 = np.asarray([truth["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)

    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46",
        prefix="stl-common-contact-alignment-scan",
    )
    output = location.native_path
    proxy_scene = output / "full_x2_stl_proxy_contact.xml"
    proxy_info = comparison.build_proxy_scene(raw_scene, proxy_scene)
    raw_model = mujoco.MjModel.from_xml_path(str(raw_scene))
    proxy_model = mujoco.MjModel.from_xml_path(str(proxy_scene))
    reference = mujoco.MjData(raw_model)
    pre.apply_prior_state(raw_model, reference, state, arm0)
    apply_stl_kinematic_pose(mujoco, raw_model, reference, trajectory, float(args.command_percent))
    narrow = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, PADS[0])
    wide = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, PADS[1])
    axis = reference.xpos[wide] - reference.xpos[narrow]
    axis /= np.linalg.norm(axis)

    candidates = []
    for offset_mm in np.linspace(-4.0, 4.0, 17):
        contact = pre.solve_gripper_translation_ik(
            raw_model,
            trajectory,
            state,
            args.command_percent,
            axis * (float(offset_mm) / 1000.0),
            cylinder_lift_m=0.0,
            start_arm_override=arm0,
        )
        arm = np.asarray([contact["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
        proxy_contact = pre.evaluate_pose(
            proxy_model, trajectory, state, args.command_percent, arm_override=arm
        )
        raw_depth = pad_depths(contact)
        proxy_depth = pad_depths(proxy_contact)
        transport = []
        for fraction in np.linspace(0.0, 1.0, 5):
            lift_m = float(0.05 * fraction)
            raw_lift = pre.solve_gripper_translation_ik(
                raw_model,
                trajectory,
                state,
                args.command_percent,
                axis * (float(offset_mm) / 1000.0) + np.asarray([0.0, 0.0, lift_m]),
                cylinder_lift_m=lift_m,
                start_arm_override=arm,
            )
            lift_arm = np.asarray(
                [raw_lift["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float
            )
            proxy_lift = pre.evaluate_pose(
                proxy_model,
                trajectory,
                state,
                args.command_percent,
                cylinder_lift_m=lift_m,
                arm_override=lift_arm,
            )
            transport.append(
                {
                    "lift_m": lift_m,
                    "raw": raw_lift,
                    "proxy": proxy_lift,
                    "common_pad_only": bool(valid(raw_lift) and valid(proxy_lift)),
                    "table_detached_after_start": bool(
                        fraction == 0.0
                        or (
                            raw_lift["table_contact_count"] == 0
                            and proxy_lift["table_contact_count"] == 0
                        )
                    ),
                }
            )
        path_pass = all(
            item["common_pad_only"] and item["table_detached_after_start"]
            for item in transport
        )
        depths = [proxy_depth[pad] for pad in PADS]
        balance_mm = (
            abs(float(depths[0]) - float(depths[1])) * 1000.0
            if all(value is not None for value in depths)
            else None
        )
        candidates.append(
            {
                "lateral_offset_mm": float(offset_mm),
                "arm_q_rad": contact["arm_q_rad"],
                "raw_contact": contact,
                "proxy_contact": proxy_contact,
                "raw_pad_depth_m": raw_depth,
                "proxy_pad_depth_m": proxy_depth,
                "proxy_pad_depth_balance_mm": balance_mm,
                "arm_transport_samples": transport,
                "arm_transport_path_pass": path_pass,
            }
        )
    eligible = [
        item
        for item in candidates
        if valid(item["raw_contact"])
        and valid(item["proxy_contact"])
        and item["arm_transport_path_pass"]
        and item["proxy_pad_depth_balance_mm"] is not None
    ]
    best = min(eligible, key=lambda item: float(item["proxy_pad_depth_balance_mm"])) if eligible else None
    result = {
        "schema": "robotsim.issue46.stl-common-contact-alignment-scan.v1",
        "classification": "READ_ONLY_STATIC_CONTACT_ALIGNMENT",
        "mj_step_calls": 0,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "baseline_raw_scene": {"path": str(raw_scene), "sha256": digest(raw_scene)},
        "proxy_scene": {"path": str(proxy_scene), "sha256": digest(proxy_scene), **proxy_info},
        "prior_config": {"path": str(args.prior_config), "sha256": digest(args.prior_config)},
        "command_percent": args.command_percent,
        "pad_axis_world": axis.tolist(),
        "candidates": candidates,
        "best_balanced_candidate": best,
        "gates": {
            "RAW_PROXY_COMMON_CONTACT_ALIGNMENT": "PASS" if eligible else "BLOCKED",
            "RAW_PROXY_COMMON_ARM_TRANSPORT_50MM": "PASS" if eligible else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The balance score compares static source-proxy penetration depths, not dynamic contact forces.",
            "No model or controller parameter was changed and no physics step was executed.",
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
                "best_offset_mm": best["lateral_offset_mm"] if best else None,
                "best_balance_mm": best["proxy_pad_depth_balance_mm"] if best else None,
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
