#!/usr/bin/env python3
"""Find the deepest source-table CoACD pad-only cylinder contact candidate.

This is a bounded static scan over source-table commands 20..42 and a small
lateral registration interval.  For each command it carries only the best
pad-only candidate into a five-sample 50 mm transport test.  No physics step,
source geometry change, contact exclusion, or actuator change occurs here.
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


def contact_depths(item: dict[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {pad: None for pad in PADS}
    for contact in item["pad_contacts"]:
        for body in contact["body_names"]:
            if body in result:
                value = float(contact["distance_m"])
                prior = result[body]
                result[body] = value if prior is None else min(prior, value)
    return result


def candidate_score(item: dict[str, object]) -> tuple[float, float]:
    values = [abs(float(value)) for value in item["pad_depth_m"].values() if value is not None]
    return (min(values), -abs(values[0] - values[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--min-command-percent", type=int, default=20)
    parser.add_argument("--max-command-percent", type=int, default=42)
    parser.add_argument("--offset-limit-mm", type=float, default=2.0)
    parser.add_argument("--offset-step-mm", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.min_command_percent <= args.max_command_percent <= 100:
        raise SystemExit("command range must be within 0..100")
    if args.offset_limit_mm <= 0 or args.offset_step_mm <= 0:
        raise SystemExit("offset limit and step must be positive")
    if not args.overlay_static_result.is_file() or not args.prior_config.is_file():
        raise SystemExit("overlay static result or prior config is missing")

    static = json.loads(args.overlay_static_result.read_text(encoding="utf-8"))
    required = ("LOCKED_COACD_PROXY_HASHES", "OVERLAY_VISUAL_GEOMS_UNCHANGED", "OVERLAY_COMPILED")
    if any(static["gates"].get(gate) != "PASS" for gate in required):
        raise SystemExit("overlay integrity preflight has not passed")
    scene = Path(static["generated_scene"]["path"])
    if not scene.is_file() or sha256(scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("overlay scene is missing or changed")
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    pre = load_preflight_module()
    model = mujoco.MjModel.from_xml_path(str(scene))
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    arm_seed = np.asarray(
        [static["aligned_contact_static"]["arm_q_rad"][name] for name in pre.ARM_JOINTS],
        dtype=float,
    )
    reference = mujoco.MjData(model)
    pre.apply_prior_state(model, reference, state, arm_seed)
    apply_stl_kinematic_pose(mujoco, model, reference, trajectory, args.max_command_percent)
    narrow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADS[0])
    wide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADS[1])
    lateral_axis = reference.xpos[wide] - reference.xpos[narrow]
    lateral_axis /= np.linalg.norm(lateral_axis)

    offsets = np.arange(
        -args.offset_limit_mm,
        args.offset_limit_mm + 0.5 * args.offset_step_mm,
        args.offset_step_mm,
    )
    by_command = []
    for command in range(args.min_command_percent, args.max_command_percent + 1):
        candidates = []
        for offset_mm in offsets:
            item = pre.solve_gripper_translation_ik(
                model,
                trajectory,
                state,
                command,
                lateral_axis * (float(offset_mm) / 1000.0),
                cylinder_lift_m=0.0,
                start_arm_override=arm_seed,
            )
            item["lateral_offset_mm"] = float(offset_mm)
            item["pad_depth_m"] = contact_depths(item)
            if valid_contact(item) and item["table_contact_count"] == 1 and all(item["pad_depth_m"].values()):
                candidates.append(item)
        best_static = max(candidates, key=candidate_score, default=None)
        transport = []
        transport_pass = False
        if best_static is not None:
            arm = np.asarray([best_static["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
            offset = lateral_axis * (float(best_static["lateral_offset_mm"]) / 1000.0)
            for fraction in np.linspace(0.0, 1.0, 5):
                lift_m = float(fraction * 0.05)
                sample = pre.solve_gripper_translation_ik(
                    model,
                    trajectory,
                    state,
                    command,
                    offset + np.asarray([0.0, 0.0, lift_m]),
                    cylinder_lift_m=lift_m,
                    start_arm_override=arm,
                )
                sample["fraction"] = float(fraction)
                transport.append(sample)
            transport_pass = all(
                valid_contact(sample)
                and sample["gripper_base_position_error_m"] <= 1e-4
                and sample["gripper_base_orientation_error_rad"] <= 1e-3
                and (sample["fraction"] == 0.0 or sample["table_contact_count"] == 0)
                for sample in transport
            )
        by_command.append(
            {
                "command_percent": command,
                "static_candidate_count": len(candidates),
                "best_static_candidate": best_static,
                "transport_samples": transport,
                "transport_path_pass": transport_pass,
            }
        )
    viable = [item for item in by_command if item["transport_path_pass"]]
    deepest = min(viable, key=lambda item: int(item["command_percent"]), default=None)
    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="stl-coacd-command-alignment-search"
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl-coacd-command-alignment-search.v1",
        "classification": "SIMULATION_ONLY_STATIC_SOURCE_STL_CONTACT_SEARCH",
        "mj_step_calls": 0,
        "overlay_static_result": {
            "path": str(args.overlay_static_result),
            "sha256": sha256(args.overlay_static_result),
        },
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "command_range": [args.min_command_percent, args.max_command_percent],
        "lateral_axis_world": lateral_axis.tolist(),
        "offset_limit_mm": args.offset_limit_mm,
        "offset_step_mm": args.offset_step_mm,
        "results_by_command": by_command,
        "deepest_viable_candidate": deepest,
        "gates": {
            "COACD_OVERLAY_INTEGRITY": "PASS",
            "DEEPER_PAD_ONLY_STATIC_CANDIDATE": "PASS" if viable else "BLOCKED",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS" if viable else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "This scan ranks source-table positions by static penetration only.",
            "It does not alter collision geometry, friction, model topology, or controller gains.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "commands_with_static_candidate": [
                    item["command_percent"] for item in by_command if item["best_static_candidate"] is not None
                ],
                "commands_with_transport_candidate": [item["command_percent"] for item in viable],
                "deepest_command_percent": None if deepest is None else deepest["command_percent"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
