#!/usr/bin/env python3
"""Bounded source-STL CoACD contact-pose search around the reviewed approach.

The search keeps the approved hand orientation and source trajectory fixed. It
only asks arm IK for small Cartesian translations of the hand base, then checks
real CoACD contacts using ``mj_forward``.  It is not a mechanism or collision
parameter search and never calls ``mj_step``.
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


def contact_geometry(model, trajectory, pre, state, arm_q: np.ndarray, command: int):
    data = mujoco.MjData(model)
    pre.apply_prior_state(model, data, state, arm_q)
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, command)
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    positions: dict[str, list[np.ndarray]] = {pad: [] for pad in PADS}
    normals: dict[str, list[np.ndarray]] = {pad: [] for pad in PADS}
    robot_table_contacts = 0
    for contact in data.contact:
        body_ids = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            for body in body_ids
        )
        if "raicom_table" in names and any(name.startswith(("R_hand_", "right_")) for name in names):
            robot_table_contacts += 1
        if cylinder not in body_ids:
            continue
        for pad in PADS:
            if pad not in names:
                continue
            positions[pad].append(np.asarray(contact.pos, dtype=float))
            normals[pad].append(np.asarray(contact.frame, dtype=float).reshape(3, 3)[0].copy())
    if not all(positions[pad] for pad in PADS):
        return {"robot_table_contact_count": robot_table_contacts, "height_delta_mm": None, "normal_dot": None}
    centers = {pad: np.mean(np.vstack(positions[pad]), axis=0) for pad in PADS}
    directions = {pad: np.mean(np.vstack(normals[pad]), axis=0) for pad in PADS}
    directions = {pad: value / np.linalg.norm(value) for pad, value in directions.items()}
    return {
        "robot_table_contact_count": robot_table_contacts,
        "pad_contact_centers_world_m": {pad: centers[pad].tolist() for pad in PADS},
        "pad_contact_normals_world": {pad: directions[pad].tolist() for pad in PADS},
        "height_delta_mm": abs(float(centers[PADS[0]][2] - centers[PADS[1]][2])) * 1000.0,
        "normal_dot": float(np.dot(directions[PADS[0]], directions[PADS[1]])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--command-percent", type=int, default=42)
    parser.add_argument("--x-limit-mm", type=float, default=6.0)
    parser.add_argument("--y-limit-mm", type=float, default=6.0)
    parser.add_argument("--z-limit-mm", type=float, default=20.0)
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--transport-candidate-count", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.command_percent <= 100:
        raise SystemExit("command percent must be in 0..100")
    if min(args.x_limit_mm, args.y_limit_mm, args.z_limit_mm, args.step_mm) <= 0:
        raise SystemExit("search limits and step must be positive")
    if not args.overlay_static_result.is_file() or not args.prior_config.is_file():
        raise SystemExit("overlay static result or prior config is missing")

    static = json.loads(args.overlay_static_result.read_text(encoding="utf-8"))
    if static["gates"].get("OVERLAY_COMPILED") != "PASS":
        raise SystemExit("CoACD overlay was not statically compiled")
    scene = Path(static["generated_scene"]["path"])
    if not scene.is_file() or sha256(scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("overlay scene is missing or changed")
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    pre = load_preflight_module()
    model = mujoco.MjModel.from_xml_path(str(scene))
    trajectory = load_registered_trajectory(
        load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO).package_path
    )
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    arm_start = np.asarray(
        [static["aligned_contact_static"]["arm_q_rad"][name] for name in pre.ARM_JOINTS],
        dtype=float,
    )
    values = lambda limit: np.arange(-limit, limit + 0.5 * args.step_mm, args.step_mm)
    raw_candidates = []
    for dx in values(args.x_limit_mm):
        for dy in values(args.y_limit_mm):
            for dz in values(args.z_limit_mm):
                translation = np.asarray([dx, dy, dz], dtype=float) / 1000.0
                item = pre.solve_gripper_translation_ik(
                    model,
                    trajectory,
                    state,
                    args.command_percent,
                    translation,
                    cylinder_lift_m=0.0,
                    start_arm_override=arm_start,
                )
                arm = np.asarray([item["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
                geometry = contact_geometry(model, trajectory, pre, state, arm, args.command_percent)
                raw_candidates.append(
                    {
                        "translation_world_mm": [float(dx), float(dy), float(dz)],
                        "arm_q_rad": item["arm_q_rad"],
                        "contact": item,
                        "geometry": geometry,
                    }
                )
    eligible = [
        item
        for item in raw_candidates
        if valid_contact(item["contact"])
        and item["contact"]["table_contact_count"] == 1
        and item["geometry"]["robot_table_contact_count"] == 0
        and item["geometry"]["height_delta_mm"] is not None
        and item["geometry"]["normal_dot"] is not None
        and item["geometry"]["normal_dot"] <= -0.8
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            float(item["geometry"]["height_delta_mm"]),
            -min(abs(float(contact["distance_m"])) for contact in item["contact"]["pad_contacts"]),
        ),
    )[: args.transport_candidate_count]
    transport_candidates = []
    for item in ranked:
        arm = np.asarray([item["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
        translation = np.asarray(item["translation_world_mm"], dtype=float) / 1000.0
        samples = []
        for fraction in np.linspace(0.0, 1.0, 5):
            lift_m = float(fraction * 0.05)
            sample = pre.solve_gripper_translation_ik(
                model,
                trajectory,
                state,
                args.command_percent,
                translation + np.asarray([0.0, 0.0, lift_m]),
                cylinder_lift_m=lift_m,
                start_arm_override=arm,
            )
            sample["fraction"] = float(fraction)
            samples.append(sample)
        path_pass = all(
            valid_contact(sample)
            and sample["gripper_base_position_error_m"] <= 1e-4
            and sample["gripper_base_orientation_error_rad"] <= 1e-3
            and (sample["fraction"] == 0.0 or sample["table_contact_count"] == 0)
            for sample in samples
        )
        transport_candidates.append(item | {"transport_samples": samples, "transport_path_pass": path_pass})
    viable = [item for item in transport_candidates if item["transport_path_pass"]]
    best = min(
        viable,
        key=lambda item: float(item["geometry"]["height_delta_mm"]),
        default=None,
    )
    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="stl-coacd-contact-pose-search"
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl-coacd-contact-pose-search.v1",
        "classification": "SIMULATION_ONLY_STATIC_SOURCE_STL_CONTACT_SEARCH",
        "mj_step_calls": 0,
        "overlay_static_result": {
            "path": str(args.overlay_static_result),
            "sha256": sha256(args.overlay_static_result),
        },
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "command_percent": args.command_percent,
        "search_range_mm": {
            "x": args.x_limit_mm,
            "y": args.y_limit_mm,
            "z": args.z_limit_mm,
            "step": args.step_mm,
        },
        "candidate_count": len(raw_candidates),
        "eligible_contact_candidates": eligible,
        "transport_candidates": transport_candidates,
        "best_candidate": best,
        "gates": {
            "COACD_OVERLAY_INTEGRITY": "PASS",
            "OPPOSING_PAD_CONTACT_STATIC": "PASS" if eligible else "BLOCKED",
            "LEVEL_PAD_CONTACT_STATIC": "PASS" if best else "BLOCKED",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS" if best else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The local pose search does not alter the source STL hand or collision model.",
            "A static contact configuration remains insufficient to claim dynamic force closure or a lift.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "candidate_count": len(raw_candidates),
                "eligible_count": len(eligible),
                "transport_viable_count": len(viable),
                "best_translation_world_mm": None if best is None else best["translation_world_mm"],
                "best_height_delta_mm": None if best is None else best["geometry"]["height_delta_mm"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
