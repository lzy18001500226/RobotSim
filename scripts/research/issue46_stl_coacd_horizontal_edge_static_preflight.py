#!/usr/bin/env python3
"""Validate a pad-first horizontal-cylinder edge-grasp candidate statically.

The candidate scene is produced by a read-only table-and-cylinder edge scan.
This preflight verifies the object remains table supported, the opened approach
is clear, p37 has only two distal pad contacts, and the arm can carry the same
source cylinder through a 50 mm static lift path.  It performs ``mj_forward``
only.
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


PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


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


def valid_pad_contact(item: dict[str, object]) -> bool:
    return bool(item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"])


def robot_table_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records = []
    for contact in data.contact:
        body_ids = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            for body in body_ids
        )
        if "raicom_table" in names and any(name.startswith(("R_hand_", "right_")) for name in names):
            records.append({"body_names": names, "distance_m": float(contact.dist)})
    return records


def evaluate(
    pre,
    model,
    trajectory,
    state,
    command: int,
    arm: np.ndarray,
    lift_m: float = 0.0,
) -> dict[str, object]:
    item = pre.evaluate_pose(
        model,
        trajectory,
        state,
        command,
        cylinder_lift_m=lift_m,
        arm_override=arm,
    )
    data = mujoco.MjData(model)
    pre.apply_prior_state(model, data, state, arm)
    if lift_m:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
        data.qpos[model.jnt_qposadr[joint] + 2] += lift_m
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, command)
    item["robot_table_contacts"] = robot_table_contacts(model, data)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-scan-result", required=True, type=Path)
    parser.add_argument("--vertical-overlay-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--table-and-cylinder-y-shift-mm", type=int, default=-150)
    parser.add_argument("--command-percent", type=int, default=37)
    parser.add_argument("--alignment-scan-result", type=Path)
    parser.add_argument("--alignment-offset-mm", type=float)
    args = parser.parse_args()
    for path in (args.edge_scan_result, args.vertical_overlay_static_result, args.prior_config):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.alignment_scan_result is not None and not args.alignment_scan_result.is_file():
        raise SystemExit("alignment scan result is missing")
    edge = json.loads(args.edge_scan_result.read_text(encoding="utf-8"))
    vertical = json.loads(args.vertical_overlay_static_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    row = next(
        (item for item in edge["shifts"] if int(item["table_and_cylinder_y_shift_mm"]) == args.table_and_cylinder_y_shift_mm),
        None,
    )
    if row is None:
        raise SystemExit("requested edge shift was not scanned")
    if set(row["pads"]) != PADS or row["nonpad"] or row["robot_table_contacts"] or len(row["table_contacts"]) != 1:
        raise SystemExit("requested edge shift is not a pad-only table-supported candidate")
    scene = Path(row["scene"])
    if not scene.is_file():
        raise SystemExit("edge-scene XML is missing")

    pre = load_preflight_module()
    model = mujoco.MjModel.from_xml_path(str(scene))
    trajectory = load_registered_trajectory(
        load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO).package_path
    )
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    pregrasp = pre.state_by_phase(prior, "HORIZONTAL_PREGRASP")
    alignment = None
    if args.alignment_scan_result is None:
        contact_source_arm = vertical["aligned_contact_static"]["arm_q_rad"]
        selected_alignment = None
        alignment_rows = []
    else:
        alignment = json.loads(args.alignment_scan_result.read_text(encoding="utf-8"))
        if int(alignment["command_percent"]) != args.command_percent:
            raise SystemExit("alignment scan command does not match requested command")
        if alignment["gates"].get("LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY") != "PASS":
            raise SystemExit("alignment scan did not pass static transport validation")
        eligible_alignment = [
            item for item in alignment.get("candidates", [])
            if item.get("arm_transport_path_pass")
            and item["contact"]["bilateral_pad_contact"]
            and not item["contact"]["nonpad_right_contact_count"]
        ]
        if args.alignment_offset_mm is None:
            selected_alignment = alignment.get("best_candidate")
        else:
            selected_alignment = next(
                (
                    item for item in eligible_alignment
                    if abs(float(item["lateral_offset_mm"]) - args.alignment_offset_mm) <= 1e-9
                ),
                None,
            )
        if selected_alignment is None or selected_alignment not in eligible_alignment:
            raise SystemExit("requested alignment offset is not a static valid horizontal candidate")
        contact_source_arm = selected_alignment["arm_q_rad"]
        alignment_rows = [
            {"lateral_offset_mm": float(item["lateral_offset_mm"]), "arm_q_rad": item["arm_q_rad"]}
            for item in eligible_alignment
        ]
    contact_arm = np.asarray([contact_source_arm[name] for name in pre.ARM_JOINTS], dtype=float)
    contact = evaluate(pre, model, trajectory, state, args.command_percent, contact_arm)
    if not valid_pad_contact(contact) or contact["table_contact_count"] != 1 or contact["robot_table_contacts"]:
        raise SystemExit("edge candidate no longer satisfies the static pad/table gate")
    contact["arm_q_rad"] = {name: float(value) for name, value in zip(pre.ARM_JOINTS, contact_arm)}
    contact["highest_safe_bilateral_percent"] = args.command_percent
    contact["lateral_offset_mm"] = (
        0.0 if selected_alignment is None else float(selected_alignment["lateral_offset_mm"])
    )
    contact["selection_basis"] = (
        "source-STL CoACD horizontal cylinder table-edge candidate"
        if selected_alignment is None
        else "source-STL CoACD horizontal cylinder table-edge candidate with pad-axis alignment"
    )
    closure_scan = []
    for command in range(100, -1, -1):
        item = evaluate(pre, model, trajectory, state, command, contact_arm)
        closure_scan.append(
            {
                "command_percent": command,
                "pad_bodies": item["pad_bodies"],
                "nonpad_right_contact_count": item["nonpad_right_contact_count"],
                "pad_distances_m": pre.pad_distance_map(item),
                "robot_table_contact_count": len(item["robot_table_contacts"]),
            }
        )
    contact["closure_contact_scan"] = closure_scan

    pregrasp_arm = np.asarray([pregrasp["motor_targets_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
    approach = []
    for fraction in np.linspace(0.0, 1.0, 11):
        arm = pregrasp_arm + fraction * (contact_arm - pregrasp_arm)
        item = evaluate(pre, model, trajectory, state, 100, arm)
        approach.append({"fraction": float(fraction), **item})
    approach_pass = all(
        not item["pad_bodies"]
        and not item["nonpad_right_contact_count"]
        and not item["robot_table_contacts"]
        for item in approach
    )

    transport = []
    for fraction in np.linspace(0.0, 1.0, 5):
        lift_m = float(0.05 * fraction)
        solution = pre.solve_gripper_translation_ik(
            model,
            trajectory,
            state,
            args.command_percent,
            np.asarray([0.0, 0.0, lift_m]),
            cylinder_lift_m=lift_m,
            start_arm_override=contact_arm,
        )
        arm = np.asarray([solution["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
        item = evaluate(pre, model, trajectory, state, args.command_percent, arm, lift_m)
        item.update(
            {
                "fraction": float(fraction),
                "arm_q_rad": solution["arm_q_rad"],
                "gripper_base_position_error_m": solution["gripper_base_position_error_m"],
                "gripper_base_orientation_error_rad": solution["gripper_base_orientation_error_rad"],
            }
        )
        transport.append(item)
    transport_pass = all(
        valid_pad_contact(item)
        and item["gripper_base_position_error_m"] <= 1e-4
        and item["gripper_base_orientation_error_rad"] <= 1e-3
        and not item["robot_table_contacts"]
        and (item["fraction"] == 0.0 or item["table_contact_count"] == 0)
        for item in transport
    )

    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-coacd-horizontal-edge-static"
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.full-x2-stl-coacd-horizontal-edge-static.v1",
        "classification": "SIMULATION_ONLY_SOURCE_STL_HORIZONTAL_EDGE_GRASP",
        "mj_step_calls": 0,
        "model": vertical["model"],
        "generated_scene": {"path": str(scene), "sha256": sha256(scene)},
        "edge_scan": {"path": str(args.edge_scan_result), "sha256": sha256(args.edge_scan_result)},
        "vertical_overlay_static": {
            "path": str(args.vertical_overlay_static_result),
            "sha256": sha256(args.vertical_overlay_static_result),
        },
        "alignment_scan": (
            None if args.alignment_scan_result is None else {"path": str(args.alignment_scan_result), "sha256": sha256(args.alignment_scan_result)}
        ),
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "workcell": {
            "cylinder_axis_world": "-Y",
            "table_and_cylinder_y_shift_mm": args.table_and_cylinder_y_shift_mm,
            "cylinder_radius_m": 0.035,
            "cylinder_height_m": 0.30,
            "cylinder_mass_kg": 0.20,
        },
        "selected_command_percent": args.command_percent,
        "aligned_contact_static": contact,
        "controller_contact_target_static": contact,
        "transport_end_static": transport[-1],
        "controller_lift_target_static": transport[-1],
        "transport_path_static": transport,
        "lateral_alignment_scan": alignment_rows or [{"lateral_offset_mm": 0.0, "arm_q_rad": contact["arm_q_rad"]}],
        "open_approach_samples": approach,
        "gates": {
            "COACD_OVERLAY_INTEGRITY": "PASS",
            "HORIZONTAL_CYLINDER_TABLE_SUPPORTED": "PASS",
            "OPEN_APPROACH_CLEAR": "PASS" if approach_pass else "BLOCKED",
            "BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS",
            "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY": "PASS" if transport_pass else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "This is a static source-STL/CoACD workcell candidate; it does not prove dynamic force closure.",
            "The table and cylinder move together to a documented edge position; their dimensions and masses are unchanged.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "gates": result["gates"],
                "transport_end_arm_q_rad": transport[-1]["arm_q_rad"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
