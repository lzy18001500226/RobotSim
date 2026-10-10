#!/usr/bin/env python3
"""Full robot/gripper contact audit for preserved Issue #46 cycle traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def external_contacts(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for contact in row.get("contacts", []):
        body1, body2 = contact["body1"], contact["body2"]
        if body1.startswith("rq_") == body2.startswith("rq_"):
            continue
        gripper = body1 if body1.startswith("rq_") else body2
        robot = body2 if body1.startswith("rq_") else body1
        if robot in {"world", "floor", "recovery_floor"}:
            continue
        result.append({**contact, "robot_body": robot, "gripper_body": gripper})
    return result


def summarize_trace(path: Path, narrow_status: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    first_row_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    wrist_events = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows.append(row)
        for contact in external_contacts(row):
            key = (contact["robot_body"], contact["gripper_body"])
            item = {**contact, "step": row["step"], "time_s": row["time_s"]}
            grouped[key].append(item)
            first_row_by_key.setdefault(key, row)
            if "wrist" in item["robot_body"]:
                wrist_events.append((item, row))
    pair_summary = []
    selected_contact = None
    selected_row = None
    for (robot, gripper), contacts in sorted(grouped.items()):
        worst = min(contacts, key=lambda item: item["distance_m"])
        first = min(contacts, key=lambda item: item["step"])
        pair_summary.append({
            "robot_body": robot,
            "gripper_body": gripper,
            "sample_count": len(contacts),
            "first_step": first["step"],
            "first_time_s": first["time_s"],
            "worst_penetration_m": max(0.0, -float(worst["distance_m"])),
            "worst_step": worst["step"],
            "worst_time_s": worst["time_s"],
            "peak_contact_normal_force_n": max(float(item["normal_force_n"]) for item in contacts),
            "worst_contact": worst,
        })
        if selected_contact is None or worst["distance_m"] < selected_contact["distance_m"]:
            selected_contact = worst
            selected_row = next(row for row in rows if row["step"] == worst["step"])
    return {
        "trace_path": str(path),
        "trace_sha256": sha256(path),
        "physics_steps": len(rows),
        "controller_runner_status_narrow_wrist_roll_guard": narrow_status,
        "external_robot_gripper_contact_sample_count": sum(map(len, grouped.values())),
        "external_robot_gripper_pairs": pair_summary,
        "no_external_robot_gripper_contact_gate": "PASS" if not grouped else "FAIL",
        "selected_worst_contact_frame": selected_contact,
        "selected_worst_contact_row": selected_row,
        "worst_wrist_contact_frame": min(wrist_events, key=lambda pair: pair[0]["distance_m"])[0]
        if wrist_events else None,
        "worst_wrist_contact_row": min(wrist_events, key=lambda pair: pair[0]["distance_m"])[1]
        if wrist_events else None,
    }


def render_row(row: dict[str, Any], xml_path: Path, out_path: Path,
               helper, runner, platform, label: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data, _ = runner.prepare_dual_data(model, platform, helper)
    for joint, value in row["qpos_rad"].items():
        data.qpos[helper.qpos_id(model, joint)] = value
    data.time = float(row["time_s"])
    mujoco.mj_forward(model, data)
    contact = external_contacts(row)[0]
    point = contact["contact_position_m"]
    png = out_path.with_suffix(".raw.png")
    helper.render(model, data, png, point, 0.36, 135, -5, size=(960, 640))
    image = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 70), fill=(20, 27, 36))
    draw.text((14, 12), label, fill=(255, 255, 255))
    draw.text((14, 38),
              f"t={row['time_s']:.3f}s | {contact['body1']} / {contact['body2']} | "
              f"gap={contact['distance_m'] * 1000:.3f} mm | "
              f"normal={contact['normal_force_n']:.2f} N",
              fill=(255, 220, 170))
    image.save(out_path)
    png.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--baseline-trace", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--baseline-xml", required=True, type=Path)
    parser.add_argument("--candidate-xml", required=True, type=Path)
    parser.add_argument("--platform-module", required=True, type=Path)
    parser.add_argument("--cycle-runner", required=True, type=Path)
    parser.add_argument("--controller-helper", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    helper = load_module(args.controller_helper.resolve(), "issue46_x2_robotiq_m0")
    platform = load_module(args.platform_module.resolve(), "issue46_x2_dual_robotiq_platform")
    runner = load_module(args.cycle_runner.resolve(), "issue46_x2_robotiq_coupler_recovery")

    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    candidate_result = json.loads(args.candidate_result.read_text(encoding="utf-8"))
    baseline_right = summarize_trace(args.baseline_trace.resolve(),
                                     baseline_result["stage_3_actuator_driven_cycles"]["right"]["status"])
    candidate_cycles = {}
    for side, record in candidate_result.get("dynamic_cycles", {}).items():
        trace = Path(record["trace_path"])
        candidate_cycles[side] = summarize_trace(trace, record["status"])

    images = {}
    if baseline_right["selected_worst_contact_row"]:
        path = out / "candidate_a_source_mount_right_worst_contact.png"
        render_row(baseline_right["selected_worst_contact_row"], args.baseline_xml.resolve(),
                   path, helper, runner, platform, "A | Pinned X2 mount | original diagnostic run")
        images["candidate_a_source_mount_right_worst_contact"] = str(path)
    right_cycle = candidate_cycles.get("right")
    if right_cycle and right_cycle["selected_worst_contact_row"]:
        path = out / "candidate_b_local_y_halfturn_right_worst_contact.png"
        render_row(right_cycle["selected_worst_contact_row"], args.candidate_xml.resolve(),
                   path, helper, runner, platform, "B | 180 deg local-Y half-turn | diagnostic candidate")
        images["candidate_b_local_y_halfturn_right_worst_contact"] = str(path)
    if right_cycle and right_cycle["worst_wrist_contact_row"]:
        row = right_cycle["worst_wrist_contact_row"]
        contact = right_cycle["worst_wrist_contact_frame"]
        row = {**row, "contacts": [contact]}
        path = out / "candidate_b_local_y_halfturn_right_wrist_contact.png"
        render_row(row, args.candidate_xml.resolve(), path, helper, runner, platform,
                   "B | 180 deg local-Y half-turn | wrist/gripper contact")
        images["candidate_b_local_y_halfturn_right_wrist_contact"] = str(path)
    if ("candidate_a_source_mount_right_worst_contact" in images
            and "candidate_b_local_y_halfturn_right_wrist_contact" in images):
        a = Image.open(images["candidate_a_source_mount_right_worst_contact"]).convert("RGB")
        b = Image.open(images["candidate_b_local_y_halfturn_right_wrist_contact"]).convert("RGB")
        paired = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (12, 16, 21))
        paired.paste(a, (0, 0))
        paired.paste(b, (a.width, 0))
        overlay = out / "right_mount_collision_before_after.png"
        paired.save(overlay)
        images["side_by_side"] = str(overlay)

    right_candidate_pass = bool(right_cycle) and (
        right_cycle["controller_runner_status_narrow_wrist_roll_guard"] == "PASS"
        and right_cycle["no_external_robot_gripper_contact_gate"] == "PASS")
    summary = {
        "classification": "SIMULATION_ONLY adapter collision audit; no physical hardware equivalence claimed",
        "scope": "Post-process preserved MuJoCo traces. No dynamics rerun and no raw evidence overwritten.",
        "baseline_candidate_a_source_mount": baseline_right,
        "candidate_b_local_y_180_halfturn_cycles": candidate_cycles,
        "candidate_b_right_mounted_motion_gate": "PASS" if right_candidate_pass else "FAIL",
        "candidate_b_left_and_both_mounted_motion_gate": {
            side: ("PASS" if cycle["controller_runner_status_narrow_wrist_roll_guard"] == "PASS"
                   and cycle["no_external_robot_gripper_contact_gate"] == "PASS" else "FAIL")
            for side, cycle in candidate_cycles.items() if side in ("left", "both")
        },
        "images": images,
        "model_hashes": {
            "candidate_a": sha256(args.baseline_xml.resolve()),
            "candidate_b": sha256(args.candidate_xml.resolve()),
        },
    }
    output = out / "full_robot_gripper_contact_audit.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(output),
        "candidate_a_right_contacts": baseline_right["external_robot_gripper_contact_sample_count"],
        "candidate_b_right_contacts": right_cycle["external_robot_gripper_contact_sample_count"] if right_cycle else None,
        "candidate_b_right_gate": summary["candidate_b_right_mounted_motion_gate"],
        "images": images,
    }, sort_keys=True))
    return 0 if right_candidate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
