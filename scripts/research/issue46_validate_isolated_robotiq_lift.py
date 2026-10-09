#!/usr/bin/env python3
"""Independently validate preserved isolated Robotiq lift evidence offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "result.json"
    trace_path = run_dir / "raw" / "isolated_physics_trace.jsonl"
    model_path = run_dir / "model" / "isolated_robotiq_bottle.xml"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    held = [row for row in rows if row["phase"] == "HOLD_50MM"]
    if not held:
        raise RuntimeError("No HOLD_50MM samples in preserved trace")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    if bottle_joint < 0:
        bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    if bottle_joint < 0:
        raise RuntimeError("Compiled evidence model has no bottle free joint")
    qadr = int(model.jnt_qposadr[bottle_joint])
    bottle_geoms = [i for i in range(model.ngeom)
                    if object_name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("bottle_")]
    table_geoms = [i for i in range(model.ngeom)
                   if object_name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("m0_table")]
    if not bottle_geoms or not table_geoms:
        raise RuntimeError("Compiled model is missing bottle or canonical table collision geoms")

    distances = []
    witness = np.zeros(6, dtype=float)
    for row in held:
        obj = row["object"]
        data.qpos[qadr:qadr + 3] = obj["freejoint_position_world_m"]
        data.qpos[qadr + 3:qadr + 7] = obj["freejoint_quaternion_wxyz"]
        mujoco.mj_forward(model, data)
        for bottle_geom in bottle_geoms:
            for table_geom in table_geoms:
                distance = float(mujoco.mj_geomDistance(
                    model, data, bottle_geom, table_geom, 1.0, witness))
                distances.append({
                    "step": row["step"],
                    "bottle_geom": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom),
                    "table_geom": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom),
                    "distance_m": distance,
                })

    max_lift = max(float(row["object_lift_from_settled_com_m"]) for row in rows)
    max_hold_slip_xy = max(
        float(np.linalg.norm(np.asarray(row["object"]["center_of_mass_world_m"][:2]) -
                             np.asarray(held[0]["object"]["center_of_mass_world_m"][:2])))
        for row in held
    )
    hold_table_contacts = sum(bool(row["table_bottle_contact"]) for row in held)
    hold_bilateral = sum(bool(row["bilateral_pad_contact"]) for row in held)
    hold_non_gripping = sum(bool(row["non_gripping_bottle_contact"]) for row in held)
    hold_writes = sum(int(row["qpos_writes_after_rollout_start"]) for row in held)
    hold_limit_violations = sum(len(row["source_joint_limit_violations"]) for row in held)
    min_table_gap = min(item["distance_m"] for item in distances)
    maximum_pad_force = max(
        float(row["right_pad_normal_force_sum_n"] + row["left_pad_normal_force_sum_n"])
        for row in held
    )
    checks = {
        "preserved_result_reports_50mm_or_more": max_lift >= 0.05,
        "one_second_held_sample_count": len(held) >= 1000,
        "bilateral_pad_contact_every_hold_sample": hold_bilateral == len(held),
        "zero_table_bottle_contact_during_hold": hold_table_contacts == 0,
        "zero_non_gripping_contact_during_hold": hold_non_gripping == 0,
        "zero_active_rollout_qpos_writes_during_hold": hold_writes == 0,
        "zero_joint_limit_violations_during_hold": hold_limit_violations == 0,
        "compiled_geometry_table_clearance_nonnegative": min_table_gap >= 0.0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "ISOLATED_SIMULATION_ONLY; not source-faithful acceptance",
        "result_status": result.get("status"),
        "model_mode": result.get("fixture_identity", {}).get("constraint_model_classification"),
        "source_joint_ranges": result.get("fixture_identity", {}).get("coupler_joint_ranges_rad"),
        "simulation_derived_coupler_margin_rad": result.get("fixture_identity", {}).get(
            "coupler_limit_activation_margin_simulation_derived_rad"),
        "runtime": result.get("runtime_identity"),
        "maximum_bottle_com_lift_m": max_lift,
        "hold_samples": len(held),
        "hold_duration_s": max(float(row["time_s"]) for row in held) -
                            min(float(row["time_s"]) for row in held),
        "hold_bilateral_samples": hold_bilateral,
        "hold_table_contact_samples": hold_table_contacts,
        "hold_non_gripping_contact_samples": hold_non_gripping,
        "hold_post_start_qpos_write_count": hold_writes,
        "hold_joint_limit_violation_count": hold_limit_violations,
        "hold_xy_slip_m": max_hold_slip_xy,
        "hold_max_total_pad_normal_force_n": maximum_pad_force,
        "compiled_bottle_to_table_min_distance_m": min_table_gap,
        "compiled_bottle_to_table_pair_count": len(distances),
        "checks": checks,
        "sha256": {
            "result_json": sha256(result_path),
            "physics_trace": sha256(trace_path),
            "contact_trace": sha256(run_dir / "raw" / "isolated_contact_trace.jsonl"),
            "compiled_model": sha256(model_path),
            "video": sha256(run_dir / "isolated_grasp_lift_release.mp4"),
        },
        "evidence_files_present": {
            name: (run_dir / name).is_file()
            for name in ("result.json", "run.log", "isolated_grasp_lift_release.mp4",
                         "static_overview.png", "static_side.png", "static_closeup.png",
                         "raw/isolated_contact_trace.jsonl", "raw/isolated_physics_trace.jsonl",
                         "model/isolated_robotiq_bottle.xml")
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
