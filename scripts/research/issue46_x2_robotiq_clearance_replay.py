#!/usr/bin/env python3
"""Compute exact compiled collision-geom distances at saved IK states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0
import issue46_x2_robotiq_mounted_smoke as smoke
import issue46_x2_robotiq_reachability as reach


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args()
    candidate_dir = args.candidate_dir.resolve()
    trace_path = candidate_dir / "raw" / "candidate_trace.json"
    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    model_path = Path(raw["model"]["experimental_xml"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    smoke.set_mounted_neutral(model, data)
    qids = [m0.qpos_id(model, joint) for joint in m0.ARM]
    robot_bodies = smoke.robot_body_names(model)
    states = {}
    position_seed = raw.get("position_only_fk_seed")
    if position_seed and position_seed.get("joint_pose_rad"):
        states["position_only_fk_seed"] = position_seed["joint_pose_rad"]
    ik = raw.get("ik", {})
    selected = ik.get("pregrasp")
    if selected and selected.get("joint_pose_rad"):
        states["selected_pregrasp"] = selected["joint_pose_rad"]
    for key in ("pregrasp_from_clear_neutral", "pregrasp_from_position_seed", "pregrasp_from_orientation_seed"):
        item = ik.get(key)
        if item and item.get("joint_pose_rad"):
            states[key] = item["joint_pose_rad"]
    out: dict[str, Any] = {
        "candidate": raw.get("configuration"),
        "model_path": str(model_path), "model_sha256": sha256(model_path),
        "trace_path": str(trace_path), "trace_sha256": sha256(trace_path),
        "distance_method": "mujoco.mj_geomDistance on compiled active robot geoms against every canonical table and bottle geom",
        "states": {},
    }
    for label, q in states.items():
        data.qpos[qids] = np.asarray(q, dtype=float)
        mujoco.mj_forward(model, data)
        distances = reach.exact_environment_distances(model, data, robot_bodies, limit=24)
        out["states"][label] = {
            "q_rad": list(map(float, q)),
            "active_contacts": reach.relevant_contacts(model, data, robot_bodies),
            "minimum_robot_bottle_signed_distance_m": distances["robot_bottle"]["minimum_signed_distance_m"],
            "nearest_robot_bottle_pairs": distances["robot_bottle"]["nearest_pairs"],
            "minimum_robot_table_signed_distance_m": distances["robot_table"]["minimum_signed_distance_m"],
            "nearest_robot_table_pairs": distances["robot_table"]["nearest_pairs"],
            "minimum_pad_bottle_signed_distance_m": distances["pad_to_bottle"]["minimum_signed_distance_m"],
            "pad_bottle_pairs": distances["pad_to_bottle"]["all_pairs"],
        }
    output_path = candidate_dir / "raw" / "candidate_clearances.json"
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "states": {
        key: {"min_robot_bottle_m": value["minimum_robot_bottle_signed_distance_m"],
              "min_robot_table_m": value["minimum_robot_table_signed_distance_m"],
              "min_pad_bottle_m": value["minimum_pad_bottle_signed_distance_m"]}
        for key, value in out["states"].items()}}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
