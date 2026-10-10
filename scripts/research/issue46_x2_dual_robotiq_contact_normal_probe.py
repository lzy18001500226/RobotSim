#!/usr/bin/env python3
"""Replay only recorded first-contact steps to capture MuJoCo contact normals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def name(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def contacts(model, data) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        c = data.contact[index]
        g1, g2 = int(c.geom1), int(c.geom2)
        wrench = np.zeros(6)
        if int(c.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, index, wrench)
        rows.append({
            "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
            "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
            "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "distance_m": float(c.dist),
            "contact_normal_world": np.asarray(c.frame[:3], dtype=float).tolist(),
            "position_world_m": np.asarray(c.pos, dtype=float).tolist(),
            "contact_force_frame_n": wrench.tolist(),
        })
    return rows


def read_trace(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in
            (json.loads(line) for line in path.open(encoding="utf-8"))}


def pair_key(contact: dict[str, Any]) -> tuple[str, str, str, str]:
    return (contact["body1"], contact["geom1"], contact["body2"], contact["geom2"])


def replay(model_path: Path, reference_trace: Path, sample_steps: list[int], runner, helper,
           platform) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, _ = runner.prepare_dual_data(model, platform, helper)
    refs = helper.controller_config(model)
    targets = {item["joint"]: float(data.qpos[item["qpos_id"]]) for item in refs}
    target_vel = {joint: 0.0 for joint in targets}
    actuator = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    "rq_right_fingers_actuator"))
    expected = read_trace(reference_trace)
    by_step = {}
    step = 0
    wanted = set(sample_steps)
    for phase, count, final_target in runner.PHASES:
        for local in range(count):
            blend, _ = helper.quintic((local + 1) / count)
            if phase == "close":
                right_target = final_target * blend
            elif phase == "reopen":
                right_target = 255.0 * (1.0 - blend)
            else:
                right_target = final_target
            helper.apply_controller(model, data, refs, targets, target_vel)
            data.ctrl[actuator] = right_target
            mujoco.mj_step(model, data)
            step += 1
            if step not in wanted:
                continue
            actual_contacts = contacts(model, data)
            reference_row = expected[step]
            reference_contacts = reference_row["contacts"]
            actual_by_key = {pair_key(item): item for item in actual_contacts}
            comparisons = []
            for ref in reference_contacts:
                if not (ref["body1"].startswith("rq_") or ref["body2"].startswith("rq_")):
                    continue
                key = pair_key(ref)
                actual = actual_by_key.get(key)
                if actual is None:
                    continue
                distance_delta = actual["distance_m"] - float(ref["distance_m"])
                position_delta = float(np.linalg.norm(
                    np.asarray(actual["position_world_m"]) - np.asarray(ref["contact_position_m"])))
                comparisons.append({
                    "body_pair": [actual["body1"], actual["body2"]],
                    "geom_pair": [actual["geom1"], actual["geom2"]],
                    "distance_replay_m": actual["distance_m"],
                    "distance_trace_m": float(ref["distance_m"]),
                    "distance_delta_m": distance_delta,
                    "position_delta_m": position_delta,
                    "match_within_1um_distance_and_10um_position": (
                        abs(distance_delta) <= 1e-6 and position_delta <= 1e-5),
                    "contact_normal_world": actual["contact_normal_world"],
                    "contact_force_frame_n": actual["contact_force_frame_n"],
                })
            by_step[step] = {
                "time_s": float(data.time),
                "requested_right_control": right_target,
                "pad_gap_m": platform.mounted_pad_gap(model, data, "right"),
                "replay_contacts": actual_contacts,
                "trace_contact_match": comparisons,
            }
    missing = sorted(wanted - set(by_step))
    if missing:
        raise RuntimeError(f"Requested sample steps were not reached: {missing}")
    return {
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "reference_trace": str(reference_trace),
        "reference_trace_sha256": sha256(reference_trace),
        "replayed_steps": max(sample_steps),
        "sampled_steps": by_step,
        "all_contact_replay_matches_pass": all(
            comparison["match_within_1um_distance_and_10um_position"]
            for sample in by_step.values() for comparison in sample["trace_contact_match"]),
        "active_rollout_qpos_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-a-xml", required=True, type=Path)
    parser.add_argument("--candidate-a-trace", required=True, type=Path)
    parser.add_argument("--candidate-a-steps", nargs="+", type=int, default=[583, 584])
    parser.add_argument("--candidate-b-xml", required=True, type=Path)
    parser.add_argument("--candidate-b-trace", required=True, type=Path)
    parser.add_argument("--candidate-b-steps", nargs="+", type=int, default=[277, 292, 316, 318, 320])
    parser.add_argument("--platform-module", required=True, type=Path)
    parser.add_argument("--cycle-runner", required=True, type=Path)
    parser.add_argument("--controller-helper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    helper = load_module(args.controller_helper.resolve(), "issue46_x2_robotiq_m0")
    platform = load_module(args.platform_module.resolve(), "issue46_x2_dual_robotiq_platform")
    runner = load_module(args.cycle_runner.resolve(), "issue46_x2_robotiq_coupler_recovery")
    result = {
        "classification": "short deterministic replay for missing contact-frame normals; same control/model settings",
        "source_mount": replay(args.candidate_a_xml.resolve(), args.candidate_a_trace.resolve(),
                                args.candidate_a_steps, runner, helper, platform),
        "local_y_halfturn": replay(args.candidate_b_xml.resolve(), args.candidate_b_trace.resolve(),
                                   args.candidate_b_steps, runner, helper, platform),
        "python": sys.version,
        "mujoco": mujoco.__version__,
    }
    result["all_replayed_contacts_match_saved_trace"] = all(
        result[key]["all_contact_replay_matches_pass"]
        for key in ("source_mount", "local_y_halfturn"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_replayed_contacts_match_saved_trace": result["all_replayed_contacts_match_saved_trace"],
        "A_matches": result["source_mount"]["all_contact_replay_matches_pass"],
        "B_matches": result["local_y_halfturn"]["all_contact_replay_matches_pass"],
    }, sort_keys=True))
    return 0 if result["all_replayed_contacts_match_saved_trace"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
