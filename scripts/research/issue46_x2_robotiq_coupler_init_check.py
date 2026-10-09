#!/usr/bin/env python3
"""One bounded interior-startup check for the mounted Robotiq coupler joints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0
import issue46_x2_robotiq_mounted_smoke as smoke


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-reachable-contact-20261009/coupler-startup-correction"
)
COUPLER_JOINTS = ("rq_right_coupler_joint", "rq_left_coupler_joint")
INTERIOR_OFFSET_RAD = 0.001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def eq_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.nefc):
        if int(data.efc_type[i]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        rows.append({
            "row": i,
            "position_residual": float(data.efc_pos[i]),
            "force": float(data.efc_force[i]),
            "constraint_id": int(data.efc_id[i]),
        })
    return rows


def joint_state(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> dict[str, Any]:
    jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return {
        "qpos_rad": float(data.qpos[qa]),
        "qpos0_rad": float(model.qpos0[qa]),
        "qvel_rad_s": float(data.qvel[da]),
        "qacc_rad_s2": float(data.qacc[da]),
        "range_rad": model.jnt_range[jid].astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=m0.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=smoke.DEFAULT_STATION)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    repo = script.parents[2]
    result: dict[str, Any] = {
        "experiment": "Single interior initial-qpos correction for mounted Robotiq; SIMULATION_ONLY",
        "status": "BLOCKED",
        "correction": {
            "joint_names": list(COUPLER_JOINTS),
            "initial_qpos_rad": -INTERIOR_OFFSET_RAD,
            "derived_from_max_prior_overshoot_rad": 0.000150054301,
            "offset_to_measured_excursion_ratio": INTERIOR_OFFSET_RAD / 0.000150054301,
            "actuator_open_control": 0.0,
            "source_ranges_or_model_constraints_changed": False,
            "post_start_qpos_writes": 0,
        },
        "runner": {
            "path": str(script), "sha256": sha256(script),
            "robot_sim_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "command": sys.argv,
        },
    }
    try:
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash differs from the accepted immutable helper")
        ident = m0.identity(args, args.canonical_helper)
        canonical = m0.load_module(args.canonical_helper)
        build_args = SimpleNamespace(
            x2_root=args.x2_root, menagerie_root=args.menagerie_root,
            station_base_pos=np.asarray(args.station_base_pos, dtype=float),
        )
        base_model, adapter = m0.build_model(build_args, out, canonical)
        base_xml = Path(adapter["model_xml"])
        variant_path = out / "x2_robotiq_2f85_head_torso_exclude_simulation_only.xml"
        exclusion = smoke.create_exclusion_variant(base_xml, variant_path)
        model = mujoco.MjModel.from_xml_path(str(variant_path))
        data = mujoco.MjData(model)
        smoke.set_mounted_neutral(model, data)
        for joint in COUPLER_JOINTS:
            data.qpos[m0.qpos_id(model, joint)] = -INTERIOR_OFFSET_RAD
        grip_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, smoke.GRIP_ACTUATOR)
        data.ctrl[grip_id] = 0.0
        mujoco.mj_forward(model, data)
        pre = {
            "time_s": float(data.time),
            "coupler_joints": {j: joint_state(model, data, j) for j in COUPLER_JOINTS},
            "equality_rows": eq_rows(model, data),
            "max_abs_equality_position_residual": max((abs(x["position_residual"]) for x in eq_rows(model, data)), default=0.0),
            "contacts": smoke.describe_contacts(model, data, smoke.robot_body_names(model)),
            "source_joint_limit_violations": m0.all_limited_joint_checks(model, data,
                {k: v["velocity"] for k, v in ident["urdf_joint_limits"].items()
                 if math.isfinite(v["velocity"]) and v["velocity"] > 0})[0],
            "actuator_control": float(data.ctrl[grip_id]),
            "actuator_force_before_first_step": float(data.actuator_force[grip_id]),
            "right_pad_separation_m": smoke.pad_gap(model, data),
            "model": {"path": str(variant_path), "sha256": sha256(variant_path), "exclusion": exclusion},
        }
        write_json(out / "raw" / "startup_preflight.json", pre)
        if pre["source_joint_limit_violations"]:
            result["preflight"] = pre
            result["status"] = "FAIL_PRESTEP_LIMIT"
        else:
            # All state initialization ends before run_cycle; it performs no qpos writes.
            cycle = smoke.run_cycle(model, data, ident, out)
            result["preflight"] = pre
            result["mounted_cycle"] = cycle
            result["status"] = "PASS" if cycle["status"] == "PASS" else "FAIL_MOUNTED_CYCLE"
        result["identity"] = ident
        result["adapter"] = adapter
        result["evidence"] = {
            "result_json": str(out / "result.json"),
            "startup_preflight": str(out / "raw" / "startup_preflight.json"),
            "trace_jsonl": str(out / "raw" / "mounted_open_close_trace.jsonl"),
            "contacts_jsonl": str(out / "raw" / "mounted_open_close_contacts.jsonl"),
            "video": str(out / "mounted_open_close.mp4"),
        }
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result_json": str(out / "result.json"),
                      "failure": result.get("failure"), "evidence": result.get("evidence")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
