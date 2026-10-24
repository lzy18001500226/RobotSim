#!/usr/bin/env python3
"""Build a candidate-specific static C3 wrench audit from fresh MuJoCo contacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path("/home/lzy18001500226/robotsim-issue43-g1-physical-grasp-20261009")
EVIDENCE = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-dfq-long-grasp-recovery")
C3_DIR = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-physical-grasp-priority-p0-20261009/c3-world-pitch15-preshape-final-20261010")
OUT = C3_DIR / "wrench-audit-contact-surface-fix"
CANDIDATE = C3_DIR / "candidate_c3.json"
MODEL = C3_DIR / "compiled/g1_dfq_c3_pitch15_static_mj336.xml"
BASE = REPO / "simulation/mujoco/fixtures/g1_dfq_m0_c2/canonical_ready_wrench_input.json"
UPSTREAM = Path("/home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/upstream")
URDF = UPSTREAM / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
LEGACY_IMPL = EVIDENCE / "implementation"
LEGACY_BUILDER = LEGACY_IMPL / "build_c2_wrench_input.py"
LEGACY_SOLVER = LEGACY_IMPL / "solve_c2_robust_wrench.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for path in (CANDIDATE, MODEL, BASE, URDF, LEGACY_BUILDER, LEGACY_SOLVER):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip())
    candidate_id = str(candidate["run_id"])

    sys.path.insert(0, str(LEGACY_IMPL))
    builder = load_module("robotsim_c3_wrench_input_builder", LEGACY_BUILDER)
    builder.RUN_ID = candidate_id
    builder.RUN = OUT
    builder.MODEL_PATH = MODEL
    builder.CANDIDATE_PATH = CANDIDATE
    builder.BASE_PATH = BASE
    builder.IMPLEMENTATION = LEGACY_IMPL
    builder.runner.URDF = URDF
    builder.hand.URDF = URDF
    original_contact_record = builder.contact_record
    bottle_geom_names = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}

    def c3_contact_record(model, data, joints, body_id, bottle_com, bottle_geom, hand_geom, digit, label):
        active_pairs = []
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            for name in bottle_geom_names:
                bottle_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
                if bottle_id >= 0 and pair == {bottle_id, hand_geom}:
                    active_pairs.append((float(contact.dist), bottle_id))
        if active_pairs:
            selected_bottle_geom = min(active_pairs)[1]
        else:
            distances = []
            for name in bottle_geom_names:
                bottle_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
                if bottle_id < 0:
                    continue
                distance = float(mujoco.mj_geomDistance(
                    model, data, bottle_id, hand_geom, 0.25, np.zeros(6, dtype=float)
                ))
                distances.append((distance, bottle_id))
            if not distances:
                raise RuntimeError("candidate bottle geoms are absent from compiled C3 model")
            selected_bottle_geom = min(distances)[1]
        return original_contact_record(
            model, data, joints, body_id, bottle_com, selected_bottle_geom,
            hand_geom, digit, label
        )

    builder.contact_record = c3_contact_record
    builder.main()

    generated = OUT / "candidate_c2_wrench_input.json"
    if not generated.is_file():
        raise RuntimeError("wrench input builder produced no output")
    wrench_input_path = OUT / "candidate_c3_wrench_input.json"
    if wrench_input_path.exists():
        raise FileExistsError(wrench_input_path)
    wrench_input = json.loads(generated.read_text(encoding="utf-8"))
    wrench_input.update({
        "analysis": "fresh C3 static contact/wrench input from the exact candidate MuJoCo geometry; no table support in wrench solve",
        "candidate": candidate["candidate"],
        "candidate_static_run_id": candidate_id,
        "candidate_run_id": candidate_id,
        "run_id": candidate_id,
        "robotsim_sha": head,
        "robotsim_dirty": dirty,
        "robotsim_worktree": str(REPO),
        "candidate_json_path": str(CANDIDATE),
        "candidate_json_sha256": sha256(CANDIDATE),
        "compiled_model_sha256": sha256(MODEL),
        "normalized_compiled_model_sha256": candidate["source"]["compiled_model_sha256"],
        "candidate_static_gates": candidate["static_gates"],
        "contact_geometry_selection": "for each controlled digit, prefer its deepest exact active pair among bottle body/shoulder/neck/cap; otherwise use the nearest bottle surface",
        "upstream_shas": {
            "unitree_ros": "5994d4faef0a9cadd3287f8de0199a67eeb2a259",
            "unitree_mujoco": "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d",
            "humanoid_vla": "3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12",
        },
    })
    generated.replace(wrench_input_path)
    wrench_input_path.write_text(json.dumps(wrench_input, indent=2), encoding="utf-8")

    solver = load_module("robotsim_c3_robust_wrench_solver", LEGACY_SOLVER)
    solver.RUN_DIR = OUT
    solver.INPUT = wrench_input_path
    solver.OUTPUT = OUT / "candidate_c3_robust_wrench.json"
    solver.main()
    wrench_path = solver.OUTPUT
    wrench = json.loads(wrench_path.read_text(encoding="utf-8"))
    wrench.update({
        "analysis": "fresh C3 quasi-static robust wrench feasibility from candidate-specific MuJoCo contacts; this is not physical grasp proof",
        "candidate": candidate["candidate"],
        "candidate_static_run_id": candidate_id,
        "candidate_wrench_input_sha256": sha256(wrench_input_path),
        "robotsim_sha": head,
        "robotsim_dirty": dirty,
        "compiled_model_sha256": sha256(MODEL),
    })
    wrench_path.write_text(json.dumps(wrench, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate": candidate["candidate"],
        "candidate_static_run_id": candidate_id,
        "wrench_input": str(wrench_input_path),
        "wrench_input_sha256": sha256(wrench_input_path),
        "wrench_result": str(wrench_path),
        "wrench_result_sha256": sha256(wrench_path),
        "classification": wrench.get("classification"),
        "physics_steps": wrench.get("physics_steps"),
        "mj_step_calls": wrench.get("mj_step_calls"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
