#!/usr/bin/env python3
"""Bounded post-stop full-body hold with the reviewed STL hand overlay.

This experiment is explicitly ``SIMULATION_ONLY``.  It uses the recorded
STL-table follower trajectories as equality constraints plus one root position
servo per hand.  It maps the observed RAICOM post-stop state once before the
first ``mj_step`` and never writes hand/body qpos thereafter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402


FOLLOWER_COLUMNS = {
    "wide_root_q_rad": "wide1",
    "narrow3_q_rad": "narrow3",
    "narrow_loop_q_rad": "narrow_loop",
    "wide3_q_rad": "wide3",
    "wide_loop_q_rad": "wide_loop",
}
ROOTS = {"right": "right_claw_joint", "left": "left_claw_joint"}
FOLLOWERS = {
    "right": {
        "wide1": "R_hand_wide1_joint",
        "narrow3": "R_hand_narrow3_joint",
        "narrow_loop": "R_hand_narrow_loop_joint",
        "wide3": "R_hand_wide3_joint",
        "wide_loop": "R_hand_wide_loop_joint",
    },
    "left": {
        "wide1": "L_hand_wide1_joint",
        "narrow3": "L_hand_narrow3_joint",
        "narrow_loop": "L_hand_narrow_loop_joint",
        "wide3": "L_hand_wide3_joint",
        "wide_loop": "L_hand_wide_loop_joint",
    },
}
MAX_HOLD_SECONDS = 2.0
MAX_STEPS = 2_000
ROOT_ERROR_GATE_RAD = 0.02
FOLLOWER_RESIDUAL_GATE_RAD = 0.02
PELVIS_DRIFT_GATE_M = 0.08
MAX_QVEL_GATE_RAD_S = 30.0
LIMIT_TOLERANCE_RAD = 1e-5
INTERNAL_WRIST_HAND_EXCLUSIONS = (
    ("left_wrist_roll_link", "L_hand_narrow3_Link"),
    ("right_wrist_roll_link", "R_hand_narrow3_Link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory_coefficients(path: Path) -> dict[str, list[float]]:
    """Fit MuJoCo joint-equality coefficients to the reviewed 101 rows."""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 101:
        raise RuntimeError("registered trajectory must have 101 samples")
    root = np.asarray([float(row["root_q_rad"]) for row in rows])
    values: dict[str, list[float]] = {}
    for column, label in FOLLOWER_COLUMNS.items():
        follower = np.asarray([float(row[column]) for row in rows])
        coefficients = np.polynomial.polynomial.polyfit(root, follower, 4)
        if not np.isfinite(coefficients).all():
            raise RuntimeError(f"non-finite follower fit: {label}")
        values[label] = [float(item) for item in coefficients]
    return values


def maximum_fit_residual(path: Path, coefficients: dict[str, list[float]]) -> float:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    maximum = 0.0
    for row in rows:
        root = float(row["root_q_rad"])
        for column, label in FOLLOWER_COLUMNS.items():
            expected = float(np.polynomial.polynomial.polyval(root, coefficients[label]))
            maximum = max(maximum, abs(float(row[column]) - expected))
    return maximum


def add_dynamic_constraints(root: ET.Element, coefficients: dict[str, list[float]]) -> dict[str, object]:
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    joint_constraints, actuators = [], []
    for side in ("right", "left"):
        for label, follower in FOLLOWERS[side].items():
            name = f"{side}_stl_dynamic_{label}_follow"
            if equality.find(f"joint[@name='{name}']") is not None:
                raise RuntimeError(f"dynamic equality already exists: {name}")
            ET.SubElement(
                equality,
                "joint",
                {
                    "name": name,
                    "joint1": follower,
                    "joint2": ROOTS[side],
                    "polycoef": " ".join(f"{item:.17g}" for item in coefficients[label]),
                    "solref": "0.001 1",
                    "solimp": "0.99 0.999 0.001",
                },
            )
            joint_constraints.append(name)
        actuator_name = f"servo_{side}_claw_stl_dynamic"
        if actuator.find(f"position[@name='{actuator_name}']") is not None:
            raise RuntimeError(f"dynamic servo already exists: {actuator_name}")
        ET.SubElement(
            actuator,
            "position",
            {
                "name": actuator_name,
                "joint": ROOTS[side],
                "kp": "0.5",
                "kv": "0.05",
                "forcelimited": "true",
                "forcerange": "-0.5 0.5",
                "ctrllimited": "true",
                "ctrlrange": "-1 0",
            },
        )
        actuators.append(actuator_name)
    exclusions = []
    existing = {(item.get("body1"), item.get("body2")) for item in contact.findall("exclude")}
    for first, second in INTERNAL_WRIST_HAND_EXCLUSIONS:
        if (first, second) not in existing and (second, first) not in existing:
            ET.SubElement(contact, "exclude", body1=first, body2=second)
        exclusions.append([first, second])
    return {"equality_joint_constraints": joint_constraints, "root_position_actuators": actuators, "internal_collision_exclusions": exclusions}


def joint_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise RuntimeError(f"missing joint: {name}")
    return float(data.qpos[model.jnt_qposadr[joint]])


def follower_residuals(model: mujoco.MjModel, data: mujoco.MjData, coefficients: dict[str, list[float]]) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for side in ("right", "left"):
        root = joint_position(model, data, ROOTS[side])
        for label, follower in FOLLOWERS[side].items():
            expected = float(np.polynomial.polynomial.polyval(root, coefficients[label]))
            residuals[f"{side}:{label}"] = joint_position(model, data, follower) - expected
    return residuals


def limited_joint_violation(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    violations = {}
    for joint in range(model.njnt):
        if not bool(model.jnt_limited[joint]):
            continue
        qadr = int(model.jnt_qposadr[joint])
        value = float(data.qpos[qadr])
        low, high = model.jnt_range[joint]
        excess = max(float(low - value), float(value - high), 0.0)
        if excess > LIMIT_TOLERANCE_RAD:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint#{joint}"
            violations[name] = excess
    return violations


def hand_external_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records = []
    for contact in data.contact:
        body_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1])) or ""
        body_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2])) or ""
        a_hand, b_hand = "_hand_" in body_a, "_hand_" in body_b
        if a_hand == b_hand:
            continue
        records.append({"body_names": [body_a, body_b], "distance_m": float(contact.dist)})
    return records


def named_pair_contacts(model: mujoco.MjModel, data: mujoco.MjData, pairs: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    records = []
    expected = {frozenset(pair) for pair in pairs}
    for contact in data.contact:
        body_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1])) or ""
        body_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2])) or ""
        if frozenset((body_a, body_b)) in expected:
            records.append({"body_names": [body_a, body_b], "distance_m": float(contact.dist)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", required=True, type=Path)
    parser.add_argument("--left-command-percent", type=float, default=0.0)
    parser.add_argument("--right-command-percent", type=float, default=0.0)
    parser.add_argument("--reference-static-scene", type=Path, help="previous official-frame static overlay used to prove the scoped internal pair is not a source contact")
    parser.add_argument("--execute-dynamics", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not args.state_audit.is_file() or not all(0.0 <= value <= 100.0 for value in (args.left_command_percent, args.right_command_percent)) or not 0.1 <= args.hold_seconds <= MAX_HOLD_SECONDS:
        raise SystemExit("state audit, hand commands, or hold duration is invalid")
    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    required = ("SOURCE_POSTSTOP_STATE", "NAMED_ROBOT_STATE_MAPPING", "BILATERAL_STL_POSE", "STATIC_MJ_FORWARD")
    if any(audit.get("gates", {}).get(gate) != "PASS" for gate in required):
        raise SystemExit("post-stop audit has not passed static state-transfer gates")
    state_helper = load_module("poststop_state", REPO / "scripts/research/issue46/native_runtime/poststop_state_audit.py")
    handoff = json.loads(Path(audit["balance_handoff"]["path"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(audit["source_summary"]["path"]).read_text(encoding="utf-8"))
    source_overlay = Path(audit["mount_overlay"]["path"])
    if args.reference_static_scene is None or not args.reference_static_scene.is_file():
        raise SystemExit("a source-frame static reference scene is required for scoped internal collision exceptions")
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory_path = loaded.package_path / "trajectory.csv"
    coefficients = trajectory_coefficients(trajectory_path)
    fit_residual = maximum_fit_residual(trajectory_path, coefficients)
    source_tree = ET.parse(source_overlay)
    constraints = add_dynamic_constraints(source_tree.getroot(), coefficients)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-poststop-hold").native_path
    dynamic_overlay = output / "raicom_stl_poststop_dynamic_overlay.xml"
    ET.indent(source_tree, space="  ")
    pg.write_new_bytes(dynamic_overlay, ET.tostring(source_tree.getroot(), encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(dynamic_overlay))
    reference_model = mujoco.MjModel.from_xml_path(str(args.reference_static_scene))
    reference_data = mujoco.MjData(reference_model)
    mujoco.mj_forward(reference_model, reference_data)
    reference_pair_contacts = named_pair_contacts(reference_model, reference_data, INTERNAL_WRIST_HAND_EXCLUSIONS)
    if reference_pair_contacts:
        raise RuntimeError("source-frame static reference contains the requested internal collision pair; refusing exclusion")
    data = mujoco.MjData(model)
    mapping = state_helper.apply_named_state(model, data, handoff["native_post_stop_sample"], summary["model_metadata"])
    trajectory = load_registered_trajectory(loaded.package_path)
    left = apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.left_command_percent, side="left", forward=False)
    right = apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.right_command_percent, side="right", forward=False)
    for side, pose in (("left", left), ("right", right)):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{side}_claw_stl_dynamic")
        if actuator < 0:
            raise RuntimeError(f"dynamic root actuator missing: {side}")
        data.ctrl[actuator] = pose.positions_rad[ROOTS[side]]
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    initial_pelvis = data.xpos[pelvis].copy()
    initial_contacts = hand_external_contacts(model, data)
    initial_residuals = follower_residuals(model, data, coefficients)
    preflight = {
        "fit_max_residual_rad": fit_residual,
        "initial_follower_residual_max_rad": max(abs(value) for value in initial_residuals.values()),
        "initial_external_hand_contacts": initial_contacts,
        "initial_limit_violations": limited_joint_violation(model, data),
        "finite_qpos": bool(np.isfinite(data.qpos).all()),
        "finite_qvel": bool(np.isfinite(data.qvel).all()),
    }
    preflight_pass = (
        preflight["fit_max_residual_rad"] <= 0.006
        and preflight["initial_follower_residual_max_rad"] <= 0.006
        and not preflight["initial_external_hand_contacts"]
        and not preflight["initial_limit_violations"]
        and preflight["finite_qpos"]
        and preflight["finite_qvel"]
    )
    trace = []
    failure = None
    if args.execute_dynamics and preflight_pass:
        steps = min(MAX_STEPS, int(round(args.hold_seconds / model.opt.timestep)))
        for step in range(steps):
            mujoco.mj_step(model, data)
            residuals = follower_residuals(model, data, coefficients)
            drift = float(np.linalg.norm(data.xpos[pelvis] - initial_pelvis))
            record = {
                "step": step + 1,
                "time_s": float(data.time),
                "pelvis_drift_m": drift,
                "max_abs_qvel_rad_s": float(np.max(np.abs(data.qvel))),
                "max_follower_residual_rad": max(abs(value) for value in residuals.values()),
                "root_errors_rad": {side: joint_position(model, data, ROOTS[side]) - float(data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{side}_claw_stl_dynamic")]) for side in ("left", "right")},
                "hand_external_contacts": hand_external_contacts(model, data),
                "limit_violations": limited_joint_violation(model, data),
            }
            if step % 10 == 0 or step + 1 == steps:
                trace.append(record)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = {"gate": "NONFINITE_STATE", **record}
            elif record["pelvis_drift_m"] > PELVIS_DRIFT_GATE_M:
                failure = {"gate": "PELVIS_DRIFT", **record}
            elif record["max_abs_qvel_rad_s"] > MAX_QVEL_GATE_RAD_S:
                failure = {"gate": "EXCESSIVE_QVEL", **record}
            elif record["max_follower_residual_rad"] > FOLLOWER_RESIDUAL_GATE_RAD:
                failure = {"gate": "FOLLOWER_RESIDUAL", **record}
            elif max(abs(value) for value in record["root_errors_rad"].values()) > ROOT_ERROR_GATE_RAD:
                failure = {"gate": "ROOT_TRACKING", **record}
            elif record["limit_violations"]:
                failure = {"gate": "JOINT_LIMIT", **record}
            elif record["hand_external_contacts"]:
                failure = {"gate": "HAND_EXTERNAL_CONTACT", **record}
            if failure:
                break
    dynamic_status = "NOT_RUN" if not args.execute_dynamics else ("PASS" if preflight_pass and failure is None else "FAIL")
    result = {
        "schema": "robotsim.issue46.raicom-stl-poststop-hold.v1",
        "classification": "SIMULATION_ONLY_EXPERIMENTAL_BILATERAL_STL_FULL_BODY_HOLD",
        "mj_step_calls": 0 if not args.execute_dynamics else int(round(data.time / model.opt.timestep)),
        "state_audit": {"path": str(args.state_audit), "sha256": sha256(args.state_audit)},
        "dynamic_overlay": {"path": str(dynamic_overlay), "sha256": sha256(dynamic_overlay)},
        "reference_static_scene": {"path": str(args.reference_static_scene), "sha256": sha256(args.reference_static_scene), "scoped_pair_contacts": reference_pair_contacts},
        "trajectory": {"path": str(trajectory_path), "sha256": sha256(trajectory_path), "fit_max_residual_rad": fit_residual, "coefficients": coefficients},
        "constraints": constraints,
        "commands": {"left_command_percent": args.left_command_percent, "right_command_percent": args.right_command_percent, "hold_seconds": args.hold_seconds},
        "preflight": preflight,
        "trace": trace,
        "failure": failure,
        "gates": {"POSTSTOP_SOURCE_STATE": "PASS", "INITIAL_STL_CONSTRAINT_FIT": "PASS" if preflight_pass else "FAIL", "DYNAMIC_HOLD": dynamic_status, "OBJECT_CONTACT": "NOT_RUN", "GRASP": "NOT_RUN"},
        "limitations": ["The follower equalities are a fourth-order fit to the human-reviewed STL table, not vendor transmission data.", "The two added root position servos are SIMULATION_ONLY controls; no vendor actuator equivalence is claimed.", "Only wrist_roll↔narrow3 internal pairs are excluded, after checking they are absent from the official-frame static reference; external hand/object collisions remain enabled.", "This empty hold does not establish navigation-to-grasp, object contact, or lift."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "mj_step_calls": result["mj_step_calls"], "failure": failure}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
