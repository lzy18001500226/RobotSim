#!/usr/bin/env python3
"""Bounded Robotiq coupler source-limit diagnosis for Issue #46."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0
import issue46_x2_robotiq_mounted_smoke as smoke


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-physical-recovery-20261009/coupler"
)
GRIP_ACTUATOR = "rq_fingers_actuator"
DRIVER_JOINTS = ("rq_right_driver_joint", "rq_left_driver_joint")
COUPLER_JOINTS = ("rq_right_coupler_joint", "rq_left_coupler_joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qstate(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> dict[str, Any]:
    jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return {
        "joint": joint,
        "qpos0_rad": float(model.qpos0[qa]),
        "range_rad": model.jnt_range[jid].astype(float).tolist(),
        "ref_rad": float(model.qpos0[qa]),
        "axis_local": model.jnt_axis[jid].astype(float).tolist(),
        "qpos_rad": float(data.qpos[qa]),
        "qvel_rad_s": float(data.qvel[da]),
        "qacc_rad_s2": float(data.qacc[da]),
        "qfrc_bias_nm": float(data.qfrc_bias[da]),
        "qfrc_actuator_nm": float(data.qfrc_actuator[da]),
        "qfrc_constraint_nm": float(data.qfrc_constraint[da]),
        "joint_solref": model.jnt_solref[jid].astype(float).tolist(),
        "joint_solimp": model.jnt_solimp[jid].astype(float).tolist(),
        "armature": float(model.dof_armature[da]),
        "damping": float(model.dof_damping[da]),
        "frictionloss": float(model.dof_frictionloss[da]),
    }


def constraints(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for row in range(data.nefc):
        if int(data.efc_type[row]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        eid = int(data.efc_id[row])
        rows.append({
            "efc_row": row,
            "equality_id": eid,
            "equality_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eid),
            "residual_position": float(data.efc_pos[row]),
            "force": float(data.efc_force[row]),
            "active": bool(data.eq_active[eid]),
            "type": int(model.eq_type[eid]),
            "object1": int(model.eq_obj1id[eid]),
            "object2": int(model.eq_obj2id[eid]),
            "polycoef": model.eq_data[eid].astype(float).tolist(),
            "solref": model.eq_solref[eid].astype(float).tolist(),
            "solimp": model.eq_solimp[eid].astype(float).tolist(),
        })
    return rows


def mass_terms(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    full = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, full, data.qM)
    dofs = {joint: int(model.jnt_dofadr[m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
            for joint in (*DRIVER_JOINTS, *COUPLER_JOINTS)}
    return {
        "joint_diagonal_kg_m2": {joint: float(full[dof, dof]) for joint, dof in dofs.items()},
        "coupler_driver_submatrix_kg_m2": {
            driver: {coupler: float(full[dofs[driver], dofs[coupler]]) for coupler in COUPLER_JOINTS}
            for driver in DRIVER_JOINTS
        },
    }


def write_timestep_variant(source: Path, destination: Path, timestep: float) -> None:
    root = ET.parse(source).getroot()
    option = root.find("option")
    if option is None:
        raise RuntimeError("Compiled experiment XML has no <option>")
    option.set("timestep", f"{timestep:.9g}")
    ET.indent(root, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def actuator_target_for_driver_pose(model: mujoco.MjModel, driver_q: float) -> dict[str, Any]:
    aid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    reference = mujoco.MjData(model)
    mujoco.mj_resetData(model, reference)
    for joint in DRIVER_JOINTS:
        jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        reference.qpos[int(model.jnt_qposadr[jid])] = driver_q
    mujoco.mj_forward(model, reference)
    length = float(reference.actuator_length[aid])
    velocity = 0.0
    gain = float(model.actuator_gainprm[aid, 0])
    bias = model.actuator_biasprm[aid].astype(float)
    if abs(gain) < 1e-12:
        raise RuntimeError("Cannot invert zero actuator gain")
    command = -(float(bias[0]) + float(bias[1]) * length + float(bias[2]) * velocity) / gain
    ctrlrange = model.actuator_ctrlrange[aid].astype(float).tolist()
    if command < ctrlrange[0] or command > ctrlrange[1]:
        raise RuntimeError(f"Derived target command {command} is outside actuator range {ctrlrange}")
    return {
        "method": "compiled tendon length plus source actuator gain/bias equilibrium",
        "driver_target_rad": driver_q,
        "offline_kinematic_actuator_length_m": length,
        "actuator_gainprm": model.actuator_gainprm[aid].astype(float).tolist(),
        "actuator_biasprm": bias.tolist(),
        "actuator_ctrlrange": ctrlrange,
        "derived_control": command,
        "equilibrium_force_at_target_n": float(gain * command + bias[0] + bias[1] * length),
        "note": "The reference qpos assignment is confined to a separate offline target-mapping MjData; rollout qpos is untouched.",
    }


def initialize_constraint_consistent_interior(model: mujoco.MjModel, data: mujoco.MjData,
                                               driver_q: float) -> dict[str, Any]:
    """Create a source-valid interior OPEN state on the original equality manifold."""
    smoke.set_mounted_neutral(model, data)
    changed: dict[str, float] = {}
    for joint in DRIVER_JOINTS:
        jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        qa = int(model.jnt_qposadr[jid])
        lo, hi = map(float, model.jnt_range[jid])
        if not lo < driver_q < hi:
            raise ValueError(f"Interior driver coordinate {driver_q} is outside {joint} range [{lo}, {hi}]")
        data.qpos[qa] = driver_q
        changed[joint] = driver_q
    mujoco.mj_forward(model, data)
    loop_solutions = []
    for side, eqid in (("right", 0), ("left", 1)):
        joint_names = tuple(f"rq_{side}_{part}_joint" for part in ("coupler", "spring_link", "follower"))
        jids = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in joint_names]
        qids = np.asarray([int(model.jnt_qposadr[jid]) for jid in jids], dtype=int)
        lower = np.asarray([model.jnt_range[jid, 0] for jid in jids], dtype=float)
        upper = np.asarray([model.jnt_range[jid, 1] for jid in jids], dtype=float)
        equality_rows = np.flatnonzero((data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) &
                                       (data.efc_id == eqid))
        if len(equality_rows) != 3:
            raise RuntimeError(f"Expected three connect rows for {side} loop equality {eqid}, got {len(equality_rows)}")
        seed = data.qpos[qids].copy()

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[qids] = q
            mujoco.mj_forward(model, data)
            rows = np.flatnonzero((data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) &
                                  (data.efc_id == eqid))
            return data.efc_pos[rows].copy()

        fit = least_squares(residual, np.clip(seed, lower + 1e-10, upper - 1e-10),
                            bounds=(lower, upper), max_nfev=250,
                            xtol=1e-13, ftol=1e-13, gtol=1e-13)
        data.qpos[qids] = fit.x
        mujoco.mj_forward(model, data)
        final_rows = np.flatnonzero((data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) &
                                    (data.efc_id == eqid))
        loop_solutions.append({
            "side": side,
            "equality_id": eqid,
            "joint_names": list(joint_names),
            "joint_qpos_rad": data.qpos[qids].astype(float).tolist(),
            "range_rad": [[float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])] for jid in jids],
            "solver_success": bool(fit.success),
            "solver_message": str(fit.message),
            "evaluations": int(fit.nfev),
            "connect_residual_m": data.efc_pos[final_rows].astype(float).tolist(),
            "max_abs_connect_residual_m": float(np.max(np.abs(data.efc_pos[final_rows]))),
        })
        for joint, q in zip(joint_names, data.qpos[qids]):
            changed[joint] = float(q)
    mujoco.mj_forward(model, data)
    residual_rows = np.flatnonzero(data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    residual_values = data.efc_pos[residual_rows]
    position_violations, _ = m0.all_limited_joint_checks(model, data, {})
    contacts = smoke.describe_contacts(model, data, smoke.robot_body_names(model))
    result = {
        "initialization_only_qpos_assignments": changed,
        "no_qpos_assignments_after_rollout_start": True,
        "driver_target_rad": driver_q,
        "loop_solutions": loop_solutions,
        "all_equality_residuals_m": residual_values.astype(float).tolist(),
        "max_abs_equality_residual": float(np.max(np.abs(residual_values))) if len(residual_values) else 0.0,
        "source_position_limit_violations": position_violations,
        "initial_robot_self_contacts": contacts["robot_self_contacts"],
        "initial_robot_bottle_contacts": contacts["robot_bottle_contacts"],
        "pad_separation_m": smoke.pad_gap(model, data),
    }
    if (not all(row["solver_success"] for row in loop_solutions) or
            result["max_abs_equality_residual"] > 1e-8 or position_violations or
            contacts["robot_self_contacts"] or contacts["robot_bottle_contacts"]):
        raise RuntimeError(f"Constraint-consistent interior initialization failed gates: {result}")
    return result


def record_state(model: mujoco.MjModel, data: mujoco.MjData, label: str,
                 control: float, controller_commands: list[dict[str, Any]]) -> dict[str, Any]:
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
              for jid in range(model.njnt)
              if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
              and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").startswith("rq_")]
    aid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    return {
        "label": label,
        "time_s": float(data.time),
        "actuator_control": control,
        "actuator_force_n": float(data.actuator_force[aid]),
        "qfrc_actuator": data.qfrc_actuator.astype(float).tolist(),
        "qfrc_constraint": data.qfrc_constraint.astype(float).tolist(),
        "qfrc_bias": data.qfrc_bias.astype(float).tolist(),
        "all_robotiq_joint_state": [qstate(model, data, joint) for joint in joints],
        "coupler_constraint_rows": constraints(model, data),
        "effective_mass_terms": mass_terms(model, data),
        "x2_controller_commands": controller_commands,
        "qpos_writes_after_rollout_start": 0,
    }


def run_probe(model: mujoco.MjModel, ident: dict[str, Any], name: str,
              command: float, duration_s: float, out: Path,
              interior_driver_target: float | None = None) -> dict[str, Any]:
    data = mujoco.MjData(model)
    initialization = ({"kind": "source reset neutral"} if interior_driver_target is None else
                      {"kind": "constraint-consistent interior OPEN", **initialize_constraint_consistent_interior(
                          model, data, interior_driver_target)})
    if interior_driver_target is None:
        smoke.set_mounted_neutral(model, data)
    refs, targets = smoke.set_controller_targets(model, data)
    aid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    data.ctrl[aid] = command
    velocity_limits = {joint: info["velocity"] for joint, info in ident["urdf_joint_limits"].items()
                       if math.isfinite(info["velocity"]) and info["velocity"] > 0}
    count = int(round(duration_s / float(model.opt.timestep)))
    rows = []
    limits = []
    mujoco.mj_forward(model, data)
    controller_commands = m0.apply_controller(model, data, refs, targets, {})
    data.ctrl[aid] = command
    mujoco.mj_forward(model, data)
    rows.append(record_state(model, data, "initial_pre_step", command, controller_commands))
    for step in range(1, count + 1):
        mujoco.mj_forward(model, data)
        controller_commands = m0.apply_controller(model, data, refs, targets, {})
        data.ctrl[aid] = command
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        pos, vel = m0.all_limited_joint_checks(model, data, velocity_limits)
        violations = [{"step": step, "kind": "position", **v} for v in pos]
        violations.extend({"step": step, "kind": "velocity", **v} for v in vel)
        limits.extend(violations)
        rows.append(record_state(model, data, "post_step", command, controller_commands))
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()):
            raise RuntimeError(f"Non-finite state in {name} at step {step}")
    trace = out / "raw" / f"{name}_step_trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    coupler_samples = []
    for row in rows:
        for joint in COUPLER_JOINTS:
            state = next(item for item in row["all_robotiq_joint_state"] if item["joint"] == joint)
            coupler_samples.append((joint, state["qpos_rad"], state["qvel_rad_s"], state["qacc_rad_s2"]))
    max_positive_overshoot = {
        joint: max(0.0, max(q - 0.0 for n, q, _, _ in coupler_samples if n == joint))
        for joint in COUPLER_JOINTS
    }
    return {
        "name": name,
        "timestep_s": float(model.opt.timestep),
        "solver": int(model.opt.solver),
        "integrator": int(model.opt.integrator),
        "iterations": int(model.opt.iterations),
        "open_command": command,
        "initialization": initialization,
        "requested_duration_s": duration_s,
        "actual_duration_s": float(data.time),
        "steps_executed": len(rows) - 1,
        "source_limit_violations": limits,
        "max_positive_coupler_upper_stop_overshoot_rad": max_positive_overshoot,
        "peak_abs_qvel_rad_s": float(np.max(np.abs(data.qvel))),
        "peak_abs_qacc_rad_s2": float(np.max(np.abs(data.qacc))),
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()),
        "passed_10ms_limit_gate": not limits and abs(float(data.time) - duration_s) <= model.opt.timestep + 1e-12,
        "trace_jsonl": str(trace),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=m0.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=smoke.DEFAULT_STATION)
    parser.add_argument("--probe-seconds", type=float, default=0.010)
    parser.add_argument("--interior-driver-target", type=float, default=0.010)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment": "Issue #46 Robotiq coupler source-limit initialization diagnosis",
        "status": "BLOCKED",
        "scope": "No source joint ranges, equalities, tendon topology, collision geoms, or source files are changed.",
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "robot_sim_head": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
            "command": sys.argv,
        },
    }
    try:
        if smoke.sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical helper hash does not match accepted scene")
        ident = m0.identity(args, args.canonical_helper)
        result["identity"] = ident
        canonical = m0.load_module(args.canonical_helper)
        (out / "model").mkdir(parents=True, exist_ok=True)
        build_args = type("BuildArgs", (), {
            "x2_root": args.x2_root,
            "menagerie_root": args.menagerie_root,
            "station_base_pos": np.asarray(args.station_base_pos, dtype=float),
        })()
        base_model, adapter = m0.build_model(build_args, out / "model", canonical)
        base_xml = Path(adapter["model_xml"])
        excluded_xml = out / "model" / "x2_robotiq_head_torso_exclude_simulation_only.xml"
        result["authorized_contact_exception"] = smoke.create_exclusion_variant(base_xml, excluded_xml)
        result["adapter"] = adapter
        target_mapping = actuator_target_for_driver_pose(base_model, args.interior_driver_target)
        result["interior_open_target_mapping"] = target_mapping
        variants = [
            ("baseline_endpoint_dt1ms", 0.001, 0.0, None, None),
            ("baseline_endpoint_dt0p5ms", 0.0005, 0.0, None, None),
            ("interior_open_from_reset_dt1ms", 0.001, target_mapping["derived_control"], None, None),
            ("constraint_consistent_interior_open_dt1ms", 0.001, target_mapping["derived_control"],
             args.interior_driver_target, None),
        ]
        probes = {}
        variant_xmls = {}
        for label, timestep, command, interior_driver_target, gravity in variants:
            xml_path = out / "model" / f"{label}.xml"
            write_timestep_variant(excluded_xml, xml_path, timestep)
            if gravity is not None:
                root = ET.parse(xml_path).getroot()
                root.find("option").set("gravity", "0 0 0")
                ET.indent(root, space="  ")
                ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            variant_xmls[label] = str(xml_path)
            probes[label] = run_probe(model, ident, label, command, args.probe_seconds, out,
                                      interior_driver_target)
        gravity_xml = out / "model" / "gravity_off_endpoint_dt1ms.xml"
        write_timestep_variant(excluded_xml, gravity_xml, 0.001)
        gravity_root = ET.parse(gravity_xml).getroot()
        gravity_root.find("option").set("gravity", "0 0 0")
        ET.indent(gravity_root, space="  ")
        ET.ElementTree(gravity_root).write(gravity_xml, encoding="utf-8", xml_declaration=True)
        gravity_model = mujoco.MjModel.from_xml_path(str(gravity_xml))
        variant_xmls["gravity_off_endpoint_dt1ms"] = str(gravity_xml)
        probes["gravity_off_endpoint_dt1ms"] = run_probe(
            gravity_model, ident, "gravity_off_endpoint_dt1ms", 0.0, args.probe_seconds, out)
        result["probe_variants"] = probes
        result["variant_xml"] = variant_xmls
        interior = probes["constraint_consistent_interior_open_dt1ms"]
        if interior["passed_10ms_limit_gate"]:
            model = mujoco.MjModel.from_xml_path(variant_xmls["constraint_consistent_interior_open_dt1ms"])
            data = mujoco.MjData(model)
            initialize_constraint_consistent_interior(model, data, args.interior_driver_target)
            full_cycle = smoke.run_cycle(model, data, ident, out / "full_cycle", target_mapping["derived_control"])
            result["interior_open_full_cycle"] = full_cycle
            result["coupler_source_limit_gate"] = "PASS" if full_cycle["status"] == "PASS" else "FAIL"
        else:
            result["coupler_source_limit_gate"] = "FAIL"
            result["full_cycle"] = "NOT RUN: 10 ms interior-target probe did not pass strict source limits"
        result["status"] = "PASS" if result["coupler_source_limit_gate"] == "PASS" else "FAIL"
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "coupler_source_limit_gate": result.get("coupler_source_limit_gate"),
                      "result_json": str(out / "result.json"), "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
