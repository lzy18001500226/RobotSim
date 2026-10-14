#!/usr/bin/env python3
"""Validate a fitted-pin X2 OmniPicker linkage as a constrained MuJoCo model.

This is an isolated SIMULATION_ONLY diagnostic. The source URDF has fixed
secondary links; passive axes and distal pin sites are inferred from mesh
features, not vendor mechanical drawings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-g2-omnipicker-compatibility-20261013/"
    "x2_g2_omnipicker_review.xml"
)
DEFAULT_URDF = Path(
    "/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/"
    "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
)
DEFAULT_G2_XACRO = Path(
    "/home/lzy18001500226/.cache/robotsim/upstreams/genie_sim_robot_model/"
    "robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro"
)
DEFAULT_OUT = Path(
    "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/"
    "omnipicker-linkage-physics-20261014-run01"
)
PREFIX = "x2cand_candidate_"
DRIVER = PREFIX + "right_claw_joint"
MIMIC = PREFIX + "R_hand_wide1_joint"
PASSIVE = {
    "narrow": (
        PREFIX + "R_hand_narrow2_passive_hinge",
        PREFIX + "R_hand_narrow3_passive_hinge",
        PREFIX + "R_hand_narrow_loop_passive_hinge",
    ),
    "wide": (
        PREFIX + "R_hand_wide2_passive_hinge",
        PREFIX + "R_hand_wide3_passive_hinge",
        PREFIX + "R_hand_wide_loop_passive_hinge",
    ),
}
PINS = {
    "narrow": {
        "pad_body": PREFIX + "R_hand_narrow3_Link",
        "pad_local": [-0.0000657836, -0.0104436701, -0.0001030215],
        "loop_body": PREFIX + "R_hand_narrow_loop_Link",
        "loop_local": [0.0315228467, -0.0097562996, -0.0002187793],
    },
    "wide": {
        "pad_body": PREFIX + "R_hand_wide3_Link",
        "pad_local": [-0.0001688067, 0.0103357106, 0.0007564969],
        "loop_body": PREFIX + "R_hand_wide_loop_Link",
        "loop_local": [0.0315133385, 0.0096536578, 0.0009045486],
    },
}
ROOT_LIMIT = (-1.0, 0.0)
MIMIC_MULTIPLIER = -1.0
STATIC_TOL_M = 1.0e-5  # numerical feasibility threshold only; not vendor tolerance


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_joint_metadata(path: Path) -> dict:
    wanted = {
        "right_claw_joint", "R_hand_wide1_joint", "R_hand_narrow2_joint",
        "R_hand_narrow3_joint", "R_hand_narrow_loop_joint", "R_hand_wide2_joint",
        "R_hand_wide3_joint", "R_hand_wide_loop_joint",
    }
    root = ET.parse(path).getroot()
    rows = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in wanted:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        rows[name] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link") if joint.find("parent") is not None else None,
            "child": joint.find("child").get("link") if joint.find("child") is not None else None,
            "origin_xyz_m": (origin.get("xyz", "0 0 0").split() if origin is not None else ["0", "0", "0"]),
            "origin_rpy_rad": (origin.get("rpy", "0 0 0").split() if origin is not None else ["0", "0", "0"]),
            "axis_xyz": axis.get("xyz", "1 0 0").split() if axis is not None else None,
            "limit": limit.attrib if limit is not None else None,
            "mimic": mimic.attrib if mimic is not None else None,
        }
    return rows


def named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for item in root.iter(tag):
        if item.get("name") == name:
            return item
    raise KeyError(f"missing {tag} {name}")


def make_model(source_xml: Path, out_xml: Path) -> None:
    root = ET.parse(source_xml).getroot()
    root.set("model", "X2-OmniPicker-fitted-pin-constrained-diagnostic")
    world = root.find("worldbody")
    assert world is not None
    assembly = named(world, "body", "x2cand_assembly_base")
    for child in list(world):
        if child is not assembly and child.tag == "body":
            world.remove(child)
    assembly.set("pos", "0 0 0")

    for side, data in PINS.items():
        pad = named(assembly, "body", data["pad_body"])
        loop = named(assembly, "body", data["loop_body"])
        ET.SubElement(pad, "site", {
            "name": f"{side}_distal_pin_pad", "type": "sphere", "size": "0.0005",
            "pos": " ".join(f"{x:.10g}" for x in data["pad_local"]),
            "rgba": "1 0.1 0.1 1", "group": "3",
        })
        ET.SubElement(loop, "site", {
            "name": f"{side}_distal_pin_loop", "type": "sphere", "size": "0.0005",
            "pos": " ".join(f"{x:.10g}" for x in data["loop_local"]),
            "rgba": "0.1 0.3 1 1", "group": "3",
        })

    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", {
        "name": "source_root_mimic", "joint1": MIMIC, "joint2": DRIVER,
        "polycoef": "0 -1 0 0 0",
    })
    for side in ("narrow", "wide"):
        ET.SubElement(eq, "connect", {
            "name": f"{side}_distal_pin_loop", "site1": f"{side}_distal_pin_pad",
            "site2": f"{side}_distal_pin_loop",
        })

    # The URDF has no nonzero effort or velocity metadata for the claw.
    # This bounded, low-gain servo is explicitly diagnostic, not vendor data.
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    else:
        for child in list(actuators):
            actuators.remove(child)
    ET.SubElement(actuators, "position", {
        "name": "diagnostic_driver_position", "joint": DRIVER,
        "kp": "0.04", "kv": "0.002", "ctrllimited": "true",
        "ctrlrange": "-1 0", "forcelimited": "true", "forcerange": "-0.02 0.02",
    })
    ET.indent(root, space="  ")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n")


def obj_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise KeyError(name)
    return value


def qadr(model: mujoco.MjModel, joint: str) -> int:
    return int(model.jnt_qposadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def site_gap(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    a = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_pad")
    b = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_loop")
    return float(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]))


def geom_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    rows = []
    for index in range(data.ncon):
        c = data.contact[index]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2)
        if n1.startswith(PREFIX) or n2.startswith(PREFIX):
            rows.append({
                "body1": n1, "body2": n2,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                "signed_distance_mm": float(c.dist * 1000),
                "penetration_proxy_mm": max(0.0, -float(c.dist * 1000)),
            })
    return rows


def equality_residuals(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    rows = {}
    kind = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    for name in ("source_root_mimic", "narrow_distal_pin_loop", "wide_distal_pin_loop"):
        equality_id = obj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        ids = np.flatnonzero((data.efc_type[:data.nefc] == kind) &
                             (data.efc_id[:data.nefc] == equality_id))
        values = data.efc_pos[ids].copy()
        if name == "source_root_mimic":
            rows[name] = {"residual_rad": values.tolist(),
                          "max_abs_residual_rad": float(np.max(np.abs(values))) if len(values) else 0.0}
        else:
            rows[name] = {"residual_world_m": values.tolist(),
                          "residual_norm_mm": float(np.linalg.norm(values) * 1000.0)}
    return rows


def solve_static_pose(model: mujoco.MjModel, data: mujoco.MjData, q: float,
                      previous: dict[str, np.ndarray]) -> dict:
    data.qpos[qadr(model, DRIVER)] = q
    data.qpos[qadr(model, MIMIC)] = -q
    sides = {}
    for side, joints in PASSIVE.items():
        ads = [qadr(model, name) for name in joints]
        start = previous[side].copy()

        def residual(values: np.ndarray) -> np.ndarray:
            data.qpos[ads] = values
            mujoco.mj_forward(model, data)
            return data.site_xpos[obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_pad")] - data.site_xpos[
                obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_loop")]

        result = least_squares(residual, start, max_nfev=200, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        data.qpos[ads] = result.x
        mujoco.mj_forward(model, data)
        previous[side] = result.x.copy()
        pad_site = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_pad")
        loop_site = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_distal_pin_loop")
        delta = data.site_xpos[pad_site] - data.site_xpos[loop_site]
        jac = np.zeros((3, 3))
        eps = 1.0e-6
        for col, adr in enumerate(ads):
            base = float(data.qpos[adr])
            data.qpos[adr] = base + eps
            mujoco.mj_forward(model, data)
            plus = data.site_xpos[pad_site] - data.site_xpos[loop_site]
            data.qpos[adr] = base - eps
            mujoco.mj_forward(model, data)
            minus = data.site_xpos[pad_site] - data.site_xpos[loop_site]
            jac[:, col] = (plus - minus) / (2 * eps)
            data.qpos[adr] = base
        mujoco.mj_forward(model, data)
        u, singular_values, _ = np.linalg.svd(jac, full_matrices=False)
        rank = int(np.sum(singular_values > max(singular_values[0] * 1e-8, 1e-12)))
        projection = u[:, :rank] @ u[:, :rank].T @ delta if rank else np.zeros(3)
        unreachable = delta - projection
        sides[side] = {
            "passive_joint_rad": dict(zip(joints, map(float, result.x))),
            "site_closure_residual_mm": site_gap(model, data, side) * 1000.0,
            "site_error_vector_world_mm": (delta * 1000.0).tolist(),
            "passive_angle_jacobian_singular_values_m_per_rad": singular_values.tolist(),
            "passive_angle_jacobian_rank": rank,
            "residual_outside_passive_motion_subspace_mm": float(np.linalg.norm(unreachable) * 1000.0),
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_evaluations": int(result.nfev),
        }
    mujoco.mj_forward(model, data)
    return {
        "driver_q_rad": float(q), "mimic_q_rad": float(data.qpos[qadr(model, MIMIC)]),
        "mimic_residual_rad": float(data.qpos[qadr(model, MIMIC)] + data.qpos[qadr(model, DRIVER)]),
        "sides": sides,
        "mujoco_equality_residuals": equality_residuals(model, data),
        "active_contact_count": len(geom_contacts(model, data)),
        "contacts": geom_contacts(model, data),
    }


def run_static(model: mujoco.MjModel, out: Path, samples: int) -> dict:
    data = mujoco.MjData(model)
    previous = {key: np.zeros(3) for key in PASSIVE}
    rows = []
    for q in np.linspace(0.0, -1.0, samples):
        row = solve_static_pose(model, data, float(q), previous)
        rows.append(row)
    maximum = max(max(v["site_closure_residual_mm"] for v in r["sides"].values()) for r in rows)
    max_mimic = max(abs(r["mimic_residual_rad"]) for r in rows)
    collision_rows = [c for r in rows for c in r["contacts"]]
    max_pen = max((c["penetration_proxy_mm"] for c in collision_rows), default=0.0)
    worst_collision = max(collision_rows, key=lambda c: c["penetration_proxy_mm"], default=None)
    if worst_collision is not None:
        for row in rows:
            if worst_collision in row["contacts"]:
                worst_collision = {**worst_collision, "driver_q_rad": row["driver_q_rad"]}
                break
    closure_pass = maximum <= STATIC_TOL_M * 1000 and max_mimic <= 1e-8
    result = {
        "status": "PASS_STATIC_CONSTRAINT_FEASIBILITY" if closure_pass else "FAIL_STATIC_CONSTRAINT_FEASIBILITY",
        "sample_count": len(rows), "q_range_rad": [0.0, -1.0],
        "mujoco_version": mujoco.__version__,
        "static_numeric_threshold_mm": STATIC_TOL_M * 1000,
        "threshold_note": "Numerical feasibility threshold only; not a vendor mechanical tolerance.",
        "max_site_closure_residual_mm": maximum,
        "max_source_mimic_residual_rad": max_mimic,
        "max_passive_abs_angle_rad": max(abs(x) for row in rows for s in row["sides"].values()
                                           for x in s["passive_joint_rad"].values()),
        "max_active_hull_penetration_proxy_mm": max_pen,
        "worst_active_hull_contact": worst_collision,
        "contact_pair_occurrences": len(collision_rows),
        "distinct_contact_pairs": sorted({(c["body1"], c["body2"]) for c in collision_rows}),
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "static_scan.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "static_scan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["driver_q_rad", "mimic_residual_rad",
                                                   "narrow_closure_mm", "wide_closure_mm",
                                                   "contacts", "max_penetration_proxy_mm"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "driver_q_rad": row["driver_q_rad"],
                "mimic_residual_rad": row["mimic_residual_rad"],
                "narrow_closure_mm": row["sides"]["narrow"]["site_closure_residual_mm"],
                "wide_closure_mm": row["sides"]["wide"]["site_closure_residual_mm"],
                "contacts": row["active_contact_count"],
                "max_penetration_proxy_mm": max((c["penetration_proxy_mm"] for c in row["contacts"]), default=0.0),
            })
    return result


def smoothstep(value: float) -> float:
    x = min(1.0, max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


def run_dynamics(model: mujoco.MjModel, out: Path, loops: int, seconds_per_leg: float) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos_at_reset = data.qpos.copy()
    previous = {key: np.zeros(3) for key in PASSIVE}
    initial = solve_static_pose(model, data, 0.0, previous)
    # One-time reset initialization to a feasible closed state; no qpos writes occur after stepping starts.
    initial_seed_delta = data.qpos - qpos_at_reset
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    actuator = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "diagnostic_driver_position")
    rows = []
    peak_vel = 0.0
    peak_qacc = 0.0
    max_eq = 0.0
    max_pen = 0.0
    max_pin = 0.0
    max_mimic = 0.0
    max_root_limit_violation = 0.0
    limit_violation_steps = 0
    total_steps = 0
    phase_defs = [("open", 0.0, -1.0), ("close", -1.0, 0.0)]
    try:
        with (out / "dynamics_trace.csv").open("w", newline="") as stream:
            fields = ["step", "time_s", "phase", "target_q_rad", "driver_q_rad", "driver_qvel_rad_s",
                      "driver_qacc_rad_s2", "mimic_q_rad", "mimic_residual_rad", "actuator_force_nm",
                      "narrow2_q_rad", "narrow3_q_rad", "narrow_loop_q_rad", "wide2_q_rad",
                      "wide3_q_rad", "wide_loop_q_rad", "narrow_pin_error_mm", "wide_pin_error_mm",
                      "ncon", "candidate_contact_pairs", "max_penetration_proxy_mm", "nefc", "max_eq_residual",
                      "solver_iterations", "finite_state"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for cycle in range(loops):
                for phase, start, end in phase_defs:
                    count = max(1, math.ceil(seconds_per_leg / dt))
                    for index in range(count):
                        u = (index + 1) / count
                        target = start + (end - start) * smoothstep(u)
                        data.ctrl[actuator] = target
                        mujoco.mj_step(model, data)
                        total_steps += 1
                        contacts = geom_contacts(model, data)
                        penetration = max((x["penetration_proxy_mm"] for x in contacts), default=0.0)
                        eq_ids = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                        eq_res = float(np.max(np.abs(data.efc_pos[eq_ids]))) if len(eq_ids) else 0.0
                        max_eq = max(max_eq, eq_res)
                        max_pen = max(max_pen, penetration)
                        q_driver = float(data.qpos[qadr(model, DRIVER)])
                        q_mimic = float(data.qpos[qadr(model, MIMIC)])
                        mimic_error = q_mimic + q_driver
                        max_mimic = max(max_mimic, abs(mimic_error))
                        root_violation = max(ROOT_LIMIT[0] - q_driver, q_driver - ROOT_LIMIT[1], 0.0)
                        root_violation = max(root_violation, -q_mimic, q_mimic - 1.0, 0.0)
                        max_root_limit_violation = max(max_root_limit_violation, root_violation)
                        if root_violation > 1e-9:
                            limit_violation_steps += 1
                        pin_error = max(site_gap(model, data, "narrow"), site_gap(model, data, "wide")) * 1000.0
                        max_pin = max(max_pin, pin_error)
                        peak_vel = max(peak_vel, float(np.max(np.abs(data.qvel))))
                        peak_qacc = max(peak_qacc, float(np.max(np.abs(data.qacc))))
                        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                                      and np.isfinite(data.qacc).all())
                        q = lambda name: float(data.qpos[qadr(model, name)])
                        writer.writerow({
                            "step": total_steps, "time_s": float(data.time), "phase": f"{cycle}:{phase}",
                            "target_q_rad": target, "driver_q_rad": q_driver,
                            "driver_qvel_rad_s": float(data.qvel[model.jnt_dofadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, DRIVER)]]),
                            "driver_qacc_rad_s2": float(data.qacc[model.jnt_dofadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, DRIVER)]]),
                            "mimic_q_rad": q_mimic, "mimic_residual_rad": mimic_error,
                            "actuator_force_nm": float(data.actuator_force[actuator]),
                            "narrow2_q_rad": q(PASSIVE["narrow"][0]), "narrow3_q_rad": q(PASSIVE["narrow"][1]),
                            "narrow_loop_q_rad": q(PASSIVE["narrow"][2]), "wide2_q_rad": q(PASSIVE["wide"][0]),
                            "wide3_q_rad": q(PASSIVE["wide"][1]), "wide_loop_q_rad": q(PASSIVE["wide"][2]),
                            "narrow_pin_error_mm": site_gap(model, data, "narrow") * 1000.0,
                            "wide_pin_error_mm": site_gap(model, data, "wide") * 1000.0,
                            "ncon": int(data.ncon), "candidate_contact_pairs": len(contacts),
                            "max_penetration_proxy_mm": penetration, "nefc": int(data.nefc),
                            "max_eq_residual": eq_res, "solver_iterations": int(np.max(data.solver_niter)),
                            "finite_state": int(finite),
                        })
                        if not finite:
                            raise FloatingPointError(f"non-finite state at step {total_steps}")
    finally:
        pass
    result = {
        "status": "PASS_DYNAMIC_CONSTRAINTS" if (max_eq < 1e-5 and max_pin < STATIC_TOL_M * 1000.0
                                                      and limit_violation_steps == 0 and max_mimic < 1e-5)
        else "FAIL_DYNAMIC_CONSTRAINTS",
        "mujoco_version": mujoco.__version__, "timestep_s": dt, "steps": total_steps,
        "cycles": loops, "seconds_per_open_or_close_leg": seconds_per_leg,
        "controller": {"type": "MuJoCo position servo on root claw only", "kp_Nm_per_rad": 0.04,
                       "kv_Nm_s_per_rad": 0.002, "force_cap_Nm": 0.02,
                       "classification": "SIMULATION_DERIVED; URDF effort/velocity fields are zero"},
        "actuated_joint": DRIVER, "constrained_mimic_joint": MIMIC,
        "passive_actuators": 0, "active_rollout_qpos_writes": 0,
        "initial_closed_state_pin_residual_mm": {
            side: initial["sides"][side]["site_closure_residual_mm"] for side in ("narrow", "wide")
        },
        "one_time_initialization_qpos_seed_rad": initial_seed_delta.tolist(),
        "max_equality_residual_m": max_eq, "max_pin_site_residual_mm": max(
            max_pin, 0.0),
        "max_abs_qvel_rad_s": peak_vel, "max_abs_qacc_rad_s2": peak_qacc,
        "max_hull_penetration_proxy_mm": max_pen,
        "joint_limits": {"root_driver_rad": list(ROOT_LIMIT), "mimic_relation": "q_wide=-q_driver",
                         "passive_source_limits": "none; URDF declares these joints fixed; no limits inferred"},
        "source_limit_violations": limit_violation_steps,
        "max_source_limit_violation_rad": max_root_limit_violation,
        "max_mimic_residual_rad": max_mimic,
        "note": "Diagnostic dynamics only; not hardware fidelity or grasp validation.",
    }
    (out / "dynamics_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-xml", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--g2-xacro", type=Path, default=DEFAULT_G2_XACRO)
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--dynamics", action="store_true", help="run only if full-range static closure passes")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--leg-seconds", type=float, default=1.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    xml_out = args.out / "constrained_candidate.xml"
    make_model(args.source_xml, xml_out)
    model = mujoco.MjModel.from_xml_path(str(xml_out))
    mujoco.mj_forward(model, mujoco.MjData(model))
    static = run_static(model, args.out, args.samples)
    report = {
        "attempt_id": args.attempt_id,
        "classification": "SIMULATION_ONLY fitted-pin constrained candidate",
        "mujoco_version": mujoco.__version__,
        "input_xml": str(args.source_xml), "input_xml_sha256": sha256(args.source_xml),
        "source_urdf": str(args.source_urdf), "source_urdf_sha256": sha256(args.source_urdf),
        "source_joint_metadata": source_joint_metadata(args.source_urdf),
        "g2_xacro": str(args.g2_xacro), "g2_xacro_sha256": sha256(args.g2_xacro),
        "g2_source_revision": "AgibotTech/genie_sim_robot_model 6d5b2159806ae0e64b432b638e9ee288d8bddc8f",
        "generated_model": str(xml_out), "generated_model_sha256": sha256(xml_out),
        "mujoco_model_dimensions": {"nq": int(model.nq), "nv": int(model.nv), "neq": int(model.neq),
                                    "nu": int(model.nu), "ngeom": int(model.ngeom)},
        "equality_design": {
            "one root joint polynomial mimic": "q_wide=-q_claw, source multiplier -1, no follower actuator",
            "two site-to-site connect constraints": "fitted distal pin centers narrow and wide",
            "passive_hinges": [name for joints in PASSIVE.values() for name in joints],
            "passive_joint_ranges": "unlimited; no supported source limit exists because URDF declares them fixed",
            "pin_provenance": "STL principal-axis bore fits from previous static report; not CAD datums",
            "collision_geometry": "unchanged compiled X2 collision meshes from input candidate",
        },
        "static": {k: v for k, v in static.items() if k != "rows"},
        "physics_validation": {
            "status": "NOT_RUN_STATIC_GATE_BLOCKED" if static["status"] != "PASS_STATIC_CONSTRAINT_FEASIBILITY" else "NOT_RUN",
            "mj_step_calls": 0,
            "reason": "The full-range fitted-pin static closure gate failed; no dynamic integration was started.",
        },
    }
    if args.dynamics:
        if static["status"] != "PASS_STATIC_CONSTRAINT_FEASIBILITY":
            report["physics_validation"] = {
                "status": "NOT_RUN_STATIC_GATE_BLOCKED", "mj_step_calls": 0,
                "reason": "The full-range fitted-pin static closure gate failed; no dynamic integration was started.",
            }
            (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return 2
        report["dynamics"] = run_dynamics(model, args.out, args.cycles, args.leg_seconds)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if static["status"] == "PASS_STATIC_CONSTRAINT_FEASIBILITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
