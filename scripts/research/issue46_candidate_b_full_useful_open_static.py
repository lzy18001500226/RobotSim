#!/usr/bin/env python3
"""Static-only full-useful-range check for the frozen X2 OmniPicker Candidate B.

The original Candidate B and all manufacturer assets remain unchanged.  This
research overlay replaces only Candidate B's unverified linear passive-rate
equalities with a geometry-selected quartic joint relation.  It performs
MuJoCo forward kinematics and offline D-closure solves; it never calls
``mj_step`` and does not write passive qpos during a dynamic rollout.

The overlay is SIMULATION_ONLY and is not manufacturer transmission data.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import issue46_candidate_b_clearance_coupling as clearance
import issue46_candidate_b_functional_validation as functional
import issue46_single_motor_travel as travel
import issue46_urdf_pipeline as pipeline
from agent import path_governance as pg


REPO = Path("/home/lzy18001500226/projects/RobotSim")
EVIDENCE_ROOT = REPO / "reviews/issue-46"
BASELINE_RUN = REPO / "reviews/issue-46/candidate-b-functional-validation-20261022T140838Z"
BASE_TRAVEL = BASELINE_RUN / "travel_analysis.json"
HIGH_STATIC_RUN = REPO / "reviews/issue-46/candidate-b-clearance-physics-20261022T182958Z"
HIGH_STATIC = HIGH_STATIC_RUN / "static_preflight.json"
HIGH_STATIC_MODEL = HIGH_STATIC_RUN / "candidate_geometry_coupled.xml"
HIGH_CYLINDER_PROBES = HIGH_STATIC_RUN / "cylinder_probe_results.json"

CLASSIFICATION = "SIMULATION_ONLY_UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING"
Q_CLOSED = pipeline.PHASE
Q_USEFUL_OPEN = 0.55
Q2_CLOSED_SAFE = 0.10252605152311378
COMMAND_SAFETY_FRACTIONS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
COMMAND_SAFETY_Q2_MAX = (Q2_CLOSED_SAFE, Q2_CLOSED_SAFE, Q2_CLOSED_SAFE,
                         Q2_CLOSED_SAFE, 0.10, 0.10)
STATIC_D_GAP_TOL_MM = 0.01
STATIC_PROXY_TOL_MM = 0.01
STATIC_SAMPLES = 101
COMMAND_KEYPOINTS = (0, 25, 50, 75, 100)
FOLLOWER_JOINTS = (
    "R_hand_narrow2_joint", "R_hand_narrow3_joint", "R_hand_narrow_loop_joint",
    "R_hand_wide2_joint", "R_hand_wide3_joint", "R_hand_wide_loop_joint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_pose(percent: float) -> tuple[float, float, float]:
    """Return (fraction, q_CAD, right root) for the useful 0..100 command."""
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError("command percent must be finite and within [0, 100]")
    fraction = float(percent) / 100.0
    q_cad = Q_CLOSED + fraction * (Q_USEFUL_OPEN - Q_CLOSED)
    root = pipeline.PHASE - q_cad
    return fraction, q_cad, root


def mirrored_coefficients(narrow: np.ndarray) -> np.ndarray:
    """Mirror q_wide2(x) = -q_narrow2(-x) for the official root mimic."""
    return np.asarray([-((-1.0) ** i) * float(value)
                       for i, value in enumerate(narrow)], dtype=float)


def build_reference_path(high_static: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Join the existing Candidate B low branch to its D-closed open branch."""
    if high_static.get("status") != "PASS_STATIC_GEOMETRY_GATE":
        raise ValueError("the saved geometry-selected open branch is not a passing static input")
    if high_static.get("new_physics_steps") != 0:
        raise ValueError("the source high-open path must be static geometry evidence")
    if abs(float(high_static.get("start_q_CAD", math.nan)) - 0.20) > 1e-10:
        raise ValueError("the high-open path is not anchored at frozen Candidate B qCAD=0.20")
    law = pipeline.CandidateLaw()
    closed_value = float(law.at_q(Q_CLOSED)["R_hand_narrow2_joint"])
    correction = closed_value - Q2_CLOSED_SAFE
    samples: list[tuple[float, float, float]] = []
    for q_cad in np.linspace(Q_CLOSED, 0.20, 68):
        state = law.at_q(float(q_cad))
        # Fade a small, collision-screened correction to zero at the registered
        # source pose. This is a functional SIMULATION_ONLY choice, not a CAD mate.
        offset = correction * (-float(q_cad)) / (0.0 - Q_CLOSED) if q_cad < 0 else 0.0
        root = pipeline.PHASE - float(q_cad)
        samples.append((root, float(state["R_hand_narrow2_joint"] - offset), float(q_cad)))

    for row in high_static["static_rows"]:
        q_cad = float(row["q_CAD"])
        if q_cad < 0.20 - 1e-10 or q_cad > Q_USEFUL_OPEN + 1e-10:
            continue
        fitted_b = float(row.get("fitted_B_CAD", -float(row["B_CAD"])))
        samples.append((pipeline.PHASE - q_cad, -fitted_b, q_cad))

    roots = np.asarray([item[0] for item in samples], dtype=float)
    q2_values = np.asarray([item[1] for item in samples], dtype=float)
    if not np.all(np.isfinite(roots)) or not np.all(np.isfinite(q2_values)):
        raise ValueError("reference path contains non-finite data")

    # Preserve four useful anchors exactly: safe closed-side root, source pose,
    # frozen Candidate B qCAD=0.20, and the already-tested qCAD=0.55 open pose.
    anchor_q = (Q_CLOSED, 0.0, 0.20, Q_USEFUL_OPEN)
    anchor_y = (
        Q2_CLOSED_SAFE,
        0.0,
        float(law.at_q(0.20)["R_hand_narrow2_joint"]),
        float(high_static["static_rows"][-1]["fitted_B_CAD"]) * -1.0,
    )
    anchor_roots = np.asarray([pipeline.PHASE - q for q in anchor_q], dtype=float)
    design = np.polynomial.polynomial.polyvander(roots, 4)
    anchor_design = np.polynomial.polynomial.polyvander(anchor_roots, 4)

    safety_roots = np.asarray([
        -fraction * (Q_USEFUL_OPEN - Q_CLOSED)
        for fraction in COMMAND_SAFETY_FRACTIONS
    ], dtype=float)
    safety_design = np.polynomial.polynomial.polyvander(safety_roots, 4)

    # Minimize the worst deviation from the joined kinematic path while keeping
    # the first 6% inside the no-proxy-contact q2 envelope measured at root close.
    a_ub = np.vstack((
        np.column_stack((design, -np.ones(len(design)))),
        np.column_stack((-design, -np.ones(len(design)))),
        np.column_stack((safety_design, np.zeros(len(safety_design)))),
    ))
    b_ub = np.concatenate((q2_values, -q2_values, np.asarray(COMMAND_SAFETY_Q2_MAX)))
    a_eq = np.column_stack((anchor_design, np.zeros(len(anchor_design))))
    b_eq = np.asarray(anchor_y, dtype=float)
    result = linprog(
        c=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=[(None, None)] * 5 + [(0.0, None)], method="highs",
    )
    if not result.success:
        raise ValueError(f"could not fit a safe quartic functional coupling: {result.message}")
    coefficients = np.asarray(result.x[:5], dtype=float)
    max_fit_error = float(result.x[-1])
    return coefficients, mirrored_coefficients(coefficients), {
        "method": "minimax quartic fit with four exact anchors and six near-close proxy-clearance inequalities",
        "anchor_qCAD": list(anchor_q),
        "anchor_narrow2_urdf_rad": list(anchor_y),
        "near_close_command_fractions": list(COMMAND_SAFETY_FRACTIONS),
        "near_close_narrow2_upper_bounds_rad": list(COMMAND_SAFETY_Q2_MAX),
        "reference_sample_count": len(samples),
        "reference_qCAD_range": [float(min(x[2] for x in samples)), float(max(x[2] for x in samples))],
        "max_abs_fit_error_rad": max_fit_error,
        "classification": CLASSIFICATION,
        "limitations": [
            "The fitted relation is a functional simulation approximation, not a manufacturer transmission law.",
            "The close-side endpoint is a conservative tested proxy-clear pose and leaves nonzero jaw clearance.",
        ],
    }


def make_candidate_xml(narrow: np.ndarray, wide: np.ndarray) -> str:
    root = clearance.free_model_xml()
    root.set("model", "SIMULATION_ONLY_CANDIDATE_B_FULL_USEFUL_RANGE_QUARTIC")
    equality = root.find("equality")
    if equality is None:
        raise ValueError("Candidate B is missing its root mimic and D closure equalities")
    for name in ("candidate_full_open_narrow_q2", "candidate_full_open_wide_q2"):
        if any(item.get("name") == name for item in equality):
            raise ValueError(f"candidate equality {name} already exists")
    narrow_joint = ET.SubElement(
        equality, "joint", name="candidate_full_open_narrow_q2",
        joint1="R_hand_narrow2_joint", joint2="right_claw_joint",
        polycoef=" ".join(f"{float(x):.16g}" for x in narrow), solref="0.002 1",
    )
    narrow_joint.tail = "\n    "
    wide_joint = ET.SubElement(
        equality, "joint", name="candidate_full_open_wide_q2",
        joint1="R_hand_wide2_joint", joint2="R_hand_wide1_joint",
        polycoef=" ".join(f"{float(x):.16g}" for x in wide), solref="0.002 1",
    )
    wide_joint.tail = "\n  "
    return ET.tostring(root, encoding="unicode") + "\n"


def static_state(model, data, surfaces, q_cad: float, q2_narrow: float,
                 seed: np.ndarray) -> tuple[dict, np.ndarray]:
    b_cad = -float(q2_narrow)

    def residual(passive: np.ndarray) -> np.ndarray:
        clearance.apply_state(model, data, clearance.values_for_q(q_cad, b_cad, passive))
        return clearance.d_residual_mm(model, data)

    fit = least_squares(
        residual, seed, max_nfev=45, xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )
    if not fit.success and np.max(np.abs(fit.fun)) > STATIC_D_GAP_TOL_MM:
        raise ValueError(f"D closure solve failed at qCAD={q_cad:.9f}: {fit.message}")
    passive = np.asarray(fit.x, dtype=float)
    values = clearance.values_for_q(q_cad, b_cad, passive)
    clearance.apply_state(model, data, values)
    d_gap = float(np.max(np.abs(clearance.d_residual_mm(model, data))))
    contacts = travel.proxy_contacts(model, data)
    proxy_penetration = max((float(item["penetration_mm"]) for item in contacts), default=0.0)
    _, sections = surfaces.aperture(data)
    widths = [float(section["width_mm"]) for section in sections if section["width_mm"] is not None]
    minimum = min(widths) if len(widths) == len(sections) else None
    coupling_residual = abs(float(values["R_hand_narrow2_joint"]) - float(
        np.polynomial.polynomial.polyval(values["right_claw_joint"], CURRENT_COEFFICIENTS)
    ))
    mimic_residual = abs(float(values["R_hand_wide1_joint"]) + float(values["right_claw_joint"]))
    return {
        "qCAD": float(q_cad),
        "q_root_rad": float(values["right_claw_joint"]),
        "q2_narrow_rad": float(values["R_hand_narrow2_joint"]),
        "q3_narrow_rad": float(values["R_hand_narrow3_joint"]),
        "q_loop_narrow_rad": float(values["R_hand_narrow_loop_joint"]),
        "joint_positions_urdf_rad": values,
        "passive_CAD": passive.tolist(),
        "D_gap_mm": d_gap,
        "proxy_penetration_mm": proxy_penetration,
        "proxy_contacts": contacts,
        "five_section_widths_mm": widths,
        "min_five_section_mm": minimum,
        "coupling_residual_rad": coupling_residual,
        "root_mimic_residual_rad": mimic_residual,
    }, passive


CURRENT_COEFFICIENTS = np.asarray([
    0.10252605152311378, -0.052265209725381116,
    -7.725503403156838, -12.559461709155974,
    -2.6596551253583485,
], dtype=float)


def save_text(run: Path, name: str, value: str) -> None:
    pg.write_new_text(run / name, value, environment="wsl")


def save_json(run: Path, name: str, value: object) -> None:
    save_text(run, name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    clearance.verify_inputs()
    if not BASELINE_RUN.is_dir() or not BASE_TRAVEL.is_file() or not HIGH_STATIC.is_file():
        raise FileNotFoundError("the pinned Candidate B static/functional evidence is missing")
    if not HIGH_STATIC_MODEL.is_file() or not HIGH_CYLINDER_PROBES.is_file():
        raise FileNotFoundError("the saved high-open geometry-coupled evidence is incomplete")

    baseline_result = json.loads((BASELINE_RUN / "single_motor_result.json").read_text())
    baseline_integrity = json.loads((BASELINE_RUN / "source_integrity.json").read_text())
    travel_analysis = json.loads(BASE_TRAVEL.read_text())
    high_static = json.loads(HIGH_STATIC.read_text())
    prior_cylinders = json.loads(HIGH_CYLINDER_PROBES.read_text())
    if baseline_integrity.get("status") != "UNCHANGED":
        raise ValueError("Candidate B source-integrity record is not UNCHANGED")
    if baseline_result.get("active_actuator_count") != 1 or baseline_result.get("passive_qpos_writes_after_start") != 0:
        raise ValueError("the frozen Candidate B baseline does not match the required one-input contract")

    narrow_coeff, wide_coeff, fit_info = build_reference_path(high_static)
    # Fit output is deterministic; this cross-check prevents accidental use of
    # stale coefficients when the static reference data changes.
    if not np.allclose(narrow_coeff, CURRENT_COEFFICIENTS, rtol=0.0, atol=2e-10):
        raise ValueError("the saved static evidence now fits a different quartic; review before use")

    xml_text = make_candidate_xml(narrow_coeff, wide_coeff)
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    root_range = model.jnt_range[root_jid].astype(float).tolist()
    if model.nu != 1 or not np.allclose(root_range, [-1.0, 0.0], atol=1e-12):
        raise ValueError(f"actuator/root contract changed: nu={model.nu}, range={root_range}")
    derived = ET.parse(str(clearance.BASE_URDF)).getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    law = pipeline.CandidateLaw()
    seed = np.asarray([
        law.at_q(Q_CLOSED)["R_hand_narrow3_joint"],
        law.at_q(Q_CLOSED)["R_hand_narrow_loop_joint"],
    ], dtype=float)

    samples = []
    max_d_gap = 0.0
    max_proxy_penetration = 0.0
    max_coupling_residual = 0.0
    max_root_mimic_residual = 0.0
    max_follower_step = 0.0
    follower_step_by_name = {name: 0.0 for name in FOLLOWER_JOINTS}
    previous_values: dict[str, float] | None = None
    source_sections_dips = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, STATIC_SAMPLES)):
        percent = float(fraction * 100.0)
        _, q_cad, root_target = command_pose(percent)
        q2 = float(np.polynomial.polynomial.polyval(root_target, narrow_coeff))
        point, seed = static_state(model, data, surfaces, q_cad, q2, seed)
        if abs(point["q_root_rad"] - root_target) > 1e-12:
            raise ValueError("static root does not match the normalized command mapping")
        if point["q_root_rad"] < root_range[0] - 1e-12 or point["q_root_rad"] > root_range[1] + 1e-12:
            raise ValueError(f"root limit exceeded at command {percent:.2f}%")
        max_d_gap = max(max_d_gap, point["D_gap_mm"])
        max_proxy_penetration = max(max_proxy_penetration, point["proxy_penetration_mm"])
        max_coupling_residual = max(max_coupling_residual, point["coupling_residual_rad"])
        max_root_mimic_residual = max(max_root_mimic_residual, point["root_mimic_residual_rad"])
        if previous_values is not None:
            for name in FOLLOWER_JOINTS:
                delta = abs(point["joint_positions_urdf_rad"][name] - previous_values[name])
                follower_step_by_name[name] = max(follower_step_by_name[name], delta)
                max_follower_step = max(max_follower_step, delta)
        if samples:
            before = samples[-1]["five_section_widths_mm"]
            after = point["five_section_widths_mm"]
            for section_index, (a, b) in enumerate(zip(before, after)):
                if b < a - 0.1:
                    source_sections_dips.append({
                        "command_percent": percent,
                        "qCAD": q_cad,
                        "section_index": section_index,
                        "previous_width_mm": a,
                        "current_width_mm": b,
                        "delta_mm": b - a,
                    })
        point["command_percent"] = percent
        point["command_fraction"] = float(fraction)
        samples.append(point)
        previous_values = point["joint_positions_urdf_rad"]

    # Exact source visual STL triangle-crossing checks at the five requested
    # normalized states, compared with the registered qCAD=0 source pose.
    source_reference_values = law.at_q(0.0)
    clearance.apply_state(model, data, source_reference_values)
    source_reference = travel.SourceSurfaces(model, derived).interference(data)
    reference_pair_names = {
        tuple(sorted((pair["body1"], pair["body2"]))) for pair in source_reference
    }
    original_closed_values = law.at_q(Q_CLOSED)
    clearance.apply_state(model, data, original_closed_values)
    original_closed_contacts = travel.proxy_contacts(model, data)
    original_closed_gap = float(np.max(np.abs(clearance.d_residual_mm(model, data))))
    _, original_closed_sections = surfaces.aperture(data)
    original_closed_widths = [
        float(section["width_mm"]) for section in original_closed_sections
        if section["width_mm"] is not None
    ]
    mesh_checks = []
    for percent in COMMAND_KEYPOINTS:
        point = samples[percent]
        clearance.apply_state(model, data, point["joint_positions_urdf_rad"])
        pairs = surfaces.interference(data)
        pair_names = {tuple(sorted((pair["body1"], pair["body2"]))) for pair in pairs}
        new_names = sorted(pair_names - reference_pair_names)
        triangles = {side: surfaces.triangles(data, surfaces.gid(side)) for side in pipeline.GROUPS}
        band = {}
        for diameter in (65.0, 70.0, 75.0):
            measured = functional.cylinder_band_fit(
                triangles["narrow"], triangles["wide"], diameter, 0.115,
                x_samples=11, z_samples=3,
            )
            band[str(int(diameter))] = {
                "status": measured["status"],
                "clearance_mm": measured["clearance_mm"],
                "center_y_mm": measured.get("best_center_y_mm"),
                "band_height_mm": functional.GRIP_BAND_HEIGHT_MM,
            }
        point["band_clearance_mm"] = band
        mesh_checks.append({
            "command_percent": percent,
            "source_pair_count": len(pair_names),
            "source_pair_names": [list(name) for name in sorted(pair_names)],
            "new_pair_names_vs_source_qCAD0": [list(name) for name in new_names],
        })

    travel_rows = travel_analysis.get("rows", [])
    last_baseline = max(
        (row for row in travel_rows if float(row["q_CAD"]) <= Q_USEFUL_OPEN),
        key=lambda row: float(row["q_CAD"]),
    )
    old_open_sections = [
        float(item["width_mm"])
        for item in last_baseline["surface_sections"]
        if item["width_mm"] is not None
    ]
    prior_high_row = high_static["static_rows"][-1]
    endpoint_reference = clearance.values_for_q(
        float(prior_high_row["q_CAD"]),
        float(prior_high_row["fitted_B_CAD"]),
        np.asarray(prior_high_row["passive_CAD"], dtype=float),
    )
    endpoint_delta = max(
        abs(samples[-1]["joint_positions_urdf_rad"][name] - endpoint_reference[name])
        for name in endpoint_reference
    )
    if endpoint_delta > 1e-8:
        raise ValueError("the selected open endpoint no longer matches saved high-open geometry")
    reused_full_height = [
        {"diameter_mm": item["diameter_mm"], "height_mm": item["height_mm"],
         "status": item["status"]}
        for item in prior_cylinders["full_height_cylinders"]
    ]

    run = pg.create_unique_output_directory(
        EVIDENCE_ROOT, prefix="candidate-b-full-useful-open-static", environment="wsl"
    ).native_path
    save_text(run, "candidate_b_full_useful_range_static.xml", xml_text)
    base_text = clearance.BASE_MJCF.read_text(encoding="utf-8")
    save_text(run, "candidate_b_full_useful_range_model.diff", "".join(difflib.unified_diff(
        base_text.splitlines(keepends=True), xml_text.splitlines(keepends=True),
        fromfile=str(clearance.BASE_MJCF), tofile="candidate_b_full_useful_range_static.xml",
    )))
    save_json(run, "static_range.json", {
        "schema": "robotsim.issue46.candidate-b-full-useful-open-static.v1",
        "classification": CLASSIFICATION,
        "new_physics_steps": 0,
        "mujoco_version": mujoco.__version__,
        "actuator_count": model.nu,
        "official_root_joint_range_rad": root_range,
        "normalized_command_mapping": {
            "command_fraction": "u in [0,1]",
            "right_claw_joint_target_rad": f"{- (Q_USEFUL_OPEN-Q_CLOSED):.16g} * u",
            "qCAD": f"{Q_CLOSED:.16g} + u * {Q_USEFUL_OPEN-Q_CLOSED:.16g}",
            "range_note": "0..100% is this experimental useful range; it uses only part of the official root joint range and does not claim manufacturer full travel",
        },
        "coupling": {
            "narrow_joint1": "R_hand_narrow2_joint",
            "narrow_joint2": "right_claw_joint",
            "narrow_polycoef": narrow_coeff.tolist(),
            "wide_joint1": "R_hand_wide2_joint",
            "wide_joint2": "R_hand_wide1_joint",
            "wide_polycoef": wide_coeff.tolist(),
            **fit_info,
        },
        "frozen_candidate_b": {
            "baseline_run": str(BASELINE_RUN),
            "baseline_model_sha256": sha256(clearance.BASE_MJCF),
            "baseline_physics_status": baseline_result["status"],
            "baseline_physical_qCAD_range": baseline_result["actual_interval_CAD"],
            "source_integrity_record": baseline_integrity,
            "preserved": True,
        },
        "closed_endpoint_comparison": {
            "original_candidate_b_relation_at_official_root_zero": {
                "q_root_rad": float(original_closed_values["right_claw_joint"]),
                "q2_narrow_rad": float(original_closed_values["R_hand_narrow2_joint"]),
                "D_gap_mm": original_closed_gap,
                "max_proxy_penetration_mm": max(
                    (float(item["penetration_mm"]) for item in original_closed_contacts),
                    default=0.0,
                ),
                "proxy_contacts": original_closed_contacts,
                "five_section_widths_mm": original_closed_widths,
            },
            "new_quartic_endpoint": {
                "q_root_rad": samples[0]["q_root_rad"],
                "q2_narrow_rad": samples[0]["joint_positions_urdf_rad"]["R_hand_narrow2_joint"],
                "D_gap_mm": samples[0]["D_gap_mm"],
                "proxy_penetration_mm": samples[0]["proxy_penetration_mm"],
                "minimum_five_section_mm": samples[0]["min_five_section_mm"],
            },
            "qualification": "The original Candidate B endpoint is a static geometry probe; the new endpoint is the selected proxy-clear quartic candidate. Neither proves physical dynamic closure.",
        },
        "static_full_useful_range": {
            "sample_count": len(samples),
            "max_D_gap_mm": max_d_gap,
            "max_proxy_penetration_mm": max_proxy_penetration,
            "max_coupling_residual_rad": max_coupling_residual,
            "max_root_mimic_residual_rad": max_root_mimic_residual,
            "max_follower_step_rad_per_1pct_command": max_follower_step,
            "max_follower_step_by_joint_rad": follower_step_by_name,
            "section_minimum_dips_over_0p1mm": source_sections_dips,
            "five_command_points": [samples[percent] for percent in COMMAND_KEYPOINTS],
            "source_mesh_checks_at_five_points": mesh_checks,
            "registered_source_pair_count_at_qCAD0": len(reference_pair_names),
            "endpoint_delta_from_existing_qCAD0p55_geometry_rad": endpoint_delta,
            "reused_same-pose_155mm_full_height_cylinder_checks": reused_full_height,
        },
        "baseline_linear_candidate_boundary": travel_analysis["boundaries"]["opening"],
        "candidate_b_original_root_limits": travel_analysis["official_root_limits"],
        "root_joint_limits_changed": False,
        "source_geometry_changed": False,
        "collision_exclusions_added": 0,
        "passive_joint_limits_changed": False,
        "physics_steps": 0,
        "limitations": [
            "The 0% command is at the official closed-side root limit but leaves the measured inner-surface gap reported at the five sections; it is not a zero-gap closure.",
            "The quartic coupling is a static geometry-selected functional overlay, not manufacturer transmission data.",
            "No new dynamic stability or physical open-close test was run for this quartic overlay.",
            "Original STL triangle checks compare only the five requested command points against the qCAD=0 source-pose pair set; baseline mating/contact pairs remain unclassified.",
            "The official root joint has additional unused travel beyond the selected 100% useful endpoint; it was not extrapolated.",
        ],
    })

    csv_path = run / "normalized_static_samples.csv"
    csv_lines = ["command_percent,qCAD,right_claw_joint_rad,narrow2_rad,narrow3_rad,narrow_loop_rad,min_five_section_mm,D_gap_mm,proxy_penetration_mm,coupling_residual_rad,root_mimic_residual_rad\n"]
    for point in samples:
        values = point["joint_positions_urdf_rad"]
        csv_lines.append(
            f"{point['command_percent']:.2f},{point['qCAD']:.12g},{point['q_root_rad']:.12g},"
            f"{values['R_hand_narrow2_joint']:.12g},{values['R_hand_narrow3_joint']:.12g},"
            f"{values['R_hand_narrow_loop_joint']:.12g},{point['min_five_section_mm']:.9g},"
            f"{point['D_gap_mm']:.9g},{point['proxy_penetration_mm']:.9g},"
            f"{point['coupling_residual_rad']:.9g},{point['root_mimic_residual_rad']:.9g}\n"
        )
    save_text(run, csv_path.name, "".join(csv_lines))

    old_min = min(old_open_sections)
    opening_boundary = travel_analysis["boundaries"]["opening"]
    boundary_lo, boundary_hi = opening_boundary["analytic_reachability_boundary_q_bracket"]
    original_closed_max_penetration = max(
        (float(item["penetration_mm"]) for item in original_closed_contacts), default=0.0
    )
    original_closed_pair_names = sorted({
        " × ".join(item["geoms"]) for item in original_closed_contacts
    })
    point_rows = []
    for point in [samples[percent] for percent in COMMAND_KEYPOINTS]:
        bands = point["band_clearance_mm"]
        point_rows.append(
            f"| {point['command_percent']:.0f}% | {point['q_root_rad']:.4f} | {point['qCAD']:.4f} | "
            f"{point['min_five_section_mm']:.2f} | {bands['65']['clearance_mm']:.2f} / "
            f"{bands['70']['clearance_mm']:.2f} / {bands['75']['clearance_mm']:.2f} | "
            f"{point['D_gap_mm']:.6g} | {point['proxy_penetration_mm']:.3f} |"
        )
    report = f"""# Issue #46 — Candidate B Full Useful Opening (Static Only)

Status: `STATIC_OPEN_RANGE_PASS_WITH_CLOSE_GAP_AND_DYNAMIC_GATES_OPEN`

## Diagnosis

Frozen Candidate B keeps a linear tendon-rate equality on each side. Its analytic real-closure boundary is bracketed at `qCAD=[{boundary_lo:.16g}, {boundary_hi:.16g}]` rad: the last saved valid grid point is `qCAD={travel_analysis['boundaries']['opening']['last_grid_valid_q']:.3f}` ({old_min:.2f} mm minimum of five source-STL sections), while the next grid point has no real candidate closure. The root at that boundary is about `{pipeline.PHASE-travel_analysis['boundaries']['opening']['last_grid_valid_q']:.3f}` rad, well inside the official `[-1, 0]` range. At the last valid grid point the D gap is below 0.002 mm and the collision proxies report no penetration, so the limiting condition is the Candidate B coupling/closure equation, not root limits, the D residual tolerance, or collision at that opening endpoint.

## Minimal functional overlay

Candidate B remains unchanged. The separate MJCF overlay removes only the two per-side Candidate B tendon-rate equalities and replaces them with one mirrored quartic `q_narrow2 = P4(q_root)` / `q_wide2 = -P4(-q_wide_root)` relationship. It retains the official root joint/mimic, D-point connect constraints, original URDF/STL visual and collision meshes, existing four collision exclusions, and single root actuator. This is a SIMULATION_ONLY geometry-selected function, not a manufacturer transmission. The fit's maximum deviation from its joined reference path is `{fit_info['max_abs_fit_error_rad']:.5f} rad`.

The command mapping is `right_claw_joint = {- (Q_USEFUL_OPEN-Q_CLOSED):.12g} * u`, `u ∈ [0,1]`. `u=1` is a useful 75 mm bottle opening endpoint at `qCAD={Q_USEFUL_OPEN:.2f}`, not the manufacturer's maximum travel; the official root still has unused range to `-1 rad`.

## Static 0/25/50/75/100% measurements

Jaw width is measured from the original visual STL inner surfaces at five heights (110, 112.5, 115, 117.5, 120 mm); the table reports the narrowest section. Band values are static 10 mm-high 65/70/75 mm cylinder clearances from the same source meshes.

| Command | Root q (rad) | qCAD | Minimum 5-section width (mm) | 65/70/75 mm band clearance (mm) | Max D gap (mm) | Proxy penetration (mm) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(point_rows)}

Across 101 static samples, maximum D-site gap is `{max_d_gap:.8g} mm`, maximum collision-proxy penetration is `{max_proxy_penetration:.3f} mm`, root and mimic residuals remain within numerical tolerance, and the largest per-1%-command follower increment is `{max_follower_step:.5f} rad`. The required 5 original-STL triangle checks add no pair names relative to the registered `qCAD=0` source pose; the source pose itself has `{len(reference_pair_names)}` baseline triangle contact/intersection pair names that remain unclassified. The endpoint exactly matches the prior qCAD=0.55 geometry pose (joint-state delta `{endpoint_delta:.3g} rad`), where the existing static 155 mm full-height 65/70/75 mm cylinder probes passed. That full-height result is reused evidence for the identical end pose, not a new object-dynamics test.

## Remaining limits

- At `u=0`, the root is at its official closed-side limit, but the minimum measured jaw section remains `{samples[0]['min_five_section_mm']:.2f} mm`; the model is not fully closed. At that same root-zero pose, the original Candidate B follower relation uses q2 `{original_closed_values['R_hand_narrow2_joint']:.6f} rad` and produces `{original_closed_max_penetration:.6f} mm` maximum proxy penetration in `{'; '.join(original_closed_pair_names)}`. This overlay uses a proxy-clear endpoint instead of hiding contact.
- The five-section minimum has a localized `{len(source_sections_dips)}` dip(s) larger than 0.1 mm between the 101 static samples; see `static_range.json`. Section width is not strictly monotonic at every height as the jaws rotate. At the requested keypoints it increases from `{samples[0]['min_five_section_mm']:.2f}` to `{samples[-1]['min_five_section_mm']:.2f} mm`.
- No `mj_step` was run for the quartic overlay. The prior physical Candidate B and prior quadratic-coupling runs do not prove stability of this new law. A one-root-motor open-hold-close-hold proposal is recorded in `MODEL_ASSUMPTIONS.md`.
- Full manufacturer travel, a zero-gap physical closure, dynamic stability of this quartic, and grasp/contact remain unvalidated.

## Reproduction

```bash
cd {Path(__file__).resolve().parents[2]}
PYTHONPATH=scripts/research ~/.local/bin/uv run --offline --with mujoco==3.3.6 --with scipy --with numpy --with pillow python scripts/research/issue46_candidate_b_full_useful_open_static.py
```

The command performs static `mj_forward` kinematics and mesh/collision queries only; it never calls `mj_step`.
"""
    save_text(run, "REPORT.md", report)

    assumptions = f"""# Candidate B Full Useful-Range Overlay Assumptions

Classification: `{CLASSIFICATION}`

- Exact overlay change: remove Candidate B's two `candidate_*_rate_eq` tendon equalities and replace them with the two quartic `candidate_full_open_*_q2` joint equalities in `candidate_b_full_useful_range_static.xml`.
- Narrow coefficients: `{", ".join(f"{x:.16g}" for x in narrow_coeff)}`.
- Wide mirrored coefficients: `{", ".join(f"{x:.16g}" for x in wide_coeff)}`.
- The quartic was fit to the Candidate B closed-to-qCAD=0.20 kinematic path plus the existing D-closed qCAD=0.20..0.55 static path. Six near-close inequalities keep the first six percent at or below the tested collision-clear q2 values.
- Existing official root mimic, official root limits, CAD-measured joint origins/axes, D-point connect equalities, source STL visual/collision assets, inertias, actuator, and four inherited collision exclusions are preserved.
- One actuator remains: `servo_right_claw_joint`. There are no passive actuators. The offline static solver calculates follower coordinates only for measurement; no dynamic stepping or runtime passive-qpos writes were performed.
- The safe close-side q2 endpoint is `R_hand_narrow2_joint={Q2_CLOSED_SAFE:.9f} rad` at `right_claw_joint=0`, with a measured minimum section gap of `{samples[0]['min_five_section_mm']:.3f} mm`. This is not zero-gap closure.
- At the same official root-zero pose, the original Candidate B relation uses q2 `{original_closed_values['R_hand_narrow2_joint']:.9f} rad` and produces `{max((float(item['penetration_mm']) for item in original_closed_contacts), default=0.0):.6f} mm` maximum collision-proxy penetration in the middle-link/loop pairs. The selected quartic endpoint avoids those proxy contacts while retaining the measured nonzero gap.
- The useful open endpoint is qCAD `{Q_USEFUL_OPEN:.3f}` / root `{samples[-1]['q_root_rad']:.9f} rad`, with minimum five-section source-STL width `{samples[-1]['min_five_section_mm']:.3f} mm`. This endpoint reuses the saved full-height static cylinder pose, with a `{endpoint_delta:.3g} rad` joint-state delta.

## Proposed bounded physical validation (not run)

After separate authorization, load only this exact derived MJCF in MuJoCo 3.3.6 with gravity on, the base fixed, the single existing root position actuator (`kp=0.5`, `kv=0.05`, torque cap ±0.5 Nm, same root-only model-bias compensation as the prior candidate), and no object. Map command `u` to root `-0.6841259890668666*u`; ramp 0→100%→hold 0.5 s→0%→hold 0.5 s. Do not write passive qpos. Stop on D gap >0.05 mm, any proxy penetration beyond the model's existing source-pose contacts, root-limit violation, non-finite state, speed >5 rad/s, or coupling residual >0.01 rad. This is a proposal only; no dynamic pass is claimed.
"""
    save_text(run, "MODEL_ASSUMPTIONS.md", assumptions)
    save_text(
        run, "run_command.txt",
        f"cd {Path(__file__).resolve().parents[2]}\n"
        "PYTHONPATH=scripts/research ~/.local/bin/uv run --offline --with mujoco==3.3.6 "
        "--with scipy --with numpy --with pillow python "
        "scripts/research/issue46_candidate_b_full_useful_open_static.py\n",
    )
    save_json(run, "run_context.json", {
        "task_id": "codex-issue46-candidate-b-full-useful-open-2026-10-23",
        "attempt_id": str(uuid.uuid4()),
        "branch": "codex/issue46-full-open-static-20261023",
        "base_head": "59ba71cf50e1aff0c4d5002a55087a7c7ff7060b",
        "new_physics_steps": 0,
    })
    clearance.verify_inputs()
    manifest = pg.write_artifact_manifest(run, environment="wsl")
    print(f"EVIDENCE_DIR={run}")
    print(f"ARTIFACT_COUNT={len(manifest['artifacts'])}")
    print(f"STATIC_KINEMATIC_PROXY_CHECK={'PASS' if max_d_gap <= STATIC_D_GAP_TOL_MM and max_proxy_penetration <= STATIC_PROXY_TOL_MM else 'FAIL'}")
    print(f"CLOSE_GAP_MM={samples[0]['min_five_section_mm']:.6f}")
    print(f"OPEN_MIN5_MM={samples[-1]['min_five_section_mm']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
