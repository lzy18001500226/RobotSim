#!/usr/bin/env python3
"""Geometry-selected, single-motor continuation for frozen Candidate B.

The candidate is UNVERIFIED_EXPERIMENTAL. It keeps the official X2 assets,
joint frames, root limits, mimic, D closures, actuator, and collision setup. It
replaces Candidate B's fixed rate relation in a separate copy with a passive
relation selected by D closure, STL aperture, continuity, and collision gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import path_governance as pg
import issue46_urdf_pipeline as p
import issue46_single_motor_travel as travel
import issue46_candidate_b_functional_validation as functional
from issue46_dpoint_legal_motion import TriangleBVH

EVIDENCE_ROOT = Path(p.REPO) / "reviews/issue-46"
BASE_RUN = EVIDENCE_ROOT / "single-motor-full-travel-20261022T084103Z"
BASE_MJCF = BASE_RUN / "single_motor.xml"
BASE_URDF = BASE_RUN / "X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf"
START_Q = 0.20
Q_STEP = 0.01
SEARCH_RADIUS = 0.025
SEARCH_STEP = 0.005
MAX_PASSIVE_STEP = 0.06
STATIC_D_GAP_MM = 0.01
STATIC_PROXY_PENETRATION_MM = 0.01
ROOT_KP = 0.5
ROOT_KV = 0.05
ROOT_FORCE_NM = 0.5
MAX_JOINT_SPEED_RAD_S = 5.0
STEP_SOURCE = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_op1_cad/OP1-官网模型.STEP")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs() -> None:
    official = p.UPSTREAM / "X2-Ultra_omnipicker.urdf"
    expected = {
        official: p.URDF_HASH,
        STEP_SOURCE: "5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b",
        p.SAMPLES: p.SAMPLES_HASH,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input hash mismatch: {path}")
    integrity = json.loads((BASE_RUN / "source_integrity.json").read_text())
    if integrity.get("status") != "UNCHANGED":
        raise ValueError("frozen Candidate B source-integrity record is not UNCHANGED")
    for filename, digest in integrity["STL_sha256"].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"official STL hash mismatch: {Path(filename).name}")


def write_json(run: Path, name: str, value: object) -> None:
    pg.write_new_text(run / name, json.dumps(value, indent=2, allow_nan=False) + "\n", environment="wsl")


def values_for_q(q: float, b_cad: float, passive_cad: np.ndarray) -> dict[str, float]:
    root = p.PHASE - q
    return {
        "right_claw_joint": root,
        "R_hand_wide1_joint": -root,
        "R_hand_narrow2_joint": -b_cad,
        "R_hand_wide2_joint": b_cad,
        "R_hand_narrow3_joint": float(passive_cad[0]),
        "R_hand_wide3_joint": -float(passive_cad[0]),
        "R_hand_narrow_loop_joint": float(passive_cad[1]),
        "R_hand_wide_loop_joint": -float(passive_cad[1]),
    }


def apply_state(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, float]) -> None:
    addresses = p.addresses(model)
    for name, value in values.items():
        data.qpos[addresses[name]] = value
    mujoco.mj_forward(model, data)


def d_residual_mm(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    residuals = []
    for side in ("narrow", "wide"):
        jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_D_jaw")
        loop = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_D_loop")
        residuals.extend((data.site_xpos[jaw] - data.site_xpos[loop]) * 1000.0)
    return np.asarray(residuals, dtype=float)


def free_model_xml() -> ET.Element:
    root = ET.parse(BASE_MJCF).getroot()
    root.set("model", "UNVERIFIED_EXPERIMENTAL_GEOMETRY_CONTINUATION_FREE")
    equality = root.find("equality")
    if equality is None:
        raise ValueError("Candidate B model lacks its equality group")
    for item in list(equality):
        if item.tag == "tendon":
            equality.remove(item)
    tendon = root.find("tendon")
    if tendon is None:
        raise ValueError("Candidate B model lacks its fixed-tendon relation")
    root.remove(tendon)
    return root


def seed_at_start(model, data, law, surfaces):
    targets = law.at_q(START_Q)
    b_cad = -float(targets["R_hand_narrow2_joint"])
    passive = np.asarray([
        float(targets["R_hand_narrow3_joint"]),
        float(targets["R_hand_narrow_loop_joint"]),
    ])
    apply_state(model, data, targets)
    gap = float(np.max(np.abs(d_residual_mm(model, data))))
    width, sections = surfaces.aperture(data)
    triangles = {side: surfaces.triangles(data, surfaces.gid(side)) for side in p.GROUPS}
    cylinder = functional.cylinder_band_fit(
        triangles["narrow"], triangles["wide"], 65.0, 0.115, x_samples=11, z_samples=3
    )
    contacts = travel.proxy_contacts(model, data)
    penetration = max((float(c["penetration_mm"]) for c in contacts), default=0.0)
    if gap > STATIC_D_GAP_MM or penetration > STATIC_PROXY_PENETRATION_MM:
        raise ValueError(f"frozen Candidate B start pose fails preflight: D={gap:.6g} mm, proxy={penetration:.6g} mm")
    return b_cad, passive, float(width), sections, cylinder


def solve_passive_state(model, data, surfaces, q, b_cad, seed):
    def residual(passive):
        apply_state(model, data, values_for_q(q, b_cad, passive))
        return d_residual_mm(model, data)

    fit = least_squares(
        residual,
        seed,
        max_nfev=45,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    values = values_for_q(q, b_cad, fit.x)
    apply_state(model, data, values)
    gap = float(np.max(np.abs(d_residual_mm(model, data))))
    contacts = travel.proxy_contacts(model, data)
    penetration = max((float(c["penetration_mm"]) for c in contacts), default=0.0)
    width, sections = surfaces.aperture(data)
    valid_widths = [float(s["width_mm"]) for s in sections if s["width_mm"] is not None]
    minimum_section = min(valid_widths) if len(valid_widths) == len(sections) else None
    triangles = {side: surfaces.triangles(data, surfaces.gid(side)) for side in p.GROUPS}
    cylinder65 = functional.cylinder_band_fit(
        triangles["narrow"], triangles["wide"], 65.0, 0.115, x_samples=11, z_samples=3
    )
    return {
        "q_CAD": float(q),
        "B_CAD": float(b_cad),
        "passive_CAD": [float(fit.x[0]), float(fit.x[1])],
        "D_gap_mm": gap,
        "proxy_penetration_mm": penetration,
        "proxy_contacts": contacts,
        "section_aperture_mm": None if width is None else float(width),
        "minimum_sampled_section_aperture_mm": minimum_section,
        "65mm_band_clearance_mm": cylinder65["clearance_mm"],
        "65mm_band_pose_mm": [0.0, cylinder65.get("best_center_y_mm"), 115.0],
        "65mm_band_status": cylinder65["status"],
        "sections": sections,
        "optimizer_cost": float(fit.cost),
    }


def static_preflight(run: Path, q_scan: float) -> dict:
    free_xml = free_model_xml()
    free_path = run / "candidate_geometry_free.xml"
    free_text = ET.tostring(free_xml, encoding="unicode") + "\n"
    if free_path.exists():
        if free_path.read_text(encoding="utf-8") != free_text:
            raise ValueError("existing preflight model differs; refusing overwrite")
    else:
        pg.write_new_text(run / free_path.name, free_text, environment="wsl")
    model = mujoco.MjModel.from_xml_path(str(free_path))
    data = mujoco.MjData(model)
    derived = ET.parse(BASE_URDF).getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    law = p.CandidateLaw()
    b_cad, passive, aperture, start_sections, start_cylinder65 = seed_at_start(model, data, law, surfaces)
    start_widths = [float(s["width_mm"]) for s in start_sections if s["width_mm"] is not None]
    start_minimum_section = min(start_widths) if len(start_widths) == len(start_sections) else None
    rows = [{
        "q_CAD": START_Q,
        "B_CAD": b_cad,
        "passive_CAD": [float(passive[0]), float(passive[1])],
        "D_gap_mm": float(np.max(np.abs(d_residual_mm(model, data)))),
        "proxy_penetration_mm": 0.0,
        "proxy_contacts": [],
        "section_aperture_mm": aperture,
        "minimum_sampled_section_aperture_mm": start_minimum_section,
        "65mm_band_clearance_mm": start_cylinder65["clearance_mm"],
        "65mm_band_pose_mm": [0.0, start_cylinder65.get("best_center_y_mm"), 115.0],
        "65mm_band_status": start_cylinder65["status"],
        "sections": start_sections,
        "optimizer_cost": 0.0,
        "selection": "frozen Candidate B starting state",
    }]
    previous_values = values_for_q(START_Q, b_cad, passive)
    previous_aperture = start_minimum_section if start_minimum_section is not None else aperture
    previous_cylinder_clearance = float(start_cylinder65["clearance_mm"] or -math.inf)
    stop_reason = "STATIC_SCAN_TARGET_REACHED"
    q = START_Q + Q_STEP
    while q <= q_scan + 1e-10:
        candidates = []
        for b_value in np.arange(b_cad - SEARCH_RADIUS, b_cad + SEARCH_RADIUS + SEARCH_STEP * 0.5, SEARCH_STEP):
            row = solve_passive_state(model, data, surfaces, q, float(b_value), passive)
            if row["D_gap_mm"] > STATIC_D_GAP_MM:
                continue
            if row["proxy_penetration_mm"] > STATIC_PROXY_PENETRATION_MM:
                continue
            if row["minimum_sampled_section_aperture_mm"] is None:
                continue
            values = values_for_q(q, float(b_value), np.asarray(row["passive_CAD"]))
            max_step = max(abs(values[n] - previous_values[n]) for n in values if n not in ("right_claw_joint", "R_hand_wide1_joint"))
            if max_step > MAX_PASSIVE_STEP:
                continue
            if row["minimum_sampled_section_aperture_mm"] < previous_aperture - 0.02:
                continue
            if row["65mm_band_clearance_mm"] is None:
                continue
            if row["65mm_band_clearance_mm"] < previous_cylinder_clearance - 0.1:
                continue
            row["max_passive_step_rad"] = float(max_step)
            row["selection"] = "maximize measured 65 mm gripping-band clearance among nearby continuous D-closed states"
            row["_values"] = values
            candidates.append(row)
        if not candidates:
            stop_reason = "NO_NEARBY_CONTINUOUS_D_CLOSED_APERTURE_INCREASE"
            break
        selected = max(candidates, key=lambda r: (r["65mm_band_clearance_mm"],
                                                  r["minimum_sampled_section_aperture_mm"],
                                                  -r["max_passive_step_rad"]))
        rows.append({k: v for k, v in selected.items() if k != "_values"})
        b_cad = selected["B_CAD"]
        passive = np.asarray(selected["passive_CAD"], dtype=float)
        previous_values = selected["_values"]
        previous_aperture = selected["minimum_sampled_section_aperture_mm"]
        previous_cylinder_clearance = selected["65mm_band_clearance_mm"]
        q = round(q + Q_STEP, 10)

    if len(rows) < 3:
        raise ValueError(f"geometry continuation too short: {stop_reason}")
    root_n = np.asarray([p.PHASE - r["q_CAD"] for r in rows], dtype=float)
    b_n = -np.asarray([r["B_CAD"] for r in rows], dtype=float)
    root0, b0 = float(root_n[0]), float(b_n[0])
    dx = root_n - root0
    slope, curvature = np.linalg.lstsq(
        np.column_stack((dx, dx * dx)), b_n - b0, rcond=None
    )[0]
    rel_poly = np.polynomial.Polynomial([b0, float(slope), float(curvature)])
    coeff_n = rel_poly(np.polynomial.Polynomial([-root0, 1.0])).coef
    coeff_w = np.asarray([-coeff_n[0], coeff_n[1], -coeff_n[2]], dtype=float)
    err_n = float(np.max(np.abs(np.polynomial.polynomial.polyval(root_n, coeff_n) - b_n)))
    err_w = err_n
    coupling = {
        "narrow": [float(x) for x in coeff_n],
        "wide": [float(x) for x in coeff_w],
        "max_fit_error_rad": max(err_n, err_w),
        "fit_degree": 2,
        "anchored_at": {"q_CAD": START_Q, "right_claw_joint_rad": root0, "narrow2_joint_rad": b0},
        "equations": [
            "R_hand_narrow2_joint = polyval(right_claw_joint, narrow_coefficients)",
            "R_hand_wide2_joint = polyval(R_hand_wide1_joint, wide_coefficients)",
        ],
        "classification": "UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING",
        "selection_rule": "continue the D-closed branch from frozen Candidate B qCAD=0.20; among nearby continuous states with no collision-proxy penetration and nondecreasing original-STL aperture, maximize the measured 65 mm gripping-band clearance",
    }
    fitted_rows = []
    passive_seed = np.asarray(rows[0]["passive_CAD"], dtype=float)
    for raw in rows:
        q = float(raw["q_CAD"])
        root = p.PHASE - q
        b_urdf = float(np.polynomial.polynomial.polyval(root, coeff_n))
        fitted = solve_passive_state(model, data, surfaces, q, -b_urdf, passive_seed)
        fitted["fitted_B_CAD"] = -b_urdf
        fitted["raw_path_B_CAD"] = float(raw["B_CAD"])
        fitted["raw_path_B_fit_delta_rad"] = fitted["fitted_B_CAD"] - fitted["raw_path_B_CAD"]
        fitted["fitted_path_valid"] = (
            fitted["D_gap_mm"] <= STATIC_D_GAP_MM
            and fitted["proxy_penetration_mm"] <= STATIC_PROXY_PENETRATION_MM
        )
        fitted_rows.append(fitted)
        passive_seed = np.asarray(fitted["passive_CAD"], dtype=float)
        if not fitted["fitted_path_valid"]:
            break
    fitted_complete = len(fitted_rows) == len(rows) and all(r["fitted_path_valid"] for r in fitted_rows)
    fitted_monotonic = all(
        (b["minimum_sampled_section_aperture_mm"] or -math.inf)
        >= (a["minimum_sampled_section_aperture_mm"] or -math.inf) - 0.02
        for a, b in zip(fitted_rows, fitted_rows[1:])
    )
    max_row = max(fitted_rows, key=lambda r: r["minimum_sampled_section_aperture_mm"] or -math.inf)
    q65 = next((r["q_CAD"] for r in fitted_rows if (r["65mm_band_clearance_mm"] or -math.inf) >= 0.0), None)
    q70 = next((r["q_CAD"] for r in fitted_rows if (r["minimum_sampled_section_aperture_mm"] or 0.0) >= 70.0), None)
    q75 = next((r["q_CAD"] for r in fitted_rows if (r["minimum_sampled_section_aperture_mm"] or 0.0) >= 75.0), None)
    exact_checks = []
    baseline_pairs = set()
    for row in fitted_rows:
        apply_state(model, data, values_for_q(row["q_CAD"], row["B_CAD"], np.asarray(row["passive_CAD"])))
        pairs = surfaces.interference(data)
        pair_names = {tuple(sorted((x["body1"], x["body2"]))) for x in pairs}
        if not baseline_pairs:
            baseline_pairs = pair_names
        exact_checks.append({
            "q_CAD": row["q_CAD"],
            "pairs": pairs,
            "new_pair_names_vs_qCAD_0p20": [list(x) for x in sorted(pair_names - baseline_pairs)],
            "qualification": "original official visual STL triangle crossings at every 0.01 rad static continuation sample; baseline mating/contact pairs remain unclassified",
        })
    new_pairs = [x for x in exact_checks for x in x["new_pair_names_vs_qCAD_0p20"]]
    static_pass = (
        q65 is not None
        and fitted_complete
        and fitted_monotonic
        and not new_pairs
        and max(r["D_gap_mm"] for r in fitted_rows) <= STATIC_D_GAP_MM
        and max(r["proxy_penetration_mm"] for r in fitted_rows) <= STATIC_PROXY_PENETRATION_MM
    )
    result = {
        "schema": "robotsim.issue46.candidate-b-clearance-coupling-preflight.v1",
        "status": "PASS_STATIC_GEOMETRY_GATE" if static_pass else "BLOCKED_STATIC_GEOMETRY_GATE",
        "classification": "UNVERIFIED_EXPERIMENTAL_FUNCTIONAL_COUPLING",
        "new_physics_steps": 0,
        "frozen_baseline": {
            "run": str(BASE_RUN),
            "model_sha256": sha256(BASE_MJCF),
            "result_status": json.loads((BASE_RUN / "single_motor_result.json").read_text())["status"],
            "preserved": True,
        },
        "start_q_CAD": START_Q,
        "stop_reason": stop_reason,
        "q65_first_q_CAD": q65,
        "q70_first_q_CAD": q70,
        "q75_first_q_CAD": q75,
        "qCAD_last_static": fitted_rows[-1]["q_CAD"],
        "max_sampled_section_aperture_mm": max_row["minimum_sampled_section_aperture_mm"],
        "max_aperture_q_CAD": max_row["q_CAD"],
        "max_D_gap_mm": max(r["D_gap_mm"] for r in fitted_rows),
        "max_proxy_penetration_mm": max(r["proxy_penetration_mm"] for r in fitted_rows),
        "coupling": coupling,
        "raw_geometry_selected_rows": rows,
        "static_rows": fitted_rows,
        "fitted_path_complete": fitted_complete,
        "fitted_path_monotonic": fitted_monotonic,
        "exact_original_STL_checks": exact_checks,
        "new_original_STL_pair_names": new_pairs,
        "official_root_limits_unchanged": [-1.0, 0.0],
        "passive_joint_limits_changed": False,
        "collision_exclusions_added": 0,
        "actuator_contract": {"active_motors": 1, "passive_actuators": 0, "passive_qpos_writes_during_physics": 0},
        "limitations": [
            "This coupling is a geometry-selected functional law, not a manufacturer gear or spring law.",
            "The aperture is measured from original official URDF/STL gripping surfaces at five heights.",
            "Source-assembly mating overlaps at the qCAD=0.20 baseline remain unclassified.",
        ],
    }
    write_json(run, "static_preflight.json", result)
    variant = free_model_xml()
    eq = variant.find("equality")
    for name, joint1, joint2, coef in (
        ("geometry_narrow_coupling", "R_hand_narrow2_joint", "right_claw_joint", coeff_n),
        ("geometry_wide_coupling", "R_hand_wide2_joint", "R_hand_wide1_joint", coeff_w),
    ):
        ET.SubElement(eq, "joint", name=name, joint1=joint1, joint2=joint2,
                      polycoef=" ".join(f"{float(v):.16g}" for v in coef), solref="0.002 1")
    variant.set("model", "UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_OMNIPICKER")
    pg.write_new_text(run / "candidate_geometry_coupled.xml", ET.tostring(variant, encoding="unicode") + "\n", environment="wsl")
    return result


def cylinder_mesh(radius: float, half_height: float, center: tuple[float, float, float],
                  around: int = 96, height_steps: int = 24) -> np.ndarray:
    cx, cy, cz = center
    angles = np.linspace(0.0, 2.0 * math.pi, around, endpoint=False)
    z_values = np.linspace(cz - half_height, cz + half_height, height_steps + 1)
    rings = [
        np.column_stack((
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles),
            np.full(around, z),
        ))
        for z in z_values
    ]
    triangles = []
    for j in range(height_steps):
        low, high = rings[j], rings[j + 1]
        for i in range(around):
            nxt = (i + 1) % around
            triangles.extend(([low[i], low[nxt], high[nxt]], [low[i], high[nxt], high[i]]))
    bottom = np.asarray([cx, cy, cz - half_height])
    top = np.asarray([cx, cy, cz + half_height])
    for i in range(around):
        nxt = (i + 1) % around
        triangles.extend(([bottom, rings[0][nxt], rings[0][i]], [top, rings[-1][i], rings[-1][nxt]]))
    return np.asarray(triangles, dtype=float)


def static_cylinder_preflight(static_run: Path) -> tuple[Path, dict]:
    preflight = json.loads((static_run / "static_preflight.json").read_text())
    if preflight.get("status") != "PASS_STATIC_GEOMETRY_GATE":
        raise ValueError("cylinder checks require a passing geometry preflight")
    run = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-cylinders-corrected", environment="wsl").native_path
    for name in ("candidate_geometry_free.xml", "candidate_geometry_coupled.xml", "STATIC_PREFLIGHT.md", "static_preflight.json"):
        source = pg.validate_output_path(static_run / name, require_exists=True, expected_kind="file", environment="wsl")
        pg.write_new_bytes(run / name, source.native_path.read_bytes(), environment="wsl")
    source_model = run / "candidate_geometry_coupled.xml"
    model_sha = sha256(source_model)
    target = max(preflight["static_rows"], key=lambda row: row["q_CAD"])
    q = float(target["q_CAD"])
    values = values_for_q(q, float(target["B_CAD"]), np.asarray(target["passive_CAD"], dtype=float))
    model = mujoco.MjModel.from_xml_path(str(source_model))
    data = mujoco.MjData(model)
    apply_state(model, data, values)
    derived = ET.parse(BASE_URDF).getroot()
    geometry, triangles = functional._surface_metrics(model, derived, data)
    xlo, xhi, zlo, zhi = [float(x) / 1000.0 for x in geometry["multi_ray_grid"]["overlap_xz_mm"]]
    half_band = functional.GRIP_BAND_HEIGHT_MM / 2000.0
    z_centers = np.unique(np.append(
        np.linspace(zlo + half_band, zhi - half_band, 7),
        sum(functional.GRIP_BAND_Z_MM) / 2000.0,
    ))
    band_results = []
    for diameter in (65.0, 70.0, 75.0):
        bands = [
            functional.cylinder_band_fit(
                triangles["narrow"], triangles["wide"], diameter, float(zc), x_samples=25, z_samples=7
            )
            for zc in z_centers
        ]
        best = max(bands, key=lambda item: -math.inf if item["clearance_mm"] is None else item["clearance_mm"])
        probe = None
        if best["clearance_mm"] is not None:
            probe = functional._static_collision_probe(
                run,
                run / "candidate_geometry_coupled.xml",
                model,
                {"joint_positions": values},
                diameter,
                float(best["z_center_mm"]),
                float(best["center_x_mm"]),
                float(best["best_center_y_mm"]),
            )
        band_results.append({
            "diameter_mm": diameter,
            "best_band": best,
            "static_collision_proxy_probe": probe,
            "all_bands": [
                {"z_center_mm": item["z_center_mm"], "clearance_mm": item["clearance_mm"], "status": item["status"]}
                for item in bands
            ],
        })

    # Probe a full-height vertical cylinder while keeping its 10 mm grip band
    # inside the measured jaw region. This is static geometry only.
    full_xml = ET.parse(run / "candidate_geometry_coupled.xml")
    world = full_xml.getroot().find("worldbody")
    if world is None:
        raise ValueError("candidate MJCF lacks a worldbody")
    ET.SubElement(world, "geom", {
        "name": "probe_fullheight",
        "type": "cylinder",
        "size": "0.0375 0.0775",
        "pos": "0 0 0.17",
        "density": "0",
        "contype": "1",
        "conaffinity": "1",
        "condim": "4",
    })
    full_xml_path = run / "probe_fullheight.xml"
    pg.write_new_text(full_xml_path, ET.tostring(full_xml.getroot(), encoding="unicode") + "\n", environment="wsl")
    probe_model = mujoco.MjModel.from_xml_path(str(full_xml_path))
    probe_data = mujoco.MjData(probe_model)
    apply_state(probe_model, probe_data, values)
    probe_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, "probe_fullheight")
    probe_surfaces = travel.SourceSurfaces(probe_model, derived)
    source_meshes = []
    for gid in probe_surfaces.local:
        mesh = probe_surfaces.triangles(probe_data, gid)
        source_meshes.append({
            "gid": gid,
            "name": mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "triangles": mesh,
            "bounds": (mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))),
            "bvh": TriangleBVH(mesh),
        })
    full_height_results = []
    half_height = 0.0775
    half_band_m = functional.GRIP_BAND_HEIGHT_MM / 2000.0
    for band in band_results:
        diameter = float(band["diameter_mm"])
        best = band["best_band"]
        if best.get("clearance_mm") is None or best["clearance_mm"] < 0:
            full_height_results.append({
                "diameter_mm": diameter,
                "height_mm": 155.0,
                "status": "NO_LOCAL_GRIPPING_BAND_FIT",
                "placement": None,
                "candidate_count": 0,
                "candidates": [],
            })
            continue
        interval = best.get("feasible_center_y_interval_mm")
        if interval and float(interval[1]) >= float(interval[0]):
            y_centers = np.linspace(float(interval[0]) / 1000.0, float(interval[1]) / 1000.0, 5)
        else:
            y_centers = np.asarray([float(best["best_center_y_mm"]) / 1000.0])
        band_z = float(best["z_center_mm"]) / 1000.0
        z_centers = np.linspace(band_z - half_height + half_band_m, band_z + half_height - half_band_m, 7)
        center_x = float(best.get("center_x_mm", 0.0)) / 1000.0
        placements = sorted(
            ((center_x, float(cy), float(cz)) for cy in y_centers for cz in z_centers),
            key=lambda center: abs(center[1] - float(best["best_center_y_mm"]) / 1000.0) + abs(center[2] - band_z),
        )
        probe_model.geom_size[probe_id] = [diameter / 2000.0, half_height, 0.0]
        clear = None
        candidates = []
        for center in placements:
            probe_model.geom_pos[probe_id] = center
            mujoco.mj_forward(probe_model, probe_data)
            contacts = []
            for contact in probe_data.contact:
                if probe_id not in (int(contact.geom1), int(contact.geom2)):
                    continue
                other = int(contact.geom2) if int(contact.geom1) == probe_id else int(contact.geom1)
                contacts.append({
                    "geom": mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_GEOM, other),
                    "distance_mm": float(contact.dist) * 1000.0,
                })
            penetration_count = sum(item["distance_mm"] < -1e-4 for item in contacts)
            cyl = cylinder_mesh(diameter / 2000.0, half_height, center)
            cyl_low, cyl_high = cyl.min(axis=(0, 1)), cyl.max(axis=(0, 1))
            crossings = []
            if not penetration_count:
                cylinder_bvh = TriangleBVH(cyl)
                for source in source_meshes:
                    mesh_low, mesh_high = source["bounds"]
                    if np.any(mesh_low > cyl_high) or np.any(cyl_low > mesh_high):
                        continue
                    hit = source["bvh"].first_intersection(cylinder_bvh, eps=1e-8)
                    if hit is not None and hit["triangle_a"] is not None:
                        crossings.append({"geom": source["name"], **hit})
            candidate = {
                "center_m": [float(x) for x in center],
                "collision_proxy_contacts": contacts,
                "proxy_penetrating_contact_count": int(penetration_count),
                "minimum_proxy_contact_distance_mm": None if not contacts else min(item["distance_mm"] for item in contacts),
                "source_visual_STL_surface_crossings": crossings,
            }
            candidates.append(candidate)
            if not penetration_count and not crossings:
                clear = candidate
                break
        if clear:
            selected_tree = ET.parse(full_xml_path)
            selected_probe = selected_tree.getroot().find(".//geom[@name='probe_fullheight']")
            selected_probe.set("size", f"{diameter / 2000.0:.12g} {half_height:.12g}")
            selected_probe.set("pos", " ".join(f"{x:.12g}" for x in clear["center_m"]))
            selected_path = run / f"probe_{int(diameter)}mm_fullheight_selected.xml"
            pg.write_new_text(selected_path, ET.tostring(selected_tree.getroot(), encoding="unicode") + "\n", environment="wsl")
        full_height_results.append({
            "diameter_mm": diameter,
            "height_mm": 155.0,
            "status": "PASS_NO_PROXY_PENETRATION_OR_SOURCE_SURFACE_CROSSING" if clear else "NO_CLEAR_PLACEMENT_IN_SCANNED_WINDOW",
            "placement": clear,
            "candidate_count": len(candidates),
            "candidate_scan": candidates,
            "tested_center_x_m": center_x,
            "tested_center_y_m": [float(y_centers[0]), float(y_centers[-1]), len(y_centers)],
            "tested_center_z_m": [float(z_centers[0]), float(z_centers[-1]), len(z_centers)],
            "containment_check": "not independently tested; MuJoCo collision proxies and source visual STL surface crossings were checked",
        })
    cylinder_result = {
        "schema": "robotsim.issue46.candidate-b-static-cylinder-clearance.v1",
        "static_model_sha256": model_sha,
        "qCAD": q,
        "joint_positions_urdf_rad": values,
        "cad_to_urdf_sign_mapping": {
            "R_hand_narrow2_joint": "-B_CAD",
            "R_hand_wide2_joint": "+B_CAD",
            "source_static_row_B_CAD": float(target["B_CAD"]),
        },
        "physics_steps": 0,
        "gripping_band_height_mm": functional.GRIP_BAND_HEIGHT_MM,
        "band_results": band_results,
        "full_height_cylinders": full_height_results,
        "qualification": "static 155 mm tall vertical cylinder geometry probe; correct CAD-to-URDF signs; no object attachment, mj_step, approach, grasp, or lift",
    }
    write_json(run, "cylinder_probe_results.json", cylinder_result)
    write_json(run, "source_static_run.json", {
        "source_run": str(static_run),
        "copied_static_model_sha256": model_sha,
        "qCAD": q,
        "corrected_joint_positions_urdf_rad": values,
        "cad_to_urdf_sign_mapping": cylinder_result["cad_to_urdf_sign_mapping"],
        "preserved_first_physics_attempt": (
            json.loads((static_run / "physics_result.json").read_text())
            if (static_run / "physics_result.json").is_file()
            else None
        ),
    })
    finalize_manifest(run)
    return run, cylinder_result


def write_mp4(run: Path, frames) -> Path | None:
    if not frames:
        return None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    output = run / "geometry_coupling_open_hold_close.mp4"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x480",
        "-framerate", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read()
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if code:
        raise RuntimeError(f"ffmpeg video encoding failed: {stderr.decode(errors='replace')[-1000:]}")
    return output


def physics_run(run: Path, result: dict) -> dict:
    if result.get("status") != "PASS_STATIC_GEOMETRY_GATE":
        raise ValueError("physics is gated on PASS_STATIC_GEOMETRY_GATE")
    cylinder_path = run / "cylinder_probe_results.json"
    if not cylinder_path.is_file():
        raise ValueError("correctly mapped static cylinder preflight is required before physics")
    cylinders = json.loads(cylinder_path.read_text())
    model_path = run / "candidate_geometry_coupled.xml"
    if cylinders.get("static_model_sha256") != sha256(model_path):
        raise ValueError("static cylinder result is not bound to this exact coupled model")
    if cylinders.get("cad_to_urdf_sign_mapping", {}).get("R_hand_narrow2_joint") != "-B_CAD":
        raise ValueError("static cylinder result lacks the CAD-to-URDF joint sign mapping")
    if cylinders.get("qCAD") != float(result["qCAD_last_static"]):
        raise ValueError("static cylinder result was measured at a different qCAD endpoint")
    band65 = next(x for x in cylinders["band_results"] if x["diameter_mm"] == 65.0)
    band65_probe = band65.get("static_collision_proxy_probe")
    if (band65["best_band"]["clearance_mm"] is None
            or band65["best_band"]["clearance_mm"] < 0
            or band65_probe is None
            or band65_probe["penetrating_contact_count"] != 0):
        raise ValueError("65 mm local gripping-band probe is not collision clear")
    full_by_diameter = {
        int(x["diameter_mm"]): x for x in cylinders["full_height_cylinders"]
    }

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = p.addresses(model)
    start = min(result["static_rows"], key=lambda x: abs(x["q_CAD"] - START_Q))
    q_start = float(start["q_CAD"])
    q_end = float(result["qCAD_last_static"])
    apply_state(model, data, values_for_q(q_start, start["B_CAD"], np.asarray(start["passive_CAD"])))
    data.ctrl[0] = p.PHASE - q_start
    mujoco.mj_forward(model, data)
    if model.nu != 1 or int(model.actuator_trnid[0, 0]) != mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint"
    ):
        raise ValueError("one-active-root-motor invariant failed")
    if max(np.abs(d_residual_mm(model, data))) > 0.01:
        raise ValueError("fitted coupling start pose does not close the D sites")
    if travel.proxy_contacts(model, data):
        raise ValueError("fitted coupling start pose has collision-proxy contact")

    from PIL import Image, ImageDraw, ImageFont

    surfaces = travel.SourceSurfaces(model, ET.parse(BASE_URDF).getroot())
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 0.34, 0.0, -12.0
    options = mujoco.MjvOption()
    options.geomgroup[1] = 0
    gif_path = run / "geometry_coupling_open_hold_close.gif"
    frames = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    schedule = [(q_start, 0.5), (q_end, 2.0), (q_end, 1.0), (q_start, 2.0), (q_start, 1.0)]
    trace, phases = [], []
    max_dgap = max_pen = max_speed = max_coupling = max_force = 0.0
    step_count = 0
    previous_q = q_start
    stop = None
    max_aperture = None
    max_aperture_q = None
    open_hold_samples, return_hold_samples = [], []
    joint_dof_addresses = {
        name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in addresses
    }
    max_joint_speed_by_name = {name: 0.0 for name in addresses}
    for phase_i, (goal, duration) in enumerate(schedule):
        actual_start = p.PHASE - float(data.qpos[addresses["right_claw_joint"]])
        for i in range(round(duration / float(model.opt.timestep))):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * min(i * float(model.opt.timestep) / max(duration - 0.5, 0.001), 1.0)
            )
            q_command = previous_q + (goal - previous_q) * blend
            root_target = p.PHASE - q_command
            root_feedforward, _ = travel.root_bias_compensation(model, data)
            raw_control = root_target + root_feedforward / float(model.actuator_gainprm[0, 0])
            data.ctrl[0] = float(np.clip(raw_control, *model.actuator_ctrlrange[0]))
            mujoco.mj_step(model, data)
            step_count += 1
            mujoco.mj_forward(model, data)
            actual_q = p.PHASE - float(data.qpos[addresses["right_claw_joint"]])
            dgap = float(np.max(np.abs(d_residual_mm(model, data))))
            contacts = travel.proxy_contacts(model, data)
            penetration = max((float(c["penetration_mm"]) for c in contacts), default=0.0)
            speed = float(np.max(np.abs(data.qvel)))
            for name, dof in joint_dof_addresses.items():
                max_joint_speed_by_name[name] = max(max_joint_speed_by_name[name], abs(float(data.qvel[dof])))
            narrow_eq = float(
                data.qpos[addresses["R_hand_narrow2_joint"]]
                - np.polynomial.polynomial.polyval(
                    data.qpos[addresses["right_claw_joint"]], result["coupling"]["narrow"]
                )
            )
            wide_eq = float(
                data.qpos[addresses["R_hand_wide2_joint"]]
                - np.polynomial.polynomial.polyval(
                    data.qpos[addresses["R_hand_wide1_joint"]], result["coupling"]["wide"]
                )
            )
            coupling_error = max(abs(narrow_eq), abs(wide_eq))
            force_max = float(np.max(np.abs(data.efc_force))) if data.efc_force.size else 0.0
            root = float(data.qpos[addresses["right_claw_joint"]])
            wide_root = float(data.qpos[addresses["R_hand_wide1_joint"]])
            limit_error = max(0.0, root, -1.0 - root, -wide_root, wide_root - 1.0)
            max_dgap, max_pen = max(max_dgap, dgap), max(max_pen, penetration)
            max_speed, max_coupling, max_force = max(max_speed, speed), max(max_coupling, coupling_error), max(max_force, force_max)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop = "NONFINITE_STATE"
            elif limit_error > 1e-5:
                stop = "OFFICIAL_ROOT_LIMIT_OVERSHOOT"
            elif actual_q < q_start - 0.01 or actual_q > q_end + 0.01:
                stop = "OUTSIDE_STATIC_GEOMETRY_PATH"
            elif dgap > 0.5:
                stop = "D_POINT_CLOSURE_LOST"
            elif penetration > 0.1:
                stop = "COLLISION_PROXY_PENETRATION"
            elif coupling_error > 0.01:
                stop = "GEOMETRY_COUPLING_RESIDUAL"
            elif speed > MAX_JOINT_SPEED_RAD_S:
                stop = "EXCESSIVE_JOINT_SPEED"
            elif any(w.number for w in data.warning):
                stop = "MUJOCO_SOLVER_WARNING"
            sampled = step_count % 10 == 0 or stop is not None
            if sampled:
                row_values = {name: float(data.qpos[adr]) for name, adr in addresses.items()}
                width, sections = surfaces.aperture(data)
                if width is not None and (max_aperture is None or width > max_aperture):
                    max_aperture, max_aperture_q = float(width), float(actual_q)
                valid_widths = [float(section["width_mm"]) for section in sections if section["width_mm"] is not None]
                minimum_section_width = min(valid_widths) if len(valid_widths) == len(sections) else None
                item = {
                    "step": step_count,
                    "time_s": float(data.time),
                    "phase": phase_i,
                    "command_q_CAD": float(q_command),
                    "actual_q_CAD": float(actual_q),
                    "actual_root_rad": root,
                    "jaw_section_aperture_mm": width,
                    "minimum_five_section_aperture_mm": minimum_section_width,
                    "sections": sections,
                    "D_gap_mm": dgap,
                    "proxy_penetration_mm": penetration,
                    "max_joint_speed_rad_s": speed,
                    "geometry_coupling_residual_rad": coupling_error,
                    "root_feedforward_nm": float(root_feedforward),
                    "constraint_force_max_mixed_units": force_max,
                    "joint_positions": row_values,
                    "contacts": contacts,
                    "stop_reason": stop,
                }
                trace.append(item)
                if phase_i == 2:
                    open_hold_samples.append(item)
                elif phase_i == 4:
                    return_hold_samples.append(item)
            if step_count % 40 == 0 or stop:
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 60), fill=(12, 20, 28))
                draw.text((8, 4), "ONE ROOT MOTOR / GEOMETRY-SELECTED COUPLING", font=font, fill=(255, 207, 112))
                draw.text((8, 22), f"t={data.time:.2f}s qCAD={actual_q:+.3f} rad gravity ON", font=font, fill="white")
                draw.text((8, 40), f"D={dgap:.4f} mm collision={penetration:.4f} mm speed={speed:.3f} rad/s", font=font, fill="white")
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                if step_count % 1000 == 0 or stop:
                    frame.save(run / f"physics_{step_count:05d}.png")
            if stop:
                break
        actual_end = p.PHASE - float(data.qpos[addresses["right_claw_joint"]])
        phases.append({"phase": phase_i, "command_end_q_CAD": goal, "actual_end_q_CAD": actual_end, "completed": stop is None})
        if stop:
            break
        if abs(actual_end - goal) > 0.02:
            stop = "PHASE_TARGET_NOT_REACHED"
            phases[-1]["completed"] = False
            break
        previous_q = goal
    renderer.close()
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    mp4_path = write_mp4(run, frames)
    with (run / "physics_trace.json").open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (run / "physics_trace.csv").open("x", newline="", encoding="utf-8") as stream:
        keys = ["step", "time_s", "phase", "command_q_CAD", "actual_q_CAD", "actual_root_rad",
                "jaw_section_aperture_mm", "minimum_five_section_aperture_mm", "D_gap_mm", "proxy_penetration_mm",
                "max_joint_speed_rad_s", "geometry_coupling_residual_rad", "root_feedforward_nm",
                "constraint_force_max_mixed_units", "stop_reason"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in trace)

    def hold_summary(samples):
        if not samples:
            return {"status": "NOT_REACHED", "sample_count": 0}
        q_values = [float(x["actual_q_CAD"]) for x in samples]
        q_span = max(q_values) - min(q_values)
        speed_hold = max(float(x["max_joint_speed_rad_s"]) for x in samples)
        d_gap_hold = max(float(x["D_gap_mm"]) for x in samples)
        coupling_hold = max(float(x["geometry_coupling_residual_rad"]) for x in samples)
        penetration_hold = max(float(x["proxy_penetration_mm"]) for x in samples)
        stable = q_span <= 0.01 and speed_hold <= 0.25 and d_gap_hold <= 0.5 and coupling_hold <= 0.01 and penetration_hold <= 0.1
        return {
            "status": "PASS" if stable else "FAIL",
            "sample_count": len(samples),
            "qCAD_min": min(q_values),
            "qCAD_max": max(q_values),
            "qCAD_span": q_span,
            "D_gap_max_mm": d_gap_hold,
            "speed_max_rad_s": speed_hold,
            "coupling_residual_max_rad": coupling_hold,
            "proxy_penetration_max_mm": penetration_hold,
        }

    open_hold = hold_summary(open_hold_samples)
    return_hold = hold_summary(return_hold_samples)
    if stop is None and open_hold["status"] != "PASS":
        stop = "OPEN_POSITION_HOLD_FAILED"
    if stop is None and return_hold["status"] != "PASS":
        stop = "RETURN_POSITION_HOLD_FAILED"
    result = {
        "status": "EXPERIMENTAL_MUJOCO_PASS" if stop is None else "EXPERIMENTAL_MUJOCO_FAIL",
        "stop_reason": stop,
        "mujoco_version": mujoco.__version__,
        "physics_steps": step_count,
        "simulation_time_s": float(data.time),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "active_actuator_count": int(model.nu),
        "active_joint": "right_claw_joint",
        "passive_actuators": 0,
        "passive_qpos_writes_after_initialization": 0,
        "qCAD_tested": [q_start, q_end],
        "phases": phases,
        "open_hold": open_hold,
        "return_position_hold": return_hold,
        "closed_position_hold": {
            "status": "NOT_RUN",
            "reason": "qCAD=0.20 is the tested baseline return pose, not a fully closed gripper pose",
        },
        "max_D_gap_mm": max_dgap,
        "max_proxy_penetration_mm": max_pen,
        "max_joint_speed_rad_s": max_speed,
        "max_joint_speed_by_name_rad_s": max_joint_speed_by_name,
        "joint_speed_stop_threshold_rad_s": MAX_JOINT_SPEED_RAD_S,
        "max_geometry_coupling_residual_rad": max_coupling,
        "max_constraint_force_mixed_units": max_force,
        "root_bias_compensation": True,
        "root_controller": "Candidate B baseline kp=0.5, kv=0.05, +/-0.5 Nm cap with root-only model-bias feedforward",
        "max_measured_section_aperture_mm": max_aperture,
        "qCAD_at_max_measured_aperture": max_aperture_q,
        "max_measured_minimum_five_section_aperture_mm": max(
            (float(row["minimum_five_section_aperture_mm"]) for row in trace if row["minimum_five_section_aperture_mm"] is not None),
            default=None,
        ),
        "static_gripping_band_center_interval_width_mm": {
            str(int(item["diameter_mm"])): item["best_band"]["clearance_mm"]
            for item in cylinders["band_results"]
        },
        "static_fullheight_status": {
            str(diameter): full_by_diameter[diameter]["status"]
            for diameter in (65, 70, 75)
        },
        "object_dynamics_or_attachment": "NOT RUN",
        "video": str(gif_path),
        "video_mp4": None if mp4_path is None else str(mp4_path),
        "original_visual_STL_clearance": "static preflight checks every 0.01 rad fitted-path sample; physical trace uses the unchanged MuJoCo collision proxies",
    }
    write_json(run, "physics_result.json", result)
    return result


def finalize_manifest(run: Path) -> dict:
    artifacts = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name in {".robotsim-output-reservation.json", "artifact_paths.json"}:
            continue
        location = pg.validate_output_path(path, require_exists=True, expected_kind="file", environment="wsl")
        artifacts.append({
            "name": path.name,
            **location.as_dict(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {"schema": "robotsim.artifact-paths.v1", "artifacts": artifacts}
    pg.write_new_text(run / "artifact_paths.json", json.dumps(manifest, indent=2) + "\n", environment="wsl")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("preflight", "cylinders", "physics"), default="preflight")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--q-scan-max", type=float, default=0.55)
    args = ap.parse_args()
    verify_inputs()
    recorded = json.loads((BASE_RUN / "provenance.json").read_text())
    if sha256(BASE_MJCF) != recorded["mjcf_sha256"]:
        raise ValueError("frozen Candidate B MJCF hash mismatch")
    if args.stage in ("cylinders", "physics"):
        if args.run_dir is None:
            raise ValueError(f"{args.stage} stage requires an exact preceding run directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        source_run = location.native_path
        result = json.loads((source_run / "static_preflight.json").read_text())
        if result.get("status") != "PASS_STATIC_GEOMETRY_GATE":
            raise ValueError("static geometry preflight did not pass")
        if args.stage == "cylinders":
            cylinder_run, cylinder = static_cylinder_preflight(source_run)
            band65 = next(item for item in cylinder["band_results"] if item["diameter_mm"] == 65.0)
            full65 = next(item for item in cylinder["full_height_cylinders"] if item["diameter_mm"] == 65.0)
            print(f"CYLINDER_65MM_BAND_CLEARANCE_MM={band65['best_band']['clearance_mm']}")
            print(f"CYLINDER_65MM_FULLHEIGHT={full65['status']}")
            print(f"RUN_DIR={cylinder_run}")
            return 0 if band65["best_band"]["clearance_mm"] >= 0 else 2
        source_info = json.loads((source_run / "source_static_run.json").read_text())
        cylinder_file = source_run / "cylinder_probe_results.json"
        if not cylinder_file.is_file():
            raise ValueError("physics requires the new correctly mapped static cylinder results")
        prior = source_info.get("preserved_first_physics_attempt")
        attempt = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-clearance-physics", environment="wsl").native_path
        static_files = (
            "static_preflight.json",
            "STATIC_PREFLIGHT.md",
            "candidate_geometry_free.xml",
            "candidate_geometry_coupled.xml",
            "cylinder_probe_results.json",
        )
        copied = []
        optional_files = sorted(path.name for path in source_run.glob("probe_*_fullheight_selected.xml"))
        for name in (*static_files, *optional_files):
            source = source_run / name
            if not source.is_file():
                raise ValueError(f"missing required static evidence: {source}")
            pg.write_new_bytes(attempt / name, source.read_bytes(), environment="wsl")
            copied.append({"name": name, "sha256": sha256(attempt / name)})
        write_json(attempt, "source_static_run.json", {
            "source_run": str(source_run),
            "copied_static_artifacts": copied,
            "preserved_first_physics_attempt": prior,
            "second_attempt_controller": "root-only model-bias feedforward already used by the frozen Candidate B baseline; gains unchanged",
        })
        physics = physics_run(attempt, result)
        verify_inputs()
        report = [
            "# Candidate B Geometry-Selected Coupling Physical Validation",
            "",
            f"Status: {physics['status']}",
            "",
            f"MuJoCo 3.3.6 steps: {physics['physics_steps']}; simulation time: {physics['simulation_time_s']:.3f} s.",
            f"qCAD range: {physics['qCAD_tested'][0]:.3f} to {physics['qCAD_tested'][1]:.3f} rad.",
            f"One active root motor; passive actuators: {physics['passive_actuators']}; passive qpos writes after initialization: {physics['passive_qpos_writes_after_initialization']}.",
            f"Max D gap: {physics['max_D_gap_mm']:.6g} mm; max collision-proxy penetration: {physics['max_proxy_penetration_mm']:.6g} mm.",
            f"Max speed: {physics['max_joint_speed_rad_s']:.6g} rad/s; max coupling residual: {physics['max_geometry_coupling_residual_rad']:.6g} rad.",
            f"Max measured five-section minimum aperture: {physics['max_measured_minimum_five_section_aperture_mm']} mm.",
            f"65/70/75 mm gripping-band feasible center-interval widths: {physics['static_gripping_band_center_interval_width_mm']} mm.",
            f"Full-height 65/70/75 mm cylinder fits: {physics['static_fullheight_status']}.",
            "qCAD=0.20 is the baseline return pose, not a fully closed pose; a closed-position hold was not tested.",
            f"Open/close video: {physics['video_mp4'] or physics['video']}.",
            "Root-only model-bias feedforward is used with the frozen Candidate B servo gains because the first hold-only attempt drifted before motion began.",
            "",
            "The passive coupling is geometry-selected and UNVERIFIED_EXPERIMENTAL. No manufacturer transmission claim, object contact, grasp, or lift is made.",
        ]
        pg.write_new_text(attempt / "REPORT.md", "\n".join(report) + "\n", environment="wsl")
        finalize_manifest(attempt)
        print(f"PHYSICS_STATUS={physics['status']}")
        print(f"RUN_DIR={attempt}")
        print(f"VIDEO={physics['video_mp4'] or physics['video']}")
        return 0 if physics["status"] == "EXPERIMENTAL_MUJOCO_PASS" else 3
    if args.run_dir:
        run_location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(run_location)
        run = run_location.native_path
        if any((run / name).exists() for name in ("static_preflight.json", "STATIC_PREFLIGHT.md", "artifact_paths.json")):
            raise ValueError("preflight run already has a terminal artifact; refusing to overwrite")
    else:
        run_location = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-clearance-coupling", environment="wsl")
        run = run_location.native_path
    result = static_preflight(run, args.q_scan_max)
    lines = [
        "# Candidate B Geometry-Selected Coupling Preflight",
        "",
        f"Status: {result['status']}",
        "",
        f"Frozen Candidate B remains unchanged at {BASE_RUN}.",
        f"Static continuation: qCAD {START_Q:.3f} to {result['qCAD_last_static']:.3f} rad.",
        f"Maximum sampled official-STL section aperture: {result['max_sampled_section_aperture_mm']:.3f} mm at qCAD={result['max_aperture_q_CAD']:.3f}.",
        f"65/70/75 mm first qCAD: {result['q65_first_q_CAD']} / {result['q70_first_q_CAD']} / {result['q75_first_q_CAD']}.",
        f"Maximum D gap: {result['max_D_gap_mm']:.6g} mm; maximum collision-proxy penetration: {result['max_proxy_penetration_mm']:.6g} mm.",
        f"Quadratic coupling fit residual: {result['coupling']['max_fit_error_rad']:.6g} rad.",
        f"Static STL intersection checks: {len(result['exact_original_STL_checks'])} samples; new pair names vs qCAD=0.20 baseline: {len(result['new_original_STL_pair_names'])}.",
        f"Baseline source-mesh contact/intersection pairs remain unclassified: {len(result['exact_original_STL_checks'][0]['pairs'])}.",
        "",
        "Coupling selection is a functional geometric continuation, not a vendor law. No physics was run by this stage.",
    ]
    pg.write_new_text(run / "STATIC_PREFLIGHT.md", "\n".join(lines) + "\n", environment="wsl")
    finalize_manifest(run)
    if result["status"] != "PASS_STATIC_GEOMETRY_GATE":
        print(f"STATIC_STATUS={result['status']}")
        print(f"RUN_DIR={run}")
        return 2
    print(f"STATIC_STATUS={result['status']}")
    print(f"RUN_DIR={run}")
    print(f"Q65={result['q65_first_q_CAD']}")
    print(f"QMAX={result['qCAD_last_static']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
