#!/usr/bin/env python3
"""Evaluate two geometry-derived X2 OmniPicker bottle grasp alternatives."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRIOR = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/grasp-corridor-20261008/run-04/result.json")
DEFAULT_OUTPUT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/outcome-grasp-corridor-20261009/run-01")
APPROACH_M = 0.080
APPROACH_SAMPLES = 21
CLOSURE_SAMPLES = 101
LIFT_M = 0.030
LIFT_SAMPLES = 16
CONTACT_TOL = 1.0e-5
CLEARANCE_EPS = 1.0e-7
NARROW = "R_hand_narrow3_Link"
WIDE = "R_hand_wide3_Link"
TARGETS = {NARROW, WIDE}
ROOT_LINK = "right_elbow_link"
YAW = "right_wrist_yaw_joint"
PITCH = "right_wrist_pitch_joint"
ROLL = "right_wrist_roll_joint"
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qpos_id(model, joint: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise RuntimeError(f"Missing joint {joint}")
    return int(model.jnt_qposadr[jid])


def body_id(model, body: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise RuntimeError(f"Missing body {body}")
    return int(bid)


def geom_id(model, geom: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if gid < 0:
        raise RuntimeError(f"Missing geom {geom}")
    return int(gid)


def source_joint_range(urdf: Path, joint_name: str) -> list[float]:
    root = ET.parse(urdf).getroot()
    joint = next(x for x in root.findall("joint") if x.get("name") == joint_name)
    limit = joint.find("limit")
    if limit is None:
        raise RuntimeError(f"No source limits on {joint_name}")
    return [float(limit.get("lower")), float(limit.get("upper"))]


def assign_joint(model, data, joint: str, value: float) -> None:
    data.qpos[qpos_id(model, joint)] = float(value)


def set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture, bottle_lift=0.0):
    model.body_pos[body_id(model, ROOT_LINK)] = (root_origin + root_shift).tolist()
    targets = base.aperture_targets(float(aperture))
    assign_joint(model, data, YAW, yaw)
    assign_joint(model, data, PITCH, pitch)
    assign_joint(model, data, ROLL, roll)
    assign_joint(model, data, DRIVER, targets["right_claw_joint_target_rad"])
    assign_joint(model, data, FOLLOWER, targets["R_hand_wide1_joint_target_rad"])
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_qpos = int(model.jnt_qposadr[bottle_joint])
    data.qpos[bottle_qpos:bottle_qpos + 3] = model.qpos0[bottle_qpos:bottle_qpos + 3] + np.array([0.0, 0.0, bottle_lift])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def roll_for_minimum_vertical_insertion(axis: np.ndarray, vector: np.ndarray, limits: list[float]) -> dict[str, Any]:
    axis = axis / np.linalg.norm(axis)
    vector = vector / np.linalg.norm(vector)
    parallel = axis * float(np.dot(axis, vector))
    perpendicular = vector - parallel
    cross = np.cross(axis, vector)
    aa, bb, cc = float(perpendicular[2]), float(cross[2]), float(parallel[2])
    radius = math.hypot(aa, bb)
    if radius < 1.0e-10:
        raise RuntimeError("Wrist-roll axis cannot change insertion-axis elevation")
    phase = math.atan2(bb, aa)

    def in_range(raw_values):
        rows = []
        for value in raw_values:
            for turns in (-2, -1, 0, 1, 2):
                candidate = value + turns * 2.0 * math.pi
                if limits[0] - 1.0e-12 <= candidate <= limits[1] + 1.0e-12:
                    rotated = (vector * math.cos(candidate) + cross * math.sin(candidate)
                               + parallel * (1.0 - math.cos(candidate)))
                    rows.append({"roll_rad": float(candidate), "insertion_axis_world": rotated.tolist(),
                                 "z_error": abs(float(rotated[2]))})
        return rows

    exact_roots = []
    if abs(cc) <= radius:
        offset = math.acos(float(np.clip(-cc / radius, -1.0, 1.0)))
        exact_roots = in_range([phase + offset, phase - offset])
    if exact_roots:
        candidates = exact_roots
        selected_kind = "exact_horizontal_root"
    else:
        # z(phi)=A cos(phi)+B sin(phi)+C. Endpoints and analytic extrema
        # contain the exact minimum of |z| when no horizontal root is feasible.
        extrema = [phase + turn * math.pi for turn in (-2, -1, 0, 1, 2)]
        candidates = in_range([limits[0], limits[1], *extrema])
        selected_kind = "bounded_nearest_horizontal_extremum"
    if not candidates:
        raise RuntimeError(f"No analytic wrist-roll candidate lies in source limits {limits}")
    candidates.sort(key=lambda row: (row["z_error"], abs(row["roll_rad"]), row["roll_rad"]))
    return {"axis_world": axis.tolist(), "initial_insertion_axis_world": vector.tolist(),
            "equation_coefficients_A_B_C": [aa, bb, cc], "source_roll_limits_rad": limits,
            "exact_horizontal_roots_in_limits": exact_roots, "bounded_analytic_candidates": candidates,
            "selected_kind": selected_kind, "selected_roll_rad": candidates[0]["roll_rad"],
            "selected_insertion_axis_world": candidates[0]["insertion_axis_world"],
            "selected_vertical_component": candidates[0]["insertion_axis_world"][2]}


def contact_distances(corr, abc, model, data, target_geom: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ng = corr.collision_geom_for_body(model, NARROW)
    wg = corr.collision_geom_for_body(model, WIDE)
    return abc.geom_distance(model, data, ng, target_geom), abc.geom_distance(model, data, wg, target_geom)


def jaw_midpoint(model, data) -> np.ndarray:
    narrow_geom = next(g for g in range(model.ngeom)
                       if int(model.geom_bodyid[g]) == body_id(model, NARROW)
                       and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])))
    wide_geom = next(g for g in range(model.ngeom)
                     if int(model.geom_bodyid[g]) == body_id(model, WIDE)
                     and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])))
    segment = np.zeros(6)
    mujoco.mj_geomDistance(model, data, narrow_geom, wide_geom, 1.0, segment)
    return 0.5 * (segment[:3] + segment[3:])


def find_contact_aperture(corr, abc, base, model, data, root_origin, root_shift, target_geom, yaw, pitch, roll):
    target_center = data.geom_xpos[target_geom].copy()
    start_aperture = 0.58
    set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, start_aperture)
    root_shift = root_shift + (target_center - jaw_midpoint(model, data))
    iterations = []
    for outer in range(5):
        def at(aperture):
            set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture)
            return contact_distances(corr, abc, model, data, target_geom)

        open_rows = at(1.0)
        closed_rows = at(0.0)
        open_sum = sum(float(x["signed_distance_m"]) for x in open_rows)
        closed_sum = sum(float(x["signed_distance_m"]) for x in closed_rows)
        if open_sum < 0.0 or closed_sum > 0.0:
            return {
                "contact_bracket_found": False,
                "failure": "jaw source aperture does not bracket simultaneous contact with target component",
                "root_shift_world_m": root_shift.tolist(),
                "target_center_world_m": target_center.tolist(),
                "open_aperture_distances_m": [x["signed_distance_m"] for x in open_rows],
                "closed_aperture_distances_m": [x["signed_distance_m"] for x in closed_rows],
                "open_sum_m": open_sum,
                "closed_sum_m": closed_sum,
                "iterations": iterations,
            }
        lo, hi = 0.0, 1.0
        bisect = []
        for index in range(56):
            mid = 0.5 * (lo + hi)
            rows = at(mid)
            residual = sum(float(x["signed_distance_m"]) for x in rows)
            bisect.append({"iteration": index, "aperture_ratio": mid,
                           "narrow_distance_m": rows[0]["signed_distance_m"],
                           "wide_distance_m": rows[1]["signed_distance_m"], "sum_residual_m": residual})
            if residual > 0.0:
                hi = mid
            else:
                lo = mid
        aperture = 0.5 * (lo + hi)
        final_rows = at(aperture)
        midpoint = jaw_midpoint(model, data)
        correction = target_center - midpoint
        iterations.append({"outer_iteration": outer, "aperture_ratio": aperture,
                           "narrow_distance_m": final_rows[0]["signed_distance_m"],
                           "wide_distance_m": final_rows[1]["signed_distance_m"],
                           "jaw_midpoint_world_m": midpoint.tolist(),
                           "target_center_world_m": target_center.tolist(),
                           "root_translation_correction_world_m": correction.tolist(),
                           "correction_norm_m": float(np.linalg.norm(correction)),
                           "bisection": bisect})
        if float(np.linalg.norm(correction)) <= 2.0e-6:
            break
        root_shift = root_shift + correction
    set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture)
    return {"contact_bracket_found": True, "aperture_ratio": float(aperture),
            "root_shift_world_m": root_shift.tolist(), "target_center_world_m": target_center.tolist(),
            "contact_distances_m": [x["signed_distance_m"] for x in contact_distances(corr, abc, model, data, target_geom)],
            "iterations": iterations}


def pair_clearance(state, target_body_names=TARGETS, target_geom_name=None, allow_target=False):
    non_target = []
    invalid = 0
    for row in state["pair_rows"]:
        if not row["query_consistent"] or not math.isfinite(float(row["signed_distance_m"])):
            invalid += 1
            continue
        intended = row["body1"] in target_body_names and (target_geom_name is None or row["geom2"] == target_geom_name)
        if not (allow_target and intended):
            non_target.append(row)
    return {"minimum_non_target_m": min((r["signed_distance_m"] for r in non_target), default=None),
            "invalid_queries": invalid,
            "non_target_clear": invalid == 0 and all(r["signed_distance_m"] > CLEARANCE_EPS for r in non_target)}


def all_state_gates(corr, model, data, state, allow_target=False, target_geom_name=None):
    pairs = pair_clearance(state, target_geom_name=target_geom_name, allow_target=allow_target)
    table = state["minimum_robot_table"]
    self_rows = state["robot_self_contacts"]
    self_clear = all(float(r["distance_m"]) >= -CLEARANCE_EPS for r in self_rows)
    limits = corr.joint_limits(corr.X2_URDF, model, data)
    return {
        "source_joint_limits": bool(limits["all_within_source_limits"]),
        "non_target_bottle_clearance": pairs["non_target_clear"],
        "robot_table_clearance": table is not None and table["signed_distance_m"] > CLEARANCE_EPS,
        "no_robot_self_penetration": self_clear,
        "collision_queries_valid": pairs["invalid_queries"] == 0,
    }, {"pair_clearance": pairs, "source_joint_limits": limits,
        "minimum_robot_table": table, "robot_self_contacts": self_rows}


def record_state(corr, abc, base, model, data, arrays, name, phase, sample, root_origin, root_shift,
                 yaw, pitch, roll, aperture, target_geom_name, lift=0.0):
    set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture, lift)
    robot, bottle, table = corr.collision_sets(abc, model)
    state = corr.collision_state(abc, model, data, robot, bottle, table)
    gates, details = all_state_gates(corr, model, data, state, allow_target=phase in {"CLOSURE", "CONTACT", "LIFT"},
                                     target_geom_name=target_geom_name)
    summary = {
        "candidate": name, "phase": phase, "sample": sample, "aperture_ratio": aperture,
        "lift_m": lift, "minimum_robot_bottle_m": state["minimum_robot_bottle"]["signed_distance_m"] if state["minimum_robot_bottle"] else None,
        "minimum_robot_bottle_pair": [state["minimum_robot_bottle"]["body1"], state["minimum_robot_bottle"]["geom2"]] if state["minimum_robot_bottle"] else None,
        "minimum_non_target_robot_bottle_m": details["pair_clearance"]["minimum_non_target_m"],
        "minimum_robot_table_m": details["minimum_robot_table"]["signed_distance_m"] if details["minimum_robot_table"] else None,
        "robot_bottle_contact_count": len(state["robot_bottle_contacts"]),
        "robot_self_contact_count": len(state["robot_self_contacts"]),
        "source_limits_pass": details["source_joint_limits"]["all_within_source_limits"],
        "non_target_clear": details["pair_clearance"]["non_target_clear"],
        "robot_table_clear": gates["robot_table_clearance"],
        "no_robot_self_penetration": gates["no_robot_self_penetration"],
        "invalid_queries": details["pair_clearance"]["invalid_queries"],
    }
    arrays["summary"].append(summary)
    for row in state["pair_rows"]:
        arrays["pairwise"].append({
            "candidate": name, "phase": phase, "sample": sample, "aperture_ratio": aperture,
            "lift_m": lift, "robot_body": row["body1"], "robot_geom": row["geom1"],
            "bottle_part": row["geom2"], "signed_distance_m": row["signed_distance_m"],
            "witness_length_m": row["witness_length_m"], "query_consistent": row["query_consistent"],
        })
    arrays["states"].append({"summary": summary, "gates": gates,
                             "contacts": state["robot_bottle_contacts"]})
    return state, summary, gates, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any], output_dir: Path, candidate_filter: str) -> None:
    runtime = payload["runtime"]
    candidate_sections = []
    for result in payload["candidates"]:
        candidate = result["candidate"]
        gates = "\n".join(
            f"- `{key}`: {'PASS' if passed else 'FAIL'}"
            for key, passed in result.get("gates", {}).items()
        )
        open_distances = result.get("open", {}).get("target_jaw_distances_m", [])
        contact = result.get("contact_geometry", {})
        min_open = result.get("open", {}).get("summary", {}).get("minimum_robot_bottle_m")
        candidate_sections.append(
            f"### {candidate['name']}\n\n"
            f"- Region: {candidate['target_region']}; target geom `{candidate['target_bottle_geom']}`.\n"
            f"- Wrist yaw/pitch/roll: `{candidate['wrist_targets_rad']}` rad; source ranges: "
            f"`{candidate['source_joint_ranges_rad']}`.\n"
            f"- Analytic side-insertion direction: `{candidate['approach_direction_world']}` world; "
            f"pregrasp offset: {APPROACH_M * 1000:.1f} mm.\n"
            f"- OPEN target-jaw distances: `{[round(value * 1000, 6) for value in open_distances]}` mm.\n"
            f"- OPEN minimum robot/bottle distance: {None if min_open is None else f'{min_open * 1000:.6f} mm'}.\n"
            f"- Opposing target contact bracket: `{result['contact_solution'].get('contact_bracket_found')}`; "
            f"target jaw distances at contact: `{[contact.get('narrow_distance_m'), contact.get('wide_distance_m')]}` m.\n"
            f"- 30 mm common-translation corridor: checked statically; no dynamics or physical support claim.\n\n"
            f"{gates}\n\n"
            f"- Rendered views: {', '.join(f'`{name}`' for name in result.get('rendered_images', [])) or 'none'}."
        )
    reproduce = reproduce_command(output_dir, candidate_filter)
    failed_scope = ("Neither bounded alternative passes the complete static corridor. "
                    if payload["candidate_count"] == 2
                    else "The selected geometry hypothesis fails its static gates. ")
    decision = (
        "Static geometry passed at least one candidate. Physical contact and lifting are not run by this "
        "static-stage report; the experiment may proceed only through the separately authorized bounded "
        "physical stages."
        if payload["source_faithful_static_pass"] else
        failed_scope + "No physics was run. This does not "
        "establish global OmniPicker infeasibility. The prior source-derived collision diagnostics remain "
        "the evidence base for any separate model-design proposal; no production model was changed."
    )
    text = (
        "# Issue #46 Outcome-Driven X2 Grasp Corridor\n\n"
        "## Result\n\n"
        f"- Source-faithful static grasp: **{'PASS' if payload['source_faithful_static_pass'] else ('FAIL for both tested alternatives' if payload['candidate_count'] == 2 else 'FAIL for selected candidate')}**.\n"
        f"- Physical bilateral contact: **{payload['physical_bilateral_contact']}**.\n"
        f"- Physical 30 mm lift: **{payload['physical_30mm_lift']}**.\n"
        f"- Production/source geometry, limits, bottle, and physics properties changed: **No**.\n\n"
        f"{decision}\n\n"
        "## Identity\n\n"
        f"- RobotSim branch/HEAD at capture: `{runtime['robotsim_branch']}` / `{runtime['robotsim_head']}`.\n"
        f"- Source corridor helper SHA-256: `{runtime['runner_sha256']}`.\n"
        f"- Outcome evaluator SHA-256: `{runtime['outcome_evaluator_sha256']}`.\n"
        f"- X2 source: `{runtime['upstream_repository']}` at `{runtime['upstream_commit']}`; license: {runtime['license']}.\n"
        f"- URDF `{runtime['urdf']}` SHA-256: `{runtime['urdf_sha256']}`.\n"
        f"- MuJoCo Python/native: `{runtime['mujoco_python']}` / `{runtime['mujoco_native']}`; native library SHA-256: `{runtime['native_library_sha256']}`.\n"
        f"- Canonical helper SHA-256: `{runtime['canonical_scene_helper_sha256']}`.\n"
        f"- Prior corridor reference result SHA-256: `{payload['reference_run04_sha256']}`.\n\n"
        "## Candidates\n\n" + "\n\n".join(candidate_sections) + "\n\n"
        "## Evidence Limits\n\n"
        "The geometry hypotheses share an analytically selected source-bounded near-horizontal wrist orientation "
        "and target the canonical body, then neck where both are included. Compiled source collision geoms and all bottle components are checked "
        "along the OPEN state, pregrasp, approach, closure, and a 30 mm geometric lift query. The mount is "
        "a rigid test fixture; this is not whole-humanoid arm reachability. No physics step was taken, so "
        "the contact aperture is a static geometric bracket, not a force-bearing grasp.\n\n"
        f"## Reproduction\n\n{reproduce}\n"
    )
    path.write_text(text, encoding="utf-8")


def evaluate_candidate(corr, abc, audit, base, extracted_urdf, seed, candidate_spec, out):
    model, data, _ = abc.x2_modules_and_model(audit, base, extracted_urdf, seed)
    root_id = body_id(model, ROOT_LINK)
    root_origin = np.asarray(model.body_pos[root_id], dtype=float).copy()
    root_shift = np.zeros(3)
    yaw, pitch, roll = candidate_spec["wrist_targets_rad"]
    target_geom_name = candidate_spec["target_bottle_geom"]
    target_geom = geom_id(model, target_geom_name)
    approach_direction = np.asarray(candidate_spec["approach_direction_world"], dtype=float)
    set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, 1.0)
    target_center = data.geom_xpos[target_geom].copy()
    root_shift = root_shift + (target_center - jaw_midpoint(model, data))
    contact = find_contact_aperture(corr, abc, base, model, data, root_origin, root_shift,
                                    target_geom, yaw, pitch, roll)
    root_shift = np.asarray(contact["root_shift_world_m"], dtype=float)
    aperture = float(contact.get("aperture_ratio", 1.0))
    arrays = {"summary": [], "pairwise": [], "states": []}

    # OPEN at the target, pregrasp, and the continuous static approach path.
    open_state, open_summary, open_gates, open_details = record_state(
        corr, abc, base, model, data, arrays, candidate_spec["name"], "OPEN", 0,
        root_origin, root_shift, yaw, pitch, roll, 1.0, target_geom_name)
    open_target_distances = [float(r["signed_distance_m"])
                             for r in contact_distances(corr, abc, model, data, target_geom)]
    pregrasp_shift = root_shift + approach_direction * APPROACH_M
    pregrasp_state, _, pregrasp_gates, _ = record_state(
        corr, abc, base, model, data, arrays, candidate_spec["name"], "PREGRASP", 0,
        root_origin, pregrasp_shift, yaw, pitch, roll, 1.0, target_geom_name)
    approach = []
    for index, progress in enumerate(np.linspace(0.0, 1.0, APPROACH_SAMPLES)):
        shift = root_shift + (1.0 - progress) * APPROACH_M * approach_direction
        state, summary, gates, details = record_state(
            corr, abc, base, model, data, arrays, candidate_spec["name"], "APPROACH", index,
            root_origin, shift, yaw, pitch, roll, 1.0, target_geom_name)
        approach.append({"summary": summary, "gates": gates,
                         "robot_bottle_contacts": state["robot_bottle_contacts"]})

    if not contact["contact_bracket_found"]:
        gates = {
            "source_limits_valid": bool(open_details["source_joint_limits"]["all_within_source_limits"]),
            "OPEN_all_robot_bottle_clear": open_details["pair_clearance"]["non_target_clear"],
            "OPEN_table_self_clear": open_gates["robot_table_clearance"] and open_gates["no_robot_self_penetration"],
            "pregrasp_clear": all(pregrasp_gates.values()),
            "approach_clear": all(all(item["gates"].values()) for item in approach),
            "opposing_jaw_contact_on_target": False,
            "closure_no_non_gripping_collision": False,
            "static_30mm_lift_corridor": False,
        }
        result = {
            "candidate": candidate_spec,
            "contact_solution": contact,
            "open": {"summary": open_summary, "gates": open_gates,
                     "target_jaw_distances_m": open_target_distances},
            "pregrasp": {"summary": arrays["summary"][1], "gates": pregrasp_gates},
            "approach": [{"summary": item["summary"], "gates": item["gates"]} for item in approach],
            "closure": {"sample_count": 0, "not_run": "no source aperture bracket for opposing target contact"},
            "static_lift_corridor": [], "gates": gates, "pass": False,
            "physics_steps": 0, "arrays": arrays,
        }
        return result

    # Close only after the source aperture brackets two-sided contact.
    closure = []
    target_distances = []
    for index, ap in enumerate(np.linspace(1.0, aperture, CLOSURE_SAMPLES)):
        state, summary, gates, details = record_state(
            corr, abc, base, model, data, arrays, candidate_spec["name"], "CLOSURE", index,
            root_origin, root_shift, yaw, pitch, roll, float(ap), target_geom_name)
        rows = contact_distances(corr, abc, model, data, target_geom)
        target_distances.append([float(rows[0]["signed_distance_m"]), float(rows[1]["signed_distance_m"])])
        closure.append({"summary": summary, "gates": gates,
                        "non_target_clear": details["pair_clearance"]["non_target_clear"],
                        "target_jaw_distances_m": target_distances[-1],
                        "contacts": state["robot_bottle_contacts"]})

    set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture)
    target_rows = contact_distances(corr, abc, model, data, target_geom)
    narrow_body, wide_body = body_id(model, NARROW), body_id(model, WIDE)
    narrow_pos, wide_pos = data.xpos[narrow_body].copy(), data.xpos[wide_body].copy()
    normal = wide_pos - narrow_pos
    normal /= np.linalg.norm(normal)
    target_center = data.geom_xpos[target_geom].copy()
    witness_offsets = []
    for row in target_rows:
        point = np.asarray(row["witness_segment_world_m"][3:6], dtype=float)
        witness_offsets.append(float(np.dot(point - target_center, normal)))
    opposing = witness_offsets[0] * witness_offsets[1] < 0.0
    monotonic = all(
        target_distances[i + 1][j] <= target_distances[i][j] + 2.0e-5
        for i in range(len(target_distances) - 1) for j in range(2)
    )
    final_clearance = closure[-1]["non_target_clear"]
    contact_ok = (all(abs(float(r["signed_distance_m"])) <= CONTACT_TOL for r in target_rows)
                  and opposing and monotonic and final_clearance)

    # Static common translation of hand and free bottle is only a clearance query.
    lift = []
    for index, height in enumerate(np.linspace(0.0, LIFT_M, LIFT_SAMPLES)):
        state, summary, gates, details = record_state(
            corr, abc, base, model, data, arrays, candidate_spec["name"], "LIFT", index,
            root_origin, root_shift + np.array([0.0, 0.0, height]), yaw, pitch, roll,
            aperture, target_geom_name, float(height))
        bottle_table = corr.bottle_table_min(abc, model, data, corr.collision_sets(abc, model)[1])
        table_clear = (bottle_table is not None and
                       (bottle_table["signed_distance_m"] >= -CLEARANCE_EPS if height == 0.0
                        else bottle_table["signed_distance_m"] > CLEARANCE_EPS))
        gates["bottle_table_clearance"] = table_clear
        lift.append({"summary": summary, "gates": gates,
                     "bottle_table_clearance": bottle_table})

    gates = {
        "source_limits_valid": bool(open_details["source_joint_limits"]["all_within_source_limits"]),
        "OPEN_all_robot_bottle_clear": open_details["pair_clearance"]["non_target_clear"],
        "OPEN_table_self_clear": open_gates["robot_table_clearance"] and open_gates["no_robot_self_penetration"],
        "pregrasp_clear": all(pregrasp_gates.values()),
        "approach_clear": all(all(item["gates"].values()) for item in approach),
        "opposing_jaw_contact_on_target": contact_ok,
        "closure_no_non_gripping_collision": all(item["non_target_clear"] for item in closure),
        "static_30mm_lift_corridor": all(all(item["gates"].values()) for item in lift),
    }
    result = {
        "candidate": candidate_spec,
        "contact_solution": contact,
        "contact_geometry": {
            "target_center_world_m": target_center.tolist(),
            "test_mount_root_position_world_m": (root_origin + root_shift).tolist(),
            "jaw_midpoint_world_m": jaw_midpoint(model, data).tolist(),
            "narrow_distance_m": float(target_rows[0]["signed_distance_m"]),
            "wide_distance_m": float(target_rows[1]["signed_distance_m"]),
            "narrow_witness_offset_along_jaw_normal_m": witness_offsets[0],
            "wide_witness_offset_along_jaw_normal_m": witness_offsets[1],
            "opposite_sides": opposing,
            "jaw_normal_narrow_to_wide_world": normal.tolist(),
        },
        "open": {"summary": open_summary, "gates": open_gates,
                 "target_jaw_distances_m": open_target_distances},
        "pregrasp": {"summary": arrays["summary"][1], "gates": pregrasp_gates},
        "approach": [{"summary": item["summary"], "gates": item["gates"]} for item in approach],
        "closure": {"sample_count": len(closure), "target_distance_trace_m": target_distances,
                    "both_clearances_monotonic": monotonic, "final_non_gripping_clear": final_clearance,
                    "samples": [{"summary": item["summary"], "gates": item["gates"],
                                 "non_target_clear": item["non_target_clear"],
                                 "target_jaw_distances_m": item["target_jaw_distances_m"]}
                                for item in closure]},
        "static_lift_corridor": [{"summary": item["summary"], "gates": item["gates"],
                                  "bottle_table_clearance": item["bottle_table_clearance"]}
                                 for item in lift],
        "gates": gates,
        "pass": all(gates.values()),
        "physics_steps": 0,
        "arrays": arrays,
    }
    return result


def derive_candidates(corr, abc, audit, base, extracted_urdf, prior):
    seed = prior["candidates"][0]["candidate"]
    model, data, _ = abc.x2_modules_and_model(audit, base, extracted_urdf, seed)
    yaw, pitch = 0.0, 0.523599
    set_state(corr, base, model, data, np.asarray(model.body_pos[body_id(model, ROOT_LINK)]),
              np.zeros(3), yaw, pitch, 0.0, 1.0)
    wrist = body_id(model, "right_wrist_roll_link")
    narrow, wide = body_id(model, NARROW), body_id(model, WIDE)
    jaw_normal = data.xpos[wide] - data.xpos[narrow]
    jaw_normal /= np.linalg.norm(jaw_normal)
    wrist_to_tip = jaw_midpoint(model, data) - data.xpos[wrist]
    wrist_to_tip -= jaw_normal * float(np.dot(wrist_to_tip, jaw_normal))
    insertion = -wrist_to_tip / np.linalg.norm(wrist_to_tip)
    roll_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROLL)
    roll_axis = np.asarray(data.xaxis[roll_joint], dtype=float)
    limits = source_joint_range(corr.X2_URDF, ROLL)
    derivation = roll_for_minimum_vertical_insertion(roll_axis, insertion, limits)
    roll = float(derivation["selected_roll_rad"])
    test_specs = [
        {"name": "body_side_near_horizontal_upper_pitch", "target_bottle_geom": "bottle_body",
         "target_region": "canonical cylindrical body side at the compiled body center",
         "wrist_targets_rad": [yaw, pitch, roll], "approach_direction_world": np.asarray(derivation["selected_insertion_axis_world"], dtype=float),
         "roll_derivation": derivation},
        {"name": "neck_side_near_horizontal_upper_pitch", "target_bottle_geom": "bottle_neck",
         "target_region": "canonical neck cylinder center; all body, shoulder, and cap surfaces remain collision checked",
         "wrist_targets_rad": [yaw, pitch, roll], "approach_direction_world": np.asarray(derivation["selected_insertion_axis_world"], dtype=float),
         "roll_derivation": derivation},
    ]
    for spec in test_specs:
        spec["approach_direction_world"] = spec["approach_direction_world"].tolist()
        spec["source_joint_ranges_rad"] = {
            name: source_joint_range(corr.X2_URDF, name) for name in (YAW, PITCH, ROLL)
        }
        spec["root_rotation_world"] = seed["root_rotation_world"]
        spec["root_position_seed_world_m"] = seed["root_position_world_m"]
    return test_specs


def write_candidate_views(corr, abc, audit, base, extracted_urdf, seed, candidate, out):
    model, data, _ = abc.x2_modules_and_model(audit, base, extracted_urdf, seed)
    root_origin = np.asarray(model.body_pos[body_id(model, ROOT_LINK)], dtype=float).copy()
    shift = np.asarray(candidate["contact_solution"]["root_shift_world_m"], dtype=float)
    yaw, pitch, roll = candidate["candidate"]["wrist_targets_rad"]
    has_contact = candidate["contact_solution"].get("contact_bracket_found", False)
    aperture = float(candidate["contact_solution"].get("aperture_ratio", 1.0))
    target = candidate["candidate"]["target_bottle_geom"]
    target_geom = geom_id(model, target)
    robot, bottle, _ = corr.collision_sets(abc, model)
    written = []
    meshes = corr.stl_meshes(abc, extracted_urdf, model, data)
    view_states = [
        ("open", shift, 1.0),
        ("pregrasp", shift + APPROACH_M * np.asarray(candidate["candidate"]["approach_direction_world"]), 1.0),
    ]
    if has_contact:
        view_states.append(("contact", shift, aperture))
    for phase, root_shift, aperture_value in view_states:
        set_state(corr, base, model, data, root_origin, root_shift, yaw, pitch, roll, aperture_value)
        pose = dict(candidate["candidate"])
        pose["name"] = candidate["candidate"]["name"] + "_" + phase
        pose["tcp_target_world_m"] = data.geom_xpos[target_geom].tolist()
        narrow_body, wide_body = body_id(model, NARROW), body_id(model, WIDE)
        jaw_normal = data.xpos[wide_body] - data.xpos[narrow_body]
        jaw_normal /= np.linalg.norm(jaw_normal)
        pose["_frame"] = {
            "jaw_normal_narrow_to_wide_world": jaw_normal.tolist(),
            "insertion_axis_distal_toward_wrist_world": candidate["candidate"]["approach_direction_world"],
        }
        state = corr.collision_state(abc, model, data, robot, bottle, corr.collision_sets(abc, model)[2])
        for view in ("front", "side", "top", "closeup"):
            corr.render_view(out, pose, model, data, meshes, view, state, phase_label=phase.upper())
            written.append(f"{pose['name']}_{view}.png")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", choices=("all", "body_side_near_horizontal_upper_pitch",
                                                "neck_side_near_horizontal_upper_pitch"), default="all")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    corr = load_module("issue46_corridor_runtime", REPO / "scripts/research/issue46_source_faithful_grasp_corridor.py")
    abc = load_module("issue46_corridor_abc", corr.ABC_PATH)
    audit, base = abc.modules()
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    source_dir = out / "source"
    source_dir.mkdir()
    extracted = audit.extract_subtree(source_dir, corr.X2_URDF, ROOT_LINK)
    runtime = corr.source_identity(abc, extracted)
    runtime["outcome_evaluator_sha256"] = corr.sha(Path(__file__).resolve())
    candidates = derive_candidates(corr, abc, audit, base, Path(extracted["path"]), prior)
    if args.candidate != "all":
        candidates = [item for item in candidates if item["name"] == args.candidate]
    reference_seed = prior["candidates"][0]["candidate"]
    results = []
    for candidate in candidates:
        # Recreate the source rig from the accepted centered-jaw pose for each independent candidate.
        result = evaluate_candidate(corr, abc, audit, base, Path(extracted["path"]), reference_seed, candidate, out)
        result["rendered_images"] = write_candidate_views(corr, abc, audit, base, Path(extracted["path"]),
                                                           reference_seed, result, out)
        arrays = result.pop("arrays", None)
        if arrays:
            result["trace_files"] = {
                "state_clearance_summary_csv": f"{candidate['name']}_state_clearance.csv",
                "robot_bottle_pairwise_clearance_csv": f"{candidate['name']}_pairwise.csv",
            }
            write_csv(out / result["trace_files"]["robot_bottle_pairwise_clearance_csv"], arrays["pairwise"])
            write_csv(out / result["trace_files"]["state_clearance_summary_csv"], arrays["summary"])
        results.append(result)

    static_pass = any(item.get("pass", False) for item in results)
    payload = {
        "task": "Issue #46 outcome-driven isolated X2 OmniPicker grasp corridor",
        "selected_path": "B",
        "candidate_filter": args.candidate,
        "source_faithful_static_pass": static_pass,
        "physical_bilateral_contact": "NOT_RUN" if not static_pass else "REQUIRES_BOUNDED_PHYSICAL_STAGE",
        "physical_30mm_lift": "NOT_RUN" if not static_pass else "REQUIRES_BOUNDED_PHYSICAL_STAGE",
        "candidate_count": len(results),
        "production_model_changed": False,
        "source_geometry_changed": False,
        "source_limits_changed": False,
        "physics_steps_total": 0,
        "bottle_qpos_writes_during_physics": 0,
        "source_model_change": False,
        "unity_used": False,
        "runtime": runtime,
        "reference_run04_sha256": corr.sha(PRIOR),
        "candidates": results,
        "scope_limit": "Exactly two geometry-derived hypotheses were evaluated: source-bounded near-horizontal side insertion at the canonical body center and at the canonical neck center. The result does not establish global source-gripper infeasibility.",
    }
    write_json(out / "result.json", payload)
    write_report(out / "REPORT.md", payload, out, args.candidate)
    (out / "REPRODUCE.md").write_text(
        "# Reproduction\n\n"
        "This command performs the two bounded static geometry candidates only. "
        "It does not call `mj_step`. Run it with a new or empty output directory.\n\n"
        f"{reproduce_command(out, args.candidate)}\n\n"
        f"Pinned X2 source: `{corr.X2_ROOT}` at `{runtime['upstream_commit']}`. "
        f"Python/MuJoCo: `{runtime['python']}` / `{runtime['mujoco_python']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_faithful_static_pass": static_pass,
                      "candidate_results": [{"name": r["candidate"]["name"], "pass": r.get("pass", False), "gates": r.get("gates", {}), "contact_bracket": r["contact_solution"].get("contact_bracket_found")} for r in results],
                      "output": str(out)}, indent=2))
    return 0


def reproduce_command(out: Path, candidate: str = "all") -> str:
    candidate_arg = "" if candidate == "all" else f" --candidate {candidate}"
    return (
        "```bash\n"
        f"cd {REPO}\n"
        f"MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \\\n"
        f"  scripts/research/issue46_outcome_grasp_corridor.py \\\n"
        f"  --output {out}{candidate_arg}\n"
        "```"
    )


if __name__ == "__main__":
    raise SystemExit(main())
