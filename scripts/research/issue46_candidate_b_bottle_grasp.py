#!/usr/bin/env python3
"""Bounded bottle grasp/lift test for the current SIMULATION_ONLY Candidate B.

The gripper source model and its one-root-motor experimental coupling are copied
unchanged from the reviewed qCAD=0.20..0.55 model. This script adds a free
cylindrical test bottle and visible support pads. Physical lift runs use a
separate actuated slide fixture carrying only the gripper; no bottle qpos is
written during stepping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
EVIDENCE_ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
SOURCE_PHYSICS = EVIDENCE_ROOT / "candidate-b-clearance-physics-20261022T182958Z"
SOURCE_STATIC = EVIDENCE_ROOT / "candidate-b-clearance-coupling-20261022T182310Z"
CYLINDER_STATIC = EVIDENCE_ROOT / "candidate-b-cylinders-corrected-20261022T182717Z"
MODEL_SOURCE = SOURCE_PHYSICS / "candidate_geometry_coupled.xml"
OFFICIAL_URDF = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
EXPECTED_MODEL_SHA = "b484595375381e4945d3685bf2b2346786b634c6a13b7433f7dfcfbd21252d22"
EXPECTED_OFFICIAL_URDF_SHA = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
EXPECTED_STEP_SHA = "5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b"
STEP_SOURCE = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_op1_cad/OP1-官网模型.STEP")
NOMINAL_Q_OPEN = 0.55
NOMINAL_Q_RETURN = 0.20
ROOT_PHASE = -0.13412598906686654
MAX_SPEED = 5.0
MAX_D_GAP_MM = 0.5
MAX_COUPLING_ERROR = 0.01
MAX_UNEXPECTED_PEN_MM = 0.1
MAX_OBJECT_PEN_MM = 2.0
OBJECT_DIAMETER_MM = 70.0
OBJECT_HEIGHT_MM = 230.0
OBJECT_MASS_KG = 0.55
GRIP_FRICTION = "1.4 0.02 0.001"  # copied from existing Candidate B contact geoms
TABLE_PAD_RADIUS_M = 0.0375
TABLE_PAD_HALF_Z_M = 0.005
TABLE_INITIAL_PRELOAD_MM = 0.05

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "scripts" / "research"))
sys.path.insert(0, str(REPO / "scripts"))
import issue46_candidate_b_clearance_coupling as coupling
import issue46_candidate_b_functional_validation as functional
import issue46_single_motor_travel as travel
import issue46_urdf_pipeline as p
from agent import path_governance as pg
from issue46_dpoint_legal_motion import TriangleBVH, solid_overlap


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(run: Path, name: str, payload: object) -> None:
    pg.write_new_text(run / name, json.dumps(payload, indent=2, allow_nan=False) + "\n", environment="wsl")


def finalize_manifest(run: Path) -> dict:
    artifacts = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name in {".robotsim-output-reservation.json", "artifact_paths.json"}:
            continue
        resolved = pg.validate_output_path(path, require_exists=True, expected_kind="file", environment="wsl")
        artifacts.append({**resolved.as_dict(), "name": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {"schema": "robotsim.artifact-paths.v1", "artifacts": artifacts}
    pg.write_new_text(run / "artifact_paths.json", json.dumps(manifest, indent=2) + "\n", environment="wsl")
    return manifest


def verify_inputs() -> dict:
    expected = {
        MODEL_SOURCE: EXPECTED_MODEL_SHA,
        OFFICIAL_URDF: EXPECTED_OFFICIAL_URDF_SHA,
        STEP_SOURCE: EXPECTED_STEP_SHA,
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"pinned input SHA mismatch: {path}")
    baseline = json.loads((SOURCE_PHYSICS / "physics_result.json").read_text())
    static = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())
    cylinders = json.loads((CYLINDER_STATIC / "cylinder_probe_results.json").read_text())
    if baseline.get("status") != "EXPERIMENTAL_MUJOCO_PASS" or baseline.get("qCAD_tested") != [0.2, 0.55]:
        raise ValueError("source physical run is not the expected bounded qCAD 0.20..0.55 pass")
    if static.get("status") != "PASS_STATIC_GEOMETRY_GATE" or cylinders.get("static_model_sha256") != EXPECTED_MODEL_SHA:
        raise ValueError("source static preflight or 70 mm clearance provenance is invalid")
    cylinder70 = next(x for x in cylinders["full_height_cylinders"] if x["diameter_mm"] == OBJECT_DIAMETER_MM)
    if cylinder70["status"] != "PASS_NO_PROXY_PENETRATION_OR_SOURCE_SURFACE_CROSSING":
        raise ValueError("70 mm cylinder does not have a passing static placement")
    return {"baseline": baseline, "static": static, "cylinders": cylinders}


def parse_poly(model_root: ET.Element, name: str) -> tuple[str, str, list[float]]:
    equality = model_root.find("equality")
    item = next((x for x in equality or [] if x.get("name") == name), None)
    if item is None or item.tag != "joint":
        raise ValueError(f"missing approved coupling {name}")
    return item.get("joint1"), item.get("joint2"), [float(x) for x in item.get("polycoef").split()]


def derivative_at(coeff: list[float], x: float) -> float:
    return sum(i * coeff[i] * x ** (i - 1) for i in range(1, len(coeff)))


def joint_graph() -> tuple[dict[frozenset[str], list[str]], dict[str, tuple[str, str, str]]]:
    root = ET.parse(coupling.BASE_URDF).getroot()
    edges: dict[frozenset[str], list[str]] = {}
    details = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        name = joint.get("name")
        kind = joint.get("type")
        edges.setdefault(frozenset((parent, child)), []).append(name)
        details[name] = (parent, child, kind)
    return edges, details


def triangle_mesh_is_closed(triangles: np.ndarray, decimals: int = 8) -> bool:
    """Conservative undirected-edge multiplicity check for a triangulated STL."""
    verts = np.round(np.asarray(triangles, dtype=np.float64), decimals=decimals)
    edges = np.concatenate((verts[:, [0, 1]], verts[:, [1, 2]], verts[:, [2, 0]]), axis=0)
    reverse = (edges[:, 0, 0] > edges[:, 1, 0]) | (
        (edges[:, 0, 0] == edges[:, 1, 0]) & (edges[:, 0, 1] > edges[:, 1, 1])
    ) | (
        (edges[:, 0, 0] == edges[:, 1, 0]) & (edges[:, 0, 1] == edges[:, 1, 1]) & (edges[:, 0, 2] > edges[:, 1, 2])
    )
    canonical = edges.copy()
    canonical[reverse] = canonical[reverse, ::-1]
    unique, counts = np.unique(canonical.reshape(len(canonical), 6), axis=0, return_counts=True)
    return bool(len(unique) > 0 and np.all(counts == 2))


def mechanical_review(run: Path, sources: dict) -> dict:
    static = sources["static"]
    physics_result = sources["baseline"]
    root = ET.parse(MODEL_SOURCE).getroot()
    narrow_joint, narrow_driver, narrow_coef = parse_poly(root, "geometry_narrow_coupling")
    wide_joint, wide_driver, wide_coef = parse_poly(root, "geometry_wide_coupling")
    if (narrow_joint, narrow_driver) != ("R_hand_narrow2_joint", "right_claw_joint"):
        raise ValueError("narrow equality differs from the reviewed candidate")
    if (wide_joint, wide_driver) != ("R_hand_wide2_joint", "R_hand_wide1_joint"):
        raise ValueError("wide equality differs from the reviewed candidate")
    narrow_turn_root = -narrow_coef[1] / (2.0 * narrow_coef[2])
    narrow_turn_q = ROOT_PHASE - narrow_turn_root
    wide_turn_joint = -wide_coef[1] / (2.0 * wide_coef[2])
    wide_turn_q = ROOT_PHASE + wide_turn_joint  # wide root = -right root

    trace = json.loads((SOURCE_PHYSICS / "physics_trace.json").read_text())
    open_rows = [x for x in trace if x["phase"] == 1 and x.get("joint_positions")]
    open_rows.sort(key=lambda x: x["actual_q_CAD"])
    turning = {}
    for name in ("R_hand_narrow2_joint", "R_hand_wide2_joint", "R_hand_narrow3_joint", "R_hand_wide3_joint",
                 "R_hand_narrow_loop_joint", "R_hand_wide_loop_joint"):
        points = [(float(row["actual_q_CAD"]), float(row["joint_positions"][name])) for row in open_rows]
        slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:]) if b[0] > a[0] + 1e-8]
        reversals = []
        for i, (left, right) in enumerate(zip(slopes, slopes[1:])):
            if left * right < -1e-8:
                reversals.append({"qCAD_mid": points[min(i + 1, len(points) - 1)][0], "slope_before": left, "slope_after": right})
        turning[name] = {"sample_count": len(points), "first_angle_rad": points[0][1], "last_angle_rad": points[-1][1],
                         "observed_slope_reversals": reversals}

    edges, joint_details = joint_graph()
    pairs = static["exact_original_STL_checks"][0]["pairs"]
    closure_pairs = {
        frozenset(("R_hand_narrow3_Link", "R_hand_narrow_loop_Link")): "candidate_narrow_D",
        frozenset(("R_hand_wide3_Link", "R_hand_wide_loop_Link")): "candidate_wide_D",
    }
    exclusions = {frozenset(x) for x in p.INTERNAL_EXCLUSIONS}
    pair_review = []
    for item in pairs:
        key = frozenset((item["body1"], item["body2"]))
        if key in edges:
            relation = {"kind": "direct_URDF_joint_neighbors", "joint_names": edges[key]}
            physical = "candidate bearing/pin/joint mating zone; source-mesh triangle crossing is not volume penetration proof"
        elif key in closure_pairs:
            relation = {"kind": "D_loop_closure_neighbors", "constraint": closure_pairs[key]}
            physical = "candidate D closure mating zone; inherited pair exclusion applies; actual volume/fit remains unresolved"
        else:
            relation = {"kind": "not_directly_connected_in_URDF_or_candidate_D_closure"}
            physical = "unexplained baseline cross-link surface intersection; retain as unresolved mechanical interference candidate"
        pair_review.append({
            "body1": item["body1"], "body2": item["body2"], "source_triangle_indices": [item["source_triangle_a"], item["source_triangle_b"]],
            "joint_graph_relation": relation, "inherited_collision_exclusion": key in exclusions,
            "classification": physical,
            "volume_intersection_measured": False,
        })
    all_pose_pairs = []
    for check in static["exact_original_STL_checks"]:
        all_pose_pairs.append({"qCAD": check["q_CAD"], "pair_count": len(check["pairs"]),
                              "pair_names": sorted([sorted((x["body1"], x["body2"])) for x in check["pairs"]])})
    result = {
        "schema": "robotsim.issue46.candidate-b-mechanical-review.v1",
        "classification": "UNVERIFIED_EXPERIMENTAL",
        "model_sha256": sha(MODEL_SOURCE),
        "candidate_range_reviewed": [0.20, 0.55],
        "quadratic_coupling": {
            "narrow": {"equation": "q_narrow2 = c0 + c1*q_root + c2*q_root^2", "coefficients": narrow_coef,
                       "turn_qCAD_analytic": narrow_turn_q, "dq_droot_at_turn": derivative_at(narrow_coef, narrow_turn_root)},
            "wide": {"equation": "q_wide2 = d0 + d1*q_wide1 + d2*q_wide1^2", "coefficients": wide_coef,
                     "turn_qCAD_analytic": wide_turn_q, "dq_dwide1_at_turn": derivative_at(wide_coef, wide_turn_joint)},
            "turning_point_interpretation": "smooth passive-angle reversal of the polynomial derivative; not by itself a branch jump or singularity",
        },
        "observed_dynamic_passive_motion": turning,
        "physics_trace_summary": {
            "steps": physics_result["physics_steps"], "max_speed_rad_s": physics_result["max_joint_speed_rad_s"],
            "max_constraint_force_mixed_units": physics_result["max_constraint_force_mixed_units"],
            "max_D_gap_mm": physics_result["max_D_gap_mm"], "max_coupling_residual_rad": physics_result["max_geometry_coupling_residual_rad"],
            "solver_warning_stop": physics_result["stop_reason"],
        },
        "ten_baseline_original_STL_crossing_pairs": pair_review,
        "static_sampled_pair_evolution": all_pose_pairs,
        "new_pair_names_vs_baseline": static["new_original_STL_pair_names"],
        "collision_significance_limit": "all ten are source visual STL surface-triangle crossing witnesses; no common-volume calculation or source-design mate approval was performed",
    }
    write_json(run, "mechanical_review.json", result)
    return result


def contacts_for(model, data, bottle_gid: int | None = None) -> list[dict]:
    out = []
    for i, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = np.asarray(contact.frame[:3], dtype=float)
        on_bottle = None
        if bottle_gid is not None and bottle_gid in (g1, g2):
            sign = 1.0 if g2 == bottle_gid else -1.0
            on_bottle = (normal * float(wrench[0]) * sign).tolist()
        other = name2 if bottle_gid == g1 else name1 if bottle_gid == g2 else ""
        side = "narrow" if "R_hand_narrow" in other else "wide" if "R_hand_wide" in other else "table" if "test_table" in other else "base_or_other"
        out.append({
            "geom1": name1, "geom2": name2, "dist_mm": float(contact.dist) * 1000.0,
            "penetration_mm": max(0.0, -float(contact.dist) * 1000.0),
            "normal_geom1_to_geom2_world": normal.tolist(),
            "normal_force_N": float(wrench[0]), "tangential_force_N": float(np.linalg.norm(wrench[1:3])),
            "force_on_bottle_world_N": on_bottle, "bottle_contact_side": side,
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
        })
    return out


def build_contact_model(base_xml: Path, out_xml: Path, bottle_center: tuple[float, float, float], table_z: float,
                        table_radius_m: float = TABLE_PAD_RADIUS_M) -> None:
    tree = ET.parse(base_xml)
    root = tree.getroot()
    root.set("model", "ISOLATED_SIM_ONLY_UNVERIFIED_EXPERIMENTAL_CANDIDATE_B_BOTTLE")
    world = root.find("worldbody")
    robot = world.find("body[@name='R_omnipicker_base_link']")
    if robot is None:
        raise ValueError("expected source Candidate B root body is missing")
    world.remove(robot)
    fixture = ET.SubElement(world, "body", {"name": "isolated_lift_fixture", "mocap": "true"})
    fixture.append(robot)
    ET.SubElement(world, "geom", {
        "name": "test_table", "type": "box",
        "pos": f"{bottle_center[0]:.12g} {bottle_center[1]:.12g} {table_z - TABLE_PAD_HALF_Z_M:.12g}",
        "size": f"{table_radius_m:.12g} {table_radius_m:.12g} {TABLE_PAD_HALF_Z_M:.12g}",
        "contype": "1", "conaffinity": "1", "condim": "4", "friction": GRIP_FRICTION,
        "solref": "0.01 1", "rgba": "0.28 0.30 0.33 1", "group": "0",
    })
    bottle = ET.SubElement(world, "body", {
        "name": "test_bottle", "pos": " ".join(f"{x:.12g}" for x in bottle_center),
    })
    ET.SubElement(bottle, "freejoint", {"name": "test_bottle_free"})
    radius_m = OBJECT_DIAMETER_MM / 2000.0
    length_m = OBJECT_HEIGHT_MM / 1000.0
    inertia_xy = OBJECT_MASS_KG * (3.0 * radius_m * radius_m + length_m * length_m) / 12.0
    inertia_z = OBJECT_MASS_KG * radius_m * radius_m / 2.0
    ET.SubElement(bottle, "inertial", {
        "pos": "0 0 0", "mass": f"{OBJECT_MASS_KG:.12g}",
        "diaginertia": f"{inertia_xy:.12g} {inertia_xy:.12g} {inertia_z:.12g}",
    })
    ET.SubElement(bottle, "geom", {
        "name": "test_bottle_cylinder", "type": "cylinder",
        "size": f"{OBJECT_DIAMETER_MM / 2000.0:.12g} {OBJECT_HEIGHT_MM / 2000.0:.12g}",
        "contype": "1", "conaffinity": "1", "condim": "4",
        "friction": GRIP_FRICTION, "solref": "0.01 1", "rgba": "0.25 0.55 0.92 0.75", "group": "0",
    })
    pg.write_new_text(out_xml, ET.tostring(root, encoding="unicode") + "\n", environment="wsl")


def static_bottle_placement(model, data, derived_urdf: ET.Element, surfaces, q: float, b_cad: float, passive: np.ndarray,
                            y_center: float, table_candidates: list[float]) -> dict:
    addresses = p.addresses(model)
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")])
    candidates = []
    for table_z in table_candidates:
        center = (0.0, float(y_center), float(table_z + OBJECT_HEIGHT_MM / 2000.0 - TABLE_INITIAL_PRELOAD_MM / 1000.0))
        data.qpos[:] = model.qpos0
        coupling.apply_state(model, data, coupling.values_for_q(q, b_cad, passive))
        data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]
        data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        data.geom_xpos[table_gid]  # force a normal Python access to validate compiled geom id
        model.geom_size[table_gid] = [TABLE_PAD_RADIUS_M, TABLE_PAD_RADIUS_M, TABLE_PAD_HALF_Z_M]
        model.geom_pos[table_gid] = [center[0], center[1], float(table_z) - TABLE_PAD_HALF_Z_M]
        data.qpos[addresses.get("test_bottle_free", 0):] if False else None
        free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
        free_adr = int(model.jnt_qposadr[free_id])
        data.qpos[free_adr:free_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        rows = contacts_for(model, data, bottle_gid)
        table_touch = [x for x in rows if table_gid in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, x["geom1"]),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, x["geom2"]),
        )]
        bottle_contacts = [x for x in rows if x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder"]
        fixture_table = [x for x in rows if x["geom1"] == "test_table" or x["geom2"] == "test_table"]
        unexpected_table = [x for x in fixture_table if "test_bottle_cylinder" not in (x["geom1"], x["geom2"])]
        hand_bottle = [x for x in bottle_contacts if x["bottle_contact_side"] in ("narrow", "wide")]
        other_bottle = [x for x in bottle_contacts if x not in hand_bottle and x not in table_touch]
        clear = not unexpected_table and not hand_bottle and not other_bottle and all(x["penetration_mm"] <= MAX_UNEXPECTED_PEN_MM for x in fixture_table)
        candidates.append({
            "table_z_m": float(table_z), "bottle_center_m": list(center), "table_robot_contacts": unexpected_table,
            "initial_hand_bottle_contacts": hand_bottle, "other_bottle_contacts": other_bottle,
            "table_bottle_contacts": [x for x in table_touch if "test_bottle_cylinder" in (x["geom1"], x["geom2"])],
            "max_table_bottle_penetration_mm": max((x["penetration_mm"] for x in table_touch), default=0.0),
            "initial_open_clear": clear,
        })
        if clear and table_touch:
            return {"status": "PASS", "selected": candidates[-1], "tested": candidates}
    return {"status": "BLOCKED_NO_COLLISION_FREE_RESTING_PLACEMENT", "selected": None, "tested": candidates}


def visual_placement_scan(model, data, surfaces, open_values: dict[str, float], bottle_adr: int, bottle_gid: int,
                          table_gid: int, mocap_id: int, band_z_m: float) -> tuple[list[dict], int]:
    """Check actual visual STL crossings/containment across a compact 3D placement grid."""
    source_meshes = []
    for gid in surfaces.local:
        tris = surfaces.triangles(data, gid)
        source_meshes.append({"gid": gid, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                              "triangles": tris, "low": tris.min(axis=(0, 1)), "high": tris.max(axis=(0, 1)),
                              "bvh": TriangleBVH(tris), "closed": triangle_mesh_is_closed(tris)})
    narrow = next(x["triangles"] for x in source_meshes if "narrow3_Link__visual" in x["name"])
    wide = next(x["triangles"] for x in source_meshes if "wide3_Link__visual" in x["name"])
    results, clear_count = [], 0
    for x_center in np.linspace(-0.03, 0.03, 7):
        band = functional.cylinder_band_fit(narrow, wide, OBJECT_DIAMETER_MM, band_z_m,
                                            cx_m=float(x_center), x_samples=17, z_samples=5)
        interval = band.get("feasible_center_y_interval_mm")
        if band.get("clearance_mm") is None or band["clearance_mm"] < 0 or not interval:
            results.append({"x_center_m": float(x_center), "status": "NO_70MM_BAND_FIT", "poses": []})
            continue
        y_values = list(dict.fromkeys(round(float(y) / 1000.0, 8) for y in
                                      (band["best_center_y_mm"], interval[0], interval[1])))
        pose_results = []
        for y_center in y_values:
            for center_z in (0.14029474200784725, 0.165, 0.1886280753411806, 0.205, 0.22):
                center = (float(x_center), float(y_center), float(center_z))
                model.geom_pos[table_gid] = [center[0], center[1], center_z - OBJECT_HEIGHT_MM / 2000.0 - TABLE_PAD_HALF_Z_M]
                data.qpos[:] = model.qpos0
                for name, value in open_values.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    data.qpos[int(model.jnt_qposadr[jid])] = float(value)
                data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]
                data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
                data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(model, data)
                cylinder = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, center)
                cylinder_bvh = TriangleBVH(cylinder)
                low, high = cylinder.min(axis=(0, 1)), cylinder.max(axis=(0, 1))
                overlaps = []
                for source in source_meshes:
                    if np.any(source["low"] > high) or np.any(low > source["high"]):
                        continue
                    overlap = solid_overlap(cylinder_bvh, source["bvh"], closed_a=triangle_mesh_is_closed(cylinder),
                                            closed_b=source["closed"])
                    if overlap.get("surface_intersection_or_contact") or overlap.get("containment") is True or not overlap.get("containment_verified", True):
                        overlaps.append({"geom": source["name"], "source_mesh_closed": source["closed"], **overlap})
                bottle_proxy = [x for x in contacts_for(model, data, bottle_gid)
                                if "test_bottle_cylinder" in (x["geom1"], x["geom2"])
                                and "test_table" not in (x["geom1"], x["geom2"])]
                table_robot = [x for x in contacts_for(model, data, bottle_gid)
                               if "test_table" in (x["geom1"], x["geom2"])
                               and "test_bottle_cylinder" not in (x["geom1"], x["geom2"])]
                visually_clear = not overlaps
                if visually_clear:
                    clear_count += 1
                pose_results.append({"bottle_center_m": list(center), "support_plane_z_m": center_z - OBJECT_HEIGHT_MM / 2000.0,
                                     "band_center_interval_width_mm": band["clearance_mm"],
                                     "collision_proxy_contact_count": len(bottle_proxy),
                                     "collision_proxy_contact_geoms": sorted({c["geom2"] if c["geom1"] == "test_bottle_cylinder" else c["geom1"] for c in bottle_proxy}),
                                     "table_robot_contact_count": len(table_robot),
                                     "source_visual_STL_surface_or_containment_conflicts": overlaps,
                                     "source_visual_clear": visually_clear})
        results.append({"x_center_m": float(x_center), "status": band["status"],
                        "y_center_interval_mm": interval, "poses": pose_results})
    return results, clear_count


def make_contact_preflight(run: Path, sources: dict, review: dict) -> dict:
    pre = sources["static"]
    row = max(pre["static_rows"], key=lambda x: x["q_CAD"])
    source_cylinders = sources["cylinders"]
    band70 = next(x for x in source_cylinders["band_results"] if x["diameter_mm"] == 70.0)["best_band"]
    y_mm = float(band70["best_center_y_mm"])
    open_q = float(row["q_CAD"])
    model_file = run / "bottle_contact_fixture.xml"
    # Search a bounded placement grid over the measured jaw-band center interval.
    # The table remains a real visible collision geom; no pair is disabled.
    table_candidates = [0.105, 0.100, 0.095, 0.090, 0.085, 0.080, 0.0736280753411806,
                        0.05, 0.025, 0.0, -0.025, -0.05, -0.075]
    first_model = run / "bottle_contact_fixture.xml"
    build_contact_model(MODEL_SOURCE, first_model, (0.0, y_mm / 1000.0, table_candidates[0] + OBJECT_HEIGHT_MM / 2000.0), table_candidates[0])
    model = mujoco.MjModel.from_xml_path(str(first_model))
    data = mujoco.MjData(model)
    derived = ET.parse(coupling.BASE_URDF).getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    chosen = None
    tested = []
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
    fixture_mocap = int(model.body_mocapid[fixture_body])
    open_values = coupling.values_for_q(open_q, float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
    data.qpos[:] = model.qpos0
    coupling.apply_state(model, data, open_values)
    data.mocap_pos[fixture_mocap] = [0.0, 0.0, 0.0]
    data.mocap_quat[fixture_mocap] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    source_bvhs = []
    for gid in surfaces.local:
        mesh = surfaces.triangles(data, gid)
        source_bvhs.append({"gid": gid, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                            "mesh": mesh, "low": mesh.min(axis=(0, 1)), "high": mesh.max(axis=(0, 1)), "bvh": TriangleBVH(mesh)})
    band_z_m = 0.14029474200784725
    x_values = np.linspace(-0.03, 0.03, 7)
    for table_z in table_candidates:
        center_z = float(table_z + OBJECT_HEIGHT_MM / 2000.0 - TABLE_INITIAL_PRELOAD_MM / 1000.0)
        if not (center_z - OBJECT_HEIGHT_MM / 2000.0 <= band_z_m <= center_z + OBJECT_HEIGHT_MM / 2000.0):
            continue
        model.geom_size[table_gid] = [TABLE_PAD_RADIUS_M, TABLE_PAD_RADIUS_M, TABLE_PAD_HALF_Z_M]
        center_y_candidates = set()
        for x_center in x_values:
            band = functional.cylinder_band_fit(surfaces.triangles(data, surfaces.gid("narrow")),
                                                surfaces.triangles(data, surfaces.gid("wide")),
                                                OBJECT_DIAMETER_MM, band_z_m, cx_m=float(x_center), x_samples=13, z_samples=5)
            interval = band.get("feasible_center_y_interval_mm")
            if band.get("clearance_mm") is None or band["clearance_mm"] < 0 or not interval:
                continue
            center_y_candidates.update(float(v) / 1000.0 for v in np.linspace(interval[0], interval[1], 5))
            center_y_candidates.add(float(band["best_center_y_mm"]) / 1000.0)
        # Prefer the centered bottle, then scan outward only within measured jaw clearance.
        for x_center in x_values:
            bands = functional.cylinder_band_fit(surfaces.triangles(data, surfaces.gid("narrow")),
                                                 surfaces.triangles(data, surfaces.gid("wide")),
                                                 OBJECT_DIAMETER_MM, band_z_m, cx_m=float(x_center), x_samples=13, z_samples=5)
            interval = bands.get("feasible_center_y_interval_mm")
            if bands.get("clearance_mm") is None or bands["clearance_mm"] < 0 or not interval:
                continue
            y_grid = sorted(center_y_candidates, key=abs)
            for y_center in y_grid:
                if not interval[0] / 1000.0 - 1e-9 <= y_center <= interval[1] / 1000.0 + 1e-9:
                    continue
                center = (float(x_center), float(y_center), center_z)
                model.geom_pos[table_gid] = [center[0], center[1], float(table_z) - TABLE_PAD_HALF_Z_M]
                data.qpos[:] = model.qpos0
                values = coupling.values_for_q(open_q, float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
                coupling.apply_state(model, data, values)
                data.mocap_pos[fixture_mocap] = [0.0, 0.0, 0.0]
                data.mocap_quat[fixture_mocap] = [1.0, 0.0, 0.0, 0.0]
                data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(model, data)
                all_contacts = contacts_for(model, data, bottle_gid)
                table_robot = [x for x in all_contacts if "test_table" in (x["geom1"], x["geom2"]) and "test_bottle_cylinder" not in (x["geom1"], x["geom2"])]
                bottle_contacts = [x for x in all_contacts if "test_bottle_cylinder" in (x["geom1"], x["geom2"])]
                hand_bottle = [x for x in bottle_contacts if x["bottle_contact_side"] in ("narrow", "wide")]
                unexpected_bottle = [x for x in bottle_contacts if x["bottle_contact_side"] not in ("narrow", "wide", "table")]
                cylinder = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, center)
                cyl_low, cyl_high = cylinder.min(axis=(0, 1)), cylinder.max(axis=(0, 1))
                crossings = []
                if not hand_bottle and not table_robot:
                    cylinder_tree = TriangleBVH(cylinder)
                    for source in source_bvhs:
                        if np.any(source["low"] > cyl_high) or np.any(cyl_low > source["high"]):
                            continue
                        hit = source["bvh"].first_intersection(cylinder_tree, eps=1e-8)
                        if hit.get("triangle_a") is not None:
                            crossings.append({"geom": source["name"], **hit})
                table_bottle = [x for x in bottle_contacts if x["bottle_contact_side"] == "table"]
                clear = (not table_robot and not hand_bottle and not unexpected_bottle and not crossings
                         and len(table_bottle) > 0
                         and max((x["penetration_mm"] for x in table_bottle), default=0.0) <= MAX_UNEXPECTED_PEN_MM)
                max_obj_pen = max((x["penetration_mm"] for x in bottle_contacts), default=0.0)
                record = {"table_z_m": float(table_z), "bottle_center_m": list(center),
                          "band_center_interval_width_mm": bands["clearance_mm"],
                          "table_robot_contact_count": len(table_robot), "hand_bottle_contact_count": len(hand_bottle),
                          "other_bottle_contact_count": len(unexpected_bottle), "table_bottle_contact_count": len(table_bottle),
                          "support_preload_mm": TABLE_INITIAL_PRELOAD_MM,
                          "max_bottle_proxy_penetration_mm": max_obj_pen,
                          "sample_hand_bottle_contacts": hand_bottle[:3], "table_robot_contacts": table_robot[:3],
                          "source_visual_STL_crossings": crossings, "initial_open_clear": clear}
                tested.append(record)
                if clear:
                    chosen = record
                    break
            if chosen:
                break
        if chosen:
            model_file = run / "bottle_contact_fixture_selected.xml"
            build_contact_model(MODEL_SOURCE, model_file, tuple(chosen["bottle_center_m"]), float(chosen["table_z_m"]))
            break
    representative_visual_diagnostics = []
    for center_z in (0.1886280753411806, 0.14029474200784725, 0.115):
        center = (0.0, y_mm / 1000.0, center_z)
        data.qpos[:] = model.qpos0
        values = coupling.values_for_q(open_q, float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
        coupling.apply_state(model, data, values)
        data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[fixture_mocap] = [0.0, 0.0, 0.0]
        data.mocap_quat[fixture_mocap] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        cylinder = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, center)
        bottle_tree = TriangleBVH(cylinder)
        low, high = cylinder.min(axis=(0, 1)), cylinder.max(axis=(0, 1))
        mesh_results = []
        for gid in surfaces.local:
            tris = surfaces.triangles(data, gid)
            if np.any(tris.min(axis=(0, 1)) > high) or np.any(low > tris.max(axis=(0, 1))):
                continue
            mesh_tree = TriangleBVH(tris)
            overlap = solid_overlap(bottle_tree, mesh_tree,
                                    closed_a=triangle_mesh_is_closed(cylinder),
                                    closed_b=triangle_mesh_is_closed(tris))
            mesh_results.append({
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                "triangle_count": len(tris),
                "source_mesh_closed": triangle_mesh_is_closed(tris),
                **overlap,
            })
        representative_visual_diagnostics.append({
            "qCAD": open_q, "bottle_center_m": list(center),
            "collision_proxy_contacts": [x for x in contacts_for(model, data, bottle_gid)
                                          if "test_bottle_cylinder" in (x["geom1"], x["geom2"])],
            "source_visual_STL_overlap_tests": mesh_results,
            "qualification": "source visual STL triangle surface intersection plus sampled containment when both triangle shells are closed",
        })
    open_values = coupling.values_for_q(open_q, float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
    visual_pose_grid, visual_clear_count = visual_placement_scan(
        model, data, surfaces, open_values, bottle_adr, bottle_gid, table_gid, fixture_mocap, band_z_m
    )
    if chosen is None:
        result = {
            "schema": "robotsim.issue46.candidate-b-bottle-contact-preflight.v1",
            "status": "PREFLIGHT_BLOCKED",
            "reason": "no upright resting bottle pose was collision-free at the open gripper configuration without fixture/table intersection",
            "tested_placements": tested,
            "representative_source_visual_diagnostics": representative_visual_diagnostics,
            "source_visual_placement_scan": visual_pose_grid,
            "source_visual_clear_placement_count": visual_clear_count,
            "bottle": {"shape": "vertical cylinder", "diameter_mm": OBJECT_DIAMETER_MM, "height_mm": OBJECT_HEIGHT_MM, "mass_kg": OBJECT_MASS_KG},
            "physics_steps": 0,
        }
        write_json(run, "contact_preflight.json", result)
        return result

    # Re-load the selected final file and do a complete initial-state check.
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    open_values = coupling.values_for_q(open_q, float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
    coupling.apply_state(model, data, open_values)
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_bottle")
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")])
    data.qpos[bottle_adr:bottle_adr + 7] = [*chosen["bottle_center_m"], 1.0, 0.0, 0.0, 0.0]
    data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]
    data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    contacts = contacts_for(model, data, bottle_gid)
    source_surfaces = travel.SourceSurfaces(model, derived)
    cyl = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, tuple(chosen["bottle_center_m"]))
    cylinder_bvh = TriangleBVH(cyl)
    source_crossings = []
    c_low, c_high = cyl.min(axis=(0, 1)), cyl.max(axis=(0, 1))
    for gid in source_surfaces.local:
        mesh = source_surfaces.triangles(data, gid)
        if np.any(mesh.min(axis=(0, 1)) > c_high) or np.any(c_low > mesh.max(axis=(0, 1))):
            continue
        hit = TriangleBVH(mesh).first_intersection(cylinder_bvh, eps=1e-8)
        if hit.get("triangle_a") is not None:
            source_crossings.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid), **hit})
    dist = np.asarray(data.qpos[bottle_adr:bottle_adr + 3], dtype=float)
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-contact-preflight.v1",
        "status": "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION" if not source_crossings and not [x for x in contacts if x["bottle_contact_side"] in ("narrow", "wide")] else "PREFLIGHT_BLOCKED",
        "classification": "ISOLATED_SIM_ONLY / UNVERIFIED_EXPERIMENTAL",
        "model_sha256": sha(model_file),
        "source_coupled_model_sha256": sha(MODEL_SOURCE),
        "bottle": {"shape": "vertical solid cylinder approximation", "diameter_mm": OBJECT_DIAMETER_MM, "height_mm": OBJECT_HEIGHT_MM,
                   "mass_kg": OBJECT_MASS_KG, "density_kg_m3": OBJECT_MASS_KG / (math.pi * (OBJECT_DIAMETER_MM / 2000.0) ** 2 * (OBJECT_HEIGHT_MM / 1000.0)),
                   "solid_cylinder_diagonal_inertia_kg_m2": [OBJECT_MASS_KG * (3.0 * (OBJECT_DIAMETER_MM / 2000.0) ** 2 + (OBJECT_HEIGHT_MM / 1000.0) ** 2) / 12.0] * 2 + [OBJECT_MASS_KG * (OBJECT_DIAMETER_MM / 2000.0) ** 2 / 2.0],
                   "center_xyz_m": dist.tolist(), "freejoint": "test_bottle_free"},
        "fixture": {"type": "visible static support plane and mocap carrier for the gripper only", "table_z_m": chosen["table_z_m"],
                    "table_robot_intersection": chosen["table_robot_contacts"], "fixture_isolated_sim_only": True},
        "gripper": {"qCAD": open_q, "joint_positions_urdf_rad": open_values, "active_actuators": int(model.nu),
                    "root_actuator_joint": "right_claw_joint", "passive_actuators": 0},
        "contact_properties": {"hand_and_object_friction": GRIP_FRICTION, "changed_from_candidate_b": False,
                               "collision_exclusions_added": 0, "solver_or_actuator_changes": 0},
        "initial_contacts": contacts,
        "initial_gripper_bottle_contacts": [x for x in contacts if x["bottle_contact_side"] in ("narrow", "wide")],
        "initial_source_STL_surface_crossings": source_crossings,
        "physics_steps": 0,
        "limitations": ["cylinder is a simplified bottle envelope, not a source scan of a commercial bottle",
                        "initial source-STL check detects surface crossings, not containment or common volume"],
    }
    write_json(run, "contact_preflight.json", result)
    return result


def scan_table_fixture_through_stroke(preflight_run: Path) -> tuple[Path, dict]:
    """Find a visible bottle support-pad pose clear of the moving gripper over qCAD 0.20..0.55."""
    preflight = json.loads((preflight_run / "contact_preflight.json").read_text())
    if preflight.get("status") not in ("PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION", "PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION"):
        raise ValueError("fixture scan requires a passing bottle/open-pose preflight")
    model_path = preflight_run / "bottle_contact_fixture_selected.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")])
    sources = verify_inputs()
    static_rows = sorted(sources["static"]["static_rows"], key=lambda row: row["q_CAD"], reverse=True)
    openrow = max(sources["static"]["static_rows"], key=lambda row: row["q_CAD"])
    open_values = coupling.values_for_q(0.55, float(openrow["B_CAD"]), np.asarray(openrow["passive_CAD"]))
    data.qpos[:] = model.qpos0
    for name, value in open_values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
    mocap_id = int(model.body_mocapid[mocap_body])
    data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]
    data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    derived = ET.parse(coupling.BASE_URDF).getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    visual_scan, _ = visual_placement_scan(model, data, surfaces, open_values, bottle_adr, bottle_gid, table_gid, mocap_id, 0.14029474200784725)
    placements = [pose for group in visual_scan for pose in group.get("poses", [])
                  if pose.get("source_visual_clear") and pose.get("collision_proxy_contact_count") == 0]
    candidates = []
    for radius in (TABLE_PAD_RADIUS_M, 0.036, 0.0355, 0.025):
        model.geom_size[table_gid] = [radius, radius, TABLE_PAD_HALF_Z_M]
        for pose in placements:
            x, y, center_z = [float(v) for v in pose["bottle_center_m"]]
            table_top = center_z - OBJECT_HEIGHT_MM / 2000.0 - TABLE_INITIAL_PRELOAD_MM / 1000.0
            max_table_pen = 0.0
            first_collision = None
            pair_union = set()
            for row in static_rows:
                model.geom_pos[table_gid] = [x, y, table_top - TABLE_PAD_HALF_Z_M]
                data.qpos[:] = model.qpos0
                values = coupling.values_for_q(float(row["q_CAD"]), float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
                for name, value in values.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    data.qpos[int(model.jnt_qposadr[jid])] = float(value)
                data.qpos[bottle_adr:bottle_adr + 7] = [x, y, center_z, 1.0, 0.0, 0.0, 0.0]
                data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]
                data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(model, data)
                table_hand = []
                for contact in data.contact:
                    geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
                    if "test_table" not in geoms or "test_bottle_cylinder" in geoms:
                        continue
                    other = geoms[1] if geoms[0] == "test_table" else geoms[0]
                    if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                        table_hand.append({"geom": other, "distance_mm": float(contact.dist) * 1000.0})
                        pair_union.add(other)
                penetration = max((max(0.0, -x["distance_mm"]) for x in table_hand), default=0.0)
                max_table_pen = max(max_table_pen, penetration)
                if table_hand and first_collision is None:
                    first_collision = {"qCAD": float(row["q_CAD"]), "pairs": table_hand[:8]}
            candidates.append({
                "pad_radius_mm": radius * 1000.0,
                "bottle_center_m": [x, y, center_z],
                "table_top_z_m": table_top,
                "static_q_samples": len(static_rows),
                "table_gripper_contacts_any_sample": first_collision is not None,
                "first_contact": first_collision,
                "max_table_gripper_penetration_mm": max_table_pen,
                "gripper_collision_geom_names": sorted(pair_union),
            })
    clear = [x for x in candidates if not x["table_gripper_contacts_any_sample"]]
    run = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-bottle-support-fixture-scan", environment="wsl").native_path
    for filename in ("contact_preflight.json", "mechanical_review.json", "preflight_context.json"):
        source = preflight_run / filename
        pg.write_new_bytes(run / filename, source.read_bytes(), environment="wsl")
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-support-fixture-scan.v1",
        "status": "PASS_FIXTURE_CLEAR_OVER_STATIC_STROKE" if clear else "BLOCKED_TABLE_PAD_INTERFERES_WITH_GRIPPER_DURING_STROKE",
        "preflight_run": str(preflight_run), "source_contact_model_sha256": sha(model_path),
        "candidate_trajectory": [0.20, 0.55], "static_pose_samples": len(static_rows),
        "visible_support_pad": {"shape": "solid cylinder", "radius_mm_tested": list({x["pad_radius_mm"] for x in candidates}),
                                "thickness_mm": TABLE_PAD_HALF_Z_M * 2000.0, "friction": GRIP_FRICTION},
        "clear_candidate_count": len(clear), "clear_candidates": clear,
        "all_candidates": candidates,
        "physics_steps": 0,
        "qualification": "static MuJoCo collision-proxy sweep only; no object or gripper mj_step, no contact dynamics",
    }
    write_json(run, "support_fixture_scan.json", result)
    finalize_manifest(run)
    return run, result


def scan_fixture_lift_path(support_scan_dir: Path, preflight_run: Path) -> tuple[Path, dict]:
    """Static, no-mj_step scan of the gripper/table collision during the proposed fixture lift."""
    scan = json.loads((support_scan_dir / "support_fixture_scan.json").read_text())
    preflight = json.loads((preflight_run / "contact_preflight.json").read_text())
    if preflight.get("status") not in ("PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION", "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION"):
        raise ValueError("fixture-lift sweep requires a valid open bottle pose")
    model_path = preflight_run / "bottle_contact_fixture_selected.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")])
    source_rows = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"]
    grasp_row = min(source_rows, key=lambda row: abs(float(row["q_CAD"]) - 0.32))
    candidates = [x for x in scan["clear_candidates"] if abs(float(x["pad_radius_mm"]) - 25.0) < 1e-6]
    if not candidates:
        candidates = scan["clear_candidates"]
    results = []
    for candidate in candidates:
        x, y, center_z = [float(v) for v in candidate["bottle_center_m"]]
        table_top = float(candidate["table_top_z_m"])
        radius = float(candidate["pad_radius_mm"]) / 1000.0
        model.geom_size[table_gid] = [radius, radius, TABLE_PAD_HALF_Z_M]
        model.geom_pos[table_gid] = [x, y, table_top - TABLE_PAD_HALF_Z_M]
        contacts_by_height = []
        for rise in np.linspace(0.0, 0.060, 13):
            data.qpos[:] = model.qpos0
            values = coupling.values_for_q(float(grasp_row["q_CAD"]), float(grasp_row["B_CAD"]), np.asarray(grasp_row["passive_CAD"]))
            for name, value in values.items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[int(model.jnt_qposadr[jid])] = float(value)
            data.qpos[bottle_adr:bottle_adr + 7] = [x, y, center_z, 1.0, 0.0, 0.0, 0.0]
            data.mocap_pos[mocap_id] = [0.0, 0.0, float(rise)]
            data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            hits = []
            for contact in data.contact:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
                if "test_table" not in (g1, g2) or "test_bottle_cylinder" in (g1, g2):
                    continue
                other = g2 if g1 == "test_table" else g1
                if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                    hits.append({"geom": other, "distance_mm": float(contact.dist) * 1000.0})
            contacts_by_height.append({"fixture_rise_mm": float(rise * 1000.0), "contacts": hits})
        collisions = [x for x in contacts_by_height if x["contacts"]]
        results.append({
            "pad_radius_mm": float(candidate["pad_radius_mm"]), "bottle_center_m": [x, y, center_z],
            "qCAD": float(grasp_row["q_CAD"]), "fixture_rise_samples_mm": [0.0, 60.0, 13],
            "table_gripper_collision_sample_count": len(collisions),
            "first_collision": None if not collisions else collisions[0],
            "collisions": collisions,
            "clear_to_lift_60mm": not collisions,
        })
    run = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-bottle-fixture-lift-scan", environment="wsl").native_path
    for filename in ("support_fixture_scan.json",):
        pg.write_new_bytes(run / filename, (support_scan_dir / filename).read_bytes(), environment="wsl")
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-fixture-lift-scan.v1",
        "status": "PASS_STATIC_FIXTURE_CLEAR_60MM" if any(x["clear_to_lift_60mm"] for x in results) else "BLOCKED_STATIC_FIXTURE_INTERFERENCE",
        "source_candidate_model_sha256": sha(MODEL_SOURCE), "support_scan_run": str(support_scan_dir),
        "fixture_lift_mocap_isolation": True, "physics_steps": 0,
        "candidate_results": results,
        "note": "fixture follows a predeclared +60 mm vertical path; static checks only; no bottle or gripper dynamics were stepped",
    }
    write_json(run, "fixture_lift_scan.json", result)
    finalize_manifest(run)
    return run, result


def scan_small_support_pad_grid(preflight_run: Path, bottle_y_offset_mm: float = 0.0,
                               bottle_x_offset_mm: float = 0.0) -> tuple[Path, dict]:
    """Find clear tabletop support points by compiling every candidate geometry from MJCF."""
    preflight = json.loads((preflight_run / "contact_preflight.json").read_text())
    if preflight.get("status") not in ("PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION", "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION"):
        raise ValueError("pad grid requires a collision-free open-bottle fixture preflight")
    model_source = preflight_run / "bottle_contact_fixture_selected.xml"
    if sha(model_source) != preflight.get("model_sha256"):
        raise ValueError("source preflight model SHA mismatch")
    source_tree = ET.parse(model_source)
    source_root = source_tree.getroot()
    world = source_root.find("worldbody")
    table_template = world.find("geom[@name='test_table']") if world is not None else None
    if table_template is None:
        raise ValueError("visible source support geom test_table is missing")
    static_rows = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"]
    q_rows = sorted(static_rows, key=lambda row: row["q_CAD"], reverse=True)
    # Existing free-bottle runs reached bilateral contact at qCAD≈0.331;
    # sample the actual grasp neighborhood for the lift clearance test.
    grasp_row = min(q_rows, key=lambda row: abs(float(row["q_CAD"]) - 0.331))
    bottle_center = [float(x) for x in preflight["selected_center_xyz_m"]]
    bottle_center[0] += float(bottle_x_offset_mm) / 1000.0
    bottle_center[1] += float(bottle_y_offset_mm) / 1000.0
    table_top = float(preflight["support_pad"]["top_z_m"])
    half = 0.004
    # Expand the support polygon close to, but wholly within, the 35 mm bottle
    # base radius. Each candidate is freshly compiled from MJCF below.
    pad_axis = tuple(sorted(set((-0.028, -0.025, -0.018, -0.015, -0.008, -0.005,
                                 0.005, 0.008, 0.015, 0.018, 0.025, 0.028))))
    offsets = [(float(x), float(y)) for x in pad_axis for y in pad_axis if math.hypot(x, y) <= 0.029]
    pad_results = []
    for dx, dy in offsets:
        tree = ET.parse(model_source)
        root = tree.getroot()
        world = root.find("worldbody")
        table = world.find("geom[@name='test_table']")
        table.set("type", "box")
        table.set("pos", f"{bottle_center[0] + dx:.12g} {bottle_center[1] + dy:.12g} {table_top - TABLE_PAD_HALF_Z_M:.12g}")
        table.set("size", f"{half:.12g} {half:.12g} {TABLE_PAD_HALF_Z_M:.12g}")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_table")
        bottle_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
        bottle_adr = int(model.jnt_qposadr[bottle_jid])
        fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
        mocap_id = int(model.body_mocapid[fixture_body])
        first = None
        max_pen = 0.0

        def set_static_pose(row: dict, rise_m: float = 0.0) -> None:
            data.qpos[:] = model.qpos0
            values = coupling.values_for_q(float(row["q_CAD"]), float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
            for name, value in values.items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[int(model.jnt_qposadr[jid])] = float(value)
            data.qpos[bottle_adr:bottle_adr + 7] = [*bottle_center, 1.0, 0.0, 0.0, 0.0]
            data.mocap_pos[mocap_id] = [0.0, 0.0, float(rise_m)]
            data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)

        def table_gripper_hits() -> list[dict]:
            hits = []
            for contact in data.contact:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
                if "test_table" not in (g1, g2) or "test_bottle_cylinder" in (g1, g2):
                    continue
                other = g2 if g1 == "test_table" else g1
                if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                    hits.append({"geom": other, "distance_mm": float(contact.dist) * 1000.0})
            return hits

        for row in q_rows:
            set_static_pose(row)
            hits = table_gripper_hits()
            if hits and first is None:
                first = {"qCAD": float(row["q_CAD"]), "contacts": hits[:5]}
            max_pen = max(max_pen, max((max(0.0, -x["distance_mm"]) for x in hits), default=0.0))
        for rise in np.linspace(0.0, 0.060, 25):
            set_static_pose(grasp_row, float(rise))
            hits = table_gripper_hits()
            if hits and first is None:
                first = {"fixture_rise_mm": float(rise * 1000.0), "contacts": hits[:5]}
            max_pen = max(max_pen, max((max(0.0, -x["distance_mm"]) for x in hits), default=0.0))
        pad_results.append({"offset_from_bottle_center_mm": [dx * 1000.0, dy * 1000.0],
                            "box_pad_size_mm": [half * 2000.0, half * 2000.0, TABLE_PAD_HALF_Z_M * 2000.0],
                            "first_gripper_contact": first, "max_penetration_mm": max_pen,
                            "clear_over_static_motion_and_60mm_fixture_lift": first is None})
    clear = [x for x in pad_results if x["clear_over_static_motion_and_60mm_fixture_lift"]]
    # Find a wide tripod whose support polygon contains the bottle center.
    tripod = None
    best_area = 0.0
    for triple in itertools.combinations(clear, 3):
        points = np.asarray([x["offset_from_bottle_center_mm"] for x in triple], dtype=float)
        area = abs(float(np.cross(points[1] - points[0], points[2] - points[0]))) / 2.0
        if area <= best_area:
            continue
        matrix = np.vstack((points.T, np.ones(3)))
        try:
            bary = np.linalg.solve(matrix, np.asarray([0.0, 0.0, 1.0]))
        except np.linalg.LinAlgError:
            continue
        if np.all(bary >= 0.05):
            best_area, tripod = area, {"pads": [x["offset_from_bottle_center_mm"] for x in triple], "support_triangle_area_mm2": area,
                                       "center_of_mass_barycentric_weights": bary.tolist()}
    run = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-bottle-support-pad-grid", environment="wsl").native_path
    pg.write_new_bytes(run / "contact_preflight.json", (preflight_run / "contact_preflight.json").read_bytes(), environment="wsl")
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-support-pad-grid.v4",
        "status": "PASS_STATIC_TRIPOD_SUPPORT_PATH" if tripod else "BLOCKED_NO_STABLE_CLEAR_TRIPOD_SUPPORT",
        "pad_half_width_mm": half * 1000.0, "max_pad_center_radius_mm": 29.0,
        "bottle_center_m": bottle_center, "qCAD_static_range": [0.20, 0.55],
        "bottle_y_offset_from_prior_candidate_mm": float(bottle_y_offset_mm),
        "bottle_x_offset_from_prior_candidate_mm": float(bottle_x_offset_mm),
        "fixture_lift_qCAD": float(grasp_row["q_CAD"]), "fixture_lift_mm": 60.0,
        "clear_pad_count": len(clear), "stable_clear_tripod": tripod, "pad_candidates": pad_results,
        "source_model_sha256": sha(model_source),
        "model_handling": "each candidate pad size and pose was written to MJCF and compiled into a fresh MjModel before collision sampling",
        "supersedes": ["candidate-b-bottle-support-pad-grid-20261023T024221Z (v1 mutable-MjModel scan contradicted by compiled tripod preflight)",
                       "candidate-b-bottle-support-pad-grid-20261023T025200Z (v2 used a 26 mm center radius)",
                       "candidate-b-bottle-support-pad-grid-20261023T033341Z (v3 used only the outer 29 mm grid and found a smaller triangle)"],
        "physics_steps": 0,
        "qualification": "static collision-proxy fixture-design scan; no mj_step, no collision exclusion, no bottle force or attachment",
    }
    write_json(run, "support_pad_grid.json", result)
    finalize_manifest(run)
    return run, result


def prepare_tripod_support_candidate(preflight_run: Path, pad_grid_run: Path, bottle_y_offset_mm: float = 0.0,
                                    fixture_z_offset_mm: float = 0.0, bottle_x_offset_mm: float = 0.0) -> tuple[Path, dict]:
    """Build and statically validate three visible support pads under the free bottle."""
    source = json.loads((preflight_run / "contact_preflight.json").read_text())
    grid = json.loads((pad_grid_run / "support_pad_grid.json").read_text())
    if source.get("status") not in ("PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION", "PASS_STATIC_NO_INITIAL_OBJECT_OR_FIXTURE_CONTACT",
                                    "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION"):
        raise ValueError(f"source bottle preflight is not eligible: {source.get('status')}")
    if grid.get("status") != "PASS_STATIC_TRIPOD_SUPPORT_PATH" or not grid.get("stable_clear_tripod"):
        raise ValueError("support-pad grid did not produce a stable clear tripod")
    model_source = preflight_run / "bottle_contact_fixture_selected.xml"
    if sha(model_source) != source.get("model_sha256"):
        raise ValueError("source preflight model SHA mismatch")

    output = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-70mm-bottle-tripod-preflight", environment="wsl").native_path
    tree = ET.parse(model_source)
    root = tree.getroot()
    world = root.find("worldbody")
    table = world.find("geom[@name='test_table']") if world is not None else None
    if table is None:
        raise ValueError("visible source support geom test_table is missing")
    center = [float(x) for x in source["bottle"]["center_xyz_m"]]
    center[0] += float(bottle_x_offset_mm) / 1000.0
    center[1] += float(bottle_y_offset_mm) / 1000.0
    top_z = float(source["support_pad"]["top_z_m"])
    half_xy = float(grid["pad_half_width_mm"]) / 1000.0
    half_z = float(source["support_pad"]["thickness_mm"]) / 2000.0
    offsets = grid["stable_clear_tripod"]["pads"]
    if len(offsets) != 3:
        raise ValueError("expected exactly three support pad locations")
    pad_centers = []
    for index, offset in enumerate(offsets):
        dx, dy = (float(x) / 1000.0 for x in offset)
        if math.hypot(dx, dy) + math.sqrt(2.0) * half_xy >= OBJECT_DIAMETER_MM / 2000.0:
            raise ValueError("support pad extends beyond the bottle base footprint")
        pad_centers.append([center[0] + dx, center[1] + dy, top_z - half_z])
    for i in range(3):
        for j in range(i + 1, 3):
            if math.hypot(pad_centers[i][0] - pad_centers[j][0], pad_centers[i][1] - pad_centers[j][1]) <= 2.0 * half_xy:
                raise ValueError("support pads overlap")
    template = dict(table.attrib)
    template.update({"type": "box", "size": f"{half_xy:.12g} {half_xy:.12g} {half_z:.12g}"})
    bottle_body_xml = world.find("body[@name='test_bottle']")
    if bottle_body_xml is None:
        raise ValueError("free bottle body is missing from source preflight")
    bottle_body_xml.set("pos", " ".join(f"{x:.12g}" for x in center))
    for index, position in enumerate(pad_centers):
        attrs = dict(template)
        attrs["name"] = "test_table" if index == 0 else f"test_table_pad_{index}"
        attrs["pos"] = " ".join(f"{x:.12g}" for x in position)
        if index == 0:
            table.attrib.clear()
            table.attrib.update(attrs)
        else:
            ET.SubElement(world, "geom", attrs)
    candidate_model = output / "bottle_contact_fixture_selected.xml"
    pg.write_new_text(candidate_model, ET.tostring(root, encoding="unicode") + "\n", environment="wsl")
    model = mujoco.MjModel.from_xml_path(str(candidate_model))
    data = mujoco.MjData(model)
    if model.nu != 1 or model.nmocap != 1:
        raise ValueError(f"expected one root actuator and one isolated fixture carrier, got nu={model.nu}, nmocap={model.nmocap}")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
    mocap_id = int(model.body_mocapid[fixture_body])
    static_rows = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"]
    model_sha = sha(candidate_model)

    def set_pose(row: dict, rise_m: float = 0.0) -> None:
        data.qpos[:] = model.qpos0
        values = coupling.values_for_q(float(row["q_CAD"]), float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
        for name, value in values.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
        data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[mocap_id] = [0.0, 0.0, (float(fixture_z_offset_mm) / 1000.0) + float(rise_m)]
        data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    def table_gripper_hits() -> list[dict]:
        hits = []
        for contact in data.contact:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if "test_table" not in (g1, g2) or "test_bottle_cylinder" in (g1, g2):
                continue
            other = g2 if "test_table" in g1 else g1
            if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                hits.append({"pad_geom": g1 if "test_table" in g1 else g2, "gripper_geom": other,
                             "distance_mm": float(contact.dist) * 1000.0})
        return hits

    open_row = max(static_rows, key=lambda row: float(row["q_CAD"]))
    set_pose(open_row)
    open_table_bottle = [x for x in contacts_for(model, data, bottle_gid)
                         if "test_table" in (x["geom1"], x["geom2"])
                         and (x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder")]
    static_collisions = []
    max_d_gap = 0.0
    for row in static_rows:
        set_pose(row)
        max_d_gap = max(max_d_gap, float(np.max(np.abs(coupling.d_residual_mm(model, data)))))
        hits = table_gripper_hits()
        if hits:
            static_collisions.append({"qCAD": float(row["q_CAD"]), "contacts": hits[:8]})
    target_row = min(static_rows, key=lambda row: abs(float(row["q_CAD"]) - 0.331))
    lift_collisions = []
    for rise in np.linspace(0.0, 0.06, 25):
        set_pose(target_row, float(rise))
        max_d_gap = max(max_d_gap, float(np.max(np.abs(coupling.d_residual_mm(model, data)))))
        hits = table_gripper_hits()
        if hits:
            lift_collisions.append({"fixture_rise_mm": float(rise * 1000.0), "contacts": hits[:8]})
    # The preceding sweep leaves `data` at its last lifted pose. Restore the
    # actual open initial configuration before evaluating its bottle contacts.
    set_pose(open_row)
    initial_hand_bottle = [x for x in contacts_for(model, data, bottle_gid)
                           if (x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder")
                           and x["bottle_contact_side"] in ("narrow", "wide")]
    max_d = max(max_d_gap, float(np.max(np.abs(coupling.d_residual_mm(model, data)))))
    surfaces = travel.SourceSurfaces(model, ET.parse(coupling.BASE_URDF).getroot())
    cylinder = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, center)
    cylinder_low, cylinder_high = cylinder.min(axis=(0, 1)), cylinder.max(axis=(0, 1))
    cylinder_tree = TriangleBVH(cylinder)
    source_visual_conflicts = []
    for gid in surfaces.local:
        mesh = surfaces.triangles(data, gid)
        if np.any(mesh.min(axis=(0, 1)) > cylinder_high) or np.any(cylinder_low > mesh.max(axis=(0, 1))):
            continue
        overlap = solid_overlap(cylinder_tree, TriangleBVH(mesh), closed_a=triangle_mesh_is_closed(cylinder),
                                closed_b=triangle_mesh_is_closed(mesh))
        if overlap.get("surface_intersection_or_contact") or overlap.get("containment") is True or not overlap.get("containment_verified", True):
            source_visual_conflicts.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid), **overlap})
    passed = not static_collisions and not lift_collisions and not initial_hand_bottle and not source_visual_conflicts and max_d <= MAX_D_GAP_MM
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-tripod-preflight.v1",
        "status": "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR" if passed else "PREFLIGHT_BLOCKED",
        "classification": "ISOLATED_SIM_ONLY / UNVERIFIED_EXPERIMENTAL",
        "source_preflight": str(preflight_run), "support_pad_grid": str(pad_grid_run),
        "bottle_y_offset_from_prior_candidate_mm": float(bottle_y_offset_mm),
        "bottle_x_offset_from_prior_candidate_mm": float(bottle_x_offset_mm),
        "fixture_initial_z_offset_mm": float(fixture_z_offset_mm),
        "source_model_sha256": sha(MODEL_SOURCE), "model_sha256": model_sha,
        "bottle": {**source["bottle"], "center_xyz_m": center}, "selected_center_xyz_m": center,
        "support_fixture": {"type": "three visible non-collinear square pads", "pad_half_width_mm": half_xy * 1000.0,
                             "pad_thickness_mm": half_z * 2000.0, "top_z_m": top_z,
                             "pad_centers_xyz_m": pad_centers, "bottle_base_footprint_radius_mm": OBJECT_DIAMETER_MM / 2.0,
                             "support_triangle_area_mm2": grid["stable_clear_tripod"]["support_triangle_area_mm2"],
                             "com_barycentric_weights": grid["stable_clear_tripod"]["center_of_mass_barycentric_weights"],
                             "static_open_bottle_support_contacts": open_table_bottle},
        "static_sweep": {"qCAD_range": [min(float(x["q_CAD"]) for x in static_rows), max(float(x["q_CAD"]) for x in static_rows)],
                         "table_gripper_collision_samples": static_collisions},
        "fixture_lift_sweep": {"qCAD": float(target_row["q_CAD"]), "fixture_rise_mm": [0.0, 60.0, 25],
                               "table_gripper_collision_samples": lift_collisions},
        "initial_hand_bottle_contacts": initial_hand_bottle, "initial_source_STL_visual_conflicts": source_visual_conflicts,
        "max_D_gap_mm": max_d, "physics_steps": 0,
        "note": "static MuJoCo collision-proxy preflight only; no mj_step, no exclusions, no bottle attachment or qpos writes during physics",
    }
    for name in ("preflight_context.json", "mechanical_review.json"):
        src = preflight_run / name
        if src.is_file():
            pg.write_new_bytes(output / name, src.read_bytes(), environment="wsl")
    source_copy = dict(source)
    bottle_copy = dict(source["bottle"])
    bottle_copy["center_xyz_m"] = center
    source_copy.update({"status": result["status"], "classification": result["classification"],
                        "bottle": bottle_copy, "selected_center_xyz_m": center,
                        "model_sha256": model_sha, "contact_model_sha256": model_sha,
                        "support_pad": {"shape": "three visible square pads", "top_z_m": top_z,
                                        "thickness_mm": half_z * 2000.0, "outer_width_mm": half_xy * 2000.0,
                                        "changed_gripper_collision": False},
                        "support_fixture": result["support_fixture"],
                        "support_pad_static_sweep": result["static_sweep"],
                        "support_pad_fixture_lift_sweep": result["fixture_lift_sweep"],
                        "fixture_initial_z_offset_mm": float(fixture_z_offset_mm),
                        "initial_table_bottle_contacts": open_table_bottle,
                        "initial_table_gripper_contacts": [], "initial_hand_bottle_contacts": initial_hand_bottle,
                        "initial_source_STL_visual_conflicts": source_visual_conflicts,
                        "max_D_gap_mm": max_d,
                        "physics_steps": 0,
                        "physics_stop_gate": "stop at any table-gripper contact before bilateral bottle contact"})
    write_json(output, "contact_preflight.json", source_copy)
    pg.write_new_bytes(output / "support_pad_grid.json", (pad_grid_run / "support_pad_grid.json").read_bytes(), environment="wsl")
    finalize_manifest(output)
    return output, result


def prepare_dynamic_fixture_candidate(preflight_run: Path) -> tuple[Path, dict]:
    """Replace the kinematic mocap lift carrier with a physical slide fixture."""
    source = json.loads((preflight_run / "contact_preflight.json").read_text())
    if source.get("status") != "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR":
        raise ValueError("dynamic fixture requires a passing tripod preflight")
    model_source = preflight_run / "bottle_contact_fixture_selected.xml"
    if sha(model_source) != source.get("model_sha256"):
        raise ValueError("tripod preflight model SHA mismatch")
    tree = ET.parse(model_source)
    root = tree.getroot()
    world = root.find("worldbody")
    fixture = world.find("body[@name='isolated_lift_fixture']") if world is not None else None
    if fixture is None or fixture.get("mocap") != "true":
        raise ValueError("expected the isolated tripod preflight's mocap carrier")
    fixture.attrib.pop("mocap")
    fixture.insert(0, ET.Element("joint", {
        "name": "isolated_fixture_z_joint", "type": "slide", "axis": "0 0 1",
        "range": "0 0.06", "limited": "true", "damping": "5", "armature": "0.01",
    }))
    fixture.insert(0, ET.Element("inertial", {
        "pos": "0 0 0", "mass": "2", "diaginertia": "0.01 0.01 0.01",
    }))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("source actuator section is missing")
    ET.SubElement(actuator, "position", {
        "name": "isolated_fixture_z_servo", "joint": "isolated_fixture_z_joint",
        "kp": "1000", "kv": "100", "ctrlrange": "0 0.06", "ctrllimited": "true",
        "forcerange": "-100 100", "forcelimited": "true",
    })
    output = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-70mm-bottle-dynamic-fixture-preflight", environment="wsl").native_path
    model_file = output / "bottle_contact_dynamic_fixture.xml"
    pg.write_new_text(model_file, ET.tostring(root, encoding="unicode") + "\n", environment="wsl")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    if model.nmocap != 0 or model.nu != 2:
        raise ValueError(f"expected no mocap and hand+fixture actuators, got nmocap={model.nmocap}, nu={model.nu}")
    fixture_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "isolated_fixture_z_joint")
    fixture_qpos = int(model.jnt_qposadr[fixture_joint])
    fixture_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "isolated_fixture_z_servo")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    root_actuator_ids = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == root_joint]
    if len(root_actuator_ids) != 1 or fixture_actuator < 0 or fixture_actuator == root_actuator_ids[0]:
        raise ValueError("dynamic fixture model must have one gripper root actuator and one independent lift-fixture actuator")
    fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
    static_rows = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"]
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    center = [float(x) for x in source["bottle"]["center_xyz_m"]]
    max_d_gap = 0.0

    def set_pose(row: dict, fixture_z: float = 0.0) -> None:
        data.qpos[:] = model.qpos0
        values = coupling.values_for_q(float(row["q_CAD"]), float(row["B_CAD"]), np.asarray(row["passive_CAD"]))
        for name, value in values.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
        data.qpos[fixture_qpos] = float(fixture_z)
        data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    def table_gripper_hits() -> list[dict]:
        hits = []
        for contact in data.contact:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if "test_table" not in (g1, g2) or "test_bottle_cylinder" in (g1, g2):
                continue
            other = g2 if "test_table" in g1 else g1
            if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                hits.append({"pad_geom": g1 if "test_table" in g1 else g2, "gripper_geom": other,
                             "distance_mm": float(contact.dist) * 1000.0})
        return hits

    open_row = max(static_rows, key=lambda row: float(row["q_CAD"]))
    set_pose(open_row)
    initial_hand_bottle = [x for x in contacts_for(model, data, bottle_gid)
                           if (x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder")
                           and x["bottle_contact_side"] in ("narrow", "wide")]
    static_collisions = []
    for row in static_rows:
        set_pose(row)
        max_d_gap = max(max_d_gap, float(np.max(np.abs(coupling.d_residual_mm(model, data)))))
        hits = table_gripper_hits()
        if hits:
            static_collisions.append({"qCAD": float(row["q_CAD"]), "fixture_z_m": 0.0, "contacts": hits[:8]})
    target_row = min(static_rows, key=lambda row: abs(float(row["q_CAD"]) - 0.331))
    lift_collisions = []
    for fixture_z in np.linspace(0.0, 0.06, 25):
        set_pose(target_row, float(fixture_z))
        max_d_gap = max(max_d_gap, float(np.max(np.abs(coupling.d_residual_mm(model, data)))))
        hits = table_gripper_hits()
        if hits:
            lift_collisions.append({"fixture_z_mm": float(fixture_z * 1000.0), "contacts": hits[:8]})
    passed = not static_collisions and not lift_collisions and not initial_hand_bottle and max_d_gap <= MAX_D_GAP_MM
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-dynamic-fixture-preflight.v1",
        "status": "PASS_DYNAMIC_FIXTURE_STATIC_CLEAR" if passed else "PREFLIGHT_BLOCKED",
        "classification": "ISOLATED_SIM_ONLY / UNVERIFIED_EXPERIMENTAL",
        "source_preflight": str(preflight_run), "source_model_sha256": sha(MODEL_SOURCE),
        "model_sha256": sha(model_file), "candidate_range_qCAD": [min(float(x["q_CAD"]) for x in static_rows), max(float(x["q_CAD"]) for x in static_rows)],
        "fixture": {"type": "physical vertical slide body with position servo", "joint": "isolated_fixture_z_joint",
                    "position_actuator": "isolated_fixture_z_servo", "range_m": [0.0, 0.06], "mass_kg": 2.0,
                    "actuator_kp": 1000.0, "actuator_kv": 100.0, "forcerange_N": [-100.0, 100.0],
                    "purpose": "moves the gripper assembly only; no object force or kinematic mocap motion"},
        "static_table_gripper_collision_samples": static_collisions,
        "fixture_lift_table_gripper_collision_samples": lift_collisions,
        "initial_hand_bottle_contacts": initial_hand_bottle,
        "max_D_gap_mm": max_d_gap, "physics_steps": 0,
        "note": "static collision preflight only; fixture position is set for geometry checks, dynamics use mj_step with a real slide DoF and actuator",
    }
    for name in ("preflight_context.json", "mechanical_review.json", "support_pad_grid.json"):
        src = preflight_run / name
        if src.is_file():
            pg.write_new_bytes(output / name, src.read_bytes(), environment="wsl")
    preflight_copy = dict(source)
    preflight_copy.update({"status": "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR" if passed else "PREFLIGHT_BLOCKED",
                           "model_sha256": sha(model_file), "contact_model_sha256": sha(model_file),
                           "dynamic_fixture": result["fixture"], "fixture_initial_z_offset_mm": 0.0,
                           "support_pad_static_sweep": {"table_gripper_collision_samples": static_collisions},
                           "support_pad_fixture_lift_sweep": {"table_gripper_collision_samples": lift_collisions},
                           "max_D_gap_mm": max_d_gap, "physics_steps": 0})
    write_json(output, "contact_preflight.json", preflight_copy)
    write_json(output, "dynamic_fixture_preflight.json", result)
    finalize_manifest(output)
    return output, result


def prepare_support_candidate(support_scan_dir: Path, y_offset_mm: float = 0.0) -> tuple[Path, dict]:
    scan = json.loads((support_scan_dir / "support_fixture_scan.json").read_text())
    candidates = scan.get("all_candidates", [])
    if not candidates:
        raise ValueError("support fixture scan contains no placements")
    # Select the pad/placement with maximum closing travel before the first
    # predicted table-loop contact. The physics stage will stop at that contact.
    candidate = min(candidates, key=lambda x: (
        abs(float(x["pad_radius_mm"]) - 25.0) + abs(float(x["bottle_center_m"][0]) - 0.03) * 1000.0
        + abs(float(x["bottle_center_m"][1]) + 0.00121691) * 1000.0
    ))
    if candidate.get("first_contact") is None:
        open_run = Path(scan["preflight_run"])
        source_preflight = json.loads((open_run / "contact_preflight.json").read_text())
        center = tuple(source_preflight["bottle"]["center_xyz_m"])
    else:
        candidate_center = tuple(candidate["bottle_center_m"])
        center = (candidate_center[0], candidate_center[1] + y_offset_mm / 1000.0,
                  float(candidate["table_top_z_m"]) + OBJECT_HEIGHT_MM / 2000.0 + TABLE_INITIAL_PRELOAD_MM / 1000.0)
        source_preflight = json.loads((Path(scan["preflight_run"]) / "contact_preflight.json").read_text())
    table_top = float(candidate["table_top_z_m"])
    radius = float(candidate["pad_radius_mm"]) / 1000.0
    output = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-70mm-bottle-fixture-preflight", environment="wsl").native_path
    model_file = output / "bottle_contact_fixture_selected.xml"
    build_contact_model(MODEL_SOURCE, model_file, center, table_top, table_radius_m=radius)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    rows = json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"]
    openrow = max(rows, key=lambda x: x["q_CAD"])
    values = coupling.values_for_q(0.55, float(openrow["B_CAD"]), np.asarray(openrow["passive_CAD"]))
    coupling.apply_state(model, data, values)
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
    mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")])
    data.mocap_pos[mocap] = [0.0, 0.0, 0.0]
    data.mocap_quat[mocap] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    contacts = contacts_for(model, data, bottle_gid)
    initial_table_gripper = [x for x in contacts if "test_table" in (x["geom1"], x["geom2"])
                             and not ({x["geom1"], x["geom2"]} & {"test_bottle_cylinder"})]
    initial_hand_object = [x for x in contacts if x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder"
                           if x["bottle_contact_side"] in ("narrow", "wide")]
    surfaces = travel.SourceSurfaces(model, ET.parse(coupling.BASE_URDF).getroot())
    cylinder = coupling.cylinder_mesh(OBJECT_DIAMETER_MM / 2000.0, OBJECT_HEIGHT_MM / 2000.0, center)
    cylinder_tree = TriangleBVH(cylinder)
    cylinder_low, cylinder_high = cylinder.min(axis=(0, 1)), cylinder.max(axis=(0, 1))
    visual_conflicts = []
    for gid in surfaces.local:
        mesh = surfaces.triangles(data, gid)
        if np.any(mesh.min(axis=(0, 1)) > cylinder_high) or np.any(cylinder_low > mesh.max(axis=(0, 1))):
            continue
        overlap = solid_overlap(cylinder_tree, TriangleBVH(mesh), closed_a=triangle_mesh_is_closed(cylinder),
                                closed_b=triangle_mesh_is_closed(mesh))
        if overlap.get("surface_intersection_or_contact") or overlap.get("containment") is True or not overlap.get("containment_verified", True):
            visual_conflicts.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid), **overlap})
    table_bottle = [x for x in contacts if "test_table" in (x["geom1"], x["geom2"]) and "test_bottle_cylinder" in (x["geom1"], x["geom2"])]
    bottle_bottom = float(center[2] - OBJECT_HEIGHT_MM / 2000.0)
    support_gap_mm = (bottle_bottom - table_top) * 1000.0
    fixture_contact = candidate.get("first_contact")
    # Verify the exact selected table pad over the closing trajectory and along
    # the proposed +60 mm fixture lift, before any bottle physics starts.
    static_rows = sorted(json.loads((SOURCE_STATIC / "static_preflight.json").read_text())["static_rows"],
                         key=lambda item: item["q_CAD"], reverse=True)
    static_table_hits = []
    for motion_row in static_rows:
        data.qpos[:] = model.qpos0
        values = coupling.values_for_q(float(motion_row["q_CAD"]), float(motion_row["B_CAD"]), np.asarray(motion_row["passive_CAD"]))
        for name, value in values.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
        data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[mocap] = [0.0, 0.0, 0.0]
        data.mocap_quat[mocap] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        hits = []
        for contact in data.contact:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if "test_table" in (g1, g2) and "test_bottle_cylinder" not in (g1, g2):
                other = g2 if g1 == "test_table" else g1
                if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                    hits.append({"geom": other, "distance_mm": float(contact.dist) * 1000.0})
        if hits:
            static_table_hits.append({"qCAD": float(motion_row["q_CAD"]), "contacts": hits[:8]})
    first_static_table_hit = static_table_hits[0] if static_table_hits else None
    # At the expected object-contact region qCAD≈0.33, verify 60 mm upward
    # fixture travel does not drive the gripper into the stand.
    lift_q = 0.32
    lift_row = min(static_rows, key=lambda item: abs(float(item["q_CAD"]) - lift_q))
    fixture_lift_table_hits = []
    for dz in np.linspace(0.0, 0.06, 13):
        data.qpos[:] = model.qpos0
        values = coupling.values_for_q(float(lift_row["q_CAD"]), float(lift_row["B_CAD"]), np.asarray(lift_row["passive_CAD"]))
        for name, value in values.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
        data.qpos[bottle_adr:bottle_adr + 7] = [*center, 1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[mocap] = [0.0, 0.0, float(dz)]
        data.mocap_quat[mocap] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        hits = []
        for contact in data.contact:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if "test_table" in (g1, g2) and "test_bottle_cylinder" not in (g1, g2):
                other = g2 if g1 == "test_table" else g1
                if other.startswith("R_hand_") or other.startswith("R_omnipicker_base"):
                    hits.append({"geom": other, "distance_mm": float(contact.dist) * 1000.0})
        if hits:
            fixture_lift_table_hits.append({"fixture_rise_m": float(dz), "contacts": hits[:8]})
    expected_grasp_q = 0.33
    table_clear_to_contact = first_static_table_hit is None or float(first_static_table_hit["qCAD"]) < expected_grasp_q
    fixture_lift_clear = not fixture_lift_table_hits
    status = ("PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR" if not initial_table_gripper and not initial_hand_object
              and not visual_conflicts and -0.1 <= support_gap_mm <= 1.0 and table_clear_to_contact and fixture_lift_clear
              else "PREFLIGHT_BLOCKED")
    result = {
        "schema": "robotsim.issue46.candidate-b-bottle-fixture-preflight.v1",
        "status": status,
        "classification": "ISOLATED_SIM_ONLY / UNVERIFIED_EXPERIMENTAL",
        "source_support_scan": str(support_scan_dir),
        "source_model_sha256": sha(MODEL_SOURCE),
        "model_sha256": sha(model_file),
        "contact_model_sha256": sha(model_file),
        "bottle": {**source_preflight["bottle"], "center_xyz_m": list(center)},
        "selected_center_xyz_m": list(center),
        "support_pad": {"shape": "visible square box pad", "half_width_mm": radius * 1000.0,
                        "outer_width_mm": radius * 2000.0, "thickness_mm": TABLE_PAD_HALF_Z_M * 2000.0,
                        "top_z_m": table_top, "friction": GRIP_FRICTION, "changed_gripper_collision": False},
        "support_pad_static_sweep": {"qCAD_range": [0.20, 0.55], "first_table_gripper_contact": first_static_table_hit,
                                     "clear_through_expected_grasp_qCAD": table_clear_to_contact},
        "support_pad_fixture_lift_sweep": {"qCAD": float(lift_row["q_CAD"]), "fixture_rise_m": [0.0, 0.06, 13],
                                           "clear": fixture_lift_clear, "contacts": fixture_lift_table_hits},
        "qCAD_open": 0.55,
        "initial_table_bottle_contacts": table_bottle,
        "bottle_bottom_to_support_gap_mm": support_gap_mm,
        "initial_table_gripper_contacts": initial_table_gripper,
        "initial_hand_bottle_contacts": initial_hand_object,
        "initial_source_STL_visual_conflicts": visual_conflicts,
        "predicted_first_table_gripper_collision_on_closing": fixture_contact,
        "physics_steps": 0,
        "physics_stop_gate": "stop at the first table-gripper contact if bilateral bottle contact has not already been established",
        "notes": ["support pad is an explicit visible isolated fixture", "support pad is smaller than the bottle base (10 mm radial overhang); tipping/slip is measured, not hidden", "no gripper-table collision pairs are excluded"],
    }
    pg.write_new_bytes(output / "support_fixture_scan.json", (support_scan_dir / "support_fixture_scan.json").read_bytes(), environment="wsl")
    for filename in ("preflight_context.json", "mechanical_review.json"):
        source = support_scan_dir / filename
        if source.is_file():
            pg.write_new_bytes(output / filename, source.read_bytes(), environment="wsl")
    write_json(output, "contact_preflight.json", result)
    finalize_manifest(output)
    return output, result


def grip_only_bias(model, data) -> tuple[float, np.ndarray]:
    """Root bias projection through the hand-only equality nullspace.

    The free bottle adds six unconstrained velocity coordinates, so the original
    whole-model single-nullvector projection is inappropriate. Restricting the
    tangent calculation to the eight hand hinge DOFs preserves the same root
    controller/feedforward and ignores object free-body coordinates.
    """
    eq_mask = np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    if data.efc_J.size != data.nefc * model.nv:
        raise ValueError("expected dense MuJoCo constraint Jacobian")
    full = data.efc_J.reshape(data.nefc, model.nv)[eq_mask]
    joint_names = ["right_claw_joint", "R_hand_narrow2_joint", "R_hand_narrow3_joint", "R_hand_narrow_loop_joint",
                   "R_hand_wide1_joint", "R_hand_wide2_joint", "R_hand_wide3_joint", "R_hand_wide_loop_joint"]
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in joint_names]
    jac = full[:, dofs]
    _, singular, vh = np.linalg.svd(jac, full_matrices=True)
    rank = int(np.linalg.matrix_rank(jac, tol=1e-9))
    tangent = vh[rank:].T[:, 0]
    root_idx = 0
    if abs(tangent[root_idx]) < 1e-8:
        raise ValueError("hand-only equality tangent has no root motion")
    tangent /= tangent[root_idx]
    return float(tangent @ data.qfrc_bias[dofs]), tangent


def collect_state(model, data, qcommand: float, phase: str, step: int, bottle_gid: int, bottle_body: int,
                  bottle_adr: int, fixture_body: int, fixture_qpos: int, root_actuator_id: int,
                  fixture_actuator_id: int, root_qpos: dict[str, int], coupling_coeff: dict[str, list[float]]) -> dict:
    contacts = contacts_for(model, data, bottle_gid)
    fixture_table_contacts = [x for x in contacts if "test_table" in (x["geom1"], x["geom2"])
                              and "test_bottle_cylinder" not in (x["geom1"], x["geom2"])
                              and ("R_hand_" in x["geom1"] or "R_omnipicker_base" in x["geom1"]
                                   or "R_hand_" in x["geom2"] or "R_omnipicker_base" in x["geom2"])]
    self_contacts = [x for x in contacts if "test_bottle_cylinder" not in (x["geom1"], x["geom2"])
                     and "test_table" not in (x["geom1"], x["geom2"])
                     and ("R_hand_" in x["geom1"] or "R_omnipicker_base" in x["geom1"])
                     and ("R_hand_" in x["geom2"] or "R_omnipicker_base" in x["geom2"])]
    bottle_contacts = [x for x in contacts if x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder"]
    narrow = [x for x in bottle_contacts if x["bottle_contact_side"] == "narrow"]
    wide = [x for x in bottle_contacts if x["bottle_contact_side"] == "wide"]
    pos = data.xpos[bottle_body].copy()
    rel_pos = pos - data.xpos[fixture_body]
    root = float(data.qpos[root_qpos["right_claw_joint"]])
    q = ROOT_PHASE - root
    narrow_actual = float(data.qpos[root_qpos["R_hand_narrow2_joint"]])
    wide_actual = float(data.qpos[root_qpos["R_hand_wide2_joint"]])
    narrow_err = narrow_actual - float(np.polynomial.polynomial.polyval(root, coupling_coeff["narrow"]))
    wide1 = float(data.qpos[root_qpos["R_hand_wide1_joint"]])
    wide_err = wide_actual - float(np.polynomial.polynomial.polyval(wide1, coupling_coeff["wide"]))
    gap = float(np.max(np.abs(coupling.d_residual_mm(model, data))))
    return {
        "step": step, "time_s": float(data.time), "phase": phase,
        "command_qCAD": qcommand, "actual_qCAD": q, "root_qpos_rad": root,
        "joint_positions_rad": {name: float(data.qpos[adr]) for name, adr in root_qpos.items()},
        "joint_speeds_rad_s": {name: float(data.qvel[int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])]) for name in root_qpos},
        "D_gap_max_mm": gap, "coupling_residual_rad": {"narrow": narrow_err, "wide": wide_err},
        "root_actuator_force": float(data.actuator_force[root_actuator_id]),
        "root_actuator_control": float(data.ctrl[root_actuator_id]),
        "fixture_z_m": float(data.qpos[fixture_qpos]),
        "fixture_actuator_force_N": float(data.actuator_force[fixture_actuator_id]),
        "equality_constraint_force_max": float(np.max(np.abs(data.efc_force[np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]))
                                                  if np.any(np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) else 0.0),
        "bottle_position_world_m": pos.tolist(),
        "bottle_quaternion_wxyz": data.xquat[bottle_body].tolist(),
        "bottle_position_relative_to_fixture_m": rel_pos.tolist(),
        "bottle_contacts": bottle_contacts,
        "narrow_normal_force_sum_N": sum(max(0.0, x["normal_force_N"]) for x in narrow),
        "wide_normal_force_sum_N": sum(max(0.0, x["normal_force_N"]) for x in wide),
        "narrow_tangential_force_sum_N": sum(x["tangential_force_N"] for x in narrow),
        "wide_tangential_force_sum_N": sum(x["tangential_force_N"] for x in wide),
        "max_object_contact_penetration_mm": max((x["penetration_mm"] for x in bottle_contacts), default=0.0),
        "max_all_contact_penetration_mm": max((max(0.0, -float(c.dist) * 1000.0) for c in data.contact), default=0.0),
        "contacts": contacts,
        "fixture_table_gripper_contacts": fixture_table_contacts,
        "gripper_self_contacts": self_contacts,
        "mujoco_warning_numbers": [int(w.number) for w in data.warning],
    }


def contact_run(run: Path, preflight_run: Path, preflight: dict, review: dict, control_variant: str = "baseline",
                lift_slip_stop_mm: float = 5.0) -> dict:
    if preflight.get("status") not in ("PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION", "PASS_STATIC_NO_INITIAL_OBJECT_OR_FIXTURE_CONTACT", "PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION", "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR"):
        raise ValueError("contact physics is gated on collision-free bottle placement")
    if not preflight.get("dynamic_fixture"):
        raise ValueError("physics lift requires the preflighted physical slide fixture, not a mocap carrier")
    model_path = preflight_run / "bottle_contact_dynamic_fixture.xml"
    if sha(model_path) != preflight["model_sha256"]:
        raise ValueError("contact preflight model SHA mismatch")
    if control_variant not in ("baseline", "kp2_diagnostic", "kp4_diagnostic"):
        raise ValueError(f"unknown root controller diagnostic: {control_variant}")
    if not math.isfinite(lift_slip_stop_mm) or not 5.0 <= lift_slip_stop_mm <= 20.0:
        raise ValueError("lift slip diagnostic threshold must be between 5 and 20 mm")
    if control_variant == "baseline":
        copy_path = run / model_path.name
        pg.write_new_bytes(copy_path, model_path.read_bytes(), environment="wsl")
    else:
        tree = ET.parse(model_path)
        actuator = tree.getroot().find("actuator/position[@joint='right_claw_joint']")
        if actuator is None:
            raise ValueError("root position actuator missing from contact model")
        original = {key: actuator.get(key) for key in ("kp", "kv", "forcerange", "ctrlrange")}
        diagnostic_kp = 2.0 if control_variant == "kp2_diagnostic" else 4.0
        actuator.set("kp", f"{diagnostic_kp:.1f}")
        copy_path = run / f"candidate_root_kp{int(diagnostic_kp)}_diagnostic.xml"
        pg.write_new_text(copy_path, ET.tostring(tree.getroot(), encoding="unicode") + "\n", environment="wsl")
        write_json(run, "controller_variant.json", {
            "classification": "CONTROL_DIAGNOSTIC_ONLY / UNVERIFIED_EXPERIMENTAL",
            "changed": {"root_actuator_kp": {"baseline": float(original["kp"]), "diagnostic": diagnostic_kp}},
            "unchanged": {"root_actuator_kv": float(original["kv"]), "forcerange": original["forcerange"],
                          "ctrlrange": original["ctrlrange"], "hand_friction": GRIP_FRICTION,
                          "constraint_stiffness": "unchanged", "mechanical_model": "unchanged"},
            "purpose": "diagnose root position-servo stiffness under physical contact; not nominal acceptance and not a promoted model",
        })
    model = mujoco.MjModel.from_xml_path(str(copy_path))
    data = mujoco.MjData(model)
    if model.nu != 2 or model.nmocap != 0:
        raise ValueError(f"expected one gripper actuator plus one physical fixture actuator and no mocap, got nu={model.nu}, nmocap={model.nmocap}")
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    root_actuator_ids = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == root_joint_id]
    fixture_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "isolated_fixture_z_joint")
    fixture_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "isolated_fixture_z_servo")
    if len(root_actuator_ids) != 1 or fixture_actuator_id < 0 or fixture_actuator_id in root_actuator_ids:
        raise ValueError("expected distinct right-root and lift-fixture actuators")
    root_actuator_id = root_actuator_ids[0]
    fixture_qpos = int(model.jnt_qposadr[fixture_joint_id])
    root_names = ["right_claw_joint", "R_hand_narrow2_joint", "R_hand_narrow3_joint", "R_hand_narrow_loop_joint",
                  "R_hand_wide1_joint", "R_hand_wide2_joint", "R_hand_wide3_joint", "R_hand_wide_loop_joint"]
    root_qpos = {name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in root_names}
    root_dofs = {name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in root_names}
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_bottle_free")
    bottle_adr = int(model.jnt_qposadr[bottle_joint])
    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_bottle")
    bottle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_bottle_cylinder")
    fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "isolated_lift_fixture")
    if fixture_body < 0:
        raise ValueError("physical isolated fixture body is missing")
    sources = json.loads((CYLINDER_STATIC / "cylinder_probe_results.json").read_text())
    openrow = max(review["source_static_rows"], key=lambda x: x["q_CAD"])
    start_values = coupling.values_for_q(NOMINAL_Q_OPEN, float(openrow["B_CAD"]), np.asarray(openrow["passive_CAD"]))
    data.qpos[:] = model.qpos0
    coupling.apply_state(model, data, start_values)
    init_center = preflight["bottle"]["center_xyz_m"]
    data.qpos[bottle_adr:bottle_adr + 7] = [*init_center, 1.0, 0.0, 0.0, 0.0]
    data.qpos[fixture_qpos] = 0.0
    data.ctrl[fixture_actuator_id] = 0.0
    mujoco.mj_forward(model, data)
    max_initial_d = float(np.max(np.abs(coupling.d_residual_mm(model, data))))
    if max_initial_d > MAX_D_GAP_MM:
        raise ValueError(f"D closure fails at open initial pose: {max_initial_d} mm")

    from PIL import Image, ImageDraw, ImageFont

    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.14]
    camera.distance, camera.azimuth, camera.elevation = 0.40, 0.0, -10.0
    options = mujoco.MjvOption()
    options.geomgroup[1] = 0
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    frames = []
    trace = []
    phases = []
    first_contacts = {"narrow": None, "wide": None}
    bilateral_streak = 0
    bilateral_q = None
    hold_rows = []
    lift_relative_start = None
    lift_peak = 0.0
    stop = None
    step = 0
    q_target = NOMINAL_Q_OPEN
    open_row = max(review["source_static_rows"], key=lambda x: x["q_CAD"])
    coeff = review["coupling_coefficients"]
    phases_to_run = [
        ("OPEN_SETTLE", NOMINAL_Q_OPEN, 0.50),
        ("CLOSE_TO_CONTACT", NOMINAL_Q_RETURN, 2.00),
        ("CLOSE_TARGET_HOLD", NOMINAL_Q_RETURN, 2.00),
    ]
    active_phases = list(phases_to_run)
    close_complete = False
    for phase_index, (phase, target_q, duration) in enumerate(active_phases):
        start_q = ROOT_PHASE - float(data.qpos[root_qpos["right_claw_joint"]])
        for i in range(round(duration / float(model.opt.timestep))):
            if phase == "CLOSE_TARGET_HOLD":
                q_command = target_q
            else:
                alpha = min(i * float(model.opt.timestep) / max(duration - 0.5 if phase == "CLOSE_TO_CONTACT" else duration, 1e-6), 1.0)
                blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
                q_command = start_q + (target_q - start_q) * blend
            root_target = ROOT_PHASE - q_command
            feedforward, _ = grip_only_bias(model, data)
            data.ctrl[root_actuator_id] = float(np.clip(root_target + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
            mujoco.mj_step(model, data)
            step += 1
            mujoco.mj_forward(model, data)
            row = collect_state(model, data, q_command, phase, step, bottle_gid, bottle_body, bottle_adr,
                                fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff)
            trace.append(row)
            for side in ("narrow", "wide"):
                force_key = f"{side}_normal_force_sum_N"
                if first_contacts[side] is None and row[force_key] >= 0.25:
                    first_contacts[side] = {"time_s": row["time_s"], "qCAD": row["actual_qCAD"], "force_N": row[force_key],
                                            "bottle_contacts": [c for c in row["bottle_contacts"] if c["bottle_contact_side"] == side]}
            bilateral = row["narrow_normal_force_sum_N"] >= 0.25 and row["wide_normal_force_sum_N"] >= 0.25
            bilateral_streak = bilateral_streak + 1 if bilateral else 0
            if row["fixture_table_gripper_contacts"]:
                stop = "ISOLATED_FIXTURE_TABLE_COLLISION"
            elif any(x["penetration_mm"] > MAX_UNEXPECTED_PEN_MM for x in row["gripper_self_contacts"]):
                stop = "GRIPPER_SELF_COLLISION"
            elif row["max_object_contact_penetration_mm"] > MAX_OBJECT_PEN_MM:
                stop = "OBJECT_CONTACT_PENETRATION_LIMIT"
            elif row["max_all_contact_penetration_mm"] > MAX_OBJECT_PEN_MM:
                stop = "CONTACT_PENETRATION_LIMIT"
            elif row["D_gap_max_mm"] > MAX_D_GAP_MM:
                stop = "D_POINT_CLOSURE_LOST"
            elif max(abs(x) for x in row["coupling_residual_rad"].values()) > MAX_COUPLING_ERROR:
                stop = "COUPLING_RESIDUAL"
            elif max(abs(x) for x in row["joint_speeds_rad_s"].values()) > MAX_SPEED:
                stop = "EXCESSIVE_JOINT_SPEED"
            elif any(row["mujoco_warning_numbers"]):
                stop = "MUJOCO_WARNING"
            if stop is None and bilateral_streak >= 25 and phase in ("CLOSE_TO_CONTACT", "CLOSE_TARGET_HOLD"):
                bilateral_q = row["actual_qCAD"]
                phases.append({"phase": phase, "completed": True, "stop": "BILATERAL_CONTACT_DETECTED", "qCAD": bilateral_q, "step": step})
                close_complete = True
                break
            if step % 40 == 0 or stop or (bilateral_streak == 1):
                renderer.update_scene(data, camera=camera, scene_option=options)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 720, 64), fill=(12, 20, 28))
                draw.text((8, 5), "ISOLATED SIM ONLY / DYNAMIC SLIDE FIXTURE / FREE BOTTLE", font=font, fill=(255, 207, 112))
                draw.text((8, 26), f"{phase} t={data.time:.2f}s qCAD={row['actual_qCAD']:+.3f} m={OBJECT_MASS_KG:.2f}kg", font=font, fill="white")
                draw.text((8, 45), f"N/W={row['narrow_normal_force_sum_N']:.2f}/{row['wide_normal_force_sum_N']:.2f}N D={row['D_gap_max_mm']:.4f}mm", font=font, fill="white")
                frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
            if stop:
                break
        if not close_complete:
            actual_phase_end = ROOT_PHASE - float(data.qpos[root_qpos["right_claw_joint"]])
            phase_reached = abs(actual_phase_end - target_q) <= 0.02
            phases.append({"phase": phase, "completed": stop is None and (phase_reached or phase == "CLOSE_TO_CONTACT"),
                           "stop": stop, "command_end_qCAD": target_q, "actual_end_qCAD": actual_phase_end})
            if stop is None and phase == "CLOSE_TARGET_HOLD" and not phase_reached:
                stop = "CLOSE_TARGET_NOT_REACHED"
        if stop:
            break
        if close_complete:
            break
        if phase == "OPEN_SETTLE" and abs((ROOT_PHASE - float(data.qpos[root_qpos["right_claw_joint"]])) - target_q) > 0.02:
            stop = "OPEN_SETTLE_TARGET_NOT_REACHED"
            break

    hold_metrics = {"status": "NOT_REACHED", "sample_count": 0}
    if close_complete and not stop:
        # Keep commanding the already requested close proportion while the
        # bottle blocks further motion. Holding the contact pose itself removes
        # motor preload and lets a one-sided contact relax; this is still the
        # same single root servo and effort cap, with no passive targets.
        q_hold = NOMINAL_Q_RETURN
        for _ in range(round(1.0 / float(model.opt.timestep))):
            root_target = ROOT_PHASE - q_hold
            feedforward, _ = grip_only_bias(model, data)
            data.ctrl[root_actuator_id] = float(np.clip(root_target + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
            mujoco.mj_step(model, data)
            step += 1
            mujoco.mj_forward(model, data)
            row = collect_state(model, data, q_hold, "BILATERAL_HOLD_ON_TABLE", step, bottle_gid, bottle_body, bottle_adr,
                                fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff)
            trace.append(row)
            if step % 10 == 0:
                hold_rows.append(row)
            if max(abs(x) for x in row["coupling_residual_rad"].values()) > MAX_COUPLING_ERROR or row["D_gap_max_mm"] > MAX_D_GAP_MM:
                stop = "HOLD_CONSTRAINT_FAILURE"
                break
            if row["fixture_table_gripper_contacts"] or any(x["penetration_mm"] > MAX_UNEXPECTED_PEN_MM for x in row["gripper_self_contacts"]):
                stop = "HOLD_FIXTURE_OR_SELF_COLLISION"
                break
            if row["max_object_contact_penetration_mm"] > MAX_OBJECT_PEN_MM:
                stop = "OBJECT_CONTACT_PENETRATION_LIMIT"
                break
        if hold_rows:
            bilateral_fraction = sum(x["narrow_normal_force_sum_N"] >= 0.25 and x["wide_normal_force_sum_N"] >= 0.25 for x in hold_rows) / len(hold_rows)
            positions = np.asarray([x["bottle_position_relative_to_fixture_m"] for x in hold_rows])
            relative_slip = float(np.max(np.linalg.norm(positions - positions[0], axis=1)) * 1000.0)
            hold_metrics = {
                "status": "PASS" if bilateral_fraction >= 0.95 and relative_slip <= 5.0 and stop is None else "FAIL",
                "sample_count": len(hold_rows), "bilateral_contact_fraction": bilateral_fraction,
                "relative_bottle_slip_mm": relative_slip,
                "narrow_min_force_N": min(x["narrow_normal_force_sum_N"] for x in hold_rows),
                "wide_min_force_N": min(x["wide_normal_force_sum_N"] for x in hold_rows),
                "root_actuator_force_max_abs": max(abs(x["root_actuator_force"]) for x in hold_rows),
            }
        else:
            hold_metrics = {"status": "FAIL_NO_BILATERAL_HOLD_SAMPLES", "sample_count": 0}

    lift_metrics = {"status": "NOT_RUN", "reason": "requires stable bilateral hold"}
    release_metrics = {"status": "NOT_RUN", "reason": "requires completed lift stage"}
    # Only move the isolated gripper fixture after a stable bilateral hold.
    if hold_metrics.get("status") == "PASS" and not stop:
        initial_fixture_z = float(data.qpos[fixture_qpos])
        initial_fixture_world = data.xpos[fixture_body].copy()
        initial_bottle = data.xpos[bottle_body].copy()
        initial_bottle_fixture_relative = initial_bottle - initial_fixture_world
        lift_start = step
        lift_steps = round(3.0 / float(model.opt.timestep))
        lift_stop = None
        lift_rows = []
        bilateral_lost_steps = 0
        max_lift_relative_slip_mm = 0.0
        max_lift_d_gap_mm = 0.0
        max_lift_coupling_error = 0.0
        for i in range(lift_steps):
            alpha = min(i / max(lift_steps - 1, 1), 1.0)
            blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
            data.ctrl[fixture_actuator_id] = initial_fixture_z + 0.060 * blend
            root_target = ROOT_PHASE - q_hold
            feedforward, _ = grip_only_bias(model, data)
            data.ctrl[root_actuator_id] = float(np.clip(root_target + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
            mujoco.mj_step(model, data)
            step += 1
            mujoco.mj_forward(model, data)
            row = collect_state(model, data, q_hold, "FIXTURE_LIFT", step, bottle_gid, bottle_body, bottle_adr,
                                fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff)
            trace.append(row)
            if step % 10 == 0:
                lift_rows.append(row)
            relative_slip_mm = float(np.linalg.norm(
                (data.xpos[bottle_body] - data.xpos[fixture_body]) - initial_bottle_fixture_relative) * 1000.0)
            max_lift_relative_slip_mm = max(max_lift_relative_slip_mm, relative_slip_mm)
            bilateral = row["narrow_normal_force_sum_N"] >= 0.25 and row["wide_normal_force_sum_N"] >= 0.25
            bilateral_lost_steps = 0 if bilateral else bilateral_lost_steps + 1
            max_lift_d_gap_mm = max(max_lift_d_gap_mm, row["D_gap_max_mm"])
            max_lift_coupling_error = max(max_lift_coupling_error, max(abs(x) for x in row["coupling_residual_rad"].values()))
            if row["max_object_contact_penetration_mm"] > MAX_OBJECT_PEN_MM:
                lift_stop = "OBJECT_CONTACT_PENETRATION_LIMIT"
            elif row["fixture_table_gripper_contacts"]:
                lift_stop = "ISOLATED_FIXTURE_TABLE_COLLISION"
            elif any(x["penetration_mm"] > MAX_UNEXPECTED_PEN_MM for x in row["gripper_self_contacts"]):
                lift_stop = "GRIPPER_SELF_COLLISION"
            elif row["D_gap_max_mm"] > MAX_D_GAP_MM:
                lift_stop = "D_POINT_CLOSURE_LOST"
            elif max(abs(x) for x in row["coupling_residual_rad"].values()) > MAX_COUPLING_ERROR:
                lift_stop = "COUPLING_RESIDUAL"
            elif max(abs(x) for x in row["joint_speeds_rad_s"].values()) > MAX_SPEED:
                lift_stop = "EXCESSIVE_JOINT_SPEED"
            elif any(row["mujoco_warning_numbers"]):
                lift_stop = "MUJOCO_WARNING"
            elif relative_slip_mm > lift_slip_stop_mm:
                lift_stop = "BOTTLE_RELATIVE_SLIP_LIMIT"
            elif ((data.qpos[fixture_qpos] - initial_fixture_z) >= 0.020
                  and (data.xpos[bottle_body, 2] - initial_bottle[2]) < 0.005
                  and any(x["bottle_contact_side"] == "table" for x in row["bottle_contacts"])):
                lift_stop = "NO_LOAD_TRANSFER_BY_FIXTURE_20MM"
            elif bilateral_lost_steps >= 25:
                lift_stop = "BILATERAL_GRASP_LOST"
            if step % 40 == 0 or lift_stop:
                renderer.update_scene(data, camera=camera, scene_option=options)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 720, 64), fill=(12, 20, 28))
                draw.text((8, 5), "ISOLATED SIM ONLY / PHYSICAL SLIDE FIXTURE LIFT", font=font, fill=(255, 207, 112))
                draw.text((8, 26), f"t={data.time:.2f}s fixture=+{data.qpos[fixture_qpos]-initial_fixture_z:.3f}m bottleDz={data.xpos[bottle_body,2]-initial_bottle[2]:+.3f}m", font=font, fill="white")
                draw.text((8, 45), f"qCAD={row['actual_qCAD']:+.3f} D={row['D_gap_max_mm']:.4f}mm penetration={row['max_object_contact_penetration_mm']:.3f}mm", font=font, fill="white")
                frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
            if lift_stop:
                break
        max_dz = float(data.xpos[bottle_body, 2] - initial_bottle[2])
        table_touch = [x for x in contacts_for(model, data, bottle_gid) if x["bottle_contact_side"] == "table"]
        bilateral_fraction = sum(x["narrow_normal_force_sum_N"] >= 0.25 and x["wide_normal_force_sum_N"] >= 0.25 for x in lift_rows) / max(1, len(lift_rows))
        lift_metrics = {
            "status": "PASS" if lift_stop is None and max_dz >= 0.05 and not table_touch and bilateral_fraction >= 0.95 else "FAIL",
            "stop_reason": lift_stop, "steps": step - lift_start,
            "fixture_rise_mm": float((data.qpos[fixture_qpos] - initial_fixture_z) * 1000.0),
            "bottle_world_rise_mm": max_dz * 1000.0,
            "bottle_relative_to_fixture_end_mm": (data.xpos[bottle_body] - data.xpos[fixture_body]).tolist(),
            "max_bottle_relative_slip_mm": max_lift_relative_slip_mm,
            "relative_slip_stop_threshold_mm": float(lift_slip_stop_mm),
            "bilateral_contact_lost_consecutive_steps": bilateral_lost_steps,
            "max_D_gap_mm": max_lift_d_gap_mm,
            "max_coupling_residual_rad": max_lift_coupling_error,
            "bilateral_contact_fraction": bilateral_fraction,
            "table_contacts_at_lift_end": table_touch,
            "bottle_contacts_at_lift_end": [x for x in contacts_for(model, data, bottle_gid) if x["geom1"] == "test_bottle_cylinder" or x["geom2"] == "test_bottle_cylinder"],
        }
        if lift_metrics["status"] == "PASS" and not lift_stop:
            # Hold at the raised physical fixture position for one second.
            for _ in range(round(1.0 / float(model.opt.timestep))):
                data.ctrl[fixture_actuator_id] = initial_fixture_z + 0.060
                feedforward, _ = grip_only_bias(model, data)
                data.ctrl[root_actuator_id] = float(np.clip(ROOT_PHASE - q_hold + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
                mujoco.mj_step(model, data)
                step += 1
                mujoco.mj_forward(model, data)
                trace.append(collect_state(model, data, q_hold, "LIFT_HOLD", step, bottle_gid, bottle_body, bottle_adr,
                                           fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff))
            # Lower the physical slide back to its initial pose before opening.
            lower_initial_z = float(data.qpos[fixture_qpos])
            for i in range(lift_steps):
                alpha = min(i / max(lift_steps - 1, 1), 1.0)
                blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
                data.ctrl[fixture_actuator_id] = max(initial_fixture_z, lower_initial_z - 0.060 * blend)
                feedforward, _ = grip_only_bias(model, data)
                data.ctrl[root_actuator_id] = float(np.clip(ROOT_PHASE - q_hold + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
                mujoco.mj_step(model, data)
                step += 1
                mujoco.mj_forward(model, data)
                trace.append(collect_state(model, data, q_hold, "FIXTURE_LOWER", step, bottle_gid, bottle_body, bottle_adr,
                                           fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff))
            # Open back to the already validated qCAD=0.55 configuration.
            open_steps = round(1.5 / float(model.opt.timestep))
            start_open_q = ROOT_PHASE - float(data.qpos[root_qpos["right_claw_joint"]])
            for i in range(open_steps):
                alpha = min(i / max(open_steps - 1, 1), 1.0)
                blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
                open_command = start_open_q + (NOMINAL_Q_OPEN - start_open_q) * blend
                data.ctrl[fixture_actuator_id] = initial_fixture_z
                feedforward, _ = grip_only_bias(model, data)
                data.ctrl[root_actuator_id] = float(np.clip(ROOT_PHASE - open_command + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
                mujoco.mj_step(model, data)
                step += 1
                mujoco.mj_forward(model, data)
                trace.append(collect_state(model, data, open_command, "RELEASE_OPEN", step, bottle_gid, bottle_body, bottle_adr,
                                           fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff))
            release_rows = []
            for _ in range(round(1.0 / float(model.opt.timestep))):
                data.ctrl[fixture_actuator_id] = initial_fixture_z
                feedforward, _ = grip_only_bias(model, data)
                data.ctrl[root_actuator_id] = float(np.clip(ROOT_PHASE - NOMINAL_Q_OPEN + feedforward / float(model.actuator_gainprm[root_actuator_id, 0]), *model.actuator_ctrlrange[root_actuator_id]))
                mujoco.mj_step(model, data)
                step += 1
                mujoco.mj_forward(model, data)
                row = collect_state(model, data, NOMINAL_Q_OPEN, "RELEASE_SETTLE", step, bottle_gid, bottle_body, bottle_adr,
                                    fixture_body, fixture_qpos, root_actuator_id, fixture_actuator_id, root_qpos, coeff)
                trace.append(row)
                release_rows.append(row)
            released = [x for x in release_rows if not x["bottle_contacts"] or all(c["bottle_contact_side"] == "table" for c in x["bottle_contacts"])]
            release_metrics = {"status": "PASS" if len(released) / max(1, len(release_rows)) >= 0.95 else "FAIL",
                               "release_no_hand_contact_fraction": len(released) / max(1, len(release_rows)),
                               "final_bottle_position_world_m": data.xpos[bottle_body].tolist(),
                               "final_bottle_linear_velocity_m_s": data.qvel[bottle_adr:bottle_adr + 3].tolist(),
                               "final_bottle_contacts": contacts_for(model, data, bottle_gid)}

    renderer.close()
    if frames:
        gif_path = run / "candidate_b_70mm_bottle_grasp.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            mp4 = run / "candidate_b_70mm_bottle_grasp.mp4"
            if not mp4.exists():
                with __import__("subprocess").Popen([
                    ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(gif_path),
                    "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(mp4)
                ], stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE) as proc:
                    _, err = proc.communicate()
                    if proc.returncode:
                        raise RuntimeError(err.decode(errors="replace")[-1200:])
        except ImportError:
            mp4 = None
    else:
        gif_path, mp4 = None, None

    summary = {
        "schema": "robotsim.issue46.candidate-b-70mm-bottle-grasp.v1",
        "status": "EXPERIMENTAL_MUJOCO_PASS" if hold_metrics.get("status") == "PASS" and lift_metrics.get("status") == "PASS" and release_metrics.get("status") == "PASS" else "EXPERIMENTAL_MUJOCO_FAIL",
        "classification": "ISOLATED_SIM_ONLY / UNVERIFIED_EXPERIMENTAL",
        "mujoco_version": mujoco.__version__, "physics_steps": step, "simulation_time_s": float(data.time),
        "input_model_sha256": sha(MODEL_SOURCE), "contact_model_sha256": sha(copy_path),
        "control_variant": control_variant,
        "root_actuator": {"name": "right_claw_joint_position", "kp": float(model.actuator_gainprm[root_actuator_id, 0]),
                          "kv": float(-model.actuator_biasprm[root_actuator_id, 2]),
                          "forcerange": model.actuator_forcerange[root_actuator_id].tolist(),
                          "ctrlrange": model.actuator_ctrlrange[root_actuator_id].tolist()},
        "fixture_actuator": {"name": "isolated_fixture_z_servo", "kp": float(model.actuator_gainprm[fixture_actuator_id, 0]),
                             "kv": float(-model.actuator_biasprm[fixture_actuator_id, 2]),
                             "forcerange": model.actuator_forcerange[fixture_actuator_id].tolist(),
                             "ctrlrange": model.actuator_ctrlrange[fixture_actuator_id].tolist(),
                             "max_abs_force_N": max((abs(x["fixture_actuator_force_N"]) for x in trace), default=0.0)},
        "bottle": preflight["bottle"],
        "fixture": preflight.get("dynamic_fixture", {"type": "ISOLATED_SIM_ONLY physical slide fixture",
                                                  "support_pad": preflight.get("support_pad"), "fixture_isolated_sim_only": True}),
        "one_root_actuator": True, "external_lift_fixture_actuator": True, "total_actuator_count": model.nu,
        "passive_actuators": 0, "passive_qpos_writes_after_initialization": 0,
        "phases": phases, "first_contact": first_contacts, "bilateral_contact_detected_qCAD": bilateral_q,
        "bilateral_hold": hold_metrics, "lift": lift_metrics, "release": release_metrics,
        "max_D_gap_mm": max((x["D_gap_max_mm"] for x in trace), default=0.0),
        "max_coupling_residual_rad": max((max(abs(v) for v in x["coupling_residual_rad"].values()) for x in trace), default=0.0),
        "max_joint_speed_rad_s": max((max(abs(v) for v in x["joint_speeds_rad_s"].values()) for x in trace), default=0.0),
        "max_contact_penetration_mm": max((x["max_all_contact_penetration_mm"] for x in trace), default=0.0),
        "max_root_actuator_force_abs": max((abs(x["root_actuator_force"]) for x in trace), default=0.0),
        "trace": str(run / "bottle_contact_trace.json"), "gif": None if gif_path is None else str(gif_path),
        "mp4": None if mp4 is None else str(mp4), "stop_reason": stop,
        "limitations": ["cylindrical envelope is not a scan of a specific commercial bottle",
                        "external slide fixture moves the gripper only and is not mounted X2 arm control",
                        "experimental quadratic coupling is not manufacturer-verified transmission",
                        "static table contact supports the initial bottle pose; lift criterion requires support separation"],
    }
    with (run / "bottle_contact_trace.json").open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, allow_nan=False)
        stream.write("\n")
    keys = ["step", "time_s", "phase", "command_qCAD", "actual_qCAD", "root_actuator_force", "D_gap_max_mm",
            "narrow_normal_force_sum_N", "wide_normal_force_sum_N", "max_object_contact_penetration_mm",
            "bottle_position_world_m", "bottle_position_relative_to_fixture_m"]
    with (run / "bottle_contact_trace.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in trace:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys})
    write_json(run, "grasp_result.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("review", "preflight", "support-scan", "support-pad-grid", "tripod-preflight", "dynamic-fixture-preflight", "fixture-preflight", "fixture-lift-scan", "physics"), required=True)
    ap.add_argument("--run-dir", type=Path, help="required existing governed preflight directory for physics")
    ap.add_argument("--support-grid-dir", type=Path, help="exact completed static support-pad grid directory for tripod-preflight")
    ap.add_argument("--bottle-y-offset-mm", type=float, default=0.0,
                    help="controlled bottle placement diagnostic relative to the prior verified candidate")
    ap.add_argument("--bottle-x-offset-mm", type=float, default=0.0,
                    help="controlled bottle placement diagnostic relative to the prior verified candidate")
    ap.add_argument("--fixture-z-offset-mm", type=float, default=0.0,
                    help="isolated gripper fixture height relative to its prior placement; bottle remains free on the visible support")
    ap.add_argument("--y-offset-mm", type=float, default=0.0,
                    help="small fixture-only lateral shift used for a static fixture preflight variant")
    ap.add_argument("--control-variant", choices=("baseline", "kp2_diagnostic", "kp4_diagnostic"), default="baseline",
                    help="diagnostic variants change only root servo kp; force cap, friction, topology and geometry remain fixed")
    ap.add_argument("--lift-slip-stop-mm", type=float, default=5.0,
                    help="lift-only diagnostic stop threshold; default 5 mm, controlled exploration may use at most 20 mm")
    args = ap.parse_args()
    sources = verify_inputs()
    if args.stage == "review":
        out = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-mechanical-contact-review", environment="wsl").native_path
        review = mechanical_review(out, sources)
        write_json(out, "source_references.json", {"physics_run": str(SOURCE_PHYSICS), "static_run": str(SOURCE_STATIC),
                   "cylinder_run": str(CYLINDER_STATIC), "model_sha256": sha(MODEL_SOURCE), "official_urdf_sha256": sha(OFFICIAL_URDF), "step_sha256": sha(STEP_SOURCE)})
        finalize_manifest(out)
        print(f"REVIEW={out}")
        print(f"NARROW_TURN_QCAD={review['quadratic_coupling']['narrow']['turn_qCAD_analytic']:.9f}")
        print(f"BASELINE_CROSSING_PAIRS={len(review['ten_baseline_original_STL_crossing_pairs'])}")
        return 0
    if args.stage == "preflight":
        out = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-70mm-bottle-preflight", environment="wsl").native_path
        review = mechanical_review(out, sources)
        preflight = make_contact_preflight(out, sources, {**review, "source_static_rows": sources["static"]["static_rows"],
                                                       "coupling_coefficients": review["quadratic_coupling_coefficients"] if "quadratic_coupling_coefficients" in review else {
                                                           "narrow": parse_poly(ET.parse(MODEL_SOURCE).getroot(), "geometry_narrow_coupling")[2],
                                                           "wide": parse_poly(ET.parse(MODEL_SOURCE).getroot(), "geometry_wide_coupling")[2]},
                                                       })
        if preflight.get("status") == "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION":
            # Save the static rows and verified coupling coefficients consumed by the physics stage.
            write_json(out, "preflight_context.json", {
                "review": review,
                "source_static_rows": sources["static"]["static_rows"],
                "coupling_coefficients": {
                    "narrow": parse_poly(ET.parse(MODEL_SOURCE).getroot(), "geometry_narrow_coupling")[2],
                    "wide": parse_poly(ET.parse(MODEL_SOURCE).getroot(), "geometry_wide_coupling")[2],
                },
            })
        finalize_manifest(out)
        print(f"PREFLIGHT={preflight['status']}")
        print(f"RUN_DIR={out}")
        if preflight.get("selected"):
            print(f"TABLE_Z={preflight['selected']['table_z_m']}")
            print(f"BOTTLE_CENTER={preflight['selected']['bottle_center_m']}")
        return 0 if preflight.get("status") == "PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION" else 2
    if args.stage == "support-scan":
        if args.run_dir is None:
            raise ValueError("support-scan requires the exact bottle preflight directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        out, scan = scan_table_fixture_through_stroke(location.native_path)
        print(f"SUPPORT_SCAN={scan['status']}")
        print(f"CLEAR_CANDIDATES={scan['clear_candidate_count']}")
        print(f"RUN_DIR={out}")
        return 0 if scan["clear_candidates"] else 2
    if args.stage == "support-pad-grid":
        if args.run_dir is None:
            raise ValueError("support-pad-grid requires the exact static bottle preflight directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        out, scan = scan_small_support_pad_grid(location.native_path, args.bottle_y_offset_mm, args.bottle_x_offset_mm)
        print(f"SUPPORT_PAD_GRID={scan['status']}")
        print(f"CLEAR_PAD_COUNT={scan['clear_pad_count']}")
        print(f"STABLE_CLEAR_TRIPOD={scan['stable_clear_tripod']}")
        print(f"RUN_DIR={out}")
        return 0 if scan["stable_clear_tripod"] else 2
    if args.stage == "tripod-preflight":
        if args.run_dir is None:
            raise ValueError("tripod-preflight requires the exact static bottle preflight directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        if args.support_grid_dir is None:
            raise ValueError("tripod-preflight requires --support-grid-dir from a completed support-pad-grid stage")
        grid_location = pg.validate_output_path(args.support_grid_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(grid_location)
        out, preflight = prepare_tripod_support_candidate(location.native_path, grid_location.native_path,
                                                          args.bottle_y_offset_mm, args.fixture_z_offset_mm,
                                                          args.bottle_x_offset_mm)
        print(f"TRIPOD_PREFLIGHT={preflight['status']}")
        print(f"STATIC_TABLE_GRIPPER_CONTACT_SAMPLES={len(preflight['static_sweep']['table_gripper_collision_samples'])}")
        print(f"LIFT_TABLE_GRIPPER_CONTACT_SAMPLES={len(preflight['fixture_lift_sweep']['table_gripper_collision_samples'])}")
        print(f"RUN_DIR={out}")
        return 0 if preflight["status"] == "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR" else 2
    if args.stage == "dynamic-fixture-preflight":
        if args.run_dir is None:
            raise ValueError("dynamic-fixture-preflight requires the exact static tripod preflight directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        out, preflight = prepare_dynamic_fixture_candidate(location.native_path)
        print(f"DYNAMIC_FIXTURE_PREFLIGHT={preflight['status']}")
        print(f"STATIC_TABLE_GRIPPER_CONTACT_SAMPLES={len(preflight['static_table_gripper_collision_samples'])}")
        print(f"LIFT_TABLE_GRIPPER_CONTACT_SAMPLES={len(preflight['fixture_lift_table_gripper_collision_samples'])}")
        print(f"RUN_DIR={out}")
        return 0 if preflight["status"] == "PASS_DYNAMIC_FIXTURE_STATIC_CLEAR" else 2
    if args.stage == "fixture-preflight":
        if args.run_dir is None:
            raise ValueError("fixture-preflight requires an exact support-scan directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        out, fixture_preflight = prepare_support_candidate(location.native_path, args.y_offset_mm)
        print(f"FIXTURE_PREFLIGHT={fixture_preflight['status']}")
        print(f"TABLE_LIMIT_QCAD={fixture_preflight['predicted_first_table_gripper_collision_on_closing']}")
        print(f"RUN_DIR={out}")
        return 0 if fixture_preflight["status"] in ("PASS_STATIC_NO_INITIAL_OBJECT_OR_FIXTURE_CONTACT", "PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION") else 2
    if args.stage == "fixture-lift-scan":
        if args.run_dir is None:
            raise ValueError("fixture-lift-scan requires the exact support-scan directory")
        location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
        pg._require_reserved_directory(location)
        out, lift_scan = scan_fixture_lift_path(location.native_path, Path(json.loads((location.native_path / "support_fixture_scan.json").read_text())["preflight_run"]))
        print(f"FIXTURE_LIFT_SCAN={lift_scan['status']}")
        print(f"RUN_DIR={out}")
        return 0 if lift_scan["status"] == "PASS_STATIC_FIXTURE_CLEAR_60MM" else 2
    if args.run_dir is None:
        raise ValueError("physics stage requires an exact static bottle preflight run directory")
    location = pg.validate_output_path(args.run_dir, require_exists=True, expected_kind="directory", environment="wsl")
    pg._require_reserved_directory(location)
    preflight_run = location.native_path
    preflight = json.loads((preflight_run / "contact_preflight.json").read_text())
    context = json.loads((preflight_run / "preflight_context.json").read_text())
    if preflight.get("status") not in ("PASS_STATIC_NO_INITIAL_GRIPPER_BOTTLE_INTERSECTION", "PASS_STATIC_NO_INITIAL_OBJECT_OR_FIXTURE_CONTACT", "PASS_STATIC_OPEN_NO_GRIPPER_INTERSECTION", "PASS_STATIC_OPEN_AND_PRECONTACT_FIXTURE_CLEAR"):
        raise ValueError("physics blocked by bottle placement preflight")
    out = pg.create_unique_output_directory(EVIDENCE_ROOT, prefix="candidate-b-70mm-bottle-physics", environment="wsl").native_path
    for filename in ("contact_preflight.json", "preflight_context.json", "mechanical_review.json"):
        src = preflight_run / filename
        if not src.is_file():
            raise FileNotFoundError(src)
        pg.write_new_bytes(out / filename, src.read_bytes(), environment="wsl")
    result = contact_run(out, preflight_run, preflight, {**context["review"], "source_static_rows": context["source_static_rows"],
                                                        "coupling_coefficients": context["coupling_coefficients"]},
                         args.control_variant, args.lift_slip_stop_mm)
    report = [
        "# Candidate B 70 mm Bottle Contact / Grasp / Lift",
        "", f"Status: {result['status']}", f"MuJoCo {result['mujoco_version']}; steps={result['physics_steps']}; sim_time={result['simulation_time_s']:.3f}s.",
        f"Bottle: vertical cylinder approximation, D={OBJECT_DIAMETER_MM:g} mm, H={OBJECT_HEIGHT_MM:g} mm, mass={OBJECT_MASS_KG:g} kg.",
        f"Fixture: ISOLATED_SIM_ONLY; one gripper root actuator={result['one_root_actuator']}; separate dynamic slide actuator={result['external_lift_fixture_actuator']}; passive qpos writes={result['passive_qpos_writes_after_initialization']}.",
        f"First contacts: narrow={result['first_contact']['narrow']}; wide={result['first_contact']['wide']}.",
        f"Bilateral hold={result['bilateral_hold']['status']}; lift={result['lift']['status']}; release={result['release']['status']}.",
        f"D gap max={result['max_D_gap_mm']:.6g}mm; coupling residual max={result['max_coupling_residual_rad']:.6g}rad; max speed={result['max_joint_speed_rad_s']:.6g}rad/s.",
        f"Peak contact penetration={result['max_contact_penetration_mm']:.6g}mm; root actuator force max={result['max_root_actuator_force_abs']:.6g}N·m; fixture actuator force max={result['fixture_actuator']['max_abs_force_N']:.6g}N.",
        "No vendor transmission claim, mounted-X2 manipulation claim, or hardware grasp claim is made.",
    ]
    pg.write_new_text(out / "REPORT.md", "\n".join(report) + "\n", environment="wsl")
    manifest = finalize_manifest(out)
    print(f"GRASP_STATUS={result['status']}")
    print(f"RUN_DIR={out}")
    print(f"VIDEO={result.get('mp4') or result.get('gif')}")
    print(f"ARTIFACTS={len(manifest['artifacts'])}")
    return 0 if result["status"] == "EXPERIMENTAL_MUJOCO_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
