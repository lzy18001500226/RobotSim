#!/usr/bin/env python3
"""Bounded geometry-first X2 OmniPicker grasp redesign for Issue #46."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
VENDOR = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
URDF = VENDOR / "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
AUDIT = HERE.with_name("issue46_omnipicker_collision_audit_recovery.py")
BASELINE = HERE.with_name("issue46_omnipicker_1dof_m0.py")
STAGE1 = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-collision-audit-recovery/run-20261008-i")
DEFAULT_OUTPUT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-grasp-topology-redesign/run-20261008-1")
ROOT = "right_elbow_link"
DRIVER, FOLLOWER = "right_claw_joint", "R_hand_wide1_joint"
JAW_BODIES = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
WRIST_PAIR = ("right_wrist_yaw_link", "right_wrist_pitch_link")
DIST_TOL, CLEAR, CONTACT_TOL = 2e-6, 0.002, 0.002

def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

def n(model, obj, ident: int) -> str:
    return mujoco.mj_id2name(model, obj, int(ident)) or f"{obj.name}_{ident}"

def idof(model, obj, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise RuntimeError(f"Compiled model missing {name}")
    return int(value)

def qadr(model, name: str) -> int:
    return int(model.jnt_qposadr[idof(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

def coll(model, body: str | None = None) -> list[int]:
    bid = None if body is None else idof(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return [g for g in range(model.ngeom)
            if (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
            and (bid is None or int(model.geom_bodyid[g]) == bid)]

def geom_lowest_point(model, data, gid: int) -> np.ndarray:
    kind = int(model.geom_type[gid])
    center = np.asarray(data.geom_xpos[gid], float)
    R = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], float)
    direction = R.T @ np.array([0., 0., -1.])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[start:start + count], float)
        world = vertices @ R.T + center
        return world[int(np.argmin(world[:, 2]))]
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        local = size[:3] * np.where(direction >= 0., 1., -1.)
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local = size[0] * direction / max(float(np.linalg.norm(direction)), 1e-12)
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        weighted = size[:3] * direction
        denom = max(float(np.linalg.norm(weighted)), 1e-12)
        local = size[:3] * weighted / denom
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        local = size[0] * direction / max(float(np.linalg.norm(direction)), 1e-12)
        local = local + np.array([0., 0., -size[1] if direction[2] < 0 else size[1]])
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = direction[:2]
        radial = radial / max(float(np.linalg.norm(radial)), 1e-12) * size[0]
        local = np.array([radial[0], radial[1], -size[1] if direction[2] < 0 else size[1]])
    else:
        raise RuntimeError(f"Unsupported floor-clearance geometry type {kind} for {n(model, mujoco.mjtObj.mjOBJ_GEOM, gid)}")
    return center + R @ local

def dist(model, data, a: int, b: int) -> dict[str, Any]:
    # MuJoCo's general geomDistance does not return a usable witness for an
    # infinite plane. Use the exact compiled support point against a horizontal floor.
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    if int(model.geom_type[b]) == plane or int(model.geom_type[a]) == plane:
        plane_id, shape_id, reversed_pair = (b, a, False) if int(model.geom_type[b]) == plane else (a, b, True)
        plane_R = np.asarray(data.geom_xmat[plane_id], float).reshape(3, 3)
        if abs(float(plane_R[2, 2])) < 0.999:
            raise RuntimeError("Analytic plane clearance currently requires a horizontal floor")
        p = geom_lowest_point(model, data, shape_id)
        z = float(data.geom_xpos[plane_id][2])
        signed = float(p[2] - z)
        on_plane = np.array([p[0], p[1], z])
        seg = np.concatenate((on_plane, p)) if reversed_pair else np.concatenate((p, on_plane))
        return {"body1": n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[a])),
                "geom1": n(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                "body2": n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[b])),
                "geom2": n(model, mujoco.mjtObj.mjOBJ_GEOM, b),
                "signed_distance_m": signed, "witness_length_m": abs(signed),
                "query_consistent": True, "witness_m": seg.tolist(),
                "distance_method": "compiled support point against horizontal plane"}
    seg = np.zeros(6)
    d = float(mujoco.mj_geomDistance(model, data, a, b, 1.0, seg))
    length = float(np.linalg.norm(seg[3:] - seg[:3]))
    ok = math.isfinite(d) and abs(abs(d) - length) <= DIST_TOL + 1e-6 * length
    return {"body1": n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[a])),
            "geom1": n(model, mujoco.mjtObj.mjOBJ_GEOM, a),
            "body2": n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[b])),
            "geom2": n(model, mujoco.mjtObj.mjOBJ_GEOM, b),
            "signed_distance_m": d, "witness_length_m": length,
            "query_consistent": bool(ok), "witness_m": seg.tolist()}

def minimum(model, data, aa: list[int], bb: list[int]):
    rows = [dist(model, data, a, b) for a in aa for b in bb]
    good = [r for r in rows if r["query_consistent"]]
    return (min(good, key=lambda r: r["signed_distance_m"]) if good
            else {"signed_distance_m": -math.inf, "query_consistent": False}), rows

def rotation_z(a: float):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def load_limits():
    root = ET.parse(URDF).getroot()
    out = {}
    for joint in root.findall("joint"):
        lim = joint.find("limit")
        if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
            out[joint.get("name", "")] = [float(lim.get("lower")), float(lim.get("upper"))]
    return out

def self_penetrations(model, data):
    robot = {n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in coll(model)}
    robot -= {"world", "m0_table", "m0_bottle"}
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[c.geom1]))
        b2 = n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[c.geom2]))
        if b1 in robot and b2 in robot and float(c.dist) < -1e-8:
            out.append({"body1": b1, "body2": b2, "penetration_m": -float(c.dist),
                        "geom1": n(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                        "geom2": n(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage1", type=Path, default=STAGE1)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)

    audit = module("issue46_collision_audit_recovery", AUDIT)
    base = module("issue46_omnipicker_1dof_m0", BASELINE)
    vendor_head = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(VENDOR), "status", "--porcelain"], text=True).strip()
    if vendor_head != PIN or dirty:
        raise RuntimeError(f"Pinned vendor checkout mismatch: {vendor_head}, dirty={bool(dirty)}")
    prior_path = args.stage1 / "collision_model_audit.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    source_urdf = Path(prior["corrected_isolated_subtree"]["path"])
    if not prior["corrected_isolated_subtree"]["preserved_wrist_yaw_joint"] or sha(source_urdf) != prior["corrected_isolated_subtree"]["sha256"]:
        raise RuntimeError("Corrected elbow-root extraction is missing or fails its recorded hash")
    extracted = out / "isolated_omnipicker_source_elbow_root.urdf"
    extracted.write_bytes(source_urdf.read_bytes())
    model, data, flags = audit.compile_isolated(base, extracted, np.zeros(3), np.array([1., 0., 0., 0.]), True)
    mujoco.mj_forward(model, data)
    jaw_ids = [audit.geom_id(model, body) for body in JAW_BODIES]
    bottle = idof(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    bottle_geoms, table_geoms = coll(model, "m0_bottle"), coll(model, "m0_table")
    floor_geoms = [g for g in coll(model) if n(model, mujoco.mjtObj.mjOBJ_GEOM, g) == "floor"]
    robot_geoms = [g for g in coll(model) if n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) not in {"world", "m0_table", "m0_bottle"}]
    nonjaw = [g for g in robot_geoms if g not in jaw_ids]
    critical = [g for g in robot_geoms if any(t in n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])).lower() for t in ("wrist", "loop", "omnipicker_base"))]
    pair_ids = [audit.geom_id(model, link) for link in WRIST_PAIR]
    wrist = dist(model, data, *pair_ids)
    pair_contacts = [{"geom1": n(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
                      "geom2": n(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
                      "dist_m": float(data.contact[i].dist)}
                     for i in range(data.ncon)
                     if {int(data.contact[i].geom1), int(data.contact[i].geom2)} == set(pair_ids)]
    frame = audit.contact_frame(base, model, data)
    write_json(out / "jaw_aperture_and_witness_integrity.json", {
        "frame": {k: v for k, v in frame.items() if k != "jaw_gap_trace"},
        "jaw_gap_trace": frame["jaw_gap_trace"],
        "invalid_samples_rejected_from_interpolation": frame["distance_query_integrity"]["inconsistent_samples"],
        "source_filtered_wrist_pair": {"distance": wrist, "active_contact_rows": pair_contacts,
                                        "classification": "SOURCE_FILTERED_OVERLAP" if wrist["signed_distance_m"] < 0 and not pair_contacts else "UNEXPECTED"},
        "compile_flags": flags,
    })
    write_csv(out / "jaw_aperture_profile.csv", frame["jaw_gap_trace"])
    center = np.asarray(frame["bottle_body_center_world_m"], float)
    R0 = np.asarray(frame["local_frame"]["rotation_matrix_source_frame_to_horizontal_grasp"], float)
    midpoint_local = np.asarray(frame["local_frame"]["contact_midpoint_relative_to_root_m"], float)
    normal_local = np.asarray(frame["local_frame"]["normal_narrow_to_wide"], float)
    finger_local = np.asarray(frame["local_frame"]["finger_axis_tip_to_base_from_compiled_link_chain"], float)
    side_local = np.cross(normal_local, finger_local)
    side_local /= np.linalg.norm(side_local)
    body_half = float(frame["bottle_body_half_height_m"])
    contact_ratio = float(frame["contact_aperture_ratio"])
    limits = load_limits()

    # Three meaningful contact topologies: lower distal band, center reference, and
    # the azimuth with greatest measured critical-structure clearance.
    yaw_map = []
    for deg in range(0, 360, 15):
        R = rotation_z(math.radians(deg)) @ R0
        pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                "root_rotation_world": R, "root_position_world_m": center - R @ midpoint_local}
        audit.set_static_state(base, model, data, pose, contact_ratio)
        crit, cr = minimum(model, data, critical, bottle_geoms)
        jaws = [dist(model, data, j, bottle) for j in jaw_ids]
        radials = []
        for row in jaws:
            p = np.asarray(row["witness_m"][3:], float) - data.geom_xpos[bottle]
            p[2] = 0.
            radials.append((p / max(np.linalg.norm(p), 1e-12)).tolist() if row["query_consistent"] else None)
        dot = float(np.dot(*radials)) if all(v is not None for v in radials) else None
        yaw_map.append({"yaw_deg": deg, "critical_clearance_m": crit["signed_distance_m"],
                        "critical_body": crit.get("body1"), "jaw_distances_m": [r["signed_distance_m"] for r in jaws],
                        "radial_opposition_dot": dot,
                        "queries_valid": all(r["query_consistent"] for r in cr + jaws)})
    write_csv(out / "wrist_loop_azimuth_clearance_map.csv", yaw_map)
    eligible = [r for r in yaw_map if r["queries_valid"] and r["radial_opposition_dot"] is not None and r["radial_opposition_dot"] <= -0.85]
    selected_yaw_deg = max(eligible, key=lambda r: r["critical_clearance_m"])["yaw_deg"] if eligible else 0

    # The third topology tilts the proximal wrist away from the bottle using the
    # observed collision witness direction. The jaw midpoint remains at the target.
    center_pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                   "root_rotation_world": R0, "root_position_world_m": center - R0 @ midpoint_local}
    audit.set_static_state(base, model, data, center_pose, contact_ratio)
    center_critical, center_critical_rows = minimum(model, data, critical, bottle_geoms)
    if center_critical.get("query_consistent") and math.isfinite(center_critical["signed_distance_m"]):
        wrist_point = np.asarray(center_critical["witness_m"][:3], float)
        lever = wrist_point - center
        radial = lever.copy()
        radial[2] = 0.
        radial /= max(float(np.linalg.norm(radial)), 1e-12)
        desired = radial * 0.050
        desired -= lever * float(np.dot(desired, lever)) / max(float(np.dot(lever, lever)), 1e-12)
        rotvec = np.cross(lever, desired) / max(float(np.dot(lever, lever)), 1e-12)
        raw_angle = float(np.linalg.norm(rotvec))
        if raw_angle > math.radians(12):
            rotvec *= math.radians(12) / raw_angle
        relief_rotation = Rotation.from_rotvec(rotvec).as_matrix()
    else:
        wrist_point = np.zeros(3)
        radial = np.array([1., 0., 0.])
        desired = np.zeros(3)
        rotvec = np.zeros(3)
        raw_angle = 0.0
        relief_rotation = np.eye(3)
    write_json(out / "wrist_relief_tilt_derivation.json", {
        "source": "compiled right-elbow-root model at the source-frame center pinch",
        "critical_pair": center_critical, "collision_witness_rows": center_critical_rows,
        "wrist_witness_world_m": wrist_point.tolist(), "bottle_center_world_m": center.tolist(),
        "outward_radial_direction": radial.tolist(), "desired_proximal_displacement_m": desired.tolist(),
        "unclipped_rotation_vector_rad": rotvec.tolist(), "unclipped_angle_rad": raw_angle,
        "applied_orientation_delta": relief_rotation.tolist(),
        "jaw_contact_midpoint_remains_fixed": True,
        "purpose": "move the proximal wrist structure outward from the bottle while retaining the compiled opposing-jaw contact point",
    })
    topology = [
        {"candidate": "distal_lower_body", "axial_seed_m": -0.035,
         "yaw_seed_deg": 0, "orientation_seed": R0.tolist(),
         "reason": "lower usable cylinder band to increase separation from the bottle shoulder and wrist-side structure while staying above the table"},
        {"candidate": "central_body_reference", "axial_seed_m": 0.0,
         "yaw_seed_deg": 0, "orientation_seed": R0.tolist(),
         "reason": "center of the actual cylindrical body at the source-frame opposing-jaw contact"},
        {"candidate": "witness_derived_wrist_relief_tilt", "axial_seed_m": 0.0,
         "yaw_seed_deg": 0, "orientation_seed": (relief_rotation @ R0).tolist(),
         "reason": "tilt the proximal structure outward based on the measured wrist/bottle collision witness while preserving the jaw midpoint"},
    ]
    write_json(out / "topology_definitions.json", {
        "candidate_budget": 3, "candidates": topology, "azimuth_map": yaw_map,
        "selected_azimuth_deg_for_information_only": selected_yaw_deg,
        "selection_rule": "The first two candidates test distinct measured bottle axial contact bands. The third rotation is analytically derived from the observed wrist/bottle collision witness; it is not an arbitrary orientation seed.",
    })

    # A bounded map of real compiled collision clearance by grasp height and insertion depth.
    map_rows = []
    for axial in np.linspace(-0.065, 0.065, 27):
        for depth in np.linspace(-0.04, 0.04, 17):
            R = R0
            f, nx, s = R @ finger_local, R @ normal_local, R @ side_local
            offset = f * float(axial) + nx * float(depth)
            pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                    "root_rotation_world": R, "root_position_world_m": center + offset - R @ midpoint_local}
            audit.set_static_state(base, model, data, pose, 1.0)
            opened, _ = minimum(model, data, robot_geoms, bottle_geoms)
            audit.set_static_state(base, model, data, pose, contact_ratio)
            crit, _ = minimum(model, data, critical, bottle_geoms)
            jaws = [dist(model, data, j, bottle) for j in jaw_ids]
            nr = []
            for r in jaws:
                v = np.asarray(r["witness_m"][3:], float) - data.geom_xpos[bottle]
                v[2] = 0
                nr.append((v / max(np.linalg.norm(v), 1e-12)).tolist() if r["query_consistent"] else None)
            opposition = float(np.dot(*nr)) if all(v is not None for v in nr) else None
            map_rows.append({"jaw_axial_offset_m": float(axial), "insertion_depth_m": float(depth),
                             "open_robot_bottle_clearance_m": opened["signed_distance_m"],
                             "closed_critical_clearance_m": crit["signed_distance_m"],
                             "jaw_distances_m": [r["signed_distance_m"] for r in jaws],
                             "radial_opposition_dot": opposition,
                             "witnesses_valid": all(r["query_consistent"] for r in jaws)})
    write_csv(out / "grasp_depth_height_clearance_map.csv", map_rows)

    candidates = []
    for spec in topology:
        yaw0 = math.radians(spec["yaw_seed_deg"])
        orientation_seed = np.asarray(spec["orientation_seed"], float)
        x0 = np.array([0., 0., 0., yaw0, contact_ratio])
        lo = np.array([-0.012, -0.010, -0.008, yaw0 - math.radians(8), max(0., contact_ratio - 0.20)])
        hi = np.array([0.012, 0.010, 0.008, yaw0 + math.radians(8), min(1., contact_ratio + 0.20)])
        trace_path = out / f"{spec['candidate']}_optimizer_trace.jsonl"
        trace_path.write_text("", encoding="utf-8")
        count, started = 0, time.monotonic()

        def evaluate(x):
            nonlocal count
            count += 1
            radial, lateral, axial, yaw, aperture = map(float, x)
            R = rotation_z(yaw) @ orientation_seed
            nw, fw, sw = R @ normal_local, R @ finger_local, R @ side_local
            offset = nw * radial + sw * lateral + fw * (spec["axial_seed_m"] + axial)
            pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                    "root_rotation_world": R, "root_position_world_m": center + offset - R @ midpoint_local}
            audit.set_static_state(base, model, data, pose, 1.)
            open_obj, open_obj_rows = minimum(model, data, robot_geoms, bottle_geoms)
            open_tab, open_tab_rows = minimum(model, data, robot_geoms, table_geoms)
            open_floor, open_floor_rows = minimum(model, data, robot_geoms, floor_geoms)
            open_self = self_penetrations(model, data)
            approach = []
            for t in np.linspace(1., 0., 8):
                p = dict(pose)
                p["root_position_world_m"] = np.asarray(pose["root_position_world_m"]) + fw * (0.035 * float(t))
                audit.set_static_state(base, model, data, p, 1.)
                r, rr = minimum(model, data, robot_geoms, bottle_geoms)
                approach.append({"t": float(t), "min": r, "invalid": [v for v in rr if not v["query_consistent"]]})
            audit.set_static_state(base, model, data, pose, aperture)
            jaws = [dist(model, data, j, bottle) for j in jaw_ids]
            normals = []
            for r in jaws:
                v = np.asarray(r["witness_m"][3:], float) - data.geom_xpos[bottle]
                v[2] = 0.
                normals.append((v / max(np.linalg.norm(v), 1e-12)).tolist() if r["query_consistent"] else None)
            opposition = float(np.dot(*normals)) if all(v is not None for v in normals) else None
            nonjaw_min, nonjaw_rows = minimum(model, data, nonjaw, bottle_geoms)
            critical_min, critical_rows = minimum(model, data, critical, bottle_geoms)
            close_tab, close_tab_rows = minimum(model, data, robot_geoms, table_geoms)
            close_floor, close_floor_rows = minimum(model, data, robot_geoms, floor_geoms)
            close_self = self_penetrations(model, data)
            heights = [float(r["witness_m"][5] - center[2]) for r in jaws if r["query_consistent"]]
            closure = []
            for a in np.linspace(1., aperture, 9):
                audit.set_static_state(base, model, data, pose, float(a))
                cr = [dist(model, data, j, bottle) for j in jaw_ids]
                closure.append({"aperture": float(a), "distances_m": [r["signed_distance_m"] for r in cr],
                                "queries_valid": all(r["query_consistent"] for r in cr)})
            monotonic = all(closure[i + 1]["distances_m"][j] - closure[i]["distances_m"][j] <= 1e-5
                            for j in range(2) for i in range(8))
            audit.set_static_state(base, model, data, pose, aperture)
            limits_rows = []
            targets = base.aperture_targets(aperture)
            for joint, key in ((DRIVER, "right_claw_joint_target_rad"), (FOLLOWER, "R_hand_wide1_joint_target_rad")):
                q = float(targets[key]); lim = limits[joint]
                limits_rows.append({"joint": joint, "target_rad": q, "range_rad": lim,
                                    "inside": lim[0] <= q <= lim[1]})
            lift = []
            for dz in (0.001, 0.005, 0.030):
                p = dict(pose); p["root_position_world_m"] = np.asarray(pose["root_position_world_m"]) + [0., 0., dz]
                audit.set_static_state(base, model, data, p, aperture, dz)
                rb, rbr = minimum(model, data, robot_geoms, bottle_geoms)
                rt, rtr = minimum(model, data, robot_geoms, table_geoms)
                rf, rfr = minimum(model, data, robot_geoms, floor_geoms)
                bt, btr = minimum(model, data, bottle_geoms, table_geoms)
                lift.append({"dz_m": dz, "robot_bottle": rb, "robot_table": rt, "robot_floor": rf,
                             "bottle_table": bt, "invalid": [r for r in rbr + rtr + rfr + btr if not r["query_consistent"]]})
            pert = []
            for label, delta in (("normal_plus", nw*.002), ("normal_minus", -nw*.002),
                                 ("side_plus", sw*.002), ("side_minus", -sw*.002)):
                p = dict(pose); p["root_position_world_m"] = np.asarray(pose["root_position_world_m"]) + delta
                audit.set_static_state(base, model, data, p, 1.)
                po, porows = minimum(model, data, robot_geoms, bottle_geoms)
                audit.set_static_state(base, model, data, p, aperture)
                pj = [dist(model, data, j, bottle) for j in jaw_ids]
                pert.append({"name": label, "open_clearance_m": po["signed_distance_m"],
                             "open_queries_valid": all(r["query_consistent"] for r in porows),
                             "closed_jaw_distance_m": [r["signed_distance_m"] for r in pj],
                             "closed_queries_valid": all(r["query_consistent"] for r in pj)})
            all_rows = open_obj_rows + open_tab_rows + open_floor_rows + nonjaw_rows + critical_rows + close_tab_rows + close_floor_rows
            invalid = sum(not r["query_consistent"] for r in all_rows)
            approach_clear = min(r["min"]["signed_distance_m"] for r in approach)
            gates = {
                "source_limits": all(r["inside"] for r in limits_rows),
                "query_witness_integrity": invalid == 0 and all(r["query_consistent"] for r in jaws),
                "zero_initial_self_penetration": not open_self,
                "open_bottle_clearance": open_obj["signed_distance_m"] >= CLEAR,
                "open_table_clearance": open_tab["signed_distance_m"] >= CLEAR,
                "open_floor_clearance": open_floor["signed_distance_m"] >= CLEAR,
                "approach_corridor_clearance": approach_clear >= CLEAR,
                "two_jaw_opposing_contact": all(r["query_consistent"] and abs(r["signed_distance_m"]) <= CONTACT_TOL for r in jaws)
                    and opposition is not None and opposition <= -0.85 and len(heights) == 2
                    and all(abs(h) <= body_half - .008 for h in heights),
                "closed_nonjaw_wrist_loop_clearance": nonjaw_min["signed_distance_m"] >= CLEAR,
                "closed_table_floor_clearance": close_tab["signed_distance_m"] >= CLEAR and close_floor["signed_distance_m"] >= CLEAR,
                "monotonic_closure": monotonic,
                "30mm_static_lift_corridor": all(r["robot_table"]["signed_distance_m"] >= CLEAR
                    and r["robot_floor"]["signed_distance_m"] >= CLEAR
                    and r["robot_bottle"]["signed_distance_m"] >= -CONTACT_TOL
                    and r["bottle_table"]["signed_distance_m"] >= .020 for r in lift),
                "pose_perturbation_margin": all(r["open_queries_valid"] and r["closed_queries_valid"]
                    and r["open_clearance_m"] >= CLEAR and max(abs(x) for x in r["closed_jaw_distance_m"]) <= .004 for r in pert),
            }
            row = {"candidate": spec["candidate"], "evaluation": count,
                   "variables": {"radial_offset_m": radial, "lateral_offset_m": lateral,
                                 "axial_adjust_m": axial, "yaw_rad": yaw, "aperture": aperture},
                   "root_position_world_m": np.asarray(pose["root_position_world_m"]).tolist(),
                   "root_rotation_world": R.tolist(),
                   "open": {"robot_bottle": open_obj, "robot_table": open_tab, "robot_floor": open_floor,
                            "self_penetrations": open_self},
                   "approach_samples": approach,
                   "closed": {"jaw_distance_rows": jaws, "jaw_radial_normals": normals,
                              "normal_opposition_dot": opposition, "bottle_contact_height_m": heights,
                              "minimum_nonjaw_robot_bottle": nonjaw_min,
                              "minimum_critical_robot_bottle": critical_min,
                              "robot_table": close_tab, "robot_floor": close_floor,
                              "self_penetrations": close_self},
                   "closure": closure, "lift_corridor": lift, "perturbations": pert,
                   "source_targets": limits_rows, "invalid_distance_query_count": invalid,
                   "invalid_distance_queries": [r for r in all_rows if not r["query_consistent"]]
                       + [r for r in jaws if not r["query_consistent"]],
                   "static_gates": gates, "static_pass": all(gates.values())}
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush()
            return row

        def residual(x):
            row = evaluate(x)
            jaw = row["closed"]["jaw_distance_rows"]
            opposition = row["closed"]["normal_opposition_dot"]
            g = row["static_gates"]
            return np.array([
                jaw[0]["signed_distance_m"] / .004, jaw[1]["signed_distance_m"] / .004,
                max(0., (opposition if opposition is not None else 1.) + .85) / .25,
                max(0., CLEAR - row["open"]["robot_bottle"]["signed_distance_m"]) / .004,
                max(0., CLEAR - row["closed"]["minimum_nonjaw_robot_bottle"]["signed_distance_m"]) / .004,
                max(0., CLEAR - min(r["min"]["signed_distance_m"] for r in row["approach_samples"])) / .004,
                5. if not g["query_witness_integrity"] else 0.,
            ])
        lower = np.array([-.012, -.010, -.008, yaw0-math.radians(8), max(0., contact_ratio-.20)])
        upper = np.array([.012, .010, .008, yaw0+math.radians(8), min(1., contact_ratio+.20)])
        start = time.monotonic()
        sol = least_squares(residual, np.array([0., 0., 0., yaw0, contact_ratio]),
                            bounds=(lower, upper), max_nfev=12, diff_step=2e-3,
                            xtol=2e-4, ftol=2e-4, gtol=2e-4)
        final = evaluate(sol.x)
        final["solver"] = {"method": "bounded deterministic least-squares using measured jaw distances/normals, approach, nonjaw clearance, source limits and lift gates",
                           "status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev),
                           "njev": int(sol.njev or 0), "objective_evaluations": count,
                           "elapsed_s": time.monotonic()-start, "solution": sol.x.tolist(),
                           "lower_bounds": lower.tolist(), "upper_bounds": upper.tolist()}
        candidates.append(final)
        pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                "root_rotation_world": np.asarray(final["root_rotation_world"]),
                "root_position_world_m": np.asarray(final["root_position_world_m"])}
        audit.set_static_state(base, model, data, pose, 1.)
        audit.render_reference(model, data, out, f"{spec['candidate']}_open.png", center.tolist(), .9, 135., -20.)
        audit.set_static_state(base, model, data, pose, float(final["variables"]["aperture"]))
        audit.render_reference(model, data, out, f"{spec['candidate']}_closed_static.png", center.tolist(), .9, 135., -20.)

    write_json(out / "candidate_results.json", {"candidate_count": len(candidates), "results": candidates,
                                                "passing": [r["candidate"] for r in candidates if r["static_pass"]]})
    # Stage 6 bounded diagnostics only if all canonical candidates fail.
    diagnostics = None
    passing = [r for r in candidates if r["static_pass"]]
    if not passing:
        best = min(candidates, key=lambda r: sum(not v for v in r["static_gates"].values()))
        pose = {"root_body_id": idof(model, mujoco.mjtObj.mjOBJ_BODY, ROOT),
                "root_rotation_world": np.asarray(best["root_rotation_world"]),
                "root_position_world_m": np.asarray(best["root_position_world_m"])}
        aperture = float(best["variables"]["aperture"])
        original_radius = float(model.geom_size[bottle][0])
        diameters = []
        for d in (.050, .060, .070, .080, .090):
            model.geom_size[bottle][0] = d/2
            audit.set_static_state(base, model, data, pose, aperture)
            c, cr = minimum(model, data, critical, [bottle])
            jr = [dist(model, data, j, bottle) for j in jaw_ids]
            diameters.append({"diameter_m": d, "critical_clearance_m": c["signed_distance_m"],
                              "critical_body": c.get("body1"), "jaw_distances_m": [r["signed_distance_m"] for r in jr],
                              "queries_valid": all(r["query_consistent"] for r in cr+jr),
                              "diagnostic_only": True})
        model.geom_size[bottle][0] = original_radius
        depth = []
        R = np.asarray(best["root_rotation_world"]); f = R @ finger_local
        for x in np.linspace(-.08, .08, 33):
            p = dict(pose); p["root_position_world_m"] = np.asarray(pose["root_position_world_m"]) + f*float(x)
            audit.set_static_state(base, model, data, p, aperture)
            c, cr = minimum(model, data, critical, bottle_geoms)
            jr = [dist(model, data, j, bottle) for j in jaw_ids]
            depth.append({"finger_axis_offset_m": float(x), "critical_clearance_m": c["signed_distance_m"],
                          "critical_body": c.get("body1"), "jaw_distances_m": [r["signed_distance_m"] for r in jr],
                          "queries_valid": all(r["query_consistent"] for r in cr+jr)})
        write_csv(out / "diagnostic_diameter_map.csv", diameters)
        write_csv(out / "diagnostic_grasp_depth_obstruction.csv", depth)
        # Compare the source STL surface to the compiled convex hull at the best tested pose.
        audit.set_static_state(base, model, data, pose, aperture)
        source_root = ET.parse(URDF).getroot()
        source_links = {link.get("name", ""): link for link in source_root.findall("link")}
        bottle_center_now = np.asarray(data.geom_xpos[bottle], float).copy()
        bottle_R = np.asarray(data.geom_xmat[bottle], float).reshape(3, 3).copy()
        cyl_r, cyl_half = float(model.geom_size[bottle][0]), float(model.geom_size[bottle][1])
        from issue46_collision_audit_recovery import parse_binary_stl
        hull_rows = []
        body_names = sorted({
            n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in critical
        })
        for body in body_names:
            link = source_links.get(body)
            collision = None if link is None else link.find("collision")
            mesh = None if collision is None else collision.find("geometry/mesh")
            if mesh is None:
                hull_rows.append({"body": body, "status": "NO_SOURCE_COLLISION_MESH"})
                continue
            mesh_path = (URDF.parent / mesh.get("filename", "")).resolve()
            try:
                triangles = parse_binary_stl(mesh_path)
                scale = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()], float)
                gids = [g for g in critical
                        if n(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == body]
                gid = gids[0]
                tri = triangles * scale[None, None, :]
                samples = np.concatenate((
                    tri.reshape(-1, 3), tri.mean(axis=1),
                    0.5 * (tri[:, 0] + tri[:, 1]),
                    0.5 * (tri[:, 1] + tri[:, 2]),
                    0.5 * (tri[:, 2] + tri[:, 0]),
                ), axis=0)
                geom_R = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
                world = samples @ geom_R.T + np.asarray(data.geom_xpos[gid], float)
                local = (world - bottle_center_now) @ bottle_R
                radial = np.linalg.norm(local[:, :2], axis=1)
                q0, q1 = radial - cyl_r, np.abs(local[:, 2]) - cyl_half
                outside = np.linalg.norm(np.column_stack((np.maximum(q0, 0.), np.maximum(q1, 0.))), axis=1)
                sdf = outside + np.minimum(np.maximum(q0, q1), 0.)
                hull = dist(model, data, gid, bottle)
                hull_rows.append({
                    "body": body, "status": "SAMPLED",
                    "source_collision_mesh": str(mesh_path), "source_mesh_sha256": sha(mesh_path),
                    "source_triangle_count": int(len(triangles)),
                    "sample_count": int(len(samples)),
                    "sampled_source_stl_minimum_cylinder_sdf_m": float(np.min(sdf)),
                    "sampled_source_surface_overlaps_cylinder": bool(float(np.min(sdf)) < 0.),
                    "compiled_convex_hull_signed_distance_m": hull["signed_distance_m"],
                    "compiled_hull_distance_witness_consistent": hull["query_consistent"],
                    "interpretation": "sampled source surface overlaps" if float(np.min(sdf)) < 0.
                        else "sampled source surface clears; hull over-approximation is plausible",
                })
            except Exception as exc:
                hull_rows.append({"body": body, "status": "UNAVAILABLE",
                                  "source_collision_mesh": str(mesh_path), "error": str(exc)})
        write_json(out / "source_mesh_vs_compiled_hull_diagnostic.json", {
            "candidate": best["candidate"], "bottle_diameter_m": 0.070,
            "method": "Source STL vertices, face centroids, and edge midpoints transformed by URDF scale and compiled geom pose; sampled against finite bottle-body cylinder signed-distance function.",
            "resolution_limit": "intersections between sample points can be missed; diagnostic only, not acceptance collision geometry",
            "rows": hull_rows,
        })
        diagnostics = {"changed_diameter_is_diagnostic_only": True, "source_model_restored": True,
                       "diameter_map": diameters, "depth_obstruction_map": depth,
                       "source_stl_vs_compiled_hull": hull_rows}

    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    native = Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
    result = {
        "source_collision_model_correctness": "PASS" if wrist["signed_distance_m"] < 0 and not pair_contacts else "FAIL",
        "canonical_70mm_static_feasibility": "PASS" if passing else "FAIL",
        "actual_bilateral_physical_grasp": "NOT RUN" if not passing else "NOT RUN YET",
        "measured_load_transfer": "NOT RUN", "actual_30mm_bottle_lift": "NOT RUN",
        "full_x2_right_arm_integration": "NOT RUN",
        "wrist_pair_audit": {"signed_distance_m": wrist["signed_distance_m"],
                             "active_contacts": pair_contacts, "classification": "SOURCE_FILTERED_OVERLAP" if wrist["signed_distance_m"] < 0 and not pair_contacts else "UNEXPECTED",
                             "root": ROOT, "right_wrist_yaw_joint_preserved": True},
        "candidates": [{"name": r["candidate"], "pass": r["static_pass"],
                        "failed_gates": [k for k, v in r["static_gates"].items() if not v],
                        "closed_wrist_or_loop_clearance_m": r["closed"]["minimum_critical_robot_bottle"]["signed_distance_m"],
                        "jaw_distances_m": [x["signed_distance_m"] for x in r["closed"]["jaw_distance_rows"]],
                        "jaw_radial_normal_dot": r["closed"]["normal_opposition_dot"]} for r in candidates],
        "diagnostics": diagnostics,
        "identity": {"branch": branch, "starting_head": head, "script_sha256": sha(HERE),
                     "vendor_commit": vendor_head, "vendor_urdf_sha256": sha(URDF),
                     "vendor_worktree_clean": not bool(dirty), "mujoco_python": mujoco.__version__,
                     "mujoco_native": mujoco.mj_versionString(), "python": sys.version,
                     "python_executable": sys.executable, "platform": platform.platform(),
                     "mujoco_gl": os.environ.get("MUJOCO_GL"), "native_library": str(native),
                     "native_library_sha256": sha(native), "accepted_controller_sha256": sha(BASELINE),
                     "physical_parameters_changed": False, "source_limits_or_meshes_changed": False},
    }
    write_json(out / "result.json", result)
    (out / "experiment_commands.txt").write_text(
        f"Branch: {branch}\nStarting HEAD: {head}\nVendor SHA: {vendor_head}\n"
        f"Exact command:\nMUJOCO_GL=egl {sys.executable} {HERE} --stage1 {args.stage1} --output {out}\n",
        encoding="utf-8")
    (out / "REPORT.md").write_text(
        "# Issue #46 X2 OmniPicker Grasp Topology Redesign\n\n"
        f"- Source collision-model correctness: {result['source_collision_model_correctness']}\n"
        f"- Canonical 70 mm static feasibility: {result['canonical_70mm_static_feasibility']}\n"
        "- Physical grasp: NOT RUN unless a static candidate passes.\n"
        "- Load transfer and 30 mm lift: NOT RUN.\n"
        "- Full X2 arm integration: NOT RUN.\n\n"
        "The corrected right-elbow-root URDF and accepted physical parameters were used. "
        "Three evidence-derived grasp topologies were evaluated. Invalid signed-distance "
        "witnesses were rejected from interpolation and optimizer inputs. See raw JSONL and "
        "CSV files for the exact measurements.\n",
        encoding="utf-8")
    print(json.dumps({"static": result["canonical_70mm_static_feasibility"],
                      "candidates": result["candidates"], "output": str(out)}, indent=2, sort_keys=True))
    return 0 if passing else 3

if __name__ == "__main__":
    raise SystemExit(main())
