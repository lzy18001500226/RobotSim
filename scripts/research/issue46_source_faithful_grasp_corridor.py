#!/usr/bin/env python3
"""Bounded source-faithful X2 OmniPicker grasp-channel geometry audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
X2_ROOT = Path("/tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d")
X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
X2_URDF = X2_ROOT / X2_REL
CANONICAL = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py")
CANONICAL_SHA = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"
PRIOR_FRAME = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/overnight-frame-collision-20261008/frame-2/source_contact_frame.json")
OUTPUT_DEFAULT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/grasp-corridor-20261008")
ABC_PATH = REPO / "scripts/research/issue46_root_cause_abc.py"
BASELINE_PATH = REPO / "scripts/research/issue46_omnipicker_1dof_m0.py"
AUDIT_PATH = REPO / "scripts/research/issue46_omnipicker_collision_audit_recovery.py"
ROOT_LINK = "right_elbow_link"
PITCH = "right_wrist_pitch_joint"
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
NARROW = "R_hand_narrow3_Link"
WIDE = "R_hand_wide3_Link"
CRITICAL_LINKS = (
    "right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link",
    "R_hand_narrow_loop_Link", "R_hand_wide_loop_Link",
)
CONTACT_TOL = 1.0e-5
COLLISION_EPS = 1.0e-7
APPROACH_M = 0.080
APPROACH_N = 21
LIFT_M = 0.030
LIFT_N = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def name(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name}_{ident}"


def body_id(model, body: str) -> int:
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if ident < 0:
        raise RuntimeError(f"Missing body {body}")
    return int(ident)


def geom_id(model, geom: str) -> int:
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if ident < 0:
        raise RuntimeError(f"Missing geom {geom}")
    return int(ident)


def collision_geom_for_body(model, body: str) -> int:
    bid = body_id(model, body)
    candidates = [g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) == bid
                  and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one active collision geom on {body}, found {len(candidates)}")
    return int(candidates[0])


def qpos_id(model, joint: str) -> int:
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if ident < 0:
        raise RuntimeError(f"Missing joint {joint}")
    return int(model.jnt_qposadr[ident])


def source_identity(abc, extracted: dict[str, Any]) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(X2_ROOT), "status", "--porcelain"], text=True).strip()
    if commit != X2_PIN or dirty:
        raise RuntimeError(f"Vendor checkout mismatch: {commit=} dirty={bool(dirty)}")
    if mujoco.__version__ != "3.3.6" or mujoco.mj_versionString() != "3.3.6":
        raise RuntimeError("Expected Python and native MuJoCo 3.3.6")
    if sha(X2_URDF) != "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d":
        raise RuntimeError("Pinned source URDF hash mismatch")
    if sha(CANONICAL) != CANONICAL_SHA:
        raise RuntimeError("Accepted canonical scene helper hash mismatch")
    root = ET.parse(X2_URDF).getroot()
    links = {row.get("name", ""): row for row in root.findall("link")}
    meshes = {}
    for link_name in extracted["included_links"]:
        for index, collision in enumerate(links[link_name].findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            path = X2_URDF.parent / "meshes" / Path(mesh.get("filename", "")).name
            meshes[path.name] = {
                "sha256": sha(path), "bytes": path.stat().st_size,
                "source_link": link_name, "collision_index": index,
                "triangle_count": len(abc.stl_triangles(path)),
                "scale": [float(v) for v in mesh.get("scale", "1 1 1").split()],
            }
    native = Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
    return {
        "upstream_repository": "https://github.com/AgibotTech/agibot_x2_urdf",
        "upstream_commit": commit, "upstream_worktree_clean": not bool(dirty),
        "license": "Mulan PSL v2", "urdf": str(X2_REL), "urdf_sha256": sha(X2_URDF),
        "collision_mesh_hashes": meshes, "extracted_subtree_urdf_sha256": extracted["sha256"],
        "prior_centered_frame_json": str(PRIOR_FRAME), "prior_centered_frame_sha256": sha(PRIOR_FRAME),
        "canonical_scene_helper_sha256": sha(CANONICAL),
        "python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(), "native_library": str(native),
        "native_library_sha256": sha(native), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "robotsim_branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "robotsim_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha(Path(__file__).resolve()),
        "baseline_controller_sha256": sha(BASELINE_PATH), "conversion_audit_sha256": sha(AUDIT_PATH),
        "source_geometry_audit_sha256": sha(ABC_PATH),
    }


def set_pose(abc, base, model, data, root_base, root_shift, pitch, aperture, bottle_lift=0.0):
    model.body_pos[body_id(model, ROOT_LINK)] = (root_base + root_shift).tolist()
    targets = base.aperture_targets(float(aperture))
    data.qpos[qpos_id(model, DRIVER)] = targets["right_claw_joint_target_rad"]
    data.qpos[qpos_id(model, FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
    data.qpos[qpos_id(model, PITCH)] = float(pitch)
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bq = int(model.jnt_qposadr[bottle_joint])
    data.qpos[bq:bq + 3] = model.qpos0[bq:bq + 3] + np.array([0.0, 0.0, bottle_lift])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def collision_sets(abc, model):
    coll = abc.all_collision_geoms(model)
    bottle_body = body_id(model, "m0_bottle")
    table_body = body_id(model, "m0_table")
    bottle = [g for g in coll if int(model.geom_bodyid[g]) == bottle_body]
    table = [g for g in coll if int(model.geom_bodyid[g]) == table_body]
    robot = [g for g in coll if int(model.geom_bodyid[g]) not in {bottle_body, table_body}
             and name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) != "world"]
    if not bottle or not table or not robot:
        raise RuntimeError("Missing compiled robot, canonical bottle, or table collision geoms")
    return robot, bottle, table


def contacts(abc, model, data):
    all_rows = abc.contact_rows(model, data, with_force=False)
    robot_bottle = [
        row for row in all_rows
        if (row["body1"] == "m0_bottle" and row["body2"] not in {"m0_table", "world"})
        or (row["body2"] == "m0_bottle" and row["body1"] not in {"m0_table", "world"})
    ]
    robot_self = [
        row for row in all_rows
        if row["body1"] not in {"m0_bottle", "m0_table", "world"}
        and row["body2"] not in {"m0_bottle", "m0_table", "world"}
    ]
    return robot_bottle, robot_self, all_rows


def collision_state(abc, model, data, robot, bottle, table):
    pair_rows = []
    for rg in robot:
        for bg in bottle:
            row = abc.geom_distance(model, data, rg, bg)
            row["bottle_part"] = name(model, mujoco.mjtObj.mjOBJ_GEOM, bg)
            pair_rows.append(row)
    valid = [row for row in pair_rows if row["query_consistent"] and math.isfinite(row["signed_distance_m"])]
    jaw = {link: min(
        (row["signed_distance_m"] for row in valid
         if row["body1"] == link and row["bottle_part"] == "bottle_body"), default=None)
        for link in (NARROW, WIDE)}
    nonjaw = [row for row in valid if row["body1"] not in {NARROW, WIDE}]
    critical = [row for row in valid if row["body1"] in CRITICAL_LINKS]
    table_rows = [abc.geom_distance(model, data, rg, tg) for rg in robot for tg in table]
    table_valid = [row for row in table_rows if row["query_consistent"]]
    robot_bottle_contacts, self_contacts, all_contacts = contacts(abc, model, data)
    return {
        "pair_rows": pair_rows,
        "invalid_distance_queries": sum(not row["query_consistent"] for row in pair_rows),
        "minimum_robot_bottle": min(valid, key=lambda row: row["signed_distance_m"]) if valid else None,
        "minimum_nonjaw_robot_bottle": min(nonjaw, key=lambda row: row["signed_distance_m"]) if nonjaw else None,
        "minimum_critical_robot_bottle": min(critical, key=lambda row: row["signed_distance_m"]) if critical else None,
        "jaw_to_body_m": jaw,
        "per_bottle_part_min_m": {
            part: min((row["signed_distance_m"] for row in valid if row["bottle_part"] == part), default=None)
            for part in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap")
        },
        "minimum_robot_table": min(table_valid, key=lambda row: row["signed_distance_m"]) if table_valid else None,
        "invalid_robot_table_queries": sum(not row["query_consistent"] for row in table_rows),
        "robot_bottle_contacts": robot_bottle_contacts,
        "robot_self_contacts": self_contacts,
        "all_contact_rows": all_contacts,
    }


def joint_limits(source_urdf: Path, model, data):
    root = ET.parse(source_urdf).getroot()
    rows = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.get("name", ""))
        if jid < 0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        rows.append({
            "joint": joint.get("name"), "source_range_rad": [lo, hi], "qpos_rad": q,
            "lower_margin_rad": q-lo, "upper_margin_rad": hi-q,
            "within_source_range": lo-1e-12 <= q <= hi+1e-12,
            "source_axis": [float(v) for v in joint.find("axis").get("xyz", "0 0 1").split()]
                if joint.find("axis") is not None else None,
        })
    return {"all_within_source_limits": all(row["within_source_range"] for row in rows), "joint_rows": rows}


def derive_candidate(abc, audit, base, subtree: Path, prior, pitch: float, pitch_limits):
    reference = prior["candidate"]
    root_base = np.asarray(reference["root_position_world_m"], dtype=float)
    root_rotation = np.asarray(reference["root_rotation_world"], dtype=float)
    seed = {"name": "source-centered-reference", "root_position_world_m": root_base.tolist(),
            "root_rotation_world": root_rotation.tolist(),
            "previous_variables": {"aperture": prior["frame"]["contact_aperture_ratio"]}}
    model, data, _ = abc.x2_modules_and_model(audit, base, subtree, seed)
    robot, bottle, table = collision_sets(abc, model)
    root_id = body_id(model, ROOT_LINK)
    root_base = np.asarray(model.body_pos[root_id], dtype=float).copy()
    narrow_body, wide_body = body_id(model, NARROW), body_id(model, WIDE)
    narrow_geom = collision_geom_for_body(model, NARROW)
    wide_geom = collision_geom_for_body(model, WIDE)
    body_geom = geom_id(model, "bottle_body")
    set_pose(abc, base, model, data, root_base, np.zeros(3), pitch,
             float(prior["frame"]["contact_aperture_ratio"]))
    jaw_axis = data.xpos[wide_body] - data.xpos[narrow_body]
    jaw_axis /= np.linalg.norm(jaw_axis)
    target_center = data.geom_xpos[body_geom].copy()
    initial_segment = np.zeros(6)
    mujoco.mj_geomDistance(model, data, narrow_geom, wide_geom, 1.0, initial_segment)
    initial_midpoint = 0.5 * (initial_segment[:3] + initial_segment[3:])
    root_shift = target_center - initial_midpoint
    solver = [{
        "stage": "initial_compiled_jaw_midpoint_centering",
        "reference_aperture": float(prior["frame"]["contact_aperture_ratio"]),
        "jaw_witness_midpoint_world_m": initial_midpoint.tolist(),
        "bottle_body_center_world_m": target_center.tolist(),
        "root_translation_correction_world_m": root_shift.tolist(),
        "correction_norm_m": float(np.linalg.norm(root_shift)),
    }]
    for outer in range(8):
        def at(ap):
            set_pose(abc, base, model, data, root_base, root_shift, pitch, ap)
            dn = abc.geom_distance(model, data, narrow_geom, body_geom)
            dw = abc.geom_distance(model, data, wide_geom, body_geom)
            seg = np.zeros(6)
            gap = float(mujoco.mj_geomDistance(model, data, narrow_geom, wide_geom, 1.0, seg))
            return dn, dw, gap, seg
        low, high = 0.0, 1.0
        dn0, dw0, _, _ = at(low)
        dn1, dw1, _, _ = at(high)
        low_sum = dn0["signed_distance_m"] + dw0["signed_distance_m"]
        high_sum = dn1["signed_distance_m"] + dw1["signed_distance_m"]
        if low_sum > 0 or high_sum < 0:
            raise RuntimeError(f"No source-limited contact bracket for pitch {pitch}: {low_sum}, {high_sum}")
        bisect = []
        for iteration in range(56):
            mid = (low + high) * 0.5
            dn, dw, gap, _ = at(mid)
            residual = dn["signed_distance_m"] + dw["signed_distance_m"]
            bisect.append({"iteration": iteration, "aperture": mid, "narrow_distance_m": dn["signed_distance_m"],
                           "wide_distance_m": dw["signed_distance_m"], "sum_residual_m": residual,
                           "jaw_gap_m": gap})
            if residual > 0:
                high = mid
            else:
                low = mid
        aperture = (low + high) * 0.5
        dn, dw, gap, segment = at(aperture)
        midpoint = 0.5 * (segment[:3] + segment[3:])
        correction = target_center - midpoint
        solver.append({
            "outer_iteration": outer, "root_translation_correction_world_m": correction.tolist(),
            "correction_norm_m": float(np.linalg.norm(correction)), "aperture": aperture,
            "narrow_distance_m": dn["signed_distance_m"], "wide_distance_m": dw["signed_distance_m"],
            "jaw_gap_m": gap, "jaw_witness_midpoint_world_m": midpoint.tolist(),
            "bottle_body_center_world_m": target_center.tolist(), "bisection": bisect,
        })
        if np.linalg.norm(correction) <= 2e-6 and abs(dn["signed_distance_m"]-dw["signed_distance_m"]) <= CONTACT_TOL:
            break
        root_shift += correction
    set_pose(abc, base, model, data, root_base, root_shift, pitch, aperture)
    targets = base.aperture_targets(aperture)
    candidate = {
        "name": "wrist_pitch_lower_limit" if pitch < 0 else "wrist_pitch_upper_limit",
        "mechanical_hypothesis": "Use an exact source wrist-pitch endpoint to rotate the open jaw channel around its source pitch axis; recenter the two compiled distal collision witnesses on the bottle-body center.",
        "root_position_world_m": (root_base + root_shift).tolist(),
        "root_rotation_world": root_rotation.tolist(),
        "wrist_pitch_target_rad": float(pitch), "wrist_pitch_source_range_rad": pitch_limits,
        "wrist_yaw_target_rad": 0.0, "wrist_roll_target_rad": 0.0,
        "contact_aperture_ratio": float(aperture),
        "active_joint_targets": {
            DRIVER: {"target_rad": float(targets["right_claw_joint_target_rad"]), "source_range_rad": [-1.0, 0.0]},
            FOLLOWER: {"target_rad": float(targets["R_hand_wide1_joint_target_rad"]), "source_range_rad": [0.0, 1.0]},
        },
        "tcp_definition": "Midpoint of the compiled narrow3/wide3 closest-point segment at simultaneous opposing contact on the 70 mm bottle-body cylinder. This is an operational source-surface-derived TCP, not a vendor frame.",
        "tcp_target_world_m": target_center.tolist(),
        "root_translation_solver": solver,
        "source_limits_unchanged": True,
    }
    return candidate, model, data


def static_summary(name_value, phase, index, aperture, progress, lift, state):
    overall = state["minimum_robot_bottle"]
    nonjaw = state["minimum_nonjaw_robot_bottle"]
    critical = state["minimum_critical_robot_bottle"]
    table = state["minimum_robot_table"]
    return {
        "candidate": name_value, "phase": phase, "sample": index, "aperture_ratio": aperture,
        "approach_progress_0_to_1": progress, "rigid_common_lift_m": lift,
        "min_robot_bottle_distance_m": overall["signed_distance_m"] if overall else None,
        "min_robot_bottle_pair": [overall["body1"], overall["geom2"]] if overall else None,
        "min_nonjaw_distance_m": nonjaw["signed_distance_m"] if nonjaw else None,
        "min_wrist_loop_distance_m": critical["signed_distance_m"] if critical else None,
        "narrow3_to_body_m": state["jaw_to_body_m"][NARROW],
        "wide3_to_body_m": state["jaw_to_body_m"][WIDE],
        "bottle_component_min_m": json.dumps(state["per_bottle_part_min_m"], separators=(",", ":")),
        "invalid_queries": state["invalid_distance_queries"],
        "robot_bottle_contact_count": len(state["robot_bottle_contacts"]),
        "robot_bottle_contact_pairs": json.dumps(state["robot_bottle_contacts"], separators=(",", ":")),
        "robot_self_contact_count": len(state["robot_self_contacts"]),
        "robot_table_min_m": table["signed_distance_m"] if table else None,
    }


def bottle_table_min(abc, model, data, bottle):
    table_id = body_id(model, "m0_table")
    table = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == table_id
             and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    rows = [abc.geom_distance(model, data, bg, tg) for bg in bottle for tg in table]
    rows = [row for row in rows if row["query_consistent"]]
    return min(rows, key=lambda row: row["signed_distance_m"]) if rows else None


def stl_meshes(abc, source_urdf, model, data):
    root = ET.parse(source_urdf).getroot()
    rows = []
    for link in root.findall("link"):
        body = link.get("name", "")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
            continue
        for index, collision in enumerate(link.findall("collision")):
            node = collision.find("geometry/mesh")
            if node is None:
                continue
            path = X2_URDF.parent / "meshes" / Path(node.get("filename", "")).name
            origin = collision.find("origin")
            rows.append({
                "body": body, "collision_index": index, "path": path, "sha256": sha(path),
                "triangles": abc.stl_triangles(path),
                "scale": np.asarray([float(v) for v in node.get("scale", "1 1 1").split()]),
                "origin_xyz": np.asarray([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]),
                "origin_rotation": Rotation.from_euler("xyz", [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]).as_matrix(),
            })
    return rows


def world_triangles(mesh, model, data):
    bid = body_id(model, mesh["body"])
    tri = mesh["triangles"] * mesh["scale"][None, None, :]
    tri = tri @ mesh["origin_rotation"].T + mesh["origin_xyz"]
    return tri @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]


def primitive_triangles(model, data, gid, n=48):
    center, rot = data.geom_xpos[gid].copy(), data.geom_xmat[gid].reshape(3, 3).copy()
    size, kind = np.asarray(model.geom_size[gid]), int(model.geom_type[gid])
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    faces = []
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        rings = []
        for z in (-size[1], size[1]):
            rings.append(np.column_stack((size[0]*np.cos(theta), size[0]*np.sin(theta), np.full(n, z))))
        for i in range(n):
            j = (i+1) % n
            faces.extend([[rings[0][i], rings[0][j], rings[1][j]],
                          [rings[0][i], rings[1][j], rings[1][i]]])
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        rings = []
        for phi in np.linspace(-math.pi/2, math.pi/2, 18):
            rings.append(np.column_stack((size[0]*math.cos(phi)*np.cos(theta),
                                          size[1]*math.cos(phi)*np.sin(theta),
                                          np.full(n, size[2]*math.sin(phi)))))
        for k in range(len(rings)-1):
            for i in range(n):
                j=(i+1)%n
                faces.extend([[rings[k][i],rings[k][j],rings[k+1][j]],
                              [rings[k][i],rings[k+1][j],rings[k+1][i]]])
    if not faces:
        return np.empty((0,3,3))
    return np.asarray(faces) @ rot.T + center


LINK_COLORS = {
    "right_wrist_yaw_link": "#d64b42", "right_wrist_pitch_link": "#e26a45",
    "right_wrist_roll_link": "#c53636", "R_hand_narrow_loop_Link": "#f39a38",
    "R_hand_wide_loop_Link": "#ed7d31", NARROW: "#23a679", WIDE: "#19a6a1",
}


def render_view(out, candidate, model, data, meshes, view, open_metrics, phase_label="OPEN"):
    h, v, depth, xlabel, ylabel, title = {
        "front": (0, 2, 1, "world x (m)", "world z (m)", "Front projection, X-Z"),
        "side": (1, 2, 0, "world y (m)", "world z (m)", "Side projection, Y-Z"),
        "top": (0, 1, 2, "world x (m)", "world y (m)", "Top projection, X-Y"),
        "closeup": (0, 2, 1, "world x (m)", "world z (m)", "Distal channel close-up, X-Z"),
    }[view]
    size=(1800,1320)
    image=Image.new("RGBA",size,"#f4f5f3")
    draw=ImageDraw.Draw(image)
    overlay=Image.new("RGBA",size,(0,0,0,0))
    mesh_draw=ImageDraw.Draw(overlay,"RGBA")
    try:
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
        title_font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",25)
    except OSError:
        font=small=title_font=ImageFont.load_default()
    bounds={
        "front":((.12,.48),(.67,1.20)),
        "side":((-.22,.20),(.67,1.20)),
        "top":((.12,.48),(-.22,.22)),
        "closeup":((.20,.40),(.75,1.08)),
    }[view]
    left,top,right,bottom=130,170,1660,1110
    def project(point):
        u=float(point[h]); w=float(point[v])
        px=left+(u-bounds[0][0])/(bounds[0][1]-bounds[0][0])*(right-left)
        py=bottom-(w-bounds[1][0])/(bounds[1][1]-bounds[1][0])*(bottom-top)
        return (int(round(px)),int(round(py)))
    def rgb(hex_color):
        text=hex_color.lstrip("#")
        return tuple(int(text[i:i+2],16) for i in (0,2,4))
    draw.rectangle((left,top,right,bottom),outline="#5e696b",width=2)
    draw.text((left,35),f"{candidate['name']} | {title}",fill="#202628",font=title_font)
    draw.text((left,78),"Original pinned collision STL triangle surfaces; compiled MuJoCo distances shown below.",
              fill="#455052",font=font)
    for value in np.arange(bounds[0][0],bounds[0][1]+1e-9,.05):
        p0=project([value,bounds[1][0],0]); p1=project([value,bounds[1][1],0])
        draw.line((p0,p1),fill=(150,160,160,75),width=1)
        draw.text((p0[0]-16,bottom+8),f"{value:.2f}",fill="#556163",font=small)
    for value in np.arange(bounds[1][0],bounds[1][1]+1e-9,.05):
        p0=project([bounds[0][0],value,0]); p1=project([bounds[0][1],value,0])
        draw.line((p0,p1),fill=(150,160,160,75),width=1)
        draw.text((left-66,p0[1]-8),f"{value:.2f}",fill="#556163",font=small)
    draw.text(((left+right)//2,bottom+42),xlabel,fill="#354043",font=font,anchor="mm")
    draw.text((30,(top+bottom)//2),ylabel,fill="#354043",font=font)
    palette = {"bottle_body":"#1781bd","bottle_shoulder":"#42a9d8","bottle_neck":"#17658b","bottle_cap":"#263e4a"}
    bottle_body = body_id(model,"m0_bottle")
    bottle_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g])==bottle_body
                  and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    for gid in bottle_ids:
        gname = name(model,mujoco.mjtObj.mjOBJ_GEOM,gid)
        tri = primitive_triangles(model,data,gid)
        fill=rgb(palette.get(gname,"#1781bd"))+(85,)
        for face in tri:
            mesh_draw.polygon([project(point) for point in face],fill=fill,outline=(22,75,102,120))
    closeup_links={"right_wrist_roll_link","R_hand_narrow_loop_Link","R_hand_wide_loop_Link",NARROW,WIDE}
    for mesh in sorted(meshes,key=lambda row: float(data.xpos[body_id(model,row["body"])][depth])):
        if view=="closeup" and mesh["body"] not in closeup_links:
            continue
        tri = world_triangles(mesh,model,data)
        centers=tri.mean(axis=1)
        margin=0.02
        visible=((centers[:,h]>=bounds[0][0]-margin)&(centers[:,h]<=bounds[0][1]+margin)&
                 (centers[:,v]>=bounds[1][0]-margin)&(centers[:,v]<=bounds[1][1]+margin))
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        camera=np.zeros(3); camera[depth]=1.0
        facing=np.einsum("ij,j->i",normals,camera)>0.0
        visible &= facing
        centers=centers[visible]
        shown=tri[visible]
        order=np.argsort(centers[:,depth])
        alpha={"right_wrist_roll_link":48,NARROW:145,WIDE:145,
               "R_hand_narrow_loop_Link":100,"R_hand_wide_loop_Link":100}.get(mesh["body"],78)
        fill=rgb(LINK_COLORS.get(mesh["body"],"#626d72"))+(alpha,)
        for face in shown[order]:
            mesh_draw.polygon([project(point) for point in face],fill=fill)
    image=Image.alpha_composite(image,overlay)
    draw=ImageDraw.Draw(image)
    corners=np.array([[.1,-.3,.8],[.5,-.3,.8],[.5,.1,.8],[.1,.1,.8],[.1,-.3,.8]])
    draw.line([project(point) for point in corners],fill="#725c45",width=4)
    if v==2:
        y=project([bounds[0][0],0.,0])[1]
        draw.line((left,y,right,y),fill="#829095",width=2)
    tcp=np.asarray(candidate["tcp_target_world_m"])
    center=project(tcp)
    star=[]
    for i in range(10):
        angle=-math.pi/2+i*math.pi/5
        radius=18 if i%2==0 else 8
        star.append((center[0]+radius*math.cos(angle),center[1]+radius*math.sin(angle)))
    draw.polygon(star,fill="#ffd22f",outline="#242424")
    frame=candidate["_frame"]
    for key,color,length in (("jaw_normal_narrow_to_wide_world","#6f35b5",.055),
                             ("insertion_axis_distal_toward_wrist_world","#ff7a00",.05)):
        endpoint=project(tcp+length*np.asarray(frame[key]))
        if math.dist(center,endpoint)>2:
            draw.line((center,endpoint),fill=color,width=5)
            dx,dy=endpoint[0]-center[0],endpoint[1]-center[1]
            angle=math.atan2(dy,dx)
            for side in (-1,1):
                a=angle+side*2.5
                draw.line((endpoint,(endpoint[0]-14*math.cos(a),endpoint[1]-14*math.sin(a))),fill=color,width=5)
    overall=open_metrics["minimum_robot_bottle"]
    jaw=open_metrics["jaw_to_body_m"]
    info=(f"{phase_label} compiled minimum: {1000*overall['signed_distance_m']:.3f} mm at "
          f"{overall['body1']} vs {overall['geom2']}; distal jaw/body {1000*jaw[NARROW]:.3f} / "
          f"{1000*jaw[WIDE]:.3f} mm")
    draw.text((left,119),info,fill="#202628",font=small)
    draw.text((left,143),"Robot: pinned source collision STL; bottle: canonical primitives; values: compiled MuJoCo geoms.",
              fill="#455052",font=small)
    if view=="closeup":
        for body,text,color in ((NARROW,"narrow distal jaw","#23a679"),(WIDE,"wide distal jaw","#19a6a1"),
                                ("R_hand_narrow_loop_Link","narrow loop","#f39a38"),
                                ("R_hand_wide_loop_Link","wide loop","#ed7d31"),
                                ("right_wrist_roll_link","wrist roll","#c53636")):
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
            if bid>=0:
                pt=project(data.xpos[bid])
                label_y=max(top+105,min(bottom-30,pt[1]-18))
                draw.line((pt,(pt[0]+115,label_y+12)),fill=color,width=2)
                draw.text((pt[0]+120,label_y),text,fill=color,font=small)
    legend=[
        ("#1781bd","body"),("#42a9d8","shoulder"),("#17658b","neck"),("#263e4a","cap"),
        ("#c53636","wrist collision STL"),("#ed7d31","loop collision STL"),("#23a679","jaw collision STL"),
        ("#6f35b5","jaw normal"),("#ff7a00","insertion toward wrist"),
    ]
    lx,ly=left,bottom+82
    for i,(color,text) in enumerate(legend):
        col=i%3; row=i//3; x=lx+col*500; y=ly+row*34
        draw.rectangle((x,y,x+20,y+18),fill=color)
        draw.text((x+28,y-1),text,fill="#354043",font=small)
    draw.text((left,1290),f"{xlabel} | {ylabel} | orthographic projection | clipped original collision STL facets",
              fill="#455052",font=small)
    image.convert("RGB").save(out/f"{candidate['name']}_{view}.png")


def state_row(path, row):
    fields=list(row[0]) if row else []
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(row)


def write_pairwise(path, rows):
    fields=["candidate","phase","sample","aperture_ratio","approach_progress_0_to_1",
            "rigid_common_lift_m","robot_body","robot_geom","bottle_part","bottle_geom",
            "signed_distance_m","witness_length_m","query_consistent"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def report(result):
    parts=[]
    for item in result["candidates"]:
        op=item["open_state"]; overall=op["minimum_robot_bottle"]; critical=op["minimum_critical"]
        gates="\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key,value in item["gate_results"].items())
        parts.append(
            f"### {item['candidate']['name']}\n\n"
            f"- Wrist pitch: {item['candidate']['wrist_pitch_target_rad']:.6f} rad; source range "
            f"{item['candidate']['wrist_pitch_source_range_rad']}.\n"
            f"- Derived target aperture: {item['candidate']['contact_aperture_ratio']:.9f}.\n"
            f"- OPEN minimum compiled robot/bottle distance: {1000*overall['signed_distance_m']:.3f} mm "
            f"at {overall['body1']} versus {overall['geom2']}.\n"
            f"- OPEN minimum wrist/loop distance: {1000*critical['signed_distance_m']:.3f} mm "
            f"at {critical['body1']} versus {critical['geom2']}.\n"
            f"- At the derived contact pose, narrow3/body and wide3/body distances are "
            f"{1000*item['closure']['narrow3_to_body_m']:.4f} mm and "
            f"{1000*item['closure']['wide3_to_body_m']:.4f} mm.\n\n{gates}"
        )
    return (
        "# Issue #46 X2 Source-Faithful Grasp Corridor\n\n"
        "## Method\n\n"
        "This bounded geometry-only investigation uses the previously measured centered-jaw source pose. "
        "The two hypotheses set right_wrist_pitch_joint to its exact lower and upper URDF limits. For each, "
        "a bisection finds the source-limited jaw aperture where both compiled distal collision surfaces "
        "oppose the 70 mm bottle-body cylinder; the compiled jaw witness midpoint is translated onto the "
        "canonical body center. No random seeds, optimizer restarts, collision decomposition, source limit "
        "changes, filter changes, or production model edits were used.\n\n"
        "The operational grasp TCP is the midpoint of the closest-point segment between the compiled "
        "R_hand_narrow3_Link and R_hand_wide3_Link collision geoms at opposing contact. It is derived for "
        "this experiment and is not a vendor-defined TCP. The insertion axis points from that midpoint "
        "toward right_wrist_roll_link.\n\n"
        "Distances use compiled MuJoCo collision geoms and include bottle body, shoulder, neck and cap. "
        "The annotated projections draw the original pinned URDF collision STL triangles after applying "
        "source mesh scale, collision origin and compiled link FK. Display sampling does not affect the "
        "distance calculations. All poses are static kinematic queries; no physics step was run.\n\n"
        "## Identity\n\n"
        f"- Upstream repository and pin: {result['runtime']['upstream_repository']} at {result['runtime']['upstream_commit']}.\n"
        f"- URDF SHA-256: {result['runtime']['urdf_sha256']}.\n"
        f"- MuJoCo Python/native: {result['runtime']['mujoco_python']} / {result['runtime']['mujoco_native']}.\n"
        f"- Native library SHA-256: {result['runtime']['native_library_sha256']}.\n"
        f"- Canonical bottle: {result['bottle']['diameter_m']*1000:.1f} mm diameter, "
        f"{result['bottle']['height_m']*1000:.1f} mm overall height, {result['bottle']['mass_kg']:.3f} kg.\n"
        "- Source license: Mulan PSL v2.\n\n"
        "## Candidate results\n\n" + "\n\n".join(parts) + "\n\n"
        "## Decision\n\n"
        f"- SOURCE-FAITHFUL STATIC GRASP: {'PASS' if result['source_faithful_static_pass'] else 'FAIL'} "
        "for the tested configurations only.\n"
        f"- PHYSICAL BILATERAL CONTACT: {result['physical_bilateral_contact']}.\n"
        f"- 30 MM PHYSICAL LIFT: {result['physical_30mm_lift']}.\n"
        f"- DESIGN APPROVAL REQUIRED: {result['design_approval_required']}.\n"
        f"- Classification: {result['classification']}.\n\n"
        f"{result['scope_limit']}\n\n"
        "This result does not establish global source-gripper infeasibility. Failure is limited to the two "
        "source pitch-limit hypotheses and their derived centered contact/approach poses. Any alternate "
        "collision representation, source geometry, bottle, or end-effector change requires maintainer "
        "approval before production use.\n"
    )


def reproduce(out):
    return (
        "# Reproduction\n\n"
        "Run in WSL2 Ubuntu 22.04 from the isolated research checkout. The output directory must not "
        "already contain files.\n\n"
        "    cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008\n"
        "    MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \\\n"
        "      scripts/research/issue46_source_faithful_grasp_corridor.py \\\n"
        f"      --output {out}\n\n"
        f"Inputs: pinned source checkout {X2_ROOT}; accepted scene helper {CANONICAL}; measured reference "
        f"{PRIOR_FRAME}. The runner checks source revision and hashes, helper hash, Python/MuJoCo version, "
        "and clean source checkout. No mj_step is called.\n"
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT_DEFAULT)
    out=parser.parse_args().output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence: {out}")
    out.mkdir(parents=True,exist_ok=True)

    abc=load_module("issue46_corridor_abc",ABC_PATH)
    audit,base=abc.modules()
    prior=json.loads(PRIOR_FRAME.read_text(encoding="utf-8"))
    source_dir=out/"source"; source_dir.mkdir()
    extracted=audit.extract_subtree(source_dir,X2_URDF,ROOT_LINK)
    runtime=source_identity(abc,extracted)
    srcroot=ET.parse(X2_URDF).getroot()
    pitch_joint=next(j for j in srcroot.findall("joint") if j.get("name")==PITCH)
    limits=[float(pitch_joint.find("limit").get("lower")),float(pitch_joint.find("limit").get("upper"))]
    if limits != [-0.523599,0.523599]:
        raise RuntimeError(f"Unexpected pinned wrist pitch limits: {limits}")

    pairwise=[]; candidates=[]; states=[]; mesh_counts={}
    for pitch in limits:
        candidate,model,data=derive_candidate(abc,audit,base,Path(extracted["path"]),prior,pitch,limits)
        robot,bottle,table=collision_sets(abc,model)
        root_base=np.asarray(candidate["root_position_world_m"])

        def take(phase,index,shift,aperture,progress=None,lift=0.0):
            set_pose(abc,base,model,data,root_base,shift,pitch,aperture,lift)
            state=collision_state(abc,model,data,robot,bottle,table)
            states.append(static_summary(candidate["name"],phase,index,aperture,progress,lift,state))
            pairwise.extend({
                "candidate":candidate["name"],"phase":phase,"sample":index,"aperture_ratio":aperture,
                "approach_progress_0_to_1":progress,"rigid_common_lift_m":lift,
                "robot_body":row["body1"],"robot_geom":row["geom1"],"bottle_part":row["bottle_part"],
                "bottle_geom":row["geom2"],"signed_distance_m":row["signed_distance_m"],
                "witness_length_m":row["witness_length_m"],"query_consistent":row["query_consistent"],
            } for row in state["pair_rows"])
            return state

        # OPEN gate and source-limit check.
        opened=take("OPEN",0,np.zeros(3),1.0)
        open_summary=states[-1]
        open_limit=joint_limits(Path(extracted["path"]),model,data)
        open_min=opened["minimum_robot_bottle"]
        open_table=opened["minimum_robot_table"]
        open_ok=(open_limit["all_within_source_limits"] and open_min is not None
                 and open_min["signed_distance_m"]>COLLISION_EPS
                 and not opened["robot_bottle_contacts"] and open_table is not None
                 and open_table["signed_distance_m"]>COLLISION_EPS
                 and not any(c["distance_m"] < -COLLISION_EPS for c in opened["robot_self_contacts"])
                 and opened["invalid_distance_queries"]==0)

        # Build contact frame at derived simultaneous contact aperture.
        aperture=float(candidate["contact_aperture_ratio"])
        contact_state=take("CONTACT_GEOMETRY",0,np.zeros(3),aperture)
        narrow_body,wide_body=body_id(model,NARROW),body_id(model,WIDE)
        wrist_body=body_id(model,"right_wrist_roll_link")
        ng,wg=collision_geom_for_body(model,NARROW),collision_geom_for_body(model,WIDE)
        _,segment=abc.closest_segment(model,data,ng,wg) if hasattr(abc,"closest_segment") else (None,None)
        if segment is None:
            segment=np.zeros(6); mujoco.mj_geomDistance(model,data,ng,wg,1.0,segment)
        tcp=.5*(segment[:3]+segment[3:])
        normal=data.xpos[wide_body]-data.xpos[narrow_body]; normal/=np.linalg.norm(normal)
        wrist_to_tcp=tcp-data.xpos[wrist_body]
        finger=wrist_to_tcp-normal*np.dot(wrist_to_tcp,normal); finger/=np.linalg.norm(finger)
        insertion=-finger
        candidate["tcp_target_world_m"]=tcp.tolist()
        candidate["_frame"]={
            "jaw_normal_narrow_to_wide_world":normal.tolist(),
            "insertion_axis_distal_toward_wrist_world":insertion.tolist(),
        }
        # Capture source-limited open approach from 80 mm outside the grasp channel.
        approach=[]
        for i,p in enumerate(np.linspace(0,1,APPROACH_N)):
            state=take("APPROACH",i,(1-p)*APPROACH_M*insertion,1.0,float(p))
            approach.append(state)
        approach_ok=all(
            s["minimum_robot_bottle"] is not None
            and s["minimum_robot_bottle"]["signed_distance_m"]>COLLISION_EPS
            and not s["robot_bottle_contacts"]
            and s["minimum_robot_table"] is not None
            and s["minimum_robot_table"]["signed_distance_m"]>COLLISION_EPS
            and s["invalid_distance_queries"]==0
            for s in approach
        )

        # Full closing path terminates at first simultaneous jaw-to-body contact.
        closure=[]
        apertures=np.linspace(1.0,aperture,101)
        for i,ap in enumerate(apertures):
            closure.append(take("CLOSURE",i,np.zeros(3),float(ap)))
        narrow_dist=[s["jaw_to_body_m"][NARROW] for s in closure]
        wide_dist=[s["jaw_to_body_m"][WIDE] for s in closure]
        monotone_tol=2e-5
        narrow_monotone=all(b<=a+monotone_tol for a,b in zip(narrow_dist,narrow_dist[1:]))
        wide_monotone=all(b<=a+monotone_tol for a,b in zip(wide_dist,wide_dist[1:]))
        nonjaw_clear=all(s["minimum_nonjaw_robot_bottle"] is not None
                         and s["minimum_nonjaw_robot_bottle"]["signed_distance_m"]>COLLISION_EPS
                         and s["invalid_distance_queries"]==0 for s in closure)
        set_pose(abc,base,model,data,root_base,np.zeros(3),pitch,aperture)
        ng=next(g for g in robot if int(model.geom_bodyid[g])==narrow_body)
        wg=next(g for g in robot if int(model.geom_bodyid[g])==wide_body)
        bg=geom_id(model,"bottle_body")
        dn=abc.geom_distance(model,data,ng,bg); dw=abc.geom_distance(model,data,wg,bg)
        sn=np.zeros(6); sw=np.zeros(6)
        mujoco.mj_geomDistance(model,data,ng,bg,1.0,sn); mujoco.mj_geomDistance(model,data,wg,bg,1.0,sw)
        witness_sides=[float(np.dot(sn[3:]-tcp,normal)),float(np.dot(sw[3:]-tcp,normal))]
        opposing=(witness_sides[0]*witness_sides[1]<0)
        intended={NARROW,WIDE}
        unexpected=[]
        for s in closure:
            for c in s["robot_bottle_contacts"]:
                other=c["body1"] if c["body2"]=="m0_bottle" else c["body2"]
                bgeom=c["geom1"] if c["body2"]=="m0_bottle" else c["geom2"]
                if other not in intended or bgeom!="bottle_body": unexpected.append(c)
        jaw_contact_ok=(abs(dn["signed_distance_m"])<=CONTACT_TOL
                        and abs(dw["signed_distance_m"])<=CONTACT_TOL and opposing)
        contact_ok=(jaw_contact_ok and not unexpected and narrow_monotone and wide_monotone and nonjaw_clear)

        # Static common-translation lift corridor; no object attachment or dynamics.
        lift=[]
        bottle_table=[]
        for i,height in enumerate(np.linspace(0,LIFT_M,LIFT_N)):
            s=take("RIGID_LIFT_CORRIDOR",i,np.array([0.,0.,height]),aperture,lift=float(height))
            lift.append(s)
            bottle_table.append(bottle_table_min(abc,model,data,bottle))
        lift_ok=all(
            s["minimum_robot_table"] is not None
            and s["minimum_robot_table"]["signed_distance_m"]>COLLISION_EPS
            and s["minimum_nonjaw_robot_bottle"] is not None
            and s["minimum_nonjaw_robot_bottle"]["signed_distance_m"]>COLLISION_EPS
            and s["invalid_distance_queries"]==0
            and (height==0 or bt is not None and bt["signed_distance_m"]>COLLISION_EPS)
            for height,s,bt in zip(np.linspace(0,LIFT_M,LIFT_N),lift,bottle_table)
        )
        # Restore exact contact pose for source-surface projection and render.
        set_pose(abc,base,model,data,root_base,np.zeros(3),pitch,aperture)
        axial=axial_profile(abc,model,data,Path(extracted["path"]),tcp,insertion)
        meshes=stl_meshes(abc,Path(extracted["path"]),model,data)
        for view in ("front","side","top"):
            set_pose(abc,base,model,data,root_base,np.zeros(3),pitch,1.0)
            render_view(out,candidate,model,data,meshes,view,opened)
        set_pose(abc,base,model,data,root_base,np.zeros(3),pitch,aperture)
        contact_visual_state=collision_state(abc,model,data,robot,bottle,table)
        render_view(out,candidate,model,data,meshes,"closeup",contact_visual_state,phase_label="CONTACT_GEOMETRY")
        mesh_counts[candidate["name"]]={m["body"]:len(m["triangles"]) for m in meshes}
        gates={
            "source_valid_joint_pose":open_limit["all_within_source_limits"],
            "OPEN all robot/bottle pairs clear and no robot contact":open_ok,
            "80mm insertion approach collision-free":approach_ok,
            "both distal jaws reach opposite sides of 70mm body":bool(jaw_contact_ok),
            "closure clear except intended jaws and monotonic":bool(contact_ok),
            "30mm static rigid lift corridor":bool(lift_ok),
        }
        candidate.pop("_frame",None)
        candidates.append({
            "candidate":candidate,"gate_results":gates,"source_faithful_static_pass":all(gates.values()),
            "open_state":{
                "minimum_robot_bottle":open_min,"minimum_critical":opened["minimum_critical_robot_bottle"],
                "minimum_robot_table":open_table,"per_bottle_component_minimum_m":opened["per_bottle_part_min_m"],
                "jaw_to_body_m":opened["jaw_to_body_m"],"robot_bottle_contacts":opened["robot_bottle_contacts"],
                "robot_self_contacts":opened["robot_self_contacts"],"summary":open_summary,
            },
            "source_joint_limits":open_limit,
            "source_frame":{
                "tcp_world_m":tcp.tolist(),"jaw_normal_narrow_to_wide_world":normal.tolist(),
                "finger_axis_wrist_to_distal_world":finger.tolist(),
                "insertion_axis_distal_toward_wrist_world":insertion.tolist(),
                "tcp_definition":candidate["tcp_definition"],
            },
            "approach":{
                "direction_world":insertion.tolist(),"start_offset_world_m":(APPROACH_M*insertion).tolist(),
                "sample_count":APPROACH_N,"collision_free":approach_ok,
            },
            "closure":{
                "sample_count":len(closure),"start_aperture":1.0,"contact_aperture":aperture,
                "opposing_jaw_geometry_pass":bool(jaw_contact_ok),
                "jaw_gap_at_contact_m":float(mujoco.mj_geomDistance(model,data,ng,wg,1.0,np.zeros(6))),
                "narrow3_to_body_m":dn["signed_distance_m"],"wide3_to_body_m":dw["signed_distance_m"],
                "bottle_witness_offsets_along_jaw_normal_m":witness_sides,
                "opposite_side_witnesses":opposing,"narrow_clearance_monotone":narrow_monotone,
                "wide_clearance_monotone":wide_monotone,"nonjaw_clear_through_closure":nonjaw_clear,
                "unexpected_contact_rows":unexpected,
                "clearance_trace":[{"aperture":float(ap),"narrow3_to_body_m":nd,"wide3_to_body_m":wd}
                                   for ap,nd,wd in zip(apertures,narrow_dist,wide_dist)],
            },
            "static_lift_corridor":{
                "height_m":LIFT_M,"samples":[
                    {"height_m":float(h),"robot_table_minimum":s["minimum_robot_table"],
                     "nonjaw_bottle_minimum":s["minimum_nonjaw_robot_bottle"],"bottle_table_minimum":bt}
                    for h,s,bt in zip(np.linspace(0,LIFT_M,LIFT_N),lift,bottle_table)],
                "geometric_clearance_pass":lift_ok,
                "interpretation":"Static common translation of free bottle and gripper geometry; no physics or attachment.",
            },
            "insertion_depth_profile":axial,"physics_steps":0,"rollout_qpos_writes":0,
        })

    static_pass=all(row["source_faithful_static_pass"] for row in candidates)
    critical_failure=all(
        row["open_state"]["minimum_critical"] is not None
        and row["open_state"]["minimum_critical"]["signed_distance_m"]<0
        for row in candidates
    )
    result={
        "task":"Issue #46 bounded source-faithful X2 OmniPicker grasp corridor investigation",
        "runtime":runtime,"source_subtree":extracted,
        "bottle":{"diameter_m":.070,"height_m":.2445,"mass_kg":.570,
                  "body_world_position_m":[.300,0.,.9175],
                  "collision_components":["bottle_body","bottle_shoulder","bottle_neck","bottle_cap"]},
        "reference":{
            "json":str(PRIOR_FRAME),"sha256":sha(PRIOR_FRAME),
            "name":prior["candidate"]["name"],
            "jaw_gap_m":prior["frame"]["jaw_gap_at_contact_aperture_m"],
            "root_position_world_m":prior["candidate"]["root_position_world_m"],
            "root_rotation_world":prior["candidate"]["root_rotation_world"],
        },
        "candidate_count":len(candidates),"candidates":candidates,
        "source_faithful_static_pass":static_pass,
        "physical_bilateral_contact":"NOT RUN" if not static_pass else "CONDITIONAL STATIC PASS",
        "physical_30mm_lift":"NOT RUN" if not static_pass else "CONDITIONAL STATIC PASS",
        "design_approval_required":not static_pass,
        "classification":"B. Genuine structural interference for the tested poses" if critical_failure else
                         "C. Insufficient geometric evidence to determine compatibility",
        "scope_limit":"Failure is bounded to two mechanically derived wrist-pitch endpoint hypotheses and their centered contact/approach poses. It does not prove global source-gripper infeasibility.",
        "physics_steps_total":0,"active_rollout_qpos_writes":0,
        "pairwise_trace":"robot_bottle_pairwise_clearance.csv",
        "state_trace":"state_clearance_summary.csv",
        "annotated_views":{
            row["candidate"]["name"]:{view:f"{row['candidate']['name']}_{view}.png"
                                      for view in ("front","side","top","closeup")}
            for row in candidates
        },
        "source_collision_triangle_count_by_link":mesh_counts,
        "collision_view_rendering":"Original pinned source collision STL facets projected with compiled link FK; bottle drawn from canonical collision primitives. Facets are viewport clipped and back-face culled for orthographic clarity.",
        "production_model_changed":False,
    }
    write_pairwise(out/"robot_bottle_pairwise_clearance.csv",pairwise)
    state_row(out/"state_clearance_summary.csv",states)
    write_json(out/"result.json",result)
    report_text=(
        "# Issue #46 X2 Source-Faithful Grasp Corridor\n\n"
        "## Method\n\n"
        "The two hypotheses set right_wrist_pitch_joint to its exact lower and upper URDF limits, "
        "then solve the jaw aperture and root translation from compiled collision witnesses. The "
        "operational TCP is the midpoint between the compiled narrow3 and wide3 closest-point witnesses "
        "at opposing contact on the bottle-body cylinder; it is not a vendor-defined TCP. The insertion "
        "axis points from the distal midpoint toward right_wrist_roll_link.\n\n"
        "All clearance queries use compiled MuJoCo collision geoms and check bottle body, shoulder, neck "
        "and cap. Annotated projections show the original pinned collision STL triangles transformed by "
        "source collision origin, mesh scale and compiled link FK. Display sampling does not affect the "
        "compiled distance calculations. This run is static geometry only: no mj_step was called.\n\n"
        f"Upstream pin: {runtime['upstream_commit']}; URDF SHA-256: {runtime['urdf_sha256']}; "
        f"MuJoCo: {runtime['mujoco_python']} / {runtime['mujoco_native']}; "
        f"native library SHA-256: {runtime['native_library_sha256']}. License: Mulan PSL v2.\n\n"
        + "\n\n".join(
            f"### {r['candidate']['name']}\n\n"
            f"- Pitch: {r['candidate']['wrist_pitch_target_rad']:.6f} rad within "
            f"{r['candidate']['wrist_pitch_source_range_rad']}.\n"
            f"- OPEN minimum: {1000*r['open_state']['minimum_robot_bottle']['signed_distance_m']:.3f} mm "
            f"at {r['open_state']['minimum_robot_bottle']['body1']} versus {r['open_state']['minimum_robot_bottle']['geom2']}.\n"
            f"- OPEN wrist/loop minimum: {1000*r['open_state']['minimum_critical']['signed_distance_m']:.3f} mm "
            f"at {r['open_state']['minimum_critical']['body1']} versus {r['open_state']['minimum_critical']['geom2']}.\n"
            f"- At target contact, distal jaw/body distances: {1000*r['closure']['narrow3_to_body_m']:.4f} / "
            f"{1000*r['closure']['wide3_to_body_m']:.4f} mm.\n"
            f"- Static verdict: {'PASS' if r['source_faithful_static_pass'] else 'FAIL'}.\n"
            + "\n".join(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in r["gate_results"].items())
            for r in candidates
        )
        + "\n\n## Outcome\n\n"
        f"SOURCE-FAITHFUL STATIC GRASP: {'PASS' if static_pass else 'FAIL'} for these tested configurations.\n\n"
        f"PHYSICAL BILATERAL CONTACT: {result['physical_bilateral_contact']}.\n\n"
        f"30 MM PHYSICAL LIFT: {result['physical_30mm_lift']}.\n\n"
        f"DESIGN APPROVAL REQUIRED: {result['design_approval_required']}.\n\n"
        f"Classification: {result['classification']}.\n\n{result['scope_limit']}\n\n"
        "No global source-gripper infeasibility is claimed. Any alternate collision model, source geometry, "
        "bottle or end-effector proposal requires maintainer approval before production application.\n"
    )
    (out/"REPORT.md").write_text(report_text,encoding="utf-8")
    reproduce_text=(
        "# Reproduction\n\nRun in WSL2 Ubuntu 22.04 from the isolated research checkout. Output must be empty.\n\n"
        "    cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008\n"
        "    MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \\\n"
        "      scripts/research/issue46_source_faithful_grasp_corridor.py \\\n"
        f"      --output {out}\n\n"
        f"Inputs: pinned X2 checkout {X2_ROOT}, canonical helper {CANONICAL}, and centered-frame evidence "
        f"{PRIOR_FRAME}. Source revision and hashes are checked. The script does not step physics.\n"
    )
    (out/"REPRODUCE.md").write_text(reproduce_text,encoding="utf-8")
    return 0


def axial_profile(abc, model, data, source_urdf, tcp, axis):
    root=ET.parse(source_urdf).getroot()
    links={row.get("name",""):row for row in root.findall("link")}
    structures={}
    for body in (*CRITICAL_LINKS,NARROW,WIDE):
        collision=links[body].find("collision")
        mesh=collision.find("geometry/mesh")
        path=X2_URDF.parent/"meshes"/Path(mesh.get("filename","")).name
        tri=abc.stl_triangles(path)
        scale=np.asarray([float(v) for v in mesh.get("scale","1 1 1").split()])
        origin=collision.find("origin")
        xyz=np.asarray([float(v) for v in (origin.get("xyz","0 0 0") if origin is not None else "0 0 0").split()])
        rpy=[float(v) for v in (origin.get("rpy","0 0 0") if origin is not None else "0 0 0").split()]
        local=(tri*scale[None,None,:])@Rotation.from_euler("xyz",rpy).as_matrix().T+xyz
        bid=body_id(model,body)
        world=local@data.xmat[bid].reshape(3,3).T+data.xpos[bid]
        projection=np.einsum("ijk,k->ij",world-tcp,axis)
        structures[body]={"mesh":path.name,"sha256":sha(path),"triangle_count":len(tri),
                          "axial_extent_from_tcp_m":[float(projection.min()),float(projection.max())]}
    bottle_extents={}
    bbody=body_id(model,"m0_bottle")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid])!=bbody: continue
        gname=name(model,mujoco.mjtObj.mjOBJ_GEOM,gid)
        center=data.geom_xpos[gid]; rot=data.geom_xmat[gid].reshape(3,3)
        size=np.asarray(model.geom_size[gid])
        theta=np.linspace(0,2*math.pi,64,endpoint=False)
        if int(model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            pts=np.vstack([np.column_stack((size[0]*np.cos(theta),size[0]*np.sin(theta),np.full(64,z)))
                           for z in (-size[1],size[1])])
        elif int(model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            pts=np.vstack([np.column_stack((size[0]*math.cos(phi)*np.cos(theta),
                                            size[1]*math.cos(phi)*np.sin(theta),
                                            np.full(64,size[2]*math.sin(phi))))
                           for phi in np.linspace(-math.pi/2,math.pi/2,24)])
        else: continue
        world=pts@rot.T+center
        values=(world-tcp)@axis
        bottle_extents[gname]=[float(values.min()),float(values.max())]
    return {
        "axis_world":axis.tolist(),
        "positive_direction":"from distal jaw TCP toward right_wrist_roll_link",
        "source_collision_surface_intervals_m":structures,
        "canonical_bottle_component_intervals_m":bottle_extents,
        "interpretation":"Axial projection intervals describe the channel/insertion depth; compiled signed distances remain the collision authority.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
