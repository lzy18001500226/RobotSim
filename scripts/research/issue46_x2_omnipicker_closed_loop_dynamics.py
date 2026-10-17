#!/usr/bin/env python3
"""Compare the pinned X2 OmniPicker tree model with a source-geometry loop hypothesis.

The loop model is diagnostic only. Fixed URDF joints are promoted to passive
hinges at their source transforms, with axes supported by cylindrical mesh
features. It has no vendor passive limits, spring data, or internal coupling.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares


SOURCE_ROOT = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf")
SOURCE_URDF = SOURCE_ROOT / "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
SOURCE_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
SOURCE_MESH_DIR = SOURCE_ROOT / "X2_URDF-v1.4.0/meshes"
TEMPLATE = Path(
    "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/"
    "omnipicker-linkage-physics-20261014-df89d796-4673-445f-b003-d65f6dc689e5/"
    "constrained_candidate.xml"
)
TEMPLATE_SHA256 = "7157e89eaf030f1f987fc4f7b723b3fee1e7e5bbc0c541d326121e1e0cc0f35a"
MESH_AUDIT = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-variant-b-full-x2-geometry-feasibility-20261012/"
    "sourcefaithful/linkage_audit/linkage_audit.json"
)
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
MENAGERIE_XML_SHA256 = "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de"
MENAGERIE_XML_URL = (
    "https://github.com/google-deepmind/mujoco_menagerie/blob/"
    f"{MENAGERIE_COMMIT}/robotiq_2f85/2f85.xml"
)
DEFAULT_OUT = Path(
    "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/"
    "x2-omnipicker-closed-loop-dynamics-20261017-run01"
)

PREFIX = "x2cand_candidate_"
ROOT_DRIVER = PREFIX + "right_claw_joint"
ROOT_FOLLOWER = PREFIX + "R_hand_wide1_joint"
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
LOOP_JOINTS = {PASSIVE["narrow"][2], PASSIVE["wide"][2]}
PIN_SITES = {
    "narrow": ("narrow_distal_pin_pad", "narrow_distal_pin_loop"),
    "wide": ("wide_distal_pin_pad", "wide_distal_pin_loop"),
}
CONTACT_EXCLUSIONS = (
    ("x2cand_assembly_base", PREFIX + "R_hand_narrow1_Link"),
    ("x2cand_assembly_base", PREFIX + "R_hand_wide1_Link"),
    ("x2cand_assembly_base", PREFIX + "R_hand_narrow_loop_Link"),
    ("x2cand_assembly_base", PREFIX + "R_hand_wide_loop_Link"),
    (PREFIX + "R_hand_narrow1_Link", PREFIX + "R_hand_narrow2_Link"),
    (PREFIX + "R_hand_narrow2_Link", PREFIX + "R_hand_narrow3_Link"),
    (PREFIX + "R_hand_narrow3_Link", PREFIX + "R_hand_narrow_loop_Link"),
    (PREFIX + "R_hand_wide1_Link", PREFIX + "R_hand_wide2_Link"),
    (PREFIX + "R_hand_wide2_Link", PREFIX + "R_hand_wide3_Link"),
    (PREFIX + "R_hand_wide3_Link", PREFIX + "R_hand_wide_loop_Link"),
)
SERVO = {"kp_Nm_per_rad": 0.5, "kv_Nm_s_per_rad": 0.05, "force_cap_Nm": 0.1}
DT = 0.001
GRAVITY = (0.0, 0.0, -9.81)
INITIAL_PIN_FIT_BAND_MM = 0.25  # STL fit-scale diagnostic, not a vendor tolerance.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xml_object(root: ET.Element, tag: str, name: str) -> ET.Element:
    for node in root.iter(tag):
        if node.get("name") == name:
            return node
    raise KeyError(f"missing {tag} {name}")


def source_inventory() -> dict[str, Any]:
    doc = ET.parse(SOURCE_URDF).getroot()
    wanted = {
        "right_claw_joint", "R_hand_wide1_joint", "R_hand_narrow2_joint",
        "R_hand_narrow3_joint", "R_hand_narrow_loop_joint", "R_hand_wide2_joint",
        "R_hand_wide3_joint", "R_hand_wide_loop_joint", "R_hand_narrow4_joint",
        "R_hand_wide4_joint",
    }
    joints: dict[str, Any] = {}
    for node in doc.findall("joint"):
        name = node.get("name", "")
        if name not in wanted:
            continue
        origin, axis, limit, mimic = (node.find("origin"), node.find("axis"),
                                      node.find("limit"), node.find("mimic"))
        joints[name] = {
            "type": node.get("type"),
            "parent": node.find("parent").get("link"),
            "child": node.find("child").get("link"),
            "origin_xyz_m": [float(x) for x in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])],
            "origin_rpy_rad": [float(x) for x in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])],
            "axis_xyz": [float(x) for x in axis.get("xyz", "0 0 1").split()] if axis is not None else None,
            "limit": dict(limit.attrib) if limit is not None else None,
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        }
    return joints


def source_mesh_cylinder_audit() -> dict[str, Any]:
    """Read retained source-STL cylinder fits and verify the loop distal bore again."""
    audit = json.loads(MESH_AUDIT.read_text(encoding="utf-8"))
    if audit.get("source", {}).get("urdf_sha256") != SOURCE_URDF_SHA256:
        raise RuntimeError("Retained source mesh audit does not match the pinned URDF hash")
    selected: dict[str, Any] = {}
    queries = {
        "narrow1_narrow2_parent": ("R_hand_narrow1_Link", (0.030852, 0.018551, 0.0)),
        "narrow2_root": ("R_hand_narrow2_Link", (0.0, 0.0, 0.0)),
        "narrow2_narrow3_parent": ("R_hand_narrow2_Link", (0.018118, -0.01574, 0.0)),
        "narrow3_root": ("R_hand_narrow3_Link", (0.0, 0.0, 0.0)),
        "narrow3_terminal": ("R_hand_narrow3_Link", (0.0, -0.0104, 0.0)),
        "narrow_loop_base_bore": ("R_hand_narrow_loop_Link", (0.0, 0.0, 0.0)),
        "wide1_wide2_parent": ("R_hand_wide1_Link", (0.030852, -0.018551, 0.0)),
        "wide2_wide3_parent": ("R_hand_wide2_Link", (0.018118, 0.01574, 0.0)),
        "wide3_terminal": ("R_hand_wide3_Link", (0.0, 0.0104, 0.0)),
        "wide_loop_base_bore": ("R_hand_wide_loop_Link", (0.0, 0.0, 0.0)),
    }
    for label, (body, target) in queries.items():
        candidates = audit["link_meshes"][body]["cylindrical_surface_candidates"]
        row = min(candidates, key=lambda c: float(np.linalg.norm(np.asarray(c["center_link_local_m"]) - target)))
        selected[label] = {
            "source_body": body,
            "target_from_urdf_or_terminal_frame_m": list(target),
            "fit_center_local_m": row["center_link_local_m"],
            "fit_axis_local": row["axis_link_local"],
            "fit_radius_mm": abs(float(row["signed_radius_m"])) * 1000.0,
            "fit_rms_mm": float(row["fit_rms_m"]) * 1000.0,
            "fit_facets": int(row["facet_count"]),
            "fit_z_span_mm": [1000.0 * x for x in row["z_span_mesh_local_m"]],
            "center_delta_from_nominal_mm": (1000.0 * (np.asarray(row["center_link_local_m"]) - target)).tolist(),
        }

    # Refit the proposed narrow loop distal hole from raw binary STL faces,
    # seeded only by the old fit center; this does not inherit its fitted value.
    loop_path = SOURCE_MESH_DIR / "narrow_loop_Link.stl"
    raw = loop_path.read_bytes()
    tri_count = struct.unpack_from("<I", raw, 80)[0]
    face_dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    faces = np.frombuffer(raw, dtype=face_dtype, offset=84, count=tri_count)
    normals = faces["normal"].astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-9
    normals[valid] /= lengths[valid, None]
    centroids = faces["vertices"].astype(np.float64).mean(axis=1)
    seed = np.array([0.0315228467, -0.0097562996])
    nxy = normals[:, :2]
    vec = centroids[:, :2] - seed
    cross = np.abs(vec[:, 0] * nxy[:, 1] - vec[:, 1] * nxy[:, 0])
    signed_radius = np.sum(vec * nxy, axis=1)
    mask = (valid & (np.abs(normals[:, 2]) < 0.18) & (np.abs(cross) < 0.00025)
            & (np.abs(signed_radius + 0.0025) < 0.00030))
    ids = np.flatnonzero(mask)
    if len(ids) < 20:
        raise RuntimeError(f"Could not refit distal pin bore in {loop_path}; inliers={len(ids)}")
    nxy_fit = normals[ids, :2]
    pxy_fit = centroids[ids, :2]
    matrix = np.zeros((2 * len(ids), 3))
    values = np.zeros(2 * len(ids))
    matrix[0::2, 0] = 1.0
    matrix[1::2, 1] = 1.0
    matrix[0::2, 2] = nxy_fit[:, 0]
    matrix[1::2, 2] = nxy_fit[:, 1]
    values[0::2] = pxy_fit[:, 0]
    values[1::2] = pxy_fit[:, 1]
    fit = np.linalg.lstsq(matrix, values, rcond=None)[0]
    err = pxy_fit - (fit[:2] + fit[2] * nxy_fit)
    normal_cov = normals[ids].T @ normals[ids] / len(ids)
    eigval, eigvec = np.linalg.eigh(normal_cov)
    axis = eigvec[:, int(np.argmin(eigval))]
    if axis[2] < 0:
        axis *= -1.0
    selected["narrow_loop_distal_pin_raw_stl_refit"] = {
        "stl_sha256": sha256(loop_path),
        "seed_xy_mm": (seed * 1000.0).tolist(),
        "fit_center_xy_mm": (fit[:2] * 1000.0).tolist(),
        "fit_axis_local": axis.tolist(),
        "normal_covariance_eigenvalues": eigval.tolist(),
        "fit_radius_mm": abs(float(fit[2])) * 1000.0,
        "fit_rms_mm": float(np.sqrt(np.mean(np.sum(err * err, axis=1))) * 1000.0),
        "facet_count": int(len(ids)),
        "z_span_mm": [float(centroids[ids, 2].min() * 1000.0), float(centroids[ids, 2].max() * 1000.0)],
        "note": "Seeded raw-STL cylinder refit, not CAD datum or manufacturing tolerance.",
    }
    return {
        "retained_audit_path": str(MESH_AUDIT),
        "retained_audit_sha256": sha256(MESH_AUDIT),
        "features": selected,
    }


def add_contact_exclusions(root: ET.Element) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.Element("contact")
        world = root.find("worldbody")
        index = list(root).index(world) + 1
        root.insert(index, contact)
    existing = {frozenset((e.get("body1"), e.get("body2"))) for e in contact.findall("exclude")}
    for body1, body2 in CONTACT_EXCLUSIONS:
        key = frozenset((body1, body2))
        if key not in existing:
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
            existing.add(key)


def apply_diagnostic_runtime(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update({
        "timestep": f"{DT:.9g}",
        "gravity": "0 0 -9.81",
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": "100",
        "tolerance": "1e-10",
    })
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for item in actuator:
        if item.get("joint") == ROOT_DRIVER:
            item.attrib.update({
                "kp": str(SERVO["kp_Nm_per_rad"]),
                "kv": str(SERVO["kv_Nm_s_per_rad"]),
                "ctrlrange": "-1 0",
                "ctrllimited": "true",
                "forcerange": f"{-SERVO['force_cap_Nm']} {SERVO['force_cap_Nm']}",
                "forcelimited": "true",
            })
    add_contact_exclusions(root)


def write_variant(template: Path, path: Path, mode: str) -> Path:
    root = ET.parse(template).getroot()
    root.set("model", f"X2-OmniPicker-{mode}-diagnostic")
    for body in root.findall(".//worldbody//body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name in {joint_name for side in PASSIVE.values() for joint_name in side}:
                if mode == "tree" or (mode == "grounded_loop" and name in LOOP_JOINTS):
                    body.remove(joint)
                else:
                    # Hole axes are derived as local Z lines from source STL
                    # cylindrical surfaces. The sign is only a coordinate convention.
                    joint.attrib["axis"] = "0 0 1"
                    joint.attrib["limited"] = "false"
                    joint.attrib["damping"] = "0"
                    joint.attrib["frictionloss"] = "0"
    equality = root.find("equality")
    if equality is None:
        raise ValueError("Input candidate has no equality block")
    if mode == "tree":
        for element in list(equality):
            if element.tag == "connect":
                equality.remove(element)
    apply_diagnostic_runtime(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n", encoding="utf-8")
    return path


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise KeyError(name)
    return value


def qadr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def vadr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def site_gap(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[float, list[float]]:
    site_a, site_b = PIN_SITES[side]
    ia = object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_a)
    ib = object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_b)
    delta = data.site_xpos[ia] - data.site_xpos[ib]
    return float(np.linalg.norm(delta)), (delta * 1000.0).astype(float).tolist()


def geometric_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2)
        rows.append({
            "body1": name1,
            "body2": name2,
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2),
            "distance_mm": float(con.dist * 1000.0),
            "penetration_proxy_mm": max(0.0, -float(con.dist * 1000.0)),
        })
    return rows


def static_scan(model: mujoco.MjModel, mode: str, samples: int) -> dict[str, Any]:
    data = mujoco.MjData(model)
    rows = []
    available: dict[str, list[str]] = {}
    for side, names in PASSIVE.items():
        available[side] = [n for n in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    previous = {side: np.zeros(len(available[side]), dtype=float) for side in available}
    q_values = np.linspace(0.0, -1.0, samples)

    for q in q_values:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr(model, ROOT_DRIVER)] = float(q)
        data.qpos[qadr(model, ROOT_FOLLOWER)] = -float(q)
        side_rows = {}
        for side, names in available.items():
            addrs = [qadr(model, n) for n in names]
            site_a = object_id(model, mujoco.mjtObj.mjOBJ_SITE, PIN_SITES[side][0])
            site_b = object_id(model, mujoco.mjtObj.mjOBJ_SITE, PIN_SITES[side][1])

            def residual(values: np.ndarray) -> np.ndarray:
                data.qpos[addrs] = values
                mujoco.mj_forward(model, data)
                return data.site_xpos[site_a] - data.site_xpos[site_b]

            if names:
                solution = least_squares(
                    residual, previous[side],
                    xtol=1e-12, ftol=1e-12, gtol=1e-12,
                    max_nfev=250, diff_step=1e-5, x_scale="jac",
                )
                previous[side] = solution.x.copy()
                data.qpos[addrs] = solution.x
                optimizer_success = bool(solution.success)
                nfev = int(solution.nfev)
            else:
                optimizer_success = True
                nfev = 0
            mujoco.mj_forward(model, data)
            gap, vector = site_gap(model, data, side)
            side_rows[side] = {
                "passive_joint_rad": {n: float(data.qpos[qadr(model, n)]) for n in names},
                "closure_residual_mm": gap * 1000.0,
                "closure_error_vector_mm": vector,
                "optimizer_success": optimizer_success,
                "optimizer_evaluations": nfev,
            }
        rows.append({"driver_q_rad": float(q), "sides": side_rows})

    max_closure = max(v["closure_residual_mm"] for row in rows for v in row["sides"].values())
    continuity = {}
    for side in ("narrow", "wide"):
        joints = list(available[side])
        if not joints:
            continuity[side] = {"max_abs_step_delta_rad": 0.0, "joint": None, "q_from": None, "q_to": None}
            continue
        values = np.asarray([[row["sides"][side]["passive_joint_rad"][j] for j in joints] for row in rows])
        delta = np.diff(values, axis=0)
        if delta.size:
            idx = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
            continuity[side] = {
                "max_abs_step_delta_rad": float(abs(delta[idx])),
                "joint": joints[idx[1]],
                "q_from": rows[idx[0]]["driver_q_rad"],
                "q_to": rows[idx[0] + 1]["driver_q_rad"],
            }
        else:
            continuity[side] = {"max_abs_step_delta_rad": 0.0, "joint": None, "q_from": None, "q_to": None}
    return {
        "mode": mode,
        "sample_count": samples,
        "q_range_rad": [0.0, -1.0],
        "max_site_closure_residual_mm": max_closure,
        "passive_joint_names": available,
        "static_solution_continuity": continuity,
        "interpretation": "Static kinematic best-fit only; optimizer success does not prove a unique physical branch.",
        "rows": rows,
    }


def initialize_dynamic_candidate(model: mujoco.MjModel, closed_row: dict[str, Any]) -> tuple[mujoco.MjData, dict[str, Any]]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[qadr(model, ROOT_DRIVER)] = 0.0
    data.qpos[qadr(model, ROOT_FOLLOWER)] = 0.0
    for side in ("narrow", "wide"):
        for name, value in closed_row["sides"][side]["passive_joint_rad"].items():
            data.qpos[qadr(model, name)] = float(value)
    aid = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "diagnostic_driver_position")
    data.ctrl[aid] = 0.0
    mujoco.mj_forward(model, data)
    initial = {
        "driver_q_rad": float(data.qpos[qadr(model, ROOT_DRIVER)]),
        "follower_q_rad": float(data.qpos[qadr(model, ROOT_FOLLOWER)]),
        "passive_q_rad": {
            side: {name: float(data.qpos[qadr(model, name)])
                   for name in PASSIVE[side]
                   if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0}
            for side in ("narrow", "wide")
        },
        "pin_residual_mm": {side: site_gap(model, data, side)[0] * 1000.0 for side in ("narrow", "wide")},
        "active_contacts": geometric_contacts(model, data),
        "one_time_initialization_qpos_write_count": 2 + sum(
            len(v["passive_joint_rad"]) for v in closed_row["sides"].values()
        ),
    }
    return data, initial


def camera(model: mujoco.MjModel) -> tuple[mujoco.MjvCamera, mujoco.MjvOption]:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.075]
    cam.distance = 0.38
    cam.azimuth = 135.0
    cam.elevation = -18.0
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    return cam, opt


def render_panel(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
                 cam: mujoco.MjvCamera, opt: mujoco.MjvOption, title: str,
                 q_target: float, pin_mm: float | None) -> Image.Image:
    renderer.update_scene(data, cam, scene_option=opt)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 44), fill=(18, 27, 39))
    draw.text((14, 8), title, fill=(245, 248, 250))
    draw.text((14, 25), f"target q={q_target:+.3f} rad" + (f"  pin gap={pin_mm:.2f} mm" if pin_mm is not None else ""), fill=(225, 232, 238))
    return image


def gather_model_row(model: mujoco.MjModel, data: mujoco.MjData, step: int, phase: str,
                     target: float, actuator_id: int, passive_names: dict[str, list[str]]) -> dict[str, Any]:
    q_driver = float(data.qpos[qadr(model, ROOT_DRIVER)])
    q_follower = float(data.qpos[qadr(model, ROOT_FOLLOWER)])
    contacts = geometric_contacts(model, data)
    passive = {}
    passive_vel = {}
    passive_acc = {}
    for side, names in passive_names.items():
        passive[side] = {name: float(data.qpos[qadr(model, name)]) for name in names}
        passive_vel[side] = {name: float(data.qvel[vadr(model, name)]) for name in names}
        passive_acc[side] = {name: float(data.qacc[vadr(model, name)]) for name in names}
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all())
    root_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_DRIVER)
    follower_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_FOLLOWER)
    root_range = model.jnt_range[root_joint].astype(float).tolist()
    follower_range = model.jnt_range[follower_joint].astype(float).tolist()
    limit_violation = max(root_range[0] - q_driver, q_driver - root_range[1],
                          follower_range[0] - q_follower, q_follower - follower_range[1], 0.0)
    return {
        "step": step,
        "time_s": float(data.time),
        "phase": phase,
        "target_q_rad": target,
        "root_q_rad": q_driver,
        "root_qvel_rad_s": float(data.qvel[vadr(model, ROOT_DRIVER)]),
        "root_qacc_rad_s2": float(data.qacc[vadr(model, ROOT_DRIVER)]),
        "follower_q_rad": q_follower,
        "root_mimic_residual_rad": q_follower + q_driver,
        "passive_q_rad": passive,
        "passive_qvel_rad_s": passive_vel,
        "passive_qacc_rad_s2": passive_acc,
        "actuator_force_nm": float(data.actuator_force[actuator_id]),
        "pin_residual_mm": {side: site_gap(model, data, side)[0] * 1000.0 for side in ("narrow", "wide")
                             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PIN_SITES[side][0]) >= 0},
        "contacts": contacts,
        "ncon": int(data.ncon),
        "max_active_penetration_proxy_mm": max((c["penetration_proxy_mm"] for c in contacts), default=0.0),
        "solver_iterations_max": int(np.max(data.solver_niter)) if data.solver_niter.size else 0,
        "nefc": int(data.nefc),
        "source_root_limit_violation_rad": float(limit_violation),
        "finite_state": finite,
    }


def run_dynamic(tree_model: mujoco.MjModel, loop_model: mujoco.MjModel,
                initial_candidate: dict[str, Any], out: Path,
                leg_seconds: float, render_fps: int) -> dict[str, Any]:
    tree = mujoco.MjData(tree_model)
    mujoco.mj_resetData(tree_model, tree)
    tree.qpos[qadr(tree_model, ROOT_DRIVER)] = 0.0
    tree.qpos[qadr(tree_model, ROOT_FOLLOWER)] = 0.0
    tree_actuator = object_id(tree_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "diagnostic_driver_position")
    tree.ctrl[tree_actuator] = 0.0
    mujoco.mj_forward(tree_model, tree)
    loop, loop_initial = initialize_dynamic_candidate(loop_model, initial_candidate)
    loop_actuator = object_id(loop_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "diagnostic_driver_position")

    tree_passive = {"narrow": [], "wide": []}
    loop_passive = {
        side: [name for name in PASSIVE[side] if mujoco.mj_name2id(loop_model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0]
        for side in ("narrow", "wide")
    }
    dt = float(tree_model.opt.timestep)
    leg_steps = max(1, int(round(leg_seconds / dt)))
    stride = max(1, int(round(1.0 / (render_fps * dt))))
    phases = (("open", 0.0, -1.0), ("close", -1.0, 0.0), ("reopen", 0.0, -1.0))
    tree_renderer = mujoco.Renderer(tree_model, height=480, width=640)
    loop_renderer = mujoco.Renderer(loop_model, height=480, width=640)
    tree_cam, tree_opt = camera(tree_model)
    loop_cam, loop_opt = camera(loop_model)
    frames: list[Image.Image] = []
    milestone_frames: dict[str, Image.Image] = {}
    trace_path = out / "step_trace.jsonl"
    csv_path = out / "step_trace.csv"
    first_nonfinite = None
    total_steps = 0
    max_root_track_error = {"tree": 0.0, "closed_loop": 0.0}
    max_passive_delta = {side: {name: 0.0 for name in loop_passive[side]} for side in loop_passive}
    previous_passive = {side: {name: float(loop.qpos[qadr(loop_model, name)]) for name in loop_passive[side]}
                        for side in loop_passive}
    max_closure = {"narrow": 0.0, "wide": 0.0}
    max_penetration = {"tree": 0.0, "closed_loop": 0.0}
    max_solver_iterations = {"tree": 0, "closed_loop": 0}
    root_limit_violation_steps = {"tree": 0, "closed_loop": 0}
    passive_delta_events = 0
    qpos_write_count_after_start = 0

    def compose_frame(target: float) -> Image.Image:
        tree_image = render_panel(tree_renderer, tree_model, tree, tree_cam, tree_opt,
                                  "OFFICIAL URDF TREE APPROXIMATION", target, None)
        pin = max(site_gap(loop_model, loop, side)[0] * 1000.0 for side in ("narrow", "wide"))
        loop_image = render_panel(loop_renderer, loop_model, loop, loop_cam, loop_opt,
                                  "SIMULATION_ONLY PASSIVE-LOOP HYPOTHESIS", target, pin)
        combined = Image.new("RGB", (tree_image.width + loop_image.width, tree_image.height), (14, 19, 28))
        combined.paste(tree_image, (0, 0))
        combined.paste(loop_image, (tree_image.width, 0))
        return combined

    csv_fields = ["step", "time_s", "phase", "target_q_rad", "tree_q_rad", "tree_qvel_rad_s",
                  "tree_actuator_force_nm", "tree_contact_count", "tree_max_penetration_mm",
                  "loop_q_rad", "loop_qvel_rad_s", "loop_qacc_rad_s2", "loop_follower_q_rad",
                  "loop_mimic_residual_rad", "narrow_pin_gap_mm", "wide_pin_gap_mm",
                  "loop_actuator_force_nm", "loop_contact_count", "loop_max_penetration_mm",
                  "solver_iterations_max", "finite_state"]
    with trace_path.open("w", encoding="utf-8") as json_stream, csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=csv_fields)
        writer.writeheader()
        frames.append(compose_frame(0.0))
        milestone_frames["closed_start"] = frames[0].copy()
        for phase, start, end in phases:
            for i in range(leg_steps):
                u = (i + 1) / leg_steps
                smooth = u * u * (3.0 - 2.0 * u)
                target = start + (end - start) * smooth
                tree.ctrl[tree_actuator] = target
                loop.ctrl[loop_actuator] = target
                # No qpos assignments occur after this point; both worlds advance only with mj_step.
                mujoco.mj_step(tree_model, tree)
                mujoco.mj_step(loop_model, loop)
                total_steps += 1
                tree_row = gather_model_row(tree_model, tree, total_steps, phase, target, tree_actuator, tree_passive)
                loop_row = gather_model_row(loop_model, loop, total_steps, phase, target, loop_actuator, loop_passive)
                row = {"tree": tree_row, "closed_loop": loop_row}
                json_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                max_root_track_error["tree"] = max(max_root_track_error["tree"], abs(tree_row["root_q_rad"] - target))
                max_root_track_error["closed_loop"] = max(max_root_track_error["closed_loop"], abs(loop_row["root_q_rad"] - target))
                max_penetration["tree"] = max(max_penetration["tree"], tree_row["max_active_penetration_proxy_mm"])
                max_penetration["closed_loop"] = max(max_penetration["closed_loop"], loop_row["max_active_penetration_proxy_mm"])
                max_solver_iterations["tree"] = max(max_solver_iterations["tree"], tree_row["solver_iterations_max"])
                max_solver_iterations["closed_loop"] = max(max_solver_iterations["closed_loop"], loop_row["solver_iterations_max"])
                if tree_row["source_root_limit_violation_rad"] > 1e-9:
                    root_limit_violation_steps["tree"] += 1
                if loop_row["source_root_limit_violation_rad"] > 1e-9:
                    root_limit_violation_steps["closed_loop"] += 1
                for side in ("narrow", "wide"):
                    gap = loop_row["pin_residual_mm"].get(side, 0.0)
                    max_closure[side] = max(max_closure[side], gap)
                    for joint_name, q in loop_row["passive_q_rad"][side].items():
                        delta = abs(q - previous_passive[side][joint_name])
                        max_passive_delta[side][joint_name] = max(max_passive_delta[side][joint_name], delta)
                        if delta > 0.1:
                            passive_delta_events += 1
                        previous_passive[side][joint_name] = q
                finite = tree_row["finite_state"] and loop_row["finite_state"]
                if not finite and first_nonfinite is None:
                    first_nonfinite = total_steps
                writer.writerow({
                    "step": total_steps, "time_s": loop_row["time_s"], "phase": phase,
                    "target_q_rad": target, "tree_q_rad": tree_row["root_q_rad"],
                    "tree_qvel_rad_s": tree_row["root_qvel_rad_s"],
                    "tree_actuator_force_nm": tree_row["actuator_force_nm"],
                    "tree_contact_count": tree_row["ncon"],
                    "tree_max_penetration_mm": tree_row["max_active_penetration_proxy_mm"],
                    "loop_q_rad": loop_row["root_q_rad"], "loop_qvel_rad_s": loop_row["root_qvel_rad_s"],
                    "loop_qacc_rad_s2": loop_row["root_qacc_rad_s2"],
                    "loop_follower_q_rad": loop_row["follower_q_rad"],
                    "loop_mimic_residual_rad": loop_row["root_mimic_residual_rad"],
                    "narrow_pin_gap_mm": loop_row["pin_residual_mm"].get("narrow"),
                    "wide_pin_gap_mm": loop_row["pin_residual_mm"].get("wide"),
                    "loop_actuator_force_nm": loop_row["actuator_force_nm"],
                    "loop_contact_count": loop_row["ncon"],
                    "loop_max_penetration_mm": loop_row["max_active_penetration_proxy_mm"],
                    "solver_iterations_max": loop_row["solver_iterations_max"], "finite_state": int(finite),
                })
                if total_steps % stride == 0 or i == leg_steps - 1:
                    frame = compose_frame(target)
                    frames.append(frame)
                    if i == leg_steps - 1:
                        milestone_frames[phase] = frame.copy()
                if not finite:
                    break
            if first_nonfinite is not None:
                break
    tree_renderer.close()
    loop_renderer.close()

    gif_path = out / "tree_vs_closed_loop.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
    for name, frame in milestone_frames.items():
        frame.save(out / f"frame_{name}.png")

    return {
        "status": "FINISHED_FINITE" if first_nonfinite is None else "STOPPED_NONFINITE",
        "mj_step_calls_per_model": total_steps,
        "mj_step_calls_total": 2 * total_steps,
        "time_per_leg_s": leg_seconds,
        "phase_sequence": ["open 0→-1", "close -1→0", "reopen 0→-1"],
        "active_rollout_qpos_writes": qpos_write_count_after_start,
        "one_time_closed_state_initialization": loop_initial,
        "first_nonfinite_step": first_nonfinite,
        "max_root_tracking_error_rad": max_root_track_error,
        "max_pin_gap_mm": max_closure,
        "max_passive_step_delta_rad": max_passive_delta,
        "passive_step_delta_events_gt_0p1_rad": passive_delta_events,
        "max_active_contact_penetration_proxy_mm": max_penetration,
        "max_solver_iterations": max_solver_iterations,
        "root_source_limit_violation_steps": root_limit_violation_steps,
        "output": {
            "trace_jsonl": str(trace_path), "trace_csv": str(csv_path), "side_by_side_video_gif": str(gif_path),
            "frame_files": [str(out / f"frame_{name}.png") for name in milestone_frames],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--source-urdf", type=Path, default=SOURCE_URDF)
    parser.add_argument("--mesh-audit", type=Path, default=MESH_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--static-samples", type=int, default=101)
    parser.add_argument("--leg-seconds", type=float, default=1.0)
    parser.add_argument("--render-fps", type=int, default=20)
    args = parser.parse_args()
    if mujoco.__version__ != "3.3.6":
        raise RuntimeError(f"Expected MuJoCo 3.3.6; found {mujoco.__version__}")
    template = args.template.resolve(strict=True)
    source_urdf = args.source_urdf.resolve(strict=True)
    if sha256(source_urdf) != SOURCE_URDF_SHA256:
        raise RuntimeError("Pinned X2 OmniPicker URDF SHA-256 does not match")
    if sha256(template) != TEMPLATE_SHA256:
        raise RuntimeError("Retained diagnostic template changed; refusing to overwrite or adapt it")
    source_git = SOURCE_ROOT
    source_head = __import__("subprocess").check_output(["git", "-C", str(source_git), "rev-parse", "HEAD"], text=True).strip()
    source_status = __import__("subprocess").check_output(["git", "-C", str(source_git), "status", "--porcelain"], text=True).strip()
    if source_head != SOURCE_PIN or source_status:
        raise RuntimeError(f"Pinned vendor checkout is not clean/exact: {source_head} {source_status}")

    args.out.mkdir(parents=True, exist_ok=True)
    tree_path = write_variant(template, args.out / "source_tree_control.xml", "tree")
    grounded_path = write_variant(template, args.out / "fixed_loop_fourbar_counterfactual.xml", "grounded_loop")
    loop_path = write_variant(template, args.out / "passive_loop_candidate.xml", "closed_loop")
    tree_model = mujoco.MjModel.from_xml_path(str(tree_path))
    grounded_model = mujoco.MjModel.from_xml_path(str(grounded_path))
    loop_model = mujoco.MjModel.from_xml_path(str(loop_path))
    mesh_geometry = source_mesh_cylinder_audit()
    tree_static = {"mode": "tree", "nq": int(tree_model.nq), "neq": int(tree_model.neq), "nu": int(tree_model.nu),
                   "interpretation": "Reference serial/tree approximation; passive-looking X2 sublinks stay fixed."}
    grounded_scan = static_scan(grounded_model, "loop_link_fixed_to_ground", max(101, args.static_samples))
    loop_scan = static_scan(loop_model, "loop_link_passive_hinge", max(101, args.static_samples))
    (args.out / "fixed_loop_static_scan.json").write_text(json.dumps(grounded_scan, indent=2) + "\n", encoding="utf-8")
    (args.out / "closed_loop_static_scan.json").write_text(json.dumps(loop_scan, indent=2) + "\n", encoding="utf-8")
    closed_seed = min(loop_scan["rows"], key=lambda row: abs(row["driver_q_rad"]))
    max_initial_gap = max(v["closure_residual_mm"] for v in closed_seed["sides"].values())
    if max_initial_gap > INITIAL_PIN_FIT_BAND_MM:
        raise RuntimeError(f"Candidate initial closure {max_initial_gap:.6f} mm exceeds documented STL-fit diagnostic band {INITIAL_PIN_FIT_BAND_MM:.3f} mm; dynamics not started")

    dynamic = run_dynamic(tree_model, loop_model, closed_seed, args.out, args.leg_seconds, args.render_fps)
    source = source_inventory()
    report = {
        "attempt_id": "x2-closed-loop-20261017-run01",
        "classification": "SIMULATION_ONLY source-mesh-derived passive-loop hypothesis; not a vendor-confirmed mechanism",
        "source": {
            "repository": "https://github.com/AgibotTech/agibot_x2_urdf",
            "commit": SOURCE_PIN,
            "license": "Mulan PSL v2",
            "urdf_path": str(source_urdf), "urdf_sha256": sha256(source_urdf),
            "joint_inventory": source,
        },
        "robotiq_reference": {
            "repository": "https://github.com/google-deepmind/mujoco_menagerie",
            "commit": MENAGERIE_COMMIT,
            "model_path": "robotiq_2f85/2f85.xml",
            "model_sha256": MENAGERIE_XML_SHA256,
            "model_url": MENAGERIE_XML_URL,
            "architecture_summary": "per-side driver/coupler/spring-link/follower revolutes; connect closure; driver coupling equality; fixed split tendon actuator; explicit adjacent-link collision excludes; spring-link torsional spring data",
        },
        "x2_mesh_pivot_evidence": mesh_geometry,
        "input_diagnostic_template": {
            "path": str(template), "sha256": sha256(template),
            "source_template_role": "prior isolated source-geometry conversion; read-only input, not accepted mechanism",
        },
        "variants": {
            "tree_control": {"path": str(tree_path), "sha256": sha256(tree_path), **tree_static},
            "fixed_loop_counterfactual": {
                "path": str(grounded_path), "sha256": sha256(grounded_path),
                "static_scan": {k: v for k, v in grounded_scan.items() if k != "rows"},
            },
            "passive_loop_candidate": {
                "path": str(loop_path), "sha256": sha256(loop_path),
                "nq": int(loop_model.nq), "neq": int(loop_model.neq), "nu": int(loop_model.nu),
                "static_scan": {k: v for k, v in loop_scan.items() if k != "rows"},
            },
        },
        "dynamic": dynamic,
        "simulation_derived_settings": {
            "timestep_s": DT, "gravity_m_s2": list(GRAVITY), "integrator": "implicitfast",
            "solver": "Newton", "solver_iterations": 100,
            "driver_servo": SERVO,
            "passive_joint_limits": "none added; official URDF declares these joints fixed and no verified passive limits are available",
            "passive_joint_springs": "none added; X2 source provides no spring stiffness/reference/force data",
            "contact_exclusions": [list(x) for x in CONTACT_EXCLUSIONS],
        },
        "interpretation": {
            "source_tree_model": "The official URDF captures a one-DOF root/mimic jaw motion but does not encode internal linkage loops.",
            "fixed_loop_counterfactual": "Tested the source fixed loop-Link as ground; full-range distal closure is quantified separately.",
            "passive_loop_candidate": "Uses mesh-aligned potential revolute axes and distal pin sites. It is a topology hypothesis; any residual free passive mode or dynamic drift is evidence of missing mechanism data, not permission to add arbitrary springs or couplings.",
            "full_robot_integration": "NOT RUN; stop for human review before integration.",
        },
        "mujoco_version": mujoco.__version__,
    }
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "REPRODUCE.md").write_text(
        "# Reproduction\n\n"
        "Run from WSL2 Ubuntu 22.04 with the pinned MuJoCo 3.3.6 environment. The command creates a separate tree control, a fixed-loop counterfactual, a passive-loop candidate, static scans, one actual open/close/reopen `mj_step` run, traces, and a side-by-side GIF. It does not modify the official URDF or previous baselines.\n\n"
        "```bash\n"
        "env MUJOCO_GL=egl /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \\\n"
        f"  {Path(__file__).resolve()} \\\n"
        f"  --template {template} \\\n"
        f"  --source-urdf {source_urdf} \\\n"
        f"  --out {args.out.resolve()}\n"
        "```\n", encoding="utf-8")
    print(json.dumps({
        "result": report["dynamic"]["status"],
        "fixed_loop_max_closure_mm": grounded_scan["max_site_closure_residual_mm"],
        "movable_loop_static_max_closure_mm": loop_scan["max_site_closure_residual_mm"],
        "movable_loop_branch_delta": loop_scan["static_solution_continuity"],
        "mj_step_calls_total": dynamic["mj_step_calls_total"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    main()
