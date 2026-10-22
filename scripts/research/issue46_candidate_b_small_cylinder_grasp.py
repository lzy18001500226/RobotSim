#!/usr/bin/env python3
"""One bounded Candidate B small-cylinder contact, load-transfer, and lift trial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import issue46_urdf_pipeline as p
import issue46_single_motor_travel as travel
from issue46_candidate_b_functional_validation import copy_frozen_candidate, cylinder_band_fit
from issue46_dpoint_legal_motion import TriangleBVH

BRANCH_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = p.REPO / "reviews/issue-46/single-motor-full-travel-20261022T084103Z"
DIAMETER_M, HEIGHT_M, DENSITY_KG_M3 = 0.025, 0.210, 1000.0
OBJECT_FRICTION, TABLE_FRICTION = "1.4 0.02 0.001", "1 0.01 0.001"
FIXTURE_NECK_WORLD = np.array([0.0, 0.0, 0.203])
GRASP_Z_M = float(FIXTURE_NECK_WORLD[2])
CONTACT_BAND_HEIGHT_MM = 14.0
OPEN_CAD, CLOSE_CAD, DT = 0.2, -0.075, 0.001
SETTLE_S, CLOSE_S, HOLD_S = 0.5, 3.5, 0.75
UNLOAD_MM, UNLOAD_S, UNLOAD_HOLD_S = 0.5, 0.5, 0.2
LIFT_MM, LIFT_S, LIFT_HOLD_S = 5.0, 1.0, 0.5
CONTACT_FORCE_FLOOR_N = 0.01
MAX_ROOT_LIMIT_ERROR_RAD, MAX_D_GAP_MM = 1e-5, 1.0
MAX_COUPLING_RAD, MAX_JOINT_SPEED_RAD_S = 0.01, 25.0
MAX_PENETRATION_MM, MAX_PRELIFT_DRIFT_MM = 2.0, 5.0
TABLE_GEOM, OBJECT_BODY, OBJECT_GEOM = "small_cylinder_support_plane", "test_cylinder", "test_cylinder_collision"
FIXTURE_JOINT, FIXTURE_ACTUATOR = "fixture_lift", "fixture_lift_servo"


def cylinder_properties(diameter_m: float, height_m: float, density_kg_m3: float) -> dict:
    if not all(math.isfinite(v) and v > 0 for v in (diameter_m, height_m, density_kg_m3)):
        raise ValueError("cylinder dimensions and density must be finite and positive")
    radius = diameter_m / 2.0
    volume = math.pi * radius**2 * height_m
    mass = density_kg_m3 * volume
    i_axis = 0.5 * mass * radius**2
    i_transverse = mass * (3.0 * radius**2 + height_m**2) / 12.0
    return {"diameter_m": diameter_m, "height_m": height_m, "radius_m": radius,
            "density_kg_m3": density_kg_m3, "volume_m3": volume, "mass_kg": mass,
            "inertia_kg_m2": [i_transverse, i_transverse, i_axis]}


def force_on_geom2_world(contact_frame: np.ndarray, force_local: np.ndarray) -> np.ndarray:
    return np.asarray(contact_frame, dtype=float).reshape(3, 3).T @ np.asarray(force_local, dtype=float)[:3]


def force_on_object_world(contact_frame: np.ndarray, force_local: np.ndarray,
                          object_is_geom2: bool) -> np.ndarray:
    return (1.0 if object_is_geom2 else -1.0) * force_on_geom2_world(contact_frame, force_local)


def object_normal_world(contact_frame: np.ndarray, object_is_geom2: bool) -> np.ndarray:
    frame = np.asarray(contact_frame, dtype=float).reshape(3, 3)
    return (1.0 if object_is_geom2 else -1.0) * frame[0]


def cylinder_surface_triangles(center: np.ndarray, radius: float, height: float,
                               segments: int = 256) -> np.ndarray:
    if segments < 32:
        raise ValueError("cylinder triangulation requires at least 32 sides")
    cx, cy, cz = np.asarray(center, dtype=float)
    angle = np.arange(segments) * (2.0 * math.pi / segments)
    low_z, high_z = cz - height/2.0, cz + height/2.0
    low = np.column_stack((cx+radius*np.cos(angle), cy+radius*np.sin(angle), np.full(segments, low_z)))
    high = low.copy(); high[:, 2] = high_z
    bottom, top = np.array([cx, cy, low_z]), np.array([cx, cy, high_z])
    triangles = []
    for i in range(segments):
        j = (i+1) % segments
        triangles.extend(((low[i], low[j], high[j]), (low[i], high[j], high[i]),
                          (bottom, low[j], low[i]), (top, high[i], high[j])))
    return np.asarray(triangles, dtype=float)


def scoped_root_feedforward(model: mujoco.MjModel, data: mujoco.MjData,
                            gripper_dofs: np.ndarray, root_dof: int) -> tuple[float, float]:
    if data.efc_J.size != data.nefc * model.nv:
        raise ValueError("Candidate B compensation expects the reviewed dense equality Jacobian layout")
    equality = np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    jac = data.efc_J.reshape(data.nefc, model.nv)[equality][:, gripper_dofs]
    _, singular, vh = np.linalg.svd(jac, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-9))
    nullspace = vh[rank:]
    if nullspace.shape[0] != 1:
        raise ValueError(f"Candidate B gripper nullspace has {nullspace.shape[0]} DOFs, expected one")
    tangent = nullspace[0]
    root_local = int(np.flatnonzero(gripper_dofs == root_dof)[0])
    if abs(tangent[root_local]) < 1e-8:
        raise ValueError("Candidate B root tangent is singular")
    tangent = tangent / tangent[root_local]
    return float(tangent @ data.qfrc_bias[gripper_dofs]), float(np.linalg.norm(jac @ nullspace[0]))


def put_json(path: Path, value: dict) -> None:
    p.put(path.parent, path.name, value)


def git_identity() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(BRANCH_ROOT), *args], text=True).strip()
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def create_fixture_xml(output: Path, object_props: dict) -> tuple[Path, dict]:
    root = ET.parse(output / "single_motor.xml").getroot()
    world, actuator = root.find("worldbody"), root.find("actuator")
    base = world.find("body") if world is not None else None
    if base is None or base.get("name") != "R_omnipicker_base_link" or actuator is None:
        raise ValueError("unexpected frozen Candidate B fixture structure")
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    base_pos = FIXTURE_NECK_WORLD - rotation @ np.array([0.0, 0.0, 0.132])
    base.set("quat", p.quat(rotation)); base.set("pos", p.fmt(base_pos))
    ET.SubElement(base, "joint", name=FIXTURE_JOINT, type="slide", axis="1 0 0",
                  limited="true", range="0 0.02")
    ET.SubElement(actuator, "position", name=FIXTURE_ACTUATOR, joint=FIXTURE_JOINT,
                  kp="2000", kv="90", forcelimited="true", forcerange="-50 50",
                  ctrllimited="true", ctrlrange="0 0.02")
    ET.SubElement(world, "geom", name=TABLE_GEOM, type="plane", size="0 0 0.02",
                  friction=TABLE_FRICTION, condim="4", rgba="0.45 0.48 0.5 1")
    obj = ET.SubElement(world, "body", name=OBJECT_BODY)
    ET.SubElement(obj, "freejoint", name="test_cylinder_free")
    ET.SubElement(obj, "inertial", pos="0 0 0", mass=f"{object_props['mass_kg']:.16g}",
                  diaginertia=p.fmt(object_props["inertia_kg_m2"]))
    ET.SubElement(obj, "geom", name=OBJECT_GEOM, type="cylinder",
                  size=f"{object_props['radius_m']:.16g} {object_props['height_m']/2:.16g}",
                  density="0", friction=OBJECT_FRICTION, condim="4", rgba="0.2 0.65 0.8 1")
    path = output / "small_cylinder_trial.xml"
    p.put(output, path.name, ET.tostring(root, encoding="unicode") + "\n")
    return path, {"base_position_m": base_pos.tolist(), "base_rotation_matrix": rotation.tolist(),
                  "neck_reference_world_m": FIXTURE_NECK_WORLD.tolist(),
                  "fixture_lift_world_axis": [0.0, 0.0, 1.0],
                  "fixture_actuator": {"kp": 2000.0, "kv": 90.0, "force_limit_N": 50.0},
                  "table_friction": [1.0, 0.01, 0.001], "object_friction": [1.4, 0.02, 0.001]}


def initialize(model: mujoco.MjModel, data: mujoco.MjData, law: p.CandidateLaw,
               object_center: np.ndarray) -> dict:
    gripper_addrs = p.addresses(model)
    for name, value in law.at_q(OPEN_CAD).items():
        data.qpos[gripper_addrs[name]] = value
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    object_qpos = int(model.jnt_qposadr[object_joint])
    data.qpos[object_qpos:object_qpos+3] = object_center
    data.qpos[object_qpos+3:object_qpos+7] = [1.0, 0.0, 0.0, 0.0]
    fixture_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FIXTURE_JOINT)
    fixture_qpos = int(model.jnt_qposadr[fixture_joint])
    data.qpos[fixture_qpos] = 0.0
    mujoco.mj_forward(model, data)
    return {"gripper": gripper_addrs, "object_joint_id": object_joint,
            "object_qpos": object_qpos, "fixture_qpos": fixture_qpos}


def proxy_distances(model: mujoco.MjModel, data: mujoco.MjData, object_gid: int) -> list[dict]:
    rows = []
    for gid in range(model.ngeom):
        if gid == object_gid or (model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        if name == TABLE_GEOM:
            continue
        body = int(model.geom_bodyid[gid])
        fromto = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, object_gid, gid, 0.5, fromto))
        rows.append({"geom": name,
                     "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                     "distance_mm": distance*1000.0, "fromto_m": fromto.tolist()})
    return sorted(rows, key=lambda row: row["distance_mm"])


def source_mesh_intersections(model: mujoco.MjModel, data: mujoco.MjData,
                              surfaces: travel.SourceSurfaces,
                              object_triangles: np.ndarray) -> list[dict]:
    obj = TriangleBVH(object_triangles)
    hits = []
    for gid in sorted(surfaces.local):
        hit = TriangleBVH(surfaces.triangles(data, gid)).first_intersection(obj)
        if hit and hit["triangle_a"] is not None:
            body = int(model.geom_bodyid[gid])
            hits.append({"source_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                         "source_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                         "source_triangle": hit["triangle_a"], "cylinder_triangle": hit["triangle_b"],
                         "candidate_pairs_checked": hit["candidate_triangle_pairs_checked"]})
    return hits


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                    object_gid: int, table_gid: int) -> dict:
    sides = {"narrow": {"force": np.zeros(3), "normal": 0.0, "tangent": 0.0,
                         "weighted_normal": np.zeros(3), "count": 0, "geoms": set()},
             "wide": {"force": np.zeros(3), "normal": 0.0, "tangent": 0.0,
                      "weighted_normal": np.zeros(3), "count": 0, "geoms": set()}}
    table_force, other_force, details = np.zeros(3), np.zeros(3), []
    for cid, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if object_gid not in (g1, g2):
            continue
        obj_is_2 = g2 == object_gid
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, cid, local)
        force = force_on_object_world(contact.frame, local, obj_is_2)
        normal = object_normal_world(contact.frame, obj_is_2)
        normal_force = max(0.0, float(np.dot(force, normal)))
        tangent_force = float(np.linalg.norm(force-normal_force*normal))
        other_gid = g1 if obj_is_2 else g2
        other_body = int(model.geom_bodyid[other_gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or "world"
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_gid) or f"geom_{other_gid}"
        if other_gid == table_gid:
            side = "table"; table_force += force
        elif body_name == "R_hand_narrow3_Link":
            side = "narrow"
        elif body_name == "R_hand_wide3_Link":
            side = "wide"
        elif body_name.startswith("R_hand_"):
            side = "other_gripper"; other_force += force
        else:
            side = "other"; other_force += force
        if side in sides:
            item = sides[side]
            item["force"] += force; item["normal"] += normal_force
            item["tangent"] += tangent_force; item["weighted_normal"] += normal*normal_force
            item["count"] += 1; item["geoms"].add(geom_name)
        details.append({"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                        "other_body": body_name, "side": side,
                        "distance_mm": float(contact.dist)*1000.0,
                        "normal_on_object_world": normal.tolist(),
                        "force_on_object_world_N": force.tolist(),
                        "normal_force_N": normal_force, "tangential_force_N": tangent_force})
    result = {}
    for side, item in sides.items():
        norm = float(np.linalg.norm(item["weighted_normal"]))
        result[side] = {"force_on_object_N": item["force"].tolist(),
                        "normal_force_N": item["normal"], "tangential_force_N": item["tangent"],
                        "mean_normal_world": (item["weighted_normal"]/norm).tolist() if norm > 1e-12 else None,
                        "contact_count": item["count"], "contact_geoms": sorted(item["geoms"])}
    narrow, wide = result["narrow"]["mean_normal_world"], result["wide"]["mean_normal_world"]
    opposition = None if narrow is None or wide is None else float(np.dot(narrow, wide))
    return {"jaw_sides": result, "jaw_normal_opposition_dot": opposition,
            "table_force_on_object_N": table_force.tolist(), "table_vertical_support_N": float(table_force[2]),
            "other_force_on_object_N": other_force.tolist(),
            "hand_vertical_support_N": float(result["narrow"]["force_on_object_N"][2] +
                                             result["wide"]["force_on_object_N"][2] + other_force[2]),
            "contacts": details}


def select_object_center(data: mujoco.MjData, surfaces: travel.SourceSurfaces,
                         object_props: dict) -> tuple[np.ndarray, dict]:
    narrow = surfaces.triangles(data, surfaces.gid("narrow"))
    wide = surfaces.triangles(data, surfaces.gid("wide"))
    xlo = max(float(narrow[:, :, 0].min()), float(wide[:, :, 0].min()))
    xhi = min(float(narrow[:, :, 0].max()), float(wide[:, :, 0].max()))
    radius = object_props["radius_m"]
    center_interval = [xlo+radius, xhi-radius]
    fits = []
    if center_interval[0] <= center_interval[1]:
        candidates = [float(x) for x in np.linspace(center_interval[0], center_interval[1], 21)
                      if abs(float(x)) <= 0.010]
        for cx in sorted(candidates, key=abs):
            fit = cylinder_band_fit(narrow, wide, object_props["diameter_m"]*1000.0,
                                    GRASP_Z_M, cx_m=cx,
                                    height_mm=CONTACT_BAND_HEIGHT_MM,
                                    x_samples=41, z_samples=11)
            fits.append({"center_x_m": cx, **fit})
            if (fit["status"] == "PASS" and fit["clearance_mm"] is not None
                    and fit["clearance_mm"] >= 0.0 and fit.get("best_center_y_mm") is not None):
                center = np.array([cx, float(fit["best_center_y_mm"])/1000.0, HEIGHT_M/2.0])
                return center, {"status": "PASS", "jaw_x_overlap_m": [xlo, xhi],
                                "center_x_feasible_interval_m": center_interval,
                                "selected_center_x_m": cx, "selected_center_y_m": float(center[1]),
                                "contact_band_center_z_m": GRASP_Z_M,
                                "contact_band_height_mm": CONTACT_BAND_HEIGHT_MM,
                                "fit": fit, "candidate_x_fits": fits,
                                "placement_rule": "nearest-to-zero center within measured jaw overlap and 10 mm station offset bound"}
    return np.array([0.0, 0.0, HEIGHT_M/2.0]), {
        "status": "FAIL", "jaw_x_overlap_m": [xlo, xhi],
        "center_x_feasible_interval_m": center_interval,
        "contact_band_center_z_m": GRASP_Z_M,
        "contact_band_height_mm": CONTACT_BAND_HEIGHT_MM,
        "candidate_x_fits": fits,
        "placement_rule": "no in-bound full-width 25 mm cylinder center fits the measured source jaw surfaces"}


def static_closing_sweep(model: mujoco.MjModel, data: mujoco.MjData, law: p.CandidateLaw,
                         addresses: dict, object_gid: int) -> dict:
    jaw_geom_ids = {"narrow": [], "wide": []}
    for gid in range(model.ngeom):
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if body_name == "R_hand_narrow3_Link":
            jaw_geom_ids["narrow"].append(gid)
        elif body_name == "R_hand_wide3_Link":
            jaw_geom_ids["wide"].append(gid)
    if not all(jaw_geom_ids.values()):
        raise ValueError("no enabled collision geoms on both source jaw bodies")
    first = {"narrow": None, "wide": None}
    samples = []
    max_internal_penetration = 0.0
    object_body = int(model.geom_bodyid[object_gid])
    for cad in np.linspace(OPEN_CAD, CLOSE_CAD, 71):
        for name, value in law.at_q(float(cad)).items():
            data.qpos[addresses["gripper"][name]] = value
        mujoco.mj_forward(model, data)
        distances = {}
        for side, geom_ids in jaw_geom_ids.items():
            side_dist = [float(mujoco.mj_geomDistance(model, data, object_gid, gid, 0.5, None))
                         for gid in geom_ids]
            distances[side] = min(side_dist)
            if distances[side] <= 0.0001 and first[side] is None:
                first[side] = {"q_CAD": float(cad), "distance_mm": distances[side]*1000.0}
        internal = [max(0.0, -float(contact.dist)) for contact in data.contact
                    if object_body not in (int(model.geom_bodyid[contact.geom1]),
                                           int(model.geom_bodyid[contact.geom2]))]
        internal_max = max(internal, default=0.0)
        max_internal_penetration = max(max_internal_penetration, internal_max)
        samples.append({"q_CAD": float(cad),
                        "terminal_jaw_min_distance_mm": {side: value*1000.0 for side, value in distances.items()},
                        "internal_penetration_mm": internal_max*1000.0})
    return {"physics_steps": 0, "sample_count": len(samples),
            "first_terminal_jaw_contact": first,
            "both_terminal_jaws_reach_object_in_existing_interval": all(first.values()),
            "max_internal_collision_proxy_penetration_mm": max_internal_penetration*1000.0,
            "samples": samples}


def static_preflight(out: Path, model: mujoco.MjModel, data: mujoco.MjData,
                     addresses: dict, law: p.CandidateLaw,
                     surfaces: travel.SourceSurfaces, object_props: dict,
                     object_center: np.ndarray, placement: dict,
                     object_gid: int, table_gid: int) -> dict:
    for name, value in law.at_q(OPEN_CAD).items():
        data.qpos[addresses["gripper"][name]] = value
    data.qpos[addresses["fixture_qpos"]] = 0.0
    data.qpos[addresses["object_qpos"]:addresses["object_qpos"]+3] = object_center
    data.qpos[addresses["object_qpos"]+3:addresses["object_qpos"]+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    contact_fit = cylinder_band_fit(
        surfaces.triangles(data, surfaces.gid("narrow")),
        surfaces.triangles(data, surfaces.gid("wide")),
        object_props["diameter_m"]*1000.0, GRASP_Z_M,
        cx_m=float(object_center[0]), height_mm=CONTACT_BAND_HEIGHT_MM,
        x_samples=61, z_samples=15)
    cylinder_tris = cylinder_surface_triangles(object_center, object_props["radius_m"], object_props["height_m"])
    raw_hits = source_mesh_intersections(model, data, surfaces, cylinder_tris)
    distances = proxy_distances(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM))
    object_contacts_now = []
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    for contact in data.contact:
        if object_geom not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2) if int(contact.geom1) == object_geom else int(contact.geom1)
        object_contacts_now.append({"other_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                                    "distance_mm": float(contact.dist)*1000.0})
    table_contacts = [row for row in object_contacts_now if row["other_geom"] == TABLE_GEOM]
    table_penetration = max((max(0.0, -row["distance_mm"]) for row in table_contacts), default=0.0)
    gripper_contacts = [row for row in object_contacts_now if row["other_geom"] != TABLE_GEOM]
    sweep = static_closing_sweep(model, data, law, addresses, object_geom)
    min_proxy = min((row["distance_mm"] for row in distances), default=None)
    result = {"classification": "STATIC_FULL_OBJECT_PREFLIGHT; zero physics steps",
              "object": object_props, "object_center_world_m": object_center.tolist(),
              "full_object_z_extent_m": [float(object_center[2]-object_props["height_m"]/2.0),
                                         float(object_center[2]+object_props["height_m"]/2.0)],
              "source_face_placement_solution": placement,
              "source_STL_full_height_cylinder_intersections_at_open": raw_hits,
              "enabled_collision_geom_distances_at_open": distances,
              "minimum_open_gripper_proxy_distance_mm": min_proxy,
              "open_object_contacts": object_contacts_now,
              "open_gripper_contacts": gripper_contacts,
              "table_support_contacts": table_contacts,
              "table_support_max_penetration_mm": table_penetration,
              "static_closing_surface_contact_sweep": sweep,
              "physics_steps": 0}
    result["status"] = "PASS" if (placement["status"] == "PASS" and
                                   contact_fit["status"] == "PASS" and contact_fit["clearance_mm"] is not None and
                                   contact_fit["clearance_mm"] >= 0.0 and not raw_hits and
                                   not gripper_contacts and min_proxy is not None and min_proxy > 0.0 and
                                   bool(table_contacts) and table_penetration <= 0.2 and
                                   sweep["both_terminal_jaws_reach_object_in_existing_interval"] and
                                   sweep["max_internal_collision_proxy_penetration_mm"] <= MAX_PENETRATION_MM) else "FAIL"
    result["source_STL_contact_band_fit"] = contact_fit
    put_json(out / "static_preflight.json", result)
    return result


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
                 label: str) -> Image.Image:
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.13]
    camera.distance, camera.azimuth, camera.elevation = 0.48, 140, -16
    options = mujoco.MjvOption(); options.geomgroup[1] = 0
    renderer.update_scene(data, camera=camera, scene_option=options)
    image = Image.fromarray(renderer.render()).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(12, 20, 28))
    draw.text((8, 7), label, fill=(240, 225, 200))
    return image


def run_physics(output: Path, model: mujoco.MjModel, object_center: np.ndarray,
                object_props: dict) -> dict:
    data = mujoco.MjData(model)
    law = p.CandidateLaw()
    gripper_addrs = p.addresses(model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    fixture_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FIXTURE_JOINT)
    fixture_qpos = int(model.jnt_qposadr[fixture_joint])
    object_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    object_body = int(model.geom_bodyid[object_gid])
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    root_dof = int(model.jnt_dofadr[root_joint])
    root_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_right_claw_joint")
    fixture_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FIXTURE_ACTUATOR)
    gripper_dofs = np.asarray([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                               for name in gripper_addrs], dtype=int)
    for name, value in law.at_q(OPEN_CAD).items():
        data.qpos[gripper_addrs[name]] = value
    data.qpos[fixture_qpos] = 0.0
    data.qpos[object_qpos:object_qpos+3] = object_center
    data.qpos[object_qpos+3:object_qpos+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    object_initial = data.xpos[object_body].copy()
    weight = object_props["mass_kg"]*9.81
    for name, value in p.CandidateLaw().at_q(OPEN_CAD).items():
        data.qpos[gripper_addrs[name]] = value
    data.qpos[fixture_qpos] = 0.0
    data.qpos[object_qpos:object_qpos+3] = object_center
    data.qpos[object_qpos+3:object_qpos+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=640, width=800)
    video_path = output / "small_cylinder_trial.gif"
    video_frames = []
    rows, stages = [], []
    max_speed = max_penetration = max_dgap = max_coupling = max_limit = 0.0
    max_root_torque = max_fixture_force = 0.0
    step = 0
    stop_reason = None
    final_state_write_audit = {"object_qpos_writes_after_first_mj_step": 0,
                               "passive_joint_qpos_writes_after_first_mj_step": 0,
                               "object_freejoint": True, "object_attachment": False,
                               "weld_or_mocap_carry": False}

    def step_once(stage: str, cad: float, fixture_target: float,
                  bilateral_duration: float) -> tuple[dict, str | None]:
        nonlocal step, max_speed, max_penetration, max_dgap, max_coupling, max_limit
        nonlocal max_root_torque, max_fixture_force
        feedforward, _ = scoped_root_feedforward(model, data, gripper_dofs, root_dof)
        root_target = p.PHASE - cad
        kp = float(model.actuator_gainprm[root_actuator, 0])
        raw_control = root_target + feedforward/kp
        data.ctrl[root_actuator] = float(np.clip(raw_control, *model.actuator_ctrlrange[root_actuator]))
        data.ctrl[fixture_actuator] = fixture_target
        mujoco.mj_step(model, data)
        step += 1
        mujoco.mj_forward(model, data)
        contact = object_contacts(model, data, object_gid, table_gid)
        speed = float(np.max(np.abs(data.qvel[gripper_dofs])))
        penetration = max((max(0.0, -float(c.dist)) for c in data.contact), default=0.0)
        dgap = max(p.gaps(model, data))
        coupling = max(abs(v) for v in travel.coupled_residuals(model, data).values())
        root = float(data.qpos[gripper_addrs["right_claw_joint"]])
        limit_error = max(0.0, root, -1.0-root)
        pos = data.xpos[object_body].copy()
        drift = float(np.linalg.norm(pos[:2]-object_initial[:2]))
        rise = float(pos[2]-object_initial[2])
        acceleration = data.qacc[object_dof:object_dof+3].copy()
        net_contact = (np.asarray(contact["table_force_on_object_N"]) +
                       np.asarray(contact["jaw_sides"]["narrow"]["force_on_object_N"]) +
                       np.asarray(contact["jaw_sides"]["wide"]["force_on_object_N"]) +
                       np.asarray(contact["other_force_on_object_N"]))
        balance = net_contact + np.array([0.0, 0.0, -weight]) - object_props["mass_kg"]*acceleration
        row = {"step": step, "time_s": float(data.time), "stage": stage,
               "command_CAD": cad, "actual_CAD": p.PHASE-root,
               "root_qpos_rad": root, "root_qvel_rad_s": float(data.qvel[root_dof]),
               "root_actuator_torque_Nm": float(data.actuator_force[root_actuator]),
               "root_feedforward_Nm": feedforward,
               "fixture_lift_target_m": fixture_target,
               "fixture_lift_actual_m": float(data.qpos[fixture_qpos]),
               "fixture_lift_force_N": float(data.actuator_force[fixture_actuator]),
               "object_position_world_m": pos.tolist(), "object_quaternion_wxyz": data.xquat[object_body].tolist(),
               "object_velocity_xyz_m_s": data.qvel[object_dof:object_dof+3].tolist(),
               "object_acceleration_xyz_m_s2": acceleration.tolist(),
               "object_rise_m": rise, "object_xy_drift_m": drift, "object_weight_N": weight,
               "hand_vertical_support_N": contact["hand_vertical_support_N"],
               "table_vertical_support_N": contact["table_vertical_support_N"],
               "hand_support_fraction": contact["hand_vertical_support_N"]/weight,
               "table_support_fraction": contact["table_vertical_support_N"]/weight,
               "narrow_normal_N": contact["jaw_sides"]["narrow"]["normal_force_N"],
               "wide_normal_N": contact["jaw_sides"]["wide"]["normal_force_N"],
               "narrow_vertical_N": contact["jaw_sides"]["narrow"]["force_on_object_N"][2],
               "wide_vertical_N": contact["jaw_sides"]["wide"]["force_on_object_N"][2],
               "narrow_tangential_N": contact["jaw_sides"]["narrow"]["tangential_force_N"],
               "wide_tangential_N": contact["jaw_sides"]["wide"]["tangential_force_N"],
               "narrow_normal_world": contact["jaw_sides"]["narrow"]["mean_normal_world"],
               "wide_normal_world": contact["jaw_sides"]["wide"]["mean_normal_world"],
               "jaw_normal_opposition_dot": contact["jaw_normal_opposition_dot"],
               "narrow_contact_count": contact["jaw_sides"]["narrow"]["contact_count"],
               "wide_contact_count": contact["jaw_sides"]["wide"]["contact_count"],
               "narrow_contact_geoms": contact["jaw_sides"]["narrow"]["contact_geoms"],
               "wide_contact_geoms": contact["jaw_sides"]["wide"]["contact_geoms"],
               "bilateral_contact": _contact_force_threshold_pass(contact),
               "bilateral_contact_duration_s": bilateral_duration,
               "D_gap_mm": dgap*1000.0, "coupling_residual_rad": coupling,
               "root_limit_error_rad": limit_error, "max_gripper_joint_speed_rad_s": speed,
               "max_penetration_mm": penetration*1000.0,
               "contact_force_balance_residual_N": balance.tolist(),
               "contact_details": contact["contacts"]}
        rows.append(row)
        max_speed, max_penetration = max(max_speed, speed), max(max_penetration, penetration)
        max_dgap, max_coupling, max_limit = max(max_dgap, dgap), max(max_coupling, coupling), max(max_limit, limit_error)
        max_root_torque = max(max_root_torque, abs(float(data.actuator_force[root_actuator])))
        max_fixture_force = max(max_fixture_force, abs(float(data.actuator_force[fixture_actuator])))
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return row, "NONFINITE_STATE"
        if limit_error > MAX_ROOT_LIMIT_ERROR_RAD:
            return row, "OFFICIAL_ROOT_LIMIT_OVERSHOOT"
        if speed > MAX_JOINT_SPEED_RAD_S:
            return row, "CANDIDATE_JOINT_SPEED_STOP_GATE"
        if penetration > MAX_PENETRATION_MM/1000.0:
            return row, "COLLISION_PROXY_PENETRATION_OVER_2MM"
        if dgap > MAX_D_GAP_MM/1000.0:
            return row, "D_POINT_GAP_OVER_1MM"
        if coupling > MAX_COUPLING_RAD:
            return row, "CANDIDATE_COUPLING_RESIDUAL_OVER_0P01RAD"
        if drift > MAX_PRELIFT_DRIFT_MM/1000.0:
            return row, "OBJECT_XY_DRIFT_OVER_5MM"
        if any(int(warning.number) for warning in data.warning):
            return row, "MUJOCO_WARNING"
        return row, None

    def run_phase(name: str, q_start: float, q_end: float, duration: float,
                  lift_start: float, lift_end: float, *, smooth: bool = True) -> tuple[dict, str | None, list[dict]]:
        phase_rows = []
        bilateral_duration = 0.0
        count = max(1, int(round(duration/DT)))
        for index in range(count):
            alpha = min(index/max(1, count-1), 1.0)
            blend = 0.5-0.5*math.cos(math.pi*alpha) if smooth else alpha
            cad = q_start+(q_end-q_start)*blend
            lift = lift_start+(lift_end-lift_start)*blend
            row, stop = step_once(name, cad, lift, bilateral_duration)
            bilateral_duration = bilateral_duration+DT if row["bilateral_contact"] else 0.0
            row["bilateral_contact_duration_s"] = bilateral_duration
            phase_rows.append(row)
            if step % 40 == 0:
                video_frames.append(render_frame(renderer, model, data,
                                                 f"Candidate B 25 mm cylinder | {name} | {data.time:.3f}s"))
            if stop:
                return {"phase": name, "steps": len(phase_rows), "duration_s": len(phase_rows)*DT,
                        "bilateral_contact_duration_s": bilateral_duration,
                        "end": row}, stop, phase_rows
        frame = render_frame(renderer, model, data, f"Candidate B 25 mm cylinder | {name} | {data.time:.3f}s")
        frame.save(output / f"{name.lower()}_view.png")
        return {"phase": name, "steps": len(phase_rows), "duration_s": len(phase_rows)*DT,
                "bilateral_contact_duration_s": bilateral_duration,
                "end": phase_rows[-1]}, None, phase_rows

    phase_results, stop_reason, load_gate = [], None, {"status": "NOT_REACHED"}
    try:
        open_frame = render_frame(renderer, model, data, "Candidate B 25 mm cylinder | OPEN")
        open_frame.save(output / "preflight_open.png")
        schedule = [("SETTLE_OPEN", OPEN_CAD, OPEN_CAD, SETTLE_S, 0.0, 0.0, False),
                    ("CLOSE", OPEN_CAD, CLOSE_CAD, CLOSE_S, 0.0, 0.0, True),
                    ("HOLD", CLOSE_CAD, CLOSE_CAD, HOLD_S, 0.0, 0.0, False)]
        for name, q0, q1, duration, lift0, lift1, smooth in schedule:
            summary, stop_reason, _ = run_phase(name, q0, q1, duration, lift0, lift1, smooth=smooth)
            phase_results.append(summary)
            if stop_reason:
                break
        if stop_reason is None:
            settle_rows = [row for row in rows if row["stage"] == "SETTLE_OPEN"][-int(0.1/DT):]
            mean_table = float(np.mean([row["table_vertical_support_N"] for row in settle_rows]))
            settled = (len(settle_rows) >= int(0.08/DT) and
                       0.8*weight <= mean_table <= 1.2*weight and
                       max(abs(row["object_velocity_xyz_m_s"][2]) for row in settle_rows) <= 0.02)
            phase_results.append({"phase": "TABLE_SETTLE_GATE", "status": "PASS" if settled else "FAIL",
                                  "object_weight_N": weight, "mean_table_support_last_0p1s_N": mean_table})
            if not settled:
                stop_reason = "OBJECT_TABLE_SETTLE_GATE_FAILED"
        hold = next((item for item in phase_results if item.get("phase") == "HOLD"), None)
        if stop_reason is None and (hold is None or hold["bilateral_contact_duration_s"] < BILATERAL_HOLD_S):
            stop_reason = "BILATERAL_TERMINAL_JAW_HOLD_NOT_SUSTAINED_FOR_0P5S"
        transfer_rows = []
        if stop_reason is None:
            summary, stop_reason, transfer_rows = run_phase("LOAD_TRANSFER", CLOSE_CAD, CLOSE_CAD,
                                                             LOAD_TRANSFER_S, 0.0, UNLOAD_MM/1000.0)
            phase_results.append(summary)
            if stop_reason is None:
                window = transfer_rows[-int(0.1/DT):]
                passing = [row for row in window if row["bilateral_contact"] and
                           row["hand_support_fraction"] >= 0.8 and row["table_support_fraction"] <= 0.2]
                load_gate = {"status": "PASS" if len(passing) >= int(0.08/DT) else "FAIL",
                             "window_s": 0.1, "required_samples": int(0.08/DT), "passing_samples": len(passing),
                             "object_weight_N": weight, "hand_support_fraction_min": 0.8,
                             "table_support_fraction_max": 0.2, "bilateral_force_floor_N": CONTACT_FORCE_FLOOR_N}
                phase_results.append({"phase": "MEASURED_LOAD_TRANSFER_GATE", **load_gate})
                if load_gate["status"] != "PASS":
                    stop_reason = "MEASURED_LOAD_TRANSFER_NOT_ESTABLISHED"
        if stop_reason is None:
            summary, stop_reason, _ = run_phase("LOAD_TRANSFER_HOLD", CLOSE_CAD, CLOSE_CAD,
                                                 LOAD_HOLD_S, UNLOAD_MM/1000.0,
                                                 UNLOAD_MM/1000.0, smooth=False)
            phase_results.append(summary)
        if stop_reason is None:
            summary, stop_reason, _ = run_phase("SMALL_LIFT", CLOSE_CAD, CLOSE_CAD,
                                                 LIFT_S, UNLOAD_MM/1000.0,
                                                 LIFT_MM/1000.0)
            phase_results.append(summary)
        lift_rows = []
        if stop_reason is None:
            summary, stop_reason, lift_rows = run_phase("LIFT_HOLD", CLOSE_CAD, CLOSE_CAD,
                                                        LIFT_HOLD_S, LIFT_MM/1000.0,
                                                        LIFT_MM/1000.0, smooth=False)
            phase_results.append(summary)
            tail = lift_rows[-int(0.25/DT):]
            lift_pass = (len(tail) >= int(0.2/DT) and
                         min(row["object_rise_m"] for row in tail) >= 0.004 and
                         max(row["table_vertical_support_N"] for row in tail) <= 0.05 and
                         all(row["bilateral_contact"] for row in tail))
            phase_results.append({"phase": "FIVE_MM_LIFT_GATE", "status": "PASS" if lift_pass else "FAIL",
                                  "target_fixture_lift_mm": LIFT_MM,
                                  "object_rise_min_last_0p25s_mm": min((row["object_rise_m"] for row in tail), default=0.0)*1000.0,
                                  "table_support_max_last_0p25s_N": max((row["table_vertical_support_N"] for row in tail), default=0.0),
                                  "bilateral_samples": sum(row["bilateral_contact"] for row in tail),
                                  "samples": len(tail)})
            if not lift_pass:
                stop_reason = "FIVE_MM_PHYSICAL_LIFT_GATE_NOT_MET"
    finally:
        renderer.close()

    if video_frames:
        video_frames[0].save(video_path, save_all=True, append_images=video_frames[1:],
                             duration=40, loop=0, optimize=False)

    if rows:
        with (output / "physics_trace.csv").open("x", newline="", encoding="utf-8") as stream:
            fields = [key for key in rows[0] if key != "contact_details"]
            writer_csv = csv.DictWriter(stream, fieldnames=fields)
            writer_csv.writeheader()
            for row in rows:
                writer_csv.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                                     for key, value in row.items() if key != "contact_details"})
        with (output / "contact_trace.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                if row["step"] % 10 == 0:
                    stream.write(json.dumps({"step": row["step"], "time_s": row["time_s"],
                                             "stage": row["stage"], "contacts": row["contact_details"]},
                                            allow_nan=False) + "\n")
    return {"status": "PASS" if stop_reason is None and any(item.get("phase") == "FIVE_MM_LIFT_GATE" and item.get("status") == "PASS" for item in phase_results) else "FAIL",
            "stop_reason": stop_reason, "physics_steps": step,
            "simulation_time_s": float(data.time), "object": object_props, "object_weight_N": weight,
            "object_initial_position_world_m": initial_pos.tolist(), "state_write_audit": state_write_audit,
            "actuation": {"jaw_actuator_count": 1, "jaw_actuator": "right_claw_joint",
                          "fixture_carriage_actuator_count": 1, "fixture_carriage_is_not_a_jaw_or_object_force": True},
            "safety_maxima": {"gripper_joint_speed_rad_s": max_speed,
                              "collision_proxy_penetration_mm": max_penetration*1000.0,
                              "D_gap_mm": max_dgap*1000.0, "coupling_residual_rad": max_coupling,
                              "root_limit_error_rad": max_limit, "root_torque_Nm": max_root_torque,
                              "fixture_force_N": max_fixture_force},
            "load_transfer_gate": load_gate, "stages": phase_results,
            "final_state": rows[-1] if rows else None,
            "video": "small_cylinder_trial.gif"}


def write_report(output: Path, result: dict, reproduction: str) -> None:
    lines = ["# Issue #46 — Candidate B Small-Cylinder Physical Grasp", "",
             f"Status: `{result['status']}`; stop reason: `{result.get('stop_reason') or 'none'}`.", "",
             f"Object: {result['object']['diameter_m']*1000:.1f} mm diameter x {result['object']['height_m']*1000:.1f} mm height; "
             f"{result['object']['mass_kg']*1000:.3f} g solid cylinder at {result['object']['density_kg_m3']:.0f} kg/m3.",
             f"Physics steps: {result.get('physics_steps', 0)}; simulated time: {result.get('simulation_time_s', 0.0):.3f} s.",
             f"Static full-object preflight: `{result.get('static_preflight', {}).get('status', 'NOT RUN')}`.",
             f"Candidate B MJCF SHA-256: `{result.get('candidate_mjcf_sha256')}`.",
             f"Executed branch/head: `{result.get('identity', {}).get('branch')}` / `{result.get('identity', {}).get('head')}`.",
             "", "## Stages"]
    for stage in result.get("stages", []):
        lines.append(f"- `{stage.get('phase')}`: `{stage.get('status', 'RECORDED')}`; {stage}")
    lines.extend(["", "## Claim boundary", "",
                  "This is a separate minimal-object feasibility result. It does not change the 65–75 mm bottle acceptance criteria. Candidate B's transmission remains `UNVERIFIED_EXPERIMENTAL`; this does not validate manufacturer mechanics or hardware behavior.",
                  "The frozen Candidate B gripper MJCF, coupling, gripper joints, collision exclusions, root limits and root servo were hash-verified and not edited. The separate fixture carriage is used only for conditional vertical motion. The cylinder is free-jointed; there is no attachment, weld, mocap carry, or post-start object/passive qpos write.",
                  "", "## Reproduction", "", f"`{reproduction}`", "", f"Evidence directory: `{output}`"])
    p.put(output, "REPORT.md", "\n".join(lines) + "\n")
    p.put(output, "REPRODUCE.md", f"# Reproduction\n\n```bash\n{reproduction}\n```\n\nMuJoCo 3.3.6. The runner performs a static full-object preflight before at most one physical trial.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the full 3D static gate and stop before any mj_step")
    args = parser.parse_args()
    output = p.pg.create_unique_output_directory(p.REPO / "reviews/issue-46",
                                                  prefix="candidate-b-small-cylinder-grasp",
                                                  environment="wsl").native_path
    identity = git_identity()
    candidate = copy_frozen_candidate(output)
    provenance = json.loads((output / "provenance.json").read_text())
    object_props = cylinder_properties(DIAMETER_M, HEIGHT_M, DENSITY_KG_M3)
    trial_xml, fixture = create_fixture_xml(output, object_props)
    baseline_xml = output / "single_motor.xml"
    baseline = mujoco.MjModel.from_xml_path(str(baseline_xml))
    model = mujoco.MjModel.from_xml_path(str(trial_xml))
    if baseline.nu != 1 or model.nu != 2:
        raise ValueError(f"expected one frozen jaw actuator plus fixture carriage; got baseline={baseline.nu} trial={model.nu}")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    root_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_right_claw_joint")
    fixture_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FIXTURE_ACTUATOR)
    if int(model.actuator_trnid[root_actuator, 0]) != root_joint:
        raise ValueError("the only jaw actuator must still drive right_claw_joint")
    if not np.allclose(model.jnt_range[root_joint], [-1.0, 0.0], rtol=0.0, atol=1e-12):
        raise ValueError("Candidate B official root limits changed")
    source_eq = [mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(baseline.neq)]
    trial_eq = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)]
    if source_eq != trial_eq:
        raise ValueError("fixture construction changed Candidate B coupling/equalities")
    source_xml_root, trial_xml_root = ET.parse(baseline_xml).getroot(), ET.parse(trial_xml).getroot()
    source_excludes = [(node.get("body1"), node.get("body2")) for node in source_xml_root.findall("./contact/exclude")]
    trial_excludes = [(node.get("body1"), node.get("body2")) for node in trial_xml_root.findall("./contact/exclude")]
    if source_excludes != trial_excludes:
        raise ValueError("fixture construction changed Candidate B collision exclusions")
    source_geom_names = {mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(baseline.ngeom)}
    trial_geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    if not source_geom_names.issubset(trial_geom_names):
        raise ValueError("fixture construction omitted original Candidate B geometry")
    for body_id in range(1, baseline.nbody):
        body_name = mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_BODY, body_id)
        trial_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            if not np.allclose(getattr(baseline, attribute)[body_id], getattr(model, attribute)[trial_id],
                               rtol=0.0, atol=1e-12):
                raise ValueError(f"fixture changed Candidate B inertial data: {body_name}.{attribute}")

    derived = ET.parse(output / "X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf").getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    law = p.CandidateLaw()
    data = mujoco.MjData(model)
    gripper_addrs = p.addresses(model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    fixture_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FIXTURE_JOINT)
    object_qpos, fixture_qpos = int(model.jnt_qposadr[object_joint]), int(model.jnt_qposadr[fixture_joint])
    for name, value in law.at_q(OPEN_CAD).items():
        data.qpos[gripper_addrs[name]] = value
    data.qpos[fixture_qpos] = 0.0
    data.qpos[object_qpos:object_qpos+3] = [0.0, 0.0, HEIGHT_M/2.0]
    data.qpos[object_qpos+3:object_qpos+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    object_center, placement = select_object_center(data, surfaces, object_props)
    object_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    addresses = {"gripper": gripper_addrs, "object_joint_id": object_joint,
                 "object_qpos": object_qpos, "fixture_qpos": fixture_qpos}
    static = static_preflight(output, model, data, addresses, law, surfaces,
                              object_props, object_center, placement, object_gid, table_gid)
    identity.update({"branch_worktree": str(BRANCH_ROOT), "candidate_HEAD": identity["head"],
                     "candidate_MJCF_SHA256": provenance["mjcf_sha256"],
                     "trial_MJCF_SHA256": p.sha(trial_xml), "official_commit": provenance["source_commit"]})
    result_base = {"schema": "robotsim.issue46.candidate-b-small-cylinder-grasp.v1",
                   "run_id": str(uuid.uuid4()), "classification": "SIMULATION_ONLY_MINIMAL_OBJECT_FEASIBILITY",
                   "object": object_props, "identity": identity, "fixture": fixture,
                   "candidate_source_integrity": candidate,
                   "candidate_root_limits_unchanged": provenance["root_limits_unchanged"],
                   "candidate_collision_exclusions_unchanged": provenance["inherited_exclusions"],
                   "candidate_MJCF_SHA256": provenance["mjcf_sha256"],
                   "trial_MJCF_SHA256": p.sha(trial_xml), "static_preflight": static,
                   "jaw_actuator_count": 1, "jaw_actuator": "right_claw_joint",
                   "fixture_carriage_actuator": FIXTURE_ACTUATOR}
    reproduction = "MUJOCO_GL=egl python scripts/research/issue46_candidate_b_small_cylinder_grasp.py"
    if static["status"] != "PASS":
        result = {**result_base, "status": "STATIC_PREFLIGHT_BLOCKED",
                  "stop_reason": "FULL_3D_STATIC_GEOMETRY_GATE_FAILED", "physics_steps": 0}
        put_json(output / "result.json", result)
        write_report(output, result, reproduction)
        p.pg.write_artifact_manifest(output, environment="wsl")
        print(json.dumps({"output": str(output), "status": result["status"], "physics_steps": 0,
                          "placement": static["source_face_placement_solution"],
                          "preflight": {key: static[key] for key in ("status", "minimum_open_gripper_proxy_distance_mm",
                                                                      "source_STL_full_height_cylinder_intersections_at_open")}},
                         indent=2))
        return 2

    if args.preflight_only:
        for name, value in law.at_q(OPEN_CAD).items():
            data.qpos[gripper_addrs[name]] = value
        data.qpos[fixture_qpos] = 0.0
        data.qpos[object_qpos:object_qpos+3] = object_center
        data.qpos[object_qpos+3:object_qpos+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=640, width=800)
        render_frame(renderer, model, data, "Candidate B 25 mm cylinder | OPEN static preflight").save(output / "preflight_open.png")
        renderer.close()
        reproduction = "MUJOCO_GL=egl python scripts/research/issue46_candidate_b_small_cylinder_grasp.py --preflight-only"
        result = {"schema": "robotsim.issue46.candidate-b-small-cylinder-grasp.v1",
                  "run_id": str(uuid.uuid4()), "classification": "STATIC_FEASIBILITY_ONLY; not a physical grasp",
                  "status": "STATIC_PREFLIGHT_PASS_PHYSICS_NOT_RUN", "physics_steps": 0,
                  "object": object_props, "identity": identity, "fixture": fixture,
                  "candidate_source_integrity": candidate,
                  "candidate_MJCF_SHA256": provenance["mjcf_sha256"],
                  "trial_MJCF_SHA256": p.sha(trial_xml), "static_preflight": static,
                  "reproduction_command": reproduction}
        put_json(output / "result.json", result)
        write_report(output, result, reproduction)
        p.pg.write_artifact_manifest(output, environment="wsl")
        print(json.dumps({"output": str(output), "status": result["status"], "physics_steps": 0,
                          "object_center_world_m": object_center.tolist(),
                          "source_surface_fit": placement}, indent=2))
        return 0

    dynamic = run_physics(output, model, object_center, object_props)
    result = {**result_base, **dynamic, "reproduction_command": reproduction}
    put_json(output / "result.json", result)
    write_report(output, result, reproduction)
    p.pg.write_artifact_manifest(output, environment="wsl")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "stop_reason": result["stop_reason"], "physics_steps": result["physics_steps"],
                      "mass_kg": result["object"]["mass_kg"], "stages": result["stages"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

