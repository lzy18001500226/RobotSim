#!/usr/bin/env python3
"""Static MuJoCo 3.3.6 A/B audit of the X2 OmniPicker right-wrist collider.

This runner calls mj_resetData and mj_forward only. It contains no mj_step call.
The B variant changes only the source right-wrist collision mesh mapping.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


UPSTREAM = Path("/tmp/robotsim-asset-registry-upstreams/agibot_x2_urdf")
PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
MODEL_DIR = UPSTREAM / "X2_URDF-v1.4.0"
SOURCE_URDF = MODEL_DIR / "X2-Ultra_omnipicker.urdf"
MESH_DIR = MODEL_DIR / "meshes"
ARCHIVE = Path("/mnt/c/Users/HP/Desktop/RobotSim/reviews")
LEGACY = ARCHIVE / "issue46-native-omnipicker-grasp-frame-20261012"
LOCAL03 = ARCHIVE / "issue-46-omnipicker-contact-geometry-root-cause-20261012-attempt02"
WORKCELL = ARCHIVE / "issue46-native-omnipicker-workcell-20261011"
CANONICAL_HELPER = ARCHIVE / "issue46-x2-reference-physics-recovery-20261012/inputs/canonical_manipulation_assets.py"
OUT = ARCHIVE / "issue46-omnipicker-right-wrist-collision-ab-20261012"
LOCAL03_MJCF = LOCAL03 / "candidate_translation_y_minus_contact_offset/compiled_fixture_model.xml"
LOCAL03_RESULT = LOCAL03 / "candidate_translation_y_minus_contact_offset/candidate_result.json"
LOCAL03_TRACE = LOCAL03 / "candidate_translation_y_minus_contact_offset/B2_CLEARANCE_TRACE.jsonl"
FRONT_TRACE = WORKCELL / "attempt-94e0b6ec-94bf-4fd1-a1ed-a3c0a779e241-retry3/A_front_nominal_heading_geometry_trace.jsonl"
RESULT_0 = LEGACY / "attempt-3cf77655-03ca-4946-bdd9-ae3986d935ad-final3-a0/result.json"
RESULT_90 = LEGACY / "attempt-3cf77655-03ca-4946-bdd9-ae3986d935ad-final3-a90/result.json"
CANDIDATE_1 = ARCHIVE / "issue46-x2-reference-physics-recovery-20261012/candidate1_final2/result.json"
MUJOCO_VERSION = "3.3.6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vec(text: str | None, default: tuple[float, ...]) -> list[float]:
    return list(default) if text is None else [float(v) for v in text.split()]


def quaternion_from_matrix(matrix: list[list[float]] | np.ndarray) -> list[float]:
    r = np.asarray(matrix, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s,
             (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                 (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                 (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return q.tolist()


def extract_subtree(source: Path, root_link: str, variant: str, destination: Path) -> dict[str, Any]:
    source_root = ET.parse(source).getroot()
    joints = source_root.findall("joint")
    child_parent = {
        j.find("child").get("link"): j.find("parent").get("link")
        for j in joints if j.find("child") is not None and j.find("parent") is not None
    }
    included = {root_link}
    changed = True
    while changed:
        changed = False
        for child, parent in child_parent.items():
            if parent in included and child not in included:
                included.add(child)
                changed = True
    if root_link not in {node.get("name") for node in source_root.findall("link")}:
        raise RuntimeError(f"Missing subtree root {root_link}")
    target_root = ET.Element("robot", dict(source_root.attrib))
    for item in source_root:
        tag = item.tag.rsplit("}", 1)[-1]
        if tag == "mujoco":
            node = copy.deepcopy(item)
            compiler = node.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(node, "compiler")
            compiler.set("meshdir", str(MESH_DIR.resolve()))
            target_root.append(node)
        elif tag == "material":
            target_root.append(copy.deepcopy(item))
        elif tag == "link" and item.get("name") in included:
            node = copy.deepcopy(item)
            for mesh in node.findall(".//mesh"):
                mesh.set("filename", Path(mesh.get("filename", "")).name)
            target_root.append(node)
        elif tag == "joint":
            parent, child = item.find("parent"), item.find("child")
            if parent is not None and child is not None and parent.get("link") in included and child.get("link") in included:
                target_root.append(copy.deepcopy(item))
    patched = 0
    if variant == "B":
        link = next(x for x in target_root.findall("link") if x.get("name") == "right_wrist_roll_link")
        mesh = link.find("collision/geometry/mesh")
        if mesh is None or Path(mesh.get("filename", "")).name != "right_wrist_roll_link.stl":
            raise RuntimeError("Expected exactly the official right-wrist collision STL before patch")
        mesh.set("filename", "right_wrist_roll_extend_link.stl")
        patched = 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(target_root, space="  ")
    ET.ElementTree(target_root).write(destination, encoding="utf-8", xml_declaration=True)
    return {"path": str(destination), "sha256": sha256(destination), "root_link": root_link,
            "included_link_count": len(included), "included_joint_count": sum(1 for j in target_root.findall("joint")),
            "collision_mesh_mapping_changes": patched}


def make_full_variant(source: Path, variant: str, destination: Path) -> dict[str, Any]:
    root = ET.parse(source).getroot()
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        muj = root.find("mujoco")
        if muj is None:
            muj = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(muj, "compiler")
    compiler.set("meshdir", str(MESH_DIR.resolve()))
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", Path(mesh.get("filename", "")).name)
    patched = 0
    if variant == "B":
        link = next(x for x in root.findall("link") if x.get("name") == "right_wrist_roll_link")
        mesh = link.find("collision/geometry/mesh")
        if mesh is None or Path(mesh.get("filename", "")).name != "right_wrist_roll_link.stl":
            raise RuntimeError("Source collision-mesh mapping changed unexpectedly")
        mesh.set("filename", "right_wrist_roll_extend_link.stl")
        patched = 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return {"path": str(destination), "sha256": sha256(destination), "collision_mesh_mapping_changes": patched}


def audit_ab_pair(path_a: Path, path_b: Path, label: str) -> dict[str, Any]:
    root_a, root_b = ET.parse(path_a).getroot(), ET.parse(path_b).getroot()
    wrist_a = next(x for x in root_a.findall("link") if x.get("name") == "right_wrist_roll_link")
    wrist_b = next(x for x in root_b.findall("link") if x.get("name") == "right_wrist_roll_link")
    visual_a = Path(wrist_a.find("visual/geometry/mesh").get("filename", "")).name
    visual_b = Path(wrist_b.find("visual/geometry/mesh").get("filename", "")).name
    collision_a = Path(wrist_a.find("collision/geometry/mesh").get("filename", "")).name
    collision_b_element = wrist_b.find("collision/geometry/mesh")
    collision_b = Path(collision_b_element.get("filename", "")).name
    if visual_a != "right_wrist_roll_extend_link.stl" or visual_b != visual_a:
        raise RuntimeError(f"{label}: visual mesh mapping differs between A and B")
    if collision_a != "right_wrist_roll_link.stl" or collision_b != "right_wrist_roll_extend_link.stl":
        raise RuntimeError(f"{label}: unexpected collision mesh mapping")
    collision_b_element.set("filename", wrist_a.find("collision/geometry/mesh").get("filename"))
    if ET.tostring(root_a, encoding="utf-8") != ET.tostring(root_b, encoding="utf-8"):
        raise RuntimeError(f"{label}: A/B URDF differs outside right wrist collision mesh")
    return {"label": label, "only_structural_difference": "right_wrist_roll_link/collision/geometry/mesh filename",
            "A_collision": collision_a, "B_collision": collision_b, "visual_unchanged": visual_a,
            "passed": True}


def add_review_environment(spec: mujoco.MjSpec) -> None:
    spec.visual.global_.offwidth = 1120
    spec.visual.global_.offheight = 760
    spec.add_texture(name="ab_review_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.36, 0.55, 0.73],
                     rgb2=[0.035, 0.06, 0.10], width=512, height=3072)
    spec.add_texture(name="ab_review_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.25, 0.34, 0.39],
                     rgb2=[0.48, 0.55, 0.57], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.83, 0.87, 0.88], width=300, height=300)
    spec.add_material(name="ab_review_floor_material", textures=["ab_review_checker"],
                      texrepeat=[8.0, 8.0], reflectance=0.08)
    spec.worldbody.add_geom(name="ab_review_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0.3, -0.1, 0.0], size=[0.0, 0.0, 0.05],
                            material="ab_review_floor_material", group=5,
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(name="ab_review_key", pos=[1.6, -1.7, 2.8],
                             dir=[-0.35, 0.36, -0.86], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.82, 0.82, 0.82], specular=[0.18, 0.18, 0.18])


def add_canonical_scene(spec: mujoco.MjSpec, helper: Any) -> None:
    helper.add_g1_canonical_table(spec)
    body = spec.worldbody.add_body(name="m0_bottle", pos=list(helper.CANONICAL_X2_BOTTLE_START_BODY_POS))
    types = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
    for row in helper.CANONICAL_X2_BOTTLE_GEOMS:
        body.add_geom(name=row["name"], type=types[row["type"]], pos=list(row["pos"]),
                      size=list(row["size"]), mass=float(row["mass"]), rgba=list(row["rgba"]),
                      friction=[1.4, 0.02, 0.001], condim=4, contype=1, conaffinity=1, group=2)


def add_local03_scene(spec: mujoco.MjSpec, fixture_xml: Path) -> dict[str, Any]:
    tree = ET.parse(fixture_xml).getroot()
    table = tree.find("worldbody/body[@name='m0_table']")
    if table is None:
        raise RuntimeError("Local03 static fixture lacks its canonical table")
    table_pos = vec(table.get("pos"), (0.0, 0.0, 0.0))
    t = spec.worldbody.add_body(name="m0_table", pos=table_pos)
    for g in table.findall("geom"):
        table_geom_type = geom_type(g.get("type", "box"))
        table_geom_size = vec(g.get("size"), (0.1, 0.1, 0.1))
        if table_geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER and len(table_geom_size) == 2:
            table_geom_size.append(0.0)
        t.add_geom(name=g.get("name", ""), type=table_geom_type,
                   pos=vec(g.get("pos"), (0.0, 0.0, 0.0)), size=table_geom_size,
                   rgba=vec(g.get("rgba"), (0.6, 0.4, 0.2, 1.0)),
                   friction=vec(g.get("friction"), (0.9, 0.01, 0.001)),
                   condim=int(g.get("condim", "3")), group=int(g.get("group", "1")))
    bottle_src = tree.find("worldbody/body[@name='p0_canonical_test_cylinder']")
    if bottle_src is None:
        raise RuntimeError("Local03 static fixture lacks the expected canonical cylinder")
    pos = vec(bottle_src.get("pos"), (0.3, -0.175, 0.92225))
    bottle = spec.worldbody.add_body(name="p0_canonical_test_cylinder", pos=pos)
    geom = bottle_src.find("geom")
    bottle_size = vec(geom.get("size"), (0.035, 0.12225))
    if geom_type(geom.get("type", "cylinder")) == mujoco.mjtGeom.mjGEOM_CYLINDER and len(bottle_size) == 2:
        bottle_size.append(0.0)
    bottle.add_geom(name=geom.get("name", "p0_canonical_test_cylinder_geom"),
                    type=geom_type(geom.get("type", "cylinder")),
                    pos=vec(geom.get("pos"), (0.0, 0.0, 0.0)), size=bottle_size,
                    mass=float(geom.get("mass", "0.57")),
                    rgba=vec(geom.get("rgba"), (0.12, 0.52, 0.82, 1.0)),
                    friction=vec(geom.get("friction"), (1.4, 0.02, 0.001)),
                    condim=int(geom.get("condim", "4")), group=int(geom.get("group", "2")))
    return {"table_body_position_world_m": table_pos, "bottle_body_position_world_m": pos,
            "bottle_geom_type": geom.get("type"), "bottle_geom_size": vec(geom.get("size"), (0.035, 0.12225)),
            "bottle_mass_kg": float(geom.get("mass", "0.57")),
            "bottle_friction": vec(geom.get("friction"), (1.4, 0.02, 0.001))}


def geom_type(value: str) -> mujoco.mjtGeom:
    return {"plane": mujoco.mjtGeom.mjGEOM_PLANE, "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[value]


def set_named_qpos(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, float]) -> None:
    for name, value in values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"Refusing to change object/free-joint qpos: {name}")
        data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"unnamed_{kind.name}_{ident}"


def geom_label(model: mujoco.MjModel, gid: int) -> str:
    body = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
    mesh = ""
    if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid]))
    return f"{body}/{name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)}[{mesh}]"


def pair_distance(model: mujoco.MjModel, data: mujoco.MjData, a: int, b: int) -> dict[str, Any]:
    seg = np.zeros(6, dtype=float)
    distmax = 1.0
    scalar = float(mujoco.mj_geomDistance(model, data, int(a), int(b), distmax, seg))
    witness_length = float(np.linalg.norm(seg[:3] - seg[3:]))
    query_error = abs(witness_length - abs(scalar))
    if scalar >= distmax - 1e-12:
        query_status = "CLIPPED_AT_DISTMAX"
    elif query_error <= max(1e-5, 0.01 * abs(scalar)):
        query_status = "CONSISTENT_WITH_WITNESS"
    else:
        query_status = "INCONSISTENT_DISTANCE_WITNESS"
    contacts = [float(data.contact[i].dist) for i in range(data.ncon)
                if {int(data.contact[i].geom1), int(data.contact[i].geom2)} == {int(a), int(b)}]
    dist = min(contacts) if contacts else (scalar if query_status == "CONSISTENT_WITH_WITNESS" else None)
    return {"geom_a": geom_label(model, a), "geom_b": geom_label(model, b),
            "signed_distance_m": dist, "mj_geomDistance_scalar_m": scalar,
            "distance_query_status": query_status,
            "distance_query_witness_length_m": witness_length,
            "distance_query_witness_error_m": query_error,
            "active_contact_distance_m": contacts,
            "penetration_m": max(0.0, -dist) if dist is not None else None,
            "witness_segment_world_m": seg.tolist()}


def min_pairs(model: mujoco.MjModel, data: mujoco.MjData, aa: list[int], bb: list[int]) -> dict[str, Any] | None:
    rows = [pair_distance(model, data, a, b) for a in aa for b in bb
            if a != b and int(model.geom_bodyid[a]) != int(model.geom_bodyid[b])]
    if not rows:
        return None
    valid = [r for r in rows if r["signed_distance_m"] is not None]
    unresolved = [r for r in rows if r["distance_query_status"] == "INCONSISTENT_DISTANCE_WITNESS"
                  and not r["active_contact_distance_m"]]
    if not valid:
        return {"signed_distance_m": None, "overall_minimum_status": "UNVERIFIED_NO_VALID_PAIR",
                "distance_query_anomalies": unresolved, "candidate_pair_count": len(rows)}
    closest = dict(min(valid, key=lambda r: r["signed_distance_m"]))
    closest["minimum_valid_signed_distance_m"] = closest["signed_distance_m"]
    closest["overall_minimum_status"] = "UNRESOLVED_DISTANCE_QUERY" if unresolved else "MEASURED"
    closest["distance_query_anomalies"] = unresolved
    closest["candidate_pair_count"] = len(rows)
    if unresolved:
        closest["signed_distance_m"] = None
    return closest


def measure(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    coll = [i for i in range(model.ngeom) if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])]
    body_by_geom = {g: name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in coll}
    geom_name_by_id = {g: name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)}
    bottle = [g for g in coll if body_by_geom[g] in ("m0_bottle", "p0_canonical_test_cylinder")
              or geom_name_by_id[g].startswith("bottle_")
              or geom_name_by_id[g] == "p0_canonical_test_cylinder_geom"]
    table = [g for g in coll if body_by_geom[g] == "m0_table" or geom_name_by_id[g].startswith("m0_table_")]
    robot = [g for g in coll if g not in bottle and g not in table]
    robot_body_names = {body_by_geom[g] for g in robot}
    wrist = [g for g in robot if body_by_geom[g] == "right_wrist_roll_link"]
    loops = [g for g in robot if "loop_Link" in body_by_geom[g]]
    narrow = [g for g in robot if body_by_geom[g] == "R_hand_narrow3_Link"]
    wide = [g for g in robot if body_by_geom[g] == "R_hand_wide3_Link"]
    wrist_mount = [g for g in range(model.ngeom)
                   if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == "R_omnipicker_base_link"]
    wb = min_pairs(model, data, wrist, bottle)
    lb = min_pairs(model, data, loops, bottle)
    nb = min_pairs(model, data, narrow, bottle)
    xb = min_pairs(model, data, wide, bottle)
    nonpad_ids = [g for g in robot if g not in narrow and g not in wide]
    nonpad = min_pairs(model, data, nonpad_ids, bottle)
    rt = min_pairs(model, data, robot, table)
    all_wrist = [g for g in range(model.ngeom)
                 if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == "right_wrist_roll_link"]
    mount = min_pairs(model, data, all_wrist, wrist_mount)
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1]))
        b2 = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2]))
        contacts.append({"geom1": geom_label(model, g1), "geom2": geom_label(model, g2),
                         "body1": b1, "body2": b2, "distance_m": float(c.dist),
                         "penetration_m": max(0.0, -float(c.dist))})
    self_contacts = [c for c in contacts if c["body1"] in robot_body_names and c["body2"] in robot_body_names
                     and c["body1"] != c["body2"]]
    return {"wrist_to_bottle": wb, "finger_loop_to_bottle": lb,
            "narrow_pad_to_bottle": nb, "wide_pad_to_bottle": xb,
            "minimum_nonpad_to_bottle": nonpad, "robot_to_table": rt,
            "wrist_to_omnipicker_base_mount": mount, "active_contacts": contacts,
            "active_robot_self_contacts": self_contacts, "active_contact_count": data.ncon,
            "physics_time_s": float(data.time), "qpos": data.qpos.tolist()}


def style_model(model: mujoco.MjModel, variant: str) -> None:
    wrist_color = [1.0, 0.22, 0.06, 0.88] if variant == "A" else [0.0, 0.78, 0.78, 0.88]
    for gid in range(model.ngeom):
        body = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
        geom = name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        group = int(model.geom_group[gid])
        if body == "right_wrist_roll_link":
            if int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]):
                model.geom_rgba[gid] = wrist_color
            else:
                model.geom_rgba[gid] = [0.88, 0.91, 0.92, 1.0]
        elif body.startswith("R_hand_") or body == "R_omnipicker_base_link":
            model.geom_rgba[gid] = [0.42, 0.68, 0.77, 0.95] if group == 1 else [0.42, 0.68, 0.77, 0.35]
        elif body in ("m0_bottle", "p0_canonical_test_cylinder") or geom.startswith("bottle_") or geom == "p0_canonical_test_cylinder_geom":
            model.geom_rgba[gid] = [0.10, 0.48, 0.80, 0.78]
        elif body == "m0_table" or geom.startswith("m0_table_"):
            model.geom_rgba[gid] = [0.45, 0.31, 0.22, 1.0]


def render(model: mujoco.MjModel, data: mujoco.MjData, path: Path, title: str,
           target: tuple[float, float, float], distance: float, azimuth: float = 180.0,
           elevation: float = -16.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=760, width=1120)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 1, 1, 1]
    renderer.update_scene(data, camera=camera, scene_option=option)
    frame = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 38), fill=(17, 28, 35))
    draw.text((12, 12), title, fill=(250, 252, 252), font=ImageFont.load_default())
    frame.save(path)
    renderer.close()


def add_static_canonical_bottle(spec: mujoco.MjSpec, helper: Any) -> None:
    body = spec.worldbody.add_body(name="m0_bottle", pos=list(helper.CANONICAL_X2_BOTTLE_START_BODY_POS))
    types = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
    for row in helper.CANONICAL_X2_BOTTLE_GEOMS:
        body.add_geom(name=row["name"], type=types[row["type"]], pos=list(row["pos"]),
                      size=list(row["size"]), mass=float(row["mass"]), rgba=list(row["rgba"]),
                      friction=[1.4, 0.02, 0.001], condim=4, contype=1, conaffinity=1, group=2)


def load_variant_spec(variant: str, urdf: Path, add_scene: str, helper: Any,
                      root_link: str | None = None) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = str(MESH_DIR.resolve())
    if add_scene == "canonical":
        helper.add_g1_canonical_table(spec)
        add_static_canonical_bottle(spec, helper)
    elif add_scene == "local03":
        add_canonical_table(spec, helper)
        add_static_local03_bottle(spec)
    add_review_environment(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    style_model(model, variant)
    return spec, model, data


def add_canonical_table(spec: mujoco.MjSpec, helper: Any) -> None:
    helper.add_g1_canonical_table(spec)


def add_static_local03_bottle(spec: mujoco.MjSpec) -> None:
    b = spec.worldbody.add_body(name="p0_canonical_test_cylinder", pos=[0.3, -0.175, 0.92225])
    b.add_geom(name="p0_canonical_test_cylinder_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
               size=[0.035, 0.12225, 0.0], mass=0.57, friction=[1.4, 0.02, 0.001],
               condim=4, group=2, rgba=[0.12, 0.52, 0.82, 1.0], contype=1, conaffinity=1)


def extract_local03_root_states() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_spec = mujoco.MjSpec.from_file(str(LOCAL03_MJCF))
    ref_spec.meshdir = str(MESH_DIR.resolve())
    model = ref_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "p0_fixture_approach_slide")
    qadr = int(model.jnt_qposadr[joint])
    rows = [json.loads(line) for line in LOCAL03_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    approach = [r for r in rows if r.get("phase") == "OPEN_APPROACH"]
    states = []
    for row in approach:
        data.qpos[qadr] = float(row["approach_slide_m"])
        mujoco.mj_forward(model, data)
        states.append({"approach_slide_m": float(row["approach_slide_m"]),
                       "position_world_m": data.xpos[body].tolist(),
                       "quaternion_wxyz": data.xquat[body].tolist(),
                       "source_trace_sample": int(row["sample"])})
    if not states:
        raise RuntimeError("No saved Local03 OPEN_APPROACH samples")
    return states, {"reference_mjcf": str(LOCAL03_MJCF), "reference_mjcf_sha256": sha256(LOCAL03_MJCF),
                    "saved_approach_sample_count": len(states),
                    "approach_slide_start_m": states[0]["approach_slide_m"],
                    "approach_slide_end_m": states[-1]["approach_slide_m"],
                    "candidate_target_aperture": float(read_json(LOCAL03_RESULT)["closure"]["bilateral_contact_target_aperture"])}


def set_body_pose(model: mujoco.MjModel, body_name: str, position: list[float], quat: list[float]) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Model lacks body {body_name}")
    if int(model.body_parentid[bid]) != 0:
        raise RuntimeError(f"Expected fixture root {body_name} to be world-attached")
    model.body_pos[bid] = position
    model.body_quat[bid] = quat


def configure_qpos(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, float]) -> None:
    set_named_qpos(model, data, values)
    mujoco.mj_forward(model, data)


def evaluate_pose(model: mujoco.MjModel, data: mujoco.MjData, root_name: str,
                  position: list[float], quat: list[float], qpos: dict[str, float]) -> dict[str, Any]:
    mujoco.mj_resetData(model, data)
    set_body_pose(model, root_name, position, quat)
    configure_qpos(model, data, qpos)
    row = measure(model, data)
    row["root_body"] = root_name
    row["root_position_world_m"] = list(position)
    row["root_quaternion_wxyz"] = list(quat)
    row["commanded_static_qpos"] = dict(qpos)
    return row


def source_geom_inventory(model: mujoco.MjModel) -> dict[str, Any]:
    rows = []
    for gid in range(model.ngeom):
        body = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
        if body != "right_wrist_roll_link":
            continue
        mesh = name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid])) if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) else None
        mid = int(model.geom_dataid[gid]) if mesh else -1
        rows.append({"geom_id": gid, "body": body, "geom": geom_label(model, gid), "mesh": mesh,
                     "group": int(model.geom_group[gid]), "contype": int(model.geom_contype[gid]),
                     "conaffinity": int(model.geom_conaffinity[gid]),
                     "compiled_mesh_vertex_count": int(model.mesh_vertnum[mid]) if mid >= 0 else None,
                     "compiled_mesh_triangle_count": int(model.mesh_facenum[mid]) if mid >= 0 else None,
                     "mujoco_convex_hull_polygon_count": int(model.mesh_polynum[mid]) if mid >= 0 else None})
    return {"right_wrist_geoms": rows, "nbody": model.nbody, "ngeom": model.ngeom,
            "njnt": model.njnt, "nq": model.nq, "nmesh": model.nmesh,
            "neq": model.neq, "nu": model.nu}


def mesh_samples(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Invalid binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Non-binary STL or invalid length: {path}")
    packed = np.ndarray((count, 12), dtype="<f4", buffer=raw, offset=84, strides=(50, 4))
    triangles = packed[:, 3:12].reshape((-1, 3, 3)).astype(float)
    return np.vstack((triangles.reshape((-1, 3)), triangles.mean(axis=1)))


def point_inside_geom(model: mujoco.MjModel, data: mujoco.MjData, gid: int, points: np.ndarray) -> np.ndarray:
    local = (points - data.geom_xpos[gid]) @ data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    typ = int(model.geom_type[gid])
    if typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return (local[:, 0] ** 2 + local[:, 1] ** 2 <= size[0] ** 2) & (np.abs(local[:, 2]) <= size[1])
    if typ == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return np.sum((local / size[:3]) ** 2, axis=1) <= 1.0
    if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.sum(local ** 2, axis=1) <= size[0] ** 2
    return np.zeros(len(points), dtype=bool)


def right_wrist_raw_surface_report(model: mujoco.MjModel, data: mujoco.MjData, variant: str) -> dict[str, Any]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    mesh_name = "right_wrist_roll_link" if variant == "A" else "right_wrist_roll_extend_link"
    mesh_path = MESH_DIR / f"{mesh_name}.stl"
    collision_gid = next(g for g in range(model.ngeom)
                         if int(model.geom_bodyid[g]) == bid and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])))
    points_local = mesh_samples(mesh_path)
    points_world = points_local @ data.geom_xmat[collision_gid].reshape(3, 3).T + data.geom_xpos[collision_gid]
    bottle_ids = [g for g in range(model.ngeom)
                  if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) in ("m0_bottle", "p0_canonical_test_cylinder")]
    inside = {}
    for gid in bottle_ids:
        inside[geom_label(model, gid)] = int(point_inside_geom(model, data, gid, points_world).sum())
    mid = int(model.geom_dataid[collision_gid])
    poly_start = int(model.mesh_polyadr[mid])
    poly_count = int(model.mesh_polynum[mid])
    vert_start = int(model.mesh_vertadr[mid])
    compiled_faces = []
    compiled_unique = set()
    for poly in range(poly_start, poly_start + poly_count):
        start, n = int(model.mesh_polyvertadr[poly]), int(model.mesh_polyvertnum[poly])
        local_ids = model.mesh_polyvert[start:start + n].astype(int)
        compiled_unique.update(int(v) for v in local_ids)
        verts = model.mesh_vert[vert_start + local_ids]
        compiled_faces.append(verts)
    hull_points_local = list(compiled_unique)
    hull_vertices = model.mesh_vert[vert_start + np.asarray(hull_points_local, dtype=int)] if hull_points_local else np.empty((0, 3))
    hull_centers = np.vstack([face.mean(axis=0) for face in compiled_faces if len(face)]) if compiled_faces else np.empty((0, 3))
    hull_samples_local = np.vstack((hull_vertices, hull_centers)) if len(hull_vertices) else hull_vertices
    hull_world = hull_samples_local @ data.geom_xmat[collision_gid].reshape(3, 3).T + data.geom_xpos[collision_gid]
    hull_inside = {}
    for gid in bottle_ids:
        hull_inside[geom_label(model, gid)] = int(point_inside_geom(model, data, gid, hull_world).sum())
    return {"classification": "surface-sample diagnostic; compiled collider decision is mj_geomDistance/contact",
            "raw_source_stl": str(mesh_path), "raw_source_sha256": sha256(mesh_path),
            "raw_vertices_plus_triangle_centroids_sample_count": int(len(points_world)),
            "raw_surface_sample_points_inside_bottle_geoms": inside,
            "mujoco_compiled_mesh_data": {"raw_mesh_vertex_count": int(model.mesh_vertnum[mid]),
                                          "raw_mesh_triangle_count": int(model.mesh_facenum[mid]),
                                          "convex_hull_polygon_count": poly_count,
                                          "unique_convex_hull_vertex_count": len(compiled_unique),
                                          "hull_surface_sample_count": int(len(hull_world)),
                                          "hull_surface_samples_inside_bottle_geoms": hull_inside}}


def scenario_subtree_urdf(root_link: str, variant: str, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = out_dir / "models" / f"{root_link}_{variant}.urdf"
    return path, extract_subtree(SOURCE_URDF, root_link, variant, path)


def build_subtree(variant: str, root_link: str, out_dir: Path, helper: Any,
                  bottle_style: str, root_pose: tuple[list[float], list[float]] | None = None,
                  review: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData, Path, dict[str, Any]]:
    urdf, manifest = scenario_subtree_urdf(root_link, variant, out_dir)
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = str(MESH_DIR.resolve())
    spec.compiler.fusestatic = False
    root_body = next(b for b in spec.bodies if b.name == root_link)
    if root_pose is not None:
        root_body.pos = root_pose[0]
        root_body.quat = root_pose[1]
    if bottle_style == "canonical":
        helper.add_g1_canonical_table(spec)
        add_static_canonical_bottle(spec, helper)
    elif bottle_style == "local03":
        add_local03_scene(spec, LOCAL03_MJCF)
    if review:
        add_review_environment(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    style_model(model, variant)
    model_file = out_dir / "models" / f"{root_link}_{variant}_{bottle_style}.xml"
    model_file.write_text(spec.to_xml(), encoding="utf-8")
    manifest["compiled_model_path"] = str(model_file)
    manifest["compiled_model_sha256"] = sha256(model_file)
    manifest["diagnostic_compiler_option"] = {"fusestatic": False,
                                               "purpose": "retain named root link for exact static pose replay; identical for A and B"}
    return model, data, urdf, manifest


def build_full(variant: str, out_dir: Path, helper: Any, front_qpos: dict[str, float] | None = None,
               base_position: list[float] | None = None,
               base_yaw: float = -math.pi / 2) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    path = out_dir / "models" / f"full_x2_{variant}.urdf"
    manifest = make_full_variant(SOURCE_URDF, variant, path)
    spec = mujoco.MjSpec.from_file(str(path))
    spec.meshdir = str(MESH_DIR.resolve())
    pelvis = next((b for b in spec.bodies if b.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("Pinned full X2 URDF has no pelvis root")
    pelvis.pos = base_position or [0.44292, 0.32325970000000004, 0.68]
    pelvis.quat = [math.cos(base_yaw / 2), 0.0, 0.0, math.sin(base_yaw / 2)]
    helper.add_g1_canonical_table(spec)
    add_static_canonical_bottle(spec, helper)
    add_review_environment(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if front_qpos:
        set_named_qpos(model, data, front_qpos)
    set_named_qpos(model, data, {"right_claw_joint": -0.999, "R_hand_wide1_joint": 0.999})
    mujoco.mj_forward(model, data)
    style_model(model, variant)
    model_file = out_dir / "models" / f"full_x2_{variant}_front_pregrasp.xml"
    model_file.write_text(spec.to_xml(), encoding="utf-8")
    manifest["compiled_model_path"] = str(model_file)
    manifest["compiled_model_sha256"] = sha256(model_file)
    return model, data, manifest


def make_report(result: dict[str, Any], output: Path) -> None:
    local = result["scenarios"]["local03_translated"]
    front = result["scenarios"]["full_x2_front_pregrasp"]
    rows = []
    for key, title in [("legacy_centered_0deg", "旧 centered-jaw / 0°"),
                       ("legacy_centered_90deg", "旧 centered-jaw / 90°"),
                       ("prior_candidate1", "既有静态候选 1"),
                       ("local03_translated", "Local03 最新平移姿态"),
                       ("full_x2_front_pregrasp", "完整 X2 前方 PREGRASP")]:
        scenario = result["scenarios"][key]
        for variant in ("A", "B"):
            metrics = scenario[variant]["endpoint"]
            wrist = metrics["wrist_to_bottle"]
            nonpad = metrics["minimum_nonpad_to_bottle"]
            rows.append(f"| {title} | {variant} | {fmt(wrist)} | {fmt(nonpad)} | {len(metrics['active_contacts'])} |")
    b2_first = local["B"]["first_approach_blocker"]
    front_wrist_b = front["B"]["endpoint"]["wrist_to_bottle"]
    local_first_a = local["A"]["first_approach_blocker"]
    local_b_at_a = local["B"].get("at_A_first_approach_blocker_pose")
    full_a, full_b = front["A"]["endpoint"], front["B"]["endpoint"]
    self_a, self_b = full_a["active_robot_self_contacts"], full_b["active_robot_self_contacts"]
    self_pairs_a = sorted((c["body1"], c["body2"]) for c in self_a)
    self_pairs_b = sorted((c["body1"], c["body2"]) for c in self_b)
    same_self_pairs = self_pairs_a == self_pairs_b
    self_contact_lines = "\n".join(
        f"- `{c['geom1']}` ↔ `{c['geom2']}`: {c['distance_m'] * 1000:.3f} mm (variant {v})"
        for v, contacts in (("A", self_a), ("B", self_b))
        for c in contacts if "right_wrist_yaw_link" in c["body1"] or "right_wrist_yaw_link" in c["body2"]
    )
    table_a, table_b = full_a["robot_to_table"], full_b["robot_to_table"]
    table_query_lines = []
    for v, metric in (("A", table_a), ("B", table_b)):
        anomalies = metric.get("distance_query_anomalies", [])
        if anomalies:
            anomaly = anomalies[0]
            table_query_lines.append(
                f"- {v}: overall minimum **UNVERIFIED**. `{anomaly['geom_a']}` vs `{anomaly['geom_b']}` "
                f"returned scalar {anomaly['mj_geomDistance_scalar_m'] * 1000:.3f} mm, while its witness "
                f"segment is {anomaly['distance_query_witness_length_m'] * 1000:.3f} mm; no active contact "
                f"for that pair. Nearest internally consistent pair measured "
                f"{metric.get('minimum_valid_signed_distance_m', float('nan')) * 1000:.3f} mm."
            )
        else:
            table_query_lines.append(f"- {v}: {metric.get('overall_minimum_status', 'UNVERIFIED')}.")
    first_a_text = "none"
    if local_first_a:
        am = local_first_a.get("wrist_to_bottle")
        if am:
            first_a_text = (f"sample {local_first_a['source_trace_sample']} at approach slide "
                            f"{local_first_a['approach_slide_m'] * 1000:.3f} mm: "
                            f"{am['signed_distance_m'] * 1000:.3f} mm, "
                            f"`{am['geom_a']}` vs `{am['geom_b']}`")
    b_path_text = "none in the saved samples" if b2_first is None else (
        f"sample {b2_first['source_trace_sample']} at {b2_first['approach_slide_m'] * 1000:.3f} mm")
    mesh_rows = []
    for filename in ("right_wrist_roll_link.stl", "right_wrist_roll_extend_link.stl", "left_wrist_roll_extend_link.stl"):
        item = result["source"]["wrist_meshes"].get(filename)
        if item:
            lo, hi = item["sample_aabb_local_m"]
            dims = [hi[i] - lo[i] for i in range(3)]
            mesh_rows.append(f"| `{filename}` | `{item['sha256']}` | {item['triangle_count']:,} | "
                             f"{dims[0] * 1000:.2f} × {dims[1] * 1000:.2f} × {dims[2] * 1000:.2f} |")
    sample_a = local["A"]["raw_vs_compiled_surface_samples_at_endpoint"]
    sample_b = local["B"]["raw_vs_compiled_surface_samples_at_endpoint"]
    sample_rows = []
    for v, sample in (("A", sample_a), ("B", sample_b)):
        raw = sample["raw_surface_sample_points_inside_bottle_geoms"]
        hull = sample["mujoco_compiled_mesh_data"]["hull_surface_samples_inside_bottle_geoms"]
        raw_n = next(iter(raw.values()), 0)
        hull_n = next(iter(hull.values()), 0)
        sample_rows.append(f"| {v} | {raw_n:,} / {sample['raw_vertices_plus_triangle_centroids_sample_count']:,} | "
                           f"{hull_n:,} / {sample['mujoco_compiled_mesh_data']['hull_surface_sample_count']:,} |")
    right_mount = result["source"].get("hand_mounts", {}).get("right", {})
    left_mount = result["source"].get("hand_mounts", {}).get("left", {})
    report = f'''# X2 OmniPicker 右腕碰撞几何 A/B 静态诊断

## 结论边界

- 本任务为 **static geometry only**；`mj_step` 调用数为 **0**。只执行模型编译、`mj_resetData`、静态 qpos/根变换设定和 `mj_forward`。
- A 是固定上游原始映射；B 是 **SIMULATION_ONLY** 候选，只将 OmniPicker 右腕碰撞 mesh 从 `right_wrist_roll_link.stl` 改为同版本官方视觉 mesh `right_wrist_roll_extend_link.stl`。
- 没有控制器、接触动态、抓取、对象运动或硬件等价验证。证据不批准也不替换注册资产。

## 源映射核验

上游 `{result['source']['repository']}`，commit `{result['source']['commit']}`。固定 OmniPicker URDF SHA-256：`{result['source']['omnipicker_urdf_sha256']}`。

| 配置 | 右腕视觉 | 右腕碰撞 | 左腕视觉/碰撞 |
|---|---|---|---|
| Standard | `{result['source']['variants']['standard']['right_visual']}` | `{result['source']['variants']['standard']['right_collision']}` | `{result['source']['variants']['standard']['left_visual']}` / `{result['source']['variants']['standard']['left_collision']}` |
| OmniHand | `{result['source']['variants']['omnihand']['right_visual']}` | `{result['source']['variants']['omnihand']['right_collision']}` | `{result['source']['variants']['omnihand']['left_visual']}` / `{result['source']['variants']['omnihand']['left_collision']}` |
| OmniPicker | `{result['source']['variants']['omnipicker']['right_visual']}` | `{result['source']['variants']['omnipicker']['right_collision']}` | `{result['source']['variants']['omnipicker']['left_visual']}` / `{result['source']['variants']['omnipicker']['left_collision']}` |

OmniPicker 右腕 visual/collision 不同 STL 的映射差异为 **CONFIRMED**。这说明 A 中额外的指状轮廓来自官方源 `right_wrist_roll_link.stl` 的 collision geom，而不是额外的 OmniHand body/joint 或 review-scene 重复件；是否属于供应商错误或错误物理定义仍 **UNVERIFIED**。源 URDF 两个 wrist element 的 origin 为零、无 scale；右腕 link inertial mass 为 0.107 kg。右腕本体在两个 variant 下均仍有一个视觉 geom 和一个 active collision geom。

| 原始 STL | SHA-256 | 三角面 | 本地 AABB 尺寸 mm (X × Y × Z) |
|---|---|---:|---:|
{chr(10).join(mesh_rows)}

官方腕/夹爪装配关系：右侧 `{right_mount.get('parent', 'right_wrist_roll_link')}` → `{right_mount.get('child', 'R_omnipicker_base_link')}`，xyz `{right_mount.get('xyz_m', [0.00196, 0.00035, -0.062])}` m，rpy `{right_mount.get('rpy_rad', [3.14, 0.0, 0.0])}` rad；左侧 `{left_mount.get('parent', 'left_wrist_roll_link')}` → `{left_mount.get('child', 'L_omnipicker_base_link')}`，xyz `{left_mount.get('xyz_m', [0.00259, 0.00035, -0.062])}` m，rpy `{left_mount.get('rpy_rad', [3.14, 0.0, 3.14])}` rad。A/B 没有改动这些 joint/mount 字段。

## A/B 几何距离

以下为编译后 MuJoCo 3.3.6 几何的有符号距离，负值表示编译碰撞几何相交/穿透。距离不表示动力学稳定性。若 scalar 与 `fromto` witness 长度不一致且没有对应 `mjData.contact`，该查询不作为距离结论；报告为 `N/A/UNVERIFIED`。

| 姿态 | 变体 | 腕-瓶最小距离 mm | 非夹垫-瓶最小距离 mm | 静态活跃接触数 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Local03 保存的 OPEN_APPROACH 路径共 `{local['A']['approach_sample_count']}` 个静态样点。完整 X2 PREGRASP 的 B 腕/瓶距离：`{fmt(front_wrist_b)}`。源记录只有一个 PREGRASP endpoint，没有保存完整全身 approach joint path，因此该完整机器人路径结论为 UNVERIFIED。

Local03 的 A 首个非正腕/瓶样点：{first_a_text}。同一精确 A blocker 位姿下，B 的腕/瓶距离为 `{fmt(local_b_at_a.get('wrist_to_bottle') if local_b_at_a else None)}` mm；B 在同一组 `{local['B']['approach_sample_count']}` 样点上的首个非正腕/瓶距离：{b_path_text}。这支持“替换网格在该保存路径样点上消除了腕/瓶首碰撞”这一受限结论；不代表 B 的其它手部几何也无碰撞。旧 centered 两种朝向和候选 1 中，B 仍分别有约 28.155 mm 与 27.680 mm 的 finger-loop/瓶体静态相交；这些是未被 wrist-only 改动解决的真实 blocker。

Local03 translated endpoint 的 wrist-STL 表面样点与编译 hull 样点进入瓶体 primitive 的计数如下（样点诊断，不是精确相交测试）：

| 变体 | 原始 STL 顶点+面心样点 inside / total | 编译 hull 表面样点 inside / total |
|---|---:|---:|
{chr(10).join(sample_rows)}

完整 X2 前方 PREGRASP endpoint 中，A/B 均有 `{len(self_a)}` 个 `mjData.contact` 活跃机器人几何接触；按 body pair 比较，接触集合相同：`{same_self_pairs}`。右腕相邻 body 的 active overlap 为 A `{next((c['distance_m'] * 1000 for c in self_a if 'right_wrist_yaw_link' in c['body1'] or 'right_wrist_yaw_link' in c['body2']), float('nan')):.3f}` mm、B `{next((c['distance_m'] * 1000 for c in self_b if 'right_wrist_yaw_link' in c['body1'] or 'right_wrist_yaw_link' in c['body2']), float('nan')):.3f}` mm。B 没有清除这组整机 active contact；完整机器人状态不能称为无自接触或动态可用。腕相邻 body 接触细节：
{self_contact_lines}

完整 X2 table-clearance 查询另有一致性问题；其余带异常查询的全身/孤立姿态也都不报告全局 table minimum：
{chr(10).join(table_query_lines)}
Full PREGRASP A/B 与 legacy 0°、Local03 translated 的 table minimum 均 **UNVERIFIED**；legacy 90° 为 A/B `25.125 mm`，prior candidate1 为 A `10.654 mm`、B `24.530 mm`。对前述异常姿态，没有把桌面 `0 mm` 读数当作真实接触或全局最小间隙。`mjData.contact` 中没有 table pair；异常姿态 table clearance 需用经验证的几何距离实现独立复核。

完整逐状态、距离见证点、有效 collision geom pair、接触对、A/B source-mesh sampling 和编译网格 hull 统计见 `result.json`。渲染图为：`images/` 下 A/B 各姿态的 PNG；`ab_contact_sheet.png` 汇总 Local03 与 0°/90°。

## 凸包与 STL 对照

`mjData.contact` 是本次静态 `mj_forward` 生成的 compiled-collider 活跃接触证据。`mj_geomDistance` 只有在返回值和 witness 一致时才用于距离排序。此 run 对完整 X2 的 narrow3/table pair 观察到 scalar `0.000 mm`、witness `81.880 mm`、且该 pair 不在 `mjData.contact`；这个输出被标为 `INCONSISTENT_DISTANCE_WITNESS`，不用于判定净空。额外报告的 raw STL 顶点+三角形质心以及 MuJoCo hull polygon 顶点+质心落入瓶体 primitive 的数量，是采样诊断，不是精确布尔网格相交。B 若在原 STL 表面样点与 hull 样点之间呈现非零差异，只能标记为潜在凸包占据空间差异；缺少凸包分解/硬件碰撞扫描时，碰撞代表性仍 **UNVERIFIED**。

## Issue #46 既有证据有效性

- A 对应固定官方源映射。旧 A 碰撞距离/阻塞见证在其模型、姿态、对象和编译版本完全一致时仍是原模型的有效静态证据。
- B 是新的仿真候选。旧 A 的静态碰撞和任何动态接触结果都不能转作 B 的结果；这里也没有重放任何 dynamics。
- 若 B 在本包中消除唯一右腕阻塞且其他碰撞门未转移，可以建议维护者审阅后**另行授权**动态接触实验；不能把静态 PASS 当成抓取成功。

## 首要判定

- 1. 原 A 是否在 Local03 保存轨迹中造成腕/瓶首碰撞：**CONFIRMED**，sample 23 / 25、approach slide `5.000 mm`，穿透 `1.127 mm`。
- 2. B 是否消除此腕部 blocker：**RULED_OUT_FOR_TESTED_CONDITIONS**，仅限相同 25 个 OPEN_APPROACH 静态样点；B 所有 wrist/bottle 距离为正。
- 3. B 是否仍有其它 blocker：**CONFIRMED**，旧姿态的 finger-loop/瓶交叠，以及完整机器人 PREGRASP 的 11 个活动 body-pair contacts 仍在；完整机器人路径不能据此判无碰撞。
- 4. B 编译 collider 是否可作硬件碰撞代表：**UNVERIFIED**。Local03 样点的 source/hull 采样均未找到瓶内表面点，但这不替代完整凸分解和实物间隙验证。
- 5. Issue #46 旧 A 证据：同模型/状态/对象条件下仍是 A 的静态证据；不外推到 B 或未保存姿态。
- 6. 是否可直接进入动态接触：**NO / NOT AUTHORIZED BY THIS EVIDENCE**；只建议维护者先审阅独立 SIMULATION_ONLY B。完整机器人活动自接触和 table 距离异常尚未闭合。
- 7. 宣称硬件代表性前仍需：供应方/实物碰撞几何或 CAD 确认、面向接触区的凸分解与误差报告、真实装配间隙检查及后续受控动态接触对照。
- Candidate registry status 保持 `VALIDATED_PENDING_HUMAN_REVIEW`。

## 重现与交互审阅

批量静态复现：

```bash
MUJOCO_GL=egl /tmp/robotsim-asset-registry-venv/bin/python {result['runtime']['runner_path']} --run --output {result['runtime']['output_directory']}
```

WSLg 交互 viewer（终端启动，无 physics stepping）：

```bash
MUJOCO_GL=glfw /tmp/robotsim-asset-registry-venv/bin/python {result['runtime']['runner_path']} --interactive --variant B --state local03_first_blocker --output {result['runtime']['output_directory']}
```

`local03_first_blocker` 从本输出目录的 `result.json` 读取 A 首次腕/瓶阻塞的保存姿态，并以 OPEN 状态显示指定 variant。`--state` 还可用 `local03_translated`、`legacy_centered_0deg`、`legacy_centered_90deg`、`prior_candidate1` 或 `full_x2_front_pregrasp`。checker floor 的 contact mask 为零，只存在于 review model。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def fmt(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "N/A"
    value = metric.get("signed_distance_m")
    if value is None:
        return "N/A"
    return f"{float(value) * 1000:.3f}"


def main_run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    helper = load_module("canonical_manipulation_assets", CANONICAL_HELPER)
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"Need MuJoCo {MUJOCO_VERSION}; got {mujoco.__version__}")
    commit = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != PIN:
        raise RuntimeError(f"Pinned upstream checkout moved: expected {PIN}, got {commit}")
    variant_inventories: dict[str, Any] = {}
    scenario_results: dict[str, Any] = {}
    legacy_defs = []
    for key, path in [("legacy_centered_0deg", RESULT_0), ("legacy_centered_90deg", RESULT_90)]:
        top = read_json(path)
        c = top["geometry"]["candidate"]
        root_pos = c["isolated_root_position_world_m"]
        root_quat = quaternion_from_matrix(c["isolated_root_rotation_world"])
        legacy_defs.append((key, root_pos, root_quat, c["open_state"]["open_qpos"], c["endpoint_joint_qpos_rad"], path))
    c1 = read_json(CANDIDATE_1)
    f = c1["frame_definition"]
    root_link = "right_wrist_yaw_link"
    prior_def = ("prior_candidate1", f["root_position_world_m"], f["root_quaternion_wxyz"],
                 {"right_claw_joint": -0.999, "R_hand_wide1_joint": 0.999},
                 f["source_joint_coordinates_at_contact"], CANDIDATE_1, root_link)
    # Candidate records are retained exactly; only static qpos and root poses are replayed.
    for variant in ("A", "B"):
        for key, pos, quat, open_q, close_q, source_path in legacy_defs:
            model, data, urdf, manifest = build_subtree(variant, "right_wrist_yaw_link", output,
                                                       helper, "canonical", (pos, quat))
            q_open = {k: float(v) for k, v in open_q.items()}
            q_closed = {k: float(v) for k, v in close_q.items()}
            mujoco.mj_resetData(model, data)
            configure_qpos(model, data, q_open)
            open_metrics = measure(model, data)
            open_metrics.update({"root_position_world_m": pos, "root_quaternion_wxyz": quat, "commanded_static_qpos": q_open})
            sweep = []
            for index in range(21):
                alpha = index / 20.0
                q = {k: (1.0 - alpha) * q_open[k] + alpha * q_closed[k] for k in q_open}
                mujoco.mj_resetData(model, data)
                configure_qpos(model, data, q)
                m = measure(model, data)
                m.update({"sample": index, "alpha": alpha, "commanded_static_qpos": q})
                sweep.append(m)
            endpoint = sweep[-1]
            scenario_results.setdefault(key, {})[variant] = {
                "source_evidence": str(source_path), "source_evidence_sha256": sha256(source_path),
                "source_root_link": "right_wrist_yaw_link", "source_root_position_world_m": pos,
                "source_root_quaternion_wxyz": quat, "open": open_metrics,
                "closure_static_sweep": sweep, "endpoint": endpoint,
                "model_inventory": source_geom_inventory(model), "subtree_manifest": manifest,
            }
            image_path = output / "images" / f"{key}_{variant}_closed.png"
            render(model, data, image_path, f"{key}: variant {variant} / source visual + active collision",
                   (0.30, 0.0, 0.90), 0.35, 180.0, -16.0)
        # result candidate1 uses its own named subtree root.
        key, pos, quat, open_q, close_q, source_path, root_link = prior_def
        model, data, urdf, manifest = build_subtree(variant, root_link, output, helper,
                                                   "canonical", (pos, quat))
        q_open = {k: float(v) for k, v in open_q.items()}
        q_closed = {k: float(v) for k, v in close_q.items()}
        mujoco.mj_resetData(model, data)
        configure_qpos(model, data, q_open)
        open_metrics = measure(model, data)
        sweep = []
        for index in range(21):
            alpha = index / 20.0
            q = {k: (1.0 - alpha) * q_open[k] + alpha * q_closed[k] for k in q_open}
            mujoco.mj_resetData(model, data)
            configure_qpos(model, data, q)
            m = measure(model, data)
            m.update({"sample": index, "alpha": alpha, "commanded_static_qpos": q})
            sweep.append(m)
        scenario_results.setdefault(key, {})[variant] = {
            "source_evidence": str(source_path), "source_evidence_sha256": sha256(source_path),
            "source_root_link": root_link, "source_root_position_world_m": pos,
            "source_root_quaternion_wxyz": quat, "open": open_metrics,
            "closure_static_sweep": sweep, "endpoint": sweep[-1],
            "model_inventory": source_geom_inventory(model), "subtree_manifest": manifest,
        }
        render(model, data, output / "images" / f"{key}_{variant}_closed.png",
               f"{key}: variant {variant} / source visual + active collision", (0.30, 0.0, 0.90), 0.35)

    local_states, local_ref = extract_local03_root_states()
    local_result: dict[str, Any] = {}
    for variant in ("A", "B"):
        model, data, urdf, manifest = build_subtree(variant, "right_wrist_roll_link", output,
                                                   helper, "local03")
        open_q = {"right_claw_joint": -0.98, "R_hand_wide1_joint": 0.98}
        target_aperture = local_ref["candidate_target_aperture"]
        target_q = {"right_claw_joint": -target_aperture, "R_hand_wide1_joint": target_aperture}
        approach_rows = []
        for state in local_states:
            row = evaluate_pose(model, data, "right_wrist_roll_link", state["position_world_m"],
                                state["quaternion_wxyz"], open_q)
            row.update({"approach_slide_m": state["approach_slide_m"], "source_trace_sample": state["source_trace_sample"]})
            approach_rows.append(row)
        target_state = local_states[-1]
        sweep = []
        for i in range(21):
            alpha = i / 20.0
            q = {k: (1.0 - alpha) * open_q[k] + alpha * target_q[k] for k in open_q}
            row = evaluate_pose(model, data, "right_wrist_roll_link", target_state["position_world_m"],
                               target_state["quaternion_wxyz"], q)
            row.update({"sample": i, "alpha": alpha})
            sweep.append(row)
        first_bad = next((r for r in approach_rows if r["wrist_to_bottle"] and r["wrist_to_bottle"]["signed_distance_m"] <= 0.0), None)
        local_result[variant] = {"approach_sample_count": len(approach_rows), "approach": approach_rows,
                                 "first_approach_blocker": first_bad, "closure_static_sweep": sweep,
                                 "endpoint": sweep[-1], "source_trace": str(LOCAL03_TRACE),
                                 "source_trace_sha256": sha256(LOCAL03_TRACE),
                                 "source_candidate_result": str(LOCAL03_RESULT),
                                 "source_candidate_result_sha256": sha256(LOCAL03_RESULT),
                                 "local03_fixture_pose_extraction": local_ref,
                                 "model_inventory": source_geom_inventory(model), "subtree_manifest": manifest}
        if variant == "B" and local_result.get("A", {}).get("first_approach_blocker") is not None:
            a_blocker = local_result["A"]["first_approach_blocker"]
            same_pose = next(r for r in approach_rows
                             if r["source_trace_sample"] == a_blocker["source_trace_sample"])
            local_result[variant]["at_A_first_approach_blocker_pose"] = same_pose
            mujoco.mj_resetData(model, data)
            set_body_pose(model, "right_wrist_roll_link", same_pose["root_position_world_m"],
                          same_pose["root_quaternion_wxyz"])
            configure_qpos(model, data, open_q)
            render(model, data, output / "images" / "local03_first_blocker_B.png",
                   "Local03 exact A first-blocker pose, variant B; static OPEN",
                   (0.30, -0.12, 0.92), 0.38)
        if first_bad is not None:
            mujoco.mj_resetData(model, data)
            set_body_pose(model, "right_wrist_roll_link", first_bad["root_position_world_m"],
                          first_bad["root_quaternion_wxyz"])
            configure_qpos(model, data, open_q)
            render(model, data, output / "images" / f"local03_first_blocker_{variant}.png",
                   f"Local03 OPEN_APPROACH first wrist/bottle blocker: variant {variant}",
                   (0.30, -0.12, 0.92), 0.38)
        mujoco.mj_resetData(model, data)
        set_body_pose(model, "right_wrist_roll_link", target_state["position_world_m"],
                      target_state["quaternion_wxyz"])
        configure_qpos(model, data, target_q)
        render(model, data, output / "images" / f"local03_closed_{variant}.png",
               f"Local03 translated target closure: variant {variant}",
               (0.30, -0.12, 0.92), 0.38)
        local_result[variant]["raw_vs_compiled_surface_samples_at_endpoint"] = right_wrist_raw_surface_report(model, data, variant)
    scenario_results["local03_translated"] = local_result

    front_rows = [json.loads(line) for line in FRONT_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(front_rows) != 1 or front_rows[0].get("phase") != "PREGRASP":
        raise RuntimeError("Expected exactly one saved PREGRASP endpoint for the full-X2 witness")
    front_trace = front_rows[0]
    arm_qpos = {k: float(v) for k, v in front_trace["arm_qpos_rad"].items()}
    workcell_result_path = WORKCELL / "attempt-94e0b6ec-94bf-4fd1-a1ed-a3c0a779e241-retry3/result.json"
    workcell_result = read_json(workcell_result_path)
    base_candidate = workcell_result["candidates"][0]
    full_base_position = [float(v) for v in base_candidate["base_position_world_m"]]
    full_base_yaw = math.radians(float(base_candidate["base_yaw_deg"]))
    front_result = {}
    for variant in ("A", "B"):
        model, data, manifest = build_full(variant, output, helper, arm_qpos,
                                           full_base_position, full_base_yaw)
        metrics = measure(model, data)
        metrics["base_position_world_m"] = full_base_position
        metrics["base_yaw_rad"] = full_base_yaw
        metrics["arm_qpos_rad"] = arm_qpos
        metrics["right_hand_open_qpos"] = {"right_claw_joint": -0.999, "R_hand_wide1_joint": 0.999}
        front_result[variant] = {"source_trace": str(FRONT_TRACE), "source_trace_sha256": sha256(FRONT_TRACE),
                                 "source_trace_row": front_trace, "endpoint": metrics,
                                 "workcell_result": str(workcell_result_path),
                                 "workcell_result_sha256": sha256(workcell_result_path),
                                 "base_candidate_index": 0,
                                 "model_inventory": source_geom_inventory(model), "full_model_manifest": manifest,
                                 "approach_path_status": "UNVERIFIED: saved record contains one PREGRASP endpoint only"}
        render(model, data, output / "images" / f"full_x2_front_pregrasp_{variant}.png",
               f"Full X2 front PREGRASP, variant {variant}; static endpoint", (0.37, 0.16, 0.82), 1.90, 180, -10)
        front_result[variant]["raw_vs_compiled_surface_samples"] = right_wrist_raw_surface_report(model, data, variant)
    scenario_results["full_x2_front_pregrasp"] = front_result

    # summarize source variant mappings directly from all four official URDFs
    variant_files = {"standard": MODEL_DIR / "X2-Ultra.urdf",
                     "omnihand": MODEL_DIR / "X2-Ultra_omnihand.urdf",
                     "omnipicker": SOURCE_URDF}
    mappings = {}
    for key, path in variant_files.items():
        root = ET.parse(path).getroot()
        row = {}
        for side in ("right", "left"):
            link = next(x for x in root.findall("link") if x.get("name") == f"{side}_wrist_roll_link")
            row[f"{side}_visual"] = Path(link.find("visual/geometry/mesh").get("filename", "")).name
            row[f"{side}_collision"] = Path(link.find("collision/geometry/mesh").get("filename", "")).name
            row[f"{side}_mass_kg"] = float(link.find("inertial/mass").get("value"))
            row[f"{side}_visual_origin"] = {k: vec(link.find("visual/origin").get(k), (0, 0, 0)) for k in ("xyz", "rpy")}
            row[f"{side}_collision_origin"] = {k: vec(link.find("collision/origin").get(k), (0, 0, 0)) for k in ("xyz", "rpy")}
        mappings[key] = row
    hand_mounts = {}
    source_root = ET.parse(SOURCE_URDF).getroot()
    for side, joint_name, child in (("right", "R_omnipicker_joint", "R_omnipicker_base_link"),
                                    ("left", "L_omnipicker_joint", "L_omnipicker_base_link")):
        joint = next(x for x in source_root.findall("joint") if x.get("name") == joint_name)
        parent, child_node = joint.find("parent"), joint.find("child")
        origin = joint.find("origin")
        hand_mounts[side] = {"joint": joint_name, "parent": parent.get("link"),
                             "child": child_node.get("link"),
                             "xyz_m": vec(origin.get("xyz"), (0, 0, 0)),
                             "rpy_rad": vec(origin.get("rpy"), (0, 0, 0))}
    wrist_meshes = {}
    for filename in ("right_wrist_roll_link.stl", "right_wrist_roll_extend_link.stl", "left_wrist_roll_extend_link.stl", "right_wrist_roll_link2.stl"):
        path = MESH_DIR / filename
        samples = mesh_samples(path)
        wrist_meshes[filename] = {"sha256": sha256(path), "sample_count": int(len(samples)),
                                  "sample_aabb_local_m": (samples.min(axis=0).tolist(), samples.max(axis=0).tolist()),
                                  "triangle_count": (len(path.read_bytes()) - 84) // 50}
    ab_diff_audit = [
        audit_ab_pair(output / "models/right_wrist_yaw_link_A.urdf",
                      output / "models/right_wrist_yaw_link_B.urdf", "right_wrist_yaw_link subtree"),
        audit_ab_pair(output / "models/right_wrist_roll_link_A.urdf",
                      output / "models/right_wrist_roll_link_B.urdf", "right_wrist_roll_link subtree"),
        audit_ab_pair(output / "models/full_x2_A.urdf",
                      output / "models/full_x2_B.urdf", "full X2 model"),
    ]
    native_libs = sorted(Path(mujoco.__file__).parent.glob("libmujoco.so*"))
    if not native_libs:
        raise RuntimeError("Could not identify the MuJoCo native library")
    repo = Path("/home/lzy18001500226/robotsim-asset-registry-20261012")
    worktree_status = subprocess.run(["git", "-C", str(repo), "status", "--short"],
                                     check=True, capture_output=True, text=True).stdout.strip()
    worktree_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   check=True, capture_output=True, text=True).stdout.strip()

    result = {"status": "STATIC_A_B_COMPLETE_NO_DYNAMIC_VALIDATION",
              "asset_registry_status": "VALIDATED_PENDING_HUMAN_REVIEW",
              "physics_steps": 0, "mujoco_step_calls": 0, "state_advance_calls": 0,
              "allowed_state_calls": ["mj_resetData", "mj_forward"],
              "source": {"repository": "AgibotTech/agibot_x2_urdf", "commit": commit,
                         "source_urdf": str(SOURCE_URDF), "omnipicker_urdf_sha256": sha256(SOURCE_URDF),
                         "variants": mappings, "wrist_meshes": wrist_meshes,
                         "hand_mounts": hand_mounts,
                         "b_collision_candidate_mesh": "right_wrist_roll_extend_link.stl",
                         "b_collision_candidate_classification": "SIMULATION_ONLY"},
              "runtime": {"mujoco": mujoco.__version__, "mujoco_native_library_path": str(native_libs[0]),
                          "mujoco_native_library_sha256": sha256(native_libs[0]),
                          "python": sys.version,
                          "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
                          "worktree_head": worktree_head, "worktree_status": worktree_status,
                          "output_directory": str(output.resolve())},
              "full_x2_pregrasp_base_pose_source": {"result_path": str(workcell_result_path),
                                                    "candidate_index": 0,
                                                    "position_world_m": full_base_position,
                                                    "yaw_rad": full_base_yaw},
              "models": {"variant_A_control": {"source": str(SOURCE_URDF), "physical_mapping_diff_count": 0},
                         "variant_B_candidate": {"physical_mapping_diff_count": 1,
                                                 "only_change": "right_wrist_roll_link/collision/geometry/mesh filename: right_wrist_roll_link.stl -> right_wrist_roll_extend_link.stl"},
                         "A_B_structural_diff_audit": ab_diff_audit},
              "scenarios": scenario_results,
              "classification_semantics": {"CONFIRMED": "direct source/configuration or compiled-model evidence", 
                                            "RULED_OUT_FOR_TESTED_CONDITIONS": "no collision/blocker in sampled named states only",
                                            "UNVERIFIED": "not measured, dynamic, hardware, or exact Boolean mesh test"}}
    write_json(output / "result.json", result)
    make_report(result, output)
    contact_sheet(output)
    make_manifest(output)
    return result


def contact_sheet(output: Path) -> None:
    selected = [output / "images" / f"local03_first_blocker_{v}.png" for v in ("A", "B")]
    selected += [output / "images" / f"legacy_centered_{a}deg_{v}_closed.png" for a in ("0", "90") for v in ("A", "B")]
    thumbs = []
    for p in selected:
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 435))
        tile = Image.new("RGB", (660, 475), (238, 241, 242))
        tile.paste(im, ((660 - im.width) // 2, 34))
        ImageDraw.Draw(tile).text((10, 10), p.name, fill=(20, 26, 30), font=ImageFont.load_default())
        thumbs.append(tile)
    if thumbs:
        cols = 2
        rows = math.ceil(len(thumbs) / cols)
        sheet = Image.new("RGB", (cols * 660, rows * 475), (230, 233, 234))
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % cols) * 660, (i // cols) * 475))
        sheet.save(output / "ab_contact_sheet.png")


def make_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(output / "SHA256SUMS.json", {"files": rows})


def interactive(output: Path, variant: str, state: str) -> None:
    import mujoco.viewer
    helper = load_module("canonical_manipulation_assets", CANONICAL_HELPER)
    scenario_map = {"legacy_centered_0deg": ("right_wrist_yaw_link", RESULT_0),
                    "legacy_centered_90deg": ("right_wrist_yaw_link", RESULT_90),
                    "prior_candidate1": ("right_wrist_yaw_link", CANDIDATE_1)}
    if state.removesuffix("_open") in scenario_map:
        closed = not state.endswith("_open")
        root, path = scenario_map[state.removesuffix("_open")]
        if path == CANDIDATE_1:
            record = read_json(path)["frame_definition"]
            pos, quat = record["root_position_world_m"], record["root_quaternion_wxyz"]
            q = record["source_joint_coordinates_at_contact"] if closed else {"right_claw_joint": -0.999, "R_hand_wide1_joint": 0.999}
        else:
            c = read_json(path)["geometry"]["candidate"]
            pos, quat = c["isolated_root_position_world_m"], quaternion_from_matrix(c["isolated_root_rotation_world"])
            q = c["endpoint_joint_qpos_rad"] if closed else c["open_state"]["open_qpos"]
        model, data, _, _ = build_subtree(variant, root, output, helper, "canonical", (pos, quat))
        set_named_qpos(model, data, q)
        mujoco.mj_forward(model, data)
        target, distance = (0.30, 0.0, 0.90), 0.38
    elif state in ("local03_translated", "local03_translated_open", "local03_first_blocker"):
        if state == "local03_first_blocker":
            saved = read_json(output / "result.json")
            blocker = saved["scenarios"]["local03_translated"]["A"]["first_approach_blocker"]
            if blocker is None:
                raise RuntimeError("No saved A first approach blocker exists in this evidence output")
            local_state = {"position_world_m": blocker["root_position_world_m"],
                           "quaternion_wxyz": blocker["root_quaternion_wxyz"]}
        else:
            states, _ = extract_local03_root_states()
            local_state = states[-1]
        model, data, _, _ = build_subtree(variant, "right_wrist_roll_link", output, helper, "local03",
                                          (local_state["position_world_m"], local_state["quaternion_wxyz"]))
        is_open = state.endswith("_open") or state == "local03_first_blocker"
        q = {"right_claw_joint": -0.98, "R_hand_wide1_joint": 0.98} if is_open else {"right_claw_joint": -0.376, "R_hand_wide1_joint": 0.376}
        set_named_qpos(model, data, q)
        mujoco.mj_forward(model, data)
        target, distance = (0.30, -0.11, 0.92), 0.38
    elif state == "full_x2_front_pregrasp":
        row = json.loads(FRONT_TRACE.read_text(encoding="utf-8").splitlines()[0])
        workcell_result = read_json(WORKCELL / "attempt-94e0b6ec-94bf-4fd1-a1ed-a3c0a779e241-retry3/result.json")
        base = workcell_result["candidates"][0]
        model, data, _ = build_full(variant, output, helper, row["arm_qpos_rad"],
                                    base["base_position_world_m"], math.radians(base["base_yaw_deg"]))
        target, distance = (0.37, 0.16, 0.82), 1.90
    else:
        raise ValueError(f"Unknown state {state}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = target
        viewer.cam.distance = distance
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -14
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.025)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run saved static A/B comparison; default action")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--state", default="local03_translated")
    args = parser.parse_args()
    if args.interactive:
        interactive(args.output, args.variant, args.state)
    else:
        main_run(args.output)


if __name__ == "__main__":
    main()
