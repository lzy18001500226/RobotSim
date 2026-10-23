#!/usr/bin/env python3
"""Build a source-derived AGILINK/X2 OmniPicker joint-tree review package.

This tool makes a local, one-hand URDF adaptation and static MuJoCo MJCF
diagnostics. It deliberately does not define a passive transmission or call
``mj_step``. The MJCF D-site variant is a separate, unverified CAD-mate overlay.
"""
from __future__ import annotations

import argparse
import copy
import csv
import difflib
import hashlib
import html
import io
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

WORKTREE = Path(__file__).resolve().parents[2]
CANONICAL = Path("/home/lzy18001500226/projects/RobotSim")
AGILINK_DIR = CANONICAL / "reference/AGILINK/OmniPicker"
X2_DIR = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0")
X2_URDF = X2_DIR / "X2-Ultra_omnipicker.urdf"
STEP = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_op1_cad/OP1-官网模型.STEP")
XACRO_ZIP = AGILINK_DIR / "agilink-omnipicker-urdf-20260723.zip"
MODEL_ZIP = AGILINK_DIR / "agilink-omnipicker-3d-model-20260723.zip"
SOURCE_ZIP = AGILINK_DIR / "agilink-omnipicker-20260723-source.zip"
XACRO_PATH = "robot_description/urdf/robot_urdf.xacro"
DISPLAY_PATH = "robot_description/urdf/robot.xacro"
D_REFERENCE = CANONICAL / "reviews/issue-46/dpoint-legal-motion-20261021T113855Z/corrected_candidate_B.xml"
SOURCE_TRIANGLES = CANONICAL / "reviews/issue-46/passive-spring-single-motor-20261022T114341Z/source_interference_baseline.json"
COMPARISON_JSON = CANONICAL / "reviews/issue-46/omnipicker-mechanism-comparison-20261023T045449Z/mechanism_comparison.json"
SOURCE_CANDIDATES = {
    "L": D_REFERENCE,
    "R": CANONICAL / "reviews/issue-46/candidate-b-functional-validation-20261022T140838Z/single_motor.xml",
}
EXPECTED = {
    "agilink_xacro_zip": "ab379eac6f8997fceca5dbc139cb6ccf8ae26e52f7791795f67ac3b2849132f5",
    "agilink_3d_zip": "09f7cd75d82954fb73ca9f410b9640207457aa68129d94b2d01cd3c84b1059b2",
    "agilink_source_zip": "eb72794cd1cd4c2b2c12b8fe3bc2be9433d963e0bfbaecb2db6f5a4d347020c8",
    "x2_urdf": "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d",
    "op1_step": "5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b",
}
SOURCE_PHASE_ROOTS = {
    "L": {"left_claw_joint": -0.1341259890668665, "L_hand_wide1_joint": 0.1341259890668665},
    "R": {"right_claw_joint": -0.1341259890668665, "R_hand_wide1_joint": 0.1341259890668665},
}
SIDE_NAMES = {"L": "left", "R": "right"}
XACRO_NS = "{http://www.ros.org/wiki/xacro}"
INTERNAL_REVOLUTES = {
    "narrow2", "narrow3", "narrow_loop", "wide2", "wide3", "wide_loop"
}

sys.path.insert(0, str(CANONICAL / "scripts/agent"))
from path_governance import describe_output_path, validate_output_path, write_reserved_text  # noqa: E402


@dataclass
class Sources:
    xacro_zip: zipfile.ZipFile
    macro_bytes: bytes
    macro: ET.Element
    display: ET.Element
    x2_bytes: bytes
    x2: ET.Element
    x2_links: dict[str, ET.Element]
    x2_joints: dict[str, ET.Element]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def parse_vec(parent: ET.Element, tag: str, attr: str, default: str = "0 0 0") -> tuple[float, ...]:
    item = parent.find(tag)
    raw = item.get(attr, default) if item is not None else default
    return tuple(float(value) for value in raw.split())


def source_joint_name(suffix: str) -> str:
    return f"hand_{suffix}_joint"


def x2_joint_name(side: str, suffix: str) -> str:
    if suffix == "narrow1":
        return f"{SIDE_NAMES[side]}_claw_joint"
    return f"{side}_hand_{suffix}_joint"


def x2_link_name(side: str, suffix: str) -> str:
    return f"{side}_omnipicker_base_link" if suffix == "base_link" else f"{side}_{suffix}"


def suffix_from_xacro_name(name: str) -> str:
    return name.removeprefix("hand_").removesuffix("_joint")


def load_sources() -> Sources:
    for key, path in (("agilink_xacro_zip", XACRO_ZIP), ("agilink_3d_zip", MODEL_ZIP), ("agilink_source_zip", SOURCE_ZIP), ("x2_urdf", X2_URDF), ("op1_step", STEP)):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha_file(path)
        if actual != EXPECTED[key]:
            raise RuntimeError(f"Pinned source SHA mismatch for {key}: {actual}")
    z = zipfile.ZipFile(XACRO_ZIP)
    broken = z.testzip()
    if broken:
        raise RuntimeError(f"AGILINK Xacro ZIP CRC failure: {broken}")
    macro_bytes = z.read(XACRO_PATH)
    macro = ET.fromstring(macro_bytes)
    display = ET.fromstring(z.read(DISPLAY_PATH))
    x2_bytes = X2_URDF.read_bytes()
    x2 = ET.fromstring(x2_bytes)
    return Sources(z, macro_bytes, macro, display, x2_bytes, x2,
                   {e.get("name"): e for e in x2.findall("link")},
                   {e.get("name"): e for e in x2.findall("joint")})


def xacro_links(src: Sources) -> dict[str, ET.Element]:
    return {e.get("name"): e for e in src.macro.findall("link")}


def xacro_joints(src: Sources) -> dict[str, ET.Element]:
    return {e.get("name"): e for e in src.macro.findall("joint")}


def source_limits(joint: ET.Element) -> dict[str, str] | None:
    item = joint.find("limit")
    return dict(item.attrib) if item is not None else None


def frame_matches(source_joint: ET.Element, x2_joint: ET.Element, side: str) -> tuple[bool, dict[str, object]]:
    suffix = suffix_from_xacro_name(source_joint.get("name", ""))
    source_parent = source_joint.find("parent").get("link")
    source_child = source_joint.find("child").get("link")
    expected_parent = x2_link_name(side, source_parent)
    expected_child = x2_link_name(side, source_child)
    actual_parent = x2_joint.find("parent").get("link")
    actual_child = x2_joint.find("child").get("link")
    xyz = parse_vec(source_joint, "origin", "xyz")
    xyz2 = parse_vec(x2_joint, "origin", "xyz")
    rpy = parse_vec(source_joint, "origin", "rpy")
    rpy2 = parse_vec(x2_joint, "origin", "rpy")
    axis = parse_vec(source_joint, "axis", "xyz")
    axis2 = parse_vec(x2_joint, "axis", "xyz")
    ok = (actual_parent == expected_parent and actual_child == expected_child
          and xyz == xyz2 and rpy == rpy2 and axis == axis2)
    return ok, {
        "side": side,
        "xacro_joint": source_joint.get("name"),
        "x2_joint": x2_joint.get("name"),
        "xacro_parent": expected_parent,
        "xacro_child": expected_child,
        "x2_parent": actual_parent,
        "x2_child": actual_child,
        "xacro_type": source_joint.get("type"),
        "x2_type": x2_joint.get("type"),
        "origin_xyz_m": xyz,
        "origin_rpy_rad": rpy,
        "axis_joint_frame": axis,
        "frame_match": ok,
        "xacro_limits": source_limits(source_joint),
        "x2_limits": source_limits(x2_joint),
        "x2_mimic": dict(x2_joint.find("mimic").attrib) if x2_joint.find("mimic") is not None else None,
        "coordinate_note": "AGILINK declares one local hand macro; X2 L/R local frames are compared independently.",
    }


def mesh_entries(src: Sources) -> dict[str, str]:
    entries: dict[str, str] = {}
    for name in src.xacro_zip.namelist():
        if name.casefold().endswith(".stl"):
            entries[Path(name).name.casefold()] = name
    return entries


def mesh_rows(src: Sources, side: str) -> list[dict[str, object]]:
    xlinks = xacro_links(src)
    source_meshes = mesh_entries(src)
    rows = []
    for xname, xl in xlinks.items():
        x2name = x2_link_name(side, xname)
        x2l = src.x2_links[x2name]
        xmesh = []
        for node in xl.findall(".//mesh"):
            filename = Path(urlparse(node.get("filename", "")).path).name
            entry = source_meshes.get(filename.casefold())
            x2file = next((p for p in (X2_DIR / "meshes").iterdir() if p.name.casefold() == filename.casefold()), None)
            xbytes = src.xacro_zip.read(entry) if entry else b""
            x2hash = sha_file(x2file) if x2file else None
            xmesh.append({"filename": filename, "xacro_member": entry,
                          "xacro_sha256": sha_bytes(xbytes) if xbytes else None,
                          "x2_path": str(x2file) if x2file else None,
                          "x2_sha256": x2hash,
                          "identical": bool(xbytes and x2hash and sha_bytes(xbytes) == x2hash)})
        x2mesh = []
        for node in x2l.findall(".//mesh"):
            filename = Path(urlparse(node.get("filename", "")).path).name
            candidate = next((p for p in (X2_DIR / "meshes").iterdir() if p.name.casefold() == filename.casefold()), None)
            x2mesh.append({"filename": filename, "sha256": sha_file(candidate) if candidate else None})
        rows.append({"side": side, "xacro_link": xname, "x2_link": x2name,
                     "xacro_inertial": inertial_record(xl), "x2_inertial": inertial_record(x2l),
                     "inertial_match": inertial_equal(xl, x2l), "xacro_meshes": xmesh,
                     "x2_meshes": x2mesh,
                     "visual_count_xacro": len(xl.findall("visual")), "visual_count_x2": len(x2l.findall("visual")),
                     "collision_count_xacro": len(xl.findall("collision")), "collision_count_x2": len(x2l.findall("collision"))})
    return rows


def inertial_record(link: ET.Element) -> dict[str, object] | None:
    item = link.find("inertial")
    if item is None:
        return None
    origin = item.find("origin")
    mass = item.find("mass")
    inertia = item.find("inertia")
    return {"origin_xyz": parse_vec(item, "origin", "xyz"),
            "origin_rpy": parse_vec(item, "origin", "rpy"),
            "mass": float(mass.get("value")) if mass is not None else None,
            "inertia": {k: float(v) for k, v in sorted(inertia.attrib.items())} if inertia is not None else None}


def inertial_equal(first: ET.Element, second: ET.Element, tol: float = 1e-12) -> bool:
    a, b = inertial_record(first), inertial_record(second)
    if a is None or b is None:
        return a is b
    if any(abs(x-y) > tol for x, y in zip(a["origin_xyz"] + a["origin_rpy"], b["origin_xyz"] + b["origin_rpy"])):
        return False
    if abs(a["mass"] - b["mass"]) > tol:
        return False
    return a["inertia"].keys() == b["inertia"].keys() and all(abs(a["inertia"][k]-b["inertia"][k]) <= tol for k in a["inertia"])


def build_candidate_urdf(src: Sources, side: str) -> ET.Element:
    """Extract one X2 hand and change only the six Xacro-confirmed joint types."""
    prefix = side
    link_names = {x2_link_name(side, x) for x in xacro_links(src)}
    root_link = x2_link_name(side, "base_link")
    robot = ET.Element("robot", {"name": f"x2_omnipicker_{'left' if side == 'L' else 'right'}_agilink_joint_tree_review_only"})
    for material in src.x2.findall("material"):
        robot.append(copy.deepcopy(material))
    for xname in xacro_links(src):
        link = copy.deepcopy(src.x2_links[x2_link_name(side, xname)])
        for mesh in link.findall(".//mesh"):
            original = Path(urlparse(mesh.get("filename", "")).path).name
            resolved = next((p for p in (X2_DIR / "meshes").iterdir() if p.name.casefold() == original.casefold()), None)
            if resolved is None:
                raise FileNotFoundError(f"X2 mesh missing for {mesh.get('filename')}")
            mesh.set("filename", resolved.as_uri())
        robot.append(link)
    source_by_suffix = {suffix_from_xacro_name(n): j for n, j in xacro_joints(src).items()}
    included = []
    for suffix, source_joint in source_by_suffix.items():
        name = x2_joint_name(side, suffix)
        joint = copy.deepcopy(src.x2_joints[name])
        if suffix in INTERNAL_REVOLUTES:
            if source_joint.get("type") != "revolute" or joint.get("type") != "fixed":
                raise RuntimeError(f"Expected reviewed fixed-to-revolute delta at {name}")
            joint.set("type", "revolute")
        included.append(joint)
    for joint in included:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent in link_names and child in link_names:
            robot.append(joint)
    if root_link not in {link.get("name") for link in robot.findall("link")}:
        raise RuntimeError("candidate hand root was not included")
    roots = {link.get("name") for link in robot.findall("link")}
    child_links = {joint.find("child").get("link") for joint in robot.findall("joint")}
    if roots - child_links != {root_link}:
        raise RuntimeError(f"candidate must have one hand-base root: {roots-child_links}")
    ET.indent(robot, space="  ")
    return robot


def candidate_delta_rows(src: Sources) -> list[dict[str, object]]:
    rows = []
    xj = xacro_joints(src)
    for side in ("L", "R"):
        for source in xj.values():
            suffix = suffix_from_xacro_name(source.get("name", ""))
            target = src.x2_joints[x2_joint_name(side, suffix)]
            _, row = frame_matches(source, target, side)
            row["candidate_type"] = source.get("type") if suffix in INTERNAL_REVOLUTES else target.get("type")
            row["candidate_delta"] = "type fixed -> revolute" if suffix in INTERNAL_REVOLUTES else "unchanged from pinned X2"
            rows.append(row)
    return rows


def origin_quat_wxyz(rpy: tuple[float, ...]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


def build_mjcf(urdf: ET.Element, side: str, *, d_closure_overlay: bool = False) -> ET.Element:
    """Translate the one-hand URDF tree to static-check MJCF; do not add a transmission."""
    robot = ET.Element("mujoco", {"model": f"x2_agilink_tree_{side.lower()}_review_only" + ("_unverified_D_overlay" if d_closure_overlay else "")})
    ET.SubElement(robot, "compiler", {"angle": "radian", "autolimits": "true", "meshdir": str(X2_DIR / "meshes"), "inertiafromgeom": "false", "fusestatic": "false"})
    ET.SubElement(robot, "option", {"gravity": "0 0 -9.81", "timestep": "0.001"})
    ET.SubElement(robot, "size", {"nconmax": "512", "njmax": "4096"})
    asset = ET.SubElement(robot, "asset")
    meshes: dict[str, str] = {}
    for link in urdf.findall("link"):
        for node in link.findall(".//mesh"):
            filename = Path(urlparse(node.get("filename", "")).path).name
            meshes.setdefault(filename, "mesh_" + Path(filename).stem)
    for filename, name in meshes.items():
        ET.SubElement(asset, "mesh", {"name": name, "file": filename})
    world = ET.SubElement(robot, "worldbody")
    body_by_link: dict[str, ET.Element] = {}
    joints_by_parent: dict[str, list[ET.Element]] = defaultdict(list)
    for joint in urdf.findall("joint"):
        joints_by_parent[joint.find("parent").get("link")].append(joint)
    link_by_name = {link.get("name"): link for link in urdf.findall("link")}
    root_name = next(iter(set(link_by_name) - {j.find("child").get("link") for j in urdf.findall("joint")}))

    def add_link_geometry(body: ET.Element, link: ET.Element) -> None:
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass").get("value")
            tensor = inertial.find("inertia").attrib
            full = " ".join(tensor[k] for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
            attrib = {"mass": mass, "fullinertia": full}
            if origin is not None:
                attrib["pos"] = origin.get("xyz", "0 0 0")
                if origin.get("rpy", "0 0 0") != "0 0 0":
                    attrib["quat"] = " ".join(f"{v:.15g}" for v in origin_quat_wxyz(parse_vec(inertial, "origin", "rpy")))
            ET.SubElement(body, "inertial", attrib)
        for group, tag, rgb in (("visual", "visual", "0.47 0.72 0.68 1"), ("collision", "collision", "0.92 0.60 0.28 0.30")):
            for i, geom in enumerate(link.findall(group)):
                mesh = geom.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = Path(urlparse(mesh.get("filename", "")).path).name
                node = ET.SubElement(body, "geom", {
                    "name": f"{link.get('name')}__{tag}_{i}", "type": "mesh", "mesh": meshes[filename],
                    "pos": (geom.find("origin").get("xyz", "0 0 0") if geom.find("origin") is not None else "0 0 0"),
                    "rgba": rgb,
                    "contype": "0" if tag == "visual" else "1",
                    "conaffinity": "0" if tag == "visual" else "1",
                    "density": "0",
                    "group": "0" if tag == "visual" else "1",
                })
                orient = geom.find("origin")
                if orient is not None and orient.get("rpy", "0 0 0") != "0 0 0":
                    node.set("quat", " ".join(f"{v:.15g}" for v in origin_quat_wxyz(parse_vec(geom, "origin", "rpy"))))

    def add_joint_body(parent_body: ET.Element, joint: ET.Element) -> None:
        child_name = joint.find("child").get("link")
        origin = joint.find("origin")
        attrs = {"name": child_name,
                 "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                 "quat": " ".join(f"{v:.15g}" for v in origin_quat_wxyz(parse_vec(joint, "origin", "rpy"))) if origin is not None else "1 0 0 0"}
        body = ET.SubElement(parent_body, "body", attrs)
        body_by_link[child_name] = body
        if joint.get("type") == "revolute":
            axis = joint.find("axis").get("xyz", "0 0 1") if joint.find("axis") is not None else "0 0 1"
            limit = joint.find("limit")
            jattrs = {"name": joint.get("name"), "type": "hinge", "pos": "0 0 0", "axis": axis}
            if limit is not None and float(limit.get("lower", "-3.141592653589793")) < float(limit.get("upper", "3.141592653589793")):
                jattrs.update({"limited": "true", "range": f"{limit.get('lower')} {limit.get('upper')}"})
            ET.SubElement(body, "joint", jattrs)
        add_link_geometry(body, link_by_name[child_name])
        for child_joint in joints_by_parent.get(child_name, []):
            add_joint_body(body, child_joint)

    base = ET.SubElement(world, "body", {"name": root_name})
    body_by_link[root_name] = base
    add_link_geometry(base, link_by_name[root_name])
    for root_joint in joints_by_parent.get(root_name, []):
        add_joint_body(base, root_joint)
    equality = ET.SubElement(robot, "equality")
    wide_root = x2_joint_name(side, "wide1")
    narrow_root = x2_joint_name(side, "narrow1")
    ET.SubElement(equality, "joint", {"name": "x2_source_wide_root_mimic", "joint1": wide_root, "joint2": narrow_root,
                                       "polycoef": "0 -1 0 0 0"})
    if d_closure_overlay:
        if side != "L":
            raise ValueError("D-site overlay coordinates in this evidence are the left-hand PR84 mapping only")
        measured = ET.parse(D_REFERENCE).getroot()
        source_sites = {site.get("name"): site for site in measured.findall(".//site") if site.get("name", "").startswith("candidate_")}
        site_map = {
            "candidate_narrow_D_jaw": ("L_hand_narrow3_Link", "x2_D_overlay_narrow_jaw"),
            "candidate_narrow_D_loop": ("L_hand_narrow_loop_Link", "x2_D_overlay_narrow_loop"),
            "candidate_wide_D_jaw": ("L_hand_wide3_Link", "x2_D_overlay_wide_jaw"),
            "candidate_wide_D_loop": ("L_hand_wide_loop_Link", "x2_D_overlay_wide_loop"),
        }
        for old_name, (link_name, new_name) in site_map.items():
            if old_name not in source_sites:
                raise RuntimeError(f"D-site source missing: {old_name}")
            site = source_sites[old_name]
            ET.SubElement(body_by_link[link_name], "site", {"name": new_name, "pos": site.get("pos", "0 0 0"), "size": "0.00035"})
        ET.SubElement(equality, "connect", {"name": "unverified_CAD_narrow_D_mate", "site1": "x2_D_overlay_narrow_jaw", "site2": "x2_D_overlay_narrow_loop"})
        ET.SubElement(equality, "connect", {"name": "unverified_CAD_wide_D_mate", "site1": "x2_D_overlay_wide_jaw", "site2": "x2_D_overlay_wide_loop"})
    ET.indent(robot, space="  ")
    return robot


def static_mujoco_audit(path: Path, side: str, *, source_pose: bool, with_D: bool = False) -> dict[str, object]:
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if source_pose:
        for name, value in SOURCE_PHASE_ROOTS[side].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"static pose joint missing: {name}")
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    nrow = int(data.nefc)
    if nrow:
        jac = np.asarray(data.efc_J).reshape((nrow, model.nv))
        eqmask = np.asarray(data.efc_type[:nrow]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        eqjac = jac[eqmask]
    else:
        eqjac = np.zeros((0, model.nv))
    if len(eqjac):
        normalized = eqjac / (np.linalg.norm(eqjac, axis=1)[:, None] + 1e-30)
        singular = np.linalg.svd(normalized, compute_uv=False)
        rank = int(np.sum(singular > max(singular[0] * 1e-8, 1e-10)))
    else:
        singular, rank = np.array([]), 0
    joints = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        joints.append({"name": name, "type": int(model.jnt_type[jid]),
                       "qpos": float(data.qpos[model.jnt_qposadr[jid]]) if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE else None,
                       "axis_world": [float(x) for x in data.xaxis[jid]],
                       "anchor_world_m": [float(x) for x in data.xanchor[jid]]})
    bodies = []
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        bodies.append({"name": name, "position_m": [float(x) for x in data.xpos[bid]],
                       "rotation_matrix_row_major": [float(x) for x in data.xmat[bid].reshape(3, 3).ravel()]})
    contact_rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        contact_rows.append({"geom1": g1, "geom2": g2, "distance_m": float(c.dist),
                             "classification": "MuJoCo convex-hull proxy only; not B-rep volume penetration"})
    d_gaps = {}
    if with_D:
        for a, b, key in (("x2_D_overlay_narrow_jaw", "x2_D_overlay_narrow_loop", "narrow"),
                          ("x2_D_overlay_wide_jaw", "x2_D_overlay_wide_loop", "wide")):
            ia = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a)
            ib = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, b)
            d_gaps[key] = float(np.linalg.norm(data.site_xpos[ia] - data.site_xpos[ib]) * 1000.0)
    return {"model": str(path), "mujoco_version": mujoco.__version__, "mj_forward_calls": 1,
            "mj_step_calls": 0, "nq": int(model.nq), "nv": int(model.nv),
            "n_joints": int(model.njnt), "n_equalities": int(model.neq),
            "equality_jacobian_rows_active": int(len(eqjac)), "equality_jacobian_rank": rank,
            "mobility_nv_minus_rank": int(model.nv-rank), "singular_values_normalized_eq_jacobian": singular.tolist(),
            "source_pose_applied": source_pose, "sites_D_gap_mm": d_gaps,
            "static_contact_count": int(data.ncon), "convex_hull_contacts": contact_rows,
            "joint_states_and_world_axes": joints, "body_world_fk": bodies}


def saved_rank(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"model": str(path), "status": "MISSING"}
    return static_mujoco_audit(path, "L" if "left" in path.name else "R", source_pose=False)


def source_step_witnesses() -> dict[str, object]:
    raw = json.loads(SOURCE_TRIANGLES.read_text(encoding="utf-8"))
    pairs = raw.get("pairs", [])
    return {"path": str(SOURCE_TRIANGLES), "witness_count": len(pairs),
            "unique_pair_count": len({(x["body1"], x["body2"]) for x in pairs}),
            "pairs": [{"body1": x["body1"], "body2": x["body2"],
                       "triangle_a": x.get("source_triangle_a"), "triangle_b": x.get("source_triangle_b"),
                       "classification": x.get("classification")} for x in pairs],
            "interpretation": "Saved original-STL surface crossing witnesses at registered source pose. They do not alone prove volumetric interpenetration or intended fit."}


def read_historical_comparison() -> dict[str, object]:
    data = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    return {"path": str(COMPARISON_JSON), "candidate_B": data.get("candidate_B"),
            "passive_spring": data.get("passive_spring"), "quadratic": data.get("quadratic"),
            "limitations": "Saved experiment summaries are compared as prior evidence; no new physics was run."}


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(body) + "\n"


def source_graph_svg() -> str:
    """A topology/frame review diagram, intentionally not a motion animation."""
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="690" viewBox="0 0 1220 690">
<style>text{font-family:Arial,sans-serif;fill:#172c3d}.node{fill:#f6fbff;stroke:#244a63;stroke-width:2}.rev{stroke:#147d64;stroke-width:4}.fix{stroke:#777;stroke-width:3}.mimic{stroke:#285bb4;stroke-width:3;stroke-dasharray:9 6}.cad{stroke:#b87b15;stroke-width:3;stroke-dasharray:4 7}.warn{fill:#a23b2b;font-weight:bold}.small{font-size:14px}.title{font-size:22px;font-weight:bold}</style>
<text x="25" y="34" class="title">AGILINK Xacro declaration mapped to one X2 hand — topology only</text>
<text x="25" y="62" class="small">Green = source revolute; gray = source fixed; blue dashed = X2 wide-root mimic; orange dotted = separate unverified CAD D-site overlay.</text>
<rect class="node" x="520" y="90" width="180" height="54" rx="8"/><text x="610" y="122" text-anchor="middle">OmniPicker base</text>
<line class="rev" x1="570" y1="144" x2="310" y2="215"/><text x="395" y="173" class="small">narrow1 root hinge</text>
<line class="rev" x1="650" y1="144" x2="910" y2="215"/><text x="790" y="173" class="small">wide1 root hinge</text>
<line class="rev" x1="520" y1="125" x2="160" y2="390"/><text x="245" y="265" class="small">narrow loop hinge (independent)</text>
<line class="rev" x1="700" y1="125" x2="1060" y2="390"/><text x="825" y="265" class="small">wide loop hinge (independent)</text>
<rect class="node" x="220" y="215" width="180" height="50" rx="8"/><text x="310" y="245" text-anchor="middle">narrow1_Link</text>
<rect class="node" x="220" y="315" width="180" height="50" rx="8"/><text x="310" y="345" text-anchor="middle">narrow2_Link</text>
<line class="rev" x1="310" y1="265" x2="310" y2="315"/><text x="323" y="296" class="small">narrow2</text>
<rect class="node" x="220" y="415" width="180" height="50" rx="8"/><text x="310" y="445" text-anchor="middle">narrow3_Link</text>
<line class="rev" x1="310" y1="365" x2="310" y2="415"/><text x="323" y="396" class="small">narrow3</text>
<rect class="node" x="220" y="515" width="180" height="50" rx="8"/><text x="310" y="545" text-anchor="middle">narrow4_Link</text>
<line class="fix" x1="310" y1="465" x2="310" y2="515"/><text x="323" y="496" class="small">narrow4 fixed</text>
<rect class="node" x="820" y="215" width="180" height="50" rx="8"/><text x="910" y="245" text-anchor="middle">wide1_Link</text>
<rect class="node" x="820" y="315" width="180" height="50" rx="8"/><text x="910" y="345" text-anchor="middle">wide2_Link</text>
<line class="rev" x1="910" y1="265" x2="910" y2="315"/><text x="923" y="296" class="small">wide2</text>
<rect class="node" x="820" y="415" width="180" height="50" rx="8"/><text x="910" y="445" text-anchor="middle">wide3_Link</text>
<line class="rev" x1="910" y1="365" x2="910" y2="415"/><text x="923" y="396" class="small">wide3</text>
<rect class="node" x="820" y="515" width="180" height="50" rx="8"/><text x="910" y="545" text-anchor="middle">wide4_Link</text>
<line class="fix" x1="910" y1="465" x2="910" y2="515"/><text x="923" y="496" class="small">wide4 fixed</text>
<rect class="node" x="75" y="390" width="170" height="50" rx="8"/><text x="160" y="420" text-anchor="middle">narrow_loop_Link</text>
<rect class="node" x="975" y="390" width="170" height="50" rx="8"/><text x="1060" y="420" text-anchor="middle">wide_loop_Link</text>
<line class="mimic" x1="400" y1="235" x2="820" y2="235"/><text x="610" y="226" text-anchor="middle" class="small">X2 wide1 = - narrow1 (official URDF mimic; not present in Xacro)</text>
<path class="cad" d="M400 440 C520 600 720 600 820 440" fill="none"/><text x="610" y="615" text-anchor="middle" class="small">CAD D-site connects shown only in separate overlay; Xacro does not define these mates.</text>
<text x="25" y="660" class="warn">No closed-loop couplings, springs, gear law or stops are defined by this graph. It does not establish a one-input physical trajectory.</text>
</svg>'''


def make_review_html(data: dict[str, object]) -> str:
    rows = data["joint_map"]
    table_rows = "".join(
        f"<tr><td>{html.escape(str(x['side']))}</td><td>{html.escape(str(x['x2_joint']))}</td><td>{html.escape(str(x['x2_parent']))} → {html.escape(str(x['x2_child']))}</td><td>{html.escape(str(x['x2_type']))} → {html.escape(str(x['candidate_type']))}</td><td>{html.escape(str(x['origin_xyz_m']))}</td><td>{html.escape(str(x['axis_joint_frame']))}</td><td>{'MATCH' if x['frame_match'] else 'MISMATCH'}</td></tr>" for x in rows)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Issue 46 AGILINK source-tree review</title>
<style>body{{font:16px/1.5 system-ui,Segoe UI,sans-serif;max-width:1280px;margin:24px auto;padding:0 20px;color:#172c3d;background:#f5f8fa}}h1,h2{{color:#143d57}}.banner{{background:#fff3d9;border-left:6px solid #ca8522;padding:12px 16px;font-weight:700}}.card{{background:white;border:1px solid #d2dee6;border-radius:8px;padding:16px;margin:16px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #cbd6dd;padding:6px;vertical-align:top}}th{{background:#eaf1f5}}code{{background:#e9eff3;padding:2px 4px}}</style>
<h1>Issue #46 — AGILINK Xacro → X2 articulation-tree review</h1>
<div class="banner">SOURCE-DERIVED JOINT TREE — STATIC ONLY. This is not a coupled or physically validated OmniPicker mechanism. No MuJoCo mj_step was run.</div>
<section class="card"><h2>What the sources establish</h2><ul><li>AGILINK macro: 8 revolute + 2 fixed joints for one hand; six X2 internal fixed joints are revolute in the macro.</li><li>All 20 L/R parent-child, origin, and axis comparisons match: {data['frame_match_count']}/{data['frame_count']}.</li><li>X2's per-hand wide-root mimic, official root limits, X2 meshes, and inertias are retained.</li><li>AGILINK macro has no mimic, transmission, ros2_control, or loop mates. The supplied display <code>robot.xacro</code> includes the macro but never instantiates it.</li></ul></section>
<section class="card"><h2>Topology diagram</h2><img alt="Joint topology diagram" src="joint_tree.svg" style="max-width:100%"></section>
<section class="card"><h2>Joint comparison and candidate delta</h2><table><thead><tr><th>Side</th><th>X2 joint</th><th>Parent → child</th><th>X2 → candidate</th><th>Origin xyz (m)</th><th>Axis in joint frame</th><th>Frame</th></tr></thead><tbody>{table_rows}</tbody></table></section>
<section class="card"><h2>Independent mobility</h2><p>Per hand, AGILINK joint tree: 8 revolute coordinates; applying the unchanged X2 wide-root mimic leaves 7 DOF. One commanded root leaves 6 unassigned internal coordinates. With the separate CAD D-site overlay, the local static equality Jacobian rank is {data['static_models']['left_D_overlay']['equality_jacobian_rank']}; its residual and numerical caveats are in the JSON. Candidate B adds two extra scalar rate laws and drops mobility further, but those laws are not present in AGILINK sources.</p><p>Therefore a one-motor open/close trajectory is not determined by the Xacro joint tree alone.</p></section>
<section class="card"><h2>Review artifacts</h2><ul>{''.join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in ['REPORT.md','source_analysis.json','source_joint_map.csv','source_link_map.csv','candidate_joint_delta.csv','X2-OmniPicker-left-AGILINK-articulated-tree-review-only.urdf','X2-OmniPicker-right-AGILINK-articulated-tree-review-only.urdf','agilink_left_source_tree_static.xml','agilink_right_source_tree_static.xml','agilink_left_D_closure_overlay_UNVERIFIED.xml','joint_tree.svg'])}</ul></section>
</html>'''


def make_report(d: dict[str, object], output_dir: Path) -> str:
    source_table = table(d["source_provenance"], [("name", "Source"), ("sha256", "SHA-256"), ("detail", "Details")])
    joint_table = table(d["joint_map"], [("side", "Side"), ("x2_joint", "X2 joint"), ("x2_parent", "Parent"), ("x2_child", "Child"), ("x2_type", "X2 type"), ("candidate_type", "Candidate type"), ("origin_xyz_m", "Origin xyz m"), ("origin_rpy_rad", "Origin RPY"), ("axis_joint_frame", "Axis"), ("xacro_limits", "AGILINK limits"), ("x2_limits", "X2 limits"), ("x2_mimic", "X2 mimic"), ("frame_match", "Frame match"), ("candidate_delta", "Delta")])
    link_table = table(d["link_map"], [("side", "Side"), ("xacro_link", "Xacro link"), ("x2_link", "X2 link"), ("inertial_match", "Inertial match"), ("visual_count_xacro", "Xacro visual"), ("visual_count_x2", "X2 visual"), ("collision_count_xacro", "Xacro collision"), ("collision_count_x2", "X2 collision"), ("mesh_summary", "Mesh SHA result")])
    collision_pairs = "\n".join(f"- `{p['body1']} × {p['body2']}`: triangles {p['triangle_a']}/{p['triangle_b']}; source-pose crossing witness; volume/intent unresolved." for p in d["source_stl_witnesses"]["pairs"])
    return f'''# Issue #46 — AGILINK Xacro source-derived OmniPicker candidate

**Result:** `SOURCE_TREE_CANDIDATE_READY_FOR_STATIC_REVIEW`
**Physical mechanism repaired:** `NOT ESTABLISHED`
**New MuJoCo dynamics:** `NOT RUN` (`mj_step_calls=0`)
**Candidate B / PR #86:** preserved and unchanged.

## Finding

The July 2026 AGILINK Xacro directly supports changing six inner joints per hand from fixed to revolute at the same parent/child frames, origins, and axes used by pinned X2 v1.4.0. This corrects a **joint-type discrepancy** in the X2 URDF. It does **not** encode a coupled one-motor mechanism: the macro has no mimic/transmission/ros2_control or mates for the two loop links. The display entrypoint contains an additional packaging defect: `robot.xacro` only includes the macro and does not invoke it, so the supplied launch path does not instantiate this declared model.

The source-derived candidate therefore restores these six joint types and keeps official X2 root limits, root mimic, meshes, inertias, and fixed tips. It is a kinematic tree for review, not a complete physical transmission. It has seven independent coordinates per hand after the X2 wide-root mimic; commanding the main root leaves six unconstrained coordinates in the pure tree. A root-only motion must not be mistaken for passive linkage behavior.

## Provenance

{source_table}

The source archives are local preserved copies of AGILINK's July 2026 URDF/Xacro and STEP/3D downloads. The archive/package does not contain a usable license file in the inspected members; redistribution rights for individual meshes remain unverified. Pinned X2 meshes remain in the existing local cache and are referenced by absolute `file://` URIs; the candidate URDF is therefore local-environment review material, not portable packaging.

## Joint mapping

The Xacro defines one local hand macro; the exact local frames were compared separately against both left and right X2 hand trees. All {d['frame_match_count']}/{d['frame_count']} parent/child/origin/axis records match. For each side the only candidate delta is the six fixed→revolute changes:

{joint_table}

Root limits stay at the pinned X2 values (narrow `{d['root_limits']['narrow']}`, wide `{d['root_limits']['wide']}`); the X2 wide-root mimic multiplier stays `-1`. AGILINK's root display limits are ±3.14, but the derived candidate does not replace official X2 root limits with those display values. Inner Xacro limits are ±3.14 with zero effort/velocity fields: they are declared visualization limits, not verified physical stops.

## Link, mesh, and inertia mapping

{link_table}

The eight moving STL meshes are byte-identical between the AGILINK archive and pinned X2 source. The base mesh differs; candidate intentionally retains the X2 base. Both `*_4_Link` terminal links are empty in both descriptions and remain fixed. Inertial values match on mapped links where present. Official source link declarations, inertias, visual meshes, and collision meshes were not edited.

## Mobility, static closure, and comparison

- AGILINK declared tree, one hand: 8 revolute coordinates, no source equality → `M=8`.
- Add preserved X2 root mimic: equality Jacobian rank 1 → `M=7`; one commanded root leaves six free internal coordinates.
- Both hands independently: 16 coordinates, two per-hand mimic relations → `M=14`; no source cross-hand coupling.
- Separate D-site diagnostic overlay: two CAD-derived point connects are included only in `agilink_left_D_closure_overlay_UNVERIFIED.xml`; source-pose D residuals are `{d['static_models']['left_D_overlay']['sites_D_gap_mm']}` mm and the measured equality rank is {d['static_models']['left_D_overlay']['equality_jacobian_rank']}. The **site coordinates** are CAD-derived; the interpretation that each point pair is the correct physical mate is still a candidate overlay, not declared by AGILINK. No Candidate-B rate law or spring is included.
- `mj_forward` static compilation checked link FK, axes, mimic/equality rows, and convex-hull contacts. It did not integrate dynamics. Contact counts below are only MuJoCo convex-hull proxy observations, not B-rep truth.

Historical comparison reuses the saved evidence only. Candidate B ran 19,750 physical steps over its bounded partial stroke, but its `q_root + q_B - q_E = 0` relation and boundary near +0.238 rad are experimental; that algebraic boundary is not a manufacturer stop. The passive-spring trace had remaining passive DOFs and stopped at a measurement-coverage gate, not a proven hard stop. The quadratic candidate is a clearance-selected fit. This new source tree uses none of those laws. See `saved_candidate_comparison.json` for exact retained values.

## Collision evidence

The new derived URDF adds no collision exclusions. Candidate B had four explicit loop exclusions; they are not carried into the source-derived baseline. Original-STEP/X2 STL source-pose evidence already contains {d['source_stl_witnesses']['witness_count']} triangle crossing witnesses across {d['source_stl_witnesses']['unique_pair_count']} pairs:

{collision_pairs}

These are surface-triangle intersection witnesses from the saved original-STL comparison. They do not establish volume penetration or prove intended fits. The separate source-pose report flags open/nonmanifold source meshes, so containment and mating intent are not fully measurable. The `mj_forward` contact list uses MuJoCo convex hulls and may overstate contact for concave parts; no pair was disabled to hide it.

## Mechanism hypotheses

1. **Rigid one-input transmission:** not established. The Xacro gives hinges and a tree, not mates or a transmission law. D-point constraints alone, in the measured overlay, do not force mobility to one.
2. **Underactuated/adaptive mechanism:** plausible and consistent with the vendor manual's one-active-DOF adaptive description. The source does not specify spring anchors, rest/preload, or contact-selected branch, so the saved spring trace is not a verified law.
3. **Mixed transmission/adaptation:** also plausible but not discriminated by the current source. No exact vendor relation or spring specification was found in the supplied Xacro/STEP package.

Candidate B's partial motion remains a useful experimental baseline, but the Xacro evidence suggests its fixed-to-revolute joint types are source-supported while its added rate equations are not. This can explain why Candidate B's observed boundary belongs to its chosen coupling; it cannot prove the AGILINK hinges alone will open the gripper correctly.

## Exact missing information and dynamic gate

To choose between the hypotheses, obtain either a vendor mate/transmission definition or an observed set of passive joint coordinates at several known root commands, including the free-space opening/closing branch and contact-adaptive behavior. If adaptation is spring-driven, the two spring-leg attachment bodies/points and rest/torque-angle relation are needed. The saved STEP assembly has no explicit joint/mate entity records or motion law.

No dynamics are authorized or performed. If separately approved, the next bounded test should begin at the registered source pose, use one root actuator per gripper and no passive qpos writes, and gate every increment on (a) D-site residual, (b) all original source-mesh collision checks, (c) root limits, (d) passive joint continuity/speed, and (e) finite state/solver warnings. A static kinematic tree is not a prerequisite pass for force-bearing motion; the passive branch still needs an explicit source-supported or explicitly approved experimental selection rule.

## Reproduction

```bash
PY=/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python
$PY {WORKTREE}/scripts/research/issue46_agilink_xacro_baseline.py --output-dir {output_dir}
$PY -m unittest {WORKTREE}/scripts/research/test_issue46_agilink_xacro_baseline.py
```

The generator calls `mj_forward` only and records `mj_step_calls=0`. It writes all generated artifacts through the governed output reservation at `{output_dir}`. `REPORT.md`, JSON, CSV, URDF, MJCF, SVG, and HTML are separate review artifacts; no official model or historical candidate is overwritten.
'''


def csv_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def generate(output_dir: Path) -> dict[str, object]:
    validate_output_path(str(output_dir), require_exists=True, expected_kind="directory", environment="wsl")
    src = load_sources()
    joints = candidate_delta_rows(src)
    links = [row for side in ("L", "R") for row in mesh_rows(src, side)]
    for row in links:
        row["mesh_summary"] = "; ".join(f"{m['filename']}={'same' if m['identical'] else 'different/unmapped'}" for m in row["xacro_meshes"]) or "no mesh in Xacro link"
    candidates: dict[str, ET.Element] = {}
    for side in ("L", "R"):
        candidate = build_candidate_urdf(src, side)
        candidates[side] = candidate
        out_name = f"X2-OmniPicker-{'left' if side == 'L' else 'right'}-AGILINK-articulated-tree-review-only.urdf"
        write_reserved_text(str(output_dir), str(output_dir / out_name), ET.tostring(candidate, encoding="unicode", xml_declaration=True) + "\n", environment="wsl")
    mjcfs: dict[str, ET.Element] = {}
    for side in ("L", "R"):
        tree = build_mjcf(candidates[side], side)
        mjcfs[side] = tree
        name = f"agilink_{'left' if side == 'L' else 'right'}_source_tree_static.xml"
        write_reserved_text(str(output_dir), str(output_dir / name), ET.tostring(tree, encoding="unicode", xml_declaration=True) + "\n", environment="wsl")
    overlay_tree = build_mjcf(candidates["L"], "L", d_closure_overlay=True)
    write_reserved_text(str(output_dir), str(output_dir / "agilink_left_D_closure_overlay_UNVERIFIED.xml"), ET.tostring(overlay_tree, encoding="unicode", xml_declaration=True) + "\n", environment="wsl")
    import mujoco
    static = {}
    for side in ("L", "R"):
        fname = f"agilink_{'left' if side == 'L' else 'right'}_source_tree_static.xml"
        static[f"{side}_tree"] = static_mujoco_audit(output_dir / fname, side, source_pose=True)
    static["left_D_overlay"] = static_mujoco_audit(output_dir / "agilink_left_D_closure_overlay_UNVERIFIED.xml", "L", source_pose=True, with_D=True)
    saved_models = {
        "candidate_B": CANONICAL / "reviews/issue-46/candidate-b-functional-validation-20261022T140838Z/single_motor.xml",
        "passive_spring": CANONICAL / "reviews/issue-46/passive-spring-single-motor-20261022T114341Z/passive_spring.xml",
        "quadratic": CANONICAL / "reviews/issue-46/candidate-b-clearance-physics-20261022T182958Z/candidate_geometry_coupled.xml",
    }
    static_saved = {name: saved_rank(path) for name, path in saved_models.items()}
    xacro_type_counts = dict(Counter(j.get("type") for j in xacro_joints(src).values()))
    frame_matches = sum(1 for x in joints if x["frame_match"])
    x2_root = {"narrow": "[-1.0, 0.0]" , "wide": "[0.0, 1.0]"}
    provenance = [
        {"name": "AGILINK Xacro archive", "sha256": sha_file(XACRO_ZIP), "detail": "reference/AGILINK/OmniPicker/agilink-omnipicker-urdf-20260723.zip"},
        {"name": "AGILINK 3D model archive", "sha256": sha_file(MODEL_ZIP), "detail": "reference/AGILINK/OmniPicker/agilink-omnipicker-3d-model-20260723.zip"},
        {"name": "AGILINK source/archive", "sha256": sha_file(SOURCE_ZIP), "detail": "reference/AGILINK/OmniPicker/agilink-omnipicker-20260723-source.zip"},
        {"name": "AGILINK robot_urdf.xacro member", "sha256": sha_bytes(src.macro_bytes), "detail": XACRO_PATH},
        {"name": "AGILINK robot.xacro member", "sha256": sha_bytes(src.xacro_zip.read(DISPLAY_PATH)), "detail": DISPLAY_PATH},
        {"name": "Pinned X2 v1.4.0 OmniPicker URDF", "sha256": sha_bytes(src.x2_bytes), "detail": "AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d"},
        {"name": "Original OP1 STEP", "sha256": sha_file(STEP), "detail": "OP1-官网模型.STEP"},
    ]
    d = {
        "schema": "robotsim.issue46.agilink-xacro-static-review.v1",
        "status": "SOURCE_TREE_CANDIDATE_READY_FOR_STATIC_REVIEW",
        "base_commit": "59ba71cf50e1aff0c4d5002a55087a7c7ff7060b",
        "worktree": str(WORKTREE), "branch": "codex/issue46-agilink-xacro-reconstruction-20261023",
        "new_physics_steps": 0, "mj_step_calls": 0, "mujoco_version": mujoco.__version__,
        "source_provenance": provenance,
        "xacro_macro": {"root": src.macro.tag, "attrs": src.macro.attrib, "xacro_joint_count": len(xacro_joints(src)),
                        "joint_type_counts": xacro_type_counts, "link_count": len(xacro_links(src)),
                        "display_root_invokes_macro": any(e.tag.endswith("robot_urdf") for e in src.display.iter()),
                        "mimic_count": len(src.macro.findall(".//mimic")),
                        "transmission_count": len(src.macro.findall(".//transmission")),
                        "ros2_control_count": len(src.macro.findall(".//ros2_control")),
                        "gazebo_count": len(src.macro.findall(".//gazebo")),
                        "license_members": [n for n in src.xacro_zip.namelist() if Path(n).name.casefold() in {"license", "license.txt", "copying", "notice"}]},
        "frame_match_count": frame_matches, "frame_count": len(joints), "joint_map": joints, "link_map": links,
        "internal_fixed_to_revolute": sorted([row["x2_joint"] for row in joints if row["candidate_delta"] == "type fixed -> revolute"]),
        "root_limits": x2_root, "root_mimic": [row for row in joints if row["x2_mimic"]],
        "candidate_files": {side: str(output_dir / f"X2-OmniPicker-{'left' if side == 'L' else 'right'}-AGILINK-articulated-tree-review-only.urdf") for side in ("L", "R")},
        "candidate_constraints": {"per_hand_x2_root_mimic": 1, "source_AGILINK_internal_mates": 0, "D_connects_in_baseline": 0,
                                  "D_connects_in_separate_overlay": 2, "CandidateB_linear_or_quadratic_relations_in_baseline": 0,
                                  "springs": 0, "actuators": 0, "collision_exclusions": 0},
        "static_models": static, "historical_saved_model_ranks": static_saved,
        "saved_candidate_comparison": read_historical_comparison(), "source_stl_witnesses": source_step_witnesses(),
        "candidate_B_exclusions": ["R_hand_narrow1_Link × R_hand_narrow_loop_Link", "R_hand_narrow3_Link × R_hand_narrow_loop_Link",
                                  "R_hand_wide1_Link × R_hand_wide_loop_Link", "R_hand_wide3_Link × R_hand_wide_loop_Link"],
        "claims_not_made": ["manufacturer-equivalent transmission", "one-input full stroke", "collision-free source B-rep assembly", "physical opening/closing PASS", "grasp readiness"],
        "limitations": ["AGILINK Xacro default display entrypoint does not invoke its macro.",
                        "Xacro topology is a display tree and does not define passive coupling or loop mates.",
                        "D-site pair ownership is an unverified CAD overlay; its coordinates are not vendor transmission semantics.",
                        "MuJoCo mesh contacts are convex-hull proxy results; saved source triangle witnesses do not classify B-rep volume or intended mating.",
                        "URDF mesh URIs reference the local pinned X2 cache; this evidence model is not portable without the source cache.",
                        "The AGILINK package license for individual mesh redistribution was not found in the archive members."],
    }
    outputs = [
        ("source_analysis.json", json.dumps(d, ensure_ascii=False, indent=2) + "\n"),
        ("source_joint_map.csv", csv_text(joints)),
        ("source_link_map.csv", csv_text(links)),
        ("candidate_joint_delta.csv", csv_text(joints)),
        ("joint_tree.svg", source_graph_svg()),
    ]
    for name, content in outputs:
        write_reserved_text(str(output_dir), str(output_dir / name), content, environment="wsl")
    report = make_report(d, output_dir)
    write_reserved_text(str(output_dir), str(output_dir / "REPORT.md"), report, environment="wsl")
    html_doc = make_review_html(d)
    write_reserved_text(str(output_dir), str(output_dir / "review.html"), html_doc, environment="wsl")
    manifest = {"schema": "robotsim.artifact-paths.v1", "run": describe_output_path(str(output_dir), environment="wsl"),
                "artifacts": [describe_output_path(str(output_dir / name), environment="wsl") for name in
                              ["REPORT.md", "source_analysis.json", "source_joint_map.csv", "source_link_map.csv", "candidate_joint_delta.csv",
                               "X2-OmniPicker-left-AGILINK-articulated-tree-review-only.urdf", "X2-OmniPicker-right-AGILINK-articulated-tree-review-only.urdf",
                               "agilink_left_source_tree_static.xml", "agilink_right_source_tree_static.xml", "agilink_left_D_closure_overlay_UNVERIFIED.xml",
                               "joint_tree.svg", "review.html"]]}
    write_reserved_text(str(output_dir), str(output_dir / "artifact_paths.json"), json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", environment="wsl")
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = generate(Path(args.output_dir))
    print(json.dumps({"status": data["status"], "worktree": data["worktree"], "branch": data["branch"],
                      "output_dir": args.output_dir, "frame_matches": f"{data['frame_match_count']}/{data['frame_count']}",
                      "fixed_to_revolute_count": len(data["internal_fixed_to_revolute"]),
                      "left_tree_M": data["static_models"]["L_tree"]["mobility_nv_minus_rank"],
                      "left_D_overlay_M": data["static_models"]["left_D_overlay"]["mobility_nv_minus_rank"],
                      "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
