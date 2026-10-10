from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "mujoco"))


SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
DEFAULT_VENDOR_ROOT = Path(
    "/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf"
)
URDF_RELATIVE = Path("X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf")
HAND_SIDES = ("L", "R")
FINGER_FAMILIES = ("thumb", "index", "middle", "ring", "pinky")
MIMIC_SOLREF_DIRECT = [-10000.0, -200.0]
SIMULATION_ONLY_KP = 18.0
SIMULATION_ONLY_KV = 2.4
GEOMETRY_TOLERANCE = 2e-6
CANONICAL_SCENE_Z_SHIFT_M = -0.8
CANONICAL_SCENE_Y_SHIFT_M = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_vendor(vendor_root: Path) -> Path:
    actual = git_output(vendor_root, "rev-parse", "HEAD")
    dirty = git_output(vendor_root, "status", "--porcelain")
    if actual != SOURCE_PIN:
        raise RuntimeError(f"pinned vendor revision mismatch: expected {SOURCE_PIN}, got {actual}")
    if dirty:
        raise RuntimeError("pinned vendor checkout is dirty; refusing to read modified upstream files")
    urdf = vendor_root / URDF_RELATIVE
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    return urdf


def number_list(value: str | None, size: int, default: str) -> list[float]:
    result = np.fromstring(default if value is None else value, sep=" ", dtype=float)
    if result.size != size:
        raise ValueError(f"expected {size} values, got {value!r}")
    return result.tolist()


def quaternion_from_rpy(rpy: list[float]) -> list[float]:
    result = np.zeros(4, dtype=float)
    mujoco.mju_euler2Quat(result, np.asarray(rpy, dtype=float), "XYZ")
    return result.tolist()


def resolve_mesh(urdf: Path, filename: str) -> Path:
    tail = filename.split("://", 1)[-1]
    marker = "X2_URDF-v1.4.0/"
    if marker in tail:
        tail = tail.split(marker, 1)[1]
    else:
        parts = Path(tail).parts
        mesh_index = next((i for i, part in enumerate(parts) if part == "meshes"), None)
        if mesh_index is not None:
            tail = str(Path(*parts[mesh_index:]))
    candidates = [urdf.parent / tail, urdf.parent.parent / tail]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    basename_matches = list(urdf.parent.rglob(Path(tail).name))
    if len(basename_matches) == 1 and basename_matches[0].is_file():
        return basename_matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve pinned mesh {filename!r} from {urdf}")


def source_geom_record(node: ET.Element, role: str, link_name: str, urdf: Path) -> dict[str, Any]:
    origin = node.find("origin")
    geometry = node.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError(f"invalid {role} geometry on {link_name}")
    shape_node = geometry[0]
    record: dict[str, Any] = {
        "role": role,
        "shape": shape_node.tag,
        "pos": number_list(None if origin is None else origin.get("xyz"), 3, "0 0 0"),
        "quat": quaternion_from_rpy(
            number_list(None if origin is None else origin.get("rpy"), 3, "0 0 0")
        ),
    }
    if shape_node.tag == "mesh":
        filename = shape_node.get("filename")
        if not filename:
            raise ValueError(f"mesh without filename on {link_name}")
        path = resolve_mesh(urdf, filename)
        record.update(
            mesh_path=str(path),
            mesh_sha256=sha256(path),
            mesh_scale=number_list(shape_node.get("scale"), 3, "1 1 1"),
            source_reference=filename,
        )
    elif shape_node.tag == "box":
        record["size"] = [x / 2.0 for x in number_list(shape_node.get("size"), 3, "")]
    elif shape_node.tag == "cylinder":
        record["size"] = [float(shape_node.get("radius")), float(shape_node.get("length")) / 2.0]
    elif shape_node.tag == "sphere":
        record["size"] = [float(shape_node.get("radius"))]
    else:
        raise ValueError(f"unsupported official URDF geometry {shape_node.tag!r} on {link_name}")
    record["link"] = link_name
    return record


def source_inventory(source_root: ET.Element, urdf: Path) -> dict[str, Any]:
    links: dict[str, Any] = {}
    for link in source_root.findall("link"):
        name = str(link.get("name"))
        inertial = link.find("inertial")
        inertial_record = None
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            tensor = inertial.find("inertia")
            if mass is None or tensor is None:
                raise ValueError(f"incomplete source inertial on {name}")
            inertial_record = {
                "mass": float(mass.get("value")),
                "pos": number_list(None if origin is None else origin.get("xyz"), 3, "0 0 0"),
                "quat": quaternion_from_rpy(
                    number_list(None if origin is None else origin.get("rpy"), 3, "0 0 0")
                ),
                "tensor": [float(tensor.get(k, "0")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")],
            }
        links[name] = {
            "name": name,
            "inertial": inertial_record,
            "visuals": [source_geom_record(g, "visual", name, urdf) for g in link.findall("visual")],
            "collisions": [source_geom_record(g, "collision", name, urdf) for g in link.findall("collision")],
        }

    joints: dict[str, Any] = {}
    mimic: dict[str, Any] = {}
    for joint in source_root.findall("joint"):
        name = str(joint.get("name"))
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        relation = joint.find("mimic")
        record: dict[str, Any] = {
            "name": name,
            "type": str(joint.get("type")),
            "parent": None if parent is None else str(parent.get("link")),
            "child": None if child is None else str(child.get("link")),
            "pos": number_list(None if origin is None else origin.get("xyz"), 3, "0 0 0"),
            "quat": quaternion_from_rpy(
                number_list(None if origin is None else origin.get("rpy"), 3, "0 0 0")
            ),
            "axis": None if axis is None else number_list(axis.get("xyz"), 3, "1 0 0"),
            "range": None if limit is None else [float(limit.get("lower")), float(limit.get("upper"))],
            "effort": None if limit is None else float(limit.get("effort", "0")),
            "velocity": None if limit is None else float(limit.get("velocity", "0")),
            "mimic": None if relation is None else {
                "driver": str(relation.get("joint")),
                "multiplier": float(relation.get("multiplier", "1")),
                "offset": float(relation.get("offset", "0")),
            },
        }
        joints[name] = record
        if record["mimic"] is not None:
            mimic[name] = record["mimic"]
    return {"links": links, "joints": joints, "mimic": mimic}


def hand_subtree(source: dict[str, Any], side: str) -> set[str]:
    anchor = f"{side}_palm"
    included = {anchor}
    changed = True
    while changed:
        changed = False
        for joint in source["joints"].values():
            if joint["parent"] in included and joint["child"] not in included:
                included.add(joint["child"])
                changed = True
    return included


def build_augmented_urdf(source_urdf: Path, source_root: ET.Element, out_dir: Path) -> Path:
    root = copy.deepcopy(source_root)
    for extension in root.findall("mujoco"):
        root.remove(extension)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        {"discardvisual": "false", "strippath": "false", "fusestatic": "false"},
    )
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("official URDF contains a mesh element without a filename")
        mesh.set("filename", str(resolve_mesh(source_urdf, filename)))
    output = out_dir / "x2_ultra_dual_native_omnihand.urdf"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def normalized_record(record: dict[str, Any]) -> str:
    fields = ("role", "shape", "pos", "quat", "size", "mesh_path", "mesh_scale")

    def rounded(value: Any) -> Any:
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, float):
            return round(value, 6)
        return value

    return json.dumps({key: rounded(record.get(key)) for key in fields}, sort_keys=True, separators=(",", ":"))


def validate_morphology_records(
    source_links: dict[str, Any],
    compiled_links: dict[str, Any],
    source_joints: dict[str, Any],
    compiled_joints: dict[str, Any],
    source_mimic: dict[str, Any],
    compiled_mimic: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for name, link in source_links.items():
        compiled = compiled_links.get(name)
        if compiled is None:
            errors.append(f"missing compiled body/link: {name}")
            continue
        for role in ("visuals", "collisions"):
            expected = Counter(normalized_record(item) for item in link[role])
            observed = Counter(normalized_record(item) for item in compiled[role])
            if expected != observed:
                missing = list((expected - observed).elements())
                extra = list((observed - expected).elements())
                errors.append(f"{name} {role} parity mismatch: missing={missing[:2]} extra={extra[:2]}")
    for name, joint in source_joints.items():
        compiled = compiled_joints.get(name)
        if compiled is None:
            errors.append(f"missing compiled joint/topology record: {name}")
            continue
        if joint["type"] != compiled["type"]:
            errors.append(f"joint type mismatch: {name}")
        if joint["parent"] != compiled["parent"] or joint["child"] != compiled["child"]:
            errors.append(f"joint attachment mismatch: {name}")
        for key in ("pos", "quat"):
            if not np.allclose(joint[key], compiled[key], atol=GEOMETRY_TOLERANCE, rtol=0):
                errors.append(f"joint origin {key} mismatch: {name}")
        if joint["type"] == "revolute":
            if compiled["axis"] is None or not np.allclose(joint["axis"], compiled["axis"], atol=GEOMETRY_TOLERANCE, rtol=0):
                errors.append(f"joint axis mismatch: {name}")
        if joint["range"] is not None:
            if compiled["range"] is None or not np.allclose(joint["range"], compiled["range"], atol=GEOMETRY_TOLERANCE, rtol=0):
                errors.append(f"joint range mismatch: {name}")
    if set(source_mimic) != set(compiled_mimic):
        errors.append(
            f"mimic relation set mismatch: missing={sorted(set(source_mimic)-set(compiled_mimic))} "
            f"extra={sorted(set(compiled_mimic)-set(source_mimic))}"
        )
    for follower, relation in source_mimic.items():
        observed = compiled_mimic.get(follower)
        if observed is None:
            continue
        if relation["driver"] != observed["driver"] or not math.isclose(
            relation["multiplier"], observed["multiplier"], abs_tol=1e-8
        ) or not math.isclose(relation["offset"], observed["offset"], abs_tol=1e-8):
            errors.append(f"mimic relation mismatch: {follower}")
    for side in HAND_SIDES:
        links = {name for name in source_links if name.startswith(f"{side}_")}
        for family in FINGER_FAMILIES:
            family_links = [name for name in links if name.lower().startswith(f"{side.lower()}_{family}_")]
            if not family_links:
                errors.append(f"missing {side} {family} source link family")
                continue
            source_meshes = sum(
                item["shape"] == "mesh"
                for name in family_links
                for item in source_links[name]["visuals"]
            )
            compiled_meshes = sum(
                item["shape"] == "mesh"
                for name in family_links
                for item in compiled_links.get(name, {}).get("visuals", [])
            )
            if source_meshes == 0 or compiled_meshes == 0:
                errors.append(f"{side} {family} visual mesh family missing")
    return errors


def add_simulation_only_actuators(spec: mujoco.MjSpec, source: dict[str, Any]) -> list[str]:
    followers = set(source["mimic"])
    actuated: list[str] = []
    spec_joint_names = {str(joint.name) for joint in spec.joints if joint.name}
    for name, source_joint in source["joints"].items():
        if source_joint["type"] != "revolute" or name in followers:
            continue
        if name not in spec_joint_names:
            raise RuntimeError(f"source independent hinge was not imported: {name}")
        joint = next(j for j in spec.joints if j.name == name)
        effort = float(source_joint["effort"])
        if effort <= 0:
            raise RuntimeError(f"source hinge has no usable effort limit: {name}")
        actuator = spec.add_actuator(
            name=f"simonly_position_{name}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            ctrllimited=True,
            ctrlrange=list(joint.range),
            forcelimited=True,
            forcerange=[-effort, effort],
        )
        actuator.set_to_position(kp=SIMULATION_ONLY_KP, kv=SIMULATION_ONLY_KV)
        actuated.append(name)
    return actuated


def add_source_mimics(spec: mujoco.MjSpec, source: dict[str, Any]) -> None:
    joint_nodes = {str(joint.name): joint for joint in spec.joints if joint.name}
    for follower, relation in source["mimic"].items():
        driver = relation["driver"]
        if follower not in joint_nodes or driver not in joint_nodes:
            raise RuntimeError(f"missing imported mimic joint: {driver} -> {follower}")
        driver_ref = float(joint_nodes[driver].ref)
        follower_ref = float(joint_nodes[follower].ref)
        polynomial = [
            relation["offset"] + relation["multiplier"] * driver_ref - follower_ref,
            relation["multiplier"], 0.0, 0.0, 0.0,
        ]
        equality = spec.add_equality(
            name=f"urdf_mimic_{follower}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=follower,
            name2=driver,
            data=polynomial + [0.0] * 6,
        )
        equality.solref = list(MIMIC_SOLREF_DIRECT)


def compiled_geom_record(
    model: mujoco.MjModel,
    geom_id: int,
    link_name: str,
    mesh_assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contype = int(model.geom_contype[geom_id])
    conaffinity = int(model.geom_conaffinity[geom_id])
    role = "visual" if contype == 0 and conaffinity == 0 else "collision"
    geom_type = int(model.geom_type[geom_id])
    geom_types = {
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    }
    if geom_type not in geom_types:
        raise RuntimeError(f"unsupported compiled geom type {geom_type} on {link_name}")
    record: dict[str, Any] = {
        "role": role,
        "shape": geom_types[geom_type],
        "pos": np.asarray(model.geom_pos[geom_id], dtype=float).tolist(),
        "quat": np.asarray(model.geom_quat[geom_id], dtype=float).tolist(),
        "link": link_name,
    }
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        asset = mesh_assets.get(str(mesh_name))
        if asset is None:
            raise RuntimeError(f"compiled mesh asset missing for geom on {link_name}")
        record["mesh_path"] = str(Path(asset["file"]).resolve())
        record["mesh_scale"] = [float(x) for x in asset.get("scale", [1, 1, 1])]
    else:
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = size[:3]
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            size = size[:2]
        else:
            size = size[:1]
        record["size"] = size.tolist()
    return record


def source_inertial_tensor(record: dict[str, Any]) -> np.ndarray:
    ixx, ixy, ixz, iyy, iyz, izz = record["tensor"]
    local = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
    quat = np.asarray(record["quat"], dtype=float)
    rotation = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    return rotation @ local @ rotation.T


def compiled_inertial_tensor(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    rotation = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T


def build_compiled_inventory(
    model: mujoco.MjModel,
    spec: mujoco.MjSpec,
    source: dict[str, Any],
    model_xml_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    xml_root = ET.parse(model_xml_path).getroot()
    mesh_assets = {
        str(asset.get("name")): {
            "file": str(asset.get("file")),
            "scale": number_list(asset.get("scale"), 3, "1 1 1"),
        }
        for asset in xml_root.findall("asset/mesh")
    }
    compiled_links: dict[str, Any] = {
        name: {"name": name, "visuals": [], "collisions": []}
        for name in source["links"]
    }
    body_ids: dict[str, int] = {}
    for name in source["links"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        body_ids[name] = body_id
    compiled_body_transforms: dict[str, Any] = {}

    def visit_bodies(parent_node: ET.Element, parent_name: str | None) -> None:
        for body in parent_node.findall("body"):
            link_name = body.get("name")
            if not link_name:
                continue
            if link_name in compiled_links:
                compiled_body_transforms[link_name] = {
                    "parent": parent_name,
                    "pos": number_list(body.get("pos"), 3, "0 0 0"),
                    "quat": number_list(body.get("quat"), 4, "1 0 0 0"),
                }
                for geom in body.findall("geom"):
                    geom_type = str(geom.get("type", "sphere"))
                    role = "visual" if int(geom.get("contype", "1")) == 0 and int(geom.get("conaffinity", "1")) == 0 else "collision"
                    record: dict[str, Any] = {
                        "role": role,
                        "shape": geom_type,
                        "pos": number_list(geom.get("pos"), 3, "0 0 0"),
                        "quat": number_list(geom.get("quat"), 4, "1 0 0 0"),
                        "link": link_name,
                    }
                    if geom_type == "mesh":
                        asset = mesh_assets.get(str(geom.get("mesh")))
                        if asset is None:
                            raise RuntimeError(f"MjSpec geom references a missing mesh asset on {link_name}")
                        record["mesh_path"] = str(Path(asset["file"]).resolve())
                        record["mesh_scale"] = asset["scale"]
                    else:
                        size_count = 2 if geom_type == "cylinder" else 1 if geom_type == "sphere" else 3
                        record["size"] = number_list(geom.get("size"), size_count, "0 " * (size_count - 1) + "0")
                    compiled_links[link_name]["visuals" if role == "visual" else "collisions"].append(record)
            visit_bodies(body, link_name)

    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("generated MuJoCo spec has no worldbody")
    visit_bodies(worldbody, None)
    compiled_joints: dict[str, Any] = {}
    for name, joint in source["joints"].items():
        child = joint["child"]
        body_record = compiled_body_transforms.get(child)
        if body_record is None:
            continue
        record: dict[str, Any] = {
            "name": name,
            "type": joint["type"],
            "parent": body_record["parent"],
            "child": child,
            "pos": body_record["pos"],
            "quat": body_record["quat"],
            "axis": None,
            "range": None,
        }
        if joint["type"] == "revolute":
            body_xml = next((b for b in xml_root.findall(".//body") if b.get("name") == child), None)
            joint_xml = None if body_xml is None else next(
                (j for j in body_xml.findall("joint") if j.get("name") == name), None
            )
            if joint_xml is not None:
                record["type"] = "revolute" if joint_xml.get("type", "hinge") == "hinge" else "unexpected"
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    record["axis"] = np.asarray(model.jnt_axis[joint_id], dtype=float).tolist()
                    record["range"] = np.asarray(model.jnt_range[joint_id], dtype=float).tolist()
        compiled_joints[name] = record

    compiled_mimic: dict[str, Any] = {}
    for eq_id in range(model.neq):
        if int(model.eq_type[eq_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
        if not name or not name.startswith("urdf_mimic_"):
            continue
        follower = str(name.removeprefix("urdf_mimic_"))
        driver = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[eq_id]))
        data = np.asarray(model.eq_data[eq_id], dtype=float)
        compiled_mimic[follower] = {
            "driver": str(driver),
            "multiplier": float(data[1]),
            "offset": float(data[0]),
        }

    inertial_comparison: dict[str, Any] = {}
    for name, source_link in source["links"].items():
        expected = source_link["inertial"]
        body_id = body_ids.get(name)
        if expected is None or body_id is None:
            continue
        actual_tensor = compiled_inertial_tensor(model, body_id)
        expected_tensor = source_inertial_tensor(expected)
        record = {
            "source_mass_kg": expected["mass"],
            "compiled_mass_kg": float(model.body_mass[body_id]),
            "source_inertia_kg_m2": expected_tensor.tolist(),
            "compiled_inertia_kg_m2": actual_tensor.tolist(),
            "source_com_m": expected["pos"],
            "compiled_com_m": np.asarray(model.body_ipos[body_id], dtype=float).tolist(),
        }
        inertial_comparison[name] = record

    return compiled_links, compiled_joints, {
        "body_ids": body_ids,
        "mesh_assets": mesh_assets,
        "inertials": inertial_comparison,
        "compiled_body_transforms": compiled_body_transforms,
        "model_xml_path": str(model_xml_path),
    }


def validate_inertials(source: dict[str, Any], compiled_details: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, actual in compiled_details["inertials"].items():
        expected = source["links"][name]["inertial"]
        mass_scale = max(1.0, abs(expected["mass"]))
        if abs(actual["compiled_mass_kg"] - expected["mass"]) > 2e-7 * mass_scale:
            errors.append(f"mass mismatch on {name}")
        if not np.allclose(actual["source_com_m"], actual["compiled_com_m"], atol=2e-6, rtol=0):
            errors.append(f"inertial origin mismatch on {name}")
        if not np.allclose(actual["source_inertia_kg_m2"], actual["compiled_inertia_kg_m2"], atol=2e-7, rtol=2e-6):
            errors.append(f"inertia tensor mismatch on {name}")
    if len(compiled_details["inertials"]) != sum(link["inertial"] is not None for link in source["links"].values()):
        errors.append("compiled/source inertial record count differs")
    return errors


def geom_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(json.loads(normalized_record(record)).items())


def parse_mimic_equalities(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for eq_id in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
        if name and name.startswith("urdf_mimic_"):
            follower_id = int(model.eq_obj1id[eq_id])
            driver_id = int(model.eq_obj2id[eq_id])
            follower = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, follower_id)
            driver = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driver_id)
            multiplier = float(model.eq_data[eq_id, 1])
            raw_constant = float(model.eq_data[eq_id, 0])
            driver_qpos0 = float(model.qpos0[int(model.jnt_qposadr[driver_id])])
            follower_qpos0 = float(model.qpos0[int(model.jnt_qposadr[follower_id])])
            result[str(name.removeprefix("urdf_mimic_"))] = {
                "driver": str(driver),
                "multiplier": multiplier,
                "offset": raw_constant - multiplier * driver_qpos0 + follower_qpos0,
                "active": bool(model.eq_active0[eq_id]),
                "name1": str(follower),
                "name2": str(driver),
                "polynomial": np.asarray(model.eq_data[eq_id], dtype=float).tolist(),
            }
    return result


def configure_visual_groups(
    model: mujoco.MjModel,
    hand_links: set[str],
    robot_links: set[str],
) -> dict[str, int]:
    counts = {
        "robot_visual": 0,
        "hand_visual": 0,
        "hand_collision": 0,
        "robot_collision": 0,
        "environment": 0,
        "floor": 0,
    }
    for geom_id in range(model.ngeom):
        visual = int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name == "robotsim_render_floor":
            model.geom_group[geom_id] = 5
            counts["floor"] += 1
        elif body_name in robot_links:
            if visual and body_name in hand_links:
                model.geom_group[geom_id] = 1
                counts["hand_visual"] += 1
            elif visual:
                model.geom_group[geom_id] = 0
                counts["robot_visual"] += 1
            elif body_name in hand_links:
                model.geom_group[geom_id] = 2
                model.geom_rgba[geom_id] = [0.06, 0.72, 0.92, 0.38]
                counts["hand_collision"] += 1
            else:
                model.geom_group[geom_id] = 3
                counts["robot_collision"] += 1
        else:
            model.geom_group[geom_id] = 4
            counts["environment"] += 1
    return counts


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(name)
    data.qpos[int(model.jnt_qposadr[joint_id])] = value


def render_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source: dict[str, Any],
    hand_fraction: float,
    arm_fraction: float = 0.0,
    move_arm: bool = False,
) -> None:
    data.qpos[:] = model.qpos0
    mimic_followers = set(source["mimic"])
    arm_joint_names: list[str] = []
    for name, joint in source["joints"].items():
        if joint["type"] != "revolute":
            continue
        if any(name.startswith(f"{side}_") for side in HAND_SIDES):
            if name in mimic_followers:
                continue
            lower, upper = joint["range"]
            set_joint_qpos(model, data, name, lower + (upper - lower) * hand_fraction)
        elif move_arm and any(token in name.lower() for token in ("shoulder", "elbow", "wrist")):
            lower, upper = joint["range"]
            width = upper - lower
            ref_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos0 = float(model.qpos0[int(model.jnt_qposadr[ref_id])])
            displacement = min(0.18, width * 0.12)
            direction = 1.0 if "pitch" in name.lower() or "flex" in name.lower() else -1.0
            value = float(np.clip(qpos0 + direction * displacement * arm_fraction, lower, upper))
            set_joint_qpos(model, data, name, value)
            if abs(value - qpos0) > 1e-6:
                arm_joint_names.append(name)
    for follower, relation in source["mimic"].items():
        driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, relation["driver"])
        driver_qpos = float(data.qpos[int(model.jnt_qposadr[driver_id])])
        set_joint_qpos(
            model,
            data,
            follower,
            relation["offset"] + relation["multiplier"] * driver_qpos,
        )
    mujoco.mj_forward(model, data)


def camera_for_bounds(
    model: mujoco.MjModel,
    body_ids: dict[str, int],
    azimuth: float,
    elevation: float,
    include_environment: bool = False,
) -> mujoco.MjvCamera:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if include_environment or name in body_ids:
            points.append(np.asarray(data.geom_xpos[geom_id], dtype=float))
    if not points:
        raise RuntimeError("no compiled X2 geoms are available to frame a camera")
    coordinates = np.stack(points)
    selected = [
        geom_id
        for geom_id in range(model.ngeom)
        if include_environment
        or mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ) in body_ids
    ]
    radii = np.asarray(model.geom_rbound[selected], dtype=float)[:, None]
    lower = (coordinates - radii).min(axis=0)
    upper = (coordinates + radii).max(axis=0)
    lookat = (lower + upper) / 2.0
    radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.25)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = radius / max(math.sin(math.radians(22.0)), 0.2)
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def camera_at(model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = point
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_images(
    model: mujoco.MjModel,
    source: dict[str, Any],
    details: dict[str, Any],
    out_dir: Path,
    with_environment: bool,
) -> dict[str, str]:
    model.vis.global_.offwidth = 1440
    model.vis.global_.offheight = 900
    model.vis.headlight.ambient[:] = [0.24, 0.24, 0.24]
    model.vis.headlight.diffuse[:] = [0.48, 0.48, 0.48]
    model.vis.headlight.specular[:] = [0.06, 0.06, 0.06]
    hand_links = set().union(*(hand_subtree(source, side) for side in HAND_SIDES))
    group_counts = configure_visual_groups(model, hand_links, set(source["links"]))
    data = mujoco.MjData(model)
    render_pose(model, data, source, hand_fraction=0.05)
    body_ids = details["body_ids"]
    outputs: dict[str, str] = {}
    with mujoco.Renderer(model, height=900, width=1440) as renderer:
        view = mujoco.MjvOption()
        mujoco.mjv_defaultOption(view)

        def save(
            name: str,
            camera: mujoco.MjvCamera,
            groups: tuple[bool, bool, bool, bool, bool, bool] = (True, True, False, False, False, True),
        ) -> None:
            view.geomgroup[:] = 0
            for group, enabled in enumerate(groups):
                view.geomgroup[group] = enabled
            renderer.update_scene(data, camera=camera, scene_option=view)
            path = out_dir / name
            imageio.imwrite(path, renderer.render())
            outputs[name] = str(path)

        full_views = (
            ("full_body_front.png", 180.0, -5.0),
            ("full_body_back.png", 0.0, -5.0),
            ("full_body_side.png", 270.0, -2.0),
        )
        for name, azimuth, elevation in full_views:
            save(
                name,
                camera_for_bounds(model, body_ids, azimuth, elevation),
                (True, True, False, False, False, False),
            )

        hand_body_ids = {name: body_id for name, body_id in body_ids.items() if name in hand_links}
        both_hands_camera = camera_for_bounds(model, hand_body_ids, 180, -5)
        both_hands_camera.distance *= 0.68
        render_pose(model, data, source, hand_fraction=0.05)
        save("both_hands_open.png", both_hands_camera, (False, True, False, False, False, False))
        render_pose(model, data, source, hand_fraction=0.68)
        save("both_hands_close.png", both_hands_camera, (False, True, False, False, False, False))
        render_pose(model, data, source, hand_fraction=0.05)

        palm_points = []
        for side in HAND_SIDES:
            palm_id = body_ids[f"{side}_palm"]
            palm_points.append(np.asarray(data.xpos[palm_id], dtype=float))
            tip_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{family}_tip")
                for family in FINGER_FAMILIES
            ]
            if any(body_id < 0 for body_id in tip_ids):
                raise RuntimeError(f"official {side} hand is missing a finger-tip body")
            tip_center = np.mean([data.xpos[body_id] for body_id in tip_ids], axis=0)
            hand_center = (palm_points[-1] + tip_center) / 2.0
            camera = camera_at(model, data, hand_center, 0.34, 90 if side == "L" else 270, 0)
            render_pose(model, data, source, hand_fraction=0.05)
            save(f"{side.lower()}_hand_open.png", camera, (False, True, False, False, False, False))
            save(f"{side.lower()}_hand_collision_only.png", camera, (False, False, True, False, False, False))
            render_pose(model, data, source, hand_fraction=0.68)
            save(f"{side.lower()}_hand_close.png", camera, (False, True, False, False, False, False))
            save(
                f"{side.lower()}_hand_visual_collision_overlay.png",
                camera,
                (False, True, True, False, False, False),
            )
            render_pose(model, data, source, hand_fraction=0.05)
        overhead_camera = camera_for_bounds(model, hand_body_ids, 0, 55)
        save(
            "both_hands_from_above.png",
            overhead_camera,
            (False, True, False, False, False, False),
        )
        save(
            "both_hands_visual_collision_overlay.png",
            overhead_camera,
            (False, True, True, False, False, False),
        )

        if with_environment:
            full = camera_for_bounds(model, body_ids, 0, -8, include_environment=True)
            full.distance *= 1.3
            save("full_robot_table_bottle.png", full, (True, True, False, False, True, False))

        frames: list[np.ndarray] = []
        fps = 30
        camera = camera_for_bounds(model, body_ids, 180, -5)
        sequence = [(0.05, 0.05, 25), (0.05, 0.68, 45), (0.68, 0.68, 20), (0.68, 0.05, 45)]
        for start_fraction, hand_fraction, count in sequence:
            for index in range(count):
                interpolation = index / max(count - 1, 1)
                value = start_fraction + (hand_fraction - start_fraction) * interpolation
                arm_value = math.sin(math.pi * interpolation) * 0.65 if hand_fraction > 0.05 else 0.0
                render_pose(model, data, source, value, arm_value, move_arm=True)
                renderer.update_scene(
                    data,
                    camera=camera,
                    scene_option=view,
                )
                frames.append(renderer.render().copy())
        video_path = out_dir / "dual_omnihands_open_close.mp4"
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
        outputs[video_path.name] = str(video_path)
    return {"files": outputs, "geom_group_counts": group_counts}


def build_model(urdf: Path, source: dict[str, Any], out_dir: Path, with_environment: bool) -> tuple[mujoco.MjModel, mujoco.MjSpec, list[str]]:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 100
    spec.option.tolerance = 1e-8
    spec.option.gravity[:] = [0.0, 0.0, -9.81]
    add_source_mimics(spec, source)
    actuated = add_simulation_only_actuators(spec, source)
    spec.worldbody.add_light(
        name="robotsim_diagnostic_key_light",
        pos=[2.0, -2.0, 3.5],
        dir=[-0.5, 0.5, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_geom(
        name="robotsim_render_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, -0.81],
        size=[0.0, 0.0, 0.05],
        rgba=[0.22, 0.25, 0.28, 1.0],
        contype=0,
        conaffinity=0,
        group=5,
    )
    if with_environment:
        from canonical_manipulation_assets import (
            CANONICAL_X2_BOTTLE_START_BODY_POS,
            add_g1_canonical_table,
        )
        from issue46_native_omnihand_minimal_grasp import add_bottle

        add_g1_canonical_table(spec)
        table = next(body for body in spec.worldbody.bodies if body.name == "m0_table")
        table.pos[1] += CANONICAL_SCENE_Y_SHIFT_M
        table.pos[2] += CANONICAL_SCENE_Z_SHIFT_M
        bottle_position = list(CANONICAL_X2_BOTTLE_START_BODY_POS)
        bottle_position[1] += CANONICAL_SCENE_Y_SHIFT_M
        bottle_position[2] += CANONICAL_SCENE_Z_SHIFT_M
        add_bottle(spec, tuple(bottle_position))
    model = spec.compile()
    model.opt.timestep = 0.002
    model.opt.gravity[:] = [0.0, 0.0, -9.81]
    xml_text = spec.to_xml()
    model_xml_path = out_dir / "x2_ultra_dual_native_omnihand.mjcf.xml"
    model_xml_path.write_text(xml_text, encoding="utf-8")
    return model, spec, actuated


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and audit the pinned complete dual-OmniHand X2 morphology.")
    parser.add_argument("--vendor-root", type=Path, default=Path(os.environ.get("AGIBOT_X2_VENDOR_ROOT", DEFAULT_VENDOR_ROOT)))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("ISSUE46_DUAL_OMNIHAND_EVIDENCE_DIR", "/tmp/issue46-dual-omnihand-platform")),
    )
    parser.add_argument("--no-render", action="store_true", help="compile and write inventories without GPU screenshots/video")
    parser.add_argument("--no-environment", action="store_true", help="omit canonical table and free bottle from the rendering scene")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urdf = verify_vendor(args.vendor_root)
    source_root = ET.parse(urdf).getroot()
    source = source_inventory(source_root, urdf)
    for side in HAND_SIDES:
        included = hand_subtree(source, side)
        for family in FINGER_FAMILIES:
            if not any(name.lower().startswith(f"{side.lower()}_{family}_") for name in included):
                raise RuntimeError(f"official {side} hand subtree is missing {family} links")
    generated_urdf = build_augmented_urdf(urdf, source_root, args.output)
    model, spec, actuated = build_model(generated_urdf, source, args.output, not args.no_environment)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    compiled_links, compiled_joints, details = build_compiled_inventory(
        model, spec, source, args.output / "x2_ultra_dual_native_omnihand.mjcf.xml"
    )
    errors = validate_morphology_records(
        source["links"], compiled_links, source["joints"], compiled_joints, source["mimic"],
        {name: {"driver": name2, "multiplier": record["multiplier"], "offset": record["offset"]}
         for name, (name2, record) in ((f, (v["driver"], v)) for f, v in source["mimic"].items())},
    )
    errors.extend(validate_inertials(source, details))
    equality_data = parse_mimic_equalities(model)
    if len(equality_data) != len(source["mimic"]):
        errors.append(f"compiled mimic equality count {len(equality_data)} != source {len(source['mimic'])}")
    compiled_mimic = {
        name: {key: record[key] for key in ("driver", "multiplier", "offset")}
        for name, record in equality_data.items()
    }
    if set(source["mimic"]) != set(compiled_mimic):
        errors.append("compiled mimic follower set differs from official source")
    for follower, relation in source["mimic"].items():
        observed = compiled_mimic.get(follower)
        if observed is None:
            continue
        if relation["driver"] != observed["driver"] or not math.isclose(
            relation["multiplier"], observed["multiplier"], abs_tol=1e-8
        ) or not math.isclose(relation["offset"], observed["offset"], abs_tol=1e-8):
            errors.append(f"compiled mimic mapping differs from source: {follower}")
    for follower, record in equality_data.items():
        if not record["active"] or record["name1"] != follower or record["name2"] != source["mimic"][follower]["driver"]:
            errors.append(f"compiled mimic equality endpoint/activation mismatch: {follower}")
    expected_nq = 63 + (7 if not args.no_environment else 0)
    expected_nv = 63 + (6 if not args.no_environment else 0)
    if model.nq != expected_nq or model.nv != expected_nv or model.nu != len(actuated) or model.neq != 12:
        errors.append(
            f"unexpected compiled dimensions nq/nv/nu/neq={model.nq}/{model.nv}/{model.nu}/{model.neq}"
        )
    if len(source["links"]) != 86 or len(source["joints"]) != 85:
        errors.append("source pin no longer has the audited full-X2 link/joint dimensions")

    visual_count = sum(len(link["visuals"]) for link in compiled_links.values())
    collision_count = sum(len(link["collisions"]) for link in compiled_links.values())
    source_visual_count = sum(len(link["visuals"]) for link in source["links"].values())
    source_collision_count = sum(len(link["collisions"]) for link in source["links"].values())
    mesh_visual_count = sum(
        item["shape"] == "mesh" for link in compiled_links.values() for item in link["visuals"]
    )
    body_mass = float(sum(float(model.body_mass[details["body_ids"][name]]) for name in source["links"]))
    inventory = {
        "status": "PASS" if not errors else "FAIL",
        "source": {
            "repository": "AgibotTech/agibot_x2_urdf",
            "commit": SOURCE_PIN,
            "urdf": str(urdf),
            "urdf_sha256": sha256(urdf),
            "vendor_worktree_clean": True,
        },
        "compiler": {
            "mujoco": mujoco.__version__,
            "discardvisual": False,
            "strippath": False,
            "fusestatic": False,
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "timestep_s": 0.002,
            "integrator": "implicitfast",
            "actuators": "SIMULATION_ONLY bounded joint-position PD; official URDF has zero transmissions",
            "actuator_kp": SIMULATION_ONLY_KP,
            "actuator_kv": SIMULATION_ONLY_KV,
            "root_base": "base_link is fixed to world as in the official URDF; no extra floating-base joint is added",
            "render_scene_environment_translation_m": [0.0, CANONICAL_SCENE_Y_SHIFT_M, CANONICAL_SCENE_Z_SHIFT_M],
            "render_scene_environment_note": "table and bottle are translated together only for the morphology preview to align the canonical table height with the official X2 fixed-base frame and place the objects beside the robot; this preview translation is not a manipulation pose, and object geometry and mass are unchanged",
        },
        "dimensions": {
            "nbody": model.nbody,
            "nq": model.nq,
            "nv": model.nv,
            "robot_hinge_nq": sum(j["type"] == "revolute" for j in source["joints"].values()),
            "robot_hinge_nv": sum(j["type"] == "revolute" for j in source["joints"].values()),
            "canonical_bottle_free_nq": 7 if not args.no_environment else 0,
            "canonical_bottle_free_nv": 6 if not args.no_environment else 0,
            "nu": model.nu,
            "neq": model.neq,
            "source_links": len(source["links"]),
            "source_joints": len(source["joints"]),
            "source_revolute": sum(j["type"] == "revolute" for j in source["joints"].values()),
            "source_fixed": sum(j["type"] == "fixed" for j in source["joints"].values()),
            "source_mimic": len(source["mimic"]),
            "source_visuals": source_visual_count,
            "compiled_visuals": visual_count,
            "source_collisions": source_collision_count,
            "compiled_collisions": collision_count,
            "compiled_visual_mesh_geoms": mesh_visual_count,
            "compiled_unique_mesh_assets": model.nmesh,
            "robot_mass_kg": body_mass,
        },
        "hands": {
            side: {
                "links": sorted(hand_subtree(source, side)),
                "finger_families": {
                    family: {
                        "source_visual_meshes": sum(
                            item["shape"] == "mesh"
                            for name in hand_subtree(source, side)
                            if name.lower().startswith(f"{side.lower()}_{family}_")
                            for item in source["links"][name]["visuals"]
                        ),
                        "compiled_visual_meshes": sum(
                            item["shape"] == "mesh"
                            for name in hand_subtree(source, side)
                            if name.lower().startswith(f"{side.lower()}_{family}_")
                            for item in compiled_links[name]["visuals"]
                        ),
                    }
                    for family in FINGER_FAMILIES
                },
            }
            for side in HAND_SIDES
        },
        "mimic_equality_data": equality_data,
        "simulation_only_actuated_joints": actuated,
        "validation_errors": errors,
        "source_to_compiled": {
            "links": {
                name: {
                    "source_visuals": len(source["links"][name]["visuals"]),
                    "compiled_visuals": len(compiled_links.get(name, {}).get("visuals", [])),
                    "source_collisions": len(source["links"][name]["collisions"]),
                    "compiled_collisions": len(compiled_links.get(name, {}).get("collisions", [])),
                    "compiled_geometries": compiled_links.get(name, {}),
                }
                for name in source["links"]
            },
            "joints": compiled_joints,
            "mimic": equality_data,
            "inertials": details["inertials"],
        },
        "generated_urdf": str(generated_urdf),
        "generated_urdf_sha256": sha256(generated_urdf),
        "generated_mjcf": str(details["model_xml_path"]),
    }
    inventory_path = args.output / "source_to_compiled_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not errors and not args.no_render:
        inventory["rendering"] = render_images(model, source, details, args.output, not args.no_environment)
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": inventory["status"], "inventory": str(inventory_path), "dimensions": inventory["dimensions"], "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
