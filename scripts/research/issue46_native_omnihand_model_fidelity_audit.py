from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))
sys.path.insert(0, str(REPO_ROOT / "simulation" / "mujoco"))

import issue46_native_omnihand_minimal_grasp as fixture  # noqa: E402


SOURCE_PIN = fixture.SOURCE_PIN
URDF = fixture.URDF
VENDOR_ROOT = fixture.VENDOR_ROOT
EVIDENCE_ROOT = Path(
    os.environ.get(
        "ISSUE46_MODEL_FIDELITY_EVIDENCE_DIR",
        "/tmp/issue46-native-omnihand-model-fidelity",
    )
)
CONTACT_PALM_POSITION = np.array([0.250, 0.159, 0.9175], dtype=float)
VISUAL_CLOSE_POSE = {
    **fixture.CLOSED_POSE,
    "R_ring_pip_joint": 0.90,
    "R_pinky_pip_joint": 0.90,
}
BOTTLE_SURROUND_POSE = {
    **fixture.OPEN_POSE,
    "R_thumb_roll_joint": 0.4232,
    "R_thumb_abad_joint": -0.9307,
    "R_thumb_mcp_joint": 0.43175,
    "R_index_abad_joint": -0.1688,
    "R_index_pip_joint": 0.126,
}
POSITION_TOLERANCE_M = 2e-7
ANGLE_TOLERANCE_RAD = 2e-6
INERTIA_TOLERANCE = 2e-8


def geom_role(contype: int, conaffinity: int) -> str:
    return "visual" if contype == 0 and conaffinity == 0 else "collision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vec(element: ET.Element | None, attribute: str, default: str, size: int) -> list[float]:
    value = default if element is None else element.get(attribute, default)
    result = np.fromstring(value, sep=" ", dtype=float)
    if result.size != size:
        raise ValueError(f"expected {size} values for {attribute}: {value}")
    return result.tolist()


def rpy_quaternion(rpy: list[float]) -> np.ndarray:
    result = np.zeros(4, dtype=float)
    mujoco.mju_euler2Quat(result, np.asarray(rpy, dtype=float), "XYZ")
    return result


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    result = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def quaternion_angle_error(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def geometry_source_record(node: ET.Element, kind: str, link_name: str) -> dict[str, object]:
    origin = node.find("origin")
    geometry = node.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError(f"invalid {kind} geometry for {link_name}")
    shape = geometry[0]
    record: dict[str, object] = {
        "link": link_name,
        "kind": kind,
        "origin_xyz_m": vec(origin, "xyz", "0 0 0", 3),
        "origin_rpy_rad": vec(origin, "rpy", "0 0 0", 3),
        "shape": shape.tag,
        "shape_attributes": dict(shape.attrib),
    }
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise ValueError(f"mesh without filename for {link_name}")
        relative = filename.split("://", 1)[-1]
        if relative.startswith("X2_URDF-v1.4.0/"):
            relative = relative.split("X2_URDF-v1.4.0/", 1)[1]
        mesh_path = (URDF.parent / relative).resolve()
        if not mesh_path.is_file():
            mesh_path = (URDF.parent / Path(relative).name).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"source mesh missing: {filename}")
        record["mesh"] = {
            "source_reference": filename,
            "resolved_path": str(mesh_path),
            "sha256": sha256(mesh_path),
            "scale": vec(shape, "scale", "1 1 1", 3),
        }
    material = node.find("material")
    if material is not None:
        record["material_reference"] = material.get("name")
    return record


def source_inventory(source_root: ET.Element) -> dict[str, object]:
    included_links, _ = fixture.right_hand_subtree(source_root)
    hand_joints = [
        joint
        for joint in source_root.findall("joint")
        if joint.find("child") is not None
        and joint.find("child").get("link") in included_links
    ]
    source_links: list[dict[str, object]] = []
    for link in source_root.findall("link"):
        name = str(link.get("name"))
        if name not in included_links:
            continue
        inertial = link.find("inertial")
        inertial_record: dict[str, object] | None = None
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if mass is None or inertia is None:
                raise ValueError(f"incomplete inertial data for {name}")
            inertial_record = {
                "mass_kg": float(mass.get("value")),
                "origin_xyz_m": vec(origin, "xyz", "0 0 0", 3),
                "origin_rpy_rad": vec(origin, "rpy", "0 0 0", 3),
                "tensor_in_inertial_frame_kg_m2": [
                    float(inertia.get(key, "0"))
                    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                ],
            }
        source_links.append(
            {
                "name": name,
                "inertial": inertial_record,
                "visuals": [geometry_source_record(node, "visual", name) for node in link.findall("visual")],
                "collisions": [geometry_source_record(node, "collision", name) for node in link.findall("collision")],
            }
        )

    source_joints: list[dict[str, object]] = []
    for joint in hand_joints:
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        dynamics = joint.find("dynamics")
        source_joints.append(
            {
                "name": str(joint.get("name")),
                "type": str(joint.get("type")),
                "parent_link": None if parent is None else parent.get("link"),
                "child_link": None if child is None else child.get("link"),
                "origin_xyz_m": vec(origin, "xyz", "0 0 0", 3),
                "origin_rpy_rad": vec(origin, "rpy", "0 0 0", 3),
                "axis_xyz": None if axis is None else vec(axis, "xyz", "1 0 0", 3),
                "limit": None if limit is None else dict(limit.attrib),
                "mimic": None if mimic is None else dict(mimic.attrib),
                "dynamics": None if dynamics is None else dict(dynamics.attrib),
            }
        )
    return {
        "links": source_links,
        "joints": source_joints,
        "counts": {
            "links": len(source_links),
            "joints": len(source_joints),
            "revolute_joints": sum(joint["type"] == "revolute" for joint in source_joints),
            "visual_geometries": sum(len(link["visuals"]) for link in source_links),
            "visual_mesh_references": sum(
                record["shape"] == "mesh" for link in source_links for record in link["visuals"]
            ),
            "collision_geometries": sum(len(link["collisions"]) for link in source_links),
            "transmissions": len(source_root.findall("transmission")),
        },
    }


def source_inertia_matrix(record: dict[str, object]) -> np.ndarray:
    ixx, ixy, ixz, iyy, iyz, izz = record["tensor_in_inertial_frame_kg_m2"]
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
    rotation = quaternion_matrix(rpy_quaternion(record["origin_rpy_rad"]))
    return rotation @ tensor @ rotation.T


def compiled_inventory(
    model: mujoco.MjModel,
    details: dict[str, object],
    source: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    mjcf_root = ET.parse(str(details["model_xml_path"])).getroot()
    mesh_paths = {
        str(asset.get("name")): str(asset.get("file"))
        for asset in mjcf_root.findall("asset/mesh")
    }
    links = {str(link["name"]): link for link in source["links"]}
    hand_names = set(links)
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in hand_names
    }
    missing_bodies = [name for name, body_id in body_ids.items() if body_id < 0]
    if missing_bodies:
        raise AssertionError(f"compiled model is missing source links: {missing_bodies}")

    mjcf_bodies: dict[str, ET.Element] = {}

    def collect_bodies(parent: ET.Element) -> None:
        for body in parent.findall("body"):
            name = body.get("name")
            if name is not None:
                mjcf_bodies[name] = body
            collect_bodies(body)

    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise AssertionError("generated MJCF is missing worldbody")
    collect_bodies(worldbody)

    def local_geom_record(source_geom: dict[str, object], compiled_geom: ET.Element) -> dict[str, object]:
        def compiled_size3() -> np.ndarray:
            raw = np.fromstring(compiled_geom.get("size", "0"), sep=" ", dtype=float)
            if raw.size > 3:
                raise AssertionError("compiled geom has an unexpected size vector")
            return np.pad(raw, (0, 3 - raw.size))

        source_position = np.asarray(source_geom["origin_xyz_m"], dtype=float)
        compiled_position = vec(compiled_geom, "pos", "0 0 0", 3)
        position_error = float(np.linalg.norm(source_position - np.asarray(compiled_position)))
        source_quaternion = rpy_quaternion(source_geom["origin_rpy_rad"])
        compiled_quaternion = np.asarray(vec(compiled_geom, "quat", "1 0 0 0", 4), dtype=float)
        compiled_quaternion_norm = float(np.linalg.norm(compiled_quaternion))
        if compiled_quaternion_norm == 0.0:
            raise AssertionError("generated MJCF contains a zero geom quaternion")
        compiled_quaternion /= compiled_quaternion_norm
        orientation_error = quaternion_angle_error(source_quaternion, compiled_quaternion)
        source_shape = str(source_geom["shape"])
        compiled_shape = str(compiled_geom.get("type", "sphere"))
        shape_aliases = {"box": "box", "cylinder": "cylinder", "sphere": "sphere", "mesh": "mesh"}
        expected_shape = shape_aliases[source_shape]
        if compiled_shape != expected_shape:
            raise AssertionError(
                f"source/compiled geometry type mismatch for {source_geom['link']} {source_geom['kind']}: "
                f"{source_shape} != {compiled_shape}"
            )

        size_error = 0.0
        mesh_sha_match: bool | None = None
        if source_shape == "box":
            full = vec(ET.Element("geometry", source_geom["shape_attributes"]), "size", "", 3)
            expected_size = np.asarray(full, dtype=float) * 0.5
            compiled_size = compiled_size3()
            size_error = float(np.max(np.abs(expected_size - compiled_size)))
        elif source_shape == "cylinder":
            attrs = source_geom["shape_attributes"]
            expected_size = np.array([float(attrs["radius"]), float(attrs["length"]) * 0.5, 0.0])
            compiled_size = compiled_size3()
            size_error = float(np.max(np.abs(expected_size - compiled_size)))
        elif source_shape == "sphere":
            expected_size = np.array([float(source_geom["shape_attributes"]["radius"]), 0.0, 0.0])
            compiled_size = compiled_size3()
            size_error = float(np.max(np.abs(expected_size - compiled_size)))
        else:
            source_mesh = source_geom["mesh"]
            mesh_name = compiled_geom.get("mesh")
            mesh_file = mesh_paths.get(str(mesh_name))
            mesh_sha_match = (
                mesh_file is not None
                and Path(mesh_file).is_file()
                and sha256(Path(mesh_file)) == source_mesh["sha256"]
            )
            if not mesh_sha_match:
                raise AssertionError(
                    f"compiled mesh does not resolve to the pinned source asset for {source_geom['link']}"
                )

        if position_error > POSITION_TOLERANCE_M or orientation_error > ANGLE_TOLERANCE_RAD:
            raise AssertionError(
                f"source/compiled local transform mismatch for {source_geom['link']} {source_geom['kind']}"
            )
        if size_error > POSITION_TOLERANCE_M:
            raise AssertionError(
                f"source/compiled primitive size mismatch for {source_geom['link']} {source_geom['kind']}"
            )
        return {
            "link": source_geom["link"],
            "kind": source_geom["kind"],
            "source_shape": source_shape,
            "compiled_shape": compiled_shape,
            "source_origin_xyz_m": source_position.tolist(),
            "compiled_local_pos_m": compiled_position,
            "local_position_error_m": position_error,
            "source_origin_rpy_rad": source_geom["origin_rpy_rad"],
            "compiled_local_quat_wxyz": compiled_quaternion.tolist(),
            "local_orientation_error_rad": orientation_error,
            "primitive_size_max_abs_error_m": size_error,
            "mesh_source_hash_matches_compiled_asset": mesh_sha_match,
        }

    geometry_parity_records: list[dict[str, object]] = []
    max_geom_position_error = 0.0
    max_geom_orientation_error = 0.0
    max_primitive_size_error = 0.0
    for link_name, source_link in links.items():
        body = mjcf_bodies.get(link_name)
        if body is None:
            raise AssertionError(f"generated MJCF is missing body for source link {link_name}")
        compiled_by_kind = {"visual": [], "collision": []}
        for geom in body.findall("geom"):
            role = geom_role(int(geom.get("contype", "1")), int(geom.get("conaffinity", "1")))
            compiled_by_kind[role].append(geom)
        for role in ("visual", "collision"):
            source_geoms = source_link[f"{role}s"]
            compiled_geoms = compiled_by_kind[role]
            if len(source_geoms) != len(compiled_geoms):
                raise AssertionError(f"generated MJCF geom ordering/count mismatch for {link_name} {role}")
            for source_geom, compiled_geom in zip(source_geoms, compiled_geoms, strict=True):
                record = local_geom_record(source_geom, compiled_geom)
                geometry_parity_records.append(record)
                max_geom_position_error = max(max_geom_position_error, record["local_position_error_m"])
                max_geom_orientation_error = max(max_geom_orientation_error, record["local_orientation_error_rad"])
                max_primitive_size_error = max(max_primitive_size_error, record["primitive_size_max_abs_error_m"])

    hand_geoms: dict[str, dict[str, list[int]]] = {
        name: {"visual": [], "collision": []} for name in hand_names
    }
    all_geom_records: list[dict[str, object]] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
        if body_name not in hand_names:
            continue
        kind = geom_role(int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id]))
        hand_geoms[body_name][kind].append(geom_id)
        geom_type = int(model.geom_type[geom_id])
        data_id = int(model.geom_dataid[geom_id])
        mesh_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, data_id)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
            else None
        )
        all_geom_records.append(
            {
                "compiled_geom_id": geom_id,
                "link": body_name,
                "kind": kind,
                "type": geom_type,
                "type_name": next(
                    (name for name in dir(mujoco.mjtGeom) if name.startswith("mjGEOM_") and int(getattr(mujoco.mjtGeom, name)) == geom_type),
                    "unknown",
                ),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "group": int(model.geom_group[geom_id]),
                "pos_m": model.geom_pos[geom_id].tolist(),
                "quat_wxyz": model.geom_quat[geom_id].tolist(),
                "size_m": model.geom_size[geom_id].tolist(),
                "mesh_name": mesh_name,
                "mesh_file": None if mesh_name is None else mesh_paths.get(mesh_name),
                "mesh_sha256": (
                    None
                    if mesh_name is None
                    else sha256(Path(mesh_paths[mesh_name]))
                ),
            }
        )

    body_records: list[dict[str, object]] = []
    max_body_pos_error = 0.0
    max_body_angle_error = 0.0
    max_mass_error = 0.0
    max_inertial_pos_error = 0.0
    max_inertia_tensor_error = 0.0
    for name, source_link in links.items():
        body_id = body_ids[name]
        source_inertial = source_link["inertial"]
        compiled_tensor_rotation = quaternion_matrix(model.body_iquat[body_id])
        compiled_tensor = (
            compiled_tensor_rotation
            @ np.diag(np.asarray(model.body_inertia[body_id], dtype=float))
            @ compiled_tensor_rotation.T
        )
        if source_inertial is None:
            mass_error = None
            inertial_pos_error = None
            inertia_error = None
            if float(model.body_mass[body_id]) != 0.0 or np.any(model.body_inertia[body_id] != 0.0):
                raise AssertionError(f"compiled link has inertia absent from the pinned URDF: {name}")
        else:
            inertia_error = float(np.max(np.abs(compiled_tensor - source_inertia_matrix(source_inertial))))
            mass_error = abs(float(model.body_mass[body_id]) - float(source_inertial["mass_kg"]))
            inertial_pos_error = float(
                np.linalg.norm(model.body_ipos[body_id] - np.asarray(source_inertial["origin_xyz_m"]))
            )
            max_mass_error = max(max_mass_error, mass_error)
            max_inertial_pos_error = max(max_inertial_pos_error, inertial_pos_error)
            max_inertia_tensor_error = max(max_inertia_tensor_error, inertia_error)

        expected_visual_count = len(source_link["visuals"])
        expected_collision_count = len(source_link["collisions"])
        actual_visual_ids = hand_geoms[name]["visual"]
        actual_collision_ids = hand_geoms[name]["collision"]
        if len(actual_visual_ids) != expected_visual_count:
            raise AssertionError(
                f"visual geom count mismatch for {name}: {expected_visual_count} source, {len(actual_visual_ids)} compiled"
            )
        if len(actual_collision_ids) != expected_collision_count:
            raise AssertionError(
                f"collision geom count mismatch for {name}: {expected_collision_count} source, {len(actual_collision_ids)} compiled"
            )

        expected_parent = next(
            (joint for joint in source["joints"] if joint["child_link"] == name), None
        )
        body_pos_error: float | None = None
        body_angle_error: float | None = None
        parent_name: str | None = None
        if expected_parent is not None and expected_parent["parent_link"] in body_ids:
            parent_name = str(expected_parent["parent_link"])
            if int(model.body_parentid[body_id]) != body_ids[parent_name]:
                raise AssertionError(f"compiled kinematic parent mismatch for {name}")
            body_pos_error = float(
                np.linalg.norm(model.body_pos[body_id] - np.asarray(expected_parent["origin_xyz_m"]))
            )
            body_angle_error = quaternion_angle_error(
                model.body_quat[body_id], rpy_quaternion(expected_parent["origin_rpy_rad"])
            )
            max_body_pos_error = max(max_body_pos_error, body_pos_error)
            max_body_angle_error = max(max_body_angle_error, body_angle_error)

        body_records.append(
            {
                "name": name,
                "compiled_body_id": body_id,
                "compiled_parent": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id]))
                    if body_id > 0
                    else None
                ),
                "source_parent": parent_name,
                "compiled_body_pos_m": model.body_pos[body_id].tolist(),
                "compiled_body_quat_wxyz": model.body_quat[body_id].tolist(),
                "source_joint_origin_position_error_m": body_pos_error,
                "source_joint_origin_angle_error_rad": body_angle_error,
                "source_inertial_specified": source_inertial is not None,
                "source_mass_kg": None if source_inertial is None else source_inertial["mass_kg"],
                "compiled_mass_kg": float(model.body_mass[body_id]),
                "mass_error_kg": mass_error,
                "source_inertial_origin_xyz_m": None if source_inertial is None else source_inertial["origin_xyz_m"],
                "compiled_inertial_origin_xyz_m": model.body_ipos[body_id].tolist(),
                "inertial_origin_error_m": inertial_pos_error,
                "source_inertia_tensor_kg_m2": None if source_inertial is None else source_inertia_matrix(source_inertial).tolist(),
                "compiled_inertia_tensor_kg_m2": compiled_tensor.tolist(),
                "inertia_tensor_max_abs_error": inertia_error,
                "source_inertial_absent_compiled_mass_inertia_zero": (
                    source_inertial is None
                    and float(model.body_mass[body_id]) == 0.0
                    and not np.any(model.body_inertia[body_id] != 0.0)
                ),
                "visual_geom_ids": actual_visual_ids,
                "collision_geom_ids": actual_collision_ids,
            }
        )

    joint_records: list[dict[str, object]] = []
    max_axis_error = 0.0
    max_limit_error = 0.0
    mimic_records: list[dict[str, object]] = []
    for source_joint in source["joints"]:
        joint_name = str(source_joint["name"])
        if source_joint["type"] != "revolute":
            representation = (
                "fixed palm root is anchored by the diagnostic fixture; upstream wrist parent is omitted"
                if source_joint["child_link"] == "R_palm"
                else "fixed body transform"
            )
            joint_records.append({**source_joint, "compiled_joint_id": None, "compiled_representation": representation})
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise AssertionError(f"compiled model is missing revolute joint {joint_name}")
        source_axis = np.asarray(source_joint["axis_xyz"], dtype=float)
        axis_error = float(np.linalg.norm(np.asarray(model.jnt_axis[joint_id]) - source_axis))
        source_range = np.array(
            [float(source_joint["limit"]["lower"]), float(source_joint["limit"]["upper"])],
            dtype=float,
        )
        compiled_range = np.asarray(model.jnt_range[joint_id], dtype=float)
        limit_error = float(np.max(np.abs(compiled_range - source_range)))
        max_axis_error = max(max_axis_error, axis_error)
        max_limit_error = max(max_limit_error, limit_error)
        relation = source_joint["mimic"]
        mimic_compiled: dict[str, object] | None = None
        if relation is not None:
            eq_name = f"urdf_mimic_{joint_name}"
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            if eq_id < 0:
                raise AssertionError(f"source mimic relation is not compiled: {joint_name}")
            mimic_compiled = {
                "equality_id": eq_id,
                "name": eq_name,
                "type": int(model.eq_type[eq_id]),
                "object1_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[eq_id])),
                "object2_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[eq_id])),
                "data": model.eq_data[eq_id].tolist(),
            }
            if mimic_compiled["object1_joint"] != joint_name or mimic_compiled["object2_joint"] != relation["joint"]:
                raise AssertionError(f"compiled mimic mapping mismatch for {joint_name}")
            mimic_records.append({"source_follower": joint_name, "source_relation": relation, "compiled": mimic_compiled})
        joint_records.append(
            {
                **source_joint,
                "compiled_joint_id": joint_id,
                "compiled_axis_xyz": model.jnt_axis[joint_id].tolist(),
                "axis_l2_error": axis_error,
                "compiled_range_rad": compiled_range.tolist(),
                "position_limit_max_abs_error_rad": limit_error,
                "compiled_limited": bool(model.jnt_limited[joint_id]),
                "source_velocity_limit_rad_s": float(source_joint["limit"]["velocity"]),
                "velocity_limit_enforced_as_joint_limit": False,
                "mimic_compiled": mimic_compiled,
            }
        )

    source_visual_count = int(source["counts"]["visual_geometries"])
    source_mesh_count = int(source["counts"]["visual_mesh_references"])
    source_collision_count = int(source["counts"]["collision_geometries"])
    compiled_visual_count = sum(len(groups["visual"]) for groups in hand_geoms.values())
    compiled_collision_count = sum(len(groups["collision"]) for groups in hand_geoms.values())
    compiled_mesh_geom_count = sum(
        int(record["type"]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        for record in all_geom_records
        if record["kind"] == "visual"
    )
    if compiled_visual_count != source_visual_count:
        raise AssertionError("compiled model does not preserve every source visual geometry")
    if compiled_mesh_geom_count != source_mesh_count:
        raise AssertionError("compiled model does not preserve every source visual mesh reference")
    if compiled_collision_count != source_collision_count:
        raise AssertionError("compiled model collision geometry count differs from source")
    if any(record["kind"] == "visual" and (record["contype"] != 0 or record["conaffinity"] != 0) for record in all_geom_records):
        raise AssertionError("a rendering-only official visual geom can collide")
    if any(record["kind"] == "collision" and (record["contype"] == 0 or record["conaffinity"] == 0) for record in all_geom_records):
        raise AssertionError("a source collision geom is disabled")
    if max_body_pos_error > POSITION_TOLERANCE_M or max_body_angle_error > ANGLE_TOLERANCE_RAD:
        raise AssertionError("compiled link frames differ from source joint origins")
    if max_axis_error > ANGLE_TOLERANCE_RAD or max_limit_error > ANGLE_TOLERANCE_RAD:
        raise AssertionError("compiled source joint axes or limits differ")
    if max_mass_error > 1e-9 or max_inertial_pos_error > POSITION_TOLERANCE_M or max_inertia_tensor_error > INERTIA_TOLERANCE:
        raise AssertionError("compiled source inertial parameters differ")

    actuator_records: list[dict[str, object]] = []
    for actuator_id in range(model.nu):
        actuator_records.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator_id, 0])),
                "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
                "forcerange": model.actuator_forcerange[actuator_id].tolist(),
                "gainprm": model.actuator_gainprm[actuator_id].tolist(),
                "biasprm": model.actuator_biasprm[actuator_id].tolist(),
            }
        )

    inventory = {
        "source": {
            "repository": "AgibotTech/agibot_x2_urdf",
            "commit": SOURCE_PIN,
            "urdf_path": str(URDF),
            "urdf_sha256": sha256(URDF),
        },
        "compiled": {
            "mujoco_version": mujoco.__version__,
            "mode": details["mode"],
            "model_xml_path": str(details["model_xml_path"]),
            "model_xml_sha256": details["model_xml_sha256"],
            "dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq", "nbody", "njnt", "ngeom", "nmesh")},
            "visual_preservation": "URDF compiler discardvisual=false; strippath=false",
            "unique_mesh_asset_count": int(model.nmesh),
            "visual_geom_count": compiled_visual_count,
            "visual_mesh_geom_count": compiled_mesh_geom_count,
            "collision_geom_count": compiled_collision_count,
            "geom_type_counts": {
                str(int(geom_type)): int(np.count_nonzero(model.geom_type == geom_type))
                for geom_type in sorted(set(int(value) for value in model.geom_type))
            },
            "visual_geoms_noncolliding": True,
            "collision_geoms_enabled": True,
        },
        "parity": {
            "link_names_and_bodies": len(body_records) == source["counts"]["links"],
            "visual_geom_count": compiled_visual_count == source_visual_count,
            "visual_mesh_reference_count": compiled_mesh_geom_count == source_mesh_count,
            "collision_geom_count": compiled_collision_count == source_collision_count,
            "visuals_noncolliding": True,
            "collision_geoms_enabled": True,
            "max_link_origin_position_error_m": max_body_pos_error,
            "max_link_origin_angle_error_rad": max_body_angle_error,
            "max_joint_axis_error": max_axis_error,
            "max_joint_limit_error_rad": max_limit_error,
            "max_mass_error_kg": max_mass_error,
            "max_inertial_origin_error_m": max_inertial_pos_error,
            "max_inertia_tensor_error_kg_m2": max_inertia_tensor_error,
            "max_geometry_local_position_error_m": max_geom_position_error,
            "max_geometry_local_orientation_error_rad": max_geom_orientation_error,
            "max_primitive_size_error_m": max_primitive_size_error,
            "all_visual_mesh_asset_hashes_match": all(
                record["mesh_source_hash_matches_compiled_asset"] is not False
                for record in geometry_parity_records
            ),
            "geometry_transforms_and_shapes_match": True,
            "mimic_relations_compiled": len(mimic_records) == sum(joint["mimic"] is not None for joint in source["joints"]),
        },
        "source_inventory": source,
        "compiled_bodies": body_records,
        "compiled_joints": joint_records,
        "compiled_hand_geometries": all_geom_records,
        "source_to_compiled_geometry_parity": geometry_parity_records,
        "mimic_relations": mimic_records,
        "actuation_assumption": {
            "source_transmission_count": source["counts"]["transmissions"],
            "compiled_actuator_count": int(model.nu),
            "mode": details["mode"],
            "assumption": "RobotSim diagnostic position servos; not specified by the pinned URDF and not verified hardware semantics",
            "position_kp": fixture.FINGER_KP,
            "position_kv": fixture.FINGER_KV,
            "mimic_implementation": "MuJoCo joint equalities for source-faithful fixture; equality-free independent position actuators only in simulation-only mode",
        },
        "compiled_actuators": actuator_records,
    }
    parity = inventory["parity"]
    checks = {
        "link_frames": max_body_pos_error <= POSITION_TOLERANCE_M and max_body_angle_error <= ANGLE_TOLERANCE_RAD,
        "joint_axes_and_limits": max_axis_error <= ANGLE_TOLERANCE_RAD and max_limit_error <= ANGLE_TOLERANCE_RAD,
        "inertials": max_mass_error <= 1e-9 and max_inertial_pos_error <= POSITION_TOLERANCE_M and max_inertia_tensor_error <= INERTIA_TOLERANCE,
        "geometry_counts_and_masks": compiled_visual_count == source_visual_count and compiled_mesh_geom_count == source_mesh_count and compiled_collision_count == source_collision_count,
        "geometry_transforms_and_shapes": max_geom_position_error <= POSITION_TOLERANCE_M and max_geom_orientation_error <= ANGLE_TOLERANCE_RAD and max_primitive_size_error <= POSITION_TOLERANCE_M,
        "mimic_relations": len(mimic_records) == sum(joint["mimic"] is not None for joint in source["joints"]),
    }
    return inventory, {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "parity": parity}


def compiled_hand_geom_ids(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    visual: list[int] = []
    collision: list[int] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name.startswith("R_"):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            visual.append(geom_id)
        else:
            collision.append(geom_id)
    return visual, collision


def set_visual_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    details: dict[str, object],
    targets: dict[str, float],
) -> None:
    resolved = fixture.derived_targets(targets, details)
    for joint_name in details["source_joints"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = fixture.source_target_to_raw(
            model, details, joint_name, float(resolved[joint_name])
        )
    mujoco.mj_forward(model, data)


def display_groups(model: mujoco.MjModel) -> tuple[mujoco.MjvOption, mujoco.MjvOption, mujoco.MjvOption, mujoco.MjvOption]:
    visual_ids, collision_ids = compiled_hand_geom_ids(model)
    hand_ids = set(visual_ids + collision_ids)
    for geom_id in range(model.ngeom):
        if geom_id in visual_ids:
            model.geom_group[geom_id] = 2
        elif geom_id in collision_ids:
            model.geom_group[geom_id] = 3
        else:
            model.geom_group[geom_id] = 1
    for geom_id in collision_ids:
        model.geom_rgba[geom_id, 3] = 0.30

    options: list[mujoco.MjvOption] = []
    for visible in ((2,), (3,), (2, 3), (1, 2)):
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        option.geomgroup[:] = 0
        for group in visible:
            option.geomgroup[group] = 1
        options.append(option)
    if not hand_ids:
        raise AssertionError("no native hand geoms available for rendering")
    return tuple(options)  # type: ignore[return-value]


def render_state(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    option: mujoco.MjvOption,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render().copy()


def run_visual_audit(out_dir: Path, camera_azimuth_deg: float = 225.0) -> dict[str, object]:
    fixture.verify_vendor_pin()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, details = fixture.build_model(
        "source-faithful",
        out_dir,
        palm_position=CONTACT_PALM_POSITION,
    )
    source_root = ET.parse(URDF).getroot()
    source = source_inventory(source_root)
    inventory, validation = compiled_inventory(model, details, source)
    if validation["status"] != "PASS":
        raise AssertionError(f"model-fidelity parity failed: {validation}")

    data = mujoco.MjData(model)
    targets = {name: float(details["joint_references"][name]) for name in details["source_joints"]}
    open_targets = fixture.derived_targets({**targets, **fixture.OPEN_POSE}, details)
    close_targets = fixture.derived_targets({**targets, **VISUAL_CLOSE_POSE}, details)
    open_pose = {"OPEN": open_targets, "CLOSE": close_targets}
    visual_option, collision_option, combined_option, bottle_option = display_groups(model)

    camera = fixture.camera_for_scene()
    camera.lookat[:] = [0.250, 0.160, 0.940]
    camera.distance = 0.43
    camera.azimuth = camera_azimuth_deg
    camera.elevation = -13.0
    renderer = mujoco.Renderer(model, height=720, width=960)
    screenshots: dict[str, np.ndarray] = {}
    try:
        set_visual_pose(model, data, details, open_pose["OPEN"])
        screenshots["official_visual_only_open.png"] = render_state(renderer, data, camera, visual_option)
        screenshots["collision_only_diagnostic_open.png"] = render_state(renderer, data, camera, collision_option)
        screenshots["combined_visual_collision_open.png"] = render_state(renderer, data, camera, combined_option)
        screenshots["native_hand_open.png"] = render_state(renderer, data, camera, visual_option)

        set_visual_pose(model, data, details, open_pose["CLOSE"])
        screenshots["native_hand_close.png"] = render_state(renderer, data, camera, visual_option)
        set_visual_pose(model, data, details, fixture.derived_targets({**targets, **BOTTLE_SURROUND_POSE}, details))
        screenshots["thumb_opposition_canonical_bottle.png"] = render_state(renderer, data, camera, bottle_option)
        camera.azimuth = (camera_azimuth_deg + 90.0) % 360.0
        screenshots["thumb_opposition_canonical_bottle_opposite_view.png"] = render_state(
            renderer, data, camera, bottle_option
        )
        camera.azimuth = camera_azimuth_deg

        set_visual_pose(model, data, details, open_pose["OPEN"])
        visual_before = np.asarray(data.geom_xpos).copy()
        set_visual_pose(model, data, details, open_pose["CLOSE"])
        visual_after = np.asarray(data.geom_xpos).copy()
        family_motion: dict[str, dict[str, object]] = {}
        for family in fixture.FINGER_FAMILIES:
            ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or "").lower().startswith(f"r_{family}_")
            ]
            displacements = [float(np.linalg.norm(visual_after[i] - visual_before[i])) for i in ids]
            family_motion[family] = {
                "mesh_geom_count": len(ids),
                "max_world_position_change_m": max(displacements, default=0.0),
                "moves_with_joint": bool(displacements) and max(displacements) > 1e-4,
            }
        if not all(item["moves_with_joint"] for item in family_motion.values()):
            raise AssertionError(f"at least one official finger visual did not follow its joint: {family_motion}")

        imageio.imwrite(out_dir / "official_visual_only_open.png", screenshots["official_visual_only_open.png"])
        imageio.imwrite(out_dir / "collision_only_diagnostic_open.png", screenshots["collision_only_diagnostic_open.png"])
        imageio.imwrite(out_dir / "combined_visual_collision_open.png", screenshots["combined_visual_collision_open.png"])
        imageio.imwrite(out_dir / "native_hand_open.png", screenshots["native_hand_open.png"])
        imageio.imwrite(out_dir / "native_hand_close.png", screenshots["native_hand_close.png"])
        imageio.imwrite(out_dir / "thumb_opposition_canonical_bottle.png", screenshots["thumb_opposition_canonical_bottle.png"])
        imageio.imwrite(
            out_dir / "thumb_opposition_canonical_bottle_opposite_view.png",
            screenshots["thumb_opposition_canonical_bottle_opposite_view.png"],
        )
        imageio.imwrite(
            out_dir / "model_representation_side_by_side.png",
            np.concatenate(
                [
                    screenshots["official_visual_only_open.png"],
                    screenshots["collision_only_diagnostic_open.png"],
                    screenshots["combined_visual_collision_open.png"],
                ],
                axis=1,
            ),
        )

        video_path = out_dir / "native_omnihand_open_close_kinematic_render.mp4"
        with imageio.get_writer(video_path, fps=25, codec="libx264", quality=8) as video:
            frames = 50
            for frame_index in range(frames):
                blend = frame_index / (frames - 1)
                smooth = blend * blend * (3.0 - 2.0 * blend)
                interpolated = {
                    name: float(open_pose["OPEN"][name])
                    + smooth * (float(open_pose["CLOSE"][name]) - float(open_pose["OPEN"][name]))
                    for name in details["source_joints"]
                }
                interpolated = fixture.derived_targets(interpolated, details)
                set_visual_pose(model, data, details, interpolated)
                video.append_data(render_state(renderer, data, camera, bottle_option))
    finally:
        renderer.close()

    inventory["visual_joint_motion"] = family_motion
    inventory["visual_evidence"] = {
        "screenshots": sorted(path.name for path in out_dir.glob("*.png")),
        "video": "native_omnihand_open_close_kinematic_render.mp4",
        "video_semantics": "MuJoCo render-only joint-pose interpolation; no mj_step, no grasp/load-transfer/lift claim",
        "bottle_pose": "canonical free bottle from fixture initial state; no post-reset bottle qpos writes",
        "hand_pose": "all five fingers; source-valid joint targets within pinned URDF ranges",
        "bottle_surround_pose_source": "previously recorded geometry-first bilateral target; kinematic render only",
        "camera_azimuth_deg": camera_azimuth_deg,
        "bottle_surround_opposite_view_azimuth_deg": (camera_azimuth_deg + 90.0) % 360.0,
    }
    inventory["classification"] = {
        "visual_morphology": "PASS" if all(item["moves_with_joint"] for item in family_motion.values()) else "FAIL",
        "collision_geometry_separation": "PASS",
        "mechanical_fidelity": "PASS" if validation["status"] == "PASS" else "FAIL",
        "physical_grasp_or_pickup": "NOT TESTED",
    }
    inventory_path = out_dir / "source_to_compiled_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "status": validation["status"],
        "validation": validation,
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "evidence_directory": str(out_dir),
        "compiled_dimensions": inventory["compiled"]["dimensions"],
        "source_counts": source["counts"],
        "visual_joint_motion": family_motion,
        "classification": inventory["classification"],
    }
    result_path = out_dir / "model_fidelity_audit_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--camera-azimuth", type=float, default=225.0)
    args = parser.parse_args()
    result = run_visual_audit(args.evidence_dir, args.camera_azimuth)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
