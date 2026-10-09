from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "mujoco"))
from canonical_manipulation_assets import (  # noqa: E402
    CANONICAL_X2_BOTTLE_DIAMETER_M,
    CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M,
    CANONICAL_X2_BOTTLE_GEOMS,
    CANONICAL_X2_BOTTLE_HEIGHT_M,
    CANONICAL_X2_BOTTLE_MASS_KG,
    CANONICAL_X2_BOTTLE_START_BODY_POS,
    G1_CANONICAL_TABLE_TOP_Z,
    add_g1_canonical_table,
)


SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
VENDOR_ROOT = Path(
    os.environ.get(
        "AGIBOT_X2_VENDOR_ROOT",
        f"/tmp/robotsim-issue46-native-omnihand-{SOURCE_PIN}",
    )
)
URDF = VENDOR_ROOT / "X2_URDF-v1.4.0" / "X2-Ultra_omnihand.urdf"
OUT = Path(
    os.environ.get(
        "ISSUE46_NATIVE_EVIDENCE_DIR",
        "/tmp/issue46-native-omnihand-evidence",
    )
)
DT = 0.002
GRAVITY_M_S2 = 9.81
PALM_POSITION = np.array([0.300, 0.224, 0.9175], dtype=float)
PALM_QUATERNION_WXYZ = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
PENETRATION_ABORT_M = 0.005
JOINT_LIMIT_TOLERANCE_RAD = 0.001
MIMIC_SOLREF_DIRECT = [-10000.0, -200.0]
FINGER_KP = 18.0
FINGER_KV = 2.4
TABLE_GEOM = "m0_table_top"
BOTTLE_BODY = "bottle"
FINGER_FAMILIES = ("thumb", "index", "middle", "ring", "pinky")

OPEN_POSE = {
    "R_thumb_roll_joint": 0.80,
    "R_thumb_abad_joint": -1.70,
    "R_thumb_mcp_joint": 0.0,
    "R_index_abad_joint": -0.18,
    "R_index_pip_joint": 0.0,
    "R_middle_pip_joint": 0.0,
    "R_ring_abad_joint": 0.10,
    "R_ring_pip_joint": 0.0,
    "R_pinky_abad_joint": 0.15,
    "R_pinky_pip_joint": 0.0,
}
THUMB_POSE = {
    **OPEN_POSE,
    "R_thumb_roll_joint": 0.18,
    "R_thumb_abad_joint": -0.42,
    "R_thumb_mcp_joint": 0.38,
}
TWO_DIGIT_POSE = {
    **THUMB_POSE,
    "R_index_abad_joint": -0.09,
    "R_index_pip_joint": 0.55,
}
THREE_DIGIT_POSE = {**TWO_DIGIT_POSE, "R_middle_pip_joint": 0.55}
CLOSED_POSE = {
    **THREE_DIGIT_POSE,
    "R_thumb_roll_joint": 0.32,
    "R_thumb_abad_joint": -0.72,
    "R_thumb_mcp_joint": 0.55,
    "R_index_abad_joint": -0.10,
    "R_index_pip_joint": 0.90,
    "R_middle_pip_joint": 0.90,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def verify_vendor_pin() -> None:
    actual = git_output(VENDOR_ROOT, "rev-parse", "HEAD")
    dirty = git_output(VENDOR_ROOT, "status", "--porcelain")
    if actual != SOURCE_PIN:
        raise RuntimeError(f"vendor pin mismatch: expected {SOURCE_PIN}, got {actual}")
    if dirty:
        raise RuntimeError("pinned vendor checkout is dirty; refusing to use it")
    if not URDF.is_file():
        raise FileNotFoundError(URDF)


def right_hand_subtree(source_root: ET.Element) -> tuple[set[str], list[ET.Element]]:
    joints = source_root.findall("joint")
    included_links = {"R_palm"}
    selected_joints: list[ET.Element] = []
    changed = True
    while changed:
        changed = False
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if parent.get("link") in included_links and child.get("link") not in included_links:
                included_links.add(str(child.get("link")))
                selected_joints.append(joint)
                changed = True
    if not selected_joints or not any(j.get("name") == "R_thumb_mcp_joint" for j in selected_joints):
        raise RuntimeError("failed to extract native right-hand subtree from pinned URDF")
    return included_links, selected_joints


def write_hand_only_urdf(out_dir: Path) -> tuple[Path, dict[str, object]]:
    source_root = ET.parse(URDF).getroot()
    included_links, selected_joints = right_hand_subtree(source_root)
    links = [link for link in source_root.findall("link") if link.get("name") in included_links]
    if len(links) != len(included_links):
        raise RuntimeError("right-hand subtree references a link missing from the pinned URDF")

    minimal_root = ET.Element("robot", {"name": "x2_native_omnihand_minimal_fixture"})
    for material in source_root.findall("material"):
        minimal_root.append(material)
    for link in links:
        minimal_root.append(link)
    for joint in selected_joints:
        minimal_root.append(joint)

    mesh_paths: list[str] = []
    for mesh in minimal_root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise RuntimeError("mesh without filename in pinned right-hand subtree")
        tail = filename.split("://", 1)[-1]
        marker = "X2_URDF-v1.4.0/"
        if marker in tail:
            tail = tail.split(marker, 1)[1]
        elif tail.startswith("package://"):
            tail = tail[len("package://") :]
        candidate = (URDF.parent / tail).resolve()
        if not candidate.is_file():
            candidate = (URDF.parent / Path(tail).name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"right-hand mesh not found: {filename} -> {candidate}")
        mesh.set("filename", str(candidate))
        mesh_paths.append(str(candidate))

    output = out_dir / "native_right_omnihand_fixture.urdf"
    ET.ElementTree(minimal_root).write(output, encoding="utf-8", xml_declaration=True)
    return output, {
        "included_links": sorted(included_links),
        "included_joint_names": [str(j.get("name")) for j in selected_joints],
        "mesh_paths": sorted(set(mesh_paths)),
        "urdf_sha256": sha256(output),
    }


def source_joint_maps(source_root: ET.Element) -> tuple[dict[str, ET.Element], dict[str, dict[str, object]]]:
    joints = {
        str(joint.get("name")): joint
        for joint in source_root.findall("joint")
        if joint.get("type") == "revolute"
        and str(joint.get("name", "")).startswith("R_")
    }
    relations: dict[str, dict[str, object]] = {}
    for follower, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        driver = str(mimic.get("joint"))
        if driver not in joints:
            raise RuntimeError(f"mimic driver is not in the right hand: {driver} -> {follower}")
        relations[follower] = {
            "follower": follower,
            "driver": driver,
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }
    return joints, relations


def add_bottle(
    spec: mujoco.MjSpec,
    body_position: tuple[float, float, float] = CANONICAL_X2_BOTTLE_START_BODY_POS,
) -> None:
    if not math.isclose(
        CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M[1]
        - CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M[0],
        CANONICAL_X2_BOTTLE_HEIGHT_M,
        abs_tol=1e-12,
    ) or not math.isclose(
        sum(float(g["mass"]) for g in CANONICAL_X2_BOTTLE_GEOMS),
        CANONICAL_X2_BOTTLE_MASS_KG,
        abs_tol=1e-12,
    ):
        raise RuntimeError("canonical X2 bottle geometry, height, and mass definitions disagree")
    bottle = spec.worldbody.add_body(name=BOTTLE_BODY, pos=list(body_position))
    bottle.add_freejoint(name="bottle_free")
    geom_types = {
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }
    for geom in CANONICAL_X2_BOTTLE_GEOMS:
        bottle.add_geom(
            name=str(geom["name"]),
            type=geom_types[str(geom["type"])],
            pos=list(geom["pos"]),
            size=list(geom["size"]),
            mass=float(geom["mass"]),
            rgba=list(geom["rgba"]),
            friction=[1.4, 0.02, 0.001],
            condim=4,
            group=1,
        )


def build_model(
    mode: str,
    out_dir: Path,
    palm_position: tuple[float, float, float] | np.ndarray = PALM_POSITION,
    palm_quaternion_wxyz: tuple[float, float, float, float] | np.ndarray = PALM_QUATERNION_WXYZ,
    bottle_body_position: tuple[float, float, float] = CANONICAL_X2_BOTTLE_START_BODY_POS,
    palm_slide_z: bool = False,
) -> tuple[mujoco.MjModel, dict[str, object]]:
    generated_urdf, hand_info = write_hand_only_urdf(out_dir)
    source_root = ET.parse(URDF).getroot()
    source_joints, relations = source_joint_maps(source_root)
    spec = mujoco.MjSpec.from_file(str(generated_urdf))
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.fusestatic = False
    hand_links, hand_joints = right_hand_subtree(source_root)
    parent_by_child = {
        str(joint.find("child").get("link")): str(joint.find("parent").get("link"))
        for joint in hand_joints
    }
    exclusion_count = 0
    for descendant, parent in parent_by_child.items():
        ancestor = parent
        while ancestor in hand_links:
            spec.add_exclude(
                name=f"hand_chain_exclusion_{exclusion_count}",
                bodyname1=ancestor,
                bodyname2=descendant,
            )
            exclusion_count += 1
            ancestor = parent_by_child.get(ancestor, "")

    if len(spec.worldbody.bodies) != 1 or spec.worldbody.bodies[0].name != "R_palm":
        raise RuntimeError("filtered URDF did not compile to R_palm as its sole root body")
    palm = spec.worldbody.bodies[0]
    palm_position = np.asarray(palm_position, dtype=float)
    palm_quaternion_wxyz = np.asarray(palm_quaternion_wxyz, dtype=float)
    if palm_position.shape != (3,) or not np.isfinite(palm_position).all():
        raise ValueError("palm_position must be a finite three-vector")
    if (
        palm_quaternion_wxyz.shape != (4,)
        or not np.isfinite(palm_quaternion_wxyz).all()
        or not math.isclose(float(np.linalg.norm(palm_quaternion_wxyz)), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("palm_quaternion_wxyz must be a finite unit WXYZ quaternion")
    palm.pos = palm_position.tolist()
    palm.quat = palm_quaternion_wxyz.tolist()
    if palm_slide_z:
        palm.add_joint(
            name="fixture_palm_z",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            limited=True,
            range=[0.0, 0.05],
        )

    spec.worldbody.add_light(
        name="fixture_key_light",
        pos=[0.1, 0.0, 1.7],
        dir=[-0.1, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        rgba=[0.26, 0.29, 0.32, 1.0],
        friction=[1.0, 0.01, 0.001],
        condim=4,
        group=1,
    )
    add_g1_canonical_table(spec)
    add_bottle(spec, bottle_body_position)

    if palm_slide_z:
        palm_actuator = spec.add_actuator(
            name="fixture_palm_z_position",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target="fixture_palm_z",
            ctrllimited=True,
            ctrlrange=[0.0, 0.05],
            forcelimited=True,
            forcerange=[-300.0, 300.0],
        )
        palm_actuator.set_to_position(kp=400.0, kv=40.0)

    for follower, relation in relations.items():
        if mode == "simulation-only":
            continue
        follower_joint = next(j for j in spec.joints if j.name == follower)
        driver_joint = next(j for j in spec.joints if j.name == relation["driver"])
        multiplier = float(relation["multiplier"])
        offset = float(relation["offset"])
        polynomial = [
            offset + multiplier * float(driver_joint.ref) - float(follower_joint.ref),
            multiplier,
            0.0,
            0.0,
            0.0,
        ]
        equality = spec.add_equality(
            name=f"urdf_mimic_{follower}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=follower,
            name2=str(relation["driver"]),
            data=polynomial + [0.0] * 6,
        )
        equality.solref = list(MIMIC_SOLREF_DIRECT)

    follower_names = set(relations)
    joint_nodes = {str(j.name): j for j in spec.joints if j.name}
    for joint_name, source_joint in source_joints.items():
        joint = joint_nodes.get(joint_name)
        if joint is None:
            raise RuntimeError(f"source revolute hand joint missing in compiled fixture spec: {joint_name}")
        if mode == "source-faithful" and joint_name in follower_names:
            continue
        limit = source_joint.find("limit")
        if limit is None:
            raise RuntimeError(f"source hand joint has no hard limits: {joint_name}")
        effort = float(limit.get("effort"))
        actuator = spec.add_actuator(
            name=f"position_{joint_name}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            ctrllimited=True,
            ctrlrange=list(joint.range),
            forcelimited=True,
            forcerange=[-effort, effort],
        )
        actuator.set_to_position(kp=FINGER_KP, kv=FINGER_KV)

    model = spec.compile()
    model.opt.timestep = DT
    model.opt.gravity[:] = [0.0, 0.0, -GRAVITY_M_S2]
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = [0.38, 0.38, 0.38]
    model.vis.headlight.diffuse[:] = [0.68, 0.68, 0.68]
    model.vis.headlight.specular[:] = [0.08, 0.08, 0.08]
    for geom_name in (TABLE_GEOM, "bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_solref[geom_id] = [0.008, 1.0]

    axes: dict[str, int] = {}
    references: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for joint_name, source_joint in source_joints.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        source_axis = np.fromstring(source_joint.find("axis").get("xyz", "1 0 0"), sep=" ")
        model_axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
        cosine = float(np.dot(source_axis, model_axis) / (np.linalg.norm(source_axis) * np.linalg.norm(model_axis)))
        if abs(abs(cosine) - 1.0) > 1e-6:
            raise RuntimeError(f"source/compiled axis mismatch for {joint_name}: dot={cosine}")
        axes[joint_name] = 1 if cosine >= 0 else -1
        references[joint_name] = float(joint_nodes[joint_name].ref)
        limit = source_joint.find("limit")
        ranges[joint_name] = (float(limit.get("lower")), float(limit.get("upper")))

    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOTTLE_BODY)
    bottle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    if bottle_body_id < 0 or bottle_joint_id < 0:
        raise RuntimeError("canonical bottle did not compile as an independent free body")
    if int(model.jnt_type[bottle_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("canonical bottle joint is not free")
    if model.neq != (len(relations) if mode == "source-faithful" else 0):
        raise RuntimeError(f"unexpected equality count for {mode}: {model.neq}")
    if any(int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_JOINT) for i in range(model.neq)):
        raise RuntimeError("fixture contains a non-joint equality, including a forbidden object weld")
    if int(model.body_parentid[bottle_body_id]) != 0:
        raise RuntimeError("bottle body is not a world child")
    if not math.isclose(float(model.body_mass[bottle_body_id]), CANONICAL_X2_BOTTLE_MASS_KG, abs_tol=1e-12):
        raise RuntimeError("compiled bottle mass differs from the canonical mass")

    actuator_by_joint: dict[str, int] = {}
    for joint_name in source_joints:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"position_{joint_name}")
        if actuator_id >= 0:
            actuator_by_joint[joint_name] = actuator_id
    details: dict[str, object] = {
        "mode": mode,
        "palm_position": palm_position.tolist(),
        "palm_quaternion_wxyz": palm_quaternion_wxyz.tolist(),
        "bottle_body_position": list(bottle_body_position),
        "palm_slide_z_enabled": palm_slide_z,
        "palm_slide_z_qpos_address": (
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fixture_palm_z")])
            if palm_slide_z
            else None
        ),
        "palm_slide_z_actuator_id": (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fixture_palm_z_position")
            if palm_slide_z
            else None
        ),
        "hand_info": hand_info,
        "self_chain_exclusion_count": exclusion_count,
        "source_joints": source_joints,
        "mimic_relations": relations,
        "mimic_followers": follower_names,
        "joint_axes": axes,
        "joint_references": references,
        "joint_ranges": ranges,
        "actuator_by_joint": actuator_by_joint,
        "bottle_body_id": bottle_body_id,
        "bottle_qpos_address": int(model.jnt_qposadr[bottle_joint_id]),
        "bottle_qvel_address": int(model.jnt_dofadr[bottle_joint_id]),
        "table_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM),
    }
    model_file = out_dir / f"native_omnihand_{mode}.xml"
    model_file.write_text(spec.to_xml(), encoding="utf-8")
    details["model_xml_path"] = str(model_file)
    details["model_xml_sha256"] = sha256(model_file)
    return model, details


def source_position(model: mujoco.MjModel, data: mujoco.MjData, details: dict[str, object], joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos = int(model.jnt_qposadr[joint_id])
    sign = int(details["joint_axes"][joint_name])
    reference = float(details["joint_references"][joint_name])
    return reference + sign * (float(data.qpos[qpos]) - float(model.qpos0[qpos]))


def source_target_to_raw(model: mujoco.MjModel, details: dict[str, object], joint_name: str, value: float) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos = int(model.jnt_qposadr[joint_id])
    sign = int(details["joint_axes"][joint_name])
    reference = float(details["joint_references"][joint_name])
    low, high = details["joint_ranges"][joint_name]
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"target violates original URDF limits for {joint_name}: {value} not in [{low}, {high}]")
    return float(model.qpos0[qpos]) + sign * (value - reference)


def derived_targets(targets: dict[str, float], details: dict[str, object]) -> dict[str, float]:
    result = dict(targets)
    relations = details["mimic_relations"]
    for _ in range(len(relations) + 1):
        changed = False
        for follower, relation in relations.items():
            value = float(relation["multiplier"]) * result[str(relation["driver"])] + float(relation["offset"])
            if follower not in result or result[follower] != value:
                result[follower] = value
                changed = True
        if not changed:
            break
    return result


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def bottle_contact_report(model: mujoco.MjModel, data: mujoco.MjData, details: dict[str, object]) -> dict[str, object]:
    bottle_id = int(details["bottle_body_id"])
    table_id = int(details["table_geom_id"])
    families: set[str] = set()
    table_force_z = 0.0
    hand_force_z = 0.0
    contacts: list[dict[str, object]] = []
    max_penetration = 0.0
    table_contact_present = False
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        body1_name = body_name(model, body1)
        body2_name = body_name(model, body2)
        is_bottle_contact = body1 == bottle_id or body2 == bottle_id
        local_force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, local_force)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_on_geom2 = frame.T @ local_force[:3]
        fz_on_bottle = 0.0
        if is_bottle_contact:
            fz_on_bottle = float(force_on_geom2[2] if body2 == bottle_id else -force_on_geom2[2])
            if geom1 == table_id or geom2 == table_id:
                table_force_z += fz_on_bottle
            if body1_name == "m0_table" or body2_name == "m0_table":
                table_contact_present = True
            else:
                hand_force_z += fz_on_bottle
                other_body = body1_name if body2 == bottle_id else body2_name
                for family in FINGER_FAMILIES:
                    if family in other_body.lower():
                        families.add(family)
        penetration = max(0.0, -float(contact.dist))
        max_penetration = max(max_penetration, penetration)
        if is_bottle_contact:
            contacts.append(
                {
                    "geom1": geom_name(model, geom1),
                    "geom2": geom_name(model, geom2),
                    "body1": body1_name,
                    "body2": body2_name,
                    "distance_m": float(contact.dist),
                    "force_local_n_xyz": local_force[:3].tolist(),
                    "force_z_on_bottle_n": fz_on_bottle,
                }
            )
    return {
        "contacts": contacts,
        "finger_families": sorted(families),
        "table_support_force_z_n": table_force_z,
        "hand_force_z_on_bottle_n": hand_force_z,
        "table_contact_present": table_contact_present,
        "max_penetration_m": max_penetration,
    }


def camera_for_scene() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.300, 0.055, 0.925]
    camera.distance = 0.56
    camera.azimuth = 180.0
    camera.elevation = -17.0
    return camera


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def run_experiment(mode: str, out_dir: Path) -> dict[str, object]:
    verify_vendor_pin()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, details = build_model(mode, out_dir)
    data = mujoco.MjData(model)
    all_targets = {name: float(details["joint_references"][name]) for name in details["source_joints"]}
    all_targets.update(OPEN_POSE)
    all_targets = derived_targets(all_targets, details)
    initial_hand_qpos_assignments = 0
    for joint_name in details["source_joints"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = source_target_to_raw(
            model, details, joint_name, float(all_targets[joint_name])
        )
        initial_hand_qpos_assignments += 1
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = camera_for_scene()
    video_path = out_dir / f"native_omnihand_{mode}.mp4"
    trace_path = out_dir / f"contact_joint_trace_{mode}.jsonl"
    video = imageio.get_writer(video_path, fps=25, codec="libx264", quality=8)
    trace = trace_path.open("w", encoding="utf-8")
    frames_per_step = max(1, round(1.0 / (25.0 * DT)))
    frame_count = 0
    bottle_qpos_write_count = 0
    initial_bottle_qpos = data.qpos[
        int(details["bottle_qpos_address"]) : int(details["bottle_qpos_address"]) + 7
    ].copy()
    initial_bottle_position = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float).copy()
    bottle_mass = float(model.body_mass[int(details["bottle_body_id"])])
    robot_weight_n = bottle_mass * GRAVITY_M_S2
    phase_results: list[dict[str, object]] = []
    gates: dict[str, dict[str, object]] = {}
    stop_reason: str | None = None
    wall_start = time.monotonic()

    def render_frame(force: bool = False) -> None:
        nonlocal frame_count
        if force or int(round(data.time / DT)) % frames_per_step == 0:
            renderer.update_scene(data, camera=camera)
            video.append_data(renderer.render())
            frame_count += 1

    def step_phase(
        label: str,
        duration_s: float,
        start_targets: dict[str, float],
        end_targets: dict[str, float],
    ) -> dict[str, object]:
        nonlocal stop_reason
        steps = max(1, int(round(duration_s / DT)))
        bottle_start = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float).copy()
        max_penetration = 0.0
        max_bottle_step = 0.0
        min_table_support = math.inf
        max_hand_support = -math.inf
        max_hand_support_ratio = -math.inf
        table_support_samples: list[float] = []
        hand_support_samples: list[float] = []
        final_contacts: dict[str, object] = {}
        contact_frames = 0
        seen_families: set[str] = set()
        finger_family_samples: list[set[str]] = []
        max_joint_error: dict[str, float] = {name: 0.0 for name in details["source_joints"]}
        max_joint_speed: dict[str, float] = {name: 0.0 for name in details["source_joints"]}
        joint_error_samples: dict[str, list[float]] = {name: [] for name in details["source_joints"]}
        joint_speed_samples: dict[str, list[float]] = {name: [] for name in details["source_joints"]}
        phase_start_time = float(data.time)
        previous_bottle = bottle_start.copy()
        for step_index in range(steps):
            alpha = smoothstep((step_index + 1) / steps)
            desired = {
                joint_name: float(start_targets.get(joint_name, details["joint_references"][joint_name]))
                + alpha
                * (
                    float(end_targets.get(joint_name, details["joint_references"][joint_name]))
                    - float(start_targets.get(joint_name, details["joint_references"][joint_name]))
                )
                for joint_name in details["source_joints"]
            }
            control_targets = derived_targets(desired, details)
            for joint_name, actuator_id in details["actuator_by_joint"].items():
                data.ctrl[actuator_id] = source_target_to_raw(model, details, joint_name, control_targets[joint_name])
            mujoco.mj_step(model, data)
            finite = bool(
                np.isfinite(data.qpos).all()
                and np.isfinite(data.qvel).all()
                and np.isfinite(data.qacc).all()
                and np.isfinite(data.ctrl).all()
            )
            if not finite:
                stop_reason = f"non-finite MuJoCo state at {label} step {step_index + 1}"
            contacts = bottle_contact_report(model, data, details)
            final_contacts = contacts
            max_penetration = max(max_penetration, float(contacts["max_penetration_m"]))
            frame_families = set(contacts["finger_families"])
            finger_family_samples.append(frame_families)
            if frame_families:
                contact_frames += 1
                seen_families.update(frame_families)
            min_table_support = min(min_table_support, float(contacts["table_support_force_z_n"]))
            max_hand_support = max(max_hand_support, float(contacts["hand_force_z_on_bottle_n"]))
            table_support_samples.append(float(contacts["table_support_force_z_n"]))
            hand_support_samples.append(float(contacts["hand_force_z_on_bottle_n"]))
            max_hand_support_ratio = max(
                max_hand_support_ratio,
                float(contacts["hand_force_z_on_bottle_n"]) / robot_weight_n,
            )
            bottle_position = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float).copy()
            max_bottle_step = max(max_bottle_step, float(np.linalg.norm(bottle_position - previous_bottle)))
            previous_bottle = bottle_position
            joint_states: dict[str, dict[str, float]] = {}
            for joint_name in details["source_joints"]:
                measured = source_position(model, data, details, joint_name)
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                velocity = int(details["joint_axes"][joint_name]) * float(
                    data.qvel[int(model.jnt_dofadr[joint_id])]
                )
                target = float(control_targets[joint_name])
                error = measured - target
                max_joint_error[joint_name] = max(max_joint_error[joint_name], abs(error))
                max_joint_speed[joint_name] = max(max_joint_speed[joint_name], abs(velocity))
                joint_error_samples[joint_name].append(abs(error))
                joint_speed_samples[joint_name].append(abs(velocity))
                low, high = details["joint_ranges"][joint_name]
                if measured < low - JOINT_LIMIT_TOLERANCE_RAD or measured > high + JOINT_LIMIT_TOLERANCE_RAD:
                    stop_reason = f"source hard joint limit exceeded at {label}: {joint_name}={measured:.9f}"
                joint_states[joint_name] = {
                    "target_source_rad": target,
                    "position_source_rad": measured,
                    "velocity_source_rad_s": velocity,
                    "tracking_error_rad": error,
                }
            if max_penetration > PENETRATION_ABORT_M:
                stop_reason = (
                    f"penetration abort at {label}: {max_penetration:.6f} m > "
                    f"{PENETRATION_ABORT_M:.6f} m"
                )
            bottle_qpos = data.qpos[
                int(details["bottle_qpos_address"]) : int(details["bottle_qpos_address"]) + 7
            ]
            if not np.isfinite(bottle_qpos).all():
                stop_reason = f"non-finite bottle free-joint state at {label} step {step_index + 1}"
            trace_record = {
                "time_s": float(data.time),
                "phase": label,
                "mode": mode,
                "mujoco_ncon": int(data.ncon),
                "contacts": contacts["contacts"],
                "finger_families": contacts["finger_families"],
                "table_support_force_z_n": contacts["table_support_force_z_n"],
                "hand_force_z_on_bottle_n": contacts["hand_force_z_on_bottle_n"],
                "bottle_position_world_m": bottle_position.tolist(),
                "bottle_quaternion_wxyz": data.xquat[int(details["bottle_body_id"])].tolist(),
                "bottle_linear_velocity_world_m_s": data.cvel[int(details["bottle_body_id"])][3:].tolist(),
                "bottle_angular_velocity_world_rad_s": data.cvel[int(details["bottle_body_id"])][:3].tolist(),
                "joints": joint_states,
                "bottle_qpos_write_count": bottle_qpos_write_count,
            }
            trace.write(json.dumps(trace_record, separators=(",", ":")) + "\n")
            if stop_reason:
                break
            render_frame()
        elapsed = float(data.time) - phase_start_time
        bottle_end = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float).copy()
        final_window_steps = min(50, len(table_support_samples))
        final_window = slice(-final_window_steps, None)
        final_window_max_joint_speed = {
            name: max(values[final_window], default=math.inf)
            for name, values in joint_speed_samples.items()
        }
        final_window_max_joint_error = {
            name: max(values[final_window], default=math.inf)
            for name, values in joint_error_samples.items()
        }
        final_window_family_samples = finger_family_samples[-final_window_steps:]
        final_window_family_fractions = {
            family: sum(family in sample for sample in final_window_family_samples)
            / max(1, final_window_steps)
            for family in FINGER_FAMILIES
        }
        final_window_families = sorted(
            set().union(*final_window_family_samples) if final_window_family_samples else set()
        )
        return {
            "phase": label,
            "requested_duration_s": duration_s,
            "actual_duration_s": elapsed,
            "bottle_start_position_world_m": bottle_start.tolist(),
            "bottle_end_position_world_m": bottle_end.tolist(),
            "bottle_vertical_delta_m": float(bottle_end[2] - bottle_start[2]),
            "max_bottle_per_step_translation_m": max_bottle_step,
            "max_penetration_m": max_penetration,
            "finger_contact_families": sorted(seen_families),
            "finger_contact_fraction": contact_frames / max(1, int(round(elapsed / DT))),
            "final_100ms_finger_contact_families": final_window_families,
            "final_100ms_finger_contact_fraction_by_family": final_window_family_fractions,
            "minimum_table_support_force_z_n": min_table_support if math.isfinite(min_table_support) else None,
            "maximum_hand_force_z_on_bottle_n": max_hand_support if math.isfinite(max_hand_support) else None,
            "maximum_hand_support_fraction_of_bottle_weight": (
                max_hand_support_ratio if math.isfinite(max_hand_support_ratio) else None
            ),
            "mean_final_100ms_table_support_force_z_n": float(np.mean(table_support_samples[-50:])) if table_support_samples else None,
            "mean_final_100ms_hand_force_z_on_bottle_n": float(np.mean(hand_support_samples[-50:])) if hand_support_samples else None,
            "mean_final_100ms_hand_support_fraction_of_bottle_weight": (
                float(np.mean(hand_support_samples[-50:])) / robot_weight_n if hand_support_samples else None
            ),
            "final_table_contact": bool(final_contacts.get("table_contact_present", False)),
            "final_finger_contact_families": list(final_contacts.get("finger_families", [])),
            "maximum_joint_tracking_error_rad": max_joint_error,
            "maximum_joint_speed_rad_s": max_joint_speed,
            "final_100ms_max_joint_tracking_error_rad": final_window_max_joint_error,
            "final_100ms_max_joint_speed_rad_s": final_window_max_joint_speed,
            "stop_reason": stop_reason,
        }

    def save_frame(filename: str) -> None:
        renderer.update_scene(data, camera=camera)
        imageio.imwrite(out_dir / filename, renderer.render())

    def set_gate(name: str, passed: bool, evidence: dict[str, object]) -> bool:
        gates[name] = {"status": "PASS" if passed else "FAIL", **evidence}
        return passed

    try:
        render_frame(force=True)
        save_frame(f"initial_open_{mode}.png")
        open_phase = step_phase("stable_open_hold", 0.8, all_targets, all_targets)
        phase_results.append(open_phase)
        open_stable = (
            max(open_phase["final_100ms_max_joint_speed_rad_s"].values()) < 0.01
            and max(open_phase["final_100ms_max_joint_tracking_error_rad"].values()) < 0.01
            and abs(open_phase["bottle_vertical_delta_m"]) < 0.001
            and not open_phase["final_finger_contact_families"]
            and open_phase["max_penetration_m"] <= PENETRATION_ABORT_M
        )
        if not set_gate("stable_open_hold", open_stable, open_phase):
            stop_reason = stop_reason or "open hold did not settle within the declared gate"

        if not stop_reason:
            thumb_phase = step_phase("controlled_thumb_motion", 0.65, all_targets, THUMB_POSE)
            phase_results.append(thumb_phase)
            thumb_motion = abs(source_position(model, data, details, "R_thumb_mcp_joint") - OPEN_POSE["R_thumb_mcp_joint"])
            thumb_ok = thumb_motion >= 0.10 and thumb_phase["max_penetration_m"] <= PENETRATION_ABORT_M
            if set_gate("controlled_thumb_motion", thumb_ok, {**thumb_phase, "thumb_mcp_motion_rad": thumb_motion}):
                all_targets.update(THUMB_POSE)
            else:
                stop_reason = stop_reason or "thumb did not show bounded controlled motion"

        if not stop_reason:
            index_start = dict(all_targets)
            index_phase = step_phase("controlled_opposing_finger_motion", 0.65, index_start, TWO_DIGIT_POSE)
            phase_results.append(index_phase)
            index_motion = abs(source_position(model, data, details, "R_index_pip_joint") - OPEN_POSE["R_index_pip_joint"])
            index_ok = index_motion >= 0.10 and index_phase["max_penetration_m"] <= PENETRATION_ABORT_M
            if set_gate("controlled_opposing_finger_motion", index_ok, {**index_phase, "index_pip_motion_rad": index_motion}):
                all_targets.update(TWO_DIGIT_POSE)
            else:
                stop_reason = stop_reason or "opposing finger did not show bounded controlled motion"

        if not stop_reason:
            two_hold = step_phase("two_digit_stable_hold", 1.2, all_targets, all_targets)
            phase_results.append(two_hold)
            two_digit_joints = (
                "R_thumb_roll_joint",
                "R_thumb_abad_joint",
                "R_thumb_mcp_joint",
                "R_index_abad_joint",
                "R_index_pip_joint",
            )
            two_stable = (
                max(two_hold["final_100ms_max_joint_speed_rad_s"][j] for j in two_digit_joints) < 0.01
                and max(two_hold["final_100ms_max_joint_tracking_error_rad"][j] for j in two_digit_joints) < 0.01
            )
            if set_gate("two_digit_control_stable_before_adding_finger", two_stable, two_hold):
                middle_phase = step_phase("add_second_opposing_finger", 1.2, all_targets, THREE_DIGIT_POSE)
                phase_results.append(middle_phase)
                middle_motion = abs(source_position(model, data, details, "R_middle_pip_joint") - OPEN_POSE["R_middle_pip_joint"])
                middle_motion_ok = middle_motion >= 0.10 and middle_phase["max_penetration_m"] <= PENETRATION_ABORT_M
                if set_gate("second_opposing_finger_motion", middle_motion_ok, {**middle_phase, "middle_pip_motion_rad": middle_motion}):
                    all_targets.update(THREE_DIGIT_POSE)
                    middle_hold = step_phase("three_digit_stable_hold", 1.2, all_targets, all_targets)
                    phase_results.append(middle_hold)
                    middle_stable = (
                        max(middle_hold["final_100ms_max_joint_speed_rad_s"][j] for j in THREE_DIGIT_POSE) < 0.01
                        and max(middle_hold["final_100ms_max_joint_tracking_error_rad"][j] for j in THREE_DIGIT_POSE) < 0.01
                        and middle_hold["max_penetration_m"] <= PENETRATION_ABORT_M
                    )
                    if not set_gate("second_opposing_finger_stable_hold", middle_stable, middle_hold):
                        stop_reason = "second opposing finger did not settle within fixed-target hold"
                else:
                    stop_reason = stop_reason or "second opposing finger did not show controlled motion"
            else:
                stop_reason = "two-digit control failed stability gate; second opposing finger not added"

        if not stop_reason:
            close_phase = step_phase("multi_digit_close_motion", 0.75, all_targets, CLOSED_POSE)
            phase_results.append(close_phase)
            all_targets.update(CLOSED_POSE)
            close_motion_ok = (
                close_phase["actual_duration_s"] >= close_phase["requested_duration_s"] - DT
                and close_phase["max_penetration_m"] <= PENETRATION_ABORT_M
            )
            set_gate("bounded_multi_digit_close_motion", close_motion_ok, close_phase)
            if not close_motion_ok:
                stop_reason = stop_reason or "bounded multi-digit close motion failed limits/penetration gate"

        if not stop_reason:
            closed_hold = step_phase("stable_multi_digit_closure", 1.2, all_targets, all_targets)
            phase_results.append(closed_hold)
            close_stable = (
                max(closed_hold["final_100ms_max_joint_tracking_error_rad"][j] for j in CLOSED_POSE) < 0.01
                and max(closed_hold["final_100ms_max_joint_speed_rad_s"][j] for j in CLOSED_POSE) < 0.01
                and closed_hold["max_penetration_m"] <= PENETRATION_ABORT_M
            )
            set_gate("stable_multi_digit_closure", close_stable, closed_hold)
            if not close_stable:
                stop_reason = "multi-digit closure failed held-state tracking/stability gate"

        if not stop_reason:
            contact_hold = step_phase("real_contact_and_load_transfer", 0.65, all_targets, all_targets)
            phase_results.append(contact_hold)
            family_fractions = contact_hold["final_100ms_finger_contact_fraction_by_family"]
            bilateral = family_fractions["thumb"] >= 0.8 and any(
                family_fractions[family] >= 0.8 for family in ("index", "middle", "ring", "pinky")
            )
            set_gate("real_finger_to_bottle_contact", bilateral, contact_hold)
            baseline_table_support = float(open_phase["mean_final_100ms_table_support_force_z_n"] or 0.0)
            final_table_support = float(contact_hold["mean_final_100ms_table_support_force_z_n"] or 0.0)
            final_hand_support_fraction = float(
                contact_hold["mean_final_100ms_hand_support_fraction_of_bottle_weight"] or 0.0
            )
            table_reduction_fraction = (baseline_table_support - final_table_support) / robot_weight_n
            contact_hold["table_support_reduction_fraction_of_bottle_weight"] = table_reduction_fraction
            load_transferred = (
                bilateral and final_hand_support_fraction >= 0.10 and table_reduction_fraction >= 0.10
            )
            set_gate("contact_driven_load_transfer", load_transferred, contact_hold)
            if not bilateral:
                stop_reason = "no force-bearing thumb/opposing-finger bottle contact"
            elif not load_transferred:
                stop_reason = "finger contact did not transfer at least 10% of bottle weight"

        if not stop_reason:
            lift_levels = (("physical_lift_5mm", 0.005, 0.0045), ("physical_lift_30mm", 0.030, 0.027), ("physical_lift_50mm", 0.050, 0.045))
            for gate_name, _, required_delta in lift_levels:
                lift_phase = step_phase(gate_name, 0.7 if required_delta <= 0.005 else 0.9, all_targets, all_targets)
                phase_results.append(lift_phase)
                measured_lift = float(lift_phase["bottle_end_position_world_m"][2] - initial_bottle_position[2])
                families = set(lift_phase["finger_contact_families"])
                # Contact continuity is also visible in the raw trace; require the active hand families to persist.
                final_families = set(lift_phase["final_100ms_finger_contact_families"])
                grip_retained = "thumb" in final_families and bool(final_families.intersection({"index", "middle", "ring", "pinky"}))
                passed = (
                    measured_lift >= required_delta
                    and grip_retained
                    and not lift_phase["final_table_contact"]
                    and lift_phase["max_penetration_m"] <= PENETRATION_ABORT_M
                )
                set_gate(gate_name, passed, {**lift_phase, "measured_bottle_lift_m": measured_lift, "minimum_required_lift_m": required_delta, "grip_retained_at_end": grip_retained, "table_contact_cleared_at_end": not lift_phase["final_table_contact"], "palm_motion_m": 0.0})
                if not passed:
                    stop_reason = f"{gate_name} failed: measured bottle lift {measured_lift:.6f} m"
                    break

        render_frame(force=True)
        save_frame(f"final_{mode}.png")
    finally:
        trace.close()
        video.close()
        renderer.close()

    final_bottle_qpos = data.qpos[
        int(details["bottle_qpos_address"]) : int(details["bottle_qpos_address"]) + 7
    ].copy()
    bottle_qpos_changed = not np.array_equal(initial_bottle_qpos, final_bottle_qpos)
    result = {
        "status": "FAIL" if stop_reason else "PASS",
        "mode": mode,
        "stop_reason": stop_reason,
        "robot_sim_branch": git_output(REPO_ROOT, "branch", "--show-current"),
        "robot_sim_head": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "vendor_repository": "https://github.com/AgibotTech/agibot_x2_urdf.git",
        "vendor_commit": git_output(VENDOR_ROOT, "rev-parse", "HEAD"),
        "vendor_worktree_clean": not bool(git_output(VENDOR_ROOT, "status", "--porcelain")),
        "source_urdf": str(URDF),
        "source_urdf_sha256": sha256(URDF),
        "generated_hand_urdf_sha256": details["hand_info"]["urdf_sha256"],
        "generated_model_xml_sha256": details["model_xml_sha256"],
        "python_version": platform.python_version(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_library_sha256": next(
            (sha256(path) for path in (Path(mujoco.__file__).parent).glob("libmujoco.so*")), None
        ),
        "numpy_version": np.__version__,
        "timestep_s": DT,
        "integrator": "implicitfast",
        "gravity_m_s2": [0.0, 0.0, -GRAVITY_M_S2],
        "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq", "nbody", "njnt", "ngeom")},
        "source_hand_links": details["hand_info"]["included_links"],
        "source_hand_joint_names": details["hand_info"]["included_joint_names"],
        "source_hand_mimic_relations": list(details["mimic_relations"].values()),
        "source_joint_ranges_rad": {name: list(bounds) for name, bounds in details["joint_ranges"].items()},
        "actuated_hand_joint_names": sorted(details["actuator_by_joint"]),
        "unused_fingers_explicitly_position_held": ["ring", "pinky"],
        "palm_fixture": {
            "initial_world_position_m": PALM_POSITION.tolist(),
            "initial_quaternion_wxyz": PALM_QUATERNION_WXYZ.tolist(),
            "palm_root_fixed_to_world": True,
            "palm_motion_actuators": 0,
            "palm_motion_during_rollout_m": 0.0,
            "simulation_only_fixture": True,
        },
        "canonical_bottle": {
            "mass_kg": bottle_mass,
            "weight_n": robot_weight_n,
            "diameter_m": CANONICAL_X2_BOTTLE_DIAMETER_M,
            "height_m": CANONICAL_X2_BOTTLE_HEIGHT_M,
            "start_body_position_world_m": list(CANONICAL_X2_BOTTLE_START_BODY_POS),
            "body_qpos_writes_after_initialization": bottle_qpos_write_count,
            "initial_hand_qpos_setup_assignments_before_physics": initial_hand_qpos_assignments,
            "free_joint_state_changed_via_mj_step": bottle_qpos_changed,
            "runtime_weld_or_attachment": False,
            "bottle_mocap": False,
            "artificial_bottle_force": False,
        },
        "contact_integrity": {
            "gravity_enabled": True,
            "friction_enabled": True,
            "bottle_on_canonical_table": True,
            "maximum_penetration_abort_m": PENETRATION_ABORT_M,
            "original_joint_limits_enforced": True,
        },
        "gates": gates,
        "phases": phase_results,
        "wall_duration_s": time.monotonic() - wall_start,
        "video": str(video_path),
        "trace": str(trace_path),
        "frames": frame_count,
        "evidence_directory": str(out_dir),
        "hardware_semantics": "not verified; hand position actuators and the fixed-palm diagnostic fixture are simulation-only abstractions",
    }
    (out_dir / f"result_{mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def check_model(mode: str, out_dir: Path) -> dict[str, object]:
    verify_vendor_pin()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, details = build_model(mode, out_dir)
    data = mujoco.MjData(model)
    open_targets = {name: float(details["joint_references"][name]) for name in details["source_joints"]}
    open_targets.update(OPEN_POSE)
    open_targets = derived_targets(open_targets, details)
    for joint_name in details["source_joints"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address] = source_target_to_raw(model, details, joint_name, open_targets[joint_name])
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qacc).all():
        raise RuntimeError("model check produced non-finite initial state")
    if len(details["source_joints"]) != 16 or len(details["mimic_relations"]) != 6:
        raise RuntimeError("expected native right-hand joint/mimic topology changed")
    open_contacts = bottle_contact_report(model, data, details)
    if open_contacts["finger_families"] or open_contacts["max_penetration_m"] > PENETRATION_ABORT_M:
        raise RuntimeError(f"known OPEN pose overlaps the bottle in static model check: {open_contacts}")
    return {
        "status": "PASS",
        "mode": mode,
        "dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq", "nbody", "njnt", "ngeom")},
        "native_hand_link_count": len(details["hand_info"]["included_links"]),
        "native_revolute_joint_count": len(details["source_joints"]),
        "mimic_relation_count": len(details["mimic_relations"]),
        "actuated_joint_count": len(details["actuator_by_joint"]),
        "bottle_mass_kg": float(model.body_mass[int(details["bottle_body_id"])]),
        "equality_count": int(model.neq),
        "open_pose_bottle_contact_families": open_contacts["finger_families"],
        "open_pose_bottle_max_penetration_m": open_contacts["max_penetration_m"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source-faithful", "simulation-only"), default="source-faithful")
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    verify_vendor_pin()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check_model:
        result = check_model(args.mode, OUT)
    else:
        result = run_experiment(args.mode, OUT)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
