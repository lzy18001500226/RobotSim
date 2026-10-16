from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


SOURCE_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
SOURCE_SHA256 = "344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259"
DEFAULT_VENDOR_ROOT = Path("/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf")
URDF_RELATIVE = Path("X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf")
DRIVER = "R_thumb_mcp_joint"
FOLLOWERS = ("R_thumb_pip_joint", "R_thumb_dip_joint")
FIXTURE_LOCKS = ("R_thumb_roll_joint", "R_thumb_abad_joint")
MIMIC_TOLERANCE_RAD = 0.003
MIMIC_SOLREF_DIRECT = (-10000.0, -200.0)
DT = 0.002
GRAVITY = 9.81
DRIVER_KP = 0.04
DRIVER_KD = 0.004
DRIVER_TORQUE_CAP_NM = 0.08
HOLD_KP = 18.0
HOLD_KD = 2.4
OPEN_Q_SOURCE = 0.08
CLOSE_Q_SOURCE = 0.40
STAGES = (
    ("OPEN_HOLD_INITIAL", 0.8, "hold_open"),
    ("CLOSE", 1.0, "close"),
    ("CLOSE_HOLD", 0.8, "hold_close"),
    ("RETURN_OPEN", 1.0, "open"),
    ("OPEN_HOLD_FINAL", 0.6, "hold_open"),
)
SETTLE_WINDOW_S = 0.2
SETTLE_MAX_ERROR_RAD = 0.01
SETTLE_MAX_SPEED_RAD_S = 0.02
MOTION_MAX_ERROR_RAD = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(vendor_root: Path) -> Path:
    actual = subprocess.check_output(["git", "-C", str(vendor_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor_root), "status", "--porcelain"], text=True).strip()
    urdf = vendor_root / URDF_RELATIVE
    if actual != SOURCE_PIN:
        raise RuntimeError(f"X2 source pin mismatch: expected {SOURCE_PIN}, got {actual}")
    if dirty:
        raise RuntimeError("pinned X2 source checkout is dirty; refusing to run")
    if not urdf.is_file() or sha256(urdf) != SOURCE_SHA256:
        raise RuntimeError("pinned X2 URDF is missing or its SHA-256 changed")
    return urdf


def source_records(root: ET.Element) -> tuple[dict[str, ET.Element], dict[str, dict[str, Any]]]:
    joints: dict[str, ET.Element] = {}
    relations: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name", ""))
        if joint.get("type") != "revolute":
            continue
        joints[name] = joint
        mimic = joint.find("mimic")
        if mimic is not None:
            relations[name] = {
                "follower": name,
                "driver": str(mimic.get("joint")),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            }
    return joints, relations


def write_thumb_urdf(source_urdf: Path, out_dir: Path) -> dict[str, Any]:
    source_root = ET.parse(source_urdf).getroot()
    included_links = {"R_palm"}
    selected_joints: list[ET.Element] = []
    changed = True
    while changed:
        changed = False
        for joint in source_root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                continue
            parent_name, child_name = parent.get("link"), child.get("link")
            if parent_name in included_links and child_name not in included_links and str(child_name).startswith("R_thumb_"):
                included_links.add(str(child_name))
                selected_joints.append(copy.deepcopy(joint))
                changed = True
    expected_links = {"R_palm", "R_thumb_roll", "R_thumb_abad", "R_thumb_mcp", "R_thumb_pip", "R_thumb_dip", "R_thumb_tip"}
    if included_links != expected_links:
        raise RuntimeError(f"unexpected right-thumb source subtree: {sorted(included_links)}")

    root = ET.Element("robot", {"name": "x2_official_right_thumb_mjstep_fixture"})
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", {"discardvisual": "false", "strippath": "false"})
    for material in source_root.findall("material"):
        root.append(copy.deepcopy(material))
    links = {str(link.get("name")): link for link in source_root.findall("link")}
    for name in sorted(included_links):
        if name not in links:
            raise RuntimeError(f"source thumb link missing: {name}")
        root.append(copy.deepcopy(links[name]))
    for joint in selected_joints:
        root.append(joint)

    mesh_paths: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise RuntimeError("official thumb mesh has no filename")
        tail = filename.split("://", 1)[-1]
        marker = "X2_URDF-v1.4.0/"
        if marker in tail:
            tail = tail.split(marker, 1)[1]
        candidate = (source_urdf.parent / tail).resolve()
        if not candidate.is_file():
            candidate = (source_urdf.parent / Path(tail).name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"official thumb mesh not found: {filename}")
        mesh.set("filename", str(candidate))
        mesh_paths.append(str(candidate))

    output = out_dir / "official_right_thumb_fixture.urdf"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return {"path": str(output), "sha256": sha256(output), "links": sorted(included_links), "joints": [str(j.get("name")) for j in selected_joints], "meshes": sorted(set(mesh_paths))}


def _name_id(model: mujoco.MjModel, objtype: Any, name: str) -> int:
    result = int(mujoco.mj_name2id(model, objtype, name))
    if result < 0:
        raise RuntimeError(f"compiled model is missing {name}")
    return result


def _mapping(model: mujoco.MjModel, source_joint: ET.Element) -> dict[str, Any]:
    name = str(source_joint.get("name"))
    jid = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    axis_node = source_joint.find("axis")
    source_axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
    compiled_axis = np.asarray(model.jnt_axis[jid], dtype=float)
    sign = 1 if float(source_axis @ compiled_axis) >= 0 else -1
    limit = source_joint.find("limit")
    if limit is None:
        raise RuntimeError(f"source thumb joint lacks limits: {name}")
    qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    q0 = float(model.qpos0[qadr])
    source_ref = 0.0
    expected_range = sorted(q0 + sign * (float(limit.get(key)) - source_ref) for key in ("lower", "upper"))
    actual_range = model.jnt_range[jid].tolist()
    if not np.allclose(actual_range, expected_range, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"compiled range mismatch for {name}: {actual_range} vs {expected_range}")
    return {
        "name": name, "jid": jid, "qadr": qadr, "dadr": dadr, "qpos0": q0,
        "source_ref": source_ref, "sign": sign, "source_axis": source_axis.tolist(),
        "compiled_axis": compiled_axis.tolist(),
        "source_range": [float(limit.get("lower")), float(limit.get("upper"))],
        "compiled_range": actual_range,
        "source_effort_limit_nm": float(limit.get("effort")),
        "source_velocity_limit_rad_s": float(limit.get("velocity")),
    }


def _add_mimic_equality(spec: mujoco.MjSpec, source_joint_by_name: dict[str, ET.Element], relation: dict[str, Any], probe: mujoco.MjModel) -> dict[str, Any]:
    follower_name, driver_name = str(relation["follower"]), str(relation["driver"])
    follower_joint = next(j for j in spec.joints if j.name == follower_name)
    driver_joint = next(j for j in spec.joints if j.name == driver_name)
    follower_map = _mapping(probe, source_joint_by_name[follower_name])
    driver_map = _mapping(probe, source_joint_by_name[driver_name])
    multiplier, offset = float(relation["multiplier"]), float(relation["offset"])
    follower_ref, driver_ref = float(follower_joint.ref), float(driver_joint.ref)
    coefficient = follower_map["sign"] * multiplier * driver_map["sign"]
    constant = follower_map["sign"] * (offset + multiplier * driver_ref - follower_ref)
    spec.add_equality(
        name=f"source_mimic_{follower_name}", type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=follower_name, name2=driver_name,
        data=[constant, coefficient, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ).solref = list(MIMIC_SOLREF_DIRECT)
    return {
        "follower": follower_name, "driver": driver_name, "multiplier": multiplier,
        "offset": offset, "compiled_constant": constant, "compiled_multiplier": coefficient,
        "equality_name": f"source_mimic_{follower_name}", "solref_direct": list(MIMIC_SOLREF_DIRECT),
    }


def _add_motor(spec: mujoco.MjSpec, joint_name: str, name: str, cap: float) -> None:
    spec.add_actuator(name=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, target=joint_name,
                      ctrllimited=True, ctrlrange=[-cap, cap], forcelimited=True,
                      forcerange=[-cap, cap])


def _add_fixture_lock(spec: mujoco.MjSpec, joint_name: str) -> None:
    lock = spec.add_equality(
        name=f"fixture_lock_{joint_name}",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1=joint_name,
        data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    lock.solref = list(MIMIC_SOLREF_DIRECT)


def build_model(source_urdf: Path, out_dir: Path, full_x2: bool = False, timestep_s: float = DT,
                exclude_fixture_root_contact: bool = True) -> tuple[mujoco.MjModel, dict[str, Any]]:
    source_root = ET.parse(source_urdf).getroot()
    all_source_joints, all_relations = source_records(source_root)
    if full_x2:
        spec = mujoco.MjSpec.from_file(str(source_urdf))
        selected_joints, selected_relations = all_source_joints, all_relations
        subtree: dict[str, Any] = {"kind": "complete_official_x2_urdf"}
    else:
        subtree = write_thumb_urdf(source_urdf, out_dir)
        spec = mujoco.MjSpec.from_file(subtree["path"])
        selected_joints = {name: all_source_joints[name] for name in (DRIVER, *FOLLOWERS, *FIXTURE_LOCKS)}
        selected_relations = {name: all_relations[name] for name in FOLLOWERS}

    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    spec.option.timestep = timestep_s
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 100
    spec.option.tolerance = 1e-8
    spec.option.gravity = [0.0, 0.0, -GRAVITY]
    for joint in spec.joints:
        if joint.name in all_source_joints and joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.solref_limit = list(MIMIC_SOLREF_DIRECT)
    if not full_x2:
        palm = spec.worldbody.bodies[0]
        palm.pos = [0.30, 0.0, 0.90]
        palm.quat = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
        spec.worldbody.add_geom(name="fixture_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0.0, 0.0, 0.05], friction=[1.0, 0.01, 0.001],
                                condim=4, rgba=[0.24, 0.27, 0.30, 1.0])
        if exclude_fixture_root_contact:
            # The pinned collision proxies overlap at this fixture's reference pose.
            spec.add_exclude(name="fixture_overlapping_palm_thumb_mcp",
                             bodyname1="R_palm", bodyname2="R_thumb_mcp")

    probe = spec.compile()
    source_mappings = {name: _mapping(probe, joint) for name, joint in selected_joints.items()}
    for name in FIXTURE_LOCKS:
        _add_fixture_lock(spec, name)
    mimic_specs = [_add_mimic_equality(spec, selected_joints, selected_relations[name], probe)
                   for name in sorted(selected_relations)]

    follower_names = set(selected_relations)
    actuator_joint_names: list[str] = []
    if full_x2:
        for joint in spec.joints:
            name = joint.name
            if not name or joint.type != mujoco.mjtJoint.mjJNT_HINGE or name in follower_names or name in FIXTURE_LOCKS:
                continue
            if name == DRIVER:
                _add_motor(spec, name, "thumb_driver_motor", DRIVER_TORQUE_CAP_NM)
            else:
                source_joint = all_source_joints.get(name)
                limit = source_joint.find("limit") if source_joint is not None else None
                effort = float(limit.get("effort")) if limit is not None else 33.5
                servo = spec.add_actuator(name=f"fixture_hold_{name}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                          target=name, ctrllimited=False, forcelimited=True,
                                          forcerange=[-effort, effort])
                servo.set_to_position(kp=HOLD_KP, kv=HOLD_KD)
            actuator_joint_names.append(name)
    else:
        _add_motor(spec, DRIVER, "thumb_driver_motor", DRIVER_TORQUE_CAP_NM)
        actuator_joint_names.append(DRIVER)

    model = spec.compile()
    driver_map = _mapping(model, selected_joints[DRIVER])
    expected_equalities = len(selected_relations) + len(FIXTURE_LOCKS)
    if model.neq != expected_equalities:
        raise RuntimeError(f"expected {expected_equalities} mimic/fixture equalities, compiled {model.neq}")
    if model.nu != len(actuator_joint_names):
        raise RuntimeError(f"unexpected actuator count {model.nu}, expected {len(actuator_joint_names)}")
    actuator_ids = {str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)): i for i in range(model.nu)}
    driver_actuator = actuator_ids.get("thumb_driver_motor")
    if driver_actuator is None:
        raise RuntimeError("compiled model lost the independent thumb driver motor")
    if any(name in actuator_joint_names for name in selected_relations):
        raise RuntimeError("mimic follower unexpectedly has an independent actuator")
    if not np.allclose(model.eq_solref, np.tile(MIMIC_SOLREF_DIRECT, (model.neq, 1))):
        raise RuntimeError("compiled equality solver parameters differ from the unchanged baseline")
    info = {
        "full_x2": full_x2,
        "fixture_locks": list(FIXTURE_LOCKS),
        "fixture_contact_exclusions": [["R_palm", "R_thumb_mcp"]]
        if not full_x2 and exclude_fixture_root_contact else [],
        "dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq", "nbody", "ngeom", "nmesh")},
        "subtree": subtree, "mimic_relations": mimic_specs, "source_joint_mappings": source_mappings,
        "driver_mapping": driver_map, "driver_actuator_id": driver_actuator,
        "actuator_joint_names": actuator_joint_names, "timestep_s": float(model.opt.timestep),
        "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
        "solver_iterations": int(model.opt.iterations), "solver_tolerance": float(model.opt.tolerance),
        "gravity_m_s2": model.opt.gravity.tolist(),
    }
    return model, info


def source_position(data: mujoco.MjData, mapping: dict[str, Any]) -> float:
    return float(mapping["source_ref"] + mapping["sign"] * (data.qpos[mapping["qadr"]] - mapping["qpos0"]))


def source_velocity(data: mujoco.MjData, mapping: dict[str, Any]) -> float:
    return float(mapping["sign"] * data.qvel[mapping["dadr"]])


def source_acceleration(data: mujoco.MjData, mapping: dict[str, Any]) -> float:
    return float(mapping["sign"] * data.qacc[mapping["dadr"]])


def controller_torque(model: mujoco.MjModel, data: mujoco.MjData, info: dict[str, Any], target_source: float) -> float:
    before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
    driver = info["driver_mapping"]
    basis = {DRIVER: 1.0}
    for relation in info["mimic_relations"]:
        if relation["driver"] == DRIVER:
            basis[str(relation["follower"])] = float(relation["compiled_multiplier"])
    projected_bias = 0.0
    for joint_name, coefficient in basis.items():
        jid = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        projected_bias += coefficient * float(data.qfrc_bias[int(model.jnt_dofadr[jid])])
    target_compiled = driver["qpos0"] + driver["sign"] * (target_source - driver["source_ref"])
    error = target_compiled - float(data.qpos[driver["qadr"]])
    torque = projected_bias + DRIVER_KP * error - DRIVER_KD * float(data.qvel[driver["dadr"]])
    torque = float(np.clip(torque, -DRIVER_TORQUE_CAP_NM, DRIVER_TORQUE_CAP_NM))
    data.ctrl[info["driver_actuator_id"]] = torque
    if not np.array_equal(before_qpos, data.qpos) or not np.array_equal(before_qvel, data.qvel):
        raise RuntimeError("controller modified qpos/qvel; only mj_step may advance state")
    return torque


def stage_target(kind: str, elapsed: float, duration: float) -> float:
    if kind == "hold_open":
        return OPEN_Q_SOURCE
    if kind == "hold_close":
        return CLOSE_Q_SOURCE
    phase = min(max(elapsed / duration, 0.0), 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    if kind == "close":
        return OPEN_Q_SOURCE + blend * (CLOSE_Q_SOURCE - OPEN_Q_SOURCE)
    if kind == "open":
        return CLOSE_Q_SOURCE - blend * (CLOSE_Q_SOURCE - OPEN_Q_SOURCE)
    raise ValueError(f"unknown stage kind: {kind}")


def equality_residuals(data: mujoco.MjData, info: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    mappings = info["source_joint_mappings"]
    for relation in info["mimic_relations"]:
        driver = source_position(data, mappings[str(relation["driver"])])
        follower = source_position(data, mappings[str(relation["follower"])])
        result[str(relation["follower"])] = follower - (
            float(relation["offset"]) + float(relation["multiplier"]) * driver
        )
    return result


def fixture_lock_residuals(data: mujoco.MjData, mappings: dict[str, Any]) -> dict[str, float]:
    return {name: source_position(data, mappings[name]) - mappings[name]["source_ref"] for name in FIXTURE_LOCKS}


def efc_residuals(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    names: dict[int, str] = {
        i: str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i))
        for i in range(model.neq)
    }
    result: dict[str, float] = {}
    for row in range(data.nefc):
        if int(data.efc_type[row]) == equality_type:
            eq_id = int(data.efc_id[row])
            result[names[eq_id]] = float(data.efc_pos[row])
    return result


def max_penetration(data: mujoco.MjData) -> float:
    return max((max(0.0, -float(data.contact[i].dist)) for i in range(data.ncon)), default=0.0)


def max_limit_violation(model: mujoco.MjModel, data: mujoco.MjData, mappings: dict[str, Any]) -> float:
    maximum = 0.0
    for mapping in mappings.values():
        jid = int(mapping["jid"])
        if model.jnt_limited[jid]:
            q = float(data.qpos[mapping["qadr"]])
            low, high = model.jnt_range[jid]
            maximum = max(maximum, float(low) - q, q - float(high), 0.0)
    return maximum


def camera_for_thumb() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.30, 0.0, 0.88]
    camera.distance = 0.55
    camera.azimuth = 135.0
    camera.elevation = -15.0
    return camera


def run_experiment(source_urdf: Path, out_dir: Path, full_x2: bool = False, video: bool = True, timestep_s: float = DT) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, info = build_model(source_urdf, out_dir, full_x2=full_x2, timestep_s=timestep_s)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    source_joint_nodes = source_records(ET.parse(source_urdf).getroot())[0]
    mappings = {name: _mapping(model, node) for name, node in source_joint_nodes.items()
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0}
    info["source_joint_mappings"] = mappings
    for relation in info["mimic_relations"]:
        if relation["driver"] not in mappings or relation["follower"] not in mappings:
            raise RuntimeError("mimic mapping missing a compiled driver or follower")

    actuator_ids = {str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)): i for i in range(model.nu)}
    driver_aid = actuator_ids["thumb_driver_motor"]
    if full_x2:
        for name, aid in actuator_ids.items():
            if name.startswith("fixture_hold_"):
                joint_name = name[len("fixture_hold_"):]
                data.ctrl[aid] = mappings[joint_name]["qpos0"]

    trace_path = out_dir / ("full_x2_trace.jsonl" if full_x2 else "thumb_trace.jsonl")
    video_path = out_dir / "thumb_mjstep.mp4"
    renderer = mujoco.Renderer(model, height=480, width=640) if video and not full_x2 else None
    camera = camera_for_thumb() if renderer is not None else None
    frames: list[np.ndarray] = []
    record_count = 0
    max_residual = max_lock_residual = max_penetration_m = max_effort = max_motion_error = max_limit = 0.0
    finite_all_steps = True
    hold_records: dict[str, list[dict[str, float]]] = {}
    stage_summaries: list[dict[str, Any]] = []

    with trace_path.open("w", encoding="utf-8") as trace:
        for stage_name, duration, kind in STAGES:
            stage_start = float(data.time)
            steps = round(duration / timestep_s)
            samples: list[dict[str, float]] = []
            stage_limit = 0.0
            for step in range(steps):
                elapsed = step * timestep_s
                target = stage_target(kind, elapsed, duration)
                mujoco.mj_forward(model, data)
                command = controller_torque(model, data, info, target)
                mujoco.mj_forward(model, data)
                driver_map = mappings[DRIVER]
                q_driver = source_position(data, driver_map)
                v_driver = source_velocity(data, driver_map)
                a_driver = source_acceleration(data, driver_map)
                residuals = equality_residuals(data, info)
                fixture_residuals = fixture_lock_residuals(data, mappings)
                efc = efc_residuals(model, data)
                max_residual = max(max_residual, max((abs(x) for x in residuals.values()), default=0.0))
                max_lock_residual = max(max_lock_residual, max((abs(x) for x in fixture_residuals.values()), default=0.0))
                penetration = max_penetration(data)
                max_penetration_m = max(max_penetration_m, penetration)
                stage_limit = max(stage_limit, max_limit_violation(model, data, mappings))
                max_limit = max(max_limit, stage_limit)
                effort = abs(float(data.actuator_force[driver_aid]))
                max_effort = max(max_effort, effort)
                finite_all_steps = finite_all_steps and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                error = q_driver - target
                if kind in ("close", "open"):
                    max_motion_error = max(max_motion_error, abs(error))
                joints: dict[str, Any] = {}
                for name, mapping in mappings.items():
                    dadr = int(mapping["dadr"])
                    joints[name] = {
                        "q_source_rad": source_position(data, mapping),
                        "qdot_source_rad_s": source_velocity(data, mapping),
                        "qacc_source_rad_s2": source_acceleration(data, mapping),
                        "qfrc_bias_compiled_nm": float(data.qfrc_bias[dadr]),
                        "qfrc_constraint_compiled_nm": float(data.qfrc_constraint[dadr]),
                        "qfrc_actuator_compiled_nm": float(data.qfrc_actuator[dadr]),
                        "source_range_rad": mapping["source_range"],
                    }
                record = {
                    "step": record_count, "stage": stage_name, "time_s": float(data.time),
                    "target_source_rad": target, "driver_source_rad": q_driver,
                    "driver_velocity_source_rad_s": v_driver, "driver_acceleration_source_rad_s2": a_driver,
                    "driver_target_error_rad": error, "driver_command_torque_nm": command,
                    "driver_actuator_force_nm": float(data.actuator_force[driver_aid]),
                    "mimic_residual_source_rad": residuals, "mujoco_efc_residual_rad": efc,
                    "fixture_lock_residual_source_rad": fixture_residuals,
                    "contacts": int(data.ncon), "max_penetration_m": penetration, "joints": joints,
                }
                trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                record_count += 1
                sample = {"error": abs(error), "speed": abs(v_driver)}
                samples.append(sample)
                if kind.startswith("hold_"):
                    hold_records.setdefault(stage_name, []).append(sample)
                mujoco.mj_step(model, data)
                frame_stride = max(1, round(0.02 / timestep_s))
                if renderer is not None and record_count % frame_stride == 0:
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    frames.append(renderer.render())
            stage_summaries.append({
                "stage": stage_name, "duration_s": duration, "start_time_s": stage_start,
                "end_time_s": float(data.time), "steps": steps,
                "max_driver_tracking_error_rad": max((x["error"] for x in samples), default=0.0),
                "max_abs_driver_velocity_rad_s": max((x["speed"] for x in samples), default=0.0),
                "max_compiled_limit_violation_rad": stage_limit,
            })

    if renderer is not None:
        renderer.close()
        if frames:
            imageio.mimsave(video_path, frames, fps=30)

    def settled(stage_name: str) -> dict[str, Any]:
        samples = hold_records.get(stage_name, [])
        tail = samples[-max(1, round(SETTLE_WINDOW_S / timestep_s)):]
        max_error = max((x["error"] for x in tail), default=float("inf"))
        max_speed = max((x["speed"] for x in tail), default=float("inf"))
        return {"window_s": len(tail) * timestep_s, "max_target_error_rad": max_error,
                "max_speed_rad_s": max_speed,
                "pass": max_error <= SETTLE_MAX_ERROR_RAD and max_speed <= SETTLE_MAX_SPEED_RAD_S}

    initial_settle, close_settle, final_settle = (
        settled("OPEN_HOLD_INITIAL"), settled("CLOSE_HOLD"), settled("OPEN_HOLD_FINAL")
    )
    gates = {
        "all_mimic_residuals_below_0_003_rad": max_residual < MIMIC_TOLERANCE_RAD,
        "unused_thumb_axes_fixture_locked": max_lock_residual < MIMIC_TOLERANCE_RAD,
        "open_hold_initial_settled": initial_settle["pass"],
        "close_hold_settled": close_settle["pass"],
        "return_open_settled": final_settle["pass"],
        "motion_tracking_bounded": max_motion_error <= MOTION_MAX_ERROR_RAD,
        "source_joint_limits_preserved": max_limit <= 1e-8,
        "actuator_effort_bounded": max_effort <= DRIVER_TORQUE_CAP_NM + 1e-9,
        "finite_and_gravity_enabled": finite_all_steps and np.allclose(model.opt.gravity, [0.0, 0.0, -GRAVITY]),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "full_x2_context" if full_x2 else "minimal_right_thumb_fixture",
        "fixture_contact_exclusions": info["fixture_contact_exclusions"],
        "mujoco_version": mujoco.__version__, "dimensions": info["dimensions"],
        "source_pin": SOURCE_PIN, "source_urdf_sha256": SOURCE_SHA256,
        "physics": {key: info[key] for key in ("timestep_s", "integrator", "solver", "solver_iterations", "solver_tolerance", "gravity_m_s2")},
        "transmission": {
            "classification": "SIMULATION_ONLY compliant native MuJoCo joint-equality coupling",
            "driver": DRIVER, "followers": sorted(r["follower"] for r in relation_specs_for_driver(info)),
            "follower_actuators": 0, "follower_state_writes_during_rollout": 0,
            "equality_solref_direct": list(MIMIC_SOLREF_DIRECT),
        "hand_joint_limit_solref_direct": list(MIMIC_SOLREF_DIRECT),
            "controller": {"type": "projected-gravity feedforward plus PD torque", "kp_nm_per_rad": DRIVER_KP,
                           "kd_nm_s_per_rad": DRIVER_KD, "driver_torque_cap_nm": DRIVER_TORQUE_CAP_NM},
        },
        "settling_diagnostic_thresholds_not_established_acceptance": {
            "window_s": SETTLE_WINDOW_S, "max_target_error_rad": SETTLE_MAX_ERROR_RAD,
            "max_speed_rad_s": SETTLE_MAX_SPEED_RAD_S, "motion_tracking_error_rad": MOTION_MAX_ERROR_RAD,
        },
        "max_mimic_residual_rad": max_residual, "max_penetration_m": max_penetration_m,
        "max_fixture_lock_residual_rad": max_lock_residual,
        "max_driver_actuator_effort_nm": max_effort, "max_motion_target_error_rad": max_motion_error,
        "hold_settling": {"initial_open": initial_settle, "close": close_settle, "final_open": final_settle},
        "gates": gates, "stages": stage_summaries, "trace": str(trace_path),
        "video": str(video_path) if video_path.is_file() else None,
        "trace_sha256": sha256(trace_path), "video_sha256": sha256(video_path) if video_path.is_file() else None,
        "final_time_s": float(data.time), "steps": record_count,
    }
    result_path = out_dir / ("full_x2_result.json" if full_x2 else "thumb_result.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def relation_specs_for_driver(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in info["mimic_relations"] if r["driver"] == DRIVER]


def main() -> int:
    parser = argparse.ArgumentParser(description="Genuine MuJoCo mj_step X2 thumb mimic diagnostic")
    parser.add_argument("--vendor-root", type=Path, default=Path(os.environ.get("AGIBOT_X2_VENDOR_ROOT", DEFAULT_VENDOR_ROOT)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-x2", action="store_true", help="run the same thumb transmission in complete fixed-base X2 context")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--timestep", type=float, default=DT)
    args = parser.parse_args()
    source_urdf = verify_source(args.vendor_root)
    result = run_experiment(source_urdf, args.output, full_x2=args.full_x2, video=not args.no_video, timestep_s=args.timestep)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "gates": result["gates"],
                      "result": result["trace"].replace("_trace.jsonl", "_result.json")}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
