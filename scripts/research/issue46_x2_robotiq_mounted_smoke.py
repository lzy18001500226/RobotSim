#!/usr/bin/env python3
"""Compare the pinned X2 head contact and exercise a mounted Robotiq in isolation."""

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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-x2-head-contact-smoke-20261009"
)
DEFAULT_STATION = [-0.08, 0.0, 0.68]
EXCLUDE_NAME = "issue46_simonly_head_pitch_torso"
GRIP_ACTUATOR = "rq_fingers_actuator"
DRIVER_JOINT = "rq_right_driver_joint"
FOLLOWER_JOINT = "rq_left_driver_joint"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def body_id(model: mujoco.MjModel, body: str) -> int:
    return m0.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, body)


def joint_state(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> dict[str, float]:
    jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return {
        "qpos_rad": float(data.qpos[qadr]),
        "qvel_rad_s": float(data.qvel[dadr]),
        "qacc_rad_s2": float(data.qacc[dadr]),
        "range_rad": model.jnt_range[jid].astype(float).tolist(),
    }


def robot_body_names(model: mujoco.MjModel) -> set[str]:
    names = set()
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and (bname.endswith("_link") or bname.startswith("rq_")):
            names.add(bname)
    names.update({"pelvis", "torso_link", "head_pitch_link"})
    return names


def describe_contacts(model: mujoco.MjModel, data: mujoco.MjData, robot_names: set[str]) -> dict[str, Any]:
    rows = []
    robot_self = []
    robot_bottle = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n1, n2 = name(model, mujoco.mjtObj.mjOBJ_BODY, b1), name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        gname1, gname2 = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1), name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        force = np.zeros(6)
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, index, force)
        row = {
            "geom1_id": g1, "geom1": gname1, "body1": n1,
            "geom2_id": g2, "geom2": gname2, "body2": n2,
            "distance_m": float(contact.dist), "position_world_m": contact.pos.astype(float).tolist(),
            "frame_world": contact.frame.reshape(3, 3).astype(float).tolist(),
            "solver_force_local_n_prestep_or_current": force.astype(float).tolist()
            if contact.efc_address >= 0 else None,
        }
        rows.append(row)
        if n1 in robot_names and n2 in robot_names and b1 != b2:
            robot_self.append(row)
        if "m0_bottle" in (n1, n2):
            other = n2 if n1 == "m0_bottle" else n1
            if other in robot_names:
                robot_bottle.append(row)
    return {"all_contacts": rows, "robot_self_contacts": robot_self, "robot_bottle_contacts": robot_bottle}


def geom_pose_data(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> list[dict[str, Any]]:
    bid = body_id(model, body)
    records = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid or (model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0):
            continue
        records.append({
            "geom_id": gid, "geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "geom_type": int(model.geom_type[gid]), "size": model.geom_size[gid].astype(float).tolist(),
            "local_position_m": model.geom_pos[gid].astype(float).tolist(),
            "local_quaternion_wxyz": model.geom_quat[gid].astype(float).tolist(),
            "world_position_m": data.geom_xpos[gid].astype(float).tolist(),
            "world_rotation_matrix": data.geom_xmat[gid].reshape(3, 3).astype(float).tolist(),
            "contype": int(model.geom_contype[gid]), "conaffinity": int(model.geom_conaffinity[gid]),
        })
    return records


def set_source_neutral(model: mujoco.MjModel, data: mujoco.MjData, station: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    root = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qadr = int(model.jnt_qposadr[root])
    # A free-joint translation is the root body's world position; the XML body
    # origin is not added to this qpos as it is for a fixed body.
    data.qpos[qadr:qadr + 3] = station
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for joint in ("head_yaw_joint", "head_pitch_joint"):
        data.qpos[m0.qpos_id(model, joint)] = 0.0
    data.qpos[m0.qpos_id(model, "left_elbow_joint")] = -0.01
    data.qpos[m0.qpos_id(model, "right_elbow_joint")] = -0.01
    mujoco.mj_forward(model, data)


def set_mounted_neutral(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    for joint in ("head_yaw_joint", "head_pitch_joint"):
        data.qpos[m0.qpos_id(model, joint)] = 0.0
    data.qpos[m0.qpos_id(model, "left_elbow_joint")] = -0.01
    data.qpos[m0.qpos_id(model, "right_elbow_joint")] = -0.01
    mujoco.mj_forward(model, data)


def model_comparison(source_model: mujoco.MjModel, mounted_model: mujoco.MjModel,
                     ident: dict[str, Any], station: np.ndarray) -> dict[str, Any]:
    source_data, mounted_data = mujoco.MjData(source_model), mujoco.MjData(mounted_model)
    set_source_neutral(source_model, source_data, station)
    set_mounted_neutral(mounted_model, mounted_data)
    source_names = robot_body_names(source_model)
    mounted_names = robot_body_names(mounted_model)
    source_contacts = describe_contacts(source_model, source_data, source_names)
    mounted_contacts = describe_contacts(mounted_model, mounted_data, mounted_names)
    source_pair = source_contacts["robot_self_contacts"]
    mounted_pair = mounted_contacts["robot_self_contacts"]
    source_excludes = []
    x2_xml = ET.parse(ident["x2_checkout"]["path"] + "/" + str(m0.X2_MJCF)).getroot()
    contact = x2_xml.find("contact")
    if contact is not None:
        source_excludes = [copy.deepcopy(e.attrib) for e in contact.findall("exclude")]
    src_geom = {b: geom_pose_data(source_model, source_data, b) for b in ("head_pitch_link", "torso_link")}
    mounted_geom = {b: geom_pose_data(mounted_model, mounted_data, b) for b in ("head_pitch_link", "torso_link")}
    pose_delta = {}
    for body in src_geom:
        s, t = src_geom[body][0], mounted_geom[body][0]
        pose_delta[body] = {
            "world_position_max_abs_delta_m": float(np.max(np.abs(np.asarray(s["world_position_m"]) - np.asarray(t["world_position_m"])))),
            "world_rotation_max_abs_delta": float(np.max(np.abs(np.asarray(s["world_rotation_matrix"]) - np.asarray(t["world_rotation_matrix"])))),
        }
    source_limits, source_velocity = m0.all_limited_joint_checks(
        source_model, source_data,
        {k: v["velocity"] for k, v in ident["urdf_joint_limits"].items()
         if math.isfinite(v["velocity"]) and v["velocity"] > 0},
    )
    return {
        "head_pose_source_neutral_rad": {j: float(source_data.qpos[m0.qpos_id(source_model, j)])
                                          for j in ("head_yaw_joint", "head_pitch_joint")},
        "comparison_arm_pose_rad": {
            "left_elbow_joint": float(source_data.qpos[m0.qpos_id(source_model, "left_elbow_joint")]),
            "right_elbow_joint": float(source_data.qpos[m0.qpos_id(source_model, "right_elbow_joint")]),
        },
        "source_head_joint_ranges_rad": {
            j: source_model.jnt_range[m0.obj_id(source_model, mujoco.mjtObj.mjOBJ_JOINT, j)].astype(float).tolist()
            for j in ("head_yaw_joint", "head_pitch_joint")},
        "fixed_base_position_world_m": station.astype(float).tolist(),
        "source_model": {"ncon": int(source_data.ncon), **source_contacts, "geom_transforms": src_geom,
                         "source_explicit_excludes": source_excludes,
                         "source_joint_limit_violations": source_limits,
                         "source_joint_velocity_violations": source_velocity},
        "mounted_model_before_exclusion": {"ncon": int(mounted_data.ncon), **mounted_contacts,
                                            "geom_transforms": mounted_geom},
        "transform_deltas": pose_delta,
        "head_torso_contact_match": {
            "source_count": len([c for c in source_pair if {c["body1"], c["body2"]} == {"head_pitch_link", "torso_link"}]),
            "mounted_count": len([c for c in mounted_pair if {c["body1"], c["body2"]} == {"head_pitch_link", "torso_link"}]),
            "source_contact_distance_m": source_pair[0]["distance_m"] if source_pair else None,
            "mounted_contact_distance_m": mounted_pair[0]["distance_m"] if mounted_pair else None,
            "contact_position_max_abs_delta_m": float(np.max(np.abs(np.asarray(source_pair[0]["position_world_m"]) -
                                                                            np.asarray(mounted_pair[0]["position_world_m"]))))
                if source_pair and mounted_pair else None,
            "only_robot_self_pair_in_both_models": (
                len(source_pair) == len(mounted_pair) == 1 and
                {source_pair[0]["body1"], source_pair[0]["body2"]} == {"head_pitch_link", "torso_link"} and
                {mounted_pair[0]["body1"], mounted_pair[0]["body2"]} == {"head_pitch_link", "torso_link"}),
        },
    }


def create_exclusion_variant(source_path: Path, output_path: Path) -> dict[str, Any]:
    source_root = ET.parse(source_path).getroot()
    variant = copy.deepcopy(source_root)
    contact = variant.find("contact")
    contact_created = contact is None
    if contact is None:
        contact = ET.SubElement(variant, "contact")
    exact = []
    for node in contact.findall("exclude"):
        if {node.get("body1"), node.get("body2")} == {"head_pitch_link", "torso_link"}:
            exact.append(node)
    if exact:
        raise RuntimeError("Mounted base model already contains the authorized exclusion; refusing duplicate")
    new_node = ET.SubElement(contact, "exclude", {
        "name": EXCLUDE_NAME, "body1": "head_pitch_link", "body2": "torso_link"})
    ET.indent(variant, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(variant).write(output_path, encoding="utf-8", xml_declaration=True)
    verify = ET.parse(output_path).getroot()
    all_excludes = [dict(node.attrib) for node in (verify.find("contact") or []).findall("exclude")]
    matches = [e for e in all_excludes if {e.get("body1"), e.get("body2")} == {"head_pitch_link", "torso_link"}]
    if len(matches) != 1 or matches[0].get("name") != EXCLUDE_NAME:
        raise RuntimeError("Experimental model does not contain exactly the authorized exclusion")
    if contact_created:
        contact.remove(new_node)
        variant.remove(contact)
    else:
        contact.remove(new_node)
    def signature(element: ET.Element) -> tuple[Any, ...]:
        return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
                tuple(signature(child) for child in element))

    baseline_after_remove = signature(variant)
    baseline = signature(source_root)
    if baseline_after_remove != baseline:
        raise RuntimeError("Experimental model differs from base model beyond the one explicit exclusion")
    return {"name": EXCLUDE_NAME, "body1": "head_pitch_link", "body2": "torso_link",
            "exactly_one_added": True, "no_other_model_xml_change": True,
            "prior_exclusion_count": len(source_root.findall("./contact/exclude")),
            "variant_exclusion_count": len(all_excludes), "created_contact_section": contact_created}


def set_controller_targets(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[dict[str, Any]], dict[str, float]]:
    refs = m0.controller_config(model)
    targets = {ref["joint"]: float(data.qpos[ref["qpos_id"]]) for ref in refs}
    return refs, targets


def pad_gap(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    left = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_left_pad1", "rq_left_pad2")]
    right = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("rq_right_pad1", "rq_right_pad2")]
    return float(np.linalg.norm(np.mean(data.geom_xpos[left], axis=0) - np.mean(data.geom_xpos[right], axis=0)))


def run_cycle(model: mujoco.MjModel, data: mujoco.MjData, ident: dict[str, Any], out: Path,
              open_command: float = 0.0) -> dict[str, Any]:
    if not 0.0 <= open_command <= 255.0:
        raise ValueError("OPEN actuator command must remain inside the Menagerie ctrlrange [0, 255]")
    refs, targets = set_controller_targets(model, data)
    grip_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    driver_jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, DRIVER_JOINT)
    follower_jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, FOLLOWER_JOINT)
    driver_qadr, driver_dadr = int(model.jnt_qposadr[driver_jid]), int(model.jnt_dofadr[driver_jid])
    follower_qadr, follower_dadr = int(model.jnt_qposadr[follower_jid]), int(model.jnt_dofadr[follower_jid])
    robot_names = robot_body_names(model)
    source_velocity = {k: v["velocity"] for k, v in ident["urdf_joint_limits"].items()
                       if math.isfinite(v["velocity"]) and v["velocity"] > 0}
    joint_start = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): float(data.qpos[model.jnt_qposadr[j]])
                   for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)}
    initial_contacts = describe_contacts(model, data, robot_names)
    if initial_contacts["robot_self_contacts"]:
        raise RuntimeError("Additional robot self-contact remains after the one authorized exclusion")
    if initial_contacts["robot_bottle_contacts"]:
        raise RuntimeError("Bottle is not clear of the mounted robot at smoke-test initialization")
    if initial_contacts["all_contacts"]:
        # The only expected environmental contact in this fixture is bottle/table support.
        unexpected = [c for c in initial_contacts["all_contacts"]
                      if "m0_bottle" not in (c["body1"], c["body2"]) or
                      not (c["body1"].startswith("m0_table") or c["body2"].startswith("m0_table"))]
        if unexpected:
            raise RuntimeError(f"Unexpected active contact remains before the smoke test: {unexpected[0]}")

    trace_path = out / "raw" / "mounted_open_close_trace.jsonl"
    contact_path = out / "raw" / "mounted_open_close_contacts.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out / "mounted_open_close.mp4"), fps=25, codec="libx264", quality=7)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.05, 0.0, 0.95]
    camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -12
    phases = [
        ("INITIAL_HOLD", 0.20, open_command, open_command),
        ("OPEN", 0.25, open_command, open_command),
        ("CLOSE", 0.60, open_command, 255.0),
        ("REOPEN", 0.60, 255.0, open_command),
        ("HOLD", 0.25, open_command, open_command),
    ]
    phase_rows = []
    all_rows = []
    contact_rows = []
    limit_violations: list[dict[str, Any]] = []
    velocity_violations: list[dict[str, Any]] = []
    max_non_gripper_deviation = {joint: 0.0 for joint in joint_start if not joint.startswith("rq_")}
    max_non_gripper_speed = {joint: 0.0 for joint in max_non_gripper_deviation}
    actuator_effort_max = {name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid): 0.0 for aid in range(model.nu)}
    qvel_limit = max([float(v) for v in source_velocity.values()] or [0.0])
    total_step = 0
    first_bottle_robot_contact = None
    first_unexpected_self_contact = None
    gap_open, gap_closed, gap_reopened = None, None, None
    frame_step = 0
    try:
        with trace_path.open("w", encoding="utf-8") as trace_stream, contact_path.open("w", encoding="utf-8") as contact_stream:
            for phase, duration, grip_from, grip_to in phases:
                steps = int(round(duration / float(model.opt.timestep)))
                phase_start_time = float(data.time)
                phase_start_gap = pad_gap(model, data)
                phase_start_driver = float(data.qpos[driver_qadr])
                phase_peak_force = 0.0
                phase_contacts = 0
                for k in range(steps):
                    u = (k + 1) / steps
                    s, _ = m0.quintic(u)
                    ctrl_value = grip_from + s * (grip_to - grip_from)
                    mujoco.mj_forward(model, data)
                    m0.apply_controller(model, data, refs, targets, {})
                    data.ctrl[grip_id] = ctrl_value
                    mujoco.mj_step(model, data)
                    total_step += 1
                    ranges, velocities = m0.all_limited_joint_checks(model, data, source_velocity)
                    if ranges:
                        limit_violations.extend({"step": total_step, **item} for item in ranges)
                    if velocities:
                        velocity_violations.extend({"step": total_step, **item} for item in velocities)
                    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()):
                        raise RuntimeError(f"Non-finite state at step {total_step}")
                    contacts = describe_contacts(model, data, robot_names)
                    if contacts["robot_bottle_contacts"] and first_bottle_robot_contact is None:
                        first_bottle_robot_contact = {"step": total_step, "time_s": float(data.time),
                                                      "contacts": contacts["robot_bottle_contacts"]}
                    if contacts["robot_self_contacts"] and first_unexpected_self_contact is None:
                        first_unexpected_self_contact = {"step": total_step, "time_s": float(data.time),
                                                         "contacts": contacts["robot_self_contacts"]}
                    phase_contacts += len(contacts["all_contacts"])
                    for aid in range(model.nu):
                        aname = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                        actuator_effort_max[aname] = max(actuator_effort_max[aname], abs(float(data.actuator_force[aid])))
                    non_gripper_state = {}
                    for joint, start_q in joint_start.items():
                        jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                        q, v = float(data.qpos[qadr]), float(data.qvel[dadr])
                        if not joint.startswith("rq_"):
                            dev = abs(q - start_q)
                            max_non_gripper_deviation[joint] = max(max_non_gripper_deviation[joint], dev)
                            max_non_gripper_speed[joint] = max(max_non_gripper_speed[joint], abs(v))
                            non_gripper_state[joint] = {"qpos_rad": q, "qvel_rad_s": v}
                    contact_record = {"step": total_step, "time_s": float(data.time), "phase": phase,
                                      **contacts}
                    contact_rows.append(contact_record)
                    contact_stream.write(json.dumps(contact_record, sort_keys=True) + "\n")
                    row = {
                        "step": total_step, "time_s": float(data.time), "phase": phase,
                        "gripper_ctrl": float(data.ctrl[grip_id]),
                        "gripper_actuator_force": float(data.actuator_force[grip_id]),
                        "driver_qpos_rad": float(data.qpos[driver_qadr]),
                        "driver_qvel_rad_s": float(data.qvel[driver_dadr]),
                        "driver_qacc_rad_s2": float(data.qacc[driver_dadr]),
                        "follower_qpos_rad": float(data.qpos[follower_qadr]),
                        "follower_qvel_rad_s": float(data.qvel[follower_dadr]),
                        "follower_qacc_rad_s2": float(data.qacc[follower_dadr]),
                        "pad_separation_m": pad_gap(model, data),
                        "active_contact_count": int(data.ncon),
                        "robot_self_contact_count": len(contacts["robot_self_contacts"]),
                        "robot_bottle_contact_count": len(contacts["robot_bottle_contacts"]),
                        "non_gripper_joint_state": non_gripper_state,
                        "robotiq_joint_state": {
                            name(model, mujoco.mjtObj.mjOBJ_JOINT, jid): {
                                "qpos_rad": float(data.qpos[model.jnt_qposadr[jid]]),
                                "qvel_rad_s": float(data.qvel[model.jnt_dofadr[jid]]),
                                "qacc_rad_s2": float(data.qacc[model.jnt_dofadr[jid]]),
                                "qfrc_bias_nm": float(data.qfrc_bias[model.jnt_dofadr[jid]]),
                                "qfrc_actuator_nm": float(data.qfrc_actuator[model.jnt_dofadr[jid]]),
                                "qfrc_constraint_nm": float(data.qfrc_constraint[model.jnt_dofadr[jid]]),
                                "range_rad": model.jnt_range[jid].astype(float).tolist(),
                            }
                            for jid in range(model.njnt)
                            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").startswith("rq_")
                        },
                        "all_actuator_force": {name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid): float(data.actuator_force[aid])
                                                for aid in range(model.nu)},
                        "qpos_writes_after_rollout_start": 0,
                    }
                    all_rows.append(row)
                    trace_stream.write(json.dumps(row, sort_keys=True) + "\n")
                    phase_peak_force = max(phase_peak_force, abs(float(data.actuator_force[grip_id])))
                    if total_step - frame_step >= 40 or k == steps - 1:
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(renderer.render())
                        frame_step = total_step
                phase_end_gap = pad_gap(model, data)
                if phase == "OPEN":
                    gap_open = phase_end_gap
                elif phase == "CLOSE":
                    gap_closed = phase_end_gap
                elif phase == "REOPEN":
                    gap_reopened = phase_end_gap
                phase_rows.append({
                    "phase": phase, "duration_s": float(data.time - phase_start_time), "steps": steps,
                    "pad_separation_start_m": phase_start_gap, "pad_separation_end_m": phase_end_gap,
                    "driver_qpos_start_rad": phase_start_driver,
                    "driver_qpos_end_rad": float(data.qpos[driver_qadr]),
                    "peak_gripper_actuator_force": phase_peak_force,
                    "summed_active_contact_samples": phase_contacts,
                })
    finally:
        renderer.close()
        writer.close()

    grip_force_limit = float(np.max(np.abs(model.actuator_forcerange[grip_id]))) if model.actuator_forcelimited[grip_id] else math.inf
    source_head_limits, source_head_vel = m0.all_limited_joint_checks(model, data, source_velocity)
    open_close_motion = bool(gap_open is not None and gap_closed is not None and gap_reopened is not None and
                             gap_open - gap_closed >= 0.02 and abs(gap_reopened - gap_open) <= 0.01)
    return {
        "status": "PASS" if (open_close_motion and not limit_violations and not velocity_violations and
                              first_bottle_robot_contact is None and first_unexpected_self_contact is None and
                              actuator_effort_max.get(GRIP_ACTUATOR, 0.0) <= grip_force_limit + 1e-6) else "FAIL",
        "phase_results": phase_rows,
        "steps": total_step,
        "time_s": float(data.time),
        "driver_joint": DRIVER_JOINT,
        "follower_joint": FOLLOWER_JOINT,
        "open_command": open_command,
        "driver_initial": joint_start.get(DRIVER_JOINT),
        "driver_final": joint_state(model, data, DRIVER_JOINT),
        "follower_initial": joint_start.get(FOLLOWER_JOINT),
        "follower_final": joint_state(model, data, FOLLOWER_JOINT),
        "pad_separation_open_m": gap_open,
        "pad_separation_closed_m": gap_closed,
        "pad_separation_reopened_m": gap_reopened,
        "open_close_reopen_gate": open_close_motion,
        "gripper_actuator": GRIP_ACTUATOR,
        "gripper_actuator_forcerange": model.actuator_forcerange[grip_id].astype(float).tolist(),
        "gripper_actuator_force_limit_abs": grip_force_limit,
        "maximum_absolute_actuator_force_by_actuator": actuator_effort_max,
        "source_joint_limit_violations": limit_violations,
        "source_joint_velocity_limit_violations": velocity_violations,
        "post_run_limit_violations": source_head_limits,
        "post_run_velocity_violations": source_head_vel,
        "maximum_non_gripper_joint_deviation_rad": max_non_gripper_deviation,
        "maximum_non_gripper_joint_speed_rad_s": max_non_gripper_speed,
        "first_robot_bottle_contact": first_bottle_robot_contact,
        "first_robot_self_contact_after_exclusion": first_unexpected_self_contact,
        "bottle_robot_contact_samples": sum(bool(c["robot_bottle_contacts"]) for c in contact_rows),
        "any_robot_self_contact_samples": sum(bool(c["robot_self_contacts"]) for c in contact_rows),
        "qpos_writes_after_rollout_start": 0,
        "bottle_qpos_writes_after_rollout_start": 0,
        "trace_jsonl": str(trace_path),
        "contact_trace_jsonl": str(contact_path),
        "video": str(out / "mounted_open_close.mp4"),
        "initial_contacts": initial_contacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=m0.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=DEFAULT_STATION)
    parser.add_argument("--open-command", type=float, default=0.0,
                        help="Menagerie actuator command used for OPEN/HOLD; default preserves original endpoint test")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "static").mkdir(exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        "experiment": "Issue #46 X2 + Robotiq mounted gripper smoke; SIMULATION_ONLY",
        "status": "BLOCKED", "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
                                              "robot_sim_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                                              "robot_sim_worktree_clean": not bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
                                              "command": sys.argv},
        "known_modeling_exception": {"classification": "SIMULATION_ONLY modeling exception",
                                      "pair": ["head_pitch_link", "torso_link"],
                                      "production_or_pinned_source_modified": False},
    }
    try:
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash differs from the accepted immutable helper")
        ident = m0.identity(args, args.canonical_helper)
        result["identity"] = ident
        canonical = m0.load_module(args.canonical_helper)
        build_args = SimpleNamespace(x2_root=args.x2_root, menagerie_root=args.menagerie_root,
                                     station_base_pos=np.asarray(args.station_base_pos, dtype=float))
        base_model, adapter = m0.build_model(build_args, out, canonical)
        base_xml = Path(adapter["model_xml"])
        result["adapter"] = adapter
        result["base_mounted_model"] = {"path": str(base_xml), "sha256": sha256(base_xml)}
        station = np.asarray(args.station_base_pos, dtype=float)
        source_model = mujoco.MjModel.from_xml_path(str(args.x2_root / m0.X2_MJCF))
        comparison = model_comparison(source_model, base_model, ident, station)
        result["source_vs_mounted_head_torso"] = comparison
        result["source_vs_mounted_head_torso"]["same_source_pair_and_transforms"] = bool(
            comparison["head_torso_contact_match"]["source_count"] == 1 and
            comparison["head_torso_contact_match"]["mounted_count"] == 1 and
            all(v["world_position_max_abs_delta_m"] < 1e-9 and v["world_rotation_max_abs_delta"] < 1e-9
                for v in comparison["transform_deltas"].values()) and
            abs(comparison["head_torso_contact_match"]["source_contact_distance_m"] -
                comparison["head_torso_contact_match"]["mounted_contact_distance_m"]) < 1e-9 and
            comparison["head_torso_contact_match"]["only_robot_self_pair_in_both_models"] and
            not comparison["source_model"]["source_explicit_excludes"])
        write_json(out / "raw" / "source_vs_mounted_head_torso.json", comparison)
        if not result["source_vs_mounted_head_torso"]["same_source_pair_and_transforms"]:
            raise RuntimeError("Source/mounted neutral-head collision comparison did not validate exactly")

        before_data = mujoco.MjData(base_model)
        set_mounted_neutral(base_model, before_data)
        m0.render(base_model, before_data, out / "static" / "mounted_before_exclusion.png",
                  (0.0, 0.0, 0.95), 2.5, 135, -12)
        variant_path = out / "x2_robotiq_2f85_head_torso_exclude_simulation_only.xml"
        exception = create_exclusion_variant(base_xml, variant_path)
        variant_model = mujoco.MjModel.from_xml_path(str(variant_path))
        variant_data = mujoco.MjData(variant_model)
        set_mounted_neutral(variant_model, variant_data)
        robot_names = robot_body_names(variant_model)
        after_contacts = describe_contacts(variant_model, variant_data, robot_names)
        if after_contacts["robot_self_contacts"]:
            raise RuntimeError("Another active robot self-contact remains after the single-pair exclusion")
        if after_contacts["robot_bottle_contacts"]:
            raise RuntimeError("The bottle is not clear from the robot before the mounted smoke")
        m0.render(variant_model, variant_data, out / "static" / "mounted_after_exclusion.png",
                  (0.0, 0.0, 0.95), 2.5, 135, -12)
        result["isolated_contact_exception"] = {
            **exception,
            "base_model_sha256": sha256(base_xml),
            "experimental_model_path": str(variant_path),
            "experimental_model_sha256": sha256(variant_path),
            "all_initial_active_contacts_after_exclusion": after_contacts,
            "initial_self_contact_gate": "PASS" if not after_contacts["robot_self_contacts"] else "FAIL",
            "modeling_note": "Only the exact source-existing head_pitch_link/torso_link pair is excluded; all source geoms, limits, masks, and other collision pairs are retained.",
        }
        result["highest_gate"] = "SOURCE/MOUNTED CONTACT MATCH + ONE-PAIR EXPERIMENTAL EXCLUSION"

        if abs(float(variant_model.opt.timestep) - m0.DT) > 1e-12:
            raise RuntimeError("Experimental mounted model timestep changed unexpectedly")
        velocity_limits = {k: v["velocity"] for k, v in ident["urdf_joint_limits"].items()
                           if math.isfinite(v["velocity"]) and v["velocity"] > 0}
        ranges, velocities = m0.all_limited_joint_checks(variant_model, variant_data, velocity_limits)
        if ranges or velocities:
            raise RuntimeError(f"Initial source limit gate failed: positions={ranges[:1]}, velocities={velocities[:1]}")
        result["mounted_open_close"] = run_cycle(variant_model, variant_data, ident, out, args.open_command)
        if result["mounted_open_close"]["status"] != "PASS":
            raise RuntimeError("Mounted gripper dynamic OPEN/CLOSE cycle failed; inspect its recorded traces")
        result["highest_gate"] = "MOUNTED DYNAMIC OPEN/CLOSE"
        result["status"] = "PASS"
        result["evidence"] = {
            "result_json": str(out / "result.json"), "base_model": str(base_xml),
            "experimental_model": str(variant_path),
            "comparison_json": str(out / "raw" / "source_vs_mounted_head_torso.json"),
            "static_images": [str(out / "static" / n) for n in
                              ("mounted_before_exclusion.png", "mounted_after_exclusion.png")],
            "video": str(out / "mounted_open_close.mp4"),
        }
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "highest_gate": result.get("highest_gate"),
                      "result_json": str(out / "result.json"), "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
