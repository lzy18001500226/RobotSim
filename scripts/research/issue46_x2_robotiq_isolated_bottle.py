#!/usr/bin/env python3
"""SIMULATION_ONLY isolated Robotiq 2F-85 and canonical-bottle physics test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-physical-recovery-20261009/isolated-bottle"
)
LIFT_JOINT = "fixture_lift_joint"
LIFT_ACTUATOR = "fixture_lift_position"
GRIP_ACTUATOR = "fingers_actuator"
PAD_GEOMS = {"left_pad1", "left_pad2", "right_pad1", "right_pad2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bottle_body_center(canonical) -> np.ndarray:
    root_pos = np.asarray(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS, dtype=float)
    geom = next(g for g in canonical.CANONICAL_X2_BOTTLE_GEOMS if g["name"] == "bottle_body")
    return root_pos + np.asarray(geom["pos"], dtype=float)


def build_fixture(source_xml: Path, menagerie_root: Path, canonical, out: Path,
                  pad_midpoint_offset_world_m: np.ndarray,
                  carriage_bias_compensation: bool,
                  carriage_stiffness_scale: float) -> tuple[mujoco.MjModel, dict[str, Any]]:
    root = ET.parse(source_xml).getroot()
    root.find("compiler").set("meshdir", str((menagerie_root / "robotiq_2f85" / "assets").resolve()))
    option = root.find("option")
    option.set("timestep", "0.001")
    option.set("gravity", "0 0 -9.81")
    world = root.find("worldbody")
    mount = world.find('./body[@name="base_mount"]')
    if mount is None:
        raise RuntimeError("Pinned Menagerie model has no base_mount")
    source_mount_pos = np.asarray([float(x) for x in mount.get("pos", "0 0 0").split()], dtype=float)
    mount.remove(mount.find("joint")) if mount.find("joint") is not None else None
    world.remove(mount)
    axes = m0.robotiq_axes(source_xml)
    rotation = np.asarray(axes["mount_rotation_matrix"], dtype=float)
    quat_wxyz = np.asarray(axes["mount_quat_wxyz"], dtype=float)
    mount.set("quat", " ".join(f"{x:.12g}" for x in quat_wxyz))
    carriage = ET.SubElement(world, "body", {"name": "fixture_carriage", "pos": "0 0 0"})
    ET.SubElement(carriage, "joint", {
        "name": LIFT_JOINT, "type": "slide", "axis": "0 0 1", "range": "0 0.05",
        "limited": "true", "armature": "0.02", "damping": "1.0",
    })
    carriage.append(mount)
    actuator = root.find("actuator")
    ctrlrange = "-0.02 0.10" if carriage_bias_compensation else "0 0.05"
    lift_kp = 1200.0 * carriage_stiffness_scale
    lift_kv = 70.0 * carriage_stiffness_scale ** 0.5
    ET.SubElement(actuator, "position", {
        "name": LIFT_ACTUATOR, "joint": LIFT_JOINT,
        "kp": f"{lift_kp:.12g}", "kv": f"{lift_kv:.12g}",
        "ctrlrange": ctrlrange, "ctrllimited": "true", "forcerange": "-25 25",
        "forcelimited": "true",
    })
    m0.add_scene(root, canonical)
    provisional = out / "model" / "isolated_robotiq_bottle_origin0.xml"
    provisional.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(provisional, encoding="utf-8", xml_declaration=True)
    model0 = mujoco.MjModel.from_xml_path(str(provisional))
    data0 = mujoco.MjData(model0)
    mujoco.mj_resetData(model0, data0)
    mujoco.mj_forward(model0, data0)
    pad_ids = [m0.obj_id(model0, mujoco.mjtObj.mjOBJ_GEOM, name)
               for name in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")]
    pad_midpoint0 = np.mean(data0.geom_xpos[pad_ids], axis=0)
    target = bottle_body_center(canonical)
    carriage_pos = target + pad_midpoint_offset_world_m - pad_midpoint0
    carriage.set("pos", " ".join(f"{x:.12g}" for x in carriage_pos))
    model_path = out / "model" / "isolated_robotiq_bottle.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pad_ids = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
               for name in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")]
    pad_midpoint = np.mean(data.geom_xpos[pad_ids], axis=0)
    return model, {
        "source_model": str(source_xml), "source_model_sha256": sha256(source_xml),
        "compiled_model": str(model_path), "compiled_model_sha256": sha256(model_path),
        "source_mount_pos_local_m": source_mount_pos.tolist(),
        "mount_rotation_world_from_source": rotation.tolist(),
        "mount_quaternion_wxyz": quat_wxyz.tolist(),
        "fixture_carriage_initial_position_world_m": carriage_pos.tolist(),
        "simulation_only_pad_midpoint_offset_world_m": pad_midpoint_offset_world_m.tolist(),
        "target_bottle_body_center_world_m": target.tolist(),
        "compiled_open_pad_midpoint_world_m": pad_midpoint.tolist(),
        "pad_midpoint_error_m": (pad_midpoint - target - pad_midpoint_offset_world_m).tolist(),
        "pad_separation_open_m": float(np.linalg.norm(
            np.mean(data.geom_xpos[pad_ids[:2]], axis=0) - np.mean(data.geom_xpos[pad_ids[2:]], axis=0))),
        "lift_joint_range_m": model.jnt_range[m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)].astype(float).tolist(),
        "lift_actuator_forcerange_n": model.actuator_forcerange[m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)].astype(float).tolist(),
        "lift_actuator_gains_simulation_derived": {"kp": lift_kp, "kv": lift_kv},
        "lift_servo_stiffness_scale_simulation_derived": carriage_stiffness_scale,
        "lift_actuator_effort_cap_simulation_derived_n": 25.0,
        "lift_actuator_control_range_simulation_derived_m": model.actuator_ctrlrange[
            m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)].astype(float).tolist(),
    }


def geom_label(model: mujoco.MjModel, gid: int) -> tuple[str, str]:
    body = int(model.geom_bodyid[gid])
    return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}")


def point_velocity_world(model: mujoco.MjModel, data: mujoco.MjData,
                         body_id: int, point_world: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point_world, body_id)
    return jacp @ data.qvel


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        force_local = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, force_local)
        b1, n1 = geom_label(model, g1)
        b2, n2 = geom_label(model, g2)
        if "m0_bottle" in (b1, b2):
            other = n2 if b1 == "m0_bottle" else n1
            side = "left" if other.startswith("left_pad") else ("right" if other.startswith("right_pad") else None)
            kind = "pad_bottle" if side else ("table_bottle" if "m0_table" in (b1, b2) else "other_bottle")
        else:
            side, kind = None, "other"
        # MuJoCo stores the contact-frame axes as rows. The contact force is the
        # force on geom2; reverse it when the bottle is geom1.
        frame_axes_world = np.asarray(c.frame, dtype=float).reshape(3, 3)
        force_world_on_geom2 = frame_axes_world.T @ force_local[:3]
        if b1 == "m0_bottle":
            force_world_on_bottle = -force_world_on_geom2
        elif b2 == "m0_bottle":
            force_world_on_bottle = force_world_on_geom2
        else:
            force_world_on_bottle = None
        body1_id = int(model.geom_bodyid[g1])
        body2_id = int(model.geom_bodyid[g2])
        velocity1 = point_velocity_world(model, data, body1_id, np.asarray(c.pos))
        velocity2 = point_velocity_world(model, data, body2_id, np.asarray(c.pos))
        relative_velocity = velocity2 - velocity1
        normal_world = frame_axes_world[0]
        tangential_velocity = relative_velocity - normal_world * float(np.dot(relative_velocity, normal_world))
        rows.append({
            "geom1": n1, "body1": b1, "geom2": n2, "body2": b2,
            "kind": kind, "pad_side": side, "distance_m": float(c.dist),
            "position_world_m": c.pos.astype(float).tolist(),
            "contact_frame_axes_world": frame_axes_world.tolist(),
            "normal_world_geom1_to_geom2": normal_world.tolist(),
            "force_local_n": force_local.astype(float).tolist() if c.efc_address >= 0 else None,
            "force_world_on_geom2_n": force_world_on_geom2.astype(float).tolist() if c.efc_address >= 0 else None,
            "force_world_on_bottle_n": force_world_on_bottle.astype(float).tolist() if force_world_on_bottle is not None and c.efc_address >= 0 else None,
            "normal_force_n": float(force_local[0]) if c.efc_address >= 0 else 0.0,
            "relative_velocity_geom2_minus_geom1_world_m_s": relative_velocity.astype(float).tolist(),
            "tangential_relative_velocity_world_m_s": tangential_velocity.astype(float).tolist(),
            "tangential_relative_speed_m_s": float(np.linalg.norm(tangential_velocity)),
            "friction_coefficients": c.friction.astype(float).tolist(),
            "solref": c.solref.astype(float).tolist(),
            "solimp": c.solimp.astype(float).tolist(),
            "condim": int(c.dim),
        })
    return rows


def static_clearances(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    bottle_geoms = [gid for gid in range(model.ngeom)
                    if geom_label(model, gid)[0] == "m0_bottle" and
                    (model.geom_contype[gid] or model.geom_conaffinity[gid])]
    table_geoms = [gid for gid in range(model.ngeom)
                   if geom_label(model, gid)[0].startswith("m0_table") and
                   (model.geom_contype[gid] or model.geom_conaffinity[gid])]
    robot_geoms = [gid for gid in range(model.ngeom)
                   if geom_label(model, gid)[0] not in {"world", "m0_bottle"} and
                   not geom_label(model, gid)[0].startswith("m0_table") and
                   (model.geom_contype[gid] or model.geom_conaffinity[gid])]
    segment = np.zeros(6, dtype=np.float64)
    distances = []
    for rg in robot_geoms:
        rb, rn = geom_label(model, rg)
        for eg in bottle_geoms + table_geoms:
            eb, en = geom_label(model, eg)
            d = float(mujoco.mj_geomDistance(model, data, rg, eg, 1.0, segment))
            distances.append({"robot_body": rb, "robot_geom": rn, "environment_body": eb,
                              "environment_geom": en, "distance_m": d,
                              "intended_grip_surface": rn in PAD_GEOMS and eb == "m0_bottle"})
    pad_bottle = [d for d in distances if d["environment_body"] == "m0_bottle" and d["intended_grip_surface"]]
    non_grip = [d for d in distances if d["environment_body"] == "m0_bottle" and not d["intended_grip_surface"]]
    table = [d for d in distances if d["environment_body"].startswith("m0_table")]
    return {
        "all_robot_environment_signed_distances": distances,
        "minimum_pad_to_bottle_m": min((d["distance_m"] for d in pad_bottle), default=None),
        "minimum_non_gripping_robot_to_bottle_m": min((d["distance_m"] for d in non_grip), default=None),
        "minimum_robot_to_table_m": min((d["distance_m"] for d in table), default=None),
        "active_contacts": contact_rows(model, data),
    }


def object_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    bid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return {
        "freejoint_position_world_m": data.qpos[qa:qa + 3].astype(float).tolist(),
        "freejoint_quaternion_wxyz": data.qpos[qa + 3:qa + 7].astype(float).tolist(),
        "center_of_mass_world_m": data.subtree_com[bid].astype(float).tolist(),
        "linear_velocity_world_m_s": data.qvel[da:da + 3].astype(float).tolist(),
        "angular_velocity_world_rad_s": data.qvel[da + 3:da + 6].astype(float).tolist(),
    }


def joint_dynamic_state(model: mujoco.MjModel, data: mujoco.MjData,
                        joint_names: tuple[str, ...]) -> dict[str, Any]:
    states = {}
    for name in joint_names:
        jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        limited = bool(model.jnt_limited[jid])
        limits = model.jnt_range[jid].astype(float).tolist() if limited else None
        q = float(data.qpos[qa])
        states[name] = {
            "qpos": q, "qvel": float(data.qvel[da]), "qacc": float(data.qacc[da]),
            "qfrc_bias": float(data.qfrc_bias[da]),
            "qfrc_passive": float(data.qfrc_passive[da]),
            "qfrc_actuator": float(data.qfrc_actuator[da]),
            "qfrc_constraint": float(data.qfrc_constraint[da]),
            "limited": limited, "range": limits,
            "limit_margin": min(q - limits[0], limits[1] - q) if limits else None,
        }
    return states


def contact_force_summary(contacts: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for kind in ("pad_bottle", "table_bottle", "other_bottle"):
        rows = [c for c in contacts if c["kind"] == kind and c["force_world_on_bottle_n"] is not None]
        force = np.sum([c["force_world_on_bottle_n"] for c in rows], axis=0) if rows else np.zeros(3)
        summaries[kind] = {
            "contact_count": len(rows), "force_world_on_bottle_sum_n": force.astype(float).tolist(),
            "vertical_force_on_bottle_sum_n": float(force[2]),
            "normal_force_sum_n": float(sum(c["normal_force_n"] for c in rows)),
            "max_tangential_relative_speed_m_s": max((c["tangential_relative_speed_m_s"] for c in rows), default=0.0),
        }
    return summaries


def body_subtree_mass(model: mujoco.MjModel, root_body_id: int) -> float:
    total = 0.0
    for body_id in range(1, model.nbody):
        parent_id = body_id
        while parent_id > 0:
            if parent_id == root_body_id:
                total += float(model.body_mass[body_id])
                break
            parent_id = int(model.body_parentid[parent_id])
    return total


def carriage_state(model: mujoco.MjModel, data: mujoco.MjData,
                   target_m: float, actuator_id: int) -> dict[str, Any]:
    jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)
    qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    body_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture_carriage")
    mass_matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass_matrix, data.qM)
    geom_ids = [m0.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
                ("left_pad1", "left_pad2", "right_pad1", "right_pad2")]
    return {
        "target_position_m": float(target_m),
        "actual_joint_position_m": float(data.qpos[qa]),
        "position_error_m": float(target_m - data.qpos[qa]),
        "joint_velocity_m_s": float(data.qvel[da]),
        "joint_acceleration_m_s2": float(data.qacc[da]),
        "qfrc_bias_n": float(data.qfrc_bias[da]),
        "qfrc_passive_n": float(data.qfrc_passive[da]),
        "qfrc_actuator_n": float(data.qfrc_actuator[da]),
        "qfrc_constraint_n": float(data.qfrc_constraint[da]),
        "effective_joint_inertia_kg": float(mass_matrix[da, da]),
        "carriage_subtree_mass_kg": body_subtree_mass(model, body_id),
        "actuator_control": float(data.ctrl[actuator_id]),
        "actuator_force_n": float(data.actuator_force[actuator_id]),
        "world_position_m": data.xpos[body_id].astype(float).tolist(),
        "pad_geom_positions_world_m": {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid): data.geom_xpos[gid].astype(float).tolist()
            for gid in geom_ids
        },
    }


def phase_steps(model: mujoco.MjModel, data: mujoco.MjData, name: str,
                duration: float, grip_start: float, grip_end: float,
                lift_start: float, lift_end: float,
                writer, renderer: mujoco.Renderer, camera: mujoco.MjvCamera,
                trace, contacts, summary, baseline_com_z: float,
                lift_bias_compensation_n: float = 0.0, lift_kp: float = 1200.0) -> dict[str, Any]:
    steps = int(round(duration / float(model.opt.timestep)))
    grip_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    lift_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
    phase_start_time = float(data.time)
    bilateral_samples = 0
    table_contact_samples = 0
    non_gripping_bottle_contact_samples = 0
    bottle_rows = []
    max_lift = -float("inf")
    for k in range(steps):
        u = (k + 1) / steps
        s = m0.quintic(u)[0]
        lift_target_m = lift_start + s * (lift_end - lift_start)
        data.ctrl[grip_id] = grip_start + s * (grip_end - grip_start)
        data.ctrl[lift_id] = lift_target_m + lift_bias_compensation_n / lift_kp
        mujoco.mj_step(model, data)
        crows = contact_rows(model, data)
        left = [c for c in crows if c["kind"] == "pad_bottle" and c["pad_side"] == "left"]
        right = [c for c in crows if c["kind"] == "pad_bottle" and c["pad_side"] == "right"]
        table = [c for c in crows if c["kind"] == "table_bottle"]
        non_gripping_bottle = [c for c in crows if c["kind"] == "other_bottle"]
        both = bool(left and right)
        bilateral_samples += int(both)
        table_contact_samples += int(bool(table))
        non_gripping_bottle_contact_samples += int(bool(non_gripping_bottle))
        state = object_state(model, data)
        if trace:
            previous = trace[-1]
            sample_dt = float(data.time - previous["time_s"])
            linear_acceleration = ((np.asarray(state["linear_velocity_world_m_s"]) -
                                    np.asarray(previous["object"]["linear_velocity_world_m_s"])) / sample_dt)
            angular_acceleration = ((np.asarray(state["angular_velocity_world_rad_s"]) -
                                     np.asarray(previous["object"]["angular_velocity_world_rad_s"])) / sample_dt)
            acceleration_available = True
        else:
            linear_acceleration = np.zeros(3)
            angular_acceleration = np.zeros(3)
            acceleration_available = False
        state["acceleration_estimate_available"] = acceleration_available
        state["linear_acceleration_world_m_s2"] = linear_acceleration.astype(float).tolist()
        state["angular_acceleration_world_rad_s2"] = angular_acceleration.astype(float).tolist()
        lift = float(state["center_of_mass_world_m"][2] - baseline_com_z)
        max_lift = max(max_lift, lift)
        gripper_joints = joint_dynamic_state(model, data, (
            "right_driver_joint", "right_coupler_joint", "right_follower_joint",
            "left_driver_joint", "left_coupler_joint", "left_follower_joint"))
        contact_forces = contact_force_summary(crows)
        carriage = carriage_state(model, data, lift_target_m, lift_id)
        row = {
            "step": int(round(float(data.time) / float(model.opt.timestep))),
            "time_s": float(data.time), "phase": name,
            "gripper_control": float(data.ctrl[grip_id]),
            "gripper_actuator_force_n": float(data.actuator_force[grip_id]),
            "lift_target_m": float(lift_target_m),
            "lift_actuator_control_position_m": float(data.ctrl[lift_id]),
            "lift_actuator_force_n": float(data.actuator_force[lift_id]),
            "object": state, "object_lift_from_settled_com_m": lift,
            "left_pad_normal_force_sum_n": sum(c["normal_force_n"] for c in left),
            "right_pad_normal_force_sum_n": sum(c["normal_force_n"] for c in right),
            "table_normal_force_sum_n": sum(c["normal_force_n"] for c in table),
            "contact_force_summary": contact_forces,
            "gripper_joints": gripper_joints,
            "fixture_carriage": carriage,
            "bilateral_pad_contact": both,
            "table_bottle_contact": bool(table),
            "non_gripping_bottle_contact": bool(non_gripping_bottle),
            "contacts": crows,
            "source_joint_limit_violations": m0.all_limited_joint_checks(model, data, {})[0],
            "qpos_writes_after_rollout_start": 0,
        }
        trace.append(row)
        contacts.append({"step": row["step"], "time_s": row["time_s"], "phase": name,
                         "contacts": crows, "bilateral_pad_contact": both,
                         "left_pad_normal_force_sum_n": row["left_pad_normal_force_sum_n"],
                         "right_pad_normal_force_sum_n": row["right_pad_normal_force_sum_n"],
                         "table_normal_force_sum_n": row["table_normal_force_sum_n"],
                         "table_bottle_contact": row["table_bottle_contact"],
                         "non_gripping_bottle_contact": row["non_gripping_bottle_contact"]})
        bottle_rows.append(row)
        if k % 25 == 0 or k == steps - 1:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    return {
        "phase": name, "duration_s": float(data.time - phase_start_time), "steps": steps,
        "gripper_control_start": grip_start, "gripper_control_end": grip_end,
        "lift_target_start_m": lift_start, "lift_target_end_m": lift_end,
        "bilateral_contact_persistence_fraction": bilateral_samples / max(steps, 1),
        "bilateral_contact_samples": bilateral_samples,
        "table_contact_persistence_fraction": table_contact_samples / max(steps, 1),
        "table_contact_samples": table_contact_samples,
        "non_gripping_bottle_contact_persistence_fraction": non_gripping_bottle_contact_samples / max(steps, 1),
        "non_gripping_bottle_contact_samples": non_gripping_bottle_contact_samples,
        "max_object_lift_from_settled_com_m": max_lift,
        "final_object_state": object_state(model, data),
        "final_table_normal_force_n": bottle_rows[-1]["table_normal_force_sum_n"] if bottle_rows else None,
        "final_source_limit_violations": bottle_rows[-1]["source_joint_limit_violations"] if bottle_rows else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pad-midpoint-offset-world-m", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="simulation-only fixture placement offset from the canonical bottle-body center")
    parser.add_argument("--carriage-bias-compensation", action="store_true",
                        help="diagnostic fixture-only position-servo feedforward derived from carriage qfrc_bias")
    parser.add_argument("--carriage-servo-stiffness-scale", type=float, default=1.0,
                        help="simulation-derived fixture carriage servo stiffness scale; damping scales by sqrt(scale)")
    parser.add_argument("--airborne-clearance-target-m", type=float, default=0.001,
                        help="fixture carriage target for the >=1 mm bottle-airborne gate; bounded to 1-2 mm")
    parser.add_argument("--skip-physics", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / "raw").mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment": "SIMULATION_ONLY Robotiq 2F-85 isolated canonical bottle grasp",
        "acceptance_classification": "NON-ACCEPTANCE while the strict source-limit gate is failed",
        "status": "BLOCKED",
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()),
                   "robot_sim_head": __import__("subprocess").check_output(
                       ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
                   "command": sys.argv},
    }
    try:
        if args.carriage_servo_stiffness_scale <= 0:
            raise ValueError("carriage servo stiffness scale must be positive")
        if not 0.001 <= args.airborne_clearance_target_m <= 0.002:
            raise ValueError("airborne clearance target must be between 0.001 and 0.002 m")
        native_library = Path(mujoco.__file__).resolve().parent / "libmujoco.so.3.3.6"
        result["runtime_identity"] = {
            "python": sys.version,
            "mujoco_python_version": mujoco.__version__,
            "mujoco_native_version": mujoco.mj_versionString(),
            "mujoco_python_module": str(Path(mujoco.__file__).resolve()),
            "mujoco_native_library": str(native_library) if native_library.exists() else None,
            "mujoco_native_library_sha256": sha256(native_library) if native_library.exists() else None,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        }
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical helper hash differs from the accepted immutable helper")
        canonical = m0.load_module(args.canonical_helper)
        source = args.menagerie_root / m0.MENAGERIE_MJCF
        model, identity = build_fixture(source, args.menagerie_root, canonical, out,
                                        np.asarray(args.pad_midpoint_offset_world_m, dtype=float),
                                        args.carriage_bias_compensation,
                                        args.carriage_servo_stiffness_scale)
        result["fixture_identity"] = identity
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        clearances = static_clearances(model, data)
        source_pos, _ = m0.all_limited_joint_checks(model, data, {})
        intended_initial_contacts = [c for c in clearances["active_contacts"]
                                     if c["kind"] == "table_bottle"]
        unintended_initial_contacts = [c for c in clearances["active_contacts"]
                                       if c["kind"] not in {"table_bottle"}]
        result["static_preflight"] = {
            **clearances,
            "source_joint_position_limit_violations": source_pos,
            "intended_bottle_table_support_contacts": intended_initial_contacts,
            "unintended_initial_contacts": unintended_initial_contacts,
            "pass": bool(not source_pos and not unintended_initial_contacts and
                         clearances["minimum_pad_to_bottle_m"] is not None and
                         clearances["minimum_pad_to_bottle_m"] > 0.0 and
                         clearances["minimum_non_gripping_robot_to_bottle_m"] is not None and
                         clearances["minimum_non_gripping_robot_to_bottle_m"] >= 0.0 and
                         clearances["minimum_robot_to_table_m"] is not None and
                         clearances["minimum_robot_to_table_m"] >= 0.0),
        }
        m0.render(model, data, out / "static_overview.png", (0.30, -0.10, 0.89), 1.15, 135, -16)
        m0.render(model, data, out / "static_side.png", (0.30, -0.10, 0.88), 0.75, 90, -2)
        m0.render(model, data, out / "static_closeup.png", (0.30, -0.10, 0.88), 0.45, 135, -8)
        result["static_images"] = [str(out / n) for n in ("static_overview.png", "static_side.png", "static_closeup.png")]
        write_json(out / "result.json", result)
        if args.skip_physics or not result["static_preflight"]["pass"]:
            result["status"] = "STATIC_FAIL" if not result["static_preflight"]["pass"] else "STATIC_PASS_PHYSICS_SKIPPED"
            write_json(out / "result.json", result)
            return 0 if result["static_preflight"]["pass"] else 1

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        lift_jid = m0.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)
        lift_dof = int(model.jnt_dofadr[lift_jid])
        lift_actuator_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
        lift_kp = float(model.actuator_gainprm[lift_actuator_id, 0])
        lift_bias_compensation_n = float(data.qfrc_bias[lift_dof]) if args.carriage_bias_compensation else 0.0
        result["carriage_controller"] = {
            "classification": "SIMULATION_ONLY fixture carriage controller",
            "mode": "bias_compensated_position_servo" if args.carriage_bias_compensation else "uncompensated_position_servo_baseline",
            "target_feedforward_qfrc_bias_n": lift_bias_compensation_n,
            "position_gain_n_per_m": lift_kp,
            "velocity_gain_n_s_per_m": -float(model.actuator_biasprm[lift_actuator_id, 2]),
            "control_position_offset_m": lift_bias_compensation_n / lift_kp,
            "stiffness_scale": args.carriage_servo_stiffness_scale,
            "damping_scale_for_constant_damping_ratio": args.carriage_servo_stiffness_scale ** 0.5,
            "airborne_clearance_command_target_m": args.airborne_clearance_target_m,
            "actuator_effort_cap_n": float(model.actuator_forcerange[lift_actuator_id, 1]),
            "source_joint_limits_changed": False,
            "bottle_force_or_state_applied": False,
        }
        renderer = mujoco.Renderer(model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.30, -0.10, 0.89]
        camera.distance, camera.azimuth, camera.elevation = 1.15, 135, -16
        writer = imageio.get_writer(str(out / "isolated_grasp_lift_release.mp4"), fps=30, codec="libx264", quality=7)
        trace_rows, contact_rows_all, phases = [], [], []
        try:
            grip_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
            lift_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
            settled_start = object_state(model, data)
            settle_phase = phase_steps(model, data, "SETTLE", 0.30, 0, 0, 0, 0,
                                       writer, renderer, camera, trace_rows, contact_rows_all, phases,
                                       float(settled_start["center_of_mass_world_m"][2]),
                                       lift_bias_compensation_n, lift_kp)
            baseline = object_state(model, data)
            baseline_com_z = float(baseline["center_of_mass_world_m"][2])
            phases.append(settle_phase)
            phases.append(phase_steps(model, data, "OPEN", 0.20, 0, 0, 0, 0,
                                      writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z,
                                      lift_bias_compensation_n, lift_kp))
            phases.append(phase_steps(model, data, "CLOSE", 0.60, 0, 255, 0, 0,
                                      writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z,
                                      lift_bias_compensation_n, lift_kp))
            hold = phase_steps(model, data, "HOLD", 1.0, 255, 255, 0, 0,
                               writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z,
                               lift_bias_compensation_n, lift_kp)
            phases.append(hold)
            hold_rows = [r for r in trace_rows if r["phase"] == "HOLD"]
            hold_left_right = [r["bilateral_pad_contact"] for r in hold_rows]
            hold_start = hold_rows[0]["object"] if hold_rows else object_state(model, data)
            hold_end = hold_rows[-1]["object"] if hold_rows else object_state(model, data)
            hold_slip = float(np.linalg.norm(np.asarray(hold_end["center_of_mass_world_m"][:2]) -
                                             np.asarray(hold_start["center_of_mass_world_m"][:2])))
            bilateral_hold_pass = bool(hold_rows and sum(hold_left_right) / len(hold_left_right) >= 0.80 and
                                       any(r["left_pad_normal_force_sum_n"] > 0.05 and r["right_pad_normal_force_sum_n"] > 0.05
                                           for r in hold_rows))
            result["physical_hold"] = {**hold, "bilateral_persistence_fraction": sum(hold_left_right) / max(len(hold_left_right), 1),
                                        "bottle_com_slip_during_hold_xy_m": hold_slip,
                                        "bottle_com_height_change_during_hold_m": float(hold_end["center_of_mass_world_m"][2] - hold_start["center_of_mass_world_m"][2]),
                                        "bilateral_hold_pass": bilateral_hold_pass}
            lift_milestones = []
            first_failed_lift_gate = None
            if bilateral_hold_pass:
                previous = 0.0
                milestone_specs = ((0.001, args.airborne_clearance_target_m),
                                   (0.005, 0.005), (0.030, 0.030), (0.050, 0.050))
                for required_height, target_height in milestone_specs:
                    label_mm = int(round(required_height * 1000))
                    move = phase_steps(model, data, f"LIFT_{label_mm}MM", 0.30, 255, 255, previous, target_height,
                                       writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z,
                                       lift_bias_compensation_n, lift_kp)
                    phases.append(move)
                    hold_duration = 1.0 if required_height >= 0.050 else 0.20
                    hold_lift = phase_steps(model, data, f"HOLD_{label_mm}MM", hold_duration, 255, 255, target_height, target_height,
                                            writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z,
                                            lift_bias_compensation_n, lift_kp)
                    phases.append(hold_lift)
                    end_contacts = contact_rows(model, data)
                    table_contact_at_end = any(c["kind"] == "table_bottle" for c in end_contacts)
                    bilateral_contact_at_end = (any(c["kind"] == "pad_bottle" and c["pad_side"] == "left" for c in end_contacts) and
                                                any(c["kind"] == "pad_bottle" and c["pad_side"] == "right" for c in end_contacts))
                    non_gripping_contact_at_end = any(c["kind"] == "other_bottle" for c in end_contacts)
                    real_com_lift = float(object_state(model, data)["center_of_mass_world_m"][2] - baseline_com_z)
                    milestone_gate_pass = bool(
                        real_com_lift >= required_height and
                        hold_lift["table_contact_persistence_fraction"] == 0.0 and
                        hold_lift["non_gripping_bottle_contact_persistence_fraction"] == 0.0 and
                        hold_lift["bilateral_contact_persistence_fraction"] >= 0.80 and
                        bilateral_contact_at_end and not table_contact_at_end and not non_gripping_contact_at_end)
                    milestone = {
                        "required_bottle_lift_m": required_height,
                        "carriage_target_m": target_height,
                        "move": move, "hold": hold_lift,
                        "real_com_lift_m": real_com_lift,
                        "peak_com_lift_during_move_m": move["max_object_lift_from_settled_com_m"],
                        "table_contact_at_end": table_contact_at_end,
                        "table_contact_persistence_fraction_during_hold": hold_lift["table_contact_persistence_fraction"],
                        "non_gripping_bottle_contact_at_end": non_gripping_contact_at_end,
                        "non_gripping_bottle_contact_persistence_fraction_during_hold": hold_lift["non_gripping_bottle_contact_persistence_fraction"],
                        "bilateral_contact_at_end": bool(bilateral_contact_at_end),
                        "gate_pass": milestone_gate_pass,
                    }
                    lift_milestones.append(milestone)
                    previous = target_height
                    if not milestone_gate_pass:
                        if hold_lift["table_contact_persistence_fraction"] > 0.0:
                            first_failed_lift_gate = "BOTTLE_NOT_AIRBORNE"
                        elif hold_lift["non_gripping_bottle_contact_persistence_fraction"] > 0.0:
                            first_failed_lift_gate = "NON_GRIPPING_BOTTLE_CONTACT"
                        elif not bilateral_contact_at_end or hold_lift["bilateral_contact_persistence_fraction"] < 0.80:
                            first_failed_lift_gate = "BILATERAL_CONTACT_LOST"
                        else:
                            first_failed_lift_gate = f"LIFT_{label_mm}MM"
                        break
            result["lift_milestones"] = lift_milestones
            result["first_failed_lift_gate"] = first_failed_lift_gate
            reached_50mm_hold = bool(lift_milestones and lift_milestones[-1]["required_bottle_lift_m"] == 0.050 and
                                     lift_milestones[-1]["gate_pass"])
            release_contact_count = None
            result["release_attempted"] = reached_50mm_hold
            if reached_50mm_hold:
                held_lift_position = float(data.qpos[m0.qpos_id(model, LIFT_JOINT)])
                grip_phase = phase_steps(model, data, "RELEASE", 0.60, 255, 0, held_lift_position,
                                         held_lift_position, writer, renderer, camera,
                                         trace_rows, contact_rows_all, phases, baseline_com_z,
                                         lift_bias_compensation_n, lift_kp)
                phases.append(grip_phase)
                release_contact_count = len([c for c in contact_rows(model, data) if c["kind"] == "pad_bottle"])
                settle_final = phase_steps(model, data, "FREE_SETTLE", 1.0, 0, 0,
                                           held_lift_position, held_lift_position, writer, renderer, camera,
                                           trace_rows, contact_rows_all, phases, baseline_com_z,
                                           lift_bias_compensation_n, lift_kp)
                phases.append(settle_final)
        finally:
            writer.close()
            renderer.close()
        with (out / "raw" / "isolated_physics_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with (out / "raw" / "isolated_contact_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in contact_rows_all:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        result["phase_results"] = phases
        result["release_pad_contact_count_at_end"] = release_contact_count
        all_limit_failures = [r for r in trace_rows if r["source_joint_limit_violations"]]
        result["source_joint_limit_status"] = "FAIL" if all_limit_failures else "PASS"
        result["source_joint_limit_first_failure"] = all_limit_failures[0] if all_limit_failures else None
        result["qpos_writes_after_rollout_start"] = 0
        result["bottle_qpos_writes_after_rollout_start"] = 0
        result["runtime_bottle_attachment_or_mocap"] = False
        result["right_and_left_pad_real_contact"] = bool(result["physical_hold"]["bilateral_hold_pass"])
        reached = [m["required_bottle_lift_m"] for m in lift_milestones if m["gate_pass"]]
        result["maximum_measured_bottle_lift_m"] = max(
            (max(m["real_com_lift_m"], m["peak_com_lift_during_move_m"]) for m in lift_milestones),
            default=0.0,
        )
        result["lift_milestones_reached_and_airborne_m"] = reached
        result["diagnostic_physics_result"] = "PASS" if (bilateral_hold_pass and 0.050 in reached and
                                                            result["source_joint_limit_status"] == "PASS") else "FAIL_NON_ACCEPTANCE"
        result["status"] = "NON_ACCEPTANCE_PHYSICS_COMPLETE"
        result["evidence"] = {"result_json": str(out / "result.json"),
                              "physics_trace": str(out / "raw" / "isolated_physics_trace.jsonl"),
                              "contact_trace": str(out / "raw" / "isolated_contact_trace.jsonl"),
                              "video": str(out / "isolated_grasp_lift_release.mp4"),
                              "static_images": result["static_images"]}
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "static_preflight": result.get("static_preflight", {}).get("pass"),
                      "source_joint_limit_status": result.get("source_joint_limit_status"),
                      "bilateral_hold": result.get("right_and_left_pad_real_contact"),
                      "maximum_bottle_lift_m": result.get("maximum_measured_bottle_lift_m"),
                      "result_json": str(out / "result.json"), "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] in {"NON_ACCEPTANCE_PHYSICS_COMPLETE", "STATIC_PASS_PHYSICS_SKIPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
