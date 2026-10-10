#!/usr/bin/env python3
"""Build and validate the isolated X2 + Robotiq right-hand replacement variant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np

import issue46_x2_robotiq_m0 as experiment


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "mount-replacement-20261010"
)
OLD_APPEND_XML = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "overnight-transfer-20261010/full_episode_x022_attempt03/"
    "x2_robotiq_2f85_simulation_only.xml"
)


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"{kind.name.lower()}_{index}"


def object_inventory(model: mujoco.MjModel) -> dict[str, Any]:
    body_names = [name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    body_geom_ids: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
    body_joint_ids: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
    for geom_id in range(model.ngeom):
        body_geom_ids[int(model.geom_bodyid[geom_id])].append(geom_id)
    for joint_id in range(model.njnt):
        body_joint_ids[int(model.jnt_bodyid[joint_id])].append(joint_id)
    pelvis_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    bodies = []
    for body_id, body_name in enumerate(body_names):
        parent_id = int(model.body_parentid[body_id])
        bodies.append({
            "name": body_name,
            "parent": body_names[parent_id],
            "mass_kg": float(model.body_mass[body_id]),
            "subtree_mass_kg": float(model.body_subtreemass[body_id]),
            "inertial_position_m": model.body_ipos[body_id].tolist(),
            "inertia_principal_kg_m2": model.body_inertia[body_id].tolist(),
            "joints": [name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in body_joint_ids[body_id]],
            "geoms": [name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in body_geom_ids[body_id]],
        })
    joints = []
    for joint_id in range(model.njnt):
        joints.append({
            "name": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
            "body": body_names[int(model.jnt_bodyid[joint_id])],
            "limited": bool(model.jnt_limited[joint_id]),
            "range_rad": model.jnt_range[joint_id].tolist(),
            "axis": model.jnt_axis[joint_id].tolist(),
        })
    actuators = []
    for actuator_id in range(model.nu):
        transmission = int(model.actuator_trntype[actuator_id])
        target = None
        if transmission == int(mujoco.mjtTrn.mjTRN_JOINT):
            target_id = int(model.actuator_trnid[actuator_id, 0])
            target = name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id)
        elif transmission == int(mujoco.mjtTrn.mjTRN_TENDON):
            target_id = int(model.actuator_trnid[actuator_id, 0])
            target = name(model, mujoco.mjtObj.mjOBJ_TENDON, target_id)
        actuators.append({
            "name": name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
            "transmission_type": transmission,
            "target": target,
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange": model.actuator_forcerange[actuator_id].tolist(),
        })
    geoms = []
    for geom_id in range(model.ngeom):
        geoms.append({
            "name": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
            "body": body_names[int(model.geom_bodyid[geom_id])],
            "type": int(model.geom_type[geom_id]),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
        })
    return {
        "nbody": int(model.nbody), "njnt": int(model.njnt), "nu": int(model.nu),
        "ngeom": int(model.ngeom), "ntendon": int(model.ntendon), "neq": int(model.neq),
        "robot_subtree_mass_kg": float(model.body_subtreemass[pelvis_id]),
        "all_body_mass_sum_kg": float(np.sum(model.body_mass[1:])),
        "bodies": bodies, "joints": joints, "actuators": actuators, "geoms": geoms,
        "tendons": [name(model, mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(model.ntendon)],
        "equalities": [name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)],
    }


def has_x2_hand_name(value: str) -> bool:
    match = re.match(r"^(?:left|right|l|r)[_-]?(.*)$", value, re.I)
    if match is None:
        return False
    return bool(re.search(
        r"(?:^|[_-])(?:hand|palm|finger|thumb|index|middle|ring|pinky|claw)(?:[_-]|\d|$)",
        match.group(1), re.I,
    ))


def render(model: mujoco.MjModel, data: mujoco.MjData, path: Path, lookat,
           distance: float, azimuth: float, elevation: float) -> None:
    experiment.render(model, data, path, lookat, distance, azimuth, elevation)


def pad_gap(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    left = [experiment.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("rq_left_pad1", "rq_left_pad2")]
    right = [experiment.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
             for n in ("rq_right_pad1", "rq_right_pad2")]
    return float(np.linalg.norm(np.mean(data.geom_xpos[left], axis=0) - np.mean(data.geom_xpos[right], axis=0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=experiment.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=experiment.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=experiment.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-append-xml", type=Path, default=OLD_APPEND_XML)
    parser.add_argument("--station-base-pos", type=float, nargs=3, default=[0.3, 0.4, 0.68])
    parser.add_argument("--station-base-yaw-deg", type=float, default=-90.0)
    parser.add_argument("--coupler-limit-activation-margin-rad", type=float, default=0.001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "static").mkdir(exist_ok=True)
    (args.output_dir / "raw").mkdir(exist_ok=True)
    out = args.output_dir.resolve()
    result: dict[str, Any] = {
        "task": "Issue #46 X2 + Robotiq right-hand replacement morphology audit",
        "status": "BLOCKED", "physical_pickup": "NOT RUN",
        "transfer_place": "NOT RUN", "source_faithful_acceptance": "NOT CLAIMED",
    }
    try:
        if experiment.sha256(args.canonical_helper) != experiment.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash mismatch")
        canonical = experiment.load_module(args.canonical_helper)
        identity = experiment.identity(args, args.canonical_helper)
        result["identity"] = identity
        model_args = argparse.Namespace(
            x2_root=args.x2_root,
            menagerie_root=args.menagerie_root,
            station_base_pos=args.station_base_pos,
            station_base_yaw_deg=args.station_base_yaw_deg,
            coupler_limit_activation_margin_rad=args.coupler_limit_activation_margin_rad,
            table_center_xy=None, bottle_root_pos=None,
        )
        model, adapter = experiment.build_model(model_args, out, canonical)
        result["adapter"] = adapter
        replacement_xml = Path(adapter["model_xml"])

        source_model = mujoco.MjModel.from_xml_path(str(args.x2_root / experiment.X2_MJCF))
        old_inventory = None
        if args.baseline_append_xml.is_file():
            old_model = mujoco.MjModel.from_xml_path(str(args.baseline_append_xml))
            old_inventory = object_inventory(old_model)
        result["inventories"] = {
            "pinned_x2_source": object_inventory(source_model),
            "previous_append_assembly_historical": old_inventory,
            "corrected_replacement": object_inventory(model),
        }

        source_root = ET.parse(args.x2_root / experiment.X2_MJCF).getroot()
        replacement_root = ET.parse(replacement_xml).getroot()
        left_before = source_root.find('.//body[@name="left_wrist_roll_link"]')
        left_after = replacement_root.find('.//body[@name="left_wrist_roll_link"]')
        if left_before is None or left_after is None:
            raise RuntimeError("Left wrist body missing from source or replacement variant")
        left_xml_unchanged = ET.tostring(left_before) == ET.tostring(left_after)
        source_right_wrist = source_root.find('.//body[@name="right_wrist_roll_link"]')
        new_right_wrist = replacement_root.find('.//body[@name="right_wrist_roll_link"]')
        wrist_mesh_rows = {
            "source": [{"class": g.get("class"), "mesh": g.get("mesh")} for g in source_right_wrist.findall("geom")],
            "replacement": [{"class": g.get("class"), "mesh": g.get("mesh")} for g in new_right_wrist.findall("geom")],
        }
        source_inventory = result["inventories"]["pinned_x2_source"]
        hand_names = [body["name"] for body in source_inventory["bodies"] if has_x2_hand_name(body["name"])]
        hand_joints = [joint["name"] for joint in source_inventory["joints"] if has_x2_hand_name(joint["name"])]
        hand_actuators = [actuator["name"] for actuator in source_inventory["actuators"]
                          if has_x2_hand_name(actuator["name"])]
        source_wrist_id = experiment.obj_id(source_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        replacement_wrist_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        wrist_mass_inertia_unchanged = (
            np.isclose(source_model.body_mass[source_wrist_id], model.body_mass[replacement_wrist_id], atol=1e-12)
            and np.allclose(source_model.body_inertia[source_wrist_id], model.body_inertia[replacement_wrist_id], atol=1e-12)
            and np.allclose(source_model.body_ipos[source_wrist_id], model.body_ipos[replacement_wrist_id], atol=1e-12)
        )
        arm_limit_deltas = []
        for joint_name in experiment.ARM:
            before_id = experiment.obj_id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            after_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if (not np.array_equal(source_model.jnt_range[before_id], model.jnt_range[after_id])
                    or int(source_model.jnt_limited[before_id]) != int(model.jnt_limited[after_id])):
                arm_limit_deltas.append(joint_name)
        rq_root = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
        rq_parent = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[rq_root]))
        result["assembly_audit"] = {
            "original_x2_hand_bodies": hand_names,
            "original_x2_hand_joints": hand_joints,
            "original_x2_hand_actuators": hand_actuators,
            "original_x2_hand_bodies_present_in_pinned_mjcf": bool(hand_names),
            "original_x2_hand_joint_actuator_refs_present": bool(hand_joints or hand_actuators),
            "source_right_wrist_geom_pair": wrist_mesh_rows["source"],
            "replacement_right_wrist_geom_pair": wrist_mesh_rows["replacement"],
            "left_wrist_subtree_xml_unchanged": left_xml_unchanged,
            "right_wrist_mass_inertia_unchanged": bool(wrist_mass_inertia_unchanged),
            "right_arm_source_limit_deltas": arm_limit_deltas,
            "robotiq_root_parent": rq_parent,
            "robotiq_root_count": sum(
                name(model, mujoco.mjtObj.mjOBJ_BODY, i) == "rq_base_mount" for i in range(model.nbody)
            ),
            "right_wrist_body_direct_children": [
                name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
                if int(model.body_parentid[i]) == replacement_wrist_id
            ],
            "visual_collision_geometry_consistent": (
                wrist_mesh_rows["replacement"] == [
                    {"class": "visual", "mesh": "rq_x2_right_wrist_roll_extend_visual"},
                    {"class": "collision", "mesh": "right_wrist_roll_link"},
                ]
            ),
            "no_duplicate_x2_right_finger_bodies_in_compiled_model": not any(
                has_x2_hand_name(name(model, mujoco.mjtObj.mjOBJ_BODY, i))
                for i in range(model.nbody)
            ),
        }
        result["morphology_gate"] = {
            "right_hand_replacement": "PASS" if (
                not hand_names and not hand_joints and not hand_actuators
                and result["assembly_audit"]["visual_collision_geometry_consistent"]
                and rq_parent == "right_wrist_roll_link"
                and result["assembly_audit"]["robotiq_root_count"] == 1
                and result["assembly_audit"]["no_duplicate_x2_right_finger_bodies_in_compiled_model"]
                and wrist_mass_inertia_unchanged
            ) else "FAIL",
            "left_hand_preserved": "PASS" if left_xml_unchanged else "FAIL",
            "source_limits_preserved": "PASS" if not arm_limit_deltas else "FAIL",
            "compiled_without_invalid_references": "PASS",
            "sim_only_coupler_margin_rad": float(args.coupler_limit_activation_margin_rad),
            "source_faithful_coupler_status": "FAIL retained: original Robotiq spring torque exceeds source upper stop at reset",
        }

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        head_audit = experiment.audit_head_torso_clearance(model, data)
        result["head_torso_clearance_audit"] = {
            key: value for key, value in head_audit.items() if key != "samples"
        }
        selected_head = head_audit["selected_initial_pose"]
        data.qpos[experiment.qpos_id(model, "head_yaw_joint")] = selected_head["head_yaw_rad"]
        data.qpos[experiment.qpos_id(model, "head_pitch_joint")] = selected_head["head_pitch_rad"]
        data.qpos[experiment.qpos_id(model, "left_elbow_joint")] = -0.20
        data.qpos[experiment.qpos_id(model, "right_elbow_joint")] = -0.02
        gripper_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
        gripper_open = float(model.actuator_ctrlrange[gripper_id, 0])
        gripper_close = float(model.actuator_ctrlrange[gripper_id, 1])
        data.ctrl[gripper_id] = gripper_open
        mujoco.mj_forward(model, data)
        initial_contacts = experiment.contacts(model, data, forces=True)
        initial_gate = experiment.static_contact_gate(initial_contacts)
        result["initial_clearance"] = {
            "status": "PASS" if initial_gate["pass"] else "FAIL",
            "contacts": initial_contacts,
            "penetrating_unintended_contacts": initial_gate["penetrating_contacts"],
        }
        wrist_position = data.xpos[replacement_wrist_id].copy()
        static_dir = out / "static"
        render(model, data, static_dir / "overview_front.png", [0.12, 0.00, 0.96], 2.6, 90, -8)
        render(model, data, static_dir / "overview_side.png", [0.12, 0.00, 0.96], 2.6, 180, -8)
        gripper_root_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
        gripper_root_position = data.xpos[gripper_root_id].copy()
        render(model, data, static_dir / "right_wrist_closeup.png", gripper_root_position, 0.48, 225, 12)
        render(model, data, static_dir / "right_wrist_below_behind.png", wrist_position, 0.62, 225, 35)

        arm_qids, _, lower, upper = experiment.arm_metadata(model, identity)
        ik_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, ik_data)
        ik_data.qpos[experiment.qpos_id(model, "head_yaw_joint")] = selected_head["head_yaw_rad"]
        ik_data.qpos[experiment.qpos_id(model, "head_pitch_joint")] = selected_head["head_pitch_rad"]
        ik_data.qpos[experiment.qpos_id(model, "left_elbow_joint")] = -0.20
        ik_data.qpos[experiment.qpos_id(model, "right_elbow_joint")] = -0.02
        mujoco.mj_forward(model, ik_data)
        bottle_geom_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
        bottle_center = ik_data.geom_xpos[bottle_geom_id].copy()
        ik = experiment.solve_grasp_first_ik(
            model, ik_data, bottle_center,
            np.asarray(adapter["isolated_lift_grasp_offset_tool_local_m"], dtype=float),
            np.asarray(adapter["insertion_axis_local"], dtype=float),
            np.cross(np.asarray(adapter["insertion_axis_local"], dtype=float),
                     np.asarray(adapter["pad_opening_axis_local"], dtype=float)),
            0.05, 0.0, ik_data.qpos[arm_qids].copy(), arm_qids, lower, upper,
        )
        result["static_bottle_alignment"] = {
            "status": "PASS" if ik["gate_pass"] else "FAIL",
            "physics_run": False,
            "ik": {key: value for key, value in ik.items() if key != "evaluations"},
        }
        display = mujoco.MjData(model)
        mujoco.mj_resetData(model, display)
        display.qpos[experiment.qpos_id(model, "head_yaw_joint")] = selected_head["head_yaw_rad"]
        display.qpos[experiment.qpos_id(model, "head_pitch_joint")] = selected_head["head_pitch_rad"]
        display.qpos[experiment.qpos_id(model, "left_elbow_joint")] = -0.20
        display.qpos[experiment.qpos_id(model, "right_elbow_joint")] = -0.02
        display.qpos[arm_qids] = np.asarray(ik["q_rad"], dtype=float)
        display.ctrl[gripper_id] = gripper_open
        mujoco.mj_forward(model, display)
        open_rows = experiment.contacts(model, display)
        open_gate = experiment.static_contact_gate(open_rows)
        bottle_geom_ids = [i for i in range(model.ngeom)
                           if name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) == "m0_bottle"]
        pad_geom_ids = [experiment.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                        for n in ("rq_right_pad1", "rq_right_pad2", "rq_left_pad1", "rq_left_pad2")]
        pad_clearances = []
        for pad_id in pad_geom_ids:
            for bottle_id in bottle_geom_ids:
                distance = float(mujoco.mj_geomDistance(model, display, pad_id, bottle_id, 0.5, np.zeros(6)))
                pad_clearances.append({
                    "pad_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, pad_id),
                    "bottle_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_id),
                    "distance_m": distance,
                })
        result["static_bottle_alignment"]["open_configuration_clearance"] = {
            "status": "PASS" if open_gate["pass"] and min(row["distance_m"] for row in pad_clearances) > 0.0 else "FAIL",
            "contacts": open_rows, "pad_to_bottle_distances": pad_clearances,
        }
        result["static_bottle_alignment"]["status"] = "PASS" if (
            ik["gate_pass"] and result["static_bottle_alignment"]["open_configuration_clearance"]["status"] == "PASS"
        ) else "FAIL"
        render(model, display, static_dir / "open_aligned_canonical_bottle.png",
               [float(bottle_center[0]), float(bottle_center[1]), float(bottle_center[2])], 0.72, 135, -10)

        dry = mujoco.MjData(model)
        mujoco.mj_resetData(model, dry)
        dry.qpos[experiment.qpos_id(model, "head_yaw_joint")] = selected_head["head_yaw_rad"]
        dry.qpos[experiment.qpos_id(model, "head_pitch_joint")] = selected_head["head_pitch_rad"]
        dry.qpos[experiment.qpos_id(model, "left_elbow_joint")] = -0.20
        dry.qpos[experiment.qpos_id(model, "right_elbow_joint")] = -0.02
        bottle_joint_id = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
        bottle_qpos = int(model.jnt_qposadr[bottle_joint_id])
        # Keep the free bottle outside every collision envelope for the full dry-run horizon.
        dry.qpos[bottle_qpos:bottle_qpos + 3] = [100.0, 100.0, 1000.0]
        dry.qpos[bottle_qpos + 3:bottle_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        dry.ctrl[gripper_id] = gripper_open
        mujoco.mj_forward(model, dry)
        refs = experiment.controller_config(model)
        targets = {item["joint"]: float(dry.qpos[item["qpos_id"]]) for item in refs}
        target_velocity = {item["joint"]: 0.0 for item in refs}
        velocity_limits = {key: item["velocity"] for key, item in identity["urdf_joint_limits"].items()
                           if math.isfinite(item["velocity"]) and item["velocity"] > 0}
        phases = [
            ("open_hold", 300, gripper_open, gripper_open),
            ("close", 700, gripper_open, gripper_close),
            ("closed_hold", 300, gripper_close, gripper_close),
            ("reopen", 700, gripper_close, gripper_open),
            ("open_hold_final", 300, gripper_open, gripper_open),
        ]
        trace: list[dict[str, Any]] = []
        contact_trace: list[dict[str, Any]] = []
        peak_velocity: dict[str, float] = {}
        peak_force: dict[str, float] = {}
        peak_robot_joint_displacement: dict[str, float] = {}
        robot_joint_initial_qpos = {
            joint_name: float(dry.qpos[experiment.qpos_id(model, joint_name)])
            for joint_name in experiment.ARM
        }
        limit_violations: list[dict[str, Any]] = []
        closed_state_qpos = None
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [float(wrist_position[0]), float(wrist_position[1]), float(wrist_position[2])]
        camera.distance, camera.azimuth, camera.elevation = 1.2, 135, -10
        renderer = mujoco.Renderer(model, height=720, width=1280)
        video_path = out / "mounted_open_close.mp4"
        with imageio.get_writer(video_path, fps=30, codec="libx264", quality=8) as video:
            step = 0
            for phase_name, steps, ctrl_start, ctrl_end in phases:
                start_gap = pad_gap(model, dry)
                gaps = []
                for k in range(steps):
                    u = k / max(steps - 1, 1)
                    grip_s, _ = experiment.quintic(u)
                    mujoco.mj_forward(model, dry)
                    experiment.apply_controller(model, dry, refs, targets, target_velocity)
                    dry.ctrl[gripper_id] = ctrl_start + grip_s * (ctrl_end - ctrl_start)
                    mujoco.mj_step(model, dry)
                    ranges, velocities = experiment.all_limited_joint_checks(model, dry, velocity_limits)
                    limit_violations.extend(ranges)
                    limit_violations.extend(velocities)
                    if not (np.isfinite(dry.qpos).all() and np.isfinite(dry.qvel).all() and np.isfinite(dry.qacc).all()):
                        raise RuntimeError(f"Non-finite state in gripper-only {phase_name}")
                    rows = experiment.contacts(model, dry, forces=True)
                    if any("m0_bottle" in row["body1"] + row["body2"] for row in rows):
                        raise RuntimeError("Bottle contacted during the no-contact gripper smoke")
                    for row in rows:
                        contact_trace.append({"step": step + 1, "time_s": float(dry.time), "phase": phase_name, **row})
                    for joint_name, q_initial in robot_joint_initial_qpos.items():
                        q_now = float(dry.qpos[experiment.qpos_id(model, joint_name)])
                        peak_robot_joint_displacement[joint_name] = max(
                            peak_robot_joint_displacement.get(joint_name, 0.0), abs(q_now - q_initial)
                        )
                    gap = pad_gap(model, dry)
                    gaps.append(gap)
                    grip_force = abs(float(dry.actuator_force[gripper_id]))
                    peak_force["rq_fingers_actuator"] = max(peak_force.get("rq_fingers_actuator", 0.0), grip_force)
                    qpos_values = {}
                    for joint_name in (
                        "rq_right_driver_joint", "rq_left_driver_joint",
                        "rq_right_coupler_joint", "rq_left_coupler_joint",
                        "rq_right_spring_link_joint", "rq_left_spring_link_joint",
                        "rq_right_follower_joint", "rq_left_follower_joint",
                    ):
                        jid = experiment.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                        peak_velocity[joint_name] = max(
                            peak_velocity.get(joint_name, 0.0), abs(float(dry.qvel[dadr]))
                        )
                        qpos_values[joint_name] = float(dry.qpos[qadr])
                    trace.append({
                        "step": step + 1, "time_s": float(dry.time), "phase": phase_name,
                        "gripper_ctrl": float(dry.ctrl[gripper_id]),
                        "actuator_force": float(dry.actuator_force[gripper_id]),
                        "pad_gap_m": gap, "qpos": qpos_values,
                        "qvel_abs_max_rad_s": dict(peak_velocity),
                    })
                    if phase_name == "close" and k == steps - 1:
                        closed_state_qpos = dry.qpos.copy()
                    if step % 20 == 0 or k == steps - 1:
                        renderer.update_scene(dry, camera)
                        video.append_data(renderer.render())
                    step += 1
                result.setdefault("open_close_phase_summary", {})[phase_name] = {
                    "start_pad_gap_m": start_gap, "end_pad_gap_m": float(gaps[-1]),
                    "min_pad_gap_m": float(min(gaps)), "max_pad_gap_m": float(max(gaps)),
                }
        renderer.close()
        with (out / "raw" / "mounted_open_close_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with (out / "raw" / "mounted_open_close_contacts.jsonl").open("w", encoding="utf-8") as stream:
            for row in contact_trace:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with (out / "raw" / "mounted_open_close_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "step", "time_s", "phase", "gripper_ctrl", "actuator_force", "pad_gap_m",
            ])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in trace)

        contact_pairs: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in contact_trace:
            key = (row["body1"], row["geom1"], row["body2"], row["geom2"])
            contact_pairs.setdefault(key, []).append(row)
        contact_pair_summary = []
        for pair, rows in contact_pairs.items():
            bodies = {pair[0], pair[2]}
            is_pad_stop = bodies == {"rq_left_pad", "rq_right_pad"}
            is_wrist_interference = "right_wrist_roll_link" in bodies and any(
                body.startswith("rq_") for body in bodies
            )
            contact_pair_summary.append({
                "bodies_and_geoms": list(pair), "sample_count": len(rows),
                "first_phase": rows[0]["phase"], "first_step": rows[0]["step"],
                "min_distance_m": min(float(row["distance_m"]) for row in rows),
                "max_normal_force_n": max(float(row["normal_force_n"] or 0.0) for row in rows),
                "classification": (
                    "EXPECTED_PAD_TO_PAD_CLOSURE_STOP" if is_pad_stop else
                    "UNINTENDED_WRIST_GRIPPER_INTERFERENCE" if is_wrist_interference else
                    "OTHER_ROBOT_SELF_CONTACT"
                ),
            })
        mount_or_other_contacts = [row for row in contact_pair_summary
                                   if row["classification"] != "EXPECTED_PAD_TO_PAD_CLOSURE_STOP"]

        close_display = mujoco.MjData(model)
        mujoco.mj_resetData(model, close_display)
        close_display.qpos[experiment.qpos_id(model, "head_yaw_joint")] = selected_head["head_yaw_rad"]
        close_display.qpos[experiment.qpos_id(model, "head_pitch_joint")] = selected_head["head_pitch_rad"]
        close_display.qpos[experiment.qpos_id(model, "left_elbow_joint")] = -0.20
        close_display.qpos[experiment.qpos_id(model, "right_elbow_joint")] = -0.02
        close_display.qpos[arm_qids] = np.asarray(ik["q_rad"], dtype=float)
        for joint_id in range(model.njnt):
            joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name.startswith("rq_") and closed_state_qpos is not None:
                qadr = int(model.jnt_qposadr[joint_id])
                close_display.qpos[qadr] = closed_state_qpos[qadr]
        mujoco.mj_forward(model, close_display)
        render(model, close_display, static_dir / "close_diagnostic_aligned_bottle.png",
               [float(bottle_center[0]), float(bottle_center[1]), float(bottle_center[2])], 0.72, 135, -10)

        open_close_status = (
            not limit_violations
            and result["open_close_phase_summary"]["close"]["end_pad_gap_m"]
                < result["open_close_phase_summary"]["open_hold"]["start_pad_gap_m"]
            and result["open_close_phase_summary"]["open_hold_final"]["end_pad_gap_m"] > 0.02
            and not mount_or_other_contacts
        )
        result["open_close"] = {
            "status": "PASS" if open_close_status else "FAIL",
            "physics": "MuJoCo dynamics; fixed X2 base and neutral arm hold; bottle initialized beyond the dry-run gravity/contact horizon before the first step",
            "bottle_contact": False, "qpos_writes_after_rollout_start": 0,
            "limit_violations": limit_violations,
            "peak_joint_velocity_rad_s": peak_velocity,
            "peak_arm_joint_displacement_rad": peak_robot_joint_displacement,
            "peak_actuator_force": peak_force,
            "contact_count": len(contact_trace),
            "contact_pair_summary": contact_pair_summary,
            "unintended_contact_pairs": mount_or_other_contacts,
            "final_open_pad_gap_m": result["open_close_phase_summary"]["open_hold_final"]["end_pad_gap_m"],
            "closed_state_for_image_only": "captured from dry physical closure; recomposed for static visualization, not stepped",
        }
        result["visuals"] = {
            "overview_front": "static/overview_front.png",
            "overview_side": "static/overview_side.png",
            "right_wrist_closeup": "static/right_wrist_closeup.png",
            "right_wrist_below_behind": "static/right_wrist_below_behind.png",
            "open_aligned_canonical_bottle": "static/open_aligned_canonical_bottle.png",
            "closed_diagnostic_aligned_bottle": "static/close_diagnostic_aligned_bottle.png",
            "mounted_open_close_video": "mounted_open_close.mp4",
        }
        morphology_pass = (
            result["morphology_gate"]["right_hand_replacement"] == "PASS"
            and result["morphology_gate"]["left_hand_preserved"] == "PASS"
            and result["morphology_gate"]["source_limits_preserved"] == "PASS"
            and result["initial_clearance"]["status"] == "PASS"
            and result["static_bottle_alignment"]["open_configuration_clearance"]["status"] == "PASS"
            and open_close_status
        )
        result["status"] = "PASS_STATIC_MORPHOLOGY_READY_FOR_VISUAL_REVIEW" if morphology_pass else "FAIL"
        result["next_gate"] = "MAINTAINER_VISUAL_REVIEW_BEFORE_PICKUP_REVALIDATION"
        result["physical_pickup"] = "NOT RUN: visual morphology gate is a maintainer review checkpoint"
        result["transfer_place"] = "NOT RUN: superseded until corrected pickup is validated"
        result["report"] = str(out / "REPORT.md")
        experiment.json_write(out / "result.json", result)
        return 0 if morphology_pass else 1
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        experiment.json_write(out / "result.json", result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
