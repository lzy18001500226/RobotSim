#!/usr/bin/env python3
"""Diagnose the pinned Robotiq coupler startup and exercise dual grippers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


COUPLERS = (
    "rq_left_right_coupler_joint",
    "rq_left_left_coupler_joint",
    "rq_right_right_coupler_joint",
    "rq_right_left_coupler_joint",
)
ISOLATED_COUPLERS = (
    "right_coupler_joint",
    "left_coupler_joint",
)
PHASES = (
    ("open_hold", 100, 0.0),
    ("close", 700, 255.0),
    ("closed_hold", 100, 255.0),
    ("reopen", 700, 0.0),
    ("open_hold_final", 100, 0.0),
)
EXPECTED_BASELINE_XML_SHA256 = "1a8cbf80cf9f186009aba7df17b13f4a89c07ba7b52b567f5a593af57fc7a8c7"
EXPECTED_HELPER_SHA256 = "5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063"


def load_controller_helper(path: Path):
    spec = importlib.util.spec_from_file_location("issue46_x2_robotiq_m0", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load controller helper snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def name(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, ident) or f"{kind.name.lower()}_{ident}"


def object_id(model, kind, label: str) -> int:
    ident = int(mujoco.mj_name2id(model, kind, label))
    if ident < 0:
        raise RuntimeError(f"Missing {kind.name} {label!r}")
    return ident


def joint_state(model, data, joint_names: tuple[str, ...]) -> dict[str, Any]:
    states = {}
    for joint_name in joint_names:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        states[joint_name] = {
            "qpos0_rad": float(model.qpos0[qadr]),
            "range_rad": model.jnt_range[jid].tolist(),
            "margin_rad": float(model.jnt_margin[jid]),
            "qpos_rad": float(data.qpos[qadr]),
            "qvel_rad_s": float(data.qvel[dadr]),
            "qacc_rad_s2": float(data.qacc[dadr]),
            "qfrc_bias_nm": float(data.qfrc_bias[dadr]),
            "qfrc_passive_nm": float(data.qfrc_passive[dadr]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dadr]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[dadr]),
        }
    return states


def contacts_with_force(model, data) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        force = np.zeros(6)
        if int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, index, force)
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        rows.append({
            "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])),
            "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])),
            "distance_m": float(contact.dist),
            "normal_force_n": float(force[0]),
            "contact_position_m": contact.pos.tolist(),
        })
    return rows


def source_limit_rows(model, data, source_urdf_limits: dict[str, Any], dual):
    return dual.source_limit_audit(model, data, source_urdf_limits)


def make_snapshot(model, data, joint_names, ctrl: dict[str, float] | None = None):
    actuators = {}
    for actuator_name, value in (ctrl or {}).items():
        aid = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        actuators[actuator_name] = {
            "ctrl": float(data.ctrl[aid]),
            "requested_ctrl": value,
            "actuator_force": float(data.actuator_force[aid]),
        }
    return {
        "time_s": float(data.time),
        "states": joint_state(model, data, joint_names),
        "actuators": actuators,
        "contact_count": int(data.ncon),
        "contacts": contacts_with_force(model, data),
        "qpos_finite": bool(np.isfinite(data.qpos).all()),
        "qvel_finite": bool(np.isfinite(data.qvel).all()),
        "qacc_finite": bool(np.isfinite(data.qacc).all()),
    }


def run_isolated_case(model_path: Path, case: str, target: float | None) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    actuator = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    if target is not None:
        data.ctrl[actuator] = target
    mujoco.mj_forward(model, data)
    before = make_snapshot(model, data, ISOLATED_COUPLERS + (
        "right_spring_link_joint", "left_spring_link_joint",
        "right_driver_joint", "left_driver_joint"), {"fingers_actuator": data.ctrl[actuator]})
    mujoco.mj_step(model, data)
    after = make_snapshot(model, data, ISOLATED_COUPLERS + (
        "right_spring_link_joint", "left_spring_link_joint",
        "right_driver_joint", "left_driver_joint"), {"fingers_actuator": data.ctrl[actuator]})
    return {
        "case": case,
        "commanded_ctrl": target,
        "model_timestep_s": float(model.opt.timestep),
        "spring_joint_stiffness_nm_per_rad": {
            joint: float(model.jnt_stiffness[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
            for joint in ("right_spring_link_joint", "left_spring_link_joint")
        },
        "spring_joint_reference_rad": {
            joint: float(model.qpos_spring[int(model.jnt_qposadr[object_id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint)])])
            for joint in ("right_spring_link_joint", "left_spring_link_joint")
        },
        "before_first_step": before,
        "after_first_step": after,
    }


def prepare_dual_data(model, dual, prior):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for joint, value in (("head_yaw_joint", 0.0), ("head_pitch_joint", 0.0),
                         ("left_elbow_joint", -0.20), ("right_elbow_joint", -0.02)):
        data.qpos[prior.qpos_id(model, joint)] = value
    bottle = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free"))
    if bottle >= 0:
        bq = int(model.jnt_qposadr[bottle])
        data.qpos[bq:bq + 3] = [100.0, 100.0, 100.0]
        data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    gripper_controls = {}
    for side in ("left", "right"):
        actuator_name = f"rq_{side}_fingers_actuator"
        aid = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        data.ctrl[aid] = 0.0
        gripper_controls[actuator_name] = 0.0
    mujoco.mj_forward(model, data)
    return data, gripper_controls


def one_step_dual(model_path: Path, case: str, controller_enabled: bool, dual, prior,
                  source_limits: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, controls = prepare_dual_data(model, dual, prior)
    names = COUPLERS + (
        "rq_left_left_driver_joint", "rq_left_right_driver_joint",
        "rq_right_left_driver_joint", "rq_right_right_driver_joint",
        "rq_left_left_spring_link_joint", "rq_left_right_spring_link_joint",
        "rq_right_left_spring_link_joint", "rq_right_right_spring_link_joint",
    )
    before_reset = joint_state(model, data, COUPLERS)
    if controller_enabled:
        refs = prior.controller_config(model)
        targets = {item["joint"]: float(data.qpos[item["qpos_id"]]) for item in refs}
        prior.apply_controller(model, data, refs, targets, {joint: 0.0 for joint in targets})
        for actuator_name, value in controls.items():
            data.ctrl[object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)] = value
        mujoco.mj_forward(model, data)
    before = make_snapshot(model, data, names, controls)
    pre_limit_rows = source_limit_rows(model, data, source_limits, dual)
    mujoco.mj_step(model, data)
    after = make_snapshot(model, data, names, controls)
    post_limit_rows = source_limit_rows(model, data, source_limits, dual)
    return {
        "case": case,
        "controller_enabled": controller_enabled,
        "model_timestep_s": float(model.opt.timestep),
        "coupler_ranges_rad": {joint: model.jnt_range[object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint)].tolist() for joint in COUPLERS},
        "coupler_margins_rad": {joint: float(model.jnt_margin[object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint)]) for joint in COUPLERS},
        "after_reset_qpos": before_reset,
        "immediately_before_first_step": before,
        "immediately_after_first_step": after,
        "source_limit_rows_before_step": pre_limit_rows,
        "source_limit_rows_after_step": post_limit_rows,
        "controller_command_count": len(prior.controller_config(model)) if controller_enabled else 0,
    }


def coupler_excess(model, data) -> dict[str, float]:
    result = {}
    for joint in COUPLERS:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        qadr = int(model.jnt_qposadr[jid])
        upper = float(model.jnt_range[jid, 1])
        result[joint] = max(0.0, float(data.qpos[qadr]) - upper)
    return result


def fatal_limit_rows(rows: list[dict[str, Any]], margin: float, simulation_only: bool):
    failures = []
    for row in rows:
        violation = max(-float(row["lower_margin_rad"]), -float(row["upper_margin_rad"]), 0.0)
        if violation <= 1e-7:
            continue
        if simulation_only and row["joint"] in COUPLERS and violation <= margin + 1e-7:
            continue
        failures.append({**row, "violation_rad": violation})
    return failures


def render_video_frame(renderer, model, data, camera, label: str) -> np.ndarray:
    renderer.update_scene(data, camera)
    frame = Image.fromarray(renderer.render()).convert("RGB")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 34), fill=(18, 25, 33))
    draw.text((14, 10), label, fill=(255, 255, 255))
    return np.asarray(frame)


def run_cycle(model_path: Path, side_mode: str, margin: float, out: Path, dual, prior,
              source_limits: dict[str, Any], render: bool = True) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, _ = prepare_dual_data(model, dual, prior)
    refs = prior.controller_config(model)
    targets = {item["joint"]: float(data.qpos[item["qpos_id"]]) for item in refs}
    target_vel = {joint: 0.0 for joint in targets}
    actuator_ids = {
        side: object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"rq_{side}_fingers_actuator")
        for side in ("left", "right")
    }
    diagnostics = []
    max_excess = {joint: 0.0 for joint in COUPLERS}
    max_abs_velocity = 0.0
    max_abs_force = {side: 0.0 for side in ("left", "right")}
    max_source_violations = {joint: 0 for joint in COUPLERS}
    opening_start = {side: dual.mounted_pad_gap(model, data, side) for side in ("left", "right")}
    trace_path = out / f"{side_mode}_open_close_trace.jsonl"
    video_path = out / f"{side_mode}_open_close.mp4"
    renderer = mujoco.Renderer(model, height=720, width=1280) if render else None
    pelvis = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lookat = data.xpos[pelvis].copy() + np.array([0.0, 0.45, 0.36])
    camera = dual.make_camera(lookat, 2.25, 135, -8)
    failed = None
    step = 0
    phases: list[tuple[str, int, float, float]] = []
    for phase, count, final_target in PHASES:
        phases.append((phase, count,
                       final_target if side_mode in ("left", "both") else 0.0,
                       final_target if side_mode in ("right", "both") else 0.0))

    writer_ctx = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8) if render else None
    try:
        with trace_path.open("w", encoding="utf-8") as trace_stream:
            if render and writer_ctx is not None:
                writer_ctx.append_data(render_video_frame(renderer, model, data, camera,
                                                          f"{side_mode.upper()} | OPEN HOLD"))
            for phase, steps, left_end, right_end in phases:
                for local in range(steps):
                    blend, _ = prior.quintic((local + 1) / steps)
                    if phase in ("close",):
                        left_target = left_end * blend
                        right_target = right_end * blend
                    elif phase == "reopen":
                        left_target = left_end + (255.0 - left_end) * (1.0 - blend) if side_mode in ("left", "both") else 0.0
                        right_target = right_end + (255.0 - right_end) * (1.0 - blend) if side_mode in ("right", "both") else 0.0
                        # Reopen runs from fully closed to open, independent of the held endpoint.
                        left_target = 255.0 * (1.0 - blend) if side_mode in ("left", "both") else 0.0
                        right_target = 255.0 * (1.0 - blend) if side_mode in ("right", "both") else 0.0
                    else:
                        left_target, right_target = left_end, right_end
                    prior.apply_controller(model, data, refs, targets, target_vel)
                    data.ctrl[actuator_ids["left"]] = left_target
                    data.ctrl[actuator_ids["right"]] = right_target
                    mujoco.mj_step(model, data)
                    step += 1
                    rows = source_limit_rows(model, data, source_limits, dual)
                    all_violations = [row for row in rows if min(
                        float(row["lower_margin_rad"]), float(row["upper_margin_rad"])) < -1e-7]
                    fatal = fatal_limit_rows(rows, margin, simulation_only=True)
                    excess = coupler_excess(model, data)
                    for joint, value in excess.items():
                        max_excess[joint] = max(max_excess[joint], value)
                    for row in all_violations:
                        if row["joint"] in max_source_violations:
                            max_source_violations[row["joint"]] += 1
                    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                                  and np.isfinite(data.qacc).all())
                    contacts = contacts_with_force(model, data)
                    wrist_contacts = [row for row in contacts if row["distance_m"] < -1e-5 and
                                      (("wrist_roll_link" in row["body1"] and row["body2"].startswith("rq_"))
                                       or ("wrist_roll_link" in row["body2"] and row["body1"].startswith("rq_")))]
                    gripper_qpos = {}
                    gripper_qvel = {}
                    gripper_qacc = {}
                    generalized_forces = {}
                    for gripper_side in ("left", "right"):
                        for suffix in dual.REFERENCE_JOINTS:
                            joint = f"rq_{gripper_side}_{suffix}"
                            jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                            qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                            gripper_qpos[joint] = float(data.qpos[qadr])
                            gripper_qvel[joint] = float(data.qvel[dadr])
                            gripper_qacc[joint] = float(data.qacc[dadr])
                            generalized_forces[joint] = {
                                "qfrc_actuator_nm": float(data.qfrc_actuator[dadr]),
                                "qfrc_passive_nm": float(data.qfrc_passive[dadr]),
                                "qfrc_constraint_nm": float(data.qfrc_constraint[dadr]),
                                "qfrc_bias_nm": float(data.qfrc_bias[dadr]),
                            }
                    actuator_force = {
                        name: float(data.actuator_force[actuator_ids[name]]) for name in ("left", "right")
                    }
                    for name, value in actuator_force.items():
                        max_abs_force[name] = max(max_abs_force[name], abs(value))
                    max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(data.qvel))))
                    row = {
                        "step": step, "time_s": float(data.time), "phase": phase,
                        "left_control": left_target, "right_control": right_target,
                        "left_pad_gap_m": dual.mounted_pad_gap(model, data, "left"),
                        "right_pad_gap_m": dual.mounted_pad_gap(model, data, "right"),
                        "pad_gap_change_from_open_m": {
                            side: float(dual.mounted_pad_gap(model, data, side) - opening_start[side])
                            for side in ("left", "right")},
                        "actuator_force": actuator_force,
                        "qpos_rad": gripper_qpos, "qvel_rad_s": gripper_qvel,
                        "qacc_rad_s2": gripper_qacc, "generalized_force_nm": generalized_forces,
                        "source_limit_violations": all_violations,
                        "coupler_upper_excess_rad": excess,
                        "contacts": contacts, "unintended_wrist_gripper_contacts": wrist_contacts,
                        "finite_state": finite,
                    }
                    trace_stream.write(json.dumps(row, sort_keys=True) + "\n")
                    if render and writer_ctx is not None and (step % 10 == 0 or step == 1):
                        writer_ctx.append_data(render_video_frame(renderer, model, data, camera,
                                                                  f"{side_mode.upper()} | {phase} | t={data.time:.3f}s"))
                    if not finite:
                        failed = {"step": step, "time_s": float(data.time), "reason": "non-finite state"}
                    elif fatal:
                        failed = {"step": step, "time_s": float(data.time),
                                  "reason": "source limit exceeded beyond diagnostic margin",
                                  "limit_failures": fatal}
                    elif wrist_contacts:
                        failed = {"step": step, "time_s": float(data.time),
                                  "reason": "unexpected wrist/gripper collision", "contacts": wrist_contacts}
                    if failed:
                        break
                if failed:
                    break
    finally:
        if writer_ctx is not None:
            writer_ctx.close()
        if renderer is not None:
            renderer.close()

    final_gaps = {side: dual.mounted_pad_gap(model, data, side) for side in ("left", "right")}
    return {
        "mode": side_mode,
        "status": "PASS" if failed is None and step == sum(p[1] for p in PHASES) else "FAIL",
        "steps": step, "duration_s": float(data.time), "first_failure": failed,
        "source_faithful_status": "FAIL: pinned source upper-limit transient remains independently recorded",
        "simulation_only_activation_margin_rad": margin,
        "source_joint_ranges_unchanged": True,
        "source_limit_violations_by_coupler": max_source_violations,
        "max_coupler_upper_excess_rad": max_excess,
        "max_abs_qvel_rad_s_all_dofs": max_abs_velocity,
        "max_abs_actuator_force_by_side": max_abs_force,
        "open_pad_gap_m": opening_start, "final_pad_gap_m": final_gaps,
        "qpos_writes_after_rollout_start": 0,
        "bottle_contact": False,
        "trace_path": str(trace_path),
        "video_path": str(video_path) if render else None,
        "video_is_actuator_driven_physics": bool(render),
    }


def build_dual_source_model(out: Path, prior, dual, menagerie_xml: Path) -> Path:
    x2_path = prior.X2_DEFAULT / prior.X2_MJCF
    tool_path = prior.X2_DEFAULT / prior.X2_TOOL_URDF
    root = ET.parse(x2_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((x2_path.parent / "meshes").resolve()))
    compiler.set("autolimits", "true")
    compiler.set("angle", "radian")
    option = root.find("option")
    option.set("timestep", str(prior.DT))
    option.set("gravity", f"0 0 {-prior.GRAVITY}")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "720")
    pelvis = root.find('./worldbody/body[@name="pelvis"]')
    floating = next((item for item in pelvis.findall("freejoint")
                     if item.get("name") == "floating_base_joint"), None)
    if floating is None:
        raise RuntimeError("Pinned X2 MJCF lacks its expected floating base joint")
    pelvis.remove(floating)
    pelvis.set("pos", "0 0.08 0.68")
    men_template = ET.parse(menagerie_xml).getroot()
    mounts = dual.official_mounts(tool_path)
    axes = dual.menagerie_axes(menagerie_xml)
    for side in ("left", "right"):
        dual.set_wrist_mesh_pair(root, x2_path.parent / "meshes", side)
        dual.namespaced_menagerie(root, men_template, menagerie_xml.parent,
                                  side, mounts[side], axes)
    world = root.find("worldbody")
    ET.SubElement(world, "geom", {
        "name": "recovery_floor", "type": "plane", "size": "0 0 0.05",
        "friction": "1 0.01 0.001", "condim": "4",
    })
    ET.indent(root, space="  ")
    model_path = out / "dual_source_faithful_no_object_scene.xml"
    ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
    return model_path


def run_static_workcell_clearance(out: Path, prior, canonical_path: Path,
                                  source_root: Path, menagerie_root: Path) -> dict[str, Any]:
    canonical_path = canonical_path.resolve()
    canonical = prior.load_module(canonical_path)
    scene_args = argparse.Namespace(
        x2_root=source_root / "agibot_x2_urdf",
        menagerie_root=menagerie_root,
        station_base_pos=[0.30, 0.40, 0.68],
        station_base_yaw_deg=-90.0,
        table_center_xy=[0.30, -0.10],
        bottle_root_pos=[0.30, 0.045, 0.9175],
        target_bottle_root_pos=[0.22, 0.045, 0.9175],
        grasp_height_offset_m=0.05,
        pregrasp_retreat_m=0.02,
        coupler_limit_activation_margin_rad=0.001,
    )
    identity = prior.identity(scene_args, canonical_path)
    model, adapter = prior.build_model(scene_args, out, canonical)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    head_pose = prior.audit_head_torso_clearance(model, data)["selected_initial_pose"]
    data.qpos[prior.qpos_id(model, "head_yaw_joint")] = head_pose["head_yaw_rad"]
    data.qpos[prior.qpos_id(model, "head_pitch_joint")] = head_pose["head_pitch_rad"]
    data.qpos[prior.qpos_id(model, "left_elbow_joint")] = -0.20
    grip_id = prior.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rq_fingers_actuator")
    data.ctrl[grip_id] = 0.0
    mujoco.mj_forward(model, data)
    qids, _, lower, upper = prior.arm_metadata(model, identity)
    bottle_geom = prior.obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    bottle_center = data.geom_xpos[bottle_geom].copy()
    insertion = np.asarray(adapter["insertion_axis_local"], dtype=float)
    vertical = np.cross(insertion, np.asarray(adapter["pad_opening_axis_local"], dtype=float))
    vertical /= np.linalg.norm(vertical)
    offset = np.asarray(adapter["isolated_lift_grasp_offset_tool_local_m"], dtype=float)
    seed = data.qpos[qids].copy()
    grasp = prior.solve_grasp_first_ik(
        model, data, bottle_center, offset, insertion, vertical, 0.05, 0.0,
        seed, qids, lower, upper)
    pregrasp = prior.solve_grasp_first_ik(
        model, data, bottle_center, offset, insertion, vertical, 0.05, 0.02,
        np.asarray(grasp["q_rad"], dtype=float), qids, lower, upper)

    def collidable_geoms(predicate):
        result = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = prior.obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue
            if predicate(body_name, prior.obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)):
                result.append(geom_id)
        return result

    bottle_geoms = collidable_geoms(lambda body, geom: body == "m0_bottle")
    table_geoms = collidable_geoms(lambda body, geom: body.startswith("m0_table"))
    wrist_geoms = collidable_geoms(lambda body, geom: body == "right_wrist_roll_link")
    gripper_geoms = collidable_geoms(lambda body, geom: body.startswith("rq_"))
    nonpad_gripper_geoms = collidable_geoms(
        lambda body, geom: body.startswith("rq_") and "pad" not in geom.lower())
    right_arm_geoms = collidable_geoms(
        lambda body, geom: body.startswith("right_") or body == "torso_link")

    def nearest_pair(first: list[int], second: list[int]) -> dict[str, Any] | None:
        nearest = None
        for first_id in first:
            for second_id in second:
                fromto = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(
                    model, data, first_id, second_id, 0.5, fromto))
                if nearest is None or distance < nearest["distance_m"]:
                    nearest = {
                        "geom1": prior.obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, first_id),
                        "geom2": prior.obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, second_id),
                        "body1": prior.obj_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(model.geom_bodyid[first_id])),
                        "body2": prior.obj_name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(model.geom_bodyid[second_id])),
                        "distance_m": distance,
                        "fromto_m": fromto.tolist(),
                    }
        return nearest

    bottle_joint = prior.obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    bottle_qadr = int(model.jnt_qposadr[bottle_joint])
    initial_bottle_qpos = data.qpos[bottle_qadr:bottle_qadr + 7].copy()
    path_rows = []
    if grasp["gate_pass"] and pregrasp["gate_pass"]:
        for fraction in np.linspace(0.0, 1.0, 101):
            data.qpos[qids] = np.asarray(pregrasp["q_rad"]) + fraction * (
                np.asarray(grasp["q_rad"]) - np.asarray(pregrasp["q_rad"]))
            data.ctrl[grip_id] = 0.0
            mujoco.mj_forward(model, data)
            contacts = prior.contacts(model, data)
            gate = prior.static_contact_gate(contacts)
            path_rows.append({
                "fraction": float(fraction),
                "right_arm_qpos_rad": data.qpos[qids].astype(float).tolist(),
                "contacts": contacts,
                "unwanted_penetrations": gate["penetrating_contacts"],
                "minimum_right_wrist_to_bottle": nearest_pair(wrist_geoms, bottle_geoms),
                "minimum_nonpad_gripper_to_bottle": nearest_pair(nonpad_gripper_geoms, bottle_geoms),
                "minimum_full_gripper_to_bottle": nearest_pair(gripper_geoms, bottle_geoms),
                "minimum_right_wrist_to_table": nearest_pair(wrist_geoms, table_geoms),
                "minimum_full_gripper_to_table": nearest_pair(gripper_geoms, table_geoms),
                "minimum_right_arm_to_table": nearest_pair(right_arm_geoms, table_geoms),
                "pass": gate["pass"],
            })

    raw_path = out / "raw" / "workcell_open_approach_clearance.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(path_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    minimum = {}
    for key in (
        "minimum_right_wrist_to_bottle", "minimum_nonpad_gripper_to_bottle",
        "minimum_full_gripper_to_bottle", "minimum_right_wrist_to_table",
        "minimum_full_gripper_to_table", "minimum_right_arm_to_table",
    ):
        values = [row[key] for row in path_rows if row[key] is not None]
        minimum[key] = min(values, key=lambda item: item["distance_m"]) if values else None
    static_dir = out / "static_workcell"
    static_dir.mkdir(exist_ok=True)
    if pregrasp["gate_pass"]:
        data.qpos[qids] = np.asarray(pregrasp["q_rad"])
        mujoco.mj_forward(model, data)
        prior.render(model, data, static_dir / "open_pregrasp.png", (0.22, -0.05, 0.9), 1.8, 135, -8)
    if grasp["gate_pass"]:
        data.qpos[qids] = np.asarray(grasp["q_rad"])
        mujoco.mj_forward(model, data)
        prior.render(model, data, static_dir / "open_grasp_endpoint.png", (0.28, -0.02, 0.89), 0.85, 135, -10)
    path_pass = bool(path_rows) and all(row["pass"] for row in path_rows)
    return {
        "classification": "STATIC ONLY; no mj_step executed",
        "status": "PASS" if path_pass else "FAIL" if path_rows else "NOT RUN: grasp/pregrasp IK gate failed",
        "canonical_helper": str(canonical_path),
        "canonical_helper_sha256": sha256(canonical_path),
        "layout": {
            "station_base_position_world_m": scene_args.station_base_pos,
            "station_base_yaw_deg": scene_args.station_base_yaw_deg,
            "table_center_xy_world_m": scene_args.table_center_xy,
            "bottle_root_world_m": scene_args.bottle_root_pos,
            "table_top_z_m": 0.8,
            "bottle_nominal_diameter_m": 0.07,
            "bottle_nominal_height_m": 0.2445,
            "bottle_mass_kg": 0.57,
        },
        "source_identity": identity,
        "model_configuration": adapter["model_configuration"],
        "coupler_limit_activation_margin_simulation_derived_rad": 0.001,
        "grasp_ik": {key: grasp[key] for key in (
            "gate_pass", "q_rad", "target_position_m", "position_error_m",
            "orientation_error_rad", "vertical_error_rad", "yaw_error_rad")},
        "pregrasp_ik": {key: pregrasp[key] for key in (
            "gate_pass", "q_rad", "target_position_m", "position_error_m",
            "orientation_error_rad", "vertical_error_rad", "yaw_error_rad")},
        "approach_path_sample_count": len(path_rows),
        "bottle_qpos_writes_during_static_path": 0,
        "bottle_qpos_held_constant": bool(np.array_equal(
            data.qpos[bottle_qadr:bottle_qadr + 7], initial_bottle_qpos)),
        "approach_path_pass": path_pass,
        "first_blocking_penetration": next((
            {"fraction": row["fraction"], "contact": row["unwanted_penetrations"][0]}
            for row in path_rows if row["unwanted_penetrations"]), None),
        "minimum_compiled_geometry_distances": minimum,
        "raw_path": str(raw_path),
        "static_images": [str(p) for p in sorted(static_dir.glob("*.png"))],
        "active_rollout_bottle_qpos_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-xml", type=Path,
                        help="Optional preserved dual assembly XML; omitted for a clean source rebuild")
    parser.add_argument("--controller-helper-snapshot", required=True, type=Path)
    parser.add_argument("--menagerie-xml", required=True, type=Path)
    parser.add_argument("--coupler-margin-rad", type=float, default=0.001)
    parser.add_argument("--exclude-known-head-torso-source-contact", action="store_true",
                        help="Add the previously authorized exact-pair exclusion to the diagnostic model only")
    parser.add_argument("--canonical-helper", type=Path,
                        help="Optional immutable canonical scene helper to run the static-only workcell corridor check")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    helper_path = args.controller_helper_snapshot.resolve()
    prior = load_controller_helper(helper_path)
    sys.modules["issue46_x2_robotiq_m0"] = prior
    import issue46_x2_dual_robotiq_platform as dual

    if not 0.0 < args.coupler_margin_rad <= 0.001:
        raise ValueError("The approved diagnostic margin must be in (0, 0.001] rad")
    if sha256(helper_path) != EXPECTED_HELPER_SHA256:
        raise RuntimeError("Controller helper snapshot does not match the recorded helper SHA")
    menagerie_xml = args.menagerie_xml.resolve()
    if git_value(menagerie_xml.parents[1], "rev-parse", "HEAD") != prior.MENAGERIE_PIN:
        raise RuntimeError("Menagerie checkout SHA does not match pinned source")
    baseline_source = (args.baseline_xml.resolve() if args.baseline_xml else
                       build_dual_source_model(out, prior, dual, menagerie_xml))
    baseline_hash = sha256(baseline_source)
    baseline_classification = (
        "previous preserved dual assembly XML"
        if baseline_hash == EXPECTED_BASELINE_XML_SHA256 else
        "source-rebuilt dual X2 + two Menagerie grippers; no bottle/table scene loaded"
    )
    snapshot_baseline = out / "dual_source_faithful_input.xml"
    shutil.copy2(baseline_source, snapshot_baseline)
    baseline_hash_copy = sha256(snapshot_baseline)
    baseline_model = mujoco.MjModel.from_xml_path(str(snapshot_baseline))
    baseline_data, _ = prepare_dual_data(baseline_model, dual, prior)
    baseline_contacts = dual.contact_rows(baseline_model, baseline_data)
    head_torso_contacts = [row for row in baseline_contacts
                           if {row["body1"], row["body2"]} == {"head_pitch_link", "torso_link"}]
    if args.exclude_known_head_torso_source_contact and not any(
            row["distance_m"] < -1e-5 for row in head_torso_contacts):
        raise RuntimeError("Refusing the diagnostic exclusion: source head/torso overlap was not reproduced")
    root = ET.parse(snapshot_baseline).getroot()
    joint_nodes = {node.get("name"): node for node in root.iter("joint") if node.get("name")}
    baseline_model_check = mujoco.MjModel.from_xml_path(str(snapshot_baseline))
    for joint in COUPLERS:
        if joint not in joint_nodes:
            raise RuntimeError(f"Baseline assembly is missing {joint}")
        jid = object_id(baseline_model_check, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not np.array_equal(baseline_model_check.jnt_range[jid], np.asarray([-1.57, 0.0])):
            raise RuntimeError(f"Source joint range for {joint} is not the original upper stop 0 rad")
    diagnostic_xml = out / "dual_simulation_only_margin_0p001.xml"
    for joint in COUPLERS:
        joint_nodes[joint].set("margin", f"{args.coupler_margin_rad:.12g}")
    if args.exclude_known_head_torso_source_contact:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        existing = [item for item in contact.findall("exclude")
                    if {item.get("body1"), item.get("body2")} ==
                    {"head_pitch_link", "torso_link"}]
        if existing:
            raise RuntimeError("Baseline already contains the exact head/torso exclusion")
        ET.SubElement(contact, "exclude", {
            "body1": "head_pitch_link", "body2": "torso_link"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(diagnostic_xml, encoding="utf-8", xml_declaration=True)

    source_limits = dual.source_joint_limits(prior.X2_DEFAULT / prior.X2_URDF)
    isolated = {
        "source_no_controller_default_open": run_isolated_case(menagerie_xml, "default_open", None),
        "source_explicit_open_command": run_isolated_case(menagerie_xml, "explicit_open", 0.0),
        "source_close_command_startup": run_isolated_case(menagerie_xml, "close_startup", 255.0),
    }
    dual_startup = {}
    for mode, model_path in (("source_faithful", snapshot_baseline),
                             ("simulation_only_margin_0p001", diagnostic_xml)):
        for controller_enabled in (False, True):
            key = f"{mode}_{'controller' if controller_enabled else 'no_controller'}"
            dual_startup[key] = one_step_dual(
                model_path, key, controller_enabled, dual, prior, source_limits)

    diagnostic_first_step = dual_startup["simulation_only_margin_0p001_controller"]
    first_step_excess = {
        row["joint"]: max(-float(row["lower_margin_rad"]),
                          -float(row["upper_margin_rad"]), 0.0)
        for row in diagnostic_first_step["source_limit_rows_after_step"]
        if row["joint"] in COUPLERS
        and (row["lower_margin_rad"] < -1e-7 or row["upper_margin_rad"] < -1e-7)
    }
    first_step_finite = all(diagnostic_first_step["immediately_after_first_step"][key]
                             for key in ("qpos_finite", "qvel_finite", "qacc_finite"))
    diagnostic_prevalidation_pass = (
        first_step_finite and all(value <= args.coupler_margin_rad + 1e-7
                                  for value in first_step_excess.values())
    )

    cycles = {}
    if diagnostic_prevalidation_pass:
        for mode in ("left", "right", "both"):
            cycles[mode] = run_cycle(diagnostic_xml, mode, args.coupler_margin_rad,
                                     out, dual, prior, source_limits, render=not args.no_video)
    else:
        cycles = {mode: {"status": "NOT RUN: diagnostic first-step prevalidation failed"}
                  for mode in ("left", "right", "both")}

    model_source = mujoco.MjModel.from_xml_path(str(snapshot_baseline))
    model_diag = mujoco.MjModel.from_xml_path(str(diagnostic_xml))
    range_compare = {}
    margin_compare = {}
    for joint in COUPLERS:
        a = object_id(model_source, mujoco.mjtObj.mjOBJ_JOINT, joint)
        b = object_id(model_diag, mujoco.mjtObj.mjOBJ_JOINT, joint)
        range_compare[joint] = {
            "source_range_rad": model_source.jnt_range[a].tolist(),
            "diagnostic_range_rad": model_diag.jnt_range[b].tolist(),
            "unchanged": bool(np.array_equal(model_source.jnt_range[a], model_diag.jnt_range[b])),
        }
        margin_compare[joint] = {
            "source_margin_rad": float(model_source.jnt_margin[a]),
            "diagnostic_margin_rad": float(model_diag.jnt_margin[b]),
        }

    repo_root = Path(__file__).resolve().parents[2]
    helper_head = subprocess.check_output([
        "git", "-C", str(repo_root), "show", f"HEAD:scripts/research/issue46_x2_robotiq_m0.py"
    ])
    result = {
        "status": "PASS" if diagnostic_prevalidation_pass and all(
            item.get("status") == "PASS" for item in cycles.values()) else "BLOCKED",
        "task": "Issue #46 dual Robotiq coupler startup recovery and dynamic verification",
        "source_faithful_verdict": "FAIL preserved: four couplers exceed the pinned upper limit at the first step",
        "simulation_only_classification": "SIMULATION_ONLY_DIAGNOSTIC; activation margin only; no limit range widening",
        "coupler_limit_activation_margin_rad": args.coupler_margin_rad,
        "runtime": {"python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(),
                    "native_library": str(Path(mujoco.__file__).with_name(
                        f"libmujoco.so.{mujoco.__version__}")),
                    "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
        "identity": {
            "robotsim_branch": git_value(repo_root, "branch", "--show-current"),
            "robotsim_head": git_value(repo_root, "rev-parse", "HEAD"),
            "runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
            "controller_helper_snapshot": str(helper_path),
            "controller_helper_snapshot_sha256": sha256(helper_path),
            "controller_helper_committed_head_sha256": hashlib.sha256(helper_head).hexdigest(),
            "controller_helper_is_dirty_worktree_snapshot": True,
            "baseline_model_source": str(baseline_source), "baseline_model_sha256": baseline_hash,
            "immutable_baseline_snapshot": str(snapshot_baseline),
            "immutable_baseline_snapshot_sha256": baseline_hash_copy,
            "diagnostic_model": str(diagnostic_xml), "diagnostic_model_sha256": sha256(diagnostic_xml),
            "menagerie_xml": str(menagerie_xml), "menagerie_xml_sha256": sha256(menagerie_xml),
            "menagerie_commit": git_value(menagerie_xml.parents[1], "rev-parse", "HEAD"),
            "x2_commit": git_value(prior.X2_DEFAULT, "rev-parse", "HEAD"),
            "canonical_helper": str(prior.CANONICAL_DEFAULT),
            "canonical_helper_sha256": (
                sha256(prior.CANONICAL_DEFAULT) if prior.CANONICAL_DEFAULT.is_file() else None),
        },
        "stage_1_menagerie_isolated_first_step": isolated,
        "stage_2_dual_assembly_source_vs_diagnostic": {
            "baseline_model_classification": baseline_classification,
            "source_head_torso_contact_at_reset": head_torso_contacts,
            "diagnostic_exact_pair_exclusion": (
                {"body1": "head_pitch_link", "body2": "torso_link",
                 "reason": "Previously authorized isolation of the verified internal source overlap"}
                if args.exclude_known_head_torso_source_contact else None),
            "source_faithful_and_margin_model_joint_ranges": range_compare,
            "joint_activation_margin_comparison": margin_compare,
            "one_step_startup_cases": dual_startup,
            "diagnostic_prevalidation_pass": diagnostic_prevalidation_pass,
            "diagnostic_first_step_coupler_excess_rad": first_step_excess,
            "source_limit_violations_recorded_not_relabelled": True,
        },
        "stage_3_actuator_driven_cycles": cycles,
        "stage_4_bottle_corridor": "NOT RUN: pass --canonical-helper for a static-only corridor audit",
        "stage_5_bottle_manipulation": "NOT RUN",
        "qpos_writes_after_rollout_start": 0,
        "bottle_contact": False,
    }
    if args.canonical_helper:
        result["stage_4_bottle_corridor"] = run_static_workcell_clearance(
            out, prior, args.canonical_helper, prior.SOURCE_CACHE,
            menagerie_xml.parents[1])
    with (out / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
