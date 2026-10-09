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
                  pad_midpoint_offset_world_m: np.ndarray) -> tuple[mujoco.MjModel, dict[str, Any]]:
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
    ET.SubElement(actuator, "position", {
        "name": LIFT_ACTUATOR, "joint": LIFT_JOINT, "kp": "1200", "kv": "70",
        "ctrlrange": "0 0.05", "ctrllimited": "true", "forcerange": "-25 25",
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
        "lift_actuator_gains_simulation_derived": {"kp": 1200.0, "kv": 70.0},
        "lift_actuator_effort_cap_simulation_derived_n": 25.0,
    }


def geom_label(model: mujoco.MjModel, gid: int) -> tuple[str, str]:
    body = int(model.geom_bodyid[gid])
    return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}")


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        force = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, force)
        b1, n1 = geom_label(model, g1)
        b2, n2 = geom_label(model, g2)
        if "m0_bottle" in (b1, b2):
            other = n2 if b1 == "m0_bottle" else n1
            side = "left" if other.startswith("left_pad") else ("right" if other.startswith("right_pad") else None)
            kind = "pad_bottle" if side else ("table_bottle" if "m0_table" in (b1, b2) else "other_bottle")
        else:
            side, kind = None, "other"
        rows.append({
            "geom1": n1, "body1": b1, "geom2": n2, "body2": b2,
            "kind": kind, "pad_side": side, "distance_m": float(c.dist),
            "position_world_m": c.pos.astype(float).tolist(),
            "force_local_n": force.astype(float).tolist() if c.efc_address >= 0 else None,
            "normal_force_n": float(force[0]) if c.efc_address >= 0 else 0.0,
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


def phase_steps(model: mujoco.MjModel, data: mujoco.MjData, name: str,
                duration: float, grip_start: float, grip_end: float,
                lift_start: float, lift_end: float,
                writer, renderer: mujoco.Renderer, camera: mujoco.MjvCamera,
                trace, contacts, summary, baseline_com_z: float) -> dict[str, Any]:
    steps = int(round(duration / float(model.opt.timestep)))
    grip_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACTUATOR)
    lift_id = m0.obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
    phase_start_time = float(data.time)
    bilateral_samples = 0
    bottle_rows = []
    max_lift = -float("inf")
    for k in range(steps):
        u = (k + 1) / steps
        s = m0.quintic(u)[0]
        data.ctrl[grip_id] = grip_start + s * (grip_end - grip_start)
        data.ctrl[lift_id] = lift_start + s * (lift_end - lift_start)
        mujoco.mj_step(model, data)
        crows = contact_rows(model, data)
        left = [c for c in crows if c["kind"] == "pad_bottle" and c["pad_side"] == "left"]
        right = [c for c in crows if c["kind"] == "pad_bottle" and c["pad_side"] == "right"]
        table = [c for c in crows if c["kind"] == "table_bottle"]
        both = bool(left and right)
        bilateral_samples += int(both)
        state = object_state(model, data)
        lift = float(state["center_of_mass_world_m"][2] - baseline_com_z)
        max_lift = max(max_lift, lift)
        row = {
            "step": int(round(float(data.time) / float(model.opt.timestep))),
            "time_s": float(data.time), "phase": name,
            "gripper_control": float(data.ctrl[grip_id]),
            "gripper_actuator_force_n": float(data.actuator_force[grip_id]),
            "lift_target_m": float(data.ctrl[lift_id]),
            "lift_actuator_force_n": float(data.actuator_force[lift_id]),
            "object": state, "object_lift_from_settled_com_m": lift,
            "left_pad_normal_force_sum_n": sum(c["normal_force_n"] for c in left),
            "right_pad_normal_force_sum_n": sum(c["normal_force_n"] for c in right),
            "table_normal_force_sum_n": sum(c["normal_force_n"] for c in table),
            "bilateral_pad_contact": both,
            "contacts": crows,
            "source_joint_limit_violations": m0.all_limited_joint_checks(model, data, {})[0],
            "qpos_writes_after_rollout_start": 0,
        }
        trace.append(row)
        contacts.append({"step": row["step"], "time_s": row["time_s"], "phase": name,
                         "contacts": crows, "bilateral_pad_contact": both,
                         "left_pad_normal_force_sum_n": row["left_pad_normal_force_sum_n"],
                         "right_pad_normal_force_sum_n": row["right_pad_normal_force_sum_n"],
                         "table_normal_force_sum_n": row["table_normal_force_sum_n"]})
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
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical helper hash differs from the accepted immutable helper")
        canonical = m0.load_module(args.canonical_helper)
        source = args.menagerie_root / m0.MENAGERIE_MJCF
        model, identity = build_fixture(source, args.menagerie_root, canonical, out,
                                        np.asarray(args.pad_midpoint_offset_world_m, dtype=float))
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
                                       float(settled_start["center_of_mass_world_m"][2]))
            baseline = object_state(model, data)
            baseline_com_z = float(baseline["center_of_mass_world_m"][2])
            phases.append(settle_phase)
            phases.append(phase_steps(model, data, "OPEN", 0.20, 0, 0, 0, 0,
                                      writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z))
            phases.append(phase_steps(model, data, "CLOSE", 0.60, 0, 255, 0, 0,
                                      writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z))
            hold = phase_steps(model, data, "HOLD", 1.0, 255, 255, 0, 0,
                               writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z)
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
            if bilateral_hold_pass:
                previous = 0.0
                for height in (0.001, 0.005, 0.030, 0.050):
                    move = phase_steps(model, data, f"LIFT_{int(height * 1000)}MM", 0.30, 255, 255, previous, height,
                                       writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z)
                    phases.append(move)
                    hold_lift = phase_steps(model, data, f"HOLD_{int(height * 1000)}MM", 0.20, 255, 255, height, height,
                                            writer, renderer, camera, trace_rows, contact_rows_all, phases, baseline_com_z)
                    phases.append(hold_lift)
                    lift_milestones.append({"target_m": height, "move": move, "hold": hold_lift,
                                            "real_com_lift_m": float(object_state(model, data)["center_of_mass_world_m"][2] - baseline_com_z),
                                            "table_contact_at_end": any(c["kind"] == "table_bottle" for c in contact_rows(model, data)),
                                            "bilateral_contact_at_end": any(c["kind"] == "pad_bottle" for c in contact_rows(model, data)) and
                                            len({c["pad_side"] for c in contact_rows(model, data) if c["kind"] == "pad_bottle"}) == 2})
                    previous = height
                    if lift_milestones[-1]["real_com_lift_m"] < height * 0.8 or not lift_milestones[-1]["bilateral_contact_at_end"]:
                        break
            result["lift_milestones"] = lift_milestones
            grip_phase = phase_steps(model, data, "RELEASE", 0.60, 255 if bilateral_hold_pass else 255, 0, float(data.qpos[m0.qpos_id(model, LIFT_JOINT)]),
                                     float(data.qpos[m0.qpos_id(model, LIFT_JOINT)]), writer, renderer, camera,
                                     trace_rows, contact_rows_all, phases, baseline_com_z)
            phases.append(grip_phase)
            release_contact_count = len([c for c in contact_rows(model, data) if c["kind"] == "pad_bottle"])
            settle_final = phase_steps(model, data, "FREE_SETTLE", 1.0, 0, 0,
                                       float(data.qpos[m0.qpos_id(model, LIFT_JOINT)]),
                                       float(data.qpos[m0.qpos_id(model, LIFT_JOINT)]), writer, renderer, camera,
                                       trace_rows, contact_rows_all, phases, baseline_com_z)
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
        reached = [m["target_m"] for m in lift_milestones if m["real_com_lift_m"] >= m["target_m"] * 0.8 and not m["table_contact_at_end"]]
        result["maximum_measured_bottle_lift_m"] = max((m["real_com_lift_m"] for m in lift_milestones), default=0.0)
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
