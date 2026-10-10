from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import issue46_native_omnihand_minimal_grasp as fixture  # noqa: E402


EVIDENCE_ROOT = Path(
    os.environ.get(
        "ISSUE46_GEOMETRY_EVIDENCE_DIR",
        "/tmp/issue46-native-omnihand-geometry-first",
    )
)
DT = fixture.DT
PENETRATION_LIMIT_M = fixture.PENETRATION_ABORT_M
CONTACT_ALPHA_STEP = 0.005
GEOMETRY_TOLERANCE_M = 0.001
FAMILIES = ("thumb", "index", "middle", "ring", "pinky")

CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "name": "prior_fixed_palm",
        "position": fixture.PALM_POSITION.copy(),
        "quaternion_wxyz": fixture.PALM_QUATERNION_WXYZ.copy(),
        "prior_dynamic_no_contact": True,
    },
    {
        "name": "offset_50x65mm",
        "position": np.array([0.250, 0.159, 0.9175], dtype=float),
        "quaternion_wxyz": fixture.PALM_QUATERNION_WXYZ.copy(),
        "prior_dynamic_no_contact": False,
    },
    {
        "name": "offset_55x70mm",
        "position": np.array([0.245, 0.154, 0.9175], dtype=float),
        "quaternion_wxyz": fixture.PALM_QUATERNION_WXYZ.copy(),
        "prior_dynamic_no_contact": False,
    },
)


def bilateral_contact_metrics(
    times_s: list[float],
    family_contacts: dict[str, list[bool]],
) -> dict[str, object]:
    sample_count = min(
        len(times_s),
        len(family_contacts.get("thumb", [])),
        len(family_contacts.get("index", [])),
    )
    simultaneous = [
        bool(family_contacts["thumb"][index] and family_contacts["index"][index])
        for index in range(sample_count)
    ]
    active_indices = [index for index, active in enumerate(simultaneous) if active]
    longest_run = 0
    current_run = 0
    for active in simultaneous:
        current_run = current_run + 1 if active else 0
        longest_run = max(longest_run, current_run)
    sample_period = float(times_s[1] - times_s[0]) if sample_count > 1 else DT
    return {
        "sample_count": sample_count,
        "simultaneous_frames": len(active_indices),
        "simultaneous_fraction": len(active_indices) / sample_count if sample_count else 0.0,
        "simultaneous_active_duration_s": len(active_indices) * sample_period,
        "longest_contiguous_duration_s": longest_run * sample_period,
        "first_simultaneous_time_s": float(times_s[active_indices[0]]) if active_indices else None,
        "last_simultaneous_time_s": float(times_s[active_indices[-1]]) if active_indices else None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_targets(
    details: dict[str, object],
    alphas: dict[str, float],
) -> dict[str, float]:
    targets = {name: float(details["joint_references"][name]) for name in details["source_joints"]}
    targets.update(fixture.OPEN_POSE)
    for joint_name, end in fixture.CLOSED_POSE.items():
        family = joint_name.split("_", 1)[1].split("_", 1)[0].lower()
        alpha = float(np.clip(alphas.get(family, 0.0), 0.0, 1.0))
        targets[joint_name] = float(fixture.OPEN_POSE[joint_name]) + alpha * (
            float(end) - float(fixture.OPEN_POSE[joint_name])
        )
    targets = fixture.derived_targets(targets, details)
    for joint_name, target in targets.items():
        low, high = details["joint_ranges"][joint_name]
        if target < low - 1e-10 or target > high + 1e-10:
            raise ValueError(f"source-valid pose exceeded {joint_name} range: {target} not in [{low}, {high}]")
    return targets


def assign_initial_hand_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    details: dict[str, object],
    targets: dict[str, float],
) -> int:
    writes = 0
    for joint_name in details["source_joints"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address] = fixture.source_target_to_raw(model, details, joint_name, targets[joint_name])
        writes += 1
    return writes


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def geometry_groups(model: mujoco.MjModel, details: dict[str, object]) -> dict[str, list[int]]:
    groups = {family: [] for family in FAMILIES}
    groups["palm"] = []
    groups["hand"] = []
    groups["bottle"] = []
    groups["table"] = []
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_palm")
    bottle_id = int(details["bottle_body_id"])
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        name = body_name(model, body_id).lower()
        if body_id == bottle_id:
            groups["bottle"].append(geom_id)
        elif name == "m0_table":
            groups["table"].append(geom_id)
        elif name.startswith("r_"):
            groups["hand"].append(geom_id)
            if body_id == palm_id:
                groups["palm"].append(geom_id)
            else:
                for family in FAMILIES:
                    if family in name:
                        groups[family].append(geom_id)
                        break
    return groups


def minimum_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    moving_geoms: list[int],
    fixed_geoms: list[int],
) -> dict[str, object]:
    best_distance = math.inf
    best_pair: tuple[int, int] | None = None
    best_segment = np.zeros(6, dtype=float)
    for first in moving_geoms:
        for second in fixed_geoms:
            segment = np.zeros(6, dtype=float)
            distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
            if distance < best_distance:
                best_distance = distance
                best_pair = (first, second)
                best_segment = segment.copy()
    if best_pair is None:
        return {"distance_m": None, "pair": None, "witness_hand_m": None, "witness_other_m": None}
    first, second = best_pair
    return {
        "distance_m": best_distance,
        "pair": {
            "hand_body": body_name(model, int(model.geom_bodyid[first])),
            "hand_geom": geom_name(model, first),
            "other_body": body_name(model, int(model.geom_bodyid[second])),
            "other_geom": geom_name(model, second),
        },
        "witness_hand_m": best_segment[:3].tolist(),
        "witness_other_m": best_segment[3:].tolist(),
    }


def set_kinematic_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    details: dict[str, object],
    targets: dict[str, float],
) -> None:
    for joint_name in details["source_joints"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = fixture.source_target_to_raw(
            model, details, joint_name, targets[joint_name]
        )
    mujoco.mj_forward(model, data)


def static_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    groups: dict[str, list[int]],
) -> dict[str, object]:
    result = {
        "family_to_bottle": {
            family: minimum_distance(model, data, groups[family], groups["bottle"])
            for family in FAMILIES
        },
        "palm_to_bottle": minimum_distance(model, data, groups["palm"], groups["bottle"]),
        "hand_to_table": minimum_distance(model, data, groups["hand"], groups["table"]),
        "palm_to_table": minimum_distance(model, data, groups["palm"], groups["table"]),
    }
    return result


def scan_family_envelope(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    details: dict[str, object],
    groups: dict[str, list[int]],
    candidate_name: str,
    output_csv: Path,
) -> dict[str, dict[str, object]]:
    rows: list[dict[str, object]] = []
    contacts: dict[str, dict[str, object]] = {}
    alphas = np.arange(0.0, 1.0 + CONTACT_ALPHA_STEP / 2.0, CONTACT_ALPHA_STEP)
    for family in FAMILIES:
        family_samples = []
        for alpha in alphas:
            targets = joint_targets(details, {family: float(alpha)})
            set_kinematic_pose(model, data, details, targets)
            distance = minimum_distance(model, data, groups[family], groups["bottle"])
            record = {
                "candidate": candidate_name,
                "family": family,
                "alpha": float(alpha),
                "minimum_distance_m": distance["distance_m"],
                "hand_body": distance["pair"]["hand_body"] if distance["pair"] else "",
                "hand_geom": distance["pair"]["hand_geom"] if distance["pair"] else "",
                "bottle_geom": distance["pair"]["other_geom"] if distance["pair"] else "",
                "witness_hand_x_m": distance["witness_hand_m"][0] if distance["witness_hand_m"] else "",
                "witness_hand_y_m": distance["witness_hand_m"][1] if distance["witness_hand_m"] else "",
                "witness_hand_z_m": distance["witness_hand_m"][2] if distance["witness_hand_m"] else "",
                "witness_bottle_x_m": distance["witness_other_m"][0] if distance["witness_other_m"] else "",
                "witness_bottle_y_m": distance["witness_other_m"][1] if distance["witness_other_m"] else "",
                "witness_bottle_z_m": distance["witness_other_m"][2] if distance["witness_other_m"] else "",
            }
            rows.append(record)
            family_samples.append((float(distance["distance_m"]), float(alpha), distance))
        feasible = [sample for sample in family_samples if sample[0] >= -GEOMETRY_TOLERANCE_M]
        if feasible:
            contacts[family] = min(feasible, key=lambda sample: abs(sample[0]))[2] | {
                "alpha": min(feasible, key=lambda sample: abs(sample[0]))[1]
            }
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return contacts


def render_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    closeup: bool = False,
) -> None:
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = fixture.camera_for_scene()
    camera.lookat[:] = [0.300, 0.035, 0.925]
    camera.distance = 0.38 if closeup else 0.52
    camera.azimuth = 180.0
    camera.elevation = -12.0
    renderer.update_scene(data, camera=camera)
    imageio.imwrite(output_path, renderer.render())
    renderer.close()


def find_bilateral_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    details: dict[str, object],
    groups: dict[str, list[int]],
    independent_contacts: dict[str, dict[str, object]],
) -> dict[str, object]:
    if "thumb" not in independent_contacts or "index" not in independent_contacts:
        return {"status": "FAIL", "reason": "native thumb or index has no source-valid contact envelope"}
    alphas = {
        "thumb": float(independent_contacts["thumb"]["alpha"]),
        "index": float(independent_contacts["index"]["alpha"]),
    }
    targets = joint_targets(details, alphas)
    set_kinematic_pose(model, data, details, targets)
    metrics = static_metrics(model, data, groups)
    distances = {family: metrics["family_to_bottle"][family]["distance_m"] for family in FAMILIES}
    targeted = ("thumb", "index")
    target_contact = all(
        distances[family] is not None and abs(float(distances[family])) <= GEOMETRY_TOLERANCE_M
        for family in targeted
    )
    other_fingers_clear = all(
        distances[family] is None or float(distances[family]) >= -GEOMETRY_TOLERANCE_M
        for family in ("middle", "ring", "pinky")
    )
    palm_clear = (
        float(metrics["palm_to_bottle"]["distance_m"]) >= 0.005
        and float(metrics["palm_to_table"]["distance_m"]) >= 0.005
    )
    table_clear = float(metrics["hand_to_table"]["distance_m"]) >= -GEOMETRY_TOLERANCE_M
    passed = target_contact and other_fingers_clear and palm_clear and table_clear
    return {
        "status": "PASS" if passed else "FAIL",
        "alphas": alphas,
        "source_joint_targets_rad": targets,
        "metrics": metrics,
        "target_thumb_and_index_surface_contact": target_contact,
        "other_fingers_nonpenetrating": other_fingers_clear,
        "palm_clearance_at_least_5mm": palm_clear,
        "whole_hand_table_clearance": table_clear,
    }


def contact_details(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bottle_body_id: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 != bottle_body_id and body2 != bottle_body_id:
            continue
        local_force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, local_force)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_on_geom2_world = frame.T @ local_force[:3]
        result.append(
            {
                "geom1": geom_name(model, geom1),
                "body1": body_name(model, body1),
                "geom2": geom_name(model, geom2),
                "body2": body_name(model, body2),
                "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
                "distance_m": float(contact.dist),
                "frame_rows": frame.tolist(),
                "force_local_n_xyz": local_force[:3].tolist(),
                "force_on_geom2_world_n_xyz": force_on_geom2_world.tolist(),
            }
        )
    return result


def run_dynamic_candidate(
    candidate: dict[str, object],
    static_result: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, details = fixture.build_model(
        "simulation-only",
        output_dir,
        palm_position=np.asarray(candidate["position"], dtype=float),
        palm_quaternion_wxyz=np.asarray(candidate["quaternion_wxyz"], dtype=float),
        palm_slide_z=True,
    )
    data = mujoco.MjData(model)
    groups = geometry_groups(model, details)
    contact_alphas = dict(static_result["alphas"])
    contact_probe_delta_alphas = {"thumb": 0.005, "index": 0.003}
    active_contact_alphas = {
        family: min(
            1.0,
            float(contact_alphas.get(family, 0.0)) + contact_probe_delta_alphas.get(family, 0.0),
        )
        for family in FAMILIES
    }
    middle_alpha = float(static_result.get("middle_alpha", 0.0))
    bottle_qpos_address = int(details["bottle_qpos_address"])
    initial_bottle_qpos = data.qpos[bottle_qpos_address : bottle_qpos_address + 7].copy()
    initial_targets = joint_targets(details, {})
    initial_hand_qpos_writes = assign_initial_hand_pose(model, data, details, initial_targets)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = fixture.camera_for_scene()
    camera.lookat[:] = [0.300, 0.035, 0.925]
    camera.distance = 0.42
    camera.azimuth = 180.0
    camera.elevation = -12.0
    trace_path = output_dir / "dynamic_trace.jsonl"
    video_path = output_dir / "dynamic_contact.mp4"
    trace = trace_path.open("w", encoding="utf-8")
    video = imageio.get_writer(video_path, fps=25, codec="libx264", quality=8)
    frames_per_step = max(1, round(1.0 / (25.0 * DT)))
    frame_count = 0
    stop_reason: str | None = None
    phase_summaries: list[dict[str, object]] = []
    contact_history: dict[str, list[bool]] = {family: [] for family in FAMILIES}
    table_support_history: list[float] = []
    hand_support_history: list[float] = []
    max_penetration = 0.0
    max_tracking_error: dict[str, float] = {name: 0.0 for name in details["source_joints"]}
    final_joint_states: dict[str, dict[str, float]] = {}
    initial_qpos_write_count = initial_hand_qpos_writes
    bottle_qpos_write_count = 0
    physics_steps = 0
    bottle_mass = float(model.body_mass[int(details["bottle_body_id"])])
    bottle_weight = bottle_mass * fixture.GRAVITY_M_S2
    phase_durations = (
        ("open_table_support_baseline", 0.4, {}, {}),
        ("thumb_index_approach", 1.4, {}, contact_alphas),
        ("contact_target_settle", 0.8, contact_alphas, contact_alphas),
        ("bounded_contact_probe", 0.25, contact_alphas, active_contact_alphas),
        ("contact_probe_hold", 0.5, active_contact_alphas, active_contact_alphas),
    )
    started = time.monotonic()

    def render_frame(force: bool = False) -> None:
        nonlocal frame_count
        if force or int(round(data.time / DT)) % frames_per_step == 0:
            renderer.update_scene(data, camera=camera)
            video.append_data(renderer.render())
            frame_count += 1

    def run_phase(label: str, duration: float, start_alphas: dict[str, float], end_alphas: dict[str, float]) -> dict[str, object]:
        nonlocal stop_reason, max_penetration, physics_steps
        steps = max(1, int(round(duration / DT)))
        phase_start = float(data.time)
        local_families = {family: [] for family in FAMILIES}
        local_times: list[float] = []
        local_table: list[float] = []
        local_hand: list[float] = []
        local_errors: dict[str, list[float]] = {name: [] for name in details["source_joints"]}
        local_max_penetration = 0.0
        for step_index in range(steps):
            alpha = fixture.smoothstep((step_index + 1) / steps)
            phase_alphas = {
                family: float(start_alphas.get(family, 0.0))
                + alpha * (float(end_alphas.get(family, 0.0)) - float(start_alphas.get(family, 0.0)))
                for family in FAMILIES
            }
            targets = joint_targets(details, phase_alphas)
            for joint_name, actuator_id in details["actuator_by_joint"].items():
                data.ctrl[actuator_id] = fixture.source_target_to_raw(
                    model, details, joint_name, targets[joint_name]
                )
            data.ctrl[int(details["palm_slide_z_actuator_id"])] = 0.0
            mujoco.mj_step(model, data)
            physics_steps += 1
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()):
                stop_reason = f"non-finite state at {label} step {step_index + 1}"
            contacts = fixture.bottle_contact_report(model, data, details)
            max_penetration = max(max_penetration, float(contacts["max_penetration_m"]))
            local_max_penetration = max(local_max_penetration, float(contacts["max_penetration_m"]))
            local_table.append(float(contacts["table_support_force_z_n"]))
            local_hand.append(float(contacts["hand_force_z_on_bottle_n"]))
            table_support_history.append(local_table[-1])
            hand_support_history.append(local_hand[-1])
            for family in FAMILIES:
                has_contact = family in set(contacts["finger_families"])
                local_families[family].append(has_contact)
                contact_history[family].append(has_contact)
            local_times.append(float(data.time))
            for joint_name in details["source_joints"]:
                measured = fixture.source_position(model, data, details, joint_name)
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                velocity = int(details["joint_axes"][joint_name]) * float(data.qvel[int(model.jnt_dofadr[joint_id])])
                error = measured - float(targets[joint_name])
                local_errors[joint_name].append(abs(error))
                max_tracking_error[joint_name] = max(max_tracking_error[joint_name], abs(error))
                low, high = details["joint_ranges"][joint_name]
                if measured < low - fixture.JOINT_LIMIT_TOLERANCE_RAD or measured > high + fixture.JOINT_LIMIT_TOLERANCE_RAD:
                    stop_reason = f"source joint limit exceeded at {label}: {joint_name}={measured:.9f}"
                final_joint_states[joint_name] = {
                    "target_source_rad": float(targets[joint_name]),
                    "position_source_rad": measured,
                    "velocity_source_rad_s": velocity,
                    "tracking_error_rad": error,
                }
            if local_max_penetration > PENETRATION_LIMIT_M:
                stop_reason = f"penetration exceeded {PENETRATION_LIMIT_M:.6f} m at {label}"
            bottle_position = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float)
            trace.write(
                json.dumps(
                    {
                        "time_s": float(data.time),
                        "phase": label,
                        "command_alphas": phase_alphas,
                        "finger_contact_families": contacts["finger_families"],
                        "contacts": contact_details(model, data, int(details["bottle_body_id"])),
                        "table_support_force_z_n": local_table[-1],
                        "hand_force_z_on_bottle_n": local_hand[-1],
                        "bottle_position_world_m": bottle_position.tolist(),
                        "bottle_quaternion_wxyz": data.xquat[int(details["bottle_body_id"])].tolist(),
                        "bottle_linear_velocity_world_m_s": data.cvel[int(details["bottle_body_id"])][3:].tolist(),
                        "bottle_angular_velocity_world_rad_s": data.cvel[int(details["bottle_body_id"])][:3].tolist(),
                        "palm_vertical_position_m": float(data.qpos[int(details["palm_slide_z_qpos_address"])]),
                        "joints": final_joint_states,
                        "bottle_qpos_write_count": bottle_qpos_write_count,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if stop_reason:
                break
            render_frame()
        n = min(50, len(local_table))
        final_window = slice(-n, None)
        bilateral = bilateral_contact_metrics(local_times, local_families)
        final_bilateral = bilateral_contact_metrics(
            local_times[-n:],
            {family: values[-n:] for family, values in local_families.items()},
        )
        summary = {
            "phase": label,
            "requested_duration_s": duration,
            "actual_duration_s": float(data.time) - phase_start,
            "contact_fraction_by_family": {
                family: float(np.mean(values)) if values else 0.0 for family, values in local_families.items()
            },
            "final_100ms_contact_fraction_by_family": {
                family: float(np.mean(values[-n:])) if values else 0.0 for family, values in local_families.items()
            },
            "simultaneous_thumb_index_contact": bilateral,
            "final_100ms_simultaneous_thumb_index_contact": final_bilateral,
            "mean_final_100ms_table_support_force_z_n": float(np.mean(local_table[final_window])) if local_table else 0.0,
            "mean_final_100ms_hand_force_z_on_bottle_n": float(np.mean(local_hand[final_window])) if local_hand else 0.0,
            "mean_final_100ms_hand_support_fraction": float(np.mean(local_hand[final_window])) / bottle_weight if local_hand else 0.0,
            "maximum_joint_tracking_error_rad": {name: max(values, default=0.0) for name, values in local_errors.items()},
            "maximum_penetration_m": local_max_penetration,
            "stop_reason": stop_reason,
        }
        phase_summaries.append(summary)
        return summary

    try:
        render_frame(force=True)
        for label, duration, start_alphas, end_alphas in phase_durations:
            run_phase(label, duration, start_alphas, end_alphas)
            if stop_reason:
                break

        contact_phase = phase_summaries[-1]
        bilateral_fraction = min(
            contact_phase["final_100ms_contact_fraction_by_family"]["thumb"],
            contact_phase["final_100ms_contact_fraction_by_family"]["index"],
        )
        bilateral_established = bilateral_fraction >= 0.8

        if bilateral_established and not stop_reason:
            middle_end = {"middle": middle_alpha}
            middle_phase = run_phase(
                "add_middle_after_bilateral_contact",
                0.9,
                active_contact_alphas,
                {**active_contact_alphas, **middle_end},
            )
            bilateral_established = bilateral_established and min(
                middle_phase["final_100ms_contact_fraction_by_family"]["thumb"],
                middle_phase["final_100ms_contact_fraction_by_family"]["index"],
            ) >= 0.8

        if bilateral_established and not stop_reason:
            squeeze_delta = 0.03
            squeeze_targets = {
                family: min(1.0, float(active_contact_alphas.get(family, 0.0)) + squeeze_delta)
                for family in FAMILIES
            }
            if phase_summaries[-1]["final_100ms_contact_fraction_by_family"].get("middle", 0.0) < 0.8:
                squeeze_targets["middle"] = float(active_contact_alphas.get("middle", 0.0))
            run_phase("bounded_load_transfer_squeeze", 0.5, squeeze_targets, squeeze_targets)
            run_phase("load_transfer_hold", 0.5, squeeze_targets, squeeze_targets)

            last_hold = phase_summaries[-1]
            open_support = float(phase_summaries[0]["mean_final_100ms_table_support_force_z_n"])
            hand_fraction = float(last_hold["mean_final_100ms_hand_support_fraction"])
            table_fraction_reduction = (open_support - float(last_hold["mean_final_100ms_table_support_force_z_n"])) / bottle_weight
            load_transferred = hand_fraction >= 0.10 and table_fraction_reduction >= 0.10
            if load_transferred and not stop_reason:
                start_palm_z = float(data.qpos[int(details["palm_slide_z_qpos_address"])])
                initial_bottle_position = np.asarray(
                    data.xpos[int(details["bottle_body_id"])], dtype=float
                ).copy()
                for label, lift, minimum in (("physical_lift_1mm", 0.001, 0.0008), ("physical_lift_5mm", 0.005, 0.0045)):
                    if label == "physical_lift_5mm" and phase_summaries[-1].get("status") == "FAIL":
                        break
                    steps = int(round((0.5 if lift <= 0.001 else 0.8) / DT))
                    phase_start = float(data.time)
                    for step_index in range(steps):
                        blend = fixture.smoothstep((step_index + 1) / steps)
                        target_palm_z = start_palm_z + blend * lift
                        targets = joint_targets(details, squeeze_targets)
                        for joint_name, actuator_id in details["actuator_by_joint"].items():
                            data.ctrl[actuator_id] = fixture.source_target_to_raw(model, details, joint_name, targets[joint_name])
                        data.ctrl[int(details["palm_slide_z_actuator_id"])] = target_palm_z
                        mujoco.mj_step(model, data)
                        physics_steps += 1
                        contacts = fixture.bottle_contact_report(model, data, details)
                        if float(contacts["max_penetration_m"]) > PENETRATION_LIMIT_M:
                            stop_reason = f"penetration exceeded limit during {label}"
                            break
                        render_frame()
                    bottle_end = np.asarray(data.xpos[int(details["bottle_body_id"])], dtype=float).copy()
                    measured_lift = float(bottle_end[2] - initial_bottle_position[2])
                    end_contacts = fixture.bottle_contact_report(model, data, details)
                    passed = (
                        measured_lift >= minimum
                        and not end_contacts["table_contact_present"]
                        and "thumb" in end_contacts["finger_families"]
                        and bool(set(end_contacts["finger_families"]).intersection({"index", "middle", "ring", "pinky"}))
                        and not stop_reason
                    )
                    phase_summaries.append(
                        {
                            "phase": label,
                            "requested_lift_m": lift,
                            "measured_bottle_lift_m": measured_lift,
                            "minimum_required_lift_m": minimum,
                            "actual_duration_s": float(data.time) - phase_start,
                            "table_contact_at_end": bool(end_contacts["table_contact_present"]),
                            "finger_contact_families_at_end": end_contacts["finger_families"],
                            "status": "PASS" if passed else "FAIL",
                            "stop_reason": stop_reason,
                        }
                    )
                    if not passed:
                        stop_reason = stop_reason or f"{label} gate failed"
                        break
            else:
                phase_summaries.append(
                    {
                        "phase": "contact_driven_load_transfer",
                        "status": "PASS" if load_transferred else "FAIL",
                        "hand_support_fraction_of_bottle_weight": hand_fraction,
                        "table_support_reduction_fraction_of_bottle_weight": table_fraction_reduction,
                        "small_lift_attempts": "not_run_without_10_percent_load_transfer",
                    }
                )
        else:
            phase_summaries.append(
                {
                    "phase": "contact_driven_load_transfer",
                    "status": "NOT_RUN",
                    "reason": "persistent thumb plus opposing-index contact gate was not met",
                    "small_lift_attempts": "not_run",
                }
            )
        render_frame(force=True)
    finally:
        trace.close()
        video.close()
        renderer.close()

    final_bottle_qpos = data.qpos[bottle_qpos_address : bottle_qpos_address + 7].copy()
    bottle_qpos_changed = not np.array_equal(initial_bottle_qpos, final_bottle_qpos)
    endpoint_geometry_metrics = static_metrics(model, data, groups)
    contact_phase = next(
        (phase for phase in phase_summaries if phase["phase"] == "contact_probe_hold"),
        next(
            (phase for phase in phase_summaries if phase["phase"] == "contact_target_settle"),
            next((phase for phase in phase_summaries if phase["phase"] == "thumb_index_approach"), {}),
        ),
    )
    final100_thumb = float(contact_phase.get("final_100ms_contact_fraction_by_family", {}).get("thumb", 0.0))
    final100_index = float(contact_phase.get("final_100ms_contact_fraction_by_family", {}).get("index", 0.0))
    result = {
        "candidate": candidate["name"],
        "mode": "simulation-only",
        "status": "PASS" if final100_thumb >= 0.8 and final100_index >= 0.8 else "FAIL",
        "persistent_thumb_index_contact": final100_thumb >= 0.8 and final100_index >= 0.8,
        "stop_reason": stop_reason,
        "candidate_palm_position_world_m": np.asarray(candidate["position"], dtype=float).tolist(),
        "candidate_palm_quaternion_wxyz": np.asarray(candidate["quaternion_wxyz"], dtype=float).tolist(),
        "contact_alpha_targets": contact_alphas,
        "bounded_contact_probe_delta_alpha": contact_probe_delta_alphas,
        "active_contact_alpha_targets_after_probe": active_contact_alphas,
        "middle_contact_alpha_target": middle_alpha,
        "palm_z_actuator": {
            "present": True,
            "range_m": [0.0, 0.05],
            "kp": 400.0,
            "kv": 40.0,
            "used_only_after_load_transfer_gate": True,
        },
        "timestep_s": DT,
        "integrator": "implicitfast",
        "source_joint_ranges_enforced": True,
        "joint_limit_tolerance_rad": fixture.JOINT_LIMIT_TOLERANCE_RAD,
        "mimic_mode": "simulation-only direct follower position actuators with source-derived targets",
        "mimic_equalities": int(model.neq),
        "actuated_hand_joint_names": sorted(details["actuator_by_joint"]),
        "initial_hand_qpos_assignments": initial_qpos_write_count,
        "bottle_qpos_writes_after_initialization": bottle_qpos_write_count,
        "bottle_qpos_changed_via_physics": bottle_qpos_changed,
        "bottle_mass_kg": bottle_mass,
        "bottle_weight_n": bottle_weight,
        "maximum_penetration_m": max_penetration,
        "physics_steps": physics_steps,
        "maximum_joint_tracking_error_rad": max_tracking_error,
        "final_joint_states": final_joint_states,
        "endpoint_kinematic_geometry": endpoint_geometry_metrics,
        "phase_results": phase_summaries,
        "wall_duration_s": time.monotonic() - started,
        "video": str(video_path),
        "trace": str(trace_path),
        "frames": frame_count,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fixture_sha256": sha256(Path(fixture.__file__).resolve()),
        "vendor_commit": fixture.git_output(fixture.VENDOR_ROOT, "rev-parse", "HEAD"),
        "source_urdf_sha256": sha256(fixture.URDF),
        "mujoco_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "evidence_directory": str(output_dir),
        "hardware_semantics": "not verified; simulation-only direct follower actuators and diagnostic palm slide",
    }
    (output_dir / "dynamic_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def analyze_candidate(candidate: dict[str, object], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, details = fixture.build_model(
        "simulation-only",
        output_dir,
        palm_position=np.asarray(candidate["position"], dtype=float),
        palm_quaternion_wxyz=np.asarray(candidate["quaternion_wxyz"], dtype=float),
    )
    data = mujoco.MjData(model)
    groups = geometry_groups(model, details)
    open_targets = joint_targets(details, {})
    initial_assignments = assign_initial_hand_pose(model, data, details, open_targets)
    mujoco.mj_forward(model, data)
    open_metrics = static_metrics(model, data, groups)
    render_state(model, data, output_dir / "static_open.png")
    contacts = scan_family_envelope(
        model,
        data,
        details,
        groups,
        str(candidate["name"]),
        output_dir / "contact_envelope.csv",
    )
    bilateral = find_bilateral_targets(model, data, details, groups, contacts)
    if bilateral.get("status") == "PASS":
        set_kinematic_pose(model, data, details, bilateral["source_joint_targets_rad"])
        render_state(model, data, output_dir / "static_bilateral_contact_target.png", closeup=True)
    result = {
        "candidate": candidate["name"],
        "palm_position_world_m": np.asarray(candidate["position"], dtype=float).tolist(),
        "palm_quaternion_wxyz": np.asarray(candidate["quaternion_wxyz"], dtype=float).tolist(),
        "prior_dynamic_no_contact": bool(candidate["prior_dynamic_no_contact"]),
        "mode": "kinematic source-valid geometry only; zero physics steps",
        "open_pose_initial_qpos_assignments": initial_assignments,
        "open_pose_metrics": open_metrics,
        "independent_family_contact_targets": contacts,
        "bilateral_target": bilateral,
        "source_joint_ranges_rad": {name: list(bounds) for name, bounds in details["joint_ranges"].items()},
        "generated_model_xml_sha256": details["model_xml_sha256"],
        "fixture_sha256": sha256(Path(fixture.__file__).resolve()),
        "vendor_commit": fixture.git_output(fixture.VENDOR_ROOT, "rev-parse", "HEAD"),
        "source_urdf_sha256": sha256(fixture.URDF),
        "mujoco_version": mujoco.__version__,
        "evidence_directory": str(output_dir),
    }
    (output_dir / "geometry_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def analyze_existing_trial(candidate_name: str, trace_path: Path) -> dict[str, object]:
    candidate = next((item for item in CANDIDATES if item["name"] == candidate_name), None)
    if candidate is None:
        raise ValueError(f"unknown candidate: {candidate_name}")
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"trial trace is empty: {trace_path}")
    endpoint = records[-1]
    with tempfile.TemporaryDirectory(prefix="issue46-endpoint-analysis-") as tempdir:
        model, details = fixture.build_model(
            "simulation-only",
            Path(tempdir),
            palm_position=np.asarray(candidate["position"], dtype=float),
            palm_quaternion_wxyz=np.asarray(candidate["quaternion_wxyz"], dtype=float),
            palm_slide_z=True,
        )
        data = mujoco.MjData(model)
        for joint_name, state in endpoint["joints"].items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = int(model.jnt_qposadr[joint_id])
            sign = int(details["joint_axes"][joint_name])
            reference = float(details["joint_references"][joint_name])
            data.qpos[qpos_address] = float(model.qpos0[qpos_address]) + sign * (
                float(state["position_source_rad"]) - reference
            )
        bottle_qpos_address = int(details["bottle_qpos_address"])
        data.qpos[bottle_qpos_address : bottle_qpos_address + 7] = (
            endpoint["bottle_position_world_m"] + endpoint["bottle_quaternion_wxyz"]
        )
        data.qpos[int(details["palm_slide_z_qpos_address"])] = endpoint["palm_vertical_position_m"]
        mujoco.mj_forward(model, data)
        endpoint_metrics = static_metrics(model, data, geometry_groups(model, details))

    max_errors: dict[str, dict[str, object]] = {}
    for record in records:
        for joint_name, state in record["joints"].items():
            previous = max_errors.get(joint_name)
            error = abs(float(state["tracking_error_rad"]))
            if previous is None or error > float(previous["absolute_error_rad"]):
                max_errors[joint_name] = {
                    "absolute_error_rad": error,
                    "signed_error_rad": float(state["tracking_error_rad"]),
                    "time_s": float(record["time_s"]),
                    "phase": record["phase"],
                }
    static_result_path = trace_path.parent.parent / "geometry_result.json"
    static_result = json.loads(static_result_path.read_text(encoding="utf-8"))
    predicted = static_result["bilateral_target"]["metrics"]["family_to_bottle"]
    actual_distances = {
        family: endpoint_metrics["family_to_bottle"][family]["distance_m"] for family in FAMILIES
    }
    result = {
        "candidate": candidate_name,
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "endpoint_time_s": endpoint["time_s"],
        "analysis_physics_steps": 0,
        "endpoint_measured_joint_positions_source_rad": {
            name: state["position_source_rad"] for name, state in endpoint["joints"].items()
        },
        "endpoint_joint_velocities_source_rad_s": {
            name: state["velocity_source_rad_s"] for name, state in endpoint["joints"].items()
        },
        "maximum_absolute_tracking_error_by_joint": max_errors,
        "predicted_surface_gaps_at_static_contact_target_m": {
            family: predicted[family]["distance_m"] for family in FAMILIES
        },
        "surface_gaps_reconstructed_from_measured_endpoint_qpos_m": actual_distances,
        "gap_increase_from_static_target_m": {
            family: float(actual_distances[family] - predicted[family]["distance_m"])
            for family in FAMILIES
        },
        "palm_to_bottle_gap_m": endpoint_metrics["palm_to_bottle"]["distance_m"],
        "hand_to_table_gap_m": endpoint_metrics["hand_to_table"]["distance_m"],
        "runtime_contact_families": endpoint["finger_contact_families"],
        "runtime_contact_details": endpoint["contacts"],
        "evidence_directory": str(trace_path.parent),
    }
    output_path = trace_path.parent / "tracking_diagnosis.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded X2 native OmniHand geometry-first contact trial")
    parser.add_argument("--candidate", choices=("auto", "prior_fixed_palm", "offset_50x65mm", "offset_55x70mm"), default="auto")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--analyze-existing-trial", type=Path)
    args = parser.parse_args()
    fixture.verify_vendor_pin()
    if args.analyze_existing_trial:
        result = analyze_existing_trial(args.candidate, args.analyze_existing_trial)
        print(json.dumps(result, indent=2))
        return 0
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "status": "BLOCKED",
        "experiment": "Issue #46 native OmniHand geometry-first contact",
        "physics_steps_before_dynamic_trial": 0,
        "candidate_limit": 3,
        "attempted_dynamic_candidates": [],
        "geometry_candidates": [],
        "vendor_commit": fixture.git_output(fixture.VENDOR_ROOT, "rev-parse", "HEAD"),
        "source_urdf_sha256": sha256(fixture.URDF),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fixture_sha256": sha256(Path(fixture.__file__).resolve()),
        "pinned_source_joint_ranges_enforced": True,
        "bottle_geometry_modified": False,
        "hand_collision_geometry_modified": False,
        "source_faithful_mimic_tuning_repeated": False,
        "evidence_directory": str(EVIDENCE_ROOT),
    }
    selected = [candidate for candidate in CANDIDATES if args.candidate == "auto" or candidate["name"] == args.candidate]
    if not selected:
        raise ValueError(f"unknown candidate selection: {args.candidate}")
    analyzed: dict[str, dict[str, object]] = {}
    for candidate in selected:
        candidate_dir = EVIDENCE_ROOT / str(candidate["name"])
        analyzed[str(candidate["name"])] = analyze_candidate(candidate, candidate_dir)
        results["geometry_candidates"].append(analyzed[str(candidate["name"])])
    if args.geometry_only:
        results["status"] = "GEOMETRY_ONLY"
    else:
        dynamic_candidates = [
            candidate
            for candidate in selected
            if not candidate["prior_dynamic_no_contact"]
            and analyzed[str(candidate["name"])]["bilateral_target"].get("status") == "PASS"
        ]
        if args.candidate != "auto":
            dynamic_candidates = [
                candidate for candidate in dynamic_candidates if candidate["name"] == args.candidate
            ]
        for candidate in dynamic_candidates:
            candidate_name = str(candidate["name"])
            static = analyzed[candidate_name]["bilateral_target"]
            dynamic = run_dynamic_candidate(
                candidate,
                {
                    **static,
                    "middle_alpha": float(
                        analyzed[candidate_name]["independent_family_contact_targets"].get("middle", {}).get("alpha", 0.0)
                    ),
                },
                EVIDENCE_ROOT / candidate_name / "dynamic",
            )
            results["attempted_dynamic_candidates"].append(dynamic)
            if dynamic["persistent_thumb_index_contact"]:
                results["status"] = "CONTACT_GATE_PASS"
                break
        if results["status"] == "BLOCKED":
            results["failure_reason"] = "no tested dynamic candidate established persistent thumb plus index contact"
    (EVIDENCE_ROOT / "geometry_first_result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "evidence_directory": results["evidence_directory"],
                "geometry_candidates": [
                    {
                        "candidate": item["candidate"],
                        "open_min_family_clearance_m": min(
                            value["distance_m"]
                            for value in item["open_pose_metrics"]["family_to_bottle"].values()
                        ),
                        "bilateral_target_status": item["bilateral_target"]["status"],
                        "contact_alphas": item["bilateral_target"].get("alphas"),
                    }
                    for item in results["geometry_candidates"]
                ],
                "dynamic_attempts": [
                    {
                        "candidate": item["candidate"],
                        "status": item["status"],
                        "persistent_thumb_index_contact": item["persistent_thumb_index_contact"],
                        "physics_steps": item["physics_steps"],
                        "stop_reason": item["stop_reason"],
                    }
                    for item in results["attempted_dynamic_candidates"]
                ],
            },
            indent=2,
        )
    )
    return 0 if results["status"] in {"CONTACT_GATE_PASS", "GEOMETRY_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
