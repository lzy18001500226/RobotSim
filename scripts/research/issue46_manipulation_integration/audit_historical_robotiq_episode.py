#!/usr/bin/env python3
"""Apply the current episode gate to a saved X2/Robotiq rollout without simulation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
REVIEWS = Path("/home/lzy18001500226/projects/RobotSim/reviews")
sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import path_governance as pg

if __package__:
    from .episode_gate import EpisodeCriteria, EpisodeFrame, ManipulationPhase, PHASE_ORDER, evaluate_episode_evidence
else:
    from episode_gate import EpisodeCriteria, EpisodeFrame, ManipulationPhase, PHASE_ORDER, evaluate_episode_evidence


ARM_JOINTS = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
REQUIRED_CONTACT_FRACTION = 0.80
MINIMUM_SUPPORT_N = 0.57 * 9.81
MINIMUM_TABLE_UNLOADING_N = 3.0
MAXIMUM_RELATIVE_TRANSLATION_M = 0.015
MINIMUM_RELEASE_OPENING_M = 0.0884


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_vec(value: str | list | tuple, expected: int) -> tuple[float, ...]:
    values = json.loads(value) if isinstance(value, str) else value
    result = tuple(float(item) for item in values)
    if len(result) != expected or not all(math.isfinite(item) for item in result):
        raise ValueError(f"expected {expected} finite values")
    return result


def infer_timestep_s(rows: list[dict]) -> float:
    times = [float(row["time_s"]) for row in rows]
    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    if not intervals or any(not math.isfinite(value) or value <= 0.0 for value in intervals):
        raise ValueError("physics trace timestamps are not strictly increasing")
    dt = sorted(intervals)[len(intervals) // 2]
    if max(abs(value - dt) for value in intervals) > 1e-8:
        raise ValueError("physics trace does not support one fixed timestep for step-index reconstruction")
    return dt


def sample_step_index(row: dict, timestep_s: float) -> int:
    """Recover the post-step index from time; historical integer step had truncation duplicates."""
    return int(round(float(row["time_s"]) / timestep_s))


def load_contact_events(path: Path, timestep_s: float) -> tuple[dict[int, dict], list[dict]]:
    by_step: dict[int, dict] = defaultdict(lambda: {
        "pad_active": set(), "force_roles": set(), "table_contact": False,
        "left_hand_contact": False, "nonpad_bodies": set(),
    })
    bottle_rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            bodies = {str(row.get("body1", "")), str(row.get("body2", ""))}
            if "m0_bottle" not in bodies:
                continue
            step = sample_step_index(row, timestep_s)
            other = next((body for body in bodies if body != "m0_bottle"), "")
            bottle_rows.append(row)
            event = by_step[step]
            if other == "m0_table":
                event["table_contact"] = True
                continue
            if other in {"rq_left_pad", "rq_right_pad"}:
                role = "left_pad" if other == "rq_left_pad" else "right_pad"
                event["pad_active"].add(role)
                if float(row.get("normal_force_n") or 0.0) > 0.0:
                    event["force_roles"].add(role)
                continue
            if other.startswith("rq_"):
                event["nonpad_bodies"].add(other)
                continue
            if other.startswith("left_") or "left_hand" in other or "left_wrist" in other:
                event["left_hand_contact"] = True
            event["nonpad_bodies"].add(other)
    return dict(by_step), bottle_rows


def _phase_rows(rows: list[dict], phase: str) -> list[dict]:
    return [row for row in rows if row["phase"] == phase]


def source_arm_trace_summary(rows: list[dict], joint_limits: dict, timestep_s: float) -> dict:
    max_error = {joint: 0.0 for joint in ARM_JOINTS}
    max_qvel = {joint: 0.0 for joint in ARM_JOINTS}
    position_violations = []
    velocity_violations = []
    coverage = 0
    for row in rows:
        try:
            qpos = parse_vec(row["right_arm_qpos_rad"], len(ARM_JOINTS))
            qvel = parse_vec(row["right_arm_qvel_rad_s"], len(ARM_JOINTS))
            target = parse_vec(row["right_arm_target_rad"], len(ARM_JOINTS))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        coverage += 1
        for index, joint in enumerate(ARM_JOINTS):
            limit = joint_limits.get(joint)
            if not limit:
                continue
            error = abs(target[index] - qpos[index])
            speed = abs(qvel[index])
            max_error[joint] = max(max_error[joint], error)
            max_qvel[joint] = max(max_qvel[joint], speed)
            if qpos[index] < float(limit["lower"]) or qpos[index] > float(limit["upper"]):
                position_violations.append({"step": sample_step_index(row, timestep_s), "joint": joint, "q_rad": qpos[index],
                                            "source_range_rad": [float(limit["lower"]), float(limit["upper"])]})
            if speed > float(limit["velocity"]):
                velocity_violations.append({"step": sample_step_index(row, timestep_s), "joint": joint, "qvel_rad_s": qvel[index],
                                            "source_limit_rad_s": float(limit["velocity"])})
    return {
        "scope": "right-arm joints present in saved CSV only; not a full-model limit certificate",
        "sample_count": coverage,
        "trace_row_count": len(rows),
        "position_status": "PASS_PARTIAL_RIGHT_ARM" if coverage == len(rows) and not position_violations else "BLOCKED_OR_FAIL",
        "velocity_status": "PASS_PARTIAL_RIGHT_ARM" if coverage == len(rows) and not velocity_violations else "BLOCKED_OR_FAIL",
        "max_abs_tracking_error_rad_by_joint": max_error,
        "max_abs_qvel_rad_s_by_joint": max_qvel,
        "position_violation_count": len(position_violations),
        "position_violation_examples": position_violations[:10],
        "velocity_violation_count": len(velocity_violations),
        "velocity_violation_examples": velocity_violations[:10],
    }


def build_episode_frames(
    rows: list[dict], contact_by_step: dict[int, dict], history: dict, model_sha: str, timestep_s: float,
) -> tuple[list[EpisodeFrame], dict]:
    by_name = {phase: _phase_rows(rows, phase) for phase in (
        "pregrasp_settle", "approach", "close", "hold", "lift_080mm_cartesian_path",
        "mounted_lift_hold_1s", "loaded_transfer", "lower_to_target", "release_at_target",
        "open_hold_at_target", "open_gripper_retract", "free_physics_settle",
    )}
    bilateral_steps = [
        sample_step_index(row, timestep_s) for row in by_name["close"] + by_name["hold"]
        if {"left_pad", "right_pad"}.issubset(contact_by_step.get(sample_step_index(row, timestep_s), {}).get("force_roles", set()))
    ]
    first_bilateral_step = min(bilateral_steps) if bilateral_steps else None
    contact_rows = []
    if first_bilateral_step is not None:
        contact_rows.extend(row for row in by_name["close"] if sample_step_index(row, timestep_s) >= first_bilateral_step)
        contact_rows.extend(by_name["hold"])

    lift_rows = by_name["lift_080mm_cartesian_path"]
    first_air_index = next((index for index, row in enumerate(lift_rows)
                            if not contact_by_step.get(sample_step_index(row, timestep_s), {}).get("table_contact", False)), None)
    if first_air_index is None:
        load_rows, airborne_rows = lift_rows, []
    else:
        load_rows, airborne_rows = lift_rows[:first_air_index], lift_rows[first_air_index:]

    release_candidates = by_name["release_at_target"] + by_name["open_hold_at_target"] + by_name["open_gripper_retract"]
    first_open_index = next((index for index, row in enumerate(release_candidates)
                             if float(row["pad_gap_m"]) >= MINIMUM_RELEASE_OPENING_M), None)
    release_rows = release_candidates[first_open_index:] if first_open_index is not None else []

    history_adapter = history["adapter"]
    initial_root = parse_vec(history_adapter["bottle_root_position_world_m"], 3)
    initial_com = parse_vec(history["initial_bottle_com_m"], 3)
    com_offset = tuple(initial_com[i] - initial_root[i] for i in range(3))
    target_root = parse_vec(history["stages"]["loaded_transfer"]["target_bottle_root_m"], 3)
    target_com = tuple(target_root[i] + com_offset[i] for i in range(3))
    model_id = f"X2+Robotiq2F85:{history_adapter['model_configuration']}:{model_sha}"

    first_contact_row = next((row for row in sorted(rows, key=lambda item: int(item["step"]))
                              if first_bilateral_step is not None and sample_step_index(row, timestep_s) == first_bilateral_step), None)
    slip_reference = parse_vec(first_contact_row["root_to_tcp_local_m"], 3) if first_contact_row else None
    integrity = history["physical_integrity"]

    phase_groups = {
        ManipulationPhase.PREGRASP: by_name["pregrasp_settle"],
        ManipulationPhase.APPROACH: by_name["approach"],
        ManipulationPhase.CONTACT: sorted(contact_rows, key=lambda row: sample_step_index(row, timestep_s)),
        ManipulationPhase.LOAD_TRANSFER: load_rows,
        ManipulationPhase.LIFT_50MM: airborne_rows,
        ManipulationPhase.HOLD: by_name["mounted_lift_hold_1s"],
        ManipulationPhase.TRANSFER: by_name["loaded_transfer"],
        ManipulationPhase.PLACE: by_name["lower_to_target"],
        ManipulationPhase.RELEASE: release_rows,
        ManipulationPhase.SETTLE: by_name["free_physics_settle"],
    }

    frames: list[EpisodeFrame] = []
    for phase in PHASE_ORDER:
        for row in phase_groups[phase]:
            step = sample_step_index(row, timestep_s)
            contact = contact_by_step.get(step, {})
            root_to_tcp = parse_vec(row["root_to_tcp_local_m"], 3)
            slip = math.dist(root_to_tcp, slip_reference) if slip_reference is not None else None
            qw, qx, qy, qz = parse_vec(row["bottle_quat_wxyz"], 4)
            frames.append(EpisodeFrame(
                phase=phase,
                simulation_time_s=float(row["time_s"]),
                physics_step_index=step,
                model_id=model_id,
                model_sha256=model_sha,
                bottle_com_world_m=(float(row["bottle_com_x_m"]), float(row["bottle_com_y_m"]), float(row["bottle_com_z_m"])),
                bottle_quat_world_xyzw=(qx, qy, qz, qw),
                bottle_linear_speed_m_s=float(row["bottle_linear_speed_m_s"]),
                bottle_angular_speed_rad_s=float(row["bottle_angular_speed_rad_s"]),
                target_bottle_com_world_m=target_com if phase is ManipulationPhase.PLACE else None,
                force_bearing_contact_roles=frozenset(contact.get("force_roles", set())),
                tool_object_contact_active=bool(contact.get("pad_active", set())),
                left_hand_contact=bool(contact.get("left_hand_contact", False)),
                table_contact=bool(contact.get("table_contact", False)),
                # Trace has contact-local force scalars but no contact-frame orientation to project onto world +Z.
                tool_upward_support_n=None,
                table_reaction_vertical_n=float(row["table_normal_force_n"]),
                max_relative_slip_m=slip,
                # Dynamic rows do not contain per-sample full-scene distance measurements.
                minimum_approach_clearance_m=None,
                tool_opening_m=float(row["pad_gap_m"]),
                # Only seven right-arm states and two driver positions were recorded, not all model joints.
                source_position_limits_pass=None,
                source_velocity_limits_pass=None,
                actuator_limits_pass=None,
                follower_constraints_pass=None,
                bottle_qpos_writes_after_start=int(row["qpos_writes_after_rollout_start"]),
                bottle_attachment_active=bool(integrity["bottle_weld_or_equality"]),
                mocap_carry_active=bool(integrity["bottle_mocap_or_follow"]),
                hidden_support_active=bool(integrity["hidden_support"]),
            ))

    mapping = {
        "first_bilateral_force_bearing_step": first_bilateral_step,
        "first_bilateral_force_bearing_time_s": (
            float(first_contact_row["time_s"]) if first_contact_row else None
        ),
        "lift_path_sample_count": len(lift_rows),
        "first_airborne_lift_step": int(lift_rows[first_air_index]["step"]) if first_air_index is not None else None,
        "load_transfer_sample_count": len(load_rows),
        "airborne_lift_sample_count": len(airborne_rows),
        "first_fully_open_release_step": int(release_candidates[first_open_index]["step"]) if first_open_index is not None else None,
        "release_sample_count": len(release_rows),
        "bottle_com_target_world_m": list(target_com),
        "relative_slip_definition": "translation change of saved bottle-root-to-TCP vector in TCP coordinates from first simultaneous force-bearing pad contact; excludes contact-point migration caused by rotation",
        "phase_mapping": {
            "pregrasp_settle": "PREGRASP",
            "approach": "APPROACH",
            "close from first simultaneous force-bearing pad contact plus hold": "CONTACT",
            "lift path while bottle-table contact is logged": "LOAD_TRANSFER",
            "lift path beginning at first logged table-contact-free sample": "LIFT_50MM",
            "mounted_lift_hold_1s": "HOLD",
            "loaded_transfer": "TRANSFER",
            "lower_to_target": "PLACE",
            "release/open/retract beginning at first >=88.4 mm pad gap": "RELEASE",
            "free_physics_settle": "SETTLE",
        },
    }
    return frames, mapping


def contact_summary(rows: list[dict], contacts: list[dict], contact_by_step: dict[int, dict], timestep_s: float) -> dict:
    phase_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        phase_rows[row["phase"]].append(row)
    force_peaks = {"left_pad": 0.0, "right_pad": 0.0}
    pair_counts = Counter()
    nonpad = Counter()
    for row in contacts:
        first, second = str(row.get("body1", "")), str(row.get("body2", ""))
        other = second if first == "m0_bottle" else first
        pair_counts[other] += 1
        if other in {"rq_left_pad", "rq_right_pad"}:
            role = "left_pad" if other == "rq_left_pad" else "right_pad"
            force_peaks[role] = max(force_peaks[role], float(row.get("normal_force_n") or 0.0))
        elif other != "m0_table":
            nonpad[other] += 1

    stage_stats = {}
    for phase, group in phase_rows.items():
        step_events = [contact_by_step.get(sample_step_index(row, timestep_s), {}) for row in group]
        table = [float(row["table_normal_force_n"]) for row in group]
        z = [float(row["bottle_com_z_m"]) for row in group]
        linear_speed = [float(row["bottle_linear_speed_m_s"]) for row in group]
        angular_speed = [float(row["bottle_angular_speed_rad_s"]) for row in group]
        pad_fraction = (sum(bool(event.get("pad_active")) for event in step_events) / len(group)) if group else 0.0
        stage_stats[phase] = {
            "samples": len(group), "start_time_s": float(group[0]["time_s"]), "end_time_s": float(group[-1]["time_s"]),
            "bottle_com_z_min_m": min(z), "bottle_com_z_max_m": max(z),
            "mean_table_normal_force_n": sum(table) / len(table), "min_table_normal_force_n": min(table),
            "max_table_normal_force_n": max(table), "any_pad_contact_fraction": pad_fraction,
            "max_linear_speed_m_s": max(linear_speed), "terminal_linear_speed_m_s": linear_speed[-1],
            "max_angular_speed_rad_s": max(angular_speed), "terminal_angular_speed_rad_s": angular_speed[-1],
        }
    return {
        "contact_trace_rows_involving_bottle": len(contacts),
        "contact_row_counts_by_other_body": dict(pair_counts),
        "nonpad_robot_bottle_contact_row_counts": dict(nonpad),
        "peak_recorded_pad_normal_force_n": force_peaks,
        "left_hand_bottle_contact_observed": any(event.get("left_hand_contact", False) for event in contact_by_step.values()),
        "table_contact_sample_count": sum(bool(event.get("table_contact", False)) for event in contact_by_step.values()),
        "phase_statistics": stage_stats,
    }


def audit(history_path: Path, physics_trace_path: Path, contact_trace_path: Path,
          static_result_path: Path) -> dict:
    history_bytes = history_path.read_bytes()
    static_bytes = static_result_path.read_bytes()
    history = json.loads(history_bytes)
    static = json.loads(static_bytes)
    if history.get("status") != "PASS":
        raise ValueError("expected the saved historical episode's own status to be PASS")
    if static.get("schema") != "robotsim.issue46.robotiq_static_approach_qualification.v1":
        raise ValueError("static qualification has an unexpected schema")
    identity = history["identity"]
    model_sha = str(history["adapter"]["model_xml_sha256"])
    if static["input_identity"]["diagnostic_model_sha256"] != model_sha:
        raise ValueError("historical and static evidence refer to different diagnostic models")
    if static["input_identity"]["historical_result_sha256"] != hashlib.sha256(history_bytes).hexdigest():
        raise ValueError("static result does not reference the exact historical result file")

    with physics_trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("historical physics trace is empty")
    timestep_s = infer_timestep_s(rows)
    contact_by_step, contacts = load_contact_events(contact_trace_path, timestep_s)
    frames, mapping = build_episode_frames(rows, contact_by_step, history, model_sha, timestep_s)
    criteria = EpisodeCriteria(
        required_contact_groups=(("left_pad",), ("right_pad",)),
        minimum_contact_fraction=REQUIRED_CONTACT_FRACTION,
        minimum_tool_upward_support_n=MINIMUM_SUPPORT_N,
        minimum_table_unloading_n=MINIMUM_TABLE_UNLOADING_N,
        maximum_relative_slip_m=MAXIMUM_RELATIVE_TRANSLATION_M,
        minimum_approach_clearance_m=0.002,
        minimum_lift_m=0.05,
        minimum_hold_duration_s=1.0,
        place_target_xy_world_m=(mapping["bottle_com_target_world_m"][0], mapping["bottle_com_target_world_m"][1]),
        maximum_place_xy_error_m=0.020,
        maximum_upright_error_rad=math.radians(5.0),
        minimum_release_opening_m=MINIMUM_RELEASE_OPENING_M,
        minimum_settle_duration_s=3.0,
        maximum_settle_linear_speed_m_s=0.020,
        maximum_settle_angular_speed_rad_s=0.050,
        maximum_settle_position_drift_m=0.020,
    )
    gate = evaluate_episode_evidence(frames, criteria)
    blocker_counts = Counter(item.split(":", 1)[0] for item in gate.blockers)
    arm_limits = source_arm_trace_summary(rows, identity["urdf_joint_limits"], timestep_s)
    c_by_phase = contact_summary(rows, contacts, contact_by_step, timestep_s)
    frame_steps = [frame.physics_step_index for frame in frames]
    step_anomalies = [
        {"index": index, "previous_step": earlier, "step": later}
        for index, (earlier, later) in enumerate(zip(frame_steps, frame_steps[1:]), start=1)
        if later <= earlier
    ]
    phase_ranges = {}
    for phase in PHASE_ORDER:
        phase_frames = [frame for frame in frames if frame.phase is phase]
        if phase_frames:
            phase_ranges[phase.value] = {
                "count": len(phase_frames), "first_step": phase_frames[0].physics_step_index,
                "last_step": phase_frames[-1].physics_step_index,
            }
    force_local_present = bool(contacts) and all("force_local_n" in row for row in contacts)
    raw_phase_rows = defaultdict(list)
    for row in rows:
        raw_phase_rows[row["phase"]].append(row)
    raw_lift_rows = raw_phase_rows["lift_080mm_cartesian_path"]
    load_count = mapping["load_transfer_sample_count"]
    load_prefix = raw_lift_rows[:load_count]
    airborne_rows = raw_lift_rows[load_count:]
    bilateral_fraction = lambda sample_rows: (
        sum({"left_pad", "right_pad"}.issubset(contact_by_step.get(sample_step_index(row, timestep_s), {}).get("force_roles", set()))
            for row in sample_rows) / len(sample_rows) if sample_rows else None
    )
    hold_rows = raw_phase_rows["hold"]
    first_bilateral = next((row for row in rows if sample_step_index(row, timestep_s)
                            == mapping["first_bilateral_force_bearing_step"]), None)
    contact_base_z = float(first_bilateral["bottle_com_z_m"]) if first_bilateral else None
    lift_peak_z = max((float(row["bottle_com_z_m"]) for row in airborne_rows), default=None)
    mean_table = lambda sample_rows: (
        sum(float(row["table_normal_force_n"]) for row in sample_rows) / len(sample_rows) if sample_rows else None
    )
    semantic_metrics = {
        "contact_hold": {
            "sample_count": len(hold_rows), "bilateral_force_bearing_fraction": bilateral_fraction(hold_rows),
            "mean_table_normal_force_n": mean_table(hold_rows),
            "max_pad_normal_force_sum_n": max((float(row["pad_normal_force_sum_n"]) for row in hold_rows), default=None),
        },
        "table_unloading_prefix": {
            "sample_count": len(load_prefix), "mean_table_normal_force_n": mean_table(load_prefix),
            "mean_table_reaction_reduction_from_contact_hold_n": (
                mean_table(hold_rows) - mean_table(load_prefix)
                if hold_rows and load_prefix else None
            ),
            "bilateral_force_bearing_fraction": bilateral_fraction(load_prefix),
            "world_upward_hand_support_n": None,
        },
        "airborne_lift": {
            "sample_count": len(airborne_rows),
            "bottle_com_rise_from_first_bilateral_contact_m": (
                lift_peak_z - contact_base_z if lift_peak_z is not None and contact_base_z is not None else None
            ),
            "table_contact_free_samples": sum(
                not contact_by_step.get(sample_step_index(row, timestep_s), {}).get("table_contact", False)
                for row in airborne_rows
            ),
            "bilateral_force_bearing_fraction": bilateral_fraction(airborne_rows),
            "world_upward_hand_support_n": None,
        },
        "recorded_pad_force_limits": {
            "peak_pad_normal_force_sum_n_all_episode": max(float(row["pad_normal_force_sum_n"]) for row in rows),
            "note": "normal-force magnitude sum only; not a world vertical support measurement",
        },
    }
    head_torso = history.get("head_torso_clearance_audit", {})
    head_best = head_torso.get("best_sample", {})
    result = {
        "schema": "robotsim.issue46.robotiq_historical_episode_gate_audit.v1",
        "classification": "READ_ONLY_HISTORICAL_TRACE_GATE_REPLAY_NO_SIMULATION",
        "run_id": str(uuid.uuid4()),
        "physics_steps_executed": 0,
        "historical_physics_samples_evaluated": len(rows),
        "episode_gate": {
            "status": gate.status,
            "stage_status": dict(gate.stage_status),
            "last_physics_step_index": gate.last_physics_step_index,
            "blocker_count": len(gate.blockers),
            "blocker_counts_by_code": dict(blocker_counts),
            "blocker_examples": list(gate.blockers[:40]),
            "criteria": {
                "required_contact_groups": [["left_pad"], ["right_pad"]],
                "minimum_contact_fraction": REQUIRED_CONTACT_FRACTION,
                "minimum_tool_upward_support_n": MINIMUM_SUPPORT_N,
                "minimum_table_unloading_n": MINIMUM_TABLE_UNLOADING_N,
                "maximum_relative_translation_m": MAXIMUM_RELATIVE_TRANSLATION_M,
                "minimum_approach_clearance_m": 0.002,
                "minimum_lift_m": 0.05,
                "minimum_hold_duration_s": 1.0,
                "minimum_release_opening_m": MINIMUM_RELEASE_OPENING_M,
                "minimum_settle_duration_s": 3.0,
            },
        },
        "trace_mapping": mapping,
        "trace_sequence_validation": {
            "frame_count": len(frames), "inferred_physics_timestep_s": timestep_s,
            "raw_step_duplicate_count": sum(1 for earlier, later in zip(rows, rows[1:])
                                             if int(earlier["step"]) == int(later["step"])),
            "raw_step_index_reconstructed_from_time": True,
            "nonpositive_step_count": sum(step <= 0 for step in frame_steps),
            "nonmonotonic_step_count": len(step_anomalies), "nonmonotonic_examples": step_anomalies[:10],
            "phase_step_ranges": phase_ranges,
        },
        "historical_measured_summary": c_by_phase,
        "semantic_stage_metrics": semantic_metrics,
        "partial_source_arm_validation": arm_limits,
        "evidence_gaps": {
            "dynamic_approach_clearance_per_sample": "MISSING; historical rollout CSV/contact trace has no geometric clearance measurement. New static candidate result is reported separately and is not retroactively applied to the old rollout.",
            "world_upward_hand_support": "MISSING; contact trace stores local force scalars but not contact-frame orientation/world force vectors, so pad normal sum is not vertical support.",
            "all_source_joint_positions_and_velocities": "MISSING; CSV contains the seven right-arm qpos/qvel vectors and two gripper driver positions, not all hand/coupler/follower qpos/qvel.",
            "all_actuator_limits_and_efforts": "MISSING; saved trace has a gripper actuator-force scalar and arm tracking error, not per-actuator control/force for the complete model.",
            "follower_constraint_residuals": "MISSING; follower and coupler states/residuals were not recorded per sample.",
            "contact_world_normals_and_tangential_vectors": "MISSING; contact frame orientation is absent, so force direction and per-contact world tangential load cannot be reconstructed.",
        },
        "historical_exception_audit": {
            **static["historical_exception_audit"],
            "head_torso_grid_summary": {
                "sample_count": head_torso.get("sample_count"),
                "geometrically_clear_samples": head_torso.get("geometrically_clear_samples"),
                "best_sample": {key: head_best.get(key) for key in
                                ("head_pitch_rad", "head_yaw_rad", "minimum_pair_distance_m")},
                "classification": "diagnostic-only exclusion; strict full-X2 acceptance unresolved",
            },
            "source_faithful_coupler_status": history["adapter"].get("source_faithful_coupler_baseline_status"),
            "coupler_margin_simulation_derived_rad": history["adapter"].get("coupler_limit_activation_margin_simulation_derived_rad"),
            "official_source_head_torso_neutral_geometry_distances": history["adapter"].get("official_source_head_torso_neutral_geometry_distances"),
        },
        "static_approach_qualification": {
            "status": static["status"],
            "run_id": static["run_id"],
            "gates": static["gates"],
            "baseline_20mm_swept": {
                group: {
                    "pass": value.get("pass"),
                    "minimum_swept_clearance_lower_bound_m": value.get("minimum_swept_clearance_lower_bound_m"),
                    "worst_pair": value.get("worst_pairs"),
                }
                for group, value in static["baseline_20mm"]["swept"].items()
            },
            "candidate_25mm_retreat_swept": {
                group: {
                    "pass": value.get("pass"),
                    "minimum_swept_clearance_lower_bound_m": value.get("minimum_swept_clearance_lower_bound_m"),
                    "worst_pair": value.get("worst_pairs"),
                }
                for group, value in static["candidate_25mm_retreat"]["swept"].items()
            },
            "candidate_internal_mechanism_clearance": static["candidate_25mm_retreat"]["internal_mechanism_clearance"]["worst_pair"],
            "candidate_right_arm_pregrasp_qpos_rad": static["candidate_25mm_retreat"]["ik"]["joint_positions_rad"],
            "candidate_right_arm_grasp_qpos_rad": static["baseline_20mm"]["approach_qpos_rad"],
        },
        "reported_0_005701_mm_clearance_reconciliation": {
            "exact_witness_found": static.get("legacy_minimum_reconciliation") is not None,
            "witness": static.get("legacy_minimum_reconciliation"),
            "additional_nearest_aabb_sample_check": {
                "sample": static["baseline_20mm"]["source_spring_to_bottle_checks"]["closest_sample"],
                "source_stl_distance_bound": static["baseline_20mm"]["source_spring_to_bottle_checks"]["source_triangle_distance_at_closest_hull_sample"],
            },
            "interpretation": "The reported 0.005701 mm is an AABB lower bound at sample 20, not the physical spring/bottle distance. At the identical model and qpos, the compiled hull projection lower bound is about 29.44 mm, mj_geomDistance is about 29.45 mm, and source STL is bounded above 20.03 mm. The actual path still has a separate near-2 mm limiting pair elsewhere.",
        },
        "input_identity": {
            "historical_result_path": str(history_path), "historical_result_sha256": hashlib.sha256(history_bytes).hexdigest(),
            "physics_trace_path": str(physics_trace_path), "physics_trace_sha256": sha256(physics_trace_path),
            "contact_trace_path": str(contact_trace_path), "contact_trace_sha256": sha256(contact_trace_path),
            "static_result_path": str(static_result_path), "static_result_sha256": hashlib.sha256(static_bytes).hexdigest(),
            "model_id": f"X2+Robotiq2F85:{history['adapter']['model_configuration']}",
            "model_sha256": model_sha,
            "x2_commit": identity["x2_checkout"]["head"],
            "menagerie_commit": identity["menagerie_checkout"]["head"],
            "mujoco_version": identity["mujoco_native"],
            "mujoco_native_library_sha256": identity["native_library_sha256"],
            "force_local_contact_data_present": force_local_present,
        },
        "limits": {
            "bottle_qpos_writes_after_start_max": max(int(row["qpos_writes_after_rollout_start"]) for row in rows),
            "bottle_freejoint": bool(history["physical_integrity"]["bottle_dynamic_freejoint"]),
            "bottle_weld_or_equality": bool(history["physical_integrity"]["bottle_weld_or_equality"]),
            "bottle_mocap_or_follow": bool(history["physical_integrity"]["bottle_mocap_or_follow"]),
            "hidden_support": bool(history["physical_integrity"]["hidden_support"]),
            "left_hand_assistance_claim": bool(history["physical_integrity"]["left_hand_assistance"]),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-result", required=True)
    parser.add_argument("--physics-trace", required=True)
    parser.add_argument("--contact-trace", required=True)
    parser.add_argument("--static-result", required=True)
    parser.add_argument("--output-parent", default=str(REVIEWS / "issue-46"))
    args = parser.parse_args()
    inputs = {
        "historical result": Path(args.historical_result),
        "historical physics trace": Path(args.physics_trace),
        "historical contact trace": Path(args.contact_trace),
        "static qualification result": Path(args.static_result),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    result = audit(inputs["historical result"], inputs["historical physics trace"],
                   inputs["historical contact trace"], inputs["static qualification result"])
    output_parent = pg.validate_output_path(args.output_parent, authorized_roots=(str(REVIEWS),),
                                            require_exists=True, expected_kind="directory", environment="wsl")
    output = pg.create_unique_output_directory(output_parent.wsl_path, prefix="robotiq-historical-episode-gate-audit",
                                               authorized_roots=(str(REVIEWS),), environment="wsl")
    result["run_id"] = str(uuid.uuid4())
    result["runner_path"] = str(Path(__file__).resolve())
    result["runner_sha256"] = sha256(Path(__file__).resolve())
    result["episode_gate_source_sha256"] = sha256(Path(__file__).with_name("episode_gate.py"))
    result["output"] = output.as_dict()
    report = [
        "# Historical X2 + Robotiq episode gate audit",
        "",
        f"Audit run ID: {result['run_id']}",
        f"Historical result status (original runner): {json.loads(inputs['historical result'].read_text())['status']}",
        f"Current episode-gate status: {result['episode_gate']['status']}",
        "Physics steps executed by this audit: 0",
        f"Saved physics samples reviewed: {result['historical_physics_samples_evaluated']}",
        "",
        "## Stage results",
        "",
    ]
    report.extend(f"- `{phase}`: `{status}`" for phase, status in result["episode_gate"]["stage_status"].items())
    static_candidate = result["static_approach_qualification"]
    static_internal = static_candidate["candidate_internal_mechanism_clearance"]
    candidate_sweep = static_candidate["candidate_25mm_retreat_swept"]
    exception = result["historical_exception_audit"]
    sequence = result["trace_sequence_validation"]
    semantic = result["semantic_stage_metrics"]
    legacy_witness = result["reported_0_005701_mm_clearance_reconciliation"]["witness"]
    report.extend([
        "",
        "The historical runner reported its own PASS, including a 56.30 mm peak COM rise, 100% bilateral pad-contact fraction during lift, transfer, release and free settle. This independent stricter gate replay is BLOCKED because the saved traces omit world-projected hand support, per-frame approach clearance, full-model joint/actuator limits and follower residuals. The missing fields remain unknown; they are not inferred from pad-normal sums or commands.",
        "",
        f"The raw integer step column repeats {sequence['raw_step_duplicate_count']} times while timestamps advance at {sequence['inferred_physics_timestep_s']:.9f} s. The audit reconstructed unique monotonic post-step indices from time; no physical samples were removed.",
        "",
        f"Contact hold: {semantic['contact_hold']['bilateral_force_bearing_fraction'] * 100:.1f}% bilateral force-bearing samples, mean table reaction {semantic['contact_hold']['mean_table_normal_force_n']:.3f} N. Table-supported load prefix: {semantic['table_unloading_prefix']['sample_count']} samples, mean table reaction {semantic['table_unloading_prefix']['mean_table_normal_force_n']:.3f} N, mean reduction {semantic['table_unloading_prefix']['mean_table_reaction_reduction_from_contact_hold_n']:.3f} N. Upward hand support remains unmeasured.",
        f"Historical airborne section reached {semantic['airborne_lift']['bottle_com_rise_from_first_bilateral_contact_m'] * 1000:.2f} mm COM rise over {semantic['airborne_lift']['table_contact_free_samples']} table-contact-free samples; this is not a new run or a measured upward-force vector.",
        "",
        "## Static candidate",
        "",
        f"The separately evaluated 25 mm retreat path is `{static_candidate['status']}`. It has 101 linear arm-space configurations, passes source position limits, preserves the historical final grasp vector, and changes only retreat distance from 20 to 25 mm. It is a static witness, not a timed or dynamically tracked trajectory.",
        "",
        f"Swept lower bounds: all robot/bottle {candidate_sweep['robot_bottle']['minimum_swept_clearance_lower_bound_m']['robot_bottle'] * 1000:.6f} mm; non-pad robot/bottle {candidate_sweep['robot_bottle_nonpad']['minimum_swept_clearance_lower_bound_m']['robot_bottle_nonpad'] * 1000:.6f} mm; robot/table {candidate_sweep['robot_table']['minimum_swept_clearance_lower_bound_m']['robot_table'] * 1000:.6f} mm; external robot/self {candidate_sweep['robot_self']['minimum_swept_clearance_lower_bound_m']['robot_self'] * 1000:.6f} mm.",
        f"The strict internal spring/pad margin is blocked at {static_internal['minimum_clearance_lower_bound_m'] * 1000:.6f} mm for `{static_internal['body1']}/{static_internal['geom1']}` to `{static_internal['body2']}/{static_internal['geom2']}`. This positive gap is invariant over all 101 arm states and no active contact was seen; an arm-only pose/path change cannot increase it.",
        "",
        f"The 0.005701 mm value is confirmed in prior static run `{legacy_witness['prior_static_run_id']}` at sample {legacy_witness['sample_index']} (20% path): `{legacy_witness['body_pair'][0]}/{legacy_witness['geom_pair'][0]}` to `{legacy_witness['body_pair'][1]}/{legacy_witness['geom_pair'][1]}`. It is the conservative AABB lower bound. At the identical qpos, independent compiled-hull SAT gives {legacy_witness['independent_current_compiled_proxy_measurement']['clearance_lower_bound_mm']:.3f} mm, `mj_geomDistance` gives {legacy_witness['independent_current_compiled_proxy_measurement']['mujoco_geom_distance_m'] * 1000:.3f} mm, and registered source STL is bounded [{legacy_witness['source_stl_surface_distance_bound']['lower_bound_m'] * 1000:.3f}, {legacy_witness['source_stl_surface_distance_bound']['upper_bound_m'] * 1000:.3f}] mm. Therefore 0.005701 mm is not the actual source-surface clearance at that sample. The overall path still has a different limiting pad/bottle clearance only just above 2 mm.",
        "",
        f"External clearances are only micrometres above 2 mm. Historical maximum tracking errors were right wrist yaw {result['partial_source_arm_validation']['max_abs_tracking_error_rad_by_joint']['right_wrist_yaw_joint']:.6f} rad and right shoulder pitch {result['partial_source_arm_validation']['max_abs_tracking_error_rad_by_joint']['right_shoulder_pitch_joint']:.6f} rad. The new candidate sweep does not include that dynamic tracking error, so it is not a robust execution margin.",
        "",
        f"Historical exceptions remain diagnostic-only: head/torso excluded, {exception['head_torso_grid_summary']['geometrically_clear_samples']}/{exception['head_torso_grid_summary']['sample_count']} sampled poses clear (best overlap {exception['head_torso_grid_summary']['best_sample']['minimum_pair_distance_m'] * 1000:.3f} mm); Robotiq adapter mount xyz `{exception['wrist_mount']['translation_m']} m`, rpy `{exception['wrist_mount']['rpy']}` lacks manufacturer adapter provenance; +{exception['coupler_margin_rad']:.3f} rad coupler margin is simulation-derived and the source-faithful baseline reported `{exception['source_faithful_coupler_status']}`.",
        "",
        "## Recovery of force direction",
        "",
        "`contact_trace.jsonl` records contact-local force scalars and normal-force magnitude, but does not record the contact-frame basis. World +Z support and per-contact world tangential force therefore cannot be reconstructed. The aggregate pad-normal-force column is not treated as vertical hand support.",
        "",
        "## Integrity",
        "",
        f"Bottle qpos writes after rollout start: {result['limits']['bottle_qpos_writes_after_start_max']}",
        f"Bottle freejoint: {result['limits']['bottle_freejoint']}; weld/equality: {result['limits']['bottle_weld_or_equality']}; mocap/follow: {result['limits']['bottle_mocap_or_follow']}; hidden support: {result['limits']['hidden_support']}; left-hand assistance claim: {result['limits']['left_hand_assistance_claim']}.",
        "",
        "## Reproduction",
        "",
        "See `REPRODUCE.md`. This command reads saved evidence only and calls no MuJoCo stepping function.",
        "",
    ])
    reproduce = [
        f"cd {ROOT}",
        "PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_historical_robotiq_episode.py \\",
        f"  --historical-result {inputs['historical result']} \\",
        f"  --physics-trace {inputs['historical physics trace']} \\",
        f"  --contact-trace {inputs['historical contact trace']} \\",
        f"  --static-result {inputs['static qualification result']}",
    ]
    blocker_lines = [f"{key}: {count}" for key, count in result["episode_gate"]["blocker_counts_by_code"].items()]
    run_log = [
        f"audit_run_id={result['run_id']}", f"gate_status={result['episode_gate']['status']}",
        "physics_steps_executed=0", f"frame_count={result['historical_physics_samples_evaluated']}",
        "blockers:", *blocker_lines,
    ]
    pg.write_new_text(output.native_path / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPORT.md", "\n".join(report), authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPRODUCE.md", "\n".join(reproduce) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "run.log", "\n".join(run_log) + "\n",
                      authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(output.native_path, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({
        "audit_run_id": result["run_id"], "gate_status": result["episode_gate"]["status"],
        "physics_steps_executed": 0, "frame_count": result["historical_physics_samples_evaluated"],
        "stage_status": result["episode_gate"]["stage_status"],
        "blocker_counts_by_code": result["episode_gate"]["blocker_counts_by_code"],
        "output": output.as_dict(), "manifest_count": len(manifest["artifacts"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
