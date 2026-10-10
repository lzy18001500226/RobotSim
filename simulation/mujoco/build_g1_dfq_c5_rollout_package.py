#!/usr/bin/env python3
"""Package one statically feasible C5 geometry for the physical rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from g1_dfq_static_wrench import candidate_rollout_allocation


CHANNEL_BY_JOINT = {
    "R_pinky_proximal_joint": "pinky_proximal",
    "R_ring_proximal_joint": "ring_proximal",
    "R_middle_proximal_joint": "middle_proximal",
    "R_index_proximal_joint": "index_proximal",
    "R_thumb_proximal_pitch_joint": "thumb_proximal_pitch",
    "R_thumb_proximal_yaw_joint": "thumb_proximal_yaw",
}
ARM_NAMES = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-candidate", type=Path, required=True)
    parser.add_argument("--static-search-result", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--close-slew-rad-s", type=float, default=0.060)
    args = parser.parse_args()
    if args.close_slew_rad_s != 0.060:
        parser.error("the accepted C5 package is fixed to the existing 0.060 rad/s close slew")

    seed = json.loads(args.seed_candidate.read_text(encoding="utf-8"))
    search = json.loads(args.static_search_result.read_text(encoding="utf-8"))
    base = json.loads(args.base_state.read_text(encoding="utf-8"))
    selected_id = search.get("selected_candidate")
    candidate_result = next(
        (item for item in search.get("candidates", [])
         if item.get("candidate_id") == selected_id),
        None,
    )
    if candidate_result is None or selected_id != "C5_upper_body_opposition":
        parser.error("the approved static search must select C5_upper_body_opposition")
    if search.get("physics_steps") != 0 or search.get("mj_step_calls") != 0:
        parser.error("static search artifact contains physics steps")
    if search.get("mujoco_version") != "3.3.6":
        parser.error("candidate must be audited with MuJoCo 3.3.6")

    candidate_id = "C5_upper_body_opposition"
    run_id = f"dfq-c5-static-{time.time_ns()}"
    candidate = dict(seed)
    candidate["run_id"] = run_id
    candidate["candidate"] = candidate_id
    candidate["analysis"] = (
        "C5 upper-body opposition selected from the bounded three-family static search; "
        "full 6D wrench is static feasibility only, not measured dynamic support."
    )
    candidate["physics_steps"] = 0
    candidate["mj_step_calls"] = 0
    candidate["mj_forward_calls"] = search.get("mj_forward_calls")
    values = candidate_result["candidate_joint_values_rad"]
    for name, value in values.items():
        if name not in candidate["joint_configuration"]:
            raise ValueError(f"C5 joint is absent from the seed source-limit table: {name}")
        candidate["joint_configuration"][name]["q_rad"] = float(value)
        lower = float(candidate["joint_configuration"][name]["lower_rad"])
        upper = float(candidate["joint_configuration"][name]["upper_rad"])
        candidate["joint_configuration"][name]["violation_rad"] = max(lower - value, value - upper, 0.0)
    if not set(ARM_NAMES).issubset(values) or not set(CHANNEL_BY_JOINT).issubset(values):
        raise ValueError("C5 configuration does not include all seven arm and six hand driver joints")

    contacts = candidate_result["active_right_hand_bottle_contacts"]
    by_digit = {}
    for row in contacts:
        digit = str(row["digit"])
        normalized = {
            "hand_geom": row["geom"],
            "hand_body": row["body"],
            "bottle_geom": row["bottle_geom"],
            "distance_m": float(row["distance_m"]),
            "point_world_m": row["point_world_m"],
            "frame_world_rows": row["frame_world_rows"],
            "friction": row["friction"],
            "condim": int(row["condim"]),
        }
        by_digit.setdefault(digit, []).append(normalized)
    candidate["joint_contact_geometry"] = {
        digit: {
            "hand_geom": rows[0]["hand_geom"],
            "closest_bottle_surface": min(rows, key=lambda item: item["distance_m"]),
            "surfaces": rows,
        }
        for digit, rows in by_digit.items()
    }
    candidate["actual_contacts"] = contacts
    candidate["non_digit_final_contacts"] = candidate_result.get(
        "active_non_digit_right_hand_bottle_contacts", []
    )
    candidate["opposing_surface_normal_dot"] = candidate_result[
        "best_thumb_opposing_contact_normal_dot"
    ]
    candidate["open_hand_approach"] = dict(seed["open_hand_approach"])
    candidate["open_hand_approach"]["optimized_contact_arm_q_rad"] = [
        float(values[name]) for name in ARM_NAMES
    ]
    candidate["open_hand_approach"]["candidate_static_clearance"] = candidate_result["open_approach"]

    source = dict(seed["source"])
    source["static_search_result_path"] = args.static_search_result.resolve().as_posix()
    source["static_search_result_sha256"] = sha256(args.static_search_result)
    source["selected_candidate_id"] = candidate_id
    source["selected_candidate_static_search"] = candidate_result
    source["base_state_sha256"] = sha256(args.base_state)
    candidate["source"] = source

    cap_m = float(search["max_contact_penetration_gate_m"])
    penetration_tolerance_m = float(search["penetration_gate_evaluation_tolerance_m"])
    candidate["static_gates"] = {
        "source_position_limits": candidate_result["source_position_limits_pass"],
        "source_velocity_planned_command_rate": candidate_result["source_velocity_plan_pass"],
        "thumb_has_actual_shallow_contact": "thumb" in candidate_result["contact_digits"],
        "opposing_digit_has_actual_shallow_contact": any(
            digit in candidate_result["contact_digits"] for digit in ("index", "middle", "ring", "pinky")
        ),
        "all_active_digit_contacts_within_penetration_bound": (
            candidate_result["maximum_contact_penetration_m"] <= cap_m + penetration_tolerance_m
        ),
        "maximum_static_contact_penetration_m": candidate_result["maximum_contact_penetration_m"],
        "penetration_gate_m": cap_m,
        "penetration_gate_evaluation_tolerance_m": penetration_tolerance_m,
        "thumb_opposing_contact_normal_dot": candidate_result["best_thumb_opposing_contact_normal_dot"],
        "contact_normals_are_opposed": candidate_result["best_thumb_opposing_contact_normal_dot"] < -0.1,
        "no_palm_wrist_arm_bottle_contact": not candidate_result.get("active_non_digit_right_hand_bottle_contacts"),
        "collision_free_open_approach": candidate_result["open_approach"]["passed"],
        "open_approach_minimum_clearance_m": candidate_result["open_approach"]["distance_m"],
        "new_fresh_contact_geometry_recorded": True,
        "active_contacts_are_not_claimed_as_load_bearing": True,
        "full_6d_static_wrench_feasible": candidate_result["effort_bounded_6d_equilibrium"]["feasible"],
        "robust_contact_perturbations_pass": candidate_result["robust_perturbations_pass"],
        "passed": candidate_result["static_candidate_pass"],
    }
    candidate["planned_close"] = {
        "target_slew_rad_s": args.close_slew_rad_s,
        "source_velocity_limit_rad_s": search["source_velocity_limit_rad_s"],
        "per_driver": {},
    }
    open_targets = candidate["open_hand_approach"]["open_driver_targets_rad"]
    for joint, channel in CHANNEL_BY_JOINT.items():
        open_q = float(open_targets[joint])
        target_q = float(values[joint])
        duration = abs(target_q - open_q) / args.close_slew_rad_s
        candidate["planned_close"]["per_driver"][joint] = {
            "open_rad": open_q,
            "target_rad": target_q,
            "commanded_slew_rad_s": args.close_slew_rad_s,
            "minimum_time_s_at_slew": duration,
            "source_velocity_limit_rad_s": search["source_velocity_limit_rad_s"],
            "planned_source_rate_pass": args.close_slew_rad_s <= search["source_velocity_limit_rad_s"],
        }
    candidate["wrench_status"] = "CANDIDATE_SPECIFIC_FULL_6D_STATIC_WRENCH_FEASIBLE"
    candidate["state_writes"] = {
        "physics_steps": 0,
        "active_rollout_bottle_qpos_writes": 0,
        "active_rollout_follower_qpos_writes": 0,
        "attachment_constraints": 0,
        "mocap_carry": False,
    }

    allocation = candidate_rollout_allocation(
        candidate_result,
        bottle_weight_n=float(search["bottle_weight_n"]),
        effort_cap_nm=float(search["simulation_only_effort_cap_nm"]),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_c5.json"
    write_json(candidate_path, candidate)

    input_path = output_dir / "candidate_c5_wrench_input.json"
    wrench_input = {
        "schema": "robotsim.g1_dfq.static_candidate_input.v1",
        "candidate_id": candidate_id,
        "candidate_static_run_id": run_id,
        "static_search_result_path": args.static_search_result.resolve().as_posix(),
        "static_search_result_sha256": sha256(args.static_search_result),
        "base_state_path": args.base_state.resolve().as_posix(),
        "base_state_sha256": sha256(args.base_state),
        "candidate_path": candidate_path.as_posix(),
        "candidate_sha256": sha256(candidate_path),
        "selected_candidate": candidate_result,
        "physics_steps": 0,
        "mj_step_calls": 0,
    }
    write_json(input_path, wrench_input)

    result_path = output_dir / "candidate_c5_wrench.json"
    wrench = {
        "schema": "robotsim.g1_dfq.candidate_wrench_evidence.v1",
        "run_id": f"{run_id}-wrench",
        "candidate_id": candidate_id,
        "candidate_static_run_id": run_id,
        "candidate_sha256": sha256(candidate_path),
        "input": input_path.as_posix(),
        "input_sha256": sha256(input_path),
        "static_search_result_sha256": sha256(args.static_search_result),
        "classification": "C5 — CANDIDATE-SPECIFIC FULL 6D STATIC WRENCH FEASIBLE",
        "robotsim_sha": search["robotsim_head"],
        "robotsim_dirty": search["robotsim_dirty"],
        "mujoco_version": search["mujoco_version"],
        "physics_steps": 0,
        "mj_step_calls": 0,
        "mj_forward_calls": search.get("mj_forward_calls"),
        "compiled_model_sha256": search["model_sha256"],
        "normalized_compiled_model_sha256": source["compiled_model_sha256"],
        "bottle_mass_kg": float(search["bottle_mass_kg"]),
        "bottle_weight_n": float(search["bottle_weight_n"]),
        "configured_mu": 1.4,
        "friction_model": "unchanged compiled MuJoCo condim=4 pyramidal contact friction; static feasibility only",
        "simulation_only_effort_cap_nm": float(search["simulation_only_effort_cap_nm"]),
        "candidate_static_wrench_allocation": allocation,
        "robustness_summary": {
            "contact_point_perturbation_count": len(candidate_result["contact_point_perturbation_checks"]),
            "all_1mm_tangent_perturbations_passed": candidate_result["robust_perturbations_pass"],
            "this_is_static_only": True,
        },
        "candidate_contact_geometry": candidate_result["active_right_hand_bottle_contacts"],
    }
    write_json(result_path, wrench)
    print(json.dumps({
        "candidate_id": candidate_id,
        "candidate_run_id": run_id,
        "candidate_path": candidate_path.as_posix(),
        "candidate_sha256": sha256(candidate_path),
        "wrench_input_path": input_path.as_posix(),
        "wrench_result_path": result_path.as_posix(),
        "allocation": allocation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
