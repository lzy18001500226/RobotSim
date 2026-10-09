"""Evidence gates for the G1 DFQ isolated physical-grasp runner."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.optimize import linprog


CHANNELS = (
    "pinky_proximal",
    "ring_proximal",
    "middle_proximal",
    "index_proximal",
    "thumb_proximal_pitch",
    "thumb_proximal_yaw",
)

GATE_NAMES = (
    "HAND_ACTUATION",
    "FINGER_BOTTLE_CONTACT",
    "CONTACT_FORCE_WITHIN_CAP",
    "UPWARD_HAND_SUPPORT",
    "TABLE_LOAD_TRANSFER",
    "AIRBORNE_1MM",
    "AIRBORNE_5MM",
    "PHYSICAL_LIFT_30MM",
    "HOLD_1S",
    "PHYSICAL_RELEASE",
    "PR_CI",
)

_RESULTANT_DIRECTIONS = np.asarray([
    np.asarray(vector, dtype=float) / np.linalg.norm(vector)
    for vector in product(range(-2, 3), repeat=3)
    if vector != (0, 0, 0)
], dtype=float)
_RESULTANT_COVER_FACTOR = 0.90


def effort_handoff_references(commanded_position_targets: dict[str, float]) -> dict[str, float]:
    """Preserve the final bounded position commands when entering effort mode."""
    missing = [channel for channel in CHANNELS if channel not in commanded_position_targets]
    if missing:
        raise ValueError(f"missing effort handoff targets for {missing}")
    references = {channel: float(commanded_position_targets[channel]) for channel in CHANNELS}
    if not np.all(np.isfinite(list(references.values()))):
        raise ValueError("effort handoff targets must be finite")
    return references


def minimum_safe_brake_scale(peak_at_scale, ceiling: float, *, iterations: int = 24):
    """Return the least braking scale meeting a speed ceiling on a monotone interval."""
    if float(peak_at_scale(0.0)) <= ceiling:
        return 0.0
    if float(peak_at_scale(1.0)) > ceiling:
        return None
    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if float(peak_at_scale(middle)) <= ceiling:
            high = middle
        else:
            low = middle
    return high


def progressive_load_transfer_summary(
    baseline_table_normal_n: float,
    baseline_table_sigma_n: float,
    baseline_hand_support_n: float,
    stage_rows: list[dict],
    bottle_weight_n: float,
    *,
    minimum_change_fraction: float = 0.01,
    maximum_noise_fraction: float = 0.10,
) -> dict:
    """Require measurable table unloading and matching upward hand support."""
    if not stage_rows or bottle_weight_n <= 0.0:
        return {"passed": False, "reason": "missing_stage_samples_or_invalid_weight"}
    table_values = np.asarray([
        float(row["table_normal_force_n"]) for row in stage_rows
    ], dtype=float)
    hand_values = np.asarray([
        float(row["hand_vertical_support_force_n"]) for row in stage_rows
    ], dtype=float)
    if not np.all(np.isfinite(table_values)) or not np.all(np.isfinite(hand_values)):
        return {"passed": False, "reason": "non_finite_load_transfer_measurement"}
    contact_windows_valid = all(bool(row.get("contact_window_valid", 1)) for row in stage_rows)
    left_hand_clear = all(not bool(row.get("left_hand_contact", False)) for row in stage_rows)
    table_mean = float(np.mean(table_values))
    hand_mean = float(np.mean(hand_values))
    table_drop = float(baseline_table_normal_n - table_mean)
    hand_rise = float(hand_mean - baseline_hand_support_n)
    minimum_detectable_change = max(
        3.0 * max(float(baseline_table_sigma_n), 0.0),
        minimum_change_fraction * bottle_weight_n,
    )
    noise_usable = minimum_detectable_change <= maximum_noise_fraction * bottle_weight_n
    passed = bool(
        noise_usable
        and contact_windows_valid
        and left_hand_clear
        and table_drop >= minimum_detectable_change
        and hand_rise >= minimum_detectable_change
    )
    return {
        "passed": passed,
        "reason": "measured_table_unload_matches_upward_hand_support"
        if passed else "load_transfer_below_noise_aware_threshold",
        "baseline_table_normal_force_n": float(baseline_table_normal_n),
        "baseline_table_sigma_n": float(baseline_table_sigma_n),
        "baseline_hand_vertical_support_force_n": float(baseline_hand_support_n),
        "stage_mean_table_normal_force_n": table_mean,
        "stage_mean_hand_vertical_support_force_n": hand_mean,
        "measured_table_unload_n": table_drop,
        "measured_hand_support_increase_n": hand_rise,
        "minimum_detectable_change_n": minimum_detectable_change,
        "maximum_allowed_baseline_noise_n": maximum_noise_fraction * bottle_weight_n,
        "contact_windows_valid": contact_windows_valid,
        "left_hand_clear": left_hand_clear,
        "sample_count": int(len(stage_rows)),
    }


def load_ready_summary(
    stage_rows: list[dict],
    bottle_weight_n: float,
    *,
    minimum_hand_support_fraction: float = 0.85,
    maximum_table_remainder_fraction: float = 0.02,
    force_balance_tolerance_fraction: float = 0.05,
    horizontal_force_tolerance_fraction: float = 0.10,
    torque_tolerance_nm: float = 0.08,
    maximum_friction_utilization: float = 0.90,
    maximum_qvel_rad_s: float = 0.5,
    maximum_effort_nm: float = 0.531,
    maximum_mimic_error_rad: float = 0.003,
) -> dict:
    """Validate measured, table-independent load support over a dwell window."""
    if not stage_rows or bottle_weight_n <= 0.0:
        return {"passed": False, "reason": "missing_stage_samples_or_invalid_weight"}
    table_force = np.asarray([float(row["table_normal_force_n"]) for row in stage_rows])
    hand_force = np.asarray([float(row["hand_vertical_support_force_n"]) for row in stage_rows])
    if not np.all(np.isfinite(table_force)) or not np.all(np.isfinite(hand_force)):
        return {"passed": False, "reason": "non_finite_load_ready_measurement"}
    table_mean = float(np.mean(table_force))
    hand_mean = float(np.mean(hand_force))
    force_balance = abs(hand_mean + table_mean - bottle_weight_n)
    max_horizontal = max(float(row["horizontal_hand_force_n"]) for row in stage_rows)
    max_torque = max(float(row["hand_torque_norm_nm"]) for row in stage_rows)
    max_friction = max(float(row["maximum_friction_utilization"]) for row in stage_rows)
    max_qvel = max(float(row["maximum_hand_qvel_rad_s"]) for row in stage_rows)
    max_effort = max(float(row["maximum_applied_driver_effort_nm"]) for row in stage_rows)
    max_mimic = max(float(row["maximum_mimic_error_rad"]) for row in stage_rows)
    all_airborne = all(not bool(row["table_supported"]) for row in stage_rows)
    all_contact_valid = all(bool(row["contact_window_valid"]) for row in stage_rows)
    left_clear = all(not bool(row["left_hand_contact"]) for row in stage_rows)
    checks = {
        "hand_support_fraction": hand_mean / bottle_weight_n >= minimum_hand_support_fraction,
        "table_remainder_fraction": table_mean / bottle_weight_n <= maximum_table_remainder_fraction,
        "table_geometry_clear": all_airborne,
        "force_balance": force_balance <= force_balance_tolerance_fraction * bottle_weight_n,
        "horizontal_force": max_horizontal <= horizontal_force_tolerance_fraction * bottle_weight_n,
        "wrench_torque": max_torque <= torque_tolerance_nm,
        "friction_utilization": max_friction <= maximum_friction_utilization,
        "source_velocity": max_qvel <= maximum_qvel_rad_s,
        "simulation_effort": max_effort <= maximum_effort_nm,
        "mimic_error": max_mimic <= maximum_mimic_error_rad,
        "force_bearing_contacts": all_contact_valid,
        "left_hand_clear": left_clear,
    }
    return {
        "passed": all(checks.values()),
        "reason": "measured_hand_supported_airborne_load_ready" if all(checks.values())
        else "load_ready_measurement_gate_failed",
        "checks": checks,
        "mean_hand_vertical_support_force_n": hand_mean,
        "mean_table_normal_force_n": table_mean,
        "hand_supported_fraction": hand_mean / bottle_weight_n,
        "remaining_table_support_fraction": table_mean / bottle_weight_n,
        "force_balance_error_n": force_balance,
        "maximum_horizontal_hand_force_n": max_horizontal,
        "maximum_hand_torque_norm_nm": max_torque,
        "maximum_friction_utilization": max_friction,
        "maximum_hand_qvel_rad_s": max_qvel,
        "maximum_applied_driver_effort_nm": max_effort,
        "maximum_mimic_error_rad": max_mimic,
        "sample_count": int(len(stage_rows)),
    }


def select_effort_feedforward(channel: str, static_efforts: dict, load_efforts: dict,
                              load_build_schedule: bool) -> float:
    """Use the measured-contact allocation only after the load phase starts."""
    source = load_efforts if load_build_schedule else static_efforts
    if channel not in source:
        raise KeyError(f"missing simulation effort allocation for {channel}")
    return float(source[channel])


def solve_live_contact_wrench_allocation(
    contact_bases_world: list[np.ndarray],
    contact_positions_world: list[np.ndarray],
    contact_friction: list[tuple[float, float]],
    contact_driver_torque_maps: list[np.ndarray],
    bottle_com_world: np.ndarray,
    target_vertical_force_n: float,
    base_driver_effort_nm: np.ndarray,
    *,
    driver_effort_limit_nm: float,
    total_normal_limit_n: float,
    single_contact_limit_n: float,
    total_contact_limit_n: float,
    horizontal_force_tolerance_n: float,
    torque_tolerance_nm: float,
    friction_utilization_limit: float,
    friction_sides: int = 16,
) -> dict:
    """Allocate a vertical bottle support wrench from current contact frames.

    Decision variables are local normal/tangent forces plus conservative
    absolute tangent bounds. The returned generalized efforts use the supplied
    live hand Jacobians and include the current position/damping/bias effort.
    """
    count = len(contact_bases_world)
    base = np.asarray(base_driver_effort_nm, dtype=float)
    if count == 0 or any(len(values) != count for values in (
        contact_positions_world, contact_friction, contact_driver_torque_maps
    )):
        return {"success": False, "reason": "no_current_right_hand_bottle_contacts"}
    if (not np.isfinite(target_vertical_force_n) or target_vertical_force_n < 0.0
            or base.ndim != 1 or not np.all(np.isfinite(base))):
        return {"success": False, "reason": "invalid_target_or_effort_input"}
    if friction_sides < 8 or friction_sides % 2:
        raise ValueError("friction_sides must be an even number >= 8")

    channel_count = base.size
    variable_count = 6 * count
    wrench_map = np.zeros((6, variable_count), dtype=float)
    effort_map = np.zeros((channel_count, variable_count), dtype=float)
    bounds: list[tuple[float | None, float | None]] = []
    objective = np.zeros(variable_count, dtype=float)
    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []

    for index, (basis_raw, position_raw, friction_raw, torque_raw) in enumerate(zip(
        contact_bases_world, contact_positions_world, contact_friction,
        contact_driver_torque_maps,
    )):
        basis = np.asarray(basis_raw, dtype=float)
        position = np.asarray(position_raw, dtype=float)
        friction = np.asarray(friction_raw, dtype=float)
        torque = np.asarray(torque_raw, dtype=float)
        if basis.shape != (3, 3) or position.shape != (3,) or friction.shape != (2,):
            return {"success": False, "reason": f"malformed_contact_geometry_{index}"}
        if torque.shape != (channel_count, 3) or not all(np.all(np.isfinite(item)) for item in (basis, position, friction, torque)):
            return {"success": False, "reason": f"malformed_contact_jacobian_{index}"}
        if np.any(friction <= 0.0):
            return {"success": False, "reason": f"invalid_contact_friction_{index}"}

        start = 6 * index
        normal, tangent_1, tangent_2, abs_1, abs_2, resultant = range(start, start + 6)
        for local_axis, variable in enumerate((normal, tangent_1, tangent_2)):
            force = basis[:, local_axis]
            wrench_map[:3, variable] = force
            wrench_map[3:, variable] = np.cross(position - bottle_com_world, force)
            effort_map[:, variable] = torque[:, local_axis]
        objective[normal] = 1.0
        objective[abs_1] = 0.02
        objective[abs_2] = 0.02
        objective[resultant] = 0.01
        bounds.extend(((0.0, None), (None, None), (None, None), (0.0, None), (0.0, None),
                       (0.0, single_contact_limit_n)))

        inequalities.extend([
            _row(variable_count, tangent_1, 1.0, abs_1, -1.0),
            _row(variable_count, tangent_1, -1.0, abs_1, -1.0),
            _row(variable_count, tangent_2, 1.0, abs_2, -1.0),
            _row(variable_count, tangent_2, -1.0, abs_2, -1.0),
        ])
        inequality_rhs.extend((0.0, 0.0, 0.0, 0.0))

        for direction in _RESULTANT_DIRECTIONS:
            inequalities.append(_row(
                variable_count,
                normal, direction[0], tangent_1, direction[1], tangent_2, direction[2],
                resultant, -_RESULTANT_COVER_FACTOR,
            ))
            inequality_rhs.append(0.0)

        # An inscribed regular friction polygon avoids overestimating the
        # elliptical Coulomb cone supported by MuJoCo's two sliding axes.
        apothem = friction_utilization_limit * math.cos(math.pi / friction_sides)
        for side in range(friction_sides):
            theta = 2.0 * math.pi * side / friction_sides
            row = np.zeros(variable_count, dtype=float)
            row[tangent_1] = math.cos(theta) / friction[0]
            row[tangent_2] = math.sin(theta) / friction[1]
            row[normal] = -apothem
            inequalities.append(row)
            inequality_rhs.append(0.0)

    # Keep total normal and contact-resultant bounds inside the existing caps,
    # with reserve left for contact uncertainty.
    normal_row = np.zeros(variable_count, dtype=float)
    total_resultant_row = np.zeros(variable_count, dtype=float)
    for index in range(count):
        offset = 6 * index
        normal_row[offset] = 1.0
        total_resultant_row[offset + 5] = 1.0
    inequalities.extend((normal_row, total_resultant_row))
    inequality_rhs.extend((total_normal_limit_n, total_contact_limit_n))

    for axis in (0, 1):
        inequalities.extend((wrench_map[axis], -wrench_map[axis]))
        inequality_rhs.extend((horizontal_force_tolerance_n, horizontal_force_tolerance_n))
    for axis in (3, 4, 5):
        inequalities.extend((wrench_map[axis], -wrench_map[axis]))
        inequality_rhs.extend((torque_tolerance_nm, torque_tolerance_nm))

    for channel in range(channel_count):
        row = effort_map[channel]
        inequalities.extend((row, -row))
        inequality_rhs.extend((
            driver_effort_limit_nm - base[channel],
            driver_effort_limit_nm + base[channel],
        ))

    equality = np.zeros((1, variable_count), dtype=float)
    equality[0] = wrench_map[2]
    solution = linprog(
        objective,
        A_ub=np.asarray(inequalities),
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality,
        b_eq=np.asarray([target_vertical_force_n]),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        return {
            "success": False,
            "reason": "live_contact_wrench_infeasible",
            "solver_status": int(solution.status),
            "solver_message": str(solution.message),
            "target_vertical_force_n": float(target_vertical_force_n),
            "contact_count": count,
        }

    values = solution.x
    predicted_wrench = wrench_map @ values
    driver_efforts = base + effort_map @ values
    per_contact = []
    maximum_friction_utilization = 0.0
    for index, (friction_raw, basis_raw) in enumerate(zip(contact_friction, contact_bases_world)):
        offset = 6 * index
        normal, tangent_1, tangent_2 = values[offset:offset + 3]
        friction = np.asarray(friction_raw, dtype=float)
        tangent = np.asarray([tangent_1, tangent_2], dtype=float)
        utilization = float(np.linalg.norm(tangent / friction) / max(float(normal), 1e-12))
        maximum_friction_utilization = max(maximum_friction_utilization, utilization)
        local = np.asarray([normal, tangent_1, tangent_2])
        world_force = np.asarray(basis_raw, dtype=float) @ local
        actual_resultant = float(np.linalg.norm(local))
        per_contact.append({
            "normal_force_n": float(normal),
            "tangent_force_local_n": tangent.tolist(),
            "predicted_force_world_on_bottle_n": world_force.tolist(),
            "friction_utilization": utilization,
            "actual_resultant_n": actual_resultant,
            "conservative_resultant_bound_n": float(values[offset + 5]),
        })
    return {
        "success": True,
        "target_vertical_force_n": float(target_vertical_force_n),
        "predicted_wrench_on_bottle": {
            "force_n": predicted_wrench[:3].tolist(),
            "torque_about_com_nm": predicted_wrench[3:].tolist(),
        },
        "driver_effort_nm": driver_efforts.tolist(),
        "driver_effort_headroom_nm": (driver_effort_limit_nm - np.abs(driver_efforts)).tolist(),
        "total_normal_force_n": float(sum(item["normal_force_n"] for item in per_contact)),
        "total_conservative_resultant_bound_n": float(sum(
            item["conservative_resultant_bound_n"] for item in per_contact
        )),
        "maximum_friction_utilization": maximum_friction_utilization,
        "resultant_cover_factor": _RESULTANT_COVER_FACTOR,
        "per_contact": per_contact,
        "solver_objective": float(solution.fun),
        "solver_message": str(solution.message),
    }


def _row(size: int, *index_values: float) -> np.ndarray:
    row = np.zeros(size, dtype=float)
    for index, value in zip(index_values[::2], index_values[1::2]):
        row[int(index)] = float(value)
    return row


def is_contact_control_state(state: str) -> bool:
    """Recognize both position-preload and intentional simulation torque control."""
    return state in {"CONTACT_PRELOAD", "SIMULATION_ONLY_TORQUE_IMPEDANCE"}


def contact_window_allows_transfer(contact_window: dict, left_contact: bool = False) -> bool:
    """Use the validated persistence window as the transfer contact gate."""
    return (
        not left_contact
        and not bool(contact_window.get("left_contact_seen", True))
        and bool(contact_window.get("valid", False))
        and bool(contact_window.get("thumb_qualifies", False))
        and bool(contact_window.get("valid_opposing_digits", []))
    )


def _event_map(result: dict) -> dict[str, dict]:
    return {
        str(event.get("event")): event
        for event in result.get("events", [])
        if isinstance(event, dict) and event.get("event")
    }

def _gate(status: str, reason: str, evidence: dict | None = None) -> dict:
    return {
        "status": status,
        "verified": status == "PASS",
        "reason": reason,
        "evidence": evidence or {},
    }


def evaluate_gates(result: dict, *, hand_motion_threshold_rad: float = 0.001) -> dict:
    """Derive review gates from measured motion and completed physical events."""
    events = _event_map(result)
    motion = result.get("measured_hand_motion", {}).get("CLOSE", {}).get("channels", {})
    hand_excursions = {
        channel: float(values.get("maximum_qpos_rad", 0.0))
        - float(values.get("minimum_qpos_rad", 0.0))
        for channel, values in motion.items()
    }
    missing_channels = [
        channel for channel in CHANNELS
        if hand_excursions.get(channel, 0.0) < hand_motion_threshold_rad
    ]
    hand_event = events.get("CLOSE_COMPLETE")
    hand_pass = hand_event is not None and not missing_channels

    hold = events.get("HOLD_VALIDATED", {})
    contact_fraction = float(hold.get("contact_fraction", 0.0))
    contact_pass = (
        hold != {}
        and contact_fraction >= 0.80
        and int(hold.get("simultaneous_contact_steps", 0)) > 0
    )

    contact_caps = result.get("contact_force_cap_gate", {})
    contact_caps_pass = bool(contact_caps.get("passed")) and not contact_caps.get("first_failure")
    support_event = events.get("UPWARD_HAND_SUPPORT_ESTABLISHED", {})
    support_pass = (
        support_event != {}
        and float(support_event.get("measured_hand_support_n", 0.0)) > 0.0
    )
    transfer = result.get("progressive_load_transfer", {})
    load_ready = events.get("LOAD_READY", {})
    transfer_pass = bool(transfer.get("passed")) and load_ready != {}

    airborne_1mm = events.get("AIRBORNE_1MM_REACHED", {})
    airborne_1mm_pass = (
        airborne_1mm != {}
        and float(airborne_1mm.get("lift_m", 0.0)) >= 0.0009
        and not bool(airborne_1mm.get("table_supported", True))
    )
    airborne_5mm = events.get("AIRBORNE_5MM_REACHED", {})
    airborne_5mm_pass = (
        airborne_5mm != {}
        and float(airborne_5mm.get("lift_m", 0.0)) >= 0.0048
        and not bool(airborne_5mm.get("table_supported", True))
    )

    lift = events.get("SHORT_LIFT_REACHED", {})
    lift_m = float(lift.get("lift_m", 0.0))
    lift_pass = lift != {} and lift_m >= 0.030 and not bool(lift.get("table_supported", True))

    post_hold = events.get("POST_LIFT_HOLD_COMPLETE", {})
    hold_pass = (
        post_hold != {}
        and float(post_hold.get("duration_s", 0.0)) >= 1.0
        and float(post_hold.get("lift_m", 0.0)) >= 0.030
    )

    release_event = events.get("RELEASE_COMPLETE", {})
    release = result.get("release", {})
    release_pass = (
        release_event != {}
        and release.get("passed") is True
        and release.get("final_right_hand_bottle_contact") is False
        and release.get("final_table_supported") is True
        and release.get("fingers_opened_time_s") is not None
    )

    first_failure = result.get("error")
    pr_ci_status = str(result.get("pr_ci", {}).get("status", "NOT RUN"))
    if pr_ci_status not in {"PASS", "FAIL", "NOT RUN"}:
        pr_ci_status = "FAIL"
    return {
        "HAND_ACTUATION": _gate(
            "PASS" if hand_pass else "FAIL",
            "all six commanded right-hand drivers showed measured closure motion"
            if hand_pass else "measured close motion is missing or below the 0.001 rad gate",
            {
                "close_complete_event": hand_event,
                "minimum_motion_rad": hand_motion_threshold_rad,
                "driver_excursion_rad": hand_excursions,
                "missing_or_insufficient_channels": missing_channels,
            },
        ),
        "FINGER_BOTTLE_CONTACT": _gate(
            "PASS" if contact_pass else "FAIL",
            "one-second HOLD validated simultaneous thumb and opposing-digit bottle contact"
            if contact_pass else "validated thumb-plus-opposing-digit HOLD was not reached",
            {
                "hold_validated_event": hold or None,
                "required_contact_fraction": 0.80,
                "measured_contact_fraction": contact_fraction,
                "first_failure": first_failure,
            },
        ),
        "CONTACT_FORCE_WITHIN_CAP": _gate(
            "PASS" if contact_caps_pass else "FAIL",
            "measured total normal, per-contact resultant, total resultant, and applied effort stayed within hard caps"
            if contact_caps_pass else "a hard contact/effort cap failed or complete measurements are missing",
            contact_caps,
        ),
        "UPWARD_HAND_SUPPORT": _gate(
            "PASS" if support_pass else "FAIL",
            "measured right-hand vertical support increased during progressive transfer"
            if support_pass else "positive measured upward hand support was not established",
            support_event or {"first_failure": first_failure},
        ),
        "TABLE_LOAD_TRANSFER": _gate(
            "PASS" if transfer_pass else "FAIL",
            "noise-aware table unloading and matching upward hand support reached LOAD_READY"
            if transfer_pass else "progressive table-to-hand transfer or LOAD_READY was not reached",
            {"progressive_transfer": transfer, "load_ready_event": load_ready or None,
             "first_failure": first_failure},
        ),
        "AIRBORNE_1MM": _gate(
            "PASS" if airborne_1mm_pass else "FAIL",
            "free bottle rose at least 1 mm with table support absent"
            if airborne_1mm_pass else "1 mm table-independent pickup was not completed",
            {"event": airborne_1mm or None, "first_failure": first_failure},
        ),
        "AIRBORNE_5MM": _gate(
            "PASS" if airborne_5mm_pass else "FAIL",
            "free bottle rose at least 5 mm with table support absent"
            if airborne_5mm_pass else "5 mm table-independent lift was not completed",
            {"event": airborne_5mm or None, "first_failure": first_failure},
        ),
        "PHYSICAL_LIFT_30MM": _gate(
            "PASS" if lift_pass else "FAIL",
            "free bottle rose at least 30 mm and had lost table support"
            if lift_pass else "30 mm free-physics lift was not completed; see first failure and event trace",
            {
                "short_lift_event": lift or None,
                "measured_peak_lift_m": result.get("peak_lift_before_release_m", lift_m),
                "table_supported_at_gate": lift.get("table_supported"),
                "first_failure": first_failure,
            },
        ),
        "HOLD_1S": _gate(
            "PASS" if hold_pass else "FAIL",
            "bottle remained lifted during the completed one-second hold"
            if hold_pass else "post-lift one-second hold was not completed",
            {"post_lift_hold_event": post_hold or None, "first_failure": first_failure},
        ),
        "PHYSICAL_RELEASE": _gate(
            "PASS" if release_pass else "FAIL",
            "fingers opened, hand contact cleared, and bottle settled on the table under free physics"
            if release_pass else "physical release and free-physics settle were not completed",
            {"release_event": release_event or None, "release_result": release or None,
             "first_failure": first_failure},
        ),
        "PR_CI": _gate(
            pr_ci_status,
            "all required GitHub checks passed for the pushed PR head"
            if pr_ci_status == "PASS"
            else "GitHub PR checks are not all green",
            result.get("pr_ci", {}),
        ),
    }
