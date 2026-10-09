"""Evidence gates for the G1 DFQ isolated physical-grasp runner."""

from __future__ import annotations


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
    "ISOLATED_PHYSICAL_LIFT_30MM",
    "HOLD_1S",
    "PHYSICAL_RELEASE",
    "FULL_WORKCELL_PICKUP",
)


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


def select_effort_feedforward(channel: str, static_efforts: dict, load_efforts: dict,
                              load_build_schedule: bool) -> float:
    """Use the measured-contact allocation only after the load phase starts."""
    source = load_efforts if load_build_schedule else static_efforts
    if channel not in source:
        raise KeyError(f"missing simulation effort allocation for {channel}")
    return float(source[channel])


def is_contact_control_state(state: str) -> bool:
    """Recognize both position-preload and intentional simulation torque control."""
    return state in {"CONTACT_PRELOAD", "SIMULATION_ONLY_TORQUE_IMPEDANCE"}


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

    full_workcell_pass = result.get("full_workcell_integration_verified") is True
    full_workcell_status = "PASS" if full_workcell_pass else "NOT RUN"

    first_failure = result.get("error")
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
        "ISOLATED_PHYSICAL_LIFT_30MM": _gate(
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
        "FULL_WORKCELL_PICKUP": _gate(
            full_workcell_status,
            "full-workcell pickup requires a separate station integration run"
            if not full_workcell_pass else "full-workcell pickup was independently verified",
            result.get("full_workcell_evidence", {}),
        ),
    }
