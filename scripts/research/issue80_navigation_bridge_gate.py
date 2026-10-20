"""Offline contract for gating PR #82 Twist proposals before an MC bridge.

This module has no ROS publisher or service client. It is a testable adapter
contract, not a live bridge and not evidence of physical movement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from issue80_m1_first_motion import (
    MC_FSM_STABLE,
    McFsmStateCode,
    VELOCITY_SOURCE_NAME,
    snapshot_fault,
)
from issue80_m1_mode_logic import MODE_LD


MAX_FORWARD_MPS = 0.20
MAX_ANGULAR_RPS = 0.30
PROPOSAL_TIMEOUT_S = 0.20
NAV_STATUS_TIMEOUT_S = 30.0
STATE_TIMEOUT_S = 0.50
MIN_ROOT_Z_M = 0.52
MAX_TILT_DEG = 15.0
SUPPORTED_ACTIVE_STATES = frozenset({"ALIGN", "DRIVE", "FINAL_ALIGN", "SETTLING"})
ROTATION_ONLY_STATES = frozenset({"ALIGN", "FINAL_ALIGN"})


def _zero_result(reason: str, *, transport_verified: bool) -> dict[str, Any]:
    return {
        "decision": "ZERO" if transport_verified else "BLOCKED",
        "reason": reason,
        "forward_to_mc": bool(transport_verified),
        "linear_x": 0.0,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.0,
        "physical_motion_status": "NOT_MEASURED",
    }


def gate_navigation_twist_proposal(
    proposal: Mapping[str, Any] | None,
    *,
    nav_status: str | None,
    nav_status_received_steady_s: float | None,
    received_steady_s: float | None,
    now_steady_s: float,
    state_snapshot: Mapping[str, Any] | None,
    joint_limits: Mapping[str, Mapping[str, float]],
    source_registration_verified: bool,
    transport_preflight_verified: bool,
    same_sample_handshake_confirmed: bool,
    cancelled: bool = False,
) -> dict[str, Any]:
    """Return a bounded command only when the proposal and MC state are ready.

    PR #82 publishes unstamped ``Twist`` values and status on a separate topic.
    A future bridge must stamp proposal receipt with its local monotonic clock,
    preserve the adapter status, and pass a contemporaneous MC state snapshot.
    MC graph endpoint uniqueness/QoS/isolation, including exactly one proposal
    producer, one MC command publisher and one verified MC subscriber, are
    summarized by the verified transport preflight; this helper creates no
    endpoint. Status is change-published, so its receipt age is bounded by the
    adapter's 30-second goal timeout; Twist receipt age is limited to 200 ms.
    """
    if not transport_preflight_verified:
        return _zero_result("transport_preflight_unverified", transport_verified=False)
    if not math.isfinite(now_steady_s):
        return _zero_result("invalid_bridge_clock", transport_verified=True)
    try:
        status_received = float(nav_status_received_steady_s)
    except (TypeError, ValueError):
        return _zero_result("nav_status_receive_time_invalid", transport_verified=True)
    if not math.isfinite(status_received) or status_received > now_steady_s:
        return _zero_result("nav_status_receive_time_invalid", transport_verified=True)
    if now_steady_s - status_received > NAV_STATUS_TIMEOUT_S:
        return _zero_result("nav_status_stale", transport_verified=True)
    if cancelled:
        return _zero_result("cancelled", transport_verified=True)

    status = nav_status.upper() if isinstance(nav_status, str) else ""
    if status not in SUPPORTED_ACTIVE_STATES:
        return _zero_result("proposal_status_not_active", transport_verified=True)

    if proposal is None or received_steady_s is None:
        return _zero_result("proposal_missing", transport_verified=True)
    try:
        received = float(received_steady_s)
    except (TypeError, ValueError):
        return _zero_result("proposal_receive_time_invalid", transport_verified=True)
    if not math.isfinite(received) or received > now_steady_s:
        return _zero_result("proposal_receive_time_invalid", transport_verified=True)
    if now_steady_s - received > PROPOSAL_TIMEOUT_S:
        return _zero_result("proposal_stale", transport_verified=True)

    try:
        command = {
            key: float(proposal[key])
            for key in (
                "linear_x", "linear_y", "linear_z",
                "angular_x", "angular_y", "angular_z",
            )
        }
    except (KeyError, TypeError, ValueError):
        return _zero_result("proposal_fields_invalid", transport_verified=True)
    if not all(math.isfinite(value) for value in command.values()):
        return _zero_result("proposal_nonfinite", transport_verified=True)
    if command["linear_y"] != 0.0 or command["linear_z"] != 0.0 or any(
        command[key] != 0.0 for key in ("angular_x", "angular_y")
    ):
        return _zero_result("proposal_nonplanar", transport_verified=True)
    if (
        command["linear_x"] < 0.0
        or command["linear_x"] > MAX_FORWARD_MPS
        or abs(command["angular_z"]) > MAX_ANGULAR_RPS
    ):
        return _zero_result("proposal_outside_adapter_bounds", transport_verified=True)
    if status in ROTATION_ONLY_STATES and command["linear_x"] != 0.0:
        return _zero_result("rotation_state_has_forward_command", transport_verified=True)
    if status == "SETTLING" and any(command.values()):
        return _zero_result("settling_state_has_nonzero_command", transport_verified=True)

    if not source_registration_verified:
        return _zero_result("source_registration_unverified", transport_verified=True)
    if not same_sample_handshake_confirmed:
        return _zero_result("mode_source_handshake_unconfirmed", transport_verified=True)
    if not isinstance(state_snapshot, Mapping):
        return _zero_result("state_snapshot_missing", transport_verified=True)

    fault = snapshot_fault(
        state_snapshot,
        joint_limits,
        state_timeout_s=STATE_TIMEOUT_S,
        min_root_z=MIN_ROOT_Z_M,
        max_tilt_deg=MAX_TILT_DEG,
        expected_source=VELOCITY_SOURCE_NAME,
        required_forward_mps=command["linear_x"],
    )
    if fault:
        return _zero_result(f"unsafe_mc_snapshot:{fault}", transport_verified=True)
    if (
        state_snapshot.get("get_mode_desc") != MODE_LD
        or state_snapshot.get("common_mode_desc") != MODE_LD
    ):
        return _zero_result("locomotion_mode_not_confirmed", transport_verified=True)
    if state_snapshot.get("fsm_state") not in (
        MC_FSM_STABLE,
        int(McFsmStateCode.MOVING),
    ):
        return _zero_result("fsm_not_stable_or_moving", transport_verified=True)
    if (
        state_snapshot.get("get_source_name") != VELOCITY_SOURCE_NAME
        or state_snapshot.get("common_source_name") != VELOCITY_SOURCE_NAME
    ):
        return _zero_result("expected_source_not_selected", transport_verified=True)

    bounds = state_snapshot.get("velocity_bounds")
    if not isinstance(bounds, Mapping):
        return _zero_result("live_velocity_bounds_missing", transport_verified=True)
    for field, value in (
        ("forward", command["linear_x"]),
        ("lateral", command["linear_y"]),
        ("angular", command["angular_z"]),
    ):
        pair = bounds.get(field)
        if not isinstance(pair, Mapping):
            return _zero_result(f"live_{field}_bounds_missing", transport_verified=True)
        try:
            minimum = float(pair["min_value"])
            maximum = float(pair["max_value"])
        except (KeyError, TypeError, ValueError):
            return _zero_result(f"live_{field}_bounds_invalid", transport_verified=True)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            return _zero_result(f"live_{field}_bounds_invalid", transport_verified=True)
        if value < minimum or value > maximum:
            return _zero_result(f"proposal_outside_live_{field}_bounds", transport_verified=True)

    return {
        "decision": "ALLOW_PROPOSAL",
        "reason": "all_offline_bridge_gates_passed",
        "forward_to_mc": True,
        **command,
        "physical_motion_status": "NOT_MEASURED",
    }
