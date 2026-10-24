"""Pure protocol classification for Issue 80's simulation-only first-motion probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MODE_SD = "STAND_DEFAULT"
MODE_LD = "LOCOMOTION_DEFAULT"
OFFICIAL_VELOCITY_SOURCE = "node.locomotion_velocity"
OFFICIAL_ACTION_SOURCE = "rc"
SUCCESS_STATUS = 1
TASK_STATES = {
    0: "UNKNOWN",
    1: "SUCCESS",
    2: "FAILURE",
    3: "ABORTED",
    4: "TIMEOUT",
    5: "INVALID",
    6: "IN_MANUAL",
    100: "NOT_READY",
    200: "PENDING",
    300: "CREATED",
    400: "RUNNING",
}


def is_valid_input_source_name(value: str) -> bool:
    """The live first-motion harness is pinned to the vendor example identity."""
    return value == OFFICIAL_VELOCITY_SOURCE


def assess_input_source_add(
    response_header_code: int,
    task_id: int,
    state_value: int,
) -> dict[str, Any]:
    """Report RPC acceptance separately from the asynchronous Add task result."""
    code = int(response_header_code)
    state = int(state_value)
    if code != 0:
        assessment = "ADD_REJECTED"
    elif state == SUCCESS_STATUS:
        assessment = "ADD_TASK_SUCCEEDED"
    elif state in (2, 3, 4, 5, 6):
        assessment = "ADD_TASK_TERMINAL_FAILURE"
    elif state == 100:
        assessment = "ADD_TASK_NOT_READY"
    elif state in (0, 200, 300, 400):
        assessment = "ADD_ACCEPTED_TASK_NOT_TERMINAL"
    else:
        assessment = "ADD_TASK_STATE_UNRECOGNIZED"
    return {
        "response_header_code": code,
        "task_id": int(task_id),
        "state_value": state,
        "state_name": TASK_STATES.get(state, "UNRECOGNIZED"),
        "assessment": assessment,
        "registration_operation_succeeded": code == 0,
        "task_state_is_success": state == SUCCESS_STATUS,
        "enable_fallback_allowed": code != 0 or state in (2, 3, 4, 5, 6),
    }


def classify_set_mc_action_response(
    response_header_code: int,
    response_status_value: int,
    response_message: str,
) -> dict[str, Any]:
    """Interpret both the detailed result code and common status; do not claim application."""
    code = int(response_header_code)
    status = int(response_status_value)
    accepted = code == 0 and status == SUCCESS_STATUS
    return {
        "response_header_code": code,
        "response_status_value": status,
        "response_message": str(response_message),
        "message": str(response_message),
        "accepted_by_service": accepted,
        "applied_mode_confirmed": False,
    }


def classify_ld_transition(
    set_result: Mapping[str, Any] | None,
    mode_observations: Sequence[Mapping[str, Any]],
    common_mode: str | None,
    *,
    readback_error: bool = False,
) -> str:
    """Separate service rejection, request acceptance, and later mode application."""
    if not set_result:
        return "LD_REQUEST_NOT_SENT"
    header_code = set_result.get("response_header_code")
    if header_code is not None and int(header_code) != 0:
        return f"REJECTED_HEADER_CODE_{int(header_code)}"
    if set_result.get("response_status_value") != SUCCESS_STATUS:
        return "REJECTED_STATUS"
    if readback_error:
        return "READBACK_INTERFACE_ERROR"

    last_action = next(
        (item for item in reversed(mode_observations) if item.get("action_desc") is not None),
        None,
    )
    get_mode = last_action.get("action_desc") if last_action else None
    if last_action and last_action.get("response_header_code") != 0:
        return "READBACK_INTERFACE_ERROR"
    if get_mode and common_mode and get_mode not in {MODE_SD, MODE_LD}:
        return "UNRELATED_GET_MODE"
    if common_mode and common_mode not in {MODE_SD, MODE_LD}:
        return "UNRELATED_COMMON_MODE"
    if get_mode and common_mode and get_mode != common_mode:
        return "READBACK_TRANSITION_IN_PROGRESS"
    if get_mode == MODE_LD and common_mode == MODE_LD:
        return "LD_CONFIRMED"
    if get_mode == MODE_SD and common_mode == MODE_SD:
        return "ACCEPTED_WAITING_FOR_VALID_LOCOMOTION_INPUT"
    return "ACCEPTED_MODE_READBACK_PENDING"
