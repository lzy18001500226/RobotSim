"""Pure mode-transition classification for the Issue 80 research probe."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


MODE_LD = "LOCOMOTION_DEFAULT"
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
SDK_INPUT_SOURCE_NAME = "node.set_mc_input_source"


def is_valid_input_source_name(value: str) -> bool:
    """Accept a unique research source or the pinned SDK example identity."""
    return value == SDK_INPUT_SOURCE_NAME or re.fullmatch(
        r"issue80_m1_[A-Za-z0-9_]{8,40}", value
    ) is not None


def assess_input_source_add(
    response_header_code: int,
    task_id: int,
    state_value: int,
) -> dict[str, Any]:
    """Separate RPC transport success from the task's registration state."""
    state = int(state_value)
    if int(response_header_code) != 0:
        assessment = "RPC_FAILED"
    elif state == SUCCESS_STATUS:
        assessment = "SERVER_STATE_SUCCESS"
    elif state in (200, 300, 400):
        assessment = "TASK_PENDING_OR_RUNNING"
    elif state == 0 and int(task_id) > 0:
        assessment = "RPC_ACK_STATE_UNKNOWN"
    else:
        assessment = "SERVER_STATE_NOT_SUCCESS"
    return {
        "response_header_code": int(response_header_code),
        "task_id": int(task_id),
        "state_value": state,
        "state_name": TASK_STATES.get(state, "UNRECOGNIZED"),
        "assessment": assessment,
        "registration_confirmed": assessment == "SERVER_STATE_SUCCESS",
    }


def source_arbitration_confirmed(
    current_source_name: str,
    common_source_name: str,
    expected_source_name: str,
    common_state_age_s: float | None,
    *,
    max_age_s: float = 0.5,
) -> bool:
    """Require matching current-source service and fresh McCommonState evidence."""
    return (
        current_source_name == expected_source_name
        and common_source_name == expected_source_name
        and common_state_age_s is not None
        and 0.0 <= float(common_state_age_s) <= max_age_s
    )


def classify_ld_transition(
    set_result: Mapping[str, Any] | None,
    mode_observations: Sequence[Mapping[str, Any]],
    common_mode: str | None,
    *,
    readback_error: bool = False,
) -> str:
    """Do not treat an RPC acknowledgement as proof of a live LD transition."""
    if not set_result:
        return "LD_REQUEST_NOT_SENT"
    if set_result.get("response_status_value") != SUCCESS_STATUS:
        return "REJECTED"
    if readback_error:
        return "READBACK_INTERFACE_ERROR"

    last_action = next(
        (item for item in reversed(mode_observations) if item.get("action_desc") is not None),
        None,
    )
    get_mode = last_action.get("action_desc") if last_action else None
    if last_action and last_action.get("response_header_code") != 0:
        return "READBACK_INTERFACE_ERROR"
    if get_mode and common_mode and get_mode != common_mode:
        return "READBACK_INTERFACE_INCONSISTENCY"
    if get_mode == MODE_LD and common_mode == MODE_LD:
        return "LD_CONFIRMED"
    return "ACCEPTED_BUT_NOT_TRANSITIONED"
