"""Pure mode-transition classification for the Issue 80 research probe."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


MODE_LD = "LOCOMOTION_DEFAULT"
SUCCESS_STATUS = 1
SDK_INPUT_SOURCE_NAME = "node.set_mc_input_source"


def is_valid_input_source_name(value: str) -> bool:
    """Accept a unique research source or the pinned SDK example identity."""
    return value == SDK_INPUT_SOURCE_NAME or re.fullmatch(
        r"issue80_m1_[A-Za-z0-9_]{8,40}", value
    ) is not None


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
