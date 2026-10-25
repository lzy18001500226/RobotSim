"""Pure protocol classification for Issue 80's simulation-only first-motion probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any


MODE_SD = "STAND_DEFAULT"
MODE_LD = "LOCOMOTION_DEFAULT"
OFFICIAL_VELOCITY_SOURCE = "node.locomotion_velocity"
OFFICIAL_ACTION_SOURCE = "rc"
SUCCESS_STATUS = 1
MODE_IDLE_STATUS = 0
MODE_RUNNING_STATUS = 100
MODE_TRANSITION_STATUS = 200
READBACK_SYNC_MAX_S = 0.20

STAND_VERIFIED = "STAND_VERIFIED"
LOCOMOTION_REQUESTED = "LOCOMOTION_REQUESTED"
LOCOMOTION_CONFIRMED = "LOCOMOTION_CONFIRMED"
MOTION_ACTIVE = "MOTION_ACTIVE"
ZERO_AND_STOP_REQUESTED = "ZERO_AND_STOP_REQUESTED"
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


def classify_locomotion_activation_readbacks(
    snapshot: Mapping[str, Any],
    *,
    expected_source: str = OFFICIAL_VELOCITY_SOURCE,
    maximum_readback_age_s: float = 0.5,
    maximum_sync_skew_s: float = READBACK_SYNC_MAX_S,
) -> dict[str, Any]:
    """Require fresh, agreeing mode/source readbacks before any motion is enabled.

    STAND and LOCOMOTION are both legitimate while a mode request is pending.
    Mismatched but individually valid values remain pending; unrelated modes or
    conflicting selected sources are fatal. Only status=100 on both mode
    readbacks and the same active source on both source readbacks confirms LD.
    """
    try:
        now = float(snapshot["sample_monotonic_s"])
        readback_times = snapshot["readback_monotonic_s"]
        state_ages = snapshot["state_age_s"]
        get_mode_time = float(readback_times["get_mode"])
        get_source_time = float(readback_times["get_source"])
        common_age = float(state_ages["common"])
        max_age = float(maximum_readback_age_s)
        max_skew = float(maximum_sync_skew_s)
        get_mode_age = now - get_mode_time
        get_source_age = now - get_source_time
        common_time = now - common_age
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "classification": "PENDING_READBACK_UNAVAILABLE",
            "reason": "required mode/source/common timestamps are missing or malformed",
            "motion_allowed": False,
        }
    numeric = (
        now, get_mode_age, get_source_age, common_age, max_age, max_skew,
        common_time, get_mode_time, get_source_time,
    )
    if (
        not all(math.isfinite(value) for value in numeric)
        or max_age <= 0.0 or max_skew < 0.0
        or get_mode_age < 0.0 or get_source_age < 0.0 or common_age < 0.0
    ):
        return {
            "classification": "PENDING_READBACK_INVALID_TIME",
            "reason": "readback timestamps are nonfinite, future-dated, or have invalid limits",
            "motion_allowed": False,
        }

    get_mode = snapshot.get("get_mode_desc")
    common_mode = snapshot.get("common_mode_desc")
    get_source = snapshot.get("get_source_name", "")
    common_source = snapshot.get("common_source_name", "")
    get_code = snapshot.get("get_mode_header_code")
    source_code = snapshot.get("get_source_header_code")
    get_status = snapshot.get("get_mode_status_value")
    common_status = snapshot.get("common_mode_status_value")
    result: dict[str, Any] = {
        "get_mode_desc": get_mode,
        "common_mode_desc": common_mode,
        "get_mode_header_code": get_code,
        "get_source_header_code": source_code,
        "get_mode_status_value": get_status,
        "common_mode_status_value": common_status,
        "get_source_name": get_source,
        "common_source_name": common_source,
        "get_mode_age_s": get_mode_age,
        "get_source_age_s": get_source_age,
        "common_state_age_s": common_age,
        "mode_source_skew_s": abs(get_mode_time - get_source_time),
        "mode_common_skew_s": abs(get_mode_time - common_time),
        "source_common_skew_s": abs(get_source_time - common_time),
        "motion_allowed": False,
    }
    allowed_modes = {MODE_SD, MODE_LD}
    if get_mode not in allowed_modes or common_mode not in allowed_modes:
        return {
            "classification": "ERROR_UNRELATED_MODE",
            "reason": "GetMcAction or CommonState reports a mode outside STAND/LOCOMOTION",
            **result,
        }
    if (
        get_source not in ("", expected_source)
        or common_source not in ("", expected_source)
    ):
        return {
            "classification": "ERROR_SOURCE_CONFLICT",
            "reason": "a selected-source readback conflicts with the registered locomotion source",
            **result,
        }
    if get_code != 0 or source_code != 0:
        return {
            "classification": "PENDING_SERVICE_READBACK_ERROR",
            "reason": "GetMcAction or GetCurrentInputSource did not return header code zero",
            **result,
        }
    if max(get_mode_age, get_source_age, common_age) > max_age:
        return {
            "classification": "PENDING_STALE_READBACK",
            "reason": "one or more mode/source/common readbacks exceed the freshness bound",
            **result,
        }
    if max(
        result["mode_source_skew_s"],
        result["mode_common_skew_s"],
        result["source_common_skew_s"],
    ) > max_skew:
        return {
            "classification": "PENDING_READBACK_SKEW",
            "reason": "mode, source and CommonState readbacks are not contemporaneous",
            **result,
        }
    if get_mode != common_mode:
        return {
            "classification": "PENDING_MODE_DISAGREEMENT",
            "reason": "GetMcAction and CommonState disagree; the request is not applied/confirmed",
            **result,
        }
    if get_source != common_source:
        return {
            "classification": "PENDING_SOURCE_DISAGREEMENT",
            "reason": "GetCurrentInputSource and CommonState disagree",
            **result,
        }
    if get_mode == MODE_SD:
        return {
            "classification": "PENDING_STAND_REMAINS_ACTIVE",
            "reason": "the locomotion service response has not changed the live mode",
            **result,
        }
    if get_source != expected_source:
        return {
            "classification": "PENDING_SOURCE_NOT_SELECTED",
            "reason": "LOCOMOTION is visible but the expected developer source is not selected",
            **result,
        }
    if get_status != MODE_RUNNING_STATUS or common_status != MODE_RUNNING_STATUS:
        return {
            "classification": "PENDING_ACTION_NOT_RUNNING",
            "reason": "both mode readbacks must report the installed RUNNING status value 100",
            **result,
        }
    return {
        **result,
        "classification": "LOCOMOTION_CONFIRMED",
        "reason": "fresh, synchronized GetMcAction/CommonState and source readbacks agree",
        "motion_allowed": True,
    }


@dataclass
class MCLocomotionAbortLifecycle:
    """Fail-closed activation/stop state machine for one M1 window lifecycle."""

    state: str = "UNINITIALIZED"
    transitions: list[dict[str, Any]] = field(default_factory=list)
    stand_request_count: int = 0
    late_locomotion_after_abort: bool = False
    locomotion_confirmed_before_stop: bool = False
    rearm_forbidden: bool = False
    fault: str | None = None

    _ALLOWED: dict[str, set[str]] = field(default_factory=lambda: {
        "UNINITIALIZED": {STAND_VERIFIED, ZERO_AND_STOP_REQUESTED},
        STAND_VERIFIED: {LOCOMOTION_REQUESTED, ZERO_AND_STOP_REQUESTED},
        LOCOMOTION_REQUESTED: {LOCOMOTION_CONFIRMED, ZERO_AND_STOP_REQUESTED},
        LOCOMOTION_CONFIRMED: {MOTION_ACTIVE, ZERO_AND_STOP_REQUESTED},
        MOTION_ACTIVE: {ZERO_AND_STOP_REQUESTED},
        ZERO_AND_STOP_REQUESTED: {STAND_VERIFIED},
    }, repr=False)

    @property
    def nonzero_motion_allowed(self) -> bool:
        return self.state == MOTION_ACTIVE

    def _transition(self, target: str, *, now_s: float, reason: str) -> None:
        if target not in self._ALLOWED.get(self.state, set()):
            raise ValueError(f"invalid MC lifecycle transition {self.state}->{target}")
        self.transitions.append({
            "from": self.state,
            "to": target,
            "monotonic_s": float(now_s),
            "reason": str(reason),
        })
        self.state = target

    def verify_initial_stand(self, *, confirmed: bool, now_s: float) -> bool:
        if not confirmed:
            self.fault = "initial_STAND_NOT_CONFIRMED"
            return False
        if self.state != "UNINITIALIZED":
            raise ValueError("initial STAND can only be established from UNINITIALIZED")
        self._transition(STAND_VERIFIED, now_s=now_s, reason="fresh initial STAND/FSM/body gates passed")
        return True

    def request_locomotion(self, *, accepted: bool, now_s: float) -> bool:
        if self.rearm_forbidden:
            raise ValueError("activation abort is unresolved; a new lifecycle is required")
        if self.state != STAND_VERIFIED:
            raise ValueError("LOCOMOTION request requires STAND_VERIFIED")
        if not accepted:
            self.fault = "LOCOMOTION_SERVICE_REQUEST_REJECTED"
            self.rearm_forbidden = True
            self._transition(
                ZERO_AND_STOP_REQUESTED, now_s=now_s,
                reason="SetMcAction was not accepted; fail closed without motion",
            )
            return False
        self._transition(
            LOCOMOTION_REQUESTED, now_s=now_s,
            reason="SetMcAction accepted by service; application remains unconfirmed",
        )
        return True

    def observe_locomotion_readback(
        self, observation: Mapping[str, Any], *, now_s: float,
    ) -> str:
        classification = str(observation.get("classification", "PENDING_READBACK_UNAVAILABLE"))
        if self.state == LOCOMOTION_REQUESTED:
            if classification == "LOCOMOTION_CONFIRMED":
                self._transition(
                    LOCOMOTION_CONFIRMED, now_s=now_s,
                    reason="fresh synchronized mode and source readbacks confirmed LOCOMOTION",
                )
            elif classification.startswith("ERROR_"):
                self.fault = str(observation.get("reason") or classification)
                self.rearm_forbidden = True
                self._transition(
                    ZERO_AND_STOP_REQUESTED, now_s=now_s,
                    reason="activation readback contains an unrelated mode or conflicting source",
                )
        elif self.state == ZERO_AND_STOP_REQUESTED and classification in {
            "LOCOMOTION_CONFIRMED", "LATE_LOCOMOTION_MODE_OBSERVED",
        }:
            self.late_locomotion_after_abort = True
            self.rearm_forbidden = True
            self.transitions.append({
                "from": ZERO_AND_STOP_REQUESTED,
                "to": ZERO_AND_STOP_REQUESTED,
                "monotonic_s": float(now_s),
                "reason": str(
                    observation.get("reason")
                    or "late LOCOMOTION mode readback after abort; zero-only, no rearm"
                ),
                "classification": classification,
                "late_locomotion_after_abort": True,
            })
        return classification

    def expire_activation(self, *, now_s: float, reason: str) -> None:
        if self.state == LOCOMOTION_REQUESTED:
            self.fault = str(reason)
            self.rearm_forbidden = True
            self._transition(
                ZERO_AND_STOP_REQUESTED, now_s=now_s,
                reason="bounded LOCOMOTION activation expired; zero and request STAND recovery",
            )

    def begin_motion(self, *, now_s: float) -> None:
        if self.rearm_forbidden:
            raise ValueError("activation abort is unresolved; motion rearm is forbidden")
        if self.state != LOCOMOTION_CONFIRMED:
            raise ValueError("nonzero motion is forbidden before LOCOMOTION_CONFIRMED")
        self._transition(
            MOTION_ACTIVE, now_s=now_s,
            reason="bounded motion entered only after fresh mode/source confirmation",
        )

    def request_zero_and_stop(self, *, now_s: float, reason: str) -> None:
        if self.state == ZERO_AND_STOP_REQUESTED:
            return
        if self.state == "UNINITIALIZED":
            self.fault = str(reason)
            return
        if self.state in {LOCOMOTION_CONFIRMED, MOTION_ACTIVE}:
            self.locomotion_confirmed_before_stop = True
        self._transition(
            ZERO_AND_STOP_REQUESTED, now_s=now_s,
            reason=str(reason),
        )

    def begin_stand_request_once(self, *, now_s: float) -> bool:
        if self.state != ZERO_AND_STOP_REQUESTED:
            return False
        if self.stand_request_count != 0:
            return False
        self.stand_request_count = 1
        self.transitions.append({
            "from": ZERO_AND_STOP_REQUESTED,
            "to": ZERO_AND_STOP_REQUESTED,
            "monotonic_s": float(now_s),
            "reason": "one validated STAND recovery request started while zero pump remains active",
            "stand_request_attempt": 1,
        })
        return True

    def confirm_stand_after_stop(
        self, *, stand_readback_pass: bool, physical_stop_gate_pass: bool, now_s: float,
    ) -> bool:
        if (
            self.state != ZERO_AND_STOP_REQUESTED
            or self.stand_request_count != 1
            or not stand_readback_pass
            or not physical_stop_gate_pass
        ):
            return False
        self._transition(
            STAND_VERIFIED, now_s=now_s,
            reason="fresh STAND/FSM/body and complete composite physical stop gate passed",
        )
        self.stand_request_count = 0
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "nonzero_motion_allowed": self.nonzero_motion_allowed,
            "stand_request_count": self.stand_request_count,
            "late_locomotion_after_abort": self.late_locomotion_after_abort,
            "locomotion_confirmed_before_stop": self.locomotion_confirmed_before_stop,
            "rearm_forbidden": self.rearm_forbidden,
            "fault": self.fault,
            "transitions": list(self.transitions),
        }


def run_zero_only_locomotion_activation_observation(
    *,
    sample: Any,
    publish_zero: Any,
    lifecycle: MCLocomotionAbortLifecycle,
    safety_fault: Any,
    expected_source: str = OFFICIAL_VELOCITY_SOURCE,
    timeout_s: float = 2.0,
    maximum_readback_age_s: float = 0.5,
    maximum_sync_skew_s: float = READBACK_SYNC_MAX_S,
    publish_hz: float = 50.0,
    clock: Any,
    sleep: Any,
    on_sample: Any = None,
) -> dict[str, Any]:
    """Observe a pending LD request with zero commands only; never publishes motion."""
    try:
        duration = float(timeout_s)
        maximum_age = float(maximum_readback_age_s)
        maximum_skew = float(maximum_sync_skew_s)
        frequency = float(publish_hz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("zero-only activation limits must be numeric") from exc
    if (
        not all(math.isfinite(value) for value in (duration, maximum_age, maximum_skew, frequency))
        or not 0.0 < duration <= 3.5
        or maximum_age <= 0.0
        or maximum_skew < 0.0
        or frequency != 50.0
    ):
        raise ValueError("zero-only activation limits are outside the bounded 50 Hz protocol")
    if lifecycle.state != LOCOMOTION_REQUESTED:
        raise ValueError("zero-only activation observation requires LOCOMOTION_REQUESTED")

    started = float(clock())
    deadline = started + duration
    next_publish = started
    zero_calls = 0
    publish_errors: list[str] = []
    observations: list[dict[str, Any]] = []
    terminal_reason: str | None = None
    while float(clock()) < deadline:
        now = float(clock())
        if now >= next_publish:
            try:
                publish_zero()
                zero_calls += 1
            except Exception as exc:
                terminal_reason = f"zero_publish_exception:{type(exc).__name__}:{exc}"
                publish_errors.append(terminal_reason)
                lifecycle.rearm_forbidden = True
                lifecycle.request_zero_and_stop(now_s=now, reason=terminal_reason)
                break
            next_publish = now + 1.0 / frequency
        try:
            snapshot = sample()
        except Exception as exc:
            terminal_reason = f"activation_observation_exception:{type(exc).__name__}:{exc}"
            lifecycle.rearm_forbidden = True
            lifecycle.request_zero_and_stop(now_s=float(clock()), reason=terminal_reason)
            break
        try:
            safety = safety_fault(snapshot)
        except Exception as exc:
            safety = f"activation_safety_callback_exception:{type(exc).__name__}:{exc}"
        classification = classify_locomotion_activation_readbacks(
            snapshot,
            expected_source=expected_source,
            maximum_readback_age_s=maximum_age,
            maximum_sync_skew_s=maximum_skew,
        )
        row = {
            "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
            "safety_fault": safety,
            "raw_snapshot": dict(snapshot),
            **classification,
        }
        observations.append(row)
        if callable(on_sample):
            try:
                on_sample(row)
            except Exception as exc:
                terminal_reason = f"activation_sample_record_exception:{type(exc).__name__}:{exc}"
                lifecycle.rearm_forbidden = True
                lifecycle.fault = terminal_reason
                lifecycle.request_zero_and_stop(now_s=float(clock()), reason=terminal_reason)
                break
        lifecycle.observe_locomotion_readback(classification, now_s=float(clock()))
        if safety:
            terminal_reason = f"activation_safety_gate:{safety}"
            lifecycle.rearm_forbidden = True
            lifecycle.fault = terminal_reason
            lifecycle.request_zero_and_stop(now_s=float(clock()), reason=terminal_reason)
            break
        if lifecycle.state == LOCOMOTION_CONFIRMED:
            terminal_reason = "LOCOMOTION_CONFIRMED_BY_ZERO_ONLY_OBSERVATION"
            break
        if lifecycle.state == ZERO_AND_STOP_REQUESTED:
            terminal_reason = lifecycle.fault or classification.get("reason") or "activation_readback_rejected"
            break
        remaining = deadline - float(clock())
        if remaining > 0.0:
            sleep(min(0.005, remaining))

    if lifecycle.state == LOCOMOTION_REQUESTED:
        terminal_reason = terminal_reason or "LOCOMOTION_ACTIVATION_TIMEOUT_WITH_ZERO_ONLY_COMMANDS"
        lifecycle.expire_activation(now_s=float(clock()), reason=terminal_reason)
    return {
        "status": "PASS" if lifecycle.state == LOCOMOTION_CONFIRMED else "BLOCKED",
        "classification": "ZERO_ONLY_MUJOCO_MC_MODE_TRANSITION_VERIFICATION",
        "activation_wait_s": max(0.0, float(clock()) - started),
        "activation_timeout_s": duration,
        "publish_hz": frequency,
        "zero_publish_calls_returned": zero_calls,
        "nonzero_publish_calls_returned": 0,
        "publish_errors": publish_errors,
        "terminal_reason": terminal_reason,
        "final_lifecycle": lifecycle.snapshot(),
        "observations": observations,
    }
