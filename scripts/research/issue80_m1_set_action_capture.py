#!/usr/bin/env python3
"""Pre-arm one ROS 2 observer/client and run the source-supported X2 stand handoff.

The process starts before the GUI Reset, subscribes to the same live state used
by the standing gate, and creates SetMcAction/GetMcAction clients before Reset.
After a verified Reset event it performs one PD -> JD -> SD sequence:

* request JOINT_DEFAULT once after the existing fresh upright pose gate;
* request STAND_DEFAULT once only after the MC reports JOINT_DEFAULT and its
  installed McBodyPoseStatus.STAND value with fresh, safe odometry;
* never retry a mutating request and never infer success from the service reply
  alone.

This script is simulation-only. It does not write robot state or alter models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from issue80_m1_first_motion import PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S


REQUIRED_TOPICS = frozenset({"odom", "imu", "common", "leg_state", "leg_command"})
ROOT_Z_MIN_M = 0.52
MAX_TILT_DEG = 15.0
MAX_STATE_AGE_S = 0.5
REQUIRED_UPRIGHT_ODOM_SAMPLES = 3
REQUIRED_STAND_COMMON_SAMPLES = 3
NATIVE_SUPPORT_REQUIRED_SAMPLES = 2


def _native_pelvis_measurements(sample: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract fail-closed world-frame pelvis quantities from native MjData."""
    pelvis = sample.get("pelvis")
    state = pelvis.get("state", {}) if isinstance(pelvis, Mapping) else {}
    position = state.get("world_position_m") if isinstance(state, Mapping) else None
    quaternion = state.get("world_quaternion_wxyz") if isinstance(state, Mapping) else None
    velocity = state.get("world_spatial_velocity_rot_lin") if isinstance(state, Mapping) else None
    result: dict[str, float | None] = {
        "pelvis_x_m": None,
        "pelvis_y_m": None,
        "pelvis_root_z_m": None,
        "pelvis_roll_deg": None,
        "pelvis_pitch_deg": None,
        "pelvis_tilt_deg": None,
        "pelvis_planar_speed_world_m_s": None,
        "pelvis_vertical_speed_world_m_s": None,
    }
    try:
        if isinstance(position, (list, tuple)) and len(position) == 3:
            x, y, z = (float(value) for value in position)
            if all(math.isfinite(value) for value in (x, y, z)):
                result.update({"pelvis_x_m": x, "pelvis_y_m": y, "pelvis_root_z_m": z})
        if isinstance(quaternion, (list, tuple)) and len(quaternion) == 4:
            w, x, y, z = (float(value) for value in quaternion)
            if all(math.isfinite(value) for value in (w, x, y, z)):
                norm = math.sqrt(w * w + x * x + y * y + z * z)
                if norm > 0.0:
                    w, x, y, z = (value / norm for value in (w, x, y, z))
                    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
                    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
                    result.update({
                        "pelvis_roll_deg": roll,
                        "pelvis_pitch_deg": pitch,
                        "pelvis_tilt_deg": max(abs(roll), abs(pitch)),
                    })
        if isinstance(velocity, (list, tuple)) and len(velocity) == 6:
            vx, vy, vz = (float(velocity[index]) for index in (3, 4, 5))
            if all(math.isfinite(value) for value in (vx, vy, vz)):
                result.update({
                    "pelvis_planar_speed_world_m_s": math.hypot(vx, vy),
                    "pelvis_vertical_speed_world_m_s": vz,
                })
    except (TypeError, ValueError):
        pass
    return result


def native_foot_support(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one captured native-MjData sample without inventing contacts.

    The capture hook records only selected ankle-roll-link to floor contacts.
    A supported sample requires at least one positive normal force on both
    sides.  It is an initialization barrier, not a claim that force-plate
    calibration or active standing has been validated.
    """
    contacts = sample.get("contacts")
    side_normals = {"left": [], "right": []}
    if isinstance(contacts, list):
        for contact in contacts:
            if not isinstance(contact, Mapping):
                continue
            try:
                normal = float(contact.get("normal_force_n"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(normal) or normal <= 0.0:
                continue
            for side in side_normals:
                if contact.get(f"involves_selected_{side}_foot") is True:
                    side_normals[side].append(normal)
    measurements = _native_pelvis_measurements(sample)
    planar_speed = measurements["pelvis_planar_speed_world_m_s"]
    vertical_speed = measurements["pelvis_vertical_speed_world_m_s"]
    root_z = measurements["pelvis_root_z_m"]
    tilt = measurements["pelvis_tilt_deg"]
    bilateral = bool(side_normals["left"] and side_normals["right"])
    # A contact impulse plus upward root velocity is not a stable handoff.
    # Reuse the established physical-stop vertical-speed ceiling and require
    # the native root itself to satisfy the existing height/tilt gates.  This
    # deliberately waits through the first impact/rebound, while preserving
    # the existing 0.20 m/s planar-speed safety ceiling.
    settled = (
        bilateral
        and planar_speed is not None and planar_speed <= 0.20
        and vertical_speed is not None
        and abs(vertical_speed) <= PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S
        and root_z is not None and root_z >= ROOT_Z_MIN_M
        and tilt is not None and tilt < MAX_TILT_DEG
    )
    return {
        "supported": bilateral,
        "safe_to_begin_ros_upright_gate": settled,
        "left_normal_force_n": sum(side_normals["left"]),
        "right_normal_force_n": sum(side_normals["right"]),
        "left_contact_count": len(side_normals["left"]),
        "right_contact_count": len(side_normals["right"]),
        "captured_foot_floor_contact_count": sample.get("captured_foot_floor_contact_count"),
        "contact_count": sample.get("contact_count"),
        **measurements,
        "vertical_speed_within_existing_physical_stop_limit": (
            vertical_speed is not None
            and abs(vertical_speed) <= PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S
        ),
        "native_root_within_existing_height_tilt_gate": (
            root_z is not None and root_z >= ROOT_Z_MIN_M
            and tilt is not None and tilt < MAX_TILT_DEG
        ),
    }


def native_root_within_existing_safety_gate(support: Mapping[str, Any]) -> bool:
    """Apply the unchanged root height/tilt/planar-speed safety limits."""
    try:
        return (
            float(support["pelvis_planar_speed_world_m_s"]) <= 0.20
            and float(support["pelvis_root_z_m"]) >= ROOT_Z_MIN_M
            and float(support["pelvis_tilt_deg"]) < MAX_TILT_DEG
        )
    except (KeyError, TypeError, ValueError):
        return False


def native_synchronized_keyframe_state(
    sample: Mapping[str, Any], expected: Mapping[str, Any], *, tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Verify the native Load key result before physics is released.

    Positive contact force is intentionally not required: the paused native
    viewer has run ``mj_forward`` but has not integrated a contact impulse.
    The qpos/qvel identity, tangent bilateral contacts, and unchanged root
    safety limits are authoritative for this pre-step gate.
    """
    try:
        qpos = [float(value) for value in sample["qpos"]]
        qvel = [float(value) for value in sample["qvel"]]
        expected_qpos = [float(value) for value in expected["qpos"]]
        expected_qvel = [float(value) for value in expected["qvel"]]
    except (KeyError, TypeError, ValueError):
        return {"eligible": False, "failure_reason": "missing_or_non_numeric_keyframe_state"}
    qpos_match = (
        len(qpos) == len(expected_qpos)
        and all(math.isfinite(value) for value in qpos)
        and max((abs(a - b) for a, b in zip(qpos, expected_qpos)), default=math.inf) <= tolerance
    )
    qvel_match = (
        len(qvel) == len(expected_qvel)
        and all(math.isfinite(value) for value in qvel)
        and max((abs(a - b) for a, b in zip(qvel, expected_qvel)), default=math.inf) <= tolerance
    )
    support = native_foot_support(sample)
    tangent = {"left": False, "right": False}
    for contact in sample.get("contacts", []):
        if not isinstance(contact, Mapping):
            continue
        try:
            distance = float(contact.get("distance_m"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance) or abs(distance) > tolerance:
            continue
        for side in tangent:
            if contact.get(f"involves_selected_{side}_foot") is True:
                tangent[side] = True
    eligible = (
        qpos_match and qvel_match and native_root_within_existing_safety_gate(support)
        and all(tangent.values())
    )
    return {
        "eligible": eligible,
        "qpos_matches_expected": qpos_match,
        "qvel_matches_expected": qvel_match,
        "tangent_selected_foot_contacts": tangent,
        "native_root_within_existing_safety_gate": native_root_within_existing_safety_gate(support),
        "support": support,
        "failure_reason": (
            None if eligible else "synchronized_keyframe_state_not_verified"
        ),
    }


def ros_leg_state_matches_synchronized_keyframe(
    leg_state: Mapping[str, Any], expected: Mapping[str, Any], *, tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Require the paused simulator's published leg state to match the loaded keyframe.

    The native ``MjData`` keyframe check is the physical authority.  This
    companion check proves that the simulator's ROS joint-state publisher has
    observed the same source-MC nominal leg configuration before JD is sent.
    It deliberately does not infer an odometry frame equivalence from a paused
    sample, because the installed odom convention has not been established.
    """
    try:
        names = [str(value) for value in expected["source_nominal_joint_names"]]
        values = [float(value) for value in expected["source_nominal_joint_values_rad"]]
        joints = leg_state["joints"]
    except (KeyError, TypeError, ValueError):
        return {
            "eligible": False,
            "failure_reason": "missing_or_invalid_ros_leg_state_or_keyframe_spec",
        }
    if len(names) != len(values) or len(names) < 12 or not isinstance(joints, list):
        return {
            "eligible": False,
            "failure_reason": "invalid_ros_leg_state_or_keyframe_joint_inventory",
        }
    expected_legs = dict(zip(names[:12], values[:12]))
    observed: dict[str, float] = {}
    for joint in joints:
        if not isinstance(joint, Mapping):
            continue
        try:
            name = str(joint["name"])
            position = float(joint["position"])
        except (KeyError, TypeError, ValueError):
            continue
        if name in expected_legs and math.isfinite(position):
            observed[name] = position
    missing = sorted(set(expected_legs) - set(observed))
    residuals = {
        name: abs(observed[name] - target)
        for name, target in expected_legs.items()
        if name in observed
    }
    maximum_residual = max(residuals.values(), default=math.inf)
    eligible = not missing and maximum_residual <= tolerance
    return {
        "eligible": eligible,
        "expected_leg_joint_count": len(expected_legs),
        "observed_matching_leg_joint_count": len(observed),
        "missing_leg_joint_names": missing,
        "maximum_abs_position_residual_rad": maximum_residual,
        "position_tolerance_rad": tolerance,
        "failure_reason": None if eligible else "ros_leg_state_does_not_match_loaded_nominal_keyframe",
    }


def read_complete_native_capture_rows(
    path: Path, offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Read newly completed append-only native capture rows once.

    A partial final JSON record remains unread until the hook writes its
    newline.  This prevents a Reset observer from treating an interrupted
    append as a state-generation transition.
    """
    if offset < 0:
        raise ValueError("native capture offset must be non-negative")
    if not path.exists():
        return offset, []
    if path.is_symlink() or not path.is_file():
        raise ValueError("native capture trace is not a regular file")
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read()
        next_offset = stream.tell()
    rows: list[dict[str, Any]] = []
    for raw in payload.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            next_offset -= len(raw)
            break
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("native capture contains an invalid complete JSON row") from exc
        if not isinstance(decoded, dict):
            raise ValueError("native capture contains a non-object row")
        rows.append(decoded)
    return next_offset, rows


def standing_observation_window(
    *,
    reset_dispatch_monotonic_ns: int,
    stand_default_confirmed_monotonic_ns: int | None,
    duration_s: float,
) -> dict[str, int | str]:
    """Anchor the baseline window to positive STAND readback when available."""
    if (
        isinstance(reset_dispatch_monotonic_ns, bool)
        or not isinstance(reset_dispatch_monotonic_ns, int)
        or reset_dispatch_monotonic_ns <= 0
        or not math.isfinite(float(duration_s))
        or duration_s <= 0.0
    ):
        raise ValueError("standing observation requires a valid Reset timestamp and positive duration")
    if stand_default_confirmed_monotonic_ns is None:
        anchor_ns = reset_dispatch_monotonic_ns
        anchor_name = "RESET_FALLBACK_NO_CONFIRMED_STAND"
    else:
        if (
            isinstance(stand_default_confirmed_monotonic_ns, bool)
            or not isinstance(stand_default_confirmed_monotonic_ns, int)
            or stand_default_confirmed_monotonic_ns < reset_dispatch_monotonic_ns
        ):
            raise ValueError("STAND_DEFAULT confirmation must follow the Reset event")
        anchor_ns = stand_default_confirmed_monotonic_ns
        anchor_name = "STAND_DEFAULT_GETMC_READBACK"
    duration_ns = int(float(duration_s) * 1e9)
    return {
        "anchor": anchor_name,
        "anchor_monotonic_ns": anchor_ns,
        "deadline_monotonic_ns": anchor_ns + duration_ns,
        "duration_ns": duration_ns,
    }


def write_complete_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one complete result atomically, without replacing prior evidence."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {name: json_safe(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def odom_pose(msg: Any) -> dict[str, Any]:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return {
        "x": float(p.x), "y": float(p.y), "z": float(p.z),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),
        "linear_velocity": json_safe(msg.twist.twist.linear),
        "angular_velocity": json_safe(msg.twist.twist.angular),
    }


def pose_is_upright(data: dict[str, Any]) -> bool:
    try:
        return (
            float(data["z"]) >= ROOT_Z_MIN_M
            and max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"]))) < MAX_TILT_DEG
        )
    except (KeyError, TypeError, ValueError):
        return False


def command_is_effective(data: dict[str, Any]) -> bool:
    """Whether any leg-command target/gain field is nonzero."""
    joints = data.get("joints", [])
    if not isinstance(joints, list):
        return False
    fields = ("position", "velocity", "effort", "stiffness", "damping")
    for joint in joints:
        if not isinstance(joint, dict):
            continue
        for field in fields:
            try:
                if abs(float(joint.get(field, 0.0) or 0.0)) > 1e-12:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def installed_body_stand_value(message_type: Any) -> int:
    """Read the STAND constant from the installed generated ROS message type."""
    instance = message_type()
    value = getattr(instance, "MC_BODY_POSE_STAND", None)
    if value is None:
        raise ValueError("installed McBodyPoseStatus has no MC_BODY_POSE_STAND constant")
    return int(value)


def _raw_fsm_state_value(common_state: Mapping[str, Any]) -> Any:
    """Preserve the installed CommonState FSM field without assigning semantics."""
    fsm_state = common_state.get("fsm_state")
    if isinstance(fsm_state, Mapping):
        return fsm_state.get("current_state", fsm_state.get("value"))
    return fsm_state


def assess_pre_action_state(
    latest: dict[str, dict[str, Any]],
    *,
    now_monotonic_ns: int,
    upright_odom_samples: int,
    service_ready: bool,
    reset_dispatch_monotonic_ns: int | None = None,
    required_upright_odom_samples: int = REQUIRED_UPRIGHT_ODOM_SAMPLES,
) -> dict[str, Any]:
    """Apply the existing freshness/upright gate, optionally after Reset only."""
    post_reset = {
        key: row for key, row in latest.items()
        if reset_dispatch_monotonic_ns is None
        or int(row.get("wall_monotonic_ns", 0)) > reset_dispatch_monotonic_ns
    }
    ages = {
        key: (now_monotonic_ns - int(row["wall_monotonic_ns"])) / 1e9
        for key, row in post_reset.items()
    }
    fresh = REQUIRED_TOPICS.issubset(post_reset) and all(
        0.0 <= ages[key] <= MAX_STATE_AGE_S for key in REQUIRED_TOPICS
    )
    odom = post_reset.get("odom", {}).get("data", {})
    pose_ok = pose_is_upright(odom)
    if not fresh:
        reason = "required_post_reset_state_missing_or_stale"
    elif not pose_ok:
        reason = "live_posture_outside_existing_gate"
    elif upright_odom_samples < required_upright_odom_samples:
        reason = "upright_odom_samples_insufficient"
    elif not service_ready:
        reason = "SetMcAction_service_not_ready"
    else:
        reason = None
    return {
        "eligible": reason is None,
        "failure_reason": reason,
        "state_ages_s": ages,
        "fresh_required_topics": fresh,
        "post_reset_required_topics": sorted(post_reset),
        "posture_gate_pass": pose_ok,
        "upright_odom_samples": upright_odom_samples,
        "required_upright_odom_samples": required_upright_odom_samples,
        "service_ready": service_ready,
        "pose": odom if odom else None,
        "raw_common_state": post_reset.get("common", {}).get("data"),
        "imu": post_reset.get("imu", {}).get("data"),
        "leg_state": post_reset.get("leg_state", {}).get("data"),
        "leg_command": post_reset.get("leg_command", {}).get("data"),
    }


def assess_joint_to_stand_state(
    latest: dict[str, dict[str, Any]],
    *,
    now_monotonic_ns: int,
    reset_dispatch_monotonic_ns: int,
    upright_odom_samples: int,
    stand_common_samples: int,
    set_service_ready: bool,
    get_service_ready: bool,
    get_mode_desc: str | None,
    get_mode_status: int | None,
    body_stand_value: int,
    required_upright_odom_samples: int = REQUIRED_UPRIGHT_ODOM_SAMPLES,
) -> dict[str, Any]:
    """Gate SD on fresh physical posture and the installed MC stand classifier."""
    base = assess_pre_action_state(
        latest,
        now_monotonic_ns=now_monotonic_ns,
        upright_odom_samples=upright_odom_samples,
        service_ready=set_service_ready and get_service_ready,
        reset_dispatch_monotonic_ns=reset_dispatch_monotonic_ns,
        required_upright_odom_samples=required_upright_odom_samples,
    )
    common = base.get("raw_common_state") or {}
    action = common.get("action_info", {}) if isinstance(common, dict) else {}
    body = common.get("body_status", {}) if isinstance(common, dict) else {}
    action_desc = action.get("action_desc")
    action_status = action.get("status", {}).get("value")
    body_status = body.get("value")
    body_is_stand = body_status == body_stand_value
    mode_is_joint = action_desc == "JOINT_DEFAULT" and action_status == 100
    get_readback_is_joint = get_mode_desc == "JOINT_DEFAULT" and get_mode_status == 100
    if not base["fresh_required_topics"]:
        reason = base["failure_reason"]
    elif not base["posture_gate_pass"]:
        reason = base["failure_reason"]
    elif not mode_is_joint or not get_readback_is_joint:
        reason = "JOINT_DEFAULT_not_confirmed_by_common_state_and_GetMcAction"
    elif not body_is_stand:
        reason = "MC_body_status_not_STAND"
    elif stand_common_samples < REQUIRED_STAND_COMMON_SAMPLES:
        reason = "fresh_STAND_body_status_samples_insufficient"
    elif not set_service_ready or not get_service_ready:
        reason = "required_MC_service_not_ready"
    else:
        reason = None
    return {
        "eligible": reason is None,
        "failure_reason": reason,
        "state_ages_s": base["state_ages_s"],
        "fresh_required_topics": base["fresh_required_topics"],
        "posture_gate_pass": base["posture_gate_pass"],
        "pose": base["pose"],
        "upright_odom_samples": upright_odom_samples,
        "stand_common_samples": stand_common_samples,
        "required_stand_common_samples": REQUIRED_STAND_COMMON_SAMPLES,
        "body_stand_value_from_installed_aimdk_msgs": body_stand_value,
        "raw_body_status": body_status,
        "raw_action_desc": action_desc,
        "raw_action_status": action_status,
        "GetMcAction_action_desc": get_mode_desc,
        "GetMcAction_status": get_mode_status,
        "set_service_ready": set_service_ready,
        "get_service_ready": get_service_ready,
        "raw_common_state": common,
        "imu": base["imu"],
        "leg_state": base["leg_state"],
        "leg_command": base["leg_command"],
    }


def set_action_record(
    request: Any,
    response: Any,
    *,
    success_status: int,
    submitted_utc: str,
    submitted_monotonic_ns: int,
    returned_utc: str,
    returned_monotonic_ns: int,
) -> dict[str, Any]:
    """Keep full nested CommonResponse and explicit status/code fields."""
    common = response.response
    status_value = int(common.status.value)
    header_code = int(common.header.code)
    return {
        "transport_state": "RESPONSE_RECEIVED",
        "wall_utc_submitted": submitted_utc,
        "wall_monotonic_ns_submitted": submitted_monotonic_ns,
        "wall_utc_returned": returned_utc,
        "wall_monotonic_ns_returned": returned_monotonic_ns,
        "request": json_safe(request),
        "response": json_safe(common),
        "response_status_value": status_value,
        "response_header_code": header_code,
        "response_message": str(common.message),
        "service_success": status_value == int(success_status),
        "action_result_success": header_code == 0,
    }


def get_action_record(
    request: Any,
    response: Any,
    *,
    submitted_utc: str,
    submitted_monotonic_ns: int,
    returned_utc: str,
    returned_monotonic_ns: int,
) -> dict[str, Any]:
    info = response.info
    return {
        "transport_state": "RESPONSE_RECEIVED",
        "wall_utc_submitted": submitted_utc,
        "wall_monotonic_ns_submitted": submitted_monotonic_ns,
        "wall_utc_returned": returned_utc,
        "wall_monotonic_ns_returned": returned_monotonic_ns,
        "request": json_safe(request),
        "response": json_safe(response),
        "response_header_code": int(response.header.code),
        "action_desc": str(info.action_desc),
        "action_status_value": int(info.status.value),
    }


def elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return (int(end_ns) - int(start_ns)) / 1e6


def summarize_takeover_timing(events: dict[str, Any]) -> dict[str, Any]:
    """Compute wall-monotonic takeover intervals without mixing ROS time."""
    reset_ns = events.get("reset_dispatch_monotonic_ns")
    first_valid_ns = events.get("first_valid_post_reset_state_monotonic_ns")
    first_upright_ns = events.get("first_upright_post_reset_monotonic_ns")
    jd = events.get("joint_default", {}) or {}
    sd = events.get("stand_default", {}) or {}
    first_cmd_jd = events.get("first_effective_leg_command_after_jd_monotonic_ns")
    first_cmd_sd = events.get("first_effective_leg_command_after_sd_monotonic_ns")
    unsafe = events.get("first_unsafe_post_reset_monotonic_ns")
    native_reset_ns = events.get("native_reset_generation_host_monotonic_ns")
    native_support_ns = events.get("native_support_ready_host_monotonic_ns")
    return {
        "reset_dispatch_to_native_generation_ms": elapsed_ms(reset_ns, native_reset_ns),
        "native_generation_to_bilateral_support_ms": elapsed_ms(native_reset_ns, native_support_ns),
        "native_support_to_JD_request_ms": elapsed_ms(native_support_ns, jd.get("wall_monotonic_ns_submitted")),
        "reset_to_first_valid_state_ms": elapsed_ms(reset_ns, first_valid_ns),
        "reset_to_first_upright_sample_ms": elapsed_ms(reset_ns, first_upright_ns),
        "reset_to_JD_request_ms": elapsed_ms(reset_ns, jd.get("wall_monotonic_ns_submitted")),
        "JD_request_to_response_ms": elapsed_ms(jd.get("wall_monotonic_ns_submitted"), jd.get("wall_monotonic_ns_returned")),
        "JD_response_to_first_effective_leg_command_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), first_cmd_jd),
        "JD_response_to_MC_STAND_classification_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), events.get("body_stand_ready_monotonic_ns")),
        "reset_to_SD_request_ms": elapsed_ms(reset_ns, sd.get("wall_monotonic_ns_submitted")),
        "SD_request_to_response_ms": elapsed_ms(sd.get("wall_monotonic_ns_submitted"), sd.get("wall_monotonic_ns_returned")),
        "SD_response_to_first_effective_leg_command_ms": elapsed_ms(sd.get("wall_monotonic_ns_returned"), first_cmd_sd),
        "reset_to_first_unsafe_posture_ms": elapsed_ms(reset_ns, unsafe),
        "JD_response_to_first_unsafe_posture_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), unsafe),
        "SD_response_to_first_unsafe_posture_ms": elapsed_ms(sd.get("wall_monotonic_ns_returned"), unsafe),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="STAND_DEFAULT")
    parser.add_argument("--source", default="node.set_mc_action")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reset-event-file", type=Path, default=None)
    parser.add_argument(
        "--static-nominal-start-event-file", type=Path, default=None,
        help=(
            "parent-written pre-physics static-nominal-start barrier; this path "
            "does not dispatch or claim a native MuJoCo Reset"
        ),
    )
    parser.add_argument(
        "--synchronized-keyframe-event-file", type=Path, default=None,
        help="native MuJoCo Load key dispatch record for the paused synchronized-init path",
    )
    parser.add_argument(
        "--synchronized-keyframe-spec", type=Path, default=None,
        help="read-only source-MC-derived keyframe specification staged by the parent",
    )
    parser.add_argument(
        "--physics-release-event-file", type=Path, default=None,
        help="parent-written verified native Play/Pause release event for synchronized initialization",
    )
    parser.add_argument(
        "--synchronized-ready-file", type=Path, default=None,
        help="child-written JD-control readiness barrier consumed by the parent before unpausing physics",
    )
    parser.add_argument("--set-timeout-s", type=float, default=2.0)
    parser.add_argument("--get-timeout-s", type=float, default=0.75)
    parser.add_argument("--ready-timeout-s", type=float, default=3.0)
    parser.add_argument("--reset-wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--upright-timeout-s", type=float, default=4.0)
    parser.add_argument("--joint-stand-timeout-s", type=float, default=4.0)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument(
        "--native-mjdata-trace", type=Path, default=None,
        help="append-only read-only native MjData capture trace staged by the parent runner",
    )
    parser.add_argument(
        "--require-native-reset-generation",
        action="store_true",
        help="require a fresh native reset generation and bilateral foot-floor support before JD",
    )
    args = parser.parse_args()
    limits = (
        (args.set_timeout_s, 0.0, 4.0, "--set-timeout-s"),
        (args.get_timeout_s, 0.0, 2.0, "--get-timeout-s"),
        (args.ready_timeout_s, 0.0, 5.0, "--ready-timeout-s"),
        (args.reset_wait_timeout_s, 0.0, 60.0, "--reset-wait-timeout-s"),
        (args.upright_timeout_s, 0.0, 5.0, "--upright-timeout-s"),
        (args.joint_stand_timeout_s, 0.0, 6.0, "--joint-stand-timeout-s"),
        (args.duration_s, 0.0, 30.0, "--duration-s"),
    )
    for value, lower, upper, name in limits:
        if not math.isfinite(value) or value <= lower or value > upper:
            parser.error(f"{name} must be in ({lower}, {upper}]")
    if args.action != "STAND_DEFAULT":
        parser.error("this capture only supports the reviewed PD -> JD -> SD standing handoff")
    if args.require_native_reset_generation and args.native_mjdata_trace is None:
        parser.error("--require-native-reset-generation requires --native-mjdata-trace")
    synchronized_paths = (
        args.synchronized_keyframe_event_file,
        args.synchronized_keyframe_spec,
        args.physics_release_event_file,
        args.synchronized_ready_file,
    )
    synchronized = any(path is not None for path in synchronized_paths)
    static_nominal_start = args.static_nominal_start_event_file is not None
    if static_nominal_start and (synchronized or args.reset_event_file is not None):
        parser.error("static nominal-start initialization is mutually exclusive with Reset and keyframe initialization")
    if synchronized:
        if any(path is None for path in synchronized_paths):
            parser.error("synchronized initialization requires all keyframe/release/ready paths")
        if args.reset_event_file is not None:
            parser.error("synchronized initialization cannot also consume a Reset event")
        if not args.require_native_reset_generation:
            parser.error("synchronized initialization requires native MjData capture generation evidence")
    elif not static_nominal_start and args.reset_event_file is None:
        parser.error("--reset-event-file is required outside synchronized initialization")

    import rclpy
    from rclpy.node import Node
    from aimdk_msgs.msg import (
        CommonRequest, CommonState, JointCommandArray, JointStateArray,
        McActionCommand, McBodyPoseStatus, McCommonState, RequestHeader,
    )
    from aimdk_msgs.srv import GetMcAction, SetMcAction
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Imu

    try:
        body_stand_value = installed_body_stand_value(McBodyPoseStatus)
    except (TypeError, ValueError, AttributeError) as exc:
        print(json.dumps({
            "status": "BLOCKED_INSTALLED_BODY_STATUS_CONSTANT_MISSING",
            "expected_constant": "aimdk_msgs.msg.McBodyPoseStatus.MC_BODY_POSE_STAND",
            "error": f"{type(exc).__name__}: {exc}",
        }, separators=(",", ":")), flush=True)
        return 2

    class TakeoverCapture(Node):
        def __init__(self) -> None:
            super().__init__("issue80_m1_standing_takeover_capture")
            self.output_dir = args.output_dir.resolve(strict=True)
            self.synchronized_initialization = bool(args.synchronized_keyframe_event_file)
            self.static_nominal_start_initialization = bool(
                args.static_nominal_start_event_file
            )
            self.reset_event_file = (
                args.reset_event_file.resolve(strict=False)
                if args.reset_event_file is not None else None
            )
            self.synchronized_keyframe_event_file = (
                args.synchronized_keyframe_event_file.resolve(strict=False)
                if args.synchronized_keyframe_event_file is not None else None
            )
            self.static_nominal_start_event_file = (
                args.static_nominal_start_event_file.resolve(strict=False)
                if args.static_nominal_start_event_file is not None else None
            )
            self.physics_release_event_file = (
                args.physics_release_event_file.resolve(strict=False)
                if args.physics_release_event_file is not None else None
            )
            self.synchronized_ready_file = (
                args.synchronized_ready_file.resolve(strict=False)
                if args.synchronized_ready_file is not None else None
            )
            self.synchronized_keyframe_spec = self._load_synchronized_keyframe_spec(
                args.synchronized_keyframe_spec
            ) if self.synchronized_initialization else None
            self.synchronized_ros_leg_state_ready: dict[str, Any] | None = None
            self.trace_path = self.output_dir / "reset_state_trace.jsonl"
            self.ready_path = self.output_dir / "reset_watch_ready.json"
            self.summary_path = self.output_dir / "reset_state_summary.json"
            self.result_path = self.output_dir / "set_action_capture_result.json"
            self.trace = self.trace_path.open("x", encoding="utf-8")
            self.started_utc = datetime.now(timezone.utc).isoformat()
            self.started_ns = time.monotonic_ns()
            self.latest: dict[str, dict[str, Any]] = {}
            self.counts: Counter[str] = Counter()
            self.clocks: list[dict[str, Any]] = []
            self.reset_dispatch: dict[str, Any] | None = None
            self.reset_ns: int | None = None
            self.physics_release: dict[str, Any] | None = None
            self.physics_release_ns: int | None = None
            self.post_reset_epoch_ns: int | None = None
            self.native_trace_path = (
                args.native_mjdata_trace.resolve(strict=False)
                if args.native_mjdata_trace is not None else None
            )
            self.require_native_reset_generation = bool(args.require_native_reset_generation)
            self.required_post_reset_upright_samples = (
                1 if self.require_native_reset_generation else REQUIRED_UPRIGHT_ODOM_SAMPLES
            )
            self.native_trace_offset = 0
            self.native_baseline_generation: int | None = None
            self.native_reset_generation: int | None = None
            self.native_reset_host_monotonic_ns: int | None = None
            self.native_reset_sample: dict[str, Any] | None = None
            self.native_support_samples = 0
            self.native_support_ready_ns: int | None = None
            self.native_support_sample: dict[str, Any] | None = None
            self.native_keyframe_ready_ns: int | None = None
            self.native_keyframe_sample: dict[str, Any] | None = None
            self.native_keyframe_assessment: dict[str, Any] | None = None
            self.first_native_step_after_release: dict[str, Any] | None = None
            self.first_native_actuation_after_release: dict[str, Any] | None = None
            self.post_jd_control_epoch_ns: int | None = None
            self.native_handoff_safety_lost: dict[str, Any] | None = None
            self.native_unsafe_after_jd: dict[str, Any] | None = None
            self.pending_jd_submission_ns: int | None = None
            self.first_valid_state_ns: int | None = None
            self.first_upright_ns: int | None = None
            self.first_unsafe_ns: int | None = None
            self.first_unsafe_row: dict[str, Any] | None = None
            self.first_effective_after_jd: int | None = None
            self.first_effective_after_sd: int | None = None
            self.stand_default_confirmed_ns: int | None = None
            self.upright_samples = 0
            self.stand_samples = 0
            self.set_attempted: set[str] = set()
            self.set_records: dict[str, dict[str, Any]] = {}
            self.get_records: list[dict[str, Any]] = []
            self.get_mode_desc: str | None = None
            self.get_mode_status: int | None = None
            self.result: dict[str, Any] | None = None
            self.error: str | None = None
            self.observer_ready = False
            self.qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.set_client = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")
            self.get_client = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
            self.topic_specs = (
                ("odom", "/aima/hal/odom/state", Odometry),
                ("imu", "/aima/hal/imu/torso/state", Imu),
                ("common", "/aima/mc/common/state", McCommonState),
                ("leg_state", "/aima/hal/joint/leg/state", JointStateArray),
                ("leg_command", "/aima/hal/joint/leg/command", JointCommandArray),
            )
            for key, topic, msg_type in self.topic_specs:
                self.create_subscription(
                    msg_type, topic, lambda msg, name=key: self.store(name, msg), self.qos
                )
            self.create_subscription(Clock, "/clock", self.store_clock, self.qos)
            self.subscriptions_created_utc = datetime.now(timezone.utc).isoformat()
            self.subscriptions_created_ns = time.monotonic_ns()

        @staticmethod
        def _load_synchronized_keyframe_spec(path_value: Path | None) -> dict[str, Any]:
            if path_value is None:
                raise ValueError("synchronized keyframe spec path is missing")
            path = path_value.resolve(strict=True)
            if path.is_symlink() or not path.is_file():
                raise ValueError("synchronized keyframe spec is not a regular file")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                qpos = [float(item) for item in value["qpos"]]
                qvel = [float(item) for item in value["qvel"]]
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("synchronized keyframe spec is invalid") from exc
            if (
                value.get("schema") != "robotsim.issue80.synchronized-nominal-keyframe.v1"
                or value.get("key_name") != "issue80_nominal_stand_v1"
                or not qpos or not qvel
                or not all(math.isfinite(item) for item in qpos + qvel)
            ):
                raise ValueError("synchronized keyframe spec violates the reviewed schema")
            return {**value, "qpos": qpos, "qvel": qvel, "path": str(path)}

        def write_trace(self, row: dict[str, Any]) -> None:
            self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self.trace.flush()

        def _post_reset_epoch(self) -> int | None:
            return self.post_reset_epoch_ns if self.post_reset_epoch_ns is not None else self.reset_ns

        def _assessment_epoch(self) -> int | None:
            """Ignore queued pre-JD telemetry after a native MjData handoff."""
            return self.physics_release_ns or self.post_jd_control_epoch_ns or self._post_reset_epoch()

        def _clear_assessment_counters(self) -> None:
            self.upright_samples = 0
            self.stand_samples = 0
            self.first_valid_state_ns = None
            self.first_upright_ns = None
            self.first_unsafe_ns = None
            self.first_unsafe_row = None

        def ros_transport_liveness(self, now_ns: int) -> dict[str, Any]:
            """Check delivery freshness without treating queued ROS data as current physics."""
            rows = {key: self.latest[key] for key in REQUIRED_TOPICS if key in self.latest}
            ages = {
                key: (now_ns - int(row.get("wall_monotonic_ns", 0))) / 1e9
                for key, row in rows.items()
            }
            fresh = REQUIRED_TOPICS.issubset(rows) and all(
                0.0 <= ages[key] <= MAX_STATE_AGE_S for key in REQUIRED_TOPICS
            )
            return {
                "fresh_required_topics": fresh,
                "state_ages_s": ages,
                "topic_keys": sorted(rows),
                "interpretation": (
                    "transport_liveness_only; physical JD eligibility is native_MjData "
                    "until a post-JD effective leg-command epoch is observed"
                ),
            }

        def poll_native_capture(self) -> None:
            """Consume only completed evidence-local native samples once."""
            if self.native_trace_path is None:
                return
            self.native_trace_offset, rows = read_complete_native_capture_rows(
                self.native_trace_path, self.native_trace_offset
            )
            for row in rows:
                generation = row.get("generation")
                host_ns = row.get("host_monotonic_ns")
                if not isinstance(generation, int) or not isinstance(host_ns, int):
                    continue
                if self.reset_ns is None:
                    if self.native_baseline_generation is None:
                        self.native_baseline_generation = generation
                    else:
                        self.native_baseline_generation = max(self.native_baseline_generation, generation)
                    continue
                if self.native_baseline_generation is None:
                    continue
                if self.native_reset_generation is None and generation > self.native_baseline_generation:
                    self.native_reset_generation = generation
                    self.native_reset_host_monotonic_ns = host_ns
                    self.native_reset_sample = row
                    self.record_event(
                        "native_reset_generation_observed",
                        generation=generation,
                        native_host_monotonic_ns=host_ns,
                        sample=row,
                    )
                if generation != self.native_reset_generation:
                    continue
                support = native_foot_support(row)
                if self.synchronized_initialization:
                    assert self.synchronized_keyframe_spec is not None
                    assessment = native_synchronized_keyframe_state(
                        row, self.synchronized_keyframe_spec
                    )
                    if self.native_keyframe_ready_ns is None and assessment["eligible"]:
                        self.native_keyframe_ready_ns = host_ns
                        self.native_keyframe_sample = row
                        self.native_keyframe_assessment = assessment
                        self.post_reset_epoch_ns = host_ns
                        self._clear_assessment_counters()
                        self.record_event(
                            "native_synchronized_keyframe_ready",
                            generation=generation,
                            native_host_monotonic_ns=host_ns,
                            assessment=assessment,
                        )
                else:
                    if support["safe_to_begin_ros_upright_gate"]:
                        self.native_support_samples += 1
                        self.native_support_sample = {"sample": row, "support": support}
                    else:
                        self.native_support_samples = 0
                    if (
                        self.native_support_ready_ns is None
                        and self.native_support_samples >= NATIVE_SUPPORT_REQUIRED_SAMPLES
                    ):
                        self.native_support_ready_ns = host_ns
                        self.post_reset_epoch_ns = host_ns
                        self._clear_assessment_counters()
                        self.record_event(
                            "native_bilateral_foot_support_ready",
                            generation=generation,
                            native_host_monotonic_ns=host_ns,
                            required_samples=NATIVE_SUPPORT_REQUIRED_SAMPLES,
                            required_post_reset_ros_upright_samples=self.required_post_reset_upright_samples,
                            support=self.native_support_sample,
                        )
                jd_ns = self.pending_jd_submission_ns or self.set_records.get(
                    "JOINT_DEFAULT", {}
                ).get("wall_monotonic_ns_submitted")
                native_ready_ns = (
                    self.native_keyframe_ready_ns
                    if self.synchronized_initialization else self.native_support_ready_ns
                )
                if (
                    native_ready_ns is not None
                    and jd_ns is None
                    and (not self.synchronized_initialization or self.physics_release_ns is not None)
                    and host_ns > native_ready_ns
                    and not native_root_within_existing_safety_gate(support)
                    and self.native_handoff_safety_lost is None
                ):
                    self.native_handoff_safety_lost = {"sample": row, "support": support}
                    self.record_event(
                        "native_pre_JD_physical_safety_fault",
                        native_host_monotonic_ns=host_ns,
                        support=self.native_handoff_safety_lost,
                    )
                if isinstance(jd_ns, int) and host_ns > jd_ns and (
                    not self.synchronized_initialization or self.physics_release_ns is not None
                ):
                    if not native_root_within_existing_safety_gate(support) and self.native_unsafe_after_jd is None:
                        self.native_unsafe_after_jd = {"sample": row, "support": support}
                        self.record_event(
                            "native_post_JD_physical_safety_fault",
                            native_host_monotonic_ns=host_ns,
                            support=self.native_unsafe_after_jd,
                        )
                if self.physics_release_ns is not None and host_ns > self.physics_release_ns:
                    if self.first_native_step_after_release is None and isinstance(row.get("mj_step_call_index"), int):
                        self.first_native_step_after_release = row
                        self.record_event(
                            "native_first_step_after_physics_release",
                            native_host_monotonic_ns=host_ns,
                            sample=row,
                        )
                    force = row.get("actuator_force", [])
                    try:
                        max_force = max(abs(float(value)) for value in force)
                    except (TypeError, ValueError):
                        max_force = 0.0
                    if self.first_native_actuation_after_release is None and max_force > 1e-8:
                        self.first_native_actuation_after_release = row
                        self.record_event(
                            "native_first_actuation_after_physics_release",
                            native_host_monotonic_ns=host_ns,
                            max_abs_actuator_force=max_force,
                            sample=row,
                        )

        def native_reset_support_record(self) -> dict[str, Any]:
            return {
                "required": self.require_native_reset_generation,
                "trace": str(self.native_trace_path) if self.native_trace_path else None,
                "baseline_generation": self.native_baseline_generation,
                "reset_generation": self.native_reset_generation,
                "reset_host_monotonic_ns": self.native_reset_host_monotonic_ns,
                "support_samples": self.native_support_samples,
                "support_ready_host_monotonic_ns": self.native_support_ready_ns,
                "support": self.native_support_sample,
                "synchronized_initialization": self.synchronized_initialization,
                "synchronized_keyframe_spec": self.synchronized_keyframe_spec,
                "synchronized_keyframe_ready_host_monotonic_ns": self.native_keyframe_ready_ns,
                "synchronized_keyframe_assessment": self.native_keyframe_assessment,
                "synchronized_ros_leg_state_ready": self.synchronized_ros_leg_state_ready,
                "physics_release_monotonic_ns": self.physics_release_ns,
                "first_native_step_after_physics_release": self.first_native_step_after_release,
                "first_native_actuation_after_physics_release": self.first_native_actuation_after_release,
                "required_post_reset_ros_upright_samples": self.required_post_reset_upright_samples,
                "post_JD_effective_command_epoch_monotonic_ns": self.post_jd_control_epoch_ns,
                "native_pre_JD_physical_safety_fault": self.native_handoff_safety_lost,
                "native_post_JD_physical_safety_fault": self.native_unsafe_after_jd,
            }

        def wait_for_native_reset_support(self, deadline: float) -> bool:
            if not self.require_native_reset_generation:
                return True
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                try:
                    self.poll_native_capture()
                except (OSError, ValueError) as exc:
                    self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                    return False
                if self.synchronized_initialization and self.native_keyframe_ready_ns is not None:
                    return True
                if not self.synchronized_initialization and self.native_support_ready_ns is not None:
                    return True
            if self.native_reset_generation is None:
                self.error = "native MjData did not observe a post-dispatch initialization generation"
            elif self.synchronized_initialization:
                self.error = "native MjData did not verify the source-MC nominal keyframe while physics was paused"
            else:
                self.error = "native MjData did not observe consecutive bilateral foot support after reset"
            return False

        def store(self, key: str, msg: Any) -> None:
            now_utc = datetime.now(timezone.utc).isoformat()
            now_ns = time.monotonic_ns()
            data = odom_pose(msg) if key == "odom" else json_safe(msg)
            row = {
                "event": "state",
                "topic_key": key,
                "wall_utc": now_utc,
                "wall_monotonic_ns": now_ns,
                "header_stamp": json_safe(msg.header.stamp) if hasattr(msg, "header") else None,
                "data": data,
            }
            self.latest[key] = row
            self.counts[key] += 1
            self.write_trace(row)
            epoch_ns = self._assessment_epoch()
            # Native capture is synchronous with the simulator.  Before JD
            # produces a real leg command, ROS can still be draining messages
            # generated before Reset; retain those rows as evidence but do not
            # classify their pose as the current physical state.
            can_assess_ros_physics = not (
                self.require_native_reset_generation
                and self.post_jd_control_epoch_ns is None
            )
            if can_assess_ros_physics and epoch_ns is not None and now_ns > epoch_ns and key == "odom":
                if pose_is_upright(data):
                    self.upright_samples += 1
                    if self.first_upright_ns is None:
                        self.first_upright_ns = now_ns
                else:
                    self.upright_samples = 0
                    if self.first_upright_ns is not None and self.first_unsafe_ns is None:
                        self.first_unsafe_ns = now_ns
                        self.first_unsafe_row = row
                if self.first_valid_state_ns is None and self.fresh_topics_after_reset(now_ns):
                    self.first_valid_state_ns = now_ns
            if can_assess_ros_physics and epoch_ns is not None and now_ns > epoch_ns and key == "common":
                common = data
                if (
                    common.get("action_info", {}).get("action_desc") == "JOINT_DEFAULT"
                    and int(common.get("action_info", {}).get("status", {}).get("value", -1)) == 100
                    and int(common.get("body_status", {}).get("value", -1)) == body_stand_value
                ):
                    self.stand_samples += 1
                else:
                    self.stand_samples = 0
            if key == "leg_command" and command_is_effective(data):
                if self.first_effective_after_jd is None:
                    response_ns = self.set_records.get("JOINT_DEFAULT", {}).get("wall_monotonic_ns_returned")
                    if response_ns is not None and now_ns > int(response_ns):
                        self.first_effective_after_jd = now_ns
                        if self.require_native_reset_generation:
                            self.post_jd_control_epoch_ns = now_ns
                            self._clear_assessment_counters()
                            self.record_event(
                                "post_JD_effective_leg_command_epoch",
                                wall_monotonic_ns=now_ns,
                                source="fresh_effective_leg_command",
                            )
                if self.first_effective_after_sd is None:
                    response_ns = self.set_records.get("STAND_DEFAULT", {}).get("wall_monotonic_ns_returned")
                    if response_ns is not None and now_ns > int(response_ns):
                        self.first_effective_after_sd = now_ns

        def store_clock(self, msg: Any) -> None:
            row = {
                "event": "clock",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                "clock": json_safe(msg.clock),
            }
            self.clocks.append(row)
            self.write_trace(row)

        def _post_reset_latest(self) -> dict[str, dict[str, Any]]:
            epoch_ns = self._assessment_epoch()
            if epoch_ns is None:
                return {}
            return {
                key: row for key, row in self.latest.items()
                if int(row.get("wall_monotonic_ns", 0)) > epoch_ns
            }

        def fresh_topics_after_reset(self, now_ns: int) -> bool:
            current = self._post_reset_latest()
            return REQUIRED_TOPICS.issubset(current) and all(
                0.0 <= (now_ns - int(current[key]["wall_monotonic_ns"])) / 1e9 <= MAX_STATE_AGE_S
                for key in REQUIRED_TOPICS
            )

        def record_event(self, event_name: str, **payload: Any) -> None:
            self.write_trace({
                "event": event_name,
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                **payload,
            })

        def write_json_once(self, path: Path, value: dict[str, Any]) -> None:
            write_complete_json(path, value)

        def spin_for(self, timeout_s: float) -> None:
            rclpy.spin_once(self, timeout_sec=max(0.0, timeout_s))

        def mode_readback(self, timeout_s: float, *, monitor_native_handoff: bool = False) -> dict[str, Any]:
            request = GetMcAction.Request()
            request.request = CommonRequest()
            request.request.header.stamp = self.get_clock().now().to_msg()
            submitted_utc = datetime.now(timezone.utc).isoformat()
            submitted_ns = time.monotonic_ns()
            future = self.get_client.call_async(request)
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                self.spin_for(min(0.01, max(0.0, deadline - time.monotonic())))
                if monitor_native_handoff:
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        break
                    if self.native_handoff_safety_lost is not None:
                        self.error = "native physical safety gate faulted while reading pre-JD MC mode"
                        break
            returned_utc = datetime.now(timezone.utc).isoformat()
            returned_ns = time.monotonic_ns()
            if self.native_handoff_safety_lost is not None and monitor_native_handoff:
                record = {
                    "transport_state": "GET_MC_ACTION_ABORTED_NATIVE_SAFETY",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "response": None,
                    "action_desc": None,
                    "action_status_value": None,
                    "native_mjdata": self.native_reset_support_record(),
                }
            elif not future.done() or future.result() is None:
                record = {
                    "transport_state": "GET_MC_ACTION_TIMEOUT",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "response": None,
                    "action_desc": None,
                    "action_status_value": None,
                }
            else:
                record = get_action_record(
                    request,
                    future.result(),
                    submitted_utc=submitted_utc,
                    submitted_monotonic_ns=submitted_ns,
                    returned_utc=returned_utc,
                    returned_monotonic_ns=returned_ns,
                )
                self.get_mode_desc = record["action_desc"]
                self.get_mode_status = record["action_status_value"]
            self.get_records.append(record)
            self.record_event("get_mc_action_readback", record=record)
            return record

        def set_action_once(self, action_desc: str, timeout_s: float, pre_action: dict[str, Any]) -> dict[str, Any]:
            if action_desc in self.set_attempted:
                raise RuntimeError(f"mutating SetMcAction retry blocked for {action_desc}")
            if action_desc not in {"JOINT_DEFAULT", "STAND_DEFAULT"}:
                raise ValueError(f"unsupported standing takeover stage: {action_desc}")
            self.set_attempted.add(action_desc)
            request = SetMcAction.Request()
            request.header = RequestHeader()
            request.header.stamp = self.get_clock().now().to_msg()
            request.source = args.source
            request.command = McActionCommand()
            request.command.action_desc = action_desc
            submitted_utc = datetime.now(timezone.utc).isoformat()
            submitted_ns = time.monotonic_ns()
            self.record_event(
                "set_mc_action_request",
                action_desc=action_desc,
                submitted_wall_utc=submitted_utc,
                submitted_monotonic_ns=submitted_ns,
                pre_action_state=pre_action,
            )
            if action_desc == "JOINT_DEFAULT":
                self.pending_jd_submission_ns = submitted_ns
            future = self.set_client.call_async(request)
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                self.spin_for(min(0.01, max(0.0, deadline - time.monotonic())))
                if self.require_native_reset_generation:
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        break
                    if self.native_unsafe_after_jd is not None:
                        record = {
                            "transport_state": "REQUEST_OUTCOME_UNKNOWN_NATIVE_PHYSICAL_SAFETY",
                            "wall_utc_submitted": submitted_utc,
                            "wall_monotonic_ns_submitted": submitted_ns,
                            "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
                            "wall_monotonic_ns_returned": time.monotonic_ns(),
                            "request": json_safe(request),
                            "pre_action_state": pre_action,
                            "response": None,
                            "response_status_value": None,
                            "response_header_code": None,
                            "response_message": "native physical safety gate faulted while the single request outcome was pending; it will not be retried",
                            "service_success": False,
                            "action_result_success": False,
                            "requested_action_desc": action_desc,
                            "native_mjdata": self.native_reset_support_record(),
                        }
                        self.set_records[action_desc] = record
                        self.record_event("set_mc_action_response", action_desc=action_desc, record=record)
                        return record
                if self.first_unsafe_ns is not None and self.first_unsafe_ns > submitted_ns:
                    record = {
                        "transport_state": "REQUEST_OUTCOME_UNKNOWN_POSTURE_UNSAFE",
                        "wall_utc_submitted": submitted_utc,
                        "wall_monotonic_ns_submitted": submitted_ns,
                        "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns_returned": time.monotonic_ns(),
                        "request": json_safe(request),
                        "pre_action_state": pre_action,
                        "response": None,
                        "response_status_value": None,
                        "response_header_code": None,
                        "response_message": "posture crossed the existing gate while waiting for this single request; outcome is unknown and it will not be retried",
                        "service_success": False,
                        "action_result_success": False,
                        "requested_action_desc": action_desc,
                    }
                    self.set_records[action_desc] = record
                    self.record_event("set_mc_action_response", action_desc=action_desc, record=record)
                    return record
            returned_utc = datetime.now(timezone.utc).isoformat()
            returned_ns = time.monotonic_ns()
            if not future.done():
                record = {
                    "transport_state": "REQUEST_OUTCOME_UNKNOWN_TIMEOUT",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "pre_action_state": pre_action,
                    "response": None,
                    "response_status_value": None,
                    "response_header_code": None,
                    "response_message": "request was sent once; no retry is permitted",
                    "service_success": False,
                    "action_result_success": False,
                }
            elif future.result() is None:
                record = {
                    "transport_state": "EMPTY_RESPONSE",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "pre_action_state": pre_action,
                    "response": None,
                    "response_status_value": None,
                    "response_header_code": None,
                    "response_message": "service future completed without a response",
                    "service_success": False,
                    "action_result_success": False,
                }
            else:
                record = set_action_record(
                    request,
                    future.result(),
                    success_status=CommonState.SUCCESS,
                    submitted_utc=submitted_utc,
                    submitted_monotonic_ns=submitted_ns,
                    returned_utc=returned_utc,
                    returned_monotonic_ns=returned_ns,
                )
                record["pre_action_state"] = pre_action
            record["requested_action_desc"] = action_desc
            self.set_records[action_desc] = record
            self.record_event("set_mc_action_response", action_desc=action_desc, record=record)
            return record

        def wait_reset_event(self) -> dict[str, Any] | None:
            deadline = time.monotonic() + args.reset_wait_timeout_s
            event_file = (
                self.synchronized_keyframe_event_file
                if self.synchronized_initialization else
                self.static_nominal_start_event_file
                if self.static_nominal_start_initialization else self.reset_event_file
            )
            assert event_file is not None
            expected_status = (
                "LOAD_KEY_BUTTON_CLICKED"
                if self.synchronized_initialization else
                "STATIC_NOMINAL_START_READY"
                if self.static_nominal_start_initialization else None
            )
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                if not event_file.exists():
                    continue
                try:
                    payload = json.loads(event_file.read_text(encoding="utf-8"))
                    event = payload.get("event", {})
                    dispatch_ns = int(event.get("dispatch_monotonic_ns", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                accepted = (
                    event.get("status") == expected_status
                    if expected_status is not None
                    else event.get("status") in {"KEY_DISPATCHED", "RESET_BUTTON_CLICKED"}
                )
                if not accepted or dispatch_ns <= 0:
                    self.error = (
                        "synchronized keyframe event did not contain a verified native Load key input"
                        if self.synchronized_initialization
                        else "static nominal-start event did not contain a verified parent readiness barrier"
                        if self.static_nominal_start_initialization
                        else "Reset event file did not contain a verified simulator Reset input event"
                    )
                    return None
                self.reset_dispatch = payload
                self.reset_ns = dispatch_ns
                self.record_event(
                    "synchronized_keyframe_barrier_observed" if self.synchronized_initialization
                    else "static_nominal_start_barrier_observed" if self.static_nominal_start_initialization
                    else "reset_barrier_observed",
                    reset_dispatch=payload,
                )
                return payload
            self.error = (
                "verified native Load key event file was not observed before timeout"
                if self.synchronized_initialization
                else "verified static nominal-start barrier was not observed before timeout"
                if self.static_nominal_start_initialization
                else "verified Reset event file was not observed before timeout"
            )
            return None

        def wait_pre_reset_ready(self) -> bool:
            deadline = time.monotonic() + args.ready_timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                if self.require_native_reset_generation:
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData pre-Reset baseline is unreadable: {type(exc).__name__}: {exc}"
                        break
                now_ns = time.monotonic_ns()
                fresh = [
                    key for key, row in self.latest.items()
                    if key in REQUIRED_TOPICS and 0.0 <= (now_ns - int(row["wall_monotonic_ns"])) / 1e9 <= MAX_STATE_AGE_S
                ]
                if len(fresh) != len(REQUIRED_TOPICS):
                    continue
                set_ready = bool(self.set_client.service_is_ready())
                get_ready = bool(self.get_client.service_is_ready())
                if not set_ready or not get_ready:
                    continue
                if self.require_native_reset_generation and self.native_baseline_generation is None:
                    continue
                pre_mode = self.mode_readback(args.get_timeout_s)
                if pre_mode.get("transport_state") != "RESPONSE_RECEIVED":
                    self.error = "GetMcAction pre-Reset readback failed"
                elif pre_mode.get("action_desc") != "PASSIVE_DEFAULT" or pre_mode.get("action_status_value") != 100:
                    self.error = (
                        "unexpected pre-Reset mode/readiness: "
                        f"action={pre_mode.get('action_desc')!r} status={pre_mode.get('action_status_value')!r}"
                    )
                else:
                    self.observer_ready = True
                ready = {
                    "schema": "robotsim.issue80.takeover-watch-ready.v1",
                    "status": "READY" if self.observer_ready else "BLOCKED_PRE_RESET_MODE_OR_SERVICE",
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_monotonic_ns": time.monotonic_ns(),
                    "node_started_wall_utc": self.started_utc,
                    "node_started_monotonic_ns": self.started_ns,
                    "subscriptions_created_wall_utc": self.subscriptions_created_utc,
                    "subscriptions_created_monotonic_ns": self.subscriptions_created_ns,
                    "fresh_topics": sorted(fresh),
                    "set_mc_action_service_ready": set_ready,
                    "get_mc_action_service_ready": get_ready,
                    "pre_reset_get_mc_action": pre_mode,
                    "raw_common_state": self.latest.get("common", {}).get("data"),
                    "native_mjdata": self.native_reset_support_record(),
                    "error": self.error,
                }
                self.write_json_once(self.ready_path, ready)
                self.record_event("takeover_client_ready", readiness=ready)
                return self.observer_ready
            ready = {
                "schema": "robotsim.issue80.takeover-watch-ready.v1",
                "status": "BLOCKED_PRE_RESET_READINESS_TIMEOUT",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                "fresh_topics": sorted(self.latest),
                "set_mc_action_service_ready": bool(self.set_client.service_is_ready()),
                "get_mc_action_service_ready": bool(self.get_client.service_is_ready()),
                "error": "required state streams and both MC clients did not become ready before Reset",
            }
            self.write_json_once(self.ready_path, ready)
            self.error = ready["error"]
            return False

        def wait_for_post_reset_upright(self) -> dict[str, Any] | None:
            assert self.reset_ns is not None
            deadline = min(
                self.reset_ns / 1e9 + args.upright_timeout_s,
                time.monotonic() + args.upright_timeout_s,
            )
            last: dict[str, Any] = {"status": "POST_RESET_STATE_NOT_READY"}
            if not self.wait_for_native_reset_support(deadline):
                return {
                    "status": "POST_RESET_NATIVE_GENERATION_OR_SUPPORT_NOT_CONFIRMED",
                    "reset_dispatch_monotonic_ns": self.reset_ns,
                    "native_mjdata": self.native_reset_support_record(),
                    "error": self.error,
                }
            if self.synchronized_initialization:
                while rclpy.ok() and time.monotonic() < deadline:
                    self.spin_for(0.005)
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        break
                    now_ns = time.monotonic_ns()
                    transport = self.ros_transport_liveness(now_ns)
                    if not transport["fresh_required_topics"]:
                        last = {
                            "status": "SYNCHRONIZED_KEYFRAME_WAITING_FOR_ROS_TRANSPORT",
                            "initialization_dispatch_monotonic_ns": self.reset_ns,
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                        }
                        continue
                    leg_state_row = self.latest.get("leg_state")
                    native_keyframe_ns = self.native_keyframe_ready_ns
                    if (
                        not isinstance(leg_state_row, Mapping)
                        or native_keyframe_ns is None
                        or int(leg_state_row.get("wall_monotonic_ns", 0)) <= native_keyframe_ns
                    ):
                        last = {
                            "status": "SYNCHRONIZED_KEYFRAME_WAITING_FOR_POST_KEYFRAME_LEG_STATE",
                            "initialization_dispatch_monotonic_ns": self.reset_ns,
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                        }
                        continue
                    assert self.synchronized_keyframe_spec is not None
                    ros_leg_state = ros_leg_state_matches_synchronized_keyframe(
                        leg_state_row.get("data", {}), self.synchronized_keyframe_spec,
                    )
                    if not ros_leg_state["eligible"]:
                        last = {
                            "status": "SYNCHRONIZED_KEYFRAME_WAITING_FOR_MATCHING_ROS_LEG_STATE",
                            "initialization_dispatch_monotonic_ns": self.reset_ns,
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                            "ros_leg_state": ros_leg_state,
                            "leg_state_wall_monotonic_ns": leg_state_row.get("wall_monotonic_ns"),
                        }
                        continue
                    if self.synchronized_ros_leg_state_ready is None:
                        self.synchronized_ros_leg_state_ready = {
                            "wall_utc": datetime.now(timezone.utc).isoformat(),
                            "wall_monotonic_ns": now_ns,
                            "leg_state_wall_monotonic_ns": leg_state_row.get("wall_monotonic_ns"),
                            "assessment": ros_leg_state,
                        }
                        self.record_event(
                            "synchronized_ros_leg_state_matches_keyframe",
                            readiness=self.synchronized_ros_leg_state_ready,
                        )
                    readback = self.mode_readback(args.get_timeout_s)
                    if (
                        readback.get("transport_state") != "RESPONSE_RECEIVED"
                        or readback.get("action_desc") != "PASSIVE_DEFAULT"
                        or readback.get("action_status_value") != 100
                    ):
                        self.error = "GetMcAction did not retain PASSIVE_DEFAULT/status=100 before synchronized JD"
                        break
                    last = {
                        "status": "SYNCHRONIZED_KEYFRAME_READY",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns": now_ns,
                        "initialization_dispatch_monotonic_ns": self.reset_ns,
                        "snapshot": {
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                            "ros_leg_state_matches_keyframe": self.synchronized_ros_leg_state_ready,
                            "get_mc_action": readback,
                            "physics_stepping_released": False,
                            "physical_state_authority": "native_MjData_keyframe",
                        },
                    }
                    self.record_event("synchronized_keyframe_gate", snapshot=last)
                    return last
                if self.error is None:
                    self.error = "synchronized keyframe was verified but MC transport/mode was not ready"
                last.update({"error": self.error, "timeout_s": args.upright_timeout_s})
                return last
            if self.require_native_reset_generation:
                # Native MjData has already established the physical handoff
                # condition.  Do not wait for queued ROS pose samples here:
                # the previous run showed they can describe the pre-Reset
                # fallen state long after the native simulator has advanced.
                # ROS remains required as a live transport and is re-anchored
                # at the first effective JD leg command below.
                while rclpy.ok() and time.monotonic() < deadline:
                    self.spin_for(0.005)
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        break
                    if self.native_handoff_safety_lost is not None:
                        self.error = "native physical safety gate faulted before JD could be issued"
                        break
                    now_ns = time.monotonic_ns()
                    transport = self.ros_transport_liveness(now_ns)
                    if not transport["fresh_required_topics"]:
                        last = {
                            "status": "POST_RESET_NATIVE_SUPPORT_WAITING_FOR_ROS_TRANSPORT",
                            "reset_dispatch_monotonic_ns": self.reset_ns,
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                        }
                        continue
                    readback = self.mode_readback(args.get_timeout_s, monitor_native_handoff=True)
                    if (
                        readback.get("transport_state") != "RESPONSE_RECEIVED"
                        or readback.get("action_desc") != "PASSIVE_DEFAULT"
                        or readback.get("action_status_value") != 100
                    ):
                        self.error = "GetMcAction did not retain PASSIVE_DEFAULT/status=100 at native JD handoff"
                        last = {
                            "status": "POST_RESET_NATIVE_SUPPORT_BAD_PRE_JD_MODE",
                            "reset_dispatch_monotonic_ns": self.reset_ns,
                            "native_mjdata": self.native_reset_support_record(),
                            "ros_transport": transport,
                            "get_mc_action": readback,
                            "error": self.error,
                        }
                        break
                    support = self.native_reset_support_record()
                    last = {
                        "status": "POST_RESET_NATIVE_SUPPORT_READY",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns": now_ns,
                        "reset_dispatch_monotonic_ns": self.reset_ns,
                        "post_reset_epoch_monotonic_ns": self._post_reset_epoch(),
                        "snapshot": {
                            "native_mjdata": support,
                            "ros_transport": transport,
                            "get_mc_action": readback,
                            "physical_state_authority": "native_MjData",
                            "ros_pose_generation": "PENDING_POST_JD_EFFECTIVE_COMMAND",
                        },
                    }
                    self.record_event("post_reset_native_support_gate", snapshot=last)
                    return last
                if self.error is None:
                    self.error = "native support was observed but ROS transport or PASSIVE_DEFAULT was not ready before JD"
                last.update({"error": self.error, "timeout_s": args.upright_timeout_s})
                return last
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                now_ns = time.monotonic_ns()
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed the existing gate after a transient upright pose and before JD"
                    return {
                        "status": "POST_RESET_UNSAFE_BEFORE_JD",
                        "reset_dispatch_monotonic_ns": self.reset_ns,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                        "first_unsafe_monotonic_ns": self.first_unsafe_ns,
                        "error": self.error,
                    }
                latest = self._post_reset_latest()
                gate = assess_pre_action_state(
                    latest,
                    now_monotonic_ns=now_ns,
                    upright_odom_samples=self.upright_samples,
                    service_ready=self.set_client.service_is_ready() and self.get_client.service_is_ready(),
                    reset_dispatch_monotonic_ns=self._post_reset_epoch(),
                    required_upright_odom_samples=self.required_post_reset_upright_samples,
                )
                common = latest.get("common", {}).get("data", {})
                action = common.get("action_info", {})
                mode = action.get("action_desc")
                last = {
                    "status": "POST_RESET_UPRIGHT_NOT_YET_CONFIRMED",
                    "reset_dispatch_monotonic_ns": self.reset_ns,
                    "post_reset_epoch_monotonic_ns": self._post_reset_epoch(),
                    "native_mjdata": self.native_reset_support_record(),
                    "snapshot": gate,
                    "raw_action_desc": mode,
                    "raw_action_status": action.get("status", {}).get("value"),
                }
                if gate["eligible"] and mode == "PASSIVE_DEFAULT" and action.get("status", {}).get("value") == 100:
                    last.update({
                        "status": "POST_RESET_UPRIGHT_OBSERVED",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns": now_ns,
                        "reset_to_upright_gate_ms": elapsed_ms(self.reset_ns, now_ns),
                        "raw_common_state": common,
                    })
                    self.record_event("post_reset_upright_gate", snapshot=last)
                    return last
            self.error = "no eligible post-Reset upright state under the unchanged height/tilt/freshness gate"
            last.update({"status": "POST_RESET_UPRIGHT_TIMEOUT", "timeout_s": args.upright_timeout_s})
            return last

        def confirm_mode(self, expected: str, timeout_s: float) -> dict[str, Any] | None:
            deadline = time.monotonic() + timeout_s
            last: dict[str, Any] | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                if self.require_native_reset_generation:
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        return last
                    if self.native_unsafe_after_jd is not None:
                        self.error = "native physical safety gate faulted while confirming MC mode"
                        return last
                if self.first_unsafe_ns is not None:
                    return last
                last = self.mode_readback(args.get_timeout_s)
                if last.get("transport_state") == "RESPONSE_RECEIVED" and last.get("action_desc") == expected and last.get("action_status_value") == 100:
                    return last
                if self.first_unsafe_ns is not None:
                    return last
                self.spin_for(0.05)
            return last

        def wait_for_synchronized_jd_control_ready(self, deadline: float) -> bool:
            """Wait for a real MC leg command while native physics remains paused."""
            if not self.synchronized_initialization:
                return True
            assert self.synchronized_ready_file is not None
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                try:
                    self.poll_native_capture()
                except (OSError, ValueError) as exc:
                    self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                    return False
                if self.first_effective_after_jd is None:
                    continue
                now_ns = time.monotonic_ns()
                transport = self.ros_transport_liveness(now_ns)
                if not transport["fresh_required_topics"]:
                    continue
                readback = self.mode_readback(args.get_timeout_s)
                if (
                    readback.get("transport_state") != "RESPONSE_RECEIVED"
                    or readback.get("action_desc") != "JOINT_DEFAULT"
                    or readback.get("action_status_value") != 100
                ):
                    continue
                ready = {
                    "schema": "robotsim.issue80.synchronized-jd-ready.v1",
                    "status": "JD_CONTROL_READY_WHILE_PHYSICS_PAUSED",
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_monotonic_ns": now_ns,
                    "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                    "native_mjdata": self.native_reset_support_record(),
                    "ros_transport": transport,
                    "ros_leg_state_matches_keyframe": self.synchronized_ros_leg_state_ready,
                    "get_mc_action": readback,
                    "physics_stepping_released": False,
                }
                self.write_json_once(self.synchronized_ready_file, ready)
                self.record_event("synchronized_jd_control_ready", readiness=ready)
                return True
            self.error = "JOINT_DEFAULT did not produce an effective leg command while native physics was paused"
            return False

        def wait_for_physics_release(self, deadline: float) -> bool:
            """Accept exactly the parent-verified native Play/Pause release event."""
            if not self.synchronized_initialization:
                return True
            assert self.physics_release_event_file is not None
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                if not self.physics_release_event_file.exists():
                    continue
                try:
                    payload = json.loads(self.physics_release_event_file.read_text(encoding="utf-8"))
                    event = payload.get("event", {})
                    dispatch_ns = int(event.get("dispatch_monotonic_ns", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if event.get("status") != "KEY_DISPATCHED" or event.get("key") != "space" or dispatch_ns <= 0:
                    self.error = "physics release event is not a verified native MuJoCo Play/Pause dispatch"
                    return False
                self.physics_release = payload
                self.physics_release_ns = dispatch_ns
                self._clear_assessment_counters()
                self.record_event("synchronized_physics_release_observed", physics_release=payload)
                return True
            self.error = "verified native physics-release event was not observed before timeout"
            return False

        def wait_for_native_release_actuation(self, deadline: float) -> bool:
            """Require the first resumed mj_step and nonzero MC actuation before JD qualification."""
            if not self.synchronized_initialization:
                return True
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                try:
                    self.poll_native_capture()
                except (OSError, ValueError) as exc:
                    self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                    return False
                if self.native_unsafe_after_jd is not None:
                    self.error = "native physical safety gate faulted immediately after physics release"
                    return False
                if (
                    self.first_native_step_after_release is not None
                    and self.first_native_actuation_after_release is not None
                ):
                    return True
            self.error = "physics release did not yield a native mj_step plus nonzero MC actuation before the bounded deadline"
            return False

        def wait_for_joint_stand(self, timeout_s: float) -> dict[str, Any] | None:
            assert self.reset_ns is not None
            deadline = time.monotonic() + timeout_s
            last: dict[str, Any] | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                if self.require_native_reset_generation:
                    try:
                        self.poll_native_capture()
                    except (OSError, ValueError) as exc:
                        self.error = f"native MjData capture became unreadable: {type(exc).__name__}: {exc}"
                        return {"eligible": False, "failure_reason": self.error}
                    if self.native_unsafe_after_jd is not None:
                        self.error = "native physical safety gate faulted while confirming JOINT_DEFAULT"
                        return {
                            "eligible": False,
                            "failure_reason": self.error,
                            "native_mjdata": self.native_reset_support_record(),
                        }
                    if self.post_jd_control_epoch_ns is None:
                        # Do not assess queued ROS state until the MC has
                        # emitted a new effective JD command.
                        continue
                now_ns = time.monotonic_ns()
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed existing height/tilt gate after the first upright post-Reset pose"
                    return {
                        "eligible": False,
                        "failure_reason": self.error,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                    }
                latest = self._post_reset_latest()
                last_common = latest.get("common", {}).get("data", {})
                common_action = last_common.get("action_info", {}).get("action_desc")
                common_status = last_common.get("action_info", {}).get("status", {}).get("value")
                readback_desc = self.get_mode_desc
                readback_status = self.get_mode_status
                if self.stand_samples >= REQUIRED_STAND_COMMON_SAMPLES:
                    last_get_ns = int(self.get_records[-1].get("wall_monotonic_ns_returned", 0)) if self.get_records else 0
                    if now_ns - last_get_ns > 150_000_000:
                        readback = self.mode_readback(args.get_timeout_s)
                        readback_desc = readback.get("action_desc")
                        readback_status = readback.get("action_status_value")
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed the existing gate while confirming the JOINT_DEFAULT stand state"
                    return {
                        "eligible": False,
                        "failure_reason": self.error,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                    }
                last = assess_joint_to_stand_state(
                    latest,
                    now_monotonic_ns=now_ns,
                    reset_dispatch_monotonic_ns=self._assessment_epoch() or self.reset_ns,
                    upright_odom_samples=self.upright_samples,
                    stand_common_samples=self.stand_samples,
                    set_service_ready=self.set_client.service_is_ready(),
                    get_service_ready=self.get_client.service_is_ready(),
                    get_mode_desc=readback_desc,
                    get_mode_status=readback_status,
                    body_stand_value=body_stand_value,
                    required_upright_odom_samples=self.required_post_reset_upright_samples,
                )
                if last["eligible"]:
                    last["wall_utc"] = datetime.now(timezone.utc).isoformat()
                    last["wall_monotonic_ns"] = now_ns
                    last["native_mjdata"] = self.native_reset_support_record()
                    return last
            self.error = "JOINT_DEFAULT did not produce fresh upright MC STAND status within its bounded window"
            if last is None:
                last = {"eligible": False, "failure_reason": self.error}
            else:
                last["eligible"] = False
                last["failure_reason"] = self.error
            return last

        def _write_final_result(self, status: str, *, block_stage: str | None = None, error: str | None = None,
                                pre_reset: dict[str, Any] | None = None,
                                upright_gate: dict[str, Any] | None = None,
                                stand_gate: dict[str, Any] | None = None) -> None:
            if self.result is not None:
                return
            # A blocked pre-JD synchronized barrier has no action record.  It
            # must still persist its concrete blocker rather than throwing
            # while formatting diagnostic timing fields.
            jd = self.set_records.get("JOINT_DEFAULT") or {}
            sd = self.set_records.get("STAND_DEFAULT") or {}
            timing_events = {
                "reset_dispatch_monotonic_ns": self.reset_ns,
                "native_reset_generation_host_monotonic_ns": self.native_reset_host_monotonic_ns,
                "native_support_ready_host_monotonic_ns": self.native_support_ready_ns,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "joint_default": jd,
                "stand_default": sd,
                "body_stand_ready_monotonic_ns": (stand_gate or {}).get("wall_monotonic_ns"),
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "first_unsafe_post_reset_monotonic_ns": self.first_unsafe_ns,
            }
            self.result = {
                "schema": "robotsim.issue80.standing-takeover-capture.v1",
                "status": status,
                "blocking_stage": block_stage,
                "error": error,
                "source": args.source,
                "pre_reset_get_mc_action": pre_reset,
                "reset_dispatch": self.reset_dispatch,
                "native_mjdata": self.native_reset_support_record(),
                "post_reset_upright_gate": upright_gate,
                "joint_default_set_mc_action": jd,
                "joint_default_get_mc_action_readbacks": [r for r in self.get_records if r.get("wall_monotonic_ns_submitted", 0) > (self.reset_ns or 0) and r.get("action_desc") == "JOINT_DEFAULT"],
                "joint_to_stand_gate": stand_gate,
                "stand_default_set_mc_action": sd,
                "stand_default_mode_confirmed_monotonic_ns": self.stand_default_confirmed_ns,
                "get_mc_action_readbacks": list(self.get_records),
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "first_unsafe_post_reset": self.first_unsafe_row,
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "timing_ms": summarize_takeover_timing(timing_events),
                "synchronized_initialization_timing_ms": {
                    "keyframe_dispatch_to_native_keyframe_ms": elapsed_ms(
                        self.reset_ns, self.native_keyframe_ready_ns
                    ),
                    "JD_response_to_effective_command_while_paused_ms": elapsed_ms(
                        jd.get("wall_monotonic_ns_returned"), self.first_effective_after_jd
                    ),
                    "physics_release_to_first_native_step_ms": elapsed_ms(
                        self.physics_release_ns,
                        (self.first_native_step_after_release or {}).get("host_monotonic_ns"),
                    ),
                    "physics_release_to_first_native_actuation_ms": elapsed_ms(
                        self.physics_release_ns,
                        (self.first_native_actuation_after_release or {}).get("host_monotonic_ns"),
                    ),
                } if self.synchronized_initialization else None,
                "limitations": [
                    "ROS graph exposes no foot-contact or ground-reaction-force topic in the captured runtime inventory.",
                    "MC body_status=STAND is retained as the installed AimDK classifier; it is not a direct force-plate measurement.",
                    "Header stamps are kept raw; all elapsed timing uses host monotonic receipt/submission timestamps.",
                ],
            }
            self.write_json_once(self.result_path, self.result)
            self.record_event("takeover_result", result=self.result)

        def run_takeover(self) -> int:
            pre_reset_mode: dict[str, Any] | None = None
            if not self.wait_pre_reset_ready():
                self._write_final_result("BLOCKED_PRE_RESET_READINESS", block_stage="PRE_RESET", error=self.error)
                return 2
            readiness = json.loads(self.ready_path.read_text(encoding="utf-8"))
            pre_reset_mode = readiness.get("pre_reset_get_mc_action")
            event_doc = self.wait_reset_event()
            if event_doc is None:
                self._write_final_result("BLOCKED_RESET_EVENT", block_stage="RESET", error=self.error, pre_reset=pre_reset_mode)
                return 2
            upright_gate = self.wait_for_post_reset_upright()
            if upright_gate is None or upright_gate.get("status") not in {
                "POST_RESET_UPRIGHT_OBSERVED",
                "POST_RESET_NATIVE_SUPPORT_READY",
                "SYNCHRONIZED_KEYFRAME_READY",
            }:
                self._write_final_result("BLOCKED_POST_RESET_UPRIGHT", block_stage="POST_RESET_UPRIGHT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            jd_pre = upright_gate["snapshot"]
            jd = self.set_action_once("JOINT_DEFAULT", args.set_timeout_s, jd_pre)
            if jd.get("transport_state") != "RESPONSE_RECEIVED" or jd.get("service_success") is not True or jd.get("action_result_success") is not True:
                self.error = "JOINT_DEFAULT SetMcAction did not return a successful status/code pair"
                self._write_final_result("BLOCKED_JOINT_DEFAULT_REJECTED", block_stage="JOINT_DEFAULT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            jd_readback = self.confirm_mode("JOINT_DEFAULT", 1.5)
            if jd_readback is None or jd_readback.get("action_desc") != "JOINT_DEFAULT" or jd_readback.get("action_status_value") != 100:
                self.error = "GetMcAction did not confirm JOINT_DEFAULT/status=100 after the single JD request"
                self._write_final_result("BLOCKED_JOINT_DEFAULT_READBACK", block_stage="JOINT_DEFAULT_READBACK", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            if self.synchronized_initialization:
                release_deadline = time.monotonic() + args.joint_stand_timeout_s
                if not self.wait_for_synchronized_jd_control_ready(release_deadline):
                    self._write_final_result("BLOCKED_SYNCHRONIZED_JD_CONTROL", block_stage="SYNCHRONIZED_JD_CONTROL", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                    return 2
                if not self.wait_for_physics_release(time.monotonic() + args.reset_wait_timeout_s):
                    self._write_final_result("BLOCKED_SYNCHRONIZED_PHYSICS_RELEASE", block_stage="SYNCHRONIZED_PHYSICS_RELEASE", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                    return 2
                if not self.wait_for_native_release_actuation(time.monotonic() + args.joint_stand_timeout_s):
                    self._write_final_result("BLOCKED_SYNCHRONIZED_RELEASE_ACTUATION", block_stage="SYNCHRONIZED_RELEASE_ACTUATION", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                    return 2
            stand_gate = self.wait_for_joint_stand(args.joint_stand_timeout_s)
            if not stand_gate.get("eligible"):
                self._write_final_result("BLOCKED_STAND_PRECONDITION", block_stage="BODY_STAND_GATE", error=str(stand_gate.get("failure_reason")), pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            sd = self.set_action_once("STAND_DEFAULT", args.set_timeout_s, stand_gate)
            if sd.get("transport_state") != "RESPONSE_RECEIVED" or sd.get("service_success") is not True or sd.get("action_result_success") is not True:
                self.error = "STAND_DEFAULT SetMcAction did not return a successful status/code pair"
                self._write_final_result("BLOCKED_STAND_DEFAULT_REJECTED", block_stage="STAND_DEFAULT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            sd_readback = self.confirm_mode("STAND_DEFAULT", 2.0)
            if sd_readback is None or sd_readback.get("action_desc") != "STAND_DEFAULT" or sd_readback.get("action_status_value") != 100:
                self.error = "GetMcAction did not confirm STAND_DEFAULT/status=100 after the single SD request"
                self._write_final_result("BLOCKED_STAND_DEFAULT_READBACK", block_stage="STAND_DEFAULT_READBACK", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            self.stand_default_confirmed_ns = int(
                sd_readback.get("wall_monotonic_ns_returned") or time.monotonic_ns()
            )
            self._write_final_result("TAKEOVER_SEQUENCE_COMPLETE", pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
            return 0

        def capture(self) -> dict[str, Any]:
            status = "TRACE_CAPTURED"
            start = time.monotonic()
            observation = None
            if self.reset_ns is not None:
                observation = standing_observation_window(
                    reset_dispatch_monotonic_ns=self.reset_ns,
                    stand_default_confirmed_monotonic_ns=self.stand_default_confirmed_ns,
                    duration_s=args.duration_s,
                )
            # Count the baseline from the positive STAND readback; the Reset-to-SD
            # handoff is not part of the requested standing observation window.
            while (
                rclpy.ok()
                and observation is not None
                and time.monotonic_ns() < int(observation["deadline_monotonic_ns"])
            ):
                if self.first_unsafe_ns is not None:
                    status = "BLOCKED_UNSAFE_STAND_BASELINE"
                    break
                self.spin_for(0.005)
            if self.reset_ns is None:
                status = "BLOCKED_NO_RESET_EVENT"
            elif self.stand_default_confirmed_ns is None:
                status = "STAND_DEFAULT_NOT_CONFIRMED"
            capture_end_ns = time.monotonic_ns()
            final_mode_readback = (
                self.mode_readback(args.get_timeout_s)
                if self.reset_ns is not None and rclpy.ok() else None
            )
            final_common_observation = self.latest.get("common")
            final_common = (final_common_observation or {}).get("data")
            if observation is not None:
                anchor_ns = int(observation["anchor_monotonic_ns"])
                unsafe_ns = self.first_unsafe_ns
                effective_end_ns = min(capture_end_ns, unsafe_ns) if unsafe_ns is not None else capture_end_ns
                standing_observation_s = max(0.0, (effective_end_ns - anchor_ns) / 1e9)
                complete = (
                    self.stand_default_confirmed_ns is not None
                    and unsafe_ns is None
                    and standing_observation_s + 1e-6 >= args.duration_s
                )
                if status == "TRACE_CAPTURED":
                    status = "STAND_DEFAULT_BASELINE_COMPLETE" if complete else "STAND_DEFAULT_BASELINE_INCOMPLETE"
                standing_observation = {
                    **observation,
                    "observed_end_monotonic_ns": effective_end_ns,
                    "observed_duration_s": standing_observation_s,
                    "minimum_duration_s": args.duration_s,
                    "complete": complete,
                    "first_unsafe_monotonic_ns": unsafe_ns,
                    "expected_body_status": body_stand_value,
                    "initial_stand_mode_readback": next(
                        (row for row in self.get_records if row.get("action_desc") == "STAND_DEFAULT" and row.get("action_status_value") == 100),
                        None,
                    ),
                    "final_stand_mode_readback": final_mode_readback,
                }
            else:
                standing_observation = {
                    "anchor": "NO_RESET_EVENT",
                    "observed_end_monotonic_ns": capture_end_ns,
                    "observed_duration_s": 0.0,
                    "minimum_duration_s": args.duration_s,
                    "complete": False,
                    "first_unsafe_monotonic_ns": self.first_unsafe_ns,
                }
            self.trace.flush()
            odom_rows = []
            common_rows = []
            for line in self.trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "state" and row.get("topic_key") == "odom":
                    odom_rows.append(row)
                elif row.get("event") == "state" and row.get("topic_key") == "common":
                    common_rows.append(row)
            if observation is not None:
                stand_start_ns = int(observation["anchor_monotonic_ns"])
                stand_end_ns = int(standing_observation["observed_end_monotonic_ns"])
                common_window = [
                    row for row in common_rows
                    if stand_start_ns <= int(row.get("wall_monotonic_ns", 0)) <= stand_end_ns
                ]
                action_descriptions = []
                action_statuses = []
                fsm_values = []
                body_values = []
                input_sources = []
                for row in common_window:
                    common = row.get("data") or {}
                    action = common.get("action_info") or {}
                    action_descriptions.append(action.get("action_desc"))
                    status_value = (action.get("status") or {}).get("value")
                    action_statuses.append(status_value)
                    fsm_values.append(_raw_fsm_state_value(common))
                    body_values.append((common.get("body_status") or {}).get("value"))
                    input_sources.append(common.get("input_source") or {})
                standing_observation.update({
                    "common_state_sample_count": len(common_window),
                    "common_mode_desc_values": sorted({str(value) for value in action_descriptions}),
                    "common_mode_status_values": sorted({value for value in action_statuses}, key=str),
                    "raw_fsm_state_values": sorted({value for value in fsm_values}, key=str),
                    "raw_body_status_values": sorted({value for value in body_values}, key=str),
                    "input_source_values": input_sources,
                })
                standing_observation["observer_status"] = status
            post_odom = [r for r in odom_rows if self.reset_ns is not None and int(r.get("wall_monotonic_ns", 0)) > self.reset_ns]
            z = [float(r.get("data", {}).get("z")) for r in post_odom if isinstance(r.get("data", {}).get("z"), (int, float))]
            jd = self.set_records.get("JOINT_DEFAULT", {})
            sd = self.set_records.get("STAND_DEFAULT", {})
            stand_gate = (self.result or {}).get("joint_to_stand_gate") or {}
            timing_events = {
                "reset_dispatch_monotonic_ns": self.reset_ns,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "joint_default": jd,
                "stand_default": sd,
                "body_stand_ready_monotonic_ns": stand_gate.get("wall_monotonic_ns"),
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "first_unsafe_post_reset_monotonic_ns": self.first_unsafe_ns,
            }
            summary = {
                "schema": "robotsim.issue80.reset-state-trace.v1",
                "observer_status": status,
                "takeover_status": (self.result or {}).get("status", "NOT_COMPLETED"),
                "duration_limit_s": args.duration_s,
                "standing_baseline_observation": standing_observation,
                "stand_default_mode_confirmed_monotonic_ns": self.stand_default_confirmed_ns,
                "initial_stand_mode_readback": next(
                    (row for row in self.get_records if row.get("action_desc") == "STAND_DEFAULT" and row.get("action_status_value") == 100),
                    None,
                ),
                "final_stand_mode_readback": final_mode_readback,
                "final_common_state_observation": final_common_observation,
                "final_common_state": final_common,
                "final_common_input_source": (final_common or {}).get("input_source"),
                "locomotion_velocity_publisher_created": False,
                "wall_elapsed_s": time.monotonic() - start,
                "sample_counts": dict(self.counts),
                "clock_messages": len(self.clocks),
                "clock_topic_exposed": bool(self.clocks),
                "odom_sample_count": len(post_odom),
                "odom_z_min_m": min(z) if z else None,
                "odom_z_max_m": max(z) if z else None,
                "odom_first": post_odom[0].get("data") if post_odom else None,
                "odom_last": post_odom[-1].get("data") if post_odom else None,
                "first_odom_wall_utc": post_odom[0].get("wall_utc") if post_odom else None,
                "last_odom_wall_utc": post_odom[-1].get("wall_utc") if post_odom else None,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "first_unsafe_post_reset": self.first_unsafe_row,
                "takeover_result": self.result,
                "timing_ms": summarize_takeover_timing(timing_events),
                "limitations": [
                    "Header timestamps are recorded as fields; elapsed intervals use host monotonic time.",
                    "This observer does not write qpos, joint commands, or contact forces.",
                    "No locomotion velocity publisher is created by this observer.",
                    "Input-source monitoring uses live McCommonState telemetry; no undocumented source-registration or command-receipt acknowledgement is inferred.",
                    "The captured ROS graph has no dedicated foot-contact/ground-reaction-force topic.",
                ],
            }
            self.write_json_once(self.summary_path, summary)
            self.record_event("takeover_trace_complete", summary=summary)
            return summary

    rclpy.init()
    node = TakeoverCapture()
    try:
        action_rc = node.run_takeover()
        node.capture()
        print(json.dumps(node.result or {"status": "NO_TAKEOVER_RESULT", "error": node.error}, separators=(",", ":"), allow_nan=False), flush=True)
        return action_rc
    except Exception as exc:
        node.error = f"{type(exc).__name__}: {exc}"
        if node.result is None:
            node._write_final_result("BLOCKED_CLIENT_EXCEPTION", block_stage="CLIENT", error=node.error)
        try:
            if not node.summary_path.exists():
                node.capture()
        except Exception:
            pass
        print(json.dumps({"status": "BLOCKED_CLIENT_EXCEPTION", "error": node.error}, separators=(",", ":"), allow_nan=False), flush=True)
        return 2
    finally:
        node.trace.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
