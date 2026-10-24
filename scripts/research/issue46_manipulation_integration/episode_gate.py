"""Model-independent acceptance gate for measured bottle manipulation traces."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean
from typing import Mapping, Sequence


class ManipulationPhase(str, Enum):
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CONTACT = "CONTACT"
    LOAD_TRANSFER = "LOAD_TRANSFER"
    LIFT_50MM = "LIFT_50MM"
    HOLD = "HOLD"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    SETTLE = "SETTLE"


PHASE_ORDER = tuple(ManipulationPhase)


@dataclass(frozen=True)
class EpisodeFrame:
    phase: ManipulationPhase
    simulation_time_s: float
    physics_step_index: int
    model_id: str
    model_sha256: str
    bottle_com_world_m: tuple[float, float, float]
    bottle_quat_world_xyzw: tuple[float, float, float, float]
    bottle_linear_speed_m_s: float | None
    bottle_angular_speed_rad_s: float | None
    target_bottle_com_world_m: tuple[float, float, float] | None
    force_bearing_contact_roles: frozenset[str]
    tool_object_contact_active: bool | None
    left_hand_contact: bool | None
    table_contact: bool | None
    tool_upward_support_n: float | None
    table_reaction_vertical_n: float | None
    max_relative_slip_m: float | None
    minimum_approach_clearance_m: float | None
    tool_opening_m: float | None
    source_position_limits_pass: bool | None
    source_velocity_limits_pass: bool | None
    actuator_limits_pass: bool | None
    follower_constraints_pass: bool | None
    bottle_qpos_writes_after_start: int | None
    bottle_attachment_active: bool | None
    mocap_carry_active: bool | None
    hidden_support_active: bool | None


@dataclass(frozen=True)
class EpisodeCriteria:
    required_contact_groups: tuple[tuple[str, ...], ...]
    minimum_contact_fraction: float
    minimum_tool_upward_support_n: float
    minimum_table_unloading_n: float
    maximum_relative_slip_m: float
    minimum_approach_clearance_m: float = 0.002
    minimum_lift_m: float = 0.05
    minimum_hold_duration_s: float = 1.0
    place_target_xy_world_m: tuple[float, float] | None = None
    maximum_place_xy_error_m: float = 0.020
    maximum_upright_error_rad: float = 0.08726646259971647
    minimum_release_opening_m: float = 0.0884
    minimum_settle_duration_s: float = 3.0
    maximum_settle_linear_speed_m_s: float = 0.020
    maximum_settle_angular_speed_rad_s: float = 0.050
    maximum_settle_position_drift_m: float = 0.020

    def __post_init__(self) -> None:
        if not self.required_contact_groups or any(not group for group in self.required_contact_groups):
            raise ValueError("at least one nonempty force-bearing contact group is required")
        values = (
            self.minimum_contact_fraction,
            self.minimum_tool_upward_support_n,
            self.minimum_table_unloading_n,
            self.maximum_relative_slip_m,
            self.minimum_approach_clearance_m,
            self.minimum_lift_m,
            self.minimum_hold_duration_s,
            self.maximum_place_xy_error_m,
            self.maximum_upright_error_rad,
            self.minimum_release_opening_m,
            self.minimum_settle_duration_s,
            self.maximum_settle_linear_speed_m_s,
            self.maximum_settle_angular_speed_rad_s,
            self.maximum_settle_position_drift_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("episode criteria must be finite and nonnegative")
        if self.minimum_contact_fraction > 1.0:
            raise ValueError("minimum_contact_fraction must be in [0, 1]")
        if (self.minimum_contact_fraction <= 0.0 or self.minimum_tool_upward_support_n <= 0.0
                or self.minimum_table_unloading_n <= 0.0 or self.minimum_lift_m < 0.05
                or self.minimum_hold_duration_s < 1.0 or self.minimum_approach_clearance_m < 0.002):
            raise ValueError("criteria may not weaken contact, load, clearance, 50 mm lift, or 1 s hold gates")
        if self.minimum_settle_duration_s < 1.0:
            raise ValueError("settle duration must be at least 1 s")
        if self.place_target_xy_world_m is not None and (
            len(self.place_target_xy_world_m) != 2
            or not all(math.isfinite(float(value)) for value in self.place_target_xy_world_m)
        ):
            raise ValueError("place target must be a finite world XY point")


@dataclass(frozen=True)
class EpisodeGateResult:
    status: str
    stage_status: Mapping[str, str]
    blockers: tuple[str, ...]
    last_physics_step_index: int


def _contact_fraction(frames: Sequence[EpisodeFrame], groups: tuple[tuple[str, ...], ...]) -> float:
    if not frames:
        return 0.0
    good = 0
    for frame in frames:
        roles = frame.force_bearing_contact_roles
        good += int(frame.tool_object_contact_active and all(any(name in roles for name in group) for group in groups))
    return good / len(frames)


def _finite_optional(values: Sequence[float | None]) -> bool:
    return bool(values) and all(value is not None and math.isfinite(float(value)) for value in values)


def _sampled_duration_s(frames: Sequence[EpisodeFrame]) -> float:
    """Duration covered by post-step samples, including the first sampled step."""
    if len(frames) < 2:
        return 0.0
    elapsed = float(frames[-1].simulation_time_s) - float(frames[0].simulation_time_s)
    intervals = [
        (float(later.simulation_time_s) - float(earlier.simulation_time_s))
        / (int(later.physics_step_index) - int(earlier.physics_step_index))
        for earlier, later in zip(frames, frames[1:])
        if int(later.physics_step_index) > int(earlier.physics_step_index)
        and float(later.simulation_time_s) > float(earlier.simulation_time_s)
    ]
    covered = elapsed + (sorted(intervals)[len(intervals) // 2] if intervals else 0.0)
    return max(0.0, round(covered, 9))


def _upright_error_rad(quaternion_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    world_vertical_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, world_vertical_z)))


def evaluate_episode_evidence(
    frames: Sequence[EpisodeFrame],
    criteria: EpisodeCriteria,
) -> EpisodeGateResult:
    """Gate measured post-step evidence; commands and kinematic playback are insufficient."""
    statuses = {phase.value: "NOT_RUN" for phase in PHASE_ORDER}
    if not frames:
        return EpisodeGateResult("NOT_RUN", statuses, ("NO_POST_STEP_EVIDENCE",), 0)

    blockers: list[str] = []
    phase_ranks = [PHASE_ORDER.index(frame.phase) for frame in frames]
    if phase_ranks != sorted(phase_ranks):
        blockers.append("PHASE_ORDER_INVALID")
    times = [float(frame.simulation_time_s) for frame in frames]
    steps = [int(frame.physics_step_index) for frame in frames]
    if any(not math.isfinite(value) for value in times) or any(b <= a for a, b in zip(times, times[1:])):
        blockers.append("SIMULATION_TIME_ORDER_INVALID")
    if any(step <= 0 for step in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
        blockers.append("POST_STEP_SAMPLE_REQUIRED")

    identities = {(frame.model_id, frame.model_sha256) for frame in frames}
    if len(identities) != 1 or any(
        not model_id or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower())
        for model_id, sha in identities
    ):
        blockers.append("MODEL_IDENTITY_MISSING_OR_CHANGED")

    for index, frame in enumerate(frames):
        if len(frame.bottle_com_world_m) != 3 or not all(math.isfinite(float(v)) for v in frame.bottle_com_world_m):
            blockers.append(f"BOTTLE_POSE_MISSING:{index}")
        quat = frame.bottle_quat_world_xyzw
        quat_norm = math.sqrt(sum(float(value) ** 2 for value in quat)) if len(quat) == 4 else 0.0
        if not all(math.isfinite(float(value)) for value in quat) or abs(quat_norm - 1.0) > 1e-3:
            blockers.append(f"BOTTLE_ORIENTATION_MISSING_OR_INVALID:{index}")
        if frame.left_hand_contact is None:
            blockers.append(f"LEFT_HAND_CONTACT_EVIDENCE_MISSING:{index}")
        elif frame.left_hand_contact:
            blockers.append(f"LEFT_HAND_CONTACT:{index}")
        if frame.bottle_qpos_writes_after_start is None:
            blockers.append(f"BOTTLE_QPOS_WRITE_EVIDENCE_MISSING:{index}")
        elif frame.bottle_qpos_writes_after_start != 0:
            blockers.append(f"BOTTLE_QPOS_WRITE:{index}")
        if any(value is None for value in (frame.bottle_attachment_active, frame.mocap_carry_active, frame.hidden_support_active)):
            blockers.append(f"OBJECT_SUPPORT_INTEGRITY_EVIDENCE_MISSING:{index}")
        elif frame.bottle_attachment_active or frame.mocap_carry_active or frame.hidden_support_active:
            blockers.append(f"ARTIFICIAL_OBJECT_SUPPORT:{index}")
        if frame.source_position_limits_pass is None:
            blockers.append(f"SOURCE_POSITION_LIMIT_EVIDENCE_MISSING:{index}")
        elif not frame.source_position_limits_pass:
            blockers.append(f"SOURCE_POSITION_LIMIT:{index}")
        if frame.source_velocity_limits_pass is None:
            blockers.append(f"SOURCE_VELOCITY_LIMIT_EVIDENCE_MISSING:{index}")
        elif not frame.source_velocity_limits_pass:
            blockers.append(f"SOURCE_VELOCITY_LIMIT:{index}")
        if frame.actuator_limits_pass is None:
            blockers.append(f"ACTUATOR_LIMIT_EVIDENCE_MISSING:{index}")
        elif not frame.actuator_limits_pass:
            blockers.append(f"ACTUATOR_LIMIT:{index}")
        if frame.follower_constraints_pass is None:
            blockers.append(f"FOLLOWER_CONSTRAINT_EVIDENCE_MISSING:{index}")
        elif not frame.follower_constraints_pass:
            blockers.append(f"FOLLOWER_CONSTRAINT:{index}")

    by_phase = {phase: [frame for frame in frames if frame.phase is phase] for phase in PHASE_ORDER}
    for phase in PHASE_ORDER[:2]:
        group = by_phase[phase]
        if group:
            if any(frame.minimum_approach_clearance_m is None for frame in group):
                statuses[phase.value] = "BLOCKED_MISSING_EVIDENCE"
                blockers.append(f"{phase.value}_CLEARANCE_EVIDENCE_MISSING")
                continue
            clear = all(
                math.isfinite(frame.minimum_approach_clearance_m)
                and frame.minimum_approach_clearance_m >= criteria.minimum_approach_clearance_m
                for frame in group
            )
            if any(frame.tool_object_contact_active is None for frame in group):
                statuses[phase.value] = "BLOCKED_MISSING_EVIDENCE"
                blockers.append(f"{phase.value}_OBJECT_CONTACT_EVIDENCE_MISSING")
                continue
            contact_free = not any(frame.tool_object_contact_active is True for frame in group)
            statuses[phase.value] = "PASS" if clear else "FAIL"
            if not clear:
                blockers.append(f"{phase.value}_COLLISION_CLEARANCE")
            if not contact_free:
                statuses[phase.value] = "FAIL"
                blockers.append(f"{phase.value}_PREMATURE_OBJECT_CONTACT")

    contact = by_phase[ManipulationPhase.CONTACT]
    contact_fraction = _contact_fraction(contact, criteria.required_contact_groups)
    if contact:
        if any(frame.tool_object_contact_active is None for frame in contact):
            statuses[ManipulationPhase.CONTACT.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("CONTACT_PAIR_EVIDENCE_MISSING")
        else:
            statuses[ManipulationPhase.CONTACT.value] = "PASS" if contact_fraction >= criteria.minimum_contact_fraction else "FAIL"
        if statuses[ManipulationPhase.CONTACT.value] == "FAIL":
            blockers.append("FORCE_BEARING_CONTACT_PERSISTENCE")

    load = by_phase[ManipulationPhase.LOAD_TRANSFER]
    if load:
        required = _finite_optional([frame.tool_upward_support_n for frame in load])
        baseline = _finite_optional([frame.table_reaction_vertical_n for frame in contact])
        loaded = _finite_optional([frame.table_reaction_vertical_n for frame in load])
        slip = _finite_optional([frame.max_relative_slip_m for frame in load])
        if required and baseline and loaded and slip and all(frame.tool_object_contact_active is not None for frame in load):
            support_pass = min(float(frame.tool_upward_support_n) for frame in load) >= criteria.minimum_tool_upward_support_n
            unloading = mean(float(frame.table_reaction_vertical_n) for frame in contact) - mean(
                float(frame.table_reaction_vertical_n) for frame in load
            )
            unloading_pass = unloading >= criteria.minimum_table_unloading_n
            contact_ok = _contact_fraction(load, criteria.required_contact_groups) >= criteria.minimum_contact_fraction
            slip_ok = max(float(frame.max_relative_slip_m) for frame in load) <= criteria.maximum_relative_slip_m
            statuses[ManipulationPhase.LOAD_TRANSFER.value] = (
                "PASS" if support_pass and unloading_pass and contact_ok and slip_ok else "FAIL"
            )
            if not support_pass:
                blockers.append("MEASURED_UPWARD_TOOL_SUPPORT")
            if not unloading_pass:
                blockers.append("MEASURED_TABLE_UNLOADING")
            if not contact_ok:
                blockers.append("LOAD_TRANSFER_CONTACT_PERSISTENCE")
            if not slip_ok:
                blockers.append("LOAD_TRANSFER_RELATIVE_SLIP")
        else:
            statuses[ManipulationPhase.LOAD_TRANSFER.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("LOAD_TRANSFER_FORCE_EVIDENCE_MISSING")

    contact_base = contact[0].bottle_com_world_m[2] if contact else None
    lift = by_phase[ManipulationPhase.LIFT_50MM]
    if lift:
        support_measured = _finite_optional([frame.tool_upward_support_n for frame in lift])
        table_reaction_measured = _finite_optional([frame.table_reaction_vertical_n for frame in lift])
        slip_measured = _finite_optional([frame.max_relative_slip_m for frame in lift])
        table_contact_measured = all(frame.table_contact is not None for frame in lift)
        contact_state_measured = all(frame.tool_object_contact_active is not None for frame in lift)
        if contact_base is None or not support_measured or not table_reaction_measured or not slip_measured or not table_contact_measured or not contact_state_measured:
            statuses[ManipulationPhase.LIFT_50MM.value] = "BLOCKED_MISSING_EVIDENCE"
            if contact_base is None:
                blockers.append("LIFT_CONTACT_BASELINE_MISSING")
            if not support_measured:
                blockers.append("LIFT_UPWARD_SUPPORT_EVIDENCE_MISSING")
            if not table_reaction_measured:
                blockers.append("LIFT_TABLE_REACTION_EVIDENCE_MISSING")
            if not slip_measured:
                blockers.append("LIFT_SLIP_EVIDENCE_MISSING")
            if not table_contact_measured:
                blockers.append("LIFT_TABLE_CONTACT_EVIDENCE_MISSING")
            if not contact_state_measured:
                blockers.append("LIFT_CONTACT_STATE_EVIDENCE_MISSING")
        else:
            rise = lift[-1].bottle_com_world_m[2] - contact_base
            contact_ok = _contact_fraction(lift, criteria.required_contact_groups) >= criteria.minimum_contact_fraction
            support_ok = min(float(frame.tool_upward_support_n) for frame in lift) >= criteria.minimum_tool_upward_support_n
            table_clear = not any(frame.table_contact for frame in lift)
            slip_ok = max(float(frame.max_relative_slip_m) for frame in lift) <= criteria.maximum_relative_slip_m
            passed = rise >= criteria.minimum_lift_m and contact_ok and support_ok and table_clear and slip_ok
            statuses[ManipulationPhase.LIFT_50MM.value] = "PASS" if passed else "FAIL"
            if rise < criteria.minimum_lift_m:
                blockers.append("MEASURED_BOTTLE_LIFT_BELOW_50MM")
            if not contact_ok:
                blockers.append("LIFT_CONTACT_PERSISTENCE")
            if not support_ok:
                blockers.append("LIFT_MEASURED_UPWARD_TOOL_SUPPORT")
            if not table_clear:
                blockers.append("BOTTLE_STILL_TABLE_SUPPORTED_DURING_LIFT")
            if not slip_ok:
                blockers.append("LIFT_RELATIVE_SLIP")

    hold = by_phase[ManipulationPhase.HOLD]
    if hold:
        support_measured = _finite_optional([frame.tool_upward_support_n for frame in hold])
        table_reaction_measured = _finite_optional([frame.table_reaction_vertical_n for frame in hold])
        slip_measured = _finite_optional([frame.max_relative_slip_m for frame in hold])
        table_contact_measured = all(frame.table_contact is not None for frame in hold)
        contact_state_measured = all(frame.tool_object_contact_active is not None for frame in hold)
        if contact_base is None or not support_measured or not table_reaction_measured or not slip_measured or not table_contact_measured or not contact_state_measured:
            statuses[ManipulationPhase.HOLD.value] = "BLOCKED_MISSING_EVIDENCE"
            if contact_base is None:
                blockers.append("HOLD_CONTACT_BASELINE_MISSING")
            if not support_measured:
                blockers.append("HOLD_UPWARD_SUPPORT_EVIDENCE_MISSING")
            if not table_reaction_measured:
                blockers.append("HOLD_TABLE_REACTION_EVIDENCE_MISSING")
            if not slip_measured:
                blockers.append("HOLD_SLIP_EVIDENCE_MISSING")
            if not table_contact_measured:
                blockers.append("HOLD_TABLE_CONTACT_EVIDENCE_MISSING")
            if not contact_state_measured:
                blockers.append("HOLD_CONTACT_STATE_EVIDENCE_MISSING")
        else:
            duration = _sampled_duration_s(hold)
            minimum_rise = min(frame.bottle_com_world_m[2] - contact_base for frame in hold)
            contact_ok = _contact_fraction(hold, criteria.required_contact_groups) >= criteria.minimum_contact_fraction
            support_ok = min(float(frame.tool_upward_support_n) for frame in hold) >= criteria.minimum_tool_upward_support_n
            table_clear = not any(frame.table_contact for frame in hold)
            slip_ok = max(float(frame.max_relative_slip_m) for frame in hold) <= criteria.maximum_relative_slip_m
            passed = (duration >= criteria.minimum_hold_duration_s and minimum_rise >= criteria.minimum_lift_m
                      and contact_ok and support_ok and table_clear and slip_ok)
            statuses[ManipulationPhase.HOLD.value] = "PASS" if passed else "FAIL"
            if duration < criteria.minimum_hold_duration_s:
                blockers.append("AIRBORNE_HOLD_DURATION")
            if minimum_rise < criteria.minimum_lift_m:
                blockers.append("AIRBORNE_HOLD_HEIGHT")
            if not contact_ok:
                blockers.append("HOLD_CONTACT_PERSISTENCE")
            if not support_ok:
                blockers.append("HOLD_MEASURED_UPWARD_TOOL_SUPPORT")
            if not table_clear:
                blockers.append("BOTTLE_TABLE_CONTACT_DURING_HOLD")
            if not slip_ok:
                blockers.append("HOLD_RELATIVE_SLIP")

    transfer = by_phase[ManipulationPhase.TRANSFER]
    if transfer:
        support = _finite_optional([frame.tool_upward_support_n for frame in transfer])
        table_reaction = _finite_optional([frame.table_reaction_vertical_n for frame in transfer])
        slip = _finite_optional([frame.max_relative_slip_m for frame in transfer])
        table_state = all(frame.table_contact is not None for frame in transfer)
        contact_state = all(frame.tool_object_contact_active is not None for frame in transfer)
        if not support or not table_reaction or not slip or not table_state or not contact_state:
            statuses[ManipulationPhase.TRANSFER.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("TRANSFER_MEASUREMENTS_MISSING")
        else:
            contact_ok = _contact_fraction(transfer, criteria.required_contact_groups) >= criteria.minimum_contact_fraction
            support_ok = min(float(frame.tool_upward_support_n) for frame in transfer) >= criteria.minimum_tool_upward_support_n
            table_clear = not any(frame.table_contact is True for frame in transfer)
            slip_ok = max(float(frame.max_relative_slip_m) for frame in transfer) <= criteria.maximum_relative_slip_m
            passed = contact_ok and support_ok and table_clear and slip_ok
            statuses[ManipulationPhase.TRANSFER.value] = "PASS" if passed else "FAIL"
            if not contact_ok:
                blockers.append("TRANSFER_CONTACT_PERSISTENCE")
            if not support_ok:
                blockers.append("TRANSFER_MEASURED_UPWARD_TOOL_SUPPORT")
            if not table_clear:
                blockers.append("TRANSFER_TABLE_SUPPORT_PRESENT")
            if not slip_ok:
                blockers.append("TRANSFER_RELATIVE_SLIP")

    place = by_phase[ManipulationPhase.PLACE]
    if place:
        target_rows = [frame.target_bottle_com_world_m for frame in place]
        table_reaction = _finite_optional([frame.table_reaction_vertical_n for frame in place])
        table_state = all(frame.table_contact is not None for frame in place)
        if (criteria.place_target_xy_world_m is None or not _finite_optional(
                [value for position in target_rows if position is not None for value in position])
                or any(position is None for position in target_rows) or not table_reaction or not table_state):
            statuses[ManipulationPhase.PLACE.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("PLACE_TARGET_OR_TABLE_EVIDENCE_MISSING")
        else:
            final = place[-1]
            xy_error = math.hypot(
                final.bottle_com_world_m[0] - criteria.place_target_xy_world_m[0],
                final.bottle_com_world_m[1] - criteria.place_target_xy_world_m[1],
            )
            upright_error = _upright_error_rad(final.bottle_quat_world_xyzw)
            supported = any(frame.table_contact is True for frame in place)
            passed = (xy_error <= criteria.maximum_place_xy_error_m
                      and upright_error <= criteria.maximum_upright_error_rad and supported)
            statuses[ManipulationPhase.PLACE.value] = "PASS" if passed else "FAIL"
            if xy_error > criteria.maximum_place_xy_error_m:
                blockers.append("PLACE_TARGET_XY_ERROR")
            if upright_error > criteria.maximum_upright_error_rad:
                blockers.append("PLACE_UPRIGHT_ERROR")
            if not supported:
                blockers.append("PLACE_TABLE_SUPPORT_MISSING")

    release = by_phase[ManipulationPhase.RELEASE]
    if release:
        opening = _finite_optional([frame.tool_opening_m for frame in release])
        table_reaction = _finite_optional([frame.table_reaction_vertical_n for frame in release])
        table_state = all(frame.table_contact is not None for frame in release)
        contact_state = all(frame.tool_object_contact_active is not None for frame in release)
        if not opening or not table_reaction or not table_state or not contact_state:
            statuses[ManipulationPhase.RELEASE.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("RELEASE_OPENING_OR_CONTACT_EVIDENCE_MISSING")
        else:
            opened = min(float(frame.tool_opening_m) for frame in release) >= criteria.minimum_release_opening_m
            contact_lost = not any(frame.tool_object_contact_active is True for frame in release)
            supported = any(frame.table_contact is True for frame in release)
            passed = opened and contact_lost and supported
            statuses[ManipulationPhase.RELEASE.value] = "PASS" if passed else "FAIL"
            if not opened:
                blockers.append("RELEASE_FINGER_OPENING")
            if not contact_lost:
                blockers.append("RELEASE_CONTACT_PERSISTED")
            if not supported:
                blockers.append("RELEASE_TABLE_SUPPORT_MISSING")

    settle = by_phase[ManipulationPhase.SETTLE]
    if settle:
        speeds_measured = _finite_optional([
            value for frame in settle for value in (frame.bottle_linear_speed_m_s, frame.bottle_angular_speed_rad_s)
        ])
        table_state = all(frame.table_contact is not None for frame in settle)
        contact_state = all(frame.tool_object_contact_active is not None for frame in settle)
        if not speeds_measured or not table_state or not contact_state:
            statuses[ManipulationPhase.SETTLE.value] = "BLOCKED_MISSING_EVIDENCE"
            blockers.append("SETTLE_SPEED_OR_CONTACT_EVIDENCE_MISSING")
        else:
            duration = _sampled_duration_s(settle)
            drift = max(math.dist(frame.bottle_com_world_m, settle[0].bottle_com_world_m) for frame in settle)
            table_supported = all(frame.table_contact is True for frame in settle)
            contact_lost = not any(frame.tool_object_contact_active is True for frame in settle)
            speed_pass = (max(float(frame.bottle_linear_speed_m_s) for frame in settle)
                          <= criteria.maximum_settle_linear_speed_m_s
                          and max(float(frame.bottle_angular_speed_rad_s) for frame in settle)
                          <= criteria.maximum_settle_angular_speed_rad_s)
            passed = (duration >= criteria.minimum_settle_duration_s and drift <= criteria.maximum_settle_position_drift_m
                      and table_supported and contact_lost and speed_pass)
            statuses[ManipulationPhase.SETTLE.value] = "PASS" if passed else "FAIL"
            if duration < criteria.minimum_settle_duration_s:
                blockers.append("SETTLE_DWELL_TOO_SHORT")
            if drift > criteria.maximum_settle_position_drift_m:
                blockers.append("SETTLE_POSITION_DRIFT")
            if not table_supported:
                blockers.append("SETTLE_TABLE_SUPPORT_MISSING")
            if not contact_lost:
                blockers.append("SETTLE_TOOL_CONTACT_REMAINS")
            if not speed_pass:
                blockers.append("SETTLE_SPEED_ABOVE_LIMIT")

    required = tuple(phase.value for phase in PHASE_ORDER)
    missing = [name for name in required if statuses[name] == "NOT_RUN"]
    if missing:
        blockers.append("STAGES_NOT_RUN:" + ",".join(missing))
    if blockers:
        return EpisodeGateResult("BLOCKED", statuses, tuple(dict.fromkeys(blockers)), max(steps, default=0))
    return EpisodeGateResult("PASS", statuses, (), max(steps, default=0))
