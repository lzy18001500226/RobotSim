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
    force_bearing_contact_roles: frozenset[str]
    tool_object_contact_active: bool
    left_hand_contact: bool
    table_contact: bool
    tool_upward_support_n: float | None
    table_reaction_vertical_n: float | None
    max_relative_slip_m: float | None
    minimum_approach_clearance_m: float | None
    source_position_limits_pass: bool
    source_velocity_limits_pass: bool
    actuator_limits_pass: bool
    follower_constraints_pass: bool
    bottle_qpos_writes_after_start: int
    bottle_attachment_active: bool
    mocap_carry_active: bool
    hidden_support_active: bool


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
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("episode criteria must be finite and nonnegative")
        if self.minimum_contact_fraction > 1.0:
            raise ValueError("minimum_contact_fraction must be in [0, 1]")
        if (self.minimum_contact_fraction <= 0.0 or self.minimum_tool_upward_support_n <= 0.0
                or self.minimum_table_unloading_n <= 0.0 or self.minimum_lift_m < 0.05
                or self.minimum_hold_duration_s < 1.0 or self.minimum_approach_clearance_m < 0.002):
            raise ValueError("criteria may not weaken contact, load, clearance, 50 mm lift, or 1 s hold gates")


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
        if frame.left_hand_contact:
            blockers.append(f"LEFT_HAND_CONTACT:{index}")
        if frame.bottle_qpos_writes_after_start != 0:
            blockers.append(f"BOTTLE_QPOS_WRITE:{index}")
        if frame.bottle_attachment_active or frame.mocap_carry_active or frame.hidden_support_active:
            blockers.append(f"ARTIFICIAL_OBJECT_SUPPORT:{index}")
        if not frame.source_position_limits_pass:
            blockers.append(f"SOURCE_POSITION_LIMIT:{index}")
        if not frame.source_velocity_limits_pass:
            blockers.append(f"SOURCE_VELOCITY_LIMIT:{index}")
        if not frame.actuator_limits_pass:
            blockers.append(f"ACTUATOR_LIMIT:{index}")
        if not frame.follower_constraints_pass:
            blockers.append(f"FOLLOWER_CONSTRAINT:{index}")

    by_phase = {phase: [frame for frame in frames if frame.phase is phase] for phase in PHASE_ORDER}
    for phase in PHASE_ORDER[:2]:
        group = by_phase[phase]
        if group:
            clear = all(
                frame.minimum_approach_clearance_m is not None
                and math.isfinite(frame.minimum_approach_clearance_m)
                and frame.minimum_approach_clearance_m >= criteria.minimum_approach_clearance_m
                for frame in group
            )
            contact_free = not any(frame.tool_object_contact_active for frame in group)
            statuses[phase.value] = "PASS" if clear else "FAIL"
            if not clear:
                blockers.append(f"{phase.value}_COLLISION_CLEARANCE")
            if not contact_free:
                statuses[phase.value] = "FAIL"
                blockers.append(f"{phase.value}_PREMATURE_OBJECT_CONTACT")

    contact = by_phase[ManipulationPhase.CONTACT]
    contact_fraction = _contact_fraction(contact, criteria.required_contact_groups)
    if contact:
        statuses[ManipulationPhase.CONTACT.value] = (
            "PASS" if contact_fraction >= criteria.minimum_contact_fraction else "FAIL"
        )
        if statuses[ManipulationPhase.CONTACT.value] == "FAIL":
            blockers.append("FORCE_BEARING_CONTACT_PERSISTENCE")

    load = by_phase[ManipulationPhase.LOAD_TRANSFER]
    if load:
        required = _finite_optional([frame.tool_upward_support_n for frame in load])
        baseline = _finite_optional([frame.table_reaction_vertical_n for frame in contact])
        loaded = _finite_optional([frame.table_reaction_vertical_n for frame in load])
        slip = _finite_optional([frame.max_relative_slip_m for frame in load])
        if required and baseline and loaded and slip:
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
            statuses[ManipulationPhase.LOAD_TRANSFER.value] = "FAIL"
            blockers.append("LOAD_TRANSFER_FORCE_EVIDENCE_MISSING")

    contact_base = contact[0].bottle_com_world_m[2] if contact else None
    lift = by_phase[ManipulationPhase.LIFT_50MM]
    if lift:
        support_measured = _finite_optional([frame.tool_upward_support_n for frame in lift])
        table_reaction_measured = _finite_optional([frame.table_reaction_vertical_n for frame in lift])
        slip_measured = _finite_optional([frame.max_relative_slip_m for frame in lift])
        if contact_base is None or not support_measured or not table_reaction_measured or not slip_measured:
            statuses[ManipulationPhase.LIFT_50MM.value] = "FAIL"
            if contact_base is None:
                blockers.append("LIFT_CONTACT_BASELINE_MISSING")
            if not support_measured:
                blockers.append("LIFT_UPWARD_SUPPORT_EVIDENCE_MISSING")
            if not table_reaction_measured:
                blockers.append("LIFT_TABLE_REACTION_EVIDENCE_MISSING")
            if not slip_measured:
                blockers.append("LIFT_SLIP_EVIDENCE_MISSING")
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
        if contact_base is None or not support_measured or not table_reaction_measured or not slip_measured:
            statuses[ManipulationPhase.HOLD.value] = "FAIL"
            if contact_base is None:
                blockers.append("HOLD_CONTACT_BASELINE_MISSING")
            if not support_measured:
                blockers.append("HOLD_UPWARD_SUPPORT_EVIDENCE_MISSING")
            if not table_reaction_measured:
                blockers.append("HOLD_TABLE_REACTION_EVIDENCE_MISSING")
            if not slip_measured:
                blockers.append("HOLD_SLIP_EVIDENCE_MISSING")
        else:
            duration = hold[-1].simulation_time_s - hold[0].simulation_time_s
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

    required = tuple(phase.value for phase in PHASE_ORDER)
    missing = [name for name in required if statuses[name] == "NOT_RUN"]
    if missing:
        blockers.append("STAGES_NOT_RUN:" + ",".join(missing))
    if blockers:
        return EpisodeGateResult("BLOCKED", statuses, tuple(dict.fromkeys(blockers)), max(steps, default=0))
    return EpisodeGateResult("PASS", statuses, (), max(steps, default=0))
