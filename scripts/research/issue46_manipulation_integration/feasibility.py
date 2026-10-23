"""Fail-closed, static-only grasp geometry feasibility contract."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Mapping

from .approach import ToolPose
from .contracts import ContractError, PoseSample


class FeasibilityError(ValueError):
    pass


class StaticGateStatus(str, Enum):
    PASS_STATIC_GEOMETRY = "PASS_STATIC_GEOMETRY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ObjectGeometry:
    object_id: str
    shape: str
    dimensions_m: tuple[float, ...]
    object_frame: str
    collision_geom_names: tuple[str, ...]
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        expected = {"cylinder": 2, "box": 3, "mesh": 3}
        if not self.object_id or not self.object_frame:
            raise FeasibilityError("object_id and object_frame are required")
        if self.shape not in expected or len(self.dimensions_m) != expected[self.shape]:
            raise FeasibilityError("shape must have its explicit dimensions: cylinder=(diameter,height), box/mesh=(x,y,z)")
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in self.dimensions_m):
            raise FeasibilityError("object dimensions must be finite positive meters")
        if not self.collision_geom_names or len(set(self.collision_geom_names)) != len(self.collision_geom_names):
            raise FeasibilityError("object collision geom inventory must be nonempty and unique")
        if self.mass_kg is not None and (not math.isfinite(self.mass_kg) or self.mass_kg <= 0):
            raise FeasibilityError("object mass must be finite and positive")


@dataclass(frozen=True)
class ContactBand:
    """Candidate contact interval along a normalized object-local axis."""
    object_frame: str
    axis_xyz: tuple[float, float, float]
    minimum_m: float
    maximum_m: float
    status: str = "CANDIDATE_NOT_PHYSICALLY_VALIDATED"

    def __post_init__(self) -> None:
        if not self.object_frame or not math.isfinite(self.minimum_m) or not math.isfinite(self.maximum_m):
            raise FeasibilityError("contact band frame and limits are required")
        if self.minimum_m > self.maximum_m:
            raise FeasibilityError("contact band minimum exceeds maximum")
        if len(self.axis_xyz) != 3 or not all(math.isfinite(float(v)) for v in self.axis_xyz):
            raise FeasibilityError("contact band axis must have three finite values")
        if abs(math.sqrt(sum(float(v)**2 for v in self.axis_xyz)) - 1.0) > 1e-6:
            raise FeasibilityError("contact band axis must be normalized")


@dataclass(frozen=True)
class JointCommandProfile:
    """Observed/configured joint state at one input sample, including passive followers.

    The map records kinematic state for collision evaluation; it does not authorize
    independently commanding or writing a mimic/follower joint.
    """
    step_index: int
    command_value: float
    joint_positions_rad: Mapping[str, float]


@dataclass(frozen=True)
class GripperContract:
    model_identity: str
    command_name: str
    command_min: float
    command_max: float
    open_value: float
    close_value: float
    joint_names: tuple[str, ...]
    commanded_joint_names: tuple[str, ...]
    derived_joint_constraints: Mapping[str, str]
    source_joint_limits_rad: Mapping[str, tuple[float, float]]
    open_joint_positions_rad: Mapping[str, float]
    close_profile_open_to_close: tuple[JointCommandProfile, ...]
    range_evidence_ref: str
    range_verified_static: bool
    pad_geoms_by_side: Mapping[str, tuple[str, ...]]
    linkage_geom_names: tuple[str, ...]
    other_gripper_geom_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (self.command_min, self.command_max, self.open_value, self.close_value)
        if not self.model_identity or not self.command_name or not all(math.isfinite(v) for v in values):
            raise FeasibilityError("gripper model identity and finite command range are required")
        if not self.range_evidence_ref or not isinstance(self.range_verified_static, bool):
            raise FeasibilityError("gripper open/close range needs an evidence reference and explicit verification status")
        if self.command_min >= self.command_max:
            raise FeasibilityError("gripper command range is empty")
        if not self.command_min <= self.open_value <= self.command_max:
            raise FeasibilityError("open command lies outside the supplied range")
        if not self.command_min <= self.close_value <= self.command_max or self.open_value == self.close_value:
            raise FeasibilityError("close command must lie in range and differ from open")
        direction = self.close_value-self.open_value
        if set(self.pad_geoms_by_side) != {"narrow", "wide"}:
            raise FeasibilityError("opposing pad groups named narrow and wide are required")
        if not all(self.pad_geoms_by_side.values()) or not self.linkage_geom_names:
            raise FeasibilityError("pad and linkage collision geometry names are required")
        names = set(self.joint_names)
        commanded = set(self.commanded_joint_names)
        derived = set(self.derived_joint_constraints)
        if not names or len(names) != len(self.joint_names):
            raise FeasibilityError("gripper joint_names must be nonempty and unique")
        if not commanded or len(commanded) != len(self.commanded_joint_names) or not commanded <= names:
            raise FeasibilityError("commanded joint inventory is invalid")
        if (not derived <= names or commanded & derived or commanded | derived != names
                or any(not value for value in self.derived_joint_constraints.values())):
            raise FeasibilityError("every non-commanded joint must name its mimic/equality/loop constraint")
        if not commanded <= set(self.source_joint_limits_rad) or not set(self.source_joint_limits_rad) <= names:
            raise FeasibilityError("each commanded joint needs a source range; other declared ranges must belong to this gripper")
        all_geoms = self.all_collision_geom_names
        if len(all_geoms) != len(set(all_geoms)):
            raise FeasibilityError("gripper collision geom groups must not overlap or duplicate names")
        _validate_complete_joint_map(self.open_joint_positions_rad, self.joint_names,
                                     self.source_joint_limits_rad, derived, "open")
        if len(self.close_profile_open_to_close) < 2:
            raise FeasibilityError("closure profile must include open and close endpoints")
        profile = self.close_profile_open_to_close
        if [x.step_index for x in profile] != list(range(len(profile))):
            raise FeasibilityError("closure sample indices must be contiguous from zero")
        if not math.isclose(profile[0].command_value, self.open_value, abs_tol=1e-12):
            raise FeasibilityError("closure profile must start at the stated open command")
        if not math.isclose(profile[-1].command_value, self.close_value, abs_tol=1e-12):
            raise FeasibilityError("closure profile must end at the stated close command")
        if dict(profile[0].joint_positions_rad) != dict(self.open_joint_positions_rad):
            raise FeasibilityError("profile open state must match open_joint_positions_rad")
        for sample in profile:
            if not math.isfinite(sample.command_value) or not self.command_min <= sample.command_value <= self.command_max:
                raise FeasibilityError("closure command outside supplied range")
            _validate_complete_joint_map(sample.joint_positions_rad, self.joint_names,
                                         self.source_joint_limits_rad, derived, "closure")
        progress = [(x.command_value-self.open_value)/direction for x in profile]
        if progress[0] != 0.0 or progress[-1] != 1.0 or any(b <= a for a, b in zip(progress, progress[1:])):
            raise FeasibilityError("closure commands must progress monotonically from open to close")

    @property
    def all_collision_geom_names(self) -> tuple[str, ...]:
        return tuple(name for side in ("narrow", "wide") for name in self.pad_geoms_by_side[side]) + tuple(
            self.linkage_geom_names) + tuple(self.other_gripper_geom_names)


def _validate_complete_joint_map(values: Mapping[str, float], joint_names: tuple[str, ...],
                                 limits: Mapping[str, tuple[float, float]], derived: set[str], label: str) -> None:
    if set(values) != set(joint_names):
        raise FeasibilityError(f"{label} joint map must provide every and only named gripper joint")
    for name, value in values.items():
        if name not in limits:
            if name not in derived or not math.isfinite(float(value)):
                raise FeasibilityError(f"joint {name} lacks a source range and a declared derived constraint")
            continue
        lo, hi = limits[name]
        if not all(math.isfinite(float(v)) for v in (lo, hi, value)) or lo >= hi:
            raise FeasibilityError(f"invalid source joint range/value for {name}")
        if value < lo or value > hi:
            raise FeasibilityError(f"joint {name} position {value} outside [{lo}, {hi}]")


@dataclass(frozen=True)
class GeometryPair:
    geom_a: str
    geom_b: str
    signed_distance_m: float
    classification: str

    def __post_init__(self) -> None:
        if not self.geom_a or not self.geom_b or self.geom_a == self.geom_b or not self.classification:
            raise FeasibilityError("collision pair needs two distinct geom names")
        if not math.isfinite(self.signed_distance_m):
            raise FeasibilityError("collision pair distance must be finite")

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.geom_a, self.geom_b)))


def collision_inventory_digest(model_identity: str, geom_names: tuple[str, ...],
                               enabled_pairs: tuple[tuple[str, str], ...]) -> str:
    canonical = {
        "model_identity": model_identity,
        "geom_names": sorted(geom_names),
        "enabled_pairs": sorted(tuple(sorted(pair)) for pair in enabled_pairs),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CollisionInventory:
    """Complete enabled-pair inventory emitted from one compiled collision model."""
    model_identity: str
    geom_names: tuple[str, ...]
    enabled_pairs: tuple[tuple[str, str], ...]
    sha256: str

    def __post_init__(self) -> None:
        if not self.model_identity or not self.geom_names or len(set(self.geom_names)) != len(self.geom_names):
            raise FeasibilityError("collision inventory needs model identity and unique geom names")
        pair_keys = tuple(tuple(sorted(x)) for x in self.enabled_pairs)
        if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(self.geom_names) for pair in pair_keys):
            raise FeasibilityError("collision inventory has an invalid pair or unknown geom")
        if len(set(pair_keys)) != len(pair_keys):
            raise FeasibilityError("collision inventory contains duplicate pairs")
        expected = collision_inventory_digest(self.model_identity, self.geom_names, self.enabled_pairs)
        if self.sha256 != expected:
            raise FeasibilityError("collision inventory SHA-256 does not match its geom/pair contents")


@dataclass(frozen=True)
class ApproachObservation:
    sample_index: int
    phase: str
    robot_model_identity: str
    tool_pose: ToolPose
    object_pose: PoseSample
    robot_joint_positions_rad: Mapping[str, float]
    collision_pairs: tuple[GeometryPair, ...]
    active_contact_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def minimum_clearance_m(self) -> float:
        return min((x.signed_distance_m for x in self.collision_pairs), default=math.inf)

    @property
    def nearest_pair(self) -> GeometryPair | None:
        return min(self.collision_pairs, key=lambda x: x.signed_distance_m, default=None)


@dataclass(frozen=True)
class ClosureObservation:
    step_index: int
    command_value: float
    pad_clearance_m_by_side: Mapping[str, float]
    pad_contact_points_object_m_by_side: Mapping[str, tuple[tuple[float, float, float], ...]]
    pad_contact_geom_names_by_side: Mapping[str, tuple[str, ...]]
    linkage_min_clearance_m: float
    checked_collision_pairs: tuple[tuple[str, str], ...]
    active_linkage_pairs: tuple[GeometryPair, ...] = ()
    other_contacts: tuple[GeometryPair, ...] = ()

    def __post_init__(self) -> None:
        if self.step_index < 0 or not math.isfinite(self.command_value):
            raise FeasibilityError("closure observation index and command must be valid")
        pair_keys = [tuple(sorted(x)) for x in self.checked_collision_pairs if len(x) == 2 and x[0] != x[1]]
        if len(pair_keys) != len(self.checked_collision_pairs) or len(pair_keys) != len(set(pair_keys)):
            raise FeasibilityError("closure observation collision pairs must be unique named pairs")
        for side in ("narrow", "wide"):
            for point in self.pad_contact_points_object_m_by_side.get(side, ()):
                if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
                    raise FeasibilityError("object-frame pad contact point must contain finite xyz")


@dataclass(frozen=True)
class FeasibilityThresholds:
    minimum_path_clearance_m: float
    contact_tolerance_m: float
    minimum_object_table_margin_m: float
    minimum_source_joint_margin_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (self.minimum_path_clearance_m, self.contact_tolerance_m,
                  self.minimum_object_table_margin_m, self.minimum_source_joint_margin_rad)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise FeasibilityError("feasibility thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class GripperFeasibilityRequest:
    robot_model_identity: str
    object_geometry: ObjectGeometry
    candidate_contact_band: ContactBand | None
    gripper: GripperContract
    robot_source_joint_limits_rad: Mapping[str, tuple[float, float]]
    collision_inventory: CollisionInventory
    evaluation_steady_ns: int
    object_pose_max_age_ns: int
    approach_samples: tuple[ApproachObservation, ...]
    closure_samples: tuple[ClosureObservation, ...]
    object_table_margin_m: float
    thresholds: FeasibilityThresholds


@dataclass(frozen=True)
class GripperFeasibilityResult:
    status: StaticGateStatus
    blockers: tuple[str, ...]
    first_obstruction: GeometryPair | None
    first_bilateral_contact_step: int | None
    first_linkage_contact_step: int | None
    evidence_scope: str = "STATIC_GEOMETRY_ONLY"

    @property
    def feasible(self) -> bool:
        return self.status is StaticGateStatus.PASS_STATIC_GEOMETRY


def _check_robot_joint_map(sample: ApproachObservation, limits: Mapping[str, tuple[float, float]], margin: float) -> bool:
    if set(sample.robot_joint_positions_rad) != set(limits):
        return False
    for name, q in sample.robot_joint_positions_rad.items():
        lo, hi = limits[name]
        if not all(math.isfinite(float(v)) for v in (q, lo, hi)) or lo >= hi:
            return False
        if q < lo or q > hi or min(q-lo, hi-q) < margin:
            return False
    return True


def evaluate_gripper_feasibility(request: GripperFeasibilityRequest) -> GripperFeasibilityResult:
    """Reject incomplete geometry evidence, obstructed paths, and loop-first closure.

    This is a geometry/IK gate only. No force, friction, stable-hold, table-unload,
    or physical-lift claim is inferred from a pose or closure sample.
    """
    blockers: list[str] = []
    first_obstruction: GeometryPair | None = None
    if not request.robot_model_identity or not request.robot_source_joint_limits_rad:
        blockers.append("ROBOT_MODEL_OR_SOURCE_LIMITS_MISSING")
    if not request.collision_inventory.enabled_pairs:
        blockers.append("REQUIRED_APPROACH_COLLISION_INVENTORY_MISSING")
    if request.collision_inventory.model_identity != request.robot_model_identity:
        blockers.append("COLLISION_INVENTORY_MODEL_MISMATCH")
    if not request.gripper.range_verified_static:
        blockers.append("GRIPPER_OPEN_CLOSE_RANGE_UNVERIFIED")
    if request.evaluation_steady_ns < 0 or request.object_pose_max_age_ns < 0:
        blockers.append("OBJECT_POSE_FRESHNESS_LIMIT_INVALID")
    required_approach = {tuple(sorted(x)) for x in request.collision_inventory.enabled_pairs}
    if not request.approach_samples:
        blockers.append("APPROACH_SAMPLES_MISSING")
    else:
        ordered = sorted(request.approach_samples, key=lambda x: x.sample_index)
        if [x.sample_index for x in ordered] != list(range(len(ordered))):
            blockers.append("APPROACH_SAMPLE_ORDER_INVALID")
        if ordered[0].phase != "PREGRASP" or ordered[-1].phase != "APPROACH":
            blockers.append("APPROACH_PHASE_ORDER_INVALID")
        for sample in ordered:
            if sample.robot_model_identity != request.robot_model_identity:
                blockers.append("ROBOT_MODEL_IDENTITY_MISMATCH")
            if sample.object_pose.object_id != request.object_geometry.object_id:
                blockers.append("OBJECT_POSE_IDENTITY_MISMATCH")
            try:
                sample.object_pose.require_fresh(request.evaluation_steady_ns, request.object_pose_max_age_ns)
            except ContractError:
                blockers.append("OBJECT_POSE_STALE_OR_UNTIMED")
            if sample.tool_pose.frame_id != sample.object_pose.frame_id:
                blockers.append("TOOL_OBJECT_FRAME_MISMATCH")
            keys = [x.key for x in sample.collision_pairs]
            if len(keys) != len(set(keys)) or set(keys) != required_approach:
                blockers.append("APPROACH_COLLISION_PAIR_COVERAGE_INCOMPLETE")
            if any(tuple(sorted(x)) not in required_approach for x in sample.active_contact_pairs):
                blockers.append("ACTIVE_CONTACT_OUTSIDE_COLLISION_INVENTORY")
            if not _check_robot_joint_map(sample, request.robot_source_joint_limits_rad,
                                          request.thresholds.minimum_source_joint_margin_rad):
                blockers.append("ROBOT_SOURCE_JOINT_LIMIT_OR_MAP")
            if sample.active_contact_pairs or sample.minimum_clearance_m < request.thresholds.minimum_path_clearance_m:
                if first_obstruction is None:
                    first_obstruction = sample.nearest_pair
                blockers.append("APPROACH_COLLISION_OR_CLEARANCE")
                break

    if (not math.isfinite(request.object_table_margin_m)
            or request.object_table_margin_m < request.thresholds.minimum_object_table_margin_m):
        blockers.append("OBJECT_TABLE_MARGIN")
    band = request.candidate_contact_band
    if band is None:
        blockers.append("CONTACT_BAND_UNSPECIFIED")
    elif band.object_frame != request.object_geometry.object_frame:
        blockers.append("CONTACT_BAND_FRAME_MISMATCH")

    expected_closure = {
        tuple(sorted((a, b)))
        for a in request.gripper.all_collision_geom_names
        for b in request.object_geometry.collision_geom_names
    }
    inventory_pairs = {tuple(sorted(x)) for x in request.collision_inventory.enabled_pairs}
    if not expected_closure <= inventory_pairs:
        blockers.append("GRIPPER_OBJECT_COLLISION_PAIR_DISABLED_OR_MISSING")
    closure = sorted(request.closure_samples, key=lambda x: x.step_index)
    if not closure:
        blockers.append("CLOSURE_SAMPLES_MISSING")
    if [x.step_index for x in closure] != list(range(len(closure))):
        blockers.append("CLOSURE_SAMPLE_ORDER_INVALID")
    first_bilateral: int | None = None
    first_linkage: int | None = None
    for sample in closure:
        if not math.isfinite(sample.command_value) or not request.gripper.command_min <= sample.command_value <= request.gripper.command_max:
            blockers.append("CLOSURE_COMMAND_OUTSIDE_RANGE")
        if set(tuple(sorted(x)) for x in sample.checked_collision_pairs) != expected_closure:
            blockers.append("CLOSURE_COLLISION_PAIR_COVERAGE_INCOMPLETE")
        if not math.isfinite(sample.linkage_min_clearance_m):
            blockers.append("LINKAGE_CLEARANCE_NONFINITE")
        pad_ok = True
        for side in ("narrow", "wide"):
            clearance = sample.pad_clearance_m_by_side.get(side)
            if clearance is None or not math.isfinite(clearance) or clearance > request.thresholds.contact_tolerance_m:
                pad_ok = False
            points = sample.pad_contact_points_object_m_by_side.get(side, ())
            contact_geoms = sample.pad_contact_geom_names_by_side.get(side, ())
            allowed_geoms = set(request.gripper.pad_geoms_by_side.get(side, ()))
            band = request.candidate_contact_band
            in_band = False
            if band is not None:
                in_band = any(
                    band.minimum_m <= sum(float(point[i])*band.axis_xyz[i] for i in range(3)) <= band.maximum_m
                    for point in points
                )
            if not points or not contact_geoms or not set(contact_geoms) <= allowed_geoms or not in_band:
                pad_ok = False
        if pad_ok and first_bilateral is None:
            first_bilateral = sample.step_index
        linkage_hit = bool(sample.active_linkage_pairs) or (
            math.isfinite(sample.linkage_min_clearance_m)
            and sample.linkage_min_clearance_m <= request.thresholds.contact_tolerance_m
        )
        if linkage_hit and first_linkage is None:
            first_linkage = sample.step_index
    if first_bilateral is None:
        blockers.append("BILATERAL_PAD_CONTACT_NOT_REACHED_IN_RANGE")
    if first_linkage is not None and (first_bilateral is None or first_linkage <= first_bilateral):
        blockers.append("LINKAGE_CONTACT_PRECEDES_BILATERAL_PADS")
    if any(x.other_contacts for x in closure):
        blockers.append("UNINTENDED_GRIPPER_OBJECT_CONTACT")
    deduped = tuple(dict.fromkeys(blockers))
    return GripperFeasibilityResult(
        status=StaticGateStatus.BLOCKED if deduped else StaticGateStatus.PASS_STATIC_GEOMETRY,
        blockers=deduped,
        first_obstruction=first_obstruction,
        first_bilateral_contact_step=first_bilateral,
        first_linkage_contact_step=first_linkage,
    )
