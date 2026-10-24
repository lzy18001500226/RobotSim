"""Transport-neutral pose, transform, and navigation-to-manipulation contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

Vector3 = tuple[float, float, float]
QuaternionXYWZ = tuple[float, float, float, float]


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class PoseSource(str, Enum):
    SIM_GROUND_TRUTH = "sim_ground_truth"
    MEASUREMENT = "measurement"
    ESTIMATE = "estimate"
    TASK_TARGET = "task_target"


class TimeDomain(str, Enum):
    """Clock which assigned acquisition_time_ns; domains are never compared."""
    SIMULATION = "simulation"
    ROS = "ros"
    SENSOR = "sensor"


@dataclass(frozen=True)
class PoseSample:
    """A pose is expressed in frame_id at acquisition_time_ns.

    received_steady_ns is local monotonic receipt time and is used only for
    freshness. It must never be compared with acquisition/simulation time.
    """
    object_id: str
    frame_id: str
    position_m: Vector3
    orientation_xyzw: QuaternionXYWZ
    acquisition_time_ns: int
    source: PoseSource
    time_domain: TimeDomain
    model_identity: str | None = None
    generation: str | None = None
    sequence: int | None = None
    received_steady_ns: int | None = None
    calibration_id: str | None = None
    covariance_6x6: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.object_id or not self.frame_id:
            raise ContractError("FRAME_OR_OBJECT_MISSING", "object_id and frame_id are required")
        if not isinstance(self.source, PoseSource) or not isinstance(self.time_domain, TimeDomain):
            raise ContractError("SOURCE_OR_CLOCK_INVALID", "pose source and acquisition clock must use declared enums")
        if self.acquisition_time_ns < 0:
            raise ContractError("INVALID_STAMP", "acquisition time must be nonnegative")
        _finite_tuple(self.position_m, 3, "position")
        q = _finite_tuple(self.orientation_xyzw, 4, "quaternion")
        norm = math.sqrt(sum(x*x for x in q))
        if abs(norm - 1.0) > 1e-6:
            raise ContractError("QUATERNION_NOT_NORMALIZED", "orientation must be normalized xyzw")
        if self.sequence is not None and self.sequence < 0:
            raise ContractError("INVALID_SEQUENCE", "sequence must be nonnegative")
        if (self.generation is None) != (self.sequence is None):
            raise ContractError("INCOMPLETE_EPOCH", "generation and sequence must be supplied together")
        if self.generation is not None and not self.generation:
            raise ContractError("INVALID_GENERATION", "generation token must be nonempty")
        if self.received_steady_ns is not None and self.received_steady_ns < 0:
            raise ContractError("INVALID_RECEIVE_TIME", "steady receive time must be nonnegative")
        if self.source is PoseSource.SIM_GROUND_TRUTH and not self.model_identity:
            raise ContractError("GROUND_TRUTH_MODEL_MISSING", "ground truth must identify its model")
        if self.source in (PoseSource.MEASUREMENT, PoseSource.ESTIMATE) and not self.calibration_id:
            raise ContractError("CALIBRATION_MISSING", "measured or estimated pose must identify sensor calibration")
        if self.covariance_6x6 is not None:
            if len(self.covariance_6x6) != 6 or any(len(row) != 6 for row in self.covariance_6x6):
                raise ContractError("INVALID_COVARIANCE", "covariance must be 6 by 6")
            for row in self.covariance_6x6:
                _finite_tuple(row, 6, "covariance")
            for i in range(6):
                for j in range(6):
                    if abs(self.covariance_6x6[i][j] - self.covariance_6x6[j][i]) > 1e-9:
                        raise ContractError("INVALID_COVARIANCE", "covariance must be symmetric")

    @property
    def transport_epoch_complete(self) -> bool:
        return bool(self.generation) and self.sequence is not None

    def require_fresh(self, now_steady_ns: int, max_age_ns: int) -> None:
        if now_steady_ns < 0 or max_age_ns < 0:
            raise ContractError("INVALID_FRESHNESS_INPUT", "steady time and maximum age must be nonnegative")
        if self.received_steady_ns is None:
            raise ContractError("RECEIPT_TIME_MISSING", "monotonic receipt time is required")
        age = now_steady_ns - self.received_steady_ns
        if age < 0 or age > max_age_ns:
            raise ContractError("STALE_POSE", f"monotonic receive age {age} ns is outside limit")


@dataclass(frozen=True)
class TransformSample:
    """T_parent_child; child coordinates are mapped into parent coordinates."""
    parent_frame: str
    child_frame: str
    translation_m: Vector3
    rotation_xyzw: QuaternionXYWZ
    acquisition_time_ns: int
    authority: str
    is_static: bool = False
    time_domain: TimeDomain | None = None

    def __post_init__(self) -> None:
        if not self.parent_frame or not self.child_frame or self.parent_frame == self.child_frame:
            raise ContractError("INVALID_TRANSFORM_FRAMES", "transform requires distinct named frames")
        if not isinstance(self.is_static, bool):
            raise ContractError("INVALID_TRANSFORM_KIND", "is_static must be boolean")
        _finite_tuple(self.translation_m, 3, "translation")
        q = _finite_tuple(self.rotation_xyzw, 4, "rotation")
        if abs(math.sqrt(sum(v*v for v in q)) - 1.0) > 1e-6:
            raise ContractError("QUATERNION_NOT_NORMALIZED", "transform quaternion must be normalized xyzw")
        if not self.authority:
            raise ContractError("TRANSFORM_AUTHORITY_MISSING", "each TF edge needs one authority")
        if self.acquisition_time_ns < 0:
            raise ContractError("INVALID_STAMP", "transform stamp must be nonnegative")
        if self.is_static and self.time_domain is not None:
            raise ContractError("STATIC_TRANSFORM_HAS_CLOCK", "static transforms are timeless")
        if not self.is_static and self.time_domain is None:
            raise ContractError("TRANSFORM_CLOCK_MISSING", "dynamic transform must name its clock domain")
        if self.time_domain is not None and not isinstance(self.time_domain, TimeDomain):
            raise ContractError("TRANSFORM_CLOCK_INVALID", "dynamic transform clock must use a declared enum")


def _finite_tuple(values: Sequence[float], length: int, label: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ContractError("INVALID_VECTOR", f"{label} must have {length} components")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ContractError("NONFINITE_VECTOR", f"{label} must be finite")
    return result


def _qmul(a: QuaternionXYWZ, b: QuaternionXYWZ) -> QuaternionXYWZ:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def _rotate(q: QuaternionXYWZ, v: Vector3) -> Vector3:
    x, y, z, w = q
    qv = (x, y, z)
    # Vector form of q * (v, 0) * conjugate(q).
    cross1 = (qv[1]*v[2] - qv[2]*v[1],
              qv[2]*v[0] - qv[0]*v[2],
              qv[0]*v[1] - qv[1]*v[0])
    cross2 = (qv[1]*cross1[2] - qv[2]*cross1[1],
              qv[2]*cross1[0] - qv[0]*cross1[2],
              qv[0]*cross1[1] - qv[1]*cross1[0])
    return tuple(v[i] + 2.0*(w*cross1[i] + cross2[i]) for i in range(3))


def transform_pose(sample: PoseSample, transform: TransformSample) -> PoseSample:
    """Apply an explicitly timestamp-matched T_parent_child; never use latest TF."""
    if sample.frame_id != transform.child_frame:
        raise ContractError("FRAME_CHAIN_MISMATCH", "pose frame must equal transform child frame")
    if not transform.is_static:
        if sample.time_domain is not transform.time_domain:
            raise ContractError("TRANSFORM_CLOCK_MISMATCH", "pose and transform clocks are not comparable")
        if sample.acquisition_time_ns != transform.acquisition_time_ns:
            raise ContractError("TRANSFORM_TIME_MISMATCH", "dynamic transform must be evaluated at acquisition time")
    rotated = _rotate(transform.rotation_xyzw, sample.position_m)
    position = tuple(rotated[i] + transform.translation_m[i] for i in range(3))
    orientation = _qmul(transform.rotation_xyzw, sample.orientation_xyzw)
    norm = math.sqrt(sum(v*v for v in orientation))
    orientation = tuple(v/norm for v in orientation)
    covariance = (_transform_covariance(sample.covariance_6x6, transform.rotation_xyzw,
                                       transform.translation_m)
                  if sample.covariance_6x6 is not None else None)
    return PoseSample(
        object_id=sample.object_id,
        frame_id=transform.parent_frame,
        position_m=position,
        orientation_xyzw=orientation,
        acquisition_time_ns=sample.acquisition_time_ns,
        source=sample.source,
        time_domain=sample.time_domain,
        model_identity=sample.model_identity,
        generation=sample.generation,
        sequence=sample.sequence,
        received_steady_ns=sample.received_steady_ns,
        calibration_id=sample.calibration_id,
        covariance_6x6=covariance,
    )


def transform_pose_chain(sample: PoseSample, transforms: Sequence[TransformSample], target_frame: str) -> PoseSample:
    """Apply an explicit ordered T_parent_child chain; no inverse/latest lookup."""
    if not target_frame:
        raise ContractError("TARGET_FRAME_MISSING", "target frame is required")
    result = sample
    for transform in transforms:
        if transform.child_frame != result.frame_id:
            raise ContractError("FRAME_CHAIN_MISMATCH", "transform chain is not contiguous and parent<-child")
        result = transform_pose(result, transform)
    if result.frame_id != target_frame:
        raise ContractError("TARGET_FRAME_UNREACHED", f"chain ended at {result.frame_id}, expected {target_frame}")
    return result


def _rotation_matrix(q: QuaternionXYWZ) -> list[list[float]]:
    x, y, z, w = q
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]


def _transform_covariance(covariance, q: QuaternionXYWZ, t: Vector3):
    """Propagate [position xyz, small-angle xyz] covariance by the SE(3) adjoint."""
    r = _rotation_matrix(q)
    tx, ty, tz = t
    skew = [[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]]
    sr = [[sum(skew[i][k]*r[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    a = [[0.0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            a[i][j] = r[i][j]
            a[i][j+3] = sr[i][j]
            a[i+3][j+3] = r[i][j]
    left = [[sum(a[i][k]*covariance[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
    return tuple(tuple(sum(left[i][k]*a[j][k] for k in range(6)) for j in range(6)) for i in range(6))


class DockStatus(str, Enum):
    SUCCEEDED = "succeeded"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class DockingHandoff:
    """A navigation completion and measured pose snapshot, not a cmd_vel command."""
    goal_id: str
    status: DockStatus
    base_pose: PoseSample
    goal_pose: PoseSample
    linear_velocity_mps: Vector3
    angular_velocity_rps: Vector3
    stationary_dwell_s: float
    generation: str | None
    sequence: int | None


@dataclass(frozen=True)
class DockingLimits:
    max_pose_age_ns: int
    xy_tolerance_m: float
    yaw_tolerance_rad: float
    max_linear_speed_mps: float
    max_angular_speed_rps: float
    required_stationary_dwell_s: float

    def __post_init__(self) -> None:
        values = (self.xy_tolerance_m, self.yaw_tolerance_rad,
                  self.max_linear_speed_mps, self.max_angular_speed_rps,
                  self.required_stationary_dwell_s)
        if self.max_pose_age_ns < 0 or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ContractError("INVALID_DOCKING_LIMITS", "docking limits must be finite and nonnegative")


@dataclass(frozen=True)
class HandoffDecision:
    ready: bool
    reasons: tuple[str, ...]


def _yaw_xyzw(q: QuaternionXYWZ) -> float:
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _wrap_angle(x: float) -> float:
    return (x + math.pi) % (2*math.pi) - math.pi


def evaluate_docking_handoff(
    handoff: DockingHandoff,
    limits: DockingLimits,
    *,
    now_steady_ns: int,
) -> HandoffDecision:
    """Accept only a succeeded, fresh, measured, stationary, frame-aligned dock."""
    reasons: list[str] = []
    if handoff.status is not DockStatus.SUCCEEDED:
        reasons.append("NAVIGATION_NOT_SUCCEEDED")
    if not handoff.goal_id:
        reasons.append("GOAL_ID_MISSING")
    try:
        handoff.base_pose.require_fresh(now_steady_ns, limits.max_pose_age_ns)
    except ContractError:
        reasons.append("BASE_POSE_STALE_OR_UNTIMED")
    if handoff.base_pose.source not in (PoseSource.MEASUREMENT, PoseSource.SIM_GROUND_TRUTH):
        reasons.append("BASE_POSE_NOT_MEASURED")
    if handoff.goal_pose.source is not PoseSource.TASK_TARGET:
        reasons.append("DOCK_GOAL_NOT_AN_EXPLICIT_TARGET")
    if handoff.base_pose.frame_id != handoff.goal_pose.frame_id:
        reasons.append("BASE_GOAL_FRAME_MISMATCH")
    if handoff.base_pose.generation != handoff.generation or handoff.sequence != handoff.base_pose.sequence:
        reasons.append("HANDOFF_EPOCH_MISMATCH")
    if not handoff.generation or handoff.sequence is None:
        reasons.append("HANDOFF_EPOCH_MISSING")
    if handoff.base_pose.source is PoseSource.SIM_GROUND_TRUTH and handoff.base_pose.model_identity is None:
        reasons.append("GROUND_TRUTH_MODEL_MISSING")
    vals = (*handoff.linear_velocity_mps, *handoff.angular_velocity_rps, handoff.stationary_dwell_s)
    if not all(math.isfinite(float(v)) for v in vals):
        reasons.append("NONFINITE_DOCK_STATE")
    else:
        linear_norm = math.sqrt(sum(float(v)**2 for v in handoff.linear_velocity_mps))
        angular_norm = math.sqrt(sum(float(v)**2 for v in handoff.angular_velocity_rps))
        if linear_norm > limits.max_linear_speed_mps or angular_norm > limits.max_angular_speed_rps:
            reasons.append("BASE_NOT_STATIONARY")
        if handoff.stationary_dwell_s < limits.required_stationary_dwell_s:
            reasons.append("STATIONARY_DWELL_INSUFFICIENT")
        dx = handoff.base_pose.position_m[0] - handoff.goal_pose.position_m[0]
        dy = handoff.base_pose.position_m[1] - handoff.goal_pose.position_m[1]
        if math.hypot(dx, dy) > limits.xy_tolerance_m:
            reasons.append("DOCK_POSITION_OUTSIDE_TOLERANCE")
        yaw_error = abs(_wrap_angle(_yaw_xyzw(handoff.base_pose.orientation_xyzw) -
                                    _yaw_xyzw(handoff.goal_pose.orientation_xyzw)))
        if yaw_error > limits.yaw_tolerance_rad:
            reasons.append("DOCK_YAW_OUTSIDE_TOLERANCE")
    return HandoffDecision(not reasons, tuple(reasons))
