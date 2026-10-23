"""Transport-neutral tool-frame approach and trajectory contracts."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .contracts import PoseSample, TransformSample, Vector3, QuaternionXYWZ, ContractError


@dataclass(frozen=True)
class ToolPose:
    frame_id: str
    position_m: Vector3
    orientation_xyzw: QuaternionXYWZ

    def __post_init__(self):
        if not self.frame_id or len(self.position_m) != 3 or len(self.orientation_xyzw) != 4:
            raise ContractError("INVALID_TOOL_POSE", "tool pose requires frame and xyz/xyzw")
        vals=(*self.position_m,*self.orientation_xyzw)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ContractError("NONFINITE_TOOL_POSE", "tool pose must be finite")
        norm=math.sqrt(sum(float(q)**2 for q in self.orientation_xyzw))
        if abs(norm-1.0)>1e-6:
            raise ContractError("QUATERNION_NOT_NORMALIZED", "tool orientation must be normalized")


@dataclass(frozen=True)
class ApproachWaypoint:
    phase: str
    progress: float
    tool_pose: ToolPose


@dataclass(frozen=True)
class IKWaypoint:
    phase: str
    progress: float
    joint_positions_rad: Mapping[str, float]
    position_error_m: float
    orientation_error_rad: float
    minimum_source_limit_margin_rad: float


@dataclass(frozen=True)
class ApproachPlan:
    object_pose: PoseSample
    tool_frame: str
    approach_direction_world: Vector3
    pregrasp_standoff_m: float
    waypoints: tuple[ApproachWaypoint, ...]

    def __post_init__(self):
        if len(self.waypoints)<2:
            raise ContractError("APPROACH_WAYPOINTS_MISSING", "plan needs PREGRASP and APPROACH")
        if self.waypoints[0].phase!="PREGRASP" or self.waypoints[-1].phase!="APPROACH":
            raise ContractError("APPROACH_PHASE_ORDER", "plan must start PREGRASP and end APPROACH")
        progress = [x.progress for x in self.waypoints]
        if progress[0] != 0.0 or progress[-1] != 1.0 or any(
            not math.isfinite(x) or x < 0.0 or x > 1.0 for x in progress
        ) or any(b <= a for a, b in zip(progress, progress[1:])):
            raise ContractError("APPROACH_PROGRESS_INVALID", "waypoint progress must increase strictly from zero to one")
        if any(x.tool_pose.frame_id != self.object_pose.frame_id for x in self.waypoints):
            raise ContractError("APPROACH_FRAME_MISMATCH", "all tool waypoints and object pose must use the same parent frame")


def make_linear_approach(
    object_pose: PoseSample,
    *,
    tool_frame: str,
    target_point_world_m: Vector3,
    tool_orientation_xyzw: QuaternionXYWZ,
    approach_direction_world: Vector3,
    pregrasp_standoff_m: float,
    interpolation_samples: int = 1,
) -> ApproachPlan:
    """Build a single straight Cartesian insertion; caller must collision-check it.

    approach_direction_world points from PREGRASP toward the target. Vertical and
    lateral approaches are both supported; the function does not select one.
    """
    vals=(*target_point_world_m,*approach_direction_world,pregrasp_standoff_m)
    if not all(math.isfinite(float(v)) for v in vals) or pregrasp_standoff_m<=0:
        raise ContractError("INVALID_APPROACH", "target, direction, and standoff must be finite")
    if interpolation_samples<1:
        raise ContractError("INVALID_APPROACH", "interpolation_samples must be positive")
    norm=math.sqrt(sum(float(v)**2 for v in approach_direction_world))
    if abs(norm-1.0)>1e-6:
        raise ContractError("APPROACH_DIRECTION_NOT_NORMALIZED", "approach direction must be unit length")
    if len(target_point_world_m)!=3:
        raise ContractError("INVALID_APPROACH", "target point must be xyz")
    direction=tuple(float(v) for v in approach_direction_world)
    start=tuple(float(target_point_world_m[i])-direction[i]*pregrasp_standoff_m for i in range(3))
    points=[]
    for i in range(interpolation_samples+1):
        u=i/interpolation_samples
        point=tuple(start[j]*(1-u)+float(target_point_world_m[j])*u for j in range(3))
        phase="PREGRASP" if i==0 else ("APPROACH" if i==interpolation_samples else "APPROACH")
        points.append(ApproachWaypoint(phase,u,ToolPose(object_pose.frame_id,point,tool_orientation_xyzw)))
    return ApproachPlan(object_pose,tool_frame,direction,pregrasp_standoff_m,tuple(points))


def validate_ik_waypoints(
    waypoints: Sequence[IKWaypoint],
    *,
    expected_joint_names: Sequence[str],
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    minimum_limit_margin_rad: float,
    source_joint_limits_rad: Mapping[str, tuple[float, float]],
    maximum_joint_step_rad: float | None = None,
) -> tuple[str,...]:
    """Check a source-limit-aware IK witness; does not claim dynamic tracking."""
    blockers=[]
    if not waypoints:
        return ("IK_WAYPOINTS_MISSING",)
    tolerances = (position_tolerance_m, orientation_tolerance_rad, minimum_limit_margin_rad)
    if not all(math.isfinite(x) and x >= 0 for x in tolerances):
        return ("IK_TOLERANCE_INVALID",)
    if set(expected_joint_names) != set(source_joint_limits_rad):
        return ("SOURCE_LIMIT_MAP_MISMATCH",)
    if maximum_joint_step_rad is not None and (not math.isfinite(maximum_joint_step_rad) or maximum_joint_step_rad <= 0):
        return ("MAXIMUM_JOINT_STEP_INVALID",)
    previous = None
    last_progress = -math.inf
    for i,w in enumerate(waypoints):
        if not math.isfinite(w.progress) or w.progress <= last_progress:
            blockers.append(f"WAYPOINT_PROGRESS:{i}")
        last_progress = w.progress
        if (i == 0 and w.phase != "PREGRASP") or (i == len(waypoints)-1 and w.phase != "APPROACH"):
            blockers.append(f"WAYPOINT_PHASE:{i}")
        if tuple(w.joint_positions_rad.keys())!=tuple(expected_joint_names):
            blockers.append(f"JOINT_NAME_ORDER:{i}")
        if not all(math.isfinite(float(v)) for v in w.joint_positions_rad.values()):
            blockers.append(f"NONFINITE_JOINT:{i}")
        if set(w.joint_positions_rad) != set(source_joint_limits_rad):
            blockers.append(f"SOURCE_JOINT_MAP:{i}")
        else:
            computed_margins=[]
            for name, q in w.joint_positions_rad.items():
                lo, hi = source_joint_limits_rad[name]
                if not all(math.isfinite(float(v)) for v in (lo, hi)) or lo >= hi or q < lo or q > hi:
                    blockers.append(f"SOURCE_POSITION_LIMIT:{i}:{name}")
                    continue
                computed_margins.append(min(q-lo, hi-q))
            if computed_margins and abs(min(computed_margins)-w.minimum_source_limit_margin_rad)>1e-5:
                blockers.append(f"SOURCE_MARGIN_WITNESS_MISMATCH:{i}")
        if not math.isfinite(w.position_error_m) or w.position_error_m < 0 or w.position_error_m>position_tolerance_m:
            blockers.append(f"POSITION_IK_RESIDUAL:{i}")
        if not math.isfinite(w.orientation_error_rad) or w.orientation_error_rad < 0 or w.orientation_error_rad>orientation_tolerance_rad:
            blockers.append(f"ORIENTATION_IK_RESIDUAL:{i}")
        if (not math.isfinite(w.minimum_source_limit_margin_rad)
                or w.minimum_source_limit_margin_rad<minimum_limit_margin_rad):
            blockers.append(f"SOURCE_LIMIT_MARGIN:{i}")
        q = tuple(w.joint_positions_rad.get(name, math.nan) for name in expected_joint_names)
        if previous is not None and maximum_joint_step_rad is not None and all(math.isfinite(x) for x in (*q, *previous)):
            if max(abs(a-b) for a,b in zip(q,previous)) > maximum_joint_step_rad:
                blockers.append(f"JOINT_STEP_TOO_LARGE:{i}")
        previous = q
    return tuple(blockers)
