"""Research-only X2 manipulation integration contracts; not runtime code."""

from .approach import ApproachPlan, IKWaypoint, ToolPose, make_linear_approach, validate_ik_waypoints
from .contracts import (
    ContractError, DockingHandoff, DockingLimits, DockStatus, PoseSample, PoseSource,
    TimeDomain, TransformSample, evaluate_docking_handoff, transform_pose_chain,
)
from .feasibility import (
    CollisionInventory, ContactBand, FeasibilityThresholds, GeometryPair, GripperContract,
    GripperFeasibilityRequest, ObjectGeometry, collision_inventory_digest, evaluate_gripper_feasibility,
)

__all__ = [
    "ApproachPlan", "CollisionInventory", "ContactBand", "ContractError", "DockingHandoff",
    "DockingLimits", "DockStatus", "FeasibilityThresholds", "GeometryPair", "GripperContract",
    "GripperFeasibilityRequest", "IKWaypoint", "ObjectGeometry", "PoseSample", "PoseSource",
    "TimeDomain", "ToolPose", "TransformSample", "evaluate_docking_handoff",
    "collision_inventory_digest", "evaluate_gripper_feasibility", "make_linear_approach", "transform_pose_chain",
    "validate_ik_waypoints",
]
