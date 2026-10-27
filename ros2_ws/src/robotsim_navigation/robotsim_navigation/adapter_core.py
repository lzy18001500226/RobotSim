"""Offline adapters for nav_msgs/Odometry and geometry_msgs/PoseStamped fields."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .point_goal_controller import Goal2D, Pose2D, goal_from_pose_fields


DEFAULT_ODOM_FRAME = "odom"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_PROPOSAL_TOPIC = "/robotsim/navigation/cmd_vel_proposal"
FORBIDDEN_MC_TOPICS = frozenset({"/cmd_vel", "/aima/mc/locomotion/velocity"})


class AdapterInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OdometrySample:
    pose: Pose2D
    stamp_ns: int


def validate_proposal_topic(topic: str) -> None:
    if not topic or not topic.startswith("/"):
        raise AdapterInputError("INVALID_TOPIC", "proposal topic must be absolute")
    if topic in FORBIDDEN_MC_TOPICS:
        raise AdapterInputError(
            "MC_TOPIC_FORBIDDEN",
            "adapter must remain disconnected from live MC topics",
        )


def stamp_to_nanoseconds(stamp: Any) -> int:
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdapterInputError("INVALID_STAMP", "message has no valid builtin Time") from exc
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise AdapterInputError("INVALID_STAMP", "stamp sec/nanosec are out of range")
    return sec * 1_000_000_000 + nanosec


def _rpy_yaw_from_quaternion(q: Any) -> tuple[float, float, float]:
    try:
        values = tuple(float(getattr(q, name)) for name in ("x", "y", "z", "w"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdapterInputError("INVALID_QUATERNION", "quaternion fields are missing") from exc
    if not all(math.isfinite(value) for value in values):
        raise AdapterInputError("INVALID_QUATERNION", "quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise AdapterInputError("INVALID_QUATERNION", "quaternion norm is zero")
    x, y, z, w = (value / norm for value in values)
    roll = math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w*y - z*x))))
    yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return roll, pitch, yaw


def odometry_to_pose(
    message: Any,
    *,
    received_steady_s: float,
    expected_odom_frame: str = DEFAULT_ODOM_FRAME,
    expected_base_frame: str = DEFAULT_BASE_FRAME,
    last_stamp_ns: int | None = None,
    min_base_z_m: float = 0.52,
    max_tilt_rad: float = math.radians(15.0),
) -> OdometrySample:
    """Validate standard Odometry fields, monotonic stamps, frames, and M1 safety gate."""
    try:
        header = message.header
        frame_id = str(header.frame_id)
        child_frame_id = str(message.child_frame_id)
        stamp_ns = stamp_to_nanoseconds(header.stamp)
        pose = message.pose.pose
        position = pose.position
        orientation = pose.orientation
        x, y, z = float(position.x), float(position.y), float(position.z)
    except AdapterInputError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdapterInputError("INVALID_ODOMETRY", "Odometry fields are incomplete") from exc

    if frame_id != expected_odom_frame or child_frame_id != expected_base_frame:
        raise AdapterInputError(
            "FRAME_MISMATCH",
            f"expected {expected_odom_frame}->{expected_base_frame}, got "
            f"{frame_id}->{child_frame_id}",
        )
    if stamp_ns == 0:
        raise AdapterInputError("INVALID_STAMP", "Odometry stamp must be nonzero")
    if last_stamp_ns is not None and stamp_ns <= last_stamp_ns:
        raise AdapterInputError(
            "STAMP_REGRESSION",
            "Odometry stamp repeated or moved backwards; reset the navigation session",
        )
    if not math.isfinite(received_steady_s):
        raise AdapterInputError("INVALID_RECEIVE_TIME", "receive time must be finite")
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise AdapterInputError("INVALID_POSITION", "Odometry position must be finite")
    if not math.isfinite(min_base_z_m) or not math.isfinite(max_tilt_rad) or max_tilt_rad <= 0:
        raise AdapterInputError("INVALID_SAFETY_CONFIG", "safety limits must be finite")

    roll, pitch, yaw = _rpy_yaw_from_quaternion(orientation)
    if z < min_base_z_m:
        raise AdapterInputError("UNSAFE_HEIGHT", f"base z {z:.4f} below {min_base_z_m:.4f} m")
    if abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad:
        raise AdapterInputError(
            "UNSAFE_TILT",
            f"roll/pitch {roll:.4f}/{pitch:.4f} exceeds {max_tilt_rad:.4f} rad",
        )

    return OdometrySample(
        pose=Pose2D(x, y, yaw, frame_id, received_steady_s),
        stamp_ns=stamp_ns,
    )


def pose_stamped_to_goal(
    message: Any,
    *,
    expected_frame: str = DEFAULT_ODOM_FRAME,
    max_planar_tilt_rad: float = 0.10,
) -> Goal2D:
    """Accept only goals already expressed in odom; no transform is fabricated."""
    try:
        header = message.header
        frame_id = str(header.frame_id)
        stamp_to_nanoseconds(header.stamp)
        if not frame_id or frame_id != expected_frame:
            raise AdapterInputError(
                "FRAME_MISMATCH",
                f"goal frame {frame_id!r} != required frame {expected_frame!r}",
            )
        pose = message.pose
        position = pose.position
        orientation = pose.orientation
        return goal_from_pose_fields(
            frame_id=frame_id,
            expected_frame=expected_frame,
            position_xyz=(position.x, position.y, position.z),
            orientation_xyzw=(
                orientation.x, orientation.y, orientation.z, orientation.w
            ),
            max_planar_tilt_rad=max_planar_tilt_rad,
        )
    except AdapterInputError:
        raise
    except ValueError as exc:
        # Keep the validation reason visible while exposing a stable adapter code.
        raise AdapterInputError(
            "INVALID_GOAL", f"PoseStamped fields are invalid: {exc}"
        ) from exc
    except (AttributeError, TypeError) as exc:
        raise AdapterInputError(
            "INVALID_GOAL", "PoseStamped fields are incomplete or nonnumeric"
        ) from exc
