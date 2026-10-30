"""Pure RGB-D and grasp-candidate contract for the X2 OmniPicker pipeline.

No ROS, model weight, camera driver, or robot command is imported here.  This
module owns the frame/unit checks shared by FoundationPose, AnyGrasp and the
open GraspNet fallback before any candidate reaches whole-body IK.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class PerceptionContractError(ValueError):
    """An RGB-D frame or grasp candidate is not safe to consume."""


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or not all(
            math.isfinite(value) and value > 0.0 for value in (self.fx, self.fy)
        ) or not all(math.isfinite(value) for value in (self.cx, self.cy)):
            raise PerceptionContractError("camera intrinsics are invalid")


@dataclass(frozen=True)
class ParallelJawCandidate:
    """GraspNet/AnyGrasp convention: X approach, Y open/close, Z finger height."""

    score: float
    width_m: float
    depth_m: float
    rotation_camera: np.ndarray
    translation_camera_m: np.ndarray

    def validate(self) -> None:
        if not math.isfinite(self.score) or not math.isfinite(self.width_m) or not math.isfinite(self.depth_m):
            raise PerceptionContractError("candidate scalar is non-finite")
        if self.width_m <= 0.0 or self.depth_m < 0.0:
            raise PerceptionContractError("candidate width/depth is invalid")
        if self.rotation_camera.shape != (3, 3) or self.translation_camera_m.shape != (3,):
            raise PerceptionContractError("candidate transform has an invalid shape")
        if not np.isfinite(self.rotation_camera).all() or not np.isfinite(self.translation_camera_m).all():
            raise PerceptionContractError("candidate transform is non-finite")
        if not np.allclose(self.rotation_camera.T @ self.rotation_camera, np.eye(3), atol=1e-5):
            raise PerceptionContractError("candidate rotation is not orthonormal")
        if not np.isclose(np.linalg.det(self.rotation_camera), 1.0, atol=1e-5):
            raise PerceptionContractError("candidate rotation is not right-handed")

    @property
    def approach_camera(self) -> np.ndarray:
        return self.rotation_camera[:, 0]

    @property
    def opening_camera(self) -> np.ndarray:
        return self.rotation_camera[:, 1]

    @property
    def tip_camera_m(self) -> np.ndarray:
        return self.translation_camera_m + self.depth_m * self.approach_camera


def point_cloud_from_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: np.ndarray | None = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 4.0,
) -> np.ndarray:
    """Deproject a metric depth image into the camera frame in metres."""
    intrinsics.validate()
    if depth_m.shape != (intrinsics.height, intrinsics.width):
        raise PerceptionContractError("depth shape does not match CameraInfo")
    if mask is not None and mask.shape != depth_m.shape:
        raise PerceptionContractError("mask shape does not match depth")
    if not (0.0 < min_depth_m < max_depth_m):
        raise PerceptionContractError("depth range is invalid")
    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    if mask is not None:
        valid &= mask.astype(bool)
    z = depth_m[valid].astype(np.float64, copy=False)
    x = (columns[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (rows[valid] - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z)).astype(np.float32)


def transform_candidate_to_base(
    candidate: ParallelJawCandidate,
    t_base_camera: np.ndarray,
) -> ParallelJawCandidate:
    """Transform an AnyGrasp candidate from camera to base_link coordinates."""
    candidate.validate()
    if t_base_camera.shape != (4, 4) or not np.isfinite(t_base_camera).all():
        raise PerceptionContractError("T_base_camera is invalid")
    rotation = t_base_camera[:3, :3]
    translation = t_base_camera[:3, 3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise PerceptionContractError("T_base_camera rotation is invalid")
    return ParallelJawCandidate(
        score=candidate.score,
        width_m=candidate.width_m,
        depth_m=candidate.depth_m,
        rotation_camera=rotation @ candidate.rotation_camera,
        translation_camera_m=rotation @ candidate.translation_camera_m + translation,
    )


def filter_omnipicker_side_grasps(
    candidates_base: list[ParallelJawCandidate],
    *,
    preferred_approach_base: np.ndarray,
    max_opening_m: float,
    approach_tolerance_rad: float,
) -> list[ParallelJawCandidate]:
    """Filter generic parallel-jaw proposals into feasible OmniPicker side grasps.

    The final collision and whole-body IK tests remain mandatory; this filter
    only rejects clearly incompatible width or approach-axis candidates.
    """
    preferred = np.asarray(preferred_approach_base, dtype=float)
    norm = float(np.linalg.norm(preferred))
    if preferred.shape != (3,) or norm <= 1e-9 or not np.isfinite(preferred).all():
        raise PerceptionContractError("preferred approach axis is invalid")
    preferred /= norm
    if not (0.0 < max_opening_m and 0.0 <= approach_tolerance_rad <= math.pi):
        raise PerceptionContractError("OmniPicker filter limits are invalid")
    accepted = []
    for candidate in candidates_base:
        candidate.validate()
        if candidate.width_m > max_opening_m:
            continue
        cosine = float(np.clip(np.dot(candidate.approach_camera, preferred), -1.0, 1.0))
        if math.acos(cosine) <= approach_tolerance_rad:
            accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.score, reverse=True)
