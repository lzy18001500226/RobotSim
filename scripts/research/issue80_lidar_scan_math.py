"""Dependency-free planar scan-registration math for Issue #80."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    return rotation, target_center - source_center @ rotation.T


def scan_delta(previous: np.ndarray, current: np.ndarray, *, iterations: int = 8) -> tuple[float, float, float, float]:
    if len(previous) < 16 or len(current) < 16:
        raise ValueError("insufficient valid LiDAR points")
    previous = previous[::max(1, len(previous) // 180)]
    current = current[::max(1, len(current) // 180)]
    rotation = np.eye(2)
    translation = np.zeros(2)
    mean_error = math.inf
    for _ in range(iterations):
        transformed = current @ rotation.T + translation
        square_distance = ((transformed[:, None, :] - previous[None, :, :]) ** 2).sum(axis=2)
        nearest = previous[np.argmin(square_distance, axis=1)]
        increment_r, increment_t = rigid_fit(transformed, nearest)
        rotation = increment_r @ rotation
        translation = translation @ increment_r.T + increment_t
        mean_error = float(np.sqrt(np.min(square_distance, axis=1).mean()))
    return float(translation[0]), float(translation[1]), math.atan2(rotation[1, 0], rotation[0, 0]), mean_error


@dataclass
class Pose2:
    x: float
    y: float
    yaw: float

    def apply_delta(self, forward: float, lateral: float, yaw: float) -> "Pose2":
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2(self.x + c * forward - s * lateral,
                     self.y + s * forward + c * lateral, wrap(self.yaw + yaw))
