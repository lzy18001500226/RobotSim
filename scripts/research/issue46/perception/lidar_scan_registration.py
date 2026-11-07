"""Reused Issue80 scan registration and explicit body/sensor SE(2) composition.

Source: RobotSim 70a84cd094a8d686864a26a260d2d17d1d65ea8e,
scripts/research/issue80_lidar_scan_math.py (rigid_fit, scan_delta, Pose2).
The returned current-scan -> previous-scan transform is the positive sensor
pose increment, NOT its inverse. The old localization caller inverted it.
No robot truth, MC odometry, dynamics or motion publisher in this module.
"""
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



def matrix(pose: Pose2) -> np.ndarray:
    c,s=math.cos(pose.yaw),math.sin(pose.yaw)
    return np.array([[c,-s,pose.x],[s,c,pose.y],[0.,0.,1.]])

def pose_from_matrix(transform: np.ndarray) -> Pose2:
    if transform.shape!=(3,3) or not np.isfinite(transform).all():
        raise ValueError("invalid SE(2) transform")
    return Pose2(float(transform[0,2]),float(transform[1,2]),
                 math.atan2(transform[1,0],transform[0,0]))

def compose_body_from_scan_delta(previous_body: Pose2,
        previous_sensor_in_body: Pose2, current_sensor_in_body: Pose2,
        current_scan_to_previous: Pose2) -> Pose2:
    """T_map_Bcur = T_map_Bprev T_Bprev_Lprev T_Lprev_Lcur inv(T_Bcur_Lcur).

    Extrinsics can change with measured waist joints. Inputs are calibrated
    local transforms and scan-derived motion, never live world root truth.
    """
    return pose_from_matrix(matrix(previous_body)@matrix(previous_sensor_in_body)
        @matrix(current_scan_to_previous)@np.linalg.inv(matrix(current_sensor_in_body)))
