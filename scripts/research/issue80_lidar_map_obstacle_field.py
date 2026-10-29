#!/usr/bin/env python3
"""Map-frame static-surface comparison for the Issue #80 LiDAR obstacle layer.

Unlike the rejected raw range-subtraction experiment, this module compares
*map-frame* LiDAR points against a static surface field built from the existing
RAICOM signature database.  Robot/chest motion is already accounted for by the
map-frame point transform, so a nominal chest pose must not be used as a proxy
for a live gait pose.

This remains an offline, ``SIMULATION_ONLY`` research component.  It does not
subscribe to ROS, build a MuJoCo model, call ``mj_step``, or publish commands.
The caller must first verify that its PointCloud2 really is map-frame data and
must provide the calibrated static-signature-to-map transform recorded by the
current localization session.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from issue80_lidar_obstacle_layer import (
    GridCell,
    GridSpec,
    ObstacleLayerError,
    Pose2D,
    RAICOM_CHEST_RAYCASTER,
    RayCasterProfile,
    ray_direction_in_base,
)


@dataclass(frozen=True)
class MapPoint:
    """One finite point in an independently verified map frame."""

    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.z_m)):
            raise ObstacleLayerError("map-frame point must be finite")


def load_map_background_capture(path: Path, *, expected_frame_id: str = "map") -> tuple[MapPoint, ...]:
    """Load one retained, verified map-frame point-cloud background capture.

    This is intentionally a capture-to-capture reference, not a replacement
    for the source static signature database.  It is appropriate when the
    delivered PointCloud2 map transform is stable but the database's nominal
    joint posture does not match the active standing pose.
    """
    if path.is_symlink() or not path.is_file():
        raise ObstacleLayerError("map background capture is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObstacleLayerError("map background capture is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("status") != "PASS_NONEMPTY_POINTCLOUD":
        raise ObstacleLayerError("map background capture did not pass point-cloud validation")
    message = payload.get("last_message")
    if not isinstance(message, dict) or message.get("frame_id") != expected_frame_id:
        raise ObstacleLayerError("map background capture has an unexpected frame")
    if message.get("all_finite_xyz_retention_blocked") is not False:
        raise ObstacleLayerError("map background capture lacks a complete retained finite array")
    raw_points = message.get("all_finite_xyz")
    if not isinstance(raw_points, list):
        raise ObstacleLayerError("map background capture has no finite points")
    points: list[MapPoint] = []
    for item in raw_points:
        if not isinstance(item, list) or len(item) != 3:
            raise ObstacleLayerError("map background capture has an invalid point")
        try:
            points.append(MapPoint(*(float(value) for value in item)))
        except (TypeError, ValueError) as exc:
            raise ObstacleLayerError("map background capture has a nonnumeric point") from exc
    if not points:
        raise ObstacleLayerError("map background capture has no finite points")
    return tuple(points)


def map_background_distances_m(
    observed_points: Sequence[MapPoint], *, background_points: Sequence[MapPoint],
) -> tuple[float, ...]:
    """Return exact 3D nearest-background distances for a map-frame scan."""
    if not background_points:
        raise ObstacleLayerError("map background reference is empty")
    return tuple(
        min(
            math.sqrt(
                (point.x_m - candidate.x_m) ** 2
                + (point.y_m - candidate.y_m) ** 2
                + (point.z_m - candidate.z_m) ** 2
            )
            for candidate in background_points
        )
        for point in observed_points
    )


def novel_map_points_against_background(
    observed_points: Sequence[MapPoint], *, background_points: Sequence[MapPoint], maximum_match_distance_m: float,
) -> tuple[MapPoint, ...]:
    """Return map-frame points not explained by a measured background scan."""
    if not math.isfinite(maximum_match_distance_m) or maximum_match_distance_m <= 0.0:
        raise ObstacleLayerError("map background match distance must be finite and positive")
    distances = map_background_distances_m(observed_points, background_points=background_points)
    return tuple(point for point, distance in zip(observed_points, distances) if distance > maximum_match_distance_m)


def classify_map_frame_capture(
    *,
    message_count: int,
    nonempty_message_count: int,
    last_frame_id: str | None,
    expected_frame_id: str | None,
) -> str:
    """Classify capture completeness without accepting a wrong coordinate frame."""
    if message_count <= 0 or nonempty_message_count <= 0 or last_frame_id is None:
        return "BLOCKED_EMPTY_OR_MISSING_POINTCLOUD"
    if expected_frame_id is not None and last_frame_id != expected_frame_id:
        return "BLOCKED_POINTCLOUD_FRAME_MISMATCH"
    return "PASS_NONEMPTY_POINTCLOUD"


def decode_map_xyz(
    *,
    frame_id: str,
    data: bytes,
    width: int,
    height: int,
    point_step: int,
    field_offsets: dict[str, int],
    expected_frame_id: str = "map",
) -> tuple[MapPoint, ...]:
    """Decode finite XYZ points only after checking the declared map frame.

    This accepts primitive PointCloud2 layout data rather than ROS message
    types, keeping perception math unit-testable.  An unexpected frame label
    is a fail-closed condition, not a harmless visualization difference.
    """
    if frame_id != expected_frame_id:
        raise ObstacleLayerError(f"PointCloud2 frame_id must be {expected_frame_id!r}, got {frame_id!r}")
    if width < 0 or height < 0 or point_step < 12:
        raise ObstacleLayerError("PointCloud2 dimensions or point_step are invalid")
    if not all(axis in field_offsets for axis in ("x", "y", "z")):
        raise ObstacleLayerError("PointCloud2 lacks x/y/z fields")
    offsets = tuple(field_offsets[axis] for axis in ("x", "y", "z"))
    if any(not isinstance(offset, int) or offset < 0 or offset + 4 > point_step for offset in offsets):
        raise ObstacleLayerError("PointCloud2 x/y/z field offsets are invalid")
    result: list[MapPoint] = []
    for index in range(width * height):
        base = index * point_step
        if base + point_step > len(data):
            raise ObstacleLayerError("PointCloud2 data is shorter than its declared dimensions")
        values = tuple(struct.unpack_from("<f", data, base + offset)[0] for offset in offsets)
        if all(math.isfinite(value) for value in values):
            result.append(MapPoint(*values))
    return tuple(result)


def apply_planar_transform(transform: Pose2D, point_x_m: float, point_y_m: float) -> tuple[float, float]:
    """Apply a declared map <- static-signature SE(2) transform."""
    if not math.isfinite(point_x_m) or not math.isfinite(point_y_m):
        raise ObstacleLayerError("static point must be finite")
    cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    return (
        transform.x_m + cosine * point_x_m - sine * point_y_m,
        transform.y_m + sine * point_x_m + cosine * point_y_m,
    )


def static_surface_cells_from_database(
    candidates: np.ndarray,
    ranges: np.ndarray,
    *,
    static_to_map: Pose2D,
    grid: GridSpec,
    profile: RayCasterProfile = RAICOM_CHEST_RAYCASTER,
) -> frozenset[GridCell]:
    """Rasterize static signature hits into a map-frame surface field.

    A database candidate describes the base pose used for that signature; each
    finite range is projected using the same RayCaster convention that created
    the database.  The explicit ``static_to_map`` transform is required so a
    localization calibration offset cannot be silently discarded.
    """
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ObstacleLayerError("static candidate array must have shape (N, 3)")
    if ranges.ndim != 2 or ranges.shape != (len(candidates), profile.ray_count):
        raise ObstacleLayerError("static ranges must align with candidates and every RayCaster slot")
    cells: set[GridCell] = set()
    for candidate, row in zip(candidates, ranges):
        base = Pose2D(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        cosine, sine = math.cos(base.yaw_rad), math.sin(base.yaw_rad)
        source_origin_x = base.x_m + cosine * profile.chest_x_offset_m
        source_origin_y = base.y_m + sine * profile.chest_x_offset_m
        for index, raw_distance in enumerate(row):
            distance = float(raw_distance)
            if not math.isfinite(distance) or not (profile.minimum_range_m <= distance <= profile.maximum_range_m):
                continue
            forward, lateral = ray_direction_in_base(index, profile)
            source_x = source_origin_x + cosine * (distance * forward) - sine * (distance * lateral)
            source_y = source_origin_y + sine * (distance * forward) + cosine * (distance * lateral)
            map_x, map_y = apply_planar_transform(static_to_map, source_x, source_y)
            cell = grid.world_to_cell(map_x, map_y)
            if grid.contains(cell):
                cells.add(cell)
    return frozenset(cells)


def surface_neighborhood(
    static_surface: Iterable[GridCell],
    *,
    grid: GridSpec,
    maximum_distance_m: float,
) -> frozenset[GridCell]:
    """Return every cell within a caller-defined static-surface tolerance."""
    if not math.isfinite(maximum_distance_m) or maximum_distance_m < 0.0:
        raise ObstacleLayerError("surface-match distance must be finite and non-negative")
    radius = math.ceil(maximum_distance_m / grid.resolution_m)
    output: set[GridCell] = set()
    for source in static_surface:
        if not grid.contains(source):
            continue
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell = GridCell(source.x + dx, source.y + dy)
                if grid.contains(cell) and math.hypot(dx, dy) * grid.resolution_m <= maximum_distance_m + 1e-12:
                    output.add(cell)
    return frozenset(output)


def novel_map_points(
    observed_points: Sequence[MapPoint],
    *,
    static_surface: Iterable[GridCell],
    grid: GridSpec,
    static_match_distance_m: float,
) -> tuple[MapPoint, ...]:
    """Retain map points not explained by the known static surface field."""
    known_neighborhood = surface_neighborhood(
        static_surface, grid=grid, maximum_distance_m=static_match_distance_m,
    )
    result: list[MapPoint] = []
    for point in observed_points:
        cell = grid.world_to_cell(point.x_m, point.y_m)
        if grid.contains(cell) and cell not in known_neighborhood:
            result.append(point)
    return tuple(result)


def static_surface_distances_m(
    observed_points: Sequence[MapPoint],
    *,
    static_surface: Iterable[GridCell],
    grid: GridSpec,
) -> tuple[float, ...]:
    """Return nearest 2D static-surface-cell distance for every map point.

    This deliberately returns ``inf`` outside the bounded field rather than
    projecting or clamping an out-of-map point into a false static match.
    """
    surface = tuple(cell for cell in static_surface if grid.contains(cell))
    if not surface:
        raise ObstacleLayerError("static surface field is empty")
    result: list[float] = []
    for point in observed_points:
        cell = grid.world_to_cell(point.x_m, point.y_m)
        if not grid.contains(cell):
            result.append(math.inf)
            continue
        result.append(min(math.hypot(cell.x - candidate.x, cell.y - candidate.y) * grid.resolution_m for candidate in surface))
    return tuple(result)
