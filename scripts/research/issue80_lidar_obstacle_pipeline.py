#!/usr/bin/env python3
"""Offline bridge from verified map-frame LiDAR points to a safe detour intent.

The bridge deliberately has no ROS node, MuJoCo model, MC client, velocity
publisher, or RouteMotionBridge import.  It validates an incoming PointCloud2
layout, removes points explained by a known static surface field, requires
strictly advancing scan sequences and temporal confirmation, then returns a
static-only A* detour intent.  A separate executor must still run the existing
full-body static preflight and all current source/mode/physical-stop gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

from issue80_lidar_map_obstacle_field import (
    GridSpec,
    decode_map_xyz,
    novel_map_points_against_background,
    novel_map_points,
    MapPoint,
)
from issue80_lidar_obstacle_layer import (
    GridCell,
    NovelObstacleTracker,
    ObstacleLayerError,
    connected_cell_components,
    direct_path_intersects_blocked,
    inflate_cells,
    minimum_span_envelope,
    path_world_points,
    plan_a_star,
    simplify_grid_path,
)


class PipelineStatus(str, Enum):
    CLEAR_DIRECT_PATH = "CLEAR_DIRECT_PATH_STATIC_ONLY"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_OBSTACLE_CONFIRMATION"
    DETOUR_READY = "DETOUR_READY_STATIC_ONLY"
    OBSTACLE_CLEAR_OF_DIRECT_PATH = "OBSTACLE_CLEAR_OF_DIRECT_PATH_STATIC_ONLY"
    BLOCKED_NO_PATH = "BLOCKED_NO_DETOUR_PATH"
    BLOCKED_INVALID_OBSERVATION = "BLOCKED_INVALID_POINTCLOUD_OBSERVATION"


@dataclass(frozen=True)
class ObstaclePipelineConfig:
    grid: GridSpec
    static_surface: frozenset[GridCell]
    static_route_blocked: frozenset[GridCell]
    start: GridCell
    goal: GridCell
    static_match_distance_m: float
    obstacle_inflation_m: float
    confirmation_frames: int
    minimum_confirmed_cluster_cells: int
    map_background_points: tuple[MapPoint, ...] = ()
    minimum_obstacle_span_x_m: float = 0.0
    minimum_obstacle_span_y_m: float = 0.0
    expected_frame_id: str = "map"
    background_registration_enabled: bool = True


@dataclass(frozen=True)
class ObstaclePipelineDecision:
    status: PipelineStatus
    sequence: int
    observed_point_count: int
    novel_point_count: int
    confirmed_cells: tuple[GridCell, ...]
    obstacle_envelope_cells: tuple[GridCell, ...]
    inflated_dynamic_cells: tuple[GridCell, ...]
    detour_cells: tuple[GridCell, ...]
    detour_map_waypoints: tuple[tuple[float, float], ...]
    reason: str
    background_registration: dict[str, float | int | str] | None = None


def _nearest_xy(source: tuple[MapPoint, ...], reference: tuple[MapPoint, ...]) -> tuple[list[float], list[tuple[MapPoint, MapPoint]]]:
    distances: list[float] = []
    pairs: list[tuple[MapPoint, MapPoint]] = []
    for point in source:
        candidate = min(
            reference,
            key=lambda item: (point.x_m - item.x_m) ** 2 + (point.y_m - item.y_m) ** 2,
        )
        distances.append(math.hypot(point.x_m - candidate.x_m, point.y_m - candidate.y_m))
        pairs.append((point, candidate))
    return distances, pairs


def register_background_xy(
    observed: tuple[MapPoint, ...], *, background: tuple[MapPoint, ...],
) -> tuple[tuple[MapPoint, ...], dict[str, float | int | str]]:
    """Robustly align a labelled map cloud to its retained static background.

    The simulation RayCaster's ``world_xyz`` field has been observed to carry
    a small Reset-to-Reset planar rigid offset.  This is scan matching against
    the known static map, not a MuJoCo-pose lookup.  Unknown returns are
    trimmed before each least-squares transform fit so they cannot define the
    alignment.
    """
    if len(observed) < 80 or len(background) < 80:
        raise ObstacleLayerError("background registration needs at least 80 finite points")
    transformed = observed
    yaw = tx = ty = 0.0
    matched = 0
    for _ in range(3):
        distances, pairs = _nearest_xy(transformed, background)
        selected_count = max(80, int(0.65 * len(distances)))
        ranked = sorted(zip(distances, pairs), key=lambda item: item[0])[:selected_count]
        selected = [pair for distance, pair in ranked if distance <= 0.20]
        if len(selected) < 80:
            raise ObstacleLayerError("background registration lacks enough static correspondences")
        source_x = sum(item[0].x_m for item in selected) / len(selected)
        source_y = sum(item[0].y_m for item in selected) / len(selected)
        target_x = sum(item[1].x_m for item in selected) / len(selected)
        target_y = sum(item[1].y_m for item in selected) / len(selected)
        numerator = denominator = 0.0
        for source, target in selected:
            sx, sy = source.x_m - source_x, source.y_m - source_y
            dx, dy = target.x_m - target_x, target.y_m - target_y
            numerator += sx * dy - sy * dx
            denominator += sx * dx + sy * dy
        delta_yaw = math.atan2(numerator, denominator)
        cosine, sine = math.cos(delta_yaw), math.sin(delta_yaw)
        delta_tx = target_x - (cosine * source_x - sine * source_y)
        delta_ty = target_y - (sine * source_x + cosine * source_y)
        transformed = tuple(MapPoint(
            delta_tx + cosine * point.x_m - sine * point.y_m,
            delta_ty + sine * point.x_m + cosine * point.y_m,
            point.z_m,
        ) for point in transformed)
        cosine_total, sine_total = math.cos(delta_yaw), math.sin(delta_yaw)
        tx, ty = (
            delta_tx + cosine_total * tx - sine_total * ty,
            delta_ty + sine_total * tx + cosine_total * ty,
        )
        yaw = math.atan2(
            math.sin(yaw + delta_yaw), math.cos(yaw + delta_yaw),
        )
        matched = len(selected)
    final_distances, _ = _nearest_xy(transformed, background)
    score = sorted(final_distances)[max(0, int(0.65 * (len(final_distances) - 1)))]
    if math.hypot(tx, ty) > 0.20 or abs(yaw) > 0.15 or score > 0.03:
        raise ObstacleLayerError("background registration is outside the bounded static-map correction envelope")
    return transformed, {
        "status": "PASS_STATIC_BACKGROUND_2D_REGISTRATION",
        "translation_x_m": tx,
        "translation_y_m": ty,
        "yaw_rad": yaw,
        "static_correspondence_count": matched,
        "trimmed_xy_score_m": score,
    }


class MapFrameObstaclePipeline:
    """Stateful temporal confirmation and re-plan helper; never an executor."""

    def __init__(self, config: ObstaclePipelineConfig) -> None:
        self.config = config
        if config.start in config.static_route_blocked or config.goal in config.static_route_blocked:
            raise ObstacleLayerError("static route map already blocks its start or goal")
        if config.minimum_obstacle_span_x_m < 0.0 or config.minimum_obstacle_span_y_m < 0.0:
            raise ObstacleLayerError("minimum obstacle spans must be non-negative")
        self.tracker = NovelObstacleTracker(config.grid, confirmation_frames=config.confirmation_frames)
        self.background_registration: dict[str, float | int | str] | None = None

    def observe(
        self,
        *,
        sequence: int,
        frame_id: str,
        data: bytes,
        width: int,
        height: int,
        point_step: int,
        field_offsets: dict[str, int],
    ) -> ObstaclePipelineDecision:
        try:
            points = decode_map_xyz(
                frame_id=frame_id,
                data=data,
                width=width,
                height=height,
                point_step=point_step,
                field_offsets=field_offsets,
                expected_frame_id=self.config.expected_frame_id,
            )
            observed_points = points
            if (
                self.config.map_background_points
                and self.config.background_registration_enabled
                and len(points) >= 80
                and len(self.config.map_background_points) >= 80
            ):
                if self.background_registration is None:
                    observed_points, self.background_registration = register_background_xy(
                        points, background=self.config.map_background_points,
                    )
                else:
                    yaw = float(self.background_registration["yaw_rad"])
                    tx = float(self.background_registration["translation_x_m"])
                    ty = float(self.background_registration["translation_y_m"])
                    cosine, sine = math.cos(yaw), math.sin(yaw)
                    observed_points = tuple(MapPoint(
                        tx + cosine * point.x_m - sine * point.y_m,
                        ty + sine * point.x_m + cosine * point.y_m,
                        point.z_m,
                    ) for point in points)
            novel = (
                novel_map_points_against_background(
                    observed_points,
                    background_points=self.config.map_background_points,
                    maximum_match_distance_m=self.config.static_match_distance_m,
                )
                if self.config.map_background_points else novel_map_points(
                    points,
                    static_surface=self.config.static_surface,
                    grid=self.config.grid,
                    static_match_distance_m=self.config.static_match_distance_m,
                )
            )
            raw_confirmed = self.tracker.update_positions(sequence, ((item.x_m, item.y_m) for item in novel))
        except ObstacleLayerError as exc:
            return ObstaclePipelineDecision(
                PipelineStatus.BLOCKED_INVALID_OBSERVATION, sequence, 0, 0, (), (), (), (), (), str(exc),
                self.background_registration,
            )
        qualified_components = tuple(
            component for component in connected_cell_components(raw_confirmed)
            if len(component) >= self.config.minimum_confirmed_cluster_cells
        )
        confirmed = frozenset().union(*qualified_components) if qualified_components else frozenset()
        envelope = frozenset().union(*(
            minimum_span_envelope(
                component,
                grid=self.config.grid,
                minimum_span_x_m=self.config.minimum_obstacle_span_x_m,
                minimum_span_y_m=self.config.minimum_obstacle_span_y_m,
            )
            for component in qualified_components
        )) if qualified_components else frozenset()
        inflated = inflate_cells(
            envelope, grid=self.config.grid, radius_m=self.config.obstacle_inflation_m,
        )
        blocked = frozenset((*self.config.static_route_blocked, *inflated))
        static_direct_blocked = direct_path_intersects_blocked(
            self.config.start, self.config.goal, blocked=self.config.static_route_blocked,
        )
        if not confirmed:
            if static_direct_blocked:
                try:
                    raw = plan_a_star(
                        self.config.start, self.config.goal, grid=self.config.grid,
                        blocked=self.config.static_route_blocked,
                    )
                    detour = simplify_grid_path(raw, blocked=self.config.static_route_blocked)
                except ObstacleLayerError as exc:
                    return ObstaclePipelineDecision(
                        PipelineStatus.BLOCKED_NO_PATH, sequence, len(points), len(novel), (), (), (), (), (), str(exc),
                    self.background_registration,
                    )
                return ObstaclePipelineDecision(
                    PipelineStatus.DETOUR_READY,
                    sequence, len(points), len(novel), (), (), (), detour,
                    path_world_points(detour, self.config.grid),
                    "known static-map occupancy blocks the direct route; the resulting detour still requires independent full-body clearance before execution",
                    self.background_registration,
                )
            return ObstaclePipelineDecision(
                PipelineStatus.CLEAR_DIRECT_PATH if not novel else PipelineStatus.WAITING_FOR_CONFIRMATION,
                sequence, len(points), len(novel), (), (), (), (), (),
                "no spatially confirmed new obstacle intersects a route decision",
                self.background_registration,
            )
        if not direct_path_intersects_blocked(self.config.start, self.config.goal, blocked=blocked):
            return ObstaclePipelineDecision(
                PipelineStatus.OBSTACLE_CLEAR_OF_DIRECT_PATH,
                sequence, len(points), len(novel), tuple(sorted(confirmed)), tuple(sorted(envelope)),
                tuple(sorted(inflated)), (), (),
                "confirmed obstacle is outside the current direct route corridor",
                self.background_registration,
            )
        try:
            raw = plan_a_star(self.config.start, self.config.goal, grid=self.config.grid, blocked=blocked)
            detour = simplify_grid_path(raw, blocked=blocked)
        except ObstacleLayerError as exc:
            return ObstaclePipelineDecision(
                PipelineStatus.BLOCKED_NO_PATH, sequence, len(points), len(novel), tuple(sorted(confirmed)), tuple(sorted(envelope)), tuple(sorted(inflated)), (), (), str(exc),
                self.background_registration,
            )
        return ObstaclePipelineDecision(
            PipelineStatus.DETOUR_READY,
            sequence,
            len(points),
            len(novel),
            tuple(sorted(confirmed)), tuple(sorted(envelope)),
            tuple(sorted(inflated)),
            detour,
            path_world_points(detour, self.config.grid),
            "map-frame novel obstacle confirmed; static-only detour requires independent full-body clearance before execution",
            self.background_registration,
        )
