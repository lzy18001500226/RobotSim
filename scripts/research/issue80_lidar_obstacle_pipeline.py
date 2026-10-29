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
from typing import Iterable

from issue80_lidar_map_obstacle_field import (
    GridSpec,
    decode_map_xyz,
    novel_map_points,
)
from issue80_lidar_obstacle_layer import (
    GridCell,
    NovelObstacleTracker,
    ObstacleLayerError,
    connected_cell_components,
    inflate_cells,
    path_world_points,
    plan_a_star,
    simplify_grid_path,
)


class PipelineStatus(str, Enum):
    CLEAR_DIRECT_PATH = "CLEAR_DIRECT_PATH_STATIC_ONLY"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_OBSTACLE_CONFIRMATION"
    DETOUR_READY = "DETOUR_READY_STATIC_ONLY"
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
    expected_frame_id: str = "map"


@dataclass(frozen=True)
class ObstaclePipelineDecision:
    status: PipelineStatus
    sequence: int
    observed_point_count: int
    novel_point_count: int
    confirmed_cells: tuple[GridCell, ...]
    inflated_dynamic_cells: tuple[GridCell, ...]
    detour_cells: tuple[GridCell, ...]
    detour_map_waypoints: tuple[tuple[float, float], ...]
    reason: str


class MapFrameObstaclePipeline:
    """Stateful temporal confirmation and re-plan helper; never an executor."""

    def __init__(self, config: ObstaclePipelineConfig) -> None:
        self.config = config
        if config.start in config.static_route_blocked or config.goal in config.static_route_blocked:
            raise ObstacleLayerError("static route map already blocks its start or goal")
        self.tracker = NovelObstacleTracker(config.grid, confirmation_frames=config.confirmation_frames)

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
            novel = novel_map_points(
                points,
                static_surface=self.config.static_surface,
                grid=self.config.grid,
                static_match_distance_m=self.config.static_match_distance_m,
            )
            raw_confirmed = self.tracker.update_positions(sequence, ((item.x_m, item.y_m) for item in novel))
        except ObstacleLayerError as exc:
            return ObstaclePipelineDecision(
                PipelineStatus.BLOCKED_INVALID_OBSERVATION, sequence, 0, 0, (), (), (), (), str(exc),
            )
        qualified_components = tuple(
            component for component in connected_cell_components(raw_confirmed)
            if len(component) >= self.config.minimum_confirmed_cluster_cells
        )
        confirmed = frozenset().union(*qualified_components) if qualified_components else frozenset()
        inflated = inflate_cells(
            confirmed, grid=self.config.grid, radius_m=self.config.obstacle_inflation_m,
        )
        blocked = frozenset((*self.config.static_route_blocked, *inflated))
        if not confirmed:
            return ObstaclePipelineDecision(
                PipelineStatus.CLEAR_DIRECT_PATH if not novel else PipelineStatus.WAITING_FOR_CONFIRMATION,
                sequence, len(points), len(novel), (), (), (), (),
                "no spatially confirmed new obstacle intersects a route decision",
            )
        try:
            raw = plan_a_star(self.config.start, self.config.goal, grid=self.config.grid, blocked=blocked)
            detour = simplify_grid_path(raw, blocked=blocked)
        except ObstacleLayerError as exc:
            return ObstaclePipelineDecision(
                PipelineStatus.BLOCKED_NO_PATH, sequence, len(points), len(novel), tuple(sorted(confirmed)), tuple(sorted(inflated)), (), (), str(exc),
            )
        return ObstaclePipelineDecision(
            PipelineStatus.DETOUR_READY,
            sequence,
            len(points),
            len(novel),
            tuple(sorted(confirmed)),
            tuple(sorted(inflated)),
            detour,
            path_world_points(detour, self.config.grid),
            "map-frame novel obstacle confirmed; static-only detour requires independent full-body clearance before execution",
        )
