#!/usr/bin/env python3
"""Pure, fail-closed LiDAR obstacle extraction and grid detour planning.

This is an Issue #80 ``SIMULATION_ONLY`` research adapter.  It has no ROS,
MuJoCo, MC, or command-publisher dependency.  Its inputs are one raw
RayCaster LiDAR scan, the corresponding *known static-map* ranges, and a
LiDAR/odom pose estimate.  Its outputs are confirmed newly-occluding cells
and an A* detour made of map-frame waypoints.  A future executor may hand the
waypoints to the existing RouteMotionBridge only after its independent source,
mode, collision, and physical-stop gates pass.

The deliberately narrow detection rule is important: a return is considered
new only when it is materially *nearer* than the static-map return for the
same ray (or the static map had no return).  Existing walls, the source table,
and the source chair therefore remain static-map structure rather than being
re-discovered as surprise obstacles on every scan.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence


SCHEMA_VERSION = "robotsim.issue80.lidar-obstacle-layer.v1"


class ObstacleLayerError(ValueError):
    """The caller supplied an unsafe or internally inconsistent observation."""


@dataclass(frozen=True)
class Pose2D:
    """A planar robot pose in the declared map frame (REP-103 x-forward/y-left)."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.yaw_rad)):
            raise ObstacleLayerError("pose values must be finite")


@dataclass(frozen=True)
class RayCasterProfile:
    """The provenance-pinned chest RayCaster sampling geometry."""

    horizontal_rays: int
    vertical_rays: int
    horizontal_fov_rad: float
    vertical_fov_rad: float
    minimum_range_m: float
    maximum_range_m: float
    chest_x_offset_m: float

    def __post_init__(self) -> None:
        values = (
            self.horizontal_fov_rad,
            self.vertical_fov_rad,
            self.minimum_range_m,
            self.maximum_range_m,
            self.chest_x_offset_m,
        )
        if self.horizontal_rays < 2 or self.vertical_rays < 1:
            raise ObstacleLayerError("RayCaster must have at least two horizontal rays and one vertical ray")
        if not all(math.isfinite(value) for value in values):
            raise ObstacleLayerError("RayCaster profile must be finite")
        if not (0.0 < self.horizontal_fov_rad <= 2.0 * math.pi and 0.0 <= self.vertical_fov_rad <= math.pi):
            raise ObstacleLayerError("RayCaster field of view is invalid")
        if not (0.0 < self.minimum_range_m < self.maximum_range_m):
            raise ObstacleLayerError("RayCaster range is invalid")

    @property
    def ray_count(self) -> int:
        return self.horizontal_rays * self.vertical_rays


# This is a source-derived configuration, not a hardware sensor claim.  It
# matches the immutable provenance manifest for the existing Issue #80 overlay.
RAICOM_CHEST_RAYCASTER = RayCasterProfile(
    horizontal_rays=90,
    vertical_rays=5,
    horizontal_fov_rad=math.radians(270.0),
    vertical_fov_rad=math.radians(60.0),
    minimum_range_m=0.08,
    maximum_range_m=8.0,
    chest_x_offset_m=0.102632788961867,
)


@dataclass(frozen=True)
class NovelReturn:
    """One scan return that occludes a known static-map ray."""

    ray_index: int
    x_m: float
    y_m: float
    observed_range_m: float
    expected_range_m: float | None


def _finite_range(value: float | None, profile: RayCasterProfile) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ObstacleLayerError("range values must be numeric or None") from exc
    if not math.isfinite(number) or number < profile.minimum_range_m or number > profile.maximum_range_m:
        return None
    return number


def raw_xyz_ranges(raw_xyz: Sequence[object], profile: RayCasterProfile = RAICOM_CHEST_RAYCASTER) -> tuple[float | None, ...]:
    """Convert one append-only raw LiDAR frame to fixed ray-indexed ranges.

    Invalid/no-hit slots stay ``None``; they must never be compacted because
    static-map signatures use the original row-major ray ordering.
    """
    if len(raw_xyz) != profile.ray_count:
        raise ObstacleLayerError("raw LiDAR ray count does not match the profile")
    ranges: list[float | None] = []
    for point in raw_xyz:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            ranges.append(None)
            continue
        try:
            values = tuple(float(value) for value in point)
        except (TypeError, ValueError):
            ranges.append(None)
            continue
        if not all(math.isfinite(value) for value in values):
            ranges.append(None)
            continue
        ranges.append(_finite_range(math.sqrt(sum(value * value for value in values)), profile))
    return tuple(ranges)


def ray_direction_in_base(ray_index: int, profile: RayCasterProfile) -> tuple[float, float]:
    """Return projected forward/lateral direction for one RayCaster slot.

    The source camera makes camera ``-Z`` point forward and camera ``+X``
    point right.  The returned pair uses the RobotSim base convention x-forward
    and y-left.  Vertical pitch scales planar range but does not rotate the
    base-frame axes.
    """
    if not 0 <= ray_index < profile.ray_count:
        raise ObstacleLayerError("ray index is out of range")
    row, column = divmod(ray_index, profile.horizontal_rays)
    horizontal = (0.5 - column / (profile.horizontal_rays - 1)) * profile.horizontal_fov_rad
    vertical = 0.0 if profile.vertical_rays == 1 else (
        (0.5 - row / (profile.vertical_rays - 1)) * profile.vertical_fov_rad
    )
    planar = math.cos(vertical)
    # -camera Z is forward and -camera X is lateral-left.
    return math.cos(horizontal) * planar, math.sin(horizontal) * planar


def novel_returns_from_ranges(
    observed_ranges: Sequence[float | None],
    expected_static_ranges: Sequence[float | None],
    *,
    robot_pose: Pose2D,
    novelty_margin_m: float,
    profile: RayCasterProfile = RAICOM_CHEST_RAYCASTER,
) -> tuple[NovelReturn, ...]:
    """Extract newly occluding range returns in map coordinates.

    ``novelty_margin_m`` is a perception uncertainty gate, not a collision
    clearance.  It must be strictly positive and configured separately from
    the existing physical-stop and route-clearance thresholds.
    """
    if len(observed_ranges) != profile.ray_count or len(expected_static_ranges) != profile.ray_count:
        raise ObstacleLayerError("observed and expected scan arrays must preserve every ray slot")
    if not math.isfinite(novelty_margin_m) or novelty_margin_m <= 0.0:
        raise ObstacleLayerError("novelty margin must be finite and positive")

    cosine, sine = math.cos(robot_pose.yaw_rad), math.sin(robot_pose.yaw_rad)
    origin_x = robot_pose.x_m + cosine * profile.chest_x_offset_m
    origin_y = robot_pose.y_m + sine * profile.chest_x_offset_m
    output: list[NovelReturn] = []
    for index, (raw_observed, raw_expected) in enumerate(zip(observed_ranges, expected_static_ranges)):
        observed = _finite_range(raw_observed, profile)
        expected = _finite_range(raw_expected, profile)
        if observed is None:
            continue
        # An unknown static-map ray may be a genuine newly observed object.  A
        # farther return than a known static surface cannot be an occlusion.
        if expected is not None and observed > expected - novelty_margin_m:
            continue
        forward, lateral = ray_direction_in_base(index, profile)
        base_x, base_y = observed * forward, observed * lateral
        output.append(NovelReturn(
            ray_index=index,
            x_m=origin_x + cosine * base_x - sine * base_y,
            y_m=origin_y + sine * base_x + cosine * base_y,
            observed_range_m=observed,
            expected_range_m=expected,
        ))
    return tuple(output)


@dataclass(frozen=True, order=True)
class GridCell:
    x: int
    y: int


@dataclass(frozen=True)
class GridSpec:
    """A bounded map-frame occupancy grid; no implicit unbounded expansion."""

    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.origin_x_m, self.origin_y_m, self.resolution_m)):
            raise ObstacleLayerError("grid values must be finite")
        if self.resolution_m <= 0.0 or self.width <= 0 or self.height <= 0:
            raise ObstacleLayerError("grid resolution and dimensions must be positive")

    def contains(self, cell: GridCell) -> bool:
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def world_to_cell(self, x_m: float, y_m: float) -> GridCell:
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ObstacleLayerError("world coordinates must be finite")
        return GridCell(
            math.floor((x_m - self.origin_x_m) / self.resolution_m),
            math.floor((y_m - self.origin_y_m) / self.resolution_m),
        )

    def cell_center(self, cell: GridCell) -> tuple[float, float]:
        if not self.contains(cell):
            raise ObstacleLayerError("cell is outside the grid")
        return (
            self.origin_x_m + (cell.x + 0.5) * self.resolution_m,
            self.origin_y_m + (cell.y + 0.5) * self.resolution_m,
        )


class NovelObstacleTracker:
    """Require a strictly advancing sequence before exposing new obstacle cells."""

    def __init__(self, grid: GridSpec, *, confirmation_frames: int) -> None:
        if confirmation_frames < 1:
            raise ObstacleLayerError("confirmation_frames must be at least one")
        self.grid = grid
        self.confirmation_frames = confirmation_frames
        self._last_sequence: int | None = None
        self._hits: dict[GridCell, int] = {}

    def update(self, sequence: int, returns: Iterable[NovelReturn]) -> frozenset[GridCell]:
        return self.update_positions(sequence, ((item.x_m, item.y_m) for item in returns))

    def update_positions(self, sequence: int, positions: Iterable[tuple[float, float]]) -> frozenset[GridCell]:
        """Update confirmation state from finite map-frame planar positions."""
        if not isinstance(sequence, int) or sequence < 0:
            raise ObstacleLayerError("LiDAR sequence must be a non-negative integer")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ObstacleLayerError("LiDAR sequence must strictly advance; duplicate/reordered scans fail closed")
        self._last_sequence = sequence
        seen: set[GridCell] = set()
        for x_m, y_m in positions:
            cell = self.grid.world_to_cell(x_m, y_m)
            if self.grid.contains(cell):
                seen.add(cell)
        # A cell missing from a fresh frame loses its temporal evidence.  This
        # avoids treating a one-frame stale return as a persistent chair.
        self._hits = {cell: count for cell, count in self._hits.items() if cell in seen}
        for cell in seen:
            self._hits[cell] = self._hits.get(cell, 0) + 1
        return frozenset(cell for cell, count in self._hits.items() if count >= self.confirmation_frames)


def inflate_cells(
    occupied: Iterable[GridCell],
    *,
    grid: GridSpec,
    radius_m: float,
) -> frozenset[GridCell]:
    """Inflate occupied cells by an explicit caller-provided physical envelope."""
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ObstacleLayerError("inflation radius must be finite and non-negative")
    cell_radius = math.ceil(radius_m / grid.resolution_m)
    result: set[GridCell] = set()
    for center in occupied:
        if not grid.contains(center):
            continue
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                candidate = GridCell(center.x + dx, center.y + dy)
                if grid.contains(candidate) and math.hypot(dx, dy) * grid.resolution_m <= radius_m + 1e-12:
                    result.add(candidate)
    return frozenset(result)


def connected_cell_components(cells: Iterable[GridCell]) -> tuple[frozenset[GridCell], ...]:
    """Return deterministic eight-connected obstacle-cell components."""
    remaining = set(cells)
    output: list[frozenset[GridCell]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                GridCell(current.x + dx, current.y + dy)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if dx or dy
            }
            joined = neighbors & remaining
            remaining -= joined
            component.update(joined)
            frontier.extend(joined)
        output.append(frozenset(component))
    return tuple(sorted(output, key=lambda component: (min(component), len(component))))


def _neighbors(cell: GridCell, grid: GridSpec, blocked: frozenset[GridCell]) -> Iterable[tuple[GridCell, float]]:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        candidate = GridCell(cell.x + dx, cell.y + dy)
        if not grid.contains(candidate) or candidate in blocked:
            continue
        # A diagonal route must not squeeze between two touching blocked cells.
        if dx and dy and (GridCell(cell.x + dx, cell.y) in blocked or GridCell(cell.x, cell.y + dy) in blocked):
            continue
        yield candidate, math.hypot(dx, dy)


def plan_a_star(
    start: GridCell,
    goal: GridCell,
    *,
    grid: GridSpec,
    blocked: Iterable[GridCell],
) -> tuple[GridCell, ...]:
    """Return a reproducible, eight-connected collision-free grid path.

    This is intentionally a planner only.  It never publishes a velocity and
    cannot substitute for the existing full-body MuJoCo static-clearance audit
    before a generated detour may be executed.
    """
    frozen_blocked = frozenset(cell for cell in blocked if grid.contains(cell))
    if not grid.contains(start) or not grid.contains(goal):
        raise ObstacleLayerError("start and goal must be inside the occupancy grid")
    if start in frozen_blocked or goal in frozen_blocked:
        raise ObstacleLayerError("start or goal lies in an occupied/inflated cell")
    queue: list[tuple[float, float, GridCell]] = [(0.0, 0.0, start)]
    parent: dict[GridCell, GridCell | None] = {start: None}
    cost: dict[GridCell, float] = {start: 0.0}
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost != cost.get(current):
            continue
        if current == goal:
            path: list[GridCell] = []
            node: GridCell | None = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return tuple(reversed(path))
        for neighbor, step_cost in _neighbors(current, grid, frozen_blocked):
            candidate_cost = current_cost + step_cost
            if candidate_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = candidate_cost
            parent[neighbor] = current
            heuristic = math.hypot(goal.x - neighbor.x, goal.y - neighbor.y)
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, neighbor))
    raise ObstacleLayerError("no collision-free path exists in the bounded occupancy grid")


def path_world_points(path: Sequence[GridCell], grid: GridSpec) -> tuple[tuple[float, float], ...]:
    if not path:
        raise ObstacleLayerError("path must contain at least one cell")
    return tuple(grid.cell_center(cell) for cell in path)


def direct_path_intersects_blocked(
    start: GridCell,
    goal: GridCell,
    *,
    blocked: Iterable[GridCell],
) -> bool:
    """Conservatively sample a direct grid segment for replan decisions."""
    frozen_blocked = frozenset(blocked)
    count = max(abs(goal.x - start.x), abs(goal.y - start.y), 1)
    for index in range(count + 1):
        fraction = index / count
        cell = GridCell(round(start.x + (goal.x - start.x) * fraction), round(start.y + (goal.y - start.y) * fraction))
        if cell in frozen_blocked:
            return True
    return False


def simplify_grid_path(
    path: Sequence[GridCell],
    *,
    blocked: Iterable[GridCell],
) -> tuple[GridCell, ...]:
    """Remove only waypoints joined by a collision-free sampled line of sight."""
    if not path:
        raise ObstacleLayerError("path must contain at least one cell")
    frozen_blocked = frozenset(blocked)
    simplified: list[GridCell] = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        chosen = anchor + 1
        for candidate in range(len(path) - 1, anchor, -1):
            if not direct_path_intersects_blocked(path[anchor], path[candidate], blocked=frozen_blocked):
                chosen = candidate
                break
        simplified.append(path[chosen])
        anchor = chosen
    return tuple(simplified)
