"""Deterministic straight-insertion feasibility via node-level wavefront."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from taskbench.planners.stickpush_rh.config import SamplingConfig
from taskbench.planners.stickpush_rh.types import SceneState


@dataclass
class WavefrontGrid:
    """Configuration-space occupancy and source-distance map."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    nx: int
    ny: int
    occupied: np.ndarray  # (ny, nx) bool
    dist: np.ndarray  # (ny, nx) int32, -1 = unreachable
    source_cells: np.ndarray  # (K, 2) int32 in (ix, iy)
    source_world_xy: np.ndarray  # (K, 2) float32
    r_eff: float


@dataclass
class StraightInsertionPlan:
    """Single-segment insertion plan for a candidate push point."""

    source_xy: np.ndarray  # (2,) on opening line (inside shelf)
    approach_backoff: float
    wavefront_dist: int


def _grid_bounds(scene: SceneState, r_eff: float, resolution: float) -> tuple[float, float, float, float]:
    x_min = float(scene.shelf_front_x + r_eff)
    x_max = float(scene.shelf_back_x - r_eff)
    y_min = float(-scene.shelf_half_w + r_eff)
    y_max = float(scene.shelf_half_w - r_eff)
    if x_max <= x_min:
        x_mid = 0.5 * (x_min + x_max)
        x_min = x_mid - max(0.5 * resolution, 1e-6)
        x_max = x_mid + max(0.5 * resolution, 1e-6)
    if y_max <= y_min:
        y_mid = 0.5 * (y_min + y_max)
        y_min = y_mid - max(0.5 * resolution, 1e-6)
        y_max = y_mid + max(0.5 * resolution, 1e-6)
    return x_min, x_max, y_min, y_max


def _grid_shape(x_min: float, x_max: float, y_min: float, y_max: float, resolution: float) -> tuple[int, int]:
    nx = int(np.floor((x_max - x_min) / resolution)) + 1
    ny = int(np.floor((y_max - y_min) / resolution)) + 1
    return max(nx, 1), max(ny, 1)


def world_to_cell(grid: WavefrontGrid, xy: np.ndarray) -> tuple[int, int] | None:
    x = float(xy[0])
    y = float(xy[1])
    if x < grid.x_min or x > grid.x_max or y < grid.y_min or y > grid.y_max:
        return None
    ix = int(np.floor((x - grid.x_min) / grid.resolution))
    iy = int(np.floor((y - grid.y_min) / grid.resolution))
    ix = min(max(ix, 0), grid.nx - 1)
    iy = min(max(iy, 0), grid.ny - 1)
    return ix, iy


def cell_to_world(grid: WavefrontGrid, cell: tuple[int, int]) -> np.ndarray:
    ix, iy = int(cell[0]), int(cell[1])
    x = grid.x_min + (float(ix) + 0.5) * grid.resolution
    y = grid.y_min + (float(iy) + 0.5) * grid.resolution
    return np.array([x, y], dtype=np.float32)


def _build_occupancy(scene: SceneState, cfg: SamplingConfig) -> tuple[np.ndarray, float, float, float, float, float]:
    resolution = float(cfg.insertion_grid_resolution)
    r_eff = float(cfg.stick_radius + cfg.contact_margin + cfg.insertion_grid_safety_eps)
    x_min, x_max, y_min, y_max = _grid_bounds(scene, r_eff, resolution)
    nx, ny = _grid_shape(x_min, x_max, y_min, y_max, resolution)

    xs = x_min + (np.arange(nx, dtype=np.float32) + 0.5) * resolution
    ys = y_min + (np.arange(ny, dtype=np.float32) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    occupied = np.zeros((ny, nx), dtype=bool)
    for obj in scene.all_objects:
        if not obj.active:
            continue
        ox = float(obj.center_xyz[0])
        oy = float(obj.center_xyz[1])
        rr = float(obj.radius + r_eff)
        occupied |= ((xx - ox) ** 2 + (yy - oy) ** 2) < (rr * rr)

    return occupied, x_min, x_max, y_min, y_max, r_eff


def build_wavefront_grid(scene: SceneState, cfg: SamplingConfig) -> WavefrontGrid:
    """Build occupancy and one BFS wavefront from shelf opening."""

    occupied, x_min, x_max, y_min, y_max, r_eff = _build_occupancy(scene, cfg)
    ny, nx = occupied.shape
    dist = -np.ones((ny, nx), dtype=np.int32)

    source_cells = []
    source_world = []
    ix_open = 0
    for iy in range(ny):
        if occupied[iy, ix_open]:
            continue
        dist[iy, ix_open] = 0
        source_cells.append((ix_open, iy))

    q: deque[tuple[int, int]] = deque(source_cells)
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    while q:
        ix, iy = q.popleft()
        d = int(dist[iy, ix])
        for dx, dy in neighbors:
            nx_i = ix + dx
            ny_i = iy + dy
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                continue
            if occupied[ny_i, nx_i]:
                continue
            if dist[ny_i, nx_i] >= 0:
                continue
            dist[ny_i, nx_i] = d + 1
            q.append((nx_i, ny_i))

    source_cells_arr = (
        np.asarray(source_cells, dtype=np.int32)
        if source_cells
        else np.zeros((0, 2), dtype=np.int32)
    )
    if source_cells:
        for ix, iy in source_cells:
            x = x_min + (float(ix) + 0.5) * float(cfg.insertion_grid_resolution)
            y = y_min + (float(iy) + 0.5) * float(cfg.insertion_grid_resolution)
            source_world.append(np.array([x, y], dtype=np.float32))
    source_world_arr = np.asarray(source_world, dtype=np.float32) if source_world else np.zeros((0, 2), dtype=np.float32)

    return WavefrontGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=float(cfg.insertion_grid_resolution),
        nx=nx,
        ny=ny,
        occupied=occupied,
        dist=dist,
        source_cells=source_cells_arr,
        source_world_xy=source_world_arr,
        r_eff=r_eff,
    )


def _nearest_reachable_cell(
    grid: WavefrontGrid,
    p1_xy: np.ndarray,
    endpoint_tolerance_cells: int,
) -> tuple[int, int] | None:
    cell = world_to_cell(grid, p1_xy)
    if cell is None:
        return None
    ix0, iy0 = cell
    if (not grid.occupied[iy0, ix0]) and int(grid.dist[iy0, ix0]) >= 0:
        return ix0, iy0

    best_cell = None
    best_sq = float("inf")
    for r in range(1, int(max(endpoint_tolerance_cells, 0)) + 1):
        x0 = max(0, ix0 - r)
        x1 = min(grid.nx - 1, ix0 + r)
        y0 = max(0, iy0 - r)
        y1 = min(grid.ny - 1, iy0 + r)
        for iy in range(y0, y1 + 1):
            for ix in range(x0, x1 + 1):
                if grid.occupied[iy, ix]:
                    continue
                if int(grid.dist[iy, ix]) < 0:
                    continue
                dx = float(ix - ix0)
                dy = float(iy - iy0)
                sq = dx * dx + dy * dy
                if sq < best_sq:
                    best_sq = sq
                    best_cell = (ix, iy)
        if best_cell is not None:
            return best_cell
    return None


def line_collision_free(
    grid: WavefrontGrid,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    allow_end_occupied: bool = False,
) -> bool:
    """Check straight segment against occupied cells using super-sampling."""

    a = np.asarray(start_xy, dtype=np.float32).reshape(2)
    b = np.asarray(end_xy, dtype=np.float32).reshape(2)
    length = float(np.linalg.norm(b - a))
    if length <= 1e-9:
        return True
    # Sample denser than cell size so diagonal corner grazes are captured.
    n_steps = max(int(np.ceil(length / max(0.5 * grid.resolution, 1e-6))), 1)
    for k in range(n_steps + 1):
        t = float(k) / float(n_steps)
        p = (1.0 - t) * a + t * b
        cell = world_to_cell(grid, p)
        if cell is None:
            continue
        ix, iy = cell
        if not grid.occupied[iy, ix]:
            continue
        if allow_end_occupied and k == n_steps:
            continue
        return False
    return True


def solve_straight_insertion(
    grid: WavefrontGrid,
    scene: SceneState,
    p1_xy: np.ndarray,
    cfg: SamplingConfig,
) -> StraightInsertionPlan | None:
    """Find a deterministic straight insertion segment from opening to p1."""

    if grid.source_world_xy.shape[0] == 0:
        return None

    reachable_cell = _nearest_reachable_cell(
        grid,
        p1_xy,
        endpoint_tolerance_cells=int(cfg.insertion_endpoint_tolerance_cells),
    )
    if reachable_cell is None:
        return None
    rc_ix, rc_iy = reachable_cell
    wavefront_dist = int(grid.dist[rc_iy, rc_ix])

    p1_y = float(p1_xy[1])
    ys = grid.source_world_xy[:, 1]
    order_full = np.argsort(np.abs(ys - p1_y), kind="stable")
    max_trials = int(cfg.insertion_max_source_trials)
    if max_trials > 0:
        order_primary = order_full[:max_trials]
        order_fallback = order_full[max_trials:]
    else:
        order_primary = order_full
        order_fallback = np.zeros((0,), dtype=order_full.dtype)

    def _first_collision_free(order: np.ndarray) -> np.ndarray | None:
        for idx in order:
            source_xy = grid.source_world_xy[int(idx)]
            if not line_collision_free(grid, source_xy, p1_xy, allow_end_occupied=True):
                continue
            return source_xy
        return None

    # Fast path: closest-y opening sources first.
    source_xy = _first_collision_free(order_primary)
    if source_xy is None and order_fallback.size > 0:
        # Fallback prevents false negatives when a feasible insertion exists but
        # lies outside the truncated nearest-y source set.
        source_xy = _first_collision_free(order_fallback)
    if source_xy is not None:
        return StraightInsertionPlan(
            source_xy=np.asarray(source_xy, dtype=np.float32),
            approach_backoff=float(cfg.insertion_approach_backoff),
            wavefront_dist=wavefront_dist,
        )

    return None
