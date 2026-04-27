"""Deterministic straight-insertion feasibility on occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from taskbench.planners.stickpush_rh.config import (
    INSERTION_MODE_STRAIGHT_ONLY,
    SamplingConfig,
    canonical_insertion_mode,
)
from taskbench.planners.stickpush_rh.types import SceneState


@dataclass
class WavefrontGrid:
    """Configuration-space occupancy and opening entry candidates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    nx: int
    ny: int
    occupied: np.ndarray  # (ny, nx) bool
    source_cells: np.ndarray  # (K, 2) int32 in (ix, iy)
    source_world_xy: np.ndarray  # (K, 2) float32
    r_eff: float
    entry_x: float


@dataclass
class StraightInsertionPlan:
    """Single-segment insertion plan for a candidate push point."""

    source_xy: np.ndarray  # (2,) on opening line (inside shelf)
    approach_backoff: float
    sources_checked: int
    sources_total: int
    entry_xy_used: np.ndarray  # (2,) insertion endpoint used for feasibility/execution
    entry_shift_cells: tuple[int, int]  # (dx_cell, dy_cell) shift from original p1 cell
    entry_shift_xy: np.ndarray  # (2,) world shift from original p1


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
    """Build occupancy and free opening-line source candidates."""

    occupied, x_min, x_max, y_min, y_max, r_eff = _build_occupancy(scene, cfg)
    ny, nx = occupied.shape

    source_cells = []
    source_world = []
    ix_open = 0
    for iy in range(ny):
        if occupied[iy, ix_open]:
            continue
        source_cells.append((ix_open, iy))
        y = y_min + (float(iy) + 0.5) * float(cfg.insertion_grid_resolution)
        source_world.append(np.array([x_min, y], dtype=np.float32))

    source_cells_arr = (
        np.asarray(source_cells, dtype=np.int32)
        if source_cells
        else np.zeros((0, 2), dtype=np.int32)
    )
    source_world_arr = (
        np.asarray(source_world, dtype=np.float32)
        if source_world
        else np.zeros((0, 2), dtype=np.float32)
    )

    return WavefrontGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=float(cfg.insertion_grid_resolution),
        nx=nx,
        ny=ny,
        occupied=occupied,
        source_cells=source_cells_arr,
        source_world_xy=source_world_arr,
        r_eff=r_eff,
        entry_x=x_min,
    )


def _entry_y_interval(grid: WavefrontGrid, p1_xy: np.ndarray, stick_length: float) -> tuple[float, float] | None:
    p1 = np.asarray(p1_xy, dtype=np.float32).reshape(2)
    length = float(stick_length)
    if length <= 0.0:
        return None
    dx = float(p1[0] - grid.entry_x)
    if abs(dx) > length:
        return None
    dy_max = np.sqrt(max((length * length) - (dx * dx), 0.0))
    y_lo = max(float(grid.y_min), float(p1[1] - dy_max))
    y_hi = min(float(grid.y_max), float(p1[1] + dy_max))
    if y_lo > y_hi:
        return None
    return y_lo, y_hi


def _weighted_without_replacement_order(
    *,
    source_ys: np.ndarray,
    center_y: float,
    decay: float,
    uniform_mix: float,
    seed: int,
) -> np.ndarray:
    k = int(source_ys.shape[0])
    if k <= 1:
        return np.arange(k, dtype=np.int32)

    decay_eff = max(float(decay), 1e-6)
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    d = np.abs(np.asarray(source_ys, dtype=np.float64) - float(center_y))
    local = np.exp(-d / decay_eff)
    weights = (mix / float(k)) + (1.0 - mix) * local
    weights = np.clip(weights, 1e-12, None)

    rng = np.random.default_rng(int(seed))
    gumbel = rng.gumbel(loc=0.0, scale=1.0, size=k)
    scores = np.log(weights) + gumbel
    return np.asarray(np.argsort(scores)[::-1], dtype=np.int32)


def _candidate_seed(base_seed: int, dx_cell: int, dy_cell: int) -> int:
    if int(dx_cell) == 0 and int(dy_cell) == 0:
        return int(base_seed)
    # Deterministic per-candidate perturbation of base seed.
    mixed = (
        (int(base_seed) * 0x9E3779B97F4A7C15)
        ^ ((int(dx_cell) + 17) * 0xC2B2AE3D27D4EB4F)
        ^ ((int(dy_cell) + 31) * 0x165667B19E3779F9)
    ) & 0xFFFFFFFFFFFFFFFF
    return int(mixed)


def _endpoint_terminal_cells(grid: WavefrontGrid, end_xy: np.ndarray) -> set[tuple[int, int]]:
    p = np.asarray(end_xy, dtype=np.float64).reshape(2)
    x = float(p[0])
    y = float(p[1])
    eps = 1e-9
    if x < (grid.x_min - eps) or x > (grid.x_max + eps) or y < (grid.y_min - eps) or y > (grid.y_max + eps):
        return set()

    u = (x - grid.x_min) / grid.resolution
    v = (y - grid.y_min) / grid.resolution
    ix_guess = int(np.floor(u))
    iy_guess = int(np.floor(v))
    ix_guess = min(max(ix_guess, 0), grid.nx - 1)
    iy_guess = min(max(iy_guess, 0), grid.ny - 1)

    terminal: set[tuple[int, int]] = set()
    for iy in range(max(0, iy_guess - 1), min(grid.ny - 1, iy_guess + 1) + 1):
        y0 = grid.y_min + float(iy) * grid.resolution
        y1 = y0 + grid.resolution
        if y < (y0 - eps) or y > (y1 + eps):
            continue
        for ix in range(max(0, ix_guess - 1), min(grid.nx - 1, ix_guess + 1) + 1):
            x0 = grid.x_min + float(ix) * grid.resolution
            x1 = x0 + grid.resolution
            if x < (x0 - eps) or x > (x1 + eps):
                continue
            terminal.add((ix, iy))

    if not terminal:
        cell = world_to_cell(grid, p)
        if cell is not None:
            terminal.add(cell)
    return terminal


def _segment_cells_supercover(
    grid: WavefrontGrid,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> list[tuple[int, int]]:
    a = np.asarray(start_xy, dtype=np.float64).reshape(2)
    b = np.asarray(end_xy, dtype=np.float64).reshape(2)

    c0 = world_to_cell(grid, a)
    c1 = world_to_cell(grid, b)
    if c0 is None or c1 is None:
        return []

    ix, iy = int(c0[0]), int(c0[1])
    end_ix, end_iy = int(c1[0]), int(c1[1])
    cells: list[tuple[int, int]] = [(ix, iy)]

    if (ix, iy) == (end_ix, end_iy):
        return cells

    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    def _append(cx: int, cy: int) -> None:
        if cx < 0 or cx >= grid.nx or cy < 0 or cy >= grid.ny:
            return
        if cells and cells[-1] == (cx, cy):
            return
        cells.append((cx, cy))

    if step_x != 0:
        next_x = grid.x_min + float(ix + (1 if step_x > 0 else 0)) * grid.resolution
        t_max_x = (next_x - float(a[0])) / dx
        t_delta_x = grid.resolution / abs(dx)
    else:
        t_max_x = float("inf")
        t_delta_x = float("inf")

    if step_y != 0:
        next_y = grid.y_min + float(iy + (1 if step_y > 0 else 0)) * grid.resolution
        t_max_y = (next_y - float(a[1])) / dy
        t_delta_y = grid.resolution / abs(dy)
    else:
        t_max_y = float("inf")
        t_delta_y = float("inf")

    eps = 1e-12
    while (ix, iy) != (end_ix, end_iy):
        if t_max_x < (t_max_y - eps):
            ix += step_x
            t_max_x += t_delta_x
            if ix < 0 or ix >= grid.nx:
                return []
            _append(ix, iy)
            continue

        if t_max_y < (t_max_x - eps):
            iy += step_y
            t_max_y += t_delta_y
            if iy < 0 or iy >= grid.ny:
                return []
            _append(ix, iy)
            continue

        # Corner crossing: include both side-adjacent cells and the diagonal.
        next_ix = ix + step_x
        next_iy = iy + step_y

        if step_x != 0:
            _append(next_ix, iy)
            t_max_x += t_delta_x
        if step_y != 0:
            _append(ix, next_iy)
            t_max_y += t_delta_y

        ix = next_ix if step_x != 0 else ix
        iy = next_iy if step_y != 0 else iy
        if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny:
            return []
        _append(ix, iy)

    return cells


def line_collision_free_dda_supercover(
    grid: WavefrontGrid,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    allow_end_occupied: bool = False,
) -> bool:
    """Check straight segment against occupied cells via supercover DDA traversal."""

    a = np.asarray(start_xy, dtype=np.float32).reshape(2)
    b = np.asarray(end_xy, dtype=np.float32).reshape(2)

    if float(np.linalg.norm(b - a)) <= 1e-9:
        cell = world_to_cell(grid, b)
        if cell is None:
            return False
        ix, iy = cell
        if grid.occupied[iy, ix] and (not allow_end_occupied):
            return False
        return True

    visited = _segment_cells_supercover(grid, a, b)
    if not visited:
        return False

    terminal_cells = _endpoint_terminal_cells(grid, b) if allow_end_occupied else set()
    for ix, iy in visited:
        if not grid.occupied[iy, ix]:
            continue
        if allow_end_occupied and (ix, iy) in terminal_cells:
            continue
        return False
    return True


def line_collision_free(
    grid: WavefrontGrid,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    allow_end_occupied: bool = False,
) -> bool:
    """Backward-compatible alias to the DDA supercover line checker."""

    return line_collision_free_dda_supercover(
        grid,
        start_xy,
        end_xy,
        allow_end_occupied=allow_end_occupied,
    )


def _solve_straight_candidate(
    grid: WavefrontGrid,
    p1_xy: np.ndarray,
    cfg: SamplingConfig,
) -> tuple[np.ndarray, int, int] | None:
    interval = _entry_y_interval(grid, p1_xy, float(cfg.insertion_stick_length))
    if interval is None:
        return None
    y_lo, y_hi = interval
    p1_cell = world_to_cell(grid, p1_xy)
    if p1_cell is None:
        return None
    iy_target = int(p1_cell[1])
    source_rows = np.asarray(grid.source_cells[:, 1], dtype=np.int32)
    matches = np.flatnonzero(source_rows == iy_target)
    if matches.size == 0:
        return None
    source_idx = int(matches[0])
    source_xy = np.asarray(grid.source_world_xy[source_idx], dtype=np.float32)
    source_y = float(source_xy[1])
    if source_y < float(y_lo) or source_y > float(y_hi):
        return None
    if not line_collision_free_dda_supercover(grid, source_xy, p1_xy, allow_end_occupied=True):
        return None
    return source_xy, 1, 1


def _solve_diagonal_candidate(
    grid: WavefrontGrid,
    p1_xy: np.ndarray,
    cfg: SamplingConfig,
    *,
    deterministic_seed: int,
) -> tuple[np.ndarray, int, int] | None:
    interval = _entry_y_interval(grid, p1_xy, float(cfg.insertion_stick_length))
    if interval is None:
        return None
    y_lo, y_hi = interval

    source_ys = np.asarray(grid.source_world_xy[:, 1], dtype=np.float64)
    candidate_mask = (source_ys >= float(y_lo)) & (source_ys <= float(y_hi))
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return None

    order_local = _weighted_without_replacement_order(
        source_ys=source_ys[candidate_indices],
        center_y=float(p1_xy[1]),
        decay=float(cfg.insertion_entry_weight_decay),
        uniform_mix=float(cfg.insertion_entry_uniform_mix),
        seed=int(deterministic_seed),
    )

    ordered_indices = candidate_indices[np.asarray(order_local, dtype=np.int32)]
    sources_total = int(ordered_indices.size)
    sources_checked = 0
    for idx in ordered_indices:
        sources_checked += 1
        source_xy = np.asarray(grid.source_world_xy[int(idx)], dtype=np.float32)
        if not line_collision_free_dda_supercover(grid, source_xy, p1_xy, allow_end_occupied=True):
            continue
        return source_xy, int(sources_checked), int(sources_total)
    return None


def solve_straight_insertion(
    grid: WavefrontGrid,
    scene: SceneState,
    p1_xy: np.ndarray,
    cfg: SamplingConfig,
    *,
    deterministic_seed: int | None = None,
) -> StraightInsertionPlan | None:
    """Find one straight insertion segment from opening line to p1."""

    del scene
    if grid.source_world_xy.shape[0] == 0:
        return None

    p1 = np.asarray(p1_xy, dtype=np.float32).reshape(2)
    p1_cell = world_to_cell(grid, p1)
    if p1_cell is None:
        return None
    insertion_mode = canonical_insertion_mode(getattr(cfg, "insertion_mode", None))
    base_seed = 0 if deterministic_seed is None else int(deterministic_seed)
    candidate_offsets = (
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    ix0, iy0 = int(p1_cell[0]), int(p1_cell[1])
    for dx_cell, dy_cell in candidate_offsets:
        ix = ix0 + int(dx_cell)
        iy = iy0 + int(dy_cell)
        if ix < 0 or ix >= int(grid.nx) or iy < 0 or iy >= int(grid.ny):
            continue

        if int(dx_cell) == 0 and int(dy_cell) == 0:
            p1_candidate = np.asarray(p1, dtype=np.float32).copy()
        else:
            p1_candidate = cell_to_world(grid, (ix, iy))

        if insertion_mode == INSERTION_MODE_STRAIGHT_ONLY:
            solved = _solve_straight_candidate(grid, p1_candidate, cfg)
        else:
            solved = _solve_diagonal_candidate(
                grid,
                p1_candidate,
                cfg,
                deterministic_seed=_candidate_seed(base_seed, int(dx_cell), int(dy_cell)),
            )
        if solved is None:
            continue

        source_xy, sources_checked, sources_total = solved
        return StraightInsertionPlan(
            source_xy=np.asarray(source_xy, dtype=np.float32).copy(),
            approach_backoff=float(cfg.insertion_approach_backoff),
            sources_checked=int(sources_checked),
            sources_total=int(sources_total),
            entry_xy_used=np.asarray(p1_candidate, dtype=np.float32).copy(),
            entry_shift_cells=(int(dx_cell), int(dy_cell)),
            entry_shift_xy=(np.asarray(p1_candidate, dtype=np.float32) - np.asarray(p1, dtype=np.float32)),
        )
    return None
