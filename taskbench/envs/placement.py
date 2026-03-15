"""Utilities for fast 2D tabletop placement sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def axis_grid(lo: float, hi: float, step: float) -> np.ndarray:
    """Return evenly spaced samples over a closed interval."""
    if hi < lo:
        return np.empty((0,), dtype=np.float32)
    count = int(np.floor((hi - lo) / step)) + 1
    if count <= 1:
        return np.array([(lo + hi) * 0.5], dtype=np.float32)
    return np.linspace(lo, lo + step * (count - 1), count, dtype=np.float32)


@dataclass(frozen=True)
class PlacementGrid:
    """Rectangular lattice used for cheap non-overlapping placement."""

    xs: np.ndarray
    ys: np.ndarray
    centers_xy: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ys), len(self.xs))

    @property
    def num_cells(self) -> int:
        return int(self.centers_xy.shape[0])

    @property
    def spacing(self) -> float:
        steps = []
        if len(self.xs) > 1:
            steps.append(float(np.min(np.diff(self.xs))))
        if len(self.ys) > 1:
            steps.append(float(np.min(np.diff(self.ys))))
        return min(steps) if steps else float("inf")

    def neighbor_indices(self, index: int, radius: int = 1) -> np.ndarray:
        if radius < 1:
            return np.empty((0,), dtype=np.int32)
        n_x = len(self.xs)
        iy = index // n_x
        ix = index % n_x
        rows = range(max(0, iy - radius), min(len(self.ys), iy + radius + 1))
        cols = range(max(0, ix - radius), min(n_x, ix + radius + 1))
        neighbors = [
            row * n_x + col
            for row in rows
            for col in cols
            if not (row == iy and col == ix)
        ]
        return np.asarray(neighbors, dtype=np.int32)


def build_rect_grid(lo_xy, hi_xy, spacing: float) -> PlacementGrid:
    lo_xy = np.asarray(lo_xy, dtype=np.float32).reshape(2)
    hi_xy = np.asarray(hi_xy, dtype=np.float32).reshape(2)
    xs = axis_grid(float(lo_xy[0]), float(hi_xy[0]), spacing)
    ys = axis_grid(float(lo_xy[1]), float(hi_xy[1]), spacing)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Workspace bounds do not admit any placement cells")
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    centers_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    return PlacementGrid(xs=xs, ys=ys, centers_xy=centers_xy)


def clamp_jitter(
    requested_jitter: float, *, grid_spacing: float, min_center_distance: float
) -> float:
    if not np.isfinite(grid_spacing):
        return 0.0
    safe_jitter = max(0.0, 0.5 * (grid_spacing - min_center_distance))
    return float(min(requested_jitter, safe_jitter))


_LAYOUT_MODES: dict[str, dict[str, float | int]] = {
    "dense": {
        "strategy": "frontier",
        "seed_min": 1,
        "seed_max": 2,
        "frontier_prob": 0.94,
        "neighbor_radius": 1,
    },
    "mixed": {
        "strategy": "clustered",
        "cluster_min": 2,
        "cluster_max": 4,
        "cluster_spread_prob": 0.18,
        "neighbor_radius": 1,
        "seed_separation_cells": 2,
    },
    "medium": {
        "strategy": "clustered",
        "cluster_min": 2,
        "cluster_max": 4,
        "cluster_spread_prob": 0.18,
        "neighbor_radius": 1,
        "seed_separation_cells": 2,
    },
    "spread": {
        "strategy": "frontier",
        "seed_min": 4,
        "seed_max": 6,
        "frontier_prob": 0.38,
        "neighbor_radius": 1,
    },
}


def _occupy_frontier_index(
    index: int,
    *,
    available: np.ndarray,
    occupied: list[int],
    frontier: set[int],
    grid: PlacementGrid,
    neighbor_radius: int,
) -> bool:
    index = int(index)
    if not available[index]:
        return False
    available[index] = False
    frontier.discard(index)
    occupied.append(index)
    neighbors = grid.neighbor_indices(index, radius=neighbor_radius)
    for neighbor in neighbors:
        neighbor = int(neighbor)
        if available[neighbor]:
            frontier.add(neighbor)
    return True


def _sample_seed_indices(
    rng: np.random.Generator,
    grid: PlacementGrid,
    *,
    seed_count: int,
    min_separation_cells: int,
) -> np.ndarray:
    total = grid.num_cells
    if seed_count >= total:
        return np.arange(total, dtype=np.int32)

    candidate_order = rng.permutation(total)
    chosen: list[int] = []
    min_sep = max(0, int(min_separation_cells))
    n_x = len(grid.xs)

    def far_enough(candidate: int) -> bool:
        if min_sep <= 0:
            return True
        cy = candidate // n_x
        cx = candidate % n_x
        for existing in chosen:
            ey = existing // n_x
            ex = existing % n_x
            if max(abs(cx - ex), abs(cy - ey)) < min_sep:
                return False
        return True

    for candidate in candidate_order:
        candidate = int(candidate)
        if far_enough(candidate):
            chosen.append(candidate)
            if len(chosen) >= seed_count:
                break

    if len(chosen) < seed_count:
        remaining = [int(v) for v in candidate_order if int(v) not in chosen]
        chosen.extend(remaining[: seed_count - len(chosen)])

    return np.asarray(chosen, dtype=np.int32)


def _sample_clustered_cells(
    rng: np.random.Generator,
    grid: PlacementGrid,
    *,
    num_cells: int,
    mode: str,
    params: dict[str, float | int],
) -> tuple[np.ndarray, dict[str, object]]:
    total = grid.num_cells
    available = np.ones(total, dtype=bool)
    occupied: list[int] = []
    neighbor_radius = int(params["neighbor_radius"])

    cluster_min = int(params["cluster_min"])
    cluster_max = int(params["cluster_max"])
    cluster_count = min(num_cells, int(rng.integers(cluster_min, cluster_max + 1)))
    seed_indices = _sample_seed_indices(
        rng,
        grid,
        seed_count=cluster_count,
        min_separation_cells=int(params.get("seed_separation_cells", 0)),
    )

    frontiers: list[set[int]] = [set() for _ in range(cluster_count)]
    cluster_members: list[list[int]] = [[] for _ in range(cluster_count)]
    target_sizes = np.ones(cluster_count, dtype=np.int32)
    remaining = num_cells - cluster_count
    if remaining > 0:
        weights = rng.random(cluster_count).astype(np.float64) + 0.5
        extra = np.floor(weights / weights.sum() * remaining).astype(np.int32)
        extra_total = int(extra.sum())
        if extra_total < remaining:
            order = rng.permutation(cluster_count)
            for idx in order[: remaining - extra_total]:
                extra[int(idx)] += 1
        target_sizes += extra

    for cluster_idx, seed_index in enumerate(seed_indices):
        _occupy_frontier_index(
            int(seed_index),
            available=available,
            occupied=occupied,
            frontier=frontiers[cluster_idx],
            grid=grid,
            neighbor_radius=neighbor_radius,
        )
        cluster_members[cluster_idx].append(int(seed_index))

    cluster_spread_prob = float(params["cluster_spread_prob"])
    active_clusters = [idx for idx in range(cluster_count)]
    while len(occupied) < num_cells and active_clusters:
        rng.shuffle(active_clusters)
        next_active: list[int] = []
        for cluster_idx in active_clusters:
            if len(occupied) >= num_cells:
                break
            if len(cluster_members[cluster_idx]) >= int(target_sizes[cluster_idx]):
                continue

            frontier_indices = np.fromiter(frontiers[cluster_idx], dtype=np.int32)
            if len(frontier_indices) > 0 and rng.random() >= cluster_spread_prob:
                next_index = int(frontier_indices[rng.integers(len(frontier_indices))])
            else:
                if frontier_indices.size == 0:
                    # Restart this cluster near its seed if it stalled.
                    seed_xy = grid.centers_xy[int(seed_indices[cluster_idx])]
                    dists = np.linalg.norm(grid.centers_xy - seed_xy[None, :], axis=1)
                    remaining = np.flatnonzero(available)
                    if len(remaining) == 0:
                        break
                    order = remaining[np.argsort(dists[remaining])]
                    next_index = int(order[0])
                else:
                    remaining = np.flatnonzero(available)
                    if len(remaining) == 0:
                        break
                    next_index = int(remaining[rng.integers(len(remaining))])

            if _occupy_frontier_index(
                next_index,
                available=available,
                occupied=occupied,
                frontier=frontiers[cluster_idx],
                grid=grid,
                neighbor_radius=neighbor_radius,
            ):
                cluster_members[cluster_idx].append(int(next_index))

            if len(cluster_members[cluster_idx]) < int(target_sizes[cluster_idx]):
                next_active.append(cluster_idx)

        active_clusters = next_active
        if not active_clusters and len(occupied) < num_cells:
            remaining = np.flatnonzero(available)
            if len(remaining) == 0:
                break
            spill_count = min(num_cells - len(occupied), len(remaining))
            spill = rng.choice(remaining, size=spill_count, replace=False)
            occupied.extend(int(v) for v in spill)

    occupied_indices = np.asarray(occupied[:num_cells], dtype=np.int32)
    metadata = {
        "layout_mode": mode,
        "layout_variant": "medium_clusters",
        "seed_indices": seed_indices.astype(np.int32),
        "cluster_count": int(cluster_count),
        "cluster_target_sizes": target_sizes.astype(np.int32),
        "cluster_spread_prob": cluster_spread_prob,
        "neighbor_radius": neighbor_radius,
    }
    return occupied_indices, metadata


def sample_frontier_cells(
    rng: np.random.Generator,
    grid: PlacementGrid,
    *,
    num_cells: int,
    mode: str = "auto",
    dense_layout_prob: float = 0.55,
    mixed_layout_prob: float = 0.30,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample occupied cells by repeatedly expanding a frontier on the grid."""
    if num_cells > grid.num_cells:
        raise ValueError(
            f"Requested {num_cells} cells, but grid only has {grid.num_cells}"
        )
    if mode == "auto":
        spread_layout_prob = 1.0 - dense_layout_prob - mixed_layout_prob
        probs = np.asarray(
            [dense_layout_prob, mixed_layout_prob, spread_layout_prob], dtype=np.float64
        )
        if np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError(
                "dense_layout_prob and mixed_layout_prob must leave a positive spread"
            )
        probs /= probs.sum()
        mode = str(rng.choice(["dense", "mixed", "spread"], p=probs))
    if mode not in _LAYOUT_MODES:
        options = ", ".join(sorted(_LAYOUT_MODES))
        raise ValueError(
            f"Unsupported placement mode {mode!r}; expected one of {options}"
        )

    params = _LAYOUT_MODES[mode]
    strategy = str(params.get("strategy", "frontier"))
    if strategy == "clustered":
        return _sample_clustered_cells(
            rng,
            grid,
            num_cells=num_cells,
            mode=mode,
            params=params,
        )

    total = grid.num_cells
    available = np.ones(total, dtype=bool)
    occupied: list[int] = []
    frontier: set[int] = set()

    seed_min = int(params["seed_min"])
    seed_max = int(params["seed_max"])
    seed_count = min(num_cells, int(rng.integers(seed_min, seed_max + 1)))
    seed_indices = rng.choice(total, size=seed_count, replace=False)

    for seed_index in seed_indices:
        _occupy_frontier_index(
            int(seed_index),
            available=available,
            occupied=occupied,
            frontier=frontier,
            grid=grid,
            neighbor_radius=int(params["neighbor_radius"]),
        )

    frontier_prob = float(params["frontier_prob"])
    while len(occupied) < num_cells:
        frontier_indices = np.fromiter(frontier, dtype=np.int32)
        if len(frontier_indices) > 0 and rng.random() < frontier_prob:
            next_index = int(frontier_indices[rng.integers(len(frontier_indices))])
        else:
            remaining = np.flatnonzero(available)
            next_index = int(remaining[rng.integers(len(remaining))])
        _occupy_frontier_index(
            next_index,
            available=available,
            occupied=occupied,
            frontier=frontier,
            grid=grid,
            neighbor_radius=int(params["neighbor_radius"]),
        )

    occupied_indices = np.asarray(occupied, dtype=np.int32)
    metadata = {
        "layout_mode": mode,
        "seed_indices": np.asarray(seed_indices, dtype=np.int32),
        "frontier_prob": frontier_prob,
        "neighbor_radius": int(params["neighbor_radius"]),
    }
    return occupied_indices, metadata


def jitter_positions(
    rng: np.random.Generator,
    centers_xy: np.ndarray,
    *,
    max_jitter: float,
    lo_xy=None,
    hi_xy=None,
) -> np.ndarray:
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    positions_xy = centers_xy.copy()
    if max_jitter > 0 and len(positions_xy) > 0:
        positions_xy += rng.uniform(
            low=-max_jitter, high=max_jitter, size=positions_xy.shape
        ).astype(np.float32)
    if lo_xy is not None and hi_xy is not None:
        lo_xy = np.asarray(lo_xy, dtype=np.float32).reshape(2)
        hi_xy = np.asarray(hi_xy, dtype=np.float32).reshape(2)
        positions_xy = np.clip(positions_xy, lo_xy, hi_xy)
    return positions_xy.astype(np.float32)
