from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.config import SamplingConfig
from taskbench.planners.stickpush_rh.insertion_grid import (
    WavefrontGrid,
    build_wavefront_grid,
    line_collision_free,
    line_collision_free_dda_supercover,
    solve_straight_insertion,
    world_to_cell,
)
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


def _scene_with_objects(objects: list[ObjectState]) -> SceneState:
    target = ObjectState(
        name="target",
        center_xyz=np.array([0.72, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    all_objects = [target, *objects]
    blockers = [o for o in all_objects if o.active and (not o.is_target)]
    return SceneState(
        target=target,
        blockers=blockers,
        all_objects=all_objects,
        shelf_front_x=0.20,
        shelf_back_x=0.90,
        shelf_half_w=0.25,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def _manual_grid(*, nx: int = 3, ny: int = 3, resolution: float = 1.0) -> WavefrontGrid:
    occupied = np.zeros((ny, nx), dtype=bool)
    source_cells = np.array([(0, iy) for iy in range(ny)], dtype=np.int32)
    source_world = np.array([[0.0, float(iy) + 0.5] for iy in range(ny)], dtype=np.float32)
    return WavefrontGrid(
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=3.0,
        resolution=float(resolution),
        nx=int(nx),
        ny=int(ny),
        occupied=occupied,
        source_cells=source_cells,
        source_world_xy=source_world,
        r_eff=0.0,
        entry_x=0.0,
    )


def test_straight_insertion_solver_finds_diagonal_path():
    front_obstacle = ObjectState(
        name="front_obs",
        center_xyz=np.array([0.235, 0.08, 0.41], dtype=np.float32),
        radius=0.020,
        is_target=False,
        active=True,
    )
    scene = _scene_with_objects([front_obstacle])
    cfg = SamplingConfig(
        insertion_grid_resolution=0.003,
        insertion_grid_safety_eps=0.0,
        insertion_approach_backoff=0.07,
        insertion_stick_length=0.50,
        insertion_entry_weight_decay=0.03,
        insertion_entry_uniform_mix=0.15,
    )
    grid = build_wavefront_grid(scene, cfg)
    p1_xy = np.array([0.56, 0.08], dtype=np.float32)
    plan = solve_straight_insertion(grid, scene, p1_xy, cfg, deterministic_seed=123)

    assert plan is not None
    # Opening near p1_y is blocked by the front obstacle, so selected source y
    # should differ and produce a diagonal insertion.
    assert abs(float(plan.source_xy[1]) - float(p1_xy[1])) > 0.003
    assert plan.sources_checked <= plan.sources_total
    assert line_collision_free(grid, plan.source_xy, p1_xy, allow_end_occupied=True)


def test_straight_insertion_solver_rejects_when_stick_too_short_for_entry_line():
    scene = _scene_with_objects([])
    cfg = SamplingConfig(
        insertion_grid_resolution=0.003,
        insertion_grid_safety_eps=0.0,
        insertion_stick_length=0.10,
    )
    grid = build_wavefront_grid(scene, cfg)
    p1_xy = np.array([0.62, 0.0], dtype=np.float32)
    plan = solve_straight_insertion(grid, scene, p1_xy, cfg, deterministic_seed=7)
    assert plan is None


def test_weighted_order_is_deterministic_for_same_seed_and_state():
    front_obstacle = ObjectState(
        name="front_obs",
        center_xyz=np.array([0.235, 0.04, 0.41], dtype=np.float32),
        radius=0.020,
        is_target=False,
        active=True,
    )
    scene = _scene_with_objects([front_obstacle])
    cfg = SamplingConfig(
        insertion_grid_resolution=0.003,
        insertion_grid_safety_eps=0.0,
        insertion_stick_length=0.50,
        insertion_entry_weight_decay=0.01,
        insertion_entry_uniform_mix=0.2,
    )
    grid = build_wavefront_grid(scene, cfg)
    p1_xy = np.array([0.56, 0.04], dtype=np.float32)

    plan_a = solve_straight_insertion(grid, scene, p1_xy, cfg, deterministic_seed=777)
    plan_b = solve_straight_insertion(grid, scene, p1_xy, cfg, deterministic_seed=777)

    assert plan_a is not None and plan_b is not None
    assert np.allclose(plan_a.source_xy, plan_b.source_xy)
    assert plan_a.sources_checked == plan_b.sources_checked
    assert plan_a.sources_total == plan_b.sources_total


def test_supercover_corner_touch_rejects_side_cell_collision():
    grid = _manual_grid()
    # Line from lower-left to upper-right passes exactly through (1, 1) corner.
    # Supercover should include side-adjacent cell (ix=1, iy=0) and reject.
    grid.occupied[0, 1] = True
    a = np.array([0.2, 0.2], dtype=np.float32)
    b = np.array([2.2, 2.2], dtype=np.float32)
    assert not line_collision_free_dda_supercover(grid, a, b, allow_end_occupied=False)


def test_endpoint_occupied_allowed_but_pre_end_occupied_rejected():
    grid = _manual_grid()
    start = np.array([0.2, 0.2], dtype=np.float32)
    end = np.array([2.2, 0.2], dtype=np.float32)

    # Endpoint cell occupied is allowed when allow_end_occupied=True.
    grid.occupied[:, :] = False
    grid.occupied[0, 2] = True
    assert line_collision_free_dda_supercover(grid, start, end, allow_end_occupied=True)
    assert not line_collision_free_dda_supercover(grid, start, end, allow_end_occupied=False)

    # Pre-end occupied cell must reject even with allow_end_occupied=True.
    grid.occupied[:, :] = False
    grid.occupied[0, 1] = True
    assert not line_collision_free_dda_supercover(grid, start, end, allow_end_occupied=True)


def test_world_to_cell_returns_none_out_of_bounds():
    scene = _scene_with_objects([])
    cfg = SamplingConfig(insertion_grid_resolution=0.003)
    grid = build_wavefront_grid(scene, cfg)
    assert world_to_cell(grid, np.array([scene.shelf_front_x - 0.01, 0.0], dtype=np.float32)) is None
