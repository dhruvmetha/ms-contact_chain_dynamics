from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.config import SamplingConfig
from taskbench.planners.stickpush_rh.insertion_grid import (
    build_wavefront_grid,
    line_collision_free,
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
        insertion_max_source_trials=128,
        insertion_endpoint_tolerance_cells=2,
    )
    grid = build_wavefront_grid(scene, cfg)
    p1_xy = np.array([0.56, 0.08], dtype=np.float32)
    plan = solve_straight_insertion(grid, scene, p1_xy, cfg)

    assert plan is not None
    # The opening cell at p1_y is blocked by the front obstacle, so selected
    # source y should differ and create a diagonal straight insertion.
    assert abs(float(plan.source_xy[1]) - float(p1_xy[1])) > 0.003
    assert line_collision_free(grid, plan.source_xy, p1_xy, allow_end_occupied=True)


def test_straight_insertion_solver_rejects_unreachable_target():
    blocking_wall = ObjectState(
        name="wall_like",
        center_xyz=np.array([0.36, 0.0, 0.41], dtype=np.float32),
        radius=0.26,
        is_target=False,
        active=True,
    )
    scene = _scene_with_objects([blocking_wall])
    cfg = SamplingConfig(
        insertion_grid_resolution=0.003,
        insertion_grid_safety_eps=0.0,
        insertion_approach_backoff=0.07,
    )
    grid = build_wavefront_grid(scene, cfg)
    p1_xy = np.array([0.62, 0.0], dtype=np.float32)
    plan = solve_straight_insertion(grid, scene, p1_xy, cfg)
    assert plan is None


def test_world_to_cell_returns_none_out_of_bounds():
    scene = _scene_with_objects([])
    cfg = SamplingConfig(insertion_grid_resolution=0.003)
    grid = build_wavefront_grid(scene, cfg)
    assert world_to_cell(grid, np.array([scene.shelf_front_x - 0.01, 0.0], dtype=np.float32)) is None
