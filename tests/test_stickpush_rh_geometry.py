from __future__ import annotations

import numpy as np

import taskbench.planners.stickpush_rh.geometry as geometry
from taskbench.planners.stickpush_rh.geometry import (
    hash_scene_state,
    point_segment_distance_xy,
    ray_length_to_shelf_edge,
    rotate_xy,
    within_shelf_xy,
)
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


def _scene_for_geometry() -> SceneState:
    target = ObjectState(
        name="target",
        center_xyz=np.array([0.35, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    blocker = ObjectState(
        name="b0",
        center_xyz=np.array([0.40, 0.05, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=False,
        active=True,
    )
    return SceneState(
        target=target,
        blockers=[blocker],
        all_objects=[target, blocker],
        shelf_front_x=0.2,
        shelf_back_x=0.7,
        shelf_half_w=0.25,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def test_within_shelf_and_ray_length():
    scene = _scene_for_geometry()
    p = np.array([0.40, 0.0], dtype=np.float32)
    d = rotate_xy(np.array([1.0, 0.0], dtype=np.float32), 0.0)

    assert within_shelf_xy(p, scene)
    ray_len = ray_length_to_shelf_edge(p, d, scene)
    assert ray_len is not None
    assert np.isclose(ray_len, scene.shelf_back_x - p[0], atol=1e-6)


def test_point_segment_distance_xy_matches_expected_projection():
    seg_a = np.array([0.0, 0.0], dtype=np.float32)
    seg_b = np.array([2.0, 0.0], dtype=np.float32)
    point = np.array([1.0, 0.3], dtype=np.float32)
    assert np.isclose(point_segment_distance_xy(point, seg_a, seg_b), 0.3, atol=1e-6)


def test_scene_hash_changes_with_object_motion():
    scene = _scene_for_geometry()
    h0 = hash_scene_state(scene)
    scene.blockers[0].center_xyz = scene.blockers[0].center_xyz + np.array([0.03, 0.0, 0.0], dtype=np.float32)
    h1 = hash_scene_state(scene)
    assert h0 != h1


def test_legacy_semicircle_helpers_removed_from_geometry_module():
    removed_symbols = (
        "grasp_semicircle_center_xy",
        "point_in_front_semicircle",
        "front_semicircle_surface_intersects",
        "collect_blockers_in_clearance",
        "compute_deficit",
        "compute_node_metrics",
        "segment_intersects_front_semicircle",
        "front_semicircle_wall_intersections",
    )
    for symbol in removed_symbols:
        assert not hasattr(geometry, symbol), f"legacy symbol should be removed: {symbol}"
