from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.geometry import (
    collect_blockers_in_clearance,
    compute_node_metrics,
    front_semicircle_wall_intersections,
    front_semicircle_surface_intersects,
    point_in_front_semicircle,
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
    front_blocker = ObjectState(
        name="b_front",
        center_xyz=np.array([0.31, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=False,
        active=True,
    )
    back_blocker = ObjectState(
        name="b_back",
        center_xyz=np.array([0.41, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=False,
        active=True,
    )
    far_blocker = ObjectState(
        name="b_far",
        center_xyz=np.array([0.30, 0.08, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=False,
        active=True,
    )
    return SceneState(
        target=target,
        blockers=[front_blocker, back_blocker, far_blocker],
        all_objects=[target, front_blocker, back_blocker, far_blocker],
        shelf_front_x=0.2,
        shelf_back_x=0.7,
        shelf_half_w=0.25,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def test_front_semicircle_surface_intersection():
    scene = _scene_for_geometry()
    target = scene.target
    front, back, far = scene.blockers

    assert front_semicircle_surface_intersects(front, target, 0.05, scene.open_dir_xy)
    assert not front_semicircle_surface_intersects(back, target, 0.05, scene.open_dir_xy)
    assert not front_semicircle_surface_intersects(far, target, 0.05, scene.open_dir_xy)


def test_metrics_include_surface_intersection():
    scene = _scene_for_geometry()
    blockers = collect_blockers_in_clearance(scene, clearance_radius=0.05)
    assert [b.name for b in blockers] == ["b_front"]

    metrics = compute_node_metrics(scene, clearance_radius=0.05, pushes_used=3)
    assert metrics.blockers == 1
    assert metrics.pushes_used == 3
    assert not metrics.solved
    assert np.isclose(metrics.min_margin, 0.004, atol=1e-4)
    # Deficit is measured from grasp semicircle center (target back-line midpoint).
    assert np.isclose(metrics.deficit, 0.0100000017, atol=1e-4)


def test_point_in_front_semicircle():
    scene = _scene_for_geometry()
    target = scene.target
    r = 0.05

    # Front of target (toward opening, -X) and within radius.
    p_front_inside = np.array([0.33, 0.0], dtype=np.float32)
    # Back of target (+X) should not count even if within disk.
    p_back_inside = np.array([0.39, 0.0], dtype=np.float32)
    # Front but outside radius.
    p_front_outside = np.array([0.29, 0.0], dtype=np.float32)

    assert point_in_front_semicircle(
        p_front_inside, target, r, scene.open_dir_xy
    )
    assert not point_in_front_semicircle(
        p_back_inside, target, r, scene.open_dir_xy
    )
    assert not point_in_front_semicircle(
        p_front_outside, target, r, scene.open_dir_xy
    )


def _scene_for_wall_intersection_checks(target_xy: tuple[float, float]) -> SceneState:
    target = ObjectState(
        name="target",
        center_xyz=np.array([target_xy[0], target_xy[1], 0.41], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    return SceneState(
        target=target,
        blockers=[],
        all_objects=[target],
        shelf_front_x=0.2,
        shelf_back_x=0.45,
        shelf_half_w=0.25,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def test_front_semicircle_wall_intersections_ignore_back_wall_only_overlap():
    # Back wall can be within clearance radius from grasp center, but should not
    # fail if it lies behind the diameter line (not in front semicircle).
    scene = _scene_for_wall_intersection_checks((0.34, 0.0))
    hits = front_semicircle_wall_intersections(scene, clearance_radius=0.10)
    assert hits == []


def test_front_semicircle_wall_intersections_detect_side_wall_overlap():
    # Positive side wall intersects the front semicircle at this y-offset.
    scene = _scene_for_wall_intersection_checks((0.34, 0.20))
    hits = front_semicircle_wall_intersections(scene, clearance_radius=0.10)
    assert "side_pos" in hits
