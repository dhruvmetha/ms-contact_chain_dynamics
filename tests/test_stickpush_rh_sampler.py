from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.config import SamplingConfig
from taskbench.planners.stickpush_rh.geometry import (
    point_in_front_semicircle,
    point_segment_distance_xy,
    within_shelf_xy,
)
from taskbench.planners.stickpush_rh.sampler import StickPushSampler
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


def _scene_for_sampler() -> SceneState:
    target = ObjectState(
        name="target",
        center_xyz=np.array([0.60, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    blocker = ObjectState(
        name="b0",
        center_xyz=np.array([0.50, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=False,
        active=True,
    )
    return SceneState(
        target=target,
        blockers=[blocker],
        all_objects=[target, blocker],
        shelf_front_x=0.20,
        shelf_back_x=0.90,
        shelf_half_w=0.35,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def test_sampler_grid_count_and_z_constant():
    scene = _scene_for_sampler()
    cfg = SamplingConfig(
        clearance_radius=0.12,
        heading_degrees=(0, 45, 90, 135, 180, 225, 270, 315),
        x_approach_values=(0.04, 0.06, 0.08, 0.10),
        delta_len_fracs=(0.25, 0.50, 0.85),
        z_eps=0.001,
        forbid_target_crossing=False,
        include_target_pushes=False,
    )
    sampler = StickPushSampler(cfg)
    actions = sampler.sample_actions(scene)

    assert len(actions) > 0
    z_expected = scene.surface_z + cfg.z_eps
    for action in actions:
        assert np.isclose(action.approach_xyz[2], z_expected)
        assert np.isclose(action.entry_xyz[2], z_expected)
        assert np.isclose(action.sweep_xyz[2], z_expected)
        assert np.isclose(action.retract_xyz[2], z_expected)
        assert within_shelf_xy(action.entry_xyz[:2], scene, margin=cfg.world_min_clearance)
        assert within_shelf_xy(action.sweep_xyz[:2], scene, margin=cfg.world_min_clearance)

    # No multiplicative x_approach branching: each push geometry appears once.
    geom_keys = {
        (
            round(float(a.entry_xyz[0]), 6),
            round(float(a.entry_xyz[1]), 6),
            round(float(a.sweep_xyz[0]), 6),
            round(float(a.sweep_xyz[1]), 6),
        )
        for a in actions
    }
    assert len(geom_keys) == len(actions)

    # Non-target pushes must end outside target front semicircle clearance.
    for action in actions:
        assert not bool(action.meta.get("is_target_push", False))
        assert not point_in_front_semicircle(
            action.sweep_xyz[:2],
            scene.target,
            cfg.clearance_radius,
            scene.open_dir_xy,
        )


def test_sampler_rejects_target_crossing_segments():
    scene = _scene_for_sampler()
    cfg = SamplingConfig(
        clearance_radius=0.12,
        heading_degrees=(0, 45, 90, 135, 180, 225, 270, 315),
        x_approach_values=(0.04,),
        delta_len_fracs=(0.5,),
        forbid_target_crossing=True,
        target_avoid_margin=0.004,
        include_target_pushes=False,
    )
    sampler = StickPushSampler(cfg)
    actions = sampler.sample_actions(scene)
    assert actions

    min_clear = scene.target.radius + scene.blockers[0].radius + cfg.target_avoid_margin
    for action in actions:
        d = point_segment_distance_xy(
            scene.target.center_xyz[:2], action.entry_xyz[:2], action.sweep_xyz[:2]
        )
        assert d > min_clear


def test_sampler_includes_target_push_candidates():
    scene = _scene_for_sampler()
    cfg = SamplingConfig(
        clearance_radius=0.12,
        heading_degrees=(0, 90, 180, 270),
        x_approach_values=(0.05,),
        delta_len_fracs=(0.5,),
        include_target_pushes=True,
        forbid_target_crossing=True,
    )
    sampler = StickPushSampler(cfg)
    actions = sampler.sample_actions(scene)
    assert actions
    target_actions = [a for a in actions if bool(a.meta.get("is_target_push", False))]
    assert target_actions
    for a in target_actions:
        assert a.meta.get("target_push_policy") == "nearest_obstacle_bearing_topk"
        sampled = a.meta.get("target_push_sampled_dir_degs")
        topk = a.meta.get("target_push_topk_dir_degs")
        assert isinstance(sampled, list) and len(sampled) > 0
        assert isinstance(topk, list) and len(topk) > 0
        assert float(a.meta.get("target_push_dir_deg")) in [float(x) for x in topk]
        assert set(float(x) for x in topk).issubset(set(float(x) for x in sampled))


def test_target_push_direction_topk_prunes_branches():
    scene = _scene_for_sampler()
    cfg = SamplingConfig(
        clearance_radius=0.12,
        heading_degrees=(0, 90, 180, 270),
        delta_len_fracs=(0.5,),
        include_target_pushes=True,
        target_push_include_open_face_bias=False,
        target_push_top_k_directions=2,
        forbid_target_crossing=False,
    )
    sampler = StickPushSampler(cfg)
    actions = sampler.sample_actions(scene)

    target_actions = [a for a in actions if bool(a.meta.get("is_target_push", False))]
    assert target_actions
    dir_degs = {round(float(a.meta.get("target_push_dir_deg")), 5) for a in target_actions}
    # With one push length sample, number of target actions should match chosen top-K directions.
    assert len(dir_degs) <= 2
