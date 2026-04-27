from __future__ import annotations

import math

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig
from taskbench.planners.stickpush_rh.grasp_success import GraspSuccessEvaluator
from taskbench.planners.stickpush_rh.search import StickPushRecedingHorizonSearch
from taskbench.planners.stickpush_rh.sampler import StickPushSampler
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


def _cfg(**kwargs) -> RecedingHorizonConfig:
    base = dict(
        grasp_success_active_label="any",
        grasp_success_templates_straight_deg=(0.0,),
        grasp_success_templates_any_deg=(0.0, 20.0, -20.0),
        grasp_success_grid_resolution=0.002,
        grasp_success_front_outside_offset=0.06,
        grasp_success_finger_thickness=0.012,
        grasp_success_finger_length=0.05,
        grasp_success_jaw_open=0.06,
        grasp_success_jaw_contact=0.036,
        grasp_success_contact_pad_width=0.01,
        grasp_success_contact_pad_depth=0.01,
        grasp_success_collision_epsilon=0.0,
    )
    base.update(kwargs)
    return RecedingHorizonConfig(**base)


def _circle(
    name: str,
    x: float,
    y: float,
    *,
    radius: float = 0.018,
    is_target: bool = False,
) -> ObjectState:
    return ObjectState(
        name=name,
        center_xyz=np.array([x, y, 0.41], dtype=np.float32),
        radius=float(radius),
        is_target=bool(is_target),
        active=True,
        footprint_type="circle",
        footprint_params={"radius": float(radius)},
    )


def _obox(
    name: str,
    x: float,
    y: float,
    *,
    half_extents: tuple[float, float],
    yaw_rad: float,
    is_target: bool = False,
) -> ObjectState:
    return ObjectState(
        name=name,
        center_xyz=np.array([x, y, 0.41], dtype=np.float32),
        radius=float(max(half_extents)),
        is_target=bool(is_target),
        active=True,
        footprint_type="obox",
        footprint_params={
            "half_extents": [float(half_extents[0]), float(half_extents[1])],
            "yaw": float(yaw_rad),
        },
    )


def _scene(
    *,
    target_xy: tuple[float, float] = (0.60, 0.0),
    blockers: list[ObjectState] | None = None,
    shelf_front_x: float = 0.20,
    shelf_back_x: float = 0.90,
    shelf_half_w: float = 0.25,
) -> SceneState:
    target = _circle("target", target_xy[0], target_xy[1], is_target=True)
    blockers = list(blockers or [])
    return SceneState(
        target=target,
        blockers=blockers,
        all_objects=[target, *blockers],
        shelf_front_x=float(shelf_front_x),
        shelf_back_x=float(shelf_back_x),
        shelf_half_w=float(shelf_half_w),
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def _nearest_idx(values: np.ndarray, query: float) -> int:
    return int(np.argmin(np.abs(values - float(query))))


def _mask_at_point(occ, mask: np.ndarray, x: float, y: float) -> bool:
    xs = np.asarray(occ.xx[0, :], dtype=np.float64)
    ys = np.asarray(occ.yy[:, 0], dtype=np.float64)
    ix = _nearest_idx(xs, x)
    iy = _nearest_idx(ys, y)
    return bool(mask[iy, ix])


def _goal_diag_from_eval(result) -> dict:
    return {
        "primary_blocking_objects": list(result.primary_blocking_objects),
        "primary_min_margin": float(result.primary_min_margin),
        "primary_penetration_sum": float(result.primary_penetration_sum),
        "active_success_raw": bool(result.active_success),
        "target_shift_ok": True,
    }


class _DummyEnv:
    def __init__(self):
        self.unwrapped = object()


class _NoopProvider:
    pass


def test_clear_scene_passes():
    evaluator = GraspSuccessEvaluator(_cfg())
    result = evaluator.evaluate(_scene())

    assert result.graspable_straight
    assert result.graspable_any
    assert result.active_success
    assert result.primary_blocking_objects == []


def test_insertion_collision_reports_blocker_name():
    scene = _scene(blockers=[_circle("b_insert", 0.40, 0.036, radius=0.012)])
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
        )
    )
    result = evaluator.evaluate(scene)

    assert not result.active_success
    assert "b_insert" in result.primary_blocking_objects
    assert result.primary_failing_template is not None
    assert "collision" in str(result.primary_failing_template.fail_reason)


def test_closure_collision_reports_blocker_name():
    scene = _scene(blockers=[_circle("b_close", 0.60, 0.013, radius=0.007)])
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
        )
    )
    result = evaluator.evaluate(scene)

    assert not result.active_success
    assert "b_close" in result.primary_blocking_objects
    assert result.primary_failing_template is not None
    assert "collision" in str(result.primary_failing_template.fail_reason)


def test_wall_collision_fails_without_movable_blockers():
    scene = _scene(shelf_half_w=0.028)
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
        )
    )
    result = evaluator.evaluate(scene)

    assert not result.active_success
    assert result.primary_blocking_objects == []
    assert result.primary_failing_template is not None
    assert result.primary_failing_template.wall_blocked
    assert result.primary_penetration_sum > 0.0


def test_missing_target_contact_fails_with_contact_reason():
    scene = _scene()
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
            grasp_success_jaw_contact=0.08,
            grasp_success_contact_pad_width=0.006,
            grasp_success_contact_pad_depth=0.006,
        )
    )
    result = evaluator.evaluate(scene)

    assert not result.active_success
    assert result.primary_blocking_objects == []
    assert result.primary_failing_template is not None
    assert str(result.primary_failing_template.fail_reason).startswith("missing_")
    assert result.primary_failing_template.contact_deficit >= 1.0


def test_collision_epsilon_rejects_tiny_near_miss():
    # Blocker sits just outside insertion corridor by a narrow gap.
    scene = _scene(blockers=[_circle("b_eps", 0.30, 0.044, radius=0.002)])
    low_eps = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
            grasp_success_grid_resolution=0.001,
            grasp_success_collision_epsilon=0.0,
        )
    )
    high_eps = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(0.0,),
            grasp_success_grid_resolution=0.001,
            grasp_success_collision_epsilon=0.004,
        )
    )

    low = low_eps.evaluate(scene)
    high = high_eps.evaluate(scene)

    assert low.active_success
    assert not high.active_success
    assert "b_eps" in high.primary_blocking_objects


def test_circle_occupancy_rasterization_contains_center_and_excludes_far_point():
    blocker = _circle("b_circle", 0.50, 0.05, radius=0.02)
    scene = _scene(blockers=[blocker])
    evaluator = GraspSuccessEvaluator(_cfg(grasp_success_grid_resolution=0.002))
    occ = evaluator._build_occupancy(scene, state_hash="circle_occ")

    mask = occ.object_masks["b_circle"]
    assert _mask_at_point(occ, mask, 0.50, 0.05)
    assert not _mask_at_point(occ, mask, 0.58, 0.05)


def test_obox_occupancy_changes_with_yaw():
    blocker_0 = _obox("b_box", 0.50, 0.00, half_extents=(0.03, 0.01), yaw_rad=0.0)
    blocker_90 = _obox("b_box", 0.50, 0.00, half_extents=(0.03, 0.01), yaw_rad=math.pi / 2.0)

    evaluator = GraspSuccessEvaluator(_cfg(grasp_success_grid_resolution=0.002))
    occ0 = evaluator._build_occupancy(_scene(blockers=[blocker_0]), state_hash="obox_yaw0")
    occ90 = evaluator._build_occupancy(_scene(blockers=[blocker_90]), state_hash="obox_yaw90")

    mask0 = occ0.object_masks["b_box"]
    mask90 = occ90.object_masks["b_box"]
    probe_x, probe_y = 0.525, 0.0

    assert _mask_at_point(occ0, mask0, probe_x, probe_y)
    assert not _mask_at_point(occ90, mask90, probe_x, probe_y)


def test_mixed_shape_scene_occupancy_consistency():
    circle = _circle("b_circle", 0.47, 0.06, radius=0.015)
    box = _obox("b_box", 0.54, -0.05, half_extents=(0.02, 0.01), yaw_rad=0.35)
    scene = _scene(blockers=[circle, box])

    evaluator = GraspSuccessEvaluator(_cfg(grasp_success_grid_resolution=0.002))
    occ = evaluator._build_occupancy(scene, state_hash="mixed_shapes")

    assert "b_circle" in occ.object_masks
    assert "b_box" in occ.object_masks
    c_count = int(np.count_nonzero(occ.object_masks["b_circle"]))
    b_count = int(np.count_nonzero(occ.object_masks["b_box"]))
    union_count = int(np.count_nonzero(occ.occ_non_target))
    assert c_count > 0
    assert b_count > 0
    assert union_count >= max(c_count, b_count)


def test_graspable_any_includes_straight_templates():
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(),
        )
    )
    result = evaluator.evaluate(_scene())

    assert result.graspable_straight
    assert result.graspable_any


def test_diagonal_only_feasible_sets_straight_false_any_true():
    scene = _scene(blockers=[_circle("b_straight", 0.40, 0.03, radius=0.012)])
    evaluator = GraspSuccessEvaluator(
        _cfg(
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(8.0,),
        )
    )
    result = evaluator.evaluate(scene)

    assert not result.graspable_straight
    assert result.graspable_any


def test_active_label_switch_changes_success():
    scene = _scene(blockers=[_circle("b_straight", 0.40, 0.03, radius=0.012)])

    any_eval = GraspSuccessEvaluator(
        _cfg(
            grasp_success_active_label="any",
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(8.0,),
        )
    ).evaluate(scene)
    straight_eval = GraspSuccessEvaluator(
        _cfg(
            grasp_success_active_label="straight",
            grasp_success_templates_straight_deg=(0.0,),
            grasp_success_templates_any_deg=(8.0,),
        )
    ).evaluate(scene)

    assert any_eval.active_success
    assert not straight_eval.active_success


def test_metrics_follow_primary_blockers_and_improve_when_blocker_removed():
    cfg = _cfg(
        grasp_success_templates_straight_deg=(0.0,),
        grasp_success_templates_any_deg=(0.0,),
    )
    evaluator = GraspSuccessEvaluator(cfg)
    blocked_scene = _scene(blockers=[_circle("b0", 0.40, 0.036, radius=0.012)])
    clear_scene = _scene(blockers=[])

    blocked = evaluator.evaluate(blocked_scene)
    clear = evaluator.evaluate(clear_scene)

    planner = StickPushRecedingHorizonSearch(
        _DummyEnv(),
        executor=object(),
        cfg=cfg,
        state_provider=_NoopProvider(),
        sampler=StickPushSampler(cfg.sampling),
    )

    blocked_metrics = planner._metrics_from_goal(goal_checks=_goal_diag_from_eval(blocked), pushes_used=1)
    clear_metrics = planner._metrics_from_goal(goal_checks=_goal_diag_from_eval(clear), pushes_used=1)

    assert blocked_metrics.blockers >= 1
    assert blocked_metrics.deficit > 0.0
    assert not blocked_metrics.solved

    assert clear_metrics.blockers == 0
    assert clear_metrics.deficit == 0.0
    assert clear_metrics.solved

    # Ranking terms should improve when blockers are removed.
    assert clear_metrics.blockers <= blocked_metrics.blockers
    assert clear_metrics.deficit <= blocked_metrics.deficit
    assert clear_metrics.min_margin >= blocked_metrics.min_margin
