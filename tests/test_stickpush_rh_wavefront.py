from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig, SamplingConfig
from taskbench.planners.stickpush_rh.insertion_grid import StraightInsertionPlan
from taskbench.planners.stickpush_rh.sampler import StickPushSampler
import taskbench.planners.stickpush_rh.search as search_mod
from taskbench.planners.stickpush_rh.search import StickPushRecedingHorizonSearch
from taskbench.planners.stickpush_rh.types import ObjectState, PlannerAction, SceneState


class _DummyEnv:
    def __init__(self):
        self.unwrapped = object()


def _simple_scene() -> SceneState:
    target = ObjectState(
        name="target",
        center_xyz=np.array([0.62, 0.0, 0.41], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    blocker = ObjectState(
        name="b0",
        center_xyz=np.array([0.55, 0.0, 0.41], dtype=np.float32),
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
        shelf_half_w=0.25,
        surface_z=0.41,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


def test_wavefront_and_feasible_actions_are_cached_for_state_hash():
    cfg = RecedingHorizonConfig(
        sampling=SamplingConfig(
            use_wavefront_insertion_solver=True,
                insertion_grid_resolution=0.003,
                insertion_grid_safety_eps=0.0,
                insertion_stick_length=0.50,
                x_approach_values=(0.07,),
                insertion_approach_backoff=0.07,
                include_target_pushes=False,
            heading_degrees=(0, 45, 90, 135, 180, 225, 270, 315),
            delta_len_fracs=(0.3, 0.6, 0.9),
        )
    )
    planner = StickPushRecedingHorizonSearch(
        _DummyEnv(),
        executor=object(),
        cfg=cfg,
        state_provider=None,
        sampler=StickPushSampler(cfg.sampling),
    )
    scene = _simple_scene()
    state_hash = "same_state_hash"

    actions_1, diag_1 = planner._build_node_actions(
        scene,
        state_hash=state_hash,
        rng=np.random.default_rng(0),
    )
    actions_2, diag_2 = planner._build_node_actions(
        scene,
        state_hash=state_hash,
        rng=np.random.default_rng(1),
    )

    assert diag_1["wavefront_cache_hit"] is False
    assert diag_2["wavefront_cache_hit"] is True
    assert len(planner._wavefront_cache) == 1
    assert len(planner._feasible_action_cache) == 1
    assert len(actions_1) == len(actions_2) > 0


def test_build_node_actions_applies_shifted_entry_and_records_shift_metadata(monkeypatch):
    cfg = RecedingHorizonConfig(
        sampling=SamplingConfig(
            use_wavefront_insertion_solver=True,
            insertion_mode="straight_only",
            insertion_grid_resolution=0.003,
            insertion_stick_length=0.5,
        )
    )

    p_a = np.array([0.15, 0.0, 0.411], dtype=np.float32)
    p1 = np.array([0.40, 0.11, 0.411], dtype=np.float32)
    p2 = np.array([0.55, -0.02, 0.411], dtype=np.float32)
    p_r = np.array([0.10, -0.02, 0.411], dtype=np.float32)
    sampled_action = PlannerAction(
        blocker_name="b0",
        theta_deg=0.0,
        x_approach=0.07,
        delta_len_idx=0,
        contact_offset=0.02,
        push_len=0.15,
        approach_xyz=p_a,
        entry_xyz=p1,
        sweep_xyz=p2,
        retract_xyz=p_r,
        meta={},
    )

    class _OneActionSampler:
        def sample_actions(self, scene: SceneState, *, focus_object_names=None, focus_meta=None):
            return [sampled_action]

    forced_plan = StraightInsertionPlan(
        source_xy=np.array([0.207, -0.01], dtype=np.float32),
        approach_backoff=0.07,
        sources_checked=2,
        sources_total=9,
        entry_xy_used=np.array([0.42, 0.08], dtype=np.float32),
        entry_shift_cells=(1, -1),
        entry_shift_xy=np.array([0.02, -0.03], dtype=np.float32),
    )

    def _fake_solve(grid, scene, p1_xy, sampling_cfg, *, deterministic_seed=None):
        del grid, scene, p1_xy, sampling_cfg, deterministic_seed
        return forced_plan

    monkeypatch.setattr(search_mod, "solve_straight_insertion", _fake_solve)

    planner = StickPushRecedingHorizonSearch(
        _DummyEnv(),
        executor=object(),
        cfg=cfg,
        state_provider=None,
        sampler=_OneActionSampler(),
    )
    scene = _simple_scene()
    actions, diag = planner._build_node_actions(
        scene,
        state_hash="shifted_entry_state",
        rng=np.random.default_rng(0),
    )

    assert diag["num_sampled_actions"] == 1
    assert diag["num_feasible_actions"] == 1
    assert len(actions) == 1
    out = actions[0]
    assert np.allclose(out.entry_xyz[:2], forced_plan.entry_xy_used, atol=1e-6)
    assert np.allclose(out.sweep_xyz[:2], sampled_action.sweep_xyz[:2], atol=1e-6)
    assert np.allclose(np.asarray(out.meta["insertion_entry_xy_used"], dtype=np.float32), forced_plan.entry_xy_used)
    assert out.meta["insertion_entry_shift_cells"] == [1, -1]
    assert np.allclose(np.asarray(out.meta["insertion_entry_shift_xy"], dtype=np.float32), forced_plan.entry_shift_xy)
    assert out.meta["insertion_entry_shift_applied"] is True
