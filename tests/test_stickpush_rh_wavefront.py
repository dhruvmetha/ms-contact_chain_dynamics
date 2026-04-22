from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig, SamplingConfig
from taskbench.planners.stickpush_rh.sampler import StickPushSampler
from taskbench.planners.stickpush_rh.search import StickPushRecedingHorizonSearch
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


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
