from __future__ import annotations

import csv
import json
import numpy as np
import torch

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig, SamplingConfig
from taskbench.planners.stickpush_rh.executor import ExecutionResult
from taskbench.planners.stickpush_rh.search import StickPushRecedingHorizonSearch
from taskbench.planners.stickpush_rh.types import ObjectState, PlannerAction, SceneState


class _FakeRaw:
    def __init__(self):
        self._state = 0

    def get_state(self):
        return torch.tensor([float(self._state)], dtype=torch.float32)

    def set_state(self, state):
        self._state = int(round(float(state[0].item())))


class _FakeEnv:
    def __init__(self):
        self.unwrapped = _FakeRaw()

    def render(self):
        value = 80 if self.unwrapped._state == 0 else 180
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[:, :] = np.array([value, value, value], dtype=np.uint8)
        return img


def _make_scene(state_id: int) -> SceneState:
    target = ObjectState(
        name=f"target_{state_id}",
        center_xyz=np.array([0.55, 0.0, 0.40], dtype=np.float32),
        radius=0.018,
        is_target=True,
        active=True,
    )
    if state_id == 0:
        blockers = [
            ObjectState(
                name="b0",
                center_xyz=np.array([0.50, 0.0, 0.40], dtype=np.float32),
                radius=0.018,
                is_target=False,
                active=True,
            )
        ]
    else:
        blockers = [
            ObjectState(
                name="b0",
                center_xyz=np.array([0.67, 0.0, 0.40], dtype=np.float32),
                radius=0.018,
                is_target=False,
                active=True,
            )
        ]
    return SceneState(
        target=target,
        blockers=blockers,
        all_objects=[target, *blockers],
        shelf_front_x=0.2,
        shelf_back_x=0.8,
        shelf_half_w=0.25,
        surface_z=0.40,
        open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
    )


class _FakeProvider:
    def get_scene_state(self, env):
        return _make_scene(env.unwrapped._state)


class _FakeSampler:
    def sample_actions(
        self,
        scene: SceneState,
        *,
        focus_object_names: list[str] | None = None,
        focus_meta: dict | None = None,
    ):
        if scene.target.name.endswith("_1"):
            return []
        p_a = np.array([0.15, 0.0, scene.surface_z + 0.001], dtype=np.float32)
        p1 = np.array([0.50, 0.0, scene.surface_z + 0.001], dtype=np.float32)
        p2 = np.array([0.65, 0.0, scene.surface_z + 0.001], dtype=np.float32)
        p_r = np.array([-0.1, 0.0, scene.surface_z + 0.001], dtype=np.float32)
        return [
            PlannerAction(
                blocker_name="b0",
                theta_deg=0.0,
                x_approach=0.05,
                delta_len_idx=0,
                contact_offset=0.02,
                push_len=0.15,
                approach_xyz=p_a,
                entry_xyz=p1,
                sweep_xyz=p2,
                retract_xyz=p_r,
                meta={"next_state": 1},
            )
        ]


class _FakeExecutor:
    def __init__(self, env):
        self.env = env

    def execute(self, action, *, step_callback=None):
        self.env.unwrapped._state = int(action.meta.get("next_state", self.env.unwrapped._state))
        return ExecutionResult(success=True, steps_executed=1, stick_result=None, info={})


def test_search_smoke_writes_artifacts(tmp_path):
    env = _FakeEnv()
    cfg = RecedingHorizonConfig(
        max_executions=5,
        max_depth=5,
        artifact_root=str(tmp_path / "artifacts"),
        sampling=SamplingConfig(use_wavefront_insertion_solver=False),
    )
    planner = StickPushRecedingHorizonSearch(
        env,
        executor=_FakeExecutor(env),
        cfg=cfg,
        state_provider=_FakeProvider(),
        sampler=_FakeSampler(),
    )
    result = planner.run(seed=7)

    out_dir = tmp_path / "artifacts"
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert run_dirs
    latest = sorted(run_dirs)[-1]

    assert result.success
    assert (latest / "summary.json").exists()
    assert (latest / "final_plan.json").exists()

    with (latest / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "goal_labels" in summary
    assert "root_active_success" in summary["goal_labels"]
    assert summary["root_metrics"]["solved"] == summary["goal_labels"]["root_active_success"]

    expansions = int(summary.get("expansions", 0))
    if expansions > 0:
        assert (latest / "exp_000001" / "metrics.json").exists()
        assert (latest / "exp_000001" / "selected_only.png").exists()
        assert (latest / "exp_000001" / "frontier_pre_select.csv").exists()
        assert (latest / "exp_000001" / "frontier_post_update.csv").exists()

        with (latest / "exp_000001" / "frontier_pre_select.csv").open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert "lex_rank" in rows[0]
        assert "ucb_score" in rows[0]
        assert "snapshot_kind" in rows[0]
        assert any(str(r.get("is_selected", "")).lower() in {"true", "1"} for r in rows)
        assert all(r.get("snapshot_kind") == "pre_select" for r in rows)

        with (latest / "exp_000001" / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["selection"]["source"] == "ucb_select"
