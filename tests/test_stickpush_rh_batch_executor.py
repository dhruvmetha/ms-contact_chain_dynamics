from __future__ import annotations

import numpy as np
import torch

from taskbench.planners.stickpush_rh.executor import StickPushBatchExecutor, StickPushExecutor
from taskbench.planners.stickpush_rh.types import PlannerAction
from taskbench.skills.stick_push import StickPushResult


class _FakeShelfGeom:
    def __init__(self, front_x: float = 0.25):
        self.front_x = float(front_x)


class _FakeRaw:
    def __init__(self, *, num_envs: int, state_dim: int = 2):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.shelf_geom = _FakeShelfGeom()
        self._state = torch.zeros((self.num_envs, state_dim), dtype=torch.float32)
        self.set_state_calls: list[torch.Tensor] = []

    def set_state(self, state: torch.Tensor) -> None:
        arr = torch.as_tensor(state, dtype=torch.float32).clone()
        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        if arr.shape[0] == 1:
            arr = arr.repeat(self.num_envs, 1)
        self._state = arr
        self.set_state_calls.append(arr.clone())

    def get_state(self) -> torch.Tensor:
        return self._state.clone()


class _FakeEnv:
    def __init__(self, *, num_envs: int):
        self.unwrapped = _FakeRaw(num_envs=num_envs)


class _FakePushSkill:
    def __init__(self, raw: _FakeRaw, *, success_map: dict[tuple[float, float], bool]):
        self.raw = raw
        self._success_map = dict(success_map)
        self._call_count = 0

    def __call__(
        self,
        *,
        approach_positions: torch.Tensor,
        approach_quaternions: torch.Tensor,
        entry_positions: torch.Tensor,
        sweep_positions: torch.Tensor,
        retract_positions: torch.Tensor,
        entry_max_steps: int,
        sweep_max_steps: int,
        retract_max_steps: int,
        rest_steps: int,
        staging_max_attempts: int,
        staging_timeout: float,
        step_callback=None,
    ) -> StickPushResult:
        del (
            approach_quaternions,
            entry_positions,
            sweep_positions,
            retract_positions,
            entry_max_steps,
            sweep_max_steps,
            retract_max_steps,
            rest_steps,
            staging_max_attempts,
            staging_timeout,
            step_callback,
        )
        self._call_count += 1
        front_x = float(self.raw.shelf_geom.front_x)
        pos = approach_positions.detach().cpu().numpy()
        backoffs = front_x - pos[:, 0]
        ys = pos[:, 1]

        success = []
        for y, backoff in zip(ys, backoffs):
            key = (round(float(y), 3), round(float(backoff), 2))
            success.append(bool(self._success_map.get(key, False)))
        success_t = torch.tensor(success, dtype=torch.bool, device=approach_positions.device)

        final_dist = torch.where(
            success_t,
            torch.full_like(success_t, 0.001, dtype=torch.float32),
            torch.full_like(success_t, 0.5, dtype=torch.float32),
        )
        orient = torch.zeros_like(final_dist)

        # Encode round id and backoff in the env state so tests can verify
        # lane states are captured from the correct retry round.
        state = self.raw.get_state()
        for idx, backoff in enumerate(backoffs):
            state[idx, 0] = float(self._call_count)
            state[idx, 1] = float(backoff)
        self.raw._state = state.clone()

        return StickPushResult(
            success_mask=success_t,
            staging_success=success_t.clone(),
            entry_final_dist=final_dist.clone(),
            sweep_final_dist=final_dist.clone(),
            retract_final_dist=final_dist.clone(),
            orientation_drift=orient,
            steps_executed=5,
            phase_steps={"stage": 1},
            phase_timing_ms={"stage_plan": 1.0},
        )


def _action(y: float, *, x_approach: float = 0.07) -> PlannerAction:
    approach = np.array([0.25 - x_approach, y, 0.43], dtype=np.float32)
    entry = np.array([0.39, y, 0.43], dtype=np.float32)
    sweep = np.array([0.43, y, 0.43], dtype=np.float32)
    retract = np.array([-0.05, y, 0.43], dtype=np.float32)
    return PlannerAction(
        blocker_name=f"b_{y:+.2f}",
        theta_deg=0.0,
        x_approach=float(x_approach),
        delta_len_idx=0,
        contact_offset=0.02,
        push_len=0.04,
        approach_xyz=approach,
        entry_xyz=entry,
        sweep_xyz=sweep,
        retract_xyz=retract,
        meta={},
    )


def test_batch_executor_lane_fallback_and_round_reset():
    env = _FakeEnv(num_envs=4)
    success_map = {
        (0.0, 0.09): True,  # lane 0 succeeds only after fallback
        (0.1, 0.07): True,  # lane 1 succeeds immediately
        # lane 2 always fails
    }
    push_skill = _FakePushSkill(env.unwrapped, success_map=success_map)
    executor = StickPushBatchExecutor(
        env,
        push_skill,
        approach_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        cuda_device=torch.device("cpu"),
        timing_enabled=True,
        staging_fallback_enabled=True,
        staging_backoff_candidates=(0.05, 0.09),
        staging_fallback_min_insertion_depth=0.01,
    )

    parent_state = torch.tensor([[11.0, 22.0]], dtype=torch.float32)
    outputs = executor.execute_batch_from_parent_state(
        parent_state,
        [_action(0.0), _action(0.1), _action(0.2)],
    )
    assert len(outputs) == 3

    # Each retry round must reset env lanes to parent to avoid cross-round contamination.
    assert len(env.unwrapped.set_state_calls) == 3
    expected_tiled = parent_state.repeat(env.unwrapped.num_envs, 1)
    for call in env.unwrapped.set_state_calls:
        assert torch.allclose(call, expected_tiled)

    info0 = outputs[0].execution.info
    assert info0["staging_success"] is True
    assert info0["staging_attempted_backoffs"] == [0.07, 0.05, 0.09]
    assert info0["staging_retry_count"] == 2
    assert info0["staging_fallback_applied"] is True
    assert info0["x_approach_used"] == 0.09
    # Lane 0 resolves in round 3 and should preserve that state even if other
    # lanes continue retrying.
    assert float(outputs[0].child_state[0, 0].item()) == 3.0

    info1 = outputs[1].execution.info
    assert info1["staging_success"] is True
    assert info1["staging_attempted_backoffs"] == [0.07]
    assert info1["staging_retry_count"] == 0
    assert info1["staging_fallback_applied"] is False
    assert float(outputs[1].child_state[0, 0].item()) == 1.0

    info2 = outputs[2].execution.info
    assert info2["staging_success"] is False
    assert info2["staging_attempted_backoffs"] == [0.07, 0.05, 0.09]
    assert info2["staging_retry_count"] == 2
    assert info2["staging_fallback_applied"] is True
    assert float(outputs[2].child_state[0, 0].item()) == 3.0

    # Timing semantics:
    # - execute_total_wall is per-lane (sum of rounds that lane attempted).
    # - batch_* fields are shared per execute_batch_from_parent_state call.
    t0 = info0["timing_ms"]
    t1 = info1["timing_ms"]
    t2 = info2["timing_ms"]
    assert t0["execute_total_wall"] == t0["attempt_wall_sum"]
    assert t1["execute_total_wall"] == t1["attempt_wall_sum"]
    assert t2["execute_total_wall"] == t2["attempt_wall_sum"]
    assert t0["batch_round_count"] == 3
    assert t1["batch_round_count"] == 3
    assert t2["batch_round_count"] == 3
    assert t0["batch_round_wall_sum"] == t1["batch_round_wall_sum"] == t2["batch_round_wall_sum"]
    assert t0["batch_execute_total_wall"] == t1["batch_execute_total_wall"] == t2["batch_execute_total_wall"]
    assert t0["batch_round_wall_sum"] >= t0["execute_total_wall"]
    assert t1["batch_round_wall_sum"] >= t1["execute_total_wall"]
    assert t2["batch_round_wall_sum"] >= t2["execute_total_wall"]
    assert t1["execute_total_wall"] <= t0["execute_total_wall"]
    assert t1["execute_total_wall"] <= t2["execute_total_wall"]


def test_batch_executor_no_cross_run_contamination():
    env = _FakeEnv(num_envs=2)
    push_skill = _FakePushSkill(
        env.unwrapped,
        success_map={(0.0, 0.07): True, (0.1, 0.07): True},
    )
    executor = StickPushBatchExecutor(
        env,
        push_skill,
        approach_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        cuda_device=torch.device("cpu"),
        timing_enabled=False,
        staging_fallback_enabled=False,
    )

    parent_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    parent_b = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
    _ = executor.execute_batch_from_parent_state(parent_a, [_action(0.0), _action(0.1)])
    _ = executor.execute_batch_from_parent_state(parent_b, [_action(0.0), _action(0.1)])

    # One round per call (fallback disabled), each call must reset from its
    # own parent state.
    assert len(env.unwrapped.set_state_calls) == 2
    assert torch.allclose(env.unwrapped.set_state_calls[0], parent_a.repeat(2, 1))
    assert torch.allclose(env.unwrapped.set_state_calls[1], parent_b.repeat(2, 1))


def test_single_executor_records_safety_try_when_all_backoffs_skipped():
    env = _FakeEnv(num_envs=1)
    push_skill = _FakePushSkill(env.unwrapped, success_map={})
    executor = StickPushExecutor(
        push_skill,
        approach_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        cuda_device=torch.device("cpu"),
        timing_enabled=False,
        staging_fallback_enabled=True,
        staging_backoff_candidates=(0.05, 0.03),
        staging_fallback_min_insertion_depth=0.30,  # skips all normal backoffs
    )

    result = executor.execute(_action(0.0))
    info = result.info

    assert info["staging_attempted_backoffs"] == [0.07]
    assert info["staging_retry_count"] == 0
    assert len(info["staging_attempt_summaries"]) == 4
    assert info["staging_attempt_summaries"][0]["skipped"] is True
    assert info["staging_attempt_summaries"][1]["skipped"] is True
    assert info["staging_attempt_summaries"][2]["skipped"] is True
    assert info["staging_attempt_summaries"][3]["skipped"] is False
    assert info["staging_attempt_summaries"][3]["backoff"] == 0.07
    assert push_skill._call_count == 1
