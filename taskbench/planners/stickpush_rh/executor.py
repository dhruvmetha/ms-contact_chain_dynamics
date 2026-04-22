"""Execution wrapper around the existing StickPush skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import time

import numpy as np
import torch

from taskbench.planners.stickpush_rh.types import PlannerAction
from taskbench.skills.stick_push import StickPush, StickPushResult


@dataclass
class ExecutionResult:
    success: bool
    steps_executed: int
    stick_result: StickPushResult
    info: dict = field(default_factory=dict)


@dataclass
class BatchExecutionResult:
    execution: ExecutionResult
    child_state: torch.Tensor


def _ordered_backoffs(primary: float, candidates: tuple[float, ...]) -> list[float]:
    ordered: list[float] = []

    def _add(v: float | None):
        if v is None:
            return
        vv = float(v)
        if vv <= 0:
            return
        if any(abs(vv - x) < 1e-6 for x in ordered):
            return
        ordered.append(vv)

    _add(primary)
    for v in candidates:
        _add(v)
    return ordered


def _clone_action(action: PlannerAction) -> PlannerAction:
    return PlannerAction(
        blocker_name=str(action.blocker_name),
        theta_deg=float(action.theta_deg),
        x_approach=float(action.x_approach),
        delta_len_idx=int(action.delta_len_idx),
        contact_offset=float(action.contact_offset),
        push_len=float(action.push_len),
        approach_xyz=np.asarray(action.approach_xyz, dtype=np.float32).copy(),
        entry_xyz=np.asarray(action.entry_xyz, dtype=np.float32).copy(),
        sweep_xyz=np.asarray(action.sweep_xyz, dtype=np.float32).copy(),
        retract_xyz=np.asarray(action.retract_xyz, dtype=np.float32).copy(),
        meta=dict(action.meta),
    )


def _clone_action_with_backoff(action: PlannerAction, *, backoff: float, front_x: float) -> PlannerAction:
    cloned = _clone_action(action)
    cloned.x_approach = float(backoff)
    cloned.approach_xyz[0] = float(front_x - float(backoff))
    return cloned


class StickPushExecutor:
    """Bridges planner actions to StickPush waypoints."""

    def __init__(
        self,
        push_skill: StickPush,
        *,
        approach_quaternion_wxyz: list[float],
        cuda_device: torch.device | None = None,
        timing_enabled: bool = False,
        entry_max_steps: int = 200,
        sweep_max_steps: int = 200,
        retract_max_steps: int = 100,
        rest_steps: int = 60,
        convergence_threshold: float = 0.01,
        staging_max_attempts: int = 2,
        staging_timeout: float = 3.0,
        staging_fallback_enabled: bool = True,
        staging_backoff_candidates: tuple[float, ...] = (0.05, 0.03, 0.09, 0.11),
        staging_fallback_min_insertion_depth: float = 0.015,
    ):
        self.push_skill = push_skill
        self.q = torch.tensor([approach_quaternion_wxyz], dtype=torch.float32)
        self.cuda_device = cuda_device or torch.device("cuda:0")
        self.timing_enabled = bool(timing_enabled)
        self.entry_max_steps = int(entry_max_steps)
        self.sweep_max_steps = int(sweep_max_steps)
        self.retract_max_steps = int(retract_max_steps)
        self.rest_steps = int(rest_steps)
        self.convergence_threshold = float(convergence_threshold)
        self.staging_max_attempts = int(staging_max_attempts)
        self.staging_timeout = float(staging_timeout)
        self.staging_fallback_enabled = bool(staging_fallback_enabled)
        self.staging_backoff_candidates = tuple(float(v) for v in staging_backoff_candidates)
        self.staging_fallback_min_insertion_depth = float(staging_fallback_min_insertion_depth)

    @staticmethod
    def _build_result_info(
        result: StickPushResult,
        convergence_threshold: float,
        *,
        include_timing: bool,
    ) -> dict:
        entry_final_dist = float(result.entry_final_dist[0].item())
        sweep_final_dist = float(result.sweep_final_dist[0].item())
        retract_final_dist = float(result.retract_final_dist[0].item())
        orientation_drift = float(result.orientation_drift[0].item())
        entry_converged = entry_final_dist <= convergence_threshold
        sweep_converged = sweep_final_dist <= convergence_threshold
        info = {
            "entry_final_dist": entry_final_dist,
            "sweep_final_dist": sweep_final_dist,
            "retract_final_dist": retract_final_dist,
            "orientation_drift": orientation_drift,
            "entry_converged": bool(entry_converged),
            "sweep_converged": bool(sweep_converged),
            "staging_success": bool(result.staging_success[0].item()),
            "convergence_threshold": float(convergence_threshold),
        }
        if include_timing:
            info["phase_steps"] = {str(k): int(v) for k, v in dict(result.phase_steps).items()}
            info["phase_timing_ms"] = {str(k): float(v) for k, v in dict(result.phase_timing_ms).items()}
        return info

    def _candidate_backoffs(self, action: PlannerAction) -> list[float]:
        return _ordered_backoffs(float(action.x_approach), self.staging_backoff_candidates)

    def execute(
        self,
        action: PlannerAction,
        *,
        step_callback: Callable | None = None,
    ) -> ExecutionResult:
        t_execute_total_ns = time.perf_counter_ns() if self.timing_enabled else 0
        raw = self.push_skill.raw
        front_x = float(raw.shelf_geom.front_x)

        approach_base = np.asarray(action.approach_xyz, dtype=np.float32).copy()
        entry = np.asarray(action.entry_xyz, dtype=np.float32).copy()
        sweep = np.asarray(action.sweep_xyz, dtype=np.float32).copy()
        retract = np.asarray(action.retract_xyz, dtype=np.float32).copy()

        attempted_backoffs: list[float] = []
        attempt_summaries: list[dict] = []
        attempt_timing_ms: list[dict] = []
        chosen_backoff = float(action.x_approach)
        chosen_approach = approach_base.copy()
        chosen_result: StickPushResult | None = None
        chosen_info: dict | None = None

        backoffs = [float(action.x_approach)]
        if self.staging_fallback_enabled:
            backoffs = self._candidate_backoffs(action)

        q = self.q.to(self.cuda_device)
        entry_t = torch.from_numpy(entry).to(self.cuda_device, dtype=torch.float32).unsqueeze(0)
        sweep_t = torch.from_numpy(sweep).to(self.cuda_device, dtype=torch.float32).unsqueeze(0)
        retract_t = torch.from_numpy(retract).to(self.cuda_device, dtype=torch.float32).unsqueeze(0)

        for backoff in backoffs:
            approach = approach_base.copy()
            approach[0] = float(front_x - backoff)
            insertion_depth = float(entry[0] - approach[0])
            if insertion_depth < self.staging_fallback_min_insertion_depth:
                attempt_summaries.append(
                    {
                        "backoff": float(backoff),
                        "skipped": True,
                        "reason": "min_insertion_depth",
                        "insertion_depth": insertion_depth,
                    }
                )
                continue

            attempted_backoffs.append(float(backoff))
            approach_t = torch.from_numpy(approach).to(self.cuda_device, dtype=torch.float32).unsqueeze(0)
            t_attempt_ns = time.perf_counter_ns() if self.timing_enabled else 0
            result = self.push_skill(
                approach_positions=approach_t,
                approach_quaternions=q,
                entry_positions=entry_t,
                sweep_positions=sweep_t,
                retract_positions=retract_t,
                entry_max_steps=self.entry_max_steps,
                sweep_max_steps=self.sweep_max_steps,
                retract_max_steps=self.retract_max_steps,
                rest_steps=self.rest_steps,
                staging_max_attempts=self.staging_max_attempts,
                staging_timeout=self.staging_timeout,
                step_callback=step_callback,
            )
            attempt_wall_ms = (
                float(time.perf_counter_ns() - int(t_attempt_ns)) / 1e6 if self.timing_enabled else 0.0
            )
            info = self._build_result_info(
                result,
                self.convergence_threshold,
                include_timing=self.timing_enabled,
            )
            info["insertion_depth"] = insertion_depth
            if self.timing_enabled:
                attempt_timing_ms.append(
                    {
                        "backoff": float(backoff),
                        "attempt_wall_ms": float(attempt_wall_ms),
                        "phase_timing_ms": dict(info.get("phase_timing_ms", {})),
                    }
                )
            attempt_row = {
                "backoff": float(backoff),
                "skipped": False,
                "staging_success": bool(info["staging_success"]),
                "entry_final_dist": float(info["entry_final_dist"]),
                "sweep_final_dist": float(info["sweep_final_dist"]),
            }
            if self.timing_enabled:
                attempt_row["attempt_wall_ms"] = float(attempt_wall_ms)
                attempt_row["phase_timing_ms"] = dict(info.get("phase_timing_ms", {}))
            attempt_summaries.append(attempt_row)

            chosen_backoff = float(backoff)
            chosen_approach = approach
            chosen_result = result
            chosen_info = info
            if bool(info["staging_success"]):
                break

        if chosen_result is None or chosen_info is None:
            # Fallback safety; should not happen unless all candidates were skipped.
            approach_t = torch.from_numpy(approach_base).to(self.cuda_device, dtype=torch.float32).unsqueeze(0)
            t_attempt_ns = time.perf_counter_ns() if self.timing_enabled else 0
            chosen_result = self.push_skill(
                approach_positions=approach_t,
                approach_quaternions=q,
                entry_positions=entry_t,
                sweep_positions=sweep_t,
                retract_positions=retract_t,
                entry_max_steps=self.entry_max_steps,
                sweep_max_steps=self.sweep_max_steps,
                retract_max_steps=self.retract_max_steps,
                rest_steps=self.rest_steps,
                staging_max_attempts=self.staging_max_attempts,
                staging_timeout=self.staging_timeout,
                step_callback=step_callback,
            )
            attempt_wall_ms = (
                float(time.perf_counter_ns() - int(t_attempt_ns)) / 1e6 if self.timing_enabled else 0.0
            )
            chosen_info = self._build_result_info(
                chosen_result,
                self.convergence_threshold,
                include_timing=self.timing_enabled,
            )
            chosen_backoff = float(action.x_approach)
            chosen_approach = approach_base.copy()
            insertion_depth = float(entry[0] - chosen_approach[0])
            chosen_info["insertion_depth"] = insertion_depth
            attempted_backoffs.append(float(chosen_backoff))
            attempt_row = {
                "backoff": float(chosen_backoff),
                "skipped": False,
                "staging_success": bool(chosen_info["staging_success"]),
                "entry_final_dist": float(chosen_info["entry_final_dist"]),
                "sweep_final_dist": float(chosen_info["sweep_final_dist"]),
            }
            if self.timing_enabled:
                attempt_row["attempt_wall_ms"] = float(attempt_wall_ms)
                attempt_row["phase_timing_ms"] = dict(chosen_info.get("phase_timing_ms", {}))
            attempt_summaries.append(attempt_row)
            if self.timing_enabled:
                attempt_timing_ms.append(
                    {
                        "backoff": float(chosen_backoff),
                        "attempt_wall_ms": float(attempt_wall_ms),
                        "phase_timing_ms": dict(chosen_info.get("phase_timing_ms", {})),
                    }
                )

        chosen_info["staging_attempted_backoffs"] = attempted_backoffs
        chosen_info["staging_attempt_summaries"] = attempt_summaries
        if self.timing_enabled:
            chosen_info["staging_attempt_timing_ms"] = attempt_timing_ms
        chosen_info["staging_used_backoff"] = float(chosen_backoff)
        chosen_info["staging_retry_count"] = max(0, len(attempted_backoffs) - 1)
        chosen_info["staging_fallback_applied"] = bool(abs(chosen_backoff - float(action.x_approach)) > 1e-6)
        chosen_info["approach_xyz_used"] = np.asarray(chosen_approach, dtype=np.float32).tolist()
        chosen_info["x_approach_used"] = float(chosen_backoff)
        chosen_info["entry_xyz_used"] = entry.tolist()
        chosen_info["sweep_xyz_used"] = sweep.tolist()
        chosen_info["retract_xyz_used"] = retract.tolist()
        if self.timing_enabled:
            chosen_info["timing_ms"] = {
                "execute_total_wall": float(time.perf_counter_ns() - int(t_execute_total_ns)) / 1e6,
                "attempt_wall_sum": float(
                    sum(float(a.get("attempt_wall_ms", 0.0)) for a in attempt_timing_ms)
                ),
            }

        return ExecutionResult(
            success=bool(chosen_result.success_mask[0].item() and bool(chosen_info["entry_converged"])),
            steps_executed=int(chosen_result.steps_executed),
            stick_result=chosen_result,
            info=chosen_info,
        )


class StickPushBatchExecutor:
    """Execute multiple planner actions in parallel from one parent state."""

    def __init__(
        self,
        env,
        push_skill: StickPush,
        *,
        approach_quaternion_wxyz: list[float],
        cuda_device: torch.device | None = None,
        timing_enabled: bool = False,
        entry_max_steps: int = 200,
        sweep_max_steps: int = 200,
        retract_max_steps: int = 100,
        rest_steps: int = 60,
        convergence_threshold: float = 0.01,
        staging_max_attempts: int = 2,
        staging_timeout: float = 3.0,
        staging_fallback_enabled: bool = True,
        staging_backoff_candidates: tuple[float, ...] = (0.05, 0.03, 0.09, 0.11),
        staging_fallback_min_insertion_depth: float = 0.015,
    ):
        self.env = env
        self.raw = env.unwrapped
        self.push_skill = push_skill
        self.q = torch.tensor([approach_quaternion_wxyz], dtype=torch.float32)
        self.cuda_device = cuda_device or torch.device("cuda:0")
        self.timing_enabled = bool(timing_enabled)
        self.entry_max_steps = int(entry_max_steps)
        self.sweep_max_steps = int(sweep_max_steps)
        self.retract_max_steps = int(retract_max_steps)
        self.rest_steps = int(rest_steps)
        self.convergence_threshold = float(convergence_threshold)
        self.staging_max_attempts = int(staging_max_attempts)
        self.staging_timeout = float(staging_timeout)
        self.staging_fallback_enabled = bool(staging_fallback_enabled)
        self.staging_backoff_candidates = tuple(float(v) for v in staging_backoff_candidates)
        self.staging_fallback_min_insertion_depth = float(staging_fallback_min_insertion_depth)

    def _candidate_backoffs(self, action: PlannerAction) -> list[float]:
        if not self.staging_fallback_enabled:
            return [float(action.x_approach)]
        return _ordered_backoffs(float(action.x_approach), self.staging_backoff_candidates)

    def _run_attempt_round(
        self,
        *,
        parent_dev: torch.Tensor,
        parent_state_2d: torch.Tensor,
        actions: list[PlannerAction],
        batch_size: int,
    ) -> tuple[StickPushResult, torch.Tensor, float]:
        # Always reset to the same parent state before each round to avoid
        # cross-round contamination from previous failed attempts.
        tiled_state = parent_dev.repeat(batch_size, 1)
        self.raw.set_state(tiled_state)

        active = int(len(actions))
        ref_action = actions[0]
        approach_np = np.stack([np.asarray(a.approach_xyz, dtype=np.float32) for a in actions], axis=0)
        entry_np = np.stack([np.asarray(a.entry_xyz, dtype=np.float32) for a in actions], axis=0)
        sweep_np = np.stack([np.asarray(a.sweep_xyz, dtype=np.float32) for a in actions], axis=0)
        retract_np = np.stack([np.asarray(a.retract_xyz, dtype=np.float32) for a in actions], axis=0)

        if active < batch_size:
            n_pad = int(batch_size - active)
            approach_pad = np.repeat(np.asarray(ref_action.approach_xyz, dtype=np.float32)[None, :], n_pad, axis=0)
            entry_pad = np.repeat(np.asarray(ref_action.entry_xyz, dtype=np.float32)[None, :], n_pad, axis=0)
            sweep_pad = np.repeat(np.asarray(ref_action.sweep_xyz, dtype=np.float32)[None, :], n_pad, axis=0)
            retract_pad = np.repeat(np.asarray(ref_action.retract_xyz, dtype=np.float32)[None, :], n_pad, axis=0)
            approach_np = np.concatenate([approach_np, approach_pad], axis=0)
            entry_np = np.concatenate([entry_np, entry_pad], axis=0)
            sweep_np = np.concatenate([sweep_np, sweep_pad], axis=0)
            retract_np = np.concatenate([retract_np, retract_pad], axis=0)

        q = self.q.to(self.cuda_device, dtype=torch.float32).expand(batch_size, -1).contiguous()
        approach_t = torch.from_numpy(approach_np).to(self.cuda_device, dtype=torch.float32)
        entry_t = torch.from_numpy(entry_np).to(self.cuda_device, dtype=torch.float32)
        sweep_t = torch.from_numpy(sweep_np).to(self.cuda_device, dtype=torch.float32)
        retract_t = torch.from_numpy(retract_np).to(self.cuda_device, dtype=torch.float32)

        t_exec_ns = time.perf_counter_ns() if self.timing_enabled else 0
        result = self.push_skill(
            approach_positions=approach_t,
            approach_quaternions=q,
            entry_positions=entry_t,
            sweep_positions=sweep_t,
            retract_positions=retract_t,
            entry_max_steps=self.entry_max_steps,
            sweep_max_steps=self.sweep_max_steps,
            retract_max_steps=self.retract_max_steps,
            rest_steps=self.rest_steps,
            staging_max_attempts=self.staging_max_attempts,
            staging_timeout=self.staging_timeout,
        )
        exec_wall_ms = (
            float(time.perf_counter_ns() - int(t_exec_ns)) / 1e6 if self.timing_enabled else 0.0
        )
        final_states = self.raw.get_state().clone().to(device=parent_state_2d.device)
        return result, final_states, float(exec_wall_ms)

    @staticmethod
    def _build_result_info_at_index(
        result: StickPushResult,
        idx: int,
        convergence_threshold: float,
        *,
        include_timing: bool,
    ) -> dict:
        entry_final_dist = float(result.entry_final_dist[idx].item())
        sweep_final_dist = float(result.sweep_final_dist[idx].item())
        retract_final_dist = float(result.retract_final_dist[idx].item())
        orientation_drift = float(result.orientation_drift[idx].item())
        entry_converged = entry_final_dist <= convergence_threshold
        sweep_converged = sweep_final_dist <= convergence_threshold
        info = {
            "entry_final_dist": entry_final_dist,
            "sweep_final_dist": sweep_final_dist,
            "retract_final_dist": retract_final_dist,
            "orientation_drift": orientation_drift,
            "entry_converged": bool(entry_converged),
            "sweep_converged": bool(sweep_converged),
            "staging_success": bool(result.staging_success[idx].item()),
            "convergence_threshold": float(convergence_threshold),
        }
        if include_timing:
            info["phase_steps"] = {str(k): int(v) for k, v in dict(result.phase_steps).items()}
            info["phase_timing_ms"] = {str(k): float(v) for k, v in dict(result.phase_timing_ms).items()}
        return info

    def execute_batch_from_parent_state(
        self,
        parent_state: torch.Tensor,
        actions: list[PlannerAction],
    ) -> list[BatchExecutionResult]:
        if not actions:
            return []
        batch_size = int(self.raw.num_envs)
        if len(actions) > batch_size:
            raise ValueError(f"Got {len(actions)} actions but batch env has {batch_size} lanes")

        parent_state_2d = parent_state
        if parent_state_2d.ndim == 1:
            parent_state_2d = parent_state_2d.unsqueeze(0)

        parent_dev = parent_state_2d.to(device=self.raw.device, dtype=torch.float32)
        t_batch_total_ns = time.perf_counter_ns() if self.timing_enabled else 0
        batch_round_wall_sum_ms = 0.0
        batch_round_count = 0
        front_x = float(self.raw.shelf_geom.front_x)

        lane_records: list[dict] = []
        for action in actions:
            lane_records.append(
                {
                    "base_action": _clone_action(action),
                    "next_backoffs": self._candidate_backoffs(action),
                    "backoff_cursor": 0,
                    "attempted_backoffs": [],
                    "attempt_summaries": [],
                    "attempt_timing_ms": [],
                    "chosen_result": None,
                    "chosen_slot": None,
                    "chosen_info": None,
                    "chosen_action": None,
                    "chosen_state": None,
                    "attempt_wall_sum": 0.0,
                    "resolved": False,
                }
            )

        unresolved_lane_ids = set(range(len(actions)))
        while unresolved_lane_ids:
            round_lane_ids: list[int] = []
            round_actions: list[PlannerAction] = []

            for lane_id in sorted(unresolved_lane_ids):
                rec = lane_records[lane_id]
                selected_action: PlannerAction | None = None

                while rec["backoff_cursor"] < len(rec["next_backoffs"]):
                    backoff = float(rec["next_backoffs"][rec["backoff_cursor"]])
                    rec["backoff_cursor"] += 1
                    candidate = _clone_action_with_backoff(
                        rec["base_action"],
                        backoff=backoff,
                        front_x=front_x,
                    )
                    insertion_depth = float(candidate.entry_xyz[0] - candidate.approach_xyz[0])
                    if insertion_depth < self.staging_fallback_min_insertion_depth:
                        rec["attempt_summaries"].append(
                            {
                                "backoff": float(backoff),
                                "skipped": True,
                                "reason": "min_insertion_depth",
                                "insertion_depth": insertion_depth,
                            }
                        )
                        continue
                    selected_action = candidate
                    break

                if selected_action is None:
                    rec["resolved"] = True
                    continue

                round_lane_ids.append(int(lane_id))
                round_actions.append(selected_action)

            if not round_actions:
                break

            result, final_states, exec_wall_ms = self._run_attempt_round(
                parent_dev=parent_dev,
                parent_state_2d=parent_state_2d,
                actions=round_actions,
                batch_size=batch_size,
            )
            batch_round_count += 1
            batch_round_wall_sum_ms += float(exec_wall_ms)

            for slot, lane_id in enumerate(round_lane_ids):
                rec = lane_records[lane_id]
                action = round_actions[slot]
                info = self._build_result_info_at_index(
                    result,
                    slot,
                    self.convergence_threshold,
                    include_timing=self.timing_enabled,
                )
                insertion_depth = float(action.entry_xyz[0] - action.approach_xyz[0])
                info["insertion_depth"] = insertion_depth
                rec["attempted_backoffs"].append(float(action.x_approach))
                rec["attempt_summaries"].append(
                    {
                        "backoff": float(action.x_approach),
                        "skipped": False,
                        "staging_success": bool(info["staging_success"]),
                        "entry_final_dist": float(info["entry_final_dist"]),
                        "sweep_final_dist": float(info["sweep_final_dist"]),
                    }
                )
                if self.timing_enabled:
                    rec["attempt_timing_ms"].append(
                        {
                            "backoff": float(action.x_approach),
                            "attempt_wall_ms": float(exec_wall_ms),
                            "phase_timing_ms": dict(info.get("phase_timing_ms", {})),
                        }
                    )
                rec["attempt_wall_sum"] = float(rec["attempt_wall_sum"]) + float(exec_wall_ms)
                rec["chosen_result"] = result
                rec["chosen_slot"] = int(slot)
                rec["chosen_info"] = dict(info)
                rec["chosen_action"] = _clone_action(action)
                rec["chosen_state"] = final_states[slot : slot + 1].clone()

                if bool(info["staging_success"]):
                    rec["resolved"] = True

            unresolved_lane_ids = {i for i in unresolved_lane_ids if not lane_records[i]["resolved"]}

        # Safety parity with single-action executor: if every candidate backoff
        # was skipped, still run one attempt with the original approach.
        safety_lane_ids = [i for i, rec in enumerate(lane_records) if rec["chosen_info"] is None]
        if safety_lane_ids:
            safety_actions = [_clone_action(lane_records[i]["base_action"]) for i in safety_lane_ids]
            result, final_states, exec_wall_ms = self._run_attempt_round(
                parent_dev=parent_dev,
                parent_state_2d=parent_state_2d,
                actions=safety_actions,
                batch_size=batch_size,
            )
            batch_round_count += 1
            batch_round_wall_sum_ms += float(exec_wall_ms)
            for slot, lane_id in enumerate(safety_lane_ids):
                rec = lane_records[lane_id]
                action = safety_actions[slot]
                info = self._build_result_info_at_index(
                    result,
                    slot,
                    self.convergence_threshold,
                    include_timing=self.timing_enabled,
                )
                insertion_depth = float(action.entry_xyz[0] - action.approach_xyz[0])
                info["insertion_depth"] = insertion_depth
                rec["attempted_backoffs"].append(float(action.x_approach))
                rec["attempt_summaries"].append(
                    {
                        "backoff": float(action.x_approach),
                        "skipped": False,
                        "staging_success": bool(info["staging_success"]),
                        "entry_final_dist": float(info["entry_final_dist"]),
                        "sweep_final_dist": float(info["sweep_final_dist"]),
                    }
                )
                if self.timing_enabled:
                    rec["attempt_timing_ms"].append(
                        {
                            "backoff": float(action.x_approach),
                            "attempt_wall_ms": float(exec_wall_ms),
                            "phase_timing_ms": dict(info.get("phase_timing_ms", {})),
                        }
                    )
                rec["attempt_wall_sum"] = float(rec["attempt_wall_sum"]) + float(exec_wall_ms)
                rec["chosen_result"] = result
                rec["chosen_slot"] = int(slot)
                rec["chosen_info"] = dict(info)
                rec["chosen_action"] = _clone_action(action)
                rec["chosen_state"] = final_states[slot : slot + 1].clone()

        outputs: list[BatchExecutionResult] = []
        active = int(len(actions))
        batch_execute_total_wall_ms = (
            float(time.perf_counter_ns() - int(t_batch_total_ns)) / 1e6 if self.timing_enabled else 0.0
        )
        for idx, rec in enumerate(lane_records):
            chosen_result: StickPushResult = rec["chosen_result"]
            chosen_slot = int(rec["chosen_slot"])
            chosen_info = dict(rec["chosen_info"])
            chosen_action: PlannerAction = rec["chosen_action"]
            base_action: PlannerAction = rec["base_action"]
            chosen_backoff = float(chosen_action.x_approach)

            chosen_info["staging_attempted_backoffs"] = [float(v) for v in rec["attempted_backoffs"]]
            chosen_info["staging_attempt_summaries"] = list(rec["attempt_summaries"])
            if self.timing_enabled:
                chosen_info["staging_attempt_timing_ms"] = list(rec["attempt_timing_ms"])
            chosen_info["staging_used_backoff"] = float(chosen_backoff)
            chosen_info["staging_retry_count"] = max(0, len(rec["attempted_backoffs"]) - 1)
            chosen_info["staging_fallback_applied"] = bool(
                abs(float(chosen_backoff) - float(base_action.x_approach)) > 1e-6
            )
            chosen_info["approach_xyz_used"] = np.asarray(chosen_action.approach_xyz, dtype=np.float32).tolist()
            chosen_info["x_approach_used"] = float(chosen_backoff)
            chosen_info["entry_xyz_used"] = np.asarray(chosen_action.entry_xyz, dtype=np.float32).tolist()
            chosen_info["sweep_xyz_used"] = np.asarray(chosen_action.sweep_xyz, dtype=np.float32).tolist()
            chosen_info["retract_xyz_used"] = np.asarray(chosen_action.retract_xyz, dtype=np.float32).tolist()
            chosen_info["batch_lane"] = int(idx)
            chosen_info["batch_active"] = int(active)
            chosen_info["batch_capacity"] = int(batch_size)
            if self.timing_enabled:
                # Per-action/lane timing: sum of wall times for rounds where this lane
                # was actually attempted (includes fallback retries for this lane).
                lane_execute_total_wall_ms = float(rec["attempt_wall_sum"])
                # Batch-level timing: shared across all lanes for this call.
                chosen_info["timing_ms"] = {
                    "execute_total_wall": lane_execute_total_wall_ms,
                    "attempt_wall_sum": lane_execute_total_wall_ms,
                    "batch_execute_total_wall": float(batch_execute_total_wall_ms),
                    "batch_round_wall_sum": float(batch_round_wall_sum_ms),
                    "batch_round_count": int(batch_round_count),
                }

            execution = ExecutionResult(
                success=bool(chosen_result.success_mask[chosen_slot].item() and bool(chosen_info["entry_converged"])),
                steps_executed=int(chosen_result.steps_executed),
                stick_result=chosen_result,
                info=chosen_info,
            )
            child_state = rec["chosen_state"].to(device=parent_state_2d.device).clone()
            outputs.append(BatchExecutionResult(execution=execution, child_state=child_state))
        return outputs
