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

        _add(action.x_approach)
        for v in self.staging_backoff_candidates:
            _add(v)
        return ordered

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
