"""Receding-horizon search loop for shelf stick pushing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig
from taskbench.planners.stickpush_rh.executor import StickPushBatchExecutor, StickPushExecutor
from taskbench.planners.stickpush_rh.frontier_ucb import FrontierUCB
from taskbench.planners.stickpush_rh.geometry import (
    compute_node_metrics,
    front_semicircle_wall_intersections,
    grasp_semicircle_center_xy,
    hash_scene_state,
)
from taskbench.planners.stickpush_rh.insertion_grid import (
    WavefrontGrid,
    build_wavefront_grid,
    solve_straight_insertion,
)
from taskbench.planners.stickpush_rh.io import (
    build_artifact_dir,
    save_image,
    save_json,
    save_video,
    serialize_action,
    serialize_metrics,
    write_frontier_csv,
)
from taskbench.planners.stickpush_rh.ranking import metrics_key, progress_reward
from taskbench.planners.stickpush_rh.sampler import StickPushSampler
from taskbench.planners.stickpush_rh.state_provider import GTSceneStateProvider
from taskbench.planners.stickpush_rh.types import NodeMetrics, PlannerAction, SearchNode
from taskbench.planners.stickpush_rh.viz import capture_rgb_frame, render_candidate_map


def _elapsed_ms(start_ns: int) -> float:
    return float(time.perf_counter_ns() - int(start_ns)) / 1e6


def _stats_ms(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "sum_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum_ms": float(arr.sum()),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _stats_scalar(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@dataclass
class PlannerRunResult:
    success: bool
    solved_node_id: int | None
    expansions: int
    executions: int
    solve_depth: int | None
    artifact_dir: str
    plan_actions: list[PlannerAction] = field(default_factory=list)
    plan_metrics: list[NodeMetrics] = field(default_factory=list)
    info: dict = field(default_factory=dict)


class StickPushRecedingHorizonSearch:
    """UCB-guided receding-horizon search over sampled stick-push actions."""

    def __init__(
        self,
        env,
        *,
        executor: StickPushExecutor,
        batch_executor: StickPushBatchExecutor | None = None,
        cfg: RecedingHorizonConfig,
        state_provider: GTSceneStateProvider | None = None,
        sampler: StickPushSampler | None = None,
        ucb: FrontierUCB | None = None,
    ):
        self.env = env
        self.raw = env.unwrapped
        self.cfg = cfg
        self.executor = executor
        self.batch_executor = batch_executor
        self.state_provider = state_provider or GTSceneStateProvider(env_index=0)
        self.sampler = sampler or StickPushSampler(cfg.sampling)
        self.ucb = ucb or FrontierUCB(cfg.ucb)
        self.timing_enabled = bool(self.cfg.visual.save_timing_diagnostics)
        self._wavefront_cache: dict[str, WavefrontGrid] = {}
        self._feasible_action_cache: dict[str, list[PlannerAction]] = {}
        self._action_diag_cache: dict[str, dict] = {}

    def _expandable_node_ids(self, nodes: dict[int, SearchNode]) -> list[int]:
        return [
            node_id
            for node_id, node in nodes.items()
            if node.untried_actions and node.depth < self.cfg.max_depth
        ]

    def _backprop(self, nodes: dict[int, SearchNode], start_node_id: int, reward: float) -> None:
        cur = start_node_id
        while cur is not None:
            n = nodes[cur]
            n.visits += 1
            n.reward_sum += float(reward)
            cur = n.parent_id

    def _recover_chain(
        self, nodes: dict[int, SearchNode], leaf_id: int
    ) -> tuple[list[PlannerAction], list[NodeMetrics], list[int]]:
        node_ids: list[int] = []
        actions_rev: list[PlannerAction] = []
        metrics_rev: list[NodeMetrics] = []
        cur = leaf_id
        while cur is not None:
            node = nodes[cur]
            node_ids.append(cur)
            metrics_rev.append(node.metrics)
            if node.parent_action is not None:
                actions_rev.append(node.parent_action)
            cur = node.parent_id
        node_ids.reverse()
        actions_rev.reverse()
        metrics_rev.reverse()
        return actions_rev, metrics_rev, node_ids

    def _frontier_rows(self, nodes: dict[int, SearchNode]) -> list[dict]:
        total_visits = max(1, sum(max(1, n.visits) for n in nodes.values()))
        rows = []
        for node_id, node in nodes.items():
            if not node.untried_actions:
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "visits": node.visits,
                    "reward_sum": node.reward_sum,
                    "ucb_score": self.ucb.score(node, total_visits),
                    "n_untried": len(node.untried_actions),
                    "state_hash": node.state_hash,
                    **serialize_metrics(node.metrics),
                }
            )
        rows.sort(key=lambda row: row["ucb_score"], reverse=True)
        return rows

    def _best_node_id(self, nodes: dict[int, SearchNode]) -> int:
        return min(nodes.keys(), key=lambda nid: (metrics_key(nodes[nid].metrics), nodes[nid].depth, nid))

    def _target_goal_checks(
        self,
        scene,
        *,
        root_target_xy: np.ndarray,
    ) -> dict:
        target_xy = scene.target.center_xyz[:2]
        shift_xy = float(np.linalg.norm(target_xy - root_target_xy))
        shift_ok = shift_xy <= float(self.cfg.max_target_shift_xy)
        grasp_center_xy = grasp_semicircle_center_xy(scene.target, scene.open_dir_xy)

        # Front side is open; wall checks are limited to intersections with the
        # front semicircle region (same region used for blocker clearance).
        side_wall_dist = float(scene.shelf_half_w - abs(float(grasp_center_xy[1])))
        back_wall_dist = float(scene.shelf_back_x - float(grasp_center_xy[0]))
        wall_radius = float(self.cfg.sampling.clearance_radius)
        wall_hits = front_semicircle_wall_intersections(scene, wall_radius)
        wall_in_radius = bool(len(wall_hits) > 0)

        return {
            "target_xy": target_xy.tolist(),
            "grasp_center_xy": grasp_center_xy.tolist(),
            "target_shift_xy": shift_xy,
            "target_shift_ok": bool(shift_ok),
            "side_wall_dist": side_wall_dist,
            "back_wall_dist": back_wall_dist,
            "wall_radius_threshold": wall_radius,
            "wall_in_radius": wall_in_radius,
            "wall_intersections": wall_hits,
            "target_wall_margin": float(self.cfg.target_wall_margin),
            "max_target_shift_xy": float(self.cfg.max_target_shift_xy),
        }

    def _goal_satisfied(self, metrics: NodeMetrics, goal_checks: dict) -> bool:
        solved_by_clearance = bool(metrics.blockers == 0)
        wall_clear = not bool(goal_checks.get("wall_in_radius", False))
        if self.cfg.require_target_shift_limit_for_success and not goal_checks.get("target_shift_ok", False):
            return False
        return bool(solved_by_clearance and wall_clear)

    @staticmethod
    def _shuffle_actions(
        actions: list[PlannerAction],
        rng: np.random.Generator,
    ) -> list[PlannerAction]:
        if len(actions) <= 1:
            return list(actions)
        idx = rng.permutation(len(actions))
        return [actions[int(i)] for i in idx]

    @staticmethod
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

    @staticmethod
    def _capture_rgb_frame_from_env_index(env, env_index: int) -> np.ndarray:
        try:
            frame = env.render()
        except RuntimeError as exc:
            if "render_mode is not set" in str(exc):
                return np.zeros((512, 512, 3), dtype=np.uint8)
            raise
        if frame is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        if hasattr(frame, "detach") and hasattr(frame, "cpu"):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)
        if arr.ndim == 4:
            idx = int(max(0, min(int(env_index), int(arr.shape[0]) - 1)))
            arr = arr[idx]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _build_node_actions(
        self,
        scene,
        *,
        state_hash: str,
        rng: np.random.Generator,
    ) -> tuple[list[PlannerAction], dict]:
        timing_enabled = bool(self.timing_enabled)
        t_total_ns = time.perf_counter_ns()

        if not self.cfg.sampling.use_wavefront_insertion_solver:
            t_sampling_ns = time.perf_counter_ns()
            sampled_actions = self.sampler.sample_actions(scene)
            sampling_ms = _elapsed_ms(t_sampling_ns)
            t_shuffle_ns = time.perf_counter_ns()
            shuffled = self._shuffle_actions(sampled_actions, rng)
            shuffle_ms = _elapsed_ms(t_shuffle_ns)
            diagnostics = {
                "num_sampled_actions": int(len(sampled_actions)),
                "num_feasible_actions": int(len(sampled_actions)),
                "wavefront_used": False,
                "wavefront_cache_hit": False,
                "action_cache_hit": False,
                "rejected_infeasible_insertion": 0,
                "num_insertion_checks": 0,
                "num_insertion_solved": 0,
                "timing_ms": {
                    "total": _elapsed_ms(t_total_ns),
                    "sampling": sampling_ms,
                    "wavefront_build": 0.0,
                    "insertion_filter": 0.0,
                    "insertion_solver_only": 0.0,
                    "cache_clone": 0.0,
                    "shuffle": shuffle_ms,
                },
            }
            if not timing_enabled:
                diagnostics.pop("timing_ms", None)
            return shuffled, diagnostics

        if state_hash in self._feasible_action_cache:
            t_cache_clone_ns = time.perf_counter_ns()
            cached = [self._clone_action(a) for a in self._feasible_action_cache[state_hash]]
            cache_clone_ms = _elapsed_ms(t_cache_clone_ns)
            t_shuffle_ns = time.perf_counter_ns()
            shuffled = self._shuffle_actions(cached, rng)
            shuffle_ms = _elapsed_ms(t_shuffle_ns)
            diagnostics = {
                "num_sampled_actions": int(len(cached)),
                "num_feasible_actions": int(len(cached)),
                "wavefront_used": True,
                "wavefront_cache_hit": True,
                "action_cache_hit": True,
                "rejected_infeasible_insertion": 0,
                "num_insertion_checks": 0,
                "num_insertion_solved": int(len(cached)),
                "timing_ms": {
                    "total": _elapsed_ms(t_total_ns),
                    "sampling": 0.0,
                    "wavefront_build": 0.0,
                    "insertion_filter": 0.0,
                    "insertion_solver_only": 0.0,
                    "cache_clone": cache_clone_ms,
                    "shuffle": shuffle_ms,
                },
            }
            if not timing_enabled:
                diagnostics.pop("timing_ms", None)
            return shuffled, diagnostics

        t_sampling_ns = time.perf_counter_ns()
        sampled_actions = self.sampler.sample_actions(scene)
        sampling_ms = _elapsed_ms(t_sampling_ns)
        diagnostics = {
            "num_sampled_actions": int(len(sampled_actions)),
            "num_feasible_actions": int(len(sampled_actions)),
            "wavefront_used": True,
            "wavefront_cache_hit": False,
            "action_cache_hit": False,
            "rejected_infeasible_insertion": 0,
            "num_insertion_checks": 0,
            "num_insertion_solved": 0,
        }

        wavefront_build_ms = 0.0
        if state_hash in self._wavefront_cache:
            wavefront = self._wavefront_cache[state_hash]
            diagnostics["wavefront_cache_hit"] = True
        else:
            t_wavefront_ns = time.perf_counter_ns()
            wavefront = build_wavefront_grid(scene, self.cfg.sampling)
            self._wavefront_cache[state_hash] = wavefront
            wavefront_build_ms = _elapsed_ms(t_wavefront_ns)

        t_insertion_filter_ns = time.perf_counter_ns()
        insertion_solver_only_ms = 0.0
        insertion_checks = 0
        feasible_actions: list[PlannerAction] = []
        for action in sampled_actions:
            insertion_checks += 1
            t_solver_ns = time.perf_counter_ns()
            plan = solve_straight_insertion(
                wavefront,
                scene,
                action.entry_xyz[:2],
                self.cfg.sampling,
            )
            insertion_solver_only_ms += _elapsed_ms(t_solver_ns)
            if plan is None:
                continue

            z_push = float(scene.surface_z + self.cfg.sampling.z_eps)
            approach = np.array(
                [scene.shelf_front_x - float(plan.approach_backoff), float(plan.source_xy[1]), z_push],
                dtype=np.float32,
            )
            entry = np.asarray(action.entry_xyz, dtype=np.float32).copy()
            sweep = np.asarray(action.sweep_xyz, dtype=np.float32).copy()
            entry[2] = z_push
            sweep[2] = z_push
            retract = np.array(
                [scene.shelf_front_x - float(self.cfg.sampling.retract_backoff), float(sweep[1]), z_push],
                dtype=np.float32,
            )
            insertion_depth = float(entry[0] - approach[0])
            if insertion_depth < float(self.cfg.sampling.min_insertion_depth):
                continue

            meta = dict(action.meta)
            meta["insertion_wavefront_dist"] = int(plan.wavefront_dist)
            meta["insertion_source_xy"] = np.asarray(plan.source_xy, dtype=np.float32).tolist()
            meta["insertion_backoff"] = float(plan.approach_backoff)
            meta["insertion_solver"] = "wavefront_straight_los"

            feasible_actions.append(
                PlannerAction(
                    blocker_name=action.blocker_name,
                    theta_deg=float(action.theta_deg),
                    x_approach=float(plan.approach_backoff),
                    delta_len_idx=int(action.delta_len_idx),
                    contact_offset=float(action.contact_offset),
                    push_len=float(action.push_len),
                    approach_xyz=approach,
                    entry_xyz=entry,
                    sweep_xyz=sweep,
                    retract_xyz=retract,
                    meta=meta,
                )
            )

        insertion_filter_ms = _elapsed_ms(t_insertion_filter_ns)
        diagnostics["num_feasible_actions"] = int(len(feasible_actions))
        diagnostics["rejected_infeasible_insertion"] = int(len(sampled_actions) - len(feasible_actions))
        diagnostics["num_insertion_checks"] = int(insertion_checks)
        diagnostics["num_insertion_solved"] = int(len(feasible_actions))
        diagnostics["timing_ms"] = {
            "total": 0.0,
            "sampling": sampling_ms,
            "wavefront_build": wavefront_build_ms,
            "insertion_filter": insertion_filter_ms,
            "insertion_solver_only": insertion_solver_only_ms,
            "cache_clone": 0.0,
            "shuffle": 0.0,
        }
        self._feasible_action_cache[state_hash] = [self._clone_action(a) for a in feasible_actions]
        self._action_diag_cache[state_hash] = dict(diagnostics)
        t_shuffle_ns = time.perf_counter_ns()
        shuffled = self._shuffle_actions([self._clone_action(a) for a in feasible_actions], rng)
        shuffle_ms = _elapsed_ms(t_shuffle_ns)
        diagnostics["timing_ms"]["shuffle"] = shuffle_ms
        diagnostics["timing_ms"]["total"] = _elapsed_ms(t_total_ns)
        if not timing_enabled:
            diagnostics.pop("timing_ms", None)
        return shuffled, diagnostics

    def run(self, seed: int | None = None) -> PlannerRunResult:
        t_run_ns = time.perf_counter_ns()
        rng = np.random.default_rng(seed)
        artifacts = build_artifact_dir(self.cfg.artifact_root, seed)
        root_state = self.raw.get_state().clone()
        timing_enabled = bool(self.timing_enabled)

        action_gen_rows: list[dict] = []
        expansion_timing_rows: list[dict] = []
        action_gen_total_ms: list[float] = []
        action_gen_sampling_ms: list[float] = []
        action_gen_wavefront_build_ms: list[float] = []
        action_gen_insertion_filter_ms: list[float] = []
        action_gen_insertion_solver_ms: list[float] = []
        action_gen_cache_clone_ms: list[float] = []
        action_gen_shuffle_ms: list[float] = []
        action_gen_cache_hits = 0
        action_gen_wavefront_cache_hits = 0
        execution_ms_values: list[float] = []
        expansion_total_ms_values: list[float] = []
        expansion_prep_ms_values: list[float] = []
        expansion_child_eval_ms_values: list[float] = []
        expansion_child_action_gen_ms_values: list[float] = []
        expansion_artifact_io_ms_values: list[float] = []
        frontier_expandable_scan_ms_values: list[float] = []
        frontier_select_ms_values: list[float] = []
        frontier_candidate_count_values: list[float] = []
        execution_internal_total_ms_values: list[float] = []
        execution_phase_stage_plan_ms_values: list[float] = []
        execution_phase_stage_execute_ms_values: list[float] = []
        execution_phase_entry_ms_values: list[float] = []
        execution_phase_sweep_ms_values: list[float] = []
        execution_phase_retract_nudge_ms_values: list[float] = []
        execution_phase_retract_pull_ms_values: list[float] = []
        execution_phase_return_to_rest_ms_values: list[float] = []
        execution_phase_motion_total_excl_plan_ms_values: list[float] = []
        execution_phase_motion_total_incl_plan_ms_values: list[float] = []
        reject_reason_counts: dict[str, int] = {}

        def _record_action_generation(
            *,
            stage: str,
            node_id: int,
            depth: int,
            diagnostics: dict,
        ) -> None:
            nonlocal action_gen_cache_hits, action_gen_wavefront_cache_hits
            timing = diagnostics.get("timing_ms", {}) if isinstance(diagnostics, dict) else {}
            row = {
                "stage": str(stage),
                "node_id": int(node_id),
                "depth": int(depth),
                "num_sampled_actions": int(diagnostics.get("num_sampled_actions", 0)),
                "num_feasible_actions": int(diagnostics.get("num_feasible_actions", 0)),
                "rejected_infeasible_insertion": int(diagnostics.get("rejected_infeasible_insertion", 0)),
                "num_insertion_checks": int(diagnostics.get("num_insertion_checks", 0)),
                "num_insertion_solved": int(diagnostics.get("num_insertion_solved", 0)),
                "wavefront_used": bool(diagnostics.get("wavefront_used", False)),
                "wavefront_cache_hit": bool(diagnostics.get("wavefront_cache_hit", False)),
                "action_cache_hit": bool(diagnostics.get("action_cache_hit", False)),
                "timing_ms": {
                    "total": float(timing.get("total", 0.0)),
                    "sampling": float(timing.get("sampling", 0.0)),
                    "wavefront_build": float(timing.get("wavefront_build", 0.0)),
                    "insertion_filter": float(timing.get("insertion_filter", 0.0)),
                    "insertion_solver_only": float(timing.get("insertion_solver_only", 0.0)),
                    "cache_clone": float(timing.get("cache_clone", 0.0)),
                    "shuffle": float(timing.get("shuffle", 0.0)),
                },
            }
            action_gen_rows.append(row)
            action_gen_total_ms.append(float(row["timing_ms"]["total"]))
            action_gen_sampling_ms.append(float(row["timing_ms"]["sampling"]))
            action_gen_wavefront_build_ms.append(float(row["timing_ms"]["wavefront_build"]))
            action_gen_insertion_filter_ms.append(float(row["timing_ms"]["insertion_filter"]))
            action_gen_insertion_solver_ms.append(float(row["timing_ms"]["insertion_solver_only"]))
            action_gen_cache_clone_ms.append(float(row["timing_ms"]["cache_clone"]))
            action_gen_shuffle_ms.append(float(row["timing_ms"]["shuffle"]))
            if bool(row["action_cache_hit"]):
                action_gen_cache_hits += 1
            if bool(row["wavefront_cache_hit"]):
                action_gen_wavefront_cache_hits += 1

        root_scene = self.state_provider.get_scene_state(self.env)
        root_metrics = compute_node_metrics(
            root_scene, self.cfg.sampling.clearance_radius, pushes_used=0
        )
        root_target_xy = root_scene.target.center_xyz[:2].copy()
        root_goal_diag = self._target_goal_checks(root_scene, root_target_xy=root_target_xy)
        root_goal_diag["solved_by_clearance"] = bool(root_metrics.blockers == 0)
        root_goal_diag["wall_clear"] = bool(not root_goal_diag["wall_in_radius"])
        root_metrics.solved = self._goal_satisfied(root_metrics, root_goal_diag)
        root_state_hash = hash_scene_state(root_scene)
        root_actions, root_action_diag = self._build_node_actions(
            root_scene,
            state_hash=root_state_hash,
            rng=rng,
        )
        _record_action_generation(stage="root", node_id=0, depth=0, diagnostics=root_action_diag)

        nodes: dict[int, SearchNode] = {
            0: SearchNode(
                node_id=0,
                parent_id=None,
                parent_action=None,
                sim_state=root_state,
                metrics=root_metrics,
                depth=0,
                visits=0,
                reward_sum=0.0,
                untried_actions=list(root_actions),
                child_ids=[],
                state_hash=root_state_hash,
            )
        }
        next_node_id = 1
        solved_node_id: int | None = 0 if root_metrics.solved else None
        expansions = 0
        executions = 0
        parallel_frontier_enabled = bool(self.cfg.parallel_frontier_enabled)
        parallel_frontier_batch_size = max(1, int(self.cfg.parallel_frontier_batch_size))
        pending_exec_queue: list[dict] = []

        t_root_artifact_io_ns = time.perf_counter_ns()
        save_image(artifacts / "root_before.png", capture_rgb_frame(self.env))
        save_image(
            artifacts / "root_candidates.png",
            render_candidate_map(
                root_scene,
                root_actions,
                selected_action=None,
                clearance_radius=self.cfg.sampling.clearance_radius,
                max_candidates_drawn=self.cfg.visual.max_candidates_drawn,
            ),
        )
        save_json(
            artifacts / "root_metrics.json",
            {
                "metrics": serialize_metrics(root_metrics),
                "num_candidates": len(root_actions),
                **root_action_diag,
                "state_hash": nodes[0].state_hash,
                **root_goal_diag,
            },
        )
        root_artifact_io_ms = _elapsed_ms(t_root_artifact_io_ns)

        t_search_loop_ns = time.perf_counter_ns()
        while solved_node_id is None and executions < self.cfg.max_executions:
            t_expansion_ns = time.perf_counter_ns()
            t_prep_ns = time.perf_counter_ns()
            t_frontier_scan_ns = time.perf_counter_ns()
            candidate_ids = self._expandable_node_ids(nodes)
            frontier_expandable_scan_ms = _elapsed_ms(t_frontier_scan_ns)
            if (not candidate_ids) and (not pending_exec_queue):
                break
            frontier_candidate_count = int(len(candidate_ids))
            if pending_exec_queue:
                queued = pending_exec_queue.pop(0)
                node_id = int(queued["node_id"])
                node = nodes[node_id]
                action = queued["action"]
                exec_result = queued["exec_result"]
                scene_before = queued["scene_before"]
                before_frame = queued["before_frame"]
                prep_ms = 0.0
                frontier_select_ms = 0.0
                child_state_override = queued.get("child_state")
                if child_state_override is not None:
                    self.raw.set_state(child_state_override.clone())
                exec_ms = float(queued.get("exec_ms", 0.0))
            else:
                t_frontier_select_ns = time.perf_counter_ns()
                node_id = self.ucb.select(nodes, candidate_ids)
                frontier_select_ms = _elapsed_ms(t_frontier_select_ns)
                node = nodes[node_id]

                self.raw.set_state(node.sim_state.clone())
                scene_before = self.state_provider.get_scene_state(self.env)
                before_frame = capture_rgb_frame(self.env)

                if not node.untried_actions:
                    continue

                remaining_budget = int(max(0, self.cfg.max_executions - executions))
                batch_count = 1
                if (
                    parallel_frontier_enabled
                    and parallel_frontier_batch_size > 1
                    and self.batch_executor is not None
                ):
                    batch_count = min(
                        int(parallel_frontier_batch_size),
                        int(len(node.untried_actions)),
                        int(remaining_budget),
                    )
                actions_to_execute = [
                    node.untried_actions.pop(0) for _ in range(max(1, int(batch_count)))
                ]
                prep_ms = _elapsed_ms(t_prep_ns)

                if len(actions_to_execute) > 1 and self.batch_executor is not None:
                    t_exec_ns = time.perf_counter_ns()
                    batch_outputs = self.batch_executor.execute_batch_from_parent_state(
                        node.sim_state,
                        actions_to_execute,
                    )
                    exec_total_ms = _elapsed_ms(t_exec_ns)
                    exec_per_action_ms = float(exec_total_ms / max(1, len(batch_outputs)))
                    pending_exec_queue.extend(
                        [
                            {
                                "node_id": int(node_id),
                                "action": actions_to_execute[i],
                                "exec_result": batch_outputs[i].execution,
                                "child_state": batch_outputs[i].child_state,
                                "scene_before": scene_before,
                                "before_frame": before_frame,
                                "exec_ms": float(exec_per_action_ms),
                            }
                            for i in range(1, len(batch_outputs))
                        ]
                    )
                    action = actions_to_execute[0]
                    exec_result = batch_outputs[0].execution
                    self.raw.set_state(batch_outputs[0].child_state.clone())
                    exec_ms = float(exec_per_action_ms)
                else:
                    action = actions_to_execute[0]
                    t_exec_ns = time.perf_counter_ns()
                    exec_result = self.executor.execute(action)
                    exec_ms = _elapsed_ms(t_exec_ns)
            exec_internal_timing = dict(exec_result.info.get("timing_ms", {}))
            exec_phase_timing = dict(exec_result.info.get("phase_timing_ms", {}))
            exec_internal_total_ms = float(exec_internal_timing.get("execute_total_wall", 0.0))
            exec_phase_stage_plan_ms = float(exec_phase_timing.get("stage_plan", 0.0))
            exec_phase_stage_execute_ms = float(exec_phase_timing.get("stage_execute", 0.0))
            exec_phase_entry_ms = float(exec_phase_timing.get("entry_move", 0.0))
            exec_phase_sweep_ms = float(exec_phase_timing.get("sweep_move", 0.0))
            exec_phase_retract_nudge_ms = float(exec_phase_timing.get("retract_nudge", 0.0))
            exec_phase_retract_pull_ms = float(exec_phase_timing.get("retract_pull", 0.0))
            exec_phase_return_to_rest_ms = float(exec_phase_timing.get("return_to_rest", 0.0))
            exec_phase_motion_total_excl_plan_ms = float(
                exec_phase_timing.get("motion_total_excluding_stage_plan", 0.0)
            )
            exec_phase_motion_total_incl_plan_ms = float(
                exec_phase_timing.get("motion_total_including_stage_plan", 0.0)
            )
            executed_action = self._clone_action(action)
            used_approach = exec_result.info.get("approach_xyz_used")
            if used_approach is not None:
                executed_action.approach_xyz = np.asarray(used_approach, dtype=np.float32).copy()
            if "x_approach_used" in exec_result.info:
                executed_action.x_approach = float(exec_result.info["x_approach_used"])
            if "staging_used_backoff" in exec_result.info:
                executed_action.meta["staging_used_backoff"] = float(exec_result.info["staging_used_backoff"])
            if "staging_fallback_applied" in exec_result.info:
                executed_action.meta["staging_fallback_applied"] = bool(exec_result.info["staging_fallback_applied"])
            candidate_snapshot = [executed_action] + list(node.untried_actions)
            reject_reason: str | None = None
            if not bool(exec_result.info.get("staging_success", True)):
                reject_reason = "staging_failed"
            elif not bool(exec_result.info.get("entry_converged", True)):
                reject_reason = "insert_not_converged"

            child_eval_ms = 0.0
            child_action_generation_ms = 0.0
            t_child_eval_ns = time.perf_counter_ns()
            if reject_reason is not None:
                # Reject samples that fail to reach insertion waypoint.
                self._backprop(nodes, node_id, reward=0.0)
                child_eval_ms = _elapsed_ms(t_child_eval_ns)
                expansions += 1
                executions += 1
                exp_dir = artifacts / f"exp_{expansions:06d}"

                t_artifact_io_ns = time.perf_counter_ns()
                after_frame = capture_rgb_frame(self.env)
                if self.cfg.visual.save_expansion_images:
                    save_image(exp_dir / "before.png", before_frame)
                    save_image(
                        exp_dir / "candidates.png",
                        render_candidate_map(
                            scene_before,
                            candidate_snapshot,
                            selected_action=executed_action,
                            clearance_radius=self.cfg.sampling.clearance_radius,
                            max_candidates_drawn=self.cfg.visual.max_candidates_drawn,
                        ),
                    )
                    save_image(
                        exp_dir / "selected_only.png",
                        render_candidate_map(
                            scene_before,
                            [executed_action],
                            selected_action=executed_action,
                            clearance_radius=self.cfg.sampling.clearance_radius,
                            max_candidates_drawn=1,
                        ),
                    )
                    save_image(exp_dir / "after.png", after_frame)

                save_json(
                    exp_dir / "metrics.json",
                    {
                        "parent_node_id": node_id,
                        "child_node_id": None,
                        "selected_action": serialize_action(executed_action),
                        "parent_metrics": serialize_metrics(node.metrics),
                        "execution": {
                            "success": exec_result.success,
                            "steps_executed": exec_result.steps_executed,
                            "info": exec_result.info,
                        },
                        "rejected": True,
                        "rejected_reason": reject_reason,
                    },
                )
                if self.cfg.visual.save_frontier_csv:
                    write_frontier_csv(
                        exp_dir / "frontier.csv",
                        self._frontier_rows(nodes),
                    )
                artifact_io_ms = _elapsed_ms(t_artifact_io_ns)
                total_ms = _elapsed_ms(t_expansion_ns)
                timing_ms = {
                    "frontier_expandable_scan": frontier_expandable_scan_ms,
                    "frontier_select": frontier_select_ms,
                    "prep": prep_ms,
                    "execute": exec_ms,
                    "child_eval": child_eval_ms,
                    "child_action_generation": child_action_generation_ms,
                    "artifact_io": artifact_io_ms,
                    "total": total_ms,
                }
                frontier_expandable_scan_ms_values.append(frontier_expandable_scan_ms)
                frontier_select_ms_values.append(frontier_select_ms)
                frontier_candidate_count_values.append(float(frontier_candidate_count))
                execution_ms_values.append(exec_ms)
                execution_internal_total_ms_values.append(exec_internal_total_ms)
                execution_phase_stage_plan_ms_values.append(exec_phase_stage_plan_ms)
                execution_phase_stage_execute_ms_values.append(exec_phase_stage_execute_ms)
                execution_phase_entry_ms_values.append(exec_phase_entry_ms)
                execution_phase_sweep_ms_values.append(exec_phase_sweep_ms)
                execution_phase_retract_nudge_ms_values.append(exec_phase_retract_nudge_ms)
                execution_phase_retract_pull_ms_values.append(exec_phase_retract_pull_ms)
                execution_phase_return_to_rest_ms_values.append(exec_phase_return_to_rest_ms)
                execution_phase_motion_total_excl_plan_ms_values.append(
                    exec_phase_motion_total_excl_plan_ms
                )
                execution_phase_motion_total_incl_plan_ms_values.append(
                    exec_phase_motion_total_incl_plan_ms
                )
                expansion_total_ms_values.append(total_ms)
                expansion_prep_ms_values.append(prep_ms)
                expansion_child_eval_ms_values.append(child_eval_ms)
                expansion_child_action_gen_ms_values.append(child_action_generation_ms)
                expansion_artifact_io_ms_values.append(artifact_io_ms)
                reject_reason_counts[reject_reason] = int(reject_reason_counts.get(reject_reason, 0) + 1)
                expansion_timing_rows.append(
                    {
                        "expansion": int(expansions),
                        "parent_node_id": int(node_id),
                        "child_node_id": None,
                        "parent_depth": int(node.depth),
                        "rejected": True,
                        "rejected_reason": str(reject_reason),
                        "selected_is_target_push": bool(executed_action.meta.get("is_target_push", False)),
                        "execution_steps": int(exec_result.steps_executed),
                        "timing_ms": timing_ms,
                        "execution_timing_ms": {
                            "executor_internal_total": exec_internal_total_ms,
                            "phase": {
                                "stage_plan": exec_phase_stage_plan_ms,
                                "stage_execute": exec_phase_stage_execute_ms,
                                "entry_move": exec_phase_entry_ms,
                                "sweep_move": exec_phase_sweep_ms,
                                "retract_nudge": exec_phase_retract_nudge_ms,
                                "retract_pull": exec_phase_retract_pull_ms,
                                "return_to_rest": exec_phase_return_to_rest_ms,
                                "motion_total_excluding_stage_plan": (
                                    exec_phase_motion_total_excl_plan_ms
                                ),
                                "motion_total_including_stage_plan": (
                                    exec_phase_motion_total_incl_plan_ms
                                ),
                            },
                        },
                    }
                )
                continue

            scene_after = self.state_provider.get_scene_state(self.env)
            child_state = self.raw.get_state().clone()
            child_metrics = compute_node_metrics(
                scene_after,
                self.cfg.sampling.clearance_radius,
                pushes_used=node.metrics.pushes_used + 1,
            )
            child_goal_diag = self._target_goal_checks(scene_after, root_target_xy=root_target_xy)
            child_goal_diag["solved_by_clearance"] = bool(child_metrics.blockers == 0)
            child_goal_diag["wall_clear"] = bool(not child_goal_diag["wall_in_radius"])
            child_metrics.solved = self._goal_satisfied(child_metrics, child_goal_diag)

            child_depth = node.depth + 1
            child_actions = []
            child_action_diag = {
                "num_sampled_actions": 0,
                "num_feasible_actions": 0,
                "wavefront_used": bool(self.cfg.sampling.use_wavefront_insertion_solver),
                "wavefront_cache_hit": False,
                "action_cache_hit": False,
                "rejected_infeasible_insertion": 0,
                "num_insertion_checks": 0,
                "num_insertion_solved": 0,
            }
            if timing_enabled:
                child_action_diag["timing_ms"] = {
                    "total": 0.0,
                    "sampling": 0.0,
                    "wavefront_build": 0.0,
                    "insertion_filter": 0.0,
                    "insertion_solver_only": 0.0,
                    "cache_clone": 0.0,
                    "shuffle": 0.0,
                }
            child_actions_built = False
            if not child_metrics.solved and child_depth < self.cfg.max_depth:
                if (not self.cfg.prune_target_invalid_nodes) or child_goal_diag["target_shift_ok"]:
                    child_state_hash = hash_scene_state(scene_after)
                    t_child_action_gen_ns = time.perf_counter_ns()
                    child_actions, child_action_diag = self._build_node_actions(
                        scene_after,
                        state_hash=child_state_hash,
                        rng=rng,
                    )
                    child_action_generation_ms = _elapsed_ms(t_child_action_gen_ns)
                    child_actions_built = True
                else:
                    child_state_hash = hash_scene_state(scene_after)
            else:
                child_state_hash = hash_scene_state(scene_after)

            child_id = next_node_id
            next_node_id += 1
            child_node = SearchNode(
                node_id=child_id,
                parent_id=node_id,
                parent_action=executed_action,
                sim_state=child_state,
                metrics=child_metrics,
                depth=child_depth,
                visits=0,
                reward_sum=0.0,
                untried_actions=list(child_actions),
                child_ids=[],
                state_hash=child_state_hash,
            )
            nodes[child_id] = child_node
            node.child_ids.append(child_id)
            if child_actions_built:
                _record_action_generation(
                    stage="child",
                    node_id=child_id,
                    depth=child_depth,
                    diagnostics=child_action_diag,
                )

            reward = progress_reward(
                node.metrics,
                child_metrics,
                solved_bonus=self.cfg.ucb.solved_bonus,
            )
            self._backprop(nodes, node_id, reward)
            child_eval_ms = _elapsed_ms(t_child_eval_ns)

            expansions += 1
            executions += 1
            exp_dir = artifacts / f"exp_{expansions:06d}"

            t_artifact_io_ns = time.perf_counter_ns()
            after_frame = capture_rgb_frame(self.env)
            if self.cfg.visual.save_expansion_images:
                save_image(exp_dir / "before.png", before_frame)
                save_image(
                    exp_dir / "candidates.png",
                    render_candidate_map(
                        scene_before,
                        candidate_snapshot,
                        selected_action=executed_action,
                        clearance_radius=self.cfg.sampling.clearance_radius,
                        max_candidates_drawn=self.cfg.visual.max_candidates_drawn,
                    ),
                )
                save_image(
                    exp_dir / "selected_only.png",
                    render_candidate_map(
                        scene_before,
                        [executed_action],
                        selected_action=executed_action,
                        clearance_radius=self.cfg.sampling.clearance_radius,
                        max_candidates_drawn=1,
                    ),
                )
                save_image(exp_dir / "after.png", after_frame)

            save_json(
                exp_dir / "metrics.json",
                {
                    "parent_node_id": node_id,
                    "child_node_id": child_id,
                    "selected_action": serialize_action(executed_action),
                    "parent_metrics": serialize_metrics(node.metrics),
                    "child_metrics": serialize_metrics(child_metrics),
                    "execution": {
                        "success": exec_result.success,
                        "steps_executed": exec_result.steps_executed,
                        "info": exec_result.info,
                    },
                    "goal_checks": child_goal_diag,
                    "action_generation": child_action_diag,
                },
            )
            if self.cfg.visual.save_frontier_csv:
                write_frontier_csv(
                    exp_dir / "frontier.csv",
                    self._frontier_rows(nodes),
                )
            artifact_io_ms = _elapsed_ms(t_artifact_io_ns)
            total_ms = _elapsed_ms(t_expansion_ns)
            timing_ms = {
                "frontier_expandable_scan": frontier_expandable_scan_ms,
                "frontier_select": frontier_select_ms,
                "prep": prep_ms,
                "execute": exec_ms,
                "child_eval": child_eval_ms,
                "child_action_generation": child_action_generation_ms,
                "artifact_io": artifact_io_ms,
                "total": total_ms,
            }
            frontier_expandable_scan_ms_values.append(frontier_expandable_scan_ms)
            frontier_select_ms_values.append(frontier_select_ms)
            frontier_candidate_count_values.append(float(frontier_candidate_count))
            execution_ms_values.append(exec_ms)
            execution_internal_total_ms_values.append(exec_internal_total_ms)
            execution_phase_stage_plan_ms_values.append(exec_phase_stage_plan_ms)
            execution_phase_stage_execute_ms_values.append(exec_phase_stage_execute_ms)
            execution_phase_entry_ms_values.append(exec_phase_entry_ms)
            execution_phase_sweep_ms_values.append(exec_phase_sweep_ms)
            execution_phase_retract_nudge_ms_values.append(exec_phase_retract_nudge_ms)
            execution_phase_retract_pull_ms_values.append(exec_phase_retract_pull_ms)
            execution_phase_return_to_rest_ms_values.append(exec_phase_return_to_rest_ms)
            execution_phase_motion_total_excl_plan_ms_values.append(
                exec_phase_motion_total_excl_plan_ms
            )
            execution_phase_motion_total_incl_plan_ms_values.append(
                exec_phase_motion_total_incl_plan_ms
            )
            expansion_total_ms_values.append(total_ms)
            expansion_prep_ms_values.append(prep_ms)
            expansion_child_eval_ms_values.append(child_eval_ms)
            expansion_child_action_gen_ms_values.append(child_action_generation_ms)
            expansion_artifact_io_ms_values.append(artifact_io_ms)
            expansion_timing_rows.append(
                {
                    "expansion": int(expansions),
                    "parent_node_id": int(node_id),
                    "child_node_id": int(child_id),
                    "parent_depth": int(node.depth),
                    "child_depth": int(child_depth),
                    "rejected": False,
                    "rejected_reason": None,
                    "selected_is_target_push": bool(executed_action.meta.get("is_target_push", False)),
                    "execution_steps": int(exec_result.steps_executed),
                    "timing_ms": timing_ms,
                    "execution_timing_ms": {
                        "executor_internal_total": exec_internal_total_ms,
                        "phase": {
                            "stage_plan": exec_phase_stage_plan_ms,
                            "stage_execute": exec_phase_stage_execute_ms,
                            "entry_move": exec_phase_entry_ms,
                            "sweep_move": exec_phase_sweep_ms,
                            "retract_nudge": exec_phase_retract_nudge_ms,
                            "retract_pull": exec_phase_retract_pull_ms,
                            "return_to_rest": exec_phase_return_to_rest_ms,
                            "motion_total_excluding_stage_plan": (
                                exec_phase_motion_total_excl_plan_ms
                            ),
                            "motion_total_including_stage_plan": (
                                exec_phase_motion_total_incl_plan_ms
                            ),
                        },
                    },
                    "child_action_generation": {
                        "num_sampled_actions": int(child_action_diag.get("num_sampled_actions", 0)),
                        "num_feasible_actions": int(child_action_diag.get("num_feasible_actions", 0)),
                        "rejected_infeasible_insertion": int(child_action_diag.get("rejected_infeasible_insertion", 0)),
                        "action_cache_hit": bool(child_action_diag.get("action_cache_hit", False)),
                        "wavefront_cache_hit": bool(child_action_diag.get("wavefront_cache_hit", False)),
                        "timing_ms": dict(child_action_diag.get("timing_ms", {})),
                    },
                }
            )
            if child_metrics.solved:
                solved_node_id = child_id
                break
        search_loop_ms = _elapsed_ms(t_search_loop_ns)

        best_node_id = solved_node_id if solved_node_id is not None else self._best_node_id(nodes)
        plan_actions, plan_metrics, plan_node_ids = self._recover_chain(nodes, best_node_id)

        # Final replay from root for deterministic verification video.
        t_replay_ns = time.perf_counter_ns()
        replay_frames = []
        self.raw.set_state(root_state.clone())
        replay_frames.append(capture_rgb_frame(self.env))

        def _replay_cb(step, obs, rew):
            if step % max(1, self.cfg.visual.replay_stride) == 0:
                replay_frames.append(capture_rgb_frame(self.env))

        for action in plan_actions:
            self.executor.execute(action, step_callback=_replay_cb)
            replay_frames.append(capture_rgb_frame(self.env))

        save_video(artifacts / "final_replay.mp4", replay_frames, fps=20)
        replay_video_ms = _elapsed_ms(t_replay_ns)

        timing_summary: dict | None = None
        if timing_enabled:
            timing_summary = {
                "run_total_ms": _elapsed_ms(t_run_ns),
                "search_loop_ms": search_loop_ms,
                "root_artifact_io_ms": root_artifact_io_ms,
                "replay_video_ms": replay_video_ms,
                "action_generation_calls": int(len(action_gen_rows)),
                "action_generation_action_cache_hits": int(action_gen_cache_hits),
                "action_generation_wavefront_cache_hits": int(action_gen_wavefront_cache_hits),
                "action_generation_ms": {
                    "total": _stats_ms(action_gen_total_ms),
                    "sampling": _stats_ms(action_gen_sampling_ms),
                    "wavefront_build": _stats_ms(action_gen_wavefront_build_ms),
                    "insertion_filter": _stats_ms(action_gen_insertion_filter_ms),
                    "insertion_solver_only": _stats_ms(action_gen_insertion_solver_ms),
                    "cache_clone": _stats_ms(action_gen_cache_clone_ms),
                    "shuffle": _stats_ms(action_gen_shuffle_ms),
                },
                "expansion_ms": {
                    "total": _stats_ms(expansion_total_ms_values),
                    "frontier_expandable_scan": _stats_ms(frontier_expandable_scan_ms_values),
                    "frontier_select": _stats_ms(frontier_select_ms_values),
                    "frontier_candidate_count": _stats_scalar(frontier_candidate_count_values),
                    "prep": _stats_ms(expansion_prep_ms_values),
                    "execute": _stats_ms(execution_ms_values),
                    "execute_internal_total": _stats_ms(execution_internal_total_ms_values),
                    "child_eval": _stats_ms(expansion_child_eval_ms_values),
                    "child_action_generation": _stats_ms(expansion_child_action_gen_ms_values),
                    "artifact_io": _stats_ms(expansion_artifact_io_ms_values),
                },
                "execution_phase_ms": {
                    "stage_plan": _stats_ms(execution_phase_stage_plan_ms_values),
                    "stage_execute": _stats_ms(execution_phase_stage_execute_ms_values),
                    "entry_move": _stats_ms(execution_phase_entry_ms_values),
                    "sweep_move": _stats_ms(execution_phase_sweep_ms_values),
                    "retract_nudge": _stats_ms(execution_phase_retract_nudge_ms_values),
                    "retract_pull": _stats_ms(execution_phase_retract_pull_ms_values),
                    "return_to_rest": _stats_ms(execution_phase_return_to_rest_ms_values),
                    "motion_total_excluding_stage_plan": _stats_ms(
                        execution_phase_motion_total_excl_plan_ms_values
                    ),
                    "motion_total_including_stage_plan": _stats_ms(
                        execution_phase_motion_total_incl_plan_ms_values
                    ),
                },
                "reject_reason_counts": reject_reason_counts,
            }
            save_json(
                artifacts / "timing_diagnostics.json",
                {
                    "timing_summary": timing_summary,
                    "action_generation": action_gen_rows,
                    "expansions": expansion_timing_rows,
                },
            )

        save_json(
            artifacts / "final_plan.json",
            {
                "success": solved_node_id is not None,
                "leaf_node_id": best_node_id,
                "node_path": plan_node_ids,
                "actions": [serialize_action(a) for a in plan_actions],
                "metrics_path": [serialize_metrics(m) for m in plan_metrics],
            },
        )

        summary = {
            "success": solved_node_id is not None,
            "solved_node_id": solved_node_id,
            "best_node_id": best_node_id,
            "expansions": expansions,
            "executions": executions,
            "solve_depth": nodes[best_node_id].depth,
            "num_nodes": len(nodes),
            "artifact_dir": str(artifacts),
            "parallel_frontier": {
                "enabled": bool(parallel_frontier_enabled),
                "batch_size": int(parallel_frontier_batch_size),
                "execution_mode": (
                    "batched_env_gpu_parallel"
                    if (
                        parallel_frontier_enabled
                        and parallel_frontier_batch_size > 1
                        and self.batch_executor is not None
                    )
                    else "sequential_single_env"
                ),
            },
            "root_metrics": serialize_metrics(root_metrics),
            "best_metrics": serialize_metrics(nodes[best_node_id].metrics),
        }
        if timing_enabled and timing_summary is not None:
            summary["timing"] = timing_summary
        save_json(artifacts / "summary.json", summary)

        return PlannerRunResult(
            success=solved_node_id is not None,
            solved_node_id=solved_node_id,
            expansions=expansions,
            executions=executions,
            solve_depth=nodes[best_node_id].depth,
            artifact_dir=str(Path(artifacts).resolve()),
            plan_actions=plan_actions,
            plan_metrics=plan_metrics,
            info=summary,
        )
