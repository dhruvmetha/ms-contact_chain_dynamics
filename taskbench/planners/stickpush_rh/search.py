"""Receding-horizon search loop for shelf stick pushing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig
from taskbench.planners.stickpush_rh.executor import StickPushExecutor
from taskbench.planners.stickpush_rh.frontier_ucb import FrontierUCB
from taskbench.planners.stickpush_rh.geometry import (
    compute_node_metrics,
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
        cfg: RecedingHorizonConfig,
        state_provider: GTSceneStateProvider | None = None,
        sampler: StickPushSampler | None = None,
        ucb: FrontierUCB | None = None,
    ):
        self.env = env
        self.raw = env.unwrapped
        self.cfg = cfg
        self.executor = executor
        self.state_provider = state_provider or GTSceneStateProvider(env_index=0)
        self.sampler = sampler or StickPushSampler(cfg.sampling)
        self.ucb = ucb or FrontierUCB(cfg.ucb)
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

        # Front side is open; only side/back walls are considered.
        side_wall_dist = float(scene.shelf_half_w - abs(float(grasp_center_xy[1])))
        back_wall_dist = float(scene.shelf_back_x - float(grasp_center_xy[0]))
        wall_radius = float(self.cfg.sampling.clearance_radius)
        wall_in_radius = bool((side_wall_dist <= wall_radius) or (back_wall_dist <= wall_radius))

        return {
            "target_xy": target_xy.tolist(),
            "grasp_center_xy": grasp_center_xy.tolist(),
            "target_shift_xy": shift_xy,
            "target_shift_ok": bool(shift_ok),
            "side_wall_dist": side_wall_dist,
            "back_wall_dist": back_wall_dist,
            "wall_radius_threshold": wall_radius,
            "wall_in_radius": wall_in_radius,
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

    def _build_node_actions(
        self,
        scene,
        *,
        state_hash: str,
        rng: np.random.Generator,
    ) -> tuple[list[PlannerAction], dict]:
        if not self.cfg.sampling.use_wavefront_insertion_solver:
            sampled_actions = self.sampler.sample_actions(scene)
            diagnostics = {
                "num_sampled_actions": int(len(sampled_actions)),
                "num_feasible_actions": int(len(sampled_actions)),
                "wavefront_used": False,
                "wavefront_cache_hit": False,
                "rejected_infeasible_insertion": 0,
            }
            return self._shuffle_actions(sampled_actions, rng), diagnostics

        if state_hash in self._feasible_action_cache:
            cached = [self._clone_action(a) for a in self._feasible_action_cache[state_hash]]
            diagnostics = dict(
                self._action_diag_cache.get(
                    state_hash,
                    {
                        "num_sampled_actions": int(len(cached)),
                        "num_feasible_actions": int(len(cached)),
                        "wavefront_used": True,
                        "wavefront_cache_hit": True,
                        "rejected_infeasible_insertion": 0,
                    },
                )
            )
            diagnostics["wavefront_cache_hit"] = True
            return self._shuffle_actions(cached, rng), diagnostics

        sampled_actions = self.sampler.sample_actions(scene)
        diagnostics = {
            "num_sampled_actions": int(len(sampled_actions)),
            "num_feasible_actions": int(len(sampled_actions)),
            "wavefront_used": True,
            "wavefront_cache_hit": False,
            "rejected_infeasible_insertion": 0,
        }

        if state_hash in self._wavefront_cache:
            wavefront = self._wavefront_cache[state_hash]
            diagnostics["wavefront_cache_hit"] = True
        else:
            wavefront = build_wavefront_grid(scene, self.cfg.sampling)
            self._wavefront_cache[state_hash] = wavefront

        feasible_actions: list[PlannerAction] = []
        for action in sampled_actions:
            plan = solve_straight_insertion(
                wavefront,
                scene,
                action.entry_xyz[:2],
                self.cfg.sampling,
            )
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

        diagnostics["num_feasible_actions"] = int(len(feasible_actions))
        diagnostics["rejected_infeasible_insertion"] = int(len(sampled_actions) - len(feasible_actions))
        self._feasible_action_cache[state_hash] = [self._clone_action(a) for a in feasible_actions]
        self._action_diag_cache[state_hash] = dict(diagnostics)
        shuffled = self._shuffle_actions([self._clone_action(a) for a in feasible_actions], rng)
        return shuffled, diagnostics

    def run(self, seed: int | None = None) -> PlannerRunResult:
        rng = np.random.default_rng(seed)
        artifacts = build_artifact_dir(self.cfg.artifact_root, seed)
        root_state = self.raw.get_state().clone()

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

        while solved_node_id is None and executions < self.cfg.max_executions:
            candidate_ids = self._expandable_node_ids(nodes)
            if not candidate_ids:
                break

            node_id = self.ucb.select(nodes, candidate_ids)
            node = nodes[node_id]

            self.raw.set_state(node.sim_state.clone())
            scene_before = self.state_provider.get_scene_state(self.env)
            before_frame = capture_rgb_frame(self.env)

            if not node.untried_actions:
                continue
            action = node.untried_actions.pop(0)

            exec_result = self.executor.execute(action)
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

            if reject_reason is not None:
                # Reject samples that fail to reach insertion waypoint.
                self._backprop(nodes, node_id, reward=0.0)
                expansions += 1
                executions += 1
                exp_dir = artifacts / f"exp_{expansions:06d}"

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
                "rejected_infeasible_insertion": 0,
            }
            if not child_metrics.solved and child_depth < self.cfg.max_depth:
                if (not self.cfg.prune_target_invalid_nodes) or child_goal_diag["target_shift_ok"]:
                    child_state_hash = hash_scene_state(scene_after)
                    child_actions, child_action_diag = self._build_node_actions(
                        scene_after,
                        state_hash=child_state_hash,
                        rng=rng,
                    )
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

            reward = progress_reward(
                node.metrics,
                child_metrics,
                solved_bonus=self.cfg.ucb.solved_bonus,
            )
            self._backprop(nodes, node_id, reward)

            expansions += 1
            executions += 1
            exp_dir = artifacts / f"exp_{expansions:06d}"

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

            if child_metrics.solved:
                solved_node_id = child_id
                break

        best_node_id = solved_node_id if solved_node_id is not None else self._best_node_id(nodes)
        plan_actions, plan_metrics, plan_node_ids = self._recover_chain(nodes, best_node_id)

        # Final replay from root for deterministic verification video.
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
            "root_metrics": serialize_metrics(root_metrics),
            "best_metrics": serialize_metrics(nodes[best_node_id].metrics),
        }
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
