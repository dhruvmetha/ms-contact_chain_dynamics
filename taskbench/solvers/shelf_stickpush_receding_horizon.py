"""Receding-horizon planner solver for shelf Panda stick pushing."""

from __future__ import annotations

import torch

from taskbench.planners.stickpush_rh.config import (
    RecedingHorizonConfig,
    SamplingConfig,
    UCBConfig,
    VisualConfig,
)
from taskbench.planners.stickpush_rh.executor import StickPushExecutor
from taskbench.planners.stickpush_rh.search import StickPushRecedingHorizonSearch
from taskbench.planners.stickpush_rh.state_provider import GTSceneStateProvider
from taskbench.skills.curobo_motion import get_arm_joint_names, setup_curobo_planner
from taskbench.skills.stick_push import StickPush
from taskbench.solver import BaseSolver, SolverResult, register_solver
from taskbench.solvers.shelf_panda_stick_push_batched import (
    Q_INTO_SHELF,
    REST_QPOS,
    ROBOT_UID,
    SAFE_QPOS,
)

import taskbench.agents.panda_stick_long  # noqa: F401


@register_solver("shelf_stickpush_receding_horizon")
class ShelfStickPushRecedingHorizonSolver(BaseSolver):
    """Shelf stick solver using UCB-guided receding horizon search."""

    def __init__(
        self,
        *,
        max_executions: int = 1000,
        max_depth: int = 256,
        clearance_radius: float = 0.10,
        heading_degrees: list[float] | None = None,
        x_approach_values: list[float] | None = None,
        insertion_approach_backoff: float = 0.07,
        delta_len_fracs: list[float] | None = None,
        contact_margin: float = 0.003,
        stick_radius: float = 0.004,
        min_push_len: float = 0.03,
        min_insertion_depth: float = 0.02,
        retract_backoff: float = 0.30,
        z_eps: float = 0.03,
        forbid_target_crossing: bool = True,
        target_avoid_margin: float = 0.004,
        include_target_pushes: bool = True,
        target_push_open_dir_bias: float = 0.5,
        target_push_direction_deltas_deg: list[float] | None = None,
        target_push_include_open_face_bias: bool = True,
        target_push_top_k_directions: int = 3,
        target_push_wall_weight: float = 1.0,
        target_push_corridor_weight: float = 0.8,
        target_push_wall_penalty_weight: float = 0.05,
        use_wavefront_insertion_solver: bool = True,
        insertion_grid_resolution: float = 0.003,
        insertion_grid_safety_eps: float = 0.0,
        insertion_endpoint_tolerance_cells: int = 2,
        insertion_max_source_trials: int = 64,
        max_target_shift_xy: float = 0.02,
        target_wall_margin: float = 0.01,
        prune_target_invalid_nodes: bool = False,
        require_target_shift_limit_for_success: bool = False,
        ucb_exploration_c: float = 1.2,
        ucb_untried_bonus: float = 1.0,
        ucb_solved_bonus: float = 10.0,
        visual_max_candidates_drawn: int = 300,
        visual_save_images: bool = True,
        visual_save_frontier_csv: bool = True,
        visual_replay_stride: int = 2,
        diagnostics_timing_enabled: bool = False,
        entry_max_steps: int = 200,
        sweep_max_steps: int = 200,
        retract_max_steps: int = 100,
        rest_steps: int = 60,
        convergence_threshold: float = 0.01,
        staging_max_attempts: int = 2,
        staging_timeout: float = 3.0,
        staging_fallback_enabled: bool = True,
        staging_backoff_candidates: list[float] | None = None,
        staging_fallback_min_insertion_depth: float = 0.015,
        artifact_root: str = "artifacts/stickpush_rh",
    ):
        self.max_executions = int(max_executions)
        self.max_depth = int(max_depth)
        self.clearance_radius = float(clearance_radius)
        self.heading_degrees = heading_degrees
        self.x_approach_values = x_approach_values
        self.insertion_approach_backoff = float(insertion_approach_backoff)
        self.delta_len_fracs = delta_len_fracs
        self.contact_margin = float(contact_margin)
        self.stick_radius = float(stick_radius)
        self.min_push_len = float(min_push_len)
        self.min_insertion_depth = float(min_insertion_depth)
        self.retract_backoff = float(retract_backoff)
        self.z_eps = float(z_eps)
        self.forbid_target_crossing = bool(forbid_target_crossing)
        self.target_avoid_margin = float(target_avoid_margin)
        self.include_target_pushes = bool(include_target_pushes)
        self.target_push_open_dir_bias = float(target_push_open_dir_bias)
        self.target_push_direction_deltas_deg = (
            list(target_push_direction_deltas_deg)
            if target_push_direction_deltas_deg is not None
            else list(SamplingConfig().target_push_direction_deltas_deg)
        )
        self.target_push_include_open_face_bias = bool(target_push_include_open_face_bias)
        self.target_push_top_k_directions = int(target_push_top_k_directions)
        self.target_push_wall_weight = float(target_push_wall_weight)
        self.target_push_corridor_weight = float(target_push_corridor_weight)
        self.target_push_wall_penalty_weight = float(target_push_wall_penalty_weight)
        self.use_wavefront_insertion_solver = bool(use_wavefront_insertion_solver)
        self.insertion_grid_resolution = float(insertion_grid_resolution)
        self.insertion_grid_safety_eps = float(insertion_grid_safety_eps)
        self.insertion_endpoint_tolerance_cells = int(insertion_endpoint_tolerance_cells)
        self.insertion_max_source_trials = int(insertion_max_source_trials)
        self.max_target_shift_xy = float(max_target_shift_xy)
        self.target_wall_margin = float(target_wall_margin)
        self.prune_target_invalid_nodes = bool(prune_target_invalid_nodes)
        self.require_target_shift_limit_for_success = bool(require_target_shift_limit_for_success)
        self.ucb_exploration_c = float(ucb_exploration_c)
        self.ucb_untried_bonus = float(ucb_untried_bonus)
        self.ucb_solved_bonus = float(ucb_solved_bonus)
        self.visual_max_candidates_drawn = int(visual_max_candidates_drawn)
        self.visual_save_images = bool(visual_save_images)
        self.visual_save_frontier_csv = bool(visual_save_frontier_csv)
        self.visual_replay_stride = int(visual_replay_stride)
        self.diagnostics_timing_enabled = bool(diagnostics_timing_enabled)
        self.entry_max_steps = int(entry_max_steps)
        self.sweep_max_steps = int(sweep_max_steps)
        self.retract_max_steps = int(retract_max_steps)
        self.rest_steps = int(rest_steps)
        self.convergence_threshold = float(convergence_threshold)
        self.staging_max_attempts = int(staging_max_attempts)
        self.staging_timeout = float(staging_timeout)
        self.staging_fallback_enabled = bool(staging_fallback_enabled)
        self.staging_backoff_candidates = (
            list(staging_backoff_candidates) if staging_backoff_candidates is not None else [0.05, 0.03, 0.09, 0.11]
        )
        self.staging_fallback_min_insertion_depth = float(staging_fallback_min_insertion_depth)
        self.artifact_root = artifact_root

    def _build_planner_config(self) -> RecedingHorizonConfig:
        approach_backoff = self.insertion_approach_backoff
        if self.x_approach_values is not None and len(self.x_approach_values) > 0:
            # Backward compatibility: legacy x_approach list now maps to one deterministic backoff.
            approach_backoff = float(self.x_approach_values[0])

        sampling = SamplingConfig(
            clearance_radius=self.clearance_radius,
            heading_degrees=tuple(self.heading_degrees) if self.heading_degrees is not None else SamplingConfig().heading_degrees,
            x_approach_values=tuple(self.x_approach_values) if self.x_approach_values is not None else SamplingConfig().x_approach_values,
            insertion_approach_backoff=approach_backoff,
            delta_len_fracs=tuple(self.delta_len_fracs) if self.delta_len_fracs is not None else SamplingConfig().delta_len_fracs,
            contact_margin=self.contact_margin,
            stick_radius=self.stick_radius,
            retract_backoff=self.retract_backoff,
            z_eps=self.z_eps,
            forbid_target_crossing=self.forbid_target_crossing,
            target_avoid_margin=self.target_avoid_margin,
            include_target_pushes=self.include_target_pushes,
            target_push_open_dir_bias=self.target_push_open_dir_bias,
            target_push_direction_deltas_deg=tuple(self.target_push_direction_deltas_deg),
            target_push_include_open_face_bias=self.target_push_include_open_face_bias,
            target_push_top_k_directions=self.target_push_top_k_directions,
            target_push_wall_weight=self.target_push_wall_weight,
            target_push_corridor_weight=self.target_push_corridor_weight,
            target_push_wall_penalty_weight=self.target_push_wall_penalty_weight,
            use_wavefront_insertion_solver=self.use_wavefront_insertion_solver,
            insertion_grid_resolution=self.insertion_grid_resolution,
            insertion_grid_safety_eps=self.insertion_grid_safety_eps,
            insertion_endpoint_tolerance_cells=self.insertion_endpoint_tolerance_cells,
            insertion_max_source_trials=self.insertion_max_source_trials,
            min_push_len=self.min_push_len,
            min_insertion_depth=self.min_insertion_depth,
        )
        ucb = UCBConfig(
            exploration_c=self.ucb_exploration_c,
            untried_bonus=self.ucb_untried_bonus,
            solved_bonus=self.ucb_solved_bonus,
        )
        visual = VisualConfig(
            save_expansion_images=self.visual_save_images,
            save_frontier_csv=self.visual_save_frontier_csv,
            replay_stride=self.visual_replay_stride,
            max_candidates_drawn=self.visual_max_candidates_drawn,
            save_timing_diagnostics=self.diagnostics_timing_enabled,
        )
        return RecedingHorizonConfig(
            max_executions=self.max_executions,
            max_depth=self.max_depth,
            artifact_root=self.artifact_root,
            max_target_shift_xy=self.max_target_shift_xy,
            target_wall_margin=self.target_wall_margin,
            prune_target_invalid_nodes=self.prune_target_invalid_nodes,
            require_target_shift_limit_for_success=self.require_target_shift_limit_for_success,
            sampling=sampling,
            ucb=ucb,
            visual=visual,
        )

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        raw = env.unwrapped
        device = raw.device
        cuda_device = torch.device("cuda:0")
        n_arm = len(get_arm_joint_names(ROBOT_UID))

        # Rest pose initialization matches existing shelf stick solvers.
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32)
        raw.agent.robot.set_qpos(qpos)
        raw.agent.set_control_mode("pd_joint_pos")
        raw.agent.controller.reset()
        rest_action = torch.zeros(1, n_arm, device=device)
        rest_action[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32, device=device)
        for _ in range(10):
            env.step(rest_action)

        # Shelf collision world, same convention as current stick solvers.
        from curobo.geom.types import Cuboid, WorldConfig

        g = raw.shelf_geom
        robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
        shelf_center = [float(g.center_x - robot_base[0]), float(-robot_base[1]), float(g.ceil_z / 2 - robot_base[2])]
        shelf_dims = [float(g.depth + 0.04), float(2 * g.half_w + 0.04), float(g.ceil_z + 0.04)]
        world = WorldConfig(
            cuboid=[
                Cuboid(
                    name="shelf",
                    pose=[shelf_center[0], shelf_center[1], shelf_center[2], 1, 0, 0, 0],
                    dims=shelf_dims,
                )
            ]
        )
        motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world, n_envs=1, warmup=False)
        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda_device).unsqueeze(0)

        push_skill = StickPush(
            env,
            motion_gen,
            robot_uid=ROBOT_UID,
            n_envs=1,
            n_arm_joints=n_arm,
            rest_qpos=REST_QPOS,
            robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        executor = StickPushExecutor(
            push_skill,
            approach_quaternion_wxyz=Q_INTO_SHELF,
            cuda_device=cuda_device,
            timing_enabled=self.diagnostics_timing_enabled,
            entry_max_steps=self.entry_max_steps,
            sweep_max_steps=self.sweep_max_steps,
            retract_max_steps=self.retract_max_steps,
            rest_steps=self.rest_steps,
            convergence_threshold=self.convergence_threshold,
            staging_max_attempts=self.staging_max_attempts,
            staging_timeout=self.staging_timeout,
            staging_fallback_enabled=self.staging_fallback_enabled,
            staging_backoff_candidates=tuple(self.staging_backoff_candidates),
            staging_fallback_min_insertion_depth=self.staging_fallback_min_insertion_depth,
        )
        planner_cfg = self._build_planner_config()
        planner = StickPushRecedingHorizonSearch(
            env,
            executor=executor,
            cfg=planner_cfg,
            state_provider=GTSceneStateProvider(env_index=0),
        )

        run_result = planner.run(seed=seed)
        return SolverResult(
            success=run_result.success,
            elapsed_steps=run_result.executions,
            info={
                "solve_depth": run_result.solve_depth,
                "expansions": run_result.expansions,
                "executions": run_result.executions,
                "artifact_dir": run_result.artifact_dir,
                "solved_node_id": run_result.solved_node_id,
                "plan_length": len(run_result.plan_actions),
            },
        )
