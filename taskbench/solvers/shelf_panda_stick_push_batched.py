"""GPU-batched shelf push solver using StickPush skill.

Runs N parallel environments with multi-push episodes.
Each push: stage → insert → sweep → nudge-back → retract → pullback+rest.

Usage:
    uv run python -m taskbench.run task=shelf_panda_stick \
        solver=shelf_panda_stick_push_batched \
        runtime.num_envs=64 run.num_episodes=5 \
        run.solver_kwargs.pushes_per_episode=3
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from taskbench.skills.curobo_motion import (
    get_arm_joint_names,
    setup_curobo_planner,
)
from taskbench.skills.stick_push import StickPush, StickPushTraceDebug
from taskbench.skills.stick_push_trace_viz import (
    build_trace_viz_spec,
    render_trace_viz_image,
    save_trace_viz_image,
)
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.shelf_panda_stick_push_batched")

ROBOT_UID = "panda_stick_long"
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]
REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
SAFE_QPOS = [0.0002, -1.2996, 0.0003, -2.4930, -0.0009, 1.0177, 0.0]

import taskbench.agents.panda_stick_long  # noqa: F401


def _sample_pushes(rng, N, shelf):
    """Sample N random push targets."""
    margin, y_margin = 0.02, 0.04
    surface_z = shelf["floor_z"] + shelf["thickness"]
    # Push below the cylinder COM (4.5cm) to reduce tipping torque while
    # keeping margin above the shelf floor (stick radius = 5mm).
    push_z = surface_z + 0.025
    max_x = shelf["front_x"] + shelf["depth"] - margin

    p1 = np.zeros((N, 3))
    p2 = np.zeros((N, 3))
    for i in range(N):
        push_type = rng.random()
        if push_type < 0.4:
            # Horizontal
            x = rng.uniform(shelf["front_x"] + margin, max_x)
            y1 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            while abs(y2 - y1) < 0.06:
                y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            p1[i] = [x, y1, push_z]
            p2[i] = [x, y2, push_z]
        elif push_type < 0.7:
            # Diagonal
            x1 = rng.uniform(shelf["front_x"] + margin, max_x)
            y1 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            x2 = rng.uniform(shelf["front_x"] + margin, max_x)
            y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            while (x2 - x1) ** 2 + (y2 - y1) ** 2 < 0.06 ** 2:
                x2 = rng.uniform(shelf["front_x"] + margin, max_x)
                y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            p1[i] = [x1, y1, push_z]
            p2[i] = [x2, y2, push_z]
        else:
            # Depth
            y = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            x1 = rng.uniform(shelf["front_x"] + margin, shelf["front_x"] + shelf["depth"] * 0.5)
            x2 = rng.uniform(x1 + 0.04, max_x)
            p1[i] = [x1, y, push_z]
            p2[i] = [x2, y, push_z]
    return p1, p2


@register_solver("shelf_panda_stick_push_batched")
class ShelfPandaStickPushBatchedSolver(BaseSolver):
    """GPU-batched multi-push solver using StickPush skill.

    Runs N envs in parallel, multiple pushes per episode.
    Cylinders are randomized per env (count + placement).
    """

    def __init__(
        self,
        pushes_per_episode: int = 3,
        trace_viz_enabled: bool = False,
        trace_viz_root: str = "artifacts/stickpush_trace",
        trace_viz_max_envs_per_push: int = 4,
        trace_viz_save_only_violations: bool = True,
        trace_viz_sample_stride: int = 2,
    ):
        self.pushes_per_episode = int(pushes_per_episode)
        self.trace_viz_enabled = bool(trace_viz_enabled)
        self.trace_viz_root = str(trace_viz_root)
        self.trace_viz_max_envs_per_push = max(1, int(trace_viz_max_envs_per_push))
        self.trace_viz_save_only_violations = bool(trace_viz_save_only_violations)
        self.trace_viz_sample_stride = max(1, int(trace_viz_sample_stride))
        self._trace_viz_run_dir: Path | None = None

    def _active_cylinders_xy_r(self, raw, env_idx: int) -> np.ndarray:
        """Return active cylinder disks for one env as (M, 3): x, y, radius."""
        if not hasattr(raw, "shelf_objects") or not hasattr(raw, "cyl_spec"):
            return np.zeros((0, 3), dtype=np.float32)
        radius = float(getattr(raw.cyl_spec, "radius", 0.018))
        rows: list[list[float]] = []
        for actor in raw.shelf_objects:
            p = actor.pose.p[int(env_idx)].detach().cpu().numpy()
            if float(p[0]) < 2.0:
                rows.append([float(p[0]), float(p[1]), radius])
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _select_trace_viz_envs(self, trace_debug: StickPushTraceDebug) -> list[int]:
        """Choose envs to render: violations first, then largest offtrack."""
        if trace_debug.entry.metrics is None or trace_debug.sweep.metrics is None or trace_debug.retract.metrics is None:
            return []
        entry_max = trace_debug.entry.metrics.max_offtrack.detach().cpu().numpy()
        sweep_max = trace_debug.sweep.metrics.max_offtrack.detach().cpu().numpy()
        retract_max = trace_debug.retract.metrics.max_offtrack.detach().cpu().numpy()
        score = np.maximum(np.maximum(entry_max, sweep_max), retract_max)

        entry_bad = trace_debug.entry.metrics.offtrack_violation.detach().cpu().numpy().astype(bool)
        sweep_bad = trace_debug.sweep.metrics.offtrack_violation.detach().cpu().numpy().astype(bool)
        retract_bad = trace_debug.retract.metrics.offtrack_violation.detach().cpu().numpy().astype(bool)
        any_bad = entry_bad | sweep_bad | retract_bad

        if self.trace_viz_save_only_violations and bool(any_bad.any()):
            candidates = np.where(any_bad)[0]
        else:
            candidates = np.arange(score.shape[0], dtype=int)
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(-score[candidates])]
        return [int(x) for x in order[: self.trace_viz_max_envs_per_push]]

    def _save_trace_viz_push(
        self,
        *,
        raw,
        shelf_geom,
        seed: int | None,
        push_idx: int,
        trace_debug: StickPushTraceDebug,
    ) -> None:
        """Render and save per-env planned-vs-actual trace images for one push."""
        if self._trace_viz_run_dir is None:
            return
        push_dir = self._trace_viz_run_dir / f"push_{int(push_idx):03d}"
        push_dir.mkdir(parents=True, exist_ok=True)
        selected_envs = self._select_trace_viz_envs(trace_debug)
        rows = []
        for env_idx in selected_envs:
            blockers = self._active_cylinders_xy_r(raw, env_idx)
            spec = build_trace_viz_spec(
                env_idx=env_idx,
                trace_debug=trace_debug,
                shelf_front_x=float(shelf_geom.front_x),
                shelf_back_x=float(shelf_geom.back_x),
                shelf_half_w=float(shelf_geom.half_w),
                blocker_disks_xy_r=blockers,
            )
            image = render_trace_viz_image(spec)
            image_path = push_dir / f"env_{int(env_idx):04d}.png"
            save_trace_viz_image(image_path, image)
            rows.append(
                {
                    "env_idx": int(env_idx),
                    "seed": (None if seed is None else int(seed)),
                    "push_idx": int(push_idx),
                    "image_path": str(image_path),
                    "entry_max_offtrack": float(trace_debug.entry.metrics.max_offtrack[env_idx].item()),
                    "sweep_max_offtrack": float(trace_debug.sweep.metrics.max_offtrack[env_idx].item()),
                    "retract_max_offtrack": float(trace_debug.retract.metrics.max_offtrack[env_idx].item()),
                    "entry_violation": bool(trace_debug.entry.metrics.offtrack_violation[env_idx].item()),
                    "sweep_violation": bool(trace_debug.sweep.metrics.offtrack_violation[env_idx].item()),
                    "retract_violation": bool(trace_debug.retract.metrics.offtrack_violation[env_idx].item()),
                }
            )
        manifest_path = push_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump({"push_idx": int(push_idx), "env_rows": rows}, f, indent=2, sort_keys=True)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        raw = env.unwrapped
        device = raw.device
        cuda = torch.device("cuda:0")
        N = raw.num_envs
        n_arm = len(get_arm_joint_names(ROBOT_UID))
        rng = np.random.default_rng(seed)

        # Set rest qpos
        rest = torch.tensor(REST_QPOS, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[:, :n_arm] = rest
        raw.agent.robot.set_qpos(qpos)
        raw.agent.set_control_mode("pd_joint_pos")
        raw.agent.controller.reset()
        for _ in range(10):
            env.step(rest)

        # Shelf collision box for cuRobo
        from curobo.geom.types import Cuboid, WorldConfig
        g = raw.shelf_geom
        robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
        sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
        sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
        world = WorldConfig(cuboid=[Cuboid(
            name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
        )])

        os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")
        motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world, n_envs=1, warmup=False)
        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

        # Create skill
        push_skill = StickPush(
            env, motion_gen,
            robot_uid=ROBOT_UID, n_envs=N, n_arm_joints=n_arm,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )
        self._trace_viz_run_dir = None
        if self.trace_viz_enabled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_tag = f"seed{seed}" if seed is not None else "seednone"
            self._trace_viz_run_dir = Path(self.trace_viz_root) / f"{ts}_{seed_tag}_N{int(N)}"
            self._trace_viz_run_dir.mkdir(parents=True, exist_ok=True)

        # Shelf config for push sampling
        shelf_dict = dict(
            front_x=g.front_x, depth=g.depth, half_w=g.half_w,
            floor_z=g.floor_z, thickness=g.thickness, inner_h=g.inner_h,
        )

        total_success = 0
        total_pushes = 0

        for push_idx in range(self.pushes_per_episode):
            push_rng = np.random.default_rng(seed + push_idx * 100 if seed else push_idx)
            p1_np, p2_np = _sample_pushes(push_rng, N, shelf_dict)
            approach_np = np.stack([
                np.full(N, g.front_x - 0.05), p1_np[:, 1], p1_np[:, 2],
            ], axis=1)
            retract_np = np.stack([
                np.full(N, g.front_x - 0.30), p2_np[:, 1], p2_np[:, 2],
            ], axis=1)

            result = push_skill(
                approach_positions=torch.tensor(approach_np, device=cuda, dtype=torch.float32),
                approach_quaternions=torch.tensor(Q_INTO_SHELF, device=cuda, dtype=torch.float32).unsqueeze(0).expand(N, -1),
                entry_positions=torch.tensor(p1_np, device=cuda, dtype=torch.float32),
                sweep_positions=torch.tensor(p2_np, device=cuda, dtype=torch.float32),
                retract_positions=torch.tensor(retract_np, device=cuda, dtype=torch.float32),
                collect_trace_debug=self.trace_viz_enabled,
                trace_sample_stride=self.trace_viz_sample_stride,
            )
            if self.trace_viz_enabled and result.trace_debug is not None:
                try:
                    self._save_trace_viz_push(
                        raw=raw,
                        shelf_geom=g,
                        seed=seed,
                        push_idx=push_idx,
                        trace_debug=result.trace_debug,
                    )
                except Exception:
                    logger.exception("Trace-viz artifact write failed for push %d", push_idx)

            n_ok = result.success_mask.sum().item()
            total_success += n_ok
            total_pushes += N
            logger.info("Push %d/%d: %d/%d success, sweep_dist=%.4f, orient=%.4f",
                        push_idx + 1, self.pushes_per_episode, n_ok, N,
                        result.sweep_final_dist.mean().item(),
                        result.orientation_drift.mean().item())

        success_rate = total_success / max(total_pushes, 1)
        return SolverResult(
            success=success_rate > 0.5,
            elapsed_steps=total_pushes,
            info={
                "n_success": total_success,
                "n_total": total_pushes,
                "success_rate": success_rate,
                "trace_viz_dir": (
                    None if self._trace_viz_run_dir is None else str(self._trace_viz_run_dir)
                ),
            },
        )
