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

import numpy as np
import torch

from taskbench.skills.curobo_motion import (
    get_arm_joint_names,
    setup_curobo_planner,
)
from taskbench.skills.stick_push import StickPush
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
    push_z = surface_z + 0.045
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

    def __init__(self, pushes_per_episode: int = 3):
        self.pushes_per_episode = int(pushes_per_episode)

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
            )

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
            },
        )
