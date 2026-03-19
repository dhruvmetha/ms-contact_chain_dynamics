"""Batched skill context for GPU-parallel operation.

Analogous to ``SkillContext`` but for N parallel environments using cuRobo
instead of mplib. The cuRobo ``MotionGen`` is created once at construction
(warmup is expensive) and persisted across episodes — only obstacle poses
are updated per-episode via ``update_world()``.

Usage::

    ctx = BatchedSkillContext(env, robot_uid="panda", n_envs=256)
    ctx.reset(seed=42)
    result = ctx.push(approach_pos, approach_quat, push_pos, push_quat)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from taskbench.envs import get_objects
from taskbench.skills.batched_primitives import BatchedMove, BatchedPush
from taskbench.skills.curobo_motion import (
    setup_curobo_planner,
    update_world,
)
from taskbench.skills.curobo_world import build_world_configs
from taskbench.skills.robot_config import RobotConfig, get_robot_config

logger = logging.getLogger("taskbench.skills.batched_context")


class BatchedSkillContext:
    """Shared context for batched skill-based solvers.

    Holds the GPU-vectorized env, cuRobo MotionGen, robot config,
    and pre-bound batched skill instances.

    The planner is created at construction time (expensive warmup) and
    reused across episodes. Call ``reset()`` to re-initialize environments
    and update the collision world.

    Args:
        env: GPU-vectorized ManiSkill env (via ``make_env()``).
        robot_uid: Robot identifier (e.g. "panda").
        n_envs: Number of parallel environments.
        interpolation_dt: Timestep for cuRobo trajectory interpolation.
        step_callback: Optional callable invoked after each env.step().
    """

    def __init__(
        self,
        env,
        *,
        robot_uid: str = "panda",
        n_envs: int = 1,
        interpolation_dt: float = 0.02,
        step_callback: Optional[Callable] = None,
    ):
        self.env = env
        self.robot_uid = robot_uid
        self.n_envs = n_envs
        self.step_callback = step_callback

        # Get robot config from the env
        from taskbench.skills.curobo_motion import get_arm_joint_names

        self.robot_config: RobotConfig = get_robot_config(env)
        self.n_arm_joints = len(get_arm_joint_names(robot_uid))

        # Build initial world config (will be updated per-episode in reset())
        world_configs = build_world_configs(env, n_envs)

        # Create cuRobo planner (expensive — done once)
        self.motion_gen = setup_curobo_planner(
            robot_uid,
            world_configs=world_configs,
            n_envs=n_envs,
            interpolation_dt=interpolation_dt,
        )

        self.objects: dict[str, object] = {}
        self._build_skills()

    def _build_skills(self):
        """Create batched skill instances with current planner."""
        kw = dict(
            robot_uid=self.robot_uid,
            n_envs=self.n_envs,
            n_arm_joints=self.n_arm_joints,
        )
        self.move_skill = BatchedMove(
            self.env, self.motion_gen, **kw,
        )
        self.push_skill = BatchedPush(
            self.env, self.motion_gen,
            move=self.move_skill,
            gripper_closed=self.robot_config.gripper_closed,
            gripper_open=self.robot_config.gripper_open,
            **kw,
        )

    def reset(self, seed=None):
        """Reset all envs and update the cuRobo collision world.

        The planner is NOT recreated (warmup is expensive). Only obstacle
        poses are updated. On first reset, the world update is skipped
        because the planner was just constructed with the same world configs.
        """
        self.env.reset(seed=seed)

        # Always update the collision world after reset — object poses change.
        # (On first reset, the pre-construction world had stale build-time
        # poses, not the post-reset table-level poses.)
        world_configs = build_world_configs(self.env, self.n_envs)
        update_world(self.motion_gen, world_configs)

        # Refresh object references
        try:
            self.objects = get_objects(self.env)
        except (NotImplementedError, AttributeError):
            self.objects = {}

    def move(
        self,
        goal_positions: torch.Tensor,
        goal_quaternions: torch.Tensor,
        gripper_state: float,
        **kwargs,
    ):
        """Plan and execute batched motion to goal poses."""
        return self.move_skill(
            goal_positions, goal_quaternions, gripper_state, **kwargs,
        )

    def push(
        self,
        approach_positions: torch.Tensor,
        approach_quaternions: torch.Tensor,
        push_positions: torch.Tensor,
        push_quaternions: torch.Tensor,
        **kwargs,
    ):
        """Execute batched push across all N envs."""
        return self.push_skill(
            approach_positions, approach_quaternions,
            push_positions, push_quaternions,
            **kwargs,
        )
