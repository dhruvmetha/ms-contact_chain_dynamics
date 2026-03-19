"""Batched manipulation skills for GPU-parallel operation.

These skills operate on N environments simultaneously using cuRobo for
collision-aware motion planning. They are the batched counterparts of
the skills in ``primitives.py``.

Primary use case: massively parallel contact data collection (push tasks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from taskbench.skills.curobo_motion import (
    _sapien_qpos_to_cu_joint_state,
    batched_actuate_gripper,
    batched_follow_path,
    batched_ik,
    batched_linear_push,
    batched_move_to_pose,
)

logger = logging.getLogger("taskbench.skills.batched_primitives")


@dataclass
class BatchedSkillResult:
    """Result from a batched skill across N envs."""
    success_mask: torch.Tensor  # (N,) bool — per-env success
    n_success: int = 0
    n_total: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    steps_executed: int = 0


@dataclass
class BatchedMoveResult(BatchedSkillResult):
    pass


@dataclass
class BatchedPushResult(BatchedSkillResult):
    contact_detected: Optional[torch.Tensor] = None  # (N,) bool
    peak_contact_forces: Optional[torch.Tensor] = None  # (N,)
    object_displacements: Optional[torch.Tensor] = None  # (N, n_obj, 3)


class BatchedMove:
    """Plan and execute motion across N envs using cuRobo.

    Args:
        env: GPU-vectorized ManiSkill env.
        motion_gen: cuRobo MotionGen instance (pre-warmed).
        robot_uid: Robot identifier (e.g. "panda").
        n_envs: Number of parallel environments.
        n_arm_joints: Number of arm DOF.
    """

    def __init__(self, env, motion_gen, *, robot_uid: str, n_envs: int,
                 n_arm_joints: int = 7):
        self.env = env
        self.motion_gen = motion_gen
        self.robot_uid = robot_uid
        self.n_envs = n_envs
        self.n_arm_joints = n_arm_joints
        # Cache robot base position for world→robot frame transform
        raw = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot_base_pos = raw.agent.robot.pose.p[0].detach().clone().to(dtype=torch.float32)

    def __call__(
        self,
        goal_positions: torch.Tensor,
        goal_quaternions: torch.Tensor,
        gripper_state: float,
        *,
        use_batch_env: bool = True,
        step_callback=None,
    ) -> BatchedMoveResult:
        """Plan and execute motions to goal poses across all envs.

        Args:
            goal_positions: (N, 3) target TCP positions.
            goal_quaternions: (N, 4) target TCP quaternions (wxyz).
            gripper_state: Gripper action value (open or closed).
            use_batch_env: Use per-env collision worlds.
            step_callback: Optional callback after each env.step().

        Returns:
            BatchedMoveResult with per-env success mask.
        """
        # Get current joint state
        start_state = _sapien_qpos_to_cu_joint_state(
            self.env, self.robot_uid, self.n_envs
        )

        # Plan batch (transform goals from world frame to robot base frame)
        plan_result = batched_move_to_pose(
            self.motion_gen,
            start_state,
            goal_positions,
            goal_quaternions,
            self.n_envs,
            use_batch_env=use_batch_env,
            robot_base_position=self.robot_base_pos,
        )

        plan_success = plan_result["success"]
        n_plan_ok = int(plan_success.sum().item())
        if n_plan_ok == 0:
            logger.warning("All %d planning queries failed", self.n_envs)
            return BatchedMoveResult(
                success_mask=plan_success,
                n_success=0,
                n_total=self.n_envs,
                failure_reasons=["all_plans_failed"],
            )

        logger.debug("Planning: %d/%d succeeded", n_plan_ok, self.n_envs)

        # Execute trajectories
        exec_result = batched_follow_path(
            self.env,
            plan_result["trajectories"],
            gripper_state,
            self.n_envs,
            n_arm_joints=self.n_arm_joints,
            step_callback=step_callback,
        )

        return BatchedMoveResult(
            success_mask=plan_success,
            n_success=n_plan_ok,
            n_total=self.n_envs,
            steps_executed=exec_result["steps_executed"],
        )


class BatchedPush:
    """Batched push skill for contact data collection.

    Executes the full push sequence across N envs:
    1. Close grippers
    2. Plan and execute approach trajectories (collision-aware)
    3. Plan and execute push trajectories (targets excluded from collision)
    4. Lift to disengage

    Args:
        env: GPU-vectorized ManiSkill env.
        motion_gen: cuRobo MotionGen instance.
        robot_uid: Robot identifier.
        n_envs: Number of parallel environments.
        n_arm_joints: Number of arm DOF.
        gripper_closed: Action value for closed gripper.
        gripper_open: Action value for open gripper.
    """

    def __init__(self, env, motion_gen, *, move: BatchedMove,
                 robot_uid: str, n_envs: int,
                 n_arm_joints: int = 7,
                 gripper_closed: float = -1.0,
                 gripper_open: float = 1.0):
        self.env = env
        self.motion_gen = motion_gen
        self.robot_uid = robot_uid
        self.n_envs = n_envs
        self.n_arm_joints = n_arm_joints
        self.gripper_closed = gripper_closed
        self.gripper_open = gripper_open
        self.move = move

    def __call__(
        self,
        approach_positions: torch.Tensor,
        approach_quaternions: torch.Tensor,
        push_positions: torch.Tensor,
        push_quaternions: torch.Tensor,
        *,
        lift_height: float = 0.08,
        settle_steps: int = 8,
        record_callback=None,
        step_callback=None,
    ) -> BatchedPushResult:
        """Execute batched push across all N envs.

        Args:
            approach_positions: (N, 3) pre-contact positions.
            approach_quaternions: (N, 4) pre-contact orientations (wxyz).
            push_positions: (N, 3) end-of-push positions.
            push_quaternions: (N, 4) end-of-push orientations (wxyz).
            lift_height: Height to lift above push pose after contact.
            settle_steps: Steps to hold after lift for physics settling.
            record_callback: Optional callback for recording after each phase.
            step_callback: Optional callback after each env.step().

        Returns:
            BatchedPushResult with per-env success and contact data.
        """
        total_steps = 0

        # Phase 1: Close grippers across all envs
        batched_actuate_gripper(
            self.env, self.gripper_closed, self.n_envs,
            n_arm_joints=self.n_arm_joints, steps=6,
            step_callback=step_callback,
        )
        total_steps += 6
        if record_callback:
            record_callback("gripper_close")

        # Phase 2: Approach (collision-aware via cuRobo, can curve freely)
        approach_result = self.move(
            approach_positions,
            approach_quaternions,
            self.gripper_closed,
            step_callback=step_callback,
        )
        total_steps += approach_result.steps_executed
        if record_callback:
            record_callback("approach")

        # Phase 3: Push (straight-line via linear joint interpolation)
        # Solve IK for the push goal to get target joint positions,
        # then linearly interpolate from current joints to goal joints.
        # This gives a near-straight Cartesian path without cuRobo's
        # trajectory optimizer arcing over the objects.
        raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        current_arm = raw.agent.robot.get_qpos()[:, :self.n_arm_joints].detach().clone()

        ik_result = batched_ik(
            self.move.motion_gen,
            push_positions,
            push_quaternions,
            self.n_envs,
            robot_uid=self.robot_uid,
            robot_base_position=self.move.robot_base_pos,
            seed_joints=current_arm,
        )
        # For envs where IK failed, keep current joints (no push)
        push_joints = ik_result["joint_positions"].to(device=current_arm.device)
        ik_failed = ~ik_result["success"].to(device=current_arm.device)
        push_joints[ik_failed] = current_arm[ik_failed]

        push_exec = batched_linear_push(
            self.env,
            start_arm_joints=current_arm,
            end_arm_joints=push_joints,
            gripper_state=self.gripper_closed,
            n_envs=self.n_envs,
            n_arm_joints=self.n_arm_joints,
            step_callback=step_callback,
        )
        total_steps += push_exec["steps_executed"]
        if record_callback:
            record_callback("push")

        # Phase 4: Lift to disengage (cuRobo, collision-aware)
        if lift_height > 0:
            lift_positions = push_positions.clone()
            lift_positions[:, 2] += lift_height
            lift_result = self.move(
                lift_positions,
                push_quaternions,
                self.gripper_closed,
                step_callback=step_callback,
            )
            total_steps += lift_result.steps_executed
            if record_callback:
                record_callback("lift")

        # Phase 5: Settle
        if settle_steps > 0:
            batched_actuate_gripper(
                self.env, self.gripper_open, self.n_envs,
                n_arm_joints=self.n_arm_joints, steps=settle_steps,
                step_callback=step_callback,
            )
            total_steps += settle_steps
            if record_callback:
                record_callback("settle")

        # Combine success: approach plan + push IK must both succeed per-env
        combined_success = approach_result.success_mask & ik_result["success"]
        n_success = int(combined_success.sum().item())

        return BatchedPushResult(
            success_mask=combined_success,
            n_success=n_success,
            n_total=self.n_envs,
            steps_executed=total_steps,
        )
