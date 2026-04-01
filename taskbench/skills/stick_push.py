"""StickPush skill: stage → insert → sweep → retract → rest.

Reusable push primitive for stick end-effectors. Works for both
shelf push (horizontal) and top-down bin push (vertical) — the
caller provides waypoints, the skill executes the motion sequence.

Supports single-env and GPU-batched (N envs) operation.

Usage:
    push = StickPush(env, motion_gen, robot_uid="panda_stick_long",
                     n_envs=1, rest_qpos=REST, robot_base_pos=base,
                     safe_start_qpos=SAFE)
    result = push(approach, quat, entry, sweep, retract)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable

import torch

from taskbench.skills.curobo_motion import (
    batched_ee_delta_move,
    batched_follow_path,
    get_arm_joint_names,
    _world_to_curobo_frame,
)

logger = logging.getLogger("taskbench.skills.stick_push")


@dataclass
class StickPushResult:
    """Result of a stick push execution."""
    success_mask: torch.Tensor       # (N,) per-env overall success
    staging_success: torch.Tensor    # (N,) bool — cuRobo staging succeeded
    entry_final_dist: torch.Tensor   # (N,) float — distance to p1 after entry
    sweep_final_dist: torch.Tensor   # (N,) float — distance to p2 after sweep
    retract_final_dist: torch.Tensor # (N,) float — distance to retract point
    orientation_drift: torch.Tensor  # (N,) float — max quat deviation from goal
    steps_executed: int              # total sim steps


class StickPush:
    """Five-phase stick push: stage → insert → sweep → retract → rest.

    Phase 1 (Stage): cuRobo plans collision-free path from rest to approach
        point. Executed via pd_joint_pos.
    Phase 2 (Insert): EE delta control pushes stick from approach to entry
        point (p1) inside the workspace.
    Phase 3 (Sweep): EE delta control sweeps stick from p1 to p2.
    Phase 4 (Retract): EE delta control pulls stick back to retract point.
    Phase 5 (Rest): pd_joint_pos drives arm back to rest configuration.

    Args:
        env: ManiSkill env (single or GPU-vectorized).
        motion_gen: cuRobo MotionGen instance (for staging).
        robot_uid: Robot identifier (e.g. "panda_stick_long").
        n_envs: Number of parallel environments.
        n_arm_joints: Number of arm DOF (default 7 for Panda).
        rest_qpos: Joint configuration to return to after push.
        robot_base_pos: (1, 3) tensor, robot base position on CUDA.
        safe_start_qpos: Joint config that passes cuRobo self-collision
            check (may differ slightly from rest_qpos).
    """

    def __init__(
        self,
        env,
        motion_gen,
        *,
        robot_uid: str,
        n_envs: int,
        n_arm_joints: int = 7,
        rest_qpos: list[float],
        robot_base_pos: torch.Tensor,
        safe_start_qpos: list[float],
    ):
        self.env = env
        self.raw = env.unwrapped
        self.motion_gen = motion_gen
        self.robot_uid = robot_uid
        self.n_envs = n_envs
        self.n_arm = n_arm_joints
        self.rest_qpos = rest_qpos
        self.robot_base_pos = robot_base_pos
        self.safe_start_qpos = safe_start_qpos
        self.device = self.raw.device
        self.cuda = robot_base_pos.device

    def __call__(
        self,
        approach_positions: torch.Tensor,   # (N, 3)
        approach_quaternions: torch.Tensor,  # (N, 4) wxyz
        entry_positions: torch.Tensor,       # (N, 3) p1
        sweep_positions: torch.Tensor,       # (N, 3) p2
        retract_positions: torch.Tensor,     # (N, 3)
        *,
        entry_max_steps: int = 100,
        sweep_max_steps: int = 200,
        retract_max_steps: int = 100,
        rest_steps: int = 60,
        max_delta: float = 0.03,
        convergence_threshold: float = 0.01,
        staging_max_attempts: int = 10,
        staging_timeout: float = 30.0,
        step_callback: Optional[Callable] = None,
    ) -> StickPushResult:
        """Execute a stick push across N environments.

        Args:
            approach_positions: (N, 3) world-frame approach points.
            approach_quaternions: (N, 4) desired EE orientation (wxyz).
            entry_positions: (N, 3) push start points (p1).
            sweep_positions: (N, 3) push end points (p2).
            retract_positions: (N, 3) retract points (outside workspace).
            entry_max_steps: Max steps for insert phase.
            sweep_max_steps: Max steps for sweep phase.
            retract_max_steps: Max steps for retract phase.
            rest_steps: Steps to hold rest_qpos at the end.
            max_delta: Per-step EE position delta clipping bound.
            convergence_threshold: Distance threshold for convergence.
            staging_max_attempts: cuRobo planning attempts.
            staging_timeout: cuRobo planning timeout (seconds).
            step_callback: Optional (step, obs, rewards) → None.

        Returns:
            StickPushResult with per-env telemetry.
        """
        total_steps = 0
        eef_trace = []  # track TCP position every step

        def _record_eef():
            tcp = self.raw.agent.tcp.pose.p[0].cpu().numpy().copy()
            eef_trace.append(tcp)

        def _tracking_callback(step, obs, rew):
            _record_eef()
            if step_callback is not None:
                step_callback(step, obs, rew)

        def _log_orientation(label):
            q = self.raw.agent.tcp.pose.q  # (N, 4)
            q_ref = approach_quaternions.to(self.device)
            dots = torch.abs(torch.sum(q * q_ref, dim=1))
            drift = 1.0 - torch.clamp(dots, max=1.0)
            tcp = self.raw.agent.tcp.pose.p[0].cpu().numpy()
            quat = q[0].cpu().numpy()
            logger.info("  %s: TCP=[%.3f,%.3f,%.3f] quat=[%.3f,%.3f,%.3f,%.3f] drift=%.4f",
                        label, *tcp, *quat, drift.mean().item())

        # --- Phase 1: Stage (cuRobo, pd_joint_pos) ---
        staging_success, staging_steps = self._stage(
            approach_positions, approach_quaternions,
            max_attempts=staging_max_attempts,
            timeout=staging_timeout,
            step_callback=_tracking_callback,
        )
        total_steps += staging_steps
        _log_orientation("After stage")

        # --- Phase 2: Insert (pd_ee_delta_pose) ---
        self.raw.agent.set_control_mode("pd_ee_delta_pose")
        self.raw.agent.controller.reset()
        _log_orientation("After mode switch (before insert)")

        entry_result = batched_ee_delta_move(
            self.env, entry_positions,
            target_quaternions=approach_quaternions,
            max_steps=entry_max_steps, max_delta=max_delta,
            convergence_threshold=convergence_threshold,
            step_callback=_tracking_callback,
        )
        total_steps += entry_result["steps_executed"]
        logger.info("  Insert: %d steps, mean_dist=%.4f, converged=%d/%d",
                     entry_result["steps_executed"],
                     entry_result["final_dists"].mean().item(),
                     entry_result["converged"].sum().item(), self.n_envs)
        _log_orientation("After insert")

        # --- Phase 3: Sweep (pd_ee_delta_pose) ---
        sweep_result = batched_ee_delta_move(
            self.env, sweep_positions,
            target_quaternions=approach_quaternions,
            max_steps=sweep_max_steps, max_delta=max_delta,
            convergence_threshold=convergence_threshold,
            step_callback=_tracking_callback,
        )
        total_steps += sweep_result["steps_executed"]
        logger.info("  Sweep: %d steps, mean_dist=%.4f, converged=%d/%d",
                     sweep_result["steps_executed"],
                     sweep_result["final_dists"].mean().item(),
                     sweep_result["converged"].sum().item(), self.n_envs)
        _log_orientation("After sweep")

        # --- Phase 4: Retract (pd_ee_delta_pose) ---
        # 4a: Small reverse along sweep direction to disengage from cylinders
        tcp_now = self.raw.agent.tcp.pose.p.clone()
        sweep_dir = (sweep_positions.to(self.device) - entry_positions.to(self.device))
        sweep_len = torch.norm(sweep_dir, dim=1, keepdim=True).clamp(min=1e-6)
        sweep_unit = sweep_dir / sweep_len
        nudge_back = tcp_now - sweep_unit * 0.03  # 3cm back along sweep direction
        nudge_back[:, 2] = tcp_now[:, 2]  # keep same z
        batched_ee_delta_move(
            self.env, nudge_back,
            target_quaternions=approach_quaternions,
            max_steps=20, max_delta=max_delta,
            convergence_threshold=convergence_threshold,
            step_callback=_tracking_callback,
        )
        total_steps += 20

        # 4b: Pull straight out from current position (just change X)
        tcp_after_nudge = self.raw.agent.tcp.pose.p.clone()
        retract_from_here = tcp_after_nudge.clone()
        retract_from_here[:, 0] = retract_positions.to(self.device)[:, 0]  # pull X to outside shelf
        retract_from_here[:, 2] = retract_positions.to(self.device)[:, 2]  # keep target Z
        retract_result = batched_ee_delta_move(
            self.env, retract_from_here,
            target_quaternions=approach_quaternions,
            max_steps=retract_max_steps, max_delta=max_delta,
            convergence_threshold=convergence_threshold,
            step_callback=_tracking_callback,
        )
        total_steps += retract_result["steps_executed"]
        logger.info("  Retract: %d steps, mean_dist=%.4f, converged=%d/%d",
                     retract_result["steps_executed"],
                     retract_result["final_dists"].mean().item(),
                     retract_result["converged"].sum().item(), self.n_envs)

        _log_orientation("After retract")

        # Orientation correction is built into every EE delta step
        # (insert, sweep, retract all actively track target_quaternions)
        q = self.raw.agent.tcp.pose.q  # (N, 4)
        q_ref = approach_quaternions.to(self.device)
        dots = torch.abs(torch.sum(q * q_ref, dim=1))
        orientation_drift = 1.0 - torch.clamp(dots, max=1.0)

        # --- Phase 5: Smooth return to safe start qpos for next staging ---
        self.raw.agent.set_control_mode("pd_joint_pos")
        self.raw.agent.controller.reset()

        current_qpos = self.raw.agent.robot.get_qpos()[:, :self.n_arm].clone()
        safe_target = torch.tensor(
            self.safe_start_qpos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.n_envs, -1)
        for i in range(rest_steps):
            alpha = min((i + 1) / rest_steps, 1.0)
            interp = current_qpos + alpha * (safe_target - current_qpos)
            self.env.step(interp)
            _record_eef()
            if step_callback is not None:
                step_callback(0, None, None)
        total_steps += rest_steps
        logger.info("  Rest: %d steps, orient_drift=%.4f", rest_steps, orientation_drift.mean().item())

        # Dump EEF trace summary
        import numpy as np
        trace = np.array(eef_trace)
        if len(trace) > 0:
            # Find big jumps (>5cm between consecutive frames)
            diffs = np.linalg.norm(np.diff(trace, axis=0), axis=1)
            jumps = np.where(diffs > 0.05)[0]
            logger.info("  EEF trace: %d frames, x=[%.3f,%.3f], y=[%.3f,%.3f], z=[%.3f,%.3f]",
                        len(trace), trace[:,0].min(), trace[:,0].max(),
                        trace[:,1].min(), trace[:,1].max(),
                        trace[:,2].min(), trace[:,2].max())
            if len(jumps) > 0:
                logger.info("  EEF JUMPS >5cm at frames: %s", jumps.tolist()[:10])

        # Overall success: staging + sweep converged
        success_mask = staging_success & sweep_result["converged"]

        return StickPushResult(
            success_mask=success_mask,
            staging_success=staging_success,
            entry_final_dist=entry_result["final_dists"],
            sweep_final_dist=sweep_result["final_dists"],
            retract_final_dist=retract_result["final_dists"],
            orientation_drift=orientation_drift,
            steps_executed=total_steps,
        )

    def _stage(
        self,
        approach_positions: torch.Tensor,
        approach_quaternions: torch.Tensor,
        *,
        max_attempts: int = 10,
        timeout: float = 30.0,
        step_callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, int]:
        """Plan and execute staging: rest → approach point via cuRobo.

        Returns (staging_success (N,) bool, steps_executed int).
        """
        from curobo.types.math import Pose as CuPose
        from curobo.types.robot import JointState as CuJointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        N = self.n_envs
        n_arm = self.n_arm

        # Use safe start qpos that passes cuRobo self-collision check
        start_pos = torch.tensor(
            self.safe_start_qpos, device=self.cuda, dtype=torch.float32
        ).unsqueeze(0).expand(N, -1).clone()

        start_state = CuJointState(
            position=start_pos.clone(),
            velocity=torch.zeros(N, n_arm, device=self.cuda),
            acceleration=torch.zeros(N, n_arm, device=self.cuda),
            joint_names=get_arm_joint_names(self.robot_uid),
        )

        # Build goal
        goal_pos = _world_to_curobo_frame(
            approach_positions.clone().to(self.cuda), self.robot_base_pos
        )
        goal_pose = CuPose(
            position=goal_pos.clone(),
            quaternion=approach_quaternions.clone().to(self.cuda),
        )

        plan_config = MotionGenPlanConfig(
            max_attempts=max_attempts, enable_graph=False,
            timeout=timeout, enable_opt=True,
        )

        # Plan
        if N == 1:
            result = self.motion_gen.plan_single(start_state, goal_pose, plan_config)
        else:
            result = self.motion_gen.plan_batch(start_state, goal_pose, plan_config)

        staging_success = result.success.cpu().to(dtype=torch.bool)
        n_ok = staging_success.sum().item()
        logger.info("  Stage: %d/%d planned", n_ok, N)

        if n_ok == 0:
            return staging_success.to(self.device), 0

        # Extract trajectories
        trajectories = result.optimized_plan.position  # (N, T, n_arm) or (T, n_arm)
        if trajectories.ndim == 2:
            trajectories = trajectories.unsqueeze(0)  # (1, T, n_arm)
        T = trajectories.shape[1]

        # For failed envs, hold at start qpos
        for i in range(N):
            if not staging_success[i]:
                trajectories[i] = start_pos[i].unsqueeze(0).expand(T, -1)

        # Execute via pd_joint_pos
        traj_dev = trajectories.to(device=self.device, dtype=torch.float32)
        for t in range(T):
            self.env.step(traj_dev[:, t, :])
            if step_callback is not None:
                step_callback(t, None, None)

        # Settle
        settle = 10
        for _ in range(settle):
            self.env.step(traj_dev[:, -1, :])
            if step_callback is not None:
                step_callback(0, None, None)

        return staging_success.to(self.device), T + settle
