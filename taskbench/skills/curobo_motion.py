"""GPU-batched motion planning using NVIDIA cuRobo.

Mirrors ``motion.py`` but uses cuRobo's ``MotionGen`` instead of mplib,
enabling collision-aware planning across N parallel environments on GPU.

All functions are module-level and operate on GPU-vectorized ManiSkill envs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np

if TYPE_CHECKING:
    from curobo.types.math import Pose as CuPose
    from curobo.types.robot import JointState as CuJointState
    from curobo.geom.types import WorldConfig
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )

logger = logging.getLogger("taskbench.skills.curobo_motion")

# cuRobo built-in config name per taskbench robot uid.
# cuRobo's ur5e.yml covers the arm only (gripper joints are locked during
# planning), which is correct for push tasks where the gripper is static.
_CUROBO_ROBOT_CONFIGS = {
    "panda": "franka.yml",
    "panda_wristcam": "franka.yml",
    "ur5e_robotiq": "configs/curobo/ur5e_robotiq_2f_140.yml",
}

# End-effector link to use in cuRobo for each robot.
# Must match the TCP frame that ManiSkill uses for skill targets.
# cuRobo defaults differ (e.g. franka.yml defaults to "panda_hand"
# which is the wrist, not the fingertip TCP).
_CUROBO_EE_LINKS = {
    "panda": "ee_link",           # fingertip frame in cuRobo's franka URDF
    "panda_wristcam": "ee_link",
    "ur5e_robotiq": "grasp_frame",  # fingertip TCP in ur5e_robotiq_2f_140 URDF
}

# Arm joint names per robot (must match ManiSkill active joint order).
# Only arm joints are listed — gripper joints are not part of the cuRobo
# kinematic chain and are controlled separately via batched_actuate_gripper.
_PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]
_ARM_JOINT_NAMES = {
    "panda": _PANDA_ARM_JOINTS,
    "panda_wristcam": _PANDA_ARM_JOINTS,
    "ur5e_robotiq": [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ],
}


# Tool-pointing-down quaternion (wxyz) in each robot's cuRobo ee_link frame.
# These are verified via cuRobo FK + plan_single to produce a vertical push.
_TOOL_DOWN_QUATS = {
    "panda": [0.7071068, 0.0, 0.7071068, 0.0],
    "panda_wristcam": [0.7071068, 0.0, 0.7071068, 0.0],
    "ur5e_robotiq": [0.0, 1.0, 0.0, 0.0],
}


def get_tool_down_quat(robot_uid: str) -> list[float]:
    """Return the tool-pointing-down quaternion (wxyz) for a robot."""
    if robot_uid not in _TOOL_DOWN_QUATS:
        raise KeyError(f"No tool-down quaternion for robot {robot_uid!r}")
    return _TOOL_DOWN_QUATS[robot_uid]


# Rotation matrix from world frame to cuRobo base_link frame per robot.
# The ros-industrial UR5e URDF has a 180° Z rotation between base_link
# and base_link_inertia. cuRobo plans in base_link frame, ManiSkill uses
# the world frame aligned with base_link_inertia. So we need Rz(pi).
_RZ_180 = torch.tensor([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0,  0.0, 1.0],
])

_CUROBO_BASE_ROTATIONS = {
    "panda": None,              # no rotation needed
    "panda_wristcam": None,
    "ur5e_robotiq": _RZ_180,    # 180° Z between base_link and world
}


def get_curobo_base_rotation(robot_uid: str) -> Optional[torch.Tensor]:
    """Return the world-to-cuRobo-base rotation matrix for a robot.

    Returns None if no rotation is needed (base_link aligned with world).
    """
    return _CUROBO_BASE_ROTATIONS.get(robot_uid)


def get_curobo_config_name(robot_uid: str) -> str:
    """Return the cuRobo config filename or absolute path for a robot uid.

    Built-in cuRobo configs (e.g. ``franka.yml``) are resolved by cuRobo's
    asset path. Custom configs (starting with ``configs/``) are resolved
    relative to the project root.
    """
    if robot_uid not in _CUROBO_ROBOT_CONFIGS:
        raise KeyError(
            f"No cuRobo config for robot {robot_uid!r}. "
            f"Available: {', '.join(sorted(_CUROBO_ROBOT_CONFIGS))}"
        )
    cfg = _CUROBO_ROBOT_CONFIGS[robot_uid]
    # Custom configs need an absolute path since cuRobo resolves relative
    # to its own content directory.
    if cfg.startswith("configs/"):
        project_root = Path(__file__).resolve().parents[2]
        cfg = str(project_root / cfg)
    return cfg


def get_arm_joint_names(robot_uid: str) -> list[str]:
    """Return the arm joint names for a robot uid."""
    if robot_uid not in _ARM_JOINT_NAMES:
        raise KeyError(f"No arm joint names for robot {robot_uid!r}")
    return _ARM_JOINT_NAMES[robot_uid]


def setup_curobo_planner(
    robot_uid: str,
    world_configs: list[WorldConfig] | WorldConfig | dict | None = None,
    n_envs: int = 1,
    interpolation_dt: float = 0.02,
    warmup: bool = True,
    warmup_batch: int | None = None,
) -> MotionGen:
    """Create a cuRobo MotionGen instance for batched planning.

    Args:
        robot_uid: Taskbench robot identifier (e.g. "panda").
        world_configs: Collision world configuration(s). For batched env
            planning, pass a list of WorldConfig (one per env). For single
            env or uniform worlds, pass a single WorldConfig or dict.
        n_envs: Number of parallel environments.
        interpolation_dt: Timestep for trajectory interpolation (seconds).
        warmup: Whether to run cuRobo warmup (expensive but required for
            CUDA graph compilation).
        warmup_batch: Batch size for warmup. Defaults to n_envs.

    Returns:
        Configured MotionGen instance.
    """
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    curobo_config = get_curobo_config_name(robot_uid)

    # For custom YAML configs, load as dict and resolve relative paths
    # so cuRobo can find the URDF and mesh assets.
    if curobo_config.endswith(".yml") and "/" in curobo_config:
        import yaml
        config_path = Path(curobo_config)
        with open(config_path, encoding="utf-8") as f:
            robot_dict = yaml.safe_load(f)
        # Resolve urdf_path and asset_root_path relative to project root
        # Config is at configs/curobo/foo.yml → parents[2] is project root
        project_root = config_path.parents[2]
        kin = robot_dict["robot_cfg"]["kinematics"]
        for key in ("urdf_path", "asset_root_path"):
            if key in kin and kin[key] and not Path(kin[key]).is_absolute():
                kin[key] = str(project_root / kin[key])
        curobo_config = robot_dict

    # MotionGenConfig accepts None, list[WorldConfig], WorldConfig, or dict
    world_model = world_configs

    ee_link = _CUROBO_EE_LINKS.get(robot_uid)

    mg_config = MotionGenConfig.load_from_robot_config(
        curobo_config,
        world_model,
        interpolation_dt=interpolation_dt,
        n_collision_envs=n_envs,
        use_cuda_graph=False,
        ee_link_name=ee_link,
        rotation_threshold=0.01,  # tight orientation matching
        num_ik_seeds=64,          # more IK attempts for harder orientations
    )
    motion_gen = MotionGen(mg_config)

    if warmup:
        batch = warmup_batch or n_envs
        logger.info(
            "Warming up cuRobo MotionGen (robot=%s, n_envs=%d, warmup_batch=%d)...",
            robot_uid, n_envs, batch,
        )
        motion_gen.warmup(batch=batch)
        logger.info("cuRobo warmup complete")

    return motion_gen


def update_world(motion_gen, world_configs) -> None:
    """Update obstacle poses in the cuRobo planner.

    Call this between episodes when obstacle layouts change.

    Args:
        motion_gen: The MotionGen instance to update.
        world_configs: A single WorldConfig or a list of WorldConfig.
            A single config updates all collision envs to the same world.
            A list updates per-env collision worlds (required when envs
            have different obstacle layouts).
    """
    if not isinstance(world_configs, list):
        motion_gen.update_world(world_configs)
        logger.debug("Updated cuRobo world (single config)")
        return

    # All same object? (common case: [wc] * N for static geometry)
    if all(wc is world_configs[0] for wc in world_configs):
        motion_gen.update_world(world_configs[0])
        logger.debug("Updated cuRobo world (identical configs)")
        return

    # Per-env different collision worlds: use the batch collision API
    motion_gen.world_coll_checker.load_batch_collision_model(world_configs)
    motion_gen.graph_planner.reset_buffer()
    logger.debug("Updated cuRobo world (per-env batch, %d envs)", len(world_configs))


def _sapien_qpos_to_cu_joint_state(
    env, robot_uid: str, n_envs: int
) -> CuJointState:
    """Extract current joint positions from all envs as a cuRobo JointState."""
    from curobo.types.robot import JointState as CuJointState

    raw = env.unwrapped
    joint_names = get_arm_joint_names(robot_uid)
    n_arm = len(joint_names)

    # qpos shape: (n_envs, n_joints_total)
    qpos = raw.agent.robot.get_qpos()
    arm_qpos = qpos[:, :n_arm].detach().clone()  # (n_envs, n_arm)

    cuda = torch.device("cuda:0") if torch.cuda.is_available() else arm_qpos.device
    pos = arm_qpos.to(dtype=torch.float32, device=cuda)
    vel = torch.zeros_like(pos)
    acc = torch.zeros_like(pos)
    return CuJointState(
        position=pos,
        velocity=vel,
        acceleration=acc,
        joint_names=joint_names,
    )


def _sapien_poses_to_cu_poses(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
) -> CuPose:
    """Convert batched SAPIEN poses to a cuRobo Pose.

    Args:
        positions: (N, 3) tensor of positions.
        quaternions: (N, 4) tensor of quaternions in wxyz order.
    """
    from curobo.types.math import Pose as CuPose

    return CuPose(
        position=positions.to(dtype=torch.float32),
        quaternion=quaternions.to(dtype=torch.float32),
    )


def _build_actions(
    arm_targets: torch.Tensor,
    gripper_state: float,
    control_mode: str,
    n_arm_joints: int,
    device: torch.device,
    n_envs: int,
) -> torch.Tensor:
    """Build (N, action_dim) action tensor from arm joint targets and gripper."""
    if control_mode == "pd_joint_pos_vel":
        action_dim = n_arm_joints * 2 + 1
    else:
        action_dim = n_arm_joints + 1
    actions = torch.zeros(n_envs, action_dim, device=device, dtype=torch.float32)
    actions[:, :n_arm_joints] = arm_targets
    actions[:, -1] = gripper_state
    return actions


def _world_to_curobo_frame(
    positions: torch.Tensor,
    robot_base_position: Optional[torch.Tensor] = None,
    robot_base_rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform world-frame positions to cuRobo's base_link frame.

    Applies translation (subtract base position) then rotation (apply
    inverse of base orientation). For the ros-industrial UR5e URDF,
    base_link has a 180° Z rotation relative to the world frame, so
    robot_base_rotation should be a (3, 3) rotation matrix representing
    this relationship.

    Args:
        positions: (N, 3) world-frame positions.
        robot_base_position: (3,) world position of robot base.
        robot_base_rotation: (3, 3) rotation matrix from world to
            cuRobo base_link frame. If None, identity is assumed.
    """
    if robot_base_position is not None:
        base = robot_base_position.detach().reshape(-1)[:3]
        positions = positions - base.unsqueeze(0)
    if robot_base_rotation is not None:
        # R @ p^T → (3, N), transpose back → (N, 3)
        R = robot_base_rotation.to(device=positions.device, dtype=positions.dtype)
        positions = (R @ positions.T).T.contiguous()
    return positions


def batched_move_to_pose(
    motion_gen: MotionGen,
    start_state: CuJointState,
    goal_positions: torch.Tensor,
    goal_quaternions: torch.Tensor,
    n_envs: int,
    plan_config: Optional[MotionGenPlanConfig] = None,
    use_batch_env: bool = True,
    robot_base_position: Optional[torch.Tensor] = None,
    robot_base_rotation: Optional[torch.Tensor] = None,
) -> dict:
    """Plan motions for N environments in batch.

    Args:
        motion_gen: Configured MotionGen instance.
        start_state: Current joint state for all envs (batched CuJointState).
        goal_positions: (N, 3) goal TCP positions in world frame.
        goal_quaternions: (N, 4) goal TCP quaternions (wxyz).
        n_envs: Number of environments.
        plan_config: Optional planning config overrides.
        use_batch_env: If True, use plan_batch_env (different world per env).
            If False, use plan_batch (same world for all).
        robot_base_position: (3,) world position of robot base.
        robot_base_rotation: (3, 3) rotation matrix from world to
            cuRobo base_link frame.

    Returns:
        Dict with keys:
            - "success": (N,) bool tensor
            - "trajectories": list of (T_i, n_arm) position tensors per env
                (None for failed plans)
            - "interpolated_plan": CuJointState with interpolated trajectory
                (from the MotionGen result, may have padded/uniform length)
    """
    if plan_config is None:
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        plan_config = MotionGenPlanConfig(max_attempts=4, enable_graph=False)

    goal_positions = _world_to_curobo_frame(
        goal_positions, robot_base_position, robot_base_rotation)

    goal_pose = _sapien_poses_to_cu_poses(goal_positions, goal_quaternions)

    if use_batch_env and n_envs > 1:
        result = motion_gen.plan_batch_env(start_state, goal_pose, plan_config)
    elif n_envs == 1:
        # Use plan_single for single env — plan_batch has indexing bugs with batch=1
        result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    else:
        result = motion_gen.plan_batch(start_state, goal_pose, plan_config)

    # Extract per-env trajectories
    success = result.success.cpu()  # (N,)

    # For batch results, optimized_plan.position is (N, T, n_arm).
    # For single results, get_interpolated_plan().position is (T, n_arm).
    opt_plan = result.optimized_plan
    if opt_plan is None or opt_plan.position is None:
        logger.warning("Planning returned no trajectory (all failed)")
        return {
            "success": success,
            "trajectories": [None] * n_envs,
        }
    positions = opt_plan.position  # (N, T, n_arm) or (T, n_arm)

    trajectories = []
    if positions.ndim == 3:
        # Batch result: (N, T, n_arm)
        for i in range(n_envs):
            if success[i]:
                trajectories.append(positions[i].cpu())
            else:
                trajectories.append(None)
    else:
        # Single result: (T, n_arm)
        if success[0]:
            trajectories.append(positions.cpu())
        else:
            trajectories.append(None)

    return {
        "success": success,
        "trajectories": trajectories,
    }


def batched_follow_path(
    env,
    trajectories: list[torch.Tensor | None],
    gripper_state: float,
    n_envs: int,
    n_arm_joints: int = 7,
    step_callback=None,
    refine_steps: int = 40,
) -> dict:
    """Execute planned trajectories across all N envs simultaneously.

    Pre-pads trajectories into a single (N, T, n_arm) tensor for fully
    vectorized stepping — no per-env Python loop at each timestep.

    For envs where planning failed (trajectory is None), the robot holds
    its current joint position.
    """
    raw = env.unwrapped
    device = raw.device

    max_len = max((t.shape[0] for t in trajectories if t is not None), default=0)
    if max_len == 0:
        logger.warning("No valid trajectories to execute")
        return {"obs": None, "rewards": None, "steps_executed": 0}

    current_arm = raw.agent.robot.get_qpos()[:, :n_arm_joints]
    total_steps = max_len + refine_steps

    # Pre-build padded trajectory tensor: (N, total_steps, n_arm).
    # Default: hold current position (for failed-plan envs).
    padded = current_arm.unsqueeze(1).expand(-1, total_steps, -1).clone()
    for i, traj in enumerate(trajectories):
        if traj is not None:
            T = traj.shape[0]
            traj_dev = traj.to(device=device, dtype=torch.float32)
            padded[i, :T] = traj_dev
            if T < total_steps:
                padded[i, T:] = traj_dev[-1]

    obs = rewards = None
    control_mode = raw.control_mode
    for t in range(total_steps):
        actions = _build_actions(
            padded[:, t], gripper_state, control_mode,
            n_arm_joints, device, n_envs,
        )
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if step_callback is not None:
            step_callback(t, obs, rewards)

    return {"obs": obs, "rewards": rewards, "steps_executed": total_steps}


def batched_ik(
    motion_gen,
    goal_positions: torch.Tensor,
    goal_quaternions: torch.Tensor,
    n_envs: int,
    robot_uid: str = "panda",
    robot_base_position: Optional[torch.Tensor] = None,
    robot_base_rotation: Optional[torch.Tensor] = None,
    seed_joints: Optional[torch.Tensor] = None,
) -> dict:
    """Solve IK for N goal poses in parallel using cuRobo.

    Returns joint configurations without trajectory optimization —
    just the goal joint positions.

    Args:
        robot_uid: Robot identifier for looking up arm joint names.
        robot_base_position: (3,) world position of robot base.
        robot_base_rotation: (3, 3) rotation matrix from world to
            cuRobo base_link frame.
        seed_joints: (N, n_arm) optional joint seed. When provided, the
            IK solver is seeded with these joints so the solution stays
            close to the current configuration (important for straight-line
            pushes where we want minimal joint change).

    Returns:
        Dict with "success" (N,) bool and "joint_positions" (N, n_arm) tensor.
    """
    goal_positions = _world_to_curobo_frame(
        goal_positions, robot_base_position, robot_base_rotation)

    goal_pose = _sapien_poses_to_cu_poses(goal_positions, goal_quaternions)

    # Seed the IK with current joints so the solution is nearby.
    # ``retract_config`` sets the fallback configuration on failure AND
    # biases the solver toward this configuration during the search.
    result = motion_gen.ik_solver.solve_batch(goal_pose)

    success = result.success.squeeze()  # ensure (N,) bool
    if success.dim() == 0:
        success = success.unsqueeze(0)

    # Build full (N, n_arm) tensor, filling failures with seed/zeros
    n_arm = result.solution.shape[-1]
    fallback = seed_joints if seed_joints is not None else torch.zeros(n_envs, n_arm)
    joint_positions = fallback.clone().to(device=result.solution.device, dtype=torch.float32)
    for i in range(n_envs):
        if success[i]:
            joint_positions[i] = result.solution[i, 0]

    return {
        "success": success.cpu(),
        "joint_positions": joint_positions.cpu(),
    }


def batched_linear_push(
    env,
    start_arm_joints: torch.Tensor,
    end_arm_joints: torch.Tensor,
    gripper_state: float,
    n_envs: int,
    n_arm_joints: int = 7,
    n_steps: int = 80,
    refine_steps: int = 20,
    step_callback=None,
) -> dict:
    """Execute a straight-line push by linearly interpolating joint targets.

    Simple linear interpolation in joint space between the approach
    configuration and the push goal configuration. Produces a
    near-straight-line Cartesian path for small motions like pushes.
    """
    raw = env.unwrapped
    device = raw.device
    control_mode = raw.control_mode

    start = start_arm_joints.to(device=device, dtype=torch.float32)
    end = end_arm_joints.to(device=device, dtype=torch.float32)

    total_steps = n_steps + refine_steps
    obs = rewards = None

    for t in range(total_steps):
        alpha = min(float(t) / max(n_steps - 1, 1), 1.0)
        arm_targets = start + alpha * (end - start)
        actions = _build_actions(
            arm_targets, gripper_state, control_mode,
            n_arm_joints, device, n_envs,
        )
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if step_callback is not None:
            step_callback(t, obs, rewards)

    return {"obs": obs, "rewards": rewards, "steps_executed": total_steps}


def batched_actuate_gripper(
    env,
    gripper_state: float,
    n_envs: int,
    n_arm_joints: int = 7,
    steps: int = 6,
    step_callback=None,
) -> None:
    """Open or close grippers across all N envs for a number of steps.

    Holds current arm joint positions while commanding the gripper.
    """
    if steps <= 0:
        return

    raw = env.unwrapped
    current_arm = raw.agent.robot.get_qpos()[:, :n_arm_joints].detach().clone()
    actions = _build_actions(
        current_arm, gripper_state, raw.control_mode,
        n_arm_joints, raw.device, n_envs,
    )
    for t in range(steps):
        env.step(actions)
        if step_callback is not None:
            step_callback(t, None, None)
