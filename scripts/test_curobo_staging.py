"""Execute just the staging plan and render a video to see the gripper orientation.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:10:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_curobo_staging.py'
"""
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _sapien_qpos_to_cu_joint_state,
    batched_move_to_pose, batched_follow_path, batched_actuate_gripper,
    get_arm_joint_names,
)
from taskbench.skills.curobo_world import shelf_world
from taskbench.skills.robot_config import get_robot_config
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
import os

os.makedirs("videos/staging_test", exist_ok=True)

cuda = torch.device("cuda:0")
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[-0.4, 0, 0, 0, 0, 0, 1],
               num_objects=5, render_mode="rgb_array", obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none",
               shelf={"floor_z": 0.20, "inner_h": 0.40},
               cylinder={"radius": 0.03, "half_length": 0.04})
env.reset(seed=42)
raw = env.unwrapped
rc = get_robot_config(env)
g = raw.shelf_geom
n_arm = len(get_arm_joint_names("ur5e_robotiq"))

# Render initial state
img = env.render()
img_np = img[0].cpu().numpy() if hasattr(img, 'cpu') else img
if img_np.max() <= 1.0:
    img_np = (img_np * 255).astype(np.uint8)
Image.fromarray(img_np).save("videos/staging_test/00_home.png")
print("Saved home pose")

# Setup cuRobo
world = shelf_world(env, n_envs=1, include_objects=False)[0]
mg = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

# Close gripper
batched_actuate_gripper(env, rc.gripper_closed, n_envs=1, n_arm_joints=n_arm)

# Render after gripper close
img = env.render()
img_np = img[0].cpu().numpy()
if img_np.max() <= 1.0:
    img_np = (img_np * 255).astype(np.uint8)
Image.fromarray(img_np).save("videos/staging_test/01_gripper_closed.png")
print("Saved gripper closed")

# Plan staging
Q_INTO_SHELF = [0.0, -0.7071068, 0.7071068, 0.0]  # horizontal gripper into shelf
staging_p = [g.front_x - 0.05, 0.0, g.surface_z + 0.05]
print(f"Staging target (world): {staging_p}")
print(f"Staging target (base): {[staging_p[i] - robot_base_pos[0, i].item() for i in range(3)]}")

start_state = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
staging_pos = torch.tensor([staging_p], dtype=torch.float32, device=cuda)
q_tensor = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda)

plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=True, timeout=10.0)
staging_result = batched_move_to_pose(
    mg, start_state, staging_pos, q_tensor,
    n_envs=1, use_batch_env=False,
    robot_base_position=robot_base_pos,
    plan_config=plan_config,
)

if staging_result["success"][0]:
    print("Staging SUCCEEDED — executing trajectory")
    batched_follow_path(
        env, staging_result["trajectories"],
        rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
        refine_steps=30,
    )

    # Render at staging pose
    img = env.render()
    img_np = img[0].cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save("videos/staging_test/02_staging.png")
    print("Saved staging pose")

    # Now try approach
    p1 = [g.front_x + 0.02, 0.0, g.surface_z + 0.05]  # just 2cm inside
    print(f"Approach target (world): {p1}")
    from curobo.types.robot import JointState as CuJointState
    staging_end = staging_result["trajectories"][0][-1].unsqueeze(0).to(device=cuda, dtype=torch.float32)
    staging_end_state = CuJointState.from_position(staging_end, joint_names=get_arm_joint_names("ur5e_robotiq"))

    approach_pos = torch.tensor([p1], dtype=torch.float32, device=cuda)
    approach_result = batched_move_to_pose(
        mg, staging_end_state, approach_pos, q_tensor,
        n_envs=1, use_batch_env=False,
        robot_base_position=robot_base_pos,
        plan_config=plan_config,
    )

    if approach_result["success"][0]:
        print("Approach SUCCEEDED — executing")
        batched_follow_path(
            env, approach_result["trajectories"],
            rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
            refine_steps=30,
        )
        img = env.render()
        img_np = img[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save("videos/staging_test/03_approach.png")
        print("Saved approach pose")
    else:
        print("Approach FAILED")
else:
    print("Staging FAILED")

env.close()
print("Done — check videos/staging_test/")
