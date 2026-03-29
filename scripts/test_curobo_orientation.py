"""Find the correct EE quaternion for 'gripper pointing into shelf (+X)'.

Renders the robot at several joint configs and saves images + FK results.
Look at the images to find which config has the gripper pointing forward,
then use that quaternion as Q_INTO_SHELF.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:10:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_curobo_orientation.py'
"""
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.robot import JointState as CuJointState

# Setup env with rendering
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[-0.1, 0, 0, 0, 0, 0, 1],
               num_objects=0, render_mode="rgb_array", obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none",
               shelf={"floor_z": 0.20, "inner_h": 0.30})
env.reset(seed=42)
raw = env.unwrapped

# Setup cuRobo for FK
cuda = torch.device("cuda:0")
mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)

# Test configs — vary wrist joints to find horizontal gripper
configs = {
    "home_original": [0, -1.5708, 1.5708, -1.5708, -1.5708, 3.1416],
    "forward_w3_pi": [0, -1.0, 1.0, -1.5708, -1.5708, 3.1416],
    "forward_w3_0":  [0, -1.0, 1.0, -1.5708, -1.5708, 0],
    "forward_w3_pi2":[0, -1.0, 1.0, -1.5708, -1.5708, 1.5708],
    "forward_w2_0":  [0, -1.0, 1.0, -1.5708, 0, 0],
    "forward_w2_pi": [0, -1.0, 1.0, -1.5708, 0, 3.1416],
    "forward_w1_0":  [0, -1.0, 1.0, 0, -1.5708, 0],
    "straight_out":  [0, -1.5708, 0, 0, -1.5708, 0],
}

import os
os.makedirs("videos/orientation_test", exist_ok=True)

for name, qpos_arm in configs.items():
    # Set robot joints
    full_qpos = qpos_arm + [0] * 8  # 6 arm + 8 gripper
    qpos_tensor = torch.tensor([full_qpos], dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos_tensor)

    # Step to update rendering
    for _ in range(10):
        action = torch.zeros(1, raw.agent.action_space.shape[0])
        action[0, :6] = torch.tensor(qpos_arm, dtype=torch.float32)
        env.step(action)

    # Render
    img = env.render()
    if img is not None:
        img_np = img[0].cpu().numpy() if hasattr(img, 'cpu') else img
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f"videos/orientation_test/{name}.png")

    # FK via cuRobo
    state = CuJointState.from_position(
        torch.tensor([qpos_arm], dtype=torch.float32, device=cuda),
        joint_names=get_arm_joint_names("ur5e_robotiq"),
    )
    kin = mg.kinematics.get_state(state.position)
    p = kin.ee_position[0].cpu().numpy()
    q = kin.ee_quaternion[0].cpu().numpy()
    print(f"{name:20s}: pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] quat=[{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f}]")

env.close()
print("\nImages saved to videos/orientation_test/")
print("Look at each image to find which config has the gripper pointing horizontally into the shelf (+X)")
