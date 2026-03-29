"""Set robot joints and render close-up images to understand joint effects.

Run locally (no GPU needed, no cuRobo):
    uv run python scripts/test_joint_sweep.py
"""
import torch
import numpy as np
import gymnasium as gym
from PIL import Image
import os

import taskbench.envs

os.makedirs("videos/joint_sweep", exist_ok=True)

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[-0.1, 0, 0, 0, 0, 0, 1],
               num_objects=0, render_mode="rgb_array", obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none",
               shelf={"floor_z": 0.20, "inner_h": 0.30})
env.reset(seed=42)
raw = env.unwrapped

joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

configs = {
    "all_zero":     [0, 0, 0, 0, 0, 0],
    "j1_neg90":     [0, -1.5708, 0, 0, 0, 0],
    "j12_neg90":    [0, -1.5708, 1.5708, 0, 0, 0],
    "j123_neg90":   [0, -1.5708, 1.5708, -1.5708, 0, 0],
    "j1234_neg90":  [0, -1.5708, 1.5708, -1.5708, -1.5708, 0],
    "original_home":[0, -1.5708, 1.5708, -1.5708, -1.5708, 3.1416],
    "fwd_w3_0":     [0, -1.0, 1.0, -1.5708, -1.5708, 0],
    "fwd_w3_pi2":   [0, -1.0, 1.0, -1.5708, -1.5708, 1.5708],
    "fwd_w3_pi":    [0, -1.0, 1.0, -1.5708, -1.5708, 3.1416],
}

for name, qpos_arm in configs.items():
    full_qpos = qpos_arm + [0] * 8
    qpos_tensor = torch.tensor([full_qpos], dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos_tensor)

    # Step a few times to let rendering update
    action = torch.zeros(1, raw.agent.action_space.shape[0])
    action[0, :6] = torch.tensor(qpos_arm, dtype=torch.float32)
    for _ in range(15):
        env.step(action)

    img = env.render()
    if img is not None:
        img_np = img
        if hasattr(img_np, 'cpu'):
            img_np = img_np[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f"videos/joint_sweep/{name}.png")
        print(f"{name:20s}: saved  joints={[round(x,2) for x in qpos_arm]}")

env.close()
print("\nImages saved to videos/joint_sweep/")
