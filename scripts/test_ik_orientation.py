"""Try ALL common quaternions via cuRobo IK, set the joints, render, and see which one is horizontal.

No planning — just IK + set_qpos + render.
"""
import torch
import numpy as np
import gymnasium as gym
from PIL import Image
import os

import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.math import Pose as CuPose

os.makedirs("videos/ik_orient", exist_ok=True)
cuda = torch.device("cuda:0")

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[-0.4, 0, 0, 0, 0, 0, 1],
               num_objects=0, render_mode="rgb_array", obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none",
               shelf={"floor_z": 0.20, "inner_h": 0.40})
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)

# Target position: just in front of shelf opening, in BASE frame
# Robot at -0.4, shelf front at 0.42, so base x = 0.42 + 0.4 = 0.82
# But that's too far. Use a closer point.
# Base frame: x=0.5, y=0, z=0.26 (which is world x=0.1, reachable)
target_pos = [0.5, 0.0, 0.26]

# Try many quaternions (wxyz format)
quats = {
    "identity":        [1, 0, 0, 0],
    "rot_x_90":        [0.7071, 0.7071, 0, 0],
    "rot_x_neg90":     [0.7071, -0.7071, 0, 0],
    "rot_y_90":        [0.7071, 0, 0.7071, 0],
    "rot_y_neg90":     [0.7071, 0, -0.7071, 0],
    "rot_z_90":        [0.7071, 0, 0, 0.7071],
    "rot_z_neg90":     [0.7071, 0, 0, -0.7071],
    "rot_x_180":       [0, 1, 0, 0],
    "rot_y_180":       [0, 0, 1, 0],
    "rot_z_180":       [0, 0, 0, 1],
    "fwd_w3_0_fk":     [0, -0.7071, 0.7071, 0],
    "fwd_w3_pi_fk":    [0, 0.7071, 0.7071, 0],
    "fwd_w3_pi2_fk":   [0, 0, 1, 0],
    "neg_identity":    [-1, 0, 0, 0],
    "xy_45":           [0.5, 0.5, 0.5, 0.5],
    "xy_neg45":        [0.5, -0.5, 0.5, -0.5],
}

joint_names = get_arm_joint_names("ur5e_robotiq")

for name, quat in quats.items():
    pos_t = torch.tensor([target_pos], dtype=torch.float32, device=cuda)
    q_t = torch.tensor([quat], dtype=torch.float32, device=cuda)
    goal = CuPose(position=pos_t, quaternion=q_t)

    result = mg.ik_solver.solve_batch(goal)
    if not result.success[0]:
        print(f"{name:20s}: IK FAILED")
        continue

    # Get joint solution
    joints = result.solution[0, 0].cpu().numpy()

    # Set robot joints and render
    full_qpos = list(joints) + [0] * 8
    qpos_tensor = torch.tensor([full_qpos], dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos_tensor)

    action = torch.zeros(1, raw.agent.action_space.shape[0])
    action[0, :6] = torch.tensor(joints, dtype=torch.float32)
    for _ in range(15):
        env.step(action)

    img = env.render()
    if img is not None:
        img_np = img
        if hasattr(img_np, 'cpu'):
            img_np = img_np[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)

        full_img = Image.fromarray(img_np)
        # Save full and zoomed left view
        full_img.save(f"videos/ik_orient/{name}.png")
        w, h = full_img.size
        left = full_img.crop((0, 0, w // 3, h))
        left = left.resize((left.width * 3, left.height * 3), Image.NEAREST)
        left.save(f"videos/ik_orient/{name}_zoom.png")

    print(f"{name:20s}: IK OK  joints=[{', '.join(f'{j:.2f}' for j in joints)}]")

env.close()
print("\nDone — check videos/ik_orient/ for zoomed images")
