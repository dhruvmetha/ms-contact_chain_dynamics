"""Test if cuRobo IK targets grasp_frame or its parent link."""
import torch
import numpy as np
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.robot import JointState as CuJointState
from curobo.types.math import Pose as CuPose

cuda = torch.device("cuda:0")

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0, 0, 0, 1, 0, 0, 0],
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
n_arm = len(get_arm_joint_names("ur5e_robotiq"))

# IK for a target
Q = [0.7071, 0, -0.7071, 0]
target = [0.55, -0.10, 0.36]
pos = torch.tensor([target], dtype=torch.float32, device=cuda)
q_t = torch.tensor([Q], dtype=torch.float32, device=cuda)
goal = CuPose(position=pos, quaternion=q_t)
r = mg.ik_solver.solve_batch(goal)

if r.success[0]:
    joints = r.solution[0, 0]

    # cuRobo FK at this solution
    kin = mg.kinematics.get_state(joints.unsqueeze(0))
    cu_pos = kin.ee_position[0].cpu().numpy()
    print(f"cuRobo FK at IK solution: [{cu_pos[0]:.4f}, {cu_pos[1]:.4f}, {cu_pos[2]:.4f}]")
    print(f"Target:                   [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
    fk_err = np.linalg.norm(cu_pos - np.array(target))
    print(f"cuRobo FK error: {fk_err:.4f}m")

    # Set in ManiSkill and check ALL relevant links
    n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm
    full_qpos = torch.cat([joints.cpu().unsqueeze(0), torch.zeros(1, n_gripper)], dim=1)
    raw.agent.robot.set_qpos(full_qpos)
    for _ in range(10):
        env.step(torch.zeros(1, raw.agent.action_space.shape[0]))

    # Check every link from wrist to grasp_frame
    for link_name in ["wrist_3_link", "flange", "tool0",
                       "robotiq_arg2f_base_link", "grasp_frame"]:
        link = None
        for l in raw.agent.robot.get_links():
            if l.name == link_name:
                link = l
                break
        if link:
            p = link.pose.p[0].cpu().numpy()
            print(f"  {link_name:30s}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")

env.close()
