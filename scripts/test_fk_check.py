"""Verify quaternion transform: base_link + Rz(180) position + q_world * conj(q_rz180)."""
import torch
import numpy as np
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, get_arm_joint_names, _world_to_curobo_frame,
)
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
import gymnasium as gym
import taskbench.envs

cuda = torch.device("cuda:0")
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu", robot_uids="ur5e_robotiq",
               robot_base_pose=[0,0,0,0,0,0,1], num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

# Rz(180) for positions
RZ180 = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]], dtype=torch.float32)

# Quaternion transform: q_curobo = q_world * conjugate(q_rz180)
q_rz180_conj = torch.tensor([0.0, 0.0, 0.0, -1.0], device=cuda)
def qmul(q1, q2):
    w1,x1,y1,z1 = q1[...,0],q1[...,1],q1[...,2],q1[...,3]
    w2,x2,y2,z2 = q2[...,0],q2[...,1],q2[...,2],q2[...,3]
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

Q_WORLD = [0.7071, 0, -0.7071, 0]  # gripper points +X in world
q_world = torch.tensor([Q_WORLD], dtype=torch.float32, device=cuda)
q_curobo = qmul(q_world, q_rz180_conj.unsqueeze(0))
print(f"Q_WORLD:  {Q_WORLD}")
print(f"Q_CUROBO: [{q_curobo[0,0]:.4f}, {q_curobo[0,1]:.4f}, {q_curobo[0,2]:.4f}, {q_curobo[0,3]:.4f}]")

# IK with transformed position + quaternion
target_world = [0.55, -0.10, 0.36]
pos = torch.tensor([target_world], dtype=torch.float32, device=cuda)
pos_base = _world_to_curobo_frame(pos, robot_base_pos, RZ180)
print(f"\nTarget world: {target_world}")
print(f"Target curobo: [{pos_base[0,0]:.4f}, {pos_base[0,1]:.4f}, {pos_base[0,2]:.4f}]")

goal = CuPose(position=pos_base, quaternion=q_curobo)
r = mg.ik_solver.solve_batch(goal)
print(f"IK: {'OK' if r.success[0] else 'FAIL'}")

if r.success[0]:
    joints = r.solution[0, 0].unsqueeze(0)

    # Set joints in ManiSkill and check TCP
    n_arm = len(get_arm_joint_names("ur5e_robotiq"))
    n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm
    full_qpos = torch.cat([joints.cpu(), torch.zeros(1, n_gripper)], dim=1)
    raw.agent.robot.set_qpos(full_qpos)
    for _ in range(5):
        env.step(torch.zeros(1, raw.agent.action_space.shape[0]))

    tcp_pos = raw.agent.tcp.pose.p[0].cpu().numpy()
    print(f"\nManiSkill TCP: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
    print(f"Target world:  [{target_world[0]:.4f}, {target_world[1]:.4f}, {target_world[2]:.4f}]")
    err = np.linalg.norm(tcp_pos - np.array(target_world))
    print(f"Error: {err:.4f}m  {'MATCH' if err < 0.02 else 'MISMATCH'}")

env.close()
