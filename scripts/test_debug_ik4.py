"""Test: robot_base_pose identity [1,0,0,0] + base_link in cuRobo."""
import torch
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.robot import JointState as CuJointState
from curobo.types.math import Pose as CuPose

cuda = torch.device("cuda:0")

# Use base_link in cuRobo config
# Temporarily override — read the config and change base_link
import yaml
from pathlib import Path
config_path = Path("configs/curobo/ur5e_robotiq_2f_85.yml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)
print(f"Current base_link: {cfg['robot_cfg']['kinematics']['base_link']}")

# Test with IDENTITY robot_base_pose [1,0,0,0]
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0, 0, 0, 1, 0, 0, 0],  # IDENTITY quaternion
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped

print(f"robot.pose.p: {raw.agent.robot.pose.p[0].cpu().numpy()}")
print(f"robot.pose.q: {raw.agent.robot.pose.q[0].cpu().numpy()}")

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
print(f"cuRobo base_link: {mg.kinematics.base_link}")

n_arm = len(get_arm_joint_names("ur5e_robotiq"))
arm_q = raw.agent.robot.get_qpos()[0, :n_arm].detach().cpu()
state = CuJointState.from_position(
    arm_q.unsqueeze(0).to(cuda, dtype=torch.float32),
    joint_names=get_arm_joint_names("ur5e_robotiq"))
kin = mg.kinematics.get_state(state.position)
cu = kin.ee_position[0].cpu().numpy()
ms = raw.agent.tcp.pose.p[0].cpu().numpy()
err = ((cu - ms)**2).sum()**0.5
print(f"\ncuRobo FK: [{cu[0]:.4f}, {cu[1]:.4f}, {cu[2]:.4f}]")
print(f"ManiSkill: [{ms[0]:.4f}, {ms[1]:.4f}, {ms[2]:.4f}]")
print(f"Error: {err:.4f}m  {'MATCH' if err < 0.01 else 'MISMATCH'}")

# IK test
Q = [0.7071, 0, -0.7071, 0]
target = [0.55, -0.10, 0.36]
pos = torch.tensor([target], dtype=torch.float32, device=cuda)
q_t = torch.tensor([Q], dtype=torch.float32, device=cuda)
goal = CuPose(position=pos, quaternion=q_t)
r = mg.ik_solver.solve_batch(goal)
print(f"\nIK at {target}: {'OK' if r.success[0] else 'FAIL'}")

if r.success[0]:
    joints = r.solution[0, 0]
    n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm
    full_qpos = torch.cat([joints.cpu().unsqueeze(0), torch.zeros(1, n_gripper)], dim=1)
    raw.agent.robot.set_qpos(full_qpos)
    for _ in range(10):
        env.step(torch.zeros(1, raw.agent.action_space.shape[0]))
    tcp = raw.agent.tcp.pose.p[0].cpu().numpy()
    print(f"ManiSkill TCP after IK: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]")
    print(f"Target:                 [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
    ik_err = ((tcp - target)**2).sum()**0.5
    print(f"IK error: {ik_err:.4f}m  {'MATCH' if ik_err < 0.02 else 'MISMATCH'}")

env.close()
