"""Compare cuRobo FK vs ManiSkill FK at multiple joint configs."""
import torch
import numpy as np
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.robot import JointState as CuJointState

cuda = torch.device("cuda:0")

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0, 0, 0, 1, 0, 0, 0],
               # Use 2F-140 URDF instead to test if arm portion is correct
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
n_arm = len(get_arm_joint_names("ur5e_robotiq"))
n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm
joint_names = get_arm_joint_names("ur5e_robotiq")

# Test configs: vary one joint at a time from home
home = [0, -1.5708, 0, -1.5708, 0, 0]
configs = {
    "home":         home,
    "all_zeros":    [0, 0, 0, 0, 0, 0],
    "j0=0.5":      [0.5, -1.5708, 0, -1.5708, 0, 0],
    "j0=1.0":      [1.0, -1.5708, 0, -1.5708, 0, 0],
    "j0=pi":       [3.14159, -1.5708, 0, -1.5708, 0, 0],
    "j1=-1.0":     [0, -1.0, 0, -1.5708, 0, 0],
    "j2=0.5":      [0, -1.5708, 0.5, -1.5708, 0, 0],
    "j3=-1.0":     [0, -1.5708, 0, -1.0, 0, 0],
    "j4=0.5":      [0, -1.5708, 0, -1.5708, 0.5, 0],
    "j5=0.5":      [0, -1.5708, 0, -1.5708, 0, 0.5],
    "retract":     [0, -2.2, 1.0, -1.383, -1.57, 0],
    "reach_fwd":   [0, -1.0, 0.5, -1.0, -1.5708, 0],
}

print(f"{'Config':<15s} {'cuRobo FK':>30s} {'ManiSkill FK':>30s} {'Error':>8s}")
print("-" * 90)

for name, joints in configs.items():
    # cuRobo FK
    j_t = torch.tensor([joints], dtype=torch.float32, device=cuda)
    state = CuJointState.from_position(j_t, joint_names=joint_names)
    kin = mg.kinematics.get_state(state.position)
    cu = kin.ee_position[0].cpu().numpy()

    # ManiSkill FK — set joints and read TCP
    full_qpos = torch.tensor([joints + [0]*n_gripper], dtype=torch.float32)
    raw.agent.robot.set_qpos(full_qpos)
    for _ in range(5):
        env.step(torch.zeros(1, raw.agent.action_space.shape[0]))
    ms = raw.agent.tcp.pose.p[0].cpu().numpy()

    err = np.linalg.norm(cu - ms)
    match = "OK" if err < 0.01 else "BAD"
    print(f"{name:<15s} [{cu[0]:7.4f},{cu[1]:7.4f},{cu[2]:7.4f}] "
          f"[{ms[0]:7.4f},{ms[1]:7.4f},{ms[2]:7.4f}] {err:7.4f} {match}")

env.close()
