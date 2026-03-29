"""Thorough FK comparison — many random joint configs."""
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
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
n_arm = len(get_arm_joint_names("ur5e_robotiq"))
n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm

# Named configs
named_configs = {
    "home":        [0, -1.5708, 0, -1.5708, 0, 0],
    "all_zeros":   [0, 0, 0, 0, 0, 0],
    "j0=1.0":      [1.0, -1.5708, 0, -1.5708, 0, 0],
    "j0=pi":       [3.14159, -1.5708, 0, -1.5708, 0, 0],
    "j1=-1.0":     [0, -1.0, 0, -1.5708, 0, 0],
    "j2=0.5":      [0, -1.5708, 0.5, -1.5708, 0, 0],
    "j3=-1.0":     [0, -1.5708, 0, -1.0, 0, 0],
    "j4=0.5":      [0, -1.5708, 0, -1.5708, 0.5, 0],
    "j5=0.5":      [0, -1.5708, 0, -1.5708, 0, 0.5],
    "retract":     [0, -2.2, 1.0, -1.383, -1.57, 0],
    "reach_fwd":   [0, -1.0, 0.5, -1.0, -1.5708, 0],
    "reach_side":  [1.5708, -1.0, 0.5, -1.0, -1.5708, 0],
}

# Random configs
rng = np.random.default_rng(42)
for i in range(20):
    joints = rng.uniform(-3.14, 3.14, size=6).tolist()
    named_configs[f"rand_{i}"] = joints

print(f"{'Config':<12s} {'cuRobo FK':>30s} {'ManiSkill FK':>30s} {'Error':>8s}")
print("-" * 90)

max_err = 0
all_ok = True
for name, joints in named_configs.items():
    # cuRobo FK
    j_t = torch.tensor([joints], dtype=torch.float32, device=cuda)
    kin = mg.kinematics.get_state(j_t)
    cu = kin.ee_position[0].cpu().numpy()

    # ManiSkill FK — set_qpos and read directly
    full_qpos = torch.tensor([joints + [0]*n_gripper], dtype=torch.float32)
    raw.agent.robot.set_qpos(full_qpos)
    ms = raw.agent.tcp.pose.p[0].cpu().numpy()

    err = np.linalg.norm(cu - ms)
    max_err = max(max_err, err)
    ok = err < 0.001
    if not ok:
        all_ok = False
    print(f"{name:<12s} [{cu[0]:7.4f},{cu[1]:7.4f},{cu[2]:7.4f}] "
          f"[{ms[0]:7.4f},{ms[1]:7.4f},{ms[2]:7.4f}] {err:7.5f} {'OK' if ok else 'BAD'}")

print(f"\nMax error: {max_err:.6f}m")
print(f"Result: {'ALL MATCH' if all_ok else 'MISMATCH FOUND'}")

# Also check quaternion agreement
print(f"\n{'Config':<12s} {'cuRobo quat':>35s} {'ManiSkill quat':>35s} {'Err':>8s}")
print("-" * 95)
for name in ["home", "all_zeros", "retract", "rand_0", "rand_5"]:
    joints = named_configs[name]
    j_t = torch.tensor([joints], dtype=torch.float32, device=cuda)
    kin = mg.kinematics.get_state(j_t)
    cu_q = kin.ee_quaternion[0].cpu().numpy()

    full_qpos = torch.tensor([joints + [0]*n_gripper], dtype=torch.float32)
    raw.agent.robot.set_qpos(full_qpos)
    ms_q = raw.agent.tcp.pose.q[0].cpu().numpy()

    q_err = min(np.linalg.norm(cu_q - ms_q), np.linalg.norm(cu_q + ms_q))  # handle sign ambiguity
    print(f"{name:<12s} [{cu_q[0]:6.3f},{cu_q[1]:6.3f},{cu_q[2]:6.3f},{cu_q[3]:6.3f}] "
          f"[{ms_q[0]:6.3f},{ms_q[1]:6.3f},{ms_q[2]:6.3f},{ms_q[3]:6.3f}] {q_err:7.5f}")

env.close()
