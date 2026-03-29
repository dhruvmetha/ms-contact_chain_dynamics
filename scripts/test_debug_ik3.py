"""Test the EXACT positions that fail in the solver."""
import torch
from taskbench.skills.curobo_motion import setup_curobo_planner
from curobo.types.math import Pose as CuPose
from taskbench.skills.curobo_world import shelf_world
import gymnasium as gym
import taskbench.envs

cuda = torch.device("cuda:0")
Q = [0.7071, 0, -0.7071, 0]
q_t = torch.tensor([Q], dtype=torch.float32, device=cuda)

# Positions that FAIL in solver
fail_positions = [
    [0.589, -0.137, 0.360],
    [0.585, -0.081, 0.360],
    [0.620, -0.140, 0.360],
    [0.582, -0.125, 0.360],
    [0.588, 0.041, 0.360],
]

# Positions that SUCCEED in standalone
ok_positions = [
    [0.55, -0.10, 0.36],
    [0.55, 0.00, 0.36],
    [0.58, -0.05, 0.36],
]

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state",
               shelf=dict(front_x=0.50, depth=0.20, half_w=0.40,
                          floor_z=0.30, thickness=0.01, inner_h=0.35))
env.reset(seed=42)

# Test 1: Single free planner (like standalone test)
print("=== Single free planner ===")
mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
print("Fail positions:")
for p in fail_positions:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    r = mg.ik_solver.solve_batch(CuPose(position=pos, quaternion=q_t))
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")
print("OK positions:")
for p in ok_positions:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    r = mg.ik_solver.solve_batch(CuPose(position=pos, quaternion=q_t))
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")

# Test 2: Two planners (like solver does)
print("\n=== After creating collision planner first ===")
world = shelf_world(env, n_envs=1, include_objects=True)[0]
mg_coll = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)
mg_free = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
print("Fail positions:")
for p in fail_positions:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    r = mg_free.ik_solver.solve_batch(CuPose(position=pos, quaternion=q_t))
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")
print("OK positions:")
for p in ok_positions:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    r = mg_free.ik_solver.solve_batch(CuPose(position=pos, quaternion=q_t))
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")

env.close()
