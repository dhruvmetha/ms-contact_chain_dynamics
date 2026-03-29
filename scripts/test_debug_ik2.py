"""Debug: Compare motion_gen_free IK when created alone vs after motion_gen."""
import torch
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.math import Pose as CuPose
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_world import shelf_world

cuda = torch.device("cuda:0")
Q = [0.7071, 0, -0.7071, 0]
targets = [
    [0.55, -0.10, 0.36],
    [0.55, 0.00, 0.36],
    [0.55, 0.10, 0.36],
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

# Test 1: Create ONLY motion_gen_free (no collision planner before it)
print("=== motion_gen_free ALONE ===")
mg1 = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
q_t = torch.tensor([Q], dtype=torch.float32, device=cuda)
for t in targets:
    pos = torch.tensor([t], dtype=torch.float32, device=cuda)
    goal = CuPose(position=pos, quaternion=q_t)
    r = mg1.ik_solver.solve_batch(goal)
    print(f"  {t}: {'OK' if r.success[0] else 'FAIL'}")

# Test 2: Create motion_gen (collision) THEN motion_gen_free
print("\n=== motion_gen_free AFTER collision planner ===")
world = shelf_world(env, n_envs=1, include_objects=False)[0]
mg_coll = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)
mg2 = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
for t in targets:
    pos = torch.tensor([t], dtype=torch.float32, device=cuda)
    goal = CuPose(position=pos, quaternion=q_t)
    r = mg2.ik_solver.solve_batch(goal)
    print(f"  {t}: {'OK' if r.success[0] else 'FAIL'}")

# Test 3: Same as test 2 but with warmup=False on the second planner
print("\n=== motion_gen_free AFTER collision (no warmup) ===")
mg3 = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1, warmup=False)
mg3.warmup(batch=1)
for t in targets:
    pos = torch.tensor([t], dtype=torch.float32, device=cuda)
    goal = CuPose(position=pos, quaternion=q_t)
    r = mg3.ik_solver.solve_batch(goal)
    print(f"  {t}: {'OK' if r.success[0] else 'FAIL'}")

env.close()
