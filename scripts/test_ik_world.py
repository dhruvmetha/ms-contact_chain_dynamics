"""Test if WorldConfig() properly clears collision for IK."""
import torch
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, get_curobo_base_rotation,
    _world_to_curobo_frame, update_world,
)
from taskbench.skills.curobo_world import shelf_world
from curobo.types.math import Pose as CuPose
from curobo.geom.types import WorldConfig
import gymnasium as gym
import taskbench.envs

cuda = torch.device("cuda:0")
base_rot = get_curobo_base_rotation("ur5e_robotiq")

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state")
env.reset(seed=42)
raw = env.unwrapped
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

world = shelf_world(env, n_envs=1, include_objects=False)[0]
mg = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)

Q = [0.7071, 0, -0.7071, 0]
test_points = [
    [0.55, -0.10, 0.36],
    [0.60, 0.0, 0.36],
    [0.55, 0.20, 0.36],
]
q = torch.tensor([Q], dtype=torch.float32, device=cuda)

print("=== With shelf collision world ===")
for p in test_points:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    pos_base = _world_to_curobo_frame(pos, robot_base_pos, base_rot)
    goal = CuPose(position=pos_base, quaternion=q)
    r = mg.ik_solver.solve_batch(goal)
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")

print("\n=== After update_world(WorldConfig()) ===")
update_world(mg, WorldConfig())
for p in test_points:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    pos_base = _world_to_curobo_frame(pos, robot_base_pos, base_rot)
    goal = CuPose(position=pos_base, quaternion=q)
    r = mg.ik_solver.solve_batch(goal)
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")

print("\n=== Fresh planner, no world ===")
mg2 = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
for p in test_points:
    pos = torch.tensor([p], dtype=torch.float32, device=cuda)
    pos_base = _world_to_curobo_frame(pos, robot_base_pos, base_rot)
    goal = CuPose(position=pos_base, quaternion=q)
    r = mg2.ik_solver.solve_batch(goal)
    print(f"  {p}: {'OK' if r.success[0] else 'FAIL'}")

env.close()
