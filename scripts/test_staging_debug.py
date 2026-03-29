"""Debug staging planning with base_link_inertia."""
import torch
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, get_arm_joint_names,
    _sapien_qpos_to_cu_joint_state, batched_move_to_pose,
)
from taskbench.skills.curobo_world import shelf_world
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
import gymnasium as gym
import taskbench.envs

cuda = torch.device("cuda:0")

env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, control_mode="pd_joint_pos",
               reward_mode="none", obs_mode="state",
               shelf=dict(front_x=0.50, depth=0.20, half_w=0.40,
                          floor_z=0.30, thickness=0.01, inner_h=0.35))
env.reset(seed=42)
raw = env.unwrapped
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)
print(f"Robot base pos: {robot_base_pos}")

# Test 1: No collision world
print("\n=== No collision world ===")
mg_free = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
target = torch.tensor([[0.45, 0.0, 0.36]], dtype=torch.float32, device=cuda)
q = torch.tensor([[0.7071, 0, -0.7071, 0]], dtype=torch.float32, device=cuda)
plan_cfg = MotionGenPlanConfig(max_attempts=10, enable_graph=True, timeout=10.0)

result = batched_move_to_pose(mg_free, start, target, q,
    n_envs=1, use_batch_env=False,
    robot_base_position=robot_base_pos,
    plan_config=plan_cfg)
print(f"No collision: {'OK' if result['success'][0] else 'FAIL'}")

# Test 2: With shelf collision
print("\n=== With shelf collision ===")
world = shelf_world(env, n_envs=1, include_objects=False)[0]
print(f"Cuboids: {len(world.cuboid)}")
for c in world.cuboid:
    print(f"  {c.name}: pose={c.pose[:3]}, dims={c.dims}")

mg_coll = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)
start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
result = batched_move_to_pose(mg_coll, start, target, q,
    n_envs=1, use_batch_env=False,
    robot_base_position=robot_base_pos,
    plan_config=plan_cfg)
print(f"With collision: {'OK' if result['success'][0] else 'FAIL'}")

# Test 3: Without robot_base_position subtraction
print("\n=== Without base pos subtraction ===")
result = batched_move_to_pose(mg_coll, start, target, q,
    n_envs=1, use_batch_env=False,
    plan_config=plan_cfg)
print(f"No base subtraction: {'OK' if result['success'][0] else 'FAIL'}")

env.close()
