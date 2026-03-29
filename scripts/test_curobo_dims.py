"""Debug: is cuRobo Cuboid.dims full-size or half-size?"""
import torch
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _sapien_qpos_to_cu_joint_state, batched_move_to_pose,
)
from taskbench.skills.curobo_world import _actor_box_params, _make_cuboid
from curobo.geom.types import WorldConfig, Cuboid

cuda = torch.device("cuda:0")
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, render_mode=None, obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none", open_top=True)
env.reset(seed=42)
raw = env.unwrapped

# Check what _make_cuboid produces
for actor in raw.scene.get_all_actors():
    if "shelf_bottom" in actor.name:
        center, half_size = _actor_box_params(actor)
        print(f"SAPIEN shelf_bottom: center={center}, half_size={half_size}")
        cuboid = _make_cuboid(actor.name, center, half_size)
        print(f"cuRobo cuboid: pose={cuboid.pose}, dims={cuboid.dims}")
        break

# cuRobo Cuboid.dims is FULL size (not half).
# If _make_cuboid passes half_size as dims, the obstacle is half the real size.
# That means the collision volume extends ±(half_size/2) from center instead of ±half_size.
# For shelf_bottom: thickness=0.01m, so half_size_z=0.01
#   Wrong (half): extends ±0.005 from z=0.55 -> z=[0.545, 0.555]
#   Right (full): extends ±0.01 from z=0.55 -> z=[0.54, 0.56]
# Both are thin — probably not the issue.
# BUT for the Y dimension: half_w=0.5
#   Wrong (half): extends ±0.25 from y=0 -> y=[-0.25, 0.25]
#   Right (full): extends ±0.5 from y=0 -> y=[-0.5, 0.5]
# The WRONG version makes the shelf bottom only 0.5m wide instead of 1.0m!

print("\n=== Read _make_cuboid source ===")
import inspect
print(inspect.getsource(_make_cuboid))

# Test with correct full-size dims
print("\n=== Testing with half-size vs full-size dims ===")
Q = [0.5, 0.5, 0.5, 0.5]
pos = torch.tensor([[0.4, 0.0, 0.7]], dtype=torch.float32, device=cuda)
q = torch.tensor([Q], dtype=torch.float32, device=cuda)

# Version 1: dims = half_size (what _make_cuboid currently does)
bottom_half = Cuboid(
    name="shelf_bottom",
    pose=[center[0], center[1], center[2], 1, 0, 0, 0],
    dims=[half_size[0], half_size[1], half_size[2]],
)

# Version 2: dims = full_size
bottom_full = Cuboid(
    name="shelf_bottom",
    pose=[center[0], center[1], center[2], 1, 0, 0, 0],
    dims=[half_size[0]*2, half_size[1]*2, half_size[2]*2],
)

for label, cuboid in [("half_size_dims", bottom_half), ("full_size_dims", bottom_full)]:
    print(f"\n  {label}: dims={cuboid.dims}")
    wc = WorldConfig(cuboid=[cuboid])
    mg = setup_curobo_planner("ur5e_robotiq", world_configs=wc, n_envs=1)
    start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
    start.position = start.position.to(cuda)
    r = batched_move_to_pose(mg, start, pos, q, n_envs=1, use_batch_env=False)
    print(f"  Result: {'OK' if r['success'][0] else 'FAIL'}")

# Test with NO obstacles as sanity check
print("\n  no_obstacles:")
mg_empty = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
start.position = start.position.to(cuda)
r = batched_move_to_pose(mg_empty, start, pos, q, n_envs=1, use_batch_env=False)
print(f"  Result: {'OK' if r['success'][0] else 'FAIL'}")

env.close()
