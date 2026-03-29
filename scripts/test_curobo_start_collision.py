"""Find which collision sphere at home qpos intersects with the shelf."""
import torch
import numpy as np
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, _sapien_qpos_to_cu_joint_state
from taskbench.skills.curobo_world import shelf_world

cuda = torch.device("cuda:0")
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, render_mode=None, obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none", open_top=True)
env.reset(seed=42)
raw = env.unwrapped

# Get the world config
world = shelf_world(env, n_envs=1, include_objects=False)[0]

# Setup planner
mg = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)

# Get home joint state
start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
start.position = start.position.to(cuda)

# Get FK + collision spheres at home position
kin_model = mg.kinematics
q_home = start.position
kin_state = kin_model.get_state(q_home)

# Get sphere positions
# The kinematics model should provide link sphere positions
spheres = kin_state.link_spheres_tensor  # shape: (batch, n_spheres, 4) -> x,y,z,r
print(f"Total collision spheres: {spheres.shape[1]}")

# Get sphere names/link mapping from the config
import yaml
import os
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "configs/curobo/ur5e_robotiq_2f_85.yml")
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

coll_spheres = config["robot_cfg"]["kinematics"]["collision_spheres"]
sphere_buffer = config["robot_cfg"]["kinematics"].get("collision_sphere_buffer", 0.005)
print(f"Collision sphere buffer: {sphere_buffer}")

# Map sphere index to link name
sphere_idx = 0
sphere_map = []
for link_name, link_spheres in coll_spheres.items():
    for s in link_spheres:
        sphere_map.append((link_name, s["center"], s["radius"]))
        sphere_idx += 1

# Check each cuboid against each sphere
print(f"\n=== Checking {len(sphere_map)} spheres against {len(world.cuboid)} cuboids ===")
for cuboid in world.cuboid:
    cx, cy, cz = cuboid.pose[0], cuboid.pose[1], cuboid.pose[2]
    dx, dy, dz = cuboid.dims[0]/2, cuboid.dims[1]/2, cuboid.dims[2]/2

    for i in range(spheres.shape[1]):
        sx, sy, sz, sr = spheres[0, i].tolist()
        # Add buffer
        sr_buffered = sr + sphere_buffer

        # Check sphere-AABB intersection
        closest_x = max(cx - dx, min(sx, cx + dx))
        closest_y = max(cy - dy, min(sy, cy + dy))
        closest_z = max(cz - dz, min(sz, cz + dz))
        dist = ((sx - closest_x)**2 + (sy - closest_y)**2 + (sz - closest_z)**2)**0.5

        if dist < sr_buffered:
            link_name = sphere_map[i][0] if i < len(sphere_map) else f"sphere_{i}"
            config_r = sphere_map[i][2] if i < len(sphere_map) else "?"
            print(f"  COLLISION: {cuboid.name} <-> {link_name} sphere {i}")
            print(f"    sphere pos=[{sx:.3f},{sy:.3f},{sz:.3f}] r={sr:.3f} (config r={config_r})")
            print(f"    cuboid center=[{cx:.3f},{cy:.3f},{cz:.3f}] half=[{dx:.3f},{dy:.3f},{dz:.3f}]")
            print(f"    distance={dist:.4f}, buffered_radius={sr_buffered:.4f}")

env.close()
