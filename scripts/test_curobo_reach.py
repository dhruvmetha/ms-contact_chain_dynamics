"""Systematic cuRobo integration test for UR5e + shelf.

Run on GPU node:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:15:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_curobo_reach.py'
"""
import torch
import numpy as np
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _sapien_qpos_to_cu_joint_state,
    get_arm_joint_names,
)
from taskbench.skills.curobo_world import shelf_world, _make_cuboid
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.geom.types import WorldConfig, Cuboid
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

cuda = torch.device("cuda:0")

# ── Setup ─────────────────────────────────────────────────────────
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[-0.1,0,0,0,0,0,1],
               num_objects=0, render_mode=None, obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none",
               shelf={"floor_z": 0.20, "inner_h": 0.70})
env.reset(seed=42)
raw = env.unwrapped
g = raw.shelf_geom

robot_base_world = raw.agent.robot.pose.p[0].cpu().numpy()
print(f"Robot base (world): {robot_base_world}")
print(f"Shelf front_x={g.front_x}, surface_z={g.surface_z}, ceil_z={g.ceil_z}")
print(f"Shelf depth={g.depth}, half_w={g.half_w}")

home_qpos = raw.agent.robot.get_qpos().cpu().numpy()[0]
n_arm = len(get_arm_joint_names("ur5e_robotiq"))
home_arm = home_qpos[:n_arm]
print(f"Home arm qpos: {home_arm}")

# Natural orientation from FK
Q_EE = [0.0, 0.7071068, 0.7071068, 0.0]

# ── Test 1: No obstacles — verify basic reachability ──────────────
print("\n" + "="*60)
print("TEST 1: No obstacles — basic reachability in BASE frame")
print("="*60)

mg_empty = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
start = CuJointState.from_position(
    torch.tensor([home_arm], dtype=torch.float32, device=cuda),
    joint_names=get_arm_joint_names("ur5e_robotiq"),
)

# FK to verify EE pose at home
kin_state = mg_empty.kinematics.get_state(start.position)
print(f"Home EE pos (base): {kin_state.ee_position.cpu().numpy()}")
print(f"Home EE quat (base): {kin_state.ee_quaternion.cpu().numpy()}")

pc = MotionGenPlanConfig(max_attempts=10, enable_graph=False)
q = torch.tensor([Q_EE], dtype=torch.float32, device=cuda)

# Grid of base-frame targets
print("\nReachability grid (base frame, no obstacles):")
for x in [0.2, 0.3, 0.4, 0.5, 0.6]:
    row = []
    for z in [0.3, 0.5, 0.7]:
        goal = CuPose(
            position=torch.tensor([[x, 0, z]], dtype=torch.float32, device=cuda),
            quaternion=q,
        )
        r = mg_empty.plan_batch(start, goal, pc)
        row.append("OK" if r.success[0] else "  ")
    print(f"  x={x:.1f}: z=0.3[{row[0]}] z=0.5[{row[1]}] z=0.7[{row[2]}]")

# ── Test 2: FK comparison with SAPIEN ─────────────────────────────
print("\n" + "="*60)
print("TEST 2: FK comparison — cuRobo vs SAPIEN")
print("="*60)

# SAPIEN TCP pose at home
sapien_tcp_p = raw.agent.tcp.pose.p[0].cpu().numpy()
sapien_tcp_q = raw.agent.tcp.pose.q[0].cpu().numpy()
print(f"SAPIEN TCP pos (world): {sapien_tcp_p}")
print(f"SAPIEN TCP quat (world): {sapien_tcp_q}")
print(f"SAPIEN TCP pos (base):  {sapien_tcp_p - robot_base_world}")

curobo_ee_p = kin_state.ee_position[0].cpu().numpy()
curobo_ee_q = kin_state.ee_quaternion[0].cpu().numpy()
print(f"cuRobo EE pos (base):   {curobo_ee_p}")
print(f"cuRobo EE quat (base):  {curobo_ee_q}")

pos_diff = np.linalg.norm(sapien_tcp_p - robot_base_world - curobo_ee_p)
print(f"Position difference: {pos_diff:.4f}m")
if pos_diff > 0.05:
    print("  WARNING: Large FK mismatch — EE link frames may differ!")

# ── Test 3: Add shelf obstacles incrementally ─────────────────────
print("\n" + "="*60)
print("TEST 3: Incremental obstacle test")
print("="*60)

# Target: just in front of shelf opening in base frame
staging_z = g.surface_z + 0.05
staging_base = [g.front_x - 0.05 - robot_base_world[0], 0, staging_z]
print(f"Staging target (base): {[round(x,3) for x in staging_base]}")

goal = CuPose(
    position=torch.tensor([staging_base], dtype=torch.float32, device=cuda),
    quaternion=q,
)

# Get all shelf cuboids in base frame
world_full = shelf_world(env, n_envs=1, include_objects=False)[0]
all_cuboids = world_full.cuboid

for i in range(len(all_cuboids) + 1):
    if i == 0:
        label = "no obstacles"
        wc = None
    else:
        cuboids = all_cuboids[:i]
        label = ", ".join(c.name.replace("scene-0_", "") for c in cuboids)
        wc = WorldConfig(cuboid=list(cuboids))

    mg = setup_curobo_planner("ur5e_robotiq", world_configs=wc, n_envs=1)
    s = CuJointState.from_position(
        torch.tensor([home_arm], dtype=torch.float32, device=cuda),
        joint_names=get_arm_joint_names("ur5e_robotiq"),
    )
    pc_graph = MotionGenPlanConfig(max_attempts=5, enable_graph=True)
    r = mg.plan_batch(s, goal, pc_graph)
    status = "OK" if r.success[0] else "FAIL"
    print(f"  [{status}] {label}")
    if status == "FAIL" and hasattr(r, 'status'):
        print(f"       -> Status: {r.status}")

# Also test: bottom + left + right walls only (skip back wall)
print("\nCustom subsets:")
by_name = {c.name.replace("scene-0_", ""): c for c in all_cuboids}
subsets = [
    ("bottom only", ["shelf_bottom"]),
    ("bottom + left + right", ["shelf_bottom", "shelf_left", "shelf_right"]),
    ("bottom + back", ["shelf_bottom", "shelf_back"]),
    ("all walls (no legs)", ["shelf_bottom", "shelf_back", "shelf_left", "shelf_right"]),
    ("all", [c.name.replace("scene-0_", "") for c in all_cuboids]),
]
for label, names in subsets:
    cuboids = [by_name[n] for n in names if n in by_name]
    wc = WorldConfig(cuboid=cuboids)
    mg = setup_curobo_planner("ur5e_robotiq", world_configs=wc, n_envs=1)
    s = CuJointState.from_position(
        torch.tensor([home_arm], dtype=torch.float32, device=cuda),
        joint_names=get_arm_joint_names("ur5e_robotiq"),
    )
    r = mg.plan_batch(s, goal, pc_graph)
    status = "OK" if r.success[0] else "FAIL"
    print(f"  [{status}] {label}")

env.close()
print("\nDone.")
