"""Debug: Why does standalone IK work but solver IK fails?

Replicate EXACTLY what the solver does, step by step, and check where it breaks.
"""
import torch
import numpy as np
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, get_arm_joint_names,
    _sapien_qpos_to_cu_joint_state, _world_to_curobo_frame,
)
from taskbench.skills.curobo_world import shelf_world
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState

cuda = torch.device("cuda:0")

# Create env EXACTLY like the solver does (via make_single_env → Hydra config)
env = gym.make(
    "ShelfEnv-v1", num_envs=1, sim_backend="cpu",
    robot_uids="ur5e_robotiq",
    robot_base_pose=[0, 0, 0, 0, 0, 0, 1],  # same as shelf.yaml
    num_objects=5,
    control_mode="pd_joint_pos",
    reward_mode="none",
    obs_mode="state",
    shelf=dict(front_x=0.50, depth=0.20, half_w=0.40,
               floor_z=0.30, thickness=0.01, inner_h=0.35),
)
env.reset(seed=43)  # seed=42+1 like run.py does
raw = env.unwrapped

print(f"Robot base pose p: {raw.agent.robot.pose.p[0].cpu().numpy()}")
print(f"Robot base pose q: {raw.agent.robot.pose.q[0].cpu().numpy()}")
print(f"Shelf: front_x={raw.shelf_geom.front_x}, back_x={raw.shelf_geom.back_x:.2f}")

robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)
print(f"robot_base_pos on cuda: {robot_base_pos}")

# Create BOTH planners like solver does
world_with_objects = shelf_world(env, n_envs=1, include_objects=True)[0]
print(f"\nCollision cuboids: {len(world_with_objects.cuboid)}")
for c in world_with_objects.cuboid:
    print(f"  {c.name}: pose={[round(x,3) for x in c.pose[:3]]}, dims={[round(x,3) for x in c.dims]}")

print("\n=== Creating motion_gen (collision-aware) ===")
motion_gen = setup_curobo_planner("ur5e_robotiq", world_configs=world_with_objects, n_envs=1)

print("\n=== Creating motion_gen_free (no collision) ===")
motion_gen_free = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)

# Check cuRobo config
print(f"\nmotion_gen base_link: {motion_gen.kinematics.base_link}")
print(f"motion_gen_free base_link: {motion_gen_free.kinematics.base_link}")
print(f"motion_gen ee_link: {motion_gen.kinematics.ee_link}")

# FK check at home config
n_arm = len(get_arm_joint_names("ur5e_robotiq"))
arm_q = raw.agent.robot.get_qpos()[0, :n_arm].detach().cpu()
print(f"\nHome joints: {arm_q.numpy()}")

state = CuJointState.from_position(
    arm_q.unsqueeze(0).to(cuda, dtype=torch.float32),
    joint_names=get_arm_joint_names("ur5e_robotiq"),
)
kin = motion_gen_free.kinematics.get_state(state.position)
cu_pos = kin.ee_position[0].cpu().numpy()
ms_pos = raw.agent.tcp.pose.p[0].cpu().numpy()
print(f"cuRobo FK:  [{cu_pos[0]:.4f}, {cu_pos[1]:.4f}, {cu_pos[2]:.4f}]")
print(f"ManiSkill:  [{ms_pos[0]:.4f}, {ms_pos[1]:.4f}, {ms_pos[2]:.4f}]")
err = np.linalg.norm(cu_pos - ms_pos)
print(f"FK error: {err:.4f}m")

# Sample a push like the solver does
from taskbench.solvers.shelf_random_push_curobo import _sample_push
rng = np.random.default_rng(43)
g = raw.shelf_geom
Q_INTO_SHELF = [0.7071, 0.0, -0.7071, 0.0]
q_tensor = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda)

for i in range(10):
    p1, p2 = _sample_push(rng, g, margin=0.03)
    print(f"\nPush {i}: p1={[round(x,3) for x in p1]}, p2={[round(x,3) for x in p2]}")

    # IK at p1 — EXACTLY like solver does
    p1_pos = torch.tensor([p1], dtype=torch.float32, device=cuda)
    p1_pos_base = _world_to_curobo_frame(p1_pos, robot_base_pos)
    print(f"  p1 world: {[round(x,3) for x in p1]}")
    print(f"  p1 curobo: [{p1_pos_base[0,0]:.3f}, {p1_pos_base[0,1]:.3f}, {p1_pos_base[0,2]:.3f}]")

    p1_goal = CuPose(position=p1_pos_base, quaternion=q_tensor)
    p1_ik = motion_gen_free.ik_solver.solve_batch(p1_goal)
    print(f"  IK with motion_gen_free: {'OK' if p1_ik.success[0] else 'FAIL'}")

    # Also try without position transform
    p1_goal_raw = CuPose(position=p1_pos, quaternion=q_tensor)
    p1_ik_raw = motion_gen_free.ik_solver.solve_batch(p1_goal_raw)
    print(f"  IK without transform:    {'OK' if p1_ik_raw.success[0] else 'FAIL'}")

    if p1_ik_raw.success[0]:
        # Check where this IK solution puts the TCP in ManiSkill
        joints = p1_ik_raw.solution[0, 0]
        kin_check = motion_gen_free.kinematics.get_state(joints.unsqueeze(0))
        fk_pos = kin_check.ee_position[0].cpu().numpy()
        print(f"  FK of raw IK: [{fk_pos[0]:.3f}, {fk_pos[1]:.3f}, {fk_pos[2]:.3f}]")

env.close()
