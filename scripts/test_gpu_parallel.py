"""Massively parallel multi-push data collection.

Each episode:
  - Random cylinder count (1-15), random packing density
  - 3 pushes per episode (scene evolves between pushes)
  - Push 1: cuRobo staging from rest + EE push
  - Push 2-3: EE staging from retract position + EE push
  - Full variety: horizontal, diagonal, depth pushes
"""
import sys
import torch
import time
import numpy as np
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode

import taskbench.envs
import taskbench.agents.panda_stick_long

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_PUSHES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
RECORD = "--video" in sys.argv

SHELF = dict(front_x=0.25, depth=0.20, half_w=0.25, floor_z=0.40, thickness=0.01, inner_h=0.30)
SURFACE_Z = SHELF["floor_z"] + SHELF["thickness"]
PUSH_Z = SURFACE_Z + 0.045
ROBOT_UID = "panda_stick_long"
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]
REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
SAFE_QPOS = [0.0002, -1.2996, 0.0003, -2.4930, -0.0009, 1.0177, 0.0]
MAX_CYLS = 15

gpu = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {gpu} ({mem_gb:.1f} GB), N={N}, episodes={N_EPISODES}, pushes/ep={N_PUSHES}")

# === Create env with max cylinders ===
t0 = time.time()
env = gym.make(
    "ShelfEnv-v1", num_envs=N, robot_uids=ROBOT_UID,
    robot_base_pose=[-0.4, 0.15, 0.0],
    num_objects=MAX_CYLS, reward_mode="none", obs_mode="state",
    render_mode="rgb_array" if RECORD else None,
    control_mode="pd_joint_pos",
    shelf=SHELF,
)
if RECORD:
    env = RecordEpisode(
        env, output_dir="videos/parallel_push",
        save_trajectory=False, save_video=True,
        save_on_reset=False, video_fps=30,
        max_steps_per_video=1200,
    )
obs, _ = env.reset(seed=42)
raw = env.unwrapped
device = raw.device
print(f"Env init: {time.time()-t0:.1f}s")

# === cuRobo planner (once) ===
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _world_to_curobo_frame, get_arm_joint_names,
)
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

g = raw.shelf_geom
robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
world = WorldConfig(cuboid=[Cuboid(
    name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
)])
mg = setup_curobo_planner(ROBOT_UID, world_configs=world, n_envs=1, warmup=False)
cuda = torch.device("cuda:0")
n_arm = len(get_arm_joint_names(ROBOT_UID))
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)
goal_quat = torch.tensor(Q_INTO_SHELF, device=cuda, dtype=torch.float32).unsqueeze(0).expand(N, -1)
plan_config = MotionGenPlanConfig(max_attempts=10, enable_graph=False, timeout=30.0, enable_opt=True)
print("cuRobo ready")


def sample_push(rng):
    """Sample N random push targets."""
    margin, y_margin = 0.02, 0.04
    max_x = SHELF["front_x"] + SHELF["depth"] - margin
    p1 = np.zeros((N, 3))
    p2 = np.zeros((N, 3))
    for i in range(N):
        push_type = rng.random()
        if push_type < 0.4:
            # Horizontal
            x = rng.uniform(SHELF["front_x"] + margin, max_x)
            y1 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            while abs(y2 - y1) < 0.06:
                y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            p1[i] = [x, y1, PUSH_Z]
            p2[i] = [x, y2, PUSH_Z]
        elif push_type < 0.7:
            # Diagonal
            x1 = rng.uniform(SHELF["front_x"] + margin, max_x)
            y1 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            x2 = rng.uniform(SHELF["front_x"] + margin, max_x)
            y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            while (x2 - x1) ** 2 + (y2 - y1) ** 2 < 0.06 ** 2:
                x2 = rng.uniform(SHELF["front_x"] + margin, max_x)
                y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            p1[i] = [x1, y1, PUSH_Z]
            p2[i] = [x2, y2, PUSH_Z]
        else:
            # Depth
            y = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            x1 = rng.uniform(SHELF["front_x"] + margin, SHELF["front_x"] + SHELF["depth"] * 0.5)
            x2 = rng.uniform(x1 + 0.04, max_x)
            p1[i] = [x1, y, PUSH_Z]
            p2[i] = [x2, y, PUSH_Z]
    return p1, p2


def batched_ee_move(targets, max_steps=200, max_delta=0.03, label="", converge_thresh=0.005):
    """Move all N TCPs toward targets. Early stop when all converge."""
    targets_dev = targets.to(device)
    t0 = time.time()
    steps = 0
    for step in range(max_steps):
        tcp = raw.agent.tcp.pose.p
        delta = torch.clamp(targets_dev - tcp, -max_delta, max_delta)
        dists = torch.norm(targets_dev - tcp, dim=1)
        if (dists < converge_thresh).all():
            steps = step
            break
        action = torch.zeros(N, 6, device=device)
        action[:, :3] = delta
        env.step(action)
        steps = step + 1
    dists = torch.norm(targets_dev - raw.agent.tcp.pose.p, dim=1)
    dt = time.time() - t0
    print(f"    {label}: {dt:.1f}s ({steps} steps), "
          f"mean={dists.mean():.4f}, converged={int((dists < 0.01).sum())}/{N}")


def curobo_staging(approach_t):
    """Plan and execute staging from rest to approach points."""
    exact_qpos = torch.tensor(SAFE_QPOS, device=cuda, dtype=torch.float32).unsqueeze(0).expand(N, -1).clone()
    start_state = CuJointState(
        position=exact_qpos.clone(),
        velocity=torch.zeros(N, n_arm, device=cuda),
        acceleration=torch.zeros(N, n_arm, device=cuda),
        joint_names=get_arm_joint_names(ROBOT_UID),
    )
    goal_pos = _world_to_curobo_frame(approach_t.clone(), robot_base_pos)
    goal_pose = CuPose(position=goal_pos.clone(), quaternion=goal_quat.clone())

    result = mg.plan_batch(start_state, goal_pose, plan_config)
    n_ok = result.success.sum().item()

    if n_ok == 0:
        return False

    # Execute trajectories
    trajectories = result.optimized_plan.position
    T = trajectories.shape[1]
    for i in range(N):
        if not result.success[i]:
            trajectories[i] = exact_qpos[i].unsqueeze(0).expand(T, -1)
    traj_dev = trajectories.to(device=device, dtype=torch.float32)
    for t in range(T):
        env.step(traj_dev[:, t, :])
    for _ in range(10):
        env.step(traj_dev[:, -1, :])
    print(f"    Staging: {n_ok}/{N} planned")
    return True


def place_cylinders(rng, env_idx):
    """Place 1-15 cylinders per env with random density. Hide extras underground."""
    from mani_skill.utils.structs.pose import Pose as MSPose

    b = len(env_idx)
    margin = raw.cyl_spec.radius + 0.01
    x_lo = g.front_x + margin
    x_hi = g.back_x - margin
    y_lo = -g.half_w + margin
    y_hi = g.half_w - margin
    z = g.surface_z + raw.cyl_spec.half_length

    # Random count and density per env
    counts = rng.integers(1, MAX_CYLS + 1, size=b)
    # Density: min_dist from 2.0x to 4.0x cylinder radius
    density_mult = rng.uniform(2.0, 4.0, size=b)
    min_dists = density_mult * raw.cyl_spec.radius

    CYL_UPRIGHT_Q = [0.7071068, 0, 0.7071068, 0]

    placed = torch.zeros(b, 0, 2, device=raw.device)
    for i, obj in enumerate(raw.shelf_objects):
        pos = torch.zeros(b, 3, device=raw.device)
        active = torch.tensor([i < counts[e] for e in range(b)], device=raw.device)

        if active.any():
            min_d = torch.tensor(min_dists, device=raw.device, dtype=torch.float32)

            # Try placement
            for _ in range(200):
                xy = torch.rand(b, 2, device=raw.device)
                xy[:, 0] = xy[:, 0] * (x_hi - x_lo) + x_lo
                xy[:, 1] = xy[:, 1] * (y_hi - y_lo) + y_lo
                if placed.shape[1] == 0:
                    break
                diffs = placed - xy.unsqueeze(1)
                dists_2d = torch.norm(diffs, dim=2)
                min_neighbor = dists_2d.min(dim=1).values
                if (min_neighbor[active] > min_d[active]).all():
                    break

            pos[active, 0] = xy[active, 0]
            pos[active, 1] = xy[active, 1]
            pos[active, 2] = z

        # Hide inactive cylinders underground
        pos[~active, 2] = -1.0

        placed = torch.cat([placed, xy.unsqueeze(1)], dim=1)
        obj.set_pose(MSPose.create_from_pq(p=pos, q=CYL_UPRIGHT_Q))

    print(f"    Cylinders: counts={counts.min()}-{counts.max()}, "
          f"density_mult={density_mult.min():.1f}-{density_mult.max():.1f}")


# === Episode loop ===
print(f"\n{'='*60}")
total_t0 = time.time()
total_pushes = 0

for ep in range(N_EPISODES):
    ep_t0 = time.time()
    rng = np.random.default_rng(42 + ep * 1000)
    print(f"\nEp {ep+1}/{N_EPISODES}:")

    # Reset
    env_idx = np.arange(N)
    if ep > 0:
        obs, _ = env.reset(seed=42 + ep * 1000)

    # Place cylinders with random count + density
    place_cylinders(rng, env_idx)

    # Set rest qpos
    rest = torch.tensor(REST_QPOS, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)
    qpos = raw.agent.robot.get_qpos().clone()
    qpos[:, :7] = rest
    raw.agent.robot.set_qpos(qpos)
    for _ in range(10):
        env.step(rest)

    for push_idx in range(N_PUSHES):
        push_rng = np.random.default_rng(42 + ep * 1000 + push_idx * 100)
        p1_np, p2_np = sample_push(push_rng)
        approach_np = np.stack([
            np.full(N, SHELF["front_x"] - 0.05), p1_np[:, 1], np.full(N, PUSH_Z),
        ], axis=1)
        p1_t = torch.tensor(p1_np, device=cuda, dtype=torch.float32)
        p2_t = torch.tensor(p2_np, device=cuda, dtype=torch.float32)
        approach_t = torch.tensor(approach_np, device=cuda, dtype=torch.float32)

        print(f"  Push {push_idx+1}/{N_PUSHES}:")

        if push_idx == 0:
            # First push: cuRobo staging from rest
            if not curobo_staging(approach_t):
                print("    Staging failed, skipping episode")
                break

            raw.agent.set_control_mode("pd_ee_delta_pose")
            raw.agent.controller.reset()
        else:
            # Subsequent pushes: EE move from current position to new approach
            batched_ee_move(approach_t, max_steps=150, label="Re-approach")

        # Entry + Sweep + Retract
        batched_ee_move(p1_t, max_steps=100, label="Entry")
        batched_ee_move(p2_t, max_steps=200, max_delta=0.03, label="Sweep")

        retract_t = torch.zeros_like(p2_t)
        retract_t[:, 0] = SHELF["front_x"] - 0.05
        retract_t[:, 1] = raw.agent.tcp.pose.p[:, 1]
        retract_t[:, 2] = PUSH_Z
        batched_ee_move(retract_t, max_steps=150, label="Retract")

        total_pushes += N

    # Switch back to joint control for next episode
    raw.agent.set_control_mode("pd_joint_pos")
    raw.agent.controller.reset()

    # Check quality
    q = raw.agent.tcp.pose.q
    q_ref = torch.tensor([-0.5, 0.5, -0.5, 0.5], device=device)
    drift = 1.0 - torch.abs(torch.sum(q * q_ref, dim=1))
    ep_dt = time.time() - ep_t0
    print(f"  orient_drift={drift.mean():.4f}, time={ep_dt:.1f}s")

    if RECORD and hasattr(env, "flush_video"):
        env.flush_video()

total_dt = time.time() - total_t0
print(f"\n{'='*60}")
print(f"Done! {N_EPISODES} ep x {N_PUSHES} pushes x {N} envs = {total_pushes} total pushes")
print(f"  {total_dt:.0f}s total, {total_dt / max(total_pushes, 1):.3f}s per push")
print(f"  GPU: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

env.close()
