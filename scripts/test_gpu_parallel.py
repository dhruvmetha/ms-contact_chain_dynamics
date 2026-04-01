"""Massively parallel multi-push data collection using StickPush skill.

Each episode:
  - Random cylinder count (1-15), random packing density
  - N_PUSHES pushes per episode (scene evolves between pushes)
  - Each push: StickPush skill (stage → insert → sweep → retract → rest)

Usage:
    srun --partition=unlimited --gres=gpu:a6000:1 --time=00:30:00 bash -c \
        'export CUDA_HOME=/usr/local/cuda-12.6 && \
         export CPATH=/usr/local/cuda-12.6/targets/x86_64-linux/include:$CPATH && \
         export CUROBO_TORCH_CUDA_GRAPH_RESET=1 && \
         uv run python scripts/test_gpu_parallel.py 4 1 10 --video'
"""
import sys
import logging
import torch
import time
import numpy as np
import gymnasium as gym
from datetime import datetime
from mani_skill.utils.wrappers import RecordEpisode

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

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

# === Create env ===
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
    video_dir = f"videos/parallel_push/{datetime.now().strftime('%Y%m%d_%H%M%S')}_N{N}_ep{N_EPISODES}_push{N_PUSHES}"
    print(f"Videos -> {video_dir}")
    env = RecordEpisode(
        env, output_dir=video_dir,
        save_trajectory=False, save_video=True,
        save_on_reset=False, video_fps=30,
        max_steps_per_video=N_PUSHES * 600,
    )
obs, _ = env.reset(seed=42)
raw = env.unwrapped
device = raw.device
print(f"Env init: {time.time()-t0:.1f}s")

# === cuRobo planner ===
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.geom.types import Cuboid, WorldConfig

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
print("cuRobo ready")

# === StickPush skill ===
from taskbench.skills.stick_push import StickPush

push_skill = StickPush(
    env, mg,
    robot_uid=ROBOT_UID, n_envs=N, n_arm_joints=n_arm,
    rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
    safe_start_qpos=SAFE_QPOS,
)
print("StickPush skill ready")


def sample_push(rng):
    """Sample N random push targets."""
    margin, y_margin = 0.02, 0.04
    max_x = SHELF["front_x"] + SHELF["depth"] - margin
    p1 = np.zeros((N, 3))
    p2 = np.zeros((N, 3))
    for i in range(N):
        push_type = rng.random()
        if push_type < 0.4:
            x = rng.uniform(SHELF["front_x"] + margin, max_x)
            y1 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            while abs(y2 - y1) < 0.06:
                y2 = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            p1[i] = [x, y1, PUSH_Z]
            p2[i] = [x, y2, PUSH_Z]
        elif push_type < 0.7:
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
            y = rng.uniform(-SHELF["half_w"] + y_margin, SHELF["half_w"] - y_margin)
            x1 = rng.uniform(SHELF["front_x"] + margin, SHELF["front_x"] + SHELF["depth"] * 0.5)
            x2 = rng.uniform(x1 + 0.04, max_x)
            p1[i] = [x1, y, PUSH_Z]
            p2[i] = [x2, y, PUSH_Z]
    return p1, p2


def place_cylinders(rng, env_idx):
    """Place 1-15 cylinders per env with random density."""
    from mani_skill.utils.structs.pose import Pose as MSPose
    b = len(env_idx)
    margin = raw.cyl_spec.radius + 0.01
    x_lo, x_hi = g.front_x + margin, g.back_x - margin
    y_lo, y_hi = -g.half_w + margin, g.half_w - margin
    z = g.surface_z + raw.cyl_spec.half_length
    counts = rng.integers(1, MAX_CYLS + 1, size=b)
    density_mult = rng.uniform(2.0, 4.0, size=b)
    min_dists = density_mult * raw.cyl_spec.radius
    CYL_Q = [0.7071068, 0, 0.7071068, 0]

    placed = torch.zeros(b, 0, 2, device=raw.device)
    for i, obj in enumerate(raw.shelf_objects):
        pos = torch.zeros(b, 3, device=raw.device)
        active = torch.tensor([i < counts[e] for e in range(b)], device=raw.device)
        if active.any():
            min_d = torch.tensor(min_dists, device=raw.device, dtype=torch.float32)
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
        pos[~active, 2] = -1.0
        placed = torch.cat([placed, xy.unsqueeze(1)], dim=1)
        obj.set_pose(MSPose.create_from_pq(p=pos, q=CYL_Q))

    print(f"    Cylinders: {counts.min()}-{counts.max()}, density={density_mult.min():.1f}-{density_mult.max():.1f}x")


# === Episode loop ===
print(f"\n{'='*60}")
total_t0 = time.time()
total_pushes = 0

for ep in range(N_EPISODES):
    ep_t0 = time.time()
    rng = np.random.default_rng(42 + ep * 1000)
    print(f"\nEp {ep+1}/{N_EPISODES}:")

    if ep > 0:
        obs, _ = env.reset(seed=42 + ep * 1000)

    place_cylinders(rng, np.arange(N))

    # Set rest qpos for first push
    rest = torch.tensor(REST_QPOS, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)
    qpos = raw.agent.robot.get_qpos().clone()
    qpos[:, :7] = rest
    raw.agent.robot.set_qpos(qpos)
    raw.agent.set_control_mode("pd_joint_pos")
    raw.agent.controller.reset()
    for _ in range(10):
        env.step(rest)

    for push_idx in range(N_PUSHES):
        push_rng = np.random.default_rng(42 + ep * 1000 + push_idx * 100)
        p1_np, p2_np = sample_push(push_rng)
        approach_np = np.stack([
            np.full(N, SHELF["front_x"] - 0.05), p1_np[:, 1], np.full(N, PUSH_Z),
        ], axis=1)
        retract_np = np.stack([
            np.full(N, SHELF["front_x"] - 0.05), p2_np[:, 1], np.full(N, PUSH_Z),
        ], axis=1)

        print(f"  Push {push_idx+1}/{N_PUSHES}:")

        result = push_skill(
            approach_positions=torch.tensor(approach_np, device=cuda, dtype=torch.float32),
            approach_quaternions=torch.tensor(Q_INTO_SHELF, device=cuda, dtype=torch.float32).unsqueeze(0).expand(N, -1),
            entry_positions=torch.tensor(p1_np, device=cuda, dtype=torch.float32),
            sweep_positions=torch.tensor(p2_np, device=cuda, dtype=torch.float32),
            retract_positions=torch.tensor(retract_np, device=cuda, dtype=torch.float32),
        )

        n_ok = result.success_mask.sum().item()
        print(f"    success={n_ok}/{N}, sweep_dist={result.sweep_final_dist.mean():.4f}, "
              f"orient={result.orientation_drift.mean():.4f}")
        total_pushes += N

    ep_dt = time.time() - ep_t0
    print(f"  episode time={ep_dt:.1f}s")

    if RECORD and hasattr(env, "flush_video"):
        env.flush_video()

total_dt = time.time() - total_t0
print(f"\n{'='*60}")
print(f"Done! {N_EPISODES} ep x {N_PUSHES} pushes x {N} envs = {total_pushes} total")
print(f"  {total_dt:.0f}s total, {total_dt / max(total_pushes, 1):.3f}s per push")
print(f"  GPU: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")
if RECORD:
    print(f"  Videos: {video_dir}/")

env.close()
