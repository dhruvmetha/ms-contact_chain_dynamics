"""Statistical push feasibility — straight-line pushes inside the shelf.

For each random push:
  1. IK at start and end (shelf-only collision, no objects)
  2. Linear joint-space interpolation (like batched_linear_push)
  3. FK at each waypoint → check EE stays inside shelf (no wall/floor/ceiling hits)

This matches what the actual solver does: straight-line push through objects,
only avoiding shelf structure.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:30:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_staging.py'
"""
import torch
import numpy as np
import gymnasium as gym
import os

import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner,
    get_arm_joint_names,
    get_curobo_base_rotation,
    batched_actuate_gripper,
    _world_to_curobo_frame,
)
from taskbench.skills.curobo_world import shelf_world
from taskbench.skills.robot_config import get_robot_config
from curobo.types.math import Pose as CuPose

os.makedirs("videos/push_stats", exist_ok=True)
cuda = torch.device("cuda:0")

Q_INTO_SHELF = [0.7071, 0, -0.7071, 0]
robot_uid = "ur5e_robotiq"
n_arm = len(get_arm_joint_names(robot_uid))
base_rot = get_curobo_base_rotation(robot_uid)

N_PUSHES = 500
SEED = 42
N_INTERP = 20  # waypoints along push path

SHELF = dict(front_x=0.50, depth=0.20, half_w=0.40, floor_z=0.30,
             thickness=0.01, inner_h=0.35)

env = gym.make(
    "ShelfEnv-v1", num_envs=1, sim_backend="cpu",
    robot_uids="ur5e_robotiq",
    robot_base_pose=[0, 0, 0, 0, 0, 0, 1],
    num_objects=0,
    render_mode="rgb_array", obs_mode="state",
    control_mode="pd_joint_pos", reward_mode="none",
    shelf=SHELF,
)
env.reset(seed=SEED)
raw = env.unwrapped
rc = get_robot_config(env)
g = raw.shelf_geom
robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

# Shelf-only collision (no objects — push goes through them)
world = shelf_world(env, n_envs=1, include_objects=False)[0]
mg = setup_curobo_planner(robot_uid, world_configs=world, n_envs=1)

print(f"Shelf: x=[{g.front_x}, {g.back_x:.2f}], y=[{-g.half_w}, {g.half_w}], "
      f"z=[{g.surface_z:.2f}, {g.ceil_z:.2f}]")

# Shelf interior bounding box (for path checking)
SHELF_X_MIN = g.front_x + 0.01
SHELF_X_MAX = g.back_x - 0.01
SHELF_Y_MIN = -g.half_w + 0.02
SHELF_Y_MAX = g.half_w - 0.02
SHELF_Z_MIN = g.surface_z + 0.01
SHELF_Z_MAX = g.ceil_z - 0.02

print(f"Interior bounds: x=[{SHELF_X_MIN:.2f},{SHELF_X_MAX:.2f}] "
      f"y=[{SHELF_Y_MIN:.2f},{SHELF_Y_MAX:.2f}] z=[{SHELF_Z_MIN:.2f},{SHELF_Z_MAX:.2f}]")

# ── Sample random pushes ───────────────────────────────────────────────
rng = np.random.default_rng(SEED)
margin = 0.03

pushes = []
while len(pushes) < N_PUSHES:
    z = rng.uniform(SHELF_Z_MIN + 0.01, SHELF_Z_MAX)
    x1 = rng.uniform(SHELF_X_MIN + margin, SHELF_X_MAX - margin)
    y1 = rng.uniform(SHELF_Y_MIN + margin, SHELF_Y_MAX - margin)

    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(0.03, 0.15)

    x2 = x1 + dist * np.cos(angle)
    y2 = y1 + dist * np.sin(angle)

    # Must stay inside shelf
    if x2 < SHELF_X_MIN + margin or x2 > SHELF_X_MAX - margin:
        continue
    if y2 < SHELF_Y_MIN + margin or y2 > SHELF_Y_MAX - margin:
        continue

    actual_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if actual_dist < 0.02:
        continue

    # Classify
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) > abs(dy) * 2:
        direction = "in_push" if dx > 0 else "pull_back"
    elif abs(dy) > abs(dx) * 2:
        direction = "left_sweep" if dy > 0 else "right_sweep"
    else:
        direction = "diagonal"

    z_frac = (z - g.surface_z) / (g.ceil_z - g.surface_z)
    z_bucket = "low" if z_frac < 0.33 else ("mid" if z_frac < 0.66 else "high")

    dist_bucket = "short" if actual_dist < 0.07 else ("medium" if actual_dist < 0.11 else "long")

    y_region = "center" if abs(y1) < 0.12 else "edge"

    pushes.append({
        "start": [float(x1), float(y1), float(z)],
        "end": [float(x2), float(y2), float(z)],
        "direction": direction,
        "z_bucket": z_bucket,
        "dist_bucket": dist_bucket,
        "y_region": y_region,
        "distance": actual_dist,
        "z": z,
    })

N = len(pushes)
print(f"\nSampled {N} pushes")

# ── Batch IK for all start + end positions ─────────────────────────────
all_positions = [p["start"] for p in pushes] + [p["end"] for p in pushes]
pos_t = torch.tensor(all_positions, dtype=torch.float32, device=cuda)
pos_base = _world_to_curobo_frame(pos_t, robot_base_pos, base_rot)
quats = torch.tensor([Q_INTO_SHELF] * len(all_positions), dtype=torch.float32, device=cuda)

# Step 1: IK for start positions
print("Batch IK for start positions...")
CHUNK = 512
start_pos = pos_base[:N]
start_quats = quats[:N]

start_ik = np.zeros(N, dtype=bool)
start_joints = torch.zeros(N, n_arm)
for s in range(0, N, CHUNK):
    e = min(s + CHUNK, N)
    goal = CuPose(position=start_pos[s:e], quaternion=start_quats[s:e])
    result = mg.ik_solver.solve_batch(goal)
    start_ik[s:e] = result.success.squeeze().cpu().numpy()
    start_joints[s:e] = result.solution[:, 0].cpu()

print(f"Start IK: {start_ik.sum()}/{N} ({100*start_ik.sum()/N:.0f}%)")

# Step 2: IK for end positions, SEEDED with start IK solutions
# This forces the end joint config to be close to the start config,
# so linear interpolation between them stays smooth.
print("Batch IK for end positions (seeded with start solutions)...")
end_pos = pos_base[N:]
end_quats = quats[N:]

end_ik = np.zeros(N, dtype=bool)
end_joints = torch.zeros(N, n_arm)
for s in range(0, N, CHUNK):
    e = min(s + CHUNK, N)
    goal = CuPose(position=end_pos[s:e], quaternion=end_quats[s:e])
    # Seed with start solutions so IK stays in same joint-space neighborhood
    # seed_config shape: (batch, num_seeds, n_arm)
    seed = start_joints[s:e].unsqueeze(1).to(device=cuda, dtype=torch.float32)
    result = mg.ik_solver.solve_batch(goal, seed_config=seed)
    end_ik[s:e] = result.success.squeeze().cpu().numpy()
    end_joints[s:e] = result.solution[:, 0].cpu()

print(f"End IK (seeded): {end_ik.sum()}/{N} ({100*end_ik.sum()/N:.0f}%)")

n_both_ik = sum(1 for i in range(N) if start_ik[i] and end_ik[i])
print(f"Both: {n_both_ik}/{N} ({100*n_both_ik/N:.0f}%)")

# Check how close start/end joints are in joint space
diffs = []
for i in range(N):
    if start_ik[i] and end_ik[i]:
        d = (start_joints[i] - end_joints[i]).abs().max().item()
        diffs.append(d)
if diffs:
    print(f"Max joint diff (start→end): mean={np.mean(diffs):.3f} "
          f"median={np.median(diffs):.3f} max={np.max(diffs):.3f} rad")

# ── Straight-line path check ───────────────────────────────────────────
# For each IK-successful push:
# 1. Interpolate joints linearly (like batched_linear_push)
# 2. FK at each waypoint
# 3. Check EE stays inside shelf bounding box
print(f"\nStraight-line path check ({N_INTERP} waypoints per push)...")

path_ok = np.zeros(N, dtype=bool)
n_checked = 0

for i in range(N):
    if not (start_ik[i] and end_ik[i]):
        continue

    n_checked += 1

    # Interpolate in joint space
    alphas = torch.linspace(0, 1, N_INTERP).unsqueeze(1)  # (N_INTERP, 1)
    js = start_joints[i].unsqueeze(0)  # (1, n_arm)
    je = end_joints[i].unsqueeze(0)
    path = (js + alphas * (je - js)).to(device=cuda, dtype=torch.float32)

    # FK at each waypoint
    kin_state = mg.kinematics.get_state(path)
    ee_pos = kin_state.ee_position.cpu().numpy()  # (N_INTERP, 3)

    # Transform back to world frame (undo Rz(180))
    ee_world = ee_pos.copy()
    ee_world[:, 0] = -ee_pos[:, 0]
    ee_world[:, 1] = -ee_pos[:, 1]

    # Check all waypoints stay inside shelf interior
    inside = (
        (ee_world[:, 0] >= SHELF_X_MIN) & (ee_world[:, 0] <= SHELF_X_MAX) &
        (ee_world[:, 1] >= SHELF_Y_MIN) & (ee_world[:, 1] <= SHELF_Y_MAX) &
        (ee_world[:, 2] >= SHELF_Z_MIN) & (ee_world[:, 2] <= SHELF_Z_MAX)
    )
    path_ok[i] = inside.all()

n_path_ok = path_ok.sum()
print(f"Path OK: {n_path_ok}/{n_checked} ({100*n_path_ok/max(n_checked,1):.0f}%)")

# ── Statistics ─────────────────────────────────────────────────────────
def report_stats(label, indices):
    n = len(indices)
    if n < 5:
        return
    n_ik = sum(1 for i in indices if start_ik[i] and end_ik[i])
    n_path = sum(1 for i in indices if path_ok[i])
    ik_pct = 100 * n_ik / n
    path_pct = 100 * n_path / n
    path_given_ik = 100 * n_path / max(n_ik, 1)
    print(f"  {label:<20s}  n={n:>3d}  IK_both={ik_pct:>4.0f}%  "
          f"Path={path_pct:>4.0f}%  Path|IK={path_given_ik:>4.0f}%")

print(f"\n{'='*70}")
print(f"RESULTS (N={N}, straight-line push, shelf-only collision)")
print(f"{'='*70}")

print(f"\nBy direction:")
for d in ["in_push", "pull_back", "left_sweep", "right_sweep", "diagonal"]:
    idx = [i for i in range(N) if pushes[i]["direction"] == d]
    report_stats(d, idx)

print(f"\nBy height:")
for z in ["low", "mid", "high"]:
    idx = [i for i in range(N) if pushes[i]["z_bucket"] == z]
    report_stats(z, idx)

print(f"\nBy distance:")
for d in ["short", "medium", "long"]:
    idx = [i for i in range(N) if pushes[i]["dist_bucket"] == d]
    report_stats(d, idx)

print(f"\nBy y-region:")
for r in ["center", "edge"]:
    idx = [i for i in range(N) if pushes[i]["y_region"] == r]
    report_stats(r, idx)

print(f"\n{'='*70}")
print(f"OVERALL: {n_path_ok}/{N} = {100*n_path_ok/N:.0f}% feasible pushes")
print(f"  IK both endpoints: {n_both_ik}/{N} = {100*n_both_ik/N:.0f}%")
print(f"  Path given IK: {n_path_ok}/{n_checked} = {100*n_path_ok/max(n_checked,1):.0f}%")
print(f"{'='*70}")

# ── Visualize ──────────────────────────────────────────────────────────
import sapien
import sapien.render
from PIL import Image, ImageDraw

env.reset(seed=SEED)
batched_actuate_gripper(env, rc.gripper_closed, n_envs=1, n_arm_joints=n_arm, steps=3)

green_mat = sapien.render.RenderMaterial(base_color=[0, 0.9, 0, 0.8])
red_mat = sapien.render.RenderMaterial(base_color=[0.9, 0, 0, 0.5])

for i in range(N):
    if not (start_ik[i] and end_ik[i]):
        continue
    mat = green_mat if path_ok[i] else red_mat
    b = raw.scene.create_actor_builder()
    b.add_sphere_visual(radius=0.005, material=mat)
    b.initial_pose = sapien.Pose(p=pushes[i]["start"])
    b.build_static(name=f"s_{i}")
    b = raw.scene.create_actor_builder()
    b.add_sphere_visual(radius=0.003, material=mat)
    b.initial_pose = sapien.Pose(p=pushes[i]["end"])
    b.build_static(name=f"e_{i}")

arm_q = raw.agent.robot.get_qpos()[0, :n_arm].detach()
for _ in range(5):
    actions = torch.zeros(1, n_arm + 1, device=raw.device, dtype=torch.float32)
    actions[0, :n_arm] = arm_q
    actions[0, -1] = rc.gripper_closed
    env.step(actions)

img = env.render()
if img is not None:
    img_np = img[0].cpu().numpy() if hasattr(img, 'cpu') else img
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10),
              f"N={N} straight-line pushes\n"
              f"Path OK: {n_path_ok}/{N} ({100*n_path_ok/N:.0f}%)\n"
              f"Green=feasible, Red=infeasible",
              fill=(255, 255, 0))
    pil_img.save("videos/push_stats/push_feasibility.png")
    print(f"\nImage: videos/push_stats/push_feasibility.png")

env.close()
print("Done")
