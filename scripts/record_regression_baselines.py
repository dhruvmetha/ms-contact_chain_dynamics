"""Record regression baselines from the current working solver.

Runs the shelf_panda_stick_push_curobo solver with seeds 42-52,
captures per-step TCP trajectory + object positions + push waypoints.
Saves to data/regression/ for use as regression test ground truth.

Run BEFORE refactoring the solver into the StickPush skill.

Usage:
    srun --partition=unlimited --gres=gpu:a6000:1 --time=00:30:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && \
        export CPATH=/usr/local/cuda-12.6/targets/x86_64-linux/include:$CPATH && \
        uv run python scripts/record_regression_baselines.py'
"""
import os
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

import taskbench.envs
import taskbench.agents.panda_stick_long

from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _world_to_curobo_frame, get_arm_joint_names,
    _sapien_qpos_to_cu_joint_state, batched_follow_path,
)
from taskbench.skills.curobo_world import shelf_world
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

ROBOT_UID = "panda_stick_long"
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]
REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
SAFE_QPOS = [0.0002, -1.2996, 0.0003, -2.4930, -0.0009, 1.0177, 0.0]
SHELF = dict(front_x=0.25, depth=0.20, half_w=0.25, floor_z=0.40, thickness=0.01, inner_h=0.30)
SURFACE_Z = SHELF["floor_z"] + SHELF["thickness"]
PUSH_Z = SURFACE_Z + 0.045

SEEDS = list(range(42, 53))
OUT_DIR = Path("data/regression")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_push(rng, shelf):
    """Same sampling as the solver."""
    margin, y_margin = 0.045, 0.12
    max_x = shelf["front_x"] + shelf["depth"] * 0.7
    z = PUSH_Z

    for _ in range(100):
        if rng.random() < 0.5:
            x = rng.uniform(shelf["front_x"] + margin, max_x)
            y1 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            x1, x2 = x, x
        else:
            x1 = rng.uniform(shelf["front_x"] + margin, max_x)
            y1 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)
            x2 = rng.uniform(shelf["front_x"] + margin, max_x)
            y2 = rng.uniform(-shelf["half_w"] + y_margin, shelf["half_w"] - y_margin)

        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if dist >= 0.08:
            return [x1, y1, z], [x2, y2, z]

    x = rng.uniform(shelf["front_x"] + margin, max_x)
    y1 = -shelf["half_w"] + y_margin
    y2 = shelf["half_w"] - y_margin
    return [x, y1, z], [x, y2, z]


# Create env (single, CPU — matches the solver)
env = gym.make(
    "ShelfEnv-v1", num_envs=1, robot_uids=ROBOT_UID,
    robot_base_pose=[-0.4, 0.15, 0.0],
    num_objects=10, reward_mode="none", obs_mode="state_dict",
    render_mode="rgb_array", sim_backend="cpu",
    control_mode="pd_joint_pos",
    shelf=SHELF,
)
raw = env.unwrapped
device = raw.device
cuda = torch.device("cuda:0")
n_arm = len(get_arm_joint_names(ROBOT_UID))

print(f"Recording baselines for seeds {SEEDS}")

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    obs, _ = env.reset(seed=seed)

    # Set rest qpos + settle
    qpos = raw.agent.robot.get_qpos().clone()
    qpos[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos)
    rest_action = torch.zeros(1, n_arm, device=device)
    rest_action[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32, device=device)
    for _ in range(10):
        env.step(rest_action)

    # Build collision world + planner
    g = raw.shelf_geom
    robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
    sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
    sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
    world_shelf = WorldConfig(cuboid=[Cuboid(
        name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
    )])
    motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world_shelf, n_envs=1)
    robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)
    q_tensor = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda)

    # Sample push
    rng = np.random.default_rng(seed)
    p1, p2 = sample_push(rng, SHELF)
    approach_p = [g.front_x - 0.05, p1[1], p1[2]]
    retract_p = [g.front_x - 0.05, p2[1], p2[2]]

    print(f"  p1={[round(x,3) for x in p1]}, p2={[round(x,3) for x in p2]}")

    # --- Stage ---
    start_state = _sapien_qpos_to_cu_joint_state(env, ROBOT_UID, 1)
    approach_base = _world_to_curobo_frame(
        torch.tensor([approach_p], dtype=torch.float32, device=cuda), robot_base_pos
    )
    goal_pose = CuPose(position=approach_base, quaternion=q_tensor)
    plan_config = MotionGenPlanConfig(
        max_attempts=10, enable_graph=False, timeout=15.0, enable_opt=True,
    )
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    if not result.success[0]:
        print(f"  Staging FAILED, skipping")
        continue

    staging_traj = result.optimized_plan.position.cpu()

    # Execute staging
    tcp_trace = []
    quat_trace = []
    qpos_trace = []

    def record_step():
        tcp_trace.append(raw.agent.tcp.pose.p[0].cpu().numpy().copy())
        quat_trace.append(raw.agent.tcp.pose.q[0].cpu().numpy().copy())
        qpos_trace.append(raw.agent.robot.get_qpos()[0, :n_arm].cpu().numpy().copy())

    record_step()
    batched_follow_path(
        env, [staging_traj],
        gripper_state=None, n_envs=1, n_arm_joints=n_arm,
        step_callback=lambda t, o, r: record_step(), refine_steps=10,
    )

    # --- Switch to EE delta ---
    raw.agent.set_control_mode("pd_ee_delta_pose")
    raw.agent.controller.reset()

    def ee_move(target, max_steps, max_delta=0.03):
        target = np.array(target)
        for _ in range(max_steps):
            tcp = raw.agent.tcp.pose.p[0].cpu().numpy()
            delta = np.clip(target - tcp, -max_delta, max_delta)
            if np.linalg.norm(target - tcp) < 0.003:
                break
            action = torch.zeros(1, 6, device=device)
            action[0, 0] = delta[0]
            action[0, 1] = delta[1]
            action[0, 2] = delta[2]
            env.step(action)
            record_step()

    # Entry
    ee_move(p1, 100)
    entry_tcp = raw.agent.tcp.pose.p[0].cpu().numpy().copy()

    # Sweep
    ee_move(p2, 200)
    sweep_tcp = raw.agent.tcp.pose.p[0].cpu().numpy().copy()

    # Retract
    ee_move(retract_p, 100)
    retract_tcp = raw.agent.tcp.pose.p[0].cpu().numpy().copy()

    # Object positions
    obj_positions = []
    for obj in raw.shelf_objects:
        obj_positions.append(obj.pose.p[0].cpu().numpy().copy())

    # Switch back to joint control
    raw.agent.set_control_mode("pd_joint_pos")
    raw.agent.controller.reset()

    # Save
    out_path = OUT_DIR / f"stick_push_seed{seed}.npz"
    np.savez(
        out_path,
        seed=seed,
        p1=np.array(p1),
        p2=np.array(p2),
        approach=np.array(approach_p),
        retract=np.array(retract_p),
        tcp_trace=np.array(tcp_trace),
        quat_trace=np.array(quat_trace),
        qpos_trace=np.array(qpos_trace),
        entry_tcp=entry_tcp,
        sweep_tcp=sweep_tcp,
        retract_tcp=retract_tcp,
        obj_positions=np.array(obj_positions),
    )
    print(f"  Saved {out_path} ({len(tcp_trace)} frames)")

env.close()
print(f"\nDone! Baselines saved to {OUT_DIR}/")
