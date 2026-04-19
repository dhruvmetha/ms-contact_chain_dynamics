#!/usr/bin/env python3
"""Debug entrypoint for shelf stick-push receding-horizon planning.

Usage:
    uv run python scripts/debug_stickpush_rh_plan.py --seed 42 --max-executions 200
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import taskbench.agents.panda_stick_long  # noqa: F401
import taskbench.envs  # noqa: F401
from taskbench.solvers.shelf_stickpush_receding_horizon import (
    ShelfStickPushRecedingHorizonSolver,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-executions", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--clearance-radius", type=float, default=0.10)
    parser.add_argument("--artifact-root", type=str, default="artifacts/stickpush_rh")
    args = parser.parse_args()

    env = gym.make(
        "ShelfEnv-v1",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="panda_stick_long",
        robot_base_pose=[-0.4, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0],
        obs_mode="state",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
    )
    solver = ShelfStickPushRecedingHorizonSolver(
        max_executions=args.max_executions,
        max_depth=args.max_depth,
        clearance_radius=args.clearance_radius,
        artifact_root=args.artifact_root,
    )
    result = solver.solve(env, seed=args.seed)
    env.close()

    print("success:", result.success)
    print("elapsed_steps:", result.elapsed_steps)
    for k, v in result.info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
