#!/usr/bin/env python3
"""Replay a saved `final_plan.json` through StickPush in ShelfEnv-v1.

Usage:
    uv run python scripts/replay_stickpush_rh_plan.py \
        --plan artifacts/stickpush_rh/<run>/final_plan.json --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import taskbench.agents.panda_stick_long  # noqa: F401
import taskbench.envs  # noqa: F401
from taskbench.planners.stickpush_rh.executor import StickPushExecutor
from taskbench.planners.stickpush_rh.types import PlannerAction
from taskbench.skills.curobo_motion import get_arm_joint_names, setup_curobo_planner
from taskbench.skills.stick_push import StickPush
from taskbench.solvers.shelf_panda_stick_push_batched import (
    Q_INTO_SHELF,
    REST_QPOS,
    ROBOT_UID,
    SAFE_QPOS,
)


def _load_actions(plan_path: Path) -> list[PlannerAction]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    out = []
    for row in data.get("actions", []):
        out.append(
            PlannerAction(
                blocker_name=row["blocker_name"],
                theta_deg=float(row["theta_deg"]),
                x_approach=float(row["x_approach"]),
                delta_len_idx=int(row["delta_len_idx"]),
                contact_offset=float(row["contact_offset"]),
                push_len=float(row["push_len"]),
                approach_xyz=np.asarray(row["approach_xyz"], dtype=np.float32),
                entry_xyz=np.asarray(row["entry_xyz"], dtype=np.float32),
                sweep_xyz=np.asarray(row["sweep_xyz"], dtype=np.float32),
                retract_xyz=np.asarray(row["retract_xyz"], dtype=np.float32),
                meta=row.get("meta", {}),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    actions = _load_actions(plan_path)
    if not actions:
        raise RuntimeError(f"No actions in {plan_path}")

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
    env.reset(seed=args.seed)
    raw = env.unwrapped
    n_arm = len(get_arm_joint_names(ROBOT_UID))
    qpos = raw.agent.robot.get_qpos().clone()
    qpos[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32)
    raw.agent.robot.set_qpos(qpos)
    raw.agent.set_control_mode("pd_joint_pos")
    raw.agent.controller.reset()
    rest_action = torch.zeros(1, n_arm, device=raw.device)
    rest_action[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32, device=raw.device)
    for _ in range(10):
        env.step(rest_action)

    from curobo.geom.types import Cuboid, WorldConfig

    g = raw.shelf_geom
    robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
    sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
    sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
    world = WorldConfig(
        cuboid=[Cuboid(name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd)]
    )
    motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world, n_envs=1, warmup=False)
    robot_base_pos = raw.agent.robot.pose.p[0].to(device="cuda:0").unsqueeze(0)
    push_skill = StickPush(
        env,
        motion_gen,
        robot_uid=ROBOT_UID,
        n_envs=1,
        n_arm_joints=n_arm,
        rest_qpos=REST_QPOS,
        robot_base_pos=robot_base_pos,
        safe_start_qpos=SAFE_QPOS,
    )
    executor = StickPushExecutor(push_skill, approach_quaternion_wxyz=Q_INTO_SHELF)

    print(f"Replaying {len(actions)} actions from {plan_path}")
    for i, action in enumerate(actions, start=1):
        result = executor.execute(action)
        print(
            f"[{i:02d}] blocker={action.blocker_name} theta={action.theta_deg:.1f} "
            f"push_len={action.push_len:.3f} success={result.success} "
            f"sweep_dist={result.info['sweep_final_dist']:.4f}"
        )
    env.close()


if __name__ == "__main__":
    main()
