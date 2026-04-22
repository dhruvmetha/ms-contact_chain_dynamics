from __future__ import annotations

import gymnasium as gym

import taskbench.agents.panda_stick_long  # noqa: F401
import taskbench.envs  # noqa: F401
from taskbench.planners.stickpush_rh.state_provider import GTSceneStateProvider


def test_shelf_env_fixed_active_object_count_enforced():
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
        num_objects=20,
        active_object_count=13,
    )
    provider = GTSceneStateProvider(env_index=0)
    try:
        for seed in (0, 1, 2, 3, 4):
            env.reset(seed=seed)
            scene = provider.get_scene_state(env)
            active = [obj for obj in scene.all_objects if obj.active]
            assert len(active) == 13
            assert scene.target.active
    finally:
        env.close()

