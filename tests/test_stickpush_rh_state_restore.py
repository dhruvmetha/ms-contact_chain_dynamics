from __future__ import annotations

import numpy as np
import gymnasium as gym

import taskbench.envs  # noqa: F401


def test_raw_state_snapshot_restore_is_deterministic():
    env = gym.make(
        "ShelfEnv-v1",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="panda",
        robot_base_pose=[-0.615, 0.0, 0.0],
        obs_mode="state",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_objects=6,
    )
    env.reset(seed=123)
    raw = env.unwrapped

    ref_target = raw.target_object.pose.p[0].cpu().numpy().copy()
    ref_objects = np.stack([obj.pose.p[0].cpu().numpy().copy() for obj in raw.shelf_objects], axis=0)
    ref_tcp = raw.agent.tcp.pose.p[0].cpu().numpy().copy()
    state = raw.get_state().clone()

    action = env.action_space.sample()
    env.step(action)

    raw.set_state(state)
    cur_target = raw.target_object.pose.p[0].cpu().numpy().copy()
    cur_objects = np.stack([obj.pose.p[0].cpu().numpy().copy() for obj in raw.shelf_objects], axis=0)
    cur_tcp = raw.agent.tcp.pose.p[0].cpu().numpy().copy()
    env.close()

    assert np.allclose(cur_target, ref_target, atol=1e-6)
    assert np.allclose(cur_objects, ref_objects, atol=1e-6)
    assert np.allclose(cur_tcp, ref_tcp, atol=1e-6)
