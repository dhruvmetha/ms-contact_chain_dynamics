#!/usr/bin/env python3
"""View any registered env — GUI window or saved video.

Usage:
    # GUI window (needs display)
    uv run python scripts/view_env.py task=shelf
    uv run python scripts/view_env.py task=shelf task.num_objects=0

    # Save video
    uv run python scripts/view_env.py task=shelf video=shelf_scene.mp4
    uv run python scripts/view_env.py task=tabletop_retrieval video=videos/tabletop.mp4

    # Control duration
    uv run python scripts/view_env.py task=shelf video=out.mp4 steps=200
"""

import sys
from pathlib import Path

import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

import taskbench.envs  # noqa: F401
from mani_skill.utils.wrappers import RecordEpisode


def _pop_extra_args(cfg):
    """Pop view_env-specific keys that aren't part of the taskbench config."""
    OmegaConf.set_struct(cfg, False)
    video = cfg.pop("video", None)
    steps = cfg.pop("steps", 500)
    return video, int(steps)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    video, steps = _pop_extra_args(cfg)

    cfg.runtime.num_envs = 1
    cfg.runtime.control_mode = "pd_joint_delta_pos"

    kwargs = OmegaConf.to_container(cfg.task, resolve=True)
    env_id = kwargs.pop("env_id")

    if video:
        video_path = Path(video)
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=cfg.runtime.obs_mode,
            control_mode=cfg.runtime.control_mode,
            reward_mode=cfg.runtime.reward_mode,
            render_mode="rgb_array",
            sim_backend="cpu",
            **kwargs,
        )
        env = RecordEpisode(
            env,
            output_dir=str(video_path.parent or "."),
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            record_reward=False,
            video_fps=30,
        )
        env.reset(seed=cfg.seed)
        hold_action = env.action_space.sample() * 0
        for _ in range(steps):
            env.step(hold_action)
        env.flush_video()
        # RecordEpisode names files automatically — rename to requested name
        generated = sorted(Path(video_path.parent or ".").glob("*.mp4"))
        if generated:
            latest = generated[-1]
            latest.rename(video_path)
            print(f"Saved {video_path} ({steps} steps)")
    else:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=cfg.runtime.obs_mode,
            control_mode=cfg.runtime.control_mode,
            reward_mode=cfg.runtime.reward_mode,
            render_mode="human",
            sim_backend="cpu",
            **kwargs,
        )
        env.reset(seed=cfg.seed)
        hold_action = env.action_space.sample() * 0
        print(f"Viewing {env_id} — close the window to exit.")
        try:
            while True:
                env.step(hold_action)
                env.render()
        except KeyboardInterrupt:
            pass

    env.close()


if __name__ == "__main__":
    main()
