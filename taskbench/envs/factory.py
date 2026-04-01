import gc

import gymnasium as gym
import torch
from omegaconf import OmegaConf

from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import taskbench.envs  # noqa: F401 — register custom envs


def cleanup_env(env):
    """Close a ManiSkill env and free all GPU resources.

    SAPIEN's PhysX GPU cache leaks memory across sequential env
    creation/destruction cycles. This function explicitly clears
    the cache after closing to prevent accumulation.
    """
    env.close()
    try:
        import sapien.physx as physx
        physx.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _task_kwargs(cfg):
    """Extract env constructor kwargs from the task config.

    Everything under ``cfg.task`` except ``env_id`` is forwarded to the
    env constructor as **kwargs.  Uses ``to_container()`` to cleanly
    convert nested structured configs to plain dicts.
    """
    container = OmegaConf.to_container(cfg.task, resolve=True)
    container.pop("env_id", None)
    return container


def make_env(cfg):
    """Create a vectorized ManiSkill env with optional video recording."""
    rt = cfg.runtime
    need_render = rt.record_video or rt.render_mode == "human"
    render_mode = rt.render_mode if need_render else None

    kwargs = _task_kwargs(cfg)

    env = gym.make(
        cfg.task.env_id,
        obs_mode=rt.obs_mode,
        control_mode=rt.control_mode,
        reward_mode=rt.reward_mode,
        num_envs=rt.num_envs,
        max_episode_steps=rt.max_episode_steps,
        render_mode=render_mode,
        **kwargs,
    )

    if rt.record_video and rt.render_mode != "human":
        from datetime import datetime
        video_dir = rt.get("video_dir", "videos")
        video_dir = f"{video_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=rt.max_episode_steps,
        )

    env = ManiSkillVectorEnv(env, auto_reset=True, record_metrics=True)
    return env


def make_batched_env(cfg):
    """Create a GPU-vectorized env for batched solvers.

    Like ``make_env`` but without ``ManiSkillVectorEnv`` wrapper
    (batched solvers manage resets directly to avoid partial-reset conflicts).
    """
    rt = cfg.runtime
    need_render = rt.record_video or rt.render_mode == "human"
    render_mode = rt.render_mode if need_render else None

    kwargs = _task_kwargs(cfg)

    env = gym.make(
        cfg.task.env_id,
        obs_mode=rt.obs_mode,
        control_mode=rt.control_mode,
        reward_mode=rt.reward_mode,
        num_envs=rt.num_envs,
        max_episode_steps=rt.max_episode_steps,
        render_mode=render_mode,
        **kwargs,
    )

    if rt.record_video and rt.render_mode != "human":
        from datetime import datetime
        video_dir = rt.get("video_dir", "videos")
        video_dir = f"{video_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_batched"
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            video_fps=30,
            max_steps_per_video=rt.get("max_steps_per_video", 3000),
        )

    return env


def make_single_env(cfg):
    """Create a single raw gym env for use with the motion planner.

    Forces ``num_envs=1`` and ``sim_backend="cpu"`` so that mplib can
    access ``env.unwrapped`` attributes directly.
    """
    rt = cfg.runtime
    need_render = rt.record_video or rt.render_mode == "human"
    render_mode = rt.render_mode if need_render else None

    kwargs = _task_kwargs(cfg)

    env = gym.make(
        cfg.task.env_id,
        obs_mode=rt.obs_mode,
        control_mode=rt.control_mode,
        reward_mode=rt.reward_mode,
        num_envs=1,
        max_episode_steps=rt.max_episode_steps,
        render_mode=render_mode,
        sim_backend="cpu",
        **kwargs,
    )

    if rt.record_video and rt.render_mode != "human":
        from datetime import datetime
        video_dir = rt.get("video_dir", "videos")
        video_dir = f"{video_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            record_reward=False,
            video_fps=30,
            render_substeps=False,
        )

    return env
