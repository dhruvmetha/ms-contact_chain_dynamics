import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from taskbench.envs.factory import make_env, make_single_env
from taskbench.logger import Logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_random(config, logger: Logger):
    """Run episodes with random actions using the vectorized env."""
    env = make_env(config)
    num_envs = config.runtime.num_envs
    target_episodes = config.run.num_episodes

    episodes_done = 0
    all_returns = []
    all_lengths = []
    all_successes = []

    obs, info = env.reset(seed=config.seed)
    ep_returns = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=int)
    ep_success_once = np.zeros(num_envs, dtype=bool)

    while episodes_done < target_episodes:
        actions = env.action_space.sample()
        obs, rewards, terminations, truncations, infos = env.step(actions)

        if config.runtime.render_mode == "human":
            env.render()

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(terminations, torch.Tensor):
            terminations = terminations.cpu().numpy()
        if isinstance(truncations, torch.Tensor):
            truncations = truncations.cpu().numpy()

        ep_returns += rewards
        ep_lengths += 1

        if "success" in infos:
            success = infos["success"]
            if isinstance(success, torch.Tensor):
                success = success.cpu().numpy()
            ep_success_once |= success.astype(bool)

        dones = terminations | truncations
        done_indices = np.where(dones)[0]

        for idx in done_indices:
            if episodes_done >= target_episodes:
                break
            episodes_done += 1
            ep_ret = float(ep_returns[idx])
            ep_len = int(ep_lengths[idx])
            ep_succ = bool(ep_success_once[idx])

            all_returns.append(ep_ret)
            all_lengths.append(ep_len)
            all_successes.append(ep_succ)

            logger.log_episode(
                {"episode/return": ep_ret, "episode/length": ep_len, "episode/success": int(ep_succ)},
                step=episodes_done,
            )

            if episodes_done % 20 == 0 or episodes_done == target_episodes:
                print(f"[Episode {episodes_done}/{target_episodes}]  return={ep_ret:.2f}  len={ep_len}  success={ep_succ}")

            ep_returns[idx] = 0.0
            ep_lengths[idx] = 0
            ep_success_once[idx] = False

    env.close()
    return all_returns, all_lengths, all_successes


def run_batched(config, logger: Logger):
    """Run episodes with a batched solver using GPU-vectorized envs and cuRobo."""
    from taskbench.solver import get_solver

    solver_kwargs = OmegaConf.select(config.run, "solver_kwargs", default={}) or {}
    solver = get_solver(config.run.solver, **solver_kwargs)

    env = make_env(config)
    recording = config.runtime.record_video
    target_episodes = config.run.num_episodes

    all_returns = []
    all_lengths = []
    all_successes = []

    for ep in range(1, target_episodes + 1):
        result = solver.solve(env, seed=config.seed + ep, cfg=config)

        if recording:
            # ManiSkillVectorEnv._env is the RecordEpisode wrapper
            inner = env._env if hasattr(env, "_env") else env
            if hasattr(inner, "flush_video"):
                inner.flush_video()

        all_returns.append(result.reward)
        all_lengths.append(result.elapsed_steps)
        all_successes.append(result.success)

        logger.log_episode(
            {
                "episode/return": result.reward,
                "episode/length": result.elapsed_steps,
                "episode/success": int(result.success),
            },
            step=ep,
        )

        n_success = result.info.get("n_success", 0)
        n_total = result.info.get("n_total", 0)
        rate = result.info.get("success_rate", 0.0)
        print(
            f"[Episode {ep}/{target_episodes}]  "
            f"batch_success={n_success}/{n_total}  "
            f"rate={rate:.2f}"
        )

    env.close()
    return all_returns, all_lengths, all_successes


def run_solver(config, logger: Logger):
    """Run episodes with a registered solver (motion planner)."""
    from taskbench.solver import get_solver

    solver_kwargs = OmegaConf.select(config.run, "solver_kwargs", default={}) or {}
    solver = get_solver(config.run.solver, **solver_kwargs)

    # Let the solver patch config before env creation (e.g. replay solver
    # merges task config from the demo HDF5).
    if hasattr(solver, "apply_demo_config"):
        solver.apply_demo_config(config)

    env = make_single_env(config)
    target_episodes = config.run.num_episodes

    all_returns = []
    all_lengths = []
    all_successes = []

    recording = config.runtime.record_video

    for ep in range(1, target_episodes + 1):
        result = solver.solve(env, seed=config.seed + ep, cfg=config)

        if recording:
            env.flush_video()

        all_returns.append(result.reward)
        all_lengths.append(result.elapsed_steps)
        all_successes.append(result.success)

        logger.log_episode(
            {
                "episode/return": result.reward,
                "episode/length": result.elapsed_steps,
                "episode/success": int(result.success),
            },
            step=ep,
        )

        # Print extra info keys generically
        extras = []
        if result.failure_reason:
            extras.append(f"failure_reason={result.failure_reason}")
        for k, v in result.info.items():
            # Guard against tensor truthiness (ambiguous for multi-element)
            try:
                if v:
                    extras.append(f"{k}={v}")
            except (RuntimeError, ValueError):
                extras.append(f"{k}={v}")

        rate = np.mean(all_successes)
        extra_str = "  " + "  ".join(extras) if extras else ""
        print(f"[Episode {ep}/{target_episodes}]  success={result.success}  cumulative_rate={rate:.2f}{extra_str}")

    env.close()
    return all_returns, all_lengths, all_successes


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = Logger(cfg)

    mode = OmegaConf.select(cfg.run, "mode", default=None)
    if mode is None:
        # Infer mode from solver name and flags for backward compat
        if cfg.run.solver == "random":
            mode = "random"
        elif OmegaConf.select(cfg.run, "batched", default=False):
            mode = "batched"
        else:
            mode = "single"

    if mode == "random":
        all_returns, all_lengths, all_successes = run_random(cfg, logger)
    elif mode == "batched":
        all_returns, all_lengths, all_successes = run_batched(cfg, logger)
    elif mode == "single":
        all_returns, all_lengths, all_successes = run_solver(cfg, logger)
    else:
        raise ValueError(f"Unknown run.mode={mode!r}; expected 'single', 'batched', or 'random'")

    # Summary
    summary = {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_length": float(np.mean(all_lengths)),
        "success_rate": float(np.mean(all_successes)),
        "num_episodes": len(all_returns),
    }

    print("\n===== Summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    logger.log_summary(summary)
    logger.finish()


if __name__ == "__main__":
    main()
