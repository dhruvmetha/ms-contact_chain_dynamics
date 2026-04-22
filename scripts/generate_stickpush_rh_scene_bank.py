#!/usr/bin/env python3
"""Generate a ShelfEnv-v1 scene bank for stickpush RH heuristic testing.

Usage:
    uv run python scripts/generate_stickpush_rh_scene_bank.py \
        --config configs/solver/shelf_stickpush_rh_scene_bank.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

import taskbench.agents.panda_stick_long  # noqa: F401
import taskbench.envs  # noqa: F401
from taskbench.planners.stickpush_rh.geometry import hash_scene_state
from taskbench.planners.stickpush_rh.state_provider import GTSceneStateProvider


DEFAULT_CONFIG = Path("configs/solver/shelf_stickpush_rh_scene_bank.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Optional OmegaConf dotlist override, e.g. scene_bank.num_scenes_if_fast=20",
    )
    return parser.parse_args()


def _load_cfg(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    node = cfg.get("scene_bank", cfg)
    return OmegaConf.to_container(node, resolve=True)


def _make_env(cfg: dict[str, Any]):
    env_kwargs = {
        "num_envs": 1,
        "sim_backend": str(cfg.get("sim_backend", "cpu")),
        "robot_uids": str(cfg.get("robot_uid", "panda_stick_long")),
        "robot_base_pose": list(cfg.get("robot_base_pose", [-0.4, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0])),
        "obs_mode": str(cfg.get("obs_mode", "state")),
        "reward_mode": str(cfg.get("reward_mode", "none")),
        "control_mode": str(cfg.get("control_mode", "pd_joint_pos")),
        "render_mode": str(cfg.get("render_mode", "rgb_array")),
        "num_objects": int(cfg.get("num_objects", 20)),
    }
    active_count = cfg.get("active_object_count", None)
    if active_count is not None:
        env_kwargs["active_object_count"] = int(active_count)
    shelf_cfg = cfg.get("shelf", None)
    if shelf_cfg is not None:
        env_kwargs["shelf"] = dict(shelf_cfg)
    cylinder_cfg = cfg.get("cylinder", None)
    if cylinder_cfg is not None:
        env_kwargs["cylinder"] = dict(cylinder_cfg)

    return gym.make(str(cfg.get("env_id", "ShelfEnv-v1")), **env_kwargs)


def _scene_active_count(scene) -> int:
    return int(sum(1 for obj in scene.all_objects if obj.active))


def _benchmark_generation(env, provider: GTSceneStateProvider, *, trials: int, seed_start: int) -> dict[str, float]:
    warmups = min(3, max(1, trials))
    for warm_seed in range(seed_start, seed_start + warmups):
        env.reset(seed=int(warm_seed))
        provider.get_scene_state(env)

    timings = []
    for i in range(trials):
        seed = int(seed_start + i)
        t0 = time.perf_counter()
        env.reset(seed=seed)
        provider.get_scene_state(env)
        timings.append(time.perf_counter() - t0)

    arr = np.asarray(timings, dtype=np.float64)
    return {
        "trials": int(trials),
        "mean_sec": float(arr.mean()),
        "median_sec": float(np.median(arr)),
        "p95_sec": float(np.percentile(arr, 95)),
        "max_sec": float(arr.max()),
    }


def _serialize_scene(seed: int, state_hash: str, scene) -> dict[str, Any]:
    objects = []
    for obj in sorted(scene.all_objects, key=lambda o: o.name):
        objects.append(
            {
                "name": str(obj.name),
                "center_xyz": [float(obj.center_xyz[0]), float(obj.center_xyz[1]), float(obj.center_xyz[2])],
                "radius": float(obj.radius),
                "is_target": bool(obj.is_target),
                "active": bool(obj.active),
            }
        )

    return {
        "seed": int(seed),
        "state_hash": str(state_hash),
        "num_active_objects": int(_scene_active_count(scene)),
        "target_name": str(scene.target.name),
        "target_center_xyz": [
            float(scene.target.center_xyz[0]),
            float(scene.target.center_xyz[1]),
            float(scene.target.center_xyz[2]),
        ],
        "shelf": {
            "front_x": float(scene.shelf_front_x),
            "back_x": float(scene.shelf_back_x),
            "half_w": float(scene.shelf_half_w),
            "surface_z": float(scene.surface_z),
        },
        "open_dir_xy": [float(scene.open_dir_xy[0]), float(scene.open_dir_xy[1])],
        "objects": objects,
    }


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config, list(args.overrides))
    out_path = Path(str(cfg["out_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = _make_env(cfg)
    provider = GTSceneStateProvider(env_index=0)

    try:
        benchmark_trials = int(cfg.get("benchmark_trials", 100))
        seed_start = int(cfg.get("seed_start", 0))
        benchmark = _benchmark_generation(
            env,
            provider,
            trials=benchmark_trials,
            seed_start=seed_start,
        )

        reference_seed = int(cfg.get("reference_seed", 0))
        env.reset(seed=reference_seed)
        reference_scene = provider.get_scene_state(env)
        reference_state_hash = str(hash_scene_state(reference_scene))
        reference_active = int(_scene_active_count(reference_scene))
        reference_target_name = str(reference_scene.target.name)

        filter_active_count = cfg.get("filter_active_object_count", None)
        if filter_active_count is None:
            target_active_count = int(reference_active)
        else:
            target_active_count = int(filter_active_count)

        slow_threshold_sec = float(cfg.get("slow_threshold_sec", 60.0))
        num_scenes_if_fast = int(cfg.get("num_scenes_if_fast", 100))
        num_scenes_if_slow = int(cfg.get("num_scenes_if_slow", 10))
        selected_num_scenes = (
            num_scenes_if_slow if float(benchmark["mean_sec"]) >= slow_threshold_sec else num_scenes_if_fast
        )

        require_unique = bool(cfg.get("require_unique_state_hash", True))
        max_seed = int(cfg.get("max_seed", 200000))
        rows: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        attempted = 0
        filtered_wrong_count = 0
        duplicates = 0
        accepted = 0
        seed = seed_start

        while accepted < selected_num_scenes:
            if seed > max_seed:
                raise RuntimeError(
                    f"Reached max_seed={max_seed} before collecting "
                    f"{selected_num_scenes} scenes (accepted={accepted}, attempted={attempted})."
                )
            attempted += 1
            env.reset(seed=seed)
            scene = provider.get_scene_state(env)

            active_count = _scene_active_count(scene)
            if active_count != target_active_count:
                filtered_wrong_count += 1
                seed += 1
                continue

            state_hash = str(hash_scene_state(scene))
            if require_unique and (state_hash in seen_hashes):
                duplicates += 1
                seed += 1
                continue
            seen_hashes.add(state_hash)

            rows.append(_serialize_scene(seed=seed, state_hash=state_hash, scene=scene))
            accepted += 1
            seed += 1

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "env_id": str(cfg.get("env_id", "ShelfEnv-v1")),
            "generator": "scripts/generate_stickpush_rh_scene_bank.py",
            "config_path": str(args.config),
            "config_overrides": list(args.overrides),
            "selection_policy": {
                "reference_seed": int(reference_seed),
                "reference_state_hash": reference_state_hash,
                "reference_target_name": reference_target_name,
                "reference_active_objects": int(reference_active),
                "target_active_objects": int(target_active_count),
                "require_unique_state_hash": bool(require_unique),
                "seed_start": int(seed_start),
                "max_seed": int(max_seed),
                "active_object_count_env_param": cfg.get("active_object_count", None),
            },
            "generation_benchmark": benchmark,
            "generation_choice": {
                "slow_threshold_sec": float(slow_threshold_sec),
                "num_scenes_if_fast": int(num_scenes_if_fast),
                "num_scenes_if_slow": int(num_scenes_if_slow),
                "selected_num_scenes": int(selected_num_scenes),
            },
            "collection_stats": {
                "attempted_resets": int(attempted),
                "accepted_scenes": int(accepted),
                "filtered_wrong_object_count": int(filtered_wrong_count),
                "duplicates_skipped": int(duplicates),
                "next_seed": int(seed),
            },
            "scenes": rows,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    finally:
        env.close()

    print(f"wrote: {out_path}")
    print(f"selected_num_scenes: {selected_num_scenes}")
    print(f"target_active_objects: {target_active_count}")
    print(f"mean_generation_sec: {float(benchmark['mean_sec']):.6f}")


if __name__ == "__main__":
    main()

