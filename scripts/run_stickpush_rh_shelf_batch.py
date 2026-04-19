#!/usr/bin/env python3
"""Run stick-push RH planner over many unique shelf scenes.

Behavior:
1) Samples unique ShelfEnv-v1 scenes by seed/state hash.
2) Runs full shelf_stickpush_receding_horizon pipeline on each unique scene.
3) If scene is already solved at root (target already reachable), deletes run artifacts.
4) Otherwise keeps one folder per scene in a single parent directory:
   - scene_XXX_seedYYY_success
   - scene_XXX_seedYYY_fail

Usage:
    uv run python scripts/run_stickpush_rh_shelf_batch.py \
        --num-scenes 100 --num-objects 6 --max-executions 1000
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import taskbench.agents.panda_stick_long  # noqa: F401
import taskbench.envs  # noqa: F401
from taskbench.planners.stickpush_rh.geometry import hash_scene_state
from taskbench.planners.stickpush_rh.state_provider import GTSceneStateProvider
from taskbench.solvers.shelf_stickpush_receding_horizon import (
    ShelfStickPushRecedingHorizonSolver,
)


DEFAULT_SHELF = {
    "front_x": 0.25,
    "depth": 0.20,
    "half_w": 0.25,
    "floor_z": 0.40,
    "thickness": 0.01,
    "inner_h": 0.30,
}

DEFAULT_CYLINDER = {
    "radius": 0.018,
    "half_length": 0.045,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--num-objects", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-seed", type=int, default=20000)
    parser.add_argument("--max-executions", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=256)
    parser.add_argument("--clearance-radius", type=float, default=0.10)
    parser.add_argument("--visual-save-images", action="store_true")
    parser.add_argument("--visual-save-frontier-csv", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="artifacts/stickpush_rh_shelf_batch",
    )
    return parser.parse_args()


def _make_env(*, num_objects: int):
    return gym.make(
        "ShelfEnv-v1",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="panda_stick_long",
        robot_base_pose=[-0.4, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0],
        obs_mode="state",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_objects=int(num_objects),
        shelf=dict(DEFAULT_SHELF),
        cylinder=dict(DEFAULT_CYLINDER),
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = Path(args.artifact_root) / f"{ts}_N{args.num_scenes}_obj{args.num_objects}"
    tmp_root = parent_dir / "_tmp_runs"
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    env = _make_env(num_objects=args.num_objects)
    provider = GTSceneStateProvider(env_index=0)
    solver = ShelfStickPushRecedingHorizonSolver(
        max_executions=int(args.max_executions),
        max_depth=int(args.max_depth),
        clearance_radius=float(args.clearance_radius),
        visual_save_images=bool(args.visual_save_images),
        visual_save_frontier_csv=bool(args.visual_save_frontier_csv),
        artifact_root=str(tmp_root),
    )

    seen_hashes: set[str] = set()
    manifest_rows: list[dict] = []
    duplicates_skipped = 0
    root_solved_skipped = 0
    kept_success = 0
    kept_fail = 0

    seed = int(args.seed_start)
    unique_count = 0

    try:
        while unique_count < int(args.num_scenes):
            if seed > int(args.max_seed):
                raise RuntimeError(
                    f"Reached max_seed={args.max_seed} before collecting "
                    f"{args.num_scenes} unique scenes (collected={unique_count})"
                )

            env.reset(seed=seed)
            scene = provider.get_scene_state(env)
            state_hash = hash_scene_state(scene)
            if state_hash in seen_hashes:
                duplicates_skipped += 1
                seed += 1
                continue

            seen_hashes.add(state_hash)
            unique_count += 1
            scene_idx = unique_count

            print(
                f"[scene {scene_idx:03d}/{args.num_scenes}] seed={seed} "
                f"hash={state_hash[:12]}...",
                flush=True,
            )
            result = solver.solve(env, seed=seed)
            run_dir = Path(result.info["artifact_dir"])
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Missing summary at {summary_path}")
            summary = _load_json(summary_path)

            root_metrics = summary.get("root_metrics", {})
            root_solved = bool(root_metrics.get("solved", False))
            executions = int(summary.get("executions", result.elapsed_steps))

            row = {
                "scene_index": scene_idx,
                "seed": seed,
                "state_hash": state_hash,
                "planner_success": bool(result.success),
                "executions": executions,
                "solve_depth": summary.get("solve_depth"),
                "root_metrics": root_metrics,
                "best_metrics": summary.get("best_metrics"),
            }

            if root_solved and executions == 0:
                root_solved_skipped += 1
                row["status"] = "root_solved_skipped"
                row["artifact_dir"] = None
                if run_dir.exists():
                    shutil.rmtree(run_dir)
            else:
                status = "success" if bool(result.success) else "fail"
                if status == "success":
                    kept_success += 1
                else:
                    kept_fail += 1
                dst = parent_dir / f"scene_{scene_idx:03d}_seed{seed}_{status}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(run_dir), str(dst))
                row["status"] = status
                row["artifact_dir"] = str(dst.resolve())

            manifest_rows.append(row)
            seed += 1

            manifest_payload = {
                "parent_dir": str(parent_dir.resolve()),
                "num_scenes_requested": int(args.num_scenes),
                "num_scenes_unique_processed": int(unique_count),
                "duplicates_skipped": int(duplicates_skipped),
                "root_solved_skipped": int(root_solved_skipped),
                "kept_success": int(kept_success),
                "kept_fail": int(kept_fail),
                "seed_start": int(args.seed_start),
                "next_seed": int(seed),
                "max_seed": int(args.max_seed),
                "max_executions": int(args.max_executions),
                "max_depth": int(args.max_depth),
                "clearance_radius": float(args.clearance_radius),
                "num_objects": int(args.num_objects),
                "visual_save_images": bool(args.visual_save_images),
                "visual_save_frontier_csv": bool(args.visual_save_frontier_csv),
                "scenes": manifest_rows,
            }
            with (parent_dir / "suite_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest_payload, f, indent=2, sort_keys=True)

    finally:
        env.close()
        if tmp_root.exists() and not any(tmp_root.iterdir()):
            tmp_root.rmdir()

    final_summary = {
        "parent_dir": str(parent_dir.resolve()),
        "num_scenes_requested": int(args.num_scenes),
        "num_scenes_unique_processed": int(unique_count),
        "duplicates_skipped": int(duplicates_skipped),
        "root_solved_skipped": int(root_solved_skipped),
        "kept_success": int(kept_success),
        "kept_fail": int(kept_fail),
        "kept_total": int(kept_success + kept_fail),
    }
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
