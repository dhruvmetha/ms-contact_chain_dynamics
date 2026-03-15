#!/usr/bin/env python3
"""Generate sharded open-table bottle-clutter scene libraries."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v3 as iio
import mani_skill.envs  # noqa: F401 - registers ManiSkill envs
import numpy as np
import sapien
from PIL import Image
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

import taskbench.envs  # noqa: F401 - registers taskbench envs
from taskbench.envs.open_table_defaults import (BOTTLE_BODY_HALF_LENGTH,
                                                BOTTLE_BODY_RADIUS,
                                                BOTTLE_VISUAL_STYLE,
                                                DENSE_LAYOUT_PROB, EDGE_MARGIN,
                                                MIXED_LAYOUT_PROB,
                                                PANDA_TABLE_DEFAULTS,
                                                PLACEMENT_CLEARANCE,
                                                PLACEMENT_JITTER,
                                                PLACEMENT_SPACING, RANDOM_YAW)
from taskbench.envs.open_table_push import (BOTTLE_UPRIGHT_Q,
                                            YCB_MUSTARD_BOTTLE_ID,
                                            _load_ycb_metadata)
from taskbench.envs.open_table_scene import (COMPACT_OPEN_TABLE_CENTER_XY,
                                             COMPACT_OPEN_TABLE_SIZE_XY)
from taskbench.envs.placement import (build_rect_grid, clamp_jitter,
                                      jitter_positions, sample_frontier_cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sharded OpenTableBottleClutter scene libraries."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for dataset outputs.",
    )
    parser.add_argument(
        "--total-scenes",
        type=int,
        default=100_000,
        help="Total number of scenes across all shards.",
    )
    parser.add_argument(
        "--scenes-per-shard",
        type=int,
        default=5_000,
        help="Number of scenes written by each shard.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        required=True,
        help="Zero-based shard index within the dataset.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=10_000,
        help="Base seed; each scene uses base_seed + global_scene_index.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="OpenTableBottleClutter-v1",
        help="Environment ID to instantiate.",
    )
    parser.add_argument(
        "--num-bottles",
        type=int,
        default=20,
        help="Number of bottles to place per scene.",
    )
    parser.add_argument(
        "--layout-mode",
        type=str,
        default="mixed",
        choices=("dense", "mixed", "medium", "spread", "auto"),
        help="Layout sampler mode.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square output image size in pixels when RGB export is enabled.",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Store top-down RGB frames alongside scene state.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=("gzip", "lzf", "none"),
        help="HDF5 compression mode for stored datasets.",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        help="Gzip compression level when compression=gzip.",
    )
    parser.add_argument(
        "--save-preview-pngs",
        action="store_true",
        help="Also save a few preview PNGs per shard for visual inspection.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="Number of preview PNGs to save when --save-preview-pngs is set.",
    )
    parser.add_argument(
        "--hide-robot-in-preview",
        action="store_true",
        help="Render preview PNGs with the Panda moved offstage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing shard if present.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Log progress every N scenes.",
    )
    return parser.parse_args()


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _resize_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    if rgb.shape[0] == image_size and rgb.shape[1] == image_size:
        return rgb.astype(np.uint8, copy=False)
    pil_image = Image.fromarray(rgb)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    resized = pil_image.resize((image_size, image_size), resample=resample)
    return np.asarray(resized, dtype=np.uint8)


def _coerce_render_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def _render_scene(env, image_size: int) -> np.ndarray:
    frame = _coerce_render_frame(env.render())
    return _resize_rgb(frame, image_size)


def _render_preview_scene(env, image_size: int, *, hide_robot: bool) -> np.ndarray:
    if not hide_robot:
        return _render_scene(env, image_size)

    raw = env.unwrapped
    robot = raw.agent.robot
    root_position = robot.pose.p[0].detach().cpu().numpy().astype(np.float32)
    root_quat = robot.pose.q[0].detach().cpu().numpy().astype(np.float32)
    try:
        robot.set_root_pose(sapien.Pose([-5.0, 0.0, 0.0], root_quat.tolist()))
        return _render_scene(env, image_size)
    finally:
        robot.set_root_pose(sapien.Pose(root_position.tolist(), root_quat.tolist()))


def _use_algorithmic_scene_sampler(args: argparse.Namespace) -> bool:
    return (not _should_render(args)) and args.env_id == "OpenTableBottleClutter-v1"


def _algorithmic_workspace_bounds() -> tuple[np.ndarray, np.ndarray]:
    center_xy = np.asarray(COMPACT_OPEN_TABLE_CENTER_XY, dtype=np.float32)
    half_extents_xy = 0.5 * np.asarray(
        COMPACT_OPEN_TABLE_SIZE_XY[[1, 0]], dtype=np.float32
    )
    lo_xy = center_xy - half_extents_xy
    hi_xy = center_xy + half_extents_xy
    return lo_xy.astype(np.float32), hi_xy.astype(np.float32)


def _algorithmic_visual_footprint_radius() -> float:
    radius = float(BOTTLE_BODY_RADIUS)
    meta = _load_ycb_metadata(YCB_MUSTARD_BOTTLE_ID)
    bbox = meta["bbox"]
    half_extent_x = max(abs(float(bbox["min"][0])), abs(float(bbox["max"][0])))
    half_extent_y = max(abs(float(bbox["min"][1])), abs(float(bbox["max"][1])))
    half_extent_xy = max(half_extent_x, half_extent_y)
    if half_extent_xy <= 0:
        raise ValueError(f"Invalid YCB bottle metadata for {YCB_MUSTARD_BOTTLE_ID!r}")
    scale = radius / half_extent_xy
    circumscribed_radius = scale * float(np.hypot(half_extent_x, half_extent_y))
    return max(radius, circumscribed_radius)


def _sample_algorithmic_bottle_quaternion(rng: np.random.Generator) -> np.ndarray:
    if not RANDOM_YAW:
        return np.asarray(BOTTLE_UPRIGHT_Q, dtype=np.float32)
    yaw = float(rng.uniform(-np.pi, np.pi))
    yaw_q = euler2quat(0.0, 0.0, yaw)
    return np.asarray(qmult(yaw_q, BOTTLE_UPRIGHT_Q), dtype=np.float32)


def _sample_algorithmic_scene_spec(
    args: argparse.Namespace, *, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    workspace_lo_xy, workspace_hi_xy = _algorithmic_workspace_bounds()
    footprint_margin = EDGE_MARGIN + _algorithmic_visual_footprint_radius()
    placement_lo_xy = workspace_lo_xy + footprint_margin
    placement_hi_xy = workspace_hi_xy - footprint_margin
    if np.any(placement_hi_xy < placement_lo_xy):
        raise ValueError("Configured edge margin leaves no usable tabletop area")

    placement_grid = build_rect_grid(
        placement_lo_xy, placement_hi_xy, PLACEMENT_SPACING
    )
    min_center_distance = 2.0 * float(BOTTLE_BODY_RADIUS) + PLACEMENT_CLEARANCE
    placement_jitter = clamp_jitter(
        PLACEMENT_JITTER,
        grid_spacing=placement_grid.spacing,
        min_center_distance=min_center_distance,
    )
    occupied_indices, layout_meta = sample_frontier_cells(
        rng,
        placement_grid,
        num_cells=args.num_bottles,
        mode=args.layout_mode,
        dense_layout_prob=DENSE_LAYOUT_PROB,
        mixed_layout_prob=MIXED_LAYOUT_PROB,
    )
    rng.shuffle(occupied_indices)
    centers_xy = placement_grid.centers_xy[occupied_indices]
    positions_xy = jitter_positions(
        rng,
        centers_xy,
        max_jitter=placement_jitter,
        lo_xy=placement_lo_xy,
        hi_xy=placement_hi_xy,
    )
    positions_xyz = np.concatenate(
        [
            positions_xy,
            np.full(
                (args.num_bottles, 1),
                BOTTLE_BODY_HALF_LENGTH,
                dtype=np.float32,
            ),
        ],
        axis=1,
    ).astype(np.float32)
    quats_wxyz = np.stack(
        [_sample_algorithmic_bottle_quaternion(rng) for _ in range(args.num_bottles)],
        axis=0,
    ).astype(np.float32)
    scene_layout = {
        **layout_meta,
        "occupied_indices": occupied_indices.astype(np.int32),
        "positions_xy": positions_xy.astype(np.float32),
    }
    return {
        "env_id": args.env_id,
        "num_bottles": int(args.num_bottles),
        "target_idx": int(args.num_bottles - 1),
        "object_names": [f"bottle_{i}" for i in range(args.num_bottles)],
        "object_positions_xyz": positions_xyz,
        "object_quats_wxyz": quats_wxyz,
        "robot_root_position_xyz": PANDA_TABLE_DEFAULTS.root_position_array(),
        "robot_root_quat_wxyz": PANDA_TABLE_DEFAULTS.root_quat_array(),
        "robot_qpos": PANDA_TABLE_DEFAULTS.home_qpos_array(),
        "workspace_lo_xy": workspace_lo_xy,
        "workspace_hi_xy": workspace_hi_xy,
        "layout": scene_layout,
    }


def _should_render(args: argparse.Namespace) -> bool:
    return bool(args.include_rgb or args.save_preview_pngs)


def _compression_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": args.compression}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.gzip_level)
        kwargs["shuffle"] = True
    return kwargs


def _write_root_config(
    output_root: Path,
    *,
    args: argparse.Namespace,
    shard_count: int,
    env_render_size: int | None,
) -> None:
    config = {
        "env_id": args.env_id,
        "total_scenes": int(args.total_scenes),
        "scenes_per_shard": int(args.scenes_per_shard),
        "shard_count": int(shard_count),
        "num_bottles": int(args.num_bottles),
        "layout_mode": args.layout_mode,
        "base_seed": int(args.base_seed),
        "image_size": int(args.image_size) if _should_render(args) else None,
        "include_rgb": bool(args.include_rgb),
        "hide_robot_in_preview": bool(args.hide_robot_in_preview),
        "generation_backend": (
            "algorithmic_sampler" if _use_algorithmic_scene_sampler(args) else "env"
        ),
        "source_render_size": env_render_size,
        "compression": args.compression,
        "gzip_level": int(args.gzip_level),
    }
    config_path = output_root / "dataset_config.json"
    try:
        with config_path.open("x", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    except FileExistsError:
        with config_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != config:
            raise RuntimeError(
                f"Existing dataset_config.json at {config_path} does not match the "
                "requested generation settings"
            )


def _acquire_shard_lock(lock_path: Path) -> int | None:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    payload = {
        "pid": os.getpid(),
        "created_at_unix": time.time(),
    }
    os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
    os.write(fd, b"\n")
    return fd


def _release_shard_lock(lock_path: Path, fd: int | None) -> None:
    if fd is not None:
        os.close(fd)
    lock_path.unlink(missing_ok=True)


def _init_env(args: argparse.Namespace):
    render_mode = "rgb_array" if _should_render(args) else None
    return gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_pos",
        reward_mode="none",
        render_mode=render_mode,
        sim_backend="cpu",
        num_bottles=args.num_bottles,
        layout_mode=args.layout_mode,
    )


def main() -> None:
    args = parse_args()

    if args.total_scenes <= 0:
        raise ValueError("--total-scenes must be positive")
    if args.scenes_per_shard <= 0:
        raise ValueError("--scenes-per-shard must be positive")
    if _should_render(args) and args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    shard_count = math.ceil(args.total_scenes / args.scenes_per_shard)
    if args.shard_index < 0 or args.shard_index >= shard_count:
        raise ValueError(
            f"--shard-index must be in [0, {shard_count - 1}], got {args.shard_index}"
        )

    output_root = args.output_root.resolve()
    shard_dir = output_root / "shards"
    manifest_dir = output_root / "manifests"
    preview_dir = output_root / "previews"
    for path in (output_root, shard_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_preview_pngs:
        preview_dir.mkdir(parents=True, exist_ok=True)

    start_index = args.shard_index * args.scenes_per_shard
    scene_count = min(args.scenes_per_shard, args.total_scenes - start_index)
    end_index = start_index + scene_count

    shard_stem = f"scenes_{start_index:06d}_{end_index - 1:06d}"
    shard_path = shard_dir / f"{shard_stem}.hdf5"
    manifest_path = manifest_dir / f"{shard_stem}.json"
    lock_path = shard_dir / f"{shard_stem}.lock"
    shard_exists = shard_path.exists()
    manifest_exists = manifest_path.exists()
    if (shard_exists or manifest_exists) and not args.overwrite:
        if shard_exists and manifest_exists:
            print(f"[skip] {shard_path} already exists")
            return
        raise RuntimeError(
            f"Detected partial shard state for {shard_stem}: "
            f"hdf5_exists={shard_exists}, manifest_exists={manifest_exists}"
        )

    lock_fd = _acquire_shard_lock(lock_path)
    if lock_fd is None:
        print(f"[skip] lock exists for {shard_stem}; another worker is handling it")
        return

    temp_manifest_path: Path | None = None
    temp_path: Path | None = None
    try:
        for stale_temp in shard_dir.glob(f"{shard_stem}.*.tmp.hdf5"):
            stale_temp.unlink(missing_ok=True)
        for stale_temp in manifest_dir.glob(f"{shard_stem}.*.tmp.json"):
            stale_temp.unlink(missing_ok=True)

        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f"{shard_stem}.", suffix=".tmp.hdf5", dir=str(shard_dir)
        )
        os.close(temp_fd)
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()

        env = None
        raw = None
        if not _use_algorithmic_scene_sampler(args):
            env = _init_env(args)
            raw = env.unwrapped

        compression_kwargs = _compression_kwargs(args)
        rgb_ds = None
        pos_ds = None
        quat_ds = None
        robot_root_pos_ds = None
        robot_root_quat_ds = None
        robot_qpos_ds = None
        seed_ds = None
        layout_mode_ds = None
        layout_variant_ds = None
        scene_json_ds = None
        dt = h5py.string_dtype(encoding="utf-8")
        source_render_size: int | None = None

        start_time = time.time()
        try:
            with h5py.File(temp_path, "w") as f:
                meta = f.create_group("metadata")
                meta.attrs["env_id"] = args.env_id
                meta.attrs["num_scenes"] = int(scene_count)
                meta.attrs["num_bottles"] = int(args.num_bottles)
                meta.attrs["layout_mode"] = args.layout_mode
                meta.attrs["base_seed"] = int(args.base_seed)
                meta.attrs["shard_index"] = int(args.shard_index)
                meta.attrs["start_index"] = int(start_index)
                meta.attrs["end_index"] = int(end_index)
                meta.attrs["include_rgb"] = bool(args.include_rgb)
                meta.attrs["hide_robot_in_preview"] = bool(args.hide_robot_in_preview)
                meta.attrs["generation_backend"] = (
                    "algorithmic_sampler"
                    if _use_algorithmic_scene_sampler(args)
                    else "env"
                )
                meta.attrs["bottle_visual_style"] = (
                    BOTTLE_VISUAL_STYLE
                    if raw is None
                    else getattr(raw, "bottle_visual_style", "")
                )
                meta.attrs["target_index"] = int(
                    args.num_bottles - 1
                    if raw is None
                    else getattr(raw, "target_idx", -1)
                )
                if _should_render(args):
                    meta.attrs["image_size"] = int(args.image_size)

                objects_grp = f.create_group("objects")
                robot_grp = f.create_group("robot")
                scene_grp = f.create_group("scene")

                for local_idx in range(scene_count):
                    global_idx = start_index + local_idx
                    seed = int(args.base_seed + global_idx)
                    if raw is None:
                        scene_spec = _sample_algorithmic_scene_spec(args, seed=seed)
                    else:
                        env.reset(seed=seed)
                        scene_spec = raw.get_scene_spec()
                    names = list(scene_spec["object_names"])
                    positions = np.asarray(
                        scene_spec["object_positions_xyz"], dtype=np.float32
                    )
                    quats = np.asarray(
                        scene_spec["object_quats_wxyz"], dtype=np.float32
                    )
                    robot_root_position = np.asarray(
                        scene_spec["robot_root_position_xyz"], dtype=np.float32
                    )
                    robot_root_quat = np.asarray(
                        scene_spec["robot_root_quat_wxyz"], dtype=np.float32
                    )
                    robot_qpos = np.asarray(scene_spec["robot_qpos"], dtype=np.float32)
                    scene_layout = scene_spec["layout"]
                    scene_layout_json = json.dumps(
                        _jsonify(scene_layout), sort_keys=True
                    )
                    layout_mode = str(scene_layout.get("layout_mode", args.layout_mode))
                    layout_variant = str(scene_layout.get("layout_variant", ""))
                    rgb = None
                    if _should_render(args):
                        rgb = _render_scene(env, args.image_size)

                    if pos_ds is None:
                        if _should_render(args):
                            source_frame = _coerce_render_frame(env.render())
                            source_render_size = int(source_frame.shape[0])
                        _write_root_config(
                            output_root,
                            args=args,
                            shard_count=shard_count,
                            env_render_size=source_render_size,
                        )
                        if source_render_size is not None:
                            meta.attrs["source_render_size"] = source_render_size
                        meta.attrs["workspace_lo_xy"] = np.asarray(
                            scene_spec["workspace_lo_xy"], dtype=np.float32
                        )
                        meta.attrs["workspace_hi_xy"] = np.asarray(
                            scene_spec["workspace_hi_xy"], dtype=np.float32
                        )

                        if args.include_rgb:
                            rgb_ds = f.create_dataset(
                                "rgb",
                                shape=(
                                    scene_count,
                                    rgb.shape[0],
                                    rgb.shape[1],
                                    rgb.shape[2],
                                ),
                                dtype=np.uint8,
                                chunks=(1, rgb.shape[0], rgb.shape[1], rgb.shape[2]),
                                **compression_kwargs,
                            )
                        pos_ds = objects_grp.create_dataset(
                            "position_xyz",
                            shape=(scene_count, positions.shape[0], positions.shape[1]),
                            dtype=np.float32,
                            chunks=(1, positions.shape[0], positions.shape[1]),
                            **compression_kwargs,
                        )
                        quat_ds = objects_grp.create_dataset(
                            "quat_wxyz",
                            shape=(scene_count, quats.shape[0], quats.shape[1]),
                            dtype=np.float32,
                            chunks=(1, quats.shape[0], quats.shape[1]),
                            **compression_kwargs,
                        )
                        robot_root_pos_ds = robot_grp.create_dataset(
                            "root_position_xyz",
                            shape=(scene_count, robot_root_position.shape[0]),
                            dtype=np.float32,
                            chunks=(1, robot_root_position.shape[0]),
                            **compression_kwargs,
                        )
                        robot_root_quat_ds = robot_grp.create_dataset(
                            "root_quat_wxyz",
                            shape=(scene_count, robot_root_quat.shape[0]),
                            dtype=np.float32,
                            chunks=(1, robot_root_quat.shape[0]),
                            **compression_kwargs,
                        )
                        robot_qpos_ds = robot_grp.create_dataset(
                            "qpos",
                            shape=(scene_count, robot_qpos.shape[0]),
                            dtype=np.float32,
                            chunks=(1, robot_qpos.shape[0]),
                            **compression_kwargs,
                        )
                        seed_ds = f.create_dataset(
                            "seed", shape=(scene_count,), dtype=np.int64
                        )
                        layout_mode_ds = scene_grp.create_dataset(
                            "layout_mode", shape=(scene_count,), dtype=dt
                        )
                        layout_variant_ds = scene_grp.create_dataset(
                            "layout_variant", shape=(scene_count,), dtype=dt
                        )
                        scene_json_ds = scene_grp.create_dataset(
                            "layout_json", shape=(scene_count,), dtype=dt
                        )
                        objects_grp.create_dataset(
                            "name", data=np.asarray(names, dtype=object), dtype=dt
                        )

                    pos_ds[local_idx] = positions
                    quat_ds[local_idx] = quats
                    robot_root_pos_ds[local_idx] = robot_root_position
                    robot_root_quat_ds[local_idx] = robot_root_quat
                    robot_qpos_ds[local_idx] = robot_qpos
                    seed_ds[local_idx] = seed
                    layout_mode_ds[local_idx] = layout_mode
                    layout_variant_ds[local_idx] = layout_variant
                    scene_json_ds[local_idx] = scene_layout_json
                    if args.include_rgb:
                        rgb_ds[local_idx] = rgb

                    if args.save_preview_pngs and local_idx < args.preview_count:
                        preview_path = (
                            preview_dir / f"{shard_stem}_scene_{local_idx:04d}.png"
                        )
                        preview_rgb = _render_preview_scene(
                            env,
                            args.image_size,
                            hide_robot=bool(args.hide_robot_in_preview),
                        )
                        iio.imwrite(preview_path, preview_rgb)

                    if args.progress_every > 0 and (
                        local_idx == 0
                        or (local_idx + 1) % args.progress_every == 0
                        or local_idx + 1 == scene_count
                    ):
                        elapsed = time.time() - start_time
                        rate = (local_idx + 1) / max(elapsed, 1e-6)
                        print(
                            f"[shard {args.shard_index:03d}] "
                            f"{local_idx + 1}/{scene_count} scenes "
                            f"({rate:.2f} scenes/s)"
                        )

                elapsed = time.time() - start_time
                meta.attrs["elapsed_sec"] = float(elapsed)
                meta.attrs["scenes_per_sec"] = float(scene_count / max(elapsed, 1e-6))

            temp_path.replace(shard_path)
        finally:
            if env is not None:
                env.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        manifest = {
            "shard_index": int(args.shard_index),
            "start_index": int(start_index),
            "end_index": int(end_index),
            "scene_count": int(scene_count),
            "base_seed": int(args.base_seed),
            "seed_start": int(args.base_seed + start_index),
            "seed_end_exclusive": int(args.base_seed + end_index),
            "num_bottles": int(args.num_bottles),
            "layout_mode": args.layout_mode,
            "include_rgb": bool(args.include_rgb),
            "hide_robot_in_preview": bool(args.hide_robot_in_preview),
            "generation_backend": (
                "algorithmic_sampler" if _use_algorithmic_scene_sampler(args) else "env"
            ),
            "image_size": int(args.image_size) if _should_render(args) else None,
            "compression": args.compression,
            "hdf5_path": str(shard_path),
        }
        manifest_fd, manifest_name = tempfile.mkstemp(
            prefix=f"{shard_stem}.", suffix=".tmp.json", dir=str(manifest_dir)
        )
        os.close(manifest_fd)
        temp_manifest_path = Path(manifest_name)
        with temp_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        temp_manifest_path.replace(manifest_path)
        temp_manifest_path = None

        print(f"[done] wrote {scene_count} scenes to {shard_path}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if temp_manifest_path is not None:
            temp_manifest_path.unlink(missing_ok=True)
        _release_shard_lock(lock_path, lock_fd)


if __name__ == "__main__":
    main()
