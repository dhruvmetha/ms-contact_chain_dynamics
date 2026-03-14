#!/usr/bin/env python3
"""Render a named suite of open-table push videos."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = REPO_ROOT / "videos"
SUCCESS_DIR = REPO_ROOT / "data" / "success"
FAILURE_DIR = REPO_ROOT / "data" / "failure"
TEMP_VIDEO = VIDEO_DIR / "0.mp4"
TEMP_SUCCESS = SUCCESS_DIR / "open_table_push_seed43.hdf5"
TEMP_FAILURE = FAILURE_DIR / "open_table_push_seed43.hdf5"


@dataclass(frozen=True)
class VideoCase:
    name: str
    solver: str
    overrides: tuple[str, ...]


CASES = (
    VideoCase(
        name="open_table_push_vertical_east",
        solver="open_table_push",
        overrides=(
            "env.extra_kwargs.push_axis=x",
            "env.extra_kwargs.push_angle_deg=0.0",
            "env.extra_kwargs.wrist_orientation=vertical",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.05",
            "env.extra_kwargs.staging_backoff=0.0",
            "env.extra_kwargs.row_origin_x=0.08",
            "env.extra_kwargs.row_origin_y=-0.06",
            "env.extra_kwargs.row_spacing=0.048",
            "+env.extra_kwargs.end_margin=0.03",
            "env.extra_kwargs.staging_height=0.14",
            "env.extra_kwargs.table_clearance=0.0",
            "run.solver_kwargs.approach_speed_scale=1.5",
            "run.solver_kwargs.push_speed_scale=2.0",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_vertical_north",
        solver="open_table_push",
        overrides=(
            "env.extra_kwargs.push_axis=y",
            "env.extra_kwargs.push_angle_deg=90.0",
            "env.extra_kwargs.wrist_orientation=vertical",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.06",
            "env.extra_kwargs.staging_backoff=0.0",
            "env.extra_kwargs.row_origin_x=0.16",
            "env.extra_kwargs.row_origin_y=-0.06",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_vertical_south",
        solver="open_table_push",
        overrides=(
            "env.extra_kwargs.push_axis=y",
            "env.extra_kwargs.push_angle_deg=270.0",
            "env.extra_kwargs.wrist_orientation=vertical",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.06",
            "env.extra_kwargs.staging_backoff=0.0",
            "env.extra_kwargs.row_origin_x=0.14",
            "env.extra_kwargs.row_origin_y=-0.03",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_vertical_west",
        solver="open_table_push",
        overrides=(
            "env.extra_kwargs.push_axis=x",
            "env.extra_kwargs.push_angle_deg=160.0",
            "env.extra_kwargs.wrist_orientation=vertical",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "+env.extra_kwargs.approach_axis_x=0.1",
            "+env.extra_kwargs.approach_axis_y=0.0",
            "+env.extra_kwargs.approach_axis_z=-0.98",
            "env.extra_kwargs.use_staging=false",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.06",
            "env.extra_kwargs.row_origin_x=0.12",
            "env.extra_kwargs.row_origin_y=-0.06",
            "env.extra_kwargs.row_spacing=0.048",
            "+env.extra_kwargs.end_margin=0.03",
            "run.solver_kwargs.push_speed_scale=1.2",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_horizontal_forward",
        solver="open_table_push_horizontal",
        overrides=(
            "env.extra_kwargs.push_axis=x",
            "env.extra_kwargs.push_angle_deg=0.0",
            "env.extra_kwargs.wrist_orientation=horizontal",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=false",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.06",
            "env.extra_kwargs.table_clearance=0.010",
            "env.extra_kwargs.push_height_offset=0.010",
            "+env.extra_kwargs.row_origin_x=0.06",
            "+env.extra_kwargs.row_origin_y=-0.16",
            "env.extra_kwargs.row_spacing=0.048",
            "env.extra_kwargs.end_margin=0.03",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_horizontal_left",
        solver="open_table_push_horizontal",
        overrides=(
            "env.extra_kwargs.push_axis=y",
            "env.extra_kwargs.push_angle_deg=90.0",
            "env.extra_kwargs.wrist_orientation=horizontal",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.04",
            "env.extra_kwargs.staging_backoff=0.02",
            "+env.extra_kwargs.row_origin_x=0.08",
            "+env.extra_kwargs.row_origin_y=-0.03",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_horizontal_right",
        solver="open_table_push_horizontal",
        overrides=(
            "env.extra_kwargs.push_axis=y",
            "env.extra_kwargs.push_angle_deg=270.0",
            "env.extra_kwargs.wrist_orientation=horizontal",
            "env.extra_kwargs.tool_spin_deg=0.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.04",
            "env.extra_kwargs.staging_backoff=0.0",
            "+env.extra_kwargs.row_origin_x=0.06",
            "+env.extra_kwargs.row_origin_y=-0.12",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_vertical_forward_spin90",
        solver="open_table_push",
        overrides=(
            "env.extra_kwargs.push_axis=x",
            "env.extra_kwargs.push_angle_deg=0.0",
            "env.extra_kwargs.wrist_orientation=vertical",
            "env.extra_kwargs.tool_spin_deg=90.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.12",
            "env.extra_kwargs.staging_backoff=0.04",
            "env.extra_kwargs.row_origin_x=0.10",
            "env.extra_kwargs.row_origin_y=-0.03",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
    VideoCase(
        name="open_table_push_horizontal_forward_spin90",
        solver="open_table_push_horizontal",
        overrides=(
            "env.extra_kwargs.push_axis=x",
            "env.extra_kwargs.push_angle_deg=0.0",
            "env.extra_kwargs.wrist_orientation=horizontal",
            "env.extra_kwargs.tool_spin_deg=90.0",
            "env.extra_kwargs.use_staging=true",
            "env.extra_kwargs.num_cylinders=2",
            "env.extra_kwargs.approach_gap=0.06",
            "env.extra_kwargs.staging_backoff=0.02",
            "env.extra_kwargs.table_clearance=0.010",
            "env.extra_kwargs.push_height_offset=0.010",
            "env.extra_kwargs.robot_init_qpos_noise=0.0",
            "+env.extra_kwargs.row_origin_x=0.08",
            "+env.extra_kwargs.row_origin_y=-0.12",
            "+env.extra_kwargs.staging_approach_axis_x=0.8",
            "+env.extra_kwargs.staging_approach_axis_y=0.0",
            "+env.extra_kwargs.staging_approach_axis_z=-0.6",
            "run.solver_kwargs.settle_steps=8",
        ),
    ),
)


def _clean_temp_artifacts() -> None:
    for path in (TEMP_VIDEO, TEMP_SUCCESS, TEMP_FAILURE):
        path.unlink(missing_ok=True)


def _copy_trace(case_name: str) -> None:
    if TEMP_FAILURE.exists():
        raise RuntimeError(f"{case_name} failed; see {TEMP_FAILURE}")
    if not TEMP_SUCCESS.exists():
        raise RuntimeError(f"{case_name} did not produce {TEMP_SUCCESS}")
    target = SUCCESS_DIR / f"{case_name}_seed43.hdf5"
    shutil.copy2(TEMP_SUCCESS, target)
    TEMP_SUCCESS.unlink(missing_ok=True)


def _move_video(case_name: str) -> None:
    if not TEMP_VIDEO.exists():
        raise RuntimeError(f"{case_name} did not produce {TEMP_VIDEO}")
    target = VIDEO_DIR / f"{case_name}.mp4"
    target.unlink(missing_ok=True)
    shutil.move(TEMP_VIDEO, target)


def run_case(case: VideoCase) -> None:
    _clean_temp_artifacts()
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "taskbench.run",
        f"solver={case.solver}",
        "run.num_episodes=1",
        "seed=42",
        "env.record_video=true",
        *case.overrides,
    ]
    print(f"[render] {case.name}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    _move_video(case.name)
    _copy_trace(case.name)


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SUCCESS_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        run_case(case)
    print(f"Rendered {len(CASES)} videos into {VIDEO_DIR}")


if __name__ == "__main__":
    main()
