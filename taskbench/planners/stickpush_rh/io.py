"""Artifact I/O helpers for stick-push RH planner."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from taskbench.planners.stickpush_rh.ranking import metrics_to_dict
from taskbench.planners.stickpush_rh.types import NodeMetrics, PlannerAction


def _to_jsonable(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def build_artifact_dir(root: str | Path, seed: int | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"_seed{seed}" if seed is not None else ""
    out_dir = Path(root) / f"{ts}{seed_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    iio.imwrite(path, arr)


def save_video(path: str | Path, frames: list[np.ndarray], fps: int = 20) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    stack = [np.asarray(f, dtype=np.uint8) for f in frames]
    iio.imwrite(path, stack, fps=fps)


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, sort_keys=True)


def write_frontier_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(_to_jsonable(row))


def serialize_action(action: PlannerAction) -> dict:
    return {
        "blocker_name": action.blocker_name,
        "theta_deg": float(action.theta_deg),
        "x_approach": float(action.x_approach),
        "delta_len_idx": int(action.delta_len_idx),
        "contact_offset": float(action.contact_offset),
        "push_len": float(action.push_len),
        "approach_xyz": action.approach_xyz.tolist(),
        "entry_xyz": action.entry_xyz.tolist(),
        "sweep_xyz": action.sweep_xyz.tolist(),
        "retract_xyz": action.retract_xyz.tolist(),
        "meta": _to_jsonable(action.meta),
    }


def serialize_metrics(metrics: NodeMetrics) -> dict:
    return metrics_to_dict(metrics)

