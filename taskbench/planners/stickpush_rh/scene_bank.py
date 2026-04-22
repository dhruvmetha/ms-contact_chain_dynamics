"""Scene-bank loader utilities for stickpush RH heuristic testing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


DEFAULT_SCENE_BANK_PATH = (
    Path(__file__).resolve().parent / "test_data" / "scene_bank_shelf_env_v1_seed0_active13_n100.json"
)


def load_scene_bank_payload(path: str | Path = DEFAULT_SCENE_BANK_PATH) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def payload_scene_to_state(scene_row: dict) -> SceneState:
    objects: list[ObjectState] = []
    target: ObjectState | None = None
    for row in scene_row.get("objects", []):
        obj = ObjectState(
            name=str(row["name"]),
            center_xyz=np.asarray(row["center_xyz"], dtype=np.float32),
            radius=float(row["radius"]),
            is_target=bool(row["is_target"]),
            active=bool(row["active"]),
        )
        objects.append(obj)
        if obj.active and obj.is_target:
            target = obj

    if target is None:
        raise ValueError("Scene row does not contain an active target object.")

    blockers = [o for o in objects if o.active and (not o.is_target)]
    shelf = scene_row["shelf"]
    return SceneState(
        target=target,
        blockers=blockers,
        all_objects=objects,
        shelf_front_x=float(shelf["front_x"]),
        shelf_back_x=float(shelf["back_x"]),
        shelf_half_w=float(shelf["half_w"]),
        surface_z=float(shelf["surface_z"]),
        open_dir_xy=np.asarray(scene_row["open_dir_xy"], dtype=np.float32),
    )


def load_scene_bank_states(path: str | Path = DEFAULT_SCENE_BANK_PATH) -> list[SceneState]:
    payload = load_scene_bank_payload(path)
    return [payload_scene_to_state(row) for row in payload.get("scenes", [])]

