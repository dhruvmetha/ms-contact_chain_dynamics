"""Geometry helpers for stick-push RH search."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from taskbench.planners.stickpush_rh.types import ObjectState, SceneState

_EPS = 1e-8


def normalize_xy(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(vv))
    if n < _EPS:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vv / n


def rotate_xy(v: np.ndarray, theta_deg: float) -> np.ndarray:
    t = math.radians(float(theta_deg))
    c, s = math.cos(t), math.sin(t)
    x, y = np.asarray(v, dtype=np.float32).reshape(2)
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def within_shelf_xy(p_xy: np.ndarray, scene: SceneState, margin: float = 0.0) -> bool:
    x, y = float(p_xy[0]), float(p_xy[1])
    return (
        scene.shelf_front_x + margin <= x <= scene.shelf_back_x - margin
        and -scene.shelf_half_w + margin <= y <= scene.shelf_half_w - margin
    )


def ray_length_to_shelf_edge(
    p_xy: np.ndarray, d_xy: np.ndarray, scene: SceneState, margin: float = 0.0
) -> float | None:
    """Return positive ray length from p_xy to first shelf boundary hit."""

    p = np.asarray(p_xy, dtype=np.float32).reshape(2)
    d = normalize_xy(d_xy)
    candidates: list[float] = []

    if abs(float(d[0])) > _EPS:
        x_hit = scene.shelf_back_x - margin if d[0] > 0 else scene.shelf_front_x + margin
        tx = (x_hit - float(p[0])) / float(d[0])
        if tx > 0:
            y_at_tx = float(p[1]) + tx * float(d[1])
            if -scene.shelf_half_w + margin <= y_at_tx <= scene.shelf_half_w - margin:
                candidates.append(tx)

    if abs(float(d[1])) > _EPS:
        y_hit = scene.shelf_half_w - margin if d[1] > 0 else -scene.shelf_half_w + margin
        ty = (y_hit - float(p[1])) / float(d[1])
        if ty > 0:
            x_at_ty = float(p[0]) + ty * float(d[0])
            if scene.shelf_front_x + margin <= x_at_ty <= scene.shelf_back_x - margin:
                candidates.append(ty)

    if not candidates:
        return None
    return float(min(candidates))


def center_distance_xy(a: ObjectState, b: ObjectState) -> float:
    return float(np.linalg.norm(a.center_xyz[:2] - b.center_xyz[:2]))


def surface_margin(a: ObjectState, b: ObjectState) -> float:
    return center_distance_xy(a, b) - (float(a.radius) + float(b.radius))


def point_segment_distance_xy(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Distance from 2D point to 2D segment."""

    p = np.asarray(point, dtype=np.float32).reshape(2)
    a = np.asarray(seg_a, dtype=np.float32).reshape(2)
    b = np.asarray(seg_b, dtype=np.float32).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < _EPS:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def hash_scene_state(scene: SceneState, quantization: float = 0.005) -> str:
    """Hash active object XY positions for loop detection/diagnostics."""

    rows = []
    for obj in sorted(scene.all_objects, key=lambda o: o.name):
        if not obj.active:
            continue
        qx = int(round(float(obj.center_xyz[0]) / quantization))
        qy = int(round(float(obj.center_xyz[1]) / quantization))
        rows.append((obj.name, qx, qy))
    payload = repr(rows).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
