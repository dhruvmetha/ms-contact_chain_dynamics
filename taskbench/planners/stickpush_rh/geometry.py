"""Geometry helpers and metric computation for stick-push RH search."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from taskbench.planners.stickpush_rh.types import NodeMetrics, ObjectState, SceneState

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


def grasp_semicircle_center_xy(target: ObjectState, open_dir_xy: np.ndarray) -> np.ndarray:
    """Center of grasp-clearance semicircle at the target back-line midpoint."""

    open_dir = normalize_xy(np.asarray(open_dir_xy, dtype=np.float32))
    return target.center_xyz[:2].astype(np.float32) - open_dir * float(target.radius)


def point_in_front_semicircle(
    point_xy: np.ndarray,
    target: ObjectState,
    clearance_radius: float,
    open_dir_xy: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    """Whether a 2D point lies inside/on the target front semicircle.

    The front semicircle is defined by:
    - center at target back-line midpoint (`grasp_semicircle_center_xy`)
    - radius = `clearance_radius`
    - front half toward `open_dir_xy`
    """

    p = np.asarray(point_xy, dtype=np.float32).reshape(2)
    center_xy = grasp_semicircle_center_xy(target, open_dir_xy)
    d = p - center_xy
    dist = float(np.linalg.norm(d))
    in_disk = dist <= max(0.0, float(clearance_radius) - float(margin))
    in_front = float(np.dot(d, normalize_xy(open_dir_xy))) >= 0.0
    return bool(in_disk and in_front)


def front_semicircle_surface_intersects(
    blocker: ObjectState,
    target: ObjectState,
    clearance_radius: float,
    open_dir_xy: np.ndarray,
) -> bool:
    """Blocker intersects front semicircle around the target (surface-aware)."""

    if not blocker.active:
        return False

    center_xy = grasp_semicircle_center_xy(target, open_dir_xy)
    d = blocker.center_xyz[:2] - center_xy
    dist = float(np.linalg.norm(d))
    intersects_disk = dist <= float(clearance_radius) + float(blocker.radius)
    front_half_intersects = float(np.dot(d, open_dir_xy)) >= -float(blocker.radius)
    return bool(intersects_disk and front_half_intersects)


def collect_blockers_in_clearance(scene: SceneState, clearance_radius: float) -> list[ObjectState]:
    return [
        b
        for b in scene.blockers
        if front_semicircle_surface_intersects(
            blocker=b,
            target=scene.target,
            clearance_radius=clearance_radius,
            open_dir_xy=scene.open_dir_xy,
        )
    ]


def compute_deficit(scene: SceneState, blockers: list[ObjectState], clearance_radius: float) -> float:
    center_xy = grasp_semicircle_center_xy(scene.target, scene.open_dir_xy)
    total = 0.0
    for b in blockers:
        d = float(np.linalg.norm(b.center_xyz[:2] - center_xy))
        total += max(0.0, float(clearance_radius) + float(b.radius) - d)
    return float(total)


def compute_node_metrics(scene: SceneState, clearance_radius: float, pushes_used: int) -> NodeMetrics:
    blockers = collect_blockers_in_clearance(scene, clearance_radius)

    if blockers:
        min_margin = min(surface_margin(scene.target, b) for b in blockers)
    else:
        min_margin = float("inf")

    deficit = compute_deficit(scene, blockers, clearance_radius)
    return NodeMetrics(
        blockers=len(blockers),
        min_margin=float(min_margin),
        deficit=float(deficit),
        pushes_used=int(pushes_used),
        solved=len(blockers) == 0,
    )


def segment_intersects_front_semicircle(
    seg_a_xy: np.ndarray,
    seg_b_xy: np.ndarray,
    target: ObjectState,
    clearance_radius: float,
    open_dir_xy: np.ndarray,
    *,
    eps: float = 1e-9,
) -> bool:
    """Whether a line segment intersects the target front semicircle region."""

    a = np.asarray(seg_a_xy, dtype=np.float64).reshape(2)
    b = np.asarray(seg_b_xy, dtype=np.float64).reshape(2)
    center = np.asarray(grasp_semicircle_center_xy(target, open_dir_xy), dtype=np.float64)
    u = np.asarray(normalize_xy(open_dir_xy), dtype=np.float64)
    r = float(clearance_radius)

    # p(t) = a + t*(b-a), t in [0,1]
    d = b - a
    q = a - center
    A = float(np.dot(d, d))

    # Degenerate segment: check endpoint directly.
    if A <= eps:
        in_disk = float(np.linalg.norm(q)) <= r + eps
        in_front = float(np.dot(q, u)) >= -eps
        return bool(in_disk and in_front)

    # Disk constraint: ||q + t d||^2 <= r^2
    B = float(2.0 * np.dot(d, q))
    C = float(np.dot(q, q) - r * r)
    disc = float(B * B - 4.0 * A * C)
    if disc < -eps:
        return False
    if disc < 0.0:
        disc = 0.0
    s = math.sqrt(disc)
    t0 = float((-B - s) / (2.0 * A))
    t1 = float((-B + s) / (2.0 * A))
    disk_lo = max(0.0, min(t0, t1))
    disk_hi = min(1.0, max(t0, t1))
    if disk_hi < disk_lo - eps:
        return False

    # Front-half constraint: dot(q + t d, u) >= 0
    L = float(np.dot(d, u))
    M = float(np.dot(q, u))
    front_lo = 0.0
    front_hi = 1.0
    if abs(L) <= eps:
        if M < -eps:
            return False
    elif L > 0.0:
        front_lo = max(front_lo, float(-M / L))
    else:
        front_hi = min(front_hi, float(-M / L))
    if front_hi < front_lo - eps:
        return False

    lo = max(disk_lo, front_lo)
    hi = min(disk_hi, front_hi)
    return bool(hi >= lo - eps)


def front_semicircle_wall_intersections(scene: SceneState, clearance_radius: float) -> list[str]:
    """Return side/back shelf walls intersecting the target front semicircle."""

    fx = float(scene.shelf_front_x)
    bx = float(scene.shelf_back_x)
    hw = float(scene.shelf_half_w)
    segments = {
        "side_pos": (np.array([fx, hw], dtype=np.float64), np.array([bx, hw], dtype=np.float64)),
        "side_neg": (np.array([fx, -hw], dtype=np.float64), np.array([bx, -hw], dtype=np.float64)),
        "back": (np.array([bx, -hw], dtype=np.float64), np.array([bx, hw], dtype=np.float64)),
    }

    hits: list[str] = []
    for name, (a, b) in segments.items():
        if segment_intersects_front_semicircle(
            a,
            b,
            scene.target,
            clearance_radius,
            scene.open_dir_xy,
        ):
            hits.append(name)
    return hits


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
