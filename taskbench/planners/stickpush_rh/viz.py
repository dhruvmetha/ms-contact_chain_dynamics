"""Visualization utilities for stick-push RH planner artifacts."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from taskbench.planners.stickpush_rh.geometry import grasp_semicircle_center_xy
from taskbench.planners.stickpush_rh.types import PlannerAction, SceneState


def capture_rgb_frame(env) -> np.ndarray:
    try:
        frame = env.render()
    except RuntimeError as exc:
        if "render_mode is not set" in str(exc):
            return np.zeros((512, 512, 3), dtype=np.uint8)
        raise
    if frame is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _world_to_canvas(
    xy: np.ndarray, scene: SceneState, transform: tuple[float, float, float]
) -> tuple[float, float]:
    # Camera-aligned top-down map:
    # - Horizontal axis follows shelf Y (mirrored so +Y is left, matching render).
    # - Vertical axis follows shelf X with front (open face) at bottom.
    left, top, scale = transform
    x, y = float(xy[0]), float(xy[1])
    px = left + (scene.shelf_half_w - y) * scale
    py = top + (scene.shelf_back_x - x) * scale
    return px, py


def _canvas_transform(scene: SceneState, width: int, height: int, pad: int) -> tuple[float, float, float]:
    # Horizontal span uses shelf width (Y), vertical span uses shelf depth (X).
    y_span = max(2.0 * scene.shelf_half_w, 1e-6)
    x_span = max(scene.shelf_back_x - scene.shelf_front_x, 1e-6)
    scale = min((width - 2 * pad) / y_span, (height - 2 * pad) / x_span)
    usable_w = y_span * scale
    usable_h = x_span * scale
    left = 0.5 * (width - usable_w)
    top = 0.5 * (height - usable_h)
    return float(left), float(top), float(scale)


def _radius_px(radius_world: float, transform: tuple[float, float, float]) -> float:
    return float(radius_world) * float(transform[2])


def _draw_arrow(draw: ImageDraw.ImageDraw, p1, p2, color, width: int = 2):
    draw.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    head_len = 10
    a1 = (p2[0] - head_len * math.cos(ang - 0.35), p2[1] - head_len * math.sin(ang - 0.35))
    a2 = (p2[0] - head_len * math.cos(ang + 0.35), p2[1] - head_len * math.sin(ang + 0.35))
    draw.polygon([p2, a1, a2], fill=color)


def _extract_target_push_debug(
    candidates: list[PlannerAction], selected_action: PlannerAction | None
) -> dict | None:
    pool: list[PlannerAction] = list(candidates)
    if selected_action is not None:
        pool.append(selected_action)
    for a in pool:
        meta = a.meta or {}
        sampled = meta.get("target_push_sampled_dir_degs")
        if sampled is None:
            continue
        topk = meta.get("target_push_topk_dir_degs") or []
        nearest = meta.get("target_push_nearest_bearing_deg")
        return {
            "sampled": [float(x) for x in sampled],
            "topk": [float(x) for x in topk],
            "nearest": (None if nearest is None else float(nearest)),
        }
    return None


def _draw_target_direction_overlay(
    draw: ImageDraw.ImageDraw,
    scene: SceneState,
    transform: tuple[float, float, float],
    *,
    clearance_radius: float,
    debug: dict | None,
):
    if debug is None:
        return
    t_xy = scene.target.center_xyz[:2].astype(np.float32)
    p0 = _world_to_canvas(t_xy, scene, transform)
    ray_len = float(clearance_radius) + float(scene.target.radius) + 0.08

    def _p_from_deg(deg: float):
        r = math.radians(float(deg))
        u = np.array([math.cos(r), math.sin(r)], dtype=np.float32)
        return _world_to_canvas(t_xy + u * ray_len, scene, transform)

    for deg in debug.get("sampled", []):
        p = _p_from_deg(float(deg))
        draw.line([p0, p], fill=(120, 140, 160, 140), width=1)
    for deg in debug.get("topk", []):
        p = _p_from_deg(float(deg))
        draw.line([p0, p], fill=(20, 150, 210, 220), width=3)
    nearest = debug.get("nearest")
    if nearest is not None:
        p = _p_from_deg(float(nearest))
        draw.line([p0, p], fill=(245, 140, 30, 230), width=4)


def render_candidate_map(
    scene: SceneState,
    candidates: list[PlannerAction],
    *,
    selected_action: PlannerAction | None = None,
    clearance_radius: float,
    max_candidates_drawn: int = 300,
    width: int = 900,
    height: int = 900,
) -> np.ndarray:
    bg = Image.new("RGB", (width, height), color=(250, 250, 245))
    draw = ImageDraw.Draw(bg, "RGBA")
    pad = 60
    transform = _canvas_transform(scene, width, height, pad)

    # Shelf bounds
    p_a = _world_to_canvas(
        np.array([scene.shelf_front_x, -scene.shelf_half_w], dtype=np.float32),
        scene,
        transform,
    )
    p_b = _world_to_canvas(
        np.array([scene.shelf_back_x, scene.shelf_half_w], dtype=np.float32),
        scene,
        transform,
    )
    x0, x1 = sorted([p_a[0], p_b[0]])
    y0, y1 = sorted([p_a[1], p_b[1]])
    draw.rectangle([(x0, y0), (x1, y1)], outline=(20, 20, 20, 255), width=3)

    # Target + clearance semicircle (centered at grasp back-line midpoint).
    t_xy = scene.target.center_xyz[:2]
    tx, ty = _world_to_canvas(t_xy, scene, transform)
    gc_xy = grasp_semicircle_center_xy(scene.target, scene.open_dir_xy)
    gx, gy = _world_to_canvas(gc_xy, scene, transform)
    t_r = _radius_px(scene.target.radius, transform)
    clr_r = _radius_px(clearance_radius, transform)

    draw.ellipse([gx - clr_r, gy - clr_r, gx + clr_r, gy + clr_r], outline=(220, 120, 20, 160), width=2)
    # Back-line (diameter line) for the semicircle boundary.
    draw.line([(gx - clr_r, gy), (gx + clr_r, gy)], fill=(220, 120, 20, 120), width=2)
    draw.ellipse([tx - t_r, ty - t_r, tx + t_r, ty + t_r], fill=(220, 40, 40, 255))

    # Blockers
    for b in scene.blockers:
        bx, by = _world_to_canvas(b.center_xyz[:2], scene, transform)
        br = _radius_px(b.radius, transform)
        color = (80, 120, 220, 220) if b.active else (180, 180, 180, 120)
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=color, outline=(30, 30, 40, 180))

    # Candidate arrows
    if candidates:
        if len(candidates) > max_candidates_drawn:
            idxs = np.linspace(0, len(candidates) - 1, max_candidates_drawn, dtype=int).tolist()
            draw_actions = [candidates[i] for i in idxs]
        else:
            draw_actions = candidates

        for a in draw_actions:
            p1 = _world_to_canvas(a.entry_xyz[:2], scene, transform)
            p2 = _world_to_canvas(a.sweep_xyz[:2], scene, transform)
            _draw_arrow(draw, p1, p2, color=(50, 80, 130, 110), width=2)

    # Debug overlay for target-push direction sampling (if metadata available).
    _draw_target_direction_overlay(
        draw,
        scene,
        transform,
        clearance_radius=clearance_radius,
        debug=_extract_target_push_debug(candidates, selected_action),
    )

    if selected_action is not None:
        p1 = _world_to_canvas(selected_action.entry_xyz[:2], scene, transform)
        p2 = _world_to_canvas(selected_action.sweep_xyz[:2], scene, transform)
        _draw_arrow(draw, p1, p2, color=(230, 30, 30, 255), width=4)
        pa = _world_to_canvas(selected_action.approach_xyz[:2], scene, transform)
        pr = _world_to_canvas(selected_action.retract_xyz[:2], scene, transform)
        draw.line([pa, p1], fill=(0, 170, 0, 220), width=3)
        draw.line([p2, pr], fill=(140, 40, 180, 220), width=3)

    return np.asarray(bg, dtype=np.uint8)
