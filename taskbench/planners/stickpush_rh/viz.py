"""Visualization utilities for stick-push RH planner artifacts."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from taskbench.planners.stickpush_rh.geometry import normalize_xy
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
    left, top, scale = transform
    x, y = float(xy[0]), float(xy[1])
    px = left + (scene.shelf_half_w - y) * scale
    py = top + (scene.shelf_back_x - x) * scale
    return px, py


def _canvas_transform(scene: SceneState, width: int, height: int, pad: int) -> tuple[float, float, float]:
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


def _rotated_rect_corners(
    center_xy: np.ndarray,
    u_xy: np.ndarray,
    v_xy: np.ndarray,
    half_len_u: float,
    half_len_v: float,
) -> list[np.ndarray]:
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    u = normalize_xy(np.asarray(u_xy, dtype=np.float64).reshape(2)).astype(np.float64)
    v = normalize_xy(np.asarray(v_xy, dtype=np.float64).reshape(2)).astype(np.float64)
    return [
        c + half_len_u * u + half_len_v * v,
        c + half_len_u * u - half_len_v * v,
        c - half_len_u * u - half_len_v * v,
        c - half_len_u * u + half_len_v * v,
    ]


def _draw_oriented_rect(
    draw: ImageDraw.ImageDraw,
    scene: SceneState,
    transform: tuple[float, float, float],
    *,
    center_xy: np.ndarray,
    u_xy: np.ndarray,
    v_xy: np.ndarray,
    half_len_u: float,
    half_len_v: float,
    fill,
    outline,
):
    corners = _rotated_rect_corners(center_xy, u_xy, v_xy, half_len_u, half_len_v)
    pts = [_world_to_canvas(np.asarray(p, dtype=np.float32), scene, transform) for p in corners]
    draw.polygon(pts, fill=fill, outline=outline)


def _draw_object(
    draw: ImageDraw.ImageDraw,
    scene: SceneState,
    transform: tuple[float, float, float],
    obj,
):
    shape = str(getattr(obj, "footprint_type", "circle") or "circle")
    params = dict(getattr(obj, "footprint_params", {}) or {})
    if shape == "obox":
        he = np.asarray(params.get("half_extents", [obj.radius, obj.radius]), dtype=np.float32).reshape(-1)
        if he.size < 2:
            he = np.array([obj.radius, obj.radius], dtype=np.float32)
        yaw = float(params.get("yaw", 0.0))
        u = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        v = np.array([-u[1], u[0]], dtype=np.float32)
        fill = (80, 120, 220, 220) if obj.active else (180, 180, 180, 120)
        _draw_oriented_rect(
            draw,
            scene,
            transform,
            center_xy=obj.center_xyz[:2],
            u_xy=u,
            v_xy=v,
            half_len_u=float(he[0]),
            half_len_v=float(he[1]),
            fill=fill,
            outline=(30, 30, 40, 180),
        )
        return

    bx, by = _world_to_canvas(obj.center_xyz[:2], scene, transform)
    br = _radius_px(obj.radius, transform)
    color = (80, 120, 220, 220) if obj.active else (180, 180, 180, 120)
    draw.ellipse([bx - br, by - br, bx + br, by + br], fill=color, outline=(30, 30, 40, 180))


def _draw_stencil_overlay(
    draw: ImageDraw.ImageDraw,
    scene: SceneState,
    transform: tuple[float, float, float],
    *,
    goal_overlay: dict,
):
    goal_labels = dict(goal_overlay.get("goal_labels", {}) or {})
    tmpl = goal_labels.get("display_template")
    if not isinstance(tmpl, dict):
        return
    stencil = dict(goal_overlay.get("stencil_cfg", {}) or {})

    front_offset = float(stencil.get("front_outside_offset", 0.06))
    finger_t = float(stencil.get("finger_thickness", 0.012))
    finger_l = float(stencil.get("finger_length", 0.05))
    jaw_open = float(stencil.get("jaw_open", 0.06))
    jaw_contact = float(stencil.get("jaw_contact", 0.036))
    pad_w = float(stencil.get("contact_pad_width", 0.01))
    pad_d = float(stencil.get("contact_pad_depth", 0.01))

    u = normalize_xy(np.asarray(tmpl.get("insertion_dir_xy", scene.open_dir_xy), dtype=np.float32))
    v = normalize_xy(np.array([-float(u[1]), float(u[0])], dtype=np.float32))
    t_xy = np.asarray(scene.target.center_xyz[:2], dtype=np.float32)

    entry_xy = tmpl.get("entry_xy")
    if entry_xy is None:
        return
    entry_xy = np.asarray(entry_xy, dtype=np.float32)
    outside = entry_xy + float(front_offset) * u
    path_len = float(np.linalg.norm(t_xy - outside))
    path_mid = 0.5 * (outside + t_xy)

    open_off = 0.5 * jaw_open + 0.5 * finger_t
    close_off = 0.5 * jaw_contact + 0.5 * finger_t
    half_t = 0.5 * finger_t
    half_l = 0.5 * finger_l

    # Insertion sweep corridors (open fingers).
    corridor_half_l = 0.5 * path_len + half_l
    for sgn in (-1.0, 1.0):
        _draw_oriented_rect(
            draw,
            scene,
            transform,
            center_xy=path_mid + sgn * open_off * v,
            u_xy=u,
            v_xy=v,
            half_len_u=corridor_half_l,
            half_len_v=half_t,
            fill=(30, 180, 70, 60),
            outline=(20, 120, 50, 180),
        )

    # Final finger bodies.
    for sgn in (-1.0, 1.0):
        _draw_oriented_rect(
            draw,
            scene,
            transform,
            center_xy=t_xy + sgn * close_off * v,
            u_xy=u,
            v_xy=v,
            half_len_u=half_l,
            half_len_v=half_t,
            fill=(20, 110, 220, 90),
            outline=(20, 80, 170, 220),
        )

    # Closure sweep bands.
    closure_half_v = abs(0.5 * jaw_open - 0.5 * jaw_contact) + half_t
    for sgn in (-1.0, 1.0):
        _draw_oriented_rect(
            draw,
            scene,
            transform,
            center_xy=t_xy + sgn * (0.25 * (jaw_open + jaw_contact) + half_t) * v,
            u_xy=u,
            v_xy=v,
            half_len_u=half_l,
            half_len_v=closure_half_v,
            fill=(240, 130, 30, 45),
            outline=(220, 120, 20, 200),
        )

    # Contact pads.
    for sgn in (-1.0, 1.0):
        _draw_oriented_rect(
            draw,
            scene,
            transform,
            center_xy=t_xy + sgn * 0.5 * jaw_contact * v,
            u_xy=u,
            v_xy=v,
            half_len_u=0.5 * pad_d,
            half_len_v=0.5 * pad_w,
            fill=(250, 220, 40, 170),
            outline=(180, 150, 30, 230),
        )

    # Entry ray.
    p_entry = _world_to_canvas(entry_xy, scene, transform)
    p_t = _world_to_canvas(t_xy, scene, transform)
    p_out = _world_to_canvas(outside, scene, transform)
    draw.line([p_out, p_entry, p_t], fill=(210, 50, 50, 230), width=2)

    reason = str(tmpl.get("fail_reason", "pass" if bool(tmpl.get("passed", False)) else "fail"))
    blocks = tmpl.get("blocking_object_names", [])
    block_txt = "" if not blocks else f" blockers={','.join(str(x) for x in blocks[:3])}"
    status_txt = f"[{goal_labels.get('active_label', 'any')}] {'PASS' if bool(tmpl.get('passed', False)) else 'FAIL'} reason={reason}{block_txt}"
    draw.rectangle([(8, 8), (min(860, 20 + 8 * len(status_txt)), 34)], fill=(255, 255, 255, 200), outline=(40, 40, 40, 180))
    draw.text((12, 14), status_txt, fill=(20, 20, 20, 255))


def render_candidate_map(
    scene: SceneState,
    candidates: list[PlannerAction],
    *,
    selected_action: PlannerAction | None = None,
    goal_overlay: dict | None = None,
    max_candidates_drawn: int = 300,
    width: int = 900,
    height: int = 900,
) -> np.ndarray:
    bg = Image.new("RGB", (width, height), color=(250, 250, 245))
    draw = ImageDraw.Draw(bg, "RGBA")
    pad = 60
    transform = _canvas_transform(scene, width, height, pad)

    # Shelf bounds.
    p_a = _world_to_canvas(np.array([scene.shelf_front_x, -scene.shelf_half_w], dtype=np.float32), scene, transform)
    p_b = _world_to_canvas(np.array([scene.shelf_back_x, scene.shelf_half_w], dtype=np.float32), scene, transform)
    x0, x1 = sorted([p_a[0], p_b[0]])
    y0, y1 = sorted([p_a[1], p_b[1]])
    draw.rectangle([(x0, y0), (x1, y1)], outline=(20, 20, 20, 255), width=3)

    # Target + blockers.
    _draw_object(draw, scene, transform, scene.target)
    for b in scene.blockers:
        _draw_object(draw, scene, transform, b)

    # Candidate arrows.
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

    if selected_action is not None:
        p1 = _world_to_canvas(selected_action.entry_xyz[:2], scene, transform)
        p2 = _world_to_canvas(selected_action.sweep_xyz[:2], scene, transform)
        _draw_arrow(draw, p1, p2, color=(230, 30, 30, 255), width=4)
        pa = _world_to_canvas(selected_action.approach_xyz[:2], scene, transform)
        pr = _world_to_canvas(selected_action.retract_xyz[:2], scene, transform)
        draw.line([pa, p1], fill=(0, 170, 0, 220), width=3)
        draw.line([p2, pr], fill=(140, 40, 180, 220), width=3)

    if isinstance(goal_overlay, dict):
        _draw_stencil_overlay(draw, scene, transform, goal_overlay=goal_overlay)

    return np.asarray(bg, dtype=np.uint8)
