"""Render planned-segment vs actual-trace diagnostics for StickPush."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from taskbench.skills.stick_push import StickPushTraceDebug


@dataclass
class TraceVizPhase:
    phase: str
    planned_start_xyz: np.ndarray  # (3,)
    planned_end_xyz: np.ndarray    # (3,)
    actual_trace_xyz: np.ndarray   # (T, 3)
    max_offtrack: float
    final_offtrack: float
    final_along_error: float
    offtrack_violation: bool


@dataclass
class TraceVizSpec:
    env_idx: int
    shelf_front_x: float
    shelf_back_x: float
    shelf_half_w: float
    phases: list[TraceVizPhase]
    blocker_disks_xy_r: np.ndarray | None = None  # (M, 3) => x, y, radius


def build_trace_viz_spec(
    *,
    env_idx: int,
    trace_debug: StickPushTraceDebug,
    shelf_front_x: float,
    shelf_back_x: float,
    shelf_half_w: float,
    blocker_disks_xy_r: np.ndarray | None = None,
) -> TraceVizSpec:
    """Build a render spec for one environment from StickPush trace debug."""

    def _phase_row(phase_obj) -> TraceVizPhase:
        m = phase_obj.metrics
        assert m is not None
        return TraceVizPhase(
            phase=str(phase_obj.phase),
            planned_start_xyz=np.asarray(phase_obj.planned_start_xyz[env_idx].detach().cpu().numpy(), dtype=np.float32),
            planned_end_xyz=np.asarray(phase_obj.planned_end_xyz[env_idx].detach().cpu().numpy(), dtype=np.float32),
            actual_trace_xyz=np.asarray(phase_obj.actual_trace_xyz[env_idx], dtype=np.float32),
            max_offtrack=float(m.max_offtrack[env_idx].item()),
            final_offtrack=float(m.final_offtrack[env_idx].item()),
            final_along_error=float(m.final_along_error[env_idx].item()),
            offtrack_violation=bool(m.offtrack_violation[env_idx].item()),
        )

    phases = [_phase_row(trace_debug.entry), _phase_row(trace_debug.sweep), _phase_row(trace_debug.retract)]
    blocker_arr = None
    if blocker_disks_xy_r is not None:
        blocker_arr = np.asarray(blocker_disks_xy_r, dtype=np.float32)
    return TraceVizSpec(
        env_idx=int(env_idx),
        shelf_front_x=float(shelf_front_x),
        shelf_back_x=float(shelf_back_x),
        shelf_half_w=float(shelf_half_w),
        phases=phases,
        blocker_disks_xy_r=blocker_arr,
    )


def render_trace_viz_image(
    spec: TraceVizSpec,
    *,
    width: int = 1280,
    height: int = 960,
    pad: int = 56,
) -> np.ndarray:
    """Render one top-down diagnostic image as RGB uint8."""

    w = int(width)
    h = int(height)
    p = int(pad)
    img = Image.new("RGB", (w, h), (248, 248, 244))
    draw = ImageDraw.Draw(img)

    front_x = float(spec.shelf_front_x)
    back_x = float(spec.shelf_back_x)
    half_w = float(spec.shelf_half_w)
    x_span = max(back_x - front_x, 1e-6)
    y_span = max(2.0 * half_w, 1e-6)
    usable_w = float(w - 2 * p)
    usable_h = float(h - 2 * p)
    scale = min(usable_w / x_span, usable_h / y_span)
    left = 0.5 * (w - x_span * scale)
    top = 0.5 * (h - y_span * scale)

    def _xy_to_px(xy: np.ndarray) -> tuple[float, float]:
        x = float(xy[0])
        y = float(xy[1])
        px = left + (x - front_x) * scale
        py = top + (half_w - y) * scale
        return float(px), float(py)

    def _rad_px(rad: float) -> float:
        return float(rad) * scale

    # Shelf boundary.
    p0 = _xy_to_px(np.array([front_x, -half_w], dtype=np.float32))
    p1 = _xy_to_px(np.array([back_x, half_w], dtype=np.float32))
    x0, x1 = sorted([p0[0], p1[0]])
    y0, y1 = sorted([p0[1], p1[1]])
    draw.rectangle([(x0, y0), (x1, y1)], outline=(26, 26, 26), width=4)

    # Optional blocker overlays.
    if spec.blocker_disks_xy_r is not None and spec.blocker_disks_xy_r.size > 0:
        for row in spec.blocker_disks_xy_r:
            cx, cy, rr = float(row[0]), float(row[1]), float(row[2])
            bx, by = _xy_to_px(np.array([cx, cy], dtype=np.float32))
            rpx = max(2.0, _rad_px(rr))
            draw.ellipse(
                [(bx - rpx, by - rpx), (bx + rpx, by + rpx)],
                fill=(120, 150, 210),
                outline=(35, 45, 70),
                width=2,
            )

    phase_style = {
        "entry": {"planned": (40, 140, 70), "actual": (55, 210, 95)},
        "sweep": {"planned": (220, 120, 30), "actual": (255, 175, 50)},
        "retract": {"planned": (135, 55, 170), "actual": (190, 95, 230)},
    }

    y_text = 16
    draw.text((16, y_text), f"env={spec.env_idx}", fill=(0, 0, 0))
    y_text += 18

    for ph in spec.phases:
        style = phase_style.get(ph.phase, {"planned": (60, 60, 60), "actual": (130, 130, 130)})

        p_start = _xy_to_px(ph.planned_start_xyz[:2])
        p_end = _xy_to_px(ph.planned_end_xyz[:2])
        draw.line([p_start, p_end], fill=style["planned"], width=5)

        if ph.actual_trace_xyz.shape[0] >= 2:
            pts = [_xy_to_px(pt[:2]) for pt in ph.actual_trace_xyz]
            draw.line(pts, fill=style["actual"], width=3)

        sr = 5
        draw.ellipse(
            [(p_start[0] - sr, p_start[1] - sr), (p_start[0] + sr, p_start[1] + sr)],
            fill=style["planned"],
            outline=(15, 15, 15),
        )
        draw.ellipse(
            [(p_end[0] - sr, p_end[1] - sr), (p_end[0] + sr, p_end[1] + sr)],
            fill=style["actual"],
            outline=(15, 15, 15),
        )

        draw.text(
            (16, y_text),
            (
                f"{ph.phase}: max_off={ph.max_offtrack:.4f} "
                f"final_off={ph.final_offtrack:.4f} "
                f"along_err={ph.final_along_error:.4f} "
                f"violation={int(ph.offtrack_violation)}"
            ),
            fill=style["planned"],
        )
        y_text += 18

    return np.asarray(img, dtype=np.uint8)


def save_trace_viz_image(path: str | Path, image: np.ndarray) -> None:
    """Save a trace-viz image to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(p)
