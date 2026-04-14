"""Camera data extraction utilities for visual data collection.

Ported and adapted from v4world-tmp's camera pipeline for shelf pushing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

_SHELF_STATIC_PREFIXES = ("shelf_", "leg_")


def _classify_segmentation_ids(env):
    """Classify segmentation IDs into background, static, and robot sets.

    Returns:
        (background_ids, static_ids, robot_ids)
    """
    seg_id2name = {
        k: v.name for k, v in env.unwrapped.segmentation_id_map.items()
    }
    robot_link_names = {
        link.get_name() for link in env.unwrapped.agent.robot.get_links()
    }

    background_ids = [0]
    static_ids = []
    robot_ids = []
    for seg_id, seg_name in seg_id2name.items():
        if seg_name == "ground":
            background_ids.append(seg_id)
        elif any(seg_name.startswith(p) for p in _SHELF_STATIC_PREFIXES):
            static_ids.append(seg_id)
        elif seg_name in robot_link_names:
            robot_ids.append(seg_id)

    return background_ids, static_ids, robot_ids


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def process_camera_data(
    env,
    cam_name: str,
    cam_data: Dict,
    sensor_params: Dict,
    video_streams: Dict,
    metadata_arrays: Dict,
    env_idx: int = 0,
):
    """Extract RGB, depth, masks, intrinsics, extrinsics from one camera.

    Adapted from v4world-tmp ``collect_trajectories.process_camera_data``.
    Handles GPU-batched envs by indexing into the batch with *env_idx*.

    Args:
        env: ManiSkill env (raw or wrapped).
        cam_name: Camera uid (e.g. ``"base_camera"``).
        cam_data: ``obs["sensor_data"][cam_name]`` dict.
        sensor_params: ``obs["sensor_param"]`` dict.
        video_streams: Accumulator — lists keyed ``{cam}_{suffix}``.
        metadata_arrays: Accumulator — lists keyed ``{cam}_intrinsics`` etc.
        env_idx: Which sub-env to extract in GPU-batched mode.
    """
    # --- RGB ---
    rgb = cam_data["rgb"].cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[env_idx]
    video_streams[f"{cam_name}_rgb"].append(rgb)

    # --- Depth (float32 meters → int16 millimetres for VideoRecorder) ---
    depth = cam_data["depth"].cpu().numpy()
    if depth.ndim == 4:
        depth = depth[env_idx, :, :, 0]
    elif depth.ndim == 3:
        depth = depth[:, :, 0]
    depth_mm = (depth * 1000.0).astype(np.int16)
    video_streams[f"{cam_name}_depth"].append(depth_mm)

    # --- Segmentation → masks ---
    seg = cam_data["segmentation"].cpu().numpy()
    if seg.ndim == 4:
        seg = seg[env_idx, :, :, 0]
    elif seg.ndim == 3:
        seg = seg[:, :, 0]

    background_ids, static_ids, robot_ids = _classify_segmentation_ids(env)

    robot_mask = np.isin(seg, robot_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_robot_mask"].append(robot_mask)

    foreground_mask = (~np.isin(seg, background_ids)).astype(np.uint8) * 255
    video_streams[f"{cam_name}_foreground_mask"].append(foreground_mask)

    static_mask = np.isin(seg, static_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_static_mask"].append(static_mask)

    # --- Camera parameters ---
    if cam_name in sensor_params:
        cam_params = sensor_params[cam_name]
        if "intrinsic_cv" in cam_params:
            intr = cam_params["intrinsic_cv"].cpu().numpy()
            if intr.ndim == 3:
                intr = intr[env_idx]
            metadata_arrays[f"{cam_name}_intrinsics"].append(intr)
        if "extrinsic_cv" in cam_params:
            extr = cam_params["extrinsic_cv"].cpu().numpy()
            if extr.ndim == 3:
                extr = extr[env_idx]
            metadata_arrays[f"{cam_name}_extrinsics"].append(extr)


# ---------------------------------------------------------------------------
# Stream initialisation helper
# ---------------------------------------------------------------------------

_STREAM_SUFFIXES = (
    "_rgb",
    "_depth",
    "_robot_mask",
    "_foreground_mask",
    "_static_mask",
)


def init_camera_streams(
    camera_names: List[str],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Create empty accumulator dicts for :func:`process_camera_data`.

    Returns:
        ``(video_streams, metadata_arrays)`` — each is a dict of empty lists.
    """
    video_streams: Dict[str, list] = {}
    metadata_arrays: Dict[str, list] = {}
    for cam in camera_names:
        for suffix in _STREAM_SUFFIXES:
            video_streams[f"{cam}{suffix}"] = []
        metadata_arrays[f"{cam}_intrinsics"] = []
        metadata_arrays[f"{cam}_extrinsics"] = []
    return video_streams, metadata_arrays


# ---------------------------------------------------------------------------
# Extrinsics utility  (ported from v4world-tmp utils/cam.py)
# ---------------------------------------------------------------------------


def convert_extrinsics_3x4_to_4x4(extrinsics_data: np.ndarray) -> np.ndarray:
    """Convert extrinsics matrices from (..., 3, 4) to (..., 4, 4).

    Appends ``[0, 0, 0, 1]`` as the last row.
    """
    if not isinstance(extrinsics_data, np.ndarray):
        return extrinsics_data

    orig_shape = extrinsics_data.shape
    if len(orig_shape) >= 2 and orig_shape[-2:] == (3, 4):
        flat = extrinsics_data.reshape(-1, 3, 4)
        out = np.zeros((flat.shape[0], 4, 4), dtype=extrinsics_data.dtype)
        out[:, :3, :] = flat
        out[:, 3, 3] = 1.0
        return out.reshape(orig_shape[:-2] + (4, 4))
    return extrinsics_data


# ---------------------------------------------------------------------------
# Depth → point cloud  (ported from v4world-tmp datalib/traj2rdd.py)
# ---------------------------------------------------------------------------


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    max_depth: float = 2.0,
    foreground_mask: Optional[np.ndarray] = None,
    attrs: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Convert a depth image to a 3-D point cloud.

    Args:
        depth: ``(H, W)`` depth in **metres**.
        rgb: ``(H, W, 3)`` colour image or ``None``.
        intrinsics: ``(3, 3)`` OpenCV intrinsics matrix.
        extrinsics: ``(4, 4)`` **world-to-camera** transform, or ``None``.
        max_depth: Clip points beyond this distance.
        foreground_mask: ``(H, W)`` boolean mask for foreground pixels.
        attrs: Extra per-pixel attribute arrays to sample.

    Returns:
        ``(points, colors, out_attrs)``
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    valid = (depth > 0) & (depth <= max_depth) & np.isfinite(depth)
    if foreground_mask is not None:
        if foreground_mask.dtype != bool:
            foreground_mask = foreground_mask.astype(bool)
        valid = valid & foreground_mask

    u_v, v_v, d_v = u[valid], v[valid], depth[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (u_v - cx) * d_v / fx
    y_cam = (v_v - cy) * d_v / fy
    points_cam = np.stack([x_cam, y_cam, d_v], axis=-1)

    if extrinsics is not None:
        c2w = np.linalg.inv(extrinsics)
        pts_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1))], axis=-1
        )
        points = (c2w @ pts_h.T).T[:, :3]
    else:
        points = points_cam

    colors = None
    if rgb is not None:
        colors = rgb[valid]
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0

    out_attrs = None
    if attrs is not None:
        out_attrs = [a[valid] for a in attrs]

    return points, colors, out_attrs
