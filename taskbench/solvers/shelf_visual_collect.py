"""Visual data collection solver for shelf pushing.

Runs GPU-batched stick pushes (same as shelf_panda_stick_push_batched)
while collecting visual data. Supports two collection modes:

- "zarr": Saves (s, a, s') pairs to a flat zarr dataset. Each push appends
  N rows (one per env). Best for large-scale first+last frame collection.
- "full": Saves full-trajectory videos + metadata per push per env to
  ManiSkillTrajectoryDataset (MP4 + HDF5). Best for video prediction.

Designed for massively parallel collection:
- cuRobo planner + StickPush skill cached across episodes (no re-init)
- Resume support: detects existing data and continues from there
- Worker ID support: multiple SLURM jobs write to separate files/subdirs

Usage:
    # Zarr mode (recommended for world model training)
    uv run python -m taskbench.run task=shelf_panda_stick \
        solver=shelf_visual_collect run.solver_kwargs.collect_mode=zarr

    # Full trajectory mode
    uv run python -m taskbench.run task=shelf_panda_stick \
        solver=shelf_visual_collect run.solver_kwargs.collect_mode=full

    # Multi-GPU parallel collection
    bash scripts/launch_visual_collect.sh 4 50
"""

import logging
import os
from datetime import datetime

import numpy as np
import torch

from taskbench.camera_utils import init_camera_streams, process_camera_data
from taskbench.skills.curobo_motion import get_arm_joint_names, setup_curobo_planner
from taskbench.skills.stick_push import StickPush
from taskbench.solver import BaseSolver, SolverResult, register_solver
from taskbench.solvers.shelf_panda_stick_push_batched import (
    Q_INTO_SHELF,
    REST_QPOS,
    ROBOT_UID,
    SAFE_QPOS,
    _sample_pushes,
)
from taskbench.trajectory_dataset import ManiSkillTrajectoryDataset, TrajectoryData

import taskbench.agents.panda_stick_long  # noqa: F401

logger = logging.getLogger("taskbench.solvers.shelf_visual_collect")

DEFAULT_CAMERA_NAMES = ["top_camera"]


def _sample_pushes_mixed(rng, N, shelf, cylinder_positions, radius=0.05,
                         cyl_radius=0.018, stick_radius=0.005,
                         insert_buffer=0.005):
    """Sample pushes with three strategies (1/3 each):

    1. p1 within `radius` of a random cylinder, p2 random
    2. p1 random, p2 within `radius` of a random cylinder
    3. Fully random (no bias)

    p1 is guaranteed to lie on a clear straight-line insertion corridor:
    no active cylinder lies on the +X path from (front_x - 0.05, p1_y, p1_z)
    to p1 within (cyl_radius + stick_radius + insert_buffer) in y.

    Args:
        rng: numpy random generator
        N: number of envs
        shelf: shelf geometry dict
        cylinder_positions: (N, max_cyl, 3) array of cylinder positions
            (hidden cylinders have x > 2.0)
        radius: jitter radius around cylinder center (meters)
        cyl_radius: cylinder body radius (meters)
        stick_radius: push stick radius (meters)
        insert_buffer: extra clearance on top of (cyl_r + stick_r) (meters)

    Returns:
        p1: (N, 3) entry positions
        p2: (N, 3) sweep endpoints
        clear_mask: (N,) bool — True if p1 has a clear insertion path. Envs
            with False should be excluded from the dataset (200-try cap fired).
    """
    # Start with fully random pushes
    p1, p2 = _sample_pushes(rng, N, shelf)

    margin = 0.02
    y_clearance = cyl_radius + stick_radius + insert_buffer
    x_pad = cyl_radius + stick_radius + insert_buffer  # cylinder past p1
    approach_x = shelf["front_x"] - 0.05
    x_back = approach_x - stick_radius

    def _insertion_clear(pt_xy, cyls_xy_active):
        """True iff the +X insertion corridor at y=pt_xy[1] up to x=pt_xy[0]
        is free of every active cylinder."""
        if len(cyls_xy_active) == 0:
            return True
        cx = cyls_xy_active[:, 0]
        cy = cyls_xy_active[:, 1]
        in_y_band = np.abs(cy - pt_xy[1]) < y_clearance
        in_x_range = (cx >= x_back) & (cx <= pt_xy[0] + x_pad)
        return not bool((in_y_band & in_x_range).any())

    clear_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        strategy = rng.random()
        cyls = cylinder_positions[i]  # (max_cyl, 3)
        active = cyls[:, 0] < 2.0
        active_xy = cyls[active, :2]

        # Ensure the initial random p1 has a clear insertion corridor;
        # if not, resample p1 uniformly in-shelf.
        if not _insertion_clear(p1[i, :2], active_xy):
            for _ in range(200):
                p1[i, 0] = rng.uniform(shelf["front_x"] + margin,
                                        shelf["front_x"] + shelf["depth"] - margin)
                p1[i, 1] = rng.uniform(-shelf["half_w"] + margin,
                                        shelf["half_w"] - margin)
                if _insertion_clear(p1[i, :2], active_xy):
                    break

        if active.any() and strategy < 0.25:
            # Strategy 1: p1 near a chosen cylinder but with clear corridor.
            cyl_idx = rng.choice(np.where(active)[0])
            cyl_pos = cyls[cyl_idx]
            for _ in range(200):
                cand = np.array([
                    cyl_pos[0] + rng.uniform(-radius, radius),
                    cyl_pos[1] + rng.uniform(-radius, radius),
                ])
                cand[0] = np.clip(cand[0], shelf["front_x"] + margin,
                                   shelf["front_x"] + shelf["depth"] - margin)
                cand[1] = np.clip(cand[1], -shelf["half_w"] + margin,
                                   shelf["half_w"] - margin)
                if _insertion_clear(cand, active_xy):
                    p1[i, 0] = cand[0]
                    p1[i, 1] = cand[1]
                    break
            # If we never found a clear cand, keep the pre-checked p1
            # (still corridor-clear from the block above).
        elif active.any() and strategy < 0.50:
            # Strategy 2: p1 stays where it is, p2 jitters near a cylinder.
            cyl_idx = rng.choice(np.where(active)[0])
            cyl_pos = cyls[cyl_idx]
            p2[i, 0] = cyl_pos[0] + rng.uniform(-radius, radius)
            p2[i, 1] = cyl_pos[1] + rng.uniform(-radius, radius)
            p2[i, 0] = np.clip(p2[i, 0], shelf["front_x"] + margin,
                                shelf["front_x"] + shelf["depth"] - margin)
            p2[i, 1] = np.clip(p2[i, 1], -shelf["half_w"] + margin,
                                shelf["half_w"] - margin)
        # else: strategy 3 — keep fully random

        # Ensure minimum sweep distance
        while np.linalg.norm(p2[i, :2] - p1[i, :2]) < 0.06:
            p2[i, 0] = rng.uniform(shelf["front_x"] + margin,
                                    shelf["front_x"] + shelf["depth"] - margin)
            p2[i, 1] = rng.uniform(-shelf["half_w"] + margin,
                                    shelf["half_w"] - margin)
            p2[i, 2] = p1[i, 2]

        clear_mask[i] = _insertion_clear(p1[i, :2], active_xy)

    return p1, p2, clear_mask


def _scene_validity_masks(cyl_pos, cyl_quat, shelf_geom, tilt_cos=0.5):
    """Vectorized scene-validity check across N envs and num_cyl cylinders.

    NB: cylinders in this env are spawned via CYL_UPRIGHT_Q which aligns the
    cylinder's BODY +X axis with world +Z (i.e. the height axis is body-X,
    not body-Z). So upright-ness is measured by the world-z component of the
    body x-axis: `2*(qx*qz + qw*qy)`. For CYL_UPRIGHT_Q this is 1.0; for a
    cylinder lying on its side it's 0.

    Args:
        cyl_pos: (N, num_cyl, 3) cylinder positions (hidden cyl: x > 2.0)
        cyl_quat: (N, num_cyl, 4) cylinder quaternions in (qw, qx, qy, qz)
        shelf_geom: object with attrs front_x, back_x, half_w, surface_z
        tilt_cos: minimum world-z component of body x-axis (cos(60°) = 0.5)

    Returns:
        valid: (N,) bool — True iff every active cylinder is in shelf bounds
            and upright. An env with no active cylinders is valid by default.
    """
    g = shelf_geom
    active = cyl_pos[..., 0] < 2.0  # (N, num_cyl)
    off_shelf = (
        (cyl_pos[..., 0] < g.front_x - 0.02)
        | (cyl_pos[..., 0] > g.back_x + 0.02)
        | (cyl_pos[..., 1] < -g.half_w - 0.02)
        | (cyl_pos[..., 1] > g.half_w + 0.02)
        | (cyl_pos[..., 2] < g.surface_z - 0.01)
    )
    qw, qx, qy, qz = (
        cyl_quat[..., 0], cyl_quat[..., 1],
        cyl_quat[..., 2], cyl_quat[..., 3],
    )
    # |R[2,0]| = |2(qx*qz - qw*qy)| — magnitude of world-z projection of
    # body-x. Use abs() because the spawn rotation aligns body-X with -world-Z;
    # a flipped cylinder (body-X = +world-Z) is equally upright.
    body_x_world_z = np.abs(2.0 * (qx * qz - qw * qy))
    toppled = body_x_world_z < tilt_cos
    bad = active & (off_shelf | toppled)
    return ~bad.any(axis=1)  # (N,)


def _extract_frame(obs, camera_name, env_indices, device="cpu"):
    """Extract RGB, depth, and instance segmentation from a batched obs.

    Returns (rgb, depth, seg) where:
        rgb: (len(env_indices), H, W, 3) uint8
        depth: (len(env_indices), H, W) float32 in meters
        seg: (len(env_indices), H, W) int16 — per-pixel actor ID
    """
    sensor_data = obs["sensor_data"][camera_name]
    rgb = sensor_data["rgb"].cpu().numpy()      # (N, H, W, 3) or (H, W, 3)
    depth = sensor_data["depth"].cpu().numpy()   # (N, H, W, 1) or (H, W, 1)
    seg = sensor_data["segmentation"].cpu().numpy()  # (N, H, W, 1) or (H, W, 1)

    if rgb.ndim == 4:
        rgb = rgb[env_indices]
    if depth.ndim == 4:
        depth = depth[env_indices, :, :, 0]
    elif depth.ndim == 3:
        depth = depth[:, :, 0]
    if seg.ndim == 4:
        seg = seg[env_indices, :, :, 0]
    elif seg.ndim == 3:
        seg = seg[:, :, 0]

    return rgb, depth, seg.astype(np.int16)


def _extract_camera_params(obs, camera_name, env_indices):
    """Extract intrinsics and extrinsics for given env indices.

    Returns (intrinsics, extrinsics) where:
        intrinsics: (len(env_indices), 3, 3) float32
        extrinsics: (len(env_indices), 3, 4) float32
    """
    sensor_params = obs["sensor_param"]
    cam_params = sensor_params[camera_name]

    intr = cam_params["intrinsic_cv"].cpu().numpy()
    extr = cam_params["extrinsic_cv"].cpu().numpy()

    if intr.ndim == 3:
        intr = intr[env_indices]
    if extr.ndim == 3:
        extr = extr[env_indices]

    return intr, extr


@register_solver("shelf_visual_collect")
class ShelfVisualCollectSolver(BaseSolver):
    """GPU-batched shelf push solver with visual data collection.

    Same push logic as ShelfPandaStickPushBatchedSolver, but captures
    visual data and writes to zarr (first+last) or MP4+HDF5 (full trajectory).
    """

    def __init__(
        self,
        pushes_per_episode: int = 3,
        n_collect_envs: int = -1,
        output_dir: str = "data/visual_trajectories",
        worker_id: int = -1,
        collect_mode: str = "zarr",
        interaction_pos_thresh: float = 0.005,
    ):
        self.pushes_per_episode = int(pushes_per_episode)
        self.n_collect_envs = int(n_collect_envs)
        self.output_dir = output_dir
        self.worker_id = int(worker_id)
        assert collect_mode in ("zarr", "full"), \
            f"collect_mode must be 'zarr' or 'full', got {collect_mode!r}"
        self.collect_mode = collect_mode
        self.interaction_pos_thresh = float(interaction_pos_thresh)

        # Zarr store (lazy init)
        self._zarr_store = None
        self._zarr_count = 0

        # Full-mode dataset (lazy init)
        self._dataset = None
        self._traj_counter = 0

        # Cached across episodes
        self._push_skill = None
        self._motion_gen = None
        self._shelf_dict = None
        self._n_arm = None
        self._camera_names = None
        self._N = None

    # ------------------------------------------------------------------
    # Zarr dataset management
    # ------------------------------------------------------------------

    def _init_zarr(self, N, H, W, camera_name):
        """Lazy-init zarr store with resume support."""
        import zarr

        if self.worker_id >= 0:
            zarr_path = f"{self.output_dir}/worker_{self.worker_id:03d}.zarr"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zarr_path = f"{self.output_dir}/{ts}.zarr"

        os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)
        self._zarr_store = zarr.open(zarr_path, mode="a")

        # Create or resume datasets
        if "rgb_before" in self._zarr_store:
            self._zarr_count = self._zarr_store["rgb_before"].shape[0]
            logger.info(
                "Resuming zarr: %d existing samples in %s",
                self._zarr_count,
                zarr_path,
            )
        else:
            chunk_n = min(N, 256)
            self._zarr_store.create_dataset(
                "rgb_before", shape=(0, H, W, 3), chunks=(chunk_n, H, W, 3),
                dtype="uint8",
            )
            self._zarr_store.create_dataset(
                "rgb_after", shape=(0, H, W, 3), chunks=(chunk_n, H, W, 3),
                dtype="uint8",
            )
            self._zarr_store.create_dataset(
                "depth_before", shape=(0, H, W), chunks=(chunk_n, H, W),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "depth_after", shape=(0, H, W), chunks=(chunk_n, H, W),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "entry_position", shape=(0, 3), chunks=(chunk_n, 3),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "sweep_position", shape=(0, 3), chunks=(chunk_n, 3),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "intrinsics", shape=(0, 3, 3), chunks=(chunk_n, 3, 3),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "extrinsics", shape=(0, 3, 4), chunks=(chunk_n, 3, 4),
                dtype="float32",
            )
            self._zarr_store.create_dataset(
                "seg_before", shape=(0, H, W), chunks=(chunk_n, H, W),
                dtype="int16",
            )
            self._zarr_store.create_dataset(
                "seg_after", shape=(0, H, W), chunks=(chunk_n, H, W),
                dtype="int16",
            )
            logger.info("Created zarr dataset: %s", zarr_path)

    def _write_zarr_batch(self, rgb_before, rgb_after, depth_before, depth_after,
                          entry_positions, sweep_positions, intrinsics, extrinsics,
                          seg_before, seg_after):
        """Append a batch of (s, a, s') pairs to the zarr store."""
        n = rgb_before.shape[0]
        self._zarr_store["rgb_before"].append(rgb_before)
        self._zarr_store["rgb_after"].append(rgb_after)
        self._zarr_store["depth_before"].append(depth_before)
        self._zarr_store["depth_after"].append(depth_after)
        self._zarr_store["entry_position"].append(entry_positions)
        self._zarr_store["sweep_position"].append(sweep_positions)
        self._zarr_store["intrinsics"].append(intrinsics)
        self._zarr_store["extrinsics"].append(extrinsics)
        self._zarr_store["seg_before"].append(seg_before)
        self._zarr_store["seg_after"].append(seg_after)
        self._zarr_count += n

    # ------------------------------------------------------------------
    # Full-trajectory dataset management
    # ------------------------------------------------------------------

    def _init_dataset(self, N):
        """Lazy-init ManiSkillTrajectoryDataset with resume support."""
        import glob

        if self.worker_id >= 0:
            dataset_dir = f"{self.output_dir}/worker_{self.worker_id:03d}"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_dir = f"{self.output_dir}/{ts}_N{N}"

        existing_dirs = glob.glob(os.path.join(dataset_dir, "traj_*"))
        n_existing = len(existing_dirs)

        self._dataset = ManiSkillTrajectoryDataset(
            root_dir=dataset_dir, force_reindex=True
        )

        if n_existing > 0:
            self._traj_counter = n_existing
            logger.info(
                "Resuming: found %d existing trajectories in %s",
                self._traj_counter,
                dataset_dir,
            )
        else:
            logger.info("Visual dataset: %s", dataset_dir)

    # ------------------------------------------------------------------
    # Planner + skill init (shared by both modes)
    # ------------------------------------------------------------------

    def _init_planner_and_skill(self, env, device, cuda, N):
        """One-time init of cuRobo planner and StickPush skill."""
        from curobo.geom.types import Cuboid, WorldConfig

        raw = env.unwrapped
        n_arm = len(get_arm_joint_names(ROBOT_UID))
        g = raw.shelf_geom
        robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
        sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
        sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
        world = WorldConfig(
            cuboid=[
                Cuboid(
                    name="shelf",
                    pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0],
                    dims=sd,
                )
            ]
        )

        os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")
        self._motion_gen = setup_curobo_planner(
            ROBOT_UID, world_configs=world, n_envs=1, warmup=False
        )
        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda).unsqueeze(0)

        self._push_skill = StickPush(
            env,
            self._motion_gen,
            robot_uid=ROBOT_UID,
            n_envs=N,
            n_arm_joints=n_arm,
            rest_qpos=REST_QPOS,
            robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )

        self._shelf_dict = dict(
            front_x=g.front_x,
            depth=g.depth,
            half_w=g.half_w,
            floor_z=g.floor_z,
            thickness=g.thickness,
            inner_h=g.inner_h,
        )
        self._n_arm = n_arm
        self._N = N
        logger.info("Initialized cuRobo planner + StickPush skill (cached)")

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        """Execute multi-push episode with visual data collection."""
        env.reset(seed=seed)
        raw = env.unwrapped
        device = raw.device
        cuda = torch.device("cuda:0")
        N = raw.num_envs

        # --- Robot reset (every episode) ---
        n_arm = self._n_arm or len(get_arm_joint_names(ROBOT_UID))
        rest = (
            torch.tensor(REST_QPOS, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1)
        )
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[:, :n_arm] = rest
        raw.agent.robot.set_qpos(qpos)
        raw.agent.set_control_mode("pd_joint_pos")
        raw.agent.controller.reset()
        for _ in range(10):
            env.step(rest)

        # --- One-time init: planner + skill ---
        if self._push_skill is None:
            self._init_planner_and_skill(env, device, cuda, N)

        push_skill = self._push_skill
        shelf_dict = self._shelf_dict
        g = raw.shelf_geom

        # --- One-time init: camera names ---
        if self._camera_names is None:
            self._camera_names = DEFAULT_CAMERA_NAMES
            if cfg is not None:
                cfg_cams = getattr(cfg.task, "camera_names", None)
                if cfg_cams is not None and len(cfg_cams) > 0:
                    self._camera_names = list(cfg_cams)
                elif cfg_cams is not None:
                    logger.warning(
                        "camera_names is empty; using default: %s",
                        DEFAULT_CAMERA_NAMES,
                    )
        camera_name = self._camera_names[0]  # primary camera for zarr mode

        n_collect = N if self.n_collect_envs < 0 else min(self.n_collect_envs, N)
        collect_envs = list(range(n_collect))
        collect_indices = np.array(collect_envs)

        if self.collect_mode == "zarr":
            return self._solve_zarr(
                env, raw, device, cuda, N, n_arm, rest, push_skill,
                shelf_dict, g, camera_name, collect_indices, seed, cfg,
            )
        else:
            return self._solve_full(
                env, raw, device, cuda, N, n_arm, rest, push_skill,
                shelf_dict, g, self._camera_names, collect_envs, seed, cfg,
            )

    # ------------------------------------------------------------------
    # Zarr collection: first + last frame per push
    # ------------------------------------------------------------------

    def _solve_zarr(self, env, raw, device, cuda, N, n_arm, rest,
                    push_skill, shelf_dict, g, camera_name,
                    collect_indices, seed, cfg):
        """Collect (s, a, s') pairs and write to zarr."""

        # Grab a frame to get resolution, init zarr
        obs_init = env.step(rest)[0]
        if self._zarr_store is None:
            sample_rgb = obs_init["sensor_data"][camera_name]["rgb"]
            H, W = sample_rgb.shape[1], sample_rgb.shape[2]
            self._init_zarr(len(collect_indices), H, W, camera_name)

        total_success = 0
        total_pushes = 0

        for push_idx in range(self.pushes_per_episode):
            push_rng = np.random.default_rng(
                seed + push_idx * 100 if seed is not None else push_idx
            )
            # Get cylinder positions for biased sampling
            cyl_pos = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            p1_np, p2_np, clear_mask = _sample_pushes_mixed(
                push_rng, N, shelf_dict, cyl_pos
            )
            approach_np = np.stack(
                [np.full(N, g.front_x - 0.05), p1_np[:, 1], p1_np[:, 2]], axis=1
            )
            retract_np = np.stack(
                [np.full(N, g.front_x - 0.30), p2_np[:, 1], p2_np[:, 2]], axis=1
            )

            # Check scene validity BEFORE push: reject envs with toppled/off-shelf cylinders
            cyl_pos_before = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            cyl_quat_before = np.stack([
                actor.pose.q.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 4)

            # Vectorized pre-push scene validity (off-shelf + toppled).
            scene_ok_pre = _scene_validity_masks(
                cyl_pos_before, cyl_quat_before, g
            )
            pre_valid = clear_mask[collect_indices] & scene_ok_pre[collect_indices]

            # Capture "before" frame
            obs_before = env.step(rest)[0]
            rgb_before, depth_before, seg_before = _extract_frame(
                obs_before, camera_name, collect_indices
            )
            intrinsics, extrinsics = _extract_camera_params(
                obs_before, camera_name, collect_indices
            )

            # Scale sweep steps: 200 base + extra for long sweeps
            # Each waypoint needs ~30 steps to converge, with 10 waypoints
            max_sweep_dist = np.max(np.linalg.norm(p2_np - p1_np, axis=1))
            sweep_steps = max(300, int(max_sweep_dist / 0.03 * 30))

            # Execute push (no callback needed — we only want first+last)
            result = push_skill(
                approach_positions=torch.tensor(
                    approach_np, device=cuda, dtype=torch.float32
                ),
                approach_quaternions=torch.tensor(
                    Q_INTO_SHELF, device=cuda, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(N, -1),
                entry_positions=torch.tensor(
                    p1_np, device=cuda, dtype=torch.float32
                ),
                sweep_positions=torch.tensor(
                    p2_np, device=cuda, dtype=torch.float32
                ),
                retract_positions=torch.tensor(
                    retract_np, device=cuda, dtype=torch.float32
                ),
                entry_max_steps=200,
                sweep_max_steps=sweep_steps,
            )

            # Capture "after" frame
            obs_after = env.step(rest)[0]
            rgb_after, depth_after, seg_after = _extract_frame(
                obs_after, camera_name, collect_indices
            )

            # Filter: only write envs where all cylinders are upright and in shelf
            cyl_pos_after = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            cyl_quat_after = np.stack([
                actor.pose.q.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 4)

            # Vectorized post-push scene validity.
            scene_ok_post = _scene_validity_masks(
                cyl_pos_after, cyl_quat_after, g
            )
            valid_mask = pre_valid & scene_ok_post[collect_indices]

            n_valid = valid_mask.sum()
            if n_valid > 0:
                self._write_zarr_batch(
                    rgb_before[valid_mask], rgb_after[valid_mask],
                    depth_before[valid_mask], depth_after[valid_mask],
                    p1_np[collect_indices[valid_mask]].astype(np.float32),
                    p2_np[collect_indices[valid_mask]].astype(np.float32),
                    intrinsics[valid_mask].astype(np.float32),
                    extrinsics[valid_mask].astype(np.float32),
                    seg_before[valid_mask], seg_after[valid_mask],
                )

            n_ok = result.success_mask.sum().item()
            total_success += n_ok
            total_pushes += N
            n_rejected = len(collect_indices) - n_valid
            n_blocked = int((~clear_mask[collect_indices]).sum())

            logger.info(
                "Push %d/%d: %d/%d success, %d rejected (toppled/off-shelf/blocked, %d blocked-insertion), zarr total: %d samples",
                push_idx + 1,
                self.pushes_per_episode,
                n_ok,
                N,
                n_rejected,
                n_blocked,
                self._zarr_count,
            )

        success_rate = total_success / max(total_pushes, 1)
        return SolverResult(
            success=success_rate > 0.5,
            elapsed_steps=total_pushes,
            info={
                "n_success": total_success,
                "n_total": total_pushes,
                "success_rate": success_rate,
                "zarr_samples": self._zarr_count,
            },
        )

    # ------------------------------------------------------------------
    # Full-trajectory collection: all frames, MP4 + HDF5
    # ------------------------------------------------------------------

    def _solve_full(self, env, raw, device, cuda, N, n_arm, rest,
                    push_skill, shelf_dict, g, camera_names,
                    collect_envs, seed, cfg):
        """Collect full trajectory and write to ManiSkillTrajectoryDataset."""

        if self._dataset is None:
            self._init_dataset(N)

        total_success = 0
        total_pushes = 0

        for push_idx in range(self.pushes_per_episode):
            push_rng = np.random.default_rng(
                seed + push_idx * 100 if seed is not None else push_idx
            )
            # Get cylinder positions for biased sampling
            cyl_pos = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            p1_np, p2_np, clear_mask = _sample_pushes_mixed(
                push_rng, N, shelf_dict, cyl_pos
            )
            approach_np = np.stack(
                [np.full(N, g.front_x - 0.05), p1_np[:, 1], p1_np[:, 2]], axis=1
            )
            retract_np = np.stack(
                [np.full(N, g.front_x - 0.30), p2_np[:, 1], p2_np[:, 2]], axis=1
            )

            # Snapshot cylinder state before the push for interaction filter
            cyl_pos_before = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            cyl_quat_before = np.stack([
                actor.pose.q.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 4)

            # Per-push visual streams
            push_streams = {}
            for eidx in collect_envs:
                vs, md = init_camera_streams(camera_names)
                md["qpos"] = []
                md["tcp_pos"] = []
                md["tcp_quat"] = []
                push_streams[eidx] = (vs, md)

            def _visual_callback(step, obs, rew):
                if obs is None:
                    return
                sensor_data = obs.get("sensor_data", {})
                sensor_params = obs.get("sensor_param", {})
                if not sensor_data:
                    return
                # Batched GPU→CPU: one transfer per tensor instead of one per env.
                qpos_np = raw.agent.robot.get_qpos().cpu().numpy()      # (N, dof)
                tcp_p_np = raw.agent.tcp.pose.p.cpu().numpy()           # (N, 3)
                tcp_q_np = raw.agent.tcp.pose.q.cpu().numpy()           # (N, 4)
                obj_pq = {
                    name: (actor.pose.p.cpu().numpy(), actor.pose.q.cpu().numpy())
                    for name, actor in raw.get_objects().items()
                }
                for eidx in collect_envs:
                    vs, md = push_streams[eidx]
                    for cam in camera_names:
                        if cam in sensor_data:
                            process_camera_data(
                                env, cam, sensor_data[cam],
                                sensor_params, vs, md, env_idx=eidx,
                            )
                    md["qpos"].append(qpos_np[eidx].copy())
                    md["tcp_pos"].append(tcp_p_np[eidx].copy())
                    md["tcp_quat"].append(tcp_q_np[eidx].copy())
                    for name, (p_np, q_np) in obj_pq.items():
                        md.setdefault(f"{name}_pos", []).append(p_np[eidx].copy())
                        md.setdefault(f"{name}_quat", []).append(q_np[eidx].copy())

            # Scale sweep steps: 200 base + extra for long sweeps
            # Each waypoint needs ~30 steps to converge, with 10 waypoints
            max_sweep_dist = np.max(np.linalg.norm(p2_np - p1_np, axis=1))
            sweep_steps = max(300, int(max_sweep_dist / 0.03 * 30))

            # Execute push
            result = push_skill(
                approach_positions=torch.tensor(
                    approach_np, device=cuda, dtype=torch.float32
                ),
                approach_quaternions=torch.tensor(
                    Q_INTO_SHELF, device=cuda, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(N, -1),
                entry_positions=torch.tensor(
                    p1_np, device=cuda, dtype=torch.float32
                ),
                sweep_positions=torch.tensor(
                    p2_np, device=cuda, dtype=torch.float32
                ),
                retract_positions=torch.tensor(
                    retract_np, device=cuda, dtype=torch.float32
                ),
                entry_max_steps=200,
                sweep_max_steps=sweep_steps,
                step_callback=_visual_callback,
            )

            n_ok = result.success_mask.sum().item()
            total_success += n_ok
            total_pushes += N

            # Post-push cylinder state for interaction filter
            cyl_pos_after = np.stack([
                actor.pose.p.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 3)
            cyl_quat_after = np.stack([
                actor.pose.q.cpu().numpy()
                for actor in raw.get_objects().values()
            ], axis=1)  # (N, num_cyl, 4)

            pos_thresh = self.interaction_pos_thresh
            # Vectorized validity & interaction masks (shape (N,) each).
            scene_ok_pre = _scene_validity_masks(
                cyl_pos_before, cyl_quat_before, g
            )
            scene_ok_post = _scene_validity_masks(
                cyl_pos_after, cyl_quat_after, g
            )
            active_pre = cyl_pos_before[..., 0] < 2.0  # (N, num_cyl)
            has_active = active_pre.any(axis=1)
            disp = np.linalg.norm(cyl_pos_after - cyl_pos_before, axis=-1)  # (N, num_cyl)
            moved_any = (disp > pos_thresh) & active_pre
            interaction = moved_any.any(axis=1)

            def _env_has_interaction(eidx):
                if not bool(clear_mask[eidx]):
                    return False, "blocked_insertion"
                if not bool(has_active[eidx]):
                    return False, "no_cyl"
                if not bool(scene_ok_pre[eidx]):
                    return False, "bad_pre_state"
                if not bool(scene_ok_post[eidx]):
                    return False, "bad_post_state"
                if not bool(interaction[eidx]):
                    return False, "no_interaction"
                return True, "ok"

            # Write trajectories
            n_written = 0
            n_empty = 0
            n_bad_state = 0
            n_blocked = 0
            n_bad_pre = 0
            for eidx in collect_envs:
                vs, md = push_streams[eidx]
                video_np = {
                    k: np.array(v) for k, v in vs.items() if len(v) > 0
                }
                if not video_np:
                    continue

                ok, reason = _env_has_interaction(eidx)
                if not ok:
                    if reason == "no_interaction":
                        n_empty += 1
                    elif reason == "blocked_insertion":
                        n_blocked += 1
                    elif reason == "bad_pre_state":
                        n_bad_pre += 1
                    elif reason == "bad_post_state":
                        n_bad_state += 1
                    else:
                        n_bad_state += 1
                    continue
                meta_np = {}
                for k, v in md.items():
                    if isinstance(v, list) and len(v) > 0:
                        try:
                            meta_np[k] = np.array(v)
                        except ValueError:
                            pass
                meta_np["push_idx"] = push_idx
                meta_np["episode_seed"] = seed if seed is not None else -1
                meta_np["env_idx"] = eidx
                meta_np["entry_position"] = p1_np[eidx].copy()
                meta_np["sweep_position"] = p2_np[eidx].copy()
                if self.worker_id >= 0:
                    meta_np["worker_id"] = self.worker_id

                traj_data = TrajectoryData(
                    success=bool(result.success_mask[eidx].item()),
                    video_streams=video_np,
                    metadata=meta_np,
                )
                self._dataset.write_trajectory(
                    f"{self._traj_counter:06d}", traj_data
                )
                self._traj_counter += 1
                n_written += 1

            logger.info(
                "Push %d/%d: %d/%d success, wrote %d (rej: %d empty, %d bad_state, %d blocked, %d bad_pre), total: %d",
                push_idx + 1,
                self.pushes_per_episode,
                n_ok,
                N,
                n_written,
                n_empty,
                n_bad_state,
                n_blocked,
                n_bad_pre,
                self._traj_counter,
            )

        success_rate = total_success / max(total_pushes, 1)
        return SolverResult(
            success=success_rate > 0.5,
            elapsed_steps=total_pushes,
            info={
                "n_success": total_success,
                "n_total": total_pushes,
                "success_rate": success_rate,
                "trajectories_written": self._traj_counter,
            },
        )
