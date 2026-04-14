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


def _sample_pushes_mixed(rng, N, shelf, cylinder_positions, radius=0.05):
    """Sample pushes with three strategies (1/3 each):

    1. p1 within `radius` of a random cylinder, p2 random
    2. p1 random, p2 within `radius` of a random cylinder
    3. Fully random (no bias)

    Args:
        rng: numpy random generator
        N: number of envs
        shelf: shelf geometry dict
        cylinder_positions: (N, max_cyl, 3) array of cylinder positions
            (hidden cylinders have x > 2.0)
        radius: jitter radius around cylinder center (meters)
    """
    # Start with fully random pushes
    p1, p2 = _sample_pushes(rng, N, shelf)

    for i in range(N):
        strategy = rng.random()
        cyls = cylinder_positions[i]  # (max_cyl, 3)
        active = cyls[:, 0] < 2.0
        if not active.any():
            continue  # no cylinders, keep random

        # Pick a random active cylinder
        cyl_idx = rng.choice(np.where(active)[0])
        cyl_pos = cyls[cyl_idx]

        margin = 0.02
        if strategy < 0.25:
            # Strategy 1: p1 near cylinder, p2 random
            p1[i, 0] = cyl_pos[0] + rng.uniform(-radius, radius)
            p1[i, 1] = cyl_pos[1] + rng.uniform(-radius, radius)
            # Clamp to shelf bounds
            p1[i, 0] = np.clip(p1[i, 0], shelf["front_x"] + margin,
                                shelf["front_x"] + shelf["depth"] - margin)
            p1[i, 1] = np.clip(p1[i, 1], -shelf["half_w"] + margin,
                                shelf["half_w"] - margin)
        elif strategy < 0.50:
            # Strategy 2: p1 random, p2 near cylinder
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

    return p1, p2


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
            p1_np, p2_np = _sample_pushes_mixed(
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

            pre_valid = np.ones(len(collect_indices), dtype=bool)
            for ci, eidx in enumerate(collect_indices):
                for cyl_i in range(cyl_pos_before.shape[1]):
                    cp = cyl_pos_before[eidx, cyl_i]
                    cq = cyl_quat_before[eidx, cyl_i]
                    if cp[0] > 2.0:
                        continue
                    if (cp[0] < g.front_x - 0.02 or cp[0] > g.back_x + 0.02
                            or cp[1] < -g.half_w - 0.02 or cp[1] > g.half_w + 0.02
                            or cp[2] < g.surface_z - 0.01):
                        pre_valid[ci] = False
                        break
                    if abs(cq[0]) < 0.5:
                        pre_valid[ci] = False
                        break

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

            valid_mask = pre_valid.copy()
            for ci, eidx in enumerate(collect_indices):
                for cyl_i in range(cyl_pos_after.shape[1]):
                    cp = cyl_pos_after[eidx, cyl_i]
                    cq = cyl_quat_after[eidx, cyl_i]
                    if cp[0] > 2.0:
                        continue  # hidden cylinder, skip
                    # Check: inside shelf bounds (with margin)
                    if (cp[0] < g.front_x - 0.02 or cp[0] > g.back_x + 0.02
                            or cp[1] < -g.half_w - 0.02 or cp[1] > g.half_w + 0.02
                            or cp[2] < g.surface_z - 0.01):
                        valid_mask[ci] = False
                        break
                    # Check: upright (w component of quaternion close to ±0.707
                    # for a cylinder standing on its end, z-axis aligned)
                    # Upright quaternion has |qw| > 0.5 (within ~60° of vertical)
                    if abs(cq[0]) < 0.5:
                        valid_mask[ci] = False
                        break

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

            logger.info(
                "Push %d/%d: %d/%d success, %d rejected (toppled/off-shelf), zarr total: %d samples",
                push_idx + 1,
                self.pushes_per_episode,
                n_ok,
                N,
                n_rejected,
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
            p1_np, p2_np = _sample_pushes_mixed(
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
                for eidx in collect_envs:
                    vs, md = push_streams[eidx]
                    for cam in camera_names:
                        if cam in sensor_data:
                            process_camera_data(
                                env, cam, sensor_data[cam],
                                sensor_params, vs, md, env_idx=eidx,
                            )
                    md["qpos"].append(
                        raw.agent.robot.get_qpos()[eidx].cpu().numpy().copy()
                    )
                    md["tcp_pos"].append(
                        raw.agent.tcp.pose.p[eidx].cpu().numpy().copy()
                    )
                    md["tcp_quat"].append(
                        raw.agent.tcp.pose.q[eidx].cpu().numpy().copy()
                    )
                    for name, actor in raw.get_objects().items():
                        md.setdefault(f"{name}_pos", []).append(
                            actor.pose.p[eidx].cpu().numpy().copy()
                        )
                        md.setdefault(f"{name}_quat", []).append(
                            actor.pose.q[eidx].cpu().numpy().copy()
                        )

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

            def _env_has_interaction(eidx):
                active = cyl_pos_before[eidx, :, 0] < 2.0
                if not active.any():
                    return False, "no_cyl"
                # reject if any active cylinder ended toppled or off-shelf
                for ci in np.where(active)[0]:
                    cp = cyl_pos_after[eidx, ci]
                    cq = cyl_quat_after[eidx, ci]
                    if (cp[0] < g.front_x - 0.02 or cp[0] > g.back_x + 0.02
                            or cp[1] < -g.half_w - 0.02
                            or cp[1] > g.half_w + 0.02
                            or cp[2] < g.surface_z - 0.01):
                        return False, "off_shelf"
                    if abs(cq[0]) < 0.5:
                        return False, "toppled"
                # require at least one active cylinder to have moved meaningfully
                for ci in np.where(active)[0]:
                    dp = np.linalg.norm(
                        cyl_pos_after[eidx, ci] - cyl_pos_before[eidx, ci]
                    )
                    if dp > pos_thresh:
                        return True, "ok"
                return False, "no_interaction"

            # Write trajectories
            n_written = 0
            n_empty = 0
            n_bad_state = 0
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
                "Push %d/%d: %d/%d success, wrote %d (rej: %d empty, %d bad_state), total: %d",
                push_idx + 1,
                self.pushes_per_episode,
                n_ok,
                N,
                n_written,
                n_empty,
                n_bad_state,
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
