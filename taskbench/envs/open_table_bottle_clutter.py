"""Dense open-table bottle clutter for random scene generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from taskbench.envs.open_table_defaults import (
    BOTTLE_BALLAST_HALF_LENGTH_BASE, BOTTLE_BALLAST_OFFSET_BASE,
    BOTTLE_BALLAST_RADIUS_BASE, BOTTLE_BODY_HALF_LENGTH_BASE,
    BOTTLE_BODY_RADIUS_BASE, BOTTLE_NECK_HALF_LENGTH_BASE,
    BOTTLE_NECK_OFFSET_BASE, BOTTLE_NECK_RADIUS_BASE, BOTTLE_SCALE,
    DENSE_LAYOUT_PROB, EDGE_MARGIN, MIXED_LAYOUT_PROB, PANDA_TABLE_DEFAULTS,
    PLACEMENT_CLEARANCE, PLACEMENT_JITTER, PLACEMENT_SPACING)
from taskbench.envs.bottle_builder import (
    BOTTLE_UPRIGHT_Q,
    build_bottle_actor,
    get_visual_footprint_radius,
)
from taskbench.envs.open_table_env import TARGET_COLOR, OpenTableEnv
from taskbench.envs.open_table_scene import (COMPACT_OPEN_TABLE_CENTER_XY,
                                             COMPACT_OPEN_TABLE_SIZE_XY)
from taskbench.envs.placement import (PlacementGrid, build_rect_grid,
                                      clamp_jitter, jitter_positions,
                                      sample_frontier_cells)

BOTTLE_COLOR = [0.20, 0.40, 0.85, 1.0]


@register_env(
    "OpenTableBottleClutter-v1", max_episode_steps=1400, asset_download_ids=["ycb"]
)
class OpenTableBottleClutterEnv(OpenTableEnv):
    """Open table with many bottles placed from a fast clutter sampler.

    The env keeps the bottle physics from ``OpenTableEnv`` but switches
    placement from a short deterministic row to a full-table lattice. A
    lightweight frontier sampler chooses occupied lattice cells so some scenes
    are clustered and others are more diffuse without relying on expensive
    rejection sampling.
    """

    # Mirrored from ManiSkill's TableSceneBuilder collision box after the
    # builder rotates the table by +90 degrees about Z.
    OPEN_TABLE_CENTER_XY = COMPACT_OPEN_TABLE_CENTER_XY.copy()
    OPEN_TABLE_HALF_EXTENTS_XY = 0.5 * COMPACT_OPEN_TABLE_SIZE_XY[[1, 0]]

    def __init__(
        self,
        *args,
        num_bottles: int = 15,
        use_full_table_workspace: bool = True,
        workspace_center_x: float | None = None,
        workspace_center_y: float | None = None,
        workspace_half_extent_x: float | None = None,
        workspace_half_extent_y: float | None = None,
        edge_margin: float = EDGE_MARGIN,
        placement_spacing: float = PLACEMENT_SPACING,
        placement_clearance: float = PLACEMENT_CLEARANCE,
        placement_jitter: float = PLACEMENT_JITTER,
        bottle_scale: float = BOTTLE_SCALE,
        layout_mode: str = "auto",
        dense_layout_prob: float = DENSE_LAYOUT_PROB,
        mixed_layout_prob: float = MIXED_LAYOUT_PROB,
        num_launch_corridors: int = 0,
        random_yaw: bool = True,
        movement_success_threshold: float = 0.02,
        **kwargs,
    ):
        self.bottle_scale = float(bottle_scale)
        if self.bottle_scale <= 0:
            raise ValueError("bottle_scale must be positive")

        kwargs.setdefault(
            "bottle_body_radius", BOTTLE_BODY_RADIUS_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_body_half_length", BOTTLE_BODY_HALF_LENGTH_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_neck_radius", BOTTLE_NECK_RADIUS_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_neck_half_length", BOTTLE_NECK_HALF_LENGTH_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_neck_offset", BOTTLE_NECK_OFFSET_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_ballast_radius", BOTTLE_BALLAST_RADIUS_BASE * self.bottle_scale
        )
        kwargs.setdefault(
            "bottle_ballast_half_length",
            BOTTLE_BALLAST_HALF_LENGTH_BASE * self.bottle_scale,
        )
        kwargs.setdefault(
            "bottle_ballast_offset", BOTTLE_BALLAST_OFFSET_BASE * self.bottle_scale
        )
        kwargs.setdefault("object_kind", "bottle")
        kwargs.setdefault("num_cylinders", num_bottles)

        self.num_bottles = int(num_bottles)
        self.use_full_table_workspace = bool(use_full_table_workspace)
        self.workspace_center_x = (
            None if workspace_center_x is None else float(workspace_center_x)
        )
        self.workspace_center_y = (
            None if workspace_center_y is None else float(workspace_center_y)
        )
        self.workspace_half_extent_x = (
            None if workspace_half_extent_x is None else float(workspace_half_extent_x)
        )
        self.workspace_half_extent_y = (
            None if workspace_half_extent_y is None else float(workspace_half_extent_y)
        )
        self.edge_margin = float(edge_margin)
        self.placement_spacing = float(placement_spacing)
        self.placement_clearance = float(placement_clearance)
        self.requested_placement_jitter = float(placement_jitter)
        self.layout_mode = str(layout_mode)
        self.dense_layout_prob = float(dense_layout_prob)
        self.mixed_layout_prob = float(mixed_layout_prob)
        self.num_launch_corridors = int(num_launch_corridors)
        self.random_yaw = bool(random_yaw)
        self.movement_success_threshold = float(movement_success_threshold)

        bottle_radius = float(kwargs.get("bottle_body_radius", 0.020))
        self.min_center_distance = 2.0 * bottle_radius + self.placement_clearance
        if self.placement_spacing < self.min_center_distance:
            raise ValueError(
                "placement_spacing must be at least 2 * bottle_body_radius + "
                f"placement_clearance ({self.min_center_distance:.4f})"
            )

        self._workspace_lo_xy = np.zeros((2,), dtype=np.float32)
        self._workspace_hi_xy = np.zeros((2,), dtype=np.float32)
        self._placement_lo_xy = np.zeros((2,), dtype=np.float32)
        self._placement_hi_xy = np.zeros((2,), dtype=np.float32)
        self._placement_grid: PlacementGrid | None = None
        self._grid_centers_xy = np.empty((0, 2), dtype=np.float32)
        self.placement_jitter = 0.0

        super().__init__(*args, **kwargs)
        self.initial_positions_xy = np.zeros((self.num_bottles, 2), dtype=np.float32)
        self.initial_positions_xy_batched = torch.zeros(
            self.num_envs, self.num_bottles, 2, device=self.device
        )
        self.launch_corridors: list[dict[str, object]] = []
        self.scene_layout: dict[str, object] = {}

    def _workspace_bounds_from_config(self) -> tuple[np.ndarray, np.ndarray]:
        if self.use_full_table_workspace:
            center_xy = self.OPEN_TABLE_CENTER_XY
            half_extents_xy = self.OPEN_TABLE_HALF_EXTENTS_XY
        else:
            if None in (
                self.workspace_center_x,
                self.workspace_center_y,
                self.workspace_half_extent_x,
                self.workspace_half_extent_y,
            ):
                raise ValueError(
                    "workspace_center_* and workspace_half_extent_* must all be set "
                    "when use_full_table_workspace is false"
                )
            center_xy = np.array(
                [self.workspace_center_x, self.workspace_center_y], dtype=np.float32
            )
            half_extents_xy = np.array(
                [self.workspace_half_extent_x, self.workspace_half_extent_y],
                dtype=np.float32,
            )
        lo_xy = center_xy - half_extents_xy
        hi_xy = center_xy + half_extents_xy
        return lo_xy.astype(np.float32), hi_xy.astype(np.float32)

    def _configure_placement_workspace(self) -> None:
        (
            self._workspace_lo_xy,
            self._workspace_hi_xy,
        ) = self._workspace_bounds_from_config()
        footprint_margin = self.edge_margin + get_visual_footprint_radius(self.obj)
        self._placement_lo_xy = self._workspace_lo_xy + footprint_margin
        self._placement_hi_xy = self._workspace_hi_xy - footprint_margin
        if np.any(self._placement_hi_xy < self._placement_lo_xy):
            raise ValueError("edge_margin leaves no usable tabletop area")

        self._placement_grid = build_rect_grid(
            self._placement_lo_xy, self._placement_hi_xy, self.placement_spacing
        )
        self.placement_jitter = clamp_jitter(
            self.requested_placement_jitter,
            grid_spacing=self._placement_grid.spacing,
            min_center_distance=self.min_center_distance,
        )
        self._grid_centers_xy = self._placement_grid.centers_xy.copy()
        if len(self._grid_centers_xy) < self.num_bottles:
            raise ValueError(
                f"Workspace only supports {len(self._grid_centers_xy)} bottles at "
                f"spacing {self.placement_spacing:.3f}, but num_bottles={self.num_bottles}"
            )

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._configure_placement_workspace()

    def _build_bottle(self, idx: int):
        color = TARGET_COLOR if idx == self.target_idx else BOTTLE_COLOR
        return build_bottle_actor(self.scene, self.obj, idx, color)

    def get_objects(self) -> dict[str, object]:
        return {obj.name: obj for obj in self.cylinders}

    def get_workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._workspace_lo_xy.copy(), self._workspace_hi_xy.copy()

    def get_grid_centers(self) -> np.ndarray:
        return self._grid_centers_xy.copy()

    def get_launch_corridors(self) -> list[dict[str, object]]:
        return []

    def get_scene_layout(self) -> dict[str, object]:
        layout = dict(self.scene_layout)
        if "occupied_indices" in layout:
            layout["occupied_indices"] = np.asarray(
                layout["occupied_indices"], dtype=np.int32
            ).copy()
        if "seed_indices" in layout:
            layout["seed_indices"] = np.asarray(
                layout["seed_indices"], dtype=np.int32
            ).copy()
        if "positions_xy" in layout:
            layout["positions_xy"] = np.asarray(
                layout["positions_xy"], dtype=np.float32
            ).copy()
        return layout

    def is_inside_workspace(self, xy, margin: float = 0.0) -> bool:
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
        margin = float(margin)
        return bool(
            np.all(xy >= self._workspace_lo_xy + margin)
            and np.all(xy <= self._workspace_hi_xy - margin)
        )

    def get_bottle_positions_xy(self) -> tuple[list[str], np.ndarray]:
        """Return bottle XY positions for env 0 (single-env helper)."""
        names = [obj.name for obj in self.cylinders]
        positions = np.stack(
            [
                obj.pose.p[0, :2].detach().cpu().numpy().astype(np.float32)
                for obj in self.cylinders
            ],
            axis=0,
        )
        return names, positions

    def _bottle_positions_xy_batched(self) -> torch.Tensor:
        """Return bottle XY positions for all envs. Shape: (N, num_bottles, 2)."""
        return torch.stack(
            [obj.pose.p[:, :2] for obj in self.cylinders], dim=1
        )

    def get_scene_spec(self) -> dict[str, object]:
        object_names = [obj.name for obj in self.cylinders]
        object_positions = np.stack(
            [
                obj.pose.p[0].detach().cpu().numpy().astype(np.float32)
                for obj in self.cylinders
            ],
            axis=0,
        )
        object_quats = np.stack(
            [
                obj.pose.q[0].detach().cpu().numpy().astype(np.float32)
                for obj in self.cylinders
            ],
            axis=0,
        )
        robot_root_position = (
            self.agent.robot.pose.p[0].detach().cpu().numpy().astype(np.float32)
        )
        robot_root_quat = (
            self.agent.robot.pose.q[0].detach().cpu().numpy().astype(np.float32)
        )
        robot_qpos = (
            self.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
        )
        workspace_lo, workspace_hi = self.get_workspace_bounds()
        return {
            "env_id": "OpenTableBottleClutter-v1",
            "num_bottles": int(self.num_bottles),
            "target_idx": int(self.target_idx),
            "object_names": list(object_names),
            "object_positions_xyz": object_positions,
            "object_quats_wxyz": object_quats,
            "robot_root_position_xyz": robot_root_position,
            "robot_root_quat_wxyz": robot_root_quat,
            "robot_qpos": robot_qpos,
            "workspace_lo_xy": workspace_lo,
            "workspace_hi_xy": workspace_hi,
            "layout": self.get_scene_layout(),
        }

    def apply_scene_spec(self, scene_spec: dict[str, object]) -> None:
        target_idx = int(scene_spec.get("target_idx", self.target_idx))
        if target_idx != self.target_idx:
            raise ValueError(
                f"scene target_idx={target_idx} does not match env target_idx={self.target_idx}"
            )

        object_positions = np.asarray(
            scene_spec["object_positions_xyz"], dtype=np.float32
        )
        object_quats = np.asarray(scene_spec["object_quats_wxyz"], dtype=np.float32)
        if object_positions.shape != (self.num_bottles, 3):
            raise ValueError(
                "scene object_positions_xyz must have shape "
                f"({self.num_bottles}, 3), got {object_positions.shape}"
            )
        if object_quats.shape != (self.num_bottles, 4):
            raise ValueError(
                "scene object_quats_wxyz must have shape "
                f"({self.num_bottles}, 4), got {object_quats.shape}"
            )

        robot_root_position = np.asarray(
            scene_spec["robot_root_position_xyz"], dtype=np.float32
        )
        robot_root_quat = np.asarray(
            scene_spec["robot_root_quat_wxyz"], dtype=np.float32
        )
        robot_qpos = np.asarray(scene_spec["robot_qpos"], dtype=np.float32)

        self.agent.robot.set_root_pose(
            sapien.Pose(robot_root_position.tolist(), robot_root_quat.tolist())
        )
        robot_qpos_t = torch.as_tensor(
            robot_qpos, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.agent.robot.set_qpos(robot_qpos_t)
        if hasattr(self.agent.robot, "set_qvel"):
            self.agent.robot.set_qvel(torch.zeros_like(robot_qpos_t))

        for obj, position, quat in zip(self.cylinders, object_positions, object_quats):
            obj.set_pose(sapien.Pose(position.tolist(), quat.tolist()))

        self.initial_positions_xy_batched = self._bottle_positions_xy_batched()
        _, self.initial_positions_xy = self.get_bottle_positions_xy()
        layout = scene_spec.get("layout", {})
        self.scene_layout = dict(layout) if isinstance(layout, dict) else {}

    def _sample_bottle_quaternion(self) -> np.ndarray:
        if not self.random_yaw:
            return np.asarray(BOTTLE_UPRIGHT_Q, dtype=np.float32)
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        yaw_q = euler2quat(0.0, 0.0, yaw)
        return np.asarray(qmult(yaw_q, BOTTLE_UPRIGHT_Q), dtype=np.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self._placement_grid is None:
            raise RuntimeError("Placement grid is not configured")

        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._reset_robot(env_idx)

            # Validate that ManiSkill's scene builder still places the robot
            # where we expect.  The algorithmic scene sampler in
            # generate_clutter_scene_library.py bakes these values into HDF5
            # files, so a silent upstream change would corrupt generated data.
            # Only applies to the Panda robot — other robots have different defaults.
            # Only validate on full resets (env_idx covers all envs) to avoid
            # reading env 0 state during a partial reset of a different env.
            _is_panda = getattr(self.agent, "uid", "") in ("panda", "panda_wristcam")
            _is_full_reset = len(env_idx) == self.num_envs
            if _is_panda and _is_full_reset:
                _actual_pos = self.agent.robot.pose.p[0].detach().cpu().numpy()
                _actual_qpos = self.agent.robot.get_qpos()[0].detach().cpu().numpy()
                if not np.allclose(
                    _actual_pos, PANDA_TABLE_DEFAULTS.root_position, atol=1e-3
                ):
                    raise RuntimeError(
                        f"Robot root pose drifted from expected "
                        f"{PANDA_TABLE_DEFAULTS.root_position}: got {_actual_pos.tolist()}"
                    )
                if not np.allclose(
                    _actual_qpos, PANDA_TABLE_DEFAULTS.home_qpos, atol=1e-3
                ):
                    raise RuntimeError(
                        f"Robot home qpos drifted from expected: got {_actual_qpos.tolist()}"
                    )

            z = self._get_table_top_z() + self.obj.body_half_length
            self.launch_corridors = []

            occupied_indices, layout_meta = sample_frontier_cells(
                self.np_random,
                self._placement_grid,
                num_cells=self.num_bottles,
                mode=self.layout_mode,
                dense_layout_prob=self.dense_layout_prob,
                mixed_layout_prob=self.mixed_layout_prob,
            )
            self.np_random.shuffle(occupied_indices)
            centers_xy = self._placement_grid.centers_xy[occupied_indices]
            positions_xy = jitter_positions(
                self.np_random,
                centers_xy,
                max_jitter=self.placement_jitter,
                lo_xy=self._placement_lo_xy,
                hi_xy=self._placement_hi_xy,
            )

            for idx, xy in enumerate(positions_xy):
                q = self._sample_bottle_quaternion()
                self.cylinders[idx].set_pose(sapien.Pose([xy[0], xy[1], z], q))

            self.initial_positions_xy_batched = self._bottle_positions_xy_batched()
            _, self.initial_positions_xy = self.get_bottle_positions_xy()
            self.scene_layout = {
                **layout_meta,
                "occupied_indices": occupied_indices.astype(np.int32),
                "positions_xy": positions_xy.astype(np.float32),
            }

    def evaluate(self):
        current_xy = self._bottle_positions_xy_batched()  # (N, B, 2)
        deltas = current_xy - self.initial_positions_xy_batched  # (N, B, 2)
        displacement = torch.linalg.norm(deltas, dim=-1)  # (N, B)
        max_displacement = displacement.max(dim=-1).values  # (N,)
        moved_bottles = (displacement > self.movement_success_threshold).sum(dim=-1)  # (N,)
        success = max_displacement > self.movement_success_threshold  # (N,)
        return {
            "success": success.to(dtype=torch.bool),
            "max_displacement": max_displacement.to(dtype=torch.float32),
            "moved_bottles": moved_bottles.to(dtype=torch.int32),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        for obj in self.cylinders:
            obs[f"{obj.name}_pose"] = obj.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)
