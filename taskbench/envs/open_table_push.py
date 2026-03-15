"""Open-table push environment for quick offline push testing.

An open table sits in front of the Panda arm. A short row of upright
objects is placed on the tabletop and can be pushed along any planar
heading with either a vertical or horizontal wrist pose.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.quaternions import qinverse

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

from taskbench.envs.base import TaskEnv
from taskbench.envs.open_table_scene import (
    COMPACT_OPEN_TABLE_CENTER_XY,
    CompactOpenTableSceneBuilder,
)
from taskbench.skills.motion import make_linear_push_plan, tcp_height_for_table_clearance

TARGET_COLOR = [0.90, 0.15, 0.15, 1.0]
OBSTACLE_COLOR = [0.20, 0.40, 0.85, 1.0]
CYLINDER_UPRIGHT_Q = [0.7071068, 0.0, 0.7071068, 0.0]
BOTTLE_UPRIGHT_Q = [0.7071068, 0.0, -0.7071068, 0.0]
YCB_MUSTARD_BOTTLE_ID = "006_mustard_bottle"


@lru_cache(maxsize=None)
def _load_ycb_metadata(model_id: str) -> dict:
    metadata_path = Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    model_db = load_json(metadata_path)
    return model_db[model_id]


@register_env("OpenTablePush-v1", max_episode_steps=320, asset_download_ids=["ycb"])
class OpenTablePushEnv(TaskEnv):
    """Open table with a line of cylinders for deterministic push tests."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Union[Panda]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        num_cylinders: int = 3,
        object_kind: str = "bottle",
        object_density: float = 1800.0,
        cylinder_radius: float = 0.018,
        cylinder_half_length: float = 0.045,
        bottle_body_radius: float = 0.020,
        bottle_body_half_length: float = 0.050,
        bottle_neck_radius: float = 0.009,
        bottle_neck_half_length: float = 0.016,
        bottle_neck_offset: float = 0.046,
        bottle_neck_density_scale: float = 0.5,
        bottle_ballast_radius: float = 0.018,
        bottle_ballast_half_length: float = 0.010,
        bottle_ballast_offset: float = 0.032,
        bottle_ballast_density_scale: float = 4.0,
        bottle_visual_style: str = "ycb_mustard",
        ycb_bottle_model_id: str = YCB_MUSTARD_BOTTLE_ID,
        push_axis: str = "y",
        push_angle_deg: float | None = None,
        wrist_orientation: str = "vertical",
        tool_spin_deg: float = 0.0,
        approach_axis_x: float | None = None,
        approach_axis_y: float | None = None,
        approach_axis_z: float | None = None,
        staging_wrist_orientation: str | None = None,
        staging_tool_spin_deg: float | None = None,
        staging_approach_axis_x: float | None = None,
        staging_approach_axis_y: float | None = None,
        staging_approach_axis_z: float | None = None,
        use_staging: bool = True,
        row_x: float = 0.20,
        row_y: float = 0.0,
        row_start_x: float = 0.16,
        row_start_y: float = -0.03,
        row_origin_x: float | None = None,
        row_origin_y: float | None = None,
        row_spacing: float = 0.050,
        success_displacement: float = 0.04,
        table_clearance: float = 0.0,
        push_height_offset: float = 0.0,
        approach_gap: float = 0.08,
        end_margin: float = 0.05,
        staging_height: float = 0.14,
        staging_backoff: float = 0.0,
        **kwargs,
    ):
        if push_axis not in {"x", "y"}:
            raise ValueError(f"Unsupported push_axis={push_axis!r}; expected 'x' or 'y'")
        if object_kind not in {"cylinder", "bottle"}:
            raise ValueError(
                f"Unsupported object_kind={object_kind!r}; expected 'cylinder' or 'bottle'"
            )
        if wrist_orientation not in {"vertical", "horizontal"}:
            options = "horizontal, vertical"
            raise ValueError(
                f"Unsupported wrist_orientation={wrist_orientation!r}; expected one of {options}"
            )
        if staging_wrist_orientation is not None and staging_wrist_orientation not in {
            "vertical",
            "horizontal",
        }:
            options = "horizontal, vertical"
            raise ValueError(
                "Unsupported staging_wrist_orientation="
                f"{staging_wrist_orientation!r}; expected one of {options}"
            )
        self.num_cylinders = num_cylinders
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_kind = object_kind
        self.object_density = float(object_density)
        self.cylinder_radius = cylinder_radius
        self.cylinder_half_length = cylinder_half_length
        self.bottle_body_radius = float(bottle_body_radius)
        self.bottle_body_half_length = float(bottle_body_half_length)
        self.bottle_neck_radius = float(bottle_neck_radius)
        self.bottle_neck_half_length = float(bottle_neck_half_length)
        self.bottle_neck_offset = float(bottle_neck_offset)
        self.bottle_neck_density_scale = float(bottle_neck_density_scale)
        self.bottle_ballast_radius = float(bottle_ballast_radius)
        self.bottle_ballast_half_length = float(bottle_ballast_half_length)
        self.bottle_ballast_offset = float(bottle_ballast_offset)
        self.bottle_ballast_density_scale = float(bottle_ballast_density_scale)
        self.bottle_visual_style = str(bottle_visual_style)
        self.ycb_bottle_model_id = str(ycb_bottle_model_id)
        self.push_axis = push_axis
        if push_angle_deg is None:
            push_angle_deg = 0.0 if push_axis == "x" else 90.0
        self.push_angle_deg = float(push_angle_deg)
        self.wrist_orientation = wrist_orientation
        self.tool_spin_deg = float(tool_spin_deg)
        if any(v is not None for v in (approach_axis_x, approach_axis_y, approach_axis_z)):
            if None in (approach_axis_x, approach_axis_y, approach_axis_z):
                raise ValueError(
                    "approach_axis_x, approach_axis_y, and approach_axis_z must all be set together"
                )
            self.approach_axis = np.array(
                [approach_axis_x, approach_axis_y, approach_axis_z],
                dtype=np.float32,
            )
        else:
            self.approach_axis = None
        if any(
            v is not None
            for v in (
                staging_approach_axis_x,
                staging_approach_axis_y,
                staging_approach_axis_z,
            )
        ):
            if None in (
                staging_approach_axis_x,
                staging_approach_axis_y,
                staging_approach_axis_z,
            ):
                raise ValueError(
                    "staging_approach_axis_x, staging_approach_axis_y, and "
                    "staging_approach_axis_z must all be set together"
                )
            self.staging_approach_axis = np.array(
                [
                    staging_approach_axis_x,
                    staging_approach_axis_y,
                    staging_approach_axis_z,
                ],
                dtype=np.float32,
            )
        else:
            self.staging_approach_axis = None
        self.staging_wrist_orientation = staging_wrist_orientation
        self.staging_tool_spin_deg = (
            None if staging_tool_spin_deg is None else float(staging_tool_spin_deg)
        )
        self.use_staging = use_staging
        self.row_x = row_x
        self.row_y = row_y
        self.row_start_x = row_start_x
        self.row_start_y = row_start_y
        if row_origin_x is None:
            row_origin_x = row_start_x if push_axis == "x" else row_x
        if row_origin_y is None:
            row_origin_y = row_y if push_axis == "x" else row_start_y
        self.row_origin_x = float(row_origin_x)
        self.row_origin_y = float(row_origin_y)
        self.row_spacing = row_spacing
        self.success_displacement = success_displacement
        self.table_clearance = float(table_clearance)
        self.push_height_offset = float(push_height_offset)
        self.approach_gap = approach_gap
        self.end_margin = end_margin
        self.staging_height = staging_height
        self.staging_backoff = float(staging_backoff)
        self.target_idx = num_cylinders - 1
        angle_rad = np.deg2rad(self.push_angle_deg)
        self.push_direction_xy = np.array(
            [np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32
        )
        self.row_origin_xy = np.array(
            [self.row_origin_x, self.row_origin_y], dtype=np.float32
        )
        self.initial_target_xy = (
            self.row_origin_xy + self.push_direction_xy * row_spacing * self.target_idx
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=20,
                solver_velocity_iterations=5,
            )
        )

    @property
    def _default_sensor_configs(self):
        center_x, center_y = COMPACT_OPEN_TABLE_CENTER_XY
        pose = sapien_utils.look_at(
            eye=[float(center_x), float(center_y) + 1e-3, 1.15],
            target=[float(center_x), float(center_y), 0.0],
        )
        return [CameraConfig("base_camera", pose, 128, 128, 1.0, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        center_x, center_y = COMPACT_OPEN_TABLE_CENTER_XY
        pose = sapien_utils.look_at(
            eye=[float(center_x), float(center_y) + 1e-3, 1.25],
            target=[float(center_x), float(center_y), 0.0],
        )
        return CameraConfig("render_camera", pose, 1024, 1024, 0.95, 0.01, 100)

    def _build_cylinder(self, idx: int):
        color = TARGET_COLOR if idx == self.target_idx else OBSTACLE_COLOR
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color)
        builder.add_cylinder_collision(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            density=self.object_density,
        )
        builder.add_cylinder_visual(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            material=mat,
        )
        builder.initial_pose = sapien.Pose([0, 0, 1.0 + idx * 0.1])
        return builder.build(name=f"cyl_{idx}")

    def _build_bottle_actor(self, idx: int, color) -> object:
        builder = self.scene.create_actor_builder()
        self._add_bottle_collision_geometry(builder)
        self._add_bottle_visual_geometry(
            builder, sapien.render.RenderMaterial(base_color=color)
        )
        builder.initial_pose = sapien.Pose([0, 0, 1.0 + idx * 0.1])
        return builder.build(name=f"bottle_{idx}")

    def _add_bottle_collision_geometry(self, builder) -> None:
        builder.add_cylinder_collision(
            radius=self.bottle_body_radius,
            half_length=self.bottle_body_half_length,
            density=self.object_density,
        )
        neck_pose = sapien.Pose([self.bottle_neck_offset, 0, 0])
        builder.add_cylinder_collision(
            pose=neck_pose,
            radius=self.bottle_neck_radius,
            half_length=self.bottle_neck_half_length,
            density=self.object_density * self.bottle_neck_density_scale,
        )
        ballast_pose = sapien.Pose([-self.bottle_ballast_offset, 0, 0])
        builder.add_cylinder_collision(
            pose=ballast_pose,
            radius=self.bottle_ballast_radius,
            half_length=self.bottle_ballast_half_length,
            density=self.object_density * self.bottle_ballast_density_scale,
        )

    def _add_bottle_visual_geometry(self, builder, material) -> None:
        if self.bottle_visual_style == "primitive":
            builder.add_cylinder_visual(
                radius=self.bottle_body_radius,
                half_length=self.bottle_body_half_length,
                material=material,
            )
            neck_pose = sapien.Pose([self.bottle_neck_offset, 0, 0])
            builder.add_cylinder_visual(
                pose=neck_pose,
                radius=self.bottle_neck_radius,
                half_length=self.bottle_neck_half_length,
                material=material,
            )
            return
        if self.bottle_visual_style != "ycb_mustard":
            raise ValueError(
                f"Unsupported bottle_visual_style={self.bottle_visual_style!r}"
            )

        scale = self._get_ycb_bottle_visual_scale(self.ycb_bottle_model_id)
        mesh_pose = self._get_ycb_bottle_visual_pose(
            self.ycb_bottle_model_id, scale=scale
        )
        mesh_path = (
            Path(ASSET_DIR)
            / "assets"
            / "mani_skill2_ycb"
            / "models"
            / self.ycb_bottle_model_id
            / "textured.obj"
        )
        builder.add_visual_from_file(
            filename=str(mesh_path),
            pose=mesh_pose,
            scale=[scale] * 3,
            material=material,
        )

    def _get_visual_footprint_radius(self) -> float:
        radius = float(self.bottle_body_radius)
        if self.bottle_visual_style != "ycb_mustard":
            return radius
        meta = _load_ycb_metadata(self.ycb_bottle_model_id)
        bbox = meta["bbox"]
        half_extent_x = max(
            abs(float(bbox["min"][0])),
            abs(float(bbox["max"][0])),
        )
        half_extent_y = max(
            abs(float(bbox["min"][1])),
            abs(float(bbox["max"][1])),
        )
        scale = self._get_ycb_bottle_visual_scale(self.ycb_bottle_model_id)
        circumscribed_radius = scale * float(np.hypot(half_extent_x, half_extent_y))
        return max(radius, circumscribed_radius)

    def _get_ycb_bottle_visual_scale(self, model_id: str) -> float:
        meta = _load_ycb_metadata(model_id)
        bbox = meta["bbox"]
        half_extent_xy = max(
            abs(float(bbox["min"][0])),
            abs(float(bbox["max"][0])),
            abs(float(bbox["min"][1])),
            abs(float(bbox["max"][1])),
        )
        if half_extent_xy <= 0:
            raise ValueError(f"Invalid YCB bottle metadata for {model_id!r}")
        return float(self.bottle_body_radius / half_extent_xy)

    def _get_ycb_bottle_visual_pose(self, model_id: str, *, scale: float) -> sapien.Pose:
        meta = _load_ycb_metadata(model_id)
        bbox = meta["bbox"]
        bottom_z = float(bbox["min"][2]) * scale
        world_z_offset = -self.bottle_body_half_length - bottom_z
        local_x_offset = world_z_offset
        return sapien.Pose(
            p=[local_x_offset, 0.0, 0.0],
            q=qinverse(BOTTLE_UPRIGHT_Q),
        )

    def _build_bottle(self, idx: int):
        color = TARGET_COLOR if idx == self.target_idx else OBSTACLE_COLOR
        return self._build_bottle_actor(idx, color)

    def _build_object(self, idx: int):
        if self.object_kind == "bottle":
            return self._build_bottle(idx)
        return self._build_cylinder(idx)

    def _load_scene(self, options: dict):
        self.table_scene = CompactOpenTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cylinders = [self._build_object(i) for i in range(self.num_cylinders)]
        self.target_object = self.cylinders[self.target_idx]

    def get_objects(self) -> dict[str, object]:
        return {f"cyl_{i}": obj for i, obj in enumerate(self.cylinders)}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            if self.object_kind == "bottle":
                z = self.bottle_body_half_length
            else:
                z = self.cylinder_half_length
            for i, obj in enumerate(self.cylinders):
                xy = self.row_origin_xy + self.push_direction_xy * (i * self.row_spacing)
                upright_q = (
                    BOTTLE_UPRIGHT_Q if self.object_kind == "bottle" else CYLINDER_UPRIGHT_Q
                )
                obj.set_pose(sapien.Pose([xy[0], xy[1], z], upright_q))
            self.initial_target_xy = (
                self.target_object.pose.p[0, :2].detach().cpu().numpy().astype(np.float32)
            )

    def get_push_plan(self):
        """Return a deterministic push plan for the configured push heading.

        With ``use_staging=True`` the solver should stage at ``staging_pose``
        first, descend to ``approach_pose``, then sweep to ``push_pose``.
        """
        z = tcp_height_for_table_clearance(
            self.agent,
            self.push_direction_xy,
            wrist_orientation=self.wrist_orientation,
            tool_spin_deg=self.tool_spin_deg,
            approach_axis=self.approach_axis,
            table_z=0.0,
            table_clearance=self.table_clearance,
        ) + self.push_height_offset
        push_distance = (self.num_cylinders - 1) * self.row_spacing + self.end_margin
        contact_position = np.array(
            [self.row_origin_xy[0], self.row_origin_xy[1], z], dtype=np.float32
        )
        hover_height = self.staging_height if self.use_staging else None
        staging_height = self.staging_height if self.use_staging else None
        plan = make_linear_push_plan(
            self.agent,
            contact_position=contact_position,
            push_distance=push_distance,
            push_direction=self.push_direction_xy,
            wrist_orientation=self.wrist_orientation,
            tool_spin_deg=self.tool_spin_deg,
            approach_axis=self.approach_axis,
            approach_gap=self.approach_gap,
            hover_height=hover_height,
            staging_height=staging_height,
            staging_backoff=self.staging_backoff,
            staging_wrist_orientation=self.staging_wrist_orientation,
            staging_tool_spin_deg=self.staging_tool_spin_deg,
            staging_approach_axis=self.staging_approach_axis,
        )
        return plan.as_skill_kwargs()

    def get_push_poses(self):
        """Return the approach and sweep poses for compatibility."""
        plan = self.get_push_plan()
        return plan["approach_pose"], plan["push_pose"]

    def evaluate(self):
        target_xy = self.target_object.pose.p[0, :2].detach().cpu().numpy().astype(np.float32)
        delta_xy = target_xy - self.initial_target_xy
        target_displacement = float(np.dot(delta_xy, self.push_direction_xy))
        success = target_displacement > self.success_displacement
        info = {
            "success": torch.tensor([success], device=self.device, dtype=torch.bool),
            "target_displacement": torch.tensor(
                [target_displacement],
                device=self.device,
                dtype=torch.float32,
            ),
        }
        info["target_displacement_x"] = torch.tensor(
            [float(delta_xy[0])], device=self.device, dtype=torch.float32
        )
        info["target_displacement_y"] = torch.tensor(
            [float(delta_xy[1])], device=self.device, dtype=torch.float32
        )
        return info

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        for i, obj in enumerate(self.cylinders):
            obs[f"cyl_{i}_pose"] = obj.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)
