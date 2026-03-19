"""Shared base for open-table environments with bottle/cylinder actors.

Handles table scene creation, object building, camera setup, and sim config.
Subclasses define task-specific placement, evaluation, and success criteria.
"""

from typing import Any

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import SceneConfig, SimConfig

from taskbench.envs.base import TaskEnv
from taskbench.envs.bottle_builder import (
    BOTTLE_UPRIGHT_Q,
    ObjectGeometry,
    YCB_MUSTARD_BOTTLE_ID,
    build_bottle_actor,
)
from taskbench.envs.open_table_defaults import (
    BOTTLE_BALLAST_HALF_LENGTH_BASE, BOTTLE_BALLAST_OFFSET_BASE,
    BOTTLE_BALLAST_RADIUS_BASE, BOTTLE_BODY_HALF_LENGTH_BASE,
    BOTTLE_BODY_RADIUS_BASE, BOTTLE_NECK_HALF_LENGTH_BASE,
    BOTTLE_NECK_OFFSET_BASE, BOTTLE_NECK_RADIUS_BASE, BOTTLE_VISUAL_STYLE)
from taskbench.envs.open_table_scene import (COMPACT_OPEN_TABLE_CENTER_XY,
                                             CompactOpenTableSceneBuilder)

TARGET_COLOR = [0.90, 0.15, 0.15, 1.0]
OBSTACLE_COLOR = [0.20, 0.40, 0.85, 1.0]
CYLINDER_UPRIGHT_Q = [0.7071068, 0.0, 0.7071068, 0.0]


class OpenTableEnv(TaskEnv):
    """Base for open-table environments with N bottle/cylinder actors.

    Handles: table scene, object building, cameras, sim config.
    Does NOT define: object placement, evaluation, or task-specific state.
    """

    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        num_cylinders: int = 3,
        # Object geometry
        object_kind: str = "bottle",
        object_density: float = 1800.0,
        cylinder_radius: float = 0.018,
        cylinder_half_length: float = 0.045,
        bottle_body_radius: float = BOTTLE_BODY_RADIUS_BASE,
        bottle_body_half_length: float = BOTTLE_BODY_HALF_LENGTH_BASE,
        bottle_neck_radius: float = BOTTLE_NECK_RADIUS_BASE,
        bottle_neck_half_length: float = BOTTLE_NECK_HALF_LENGTH_BASE,
        bottle_neck_offset: float = BOTTLE_NECK_OFFSET_BASE,
        bottle_neck_density_scale: float = 0.5,
        bottle_ballast_radius: float = BOTTLE_BALLAST_RADIUS_BASE,
        bottle_ballast_half_length: float = BOTTLE_BALLAST_HALF_LENGTH_BASE,
        bottle_ballast_offset: float = BOTTLE_BALLAST_OFFSET_BASE,
        bottle_ballast_density_scale: float = 4.0,
        bottle_visual_style: str = BOTTLE_VISUAL_STYLE,
        ycb_bottle_model_id: str = YCB_MUSTARD_BOTTLE_ID,
        **kwargs,
    ):
        if object_kind not in {"cylinder", "bottle"}:
            raise ValueError(f"Unsupported object_kind={object_kind!r}")

        self.num_cylinders = num_cylinders
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_idx = num_cylinders - 1

        self.obj = ObjectGeometry(
            kind=object_kind,
            density=float(object_density),
            cylinder_radius=cylinder_radius,
            cylinder_half_length=cylinder_half_length,
            body_radius=float(bottle_body_radius),
            body_half_length=float(bottle_body_half_length),
            neck_radius=float(bottle_neck_radius),
            neck_half_length=float(bottle_neck_half_length),
            neck_offset=float(bottle_neck_offset),
            neck_density_scale=float(bottle_neck_density_scale),
            ballast_radius=float(bottle_ballast_radius),
            ballast_half_length=float(bottle_ballast_half_length),
            ballast_offset=float(bottle_ballast_offset),
            ballast_density_scale=float(bottle_ballast_density_scale),
            visual_style=str(bottle_visual_style),
            ycb_model_id=str(ycb_bottle_model_id),
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
            eye=[float(center_x) + 0.6, float(center_y) - 0.4, 0.45],
            target=[float(center_x) + 0.05, float(center_y), 0.0],
        )
        return CameraConfig("render_camera", pose, 1024, 1024, 0.95, 0.01, 100)

    def _build_cylinder(self, idx: int):
        color = TARGET_COLOR if idx == self.target_idx else OBSTACLE_COLOR
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color)
        builder.add_cylinder_collision(
            radius=self.obj.cylinder_radius,
            half_length=self.obj.cylinder_half_length,
            density=self.obj.density,
        )
        builder.add_cylinder_visual(
            radius=self.obj.cylinder_radius,
            half_length=self.obj.cylinder_half_length,
            material=mat,
        )
        builder.initial_pose = sapien.Pose([0, 0, 1.0 + idx * 0.1])
        return builder.build(name=f"cyl_{idx}")

    def _build_bottle(self, idx: int):
        color = TARGET_COLOR if idx == self.target_idx else OBSTACLE_COLOR
        return build_bottle_actor(self.scene, self.obj, idx, color)

    def _build_object(self, idx: int):
        if self.obj.kind == "bottle":
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

    def _get_table_top_z(self):
        return self.table_scene.table_top_z

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
