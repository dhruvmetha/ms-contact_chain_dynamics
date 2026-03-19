"""Tabletop target retrieval — retrieve a target object from clutter on a table.

Objects are placed in a row (or loaded from a scene file). The last object
is the target (red). Success = target lifted above the table surface.
"""

from dataclasses import dataclass

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env

from taskbench.envs.bottle_builder import BOTTLE_UPRIGHT_Q
from taskbench.envs.open_table_env import CYLINDER_UPRIGHT_Q, OpenTableEnv


@dataclass
class RowLayout:
    """Spatial placement of the object row on the table."""
    origin_x: float
    origin_y: float
    spacing: float = 0.050
    angle_deg: float = 90.0


@register_env("TabletopRetrieval-v1", max_episode_steps=320, asset_download_ids=["ycb"])
class TabletopRetrievalEnv(OpenTableEnv):
    """Open table with a row of objects — retrieve the target.

    Objects are arranged in a row on the table. The last object is the
    target (red). The solver must retrieve it — how it does so (push
    obstacles aside, pick directly, rearrange) is the solver's decision.

    Success: target lifted above ``table_top_z + lift_threshold``.
    """

    def __init__(
        self,
        *args,
        # Row layout
        row_origin_x: float = 0.16,
        row_origin_y: float = -0.06,
        row_spacing: float = 0.050,
        row_angle_deg: float = 90.0,
        # Success criterion
        lift_threshold: float = 0.05,
        # Scene file
        scene_file: str | None = None,
        **kwargs,
    ):
        # Scene file (optional)
        self._scene_file = scene_file
        self._scene_data = None
        self._scene_index = 0
        if scene_file is not None:
            import json
            with open(scene_file) as f:
                self._scene_data = json.load(f)["scenes"]

        self.lift_threshold = float(lift_threshold)

        self.row = RowLayout(
            origin_x=float(row_origin_x),
            origin_y=float(row_origin_y),
            spacing=row_spacing,
            angle_deg=float(row_angle_deg),
        )

        angle_rad = np.deg2rad(self.row.angle_deg)
        self.row_direction_xy = np.array(
            [np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32
        )
        self.row_origin_xy = np.array(
            [self.row.origin_x, self.row.origin_y], dtype=np.float32
        )

        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._reset_robot(env_idx)

            if self._scene_data is not None:
                self._place_from_scene_data()
            else:
                self._place_row_layout()

    def _place_row_layout(self):
        """Place objects in a deterministic row."""
        table_top_z = self._get_table_top_z()
        if self.obj.kind == "bottle":
            z = table_top_z + self.obj.body_half_length
        else:
            z = table_top_z + self.obj.cylinder_half_length
        upright_q = (
            BOTTLE_UPRIGHT_Q if self.obj.kind == "bottle" else CYLINDER_UPRIGHT_Q
        )
        for i, obj in enumerate(self.cylinders):
            xy = self.row_origin_xy + self.row_direction_xy * (
                i * self.row.spacing
            )
            obj.set_pose(sapien.Pose([xy[0], xy[1], z], upright_q))

    def _place_from_scene_data(self):
        """Place objects from a precomputed scene file."""
        scene = self._scene_data[self._scene_index % len(self._scene_data)]
        self._scene_index += 1
        positions = scene["positions"]
        quaternions = scene.get("quaternions")
        upright_q = (
            BOTTLE_UPRIGHT_Q if self.obj.kind == "bottle" else CYLINDER_UPRIGHT_Q
        )
        for i, obj in enumerate(self.cylinders):
            p = positions[i] if i < len(positions) else [0, 0, 1.0]
            q = quaternions[i] if quaternions and i < len(quaternions) else upright_q
            obj.set_pose(sapien.Pose(p, q))

    def get_scene_info(self):
        """Return scene geometry for solver planning."""
        return {
            "target_name": f"cyl_{self.target_idx}",
            "row_direction_xy": self.row_direction_xy.copy(),
            "row_origin_xy": self.row_origin_xy.copy(),
            "num_objects": self.num_cylinders,
            "row_spacing": self.row.spacing,
            "table_top_z": self._get_table_top_z(),
        }

    def evaluate(self):
        table_top_z = self._get_table_top_z()
        target_z = self.target_object.pose.p[:, 2]  # (N,)
        lifted = target_z > table_top_z + self.lift_threshold  # (N,) bool
        grasped = self.agent.is_grasping(self.target_object)  # (N,) bool

        return {
            "success": (lifted | grasped).to(dtype=torch.bool),
            "target_z": target_z.to(dtype=torch.float32),
            "target_lifted": lifted.to(dtype=torch.bool),
            "target_grasped": grasped.to(dtype=torch.bool),
        }
