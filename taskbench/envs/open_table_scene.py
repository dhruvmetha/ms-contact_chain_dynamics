"""Shared compact scene builder for open-table taskbench environments."""

from __future__ import annotations

import numpy as np
import sapien
import sapien.render

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import TableSceneBuilder

COMPACT_OPEN_TABLE_CENTER_XY = np.array([-0.15, 0.0], dtype=np.float32)
COMPACT_OPEN_TABLE_SIZE_XY = np.array([1.02, 0.72], dtype=np.float32)
COMPACT_OPEN_TABLE_HEIGHT = 0.78
COMPACT_OPEN_TABLE_TOP_THICKNESS = 0.045
COMPACT_OPEN_TABLE_LEG_THICKNESS = 0.055


class CompactOpenTableSceneBuilder(TableSceneBuilder):
    """Smaller open table with the same robot initialization as ManiSkill's default."""

    def build(self):
        size_x, size_y = COMPACT_OPEN_TABLE_SIZE_XY
        height = COMPACT_OPEN_TABLE_HEIGHT
        top_thickness = COMPACT_OPEN_TABLE_TOP_THICKNESS
        leg_thickness = COMPACT_OPEN_TABLE_LEG_THICKNESS
        center_x, center_y = COMPACT_OPEN_TABLE_CENTER_XY

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, height / 2.0]),
            half_size=[size_x / 2.0, size_y / 2.0, height / 2.0],
        )

        wood_mat = sapien.render.RenderMaterial(base_color=[0.71, 0.58, 0.43, 1.0])
        metal_mat = sapien.render.RenderMaterial(base_color=[0.20, 0.22, 0.24, 1.0])

        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, 0.0, height - top_thickness / 2.0]),
            half_size=[size_x / 2.0, size_y / 2.0, top_thickness / 2.0],
            material=wood_mat,
        )

        leg_height = max(0.02, height - top_thickness)
        leg_center_z = leg_height / 2.0
        leg_offset_x = size_x / 2.0 - leg_thickness / 2.0 - 0.06
        leg_offset_y = size_y / 2.0 - leg_thickness / 2.0 - 0.06
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                builder.add_box_visual(
                    pose=sapien.Pose(
                        p=[
                            sign_x * leg_offset_x,
                            sign_y * leg_offset_y,
                            leg_center_z,
                        ]
                    ),
                    half_size=[leg_thickness / 2.0, leg_thickness / 2.0, leg_height / 2.0],
                    material=metal_mat,
                )

        builder.initial_pose = sapien.Pose([center_x, center_y, -height])
        self.table = builder.build_kinematic(name="table-workspace")

        self.table_length = float(size_x)
        self.table_width = float(size_y)
        self.table_height = float(height)

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-self.table_height,
        )
        self.scene_objects = [self.table, self.ground]
