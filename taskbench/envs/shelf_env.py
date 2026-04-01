"""Cluttered shelf environment for ManiSkill3.

An enclosed shelf on legs sits in front of the Panda robot, open toward
the robot (-X face).  Inside: 19 blue cylinders and 1 red target.  The
robot must reach in from the front, push blue cylinders aside, and extract
the red one.  CPU-only, single-env.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sapien
import sapien.render
import torch


from taskbench.envs.base import TaskEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# Colors
BLUE_COLOR = [0.20, 0.40, 0.85, 1.0]
RED_COLOR = [0.90, 0.10, 0.10, 1.0]
WOOD_COLOR = [0.55, 0.35, 0.10, 1.0]

CYL_UPRIGHT_Q = [0.7071068, 0, 0.7071068, 0]  # rotate +X axis -> +Z


# ── Structured configs ──────────────────────────────────────────────

@dataclass
class ShelfGeometry:
    """Shelf enclosure dimensions (meters)."""
    front_x: float = 0.20       # front edge X (closest to robot)
    depth: float = 0.30         # depth along +X
    half_w: float = 0.25        # half-width along Y
    floor_z: float = 0.30       # bottom board height above ground
    thickness: float = 0.01     # board/wall thickness
    inner_h: float = 0.25       # interior height

    # Derived — computed in __post_init__
    back_x: float = field(init=False)
    center_x: float = field(init=False)
    surface_z: float = field(init=False)
    ceil_z: float = field(init=False)
    leg_height: float = field(init=False)

    def __post_init__(self):
        self.back_x = self.front_x + self.depth
        self.center_x = self.front_x + self.depth / 2
        self.surface_z = self.floor_z + self.thickness
        self.ceil_z = self.floor_z + 2 * self.thickness + self.inner_h
        self.leg_height = self.floor_z


@dataclass
class CylinderSpec:
    """Cylinder object dimensions."""
    radius: float = 0.018       # 3.6cm diameter
    half_length: float = 0.045  # 9cm tall


@dataclass
class ShelfTaskConfig:
    """Top-level structured config for the shelf task."""
    env_id: str = "ShelfEnv-v1"
    robot_uids: str = "panda"
    robot_base_pose: list[float] = field(default_factory=lambda: [-0.615, 0.0, 0.0])
    num_objects: int = 20
    shelf: ShelfGeometry = field(default_factory=ShelfGeometry)
    cylinder: CylinderSpec = field(default_factory=CylinderSpec)


@register_env("ShelfEnv-v1", max_episode_steps=300)
class ShelfEnv(TaskEnv):
    """Enclosed shelf with 19 blue cylinders and 1 red target.

    The shelf faces the robot (open toward -X).  The robot reaches in
    to find and extract the red cylinder.

    Args:
        num_objects: Total number of cylinders (default: 20).
        robot_uids: Robot to use (default: "panda").
    """

    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, robot_uids="panda", num_objects: int = 20,
                 shelf=None, cylinder=None, open_top: bool = False, **kwargs):
        self.shelf_geom = ShelfGeometry(**(shelf or {}))
        self.cyl_spec = CylinderSpec(**(cylinder or {}))
        self.num_objects = num_objects
        self.open_top = open_top
        self.target_idx = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=1,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Sim / camera configs
    # ------------------------------------------------------------------

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
        # Looking from behind the robot toward the shelf
        pose = sapien_utils.look_at(eye=[0.10, 0.0, 0.55], target=[0.55, 0.0, 0.42])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        g = self.shelf_geom
        mid_z = g.surface_z + g.inner_h / 2

        # Close front-right: looking into the shelf opening
        front_close = sapien_utils.look_at(
            [g.front_x - 0.25, -0.35, mid_z + 0.15],
            [g.center_x, 0.0, mid_z],
        )
        # Close front-left: opposite angle
        front_left = sapien_utils.look_at(
            [g.front_x - 0.20, 0.35, mid_z + 0.10],
            [g.center_x, 0.0, mid_z],
        )
        # Wide top-down: shows robot + shelf
        wide_top = sapien_utils.look_at(
            [g.front_x - 0.15, 0.001, g.ceil_z + 0.60],
            [g.front_x + 0.05, 0, g.surface_z],
        )
        return [
            CameraConfig("render_topdown", wide_top, 512, 512, 1.2, 0.01, 100),
        ]

    # ------------------------------------------------------------------
    # Collision geometry (for motion planner)
    # ------------------------------------------------------------------

    def get_collision_boxes(self):
        """Return shelf collision geometry as a list of (name, center, half_size).

        Each entry is a box: (str, [x,y,z], [hx,hy,hz]).
        Solvers can pass these to the motion planner as obstacles.
        """
        g = self.shelf_geom
        cx = g.center_x
        hw = g.half_w
        fz = g.floor_z
        ih = g.inner_h
        t = g.thickness
        d = g.depth

        boxes = [
            ("shelf_bottom", [cx, 0, fz], [d / 2, hw, t]),
            ("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t]),
            ("shelf_back", [g.back_x, 0, fz + t + ih / 2], [t, hw, ih / 2]),
            ("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
            ("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
        ]

        leg_r = 0.015
        lh = g.leg_height / 2
        for li, (dx, dy) in enumerate([(-d / 2 + 0.02, -hw + 0.02),
                                         (-d / 2 + 0.02, hw - 0.02),
                                         (d / 2 - 0.02, -hw + 0.02),
                                         (d / 2 - 0.02, hw - 0.02)]):
            boxes.append((f"leg_{li}", [cx + dx, dy, lh], [leg_r, leg_r, lh]))

        return boxes

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _build_shelf(self):
        """Build enclosed shelf on legs, open toward -X (facing robot)."""
        mat = sapien.render.RenderMaterial(base_color=WOOD_COLOR)
        parts = []

        g = self.shelf_geom
        cx = g.center_x
        hw = g.half_w
        fz = g.floor_z
        ih = g.inner_h
        t = g.thickness
        d = g.depth

        def _box(name, center, half_size, visual=True):
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=half_size)
            if visual:
                b.add_box_visual(half_size=half_size, material=mat)
            b.initial_pose = sapien.Pose(p=center)
            parts.append(b.build_static(name=name))

        # Bottom board
        _box("shelf_bottom", [cx, 0, fz], [d / 2, hw, t])
        # Top board — collision only (no visual so cameras can see inside)
        if not self.open_top:
            _box("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t], visual=False)
        # Back wall (+X side, far from robot)
        _box("shelf_back", [g.back_x, 0, fz + t + ih / 2], [t, hw, ih / 2])
        # Left wall (-Y)
        _box("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2])
        # Right wall (+Y)
        _box("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2])

        # 4 legs
        leg_r = 0.015  # leg radius approximated as thin box
        lh = g.leg_height / 2
        for li, (dx, dy) in enumerate([(-d / 2 + 0.02, -hw + 0.02),
                                         (-d / 2 + 0.02, hw - 0.02),
                                         (d / 2 - 0.02, -hw + 0.02),
                                         (d / 2 - 0.02, hw - 0.02)]):
            _box(f"leg_{li}", [cx + dx, dy, lh], [leg_r, leg_r, lh])

        return parts

    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        return {f"cyl_{i}": obj for i, obj in enumerate(self.shelf_objects)}

    def _load_scene(self, options: dict):
        build_ground(self.scene, altitude=0.0)
        self.shelf_parts = self._build_shelf()

        # Pick target before building so we can color it red
        self.shelf_objects = []
        self.target_object = None
        if self.num_objects > 0:
            r = self.cyl_spec.radius
            hl = self.cyl_spec.half_length
            self.target_idx = self.np_random.integers(0, self.num_objects)
            for i in range(self.num_objects):
                color = RED_COLOR if i == self.target_idx else BLUE_COLOR
                cyl_mat = sapien.render.RenderMaterial(base_color=color)
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_collision(radius=r, half_length=hl)
                builder.add_cylinder_visual(radius=r, half_length=hl, material=cyl_mat)
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0 + i * 0.1])
                self.shelf_objects.append(builder.build(name=f"cyl_{i}"))
            self.target_object = self.shelf_objects[self.target_idx]

    # ------------------------------------------------------------------
    # Episode init
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._reset_robot(env_idx)
        if self.num_objects == 0:
            return

        from mani_skill.utils.structs.pose import Pose as MSPose

        g = self.shelf_geom
        margin = self.cyl_spec.radius + 0.01
        x_lo = g.front_x + margin
        x_hi = g.back_x - margin
        y_lo = -g.half_w + margin
        y_hi = g.half_w - margin
        z = g.surface_z + self.cyl_spec.half_length
        min_dist = self.cyl_spec.radius * 2.5  # no overlap
        b = len(env_idx)

        # Track placed positions for overlap rejection (b, i, 2)
        placed = torch.zeros(b, 0, 2, device=self.device)

        for i, obj in enumerate(self.shelf_objects):
            # Rejection sample: generate candidates until no overlap
            pos = torch.zeros(b, 3, device=self.device)
            for _ in range(200):
                # Random (x, y) per env
                xy = torch.rand(b, 2, device=self.device)
                xy[:, 0] = xy[:, 0] * (x_hi - x_lo) + x_lo
                xy[:, 1] = xy[:, 1] * (y_hi - y_lo) + y_lo

                if placed.shape[1] == 0:
                    break  # first object, no overlap check needed

                # Check min distance to all previously placed objects
                # placed: (b, i, 2), xy: (b, 2) -> (b, 1, 2)
                diffs = placed - xy.unsqueeze(1)  # (b, i, 2)
                dists = torch.norm(diffs, dim=2)   # (b, i)
                min_dists = dists.min(dim=1).values  # (b,)
                if (min_dists > min_dist).all():
                    break

            pos[:, 0] = xy[:, 0]
            pos[:, 1] = xy[:, 1]
            pos[:, 2] = z
            placed = torch.cat([placed, xy.unsqueeze(1)], dim=1)

            obj.set_pose(MSPose.create_from_pq(p=pos, q=CYL_UPRIGHT_Q))

    # ------------------------------------------------------------------
    # Evaluation / obs / reward
    # ------------------------------------------------------------------

    def evaluate(self):
        if self.target_object is None:
            success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            target_z = self.target_object.pose.p[:, 2]  # (N,)
            success = target_z > self.shelf_geom.ceil_z + 0.05
        return {
            "success": success.to(dtype=torch.bool),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.shelf_objects:
            obj_poses = torch.cat(
                [obj.pose.p for obj in self.shelf_objects], dim=-1
            )
            obs["obj_poses"] = obj_poses
            obs["target_idx"] = torch.full(
                (self.num_envs, 1), self.target_idx, device=self.device, dtype=torch.float32
            )
            obs["target_pos"] = self.target_object.pose.p
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)


if __name__ == "__main__":
    import gymnasium as gym

    import taskbench.envs  # noqa: F401

    env = gym.make(
        "ShelfEnv-v1",
        num_envs=1,
        render_mode="human",
        robot_uids="panda",
        robot_base_pose=[-0.615, 0, 0],
    )
    obs, _ = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()
    env.close()
