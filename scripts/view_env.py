#!/usr/bin/env python3
"""View any registered env — GUI window or saved video.

Usage:
    # GUI window (needs display) — includes interactive sliders for
    # robot base pose and shelf position
    uv run python scripts/view_env.py task=shelf
    uv run python scripts/view_env.py task=shelf task.num_objects=0

    # Save video
    uv run python scripts/view_env.py task=shelf video=shelf_scene.mp4
    uv run python scripts/view_env.py task=tabletop_retrieval video=videos/tabletop.mp4

    # Control duration
    uv run python scripts/view_env.py task=shelf video=out.mp4 steps=200
"""

import sys
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import sapien
import torch
from omegaconf import DictConfig, OmegaConf
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin

import taskbench.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose as MSPose
from mani_skill.utils.wrappers import RecordEpisode


def _pop_extra_args(cfg):
    """Pop view_env-specific keys that aren't part of the taskbench config."""
    OmegaConf.set_struct(cfg, False)
    video = cfg.pop("video", None)
    steps = cfg.pop("steps", 500)
    return video, int(steps)


# ── Interactive scene config plugin ──────────────────────────────────


class SceneConfigPlugin(Plugin):
    """Viewer plugin with sliders to adjust robot base pose and shelf position.

    Drag sliders to move the robot / shelf in real-time, then click
    "Print YAML" to copy the values back into your task config.
    """

    def __init__(self, inner_env):
        self.env = inner_env
        self.ui_window = None

        # Robot base pose
        rbp = list(inner_env._robot_base_pose)
        self.robot_x = float(rbp[0])
        self.robot_y = float(rbp[1])
        self.robot_z = float(rbp[2])
        self._robot_quat = (
            [float(v) for v in rbp[3:7]] if len(rbp) == 7 else [1.0, 0.0, 0.0, 0.0]
        )

        # Shelf (if present)
        self.has_shelf = hasattr(inner_env, "shelf_parts") and hasattr(
            inner_env, "shelf_geom"
        )
        if self.has_shelf:
            g = inner_env.shelf_geom
            self.shelf_front_x = float(g.front_x)
            self.shelf_floor_z = float(g.floor_z)
            self._orig_front_x = g.front_x
            self._orig_floor_z = g.floor_z
            self._orig_shelf_poses = [
                sapien.Pose(
                    p.pose.p.cpu().numpy().flatten(),
                    p.pose.q.cpu().numpy().flatten(),
                )
                for p in inner_env.shelf_parts
            ]

        self._robot_frame = None
        self._shelf_bbox = None
        self._shelf_frame = None

    def init(self, viewer):
        super().init(viewer)
        # Coordinate frame at robot base
        robot_pose = sapien.Pose(
            [self.robot_x, self.robot_y, self.robot_z], self._robot_quat
        )
        self._robot_frame = viewer.add_coordinate_frame(
            robot_pose, length=0.15, radius=0.005
        )
        # Shelf bounding box + frame at open face
        if self.has_shelf:
            bbox_pose, bbox_half = self._shelf_bbox_params()
            self._shelf_bbox = viewer.add_bounding_box(
                bbox_pose,
                bbox_half,
                color=np.array([1.0, 0.8, 0.0, 1.0]),
                line_width=2.0,
            )
            self._shelf_frame = viewer.add_coordinate_frame(
                sapien.Pose(
                    [
                        self.shelf_front_x,
                        0,
                        self.shelf_floor_z + self.env.shelf_geom.thickness,
                    ]
                ),
                length=0.10,
                radius=0.004,
            )

    # ── UI ────────────────────────────────────────────────────────

    def build(self):
        if self.ui_window is not None:
            return  # sliders are bound — only build once
        self.ui_window = (
            R.UIWindow().Label("Scene Config").Pos(10, 420).Size(350, 300)
        )
        self.ui_window.append(
            R.UISection()
            .Label("Robot Base Pose")
            .Expanded(True)
            .append(
                R.UISliderFloat()
                .Label("X##robot")
                .Min(-2.0)
                .Max(2.0)
                .Bind(self, "robot_x"),
                R.UISliderFloat()
                .Label("Y##robot")
                .Min(-2.0)
                .Max(2.0)
                .Bind(self, "robot_y"),
                R.UISliderFloat()
                .Label("Z##robot")
                .Min(-0.5)
                .Max(1.5)
                .Bind(self, "robot_z"),
            ),
        )
        if self.has_shelf:
            self.ui_window.append(
                R.UISection()
                .Label("Shelf Position")
                .Expanded(True)
                .append(
                    R.UISliderFloat()
                    .Label("front_x")
                    .Min(0.0)
                    .Max(2.0)
                    .Bind(self, "shelf_front_x"),
                    R.UISliderFloat()
                    .Label("floor_z")
                    .Min(0.0)
                    .Max(1.5)
                    .Bind(self, "shelf_floor_z"),
                ),
            )
        self.ui_window.append(
            R.UIButton().Label("Print YAML").Callback(self._print_yaml),
        )

    def get_ui_windows(self):
        self.build()
        return [self.ui_window] if self.ui_window else []

    # ── Per-frame update ─────────────────────────────────────────

    def before_render(self):
        self._apply_robot_pose()
        if self.has_shelf:
            self._apply_shelf_poses()

    def _apply_robot_pose(self):
        p = torch.tensor(
            [[self.robot_x, self.robot_y, self.robot_z]], dtype=torch.float32
        )
        q = torch.tensor([self._robot_quat], dtype=torch.float32)
        self.env.agent.robot.set_pose(MSPose.create_from_pq(p=p, q=q))
        if self._robot_frame is not None:
            self._robot_frame.set_position(
                [self.robot_x, self.robot_y, self.robot_z]
            )
            self._robot_frame.set_rotation(self._robot_quat)

    def _apply_shelf_poses(self):
        dx = self.shelf_front_x - self._orig_front_x
        dz = self.shelf_floor_z - self._orig_floor_z
        for part, orig_pose in zip(self.env.shelf_parts, self._orig_shelf_poses):
            new_p = [
                float(orig_pose.p[0]) + dx,
                float(orig_pose.p[1]),
                float(orig_pose.p[2]) + dz,
            ]
            part.pose = sapien.Pose(new_p, orig_pose.q)
        # Update overlays
        if self._shelf_bbox is not None:
            bbox_pose, bbox_half = self._shelf_bbox_params()
            self.viewer.update_bounding_box(self._shelf_bbox, bbox_pose, bbox_half)
        if self._shelf_frame is not None:
            self._shelf_frame.set_position(
                [
                    self.shelf_front_x,
                    0,
                    self.shelf_floor_z + self.env.shelf_geom.thickness,
                ]
            )

    def _shelf_bbox_params(self):
        g = self.env.shelf_geom
        cx = self.shelf_front_x + g.depth / 2
        surface_z = self.shelf_floor_z + g.thickness
        ceil_z = self.shelf_floor_z + 2 * g.thickness + g.inner_h
        center_z = (surface_z + ceil_z) / 2
        half_z = (ceil_z - surface_z) / 2
        pose = sapien.Pose([cx, 0, center_z])
        half_size = np.array([g.depth / 2, g.half_w, half_z])
        return pose, half_size

    # ── Print ────────────────────────────────────────────────────

    def _print_yaml(self, _):
        q = self._robot_quat
        has_quat = q != [1.0, 0.0, 0.0, 0.0]
        if has_quat:
            pose_str = (
                f"[{self.robot_x:.4f}, {self.robot_y:.4f}, {self.robot_z:.4f}, "
                f"{q[0]}, {q[1]}, {q[2]}, {q[3]}]"
            )
        else:
            pose_str = (
                f"[{self.robot_x:.4f}, {self.robot_y:.4f}, {self.robot_z:.4f}]"
            )
        print("\n# ─── Current scene config (paste into YAML) ───")
        print(f"  robot_base_pose: {pose_str}")
        if self.has_shelf:
            print(f"  shelf:")
            print(f"    front_x: {self.shelf_front_x:.4f}")
            print(f"    floor_z: {self.shelf_floor_z:.4f}")
        print("# ──────────────────────────────────────────────\n")


# ── Entry point ──────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    video, steps = _pop_extra_args(cfg)

    cfg.runtime.num_envs = 1
    cfg.runtime.control_mode = "pd_joint_delta_pos"

    kwargs = OmegaConf.to_container(cfg.task, resolve=True)
    env_id = kwargs.pop("env_id")

    if video:
        video_path = Path(video)
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=cfg.runtime.obs_mode,
            control_mode=cfg.runtime.control_mode,
            reward_mode=cfg.runtime.reward_mode,
            render_mode="rgb_array",
            sim_backend="cpu",
            **kwargs,
        )
        env = RecordEpisode(
            env,
            output_dir=str(video_path.parent or "."),
            save_trajectory=False,
            save_video=True,
            save_on_reset=False,
            record_reward=False,
            video_fps=30,
        )
        env.reset(seed=cfg.seed)
        hold_action = env.action_space.sample() * 0
        for _ in range(steps):
            env.step(hold_action)
        env.flush_video()
        # RecordEpisode names files automatically — rename to requested name
        generated = sorted(Path(video_path.parent or ".").glob("*.mp4"))
        if generated:
            latest = generated[-1]
            latest.rename(video_path)
            print(f"Saved {video_path} ({steps} steps)")
    else:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=cfg.runtime.obs_mode,
            control_mode=cfg.runtime.control_mode,
            reward_mode=cfg.runtime.reward_mode,
            render_mode="human",
            sim_backend="cpu",
            **kwargs,
        )
        env.reset(seed=cfg.seed)
        hold_action = env.action_space.sample() * 0
        # First render creates the viewer
        viewer = env.render()
        # Inject interactive config plugin (only for TaskEnv subclasses)
        inner = env.unwrapped
        if hasattr(inner, "_robot_base_pose"):
            plugin = SceneConfigPlugin(inner)
            plugin.init(viewer)
            viewer.plugins.append(plugin)
        print(f"Viewing {env_id} — close the window to exit.")
        print("  Use the 'Scene Config' panel to adjust positions.")
        print("  Click 'Print YAML' to get values for your config.")
        try:
            while True:
                env.step(hold_action)
                env.render()
        except KeyboardInterrupt:
            pass

    env.close()


if __name__ == "__main__":
    main()
