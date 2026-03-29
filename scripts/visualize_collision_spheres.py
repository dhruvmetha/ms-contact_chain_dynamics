"""Visualize cuRobo collision spheres on the robot in ManiSkill.

Places semi-transparent spheres at each collision sphere position at
multiple joint configs. Renders from multiple angles.

Run on GPU (needs rendering):
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:10:00 --mem=32G \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/visualize_collision_spheres.py'
"""
import torch
import numpy as np
import gymnasium as gym
import os
import sapien
import sapien.render
from PIL import Image, ImageDraw
import yaml

import taskbench.envs
from taskbench.skills.curobo_motion import setup_curobo_planner, get_arm_joint_names
from curobo.types.robot import JointState as CuJointState
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv

os.makedirs("videos/collision_viz", exist_ok=True)
cuda = torch.device("cuda:0")

# ── Minimal env with close cameras ─────────────────────────────────────
@register_env("CollisionViz-v1", max_episode_steps=100)
class CollisionVizEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reconfiguration_freq=1, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        front = sapien_utils.look_at(eye=[0.15, -0.8, 0.5], target=[0.3, 0.0, 0.3])
        top = sapien_utils.look_at(eye=[0.3, 0.001, 1.5], target=[0.3, 0.0, 0.0])
        side = sapien_utils.look_at(eye=[1.0, 0.0, 0.5], target=[0.0, 0.0, 0.3])
        return [
            CameraConfig("front", front, 640, 640, 1.0, 0.01, 100),
            CameraConfig("top", top, 640, 640, 1.0, 0.01, 100),
            CameraConfig("side", side, 640, 640, 1.0, 0.01, 100),
        ]

    def _load_scene(self, options):
        from mani_skill.utils.building.ground import build_ground
        build_ground(self.scene, altitude=0.0)

    def _initialize_episode(self, env_idx, options):
        qpos = self.agent.keyframes["rest"].qpos
        self.agent.robot.set_qpos(qpos)

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}
    def _get_obs_extra(self, info):
        return {}
    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)


env = gym.make("CollisionViz-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq",
               render_mode="rgb_array", obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none")
env.reset(seed=42)
raw = env.unwrapped

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)
n_arm = len(get_arm_joint_names("ur5e_robotiq"))

# ── Read collision spheres from config ─────────────────────────────────
with open("configs/curobo/ur5e_robotiq_2f_85.yml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

collision_spheres = cfg["robot_cfg"]["kinematics"]["collision_spheres"]

# ── Visualize at multiple configs ──────────────────────────────────────
configs = {
    "upright": [0, -1.5708, 0, -1.5708, 0, 0],
    "reach_fwd": [0, -1.0, 0.5, -1.0, -1.5708, 0],
    "retract": [0, -2.2, 1.0, -1.383, -1.57, 0],
}

for config_name, joints in configs.items():
    print(f"\nConfig: {config_name}")

    # Set robot joints
    n_gripper = raw.agent.robot.get_qpos().shape[1] - n_arm
    full_qpos = torch.tensor([joints + [0] * n_gripper], dtype=torch.float32)
    raw.agent.robot.set_qpos(full_qpos)

    # Get cuRobo FK for all links to find sphere world positions
    j_t = torch.tensor([joints], dtype=torch.float32, device=cuda)
    state = CuJointState.from_position(j_t, joint_names=get_arm_joint_names("ur5e_robotiq"))

    # Get sphere positions in world frame via cuRobo
    kin_state = mg.kinematics.get_state(j_t)
    link_spheres = kin_state.link_spheres_tensor  # (1, n_spheres, 4) [x,y,z,r]
    spheres_np = link_spheres[0].cpu().numpy()  # (n_spheres, 4)

    print(f"  Total spheres from cuRobo: {spheres_np.shape[0]}")

    # Clear previous markers
    # (Can't easily delete actors in SAPIEN, so we create new ones each time)

    # Place visual spheres
    arm_mat = sapien.render.RenderMaterial(base_color=[1.0, 0.3, 0.3, 0.4])  # red, semi-transparent
    gripper_mat = sapien.render.RenderMaterial(base_color=[0.3, 0.3, 1.0, 0.3])  # blue, semi-transparent
    disabled_mat = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 0.2])  # gray, very transparent

    n_placed = 0
    for i in range(spheres_np.shape[0]):
        x, y, z, r = spheres_np[i]
        if r <= 0:
            # Disabled sphere — show faintly
            mat = disabled_mat
            r = 0.005  # tiny dot to show position
        elif r > 0.03:
            mat = arm_mat
        else:
            mat = gripper_mat

        b = raw.scene.create_actor_builder()
        b.add_sphere_visual(radius=float(r), material=mat)
        b.initial_pose = sapien.Pose(p=[float(x), float(y), float(z)])
        b.build_static(name=f"csphere_{config_name}_{i}")
        n_placed += 1

    print(f"  Placed {n_placed} visual spheres")

    # Step to render
    for _ in range(3):
        env.step(torch.zeros(1, raw.agent.action_space.shape[0]))

    # Render
    img = env.render()
    if img is not None:
        img_np = img[0].cpu().numpy() if hasattr(img, 'cpu') else img
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), f"{config_name}\n{n_placed} collision spheres\n"
                  f"Red=arm, Blue=gripper, Gray=disabled",
                  fill=(255, 255, 0))
        pil_img.save(f"videos/collision_viz/{config_name}.png")
        print(f"  Saved: videos/collision_viz/{config_name}.png")

env.close()
print("\nDone — check videos/collision_viz/")
