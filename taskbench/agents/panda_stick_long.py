"""Panda arm with a long stick end-effector (no gripper).

Bigger stick than ManiSkill's built-in PandaStick:
  - radius: 15mm (vs 8mm)
  - length: 20cm (vs 10cm)
  - TCP at stick tip: 25cm from hand (vs 15cm)

Same 7-DOF arm, same controllers as PandaStick.
"""

from copy import deepcopy
from typing import Tuple
import pathlib

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


@register_agent()
class PandaStickLong(BaseAgent):
    uid = "panda_stick_long"
    urdf_path = str(_DATA_DIR / "robots/panda_stick_long/panda_stick_long.urdf")
    urdf_config = dict()
    fix_root_link = True

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]

    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, use_delta=True,
        )
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1, pos_upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1, pos_upper=0.1,
            rot_lower=-0.1, rot_upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None, pos_upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path,
            use_delta=False, normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
            pd_ee_pose=dict(arm=arm_pd_ee_pose),
        )
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :7]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
