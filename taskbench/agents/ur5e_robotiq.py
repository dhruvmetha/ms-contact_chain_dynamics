"""UR5e + Robotiq 2F-140 agent for ManiSkill3 (URDF-based).

Uses the clean ros-industrial URDF for UR5e + Robotiq 2F-140 gripper.
The same URDF is used by both ManiSkill (simulation) and cuRobo (planning),
eliminating any frame mismatch between planner and simulator.

The Robotiq 2F-140 parallel-jaw linkage uses one actuated joint
(finger_joint) with 5 mimic joints, following ManiSkill's XArm6Robotiq
pattern: active mimic pair + passive inner joints.

Controller gains are from the UR5e ros-industrial URDF:
  - Shoulder/elbow joints (1-3): kp=2000, kd=400, forcerange=150 Nm
  - Wrist joints (4-6):          kp=500,  kd=100, forcerange=28  Nm
"""

from copy import deepcopy

import numpy as np
import sapien
import sapien.core as sapien_core
import torch

import pathlib

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

from taskbench.skills.robot_config import RobotConfig

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


@register_agent()
class UR5eRobotiq(BaseAgent):
    uid = "ur5e_robotiq"
    urdf_path = str(_DATA_DIR / "robots/ur5e_robotiq_2f_85/ur5e_robotiq_2f_85.urdf")
    fix_root_link = True

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=5.0, dynamic_friction=5.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    # -- Taskbench metadata (single source of truth for skill/planning) --
    taskbench_config = RobotConfig(
        move_group="grasp_frame",
        finger_length=0.05,
        gripper_link_names=frozenset({
            "robotiq_arg2f_base_link",
            "left_outer_knuckle", "left_outer_finger",
            "left_inner_finger", "left_inner_finger_pad",
            "left_inner_knuckle",
            "right_outer_knuckle", "right_outer_finger",
            "right_inner_finger", "right_inner_finger_pad",
            "right_inner_knuckle",
        }),
        gripper_open=0.0,
        gripper_closed=0.81,
    )
    table_scene_base_pose = sapien.Pose([-0.56, 0, 0], [0, 0, 0, 1])

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0,     # arm: upright
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            # gripper: open
            ]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # UR5e actuator gains
    # Shoulder/elbow: kp=2000, kd=400, forcerange=150 Nm
    # Wrist joints:   kp=500,  kd=100, forcerange=28 Nm
    arm_stiffness = (2000, 2000, 2000, 500, 500, 500)
    arm_damping = (400, 400, 400, 100, 100, 100)
    arm_force_limit = (150, 150, 150, 28, 28, 28)
    arm_friction = 0.0

    gripper_stiffness = 1e5
    gripper_damping = 2000
    gripper_force_limit = 20
    gripper_friction = 1
    ee_link_name = "grasp_frame"

    @property
    def _controller_configs(self):
        # ------------------------------------------------------------------ #
        # Arm controllers
        # ------------------------------------------------------------------ #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # EE controllers
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
        )
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # ------------------------------------------------------------------ #
        # Gripper controllers (Robotiq 2F-140, ros-industrial linkage)
        # ------------------------------------------------------------------ #
        # Active mimic pair: left_outer_knuckle drives, right_outer_knuckle follows
        finger_joint_names = [
            "left_outer_knuckle_joint",
            "right_outer_knuckle_joint",
        ]
        mimic_config = dict(
            right_outer_knuckle_joint=dict(
                joint="left_outer_knuckle_joint", multiplier=1.0, offset=0.0
            ),
        )
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            finger_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
            mimic=mimic_config,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.15,
            upper=0.15,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
            mimic=mimic_config,
        )

        # Passive inner joints (driven by linkage kinematics)
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=5.0,
        )

        # ------------------------------------------------------------------ #
        # Combined controller configs
        # ------------------------------------------------------------------ #
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
        )

        return deepcopy(controller_configs)

    def _after_loading_articulation(self):
        # Create drive constraints for the Robotiq 4-bar linkage.
        # Without these, the passive inner finger joints free-float.
        # Drive poses from ManiSkill's FloatingRobotiq2F85Gripper agent.
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        # Right finger drive
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()
        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)
        right_drive.set_drive_property_x(stiffness=5e3, damping=200)
        right_drive.set_drive_property_y(stiffness=5e3, damping=200)
        right_drive.set_drive_property_z(stiffness=5e3, damping=200)

        # Left finger drive
        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()
        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)
        left_drive.set_drive_property_x(stiffness=5e3, damping=200)
        left_drive.set_drive_property_y(stiffness=5e3, damping=200)
        left_drive.set_drive_property_z(stiffness=5e3, damping=200)

        # Disable self-collisions between gripper links
        gripper_links = [
            "robotiq_arg2f_base_link",
            "left_outer_knuckle", "left_outer_finger",
            "left_inner_finger", "left_inner_finger_pad",
            "left_inner_knuckle",
            "right_outer_knuckle", "right_outer_finger",
            "right_inner_finger", "right_inner_finger_pad",
            "right_inner_knuckle",
            "tool0",
            "wrist_3_link",
            "wrist_2_link",
        ]
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
