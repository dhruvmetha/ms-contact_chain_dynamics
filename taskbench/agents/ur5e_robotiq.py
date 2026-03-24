"""UR5e + Robotiq 2F-85 agent for ManiSkill3 (MJCF-based).

Uses official MuJoCo Menagerie models for the UR5e arm and Robotiq 2F-85
gripper. The MJCF is loaded by SAPIEN and a kinematic URDF is auto-exported
for mplib motion planning and EE pose controllers.

The Robotiq's 4-bar linkage is approximated via PDJointPosMimicController
with coupling ratios calibrated from MuJoCo simulation.

Controller gains are from MuJoCo Menagerie actuator definitions:
  - Shoulder/elbow joints (1-3): kp=2000, kd=400, forcerange=150 Nm
  - Wrist joints (4-6):          kp=500,  kd=100, forcerange=28  Nm
  - Gripper:                     forcerange ~20 N per joint for mimic stability
"""

import atexit
from copy import deepcopy
import os
import re
import tempfile

import numpy as np
import sapien.core as sapien
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
    mjcf_path = str(_DATA_DIR / "robots/ur5e_robotiq/ur5e_robotiq.xml")
    urdf_path = None  # auto-exported in _after_loading_articulation

    urdf_config = dict()

    # -- Taskbench metadata (single source of truth for skill/planning) --
    taskbench_config = RobotConfig(
        move_group="eef",
        finger_length=0.035,
        gripper_link_names=frozenset({
            "robotiq_base_mount", "robotiq_base",
            "right_driver", "right_coupler", "right_spring_link",
            "right_follower", "right_pad", "right_silicone_pad",
            "left_driver", "left_coupler", "left_spring_link",
            "left_follower", "left_pad", "left_silicone_pad",
        }),
        gripper_open=0.0,
        gripper_closed=-0.8,
    )
    table_scene_base_pose = sapien.Pose([-0.56, 0, 0], [0, 0, 0, 1])

    keyframes = dict(
        rest=Keyframe(
            # UR5e "elbow-up" home pose + gripper open (8 gripper joints)
            qpos=np.array([
                0,           # shoulder_pan_joint
                -1.5708,     # shoulder_lift_joint
                1.5708,      # elbow_joint
                -1.5708,     # wrist_1_joint
                -1.5708,     # wrist_2_joint
                np.pi,       # wrist_3_joint (aligned with build_grasp_pose)
                # 8 gripper joints (all zero = open)
                0, 0, 0, 0, 0, 0, 0, 0,
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

    # MuJoCo Menagerie UR5e actuator gains (per-joint tuples)
    # Shoulder/elbow: kp=2000, kd=400, forcerange=150 Nm
    # Wrist joints:   kp=500,  kd=100, forcerange=28 Nm
    arm_stiffness = (2000, 2000, 2000, 500, 500, 500)
    arm_damping = (400, 400, 400, 100, 100, 100)
    arm_force_limit = (150, 150, 150, 28, 28, 28)
    arm_friction = 0.0

    gripper_stiffness = 1e5
    gripper_damping = 2000
    # Menagerie 2F-85: 5N tendon; ~20N per-joint for mimic stability
    gripper_force_limit = 20
    gripper_friction = 1
    ee_link_name = "eef"

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

        # EE controllers (require urdf_path, set in _after_loading_articulation)
        if self.urdf_path is not None:
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
        # Gripper controllers (Robotiq 2F-85 Menagerie 4-bar linkage)
        # ------------------------------------------------------------------ #
        finger_joint_names = [
            "right_driver_joint",
            "right_spring_link_joint",
            "left_driver_joint",
            "left_spring_link_joint",
            "right_coupler_joint",
            "right_follower_joint",
            "left_coupler_joint",
            "left_follower_joint",
        ]
        mimic_config = dict(
            left_driver_joint=dict(
                joint="right_driver_joint", multiplier=1.0, offset=0.0
            ),
            right_spring_link_joint=dict(
                joint="right_driver_joint", multiplier=0.9, offset=0.0
            ),
            left_spring_link_joint=dict(
                joint="right_driver_joint", multiplier=0.9, offset=0.0
            ),
            right_coupler_joint=dict(
                joint="right_driver_joint", multiplier=0.0, offset=0.0
            ),
            left_coupler_joint=dict(
                joint="right_driver_joint", multiplier=0.0, offset=0.0
            ),
            right_follower_joint=dict(
                joint="right_driver_joint", multiplier=-0.76, offset=0.0
            ),
            left_follower_joint=dict(
                joint="right_driver_joint", multiplier=-0.76, offset=0.0
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

        # ------------------------------------------------------------------ #
        # Combined controller configs
        # ------------------------------------------------------------------ #
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper_active=finger_mimic_pd_joint_pos,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper_active=finger_mimic_pd_joint_pos,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper_active=finger_mimic_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper_active=finger_mimic_pd_joint_delta_pos,
            ),
        )

        # Add EE controllers if URDF is available
        if self.urdf_path is not None:
            controller_configs.update(dict(
                pd_ee_delta_pos=dict(
                    arm=arm_pd_ee_delta_pos,
                    gripper_active=finger_mimic_pd_joint_delta_pos,
                    ),
                pd_ee_delta_pose=dict(
                    arm=arm_pd_ee_delta_pose,
                    gripper_active=finger_mimic_pd_joint_pos,
                    ),
                pd_ee_pose=dict(
                    arm=arm_pd_ee_pose,
                    gripper_active=finger_mimic_pd_joint_pos,
                    ),
                pd_ee_target_delta_pos=dict(
                    arm=arm_pd_ee_target_delta_pos,
                    gripper_active=finger_mimic_pd_joint_delta_pos,
                    ),
                pd_ee_target_delta_pose=dict(
                    arm=arm_pd_ee_target_delta_pose,
                    gripper_active=finger_mimic_pd_joint_delta_pos,
                    ),
            ))

        return deepcopy(controller_configs)

    def _after_loading_articulation(self):
        # --- Auto-export kinematic URDF for mplib and EE controllers ---
        from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf

        urdf_str = export_kinematic_chain_urdf(
            self.robot._objs[0], force_fix_root=True
        )
        prefix = "scene-0-ur5e_robotiq_"
        urdf_str = urdf_str.replace(prefix, "")
        # Fix remaining empty-name fixed joints (name="")
        counter = [0]
        def _unique_fix(m):
            counter[0] += 1
            return f'joint name="fixed_{counter[0]}"'
        urdf_str = re.sub(r'joint name=""', _unique_fix, urdf_str)
        # Fix robot name too
        urdf_str = urdf_str.replace('robot name=""', 'robot name="ur5e_robotiq"')

        # --- Inject collision geometry from SAPIEN shapes ---
        # TODO: fix reference frame mismatch — disabled until debugged
        # urdf_str = self._inject_collision_geometry(urdf_str)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".urdf", prefix="ur5e_robotiq_", delete=False, mode="w"
        )
        tmp.write(urdf_str)
        tmp.flush()
        self.urdf_path = tmp.name
        self._urdf_tmp = tmp  # prevent GC

        # --- Generate a companion SRDF for mplib planning ---
        srdf_str = '<?xml version="1.0" ?>\n<robot name="ur5e_robotiq">\n'
        srdf_str += '  <group name="eef">\n'
        srdf_str += '    <joint name="virtual_joint"/>\n'
        for jn in self.arm_joint_names:
            srdf_str += f'    <joint name="{jn}"/>\n'
        srdf_str += '  </group>\n'
        srdf_str += '  <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="base_link"/>\n'
        # Disable gripper self-collisions for planning
        gripper_srdf_links = [
            "robotiq_base_mount", "robotiq_base",
            "right_driver", "right_coupler", "right_spring_link",
            "right_follower", "right_pad", "right_silicone_pad",
            "left_driver", "left_coupler", "left_spring_link",
            "left_follower", "left_pad", "left_silicone_pad",
        ]
        for i, l1 in enumerate(gripper_srdf_links):
            for l2 in gripper_srdf_links[i + 1:]:
                srdf_str += f'  <disable_collisions link1="{l1}" link2="{l2}" reason="Adjacent"/>\n'
        # Disable wrist-gripper collisions
        for wl in ["wrist_1_link", "wrist_2_link", "wrist_3_link"]:
            for gl in gripper_srdf_links:
                srdf_str += f'  <disable_collisions link1="{wl}" link2="{gl}" reason="Never"/>\n'
        srdf_str += '</robot>\n'
        srdf_path = self.urdf_path.replace(".urdf", ".srdf")
        with open(srdf_path, "w") as f:
            f.write(srdf_str)
        self._srdf_path = srdf_path

        def _cleanup_temp_files(urdf=tmp.name, srdf=srdf_path):
            for path in (urdf, srdf):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        atexit.register(_cleanup_temp_files)

        # --- Disable self-collisions between gripper links ---
        gripper_links = [
            "robotiq_base_mount",
            "robotiq_base",
            "right_driver",
            "right_coupler",
            "right_spring_link",
            "right_follower",
            "right_pad",
            "right_silicone_pad",
            "left_driver",
            "left_coupler",
            "left_spring_link",
            "left_follower",
            "left_pad",
            "left_silicone_pad",
            "wrist_3_link",
            "wrist_2_link",
        ]
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _inject_collision_geometry(self, urdf_str: str) -> str:
        """Add collision shapes from SAPIEN into the exported URDF.

        The auto-exported URDF has no collision geometry, so mplib treats
        the robot as a point.  This reads the actual PhysX collision shapes
        from each link and injects them as URDF <collision> elements.

        Uses string insertion (not XML re-serialization) to avoid breaking
        the URDF structure that mplib's parser depends on.

        Capsules are approximated as cylinders (URDF doesn't support capsules).
        """
        from transforms3d.euler import quat2euler

        for link in self.robot.get_links():
            name = link.get_name()
            collision_xml = ""
            for comp in link._objs[0].entity.components:
                if not hasattr(comp, "get_collision_shapes"):
                    continue
                for shape in comp.get_collision_shapes():
                    stype = type(shape).__name__
                    pose = shape.get_local_pose()
                    p = pose.p
                    rpy = quat2euler([pose.q[0], pose.q[1], pose.q[2], pose.q[3]])

                    if "Capsule" in stype or "Cylinder" in stype:
                        hl = shape.half_length if hasattr(shape, "half_length") else 0.01
                        geom_xml = f'<cylinder radius="{shape.radius}" length="{hl * 2}"/>'
                    elif "Box" in stype:
                        hs = shape.half_size
                        geom_xml = f'<box size="{hs[0]*2} {hs[1]*2} {hs[2]*2}"/>'
                    elif "Sphere" in stype:
                        geom_xml = f'<sphere radius="{shape.radius}"/>'
                    else:
                        continue

                    collision_xml += (
                        f'<collision>'
                        f'<origin xyz="{p[0]} {p[1]} {p[2]}" '
                        f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'
                        f'<geometry>{geom_xml}</geometry>'
                        f'</collision>'
                    )

            if collision_xml:
                # Insert just before the closing </link> tag for this link
                tag = f'<link name="{name}">'
                # Find the link and its closing tag, insert before </link>
                idx = urdf_str.find(tag)
                if idx == -1:
                    continue
                # Find the closing </link> after this opening tag
                close_idx = urdf_str.find("</link>", idx)
                if close_idx == -1:
                    continue
                urdf_str = urdf_str[:close_idx] + collision_xml + urdf_str[close_idx:]

        return urdf_str

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_pad"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_pad"
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
        qvel = self.robot.get_qvel()[..., :-8]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
