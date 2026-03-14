"""Base class for taskbench environments."""

from abc import ABCMeta, abstractmethod

import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv


class TaskEnv(BaseEnv, metaclass=ABCMeta):
    """Base class for all taskbench custom environments.

    Subclasses must implement ``get_objects()`` to expose their
    manipulable objects with canonical names.
    """

    def _default_initial_agent_poses(self):
        """Return safe build-time robot base poses when known.

        ManiSkill emits a noisy warning and spawns the robot at the origin if
        no initial pose is provided during articulation build. For this repo,
        most tasks later reset the robot to a fixed base pose anyway, so we
        provide those poses up front to avoid transient spawn collisions and
        warning spam.
        """
        robot_uids = self.robot_uids
        if robot_uids is None:
            return None

        pose_by_uid = {
            "panda": sapien.Pose([-0.615, 0.0, 0.0]),
            "panda_wristcam": sapien.Pose([-0.615, 0.0, 0.0]),
            "panda_stick": sapien.Pose([-0.615, 0.0, 0.0]),
            "xarm6_allegro_left": sapien.Pose([-0.522, 0.0, 0.0]),
            "xarm6_allegro_right": sapien.Pose([-0.522, 0.0, 0.0]),
            "xarm6_robotiq": sapien.Pose([-0.522, 0.0, 0.0]),
            "xarm6_nogripper": sapien.Pose([-0.522, 0.0, 0.0]),
            "fetch": sapien.Pose([-1.05, 0.0, -0.9196429]),
            "widowxai": sapien.Pose(),
            "widowxai_wristcam": sapien.Pose(),
            "so100": sapien.Pose([-0.725, 0.0, 0.0], q=euler2quat(0, 0, 3.141592653589793 / 2)),
        }

        if isinstance(robot_uids, tuple):
            poses = []
            for idx, uid in enumerate(robot_uids):
                if uid == "panda":
                    yaw = 3.141592653589793 / 2 if idx == 0 else -3.141592653589793 / 2
                    y = -0.75 if idx == 0 else 0.75
                    poses.append(sapien.Pose([0.0, y, 0.0], q=euler2quat(0, 0, yaw)))
                elif uid == "panda_wristcam":
                    yaw = 3.141592653589793 / 2 if idx == 0 else -3.141592653589793 / 2
                    y = -0.75 if idx == 0 else 0.75
                    poses.append(sapien.Pose([0.0, y, 0.0], q=euler2quat(0, 0, yaw)))
                else:
                    pose = pose_by_uid.get(uid)
                    if pose is None:
                        return None
                    poses.append(pose)
            return poses

        return pose_by_uid.get(robot_uids)

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate: bool = False):
        """Load the robot with a safe initial base pose when available."""
        if initial_agent_poses is None:
            initial_agent_poses = self._default_initial_agent_poses()
        return super()._load_agent(
            options,
            initial_agent_poses=initial_agent_poses,
            build_separate=build_separate,
        )

    @abstractmethod
    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        ...
