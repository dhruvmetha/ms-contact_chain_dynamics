"""Base class for taskbench environments."""

from abc import ABCMeta, abstractmethod

import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv


class TaskEnv(BaseEnv, metaclass=ABCMeta):
    """Base class for all taskbench custom environments.

    Subclasses must implement ``get_objects()`` to expose their
    manipulable objects with canonical names.

    All subclasses receive ``robot_base_pose`` from the task YAML via
    ``**kwargs``.  This class intercepts it before ``BaseEnv.__init__``.
    """

    def __init__(self, *args, robot_base_pose=None, **kwargs):
        if robot_base_pose is None:
            raise ValueError(
                "robot_base_pose is required — set it in the task YAML "
                "(e.g., task.robot_base_pose: [-0.615, 0.0, 0.0])"
            )
        self._robot_base_pose = list(robot_base_pose)
        super().__init__(*args, **kwargs)

    def _robot_base_sapien_pose(self):
        """Build a ``sapien.Pose`` from the task-config ``robot_base_pose``.

        Accepts 3-element ``[x, y, z]`` (identity quaternion) or
        7-element ``[x, y, z, qw, qx, qy, qz]``.
        """
        n = len(self._robot_base_pose)
        if n not in (3, 7):
            raise ValueError(
                f"robot_base_pose must have 3 or 7 elements, got {n}"
            )
        p = self._robot_base_pose[:3]
        if n == 7:
            return sapien.Pose(p, self._robot_base_pose[3:])
        return sapien.Pose(p)

    def _default_initial_agent_poses(self):
        """Return the robot base pose declared in the task config."""
        return self._robot_base_sapien_pose()

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate: bool = False):
        """Load the robot with the task-config base pose."""
        if initial_agent_poses is None:
            initial_agent_poses = self._default_initial_agent_poses()
        return super()._load_agent(
            options,
            initial_agent_poses=initial_agent_poses,
            build_separate=build_separate,
        )

    def _reset_robot(self, env_idx):
        """Reset robot base pose and joint configuration for the given envs.

        Sets the robot base pose from the task config and applies rest
        keyframe qpos for custom agents (indicated by the agent class
        defining ``table_scene_base_pose``).

        Only modifies the envs specified by ``env_idx`` so that partial
        batched resets don't clobber running envs.
        """
        raw = self.unwrapped if hasattr(self, "unwrapped") else self

        # Apply rest qpos for custom agents not handled by TableSceneBuilder
        agent_cls = type(raw.agent)
        if getattr(agent_cls, "table_scene_base_pose", None) is not None:
            keyframes = getattr(agent_cls, "keyframes", {})
            rest = keyframes.get("rest")
            if rest is not None and rest.qpos is not None:
                rest_qpos = torch.tensor(
                    rest.qpos, dtype=torch.float32, device=raw.device
                )
                full_qpos = raw.agent.robot.get_qpos()  # (N, D)
                full_qpos[env_idx] = rest_qpos
                raw.agent.robot.set_qpos(full_qpos)

        # Apply robot base pose from task config
        pose = self._robot_base_sapien_pose()
        base_p = torch.tensor(
            pose.p, dtype=torch.float32, device=raw.device
        )
        base_q = torch.tensor(
            pose.q, dtype=torch.float32, device=raw.device
        )
        cur_p = raw.agent.robot.pose.p.clone()
        cur_q = raw.agent.robot.pose.q.clone()
        cur_p[env_idx] = base_p
        cur_q[env_idx] = base_q
        from mani_skill.utils.structs.pose import Pose as MSPose
        raw.agent.robot.set_pose(MSPose.create_from_pq(p=cur_p, q=cur_q))

    @abstractmethod
    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        ...
