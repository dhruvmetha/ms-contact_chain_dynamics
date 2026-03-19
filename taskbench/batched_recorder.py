"""Batched state recorder for N parallel environments.

Captures ``(N, T, ...)`` tensors of simulation state from GPU-vectorized
ManiSkill envs. Accumulates data on GPU and transfers to CPU once at
episode end to minimize per-step overhead.

Saves as HDF5 with an env-index dimension::

    episode.hdf5
    ├── metadata/          attrs: seed, env_id, n_envs, num_frames
    ├── robot/
    │   ├── qpos           (N, T, n_joints)
    │   ├── tcp_pos        (N, T, 3)
    │   └── tcp_quat       (N, T, 4)
    ├── objects/
    │   ├── cyl_0/
    │   │   ├── pos        (N, T, 3)
    │   │   └── quat       (N, T, 4)
    │   └── ...
    └── skill              (T,) per-frame skill labels
"""

import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

logger = logging.getLogger("taskbench.batched_recorder")


# Batched field extractors: take (raw_env,) and return (N, ...) tensors.
# These stay on GPU — no .cpu() call per step.
_BATCHED_FIELD_EXTRACTORS = {
    "qpos": lambda raw: raw.agent.robot.get_qpos(),
    "qvel": lambda raw: raw.agent.robot.get_qvel(),
    "tcp_pos": lambda raw: raw.agent.tcp.pose.p,
    "tcp_quat": lambda raw: raw.agent.tcp.pose.q,
    "gripper_qpos": lambda raw: raw.agent.robot.get_qpos()[:, -2:],
}


class BatchedStateRecorder:
    """Records simulation state from N parallel GPU envs.

    Accumulates tensors on GPU and transfers to CPU at save time.

    Args:
        env: GPU-vectorized ManiSkill env.
        n_envs: Number of parallel environments.
        objects: Dict mapping name → actor for tracked objects.
            If None, no object poses are recorded.
        robot_fields: List of robot state fields to record.
            Available: qpos, qvel, tcp_pos, tcp_quat, gripper_qpos.
    """

    def __init__(
        self,
        env,
        n_envs: int,
        objects: Optional[Dict[str, object]] = None,
        robot_fields: Optional[List[str]] = None,
    ):
        self.env = env
        self.raw = env.unwrapped if hasattr(env, "unwrapped") else env
        self.n_envs = n_envs
        self.objects = objects or {}
        self.robot_fields = robot_fields or []
        self._skill = ""

        # GPU tensor accumulators: field_name → list of (N, ...) tensors
        self._buffers: Dict[str, list] = {}
        self._skill_labels: List[str] = []

        for field in self.robot_fields:
            if field not in _BATCHED_FIELD_EXTRACTORS:
                available = ", ".join(sorted(_BATCHED_FIELD_EXTRACTORS))
                raise ValueError(
                    f"Unknown robot field {field!r}. Available: {available}"
                )

    def set_skill(self, label: str):
        """Set the current skill label for subsequent frames."""
        self._skill = label

    def record(self):
        """Capture one frame of state from all N envs.

        Tensors stay on GPU. Call after each env.step().
        """
        raw = self.raw

        # Robot fields — clone() is essential: ManiSkill reuses the same
        # GPU buffer across steps, so detach() alone would give a view into
        # memory that gets overwritten on the next step.
        for field in self.robot_fields:
            tensor = _BATCHED_FIELD_EXTRACTORS[field](raw)
            self._buffers.setdefault(field, []).append(tensor.detach().clone())

        # Object poses — same: pose tensors are internal SAPIEN buffers
        # updated in-place each step.
        for name, actor in self.objects.items():
            pos = actor.pose.p.detach().clone()  # (N, 3)
            quat = actor.pose.q.detach().clone()  # (N, 4)
            self._buffers.setdefault(f"{name}_pos", []).append(pos)
            self._buffers.setdefault(f"{name}_quat", []).append(quat)

        self._skill_labels.append(self._skill)

    def save(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        hydra_cfg=None,
    ):
        """Transfer GPU data to CPU and save to HDF5.

        Args:
            path: Output file path.
            metadata: Optional episode metadata dict.
            hydra_cfg: Optional Hydra config for reproducibility.
        """
        if not self._skill_labels:
            logger.warning("No frames recorded, skipping save")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        num_frames = len(self._skill_labels)

        # Stack and transfer to CPU (one bulk transfer per field)
        cpu_data: Dict[str, np.ndarray] = {}
        for key, tensor_list in self._buffers.items():
            # Stack: list of (N, ...) → (T, N, ...), then transpose to (N, T, ...)
            stacked = torch.stack(tensor_list, dim=0)  # (T, N, ...)
            stacked = stacked.permute(1, 0, *range(2, stacked.ndim))  # (N, T, ...)
            cpu_data[key] = stacked.cpu().numpy()

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["num_frames"] = num_frames
            meta.attrs["n_envs"] = self.n_envs
            meta.attrs["control_freq"] = self.raw.control_freq
            meta.attrs["env_id"] = self.raw.spec.id if self.raw.spec else ""
            if metadata:
                for key, value in metadata.items():
                    meta.attrs[key] = value
            if hydra_cfg is not None:
                from omegaconf import OmegaConf
                meta.attrs["hydra_config"] = OmegaConf.to_yaml(hydra_cfg)

            # Robot state
            if self.robot_fields:
                robot_grp = f.create_group("robot")
                for field in self.robot_fields:
                    if field in cpu_data:
                        robot_grp.create_dataset(
                            field, data=cpu_data[field], compression="gzip"
                        )

            # Object poses
            if self.objects:
                obj_grp = f.create_group("objects")
                for name in self.objects:
                    name_grp = obj_grp.create_group(name)
                    pos_key = f"{name}_pos"
                    quat_key = f"{name}_quat"
                    if pos_key in cpu_data:
                        name_grp.create_dataset(
                            "pos", data=cpu_data[pos_key], compression="gzip"
                        )
                    if quat_key in cpu_data:
                        name_grp.create_dataset(
                            "quat", data=cpu_data[quat_key], compression="gzip"
                        )

            # Per-frame skill labels
            dt = h5py.string_dtype()
            f.create_dataset("skill", data=self._skill_labels, dtype=dt)

        logger.info(
            "Saved %d frames × %d envs to %s", num_frames, self.n_envs, path
        )

    def reset(self, objects=None):
        """Clear buffers and optionally update tracked objects for a new episode."""
        self.clear()
        if objects is not None:
            self.objects = objects

    def clear(self):
        """Discard all recorded frames."""
        self._buffers.clear()
        self._skill_labels.clear()
