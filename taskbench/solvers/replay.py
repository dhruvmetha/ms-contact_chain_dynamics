"""Replay solver — re-execute a recorded skill program from an HDF5 demo.

Usage:
    uv run python -m taskbench.run solver=replay \\
        run.solver_kwargs.demo_path=data/success/episode_seed45.hdf5 \\
        task.num_cubes=5

    # With video recording:
    uv run python -m taskbench.run solver=replay \\
        run.solver_kwargs.demo_path=data/success/episode_seed45.hdf5 \\
        task.num_cubes=5 runtime.record_video=true

Note: task.num_cubes must match the demo. The solver will raise a clear
error if objects are missing.
"""

import json
import logging

import h5py
import numpy as np
import sapien
from omegaconf import OmegaConf

from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.replay")


def _deserialize_arg(value):
    """Convert a JSON-deserialized value back to its original type."""
    if isinstance(value, dict) and value.get("_type") == "pose":
        return sapien.Pose(
            np.array(value["p"], dtype=np.float32),
            np.array(value["q"], dtype=np.float32),
        )
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            return [_deserialize_arg(v) for v in value]
        if len(value) > 0 and isinstance(value[0], list):
            # Tuple of arrays, e.g. (p, q) for PoseLike
            return tuple(np.array(v, dtype=np.float32) for v in value)
        return value
    return value


def _deserialize_kwargs(args_json):
    """Deserialize a JSON string of skill kwargs."""
    raw = json.loads(args_json)
    return {k: _deserialize_arg(v) for k, v in raw.items()}


@register_solver("replay")
class ReplaySolver(BaseSolver):
    """Replay a recorded skill program from an HDF5 demonstration.

    Args:
        demo_path: Path to the HDF5 file containing the recorded program.
    """

    def __init__(self, demo_path: str):
        self.demo_path = demo_path

    def apply_demo_config(self, cfg):
        """Merge task config from the demo HDF5 into the running config.

        Forwards all task keys from the demo so the env matches the demo.
        Call this before env creation.
        """
        if not self.demo_path:
            return
        with h5py.File(self.demo_path, "r") as f:
            hydra_yaml = f["metadata"].attrs.get("hydra_config", "")
            if isinstance(hydra_yaml, bytes):
                hydra_yaml = hydra_yaml.decode()
            if not hydra_yaml:
                return
            demo_cfg = OmegaConf.create(hydra_yaml)

        # Forward all task.* keys from the demo config.
        # Fall back to legacy env.* for old recordings.
        demo_task = OmegaConf.select(demo_cfg, "task", default=None)
        if demo_task is None:
            demo_task = OmegaConf.select(demo_cfg, "env", default=None)
        if demo_task is None:
            return
        for key in demo_task:
            val = OmegaConf.select(demo_task, key)
            if val is not None:
                OmegaConf.update(cfg, f"task.{key}", val)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        if not self.demo_path:
            raise ValueError(
                "demo_path is required for ReplaySolver. "
                "Pass run.solver_kwargs.demo_path=<path>"
            )
        with h5py.File(self.demo_path, "r") as f:
            meta = f["metadata"].attrs
            demo_seed = int(meta["seed"])
            demo_env_id = meta.get("env_id", "")
            if isinstance(demo_env_id, bytes):
                demo_env_id = demo_env_id.decode()
            demo_config_yaml = meta.get("hydra_config", "")
            if isinstance(demo_config_yaml, bytes):
                demo_config_yaml = demo_config_yaml.decode()
            # Read num_cubes from objects group
            demo_num_cubes = len(f["objects"]) if "objects" in f else None
            if "program" not in f or "program/skill" not in f:
                logger.error("Demo %s has no recorded skill calls", self.demo_path)
                return SolverResult(
                    success=False, failure_reason="empty_demo"
                )
            skill_names = [s.decode() if isinstance(s, bytes) else s
                           for s in f["program/skill"]]
            skill_args = [s.decode() if isinstance(s, bytes) else s
                          for s in f["program/args"]]

        # Validate env matches the demo
        actual_env_id = env.unwrapped.spec.id if env.unwrapped.spec else ""
        if demo_env_id and actual_env_id and demo_env_id != actual_env_id:
            raise ValueError(
                f"Demo was recorded on {demo_env_id} but env is {actual_env_id}. "
                f"Override with task.env_id={demo_env_id}"
            )

        if demo_config_yaml:
            logger.info("Demo config:\n%s", demo_config_yaml)

        # Collect object names from the demo program
        demo_objects = set()
        for args_json in skill_args:
            kwargs = json.loads(args_json)
            if "obj_name" in kwargs:
                demo_objects.add(kwargs["obj_name"])

        logger.info(
            "Replaying %s: %d skill calls, seed=%d",
            self.demo_path, len(skill_names), demo_seed,
        )
        if demo_objects:
            logger.info("Demo uses objects: %s", sorted(demo_objects))

        ctx = SkillContext(env)
        ctx.reset(seed=demo_seed)

        # Validate that the env has all objects the demo needs
        if demo_objects:
            missing = demo_objects - set(ctx.objects.keys())
            if missing:
                hint = f"task.num_cubes={demo_num_cubes}" if demo_num_cubes else ""
                raise ValueError(
                    f"Demo requires objects {sorted(missing)} not found in env. "
                    f"Env has {sorted(ctx.objects.keys())}. "
                    f"Try: uv run python -m taskbench.run solver=replay "
                    f"run.solver_kwargs.demo_path={self.demo_path} {hint}"
                )

        for i, (skill_name, args_json) in enumerate(zip(skill_names, skill_args)):
            kwargs = _deserialize_kwargs(args_json)
            logger.info("Step %d/%d: %s(%s)", i + 1, len(skill_names), skill_name, kwargs)

            skill_fn = getattr(ctx, skill_name, None)
            if skill_fn is None:
                logger.error("Unknown skill: %s", skill_name)
                return SolverResult(
                    success=False,
                    failure_reason=f"unknown_skill:{skill_name}",
                )

            result = skill_fn(**kwargs)
            if not result.success:
                logger.warning("Step %d failed: %s", i + 1, result.failure_reason)
                return SolverResult(
                    success=False,
                    failure_reason=result.failure_reason,
                    info={"steps_completed": i},
                )

        logger.info("Replay complete: all %d skills executed", len(skill_names))
        return SolverResult(success=True)
