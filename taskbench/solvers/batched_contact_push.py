"""GPU-batched push data collection using cuRobo planning.

Collects contact data across N parallel environments simultaneously.
Uses cuRobo for collision-aware trajectory planning on GPU, enabling
massively parallel data collection for training a contact world model.
"""

import logging
import os

import numpy as np
import torch

from taskbench.batched_recorder import BatchedStateRecorder
from taskbench.skills.batched_context import BatchedSkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.batched_contact_push")


def _build_push_poses(env, n_envs: int, push_distance: float = 0.10,
                      approach_gap: float = 0.06,
                      push_randomization: bool = False,
                      seed: int = 0):
    """Build approach and push target poses for all N envs.

    Queries the env's ``get_scene_info()`` for scene geometry and computes
    GPU-batched approach/push poses. The tool orientation points straight
    down (vertical push).

    Returns (approach_positions, approach_quaternions,
             push_positions, push_quaternions) as (N, ...) tensors.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    device = raw.device

    if not hasattr(raw, "get_scene_info"):
        raise TypeError(
            f"{type(raw).__name__} does not implement get_scene_info(). "
            f"batched_contact_push requires TabletopRetrievalEnv "
            f"(task=tabletop_retrieval)."
        )
    scene = raw.get_scene_info()
    pd = np.asarray(scene["row_direction_xy"], dtype=np.float32).flatten()[:2]
    push_dir = torch.tensor([pd[0], pd[1], 0.0], device=device)
    origin = np.asarray(scene["row_origin_xy"], dtype=np.float32).flatten()[:2]
    contact_xy = torch.tensor([origin[0], origin[1]], device=device)

    # Contact height from the first object's z position
    objects = getattr(raw, "cylinders", None)
    if objects and len(objects) > 0:
        contact_z = float(objects[0].pose.p[0, 2].cpu())
    else:
        contact_z = 0.05

    # Approach: before the contact point along the push direction
    approach_xy = contact_xy - push_dir[:2] * approach_gap
    approach_p = torch.tensor(
        [approach_xy[0].item(), approach_xy[1].item(), contact_z],
        device=device,
    ).unsqueeze(0).expand(n_envs, -1).clone()

    # Push end: past the objects along the push direction
    push_xy = contact_xy + push_dir[:2] * push_distance
    push_p = torch.tensor(
        [push_xy[0].item(), push_xy[1].item(), contact_z],
        device=device,
    ).unsqueeze(0).expand(n_envs, -1).clone()

    # Tool-pointing-down orientation in cuRobo's ee_link frame.
    # Robot-specific because each robot's ee_link has a different
    # local frame convention.
    from taskbench.skills.curobo_motion import get_tool_down_quat
    tool_down_quat = torch.tensor(get_tool_down_quat(raw.agent.uid), device=device)
    goal_quat = tool_down_quat.unsqueeze(0).expand(n_envs, -1).clone()

    # Per-env randomization
    if push_randomization and n_envs > 1:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Per-env contact point noise — shifts the contact origin so each
        # env pushes from a slightly different position/height.  Both approach
        # and push endpoints are derived from the same contact offset so the
        # push vector (direction + distance) is preserved.
        contact_noise = torch.zeros(n_envs, 3, device=device)
        contact_noise[:, :2] = (torch.rand(n_envs, 2, device=device, generator=rng) - 0.5) * 0.02
        contact_noise[:, 2] = (torch.rand(n_envs, device=device, generator=rng) - 0.5) * 0.01

        # Per-env push distance noise — varies how far each env pushes.
        dist_noise = (torch.rand(n_envs, 1, device=device, generator=rng) - 0.5) * 0.04
        push_dir_3d = push_dir.unsqueeze(0).expand(n_envs, -1).clone()

        approach_p = approach_p + contact_noise
        push_p = push_p + contact_noise + dist_noise * push_dir_3d

    return (
        approach_p.contiguous(),
        goal_quat,
        push_p.contiguous(),
        goal_quat,
    )


@register_solver("batched_contact_push")
class BatchedContactPushSolver(BaseSolver):
    """GPU-batched push data collection using cuRobo planning.

    Creates a GPU-vectorized env, sets up cuRobo planner once (expensive
    warmup), then per-episode: randomizes scenes, plans + executes batched
    pushes, and records contact data.
    """

    def __init__(
        self,
        push_distance: float = 0.10,
        approach_gap: float = 0.06,
        lift_height: float = 0.08,
        settle_steps: int = 8,
        push_randomization: bool = True,
        interpolation_dt: float = 0.02,
        save_data: bool = True,
        data_dir: str = "data/batched",
    ):
        self.push_distance = push_distance
        self.approach_gap = approach_gap
        self.lift_height = lift_height
        self.settle_steps = settle_steps
        self.push_randomization = push_randomization
        self.interpolation_dt = interpolation_dt
        self.save_data = save_data
        self.data_dir = data_dir
        self._ctx = None  # Persisted across episodes (warmup is expensive)
        self._recorder = None

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        """Run one batched episode of push data collection.

        Note: Unlike other solvers, this solver receives a GPU-vectorized
        env (from ``run_batched``), not a single-env.
        """
        raw = env.unwrapped if hasattr(env, "unwrapped") else env
        robot_uid = raw.agent.uid
        n_envs = raw.num_envs

        # Create context once, reuse across episodes (warmup is expensive)
        if self._ctx is None:
            self._ctx = BatchedSkillContext(
                env,
                robot_uid=robot_uid,
                n_envs=n_envs,
                interpolation_dt=self.interpolation_dt,
            )
        ctx = self._ctx

        # Reset all envs (populates ctx.objects)
        ctx.reset(seed=seed)

        # Reuse recorder across episodes, just clear buffers + update objects
        if self._recorder is None:
            self._recorder = BatchedStateRecorder(
                env,
                n_envs=n_envs,
                objects=ctx.objects,
                robot_fields=["qpos", "tcp_pos", "tcp_quat"],
            )
        else:
            self._recorder.reset(objects=ctx.objects)
        recorder = self._recorder
        recorder.record()

        # Build push poses for all envs
        approach_pos, approach_quat, push_pos, push_quat = _build_push_poses(
            env, n_envs,
            push_distance=self.push_distance,
            approach_gap=self.approach_gap,
            push_randomization=self.push_randomization,
            seed=seed or 0,
        )

        # Execute batched push
        recorder.set_skill("push")
        push_result = ctx.push(
            approach_pos, approach_quat,
            push_pos, push_quat,
            lift_height=self.lift_height,
            settle_steps=self.settle_steps,
            step_callback=lambda t, obs, rew: recorder.record(),
        )

        # Evaluate results
        n_success = push_result.n_success
        success_rate = n_success / n_envs

        # Save data
        if self.save_data:
            os.makedirs(self.data_dir, exist_ok=True)
            recorder.save(
                os.path.join(self.data_dir, f"batch_push_seed{seed}.hdf5"),
                metadata={
                    "seed": seed or 0,
                    "solver": "batched_contact_push",
                    "n_envs": n_envs,
                    "n_success": n_success,
                    "success_rate": success_rate,
                },
                hydra_cfg=cfg,
            )

        logger.info(
            "Batched push: %d/%d envs succeeded (%.1f%%)",
            n_success, n_envs, success_rate * 100,
        )

        return SolverResult(
            success=success_rate > 0.5,
            reward=success_rate,
            elapsed_steps=push_result.steps_executed,
            info={
                "n_success": n_success,
                "n_total": n_envs,
                "success_rate": success_rate,
            },
        )
