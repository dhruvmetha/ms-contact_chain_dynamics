"""Unified cube-stacking solver using pick/place skill objects.

Loops over ``pick()`` / ``place()`` calls to stack N cubes into a tower.
The 2-cube case (``StackCube-v1``) is just ``n=2``.
"""

import logging

import numpy as np
import sapien

from taskbench.recorder import StateRecorder
from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.stack_n_cubes")


@register_solver("stack_cubes")
class StackCubesSolver(BaseSolver):
    """Stack N cubes into a tower using sequential pick-place primitives."""

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        """Stack all cubes into a tower using N-1 pick-place operations."""
        self._cfg = cfg
        # Set up recorder and skill context
        ctx = SkillContext(env)
        ctx.reset(seed=seed)

        assert env.unwrapped.control_mode in [
            "pd_joint_pos",
            "pd_joint_pos_vel",
        ], f"Unsupported control mode: {env.unwrapped.control_mode}"

        raw = env.unwrapped
        objects = ctx.objects
        base_name = "cube_0"
        other_names = [n for n in objects.keys() if n != base_name]
        rng = np.random.default_rng(seed)
        rng.shuffle(other_names)
        n = len(objects)
        total_steps = n - 1

        # Set up state recorder and rebind skills with callback
        recorder = StateRecorder(
            env,
            objects=objects,
            robot_fields=[
                "qpos",
                "qf",
                "arm_force_limit",
                "joint_load_l2",
                "tcp_pos",
                "tcp_quat",
                "gripper_qpos",
                "gripper_contact_force",
            ],
        )
        ctx.step_callback = recorder.record
        ctx._build_skills()
        recorder.record()  # initial state

        logger.info(f"Starting sequential stacking: {n} cubes, {total_steps} pick-place steps (seed={seed})")
        logger.info(f"Base: {base_name}, pick order: {other_names}")

        target_name = base_name
        for i, pick_name in enumerate(other_names):
            pick_actor = objects[pick_name]
            target_actor = objects[target_name]

            # Dynamic lift height scales with stack progress
            cube_height = (raw.cube_half_size[2] * 2).item()
            lift_height = max(0.1, (i + 2) * cube_height + 0.05)

            logger.info(f"Step {i+1}/{total_steps}: picking {pick_name}...")
            pick_kwargs = {"obj_name": pick_name, "lift_height": lift_height}
            recorder.record_skill_call("pick", pick_kwargs)
            pick_result = ctx.pick(pick_name, lift_height=lift_height)

            if not pick_result.success:
                logger.warning(
                    f"Step {i+1}/{total_steps} FAILED: {pick_result.failure_reason}"
                )
                result = SolverResult(
                    success=False,
                    info={"cubes_stacked": i},
                    failure_reason=pick_result.failure_reason,
                )
                self._save_recording(recorder, seed, result)
                return result

            # Compute release pose: above target at lift height
            goal_pose = target_actor.pose * sapien.Pose([0, 0, cube_height])
            offset = (goal_pose.p - pick_actor.pose.p).cpu().numpy()[0]
            release_p = pick_result.lift_pose.p + offset
            release_q = pick_result.lift_pose.q

            logger.info(f"Step {i+1}/{total_steps}: placing {pick_name} on {target_name}...")
            place_target = (release_p, release_q)
            place_kwargs = {
                "target_pose": place_target,
                "retract_height": float(pick_result.lift_pose.p[2]),
            }
            recorder.record_skill_call("place", place_kwargs)
            place_result = ctx.place(
                place_target,
                retract_height=pick_result.lift_pose.p[2],
            )

            if not place_result.success:
                logger.warning(
                    f"Step {i+1}/{total_steps} FAILED: {place_result.failure_reason}"
                )
                result = SolverResult(
                    success=False,
                    info={"cubes_stacked": i},
                    failure_reason=place_result.failure_reason,
                )
                self._save_recording(recorder, seed, result)
                return result

            # Verify placement (task-specific: check cube Z-height)
            expected_z = (target_actor.pose.p[..., 2] + cube_height).cpu().numpy().item()
            actual_z = pick_actor.pose.p[..., 2].cpu().numpy().item()
            if abs(actual_z - expected_z) > 0.01:
                logger.warning(
                    f"Placement check failed: expected_z={expected_z:.4f}, "
                    f"actual_z={actual_z:.4f}, diff={abs(actual_z - expected_z):.4f}"
                )
                result = SolverResult(
                    success=False,
                    info={"cubes_stacked": i},
                    failure_reason="placement_check_failed",
                )
                self._save_recording(recorder, seed, result)
                return result

            logger.info(f"Step {i+1}/{total_steps} complete")
            target_name = pick_name  # next placement goes on top of this cube

        logger.info(f"Stacking complete: {total_steps}/{total_steps} cubes placed")
        result = SolverResult(
            success=True,
            info={"cubes_stacked": total_steps},
        )
        self._save_recording(recorder, seed, result)
        return result

    def _save_recording(self, recorder, seed, result):
        """Save state recording to data/success/ or data/failure/."""
        tag = "success" if result.success else "failure"
        recorder.save(
            f"data/{tag}/episode_seed{seed}.hdf5",
            metadata={
                "seed": seed,
                "solver": "stack_cubes",
                "success": result.success,
                "failure_reason": result.failure_reason or "",
            },
            hydra_cfg=self._cfg,
        )
