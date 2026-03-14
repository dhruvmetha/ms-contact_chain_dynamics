"""Offline push smoke test for the open-table clutter environment."""

import logging
import os

from taskbench.recorder import StateRecorder
from taskbench.skills.context import SkillContext
from taskbench.skills.motion import hold_current_pose
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.open_table_push")


@register_solver("open_table_push")
class OpenTablePushSolver(BaseSolver):
    """Execute a single deterministic push on the open-table cylinder row."""

    def __init__(
        self,
        clearance_height: float = 0.0,
        lift_height: float = 0.0,
        effort_scale: float = 1.0,
        effort_scale_end: float | None = None,
        min_contact_force: float = 0.01,
        open_gripper_after_push: bool = False,
        staging_speed_scale: float = 1.0,
        hover_speed_scale: float = 1.0,
        approach_speed_scale: float = 1.0,
        push_speed_scale: float = 1.0,
        settle_steps: int = 60,
    ):
        self.clearance_height = clearance_height
        self.lift_height = lift_height
        self.effort_scale = effort_scale
        self.effort_scale_end = effort_scale_end
        self.min_contact_force = min_contact_force
        self.open_gripper_after_push = open_gripper_after_push
        self.staging_speed_scale = staging_speed_scale
        self.hover_speed_scale = hover_speed_scale
        self.approach_speed_scale = approach_speed_scale
        self.push_speed_scale = push_speed_scale
        self.settle_steps = settle_steps

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        self._cfg = cfg
        ctx = SkillContext(env)
        ctx.reset(seed=seed)
        raw = env.unwrapped

        recorder = StateRecorder(
            env,
            objects=ctx.objects,
            robot_fields=[
                "qpos",
                "arm_force_limit",
                "joint_load_l2",
                "gripper_contact_force",
                "tcp_pos",
                "tcp_quat",
            ],
        )
        ctx.step_callback = recorder.record
        ctx._build_skills()
        recorder.record()

        if hasattr(raw, "get_push_plan"):
            push_plan = raw.get_push_plan()
            staging_pose = push_plan.get("staging_pose")
            hover_pose = push_plan.get("hover_pose")
            approach_pose = push_plan["approach_pose"]
            push_pose = push_plan["push_pose"]
        elif hasattr(raw, "get_push_poses"):
            staging_pose = None
            hover_pose = None
            approach_pose, push_pose = raw.get_push_poses()
        else:
            raise RuntimeError(
                f"{type(raw).__name__} does not expose get_push_plan() or get_push_poses()"
            )

        push_kwargs = {
            "staging_pose": staging_pose,
            "hover_pose": hover_pose,
            "approach_pose": approach_pose,
            "push_pose": push_pose,
            "clearance_height": self.clearance_height,
            "lift_height": self.lift_height,
            "effort_scale": self.effort_scale,
            "effort_scale_end": (
                self.effort_scale if self.effort_scale_end is None else self.effort_scale_end
            ),
            "min_contact_force": self.min_contact_force,
            "open_gripper_after_push": self.open_gripper_after_push,
            "staging_speed_scale": self.staging_speed_scale,
            "hover_speed_scale": self.hover_speed_scale,
            "approach_speed_scale": self.approach_speed_scale,
            "push_speed_scale": self.push_speed_scale,
        }
        recorder.record_skill_call("push", push_kwargs)
        push_result = ctx.push(**push_kwargs)

        hold_current_pose(
            env,
            ctx.planner,
            ctx.robot_config.gripper_open,
            steps=self.settle_steps,
            step_callback=recorder.record,
        )

        info = raw.evaluate()
        success = bool(info["success"].item())
        target_displacement = float(info["target_displacement"].item())
        target_displacement_y = float(
            info.get("target_displacement_y", info["target_displacement"]).item()
        )

        result = SolverResult(
            success=success,
            reward=float(success),
            elapsed_steps=len(recorder.frames),
            info={
                "target_displacement": target_displacement,
                "target_displacement_y": target_displacement_y,
                "push_distance": push_result.push_distance,
                "planar_push_distance": push_result.planar_push_distance,
                "arm_force_limit_mean": push_result.arm_force_limit_mean,
                "contact_force_peak": push_result.contact_force_peak,
                "joint_load_l2_peak": push_result.joint_load_l2_peak,
            },
            failure_reason=None if success else (
                push_result.failure_reason or "target_displacement_below_threshold"
            ),
        )
        self._save_recording(recorder, seed, result)
        logger.info(
            "Open-table push: success=%s target_d=%.3f peak_contact=%.3f peak_joint_load=%.3f",
            success,
            target_displacement,
            push_result.contact_force_peak,
            push_result.joint_load_l2_peak,
        )
        return result

    def _save_recording(self, recorder, seed, result):
        tag = "success" if result.success else "failure"
        os.makedirs(f"data/{tag}", exist_ok=True)
        recorder.save(
            f"data/{tag}/open_table_push_seed{seed}.hdf5",
            metadata={
                "seed": seed,
                "solver": "open_table_push",
                "success": result.success,
                "failure_reason": result.failure_reason or "",
            },
            hydra_cfg=self._cfg,
        )
