"""Push-based retrieval solver for the tabletop retrieval task.

One strategy for tabletop retrieval: push obstacles aside to clear a path
to the target. Queries ``get_scene_info()`` for scene geometry and computes
push waypoints via ``make_linear_push_plan()``.
"""

import logging

import numpy as np

from taskbench.recorder import StateRecorder
from taskbench.skills.context import SkillContext
from taskbench.skills.motion import (
    hold_current_pose,
    make_linear_push_plan,
    tcp_height_for_table_clearance,
)
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.tabletop_push")


def _parse_axis_triple(x, y, z) -> np.ndarray | None:
    values = (x, y, z)
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("approach_axis x, y, z must all be set together")
    return np.array(values, dtype=np.float32)


@register_solver("tabletop_push")
class TabletopPushSolver(BaseSolver):
    """Push-based retrieval: push obstacles aside on the tabletop."""

    def __init__(
        self,
        # Trajectory planning
        wrist_orientation: str = "vertical",
        tool_spin_deg: float = 0.0,
        approach_axis_x: float | None = None,
        approach_axis_y: float | None = None,
        approach_axis_z: float | None = None,
        table_clearance: float = 0.006,
        push_height_offset: float = 0.0,
        approach_gap: float = 0.06,
        end_margin: float = 0.05,
        # Staging
        use_staging: bool = True,
        staging_height: float = 0.18,
        staging_backoff: float = 0.0,
        staging_wrist_orientation: str | None = None,
        staging_tool_spin_deg: float | None = None,
        staging_approach_axis_x: float | None = None,
        staging_approach_axis_y: float | None = None,
        staging_approach_axis_z: float | None = None,
        # Execution dynamics
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
        self.wrist_orientation = wrist_orientation
        self.tool_spin_deg = float(tool_spin_deg)
        self.approach_axis = _parse_axis_triple(
            approach_axis_x, approach_axis_y, approach_axis_z
        )
        self.table_clearance = float(table_clearance)
        self.push_height_offset = float(push_height_offset)
        self.approach_gap = float(approach_gap)
        self.end_margin = float(end_margin)
        self.use_staging = use_staging
        self.staging_height = float(staging_height)
        self.staging_backoff = float(staging_backoff)
        self.staging_wrist_orientation = staging_wrist_orientation
        self.staging_tool_spin_deg = (
            None if staging_tool_spin_deg is None else float(staging_tool_spin_deg)
        )
        self.staging_approach_axis = _parse_axis_triple(
            staging_approach_axis_x, staging_approach_axis_y, staging_approach_axis_z
        )
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

    def _build_push_plan(self, env):
        """Compute push trajectory from scene geometry."""
        raw = env.unwrapped
        scene = raw.get_scene_info()

        z = (
            tcp_height_for_table_clearance(
                raw.agent,
                scene["row_direction_xy"],
                wrist_orientation=self.wrist_orientation,
                tool_spin_deg=self.tool_spin_deg,
                approach_axis=self.approach_axis,
                table_z=scene["table_top_z"],
                table_clearance=self.table_clearance,
            )
            + self.push_height_offset
        )
        push_distance = (
            (scene["num_objects"] - 1) * scene["row_spacing"] + self.end_margin
        )
        contact_position = np.array(
            [scene["row_origin_xy"][0], scene["row_origin_xy"][1], z],
            dtype=np.float32,
        )
        staging_height = (
            (scene["table_top_z"] + self.staging_height) if self.use_staging else None
        )
        plan = make_linear_push_plan(
            raw.agent,
            contact_position=contact_position,
            push_distance=push_distance,
            push_direction=scene["row_direction_xy"],
            wrist_orientation=self.wrist_orientation,
            tool_spin_deg=self.tool_spin_deg,
            approach_axis=self.approach_axis,
            approach_gap=self.approach_gap,
            staging_height=staging_height,
            staging_backoff=self.staging_backoff,
            staging_wrist_orientation=self.staging_wrist_orientation,
            staging_tool_spin_deg=self.staging_tool_spin_deg,
            staging_approach_axis=self.staging_approach_axis,
        )
        return plan.as_skill_kwargs()

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
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
        recorder.record()

        push_plan = self._build_push_plan(env)

        push_kwargs = {
            **push_plan,
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

        settle_gripper = (
            ctx.robot_config.gripper_open
            if self.open_gripper_after_push
            else ctx.robot_config.gripper_closed
        )
        hold_current_pose(
            env,
            ctx.planner,
            settle_gripper,
            steps=self.settle_steps,
            step_callback=recorder.record,
        )

        info = raw.evaluate()
        success = bool(info["success"].item())

        result = SolverResult(
            success=success,
            reward=float(success),
            elapsed_steps=len(recorder.frames),
            info={
                "target_lifted": bool(info["target_lifted"].item()),
                "target_grasped": bool(info["target_grasped"].item()),
                "push_distance": push_result.push_distance,
                "contact_force_peak": push_result.contact_force_peak,
            },
            failure_reason=None if success else (
                push_result.failure_reason or "target_not_retrieved"
            ),
        )
        self.save_recording(recorder, seed, result, cfg=cfg)
        logger.info(
            "Tabletop push: success=%s lifted=%s grasped=%s",
            success,
            info["target_lifted"].item(),
            info["target_grasped"].item(),
        )
        return result
