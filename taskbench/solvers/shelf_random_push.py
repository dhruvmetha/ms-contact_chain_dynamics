"""Random push solver for the shelf environment.

Samples two random points inside the shelf at the same height,
executes a push between them. One push per episode.
- RRT (plan_pose) to reach the start point
- Straight line (plan_screw) to sweep to the end point

Usage:
    uv run python -m taskbench.run task=shelf solver=shelf_random_push
    uv run python -m taskbench.run task=shelf solver=shelf_random_push \
        run.num_episodes=100
"""

import logging

import numpy as np
import sapien

from taskbench.recorder import StateRecorder
from taskbench.skills.context import SkillContext
from taskbench.skills.motion import (
    actuate_gripper,
    add_collision_boxes,
    follow_path,
    hold_current_pose,
    move_to_pose,
    move_to_pose_rrt,
    sapien_to_mplib_pose,
)
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.shelf_random_push")

# Gripper pointing into the shelf (+X direction), fingers vertical, wxyz
Q_INTO_SHELF = [0.5, 0.5, 0.5, 0.5]


def _sample_push(rng, shelf_geom, margin=0.03, y_margin=0.25, depth_frac=0.6,
                 z_offset=0.12):
    """Sample start and end points for a push at a fixed height.

    Both points share the same Z (pushing in a horizontal plane).

    Args:
        z_offset: height above shelf surface (meters).
    """
    g = shelf_geom
    max_x = g.front_x + g.depth * depth_frac
    z = g.surface_z + z_offset

    x1 = rng.uniform(g.front_x + margin, max_x)
    y1 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)

    x2 = rng.uniform(g.front_x + margin, max_x)
    y2 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)

    return [x1, y1, z], [x2, y2, z]


@register_solver("shelf_random_push")
class ShelfRandomPushSolver(BaseSolver):
    """One random push per episode. Reset between pushes."""

    def __init__(
        self,
        settle_steps: int = 60,
        margin: float = 0.03,
    ):
        self.settle_steps = int(settle_steps)
        self.margin = float(margin)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        ctx = SkillContext(env)
        ctx.reset(seed=seed)
        raw = env.unwrapped
        rc = ctx.robot_config
        rng = np.random.default_rng(seed)

        # Load shelf collision geometry into the motion planner
        boxes = raw.get_collision_boxes()
        add_collision_boxes(ctx.planner, boxes, resolution=0.005)

        # Set up recording
        recorder = StateRecorder(
            env,
            objects=ctx.objects,
            robot_fields=["qpos", "tcp_pos", "tcp_quat", "gripper_contact_force"],
        )
        ctx.step_callback = recorder.record
        recorder.record()

        # Sample push — resample until the full plan is feasible
        g = raw.shelf_geom
        max_attempts = 50
        p1, p2 = None, None
        for attempt in range(max_attempts):
            p1, p2 = _sample_push(rng, raw.shelf_geom, margin=self.margin)
            approach_pose = sapien.Pose(p=p1, q=Q_INTO_SHELF)
            push_pose = sapien.Pose(p=p2, q=Q_INTO_SHELF)
            staging_pose = sapien.Pose(
                p=[g.front_x - 0.05, p1[1], p1[2]], q=Q_INTO_SHELF,
            )

            # Plan the full chain — save plans for execution
            ts = raw.control_timestep

            # Plan staging (RRT)
            qpos = raw.agent.robot.get_qpos().cpu().numpy()[0]
            staging_goal = sapien_to_mplib_pose(staging_pose)
            plan_staging = ctx.planner.plan_pose(
                staging_goal, qpos, time_step=ts, planning_time=1.0,
            )
            if plan_staging["status"] != "Success":
                continue

            # Plan approach (screw from staging end)
            approach_goal = sapien_to_mplib_pose(approach_pose)
            plan_approach = ctx.planner.plan_screw(
                approach_goal, plan_staging["position"][-1], time_step=ts,
            )
            if plan_approach["status"] != "Success":
                continue

            # Plan sweep (screw from approach end, slow)
            push_goal = sapien_to_mplib_pose(push_pose)
            plan_sweep = ctx.planner.plan_screw(
                push_goal, plan_approach["position"][-1], time_step=ts * 0.3,
            )
            if plan_sweep["status"] != "Success":
                continue
            logger.info(
                "Push (attempt %d): [%.2f, %.2f, %.2f] -> [%.2f, %.2f, %.2f]",
                attempt + 1, *p1, *p2,
            )
            break
        else:
            logger.info("  -> no feasible push found in %d attempts", max_attempts)
            return SolverResult(success=False, failure_reason="no_feasible_push",
                                elapsed_steps=len(recorder.frames))

        # Visualize sampled points as colored spheres
        green_mat = sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
        red_mat = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        for pt, mat, name in [(p1, red_mat, "marker_start"),
                               (p2, green_mat, "marker_end")]:
            b = raw.scene.create_actor_builder()
            b.add_sphere_visual(radius=0.015, material=mat)
            b.initial_pose = sapien.Pose(p=pt)
            b.build_static(name=name)

        recorder.record_skill_call("push", {
            "approach_pose": p1,
            "push_pose": p2,
        })

        # Close gripper
        actuate_gripper(env, ctx.planner, rc.gripper_closed,
                        step_callback=recorder.record)

        # Execute the pre-planned paths
        follow_path(env, plan_staging, rc.gripper_closed, rc,
                    step_callback=recorder.record)
        follow_path(env, plan_approach, rc.gripper_closed, rc,
                    step_callback=recorder.record)
        follow_path(env, plan_sweep, rc.gripper_closed, rc,
                    monitor_contacts=True,
                    allowed_contact_links=rc.gripper_link_names,
                    step_callback=recorder.record)

        # Settle
        hold_current_pose(
            env, ctx.planner, rc.gripper_closed,
            steps=self.settle_steps,
            step_callback=recorder.record,
        )

        logger.info("  -> OK")
        solver_result = SolverResult(
            success=True,
            elapsed_steps=len(recorder.frames),
        )
        self.save_recording(recorder, seed, solver_result, cfg=cfg)
        return solver_result
