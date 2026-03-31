"""Random push solver for the shelf environment using the Panda Stick + cuRobo.

Simple two-phase approach:
  1. MotionGen: plan from rest -> approach point (in front of shelf)
  2. Cartesian push: dense IK waypoints from approach -> push end

No gripper — the stick tip pushes objects directly.

Usage:
    uv run python -m taskbench.run task=shelf_panda_stick solver=shelf_panda_stick_push_curobo
"""

import logging
from pathlib import Path

import numpy as np
import torch

from taskbench.recorder import StateRecorder
from taskbench.skills.curobo_motion import (
    batched_cartesian_push,
    batched_follow_path,
    batched_move_to_pose,
    get_arm_joint_names,
    setup_curobo_planner,
    _sapien_qpos_to_cu_joint_state,
)
from taskbench.skills.curobo_world import shelf_world
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.shelf_panda_stick_push_curobo")

ROBOT_UID = "panda_stick_long"

# EE orientation for forward reach into shelf (wxyz).
# Maps panda_hand_tcp local Z (stick direction) -> world +X (into shelf).
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]

# Import the agent so it gets registered
import taskbench.agents.panda_stick_long  # noqa: F401

# cuRobo retract config — the drifted version after stepping passes collision check.
PANDA_REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]


def _sample_push(rng, shelf_geom, margin=0.03, y_margin=0.12, depth_frac=0.7,
                 z_offset=0.045, min_distance=0.08):
    """Sample start and end points for a push at a fixed height."""
    g = shelf_geom
    max_x = g.front_x + g.depth * depth_frac
    z = g.surface_z + z_offset

    for _ in range(100):
        if rng.random() < 0.5:
            x = rng.uniform(g.front_x + margin, max_x)
            y1 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)
            y2 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)
            x1, x2 = x, x
        else:
            x1 = rng.uniform(g.front_x + margin, max_x)
            y1 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)
            x2 = rng.uniform(g.front_x + margin, max_x)
            y2 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)

        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if dist >= min_distance:
            return [x1, y1, z], [x2, y2, z]

    x = rng.uniform(g.front_x + margin, max_x)
    y1 = -g.half_w + y_margin
    y2 = g.half_w - y_margin
    return [x, y1, z], [x, y2, z]


@register_solver("shelf_panda_stick_push_curobo")
class ShelfPandaStickPushCuroboSolver(BaseSolver):
    """One random push per episode using the Panda Stick + cuRobo.

    Two-phase: MotionGen staging + dense Cartesian push.
    """

    def __init__(
        self,
        settle_steps: int = 60,
        margin: float = 0.03,
    ):
        self.settle_steps = int(settle_steps)
        self.margin = float(margin)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        raw = env.unwrapped
        rng = np.random.default_rng(seed)

        device = raw.device
        cuda_device = torch.device("cuda:0")
        n_arm = len(get_arm_joint_names(ROBOT_UID))

        # Set rest pose and let it settle slightly — the drifted qpos from PD
        # control avoids the exact retract config (which cuRobo flags as in-collision).
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[0, :n_arm] = torch.tensor(PANDA_REST_QPOS, dtype=torch.float32)
        raw.agent.robot.set_qpos(qpos)
        rest_action = torch.zeros(1, n_arm, device=device)
        rest_action[0, :n_arm] = torch.tensor(PANDA_REST_QPOS, dtype=torch.float32, device=device)
        for _ in range(10):
            env.step(rest_action)
        del rest_action

        # Collision world: one solid box for the entire shelf.
        # Simpler, fewer collision primitives, arm routes around it.
        from curobo.geom.types import Cuboid, WorldConfig
        g = raw.shelf_geom
        robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
        sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
        sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
        world_shelf = WorldConfig(cuboid=[Cuboid(
            name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
        )])
        motion_gen = setup_curobo_planner(
            ROBOT_UID, world_configs=world_shelf, n_envs=1,
        )

        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda_device).unsqueeze(0)
        q_tensor = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda_device)

        # Recording
        from taskbench.envs import get_objects
        objects = get_objects(env)
        recorder = StateRecorder(
            env, objects=objects,
            robot_fields=["qpos", "tcp_pos", "tcp_quat"],
        )
        step_cb = lambda t, obs, rew: recorder.record()
        recorder.record()

        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        g = raw.shelf_geom
        max_attempts = 50

        from curobo.types.math import Pose as CuPose
        from taskbench.skills.curobo_motion import _world_to_curobo_frame

        start_state = _sapien_qpos_to_cu_joint_state(env, ROBOT_UID, 1)
        logger.info("Start qpos: %s", start_state.position.cpu().tolist())
        valid, status = motion_gen.check_start_state(start_state)
        logger.info("Start state valid: %s, status: %s", valid, status)

        for attempt in range(max_attempts):
            p1, p2 = _sample_push(rng, g, margin=self.margin)
            approach_p = [g.front_x - 0.05, p1[1], p1[2]]

            # === PHASE 1: MotionGen staging (rest -> approach point) ===
            approach_pos = torch.tensor([approach_p], dtype=torch.float32, device=cuda_device)
            approach_base = _world_to_curobo_frame(approach_pos, robot_base_pos)
            goal_pose = CuPose(position=approach_base, quaternion=q_tensor)

            plan_config = MotionGenPlanConfig(
                max_attempts=10, enable_graph=False, timeout=15.0,
                enable_opt=True,
            )

            import time
            t0 = time.time()
            result = motion_gen.plan_single(start_state, goal_pose, plan_config)
            dt = time.time() - t0
            logger.info("  attempt %d: plan_single took %.3fs, success=%s",
                        attempt + 1, dt, result.success.cpu().numpy())

            if not result.success[0]:
                continue

            # Extract trajectory
            opt_plan = result.optimized_plan
            if opt_plan is None or opt_plan.position is None:
                logger.info("  attempt %d: no trajectory", attempt + 1)
                continue
            staging_traj = opt_plan.position.cpu()  # (T, 7)

            logger.info(
                "Push (attempt %d): approach=[%.2f, %.2f, %.2f] "
                "p1=[%.2f, %.2f, %.2f] -> p2=[%.2f, %.2f, %.2f]",
                attempt + 1, *approach_p, *p1, *p2,
            )
            break
        else:
            logger.info("  -> no feasible approach in %d attempts", max_attempts)
            return SolverResult(success=False, failure_reason="no_feasible_approach",
                                elapsed_steps=len(recorder.frames))

        # Visualize markers
        import sapien
        import sapien.render
        green_mat = sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
        red_mat = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        for pt, mat, name in [(p1, red_mat, "marker_start"),
                               (p2, green_mat, "marker_end")]:
            b = raw.scene.create_actor_builder()
            b.add_sphere_visual(radius=0.015, material=mat)
            b.initial_pose = sapien.Pose(p=pt)
            b.build_static(name=name)

        recorder.record_skill_call("push", {
            "approach_pose": approach_p, "push_start": p1, "push_end": p2,
        })

        # === EXECUTION ===

        # Phase 1: Staging (MotionGen trajectory)
        batched_follow_path(
            env, [staging_traj],
            gripper_state=None, n_envs=1, n_arm_joints=n_arm,
            step_callback=step_cb, refine_steps=10,
        )

        # Phase 2+3: Entry + Sweep via EE delta pose control.
        # Switch from pd_joint_pos to pd_ee_delta_pose — the Jacobian controller
        # is continuous (no IK branch flips) and maintains stick orientation.
        raw.agent.set_control_mode("pd_ee_delta_pose")
        raw.agent.controller.reset()

        def _ee_move_to(target_pos, steps, max_delta=0.03, phase_name="move"):
            """Drive TCP toward target_pos with zero rotation delta."""
            tcp_start = raw.agent.tcp.pose.p[0].cpu().numpy()
            q_start = raw.agent.tcp.pose.q[0].cpu().numpy()
            logger.info("  %s: start TCP=[%.3f, %.3f, %.3f] target=[%.3f, %.3f, %.3f] "
                        "dist=%.3f quat=[%.3f, %.3f, %.3f, %.3f]",
                        phase_name, *tcp_start, *target_pos,
                        np.linalg.norm(target_pos - tcp_start), *q_start)
            n_steps = 0
            for i in range(steps):
                tcp = raw.agent.tcp.pose.p[0].cpu().numpy()
                dist = np.linalg.norm(target_pos - tcp)
                if dist < 0.003:
                    logger.info("  %s: converged at step %d, dist=%.4f", phase_name, i, dist)
                    break
                delta = np.clip(target_pos - tcp, -max_delta, max_delta)
                action = torch.zeros(1, 6, device=device)
                action[0, 0] = delta[0]
                action[0, 1] = delta[1]
                action[0, 2] = delta[2]
                # rotation deltas = 0 → maintain current orientation
                obs, rew, _, _, _ = env.step(action)
                n_steps += 1
                if step_cb:
                    step_cb(0, obs, rew)
                if i % 20 == 0:
                    q_now = raw.agent.tcp.pose.q[0].cpu().numpy()
                    qpos = raw.agent.robot.get_qpos()[0, :7].cpu().numpy()
                    logger.info("  %s step %3d: TCP=[%.3f, %.3f, %.3f] dist=%.3f "
                                "quat=[%.3f, %.3f, %.3f, %.3f] "
                                "joints=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                                phase_name, i, *tcp, dist, *q_now, *qpos)

            tcp_end = raw.agent.tcp.pose.p[0].cpu().numpy()
            q_end = raw.agent.tcp.pose.q[0].cpu().numpy()
            final_dist = np.linalg.norm(target_pos - tcp_end)
            logger.info("  %s: done in %d steps, TCP=[%.3f, %.3f, %.3f] "
                        "final_dist=%.4f quat=[%.3f, %.3f, %.3f, %.3f]",
                        phase_name, n_steps, *tcp_end, final_dist, *q_end)
            return n_steps

        # Log staging endpoint
        staging_tcp = raw.agent.tcp.pose.p[0].cpu().numpy()
        staging_quat = raw.agent.tcp.pose.q[0].cpu().numpy()
        logger.info("After staging: TCP=[%.3f, %.3f, %.3f] quat=[%.3f, %.3f, %.3f, %.3f]",
                     *staging_tcp, *staging_quat)

        # Entry: approach -> p1 (stick enters shelf)
        _ee_move_to(np.array(p1), steps=100, phase_name="entry")

        # Sweep: p1 -> p2 (push inside shelf)
        _ee_move_to(np.array(p2), steps=200, max_delta=0.03, phase_name="sweep")

        # Settle
        logger.info("  Settling for %d steps...", self.settle_steps)
        for i in range(self.settle_steps):
            env.step(torch.zeros(1, 6, device=device))
            if step_cb:
                step_cb(0, None, None)
        settle_tcp = raw.agent.tcp.pose.p[0].cpu().numpy()
        logger.info("  After settle: TCP=[%.3f, %.3f, %.3f]", *settle_tcp)

        logger.info("  -> OK")

        solver_result = SolverResult(
            success=True,
            elapsed_steps=len(recorder.frames),
        )
        self.save_recording(recorder, seed, solver_result)
        return solver_result
