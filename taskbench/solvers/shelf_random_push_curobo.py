"""Random push solver for the shelf environment using cuRobo.

Three-phase motion:
  1. Staging: cuRobo collision-aware path from home → shelf opening
  2. Entry: Cartesian straight-line staging → p1, collision-checked (shelf + objects)
  3. Sweep: Cartesian straight-line p1 → p2, collision-checked (shelf only, push through objects)

Backward search: verify all IK during planning, reject infeasible samples fast.

Usage:
    uv run python -m taskbench.run task=shelf solver=shelf_random_push_curobo
"""

import logging
import os

import numpy as np
import torch

from taskbench.recorder import StateRecorder
from taskbench.skills.curobo_motion import (
    batched_actuate_gripper,
    batched_follow_path,
    batched_move_to_pose,
    get_arm_joint_names,
    setup_curobo_planner,
)
from taskbench.skills.curobo_world import shelf_world
from taskbench.skills.robot_config import get_robot_config
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.shelf_random_push_curobo")

# EE orientation for horizontal forward reach into shelf, wxyz
Q_INTO_SHELF = [0.7071068, 0.0, -0.7071068, 0.0]


def _sample_push(rng, shelf_geom, margin=0.03, y_margin=0.03, depth_frac=0.8,
                 z_offset=0.05):
    """Sample start and end points for a push at a fixed height."""
    g = shelf_geom
    max_x = g.front_x + g.depth * depth_frac
    z = g.surface_z + z_offset

    x1 = rng.uniform(g.front_x + margin, max_x)
    y1 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)

    x2 = rng.uniform(g.front_x + margin, max_x)
    y2 = rng.uniform(-g.half_w + y_margin, g.half_w - y_margin)

    return [x1, y1, z], [x2, y2, z]


@register_solver("shelf_random_push_curobo")
class ShelfRandomPushCuroboSolver(BaseSolver):
    """One random push per episode using cuRobo for planning."""

    def __init__(
        self,
        settle_steps: int = 60,
        margin: float = 0.03,
        sweep_steps: int = 80,
    ):
        self.settle_steps = int(settle_steps)
        self.margin = float(margin)
        self.sweep_steps = int(sweep_steps)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        raw = env.unwrapped
        rc = get_robot_config(env)
        rng = np.random.default_rng(seed)
        robot_uid = raw.agent.__class__.__name__
        if "UR5e" in robot_uid or "ur5e" in robot_uid:
            robot_uid = "ur5e_robotiq"
        elif "Panda" in robot_uid or "panda" in robot_uid:
            robot_uid = "panda"
        else:
            robot_uid = cfg.task.robot_uids if cfg else "panda"

        device = raw.device
        cuda_device = torch.device("cuda:0")
        n_arm = len(get_arm_joint_names(robot_uid))

        # Three collision worlds:
        # 1. shelf + objects: for staging (avoid everything)
        # 2. shelf + objects: for entry IK (avoid everything)
        # 3. shelf only: for sweep IK (push through objects)
        world_full = shelf_world(env, n_envs=1, include_objects=True)[0]
        world_shelf = shelf_world(env, n_envs=1, include_objects=False)[0]

        # Three planners:
        # - motion_gen_full: collision-aware (shelf + objects) for staging path planning
        # - motion_gen_shelf: collision-aware (shelf only) for sweep IK
        # - motion_gen_entry: collision-aware (shelf + objects) for entry IK
        motion_gen_full = setup_curobo_planner(
            robot_uid, world_configs=world_full, n_envs=1,
        )
        motion_gen_shelf = setup_curobo_planner(
            robot_uid, world_configs=world_shelf, n_envs=1,
        )

        # Robot base position
        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda_device).unsqueeze(0)
        q_tensor = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda_device)

        # Recording
        from taskbench.envs import get_objects
        objects = get_objects(env)
        recorder = StateRecorder(
            env, objects=objects,
            robot_fields=["qpos", "tcp_pos", "tcp_quat", "gripper_contact_force"],
        )
        step_cb = lambda t, obs, rew: recorder.record()
        recorder.record()

        # Close gripper
        batched_actuate_gripper(
            env, rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
            step_callback=step_cb,
        )

        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        from taskbench.skills.curobo_motion import (
            _sapien_qpos_to_cu_joint_state, _world_to_curobo_frame,
        )
        from curobo.types.math import Pose as CuPose

        staging_plan_config = MotionGenPlanConfig(
            max_attempts=10, enable_graph=True, timeout=10.0,
        )

        N_PUSH_WP = 20
        N_ENTRY_WP = 10

        g = raw.shelf_geom
        max_attempts = 50
        p1, p2 = None, None
        staging_result = None
        push_joint_path = None
        entry_joint_path = None

        for attempt in range(max_attempts):
            p1, p2 = _sample_push(rng, g, margin=self.margin)
            staging_p = [g.front_x - 0.05, p1[1], p1[2]]

            # === PHASE 3 first (backward): Sweep IK (shelf collision only) ===
            p1_t = torch.tensor(p1, dtype=torch.float32, device=cuda_device)
            p2_t = torch.tensor(p2, dtype=torch.float32, device=cuda_device)
            push_alphas = torch.linspace(0, 1, N_PUSH_WP, device=cuda_device)
            push_positions = p1_t.unsqueeze(0) + push_alphas.unsqueeze(1) * (p2_t - p1_t).unsqueeze(0)

            push_joint_path = []
            prev = None
            push_ok = True
            for wi in range(N_PUSH_WP):
                wp = push_positions[wi].unsqueeze(0)
                wp_base = _world_to_curobo_frame(wp, robot_base_pos)
                goal = CuPose(position=wp_base, quaternion=q_tensor)

                if prev is not None:
                    seed = prev.unsqueeze(0).unsqueeze(0)
                    result = motion_gen_shelf.ik_solver.solve_batch(goal, seed_config=seed)
                else:
                    result = motion_gen_shelf.ik_solver.solve_batch(goal)

                if not result.success[0]:
                    result = motion_gen_shelf.ik_solver.solve_batch(goal)

                if not result.success[0]:
                    push_ok = False
                    break

                joints = result.solution[0, 0].to(device=cuda_device, dtype=torch.float32)
                push_joint_path.append(joints.unsqueeze(0))
                prev = joints

            if not push_ok:
                logger.info("  attempt %d: sweep IK failed at wp %d/%d",
                            attempt + 1, wi + 1, N_PUSH_WP)
                continue

            p1_joints = push_joint_path[0]

            # === PHASE 2 (backward): Entry IK (shelf + objects collision) ===
            stg_t = torch.tensor(staging_p, dtype=torch.float32, device=cuda_device)
            entry_alphas = torch.linspace(0, 1, N_ENTRY_WP, device=cuda_device)
            entry_positions = stg_t.unsqueeze(0) + entry_alphas.unsqueeze(1) * (p1_t - stg_t).unsqueeze(0)

            entry_joint_path = []
            prev = p1_joints[0]
            entry_ok = True

            # Solve in reverse (from p1 back to staging) for better seeding
            for wi in range(N_ENTRY_WP - 1, -1, -1):
                wp = entry_positions[wi].unsqueeze(0)
                wp_base = _world_to_curobo_frame(wp, robot_base_pos)
                goal = CuPose(position=wp_base, quaternion=q_tensor)

                seed = prev.unsqueeze(0).unsqueeze(0)
                result = motion_gen_full.ik_solver.solve_batch(goal, seed_config=seed)

                if not result.success[0]:
                    result = motion_gen_full.ik_solver.solve_batch(goal)

                if not result.success[0]:
                    entry_ok = False
                    break

                joints = result.solution[0, 0].to(device=cuda_device, dtype=torch.float32)
                entry_joint_path.append(joints.unsqueeze(0))
                prev = joints

            if not entry_ok:
                logger.info("  attempt %d: entry IK failed at wp %d/%d",
                            attempt + 1, N_ENTRY_WP - wi, N_ENTRY_WP)
                continue

            entry_joint_path = entry_joint_path[::-1]  # reverse to staging→p1 order

            # === PHASE 1 (backward): Staging path (collision-aware planning) ===
            start_state = _sapien_qpos_to_cu_joint_state(env, robot_uid, 1)
            stg_pos = torch.tensor([staging_p], dtype=torch.float32, device=cuda_device)

            staging_result = batched_move_to_pose(
                motion_gen_full, start_state,
                stg_pos, q_tensor,
                n_envs=1, use_batch_env=False,
                robot_base_position=robot_base_pos,
                plan_config=staging_plan_config,
            )
            if not staging_result["success"][0]:
                logger.info("  attempt %d: staging plan failed", attempt + 1)
                continue

            logger.info(
                "Push (attempt %d): [%.2f, %.2f, %.2f] -> [%.2f, %.2f, %.2f] "
                "(entry %d/%d, sweep %d/%d IK OK)",
                attempt + 1, *p1, *p2,
                N_ENTRY_WP, N_ENTRY_WP, N_PUSH_WP, N_PUSH_WP,
            )
            break
        else:
            logger.info("  -> no feasible push found in %d attempts", max_attempts)
            return SolverResult(success=False, failure_reason="no_feasible_push",
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
            "approach_pose": p1, "push_pose": p2,
        })

        # === EXECUTION: replay pre-computed joint paths ===

        # Phase 1: Staging (collision-free planned path)
        batched_follow_path(
            env, staging_result["trajectories"],
            rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
            step_callback=step_cb, refine_steps=10,
        )

        # Phase 2: Entry (pre-computed Cartesian joint path)
        entry_traj = torch.cat(entry_joint_path, dim=0)
        batched_follow_path(
            env, [entry_traj],
            rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
            step_callback=step_cb, refine_steps=5,
        )

        # Phase 3: Sweep (pre-computed Cartesian joint path)
        push_traj = torch.cat(push_joint_path, dim=0)
        batched_follow_path(
            env, [push_traj],
            rc.gripper_closed, n_envs=1, n_arm_joints=n_arm,
            step_callback=step_cb, refine_steps=self.settle_steps,
        )

        logger.info("  -> OK")

        # Save video
        if hasattr(env, 'render_images') and len(env.render_images) > 1:
            from mani_skill.utils.visualization.misc import images_to_video
            video_dir = cfg.runtime.get("video_dir", "videos") if cfg else "videos"
            if not os.path.isabs(video_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                video_dir = os.path.join(project_root, video_dir)
            os.makedirs(video_dir, exist_ok=True)
            import time
            video_name = f"push_{int(time.time())}"
            images_to_video(env.render_images, video_dir, video_name, fps=30, verbose=True)
            logger.info("  Video saved: %s/%s.mp4 (%d frames)",
                         video_dir, video_name, len(env.render_images))

        solver_result = SolverResult(
            success=True,
            elapsed_steps=len(recorder.frames),
        )
        self.save_recording(recorder, seed, solver_result, cfg=cfg)
        return solver_result
