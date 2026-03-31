"""Shelf push solver using the Panda Stick + StickPush skill.

Samples random push targets inside the shelf, delegates execution
to the StickPush skill (stage → insert → sweep → retract → rest).

Usage:
    uv run python -m taskbench.run task=shelf_panda_stick solver=shelf_panda_stick_push_curobo
"""

import logging
from pathlib import Path

import numpy as np
import torch

from taskbench.recorder import StateRecorder
from taskbench.skills.curobo_motion import (
    get_arm_joint_names,
    setup_curobo_planner,
)
from taskbench.skills.stick_push import StickPush
from taskbench.solver import BaseSolver, SolverResult, register_solver

logger = logging.getLogger("taskbench.solvers.shelf_panda_stick_push_curobo")

ROBOT_UID = "panda_stick_long"
Q_INTO_SHELF = [0.5, -0.5, 0.5, -0.5]
REST_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
SAFE_QPOS = [0.0002, -1.2996, 0.0003, -2.4930, -0.0009, 1.0177, 0.0]

import taskbench.agents.panda_stick_long  # noqa: F401


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
    """One random push per episode using the StickPush skill."""

    def __init__(self, settle_steps: int = 60, margin: float = 0.03):
        self.settle_steps = int(settle_steps)
        self.margin = float(margin)

    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        env.reset(seed=seed)
        raw = env.unwrapped
        rng = np.random.default_rng(seed)

        device = raw.device
        cuda_device = torch.device("cuda:0")
        n_arm = len(get_arm_joint_names(ROBOT_UID))

        # Set rest qpos + settle
        qpos = raw.agent.robot.get_qpos().clone()
        qpos[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32)
        raw.agent.robot.set_qpos(qpos)
        rest_action = torch.zeros(1, n_arm, device=device)
        rest_action[0, :n_arm] = torch.tensor(REST_QPOS, dtype=torch.float32, device=device)
        for _ in range(10):
            env.step(rest_action)

        # Collision world: solid shelf box
        from curobo.geom.types import Cuboid, WorldConfig
        g = raw.shelf_geom
        robot_base = raw.agent.robot.pose.p[0].cpu().numpy()
        sc = np.array([g.center_x, 0.0, g.ceil_z / 2]) - robot_base[:3]
        sd = [g.depth + 0.04, 2 * g.half_w + 0.04, g.ceil_z + 0.04]
        world_shelf = WorldConfig(cuboid=[Cuboid(
            name="shelf", pose=[float(sc[0]), float(sc[1]), float(sc[2]), 1, 0, 0, 0], dims=sd,
        )])
        motion_gen = setup_curobo_planner(ROBOT_UID, world_configs=world_shelf, n_envs=1)
        robot_base_pos = raw.agent.robot.pose.p[0].to(device=cuda_device).unsqueeze(0)

        # Recording
        from taskbench.envs import get_objects
        objects = get_objects(env)
        recorder = StateRecorder(
            env, objects=objects,
            robot_fields=["qpos", "tcp_pos", "tcp_quat"],
        )
        step_cb = lambda t, obs, rew: recorder.record()
        recorder.record()

        # Create skill
        push_skill = StickPush(
            env, motion_gen,
            robot_uid=ROBOT_UID, n_envs=1, n_arm_joints=n_arm,
            rest_qpos=REST_QPOS, robot_base_pos=robot_base_pos,
            safe_start_qpos=SAFE_QPOS,
        )

        # Sample push targets
        p1, p2 = _sample_push(rng, g, margin=self.margin)
        approach_p = [g.front_x - 0.05, p1[1], p1[2]]
        retract_p = [g.front_x - 0.05, p2[1], p2[2]]

        logger.info("Push: approach=[%.2f, %.2f, %.2f] p1=[%.2f, %.2f, %.2f] "
                     "-> p2=[%.2f, %.2f, %.2f]", *approach_p, *p1, *p2)

        # Visualize markers
        import sapien
        import sapien.render
        red_mat = sapien.render.RenderMaterial(base_color=[0.9, 0.1, 0.1, 1])
        green_mat = sapien.render.RenderMaterial(base_color=[0.1, 0.9, 0.1, 1])
        for pt, mat, name in [(p1, red_mat, "marker_start"),
                               (p2, green_mat, "marker_end")]:
            b = raw.scene.create_actor_builder()
            b.add_sphere_visual(radius=0.015, material=mat)
            b.initial_pose = sapien.Pose(p=pt)
            b.build_static(name=name)

        recorder.record_skill_call("push", {
            "approach_pose": approach_p, "push_start": p1, "push_end": p2,
        })

        # Execute push
        result = push_skill(
            approach_positions=torch.tensor([approach_p], dtype=torch.float32, device=cuda_device),
            approach_quaternions=torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda_device),
            entry_positions=torch.tensor([p1], dtype=torch.float32, device=cuda_device),
            sweep_positions=torch.tensor([p2], dtype=torch.float32, device=cuda_device),
            retract_positions=torch.tensor([retract_p], dtype=torch.float32, device=cuda_device),
            rest_steps=self.settle_steps,
            step_callback=step_cb,
        )

        success = result.success_mask[0].item()
        logger.info("  -> %s (orient_drift=%.4f, sweep_dist=%.4f)",
                     "OK" if success else "FAIL",
                     result.orientation_drift[0].item(),
                     result.sweep_final_dist[0].item())

        solver_result = SolverResult(
            success=success,
            elapsed_steps=result.steps_executed,
        )
        self.save_recording(recorder, seed, solver_result)
        return solver_result
