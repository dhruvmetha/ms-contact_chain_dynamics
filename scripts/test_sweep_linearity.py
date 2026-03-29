"""Check how straight the joint-space interpolation is in Cartesian space.

Solves IK at start and end of a push, interpolates joints, computes FK
at each step, and measures deviation from the straight Cartesian line.

Run on GPU:
    srun --gres=gpu:a4000:1 --partition=unlimited --nodelist=rlab4 --time=00:10:00 \
        bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH && \
        uv run python scripts/test_sweep_linearity.py'
"""
import torch
import numpy as np

from taskbench.skills.curobo_motion import (
    setup_curobo_planner,
    get_arm_joint_names,
    get_curobo_base_rotation,
    _world_to_curobo_frame,
)
from curobo.types.math import Pose as CuPose

cuda = torch.device("cuda:0")
robot_uid = "ur5e_robotiq"
n_arm = len(get_arm_joint_names(robot_uid))
base_rot = get_curobo_base_rotation(robot_uid)

Q_INTO_SHELF = [0.7071, 0, -0.7071, 0]
mg = setup_curobo_planner(robot_uid, world_configs=None, n_envs=1)

# Test several push distances and directions
pushes = [
    ("short_lateral",  [0.55, -0.05, 0.36], [0.55,  0.05, 0.36]),  # 10cm lateral
    ("long_lateral",   [0.55, -0.15, 0.36], [0.55,  0.15, 0.36]),  # 30cm lateral
    ("short_depth",    [0.52,  0.0,  0.36], [0.60,  0.0,  0.36]),  # 8cm depth
    ("long_depth",     [0.52,  0.0,  0.36], [0.68,  0.0,  0.36]),  # 16cm depth
    ("diagonal",       [0.52, -0.10, 0.36], [0.62,  0.10, 0.36]),  # diagonal
    ("short_5cm",      [0.55,  0.0,  0.36], [0.60,  0.0,  0.36]),  # 5cm
]

N_INTERP = 80  # same as solver

for name, p1, p2 in pushes:
    # Transform to cuRobo frame
    p1_t = torch.tensor([p1], dtype=torch.float32, device=cuda)
    p2_t = torch.tensor([p2], dtype=torch.float32, device=cuda)
    p1_base = _world_to_curobo_frame(p1_t, robot_base_rotation=base_rot)
    p2_base = _world_to_curobo_frame(p2_t, robot_base_rotation=base_rot)
    q_t = torch.tensor([Q_INTO_SHELF], dtype=torch.float32, device=cuda)

    # IK at start
    goal1 = CuPose(position=p1_base, quaternion=q_t)
    r1 = mg.ik_solver.solve_batch(goal1)
    if not r1.success[0]:
        print(f"{name:20s}: start IK FAILED")
        continue
    j1 = r1.solution[0, 0]

    # IK at end, seeded with start
    goal2 = CuPose(position=p2_base, quaternion=q_t)
    r2 = mg.ik_solver.solve_batch(goal2, seed_config=j1.unsqueeze(0).unsqueeze(0))
    if not r2.success[0]:
        print(f"{name:20s}: end IK FAILED")
        continue
    j2 = r2.solution[0, 0]

    # Joint diff
    joint_diff = (j1 - j2).abs().max().item()

    # Interpolate in joint space
    alphas = torch.linspace(0, 1, N_INTERP).unsqueeze(1).to(cuda)
    path = j1.unsqueeze(0) + alphas * (j2 - j1).unsqueeze(0)

    # FK at each step
    kin = mg.kinematics.get_state(path)
    ee_pos = kin.ee_position.cpu().numpy()  # (N_INTERP, 3) in cuRobo frame

    # Convert back to world frame (undo Rz180)
    ee_world = ee_pos.copy()
    ee_world[:, 0] = -ee_pos[:, 0]
    ee_world[:, 1] = -ee_pos[:, 1]

    # Ideal straight line: p1 to p2
    p1_np = np.array(p1)
    p2_np = np.array(p2)
    push_vec = p2_np - p1_np
    push_len = np.linalg.norm(push_vec)
    push_dir = push_vec / push_len

    # Deviation from straight line at each step
    deviations = []
    for i in range(N_INTERP):
        # Project onto line
        v = ee_world[i] - p1_np
        proj = np.dot(v, push_dir)
        closest = p1_np + proj * push_dir
        dev = np.linalg.norm(ee_world[i] - closest)
        deviations.append(dev)

    max_dev = max(deviations)
    mean_dev = np.mean(deviations)

    # Also check start/end accuracy
    start_err = np.linalg.norm(ee_world[0] - p1_np)
    end_err = np.linalg.norm(ee_world[-1] - p2_np)

    print(f"{name:20s}: push={push_len*100:.0f}cm  joint_diff={joint_diff:.2f}rad  "
          f"max_dev={max_dev*1000:.1f}mm  mean_dev={mean_dev*1000:.1f}mm  "
          f"start_err={start_err*1000:.1f}mm  end_err={end_err*1000:.1f}mm")

print("\nDone")
