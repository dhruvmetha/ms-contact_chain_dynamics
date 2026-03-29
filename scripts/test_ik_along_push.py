"""Test IK along a push line to see where it fails."""
import torch
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, get_curobo_base_rotation, _world_to_curobo_frame
)
from curobo.types.math import Pose as CuPose

cuda = torch.device("cuda:0")
base_rot = get_curobo_base_rotation("ur5e_robotiq")

mg = setup_curobo_planner("ur5e_robotiq", world_configs=None, n_envs=1)

p1 = [0.57, -0.01, 0.36]
p2 = [0.62, 0.12, 0.36]
Q = [0.7071, 0, -0.7071, 0]

N = 20
positions = []
for i in range(N):
    a = i / (N - 1)
    positions.append([p1[j] + a * (p2[j] - p1[j]) for j in range(3)])

# Batch IK (no seeding)
pos_t = torch.tensor(positions, dtype=torch.float32, device=cuda)
pos_base = _world_to_curobo_frame(pos_t, robot_base_rotation=base_rot)
quats = torch.tensor([Q] * N, dtype=torch.float32, device=cuda)
goal = CuPose(position=pos_base, quaternion=quats)

result = mg.ik_solver.solve_batch(goal)
success = result.success.squeeze().cpu().numpy()

print("Batch IK (no seeding):")
for i in range(N):
    p = positions[i]
    status = "OK" if success[i] else "FAIL"
    print(f"  wp{i:2d}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] {status}")
print(f"Total: {success.sum()}/{N}")

# Sequential IK (seeded)
print("\nSequential IK (seeded):")
prev = None
n_ok = 0
for i in range(N):
    pos_i = torch.tensor([positions[i]], dtype=torch.float32, device=cuda)
    pos_i_base = _world_to_curobo_frame(pos_i, robot_base_rotation=base_rot)
    q_i = torch.tensor([Q], dtype=torch.float32, device=cuda)
    goal_i = CuPose(position=pos_i_base, quaternion=q_i)

    if prev is not None:
        seed = prev.unsqueeze(0).unsqueeze(0)
        r = mg.ik_solver.solve_batch(goal_i, seed_config=seed)
    else:
        r = mg.ik_solver.solve_batch(goal_i)

    ok = r.success[0].item()
    if ok:
        prev = r.solution[0, 0]
        n_ok += 1

    p = positions[i]
    print(f"  wp{i:2d}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] {'OK' if ok else 'FAIL'}")

print(f"Total: {n_ok}/{N}")
