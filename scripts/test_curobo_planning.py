"""Debug cuRobo planning around shelf bottom."""
import torch
import gymnasium as gym
import taskbench.envs
from taskbench.skills.curobo_motion import (
    setup_curobo_planner, _sapien_qpos_to_cu_joint_state, batched_move_to_pose,
    get_arm_joint_names,
)
from taskbench.skills.curobo_world import shelf_world
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

cuda = torch.device("cuda:0")
env = gym.make("ShelfEnv-v1", num_envs=1, sim_backend="cpu",
               robot_uids="ur5e_robotiq", robot_base_pose=[0,0,0,0,0,0,1],
               num_objects=0, render_mode=None, obs_mode="state",
               control_mode="pd_joint_pos", reward_mode="none", open_top=True)
env.reset(seed=42)
raw = env.unwrapped
g = raw.shelf_geom

# Full shelf world
world = shelf_world(env, n_envs=1, include_objects=False)[0]

# Use setup_curobo_planner (handles path resolution)
mg = setup_curobo_planner("ur5e_robotiq", world_configs=world, n_envs=1)
print("cuRobo warmed up")

start = _sapien_qpos_to_cu_joint_state(env, "ur5e_robotiq", 1)
start.position = start.position.to(cuda)
print(f"Start qpos: {start.position}")

Q = [0.5, 0.5, 0.5, 0.5]
q = torch.tensor([Q], dtype=torch.float32, device=cuda)

# Use plan_single with graph search enabled
plan_config = MotionGenPlanConfig(enable_graph=True, max_attempts=10)

# Test various targets with plan_single directly
targets = [
    ("in_front", [0.3, 0.0, 0.4]),     # well below shelf, in front
    ("below_shelf", [0.4, 0.0, 0.4]),   # below shelf height
    ("shelf_height", [0.4, 0.0, 0.7]),  # at shelf interior height
    ("staging", [g.front_x - 0.05, 0.0, 0.7]),  # just outside shelf
    ("inside", [g.front_x + 0.05, 0.0, 0.7]),   # just inside
]

print("\n=== Direct plan_single tests ===")
for name, pos in targets:
    goal = CuPose(
        position=torch.tensor([pos], dtype=torch.float32, device=cuda),
        quaternion=q,
    )
    result = mg.plan_single(start, goal, plan_config)
    ok = result.success.item()
    print(f"  {name:15s} pos={pos} -> {'OK' if ok else 'FAIL'} status={result.status}")

# Two-step approach: first go below shelf, then into the opening
print("\n=== Two-step: below shelf -> staging -> inside ===")
# Step 1: go to a point below the shelf (no obstruction)
below_goal = CuPose(
    position=torch.tensor([[0.4, 0.0, 0.4]], dtype=torch.float32, device=cuda),
    quaternion=q,
)
r1 = mg.plan_single(start, below_goal, plan_config)
print(f"  Step 1 (below shelf): {'OK' if r1.success.item() else 'FAIL'}")

if r1.success:
    # Step 2: from below, go to staging
    end_joints = r1.interpolated_plan.position[-1].unsqueeze(0)
    state2 = CuJointState.from_position(end_joints, joint_names=get_arm_joint_names("ur5e_robotiq"))
    staging_goal = CuPose(
        position=torch.tensor([[g.front_x - 0.05, 0.0, 0.7]], dtype=torch.float32, device=cuda),
        quaternion=q,
    )
    r2 = mg.plan_single(state2, staging_goal, plan_config)
    print(f"  Step 2 (staging):     {'OK' if r2.success.item() else 'FAIL'} status={r2.status}")

    if r2.success:
        # Step 3: from staging, go inside
        end_joints2 = r2.interpolated_plan.position[-1].unsqueeze(0)
        state3 = CuJointState.from_position(end_joints2, joint_names=get_arm_joint_names("ur5e_robotiq"))
        inside_goal = CuPose(
            position=torch.tensor([[g.front_x + 0.05, 0.0, 0.7]], dtype=torch.float32, device=cuda),
            quaternion=q,
        )
        r3 = mg.plan_single(state3, inside_goal, plan_config)
        print(f"  Step 3 (inside):      {'OK' if r3.success.item() else 'FAIL'} status={r3.status}")

env.close()
