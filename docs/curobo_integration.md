# cuRobo Integration Reference

How cuRobo is used for GPU-batched motion planning in robotics simulation (Isaac Sim, ManiSkill).

## cuRobo Core API

### MotionGen — Trajectory Planning

The main entry point. Create once (warmup is expensive), reuse across episodes.

```python
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState

# Configure
config = MotionGenConfig.load_from_robot_config(
    "franka.yml",              # robot config (built-in or custom dict/path)
    world_model,               # WorldConfig, list[WorldConfig], dict, or None
    interpolation_dt=0.01,     # trajectory interpolation timestep
    n_collision_envs=N,        # number of parallel collision worlds
    use_cuda_graph=False,      # disable if world changes dynamically
    ee_link_name="ee_link",    # end-effector frame (overrides robot config default)
)

motion_gen = MotionGen(config)
motion_gen.warmup(batch=N)     # CUDA kernel compilation — expensive, do once
```

**Three planning modes:**

| Method | Use case | World |
|---|---|---|
| `plan_single(start, goal, cfg)` | One trajectory | Single world |
| `plan_batch(start, goal, cfg)` | N trajectories | Same world for all |
| `plan_batch_env(start, goal, cfg)` | N trajectories | Different world per env |

```python
# plan_batch_env: each env has its own collision world
start_state = JointState.from_position(
    qpos_tensor,  # (N, n_arm) float32 CUDA
    joint_names=["joint1", "joint2", ...],
)
goal_pose = Pose(position=pos_tensor, quaternion=quat_tensor)  # wxyz order

plan_config = MotionGenPlanConfig(max_attempts=4, enable_graph=False)
result = motion_gen.plan_batch_env(start_state, goal_pose, plan_config)

result.success          # (N,) bool tensor
result.optimized_plan   # .position is (N, T, n_arm) joint trajectory
```

**Updating world between episodes** (cheaper than recreating MotionGen):

```python
# All envs same world:
motion_gen.update_world(new_world_config)  # single WorldConfig

# Per-env different worlds (batch collision update):
motion_gen.world_coll_checker.load_batch_collision_model([wc1, wc2, ...])
motion_gen.graph_planner.reset_buffer()  # invalidate cached roadmaps

# Per-env single env update:
motion_gen.world_coll_checker.load_collision_model(wc, env_idx=i)
```

### WorldConfig — Collision Geometry

Defines obstacles for collision avoidance. Supports cuboids, meshes, capsules, cylinders, spheres.

```python
from curobo.geom.types import WorldConfig, Cuboid, Mesh

# Cuboid: pose is [x, y, z, qw, qx, qy, qz], dims are full extents
table = Cuboid(name="table", dims=[1.0, 0.6, 0.02], pose=[0, 0, 0.78, 1, 0, 0, 0])
world_config = WorldConfig(cuboid=[table])

# For batched envs (N different worlds):
# Pass list[WorldConfig] to MotionGenConfig.load_from_robot_config
config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    [world_config_1, world_config_2, ...],  # one per env
    n_collision_envs=N,
)

# Mesh obstacles (from OBJ files)
mesh = Mesh(name="scene", file_path="scene.obj", pose=[...], scale=[1, 1, 1])
world_config = WorldConfig(mesh=[mesh], cuboid=[table])
```

**Isaac Sim shortcut** — parse USD scene directly:

```python
from curobo.util.usd_helper import UsdHelper
usd_helper = UsdHelper(usd_file_path)
world_config = WorldConfig.from_usd_stage(usd_helper.stage)
```

### IKSolver — Batched Inverse Kinematics

Accessed through `motion_gen.ik_solver`. Solves N IK problems in parallel.

```python
goal_pose = Pose(position=positions, quaternion=quaternions)  # (N, 3), (N, 4) wxyz

# Seed with current joints to get nearby solutions
result = motion_gen.ik_solver.solve_batch(goal_pose, retract_config=seed_joints)

result.success      # (N,) bool
result.solution     # (N, num_seeds, n_arm) — best solution at [:, 0]
```

### JointState and Pose Types

```python
# JointState: always provide joint_names so cuRobo maps to its internal order
state = JointState.from_position(
    torch.zeros(N, 7).cuda(),
    joint_names=["panda_joint1", ..., "panda_joint7"],
)

# Pose: quaternion is wxyz (not xyzw)
pose = Pose(
    position=torch.tensor([[0.4, 0.0, 0.3]]).cuda(),
    quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).cuda(),
)
# or from a flat list [x, y, z, qw, qx, qy, qz]
pose = Pose.from_list([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
```

## Integration with ManiSkill

### Coordinate Frame Alignment

cuRobo plans in the **robot base frame**. ManiSkill uses a **world frame** where the robot base has a non-zero offset. Transform goals before planning:

```python
robot_base_pos = env.unwrapped.agent.robot.pose.p[0]  # (3,) world position of robot base
goal_in_robot_frame = goal_world_pos - robot_base_pos
```

### Extracting Joint State from ManiSkill

ManiSkill's `get_qpos()` returns all joints (arm + gripper). cuRobo only needs arm joints:

```python
qpos = env.unwrapped.agent.robot.get_qpos()  # (N, total_joints)
arm_qpos = qpos[:, :n_arm_joints]             # arm joints are first by convention

state = JointState.from_position(arm_qpos.float(), joint_names=arm_joint_names)
```

### Control Mode

Motion planning requires **position control** — use `pd_joint_pos` or `pd_joint_pos_vel`:

```python
env = gym.make("TaskEnv-v1", control_mode="pd_joint_pos", num_envs=256)
```

Action format: `[arm_joint_targets, gripper_action]` as `(N, n_arm + 1)` tensor.

For `pd_joint_pos_vel`: `[arm_pos, arm_vel, gripper]` as `(N, 2*n_arm + 1)`.

### Executing cuRobo Trajectories in ManiSkill

cuRobo returns waypoints; step through them with `env.step()`:

```python
trajectory = result.optimized_plan.position  # (N, T, n_arm)

for t in range(T):
    actions = torch.zeros(N, action_dim, device=device)
    actions[:, :n_arm] = trajectory[:, t]
    actions[:, -1] = gripper_state
    obs, rew, term, trunc, info = env.step(actions)
```

Add **refine steps** (holding the final waypoint) so the PD controller converges.

### Building WorldConfig from ManiSkill Scenes

ManiSkill scenes use SAPIEN actors. Convert to cuRobo cuboids manually:

```python
import sapien.physx as physx

comp = actor.find_component_by_type(physx.PhysxRigidStaticComponent)
shape = comp.get_collision_shapes()[0]
half_size = shape.half_size  # (3,)
world_pose = actor.pose * shape.get_local_pose()
center = world_pose.p  # (3,)

cuboid = Cuboid(
    name=actor.name,
    dims=(half_size * 2).tolist(),  # full extents
    pose=[*center, 1, 0, 0, 0],    # [x,y,z, qw,qx,qy,qz]
)
```

**Key rule**: Exclude target objects from the collision world so the planner can reach them.

## Robot Configuration

### Built-in vs Custom Configs

cuRobo ships configs for common robots (`franka.yml`, `ur5e.yml`, etc.). For custom setups:

```yaml
# configs/curobo/my_robot.yml
robot_cfg:
  kinematics:
    urdf_path: "path/to/robot.urdf"
    asset_root_path: "path/to/meshes/"
    base_link: "base_link"
    ee_link: "tool_frame"
    lock_joints:
      gripper_joint_1: 0.0  # lock during planning
      gripper_joint_2: 0.0
    cspace:
      joint_names: ["joint1", "joint2", ...]
      retract_config: [0, -1.57, 0, -1.57, 0, 0]  # home pose
      # velocity/acceleration/jerk limits per joint
  collision:
    collision_link_names: ["link1", "link2", ...]
    collision_spheres:
      link1:
        - center: [0, 0, 0]
          radius: 0.05
    self_collision_ignore:
      link1: ["link2"]  # skip adjacent link checks
```

### Gripper Handling

Lock gripper joints during planning (they're controlled separately):

```yaml
lock_joints:
  panda_finger_joint1: 0.04
  panda_finger_joint2: 0.04
```

Control grippers by holding arm joints and commanding only the gripper action:

```python
actions[:, :n_arm] = current_arm_qpos  # hold arm position
actions[:, -1] = gripper_state          # -1 closed, +1 open (ManiSkill convention)
```

## Performance Best Practices

1. **Warmup once, reuse MotionGen** — CUDA kernel compilation takes seconds. Create the planner once and persist it across episodes. Only call `update_world()` between episodes.

2. **Disable CUDA graphs when worlds change** — `use_cuda_graph=False` in MotionGenConfig. CUDA graphs require fixed computation graphs; dynamic world updates break this.

3. **Keep data on GPU** — avoid `.cpu()` per step. Accumulate tensors on GPU and do a single bulk transfer at episode end.

4. **Vectorize action construction** — pre-pad variable-length trajectories into a single `(N, T, n_arm)` tensor. Avoid per-env Python loops inside the step loop.

5. **Use `plan_batch_env` for per-env collision worlds** — even if all worlds are identical, this is the correct API when `n_collision_envs > 1`.

6. **Straight-line motions skip the planner** — for contact tasks (pushing), use joint-space linear interpolation instead of cuRobo's trajectory optimizer, which would arc around the target object.

7. **`interpolation_dt`** — controls trajectory density. Smaller values (0.002) give finer control but longer trajectories. Match to the simulation timestep for smooth execution.
