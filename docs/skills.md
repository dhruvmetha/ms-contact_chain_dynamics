# Skills

Skills are composable manipulation primitives. Use them through `SkillContext` — it bundles the env, motion planner, and object references so you only pass task-specific parameters.

## SkillContext

```python
from taskbench.skills.context import SkillContext

ctx = SkillContext(env, step_callback=recorder.record)
ctx.reset(seed=42)

# Skills are ready
pick_result = ctx.pick("cube_1", lift_height=0.15)
if pick_result.success:
    ctx.place(target_pose, retract_height=0.2)
```

`ctx.reset(seed)` handles: env reset, planner creation, object discovery, and skill re-initialization. You must call it before using any skills.

After reset, `ctx.objects` is a `dict[str, Actor]` mapping object names to SAPIEN actors (from the env's `get_objects()` method).

## Available Skills

### pick

```python
pick_result = ctx.pick(obj_name, *, lift_height=0.1, verify_grasp=True)
```

Grasp an object by name and lift it. Tries multiple grasp angles (6 candidates around the Z-axis) until one succeeds.

**Parameters:**
- `obj_name` — string name of the object (resolved via `ctx.objects`)
- `lift_height` — how high to lift after grasping (meters)
- `verify_grasp` — check that the object is actually held after closing fingers

**Returns:** `PickResult` with:
- `success`, `failure_reason`
- `grasp_pose` — the pose used for grasping
- `lift_pose` — the pose after lifting (useful for computing place targets)
- `obj_size` — bounding box of the grasped object

### place

```python
place_result = ctx.place(target_pose, *, settling_steps=10, retract_height=None)
```

Move to a target pose, release the object, wait for it to settle, then retract.

**Parameters:**
- `target_pose` — where to place. Accepts `sapien.Pose` or `(position, quaternion)` tuple
- `settling_steps` — steps to wait after releasing
- `retract_height` — Z height to retract to after placing (default: lift back up)

**Returns:** `PlaceResult` with `success`, `failure_reason`

### move

```python
move_result = ctx.move(
    target_pose,
    *,
    gripper_open=True,
    monitor_contacts=True,
    time_step_scale=1.0,
    contact_force_threshold=0.01,
)
```

Move the end-effector to a target pose using screw-based motion planning.

**Parameters:**
- `target_pose` — target end-effector pose
- `gripper_open` — gripper state during motion
- `monitor_contacts` — abort if unexpected contacts occur
- `time_step_scale` — planner/control waypoint spacing multiplier; values above `1.0` move faster with fewer waypoints
- `contact_force_threshold` — minimum disallowed contact magnitude treated as a collision

**Returns:** `MoveResult` with `success`, `failure_reason`, plus collision diagnostics (`contact_link`, `contact_entity`, `contact_force`) when the move aborts on contact.

### push

```python
push_result = ctx.push(
    approach_pose,
    push_pose,
    staging_pose=None,
    hover_pose=None,
    *,
    clearance_height=0.1,
    lift_height=0.1,
    effort_scale=1.0,
    effort_scale_end=1.0,
    min_contact_force=0.0,
    staging_speed_scale=1.0,
    clearance_speed_scale=1.0,
    approach_speed_scale=1.0,
    push_speed_scale=1.0,
    lift_speed_scale=1.0,
)
```

Approach an object and sweep it to a target position with a straight-line Cartesian motion between `approach_pose` and `push_pose`.

**Parameters:**
- `staging_pose` — optional pre-push pose, useful for vertical pushes that should descend straight down first
- `hover_pose` — optional pose directly above the approach point for a clean vertical lowering phase
- `approach_pose` — where to position before pushing
- `push_pose` — where to push to (the sweep target)
- `clearance_height` — height to lift before approaching
- `lift_height` — height to lift after pushing
- `effort_scale` — scale factor on the arm's drive force limit during the push
- `effort_scale_end` — optional end scale for a linear ramp across the push
- `min_contact_force` — require at least this peak contact force for success
- `*_speed_scale` — phase-wise waypoint spacing multipliers; values above `1.0` run faster with fewer waypoints

Setting `clearance_height=0` or `lift_height=0` skips that lift phase entirely, which is useful when a solver already stages the tool above the push start pose.

**Returns:** `PushResult` with `success`, `failure_reason`, plus:
- `push_distance` / `planar_push_distance` — executed start-to-target distance
- `arm_force_limit_start/end/mean` — commanded arm drive force-limit schedule
- `contact_force_peak` / `contact_force_mean` — estimated gripper-object contact force from PhysX impulses
- `joint_load_l2_peak` / `joint_load_l2_mean` — aggregate internal joint-load proxy during the sweep
- `contact_objects` — object names touched during the push

## Task-Space Push Planning

Pushes are best parameterized in task space, not by robot joint indices. The
recommended model is:

- contact point in workspace
- push direction in the plane
- contact height
- tool approach axis (`vertical`, `horizontal`, or a custom axis)
- tool spin/roll around that axis

Use `ctx.plan_linear_push(...)` to build poses from those parameters:

```python
plan = ctx.plan_linear_push(
    contact_position=[0.20, -0.03, 0.045],
    push_angle_deg=35.0,
    push_distance=0.12,
    approach_gap=0.05,
    wrist_orientation="vertical",
    tool_spin_deg=20.0,
    hover_height=0.16,
    staging_height=0.16,
    staging_backoff=0.04,
)
push_result = ctx.push(**plan.as_skill_kwargs())
```

This keeps the skill general across vertical pushes, horizontal pushes, angled
pushes, and different tool rolls while leaving IK/planning to choose the
actual joints.

## PoseLike

All skills accept poses as either `sapien.Pose` or `(position, quaternion)` tuples:

```python
ctx.place(sapien.Pose([0.1, 0.0, 0.2], [1, 0, 0, 0]))
ctx.place(([0.1, 0.0, 0.2], [1, 0, 0, 0]))
```

Quaternion format is `[w, x, y, z]` (SAPIEN convention).

## Result Dataclasses

All skills return a `SkillResult` subclass:

```python
@dataclass
class SkillResult:
    success: bool
    failure_reason: Optional[str] = None
    step_result: Optional[tuple] = None  # last (obs, rew, term, trunc, info)
```

`PickResult` adds `grasp_pose`, `lift_pose`, and `obj_size` — used by downstream skills (e.g., computing where to place).

## RobotConfig

Robot-specific constants (move group, finger length, gripper links) are stored in `RobotConfig`, not hardcoded:

```python
from taskbench.skills.robot_config import ROBOT_CONFIGS

ROBOT_CONFIGS = {
    "panda": RobotConfig(move_group="panda_hand_tcp", finger_length=0.025, ...),
    "panda_wristcam": RobotConfig(...),
}
```

`SkillContext` auto-detects the robot from `env.unwrapped.agent.uid`.

## Motion Primitives (Lower Level)

`taskbench/skills/motion.py` provides the functions that skills are built on:

| Function | Description |
|----------|-------------|
| `setup_planner(env, robot_config)` | Create mplib Planner from env's robot |
| `move_to_pose(env, planner, pose, gripper_state, robot_config, ...)` | Plan + execute straight-line screw motion |
| `follow_path(env, result, gripper_state, robot_config, ...)` | Execute a pre-planned trajectory |
| `actuate_gripper(env, planner, gripper_state, steps=6)` | Open/close gripper |
| `build_action(env, qpos, gripper_state)` | Build action array for pd_joint_pos |
| `attach_object(planner, size)` / `detach_object(planner)` | Inform planner about held objects |
