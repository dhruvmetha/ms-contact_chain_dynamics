# Taskbench Architecture

Taskbench is a robotics research testbed for evaluating solvers on ManiSkill3 manipulation tasks. It provides a pluggable framework for defining environments, composing manipulation skills, and recording demonstrations.

## Directory Structure

```
configs/
  default.yaml                      # Base Hydra config (seed, runtime, logging, run)
  task/                             # Task config group (what: scene, objects, success)
    open_table_push.yaml
    stack_cubes.yaml
  solver/                           # Solver config group (how: trajectory, execution)
    open_table_push.yaml
    stack_cubes.yaml

taskbench/                          # Core framework
  run.py                            # Entry point (@hydra.main)
  solver.py                         # BaseSolver ABC, SolverResult, @register_solver
  recorder.py                       # StateRecorder for episode capture (HDF5)
  batched_recorder.py               # GPU-batched recording for vectorized envs
  logger.py                         # WandB logging wrapper
  envs/
    base.py                         # TaskEnv — base class (get_objects, get_push_task)
    factory.py                      # make_env(cfg) / make_single_env(cfg)
    __init__.py                     # get_objects() dispatch + env registration
    open_table_push.py              # OpenTablePush-v1 (push row of bottles)
    open_table_bottle_clutter.py    # OpenTableBottleClutter-v1 (dense bottle clutter)
    open_table_scene.py             # Shared compact table scene builder
    stack_n_cube.py                 # StackNCube-v1 (parameterized N-cube)
    stack_cube_distractor.py        # StackCubeDistractor-v1 (2-cube + distractor)
    shelf_env.py                    # ShelfEnv-v1 (enclosed shelf with cylinders)
    bin_with_objects.py             # BinWithObjects-v1 (bin of primitives + YCB)
  skills/
    context.py                      # SkillContext — bundles env + planner + skills
    primitives.py                   # Composable skill objects (Pick, Place, Push, Move)
    motion.py                       # Low-level mplib helpers (plan_screw, follow_path)
    robot_config.py                 # RobotConfig dataclass + auto-discovery
    batched_context.py              # BatchedSkillContext (cuRobo, GPU-parallel)
    batched_primitives.py           # BatchedMove, BatchedPush (GPU-parallel)
    curobo_motion.py                # GPU-batched planning with cuRobo
    curobo_world.py                 # Collision world setup for cuRobo
  solvers/
    open_table_push.py              # OpenTablePushSolver — single-env mplib push
    batched_contact_push.py         # BatchedContactPushSolver — GPU-batched cuRobo push
    stack_n_cubes.py                # StackCubesSolver — sequential pick-place
    replay.py                       # ReplaySolver — replay HDF5 demos
    demo_recorder.py                # DemoRecorderSolver — interactive viewer
    shelf_reachability.py           # ShelfReachabilitySolver — grid sweep
  agents/
    ur5e_robotiq.py                 # UR5e + Robotiq 2F-85 agent
    __init__.py                     # Agent discovery (pkgutil)
```

## Config Structure

Three config sections, three concerns:

```yaml
task:       # what's in the world (env_id, objects, layout, success criterion)
  env_id: OpenTablePush-v1
  push_axis: y
  num_cylinders: 2

runtime:    # how to run the simulation (control mode, parallelism, recording)
  control_mode: pd_joint_pos
  num_envs: 1
  record_video: true

run:        # what solver to use and its parameters
  solver: open_table_push
  solver_kwargs:
    wrist_orientation: vertical
    approach_gap: 0.06
```

Two config groups compose independently:

```bash
uv run python -m taskbench.run task=open_table_push solver=open_table_push
uv run python -m taskbench.run task=open_table_push_x solver=open_table_push_horizontal
uv run python -m taskbench.run task=open_table_push_ur5e solver=open_table_push
```

Override any param from CLI:

```bash
task.num_cylinders=5       # task params (flat, no nesting)
task.bottle_body_half_length=0.08
task.scene_file=layouts.json
run.solver_kwargs.effort_scale=0.5
runtime.record_video=false
```

---

## Core Flow

`taskbench/run.py` is the entry point (`@hydra.main`). It dispatches based on `cfg.run.solver`:

- `"random"` → `run_random()` — creates a vectorized env via `make_env()`, samples random actions
- `batched` flag → `run_batched()` — GPU-vectorized env, cuRobo planner
- Any other value → `run_solver()` — auto-discovers solver, creates a single CPU env, calls `solver.solve()` per episode

The runner owns the episode loop. The solver owns everything inside an episode — resets, planning, execution, evaluation.

---

## Environments

### TaskEnv Base Class

All custom environments inherit from `TaskEnv` and must implement `get_objects()`:

```python
from taskbench.envs.base import TaskEnv

class TaskEnv(BaseEnv, metaclass=ABCMeta):
    @abstractmethod
    def get_objects(self) -> dict[str, object]:
        """Return a name -> actor mapping for all manipulable objects."""
        ...

    def get_push_task(self) -> dict:
        """Return scene geometry for push-capable envs (optional)."""
        raise NotImplementedError
```

Push-capable envs implement `get_push_task()` to expose scene info without computing trajectories:

```python
def get_push_task(self):
    return {
        "push_direction_xy": self.push_direction_xy,
        "row_origin_xy": self.row_origin_xy,
        "num_objects": self.num_cylinders,
        "row_spacing": self.row.spacing,
        "table_top_z": self._get_table_top_z(),
    }
```

### Registered Environments

| Env ID | Class | Notes |
|--------|-------|-------|
| `StackCube-v1` | Built-in ManiSkill | 2-cube stacking |
| `StackNCube-v1` | `StackNCubeEnv` | Parameterized N-cube (2-6) |
| `StackCubeDistractor-v1` | `StackCubeDistractorEnv` | 2-cube + blue distractor |
| `ShelfEnv-v1` | `ShelfEnv` | Enclosed shelf, 19 blue + 1 red cylinder |
| `BinWithObjects-v1` | `BinWithObjectsEnv` | Bin with ~30 random primitives + YCB objects |
| `OpenTablePush-v1` | `OpenTablePushEnv` | Row of bottles on open table for push tests |
| `OpenTableBottleClutter-v1` | `OpenTableBottleClutterEnv` | Dense random bottle placements |

### Environment Factories

```python
from taskbench.envs.factory import make_env, make_single_env

# Vectorized env for RL / batched data collection (GPU backend)
env = make_env(cfg)

# Single raw env for motion planner (num_envs=1, CPU backend, no vector wrapper)
env = make_single_env(cfg)
```

The factory reads `cfg.task` for env constructor kwargs and `cfg.runtime` for framework params. Motion-planner solvers **must** use `make_single_env()`.

---

## Solvers

### BaseSolver and @register_solver

```python
from taskbench.solver import BaseSolver, SolverResult, register_solver

@register_solver("my_task")
class MyTaskSolver(BaseSolver):
    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        ...
```

The `@register_solver` decorator adds the class to `SOLVER_REGISTRY`. Solvers under `taskbench/solvers/` are auto-discovered via `pkgutil.walk_packages` on first call to `get_solver()` — no manual imports needed.

Solvers own the episode lifecycle — resets, evaluation, recording. The runner trusts the solver's `SolverResult`.

### SolverResult

```python
@dataclass
class SolverResult:
    success: bool
    reward: float = 0.0
    elapsed_steps: int = 0
    info: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
```

---

## Skills

See [docs/skills.md](skills.md) for full documentation of all skills, parameters, and return types.

### SkillContext

```python
ctx = SkillContext(env)
ctx.reset(seed=42)                        # env.reset + planner setup
ctx.step_callback = recorder.record       # propagates to all skills
ctx.pick("cube_1", lift_height=0.15)      # → PickResult
ctx.place(target_pose)                    # → PlaceResult
ctx.push(approach_pose, push_pose, ...)   # → PushResult
```

`ctx.initialize()` sets up planner/objects from current env state without resetting — for search-based solvers that manage their own resets.

---

## Adding a New Task

A "task" consists of up to three pieces: an environment, a solver, and Hydra configs. **No edits to core files required.**

### 1. Create the Environment

Create `taskbench/envs/my_task.py` and register the import in `taskbench/envs/__init__.py`.

### 2. Create the Solver

Create `taskbench/solvers/my_solver.py` with `@register_solver("my_solver")`.

### 3. Create the Hydra Configs

`configs/task/my_task.yaml`:

```yaml
# @package _global_
task:
  env_id: MyTask-v1
  my_param: 42

runtime:
  max_episode_steps: 200
  reward_mode: none
```

`configs/solver/my_solver.yaml`:

```yaml
# @package _global_
run:
  solver: my_solver

runtime:
  control_mode: pd_joint_pos
  num_envs: 1
```

### 4. Run

```bash
uv run python -m taskbench.run task=my_task solver=my_solver
```

---

## State Recording

See [docs/demos.md](demos.md) for full documentation on recording, HDF5 format, parsing, and replay.

---

## Constraints

- **mplib 0.2.x**: uses `mplib.pymp.Pose` objects (not numpy). SAPIEN poses are batched — must `.flatten()` before passing to mplib.
- **Motion planner requires**: `num_envs=1`, `sim_backend="cpu"`, `pd_joint_pos` control mode.
- **Video recording with planner**: use `save_on_reset=False` on `RecordEpisode`, call `env.flush_video()` manually.
- **numpy < 2.0** required by mplib 0.2.1.
