# Taskbench Architecture

Taskbench is a robotics research testbed for evaluating solvers on ManiSkill3 manipulation tasks. It provides a pluggable framework for defining environments, composing manipulation skills, and recording demonstrations.

## Directory Structure

```
taskbench/                          # Core framework
  run.py                            # Entry point (@hydra.main)
  solver.py                         # BaseSolver ABC, SolverResult, @register_solver, auto-discovery
  recorder.py                       # StateRecorder for episode capture (HDF5)
  logger.py                         # WandB logging wrapper
  envs/
    base.py                         # TaskEnv — base class with get_objects()
    factory.py                      # make_env() / make_single_env()
    __init__.py                     # get_objects() dispatch + env registration
    stack_n_cube.py                 # StackNCube-v1 (parameterized N-cube)
    stack_cube_distractor.py        # StackCubeDistractor-v1 (2-cube + distractor)
    shelf_env.py                    # ShelfEnv-v1 (enclosed shelf with cylinders)
    bin_with_objects.py             # BinWithObjects-v1 (bin of primitives + YCB)
    open_table_bottle_clutter.py    # OpenTableBottleClutter-v1 (dense bottle clutter)
  skills/
    robot_config.py                 # RobotConfig dataclass + registry
    context.py                      # SkillContext — bundles env + planner + skills
    motion.py                       # Low-level mplib helpers
    primitives.py                   # Composable skill objects (Pick, Place, etc.)

  solvers/
    stack_n_cubes.py                # StackCubesSolver (@register_solver)
    replay.py                       # ReplaySolver (@register_solver)
    demo_recorder.py                # DemoRecorderSolver (@register_solver)
    shelf_reachability.py           # ShelfReachabilitySolver (@register_solver)
    dense_random_push.py            # DenseRandomPushSolver (@register_solver)

configs/
  default.yaml                      # Default Hydra config (YAML-only, no Python dataclasses)
  solver/                           # Hydra config group (one YAML per solver)
    random.yaml
    stack_cubes.yaml
    replay.yaml
    demo_recorder.yaml
    shelf_reachability.yaml
```

## Core Flow

`taskbench/run.py` is the entry point (`@hydra.main`). It dispatches based on `cfg.run.solver`:

- `"random"` -> `run_random()` — creates a vectorized env via `make_env()`, samples random actions
- Any other value -> `run_solver()` — looks up the solver via `get_solver()`, creates a single CPU env via `make_single_env()`, calls `solver.solve()` per episode

```bash
uv run python -m taskbench.run                                      # random baseline
uv run python -m taskbench.run solver=stack_cubes                   # registered solver
uv run python -m taskbench.run solver=stack_cubes env.num_cubes=4
```

---

## Skills

See [docs/skills.md](skills.md) for full documentation of all skills, parameters, and return types.

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
```

### Registered Environments

| Env ID | Class | get_objects() | Notes |
|--------|-------|---------------|-------|
| `StackCube-v1` | Built-in ManiSkill | `{"cube_0": cubeB, "cube_1": cubeA}` | 2-cube stacking (fallback in `get_objects()` dispatch) |
| `StackNCube-v1` | `StackNCubeEnv` | `{"cube_0": ..., "cube_N": ...}` | Parameterized N-cube (2-6), cube_0 is always green (base), any tower order valid. `env.num_cubes=N` |
| `StackCubeDistractor-v1` | `StackCubeDistractorEnv` | `{"cube_0": green, "cube_1": red, "cube_2": blue}` | 2-cube stacking + blue distractor |
| `ShelfEnv-v1` | `ShelfEnv` | `{"cyl_0": ..., "cyl_19": ...}` | Enclosed shelf, 19 blue + 1 red cylinder |
| `BinWithObjects-v1` | `BinWithObjectsEnv` | `{obj.name: obj, ...}` | Bin with ~30 random primitives + YCB objects |
| `OpenTableBottleClutter-v1` | `OpenTableBottleClutterEnv` | `{"bottle_0": ..., "bottle_N": ...}` | Open table with dense random bottle placements for push-data collection |

### Environment Factories

```python
from taskbench.envs.factory import make_env, make_single_env

# Vectorized env for RL (multiple parallel envs, GPU backend)
env = make_env(cfg.env)

# Single raw env for motion planner (num_envs=1, CPU backend, no vector wrapper)
env = make_single_env(cfg.env)
```

Motion-planner solvers **must** use `make_single_env()` because mplib requires `num_envs=1`, `sim_backend="cpu"`, and direct access to `env.unwrapped`.

---

## Solvers

### BaseSolver and @register_solver

```python
from taskbench.solver import BaseSolver, SolverResult, register_solver

@register_solver("my_task")
class MyTaskSolver(BaseSolver):
    def solve(self, env, seed=None) -> SolverResult:
        ...
```

The `@register_solver` decorator adds the class to `SOLVER_REGISTRY`. Solvers under `taskbench/solvers/` are auto-discovered via `pkgutil.walk_packages` on first call to `get_solver()` — no manual imports needed.

### SolverResult

```python
@dataclass
class SolverResult:
    success: bool                           # task success
    reward: float = 0.0                     # cumulative reward
    elapsed_steps: int = 0                  # steps taken
    info: dict = field(default_factory=dict) # task-specific data
    failure_reason: Optional[str] = None    # human-readable reason
```

---

## Adding a New Task

A "task" consists of up to three pieces: an environment, a solver, and a Hydra config. **No edits to core files required.**

### 1. Create the Environment

Create `taskbench/envs/my_task.py`:

```python
import torch
import sapien
from mani_skill.utils.registration import register_env
from taskbench.envs.base import TaskEnv

@register_env("MyTask-v1", max_episode_steps=200)
class MyTaskEnv(TaskEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def get_objects(self) -> dict[str, object]:
        return {"target": self.target_obj}

    def _load_scene(self, options: dict):
        # Build scene actors
        ...

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Randomize object poses each episode
        ...

    def evaluate(self):
        # Return {"success": torch.tensor([bool], device=self.device)}
        ...

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
```

Register the env by adding an import to `taskbench/envs/__init__.py`:

```python
import taskbench.envs.my_task  # noqa: F401
```

### 2. Create the Solver

Create `taskbench/solvers/my_task.py`:

```python
from taskbench.skills.context import SkillContext
from taskbench.solver import BaseSolver, SolverResult, register_solver

@register_solver("my_task")
class MyTaskSolver(BaseSolver):
    def solve(self, env, seed=None) -> SolverResult:
        ctx = SkillContext(env)
        ctx.reset(seed=seed)

        pick_result = ctx.pick("target")
        if not pick_result.success:
            return SolverResult(success=False, failure_reason=pick_result.failure_reason)

        place_result = ctx.place(goal_pose)
        if not place_result.success:
            return SolverResult(success=False, failure_reason=place_result.failure_reason)

        info = env.unwrapped.evaluate()
        return SolverResult(success=bool(info["success"].item()))
```

### 3. Create the Hydra Config

`configs/solver/my_task.yaml`:

```yaml
# @package _global_
run:
  solver: my_task

env:
  env_id: MyTask-v1
  control_mode: pd_joint_pos
  num_envs: 1
  reward_mode: none
```

### 4. Run

```bash
uv run python -m taskbench.run solver=my_task
```

The solver is auto-discovered from `taskbench/solvers/my_task.py` (no registry edits). The config is found by Hydra in `configs/solver/` (no searchpath edits).

---

## State Recording

See [docs/demos.md](demos.md) for full documentation on recording, HDF5 format, parsing, and replay.

---

## Configuration

All config is YAML-only (no Python dataclasses). `configs/default.yaml` defines the schema; solver configs in `configs/solver/` override env-specific fields.

Override any field from the command line:

```bash
uv run python -m taskbench.run seed=123 env.record_video=true run.num_episodes=10
uv run python -m taskbench.run solver=stack_cubes env.num_cubes=5
```

---

## Constraints

- **mplib 0.2.x**: uses `mplib.pymp.Pose` objects (not numpy). SAPIEN poses are batched — must `.flatten()` before passing to mplib.
- **Motion planner requires**: `num_envs=1`, `sim_backend="cpu"`, `pd_joint_pos` control mode.
- **Video recording with planner**: use `save_on_reset=False` on `RecordEpisode`, call `env.flush_video()` manually.
- **numpy < 2.0** required by mplib 0.2.1.
