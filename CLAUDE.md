# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses **uv** for dependency management. Always use `uv run` (not `source .venv/bin/activate && python`) and `uv pip` (not bare `pip`) for all operations.

```bash
# Create and sync the venv (installs all deps from pyproject.toml + uv.lock)
uv sync

# Or with dev extras (includes black, isort)
uv sync --extra dev

# Install additional packages (ALWAYS use uv pip, never bare pip)
uv pip install <package>
```

## Running

Entry point is Hydra-based. Task and solver are independent config groups. **Always use `uv run`.**

```bash
# Default run (random solver, 16 envs, 100 episodes)
uv run python -m taskbench.run

# Compose task + solver
uv run python -m taskbench.run task=tabletop_retrieval solver=tabletop_push

# Override task params (flat — no extra_kwargs nesting)
uv run python -m taskbench.run task=tabletop_retrieval solver=tabletop_push \
    task.num_cylinders=5 task.bottle_body_half_length=0.08

# Override solver params
uv run python -m taskbench.run task=tabletop_retrieval solver=tabletop_push \
    run.solver_kwargs.effort_scale=0.5

# Override runtime params
uv run python -m taskbench.run task=tabletop_retrieval solver=tabletop_push \
    runtime.record_video=false run.num_episodes=10

# Load scenes from file
uv run python -m taskbench.run task=tabletop_retrieval solver=tabletop_push \
    task.scene_file=data/scenes/layouts.json

# Other examples
uv run python -m taskbench.run task=stack_cubes solver=stack_cubes
uv run python -m taskbench.run seed=123 logging.use_wandb=true
```

Hydra writes timestamped output dirs under `outputs/`. Videos go to `videos/`.

## Formatting

```bash
uv run black taskbench/
uv run isort taskbench/
```

No tests exist in this repo.

## Architecture

**taskbench** is a robotics research testbed for evaluating solvers on ManiSkill3 manipulation tasks. See `docs/architecture.md` for full documentation.

### Config Structure

Three config groups, three concerns:

```
configs/
  default.yaml         — seed, logging, run defaults, runtime defaults
  task/                — what: robot identity + position, objects, layout, success criterion
    shelf.yaml
    stack_cubes.yaml
    tabletop_retrieval.yaml
  solver/              — how: solver name + params, solver runtime requirements
    shelf_reachability.yaml
    stack_cubes.yaml
    tabletop_push.yaml
```

Three config sections in the resolved config:
- **`task:`** — env_id, `robot_uids`, `robot_base_pose` (required), env constructor params (objects, layout, success criterion)
- **`runtime:`** — framework params (control_mode, num_envs, recording, obs_mode, reward_mode)
- **`run:`** — solver selection, num_episodes, solver_kwargs

**Task YAML is the single source of truth for scene geometry.** Every task must declare `robot_base_pose` (3-element `[x,y,z]` or 7-element `[x,y,z,qw,qx,qy,qz]`). Solver configs only set `run.*` and solver-specific `runtime.*` requirements (e.g., `control_mode`, `num_envs`). Use CLI overrides for task variants instead of creating separate YAML files.

### Core Flow

`taskbench/run.py` is the entry point (`@hydra.main`). It dispatches based on `cfg.run.solver`:
- `"random"` → `run_random()` — vectorized env (`ManiSkillVectorEnv`), samples random actions
- `batched` flag → `run_batched()` — GPU-vectorized env, cuRobo planner
- Any other value → `run_solver()` — auto-discovers solver via `@register_solver`, creates a single env, runs episode loop

### Solver System (pluggable, zero-touch)

Solvers self-register via `@register_solver("name")` decorator in `taskbench/solver.py`. Auto-discovery walks `taskbench/solvers/*.py` via `pkgutil`. Adding a new solver requires **no edits to core files**:
1. Create `taskbench/solvers/my_solver.py` with `@register_solver("my_solver")` inheriting `BaseSolver`
2. Create `configs/solver/my_solver.yaml` with solver params and runtime constraints
3. Run: `uv run python -m taskbench.run task=my_task solver=my_solver`

### Environment System

Envs inherit `TaskEnv` and must implement:
- `get_objects() → dict[str, actor]` — manipulable objects
- `get_scene_info() → dict` (optional) — scene geometry for solver planning (object positions, target, table height)

Envs define the scene and success criterion. They don't compute trajectories — solvers do that.

### Skill System

**mplib skills** (`Pick`, `Place`, `Move`, `Push`) in `taskbench/skills/primitives.py`. Use `SkillContext` for single-env CPU:
```python
ctx = SkillContext(env)
ctx.reset(seed=42)
ctx.step_callback = recorder.record
ctx.push(approach_pose, push_pose)
```

**cuRobo + EE control skills** — standalone classes for GPU-batched operation:
- **`StickPush`** (`taskbench/skills/stick_push.py`) — 5-phase push: stage (cuRobo) → insert → sweep → nudge-back → retract → smooth-rest. Active orientation correction at every step. Works single-env and GPU-batched (128+ envs).

```python
push = StickPush(env, motion_gen, robot_uid="panda_stick_long",
                 n_envs=N, rest_qpos=REST, robot_base_pos=base,
                 safe_start_qpos=SAFE)
result = push(approach, quat, entry, sweep, retract)
# result.success_mask, .sweep_final_dist, .orientation_drift
```

**`batched_ee_delta_move()`** in `curobo_motion.py` — drives N TCPs to targets using `pd_ee_delta_pose` with optional `target_quaternions` for active orientation correction.

Robot-specific constants live in `RobotConfig` (`taskbench/skills/robot_config.py`), not hardcoded. Skills accept `PoseLike` (tuples or `sapien.Pose`) and resolve objects by string name.

### Key Modules

- **`configs/default.yaml`** — Hydra config with `task:`, `runtime:`, `run:` sections.
- **`taskbench/envs/factory.py`** — `make_env(cfg)` (vectorized), `make_single_env(cfg)` (raw), `cleanup_env()` (GPU memory leak fix).
- **`taskbench/envs/base.py`** — `TaskEnv` base class: intercepts `robot_base_pose` from kwargs, provides `_default_initial_agent_poses()` and `_reset_robot()`, abstract `get_objects()`.
- **`taskbench/envs/shelf_env.py`** — `ShelfEnv`: enclosed shelf with cylinders, per-env random count (1-N) and non-overlapping placement.
- **`taskbench/solver.py`** — `BaseSolver` ABC, `SolverResult`, `@register_solver`, `discover_solvers()`.
- **`taskbench/skills/robot_config.py`** — `RobotConfig` dataclass + `get_robot_config()` (auto-discovers from agent classes).
- **`taskbench/agents/`** — Pluggable robot agents. Includes `panda_stick_long.py` (25cm stick, no gripper).
- **`taskbench/skills/context.py`** — `SkillContext` — bundles env + planner + objects + skills (mplib).
- **`taskbench/skills/stick_push.py`** — `StickPush` skill (cuRobo + EE control, GPU-batched).
- **`taskbench/skills/curobo_motion.py`** — GPU-batched motion: `setup_curobo_planner()`, `batched_ee_delta_move()`, `batched_follow_path()`, `batched_move_to_pose()`.
- **`taskbench/skills/motion.py`** — Low-level mplib helpers: `setup_planner()`, `move_to_pose()`, `build_action()`, `PoseLike`.
- **`taskbench/skills/primitives.py`** — Composable skill objects with `SkillResult` dataclasses (mplib).
- **`taskbench/recorder.py`** — `StateRecorder` for capturing simulation state to HDF5.
- **`taskbench/logger.py`** — Optional WandB logging wrapper.
- **`scripts/test_gpu_parallel.py`** — GPU-batched multi-push data collection using `StickPush`.
- **`tests/skills/test_stick_push.py`** — 12 tests (unit, integration, regression).

### Critical Constraints (mplib / ManiSkill)

- **mplib 0.2.x API**: Uses `mplib.pymp.Pose` objects (not numpy arrays) for `set_base_pose()`, `plan_screw()`, etc.
- **SAPIEN poses are batched**: Even with `num_envs=1`, pose tensors have shape `(1, 3)` / `(1, 4)` — must `.flatten()` before passing to mplib.
- **Motion planner requires**: `num_envs=1`, `sim_backend="cpu"`, `pd_joint_pos` control mode, no `ManiSkillVectorEnv` wrapper.
- **Video recording with planner**: Must use `save_on_reset=False` on `RecordEpisode` and call `env.flush_video()` manually.
- **numpy < 2.0** required by mplib 0.2.1.

### Critical Constraints (cuRobo / GPU parallel)

- **cuRobo self-collision**: Exact REST_QPOS `[0,-1.3,0,-2.5,0,1,0]` fails self-collision check. Use SAFE_QPOS (slightly drifted).
- **cuRobo `plan_batch`**: crashes if 0/N IK succeed (known bug). Check `n_success > 0` before accessing trajectories.
- **cuRobo tensors**: `.clone()` all tensors passed between episodes (internal buffer reuse conflict).
- **GPU env control mode switching**: `raw.agent.set_control_mode(mode)` + `raw.agent.controller.reset()`.
- **GPU per-env poses**: Use `Pose.create_from_pq(p=(N,3), q=(4,))` — auto-broadcasts, auto-masks to env_idx.
- **GPU per-env markers**: Can't add actors to individual sub_scenes (tensor shape mismatch).
- **GPU video recording**: Requires `max_steps_per_video` param in `RecordEpisode`.
- **SAPIEN GPU memory leak**: Call `cleanup_env()` from `taskbench/envs/factory.py` after `env.close()`.
- **srun**: Use `--partition=unlimited`, don't set `--mem` (default UNLIMITED). Set `CUDA_HOME=/usr/local/cuda-12.6` and `CPATH=.../targets/x86_64-linux/include`.
