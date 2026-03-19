"""Skill context — bundles env + planner + objects for skill execution.

Eliminates the boilerplate that every solver repeats::

    ctx = SkillContext(env, step_callback=recorder.record)
    ctx.reset(seed=42)

    ctx.pick("cube_1", lift_height=0.15)
    ctx.place(target_pose)
"""

from typing import Callable, Optional

from taskbench.envs import get_objects
from taskbench.skills.motion import make_linear_push_plan, setup_planner
from taskbench.skills.primitives import Move, Pick, Place, Push
from taskbench.skills.robot_config import RobotConfig, get_robot_config

_NOT_READY_MSG = "SkillContext.reset() must be called before using skills"


class _SkillProxy:
    """Raises a clear error when skills are accessed before reset()."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_NOT_READY_MSG)

    def __getattr__(self, name):
        raise RuntimeError(_NOT_READY_MSG)


class SkillContext:
    """Shared context for skill-based solvers.

    Holds the env, motion planner, robot config, object references, and
    pre-bound skill instances. Call ``reset()`` to re-initialize everything
    for a new episode.

    Args:
        env: Gym env (num_envs=1, sim_backend="cpu").
        step_callback: Optional callable invoked after each env.step()
            (e.g. ``recorder.record``).
    """

    def __init__(self, env, *, step_callback: Optional[Callable] = None):
        self.env = env
        self._step_callback = step_callback
        self.robot_config: RobotConfig = get_robot_config(env)
        self.planner = None
        self.objects: dict[str, object] = {}

        # Skill instances — populated by reset() → _build_skills()
        _proxy = _SkillProxy()
        self.pick: Pick = _proxy  # type: ignore[assignment]
        self.place: Place = _proxy  # type: ignore[assignment]
        self.push: Push = _proxy  # type: ignore[assignment]
        self.move: Move = _proxy  # type: ignore[assignment]

    @property
    def step_callback(self) -> Optional[Callable]:
        return self._step_callback

    @step_callback.setter
    def step_callback(self, value: Optional[Callable]):
        self._step_callback = value
        for skill in (self.pick, self.place, self.push, self.move):
            if not isinstance(skill, _SkillProxy):
                skill.step_callback = value

    def reset(self, seed=None):
        """Reset the env and rebuild planner, objects, and skills."""
        self.env.reset(seed=seed)
        self.initialize()

    def initialize(self):
        """Set up planner, objects, and skills from current env state.

        Use this after the env has already been reset externally (e.g. by
        a search-based solver that manages its own resets).
        """
        self.planner = setup_planner(self.env, self.robot_config)
        self.objects = get_objects(self.env)
        self._build_skills()

    def _build_skills(self):
        """Create skill instances with current planner/objects."""
        kw = dict(
            robot_config=self.robot_config,
            objects=self.objects,
            step_callback=self._step_callback,
        )
        self.pick = Pick(self.env, self.planner, **kw)
        self.place = Place(self.env, self.planner, **kw)
        self.push = Push(self.env, self.planner, **kw)
        self.move = Move(self.env, self.planner, **kw)

    def plan_linear_push(self, **kwargs):
        """Build a Cartesian push plan from task-space parameters.

        This is the preferred interface for push setup: define where contact
        starts, which direction to push, how far to push, and the tool
        orientation. The low-level planner handles joint-space realization.
        """
        return make_linear_push_plan(self.env.unwrapped.agent, **kwargs)
