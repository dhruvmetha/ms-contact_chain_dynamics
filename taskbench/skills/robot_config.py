"""Robot-specific constants for motion planning and skill execution.

Each supported robot has a ``RobotConfig`` — either declared on the agent
class via a ``taskbench_config`` attribute (preferred) or listed in the
``_BUILTIN_CONFIGS`` fallback table for ManiSkill built-in robots.

Use ``get_robot_config(env)`` to look up the config for the current robot.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    """Hardware-specific constants that skills and motion planning need."""

    move_group: str  # mplib move group name (from SRDF)
    finger_length: float  # depth of gripper fingers (meters)
    gripper_link_names: frozenset[str]  # links for contact detection
    gripper_open: float = 1.0  # action value for open
    gripper_closed: float = -1.0  # action value for closed


# Fallback configs for ManiSkill built-in robots that don't have agent files
# with a ``taskbench_config`` attribute.
_PANDA_CONFIG = RobotConfig(
    move_group="panda_hand_tcp",
    finger_length=0.025,
    gripper_link_names=frozenset(
        {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
    ),
)
_BUILTIN_CONFIGS: dict[str, RobotConfig] = {
    "panda": _PANDA_CONFIG,
    "panda_wristcam": _PANDA_CONFIG,
}


def _discover_config(uid: str) -> RobotConfig | None:
    """Try to find a ``taskbench_config`` on the registered agent class."""
    from mani_skill.agents.registration import REGISTERED_AGENTS

    spec = REGISTERED_AGENTS.get(uid)
    if spec is None:
        return None
    agent_cls = spec.agent_cls
    cfg = getattr(agent_cls, "taskbench_config", None)
    if isinstance(cfg, RobotConfig):
        return cfg
    return None


def get_robot_config(env) -> RobotConfig:
    """Look up the RobotConfig for the env's robot.

    Checks the agent class for a ``taskbench_config`` attribute first,
    then falls back to ``_BUILTIN_CONFIGS`` for ManiSkill built-ins.
    """
    uid = env.unwrapped.agent.uid
    # Try agent-declared config first
    cfg = _discover_config(uid)
    if cfg is not None:
        return cfg
    # Fall back to built-in table
    if uid in _BUILTIN_CONFIGS:
        return _BUILTIN_CONFIGS[uid]
    from mani_skill.agents.registration import REGISTERED_AGENTS

    available = sorted(
        set(_BUILTIN_CONFIGS)
        | {
            k
            for k, v in REGISTERED_AGENTS.items()
            if hasattr(v.agent_cls, "taskbench_config")
        }
    )
    raise KeyError(
        f"No RobotConfig for robot {uid!r}. Available: {', '.join(available)}"
    )
