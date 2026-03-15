"""Shared defaults for open-table environments and scene generation.

This module is the single source of truth for constants used by both
``OpenTableBottleClutterEnv`` and the algorithmic scene sampler in
``scripts/generate_clutter_scene_library.py``.  Changing a value here
automatically propagates to both paths.

The ``PANDA_TABLE_DEFAULTS`` dataclass mirrors the values that ManiSkill's
``TableSceneBuilder.initialize()`` applies for the Panda robot.  If a
future ManiSkill release changes these, the drift-detection assertion in
``OpenTableBottleClutterEnv._initialize_episode`` will fire immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Bottle geometry (unscaled base values, multiplied by BOTTLE_SCALE at use)
# ---------------------------------------------------------------------------
BOTTLE_SCALE: float = 1.45
BOTTLE_BODY_RADIUS_BASE: float = 0.020
BOTTLE_BODY_HALF_LENGTH_BASE: float = 0.050
BOTTLE_NECK_RADIUS_BASE: float = 0.009
BOTTLE_NECK_HALF_LENGTH_BASE: float = 0.016
BOTTLE_NECK_OFFSET_BASE: float = 0.046
BOTTLE_BALLAST_RADIUS_BASE: float = 0.018
BOTTLE_BALLAST_HALF_LENGTH_BASE: float = 0.010
BOTTLE_BALLAST_OFFSET_BASE: float = 0.032
BOTTLE_VISUAL_STYLE: str = "ycb_mustard"
RANDOM_YAW: bool = True

# ---------------------------------------------------------------------------
# Placement grid
# ---------------------------------------------------------------------------
EDGE_MARGIN: float = 0.04
PLACEMENT_SPACING: float = 0.063
PLACEMENT_CLEARANCE: float = 0.001
PLACEMENT_JITTER: float = 0.001
DENSE_LAYOUT_PROB: float = 0.40
MIXED_LAYOUT_PROB: float = 0.40

# ---------------------------------------------------------------------------
# Derived (scaled) bottle dimensions — convenience for callers that need the
# default scaled values without recomputing.
# ---------------------------------------------------------------------------
BOTTLE_BODY_RADIUS: float = BOTTLE_BODY_RADIUS_BASE * BOTTLE_SCALE
BOTTLE_BODY_HALF_LENGTH: float = BOTTLE_BODY_HALF_LENGTH_BASE * BOTTLE_SCALE


# ---------------------------------------------------------------------------
# Panda robot defaults (mirrors ManiSkill TableSceneBuilder.initialize)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PandaTableDefaults:
    """Expected Panda state after ``TableSceneBuilder.initialize()``."""

    root_position: tuple[float, ...] = (-0.615, 0.0, 0.0)
    root_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    # fmt: off
    home_qpos: tuple[float, ...] = (
        0.0,                    # joint1
        0.39269908169872414,    # joint2  (pi/8)
        0.0,                    # joint3
        -1.9634954084936208,    # joint4  (-5*pi/8)
        0.0,                    # joint5
        2.356194490192345,      # joint6  (3*pi/4)
        0.7853981633974483,     # joint7  (pi/4)
        0.04,                   # finger_left
        0.04,                   # finger_right
    )
    # fmt: on

    def root_position_array(self) -> np.ndarray:
        return np.array(self.root_position, dtype=np.float32)

    def root_quat_array(self) -> np.ndarray:
        return np.array(self.root_quat, dtype=np.float32)

    def home_qpos_array(self) -> np.ndarray:
        return np.array(self.home_qpos, dtype=np.float32)


PANDA_TABLE_DEFAULTS = PandaTableDefaults()
