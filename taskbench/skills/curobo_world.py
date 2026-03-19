"""SAPIEN scene → cuRobo WorldConfig adapters.

Each environment type has a function that extracts obstacle geometry from
the SAPIEN scene and converts it to cuRobo ``WorldConfig`` objects with
``Cuboid`` obstacles. Target objects are explicitly excluded so the robot
can intentionally make contact with them.

For ``plan_batch_env()`` with N different scene layouts, produce a
``list[WorldConfig]`` of length N.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from curobo.geom.types import Cuboid, WorldConfig

logger = logging.getLogger("taskbench.skills.curobo_world")


def _actor_box_params(actor):
    """Extract center and half-size from a box-shaped SAPIEN actor.

    Returns (center_xyz, half_size_xyz) as numpy arrays, or None
    if the actor has no box collision shape.
    """
    import sapien.physx as physx

    comp = actor.find_component_by_type(physx.PhysxRigidStaticComponent)
    if comp is None:
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        return None

    shapes = comp.get_collision_shapes()
    if not shapes:
        return None

    shape = shapes[0]
    if not hasattr(shape, "half_size"):
        return None

    half_size = np.asarray(shape.half_size, dtype=np.float64).reshape(-1)[:3]
    world_pose = actor.pose * shape.get_local_pose()
    center = np.asarray(world_pose.p, dtype=np.float64).flatten()[:3]

    return center, half_size


def _make_cuboid(name: str, center, half_size):
    """Create a cuRobo Cuboid from center and half-size.

    cuRobo Cuboid takes full dims and pose as [x, y, z, qw, qx, qy, qz].
    """
    from curobo.geom.types import Cuboid

    center = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
    half_size = np.asarray(half_size, dtype=np.float64).reshape(-1)[:3]
    dims = (half_size * 2).tolist()
    pose = [float(center[0]), float(center[1]), float(center[2]),
            1.0, 0.0, 0.0, 0.0]
    return Cuboid(name=name, dims=dims, pose=pose)


def _extract_table_cuboid(env) -> Optional["Cuboid"]:
    """Build a thin table surface slab in robot base frame.

    Uses the known table initialization constants rather than querying
    the actor pose, because GPU batched scenes may not have committed
    pose updates when this is called.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env

    # Get robot base position for world→robot frame transform
    robot_base = np.asarray(
        raw.agent.robot.pose.p[0].cpu(), dtype=np.float64
    ).flatten()[:3]

    # Compute table top z from known constants.
    # The parent TableSceneBuilder.initialize() places the table actor at
    # z = -0.9196429. The table collision box local offset = height/2 and
    # half_size_z = height/2, so table top = -0.9196429 + height.
    table_scene = getattr(raw, "table_scene", None)
    if table_scene is None:
        logger.warning("No table_scene attribute on env")
        return None

    # Prefer the computed table_top_z (set by CompactOpenTableSceneBuilder.initialize())
    # over re-deriving from the internal ManiSkill table actor offset constant.
    table_top_z = getattr(table_scene, "table_top_z", None)
    if table_top_z is None:
        table_height = getattr(table_scene, "table_height", 0.78)
        table_top_z = -0.9196429 + table_height

    # Table center xy from the scene builder (or from actor)
    table_actor = getattr(table_scene, "table", None)
    if table_actor is not None:
        from taskbench.envs.open_table_scene import COMPACT_OPEN_TABLE_CENTER_XY, COMPACT_OPEN_TABLE_SIZE_XY
        cx, cy = COMPACT_OPEN_TABLE_CENTER_XY
        sx, sy = COMPACT_OPEN_TABLE_SIZE_XY
    else:
        cx, cy = -0.12, 0.0
        sx, sy = 1.0, 0.6

    # Thin slab at the table surface
    slab_thickness = 0.02
    slab_center = np.array([cx, cy, table_top_z - slab_thickness / 2])
    slab_half = np.array([sx / 2, sy / 2, slab_thickness / 2])

    # Transform to robot base frame
    slab_center = slab_center - robot_base
    return _make_cuboid("table", slab_center, slab_half)


def open_table_world(env, n_envs: int = 1) -> list:
    """Build cuRobo WorldConfig for open-table environments.

    Includes the table as a cuboid. Bottles/cylinders are excluded since
    they are the target objects for contact.

    Args:
        env: ManiSkill env (may be vectorized).
        n_envs: Number of parallel environments.

    Returns:
        List of WorldConfig, one per env. For open-table, all envs share
        the same table geometry (object poses differ but objects are excluded).
    """
    from curobo.geom.types import WorldConfig

    table = _extract_table_cuboid(env)
    cuboids = [table] if table is not None else []

    world_config = WorldConfig(cuboid=cuboids)

    # All envs share the same static table geometry
    return [world_config] * n_envs


def shelf_world(env, n_envs: int = 1) -> list:
    """Build cuRobo WorldConfig for shelf environments.

    Includes the table and shelf panels as cuboids. Objects on the shelf
    are excluded (they are manipulation targets).
    """
    from curobo.geom.types import WorldConfig

    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    cuboids = []

    table = _extract_table_cuboid(env)
    if table is not None:
        cuboids.append(table)

    # Extract shelf panels
    for actor in raw.scene.get_all_actors():
        if "shelf" not in actor.name and "panel" not in actor.name and "leg" not in actor.name:
            continue
        params = _actor_box_params(actor)
        if params is None:
            continue
        center, half_size = params
        cuboids.append(_make_cuboid(actor.name, center, half_size))

    world_config = WorldConfig(cuboid=cuboids)
    return [world_config] * n_envs


def bin_world(env, n_envs: int = 1) -> list:
    """Build cuRobo WorldConfig for bin environments.

    Includes the table and bin walls as cuboids. Objects in the bin are
    excluded (they are manipulation targets).
    """
    from curobo.geom.types import WorldConfig

    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    cuboids = []

    table = _extract_table_cuboid(env)
    if table is not None:
        cuboids.append(table)

    # Extract bin walls
    for actor in raw.scene.get_all_actors():
        if "bin" not in actor.name and "wall" not in actor.name:
            continue
        params = _actor_box_params(actor)
        if params is None:
            continue
        center, half_size = params
        cuboids.append(_make_cuboid(actor.name, center, half_size))

    world_config = WorldConfig(cuboid=cuboids)
    return [world_config] * n_envs


# Registry of env-id prefixes to world adapter functions
_WORLD_ADAPTERS = {
    "OpenTable": open_table_world,
    "Shelf": shelf_world,
    "Bin": bin_world,
}


def build_world_configs(env, n_envs: int = 1) -> list:
    """Auto-select the right world adapter based on env_id.

    Falls back to open_table_world if no specific adapter matches.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    env_id = raw.spec.id if raw.spec else ""

    for prefix, adapter in _WORLD_ADAPTERS.items():
        if prefix in env_id:
            return adapter(env, n_envs)

    logger.warning(
        "No specific world adapter for %s, using open_table_world", env_id
    )
    return open_table_world(env, n_envs)
