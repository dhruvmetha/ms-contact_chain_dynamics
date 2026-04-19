"""Scene state providers for stick-push receding-horizon planner."""

from __future__ import annotations

import numpy as np

from taskbench.planners.stickpush_rh.geometry import collect_blockers_in_clearance
from taskbench.planners.stickpush_rh.perception_interface import SceneStateProvider
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


class GTSceneStateProvider(SceneStateProvider):
    """Reads object/shelf state directly from the simulator."""

    def __init__(self, env_index: int = 0, active_x_threshold: float = 2.0):
        self.env_index = int(env_index)
        self.active_x_threshold = float(active_x_threshold)

    def get_scene_state(self, env) -> SceneState:
        raw = env.unwrapped
        obj_map = raw.get_objects()
        target_name = f"cyl_{int(raw.target_idx)}"
        radius = float(getattr(raw.cyl_spec, "radius", 0.018))

        all_objects: list[ObjectState] = []
        target_state: ObjectState | None = None
        for name, actor in obj_map.items():
            center = actor.pose.p[self.env_index].detach().cpu().numpy().astype(np.float32)
            active = bool(center[0] < self.active_x_threshold)
            state = ObjectState(
                name=name,
                center_xyz=center,
                radius=radius,
                is_target=(name == target_name),
                active=active,
            )
            all_objects.append(state)
            if state.is_target:
                target_state = state

        if target_state is None:
            # Fallback keeps planner robust if env naming differs.
            target_state = next(obj for obj in all_objects if obj.active)
            target_state.is_target = True

        g = raw.shelf_geom
        blockers = [obj for obj in all_objects if obj.active and not obj.is_target]
        scene = SceneState(
            target=target_state,
            blockers=blockers,
            all_objects=all_objects,
            shelf_front_x=float(g.front_x),
            shelf_back_x=float(g.back_x),
            shelf_half_w=float(g.half_w),
            surface_z=float(g.surface_z),
            open_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
        )
        return scene

    def blockers_in_clearance(self, scene: SceneState, clearance_radius: float):
        return collect_blockers_in_clearance(scene, clearance_radius)

