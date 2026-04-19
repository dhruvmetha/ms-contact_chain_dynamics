"""Action sampling for stick-push receding-horizon search."""

from __future__ import annotations

import math

import numpy as np

from taskbench.planners.stickpush_rh.config import SamplingConfig
from taskbench.planners.stickpush_rh.geometry import (
    collect_blockers_in_clearance,
    normalize_xy,
    point_segment_distance_xy,
    point_in_front_semicircle,
    ray_length_to_shelf_edge,
    rotate_xy,
    within_shelf_xy,
)
from taskbench.planners.stickpush_rh.types import PlannerAction, SceneState


class StickPushSampler:
    """Generate candidate stick pushes from blocker geometry."""

    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg

    @staticmethod
    def _wrap_deg(deg: float) -> float:
        v = float(deg) % 360.0
        if v < 0:
            v += 360.0
        return v

    @staticmethod
    def _deg_from_vec(v_xy: np.ndarray) -> float:
        v = np.asarray(v_xy, dtype=np.float32).reshape(2)
        return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))

    @staticmethod
    def _vec_from_deg(deg: float) -> np.ndarray:
        r = math.radians(float(deg))
        return np.array([math.cos(r), math.sin(r)], dtype=np.float32)

    def _nearest_obstacle_bearing_deg(self, scene: SceneState) -> float:
        """Bearing from target center to nearest active non-target object."""

        t_xy = scene.target.center_xyz[:2].astype(np.float32)
        nearest_obj = None
        nearest_dist = float("inf")
        for obj in scene.all_objects:
            if (not obj.active) or obj.is_target:
                continue
            d = obj.center_xyz[:2].astype(np.float32) - t_xy
            dist = float(np.linalg.norm(d))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_obj = obj

        if nearest_obj is None:
            return self._deg_from_vec(np.asarray(scene.open_dir_xy, dtype=np.float32))
        vec = nearest_obj.center_xyz[:2].astype(np.float32) - t_xy
        return self._deg_from_vec(vec)

    def _target_direction_candidates(self, scene: SceneState) -> tuple[float, list[float]]:
        nearest_deg = self._nearest_obstacle_bearing_deg(scene)
        dirs: list[float] = []

        def _add(d: float):
            dd = self._wrap_deg(d)
            if any(abs(dd - x) < 1e-5 for x in dirs):
                return
            dirs.append(dd)

        for delta in self.cfg.target_push_direction_deltas_deg:
            _add(nearest_deg + float(delta))
        if self.cfg.target_push_include_open_face_bias:
            _add(self._deg_from_vec(np.asarray(scene.open_dir_xy, dtype=np.float32)))
        return nearest_deg, dirs

    def _score_target_direction(
        self,
        scene: SceneState,
        *,
        obj_center_xy: np.ndarray,
        obj_radius: float,
        p1_xy: np.ndarray,
        p2_cap_xy: np.ndarray,
        l_cap: float,
    ) -> tuple[float, dict]:
        # 1) Along-ray distance budget to wall.
        wall_ray = float(l_cap)

        # 2) Nearest object clearance along ray corridor (surface margin estimate).
        min_corridor = float("inf")
        for other in scene.all_objects:
            if (not other.active) or other.is_target:
                continue
            d = point_segment_distance_xy(
                other.center_xyz[:2].astype(np.float32),
                p1_xy,
                p2_cap_xy,
            )
            clearance = d - (
                float(other.radius)
                + float(obj_radius)
                + float(self.cfg.stick_radius)
                + float(self.cfg.contact_margin)
            )
            min_corridor = min(min_corridor, float(clearance))
        if not np.isfinite(min_corridor):
            min_corridor = float(l_cap)

        # 3) Penalty if endpoint is near side/back walls.
        side_dist = float(scene.shelf_half_w - abs(float(p2_cap_xy[1])))
        back_dist = float(scene.shelf_back_x - float(p2_cap_xy[0]))
        side_term = 1.0 / max(1e-4, side_dist)
        back_term = 1.0 / max(1e-4, back_dist)
        wall_penalty = side_term + back_term

        score = (
            float(self.cfg.target_push_wall_weight) * wall_ray
            + float(self.cfg.target_push_corridor_weight) * min_corridor
            - float(self.cfg.target_push_wall_penalty_weight) * wall_penalty
        )
        diag = {
            "score_wall_ray": wall_ray,
            "score_corridor_min_clearance": float(min_corridor),
            "score_side_dist": side_dist,
            "score_back_dist": back_dist,
            "score_wall_penalty": float(wall_penalty),
            "score_total": float(score),
        }
        return float(score), diag

    def _target_free_space_dir(self, scene: SceneState) -> np.ndarray:
        """Direction that tends to move target away from clutter and toward opening."""

        t_xy = scene.target.center_xyz[:2].astype(np.float32)
        v = float(self.cfg.target_push_open_dir_bias) * np.asarray(
            scene.open_dir_xy, dtype=np.float32
        )
        wall_thresh = float(self.cfg.clearance_radius) + float(scene.target.radius)

        # Wall repulsion: side walls (+/-Y) and back wall (+X). Front (-X) is open.
        d_pos_y = float(scene.shelf_half_w - t_xy[1])
        d_neg_y = float(scene.shelf_half_w + t_xy[1])
        d_back = float(scene.shelf_back_x - t_xy[0])
        eps = 1e-4

        def _wall_weight(d: float) -> float:
            if d >= wall_thresh:
                return 0.0
            dd = max(d, eps)
            return (1.0 / (dd * dd)) - (1.0 / (wall_thresh * wall_thresh))

        w_pos_y = _wall_weight(d_pos_y)
        w_neg_y = _wall_weight(d_neg_y)
        w_back = _wall_weight(d_back)
        if w_pos_y > 0.0:
            v = v + np.array([0.0, -w_pos_y], dtype=np.float32)
        if w_neg_y > 0.0:
            v = v + np.array([0.0, w_neg_y], dtype=np.float32)
        if w_back > 0.0:
            v = v + np.array([-w_back, 0.0], dtype=np.float32)

        for obj in scene.blockers:
            if not obj.active:
                continue
            d = t_xy - obj.center_xyz[:2].astype(np.float32)
            dist = float(np.linalg.norm(d))
            if dist <= 1e-5:
                continue
            # Repulsion from nearby blockers; closer blockers contribute more.
            v = v + d / (dist * dist)
        return normalize_xy(v)

    def sample_actions(self, scene: SceneState) -> list[PlannerAction]:
        actions: list[PlannerAction] = []
        blockers = collect_blockers_in_clearance(scene, self.cfg.clearance_radius)
        sample_objects = list(blockers)
        if self.cfg.include_target_pushes and scene.target.active:
            sample_objects.append(scene.target)

        for obj in sample_objects:
            is_target_push = bool(obj.is_target)
            b_xy = obj.center_xyz[:2].astype(np.float32)
            t_xy = scene.target.center_xyz[:2].astype(np.float32)
            u0 = self._target_free_space_dir(scene) if is_target_push else normalize_xy(b_xy - t_xy)

            # Direction generation policy differs for target pushes.
            target_dir_scores: dict[float, tuple[float, dict]] = {}
            target_dir_sampled: list[float] = []
            target_dir_topk: list[float] = []
            target_nearest_bearing_deg: float | None = None

            if is_target_push:
                nearest_deg, dir_degs = self._target_direction_candidates(scene)
                target_nearest_bearing_deg = float(self._wrap_deg(nearest_deg))
                target_dir_sampled = [float(self._wrap_deg(d)) for d in dir_degs]
                scored_dirs: list[tuple[float, float, dict]] = []  # (score, abs_deg, diag)
                for abs_deg in dir_degs:
                    u = normalize_xy(self._vec_from_deg(abs_deg))
                    contact_offset = (
                        float(obj.radius)
                        + float(self.cfg.stick_radius)
                        + float(self.cfg.contact_margin)
                    )
                    p1_xy = b_xy - u * contact_offset
                    if not within_shelf_xy(p1_xy, scene, margin=self.cfg.world_min_clearance):
                        continue
                    ray_len = ray_length_to_shelf_edge(
                        p1_xy, u, scene, margin=self.cfg.world_min_clearance
                    )
                    if ray_len is None:
                        continue
                    l_cap = max(0.0, float(ray_len) - float(obj.radius))
                    if l_cap < float(self.cfg.min_push_len):
                        continue
                    p2_cap_xy = p1_xy + u * l_cap
                    score, diag = self._score_target_direction(
                        scene,
                        obj_center_xy=b_xy,
                        obj_radius=float(obj.radius),
                        p1_xy=p1_xy,
                        p2_cap_xy=p2_cap_xy,
                        l_cap=l_cap,
                    )
                    scored_dirs.append((float(score), float(self._wrap_deg(abs_deg)), diag))

                scored_dirs.sort(key=lambda x: x[0], reverse=True)
                k = max(1, int(self.cfg.target_push_top_k_directions))
                scored_dirs = scored_dirs[:k]
                target_dir_topk = [d[1] for d in scored_dirs]
                target_dir_scores = {d[1]: (d[0], d[2]) for d in scored_dirs}
                dir_iter: list[tuple[float, np.ndarray]] = [
                    (deg, normalize_xy(self._vec_from_deg(deg))) for deg in target_dir_topk
                ]
            else:
                dir_iter = [
                    (float(theta_deg), rotate_xy(u0, theta_deg)) for theta_deg in self.cfg.heading_degrees
                ]

            for theta_tag, u in dir_iter:
                contact_offset = (
                    float(obj.radius) + float(self.cfg.stick_radius) + float(self.cfg.contact_margin)
                )
                p1_xy = b_xy - u * contact_offset
                if not within_shelf_xy(p1_xy, scene, margin=self.cfg.world_min_clearance):
                    continue

                ray_len = ray_length_to_shelf_edge(
                    p1_xy, u, scene, margin=self.cfg.world_min_clearance
                )
                if ray_len is None:
                    continue

                l_cap = max(0.0, float(ray_len) - float(obj.radius))
                if l_cap < float(self.cfg.min_push_len):
                    continue

                # If the sampled heading points toward the target, bias to longer pushes
                # so the blocker can still be moved out of the clearance region.
                toward_target = (not is_target_push) and float(np.dot(u, u0)) < 0.0
                dist_bt = float(np.linalg.norm(b_xy - t_xy))
                base_push = float(self.cfg.min_push_len)
                if toward_target:
                    escape = max(
                        0.0,
                        float(self.cfg.clearance_radius) + float(obj.radius) + float(self.cfg.contact_margin) - dist_bt,
                    )
                    base_push += escape
                base_push = min(base_push, l_cap)

                for delta_len_idx, frac in enumerate(self.cfg.delta_len_fracs):
                    push_len = base_push + float(frac) * max(0.0, l_cap - base_push)
                    push_len = float(np.clip(push_len, self.cfg.min_push_len, l_cap))
                    p2_xy = p1_xy + u * push_len
                    if not within_shelf_xy(p2_xy, scene, margin=self.cfg.world_min_clearance):
                        continue
                    # For non-target pushes, require sweep endpoint to finish outside
                    # the target grasp-clearance semicircle.
                    if (not is_target_push) and point_in_front_semicircle(
                        p2_xy,
                        scene.target,
                        self.cfg.clearance_radius,
                        scene.open_dir_xy,
                    ):
                        continue
                    if self.cfg.forbid_target_crossing and (not is_target_push):
                        target_clearance = (
                            float(scene.target.radius)
                            + float(obj.radius)
                            + float(self.cfg.target_avoid_margin)
                        )
                        seg_dist = point_segment_distance_xy(
                            scene.target.center_xyz[:2], p1_xy, p2_xy
                        )
                        if seg_dist <= target_clearance:
                            continue

                    z_push = float(scene.surface_z + self.cfg.z_eps)
                    p1 = np.array([p1_xy[0], p1_xy[1], z_push], dtype=np.float32)
                    p2 = np.array([p2_xy[0], p2_xy[1], z_push], dtype=np.float32)
                    x_approach = float(self.cfg.insertion_approach_backoff)
                    if len(self.cfg.x_approach_values) > 0:
                        x_approach = float(self.cfg.x_approach_values[0])

                    approach = np.array(
                        [scene.shelf_front_x - x_approach, p1[1], z_push],
                        dtype=np.float32,
                    )
                    retract = np.array(
                        [scene.shelf_front_x - float(self.cfg.retract_backoff), p2[1], z_push],
                        dtype=np.float32,
                    )

                    insertion_depth = float(p1[0] - approach[0])
                    if insertion_depth < float(self.cfg.min_insertion_depth):
                        continue

                    actions.append(
                        PlannerAction(
                            blocker_name=obj.name,
                            theta_deg=float(theta_tag),
                            x_approach=float(x_approach),
                            delta_len_idx=int(delta_len_idx),
                            contact_offset=float(contact_offset),
                            push_len=float(push_len),
                            approach_xyz=approach,
                            entry_xyz=p1,
                            sweep_xyz=p2,
                            retract_xyz=retract,
                            meta={
                                "u0_xy": u0.tolist(),
                                "u_xy": u.tolist(),
                                "p1_xy": p1_xy.tolist(),
                                "p2_xy": p2_xy.tolist(),
                                "l_cap": float(l_cap),
                                "toward_target": bool(toward_target),
                                "is_target_push": bool(is_target_push),
                                "sample_object": obj.name,
                                "target_push_policy": (
                                    "nearest_obstacle_bearing_topk"
                                    if is_target_push
                                    else "default_blocker_heading_grid"
                                ),
                                "target_push_nearest_bearing_deg": (
                                    float(target_nearest_bearing_deg)
                                    if target_nearest_bearing_deg is not None
                                    else None
                                ),
                                "target_push_dir_deg": (
                                    float(self._wrap_deg(theta_tag)) if is_target_push else None
                                ),
                                "target_push_sampled_dir_degs": (
                                    [float(x) for x in target_dir_sampled]
                                    if is_target_push
                                    else None
                                ),
                                "target_push_topk_dir_degs": (
                                    [float(x) for x in target_dir_topk]
                                    if is_target_push
                                    else None
                                ),
                                "target_push_dir_score": (
                                    float(target_dir_scores.get(float(self._wrap_deg(theta_tag)), (0.0, {}))[0])
                                    if is_target_push
                                    else None
                                ),
                                "target_push_score_diag": (
                                    dict(target_dir_scores.get(float(self._wrap_deg(theta_tag)), (0.0, {}))[1])
                                    if is_target_push
                                    else None
                                ),
                            },
                        )
                    )
        return actions
