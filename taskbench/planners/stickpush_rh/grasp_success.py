"""Fast shape-aware graspability heuristic for RH stick-push search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from taskbench.planners.stickpush_rh.config import RecedingHorizonConfig
from taskbench.planners.stickpush_rh.geometry import hash_scene_state, normalize_xy, rotate_xy
from taskbench.planners.stickpush_rh.types import ObjectState, SceneState


@dataclass
class TemplateCheckResult:
    angle_deg: float
    passed: bool
    fail_reason: str | None
    insertion_dir_xy: np.ndarray
    entry_xy: np.ndarray | None
    blocking_object_names: list[str] = field(default_factory=list)
    wall_blocked: bool = False
    left_contact: bool = False
    right_contact: bool = False
    movable_overlap: float = 0.0
    wall_overlap: float = 0.0
    contact_deficit: float = 0.0
    total_violation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "angle_deg": float(self.angle_deg),
            "passed": bool(self.passed),
            "fail_reason": self.fail_reason,
            "insertion_dir_xy": np.asarray(self.insertion_dir_xy, dtype=np.float32).tolist(),
            "entry_xy": (
                None
                if self.entry_xy is None
                else np.asarray(self.entry_xy, dtype=np.float32).tolist()
            ),
            "blocking_object_names": [str(x) for x in self.blocking_object_names],
            "wall_blocked": bool(self.wall_blocked),
            "left_contact": bool(self.left_contact),
            "right_contact": bool(self.right_contact),
            "movable_overlap": float(self.movable_overlap),
            "wall_overlap": float(self.wall_overlap),
            "contact_deficit": float(self.contact_deficit),
            "total_violation": float(self.total_violation),
        }


@dataclass
class GraspSuccessResult:
    graspable_straight: bool
    graspable_any: bool
    active_label: str
    active_success: bool
    active_template_results: list[TemplateCheckResult]
    all_template_results: list[TemplateCheckResult]
    primary_failing_template: TemplateCheckResult | None
    display_template: TemplateCheckResult | None
    primary_blocking_objects: list[str]
    blocking_objects_union: list[str]
    primary_penetration_sum: float
    primary_min_margin: float

    def to_dict(self) -> dict:
        return {
            "graspable_straight": bool(self.graspable_straight),
            "graspable_any": bool(self.graspable_any),
            "active_label": str(self.active_label),
            "active_success": bool(self.active_success),
            "primary_blocking_objects": [str(x) for x in self.primary_blocking_objects],
            "blocking_objects_union": [str(x) for x in self.blocking_objects_union],
            "primary_penetration_sum": float(self.primary_penetration_sum),
            "primary_min_margin": float(self.primary_min_margin),
            "primary_failing_template": (
                None if self.primary_failing_template is None else self.primary_failing_template.to_dict()
            ),
            "display_template": (
                None if self.display_template is None else self.display_template.to_dict()
            ),
            "template_results_active": [r.to_dict() for r in self.active_template_results],
            "template_results_all": [r.to_dict() for r in self.all_template_results],
        }


@dataclass
class _OccupancyCache:
    xx: np.ndarray
    yy: np.ndarray
    resolution: float
    inside_open_region: np.ndarray
    wall_occ: np.ndarray
    occ_target: np.ndarray
    occ_non_target: np.ndarray
    occ_non_target_or_wall: np.ndarray
    object_masks: dict[str, np.ndarray]
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class GraspSuccessEvaluator:
    """Compute graspability labels and blocker diagnostics from occupancy."""

    def __init__(self, cfg: RecedingHorizonConfig):
        self.cfg = cfg
        self._cache: dict[str, _OccupancyCache] = {}

    @staticmethod
    def _object_mask(
        scene: SceneState,
        obj: ObjectState,
        xx: np.ndarray,
        yy: np.ndarray,
    ) -> np.ndarray:
        center = np.asarray(obj.center_xyz[:2], dtype=np.float64).reshape(2)
        shape = str(getattr(obj, "footprint_type", "circle") or "circle")
        params = dict(getattr(obj, "footprint_params", {}) or {})
        if shape == "obox":
            he = np.asarray(params.get("half_extents", [obj.radius, obj.radius]), dtype=np.float64).reshape(-1)
            if he.size < 2:
                he = np.array([float(obj.radius), float(obj.radius)], dtype=np.float64)
            yaw = float(params.get("yaw", 0.0))
            c, s = math.cos(yaw), math.sin(yaw)
            u = np.array([c, s], dtype=np.float64)
            v = np.array([-s, c], dtype=np.float64)
            dx = xx - center[0]
            dy = yy - center[1]
            du = dx * u[0] + dy * u[1]
            dv = dx * v[0] + dy * v[1]
            return (np.abs(du) <= float(he[0])) & (np.abs(dv) <= float(he[1]))

        # Default footprint: circle.
        rad = float(params.get("radius", obj.radius))
        return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= (rad * rad)

    def _build_occupancy(self, scene: SceneState, *, state_hash: str) -> _OccupancyCache:
        cached = self._cache.get(state_hash)
        if cached is not None:
            return cached

        res = float(max(1e-4, self.cfg.grasp_success_grid_resolution))
        front_pad = float(max(0.0, self.cfg.grasp_success_front_outside_offset))
        x_min = float(scene.shelf_front_x - front_pad)
        x_max = float(scene.shelf_back_x + res)
        y_min = float(-scene.shelf_half_w - res)
        y_max = float(scene.shelf_half_w + res)

        nx = int(max(1, np.floor((x_max - x_min) / res) + 1))
        ny = int(max(1, np.floor((y_max - y_min) / res) + 1))
        xs = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * res
        ys = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys, indexing="xy")

        inside_open_region = (xx < float(scene.shelf_front_x)) | (
            (xx <= float(scene.shelf_back_x)) & (np.abs(yy) <= float(scene.shelf_half_w))
        )
        wall_occ = ~inside_open_region

        occ_target = np.zeros_like(xx, dtype=bool)
        occ_non_target = np.zeros_like(xx, dtype=bool)
        object_masks: dict[str, np.ndarray] = {}

        for obj in scene.all_objects:
            if not obj.active:
                continue
            mask = self._object_mask(scene, obj, xx, yy)
            if obj.is_target:
                occ_target |= mask
            else:
                occ_non_target |= mask
                object_masks[str(obj.name)] = mask

        occ_non_target_or_wall = occ_non_target | wall_occ
        cache = _OccupancyCache(
            xx=xx,
            yy=yy,
            resolution=res,
            inside_open_region=inside_open_region,
            wall_occ=wall_occ,
            occ_target=occ_target,
            occ_non_target=occ_non_target,
            occ_non_target_or_wall=occ_non_target_or_wall,
            object_masks=object_masks,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        self._cache[state_hash] = cache
        return cache

    @staticmethod
    def _rect_mask(
        xx: np.ndarray,
        yy: np.ndarray,
        *,
        center: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        half_len_u: float,
        half_len_v: float,
        eps: float = 0.0,
    ) -> np.ndarray:
        c = np.asarray(center, dtype=np.float64).reshape(2)
        uu = normalize_xy(np.asarray(u, dtype=np.float64).reshape(2))
        vv = normalize_xy(np.asarray(v, dtype=np.float64).reshape(2))
        dx = xx - float(c[0])
        dy = yy - float(c[1])
        du = dx * float(uu[0]) + dy * float(uu[1])
        dv = dx * float(vv[0]) + dy * float(vv[1])
        return (np.abs(du) <= float(half_len_u) + float(eps)) & (
            np.abs(dv) <= float(half_len_v) + float(eps)
        )

    def _entry_on_front(
        self,
        scene: SceneState,
        target_xy: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray | None:
        # target + s*u intersects front plane x=front_x
        ux = float(u[0])
        if abs(ux) < 1e-8:
            return None
        s = (float(scene.shelf_front_x) - float(target_xy[0])) / ux
        if s <= 0.0:
            return None
        p = np.asarray(target_xy, dtype=np.float64).reshape(2) + s * np.asarray(u, dtype=np.float64).reshape(2)
        if abs(float(p[1])) > float(scene.shelf_half_w):
            return None
        return p.astype(np.float64)

    def _check_template(
        self,
        scene: SceneState,
        occ: _OccupancyCache,
        *,
        angle_deg: float,
    ) -> TemplateCheckResult:
        target_xy = np.asarray(scene.target.center_xyz[:2], dtype=np.float64)
        open_dir = normalize_xy(np.asarray(scene.open_dir_xy, dtype=np.float64))
        u = normalize_xy(rotate_xy(open_dir, float(angle_deg)).astype(np.float64))
        v = normalize_xy(np.array([-u[1], u[0]], dtype=np.float64))
        eps = float(max(0.0, self.cfg.grasp_success_collision_epsilon))

        entry = self._entry_on_front(scene, target_xy, u)
        if entry is None:
            return TemplateCheckResult(
                angle_deg=float(angle_deg),
                passed=False,
                fail_reason="invalid_entry",
                insertion_dir_xy=u.astype(np.float32),
                entry_xy=None,
                total_violation=1.0,
            )

        outside = entry + float(self.cfg.grasp_success_front_outside_offset) * u
        path_len = float(np.linalg.norm(target_xy - outside))
        path_mid = 0.5 * (outside + target_xy)
        finger_t = float(self.cfg.grasp_success_finger_thickness)
        finger_l = float(self.cfg.grasp_success_finger_length)
        jaw_open = float(self.cfg.grasp_success_jaw_open)
        jaw_contact = float(self.cfg.grasp_success_jaw_contact)
        pad_w = float(self.cfg.grasp_success_contact_pad_width)
        pad_d = float(self.cfg.grasp_success_contact_pad_depth)

        open_offset = 0.5 * jaw_open + 0.5 * finger_t
        close_offset = 0.5 * jaw_contact + 0.5 * finger_t
        half_w = 0.5 * finger_t
        half_l = 0.5 * finger_l
        corridor_half_l = 0.5 * path_len + half_l

        # Insertion sweep (fingers open).
        corridor_masks: list[np.ndarray] = []
        for sgn in (-1.0, 1.0):
            center = path_mid + sgn * open_offset * v
            corridor_masks.append(
                self._rect_mask(
                    occ.xx,
                    occ.yy,
                    center=center,
                    u=u,
                    v=v,
                    half_len_u=corridor_half_l,
                    half_len_v=half_w,
                    eps=eps,
                )
            )

        # Final finger bodies (gripper closed to contact gap).
        finger_masks: list[np.ndarray] = []
        for sgn in (-1.0, 1.0):
            center = target_xy + sgn * close_offset * v
            finger_masks.append(
                self._rect_mask(
                    occ.xx,
                    occ.yy,
                    center=center,
                    u=u,
                    v=v,
                    half_len_u=half_l,
                    half_len_v=half_w,
                    eps=eps,
                )
            )

        # Closure sweep at final pose.
        closure_half_w = abs(0.5 * jaw_open - 0.5 * jaw_contact) + half_w
        closure_masks: list[np.ndarray] = []
        for sgn in (-1.0, 1.0):
            center = target_xy + sgn * (0.25 * (jaw_open + jaw_contact) + 0.5 * finger_t) * v
            closure_masks.append(
                self._rect_mask(
                    occ.xx,
                    occ.yy,
                    center=center,
                    u=u,
                    v=v,
                    half_len_u=half_l,
                    half_len_v=closure_half_w,
                    eps=eps,
                )
            )

        # Contact pads at inner finger faces.
        left_pad = self._rect_mask(
            occ.xx,
            occ.yy,
            center=target_xy - 0.5 * jaw_contact * v,
            u=u,
            v=v,
            half_len_u=0.5 * pad_d,
            half_len_v=0.5 * pad_w,
            eps=eps,
        )
        right_pad = self._rect_mask(
            occ.xx,
            occ.yy,
            center=target_xy + 0.5 * jaw_contact * v,
            u=u,
            v=v,
            half_len_u=0.5 * pad_d,
            half_len_v=0.5 * pad_w,
            eps=eps,
        )

        eval_masks = corridor_masks + finger_masks + closure_masks
        collision_mask = np.zeros_like(occ.xx, dtype=bool)
        for m in eval_masks:
            collision_mask |= m

        wall_blocked = bool(np.any(collision_mask & occ.wall_occ))
        wall_overlap = float(np.count_nonzero(collision_mask & occ.wall_occ)) * occ.resolution

        moving_overlap = float(np.count_nonzero(collision_mask & occ.occ_non_target)) * occ.resolution
        blocking_names = []
        for name, mask in occ.object_masks.items():
            if bool(np.any(collision_mask & mask)):
                blocking_names.append(str(name))

        left_contact = bool(np.any(left_pad & occ.occ_target))
        right_contact = bool(np.any(right_pad & occ.occ_target))
        contact_deficit = 0.0
        if not left_contact:
            contact_deficit += 1.0
        if not right_contact:
            contact_deficit += 1.0

        fail_reason = None
        if moving_overlap > 0.0 or wall_overlap > 0.0:
            # Prefer finer labels when possible.
            if moving_overlap > 0.0:
                # Heuristic split: if any corridor overlap, classify as approach collision.
                approach_hit = any(bool(np.any(m & occ.occ_non_target_or_wall)) for m in corridor_masks)
                closure_hit = any(bool(np.any(m & occ.occ_non_target_or_wall)) for m in closure_masks)
                if approach_hit:
                    fail_reason = "approach_collision"
                elif closure_hit:
                    fail_reason = "closure_collision"
                else:
                    fail_reason = "finger_pose_collision"
            else:
                approach_hit = any(bool(np.any(m & occ.wall_occ)) for m in corridor_masks)
                closure_hit = any(bool(np.any(m & occ.wall_occ)) for m in closure_masks)
                if approach_hit:
                    fail_reason = "approach_collision"
                elif closure_hit:
                    fail_reason = "closure_collision"
                else:
                    fail_reason = "finger_pose_collision"
        elif not left_contact:
            fail_reason = "missing_left_contact"
        elif not right_contact:
            fail_reason = "missing_right_contact"

        passed = fail_reason is None
        total_violation = float(moving_overlap + wall_overlap + contact_deficit)
        return TemplateCheckResult(
            angle_deg=float(angle_deg),
            passed=bool(passed),
            fail_reason=fail_reason,
            insertion_dir_xy=u.astype(np.float32),
            entry_xy=np.asarray(entry, dtype=np.float32),
            blocking_object_names=sorted(set(blocking_names)),
            wall_blocked=bool(wall_blocked),
            left_contact=bool(left_contact),
            right_contact=bool(right_contact),
            movable_overlap=float(moving_overlap),
            wall_overlap=float(wall_overlap),
            contact_deficit=float(contact_deficit),
            total_violation=float(total_violation),
        )

    @staticmethod
    def _primary_failing_template(rows: list[TemplateCheckResult]) -> TemplateCheckResult | None:
        failing = [r for r in rows if not r.passed]
        if not failing:
            return None
        return min(failing, key=lambda r: (float(r.total_violation), len(r.blocking_object_names), abs(float(r.angle_deg))))

    @staticmethod
    def _first_passing_template(rows: list[TemplateCheckResult]) -> TemplateCheckResult | None:
        for r in rows:
            if r.passed:
                return r
        return None

    def evaluate(
        self,
        scene: SceneState,
        *,
        state_hash: str | None = None,
        active_label_override: str | None = None,
    ) -> GraspSuccessResult:
        if state_hash is None:
            state_hash = hash_scene_state(scene)
        occ = self._build_occupancy(scene, state_hash=state_hash)

        straight_angles: list[float] = [float(a) for a in self.cfg.grasp_success_templates_straight_deg]
        if not straight_angles:
            straight_angles = [0.0]

        any_angles: list[float] = []
        seen_any: set[float] = set()
        # `graspable_any` is required to include the straight templates.
        for angle in [*straight_angles, *[float(a) for a in self.cfg.grasp_success_templates_any_deg]]:
            key = float(angle)
            if key in seen_any:
                continue
            seen_any.add(key)
            any_angles.append(key)

        straight_rows = [self._check_template(scene, occ, angle_deg=angle) for angle in straight_angles]
        any_rows = [self._check_template(scene, occ, angle_deg=angle) for angle in any_angles]
        all_rows = list(any_rows)

        graspable_straight = bool(any(r.passed for r in straight_rows))
        graspable_any = bool(any(r.passed for r in any_rows))
        label_source = active_label_override if active_label_override is not None else self.cfg.grasp_success_active_label
        active_label = str(label_source).strip().lower()
        if active_label not in {"straight", "any"}:
            active_label = "any"
        active_rows = straight_rows if active_label == "straight" else any_rows
        active_success = graspable_straight if active_label == "straight" else graspable_any

        primary_fail: TemplateCheckResult | None = None
        if not active_success:
            primary_fail = self._primary_failing_template(active_rows)
        display = self._first_passing_template(active_rows)
        if display is None:
            display = primary_fail

        blocking_union: set[str] = set()
        for row in active_rows:
            blocking_union.update(str(x) for x in row.blocking_object_names)
        primary_blockers = [] if primary_fail is None else sorted(set(primary_fail.blocking_object_names))

        if primary_fail is None:
            primary_penetration_sum = 0.0
            primary_min_margin = float("inf")
        else:
            if len(primary_fail.blocking_object_names) > 0:
                primary_penetration_sum = float(primary_fail.movable_overlap)
            else:
                primary_penetration_sum = float(primary_fail.wall_overlap + primary_fail.contact_deficit)
            primary_min_margin = -float(primary_penetration_sum) if primary_penetration_sum > 0.0 else 0.0

        return GraspSuccessResult(
            graspable_straight=bool(graspable_straight),
            graspable_any=bool(graspable_any),
            active_label=active_label,
            active_success=bool(active_success),
            active_template_results=active_rows,
            all_template_results=all_rows,
            primary_failing_template=primary_fail,
            display_template=display,
            primary_blocking_objects=primary_blockers,
            blocking_objects_union=sorted(blocking_union),
            primary_penetration_sum=float(primary_penetration_sum),
            primary_min_margin=float(primary_min_margin),
        )
