"""Configuration dataclasses for stick-push receding-horizon search."""

from __future__ import annotations

from dataclasses import dataclass, field

INSERTION_MODE_DIAGONAL_ALLOWED = "diagonal_allowed"
INSERTION_MODE_STRAIGHT_ONLY = "straight_only"
_VALID_INSERTION_MODES = {
    INSERTION_MODE_DIAGONAL_ALLOWED,
    INSERTION_MODE_STRAIGHT_ONLY,
}


def canonical_insertion_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower()
    if value not in _VALID_INSERTION_MODES:
        return INSERTION_MODE_DIAGONAL_ALLOWED
    return value


def active_label_for_insertion_mode(mode: str | None) -> str:
    canonical = canonical_insertion_mode(mode)
    if canonical == INSERTION_MODE_STRAIGHT_ONLY:
        return "straight"
    return "any"


@dataclass
class SamplingConfig:
    """Sampling knobs for planner action generation."""

    heading_degrees: tuple[float, ...] = (
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    )
    x_approach_values: tuple[float, ...] = (0.05, 0.07, 0.09, 0.11)
    insertion_approach_backoff: float = 0.07
    delta_len_fracs: tuple[float, ...] = (0.30, 0.60, 0.90)
    contact_margin: float = 0.003
    stick_radius: float = 0.004
    retract_backoff: float = 0.30
    z_eps: float = 0.03
    min_push_len: float = 0.03
    min_insertion_depth: float = 0.02
    world_min_clearance: float = 0.001
    forbid_target_crossing: bool = True
    target_avoid_margin: float = 0.004
    include_target_pushes: bool = True
    target_push_open_dir_bias: float = 0.5
    target_push_direction_deltas_deg: tuple[float, ...] = (
        -180.0,
        -150.0,
        -210.0,
        -90.0,
        90.0,
        120.0,
        -120.0,
    )
    target_push_include_open_face_bias: bool = True
    target_push_top_k_directions: int = 3
    target_push_wall_weight: float = 1.0
    target_push_corridor_weight: float = 0.8
    target_push_wall_penalty_weight: float = 0.05
    use_wavefront_insertion_solver: bool = True
    insertion_grid_resolution: float = 0.003
    insertion_grid_safety_eps: float = 0.0
    insertion_stick_length: float = 0.25
    insertion_entry_weight_decay: float = 0.04
    insertion_entry_uniform_mix: float = 0.15
    insertion_mode: str = INSERTION_MODE_DIAGONAL_ALLOWED  # {"diagonal_allowed", "straight_only"}


@dataclass
class UCBConfig:
    """Frontier UCB selection knobs."""

    exploration_c: float = 1.2
    untried_bonus: float = 1.0
    solved_bonus: float = 10.0


@dataclass
class VisualConfig:
    """Artifact and visualization knobs."""

    save_expansion_images: bool = True
    save_frontier_csv: bool = True
    replay_stride: int = 2
    max_candidates_drawn: int = 300
    save_timing_diagnostics: bool = False


@dataclass
class RecedingHorizonConfig:
    """Top-level planner config."""

    max_executions: int = 1000
    max_depth: int = 256
    parallel_frontier_enabled: bool = False
    parallel_frontier_batch_size: int = 1
    artifact_root: str = "artifacts/stickpush_rh"
    max_target_shift_xy: float = 0.02
    target_wall_margin: float = 0.01
    prune_target_invalid_nodes: bool = False
    require_target_shift_limit_for_success: bool = False
    grasp_success_active_label: str = "any"  # {"straight", "any"}
    grasp_success_templates_straight_deg: tuple[float, ...] = (0.0,)
    grasp_success_templates_any_deg: tuple[float, ...] = (0.0, -20.0, 20.0, -40.0, 40.0)
    grasp_success_grid_resolution: float = 0.003
    grasp_success_front_outside_offset: float = 0.06
    grasp_success_finger_thickness: float = 0.012
    grasp_success_finger_length: float = 0.05
    grasp_success_jaw_open: float = 0.06
    grasp_success_jaw_contact: float = 0.036
    grasp_success_contact_pad_width: float = 0.01
    grasp_success_contact_pad_depth: float = 0.01
    grasp_success_collision_epsilon: float = 0.0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    ucb: UCBConfig = field(default_factory=UCBConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)

    @property
    def effective_grasp_success_active_label(self) -> str:
        return active_label_for_insertion_mode(self.sampling.insertion_mode)
