"""Dataclasses for the stick-push receding-horizon planner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ObjectState:
    name: str
    center_xyz: np.ndarray
    radius: float
    is_target: bool
    active: bool
    footprint_type: str = "circle"
    footprint_params: dict = field(default_factory=dict)


@dataclass
class SceneState:
    target: ObjectState
    blockers: list[ObjectState]
    all_objects: list[ObjectState]
    shelf_front_x: float
    shelf_back_x: float
    shelf_half_w: float
    surface_z: float
    open_dir_xy: np.ndarray


@dataclass
class PlannerAction:
    blocker_name: str
    theta_deg: float
    x_approach: float
    delta_len_idx: int
    contact_offset: float
    push_len: float
    approach_xyz: np.ndarray
    entry_xyz: np.ndarray
    sweep_xyz: np.ndarray
    retract_xyz: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class NodeMetrics:
    blockers: int
    min_margin: float
    deficit: float
    pushes_used: int
    solved: bool


@dataclass
class SearchNode:
    node_id: int
    parent_id: int | None
    parent_action: PlannerAction | None
    sim_state: torch.Tensor
    metrics: NodeMetrics
    depth: int
    visits: int
    reward_sum: float
    untried_actions: list[PlannerAction] = field(default_factory=list)
    child_ids: list[int] = field(default_factory=list)
    state_hash: str = ""
