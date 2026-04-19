from __future__ import annotations

import numpy as np
import torch

from taskbench.planners.stickpush_rh.config import UCBConfig
from taskbench.planners.stickpush_rh.frontier_ucb import FrontierUCB
from taskbench.planners.stickpush_rh.types import NodeMetrics, PlannerAction, SearchNode


def _dummy_action() -> PlannerAction:
    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return PlannerAction(
        blocker_name="b0",
        theta_deg=0.0,
        x_approach=0.05,
        delta_len_idx=0,
        contact_offset=0.02,
        push_len=0.05,
        approach_xyz=p.copy(),
        entry_xyz=p.copy(),
        sweep_xyz=p.copy(),
        retract_xyz=p.copy(),
        meta={},
    )


def _node(node_id: int, *, untried: int, visits: int, reward: float) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        parent_id=None,
        parent_action=None,
        sim_state=torch.tensor([0.0]),
        metrics=NodeMetrics(blockers=1, min_margin=0.0, deficit=1.0, pushes_used=0, solved=False),
        depth=0,
        visits=visits,
        reward_sum=reward,
        untried_actions=[_dummy_action() for _ in range(untried)],
        child_ids=[],
        state_hash=f"s{node_id}",
    )


def test_ucb_prefers_node_with_untried_actions():
    ucb = FrontierUCB(UCBConfig(exploration_c=0.0, untried_bonus=1.0, solved_bonus=10.0))
    nodes = {
        0: _node(0, untried=0, visits=10, reward=10.0),
        1: _node(1, untried=1, visits=10, reward=10.0),
    }
    assert ucb.select(nodes, [0, 1]) == 1


def test_ucb_switches_when_first_node_has_no_untried():
    ucb = FrontierUCB(UCBConfig(exploration_c=1.0, untried_bonus=0.5, solved_bonus=10.0))
    nodes = {
        0: _node(0, untried=1, visits=1, reward=0.0),
        1: _node(1, untried=1, visits=1, reward=0.0),
    }
    first = ucb.select(nodes, [0, 1])
    nodes[first].untried_actions = []
    second = ucb.select(nodes, [0, 1])
    assert first != second

