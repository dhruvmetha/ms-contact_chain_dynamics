"""Lexicographic-first frontier selection with visit-based exploration."""

from __future__ import annotations

import math
from typing import Iterable

from taskbench.planners.stickpush_rh.config import UCBConfig
from taskbench.planners.stickpush_rh.ranking import metrics_key
from taskbench.planners.stickpush_rh.types import SearchNode


class FrontierUCB:
    def __init__(self, cfg: UCBConfig):
        self.cfg = cfg

    def score(self, node: SearchNode, total_visits: int, *, lex_rank: int) -> float:
        if not node.untried_actions:
            return float("-inf")
        # Primary priority comes from lexicographic ordering over scene metrics.
        # rank=0 is best, so higher score for lower rank.
        lex_score = -float(lex_rank)
        # Exploration discourages repeatedly selecting the same node.
        explore = float(self.cfg.exploration_c) * math.sqrt(
            math.log(float(total_visits) + 1.0) / (float(node.visits) + 1.0)
        )
        untried = float(self.cfg.untried_bonus) if node.untried_actions else 0.0
        return lex_score + explore + untried

    def select(self, nodes: dict[int, SearchNode], candidate_ids: Iterable[int]) -> int:
        ids = list(candidate_ids)
        if not ids:
            raise ValueError("FrontierUCB.select requires at least one candidate id")

        total_visits = max(1, sum(max(n.visits, 1) for n in nodes.values()))
        ranked = sorted(
            ids,
            key=lambda node_id: (
                metrics_key(nodes[node_id].metrics),
                nodes[node_id].depth,
                node_id,
            ),
        )
        rank_map = {node_id: rank for rank, node_id in enumerate(ranked)}
        scored = [
            (
                self.score(nodes[node_id], total_visits, lex_rank=rank_map[node_id]),
                -rank_map[node_id],
                -nodes[node_id].depth,
                -node_id,
                node_id,
            )
            for node_id in ids
        ]
        scored.sort(reverse=True)
        return int(scored[0][-1])
