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

    def _score_terms(self, node: SearchNode, total_visits: int, *, lex_rank: int) -> tuple[float, float, float]:
        # Primary priority comes from lexicographic ordering over scene metrics.
        # rank=0 is best, so higher score for lower rank.
        lex_score = -float(lex_rank)
        # Exploration discourages repeatedly selecting the same node.
        explore = float(self.cfg.exploration_c) * math.sqrt(
            math.log(float(total_visits) + 1.0) / (float(node.visits) + 1.0)
        )
        untried = float(self.cfg.untried_bonus) if node.untried_actions else 0.0
        return lex_score, explore, untried

    def score(self, node: SearchNode, total_visits: int, *, lex_rank: int) -> float:
        if not node.untried_actions:
            return float("-inf")
        lex_score, explore, untried = self._score_terms(node, total_visits, lex_rank=lex_rank)
        return lex_score + explore + untried

    def ranked_candidates(self, nodes: dict[int, SearchNode], candidate_ids: Iterable[int]) -> list[dict]:
        ids = list(candidate_ids)
        if not ids:
            return []

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
        rows: list[dict] = []
        for node_id in ids:
            node = nodes[node_id]
            lex_rank = int(rank_map[node_id])
            ucb_score = self.score(node, total_visits, lex_rank=lex_rank)
            if math.isinf(ucb_score) and ucb_score < 0.0:
                lex_score = float("-inf")
                explore_bonus = 0.0
                untried_bonus = 0.0
            else:
                lex_score, explore_bonus, untried_bonus = self._score_terms(
                    node,
                    total_visits,
                    lex_rank=lex_rank,
                )
            rows.append(
                {
                    "node_id": int(node_id),
                    "lex_rank": int(lex_rank),
                    "lex_score": float(lex_score),
                    "explore_bonus": float(explore_bonus),
                    "untried_bonus": float(untried_bonus),
                    "ucb_score": float(ucb_score),
                    "depth": int(node.depth),
                }
            )
        rows.sort(
            key=lambda row: (
                float(row["ucb_score"]),
                -int(row["lex_rank"]),
                -int(row["depth"]),
                -int(row["node_id"]),
            ),
            reverse=True,
        )
        for order, row in enumerate(rows):
            row["rank_order"] = int(order)
            row["total_visits"] = int(total_visits)
        return rows

    def select(self, nodes: dict[int, SearchNode], candidate_ids: Iterable[int]) -> int:
        rows = self.ranked_candidates(nodes, candidate_ids)
        if not rows:
            raise ValueError("FrontierUCB.select requires at least one candidate id")
        return int(rows[0]["node_id"])
