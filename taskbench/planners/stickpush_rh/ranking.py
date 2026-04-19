"""Ranking helpers for stick-push RH nodes."""

from __future__ import annotations

import math

from taskbench.planners.stickpush_rh.types import NodeMetrics


def metrics_key(m: NodeMetrics) -> tuple[float, float, float, int]:
    """Lexicographic key: lower blockers, higher min_margin, lower deficit, lower pushes."""

    min_margin = m.min_margin
    if math.isinf(min_margin):
        min_margin = 1e9
    return (float(m.blockers), -float(min_margin), float(m.deficit), int(m.pushes_used))


def dominates(a: NodeMetrics, b: NodeMetrics) -> bool:
    return metrics_key(a) < metrics_key(b)


def progress_reward(
    parent: NodeMetrics,
    child: NodeMetrics,
    *,
    solved_bonus: float = 10.0,
) -> float:
    reward = 1.0 if dominates(child, parent) else 0.0
    if child.solved:
        reward += float(solved_bonus)
    return float(reward)


def metrics_to_dict(m: NodeMetrics) -> dict:
    return {
        "blockers": int(m.blockers),
        "min_margin": float(m.min_margin),
        "deficit": float(m.deficit),
        "pushes_used": int(m.pushes_used),
        "solved": bool(m.solved),
    }

