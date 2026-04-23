"""
Per-episode graders.

CRITICAL: all returned scores MUST be strictly in (0.01, 0.99).
The OpenEnv validator rejects exact 0.0 and 1.0. Use _clamp() on every path.
"""
from __future__ import annotations

from infinite_dom.config import CONFIG
from infinite_dom.models import DOMState

SCORE_MIN = 0.01
SCORE_MAX = 0.99
_EPSILON = 1e-6


def _clamp(score: float) -> float:
    if score <= SCORE_MIN:
        return SCORE_MIN + _EPSILON
    if score >= SCORE_MAX:
        return SCORE_MAX - _EPSILON
    return score


def _completion_rate(state: DOMState) -> float:
    if not state.task_graph_total:
        return 0.0
    return len(state.task_graph_completed) / len(state.task_graph_total)


def _efficiency(state: DOMState) -> float:
    if CONFIG.max_steps == 0:
        return 0.0
    return max(0.0, 1.0 - (state.step_count / CONFIG.max_steps))


def grade_generic(state: DOMState) -> float:
    """Generic scoring — works for all tasks, weights completion highest."""
    completion = _completion_rate(state)
    efficiency = _efficiency(state)
    failure_penalty = min(state.failed_action_count * 0.05, 0.3)
    score = (0.7 * completion) + (0.3 * efficiency) - failure_penalty
    return _clamp(score)


def grade_task_1(state: DOMState) -> float:
    return grade_generic(state)


def grade_task_2(state: DOMState) -> float:
    return grade_generic(state)


def grade_task_3(state: DOMState) -> float:
    return grade_generic(state)


def grade_task_4(state: DOMState) -> float:
    base = grade_generic(state)
    if state.failed_action_count == 0 and len(state.task_graph_completed) > 0:
        base += 0.05
    return _clamp(base)


GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
    4: grade_task_4,
}


def grade(task_id: int, state: DOMState) -> float:
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return GRADERS[task_id](state)
