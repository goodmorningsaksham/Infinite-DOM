"""
Reward function for the Infinite DOM.

Design principles:
  A. Dense progression signal — reward each time a new task-graph node is completed
  B. Small step penalty — prevent wandering
  C. Invalid-action penalty — teach action-space discipline without killing exploration
  D. Completion bonus — terminal task success gets a bonus
  E. Anti-thrash penalty — repeated failed actions are penalized

Returns both the scalar total and a component breakdown for debugging and W&B logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    progression: float
    step: float
    invalid: float
    completion: float
    thrash: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "progression": self.progression,
            "step": self.step,
            "invalid": self.invalid,
            "completion": self.completion,
            "thrash": self.thrash,
            "total": self.total,
        }


STEP_PENALTY = -0.01
INVALID_ACTION_PENALTY = -0.05
COMPLETION_BONUS = 1.0
THRASH_PENALTY = -0.2
THRASH_THRESHOLD = 3


def _is_thrashing(action_history: list[dict[str, Any]]) -> bool:
    """Detect degenerate loops — N identical failed actions in a row."""
    if len(action_history) < THRASH_THRESHOLD:
        return False
    last_n = action_history[-THRASH_THRESHOLD:]
    all_failed = all(not a.get("succeeded", True) for a in last_n)
    all_same = len(
        {(a.get("action_type"), a.get("element_ref"), a.get("text_value", "")) for a in last_n}
    ) == 1
    return all_failed and all_same


def compute_reward(
    newly_completed_nodes: list[str],
    node_weights: dict[str, float],
    action_succeeded: bool,
    is_episode_complete: bool,
    action_history: list[dict[str, Any]],
) -> RewardBreakdown:
    progression = sum(node_weights.get(n, 0.0) for n in newly_completed_nodes)
    step = STEP_PENALTY
    invalid = 0.0 if action_succeeded else INVALID_ACTION_PENALTY
    completion = COMPLETION_BONUS if is_episode_complete else 0.0
    thrash = THRASH_PENALTY if _is_thrashing(action_history) else 0.0

    total = progression + step + invalid + completion + thrash
    return RewardBreakdown(
        progression=progression,
        step=step,
        invalid=invalid,
        completion=completion,
        thrash=thrash,
        total=total,
    )
