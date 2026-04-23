## PHASE 3 — Task Graph, Rewards, Graders

### 3.1 Goal

Pure logic layer — no browser, no generator yet. Build:
1. A task graph data structure that represents the semantic steps of a task and can be checked for completion given some observed "browser state"
2. A reward function that computes per-step rewards
3. Grader functions (per-task episode-end scoring, strictly in `(0.01, 0.99)`)

This phase is high-leverage: everything else depends on these semantics being right.

### 3.2 Execution Checklist

#### 3.2.1 Create `infinite_dom/task_graph.py`

```python
"""
Task graph: the semantic representation of a task.

A task is a set of nodes (semantic checkpoints) that must be satisfied.
Each node is satisfied when some predicate over the current browser
state is true. The graph has soft ordering (edges are hints, not
hard dependencies) — some nodes can be completed in any order, but
final completion requires all nodes to be marked complete.

This module is browser-agnostic; it takes an opaque BrowserState dict
with keys describing the observed page (input values, selected options,
URL, visible-element facts) and checks node predicates against it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


BrowserState = dict[str, Any]
"""
Opaque dict describing what the environment can observe about the browser.
Expected (but not required) keys the generator + browser driver populate:

- inputs: dict[str, str]       # fieldName -> current value
- selected_options: dict[str, str]  # dropdownName -> selected value
- url: str
- visible_text: set[str]       # normalized-lowercased visible text tokens
- clicked_element_refs: list[str]  # history of clicked refs
- confirmation_visible: bool
- booking_id_visible: str | None
"""


@dataclass
class TaskNode:
    node_id: str
    description: str
    predicate: Callable[[BrowserState], bool]
    weight: float = 0.1  # reward weight when this node is first satisfied


@dataclass
class TaskGraph:
    task_id: int
    template_id: str
    instruction: str
    nodes: list[TaskNode]
    edges: list[tuple[str, str]] = field(default_factory=list)
    # Task-level parameters (origin, destination, class) chosen at generation time
    params: dict[str, str] = field(default_factory=dict)

    @property
    def all_node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    @property
    def node_weights(self) -> dict[str, float]:
        return {n.node_id: n.weight for n in self.nodes}

    def get_completed_nodes(self, browser_state: BrowserState) -> list[str]:
        """Return list of node IDs whose predicates evaluate to True."""
        completed = []
        for node in self.nodes:
            try:
                if node.predicate(browser_state):
                    completed.append(node.node_id)
            except Exception:
                # Predicate errors count as "not completed"
                continue
        return completed

    def is_fully_complete(self, browser_state: BrowserState) -> bool:
        return set(self.get_completed_nodes(browser_state)) == set(self.all_node_ids)


# =========================================================================
# Canonical task graph factory for the booking flow archetype
# =========================================================================


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def make_booking_flow_graph(
    task_id: int,
    template_id: str,
    origin: str,
    destination: str,
    seat_class: str,
    instruction: str,
) -> TaskGraph:
    """
    Construct the canonical booking-flow task graph.

    Nodes represent semantic checkpoints:
      1. origin_set        — origin field contains the target origin
      2. destination_set   — destination field contains the target destination
      3. class_selected    — class dropdown has the target class selected
      4. search_submitted  — search results page has loaded
      5. booking_confirmed — confirmation element contains a booking reference
    """
    target_origin = _norm(origin)
    target_destination = _norm(destination)
    target_class = _norm(seat_class)

    def _any_input_contains(state: BrowserState, semantic_names: tuple[str, ...], target: str) -> bool:
        inputs: dict[str, str] = state.get("inputs", {})
        for name, val in inputs.items():
            if any(sem in _norm(name) for sem in semantic_names):
                if target in _norm(val):
                    return True
        return False

    def origin_predicate(state: BrowserState) -> bool:
        return _any_input_contains(
            state,
            ("origin", "from", "depart", "start"),
            target_origin,
        )

    def destination_predicate(state: BrowserState) -> bool:
        return _any_input_contains(
            state,
            ("destination", "to", "arrive", "end"),
            target_destination,
        )

    def class_predicate(state: BrowserState) -> bool:
        sel: dict[str, str] = state.get("selected_options", {})
        for name, val in sel.items():
            name_norm = _norm(name)
            val_norm = _norm(val)
            if any(sem in name_norm for sem in ("class", "cabin", "fare", "seat")):
                if target_class in val_norm:
                    return True
        return False

    def search_predicate(state: BrowserState) -> bool:
        # Either the URL changed to include 'results', or a visible results token appeared
        url = _norm(state.get("url", ""))
        if "results" in url or "search" in url:
            return True
        vis: set[str] = state.get("visible_text", set())
        vis_norm = {_norm(v) for v in vis}
        return any(tok in vis_norm for tok in {"results", "available trains", "found trains"})

    def confirmation_predicate(state: BrowserState) -> bool:
        if state.get("confirmation_visible"):
            return True
        bid = state.get("booking_id_visible")
        return bool(bid)

    nodes = [
        TaskNode("origin_set", "Enter origin", origin_predicate, weight=0.15),
        TaskNode("destination_set", "Enter destination", destination_predicate, weight=0.15),
        TaskNode("class_selected", "Select class", class_predicate, weight=0.15),
        TaskNode("search_submitted", "Submit search", search_predicate, weight=0.2),
        TaskNode("booking_confirmed", "Confirm booking", confirmation_predicate, weight=0.35),
    ]

    edges = [
        ("origin_set", "destination_set"),
        ("destination_set", "class_selected"),
        ("class_selected", "search_submitted"),
        ("search_submitted", "booking_confirmed"),
    ]

    return TaskGraph(
        task_id=task_id,
        template_id=template_id,
        instruction=instruction,
        nodes=nodes,
        edges=edges,
        params={
            "origin": origin,
            "destination": destination,
            "class": seat_class,
        },
    )
```

#### 3.2.2 Create `infinite_dom/reward_calculator.py`

```python
"""
Reward function for the Infinite DOM.

Design principles (from master plan §9):
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
THRASH_THRESHOLD = 3  # same failed action this many times in a row triggers thrash


def _is_thrashing(action_history: list[dict[str, Any]]) -> bool:
    """Detect degenerate loops — N identical failed actions in a row."""
    if len(action_history) < THRASH_THRESHOLD:
        return False
    last_n = action_history[-THRASH_THRESHOLD:]
    all_failed = all(not a.get("succeeded", True) for a in last_n)
    all_same = len({(a.get("action_type"), a.get("element_ref"), a.get("text_value", "")) for a in last_n}) == 1
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
```

#### 3.2.3 Create `infinite_dom/graders.py`

```python
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


def _clamp(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, score))


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
    # Same as task 1 — label drift alone doesn't change scoring
    return grade_generic(state)


def grade_task_3(state: DOMState) -> float:
    # Structural drift — still same metric, but completion is harder to earn
    return grade_generic(state)


def grade_task_4(state: DOMState) -> float:
    # Full chaos — if agent survived distractors with high completion, modest bonus
    base = grade_generic(state)
    # Tiny bonus for surviving with zero failed actions despite chaos
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
```

### 3.3 Tests

Create `tests/test_task_graph.py`:

```python
"""Phase 3 tests — task graph logic."""
from infinite_dom.task_graph import make_booking_flow_graph


class TestBookingFlowGraph:
    def _make(self):
        return make_booking_flow_graph(
            task_id=1,
            template_id="booking_flow",
            origin="Bengaluru",
            destination="Mumbai",
            seat_class="2AC",
            instruction="Book a 2AC ticket from Bengaluru to Mumbai",
        )

    def test_graph_has_five_nodes(self):
        g = self._make()
        assert len(g.nodes) == 5
        assert set(g.all_node_ids) == {
            "origin_set", "destination_set", "class_selected",
            "search_submitted", "booking_confirmed",
        }

    def test_origin_predicate_hits_from_field(self):
        g = self._make()
        state = {"inputs": {"From Station": "Bengaluru (SBC)"}}
        assert "origin_set" in g.get_completed_nodes(state)

    def test_origin_predicate_misses_unrelated_field(self):
        g = self._make()
        state = {"inputs": {"Newsletter Email": "Bengaluru"}}
        # "Newsletter Email" doesn't contain origin/from/depart/start
        assert "origin_set" not in g.get_completed_nodes(state)

    def test_class_predicate_via_selected_options(self):
        g = self._make()
        state = {"selected_options": {"Fare Class": "2AC - Second AC"}}
        assert "class_selected" in g.get_completed_nodes(state)

    def test_search_via_url(self):
        g = self._make()
        state = {"url": "http://localhost:9000/results?trains=..."}
        assert "search_submitted" in g.get_completed_nodes(state)

    def test_search_via_visible_text(self):
        g = self._make()
        state = {"url": "...", "visible_text": {"Available Trains", "Time"}}
        assert "search_submitted" in g.get_completed_nodes(state)

    def test_confirmation_via_booking_id(self):
        g = self._make()
        state = {"booking_id_visible": "PNR 1234567890"}
        assert "booking_confirmed" in g.get_completed_nodes(state)

    def test_fully_complete_when_all_satisfied(self):
        g = self._make()
        state = {
            "inputs": {"From": "Bengaluru", "To": "Mumbai"},
            "selected_options": {"Class": "2AC"},
            "url": "/results",
            "confirmation_visible": True,
        }
        assert g.is_fully_complete(state)

    def test_not_fully_complete_with_partial(self):
        g = self._make()
        state = {
            "inputs": {"From": "Bengaluru", "To": "Mumbai"},
            "selected_options": {"Class": "2AC"},
        }
        assert not g.is_fully_complete(state)

    def test_predicate_error_safe(self):
        # Construct a browser state missing expected keys — should not crash
        g = self._make()
        state = {}
        completed = g.get_completed_nodes(state)
        assert completed == []
```

Create `tests/test_reward.py`:

```python
"""Phase 3 tests — reward function."""
from infinite_dom.reward_calculator import (
    COMPLETION_BONUS,
    INVALID_ACTION_PENALTY,
    STEP_PENALTY,
    THRASH_PENALTY,
    compute_reward,
)


NODE_WEIGHTS = {
    "origin_set": 0.15,
    "destination_set": 0.15,
    "class_selected": 0.15,
    "search_submitted": 0.2,
    "booking_confirmed": 0.35,
}


class TestCompute:
    def test_no_progress_successful_action(self):
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.progression == 0.0
        assert r.step == STEP_PENALTY
        assert r.invalid == 0.0
        assert r.total == STEP_PENALTY

    def test_one_node_progress(self):
        r = compute_reward(
            newly_completed_nodes=["origin_set"],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.progression == 0.15
        assert r.total == 0.15 + STEP_PENALTY

    def test_invalid_action_adds_penalty(self):
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.invalid == INVALID_ACTION_PENALTY

    def test_episode_completion_bonus(self):
        r = compute_reward(
            newly_completed_nodes=["booking_confirmed"],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=True,
            action_history=[],
        )
        assert r.completion == COMPLETION_BONUS
        assert r.total == 0.35 + STEP_PENALTY + COMPLETION_BONUS

    def test_thrash_detection(self):
        # Three identical failed actions in a row
        failed = {"action_type": "click", "element_ref": "btn_x", "succeeded": False}
        history = [failed, failed, failed]
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=history,
        )
        assert r.thrash == THRASH_PENALTY

    def test_no_thrash_when_actions_differ(self):
        a1 = {"action_type": "click", "element_ref": "btn_1", "succeeded": False}
        a2 = {"action_type": "click", "element_ref": "btn_2", "succeeded": False}
        a3 = {"action_type": "click", "element_ref": "btn_3", "succeeded": False}
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=[a1, a2, a3],
        )
        assert r.thrash == 0.0


class TestAntiGaming:
    """Pathological policies should score low."""

    def test_random_policy_mostly_negative(self):
        total = 0.0
        # Simulate 20 random actions, mostly failing, no progress
        history = []
        for _ in range(20):
            history.append({"action_type": "click", "element_ref": f"r{_}", "succeeded": False})
            r = compute_reward(
                newly_completed_nodes=[],
                node_weights=NODE_WEIGHTS,
                action_succeeded=False,
                is_episode_complete=False,
                action_history=history,
            )
            total += r.total
        assert total < 0, f"Random failing policy should accumulate negative reward, got {total}"
```

Create `tests/test_graders.py`:

```python
"""Phase 3 tests — graders."""
import pytest

from infinite_dom.graders import SCORE_MAX, SCORE_MIN, grade
from infinite_dom.models import DOMState


def _state(**overrides):
    base = dict(
        episode_id="ep1",
        step_count=10,
        task_id=1,
        template_id="booking_flow",
        seed=42,
        task_graph_completed=["origin_set", "destination_set"],
        task_graph_total=[
            "origin_set", "destination_set", "class_selected",
            "search_submitted", "booking_confirmed",
        ],
    )
    base.update(overrides)
    return DOMState(**base)


class TestGraderBounds:
    @pytest.mark.parametrize("task_id", [1, 2, 3, 4])
    def test_grade_in_open_unit_interval(self, task_id):
        # Perfect episode
        s = _state(
            task_id=task_id,
            task_graph_completed=[
                "origin_set", "destination_set", "class_selected",
                "search_submitted", "booking_confirmed",
            ],
            step_count=5,
            failed_action_count=0,
        )
        score = grade(task_id, s)
        assert SCORE_MIN < score < SCORE_MAX, f"Task {task_id} perfect score out of bounds: {score}"

    @pytest.mark.parametrize("task_id", [1, 2, 3, 4])
    def test_grade_never_reaches_zero(self, task_id):
        s = _state(
            task_id=task_id,
            task_graph_completed=[],
            step_count=25,
            failed_action_count=10,
        )
        score = grade(task_id, s)
        assert score >= SCORE_MIN

    @pytest.mark.parametrize("task_id", [1, 2, 3, 4])
    def test_grade_never_reaches_one(self, task_id):
        s = _state(
            task_id=task_id,
            task_graph_completed=[
                "origin_set", "destination_set", "class_selected",
                "search_submitted", "booking_confirmed",
            ],
            step_count=1,
            failed_action_count=0,
        )
        score = grade(task_id, s)
        assert score <= SCORE_MAX


class TestGraderMonotonicity:
    def test_more_completion_scores_higher(self):
        low = _state(task_graph_completed=["origin_set"])
        high = _state(task_graph_completed=[
            "origin_set", "destination_set", "class_selected",
        ])
        assert grade(1, high) > grade(1, low)

    def test_failures_reduce_score(self):
        clean = _state(failed_action_count=0)
        messy = _state(failed_action_count=5)
        assert grade(1, clean) > grade(1, messy)


def test_unknown_task_id_raises():
    with pytest.raises(ValueError):
        grade(99, _state())
```

### 3.4 Milestone Gate 3

Run: `pytest tests/test_task_graph.py tests/test_reward.py tests/test_graders.py -v`

**GATE CRITERIA:** All tests pass. No warnings about Pydantic validation. Log results in BUILD_LOG.md.

---

