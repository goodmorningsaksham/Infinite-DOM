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
