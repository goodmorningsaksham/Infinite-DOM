"""Phase 2 tests — data model correctness."""
import pytest
from pydantic import ValidationError

from infinite_dom.models import ActionType, DOMAction, DOMObservation, DOMState


class TestDOMAction:
    def test_valid_click(self):
        a = DOMAction(action_type=ActionType.CLICK, element_ref="btn_3")
        assert a.action_type == ActionType.CLICK
        assert a.element_ref == "btn_3"
        assert a.text_value == ""
        assert a.scroll_delta == 0

    def test_valid_type(self):
        a = DOMAction(
            action_type=ActionType.TYPE,
            element_ref="inp_1",
            text_value="Bengaluru",
        )
        assert a.text_value == "Bengaluru"

    def test_valid_scroll(self):
        a = DOMAction(action_type=ActionType.SCROLL, scroll_delta=500)
        assert a.scroll_delta == 500

    def test_scroll_delta_bounds(self):
        with pytest.raises(ValidationError):
            DOMAction(action_type=ActionType.SCROLL, scroll_delta=3000)

    def test_action_type_rejects_invalid_string(self):
        with pytest.raises(ValidationError):
            DOMAction(action_type="not_a_real_action")  # type: ignore

    def test_serializes_to_json(self):
        a = DOMAction(action_type=ActionType.CLICK, element_ref="btn_3")
        j = a.model_dump_json()
        assert "click" in j
        assert "btn_3" in j


class TestDOMObservation:
    def test_minimum_fields(self):
        o = DOMObservation(
            a11y_tree="[role=main]",
            task_instruction="Do a thing",
            done=False,
            reward=0.0,
        )
        assert o.a11y_tree == "[role=main]"
        assert o.task_instruction == "Do a thing"
        assert o.done is False
        assert o.reward == 0.0
        assert o.task_progress == []

    def test_done_must_be_bool(self):
        with pytest.raises(ValidationError):
            DOMObservation(
                a11y_tree="x",
                task_instruction="x",
                done="yes",  # type: ignore
                reward=0.0,
            )

    def test_step_count_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            DOMObservation(
                a11y_tree="x",
                task_instruction="x",
                done=False,
                reward=0.0,
                step_count=-1,
            )


class TestDOMState:
    def test_minimum_fields(self):
        s = DOMState(
            episode_id="ep_123",
            step_count=0,
            task_id=1,
            template_id="booking_flow",
            seed=42,
        )
        assert s.task_id == 1
        assert s.task_graph_completed == []
        assert s.failed_action_count == 0
        assert s.terminated_reason == ""
