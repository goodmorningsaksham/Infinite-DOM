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
        g = self._make()
        state = {}
        completed = g.get_completed_nodes(state)
        assert completed == []
