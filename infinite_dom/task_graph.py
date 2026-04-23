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
    weight: float = 0.1


@dataclass
class TaskGraph:
    task_id: int
    template_id: str
    instruction: str
    nodes: list[TaskNode]
    edges: list[tuple[str, str]] = field(default_factory=list)
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
                continue
        return completed

    def is_fully_complete(self, browser_state: BrowserState) -> bool:
        return set(self.get_completed_nodes(browser_state)) == set(self.all_node_ids)


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

    def _any_input_contains(
        state: BrowserState, semantic_names: tuple[str, ...], target: str
    ) -> bool:
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
