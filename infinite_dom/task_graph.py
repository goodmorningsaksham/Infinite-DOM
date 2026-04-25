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
- error_messages: list[str]    # validation error messages currently shown
- cart_items: list[str]        # items in the cart (e-commerce)
- order_id_visible: str | None # order ID (e-commerce)
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


# ---------------------------------------------------------------------------
# Booking flow graph (original simple version for tasks 1-4)
# ---------------------------------------------------------------------------

def make_booking_flow_graph(
    task_id: int,
    template_id: str,
    origin: str,
    destination: str,
    seat_class: str,
    instruction: str,
    **extra_params: str,
) -> TaskGraph:
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
            state, ("origin", "from", "depart", "start"), target_origin,
        )

    def destination_predicate(state: BrowserState) -> bool:
        return _any_input_contains(
            state, ("destination", "to", "arrive", "end", "going"), target_destination,
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

    # --- Build node list, conditionally adding nodes based on extra_params ---
    nodes = [
        TaskNode("origin_set", "Enter origin", origin_predicate, weight=0.12),
        TaskNode("destination_set", "Enter destination", destination_predicate, weight=0.12),
        TaskNode("class_selected", "Select class", class_predicate, weight=0.12),
    ]

    edges = [
        ("origin_set", "destination_set"),
        ("destination_set", "class_selected"),
    ]

    # Conditional: trip type (round-trip adds return date)
    trip_type = extra_params.get("trip_type", "")
    if trip_type == "round_trip":
        target_return = _norm(extra_params.get("return_date", ""))

        def return_date_predicate(state: BrowserState) -> bool:
            inputs: dict[str, str] = state.get("inputs", {})
            for name, val in inputs.items():
                if any(sem in _norm(name) for sem in ("return", "back", "coming")):
                    if target_return and target_return in _norm(val):
                        return True
            return False

        nodes.append(TaskNode("return_date_set", "Enter return date", return_date_predicate, weight=0.08))
        edges.append(("class_selected", "return_date_set"))
        edges.append(("return_date_set", "search_submitted"))
    else:
        edges.append(("class_selected", "search_submitted"))

    # Conditional: must pick the correct train from results (reasoning-based)
    target_train = extra_params.get("target_train", "")
    if target_train:
        target_train_norm = _norm(target_train)

        def train_selected_predicate(state: BrowserState) -> bool:
            vis: set[str] = state.get("visible_text", set())
            vis_lower = " ".join(_norm(v) for v in vis)
            url = _norm(state.get("url", ""))
            if "confirm" in url and target_train_norm in vis_lower:
                return True
            selected = state.get("selected_train", "")
            if target_train_norm in _norm(selected):
                return True
            return False

        nodes.append(TaskNode("search_submitted", "Submit search", search_predicate, weight=0.10))
        nodes.append(TaskNode("train_selected", "Select correct train", train_selected_predicate, weight=0.16))
        edges.append(("search_submitted", "train_selected"))
        edges.append(("train_selected", "booking_confirmed"))
    else:
        nodes.append(TaskNode("search_submitted", "Submit search", search_predicate, weight=0.16))
        edges.append(("search_submitted", "booking_confirmed"))

    nodes.append(TaskNode("booking_confirmed", "Confirm booking", confirmation_predicate, weight=0.30))

    # Normalize weights to sum to ~1.0
    total_w = sum(n.weight for n in nodes)
    if abs(total_w - 1.0) > 0.01:
        for n in nodes:
            n.weight = round(n.weight / total_w, 3)

    params = {"origin": origin, "destination": destination, "class": seat_class}
    params.update(extra_params)

    return TaskGraph(
        task_id=task_id,
        template_id=template_id,
        instruction=instruction,
        nodes=nodes,
        edges=edges,
        params=params,
    )


# ---------------------------------------------------------------------------
# E-commerce flow graph (tasks 5-8)
# ---------------------------------------------------------------------------

def make_ecommerce_flow_graph(
    task_id: int,
    template_id: str,
    target_product: str,
    target_category: str,
    instruction: str,
    **extra_params: str,
) -> TaskGraph:
    target_product_norm = _norm(target_product)
    target_category_norm = _norm(target_category)

    def product_searched_predicate(state: BrowserState) -> bool:
        inputs: dict[str, str] = state.get("inputs", {})
        for name, val in inputs.items():
            if any(sem in _norm(name) for sem in ("search", "query", "find", "look")):
                if target_product_norm in _norm(val) or target_category_norm in _norm(val):
                    return True
        url = _norm(state.get("url", ""))
        if "search" in url or "catalog" in url or "products" in url:
            return True
        return False

    def category_filtered_predicate(state: BrowserState) -> bool:
        sel: dict[str, str] = state.get("selected_options", {})
        for name, val in sel.items():
            if any(sem in _norm(name) for sem in ("category", "filter", "type", "department")):
                if target_category_norm in _norm(val):
                    return True
        vis: set[str] = state.get("visible_text", set())
        vis_lower = " ".join(_norm(v) for v in vis)
        if f"showing {target_category_norm}" in vis_lower or f"filtered by {target_category_norm}" in vis_lower:
            return True
        return False

    def product_selected_predicate(state: BrowserState) -> bool:
        vis: set[str] = state.get("visible_text", set())
        vis_lower = " ".join(_norm(v) for v in vis)
        url = _norm(state.get("url", ""))
        if "product" in url or "detail" in url or "item" in url:
            if target_product_norm in vis_lower:
                return True
        return False

    def added_to_cart_predicate(state: BrowserState) -> bool:
        cart = state.get("cart_items", [])
        if any(target_product_norm in _norm(item) for item in cart):
            return True
        vis: set[str] = state.get("visible_text", set())
        vis_lower = " ".join(_norm(v) for v in vis)
        return "added to cart" in vis_lower or "item in cart" in vis_lower

    def checkout_started_predicate(state: BrowserState) -> bool:
        url = _norm(state.get("url", ""))
        if "checkout" in url or "shipping" in url:
            return True
        vis: set[str] = state.get("visible_text", set())
        vis_lower = " ".join(_norm(v) for v in vis)
        return "shipping" in vis_lower or "checkout" in vis_lower

    def shipping_filled_predicate(state: BrowserState) -> bool:
        inputs: dict[str, str] = state.get("inputs", {})
        filled_count = 0
        for name, val in inputs.items():
            if any(sem in _norm(name) for sem in ("address", "street", "city", "zip", "pin", "name", "phone")):
                if len(val.strip()) > 1:
                    filled_count += 1
        return filled_count >= 3

    def order_confirmed_predicate(state: BrowserState) -> bool:
        oid = state.get("order_id_visible")
        if oid:
            return True
        vis: set[str] = state.get("visible_text", set())
        vis_lower = " ".join(_norm(v) for v in vis)
        return "order confirmed" in vis_lower or "order placed" in vis_lower

    nodes = [
        TaskNode("product_searched", "Search for product", product_searched_predicate, weight=0.08),
        TaskNode("category_filtered", "Filter by category", category_filtered_predicate, weight=0.10),
        TaskNode("product_selected", "Select correct product", product_selected_predicate, weight=0.15),
        TaskNode("added_to_cart", "Add to cart", added_to_cart_predicate, weight=0.15),
        TaskNode("checkout_started", "Proceed to checkout", checkout_started_predicate, weight=0.10),
        TaskNode("shipping_filled", "Fill shipping details", shipping_filled_predicate, weight=0.17),
        TaskNode("order_confirmed", "Confirm order", order_confirmed_predicate, weight=0.25),
    ]

    edges = [
        ("product_searched", "category_filtered"),
        ("category_filtered", "product_selected"),
        ("product_selected", "added_to_cart"),
        ("added_to_cart", "checkout_started"),
        ("checkout_started", "shipping_filled"),
        ("shipping_filled", "order_confirmed"),
    ]

    params = {"target_product": target_product, "target_category": target_category}
    params.update(extra_params)

    return TaskGraph(
        task_id=task_id,
        template_id=template_id,
        instruction=instruction,
        nodes=nodes,
        edges=edges,
        params=params,
    )
