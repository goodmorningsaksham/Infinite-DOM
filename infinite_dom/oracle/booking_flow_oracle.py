"""
Oracle for the booking flow — hand-written solver used to:
  1. Verify the environment is solvable end-to-end
  2. Generate SFT trajectories for the imitation-learning warmup phase
"""
from __future__ import annotations

from infinite_dom.models import ActionType, DOMAction
from infinite_dom.task_graph import TaskGraph


def _find_ref(a11y_tree: str, role_hint: str, name_substrings: tuple[str, ...]) -> str | None:
    """Scan the a11y tree text for a line matching role and any of the substrings."""
    wants_lower = tuple(s.lower() for s in name_substrings)
    for line in a11y_tree.split("\n"):
        if f"role={role_hint}" not in line:
            continue
        lower = line.lower()
        if any(s in lower for s in wants_lower):
            idx = line.find("ref=")
            if idx == -1:
                continue
            tail = line[idx + 4:]
            ref = tail.split(" ")[0].rstrip("]")
            return ref
    return None


def _combobox_has_value(a11y_tree: str, role_hint: str, name_substrings: tuple[str, ...], target: str) -> bool:
    """Check if a combobox's value= attribute contains the target string."""
    wants_lower = tuple(s.lower() for s in name_substrings)
    target_lower = target.lower()
    for line in a11y_tree.split("\n"):
        if f"role={role_hint}" not in line:
            continue
        lower = line.lower()
        if any(s in lower for s in wants_lower):
            val_idx = lower.find('value="')
            if val_idx != -1:
                val_str = lower[val_idx + 7:]
                val_end = val_str.find('"')
                if val_end != -1:
                    val_content = val_str[:val_end]
                    if target_lower in val_content and "select" not in val_content:
                        return True
            return False
    return False


def _textbox_has_value(a11y_tree: str, ref_id: str, target: str) -> bool:
    """Check if a textbox already has the target value filled in."""
    target_lower = target.lower()
    for line in a11y_tree.split("\n"):
        if f"ref={ref_id} " not in line:
            continue
        lower = line.lower()
        val_idx = lower.find('value="')
        if val_idx != -1:
            val_str = lower[val_idx + 7:]
            val_end = val_str.find('"')
            if val_end != -1 and target_lower in val_str[:val_end]:
                return True
        return False
    return False


def oracle_policy(obs_a11y_tree: str, task_graph: TaskGraph) -> DOMAction:
    """
    Decide the next oracle action given the current a11y tree and task graph.
    Strategy: advance the next incomplete node, in the intended order.
    """
    origin = task_graph.params["origin"]
    destination = task_graph.params["destination"]
    seat_class = task_graph.params["class"]

    for banner_names in (("accept", "cookie"), ("close", "promo"), ("dismiss",)):
        ref = _find_ref(obs_a11y_tree, "button", banner_names)
        if ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=ref)

    origin_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("from", "origin", "depart", "start"),
    )
    if origin_ref and not _textbox_has_value(obs_a11y_tree, origin_ref, origin):
        return DOMAction(action_type=ActionType.TYPE, element_ref=origin_ref, text_value=origin)

    dest_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("to", "destination", "arrive", "going"),
    )
    if dest_ref and not _textbox_has_value(obs_a11y_tree, dest_ref, destination):
        return DOMAction(action_type=ActionType.TYPE, element_ref=dest_ref, text_value=destination)

    class_ref = _find_ref(
        obs_a11y_tree, "combobox",
        ("class", "cabin", "fare", "seat"),
    )
    if class_ref and not _combobox_has_value(obs_a11y_tree, "combobox", ("class", "cabin", "fare", "seat"), seat_class):
        return DOMAction(
            action_type=ActionType.TYPE,
            element_ref=class_ref,
            text_value=seat_class,
        )

    search_ref = _find_ref(
        obs_a11y_tree, "button",
        ("search", "find", "go", "check", "look"),
    )
    if search_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=search_ref)

    book_ref = _find_ref(
        obs_a11y_tree, "button",
        ("book", "reserve", "purchase", "buy", "secure"),
    )
    if book_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=book_ref)

    confirm_ref = _find_ref(
        obs_a11y_tree, "button",
        ("confirm", "complete", "finalize", "place"),
    )
    if confirm_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=confirm_ref)

    return DOMAction(action_type=ActionType.WAIT)
