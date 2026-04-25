"""
Oracle for the booking flow — hand-written solver used to:
  1. Verify the environment is solvable end-to-end
  2. Generate SFT trajectories for the imitation-learning warmup phase

Handles: cookie banners, promo modals, newsletter popups, form validation,
conditional round-trip fields, and train selection from results.
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


def _find_ref_exact(a11y_tree: str, role_hint: str, name_exact: str) -> str | None:
    """Find a ref where the name contains the exact target (case-insensitive)."""
    target = name_exact.lower()
    for line in a11y_tree.split("\n"):
        if f"role={role_hint}" not in line:
            continue
        lower = line.lower()
        if target in lower:
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


def _page_has_text(a11y_tree: str, text: str) -> bool:
    """Check if any line in the a11y tree contains the given text."""
    target = text.lower()
    for line in a11y_tree.split("\n"):
        if target in line.lower():
            return True
    return False


def _dismiss_distractor(a11y_tree: str) -> DOMAction | None:
    """Try to dismiss cookie banners, promo modals, newsletter popups."""
    for names in (
        ("accept cookies", "accept"),
        ("close promo",),
        ("close newsletter", "no thanks"),
        ("dismiss",),
    ):
        ref = _find_ref(a11y_tree, "button", names)
        if ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=ref)
    return None


def oracle_policy(obs_a11y_tree: str, task_graph: TaskGraph) -> DOMAction:
    """
    Decide the next oracle action given the current a11y tree and task graph.
    Routes to the appropriate sub-policy based on template_id.
    """
    if task_graph.template_id == "ecommerce_flow":
        return _ecommerce_oracle(obs_a11y_tree, task_graph)
    return _booking_oracle(obs_a11y_tree, task_graph)


def _booking_oracle(obs_a11y_tree: str, task_graph: TaskGraph) -> DOMAction:
    """Oracle for booking flow tasks 1-4."""
    origin = task_graph.params["origin"]
    destination = task_graph.params["destination"]
    seat_class = task_graph.params["class"]
    trip_type = task_graph.params.get("trip_type", "")
    return_date = task_graph.params.get("return_date", "")
    target_train = task_graph.params.get("target_train", "")

    # Dismiss distractors first
    dismiss = _dismiss_distractor(obs_a11y_tree)
    if dismiss:
        return dismiss

    # On the confirmation/done page, click confirm if available
    if _page_has_text(obs_a11y_tree, "confirm your booking"):
        confirm_ref = _find_ref(
            obs_a11y_tree, "button",
            ("confirm", "complete", "finalize", "place", "pay"),
        )
        if confirm_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=confirm_ref)

    # On the results page, select the target train
    if _page_has_text(obs_a11y_tree, "available trains") and target_train:
        train_ref = _find_ref_exact(obs_a11y_tree, "button", target_train)
        if train_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=train_ref)
        book_ref = _find_ref(
            obs_a11y_tree, "button",
            ("book", "reserve", "purchase", "buy", "secure"),
        )
        if book_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=book_ref)

    # Fill the search form
    origin_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("from", "origin", "depart", "start", "leaving"),
    )
    if origin_ref and not _textbox_has_value(obs_a11y_tree, origin_ref, origin):
        return DOMAction(action_type=ActionType.TYPE, element_ref=origin_ref, text_value=origin)

    dest_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("to", "destination", "arrive", "going", "end"),
    )
    if dest_ref and not _textbox_has_value(obs_a11y_tree, dest_ref, destination):
        return DOMAction(action_type=ActionType.TYPE, element_ref=dest_ref, text_value=destination)

    class_ref = _find_ref(
        obs_a11y_tree, "combobox",
        ("class", "cabin", "fare", "seat", "coach"),
    )
    if class_ref and not _combobox_has_value(obs_a11y_tree, "combobox", ("class", "cabin", "fare", "seat", "coach"), seat_class):
        return DOMAction(
            action_type=ActionType.TYPE,
            element_ref=class_ref,
            text_value=seat_class,
        )

    # Handle round-trip: set trip type and return date
    if trip_type == "round_trip":
        trip_type_ref = _find_ref(
            obs_a11y_tree, "combobox",
            ("trip", "journey", "travel mode"),
        )
        if trip_type_ref and not _combobox_has_value(obs_a11y_tree, "combobox", ("trip", "journey", "travel"), "round"):
            return DOMAction(
                action_type=ActionType.TYPE,
                element_ref=trip_type_ref,
                text_value="round_trip",
            )

        if return_date:
            return_ref = _find_ref(
                obs_a11y_tree, "textbox",
                ("return", "coming back", "back on"),
            )
            if return_ref and not _textbox_has_value(obs_a11y_tree, return_ref, return_date):
                return DOMAction(action_type=ActionType.TYPE, element_ref=return_ref, text_value=return_date)

    # Click search button
    search_ref = _find_ref(
        obs_a11y_tree, "button",
        ("search", "find", "go", "check", "look"),
    )
    if search_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=search_ref)

    # Fallback: click any book/confirm button
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


def _ecommerce_oracle(obs_a11y_tree: str, task_graph: TaskGraph) -> DOMAction:
    """Oracle for e-commerce flow tasks 5-8."""
    target_product = task_graph.params["target_product"]
    target_category = task_graph.params["target_category"]
    shipping_name = task_graph.params.get("shipping_name", "")
    shipping_address = task_graph.params.get("shipping_address", "")
    shipping_city = task_graph.params.get("shipping_city", "")
    shipping_pin = task_graph.params.get("shipping_pin", "")
    shipping_phone = task_graph.params.get("shipping_phone", "")

    # Dismiss distractors first
    dismiss = _dismiss_distractor(obs_a11y_tree)
    if dismiss:
        return dismiss

    # On order confirmed page — done
    if _page_has_text(obs_a11y_tree, "order confirmed"):
        return DOMAction(action_type=ActionType.WAIT)

    # On checkout/shipping page — fill shipping details
    if _page_has_text(obs_a11y_tree, "shipping"):
        name_ref = _find_ref(obs_a11y_tree, "textbox", ("name", "full name", "recipient"))
        if name_ref and not _textbox_has_value(obs_a11y_tree, name_ref, shipping_name):
            return DOMAction(action_type=ActionType.TYPE, element_ref=name_ref, text_value=shipping_name)

        addr_ref = _find_ref(obs_a11y_tree, "textbox", ("address", "street", "delivery"))
        if addr_ref and not _textbox_has_value(obs_a11y_tree, addr_ref, shipping_address):
            return DOMAction(action_type=ActionType.TYPE, element_ref=addr_ref, text_value=shipping_address)

        city_ref = _find_ref(obs_a11y_tree, "textbox", ("city", "town"))
        if city_ref and not _textbox_has_value(obs_a11y_tree, city_ref, shipping_city):
            return DOMAction(action_type=ActionType.TYPE, element_ref=city_ref, text_value=shipping_city)

        pin_ref = _find_ref(obs_a11y_tree, "textbox", ("pin", "zip", "postal"))
        if pin_ref and not _textbox_has_value(obs_a11y_tree, pin_ref, shipping_pin):
            return DOMAction(action_type=ActionType.TYPE, element_ref=pin_ref, text_value=shipping_pin)

        phone_ref = _find_ref(obs_a11y_tree, "textbox", ("phone", "mobile", "contact"))
        if phone_ref and not _textbox_has_value(obs_a11y_tree, phone_ref, shipping_phone):
            return DOMAction(action_type=ActionType.TYPE, element_ref=phone_ref, text_value=shipping_phone)

        # All filled — click place order
        order_ref = _find_ref(obs_a11y_tree, "button", ("place order", "confirm order", "complete purchase", "submit order", "buy now"))
        if order_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=order_ref)

    # On cart page — proceed to checkout
    if _page_has_text(obs_a11y_tree, "your cart") or _page_has_text(obs_a11y_tree, "cart"):
        checkout_ref = _find_ref(obs_a11y_tree, "button", ("checkout", "proceed", "continue to payment", "buy now"))
        if checkout_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=checkout_ref)

    # On product detail page — add to cart
    if _page_has_text(obs_a11y_tree, target_product.lower()):
        # Check if we're on the product detail page (has "add to cart" button)
        add_ref = _find_ref(obs_a11y_tree, "button", ("add to cart", "add to bag", "add item", "put in cart"))
        if add_ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=add_ref)

    # On search/catalog page — filter by category, then select product
    category_ref = _find_ref(obs_a11y_tree, "combobox", ("category", "department", "filter", "type", "browse"))
    if category_ref and not _combobox_has_value(obs_a11y_tree, "combobox", ("category", "department", "filter", "type", "browse"), target_category):
        return DOMAction(action_type=ActionType.TYPE, element_ref=category_ref, text_value=target_category)

    # Type search query
    search_ref = _find_ref(obs_a11y_tree, "textbox", ("search", "find", "looking", "query"))
    if search_ref and not _textbox_has_value(obs_a11y_tree, search_ref, target_product):
        return DOMAction(action_type=ActionType.TYPE, element_ref=search_ref, text_value=target_product)

    # Click filter/apply button
    filter_ref = _find_ref(obs_a11y_tree, "button", ("filter", "apply"))
    if filter_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=filter_ref)

    # Click View on the target product
    view_ref = _find_ref_exact(obs_a11y_tree, "button", target_product)
    if view_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=view_ref)

    # Fallback: click any view button
    view_ref = _find_ref(obs_a11y_tree, "button", ("view",))
    if view_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=view_ref)

    return DOMAction(action_type=ActionType.WAIT)
