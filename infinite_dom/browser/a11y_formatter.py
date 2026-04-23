"""
Converts Playwright accessibility tree snapshots to the text format
that the agent sees.

Format rules:
  - Each element gets a short stable ref (role abbrev + counter): btn_1, inp_2, lnk_3
  - Hidden/invisible elements are pruned
  - Depth is indicated by 2-space indentation
  - Element properties appear in bracketed format:
      [ref=btn_1 role=button name="Search" text="Search"]
  - Token budget enforced; overflow truncated with marker
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from infinite_dom.config import CONFIG

_ROLE_TO_PREFIX = {
    "button": "btn",
    "textbox": "inp",
    "searchbox": "inp",
    "combobox": "cmb",
    "listbox": "lst",
    "option": "opt",
    "link": "lnk",
    "checkbox": "chk",
    "radio": "rad",
    "menuitem": "mnu",
    "tab": "tab",
    "dialog": "dlg",
    "heading": "hdg",
    "img": "img",
    "banner": "bnr",
    "main": "mn",
    "navigation": "nav",
    "form": "frm",
    "list": "lst",
    "listitem": "li",
    "article": "art",
    "region": "rgn",
    "group": "grp",
    "text": "txt",
    "StaticText": "txt",
}


@dataclass
class A11yRef:
    ref_id: str
    role: str
    name: str = ""
    text: str = ""
    value: str = ""
    selected_option: str = ""
    playwright_node: Any = None


@dataclass
class A11yFormatResult:
    text: str
    ref_map: dict[str, A11yRef] = field(default_factory=dict)
    truncated: bool = False


def _approx_tokens(s: str) -> int:
    return len(s) // 4


def _pick_prefix(role: str) -> str:
    return _ROLE_TO_PREFIX.get(role, "el")


def format_a11y_tree(
    snapshot: dict[str, Any],
    max_tokens: int = CONFIG.a11y_tree_max_tokens,
) -> A11yFormatResult:
    """
    Walk the Playwright accessibility snapshot and produce the agent's observation.

    `snapshot` is the dict returned by `page.accessibility.snapshot()`.
    Its schema (simplified): { role, name, value, children: [...], ... }.
    """
    result = A11yFormatResult(text="", ref_map={})
    if snapshot is None:
        result.text = "[empty a11y tree]"
        return result

    lines: list[str] = []
    counters: dict[str, itertools.count] = {}

    def next_ref(role: str) -> str:
        prefix = _pick_prefix(role)
        if prefix not in counters:
            counters[prefix] = itertools.count(1)
        return f"{prefix}_{next(counters[prefix])}"

    def walk(node: dict[str, Any], depth: int, budget_left: int) -> int:
        if not isinstance(node, dict):
            return budget_left

        role = node.get("role", "") or ""
        name = (node.get("name", "") or "").strip()
        value = (node.get("value", "") or "").strip()

        if not role:
            for child in node.get("children", []) or []:
                if budget_left <= 0:
                    break
                budget_left = walk(child, depth, budget_left)
            return budget_left

        skip_empty_structural = role in {"generic", "none", "presentation"} and not name and not value

        if not skip_empty_structural:
            ref = next_ref(role)
            parts = [f"ref={ref}", f"role={role}"]
            if name:
                parts.append(f'name="{name[:60]}"')
            if value:
                parts.append(f'value="{value[:60]}"')

            selected = ""
            if role in {"combobox", "listbox"}:
                for child in node.get("children", []) or []:
                    if child.get("selected") or child.get("checked"):
                        selected = (child.get("name", "") or "").strip()
                        if selected:
                            parts.append(f'selected="{selected[:60]}"')
                            break

            line = "  " * depth + "[" + " ".join(parts) + "]"
            lines.append(line)

            result.ref_map[ref] = A11yRef(
                ref_id=ref,
                role=role,
                name=name,
                text=name,
                value=value,
                selected_option=selected,
                playwright_node=node,
            )

            budget_left -= _approx_tokens(line)
            if budget_left <= 0:
                result.truncated = True
                return 0

        for child in node.get("children", []) or []:
            if budget_left <= 0:
                break
            budget_left = walk(child, depth + 1, budget_left)

        return budget_left

    walk(snapshot, 0, max_tokens)
    if result.truncated:
        lines.append(f"[... truncated — budget {max_tokens} tokens exhausted]")

    result.text = "\n".join(lines)
    return result
