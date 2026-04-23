## PHASE 5 — Browser Driver (Playwright)

### 5.1 Goal

A `PlaywrightDriver` that:
1. Starts a persistent Chromium context
2. Registers generated pages with the PageServer and navigates to them
3. Executes actions (click, type, scroll, wait, back) by `element_ref`
4. Produces a pruned accessibility tree snapshot with stable refs
5. Extracts a `BrowserState` dict suitable for task graph predicates

### 5.2 Key Constraints

- **Element refs must be stable across snapshots within the same page.** If `btn_3` refers to "Search" on snapshot 1, it must still refer to "Search" on snapshot 2 (assuming the element still exists). If it disappears and reappears later, the ref may change.
- **The driver must handle SPA navigation** — Alpine.js changes the page via `history.pushState` without full page reloads. The a11y tree must refresh properly.
- **Must work on Windows via WSL2** — Playwright's Chromium runs inside WSL2's Linux kernel; this is standard.

### 5.3 Execution Checklist

#### 5.3.1 Create `infinite_dom/browser/a11y_formatter.py`

```python
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
    playwright_node: Any = None  # opaque handle for action dispatch


@dataclass
class A11yFormatResult:
    text: str
    ref_map: dict[str, A11yRef] = field(default_factory=dict)
    truncated: bool = False


def _approx_tokens(s: str) -> int:
    # Rough approximation — 1 token ~ 4 chars
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

        # Filter rules: prune noise
        if not role:
            # Just recurse through children without emitting a line
            for child in node.get("children", []) or []:
                if budget_left <= 0:
                    break
                budget_left = walk(child, depth, budget_left)
            return budget_left

        # Skip purely structural roles with no name/value and no useful content
        skip_empty_structural = role in {"generic", "none", "presentation"} and not name and not value

        if not skip_empty_structural:
            ref = next_ref(role)
            parts = [f"ref={ref}", f"role={role}"]
            if name:
                parts.append(f'name="{name[:60]}"')
            if value:
                parts.append(f'value="{value[:60]}"')

            # Selected option detection (for combobox/select)
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

    remaining = walk(snapshot, 0, max_tokens)
    if result.truncated:
        lines.append(f"[... truncated — budget {max_tokens} tokens exhausted]")

    result.text = "\n".join(lines)
    return result
```

#### 5.3.2 Create `infinite_dom/browser/playwright_driver.py`

```python
"""
Playwright wrapper: drive a real Chromium instance, execute agent actions,
capture accessibility tree and browser state.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from infinite_dom.browser.a11y_formatter import A11yFormatResult, format_a11y_tree
from infinite_dom.config import CONFIG
from infinite_dom.generator.serve_html import get_page_server
from infinite_dom.models import ActionType, DOMAction

logger = logging.getLogger(__name__)


@dataclass
class BrowserSnapshot:
    a11y: A11yFormatResult
    url: str
    state_dict: dict[str, Any] = field(default_factory=dict)


class PlaywrightDriver:
    """
    One instance = one browser context = one episode. `close()` teardown
    releases resources. For now we run a single episode at a time
    (max_concurrent_envs=1 in the OpenEnv server).
    """

    def __init__(self):
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._current_gen_id: str | None = None
        self._last_snapshot: BrowserSnapshot | None = None

    # =====================================================================
    # Lifecycle
    # =====================================================================

    async def start(self) -> None:
        if self._pw is not None:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=CONFIG.playwright_headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None

    async def load_page(self, generation_id: str, html: str) -> None:
        """Register page with the local server and navigate."""
        assert self._browser is not None, "Driver not started"

        page_server = await get_page_server()
        url = page_server.register_page(generation_id, html)
        self._current_gen_id = generation_id

        # New context per episode → clean state
        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

        await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
        # Give Alpine.js one tick to initialize
        await self._page.wait_for_function(
            "() => window.Alpine !== undefined",
            timeout=5000,
        )
        # Additional small delay for initial reactive rendering
        await asyncio.sleep(0.15)

    # =====================================================================
    # Snapshot / Observation
    # =====================================================================

    async def snapshot(self) -> BrowserSnapshot:
        """Capture current state of the page."""
        assert self._page is not None, "No page loaded"

        try:
            raw = await self._page.accessibility.snapshot()
        except Exception as e:
            logger.warning("accessibility.snapshot failed: %s", e)
            raw = {"role": "WebArea", "name": "", "children": []}

        a11y = format_a11y_tree(raw)
        url = self._page.url

        state_dict = await self._extract_browser_state()
        snap = BrowserSnapshot(a11y=a11y, url=url, state_dict=state_dict)
        self._last_snapshot = snap
        return snap

    async def _extract_browser_state(self) -> dict[str, Any]:
        """Extract a task-graph-compatible dict from the live page."""
        assert self._page is not None

        # Collect inputs (map label->value)
        inputs = await self._page.evaluate(
            """() => {
                const result = {};
                document.querySelectorAll('input, textarea').forEach(el => {
                    // Try to find an associated label
                    let label = '';
                    if (el.labels && el.labels.length > 0) {
                        label = el.labels[0].textContent.trim();
                    } else if (el.getAttribute('aria-label')) {
                        label = el.getAttribute('aria-label');
                    } else if (el.getAttribute('name')) {
                        label = el.getAttribute('name');
                    } else if (el.getAttribute('placeholder')) {
                        label = el.getAttribute('placeholder');
                    }
                    if (label) {
                        result[label] = el.value || '';
                    }
                });
                return result;
            }"""
        )

        # Collect selected options (by label)
        selected = await self._page.evaluate(
            """() => {
                const result = {};
                document.querySelectorAll('select').forEach(el => {
                    let label = '';
                    if (el.labels && el.labels.length > 0) {
                        label = el.labels[0].textContent.trim();
                    } else if (el.getAttribute('aria-label')) {
                        label = el.getAttribute('aria-label');
                    } else if (el.getAttribute('name')) {
                        label = el.getAttribute('name');
                    }
                    if (label && el.value) {
                        result[label] = el.value;
                    }
                });
                return result;
            }"""
        )

        # Visible text (a set of tokens)
        visible_text = await self._page.evaluate(
            """() => {
                const text = document.body.innerText || '';
                // Split into tokens (rough), dedupe
                return Array.from(new Set(text.split(/[\\n\\r\\s]+/).filter(s => s.length > 0)));
            }"""
        )

        # Confirmation detection: look for text like "PNR" + digits or "Booking Confirmed"
        confirmation = await self._page.evaluate(
            """() => {
                const text = document.body.innerText || '';
                const lower = text.toLowerCase();
                const idMatch = text.match(/PNR\\s*\\d{6,}|BKG\\d{6,}|confirmation[:\\s]+[a-z0-9]{6,}/i);
                return {
                    visible: lower.includes('booking confirmed') || lower.includes('confirmation') || !!idMatch,
                    booking_id: idMatch ? idMatch[0] : null,
                };
            }"""
        )

        return {
            "inputs": inputs or {},
            "selected_options": selected or {},
            "visible_text": set(visible_text or []),
            "url": self._page.url,
            "confirmation_visible": confirmation.get("visible", False),
            "booking_id_visible": confirmation.get("booking_id"),
        }

    # =====================================================================
    # Action execution
    # =====================================================================

    async def execute(self, action: DOMAction) -> bool:
        """
        Execute the action. Returns True on success, False if the action
        could not be carried out (element not found, wrong type, etc.).
        """
        assert self._page is not None, "No page loaded"
        assert self._last_snapshot is not None, "Must snapshot before executing"

        try:
            if action.action_type == ActionType.WAIT:
                await asyncio.sleep(0.3)
                return True

            if action.action_type == ActionType.BACK:
                await self._page.go_back(wait_until="domcontentloaded", timeout=5000)
                await asyncio.sleep(0.1)
                return True

            if action.action_type == ActionType.SCROLL:
                delta = int(action.scroll_delta)
                await self._page.evaluate(f"window.scrollBy(0, {delta});")
                return True

            # For click / type, we need to resolve the element ref to a real element
            ref_map = self._last_snapshot.a11y.ref_map
            if action.element_ref not in ref_map:
                logger.info("Unknown element_ref: %s", action.element_ref)
                return False

            ref = ref_map[action.element_ref]
            locator = await self._resolve_locator(ref)
            if locator is None:
                return False

            if action.action_type == ActionType.CLICK:
                await locator.click(timeout=3000)
                await asyncio.sleep(0.15)
                return True

            if action.action_type == ActionType.TYPE:
                await locator.fill(action.text_value, timeout=3000)
                await asyncio.sleep(0.05)
                # For comboboxes / selects, also trigger change event
                return True

            return False
        except Exception as e:
            logger.info("Action execution failed: %s", e)
            return False

    async def _resolve_locator(self, ref):
        """
        Resolve an A11yRef to a Playwright Locator.
        Strategy: use role + name matching since the snapshot doesn't
        hand back real element handles.
        """
        assert self._page is not None

        role = ref.role
        name = ref.name

        try:
            if name:
                loc = self._page.get_by_role(role, name=name, exact=False).first
            else:
                loc = self._page.get_by_role(role).first
            # Check it's actually visible/present
            await loc.wait_for(state="visible", timeout=2000)
            return loc
        except Exception:
            # Fallback: label-based locator for form fields
            if ref.role in {"textbox", "searchbox"} and ref.name:
                try:
                    loc = self._page.get_by_label(ref.name).first
                    await loc.wait_for(state="visible", timeout=1500)
                    return loc
                except Exception:
                    pass
            return None
```

### 5.4 Tests

Create `tests/test_a11y_formatter.py`:

```python
"""Phase 5 tests — a11y tree formatter (no browser required)."""
from infinite_dom.browser.a11y_formatter import format_a11y_tree


def test_empty_snapshot():
    r = format_a11y_tree(None)
    assert "empty" in r.text.lower()


def test_single_button():
    snap = {
        "role": "WebArea",
        "name": "",
        "children": [
            {"role": "button", "name": "Search", "children": []},
        ],
    }
    r = format_a11y_tree(snap)
    assert "btn_1" in r.text
    assert "Search" in r.text
    assert "btn_1" in r.ref_map


def test_multiple_buttons_get_unique_refs():
    snap = {
        "role": "main",
        "name": "",
        "children": [
            {"role": "button", "name": "Click A"},
            {"role": "button", "name": "Click B"},
        ],
    }
    r = format_a11y_tree(snap)
    assert "btn_1" in r.ref_map
    assert "btn_2" in r.ref_map
    assert r.ref_map["btn_1"].name == "Click A"
    assert r.ref_map["btn_2"].name == "Click B"


def test_indentation_reflects_depth():
    snap = {
        "role": "main",
        "children": [
            {"role": "form", "children": [
                {"role": "button", "name": "Submit"},
            ]},
        ],
    }
    r = format_a11y_tree(snap)
    lines = r.text.split("\n")
    btn_line = next((line for line in lines if "btn_" in line), "")
    assert btn_line.startswith("  ")  # indented at least once


def test_form_elements_distinguished():
    snap = {
        "role": "form",
        "children": [
            {"role": "textbox", "name": "Email"},
            {"role": "combobox", "name": "Country"},
        ],
    }
    r = format_a11y_tree(snap)
    assert "inp_1" in r.ref_map
    assert "cmb_1" in r.ref_map
```

Create `tests/test_browser_driver.py`:

```python
"""Phase 5 tests — real browser (requires Playwright browsers installed)."""
import pytest

from infinite_dom.browser.playwright_driver import PlaywrightDriver
from infinite_dom.generator.dom_generator import DOMGenerator
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.models import ActionType, DOMAction


@pytest.mark.browser
@pytest.mark.asyncio
class TestPlaywrightDriver:
    async def test_load_and_snapshot(self):
        gen = DOMGenerator()
        page = gen.generate(task_id=1, seed=42)
        driver = PlaywrightDriver()
        try:
            await driver.start()
            await driver.load_page(page.generation_id, page.html)
            snap = await driver.snapshot()
            assert snap.a11y.text, "a11y text should not be empty"
            assert "btn_" in snap.a11y.text or "inp_" in snap.a11y.text
        finally:
            await driver.close()
            await stop_page_server()

    async def test_type_into_origin_field(self):
        gen = DOMGenerator()
        page = gen.generate(task_id=1, seed=42)
        driver = PlaywrightDriver()
        try:
            await driver.start()
            await driver.load_page(page.generation_id, page.html)
            snap = await driver.snapshot()

            # Find the first inp_X corresponding to origin (name will contain "From")
            origin_ref = None
            for ref_id, ref in snap.a11y.ref_map.items():
                if "from" in ref.name.lower() or "origin" in ref.name.lower():
                    if ref.role in {"textbox", "searchbox"}:
                        origin_ref = ref_id
                        break
            assert origin_ref is not None

            ok = await driver.execute(DOMAction(
                action_type=ActionType.TYPE,
                element_ref=origin_ref,
                text_value="Bengaluru",
            ))
            assert ok

            snap2 = await driver.snapshot()
            # Browser state should reflect the typed value
            inputs_combined = " ".join(snap2.state_dict["inputs"].values())
            assert "Bengaluru" in inputs_combined
        finally:
            await driver.close()
            await stop_page_server()

    async def test_invalid_ref_returns_false(self):
        gen = DOMGenerator()
        page = gen.generate(task_id=1, seed=42)
        driver = PlaywrightDriver()
        try:
            await driver.start()
            await driver.load_page(page.generation_id, page.html)
            await driver.snapshot()

            ok = await driver.execute(DOMAction(
                action_type=ActionType.CLICK,
                element_ref="btn_nonexistent_99",
            ))
            assert ok is False
        finally:
            await driver.close()
            await stop_page_server()
```

### 5.5 Milestone Gate 5

Run:
```bash
pytest tests/test_a11y_formatter.py -v
pytest tests/test_browser_driver.py -v -m browser
```

**GATE CRITERIA:** All tests pass, including the browser ones. If Playwright fails to launch Chromium under WSL2, run `playwright install-deps chromium` and retry. If still failing, log the exact error and raise it to user attention.

---

