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
    releases resources.
    """

    def __init__(self):
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._current_gen_id: str | None = None
        self._last_snapshot: BrowserSnapshot | None = None

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

        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

        for attempt in range(3):
            try:
                await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.info("page.goto failed (attempt %d): %s — retrying", attempt + 1, e)
                await asyncio.sleep(0.5)
        try:
            await self._page.wait_for_function(
                "() => window.Alpine !== undefined",
                timeout=5000,
            )
        except Exception:
            pass
        await asyncio.sleep(0.15)

    async def snapshot(self) -> BrowserSnapshot:
        """Capture current state of the page."""
        assert self._page is not None, "No page loaded"

        raw = await self._get_a11y_tree()
        a11y = format_a11y_tree(raw)
        url = self._page.url

        state_dict = await self._extract_browser_state()
        snap = BrowserSnapshot(a11y=a11y, url=url, state_dict=state_dict)
        self._last_snapshot = snap
        return snap

    async def _get_a11y_tree(self) -> dict[str, Any] | None:
        """Get the accessibility tree via CDP and convert to a nested dict."""
        assert self._page is not None
        try:
            cdp = await self._page.context.new_cdp_session(self._page)
            try:
                result = await cdp.send("Accessibility.getFullAXTree")
            finally:
                await cdp.detach()

            nodes = result.get("nodes", [])
            if not nodes:
                return None

            node_map: dict[str, dict] = {}
            for n in nodes:
                nid = n["nodeId"]
                if n.get("ignored"):
                    node_map[nid] = {
                        "role": "",
                        "name": "",
                        "children": [],
                        "_child_ids": n.get("childIds", []),
                    }
                    continue
                role_val = n.get("role", {}).get("value", "")
                if role_val == "RootWebArea":
                    role_val = "WebArea"
                name_val = n.get("name", {}).get("value", "") if isinstance(n.get("name"), dict) else ""
                value_val = n.get("value", {}).get("value", "") if isinstance(n.get("value"), dict) else ""

                props = {}
                for prop in n.get("properties", []):
                    pname = prop.get("name", "")
                    pval = prop.get("value", {}).get("value")
                    if pname and pval is not None:
                        props[pname] = pval

                node_map[nid] = {
                    "role": role_val,
                    "name": name_val,
                    "value": value_val,
                    "children": [],
                    "_child_ids": n.get("childIds", []),
                    "selected": props.get("selected", False),
                    "checked": props.get("checked", False),
                }

            for nid, node in node_map.items():
                for cid in node.pop("_child_ids", []):
                    if cid in node_map:
                        node["children"].append(node_map[cid])

            root_id = nodes[0]["nodeId"]
            return node_map.get(root_id)
        except Exception as e:
            logger.warning("CDP accessibility tree failed: %s", e)
            return {"role": "WebArea", "name": "", "children": []}

    async def _extract_browser_state(self) -> dict[str, Any]:
        """Extract a task-graph-compatible dict from the live page."""
        assert self._page is not None

        inputs = await self._page.evaluate(
            """() => {
                const result = {};
                document.querySelectorAll('input, textarea').forEach(el => {
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

        visible_text = await self._page.evaluate(
            """() => {
                const text = document.body.innerText || '';
                return Array.from(new Set(text.split(/[\\n\\r\\s]+/).filter(s => s.length > 0)));
            }"""
        )

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

    async def execute(self, action: DOMAction, timeout_s: float = 10.0) -> bool:
        """
        Execute the action. Returns True on success, False if the action
        could not be carried out (element not found, wrong type, etc.).
        Per-action timeout prevents any single action from hanging.
        """
        assert self._page is not None, "No page loaded"
        assert self._last_snapshot is not None, "Must snapshot before executing"

        try:
            return await asyncio.wait_for(
                self._execute_inner(action), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.info("Action timed out after %.1fs: %s", timeout_s, action.action_type)
            return False
        except Exception as e:
            logger.info("Action execution failed: %s", e)
            return False

    async def _execute_inner(self, action: DOMAction) -> bool:
        """Inner action execution logic, wrapped by execute() with timeout."""
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
            if ref.role in {"combobox", "listbox"}:
                el_id = await locator.get_attribute("id")
                selected = await self._page.evaluate(
                    """(args) => {
                        const [elId, target] = args;
                        const el = elId ? document.getElementById(elId) : null;
                        if (!el || !el.options) return false;
                        const lower = target.toLowerCase();
                        for (const opt of el.options) {
                            if (opt.text.toLowerCase().includes(lower)
                                || opt.value.toLowerCase().includes(lower)) {
                                el.value = opt.value;
                                el.dispatchEvent(new Event('input', {bubbles: true}));
                                el.dispatchEvent(new Event('change', {bubbles: true}));
                                return true;
                            }
                        }
                        return false;
                    }""",
                    [el_id, action.text_value],
                )
                if not selected:
                    return False
            else:
                await locator.fill(action.text_value, timeout=3000)
            await asyncio.sleep(0.1)
            return True

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
            await loc.wait_for(state="visible", timeout=2000)
            return loc
        except Exception:
            if ref.role in {"textbox", "searchbox"} and ref.name:
                try:
                    loc = self._page.get_by_label(ref.name).first
                    await loc.wait_for(state="visible", timeout=1500)
                    return loc
                except Exception:
                    pass
            return None
