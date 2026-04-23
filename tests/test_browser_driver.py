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
