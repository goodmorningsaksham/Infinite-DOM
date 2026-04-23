"""Phase 4 tests — generator produces valid HTML."""

import pytest
from bs4 import BeautifulSoup

from infinite_dom.generator.dom_generator import DOMGenerator


class TestDOMGenerator:
    @pytest.fixture
    def gen(self):
        return DOMGenerator()

    def test_generate_returns_html_string(self, gen):
        p = gen.generate(task_id=1, seed=42)
        assert isinstance(p.html, str)
        assert len(p.html) > 500

    def test_html_is_parseable(self, gen):
        p = gen.generate(task_id=1, seed=42)
        soup = BeautifulSoup(p.html, "html.parser")
        assert soup.find("body") is not None

    def test_task_graph_attached(self, gen):
        p = gen.generate(task_id=1, seed=42)
        assert p.task_graph is not None
        assert len(p.task_graph.nodes) == 5
        assert p.task_graph.instruction != ""

    def test_instruction_references_params(self, gen):
        p = gen.generate(task_id=1, seed=42)
        assert p.profile.origin_city in p.task_graph.instruction
        assert p.profile.destination_city in p.task_graph.instruction

    def test_html_contains_form_elements(self, gen):
        p = gen.generate(task_id=1, seed=42)
        soup = BeautifulSoup(p.html, "html.parser")
        inputs = soup.find_all("input")
        assert len(inputs) >= 2
        selects = soup.find_all("select")
        assert len(selects) >= 1

    def test_seeded_generation_deterministic(self, gen):
        p1 = gen.generate(task_id=1, seed=42)
        p2 = gen.generate(task_id=1, seed=42)
        assert p1.html == p2.html
        assert p1.profile.origin_city == p2.profile.origin_city

    def test_different_seeds_different_outputs(self, gen):
        p1 = gen.generate(task_id=2, seed=1)
        p2 = gen.generate(task_id=2, seed=2)
        differs = (
            p1.profile.origin_label != p2.profile.origin_label
            or p1.profile.origin_city != p2.profile.origin_city
            or p1.profile.css_prefix != p2.profile.css_prefix
        )
        assert differs

    def test_task_4_may_include_distractors(self, gen):
        found_cookie = False
        found_promo = False
        for seed in range(30):
            p = gen.generate(task_id=4, seed=seed)
            if p.profile.include_cookie_banner:
                found_cookie = True
            if p.profile.include_promo_modal:
                found_promo = True
            if found_cookie and found_promo:
                break
        assert found_cookie
        assert found_promo

    def test_task_1_never_has_distractors(self, gen):
        for seed in range(30):
            p = gen.generate(task_id=1, seed=seed)
            assert not p.profile.include_cookie_banner
            assert not p.profile.include_promo_modal

    def test_css_prefix_in_generated_classes(self, gen):
        p = gen.generate(task_id=2, seed=99)
        assert p.profile.css_prefix in p.html


class TestPageSavedToDisk:
    """Write generated pages to a scratch dir — useful for manual inspection."""

    def test_can_write_sample(self, tmp_path):
        g = DOMGenerator()
        out = tmp_path / "sample.html"
        p = g.generate(task_id=4, seed=123)
        out.write_text(p.html, encoding="utf-8")
        assert out.exists()
        assert out.stat().st_size > 500
