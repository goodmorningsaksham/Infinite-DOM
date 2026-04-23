"""
DOM generator — the core research contribution.

Given (task_id, seed), produces:
  - A full HTML string (rendered from the archetype template with variance)
  - A TaskGraph describing success criteria for that rendering
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, select_autoescape

from infinite_dom.config import CONFIG
from infinite_dom.generator.variance import (
    COOKIE_BANNER_TEXTS,
    PROMO_TEXTS,
    VarianceProfile,
    select_variance,
)
from infinite_dom.task_graph import TaskGraph, make_booking_flow_graph


@dataclass
class GeneratedPage:
    generation_id: str
    html: str
    task_graph: TaskGraph
    profile: VarianceProfile
    task_id: int
    seed: int


class DOMGenerator:
    def __init__(self):
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(CONFIG.templates_dir)),
            autoescape=select_autoescape(["html", "jinja"]),
        )

    def generate(self, task_id: int = 1, seed: int | None = None) -> GeneratedPage:
        """Generate one page for (task_id, seed)."""
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31 - 1)

        rng = random.Random(seed)
        profile = select_variance(task_id=task_id, seed=seed)

        cookie_banner_text = rng.choice(COOKIE_BANNER_TEXTS)
        promo_text = rng.choice(PROMO_TEXTS)

        template = self.jinja_env.get_template("booking_flow.jinja")
        html = template.render(
            profile=profile,
            p=profile.css_prefix,
            cookie_banner_text=cookie_banner_text,
            promo_text=promo_text,
        )

        instruction = (
            f"Book a {profile.selected_class} ticket from "
            f"{profile.origin_city} to {profile.destination_city}"
        )

        task_graph = make_booking_flow_graph(
            task_id=task_id,
            template_id="booking_flow",
            origin=profile.origin_city,
            destination=profile.destination_city,
            seat_class=profile.selected_class,
            instruction=instruction,
        )

        generation_id = f"gen_{task_id}_{seed}"

        return GeneratedPage(
            generation_id=generation_id,
            html=html,
            task_graph=task_graph,
            profile=profile,
            task_id=task_id,
            seed=seed,
        )
