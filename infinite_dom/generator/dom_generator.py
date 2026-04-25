"""
DOM generator — the core research contribution.

Given (task_id, seed), produces:
  - A full HTML string (rendered from the archetype template with variance)
  - A TaskGraph describing success criteria for that rendering
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, select_autoescape

from infinite_dom.config import CONFIG
from infinite_dom.generator.variance import (
    PRODUCT_CATEGORIES,
    VarianceProfile,
    select_variance,
)
from infinite_dom.task_graph import TaskGraph, make_booking_flow_graph, make_ecommerce_flow_graph


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

        profile = select_variance(task_id=task_id, seed=seed)

        if profile.template_id == "ecommerce_flow":
            return self._generate_ecommerce(task_id, seed, profile)
        return self._generate_booking(task_id, seed, profile)

    def _generate_booking(self, task_id: int, seed: int, profile: VarianceProfile) -> GeneratedPage:
        trains_json = json.dumps([
            {"id": i + 1, "name": t["name"], "time": t["time"], "price": t["price"]}
            for i, t in enumerate(profile.available_trains)
        ])

        template = self.jinja_env.get_template("booking_flow.jinja")
        html = template.render(
            profile=profile,
            p=profile.css_prefix,
            trains_json=trains_json,
        )

        instruction = (
            f"Book a {profile.selected_class} ticket from "
            f"{profile.origin_city} to {profile.destination_city}"
        )
        if profile.trip_type == "round_trip":
            instruction += f" (round-trip, returning {profile.return_date})"
        if profile.target_train and task_id >= 3:
            instruction += f". Select the {profile.target_train}."

        extra_params: dict[str, str] = {}
        if profile.trip_type == "round_trip":
            extra_params["trip_type"] = "round_trip"
            extra_params["return_date"] = profile.return_date
        if profile.target_train and task_id >= 3:
            extra_params["target_train"] = profile.target_train

        task_graph = make_booking_flow_graph(
            task_id=task_id,
            template_id="booking_flow",
            origin=profile.origin_city,
            destination=profile.destination_city,
            seat_class=profile.selected_class,
            instruction=instruction,
            **extra_params,
        )

        return GeneratedPage(
            generation_id=f"gen_{task_id}_{seed}",
            html=html,
            task_graph=task_graph,
            profile=profile,
            task_id=task_id,
            seed=seed,
        )

    def _generate_ecommerce(self, task_id: int, seed: int, profile: VarianceProfile) -> GeneratedPage:
        products_json = json.dumps([
            {"id": i + 1, "name": p[0], "price": p[1], "mrp": p[2]}
            for i, p in enumerate(profile.products_in_category)
        ])
        distractor_json = json.dumps([
            {"id": 100 + i, "name": p[0], "price": p[1], "mrp": p[2]}
            for i, p in enumerate(profile.distractor_products)
        ])
        shipping_json = json.dumps({
            "name": profile.shipping_name,
            "address": profile.shipping_address,
            "city": profile.shipping_city,
            "pin": profile.shipping_pin,
            "phone": profile.shipping_phone,
        })

        template = self.jinja_env.get_template("ecommerce_flow.jinja")
        html = template.render(
            profile=profile,
            p=profile.css_prefix,
            products_json=products_json,
            distractor_json=distractor_json,
            shipping_json=shipping_json,
            categories=PRODUCT_CATEGORIES,
        )

        instruction = (
            f"Buy '{profile.target_product}' from the {profile.target_category} category. "
            f"Ship to {profile.shipping_name} at {profile.shipping_address}, "
            f"{profile.shipping_city} {profile.shipping_pin}. Phone: {profile.shipping_phone}."
        )

        extra_params: dict[str, str] = {
            "shipping_name": profile.shipping_name,
            "shipping_address": profile.shipping_address,
            "shipping_city": profile.shipping_city,
            "shipping_pin": profile.shipping_pin,
            "shipping_phone": profile.shipping_phone,
        }

        task_graph = make_ecommerce_flow_graph(
            task_id=task_id,
            template_id="ecommerce_flow",
            target_product=profile.target_product,
            target_category=profile.target_category,
            instruction=instruction,
            **extra_params,
        )

        return GeneratedPage(
            generation_id=f"gen_{task_id}_{seed}",
            html=html,
            task_graph=task_graph,
            profile=profile,
            task_id=task_id,
            seed=seed,
        )
