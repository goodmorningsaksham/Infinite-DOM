## PHASE 4 — DOM Generator

### 4.1 Goal

Produce a `DOMGenerator` that, given a `task_id` and `seed`, generates:
1. A complete, interactive HTML page using Jinja2 + Alpine.js
2. A `TaskGraph` describing what "success" means on that page
3. Variance at the right axes: layout, labels, CSS class names, field order, distractors, ARIA correctness

### 4.2 Key Design Decisions

- **Template language:** Jinja2 (already in requirements)
- **Client-side reactivity:** Alpine.js via CDN (no build step, ~15KB)
- **Page serving:** Small aiohttp-based local server on port 9000. Each generated page gets a unique URL like `http://localhost:9000/page/<generation_id>`
- **Why not React:** React build step per episode = 30 sec × 1000 episodes = days wasted. Alpine.js gives the same interactive SPA behavior without a build step.

### 4.3 Execution Checklist

#### 4.3.1 Create `infinite_dom/generator/variance.py`

```python
"""
Variance pools for procedural DOM generation.

Each pool defines all the ways a single semantic element can surface on the page.
The generator picks from these pools using the episode seed.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass


# =========================================================================
# Label variance
# =========================================================================

ORIGIN_LABELS = [
    "From", "Origin", "Departing From", "Depart", "Starting Point",
    "From Station", "Leaving From",
]

DESTINATION_LABELS = [
    "To", "Destination", "Arriving At", "Arrive", "End Point",
    "To Station", "Going To",
]

CLASS_LABELS = [
    "Class", "Cabin", "Fare Class", "Travel Class", "Seat Type",
    "Coach Class", "Booking Class",
]

SEARCH_BUTTON_LABELS = [
    "Search", "Find Trains", "Go", "Check Availability", "Search Now",
    "Find My Train", "Look Up Trains",
]

BOOK_BUTTON_LABELS = [
    "Book", "Reserve", "Purchase", "Confirm Booking", "Buy Now",
    "Book Now", "Secure Seat",
]

CONFIRM_BUTTON_LABELS = [
    "Confirm", "Pay & Confirm", "Complete Booking", "Finalize",
    "Place Booking",
]

CLASS_OPTION_LABELS = [
    "Sleeper (SL)", "AC 3 Tier (3A)", "AC 2 Tier (2A)", "AC 1st Class (1A)",
    "Chair Car (CC)", "Second Sitting (2S)",
]

# =========================================================================
# Layout variance
# =========================================================================

LAYOUTS = ["top_nav", "sidebar_nav", "hamburger", "single_column"]

# =========================================================================
# Distractors
# =========================================================================

COOKIE_BANNER_TEXTS = [
    "We use cookies to improve your experience",
    "This site uses cookies — accept to continue",
    "Accept cookies and tracking?",
]

PROMO_TEXTS = [
    "Flat 15% off all bookings this week!",
    "New user? Get ₹500 cashback!",
    "Tatkal bookings now open — hurry!",
]

# =========================================================================
# City + class parameter pools
# =========================================================================

CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]

CLASSES = ["Sleeper", "AC 3 Tier", "AC 2 Tier", "AC 1st Class", "Chair Car"]

# Short-form code variants of each class. The graph checks for substrings
# so any of these in the selected option value satisfies the predicate.
CLASS_SHORT_FORMS = {
    "Sleeper": ["SL", "Sleeper", "sleeper"],
    "AC 3 Tier": ["3A", "AC3", "3 Tier", "ac 3 tier"],
    "AC 2 Tier": ["2A", "AC2", "2 Tier", "ac 2 tier", "2AC"],
    "AC 1st Class": ["1A", "AC1", "1st Class", "ac 1st class"],
    "Chair Car": ["CC", "Chair", "chair car"],
}


# =========================================================================
# Utility
# =========================================================================

def random_css_prefix(rng: random.Random, length: int = 8) -> str:
    """Generate a random CSS class name prefix to prevent selector memorization."""
    alphabet = string.ascii_lowercase
    return "".join(rng.choice(alphabet) for _ in range(length))


@dataclass
class VarianceProfile:
    """The selections made for a single episode."""
    layout: str
    origin_label: str
    destination_label: str
    class_label: str
    search_button_label: str
    book_button_label: str
    confirm_button_label: str
    class_option_labels: list[str]
    css_prefix: str
    include_cookie_banner: bool
    include_promo_modal: bool
    aria_mode: str  # "correct" | "noisy" | "wrong"
    field_order: list[str]  # permutation of ["origin", "destination", "class"]
    origin_city: str
    destination_city: str
    selected_class: str


def select_variance(task_id: int, seed: int) -> VarianceProfile:
    """
    Apply variance based on task difficulty.

    Task 1: clean form (minimal variance)
    Task 2: + label drift
    Task 3: + structural drift (layout + field order)
    Task 4: + distractors + potentially wrong ARIA
    """
    rng = random.Random(seed)

    # Pick origin/destination/class from distinct cities
    origin_city = rng.choice(CITIES)
    destination_city = rng.choice([c for c in CITIES if c != origin_city])
    selected_class = rng.choice(CLASSES)

    # Defaults: clean
    layout = "top_nav"
    origin_label = "From"
    destination_label = "To"
    class_label = "Class"
    search_button_label = "Search"
    book_button_label = "Book"
    confirm_button_label = "Confirm"
    class_option_labels = CLASS_OPTION_LABELS.copy()
    css_prefix = "std"
    include_cookie_banner = False
    include_promo_modal = False
    aria_mode = "correct"
    field_order = ["origin", "destination", "class"]

    if task_id >= 2:
        # Label drift
        origin_label = rng.choice(ORIGIN_LABELS)
        destination_label = rng.choice(DESTINATION_LABELS)
        class_label = rng.choice(CLASS_LABELS)
        search_button_label = rng.choice(SEARCH_BUTTON_LABELS)
        book_button_label = rng.choice(BOOK_BUTTON_LABELS)
        confirm_button_label = rng.choice(CONFIRM_BUTTON_LABELS)
        css_prefix = random_css_prefix(rng)

    if task_id >= 3:
        # Structural drift
        layout = rng.choice(LAYOUTS)
        rng.shuffle(field_order)

    if task_id >= 4:
        # Full chaos
        include_cookie_banner = rng.random() < 0.7
        include_promo_modal = rng.random() < 0.5
        aria_mode = rng.choices(["correct", "noisy", "wrong"], weights=[4, 3, 1])[0]

    return VarianceProfile(
        layout=layout,
        origin_label=origin_label,
        destination_label=destination_label,
        class_label=class_label,
        search_button_label=search_button_label,
        book_button_label=book_button_label,
        confirm_button_label=confirm_button_label,
        class_option_labels=class_option_labels,
        css_prefix=css_prefix,
        include_cookie_banner=include_cookie_banner,
        include_promo_modal=include_promo_modal,
        aria_mode=aria_mode,
        field_order=field_order,
        origin_city=origin_city,
        destination_city=destination_city,
        selected_class=selected_class,
    )
```

#### 4.3.2 Create `infinite_dom/generator/templates/_base_styles.jinja`

```jinja
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f5f5f7; color: #222; }
  .{{ p }}-header { background: #111827; color: white; padding: 1rem 2rem; }
  .{{ p }}-container { max-width: 900px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,.08); }
  .{{ p }}-field { margin-bottom: 1.25rem; }
  .{{ p }}-field label { display: block; font-weight: 600; margin-bottom: .4rem; }
  .{{ p }}-field input, .{{ p }}-field select {
    width: 100%; padding: .65rem; font-size: 1rem;
    border: 1px solid #d1d5db; border-radius: 6px;
  }
  .{{ p }}-btn {
    padding: .75rem 1.5rem; font-size: 1rem; background: #2563eb; color: white;
    border: none; border-radius: 6px; cursor: pointer;
  }
  .{{ p }}-btn:hover { background: #1d4ed8; }
  .{{ p }}-results-row { border-bottom: 1px solid #e5e7eb; padding: 1rem 0; display: flex; justify-content: space-between; align-items: center; }
  .{{ p }}-banner { background: #fef3c7; padding: .75rem 1.5rem; display: flex; justify-content: space-between; align-items: center; }
  .{{ p }}-modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.5); display: flex;
    align-items: center; justify-content: center; z-index: 100;
  }
  .{{ p }}-modal {
    background: white; padding: 2rem; border-radius: 12px; max-width: 500px;
  }
  .{{ p }}-confirmation {
    background: #d1fae5; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;
  }
  .{{ p }}-sidebar {
    width: 250px; float: left; padding: 1rem; background: #f3f4f6; min-height: 100vh;
  }
</style>
```

#### 4.3.3 Create `infinite_dom/generator/templates/booking_flow.jinja`

This is the critical file. It produces a working, interactive booking flow using Alpine.js for state. The template has conditional blocks for layout variants, field order, and distractors. Read carefully.

```jinja
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Train Booking</title>
  <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
  {% include '_base_styles.jinja' %}
</head>
<body x-data="bookingApp()">

  {# ==================== Header / Navigation ==================== #}
  {% if profile.layout == 'top_nav' %}
    <div class="{{ profile.css_prefix }}-header" role="banner">
      <h1>TrainBook Express</h1>
      <nav role="navigation">
        <a href="#" style="color:white;margin-right:1rem;">Home</a>
        <a href="#" style="color:white;margin-right:1rem;">My Bookings</a>
        <a href="#" style="color:white;">Help</a>
      </nav>
    </div>
  {% elif profile.layout == 'sidebar_nav' %}
    <div class="{{ profile.css_prefix }}-sidebar" role="navigation">
      <h2>TrainBook</h2>
      <ul>
        <li><a href="#">Book Tickets</a></li>
        <li><a href="#">My Bookings</a></li>
        <li><a href="#">Help</a></li>
      </ul>
    </div>
  {% elif profile.layout == 'hamburger' %}
    <div class="{{ profile.css_prefix }}-header">
      <button aria-label="Open menu">☰</button>
      <span style="margin-left:1rem;">TrainBook</span>
    </div>
  {% else %}
    <div style="padding:1rem 2rem;background:#111827;color:white;">
      <h1>TrainBook</h1>
    </div>
  {% endif %}

  {# ==================== Cookie banner distractor ==================== #}
  {% if profile.include_cookie_banner %}
    <div x-show="showCookieBanner" class="{{ profile.css_prefix }}-banner" role="alert">
      <span>{{ cookie_banner_text }}</span>
      <button
        class="{{ profile.css_prefix }}-btn"
        x-on:click="showCookieBanner = false"
        aria-label="Accept cookies"
      >Accept</button>
    </div>
  {% endif %}

  {# ==================== Search form ==================== #}
  <div class="{{ profile.css_prefix }}-container" x-show="currentPage === 'search'" role="main">
    <h2>Search Trains</h2>
    <form x-on:submit.prevent="submitSearch()" role="form" aria-label="Search Trains">
      {% for field in profile.field_order %}
        {% if field == 'origin' %}
          <div class="{{ profile.css_prefix }}-field">
            <label for="{{ profile.css_prefix }}_origin">{{ profile.origin_label }}</label>
            <input
              id="{{ profile.css_prefix }}_origin"
              type="text"
              name="{{ profile.origin_label | lower | replace(' ', '_') }}"
              x-model="form.origin"
              aria-label="{% if profile.aria_mode == 'wrong' %}Search query{% else %}{{ profile.origin_label }}{% endif %}"
              placeholder="Enter origin city"
            />
          </div>
        {% elif field == 'destination' %}
          <div class="{{ profile.css_prefix }}-field">
            <label for="{{ profile.css_prefix }}_destination">{{ profile.destination_label }}</label>
            <input
              id="{{ profile.css_prefix }}_destination"
              type="text"
              name="{{ profile.destination_label | lower | replace(' ', '_') }}"
              x-model="form.destination"
              aria-label="{% if profile.aria_mode == 'wrong' %}Newsletter{% else %}{{ profile.destination_label }}{% endif %}"
              placeholder="Enter destination city"
            />
          </div>
        {% elif field == 'class' %}
          <div class="{{ profile.css_prefix }}-field">
            <label for="{{ profile.css_prefix }}_class">{{ profile.class_label }}</label>
            <select
              id="{{ profile.css_prefix }}_class"
              name="{{ profile.class_label | lower | replace(' ', '_') }}"
              x-model="form.travelClass"
              aria-label="{{ profile.class_label }}"
            >
              <option value="">-- Select --</option>
              {% for option_label in profile.class_option_labels %}
                <option value="{{ option_label }}">{{ option_label }}</option>
              {% endfor %}
            </select>
          </div>
        {% endif %}
      {% endfor %}

      <button
        type="submit"
        class="{{ profile.css_prefix }}-btn"
        aria-label="{{ profile.search_button_label }}"
      >{{ profile.search_button_label }}</button>
    </form>
  </div>

  {# ==================== Results page ==================== #}
  <div class="{{ profile.css_prefix }}-container" x-show="currentPage === 'results'" role="main">
    <h2>Available Trains</h2>
    <template x-for="train in trains" :key="train.id">
      <div class="{{ profile.css_prefix }}-results-row">
        <div>
          <strong x-text="train.name"></strong><br>
          <span x-text="train.time"></span>
        </div>
        <button
          class="{{ profile.css_prefix }}-btn"
          x-on:click="selectTrain(train)"
          x-bind:aria-label="'{{ profile.book_button_label }} ' + train.name"
        >{{ profile.book_button_label }}</button>
      </div>
    </template>
    <button x-on:click="currentPage = 'search'" style="margin-top:1rem;">Back to search</button>
  </div>

  {# ==================== Confirmation page ==================== #}
  <div class="{{ profile.css_prefix }}-container" x-show="currentPage === 'confirm'" role="main">
    <h2>Confirm Your Booking</h2>
    <p>Review details below and confirm.</p>
    <ul>
      <li>From: <span x-text="form.origin"></span></li>
      <li>To: <span x-text="form.destination"></span></li>
      <li>Class: <span x-text="form.travelClass"></span></li>
      <li x-show="selectedTrain">Train: <span x-text="selectedTrain && selectedTrain.name"></span></li>
    </ul>
    <button
      class="{{ profile.css_prefix }}-btn"
      x-on:click="confirmBooking()"
      aria-label="{{ profile.confirm_button_label }}"
    >{{ profile.confirm_button_label }}</button>
  </div>

  {# ==================== Confirmation success ==================== #}
  <div class="{{ profile.css_prefix }}-container" x-show="currentPage === 'done'" role="main">
    <div class="{{ profile.css_prefix }}-confirmation">
      <h2>Booking Confirmed!</h2>
      <p>Your booking reference:</p>
      <p><strong x-text="bookingId"></strong></p>
    </div>
  </div>

  {# ==================== Promo modal distractor ==================== #}
  {% if profile.include_promo_modal %}
    <div x-show="showPromoModal" class="{{ profile.css_prefix }}-modal-overlay">
      <div class="{{ profile.css_prefix }}-modal">
        <h3>{{ promo_text }}</h3>
        <button x-on:click="showPromoModal = false" class="{{ profile.css_prefix }}-btn" aria-label="Close promo">Close</button>
      </div>
    </div>
  {% endif %}

  <script>
    function bookingApp() {
      return {
        currentPage: 'search',
        showCookieBanner: {{ 'true' if profile.include_cookie_banner else 'false' }},
        showPromoModal: {{ 'true' if profile.include_promo_modal else 'false' }},
        form: { origin: '', destination: '', travelClass: '' },
        trains: [
          { id: 1, name: 'Shatabdi Express', time: '06:00 – 12:30' },
          { id: 2, name: 'Rajdhani Express', time: '20:00 – 08:15' },
          { id: 3, name: 'Duronto Express', time: '22:30 – 10:00' },
        ],
        selectedTrain: null,
        bookingId: '',

        submitSearch() {
          if (!this.form.origin || !this.form.destination || !this.form.travelClass) {
            return;
          }
          // Simulate URL navigation via history — task graph checks `url`
          history.pushState({}, '', '/results');
          this.currentPage = 'results';
        },
        selectTrain(train) {
          this.selectedTrain = train;
          history.pushState({}, '', '/confirm');
          this.currentPage = 'confirm';
        },
        confirmBooking() {
          this.bookingId = 'PNR' + Math.floor(Math.random() * 9000000000 + 1000000000);
          history.pushState({}, '', '/done');
          this.currentPage = 'done';
        },
      };
    }
  </script>
</body>
</html>
```

**Note:** The template uses `x-on:submit` / `x-on:click` (not `@submit` / `@click`) so that serialized Alpine directives are consistently recognizable in accessibility tree output and don't confuse the Jinja parser.

#### 4.3.4 Create `infinite_dom/generator/dom_generator.py`

```python
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

        # Distractor text selection
        cookie_banner_text = rng.choice(COOKIE_BANNER_TEXTS)
        promo_text = rng.choice(PROMO_TEXTS)

        template = self.jinja_env.get_template("booking_flow.jinja")
        html = template.render(
            profile=profile,
            p=profile.css_prefix,  # shorthand used in base styles
            cookie_banner_text=cookie_banner_text,
            promo_text=promo_text,
        )

        # Instruction
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
```

#### 4.3.5 Create `infinite_dom/generator/serve_html.py`

A small local HTTP server that serves generated HTML pages on demand. Each page is registered with an ID; the browser navigates to `http://localhost:9000/page/<id>`.

```python
"""
Local HTTP server for generated pages.

Pages are registered in memory keyed by generation_id. Playwright
navigates to http://localhost:{port}/page/<generation_id> to access them.
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from infinite_dom.config import CONFIG

logger = logging.getLogger(__name__)


class PageServer:
    def __init__(self, port: int = CONFIG.page_server_port):
        self.port = port
        self.pages: dict[str, str] = {}
        self.app = web.Application()
        self.app.router.add_get("/page/{gen_id}", self._handle_page)
        self.app.router.add_get("/results", self._handle_results_placeholder)
        self.app.router.add_get("/confirm", self._handle_results_placeholder)
        self.app.router.add_get("/done", self._handle_results_placeholder)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def register_page(self, gen_id: str, html: str) -> str:
        """Register HTML under an ID. Returns the URL to navigate to."""
        self.pages[gen_id] = html
        return f"http://localhost:{self.port}/page/{gen_id}"

    def unregister_page(self, gen_id: str) -> None:
        self.pages.pop(gen_id, None)

    async def _handle_page(self, request: web.Request) -> web.Response:
        gen_id = request.match_info["gen_id"]
        html = self.pages.get(gen_id)
        if html is None:
            return web.Response(status=404, text="page not found")
        return web.Response(text=html, content_type="text/html")

    async def _handle_results_placeholder(self, request: web.Request) -> web.Response:
        """
        When Alpine does `history.pushState('/results')`, the browser
        path changes but doesn't reload. If it DOES reload (shouldn't
        happen in normal flow), return a simple placeholder so the
        task graph's URL predicate still fires.
        """
        return web.Response(text="<html><body>Results</body></html>", content_type="text/html")

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await self._site.start()
        logger.info("Page server started on port %d", self.port)

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()


# Singleton accessor
_INSTANCE: PageServer | None = None


async def get_page_server() -> PageServer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PageServer()
        await _INSTANCE.start()
    return _INSTANCE


async def stop_page_server() -> None:
    global _INSTANCE
    if _INSTANCE is not None:
        await _INSTANCE.stop()
        _INSTANCE = None
```

### 4.4 Tests

Create `tests/test_generator.py`:

```python
"""Phase 4 tests — generator produces valid HTML."""
import re

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
        assert len(p.html) > 500  # substantive

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
        assert len(inputs) >= 2  # at least origin + destination
        selects = soup.find_all("select")
        assert len(selects) >= 1  # class selector

    def test_seeded_generation_deterministic(self, gen):
        p1 = gen.generate(task_id=1, seed=42)
        p2 = gen.generate(task_id=1, seed=42)
        assert p1.html == p2.html
        assert p1.profile.origin_city == p2.profile.origin_city

    def test_different_seeds_different_outputs(self, gen):
        p1 = gen.generate(task_id=2, seed=1)
        p2 = gen.generate(task_id=2, seed=2)
        # Either labels, params, or layout should differ
        differs = (
            p1.profile.origin_label != p2.profile.origin_label
            or p1.profile.origin_city != p2.profile.origin_city
            or p1.profile.css_prefix != p2.profile.css_prefix
        )
        assert differs

    def test_task_4_may_include_distractors(self, gen):
        # With enough seeds, task 4 should produce at least one cookie banner and one promo
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
        # The prefix must appear in the class attribute of at least one element
        assert p.profile.css_prefix in p.html


class TestPageSavedToDisk:
    """Write generated pages to a scratch dir — useful for manual inspection."""

    def test_can_write_sample(self, gen=None, tmp_path=None):
        from infinite_dom.generator.dom_generator import DOMGenerator
        g = DOMGenerator()
        out = tmp_path / "sample.html"
        p = g.generate(task_id=4, seed=123)
        out.write_text(p.html, encoding="utf-8")
        assert out.exists()
        assert out.stat().st_size > 500
```

### 4.5 Milestone Gate 4

Run: `pytest tests/test_generator.py -v`

**GATE CRITERIA:** All tests pass. Also run this one-shot manual check and save the output HTML so a human can eyeball it if needed:

```bash
python -c "
from pathlib import Path
from infinite_dom.generator.dom_generator import DOMGenerator
g = DOMGenerator()
Path('generated_pages').mkdir(exist_ok=True)
for task_id in [1, 2, 3, 4]:
    for seed in [1, 2]:
        p = g.generate(task_id=task_id, seed=seed)
        Path(f'generated_pages/task{task_id}_seed{seed}.html').write_text(p.html, encoding='utf-8')
        print(f'Wrote task{task_id}_seed{seed}.html ({len(p.html)} bytes), origin={p.profile.origin_city}, dest={p.profile.destination_city}, layout={p.profile.layout}')
"
```

If the eight HTML files render in a browser with working form inputs, Phase 4 passes.

---

