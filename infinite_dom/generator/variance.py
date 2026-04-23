"""
Variance pools for procedural DOM generation.

Each pool defines all the ways a single semantic element can surface on the page.
The generator picks from these pools using the episode seed.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

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

LAYOUTS = ["top_nav", "sidebar_nav", "hamburger", "single_column"]

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

CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]

CLASSES = ["Sleeper", "AC 3 Tier", "AC 2 Tier", "AC 1st Class", "Chair Car"]

CLASS_SHORT_FORMS = {
    "Sleeper": ["SL", "Sleeper", "sleeper"],
    "AC 3 Tier": ["3A", "AC3", "3 Tier", "ac 3 tier"],
    "AC 2 Tier": ["2A", "AC2", "2 Tier", "ac 2 tier", "2AC"],
    "AC 1st Class": ["1A", "AC1", "1st Class", "ac 1st class"],
    "Chair Car": ["CC", "Chair", "chair car"],
}


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
    aria_mode: str
    field_order: list[str]
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

    origin_city = rng.choice(CITIES)
    destination_city = rng.choice([c for c in CITIES if c != origin_city])
    selected_class = rng.choice(CLASSES)

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
        origin_label = rng.choice(ORIGIN_LABELS)
        destination_label = rng.choice(DESTINATION_LABELS)
        class_label = rng.choice(CLASS_LABELS)
        search_button_label = rng.choice(SEARCH_BUTTON_LABELS)
        book_button_label = rng.choice(BOOK_BUTTON_LABELS)
        confirm_button_label = rng.choice(CONFIRM_BUTTON_LABELS)
        css_prefix = random_css_prefix(rng)

    if task_id >= 3:
        layout = rng.choice(LAYOUTS)
        rng.shuffle(field_order)

    if task_id >= 4:
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
