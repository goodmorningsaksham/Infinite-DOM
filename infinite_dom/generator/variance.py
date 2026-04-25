"""
Variance pools for procedural DOM generation.

Each pool defines all the ways a single semantic element can surface on the page.
The generator picks from these pools using the episode seed.

Tasks 1-4: Booking flow (original + enhanced with validation, conditional fields, train selection)
Tasks 5-8: E-commerce flow (product search → filter → cart → checkout → confirm)
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field as dc_field

# ---------------------------------------------------------------------------
# Booking flow pools
# ---------------------------------------------------------------------------

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

TRAIN_NAMES = [
    "Shatabdi Express", "Rajdhani Express", "Duronto Express",
    "Garib Rath Express", "Humsafar Express", "Tejas Express",
    "Vande Bharat Express", "Jan Shatabdi Express",
]

TRAIN_TIMES = [
    "06:00 – 12:30", "08:15 – 14:45", "10:30 – 17:00",
    "14:00 – 20:30", "16:45 – 23:15", "20:00 – 08:15+1",
    "22:30 – 10:00+1", "05:30 – 11:45",
]

RETURN_DATE_LABELS = [
    "Return Date", "Coming Back", "Return Journey", "Back On",
]

TRIP_TYPE_LABELS = [
    "Trip Type", "Journey Type", "Travel Mode",
]

VALIDATION_ERROR_MESSAGES = {
    "origin_empty": [
        "Please enter your departure city",
        "Origin station is required",
        "Where are you travelling from?",
    ],
    "destination_empty": [
        "Please enter your destination",
        "Destination is required",
        "Where are you going?",
    ],
    "class_empty": [
        "Please select a travel class",
        "Choose your ticket class",
        "Seat type is required",
    ],
    "same_city": [
        "Origin and destination cannot be the same",
        "Please choose different stations",
    ],
    "return_date_empty": [
        "Return date is required for round-trip",
        "Please select a return date",
    ],
}

# ---------------------------------------------------------------------------
# E-commerce pools
# ---------------------------------------------------------------------------

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Books", "Home & Kitchen",
    "Sports", "Beauty", "Toys", "Groceries",
]

PRODUCTS = {
    "Electronics": [
        ("Wireless Bluetooth Headphones", 1299, 2499),
        ("USB-C Fast Charger 65W", 799, 1499),
        ("Portable Power Bank 20000mAh", 999, 1899),
        ("Smart Watch Fitness Tracker", 2499, 4999),
        ("Laptop Stand Aluminium", 1599, 2999),
        ("Webcam HD 1080p", 1899, 3499),
    ],
    "Clothing": [
        ("Cotton Crew Neck T-Shirt", 399, 799),
        ("Slim Fit Denim Jeans", 1299, 2499),
        ("Running Shoes Lightweight", 1999, 3999),
        ("Wool Blend Sweater", 1499, 2999),
        ("Waterproof Jacket", 2499, 4999),
        ("Formal Cotton Shirt", 899, 1799),
    ],
    "Books": [
        ("Python Programming Masterclass", 499, 899),
        ("Data Structures & Algorithms", 599, 999),
        ("Machine Learning Handbook", 799, 1499),
        ("Web Development Complete Guide", 649, 1199),
        ("System Design Interview Prep", 549, 999),
        ("Clean Code Principles", 449, 849),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Water Bottle", 399, 799),
        ("Non-Stick Frying Pan Set", 1299, 2499),
        ("LED Desk Lamp Adjustable", 899, 1699),
        ("Vacuum Insulated Lunch Box", 699, 1299),
        ("Kitchen Digital Weighing Scale", 599, 1099),
        ("Ceramic Coffee Mug Set", 499, 999),
    ],
}

ECOM_SEARCH_LABELS = [
    "Search products", "Find items", "What are you looking for?",
    "Search", "Search catalog", "Type to search",
]

ECOM_CATEGORY_LABELS = [
    "Category", "Department", "Filter by Category", "Type",
    "Product Type", "Browse by",
]

ECOM_ADD_CART_LABELS = [
    "Add to Cart", "Add to Bag", "Buy", "Add", "Add Item",
    "Put in Cart",
]

ECOM_CHECKOUT_LABELS = [
    "Proceed to Checkout", "Checkout", "Buy Now", "Go to Checkout",
    "Continue to Payment",
]

ECOM_CONFIRM_LABELS = [
    "Place Order", "Confirm Order", "Complete Purchase",
    "Submit Order", "Buy Now",
]

ECOM_NAME_LABELS = ["Full Name", "Name", "Your Name", "Recipient Name"]
ECOM_ADDRESS_LABELS = ["Address", "Street Address", "Delivery Address", "Address Line"]
ECOM_CITY_LABELS = ["City", "Town", "City / Town"]
ECOM_PIN_LABELS = ["PIN Code", "Zip Code", "Postal Code", "Pincode"]
ECOM_PHONE_LABELS = ["Phone", "Mobile Number", "Contact Number", "Phone Number"]

SHIPPING_NAMES = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Gupta",
    "Vijay Singh", "Ananya Reddy", "Rohit Verma", "Kavita Nair",
]
SHIPPING_ADDRESSES = [
    "42 MG Road", "15 Park Street", "78 Nehru Place",
    "23 Brigade Road", "101 Linking Road", "9 Banjara Hills",
]
SHIPPING_PINS = ["560001", "400001", "110001", "500034", "600001", "700001"]
SHIPPING_PHONES = ["9876543210", "8765432109", "7654321098", "9988776655"]

ECOM_DISTRACTOR_ADS = [
    "LIMITED OFFER: 50% off premium membership!",
    "Free delivery on orders above ₹499",
    "Download our app for exclusive deals!",
    "Today's Deal: Flat ₹200 off on first order",
]

ECOM_NEWSLETTER_TEXTS = [
    "Subscribe to our newsletter for deals!",
    "Get 10% off — enter your email below",
    "Join our mailing list for exclusive offers",
]


def random_css_prefix(rng: random.Random, length: int = 8) -> str:
    alphabet = string.ascii_lowercase
    return "".join(rng.choice(alphabet) for _ in range(length))


@dataclass
class VarianceProfile:
    """The selections made for a single episode."""
    template_id: str = "booking_flow"
    layout: str = "top_nav"

    # Booking-specific
    origin_label: str = "From"
    destination_label: str = "To"
    class_label: str = "Class"
    search_button_label: str = "Search"
    book_button_label: str = "Book"
    confirm_button_label: str = "Confirm"
    class_option_labels: list[str] = dc_field(default_factory=lambda: CLASS_OPTION_LABELS.copy())
    origin_city: str = ""
    destination_city: str = ""
    selected_class: str = ""
    trip_type: str = "one_way"
    return_date: str = ""
    return_date_label: str = "Return Date"
    trip_type_label: str = "Trip Type"
    target_train: str = ""
    enable_validation: bool = False
    train_count: int = 3
    available_trains: list[dict] = dc_field(default_factory=list)
    validation_msg_origin: str = "Please enter your departure city"
    validation_msg_destination: str = "Please enter your destination"
    validation_msg_class: str = "Please select a travel class"
    validation_msg_same_city: str = "Origin and destination cannot be the same"
    validation_msg_return_date: str = "Return date is required for round-trip"

    # E-commerce-specific
    ecom_search_label: str = "Search products"
    ecom_category_label: str = "Category"
    ecom_add_cart_label: str = "Add to Cart"
    ecom_checkout_label: str = "Proceed to Checkout"
    ecom_confirm_label: str = "Place Order"
    ecom_name_label: str = "Full Name"
    ecom_address_label: str = "Address"
    ecom_city_label: str = "City"
    ecom_pin_label: str = "PIN Code"
    ecom_phone_label: str = "Phone"
    target_product: str = ""
    target_category: str = ""
    target_product_price: int = 0
    shipping_name: str = ""
    shipping_address: str = ""
    shipping_city: str = ""
    shipping_pin: str = ""
    shipping_phone: str = ""
    products_in_category: list[tuple[str, int, int]] = dc_field(default_factory=list)
    distractor_products: list[tuple[str, int, int]] = dc_field(default_factory=list)

    # Common
    css_prefix: str = "std"
    include_cookie_banner: bool = False
    include_promo_modal: bool = False
    include_newsletter_popup: bool = False
    include_fake_buttons: bool = False
    aria_mode: str = "correct"
    field_order: list[str] = dc_field(default_factory=lambda: ["origin", "destination", "class"])
    cookie_banner_text: str = ""
    promo_text: str = ""
    newsletter_text: str = ""
    distractor_ad_text: str = ""


def select_variance(task_id: int, seed: int) -> VarianceProfile:
    """
    Apply variance based on task difficulty.

    Tasks 1-4: Booking flow
      Task 1: Clean form (minimal variance)
      Task 2: + label drift + form validation
      Task 3: + structural drift + conditional fields (round-trip) + train selection
      Task 4: + distractors + wrong ARIA + fake buttons + more trains

    Tasks 5-8: E-commerce flow
      Task 5: Clean store (standard labels)
      Task 6: + label drift + validation
      Task 7: + structural drift + more products + distractor ads
      Task 8: + full chaos (newsletter popups, fake buttons, noisy ARIA)
    """
    rng = random.Random(seed)

    if task_id <= 4:
        return _select_booking_variance(task_id, rng)
    else:
        return _select_ecommerce_variance(task_id, rng)


def _select_booking_variance(task_id: int, rng: random.Random) -> VarianceProfile:
    origin_city = rng.choice(CITIES)
    destination_city = rng.choice([c for c in CITIES if c != origin_city])
    selected_class = rng.choice(CLASSES)

    # Default: 3 trains for simple tasks
    default_train_names = rng.sample(TRAIN_NAMES, k=3)
    default_trains = [
        {"name": n, "time": rng.choice(TRAIN_TIMES), "price": rng.randint(250, 2500)}
        for n in default_train_names
    ]

    profile = VarianceProfile(
        template_id="booking_flow",
        origin_city=origin_city,
        destination_city=destination_city,
        selected_class=selected_class,
        cookie_banner_text=rng.choice(COOKIE_BANNER_TEXTS),
        promo_text=rng.choice(PROMO_TEXTS),
        available_trains=default_trains,
    )

    if task_id >= 2:
        profile.origin_label = rng.choice(ORIGIN_LABELS)
        profile.destination_label = rng.choice(DESTINATION_LABELS)
        profile.class_label = rng.choice(CLASS_LABELS)
        profile.search_button_label = rng.choice(SEARCH_BUTTON_LABELS)
        profile.book_button_label = rng.choice(BOOK_BUTTON_LABELS)
        profile.confirm_button_label = rng.choice(CONFIRM_BUTTON_LABELS)
        profile.css_prefix = random_css_prefix(rng)
        profile.enable_validation = True
        profile.validation_msg_origin = rng.choice(VALIDATION_ERROR_MESSAGES["origin_empty"])
        profile.validation_msg_destination = rng.choice(VALIDATION_ERROR_MESSAGES["destination_empty"])
        profile.validation_msg_class = rng.choice(VALIDATION_ERROR_MESSAGES["class_empty"])
        profile.validation_msg_same_city = rng.choice(VALIDATION_ERROR_MESSAGES["same_city"])
        profile.validation_msg_return_date = rng.choice(VALIDATION_ERROR_MESSAGES["return_date_empty"])

    if task_id >= 3:
        profile.layout = rng.choice(LAYOUTS)
        profile.field_order = ["origin", "destination", "class"]
        rng.shuffle(profile.field_order)

        if rng.random() < 0.5:
            profile.trip_type = "round_trip"
            day = rng.randint(1, 28)
            month = rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
            profile.return_date = f"{day} {month}"
            profile.return_date_label = rng.choice(RETURN_DATE_LABELS)
            profile.trip_type_label = rng.choice(TRIP_TYPE_LABELS)
            profile.field_order.append("trip_type")
            profile.field_order.append("return_date")

        train_names = rng.sample(TRAIN_NAMES, k=min(5, len(TRAIN_NAMES)))
        profile.target_train = rng.choice(train_names)
        profile.train_count = len(train_names)
        profile.available_trains = [
            {"name": n, "time": rng.choice(TRAIN_TIMES), "price": rng.randint(250, 2500)}
            for n in train_names
        ]

    if task_id >= 4:
        profile.include_cookie_banner = rng.random() < 0.7
        profile.include_promo_modal = rng.random() < 0.5
        profile.include_fake_buttons = True
        profile.aria_mode = rng.choices(["correct", "noisy", "wrong"], weights=[3, 4, 2])[0]
        profile.distractor_ad_text = rng.choice(ECOM_DISTRACTOR_ADS)
        profile.newsletter_text = rng.choice(ECOM_NEWSLETTER_TEXTS)
        profile.include_newsletter_popup = rng.random() < 0.4

        train_names = rng.sample(TRAIN_NAMES, k=min(7, len(TRAIN_NAMES)))
        profile.target_train = rng.choice(train_names)
        profile.train_count = len(train_names)
        profile.available_trains = [
            {"name": n, "time": rng.choice(TRAIN_TIMES), "price": rng.randint(250, 2500)}
            for n in train_names
        ]

    return profile


def _select_ecommerce_variance(task_id: int, rng: random.Random) -> VarianceProfile:
    target_category = rng.choice(list(PRODUCTS.keys()))
    products_in_cat = PRODUCTS[target_category]
    target_idx = rng.randint(0, len(products_in_cat) - 1)
    target_name, target_sale, target_mrp = products_in_cat[target_idx]

    # Get distractor products from other categories
    other_cats = [c for c in PRODUCTS.keys() if c != target_category]
    distractor_cat = rng.choice(other_cats)
    distractor_products = rng.sample(PRODUCTS[distractor_cat], k=min(3, len(PRODUCTS[distractor_cat])))

    shipping_city = rng.choice(CITIES)

    profile = VarianceProfile(
        template_id="ecommerce_flow",
        target_product=target_name,
        target_category=target_category,
        target_product_price=target_sale,
        products_in_category=list(products_in_cat),
        distractor_products=distractor_products,
        shipping_name=rng.choice(SHIPPING_NAMES),
        shipping_address=rng.choice(SHIPPING_ADDRESSES),
        shipping_city=shipping_city,
        shipping_pin=rng.choice(SHIPPING_PINS),
        shipping_phone=rng.choice(SHIPPING_PHONES),
        cookie_banner_text=rng.choice(COOKIE_BANNER_TEXTS),
        promo_text=rng.choice(PROMO_TEXTS),
        newsletter_text=rng.choice(ECOM_NEWSLETTER_TEXTS),
        distractor_ad_text=rng.choice(ECOM_DISTRACTOR_ADS),
    )

    if task_id >= 6:
        profile.ecom_search_label = rng.choice(ECOM_SEARCH_LABELS)
        profile.ecom_category_label = rng.choice(ECOM_CATEGORY_LABELS)
        profile.ecom_add_cart_label = rng.choice(ECOM_ADD_CART_LABELS)
        profile.ecom_checkout_label = rng.choice(ECOM_CHECKOUT_LABELS)
        profile.ecom_confirm_label = rng.choice(ECOM_CONFIRM_LABELS)
        profile.ecom_name_label = rng.choice(ECOM_NAME_LABELS)
        profile.ecom_address_label = rng.choice(ECOM_ADDRESS_LABELS)
        profile.ecom_city_label = rng.choice(ECOM_CITY_LABELS)
        profile.ecom_pin_label = rng.choice(ECOM_PIN_LABELS)
        profile.ecom_phone_label = rng.choice(ECOM_PHONE_LABELS)
        profile.css_prefix = random_css_prefix(rng)
        profile.enable_validation = True

    if task_id >= 7:
        profile.layout = rng.choice(LAYOUTS)
        profile.include_cookie_banner = rng.random() < 0.5
        profile.distractor_products = rng.sample(
            PRODUCTS[rng.choice(other_cats)],
            k=min(4, len(PRODUCTS[distractor_cat])),
        )

    if task_id >= 8:
        profile.include_promo_modal = rng.random() < 0.6
        profile.include_newsletter_popup = rng.random() < 0.5
        profile.include_fake_buttons = True
        profile.aria_mode = rng.choices(["correct", "noisy", "wrong"], weights=[3, 4, 2])[0]

    return profile
