"""
FastAPI app for the Infinite DOM environment.

Uses OpenEnv's create_app to auto-generate REST + WebSocket endpoints,
and adds a `/` route for the HuggingFace Space dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    try:
        from openenv.http_server import create_app
    except ImportError:
        from openenv_core.http_server import create_app  # type: ignore

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment
from infinite_dom.models import DOMAction, DOMObservation

app = create_app(
    InfiniteDOMEnvironment,
    DOMAction,
    DOMObservation,
    env_name="infinite_dom",
    max_concurrent_envs=1,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_DESCRIPTIONS = {
    1: "Booking: clean form — standard labels, no distractors",
    2: "Booking: label drift — randomised labels + form validation",
    3: "Booking: structural drift — shuffled fields, conditional round-trip, train selection",
    4: "Booking: full chaos — distractors, noisy ARIA, fake buttons, 7 trains",
    5: "E-commerce: clean store — search, filter, cart, checkout",
    6: "E-commerce: label drift — randomised labels + checkout validation",
    7: "E-commerce: structural drift — layout changes, more distractors, more products",
    8: "E-commerce: full chaos — newsletter popups, fake buttons, noisy ARIA",
}

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def dashboard_root() -> str:
    return _DASHBOARD_HTML


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "env": "infinite_dom", "version": "1.0.0"})


@app.get("/tasks")
async def list_tasks():
    return JSONResponse({
        "tasks": [
            {"task_id": tid, "description": desc}
            for tid, desc in TASK_DESCRIPTIONS.items()
        ]
    })
