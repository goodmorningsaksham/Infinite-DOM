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
    1: "Clean booking form — standard labels, no distractors",
    2: "Label drift — randomised field labels and button text",
    3: "Structural drift — randomised layout and field order",
    4: "Full chaos — distractors, noisy ARIA, everything randomised",
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
