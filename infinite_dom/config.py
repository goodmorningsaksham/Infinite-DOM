"""Runtime configuration pulled from environment variables with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    host: str = os.environ.get("INFINITE_DOM_HOST", "0.0.0.0")
    port: int = int(os.environ.get("INFINITE_DOM_PORT", "8000"))
    page_server_port: int = int(os.environ.get("INFINITE_DOM_PAGE_SERVER_PORT", "9000"))
    max_steps: int = int(os.environ.get("INFINITE_DOM_MAX_STEPS", "35"))
    playwright_headless: bool = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
    max_failed_actions: int = 5
    a11y_tree_max_tokens: int = 1500
    templates_dir: Path = Path(__file__).parent / "generator" / "templates"


CONFIG = Config()
