## PHASE 1 — Project Scaffold

### 1.1 Goal

All directory structure exists. All dependency files exist. Python venv created. All dependencies installed. `BUILD_LOG.md` created.

### 1.2 Execution Checklist

Run commands in this order. Record each in BUILD_LOG.md.

#### 1.2.1 Create directory tree

Create all directories listed in §1 (project root structure). Empty `__init__.py` files in every Python package directory.

#### 1.2.2 Create `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Testing
.pytest_cache/
.coverage
htmlcov/
.ruff_cache/

# Env
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project artifacts
generated_pages/
training/checkpoints/
training/data/
*.log
wandb/
runs/

# Playwright
playwright-report/
test-results/
.playwright/
```

#### 1.2.3 Create `.env.example`

```
# HuggingFace token for model downloads and Space deployment
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=

# Optional: Weights & Biases for training logs
# If empty, training falls back to local matplotlib logging
# Get from: https://wandb.ai/authorize
WANDB_API_KEY=

# Server config
INFINITE_DOM_HOST=0.0.0.0
INFINITE_DOM_PORT=8000

# Generator config
INFINITE_DOM_PAGE_SERVER_PORT=9000
INFINITE_DOM_MAX_STEPS=25

# Playwright
PLAYWRIGHT_BROWSERS_PATH=0
PLAYWRIGHT_HEADLESS=true
```

#### 1.2.4 Create `requirements.txt`

Pin to versions known to work together. Use these exact versions unless one fails to install, in which case bump minor version and log it.

```
# OpenEnv — use latest; install from GitHub if not on PyPI
openenv @ git+https://github.com/meta-pytorch/OpenEnv.git

# Web framework
fastapi==0.115.4
uvicorn[standard]==0.32.0
pydantic==2.9.2

# Templating
jinja2==3.1.4

# Browser automation
playwright==1.48.0

# HTTP client (for serving generated pages)
aiohttp==3.10.10
httpx==0.27.2

# Utilities
python-dotenv==1.0.1
PyYAML==6.0.2
rich==13.9.3

# For notebook-style training work (full training loaded separately on Colab)
numpy==1.26.4
```

**Note on OpenEnv install:** If `pip install git+https://github.com/meta-pytorch/OpenEnv.git` fails, try `pip install openenv-core` as the PyPI fallback. Log whichever worked in BUILD_LOG.md. If both fail, stop and alert user — this is a blocker.

#### 1.2.5 Create `requirements-dev.txt`

```
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-timeout==2.3.1
ruff==0.7.1
ipykernel==6.29.5
beautifulsoup4==4.12.3  # Used in tests to parse generated HTML
```

#### 1.2.6 Create `pyproject.toml`

```toml
[project]
name = "infinite-dom"
version = "0.1.0"
description = "Procedurally generated DOM training environment for web agents"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]  # Line length handled by formatter

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
timeout = 60
markers = [
    "slow: marks tests as slow (deselect with -m 'not slow')",
    "browser: marks tests that require a real browser",
    "e2e: marks end-to-end tests",
]
```

#### 1.2.7 Create `Dockerfile`

```dockerfile
# Use Playwright's official image — comes with Chromium preinstalled
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY infinite_dom/ ./infinite_dom/
COPY inference.py client.py openenv.yaml ./
COPY scripts/ ./scripts/

EXPOSE 8000 9000

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_HEADLESS=true \
    INFINITE_DOM_HOST=0.0.0.0 \
    INFINITE_DOM_PORT=8000

CMD ["uvicorn", "infinite_dom.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 1.2.8 Create `.dockerignore`

```
.git
.venv
venv
__pycache__
*.pyc
tests/
training/
.env
.pytest_cache
.ruff_cache
BUILD_LOG.md
README.md
```

#### 1.2.9 Create `openenv.yaml` (initial version — will expand in Phase 7)

```yaml
spec_version: 1
name: infinite_dom
display_name: "The Infinite DOM"
description: >
  A procedurally generated training environment for web agents.
  Every episode spins up a fresh, live, interactive web application
  with randomized layout, labels, and structure while preserving the
  underlying task semantics. Agents must learn semantic understanding
  rather than positional memorization to succeed across variants.
type: space
runtime: fastapi
app: infinite_dom.server.app:app
port: 8000
tasks:
  - id: task_1_clean_form
    name: Clean Form
    difficulty: easy
    description: >
      Baseline booking flow with clean ARIA, no distractors,
      standard label set.
  - id: task_2_label_drift
    name: Label Drift
    difficulty: medium
    description: >
      Same task structure as Task 1 but with randomized button and
      field labels drawn from a synonym pool.
  - id: task_3_structural_drift
    name: Structural Drift
    difficulty: hard
    description: >
      Label drift plus randomized layout and field order.
  - id: task_4_full_chaos
    name: Full Chaos
    difficulty: expert
    description: >
      All variance enabled plus distractors (cookie banner, promo modals)
      and occasionally misleading ARIA labels.
action_space:
  type: structured
  fields:
    action_type:
      type: string
      choices: [click, type, scroll, wait, back]
    element_ref:
      type: string
    text_value:
      type: string
    scroll_delta:
      type: integer
      min: -2000
      max: 2000
observation_space:
  fields:
    a11y_tree:
      type: string
      description: "Accessibility tree serialized as indented text"
    task_instruction:
      type: string
    task_progress:
      type: list[string]
    step_count:
      type: integer
```

#### 1.2.10 Create `scripts/setup_dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "==> Infinite DOM dev setup"
echo "==> Target: WSL2 Ubuntu 22.04"

# Python version check
if ! python3.11 --version > /dev/null 2>&1; then
    echo "ERROR: Python 3.11 not found. Install with: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "==> Creating virtualenv"
    python3.11 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing runtime requirements"
pip install -r requirements.txt

echo "==> Installing dev requirements"
pip install -r requirements-dev.txt

echo "==> Installing Playwright browsers"
playwright install chromium
playwright install-deps chromium

echo "==> Done. Activate with: source .venv/bin/activate"
```

Make it executable: `chmod +x scripts/setup_dev.sh`

#### 1.2.11 Create `BUILD_LOG.md` stub

```markdown
# Infinite DOM Build Log

Audit trail for autonomous build. One section per phase.
```

### 1.3 Commands to Run

```bash
cd infinite-dom
bash scripts/setup_dev.sh
source .venv/bin/activate
playwright --version   # Verify Playwright installed
python -c "import fastapi, pydantic, jinja2, playwright; print('OK')"
```

Also attempt:
```bash
python -c "from openenv.core.env_server.interfaces import Environment; print('OpenEnv OK')"
```

If the OpenEnv import fails, try the PyPI fallback in requirements.txt and reinstall.

### 1.4 Milestone Gate 1

Create `tests/test_phase1_scaffold.py`:

```python
"""Phase 1 gate: scaffold integrity."""
import importlib
from pathlib import Path


def test_all_required_directories_exist():
    required = [
        "infinite_dom",
        "infinite_dom/generator",
        "infinite_dom/generator/templates",
        "infinite_dom/browser",
        "infinite_dom/environment",
        "infinite_dom/oracle",
        "infinite_dom/server",
        "training",
        "tests",
        "scripts",
    ]
    for d in required:
        assert Path(d).is_dir(), f"Missing directory: {d}"


def test_required_root_files_exist():
    required = [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "Dockerfile",
        ".dockerignore",
        "openenv.yaml",
        ".gitignore",
        ".env.example",
        "BUILD_LOG.md",
    ]
    for f in required:
        assert Path(f).is_file(), f"Missing file: {f}"


def test_all_init_py_files_exist():
    packages = [
        "infinite_dom",
        "infinite_dom/generator",
        "infinite_dom/browser",
        "infinite_dom/environment",
        "infinite_dom/oracle",
        "infinite_dom/server",
        "tests",
    ]
    for p in packages:
        assert Path(f"{p}/__init__.py").is_file(), f"Missing __init__.py: {p}"


def test_core_imports_available():
    for mod in ["fastapi", "pydantic", "jinja2", "playwright", "yaml"]:
        importlib.import_module(mod)
```

Run: `pytest tests/test_phase1_scaffold.py -v`

**GATE CRITERIA:** All 4 tests pass. If any fail, fix before advancing.

### 1.5 Phase 1 BUILD_LOG Entry

Write in BUILD_LOG.md exactly which deps installed, which version of OpenEnv was used (git or PyPI), and any deviations.

---

