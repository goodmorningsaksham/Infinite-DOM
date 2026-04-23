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
