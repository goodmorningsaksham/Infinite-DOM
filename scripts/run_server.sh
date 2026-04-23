#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=$(pwd)
uvicorn infinite_dom.server.app:app --host 0.0.0.0 --port 8000 --reload
