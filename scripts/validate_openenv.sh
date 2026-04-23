#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
if command -v openenv &> /dev/null; then
    openenv validate .
else
    echo "[WARN] openenv CLI not found — install may have failed"
    echo "[WARN] skipping validation; running smoke test instead"
    python scripts/smoke_test.py
fi
