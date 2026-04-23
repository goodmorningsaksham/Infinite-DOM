## PHASE 7 — OpenEnv Server + Oracle + Training Data

### 7.1 Goal

- Wire the environment into an OpenEnv FastAPI server via `create_app()`
- Build the hardcoded oracle solver for the booking flow
- Generate an SFT dataset (observation, action) pairs from oracle trajectories
- Final `openenv validate .` passes

### 7.2 Execution Checklist

#### 7.2.1 Create `infinite_dom/server/dashboard.html`

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Infinite DOM — Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 2rem; background:#f8fafc; }
    h1 { color:#111827; }
    .badge { background:#dbeafe; color:#1e3a8a; padding:0.25rem 0.5rem; border-radius:4px; font-size:0.85rem; }
    pre { background:#1f2937; color:#f9fafb; padding:1rem; border-radius:8px; overflow:auto; }
    .card { background:white; padding:1.5rem; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,.1); margin-bottom:1rem; }
    .endpoint { font-family:monospace; background:#f1f5f9; padding:0.25rem 0.5rem; border-radius:4px; }
  </style>
</head>
<body>
  <h1>The Infinite DOM <span class="badge">OpenEnv</span></h1>
  <p>
    A procedurally generated training environment for web agents. Every episode
    spins up a fresh, live, interactive web application — same task, different
    layout, different labels, different structure.
  </p>

  <div class="card">
    <h2>OpenEnv endpoints</h2>
    <ul>
      <li><span class="endpoint">POST /reset</span> — start a new episode</li>
      <li><span class="endpoint">POST /step</span> — execute an action</li>
      <li><span class="endpoint">GET /state</span> — full episode state (grading)</li>
      <li><span class="endpoint">GET /health</span> — liveness probe</li>
      <li><span class="endpoint">GET /schema</span> — action + observation schema</li>
      <li><span class="endpoint">GET /metadata</span> — environment metadata</li>
      <li><span class="endpoint">WS /ws</span> — WebSocket session</li>
    </ul>
  </div>

  <div class="card">
    <h2>Tasks</h2>
    <ul>
      <li><strong>Task 1 — Clean Form:</strong> standard layout, correct ARIA, no distractors</li>
      <li><strong>Task 2 — Label Drift:</strong> randomized button + field labels</li>
      <li><strong>Task 3 — Structural Drift:</strong> label drift + randomized layout + field order</li>
      <li><strong>Task 4 — Full Chaos:</strong> everything + distractors + misleading ARIA</li>
    </ul>
  </div>

  <div class="card">
    <h2>Quickstart</h2>
<pre>curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": 1, "seed": 42}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action_type": "type", "element_ref": "inp_1", "text_value": "Bengaluru"}'
curl http://localhost:8000/state</pre>
  </div>
</body>
</html>
```

#### 7.2.2 Create `infinite_dom/server/app.py`

```python
"""
FastAPI app for the Infinite DOM environment.

Uses OpenEnv's create_app to auto-generate REST + WebSocket endpoints,
and adds a `/` route for the HuggingFace Space dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.responses import HTMLResponse

# OpenEnv with fallbacks
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


_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def dashboard_root() -> str:
    return _DASHBOARD_HTML
```

#### 7.2.3 Create `inference.py` (top-level)

```python
"""
OpenEnv-required top-level inference entrypoint.

Loads the environment, runs a single evaluation episode using a
hand-written heuristic policy (placeholder), and emits the required
stdout log format:

    [RESET] ...
    [STEP N] action=... reward=... done=...
    [END] task_id=... steps=N score=X.XX rewards=...
"""
from __future__ import annotations

import asyncio
import sys

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.graders import grade
from infinite_dom.models import ActionType, DOMAction


def heuristic_action(obs) -> DOMAction:
    """
    Stub heuristic — picks an obvious action from the a11y tree.
    Real agents will replace this with their policy.
    """
    # Find first textbox without a value, type into it
    for line in obs.a11y_tree.split("\n"):
        if "ref=inp_" in line and 'value=""' in line:
            ref = line.split("ref=")[1].split(" ")[0].rstrip("]")
            return DOMAction(action_type=ActionType.TYPE, element_ref=ref, text_value="Mumbai")
    # Otherwise click first button
    for line in obs.a11y_tree.split("\n"):
        if "ref=btn_" in line:
            ref = line.split("ref=")[1].split(" ")[0].rstrip("]")
            return DOMAction(action_type=ActionType.CLICK, element_ref=ref)
    return DOMAction(action_type=ActionType.WAIT)


def run_episode(task_id: int = 1, seed: int = 42) -> None:
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=task_id, seed=seed)
        print(f"[RESET] task_id={task_id} seed={seed} instruction={obs.task_instruction!r}")
        total_reward = 0.0
        step_num = 0
        while not obs.done:
            action = heuristic_action(obs)
            obs = env.step(action)
            step_num += 1
            total_reward += obs.reward or 0.0
            print(f"[STEP {step_num}] action={action.action_type.value} "
                  f"ref={action.element_ref} reward={obs.reward:.3f} done={obs.done}")
        score = grade(task_id, env.state)
        print(f"[END] task_id={task_id} steps={step_num} score={score:.4f} "
              f"rewards={total_reward:.3f}")
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())


if __name__ == "__main__":
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_episode(task_id=task_id, seed=seed)
```

#### 7.2.4 Create `infinite_dom/oracle/booking_flow_oracle.py`

A hardcoded solver that demonstrates what the agent is supposed to learn.

```python
"""
Oracle for the booking flow — hand-written solver used to:
  1. Verify the environment is solvable end-to-end
  2. Generate SFT trajectories for the imitation-learning warmup phase
"""
from __future__ import annotations

from dataclasses import dataclass

from infinite_dom.models import ActionType, DOMAction
from infinite_dom.task_graph import TaskGraph


@dataclass
class OracleStep:
    """One step in the oracle trajectory: match a line in the a11y tree, then act."""
    action_type: ActionType
    # Match either by role+name substring, or by literal ref if determined on-line
    role_hint: str = ""
    name_substrings: tuple[str, ...] = ()
    text_value_template: str = ""  # may use {origin}, {destination}, {class}
    scroll_delta: int = 0


def _find_ref(a11y_tree: str, role_hint: str, name_substrings: tuple[str, ...]) -> str | None:
    """Scan the a11y tree text for a line matching role and any of the substrings."""
    wants_lower = tuple(s.lower() for s in name_substrings)
    for line in a11y_tree.split("\n"):
        if f"role={role_hint}" not in line:
            continue
        lower = line.lower()
        if any(s in lower for s in wants_lower):
            # Extract ref
            idx = line.find("ref=")
            if idx == -1:
                continue
            tail = line[idx + 4:]
            ref = tail.split(" ")[0].rstrip("]")
            return ref
    return None


def oracle_policy(obs_a11y_tree: str, task_graph: TaskGraph) -> DOMAction:
    """
    Decide the next oracle action given the current a11y tree and task graph.

    Strategy: advance the next incomplete node, in the intended order.
    """
    origin = task_graph.params["origin"]
    destination = task_graph.params["destination"]
    seat_class = task_graph.params["class"]

    # Dismiss distractor banners / modals first if present
    for banner_names in (("accept", "cookie"), ("close", "promo"), ("dismiss",)):
        ref = _find_ref(obs_a11y_tree, "button", banner_names)
        if ref:
            return DOMAction(action_type=ActionType.CLICK, element_ref=ref)

    completed = set(task_graph.get_completed_nodes_from_tree_hack(obs_a11y_tree)) \
        if hasattr(task_graph, "get_completed_nodes_from_tree_hack") else set()
    # Simpler: we don't have direct browser_state here, just act in the natural order

    # Step 1: Type origin
    origin_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("from", "origin", "depart", "start"),
    )
    if origin_ref and f'value="{origin}"' not in obs_a11y_tree and origin.lower() not in obs_a11y_tree.lower():
        return DOMAction(action_type=ActionType.TYPE, element_ref=origin_ref, text_value=origin)

    # Step 2: Type destination
    dest_ref = _find_ref(
        obs_a11y_tree, "textbox",
        ("to", "destination", "arrive", "going"),
    )
    if dest_ref and destination.lower() not in obs_a11y_tree.lower():
        return DOMAction(action_type=ActionType.TYPE, element_ref=dest_ref, text_value=destination)

    # Step 3: Select class
    class_ref = _find_ref(
        obs_a11y_tree, "combobox",
        ("class", "cabin", "fare", "seat"),
    )
    # If we see class combobox but no selected="..." matching class short-form, set it
    if class_ref:
        # Heuristic: if the tree shows an un-selected combobox, type/fill the class name
        # We rely on the combobox accepting text input; this works for native <select> too
        # because the driver's `fill` path becomes a set-value equivalent for simple cases.
        return DOMAction(
            action_type=ActionType.TYPE,
            element_ref=class_ref,
            text_value=seat_class,
        )

    # Step 4: Click search button
    search_ref = _find_ref(
        obs_a11y_tree, "button",
        ("search", "find", "go", "check", "look"),
    )
    if search_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=search_ref)

    # Step 5: Click first book button on results
    book_ref = _find_ref(
        obs_a11y_tree, "button",
        ("book", "reserve", "purchase", "buy", "secure"),
    )
    if book_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=book_ref)

    # Step 6: Click confirm
    confirm_ref = _find_ref(
        obs_a11y_tree, "button",
        ("confirm", "complete", "finalize", "place"),
    )
    if confirm_ref:
        return DOMAction(action_type=ActionType.CLICK, element_ref=confirm_ref)

    # Fallback
    return DOMAction(action_type=ActionType.WAIT)
```

#### 7.2.5 Create `tests/test_oracle.py`

```python
"""Phase 7 — oracle solves the booking flow on Task 1."""
import asyncio

import pytest

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.graders import grade
from infinite_dom.oracle.booking_flow_oracle import oracle_policy


@pytest.mark.e2e
@pytest.mark.browser
@pytest.mark.parametrize("seed", [1, 7, 42, 101, 999])
def test_oracle_solves_task_1(seed):
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=seed)
        done = False
        max_steps = 20
        i = 0
        while not done and i < max_steps:
            action = oracle_policy(obs.a11y_tree, env._current_page.task_graph)  # type: ignore
            obs = env.step(action)
            done = obs.done
            i += 1
        # Oracle should get most nodes
        assert len(env.state.task_graph_completed) >= 3, \
            f"seed={seed}: oracle only completed {env.state.task_graph_completed}"
        score = grade(1, env.state)
        assert score > 0.3
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())
```

#### 7.2.6 Create `training/generate_oracle_data.py`

```python
"""
Run oracle against N episodes across all tasks, record (observation, action) pairs
as JSONL for SFT warmup training later.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.oracle.booking_flow_oracle import oracle_policy


OUT_DIR = Path("training/data")


def run(num_episodes: int = 30, tasks: tuple[int, ...] = (1, 2)) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "oracle_trajectories.jsonl"

    env = InfiniteDOMEnvironment()
    total_written = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for task_id in tasks:
                for ep in range(num_episodes):
                    seed = ep * 13 + task_id * 1000
                    obs = env.reset(task_id=task_id, seed=seed)
                    instruction = obs.task_instruction
                    steps_in_ep = 0
                    while not obs.done and steps_in_ep < 25:
                        action = oracle_policy(obs.a11y_tree, env._current_page.task_graph)  # type: ignore
                        record = {
                            "task_id": task_id,
                            "seed": seed,
                            "step": steps_in_ep,
                            "instruction": instruction,
                            "observation": obs.a11y_tree,
                            "action": action.model_dump(),
                        }
                        f.write(json.dumps(record) + "\n")
                        total_written += 1
                        obs = env.step(action)
                        steps_in_ep += 1
                    print(f"[task={task_id} ep={ep} seed={seed}] steps={steps_in_ep} "
                          f"completed={len(env.state.task_graph_completed)}/{len(env.state.task_graph_total)}")
        print(f"\n[DONE] wrote {total_written} records to {out_path}")
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run(num_episodes=n)
```

#### 7.2.7 Create `client.py` (top-level)

```python
"""
Convenience client wrapper. Agents use EnvClient from OpenEnv directly;
this module re-exports for discoverability.
"""
try:
    from openenv.core.env_client import EnvClient
except ImportError:
    try:
        from openenv.env_client import EnvClient
    except ImportError:
        from openenv_core.env_client import EnvClient  # type: ignore


__all__ = ["EnvClient"]
```

#### 7.2.8 Create `training/train_infinite_dom.ipynb` — a NOTEBOOK STUB

Notebooks are JSON. Create this file with a minimal structure — the user will fill in training code during the hackathon when compute credits are available. The stub must include:
- A markdown intro cell
- A cell that installs Unsloth + TRL
- A cell that connects to the env via EnvClient
- A cell that runs an oracle-vs-random baseline evaluation
- Placeholder cells for SFT warmup, GRPO training, and eval

Exact JSON structure — write to `training/train_infinite_dom.ipynb`:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Infinite DOM — Training Notebook (stub)\n",
        "\n",
        "Run on Colab T4 (prep smoke test) or A100 (full training during hackathon).\n",
        "\n",
        "## Phases\n",
        "1. Install dependencies (Unsloth + TRL)\n",
        "2. Connect to the Infinite DOM env\n",
        "3. Baseline eval — random vs oracle\n",
        "4. SFT warmup on oracle trajectories\n",
        "5. GRPO RL with curriculum\n",
        "6. Held-out generalization eval\n",
        "7. WebArena transfer eval (stretch)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 1 — Install deps\n",
        "!pip install -q unsloth\n",
        "!pip install -q trl transformers accelerate peft\n",
        "!pip install -q httpx pydantic"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 2 — Connect to running Infinite DOM server\n",
        "# Server must be running at INFINITE_DOM_URL (default: http://localhost:8000)\n",
        "import os\n",
        "INFINITE_DOM_URL = os.environ.get('INFINITE_DOM_URL', 'http://localhost:8000')\n",
        "\n",
        "# TODO: EnvClient import path may differ by version\n",
        "from openenv.core.env_client import EnvClient\n",
        "client = EnvClient(INFINITE_DOM_URL)\n",
        "print(client.metadata())"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 3 — Baseline eval placeholder\n",
        "# TODO: run N episodes with a random policy vs the oracle, record completion rates\n",
        "# See infinite_dom/oracle/booking_flow_oracle.py for the oracle reference"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 4 — SFT warmup (TODO hackathon day)\n",
        "# Load oracle trajectories from training/data/oracle_trajectories.jsonl\n",
        "# Fine-tune Qwen-2.5-3B-Instruct via Unsloth LoRA on (obs, action_json) pairs"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 5 — GRPO curriculum training (TODO hackathon day)\n",
        "# Progressively train on Tasks 1 -> 4 with GRPO from TRL"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "# Cell 6 — Held-out eval + WebArena eval (stretch)\n",
        "# Plot training reward curve and seen-vs-unseen generalization gap"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.11"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

#### 7.2.9 Create `scripts/run_server.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=$(pwd)
uvicorn infinite_dom.server.app:app --host 0.0.0.0 --port 8000 --reload
```

#### 7.2.10 Create `scripts/smoke_test.py`

```python
"""Smoke test — start server in subprocess, hit endpoints, shut down."""
import subprocess
import time

import httpx


def main():
    proc = subprocess.Popen(
        ["uvicorn", "infinite_dom.server.app:app", "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # Wait for server to start
        for _ in range(30):
            try:
                r = httpx.get("http://127.0.0.1:8001/health", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        print("[OK] /health responded")

        # Reset
        r = httpx.post(
            "http://127.0.0.1:8001/reset",
            json={"task_id": 1, "seed": 42},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json()
        print(f"[OK] /reset returned instruction={obs.get('task_instruction', '')!r}")

        # Dashboard
        r = httpx.get("http://127.0.0.1:8001/", timeout=5)
        r.raise_for_status()
        assert "Infinite DOM" in r.text
        print("[OK] / (dashboard) responded")

        print("\n[SMOKE TEST PASSED]")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
```

#### 7.2.11 Create `scripts/validate_openenv.sh`

```bash
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
```

### 7.3 Milestone Gate 7 — FINAL

Run all of these in sequence. Each must pass:

1. `pytest tests/test_oracle.py -v -m "e2e and browser"`
2. `python training/generate_oracle_data.py 10` — should write a JSONL file
3. `python scripts/smoke_test.py` — server responds to /health, /reset, /
4. `bash scripts/validate_openenv.sh` — passes or logs clear reason

**GATE CRITERIA:** All four pass. If `openenv validate` is unavailable (CLI not installed), log this in BUILD_LOG.md and treat the smoke test as the substitute gate.

---

