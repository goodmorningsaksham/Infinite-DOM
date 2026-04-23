---
title: The Infinite DOM
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# The Infinite DOM

**Procedurally generated training environment for web agents that don't break when websites change.**

> **OpenEnv Hackathon India 2026**
> - **Primary theme:** #3.1 World Modeling — Professional Tasks
> - **Bonus sub-theme:** Patronus AI — Consumer Workflows with Schema Drift
> - **Secondary theme:** #2 Long-Horizon Planning & Instruction Following

## The Problem

Every web agent — Claude for Chrome, OpenAI Operator, Google Mariner — fails the moment a site updates its layout, renames a button, or shuffles its form fields. Existing benchmarks (WebArena, MiniWob++) use fixed DOMs that agents memorize instead of understand. There is no training environment that exposes agents to the structural variance they'll face in the real world.

## What It Does

Each episode spins up a **fresh, live, interactive web page** served by a real Chromium browser. The task is always the same — "book a train ticket" — but the page is procedurally regenerated with randomized:

- **Labels:** "Search" → "Find Trains" → "Check Availability" → "Look Up Trains"
- **Layout:** top nav, sidebar, hamburger menu, single column
- **Field order:** origin/destination/class shuffled
- **CSS classes:** random prefixes prevent selector memorization
- **Distractors:** cookie banners, promo modals, misleading ARIA labels

The agent observes an **accessibility tree** (not raw HTML) and emits JSON actions (`click`, `type`, `scroll`). Rewards come from a **semantic task graph** — the agent must complete checkpoints (enter origin, select class, submit search, confirm booking) regardless of how they're presented.

This is a textual match for the **Patronus AI "Consumer Workflows with Schema Drift"** sub-theme: a multi-step consumer workflow where the underlying data schemas, UI contracts, and presentation rules change between episodes.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenEnv Server (FastAPI)              │
│  POST /reset  ─► InfiniteDOMEnvironment.reset()         │
│  POST /step   ─► InfiniteDOMEnvironment.step()          │
│  GET  /state  ─► DOMState (for grading)                 │
│  WS   /ws     ─► persistent session                     │
└─────────┬──────────────────────────────────┬────────────┘
          │                                  │
          ▼                                  ▼
┌──────────────────┐              ┌──────────────────────┐
│  DOM Generator   │              │   Playwright Driver   │
│                  │              │                       │
│ Jinja2 template  │──── HTML ──►│ Chromium (headless)   │
│ + Alpine.js      │              │ CDP a11y tree         │
│ + VarianceProfile│              │ JS state extraction   │
└──────────────────┘              └───────────┬──────────┘
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │   Task Graph +    │
                                   │   Reward Calc +   │
                                   │   Graders         │
                                   └──────────────────┘
```

## Task Curriculum

| Task | Name | Difficulty | What Changes |
|------|------|-----------|-------------|
| 1 | Clean Form | Easy | Nothing — standard labels, layout, ARIA |
| 2 | Label Drift | Medium | Button/field labels randomized from synonym pools |
| 3 | Structural Drift | Hard | + Layout and field order randomized |
| 4 | Full Chaos | Expert | + Cookie banners, promo modals, misleading ARIA |

## Reward Design

Dense, multi-component reward signal:

- **Progression** (+0.15–0.35): each task-graph node completed for the first time
- **Step penalty** (−0.01): prevents wandering
- **Invalid action penalty** (−0.05): teaches action-space discipline
- **Completion bonus** (+1.0): full task completion
- **Anti-thrash penalty** (−0.2): repeated identical failed actions

Grader scores are strictly in (0.01, 0.99) per OpenEnv requirements.

## Status (Prep-Days Build)

- [x] OpenEnv-compliant environment (Pydantic models, env class, FastAPI server)
- [x] Booking-flow archetype with 4-level variance curriculum
- [x] Playwright-driven real browser with CDP accessibility tree extraction
- [x] Semantic task graph with 5 checkpoints per episode
- [x] Oracle solver passing on Task 1 (5/5 seeds, ≥3 nodes completed)
- [x] SFT data generation pipeline (250+ observation-action pairs)
- [x] 73 tests passing (62 unit + 11 browser/e2e)
- [ ] Training runs with Unsloth/TRL (hackathon days — compute credits needed)
- [ ] HuggingFace Space deployment (hackathon days)
- [ ] WebArena transfer evaluation (stretch)
- [ ] Additional task archetypes: search-filter-select, modal-nested (stretch)

## Quickstart

```bash
# Clone and setup
git clone <repo-url> && cd infinite-dom
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
playwright install chromium

# Run tests
pytest tests/ -v -m "not browser"     # unit tests (no browser needed)
pytest tests/ -v --timeout=120        # all tests including browser

# Start the server
PYTHONPATH=. uvicorn infinite_dom.server.app:app --host 0.0.0.0 --port 8000

# In another terminal — interact
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"seed": 42, "task_id": 1}'
```

## Generate Training Data

```bash
PYTHONPATH=. python training/generate_oracle_data.py 100
# Writes observation-action pairs to training/data/oracle_trajectories.jsonl
```

## Key Files

| File | Purpose |
|------|---------|
| `infinite_dom/models.py` | OpenEnv-compliant Pydantic models (Action, Observation, State) |
| `infinite_dom/task_graph.py` | Semantic task graph with predicate-based completion |
| `infinite_dom/reward_calculator.py` | Dense multi-component reward function |
| `infinite_dom/graders.py` | Per-episode scoring in (0.01, 0.99) |
| `infinite_dom/generator/dom_generator.py` | Procedural HTML generation with Jinja2 |
| `infinite_dom/generator/variance.py` | Label/layout/distractor variance pools |
| `infinite_dom/browser/playwright_driver.py` | Chromium automation + CDP a11y tree |
| `infinite_dom/environment/infinite_dom_env.py` | OpenEnv Environment class |
| `infinite_dom/oracle/booking_flow_oracle.py` | Hand-written solver for SFT data |
| `inference.py` | OpenEnv evaluation entrypoint |

## Why This Matters

Web agents today are brittle. They memorize CSS selectors and DOM paths instead of understanding what a "search button" or "origin field" is semantically. The Infinite DOM forces agents to develop genuine semantic understanding by making every episode structurally unique while keeping the task semantics constant. If an agent can book a ticket across 1000 different page layouts, it can handle the real web.

## License

MIT
