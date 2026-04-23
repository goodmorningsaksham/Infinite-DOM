# CLAUDE.md — Infinite DOM Autonomous Build

> **You are Claude Code. This file is the entry point. The detailed build plan is split across phase files in `build/` — read each one ONLY when you begin that phase. This keeps context lean. Do not read all phase files upfront.**

---

## 0. Meta-Instructions (Read Once, Apply Always)

### 0.1 Execution Model

You will execute **seven build phases** in strict sequential order, with an optional eighth. For each phase:

1. Read `build/PHASE_<N>.md` using the Read tool at the start of the phase
2. Execute every step in the phase file
3. Run the phase's milestone gate (an automated test)
4. Only advance when the gate passes
5. Append an entry to `BUILD_LOG.md` summarizing what you did

**Do not read ahead.** Do not read `build/PHASE_3.md` while in Phase 1. Context budget is limited; reading phases on demand is how this document is designed to work.

### 0.2 Anti-Shortcut Rules — NON-NEGOTIABLE

You have a tendency toward these shortcuts. Do not take them:

1. **Do not stub functions with `pass`.** Every function must be implemented. If you genuinely cannot implement something, raise `NotImplementedError("REASON: <why>")` so it is visible.
2. **Do not write placeholder templates.** HTML templates must produce real, interactive, working pages.
3. **Do not skip or disable tests.** Each phase ends with a pytest run. Fix failures. Do not comment out assertions.
4. **Do not invent library APIs.** When uncertain, consult `build/REFERENCE.md` for verified patterns or write a small verification script.
5. **Do not re-architect.** The architecture is fixed. If you disagree, note it in `BUILD_LOG.md` under "Deviations" and continue following the spec.
6. **Do not claim a gate passed without running it.** Run the test. Copy the actual output into `BUILD_LOG.md`.
7. **Do not mark incomplete work as "done".** If something is partial, mark it explicitly as "PARTIAL — reason".

### 0.3 BUILD_LOG.md — Your Audit Trail

Create `BUILD_LOG.md` at the project root during Phase 1. After each phase, append:

```
## Phase N — <phase name>
- Started: <ISO timestamp>
- Completed: <ISO timestamp>
- Files created: <list>
- Files modified: <list>
- Tests run: <command(s)> — <pass/fail summary>
- Issues encountered: <list, or "none">
- Deviations from spec: <list, or "none">
- Gate status: PASS / FAIL
```

If blocked, add a `BLOCKED` section with the exact command and error and stop.

### 0.4 Environment Assumptions (Confirmed with User)

| Aspect | Value |
|--------|-------|
| OS | Windows 10/11 with WSL2 Ubuntu 22.04 |
| Python | 3.11 (install in a venv) |
| Node | Not required (Jinja2 + Alpine.js, no React build step) |
| HuggingFace | User has account, will supply `HF_TOKEN` in `.env` later |
| W&B | User has no account — use matplotlib fallback, gate W&B behind `WANDB_API_KEY` env var |
| Billing | Zero-cost path only. No paid services. Training happens during hackathon on provided credits. |

### 0.5 What You Are Building (One Paragraph)

The Infinite DOM is an OpenEnv-compliant reinforcement learning environment that procedurally generates live, interactive web applications on each episode. An LLM agent observes the accessibility tree of a generated page, emits JSON actions (click / type / scroll), and is rewarded for traversing a semantic task graph (for example, "book a ticket"). Each episode's DOM has randomized layout, labels, class names, and distractors while preserving task semantics — preventing the agent from memorizing structure and forcing it to learn semantic understanding. The environment exposes standard OpenEnv REST and WebSocket endpoints via FastAPI, runs a real headless Chromium via Playwright, and ships as a Docker container deployable to HuggingFace Spaces. This document covers the prep-days build: infrastructure plus one working task archetype (multi-step booking flow) end-to-end, ready for the 48-hour hackathon where training, deployment, and polish happen.

### 0.6 Available Skills

An `openenv-builder` skill is available under `.claude/skills/openenv-builder/`. Its `SKILL.md`, `ARCHITECTURE.md`, and `SUBMISSION.md` are authoritative references for OpenEnv conventions — data model rules, the three required methods, server factory semantics, grader score clamping, and HuggingFace Space deployment. If you encounter ambiguity about OpenEnv behavior during any phase (especially Phase 2 for models, Phase 6 for the environment class, and Phase 7 for the server and graders), consult this skill's reference files before guessing.

Do NOT invoke the `hackathon-expert` skill during the build. It is scoped to pitch, demo, and submission strategy work that happens after Phase 7 and after the prep build is complete.

### 0.7 Context Management

This build is long and detail-dense. Context can fill up, especially during the largest phases (Phase 4 — DOM generator, and Phase 7 — server + oracle + data generation). If you notice your available context shrinking to the point where it threatens the quality of work in the current phase, do one of the following:

1. Briefly summarize your progress and the outstanding checklist for the current phase
2. Recommend to the user that they run `/compact` to compress the conversation history
3. Then continue from where you left off

Do not silently degrade quality to fit context. Flag the issue and let the user decide. `/compact` is a user-invoked command, so you cannot call it yourself — you ask for it.

### 0.8 Hackathon Requirements Reference

A file named `minimum-requirements.md` (or similar — check the project root) contains the official OpenEnv Hackathon India 2026 themes, sub-themes, judging criteria, and submission requirements. It is authoritative for what this project must satisfy to be competitive.

Read it ONCE at the start of Phase 1 to understand the constraints. Do not re-read it every phase — it does not change. The key facts you need to internalize from it:

- **Primary theme claim:** #3.1 World Modeling — Professional Tasks
- **Primary bonus sub-theme:** Patronus AI "Consumer Workflows with Schema Drift" — the Infinite DOM is a textual match for this. Call it out explicitly in the `openenv.yaml` description and in the README (Phase 8).
- **Secondary theme:** #2 Long-Horizon Planning & Instruction Following
- **Minimum requirements that affect this build:**
  - Must use OpenEnv (latest release) — handled by the import fallbacks in §5
  - Training script using Unsloth or HF TRL in Colab — stub exists in `training/train_infinite_dom.ipynb`; completed during the hackathon itself
  - Hosted on HuggingFace Spaces — Dockerfile and `openenv.yaml` are prepared in prep, deployment happens during the hackathon
  - README with problem motivation, environment explanation, results, and links — Phase 8

If `minimum-requirements.md` is not present at the project root, note it in `BUILD_LOG.md` under Phase 1 and proceed — the constraints above are the bits that affect your build.

---

## 1. Target Repository Layout

By the end of all phases, the repository will look exactly like this. Use this as a reference when creating files — do not create files outside this structure without logging a deviation.

```
infinite-dom/
├── CLAUDE.md                        # This file (entry point)
├── minimum-requirements.md          # Hackathon themes + judging criteria (authoritative)
├── .claude/
│   └── skills/
│       ├── openenv-builder/         # Reference skill — consult when needed
│       │   ├── SKILL.md
│       │   ├── ARCHITECTURE.md
│       │   └── SUBMISSION.md
│       └── hackathon-expert/        # DO NOT invoke during build
│           └── hackathon-expert.md
├── build/                           # Phase specifications (user-provided)
│   ├── PHASE_1.md
│   ├── PHASE_2.md
│   ├── PHASE_3.md
│   ├── PHASE_4.md
│   ├── PHASE_5.md
│   ├── PHASE_6.md
│   ├── PHASE_7.md
│   ├── PHASE_8.md
│   └── REFERENCE.md
│
├── README.md                        # User-facing readme (Phase 8)
├── BUILD_LOG.md                     # Your audit trail
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
├── openenv.yaml
├── pyproject.toml
│
├── infinite_dom/
│   ├── __init__.py
│   ├── models.py
│   ├── config.py
│   ├── task_graph.py
│   ├── reward_calculator.py
│   ├── graders.py
│   │
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── dom_generator.py
│   │   ├── variance.py
│   │   ├── serve_html.py
│   │   └── templates/
│   │       ├── booking_flow.jinja
│   │       └── _base_styles.jinja
│   │
│   ├── browser/
│   │   ├── __init__.py
│   │   ├── playwright_driver.py
│   │   └── a11y_formatter.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   └── infinite_dom_env.py
│   │
│   ├── oracle/
│   │   ├── __init__.py
│   │   └── booking_flow_oracle.py
│   │
│   └── server/
│       ├── __init__.py
│       ├── app.py
│       └── dashboard.html
│
├── inference.py                     # Top-level OpenEnv entrypoint
├── client.py                        # EnvClient wrapper
│
├── training/
│   ├── __init__.py
│   ├── generate_oracle_data.py
│   ├── train_infinite_dom.ipynb
│   └── README.md
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_phase1_scaffold.py
│   ├── test_models.py
│   ├── test_task_graph.py
│   ├── test_reward.py
│   ├── test_graders.py
│   ├── test_generator.py
│   ├── test_a11y_formatter.py
│   ├── test_browser_driver.py
│   ├── test_oracle.py
│   └── test_environment_e2e.py
│
└── scripts/
    ├── setup_dev.sh
    ├── run_server.sh
    ├── smoke_test.py
    └── validate_openenv.sh
```

---

## 2. Phase Index

Work through these in order. For each phase, the first step is to read the corresponding `build/PHASE_<N>.md` file.

| Phase | Spec File | Goal | Hard Dependency |
|-------|-----------|------|-----------------|
| 1 | `build/PHASE_1.md` | Scaffold: directory tree, config files, dependencies installed, venv working | — |
| 2 | `build/PHASE_2.md` | Pydantic data models — `DOMAction`, `DOMObservation`, `DOMState` | Phase 1 |
| 3 | `build/PHASE_3.md` | Pure-logic layer: task graph, reward function, graders (no browser yet) | Phase 2 |
| 4 | `build/PHASE_4.md` | DOM generator: working booking_flow template producing valid HTML | Phase 3 |
| 5 | `build/PHASE_5.md` | Playwright browser driver + accessibility tree formatter | Phase 4 |
| 6 | `build/PHASE_6.md` | `InfiniteDOMEnvironment` class implementing OpenEnv `reset`/`step`/`state` | Phase 5 |
| 7 | `build/PHASE_7.md` | OpenEnv FastAPI server, oracle solver, SFT data generator, final validation | Phase 6 |
| 8 (optional) | `build/PHASE_8.md` | README + final polish, if time remains | Phase 7 |

Supporting reference material (read only when needed):
- `build/REFERENCE.md` — verified API patterns for Playwright, FastAPI, Pydantic, OpenEnv, Jinja2

---

## 3. Milestone Gate Rules

Every phase ends with at least one pytest command (the "gate"). The rules:

- **Run the gate test exactly as written in the phase file.** Do not modify it.
- **If the gate passes cleanly, record the output and advance.**
- **If the gate fails, debug the code, not the test.** The test encodes the specification — if the test is wrong, stop and note it in `BUILD_LOG.md`.
- **Do not comment out a failing test to advance.** That counts as a fake pass.
- **If you cannot fix it after reasonable effort, stop.** Write the exact command, exact error, and what you tried to `BUILD_LOG.md` under a `BLOCKED` heading.

---

## 4. Critical-Path vs Non-Critical-Path Items

If you hit an unsolvable blocker, these items cannot be skipped — they are critical path:

- OpenEnv imports work (any of the fallback paths in `build/REFERENCE.md`)
- Playwright Chromium launches successfully
- Generator produces renderable HTML
- `InfiniteDOMEnvironment.reset()` and `.step()` complete without crashing
- Oracle achieves at least 30% task completion on Task 1

These items may be marked `TODO` if blocked — they are non-critical path:

- Second and third task archetypes (beyond booking_flow)
- WebArena integration
- Full training notebook cells (stubs are acceptable for prep days)
- Advanced dashboard UI polish
- Docker image build optimization

---

## 5. Import Path Fallbacks (OpenEnv)

OpenEnv's exact import path may vary by version. In any file that imports from OpenEnv, use this pattern:

```python
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        from openenv.types import Action, Observation, State
    except ImportError:
        from openenv_core.types import Action, Observation, State  # type: ignore
```

Log in `BUILD_LOG.md` which of the three paths succeeded during Phase 1 installation. Use that path consistently across all files thereafter.

---

## 6. Failure Policy

When blocked:

1. Do NOT fake a pass.
2. Do NOT skip the phase.
3. Do NOT silently implement something else instead.

Instead:

1. Write the exact command and error to `BUILD_LOG.md` under `## BLOCKED`
2. List what you tried
3. Stop execution and wait for user guidance

Fake passes are detected easily by the user. Honest blockers are fixable.

---

## 7. Phase Completion Checklist (Tear-Off)

After each phase, verify:

- [ ] All files listed in the phase spec exist
- [ ] No `TODO` markers hiding missing functionality in critical-path code
- [ ] Milestone gate test passes (with real output logged)
- [ ] `BUILD_LOG.md` entry for this phase written
- [ ] No new lint errors introduced: `ruff check infinite_dom/ tests/`

Then, and only then, proceed to the next phase.

---

## 8. Hackathon-Day Follow-Up (Not in Scope)

After your autonomous build finishes, the user will complete these during the 48-hour hackathon itself:

- Run actual RL training on hackathon-provided compute
- Deploy environment to HuggingFace Space
- Record 2-minute demo video
- Add real training plots to README
- Execute live pitch including real-website closing demo

Your job ends when Phase 7 (and optionally Phase 8) is complete and all gates pass. You deliver a tested, OpenEnv-compliant environment the user can train against without debugging during the event.

---

**BEGIN AT PHASE 1.** Read `build/PHASE_1.md` now and execute it.