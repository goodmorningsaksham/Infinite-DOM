## PHASE 6 — Environment Class

### 6.1 Goal

Implement `InfiniteDOMEnvironment` — a concrete subclass of OpenEnv's `Environment` implementing `reset()`, `step()`, and the `state` property. End-to-end episode test passes.

### 6.2 Execution Checklist

#### 6.2.1 Create `infinite_dom/environment/infinite_dom_env.py`

```python
"""
The Infinite DOM environment — ties together generator, browser, task graph, rewards.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

# Import OpenEnv with fallbacks (mirror models.py strategy)
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    try:
        from openenv.interfaces import Environment
    except ImportError:
        from openenv_core.interfaces import Environment  # type: ignore

from infinite_dom.browser.playwright_driver import BrowserSnapshot, PlaywrightDriver
from infinite_dom.config import CONFIG
from infinite_dom.generator.dom_generator import DOMGenerator, GeneratedPage
from infinite_dom.models import DOMAction, DOMObservation, DOMState
from infinite_dom.reward_calculator import compute_reward

logger = logging.getLogger(__name__)


class InfiniteDOMEnvironment(Environment):
    """
    OpenEnv-compliant environment.

    Each reset() generates a new procedural page and re-navigates the browser.
    Each step() executes one action, updates task progress, and emits a reward.
    """

    def __init__(self):
        self._generator = DOMGenerator()
        self._driver = PlaywrightDriver()
        self._driver_started = False
        self._current_page: GeneratedPage | None = None
        self._state: DOMState | None = None

    # =====================================================================
    # Required OpenEnv methods
    # =====================================================================

    def reset(self, task_id: int = 1, seed: int | None = None, **kwargs) -> DOMObservation:
        return asyncio.get_event_loop().run_until_complete(self._reset_async(task_id, seed))

    def step(self, action: DOMAction, **kwargs) -> DOMObservation:
        return asyncio.get_event_loop().run_until_complete(self._step_async(action))

    @property
    def state(self) -> DOMState:
        if self._state is None:
            raise RuntimeError("state accessed before reset()")
        return self._state

    # =====================================================================
    # Async implementations
    # =====================================================================

    async def _reset_async(self, task_id: int, seed: int | None) -> DOMObservation:
        if not self._driver_started:
            await self._driver.start()
            self._driver_started = True

        page = self._generator.generate(task_id=task_id, seed=seed)
        self._current_page = page

        self._state = DOMState(
            episode_id=f"ep_{uuid.uuid4().hex[:12]}",
            step_count=0,
            task_id=task_id,
            template_id=page.task_graph.template_id,
            seed=page.seed,
            task_graph_total=page.task_graph.all_node_ids,
            task_graph_completed=[],
            action_history=[],
            episode_reward=0.0,
            failed_action_count=0,
        )

        await self._driver.load_page(page.generation_id, page.html)
        snap = await self._driver.snapshot()

        # Check any pre-completed nodes (should be none, but compute defensively)
        completed = page.task_graph.get_completed_nodes(snap.state_dict)
        self._state.task_graph_completed = completed

        return DOMObservation(
            a11y_tree=snap.a11y.text,
            task_instruction=page.task_graph.instruction,
            task_progress=completed,
            step_count=0,
            last_action_repr="",
            page_url=snap.url,
            done=False,
            reward=0.0,
            metadata={
                "template_id": page.task_graph.template_id,
                "task_id": task_id,
                "seed": page.seed,
            },
        )

    async def _step_async(self, action: DOMAction) -> DOMObservation:
        assert self._state is not None and self._current_page is not None

        prev_completed = set(self._state.task_graph_completed)

        # Execute action
        action_succeeded = await self._driver.execute(action)

        # Record action in history
        self._state.action_history.append({
            "action_type": action.action_type.value,
            "element_ref": action.element_ref,
            "text_value": action.text_value,
            "scroll_delta": action.scroll_delta,
            "succeeded": action_succeeded,
        })
        if not action_succeeded:
            self._state.failed_action_count += 1

        # New observation
        snap = await self._driver.snapshot()
        new_completed = self._current_page.task_graph.get_completed_nodes(snap.state_dict)
        newly_done = [n for n in new_completed if n not in prev_completed]

        self._state.task_graph_completed = new_completed
        self._state.step_count += 1

        # Termination
        is_complete = set(new_completed) == set(self._state.task_graph_total)
        timeout = self._state.step_count >= CONFIG.max_steps
        stuck = self._state.failed_action_count >= CONFIG.max_failed_actions

        done = is_complete or timeout or stuck
        if done:
            if is_complete:
                self._state.terminated_reason = "success"
            elif timeout:
                self._state.terminated_reason = "timeout"
            else:
                self._state.terminated_reason = "stuck"

        # Reward
        breakdown = compute_reward(
            newly_completed_nodes=newly_done,
            node_weights=self._current_page.task_graph.node_weights,
            action_succeeded=action_succeeded,
            is_episode_complete=is_complete,
            action_history=self._state.action_history,
        )
        self._state.episode_reward += breakdown.total

        return DOMObservation(
            a11y_tree=snap.a11y.text,
            task_instruction=self._current_page.task_graph.instruction,
            task_progress=new_completed,
            step_count=self._state.step_count,
            last_action_repr=str(action.model_dump()),
            page_url=snap.url,
            done=done,
            reward=breakdown.total,
            metadata={
                "reward_breakdown": breakdown.as_dict(),
                "newly_completed_nodes": newly_done,
                "action_succeeded": action_succeeded,
                "terminated_reason": self._state.terminated_reason,
            },
        )

    # =====================================================================
    # Teardown
    # =====================================================================

    async def shutdown(self) -> None:
        if self._driver_started:
            await self._driver.close()
            self._driver_started = False
```

### 6.3 Tests

Create `tests/test_environment_e2e.py`:

```python
"""Phase 6 — full end-to-end episode test."""
import asyncio

import pytest

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.models import ActionType, DOMAction


@pytest.mark.e2e
@pytest.mark.browser
def test_reset_returns_valid_observation():
    """Loop reuse: run the async work inside a fresh event loop via .reset()."""
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=101)
        assert obs.a11y_tree
        assert obs.task_instruction
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.step_count == 0
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())


@pytest.mark.e2e
@pytest.mark.browser
def test_one_step_produces_feedback():
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=101)
        # Find a textbox to type into
        ref_id = None
        # Parse the a11y_tree and pull first inp_
        for line in obs.a11y_tree.split("\n"):
            if "ref=inp_" in line:
                ref_id = line.split("ref=")[1].split(" ")[0].rstrip("]")
                break
        assert ref_id is not None

        obs2 = env.step(DOMAction(
            action_type=ActionType.TYPE,
            element_ref=ref_id,
            text_value="Mumbai",
        ))
        assert obs2.step_count == 1
        # Either the step was valid and reward is ~step penalty, or invalid and ~-0.06
        assert -0.1 <= obs2.reward <= 1.0  # loose — we just want feedback
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())


@pytest.mark.e2e
@pytest.mark.browser
def test_timeout_termination():
    """Smash through max_steps with null actions — episode must terminate."""
    from infinite_dom.config import CONFIG

    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=55)
        done = False
        for _ in range(CONFIG.max_steps + 1):
            obs = env.step(DOMAction(action_type=ActionType.WAIT))
            if obs.done:
                done = True
                break
        assert done
        assert env.state.terminated_reason in {"timeout", "stuck"}
    finally:
        asyncio.get_event_loop().run_until_complete(env.shutdown())
        asyncio.get_event_loop().run_until_complete(stop_page_server())
```

### 6.4 Milestone Gate 6

Run: `pytest tests/test_environment_e2e.py -v -m "e2e and browser"`

**GATE CRITERIA:** All three e2e tests pass.

---

