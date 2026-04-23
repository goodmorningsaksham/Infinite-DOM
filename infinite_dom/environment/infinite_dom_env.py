"""
The Infinite DOM environment — ties together generator, browser, task graph, rewards.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    try:
        from openenv.interfaces import Environment
    except ImportError:
        from openenv_core.interfaces import Environment  # type: ignore

from infinite_dom.browser.playwright_driver import PlaywrightDriver
from infinite_dom.config import CONFIG
from infinite_dom.generator.dom_generator import DOMGenerator, GeneratedPage
from infinite_dom.models import DOMAction, DOMObservation, DOMState
from infinite_dom.reward_calculator import compute_reward

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


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

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> DOMObservation:
        task_id = kwargs.get("task_id", 1)
        return _run_async(self._reset_async(task_id, seed))

    def step(self, action: DOMAction, timeout_s: float | None = None, **kwargs) -> DOMObservation:
        return _run_async(self._step_async(action))

    @property
    def state(self) -> DOMState:
        if self._state is None:
            raise RuntimeError("state accessed before reset()")
        return self._state

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shutdown())
        except RuntimeError:
            _run_async(self.shutdown())

    async def reset_async(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> DOMObservation:
        task_id = kwargs.get("task_id", 1)
        return await self._reset_async(task_id, seed)

    async def step_async(self, action: DOMAction, timeout_s: float | None = None, **kwargs) -> DOMObservation:
        return await self._step_async(action)

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

        action_succeeded = await self._driver.execute(action)

        self._state.action_history.append({
            "action_type": action.action_type.value,
            "element_ref": action.element_ref,
            "text_value": action.text_value,
            "scroll_delta": action.scroll_delta,
            "succeeded": action_succeeded,
        })
        if not action_succeeded:
            self._state.failed_action_count += 1

        snap = await self._driver.snapshot()
        new_completed = self._current_page.task_graph.get_completed_nodes(snap.state_dict)
        newly_done = [n for n in new_completed if n not in prev_completed]

        self._state.task_graph_completed = new_completed
        self._state.step_count += 1

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

    async def shutdown(self) -> None:
        if self._driver_started:
            await self._driver.close()
            self._driver_started = False
