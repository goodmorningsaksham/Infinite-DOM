"""Phase 6 — full end-to-end episode test."""
import pytest

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment, _run_async
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.models import ActionType, DOMAction


@pytest.mark.e2e
@pytest.mark.browser
def test_reset_returns_valid_observation():
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=101)
        assert obs.a11y_tree
        assert obs.task_instruction
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.step_count == 0
    finally:
        _run_async(env.shutdown())
        _run_async(stop_page_server())


@pytest.mark.e2e
@pytest.mark.browser
def test_one_step_produces_feedback():
    env = InfiniteDOMEnvironment()
    try:
        obs = env.reset(task_id=1, seed=101)
        ref_id = None
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
        assert -0.1 <= obs2.reward <= 1.0
    finally:
        _run_async(env.shutdown())
        _run_async(stop_page_server())


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
        _run_async(env.shutdown())
        _run_async(stop_page_server())
