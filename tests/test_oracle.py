"""Phase 7 — oracle solves the booking flow on Task 1."""
import pytest

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment, _run_async
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
            action = oracle_policy(obs.a11y_tree, env._current_page.task_graph)
            obs = env.step(action)
            done = obs.done
            i += 1
        assert len(env.state.task_graph_completed) >= 3, \
            f"seed={seed}: oracle only completed {env.state.task_graph_completed}"
        score = grade(1, env.state)
        assert score > 0.3
    finally:
        _run_async(env.shutdown())
        _run_async(stop_page_server())
