"""Phase 3 tests — reward function."""
from infinite_dom.reward_calculator import (
    COMPLETION_BONUS,
    INVALID_ACTION_PENALTY,
    STEP_PENALTY,
    THRASH_PENALTY,
    compute_reward,
)

NODE_WEIGHTS = {
    "origin_set": 0.15,
    "destination_set": 0.15,
    "class_selected": 0.15,
    "search_submitted": 0.2,
    "booking_confirmed": 0.35,
}


class TestCompute:
    def test_no_progress_successful_action(self):
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.progression == 0.0
        assert r.step == STEP_PENALTY
        assert r.invalid == 0.0
        assert r.total == STEP_PENALTY

    def test_one_node_progress(self):
        r = compute_reward(
            newly_completed_nodes=["origin_set"],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.progression == 0.15
        assert r.total == 0.15 + STEP_PENALTY

    def test_invalid_action_adds_penalty(self):
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=[],
        )
        assert r.invalid == INVALID_ACTION_PENALTY

    def test_episode_completion_bonus(self):
        r = compute_reward(
            newly_completed_nodes=["booking_confirmed"],
            node_weights=NODE_WEIGHTS,
            action_succeeded=True,
            is_episode_complete=True,
            action_history=[],
        )
        assert r.completion == COMPLETION_BONUS
        assert r.total == 0.35 + STEP_PENALTY + COMPLETION_BONUS

    def test_thrash_detection(self):
        failed = {"action_type": "click", "element_ref": "btn_x", "succeeded": False}
        history = [failed, failed, failed]
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=history,
        )
        assert r.thrash == THRASH_PENALTY

    def test_no_thrash_when_actions_differ(self):
        a1 = {"action_type": "click", "element_ref": "btn_1", "succeeded": False}
        a2 = {"action_type": "click", "element_ref": "btn_2", "succeeded": False}
        a3 = {"action_type": "click", "element_ref": "btn_3", "succeeded": False}
        r = compute_reward(
            newly_completed_nodes=[],
            node_weights=NODE_WEIGHTS,
            action_succeeded=False,
            is_episode_complete=False,
            action_history=[a1, a2, a3],
        )
        assert r.thrash == 0.0


class TestAntiGaming:
    """Pathological policies should score low."""

    def test_random_policy_mostly_negative(self):
        total = 0.0
        history = []
        for i in range(20):
            history.append({"action_type": "click", "element_ref": f"r{i}", "succeeded": False})
            r = compute_reward(
                newly_completed_nodes=[],
                node_weights=NODE_WEIGHTS,
                action_succeeded=False,
                is_episode_complete=False,
                action_history=history,
            )
            total += r.total
        assert total < 0, f"Random failing policy should accumulate negative reward, got {total}"
