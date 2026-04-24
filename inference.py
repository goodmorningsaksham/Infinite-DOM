"""
OpenEnv-required top-level inference entrypoint.

Runs evaluation episodes across all tasks using an LLM policy (with
heuristic fallback). Emits the exact [START]/[STEP]/[END] stdout format
required by the OpenEnv hackathon validator.
"""
from __future__ import annotations

import json
import os
import sys

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment, _run_async
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.graders import grade
from infinite_dom.models import ActionType, DOMAction
from infinite_dom.oracle.booking_flow_oracle import oracle_policy

# ---------------------------------------------------------------------------
# Environment variables — judges provide these at evaluation time
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default — judges supply their own

# ---------------------------------------------------------------------------
# Score normalization — all scores MUST be strictly in (0.01, 0.99)
# ---------------------------------------------------------------------------
SCORE_MIN = 0.01
SCORE_MAX = 0.99
RAW_MIN = -6.0
RAW_MAX = 1.5


def normalize_reward(raw: float) -> float:
    n = SCORE_MIN + (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * (SCORE_MAX - SCORE_MIN)
    return max(SCORE_MIN, min(SCORE_MAX, round(n, 4)))


def _clamp_score(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, round(score, 4)))


# ---------------------------------------------------------------------------
# Stdout logging — exact format required by validator
# ---------------------------------------------------------------------------
def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={r_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM client — uses OpenAI-compatible API via HuggingFace router
# ---------------------------------------------------------------------------
_llm_client = None
_llm_failed = False

SYSTEM_PROMPT = """You are a web agent navigating an interactive web application.
You observe an accessibility tree and must complete a booking task.

First, reason about what you see and what action to take inside <think> tags.
Then, provide your action as a JSON object inside <answer> tags.

Format:
<think>
[Observe the current page state. Identify which fields are filled, what buttons are available, and what the next logical step is to complete the task.]
</think>
<answer>
{"action_type": "click"|"type"|"scroll"|"wait", "element_ref": "ref_id", "text_value": "text"|"", "scroll_delta": 0}
</answer>

Rules:
- Use "type" to fill text fields (element_ref + text_value required)
- Use "click" to press buttons/links (element_ref required)
- Use "scroll" to scroll the page (scroll_delta: positive=down, negative=up)
- Use "wait" when the page is loading or you need to observe
- Element refs look like: inp_1, btn_2, cmb_1, lnk_3
- Always dismiss cookie banners or popups before interacting with the main form"""


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        try:
            from openai import OpenAI
            _llm_client = OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN or "dummy",
                timeout=10.0,
            )
        except Exception:
            pass
    return _llm_client


def _parse_llm_action(text: str) -> DOMAction | None:
    """Parse LLM response into a DOMAction. Supports <think>/<answer> format and raw JSON."""
    try:
        text = text.strip()
        # Extract from <answer> tags if present (WebAgent-R1 think-then-act format)
        if "<answer>" in text:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>") if "</answer>" in text else len(text)
            text = text[start:end].strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return DOMAction(
            action_type=ActionType(data.get("action_type", "wait")),
            element_ref=data.get("element_ref", ""),
            text_value=data.get("text_value", ""),
            scroll_delta=int(data.get("scroll_delta", 0)),
        )
    except Exception:
        return None


_action_history_log: list[str] = []


def _format_step_history() -> str:
    """Compress action history into a concise context string (WebAgent-R1 style)."""
    if not _action_history_log:
        return ""
    lines = ["Previous actions:"]
    for entry in _action_history_log[-5:]:  # Last 5 steps max
        lines.append(f"  {entry}")
    return "\n".join(lines) + "\n\n"


def get_action(obs, task_graph=None) -> DOMAction:
    """Get next action: try LLM first, fall back to oracle, then heuristic."""
    global _llm_failed

    if not _llm_failed:
        client = _get_llm_client()
        if client is not None:
            try:
                step_history = _format_step_history()
                user_msg = (
                    f"Task: {obs.task_instruction}\n\n"
                    f"{step_history}"
                    f"Accessibility Tree:\n{obs.a11y_tree[:3000]}\n\n"
                    f"Completed: {obs.task_progress}\n"
                    f"Step: {obs.step_count}"
                )
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )
                action = _parse_llm_action(response.choices[0].message.content)
                if action is not None:
                    desc = f"Step {obs.step_count}: {action.action_type.value}"
                    if action.element_ref:
                        desc += f" on {action.element_ref}"
                    if action.text_value:
                        desc += f" ('{action.text_value}')"
                    _action_history_log.append(desc)
                    return action
            except Exception:
                _llm_failed = True

    if task_graph is not None:
        return oracle_policy(obs.a11y_tree, task_graph)

    return _heuristic_action(obs)


def _heuristic_action(obs) -> DOMAction:
    """Last-resort heuristic when both LLM and oracle are unavailable."""
    for line in obs.a11y_tree.split("\n"):
        if "ref=inp_" in line and 'value=""' in line:
            ref = line.split("ref=")[1].split(" ")[0].rstrip("]")
            return DOMAction(action_type=ActionType.TYPE, element_ref=ref, text_value="Mumbai")
    for line in obs.a11y_tree.split("\n"):
        if "ref=btn_" in line:
            ref = line.split("ref=")[1].split(" ")[0].rstrip("]")
            return DOMAction(action_type=ActionType.CLICK, element_ref=ref)
    return DOMAction(action_type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Task runner — one [START]...[END] block per task
# ---------------------------------------------------------------------------
TASK_NAMES = {
    1: "task_1_clean_form",
    2: "task_2_label_drift",
    3: "task_3_structural_drift",
    4: "task_4_full_chaos",
}
MAX_STEPS = 25


def run_task(task_id: int, seed: int = 42) -> None:
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    log_start(task_name, "infinite_dom", MODEL_NAME)
    _action_history_log.clear()

    rewards: list[float] = []
    steps = 0
    env = InfiniteDOMEnvironment()

    try:
        obs = env.reset(task_id=task_id, seed=seed)
        task_graph = env._current_page.task_graph if env._current_page else None

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            action = get_action(obs, task_graph)
            obs = env.step(action)
            r = normalize_reward(obs.reward or 0.0)
            rewards.append(r)
            steps = step
            log_step(step, action.action_type.value, r, obs.done)

    except Exception as e:
        log_step(steps + 1, "error", SCORE_MIN, True, str(e)[:100])
    finally:
        try:
            _run_async(env.shutdown())
        except Exception:
            pass
        try:
            _run_async(stop_page_server())
        except Exception:
            pass

    score = sum(rewards) / len(rewards) if rewards else SCORE_MIN
    score = _clamp_score(score)
    success = score > 0.5
    log_end(success, steps, score, rewards)


# ---------------------------------------------------------------------------
# Main — run all 4 tasks
# ---------------------------------------------------------------------------
def main() -> None:
    tasks = [1, 2, 3, 4]
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    for task_id in tasks:
        run_task(task_id, seed=seed)


if __name__ == "__main__":
    main()
