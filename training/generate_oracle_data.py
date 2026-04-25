"""
Run oracle against N episodes across all tasks, record (observation, action) pairs
as JSONL for SFT warmup training later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment, _run_async
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.oracle.booking_flow_oracle import oracle_policy


OUT_DIR = Path("training/data")


def run(num_episodes: int = 30, tasks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)) -> None:
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
                    while not obs.done and steps_in_ep < 35:
                        action = oracle_policy(obs.a11y_tree, env._current_page.task_graph)
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
        _run_async(env.shutdown())
        _run_async(stop_page_server())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tasks_arg = tuple(int(t) for t in sys.argv[2].split(",")) if len(sys.argv) > 2 else (1, 2, 3, 4, 5, 6, 7, 8)
    run(num_episodes=n, tasks=tasks_arg)
