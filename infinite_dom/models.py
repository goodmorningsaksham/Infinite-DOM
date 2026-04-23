"""
Pydantic models for OpenEnv compliance.

These are the contracts between the agent, the environment server, and graders.
The OpenEnv base classes already define certain fields — do not redeclare.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        from openenv.types import Action, Observation, State
    except ImportError:
        from openenv_core.types import Action, Observation, State  # type: ignore


class ActionType(str, Enum):
    """Valid action types the agent can emit."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    BACK = "back"


class DOMAction(Action):
    """
    What the agent controls in a single step.

    Actions reference DOM elements by the short `element_ref` identifiers
    (e.g., 'btn_7', 'inp_2') that appear in the accessibility tree
    observation — NOT by CSS selectors or DOM paths. This forces the
    agent to reason from what it currently sees, not from memorized
    selectors.
    """
    action_type: ActionType = Field(
        description="The kind of action to execute."
    )
    element_ref: str = Field(
        default="",
        description="ID of target element as shown in the a11y tree observation. Required for click, type, back.",
    )
    text_value: str = Field(
        default="",
        description="Text to type. Only used when action_type=type.",
    )
    scroll_delta: int = Field(
        default=0,
        ge=-2000,
        le=2000,
        description="Pixels to scroll vertically. Only used when action_type=scroll.",
    )


class DOMObservation(Observation):
    """
    What the agent sees after a step.

    Inherits from OpenEnv's Observation base:
      - done: bool
      - reward: float | None
      - metadata: dict[str, Any]

    Do not redeclare those here.
    """
    model_config = ConfigDict(strict=True)
    a11y_tree: str = Field(
        description="Accessibility tree serialized as indented text with element refs.",
    )
    task_instruction: str = Field(
        description="Natural language goal the agent must accomplish.",
    )
    task_progress: list[str] = Field(
        default_factory=list,
        description="List of task-graph node IDs completed so far.",
    )
    last_action_repr: str = Field(
        default="",
        description="String representation of the agent's previous action.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="How many steps have elapsed in this episode.",
    )
    page_url: str = Field(
        default="",
        description="Current browser URL. For debugging only; not intended as policy input.",
    )


class DOMState(State):
    """
    Full episode state used for grading.

    Inherits from OpenEnv's State base:
      - episode_id: str
      - step_count: int

    Do not redeclare those here.
    """
    task_id: int = Field(description="Curriculum task (1–4).")
    template_id: str = Field(description="Which archetype template generated this episode.")
    seed: int = Field(description="Random seed used for generation.")
    task_graph_completed: list[str] = Field(default_factory=list)
    task_graph_total: list[str] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    episode_reward: float = Field(default=0.0)
    failed_action_count: int = Field(default=0)
    is_generalization_episode: bool = Field(
        default=False,
        description="True if the template was held out of training.",
    )
    terminated_reason: str = Field(
        default="",
        description="Empty while running; set on termination: 'success' | 'timeout' | 'stuck'",
    )
