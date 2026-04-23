## PHASE 2 — Data Models

### 2.1 Goal

All three Pydantic models (`DOMAction`, `DOMObservation`, `DOMState`) implemented correctly. Constants/enums defined. Validation rules correct. Tests passing.

### 2.2 Key Reference: OpenEnv Model Rules

From the OpenEnv spec, your three models inherit from base classes that already provide certain fields. DO NOT REDECLARE these fields:

- `Observation` base class already has: `done: bool`, `reward: float | None`, `metadata: dict`
- `State` base class already has: `episode_id: str`, `step_count: int`
- `Action` base class has no pre-set fields you'd collide with, but has serialization hooks

If the OpenEnv import path differs in the installed version, adapt but preserve the inheritance structure. Valid import patterns to try:

```python
# Try in order, use the first that works:
from openenv.core.env_server.types import Action, Observation, State
from openenv.types import Action, Observation, State
from openenv_core.types import Action, Observation, State
```

Log which import path worked in BUILD_LOG.md.

### 2.3 Execution Checklist

#### 2.3.1 Create `infinite_dom/config.py`

```python
"""Runtime configuration pulled from environment variables with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    host: str = os.environ.get("INFINITE_DOM_HOST", "0.0.0.0")
    port: int = int(os.environ.get("INFINITE_DOM_PORT", "8000"))
    page_server_port: int = int(os.environ.get("INFINITE_DOM_PAGE_SERVER_PORT", "9000"))
    max_steps: int = int(os.environ.get("INFINITE_DOM_MAX_STEPS", "25"))
    playwright_headless: bool = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
    max_failed_actions: int = 5
    a11y_tree_max_tokens: int = 1500
    templates_dir: Path = Path(__file__).parent / "generator" / "templates"


CONFIG = Config()
```

#### 2.3.2 Create `infinite_dom/models.py`

```python
"""
Pydantic models for OpenEnv compliance.

These are the contracts between the agent, the environment server, and graders.
The OpenEnv base classes already define certain fields — do not redeclare.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

# Import OpenEnv base classes with fallback paths
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
```

### 2.4 Tests

Create `tests/test_models.py`:

```python
"""Phase 2 tests — data model correctness."""
import pytest
from pydantic import ValidationError

from infinite_dom.models import ActionType, DOMAction, DOMObservation, DOMState


class TestDOMAction:
    def test_valid_click(self):
        a = DOMAction(action_type=ActionType.CLICK, element_ref="btn_3")
        assert a.action_type == ActionType.CLICK
        assert a.element_ref == "btn_3"
        assert a.text_value == ""
        assert a.scroll_delta == 0

    def test_valid_type(self):
        a = DOMAction(
            action_type=ActionType.TYPE,
            element_ref="inp_1",
            text_value="Bengaluru",
        )
        assert a.text_value == "Bengaluru"

    def test_valid_scroll(self):
        a = DOMAction(action_type=ActionType.SCROLL, scroll_delta=500)
        assert a.scroll_delta == 500

    def test_scroll_delta_bounds(self):
        with pytest.raises(ValidationError):
            DOMAction(action_type=ActionType.SCROLL, scroll_delta=3000)

    def test_action_type_rejects_invalid_string(self):
        with pytest.raises(ValidationError):
            DOMAction(action_type="not_a_real_action")  # type: ignore

    def test_serializes_to_json(self):
        a = DOMAction(action_type=ActionType.CLICK, element_ref="btn_3")
        j = a.model_dump_json()
        assert "click" in j
        assert "btn_3" in j


class TestDOMObservation:
    def test_minimum_fields(self):
        o = DOMObservation(
            a11y_tree="[role=main]",
            task_instruction="Do a thing",
            done=False,
            reward=0.0,
        )
        assert o.a11y_tree == "[role=main]"
        assert o.task_instruction == "Do a thing"
        assert o.done is False
        assert o.reward == 0.0
        assert o.task_progress == []

    def test_done_must_be_bool(self):
        with pytest.raises(ValidationError):
            DOMObservation(
                a11y_tree="x",
                task_instruction="x",
                done="yes",  # type: ignore
                reward=0.0,
            )

    def test_step_count_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            DOMObservation(
                a11y_tree="x",
                task_instruction="x",
                done=False,
                reward=0.0,
                step_count=-1,
            )


class TestDOMState:
    def test_minimum_fields(self):
        s = DOMState(
            episode_id="ep_123",
            step_count=0,
            task_id=1,
            template_id="booking_flow",
            seed=42,
        )
        assert s.task_id == 1
        assert s.task_graph_completed == []
        assert s.failed_action_count == 0
        assert s.terminated_reason == ""
```

Run: `pytest tests/test_models.py -v`

### 2.5 Milestone Gate 2

All tests in `tests/test_models.py` pass. Log the result.

---

