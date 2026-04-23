---
name: openenv-builder
description: Build production-grade OpenEnv reinforcement learning environments from scratch. Use this skill whenever the user wants to create an RL environment, build an OpenEnv-compatible project, design observation/action spaces, implement reward functions, create task curricula, wrap a simulator in OpenEnv, build graders, prepare for OpenEnv hackathon submission, or debug OpenEnv environment issues. Also trigger when the user mentions OpenEnv, RL environments, simulated environments, env_server, create_app, step/reset/state patterns, reward shaping, task graders, or HuggingFace Spaces deployment of environments. If the user says "build an environment", "create tasks", "design rewards", "wrap my simulator", "OpenEnv hackathon", or "submission checklist", use this skill.
---

# OpenEnv Environment Builder

You are an expert at building production-grade reinforcement learning environments using the OpenEnv framework. You understand the complete architecture from simulator wrapping through HuggingFace Spaces deployment and hackathon submission.

## When to Read Reference Files

This skill has two reference files in the `references/` directory:

- **Read `references/ARCHITECTURE.md`** when: building a new environment from scratch, setting up file structure, implementing models/environment/server/graders, or debugging import/routing issues. This is the complete file-by-file blueprint.

- **Read `references/SUBMISSION.md`** when: preparing for hackathon submission, fixing validator errors, writing inference.py, debugging score range issues, deploying to HuggingFace Spaces, or running pre-validation checks.

---

## Core Concepts

### What Is an OpenEnv Environment?

An OpenEnv environment is a simulation that an agent interacts with in a loop:

```
Agent observes state → picks action → Environment processes action →
returns new observation + reward → repeat until done
```

OpenEnv standardizes this loop so any OpenEnv-compatible agent works with any OpenEnv environment. The framework provides:

- **Base classes** for data models (Action, Observation, State)
- **A server factory** (`create_app()`) that auto-generates API endpoints
- **A WebSocket protocol** for persistent agent-environment sessions
- **A client library** (`EnvClient`) for agents to connect

You build the simulation logic. OpenEnv handles the communication plumbing.

### The Three Required Methods

Every OpenEnv environment class must implement exactly three things:

```python
from openenv.core.env_server.interfaces import Environment

class MyEnvironment(Environment):
    
    def reset(self, task_id=1, seed=None, **kwargs):
        """Initialize a new episode. Returns first Observation."""
        # Create/reset your simulator
        # Return initial observation
    
    def step(self, action, **kwargs):
        """Process one action. Returns new Observation with reward and done."""
        # Feed action to simulator
        # Get new state
        # Compute reward
        # Check if episode is done
        # Return observation
    
    @property
    def state(self):
        """Return full episode state for grading. Returns State object."""
        # Return complete episode history and metrics
```

### The Three Required Models

Data flows through typed Pydantic models that inherit from OpenEnv base classes:

```python
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class MyAction(Action):
    """What the agent controls."""
    # Your action fields here
    
class MyObservation(Observation):
    """What the agent sees. Inherits: done (bool), reward (float|None), metadata (dict)"""
    # Your observation fields here

class MyState(State):
    """Full episode state for grading. Inherits: episode_id (str), step_count (int)"""
    # Your state fields here
```

**Critical rule**: The Observation base class already has `done`, `reward`, and `metadata` fields. Don't redeclare them — just add your domain-specific fields.

### The Server Factory

One line creates all API endpoints:

```python
from openenv.core.env_server.http_server import create_app

app = create_app(
    MyEnvironment,      # Your environment class
    MyAction,           # Your action model
    MyObservation,      # Your observation model  
    env_name="my_env",
    max_concurrent_envs=1,
)
```

This auto-generates: `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /schema`, `GET /metadata`, `WS /ws`

---

## Building an Environment: Step-by-Step

### Step 1: Choose Your Simulator

The simulator is the physics engine of your environment. It models the world the agent interacts with. Examples:

| Domain | Simulator | What It Models |
|--------|-----------|---------------|
| Diabetes | simglucose | Glucose-insulin dynamics in T1D patients |
| Kubernetes | minikube | Container orchestration and failures |
| Traffic | SUMO | Vehicle flow and signal control |
| Finance | Custom | Market dynamics and portfolio risk |

Your simulator must be: deterministic (given same inputs → same outputs), steppable (advance by fixed time increments), and observable (expose internal state).

### Step 2: Define the Observation Space

The observation is the agent's window into the world. Design principles:

1. **Include everything the agent needs to decide.** If a human expert would look at it, include it.
2. **Add temporal context.** A single reading is often insufficient — include history windows.
3. **Distinguish noisy from true values.** Agent sees noisy data; reward uses true data.
4. **Hide information strategically.** In harder tasks, remove fields that make the problem easy.
5. **Include the agent's own last action.** Agents need to know what they just did.

```python
class MyObservation(Observation):
    # Primary measurement (what the sensor shows — possibly noisy)
    measurement: float = Field(description="Current sensor reading")
    
    # Temporal context (enables reasoning without RNN)
    history_window: list[float] = Field(default_factory=list,
        description="Last N readings for temporal reasoning")
    
    # Announced future events (for easier tasks)
    event_announced: bool = Field(default=False)
    
    # Agent's own last action (so it knows what it did)
    last_action_value: float = Field(default=0.0)
    
    # Debug/research field (not used for decisions)
    true_measurement: float = Field(default=0.0,
        description="Pre-noise ground truth — debug only")
```

### Step 3: Define the Action Space

Keep it minimal — only the controls that matter.

```python
class MyAction(Action):
    continuous_control: float = Field(ge=0.0, le=10.0,
        description="Main continuous control value")
    discrete_choice: float = Field(ge=0.0, le=5.0,
        description="Secondary control")
```

Use `ge`/`le` constraints to define valid ranges. The agent can send any value within these bounds.

### Step 4: Design the Reward Function

**This is the most important design decision.** The reward function teaches the agent what to optimize.

Design principles:

1. **Dense rewards** — give feedback every step, not just at episode end
2. **Asymmetric penalties** — if one failure mode is more dangerous, penalize it more
3. **Anti-exploitation** — prevent reward hacking with specific penalties
4. **Recovery incentives** — reward correcting problems, not just avoiding them
5. **Decomposed components** — track each reward component separately for debugging

```python
def calculate_reward(current_value, prev_value, agent_action, ...):
    # Component 1: Primary goal
    if in_target_range(current_value):
        primary = +1.0
    else:
        primary = 0.0
    
    # Component 2: Asymmetric penalty (one direction is worse)
    if dangerously_low(current_value):
        danger_penalty = -3.0   # life-threatening
    elif dangerously_high(current_value):
        danger_penalty = -0.5   # damaging but slower
    else:
        danger_penalty = 0.0
    
    # Component 3: Anti-exploitation
    if agent_caused_the_problem(agent_action, current_value):
        exploit_penalty = -3.0
    else:
        exploit_penalty = 0.0
    
    # Component 4: Recovery bonus
    if was_bad and now_good and within_time_window:
        recovery = +0.5
    else:
        recovery = 0.0
    
    total = primary + danger_penalty + exploit_penalty + recovery
    return RewardObject(primary, danger_penalty, exploit_penalty, recovery, total)
```

### Step 5: Design the Task Curriculum

Progressive difficulty is what makes an environment interesting. Each task should isolate a specific challenge:

```
Task 1 (Easy):    Single scenario, full information, no disruptions
Task 2 (Medium):  Same scenario + disruptions (announced in advance)
Task 3 (Hard):    Random scenarios + disruptions (unannounced)  
Task 4 (Expert):  Random scenarios + hidden adversity + no information
```

Pattern: **Add one new challenge per task level.** Don't pile everything on at once.

### Step 6: Build the Graders

Graders score completed episodes. They are SEPARATE from rewards:

- **Rewards** = per-step training signal (shaped, dense, includes bonuses)
- **Graders** = per-episode evaluation metric (simple, deterministic, for judges)

```python
SCORE_MIN = 0.01  # Validator requires strictly > 0
SCORE_MAX = 0.99  # Validator requires strictly < 1

def _clamp(score):
    return max(SCORE_MIN, min(SCORE_MAX, score))

def score_task_1(state):
    readings = state.history[1:]  # exclude initial
    if not readings:
        return SCORE_MIN
    in_range = sum(1 for v in readings if is_good(v))
    tir = in_range / len(readings)
    score = tir - (state.bad_events * 0.1)
    return _clamp(score)

# GRADER DISPATCH
GRADERS = {1: score_task_1, 2: score_task_2, ...}

def grade(task_id, state):
    return GRADERS[task_id](state)
```

**CRITICAL**: Grader scores must be strictly between 0 and 1 (not 0.0, not 1.0). Use `_clamp()` on every return path.

### Step 7: Wire It All Together

Read `references/ARCHITECTURE.md` for the complete file structure and code patterns.

### Step 8: Add Realism Enhancements

What separates a toy environment from a research-grade one:

- **Sensor noise** — Agent sees noisy data, reward uses true data
- **Pharmacokinetic models** — Actions have delayed, non-linear effects
- **Exercise/disruption events** — External factors the agent must adapt to
- **Hidden adversity** — System properties change without notification

### Step 9: Write Tests

Test everything:
- reset() returns valid observation
- step() advances correctly
- Termination conditions work
- Rewards are in expected range for each zone
- Graders produce correct scores for known inputs
- Each enhancement (noise, events, etc.) behaves correctly

### Step 10: Build the Dashboard

A visual interface that demonstrates the environment. For HuggingFace Spaces:

- Serve HTML at `GET /` (root URL)
- Use WebSocket for real-time updates (same `/ws` as the agent)
- Include Chart.js for visualization
- Add Reset, Step, and Run Agent buttons
- Show key metrics (score, step count, events)

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| HF Space shows 404 | No route at `/` | Add `@app.get("/")` serving HTML |
| Grader scores rejected | Returns exact 0.0 or 1.0 | Clamp to (0.01, 0.99) |
| WebSocket closes immediately | `max_concurrent_envs=1` and browser tab open | Close browser before running inference |
| Rewards all 1.00 in stdout | Not normalizing for validator | Normalize to (0.01, 0.99) range |
| `[END]` missing `score=` field | Old format without score | Add `score=X.XX` between `steps=` and `rewards=` |
| Import error on deploy | Circular import or wrong path | Use `sys.path.insert(0, ...)` in app.py |
| Simulator state leaks between episodes | Not resetting all state in reset() | Clear ALL instance variables in reset() |

---

## Quick Reference

### OpenEnv Imports
```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.http_server import create_app
from openenv.core.env_client import EnvClient
```

### openenv.yaml Spec
```yaml
spec_version: 1
name: my_env
display_name: "My Environment"
description: >
  What this environment does
type: space
runtime: fastapi
app: server.app:app
port: 8000
tasks:
  - id: task_1
    name: Task Name
    difficulty: easy
    description: What this task tests
action_space:
  type: continuous
  fields:
    control: {type: float, min: 0.0, max: 10.0}
observation_space:
  fields:
    measurement: {type: float, min: 0.0, max: 1000.0}
```

### Validation Command
```bash
openenv validate .
# Must return: [OK] : Ready for multi-mode deployment
```
