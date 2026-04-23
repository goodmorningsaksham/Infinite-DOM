# OpenEnv Environment Architecture Reference

Complete file-by-file blueprint for building an OpenEnv environment. Every code pattern is production-tested.

## Table of Contents

1. [Project Structure](#project-structure)
2. [models.py — Data Models](#modelspy)
3. [server/constants.py — Configuration](#serverconstantspy)
4. [server/patient_manager.py — Simulator Wrapper](#serversimulator_wrapperpy)
5. [server/reward_calculator.py — Reward Function](#serverreward_calculatorpy)
6. [server/environment.py — Core Environment](#serverenvironmentpy)
7. [server/graders.py — Task Graders](#servergraderspy)
8. [server/app.py — FastAPI Server](#serverapspy)
9. [client.py — Agent Client](#clientpy)
10. [inference.py — Submission Script](#inferencepy)
11. [openenv.yaml — Spec File](#openenvyaml)
12. [Dockerfile — Deployment](#dockerfile)
13. [tests/ — Test Suite](#tests)

---

## Project Structure

```
my_env/
├── inference.py              # Agent script (MUST be at root)
├── models.py                 # Action, Observation, State, Reward models
├── client.py                 # WebSocket client for agents
├── eval.py                   # Baseline evaluation script
├── openenv.yaml              # OpenEnv spec (tasks, action/obs space)
├── Dockerfile                # HF Spaces deployment
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata
├── server/
│   ├── __init__.py           # Empty
│   ├── app.py                # FastAPI server + dashboard
│   ├── environment.py        # Core: reset/step/state
│   ├── simulator_wrapper.py  # Wraps your domain simulator
│   ├── reward_calculator.py  # Per-step reward computation
│   ├── graders.py            # Per-episode scoring (task graders)
│   ├── constants.py          # All config, thresholds, schedules
│   └── baseline_agent.py     # PID/rule-based baseline (optional)
└── tests/
    ├── test_environment.py
    ├── test_graders.py
    └── test_reward.py
```

---

## models.py

Defines all data types. Models inherit from OpenEnv base classes.

```python
"""Data models for the environment."""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class MyAction(Action):
    """What the agent controls each step."""
    control_a: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Primary continuous control (units/hr)",
    )
    control_b: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description="Secondary discrete-like control (units)",
    )


class MyObservation(Observation):
    """What the agent sees. Base class provides: done, reward, metadata."""
    
    # Primary measurement (possibly noisy)
    measurement: float = Field(default=0.0,
        description="Current sensor reading (may include noise)")
    measurement_trend: str = Field(default="stable",
        description="Rate of change: rapidly_falling/falling/stable/rising/rapidly_rising")
    
    # Temporal context
    history_window: list[float] = Field(default_factory=list,
        description="Last 12 readings for temporal reasoning")
    
    # Event announcements (task-dependent)
    event_announced: bool = Field(default=False,
        description="Upcoming event within announcement window")
    event_magnitude: float = Field(default=0.0,
        description="Size of announced event")
    
    # System state
    active_effect_units: float = Field(default=0.0,
        description="Active effect from recent actions (PK/PD model)")
    time_hours: float = Field(default=0.0,
        description="Current time in hours")
    step: int = Field(default=0,
        description="Current step number")
    
    # Agent's last action (echoed back)
    last_action_a: float = Field(default=0.0)
    last_action_b: float = Field(default=0.0)
    
    # Identity (hidden in hard tasks)
    scenario_id: Optional[str] = Field(default=None,
        description="Scenario identifier (None in hard tasks)")
    
    # Debug fields
    true_measurement: float = Field(default=0.0,
        description="Pre-noise ground truth")


class MyState(State):
    """Full episode state for grading. Base provides: episode_id, step_count."""
    task_id: int = Field(description="Current task: 1, 2, 3, or 4")
    scenario_name: str = Field(default="")
    step: int = Field(default=0)
    done: bool = Field(default=False)
    measurement_history: list[float] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)
    in_range_fraction: float = Field(default=0.0)
    bad_events: int = Field(default=0)
    severe_bad_events: int = Field(default=0)
    episode_reward_total: float = Field(default=0.0)


class MyReward:
    """Decomposed reward for debugging. NOT an OpenEnv base class."""
    def __init__(self, primary=0.0, danger_penalty=0.0, exploit_penalty=0.0,
                 recovery_bonus=0.0, step_total=0.0):
        self.primary = primary
        self.danger_penalty = danger_penalty
        self.exploit_penalty = exploit_penalty
        self.recovery_bonus = recovery_bonus
        self.step_total = step_total
```

---

## server/constants.py

All configuration in one place. No magic numbers in other files.

```python
"""Shared constants. Imported by environment, reward, graders."""

import random

# Measurement thresholds
TARGET_LOW = 70.0
TARGET_HIGH = 180.0
SEVERE_LOW = 54.0
SEVERE_HIGH = 250.0
CRITICAL_LOW = 10.0  # Simulation termination

# Episode configuration
STEPS_PER_EPISODE = 480
STEP_DURATION_MIN = 3

# Event schedule: {step_number: magnitude}
EVENT_SCHEDULE = {
    100: 50.0,  # Event 1
    200: 70.0,  # Event 2
    320: 80.0,  # Event 3
}
EVENT_ANNOUNCEMENT_STEPS = 10  # Announce N steps in advance

# Scenario pool
ALL_SCENARIOS = ["scenario_" + str(i) for i in range(1, 31)]
DEFAULT_SCENARIO = "scenario_1"
EVAL_SCENARIOS = random.Random(42).sample(ALL_SCENARIOS, 5)

# PK/PD model (pharmacokinetics)
PK_T_PEAK_MIN = 55
PK_T_END_MIN = 480
PK_HISTORY_STEPS = 160

# Disruption events
DISRUPTION_LEVELS = [0.3, 0.5, 0.7, 1.0]
DISRUPTION_DURATION_STEPS = [10, 20, 30]

# Task 4: Hidden adversity
ADVERSITY_MIN = 1.5
ADVERSITY_MAX = 2.5
ADVERSITY_ONSET_MIN = 20
ADVERSITY_ONSET_MAX = 100
```

---

## server/simulator_wrapper.py

Wraps your domain simulator. Isolates all simulator-specific code.

```python
"""Simulator wrapper. All domain-specific simulation code lives here."""

import numpy as np
from server.constants import STEP_DURATION_MIN

# Import your actual simulator
# from simglucose.patient.t1dpatient import T1DPatient
# from simglucose.sensor.cgm import CGMSensor


class SimulatorWrapper:
    """Wraps the domain simulator with noise and sensitivity support."""

    def __init__(self):
        self._simulator = None
        self._noise_std = 10.0  # Sensor noise standard deviation

    def reset(self, scenario_name: str, seed: int = None):
        """Create and initialize the simulator for a given scenario."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize your simulator here
        # self._simulator = T1DPatient.withName(scenario_name)
        
        # Get initial reading
        true_value = self._get_true_reading()
        noisy_value = self._add_noise(true_value)
        return noisy_value, true_value

    def step(self, action_a: float, action_b: float,
             event_magnitude: float = 0.0,
             sensitivity_multiplier: float = 1.0) -> tuple[float, float]:
        """Advance simulation by one step.
        
        Args:
            action_a: Primary control value (e.g., basal rate)
            action_b: Secondary control value (e.g., bolus dose)
            event_magnitude: External event input (e.g., carbs from meal)
            sensitivity_multiplier: Modifies action effectiveness
                >1.0 = actions more effective (e.g., exercise)
                <1.0 = actions less effective (e.g., illness)
        
        Returns:
            (noisy_reading, true_reading) tuple
        """
        if self._simulator is None:
            raise RuntimeError("Simulator not initialized — call reset() first")

        # Convert action units for simulator
        action_per_min = (action_a / 60.0 + action_b / STEP_DURATION_MIN) * sensitivity_multiplier

        # Run simulator for STEP_DURATION_MIN minutes
        for minute in range(STEP_DURATION_MIN):
            event_input = event_magnitude if minute == 0 else 0.0
            # self._simulator.step(action=action_per_min, event=event_input)
        
        true_value = self._get_true_reading()
        noisy_value = self._add_noise(true_value)
        return noisy_value, true_value

    def _get_true_reading(self) -> float:
        """Get the true measurement from the simulator."""
        # return self._simulator.observation.Gsub
        return 140.0  # placeholder

    def _add_noise(self, true_value: float) -> float:
        """Add sensor noise to true reading."""
        noise = np.random.normal(0, self._noise_std)
        return true_value + noise
```

---

## server/reward_calculator.py

Computes per-step reward with decomposed components.

```python
"""Per-step reward computation."""

from server.constants import TARGET_LOW, TARGET_HIGH, SEVERE_LOW, SEVERE_HIGH

RECOVERY_WINDOW = 10  # Steps allowed for recovery bonus


def calculate_step_reward(current, prev, action_b, 
                          action_b_2_steps_ago=0.0,
                          steps_since_low_start=None,
                          steps_since_high_start=None):
    """Calculate decomposed reward for one step.
    
    Returns object with: primary, danger_penalty, exploit_penalty,
                         recovery_bonus, step_total
    """
    
    # 1. Primary goal: in target range
    primary = 1.0 if TARGET_LOW <= current <= TARGET_HIGH else 0.0

    # 2. Danger penalty (asymmetric)
    if current < SEVERE_LOW:
        danger_penalty = -3.0      # Severe low — life threatening
    elif current < TARGET_LOW:
        danger_penalty = -1.0      # Mild low — dangerous
    elif current > SEVERE_HIGH:
        danger_penalty = -1.5      # Severe high
    elif current > TARGET_HIGH:
        danger_penalty = -0.5      # Mild high
    else:
        danger_penalty = 0.0

    # 3. Anti-exploitation: agent caused the crash
    exploit_penalty = 0.0
    if current < SEVERE_LOW and action_b_2_steps_ago > 5.0:
        exploit_penalty = -3.0

    # 4. Recovery bonus: corrected problem within window
    recovery_bonus = 0.0
    if (steps_since_low_start is not None
            and steps_since_low_start <= RECOVERY_WINDOW
            and current >= TARGET_LOW and prev < TARGET_LOW):
        recovery_bonus = 0.5
    elif (steps_since_high_start is not None
            and steps_since_high_start <= RECOVERY_WINDOW
            and current <= TARGET_HIGH and prev > TARGET_HIGH):
        recovery_bonus = 0.3

    total = primary + danger_penalty + exploit_penalty + recovery_bonus

    # Return as simple namespace (or use a dataclass/pydantic model)
    class Reward:
        pass
    r = Reward()
    r.primary = primary
    r.danger_penalty = danger_penalty
    r.exploit_penalty = exploit_penalty
    r.recovery_bonus = recovery_bonus
    r.step_total = total
    return r
```

---

## server/environment.py

The core environment. Implements OpenEnv's three required methods.

```python
"""Core environment implementation."""

import logging
import random
from collections import deque
from typing import Optional, Any
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import MyAction, MyObservation, MyState
from server.simulator_wrapper import SimulatorWrapper
from server.reward_calculator import calculate_step_reward
from server.constants import (
    STEPS_PER_EPISODE, STEP_DURATION_MIN, EVENT_SCHEDULE,
    EVENT_ANNOUNCEMENT_STEPS, ALL_SCENARIOS, DEFAULT_SCENARIO,
    CRITICAL_LOW, SEVERE_LOW, TARGET_LOW, TARGET_HIGH,
    PK_T_PEAK_MIN, PK_T_END_MIN, PK_HISTORY_STEPS,
    DISRUPTION_LEVELS, DISRUPTION_DURATION_STEPS,
    ADVERSITY_MIN, ADVERSITY_MAX, ADVERSITY_ONSET_MIN, ADVERSITY_ONSET_MAX,
)

logger = logging.getLogger(__name__)

# Precompute PK/PD curve once at import time
def _precompute_pk_curve():
    from scipy.stats import gamma as gamma_dist
    shape_k = 2
    scale = PK_T_PEAK_MIN / (shape_k - 1)
    time_pts = np.linspace(0, PK_T_END_MIN, PK_HISTORY_STEPS)
    return gamma_dist.cdf(time_pts, a=shape_k, scale=scale)

_PK_CURVE = _precompute_pk_curve()

MAX_CONSECUTIVE_SEVERE = 5


class MyEnvironment(Environment):
    """OpenEnv-compliant RL environment."""

    def __init__(self):
        self._sim = SimulatorWrapper()
        self._task_id = 1
        self._step_count = 0
        self._done = True
        self._measurement_history = []
        self._noisy_history = []
        self._reward_history = []
        self._action_history = []
        self._episode_reward = 0.0
        self._scenario_name = DEFAULT_SCENARIO
        self._episode_id = ""
        self._consecutive_severe = 0
        self._bad_events = 0
        self._severe_bad_events = 0
        self._high_events = 0
        self._action_effect_history = deque([0.0] * PK_HISTORY_STEPS,
                                             maxlen=PK_HISTORY_STEPS)
        # Disruption tracking
        self._current_disruption = 0.0
        self._disruption_steps_remaining = 0
        self._disruption_schedule = {}
        self._disruption_duration_map = {}
        # Recovery tracking
        self._low_start_step = None
        self._high_start_step = None
        # Task 4: hidden adversity
        self._adversity_multiplier = 1.0
        self._adversity_onset = None
        self._adversity_active = False

    # ── reset ────────────────────────────────────────────────────

    def reset(self, task_id=1, seed=None, episode_id=None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._task_id = int(task_id)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._done = False
        self._measurement_history = []
        self._noisy_history = []
        self._reward_history = []
        self._action_history = []
        self._episode_reward = 0.0
        self._consecutive_severe = 0
        self._bad_events = 0
        self._severe_bad_events = 0
        self._high_events = 0
        self._action_effect_history = deque([0.0] * PK_HISTORY_STEPS,
                                             maxlen=PK_HISTORY_STEPS)
        self._current_disruption = 0.0
        self._disruption_steps_remaining = 0
        self._low_start_step = None
        self._high_start_step = None
        self._adversity_multiplier = 1.0
        self._adversity_onset = None
        self._adversity_active = False

        # Task-specific setup
        self._setup_disruptions()
        self._setup_adversity()
        self._select_scenario()

        # Initialize simulator
        noisy, true_val = self._sim.reset(self._scenario_name, seed)
        self._measurement_history.append(true_val)
        self._noisy_history.append(noisy)

        return self._build_observation(noisy, true_val)

    # ── step ─────────────────────────────────────────────────────

    def step(self, action, timeout_s=None, **kwargs):
        if self._done:
            last_true = self._measurement_history[-1] if self._measurement_history else 0.0
            last_noisy = self._noisy_history[-1] if self._noisy_history else 0.0
            return self._build_observation(last_noisy, last_true, force_done=True)

        a_val = action.control_a
        b_val = action.control_b
        self._action_history.append((a_val, b_val))

        # Track action effects for PK/PD
        effect_this_step = (a_val * STEP_DURATION_MIN / 60.0) + b_val
        self._action_effect_history.append(effect_this_step)

        # Update disruption state
        self._update_disruptions()

        # Compute sensitivity multiplier
        sensitivity = self._compute_sensitivity()

        # Get event magnitude for this step
        event_mag = self._get_event_magnitude(self._step_count)

        # Advance simulator
        noisy, true_val = self._sim.step(
            a_val, b_val, event_mag,
            sensitivity_multiplier=sensitivity,
        )

        self._measurement_history.append(true_val)
        self._noisy_history.append(noisy)
        self._step_count += 1

        # Compute reward
        prev = self._measurement_history[-2] if len(self._measurement_history) >= 2 else true_val
        b_2_ago = self._action_history[-2][1] if len(self._action_history) >= 2 else 0.0

        steps_since_low = (self._step_count - self._low_start_step
                           if self._low_start_step is not None else None)
        steps_since_high = (self._step_count - self._high_start_step
                            if self._high_start_step is not None else None)

        reward = calculate_step_reward(
            true_val, prev, b_val, b_2_ago, steps_since_low, steps_since_high
        )
        self._reward_history.append(reward.step_total)
        self._episode_reward += reward.step_total

        # Track events
        self._track_events(true_val)

        # Check termination
        if self._step_count >= STEPS_PER_EPISODE:
            self._done = True
        if self._consecutive_severe >= MAX_CONSECUTIVE_SEVERE:
            self._done = True

        return self._build_observation(noisy, true_val, reward_value=reward.step_total)

    # ── state ────────────────────────────────────────────────────

    @property
    def state(self):
        in_range = self._compute_in_range_fraction()
        return MyState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            scenario_name=self._scenario_name,
            step=self._step_count,
            done=self._done,
            measurement_history=list(self._measurement_history),
            reward_history=list(self._reward_history),
            in_range_fraction=in_range,
            bad_events=self._bad_events,
            severe_bad_events=self._severe_bad_events,
            episode_reward_total=self._episode_reward,
        )

    # ── helpers ──────────────────────────────────────────────────

    def _build_observation(self, noisy, true_val, reward_value=None, force_done=False):
        done = force_done or self._done
        trend = self._compute_trend()
        
        # Event announcements (task 2 only)
        announced, magnitude = False, 0.0
        if self._task_id == 2:
            announced, magnitude = self._check_announcement(self._step_count)

        # Scenario identity (hidden in tasks 3/4)
        scenario_id = self._scenario_name if self._task_id in (1, 2) else None

        # Last action
        if self._action_history:
            last_a, last_b = self._action_history[-1]
        else:
            last_a, last_b = 1.0, 0.0

        # History window
        window = self._noisy_history[-12:] if self._noisy_history else []
        window = [round(v, 1) for v in window]

        return MyObservation(
            measurement=round(noisy, 2),
            measurement_trend=trend,
            history_window=window,
            event_announced=announced,
            event_magnitude=magnitude,
            active_effect_units=self._compute_active_effect(),
            time_hours=round((self._step_count * STEP_DURATION_MIN) / 60.0, 2),
            step=self._step_count,
            last_action_a=round(last_a, 4),
            last_action_b=round(last_b, 4),
            scenario_id=scenario_id,
            true_measurement=round(true_val, 2),
            done=done,
            reward=reward_value,
        )

    def _compute_active_effect(self):
        """PK/PD model: compute active effect from action history."""
        history = np.array(list(self._action_effect_history))[::-1]
        remaining = 1.0 - _PK_CURVE
        return max(0.0, round(float(np.sum(history * remaining)), 4))

    def _compute_trend(self):
        if len(self._noisy_history) < 2:
            return "stable"
        diff = self._noisy_history[-1] - self._noisy_history[-2]
        rate = diff / STEP_DURATION_MIN
        if rate < -3:    return "rapidly_falling"
        elif rate < -1:  return "falling"
        elif rate > 3:   return "rapidly_rising"
        elif rate > 1:   return "rising"
        return "stable"

    def _compute_in_range_fraction(self):
        readings = self._measurement_history[1:]
        if not readings: return 0.0
        in_range = sum(1 for v in readings if TARGET_LOW <= v <= TARGET_HIGH)
        return in_range / len(readings)

    def _track_events(self, true_val):
        if true_val < TARGET_LOW:
            self._bad_events += 1
            if self._low_start_step is None:
                self._low_start_step = self._step_count
        else:
            self._low_start_step = None

        if true_val < SEVERE_LOW:
            self._severe_bad_events += 1
            self._consecutive_severe += 1
        else:
            self._consecutive_severe = 0

        if true_val > TARGET_HIGH:
            self._high_events += 1
            if self._high_start_step is None:
                self._high_start_step = self._step_count
        else:
            self._high_start_step = None

    def _get_event_magnitude(self, step):
        if self._task_id == 1: return 0.0
        return EVENT_SCHEDULE.get(step, 0.0)

    def _check_announcement(self, step):
        for event_step, mag in EVENT_SCHEDULE.items():
            if 0 < (event_step - step) <= EVENT_ANNOUNCEMENT_STEPS:
                return True, mag
        return False, 0.0

    def _setup_disruptions(self):
        """Configure disruption events based on task."""
        self._disruption_schedule = {}
        self._disruption_duration_map = {}
        if self._task_id >= 3 and random.random() < 0.6:
            step = random.randint(60, 350)
            intensity = random.choice(DISRUPTION_LEVELS)
            duration = random.choice(DISRUPTION_DURATION_STEPS)
            self._disruption_schedule[step] = intensity
            self._disruption_duration_map[step] = duration

    def _update_disruptions(self):
        if self._step_count in self._disruption_schedule:
            self._current_disruption = self._disruption_schedule[self._step_count]
            self._disruption_steps_remaining = self._disruption_duration_map.get(
                self._step_count, 20)
        if self._disruption_steps_remaining > 0:
            self._disruption_steps_remaining -= 1
            if self._disruption_steps_remaining <= 0:
                self._current_disruption = 0.0

    def _compute_sensitivity(self):
        sensitivity = 1.0
        if self._current_disruption > 0:
            sensitivity *= 1.0 + self._current_disruption * 0.7
        if (self._task_id == 4 and self._adversity_onset is not None
                and self._step_count >= self._adversity_onset):
            self._adversity_active = True
            sensitivity *= (1.0 / self._adversity_multiplier)
        return sensitivity

    def _setup_adversity(self):
        if self._task_id == 4:
            self._adversity_multiplier = random.uniform(ADVERSITY_MIN, ADVERSITY_MAX)
            self._adversity_onset = random.randint(ADVERSITY_ONSET_MIN, ADVERSITY_ONSET_MAX)

    def _select_scenario(self):
        if self._task_id in (1, 2):
            self._scenario_name = DEFAULT_SCENARIO
        else:
            self._scenario_name = random.choice(ALL_SCENARIOS)
```

---

## server/graders.py

Deterministic scoring functions for completed episodes.

```python
"""Task graders — score completed episodes."""

from models import MyState
from server.constants import TARGET_LOW, TARGET_HIGH, SEVERE_LOW, EVENT_SCHEDULE

SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp(score):
    """Clamp to (0.01, 0.99) — validator requires strictly between 0 and 1."""
    return max(SCORE_MIN, min(SCORE_MAX, score))


def score_task_1(state):
    """Easy task grader: in-range fraction + hypo penalty."""
    history = state.measurement_history[1:]
    if not history: return SCORE_MIN
    total = len(history)
    in_range = sum(1 for v in history if TARGET_LOW <= v <= TARGET_HIGH)
    tir = in_range / total
    bonus = 0.05 if state.severe_bad_events == 0 else 0.0
    score = tir + bonus - (state.severe_bad_events * 0.1)
    return _clamp(score)


def score_task_2(state):
    """Medium task grader: TIR + post-event spike penalties."""
    history = state.measurement_history[1:]
    if not history: return SCORE_MIN
    tir = sum(1 for v in history if TARGET_LOW <= v <= TARGET_HIGH) / len(history)
    
    full = state.measurement_history
    spike_penalty = 0.0
    for event_step in EVENT_SCHEDULE:
        start = event_step
        end = min(event_step + 60, len(full))
        if start >= len(full): continue
        window = full[start:end]
        if not window: continue
        peak = max(window)
        if peak > 250: spike_penalty += 0.15
        elif peak > 200: spike_penalty += 0.08
        elif peak > 180: spike_penalty += 0.03
    
    hypo_penalty = min(0.3, state.severe_bad_events * 0.1)
    return _clamp(tir - spike_penalty - hypo_penalty)


def score_task_3(state):
    """Hard task grader: TIR with heavy hypo penalty."""
    history = state.measurement_history[1:]
    if not history: return SCORE_MIN
    tir = sum(1 for v in history if TARGET_LOW <= v <= TARGET_HIGH) / len(history)
    return _clamp(tir - state.severe_bad_events * 0.15)


def score_task_4(state):
    """Expert task grader: TIR + severe high penalty + hypo penalty."""
    history = state.measurement_history[1:]
    if not history: return SCORE_MIN
    total = len(history)
    tir = sum(1 for v in history if TARGET_LOW <= v <= TARGET_HIGH) / total
    severe_high_steps = sum(1 for v in history if v > 300.0)
    high_penalty = min(0.4, severe_high_steps / total * 2)
    return _clamp(tir - state.severe_bad_events * 0.15 - high_penalty)


GRADERS = {1: score_task_1, 2: score_task_2, 3: score_task_3, 4: score_task_4}

def grade(task_id, state):
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return grader(state)
```

---

## server/app.py

FastAPI server with OpenEnv factory + custom endpoints.

```python
"""FastAPI server for the environment."""

import logging, sys, os
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["ENABLE_WEB_INTERFACE"] = "false"  # We serve our own UI

from openenv.core.env_server.http_server import create_app
from server.environment import MyEnvironment
from models import MyAction, MyObservation

app = create_app(MyEnvironment, MyAction, MyObservation,
                 env_name="my_env", max_concurrent_envs=1)


@app.get("/tasks", tags=["Info"])
async def list_tasks():
    return [
        {"id": 1, "name": "Basic Control", "difficulty": "easy"},
        {"id": 2, "name": "Event Management", "difficulty": "medium"},
        {"id": 3, "name": "Cross-Scenario", "difficulty": "hard"},
        {"id": 4, "name": "Hidden Adversity", "difficulty": "expert"},
    ]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_ui():
    """Dashboard HTML — use WebSocket for real-time interaction."""
    return HTMLResponse(content="<html>...</html>")
```

**Critical**: Set `ENABLE_WEB_INTERFACE = "false"` and add your own `GET /` route. Without this, HuggingFace Spaces shows a 404.

---

## inference.py

The submission script. Must follow EXACT stdout format.

```python
"""Inference script — strict [START]/[STEP]/[END] format required."""

import os, json, re
from openai import OpenAI
from client import MyEnvClient
from models import MyAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default

BENCHMARK = "my_env"
SCORE_MIN, SCORE_MAX = 0.01, 0.99
RAW_REWARD_MIN, RAW_REWARD_MAX = -6.0, 1.5

TASK_NAMES = {1: "basic_control", 2: "event_management", 3: "cross_scenario"}

def normalize_reward(r):
    n = SCORE_MIN + (r - RAW_REWARD_MIN) / (RAW_REWARD_MAX - RAW_REWARD_MIN) * (SCORE_MAX - SCORE_MIN)
    return max(SCORE_MIN, min(SCORE_MAX, round(n, 4)))

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={r_str}", flush=True)

def run_task(client_openai, env_url, task_id):
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    log_start(task_name, BENCHMARK, MODEL_NAME)
    
    rewards = []
    steps_completed = 0
    
    try:
        with MyEnvClient(base_url=env_url) as env:
            result = env.reset(task_id=task_id)
            for step in range(1, 481):
                if result.done: break
                # Get LLM action or use fallback
                action = get_action(client_openai, result.observation)
                result = env.step(action)
                r = normalize_reward(result.reward or 0.0)
                rewards.append(r)
                steps_completed = step
                log_step(step, format_action(action), r, result.done)
    except Exception:
        pass
    
    score = sum(rewards) / len(rewards) if rewards else SCORE_MIN
    score = max(SCORE_MIN, min(SCORE_MAX, score))
    log_end(score > 0.5, steps_completed, score, rewards)

def main():
    env_url = os.getenv("OASIS_ENV_URL") or "http://localhost:8000"
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy", timeout=10.0)
    for tid in [1, 2, 3]:
        run_task(client, env_url, tid)

if __name__ == "__main__":
    main()
```

---

## Dockerfile

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest
WORKDIR /app/env
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="/app/env:${PYTHONPATH}"
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## openenv.yaml

```yaml
spec_version: 1
name: my_env
display_name: "My Environment"
description: >
  Description of what this environment does
type: space
runtime: fastapi
app: server.app:app
port: 8000
tasks:
  - id: task_1
    name: Basic Control
    difficulty: easy
    description: Single scenario, full information
  - id: task_2
    name: Event Management
    difficulty: medium
    description: Same scenario with disruptive events
  - id: task_3
    name: Cross-Scenario
    difficulty: hard
    description: Random scenarios, no information
action_space:
  type: continuous
  fields:
    control_a: {type: float, min: 0.0, max: 10.0}
    control_b: {type: float, min: 0.0, max: 20.0}
observation_space:
  fields:
    measurement: {type: float}
    measurement_trend: {type: string}
    event_announced: {type: bool}
```

---

## Tests

Test structure mirrors the architecture:

```python
# tests/test_environment.py
class TestReset:
    def test_returns_observation(self): ...
    def test_measurement_plausible(self): ...
    def test_done_false(self): ...

class TestStep:
    def test_advances_counter(self): ...
    def test_returns_reward(self): ...

class TestTermination:
    def test_done_at_max_steps(self): ...
    def test_emergency_termination(self): ...

# tests/test_graders.py
class TestGraderRange:
    def test_scores_in_valid_range(self): ...
    def test_empty_history(self): ...

# tests/test_reward.py
class TestRewardZones:
    def test_in_range_positive(self): ...
    def test_low_penalty(self): ...
    def test_high_penalty(self): ...
```

Run with: `python -m pytest tests/ -v`
