# OpenEnv Submission Guide & Gotchas

Everything required for passing the OpenEnv hackathon validator and avoiding rejection.

## Table of Contents

1. [Pre-Submission Checklist](#pre-submission-checklist)
2. [inference.py Requirements](#inferencepy-requirements)
3. [Stdout Format Specification](#stdout-format)
4. [Score Range Requirements](#score-range)
5. [Common Validator Failures](#common-failures)
6. [HuggingFace Spaces Deployment](#hf-spaces)
7. [Pre-Validation Script](#pre-validation)

---

## Pre-Submission Checklist

Every item must pass or submission is rejected:

| # | Check | How to Verify |
|---|-------|--------------|
| 1 | HF Space deploys and `/reset` returns 200 | `curl -X POST $SPACE_URL/reset -H "Content-Type: application/json" -d '{}'` |
| 2 | `openenv validate .` passes | Run in project root |
| 3 | Dockerfile builds | `docker build .` |
| 4 | `inference.py` at project root | `test -f inference.py` |
| 5 | Uses `from openai import OpenAI` | `grep "from openai import OpenAI" inference.py` |
| 6 | `API_BASE_URL` has default | `os.getenv("API_BASE_URL") or "https://..."` |
| 7 | `MODEL_NAME` has default | `os.getenv("MODEL_NAME") or "meta-llama/..."` |
| 8 | `HF_TOKEN` has NO default | `HF_TOKEN = os.getenv("HF_TOKEN")` — bare, no fallback |
| 9 | Stdout follows `[START]/[STEP]/[END]` format exactly | See below |
| 10 | `[END]` includes `score=` field | `score=0.XX` between `steps=` and `rewards=` |
| 11 | All rewards/scores strictly in (0, 1) | Never 0.00 or 1.00 |
| 12 | 3+ tasks with graders | Tasks enumerated, each produces score |
| 13 | `flush=True` on all print statements | Prevents buffered stdout |
| 14 | Runtime < 20 minutes | Use fallback actions when LLM fails |
| 15 | Runs on vcpu=2, memory=8gb | No GPU, no heavy ML libraries |

---

## inference.py Requirements

### Environment Variables

```python
# REQUIRED format — do not deviate
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default — judges provide their own
```

### LLM Client

```python
from openai import OpenAI  # MUST use OpenAI client, not requests/httpx

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
    timeout=10.0,
)
```

### Fallback Strategy

LLM credits may run out mid-episode. Always have a fallback:

```python
try:
    response = client.chat.completions.create(...)
    action = parse_response(response)
except Exception:
    action = FALLBACK_ACTION  # Safe default
    llm_failed = True         # Skip LLM for remaining steps
```

---

## Stdout Format

### Required Format (EXACT)

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

### Rules

- One `[START]` per task, at episode begin
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` per task, ALWAYS emitted (even on exception)
- `reward` and `rewards` formatted to 2 decimal places
- `score` formatted to 2 or 3 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the error string, or `null` if none
- All fields on a single line, no newlines within a line
- `flush=True` on every print statement

### Implementation

```python
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={r_str}", flush=True)
```

---

## Score Range Requirements

### The Rule

Every task score and every reward MUST be strictly between 0 and 1:

```
VALID:   0.01, 0.50, 0.92, 0.99
INVALID: 0.00, 1.00, -0.50, 1.50
```

### Where Scores Come From

The validator reads scores from TWO places:

1. **Per-step rewards** in `[STEP]` lines → `reward=0.92`
2. **Episode score** in `[END]` line → `score=0.85`

Both must be in (0, 1).

### How to Normalize Raw Rewards

Raw rewards may range from -6.0 to +1.5. Linear normalization preserves signal:

```python
SCORE_MIN = 0.01
SCORE_MAX = 0.99
RAW_MIN = -6.0   # worst possible raw reward
RAW_MAX = 1.5    # best possible raw reward

def normalize_reward(raw):
    n = SCORE_MIN + (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * (SCORE_MAX - SCORE_MIN)
    return max(SCORE_MIN, min(SCORE_MAX, round(n, 4)))
```

This maps: +1.0 → 0.92, -0.5 → 0.73, -3.0 → 0.40, -6.0 → 0.01

**WARNING**: Clamping to (0.001, 0.999) then formatting with `:.2f` rounds to 0.00 or 1.00. Always use (0.01, 0.99) as bounds.

### Server-Side Grader Scores

Graders must also return scores in (0, 1):

```python
SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp(score):
    return max(SCORE_MIN, min(SCORE_MAX, score))

# Apply _clamp() to EVERY return path in EVERY grader
def score_task_1(state):
    if not state.history:
        return SCORE_MIN  # NOT 0.0
    ...
    return _clamp(score)  # NOT max(0.0, min(1.0, score))
```

### Score Computation in inference.py

Score must be computed AFTER the try/except block to survive WebSocket close crashes:

```python
def run_task(client, env_url, task_id):
    log_start(...)
    rewards = []
    steps = 0
    
    try:
        with EnvClient(base_url=env_url) as env:
            result = env.reset(task_id=task_id)
            for step in range(1, MAX_STEPS + 1):
                if result.done: break
                result = env.step(action)
                r = normalize_reward(result.reward or 0.0)
                rewards.append(r)
                steps = step
                log_step(step, ..., r, result.done, ...)
    except Exception:
        pass  # WebSocket close may crash — rewards already collected

    # AFTER try/except — not inside it
    score = sum(rewards) / len(rewards) if rewards else SCORE_MIN
    score = max(SCORE_MIN, min(SCORE_MAX, score))
    success = score > 0.5
    log_end(success, steps, score, rewards)
```

---

## Common Failures

### "One or more task scores out of range"

**Cause**: Grader returned 0.0 or 1.0, or stdout reward is 0.00 or 1.00.

**Fix**: Clamp ALL scores to (0.01, 0.99). Check both graders AND inference.py normalization. Verify that `:.2f` formatting doesn't round 0.001 to 0.00 or 0.999 to 1.00.

### Double `[START]` in output

**Cause**: Exception handler in `main()` prints `[START]` again after `run_task()` already printed it.

**Fix**: Put try/except INSIDE `run_task()`, not wrapping it. `main()` should just call `run_task()` directly.

```python
# WRONG:
def main():
    for task_id in [1, 2, 3]:
        try:
            run_task(...)
        except:
            log_start(...)  # DUPLICATE
            log_end(...)

# RIGHT:
def run_task(...):
    log_start(...)
    try:
        ...
    except:
        pass
    log_end(...)

def main():
    for task_id in [1, 2, 3]:
        run_task(...)  # No try/except here
```

### WebSocket "ConnectionClosedOK" error

**Cause**: HF Spaces may close WebSocket unexpectedly during `env.close()`.

**Fix**: Wrap the entire `GlucoEnv` block in try/except. Compute score from collected rewards AFTER the except.

### Server crashes with ImportError

**Cause**: Importing graders from inside environment creates circular dependency, or wrong module path.

**Fix**: Keep imports one-directional: `environment.py` imports from `constants.py`, `reward_calculator.py`, `simulator_wrapper.py`. Never import graders into environment.

### HF Space shows 404 at root URL

**Cause**: OpenEnv's `ENABLE_WEB_INTERFACE` isn't working, and no custom root route exists.

**Fix**: Set `os.environ["ENABLE_WEB_INTERFACE"] = "false"` and add `@app.get("/")` route serving your dashboard HTML.

### score=0.01 despite good rewards

**Cause**: Score computation is inside the `try` block. WebSocket close crash triggers `except` which sets score to minimum.

**Fix**: Move score computation AFTER the try/except block, using the rewards list that was already collected.

---

## HF Spaces Deployment

### Dockerfile

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest
WORKDIR /app/env
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="/app/env:${PYTHONPATH}"
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Do NOT include**: `ENV ENABLE_WEB_INTERFACE=true` — handle in Python code instead.

### HuggingFace README Frontmatter

First lines of README.md must be:

```yaml
---
title: My Environment
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---
```

### max_concurrent_envs

Set to `1` in `create_app()`. The validator runs one task at a time. Higher values waste memory on the 8GB limit.

---

## Pre-Validation Script

Run before every submission:

```bash
# Step 1: Ping HF Space
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  $SPACE_URL/reset --max-time 30
# Must return: 200

# Step 2: Docker build
docker build .
# Must succeed

# Step 3: OpenEnv validate
openenv validate .
# Must return: [OK]
```

### Manual Inference Test

```bash
export OASIS_ENV_URL="https://your-space.hf.space"
export HF_TOKEN="hf_your_token"
python inference.py
```

Check output for:
- Exactly one `[START]` per task
- `reward=` values all in (0.01, 0.99)
- `score=` value in `[END]` line
- Exactly one `[END]` per task (even on failure)
- No `score=0.00` or `score=1.00`
