# Infinite DOM — Complete Operations Guide

Step-by-step instructions to run, test, train, deploy, and submit this project.
Every section has a **checkpoint** at the end so you know immediately if something broke.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Initial Setup](#2-initial-setup)
3. [Running Unit Tests (No Browser)](#3-running-unit-tests-no-browser)
4. [Running Browser Tests](#4-running-browser-tests)
5. [Running the Server Locally](#5-running-the-server-locally)
6. [Running the Oracle & Generating Training Data](#6-running-the-oracle--generating-training-data)
7. [Running inference.py](#7-running-inferencepy)
8. [Credentials & Environment Variables](#8-credentials--environment-variables)
9. [Updating inference.py for Submission](#9-updating-inferencepy-for-submission)
10. [Training with Unsloth/TRL](#10-training-with-unsloththf-trl)
11. [HuggingFace Space Deployment](#11-huggingface-space-deployment)
12. [Pre-Submission Validation](#12-pre-submission-validation)
13. [Reward Hacking Prevention & Safety](#13-reward-hacking-prevention--safety)
14. [Model Saving & Export](#14-model-saving--export)
15. [Troubleshooting](#15-troubleshooting)
16. [Hackathon Guide Cross-Reference](#16-hackathon-guide-cross-reference)

---

### Hackathon Guide Alignment Summary

This project follows the **Hackathon Self-Serve Guide** structure. Here's where each guide section is addressed:

| Guide § | Topic | Where Addressed |
|---------|-------|-----------------|
| §1 | Right project idea | README.md — verifiable multi-step web task with crisp reward |
| §2 | RL loop | Environment (reset→step→reward→update), inference.py |
| §3 | SFT first? | Training notebook Cell 5 (SFT), then Cell 10 (GRPO) |
| §4 | Environment design | `infinite_dom/environment/`, `task_graph.py`, `config.py` |
| §5 | OpenEnv build | `server/app.py` using `create_app()`, `openenv.yaml` |
| §6 | Simple first (curriculum) | 4 tasks: Clean Form → Label Drift → Structural → Chaos |
| §7 | Reward design | `reward_calculator.py` — 5 independent components |
| §8 | Reward hacking prevention | Section 13 of this guide + thrash detection + timeouts |
| §9 | Process-aware feedback | Per-step RewardBreakdown with component decomposition |
| §10 | Training stack | Unsloth + TRL + OpenEnv in `train_infinite_dom.ipynb` |
| §11 | GRPO/RLVR | Training notebook Cell 10 — GRPO with verifiable environment |
| §12 | Fast inference | Unsloth 4-bit quantization, per-action timeouts |
| §13 | Deploy early | Dockerfile, smoke_test.py, HF Space deployment (Section 11) |
| §14 | Scale after stable | All 73 tests pass before training; curriculum progression |
| §15 | Monitor training | Training notebook Cell 12 — loss plots, reward curves |
| §16 | Save models correctly | Section 14 of this guide + notebook Cell 6, 11 |
| §17 | Team structure | N/A (solo, but sections parallel the team split) |
| §18 | Execution plan | This guide IS the execution plan |
| §19 | Demo format | README.md baseline→trained comparison, reward curves |
| §21 | Common mistakes | Section 13 (hacking), Section 15 (troubleshooting) |

---

## 1. Prerequisites

**Required software:**

| Software | Version | Check Command |
|----------|---------|---------------|
| Python | >= 3.11 | `python --version` |
| Git | any | `git --version` |
| pip | latest | `pip --version` |
| Node.js | any (for Playwright CLI) | `node --version` |
| Docker | any (for deployment) | `docker --version` |

**Required accounts:**

| Account | URL | What For |
|---------|-----|----------|
| HuggingFace | https://huggingface.co/settings/tokens | `HF_TOKEN` — deploy Space, download models |
| Google Colab | https://colab.research.google.com | Training notebook with GPU |

**Optional accounts:**

| Account | URL | What For |
|---------|-----|----------|
| Weights & Biases | https://wandb.ai/authorize | `WANDB_API_KEY` — training charts (falls back to matplotlib) |

---

## 2. Initial Setup

### 2.1 Clone and Enter the Project

```bash
cd /path/to/your/workspace
git clone <repo-url> infinite-dom
cd infinite-dom
```

### 2.2 Create Virtual Environment

**On Linux/WSL2/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On Windows (Git Bash / MSYS2):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2.3 Upgrade pip

```bash
pip install --upgrade pip
```

### 2.4 Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

This installs: OpenEnv, FastAPI, Uvicorn, Pydantic, Jinja2, Playwright, aiohttp, httpx, PyYAML, rich, numpy.

> **Corporate proxy / SSL issues?** If you see `CERTIFICATE_VERIFY_FAILED`, run:
> ```bash
> pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
> ```

### 2.5 Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

This installs: pytest, pytest-asyncio, pytest-timeout, ruff, ipykernel, beautifulsoup4.

### 2.6 Install Playwright Chromium

```bash
playwright install chromium
```

> **Corporate proxy / SSL issues?** Run with:
> ```bash
> NODE_TLS_REJECT_UNAUTHORIZED=0 playwright install chromium
> ```

On Linux/WSL2, also install system dependencies:
```bash
playwright install-deps chromium
```

### 2.7 Create Your `.env` File

```bash
cp .env.example .env
```

Edit `.env` and fill in your HuggingFace token:
```
HF_TOKEN=hf_your_token_here
```

Leave `WANDB_API_KEY` empty for now (optional).

### CHECKPOINT 1 — Setup Verification

```bash
python -c "import fastapi, pydantic, jinja2, playwright, yaml; print('Core imports: OK')"
python -c "from openenv.core.env_server.interfaces import Environment; print('OpenEnv: OK')"
```

**Expected output:**
```
Core imports: OK
OpenEnv: OK
```

If either fails, re-run the pip install step and check for errors.

---

## 3. Running Unit Tests (No Browser)

These tests run instantly and don't need Playwright/Chromium. Run them first.

```bash
pytest tests/test_phase1_scaffold.py tests/test_models.py tests/test_task_graph.py tests/test_reward.py tests/test_graders.py tests/test_generator.py tests/test_a11y_formatter.py -v
```

### CHECKPOINT 2 — Unit Tests

**Expected output:** `62 passed` (approximately 5 seconds)

```
tests/test_phase1_scaffold.py    — 4 passed   (project structure)
tests/test_models.py             — 10 passed  (Pydantic models)
tests/test_task_graph.py         — 10 passed  (task graph logic)
tests/test_reward.py             — 7 passed   (reward function)
tests/test_graders.py            — 15 passed  (episode graders)
tests/test_generator.py          — 11 passed  (DOM generator)
tests/test_a11y_formatter.py     — 5 passed   (a11y tree formatter)
```

**If any fail:**
- `test_phase1_scaffold` failures → missing files or directories; check `infinite_dom/` tree
- `test_models` failures → Pydantic version issue; check `pip show pydantic`
- `test_generator` failures → Jinja2 template issue; check `infinite_dom/generator/templates/`

---

## 4. Running Browser Tests

These tests launch a real Chromium browser. They are slower (~10-40s each).

### 4.1 Browser Driver Tests

```bash
pytest tests/test_browser_driver.py -v -m browser --timeout=120
```

**Expected:** `3 passed` (~10s)

### 4.2 End-to-End Environment Tests

```bash
pytest tests/test_environment_e2e.py -v -m "e2e and browser" --timeout=120
```

**Expected:** `3 passed` (~20-40s)

### 4.3 Oracle Tests

```bash
pytest tests/test_oracle.py -v -m "e2e and browser" --timeout=120
```

**Expected:** `5 passed` (~40s) — oracle tested on 5 different seeds

### CHECKPOINT 3 — Browser Tests

**Expected total:** `11 passed` across all browser test files.

**If any fail:**
- `"Chromium not found"` → re-run `playwright install chromium`
- `"port 9000 already in use"` → kill process on port 9000: `lsof -ti:9000 | xargs kill`
  (Windows: `netstat -ano | findstr :9000` then `taskkill /PID <pid> /F`)
- `"timeout"` → increase `--timeout=180` and try again

---

## 5. Running the Server Locally

### 5.1 Start the Server

```bash
PYTHONPATH=. uvicorn infinite_dom.server.app:app --host 127.0.0.1 --port 8000
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."
uvicorn infinite_dom.server.app:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5.2 Test the Endpoints (in a separate terminal)

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```
Expected: `{"status":"healthy"}`

**Dashboard:**
Open http://127.0.0.1:8000/ in a browser — you should see the "Infinite DOM" dashboard page.

**Reset (start an episode):**
```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "task_id": 1}'
```
Expected: JSON with observation data (may take 5-15s on first call as Chromium starts).

### 5.3 Run the Smoke Test (automated)

With no server running (it starts its own):
```bash
PYTHONPATH=. python scripts/smoke_test.py
```

### CHECKPOINT 4 — Server Working

**Expected smoke test output:**
```
[OK] /health responded
[OK] /reset returned instruction='...'
[OK] / (dashboard) responded

[SMOKE TEST PASSED]
```

**If it fails:**
- `"server did not start"` → check port 8001 isn't in use
- `"/reset timed out"` → Playwright browser initialization is slow; this may take up to 90s on first run
- Import errors → make sure you have `PYTHONPATH=.` set

---

## 6. Running the Oracle & Generating Training Data

### 6.1 Run Oracle on a Single Episode

```bash
PYTHONPATH=. python -c "
from infinite_dom.environment.infinite_dom_env import InfiniteDOMEnvironment, _run_async
from infinite_dom.generator.serve_html import stop_page_server
from infinite_dom.oracle.booking_flow_oracle import oracle_policy
from infinite_dom.graders import grade

env = InfiniteDOMEnvironment()
try:
    obs = env.reset(task_id=1, seed=42)
    print(f'Instruction: {obs.task_instruction}')
    for i in range(20):
        action = oracle_policy(obs.a11y_tree, env._current_page.task_graph)
        obs = env.step(action)
        print(f'Step {i+1}: action={action.action_type.value} ref={action.element_ref} done={obs.done}')
        if obs.done: break
    score = grade(1, env.state)
    print(f'Completed: {env.state.task_graph_completed}')
    print(f'Score: {score:.4f}')
finally:
    _run_async(env.shutdown())
    _run_async(stop_page_server())
"
```

### 6.2 Generate SFT Training Data

```bash
PYTHONPATH=. python training/generate_oracle_data.py 30
```

This runs the oracle on 30 episodes each for Tasks 1 and 2, writing observation-action pairs.

### CHECKPOINT 5 — Oracle Data

**Expected output:**
```
[task=1 ep=0 seed=1000] steps=25 completed=4/5
...
[task=2 ep=29 seed=2377] steps=25 completed=4/5

[DONE] wrote ~1500 records to training/data/oracle_trajectories.jsonl
```

**Verify the file exists and has content:**
```bash
wc -l training/data/oracle_trajectories.jsonl
head -1 training/data/oracle_trajectories.jsonl | python -m json.tool
```

Expected: 750-1500+ lines, each a valid JSON object with keys: `task_id`, `seed`, `step`, `instruction`, `observation`, `action`.

---

## 7. Running inference.py

The current `inference.py` uses a simple heuristic policy (not an LLM).

```bash
PYTHONPATH=. python inference.py 1 42
```

Arguments: `task_id` (default 1), `seed` (default 42).

### CHECKPOINT 6 — Inference Runs

**Expected output:**
```
[RESET] task_id=1 seed=42 instruction='Book a AC 2 Tier ticket from Mumbai to Bengaluru'
[STEP 1] action=type ref=inp_1 reward=0.140 done=False
[STEP 2] action=type ref=inp_2 reward=-0.010 done=False
...
[END] task_id=1 steps=25 score=0.XXXX rewards=X.XXX
```

If this runs and prints `[END]`, inference works.

---

## 8. Credentials & Environment Variables

Here's every credential and where it goes:

### `.env` File (local development)

```bash
# REQUIRED for HF Space deployment and model access
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Get from: https://huggingface.co/settings/tokens
# Needs: "Write" permission for Space creation

# OPTIONAL — training experiment tracking
WANDB_API_KEY=
# Get from: https://wandb.ai/authorize
# Leave empty to use matplotlib fallback

# Server config (defaults are fine for local dev)
INFINITE_DOM_HOST=0.0.0.0
INFINITE_DOM_PORT=8000
INFINITE_DOM_PAGE_SERVER_PORT=9000
INFINITE_DOM_MAX_STEPS=25

# Playwright (defaults are fine)
PLAYWRIGHT_BROWSERS_PATH=0
PLAYWRIGHT_HEADLESS=true
```

### Colab Notebook Environment Variables

Set these in the notebook before training:
```python
import os
os.environ["INFINITE_DOM_URL"] = "https://your-space.hf.space"  # Your deployed HF Space URL
os.environ["HF_TOKEN"] = "hf_your_token"                        # For model downloads
# os.environ["WANDB_API_KEY"] = "your_wandb_key"                # Optional
```

### HuggingFace Space Secrets

When deploying to HF Spaces, set these as **Space Secrets** (Settings → Repository secrets):

| Secret Name | Value | Required |
|-------------|-------|----------|
| `HF_TOKEN` | Your HF write token | Yes |
| `PLAYWRIGHT_HEADLESS` | `true` | Yes |

### inference.py Environment Variables (for judges)

These are read by inference.py at submission evaluation time:

| Variable | Default | Set By |
|----------|---------|--------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Judges |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Judges |
| `HF_TOKEN` | *(none — no default)* | Judges |

---

## 9. Updating inference.py for Submission

> **STATUS: DONE.** The `inference.py` file has already been updated with all submission requirements. This section documents what was changed and why, for reference.

The `inference.py` now includes all submission requirements:

**1. Add environment variables at the top:**
```python
import os
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default — judges provide theirs
```

**2. Add OpenAI client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
    timeout=10.0,
)
```

> You need to install the openai package: `pip install openai`

**3. Fix stdout format to match EXACT spec:**
```python
def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={r_str}", flush=True)
```

**4. Add reward normalization (all rewards must be in 0.01–0.99):**
```python
SCORE_MIN = 0.01
SCORE_MAX = 0.99

def normalize_reward(raw: float) -> float:
    RAW_MIN, RAW_MAX = -6.0, 1.5
    n = SCORE_MIN + (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * (SCORE_MAX - SCORE_MIN)
    return max(SCORE_MIN, min(SCORE_MAX, round(n, 4)))
```

**5. Add LLM fallback strategy:**
```python
llm_failed = False

def get_llm_action(obs):
    global llm_failed
    if llm_failed:
        return heuristic_action(obs)  # Fallback
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a web agent..."},
                {"role": "user", "content": f"Task: {obs.task_instruction}\n\nPage:\n{obs.a11y_tree}\n\nRespond with JSON action."},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        return parse_llm_response(response.choices[0].message.content)
    except Exception:
        llm_failed = True
        return heuristic_action(obs)
```

**6. Run all 4 tasks (3+ required for submission):**
```python
def main():
    for task_id in [1, 2, 3, 4]:
        run_task(task_id, seed=42)

if __name__ == "__main__":
    main()
```

**7. Compute score AFTER try/except (not inside):**
```python
def run_task(task_id, seed):
    log_start(f"task_{task_id}", "infinite_dom", MODEL_NAME)
    rewards = []
    steps = 0
    try:
        env = InfiniteDOMEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        for step in range(1, 26):
            action = get_llm_action(obs)
            obs = env.step(action)
            r = normalize_reward(obs.reward or 0.0)
            rewards.append(r)
            steps = step
            log_step(step, action.action_type.value, r, obs.done)
            if obs.done:
                break
    except Exception as e:
        log_step(steps + 1, "error", SCORE_MIN, True, str(e))
    finally:
        _run_async(env.shutdown())
        _run_async(stop_page_server())

    # Score AFTER try/except — survives crashes
    score = sum(rewards) / len(rewards) if rewards else SCORE_MIN
    score = max(SCORE_MIN, min(SCORE_MAX, score))
    log_end(score > 0.5, steps, score, rewards)
```

### CHECKPOINT 7 — Updated Inference

After updating, test with:
```bash
PYTHONPATH=. python inference.py
```

Verify:
- [ ] Exactly one `[START]` per task
- [ ] All `reward=` values between 0.01 and 0.99 (never 0.00 or 1.00)
- [ ] `done=true` or `done=false` (lowercase)
- [ ] Exactly one `[END]` per task
- [ ] `score=` between 0.01 and 0.99
- [ ] `flush=True` on every print (no buffered output)

---

## 10. Training with Unsloth/HF TRL

Training happens on **Google Colab with GPU** (T4 for smoke test, A100 for full training).

### 10.1 Preparation (Before Colab)

**Step 1: Deploy your HF Space first** (see Section 11)

**Step 2: Generate oracle training data locally:**
```bash
PYTHONPATH=. python training/generate_oracle_data.py 100
```
This creates `training/data/oracle_trajectories.jsonl` with ~5000 records.

**Step 3: Upload training data to HuggingFace:**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='training/data/oracle_trajectories.jsonl',
    path_in_repo='oracle_trajectories.jsonl',
    repo_id='YOUR_USERNAME/infinite-dom-data',
    repo_type='dataset',
    token='hf_your_token',
)
print('Uploaded!')
"
```

### 10.2 Colab Notebook Setup

Open `training/train_infinite_dom.ipynb` in Colab, or create a new notebook.

**Cell 1 — Install dependencies:**
```python
!pip install -q unsloth
!pip install -q trl==0.12.0 transformers accelerate peft
!pip install -q httpx pydantic openai datasets
!pip install -q "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"
```

**Cell 2 — Download oracle data:**
```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="YOUR_USERNAME/infinite-dom-data",
    filename="oracle_trajectories.jsonl",
    repo_type="dataset",
)
with open(path) as f:
    records = [json.loads(line) for line in f]
print(f"Loaded {len(records)} oracle records")
```

### 10.3 SFT Warmup Training

This fine-tunes a small model on oracle trajectories to give it a warm start.

**Cell 3 — Prepare SFT dataset:**
```python
from datasets import Dataset

def format_for_sft(record):
    system = "You are a web agent. Given an accessibility tree observation and a task instruction, output a JSON action."
    user = f"Task: {record['instruction']}\n\nAccessibility Tree:\n{record['observation']}"
    action = json.dumps(record['action'])
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": action},
        ]
    }

sft_data = [format_for_sft(r) for r in records]
dataset = Dataset.from_list(sft_data)
print(f"SFT dataset: {len(dataset)} examples")
print(dataset[0]["messages"][1]["content"][:200])
```

**Cell 4 — Load model with Unsloth:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=4096,
    dtype=None,        # Auto-detect
    load_in_4bit=True, # Use QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

**Cell 5 — Train SFT:**
```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    max_seq_length=4096,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
print("SFT training complete!")
```

### CHECKPOINT 8 — SFT Training

After training completes, verify:
```python
# Quick test: does the model produce valid JSON?
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

test_prompt = "Task: Book a Sleeper ticket from Delhi to Mumbai\n\nAccessibility Tree:\n[ref=inp_1 role=textbox name=\"From\"]\n[ref=inp_2 role=textbox name=\"To\"]"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Expected: JSON-like output with `action_type`, `element_ref`, `text_value` fields.

### 10.4 GRPO Reinforcement Learning (Hackathon Day)

After SFT warmup, use GRPO with the live environment:

**Cell 6 — Connect to live environment:**
```python
import os
os.environ["INFINITE_DOM_URL"] = "https://your-space.hf.space"

from openenv.core.env_client import EnvClient
env_client = EnvClient(os.environ["INFINITE_DOM_URL"])
print(env_client.metadata())
```

**Cell 7 — GRPO training loop (simplified):**
```python
from trl import GRPOTrainer, GRPOConfig

# Define reward function that calls the environment
def reward_fn(completions, prompts):
    rewards = []
    for completion in completions:
        try:
            action = json.loads(completion)
            # Send to environment and get reward
            obs = env_client.step(action)
            rewards.append(obs.reward or 0.0)
        except Exception:
            rewards.append(-0.1)
    return rewards

grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    logging_steps=5,
)

# Train with curriculum: Task 1 → 2 → 3 → 4
for task_id in [1, 2, 3, 4]:
    print(f"\n=== Training on Task {task_id} ===")
    env_client.reset(task_id=task_id, seed=0)
    # ... run GRPO episodes ...
```

### 10.5 Save Training Plots

```python
import matplotlib.pyplot as plt

# Plot reward curve
plt.figure(figsize=(10, 6))
plt.plot(reward_history)
plt.xlabel("Training Step")
plt.ylabel("Episode Reward")
plt.title("Infinite DOM — Training Reward Curve")
plt.savefig("training_reward_curve.png", dpi=150)
plt.show()
print("Saved training_reward_curve.png")
```

Include this plot in your README and submission materials.

---

## 11. HuggingFace Space Deployment

### 11.1 Create the HF Space

**Option A — CLI:**
```bash
pip install huggingface_hub
huggingface-cli login  # Enter your HF_TOKEN

huggingface-cli repo create infinite-dom --type space --space-sdk docker
```

**Option B — Web UI:**
1. Go to https://huggingface.co/new-space
2. Name: `infinite-dom`
3. SDK: **Docker**
4. Visibility: Public
5. Create

### 11.2 Update Dockerfile for HF Spaces

The current Dockerfile uses the Playwright base image. For HF Spaces, you may want to use the OpenEnv base image instead. Edit `Dockerfile`:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infinite_dom/ ./infinite_dom/
COPY inference.py client.py openenv.yaml ./
COPY scripts/ ./scripts/

EXPOSE 8000 9000

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_HEADLESS=true \
    INFINITE_DOM_HOST=0.0.0.0 \
    INFINITE_DOM_PORT=8000

CMD ["uvicorn", "infinite_dom.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.3 Verify Docker Build Locally

```bash
docker build -t infinite-dom .
```

### CHECKPOINT 9 — Docker Build

**Expected:** Build completes without errors.

Test the container locally:
```bash
docker run -p 8000:8000 -p 9000:9000 infinite-dom
```

In another terminal:
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

### 11.4 Push to HuggingFace Space

```bash
# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/infinite-dom

# Push
git push hf main
```

Or use the HF CLI:
```bash
huggingface-cli upload YOUR_USERNAME/infinite-dom . . --repo-type space
```

### 11.5 Set Space Secrets

Go to your Space settings (https://huggingface.co/spaces/YOUR_USERNAME/infinite-dom/settings):

1. Click **"Repository secrets"**
2. Add: `HF_TOKEN` = your HuggingFace token
3. Add: `PLAYWRIGHT_HEADLESS` = `true`

### 11.6 Wait for Build

The Space will auto-build from the Dockerfile. Check the **"Logs"** tab for progress. First build takes 5-10 minutes.

### CHECKPOINT 10 — Space Deployed

Once the Space shows "Running":

```bash
# Replace with your actual Space URL
export SPACE_URL="https://YOUR_USERNAME-infinite-dom.hf.space"

# Health check
curl $SPACE_URL/health
# Expected: {"status":"healthy"}

# Dashboard
curl -s $SPACE_URL/ | head -5
# Expected: HTML containing "Infinite DOM"

# Reset test
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "task_id": 1}' \
  --max-time 60
# Expected: JSON observation (may take 30-60s first time)
```

---

## 12. Pre-Submission Validation

Run through this checklist before submitting.

### 12.1 OpenEnv Validation

```bash
openenv validate .
```

If the `openenv` CLI isn't available, run the smoke test instead:
```bash
PYTHONPATH=. python scripts/smoke_test.py
```

### 12.2 Inference Output Check

```bash
PYTHONPATH=. python inference.py 2>&1 | tee inference_output.txt
```

Verify the output file:
```bash
# Must have exactly one [START] per task
grep -c "\[START\]" inference_output.txt
# Expected: 4 (one per task)

# Must have exactly one [END] per task
grep -c "\[END\]" inference_output.txt
# Expected: 4

# No score=0.00 or score=1.00
grep "score=0.00\|score=1.00" inference_output.txt
# Expected: no output (nothing matched)

# All rewards in valid range
grep -oP "reward=\K[0-9.]+" inference_output.txt | sort -n | head -1
# Expected: >= 0.01
grep -oP "reward=\K[0-9.]+" inference_output.txt | sort -rn | head -1
# Expected: <= 0.99
```

### 12.3 Docker Build Check

```bash
docker build -t infinite-dom .
```

### 12.4 HF Space Liveness Check

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  $SPACE_URL/reset --max-time 60
# Expected: 200
```

### 12.5 Full Pre-Submission Checklist

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Unit tests pass | `pytest tests/ -m "not browser" -q` | 62 passed |
| 2 | Browser tests pass | `pytest tests/ -m browser --timeout=120 -q` | 11 passed |
| 3 | Docker builds | `docker build .` | Success |
| 4 | `inference.py` exists at root | `test -f inference.py && echo OK` | OK |
| 5 | Stdout format correct | Check inference_output.txt | [START]/[STEP]/[END] |
| 6 | All scores in (0.01, 0.99) | grep check | No 0.00 or 1.00 |
| 7 | 4 tasks defined | `grep -c "task_id" openenv.yaml` | 4 |
| 8 | HF Space responds | `curl $SPACE_URL/health` | {"status":"healthy"} |
| 9 | README has frontmatter | `head -10 README.md` | YAML with title, sdk: docker |
| 10 | Training evidence | `ls training_reward_curve.png` | File exists |
| 11 | `flush=True` on prints | `grep -c "flush=True" inference.py` | > 0 |
| 12 | `openenv validate .` | `openenv validate .` | [OK] or known issues logged |

### CHECKPOINT 11 — Submission Ready

All 12 checks pass → you are ready to submit.

---

## 13. Reward Hacking Prevention & Safety

*Hackathon Guide §8: Protect yourself against reward hacking*

The Infinite DOM has multiple layers of protection against reward hacking:

### 13.1 Multiple Independent Reward Components

The reward function (`infinite_dom/reward_calculator.py`) uses **5 independent signals**:

| Component | Value | Purpose |
|-----------|-------|---------|
| Progression | +0.15 to +0.35 | Dense signal per task-graph node completed |
| Step penalty | −0.01 | Prevents wandering/stalling |
| Invalid action | −0.05 | Teaches action discipline |
| Completion bonus | +1.0 | Terminal success signal |
| Anti-thrash | −0.2 | Catches degenerate loops (3+ identical failed actions) |

A single reward function would be easy to hack. These 5 independent signals make it much harder.

### 13.2 Anti-Cheating Checks

- **Thrash detection** (`_is_thrashing()`): Detects 3+ identical failed actions in a row
- **Max failed actions**: Episode terminates after 5 failed actions (`config.max_failed_actions`)
- **Max steps**: Episode terminates after 25 steps (`config.max_steps`)
- **Per-action timeout**: 10-second timeout on every browser action (`playwright_driver.execute()`)
- **Semantic task graph**: Progress is measured by actual browser state (input values, selected options, URL), not by action sequences — the agent can't game it by replaying a memorized action list

### 13.3 Grader Independence

The graders (`infinite_dom/graders.py`) compute scores from 3 independent dimensions:
- **Completion rate** (weight 0.7): task-graph nodes completed / total
- **Efficiency** (weight 0.3): time remaining / max steps
- **Failure penalty**: −0.05 per failed action (capped at 0.3)

All scores are clamped to (0.01, 0.99) per OpenEnv requirements.

### 13.4 What to Watch For During Training

When running GRPO training, periodically inspect generated actions:

```python
# In training loop, every 50 steps:
sample = model.generate(test_prompt)
print(f"Generated: {sample}")
# Check for:
# - Repeated identical actions
# - Actions targeting non-existent elements
# - Suspiciously high rewards with no task progress
```

---

## 14. Model Saving & Export

*Hackathon Guide §16: Save models correctly*

### 14.1 CRITICAL: Do NOT upcast 4-bit models naively

If you train with QLoRA (4-bit), **do not** do this:
```python
# WRONG — damages model quality
model = model.to(torch.float16)  # upcasts 4-bit to 16-bit
model.merge_and_unload()          # merges corrupted weights
```

### 14.2 Correct saving approaches

**Option A — Save LoRA adapters only (safest):**
```python
model.save_pretrained("./my_adapters")
tokenizer.save_pretrained("./my_adapters")
```

**Option B — Merge with Unsloth (correct 16-bit merge):**
```python
model.save_pretrained_merged("./merged_model", tokenizer, save_method="merged_16bit")
```

**Option C — Push to HuggingFace Hub:**
```python
model.push_to_hub("your_username/infinite-dom-agent", token=HF_TOKEN)
tokenizer.push_to_hub("your_username/infinite-dom-agent", token=HF_TOKEN)
```

### 14.3 Test immediately after saving

```python
# Load and test
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("./my_adapters")
FastLanguageModel.for_inference(model)

outputs = model.generate(test_input, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
# Verify it produces valid JSON actions
```

Do not leave export until the end of the hackathon. Test saves early and often.

---

## 15. Troubleshooting

### Port Already in Use

```bash
# Find what's using the port
# Linux/WSL:
lsof -ti:9000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :9000
taskkill /PID <pid> /F
```

The page server has automatic port fallback — if 9000 is busy, it picks a random free port.

### Playwright Browser Download Fails (SSL)

```bash
NODE_TLS_REJECT_UNAUTHORIZED=0 playwright install chromium
```

### "ModuleNotFoundError: No module named 'infinite_dom'"

Always set PYTHONPATH:
```bash
# Linux/macOS/WSL:
export PYTHONPATH=.

# Windows PowerShell:
$env:PYTHONPATH = "."

# Or prefix every command:
PYTHONPATH=. python <script>
```

### Tests Timeout

Increase the timeout:
```bash
pytest tests/ --timeout=300 -v
```

### "Page accessibility snapshot failed"

This is handled gracefully — the driver falls back to an empty tree. If it happens consistently, check that:
1. Playwright Chromium is installed: `playwright install chromium`
2. The page server is accessible on its port
3. Alpine.js CDN is reachable (corporate proxy may block it)

### Training on Colab: Out of Memory

- Use `load_in_4bit=True` (QLoRA)
- Reduce `per_device_train_batch_size` to 2 or 1
- Reduce `max_seq_length` to 2048
- Use gradient checkpointing: `use_gradient_checkpointing="unsloth"`

### HF Space Build Fails

Check the Space logs. Common issues:
- Missing file in COPY command → check Dockerfile
- pip install fails → check requirements.txt versions
- Port mismatch → ensure `EXPOSE 8000` and `CMD` uses `--port 8000`

---

## 16. Hackathon Guide Cross-Reference

Quick lookup: where each hackathon guide recommendation is implemented in the codebase.

| Guide § | Recommendation | Implementation File(s) |
|---------|---------------|----------------------|
| §2 RL loop | reset→step→reward→update | `infinite_dom_env.py` (reset/step), `reward_calculator.py`, `inference.py` |
| §3 SFT then RL | Warm start before RL | `train_infinite_dom.ipynb` Cell 5 (SFT) then Cell 10 (GRPO) |
| §4 Environment first | reset/step/state/reward | `infinite_dom_env.py`, `task_graph.py`, `config.py` |
| §5 OpenEnv | create_app + FastAPI | `server/app.py`, `openenv.yaml` |
| §6 Simple first | Curriculum of 4 tasks | `variance.py` (task 1-4 progressive difficulty) |
| §7 Multiple rewards | 5 independent components | `reward_calculator.py` (progression, step, invalid, completion, thrash) |
| §8 Anti-hacking | Thrash detect + timeouts | `reward_calculator.py`, `playwright_driver.py` (10s per-action timeout) |
| §9 Process feedback | Step-level breakdown | `RewardBreakdown` dataclass, metadata per observation |
| §10 TRL + Unsloth | Training stack | `train_infinite_dom.ipynb` (Unsloth QLoRA + TRL SFT/GRPO) |
| §11 GRPO | Verifiable reward RL | `train_infinite_dom.ipynb` Cell 10 (GRPO with env verifier) |
| §12 Fast inference | Unsloth 4-bit | `train_infinite_dom.ipynb` Cell 4 (load_in_4bit=True) |
| §13 Deploy early | Docker + HF Space | `Dockerfile`, `smoke_test.py`, Section 11 of this guide |
| §14 Scale after stable | Test-first approach | 73 tests pass before training |
| §15 Monitor | Loss + reward + outputs | `train_infinite_dom.ipynb` Cell 12 (plots), Cell 7 (sanity check) |
| §16 Save correctly | LoRA adapters, not upcast | Section 14 of this guide + `train_infinite_dom.ipynb` Cell 6, 11 |
| §19 Demo | Before/after comparison | `train_infinite_dom.ipynb` Cell 13 (eval), README.md |
| §21 Avoid mistakes | Multiple checks | This guide Sections 12-14 |
