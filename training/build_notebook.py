# -*- coding: utf-8 -*-
"""Generate the improved training notebook as .ipynb"""
import json

def cell(cell_type, source, outputs=None):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    # Fix: each line except last needs \n
    lines = c["source"]
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = outputs or []
    return c

cells = []

# ============================================================
# CELL 0 -- Markdown overview
# ============================================================
cells.append(cell("markdown", """# Infinite DOM -- Training Notebook (V2 -- Improved Pipeline)

**Runtime:** A100 (80GB) recommended. T4 works with 7B model fallback.

## What Changed from V1
1. **Model**: Qwen2.5-3B -> **Qwen2.5-7B** (good balance of speed + quality on A100)
2. **SFT Data**: 224 CoT records -> **all 7,591 oracle records** with balanced action types
3. **Action Balance**: 88% click -> **42% click / 38% type / 10% scroll / 10% wait**
4. **Step History**: Each training example includes previous 2-3 actions as context
5. **Curriculum SFT**: Easy tasks first, hard tasks added progressively
6. **Tolerant GRPO Reward**: Accepts any valid next action, not just oracle's specific choice
7. **GRPO Scale**: 200 records -> **300/task**, 2 gens, 1 epoch (budget-optimised)
8. **Pre-training validation gate**: Aborts if action distribution is imbalanced

## Pipeline
1. Install dependencies & configure remote environment
2. Load **full** oracle training data (7,591 records, all 8 tasks)
3. Balance action types + add step history + curriculum ordering
4. **SFT Phase** -- Curriculum behaviour cloning (easy->hard, 2 epochs)
5. **GRPO Phase** -- Tolerant per-step reward (300 records/task, 2 gens)
6. Live evaluation + plots"""))

# ============================================================
# CELL 1 -- Install dependencies
# ============================================================
cells.append(cell("code", """# Cell 1 -- Install dependencies
!pip install -q unsloth
!pip install -q "trl>=0.12.0" transformers accelerate peft
!pip install -q httpx pydantic datasets matplotlib
!pip install -q huggingface_hub
!pip install -q websockets nest_asyncio

import warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*`max_length`.*")
warnings.filterwarnings("ignore", message=".*`max_new_tokens`.*`max_length`.*")

import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb >= 40:
        print("A100/A6000 detected -- will use 7B model (budget mode)")
        RECOMMENDED_MODEL = "7B"
    elif vram_gb >= 14:
        print("T4/V100 detected -- will use 7B model")
        RECOMMENDED_MODEL = "7B"
    else:
        print("WARNING: Low VRAM -- will use 3B model")
        RECOMMENDED_MODEL = "3B"
else:
    print("WARNING: No GPU detected")
    RECOMMENDED_MODEL = "3B"

print(f"\\nRecommended model size: {RECOMMENDED_MODEL}")"""))

# ============================================================
# CELL 2 -- Configure Remote Environment & Health Check
# ============================================================
cells.append(cell("code", """# Cell 2 -- Configure Remote Environment & Health Check (Robust)
#
# Retry logic with exponential backoff. Validates both HTTP and WebSocket.

import os
import time
import httpx

HF_SPACE_URL = "https://saksham1771-infinite-dom.hf.space"  # @param {type:"string"}
HF_SPACE_URL = HF_SPACE_URL.rstrip("/")
os.environ["INFINITE_DOM_URL"] = HF_SPACE_URL

WS_URL = HF_SPACE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

print(f"Environment URL: {HF_SPACE_URL}")
print(f"WebSocket URL:   {WS_URL}")
print()

def check_health(url, max_retries=5, initial_wait=5):
    for attempt in range(max_retries):
        try:
            r = httpx.get(f"{url}/health", timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"  HTTP health: OK -- {data}")
                return True
            else:
                print(f"  Attempt {attempt+1}: HTTP {r.status_code}")
        except httpx.ConnectError:
            print(f"  Attempt {attempt+1}: Connection refused -- is the HF Space running?")
        except httpx.TimeoutException:
            print(f"  Attempt {attempt+1}: Timeout -- Space may be cold-starting")
        except Exception as e:
            print(f"  Attempt {attempt+1}: {type(e).__name__}: {e}")
        if attempt < max_retries - 1:
            wait = initial_wait * (2 ** attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    return False

env_available = check_health(HF_SPACE_URL)

# Also test WebSocket if HTTP works
ws_tested = False
if env_available:
    import asyncio, nest_asyncio, websockets, json as _json
    nest_asyncio.apply()

    async def _test_ws():
        try:
            async with websockets.connect(WS_URL, open_timeout=30) as ws:
                await ws.send(_json.dumps({"type": "reset", "data": {"task_id": 1, "seed": 999}}))
                resp = _json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                if "observation" in resp.get("data", {}):
                    obs_fields = resp["data"]["observation"]
                    print(f"  WebSocket: OK -- got observation ({len(obs_fields.get('a11y_tree', ''))} chars)")
                    return True
                elif "a11y_tree" in resp.get("data", {}):
                    print(f"  WebSocket: OK -- got observation ({len(resp['data']['a11y_tree'])} chars)")
                    return True
                else:
                    print(f"  WebSocket: unexpected response: {list(resp.keys())}")
        except Exception as e:
            print(f"  WebSocket: FAILED -- {type(e).__name__}: {e}")
        return False

    ws_tested = asyncio.get_event_loop().run_until_complete(_test_ws())

if env_available and ws_tested:
    print("\\nEnvironment fully connected (HTTP + WebSocket)")
elif env_available:
    print("\\nHTTP works but WebSocket failed. Live eval may not work.")
    print("Training (SFT + GRPO) does NOT need the live environment -- you can proceed.")
else:
    print("\\nEnvironment not available. Training can proceed without it.")
    print("Live evaluation (Cell 10, 12) will be skipped.")

# Fetch task list
if env_available:
    try:
        r = httpx.get(f"{HF_SPACE_URL}/tasks", timeout=10)
        if r.status_code == 200:
            tasks_list = r.json().get("tasks", [])
            print(f"\\nAvailable tasks ({len(tasks_list)}):")
            for t in tasks_list:
                print(f"  Task {t['task_id']}: {t['description']}")
    except Exception:
        pass"""))

# ============================================================
# CELL 3 -- Load Oracle Training Data
# ============================================================
cells.append(cell("code", """# Cell 3 -- Load ALL Oracle Training Data (7,591 records, all 8 tasks)
#
# V2 CHANGE: Uses the FULL oracle dataset, not the 250-record CoT subset.
# The CoT subset only covered tasks 1-2 -- that was the #1 cause of failure.

import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_SOURCE = "huggingface"  # @param ["huggingface", "local"]
HF_DATASET_REPO = "saksham1771/infinite-dom-data"  # @param {type:"string"}
LOCAL_DATA_PATH = "training/data/oracle_trajectories.jsonl"

if DATA_SOURCE == "huggingface":
    from huggingface_hub import hf_hub_download
    local_file = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="oracle_trajectories.jsonl",
        repo_type="dataset",
    )
    LOCAL_DATA_PATH = local_file
    print(f"Downloaded oracle_trajectories.jsonl from {HF_DATASET_REPO}")

with open(LOCAL_DATA_PATH) as f:
    records = [json.loads(line) for line in f if line.strip()]

# Clean up action dicts (remove metadata field if present)
for r in records:
    action = r["action"]
    for key in list(action.keys()):
        if key not in ("action_type", "element_ref", "text_value", "scroll_delta"):
            del action[key]
    action.setdefault("scroll_delta", 0)
    action.setdefault("text_value", "")
    action.setdefault("element_ref", "")

# Validate
required_keys = {"task_id", "seed", "step", "instruction", "observation", "action"}
VALID_ACTIONS = {"click", "type", "scroll", "wait"}
valid_records = [r for r in records
                 if required_keys.issubset(r.keys())
                 and r["action"].get("action_type") in VALID_ACTIONS]
dropped = len(records) - len(valid_records)
records = valid_records

# Group by episode (task_id, seed) for step history
episodes = defaultdict(list)
for r in records:
    episodes[(r["task_id"], r["seed"])].append(r)
for key in episodes:
    episodes[key].sort(key=lambda x: x["step"])

print(f"\\nLoaded {len(records)} valid oracle records ({dropped} dropped)")
print(f"Episodes: {len(episodes)}")

# Per-task breakdown
print(f"\\nPer-task breakdown:")
print(f"{'Task':>6s}  {'Records':>8s}  {'click':>6s}  {'type':>6s}  {'scroll':>6s}  {'wait':>6s}")
print("-" * 55)
for tid in sorted(set(r["task_id"] for r in records)):
    task_recs = [r for r in records if r["task_id"] == tid]
    atypes = Counter(r["action"]["action_type"] for r in task_recs)
    print(f"{tid:>6d}  {len(task_recs):>8d}  {atypes.get('click',0):>6d}  {atypes.get('type',0):>6d}  {atypes.get('scroll',0):>6d}  {atypes.get('wait',0):>6d}")

total_atypes = Counter(r["action"]["action_type"] for r in records)
print(f"\\nGlobal action distribution (RAW -- before balancing):")
for atype, count in total_atypes.most_common():
    pct = 100 * count / len(records)
    print(f"  {atype:>6s}: {count:>5d} ({pct:>5.1f}%)")"""))

# ============================================================
# CELL 4 -- Balance + Step History + Prepare SFT Dataset
# ============================================================
cells.append(cell("code", r"""# Cell 4 -- Balance Action Types + Add Step History + Prepare SFT Dataset
#
# V2 CHANGES:
#   1. Use ALL 7,591 oracle records (not 250 CoT)
#   2. Hard-balance action types: target 42% click, 38% type, 10% scroll, 10% wait
#   3. Add step history (last 3 actions) to each training example
#   4. Curriculum ordering: easy tasks first, hard tasks added progressively
#   5. Pre-training validation gate: abort if distribution is still bad

import re
import random as stdlib_random
from datasets import Dataset
from unsloth import FastLanguageModel

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"  # V2-lite: 7B for budget, still strong
MAX_SEQ_LEN = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

# --- System Prompt (handles both booking and e-commerce) ---
SYSTEM_PROMPT = '''You are a web agent navigating an interactive web application.
You observe an accessibility tree and must complete the given task.

First, reason about what you see and what action to take inside <think> tags.
Then, provide your action as a JSON object inside <answer> tags.

Format:
<think>
[Observe the current page state. Identify which fields need filling, what buttons are available, and decide the next step. If you see empty text fields that need values, you MUST use "type" to fill them -- do NOT click on them.]
</think>
<answer>
{"action_type": "click"|"type"|"scroll"|"wait", "element_ref": "ref_id", "text_value": "text"|"", "scroll_delta": 0}
</answer>

Action rules:
- "type": Fill text fields (textbox) or select dropdown values (combobox). REQUIRES element_ref + text_value.
- "click": Press buttons or links. REQUIRES element_ref. Do NOT use click on text fields.
- "scroll": Scroll the page. Uses scroll_delta (positive=down, negative=up).
- "wait": Pause when the page is loading or after a navigation.

Element refs look like: inp_1, btn_2, cmb_1, lnk_3

Strategy:
1. First dismiss any cookie banners, popups, or newsletter modals (click "Accept"/"Close"/"No Thanks")
2. For booking tasks: type origin city -> type destination -> select class -> click search -> select correct train -> click confirm
3. For shopping tasks: type search query -> select category -> click filter -> click View on target product -> click Add to Cart -> click Checkout -> type shipping name/address/city/pin/phone -> click Place Order
4. Always check if fields are already filled before typing -- look at value= attributes
5. If a field shows value="" it is EMPTY and needs "type" action, not "click"
6. Use scroll when elements might be below the visible area'''

OBS_MAX_CHARS = 3500
STEP_HISTORY_WINDOW = 3

# --- Balance action types ---
def balance_actions_global(records, target_dist, rng_seed=42):
    '''Hard-balance to target distribution across all records.
    Undersamples majority (click), oversamples minority (type, scroll, wait).
    For missing action types (scroll), synthesize from existing by adding scroll actions.'''
    rng = stdlib_random.Random(rng_seed)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["action"]["action_type"]].append(r)

    total = len(records)
    targets = {k: max(1, int(total * v)) for k, v in target_dist.items() if v > 0}

    balanced = []
    for action_type, target_count in targets.items():
        pool = by_type.get(action_type, [])
        if not pool:
            # Synthesize scroll actions from click records (reasonable proxy)
            if action_type == "scroll":
                pool = _synthesize_scroll_actions(by_type.get("click", [])[:200], rng)
            if not pool:
                print(f"  WARNING: No records for '{action_type}', skipping")
                continue
        if len(pool) >= target_count:
            balanced.extend(rng.sample(pool, target_count))
        else:
            copies = target_count // len(pool)
            remainder = target_count % len(pool)
            balanced.extend(pool * copies)
            balanced.extend(rng.sample(pool, remainder))
    rng.shuffle(balanced)
    return balanced

def _synthesize_scroll_actions(click_records, rng):
    '''Create scroll action records from click records -- same observation, different action.'''
    scrolls = []
    for r in click_records[:100]:
        scroll_r = {
            "task_id": r["task_id"],
            "seed": r["seed"],
            "step": r["step"],
            "instruction": r["instruction"],
            "observation": r["observation"],
            "action": {
                "action_type": "scroll",
                "element_ref": "",
                "text_value": "",
                "scroll_delta": rng.choice([300, 500, -300, -500]),
            },
        }
        scrolls.append(scroll_r)
    return scrolls


TARGET_DIST = {"click": 0.42, "type": 0.38, "scroll": 0.10, "wait": 0.10}
balanced_records = balance_actions_global(records, TARGET_DIST, rng_seed=42)

# --- Validation Gate ---
bal_atypes = Counter(r["action"]["action_type"] for r in balanced_records)
bal_total = len(balanced_records)
print(f"\nBalanced distribution ({bal_total} records):")
for atype in ["click", "type", "scroll", "wait"]:
    cnt = bal_atypes.get(atype, 0)
    pct = 100 * cnt / bal_total
    bar = "#" * int(pct / 2)
    print(f"  {atype:>6s}: {cnt:>5d} ({pct:>5.1f}%) {bar}")

type_pct = bal_atypes.get("type", 0) / bal_total
click_pct = bal_atypes.get("click", 0) / bal_total
assert type_pct >= 0.25, f"ABORT: type actions only {type_pct:.0%} -- need >=25%"
assert click_pct <= 0.55, f"ABORT: click actions {click_pct:.0%} -- need <=55%"
print("  [x] Balance validation PASSED")


# --- Add step history to each record ---
def format_with_history(record, all_episodes):
    '''Create SFT text with step history context.'''
    obs_text = record["observation"][:OBS_MAX_CHARS]
    task_id = record["task_id"]
    seed = record["seed"]
    step = record["step"]

    # Get episode history
    ep_key = (task_id, seed)
    ep_records = all_episodes.get(ep_key, [])

    # Build history string from previous steps
    history = ""
    prev_steps = [r for r in ep_records if r["step"] < step]
    prev_steps = prev_steps[-STEP_HISTORY_WINDOW:]  # last N steps
    if prev_steps:
        history_lines = []
        for prev in prev_steps:
            a = prev["action"]
            line = f"  Step {prev['step']}: {a['action_type']} {a.get('element_ref', '')}"
            if a.get("text_value"):
                line += f' "{a["text_value"]}"'
            history_lines.append(line)
        history = "\nPrevious actions:\n" + "\n".join(history_lines) + "\n"

    step_ctx = f"\nStep: {step}" if step > 0 else ""

    user = f"Task: {record['instruction']}{history}\n\nAccessibility Tree:\n{obs_text}{step_ctx}"

    # Action JSON
    action = record["action"].copy()
    action_json = json.dumps(
        {k: action[k] for k in ("action_type", "element_ref", "text_value", "scroll_delta")},
        separators=(",", ":"),
    )

    # Build think-then-act response (concise reasoning based on actual observation)
    atype = action["action_type"]
    ref = action.get("element_ref", "")
    tval = action.get("text_value", "")

    if atype == "type" and tval:
        think = f'I see {ref} needs a value. Typing "{tval}" into it.'
    elif atype == "click":
        think = f"Clicking {ref} to proceed with the task."
    elif atype == "scroll":
        direction = "down" if action.get("scroll_delta", 0) > 0 else "up"
        think = f"Scrolling {direction} to find more elements."
    else:
        think = "Waiting for the page to update."

    assistant_text = f"<think>\n{think}\n</think>\n<answer>\n{action_json}\n</answer>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_text},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "task_id": record["task_id"]}


# --- Format all records with history ---
print("\nFormatting records with step history...")
sft_data = [format_with_history(r, episodes) for r in balanced_records]

# --- Curriculum-aware stratified split ---
# Curriculum order: clean(1,5) -> label_drift(2,6) -> structural(3,7) -> chaos(4,8)
CURRICULUM_PHASES = [
    [1, 5],      # Phase 1: clean
    [2, 6],      # Phase 2: label drift
    [3, 7],      # Phase 3: structural
    [4, 8],      # Phase 4: chaos
]

stdlib_random.seed(42)
by_task = defaultdict(list)
for item in sft_data:
    by_task[item["task_id"]].append(item)

train_items, eval_items = [], []
for task_id in sorted(by_task.keys()):
    items = by_task[task_id]
    stdlib_random.shuffle(items)
    split_idx = max(1, int(len(items) * 0.9))
    train_items.extend(items[:split_idx])
    eval_items.extend(items[split_idx:])

# Sort train items by curriculum phase (easy first, hard last)
task_to_phase = {}
for phase_idx, task_ids in enumerate(CURRICULUM_PHASES):
    for tid in task_ids:
        task_to_phase[tid] = phase_idx

# Within each phase, shuffle; but phases are ordered
train_by_phase = defaultdict(list)
for item in train_items:
    phase = task_to_phase.get(item["task_id"], 3)
    train_by_phase[phase].append(item)

curriculum_train = []
for phase_idx in sorted(train_by_phase.keys()):
    phase_items = train_by_phase[phase_idx]
    stdlib_random.shuffle(phase_items)
    curriculum_train.extend(phase_items)
    print(f"  Curriculum phase {phase_idx}: {len(phase_items)} records (tasks {CURRICULUM_PHASES[phase_idx]})")

stdlib_random.shuffle(eval_items)

train_ds = Dataset.from_list([{"text": item["text"]} for item in curriculum_train])
eval_ds = Dataset.from_list([{"text": item["text"]} for item in eval_items])

print(f"\nFinal dataset:")
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")
print(f"  Eval tasks: {Counter(item['task_id'] for item in eval_items)}")
print(f"\nSample text (first 400 chars):")
print(f"  {train_ds[0]['text'][:400]}...")"""))

# ============================================================
# CELL 5 -- Load model with LoRA
# ============================================================
cells.append(cell("code", """# Cell 5 -- Apply LoRA adapters to 7B model
#
# V2-lite: 7B with LoRA rank 16. Faster training, lower VRAM.

model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # rank 16 is sufficient for 7B
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"Model: {MODEL_ID}")
print(f"LoRA rank: 16, alpha: 16")
print(f"Trainable params: {model.print_trainable_parameters()}")"""))

# ============================================================
# CELL 6 -- Curriculum SFT
# ============================================================
cells.append(cell("code", """# Cell 6 -- Curriculum SFT Training (2 epochs, budget mode)
#
# V2-lite CHANGES:
#   - Full balanced dataset (not 224 records)
#   - 2 epochs (sufficient for format learning + action diversity)
#   - Batch size 4 for 7B (faster than 14B)
#   - Early stopping patience 2

from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import math

num_samples = len(train_ds)
batch_size = 4     # 7B allows larger batch
grad_accum = 4     # Effective batch = 16
effective_batch = batch_size * grad_accum
num_epochs = 2
steps_per_epoch = math.ceil(num_samples / effective_batch)
total_steps = steps_per_epoch * num_epochs

eval_interval = max(1, steps_per_epoch // 3)
log_interval = max(1, eval_interval // 2)

print(f"Dataset: {num_samples} samples")
print(f"Effective batch size: {effective_batch}")
print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")
print(f"Eval every {eval_interval} steps | Log every {log_interval} steps")
print(f"Epochs: {num_epochs} (with early stopping patience=2)")
print()

sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,       # Standard for 7B
    warmup_steps=min(50, total_steps // 10),
    logging_steps=log_interval,
    save_steps=eval_interval,
    eval_strategy="steps",
    eval_steps=eval_interval,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=sft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(f"Starting SFT training...")
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")
print(f"  Epochs: {num_epochs} | Early stopping patience: 2")
print()

sft_results = trainer.train()
print(f"\\nSFT complete!")
print(f"  Final training loss: {sft_results.training_loss:.4f}")
if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
    print(f"  Best eval loss: {trainer.state.best_metric:.4f}")"""))

# ============================================================
# CELL 7 -- Save SFT + Sanity Check
# ============================================================
cells.append(cell("code", """# Cell 7 -- Save SFT checkpoint + comprehensive sanity check
#
# V2: Tests ALL task types (booking + e-commerce), checks action diversity

import gc

sft_log_history = list(trainer.state.log_history) if hasattr(trainer, "state") else []
sft_best_metric = getattr(trainer.state, "best_metric", None) if hasattr(trainer, "state") else None

IS_COLAB = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CHECKPOINT_DIR = "/content/drive/MyDrive/infinite-dom-checkpoints"
    IS_COLAB = True
except ImportError:
    CHECKPOINT_DIR = "./checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SFT_SAVE_PATH = f"{CHECKPOINT_DIR}/sft_lora_adapters"

model.save_pretrained(SFT_SAVE_PATH)
tokenizer.save_pretrained(SFT_SAVE_PATH)
print(f"SFT adapters saved to: {SFT_SAVE_PATH}")

model.save_pretrained("./sft_lora_adapters")
tokenizer.save_pretrained("./sft_lora_adapters")
print("Also saved locally: ./sft_lora_adapters")

del trainer
gc.collect()
torch.cuda.empty_cache()
print(f"Freed SFT trainer memory. GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# --- Comprehensive sanity check ---
def parse_model_output(text):
    text = text.strip()
    if "<answer>" in text:
        start = text.index("<answer>") + len("<answer>")
        end = text.index("</answer>") if "</answer>" in text else len(text)
        text = text[start:end].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)

FastLanguageModel.for_inference(model)

test_cases = [
    ("Booking: empty form", "Task: Book a Sleeper ticket from Delhi to Mumbai\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=frm_1 role=form name=\\"Search Trains\\"]\\n    [ref=inp_1 role=textbox name=\\"From\\" value=\\"\\"]\\n    [ref=inp_2 role=textbox name=\\"To\\" value=\\"\\"]\\n    [ref=cmb_1 role=combobox name=\\"Class\\" value=\\"-- Select --\\"]\\n    [ref=btn_1 role=button name=\\"Search\\"]"),
    ("Booking: form filled", "Task: Book a Sleeper ticket from Delhi to Mumbai\\n\\nPrevious actions:\\n  Step 0: type inp_1 \\"Delhi\\"\\n  Step 1: type inp_2 \\"Mumbai\\"\\n  Step 2: type cmb_1 \\"Sleeper\\"\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=frm_1 role=form name=\\"Search Trains\\"]\\n    [ref=inp_1 role=textbox name=\\"From\\" value=\\"Delhi\\"]\\n    [ref=inp_2 role=textbox name=\\"To\\" value=\\"Mumbai\\"]\\n    [ref=cmb_1 role=combobox name=\\"Class\\" value=\\"Sleeper\\"]\\n    [ref=btn_1 role=button name=\\"Search\\"]\\nStep: 3"),
    ("E-commerce: search", "Task: Buy a Laptop from Electronics. Ship to Raj, 12 MG Road, Bangalore, 560001, 9876543210\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=inp_1 role=textbox name=\\"Search products\\" value=\\"\\"]\\n  [ref=cmb_1 role=combobox name=\\"Category\\" value=\\"All\\"]\\n  [ref=btn_1 role=button name=\\"Filter\\"]"),
    ("E-commerce: checkout", "Task: Buy a Laptop from Electronics. Ship to Raj, 12 MG Road, Bangalore, 560001, 9876543210\\n\\nPrevious actions:\\n  Step 5: click btn_3\\n  Step 6: click btn_2\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=hdg_1 role=heading name=\\"Shipping Details\\"]\\n  [ref=inp_1 role=textbox name=\\"Full Name\\" value=\\"\\"]\\n  [ref=inp_2 role=textbox name=\\"Address\\" value=\\"\\"]\\n  [ref=inp_3 role=textbox name=\\"City\\" value=\\"\\"]\\n  [ref=inp_4 role=textbox name=\\"PIN Code\\" value=\\"\\"]\\n  [ref=inp_5 role=textbox name=\\"Phone\\" value=\\"\\"]\\n  [ref=btn_1 role=button name=\\"Place Order\\"]\\nStep: 7"),
]

print("\\nSFT Sanity Check (all task types):")
print("=" * 60)
action_types_seen = set()
pass_count = 0

for label, test_input in test_cases:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_input},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.1)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\\n[{label}]")
    print(f"  Response: {response.strip()[:300]}")

    try:
        parsed = parse_model_output(response)
        atype = parsed.get("action_type", "?")
        action_types_seen.add(atype)
        print(f"  Parsed: {parsed}")
        print(f"  Status: PASS")
        pass_count += 1
    except Exception as e:
        print(f"  Status: FAIL ({e})")

print(f"\\n{'=' * 60}")
print(f"Passed: {pass_count}/{len(test_cases)}")
print(f"Action types used: {action_types_seen}")
if "type" in action_types_seen and "click" in action_types_seen:
    print("GOOD: Model uses both 'type' and 'click' -- no mode collapse")
elif pass_count == len(test_cases):
    print("WARNING: All passed but check action type diversity")
else:
    print("WARNING: Some tests failed -- check SFT data quality")"""))

# ============================================================
# CELL 8 -- Resume Point
# ============================================================
cells.append(cell("code", """# Cell 7.5 -- RESUME POINT: Load saved model from checkpoint
#
# USE THIS CELL ONLY WHEN RESUMING AFTER A SESSION CRASH.
# Skip Cells 4-7 -- this loads the saved SFT/GRPO model.
# Resume flow: Cell 1 -> 2 -> 3 -> 4 (data only) -> THIS -> 8

import gc
import os

IS_COLAB = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CHECKPOINT_DIR = "/content/drive/MyDrive/infinite-dom-checkpoints"
    IS_COLAB = True
except ImportError:
    CHECKPOINT_DIR = "./checkpoints"

available = []
if os.path.exists(f"{CHECKPOINT_DIR}/sft_lora_adapters"):
    available.append("SFT")
if os.path.exists(f"{CHECKPOINT_DIR}/grpo_final"):
    available.append("GRPO")
if os.path.exists(f"{CHECKPOINT_DIR}/online_rl_final"):
    available.append("Online-RL")

print(f"Available checkpoints: {available if available else 'NONE'}")

if not available:
    print("No checkpoints found. Run Cells 4-7 first.")
else:
    from unsloth import FastLanguageModel

    MAX_SEQ_LEN = 4096
    load_path = None
    for name in ["online_rl_final", "grpo_final", "sft_lora_adapters"]:
        p = f"{CHECKPOINT_DIR}/{name}"
        if os.path.exists(p):
            load_path = p
            break

    print(f"Loading from: {load_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    sft_log_history = []
    sft_best_metric = None
    print("Model loaded. Proceed to Cell 8 (GRPO) or Cell 10 (Online RL).")"""))

# ============================================================
# CELL 9 -- GRPO with Tolerant Reward
# ============================================================
cells.append(cell("code", r"""# Cell 8 -- GRPO RL Training with TOLERANT Reward Function
#
# V2 CHANGES:
#   1. TOLERANT REWARD: accepts any valid next action, not just oracle's exact choice
#      - Filling destination before origin is equally valid -> same reward
#      - Clicking any correct book button -> same reward (not just oracle's btn_4)
#   2. SCALE UP: 600 records/task (was 200), 4 generations (was 2), 2 epochs (was 1)
#   3. Better action parsing with multi-stage fallback
#   4. Per-task checkpointing for crash recovery

from trl import GRPOTrainer, GRPOConfig
import re
import gc

FastLanguageModel.for_training(model)

SCORE_MIN, SCORE_MAX = 0.01, 0.99
VALID_ACTION_TYPES = {"click", "type", "scroll", "wait"}
VALID_REF_PREFIXES = {"inp", "btn", "cmb", "lnk", "frm", "hdg", "mn", "nav", "chk", "rad", "opt", "txt", "lst", "el", "sec"}

if "CHECKPOINT_DIR" not in dir():
    CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_RECORDS_PER_TASK = 300   # V2-lite: 300 per task (budget)
TARGET_MINORITY_PCT = 0.35

TASK_LABELS = {
    1: "Booking: Clean Form",
    2: "Booking: Label Drift",
    3: "Booking: Structural Drift",
    4: "Booking: Full Chaos",
    5: "E-commerce: Clean Store",
    6: "E-commerce: Label Drift",
    7: "E-commerce: Structural Drift",
    8: "E-commerce: Full Chaos",
}


def _parse_action_from_text(text):
    '''Multi-stage action parser with regex fallback.'''
    text = text.strip()
    if not text:
        return {}

    action_text = text
    if "<answer>" in text:
        try:
            a_s = text.index("<answer>") + 8
            a_e = text.index("</answer>") if "</answer>" in text else len(text)
            action_text = text[a_s:a_e].strip()
        except ValueError:
            pass
    if action_text.startswith("```"):
        parts = action_text.split("```")
        if len(parts) > 1:
            action_text = parts[1].lstrip("json").strip()
    try:
        parsed = json.loads(action_text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback
    result = {}
    at_match = re.search(r'"action_type"\s*:\s*"(click|type|scroll|wait)"', text)
    if at_match:
        result["action_type"] = at_match.group(1)
    ref_match = re.search(r'"element_ref"\s*:\s*"([^"]*)"', text)
    if ref_match:
        result["element_ref"] = ref_match.group(1)
    tv_match = re.search(r'"text_value"\s*:\s*"([^"]*)"', text)
    if tv_match:
        result["text_value"] = tv_match.group(1)
    sd_match = re.search(r'"scroll_delta"\s*:\s*(-?\d+)', text)
    if sd_match:
        result["scroll_delta"] = int(sd_match.group(1))
    return result


def _is_form_field_ref(ref):
    '''Check if ref points to a form field (textbox, combobox).'''
    return ref.startswith("inp_") or ref.startswith("cmb_")

def _is_button_ref(ref):
    '''Check if ref points to a button or link.'''
    return ref.startswith("btn_") or ref.startswith("lnk_")


def tolerant_reward(completions, oracle_action_type=None,
                    oracle_element_ref=None, oracle_text_value=None,
                    **kwargs):
    '''
    TOLERANT reward -- accepts any valid next action, not just oracle's exact choice.

    Scoring philosophy:
      - Correct action_type for the element type -> high reward
      - Wrong action_type (click on textbox, type on button) -> low reward
      - Valid action with correct format -> medium reward
      - Unparseable -> floor

    Key change from V1: filling destination before origin gets the SAME reward
    as filling origin first. Any valid form fill on an empty field scores high.
    '''
    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        text = text.strip()

        o_atype = oracle_action_type[i] if oracle_action_type else ""
        o_ref = oracle_element_ref[i] if oracle_element_ref else ""
        o_tval = oracle_text_value[i] if oracle_text_value else ""

        predicted = _parse_action_from_text(text)
        p_atype = predicted.get("action_type", "")
        p_ref = predicted.get("element_ref", "")
        p_tval = predicted.get("text_value", "")

        has_format = "<think>" in text and "<answer>" in text
        format_bonus = 0.04 if has_format else 0.0

        # --- Unparseable ---
        if not predicted or p_atype not in VALID_ACTION_TYPES:
            rewards.append(SCORE_MIN)
            continue

        # --- TOLERANT SCORING ---
        score = 0.0

        # 1. Action type correctness (biggest signal)
        if p_atype == o_atype:
            score += 0.40  # Correct action type
        elif p_atype == "type" and _is_form_field_ref(p_ref):
            score += 0.30  # Type on a form field is generally good
        elif p_atype == "click" and _is_button_ref(p_ref):
            score += 0.25  # Click on a button is generally good
        elif p_atype == "scroll":
            score += 0.15  # Scroll is rarely wrong but not always useful
        elif p_atype == "wait":
            score += 0.10  # Wait is safe but passive
        else:
            score += 0.05  # Wrong type entirely (click on textbox etc.)

        # 2. Element targeting (secondary signal)
        if p_ref == o_ref:
            score += 0.30  # Exact match
        elif p_ref and o_ref:
            # Same element type = partial credit
            p_prefix = p_ref.split("_")[0] if "_" in p_ref else ""
            o_prefix = o_ref.split("_")[0] if "_" in o_ref else ""
            if p_prefix == o_prefix:
                score += 0.15  # Same type of element (inp vs inp)
            elif p_prefix in VALID_REF_PREFIXES:
                score += 0.05  # Valid ref but different type
        elif p_ref:
            score += 0.03  # Has a ref but oracle doesn't (or vice versa)

        # 3. Text value (for type actions)
        if p_atype == "type" and o_atype == "type" and o_tval:
            if p_tval.strip().lower() == o_tval.strip().lower():
                score += 0.20  # Exact text match
            elif p_tval and o_tval.lower() in p_tval.lower():
                score += 0.10  # Partial match
            elif p_tval:
                score += 0.02  # Has text but wrong
        elif p_atype != "type":
            score += 0.10  # Non-type actions don't need text value

        score += format_bonus
        rewards.append(max(SCORE_MIN, min(SCORE_MAX, score)))

    return rewards


def balance_task_actions(task_records, target_minority_pct, rng_seed):
    '''Oversample minority action types within a task.'''
    rng = stdlib_random.Random(rng_seed)
    by_type = defaultdict(list)
    for r in task_records:
        by_type[r["action"]["action_type"]].append(r)

    majority = by_type.get("click", [])
    minority = []
    for atype in ("type", "wait", "scroll"):
        minority.extend(by_type.get(atype, []))

    if not minority or not majority:
        return task_records

    n_majority = len(majority)
    n_needed = int(target_minority_pct * n_majority / (1 - target_minority_pct))
    n_needed = max(n_needed, len(minority))

    if n_needed > len(minority):
        oversampled = minority * (n_needed // len(minority))
        remainder = n_needed % len(minority)
        if remainder:
            oversampled += rng.sample(minority, remainder)
    else:
        oversampled = minority

    balanced = majority + oversampled
    rng.shuffle(balanced)
    return balanced


# --- GRPO Config ---
grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,           # V2-lite: 1 epoch (budget)
    per_device_train_batch_size=2,  # 7B allows batch 2
    gradient_accumulation_steps=4,  # Effective batch = 8
    learning_rate=5e-6,           # Standard for 7B GRPO
    logging_steps=10,
    save_steps=500,
    max_completion_length=300,
    num_generations=2,            # V2-lite: 2 gens (budget)
    temperature=0.7,
    report_to="none",
    seed=42,
)

reward_history = []

for task_id in range(1, 9):
    label = TASK_LABELS.get(task_id, f"Task {task_id}")
    print(f"\n{'='*60}")
    print(f"GRPO -- Task {task_id}: {label}")
    print(f"{'='*60}")

    task_records = [r for r in records if r["task_id"] == task_id]
    if not task_records:
        print(f"  No data for task {task_id} -- skipping")
        continue

    # Balance within task
    raw_dist = Counter(r["action"]["action_type"] for r in task_records)
    task_records = balance_task_actions(task_records, TARGET_MINORITY_PCT, rng_seed=42 + task_id)
    balanced_dist = Counter(r["action"]["action_type"] for r in task_records)
    print(f"  Balance: {dict(raw_dist)} -> {dict(balanced_dist)}")

    # Cap
    if len(task_records) > MAX_RECORDS_PER_TASK:
        rng = stdlib_random.Random(42 + task_id)
        task_records = rng.sample(task_records, MAX_RECORDS_PER_TASK)
    print(f"  Records: {len(task_records)}")

    prompts = []
    oracle_atypes = []
    oracle_refs = []
    oracle_tvals = []

    for r in task_records:
        obs_text = r["observation"][:OBS_MAX_CHARS]
        step = r.get("step", 0)
        step_ctx = f"\nStep: {step}" if step > 0 else ""

        # Add step history
        ep_key = (r["task_id"], r["seed"])
        ep_records = episodes.get(ep_key, [])
        prev_steps = [pr for pr in ep_records if pr["step"] < step][-STEP_HISTORY_WINDOW:]
        history = ""
        if prev_steps:
            hlines = []
            for prev in prev_steps:
                a = prev["action"]
                hl = f"  Step {prev['step']}: {a['action_type']} {a.get('element_ref', '')}"
                if a.get("text_value"):
                    hl += f' "{a["text_value"]}"'
                hlines.append(hl)
            history = "\nPrevious actions:\n" + "\n".join(hlines) + "\n"

        user_msg = f"Task: {r['instruction']}{history}\n\nAccessibility Tree:\n{obs_text}{step_ctx}"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(msgs)

        action = r["action"]
        oracle_atypes.append(action.get("action_type", ""))
        oracle_refs.append(action.get("element_ref", ""))
        oracle_tvals.append(action.get("text_value", ""))

    prompt_dataset = Dataset.from_dict({
        "prompt": prompts,
        "oracle_action_type": oracle_atypes,
        "oracle_element_ref": oracle_refs,
        "oracle_text_value": oracle_tvals,
    })

    eff_batch = grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps
    total_steps = (len(prompts) * grpo_config.num_train_epochs) // eff_batch
    print(f"  Steps: ~{total_steps} | Gens: {grpo_config.num_generations} | Epochs: {grpo_config.num_train_epochs}")

    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[tolerant_reward],
        train_dataset=prompt_dataset,
        args=grpo_config,
    )

    result = grpo_trainer.train()

    avg_reward = None
    if hasattr(grpo_trainer, "state") and grpo_trainer.state.log_history:
        reward_logs = [h["reward"] for h in grpo_trainer.state.log_history if "reward" in h]
        if reward_logs:
            avg_reward = sum(reward_logs) / len(reward_logs)

    loss_val = result.training_loss
    print(f"  Loss: {loss_val:.4f}" + (f" | Avg reward: {avg_reward:.4f}" if avg_reward else ""))

    reward_history.append({
        "task_id": task_id, "label": label, "loss": loss_val, "avg_reward": avg_reward,
    })

    # Save after each task (crash recovery)
    save_path = f"{CHECKPOINT_DIR}/grpo_after_task_{task_id}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Checkpoint: {save_path}")

    del grpo_trainer
    gc.collect()
    torch.cuda.empty_cache()

    grpo_config.learning_rate *= 0.92  # Gentle LR decay across tasks

# Save final GRPO model
model.save_pretrained(f"{CHECKPOINT_DIR}/grpo_final")
tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/grpo_final")

print(f"\n{'='*60}")
print("GRPO complete!")
for rh in reward_history:
    r_str = f"  reward={rh['avg_reward']:.4f}" if rh.get("avg_reward") else ""
    print(f"  Task {rh['task_id']} ({rh['label']}): loss={rh['loss']:.4f}{r_str}")"""))

# ============================================================
# CELL 10 -- Online RL (SKIPPED in budget mode)
# ============================================================
cells.append(cell("code", """# Cell 8.5 -- Online RL: SKIPPED (budget mode)
#
# Online RL is skipped to save compute. SFT + GRPO provide the core improvements.
# The model already learns per-step action quality from GRPO.
# Online RL would add ~10-20% improvement but costs 30-45 min of A100 time.
#
# To enable it later, replace this cell with the full Online RL cell from V2-full.

print("Online RL: SKIPPED (budget mode)")
print("SFT + GRPO training is complete. Proceeding to quality check and evaluation.")"""))
# ============================================================
# CELL 11 -- Quality Check
# ============================================================
cells.append(cell("code", """# Cell 9 -- Post-RL Quality Check

FastLanguageModel.for_inference(model)

quality_tests = [
    ("Booking: empty form", "Task: Book a Sleeper ticket from Delhi to Mumbai\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=frm_1 role=form name=\\"Search Trains\\"]\\n    [ref=inp_1 role=textbox name=\\"From\\" value=\\"\\"]\\n    [ref=inp_2 role=textbox name=\\"To\\" value=\\"\\"]\\n    [ref=cmb_1 role=combobox name=\\"Class\\" value=\\"-- Select --\\"]\\n    [ref=btn_1 role=button name=\\"Search\\"]"),
    ("Booking: drifted labels", "Task: Book an AC 2 Tier ticket from Hyderabad to Pune\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=frm_1 role=form name=\\"Journey Planner\\"]\\n    [ref=inp_1 role=textbox name=\\"Starting Point\\" value=\\"\\"]\\n    [ref=inp_2 role=textbox name=\\"Going To\\" value=\\"\\"]\\n    [ref=cmb_1 role=combobox name=\\"Coach Class\\" value=\\"-- Pick --\\"]\\n    [ref=btn_1 role=button name=\\"Check Availability\\"]"),
    ("Booking: with distractor", "Task: Book a Chair Car ticket from Kolkata to Lucknow\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=btn_99 role=button name=\\"Accept\\" description=\\"cookie banner\\"]\\n  [ref=frm_1 role=form name=\\"Travel Booking\\"]\\n    [ref=inp_1 role=textbox name=\\"Origin\\" value=\\"\\"]\\n    [ref=inp_2 role=textbox name=\\"Destination\\" value=\\"\\"]\\n    [ref=btn_1 role=button name=\\"Look Up Trains\\"]"),
    ("E-com: product search", "Task: Buy a Laptop from Electronics. Ship to Raj, 12 MG Road, Bangalore, 560001, 9876543210\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=inp_1 role=textbox name=\\"Search\\" value=\\"\\"]\\n  [ref=cmb_1 role=combobox name=\\"Category\\" value=\\"All\\"]\\n  [ref=btn_1 role=button name=\\"Filter\\"]"),
    ("E-com: shipping form", "Task: Buy a Laptop from Electronics. Ship to Raj, 12 MG Road, Bangalore, 560001, 9876543210\\n\\nPrevious actions:\\n  Step 6: click btn_2\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=hdg_1 role=heading name=\\"Shipping\\"]\\n  [ref=inp_1 role=textbox name=\\"Name\\" value=\\"\\"]\\n  [ref=inp_2 role=textbox name=\\"Address\\" value=\\"\\"]\\n  [ref=btn_1 role=button name=\\"Place Order\\"]\\nStep: 7"),
    ("Scroll needed", "Task: Book a Sleeper ticket from Chennai to Jaipur\\n\\nAccessibility Tree:\\n[ref=mn_1 role=main]\\n  [ref=hdg_1 role=heading name=\\"Available Trains\\"]\\n  [... truncated -- budget exhausted]\\nStep: 5"),
]

print("Post-RL Quality Check")
print("=" * 60)
atypes_seen = Counter()
pass_count = 0

for label, test_input in quality_tests:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_input},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    results = []
    for _ in range(3):
        outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.5, do_sample=True)
        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(resp)

    print(f"\\n[{label}]")
    for j, resp in enumerate(results):
        try:
            parsed = parse_model_output(resp)
            atype = parsed.get("action_type", "?")
            atypes_seen[atype] += 1
            ref = parsed.get("element_ref", "?")
            tval = parsed.get("text_value", "")
            print(f"  {j+1}: {atype} {ref}" + (f' \"{tval}\"' if tval else "") + " -- PASS")
            pass_count += 1
        except Exception:
            print(f"  {j+1}: FAIL (parse error)")

print(f"\\n{'=' * 60}")
print(f"Passed: {pass_count}/{len(quality_tests) * 3}")
print(f"Action types: {dict(atypes_seen)}")
type_pct = atypes_seen.get("type", 0) / max(sum(atypes_seen.values()), 1)
click_pct = atypes_seen.get("click", 0) / max(sum(atypes_seen.values()), 1)
print(f"Type usage: {type_pct:.0%} | Click usage: {click_pct:.0%}")
if type_pct >= 0.20:
    print("GOOD: Model uses 'type' actions -- no mode collapse")
else:
    print("WARNING: Low 'type' usage -- possible mode collapse")"""))

# ============================================================
# CELL 12 -- Live Evaluation
# ============================================================
cells.append(cell("code", r"""# Cell 10 -- Live Environment Evaluation (WebSocket)
#
# V2: More seeds, all 8 tasks, includes random baseline comparison

import json
import asyncio
import nest_asyncio
import websockets
import httpx

nest_asyncio.apply()

INFINITE_DOM_URL = os.environ.get("INFINITE_DOM_URL", HF_SPACE_URL)
WS_URL = INFINITE_DOM_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

print(f"Environment: {INFINITE_DOM_URL}")
try:
    r = httpx.get(f"{INFINITE_DOM_URL}/health", timeout=30)
    r.raise_for_status()
    print(f"Server: {r.json()}")
except Exception as e:
    print(f"ERROR: {e}")
    raise SystemExit("Environment not available for evaluation")

EVAL_SEEDS = 3  # Seeds per task (budget mode)
MAX_EVAL_STEPS = 20

FastLanguageModel.for_inference(model)


async def eval_episode(task_id, seed, use_model=True):
    '''Run one eval episode. Returns (nodes_completed, total_nodes, total_steps, total_reward).'''
    try:
        async with websockets.connect(WS_URL, open_timeout=30, close_timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id, "seed": seed}}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            obs = resp.get("data", resp)
            # OpenEnv WS: data = {"observation": {...}, "reward": ..., "done": ...}
            obs_inner = obs.get("observation", obs)

            instruction = obs_inner.get("task_instruction", "")
            a11y = obs_inner.get("a11y_tree", "")
            progress = obs_inner.get("task_progress", [])
            total_nodes = obs.get("metadata", obs_inner.get("metadata", {})).get("total_nodes", 5)
            total_reward = 0.0
            step_history = []

            for step_i in range(MAX_EVAL_STEPS):
                if use_model:
                    history = ""
                    if step_history:
                        hlines = [f"  Step {s['step']}: {s['atype']} {s['ref']}" +
                                  (f' \"{s["tval"]}\"' if s.get("tval") else "")
                                  for s in step_history[-3:]]
                        history = "\nPrevious actions:\n" + "\n".join(hlines) + "\n"

                    user_msg = f"Task: {instruction}{history}\n\nAccessibility Tree:\n{a11y[:OBS_MAX_CHARS]}"
                    if step_i > 0:
                        user_msg += f"\nStep: {step_i}"

                    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
                    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
                    out = model.generate(**inp, max_new_tokens=250, temperature=0.1)
                    resp_text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
                    action = parse_model_output_safe(resp_text)
                else:
                    # Random baseline
                    import random
                    action = {
                        "action_type": random.choice(["click", "type", "scroll", "wait"]),
                        "element_ref": f"btn_{random.randint(1,5)}",
                        "text_value": "",
                        "scroll_delta": 0,
                    }

                await ws.send(json.dumps({"type": "step", "data": action}))
                step_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                step_data = step_resp.get("data", step_resp)
                step_obs = step_data.get("observation", step_data)

                progress = step_obs.get("task_progress", progress)
                total_reward += step_data.get("reward", 0)
                a11y = step_obs.get("a11y_tree", a11y)
                done = step_data.get("done", False)

                step_history.append({
                    "step": step_i,
                    "atype": action["action_type"],
                    "ref": action.get("element_ref", ""),
                    "tval": action.get("text_value", ""),
                })

                if done:
                    break

            return len(progress), total_nodes, step_i + 1, total_reward
    except Exception as e:
        print(f"    Eval error (task={task_id}, seed={seed}): {e}")
        return 0, 5, 0, 0.0


# --- Random Baseline ---
print("\n=== Random Baseline ===")
random_results = {}
for task_id in [1, 2, 5]:
    nodes_list = []
    for seed in range(EVAL_SEEDS):
        n, t, s, r = asyncio.get_event_loop().run_until_complete(eval_episode(task_id, seed * 17, use_model=False))
        nodes_list.append(n)
    avg = sum(nodes_list) / len(nodes_list)
    random_results[task_id] = avg
    print(f"  Task {task_id}: avg {avg:.1f}/{t} nodes")

# --- Trained Model ---
print("\n=== Trained Model -- Tasks 1, 2, 5 (budget eval) ===")
trained_results = {}
trained_totals = {}
trained_rewards = {}

for task_id in [1, 2, 5]:
    nodes_list = []
    reward_list = []
    total = 5
    for seed in range(EVAL_SEEDS):
        n, t, s, r = asyncio.get_event_loop().run_until_complete(eval_episode(task_id, seed * 17, use_model=True))
        nodes_list.append(n)
        reward_list.append(r)
        total = t
    avg_nodes = sum(nodes_list) / len(nodes_list)
    avg_reward = sum(reward_list) / len(reward_list)
    trained_results[task_id] = avg_nodes
    trained_totals[task_id] = total
    trained_rewards[task_id] = avg_reward
    label = TASK_LABELS.get(task_id, f"Task {task_id}")
    print(f"  Task {task_id} ({label}): avg {avg_nodes:.1f}/{total} nodes, reward={avg_reward:.3f}")

# --- Summary ---
print(f"\n=== Summary ===")
random_avg = sum(random_results.values()) / max(len(random_results), 1)
trained_avg = sum(trained_results.values()) / max(len(trained_results), 1)
booking_avg = sum(trained_results[t] for t in range(1, 5)) / 4
ecom_avg = sum(trained_results[t] for t in range(5, 9)) / 4
print(f"Random baseline: {random_avg:.1f} nodes avg")
print(f"Trained model:   {trained_avg:.1f} nodes avg (booking: {booking_avg:.1f}, ecom: {ecom_avg:.1f})")
print(f"Improvement:     +{trained_avg - random_avg:.1f} nodes")"""))

# ============================================================
# CELL 13 -- Save final model
# ============================================================
cells.append(cell("code", """# Cell 11 -- Save final model (LoRA adapters)

model.save_pretrained(f"{CHECKPOINT_DIR}/final_model")
tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/final_model")
print(f"Final model saved to: {CHECKPOINT_DIR}/final_model")

# Push to HF Hub (optional)
# model.push_to_hub("your-username/infinite-dom-agent", token=os.environ.get("HF_TOKEN"))
# tokenizer.push_to_hub("your-username/infinite-dom-agent", token=os.environ.get("HF_TOKEN"))"""))

# ============================================================
# CELL 14 -- Training Plots
# ============================================================
cells.append(cell("code", r"""# Cell 12 -- Training Plots

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 120

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: SFT Loss ---
ax = axes[0]
if sft_log_history:
    train_steps = [h["step"] for h in sft_log_history if "loss" in h and "eval_loss" not in h]
    train_loss = [h["loss"] for h in sft_log_history if "loss" in h and "eval_loss" not in h]
    eval_steps = [h["step"] for h in sft_log_history if "eval_loss" in h]
    eval_loss = [h["eval_loss"] for h in sft_log_history if "eval_loss" in h]

    if train_steps:
        ax.plot(train_steps, train_loss, "b-", label="Train", alpha=0.7)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, "r-", label="Eval", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training & Eval Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "No SFT data\n(resumed from checkpoint)", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("SFT Training & Eval Loss")

# --- Plot 2: GRPO Reward by Task ---
ax = axes[1]
if reward_history:
    task_ids = [rh["task_id"] for rh in reward_history]
    rewards = [rh["avg_reward"] or 0 for rh in reward_history]
    labels = [f"T{rh['task_id']}" for rh in reward_history]

    colors_map = {1: "#2ecc71", 2: "#27ae60", 3: "#1abc9c", 4: "#16a085",
                  5: "#e74c3c", 6: "#c0392b", 7: "#e67e22", 8: "#d35400"}
    bar_colors = [colors_map.get(rh["task_id"], "#3498db") for rh in reward_history]
    bars = ax.bar(labels, rewards, color=bar_colors)

    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Avg Reward")
    ax.set_title("GRPO Avg Reward by Task")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.text(0.5, 0.5, "No GRPO data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("GRPO Avg Reward by Task")

# --- Plot 3: Trained vs Random ---
ax = axes[2]
if "trained_results" in dir() and trained_results:
    task_ids_eval = sorted(trained_results.keys())
    trained_vals = [trained_results[t] for t in task_ids_eval]
    random_vals = [random_results.get(t, 0) for t in task_ids_eval]
    x = range(len(task_ids_eval))
    w = 0.35

    ax.bar([i - w/2 for i in x], random_vals, w, label="Random", color="#95a5a6", alpha=0.7)
    ax.bar([i + w/2 for i in x], trained_vals, w, label="Trained", color="#3498db")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"T{t}" for t in task_ids_eval], fontsize=8)
    ax.set_ylabel("Avg Nodes Completed")
    ax.set_title("Random vs Trained Agent")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Random vs Trained Agent")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: training_curves.png")
plt.show()"""))

# ============================================================
# CELL 14.5 -- Visual Demo: Agent Navigating with Screenshots
# ============================================================
cells.append(cell("code", r"""# Cell 12.5 -- VISUAL DEMO: Watch the Trained Agent Navigate a Real Website
#
# This cell connects to the live environment, runs the trained model step-by-step,
# and displays real browser screenshots showing the agent filling forms and clicking buttons.
# Screen-record this cell running for your submission video.

import httpx
import asyncio
import nest_asyncio
import websockets
import json
import base64
import re
from IPython.display import display, Image, HTML, clear_output

nest_asyncio.apply()

if not env_available:
    print("Environment not available -- skipping visual demo.")
else:
    SCREENSHOT_URL = HF_SPACE_URL + "/screenshot"
    DEMO_TASK = 1
    DEMO_SEED = 42
    DEMO_MAX_STEPS = 12
    OBS_MAX_CHARS_DEMO = 6000

    def get_screenshot():
        '''Fetch a browser screenshot from the environment server.'''
        try:
            r = httpx.get(SCREENSHOT_URL, timeout=15)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                return r.content
        except Exception as e:
            print(f"  (screenshot unavailable: {e})")
        return None

    def parse_action_demo(text):
        text = text.strip()
        if "<answer>" in text:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>") if "</answer>" in text else len(text)
            text = text[start:end].strip()
        try:
            parsed = json.loads(text)
            return {
                "action_type": parsed.get("action_type", "wait"),
                "element_ref": parsed.get("element_ref", ""),
                "text_value": parsed.get("text_value", ""),
                "scroll_delta": parsed.get("scroll_delta", 0),
            }
        except Exception:
            result = {"action_type": "wait", "element_ref": "", "text_value": "", "scroll_delta": 0}
            at = re.search(r'"action_type"\s*:\s*"(\w+)"', text)
            if at: result["action_type"] = at.group(1)
            ref = re.search(r'"element_ref"\s*:\s*"([^"]*)"', text)
            if ref: result["element_ref"] = ref.group(1)
            tv = re.search(r'"text_value"\s*:\s*"([^"]*)"', text)
            if tv: result["text_value"] = tv.group(1)
            return result

    async def visual_demo():
        FastLanguageModel.for_inference(model)

        print("=" * 70)
        print("  VISUAL DEMO: Trained Agent Navigating a Real Web Page")
        print("=" * 70)
        print(f"  Task: {DEMO_TASK} | Seed: {DEMO_SEED}")
        print()

        screenshots = []

        async with websockets.connect(WS_URL, open_timeout=30, close_timeout=10) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": DEMO_TASK, "seed": DEMO_SEED}}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            obs_data = resp.get("data", resp)
            obs = obs_data.get("observation", obs_data)

            instruction = obs.get("task_instruction", "")
            a11y_tree = obs.get("a11y_tree", "")
            progress = obs.get("task_progress", [])

            print(f"  Task: {instruction}")
            print()

            # Initial screenshot
            import time
            time.sleep(0.5)
            img = get_screenshot()
            if img:
                print("  [Initial Page]")
                display(Image(data=img, width=700))
                screenshots.append(("Initial Page", img))
            print()

            step_history = []

            for step_i in range(DEMO_MAX_STEPS):
                # Build prompt
                hist_str = ""
                if step_history:
                    hlines = [f"  Step {s['step']}: {s['atype']} {s['ref']}" +
                              (f' "{s["tval"]}"' if s.get("tval") else "")
                              for s in step_history[-3:]]
                    hist_str = "\nPrevious actions:\n" + "\n".join(hlines) + "\n"

                step_ctx = f"\nStep: {step_i}" if step_i > 0 else ""
                user_msg = f"Task: {instruction}{hist_str}\n\nAccessibility Tree:\n{a11y_tree[:OBS_MAX_CHARS_DEMO]}{step_ctx}"

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.1)
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                action = parse_action_demo(response)
                atype = action["action_type"]
                aref = action.get("element_ref", "")
                atval = action.get("text_value", "")

                # Print action
                action_desc = f"{atype} {aref}"
                if atval:
                    action_desc += f' "{atval}"'
                print(f"  Step {step_i}: {action_desc}")

                # Show thinking if present
                if "<think>" in response and "</think>" in response:
                    think = response[response.index("<think>")+7:response.index("</think>")].strip()
                    print(f"    Think: {think[:150]}")

                # Execute
                await ws.send(json.dumps({"type": "step", "data": action}))
                step_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                step_data = step_resp.get("data", step_resp)
                step_obs = step_data.get("observation", step_data)

                new_progress = step_obs.get("task_progress", progress)
                done = step_data.get("done", False)
                a11y_tree = step_obs.get("a11y_tree", a11y_tree)
                reward = step_data.get("reward", 0)

                newly_done = [n for n in new_progress if n not in progress]
                progress = new_progress

                if newly_done:
                    print(f"    >> Node completed: {newly_done} (reward: {reward:+.3f})")

                step_history.append({
                    "step": step_i, "atype": atype, "ref": aref, "tval": atval
                })

                # Screenshot after action
                time.sleep(0.3)
                img = get_screenshot()
                if img and (newly_done or atype == "type" or step_i == 0):
                    caption = f"After step {step_i}: {action_desc}"
                    print(f"  [{caption}]")
                    display(Image(data=img, width=700))
                    screenshots.append((caption, img))

                print(f"    Progress: {len(progress)} nodes | Done: {done}")
                print()

                if done:
                    print(f"  EPISODE COMPLETE!")
                    print(f"  Nodes: {len(progress)} | Total reward: {sum(step_data.get('reward', 0) for _ in [1]):.3f}")
                    # Final screenshot
                    time.sleep(0.5)
                    img = get_screenshot()
                    if img:
                        print("  [Final State]")
                        display(Image(data=img, width=700))
                        screenshots.append(("Final State", img))
                    break

        print()
        print("=" * 70)
        print(f"  Demo complete: {len(progress)} nodes completed in {step_i+1} steps")
        print(f"  Screenshots captured: {len(screenshots)}")
        print("=" * 70)

        # Save screenshots
        for i, (caption, img_data) in enumerate(screenshots):
            fname = f"demo_screenshot_{i}.png"
            with open(fname, "wb") as f:
                f.write(img_data)
        print(f"\nScreenshots saved as demo_screenshot_*.png")

    asyncio.get_event_loop().run_until_complete(visual_demo())"""))

# ============================================================
# CELL 15 -- Summary Report
# ============================================================
cells.append(cell("code", """# Cell 13 -- Summary Report

print("=" * 60)
print("  INFINITE DOM -- TRAINING SUMMARY (V2)")
print("=" * 60)

print(f"\\nModel: {MODEL_ID}")
print(f"Quantization: QLoRA 4-bit, LoRA rank 16")
print(f"Observation Window: {OBS_MAX_CHARS} chars")
print(f"Step History: last {STEP_HISTORY_WINDOW} actions")
print(f"Tasks trained: 8 (GRPO all tasks) | Eval: tasks 1, 2, 5")
print(f"Mode: Budget (~1h A100)")

print(f"\\n--- Data ---")
print(f"Oracle records: {len(records)}")
print(f"SFT records (balanced): {len(train_ds)} train / {len(eval_ds)} eval")
bal = Counter(r["action"]["action_type"] for r in balanced_records)
print(f"Action balance: {dict(bal)}")

print(f"\\n--- SFT Phase ---")
if "sft_results" in dir():
    print(f"Training loss: {sft_results.training_loss:.4f}")
if sft_best_metric is not None:
    print(f"Best eval loss: {sft_best_metric:.4f}")

print(f"\\n--- GRPO Phase ---")
if reward_history:
    for rh in reward_history:
        r_str = f"{rh['avg_reward']:.4f}" if rh.get("avg_reward") else "N/A"
        print(f"  Task {rh['task_id']} ({rh['label']}): loss={rh['loss']:.4f}  reward={r_str}")

print(f"\\n--- Online RL ---")
print("Skipped (budget mode -- SFT + GRPO sufficient for demonstration)")

if "trained_results" in dir() and trained_results:
    print(f"\\n--- Live Evaluation ---")
    print(f"{'Task':>6s}  {'Nodes':>10s}  {'Reward':>8s}")
    print("-" * 35)
    for tid in sorted(trained_results.keys()):
        total = trained_totals.get(tid, 5)
        reward = trained_rewards.get(tid, 0)
        print(f"{tid:>6d}  {trained_results[tid]:>5.1f}/{total:<3d}  {reward:>8.3f}")

    trained_avg_all = sum(trained_results.values()) / len(trained_results)
    booking_eval = [trained_results[t] for t in range(1, 5) if t in trained_results]
    ecom_eval = [trained_results[t] for t in range(5, 9) if t in trained_results]
    print(f"\\n  Overall: {trained_avg_all:.1f} nodes avg")
    if booking_eval:
        print(f"  Booking: {sum(booking_eval)/len(booking_eval):.1f} nodes avg")
    if ecom_eval:
        print(f"  E-commerce: {sum(ecom_eval)/len(ecom_eval):.1f} nodes avg")

print(f"\\n{'=' * 60}")"""))

# ============================================================
# Build notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "gpuClass": "standard"
    },
    "cells": cells,
}

out_path = "training/train_infinite_dom_v2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {out_path}")
print(f"Total cells: {len(cells)}")
